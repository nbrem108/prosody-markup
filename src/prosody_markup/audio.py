from __future__ import annotations

import hashlib
import io
import math
import struct
import wave
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MAX_WAV_BYTES = 256 * 1024 * 1024
_CHUNK_FRAMES = 65_536

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2

# Normalization holds the decoded signal in memory, so the bound is memory
# rather than time. Half an hour of 48 kHz audio is a few hundred megabytes of
# float64 and a few seconds of work; beyond that, refuse rather than thrash.
MAX_NORMALIZE_SECONDS = 1800.0

# Output samples computed per pass, which caps peak memory independently of
# how long the recording is.
_RESAMPLE_CHUNK = 1 << 20

_TAPS_PER_PHASE = 32
_INT16_PEAK = 32_767
_INT16_FLOOR = -32_768


class AudioError(ValueError):
    """Base class for local audio contract failures."""


class AudioInspectionError(AudioError):
    """Raised when an input cannot satisfy the local WAV inspection contract."""


class AudioNormalizationError(AudioError):
    """Raised when audio cannot be converted to the canonical pipeline format."""


@dataclass(frozen=True, slots=True)
class AudioInspection:
    schema_version: str
    container: str
    encoding: str
    source_path: str
    source_sha256: str
    duration_s: float
    sample_rate: int
    channels: int
    sample_width_bits: int
    frame_count: int
    peak_amplitude: float
    clipping: bool
    silence_ratio: float
    conversion_required: bool

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


def _decode_pcm(frames: bytes, sample_width: int) -> Iterator[int]:
    if sample_width == 1:
        yield from (value - 128 for value in frames)
        return
    if sample_width == 2:
        yield from (value[0] for value in struct.iter_unpack("<h", frames))
        return
    if sample_width == 3:
        for offset in range(0, len(frames), 3):
            value = int.from_bytes(frames[offset : offset + 3], "little", signed=False)
            if value & 0x800000:
                value -= 1 << 24
            yield value
        return
    if sample_width == 4:
        yield from (value[0] for value in struct.iter_unpack("<i", frames))
        return
    raise AudioInspectionError(f"unsupported PCM sample width: {sample_width * 8} bits")


def inspect_wav(path: Path) -> AudioInspection:
    source = path.expanduser().resolve()
    try:
        source_size = source.stat().st_size
        if source_size > MAX_WAV_BYTES:
            raise AudioInspectionError(
                f"WAV size {source_size} bytes exceeds the {MAX_WAV_BYTES}-byte safety limit"
            )
        source_bytes = source.read_bytes()
    except OSError as exc:
        raise AudioInspectionError(f"cannot read audio file: {source}") from exc

    try:
        with wave.open(io.BytesIO(source_bytes), "rb") as stream:
            if stream.getcomptype() != "NONE":
                raise AudioInspectionError("audio must be uncompressed PCM WAV")
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frame_count = stream.getnframes()
            if channels <= 0 or sample_rate <= 0:
                raise AudioInspectionError("WAV metadata must declare channels and sample rate")
            if frame_count <= 0:
                raise AudioInspectionError("WAV contains no audio frames")

            full_scale = float((1 << (sample_width * 8 - 1)) - 1)
            silence_floor = full_scale * 0.01
            peak = 0
            silent_samples = 0
            sample_count = 0
            bytes_read = 0
            remaining_frames = frame_count
            frame_width = channels * sample_width

            while remaining_frames > 0:
                requested = min(_CHUNK_FRAMES, remaining_frames)
                frames = stream.readframes(requested)
                if not frames or len(frames) % frame_width != 0:
                    raise AudioInspectionError("WAV frame data is truncated")
                bytes_read += len(frames)
                decoded_frames = len(frames) // frame_width
                remaining_frames -= decoded_frames
                for sample in _decode_pcm(frames, sample_width):
                    magnitude = abs(sample)
                    peak = max(peak, magnitude)
                    silent_samples += magnitude <= silence_floor
                    sample_count += 1
    except (EOFError, wave.Error) as exc:
        raise AudioInspectionError("audio must be a valid PCM WAV") from exc

    expected_bytes = frame_count * channels * sample_width
    if bytes_read != expected_bytes or sample_count != frame_count * channels:
        raise AudioInspectionError(
            f"WAV frame data is truncated: expected {expected_bytes} bytes, read {bytes_read}"
        )
    if sample_count == 0:
        raise AudioInspectionError("WAV contains no decodable PCM samples")

    peak_amplitude = min(1.0, peak / full_scale)
    clipping = peak_amplitude >= 0.999
    silence_ratio = silent_samples / sample_count
    duration_s = frame_count / sample_rate

    numeric_values = (peak_amplitude, silence_ratio, duration_s)
    if not all(math.isfinite(value) for value in numeric_values):
        raise AudioInspectionError("audio diagnostics must be finite")

    return AudioInspection(
        schema_version="audio-inspection@0.1.0",
        container="wav",
        encoding="pcm",
        source_path=str(source),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bits=sample_width * 8,
        frame_count=frame_count,
        peak_amplitude=peak_amplitude,
        clipping=clipping,
        silence_ratio=silence_ratio,
        conversion_required=channels != 1 or sample_rate != 16_000 or sample_width != 2,
    )


@dataclass(frozen=True, slots=True)
class NormalizationManifest:
    schema_version: str
    source_path: str
    source_sha256: str
    source_sample_rate: int
    source_channels: int
    source_sample_width_bits: int
    source_frame_count: int
    source_duration_s: float
    output_path: str
    output_sha256: str
    output_sample_rate: int
    output_channels: int
    output_sample_width_bits: int
    output_frame_count: int
    output_duration_s: float
    output_peak_amplitude: float
    samples_modified: bool
    channel_mix: str
    resampler: str
    resample_ratio: str
    filter_taps_per_phase: int

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


def _blackman_sinc(taps: int, cutoff: float) -> np.typing.NDArray[np.float64]:
    """Linear-phase windowed-sinc low-pass, normalized to unit DC gain.

    `cutoff` is in cycles per sample of the upsampled rate. A Blackman window
    keeps the stopband low enough that decimation does not fold audible energy
    back into the band the pitch extractor reads.
    """
    center = (taps - 1) / 2
    offsets = np.arange(taps, dtype=np.float64) - center
    with np.errstate(invalid="ignore", divide="ignore"):
        kernel = np.sin(2.0 * np.pi * cutoff * offsets) / (np.pi * offsets)
    kernel[offsets == 0.0] = 2.0 * cutoff

    ratios = np.arange(taps, dtype=np.float64) / (taps - 1)
    kernel *= 0.42 - 0.5 * np.cos(2.0 * np.pi * ratios) + 0.08 * np.cos(4.0 * np.pi * ratios)

    total = float(kernel.sum())
    if total == 0.0 or not math.isfinite(total):
        raise AudioNormalizationError("resampling filter is degenerate")
    return kernel / total


def _resample(
    samples: np.typing.NDArray[np.float64], source_rate: int, target_rate: int
) -> np.typing.NDArray[np.float64]:
    """Deterministic rational polyphase resampling.

    Upsample by `up`, low-pass, decimate by `down`, evaluated directly through
    the filter's polyphase decomposition so no zero-stuffed signal is built.
    Output is produced in chunks, so peak memory does not grow with duration.
    """
    if source_rate == target_rate:
        return samples

    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    taps = _TAPS_PER_PHASE * up + 1
    # Band-limit to whichever Nyquist is lower, expressed against the upsampled rate.
    kernel = _blackman_sinc(taps, 0.5 / max(up, down))

    # Phase p holds kernel[p], kernel[p + up], ... Padding to a rectangle lets
    # every phase be gathered in one indexed read.
    padded = np.zeros(up * _TAPS_PER_PHASE + up, dtype=np.float64)
    padded[:taps] = kernel
    # Compensate for the energy lost to zero-stuffing.
    phases = padded.reshape(-1, up).T * up

    center = (taps - 1) // 2
    frame_count = samples.size
    output_count = frame_count * up // down
    taps_per_phase = phases.shape[1]

    resampled = np.empty(output_count, dtype=np.float64)
    for start in range(0, output_count, _RESAMPLE_CHUNK):
        stop = min(start + _RESAMPLE_CHUNK, output_count)
        positions = np.arange(start, stop, dtype=np.int64) * down + center
        phase_index = positions % up
        base = positions // up

        # Only the first and last chunks can reach outside the signal. Copying
        # a zero-padded array to avoid the check would double peak memory,
        # which is what actually bounds how long a recording can be.
        interior = base[0] >= taps_per_phase - 1 and base[-1] < frame_count

        total = np.zeros(stop - start, dtype=np.float64)
        for offset in range(taps_per_phase):
            source_index = base - offset
            coefficients = phases[phase_index, offset]
            if interior:
                total += coefficients * samples[source_index]
                continue
            np.clip(source_index, 0, frame_count - 1, out=source_index)
            inside = (base - offset >= 0) & (base - offset < frame_count)
            total += coefficients * samples[source_index] * inside
        resampled[start:stop] = total
    return resampled


def _quantize(values: np.typing.NDArray[np.float64]) -> np.typing.NDArray[np.int16]:
    """Round half away from zero, then clamp into signed 16-bit range.

    Chunked and written in place: whole-array temporaries here were the peak of
    the whole conversion, which is what limits how long a recording can be.
    """
    quantized = np.empty(values.size, dtype=np.int16)
    for start in range(0, values.size, _RESAMPLE_CHUNK):
        stop = min(start + _RESAMPLE_CHUNK, values.size)
        block = values[start:stop] * _INT16_PEAK
        # floor(|x| + 0.5) carrying x's sign is round-half-away-from-zero.
        np.abs(block, out=block)
        np.floor(block + 0.5, out=block)
        np.copysign(block, values[start:stop], out=block)
        np.clip(block, _INT16_FLOOR, _INT16_PEAK, out=block)
        quantized[start:stop] = block.astype(np.int16)
    return quantized


def _read_mono_samples(
    source_bytes: bytes, inspection: AudioInspection
) -> np.typing.NDArray[np.float64]:
    """Decode to mono float samples in [-1, 1] by averaging channels."""
    channels = inspection.channels
    sample_width = inspection.sample_width_bits // 8
    full_scale = float(1 << (inspection.sample_width_bits - 1))

    mono = np.empty(inspection.frame_count, dtype=np.float64)
    written = 0
    with wave.open(io.BytesIO(source_bytes), "rb") as stream:
        remaining_frames = inspection.frame_count
        while remaining_frames > 0:
            requested = min(_CHUNK_FRAMES, remaining_frames)
            frames = stream.readframes(requested)
            decoded = np.fromiter(_decode_pcm(frames, sample_width), dtype=np.float64, count=-1)
            block = decoded.reshape(-1, channels)
            remaining_frames -= block.shape[0]
            mono[written : written + block.shape[0]] = block.mean(axis=1) / full_scale
            written += block.shape[0]
    return mono[:written]


def normalize_wav(path: Path, output_path: Path) -> NormalizationManifest:
    """Convert a PCM WAV to canonical mono 16 kHz signed 16-bit PCM.

    Inspection runs first, so normalization inherits its validation, provenance
    hash, and fail-closed behavior. Already-canonical audio keeps its samples
    bit-for-bit.
    """
    inspection = inspect_wav(path)
    source = Path(inspection.source_path)
    destination = output_path.expanduser().resolve()

    if destination == source:
        raise AudioNormalizationError("output path must differ from the source audio")
    if inspection.clipping:
        raise AudioNormalizationError(
            "source audio is clipped; re-record or re-export below full scale"
        )
    if inspection.duration_s > MAX_NORMALIZE_SECONDS:
        raise AudioNormalizationError(
            f"audio duration {inspection.duration_s:.1f}s exceeds the "
            f"{MAX_NORMALIZE_SECONDS:.0f}s normalization limit"
        )

    source_bytes = source.read_bytes()
    samples_modified = inspection.conversion_required

    if samples_modified:
        mono = _read_mono_samples(source_bytes, inspection)
        # The encoded source is no longer needed, and on a long recording it is
        # hundreds of megabytes competing with the decoded signal.
        source_bytes = b""
        resampled = _resample(mono, inspection.sample_rate, TARGET_SAMPLE_RATE)
        del mono
        if resampled.size == 0:
            raise AudioNormalizationError("normalization produced no audio frames")
        payload = _quantize(resampled).astype("<i2").tobytes()
        del resampled
        channel_mix = "identity" if inspection.channels == 1 else "mean"
        if inspection.sample_rate == TARGET_SAMPLE_RATE:
            resampler, ratio, taps_per_phase = "none", "1/1", 0
        else:
            divisor = math.gcd(inspection.sample_rate, TARGET_SAMPLE_RATE)
            resampler = "polyphase-blackman-sinc"
            ratio = f"{TARGET_SAMPLE_RATE // divisor}/{inspection.sample_rate // divisor}"
            taps_per_phase = _TAPS_PER_PHASE
    else:
        # Already canonical: carry the PCM frames across untouched.
        with wave.open(io.BytesIO(source_bytes), "rb") as stream:
            payload = stream.readframes(inspection.frame_count)
        channel_mix, resampler, ratio, taps_per_phase = "identity", "none", "1/1", 0

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as target:
            target.setnchannels(TARGET_CHANNELS)
            target.setsampwidth(TARGET_SAMPLE_WIDTH)
            target.setframerate(TARGET_SAMPLE_RATE)
            target.writeframes(payload)
    except OSError as exc:
        raise AudioNormalizationError(f"cannot write normalized audio: {destination}") from exc

    output_bytes = destination.read_bytes()
    frame_count = len(payload) // TARGET_SAMPLE_WIDTH
    peak = max((abs(value[0]) for value in struct.iter_unpack("<h", payload)), default=0)

    return NormalizationManifest(
        schema_version="audio-normalization@0.1.0",
        source_path=str(source),
        source_sha256=inspection.source_sha256,
        source_sample_rate=inspection.sample_rate,
        source_channels=inspection.channels,
        source_sample_width_bits=inspection.sample_width_bits,
        source_frame_count=inspection.frame_count,
        source_duration_s=inspection.duration_s,
        output_path=str(destination),
        output_sha256=hashlib.sha256(output_bytes).hexdigest(),
        output_sample_rate=TARGET_SAMPLE_RATE,
        output_channels=TARGET_CHANNELS,
        output_sample_width_bits=TARGET_SAMPLE_WIDTH * 8,
        output_frame_count=frame_count,
        output_duration_s=frame_count / TARGET_SAMPLE_RATE,
        output_peak_amplitude=min(1.0, peak / float(_INT16_PEAK)),
        samples_modified=samples_modified,
        channel_mix=channel_mix,
        resampler=resampler,
        resample_ratio=ratio,
        filter_taps_per_phase=taps_per_phase,
    )
