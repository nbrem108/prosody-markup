from __future__ import annotations

import hashlib
import io
import math
import struct
import wave
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_WAV_BYTES = 256 * 1024 * 1024
_CHUNK_FRAMES = 65_536


class AudioInspectionError(ValueError):
    """Raised when an input cannot satisfy the local WAV inspection contract."""


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
