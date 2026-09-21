from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from prosody_markup import audio as audio_module
from prosody_markup.audio import (
    MAX_WAV_BYTES,
    AudioInspectionError,
    AudioNormalizationError,
    inspect_wav,
    normalize_wav,
)


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
    duration_s: float = 0.1,
    amplitude: float = 0.5,
    frequency: float = 220.0,
) -> None:
    frame_count = int(sample_rate * duration_s)
    samples = [
        int(math.sin(2 * math.pi * frequency * index / sample_rate) * amplitude * 32767)
        for index in range(frame_count)
    ]
    frames = b"".join(
        struct.pack("<h", sample) for sample in samples for _channel in range(channels)
    )
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        target.writeframes(frames)


def test_inspect_canonical_wav_without_mutating_source(tmp_path: Path) -> None:
    audio_path = tmp_path / "canonical.wav"
    _write_wav(audio_path)
    before = hashlib.sha256(audio_path.read_bytes()).hexdigest()

    result = inspect_wav(audio_path)

    assert result.schema_version == "audio-inspection@0.1.0"
    assert result.container == "wav"
    assert result.encoding == "pcm"
    assert result.duration_s == pytest.approx(0.1)
    assert result.sample_rate == 16_000
    assert result.channels == 1
    assert result.sample_width_bits == 16
    assert result.frame_count == 1_600
    assert result.peak_amplitude == pytest.approx(0.5, abs=0.01)
    assert result.clipping is False
    assert 0 <= result.silence_ratio <= 1
    assert result.conversion_required is False
    assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == before


def test_inspect_reports_conversion_for_noncanonical_pcm(tmp_path: Path) -> None:
    audio_path = tmp_path / "stereo.wav"
    _write_wav(audio_path, sample_rate=44_100, channels=2)

    result = inspect_wav(audio_path)

    assert result.channels == 2
    assert result.sample_rate == 44_100
    assert result.conversion_required is True


def test_inspect_detects_clipping(tmp_path: Path) -> None:
    audio_path = tmp_path / "clipped.wav"
    _write_wav(audio_path, amplitude=1.0)

    result = inspect_wav(audio_path)

    assert result.clipping is True


def test_inspect_rejects_empty_wav(tmp_path: Path) -> None:
    audio_path = tmp_path / "empty.wav"
    with wave.open(str(audio_path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)

    with pytest.raises(AudioInspectionError, match="contains no audio frames"):
        inspect_wav(audio_path)


def test_inspect_rejects_malformed_input(tmp_path: Path) -> None:
    audio_path = tmp_path / "not-a-wave.wav"
    audio_path.write_text("not audio", encoding="utf-8")

    with pytest.raises(AudioInspectionError, match="valid PCM WAV"):
        inspect_wav(audio_path)


def test_inspect_rejects_files_above_safety_limit_before_reading(tmp_path: Path) -> None:
    audio_path = tmp_path / "too-large.wav"
    with audio_path.open("wb") as stream:
        stream.truncate(MAX_WAV_BYTES + 1)

    with pytest.raises(AudioInspectionError, match="exceeds the .* safety limit"):
        inspect_wav(audio_path)


def test_inspect_rejects_truncated_pcm_frames(tmp_path: Path) -> None:
    audio_path = tmp_path / "truncated.wav"
    payload = b"\x00\x00\x01"
    declared_size = len(payload) + 1
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + declared_size)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 16_000, 32_000, 2, 16)
        + b"data"
        + struct.pack("<I", declared_size)
    )
    audio_path.write_bytes(header + payload)

    with pytest.raises(AudioInspectionError, match="truncated"):
        inspect_wav(audio_path)


def test_audio_inspect_cli_emits_json(tmp_path: Path) -> None:
    audio_path = tmp_path / "canonical.wav"
    _write_wav(audio_path)

    completed = subprocess.run(
        [sys.executable, "-m", "prosody_markup.cli", "audio", "inspect", str(audio_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["sample_rate"] == 16_000
    assert payload["conversion_required"] is False


def test_audio_inspect_cli_reports_invalid_input_without_traceback(tmp_path: Path) -> None:
    audio_path = tmp_path / "not-a-wave.wav"
    audio_path.write_text("not audio", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "prosody_markup.cli", "audio", "inspect", str(audio_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "valid PCM WAV" in completed.stderr
    assert "Traceback" not in completed.stderr


def _read_samples(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as stream:
        frames = stream.readframes(stream.getnframes())
    return [value[0] for value in struct.iter_unpack("<h", frames)]


def _steady_state_rms(samples: list[int]) -> float:
    """RMS of the middle half, which excludes linear-phase filter edge ringing."""
    window = samples[len(samples) // 4 : 3 * len(samples) // 4]
    return math.sqrt(sum(value * value for value in window) / len(window))


def test_normalize_converts_to_canonical_format(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    output = tmp_path / "normalized.wav"
    _write_wav(source, sample_rate=44_100, channels=2, duration_s=0.25)

    manifest = normalize_wav(source, output)

    assert manifest.schema_version == "audio-normalization@0.1.0"
    assert manifest.output_sample_rate == 16_000
    assert manifest.output_channels == 1
    assert manifest.output_sample_width_bits == 16
    assert manifest.output_frame_count == 4_000
    assert manifest.output_duration_s == pytest.approx(0.25)
    assert manifest.samples_modified is True
    assert manifest.channel_mix == "mean"
    assert manifest.resampler == "polyphase-blackman-sinc"
    assert manifest.resample_ratio == "160/441"

    with wave.open(str(output), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getframerate() == 16_000
        assert stream.getsampwidth() == 2


def test_normalize_records_source_provenance(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    output = tmp_path / "normalized.wav"
    _write_wav(source, sample_rate=44_100, channels=2, duration_s=0.1)

    manifest = normalize_wav(source, output)

    assert manifest.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest.output_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest.source_sample_rate == 44_100
    assert manifest.source_channels == 2


def test_normalize_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    _write_wav(source, sample_rate=44_100, channels=2, duration_s=0.1)

    first = normalize_wav(source, tmp_path / "first.wav")
    second = normalize_wav(source, tmp_path / "second.wav")

    assert first.output_sha256 == second.output_sha256


def test_normalize_passes_through_canonical_audio_without_quality_loss(tmp_path: Path) -> None:
    source = tmp_path / "canonical.wav"
    output = tmp_path / "normalized.wav"
    _write_wav(source, duration_s=0.1)

    manifest = normalize_wav(source, output)

    assert manifest.samples_modified is False
    assert manifest.resampler == "none"
    assert manifest.channel_mix == "identity"
    assert _read_samples(output) == _read_samples(source)


def test_normalize_preserves_a_tone_below_the_target_nyquist(tmp_path: Path) -> None:
    source = tmp_path / "tone.wav"
    output = tmp_path / "normalized.wav"
    _write_wav(source, sample_rate=44_100, duration_s=0.25, frequency=1_000.0, amplitude=0.5)

    normalize_wav(source, output)

    expected_rms = 0.5 * 32767 / math.sqrt(2)
    assert _steady_state_rms(_read_samples(output)) == pytest.approx(expected_rms, rel=0.05)


def test_normalize_attenuates_content_above_the_target_nyquist(tmp_path: Path) -> None:
    """Decimation must not fold 15 kHz energy back into the pitch band."""
    source = tmp_path / "ultrasonic.wav"
    output = tmp_path / "normalized.wav"
    _write_wav(source, sample_rate=44_100, duration_s=0.25, frequency=15_000.0, amplitude=0.5)

    normalize_wav(source, output)

    input_rms = 0.5 * 32767 / math.sqrt(2)
    assert _steady_state_rms(_read_samples(output)) < input_rms / 1_000


def test_normalize_rejects_clipped_input(tmp_path: Path) -> None:
    source = tmp_path / "clipped.wav"
    _write_wav(source, amplitude=1.0)

    with pytest.raises(AudioNormalizationError, match="clipped"):
        normalize_wav(source, tmp_path / "normalized.wav")


def test_normalize_rejects_empty_input(tmp_path: Path) -> None:
    source = tmp_path / "empty.wav"
    with wave.open(str(source), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)

    with pytest.raises(AudioInspectionError, match="contains no audio frames"):
        normalize_wav(source, tmp_path / "normalized.wav")


def test_normalize_refuses_to_overwrite_its_source(tmp_path: Path) -> None:
    source = tmp_path / "canonical.wav"
    _write_wav(source)

    with pytest.raises(AudioNormalizationError, match="must differ from the source"):
        normalize_wav(source, source)


def test_audio_normalize_cli_writes_audio_and_sidecar_manifest(tmp_path: Path) -> None:
    source = tmp_path / "stereo.wav"
    output = tmp_path / "normalized.wav"
    _write_wav(source, sample_rate=44_100, channels=2, duration_s=0.1)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "audio",
            "normalize",
            str(source),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["output_sample_rate"] == 16_000

    sidecar = Path(str(output) + ".manifest.json")
    assert json.loads(sidecar.read_text(encoding="utf-8")) == payload
    assert inspect_wav(output).conversion_required is False


def test_audio_normalize_cli_reports_clipped_input_without_traceback(tmp_path: Path) -> None:
    source = tmp_path / "clipped.wav"
    output = tmp_path / "normalized.wav"
    _write_wav(source, amplitude=1.0)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "audio",
            "normalize",
            str(source),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "clipped" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()


def test_normalize_refuses_audio_beyond_the_duration_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The limit is memory, not patience, so it fails closed rather than thrashing."""
    source = tmp_path / "long.wav"
    _write_wav(source, sample_rate=44_100, duration_s=0.2)
    monkeypatch.setattr(audio_module, "MAX_NORMALIZE_SECONDS", 0.05)

    with pytest.raises(AudioNormalizationError, match="normalization limit"):
        normalize_wav(source, tmp_path / "out.wav")


def test_quantization_rounds_half_away_from_zero() -> None:
    """Banker's rounding would bias the waveform, so halves go outward."""
    full_scale = 32767.0
    values = np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5]) / full_scale

    quantized = audio_module._quantize(values)

    assert quantized.tolist() == [1, -1, 2, -2, 3, -3]


def test_quantization_clamps_into_range() -> None:
    quantized = audio_module._quantize(np.array([2.0, -2.0]))

    assert quantized.tolist() == [32767, -32768]


def test_resampling_is_chunk_boundary_independent(tmp_path: Path) -> None:
    """Output must not depend on how the work was divided internally."""
    source = tmp_path / "tone.wav"
    _write_wav(source, sample_rate=44_100, duration_s=0.5, frequency=300.0)

    default = normalize_wav(source, tmp_path / "default.wav").output_sha256
    audio_module._RESAMPLE_CHUNK, original = 97, audio_module._RESAMPLE_CHUNK
    try:
        tiny = normalize_wav(source, tmp_path / "tiny.wav").output_sha256
    finally:
        audio_module._RESAMPLE_CHUNK = original

    assert default == tiny
