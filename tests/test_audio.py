from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from prosody_markup.audio import MAX_WAV_BYTES, AudioInspectionError, inspect_wav


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
    duration_s: float = 0.1,
    amplitude: float = 0.5,
) -> None:
    frame_count = int(sample_rate * duration_s)
    samples = [
        int(math.sin(2 * math.pi * 220 * index / sample_rate) * amplitude * 32767)
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
