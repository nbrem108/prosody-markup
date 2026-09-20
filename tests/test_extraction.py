from __future__ import annotations

import csv
import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from prosody_markup.extract import (
    ExtractionError,
    PitchSettings,
    extract_pitch,
    write_debug_artifacts,
)
from prosody_markup.models import Document, Token

SAMPLE_RATE = 16_000


def _voiced(duration_s: float, f0_hz: float) -> np.typing.NDArray[np.float64]:
    """A harmonic-rich tone, which Praat reads as periodic (voiced) speech."""
    times = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    signal = np.zeros_like(times)
    for harmonic, amplitude in enumerate([1.0, 0.5, 0.25, 0.12], start=1):
        signal += amplitude * np.sin(2 * np.pi * f0_hz * harmonic * times)
    return 0.5 * signal / (np.abs(signal).max() + 1e-9)


def _noise(duration_s: float, seed: int = 7) -> np.typing.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.08, int(SAMPLE_RATE * duration_s))


def _silence(duration_s: float) -> np.typing.NDArray[np.float64]:
    return np.zeros(int(SAMPLE_RATE * duration_s))


def _write_signal(path: Path, parts: list[np.typing.NDArray[np.float64]]) -> None:
    signal = np.concatenate(parts) if parts else np.zeros(1)
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm.tobytes())


def _document(spans: list[tuple[str, str, float, float]]) -> Document:
    tokens = [
        Token(
            id=token_id,
            text=text,
            start=start,
            end=end,
            speaker="S1",
            asr_confidence=0.95,
            alignment_confidence=0.95,
            turn_id="turn-1",
            speaker_confidence=1.0,
        )
        for token_id, text, start, end in spans
    ]
    return Document(
        schema_version="0.1.0",
        legend_version="unassigned",
        audio={},
        provenance={"kind": "local-asr"},
        baseline={"speaker": "S1", "status": "pending-normalization"},
        tokens=tokens,
    )


def test_extracts_pitch_for_a_voiced_word(tmp_path: Path) -> None:
    audio = tmp_path / "voiced.wav"
    _write_signal(audio, [_voiced(0.4, 200.0)])
    document = _document([("w0001", "hello", 0.0, 0.4)])

    extraction = extract_pitch(document, audio)

    (word,) = extraction.words
    assert word.valid is True
    assert word.reason is None
    assert word.voiced_coverage > 0.9
    assert word.f0_mean == pytest.approx(200.0, rel=0.05)
    assert word.f0_max == pytest.approx(200.0, rel=0.05)
    assert word.f0_range == pytest.approx(0.0, abs=5.0)


def test_measurements_reach_the_token_as_raw_features(tmp_path: Path) -> None:
    audio = tmp_path / "voiced.wav"
    _write_signal(audio, [_voiced(0.4, 180.0)])
    document = _document([("w0001", "hello", 0.0, 0.4)])

    extracted = extract_pitch(document, audio).document

    token = extracted.tokens[0]
    assert token.raw_features["f0_mean_hz"] == pytest.approx(180.0, rel=0.05)
    assert set(token.raw_features) == {
        "f0_mean_hz",
        "f0_min_hz",
        "f0_max_hz",
        "f0_range_hz",
        "voiced_coverage",
    }
    assert token.feature_suppression == {}


def test_extraction_assigns_no_typography(tmp_path: Path) -> None:
    """Raw measurement only: normalized features and marks are later stages."""
    audio = tmp_path / "voiced.wav"
    _write_signal(audio, [_voiced(0.4, 200.0)])
    document = _document([("w0001", "hello", 0.0, 0.4)])

    extracted = extract_pitch(document, audio).document

    assert all(token.marks == [] for token in extracted.tokens)
    assert all(token.features == {} for token in extracted.tokens)
    assert extracted.legend_version == "unassigned"


def test_unvoiced_word_is_suppressed_with_a_reason(tmp_path: Path) -> None:
    audio = tmp_path / "silent.wav"
    _write_signal(audio, [_silence(0.4)])
    document = _document([("w0001", "hush", 0.0, 0.4)])

    extraction = extract_pitch(document, audio)

    (word,) = extraction.words
    assert word.valid is False
    assert word.reason == "unvoiced"
    assert word.f0_mean is None
    assert extraction.document.tokens[0].feature_suppression == {"pitch": "unvoiced"}


def test_noise_is_suppressed_for_insufficient_voiced_coverage(tmp_path: Path) -> None:
    audio = tmp_path / "noise.wav"
    _write_signal(audio, [_noise(0.4)])
    document = _document([("w0001", "static", 0.0, 0.4)])

    (word,) = extract_pitch(document, audio).words

    assert word.valid is False
    assert word.reason in {"unvoiced", "insufficient_voiced_coverage"}
    assert word.f0_mean is None


def test_short_window_is_suppressed_rather_than_guessed(tmp_path: Path) -> None:
    audio = tmp_path / "voiced.wav"
    _write_signal(audio, [_voiced(0.4, 200.0)])
    document = _document([("w0001", "a", 0.0, 0.02)])

    (word,) = extract_pitch(document, audio).words

    assert word.valid is False
    assert word.reason == "window_too_short"


def test_subharmonic_is_suppressed_as_octave_ambiguous(tmp_path: Path) -> None:
    """A ceiling below the true F0 makes Praat report a confident halved value.

    It reports 100 Hz for a 200 Hz tone at full voiced coverage and with no
    within-word spread, so neither a range check nor a spread check sees it.
    Only re-measuring at a raised ceiling exposes the disagreement.
    """
    audio = tmp_path / "high.wav"
    _write_signal(audio, [_voiced(0.4, 200.0)])
    document = _document([("w0001", "hello", 0.0, 0.4)])

    settings = PitchSettings(floor_hz=75.0, ceiling_hz=150.0)
    (word,) = extract_pitch(document, audio, settings).words

    assert word.valid is False
    assert word.reason == "octave_ambiguous"
    assert word.f0_mean is None


def test_honest_measurement_survives_octave_verification(tmp_path: Path) -> None:
    """The octave check must not suppress a pitch the ceiling can actually fit."""
    audio = tmp_path / "clean.wav"
    _write_signal(audio, [_voiced(0.4, 200.0)])
    document = _document([("w0001", "hello", 0.0, 0.4)])

    (word,) = extract_pitch(document, audio, PitchSettings(ceiling_hz=500.0)).words

    assert word.valid is True
    assert word.f0_mean == pytest.approx(200.0, rel=0.05)


def test_out_of_range_pitch_is_suppressed(tmp_path: Path) -> None:
    """A measurement below the configured floor is refused outright."""
    audio = tmp_path / "low.wav"
    _write_signal(audio, [_voiced(0.4, 120.0)])
    document = _document([("w0001", "hello", 0.0, 0.4)])

    (word,) = extract_pitch(document, audio, PitchSettings(floor_hz=200.0)).words

    assert word.valid is False
    assert word.f0_mean is None


def test_mixed_utterance_measures_only_the_measurable_words(tmp_path: Path) -> None:
    audio = tmp_path / "mixed.wav"
    _write_signal(audio, [_voiced(0.4, 200.0), _silence(0.3), _voiced(0.4, 220.0)])
    document = _document(
        [
            ("w0001", "first", 0.0, 0.4),
            ("w0002", "quiet", 0.4, 0.7),
            ("w0003", "third", 0.7, 1.1),
        ]
    )

    extraction = extract_pitch(document, audio)

    assert [word.valid for word in extraction.words] == [True, False, True]
    assert [word.reason for word in extraction.words] == [
        None,
        "insufficient_voiced_coverage",
        None,
    ]


def test_punctuation_tokens_are_not_measured(tmp_path: Path) -> None:
    audio = tmp_path / "voiced.wav"
    _write_signal(audio, [_voiced(0.4, 200.0)])
    document = _document([("w0001", "hello", 0.0, 0.4), ("w0002", ",", 0.4, 0.4)])

    extraction = extract_pitch(document, audio)

    assert [word.token_id for word in extraction.words] == ["w0001"]
    assert extraction.document.tokens[1].raw_features == {}


def test_extraction_records_extractor_provenance(tmp_path: Path) -> None:
    audio = tmp_path / "voiced.wav"
    _write_signal(audio, [_voiced(0.4, 200.0)])
    document = _document([("w0001", "hello", 0.0, 0.4)])

    provenance = extract_pitch(document, audio).document.provenance

    assert provenance["extractor"] == "praat-parselmouth"
    assert provenance["extraction_schema"] == "pitch-extraction@0.1.0"
    assert provenance["extractor_version"]
    assert provenance["extraction_settings"]["ceiling_hz"] == 500.0
    assert provenance["extraction_settings"]["floor_hz"] == 75.0
    assert provenance["kind"] == "local-asr"


def test_extraction_requires_canonical_audio(tmp_path: Path) -> None:
    audio = tmp_path / "stereo.wav"
    frames = b"".join(struct.pack("<hh", 1000, 1000) for _ in range(4_410))
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(44_100)
        target.writeframes(frames)

    with pytest.raises(ExtractionError, match="audio normalize"):
        extract_pitch(_document([("w0001", "hello", 0.0, 0.1)]), audio)


def test_extraction_rejects_a_transcript_past_the_recording(tmp_path: Path) -> None:
    audio = tmp_path / "voiced.wav"
    _write_signal(audio, [_voiced(0.2, 200.0)])

    with pytest.raises(ExtractionError, match="past the"):
        extract_pitch(_document([("w0001", "hello", 0.0, 9.0)]), audio)


def test_extraction_rejects_an_empty_transcript(tmp_path: Path) -> None:
    audio = tmp_path / "voiced.wav"
    _write_signal(audio, [_voiced(0.2, 200.0)])

    with pytest.raises(ExtractionError, match="no tokens"):
        extract_pitch(_document([]), audio)


def test_debug_artifacts_are_written(tmp_path: Path) -> None:
    audio = tmp_path / "mixed.wav"
    _write_signal(audio, [_voiced(0.4, 200.0), _silence(0.3)])
    document = _document([("w0001", "first", 0.0, 0.4), ("w0002", "quiet", 0.4, 0.7)])

    extraction = extract_pitch(document, audio)
    features_path, contour_path = write_debug_artifacts(extraction, tmp_path / "run-001")

    assert features_path.name == "word-features.csv"
    assert contour_path.name == "pitch-contour.csv"

    rows = list(csv.DictReader(features_path.read_text(encoding="utf-8").splitlines()))
    assert [row["token_id"] for row in rows] == ["w0001", "w0002"]
    assert rows[0]["valid"] == "True"
    assert rows[1]["reason"] == "insufficient_voiced_coverage"

    contour = list(csv.DictReader(contour_path.read_text(encoding="utf-8").splitlines()))
    assert contour
    assert all(row["token_id"] == "w0001" for row in contour)
    assert all(float(row["f0_hz"]) > 0 for row in contour)


def test_extract_cli_writes_ir_and_debug_bundle(tmp_path: Path) -> None:
    audio = tmp_path / "voiced.wav"
    _write_signal(audio, [_voiced(0.4, 200.0)])
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(_document([("w0001", "hello", 0.0, 0.4)]).to_dict()), encoding="utf-8"
    )
    output = tmp_path / "features.json"
    debug_dir = tmp_path / "run-001"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "extract",
            str(audio),
            str(transcript),
            "--output",
            str(output),
            "--debug-dir",
            str(debug_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "measured 1/1 eligible words" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["tokens"][0]["raw_features"]["f0_mean_hz"] == pytest.approx(200.0, rel=0.05)
    assert payload["provenance"]["extractor"] == "praat-parselmouth"
    assert (debug_dir / "word-features.csv").exists()
    assert (debug_dir / "pitch-contour.csv").exists()


def test_extract_cli_reports_uncanonical_audio_without_traceback(tmp_path: Path) -> None:
    audio = tmp_path / "stereo.wav"
    frames = b"".join(struct.pack("<hh", 1000, 1000) for _ in range(4_410))
    with wave.open(str(audio), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(44_100)
        target.writeframes(frames)
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(_document([("w0001", "hello", 0.0, 0.05)]).to_dict()), encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "extract",
            str(audio),
            str(transcript),
            "--output",
            str(tmp_path / "features.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "audio normalize" in completed.stderr
    assert "Traceback" not in completed.stderr
