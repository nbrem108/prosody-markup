from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from prosody_markup.cli import build_parser
from prosody_markup.transcribe import (
    FasterWhisperAdapter,
    TranscribedWord,
    TranscriptionError,
    TranscriptionResult,
    build_transcript,
)


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    duration_s: float = 1.0,
) -> None:
    frame_count = int(sample_rate * duration_s)
    frames = b"".join(
        struct.pack("<h", int(math.sin(2 * math.pi * 220 * index / sample_rate) * 0.5 * 32767))
        for index in range(frame_count)
        for _channel in range(channels)
    )
    with wave.open(str(path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(frames)


class FakeAdapter:
    """Deterministic stand-in for a real engine, fixed at the adapter boundary."""

    def __init__(self, words: list[TranscribedWord] | None = None) -> None:
        self.words = (
            words
            if words is not None
            else [
                TranscribedWord("Hello", 0.0, 0.3, 0.95, space_before=False),
                TranscribedWord("there", 0.35, 0.7, 0.88),
            ]
        )
        self.calls: list[Path] = []

    def transcribe(self, path: Path) -> TranscriptionResult:
        self.calls.append(path)
        return TranscriptionResult(
            words=list(self.words),
            model_name="fake/deterministic",
            model_version="1.0.0",
            options={"device": "cpu"},
        )


def test_build_transcript_produces_word_timestamp_ir(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)

    document = build_transcript(audio, FakeAdapter(), speaker="S1")

    assert document.schema_version == "0.1.0"
    assert [token.id for token in document.tokens] == ["w0001", "w0002"]
    assert [token.text for token in document.tokens] == ["Hello", "there"]
    assert [token.start for token in document.tokens] == [0.0, 0.35]
    assert all(token.speaker == "S1" for token in document.tokens)
    assert all(token.turn_id == "turn-1" for token in document.tokens)
    assert all(token.speaker_confidence == 1.0 for token in document.tokens)
    assert [token.space_before for token in document.tokens] == [False, True]


def test_build_transcript_records_model_and_audio_provenance(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)

    document = build_transcript(audio, FakeAdapter(), speaker="S1")

    assert document.provenance["kind"] == "local-asr"
    assert document.provenance["asr_model"] == "fake/deterministic"
    assert document.provenance["asr_model_version"] == "1.0.0"
    assert document.audio["sample_rate"] == 16_000
    assert document.audio["duration_s"] == pytest.approx(1.0)
    assert len(document.audio["sha256"]) == 64


def test_build_transcript_marks_alignment_confidence_as_inherited(tmp_path: Path) -> None:
    """Confidence provenance must not imply alignment was measured separately."""
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)

    inherited = build_transcript(audio, FakeAdapter(), speaker="S1")
    assert inherited.provenance["alignment"] == "asr-inherited"
    assert inherited.tokens[0].alignment_confidence == inherited.tokens[0].asr_confidence

    reported = build_transcript(
        audio,
        FakeAdapter([TranscribedWord("Hi", 0.0, 0.2, 0.9, alignment_confidence=0.7)]),
        speaker="S1",
    )
    assert reported.provenance["alignment"] == "adapter-reported"
    assert reported.tokens[0].alignment_confidence == 0.7


def test_build_transcript_keeps_low_confidence_words(tmp_path: Path) -> None:
    """Suppression is the legend's decision at assignment, not the adapter's."""
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)
    words = [
        TranscribedWord("certain", 0.0, 0.3, 0.99),
        TranscribedWord("mumbled", 0.35, 0.6, 0.05),
    ]

    document = build_transcript(audio, FakeAdapter(words), speaker="S1")

    assert [token.text for token in document.tokens] == ["certain", "mumbled"]
    assert document.tokens[1].asr_confidence == 0.05


def test_build_transcript_leaves_baseline_pending(tmp_path: Path) -> None:
    """Per-speaker normalization is a later stage, so the baseline is not invented."""
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)

    document = build_transcript(audio, FakeAdapter(), speaker="S1")

    assert document.baseline == {"speaker": "S1", "status": "pending-normalization"}
    assert all(token.features == {} for token in document.tokens)


def test_build_transcript_requires_canonical_audio(tmp_path: Path) -> None:
    audio = tmp_path / "stereo.wav"
    _write_wav(audio, sample_rate=44_100, channels=2, duration_s=0.2)

    with pytest.raises(TranscriptionError, match="audio normalize"):
        build_transcript(audio, FakeAdapter(), speaker="S1")


def test_build_transcript_requires_a_speaker(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)

    with pytest.raises(TranscriptionError, match="speaker identity is required"):
        build_transcript(audio, FakeAdapter(), speaker="  ")


def test_build_transcript_rejects_empty_transcription(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)

    with pytest.raises(TranscriptionError, match="produced no words"):
        build_transcript(audio, FakeAdapter([]), speaker="S1")


def test_build_transcript_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)
    words = [
        TranscribedWord("second", 0.5, 0.8, 0.9),
        TranscribedWord("first", 0.1, 0.4, 0.9),
    ]

    with pytest.raises(TranscriptionError, match="monotonic"):
        build_transcript(audio, FakeAdapter(words), speaker="S1")


def test_build_transcript_rejects_words_past_the_recording(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    _write_wav(audio, duration_s=0.5)
    words = [TranscribedWord("overrun", 0.1, 9.0, 0.9)]

    with pytest.raises(TranscriptionError, match="past the"):
        build_transcript(audio, FakeAdapter(words), speaker="S1")


def test_build_transcript_rejects_out_of_range_confidence(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)
    words = [TranscribedWord("impossible", 0.0, 0.3, 1.5)]

    with pytest.raises(TranscriptionError, match="between 0 and 1"):
        build_transcript(audio, FakeAdapter(words), speaker="S1")


def test_build_transcript_rejects_inverted_word_bounds(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)
    words = [TranscribedWord("backwards", 0.4, 0.1, 0.9)]

    with pytest.raises(TranscriptionError, match="ends before it starts"):
        build_transcript(audio, FakeAdapter(words), speaker="S1")


def _faster_whisper_installed() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def test_faster_whisper_adapter_reports_a_missing_dependency(tmp_path: Path) -> None:
    """The optional engine must fail with guidance, not an ImportError traceback."""
    if _faster_whisper_installed():
        pytest.skip("faster-whisper is installed in this environment")

    with pytest.raises(TranscriptionError, match="faster-whisper is not installed"):
        FasterWhisperAdapter().transcribe(tmp_path / "missing.wav")


def test_transcribe_document_serializes_to_ir_json(tmp_path: Path) -> None:
    audio = tmp_path / "normalized.wav"
    _write_wav(audio)

    payload = json.loads(json.dumps(build_transcript(audio, FakeAdapter(), speaker="S1").to_dict()))

    assert payload["tokens"][0]["id"] == "w0001"
    assert payload["tokens"][0]["marks"] == []
    assert payload["provenance"]["kind"] == "local-asr"
    assert payload["baseline"]["status"] == "pending-normalization"


def test_transcribe_cli_parses_its_arguments() -> None:
    args = build_parser().parse_args(
        ["transcribe", "in.wav", "--output", "out.json", "--speaker", "S1", "--turn-id", "turn-9"]
    )

    assert args.input == Path("in.wav")
    assert args.output == Path("out.json")
    assert args.speaker == "S1"
    assert args.turn_id == "turn-9"
    assert args.model == "base.en"


def test_transcribe_cli_reports_uncanonical_audio_without_traceback(tmp_path: Path) -> None:
    audio = tmp_path / "stereo.wav"
    _write_wav(audio, sample_rate=44_100, channels=2, duration_s=0.2)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "transcribe",
            str(audio),
            "--output",
            str(tmp_path / "transcript.json"),
            "--speaker",
            "S1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "audio normalize" in completed.stderr
    assert "Traceback" not in completed.stderr


@pytest.mark.integration
def test_faster_whisper_transcribes_real_audio(tmp_path: Path) -> None:
    """Opt-in only: needs faster-whisper and a local model download.

    Run with PROSODY_ASR_MODEL set and `-m integration`.
    """
    model = os.environ.get("PROSODY_ASR_MODEL")
    if not model:
        pytest.skip("set PROSODY_ASR_MODEL to run the real-model integration test")

    audio = tmp_path / "normalized.wav"
    _write_wav(audio, duration_s=2.0)

    document = build_transcript(audio, FasterWhisperAdapter(model), speaker="S1")

    assert document.provenance["asr_model"] == f"faster-whisper/{model}"
