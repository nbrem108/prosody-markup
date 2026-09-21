from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from prosody_markup.pipeline import PipelineError, run_pipeline
from prosody_markup.transcribe import TranscribedWord, TranscriptionError, TranscriptionResult

SOURCE_RATE = 44_100

# One emphatic word among otherwise level speech.
_WORDS = (
    ("I", 150.0),
    ("said", 158.0),
    ("the", 145.0),
    ("BLUE", 250.0),
    ("one", 152.0),
    ("not", 147.0),
    ("the", 155.0),
    ("red", 149.0),
    ("one", 151.0),
)
_WORD_S = 0.28
_GAP_S = 0.06


def _tone(duration_s: float, f0_hz: float) -> np.typing.NDArray[np.float64]:
    times = np.arange(int(SOURCE_RATE * duration_s)) / SOURCE_RATE
    signal = np.zeros_like(times)
    for harmonic, amplitude in enumerate([1.0, 0.5, 0.25, 0.12], start=1):
        signal += amplitude * np.sin(2 * np.pi * f0_hz * harmonic * times)
    return 0.45 * signal / (np.abs(signal).max() + 1e-9)


def _write_source(path: Path) -> list[tuple[str, float, float]]:
    """A 44.1 kHz stereo recording, so the run exercises real conversion."""
    parts: list[np.typing.NDArray[np.float64]] = []
    spans: list[tuple[str, float, float]] = []
    cursor = 0.0
    for text, f0_hz in _WORDS:
        parts.append(_tone(_WORD_S, f0_hz))
        spans.append((text, round(cursor, 3), round(cursor + _WORD_S - 0.03, 3)))
        parts.append(np.zeros(int(SOURCE_RATE * _GAP_S)))
        cursor += _WORD_S + _GAP_S

    signal = np.concatenate(parts)
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(SOURCE_RATE)
        target.writeframes(np.repeat(pcm, 2).tobytes())
    return spans


class FakeAdapter:
    def __init__(self, spans: list[tuple[str, float, float]]) -> None:
        self.spans = spans

    def transcribe(self, path: Path) -> TranscriptionResult:
        return TranscriptionResult(
            words=[
                TranscribedWord(text, start, end, 0.97, space_before=index > 0)
                for index, (text, start, end) in enumerate(self.spans)
            ],
            model_name="fake/deterministic",
            model_version="1.0.0",
        )


class BrokenAdapter:
    def transcribe(self, path: Path) -> TranscriptionResult:
        raise TranscriptionError("adapter exploded")


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, FakeAdapter]:
    audio = tmp_path / "recording.wav"
    return audio, FakeAdapter(_write_source(audio))


def test_process_runs_every_stage_to_marked_text(
    source: tuple[Path, FakeAdapter], tmp_path: Path
) -> None:
    audio, adapter = source

    result = run_pipeline(audio, adapter, tmp_path / "run-001", speaker="S1")

    assert result.rendered == "I said the *BLUE* one not the red one"
    assert result.manifest["status"] == "complete"
    assert result.manifest["completed_stages"] == [
        "inspect",
        "normalize_audio",
        "transcribe",
        "extract",
        "prominence",
        "render",
    ]


def test_process_emits_the_required_debug_bundle(
    source: tuple[Path, FakeAdapter], tmp_path: Path
) -> None:
    audio, adapter = source
    debug_dir = tmp_path / "run-001"

    run_pipeline(audio, adapter, debug_dir, speaker="S1")

    required = {
        "audio-summary.json",
        "transcript.json",
        "word-features.csv",
        "pitch-contour.csv",
        "candidates.json",
        "assigned.json",
        "rendered.md",
        "run-manifest.json",
    }
    assert required <= {path.name for path in debug_dir.iterdir()}


def test_manifest_declares_versions_for_every_stage(
    source: tuple[Path, FakeAdapter], tmp_path: Path
) -> None:
    audio, adapter = source

    manifest = run_pipeline(audio, adapter, tmp_path / "run-001", speaker="S1").manifest

    versions = manifest["versions"]
    assert versions["schema_version"] == "0.1.0"
    assert versions["legend_version"].startswith("tier1@")
    assert versions["audio_inspection_schema"] == "audio-inspection@0.1.0"
    assert versions["audio_normalization_schema"] == "audio-normalization@0.1.0"
    assert versions["extraction_schema"] == "pitch-extraction@0.1.0"
    assert versions["baseline_schema"] == "speaker-baseline@0.1.0"
    assert versions["asr_model"] == "fake/deterministic"
    assert versions["asr_model_version"] == "1.0.0"
    assert versions["extractor"] == "praat-parselmouth"
    assert versions["prosody_markup"]


def test_manifest_records_settings_and_checksums(
    source: tuple[Path, FakeAdapter], tmp_path: Path
) -> None:
    audio, adapter = source
    debug_dir = tmp_path / "run-001"

    manifest = run_pipeline(audio, adapter, debug_dir, speaker="S1").manifest

    assert manifest["settings"]["enabled_channels"] == ["pitch"]
    assert manifest["settings"]["speaker"] == "S1"
    assert manifest["settings"]["pitch"]["ceiling_hz"] == 500.0

    rows = {row["name"]: row["sha256"] for row in manifest["artifacts"]}
    assert "assigned.json" in rows
    assert all(len(digest) == 64 for digest in rows.values())
    # The manifest lists the bundle, so it cannot checksum itself.
    assert "run-manifest.json" not in rows


def test_manifest_summary_separates_candidates_from_marks(
    source: tuple[Path, FakeAdapter], tmp_path: Path
) -> None:
    audio, adapter = source

    summary = run_pipeline(audio, adapter, tmp_path / "run-001", speaker="S1").manifest["summary"]

    assert summary["words"] == len(_WORDS)
    assert summary["eligible_words"] == len(_WORDS)
    assert summary["measured_words"] == len(_WORDS)
    assert summary["candidates"] >= summary["marks"] >= 1


def test_bundle_artifacts_agree_with_each_other(
    source: tuple[Path, FakeAdapter], tmp_path: Path
) -> None:
    audio, adapter = source
    debug_dir = tmp_path / "run-001"

    result = run_pipeline(audio, adapter, debug_dir, speaker="S1")

    assigned = json.loads((debug_dir / "assigned.json").read_text(encoding="utf-8"))
    candidates = json.loads((debug_dir / "candidates.json").read_text(encoding="utf-8"))
    rendered = (debug_dir / "rendered.md").read_text(encoding="utf-8").strip()

    marked = {token["id"] for token in assigned["tokens"] if token["marks"]}
    assert {row["token_id"] for row in candidates if row["retained"]} == marked
    assert rendered == result.rendered
    assert assigned["baseline"]["method"] == "median-mad-semitones"


def test_partial_failure_keeps_what_ran_and_names_the_stage(tmp_path: Path) -> None:
    """A failed run must diagnose itself, not fabricate the missing output."""
    audio = tmp_path / "recording.wav"
    _write_source(audio)
    debug_dir = tmp_path / "run-001"

    with pytest.raises(TranscriptionError, match="adapter exploded"):
        run_pipeline(audio, BrokenAdapter(), debug_dir, speaker="S1")

    manifest = json.loads((debug_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failed_stage"] == "transcribe"
    assert "adapter exploded" in manifest["error"]
    assert manifest["completed_stages"] == ["inspect", "normalize_audio"]

    # Stages that ran left their artifacts; nothing downstream was invented.
    assert (debug_dir / "audio-summary.json").exists()
    assert not (debug_dir / "transcript.json").exists()
    assert not (debug_dir / "assigned.json").exists()
    assert not (debug_dir / "rendered.md").exists()


def test_unreadable_audio_fails_before_any_stage_output(tmp_path: Path) -> None:
    audio = tmp_path / "not-audio.wav"
    audio.write_text("not audio", encoding="utf-8")
    debug_dir = tmp_path / "run-001"

    with pytest.raises(ValueError, match="valid PCM WAV"):
        run_pipeline(audio, FakeAdapter([]), debug_dir, speaker="S1")

    manifest = json.loads((debug_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["failed_stage"] == "inspect"
    assert manifest["completed_stages"] == []
    assert manifest["artifacts"] == []


def test_unknown_render_format_is_refused_up_front(
    source: tuple[Path, FakeAdapter], tmp_path: Path
) -> None:
    audio, adapter = source

    with pytest.raises(PipelineError, match="unknown render format"):
        run_pipeline(audio, adapter, tmp_path / "run-001", speaker="S1", render_format="braille")


def test_process_renders_other_formats(source: tuple[Path, FakeAdapter], tmp_path: Path) -> None:
    audio, adapter = source
    debug_dir = tmp_path / "run-001"

    result = run_pipeline(audio, adapter, debug_dir, speaker="S1", render_format="html")

    assert "<em" in result.rendered
    assert (debug_dir / "rendered.html").exists()
    assert result.manifest["settings"]["render_format"] == "html"


# Artifacts that record where the run wrote its normalized audio, and so
# legitimately differ between two runs pointed at different directories.
_PATH_BEARING = {"audio-normalization.json", "transcript.json", "assigned.json"}


def test_process_is_reproducible(source: tuple[Path, FakeAdapter], tmp_path: Path) -> None:
    """The same input must produce the same measurements and the same marks."""
    audio, adapter = source

    first = run_pipeline(audio, adapter, tmp_path / "run-a", speaker="S1")
    second = run_pipeline(audio, adapter, tmp_path / "run-b", speaker="S1")

    assert first.rendered == second.rendered

    left = {row["name"]: row["sha256"] for row in first.manifest["artifacts"]}
    right = {row["name"]: row["sha256"] for row in second.manifest["artifacts"]}
    assert left.keys() == right.keys()
    for name in left.keys() - _PATH_BEARING:
        assert left[name] == right[name], name


def test_only_recorded_paths_differ_between_runs(
    source: tuple[Path, FakeAdapter], tmp_path: Path
) -> None:
    """Provenance records the real output path, which is all that may vary.

    Pinned deliberately: if a run ever differs in a measurement rather than a
    path, that is a reproducibility bug and this test should catch it.
    """
    audio, adapter = source
    run_pipeline(audio, adapter, tmp_path / "run-a", speaker="S1")
    run_pipeline(audio, adapter, tmp_path / "run-b", speaker="S1")

    for name in _PATH_BEARING:
        left = json.loads((tmp_path / "run-a" / name).read_text(encoding="utf-8"))
        right = json.loads((tmp_path / "run-b" / name).read_text(encoding="utf-8"))
        for payload, run in ((left, "run-a"), (right, "run-b")):
            if "output_path" in payload:
                payload["output_path"] = payload["output_path"].replace(run, "<run>")
            if "audio" in payload:
                payload["audio"]["path"] = payload["audio"]["path"].replace(run, "<run>")
        assert left == right, name
