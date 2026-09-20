from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from prosody_markup.legend import load_legend
from prosody_markup.models import Document, Token
from prosody_markup.normalize import (
    BaselineSettings,
    NormalizationError,
    compute_baseline,
    normalize_and_assign,
)

# A voice centred near 150 Hz. Natural within-session variation matters: a
# perfectly constant pitch has a median absolute deviation of exactly zero,
# which is a degenerate sample rather than a scale to measure against.
_FLAT_HZ = 150.0
_VARIATION_SEMITONES = (-1.5, -0.75, 0.0, 0.75, 1.5)
_PROMINENT_HZ = 260.0


def _natural_hz(index: int) -> float:
    offset = _VARIATION_SEMITONES[index % len(_VARIATION_SEMITONES)]
    return _FLAT_HZ * 2.0 ** (offset / 12.0)


def _natural_tokens(count: int, prominent: set[int] | None = None) -> list[Token]:
    prominent = prominent or set()
    return [
        _token(i, f"word{i}", _PROMINENT_HZ if i in prominent else _natural_hz(i))
        for i in range(count)
    ]


def _token(
    index: int,
    text: str,
    f0_mean_hz: float | None,
    *,
    coverage: float = 1.0,
    suppressed: str | None = None,
) -> Token:
    raw: dict[str, float | None] = {}
    if f0_mean_hz is not None:
        raw = {"f0_mean_hz": f0_mean_hz, "voiced_coverage": coverage}
    elif suppressed:
        raw = {"voiced_coverage": coverage}
    return Token(
        id=f"w{index:04d}",
        text=text,
        start=index * 0.3,
        end=index * 0.3 + 0.25,
        speaker="S1",
        asr_confidence=0.97,
        alignment_confidence=0.97,
        turn_id="turn-1",
        speaker_confidence=1.0,
        raw_features=raw,
        feature_suppression={"pitch": suppressed} if suppressed else {},
    )


def _document(tokens: list[Token], *, speaker: str = "S1", duration_s: float = 12.0) -> Document:
    return Document(
        schema_version="0.1.0",
        legend_version="unassigned",
        audio={"duration_s": duration_s, "sample_rate": 16_000},
        provenance={"kind": "local-asr", "extractor": "praat-parselmouth"},
        baseline={"speaker": speaker, "status": "pending-normalization"},
        tokens=tokens,
    )


def _flat_document(count: int = 12) -> Document:
    return _document([_token(i, f"word{i}", _FLAT_HZ) for i in range(count)])


def test_baseline_records_method_and_parameters() -> None:
    baseline = compute_baseline(_document([_token(i, f"w{i}", 150.0 + i) for i in range(10)]))

    assert baseline.schema_version == "speaker-baseline@0.1.0"
    assert baseline.method == "median-mad-semitones"
    assert baseline.speaker == "S1"
    assert baseline.measured_words == 10
    assert baseline.window_s == 12.0
    assert baseline.settings["min_measured_words"] == 5


def test_baseline_reaches_the_ir() -> None:
    document = _document([_token(i, f"w{i}", 150.0 + i * 3) for i in range(10)])

    result = normalize_and_assign(document, load_legend())

    recorded = result.document.baseline
    assert recorded["method"] == "median-mad-semitones"
    assert recorded["speaker"] == "S1"
    assert recorded["window_s"] == 12.0
    assert "status" not in recorded
    assert result.document.provenance["baseline_method"] == "median-mad-semitones"


def test_speaker_metadata_is_required() -> None:
    document = _document([_token(i, f"w{i}", 150.0 + i) for i in range(10)], speaker="  ")

    with pytest.raises(NormalizationError, match="baseline.speaker is required"):
        compute_baseline(document)


def test_session_duration_is_required() -> None:
    document = _document([_token(i, f"w{i}", 150.0 + i) for i in range(10)])
    document.audio = {}

    with pytest.raises(NormalizationError, match="audio.duration_s is required"):
        compute_baseline(document)


def test_too_few_measured_words_refuses_a_baseline() -> None:
    """A median over three words describes the sample, not the speaker."""
    document = _document([_token(i, f"w{i}", 150.0 + i) for i in range(3)])

    with pytest.raises(NormalizationError, match="at least 5"):
        compute_baseline(document)


def test_baseline_thresholds_are_configurable() -> None:
    """The conservatism knobs are part of the contract, and are recorded."""
    document = _document([_token(i, f"w{i}", 150.0 + i) for i in range(3)])

    baseline = compute_baseline(document, BaselineSettings(min_measured_words=3))

    assert baseline.measured_words == 3
    assert baseline.settings["min_measured_words"] == 3


def test_suppressed_words_do_not_enter_the_baseline() -> None:
    tokens = [_token(i, f"w{i}", 150.0) for i in range(6)]
    tokens += [_token(9, "noise", None, suppressed="unvoiced")]

    baseline = compute_baseline(_document(tokens))

    assert baseline.measured_words == 6


def test_a_prominent_word_is_marked() -> None:
    tokens = _natural_tokens(11, prominent={4})

    result = normalize_and_assign(_document(tokens), load_legend())

    marked = [token.id for token in result.document.tokens if token.marks]
    assert marked == ["w0004"]
    assert result.document.tokens[4].marks[0].mark == "emphasis"


def test_only_the_pitch_channel_is_enabled() -> None:
    """Duration and timing stay fixture-tested until separately validated."""
    tokens = _natural_tokens(11, prominent={4})
    for token in tokens:
        # Values that would trip the duration and timing channels if enabled.
        token.features["dur_z"] = 9.0
        token.features["pause_after"] = 9.0
        token.feature_confidence["duration"] = 0.99
        token.feature_confidence["timing"] = 0.99

    result = normalize_and_assign(_document(tokens), load_legend())

    assert result.document.provenance["enabled_channels"] == ["pitch"]
    assert {candidate.channel for candidate in result.candidates} == {"pitch"}
    marks = [mark.mark for token in result.document.tokens for mark in token.marks]
    assert set(marks) <= {"emphasis"}


def test_partly_voiced_words_fall_below_the_confidence_floor() -> None:
    """Coverage is the feature's confidence, so a half-measured word is never marked."""
    tokens = _natural_tokens(11)
    tokens[4] = _token(4, "GOT", _PROMINENT_HZ, coverage=0.40)

    result = normalize_and_assign(_document(tokens), load_legend())

    assert result.document.tokens[4].feature_confidence["pitch"] == pytest.approx(0.40)
    assert result.document.tokens[4].marks == []
    assert result.candidates == []


def test_density_cap_still_limits_marks() -> None:
    """Many prominent words must not all be marked."""
    # Five clearly prominent words, but the cap allows only four.
    tokens = _natural_tokens(30, prominent={2, 7, 12, 17, 22})

    result = normalize_and_assign(_document(tokens), load_legend())

    legend = load_legend()
    marked = [token for token in result.document.tokens if token.marks]
    assert len(result.candidates) == 5
    assert len(marked) == 4
    assert len(marked) <= max(1, int(len(tokens) * legend.density_cap))


def test_candidate_and_final_mark_artifacts_are_separate() -> None:
    tokens = _natural_tokens(30, prominent={2, 7, 12, 17, 22})

    result = normalize_and_assign(_document(tokens), load_legend())
    rows = result.candidate_rows()

    assert len(rows) == len(result.candidates)
    assert any(row["retained"] for row in rows)
    assert any(not row["retained"] for row in rows)
    marked = {token.id for token in result.document.tokens if token.marks}
    assert {row["token_id"] for row in rows if row["retained"]} == marked


def test_a_flat_speaker_produces_no_marks() -> None:
    """No usable spread means nothing can be ranked prominent."""
    result = normalize_and_assign(_flat_document(), load_legend())

    assert result.degenerate_spread is True
    assert all(token.features["f0_z"] == 0.0 for token in result.document.tokens)
    assert [token.id for token in result.document.tokens if token.marks] == []


def test_unmeasured_words_get_no_normalized_feature() -> None:
    """An absent measurement must not read as a measurement of zero prominence."""
    tokens = _natural_tokens(8)
    tokens.append(_token(9, "mumble", None, suppressed="insufficient_voiced_coverage"))

    result = normalize_and_assign(_document(tokens), load_legend())

    unmeasured = result.document.tokens[-1]
    assert "f0_z" not in unmeasured.features
    assert "pitch" not in unmeasured.feature_confidence
    assert unmeasured.feature_suppression == {"pitch": "insufficient_voiced_coverage"}
    assert unmeasured.marks == []


def test_normalization_is_robust_to_a_single_octave_error() -> None:
    """A median and MAD must not be dragged by one halved measurement."""
    clean = [_token(i, f"word{i}", _FLAT_HZ + i) for i in range(11)]
    baseline_clean = compute_baseline(_document(clean))

    polluted = list(clean)
    polluted[0] = _token(0, "word0", _FLAT_HZ / 2)
    baseline_polluted = compute_baseline(_document(polluted))

    assert baseline_polluted.center_semitones == pytest.approx(
        baseline_clean.center_semitones, abs=0.5
    )


def test_normalize_cli_writes_marks_and_candidates(tmp_path: Path) -> None:
    tokens = _natural_tokens(30, prominent={2, 7, 12, 17, 22})
    features = tmp_path / "features.json"
    features.write_text(json.dumps(_document(tokens).to_dict()), encoding="utf-8")
    output = tmp_path / "assigned.json"
    candidates = tmp_path / "candidates.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "normalize",
            str(features),
            "--output",
            str(output),
            "--candidates",
            str(candidates),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "candidates ->" in completed.stdout
    assigned = json.loads(output.read_text(encoding="utf-8"))
    rows = json.loads(candidates.read_text(encoding="utf-8"))
    assert assigned["baseline"]["method"] == "median-mad-semitones"
    assert len(rows) > sum(1 for token in assigned["tokens"] if token["marks"])


def test_normalize_cli_reports_a_missing_speaker_without_traceback(tmp_path: Path) -> None:
    document = _document([_token(i, f"w{i}", 150.0 + i) for i in range(10)], speaker="  ")
    features = tmp_path / "features.json"
    features.write_text(json.dumps(document.to_dict()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "normalize",
            str(features),
            "--output",
            str(tmp_path / "assigned.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "baseline.speaker is required" in completed.stderr
    assert "Traceback" not in completed.stderr
