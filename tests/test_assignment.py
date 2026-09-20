from __future__ import annotations

import math

import pytest

from prosody_markup.assign import assign_marks, is_eligible
from prosody_markup.legend import load_legend


def test_assigns_high_confidence_marks_and_respects_density(fixture_document):  # type: ignore[no-untyped-def]
    legend = load_legend()
    assigned = assign_marks(fixture_document, legend)
    marked = [token for token in assigned.tokens if token.marks]
    eligible = [token for token in assigned.tokens if is_eligible(token)]

    assert [token.id for token in marked] == ["t04", "t25"]
    assert len(marked) / len(eligible) <= legend.density_cap
    assert assigned.legend_version == legend.legend_version


def test_suppresses_untrusted_tokens(fixture_document):  # type: ignore[no-untyped-def]
    token = fixture_document.tokens[3]
    token.asr_confidence = 0.3
    token.features["f0_z"] = 5.0
    token.feature_confidence["pitch"] = 1.0

    assigned = assign_marks(fixture_document, load_legend())

    assert assigned.tokens[3].marks == []
    assert "low_asr_confidence" in assigned.tokens[3].suppressed


def test_density_cap_breaks_prominence_ties_on_confidence(fixture_document):  # type: ignore[no-untyped-def]
    for token in fixture_document.tokens:
        if is_eligible(token):
            token.features["f0_z"] = 3.0
            token.feature_confidence["pitch"] = 0.90
    fixture_document.tokens[3].feature_confidence["pitch"] = 0.99
    fixture_document.tokens[24].feature_confidence["pitch"] = 0.98

    assigned = assign_marks(fixture_document, load_legend())
    marked_ids = [token.id for token in assigned.tokens if token.marks]

    assert len(marked_ids) == 3
    assert {"t04", "t25"}.issubset(marked_ids)
    assert marked_ids == ["t01", "t04", "t25"]
    assert any("density_cap" in token.suppressed for token in assigned.tokens)


def test_density_cap_is_applied_per_speaker_turn(fixture_document):  # type: ignore[no-untyped-def]
    eligible = [token for token in fixture_document.tokens if is_eligible(token)]
    split = len(eligible) // 2
    for index, token in enumerate(eligible):
        token.speaker = "A" if index < split else "B"
        token.turn_id = "turn-a" if index < split else "turn-b"
        token.features["f0_z"] = 3.0
        token.feature_confidence["pitch"] = 0.95

    assigned = assign_marks(fixture_document, load_legend())

    for speaker in ("A", "B"):
        speaker_tokens = [
            token for token in assigned.tokens if token.speaker == speaker and is_eligible(token)
        ]
        marked = [token for token in speaker_tokens if token.marks]
        assert len(marked) <= math.floor(len(speaker_tokens) * 0.15)


def test_suppresses_lengthening_when_token_has_no_vowel(fixture_document):  # type: ignore[no-untyped-def]
    token = fixture_document.tokens[0]
    token.text = "hmm"
    token.features["dur_z"] = 4.0
    token.feature_confidence["duration"] = 0.99

    assigned = assign_marks(fixture_document, load_legend())

    assert not any(mark.mark == "lengthening" for mark in assigned.tokens[0].marks)
    assert "unrenderable_lengthening" in assigned.tokens[0].suppressed


def test_suppresses_uncertain_speaker_attribution(fixture_document):  # type: ignore[no-untyped-def]
    token = fixture_document.tokens[3]
    token.speaker_confidence = 0.2

    assigned = assign_marks(fixture_document, load_legend())

    assert assigned.tokens[3].marks == []
    assert "low_speaker_confidence" in assigned.tokens[3].suppressed


def test_rejects_nonfinite_features(fixture_document):  # type: ignore[no-untyped-def]
    fixture_document.tokens[3].features["f0_z"] = float("inf")

    with pytest.raises(ValueError, match="must be finite"):
        assign_marks(fixture_document, load_legend())


def test_rejects_duplicate_token_ids(fixture_document):  # type: ignore[no-untyped-def]
    fixture_document.tokens[1].id = fixture_document.tokens[0].id

    with pytest.raises(ValueError, match="nonempty and unique"):
        assign_marks(fixture_document, load_legend())


def test_density_cap_ranks_prominence_over_confidence(fixture_document):  # type: ignore[no-untyped-def]
    """A strongly stressed token in noisier audio must beat a marginal one in clean audio."""
    fixture_document.tokens = fixture_document.tokens[:10]
    for token in fixture_document.tokens:
        token.features.pop("f0_z", None)
        token.features["pause_after"] = 0.0
        token.feature_confidence.clear()

    legend = load_legend()
    eligible = [token for token in fixture_document.tokens if is_eligible(token)]
    assert math.floor(len(eligible) * legend.density_cap) == 1

    marginal = fixture_document.tokens[2]  # t03: barely over threshold, pristine confidence
    marginal.features["f0_z"] = 2.05
    marginal.feature_confidence["pitch"] = 0.99

    strong = fixture_document.tokens[8]  # t09: strongly stressed, confidence just above the floor
    strong.features["f0_z"] = 4.5
    strong.feature_confidence["pitch"] = 0.86

    assigned = assign_marks(fixture_document, legend)
    marked_ids = [token.id for token in assigned.tokens if token.marks]

    assert marked_ids == ["t09"]
    assert "density_cap" in assigned.tokens[2].suppressed
