from __future__ import annotations

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


def test_density_cap_keeps_highest_confidence_tokens(fixture_document):  # type: ignore[no-untyped-def]
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
