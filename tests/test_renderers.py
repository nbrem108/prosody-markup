from __future__ import annotations

from prosody_markup.assign import assign_marks
from prosody_markup.legend import load_legend
from prosody_markup.render import render_html, render_markdown, render_unicode


def test_markdown_fixture(fixture_document):  # type: ignore[no-untyped-def]
    assigned = assign_marks(fixture_document, load_legend())
    assert render_markdown(assigned) == (
        "No, I *got* it. It is fine. I will just redo the whole deck before the morning review "
        "with the team tonight....."
    )


def test_html_preserves_versions_and_audio_bounds(fixture_document):  # type: ignore[no-untyped-def]
    assigned = assign_marks(fixture_document, load_legend())
    rendered = render_html(assigned)
    assert 'data-schema-version="0.1.0"' in rendered
    assert 'data-legend-version="tier1@0.1.0-draft"' in rendered
    assert 'data-start="0.500" data-end="0.800"' in rendered
    assert "<em " in rendered


def test_unicode_fallback_is_distinct(fixture_document):  # type: ignore[no-untyped-def]
    assigned = assign_marks(fixture_document, load_legend())
    rendered = render_unicode(assigned)
    assert "_got_" in rendered
    assert "*got*" not in rendered
