from __future__ import annotations

from prosody_markup.models import Document


def test_document_round_trip(fixture_document):  # type: ignore[no-untyped-def]
    encoded = fixture_document.to_dict()
    decoded = Document.from_dict(encoded)
    assert decoded.to_dict() == encoded
