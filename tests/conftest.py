from __future__ import annotations

import json
from pathlib import Path

import pytest

from prosody_markup.models import Document


@pytest.fixture
def fixture_document() -> Document:
    path = Path(__file__).parents[1] / "examples" / "tier1-input.json"
    return Document.from_dict(json.loads(path.read_text(encoding="utf-8")))
