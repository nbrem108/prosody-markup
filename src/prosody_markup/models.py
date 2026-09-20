from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Mark:
    channel: str
    mark: str
    confidence: float
    strength: int = 1


@dataclass(slots=True)
class Token:
    id: str
    text: str
    start: float
    end: float
    speaker: str | None
    asr_confidence: float
    alignment_confidence: float
    turn_id: str | None = None
    speaker_confidence: float | None = None
    space_before: bool | None = None
    features: dict[str, float | None] = field(default_factory=dict)
    feature_confidence: dict[str, float] = field(default_factory=dict)
    overlap: bool = False
    marks: list[Mark] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    density_floor_retained: bool = False


@dataclass(slots=True)
class Document:
    schema_version: str
    legend_version: str
    audio: dict[str, Any]
    provenance: dict[str, Any]
    baseline: dict[str, Any]
    tokens: list[Token]
    events: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Document:
        tokens = []
        for raw in value.get("tokens", []):
            marks = [Mark(**mark) for mark in raw.get("marks", [])]
            tokens.append(Token(**{**raw, "marks": marks}))
        return cls(
            schema_version=str(value["schema_version"]),
            legend_version=str(value["legend_version"]),
            audio=dict(value.get("audio", {})),
            provenance=dict(value.get("provenance", {})),
            baseline=dict(value.get("baseline", {})),
            tokens=tokens,
            events=list(value.get("events", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
