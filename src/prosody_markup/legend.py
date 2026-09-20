from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import IO, Any

import yaml


@dataclass(frozen=True, slots=True)
class ChannelRule:
    feature: str
    mark: str
    threshold: float
    confidence_floor: float


@dataclass(frozen=True, slots=True)
class SuppressionPolicy:
    asr_confidence_floor: float
    alignment_confidence_floor: float
    speaker_confidence_floor: float
    require_speaker: bool
    suppress_overlap: bool


@dataclass(frozen=True, slots=True)
class Legend:
    schema_version: str
    legend_version: str
    density_cap: float
    min_marks_per_turn: int
    channels: dict[str, ChannelRule]
    suppression: SuppressionPolicy

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Legend:
        return cls(
            schema_version=str(raw["schema_version"]),
            legend_version=str(raw["legend_version"]),
            density_cap=float(raw["density_cap"]),
            min_marks_per_turn=int(raw.get("min_marks_per_turn", 1)),
            channels={name: ChannelRule(**rule) for name, rule in raw["channels"].items()},
            suppression=SuppressionPolicy(**raw["suppression"]),
        )


def _default_legend_stream() -> IO[str]:
    source_path = Path(__file__).resolve().parents[2] / "legend" / "tier1.yaml"
    if source_path.exists():
        return source_path.open(encoding="utf-8")
    packaged = resources.files("prosody_markup").joinpath("data/tier1.yaml")
    return packaged.open("r", encoding="utf-8")


def load_legend(path: Path | None = None) -> Legend:
    stream = path.open(encoding="utf-8") if path else _default_legend_stream()
    with stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        location = str(path) if path else "packaged Tier 1 legend"
        raise ValueError(f"Legend must be a mapping: {location}")
    return Legend.from_dict(raw)
