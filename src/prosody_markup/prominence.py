from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .assign import Candidate, assign_marks, collect_candidates
from .legend import Legend
from .models import Document

BASELINE_SCHEMA_VERSION = "speaker-baseline@0.1.0"
BASELINE_METHOD = "median-mad-semitones"

# Scales the median absolute deviation to a standard-deviation-equivalent for
# normally distributed data, so the legend's z thresholds keep their meaning.
_MAD_TO_SIGMA = 1.4826

# The channel v0.1 enables on real audio; the rest stay fixture-tested.
REAL_AUDIO_CHANNELS = {"pitch"}


class ProminenceError(ValueError):
    """Raised when a trustworthy speaker baseline cannot be established."""


@dataclass(frozen=True, slots=True)
class BaselineSettings:
    """Bounds on when a session baseline is considered trustworthy.

    Every default here prefers producing no marks over producing wrong ones,
    which is the tradeoff the v0.1 precision gate asks for.
    """

    # Below this, a median and MAD describe the sample rather than the speaker.
    min_measured_words: int = 5
    # A spread this small is a monotone sample, not a scale; dividing by it
    # would turn measurement noise into towering z-scores.
    min_spread_semitones: float = 0.25

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Baseline:
    schema_version: str
    method: str
    speaker: str
    center_semitones: float
    spread_semitones: float
    measured_words: int
    window_s: float
    settings: dict[str, float | int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _semitones(frequency_hz: float) -> float:
    """Pitch on a log scale, where a speaker's variation is symmetric.

    Prominence is perceived in ratios, not hertz, so a 20 Hz rise means
    something different for a low voice than a high one. Normalizing in
    semitones keeps one z threshold meaningful across speakers.
    """
    return 12.0 * math.log2(frequency_hz)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _measured_pitch(document: Document) -> list[tuple[int, float]]:
    """Indexes and F0 means of words the extractor actually measured."""
    measured: list[tuple[int, float]] = []
    for index, token in enumerate(document.tokens):
        if "pitch" in token.feature_suppression:
            continue
        value = token.raw_features.get("f0_mean_hz")
        if value is None or not math.isfinite(value) or value <= 0.0:
            continue
        measured.append((index, float(value)))
    return measured


def compute_baseline(document: Document, settings: BaselineSettings | None = None) -> Baseline:
    """Derive a session-level baseline for one known speaker.

    Uses a median and a scaled median absolute deviation rather than a mean and
    standard deviation: a single octave error or shout would drag those enough
    to change which words look prominent.
    """
    settings = settings or BaselineSettings()

    speaker = str(document.baseline.get("speaker") or "").strip()
    if not speaker:
        raise ProminenceError("baseline.speaker is required to normalize a session")
    window_s = document.audio.get("duration_s")
    if not isinstance(window_s, (int, float)) or not math.isfinite(window_s) or window_s <= 0:
        raise ProminenceError("audio.duration_s is required to record the baseline window")

    measured = _measured_pitch(document)
    if len(measured) < settings.min_measured_words:
        raise ProminenceError(
            f"only {len(measured)} measured words; at least {settings.min_measured_words} are "
            "needed for a session baseline"
        )

    semitones = [_semitones(value) for _index, value in measured]
    center = _median(semitones)
    spread = _median([abs(value - center) for value in semitones]) * _MAD_TO_SIGMA

    return Baseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        method=BASELINE_METHOD,
        speaker=speaker,
        center_semitones=center,
        spread_semitones=spread,
        measured_words=len(measured),
        window_s=float(window_s),
        settings=settings.to_dict(),
    )


@dataclass(frozen=True, slots=True)
class ProminenceResult:
    document: Document
    baseline: Baseline
    candidates: list[Candidate]
    degenerate_spread: bool

    def candidate_rows(self) -> list[dict[str, Any]]:
        """Candidates as an artifact, before the density cap decides anything."""
        marked = {token.id for token in self.document.tokens if token.marks}
        rows: list[dict[str, Any]] = []
        for candidate in self.candidates:
            token = self.document.tokens[candidate.token_index]
            rows.append(
                {
                    "token_id": token.id,
                    "text": token.text,
                    "start": token.start,
                    "end": token.end,
                    "channel": candidate.channel,
                    "mark": candidate.mark,
                    "confidence": candidate.confidence,
                    "strength": candidate.strength,
                    "prominence": candidate.prominence,
                    "f0_z": token.features.get("f0_z"),
                    "retained": token.id in marked,
                    "density_floor_retained": token.density_floor_retained,
                }
            )
        return rows


def normalize_and_assign(
    document: Document,
    legend: Legend,
    settings: BaselineSettings | None = None,
    channels: set[str] | None = None,
) -> ProminenceResult:
    """Turn raw per-word pitch into normalized candidates and conservative marks.

    Confidence in the normalized feature is the word's voiced coverage: the
    fraction of the word actually measured. A partly voiced word therefore
    falls below the legend's confidence floor and is never marked, which keeps
    the failure mode a missed mark rather than a wrong one.
    """
    baseline = compute_baseline(document, settings)
    settings = settings or BaselineSettings()
    degenerate = baseline.spread_semitones < settings.min_spread_semitones

    for token in document.tokens:
        token.features = {key: value for key, value in token.features.items() if key != "f0_z"}
        token.feature_confidence = {
            key: value for key, value in token.feature_confidence.items() if key != "pitch"
        }

        if "pitch" in token.feature_suppression:
            continue
        mean_hz = token.raw_features.get("f0_mean_hz")
        coverage = token.raw_features.get("voiced_coverage")
        if mean_hz is None or coverage is None or mean_hz <= 0.0:
            continue

        if degenerate:
            # A sample with no usable spread cannot rank prominence. Report a
            # flat score rather than dividing by noise and inventing outliers.
            token.features["f0_z"] = 0.0
        else:
            token.features["f0_z"] = (
                _semitones(float(mean_hz)) - baseline.center_semitones
            ) / baseline.spread_semitones
        token.feature_confidence["pitch"] = min(1.0, max(0.0, float(coverage)))

    document.baseline = {
        **document.baseline,
        **baseline.to_dict(),
        # Assignment validates this, and it is what makes the document
        # assignable at all: an un-normalized transcript has no window.
        "window_s": baseline.window_s,
    }
    document.baseline.pop("status", None)

    restricted = legend.restrict_channels(channels or REAL_AUDIO_CHANNELS)
    candidates = collect_candidates(document, restricted)
    assign_marks(document, restricted)

    document.provenance = {
        **document.provenance,
        "baseline_method": baseline.method,
        "baseline_schema": baseline.schema_version,
        "enabled_channels": sorted(restricted.channels),
        "degenerate_spread": degenerate,
    }
    return ProminenceResult(
        document=document,
        baseline=baseline,
        candidates=candidates,
        degenerate_spread=degenerate,
    )
