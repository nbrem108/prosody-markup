from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import parselmouth

from .assign import is_eligible, validate_tokens
from .audio import inspect_wav
from .models import Document

EXTRACTOR_SCHEMA_VERSION = "pitch-extraction@0.1.0"
EXTRACTOR_NAME = "praat-parselmouth"


class ExtractionError(ValueError):
    """Raised when pitch features cannot be measured from canonical audio."""


@dataclass(frozen=True, slots=True)
class PitchSettings:
    """Conservative measurement bounds, recorded with every extraction.

    Defaults deliberately prefer a missing measurement to a wrong one: a word
    that cannot be measured cleanly is suppressed rather than guessed at.
    """

    floor_hz: float = 75.0
    ceiling_hz: float = 500.0
    time_step_s: float = 0.01
    # Praat needs several pitch periods; below this a window cannot be measured.
    min_window_s: float = 0.04
    min_voiced_coverage: float = 0.5
    # Praat resolves a periodic signal to a subharmonic when the ceiling cannot
    # fit the true F0, and reports it confidently and consistently, so neither a
    # range check nor a within-word spread check detects it. Re-measure with a
    # raised ceiling and refuse the word when the two analyses disagree by more
    # than this many octaves.
    octave_verification_ratio: float = 2.0
    max_octave_disagreement: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PitchFrame:
    time_s: float
    f0_hz: float
    token_id: str


@dataclass(frozen=True, slots=True)
class WordPitch:
    """One word's measurement, or an explicit reason there is none."""

    token_id: str
    text: str
    start: float
    end: float
    voiced_coverage: float
    f0_mean: float | None
    f0_min: float | None
    f0_max: float | None
    f0_range: float | None
    valid: bool
    reason: str | None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PitchExtraction:
    document: Document
    words: list[WordPitch]
    frames: list[PitchFrame]
    settings: PitchSettings


def _measure_contour(
    path: Path, settings: PitchSettings, ceiling_hz: float | None = None
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    """Measure one pitch contour across the whole recording.

    Measuring once and slicing per word avoids re-analysing short excerpts,
    which Praat refuses outright and which would lose the surrounding context
    its periodicity search depends on.
    """
    try:
        sound = parselmouth.Sound(str(path))
        pitch = sound.to_pitch(
            time_step=settings.time_step_s,
            pitch_floor=settings.floor_hz,
            pitch_ceiling=settings.ceiling_hz if ceiling_hz is None else ceiling_hz,
        )
    except parselmouth.PraatError as exc:
        raise ExtractionError(f"pitch analysis failed: {exc}") from exc

    times = np.asarray(pitch.xs(), dtype=np.float64)
    frequencies = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    return times, frequencies


def _summarize_word(
    token_id: str,
    text: str,
    start: float,
    end: float,
    times: np.typing.NDArray[np.float64],
    frequencies: np.typing.NDArray[np.float64],
    verification: tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
    settings: PitchSettings,
) -> tuple[WordPitch, np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    def unmeasured(reason: str, coverage: float = 0.0) -> WordPitch:
        return WordPitch(
            token_id=token_id,
            text=text,
            start=start,
            end=end,
            voiced_coverage=coverage,
            f0_mean=None,
            f0_min=None,
            f0_max=None,
            f0_range=None,
            valid=False,
            reason=reason,
        )

    empty: np.typing.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    if end - start < settings.min_window_s:
        return unmeasured("window_too_short"), empty, empty

    window = (times >= start) & (times < end)
    values = frequencies[window]
    window_times = times[window]
    if values.size == 0:
        return unmeasured("no_pitch_frames"), empty, empty

    voiced = values[values > 0.0]
    coverage = float(voiced.size / values.size)
    if voiced.size == 0:
        return unmeasured("unvoiced", coverage), empty, empty
    if coverage < settings.min_voiced_coverage:
        return unmeasured("insufficient_voiced_coverage", coverage), empty, empty

    f0_min = float(voiced.min())
    f0_max = float(voiced.max())
    if f0_min < settings.floor_hz or f0_max > settings.ceiling_hz:
        return unmeasured("f0_out_of_range", coverage), empty, empty

    mean_hz = float(voiced.mean())
    check_times, check_frequencies = verification
    check_window = (check_times >= start) & (check_times < end)
    check_voiced = check_frequencies[check_window][check_frequencies[check_window] > 0.0]
    if check_voiced.size == 0:
        # The wider search should find voicing wherever the narrow one did.
        return unmeasured("octave_ambiguous", coverage), empty, empty
    disagreement = abs(math.log2(float(check_voiced.mean()) / mean_hz))
    if disagreement > settings.max_octave_disagreement:
        return unmeasured("octave_ambiguous", coverage), empty, empty

    measured = WordPitch(
        token_id=token_id,
        text=text,
        start=start,
        end=end,
        voiced_coverage=coverage,
        f0_mean=mean_hz,
        f0_min=f0_min,
        f0_max=f0_max,
        f0_range=f0_max - f0_min,
        valid=True,
        reason=None,
    )
    voiced_mask = values > 0.0
    return measured, window_times[voiced_mask], values[voiced_mask]


def extract_pitch(
    document: Document,
    audio_path: Path,
    settings: PitchSettings | None = None,
) -> PitchExtraction:
    """Attach raw per-word pitch measurements to word-timestamp IR.

    Writes raw measurements only. Normalization into comparable features and
    any mark assignment are later stages, so nothing here decides typography.
    """
    settings = settings or PitchSettings()
    inspection = inspect_wav(audio_path)
    if inspection.conversion_required:
        raise ExtractionError(
            "audio must be canonical mono 16 kHz 16-bit PCM; "
            "run 'prosody-markup audio normalize' first"
        )
    if not document.tokens:
        raise ExtractionError("transcript contains no tokens")
    validate_tokens(document.tokens)

    latest = max(token.end for token in document.tokens)
    if latest > inspection.duration_s + settings.time_step_s:
        raise ExtractionError(
            f"transcript ends at {latest:.3f}s, past the {inspection.duration_s:.3f}s recording"
        )

    source = Path(inspection.source_path)
    times, frequencies = _measure_contour(source, settings)
    # Second opinion at a raised ceiling, used only to detect subharmonics.
    verification = _measure_contour(
        source, settings, ceiling_hz=settings.ceiling_hz * settings.octave_verification_ratio
    )

    words: list[WordPitch] = []
    frames: list[PitchFrame] = []
    for token in document.tokens:
        token.raw_features = {}
        token.feature_suppression = {}
        if not is_eligible(token):
            continue

        measurement, voiced_times, voiced_values = _summarize_word(
            token.id,
            token.text,
            token.start,
            token.end,
            times,
            frequencies,
            verification,
            settings,
        )
        words.append(measurement)

        if measurement.valid:
            token.raw_features = {
                "f0_mean_hz": measurement.f0_mean,
                "f0_min_hz": measurement.f0_min,
                "f0_max_hz": measurement.f0_max,
                "f0_range_hz": measurement.f0_range,
                "voiced_coverage": measurement.voiced_coverage,
            }
            for time_s, value in zip(voiced_times, voiced_values, strict=True):
                frames.append(PitchFrame(float(time_s), float(value), token.id))
        else:
            token.raw_features = {"voiced_coverage": measurement.voiced_coverage}
            # An unmeasurable word must say so; a later stage must not read the
            # absence of features as a measurement of zero prominence.
            token.feature_suppression = {"pitch": measurement.reason or "unmeasured"}

    document.provenance = {
        **document.provenance,
        "extractor": EXTRACTOR_NAME,
        "extractor_version": str(parselmouth.VERSION),
        "extraction_schema": EXTRACTOR_SCHEMA_VERSION,
        "extraction_settings": settings.to_dict(),
    }
    return PitchExtraction(document=document, words=words, frames=frames, settings=settings)


def write_debug_artifacts(extraction: PitchExtraction, debug_dir: Path) -> list[Path]:
    """Emit the inspectable tables the v0.1 plan requires for every run."""
    debug_dir.mkdir(parents=True, exist_ok=True)

    features_path = debug_dir / "word-features.csv"
    with features_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "token_id",
                "text",
                "start",
                "end",
                "voiced_coverage",
                "f0_mean",
                "f0_min",
                "f0_max",
                "f0_range",
                "valid",
                "reason",
            ],
        )
        writer.writeheader()
        for word in extraction.words:
            writer.writerow(word.to_row())

    contour_path = debug_dir / "pitch-contour.csv"
    with contour_path.open("w", encoding="utf-8", newline="") as stream:
        contour_writer = csv.writer(stream)
        contour_writer.writerow(["time_s", "f0_hz", "token_id"])
        for frame in extraction.frames:
            contour_writer.writerow([f"{frame.time_s:.4f}", f"{frame.f0_hz:.3f}", frame.token_id])

    return [features_path, contour_path]
