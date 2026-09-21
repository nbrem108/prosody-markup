from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .assign import is_eligible
from .models import Document

EVALUATION_SCHEMA_VERSION = "evaluation-report@0.1.0"
ANNOTATION_TASKS_SCHEMA_VERSION = "annotation-tasks@0.1.0"

# The v0.1 gate, and the sample it must be measured on. A precision figure from
# a handful of utterances is not a gate result, so coverage is checked first.
PRECISION_GATE = 0.85
MIN_UTTERANCES = 100
MIN_SPEAKERS = 10
MIN_ANNOTATORS = 3


class EvaluationError(ValueError):
    """Raised when a report cannot be produced from the given runs and labels."""


@dataclass(frozen=True, slots=True)
class TaskWord:
    token_id: str
    text: str


@dataclass(frozen=True, slots=True)
class AnnotationTask:
    """One utterance to label, carrying no trace of what the system decided."""

    utterance_id: str
    speaker: str
    words: list[TaskWord]


@dataclass(frozen=True, slots=True)
class Annotation:
    annotator: str
    labels: dict[str, list[str]]
    tasks_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class Metrics:
    marked: int
    majority_positive: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_tasks(documents: dict[str, Document]) -> list[AnnotationTask]:
    """Turn assigned runs into blind labelling tasks.

    Only token ids and lexical text cross over. Marks, features, confidences and
    suppression reasons are all left behind, so an annotator cannot be anchored
    by what the system already decided.
    """
    tasks: list[AnnotationTask] = []
    for utterance_id, document in sorted(documents.items()):
        speaker = str(document.baseline.get("speaker") or "").strip()
        if not speaker:
            raise EvaluationError(f"{utterance_id}: run has no speaker, so it cannot be split")
        words = [
            TaskWord(token_id=token.id, text=token.text)
            for token in document.tokens
            if is_eligible(token)
        ]
        if not words:
            raise EvaluationError(f"{utterance_id}: run has no eligible words to label")
        tasks.append(AnnotationTask(utterance_id=utterance_id, speaker=speaker, words=words))
    return tasks


def tasks_payload(tasks: list[AnnotationTask]) -> dict[str, Any]:
    return {
        "schema_version": ANNOTATION_TASKS_SCHEMA_VERSION,
        "instructions": (
            "Mark the words you heard as prominent. Label from the audio alone, "
            "before seeing any system output."
        ),
        "tasks": [
            {
                "utterance_id": task.utterance_id,
                "speaker": task.speaker,
                "words": [asdict(word) for word in task.words],
            }
            for task in tasks
        ],
    }


def load_annotations(directory: Path) -> list[Annotation]:
    annotations: list[Annotation] = []
    for path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path.name} is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise EvaluationError(f"{path.name} must be a JSON object")
        annotator = str(raw.get("annotator") or path.stem).strip()
        labels = raw.get("labels")
        if not isinstance(labels, dict):
            raise EvaluationError(f"{path.name} must carry a 'labels' object")
        annotations.append(
            Annotation(
                annotator=annotator,
                labels={str(key): [str(item) for item in value] for key, value in labels.items()},
                tasks_sha256=raw.get("tasks_sha256"),
            )
        )
    if not annotations:
        raise EvaluationError(f"no annotation files found in {directory}")
    return annotations


def fleiss_kappa(ratings: list[tuple[int, int]]) -> float | None:
    """Agreement across a fixed number of raters on a binary judgement.

    Returns None where the statistic is undefined: with one rater, or when every
    rater chose the same category for every item, chance agreement is already
    total and the correction divides by zero.
    """
    if not ratings:
        return None
    raters = ratings[0][0] + ratings[0][1]
    if raters < 2 or any(positive + negative != raters for positive, negative in ratings):
        return None

    items = len(ratings)
    agreement = [
        (positive * positive + negative * negative - raters) / (raters * (raters - 1))
        for positive, negative in ratings
    ]
    observed = sum(agreement) / items
    positives = sum(positive for positive, _negative in ratings) / (items * raters)
    expected = positives * positives + (1 - positives) * (1 - positives)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


@dataclass(slots=True)
class WordOutcome:
    utterance_id: str
    speaker: str
    token_id: str
    text: str
    votes: int
    raters: int
    majority: bool
    marked: bool
    floor_retained: bool

    @property
    def category(self) -> str:
        """Why a false positive happened, in terms the data supports."""
        if self.floor_retained:
            return "density_floor_retained"
        if self.votes == 0:
            return "no_annotator_agreed"
        return "split_decision"


def _metrics(outcomes: list[WordOutcome]) -> Metrics:
    true_positives = sum(1 for item in outcomes if item.marked and item.majority)
    false_positives = sum(1 for item in outcomes if item.marked and not item.majority)
    false_negatives = sum(1 for item in outcomes if not item.marked and item.majority)
    marked = true_positives + false_positives
    return Metrics(
        marked=marked,
        majority_positive=true_positives + false_negatives,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=true_positives / marked if marked else None,
        recall=(
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives)
            else None
        ),
    )


@dataclass(slots=True)
class EvaluationReport:
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def gate_status(self) -> str:
        status = self.payload["gate"]["status"]
        return str(status)


def _upstream_changes(outcomes: list[WordOutcome]) -> list[str]:
    """Name what a failing gate points at, rather than leaving a bare number."""
    false_positives = [item for item in outcomes if item.marked and not item.majority]
    if not false_positives:
        return []
    counts: dict[str, int] = {}
    for item in false_positives:
        counts[item.category] = counts.get(item.category, 0) + 1

    changes: list[str] = []
    total = len(false_positives)
    if counts.get("density_floor_retained", 0) / total > 0.3:
        changes.append(
            "Most false positives survive only under the short-turn floor: reconsider "
            "min_marks_per_turn before touching thresholds."
        )
    if counts.get("no_annotator_agreed", 0) / total > 0.3:
        changes.append(
            "Most false positives were heard as prominent by no annotator: raise the pitch "
            "channel threshold or the confidence floor."
        )
    if counts.get("split_decision", 0) / total > 0.3:
        changes.append(
            "Most false positives split the annotators: the disagreement may be in the "
            "labelling task rather than in extraction."
        )
    return changes


def evaluate(
    documents: dict[str, Document],
    annotations: list[Annotation],
    *,
    tuning_speakers: set[str] | None = None,
    tasks_sha256: str | None = None,
) -> EvaluationReport:
    """Score assigned runs against majority annotator labels.

    Precision is reported first and gates the phase; recall is diagnostic. When
    the sample falls short of the required utterances, speakers, or annotators,
    the gate reports insufficient data rather than a pass, because a precision
    figure from too small a sample is not the result the gate asks for.
    """
    if not documents:
        raise EvaluationError("no runs to evaluate")

    raters = len(annotations)
    outcomes: list[WordOutcome] = []
    ratings: list[tuple[int, int]] = []
    mismatched_tasks = [
        annotation.annotator
        for annotation in annotations
        if tasks_sha256 and annotation.tasks_sha256 and annotation.tasks_sha256 != tasks_sha256
    ]

    eligible_total = 0
    feature_suppressed = 0
    legend_suppressed = 0

    for utterance_id, document in sorted(documents.items()):
        speaker = str(document.baseline.get("speaker") or "").strip()
        labelled = [
            set(annotation.labels[utterance_id])
            for annotation in annotations
            if utterance_id in annotation.labels
        ]
        for token in document.tokens:
            if not is_eligible(token):
                continue
            eligible_total += 1
            if token.feature_suppression:
                feature_suppressed += 1
            if token.suppressed:
                legend_suppressed += 1
            if len(labelled) != raters:
                # An utterance not every annotator labelled cannot carry a
                # majority, so it is left out rather than silently weighted.
                continue
            votes = sum(1 for labels in labelled if token.id in labels)
            ratings.append((votes, raters - votes))
            outcomes.append(
                WordOutcome(
                    utterance_id=utterance_id,
                    speaker=speaker,
                    token_id=token.id,
                    text=token.text,
                    votes=votes,
                    raters=raters,
                    majority=votes * 2 > raters,
                    marked=bool(token.marks),
                    floor_retained=token.density_floor_retained,
                )
            )

    speakers = sorted({outcome.speaker for outcome in outcomes})
    shortfalls: list[str] = []
    if len(documents) < MIN_UTTERANCES:
        shortfalls.append(f"{len(documents)} utterances, need {MIN_UTTERANCES}")
    if len(speakers) < MIN_SPEAKERS:
        shortfalls.append(f"{len(speakers)} speakers, need {MIN_SPEAKERS}")
    if raters < MIN_ANNOTATORS:
        shortfalls.append(f"{raters} annotators, need {MIN_ANNOTATORS}")

    overall = _metrics(outcomes)
    held_out_speakers = sorted(set(speakers) - (tuning_speakers or set()))
    held_out = (
        _metrics([item for item in outcomes if item.speaker in set(held_out_speakers)])
        if tuning_speakers
        else None
    )

    gate_metrics = held_out or overall
    if shortfalls or mismatched_tasks or gate_metrics.precision is None:
        status = "insufficient-data"
    elif gate_metrics.precision >= PRECISION_GATE:
        status = "pass"
    else:
        status = "fail"

    false_positive_categories: dict[str, int] = {}
    for item in outcomes:
        if item.marked and not item.majority:
            false_positive_categories[item.category] = (
                false_positive_categories.get(item.category, 0) + 1
            )

    payload: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "coverage": {
            "utterances": len(documents),
            "speakers": len(speakers),
            "annotators": raters,
            "scored_words": len(outcomes),
            "minimums": {
                "utterances": MIN_UTTERANCES,
                "speakers": MIN_SPEAKERS,
                "annotators": MIN_ANNOTATORS,
            },
            "meets_minimums": not shortfalls,
            "shortfalls": shortfalls,
            "annotators_with_mismatched_tasks": mismatched_tasks,
        },
        "agreement": {
            "metric": "fleiss_kappa",
            "value": fleiss_kappa(ratings),
            "raters": raters,
            "items": len(ratings),
        },
        "overall": overall.to_dict(),
        "held_out": (
            {
                "speakers": held_out_speakers,
                "tuning_speakers": sorted(tuning_speakers or set()),
                **held_out.to_dict(),
            }
            if held_out
            else None
        ),
        "per_speaker": {
            speaker: _metrics([item for item in outcomes if item.speaker == speaker]).to_dict()
            for speaker in speakers
        },
        "density": {
            "eligible_words": eligible_total,
            "marked_words": sum(1 for item in outcomes if item.marked),
            "mark_density": (
                sum(1 for item in outcomes if item.marked) / eligible_total
                if eligible_total
                else None
            ),
            "floor_retained_marks": sum(
                1 for item in outcomes if item.marked and item.floor_retained
            ),
        },
        "suppression": {
            "feature_suppressed_words": feature_suppressed,
            "legend_suppressed_words": legend_suppressed,
            "feature_suppression_rate": (
                feature_suppressed / eligible_total if eligible_total else None
            ),
            "legend_suppression_rate": (
                legend_suppressed / eligible_total if eligible_total else None
            ),
        },
        "false_positives": {
            "total": overall.false_positives,
            "by_category": false_positive_categories,
        },
        "gate": {
            "mark": "emphasis",
            "threshold": PRECISION_GATE,
            "measured_on": "held_out" if held_out else "overall",
            "measured_precision": gate_metrics.precision,
            "status": status,
            "required_upstream_changes": (
                _upstream_changes(outcomes) if status in {"fail", "insufficient-data"} else []
            ),
        },
    }
    return EvaluationReport(payload=payload)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_report(report: EvaluationReport) -> str:
    """A Markdown summary whose headline is the gate decision, not the number."""
    payload = report.payload
    gate = payload["gate"]
    coverage = payload["coverage"]
    verdict = {
        "pass": "PASS - proceed to paired exposure",
        "fail": "NO-GO - do not proceed; fix upstream",
        "insufficient-data": "INSUFFICIENT DATA - not a gate result",
    }[gate["status"]]

    lines = [
        "# Tier 1 prominence precision report",
        "",
        f"**{verdict}**",
        "",
        f"Precision {_percent(gate['measured_precision'])} against a {gate['threshold']:.0%} gate, "
        f"measured on the {gate['measured_on']} split.",
        "",
        "## Coverage",
        "",
        f"- Utterances: {coverage['utterances']} (need {coverage['minimums']['utterances']})",
        f"- Speakers: {coverage['speakers']} (need {coverage['minimums']['speakers']})",
        f"- Annotators: {coverage['annotators']} (need {coverage['minimums']['annotators']})",
        f"- Scored words: {coverage['scored_words']}",
    ]
    for shortfall in coverage["shortfalls"]:
        lines.append(f"- **Shortfall:** {shortfall}")
    for annotator in coverage["annotators_with_mismatched_tasks"]:
        lines.append(f"- **Warning:** {annotator} labelled a different task set")

    agreement = payload["agreement"]
    overall = payload["overall"]
    density = payload["density"]
    suppression = payload["suppression"]
    kappa = agreement["value"]
    lines += [
        "",
        "## Agreement",
        "",
        f"- Fleiss' kappa: {'n/a' if kappa is None else f'{kappa:.3f}'} "
        f"across {agreement['raters']} raters and {agreement['items']} items",
        "",
        "## Scores",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {_percent(overall['precision'])} |",
        f"| Recall (diagnostic) | {_percent(overall['recall'])} |",
        f"| True positives | {overall['true_positives']} |",
        f"| False positives | {overall['false_positives']} |",
        f"| False negatives | {overall['false_negatives']} |",
        f"| Mark density | {_percent(density['mark_density'])} |",
        f"| Marks kept by the short-turn floor | {density['floor_retained_marks']} |",
        f"| Feature suppression rate | {_percent(suppression['feature_suppression_rate'])} |",
        f"| Legend suppression rate | {_percent(suppression['legend_suppression_rate'])} |",
        "",
        "## False positives by category",
        "",
    ]
    categories = payload["false_positives"]["by_category"]
    if categories:
        lines += ["| Category | Count |", "|---|---|"]
        lines += [f"| {name} | {count} |" for name, count in sorted(categories.items())]
    else:
        lines.append("None.")

    lines += [
        "",
        "## Per speaker",
        "",
        "| Speaker | Precision | Recall | Marks |",
        "|---|---|---|---|",
    ]
    for speaker, metrics in sorted(payload["per_speaker"].items()):
        lines.append(
            f"| {speaker} | {_percent(metrics['precision'])} | "
            f"{_percent(metrics['recall'])} | {metrics['marked']} |"
        )

    changes = gate["required_upstream_changes"]
    if changes:
        lines += ["", "## Required upstream changes", ""]
        lines += [f"- {change}" for change in changes]
        lines += [
            "",
            "Recall must not be improved by lowering precision below the gate, and no duration, "
            "hesitation, Tier 2, or web-demo scope may be added to compensate.",
        ]
    return "\n".join(lines)
