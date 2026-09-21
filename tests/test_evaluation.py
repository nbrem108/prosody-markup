from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from prosody_markup.evaluate import (
    MIN_ANNOTATORS,
    MIN_SPEAKERS,
    MIN_UTTERANCES,
    PRECISION_GATE,
    Annotation,
    EvaluationError,
    build_tasks,
    evaluate,
    fleiss_kappa,
    load_annotations,
    render_report,
    tasks_payload,
)
from prosody_markup.models import Document, Mark, Token


def _token(index: int, text: str, *, marked: bool = False, floor: bool = False) -> Token:
    return Token(
        id=f"w{index:04d}",
        text=text,
        start=index * 0.3,
        end=index * 0.3 + 0.25,
        speaker="S1",
        asr_confidence=0.97,
        alignment_confidence=0.97,
        turn_id="turn-1",
        speaker_confidence=1.0,
        marks=[Mark(channel="pitch", mark="emphasis", confidence=0.95)] if marked else [],
        density_floor_retained=floor,
    )


def _document(speaker: str, marked: set[int], *, floor: set[int] | None = None) -> Document:
    floor = floor or set()
    return Document(
        schema_version="0.1.0",
        legend_version="tier1@0.1.0-draft",
        audio={"duration_s": 4.0},
        provenance={"kind": "local-asr"},
        baseline={"speaker": speaker, "window_s": 4.0},
        tokens=[
            _token(index, f"word{index}", marked=index in marked, floor=index in floor)
            for index in range(5)
        ],
    )


def _corpus(utterances: int = 4, speakers: int = 2) -> dict[str, Document]:
    return {
        f"run-{index:03d}": _document(f"spk-{index % speakers:02d}", marked={1})
        for index in range(utterances)
    }


def _annotations(
    documents: dict[str, Document], prominent: set[str], count: int = 3
) -> list[Annotation]:
    return [
        Annotation(
            annotator=f"ann-{number}",
            labels={utterance: sorted(prominent) for utterance in documents},
        )
        for number in range(count)
    ]


def test_tasks_carry_no_system_output() -> None:
    """Annotators must label before seeing what the system decided."""
    documents = {"run-001": _document("spk-01", marked={1, 3}, floor={3})}

    payload = tasks_payload(build_tasks(documents))

    serialized = json.dumps(payload)
    assert "emphasis" not in serialized
    assert "marks" not in serialized
    assert "density_floor_retained" not in serialized
    assert "f0_z" not in serialized
    (task,) = payload["tasks"]
    assert [word["token_id"] for word in task["words"]] == [f"w{i:04d}" for i in range(5)]
    assert set(task["words"][0]) == {"token_id", "text"}


def test_tasks_require_a_speaker() -> None:
    document = _document("", marked=set())

    with pytest.raises(EvaluationError, match="no speaker"):
        build_tasks({"run-001": document})


def test_fleiss_kappa_is_one_for_perfect_agreement() -> None:
    # Three raters, unanimous on every item, mixed across items.
    assert fleiss_kappa([(3, 0), (0, 3), (3, 0), (0, 3)]) == pytest.approx(1.0)


def test_fleiss_kappa_goes_negative_for_systematic_disagreement() -> None:
    """Raters who reliably split against each other score below chance."""
    assert fleiss_kappa([(2, 1), (1, 2), (2, 1), (1, 2)]) == pytest.approx(-1 / 3)


def test_fleiss_kappa_reports_partial_agreement() -> None:
    assert fleiss_kappa([(3, 0), (3, 0), (0, 3), (2, 1), (0, 3), (1, 2)]) == pytest.approx(
        0.5556, abs=1e-4
    )


def test_fleiss_kappa_is_undefined_without_variation() -> None:
    """Every rater choosing one category makes chance agreement total."""
    assert fleiss_kappa([(3, 0), (3, 0)]) is None


def test_fleiss_kappa_is_undefined_for_a_single_rater() -> None:
    assert fleiss_kappa([(1, 0), (0, 1)]) is None


def test_precision_counts_majority_labels() -> None:
    documents = {"run-001": _document("spk-01", marked={1, 2})}
    annotations = [
        Annotation("ann-0", {"run-001": ["w0001"]}),
        Annotation("ann-1", {"run-001": ["w0001"]}),
        Annotation("ann-2", {"run-001": ["w0002"]}),
    ]

    overall = evaluate(documents, annotations).payload["overall"]

    # w0001 has 2 of 3 votes; w0002 has 1 of 3, so it is a false positive.
    assert overall["true_positives"] == 1
    assert overall["false_positives"] == 1
    assert overall["precision"] == pytest.approx(0.5)


def test_recall_counts_missed_prominence() -> None:
    documents = {"run-001": _document("spk-01", marked=set())}
    annotations = _annotations(documents, {"w0003"})

    overall = evaluate(documents, annotations).payload["overall"]

    assert overall["false_negatives"] == 1
    assert overall["recall"] == pytest.approx(0.0)
    assert overall["precision"] is None


def test_false_positives_are_categorized() -> None:
    documents = {
        "run-001": _document("spk-01", marked={1}, floor={1}),
        "run-002": _document("spk-02", marked={2}),
        "run-003": _document("spk-03", marked={3}),
    }
    annotations = [
        Annotation("ann-0", {"run-001": [], "run-002": [], "run-003": ["w0003"]}),
        Annotation("ann-1", {"run-001": [], "run-002": [], "run-003": []}),
        Annotation("ann-2", {"run-001": [], "run-002": [], "run-003": []}),
    ]

    categories = evaluate(documents, annotations).payload["false_positives"]["by_category"]

    assert categories["density_floor_retained"] == 1
    assert categories["no_annotator_agreed"] == 1
    assert categories["split_decision"] == 1


def test_gate_refuses_to_pass_on_too_small_a_sample() -> None:
    """A precision figure from four utterances is not a gate result."""
    documents = _corpus(utterances=4, speakers=2)
    annotations = _annotations(documents, {"w0001"})

    payload = evaluate(documents, annotations).payload

    assert payload["overall"]["precision"] == pytest.approx(1.0)
    assert payload["gate"]["status"] == "insufficient-data"
    assert payload["coverage"]["meets_minimums"] is False
    assert any("utterances" in shortfall for shortfall in payload["coverage"]["shortfalls"])
    assert any("speakers" in shortfall for shortfall in payload["coverage"]["shortfalls"])


def test_gate_reports_a_shortfall_in_annotators() -> None:
    documents = _corpus()
    annotations = _annotations(documents, {"w0001"}, count=2)

    coverage = evaluate(documents, annotations).payload["coverage"]

    assert any("annotators" in shortfall for shortfall in coverage["shortfalls"])
    assert coverage["minimums"]["annotators"] == MIN_ANNOTATORS


def _sufficient_sample(precision: float) -> tuple[dict[str, Document], list[Annotation]]:
    """A sample meeting every coverage minimum, with a chosen precision."""
    documents: dict[str, Document] = {}
    labels: dict[str, list[str]] = {}
    wrong = round(MIN_UTTERANCES * (1 - precision))
    for index in range(MIN_UTTERANCES):
        name = f"run-{index:03d}"
        documents[name] = _document(f"spk-{index % MIN_SPEAKERS:02d}", marked={1})
        # A run labelled elsewhere makes the system's mark a false positive.
        labels[name] = ["w0003"] if index < wrong else ["w0001"]
    annotations = [Annotation(f"ann-{number}", dict(labels)) for number in range(MIN_ANNOTATORS)]
    return documents, annotations


def test_gate_passes_on_a_sufficient_sample_above_the_threshold() -> None:
    documents, annotations = _sufficient_sample(precision=0.90)

    payload = evaluate(documents, annotations).payload

    assert payload["coverage"]["meets_minimums"] is True
    assert payload["overall"]["precision"] == pytest.approx(0.90, abs=0.01)
    assert payload["gate"]["status"] == "pass"
    assert payload["gate"]["required_upstream_changes"] == []


def test_gate_fails_below_the_threshold_and_names_upstream_changes() -> None:
    documents, annotations = _sufficient_sample(precision=0.60)

    payload = evaluate(documents, annotations).payload

    assert payload["overall"]["precision"] == pytest.approx(0.60, abs=0.01)
    assert payload["gate"]["status"] == "fail"
    assert payload["gate"]["threshold"] == PRECISION_GATE
    assert payload["gate"]["required_upstream_changes"]


def test_held_out_split_excludes_tuning_speakers() -> None:
    documents, annotations = _sufficient_sample(precision=0.90)

    payload = evaluate(documents, annotations, tuning_speakers={"spk-00", "spk-01"}).payload

    held_out = payload["held_out"]
    assert held_out is not None
    assert "spk-00" not in held_out["speakers"]
    assert held_out["tuning_speakers"] == ["spk-00", "spk-01"]
    assert payload["gate"]["measured_on"] == "held_out"


def test_density_and_suppression_are_reported() -> None:
    documents = {"run-001": _document("spk-01", marked={1}, floor={1})}
    documents["run-001"].tokens[2].feature_suppression = {"pitch": "unvoiced"}
    documents["run-001"].tokens[3].suppressed = ["low_asr_confidence"]
    annotations = _annotations(documents, {"w0001"})

    payload = evaluate(documents, annotations).payload

    assert payload["density"]["eligible_words"] == 5
    assert payload["density"]["marked_words"] == 1
    assert payload["density"]["mark_density"] == pytest.approx(0.2)
    assert payload["density"]["floor_retained_marks"] == 1
    assert payload["suppression"]["feature_suppressed_words"] == 1
    assert payload["suppression"]["legend_suppressed_words"] == 1


def test_partly_labelled_utterances_are_left_out_rather_than_weighted() -> None:
    """An utterance not every annotator labelled cannot carry a majority."""
    documents = _corpus(utterances=2, speakers=1)
    annotations = [
        Annotation("ann-0", {"run-000": ["w0001"], "run-001": ["w0001"]}),
        Annotation("ann-1", {"run-000": ["w0001"]}),
        Annotation("ann-2", {"run-000": ["w0001"]}),
    ]

    payload = evaluate(documents, annotations).payload

    assert payload["coverage"]["scored_words"] == 5


def test_mismatched_task_sets_block_the_gate() -> None:
    documents, annotations = _sufficient_sample(precision=0.95)
    annotations[0] = Annotation(
        annotations[0].annotator, annotations[0].labels, tasks_sha256="deadbeef"
    )

    payload = evaluate(documents, annotations, tasks_sha256="cafe1234").payload

    assert payload["coverage"]["annotators_with_mismatched_tasks"] == ["ann-0"]
    assert payload["gate"]["status"] == "insufficient-data"


def test_report_headline_is_the_verdict() -> None:
    documents, annotations = _sufficient_sample(precision=0.60)

    markdown = render_report(evaluate(documents, annotations))

    assert "NO-GO" in markdown
    assert "Required upstream changes" in markdown
    assert "must not be improved by lowering precision" in markdown


def test_report_marks_an_insufficient_sample_as_not_a_result() -> None:
    documents = _corpus()
    markdown = render_report(evaluate(documents, _annotations(documents, {"w0001"})))

    assert "INSUFFICIENT DATA" in markdown
    assert "not a gate result" in markdown


def test_load_annotations_requires_a_labels_object(tmp_path: Path) -> None:
    (tmp_path / "ann.json").write_text(json.dumps({"annotator": "a"}), encoding="utf-8")

    with pytest.raises(EvaluationError, match="labels"):
        load_annotations(tmp_path)


def test_load_annotations_reports_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="no annotation files"):
        load_annotations(tmp_path)


def _write_run(root: Path, name: str, document: Document) -> None:
    run = root / name
    run.mkdir(parents=True)
    (run / "assigned.json").write_text(json.dumps(document.to_dict()), encoding="utf-8")


def test_evaluate_cli_builds_tasks_and_reports(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for index in range(3):
        _write_run(runs, f"run-{index:03d}", _document(f"spk-{index:02d}", marked={1}))

    tasks_out = tmp_path / "tasks.json"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "evaluate",
            "tasks",
            str(runs),
            "--output",
            str(tasks_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    assert "tasks_sha256:" in built.stdout
    payload: dict[str, Any] = json.loads(tasks_out.read_text(encoding="utf-8"))
    assert len(payload["tasks"]) == 3

    labels = tmp_path / "labels"
    labels.mkdir()
    for number in range(3):
        (labels / f"ann-{number}.json").write_text(
            json.dumps(
                {
                    "annotator": f"ann-{number}",
                    "labels": {f"run-{index:03d}": ["w0001"] for index in range(3)},
                }
            ),
            encoding="utf-8",
        )

    report_out = tmp_path / "report.json"
    markdown_out = tmp_path / "report.md"
    reported = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "evaluate",
            "report",
            str(runs),
            "--labels",
            str(labels),
            "--output",
            str(report_out),
            "--markdown",
            str(markdown_out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Too small a sample, so the gate withholds a verdict rather than passing.
    assert reported.returncode == 3, reported.stderr
    assert "INSUFFICIENT DATA" in reported.stdout
    assert json.loads(report_out.read_text(encoding="utf-8"))["gate"]["status"] == (
        "insufficient-data"
    )
    assert markdown_out.exists()


def test_evaluate_cli_reports_missing_runs_without_traceback(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "evaluate",
            "tasks",
            str(tmp_path),
            "--output",
            str(tmp_path / "tasks.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "no run bundles" in completed.stderr
    assert "Traceback" not in completed.stderr
