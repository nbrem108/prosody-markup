from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .audio import inspect_wav, normalize_wav
from .extract import PitchSettings, extract_pitch, write_debug_artifacts
from .legend import Legend, load_legend
from .models import Document
from .prominence import REAL_AUDIO_CHANNELS, BaselineSettings, normalize_and_assign
from .render import RENDERERS
from .transcribe import TranscriptionAdapter, build_transcript

RUN_MANIFEST_SCHEMA_VERSION = "run-manifest@0.1.0"

STAGES = ("inspect", "normalize_audio", "transcribe", "extract", "prominence", "render")


class PipelineError(ValueError):
    """Raised when a run cannot proceed and no output should be fabricated."""


def _package_version() -> str:
    try:
        return version("prosody-markup")
    except PackageNotFoundError:  # pragma: no cover - only when run from a bare checkout
        return "unknown"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(slots=True)
class RunArtifacts:
    """Files written so far, in the order they were produced."""

    directory: Path
    names: list[str] = field(default_factory=list)

    def record(self, name: str) -> None:
        if name not in self.names:
            self.names.append(name)

    def rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for name in self.names:
            path = self.directory / name
            if path.exists():
                rows.append({"name": name, "sha256": _digest(path)})
        return rows


@dataclass(slots=True)
class RunResult:
    document: Document
    rendered: str
    manifest: dict[str, Any]
    debug_dir: Path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pipeline(
    audio_path: Path,
    adapter: TranscriptionAdapter,
    debug_dir: Path,
    *,
    speaker: str,
    legend: Legend | None = None,
    channels: set[str] | None = None,
    render_format: str = "markdown",
    pitch_settings: PitchSettings | None = None,
    baseline_settings: BaselineSettings | None = None,
    turn_id: str = "turn-1",
) -> RunResult:
    """Run every local stage from source WAV to marked text.

    Each stage writes its artifact before the next begins, so a failure leaves
    the bundle it managed to produce plus a manifest naming the stage that
    failed. Nothing downstream of a failure is written, and no output is
    fabricated to fill the gap.
    """
    if render_format not in RENDERERS:
        raise PipelineError(f"unknown render format: {render_format}")

    legend = legend or load_legend()
    pitch_settings = pitch_settings or PitchSettings()
    baseline_settings = baseline_settings or BaselineSettings()
    enabled = channels or set(REAL_AUDIO_CHANNELS)

    debug_dir.mkdir(parents=True, exist_ok=True)
    artifacts = RunArtifacts(directory=debug_dir)
    completed: list[str] = []
    started_at = datetime.now(UTC).isoformat()

    def manifest(
        status: str,
        *,
        failed_stage: str | None = None,
        error: str | None = None,
        provenance: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "status": status,
            "created_at": started_at,
            "source_audio": str(audio_path),
            "stages": list(STAGES),
            "completed_stages": list(completed),
            "failed_stage": failed_stage,
            "error": error,
            "settings": {
                "speaker": speaker,
                "turn_id": turn_id,
                "enabled_channels": sorted(enabled),
                "render_format": render_format,
                "pitch": pitch_settings.to_dict(),
                "baseline": baseline_settings.to_dict(),
            },
            "versions": {
                "prosody_markup": _package_version(),
                "legend_version": legend.legend_version,
                "schema_version": legend.schema_version,
                **(provenance or {}),
            },
            "artifacts": artifacts.rows(),
            "summary": summary or {},
        }

    def fail(stage: str, exc: Exception) -> None:
        _write_json(
            debug_dir / "run-manifest.json",
            manifest("failed", failed_stage=stage, error=f"{type(exc).__name__}: {exc}"),
        )

    try:
        inspection = inspect_wav(audio_path)
        _write_json(debug_dir / "audio-summary.json", inspection.to_dict())
        artifacts.record("audio-summary.json")
        completed.append("inspect")
    except Exception as exc:
        fail("inspect", exc)
        raise

    try:
        normalized_path = debug_dir / "normalized.wav"
        normalization = normalize_wav(audio_path, normalized_path)
        _write_json(debug_dir / "audio-normalization.json", normalization.to_dict())
        artifacts.record("normalized.wav")
        artifacts.record("audio-normalization.json")
        completed.append("normalize_audio")
    except Exception as exc:
        fail("normalize_audio", exc)
        raise

    try:
        document = build_transcript(normalized_path, adapter, speaker=speaker, turn_id=turn_id)
        _write_json(debug_dir / "transcript.json", document.to_dict())
        artifacts.record("transcript.json")
        completed.append("transcribe")
    except Exception as exc:
        fail("transcribe", exc)
        raise

    try:
        extraction = extract_pitch(document, normalized_path, pitch_settings)
        for path in write_debug_artifacts(extraction, debug_dir):
            artifacts.record(path.name)
        completed.append("extract")
    except Exception as exc:
        fail("extract", exc)
        raise

    try:
        prominence = normalize_and_assign(
            extraction.document, legend, baseline_settings, channels=enabled
        )
        _write_json(debug_dir / "candidates.json", prominence.candidate_rows())
        artifacts.record("candidates.json")
        _write_json(debug_dir / "assigned.json", prominence.document.to_dict())
        artifacts.record("assigned.json")
        completed.append("prominence")
    except Exception as exc:
        fail("prominence", exc)
        raise

    try:
        rendered = RENDERERS[render_format](prominence.document)
        rendered_name = (
            "rendered.md" if render_format == "markdown" else f"rendered.{render_format}"
        )
        (debug_dir / rendered_name).write_text(rendered + "\n", encoding="utf-8")
        artifacts.record(rendered_name)
        completed.append("render")
    except Exception as exc:
        fail("render", exc)
        raise

    provenance = prominence.document.provenance
    payload = manifest(
        "complete",
        provenance={
            "audio_inspection_schema": inspection.schema_version,
            "audio_normalization_schema": normalization.schema_version,
            "asr_model": provenance.get("asr_model"),
            "asr_model_version": provenance.get("asr_model_version"),
            "extractor": provenance.get("extractor"),
            "extractor_version": provenance.get("extractor_version"),
            "extraction_schema": provenance.get("extraction_schema"),
            "baseline_schema": provenance.get("baseline_schema"),
        },
        summary={
            "words": len(prominence.document.tokens),
            "measured_words": sum(1 for word in extraction.words if word.valid),
            "eligible_words": len(extraction.words),
            "candidates": len(prominence.candidates),
            "marks": sum(1 for token in prominence.document.tokens if token.marks),
            "degenerate_spread": prominence.degenerate_spread,
        },
    )
    # The manifest lists every other artifact, so it is written last and is
    # itself the signal that a run finished.
    _write_json(debug_dir / "run-manifest.json", payload)

    return RunResult(
        document=prominence.document,
        rendered=rendered,
        manifest=payload,
        debug_dir=debug_dir,
    )
