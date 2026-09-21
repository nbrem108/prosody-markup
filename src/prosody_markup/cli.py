from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .assign import assign_marks, is_eligible
from .audio import AudioError, inspect_wav, normalize_wav
from .corpus import CorpusError, load_manifest, summarize, validate_manifest
from .extract import ExtractionError, PitchSettings, extract_pitch, write_debug_artifacts
from .legend import load_legend
from .models import Document
from .pipeline import PipelineError, run_pipeline
from .prominence import ProminenceError, normalize_and_assign
from .render import RENDERERS
from .transcribe import (
    FasterWhisperAdapter,
    TranscriptionError,
    build_transcript,
)


def _load_document(path: Path) -> Document:
    with path.open(encoding="utf-8") as stream:
        raw: Any = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("IR document must be a JSON object")
    return Document.from_dict(raw)


def _assign(args: argparse.Namespace) -> int:
    document = _load_document(args.input)
    legend = load_legend(args.legend)
    assigned = assign_marks(document, legend)
    print(RENDERERS[args.format](assigned))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    document = _load_document(args.input)
    eligible = sum(is_eligible(token) for token in document.tokens)
    payload = {
        "schema_version": document.schema_version,
        "legend_version": document.legend_version,
        "tokens": len(document.tokens),
        "eligible_tokens": eligible,
        "speakers": sorted({token.speaker for token in document.tokens if token.speaker}),
        "duration_s": document.audio.get("duration_s"),
        "provenance": document.provenance,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _audio_inspect(args: argparse.Namespace) -> int:
    inspection = inspect_wav(args.input)
    print(json.dumps(inspection.to_dict(), indent=2, sort_keys=True))
    return 0


def _audio_normalize(args: argparse.Namespace) -> int:
    manifest = normalize_wav(args.input, args.output)
    payload = manifest.to_dict()
    sidecar = Path(manifest.output_path + ".manifest.json")
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _transcribe(args: argparse.Namespace) -> int:
    adapter = FasterWhisperAdapter(args.model, device=args.device, language=args.language)
    document = build_transcript(args.input, adapter, speaker=args.speaker, turn_id=args.turn_id)
    payload = json.dumps(document.to_dict(), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"wrote {len(document.tokens)} words to {args.output}")
    return 0


def _extract(args: argparse.Namespace) -> int:
    document = _load_document(args.transcript)
    settings = PitchSettings(
        floor_hz=args.pitch_floor,
        ceiling_hz=args.pitch_ceiling,
    )
    extraction = extract_pitch(document, args.input, settings)
    payload = json.dumps(extraction.document.to_dict(), indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")

    measured = sum(1 for word in extraction.words if word.valid)
    print(f"measured {measured}/{len(extraction.words)} eligible words -> {args.output}")
    if args.debug_dir:
        for path in write_debug_artifacts(extraction, args.debug_dir):
            print(f"wrote {path}")
    return 0


def _prominence(args: argparse.Namespace) -> int:
    document = _load_document(args.input)
    legend = load_legend(args.legend)
    result = normalize_and_assign(document, legend)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.document.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.candidates:
        args.candidates.parent.mkdir(parents=True, exist_ok=True)
        args.candidates.write_text(
            json.dumps(result.candidate_rows(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    marked = sum(1 for token in result.document.tokens if token.marks)
    print(
        f"baseline {result.baseline.center_semitones:.2f}±{result.baseline.spread_semitones:.2f} "
        f"semitones from {result.baseline.measured_words} words; "
        f"{len(result.candidates)} candidates -> {marked} marks"
    )
    if result.degenerate_spread:
        print("note: spread below the usable minimum; no word was ranked prominent")
    return 0


def _process(args: argparse.Namespace) -> int:
    adapter = FasterWhisperAdapter(args.model, device=args.device, language=args.language)
    result = run_pipeline(
        args.input,
        adapter,
        args.debug_dir,
        speaker=args.speaker,
        legend=load_legend(args.legend),
        channels=set(args.channels),
        render_format=args.format,
        turn_id=args.turn_id,
    )
    print(result.rendered)
    summary = result.manifest["summary"]
    print(
        f"\n{summary['marks']} marks from {summary['candidates']} candidates "
        f"({summary['measured_words']}/{summary['eligible_words']} words measured); "
        f"bundle in {result.debug_dir}",
        file=sys.stderr,
    )
    return 0


def _corpus_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    problems = validate_manifest(manifest, args.audio_root, require_audio=args.require_audio)
    print(json.dumps(summarize(manifest), indent=2, sort_keys=True))
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\ncorpus provenance is complete", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prosody-markup")
    subparsers = parser.add_subparsers(dest="command", required=True)

    assign_parser = subparsers.add_parser("assign", help="Assign marks from normalized IR")
    assign_parser.add_argument("input", type=Path)
    assign_parser.add_argument("--legend", type=Path)
    assign_parser.add_argument("--format", choices=sorted(RENDERERS), default="markdown")
    assign_parser.set_defaults(handler=_assign)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect an IR document")
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.set_defaults(handler=_inspect)

    audio_parser = subparsers.add_parser("audio", help="Inspect and prepare local audio")
    audio_subparsers = audio_parser.add_subparsers(dest="audio_command", required=True)
    audio_inspect_parser = audio_subparsers.add_parser(
        "inspect", help="Inspect a PCM WAV without modifying it"
    )
    audio_inspect_parser.add_argument("input", type=Path)
    audio_inspect_parser.set_defaults(handler=_audio_inspect)

    audio_normalize_parser = audio_subparsers.add_parser(
        "normalize", help="Convert a PCM WAV to canonical mono 16 kHz 16-bit PCM"
    )
    audio_normalize_parser.add_argument("input", type=Path)
    audio_normalize_parser.add_argument("--output", type=Path, required=True)
    audio_normalize_parser.set_defaults(handler=_audio_normalize)

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe canonical audio into word-timestamp IR"
    )
    transcribe_parser.add_argument("input", type=Path)
    transcribe_parser.add_argument("--output", type=Path, required=True)
    transcribe_parser.add_argument("--speaker", required=True)
    transcribe_parser.add_argument("--turn-id", dest="turn_id", default="turn-1")
    transcribe_parser.add_argument("--model", default="base.en")
    transcribe_parser.add_argument("--device", default="cpu")
    transcribe_parser.add_argument("--language", default="en")
    transcribe_parser.set_defaults(handler=_transcribe)

    extract_parser = subparsers.add_parser(
        "extract", help="Measure per-word pitch features from canonical audio"
    )
    extract_parser.add_argument("input", type=Path)
    extract_parser.add_argument("transcript", type=Path)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--debug-dir", dest="debug_dir", type=Path)
    extract_parser.add_argument("--pitch-floor", dest="pitch_floor", type=float, default=75.0)
    extract_parser.add_argument("--pitch-ceiling", dest="pitch_ceiling", type=float, default=500.0)
    extract_parser.set_defaults(handler=_extract)

    prominence_parser = subparsers.add_parser(
        "prominence",
        help="Normalize measured pitch against a speaker baseline and assign marks",
    )
    prominence_parser.add_argument("input", type=Path)
    prominence_parser.add_argument("--output", type=Path, required=True)
    prominence_parser.add_argument("--candidates", type=Path)
    prominence_parser.add_argument("--legend", type=Path)
    prominence_parser.set_defaults(handler=_prominence)

    process_parser = subparsers.add_parser(
        "process", help="Run every local stage from WAV to marked text"
    )
    process_parser.add_argument("input", type=Path)
    process_parser.add_argument("--speaker", required=True)
    process_parser.add_argument("--debug-dir", dest="debug_dir", type=Path, required=True)
    process_parser.add_argument("--channels", nargs="+", default=["pitch"])
    process_parser.add_argument("--format", choices=sorted(RENDERERS), default="markdown")
    process_parser.add_argument("--turn-id", dest="turn_id", default="turn-1")
    process_parser.add_argument("--legend", type=Path)
    process_parser.add_argument("--model", default="base.en")
    process_parser.add_argument("--device", default="cpu")
    process_parser.add_argument("--language", default="en")
    process_parser.set_defaults(handler=_process)

    corpus_parser = subparsers.add_parser("corpus", help="Work with the engineering corpus")
    corpus_subparsers = corpus_parser.add_subparsers(dest="corpus_command", required=True)
    corpus_validate_parser = corpus_subparsers.add_parser(
        "validate", help="Check corpus manifest structure, rights basis, and checksums"
    )
    corpus_validate_parser.add_argument("manifest", type=Path)
    corpus_validate_parser.add_argument("--audio-root", dest="audio_root", type=Path)
    corpus_validate_parser.add_argument(
        "--require-audio", dest="require_audio", action="store_true"
    )
    corpus_validate_parser.set_defaults(handler=_corpus_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (
        AudioError,
        TranscriptionError,
        ExtractionError,
        ProminenceError,
        PipelineError,
        CorpusError,
    ) as exc:
        print(f"prosody-markup: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
