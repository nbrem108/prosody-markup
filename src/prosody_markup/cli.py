from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .assign import assign_marks, is_eligible
from .legend import load_legend
from .models import Document
from .render import RENDERERS


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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
