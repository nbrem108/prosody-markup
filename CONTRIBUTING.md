# Contributing

Thank you for helping test a convention rather than merely expanding a transcription stack.

## Before coding

Read `SPEC.md`, `docs/ARCHITECTURE.md`, and `docs/EVALS.md`. Open an issue for changes to the public
IR or legend. New marks without evidence will not be accepted.

## Setup and gate

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
uv run prosody-markup assign examples/tier1-input.json --format markdown
```

Use tests first for behavior changes. Keep fixtures deterministic and credential-free. A pull
request must identify its affected layer, validation evidence, compatibility risk, and any corpus
or model provenance changes.

## Scope boundaries

Welcome: eval fixtures, corpus provenance, precision improvements, suppression logic, renderers,
debug artifacts, and replaceable local adapters.

Usually out of scope: global text insertion, grammar cleanup, filler removal, emotion labels,
sarcasm inference, hidden cloud dependencies, and new marks without acquisition evidence.

## Licensing

Contributions to code are Apache-2.0. Contributions to `SPEC.md` and `legend/` are CC-BY-4.0.
Corpus contributions must include explicit license, consent where applicable, and provenance.
