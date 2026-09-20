# Prosody Markup

**An open, testable convention for writing how speech was said, not only what was said.**

Prosody Markup maps high-confidence acoustic cues to a small, stable typographic legend:

- perceived pitch prominence → *italics*
- lengthened vowel or drawl → letter doubling
- hesitation pause → scaled ellipsis

The durable artifact is the legend. The pipeline exists to demonstrate, test, and stress it.

> **Pre-alpha:** this repository currently implements the versioned intermediate representation,
> conservative mark assignment from normalized features, Markdown/HTML/Unicode rendering, and
> deterministic fixtures. Audio transcription and feature extraction are the next milestone; do
> not treat current fixture output as production speech analysis.

## Why this exists

Transcription preserves words while discarding pitch, duration, amplitude, and timing. Prosody
Markup tests a focused bet: readers can acquire a small typographic code through repeated paired
exposure to marked text and source audio. We optimize for acquisition, not acoustic coverage.

## Principles

1. **Iconic beats arbitrary.** Marks should resemble what they encode.
2. **Precision beats recall.** A missed mark is silence; a false mark teaches misinformation.
3. **Paired exposure teaches the code.** Every marked span in a user-facing surface must link to
   its source audio.
4. **Local-first.** Tier 1 must work offline without an account or API key.
5. **One job.** This is not a dictation app, emotion classifier, transcript cleaner, or TTS system.

## Quick start

```bash
uv sync --extra dev
uv run prosody-markup assign examples/tier1-input.json --format markdown
uv run prosody-markup inspect examples/tier1-input.json
uv run pytest
```

Expected marked text:

```text
No, I *got* it. It is fine. I will just redo the whole deck before the morning review with the team tonight....
```

## Repository map

```text
SPEC.md                 versioned legend specification (CC BY 4.0)
legend/tier1.yaml       declarative thresholds and density policy
src/prosody_markup/     IR, assignment engine, renderers, CLI
examples/               deterministic input fixtures
tests/                  contract, assignment, and renderer tests
docs/                   product brief, architecture, evals, roadmap, governance
```

## Current scope

The first implementation slice deliberately starts downstream of audio analysis. It proves that:

- normalized token features can be represented without losing provenance;
- a legend config can assign marks without hard-coded channel rules;
- confidence floors, suppression rules, and the density cap are deterministic;
- multiple renderers consume the same versioned IR.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the path to real audio.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [SPEC.md](SPEC.md). New marks require evidence; ASR
integrations must preserve the public IR contract. Issues and small, test-backed pull requests are
welcome.

## Licensing

Code is licensed under Apache License 2.0. `SPEC.md` and files under `legend/` are licensed under
CC BY 4.0 so the convention can be independently implemented, including in closed products.
Reference corpus items must carry their own recorded provenance and compatible license.
