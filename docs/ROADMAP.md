# Roadmap

## v0.1 — Does the contract survive? (current)

- Versioned IR and Tier 1 draft legend.
- Declarative assignment with suppression and density cap.
- Markdown, HTML, and Unicode renderers.
- Deterministic fixtures, tests, CLI, and buildable package.
- Next: local ASR adapter and pitch extraction on a small licensed corpus.

Exit for real-audio v0.1: at least 100 utterances from 10+ speakers, three annotators, reported
agreement, and at least 85% precision for public prominence marks.

## v0.2 — Does paired exposure work?

- Local record/upload demo with exact-span audio playback.
- Complete Tier 1 candidate channels after separate validation.
- Publish `SPEC.md` 1.0 only if collisions and rendering behavior are resolved.
- Ten people use their own audio for two weeks; exploratory comprehension is reported.

## v0.3 — Does it hold under volume?

- Multi-speaker diarization and per-speaker baselines.
- Audio-event passthrough and WebVTT.
- Discrimination harness and reading-friction tests.
- Density remains below the cap on at least 90% of conversational turns without clipping.

## v0.4 — Is it acquired?

- Longitudinal results from users past 500 marked utterances.
- Greater than 75% Tier 1 discrimination overall; every mark above chance.
- No material sustained reading-comprehension penalty.

Tier 2, inferred states, multilingual support, streaming, desktop dictation, and TTS are deliberately
unscheduled until v0.4 reports out.
