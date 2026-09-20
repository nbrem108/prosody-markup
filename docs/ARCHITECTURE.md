# Architecture

## Pipeline

1. **Transcribe:** words, timestamps, confidence, speaker attribution, optional events.
2. **Extract:** F0, RMS energy, duration relative to expected phonemes, adjacent silence.
3. **Normalize:** rolling per-speaker baselines and normalized features.
4. **Assign:** declarative thresholds, suppression rules, confidence floors, density cap.
5. **Render:** Markdown, HTML, Unicode, later WebVTT.

Stages 1–3 produce the public IR; stage 4 enriches it with versioned marks; stage 5 consumes it.
An adapter replacement should not require changes to unrelated stages.

## Initial repository slice

The initial implementation begins at stage 4 using checked normalized-feature fixtures. This
separates legend mechanics from ASR and signal-extraction uncertainty and gives future audio work
a deterministic target contract.

## Local-first boundary

Tier 1 must have an entirely local path. Cloud ASR MAY be implemented behind an adapter but MUST
NOT be required by tests, examples, or the reference acquisition loop.

## VAD boundary

Recording VAD may remove leading and trailing non-speech. It MUST NOT delete intra-utterance
silence used by the timing channel. Pause preservation is tested independently of speech trimming.

## Public IR requirements

Every document carries:

- schema and legend versions;
- audio provenance without requiring a public URI;
- ASR and alignment provenance;
- speaker/session baseline metadata;
- tokens with stable IDs, time bounds, original spacing, raw/normalized features, and confidence;
- assigned marks and suppression reasons;
- optional non-speech events.

## Debug artifacts

A future real-audio run should be able to emit transcript JSON, feature tables, pitch and energy
contours, pause map, mark candidates, final marks, and rendered output. Debug output is local and
must not silently upload audio.
