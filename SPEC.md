# Prosody Markup Legend Specification

**Specification version:** 0.1.0-draft
**SPDX-License-Identifier:** CC-BY-4.0

This document defines a rendering contract for measurable prosodic cues. It is intentionally
small. Compatibility and reader acquisition are more important than covering every acoustic
phenomenon.

## Normative language

The terms **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.

## Tier 1 legend

| Channel | Measured cue | Mark | Example |
|---|---|---|---|
| `pitch` | high-confidence perceived prominence, initially approximated by F0 deviation | emphasis/italics | `I *didn't* say that` |
| `duration` | lengthened vowel or drawl relative to expected phoneme duration | letter doubling | `riiight` |
| `timing` | hesitation pause after a token | ellipsis of 3–5 dots | `I mean... maybe` |

A channel MUST map to exactly one semantic mark. A measured cue MUST NOT silently acquire an
inferred emotional or intentional label.

## Baselines

- Features MUST be normalized per speaker and per session, never globally.
- Implementations SHOULD use a rolling baseline and MUST record the normalization window.
- Multi-speaker assignment MUST NOT proceed without speaker attribution.
- Marks encode deviation from the speaker's own baseline, not an absolute human norm.

## Suppression

An implementation MUST suppress mark assignment when any of these are true:

- ASR confidence is below the configured floor;
- alignment confidence is below the configured floor;
- speaker attribution is missing, lacks confidence, or falls below the configured confidence floor;
- speech overlaps another speaker;
- the relevant acoustic feature is absent or invalid.

Implementations SHOULD expose suppression reasons in the IR.

Implementations MUST reject non-finite features or timestamps, confidence values outside 0–1,
invalid timestamp bounds, duplicate token IDs, and missing baseline-window metadata before mark
assignment.

## Density

Marked-token density is calculated per speaker turn after punctuation restoration, excluding
punctuation-only tokens. Implementations SHOULD supply a stable `turn_id`; when absent, the
reference implementation infers turns from contiguous speaker spans. A token with one or more
marks counts once.

A turn's mark allowance is `floor(eligible_tokens * density_cap)`, where Tier 1 sets `density_cap`
to 0.15. Tier 1 output MUST NOT exceed that allowance except under the short-turn floor defined
below.

When candidates exceed the allowance, retain the candidates with the greatest prominence, where
prominence is the feature value's relative excess over its channel threshold, `(value - threshold)
/ abs(threshold)`. Confidence gates assignment through `confidence_floor`; it MUST NOT be reused as
the ranking signal, because a barely-suprathreshold token in clean audio would otherwise outrank a
strongly marked token in noisier audio. Ties break on confidence, then on token order.

### Short-turn floor

A turn whose allowance floors to zero MUST still be allowed up to `min_marks_per_turn` marks when
at least one candidate survives suppression, bounded by the turn's eligible token count. Most
conversational turns are short; flooring them to zero removes the marks readers need for
acquisition.

This floor is a deliberate and bounded exception to the cap, not a second cap. A turn retained
under it MAY exceed `density_cap`: a four-token turn carrying one mark sits at 25%. The exception
cannot fabricate a mark, because it only ever retains a candidate that already cleared its channel
threshold and `confidence_floor`; it changes how many survivors are kept, never whether a candidate
existed.

Because the floor raises mark coverage on exactly the shortest and least certain turns,
implementations MUST mark which tokens were retained only under it, and evaluation MUST report
precision both with and without those tokens. Tier 1 sets `min_marks_per_turn` to 1; a higher
value is outside the Tier 1 contract.

## Rendering

- Markdown renders `emphasis` with `*...*`.
- HTML renders semantic `<em>` and data attributes for channel, confidence, and audio bounds.
- Unicode/plain text preserves lexical text and uses `_..._` for emphasis.
- Renderers MUST NOT visually encode confidence in version 0.1. Confidence is for filtering and
  inspection, not a hidden additional mark channel.
- Every user-facing interactive renderer MUST make the exact marked audio span available within
  one action.

## Duration realization

A duration mark stores `strength` from 1–3. Renderers double the final vowel grapheme by that
strength. If no vowel is available or the token is unsuitable for deterministic transformation,
assignment MUST suppress the mark with an `unrenderable_lengthening` reason rather than invent
spelling.

## Timing realization

A timing mark stores `strength` from 1–3 and renders 3, 4, or 5 dots respectively. This scale is
experimental and must be validated before the draft reaches 1.0.

## Versioning

Every IR document MUST declare `schema_version` and `legend_version`. Rendered HTML and JSON MUST
preserve both. Breaking changes to Tier 1 require a new major legend version and acquisition data
showing why compatibility should be broken.

## Reserved marks

Bold and underline are permanently unassigned. Bold collides with author-to-reader importance;
underline collides with hyperlinks. Tier 2 and inferred-state marks are outside this draft.
