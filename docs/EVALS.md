# Evaluation plan

The eval is the runnable product contract. Results are reported by legend version, schema version,
model/adaptor versions, dataset slice, and speaker.

## Layer 1: signal validity

**Data:** clean and noisy clips with checked timestamps and acoustic measurements.
**Task:** extract and normalize F0, duration, energy, and silence.
**Scores:** extraction coverage, alignment error, invalid-feature rate, and reproducibility.

## Layer 2: rendering validity

**Data:** at least 100 candidate utterances from at least 10 speakers, labeled by at least three
listeners for perceived prominence/drawl/hesitation.
**Task:** assign and render Tier 1 marks.
**Scores:** precision first, recall second, inter-annotator agreement, mark density, suppression
rate, and categorized false positives.

Initial gate: each public mark reaches at least 85% precision against majority labels. Recall is
diagnostic and must not be improved by lowering precision below the gate.

## Layer 3: acquisition validity

**Data:** longitudinal exposures plus held-out utterances and speakers.
**Task:** show marked text and two audio clips; ask which clip produced the text.
**Scores:** forced-choice accuracy overall and per mark, learning curve by exposure count, transfer
to unseen speakers, and retention after an exposure gap.

Target: greater than 75% overall accuracy after 500 exposures, every mark above chance.

## Reading-friction guardrail

Compare marked and unmarked text on factual comprehension, reading time, disable rate, and user
reported distraction. Acquisition does not count as product success if comprehension materially
falls or users routinely disable the marks.

## Online flywheel

With explicit consent, locally captured scrub events, corrections, suppressed candidates, and
opt-out/disable events can be exported as de-identified eval candidates. Audio collection is never
implicit. Production failures promoted into fixtures retain license and consent provenance.

## Minimum eval fixture

Each fixture contains audio provenance, transcript, word timing, speaker labels, model provenance,
raw and normalized features, human labels, expected marks, rendered snapshots, and known caveats.
