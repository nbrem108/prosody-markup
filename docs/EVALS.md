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

## Running layer 2

`prosody-markup evaluate tasks` builds blind labelling tasks from run bundles, carrying only token
ids and lexical text so annotators are not anchored by system output. `prosody-markup evaluate
report` scores runs against majority labels and emits both JSON and Markdown.

Coverage is checked before the gate. A sample short of 100 utterances, 10 speakers, or 3 annotators
reports `insufficient-data` rather than a pass, however good its precision looks, because a
precision figure from too small a sample is not the result this gate asks for. False positives are
split by category so a failure points somewhere: marks kept only by the short-turn floor are
counted separately from marks no annotator agreed with, and from marks that split the annotators.

## Corpus provenance

Real-audio clips live in [`corpus/`](../corpus/README.md) behind a manifest that records, per clip,
its rights basis, source, consent reference, recording conditions, and checksum. Every clip must
declare one of three rights bases — a compatible license, a contribution agreement, or public
domain — and there is no fourth option, so unclear-rights audio cannot be added rather than merely
being discouraged. Speaker identities are pseudonymous and consent agreements are stored outside
the repository, referenced by identifier.

Validation runs in CI against the manifest alone, so provenance is enforced without the recordings
ever entering the repository. That corpus is engineering evidence for the pipeline; it is not
evidence of acquisition.

## Minimum eval fixture

Each real-audio eval fixture should contain audio provenance, transcript, word timing, speaker
labels and confidence, model provenance, raw and normalized features, human labels, expected marks,
rendered snapshots, and known caveats. The current synthetic contract fixture begins at normalized
features and is explicitly labeled as such.
