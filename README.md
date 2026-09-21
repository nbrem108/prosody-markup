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

Inspect a local PCM WAV before transcription:

```bash
uv run prosody-markup audio inspect input.wav
```

The command reports format, duration, peak level, clipping, silence ratio, source checksum, and
whether conversion to the pipeline's canonical mono 16 kHz signed 16-bit PCM format is required.
It never mutates the source file. Inspection is bounded to WAV files of at most 256 MiB and streams
decoded sample metrics in chunks from one immutable input snapshot.

Convert source audio to that canonical format:

```bash
uv run prosody-markup audio normalize input.wav --output normalized.wav
```

Channel mixing averages channels and resampling uses a deterministic rational polyphase
Blackman-windowed sinc filter, so repeated runs are byte-identical and decimation does not fold
energy above 8 kHz back into the band the pitch extractor reads. Audio that is already mono 16 kHz
signed 16-bit PCM keeps its samples unchanged. Clipped, empty, or over-long input fails closed
rather than producing degraded audio; normalization runs in pure Python and is therefore limited to
10 minutes of audio. Conversion parameters, the source checksum, and the output checksum are
written to a `<output>.manifest.json` sidecar.

Transcribe normalized audio into word-timestamp IR:

```bash
uv run prosody-markup transcribe normalized.wav --output transcript.json --speaker S1
```

Transcription runs behind a provider-neutral adapter, so no engine type reaches the rest of the
pipeline. The reference adapter is local `faster-whisper`, imported lazily and never required by
the package, its tests, or CI; install it yourself to transcribe real audio. Input must already be
canonical, so transcription never silently resamples and loses provenance. Word timestamps are
checked for monotonicity and against the recording length, and model identity, version, and audio
checksum are recorded. Low-confidence words are kept rather than dropped — suppression is the
legend's decision at assignment.

The result is valid word-timestamp IR but not yet assignable: its baseline is left pending because
per-speaker normalization is a later stage.

Measure per-word pitch, with an inspectable debug bundle:

```bash
uv run prosody-markup extract normalized.wav transcript.json \
  --output features.json --debug-dir reports/generated/run-001
```

Pitch is measured with Praat through `praat-parselmouth`, locally and with no model download. Every
eligible word gets voiced coverage, F0 mean, minimum, maximum, and range, or an explicit
suppression reason — `window_too_short`, `unvoiced`, `insufficient_voiced_coverage`,
`f0_out_of_range`, or `octave_ambiguous`. Measurements are written as raw values; normalization
into comparable features and any mark assignment are later stages, so extraction never decides
typography. `--debug-dir` writes `word-features.csv` and `pitch-contour.csv`.

Octave errors get a specific guard. When the ceiling cannot fit the true F0, Praat reports a
confident subharmonic — 100 Hz for a 200 Hz tone, at full voiced coverage and with no within-word
spread — which neither a range check nor a spread check detects. Each word is therefore re-measured
at a raised ceiling and refused when the two analyses disagree by more than half an octave.

Normalize measured pitch against a speaker baseline and assign marks:

```bash
uv run prosody-markup prominence features.json \
  --output assigned.json --candidates candidates.json
```

The session baseline uses a median and a scaled median absolute deviation over the measured words,
in semitones rather than hertz: prominence is perceived as a ratio, so one z threshold only carries
the same meaning across speakers on a log scale, and a robust centre keeps a single octave error or
shout from dragging the scale. Method, parameters, window, and word count are recorded in the IR.

Confidence in the normalized feature is the word's voiced coverage, so a partly measured word falls
below the legend's confidence floor and is never marked. A session with too few measured words, or
with no usable spread, produces no marks rather than guesses. Only the pitch channel is enabled;
duration and timing stay fixture-tested until separately validated. `--candidates` records what the
legend proposed, separately from what the density cap kept.

The whole chain, from a 44.1 kHz stereo recording to marked text:

```bash
uv run prosody-markup audio normalize recording.wav --output normalized.wav
uv run prosody-markup transcribe normalized.wav --output transcript.json --speaker S1
uv run prosody-markup extract normalized.wav transcript.json --output features.json
uv run prosody-markup prominence features.json --output assigned.json
uv run prosody-markup assign assigned.json --format markdown
```

```text
I said the *BLUE* one not the red one
```

Expected marked text:

```text
No, I *got* it. It is fine. I will just redo the whole deck before the morning review with the team tonight....
```

## Repository map

```text
SPEC.md                 versioned legend specification (CC BY 4.0)
legend/tier1.yaml       declarative thresholds and density policy
src/prosody_markup/     IR, audio, transcription, extraction, assignment, renderers, CLI
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

See [docs/ROADMAP.md](docs/ROADMAP.md) for the milestone sequence and
[docs/V0_1_REAL_AUDIO_PLAN.md](docs/V0_1_REAL_AUDIO_PLAN.md) for the ordered implementation plan.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [SPEC.md](SPEC.md). New marks require evidence; ASR
integrations must preserve the public IR contract. Issues and small, test-backed pull requests are
welcome.

## Licensing

Code is licensed under Apache License 2.0. `SPEC.md` and files under `legend/` are licensed under
CC BY 4.0 so the convention can be independently implemented, including in closed products.
Reference corpus items must carry their own recorded provenance and compatible license.
