# v0.1 Real-Audio Implementation Plan

## Goal

Prove that a fully local pipeline can place conservative pitch-prominence italics on clean,
single-speaker English WAV recordings with at least 85% precision against majority human labels.

This phase tests one claim only:

> High-confidence perceived prominence can be extracted from real audio and rendered as italics
> without teaching readers false associations.

Duration, hesitation, diarization, streaming, a web demo, and Tier 2 are outside this phase.

## Fixed scope

- Input: prerecorded PCM WAV.
- Language: English.
- Speakers: one known speaker per file.
- Processing: fully local and credential-free.
- Public mark: `pitch → emphasis` only.
- Output: marked Markdown plus versioned JSON IR and inspectable debug artifacts.
- Evaluation: speaker-held-out precision first; recall remains diagnostic.

## Architecture decisions

### Audio boundary

The first slice accepts WAV without silently converting it. `audio inspect` reports format,
duration, channel count, sample width, peak level, clipping, and whether conversion to mono 16 kHz
PCM is required. Conversion becomes a separate explicit stage so source provenance is never lost.

### ASR adapter

Start with `faster-whisper` behind a narrow adapter protocol. It must emit words with stable IDs,
text, timestamps, confidence, spacing, and one known speaker/turn. Model downloads are explicit;
tests and checked fixtures cannot require a network connection.

### Pitch extraction

Use Praat through `praat-parselmouth`. For each eligible word interval retain voiced-frame coverage,
F0 mean, maximum, range, and a validity/suppression reason. Never interpolate through long unvoiced
regions merely to make a mark candidate.

### Normalization

Begin with a session-level single-speaker baseline. Record the method and window in IR provenance.
Use a robust center/spread calculation where the available voiced sample permits it. Rolling
baselines wait until longer-form recordings demonstrate the need.

### Assignment

Reuse the existing declarative legend, validation, confidence floors, suppression rules, and
per-turn density cap. During v0.1, the process command enables only the pitch channel even though
other candidate channels remain fixture-tested.

## Ordered implementation slices

GitHub tracking:

1. [#1 — WAV input contract and inspection](https://github.com/nbrem108/prosody-markup/issues/1)
2. [#2 — Canonical audio normalization](https://github.com/nbrem108/prosody-markup/issues/2)
3. [#3 — Local word-timestamp ASR adapter](https://github.com/nbrem108/prosody-markup/issues/3)
4. [#4 — Conservative per-word pitch extraction](https://github.com/nbrem108/prosody-markup/issues/4)
5. [#5 — Single-speaker normalization and italics](https://github.com/nbrem108/prosody-markup/issues/5)
6. [#6 — End-to-end process command and debug bundle](https://github.com/nbrem108/prosody-markup/issues/6)
7. [#7 — Licensed engineering corpus](https://github.com/nbrem108/prosody-markup/issues/7)
8. [#8 — Annotation harness and precision report](https://github.com/nbrem108/prosody-markup/issues/8)

### 1. Audio contract and inspection

**Outcome:** unsupported or poor-quality WAV input fails or reports actionable diagnostics before
ASR work begins.

Create:

- `src/prosody_markup/audio.py`
- `tests/test_audio.py`

Modify:

- `src/prosody_markup/cli.py`
- `README.md`

CLI:

```bash
prosody-markup audio inspect input.wav
```

Acceptance criteria:

- Reports duration, sample rate, channels, sample width, frame count, peak amplitude, clipping,
  silence ratio, and `conversion_required`.
- Reports explicit `wav`/`pcm` format fields, enforces a 256 MiB safety limit, and calculates sample
  metrics in bounded chunks from the same immutable snapshot used for the source checksum.
- Rejects unreadable, empty, compressed, or unsupported WAV safely.
- Does not mutate the source file.
- Uses generated test audio; no binary fixture is committed.

### 2. Explicit audio normalization

**Outcome:** supported source audio is converted reproducibly to mono 16 kHz signed 16-bit PCM
while retaining source and derived-file provenance.

CLI:

```bash
prosody-markup audio normalize input.wav --output normalized.wav
```

Acceptance criteria:

- Resampling and channel mixing are deterministic.
- Existing compliant audio can pass through without quality loss.
- Source hash and conversion parameters appear in a sidecar manifest.
- Clipped or empty input fails closed.

### 3. Local ASR adapter

**Outcome:** normalized WAV becomes checked word-timestamp IR.

CLI:

```bash
prosody-markup transcribe normalized.wav --output transcript.json
```

Acceptance criteria:

- Adapter interface is independent of `faster-whisper`.
- Local model and version are recorded.
- Word timestamps are monotonic and bounded by audio duration.
- Low-confidence words remain present but are suppressible.
- Unit tests use a deterministic fake adapter; optional integration tests use an explicit local
  model marker.

### 4. Pitch extraction and debug artifacts

**Outcome:** every eligible word receives measurable pitch features or an explicit suppression
reason.

CLI:

```bash
prosody-markup extract normalized.wav transcript.json \
  --output features.json \
  --debug-dir reports/generated/run-001
```

Acceptance criteria:

- Extracts voiced coverage, F0 mean/max/range, and validity.
- Emits `word-features.csv` and `pitch-contour.csv`.
- Handles unvoiced words, octave errors, short windows, and out-of-range F0 conservatively.
- No mark assignment occurs in the extractor.

### 5. Single-speaker normalization and pitch assignment

**Outcome:** raw pitch features become normalized candidates and conservative italics.

Acceptance criteria:

- Baseline method and parameters are recorded.
- Speaker/session metadata is required.
- Only the pitch channel is enabled.
- Density and confidence suppression remain enforced.
- Candidate and final-mark artifacts are distinct.

### 6. End-to-end process command

**Outcome:** one command produces marked text and a complete diagnostic bundle.

```bash
prosody-markup process input.wav \
  --channels pitch \
  --format markdown \
  --debug-dir reports/generated/run-001
```

Required artifacts:

```text
reports/generated/run-001/
  audio-summary.json
  transcript.json
  word-features.csv
  pitch-contour.csv
  candidates.json
  assigned.json
  rendered.md
  run-manifest.json
```

### 7. Licensed engineering corpus

**Outcome:** a small non-evaluative corpus exercises real accents, voices, microphones, and
contrastive stress.

- 5–10 speakers.
- 3–5 short utterances per speaker.
- CC0, compatible CC license, public domain, or explicit contribution agreement.
- Manifest records speaker consent, source, license, text, recording conditions, and checksum.
- Corpus documentation must state that it is for engineering, not evidence of acquisition.

### 8. Annotation and v0.1 evaluation

**Outcome:** a speaker-held-out report determines whether pitch italics deserve to proceed to the
paired-exposure demo.

- At least 100 utterances from at least 10 speakers.
- At least three independent annotators.
- Annotators label perceived prominence before seeing system output.
- Report inter-annotator agreement, precision, recall, density, suppression rate, and categorized
  false positives.
- Gate: at least 85% precision for public prominence marks.

If the gate fails, tune extraction, normalization, and thresholds. Do not add other marks or build
the web demo to compensate.

## Verification gate for every slice

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
```

Every implementation PR must also run its focused test first, record the RED/GREEN evidence, and
smoke the changed CLI surface.

## Deferred until the v0.1 gate passes

- Duration/drawl and hesitation marks on real audio.
- Multi-speaker diarization.
- Rolling baselines.
- Web upload/recording demo.
- Exact-span playback and longitudinal acquisition instrumentation.
- WebVTT, streaming, multilingual support, desktop dictation, TTS, and inferred states.
