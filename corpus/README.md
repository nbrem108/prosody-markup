# v0.1 engineering corpus

This corpus exists to exercise the pipeline across real voices, accents, microphones, and
contrastive stress. **It is engineering evidence, not an acquisition result.** Nothing measured
against it says anything about whether readers learn the marks; that question belongs to the
evaluation in [`docs/EVALS.md`](../docs/EVALS.md) and to v0.4.

`manifest.yaml` ships empty and `status: draft`. Entries are added only as real recordings arrive
with their rights basis established. A placeholder speaker, consent, or checksum record does not
belong here: a fabricated provenance record is worse than an absent one, because it looks like
evidence.

## What the manifest must satisfy

Validation runs in CI against the manifest alone, so it never requires the audio:

```bash
uv run prosody-markup corpus validate corpus/manifest.yaml
```

Point it at the audio to verify checksums too. Clips whose audio is absent are skipped unless
`--require-audio` is passed, which is how a machine holding the recordings checks them:

```bash
uv run prosody-markup corpus validate corpus/manifest.yaml \
  --audio-root corpus --require-audio
```

Every clip must declare **why it may be used**, with the evidence for it. `rights_basis` is one of:

| `rights_basis` | Also requires |
|---|---|
| `license` | a `license` from the compatible set (`CC0-1.0`, `CC-BY-4.0`, `CC-BY-SA-4.0`, `CC-BY-3.0`, `public-domain`) |
| `contribution-agreement` | `consent.obtained: true`, `consent.date`, and `consent.agreement_ref` |
| `public-domain` | `source` naming the origin |

There is no fourth option, which is how "no scraped podcast or unclear-rights audio" is enforced
rather than merely asked for. A clip whose rights cannot be stated in these terms cannot be added.

## Keep personal data out of this repository

Speaker identities are pseudonymous (`spk-01`), and `consent.agreement_ref` points at an agreement
stored **outside** the repository. The manifest records *that* consent exists and how to find it,
never the contributor's name or contact details. This mirrors the de-identification rule in
`docs/EVALS.md`.

## Adding a recording

`corpus add` measures what it can and refuses what it cannot verify, so the only things you supply
are the facts only you know:

```bash
uv run prosody-markup corpus add corpus/manifest.yaml \
  --audio corpus/audio/spk-01/clip-0001.wav \
  --speaker spk-01 --accent "General American" \
  --text "I said the BLUE one, not the red one." \
  --rights-basis contribution-agreement \
  --source "recorded for this project" \
  --consent-date 2026-09-21 --agreement-ref "agreements/spk-01" \
  --device "USB condenser" --environment "quiet room"
```

The checksum and duration come from the file, never from typing. The clip id is allocated for you.
A new speaker is created on first use. The whole manifest is re-validated before anything is
written, so a clip cannot be added into a state `corpus validate` would later reject.

It refuses, rather than warns:

| Situation | Why |
|---|---|
| `contribution-agreement` without `--consent-date` and `--agreement-ref` | consent is evidenced or the clip does not go in |
| a license outside the compatible set | redistribution would not be permitted |
| audio already in the manifest under another id | one recording counted twice inflates the corpus without adding evidence |
| audio outside the corpus directory | the manifest must describe files it actually ships |
| a file that is not valid PCM WAV | unusable audio never reaches the manifest |

## A session worth recording

The corpus is meant to exercise **contrastive stress** — the same sentence said with prominence in
different places. Read sentences alone will not do that, which is why generic speech datasets are a
poor fit even where their licenses are compatible.

For each speaker, record the same sentence several times, stressing a different word each time:

- "I said the **blue** one, not the red one."
- "**I** said the blue one, not the red one."
- "I **said** the blue one, not the red one."

Three to five such takes per speaker, five to ten speakers, each clip a few seconds. Keep recording
conditions varied across speakers — different microphones and rooms are the point, not a defect.
Record mono 16 kHz WAV if you can; anything `audio normalize` accepts will do, and conversion is
lossless-by-checksum from there.

Consent before the microphone is on, not after. Store the agreement outside this repository and
reference it by identifier.

## Shape of an entry

Illustrative only — the values below are placeholders, not a record of any real person or
recording:

```yaml
speakers:
  - id: spk-01
    accent: "<broad accent description>"

clips:
  - id: clip-0001
    speaker: spk-01
    path: audio/spk-01/clip-0001.wav
    text: "<exact words spoken>"
    rights_basis: contribution-agreement
    source: "<recorded for this project | archive name and item id>"
    sha256: "<64 lowercase hex characters>"
    duration_s: 3.2
    consent:
      obtained: true
      date: "<YYYY-MM-DD>"
      agreement_ref: "<identifier of the agreement held outside this repo>"
    recording:
      device: "<microphone>"
      environment: "<room and noise conditions>"
```

## Size targets

The v0.1 plan calls for 5–10 speakers with 3–5 short utterances each. Those bounds are checked only
once the manifest declares `status: complete`, so a corpus can be assembled incrementally without
failing CI. Structure and provenance are checked at every status.

Targets, for reference: 5–10 speakers, 3–5 clips per speaker, each clip at most 30 seconds.
