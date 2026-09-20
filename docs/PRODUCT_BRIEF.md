# Product brief

## Thesis

Prosody Markup is an open-source pipeline and versioned legend that turns spoken audio into text
carrying typographic marks for how something was said. The bet is not that every mark is obvious
on first sight. It is that repeated paired exposure to text and source audio makes a small,
stable code become punctuation-like and eventually unobtrusive.

## Core bets

1. Some measurable prosodic cues can be rendered without making text distracting.
2. Readers can acquire mappings through paired exposure rather than formal instruction.
3. A small stable legend is more valuable than broad acoustic coverage.
4. False marks are more harmful than missed marks.
5. The legend and evaluation harness, not a particular ASR stack, are the durable artifacts.

## Goals

- Publish an independently implementable legend.
- Produce trusted, conservatively marked text from audio.
- Demonstrate longitudinal acquisition on unseen utterances and speakers.
- Keep normalized features, marks, confidence, provenance, and versioning in an open IR.
- Work locally end to end for Tier 1.

## Non-goals

- Keyboard replacement or global text insertion.
- Transcript cleanup, filler removal, grammar correction, or tone rewriting.
- Emotion, sarcasm, irony, or speaker-intent classification.
- Text-to-speech round trips.
- Multilingual claims before English stress-accent behavior is validated.

## Reference experience

Record or upload speech, inspect a marked transcript, and activate any mark to hear exactly the
source span. The reference UI is a review-and-learning surface, not a dictation product.

## Primary success metric

After at least 500 marked-utterance exposures, users choose which of two unseen audio clips
produced a marked text at greater than 75% accuracy, with every Tier 1 mark above chance and no
material sustained reading-comprehension penalty.
