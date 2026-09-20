from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .legend import Legend
from .models import Document, Mark, Token

_WORD = re.compile(r"[A-Za-z0-9]")
_VOWEL = re.compile(r"[aeiouyAEIOUY]")


@dataclass(frozen=True, slots=True)
class Candidate:
    token_index: int
    channel: str
    mark: str
    confidence: float
    strength: int


def is_eligible(token: Token) -> bool:
    return bool(_WORD.search(token.text))


def suppression_reasons(token: Token, legend: Legend) -> list[str]:
    policy = legend.suppression
    reasons: list[str] = []
    if token.asr_confidence < policy.asr_confidence_floor:
        reasons.append("low_asr_confidence")
    if token.alignment_confidence < policy.alignment_confidence_floor:
        reasons.append("low_alignment_confidence")
    if policy.require_speaker and not token.speaker:
        reasons.append("missing_speaker")
    if policy.require_speaker and token.speaker_confidence is None:
        reasons.append("missing_speaker_confidence")
    elif (
        token.speaker_confidence is not None
        and token.speaker_confidence < policy.speaker_confidence_floor
    ):
        reasons.append("low_speaker_confidence")
    if policy.suppress_overlap and token.overlap:
        reasons.append("overlapping_speech")
    return reasons


def _strength(channel: str, value: float, threshold: float) -> int:
    if channel == "timing":
        return max(1, min(3, 1 + int((value - threshold) / 0.35)))
    return max(1, min(3, 1 + int(value - threshold)))


def _turn_keys(tokens: list[Token]) -> dict[int, tuple[str | None, str]]:
    keys: dict[int, tuple[str | None, str]] = {}
    inferred_turn = -1
    previous_speaker: str | None = None
    for index, token in enumerate(tokens):
        if token.turn_id is not None:
            keys[index] = (token.speaker, token.turn_id)
            previous_speaker = token.speaker
            continue
        if index == 0 or token.speaker != previous_speaker:
            inferred_turn += 1
        keys[index] = (token.speaker, f"inferred-{inferred_turn}")
        previous_speaker = token.speaker
    return keys


def _validate_document(document: Document) -> None:
    if not document.baseline:
        raise ValueError("Document baseline metadata is required")
    window = document.baseline.get("window_s")
    if not isinstance(window, (int, float)) or not math.isfinite(window) or window <= 0:
        raise ValueError("baseline.window_s must be a positive finite number")

    seen_ids: set[str] = set()
    for token in document.tokens:
        if not token.id or token.id in seen_ids:
            raise ValueError(f"Token IDs must be nonempty and unique: {token.id!r}")
        seen_ids.add(token.id)
        if not math.isfinite(token.start) or not math.isfinite(token.end):
            raise ValueError(f"Token {token.id} timestamps must be finite")
        if token.start < 0 or token.end < token.start:
            raise ValueError(f"Token {token.id} has invalid timestamp bounds")
        confidences = {
            "asr_confidence": token.asr_confidence,
            "alignment_confidence": token.alignment_confidence,
            **token.feature_confidence,
        }
        speaker_confidence = token.speaker_confidence
        if speaker_confidence is not None:
            confidences["speaker_confidence"] = float(speaker_confidence)
        for name, value in confidences.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Token {token.id} {name} must be finite and between 0 and 1")
        for name, feature_value in token.features.items():
            if feature_value is not None and not math.isfinite(feature_value):
                raise ValueError(f"Token {token.id} feature {name} must be finite")


def assign_marks(document: Document, legend: Legend) -> Document:
    if document.schema_version != legend.schema_version:
        raise ValueError(
            f"Schema mismatch: document={document.schema_version}, legend={legend.schema_version}"
        )
    _validate_document(document)

    candidates: list[Candidate] = []
    turn_keys = _turn_keys(document.tokens)
    eligible_by_turn: dict[tuple[str | None, str], int] = {}
    for index, token in enumerate(document.tokens):
        if is_eligible(token):
            key = turn_keys[index]
            eligible_by_turn[key] = eligible_by_turn.get(key, 0) + 1

    for index, token in enumerate(document.tokens):
        token.marks.clear()
        token.suppressed = suppression_reasons(token, legend)
        if token.suppressed or not is_eligible(token):
            continue
        for channel, rule in legend.channels.items():
            value = token.features.get(rule.feature)
            confidence = token.feature_confidence.get(channel, 0.0)
            if value is None:
                continue
            if value >= rule.threshold and confidence >= rule.confidence_floor:
                if rule.mark == "lengthening" and not _VOWEL.search(token.text):
                    token.suppressed.append("unrenderable_lengthening")
                    continue
                candidates.append(
                    Candidate(
                        token_index=index,
                        channel=channel,
                        mark=rule.mark,
                        confidence=confidence,
                        strength=_strength(channel, value, rule.threshold),
                    )
                )

    candidate_tokens: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        candidate_tokens.setdefault(candidate.token_index, []).append(candidate)

    candidates_by_turn: dict[tuple[str | None, str], list[int]] = {}
    for index in candidate_tokens:
        candidates_by_turn.setdefault(turn_keys[index], []).append(index)

    retained: set[int] = set()
    for key, token_indexes in candidates_by_turn.items():
        allowed = math.floor(eligible_by_turn[key] * legend.density_cap)
        ranked_tokens = sorted(
            token_indexes,
            key=lambda index: (-max(item.confidence for item in candidate_tokens[index]), index),
        )
        retained.update(ranked_tokens[:allowed])

    for index, items in candidate_tokens.items():
        token = document.tokens[index]
        if index not in retained:
            token.suppressed.append("density_cap")
            continue
        for item in sorted(items, key=lambda candidate: candidate.channel):
            token.marks.append(
                Mark(
                    channel=item.channel,
                    mark=item.mark,
                    confidence=item.confidence,
                    strength=item.strength,
                )
            )

    document.legend_version = legend.legend_version
    return document
