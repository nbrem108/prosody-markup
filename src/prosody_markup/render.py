from __future__ import annotations

import html
import json
import re
from collections.abc import Callable

from .models import Document, Mark, Token

_VOWEL = re.compile(r"([aeiouyAEIOUY])(?!.*[aeiouyAEIOUY])")
_NO_SPACE_BEFORE = {".", ",", "!", "?", ":", ";", ")", "]", "}"}
_NO_SPACE_AFTER = {"(", "[", "{"}


def _lengthen(text: str, strength: int) -> str:
    match = _VOWEL.search(text)
    if not match:
        return text
    return text[: match.end()] + match.group(1) * strength + text[match.end() :]


def _apply_lexical_marks(token: Token) -> str:
    text = token.text
    for mark in token.marks:
        if mark.mark == "lengthening":
            text = _lengthen(text, mark.strength)
    for mark in token.marks:
        if mark.mark == "hesitation":
            text += "." * (2 + mark.strength)
    return text


def _join(parts: list[tuple[str, str, bool | None]]) -> str:
    output = ""
    previous_raw = ""
    for rendered, raw, space_before in parts:
        use_heuristic_no_space = space_before is None and (
            raw in _NO_SPACE_BEFORE or previous_raw in _NO_SPACE_AFTER
        )
        if not output or space_before is False or use_heuristic_no_space:
            output += rendered
        else:
            output += " " + rendered
        previous_raw = raw
    return output


def render_markdown(document: Document) -> str:
    parts: list[tuple[str, str, bool | None]] = []
    for index, token in enumerate(document.tokens):
        if _is_redundant_period(document, index):
            continue
        text = _apply_lexical_marks(token)
        if any(mark.mark == "emphasis" for mark in token.marks):
            text = f"*{text}*"
        parts.append((text, token.text, token.space_before))
    return _join(parts)


def render_unicode(document: Document) -> str:
    parts: list[tuple[str, str, bool | None]] = []
    for index, token in enumerate(document.tokens):
        if _is_redundant_period(document, index):
            continue
        text = _apply_lexical_marks(token)
        if any(mark.mark == "emphasis" for mark in token.marks):
            text = f"_{text}_"
        parts.append((text, token.text, token.space_before))
    return _join(parts)


def _mark_data(marks: list[Mark]) -> tuple[str, str]:
    channels = ",".join(mark.channel for mark in marks)
    confidence = max((mark.confidence for mark in marks), default=0.0)
    return channels, f"{confidence:.3f}"


def _is_redundant_period(document: Document, index: int) -> bool:
    if index == 0 or document.tokens[index].text != ".":
        return False
    return any(mark.mark == "hesitation" for mark in document.tokens[index - 1].marks)


def render_html(document: Document) -> str:
    parts: list[tuple[str, str, bool | None]] = []
    for index, token in enumerate(document.tokens):
        if _is_redundant_period(document, index):
            continue
        text = html.escape(_apply_lexical_marks(token))
        if token.marks:
            channels, confidence = _mark_data(token.marks)
            attrs = (
                f'data-token-id="{html.escape(token.id)}" '
                f'data-channels="{html.escape(channels)}" '
                f'data-confidence="{confidence}" '
                f'data-start="{token.start:.3f}" data-end="{token.end:.3f}"'
            )
            if any(mark.mark == "emphasis" for mark in token.marks):
                text = f"<em {attrs}>{text}</em>"
            else:
                text = f"<span {attrs}>{text}</span>"
        parts.append((text, token.text, token.space_before))
    body = _join(parts)
    return (
        f'<article data-schema-version="{html.escape(document.schema_version)}" '
        f'data-legend-version="{html.escape(document.legend_version)}">{body}</article>'
    )


def render_json(document: Document) -> str:
    return json.dumps(document.to_dict(), indent=2, sort_keys=True)


RENDERERS: dict[str, Callable[[Document], str]] = {
    "markdown": render_markdown,
    "html": render_html,
    "unicode": render_unicode,
    "json": render_json,
}
