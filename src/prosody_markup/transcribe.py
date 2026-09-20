from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .assign import validate_tokens
from .audio import TARGET_SAMPLE_RATE, inspect_wav
from .models import Document, Token

SCHEMA_VERSION = "0.1.0"
TRANSCRIPT_LEGEND_VERSION = "unassigned"

# Word boundaries are compared against a duration derived from an integer frame
# count, so allow a sample-scale tolerance rather than demanding exact equality.
_BOUNDS_TOLERANCE_S = 1.0 / TARGET_SAMPLE_RATE


class TranscriptionError(ValueError):
    """Raised when transcription cannot produce a valid word-timestamp IR."""


@dataclass(frozen=True, slots=True)
class TranscribedWord:
    """One word from an adapter, before it becomes IR.

    `alignment_confidence` is separate from `confidence` because an adapter that
    aligns independently of recognition can report both. When it is None the
    transcript records that alignment confidence was inherited from ASR rather
    than measured, so downstream suppression is not misled about its provenance.
    """

    text: str
    start: float
    end: float
    confidence: float
    space_before: bool = True
    alignment_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """An adapter's output, in terms this package owns."""

    words: list[TranscribedWord]
    model_name: str
    model_version: str
    language: str = "en"
    options: dict[str, str] = field(default_factory=dict)


class TranscriptionAdapter(Protocol):
    """The transcription boundary.

    Deliberately states nothing about any particular engine: swapping the
    implementation must not reach into other pipeline stages.
    """

    def transcribe(self, path: Path) -> TranscriptionResult: ...


def _validate_words(words: list[TranscribedWord], duration_s: float) -> None:
    if not words:
        raise TranscriptionError("transcription produced no words")

    previous_start = 0.0
    previous_end = 0.0
    for position, word in enumerate(words):
        label = f"word {position} ({word.text!r})"
        if not word.text.strip():
            raise TranscriptionError(f"{label} has no text")
        for name, value in (
            ("confidence", word.confidence),
            ("alignment_confidence", word.alignment_confidence),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise TranscriptionError(f"{label} {name} must be between 0 and 1")
        if word.start < 0.0:
            raise TranscriptionError(f"{label} starts before the recording")
        if word.end < word.start:
            raise TranscriptionError(f"{label} ends before it starts")
        if word.end > duration_s + _BOUNDS_TOLERANCE_S:
            raise TranscriptionError(
                f"{label} ends at {word.end:.3f}s, past the {duration_s:.3f}s recording"
            )
        if word.start < previous_start or word.end < previous_end:
            raise TranscriptionError(f"{label} breaks monotonic word timing")
        previous_start, previous_end = word.start, word.end


def build_transcript(
    path: Path,
    adapter: TranscriptionAdapter,
    *,
    speaker: str,
    turn_id: str = "turn-1",
) -> Document:
    """Transcribe canonical audio into word-timestamp IR.

    The audio must already be normalized, so transcription never silently
    resamples and lose provenance. Low-confidence words are kept: suppression is
    the legend's decision at assignment, not the adapter's here.

    Baseline metadata is left pending because per-speaker normalization is a
    later stage; the document is therefore valid word-timestamp IR but not yet
    ready for assignment.
    """
    if not speaker.strip():
        raise TranscriptionError("speaker identity is required for single-speaker v0.1 audio")

    inspection = inspect_wav(path)
    if inspection.conversion_required:
        raise TranscriptionError(
            "audio must be canonical mono 16 kHz 16-bit PCM; "
            "run 'prosody-markup audio normalize' first"
        )

    result = adapter.transcribe(Path(inspection.source_path))
    _validate_words(result.words, inspection.duration_s)

    tokens: list[Token] = []
    for position, word in enumerate(result.words, start=1):
        alignment_confidence = (
            word.confidence if word.alignment_confidence is None else word.alignment_confidence
        )
        tokens.append(
            Token(
                id=f"w{position:04d}",
                text=word.text.strip(),
                start=word.start,
                end=word.end,
                speaker=speaker,
                asr_confidence=word.confidence,
                alignment_confidence=alignment_confidence,
                turn_id=turn_id,
                # A single known speaker per file is a fixed v0.1 precondition,
                # not a diarization result.
                speaker_confidence=1.0,
                space_before=word.space_before,
            )
        )
    validate_tokens(tokens)

    alignment_source = (
        "asr-inherited"
        if all(word.alignment_confidence is None for word in result.words)
        else "adapter-reported"
    )
    return Document(
        schema_version=SCHEMA_VERSION,
        legend_version=TRANSCRIPT_LEGEND_VERSION,
        audio={
            "path": inspection.source_path,
            "sha256": inspection.source_sha256,
            "duration_s": inspection.duration_s,
            "sample_rate": inspection.sample_rate,
            "channels": inspection.channels,
        },
        provenance={
            "kind": "local-asr",
            "asr_model": result.model_name,
            "asr_model_version": result.model_version,
            "asr_options": dict(result.options),
            "language": result.language,
            "alignment": alignment_source,
            "speaker_source": "declared",
        },
        baseline={"speaker": speaker, "status": "pending-normalization"},
        tokens=tokens,
    )


class FasterWhisperAdapter:
    """Local `faster-whisper` transcription, kept behind the adapter boundary.

    The import is deferred so the package, its tests, and CI never require the
    dependency or a model download.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

    def transcribe(self, path: Path) -> TranscriptionResult:
        try:
            import faster_whisper
        except ImportError as exc:  # pragma: no cover - exercised without the extra installed
            raise TranscriptionError(
                "faster-whisper is not installed; install it locally to transcribe real audio"
            ) from exc

        model = faster_whisper.WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )
        segments, _info = model.transcribe(str(path), language=self.language, word_timestamps=True)

        words: list[TranscribedWord] = []
        for segment in segments:
            for word in segment.words or ():
                text = str(word.word)
                words.append(
                    TranscribedWord(
                        text=text,
                        start=float(word.start),
                        end=float(word.end),
                        # Clamp: probabilities outside [0, 1] would fail the IR
                        # boundary, and a model quirk should not look like corrupt audio.
                        confidence=min(1.0, max(0.0, float(word.probability))),
                        space_before=text.startswith(" "),
                    )
                )

        return TranscriptionResult(
            words=words,
            model_name=f"faster-whisper/{self.model_size}",
            model_version=str(getattr(faster_whisper, "__version__", "unknown")),
            language=self.language,
            options={"device": self.device, "compute_type": self.compute_type},
        )
