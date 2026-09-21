from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CORPUS_SCHEMA_VERSION = "corpus-manifest@0.1.0"

# v0.1 targets from the real-audio plan, enforced once a corpus declares itself
# complete rather than while it is still being assembled.
MIN_SPEAKERS = 5
MAX_SPEAKERS = 10
MIN_CLIPS_PER_SPEAKER = 3
MAX_CLIPS_PER_SPEAKER = 5
MAX_CLIP_SECONDS = 30.0

# How a clip may be usable at all. Anything outside this set is unclear rights,
# which is the case this list exists to refuse.
RIGHTS_BASES = frozenset({"license", "contribution-agreement", "public-domain"})

# Licenses compatible with redistributing the corpus alongside this project.
ALLOWED_LICENSES = frozenset(
    {
        "CC0-1.0",
        "CC-BY-4.0",
        "CC-BY-SA-4.0",
        "CC-BY-3.0",
        "public-domain",
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CorpusError(ValueError):
    """Raised when a corpus manifest is malformed or its provenance is incomplete."""


@dataclass(frozen=True, slots=True)
class Speaker:
    id: str
    accent: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class Clip:
    id: str
    speaker: str
    path: str
    text: str
    rights_basis: str
    sha256: str
    source: str
    duration_s: float | None = None
    license: str | None = None
    license_url: str | None = None
    consent: dict[str, Any] = field(default_factory=dict)
    recording: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    schema_version: str
    name: str
    status: str
    purpose: str
    speakers: list[Speaker]
    clips: list[Clip]

    @property
    def clips_by_speaker(self) -> dict[str, list[Clip]]:
        grouped: dict[str, list[Clip]] = {speaker.id: [] for speaker in self.speakers}
        for clip in self.clips:
            grouped.setdefault(clip.speaker, []).append(clip)
        return grouped


def load_manifest(path: Path) -> CorpusManifest:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusError(f"cannot read corpus manifest: {path}") from exc
    except yaml.YAMLError as exc:
        raise CorpusError(f"corpus manifest is not valid YAML: {path}") from exc

    if not isinstance(raw, dict):
        raise CorpusError("corpus manifest must be a mapping")

    schema_version = str(raw.get("schema_version", ""))
    if schema_version != CORPUS_SCHEMA_VERSION:
        raise CorpusError(
            f"unsupported corpus schema: {schema_version!r}, expected {CORPUS_SCHEMA_VERSION!r}"
        )
    status = str(raw.get("status", "")).strip()
    if status not in {"draft", "complete"}:
        raise CorpusError(f"corpus status must be 'draft' or 'complete', got {status!r}")

    speakers = [Speaker(**entry) for entry in raw.get("speakers") or []]
    clips = [Clip(**entry) for entry in raw.get("clips") or []]
    return CorpusManifest(
        schema_version=schema_version,
        name=str(raw.get("name", "")),
        status=status,
        purpose=str(raw.get("purpose", "")),
        speakers=speakers,
        clips=clips,
    )


def _check_rights(clip: Clip) -> list[str]:
    """Every clip must name why it may be used, with the evidence for it."""
    problems: list[str] = []
    if clip.rights_basis not in RIGHTS_BASES:
        problems.append(
            f"{clip.id}: rights_basis must be one of {sorted(RIGHTS_BASES)}, "
            f"got {clip.rights_basis!r}"
        )
        return problems

    if not clip.source.strip():
        problems.append(f"{clip.id}: source is required so the recording's origin is auditable")

    if clip.rights_basis == "license":
        if not clip.license:
            problems.append(f"{clip.id}: rights_basis 'license' requires a license")
        elif clip.license not in ALLOWED_LICENSES:
            problems.append(
                f"{clip.id}: license {clip.license!r} is not in the compatible set "
                f"{sorted(ALLOWED_LICENSES)}"
            )
    elif clip.rights_basis == "contribution-agreement":
        consent = clip.consent or {}
        if consent.get("obtained") is not True:
            problems.append(f"{clip.id}: contributed audio requires consent.obtained: true")
        if not str(consent.get("agreement_ref") or "").strip():
            problems.append(
                f"{clip.id}: contributed audio requires consent.agreement_ref naming the "
                "stored agreement"
            )
        if not str(consent.get("date") or "").strip():
            problems.append(f"{clip.id}: contributed audio requires consent.date")
    return problems


def _check_path(clip: Clip) -> list[str]:
    problems: list[str] = []
    candidate = Path(clip.path)
    if not clip.path.strip():
        problems.append(f"{clip.id}: path is required")
    elif candidate.is_absolute() or ".." in candidate.parts:
        problems.append(f"{clip.id}: path must stay inside the corpus, got {clip.path!r}")
    return problems


def validate_manifest(
    manifest: CorpusManifest,
    audio_root: Path | None = None,
    *,
    require_audio: bool = False,
) -> list[str]:
    """Check structure and provenance, and report every problem found.

    Audio itself is optional so this runs in CI against the manifest alone.
    When a clip's audio is present its checksum is verified, so provenance can
    be trusted without the recordings ever entering the repository.
    """
    problems: list[str] = []
    if not manifest.purpose.strip():
        problems.append("purpose is required and must state what this corpus is evidence of")

    speaker_ids = {speaker.id for speaker in manifest.speakers}
    if len(speaker_ids) != len(manifest.speakers):
        problems.append("speaker ids must be unique")

    seen: set[str] = set()
    for clip in manifest.clips:
        if clip.id in seen:
            problems.append(f"{clip.id}: duplicate clip id")
        seen.add(clip.id)

        if clip.speaker not in speaker_ids:
            problems.append(f"{clip.id}: unknown speaker {clip.speaker!r}")
        if not clip.text.strip():
            problems.append(f"{clip.id}: text is required")
        if not _SHA256.match(clip.sha256 or ""):
            problems.append(f"{clip.id}: sha256 must be 64 lowercase hex characters")
        if clip.duration_s is not None and not 0.0 < clip.duration_s <= MAX_CLIP_SECONDS:
            problems.append(
                f"{clip.id}: duration_s must be above 0 and at most {MAX_CLIP_SECONDS:.0f}s"
            )

        problems.extend(_check_rights(clip))
        problems.extend(_check_path(clip))

        if audio_root is not None:
            audio_path = audio_root / clip.path
            if audio_path.exists():
                digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
                if digest != clip.sha256:
                    problems.append(
                        f"{clip.id}: audio checksum mismatch; manifest says {clip.sha256[:12]}…, "
                        f"file is {digest[:12]}…"
                    )
            elif require_audio:
                problems.append(f"{clip.id}: audio missing at {audio_path}")

    if manifest.status == "complete":
        problems.extend(check_readiness(manifest))
    return problems


def check_readiness(manifest: CorpusManifest) -> list[str]:
    """Check the v0.1 size targets, which only bind once a corpus claims to be complete."""
    problems: list[str] = []
    speaker_count = len(manifest.speakers)
    if not MIN_SPEAKERS <= speaker_count <= MAX_SPEAKERS:
        problems.append(
            f"a complete corpus needs {MIN_SPEAKERS}-{MAX_SPEAKERS} speakers, got {speaker_count}"
        )
    for speaker_id, clips in manifest.clips_by_speaker.items():
        if not MIN_CLIPS_PER_SPEAKER <= len(clips) <= MAX_CLIPS_PER_SPEAKER:
            problems.append(
                f"{speaker_id}: a complete corpus needs {MIN_CLIPS_PER_SPEAKER}-"
                f"{MAX_CLIPS_PER_SPEAKER} clips per speaker, got {len(clips)}"
            )
    return problems


def summarize(manifest: CorpusManifest) -> dict[str, Any]:
    grouped = manifest.clips_by_speaker
    return {
        "schema_version": manifest.schema_version,
        "name": manifest.name,
        "status": manifest.status,
        "speakers": len(manifest.speakers),
        "clips": len(manifest.clips),
        "clips_per_speaker": {speaker: len(clips) for speaker, clips in sorted(grouped.items())},
        "rights_bases": sorted({clip.rights_basis for clip in manifest.clips}),
    }


# Kept with the writer so an ingested manifest never loses the reason it is
# strict. It stays true whether the corpus is empty or full.
_MANIFEST_HEADER = """# v0.1 engineering corpus.
#
# Entries are added only as real recordings arrive with their rights basis
# established. No placeholder speaker, consent, or checksum record belongs
# here: a fabricated provenance record is worse than an absent one, because
# it reads as evidence.
#
# Add a clip with 'prosody-markup corpus add'; it computes the checksum and
# refuses anything the rules in corpus/README.md would reject.
"""


def dump_manifest(manifest: CorpusManifest) -> str:
    """Serialize a manifest, keeping the header that explains its rules."""
    payload = {
        "schema_version": manifest.schema_version,
        "name": manifest.name,
        "status": manifest.status,
        "purpose": manifest.purpose,
        "speakers": [
            {key: value for key, value in asdict(speaker).items() if value is not None}
            for speaker in manifest.speakers
        ],
        "clips": [
            {key: value for key, value in asdict(clip).items() if value not in (None, {}, "")}
            for clip in manifest.clips
        ],
    }
    body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    return _MANIFEST_HEADER + body


def next_clip_id(manifest: CorpusManifest) -> str:
    used = {clip.id for clip in manifest.clips}
    index = 1
    while f"clip-{index:04d}" in used:
        index += 1
    return f"clip-{index:04d}"


def add_clip(
    manifest: CorpusManifest,
    clip: Clip,
    *,
    accent: str | None = None,
    audio_root: Path | None = None,
) -> CorpusManifest:
    """Return a manifest with one more clip, or raise if it would not be valid.

    The whole manifest is re-validated rather than just the new entry, so a
    clip can never be added into a state the validator would reject.
    """
    if any(existing.id == clip.id for existing in manifest.clips):
        raise CorpusError(f"{clip.id}: clip id is already in the manifest")
    duplicate = next(
        (existing.id for existing in manifest.clips if existing.sha256 == clip.sha256), None
    )
    if duplicate:
        raise CorpusError(
            f"{clip.id}: identical audio is already recorded as {duplicate}; "
            "the same recording must not be counted twice"
        )

    speakers = list(manifest.speakers)
    if not any(speaker.id == clip.speaker for speaker in speakers):
        speakers.append(Speaker(id=clip.speaker, accent=accent))

    updated = CorpusManifest(
        schema_version=manifest.schema_version,
        name=manifest.name,
        status=manifest.status,
        purpose=manifest.purpose,
        speakers=speakers,
        clips=[*manifest.clips, clip],
    )
    problems = validate_manifest(updated, audio_root)
    if problems:
        raise CorpusError("; ".join(problems))
    return updated
