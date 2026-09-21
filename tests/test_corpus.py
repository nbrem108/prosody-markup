from __future__ import annotations

import hashlib
import math
import struct
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from prosody_markup.corpus import (
    CORPUS_SCHEMA_VERSION,
    Clip,
    CorpusError,
    add_clip,
    check_readiness,
    dump_manifest,
    load_manifest,
    next_clip_id,
    summarize,
    validate_manifest,
)

REPO_MANIFEST = Path(__file__).parents[1] / "corpus" / "manifest.yaml"

_CONSENTED_CLIP: dict[str, Any] = {
    "id": "clip-0001",
    "speaker": "spk-01",
    "path": "audio/spk-01/clip-0001.wav",
    "text": "I said the blue one.",
    "rights_basis": "contribution-agreement",
    "source": "recorded for this project",
    "sha256": "a" * 64,
    "duration_s": 3.2,
    "consent": {
        "obtained": True,
        "date": "2026-01-01",
        "agreement_ref": "agreements/spk-01",
    },
}


def _manifest(
    clips: list[dict[str, Any]] | None = None,
    speakers: list[dict[str, Any]] | None = None,
    status: str = "draft",
) -> dict[str, Any]:
    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "name": "test corpus",
        "status": status,
        "purpose": "Engineering evidence only.",
        "speakers": speakers if speakers is not None else [{"id": "spk-01"}],
        "clips": clips if clips is not None else [dict(_CONSENTED_CLIP)],
    }


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _clip(**overrides: Any) -> dict[str, Any]:
    clip = dict(_CONSENTED_CLIP)
    clip.update(overrides)
    return clip


def test_repository_manifest_is_valid_and_empty() -> None:
    """The committed manifest holds no fabricated provenance records."""
    manifest = load_manifest(REPO_MANIFEST)

    assert manifest.status == "draft"
    assert manifest.clips == []
    assert manifest.speakers == []
    assert manifest.purpose
    assert validate_manifest(manifest) == []


def test_a_consented_clip_validates(tmp_path: Path) -> None:
    manifest = load_manifest(_write(tmp_path / "manifest.yaml", _manifest()))

    assert validate_manifest(manifest) == []
    assert summarize(manifest)["rights_bases"] == ["contribution-agreement"]


def test_contributed_audio_requires_recorded_consent(tmp_path: Path) -> None:
    clip = _clip(consent={"obtained": False, "date": "2026-01-01", "agreement_ref": "x"})
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    problems = validate_manifest(manifest)

    assert any("consent.obtained: true" in problem for problem in problems)


def test_contributed_audio_requires_an_agreement_reference(tmp_path: Path) -> None:
    clip = _clip(consent={"obtained": True, "date": "2026-01-01"})
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    problems = validate_manifest(manifest)

    assert any("agreement_ref" in problem for problem in problems)


def test_unclear_rights_are_refused(tmp_path: Path) -> None:
    """There is no fourth rights basis, which is how scraped audio is kept out."""
    clip = _clip(rights_basis="found-online", consent={})
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    problems = validate_manifest(manifest)

    assert any("rights_basis must be one of" in problem for problem in problems)


def test_incompatible_license_is_refused(tmp_path: Path) -> None:
    clip = _clip(rights_basis="license", license="All rights reserved", consent={})
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    problems = validate_manifest(manifest)

    assert any("not in the compatible set" in problem for problem in problems)


def test_licensed_clip_validates(tmp_path: Path) -> None:
    clip = _clip(rights_basis="license", license="CC0-1.0", source="archive item 42", consent={})
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    assert validate_manifest(manifest) == []


def test_source_is_always_required(tmp_path: Path) -> None:
    clip = _clip(source="  ")
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    problems = validate_manifest(manifest)

    assert any("source is required" in problem for problem in problems)


def test_clip_must_reference_a_declared_speaker(tmp_path: Path) -> None:
    clip = _clip(speaker="spk-99")
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    problems = validate_manifest(manifest)

    assert any("unknown speaker" in problem for problem in problems)


def test_duplicate_clip_ids_are_refused(tmp_path: Path) -> None:
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([_clip(), _clip()])))

    problems = validate_manifest(manifest)

    assert any("duplicate clip id" in problem for problem in problems)


def test_paths_may_not_escape_the_corpus(tmp_path: Path) -> None:
    clip = _clip(path="../../etc/passwd")
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    problems = validate_manifest(manifest)

    assert any("must stay inside the corpus" in problem for problem in problems)


def test_checksum_must_look_like_a_digest(tmp_path: Path) -> None:
    clip = _clip(sha256="nope")
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    problems = validate_manifest(manifest)

    assert any("64 lowercase hex" in problem for problem in problems)


def test_overlong_clips_are_refused(tmp_path: Path) -> None:
    clip = _clip(duration_s=120.0)
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest([clip])))

    problems = validate_manifest(manifest)

    assert any("duration_s" in problem for problem in problems)


def test_validation_runs_without_the_audio(tmp_path: Path) -> None:
    """CI must be able to check provenance without the recordings present."""
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest()))

    assert validate_manifest(manifest, audio_root=tmp_path) == []


def test_missing_audio_is_reported_when_required(tmp_path: Path) -> None:
    manifest = load_manifest(_write(tmp_path / "m.yaml", _manifest()))

    problems = validate_manifest(manifest, audio_root=tmp_path, require_audio=True)

    assert any("audio missing" in problem for problem in problems)


def test_present_audio_is_checksummed(tmp_path: Path) -> None:
    payload = b"RIFF-not-really-audio"
    audio = tmp_path / "audio" / "spk-01" / "clip-0001.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(payload)

    good = _clip(sha256=hashlib.sha256(payload).hexdigest())
    manifest = load_manifest(_write(tmp_path / "good.yaml", _manifest([good])))
    assert validate_manifest(manifest, audio_root=tmp_path, require_audio=True) == []

    stale = load_manifest(_write(tmp_path / "stale.yaml", _manifest([_clip()])))
    problems = validate_manifest(stale, audio_root=tmp_path, require_audio=True)
    assert any("checksum mismatch" in problem for problem in problems)


def test_size_targets_bind_only_when_complete(tmp_path: Path) -> None:
    """A corpus is assembled incrementally, so targets apply at completion."""
    draft = load_manifest(_write(tmp_path / "draft.yaml", _manifest()))
    assert validate_manifest(draft) == []

    complete = load_manifest(_write(tmp_path / "done.yaml", _manifest(status="complete")))
    problems = validate_manifest(complete)
    assert any("5-10 speakers" in problem for problem in problems)
    assert any("3-5 clips per speaker" in problem for problem in problems)


def test_a_complete_corpus_meeting_the_targets_passes(tmp_path: Path) -> None:
    speakers = [{"id": f"spk-{index:02d}"} for index in range(1, 6)]
    clips = [
        _clip(
            id=f"clip-{index:02d}{number}",
            speaker=f"spk-{index:02d}",
            path=f"audio/spk-{index:02d}/clip-{number}.wav",
        )
        for index in range(1, 6)
        for number in range(3)
    ]
    manifest = load_manifest(
        _write(tmp_path / "m.yaml", _manifest(clips, speakers, status="complete"))
    )

    assert check_readiness(manifest) == []
    assert validate_manifest(manifest) == []


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    payload = _manifest()
    payload["schema_version"] = "corpus-manifest@9.9.9"

    with pytest.raises(CorpusError, match="unsupported corpus schema"):
        load_manifest(_write(tmp_path / "m.yaml", payload))


def test_unknown_status_is_refused(tmp_path: Path) -> None:
    payload = _manifest()
    payload["status"] = "probably-fine"

    with pytest.raises(CorpusError, match="status must be"):
        load_manifest(_write(tmp_path / "m.yaml", payload))


def test_purpose_is_required(tmp_path: Path) -> None:
    """The corpus must state what it is evidence of, and what it is not."""
    payload = _manifest()
    payload["purpose"] = ""
    manifest = load_manifest(_write(tmp_path / "m.yaml", payload))

    problems = validate_manifest(manifest)

    assert any("purpose is required" in problem for problem in problems)


def test_corpus_cli_reports_problems_with_a_failing_exit_code(tmp_path: Path) -> None:
    path = _write(tmp_path / "m.yaml", _manifest([_clip(rights_basis="found-online")]))

    completed = subprocess.run(
        [sys.executable, "-m", "prosody_markup.cli", "corpus", "validate", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "rights_basis must be one of" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_corpus_cli_accepts_the_repository_manifest() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "prosody_markup.cli", "corpus", "validate", str(REPO_MANIFEST)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "corpus provenance is complete" in completed.stderr


def _write_audio(path: Path, *, frequency: float = 150.0, duration_s: float = 1.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = b"".join(
        struct.pack("<h", int(math.sin(2 * math.pi * frequency * index / 16_000) * 0.4 * 32767))
        for index in range(int(16_000 * duration_s))
    )
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16_000)
        target.writeframes(frames)
    return path


def _empty_manifest(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.yaml"
    _write(path, _manifest(clips=[], speakers=[]))
    return path


def _consented(audio: Path, speaker: str = "spk-01", clip_id: str = "clip-0001") -> Clip:
    return Clip(
        id=clip_id,
        speaker=speaker,
        path=str(audio.relative_to(audio.parents[2])),
        text="I said the blue one.",
        rights_basis="contribution-agreement",
        sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
        source="recorded for this project",
        duration_s=1.5,
        consent={"obtained": True, "date": "2026-01-01", "agreement_ref": "agreements/spk-01"},
    )


def test_add_clip_creates_the_speaker_and_validates(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    audio = _write_audio(root / "audio" / "spk-01" / "clip-0001.wav")
    manifest = load_manifest(_empty_manifest(root))

    updated = add_clip(manifest, _consented(audio), accent="General American", audio_root=root)

    assert [speaker.id for speaker in updated.speakers] == ["spk-01"]
    assert updated.speakers[0].accent == "General American"
    assert [clip.id for clip in updated.clips] == ["clip-0001"]
    assert validate_manifest(updated, audio_root=root, require_audio=True) == []


def test_add_clip_refuses_the_same_recording_twice(tmp_path: Path) -> None:
    """Two names for one recording would inflate the corpus without adding evidence."""
    root = tmp_path / "corpus"
    audio = _write_audio(root / "audio" / "spk-01" / "clip-0001.wav")
    manifest = add_clip(load_manifest(_empty_manifest(root)), _consented(audio), audio_root=root)

    duplicate = _consented(audio, speaker="spk-02", clip_id="clip-0002")

    with pytest.raises(CorpusError, match="identical audio is already recorded"):
        add_clip(manifest, duplicate, audio_root=root)


def test_add_clip_refuses_a_duplicate_id(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    first = _write_audio(root / "audio" / "spk-01" / "clip-0001.wav")
    second = _write_audio(root / "audio" / "spk-01" / "clip-0002.wav", frequency=200.0)
    manifest = add_clip(load_manifest(_empty_manifest(root)), _consented(first), audio_root=root)

    clash = _consented(second, clip_id="clip-0001")

    with pytest.raises(CorpusError, match="clip id is already in the manifest"):
        add_clip(manifest, clash, audio_root=root)


def test_add_clip_refuses_an_entry_the_validator_would_reject(tmp_path: Path) -> None:
    """A clip can never be added into a state validation would fail."""
    root = tmp_path / "corpus"
    audio = _write_audio(root / "audio" / "spk-01" / "clip-0001.wav")
    manifest = load_manifest(_empty_manifest(root))
    unconsented = replace(_consented(audio), consent={})

    with pytest.raises(CorpusError, match="consent.obtained"):
        add_clip(manifest, unconsented, audio_root=root)


def test_dump_manifest_round_trips_and_keeps_the_header(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    audio = _write_audio(root / "audio" / "spk-01" / "clip-0001.wav")
    manifest = add_clip(load_manifest(_empty_manifest(root)), _consented(audio), audio_root=root)

    body = dump_manifest(manifest)
    path = root / "written.yaml"
    path.write_text(body, encoding="utf-8")

    assert body.startswith("# v0.1 engineering corpus.")
    assert "fabricated provenance record" in body
    reloaded = load_manifest(path)
    assert reloaded.clips == manifest.clips
    assert reloaded.speakers == manifest.speakers


def test_next_clip_id_skips_used_ids(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    audio = _write_audio(root / "audio" / "spk-01" / "clip-0001.wav")
    manifest = add_clip(load_manifest(_empty_manifest(root)), _consented(audio), audio_root=root)

    assert next_clip_id(manifest) == "clip-0002"


def test_corpus_add_cli_computes_the_checksum_from_the_audio(tmp_path: Path) -> None:
    """The checksum is measured, never typed, so it cannot silently disagree."""
    root = tmp_path / "corpus"
    audio = _write_audio(root / "audio" / "spk-01" / "clip-0001.wav")
    manifest_path = _empty_manifest(root)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "corpus",
            "add",
            str(manifest_path),
            "--audio",
            str(audio),
            "--speaker",
            "spk-01",
            "--text",
            "I said the blue one.",
            "--rights-basis",
            "contribution-agreement",
            "--source",
            "recorded for this project",
            "--consent-date",
            "2026-01-01",
            "--agreement-ref",
            "agreements/spk-01",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    (clip,) = load_manifest(manifest_path).clips
    assert clip.sha256 == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert clip.duration_s == pytest.approx(1.5)
    assert clip.path == "audio/spk-01/clip-0001.wav"


def test_corpus_add_cli_refuses_contributed_audio_without_consent(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    audio = _write_audio(root / "audio" / "spk-01" / "clip-0001.wav")
    manifest_path = _empty_manifest(root)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "corpus",
            "add",
            str(manifest_path),
            "--audio",
            str(audio),
            "--speaker",
            "spk-01",
            "--text",
            "Words.",
            "--rights-basis",
            "contribution-agreement",
            "--source",
            "recorded",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--consent-date and --agreement-ref" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert load_manifest(manifest_path).clips == []


def test_corpus_add_cli_refuses_audio_outside_the_corpus(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _empty_manifest(root)
    outside = _write_audio(tmp_path / "elsewhere" / "clip.wav")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "corpus",
            "add",
            str(root / "manifest.yaml"),
            "--audio",
            str(outside),
            "--speaker",
            "spk-01",
            "--text",
            "Words.",
            "--rights-basis",
            "public-domain",
            "--source",
            "archive",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "must live under the corpus root" in completed.stderr


def test_corpus_add_cli_refuses_a_file_that_is_not_audio(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    _empty_manifest(root)
    fake = root / "audio" / "spk-01" / "clip.wav"
    fake.parent.mkdir(parents=True)
    fake.write_text("not audio", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "prosody_markup.cli",
            "corpus",
            "add",
            str(root / "manifest.yaml"),
            "--audio",
            str(fake),
            "--speaker",
            "spk-01",
            "--text",
            "Words.",
            "--rights-basis",
            "public-domain",
            "--source",
            "archive",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "valid PCM WAV" in completed.stderr
