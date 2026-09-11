from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from goldminer.errors import GoldMinerError
from goldminer.providers.transcription import TranscriptionProvider

from .models import CanonicalTranscript, TRANSCRIPT_SCHEMA_VERSION, validate_provider_utterances
from .parser import parse_transcript_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GoldMinerError(f"Could not fingerprint {path}: {exc}") from exc
    return digest.hexdigest()


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_transcript(path: Path) -> CanonicalTranscript:
    try:
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldMinerError(f"Could not load canonical transcript {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise GoldMinerError("Canonical transcript root must be a JSON object")
    return CanonicalTranscript.from_dict(raw)


def build_transcript(
    destination: Path,
    *,
    media_duration_ms: int,
    transcript_path: Path | None = None,
    provider: TranscriptionProvider | None = None,
    audio_path: Path | None = None,
    resume: bool = False,
) -> tuple[CanonicalTranscript, bool]:
    if transcript_path is not None:
        source_type, utterances = parse_transcript_file(transcript_path)
        source_sha256 = sha256_file(transcript_path)
        source_path: str | None = str(transcript_path)
    elif provider is not None and audio_path is not None:
        utterances = validate_provider_utterances(provider.transcribe(audio_path, media_duration_ms), media_duration_ms)
        source_type = "provider"
        source_sha256 = sha256_file(audio_path)
        source_path = None
    else:
        raise GoldMinerError(
            "No transcript was supplied and no transcription provider is configured. "
            "Pass --transcript PATH using an SRT or VTT file."
        )

    if resume and destination.is_file():
        cached = load_transcript(destination)
        if cached.source_sha256 == source_sha256 and cached.media_duration_ms == media_duration_ms:
            return cached, True

    transcript = CanonicalTranscript(
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        source_type=source_type,
        source_path=source_path,
        source_sha256=source_sha256,
        media_duration_ms=media_duration_ms,
        utterances=tuple(utterances),
    )
    transcript.validate()
    _atomic_json_write(destination, transcript.to_dict())
    reloaded = load_transcript(destination)
    return reloaded, False
