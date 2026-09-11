from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from goldminer.errors import GoldMinerError

TRANSCRIPT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class Utterance:
    id: str
    start_ms: int
    end_ms: int
    speaker: str | None
    original_text: str
    normalized_text: str

    def validate(self, *, media_duration_ms: int | None = None) -> None:
        if not self.id:
            raise GoldMinerError("Utterance id cannot be empty")
        if self.start_ms < 0:
            raise GoldMinerError(f"Utterance {self.id} has a negative start timestamp")
        if self.end_ms <= self.start_ms:
            raise GoldMinerError(f"Utterance {self.id} must have start_ms < end_ms")
        if media_duration_ms is not None and self.end_ms > media_duration_ms:
            raise GoldMinerError(
                f"Utterance {self.id} ends at {self.end_ms} ms, beyond media duration "
                f"{media_duration_ms} ms"
            )
        if not self.original_text.strip():
            raise GoldMinerError(f"Utterance {self.id} has empty original_text")
        if not self.normalized_text.strip():
            raise GoldMinerError(f"Utterance {self.id} has empty normalized_text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "speaker": self.speaker,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Utterance":
        required = {"id", "start_ms", "end_ms", "speaker", "original_text", "normalized_text"}
        missing = required.difference(value)
        if missing:
            raise GoldMinerError(f"Transcript utterance is missing fields: {', '.join(sorted(missing))}")
        if not isinstance(value["start_ms"], int) or isinstance(value["start_ms"], bool):
            raise GoldMinerError("Utterance start_ms must be an integer")
        if not isinstance(value["end_ms"], int) or isinstance(value["end_ms"], bool):
            raise GoldMinerError("Utterance end_ms must be an integer")
        speaker = value["speaker"]
        if speaker is not None and not isinstance(speaker, str):
            raise GoldMinerError("Utterance speaker must be a string or null")
        for field in ("id", "original_text", "normalized_text"):
            if not isinstance(value[field], str):
                raise GoldMinerError(f"Utterance {field} must be a string")
        return cls(
            id=value["id"],
            start_ms=value["start_ms"],
            end_ms=value["end_ms"],
            speaker=speaker,
            original_text=value["original_text"],
            normalized_text=value["normalized_text"],
        )


@dataclass(frozen=True, slots=True)
class CanonicalTranscript:
    schema_version: str
    source_type: str
    source_path: str | None
    source_sha256: str
    media_duration_ms: int
    utterances: tuple[Utterance, ...]

    def validate(self) -> None:
        if self.schema_version != TRANSCRIPT_SCHEMA_VERSION:
            raise GoldMinerError(f"Unsupported transcript schema version: {self.schema_version}")
        if self.source_type not in {"srt", "vtt", "provider"}:
            raise GoldMinerError(f"Unsupported transcript source type: {self.source_type}")
        if not self.source_sha256:
            raise GoldMinerError("Transcript source_sha256 cannot be empty")
        if self.media_duration_ms <= 0:
            raise GoldMinerError("Transcript media_duration_ms must be positive")
        if not self.utterances:
            raise GoldMinerError("Transcript contains no utterances")

        ids: set[str] = set()
        previous_start = -1
        for utterance in self.utterances:
            utterance.validate(media_duration_ms=self.media_duration_ms)
            if utterance.id in ids:
                raise GoldMinerError(f"Duplicate utterance id: {utterance.id}")
            if utterance.start_ms < previous_start:
                raise GoldMinerError("Transcript utterances must be ordered by start timestamp")
            ids.add(utterance.id)
            previous_start = utterance.start_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": {
                "type": self.source_type,
                "path": self.source_path,
                "sha256": self.source_sha256,
            },
            "media_duration_ms": self.media_duration_ms,
            "utterance_count": len(self.utterances),
            "utterances": [utterance.to_dict() for utterance in self.utterances],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalTranscript":
        try:
            source = value["source"]
            raw_utterances = value["utterances"]
            transcript = cls(
                schema_version=value["schema_version"],
                source_type=source["type"],
                source_path=source.get("path"),
                source_sha256=source["sha256"],
                media_duration_ms=value["media_duration_ms"],
                utterances=tuple(Utterance.from_dict(item) for item in raw_utterances),
            )
        except (KeyError, TypeError) as exc:
            raise GoldMinerError(f"Invalid transcript schema: missing or malformed {exc}") from exc
        if value.get("utterance_count") != len(transcript.utterances):
            raise GoldMinerError("Transcript utterance_count does not match utterances")
        transcript.validate()
        return transcript


def validate_provider_utterances(values: Sequence[Utterance], media_duration_ms: int) -> tuple[Utterance, ...]:
    utterances = tuple(values)
    for utterance in utterances:
        utterance.validate(media_duration_ms=media_duration_ms)
    return utterances
