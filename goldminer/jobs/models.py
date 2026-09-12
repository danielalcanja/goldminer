from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from goldminer.errors import GoldMinerError

JOB_SCHEMA_VERSION = "1.0"
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")


def validate_object_key(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GoldMinerError(f"{field} must be a non-empty string")
    key = value.strip()
    path = PurePosixPath(key)
    if (
        key.startswith("/")
        or "\\" in key
        or "\x00" in key
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise GoldMinerError(f"{field} must be a safe relative R2 object key")
    return key


def _bounded_integer(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GoldMinerError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise GoldMinerError(f"{field} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class JobSpec:
    job_id: str
    source_key: str
    result_prefix: str
    transcript_key: str | None = None
    top_k: int = 8
    max_duration_seconds: int = 60
    handle_ms: int = 500
    schema_version: str = JOB_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "JobSpec":
        schema_version = value.get("schema_version", JOB_SCHEMA_VERSION)
        if schema_version != JOB_SCHEMA_VERSION:
            raise GoldMinerError(
                f"Unsupported job schema_version {schema_version!r}; expected {JOB_SCHEMA_VERSION!r}"
            )

        job_id = value.get("job_id")
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise GoldMinerError(
                "job_id must be 1-100 characters using letters, numbers, underscores, or hyphens"
            )

        source_key = validate_object_key(value.get("source_key"), "source_key")
        result_prefix = validate_object_key(
            value.get("result_prefix", f"jobs/{job_id}"), "result_prefix"
        ).rstrip("/")

        transcript_value = value.get("transcript_key")
        transcript_key = (
            None
            if transcript_value is None
            else validate_object_key(transcript_value, "transcript_key")
        )

        return cls(
            job_id=job_id,
            source_key=source_key,
            result_prefix=result_prefix,
            transcript_key=transcript_key,
            top_k=_bounded_integer(value.get("top_k", 8), "top_k", minimum=1, maximum=25),
            max_duration_seconds=_bounded_integer(
                value.get("max_duration_seconds", 60),
                "max_duration_seconds",
                minimum=15,
                maximum=60,
            ),
            handle_ms=_bounded_integer(
                value.get("handle_ms", 500), "handle_ms", minimum=0, maximum=5000
            ),
            schema_version=JOB_SCHEMA_VERSION,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "source_key": self.source_key,
            "result_prefix": self.result_prefix,
            "transcript_key": self.transcript_key,
            "top_k": self.top_k,
            "max_duration_seconds": self.max_duration_seconds,
            "handle_ms": self.handle_ms,
        }


@dataclass(frozen=True, slots=True)
class UploadedArtifact:
    name: str
    key: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "key": self.key, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class JobResult:
    job_id: str
    status_key: str
    artifacts: tuple[UploadedArtifact, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": self.job_id,
            "status": "completed",
            "status_key": self.status_key,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }
