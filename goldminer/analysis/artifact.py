from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from goldminer.analysis.models import Candidate, CandidateProposal, DiscoveryWindow
from goldminer.errors import GoldMinerError
from goldminer.providers.analysis import AnalysisProvider

CANDIDATE_SCHEMA_VERSION = "1.0"
DISCOVERY_PROMPT_VERSION = "1.1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class CachedAnalysisProvider:
    provider: AnalysisProvider
    cache_dir: Path
    transcript_sha256: str

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    def _path(self, window: DiscoveryWindow) -> Path:
        digest = hashlib.sha256(
            (
                self.transcript_sha256
                + self.model
                + DISCOVERY_PROMPT_VERSION
                + window.id
                + window.utterances[0].id
                + window.utterances[-1].id
            ).encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"{window.id}-{digest}.json"

    def discover(self, window: DiscoveryWindow) -> Sequence[CandidateProposal]:
        path = self._path(window)
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                return tuple(CandidateProposal(**item) for item in raw["proposals"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                pass
        proposals = tuple(self.provider.discover(window))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            path,
            {"prompt_version": DISCOVERY_PROMPT_VERSION, "model": self.model, "proposals": [
                {
                    "detector": item.detector,
                    "start_utterance_id": item.start_utterance_id,
                    "end_utterance_id": item.end_utterance_id,
                    "core_idea": item.core_idea,
                    "evidence": item.evidence,
                }
                for item in proposals
            ]},
        )
        return proposals


def write_candidates_artifact(
    path: Path,
    *,
    transcript_sha256: str,
    provider: AnalysisProvider,
    window_count: int,
    candidates: Sequence[Candidate],
) -> None:
    _atomic_write(
        path,
        {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "prompt_version": DISCOVERY_PROMPT_VERSION,
            "provider": provider.name,
            "model": provider.model,
            "transcript_sha256": transcript_sha256,
            "window_count": window_count,
            "candidate_count": len(candidates),
            "candidates": [item.to_dict() for item in candidates],
        },
    )
    load_candidates_artifact(path)


def load_candidates_artifact(path: Path) -> tuple[dict[str, Any], tuple[Candidate, ...]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["schema_version"] != CANDIDATE_SCHEMA_VERSION:
            raise GoldMinerError("Unsupported candidates schema version")
        candidates = tuple(Candidate.from_dict(item) for item in raw["candidates"])
        if raw["candidate_count"] != len(candidates):
            raise GoldMinerError("candidate_count does not match candidates")
        return raw, candidates
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GoldMinerError(f"Could not load candidates artifact: {exc}") from exc
