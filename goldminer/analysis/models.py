from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from goldminer.errors import GoldMinerError
from goldminer.transcript.models import Utterance


@dataclass(frozen=True, slots=True)
class DiscoveryWindow:
    id: str
    utterances: tuple[Utterance, ...]

    @property
    def start_ms(self) -> int:
        return self.utterances[0].start_ms

    @property
    def end_ms(self) -> int:
        return self.utterances[-1].end_ms

    def prompt_text(self) -> str:
        return "\n".join(
            f"[{item.id}] {item.start_ms}-{item.end_ms}ms"
            f" {item.speaker or 'Unknown'}: {item.original_text}"
            for item in self.utterances
        )


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    detector: str
    start_utterance_id: str
    end_utterance_id: str
    core_idea: str
    evidence: str


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    start_ms: int
    end_ms: int
    source_utterance_ids: tuple[str, ...]
    detector_categories: tuple[str, ...]
    transcript: str
    core_idea: str
    evidence: str
    source_windows: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_sec": round((self.end_ms - self.start_ms) / 1000, 3),
            "source_utterance_ids": list(self.source_utterance_ids),
            "detector_categories": list(self.detector_categories),
            "transcript": self.transcript,
            "core_idea": self.core_idea,
            "evidence": self.evidence,
            "source_windows": list(self.source_windows),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Candidate":
        try:
            candidate = cls(
                id=str(value["id"]),
                start_ms=int(value["start_ms"]),
                end_ms=int(value["end_ms"]),
                source_utterance_ids=tuple(str(item) for item in value["source_utterance_ids"]),
                detector_categories=tuple(str(item) for item in value["detector_categories"]),
                transcript=str(value["transcript"]),
                core_idea=str(value["core_idea"]),
                evidence=str(value["evidence"]),
                source_windows=tuple(str(item) for item in value["source_windows"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GoldMinerError(f"Malformed candidate artifact: {exc}") from exc
        if candidate.start_ms < 0 or candidate.end_ms <= candidate.start_ms:
            raise GoldMinerError(f"Candidate {candidate.id} has invalid timestamps")
        if not candidate.source_utterance_ids or not candidate.detector_categories:
            raise GoldMinerError(f"Candidate {candidate.id} is missing source IDs or categories")
        return candidate
