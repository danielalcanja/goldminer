from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Callable, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from goldminer.analysis.detectors import DETECTORS
from goldminer.analysis.models import CandidateProposal, DiscoveryWindow
from goldminer.errors import GoldMinerError


class AnalysisProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def discover(self, window: DiscoveryWindow) -> Sequence[CandidateProposal]: ...


@dataclass(frozen=True, slots=True)
class FakeAnalysisProvider:
    proposals_by_window: dict[str, Sequence[CandidateProposal]]
    provider_name: str = "fake"
    model_name: str = "fixture-v1"

    @property
    def name(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return self.model_name

    def discover(self, window: DiscoveryWindow) -> Sequence[CandidateProposal]:
        return tuple(self.proposals_by_window.get(window.id, ()))


JsonTransport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


def _transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except (URLError, TimeoutError) as exc:
        return 599, json.dumps({"error": {"message": f"network error: {exc}"}}).encode()


DISCOVERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidates": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "detector": {"type": "string", "enum": list(DETECTORS)},
                    "start_utterance_id": {"type": "string"},
                    "end_utterance_id": {"type": "string"},
                    "core_idea": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "detector",
                    "start_utterance_id",
                    "end_utterance_id",
                    "core_idea",
                    "evidence",
                ],
            },
        }
    },
    "required": ["candidates"],
}


@dataclass(frozen=True, slots=True)
class OpenAIAnalysisProvider:
    api_key: str
    model_name: str = "gpt-5.4-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 300.0
    max_retries: int = 3
    transport: JsonTransport = _transport

    @classmethod
    def from_environment(cls) -> "OpenAIAnalysisProvider":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            model_name=os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-5.4-mini").strip(),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        )

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self.model_name

    def _output_text(self, response: dict[str, object]) -> str:
        direct = response.get("output_text")
        if isinstance(direct, str):
            return direct
        output = response.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            text = part.get("text")
                            if isinstance(text, str):
                                return text
        raise GoldMinerError("OpenAI analysis response contained no output text")

    def discover(self, window: DiscoveryWindow) -> Sequence[CandidateProposal]:
        if not self.api_key or self.api_key == "replace-with-new-key":
            raise GoldMinerError("OPENAI_API_KEY is not configured for candidate discovery")
        instructions = (
            "You are a short-form content analyst, not a copywriter. Search the supplied window "
            "using every detector category. Return only complete high-value moments as contiguous "
            "utterance-ID ranges. Do not invent, rewrite, or join non-contiguous speech. Optimize "
            "for discovery recall, but omit weak filler. Candidate ranges may be 15-120 seconds. A candidate "
            "must contain its explicit conclusion or payoff; never infer a solution that is not spoken inside the "
            "range. When adjacent speech forms setup, contrast, and resolution, return the broader complete arc "
            "instead of splitting it into incomplete neighboring candidates. It is acceptable to return an "
            "additional broader arc when shorter moments are independently complete."
        )
        body = json.dumps(
            {
                "model": self.model_name,
                "instructions": instructions,
                "input": (
                    "Detector categories: " + ", ".join(DETECTORS) + "\n\n" + window.prompt_text()
                ),
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "candidate_discovery",
                        "strict": True,
                        "schema": DISCOVERY_SCHEMA,
                    }
                },
                "store": False,
            }
        ).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "podcast-gold-miner/0.1.0",
        }
        status, payload = 0, b""
        for attempt in range(self.max_retries + 1):
            status, payload = self.transport(
                f"{self.base_url}/responses", headers, body, self.timeout_seconds
            )
            if 200 <= status < 300:
                break
            if status not in {408, 409, 429, 599} and status < 500:
                break
            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))
        if not 200 <= status < 300:
            try:
                message = json.loads(payload).get("error", {}).get("message", payload.decode())
            except (json.JSONDecodeError, AttributeError):
                message = payload.decode(errors="replace") or "unknown API error"
            raise GoldMinerError(f"OpenAI candidate discovery failed with HTTP {status}: {message}")
        try:
            response = json.loads(payload)
            structured = json.loads(self._output_text(response))
            raw_candidates = structured["candidates"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise GoldMinerError("OpenAI returned invalid structured candidate JSON") from exc
        if not isinstance(raw_candidates, list):
            raise GoldMinerError("OpenAI candidates must be an array")
        proposals: list[CandidateProposal] = []
        for index, value in enumerate(raw_candidates, start=1):
            try:
                proposal = CandidateProposal(
                    detector=value["detector"],
                    start_utterance_id=value["start_utterance_id"],
                    end_utterance_id=value["end_utterance_id"],
                    core_idea=value["core_idea"].strip(),
                    evidence=value["evidence"].strip(),
                )
            except (KeyError, TypeError, AttributeError) as exc:
                raise GoldMinerError(f"Malformed candidate proposal {index}") from exc
            if proposal.detector not in DETECTORS or not proposal.core_idea or not proposal.evidence:
                raise GoldMinerError(f"Invalid candidate proposal {index}")
            proposals.append(proposal)
        return tuple(proposals)
