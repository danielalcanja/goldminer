from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from copy import deepcopy
from typing import Protocol, Sequence

from goldminer.analysis.scoring import ScoredCandidate
from goldminer.errors import GoldMinerError
from goldminer.providers.analysis import JsonTransport, _transport


@dataclass(frozen=True, slots=True)
class ComparativeJudgment:
    candidate_id: str
    rank: int
    reason: str


class RerankingProvider(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    def rerank(self, candidates: Sequence[ScoredCandidate]) -> Sequence[ComparativeJudgment]: ...


@dataclass(frozen=True, slots=True)
class FakeRerankingProvider:
    order: tuple[str, ...]
    model_name: str = "fixture-v1"
    @property
    def name(self) -> str: return "fake"
    @property
    def model(self) -> str: return self.model_name
    def rerank(self, candidates: Sequence[ScoredCandidate]) -> Sequence[ComparativeJudgment]:
        return tuple(ComparativeJudgment(cid, i, "Fixture comparative judgment") for i, cid in enumerate(self.order, 1))


RERANK_SCHEMA = {"type":"object","additionalProperties":False,"properties":{"ranking":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{"candidate_id":{"type":"string"},"rank":{"type":"integer","minimum":1},"reason":{"type":"string"}},"required":["candidate_id","rank","reason"]}}},"required":["ranking"]}


@dataclass(frozen=True, slots=True)
class OpenAIRerankingProvider:
    api_key: str
    model_name: str = "gpt-5.4-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 300.0
    max_retries: int = 3
    transport: JsonTransport = _transport

    @classmethod
    def from_environment(cls) -> "OpenAIRerankingProvider":
        return cls(os.environ.get("OPENAI_API_KEY", "").strip(), os.environ.get("OPENAI_RERANK_MODEL", os.environ.get("OPENAI_SCORING_MODEL", "gpt-5.4-mini")).strip(), os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"))
    @property
    def name(self) -> str: return "openai"
    @property
    def model(self) -> str: return self.model_name

    def rerank(self, candidates: Sequence[ScoredCandidate]) -> Sequence[ComparativeJudgment]:
        if not self.api_key or self.api_key == "replace-with-new-key": raise GoldMinerError("OPENAI_API_KEY is not configured for reranking")
        sections = [f"ID {c.id} | score {c.final_score} | category {c.category} | {c.transcript}" for c in candidates]
        schema = deepcopy(RERANK_SCHEMA)
        item_schema = schema["properties"]["ranking"]["items"]
        item_schema["properties"]["candidate_id"]["enum"] = [candidate.id for candidate in candidates]
        item_schema["properties"]["rank"]["maximum"] = len(candidates)
        schema["properties"]["ranking"]["minItems"] = len(candidates)
        schema["properties"]["ranking"]["maxItems"] = len(candidates)
        body = json.dumps({"model":self.model_name,"instructions":"Comparatively rank every podcast clip from strongest to weakest for short-form publishing. Reward immediate hooks, standalone clarity, payoff, specificity, emotional or practical value, and audience fit. Return every supplied ID exactly once. Rank values express order and must be unique, but the client will normalize numbering defensively. Include a concise evidence-based reason.","input":"\n\n".join(sections),"text":{"format":{"type":"json_schema","name":"comparative_ranking","strict":True,"schema":schema}},"store":False}).encode()
        headers={"Authorization":f"Bearer {self.api_key}","Content-Type":"application/json","User-Agent":"podcast-gold-miner/0.1.0"}
        status,payload=0,b""
        for attempt in range(self.max_retries+1):
            status,payload=self.transport(f"{self.base_url}/responses",headers,body,self.timeout_seconds)
            if 200 <= status < 300: break
            if status not in {408,409,429,599} and status < 500: break
            if attempt < self.max_retries: time.sleep(min(2**attempt,8))
        if not 200 <= status < 300: raise GoldMinerError(f"OpenAI reranking failed with HTTP {status}: {payload.decode(errors='replace')}")
        try:
            response=json.loads(payload); text=response.get("output_text")
            if not isinstance(text,str):
                text=next(part["text"] for item in response["output"] for part in item.get("content",[]) if part.get("type")=="output_text")
            raw=json.loads(text)["ranking"]
            parsed=tuple(ComparativeJudgment(str(x["candidate_id"]),int(x["rank"]),str(x["reason"])) for x in raw)
        except (json.JSONDecodeError,KeyError,TypeError,ValueError,StopIteration) as exc: raise GoldMinerError("OpenAI returned invalid structured reranking JSON") from exc
        expected = {candidate.id for candidate in candidates}
        if len(parsed) != len(candidates) or {item.candidate_id for item in parsed} != expected:
            raise GoldMinerError("Reranking response must contain every candidate ID exactly once")
        # Structured output guarantees integer ranks but cannot guarantee array-wide
        # uniqueness. Preserve the model's intended ordering and normalize to 1..N.
        result = tuple(
            ComparativeJudgment(item.candidate_id, rank, item.reason)
            for rank, item in enumerate(sorted(parsed, key=lambda item: (item.rank, item.candidate_id)), 1)
        )
        validate_judgments(result, candidates)
        return result


def validate_judgments(judgments: Sequence[ComparativeJudgment], candidates: Sequence[ScoredCandidate]) -> None:
    expected={c.id for c in candidates}; actual={j.candidate_id for j in judgments}
    if actual != expected or len(judgments) != len(candidates) or {j.rank for j in judgments} != set(range(1,len(candidates)+1)) or any(not j.reason.strip() for j in judgments):
        raise GoldMinerError("Reranking response must contain every candidate exactly once with consecutive ranks")
