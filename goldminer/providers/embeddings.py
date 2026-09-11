from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time
from typing import Protocol, Sequence

from goldminer.errors import GoldMinerError
from goldminer.providers.analysis import JsonTransport, _transport


class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True, slots=True)
class FakeEmbeddingProvider:
    vectors: dict[str, tuple[float, ...]]
    model_name: str = "fixture-v1"

    @property
    def name(self) -> str: return "fake"
    @property
    def model(self) -> str: return self.model_name
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self.vectors[text] for text in texts)


@dataclass(frozen=True, slots=True)
class OpenAIEmbeddingProvider:
    api_key: str
    model_name: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 180.0
    max_retries: int = 3
    transport: JsonTransport = _transport

    @classmethod
    def from_environment(cls) -> "OpenAIEmbeddingProvider":
        return cls(os.environ.get("OPENAI_API_KEY", "").strip(), os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(), os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"))

    @property
    def name(self) -> str: return "openai"
    @property
    def model(self) -> str: return self.model_name

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not self.api_key or self.api_key == "replace-with-new-key":
            raise GoldMinerError("OPENAI_API_KEY is not configured for embeddings")
        if not texts: return ()
        body = json.dumps({"model": self.model_name, "input": list(texts), "encoding_format": "float"}).encode()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "User-Agent": "podcast-gold-miner/0.1.0"}
        status, payload = 0, b""
        for attempt in range(self.max_retries + 1):
            status, payload = self.transport(f"{self.base_url}/embeddings", headers, body, self.timeout_seconds)
            if 200 <= status < 300: break
            if status not in {408, 409, 429, 599} and status < 500: break
            if attempt < self.max_retries: time.sleep(min(2**attempt, 8))
        if not 200 <= status < 300:
            try: message = json.loads(payload).get("error", {}).get("message", payload.decode())
            except (json.JSONDecodeError, AttributeError): message = payload.decode(errors="replace") or "unknown API error"
            raise GoldMinerError(f"OpenAI embeddings failed with HTTP {status}: {message}")
        try:
            data = sorted(json.loads(payload)["data"], key=lambda item: item["index"])
            vectors = tuple(tuple(float(x) for x in item["embedding"]) for item in data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise GoldMinerError("OpenAI returned invalid embedding JSON") from exc
        if len(vectors) != len(texts) or not vectors or any(not vector for vector in vectors):
            raise GoldMinerError("Embedding response count or dimensions do not match inputs")
        dimension = len(vectors[0])
        if any(len(v) != dimension or any(not math.isfinite(x) for x in v) for v in vectors):
            raise GoldMinerError("Embedding response contains inconsistent or non-finite vectors")
        return vectors
