from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from goldminer.analysis.scoring import ScoredCandidate
from goldminer.errors import GoldMinerError
from goldminer.providers.embeddings import EmbeddingProvider
from goldminer.providers.reranking import ComparativeJudgment, RerankingProvider

RANKING_SCHEMA_VERSION = "1.0"
DEDUPE_VERSION = "1.0"
SELECTION_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class RankingConfig:
    semantic_threshold: float = 0.86
    timestamp_overlap_threshold: float = 0.60
    shortlist_size: int = 25
    max_per_category: int = 2


@dataclass(frozen=True, slots=True)
class DuplicateFamily:
    representative_id: str
    member_ids: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RankingResult:
    selected: tuple[ScoredCandidate, ...]
    families: tuple[DuplicateFamily, ...]
    shortlist_count: int
    reused_embeddings: bool
    reused_reranking: bool


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise GoldMinerError("Cannot compare embedding vectors with different dimensions")
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return 0.0 if denominator == 0 else sum(x * y for x, y in zip(left, right)) / denominator


def timestamp_overlap(left: ScoredCandidate, right: ScoredCandidate) -> float:
    overlap = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    return overlap / min(left.end_ms - left.start_ms, right.end_ms - right.start_ms)


def deduplicate(candidates: Sequence[ScoredCandidate], vectors: Sequence[Sequence[float]], config: RankingConfig) -> tuple[tuple[ScoredCandidate, ...], tuple[DuplicateFamily, ...]]:
    if len(candidates) != len(vectors):
        raise GoldMinerError("Candidate and embedding counts do not match")
    parent = list(range(len(candidates)))
    pair_reasons: dict[tuple[int, int], str] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            overlap = timestamp_overlap(candidates[left], candidates[right])
            semantic = cosine_similarity(vectors[left], vectors[right])
            labels = []
            if overlap >= config.timestamp_overlap_threshold:
                labels.append(f"timestamp_overlap={overlap:.3f}")
            if semantic >= config.semantic_threshold:
                labels.append(f"semantic_similarity={semantic:.3f}")
            if labels:
                union(left, right)
                pair_reasons[(left, right)] = ", ".join(labels)

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)
    kept: list[ScoredCandidate] = []
    families: list[DuplicateFamily] = []
    for members in groups.values():
        representative = min(members, key=lambda i: (-candidates[i].final_score, candidates[i].end_ms - candidates[i].start_ms, candidates[i].id))
        kept.append(candidates[representative])
        reasons = tuple(reason for pair, reason in pair_reasons.items() if pair[0] in members and pair[1] in members)
        families.append(DuplicateFamily(candidates[representative].id, tuple(candidates[i].id for i in members), reasons))
    return tuple(sorted(kept, key=lambda c: (-c.final_score, c.id))), tuple(sorted(families, key=lambda f: f.representative_id))


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_key(candidates: Sequence[ScoredCandidate], model: str, version: str) -> str:
    identity = model + version + "|".join(f"{c.id}:{c.start_ms}:{c.end_ms}:{c.final_score}:{c.transcript}" for c in candidates)
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def cached_embeddings(candidates: Sequence[ScoredCandidate], provider: EmbeddingProvider, cache_dir: Path) -> tuple[tuple[tuple[float, ...], ...], bool]:
    path = cache_dir / f"embeddings-{_cache_key(candidates, provider.model, DEDUPE_VERSION)}.json"
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            vectors = tuple(tuple(float(x) for x in vector) for vector in raw["vectors"])
            if len(vectors) == len(candidates):
                return vectors, True
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    vectors = tuple(tuple(float(x) for x in vector) for vector in provider.embed([f"{c.category}. {c.transcript}" for c in candidates]))
    if len(vectors) != len(candidates):
        raise GoldMinerError("Embedding provider returned the wrong number of vectors")
    _atomic_write(path, {"version": DEDUPE_VERSION, "model": provider.model, "vectors": vectors})
    return vectors, False


def _validate_judgments(candidates: Sequence[ScoredCandidate], judgments: Sequence[ComparativeJudgment]) -> None:
    expected = {candidate.id for candidate in candidates}
    actual = {judgment.candidate_id for judgment in judgments}
    ranks = {judgment.rank for judgment in judgments}
    if actual != expected or len(judgments) != len(candidates) or ranks != set(range(1, len(candidates) + 1)):
        raise GoldMinerError("Comparative reranking must return every candidate exactly once with contiguous ranks")


def cached_reranking(candidates: Sequence[ScoredCandidate], provider: RerankingProvider, cache_dir: Path) -> tuple[tuple[ComparativeJudgment, ...], bool]:
    path = cache_dir / f"rerank-{_cache_key(candidates, provider.model, SELECTION_VERSION)}.json"
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            judgments = tuple(ComparativeJudgment(item["candidate_id"], item["rank"], item["reason"]) for item in raw["ranking"])
            _validate_judgments(candidates, judgments)
            return judgments, True
        except (OSError, json.JSONDecodeError, KeyError, TypeError, GoldMinerError):
            pass
    judgments = tuple(provider.rerank(candidates))
    _validate_judgments(candidates, judgments)
    _atomic_write(path, {"version": SELECTION_VERSION, "model": provider.model, "ranking": [{"candidate_id": j.candidate_id, "rank": j.rank, "reason": j.reason} for j in judgments]})
    return judgments, False


def build_ranking(candidates: Sequence[ScoredCandidate], embedding_provider: EmbeddingProvider, reranking_provider: RerankingProvider, cache_dir: Path, top_k: int, config: RankingConfig = RankingConfig()) -> tuple[RankingResult, dict[str, Any]]:
    eligible = tuple(sorted((candidate for candidate in candidates if not candidate.rejected), key=lambda c: (-c.final_score, c.id)))
    if not eligible:
        artifact = {
            "schema_version": RANKING_SCHEMA_VERSION,
            "dedupe_version": DEDUPE_VERSION,
            "selection_version": SELECTION_VERSION,
            "embedding_provider": embedding_provider.name,
            "embedding_model": embedding_provider.model,
            "reranking_provider": reranking_provider.name,
            "reranking_model": reranking_provider.model,
            "config": {"semantic_threshold": config.semantic_threshold, "timestamp_overlap_threshold": config.timestamp_overlap_threshold, "shortlist_size": config.shortlist_size, "max_per_category": config.max_per_category, "top_k": top_k},
            "counts": {"input": len(candidates), "eligible": 0, "duplicate_families": 0, "representatives": 0, "shortlist": 0, "selected": 0},
            "duplicate_families": [],
            "final_ranking": [],
        }
        return RankingResult((), (), 0, False, False), artifact
    vectors, reused_embeddings = cached_embeddings(eligible, embedding_provider, cache_dir)
    representatives, families = deduplicate(eligible, vectors, config)
    shortlist = representatives[:config.shortlist_size]
    judgments, reused_reranking = cached_reranking(shortlist, reranking_provider, cache_dir)
    comparative = {judgment.candidate_id: judgment for judgment in judgments}
    score_order = {candidate.id: rank for rank, candidate in enumerate(shortlist, 1)}
    ordered = sorted(shortlist, key=lambda candidate: comparative[candidate.id].rank)
    selected: list[ScoredCandidate] = []
    category_counts: dict[str, int] = {}
    deferred: list[ScoredCandidate] = []
    for candidate in ordered:
        if category_counts.get(candidate.category, 0) >= config.max_per_category:
            deferred.append(candidate)
            continue
        selected.append(candidate)
        category_counts[candidate.category] = category_counts.get(candidate.category, 0) + 1
        if len(selected) == min(top_k, len(shortlist)):
            break
    for candidate in deferred + ordered:
        if len(selected) == min(top_k, len(shortlist)):
            break
        if candidate not in selected:
            selected.append(candidate)
    final_entries = []
    for rank, candidate in enumerate(selected, 1):
        judgment = comparative[candidate.id]
        agreement = abs(score_order[candidate.id] - judgment.rank)
        margin = candidate.final_score - (ordered[judgment.rank].final_score if judgment.rank < len(ordered) else 0)
        confidence = "high" if agreement <= 3 and candidate.final_score >= 75 else "medium" if agreement <= 7 and candidate.final_score >= 60 else "low"
        final_entries.append({
            "rank": rank, **candidate.to_dict(), "comparative_rank": judgment.rank,
            "comparative_reason": judgment.reason, "score_rank": score_order[candidate.id],
            "rank_agreement_delta": agreement, "score_margin_to_next_comparative": round(margin, 2),
            "confidence": confidence, "selection_reason": "Comparative rank plus category-diversity selection",
        })
    artifact = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "dedupe_version": DEDUPE_VERSION,
        "selection_version": SELECTION_VERSION,
        "embedding_provider": embedding_provider.name,
        "embedding_model": embedding_provider.model,
        "reranking_provider": reranking_provider.name,
        "reranking_model": reranking_provider.model,
        "config": {"semantic_threshold": config.semantic_threshold, "timestamp_overlap_threshold": config.timestamp_overlap_threshold, "shortlist_size": config.shortlist_size, "max_per_category": config.max_per_category, "top_k": top_k},
        "counts": {"input": len(candidates), "eligible": len(eligible), "duplicate_families": sum(len(f.member_ids) > 1 for f in families), "representatives": len(representatives), "shortlist": len(shortlist), "selected": len(selected)},
        "duplicate_families": [{"representative_id": family.representative_id, "member_ids": list(family.member_ids), "suppressed_ids": [item for item in family.member_ids if item != family.representative_id], "reasons": list(family.reasons)} for family in families if len(family.member_ids) > 1],
        "final_ranking": final_entries,
    }
    return RankingResult(tuple(selected), families, len(shortlist), reused_embeddings, reused_reranking), artifact


def write_ranking(path: Path, artifact: dict[str, Any]) -> None:
    _atomic_write(path, artifact)
