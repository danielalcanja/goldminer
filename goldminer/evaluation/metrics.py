from __future__ import annotations

import math
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from goldminer.evaluation.models import EvaluationDataset, GoldMoment, load_ranking

EVALUATION_REPORT_SCHEMA_VERSION = "1.0"
MATCH_OVERLAP_THRESHOLD = 0.50


def overlap_ratio(start_ms: int, end_ms: int, gold: GoldMoment) -> float:
    overlap = max(0, min(end_ms, gold.end_ms) - max(start_ms, gold.start_ms))
    shortest = min(end_ms - start_ms, gold.end_ms - gold.start_ms)
    return overlap / shortest if shortest > 0 else 0.0


def _best_match(candidate: dict[str, Any], gold: Sequence[GoldMoment]) -> GoldMoment | None:
    scored = [(overlap_ratio(candidate["start_ms"], candidate["end_ms"], item), item) for item in gold]
    score, result = max(scored, key=lambda pair: pair[0], default=(0.0, None))
    return result if score >= MATCH_OVERLAP_THRESHOLD else None


def _unique_matches(predictions: Sequence[dict[str, Any]], gold: Sequence[GoldMoment]) -> list[GoldMoment | None]:
    available = {item.id: item for item in gold}
    matches: list[GoldMoment | None] = []
    for candidate in predictions:
        scored = [
            (overlap_ratio(candidate["start_ms"], candidate["end_ms"], item), item)
            for item in available.values()
        ]
        score, match = max(scored, key=lambda pair: pair[0], default=(0.0, None))
        if match is not None and score >= MATCH_OVERLAP_THRESHOLD:
            matches.append(match)
            available.pop(match.id)
        else:
            matches.append(None)
    return matches


def _dcg(relevances: Sequence[int]) -> float:
    return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(relevances))


def evaluate_episode(predictions: Sequence[dict[str, Any]], gold: Sequence[GoldMoment]) -> dict[str, Any]:
    top10 = tuple(predictions[:10])
    top5 = tuple(predictions[:5])
    unique_top10 = _unique_matches(top10, gold)
    unique_top5 = _unique_matches(top5, gold)
    matched_gold = {item.id for item in gold if any(overlap_ratio(p["start_ms"], p["end_ms"], item) >= MATCH_OVERLAP_THRESHOLD for p in top10)}
    recall = len(matched_gold) / len(gold)
    precision = sum(item is not None for item in unique_top5) / len(top5) if top5 else 0.0
    relevance = [match.relevance if match else 0 for match in unique_top10]
    ideal = sorted((item.relevance for item in gold), reverse=True)[:10]
    ideal_dcg = _dcg(ideal)
    ndcg = _dcg(relevance) / ideal_dcg if ideal_dcg else 0.0
    boundary_errors = []
    groups = []
    for prediction, match in zip(top10, unique_top10):
        if match:
            boundary_errors.append(
                (abs(prediction["start_ms"] - match.start_ms) + abs(prediction["end_ms"] - match.end_ms)) / 2000
            )
    for prediction in top10:
        match = _best_match(prediction, gold)
        if match:
            groups.append(match.idea_group)
    duplicate_count = len(groups) - len(set(groups))
    return {
        "gold_count": len(gold),
        "prediction_count": len(predictions),
        "recall_at_10": round(recall, 4),
        "precision_at_5": round(precision, 4),
        "ndcg_at_10": round(ndcg, 4),
        "boundary_error_seconds": round(mean(boundary_errors), 3) if boundary_errors else None,
        "duplicate_rate": round(duplicate_count / len(top10), 4) if top10 else 0.0,
        "matched_gold_ids": sorted(matched_gold),
    }


def evaluate_dataset(dataset: EvaluationDataset) -> dict[str, Any]:
    episodes = []
    errors = []
    for episode in dataset.episodes:
        try:
            result = evaluate_episode(load_ranking(episode.ranking_path), episode.gold_moments)
            episodes.append({"episode_id": episode.id, "ranking_path": str(episode.ranking_path), **result})
        except Exception as exc:
            errors.append({"episode_id": episode.id, "error": str(exc)})
    completed = len(episodes)
    boundary_values = [item["boundary_error_seconds"] for item in episodes if item["boundary_error_seconds"] is not None]
    aggregate = {
        "recall_at_10": round(mean(item["recall_at_10"] for item in episodes), 4) if episodes else 0.0,
        "precision_at_5": round(mean(item["precision_at_5"] for item in episodes), 4) if episodes else 0.0,
        "ndcg_at_10": round(mean(item["ndcg_at_10"] for item in episodes), 4) if episodes else 0.0,
        "boundary_error_seconds": round(mean(boundary_values), 3) if boundary_values else None,
        "duplicate_rate": round(mean(item["duplicate_rate"] for item in episodes), 4) if episodes else 0.0,
        "run_reliability": round(completed / len(dataset.episodes), 4),
    }
    return {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "dataset_name": dataset.name,
        "match_overlap_threshold": MATCH_OVERLAP_THRESHOLD,
        "episode_count": len(dataset.episodes),
        "completed_episode_count": completed,
        "metrics": aggregate,
        "episodes": episodes,
        "errors": errors,
    }
