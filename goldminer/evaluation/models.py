from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from goldminer.errors import GoldMinerError

EVALUATION_DATASET_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class GoldMoment:
    id: str
    start_ms: int
    end_ms: int
    relevance: int
    idea_group: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoldMoment":
        try:
            moment = cls(
                id=str(value["id"]),
                start_ms=value["start_ms"],
                end_ms=value["end_ms"],
                relevance=value.get("relevance", 3),
                idea_group=str(value.get("idea_group", value["id"])),
            )
        except (KeyError, TypeError) as exc:
            raise GoldMinerError(f"Malformed gold moment: {exc}") from exc
        if not moment.id or not isinstance(moment.start_ms, int) or not isinstance(moment.end_ms, int):
            raise GoldMinerError("Gold moments require an ID and integer millisecond timestamps")
        if moment.start_ms < 0 or moment.end_ms <= moment.start_ms:
            raise GoldMinerError(f"Gold moment {moment.id} has an invalid timestamp range")
        if isinstance(moment.relevance, bool) or not isinstance(moment.relevance, int) or not 1 <= moment.relevance <= 3:
            raise GoldMinerError(f"Gold moment {moment.id} relevance must be an integer from 1 to 3")
        if not moment.idea_group:
            raise GoldMinerError(f"Gold moment {moment.id} requires a non-empty idea_group")
        return moment


@dataclass(frozen=True, slots=True)
class EvaluationEpisode:
    id: str
    ranking_path: Path
    gold_moments: tuple[GoldMoment, ...]


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    name: str
    episodes: tuple[EvaluationEpisode, ...]


def load_dataset(path: Path) -> EvaluationDataset:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["schema_version"] != EVALUATION_DATASET_SCHEMA_VERSION:
            raise GoldMinerError("Unsupported evaluation dataset schema version")
        name = str(raw["name"]).strip()
        episodes_raw = raw["episodes"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GoldMinerError(f"Could not load evaluation dataset: {exc}") from exc
    if not name or not isinstance(episodes_raw, list) or not episodes_raw:
        raise GoldMinerError("Evaluation dataset requires a name and at least one episode")
    episodes: list[EvaluationEpisode] = []
    seen: set[str] = set()
    for value in episodes_raw:
        try:
            episode_id = str(value["id"]).strip()
            ranking = Path(value["ranking_path"])
            ranking_path = ranking if ranking.is_absolute() else (path.parent / ranking).resolve()
            moments = tuple(GoldMoment.from_dict(item) for item in value["gold_moments"])
        except (KeyError, TypeError) as exc:
            raise GoldMinerError(f"Malformed evaluation episode: {exc}") from exc
        if not episode_id or episode_id in seen or not moments:
            raise GoldMinerError("Evaluation episode IDs must be unique and each episode needs gold moments")
        seen.add(episode_id)
        episodes.append(EvaluationEpisode(episode_id, ranking_path, moments))
    return EvaluationDataset(name, tuple(episodes))


def load_ranking(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        ranking = raw["final_ranking"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GoldMinerError(f"Could not load ranking {path}: {exc}") from exc
    if not isinstance(ranking, list):
        raise GoldMinerError(f"Ranking final_ranking must be an array: {path}")
    parsed: list[dict[str, Any]] = []
    for item in ranking:
        if not isinstance(item, dict):
            raise GoldMinerError(f"Ranking contains a malformed candidate: {path}")
        try:
            start, end = item["start_ms"], item["end_ms"]
        except KeyError as exc:
            raise GoldMinerError(f"Ranking candidate is missing timestamps: {path}") from exc
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise GoldMinerError(f"Ranking candidate has invalid timestamps: {path}")
        parsed.append(item)
    return tuple(parsed)
