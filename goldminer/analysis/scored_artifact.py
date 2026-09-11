from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from goldminer.analysis.boundaries import (
    BoundaryContext,
    apply_evaluation,
    opening_anchor_candidates,
)
from goldminer.analysis.discovery import canonical_utterance_id
from goldminer.analysis.scoring import (
    BOUNDARY_VERSION,
    DERIVED_REVIEW_VERSION,
    DIMENSION_WEIGHTS,
    EDIT_VARIANTS,
    MODEL_JUDGMENT_VERSION,
    SCORING_VERSION,
    DimensionJudgment,
    DerivedEditReview,
    ModelEvaluation,
    PenaltyJudgment,
    ScoredCandidate,
)
from goldminer.errors import GoldMinerError
from goldminer.providers.scoring import ScoringProvider
from goldminer.transcript.models import CanonicalTranscript

SCORED_ARTIFACT_SCHEMA_VERSION = "1.4"
# Each response is intentionally scoped to one idea. Structured-output models
# reliably return all three edits for one candidate, while larger batches can
# silently omit later candidates even when the JSON itself is valid.
SCORING_BATCH_SIZE = 1
MAX_DURATION_RESCUE_OVERAGE_MS = 15_000


@dataclass(frozen=True, slots=True)
class _EvaluatedVariant:
    evaluation: ModelEvaluation
    context: BoundaryContext
    scored: ScoredCandidate


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluation_to_dict(value: ModelEvaluation) -> dict[str, Any]:
    return {
        "candidate_id": value.candidate_id,
        "variant_id": value.variant_id,
        "start_utterance_id": value.start_utterance_id,
        "end_utterance_id": value.end_utterance_id,
        "start_boundary_text": value.start_boundary_text,
        "end_boundary_text": value.end_boundary_text,
        "category": value.category,
        "secondary_categories": list(value.secondary_categories),
        "dimensions": {
            name: {"raw_score": item.raw_score, "evidence": item.evidence}
            for name, item in value.dimensions.items()
        },
        "penalties": [
            {"type": item.type, "points": item.points, "evidence": item.evidence}
            for item in value.penalties
        ],
        "unsafe_extraction": value.unsafe_extraction,
        "non_contiguous_required": value.non_contiguous_required,
        "opening_is_clear": value.opening_is_clear,
        "opening_problem": value.opening_problem,
        "central_idea_is_strong": value.central_idea_is_strong,
        "central_idea_problem": value.central_idea_problem,
        "opening_evidence": value.opening_evidence,
        "central_idea_evidence": value.central_idea_evidence,
        "arc_is_complete": value.arc_is_complete,
        "payoff_type": value.payoff_type,
        "payoff_text": value.payoff_text,
        "arc_evidence": value.arc_evidence,
        "boundary_rationale": value.boundary_rationale,
        "why_selected": value.why_selected,
    }


def evaluation_from_dict(value: dict[str, Any]) -> ModelEvaluation:
    try:
        result = ModelEvaluation(
            candidate_id=value["candidate_id"],
            variant_id=value.get("variant_id", "primary"),
            start_utterance_id=value["start_utterance_id"],
            end_utterance_id=value["end_utterance_id"],
            start_boundary_text=value["start_boundary_text"],
            end_boundary_text=value["end_boundary_text"],
            category=value["category"],
            secondary_categories=tuple(value["secondary_categories"]),
            dimensions={
                name: DimensionJudgment(item["raw_score"], item["evidence"])
                for name, item in value["dimensions"].items()
            },
            penalties=tuple(
                PenaltyJudgment(item["type"], item["points"], item["evidence"])
                for item in value["penalties"]
            ),
            unsafe_extraction=value["unsafe_extraction"],
            non_contiguous_required=value["non_contiguous_required"],
            opening_is_clear=value["opening_is_clear"],
            opening_problem=value["opening_problem"],
            central_idea_is_strong=value["central_idea_is_strong"],
            central_idea_problem=value["central_idea_problem"],
            opening_evidence=value["opening_evidence"],
            central_idea_evidence=value["central_idea_evidence"],
            arc_is_complete=value["arc_is_complete"],
            payoff_type=value["payoff_type"],
            payoff_text=value["payoff_text"],
            arc_evidence=value["arc_evidence"],
            boundary_rationale=value["boundary_rationale"],
            why_selected=value["why_selected"],
        )
    except (KeyError, TypeError) as exc:
        raise GoldMinerError(f"Malformed cached evaluation: {exc}") from exc
    result.validate()
    return result


def derived_review_to_dict(value: DerivedEditReview) -> dict[str, Any]:
    return {
        "candidate_id": value.candidate_id,
        "dimensions": {
            name: {"raw_score": item.raw_score, "evidence": item.evidence}
            for name, item in value.dimensions.items()
        },
        "penalties": [
            {"type": item.type, "points": item.points, "evidence": item.evidence}
            for item in value.penalties
        ],
        "opening_is_clear": value.opening_is_clear,
        "opening_problem": value.opening_problem,
        "central_idea_is_strong": value.central_idea_is_strong,
        "central_idea_problem": value.central_idea_problem,
        "opening_evidence": value.opening_evidence,
        "central_idea_evidence": value.central_idea_evidence,
        "arc_is_complete": value.arc_is_complete,
        "payoff_type": value.payoff_type,
        "payoff_text": value.payoff_text,
        "arc_evidence": value.arc_evidence,
    }


def derived_review_from_dict(value: dict[str, Any]) -> DerivedEditReview:
    try:
        result = DerivedEditReview(
            candidate_id=value["candidate_id"],
            dimensions={
                name: DimensionJudgment(item["raw_score"], item["evidence"])
                for name, item in value["dimensions"].items()
            },
            penalties=tuple(
                PenaltyJudgment(item["type"], item["points"], item["evidence"])
                for item in value["penalties"]
            ),
            opening_is_clear=value["opening_is_clear"],
            opening_problem=value["opening_problem"],
            central_idea_is_strong=value["central_idea_is_strong"],
            central_idea_problem=value["central_idea_problem"],
            opening_evidence=value["opening_evidence"],
            central_idea_evidence=value["central_idea_evidence"],
            arc_is_complete=value["arc_is_complete"],
            payoff_type=value["payoff_type"],
            payoff_text=value["payoff_text"],
            arc_evidence=value["arc_evidence"],
        )
    except (KeyError, TypeError) as exc:
        raise GoldMinerError(f"Malformed cached derived edit review: {exc}") from exc
    result.validate()
    return result


@dataclass(frozen=True, slots=True)
class CachedScoringProvider:
    provider: ScoringProvider
    cache_dir: Path
    candidates_sha256: str

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    def _path(self, contexts: Sequence[BoundaryContext]) -> Path:
        identity = ",".join(item.candidate.id for item in contexts)
        digest = hashlib.sha256(
            (self.candidates_sha256 + self.model + MODEL_JUDGMENT_VERSION + BOUNDARY_VERSION + identity).encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"batch-{contexts[0].candidate.id}-{digest}.json"

    def evaluate(self, contexts: Sequence[BoundaryContext]) -> Sequence[ModelEvaluation]:
        path = self._path(contexts)
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                cached = tuple(evaluation_from_dict(item) for item in raw["evaluations"])
                if self._valid_for_contexts(cached, contexts):
                    return cached
            except (OSError, json.JSONDecodeError, KeyError, TypeError, GoldMinerError):
                pass
        evaluations = tuple(self.provider.evaluate(contexts))
        if not self._valid_for_contexts(evaluations, contexts):
            raise GoldMinerError("Scoring provider returned boundaries outside allowed candidate contexts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            path,
            {
                "scoring_version": SCORING_VERSION,
                "boundary_version": BOUNDARY_VERSION,
                "model": self.model,
                "evaluations": [evaluation_to_dict(item) for item in evaluations],
            },
        )
        return evaluations

    def _review_path(self, candidate: ScoredCandidate) -> Path:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "candidates_sha256": self.candidates_sha256,
                    "model": self.model,
                    "review_version": DERIVED_REVIEW_VERSION,
                    "candidate_id": candidate.id,
                    "variant_id": candidate.variant_id,
                    "start_ms": candidate.start_ms,
                    "end_ms": candidate.end_ms,
                    "transcript": candidate.transcript,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()[:20]
        return self.cache_dir / f"review-{candidate.id}-{digest}.json"

    def review_derived(
        self,
        candidates: Sequence[ScoredCandidate],
        original_ideas: Mapping[str, str],
    ) -> Sequence[DerivedEditReview]:
        results: list[DerivedEditReview] = []
        missing: list[ScoredCandidate] = []
        for candidate in candidates:
            path = self._review_path(candidate)
            if path.is_file():
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    review = derived_review_from_dict(raw["review"])
                    if review.candidate_id == candidate.id:
                        results.append(review)
                        continue
                except (OSError, json.JSONDecodeError, KeyError, TypeError, GoldMinerError):
                    pass
            missing.append(candidate)
        if missing:
            fresh = tuple(self.provider.review_derived(missing, original_ideas))
            expected = {item.id for item in missing}
            if (
                {item.candidate_id for item in fresh} != expected
                or len(fresh) != len(expected)
            ):
                raise GoldMinerError("Derived edit reviewer returned missing or duplicate candidates")
            fresh_by_id = {item.candidate_id: item for item in fresh}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for candidate in missing:
                review = fresh_by_id[candidate.id]
                review.validate()
                _atomic_write(
                    self._review_path(candidate),
                    {
                        "review_version": DERIVED_REVIEW_VERSION,
                        "model": self.model,
                        "review": derived_review_to_dict(review),
                    },
                )
                results.append(review)
        by_id = {item.candidate_id: item for item in results}
        return tuple(by_id[item.id] for item in candidates)

    def _valid_for_contexts(
        self,
        evaluations: Sequence[ModelEvaluation],
        contexts: Sequence[BoundaryContext],
    ) -> bool:
        if not evaluations and contexts:
            return False
        expected_ids = {item.candidate.id for item in contexts}
        if {item.candidate_id for item in evaluations} != expected_ids:
            return False
        pairs = {(item.candidate_id, item.variant_id) for item in evaluations}
        if len(pairs) != len(evaluations):
            return False
        by_id: dict[str, list[ModelEvaluation]] = {}
        for evaluation in evaluations:
            by_id.setdefault(evaluation.candidate_id, []).append(evaluation)
        for context in contexts:
            allowed = {item.id for item in context.utterances}
            allowed_order = {item.id: index for index, item in enumerate(context.utterances)}
            for evaluation in by_id[context.candidate.id]:
                start_id = canonical_utterance_id(evaluation.start_utterance_id)
                end_id = canonical_utterance_id(evaluation.end_utterance_id)
                if start_id not in allowed or end_id not in allowed:
                    return False
                if allowed_order[start_id] > allowed_order[end_id]:
                    return False
        return True


def _editorial_score(candidate: ScoredCandidate) -> float:
    return round(
        max(0.0, candidate.weighted_subtotal - sum(item.points for item in candidate.penalties)),
        2,
    )


def _selection_key(candidate: ScoredCandidate) -> tuple[object, ...]:
    variant_priority = {
        "duration_rescue": 0,
        "recombined": 1,
        "hook_first": 2,
        "full_arc": 3,
        "concise": 4,
        "primary": 5,
    }
    return (
        candidate.rejected,
        -_editorial_score(candidate),
        -candidate.dimensions["hook"].raw_score,
        -candidate.dimensions["payoff"].raw_score,
        candidate.end_ms - candidate.start_ms,
        variant_priority.get(candidate.variant_id, 99),
    )


def _alternative_summary(
    candidate: ScoredCandidate,
    *,
    selected: bool,
) -> dict[str, Any]:
    return {
        "variant_id": candidate.variant_id,
        "selected": selected,
        "start_ms": candidate.start_ms,
        "end_ms": candidate.end_ms,
        "duration_sec": round((candidate.end_ms - candidate.start_ms) / 1000, 3),
        "transcript": candidate.transcript,
        "editorial_score": _editorial_score(candidate),
        "final_score": candidate.final_score,
        "rejected": candidate.rejected,
        "rejection_reasons": list(candidate.rejection_reasons),
        "start_boundary_text": candidate.start_boundary_text,
        "end_boundary_text": candidate.end_boundary_text,
        "payoff_text": candidate.payoff_text,
    }


def _invalid_alternative(
    evaluation: ModelEvaluation,
    error: GoldMinerError,
) -> dict[str, Any]:
    return {
        "variant_id": evaluation.variant_id,
        "selected": False,
        "start_ms": None,
        "end_ms": None,
        "duration_sec": None,
        "transcript": "",
        "editorial_score": None,
        "final_score": 0.0,
        "rejected": True,
        "rejection_reasons": ["invalid_boundary"],
        "start_boundary_text": evaluation.start_boundary_text,
        "end_boundary_text": evaluation.end_boundary_text,
        "payoff_text": evaluation.payoff_text,
        "validation_error": str(error),
    }


def _conservative_dimensions(
    start: ModelEvaluation,
    end: ModelEvaluation,
) -> dict[str, DimensionJudgment]:
    dimensions = {
        name: (
            start.dimensions[name]
            if start.dimensions[name].raw_score <= end.dimensions[name].raw_score
            else end.dimensions[name]
        )
        for name in DIMENSION_WEIGHTS
    }
    dimensions["hook"] = start.dimensions["hook"]
    dimensions["standalone_clarity"] = start.dimensions["standalone_clarity"]
    dimensions["payoff"] = end.dimensions["payoff"]
    return dimensions


def _conservative_penalties(
    start: ModelEvaluation,
    end: ModelEvaluation,
) -> tuple[PenaltyJudgment, ...]:
    by_type: dict[str, PenaltyJudgment] = {}
    for penalty in (*start.penalties, *end.penalties):
        previous = by_type.get(penalty.type)
        if previous is None or penalty.points > previous.points:
            by_type[penalty.type] = penalty
    return tuple(by_type[name] for name in sorted(by_type))


def _combine_evaluations(
    start: ModelEvaluation,
    end: ModelEvaluation,
    *,
    candidate_id: str,
) -> ModelEvaluation:
    central_problem = (
        start.central_idea_problem
        if start.central_idea_problem != "none"
        else end.central_idea_problem
    )
    return ModelEvaluation(
        candidate_id=candidate_id,
        variant_id="recombined",
        start_utterance_id=start.start_utterance_id,
        end_utterance_id=end.end_utterance_id,
        start_boundary_text=start.start_boundary_text,
        end_boundary_text=end.end_boundary_text,
        category=start.category,
        secondary_categories=tuple(
            dict.fromkeys((*start.secondary_categories, *end.secondary_categories))
        ),
        dimensions=_conservative_dimensions(start, end),
        penalties=_conservative_penalties(start, end),
        unsafe_extraction=start.unsafe_extraction or end.unsafe_extraction,
        non_contiguous_required=(
            start.non_contiguous_required or end.non_contiguous_required
        ),
        opening_is_clear=start.opening_is_clear,
        opening_problem=start.opening_problem,
        central_idea_is_strong=(
            start.central_idea_is_strong and end.central_idea_is_strong
        ),
        central_idea_problem=central_problem,
        opening_evidence=start.opening_evidence,
        central_idea_evidence=(
            f"Opening assessment: {start.central_idea_evidence} "
            f"Payoff assessment: {end.central_idea_evidence}"
        ),
        arc_is_complete=end.arc_is_complete,
        payoff_type=end.payoff_type,
        payoff_text=end.payoff_text,
        arc_evidence=end.arc_evidence,
        boundary_rationale=(
            "Deterministically combined the strongest assessed opening with "
            "the strongest assessed ending from overlapping contiguous edits."
        ),
        why_selected="Recombined source-exact boundaries for a stronger complete arc.",
    )


def _candidate_overlap(left: BoundaryContext, right: BoundaryContext) -> float:
    overlap = max(
        0,
        min(left.candidate.end_ms, right.candidate.end_ms)
        - max(left.candidate.start_ms, right.candidate.start_ms),
    )
    shortest = min(
        left.candidate.end_ms - left.candidate.start_ms,
        right.candidate.end_ms - right.candidate.start_ms,
    )
    return overlap / shortest if shortest > 0 else 0.0


def _apply_recombined(
    start: _EvaluatedVariant,
    end: _EvaluatedVariant,
    target: BoundaryContext,
    transcript: CanonicalTranscript,
    max_duration_ms: int,
) -> _EvaluatedVariant | None:
    evaluation = _combine_evaluations(
        start.evaluation,
        end.evaluation,
        candidate_id=target.candidate.id,
    )
    try:
        scored = apply_evaluation(
            evaluation,
            target,
            transcript,
            max_duration_ms=max_duration_ms,
        )
    except GoldMinerError:
        return None
    return _EvaluatedVariant(evaluation, target, scored)


def _rescue_over_duration(
    variant: _EvaluatedVariant,
    transcript: CanonicalTranscript,
    max_duration_ms: int,
) -> _EvaluatedVariant | None:
    scored = variant.scored
    duration_ms = scored.end_ms - scored.start_ms
    if (
        scored.rejection_reasons != ("duration_out_of_bounds",)
        or duration_ms <= max_duration_ms
        or duration_ms > max_duration_ms + MAX_DURATION_RESCUE_OVERAGE_MS
    ):
        return None
    utterance_by_id = {item.id: item for item in transcript.utterances}
    # A transcription provider can place a weak setup and the actual hook in
    # the same long utterance. Search that first utterance too; requiring the
    # aligned result to move forward prevents returning the unchanged edit.
    for utterance_id in scored.source_utterance_ids:
        utterance = utterance_by_id[utterance_id]
        for opening_anchor in opening_anchor_candidates(utterance.original_text):
            evaluation = replace(
                variant.evaluation,
                variant_id="duration_rescue",
                start_utterance_id=utterance.id,
                start_boundary_text=opening_anchor,
                boundary_rationale=(
                    f"Trimmed an over-duration setup at {utterance.id} while preserving "
                    "the assessed payoff and contiguous source text."
                ),
                why_selected="Shortest complete source-exact rescue under the duration ceiling.",
            )
            try:
                repaired = apply_evaluation(
                    evaluation,
                    variant.context,
                    transcript,
                    max_duration_ms=max_duration_ms,
                )
            except GoldMinerError:
                continue
            if (
                not repaired.rejected
                and repaired.start_ms > scored.start_ms
                and repaired.start_ms >= variant.context.candidate.start_ms
            ):
                return _EvaluatedVariant(evaluation, variant.context, repaired)
    return None


def _apply_derived_review(
    variant: _EvaluatedVariant,
    review: DerivedEditReview,
    transcript: CanonicalTranscript,
    max_duration_ms: int,
) -> _EvaluatedVariant:
    if review.candidate_id != variant.scored.id:
        raise GoldMinerError(
            f"Derived review candidate ID {review.candidate_id} does not match "
            f"{variant.scored.id}"
        )
    review.validate()
    evaluation = replace(
        variant.evaluation,
        dimensions=review.dimensions,
        penalties=review.penalties,
        opening_is_clear=review.opening_is_clear,
        opening_problem=review.opening_problem,
        central_idea_is_strong=review.central_idea_is_strong,
        central_idea_problem=review.central_idea_problem,
        opening_evidence=review.opening_evidence,
        central_idea_evidence=review.central_idea_evidence,
        arc_is_complete=review.arc_is_complete,
        payoff_type=review.payoff_type,
        payoff_text=review.payoff_text,
        arc_evidence=review.arc_evidence,
        boundary_rationale=(
            f"{variant.evaluation.boundary_rationale} Final derived edit was "
            "independently reviewed against its exact retained text."
        ),
    )
    return _EvaluatedVariant(
        evaluation,
        variant.context,
        apply_evaluation(
            evaluation,
            variant.context,
            transcript,
            max_duration_ms=max_duration_ms,
        ),
    )


def _apply_identical_range_consensus(
    variants: list[_EvaluatedVariant],
) -> list[_EvaluatedVariant]:
    groups: dict[tuple[int, int, str], list[int]] = {}
    for index, variant in enumerate(variants):
        key = (
            variant.scored.start_ms,
            variant.scored.end_ms,
            " ".join(variant.scored.transcript.casefold().split()),
        )
        groups.setdefault(key, []).append(index)
    updated = list(variants)
    for indexes in groups.values():
        if len(indexes) < 2:
            continue
        judgments = {
            (
                variants[index].evaluation.central_idea_is_strong,
                variants[index].evaluation.central_idea_problem,
            )
            for index in indexes
        }
        if len(judgments) < 2:
            continue
        for index in indexes:
            scored = variants[index].scored
            reasons = tuple(
                dict.fromkeys(
                    (
                        *scored.rejection_reasons,
                        "weak_or_unclear_core_idea",
                        "conflicting_central_idea_assessment",
                    )
                )
            )
            updated[index] = replace(
                variants[index],
                scored=replace(
                    scored,
                    rejected=True,
                    final_score=0.0,
                    rejection_reasons=reasons,
                ),
            )
    return updated


def score_candidates(
    contexts: Sequence[BoundaryContext],
    transcript: CanonicalTranscript,
    provider: ScoringProvider,
    *,
    batch_size: int = SCORING_BATCH_SIZE,
    max_duration_ms: int = 60_000,
    progress=None,
) -> tuple[ScoredCandidate, ...]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    context_by_id = {item.candidate.id: item for item in contexts}
    variants_by_id: dict[str, list[_EvaluatedVariant]] = {
        item.candidate.id: [] for item in contexts
    }
    evaluations_by_id: dict[str, list[ModelEvaluation]] = {
        item.candidate.id: [] for item in contexts
    }
    invalid_by_id: dict[str, list[dict[str, Any]]] = {
        item.candidate.id: [] for item in contexts
    }
    total_batches = (len(contexts) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(contexts), batch_size), start=1):
        batch = contexts[start : start + batch_size]
        if progress:
            progress(f"Boundary scoring batch {batch_number}/{total_batches}: analyzing")
        evaluations = provider.evaluate(batch)
        expected_ids = {item.candidate.id for item in batch}
        pairs = {(item.candidate_id, item.variant_id) for item in evaluations}
        if (
            {item.candidate_id for item in evaluations} != expected_ids
            or len(pairs) != len(evaluations)
        ):
            raise GoldMinerError("Scoring provider returned missing or duplicate candidate edit variants")
        by_id: dict[str, list[ModelEvaluation]] = {}
        for evaluation in evaluations:
            by_id.setdefault(evaluation.candidate_id, []).append(evaluation)
        for context in batch:
            candidate_evaluations = by_id[context.candidate.id]
            evaluations_by_id[context.candidate.id].extend(candidate_evaluations)
            for evaluation in candidate_evaluations:
                try:
                    variants_by_id[context.candidate.id].append(
                        _EvaluatedVariant(
                            evaluation,
                            context,
                            apply_evaluation(
                                evaluation,
                                context,
                                transcript,
                                max_duration_ms=max_duration_ms,
                            ),
                        )
                    )
                except GoldMinerError as exc:
                    invalid_by_id[context.candidate.id].append(
                        _invalid_alternative(evaluation, exc)
                    )
        if progress:
            progress(f"Boundary scoring batch {batch_number}/{total_batches}: completed")

    # Conflicting judgments about the exact same edit resolve conservatively.
    for candidate_id, variants in variants_by_id.items():
        variants_by_id[candidate_id] = _apply_identical_range_consensus(variants)

    # Recombine independently assessed openings and endings inside each idea.
    for candidate_id, original_variants in tuple(variants_by_id.items()):
        context = context_by_id[candidate_id]
        known_ranges = {
            (item.scored.start_ms, item.scored.end_ms, item.scored.transcript)
            for item in original_variants
        }
        for opening in original_variants:
            for ending in original_variants:
                if opening is ending:
                    continue
                recombined = _apply_recombined(
                    opening,
                    ending,
                    context,
                    transcript,
                    max_duration_ms,
                )
                if recombined is None:
                    continue
                key = (
                    recombined.scored.start_ms,
                    recombined.scored.end_ms,
                    recombined.scored.transcript,
                )
                if key not in known_ranges:
                    variants_by_id[candidate_id].append(recombined)
                    known_ranges.add(key)

    # Also allow boundary pieces from strongly overlapping discovery candidates.
    context_pairs = tuple(contexts)
    for left in context_pairs:
        for right in context_pairs:
            if left.candidate.id == right.candidate.id or _candidate_overlap(left, right) < 0.5:
                continue
            target_id = left.candidate.id
            known_ranges = {
                (item.scored.start_ms, item.scored.end_ms, item.scored.transcript)
                for item in variants_by_id[target_id]
            }
            left_originals = tuple(
                item
                for item in variants_by_id[target_id]
                if item.evaluation.variant_id in EDIT_VARIANTS
            )
            right_originals = tuple(
                item
                for item in variants_by_id[right.candidate.id]
                if item.evaluation.variant_id in EDIT_VARIANTS
            )
            for opening in left_originals:
                for ending in right_originals:
                    recombined = _apply_recombined(
                        opening,
                        ending,
                        left,
                        transcript,
                        max_duration_ms,
                    )
                    if recombined is None:
                        continue
                    key = (
                        recombined.scored.start_ms,
                        recombined.scored.end_ms,
                        recombined.scored.transcript,
                    )
                    if key not in known_ranges:
                        variants_by_id[target_id].append(recombined)
                        known_ranges.add(key)

    # Rescue high-quality near misses that failed only because setup pushed them
    # slightly over the duration ceiling.
    for candidate_id, variants in tuple(variants_by_id.items()):
        known_ranges = {
            (item.scored.start_ms, item.scored.end_ms, item.scored.transcript)
            for item in variants
        }
        for variant in tuple(variants):
            rescued = _rescue_over_duration(variant, transcript, max_duration_ms)
            if rescued is None:
                continue
            key = (
                rescued.scored.start_ms,
                rescued.scored.end_ms,
                rescued.scored.transcript,
            )
            if key not in known_ranges:
                variants_by_id[candidate_id].append(rescued)
                known_ranges.add(key)

    # Algorithmic boundary changes can materially alter the opening, topic, or
    # payoff. Re-score whichever derived edit would win, and repeat if that
    # fresh review causes a different derived edit to become the winner.
    for context in contexts:
        candidate_id = context.candidate.id
        reviewed: set[tuple[str, int, int, str]] = set()
        while True:
            variants = variants_by_id[candidate_id]
            if not variants:
                break
            tentative = min(variants, key=lambda item: _selection_key(item.scored))
            review_key = (
                tentative.scored.variant_id,
                tentative.scored.start_ms,
                tentative.scored.end_ms,
                tentative.scored.transcript,
            )
            if (
                tentative.scored.rejected
                or tentative.scored.variant_id not in {"recombined", "duration_rescue"}
                or review_key in reviewed
            ):
                break
            if progress:
                progress(f"Derived edit {candidate_id}: validating exact cut")
            review_values = tuple(
                provider.review_derived(
                    (tentative.scored,),
                    {candidate_id: context.candidate.core_idea},
                )
            )
            if len(review_values) != 1:
                raise GoldMinerError(
                    f"Derived edit reviewer returned {len(review_values)} reviews for {candidate_id}"
                )
            reviewed_variant = _apply_derived_review(
                tentative,
                review_values[0],
                transcript,
                max_duration_ms,
            )
            variants[variants.index(tentative)] = reviewed_variant
            reviewed.add(review_key)
            if progress:
                verdict = "accepted" if not reviewed_variant.scored.rejected else "rejected"
                progress(f"Derived edit {candidate_id}: {verdict}")

    results: list[ScoredCandidate] = []
    variant_order = {
        "hook_first": 0,
        "concise": 1,
        "full_arc": 2,
        "recombined": 3,
        "duration_rescue": 4,
        "primary": 5,
    }
    for context in contexts:
        candidate_id = context.candidate.id
        variants = variants_by_id[candidate_id]
        if not variants:
            # Keep the failed idea as an auditable rejection instead of
            # aborting every other candidate in the video.
            fallback_evaluation = replace(
                evaluations_by_id[candidate_id][0],
                start_boundary_text="",
                end_boundary_text="",
                opening_is_clear=False,
                opening_problem="unresolved_reference",
                arc_is_complete=False,
                payoff_type="none",
            )
            fallback = apply_evaluation(
                fallback_evaluation,
                context,
                transcript,
                max_duration_ms=max_duration_ms,
            )
            variants.append(
                _EvaluatedVariant(
                    fallback_evaluation,
                    context,
                    replace(
                        fallback,
                        variant_id="primary",
                        rejected=True,
                        final_score=0.0,
                        rejection_reasons=tuple(
                            dict.fromkeys((*fallback.rejection_reasons, "invalid_boundary"))
                        ),
                    ),
                )
            )
        winner_variant = min((item.scored for item in variants), key=_selection_key)
        valid_summaries = [
            _alternative_summary(item.scored, selected=item.scored is winner_variant)
            for item in variants
        ]
        alternatives = tuple(
            sorted(
                (*valid_summaries, *invalid_by_id[candidate_id]),
                key=lambda item: variant_order.get(str(item["variant_id"]), 99),
            )
        )
        results.append(replace(winner_variant, alternative_edits=alternatives))
    return tuple(results)


def write_scored_artifact(
    path: Path,
    *,
    candidates_sha256: str,
    provider: ScoringProvider,
    candidates: Sequence[ScoredCandidate],
) -> None:
    _atomic_write(
        path,
        {
            "schema_version": SCORED_ARTIFACT_SCHEMA_VERSION,
            "boundary_version": BOUNDARY_VERSION,
            "scoring_version": SCORING_VERSION,
            "provider": provider.name,
            "model": provider.model,
            "weights": DIMENSION_WEIGHTS,
            "candidates_sha256": candidates_sha256,
            "candidate_count": len(candidates),
            "rejected_count": sum(item.rejected for item in candidates),
            "candidates": [item.to_dict() for item in candidates],
        },
    )


def load_scored_artifact(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["schema_version"] != SCORED_ARTIFACT_SCHEMA_VERSION:
            raise GoldMinerError("Unsupported scored artifact schema version")
        candidates = tuple(raw["candidates"])
        if raw["candidate_count"] != len(candidates):
            raise GoldMinerError("Scored candidate_count does not match candidates")
        if raw["weights"] != DIMENSION_WEIGHTS:
            raise GoldMinerError("Scored artifact weights do not match configured weights")
        return raw, candidates
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GoldMinerError(f"Could not load scored candidates artifact: {exc}") from exc
