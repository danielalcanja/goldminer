from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from goldminer.errors import GoldMinerError

SCORING_VERSION = "1.8"
BOUNDARY_VERSION = "1.3"
MODEL_JUDGMENT_VERSION = "1.4"
DERIVED_REVIEW_VERSION = "1.0"

EDIT_VARIANTS = ("hook_first", "concise", "full_arc")
INTERNAL_EDIT_VARIANTS = ("recombined", "duration_rescue")

DIMENSION_WEIGHTS: dict[str, int] = {
    "hook": 15,
    "standalone_clarity": 15,
    "payoff": 15,
    "value": 15,
    "emotion": 10,
    "novelty": 10,
    "shareability": 10,
    "audience_fit": 5,
    "delivery_density": 5,
}

PENALTY_TYPES = {
    "context_dependence",
    "slow_setup",
    "repetition_filler",
    "no_payoff",
    "generic_advice",
}

OPENING_PROBLEMS = {
    "none",
    "continuation",
    "unresolved_reference",
    "housekeeping",
    "unrelated_setup",
}

CENTRAL_IDEA_PROBLEMS = {
    "none",
    "generic",
    "incomplete",
    "garbled",
    "multiple_competing_ideas",
}

PAYOFF_TYPES = {
    "none",
    "conclusion",
    "result",
    "actionable_takeaway",
    "punchline",
    "reveal",
}


@dataclass(frozen=True, slots=True)
class DimensionJudgment:
    raw_score: int
    evidence: str

    def validate(self, name: str) -> None:
        if isinstance(self.raw_score, bool) or not isinstance(self.raw_score, int):
            raise GoldMinerError(f"Score {name} must be an integer")
        if not 0 <= self.raw_score <= 10:
            raise GoldMinerError(f"Score {name} must be between 0 and 10")
        if not self.evidence.strip():
            raise GoldMinerError(f"Score {name} requires evidence")


@dataclass(frozen=True, slots=True)
class PenaltyJudgment:
    type: str
    points: float
    evidence: str

    def validate(self) -> None:
        if self.type not in PENALTY_TYPES:
            raise GoldMinerError(f"Unsupported penalty type: {self.type}")
        if isinstance(self.points, bool) or not isinstance(self.points, (int, float)):
            raise GoldMinerError(f"Penalty {self.type} points must be numeric")
        if not 0 <= self.points <= 25:
            raise GoldMinerError(f"Penalty {self.type} must be between 0 and 25")
        if not self.evidence.strip():
            raise GoldMinerError(f"Penalty {self.type} requires evidence")


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    candidate_id: str
    start_utterance_id: str
    end_utterance_id: str
    category: str
    secondary_categories: tuple[str, ...]
    dimensions: Mapping[str, DimensionJudgment]
    penalties: tuple[PenaltyJudgment, ...]
    unsafe_extraction: bool
    non_contiguous_required: bool
    boundary_rationale: str
    why_selected: str
    opening_is_clear: bool = True
    central_idea_is_strong: bool = True
    opening_evidence: str = "Opening is understandable without prior context."
    central_idea_evidence: str = "The clip communicates a complete central idea."
    start_boundary_text: str = ""
    end_boundary_text: str = ""
    opening_problem: str = "none"
    central_idea_problem: str = "none"
    arc_is_complete: bool = True
    payoff_type: str = "conclusion"
    payoff_text: str = ""
    arc_evidence: str = "The setup is resolved by an explicit payoff."
    variant_id: str = "primary"

    def validate(self) -> None:
        if set(self.dimensions) != set(DIMENSION_WEIGHTS):
            missing = set(DIMENSION_WEIGHTS).difference(self.dimensions)
            extra = set(self.dimensions).difference(DIMENSION_WEIGHTS)
            raise GoldMinerError(f"Score dimensions mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        for name, judgment in self.dimensions.items():
            judgment.validate(name)
        for penalty in self.penalties:
            penalty.validate()
        if not isinstance(self.opening_is_clear, bool) or not isinstance(self.central_idea_is_strong, bool):
            raise GoldMinerError("Editorial quality flags must be booleans")
        if not self.opening_evidence.strip() or not self.central_idea_evidence.strip():
            raise GoldMinerError("Opening and central-idea assessments require evidence")
        if not isinstance(self.start_boundary_text, str) or not isinstance(self.end_boundary_text, str):
            raise GoldMinerError("Phrase boundary text must be strings")
        if self.opening_problem not in OPENING_PROBLEMS:
            raise GoldMinerError(f"Unsupported opening problem: {self.opening_problem}")
        if self.central_idea_problem not in CENTRAL_IDEA_PROBLEMS:
            raise GoldMinerError(f"Unsupported central idea problem: {self.central_idea_problem}")
        if not isinstance(self.arc_is_complete, bool):
            raise GoldMinerError("Narrative arc flag must be a boolean")
        if self.payoff_type not in PAYOFF_TYPES:
            raise GoldMinerError(f"Unsupported payoff type: {self.payoff_type}")
        if not isinstance(self.payoff_text, str) or not self.arc_evidence.strip():
            raise GoldMinerError("Narrative payoff assessment requires text evidence")
        if self.variant_id not in {"primary", *EDIT_VARIANTS, *INTERNAL_EDIT_VARIANTS}:
            raise GoldMinerError(f"Unsupported edit variant: {self.variant_id}")
        if not self.boundary_rationale.strip() or not self.why_selected.strip():
            raise GoldMinerError("Boundary rationale and selection explanation are required")


@dataclass(frozen=True, slots=True)
class DerivedEditReview:
    """A fresh content judgment for an algorithmically-created final edit."""

    candidate_id: str
    dimensions: Mapping[str, DimensionJudgment]
    penalties: tuple[PenaltyJudgment, ...]
    opening_is_clear: bool
    opening_problem: str
    central_idea_is_strong: bool
    central_idea_problem: str
    opening_evidence: str
    central_idea_evidence: str
    arc_is_complete: bool
    payoff_type: str
    payoff_text: str
    arc_evidence: str

    def validate(self) -> None:
        if not self.candidate_id.strip():
            raise GoldMinerError("Derived edit review requires a candidate ID")
        if set(self.dimensions) != set(DIMENSION_WEIGHTS):
            missing = set(DIMENSION_WEIGHTS).difference(self.dimensions)
            extra = set(self.dimensions).difference(DIMENSION_WEIGHTS)
            raise GoldMinerError(
                f"Derived review dimensions mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name, judgment in self.dimensions.items():
            judgment.validate(name)
        for penalty in self.penalties:
            penalty.validate()
        if not isinstance(self.opening_is_clear, bool) or not isinstance(
            self.central_idea_is_strong, bool
        ):
            raise GoldMinerError("Derived editorial quality flags must be booleans")
        if self.opening_problem not in OPENING_PROBLEMS:
            raise GoldMinerError(f"Unsupported opening problem: {self.opening_problem}")
        if self.central_idea_problem not in CENTRAL_IDEA_PROBLEMS:
            raise GoldMinerError(
                f"Unsupported central idea problem: {self.central_idea_problem}"
            )
        if not self.opening_evidence.strip() or not self.central_idea_evidence.strip():
            raise GoldMinerError("Derived opening and central-idea reviews require evidence")
        if not isinstance(self.arc_is_complete, bool):
            raise GoldMinerError("Derived narrative arc flag must be a boolean")
        if self.payoff_type not in PAYOFF_TYPES:
            raise GoldMinerError(f"Unsupported payoff type: {self.payoff_type}")
        if not isinstance(self.payoff_text, str) or not self.arc_evidence.strip():
            raise GoldMinerError("Derived narrative payoff review requires text evidence")


def calculate_score(
    dimensions: Mapping[str, DimensionJudgment], penalties: tuple[PenaltyJudgment, ...]
) -> tuple[float, float, dict[str, float]]:
    if sum(DIMENSION_WEIGHTS.values()) != 100:
        raise GoldMinerError("Scoring weights must sum to 100")
    if set(dimensions) != set(DIMENSION_WEIGHTS):
        raise GoldMinerError("All configured score dimensions are required")
    weighted_points: dict[str, float] = {}
    for name, weight in DIMENSION_WEIGHTS.items():
        dimensions[name].validate(name)
        weighted_points[name] = round(dimensions[name].raw_score * weight / 10, 2)
    for penalty in penalties:
        penalty.validate()
    subtotal = round(sum(weighted_points.values()), 2)
    final = round(max(0.0, min(100.0, subtotal - sum(item.points for item in penalties))), 2)
    return subtotal, final, weighted_points


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    id: str
    start_ms: int
    end_ms: int
    source_utterance_ids: tuple[str, ...]
    transcript: str
    category: str
    secondary_categories: tuple[str, ...]
    dimensions: Mapping[str, DimensionJudgment]
    penalties: tuple[PenaltyJudgment, ...]
    weighted_subtotal: float
    final_score: float
    rejected: bool
    rejection_reasons: tuple[str, ...]
    boundary_rationale: str
    why_selected: str
    opening_is_clear: bool = True
    central_idea_is_strong: bool = True
    opening_evidence: str = ""
    central_idea_evidence: str = ""
    start_boundary_text: str = ""
    end_boundary_text: str = ""
    opening_problem: str = "none"
    central_idea_problem: str = "none"
    arc_is_complete: bool = True
    payoff_type: str = "conclusion"
    payoff_text: str = ""
    arc_evidence: str = ""
    variant_id: str = "primary"
    alternative_edits: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        _, _, points = calculate_score(self.dimensions, self.penalties)
        return {
            "id": self.id,
            "selected_variant": self.variant_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_sec": round((self.end_ms - self.start_ms) / 1000, 3),
            "source_utterance_ids": list(self.source_utterance_ids),
            "transcript": self.transcript,
            "category": self.category,
            "secondary_categories": list(self.secondary_categories),
            "scores": {
                name: {
                    "raw_score": judgment.raw_score,
                    "weight": DIMENSION_WEIGHTS[name],
                    "weighted_points": points[name],
                    "evidence": judgment.evidence,
                }
                for name, judgment in self.dimensions.items()
            },
            "penalties": [
                {"type": item.type, "points": item.points, "evidence": item.evidence}
                for item in self.penalties
            ],
            "weighted_subtotal": self.weighted_subtotal,
            "final_score": self.final_score,
            "rejected": self.rejected,
            "rejection_reasons": list(self.rejection_reasons),
            "boundary_rationale": self.boundary_rationale,
            "why_selected": self.why_selected,
            "editorial_quality": {
                "opening_is_clear": self.opening_is_clear,
                "opening_evidence": self.opening_evidence,
                "opening_problem": self.opening_problem,
                "central_idea_is_strong": self.central_idea_is_strong,
                "central_idea_evidence": self.central_idea_evidence,
                "central_idea_problem": self.central_idea_problem,
            },
            "phrase_boundaries": {
                "start_text": self.start_boundary_text,
                "end_text": self.end_boundary_text,
            },
            "narrative_arc": {
                "is_complete": self.arc_is_complete,
                "payoff_type": self.payoff_type,
                "payoff_text": self.payoff_text,
                "evidence": self.arc_evidence,
            },
            "alternative_edits": [dict(item) for item in self.alternative_edits],
            "boundary_version": BOUNDARY_VERSION,
            "scoring_version": SCORING_VERSION,
        }
