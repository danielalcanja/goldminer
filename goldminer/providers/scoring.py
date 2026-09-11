from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Mapping, Protocol, Sequence

from goldminer.analysis.boundaries import BoundaryContext
from goldminer.analysis.scoring import (
    DIMENSION_WEIGHTS,
    EDIT_VARIANTS,
    CENTRAL_IDEA_PROBLEMS,
    DerivedEditReview,
    OPENING_PROBLEMS,
    PAYOFF_TYPES,
    PENALTY_TYPES,
    DimensionJudgment,
    ModelEvaluation,
    PenaltyJudgment,
    ScoredCandidate,
)
from goldminer.errors import GoldMinerError
from goldminer.providers.analysis import JsonTransport, _transport

ARCHETYPES = (
    "story",
    "lesson",
    "tactical",
    "contrarian_take",
    "framework",
    "failure_mistake",
    "transformation",
    "prediction",
    "question_answer",
    "analogy",
    "confession_vulnerability",
    "one_liner_quote",
)


class ScoringProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def evaluate(self, contexts: Sequence[BoundaryContext]) -> Sequence[ModelEvaluation]: ...

    def review_derived(
        self,
        candidates: Sequence[ScoredCandidate],
        original_ideas: Mapping[str, str],
    ) -> Sequence[DerivedEditReview]: ...


@dataclass(frozen=True, slots=True)
class FakeScoringProvider:
    evaluations: dict[str, ModelEvaluation | Sequence[ModelEvaluation]]
    derived_reviews: Mapping[tuple[str, int, int], DerivedEditReview] | None = None
    provider_name: str = "fake"
    model_name: str = "fixture-v1"

    @property
    def name(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return self.model_name

    def evaluate(self, contexts: Sequence[BoundaryContext]) -> Sequence[ModelEvaluation]:
        result: list[ModelEvaluation] = []
        for context in contexts:
            evaluations = self.evaluations[context.candidate.id]
            if isinstance(evaluations, ModelEvaluation):
                result.append(evaluations)
            else:
                result.extend(evaluations)
        return tuple(result)

    def review_derived(
        self,
        candidates: Sequence[ScoredCandidate],
        original_ideas: Mapping[str, str],
    ) -> Sequence[DerivedEditReview]:
        del original_ideas
        reviews: list[DerivedEditReview] = []
        for candidate in candidates:
            key = (candidate.id, candidate.start_ms, candidate.end_ms)
            configured = self.derived_reviews.get(key) if self.derived_reviews else None
            reviews.append(
                configured
                or DerivedEditReview(
                    candidate_id=candidate.id,
                    dimensions=candidate.dimensions,
                    penalties=candidate.penalties,
                    opening_is_clear=candidate.opening_is_clear,
                    opening_problem=candidate.opening_problem,
                    central_idea_is_strong=candidate.central_idea_is_strong,
                    central_idea_problem=candidate.central_idea_problem,
                    opening_evidence=candidate.opening_evidence,
                    central_idea_evidence=candidate.central_idea_evidence,
                    arc_is_complete=candidate.arc_is_complete,
                    payoff_type=candidate.payoff_type,
                    payoff_text=candidate.payoff_text,
                    arc_evidence=candidate.arc_evidence,
                )
            )
        return tuple(reviews)


def _dimension_schema() -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            name: {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "raw_score": {"type": "integer", "minimum": 0, "maximum": 10},
                    "evidence": {"type": "string"},
                },
                "required": ["raw_score", "evidence"],
            }
            for name in DIMENSION_WEIGHTS
        },
        "required": list(DIMENSION_WEIGHTS),
    }


EVALUATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "variant_id": {"type": "string", "enum": list(EDIT_VARIANTS)},
                    "start_utterance_id": {"type": "string"},
                    "end_utterance_id": {"type": "string"},
                    "start_boundary_text": {"type": "string", "minLength": 1},
                    "end_boundary_text": {"type": "string", "minLength": 1},
                    "category": {"type": "string", "enum": list(ARCHETYPES)},
                    "secondary_categories": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(ARCHETYPES)},
                    },
                    "dimensions": _dimension_schema(),
                    "penalties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string", "enum": sorted(PENALTY_TYPES)},
                                "points": {"type": "number", "minimum": 0, "maximum": 25},
                                "evidence": {"type": "string"},
                            },
                            "required": ["type", "points", "evidence"],
                        },
                    },
                    "unsafe_extraction": {"type": "boolean"},
                    "non_contiguous_required": {"type": "boolean"},
                    "opening_is_clear": {"type": "boolean"},
                    "opening_problem": {"type": "string", "enum": sorted(OPENING_PROBLEMS)},
                    "central_idea_is_strong": {"type": "boolean"},
                    "central_idea_problem": {"type": "string", "enum": sorted(CENTRAL_IDEA_PROBLEMS)},
                    "opening_evidence": {"type": "string"},
                    "central_idea_evidence": {"type": "string"},
                    "arc_is_complete": {"type": "boolean"},
                    "payoff_type": {"type": "string", "enum": sorted(PAYOFF_TYPES)},
                    "payoff_text": {"type": "string", "minLength": 1},
                    "arc_evidence": {"type": "string", "minLength": 1},
                    "boundary_rationale": {"type": "string"},
                    "why_selected": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "variant_id",
                    "start_utterance_id",
                    "end_utterance_id",
                    "start_boundary_text",
                    "end_boundary_text",
                    "category",
                    "secondary_categories",
                    "dimensions",
                    "penalties",
                    "unsafe_extraction",
                    "non_contiguous_required",
                    "opening_is_clear",
                    "opening_problem",
                    "central_idea_is_strong",
                    "central_idea_problem",
                    "opening_evidence",
                    "central_idea_evidence",
                    "arc_is_complete",
                    "payoff_type",
                    "payoff_text",
                    "arc_evidence",
                    "boundary_rationale",
                    "why_selected",
                ],
            },
        }
    },
    "required": ["evaluations"],
}


DERIVED_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "dimensions": _dimension_schema(),
                    "penalties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string", "enum": sorted(PENALTY_TYPES)},
                                "points": {"type": "number", "minimum": 0, "maximum": 25},
                                "evidence": {"type": "string"},
                            },
                            "required": ["type", "points", "evidence"],
                        },
                    },
                    "opening_is_clear": {"type": "boolean"},
                    "opening_problem": {"type": "string", "enum": sorted(OPENING_PROBLEMS)},
                    "central_idea_is_strong": {"type": "boolean"},
                    "central_idea_problem": {
                        "type": "string",
                        "enum": sorted(CENTRAL_IDEA_PROBLEMS),
                    },
                    "opening_evidence": {"type": "string"},
                    "central_idea_evidence": {"type": "string"},
                    "arc_is_complete": {"type": "boolean"},
                    "payoff_type": {"type": "string", "enum": sorted(PAYOFF_TYPES)},
                    "payoff_text": {"type": "string"},
                    "arc_evidence": {"type": "string"},
                },
                "required": [
                    "candidate_id",
                    "dimensions",
                    "penalties",
                    "opening_is_clear",
                    "opening_problem",
                    "central_idea_is_strong",
                    "central_idea_problem",
                    "opening_evidence",
                    "central_idea_evidence",
                    "arc_is_complete",
                    "payoff_type",
                    "payoff_text",
                    "arc_evidence",
                ],
            },
        }
    },
    "required": ["reviews"],
}


@dataclass(frozen=True, slots=True)
class OpenAIScoringProvider:
    api_key: str
    model_name: str = "gpt-5.4-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 300.0
    max_retries: int = 3
    transport: JsonTransport = _transport

    @classmethod
    def from_environment(cls) -> "OpenAIScoringProvider":
        return cls(
            api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            model_name=os.environ.get("OPENAI_SCORING_MODEL", os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-5.4-mini")).strip(),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        )

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self.model_name

    def _extract_text(self, response: dict[str, object]) -> str:
        if isinstance(response.get("output_text"), str):
            return response["output_text"]  # type: ignore[return-value]
        for item in response.get("output", []) if isinstance(response.get("output"), list) else []:
            if isinstance(item, dict) and isinstance(item.get("content"), list):
                for part in item["content"]:
                    if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        return part["text"]
        raise GoldMinerError("OpenAI scoring response contained no output text")

    def _request_structured(self, body: bytes, *, operation: str) -> dict[str, object]:
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
                message = json.loads(payload).get("error", {}).get(
                    "message", payload.decode()
                )
            except (json.JSONDecodeError, AttributeError):
                message = payload.decode(errors="replace") or "unknown API error"
            raise GoldMinerError(f"OpenAI {operation} failed with HTTP {status}: {message}")
        try:
            response = json.loads(payload)
            structured = json.loads(self._extract_text(response))
        except (json.JSONDecodeError, TypeError) as exc:
            raise GoldMinerError(
                f"OpenAI returned invalid structured {operation} JSON"
            ) from exc
        if not isinstance(structured, dict):
            raise GoldMinerError(f"OpenAI {operation} response must be an object")
        return structured

    def _parse_evaluation(self, value: dict[str, object]) -> ModelEvaluation:
        try:
            dimensions = {
                name: DimensionJudgment(
                    raw_score=value["dimensions"][name]["raw_score"],  # type: ignore[index]
                    evidence=value["dimensions"][name]["evidence"],  # type: ignore[index]
                )
                for name in DIMENSION_WEIGHTS
            }
            penalties = tuple(
                PenaltyJudgment(item["type"], item["points"], item["evidence"])
                for item in value["penalties"]  # type: ignore[union-attr]
            )
            evaluation = ModelEvaluation(
                candidate_id=value["candidate_id"],  # type: ignore[arg-type]
                variant_id=value["variant_id"],  # type: ignore[arg-type]
                start_utterance_id=value["start_utterance_id"],  # type: ignore[arg-type]
                end_utterance_id=value["end_utterance_id"],  # type: ignore[arg-type]
                start_boundary_text=value["start_boundary_text"],  # type: ignore[arg-type]
                end_boundary_text=value["end_boundary_text"],  # type: ignore[arg-type]
                category=value["category"],  # type: ignore[arg-type]
                secondary_categories=tuple(value["secondary_categories"]),  # type: ignore[arg-type]
                dimensions=dimensions,
                penalties=penalties,
                unsafe_extraction=value["unsafe_extraction"],  # type: ignore[arg-type]
                non_contiguous_required=value["non_contiguous_required"],  # type: ignore[arg-type]
                opening_is_clear=value["opening_is_clear"],  # type: ignore[arg-type]
                opening_problem=value["opening_problem"],  # type: ignore[arg-type]
                central_idea_is_strong=value["central_idea_is_strong"],  # type: ignore[arg-type]
                central_idea_problem=value["central_idea_problem"],  # type: ignore[arg-type]
                opening_evidence=value["opening_evidence"],  # type: ignore[arg-type]
                central_idea_evidence=value["central_idea_evidence"],  # type: ignore[arg-type]
                arc_is_complete=value["arc_is_complete"],  # type: ignore[arg-type]
                payoff_type=value["payoff_type"],  # type: ignore[arg-type]
                payoff_text=value["payoff_text"],  # type: ignore[arg-type]
                arc_evidence=value["arc_evidence"],  # type: ignore[arg-type]
                boundary_rationale=value["boundary_rationale"],  # type: ignore[arg-type]
                why_selected=value["why_selected"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError) as exc:
            raise GoldMinerError(f"Malformed scoring evaluation: {exc}") from exc
        evaluation.validate()
        if evaluation.category not in ARCHETYPES or any(item not in ARCHETYPES for item in evaluation.secondary_categories):
            raise GoldMinerError(f"Unsupported category in evaluation {evaluation.candidate_id}")
        if not isinstance(evaluation.unsafe_extraction, bool) or not isinstance(evaluation.non_contiguous_required, bool):
            raise GoldMinerError("Hard rejection flags must be booleans")
        return evaluation

    def _parse_derived_review(self, value: dict[str, object]) -> DerivedEditReview:
        try:
            review = DerivedEditReview(
                candidate_id=value["candidate_id"],  # type: ignore[arg-type]
                dimensions={
                    name: DimensionJudgment(
                        raw_score=value["dimensions"][name]["raw_score"],  # type: ignore[index]
                        evidence=value["dimensions"][name]["evidence"],  # type: ignore[index]
                    )
                    for name in DIMENSION_WEIGHTS
                },
                penalties=tuple(
                    PenaltyJudgment(item["type"], item["points"], item["evidence"])
                    for item in value["penalties"]  # type: ignore[union-attr]
                ),
                opening_is_clear=value["opening_is_clear"],  # type: ignore[arg-type]
                opening_problem=value["opening_problem"],  # type: ignore[arg-type]
                central_idea_is_strong=value["central_idea_is_strong"],  # type: ignore[arg-type]
                central_idea_problem=value["central_idea_problem"],  # type: ignore[arg-type]
                opening_evidence=value["opening_evidence"],  # type: ignore[arg-type]
                central_idea_evidence=value["central_idea_evidence"],  # type: ignore[arg-type]
                arc_is_complete=value["arc_is_complete"],  # type: ignore[arg-type]
                payoff_type=value["payoff_type"],  # type: ignore[arg-type]
                payoff_text=value["payoff_text"],  # type: ignore[arg-type]
                arc_evidence=value["arc_evidence"],  # type: ignore[arg-type]
            )
        except (KeyError, TypeError) as exc:
            raise GoldMinerError(f"Malformed derived edit review: {exc}") from exc
        review.validate()
        return review

    def evaluate(self, contexts: Sequence[BoundaryContext]) -> Sequence[ModelEvaluation]:
        if not self.api_key or self.api_key == "replace-with-new-key":
            raise GoldMinerError("OPENAI_API_KEY is not configured for scoring")
        if not contexts:
            return ()
        sections = []
        for context in contexts:
            candidate = context.candidate
            sections.append(
                f"CANDIDATE {candidate.id}\nOriginal proposed range: "
                f"{candidate.source_utterance_ids[0]}-{candidate.source_utterance_ids[-1]}\n"
                f"Discovery idea: {candidate.core_idea}\nAllowed contiguous context:\n{context.prompt_text()}"
            )
        instructions = (
            "Act as a rigorous short-form clip editor. Return exactly three distinct edits for every candidate: "
            "hook_first starts at the strongest cold-viewer hook; concise is the shortest complete useful edit; "
            "full_arc preserves the strongest setup, development, and payoff. Set variant_id accordingly. Do not "
            "return three copies of the same boundaries unless no materially different valid edit exists. For each "
            "edit, repair start and end only "
            "within its allowed contiguous utterance context. You may also trim inside the first and last "
            "utterances: start_boundary_text must be an exact consecutive quote identifying the first spoken "
            "words to retain, and end_boundary_text must be an exact consecutive quote identifying the last "
            "spoken words to retain. Copy both quotes verbatim from their named utterances; use enough words to "
            "identify each occurrence unambiguously. Search the entire allowed context for the best version, "
            "not merely the original proposed range. When neighboring utterances supply the missing cold hook, "
            "contrast, resolution, or payoff, extend into them and return one complete narrative arc instead of "
            "a partial fragment. Prefer a natural hook, complete development, "
            "and immediate payoff, usually 20-45 seconds and never longer than 60 seconds. Treat the exact "
            "first retained phrase as the first thing a cold viewer hears. Classify opening_problem as none, "
            "continuation, unresolved_reference, housekeeping, or unrelated_setup. Only use none when a cold "
            "viewer can immediately understand the opening; opening_is_clear must be false for every other value. "
            "Set opening_is_clear=false if it "
            "depends on earlier speech, starts with an unresolved continuation (such as that, and, but, because, "
            "or so), uses an unclear pronoun, or begins with housekeeping or unrelated setup. Set "
            "Classify central_idea_problem as none, generic, incomplete, garbled, or multiple_competing_ideas. "
            "central_idea_is_strong can be true only when that value is none and the range communicates one specific, worthwhile claim, story, "
            "or lesson and completes its payoff. Generic, garbled, contradictory, transcript-uncertain, or unfinished "
            "speech is not a strong central idea. Classify payoff_type as conclusion, result, actionable_takeaway, "
            "punchline, reveal, or none, and copy the explicit payoff_text verbatim from the selected range. "
            "arc_is_complete may be true only when the setup is explicitly resolved by that payoff. Do not count "
            "the second half of a contrast, an implied solution, or an unfinished next thought as a payoff. First try "
            "to repair any problem by trimming or extending within "
            "the allowed context; if that cannot produce a coherent range within 60 seconds, set the relevant flag "
            "false. A candidate whose whole utterance range exceeds 60 seconds can still be valid when exact "
            "phrase trimming brings the complete arc under 60 seconds. Score every "
            "dimension from 0-10 without inflation and cite one or two evidence sentences. Apply explicit "
            "penalties separately. Mark unsafe extraction when removing qualifications changes meaning, and "
            "mark non_contiguous_required when the idea only works by splicing separate ranges."
        )
        body = json.dumps(
            {
                "model": self.model_name,
                "instructions": instructions,
                "input": "\n\n".join(sections),
                "text": {"format": {"type": "json_schema", "name": "clip_scoring", "strict": True, "schema": EVALUATION_SCHEMA}},
                "store": False,
            }
        ).encode()
        try:
            structured = self._request_structured(body, operation="scoring")
            raw = structured["evaluations"]
        except (KeyError, TypeError) as exc:
            raise GoldMinerError("OpenAI returned invalid structured scoring JSON") from exc
        if not isinstance(raw, list):
            raise GoldMinerError("Scoring evaluations must be an array")
        evaluations = tuple(self._parse_evaluation(item) for item in raw)
        expected = {
            (item.candidate.id, variant_id)
            for item in contexts
            for variant_id in EDIT_VARIANTS
        }
        actual = {(item.candidate_id, item.variant_id) for item in evaluations}
        if actual != expected or len(evaluations) != len(expected):
            raise GoldMinerError(
                "Scoring response candidate/variant IDs mismatch; "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
        return evaluations

    def review_derived(
        self,
        candidates: Sequence[ScoredCandidate],
        original_ideas: Mapping[str, str],
    ) -> Sequence[DerivedEditReview]:
        if not self.api_key or self.api_key == "replace-with-new-key":
            raise GoldMinerError("OPENAI_API_KEY is not configured for derived edit review")
        if not candidates:
            return ()
        sections = []
        for candidate in candidates:
            sections.append(
                f"CANDIDATE {candidate.id}\n"
                f"Original discovery idea: {original_ideas[candidate.id]}\n"
                f"Exact proposed clip ({candidate.start_ms}-{candidate.end_ms}ms):\n"
                f"{candidate.transcript}"
            )
        instructions = (
            "Act as the final senior-editor quality gate for algorithmically trimmed clips. Review exactly the "
            "text supplied; do not suggest new boundaries and do not rely on an earlier version of the clip. "
            "Judge the literal first words as a cold opening and the literal last words as the ending. Mark an "
            "opening unclear when it begins mid-sentence, with a continuation, unresolved pronoun, or backward "
            "reference. Mark central_idea_problem=multiple_competing_ideas when the clip changes subjects or "
            "combines two independently postable lessons, even when both concern the broad original topic. A "
            "complete arc must contain one setup, development, and an explicit payoff that resolves that same "
            "idea. Copy payoff_text exactly from the supplied clip, or use an empty string with payoff_type=none. "
            "Score all dimensions again from 0-10 using evidence from this exact edit. Do not preserve or infer "
            "scores from any earlier candidate. Return one review for every candidate ID."
        )
        body = json.dumps(
            {
                "model": self.model_name,
                "instructions": instructions,
                "input": "\n\n".join(sections),
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "derived_clip_review",
                        "strict": True,
                        "schema": DERIVED_REVIEW_SCHEMA,
                    }
                },
                "store": False,
            }
        ).encode()
        try:
            structured = self._request_structured(body, operation="derived edit review")
            raw = structured["reviews"]
        except (KeyError, TypeError) as exc:
            raise GoldMinerError(
                "OpenAI returned invalid structured derived edit review JSON"
            ) from exc
        if not isinstance(raw, list):
            raise GoldMinerError("Derived edit reviews must be an array")
        reviews = tuple(self._parse_derived_review(item) for item in raw)
        expected = {item.id for item in candidates}
        actual = {item.candidate_id for item in reviews}
        if actual != expected or len(reviews) != len(expected):
            raise GoldMinerError(
                "Derived edit review candidate IDs mismatch; "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
        return reviews
