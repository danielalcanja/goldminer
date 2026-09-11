from __future__ import annotations

from dataclasses import dataclass
import re

from goldminer.analysis.discovery import canonical_utterance_id
from goldminer.analysis.models import Candidate
from goldminer.analysis.scoring import ModelEvaluation, ScoredCandidate, calculate_score
from goldminer.errors import GoldMinerError
from goldminer.transcript.models import CanonicalTranscript, Utterance


_WORD_RE = re.compile(r"\b\w+(?:['’]\w+)*\b", re.UNICODE)
_CONTEXTUAL_OPENING_RE = re.compile(
    r"^\s*(?:and\b|but\b|so\b|that\b|"
    r"(?:it|this)\s+(?:can|could|is|was|will|would|has|feels)\b|"
    r"(?:it|this)['’]s\b|"
    r"(?:review|test|use|do|make|take|put|move|finish)\s+(?:it|that|this)\b|"
    r"what\s+i\s+(?:see|saw|said|mean)\s+(?:there|before|here)\b|"
    r"right[\s,?.!]+(?:that|this|it|so|and|but)\b|"
    r"yeah[\s,?.!]+(?:that|and|but|so)\b)",
    re.IGNORECASE,
)
_INCOMPLETE_ENDING_RE = re.compile(
    r"(?:[,;:\-—]|\.\.\.)\s*$|"
    r"\b(?:and|but|because|so|that|which|when|if|to|of)\s*[.!?]?\s*$|"
    r"\b(?:go|do|put|get|take|move|start|finish)\s+(?:there|that|this|it)\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_LEADING_CONTEXT_RE = re.compile(
    r"^\s*(?:(?:yeah|right|well|okay|ok|and|but|so|that)\b[\s,;:.!?\-—]*)+",
    re.IGNORECASE,
)
_STRONG_OPENING_ANCHOR_RE = re.compile(
    r"\b(?:we\s+don['’]t\s+know|i\s+(?:just\s+)?(?:want|think|believe|feel|recommend)|"
    r"the\s+(?:goal|point|problem|lesson|reason|key)\s+(?:is|was)|"
    r"you\s+(?:need|should|can)\b)",
    re.IGNORECASE,
)


def trim_contextual_prefix(text: str) -> str:
    """Return an exact source suffix after disposable conversational connectives."""
    trimmed = _LEADING_CONTEXT_RE.sub("", text).strip()
    return trimmed if len(_WORD_RE.findall(trimmed)) >= 3 else text.strip()


def opening_anchor_candidates(text: str) -> tuple[str, ...]:
    """Return source-exact suffixes that may form an independent cold opening."""
    values = [trim_contextual_prefix(text)]
    values.extend(text[match.start() :].strip() for match in _STRONG_OPENING_ANCHOR_RE.finditer(text))
    return tuple(dict.fromkeys(value for value in values if value))


@dataclass(frozen=True, slots=True)
class _PhraseSpan:
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_count: int


def _token_key(value: str) -> str:
    return value.casefold().replace("’", "'")


def _find_phrase_span(text: str, phrase: str, *, use_last: bool) -> _PhraseSpan:
    text_matches = tuple(_WORD_RE.finditer(text))
    phrase_matches = tuple(_WORD_RE.finditer(phrase))
    if not text_matches or not phrase_matches:
        raise GoldMinerError(f"Phrase boundary has no alignable words: {phrase!r}")
    text_tokens = tuple(_token_key(item.group()) for item in text_matches)
    phrase_tokens = tuple(_token_key(item.group()) for item in phrase_matches)
    def positions(anchor: tuple[str, ...]) -> list[int]:
        return [
            index
            for index in range(len(text_tokens) - len(anchor) + 1)
            if text_tokens[index : index + len(anchor)] == anchor
        ]

    anchor_tokens = phrase_tokens
    matches = positions(anchor_tokens)
    # A natural phrase can cross one of the transcription provider's segment
    # boundaries. Retain only the exact portion inside the named boundary
    # utterance: the prefix for a start, or the suffix for an end.
    minimum_anchor = min(3, len(phrase_tokens))
    if not matches:
        for anchor_length in range(len(phrase_tokens) - 1, minimum_anchor - 1, -1):
            anchor_tokens = (
                phrase_tokens[-anchor_length:] if use_last else phrase_tokens[:anchor_length]
            )
            matches = positions(anchor_tokens)
            if matches:
                break
    if not matches:
        raise GoldMinerError(f"Phrase boundary could not be aligned to its utterance: {phrase!r}")
    token_start = matches[-1] if use_last else matches[0]
    token_end = token_start + len(anchor_tokens)
    char_end = text_matches[token_end - 1].end()
    while char_end < len(text) and text[char_end] in ".,!?;:)]}\"'’”":
        char_end += 1
    return _PhraseSpan(
        char_start=text_matches[token_start].start(),
        char_end=char_end,
        token_start=token_start,
        token_end=token_end,
        token_count=len(text_matches),
    )


def _full_span(text: str) -> _PhraseSpan:
    matches = tuple(_WORD_RE.finditer(text))
    if not matches:
        raise GoldMinerError("Selected utterance contains no alignable words")
    return _PhraseSpan(0, len(text), 0, len(matches), len(matches))


def _interpolated_ms(utterance: Utterance, token_position: int, token_count: int) -> int:
    duration = utterance.end_ms - utterance.start_ms
    return utterance.start_ms + round(duration * token_position / token_count)


@dataclass(frozen=True, slots=True)
class BoundaryContext:
    candidate: Candidate
    utterances: tuple[Utterance, ...]

    def prompt_text(self) -> str:
        return "\n".join(
            f"[{item.id}] {item.start_ms}-{item.end_ms}ms {item.speaker or 'Unknown'}: {item.original_text}"
            for item in self.utterances
        )


def build_boundary_context(
    candidate: Candidate,
    transcript: CanonicalTranscript,
    *,
    padding_utterances: int = 5,
    padding_ms: int | None = None,
) -> BoundaryContext:
    index = {item.id: position for position, item in enumerate(transcript.utterances)}
    start = index[candidate.source_utterance_ids[0]]
    end = index[candidate.source_utterance_ids[-1]]
    context_start = max(0, start - padding_utterances)
    context_end = min(len(transcript.utterances), end + padding_utterances + 1)
    if padding_ms is not None:
        minimum_start = candidate.start_ms - padding_ms
        maximum_end = candidate.end_ms + padding_ms
        while context_start > 0 and transcript.utterances[context_start - 1].end_ms >= minimum_start:
            context_start -= 1
        while context_end < len(transcript.utterances) and transcript.utterances[context_end].start_ms <= maximum_end:
            context_end += 1
    return BoundaryContext(candidate, transcript.utterances[context_start:context_end])


def apply_evaluation(
    evaluation: ModelEvaluation,
    context: BoundaryContext,
    transcript: CanonicalTranscript,
    *,
    max_duration_ms: int = 60_000,
) -> ScoredCandidate:
    evaluation.validate()
    if evaluation.candidate_id != context.candidate.id:
        raise GoldMinerError(
            f"Evaluation candidate ID {evaluation.candidate_id} does not match {context.candidate.id}"
        )
    start_id = canonical_utterance_id(evaluation.start_utterance_id)
    end_id = canonical_utterance_id(evaluation.end_utterance_id)
    allowed = {item.id for item in context.utterances}
    if start_id not in allowed or end_id not in allowed:
        raise GoldMinerError(f"Boundary repair for {evaluation.candidate_id} escaped its allowed context")
    index = {item.id: position for position, item in enumerate(transcript.utterances)}
    start, end = index[start_id], index[end_id]
    if start > end:
        raise GoldMinerError(f"Boundary repair for {evaluation.candidate_id} reversed its range")
    selected = transcript.utterances[start : end + 1]
    start_span = (
        _find_phrase_span(selected[0].original_text, evaluation.start_boundary_text, use_last=False)
        if evaluation.start_boundary_text.strip()
        else _full_span(selected[0].original_text)
    )
    end_span = (
        _find_phrase_span(selected[-1].original_text, evaluation.end_boundary_text, use_last=True)
        if evaluation.end_boundary_text.strip()
        else _full_span(selected[-1].original_text)
    )
    if len(selected) == 1 and start_span.char_start >= end_span.char_end:
        raise GoldMinerError(f"Phrase boundary repair for {evaluation.candidate_id} reversed its range")
    start_ms = _interpolated_ms(selected[0], start_span.token_start, start_span.token_count)
    end_ms = _interpolated_ms(selected[-1], end_span.token_end, end_span.token_count)
    if start_ms >= end_ms:
        raise GoldMinerError(f"Phrase boundary repair for {evaluation.candidate_id} has no duration")
    if len(selected) == 1:
        selected_text = selected[0].original_text[start_span.char_start : end_span.char_end]
    else:
        text_parts = [selected[0].original_text[start_span.char_start :]]
        text_parts.extend(item.original_text for item in selected[1:-1])
        text_parts.append(selected[-1].original_text[: end_span.char_end])
        selected_text = " ".join(part.strip() for part in text_parts if part.strip())
    subtotal, final, _ = calculate_score(evaluation.dimensions, evaluation.penalties)
    rejection_reasons: list[str] = []
    if evaluation.unsafe_extraction:
        rejection_reasons.append("unsafe_extraction")
    if evaluation.non_contiguous_required:
        rejection_reasons.append("non_contiguous_splice_required")
    if (
        not evaluation.opening_is_clear
        or evaluation.opening_problem != "none"
        or evaluation.dimensions["standalone_clarity"].raw_score < 7
    ):
        rejection_reasons.append("unclear_opening")
    if _CONTEXTUAL_OPENING_RE.search(selected_text):
        rejection_reasons.append("context_dependent_opening")
    if (
        not evaluation.central_idea_is_strong
        or evaluation.central_idea_problem != "none"
        or evaluation.dimensions["value"].raw_score < 7
        or evaluation.dimensions["payoff"].raw_score < 7
    ):
        rejection_reasons.append("weak_or_unclear_core_idea")
    payoff_is_present = True
    if evaluation.payoff_text.strip():
        try:
            _find_phrase_span(selected_text, evaluation.payoff_text, use_last=True)
        except GoldMinerError:
            payoff_is_present = False
    if not evaluation.arc_is_complete or evaluation.payoff_type == "none" or not payoff_is_present:
        rejection_reasons.append("incomplete_narrative_arc")
    if _INCOMPLETE_ENDING_RE.search(selected_text):
        rejection_reasons.append("incomplete_ending")
    duration_ms = end_ms - start_ms
    if duration_ms < 15_000 or duration_ms > max_duration_ms:
        rejection_reasons.append("duration_out_of_bounds")
    if final < 60:
        rejection_reasons.append("score_below_publishable_threshold")
    return ScoredCandidate(
        id=context.candidate.id,
        start_ms=start_ms,
        end_ms=end_ms,
        source_utterance_ids=tuple(item.id for item in selected),
        transcript=selected_text,
        category=evaluation.category,
        secondary_categories=evaluation.secondary_categories,
        dimensions=evaluation.dimensions,
        penalties=evaluation.penalties,
        weighted_subtotal=subtotal,
        final_score=0.0 if rejection_reasons else final,
        rejected=bool(rejection_reasons),
        rejection_reasons=tuple(rejection_reasons),
        boundary_rationale=evaluation.boundary_rationale,
        why_selected=evaluation.why_selected,
        opening_is_clear=evaluation.opening_is_clear,
        central_idea_is_strong=evaluation.central_idea_is_strong,
        opening_evidence=evaluation.opening_evidence,
        central_idea_evidence=evaluation.central_idea_evidence,
        start_boundary_text=evaluation.start_boundary_text,
        end_boundary_text=evaluation.end_boundary_text,
        opening_problem=evaluation.opening_problem,
        central_idea_problem=evaluation.central_idea_problem,
        arc_is_complete=evaluation.arc_is_complete,
        payoff_type=evaluation.payoff_type,
        payoff_text=evaluation.payoff_text,
        arc_evidence=evaluation.arc_evidence,
        variant_id=evaluation.variant_id,
    )
