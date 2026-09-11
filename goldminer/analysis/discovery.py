from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable

from goldminer.errors import GoldMinerError
from goldminer.providers.analysis import AnalysisProvider
from goldminer.transcript.models import CanonicalTranscript

from .models import Candidate, CandidateProposal, DiscoveryWindow

_WORD_RE = re.compile(r"[a-z0-9]+")
_UTTERANCE_ID_RE = re.compile(r"^[uU]0*(\d+)$")


def canonical_utterance_id(value: str) -> str:
    match = _UTTERANCE_ID_RE.fullmatch(value.strip())
    if not match:
        return value.strip()
    return f"u{int(match.group(1)):06d}"


def proposal_to_candidate(
    proposal: CandidateProposal,
    window: DiscoveryWindow,
    transcript: CanonicalTranscript,
) -> Candidate:
    start_id = canonical_utterance_id(proposal.start_utterance_id)
    end_id = canonical_utterance_id(proposal.end_utterance_id)
    window_ids = {item.id for item in window.utterances}
    if start_id not in window_ids or end_id not in window_ids:
        raise GoldMinerError(
            f"Provider proposed IDs outside {window.id}: "
            f"{proposal.start_utterance_id}-{proposal.end_utterance_id}"
        )
    id_to_index = {item.id: index for index, item in enumerate(transcript.utterances)}
    start_index = id_to_index[start_id]
    end_index = id_to_index[end_id]
    if start_index > end_index:
        raise GoldMinerError("Candidate start utterance must not follow its end utterance")
    selected = transcript.utterances[start_index : end_index + 1]
    if not selected:
        raise GoldMinerError("Candidate source range is empty")
    return Candidate(
        id="pending",
        start_ms=selected[0].start_ms,
        end_ms=selected[-1].end_ms,
        source_utterance_ids=tuple(item.id for item in selected),
        detector_categories=(proposal.detector,),
        transcript=" ".join(item.original_text for item in selected),
        core_idea=proposal.core_idea,
        evidence=proposal.evidence,
        source_windows=(window.id,),
    )


def _token_similarity(left: str, right: str) -> float:
    a, b = set(_WORD_RE.findall(left.casefold())), set(_WORD_RE.findall(right.casefold()))
    return len(a & b) / len(a | b) if a and b else 0.0


def _overlap_ratio(left: Candidate, right: Candidate) -> float:
    overlap = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    shortest = min(left.end_ms - left.start_ms, right.end_ms - right.start_ms)
    return overlap / shortest if shortest else 0.0


def merge_overlapping_candidates(candidates: list[Candidate]) -> list[Candidate]:
    retained: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.start_ms, item.end_ms)):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(retained)
                if _overlap_ratio(existing, candidate) >= 0.8
                and _token_similarity(existing.core_idea, candidate.core_idea) >= 0.4
            ),
            None,
        )
        if duplicate_index is None:
            retained.append(candidate)
            continue
        existing = retained[duplicate_index]
        preferred = min(
            (existing, candidate),
            key=lambda item: (item.end_ms - item.start_ms, -len(item.evidence)),
        )
        retained[duplicate_index] = replace(
            preferred,
            detector_categories=tuple(
                sorted(set(existing.detector_categories + candidate.detector_categories))
            ),
            source_windows=tuple(sorted(set(existing.source_windows + candidate.source_windows))),
        )
    return [replace(item, id=f"c{index:05d}") for index, item in enumerate(retained, start=1)]


def discover_candidates(
    transcript: CanonicalTranscript,
    windows: tuple[DiscoveryWindow, ...],
    provider: AnalysisProvider,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Candidate, ...]:
    proposals: list[Candidate] = []
    for index, window in enumerate(windows, start=1):
        if progress:
            progress(f"Candidate discovery window {index}/{len(windows)}: analyzing")
        for proposal in provider.discover(window):
            proposals.append(proposal_to_candidate(proposal, window, transcript))
        if progress:
            progress(f"Candidate discovery window {index}/{len(windows)}: completed")
    return tuple(merge_overlapping_candidates(proposals))
