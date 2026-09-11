from __future__ import annotations

from goldminer.transcript.models import Utterance

from .models import DiscoveryWindow


def build_discovery_windows(
    utterances: tuple[Utterance, ...],
    *,
    target_duration_ms: int = 300_000,
    overlap_ms: int = 60_000,
) -> tuple[DiscoveryWindow, ...]:
    if not utterances:
        return ()
    if target_duration_ms <= 0 or overlap_ms < 0 or overlap_ms >= target_duration_ms:
        raise ValueError("window durations must satisfy 0 <= overlap < target")

    windows: list[DiscoveryWindow] = []
    start_index = 0
    while start_index < len(utterances):
        window_start = utterances[start_index].start_ms
        target_end = window_start + target_duration_ms
        end_index = start_index
        while end_index + 1 < len(utterances) and utterances[end_index + 1].start_ms < target_end:
            end_index += 1
        window_items = utterances[start_index : end_index + 1]
        windows.append(DiscoveryWindow(f"w{len(windows) + 1:04d}", window_items))
        if end_index == len(utterances) - 1:
            break
        next_start_time = max(window_start + 1, window_items[-1].end_ms - overlap_ms)
        next_index = start_index + 1
        while next_index < len(utterances) and utterances[next_index].start_ms < next_start_time:
            next_index += 1
        start_index = min(next_index, end_index + 1)
    return tuple(windows)
