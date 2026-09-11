from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Sequence

from goldminer.errors import GoldMinerError
from goldminer.media.ffmpeg import FFmpegTools, Runner, probe_duration_ms, run_checked
from goldminer.transcript.models import Utterance

CUTTER_VERSION = "1.2"
CUT_DURATION_TOLERANCE_MS = 750
MAX_VIDEO_BITRATE = 4_000_000
MAX_AUDIO_BITRATE = 128_000
FREE_SPACE_RESERVE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ClipCut:
    rank: int
    candidate_id: str
    path: Path
    start_ms: int
    end_ms: int
    measured_duration_ms: int
    reused: bool


def stable_clip_name(rank: int, category: str, score: float) -> str:
    safe_category = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_") or "clip"
    return f"{rank:02d}_{safe_category}_{round(score):02d}.mp4"


def _fingerprint(source_sha256: str, candidate: dict[str, Any], start_ms: int, end_ms: int) -> str:
    payload = f"{CUTTER_VERSION}:{source_sha256}:{candidate['id']}:{start_ms}:{end_ms}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _speech_safe_range(
    core_start: int,
    core_end: int,
    *,
    handle_ms: int,
    media_duration_ms: int,
    utterances: Sequence[Utterance],
) -> tuple[int, int]:
    start_ms = max(0, core_start - handle_ms)
    end_ms = min(media_duration_ms, core_end + handle_ms)
    if not utterances or handle_ms == 0:
        return start_ms, end_ms

    # A phrase-level boundary inside an utterance is exact: extending backward
    # or forward would restore speech the editor deliberately removed.
    if any(item.start_ms < core_start < item.end_ms for item in utterances):
        start_ms = core_start
    else:
        previous_ends = [item.end_ms for item in utterances if item.end_ms <= core_start]
        if previous_ends:
            start_ms = max(start_ms, max(previous_ends))

    if any(item.start_ms < core_end < item.end_ms for item in utterances):
        end_ms = core_end
    else:
        next_starts = [item.start_ms for item in utterances if item.start_ms >= core_end]
        if next_starts:
            end_ms = min(end_ms, min(next_starts))
    return start_ms, end_ms


def cut_selected_clips(
    source: Path,
    selections: Sequence[dict[str, Any]],
    clips_dir: Path,
    tools: FFmpegTools,
    *,
    source_sha256: str,
    media_duration_ms: int,
    handle_ms: int = 500,
    max_duration_ms: int = 60_000,
    utterances: Sequence[Utterance] = (),
    resume: bool = False,
    runner: Runner = subprocess.run,
    progress=None,
) -> tuple[ClipCut, ...]:
    cut_ranges: list[tuple[int, int]] = []
    for item in selections:
        core_start, core_end = int(item["start_ms"]), int(item["end_ms"])
        if core_end - core_start > max_duration_ms:
            raise GoldMinerError(
                f"Candidate {item['id']} exceeds the configured {max_duration_ms / 1000:g} second maximum"
            )
        start_ms, end_ms = _speech_safe_range(
            core_start,
            core_end,
            handle_ms=handle_ms,
            media_duration_ms=media_duration_ms,
            utterances=utterances,
        )
        excess = max(0, end_ms - start_ms - max_duration_ms)
        if excess:
            trim_after = min(excess, end_ms - core_end)
            end_ms -= trim_after
            start_ms += excess - trim_after
        cut_ranges.append((start_ms, end_ms))
    total_duration_ms = sum(end - start for start, end in cut_ranges)
    estimated_bytes = round(total_duration_ms / 1000 * (MAX_VIDEO_BITRATE + MAX_AUDIO_BITRATE) / 8)
    available_bytes = shutil.disk_usage(clips_dir).free
    required_bytes = estimated_bytes + FREE_SPACE_RESERVE_BYTES
    if available_bytes < required_bytes:
        raise GoldMinerError(
            f"Not enough free disk space to cut {len(selections)} clips. "
            f"Approximately {required_bytes / (1024**2):.0f} MiB is required, but "
            f"only {available_bytes / (1024**2):.0f} MiB is available. Free disk "
            "space or choose --output on a drive with more room, then rerun with --resume."
        )
    cache_dir = clips_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[ClipCut] = []
    for candidate, (start_ms, end_ms) in zip(selections, cut_ranges):
        rank = int(candidate["rank"])
        if end_ms <= start_ms:
            raise GoldMinerError(f"Invalid cut range for candidate {candidate['id']}")
        destination = clips_dir / stable_clip_name(rank, str(candidate["category"]), float(candidate["final_score"]))
        checkpoint = cache_dir / f"{destination.stem}.json"
        fingerprint = _fingerprint(source_sha256, candidate, start_ms, end_ms)
        if resume and destination.is_file() and destination.stat().st_size > 0 and checkpoint.is_file():
            try:
                cached = json.loads(checkpoint.read_text(encoding="utf-8"))
                measured = probe_duration_ms(destination, tools, runner=runner)
                if cached.get("fingerprint") == fingerprint and abs(measured - (end_ms - start_ms)) <= CUT_DURATION_TOLERANCE_MS:
                    if progress: progress(f"Clip {rank}/{len(selections)}: reused {destination.name}")
                    results.append(ClipCut(rank, str(candidate["id"]), destination, start_ms, end_ms, measured, True))
                    continue
            except (OSError, json.JSONDecodeError, GoldMinerError):
                pass
        if progress: progress(f"Clip {rank}/{len(selections)}: cutting {destination.name}")
        temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
        try:
            run_checked([
                tools.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start_ms / 1000:.3f}", "-i", str(source),
                "-t", f"{(end_ms - start_ms) / 1000:.3f}",
                "-map", "0:v:0", "-map", "0:a:0?",
                "-vf", "scale=w='if(gte(iw,ih),min(1920,iw),-2)':h='if(gte(iw,ih),-2,min(1920,ih))'",
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-maxrate", "4M", "-bufsize", "8M",
                "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(temporary),
            ], runner=runner)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise GoldMinerError(f"FFmpeg did not produce clip {rank}")
            measured = probe_duration_ms(temporary, tools, runner=runner)
            expected = end_ms - start_ms
            if abs(measured - expected) > CUT_DURATION_TOLERANCE_MS:
                raise GoldMinerError(f"Clip {rank} duration {measured} ms differs from expected {expected} ms by more than {CUT_DURATION_TOLERANCE_MS} ms")
            os.replace(temporary, destination)
            _atomic_json(checkpoint, {"cutter_version": CUTTER_VERSION, "fingerprint": fingerprint, "measured_duration_ms": measured})
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        results.append(ClipCut(rank, str(candidate["id"]), destination, start_ms, end_ms, measured, False))
        if progress: progress(f"Clip {rank}/{len(selections)}: completed")
    return tuple(results)
