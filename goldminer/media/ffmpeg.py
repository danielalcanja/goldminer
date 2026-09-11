from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from goldminer.errors import GoldMinerError

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class FFmpegTools:
    ffmpeg: str
    ffprobe: str


def _find_executable(value: str, label: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise GoldMinerError(
            f"{label} was not found. Install FFmpeg and ensure '{value}' is executable on PATH, "
            f"or set GOLDMINER_{label.upper()} to its full path."
        )
    return resolved


def discover_tools() -> FFmpegTools:
    return FFmpegTools(
        ffmpeg=_find_executable(os.environ.get("GOLDMINER_FFMPEG", "ffmpeg"), "ffmpeg"),
        ffprobe=_find_executable(os.environ.get("GOLDMINER_FFPROBE", "ffprobe"), "ffprobe"),
    )


def run_checked(args: Sequence[str], *, runner: Runner = subprocess.run) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(list(args), capture_output=True, text=True, check=False)
    except OSError as exc:
        raise GoldMinerError(f"Could not start media tool: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise GoldMinerError(f"Media command failed: {detail}")
    return result


def validate_tools(tools: FFmpegTools, *, runner: Runner = subprocess.run) -> None:
    run_checked([tools.ffmpeg, "-version"], runner=runner)
    run_checked([tools.ffprobe, "-version"], runner=runner)


def probe_duration_ms(path: Path, tools: FFmpegTools, *, runner: Runner = subprocess.run) -> int:
    result = run_checked(
        [tools.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        runner=runner,
    )
    try:
        seconds = float(json.loads(result.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GoldMinerError(f"FFprobe returned no valid duration for: {path}") from exc
    if seconds <= 0:
        raise GoldMinerError(f"Input media has a non-positive duration: {path}")
    return round(seconds * 1000)
