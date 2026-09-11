from __future__ import annotations

import os
import subprocess
from pathlib import Path

from goldminer.errors import GoldMinerError
from goldminer.media.ffmpeg import FFmpegTools, Runner, run_checked


def extract_analysis_audio(
    source: Path,
    destination: Path,
    tools: FFmpegTools,
    *,
    resume: bool = False,
    runner: Runner = subprocess.run,
) -> bool:
    """Extract mono 16 kHz PCM audio; return True when a cached file was reused."""
    if resume and destination.is_file() and destination.stat().st_size > 44:
        return True

    temporary = destination.with_name(f".{destination.name}.tmp.wav")
    try:
        run_checked(
            [
                tools.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            runner=runner,
        )
        if not temporary.is_file() or temporary.stat().st_size <= 44:
            raise GoldMinerError("FFmpeg completed but did not produce a valid audio artifact")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return False
