from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import GoldMinerError


@dataclass(frozen=True, slots=True)
class RunConfig:
    video: Path
    output: Path
    top_k: int = 8
    transcript: Path | None = None
    no_cut: bool = False
    resume: bool = False
    handle_ms: int = 500
    max_clip_seconds: int = 60

    def validate(self) -> "RunConfig":
        video = self.video.expanduser().resolve()
        if not video.exists():
            raise GoldMinerError(f"Input video does not exist: {video}")
        if not video.is_file():
            raise GoldMinerError(f"Input video is not a regular file: {video}")
        if self.top_k < 1:
            raise GoldMinerError("--top-k must be at least 1")
        if not 0 <= self.handle_ms <= 5000:
            raise GoldMinerError("--handle-ms must be between 0 and 5000")
        if not 15 <= self.max_clip_seconds <= 120:
            raise GoldMinerError("--max-duration must be between 15 and 120 seconds")

        transcript = self.transcript.expanduser().resolve() if self.transcript else None
        if transcript is not None and (not transcript.exists() or not transcript.is_file()):
            raise GoldMinerError(f"Transcript does not exist or is not a file: {transcript}")

        return RunConfig(
            video=video,
            output=self.output.expanduser().resolve(),
            top_k=self.top_k,
            transcript=transcript,
            no_cut=self.no_cut,
            resume=self.resume,
            handle_ms=self.handle_ms,
            max_clip_seconds=self.max_clip_seconds,
        )
