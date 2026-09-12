from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    output_tail: tuple[str, ...]


class GoldMinerRunner(Protocol):
    def run(
        self,
        video: Path,
        output: Path,
        *,
        transcript: Path | None,
        top_k: int,
        max_duration_seconds: int,
        handle_ms: int,
    ) -> ProcessResult: ...


class SubprocessGoldMinerRunner:
    def __init__(
        self,
        *,
        executable: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._executable = executable or sys.executable
        self._progress = progress or (lambda line: print(line, flush=True))

    def command(
        self,
        video: Path,
        output: Path,
        *,
        transcript: Path | None,
        top_k: int,
        max_duration_seconds: int,
        handle_ms: int,
    ) -> Sequence[str]:
        command = [
            self._executable,
            "-m",
            "goldminer",
            str(video),
            "--output",
            str(output),
            "--top-k",
            str(top_k),
            "--max-duration",
            str(max_duration_seconds),
            "--handle-ms",
            str(handle_ms),
            "--resume",
        ]
        if transcript is not None:
            command.extend(["--transcript", str(transcript)])
        return command

    def run(
        self,
        video: Path,
        output: Path,
        *,
        transcript: Path | None,
        top_k: int,
        max_duration_seconds: int,
        handle_ms: int,
    ) -> ProcessResult:
        command = self.command(
            video,
            output,
            transcript=transcript,
            top_k=top_k,
            max_duration_seconds=max_duration_seconds,
            handle_ms=handle_ms,
        )
        tail: deque[str] = deque(maxlen=100)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            tail.append(line)
            self._progress(line)
        return ProcessResult(returncode=process.wait(), output_tail=tuple(tail))
