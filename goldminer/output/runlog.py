from __future__ import annotations

import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RunLogger:
    """Append-only structured diagnostics for one CLI invocation."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self._started = perf_counter()

    def write(self, event: str, status: str, **details: Any) -> None:
        record = {
            "timestamp": _utc_now(),
            "run_id": self.run_id,
            "event": event,
            "status": status,
            "elapsed_ms": round((perf_counter() - self._started) * 1000),
            **details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded.encode("utf-8"))
        finally:
            os.close(descriptor)

    def try_write(self, event: str, status: str, **details: Any) -> None:
        """Best-effort diagnostics must never replace the pipeline's real error."""
        try:
            self.write(event, status, **details)
        except OSError:
            pass

    def start(self, event: str, **details: Any) -> float:
        self.write(event, "started", **details)
        return perf_counter()

    def complete(self, event: str, started: float, **details: Any) -> None:
        self.write(event, "completed", duration_ms=round((perf_counter() - started) * 1000), **details)

    def failure(self, event: str, started: float, error: BaseException) -> None:
        self.try_write(
            event,
            "failed",
            duration_ms=round((perf_counter() - started) * 1000),
            error_type=type(error).__name__,
            error=str(error),
        )

    def environment(self) -> dict[str, str]:
        return {
            "goldminer_version": "0.1.0",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        }
