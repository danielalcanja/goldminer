from __future__ import annotations

from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import threading
from typing import Any
from urllib.parse import urlsplit

from goldminer.errors import GoldMinerError

from .models import JobSpec
from .service import JobService

_LOG = logging.getLogger(__name__)
_MAX_REQUEST_BYTES = 64 * 1024
_SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
)


def _safe_exception_message(exc: Exception) -> str:
    message = str(exc).strip() or "no exception message"
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name, "")
        if value:
            message = message.replace(value, f"<{name} redacted>")
    return f"{type(exc).__name__}: {message}"[:1000]


class JobHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service_factory: Callable[[], JobService],
        *,
        bind_and_activate: bool = True,
    ) -> None:
        super().__init__(address, JobRequestHandler, bind_and_activate=bind_and_activate)
        self._service_factory = service_factory
        self._service: JobService | None = None
        self._service_lock = threading.Lock()
        self.job_lock = threading.Lock()

    def get_service(self) -> JobService:
        with self._service_lock:
            if self._service is None:
                self._service = self._service_factory()
            return self._service


class JobRequestHandler(BaseHTTPRequestHandler):
    server: JobHTTPServer

    def _json(self, status: int, value: Any) -> None:
        body = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlsplit(self.path).path == "/healthz":
            self._json(200, {"status": "ok", "busy": self.server.job_lock.locked()})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if urlsplit(self.path).path != "/run":
            self._json(404, {"error": "not_found"})
            return
        try:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise GoldMinerError("Content-Length is required")
            length = int(raw_length)
            if not 0 < length <= _MAX_REQUEST_BYTES:
                raise GoldMinerError("Job request must be between 1 byte and 64 KiB")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise GoldMinerError("Job request must be a JSON object")
            spec = JobSpec.from_mapping(payload)
        except (ValueError, json.JSONDecodeError, GoldMinerError) as exc:
            self._json(400, {"error": "invalid_job", "message": str(exc)})
            return

        self.server.job_lock.acquire()
        try:
            result = self.server.get_service().run(spec)
            self._json(200, result.as_dict())
        except GoldMinerError as exc:
            _LOG.exception("Job %s failed", spec.job_id)
            self._json(500, {"error": "job_failed", "message": str(exc)})
        except Exception as exc:
            _LOG.exception("Job %s failed unexpectedly", spec.job_id)
            self._json(
                500,
                {
                    "error": "unexpected_worker_failure",
                    "message": _safe_exception_message(exc),
                },
            )
        except BaseException:
            _LOG.exception("Job %s failed unexpectedly", spec.job_id)
            self._json(500, {"error": "job_failed", "message": "Unexpected worker failure"})
        finally:
            self.server.job_lock.release()

    def log_message(self, format: str, *args: Any) -> None:
        _LOG.info("%s - %s", self.address_string(), format % args)


def serve(
    service_factory: Callable[[], JobService],
    host: str,
    port: int,
) -> None:
    server = JobHTTPServer((host, port), service_factory)
    _LOG.info("Gold Miner job worker listening on %s:%s", host, port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
