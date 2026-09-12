from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Sequence

from goldminer.env import load_dotenv, project_dotenv
from goldminer.errors import GoldMinerError

from .models import JobSpec
from .runner import SubprocessGoldMinerRunner
from .server import serve
from .service import JobService
from .storage import R2ObjectStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goldminer-worker",
        description="Run Gold Miner background jobs using Cloudflare R2 storage.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    server = subparsers.add_parser("serve", help="start the container HTTP server")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    run = subparsers.add_parser("run", help="run one job from a local JSON request")
    run.add_argument("job_file", type=Path)
    return parser


def _service() -> JobService:
    return JobService(
        R2ObjectStore.from_environment(),
        SubprocessGoldMinerRunner(),
        keep_workdir=os.environ.get("GOLDMINER_KEEP_WORKDIR") == "1",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        load_dotenv(project_dotenv())
        service = _service()
        if args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise GoldMinerError("--port must be between 1 and 65535")
            serve(service, args.host, args.port)
            return 0

        try:
            payload = json.loads(args.job_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise GoldMinerError(f"Could not read job file: {exc}") from exc
        if not isinstance(payload, dict):
            raise GoldMinerError("Job file must contain a JSON object")
        result = service.run(JobSpec.from_mapping(payload))
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return 0
    except GoldMinerError as exc:
        print(f"goldminer-worker: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
