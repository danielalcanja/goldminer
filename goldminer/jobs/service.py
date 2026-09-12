from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Mapping

from goldminer.errors import GoldMinerError

from .models import JOB_SCHEMA_VERSION, JobResult, JobSpec, UploadedArtifact
from .runner import GoldMinerRunner
from .storage import ObjectStore

_LOG = logging.getLogger(__name__)
_ROOT_ARTIFACTS = (
    "transcript.json",
    "candidates.json",
    "scored_candidates.json",
    "ranking.json",
    "report.html",
    "run_manifest.json",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_name(key: str, fallback: str) -> str:
    name = PurePosixPath(key).name
    return name if name not in {"", ".", ".."} else fallback


class JobService:
    def __init__(
        self,
        store: ObjectStore,
        runner: GoldMinerRunner,
        *,
        work_root: Path | None = None,
        keep_workdir: bool = False,
    ) -> None:
        configured_root = os.environ.get("GOLDMINER_WORK_ROOT", "/tmp/goldminer-jobs")
        self._work_root = (work_root or Path(configured_root)).resolve()
        self._store = store
        self._runner = runner
        self._keep_workdir = keep_workdir

    def _status_key(self, spec: JobSpec) -> str:
        return f"{spec.result_prefix}/status.json"

    def _write_status(self, spec: JobSpec, status: str, **details: Any) -> None:
        value: dict[str, Any] = {
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": spec.job_id,
            "status": status,
            "source_key": spec.source_key,
            "updated_at": _now(),
        }
        value.update(details)
        self._store.put_json(self._status_key(spec), value)

    def _artifact_paths(self, run_directory: Path) -> tuple[Path, ...]:
        artifacts = [
            run_directory / name
            for name in _ROOT_ARTIFACTS
            if (run_directory / name).is_file()
        ]
        clips = run_directory / "clips"
        if clips.is_dir():
            artifacts.extend(sorted(path for path in clips.glob("*.mp4") if path.is_file()))
        run_log = run_directory / "logs" / "run.jsonl"
        if run_log.is_file():
            artifacts.append(run_log)
        return tuple(artifacts)

    def _upload_artifacts(
        self, spec: JobSpec, run_directory: Path
    ) -> tuple[UploadedArtifact, ...]:
        uploaded: list[UploadedArtifact] = []
        for path in self._artifact_paths(run_directory):
            relative = path.relative_to(run_directory).as_posix()
            key = f"{spec.result_prefix}/artifacts/{relative}"
            self._store.upload_file(path, key)
            uploaded.append(
                UploadedArtifact(name=relative, key=key, size_bytes=path.stat().st_size)
            )
        return tuple(uploaded)

    def run(self, spec: JobSpec) -> JobResult:
        self._work_root.mkdir(parents=True, exist_ok=True)
        job_directory = self._work_root / spec.job_id
        job_directory.mkdir(parents=False, exist_ok=True)
        source = job_directory / "input" / "source" / _local_name(
            spec.source_key, "source-video"
        )
        transcript = (
            job_directory
            / "input"
            / "transcript"
            / _local_name(spec.transcript_key, "transcript.srt")
            if spec.transcript_key
            else None
        )
        run_directory = job_directory / "run"
        started_at = _now()
        completed = False
        try:
            self._write_status(spec, "running", phase="downloading", started_at=started_at)
            if not source.is_file() or source.stat().st_size == 0:
                self._store.download_file(spec.source_key, source)
            if (
                spec.transcript_key
                and transcript is not None
                and (not transcript.is_file() or transcript.stat().st_size == 0)
            ):
                self._store.download_file(spec.transcript_key, transcript)

            self._write_status(spec, "running", phase="processing", started_at=started_at)
            process = self._runner.run(
                source,
                run_directory,
                transcript=transcript,
                top_k=spec.top_k,
                max_duration_seconds=spec.max_duration_seconds,
                handle_ms=spec.handle_ms,
            )
            if process.returncode != 0:
                last_line = process.output_tail[-1] if process.output_tail else "no process output"
                raise GoldMinerError(
                    f"Gold Miner exited with code {process.returncode}: {last_line}"
                )

            self._write_status(spec, "running", phase="uploading", started_at=started_at)
            artifacts = self._upload_artifacts(spec, run_directory)
            if not any(artifact.name == "run_manifest.json" for artifact in artifacts):
                raise GoldMinerError("Gold Miner completed without producing run_manifest.json")

            completed_at = _now()
            result = JobResult(
                job_id=spec.job_id,
                status_key=self._status_key(spec),
                artifacts=artifacts,
            )
            self._write_status(
                spec,
                "completed",
                phase="completed",
                started_at=started_at,
                completed_at=completed_at,
                artifacts=[artifact.as_dict() for artifact in artifacts],
            )
            completed = True
            return result
        except BaseException as exc:
            try:
                diagnostic_artifacts = self._upload_artifacts(spec, run_directory)
                self._write_status(
                    spec,
                    "failed",
                    phase="failed",
                    started_at=started_at,
                    failed_at=_now(),
                    error={"type": type(exc).__name__, "message": str(exc)},
                    artifacts=[artifact.as_dict() for artifact in diagnostic_artifacts],
                )
            except BaseException:
                _LOG.exception("Could not persist failure status for job %s", spec.job_id)
            raise
        finally:
            if self._keep_workdir:
                _LOG.info("Keeping job work directory %s", job_directory)
            elif completed:
                shutil.rmtree(job_directory, ignore_errors=True)
            else:
                _LOG.info("Keeping failed job work directory for Workflow retry: %s", job_directory)
