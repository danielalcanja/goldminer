from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from goldminer.errors import GoldMinerError
from goldminer.jobs.models import JobSpec
from goldminer.jobs.runner import ProcessResult, SubprocessGoldMinerRunner
from goldminer.jobs.server import JobHTTPServer
from goldminer.jobs.service import JobService


class FakeStore:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = dict(objects or {})
        self.statuses: list[tuple[str, dict[str, object]]] = []

    def download_file(self, key: str, destination: Path) -> None:
        if key not in self.objects:
            raise GoldMinerError(f"missing fake object: {key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])

    def upload_file(self, source: Path, key: str) -> None:
        self.objects[key] = source.read_bytes()

    def put_json(self, key: str, value: dict[str, object]) -> None:
        copied = json.loads(json.dumps(value))
        self.statuses.append((key, copied))
        self.objects[key] = json.dumps(copied).encode("utf-8")


class FakeRunner:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "video": video,
                "output": output,
                "transcript": transcript,
                "top_k": top_k,
                "max_duration_seconds": max_duration_seconds,
                "handle_ms": handle_ms,
            }
        )
        (output / "logs").mkdir(parents=True)
        (output / "logs" / "run.jsonl").write_text('{"status":"test"}\n')
        if self.returncode == 0:
            (output / "clips").mkdir()
            (output / "clips" / "01_lesson_92.mp4").write_bytes(b"clip")
            (output / "run_manifest.json").write_text('{"status":"completed"}\n')
            (output / "report.html").write_text("<html></html>")
        return ProcessResult(self.returncode, ("fake process output",))


class JobSpecTests(unittest.TestCase):
    def test_defaults_keep_hosted_clips_at_sixty_seconds(self) -> None:
        spec = JobSpec.from_mapping(
            {"job_id": "job-123", "source_key": "uploads/job-123/video.mov"}
        )
        self.assertEqual(spec.result_prefix, "jobs/job-123")
        self.assertEqual(spec.top_k, 8)
        self.assertEqual(spec.max_duration_seconds, 60)
        self.assertEqual(spec.handle_ms, 500)

    def test_rejects_unsafe_keys_and_long_hosted_clips(self) -> None:
        with self.assertRaisesRegex(GoldMinerError, "safe relative"):
            JobSpec.from_mapping({"job_id": "job-123", "source_key": "../video.mov"})
        with self.assertRaisesRegex(GoldMinerError, "between 15 and 60"):
            JobSpec.from_mapping(
                {
                    "job_id": "job-123",
                    "source_key": "uploads/video.mov",
                    "max_duration_seconds": 61,
                }
            )


class JobServiceTests(unittest.TestCase):
    def test_downloads_runs_and_uploads_only_review_artifacts(self) -> None:
        with TemporaryDirectory() as temporary:
            work_root = Path(temporary) / "work"
            store = FakeStore(
                {
                    "uploads/job-123/video.mov": b"video",
                    "uploads/job-123/transcript.srt": b"1\n00:00:00,000 --> 00:00:01,000\nHi\n",
                }
            )
            runner = FakeRunner()
            service = JobService(store, runner, work_root=work_root)
            spec = JobSpec.from_mapping(
                {
                    "job_id": "job-123",
                    "source_key": "uploads/job-123/video.mov",
                    "transcript_key": "uploads/job-123/transcript.srt",
                    "top_k": 5,
                    "handle_ms": 0,
                }
            )

            result = service.run(spec)

            self.assertEqual(result.job_id, "job-123")
            self.assertEqual([status[1]["status"] for status in store.statuses], [
                "running", "running", "running", "completed"
            ])
            self.assertEqual(runner.calls[0]["top_k"], 5)
            self.assertEqual(runner.calls[0]["max_duration_seconds"], 60)
            self.assertEqual(runner.calls[0]["handle_ms"], 0)
            uploaded = {artifact.name for artifact in result.artifacts}
            self.assertEqual(
                uploaded,
                {"clips/01_lesson_92.mp4", "logs/run.jsonl", "report.html", "run_manifest.json"},
            )
            self.assertNotIn("jobs/job-123/artifacts/audio.wav", store.objects)
            self.assertEqual(list(work_root.iterdir()), [])

    def test_failure_uploads_diagnostic_log_and_failed_status(self) -> None:
        with TemporaryDirectory() as temporary:
            store = FakeStore({"uploads/video.mov": b"video"})
            service = JobService(
                store,
                FakeRunner(returncode=2),
                work_root=Path(temporary),
            )
            spec = JobSpec.from_mapping(
                {"job_id": "failed-job", "source_key": "uploads/video.mov"}
            )

            with self.assertRaisesRegex(GoldMinerError, "exited with code 2"):
                service.run(spec)

            self.assertEqual(store.statuses[-1][1]["status"], "failed")
            self.assertIn(
                "jobs/failed-job/artifacts/logs/run.jsonl",
                store.objects,
            )
            self.assertTrue((Path(temporary) / "failed-job" / "run").is_dir())

    def test_retry_reuses_downloaded_source_and_local_pipeline_directory(self) -> None:
        class CountingStore(FakeStore):
            def __init__(self) -> None:
                super().__init__({"uploads/video.mov": b"video"})
                self.download_count = 0

            def download_file(self, key: str, destination: Path) -> None:
                self.download_count += 1
                super().download_file(key, destination)

        class RetryRunner(FakeRunner):
            def run(  # type: ignore[override]
                self,
                video: Path,
                output: Path,
                *,
                transcript: Path | None,
                top_k: int,
                max_duration_seconds: int,
                handle_ms: int,
            ) -> ProcessResult:
                self.calls.append(
                    {
                        "video": video,
                        "output": output,
                        "transcript": transcript,
                        "top_k": top_k,
                        "max_duration_seconds": max_duration_seconds,
                        "handle_ms": handle_ms,
                    }
                )
                (output / "logs").mkdir(parents=True, exist_ok=True)
                (output / "logs" / "run.jsonl").write_text('{"status":"test"}\n')
                if len(self.calls) == 1:
                    return ProcessResult(2, ("first attempt failed",))
                (output / "clips").mkdir(exist_ok=True)
                (output / "clips" / "01_lesson_92.mp4").write_bytes(b"clip")
                (output / "run_manifest.json").write_text('{"status":"completed"}\n')
                return ProcessResult(0, ("second attempt succeeded",))

        with TemporaryDirectory() as temporary:
            store = CountingStore()
            runner = RetryRunner()
            service = JobService(store, runner, work_root=Path(temporary))
            spec = JobSpec.from_mapping(
                {"job_id": "retry-job", "source_key": "uploads/video.mov"}
            )

            with self.assertRaises(GoldMinerError):
                service.run(spec)
            result = service.run(spec)

            self.assertEqual(store.download_count, 1)
            self.assertEqual(result.job_id, "retry-job")
            self.assertEqual(runner.calls[0]["output"], runner.calls[1]["output"])
            self.assertFalse((Path(temporary) / "retry-job").exists())


class JobHTTPServerTests(unittest.TestCase):
    def test_service_configuration_is_lazy_so_the_container_can_boot(self) -> None:
        calls = 0

        def unavailable_service() -> JobService:
            nonlocal calls
            calls += 1
            raise GoldMinerError("missing worker configuration")

        server = JobHTTPServer(
            ("127.0.0.1", 0), unavailable_service, bind_and_activate=False
        )
        try:
            self.assertEqual(calls, 0)
            with self.assertRaisesRegex(GoldMinerError, "missing worker configuration"):
                server.get_service()
            self.assertEqual(calls, 1)
        finally:
            server.server_close()


class SubprocessRunnerTests(unittest.TestCase):
    def test_builds_argument_array_for_the_existing_cli(self) -> None:
        runner = SubprocessGoldMinerRunner(executable="/usr/bin/python3")
        command = runner.command(
            Path("/work/source.mov"),
            Path("/work/run"),
            transcript=Path("/work/source.srt"),
            top_k=8,
            max_duration_seconds=60,
            handle_ms=250,
        )
        self.assertEqual(command[:3], ["/usr/bin/python3", "-m", "goldminer"])
        self.assertIn("--resume", command)
        self.assertEqual(command[-2:], ["--transcript", "/work/source.srt"])


if __name__ == "__main__":
    unittest.main()
