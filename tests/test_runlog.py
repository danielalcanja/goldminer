import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from goldminer.output.runlog import RunLogger


class RunLoggerTests(unittest.TestCase):
    def test_failure_logging_never_masks_original_error_when_disk_write_fails(self):
        with tempfile.TemporaryDirectory() as td:
            logger = RunLogger(Path(td) / "run.jsonl")
            with patch("goldminer.output.runlog.os.open", side_effect=OSError("disk full")):
                logger.failure("delivery", 0.0, RuntimeError("original"))

    def test_writes_parseable_append_only_json_lines(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "logs" / "run.jsonl"
            logger = RunLogger(path)
            started = logger.start("example", source_path="input.mp4")
            logger.complete("example", started, count=3)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["status"] for record in records], ["started", "completed"])
            self.assertEqual(records[0]["run_id"], records[1]["run_id"])
            self.assertEqual(records[1]["count"], 3)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_failure_records_type_and_message(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.jsonl"
            logger = RunLogger(path)
            started = logger.start("stage")
            logger.failure("stage", started, ValueError("bad input"))
            record = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_type"], "ValueError")
            self.assertEqual(record["error"], "bad input")


if __name__ == "__main__":
    unittest.main()
