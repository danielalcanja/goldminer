import json
import tempfile
import unittest
from pathlib import Path

from goldminer.output.delivery import write_manifest, write_report


class DeliveryArtifactTests(unittest.TestCase):
    def test_report_escapes_transcript_and_supports_no_cut(self):
        selection = {
            "rank": 1, "id": "c1", "category": "lesson", "final_score": 80,
            "duration_sec": 30, "confidence": "high", "comparative_reason": "<strong>",
            "transcript": "<script>alert(1)</script>", "start_ms": 0, "end_ms": 30000,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.html"
            write_report(path, (selection,), ())
            text = path.read_text()
            self.assertNotIn("<script>alert", text)
            self.assertIn("&lt;script&gt;", text)
            self.assertIn("Video cutting was skipped", text)

    def test_manifest_is_complete_and_parseable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            path = root / "run_manifest.json"
            write_manifest(
                path, source=source, source_sha256="abc", duration_ms=1000,
                config={"top_k": 1}, counts={"selected": 1}, models={"analysis": "fake"},
                artifacts=("ranking.json", "report.html"), clips=(), no_cut=True,
            )
            value = json.loads(path.read_text())
            self.assertEqual(value["status"], "completed")
            self.assertTrue(value["no_cut"])
            self.assertEqual(value["input"]["sha256"], "abc")
            self.assertIn("config_hash", value)


if __name__ == "__main__":
    unittest.main()
