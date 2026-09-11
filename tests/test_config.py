import tempfile
import unittest
from pathlib import Path

from goldminer.config import RunConfig
from goldminer.errors import GoldMinerError


class RunConfigTests(unittest.TestCase):
    def test_missing_video_is_actionable(self):
        with self.assertRaisesRegex(GoldMinerError, "does not exist"):
            RunConfig(Path("missing.mp4"), Path("out")).validate()

    def test_top_k_must_be_positive(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "podcast.mp4"
            video.write_bytes(b"media")
            with self.assertRaisesRegex(GoldMinerError, "at least 1"):
                RunConfig(video, Path(td) / "out", top_k=0).validate()

    def test_max_duration_must_be_supported(self):
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "podcast.mp4"
            video.write_bytes(b"media")
            with self.assertRaisesRegex(GoldMinerError, "between 15 and 120"):
                RunConfig(video, Path(td) / "out", max_clip_seconds=121).validate()


if __name__ == "__main__":
    unittest.main()
