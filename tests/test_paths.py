import tempfile
import unittest
from pathlib import Path

from goldminer.errors import GoldMinerError
from goldminer.output.paths import create_run_paths, default_output


class OutputPathTests(unittest.TestCase):
    def test_default_output_is_deterministic_and_local(self):
        self.assertEqual(
            default_output(Path("show.mp4"), Path("/tmp/work")),
            Path("/tmp/work").resolve() / "show_clips",
        )

    def test_source_cannot_be_inside_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "run"
            output.mkdir()
            source = output / "podcast.mp4"
            source.write_bytes(b"media")
            with self.assertRaisesRegex(GoldMinerError, "contain the source"):
                create_run_paths(output, source)


if __name__ == "__main__":
    unittest.main()
