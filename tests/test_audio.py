import subprocess
import tempfile
import unittest
from pathlib import Path

from goldminer.errors import GoldMinerError
from goldminer.media.audio import extract_analysis_audio
from goldminer.media.ffmpeg import FFmpegTools


class AudioExtractionTests(unittest.TestCase):
    def test_extracts_to_temp_then_replaces_destination(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source file.mp4"
            source.write_bytes(b"source remains unchanged")
            original = source.read_bytes()
            destination = root / "run" / "audio.wav"
            destination.parent.mkdir()
            calls = []

            def runner(args, **kwargs):
                calls.append(args)
                Path(args[-1]).write_bytes(b"RIFF" + b"0" * 100)
                return subprocess.CompletedProcess(args, 0, "", "")

            reused = extract_analysis_audio(
                source, destination, FFmpegTools("ffmpeg", "ffprobe"), runner=runner
            )
            self.assertFalse(reused)
            self.assertTrue(destination.exists())
            self.assertEqual(source.read_bytes(), original)
            self.assertIsInstance(calls[0], list)
            self.assertIn(str(source), calls[0])

    def test_resume_reuses_nonempty_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            destination = root / "audio.wav"
            destination.write_bytes(b"RIFF" + b"0" * 100)

            def runner(*args, **kwargs):
                self.fail("runner should not be called")

            self.assertTrue(
                extract_analysis_audio(
                    source,
                    destination,
                    FFmpegTools("ffmpeg", "ffprobe"),
                    resume=True,
                    runner=runner,
                )
            )

    def test_failed_extract_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            destination = root / "audio.wav"

            def runner(args, **kwargs):
                Path(args[-1]).write_bytes(b"partial")
                return subprocess.CompletedProcess(args, 1, "", "boom")

            with self.assertRaisesRegex(GoldMinerError, "boom"):
                extract_analysis_audio(
                    source, destination, FFmpegTools("ffmpeg", "ffprobe"), runner=runner
                )
            self.assertFalse(destination.exists())
            self.assertFalse((root / ".audio.wav.tmp.wav").exists())


if __name__ == "__main__":
    unittest.main()
