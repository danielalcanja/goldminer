import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from goldminer.errors import GoldMinerError
from goldminer.media.ffmpeg import FFmpegTools
from goldminer.output.cutter import cut_selected_clips, stable_clip_name
from goldminer.transcript.models import Utterance


SELECTION = {
    "rank": 1, "id": "c00001", "category": "contrarian_take",
    "final_score": 82.4, "start_ms": 1000, "end_ms": 4000,
}


class CutterTests(unittest.TestCase):
    def test_stable_filename_is_safe(self):
        self.assertEqual(stable_clip_name(1, "Contrarian Take!", 82.4), "01_contrarian_take_82.mp4")

    def test_atomic_cut_uses_argument_array_and_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source video.mp4"
            source.write_bytes(b"source unchanged")
            original = source.read_bytes()
            clips = root / "clips"
            clips.mkdir()
            calls = []

            def runner(args, **kwargs):
                calls.append(args)
                if args[0] == "ffmpeg":
                    Path(args[-1]).write_bytes(b"playable")
                    return subprocess.CompletedProcess(args, 0, "", "")
                return subprocess.CompletedProcess(args, 0, json.dumps({"format": {"duration": "4.000"}}), "")

            result = cut_selected_clips(
                source, (SELECTION,), clips, FFmpegTools("ffmpeg", "ffprobe"),
                source_sha256="abc", media_duration_ms=10000, resume=True, runner=runner,
            )
            self.assertFalse(result[0].reused)
            self.assertEqual(result[0].start_ms, 500)
            self.assertEqual(result[0].end_ms, 4500)
            self.assertIsInstance(calls[0], list)
            self.assertEqual(source.read_bytes(), original)

            calls.clear()
            reused = cut_selected_clips(
                source, (SELECTION,), clips, FFmpegTools("ffmpeg", "ffprobe"),
                source_sha256="abc", media_duration_ms=10000, resume=True, runner=runner,
            )
            self.assertTrue(reused[0].reused)
            self.assertFalse(any(call[0] == "ffmpeg" for call in calls))

    def test_failed_cut_removes_temporary_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            clips = root / "clips"
            clips.mkdir()

            def runner(args, **kwargs):
                Path(args[-1]).write_bytes(b"partial")
                return subprocess.CompletedProcess(args, 1, "", "encoder failed")

            with self.assertRaisesRegex(GoldMinerError, "encoder failed"):
                cut_selected_clips(source, (SELECTION,), clips, FFmpegTools("ffmpeg", "ffprobe"), source_sha256="abc", media_duration_ms=10000, runner=runner)
            self.assertFalse(any(path.name.endswith(".tmp.mp4") for path in clips.iterdir()))

    def test_low_disk_space_fails_before_ffmpeg(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            clips = root / "clips"
            clips.mkdir()
            disk = shutil._ntuple_diskusage(total=1, used=1, free=1)
            with patch("goldminer.output.cutter.shutil.disk_usage", return_value=disk):
                with self.assertRaisesRegex(GoldMinerError, "Not enough free disk space"):
                    cut_selected_clips(
                        source, (SELECTION,), clips, FFmpegTools("ffmpeg", "ffprobe"),
                        source_sha256="abc", media_duration_ms=10000,
                    )

    def test_handles_never_push_final_clip_over_60_seconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            clips = root / "clips"
            clips.mkdir()
            selection = dict(SELECTION, start_ms=10000, end_ms=70000)

            def runner(args, **kwargs):
                if args[0] == "ffmpeg":
                    Path(args[-1]).write_bytes(b"playable")
                    return subprocess.CompletedProcess(args, 0, "", "")
                return subprocess.CompletedProcess(args, 0, json.dumps({"format": {"duration": "60.000"}}), "")

            result = cut_selected_clips(
                source, (selection,), clips, FFmpegTools("ffmpeg", "ffprobe"),
                source_sha256="abc", media_duration_ms=100000, runner=runner,
            )
            self.assertEqual(result[0].end_ms - result[0].start_ms, 60000)

    def test_handles_are_clamped_to_silent_transcript_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            clips = root / "clips"
            clips.mkdir()
            utterances = (
                Utterance("u000001", 0, 950, "A", "Previous speech.", "normalized"),
                Utterance("u000002", 1000, 4000, "A", "Selected speech.", "normalized"),
                Utterance("u000003", 4100, 5000, "A", "Next speech.", "normalized"),
            )

            def runner(args, **kwargs):
                if args[0] == "ffmpeg":
                    Path(args[-1]).write_bytes(b"playable")
                    return subprocess.CompletedProcess(args, 0, "", "")
                return subprocess.CompletedProcess(
                    args, 0, json.dumps({"format": {"duration": "3.150"}}), ""
                )

            result = cut_selected_clips(
                source, (SELECTION,), clips, FFmpegTools("ffmpeg", "ffprobe"),
                source_sha256="abc", media_duration_ms=10_000,
                utterances=utterances, runner=runner,
            )
            self.assertEqual(result[0].start_ms, 950)
            self.assertEqual(result[0].end_ms, 4100)

    def test_phrase_level_boundaries_do_not_restore_trimmed_speech(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            clips = root / "clips"
            clips.mkdir()
            selection = dict(SELECTION, start_ms=1500, end_ms=3500)
            utterances = (
                Utterance(
                    "u000001", 1000, 4000, "A", "Trimmed selected speech.", "normalized"
                ),
            )

            def runner(args, **kwargs):
                if args[0] == "ffmpeg":
                    Path(args[-1]).write_bytes(b"playable")
                    return subprocess.CompletedProcess(args, 0, "", "")
                return subprocess.CompletedProcess(
                    args, 0, json.dumps({"format": {"duration": "2.000"}}), ""
                )

            result = cut_selected_clips(
                source, (selection,), clips, FFmpegTools("ffmpeg", "ffprobe"),
                source_sha256="abc", media_duration_ms=10_000,
                utterances=utterances, runner=runner,
            )
            self.assertEqual(result[0].start_ms, 1500)
            self.assertEqual(result[0].end_ms, 3500)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
    def test_generated_media_cut_duration_is_within_tolerance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.mp4"
            subprocess.run([
                shutil.which("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=30:d=3",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
            ], check=True)
            clips = root / "clips"
            clips.mkdir()
            selection = dict(SELECTION, start_ms=500, end_ms=2000)
            result = cut_selected_clips(
                source, (selection,), clips,
                FFmpegTools(shutil.which("ffmpeg"), shutil.which("ffprobe")),
                source_sha256="fixture", media_duration_ms=3000, handle_ms=0,
            )
            self.assertLessEqual(abs(result[0].measured_duration_ms - 1500), 750)
            self.assertGreater(result[0].path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
