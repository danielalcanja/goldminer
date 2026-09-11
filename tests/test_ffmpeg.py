import subprocess
import unittest
from pathlib import Path

from goldminer.errors import GoldMinerError
from goldminer.media.ffmpeg import FFmpegTools, probe_duration_ms, run_checked, validate_tools


class FFmpegTests(unittest.TestCase):
    def test_tools_are_invoked_with_argument_arrays(self):
        calls = []

        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, "ok", "")

        validate_tools(FFmpegTools("ffmpeg", "ffprobe"), runner=runner)
        self.assertEqual(calls, [["ffmpeg", "-version"], ["ffprobe", "-version"]])

    def test_probe_converts_seconds_to_milliseconds(self):
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, '{"format":{"duration":"12.3456"}}', "")

        duration = probe_duration_ms(Path("input.mp4"), FFmpegTools("ffmpeg", "ffprobe"), runner=runner)
        self.assertEqual(duration, 12346)

    def test_process_failure_includes_tool_error(self):
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 1, "", "invalid media")

        with self.assertRaisesRegex(GoldMinerError, "invalid media"):
            run_checked(["ffmpeg", "-i", "bad"], runner=runner)


if __name__ == "__main__":
    unittest.main()
