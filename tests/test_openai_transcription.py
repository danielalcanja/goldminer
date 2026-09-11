import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from goldminer.errors import GoldMinerError
from goldminer.media.ffmpeg import FFmpegTools
from goldminer.providers.transcription import OpenAITranscriptionProvider


class OpenAITranscriptionProviderTests(unittest.TestCase):
    def _runner(self, args, **kwargs):
        Path(args[-1]).write_bytes(b"ID3" + b"0" * 100)
        return subprocess.CompletedProcess(args, 0, "", "")

    def test_converts_diarized_segments_to_canonical_utterances(self):
        captured = {}

        def transport(url, headers, body, timeout):
            captured.update(url=url, headers=headers, body=body, timeout=timeout)
            payload = {
                "segments": [
                    {"start": 0.25, "end": 1.5, "speaker": "A", "text": " Hello there. "},
                    {"start": 1.5, "end": 3.0, "speaker": "B", "text": "General Kenobi."},
                ]
            }
            return 200, json.dumps(payload).encode()

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            provider = OpenAITranscriptionProvider(
                api_key="test-key",
                transport=transport,
                runner=self._runner,
                tools=FFmpegTools("ffmpeg", "ffprobe"),
            )
            utterances = provider.transcribe(audio, 3000)
            self.assertEqual(len(utterances), 2)
            self.assertEqual((utterances[0].start_ms, utterances[0].end_ms), (250, 1500))
            self.assertEqual(utterances[0].speaker, "A")
            self.assertEqual(utterances[0].original_text, "Hello there.")
            self.assertIn("/audio/transcriptions", captured["url"])
            self.assertIn(b'gpt-4o-transcribe-diarize', captured["body"])
            self.assertIn(b'diarized_json', captured["body"])
            self.assertFalse((audio.parent / ".openai-transcription-0001.mp3").exists())

    def test_missing_api_key_fails_before_media_or_network_work(self):
        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "audio.wav"
            audio.write_bytes(b"RIFF")
            with self.assertRaisesRegex(GoldMinerError, "OPENAI_API_KEY"):
                OpenAITranscriptionProvider(api_key="").transcribe(audio, 1000)

    def test_api_error_is_actionable_and_upload_is_deleted(self):
        def transport(url, headers, body, timeout):
            return 401, b'{"error":{"message":"Incorrect API key"}}'

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            provider = OpenAITranscriptionProvider(
                api_key="bad-key",
                transport=transport,
                runner=self._runner,
                tools=FFmpegTools("ffmpeg", "ffprobe"),
            )
            with self.assertRaisesRegex(GoldMinerError, "HTTP 401.*Incorrect API key"):
                provider.transcribe(audio, 1000)
            self.assertFalse((audio.parent / ".openai-transcription-0001.mp3").exists())

    def test_response_without_timestamped_segments_fails(self):
        def transport(url, headers, body, timeout):
            return 200, b'{"text":"untimed text"}'

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            provider = OpenAITranscriptionProvider(
                api_key="test-key",
                transport=transport,
                runner=self._runner,
                tools=FFmpegTools("ffmpeg", "ffprobe"),
            )
            with self.assertRaisesRegex(GoldMinerError, "no timestamped segments"):
                provider.transcribe(audio, 1000)

    def test_zero_length_brief_segment_gets_minimal_valid_span(self):
        def transport(url, headers, body, timeout):
            return 200, b'{"segments":[{"start":1.5,"end":1.5,"speaker":"A","text":"Yes."}]}'

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            provider = OpenAITranscriptionProvider(
                api_key="test-key",
                transport=transport,
                runner=self._runner,
                tools=FFmpegTools("ffmpeg", "ffprobe"),
            )
            utterances = provider.transcribe(audio, 3000)
            self.assertEqual((utterances[0].start_ms, utterances[0].end_ms), (1500, 1750))
            self.assertEqual(utterances[0].original_text, "Yes.")

    def test_inverted_segment_is_rejected(self):
        def transport(url, headers, body, timeout):
            return 200, b'{"segments":[{"start":2.0,"end":1.0,"text":"Broken."}]}'

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            provider = OpenAITranscriptionProvider(
                api_key="test-key",
                transport=transport,
                runner=self._runner,
                tools=FFmpegTools("ffmpeg", "ffprobe"),
            )
            with self.assertRaisesRegex(GoldMinerError, "ends before it starts"):
                provider.transcribe(audio, 3000)

    def test_long_audio_is_chunked_and_timestamps_are_offset(self):
        requests = []

        def transport(url, headers, body, timeout):
            requests.append(body)
            return 200, b'{"segments":[{"start":1.0,"end":2.0,"speaker":"A","text":"Chunk."}]}'

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            provider = OpenAITranscriptionProvider(
                api_key="test-key",
                transport=transport,
                runner=self._runner,
                tools=FFmpegTools("ffmpeg", "ffprobe"),
                chunk_duration_seconds=1200,
            )
            utterances = provider.transcribe(audio, 2_500_000)
            self.assertEqual(len(requests), 3)
            self.assertEqual(
                [(item.start_ms, item.end_ms) for item in utterances],
                [(1000, 2000), (1_201_000, 1_202_000), (2_401_000, 2_402_000)],
            )
            self.assertEqual([item.id for item in utterances], ["u000001", "u000002", "u000003"])
            self.assertFalse(list(audio.parent.glob(".openai-transcription-*.mp3")))

    def test_completed_chunk_is_reused_without_another_paid_request(self):
        calls = []

        def transport(url, headers, body, timeout):
            calls.append(url)
            return 200, b'{"segments":[{"start":0.0,"end":1.0,"text":"Cached."}]}'

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            provider = OpenAITranscriptionProvider(
                api_key="test-key",
                transport=transport,
                runner=self._runner,
                tools=FFmpegTools("ffmpeg", "ffprobe"),
            )
            first = provider.transcribe(audio, 1000)

            def runner_must_not_run(*args, **kwargs):
                self.fail("FFmpeg must not run for a cached chunk")

            cached_provider = OpenAITranscriptionProvider(
                api_key="test-key",
                transport=lambda *args: self.fail("API must not run for a cached chunk"),
                runner=runner_must_not_run,
                tools=FFmpegTools("ffmpeg", "ffprobe"),
            )
            second = cached_provider.transcribe(audio, 1000)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 1)

    def test_timeout_is_retried_and_reported_as_goldminer_error(self):
        def transport(url, headers, body, timeout):
            raise TimeoutError("read timed out")

        with tempfile.TemporaryDirectory() as td, patch(
            "goldminer.providers.transcription.time.sleep", return_value=None
        ):
            audio = Path(td) / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            provider = OpenAITranscriptionProvider(
                api_key="test-key",
                transport=transport,
                runner=self._runner,
                tools=FFmpegTools("ffmpeg", "ffprobe"),
                max_retries=1,
            )
            with self.assertRaisesRegex(GoldMinerError, "HTTP 599.*network timeout"):
                provider.transcribe(audio, 1000)


if __name__ == "__main__":
    unittest.main()
