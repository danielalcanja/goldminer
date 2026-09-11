import json
import tempfile
import unittest
from pathlib import Path

from goldminer.errors import GoldMinerError
from goldminer.providers.transcription import FakeTranscriptionProvider
from goldminer.transcript.artifact import build_transcript, load_transcript
from goldminer.transcript.models import Utterance


class TranscriptArtifactTests(unittest.TestCase):
    def test_import_writes_and_reloads_valid_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "input.srt"
            source.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello world.\n", encoding="utf-8")
            destination = root / "transcript.json"
            transcript, reused = build_transcript(
                destination, media_duration_ms=2000, transcript_path=source
            )
            self.assertFalse(reused)
            self.assertEqual(transcript.schema_version, "1.0")
            self.assertEqual(load_transcript(destination), transcript)
            raw = json.loads(destination.read_text())
            self.assertEqual(raw["utterance_count"], 1)

    def test_resume_requires_matching_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "input.srt"
            destination = root / "transcript.json"
            source.write_text("1\n00:00:00,000 --> 00:00:01,000\nFirst.\n")
            build_transcript(destination, media_duration_ms=3000, transcript_path=source)
            _, reused = build_transcript(
                destination, media_duration_ms=3000, transcript_path=source, resume=True
            )
            self.assertTrue(reused)
            source.write_text("1\n00:00:00,000 --> 00:00:01,000\nChanged.\n")
            changed, reused = build_transcript(
                destination, media_duration_ms=3000, transcript_path=source, resume=True
            )
            self.assertFalse(reused)
            self.assertEqual(changed.utterances[0].original_text, "Changed.")

    def test_timestamp_cannot_exceed_media_duration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "input.vtt"
            source.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nToo late.\n")
            with self.assertRaisesRegex(GoldMinerError, "beyond media duration"):
                build_transcript(root / "transcript.json", media_duration_ms=4000, transcript_path=source)

    def test_fake_provider_obeys_same_validation_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            utterance = Utterance("u000001", 0, 1000, None, "Hello", "Hello")
            provider = FakeTranscriptionProvider([utterance])
            transcript, _ = build_transcript(
                root / "transcript.json",
                media_duration_ms=2000,
                provider=provider,
                audio_path=audio,
            )
            self.assertEqual(transcript.source_type, "provider")
            self.assertEqual(transcript.utterances, (utterance,))

    def test_provider_output_outside_media_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audio = root / "audio.wav"
            audio.write_bytes(b"RIFF" + b"0" * 100)
            invalid = Utterance("u000001", 0, 5000, None, "Hello", "Hello")
            with self.assertRaisesRegex(GoldMinerError, "beyond media duration"):
                build_transcript(
                    root / "transcript.json",
                    media_duration_ms=1000,
                    provider=FakeTranscriptionProvider([invalid]),
                    audio_path=audio,
                )

    def test_damaged_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "transcript.json"
            path.write_text('{"schema_version":"1.0"}')
            with self.assertRaisesRegex(GoldMinerError, "Invalid transcript schema"):
                load_transcript(path)


if __name__ == "__main__":
    unittest.main()
