import unittest
from pathlib import Path

from goldminer.errors import GoldMinerError
from goldminer.transcript.parser import parse_srt, parse_timestamp, parse_transcript_file, parse_vtt


FIXTURES = Path(__file__).parent / "fixtures"


class TranscriptParserTests(unittest.TestCase):
    def test_parses_srt_with_speaker_and_stable_ids(self):
        source_type, utterances = parse_transcript_file(FIXTURES / "sample.srt")
        self.assertEqual(source_type, "srt")
        self.assertEqual([item.id for item in utterances], ["u000001", "u000002"])
        self.assertEqual(utterances[0].speaker, "Daniel")
        self.assertEqual(utterances[0].start_ms, 500)
        self.assertEqual(utterances[0].original_text, "I thought the hard part was writing the code.")

    def test_parses_vtt_voice_markup_and_settings(self):
        source_type, utterances = parse_transcript_file(FIXTURES / "sample.vtt")
        self.assertEqual(source_type, "vtt")
        self.assertEqual(utterances[0].speaker, "Daniel")
        self.assertNotIn("<v", utterances[0].original_text)

    def test_duplicate_overlapping_fragments_collapse(self):
        utterances = parse_srt(
            "1\n00:00:01,000 --> 00:00:03,000\nSame idea\n\n"
            "2\n00:00:02,500 --> 00:00:04,000\nSame idea\n"
        )
        self.assertEqual(len(utterances), 1)
        self.assertEqual((utterances[0].start_ms, utterances[0].end_ms), (1000, 4000))

    def test_out_of_order_blocks_are_canonically_sorted(self):
        utterances = parse_vtt(
            "WEBVTT\n\n00:00:05.000 --> 00:00:06.000\nLater\n\n"
            "00:00:01.000 --> 00:00:02.000\nEarlier\n"
        )
        self.assertEqual([item.original_text for item in utterances], ["Earlier", "Later"])

    def test_malformed_timestamp_fails(self):
        with self.assertRaisesRegex(GoldMinerError, "Malformed"):
            parse_timestamp("00:61:00.000")

    def test_empty_input_fails(self):
        with self.assertRaisesRegex(GoldMinerError, "no timed utterances"):
            parse_srt("")

    def test_unsupported_extension_fails(self):
        with self.assertRaisesRegex(GoldMinerError, "Unsupported transcript format"):
            parse_transcript_file(FIXTURES / "unknown.txt")


if __name__ == "__main__":
    unittest.main()
