import unittest

from goldminer.analysis.discovery import discover_candidates, proposal_to_candidate
from goldminer.analysis.models import CandidateProposal, DiscoveryWindow
from goldminer.analysis.windows import build_discovery_windows
from goldminer.errors import GoldMinerError
from goldminer.providers.analysis import FakeAnalysisProvider
from goldminer.transcript.models import CanonicalTranscript, Utterance


def make_transcript(count=80, step_ms=5000):
    utterances = tuple(
        Utterance(
            id=f"u{index:06d}",
            start_ms=(index - 1) * step_ms,
            end_ms=index * step_ms,
            speaker="A",
            original_text=f"Utterance {index}",
            normalized_text=f"Utterance {index}",
        )
        for index in range(1, count + 1)
    )
    return CanonicalTranscript("1.0", "provider", None, "hash", count * step_ms, utterances)


class DiscoveryTests(unittest.TestCase):
    def test_windows_are_utterance_aligned_and_overlap(self):
        transcript = make_transcript()
        windows = build_discovery_windows(
            transcript.utterances, target_duration_ms=100_000, overlap_ms=20_000
        )
        self.assertGreater(len(windows), 1)
        first_ids = {item.id for item in windows[0].utterances}
        second_ids = {item.id for item in windows[1].utterances}
        self.assertTrue(first_ids & second_ids)
        self.assertTrue(all(window.utterances for window in windows))

    def test_story_crossing_nominal_window_boundary_is_discoverable(self):
        transcript = make_transcript()
        windows = build_discovery_windows(
            transcript.utterances, target_duration_ms=100_000, overlap_ms=30_000
        )
        crossing = next(
            window
            for window in windows
            if {"u000018", "u000019", "u000020", "u000021", "u000022"}.issubset(
                {item.id for item in window.utterances}
            )
        )
        proposal = CandidateProposal(
            "story", "u000018", "u000022", "A failure becomes a lesson", "Setup and payoff"
        )
        provider = FakeAnalysisProvider({crossing.id: [proposal]})
        candidates = discover_candidates(transcript, windows, provider)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_utterance_ids[0], "u000018")
        self.assertEqual(candidates[0].source_utterance_ids[-1], "u000022")

    def test_candidate_text_is_derived_from_contiguous_source(self):
        transcript = make_transcript(10)
        window = DiscoveryWindow("w0001", transcript.utterances)
        candidate = proposal_to_candidate(
            CandidateProposal("lesson", "u000003", "u000005", "Idea", "Evidence"),
            window,
            transcript,
        )
        self.assertEqual(candidate.source_utterance_ids, ("u000003", "u000004", "u000005"))
        self.assertEqual(candidate.transcript, "Utterance 3 Utterance 4 Utterance 5")

    def test_provider_cannot_cite_ids_outside_window(self):
        transcript = make_transcript(10)
        window = DiscoveryWindow("w0001", transcript.utterances[:5])
        with self.assertRaisesRegex(GoldMinerError, "outside w0001"):
            proposal_to_candidate(
                CandidateProposal("story", "u000004", "u000008", "Idea", "Evidence"),
                window,
                transcript,
            )

    def test_short_numeric_provider_ids_are_normalized(self):
        transcript = make_transcript(80)
        window = DiscoveryWindow("w0002", transcript.utterances[50:65])
        candidate = proposal_to_candidate(
            CandidateProposal("story", "u057", "u059", "Idea", "Evidence"),
            window,
            transcript,
        )
        self.assertEqual(candidate.source_utterance_ids, ("u000057", "u000058", "u000059"))

    def test_overlapping_duplicate_proposals_merge_categories(self):
        transcript = make_transcript(10)
        window = DiscoveryWindow("w0001", transcript.utterances)
        provider = FakeAnalysisProvider(
            {
                "w0001": [
                    CandidateProposal(
                        "story", "u000002", "u000007", "Failure created an important lesson", "Story"
                    ),
                    CandidateProposal(
                        "lesson", "u000002", "u000006", "An important lesson came from failure", "Lesson"
                    ),
                ]
            }
        )
        candidates = discover_candidates(transcript, (window,), provider)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].detector_categories, ("lesson", "story"))


if __name__ == "__main__":
    unittest.main()
