import json
import unittest

from goldminer.analysis.detectors import DETECTORS
from goldminer.analysis.models import DiscoveryWindow
from goldminer.errors import GoldMinerError
from goldminer.providers.analysis import OpenAIAnalysisProvider
from goldminer.transcript.models import Utterance


class OpenAIAnalysisProviderTests(unittest.TestCase):
    def _window(self):
        return DiscoveryWindow(
            "w0001",
            (
                Utterance("u000001", 0, 1000, "A", "A clear setup.", "A clear setup."),
                Utterance("u000002", 1000, 3000, "A", "A useful payoff.", "A useful payoff."),
            ),
        )

    def test_structured_response_becomes_proposal(self):
        captured = {}

        def transport(url, headers, body, timeout):
            captured["request"] = json.loads(body)
            structured = {
                "candidates": [
                    {
                        "detector": "lesson",
                        "start_utterance_id": "u000001",
                        "end_utterance_id": "u000002",
                        "core_idea": "A useful lesson",
                        "evidence": "The setup resolves with a takeaway.",
                    }
                ]
            }
            response = {"output": [{"content": [{"type": "output_text", "text": json.dumps(structured)}]}]}
            return 200, json.dumps(response).encode()

        provider = OpenAIAnalysisProvider(api_key="test", transport=transport)
        proposals = provider.discover(self._window())
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].detector, "lesson")
        request = captured["request"]
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(all(detector in request["input"] for detector in DETECTORS))

    def test_invalid_detector_is_rejected(self):
        def transport(url, headers, body, timeout):
            structured = {
                "candidates": [{
                    "detector": "viral_magic",
                    "start_utterance_id": "u000001",
                    "end_utterance_id": "u000002",
                    "core_idea": "Idea",
                    "evidence": "Evidence",
                }]
            }
            return 200, json.dumps({"output_text": json.dumps(structured)}).encode()

        with self.assertRaisesRegex(GoldMinerError, "Invalid candidate"):
            OpenAIAnalysisProvider(api_key="test", transport=transport).discover(self._window())


if __name__ == "__main__":
    unittest.main()
