import json
import unittest

from goldminer.analysis.boundaries import apply_evaluation, build_boundary_context
from goldminer.analysis.scoring import DIMENSION_WEIGHTS, EDIT_VARIANTS
from goldminer.errors import GoldMinerError
from goldminer.providers.scoring import OpenAIScoringProvider
from tests.test_scoring import candidate_fixture, evaluation_fixture, transcript_fixture


def raw_evaluation(candidate_id="c00001", variant_id="hook_first"):
    return {
        "candidate_id": candidate_id,
        "variant_id": variant_id,
        "start_utterance_id": "u000004",
        "end_utterance_id": "u000011",
        "start_boundary_text": "Exact source 4",
        "end_boundary_text": "Exact source 11",
        "category": "lesson",
        "secondary_categories": ["story"],
        "dimensions": {
            name: {"raw_score": 8, "evidence": f"Evidence for {name}."}
            for name in DIMENSION_WEIGHTS
        },
        "penalties": [
            {"type": "slow_setup", "points": 3, "evidence": "The hook takes time."}
        ],
        "unsafe_extraction": False,
        "non_contiguous_required": False,
        "opening_is_clear": True,
        "opening_problem": "none",
        "central_idea_is_strong": True,
        "central_idea_problem": "none",
        "opening_evidence": "The opening stands alone.",
        "central_idea_evidence": "The lesson is specific and complete.",
        "arc_is_complete": True,
        "payoff_type": "conclusion",
        "payoff_text": "Exact source 11",
        "arc_evidence": "The final line completes the lesson.",
        "boundary_rationale": "Natural complete range.",
        "why_selected": "Strong lesson.",
    }


def raw_derived_review(candidate_id="c00001"):
    return {
        "candidate_id": candidate_id,
        "dimensions": {
            name: {"raw_score": 8, "evidence": f"Fresh exact-edit evidence for {name}."}
            for name in DIMENSION_WEIGHTS
        },
        "penalties": [],
        "opening_is_clear": True,
        "opening_problem": "none",
        "central_idea_is_strong": True,
        "central_idea_problem": "none",
        "opening_evidence": "The literal first words establish the topic.",
        "central_idea_evidence": "The exact edit develops one idea.",
        "arc_is_complete": True,
        "payoff_type": "conclusion",
        "payoff_text": "Exact source 11",
        "arc_evidence": "The final line resolves the opening idea.",
    }


class ScoringProviderTests(unittest.TestCase):
    def test_strict_structured_batch_is_parsed(self):
        captured = {}

        def transport(url, headers, body, timeout):
            captured["body"] = json.loads(body)
            result = {
                "evaluations": [raw_evaluation(variant_id=value) for value in EDIT_VARIANTS]
            }
            return 200, json.dumps({"output_text": json.dumps(result)}).encode()

        context = build_boundary_context(candidate_fixture(), transcript_fixture())
        evaluations = OpenAIScoringProvider(api_key="test", transport=transport).evaluate([context])
        self.assertEqual(len(evaluations), 3)
        self.assertEqual(evaluations[0].dimensions["hook"].raw_score, 8)
        self.assertEqual(evaluations[0].start_boundary_text, "Exact source 4")
        self.assertTrue(captured["body"]["text"]["format"]["strict"])
        self.assertIn("trim inside", captured["body"]["instructions"])

    def test_missing_candidate_evaluation_is_rejected(self):
        def transport(url, headers, body, timeout):
            return 200, json.dumps({"output_text": '{"evaluations":[]}'}).encode()

        context = build_boundary_context(candidate_fixture(), transcript_fixture())
        with self.assertRaisesRegex(GoldMinerError, "candidate/variant IDs mismatch"):
            OpenAIScoringProvider(api_key="test", transport=transport).evaluate([context])

    def test_invalid_score_is_rejected(self):
        values = [raw_evaluation(variant_id=item) for item in EDIT_VARIANTS]
        values[0]["dimensions"]["hook"]["raw_score"] = 11

        def transport(url, headers, body, timeout):
            return 200, json.dumps(
                {"output_text": json.dumps({"evaluations": values})}
            ).encode()

        context = build_boundary_context(candidate_fixture(), transcript_fixture())
        with self.assertRaisesRegex(GoldMinerError, "between 0 and 10"):
            OpenAIScoringProvider(api_key="test", transport=transport).evaluate([context])

    def test_exact_derived_edit_is_independently_reviewed(self):
        captured = {}

        def transport(url, headers, body, timeout):
            captured["body"] = json.loads(body)
            result = {"reviews": [raw_derived_review()]}
            return 200, json.dumps({"output_text": json.dumps(result)}).encode()

        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        candidate = apply_evaluation(evaluation_fixture(), context, transcript)
        reviews = OpenAIScoringProvider(
            api_key="test", transport=transport
        ).review_derived((candidate,), {candidate.id: "A useful lesson"})

        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].dimensions["hook"].raw_score, 8)
        self.assertTrue(captured["body"]["text"]["format"]["strict"])
        self.assertIn("exact proposed clip", captured["body"]["input"].lower())
        self.assertIn("multiple_competing_ideas", captured["body"]["instructions"])


if __name__ == "__main__":
    unittest.main()
