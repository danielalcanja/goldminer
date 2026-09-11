import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from goldminer.analysis.boundaries import build_boundary_context, apply_evaluation
from goldminer.analysis.models import Candidate
from goldminer.analysis.scoring import (
    DIMENSION_WEIGHTS,
    DerivedEditReview,
    DimensionJudgment,
    ModelEvaluation,
    PenaltyJudgment,
    calculate_score,
)
from goldminer.analysis.scored_artifact import (
    SCORING_BATCH_SIZE,
    CachedScoringProvider,
    score_candidates,
)
from goldminer.errors import GoldMinerError
from goldminer.providers.scoring import FakeScoringProvider
from goldminer.transcript.models import CanonicalTranscript, Utterance


def transcript_fixture():
    utterances = tuple(
        Utterance(
            f"u{index:06d}",
            (index - 1) * 5000,
            index * 5000,
            "A",
            f"Exact source {index}.",
            f"Exact source {index}.",
        )
        for index in range(1, 16)
    )
    return CanonicalTranscript("1.0", "provider", None, "hash", 75000, utterances)


def candidate_fixture():
    return Candidate(
        "c00001",
        20000,
        50000,
        tuple(f"u{index:06d}" for index in range(5, 11)),
        ("lesson",),
        "raw",
        "A useful lesson",
        "Evidence",
        ("w0001",),
    )


def evaluation_fixture(**overrides):
    values = dict(
        candidate_id="c00001",
        start_utterance_id="u000004",
        end_utterance_id="u000011",
        category="lesson",
        secondary_categories=("story",),
        dimensions={name: DimensionJudgment(10, f"Evidence for {name}.") for name in DIMENSION_WEIGHTS},
        penalties=(),
        unsafe_extraction=False,
        non_contiguous_required=False,
        boundary_rationale="Added the required setup and payoff.",
        why_selected="Complete and useful.",
    )
    values.update(overrides)
    return ModelEvaluation(**values)


class ScoringTests(unittest.TestCase):
    def test_scoring_uses_one_candidate_per_resumable_request(self):
        self.assertEqual(SCORING_BATCH_SIZE, 1)

    def test_best_publishable_edit_variant_is_selected_and_all_are_recorded(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        dimensions = {
            name: DimensionJudgment(8, f"Evidence for {name}.")
            for name in DIMENSION_WEIGHTS
        }
        weak_dimensions = dict(dimensions)
        weak_dimensions["hook"] = DimensionJudgment(6, "The opening is less immediate.")
        evaluations = (
            evaluation_fixture(
                variant_id="hook_first",
                start_utterance_id="u000004",
                end_utterance_id="u000011",
                dimensions=weak_dimensions,
            ),
            evaluation_fixture(
                variant_id="concise",
                start_utterance_id="u000005",
                end_utterance_id="u000009",
                dimensions=dimensions,
            ),
            evaluation_fixture(
                variant_id="full_arc",
                start_utterance_id="u000004",
                end_utterance_id="u000011",
                dimensions={
                    name: DimensionJudgment(9, f"Strong evidence for {name}.")
                    for name in DIMENSION_WEIGHTS
                },
            ),
        )

        scored = score_candidates(
            (context,),
            transcript,
            FakeScoringProvider({"c00001": evaluations}),
        )

        self.assertEqual(scored[0].variant_id, "full_arc")
        self.assertGreaterEqual(len(scored[0].alternative_edits), 3)
        self.assertEqual(
            [
                item["variant_id"]
                for item in scored[0].alternative_edits
                if item["variant_id"] in {"hook_first", "concise", "full_arc"}
            ],
            ["hook_first", "concise", "full_arc"],
        )
        self.assertEqual(
            [
                item["selected"]
                for item in scored[0].alternative_edits
                if item["variant_id"] in {"hook_first", "concise", "full_arc"}
            ],
            [False, False, True],
        )

    def test_recombines_clear_opening_with_complete_ending(self):
        utterances = (
            Utterance("u000001", 0, 8_000, "A", "My long soft setup.", "normalized"),
            Utterance(
                "u000002", 8_000, 18_000, "A",
                "Use clear requirements before asking AI to build.", "normalized",
            ),
            Utterance(
                "u000003", 18_000, 28_000, "A",
                "Ask AI to test the result and then you go there.", "normalized",
            ),
            Utterance(
                "u000004", 28_000, 38_000, "A",
                "Review it yourself; that workflow works really well.", "normalized",
            ),
        )
        transcript = CanonicalTranscript("1.0", "provider", None, "hash", 38_000, utterances)
        candidate = Candidate(
            "c00001", 0, 38_000, tuple(item.id for item in utterances),
            ("framework",), "raw", "A complete AI review workflow.", "Evidence", ("w0001",),
        )
        high = {
            name: DimensionJudgment(9, f"Strong evidence for {name}.")
            for name in DIMENSION_WEIGHTS
        }
        medium = {
            name: DimensionJudgment(8, f"Evidence for {name}.")
            for name in DIMENSION_WEIGHTS
        }
        evaluations = (
            evaluation_fixture(
                variant_id="hook_first", start_utterance_id="u000001",
                end_utterance_id="u000003", dimensions=medium,
                payoff_text="go there",
            ),
            evaluation_fixture(
                variant_id="concise", start_utterance_id="u000002",
                end_utterance_id="u000003", dimensions=high,
                payoff_text="go there",
            ),
            evaluation_fixture(
                variant_id="full_arc", start_utterance_id="u000001",
                end_utterance_id="u000004", dimensions=medium,
                payoff_text="workflow works really well",
            ),
        )

        scored = score_candidates(
            (build_boundary_context(candidate, transcript),),
            transcript,
            FakeScoringProvider({"c00001": evaluations}),
        )[0]

        self.assertEqual(scored.variant_id, "recombined")
        self.assertTrue(scored.transcript.startswith("Use clear requirements"))
        self.assertTrue(scored.transcript.endswith("works really well."))

    def test_near_over_duration_candidate_is_rescued_by_trimming_setup(self):
        transcript = transcript_fixture()
        context = build_boundary_context(
            candidate_fixture(), transcript, padding_utterances=20
        )
        evaluations = tuple(
            evaluation_fixture(
                variant_id=variant_id,
                start_utterance_id="u000001",
                end_utterance_id="u000013",
            )
            for variant_id in ("hook_first", "concise", "full_arc")
        )

        scored = score_candidates(
            (context,), transcript, FakeScoringProvider({"c00001": evaluations})
        )[0]

        self.assertEqual(scored.variant_id, "duration_rescue")
        self.assertFalse(scored.rejected)
        self.assertLessEqual(scored.end_ms - scored.start_ms, 60_000)
        self.assertTrue(scored.transcript.startswith("Exact source 5."))

    def test_ai_balance_near_miss_gets_the_editorial_start_and_payoff(self):
        utterances = (
            Utterance(
                "u000001", 280_884, 286_634, "B",
                "Man, it feels overwhelming because there is so much happening,", "normalized",
            ),
            Utterance(
                "u000002", 286_834, 294_534, "B",
                "AI is changing so quickly, especially for us developers,", "normalized",
            ),
            Utterance(
                "u000003", 294_734, 306_634, "B",
                "that we don't know how much AI should we use and sometimes I use too much.",
                "normalized",
            ),
            Utterance(
                "u000004", 308_532, 322_382, "B",
                "Too much feels artificial and it does not feel premium.", "normalized",
            ),
            Utterance(
                "u000005", 323_284, 348_484, "B",
                "Too little feels slow, but the right process and middle ground can be really helpful.",
                "normalized",
            ),
        )
        transcript = CanonicalTranscript(
            "1.0", "provider", None, "hash", 350_000, utterances
        )
        candidate = Candidate(
            "c00001", 280_884, 348_484, tuple(item.id for item in utterances),
            ("lesson",), "raw", "Find the right amount of AI.", "Evidence", ("w0001",),
        )
        evaluations = tuple(
            evaluation_fixture(
                variant_id=variant_id,
                start_utterance_id="u000001",
                end_utterance_id="u000005",
                payoff_text="the right process and middle ground can be really helpful",
            )
            for variant_id in ("hook_first", "concise", "full_arc")
        )

        scored = score_candidates(
            (build_boundary_context(candidate, transcript),),
            transcript,
            FakeScoringProvider({"c00001": evaluations}),
        )[0]

        self.assertEqual(scored.variant_id, "duration_rescue")
        self.assertTrue(scored.transcript.startswith("we don't know how much AI"))
        self.assertTrue(scored.transcript.endswith("really helpful."))
        self.assertLessEqual(scored.end_ms - scored.start_ms, 60_000)

    def test_ai_balance_rescue_finds_hook_inside_first_long_utterance(self):
        utterances = (
            Utterance(
                "u000001", 280_884, 306_734, "B",
                "Man, it feels so crazy the way we are doing right now and it feels "
                "overwhelming because there is so much AI, especially for us developers "
                "that we don't know how much AI should we use and sometimes I use too much.",
                "normalized",
            ),
            Utterance(
                "u000002", 308_532, 322_382, "B",
                "Too much feels artificial and it does not feel premium.", "normalized",
            ),
            Utterance(
                "u000003", 323_284, 348_434, "B",
                "Too little feels slow, but the right process and middle ground can be really helpful.",
                "normalized",
            ),
        )
        transcript = CanonicalTranscript(
            "1.0", "provider", None, "hash", 350_000, utterances
        )
        candidate = Candidate(
            "c00001", 280_884, 348_434, tuple(item.id for item in utterances),
            ("lesson",), "raw", "Find the right amount of AI.", "Evidence", ("w0001",),
        )
        evaluations = tuple(
            evaluation_fixture(
                variant_id=variant_id,
                start_utterance_id="u000001",
                end_utterance_id="u000003",
                payoff_text="the right process and middle ground can be really helpful",
            )
            for variant_id in ("hook_first", "concise", "full_arc")
        )

        scored = score_candidates(
            (build_boundary_context(candidate, transcript),),
            transcript,
            FakeScoringProvider({"c00001": evaluations}),
        )[0]

        self.assertEqual(scored.variant_id, "duration_rescue")
        self.assertTrue(scored.transcript.startswith("we don't know how much AI"))
        self.assertLessEqual(scored.end_ms - scored.start_ms, 60_000)

    def test_winning_derived_edit_is_rejected_by_fresh_multiple_idea_review(self):
        transcript = transcript_fixture()
        context = build_boundary_context(
            candidate_fixture(), transcript, padding_utterances=20
        )
        evaluations = tuple(
            evaluation_fixture(
                variant_id=variant_id,
                start_utterance_id="u000001",
                end_utterance_id="u000013",
            )
            for variant_id in ("hook_first", "concise", "full_arc")
        )
        dimensions = {
            name: DimensionJudgment(8, f"Fresh evidence for {name} from the exact edit.")
            for name in DIMENSION_WEIGHTS
        }
        rejecting_review = DerivedEditReview(
            candidate_id="c00001",
            dimensions=dimensions,
            penalties=(),
            opening_is_clear=True,
            opening_problem="none",
            central_idea_is_strong=False,
            central_idea_problem="multiple_competing_ideas",
            opening_evidence="The literal opening is understandable.",
            central_idea_evidence="The exact edit changes from AI review to weekly delivery.",
            arc_is_complete=False,
            payoff_type="none",
            payoff_text="",
            arc_evidence="The ending resolves a different idea than the opening.",
        )
        provider = FakeScoringProvider(
            {"c00001": evaluations},
            derived_reviews={
                ("c00001", 20_000, 65_000): rejecting_review,
            },
        )

        scored = score_candidates((context,), transcript, provider)[0]

        self.assertTrue(scored.rejected)
        rejected_rescue = next(
            item
            for item in scored.alternative_edits
            if item["variant_id"] == "duration_rescue"
        )
        self.assertIn("weak_or_unclear_core_idea", rejected_rescue["rejection_reasons"])

    def test_identical_range_disagreement_rejects_the_clip_conservatively(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        evaluations = (
            evaluation_fixture(variant_id="hook_first"),
            evaluation_fixture(
                variant_id="concise",
                central_idea_is_strong=False,
                central_idea_problem="garbled",
            ),
            evaluation_fixture(variant_id="full_arc"),
        )

        scored = score_candidates(
            (context,), transcript, FakeScoringProvider({"c00001": evaluations})
        )[0]

        self.assertTrue(scored.rejected)
        self.assertIn("conflicting_central_idea_assessment", scored.rejection_reasons)

    def test_overlapping_candidates_share_strong_boundary_options(self):
        transcript = transcript_fixture()
        first = Candidate(
            "c00001", 10_000, 35_000,
            tuple(f"u{index:06d}" for index in range(3, 8)),
            ("lesson",), "raw", "Shared practical lesson", "Evidence", ("w0001",),
        )
        second = Candidate(
            "c00002", 20_000, 45_000,
            tuple(f"u{index:06d}" for index in range(5, 10)),
            ("lesson",), "raw", "Shared practical lesson", "Evidence", ("w0001",),
        )
        first_evaluations = tuple(
            evaluation_fixture(
                variant_id=variant_id,
                start_utterance_id="u000003",
                end_utterance_id="u000006",
            )
            for variant_id in ("hook_first", "concise", "full_arc")
        )
        second_evaluations = tuple(
            evaluation_fixture(
                candidate_id="c00002",
                variant_id=variant_id,
                start_utterance_id="u000004",
                end_utterance_id="u000008",
            )
            for variant_id in ("hook_first", "concise", "full_arc")
        )

        scored = score_candidates(
            (
                build_boundary_context(first, transcript),
                build_boundary_context(second, transcript),
            ),
            transcript,
            FakeScoringProvider(
                {"c00001": first_evaluations, "c00002": second_evaluations}
            ),
        )

        self.assertTrue(
            any(
                item["variant_id"] == "recombined"
                and item["transcript"].startswith("Exact source 3.")
                and item["transcript"].endswith("Exact source 8.")
                for item in scored[0].alternative_edits
            )
        )

    def test_invalid_alternative_boundary_does_not_abort_other_edits(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        dimensions = {
            name: DimensionJudgment(8, f"Evidence for {name}.")
            for name in DIMENSION_WEIGHTS
        }
        evaluations = (
            evaluation_fixture(
                variant_id="hook_first",
                start_boundary_text="words that do not exist",
            ),
            evaluation_fixture(
                variant_id="concise",
                start_utterance_id="u000005",
                end_utterance_id="u000009",
                dimensions=dimensions,
            ),
            evaluation_fixture(variant_id="full_arc"),
        )

        scored = score_candidates(
            (context,),
            transcript,
            FakeScoringProvider({"c00001": evaluations}),
        )

        self.assertEqual(scored[0].variant_id, "full_arc")
        invalid = next(
            item
            for item in scored[0].alternative_edits
            if item["variant_id"] == "hook_first"
        )
        self.assertTrue(invalid["rejected"])
        self.assertIn("invalid_boundary", invalid["rejection_reasons"])

    def test_three_edit_variants_are_validated_and_reused_from_cache(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        evaluations = tuple(
            evaluation_fixture(variant_id=variant_id)
            for variant_id in ("hook_first", "concise", "full_arc")
        )
        calls = 0

        class Provider:
            name = "fake"
            model = "fixture-v1"

            def evaluate(self, contexts):
                nonlocal calls
                calls += 1
                return evaluations

        with TemporaryDirectory() as temporary:
            cached = CachedScoringProvider(
                Provider(), Path(temporary), "candidate-artifact-hash"
            )
            self.assertEqual(len(cached.evaluate((context,))), 3)
            self.assertEqual(len(cached.evaluate((context,))), 3)

        self.assertEqual(calls, 1)

    def test_time_padding_exposes_neighboring_setup_and_payoff(self):
        transcript = transcript_fixture()
        context = build_boundary_context(
            candidate_fixture(), transcript, padding_utterances=1, padding_ms=20_000
        )
        self.assertEqual(context.utterances[0].id, "u000001")
        self.assertEqual(context.utterances[-1].id, "u000015")

    def test_weights_sum_to_100_and_golden_score_is_exact(self):
        dimensions = {name: DimensionJudgment(10, "Evidence") for name in DIMENSION_WEIGHTS}
        penalties = (PenaltyJudgment("slow_setup", 7.5, "The opening is slow."),)
        subtotal, final, points = calculate_score(dimensions, penalties)
        self.assertEqual(sum(DIMENSION_WEIGHTS.values()), 100)
        self.assertEqual(subtotal, 100)
        self.assertEqual(final, 92.5)
        self.assertEqual(points["hook"], 15)
        self.assertEqual(points["audience_fit"], 5)

    def test_score_clamps_at_zero(self):
        dimensions = {name: DimensionJudgment(0, "No evidence") for name in DIMENSION_WEIGHTS}
        _, final, _ = calculate_score(
            dimensions, (PenaltyJudgment("no_payoff", 25, "No conclusion."),)
        )
        self.assertEqual(final, 0)

    def test_boundary_can_trim_or_extend_only_on_contiguous_source(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript, padding_utterances=2)
        scored = apply_evaluation(evaluation_fixture(), context, transcript)
        self.assertEqual(scored.source_utterance_ids[0], "u000004")
        self.assertEqual(scored.source_utterance_ids[-1], "u000011")
        self.assertEqual(scored.transcript, " ".join(f"Exact source {i}." for i in range(4, 12)))
        self.assertEqual(scored.final_score, 100)

    def test_boundary_cannot_escape_allowed_context(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript, padding_utterances=1)
        with self.assertRaisesRegex(GoldMinerError, "escaped"):
            apply_evaluation(
                evaluation_fixture(start_utterance_id="u000001"), context, transcript
            )

    def test_hard_filter_sets_zero_and_records_reason(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        scored = apply_evaluation(
            evaluation_fixture(unsafe_extraction=True), context, transcript
        )
        self.assertTrue(scored.rejected)
        self.assertEqual(scored.final_score, 0)
        self.assertEqual(scored.rejection_reasons, ("unsafe_extraction",))

    def test_duration_outside_15_to_60_seconds_is_rejected(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        scored = apply_evaluation(
            evaluation_fixture(start_utterance_id="u000004", end_utterance_id="u000005"),
            context,
            transcript,
        )
        self.assertTrue(scored.rejected)
        self.assertEqual(scored.final_score, 0)
        self.assertIn("duration_out_of_bounds", scored.rejection_reasons)

    def test_clip_over_60_seconds_is_rejected_by_default(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        scored = apply_evaluation(
            evaluation_fixture(start_utterance_id="u000001", end_utterance_id="u000013"),
            build_boundary_context(candidate_fixture(), transcript, padding_utterances=20),
            transcript,
        )
        self.assertTrue(scored.rejected)
        self.assertIn("duration_out_of_bounds", scored.rejection_reasons)

    def test_phrase_boundaries_rescue_strong_arc_inside_coarse_utterances(self):
        utterances = (
            Utterance(
                "u000001",
                0,
                30_000,
                "A",
                "This setup is not useful for a cold viewer. We don't know how much AI should we use.",
                "normalized",
            ),
            Utterance(
                "u000002",
                31_000,
                70_000,
                "A",
                "Too much feels artificial and too little feels slow, but the right middle ground can be really helpful. This trailing sentence is unnecessary.",
                "normalized",
            ),
        )
        transcript = CanonicalTranscript("1.0", "provider", None, "hash", 70_000, utterances)
        candidate = Candidate(
            "c00001", 0, 70_000, ("u000001", "u000002"), ("framework",), "raw",
            "Use the right amount of AI.", "Evidence", ("w0001",),
        )
        scored = apply_evaluation(
            evaluation_fixture(
                start_utterance_id="u000001",
                end_utterance_id="u000002",
                start_boundary_text="We don't know how much AI should we use",
                end_boundary_text="the right middle ground can be really helpful",
            ),
            build_boundary_context(candidate, transcript),
            transcript,
        )
        self.assertFalse(scored.rejected)
        self.assertLess(scored.end_ms - scored.start_ms, 60_000)
        self.assertGreater(scored.start_ms, utterances[0].start_ms)
        self.assertLess(scored.end_ms, utterances[1].end_ms)
        self.assertTrue(scored.transcript.startswith("We don't know how much AI"))
        self.assertTrue(scored.transcript.endswith("really helpful."))

    def test_phrase_boundary_must_exist_in_named_utterance(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        with self.assertRaisesRegex(GoldMinerError, "could not be aligned"):
            apply_evaluation(
                evaluation_fixture(start_boundary_text="invented opening words"),
                context,
                transcript,
            )

    def test_end_phrase_can_cross_a_transcription_segment_boundary(self):
        utterances = (
            Utterance("u000001", 0, 10_000, "A", "Start here and keep going", "normalized"),
            Utterance(
                "u000002", 10_000, 20_000, "A", "until the complete payoff lands.", "normalized"
            ),
        )
        transcript = CanonicalTranscript("1.0", "provider", None, "hash", 20_000, utterances)
        candidate = Candidate(
            "c00001", 0, 20_000, ("u000001", "u000002"), ("lesson",), "raw",
            "Complete payoff.", "Evidence", ("w0001",),
        )
        scored = apply_evaluation(
            evaluation_fixture(
                start_utterance_id="u000001",
                end_utterance_id="u000002",
                start_boundary_text="Start here",
                end_boundary_text="keep going until the complete payoff lands",
                payoff_text="keep going until the complete payoff lands",
            ),
            build_boundary_context(candidate, transcript),
            transcript,
        )
        self.assertFalse(scored.rejected)
        self.assertEqual(scored.end_ms, 20_000)
        self.assertTrue(scored.transcript.endswith("complete payoff lands."))

    def test_missing_dimension_and_evidence_are_rejected(self):
        dimensions = {name: DimensionJudgment(8, "Evidence") for name in DIMENSION_WEIGHTS}
        dimensions.pop("hook")
        with self.assertRaisesRegex(GoldMinerError, "dimensions mismatch"):
            evaluation_fixture(dimensions=dimensions).validate()
        dimensions["hook"] = DimensionJudgment(8, "")
        with self.assertRaisesRegex(GoldMinerError, "requires evidence"):
            evaluation_fixture(dimensions=dimensions).validate()

    def test_unclear_opening_is_a_hard_rejection(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        scored = apply_evaluation(
            evaluation_fixture(opening_is_clear=False, opening_evidence="Depends on prior speech."),
            context,
            transcript,
        )
        self.assertTrue(scored.rejected)
        self.assertIn("unclear_opening", scored.rejection_reasons)

    def test_explicit_continuation_rejects_even_if_clear_flag_is_inconsistent(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        scored = apply_evaluation(
            evaluation_fixture(
                opening_is_clear=True,
                opening_problem="continuation",
                opening_evidence="The range begins mid-thought.",
            ),
            context,
            transcript,
        )
        self.assertTrue(scored.rejected)
        self.assertIn("unclear_opening", scored.rejection_reasons)

    def test_contextual_opening_words_are_rejected_deterministically(self):
        utterance = Utterance(
            "u000001", 0, 20_000, "A",
            "And it's complicated because using too much AI makes the work feel cheap.",
            "normalized",
        )
        transcript = CanonicalTranscript("1.0", "provider", None, "hash", 20_000, (utterance,))
        candidate = Candidate(
            "c00001", 0, 20_000, (utterance.id,), ("lesson",), utterance.original_text,
            "Too much AI can reduce quality.", "Evidence", ("w0001",),
        )
        scored = apply_evaluation(
            evaluation_fixture(start_utterance_id=utterance.id, end_utterance_id=utterance.id),
            build_boundary_context(candidate, transcript), transcript,
        )
        self.assertTrue(scored.rejected)
        self.assertIn("context_dependent_opening", scored.rejection_reasons)

    def test_cold_opening_gate_rejects_continuations_and_unresolved_pronouns(self):
        for text in (
            "And review it yourself because that is the safest workflow.",
            "Review it yourself because that is the safest workflow.",
            "It can be extremely helpful for developers who use the tool.",
            "What I see there before is a basic candidate interface that needs polish.",
        ):
            with self.subTest(text=text):
                utterance = Utterance(
                    "u000001", 0, 20_000, "A", text, "normalized"
                )
                transcript = CanonicalTranscript(
                    "1.0", "provider", None, "hash", 20_000, (utterance,)
                )
                candidate = Candidate(
                    "c00001", 0, 20_000, (utterance.id,), ("lesson",), text,
                    "A purported lesson.", "Evidence", ("w0001",),
                )
                scored = apply_evaluation(
                    evaluation_fixture(
                        start_utterance_id=utterance.id,
                        end_utterance_id=utterance.id,
                    ),
                    build_boundary_context(candidate, transcript),
                    transcript,
                )
                self.assertIn("context_dependent_opening", scored.rejection_reasons)

    def test_weak_central_idea_is_a_hard_rejection(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        scored = apply_evaluation(
            evaluation_fixture(central_idea_is_strong=False, central_idea_evidence="No complete payoff."),
            context,
            transcript,
        )
        self.assertTrue(scored.rejected)
        self.assertIn("weak_or_unclear_core_idea", scored.rejection_reasons)

    def test_explicit_garbled_idea_rejects_even_if_strong_flag_is_inconsistent(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        scored = apply_evaluation(
            evaluation_fixture(
                central_idea_is_strong=True,
                central_idea_problem="garbled",
                central_idea_evidence="The conclusion cannot be understood reliably.",
            ),
            context,
            transcript,
        )
        self.assertTrue(scored.rejected)
        self.assertIn("weak_or_unclear_core_idea", scored.rejection_reasons)

    def test_unfinished_punctuation_is_rejected_deterministically(self):
        utterance = Utterance(
            "u000001", 0, 20_000, "A",
            "A useful point starts here and develops clearly, but this ending stops,",
            "normalized",
        )
        transcript = CanonicalTranscript("1.0", "provider", None, "hash", 20_000, (utterance,))
        candidate = Candidate(
            "c00001", 0, 20_000, (utterance.id,), ("lesson",), utterance.original_text,
            "A point.", "Evidence", ("w0001",),
        )
        scored = apply_evaluation(
            evaluation_fixture(start_utterance_id=utterance.id, end_utterance_id=utterance.id),
            build_boundary_context(candidate, transcript), transcript,
        )
        self.assertTrue(scored.rejected)
        self.assertIn("incomplete_ending", scored.rejection_reasons)

    def test_vague_go_there_ending_is_rejected_deterministically(self):
        utterance = Utterance(
            "u000001", 0, 20_000, "A",
            "First define the requirements, ask AI to test everything, and then you go there.",
            "normalized",
        )
        transcript = CanonicalTranscript(
            "1.0", "provider", None, "hash", 20_000, (utterance,)
        )
        candidate = Candidate(
            "c00001", 0, 20_000, (utterance.id,), ("lesson",), utterance.original_text,
            "A workflow.", "Evidence", ("w0001",),
        )
        scored = apply_evaluation(
            evaluation_fixture(
                start_utterance_id=utterance.id,
                end_utterance_id=utterance.id,
            ),
            build_boundary_context(candidate, transcript),
            transcript,
        )
        self.assertIn("incomplete_ending", scored.rejection_reasons)

    def test_payoff_quote_must_be_inside_selected_arc(self):
        transcript = transcript_fixture()
        scored = apply_evaluation(
            evaluation_fixture(payoff_text="an invented payoff"),
            build_boundary_context(candidate_fixture(), transcript),
            transcript,
        )
        self.assertTrue(scored.rejected)
        self.assertIn("incomplete_narrative_arc", scored.rejection_reasons)

    def test_score_below_60_is_not_publishable(self):
        transcript = transcript_fixture()
        context = build_boundary_context(candidate_fixture(), transcript)
        dimensions = {name: DimensionJudgment(7, "Adequate but not exceptional evidence.") for name in DIMENSION_WEIGHTS}
        penalties = (PenaltyJudgment("slow_setup", 11, "The setup consumes too much of the clip."),)
        scored = apply_evaluation(
            evaluation_fixture(dimensions=dimensions, penalties=penalties), context, transcript
        )
        self.assertTrue(scored.rejected)
        self.assertEqual(scored.final_score, 0)
        self.assertIn("score_below_publishable_threshold", scored.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
