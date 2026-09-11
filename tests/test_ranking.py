import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from goldminer.analysis.ranking import RankingConfig, build_ranking, deduplicate, write_ranking
from goldminer.analysis.scoring import DIMENSION_WEIGHTS, DimensionJudgment, ScoredCandidate
from goldminer.errors import GoldMinerError
from goldminer.providers.embeddings import FakeEmbeddingProvider
from goldminer.providers.reranking import FakeRerankingProvider


def candidate(identifier, start, score, category="lesson", rejected=False):
    dimensions = {name: DimensionJudgment(7, "Evidence") for name in DIMENSION_WEIGHTS}
    return ScoredCandidate(
        identifier, start, start + 30000, ("u001",), identifier, category, (),
        dimensions, (), 70, score, rejected, ("test",) if rejected else (),
        "Boundary reason", "Selection reason",
    )


class RankingTests(unittest.TestCase):
    def test_three_phrasings_collapse_and_unrelated_clips_remain(self):
        clips = (
            candidate("idea-a", 0, 78),
            candidate("idea-b", 60000, 82),
            candidate("idea-c", 120000, 75),
            candidate("unrelated", 180000, 80, "story"),
        )
        vectors = ((1, 0), (.99, .01), (.98, .02), (0, 1))
        kept, families = deduplicate(clips, vectors, RankingConfig())
        self.assertEqual({item.id for item in kept}, {"idea-b", "unrelated"})
        duplicate = next(item for item in families if len(item.member_ids) == 3)
        self.assertEqual(duplicate.representative_id, "idea-b")

    def test_timestamp_overlap_collapses_to_strongest(self):
        clips = (candidate("weak", 0, 60), candidate("strong", 5000, 90))
        kept, _ = deduplicate(clips, ((1, 0), (0, 1)), RankingConfig())
        self.assertEqual([item.id for item in kept], ["strong"])

    def test_final_ranking_is_diverse_auditable_and_cached(self):
        clips = (
            candidate("a", 0, 90),
            candidate("b", 60000, 88),
            candidate("c", 120000, 86),
            candidate("d", 180000, 84, "story"),
            candidate("rejected", 240000, 99, rejected=True),
        )
        embed = FakeEmbeddingProvider({
            f"{c.category}. {c.transcript}": tuple(1.0 if column == index else 0.0 for column in range(4))
            for index, c in enumerate(clips[:-1])
        })
        rerank = FakeRerankingProvider(("a", "b", "c", "d"))
        with TemporaryDirectory() as temporary:
            result, artifact = build_ranking(clips, embed, rerank, Path(temporary), 3)
            self.assertEqual([item.id for item in result.selected], ["a", "b", "d"])
            self.assertEqual(artifact["counts"]["selected"], 3)
            self.assertNotIn("rejected", {item["id"] for item in artifact["final_ranking"]})
            write_ranking(Path(temporary) / "ranking.json", artifact)
            self.assertEqual(json.loads((Path(temporary) / "ranking.json").read_text())["schema_version"], "1.0")
            cached, _ = build_ranking(clips, embed, rerank, Path(temporary), 3)
            self.assertTrue(cached.reused_embeddings)
            self.assertTrue(cached.reused_reranking)

    def test_invalid_reranking_is_rejected(self):
        clips = (candidate("a", 0, 90), candidate("b", 60000, 80))
        embed = FakeEmbeddingProvider({"lesson. a": (1, 0), "lesson. b": (0, 1)})
        with TemporaryDirectory() as temporary:
            with self.assertRaises(GoldMinerError):
                build_ranking(clips, embed, FakeRerankingProvider(("a",)), Path(temporary), 2)

    def test_all_rejected_candidates_produce_an_empty_successful_ranking(self):
        clips = (candidate("weak", 0, 0, rejected=True),)
        with TemporaryDirectory() as temporary:
            result, artifact = build_ranking(
                clips,
                FakeEmbeddingProvider({}),
                FakeRerankingProvider(()),
                Path(temporary),
                8,
            )
        self.assertEqual(result.selected, ())
        self.assertEqual(artifact["counts"]["eligible"], 0)
        self.assertEqual(artifact["final_ranking"], [])


if __name__ == "__main__":
    unittest.main()
