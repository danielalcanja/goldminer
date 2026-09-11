import json
import unittest

from goldminer.errors import GoldMinerError
from goldminer.providers.embeddings import OpenAIEmbeddingProvider
from goldminer.providers.reranking import OpenAIRerankingProvider
from test_ranking import candidate


class EmbeddingProviderTests(unittest.TestCase):
    def test_parses_vectors_in_index_order(self):
        payload = json.dumps({"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}).encode()
        provider = OpenAIEmbeddingProvider("key", transport=lambda *_: (200, payload))
        self.assertEqual(provider.embed(("a", "b")), ((1.0, 0.0), (0.0, 1.0)))

    def test_rejects_wrong_vector_count(self):
        payload = json.dumps({"data": [{"index": 0, "embedding": [1, 0]}]}).encode()
        provider = OpenAIEmbeddingProvider("key", transport=lambda *_: (200, payload))
        with self.assertRaises(GoldMinerError):
            provider.embed(("a", "b"))

    def test_reranking_normalizes_duplicate_rank_numbers(self):
        payload = json.dumps({"output_text": json.dumps({"ranking": [
            {"candidate_id": "a", "rank": 1, "reason": "Strongest"},
            {"candidate_id": "b", "rank": 1, "reason": "Second"},
        ]})}).encode()
        provider = OpenAIRerankingProvider("key", transport=lambda *_: (200, payload))
        result = provider.rerank((candidate("a", 0, 90), candidate("b", 60000, 80)))
        self.assertEqual([item.rank for item in result], [1, 2])


if __name__ == "__main__":
    unittest.main()
