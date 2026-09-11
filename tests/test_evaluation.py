import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from goldminer.errors import GoldMinerError
from goldminer.evaluation.cli import compare_metrics, main
from goldminer.evaluation.metrics import evaluate_dataset, evaluate_episode
from goldminer.evaluation.models import GoldMoment, load_dataset


class EvaluationTests(unittest.TestCase):
    def test_metrics_cover_editorial_quality_dimensions(self):
        gold = (
            GoldMoment("g1", 1000, 11000, 3, "idea-a"),
            GoldMoment("g2", 20000, 30000, 2, "idea-b"),
        )
        predictions = (
            {"start_ms": 1500, "end_ms": 11500},
            {"start_ms": 2000, "end_ms": 10000},
            {"start_ms": 20500, "end_ms": 29500},
            {"start_ms": 40000, "end_ms": 50000},
        )
        result = evaluate_episode(predictions, gold)
        self.assertEqual(result["recall_at_10"], 1.0)
        self.assertEqual(result["precision_at_5"], 0.5)
        self.assertGreater(result["ndcg_at_10"], 0.9)
        self.assertEqual(result["boundary_error_seconds"], 0.5)
        self.assertEqual(result["duplicate_rate"], 0.25)

    def test_dataset_records_missing_ranking_as_reliability_failure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ranking.json").write_text(json.dumps({"final_ranking": [{"start_ms": 0, "end_ms": 10000}]}))
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps({
                "schema_version": "1.0", "name": "fixture",
                "episodes": [
                    {"id": "ok", "ranking_path": "ranking.json", "gold_moments": [{"id": "g1", "start_ms": 0, "end_ms": 10000}]},
                    {"id": "missing", "ranking_path": "missing.json", "gold_moments": [{"id": "g2", "start_ms": 0, "end_ms": 10000}]},
                ],
            }))
            report = evaluate_dataset(load_dataset(dataset_path))
            self.assertEqual(report["metrics"]["run_reliability"], 0.5)
            self.assertEqual(len(report["errors"]), 1)

    def test_invalid_dataset_is_rejected(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.json"
            path.write_text('{"schema_version":"9","name":"bad","episodes":[]}')
            with self.assertRaisesRegex(GoldMinerError, "Unsupported"):
                load_dataset(path)

    def test_baseline_comparison_obeys_metric_direction(self):
        baseline = {"metrics": {"recall_at_10": 0.8, "precision_at_5": 0.8, "ndcg_at_10": 0.8, "run_reliability": 1.0, "boundary_error_seconds": 2.0, "duplicate_rate": 0.1}}
        current = {"metrics": {"recall_at_10": 0.7, "precision_at_5": 0.8, "ndcg_at_10": 0.8, "run_reliability": 1.0, "boundary_error_seconds": 3.0, "duplicate_rate": 0.05}}
        comparison, regressed = compare_metrics(current, baseline, 0.0)
        self.assertTrue(regressed)
        self.assertTrue(comparison["recall_at_10"]["regression"])
        self.assertTrue(comparison["boundary_error_seconds"]["regression"])
        self.assertFalse(comparison["duplicate_rate"]["regression"])

    def test_cli_writes_report_without_network(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ranking.json").write_text(json.dumps({"final_ranking": [{"start_ms": 0, "end_ms": 10000}]}))
            (root / "dataset.json").write_text(json.dumps({
                "schema_version": "1.0", "name": "fixture",
                "episodes": [{"id": "ep", "ranking_path": "ranking.json", "gold_moments": [{"id": "g", "start_ms": 0, "end_ms": 10000}]}],
            }))
            output = root / "result.json"
            self.assertEqual(main([str(root / "dataset.json"), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["metrics"]["recall_at_10"], 1.0)


if __name__ == "__main__":
    unittest.main()
