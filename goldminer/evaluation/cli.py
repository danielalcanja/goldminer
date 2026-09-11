from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from goldminer.errors import GoldMinerError
from goldminer.evaluation.metrics import evaluate_dataset
from goldminer.evaluation.models import load_dataset

HIGHER_IS_BETTER = ("recall_at_10", "precision_at_5", "ndcg_at_10", "run_reliability")
LOWER_IS_BETTER = ("boundary_error_seconds", "duplicate_rate")


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compare_metrics(current: dict[str, Any], baseline: dict[str, Any], tolerance: float) -> tuple[dict[str, Any], bool]:
    current_metrics = current["metrics"]
    baseline_metrics = baseline["metrics"]
    comparison: dict[str, Any] = {}
    regressed = False
    for name in HIGHER_IS_BETTER:
        delta = float(current_metrics[name]) - float(baseline_metrics[name])
        failed = delta < -tolerance
        comparison[name] = {"baseline": baseline_metrics[name], "current": current_metrics[name], "improvement": round(delta, 4), "regression": failed}
        regressed = regressed or failed
    for name in LOWER_IS_BETTER:
        current_value, baseline_value = current_metrics[name], baseline_metrics[name]
        if current_value is None or baseline_value is None:
            failed = current_value is not None and baseline_value is None
            improvement = None
        else:
            improvement = round(float(baseline_value) - float(current_value), 4)
            failed = improvement < -tolerance
        comparison[name] = {"baseline": baseline_value, "current": current_value, "improvement": improvement, "regression": failed}
        regressed = regressed or failed
    return comparison, regressed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goldminer-evaluate",
        description="Measure Gold Miner rankings against human editorial labels.",
    )
    parser.add_argument("dataset", type=Path, help="versioned evaluation dataset JSON")
    parser.add_argument("--output", type=Path, help="evaluation report JSON (default: evaluation_report.json beside dataset)")
    parser.add_argument("--baseline", type=Path, help="prior evaluation report to compare")
    parser.add_argument("--max-regression", type=float, default=0.0, help="allowed metric regression before failure (default: 0)")
    parser.add_argument("--fail-on-regression", action="store_true", help="exit with status 1 when a metric exceeds allowed regression")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_regression < 0:
        print("goldminer-evaluate: error: --max-regression must be nonnegative", file=sys.stderr)
        return 2
    try:
        dataset = load_dataset(args.dataset.expanduser().resolve())
        report = evaluate_dataset(dataset)
        regressed = False
        if args.baseline:
            try:
                baseline = json.loads(args.baseline.expanduser().resolve().read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise GoldMinerError(f"Could not load baseline report: {exc}") from exc
            comparison, regressed = compare_metrics(report, baseline, args.max_regression)
            report["baseline_path"] = str(args.baseline.expanduser().resolve())
            report["comparison"] = comparison
            report["regression_detected"] = regressed
        output = args.output.expanduser().resolve() if args.output else args.dataset.expanduser().resolve().with_name("evaluation_report.json")
        _atomic_write(output, report)
    except GoldMinerError as exc:
        print(f"goldminer-evaluate: error: {exc}", file=sys.stderr)
        return 2
    metrics = report["metrics"]
    print(f"Evaluation complete: {report['completed_episode_count']}/{report['episode_count']} episodes")
    print(f"Recall@10: {metrics['recall_at_10']:.4f}")
    print(f"Precision@5: {metrics['precision_at_5']:.4f}")
    print(f"NDCG@10: {metrics['ndcg_at_10']:.4f}")
    print(f"Boundary error: {metrics['boundary_error_seconds']} seconds")
    print(f"Duplicate rate: {metrics['duplicate_rate']:.4f}")
    print(f"Run reliability: {metrics['run_reliability']:.4f}")
    print(f"Report: {output}")
    if regressed:
        print("Regression detected against baseline", file=sys.stderr)
        return 1 if args.fail_on_regression else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
