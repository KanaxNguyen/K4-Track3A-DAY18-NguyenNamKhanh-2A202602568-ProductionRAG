"""Retry missing measurements using saved answers; preserve all valid scores."""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.m4_eval import evaluate_ragas, failure_analysis, save_report

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def repair(path: Path) -> bool:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["per_question"]
    retries = []
    for index, row in enumerate(rows):
        missing = [key for key in METRICS if not isinstance(row.get(key), (int, float))
                   or not math.isfinite(row[key])]
        if not missing:
            continue
        print(f"Retrying question {index + 1}: {', '.join(missing)}", flush=True)
        measured = evaluate_ragas([row["question"]], [row["answer"]], [row["contexts"]], [row["ground_truth"]])
        fresh = asdict(measured["per_question"][0])
        for key in missing:
            if measured["evaluation_status"] == "completed":
                row[key] = fresh[key]
        retries.append({"question_number": index + 1, "metrics": missing,
                        "retry_status": measured["evaluation_status"]})
    invalid = {key: sum(row.get(key) is None for row in rows) for key in METRICS}
    aggregates = {}
    for key in METRICS:
        valid = [row[key] for row in rows if row.get(key) is not None]
        aggregates[key] = sum(valid) / len(valid) if valid else 0.0
    complete = not any(invalid.values())
    results = {**aggregates, "per_question": rows, "evaluation_status": "completed" if complete else "partial",
               "invalid_metric_counts": invalid,
               "run_metadata": {**report.get("run_metadata", {}), "evaluation_retries": retries,
                                "ragas_max_tokens": 2048, "ragas_max_workers": 1}}
    failures = failure_analysis(rows, bottom_n=5) if complete and path.name == "ragas_report.json" else []
    save_report(results, failures, str(path))
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    sys.exit(0 if repair(parser.parse_args().report) else 1)
