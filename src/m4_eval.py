from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from config import LLM_API_KEY, LLM_BASE_URL, LLM_EMBEDDING_MODEL, LLM_MODEL
    keys = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    empty = {key: 0.0 for key in keys}
    empty["per_question"] = []
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
    if not questions:
        return empty
    if not LLM_API_KEY:
        print("  ⚠️  Skipping RAGAS: set OPENROUTER_API_KEY in .env")
        per_question = [EvalResult(q, a, c, gt, 0.0, 0.0, 0.0, 0.0)
                        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)]
        return {**{key: 0.0 for key in keys}, "per_question": per_question,
                "evaluation_status": "skipped_missing_api_key"}
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig

        llm_kwargs = {"model": LLM_MODEL, "api_key": LLM_API_KEY, "temperature": 0,
                      "timeout": 60, "max_retries": 2, "max_tokens": 2048}
        embedding_kwargs = {"model": LLM_EMBEDDING_MODEL, "api_key": LLM_API_KEY,
                            "check_embedding_ctx_length": False, "request_timeout": 60, "max_retries": 2}
        if LLM_BASE_URL:
            llm_kwargs["base_url"] = LLM_BASE_URL
            embedding_kwargs["base_url"] = LLM_BASE_URL
        evaluator_llm = ChatOpenAI(**llm_kwargs)
        evaluator_embeddings = OpenAIEmbeddings(**embedding_kwargs)
        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                          llm=evaluator_llm, embeddings=evaluator_embeddings,
                          run_config=RunConfig(timeout=120, max_retries=2, max_workers=1))
        df = result.to_pandas()
        per_question = []
        invalid_counts = dict.fromkeys(keys, 0)
        for i, row in df.iterrows():
            values = {}
            for key in keys:
                try:
                    score = float(row.get(key, float("nan")))
                except (TypeError, ValueError):
                    score = float("nan")
                if math.isfinite(score):
                    values[key] = score
                else:
                    values[key] = None
                    invalid_counts[key] += 1
            per_question.append(EvalResult(
                question=questions[int(i)], answer=answers[int(i)], contexts=contexts[int(i)],
                ground_truth=ground_truths[int(i)], **values))
        aggregate = {}
        for key in keys:
            valid = [getattr(item, key) for item in per_question if getattr(item, key) is not None]
            aggregate[key] = sum(valid) / len(valid) if valid else 0.0
        complete = len(per_question) == len(questions) and not any(invalid_counts.values())
        return {**aggregate, "per_question": per_question,
                "evaluation_status": "completed" if complete else "partial",
                "invalid_metric_counts": invalid_counts}
    except Exception as exc:  # noqa: BLE001 - optional API/dependency errors return an explicit skipped report
        print(f"  ⚠️  RAGAS evaluation failed: {type(exc).__name__}, HTTP {getattr(exc, 'status_code', None)}")
        empty["per_question"] = [EvalResult(q, a, c, gt, 0.0, 0.0, 0.0, 0.0)
                                 for q, a, c, gt in zip(questions, answers, contexts, ground_truths)]
        empty["evaluation_status"] = "failed"
        return empty


def _safe_float(value) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 5) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if bottom_n <= 0:
        return []
    diagnostic_tree = {
        "faithfulness": ("Answer may contain claims unsupported by retrieved context",
                         "Constrain generation to cited context and remove unsupported details"),
        "context_recall": ("Retrieved context is missing relevant facts",
                           "Improve chunk boundaries, query coverage, and hybrid retrieval"),
        "context_precision": ("Retrieved context contains distracting or irrelevant chunks",
                              "Rerank candidates and apply source or metadata filters"),
        "answer_relevancy": ("Answer does not directly address the user question",
                             "Tighten the answer prompt and preserve question intent"),
    }
    def value(item, key):
        return item.get(key, 0.0) if isinstance(item, dict) else getattr(item, key, 0.0)

    records = []
    for item in eval_results:
        values = {key: _safe_float(value(item, key)) for key in diagnostic_tree}
        worst = min(values, key=values.get)
        diagnosis, fix = diagnostic_tree[worst]
        avg = sum(values.values()) / len(values)
        records.append({"question": value(item, "question") or "", "answer": value(item, "answer") or "",
                        "contexts": value(item, "contexts") or [], "ground_truth": value(item, "ground_truth") or "",
                        "average_score": avg, "worst_metric": worst, "score": values[worst],
                        "diagnosis": diagnosis, "suggested_fix": fix,
                        "diagnostic_tree": {"output_correct": "Review answer against ground truth",
                                            "context_relevant": "Review contexts for supporting evidence",
                                            "retrieval_issue": worst in {"context_recall", "context_precision"},
                                            "next_action": fix}})
    return sorted(records, key=lambda record: (record["average_score"], record["score"]))[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k not in {"per_question", "run_metadata"}},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [asdict(item) if isinstance(item, EvalResult) else item
                         for item in results.get("per_question", [])],
        "run_metadata": results.get("run_metadata", {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
