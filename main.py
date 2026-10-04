"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import argparse
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def preflight():
    from huggingface_hub import snapshot_download
    from qdrant_client import QdrantClient

    from config import EMBEDDING_MODEL, LLM_API_KEY, QDRANT_HOST, QDRANT_PORT

    if not LLM_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY in the ignored local .env before live benchmarking")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
    try:
        client.get_collections()
    finally:
        client.close()
    for model in ("sentence-transformers/all-MiniLM-L6-v2", EMBEDDING_MODEL, "BAAI/bge-reranker-v2-m3"):
        snapshot_download(model, local_files_only=True)
    print("Preflight: OpenRouter configured, Qdrant reachable, all 3 model snapshots cached", flush=True)


def main(reuse_baseline: bool = False):
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)
    preflight()

    # Step 1: Basic Baseline
    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline
    if reuse_baseline:
        from config import EMBEDDING_MODEL, LLM_MODEL, TEST_SET_PATH
        from src.benchmarks import corpus_fingerprint

        with open("reports/naive_baseline_report.json", encoding="utf-8") as f:
            saved = json.load(f)
        with open(TEST_SET_PATH, encoding="utf-8") as f:
            expected = json.load(f)
        rows = saved.get("per_question", [])
        matching = len(rows) == len(expected) and all(
            row["question"] == item["question"] and row["ground_truth"] == item["ground_truth"]
            for row, item in zip(rows, expected))
        runtime = saved.get("run_metadata", {})
        if not (matching and saved["aggregate"].get("evaluation_status") == "completed"
                and runtime.get("llm_model") == LLM_MODEL and runtime.get("embedding_model") == EMBEDDING_MODEL
                and runtime.get("input_fingerprint") == corpus_fingerprint()):
            raise RuntimeError("Cannot reuse an incomplete baseline or a report with a different test set/model")
        naive_results = {**saved["aggregate"], "per_question": rows, "run_metadata": runtime}
        print("Reusing measured baseline; corpus must be unchanged.", flush=True)
    else:
        naive_results = run_baseline()

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline
    search, reranker = build_pipeline()
    prod_results = evaluate_pipeline(search, reranker)

    # Ensure reports are located in reports/
    for f in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(f):
            os.replace(f, f"reports/{f}")

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "✓" if p >= 0.75 else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")

    from src.benchmarks import benchmark_components, save_benchmark_summary

    print("\n📌 STEP 4: Chunking and warm reranker benchmarks", flush=True)
    benchmark_components(search, reranker)
    elapsed = time.time() - start
    complete = save_benchmark_summary(naive_results, prod_results, elapsed)
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Viết analysis/reflections/reflection_[HọTên].md")
    print("  3. Chạy: python check_lab.py")
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-baseline", action="store_true",
                        help="Reuse a completed baseline from the same unchanged corpus/test set/model")
    sys.exit(0 if main(parser.parse_args().reuse_baseline) else 1)
