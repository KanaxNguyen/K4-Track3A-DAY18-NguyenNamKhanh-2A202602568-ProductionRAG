"""Reproducible component benchmarks and the final baseline comparison."""

import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

from config import DATA_DIR, EMBEDDING_MODEL, RERANK_TOP_K, TEST_SET_PATH
from src.m1_chunking import compare_strategies, load_documents
from src.m3_rerank import benchmark_reranker
from src.m4_eval import load_test_set

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def corpus_fingerprint() -> str:
    """Hash benchmark inputs so a baseline cannot be reused after data changes."""
    paths = sorted(Path(DATA_DIR).glob("*.md")) + sorted(Path(DATA_DIR).glob("*.pdf")) + [Path(TEST_SET_PATH)]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _save(name: str, data: dict):
    path = Path("reports") / name
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report saved to {path}", flush=True)


def benchmark_components(search, reranker):
    documents = load_documents()
    started = time.perf_counter()
    stats = compare_strategies(documents)
    _save("chunking_report.json", {"documents": len(documents),
                                   "scope": "joined corpus, compare_strategies()",
                                   "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
                                   "elapsed_ms": (time.perf_counter() - started) * 1000,
                                   "strategies": stats})
    manifest = [{"source": doc["metadata"]["source"], "text_characters": len(doc["text"]),
                 "text_sha256": hashlib.sha256(doc["text"].encode()).hexdigest()} for doc in documents]
    _save("input_manifest.json", {"indexed_documents": manifest,
                                  "skipped_scan_pdfs": ["BCTC.pdf",
                                                        "Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf"],
                                  "test_set_sha256": hashlib.sha256(Path("test_set.json").read_bytes()).hexdigest()})
    query = load_test_set()[0]["question"]
    candidates = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in search.search(query)]
    latency = benchmark_reranker(reranker, query, candidates, n_runs=5)
    _save("reranker_benchmark.json", {"model": reranker.model_name, "backend": reranker.last_backend,
                                      "candidates": len(candidates), "top_k": RERANK_TOP_K, "runs": 5,
                                      "model_load_included": False, "warm_latency": latency})


def save_benchmark_summary(naive: dict, production: dict, elapsed_s: float):
    measured_at = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    rows = {metric: {"naive": naive[metric], "production": production[metric],
                     "delta": production[metric] - naive[metric]} for metric in METRICS}
    complete = all(result.get("evaluation_status") == "completed" for result in (naive, production))
    _save("benchmark_summary.json", {"measured_at": measured_at, "elapsed_seconds": elapsed_s,
                                     "evaluation_complete": complete, "comparison": rows,
                                     "production_metrics_at_least_070": sum(production[k] >= 0.70 for k in METRICS),
                                     "naive_runtime": naive.get("run_metadata", {}),
                                     "production_runtime": production.get("run_metadata", {})})
    packages = ("sentence-transformers", "qdrant-client", "ragas", "langchain-openai", "openai", "pytest")
    _save("environment_report.json", {"measured_at": measured_at, "python": sys.version.split()[0],
                                      "platform": platform.platform(), "architecture": platform.machine(),
                                      "embedding_model": EMBEDDING_MODEL,
                                      "packages": {name: version(name) for name in packages}})
    return complete
