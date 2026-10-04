from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self.last_backend = "not_loaded"

    def _load_model(self):
        if self._model is None:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(self.model_name, local_files_only=True)
            from sentence_transformers import CrossEncoder
            # Keep the model on CPU for predictable memory use on student laptops.
            self._model = CrossEncoder(model_path, device="cpu")
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        pairs = [(query, str(doc.get("text", ""))) for doc in documents]
        try:
            scores = self._load_model().predict(pairs, show_progress_bar=False)
            self.last_backend = "cross_encoder"
            if isinstance(scores, (int, float)):
                scores = [scores]
        except Exception as exc:  # noqa: BLE001 - missing model uses deterministic local fallback
            # A lexical fallback keeps the pipeline operable before model weights
            # have been downloaded. The score is explicitly only a fallback rank.
            print(f"  ⚠️  Cross-encoder unavailable; using token overlap ({exc})")
            self.last_backend = "token_overlap"
            qterms = set(_terms(query))
            scores = []
            for _, text in pairs:
                terms = set(_terms(text))
                scores.append(len(qterms & terms) / max(1, len(qterms | terms)))
        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [RerankResult(str(doc.get("text", "")), float(doc.get("score", 0.0)), float(score),
                             dict(doc.get("metadata", {})), i)
                for i, (score, doc) in enumerate(scored[:top_k])]


def _terms(text: str) -> list[str]:
    import re
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            if self._model is None:
                from flashrank import Ranker
                self._model = Ranker()
            from flashrank import RerankRequest
            passages = [{"id": i, "text": str(doc.get("text", "")), "meta": doc.get("metadata", {})}
                        for i, doc in enumerate(documents)]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            by_id = {i: doc for i, doc in enumerate(documents)}
            return [RerankResult(item["text"], float(by_id[int(item["id"])].get("score", 0.0)),
                                 float(item["score"]), dict(item.get("meta", {})), rank)
                    for rank, item in enumerate(results[:top_k])]
        except Exception as exc:  # noqa: BLE001 - keep the secondary reranker optional
            print(f"  ⚠️  FlashRank unavailable; using cross-encoder fallback ({exc})")
            return CrossEncoderReranker().rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
