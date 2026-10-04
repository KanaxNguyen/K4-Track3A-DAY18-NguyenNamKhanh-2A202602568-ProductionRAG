from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
    except Exception:  # noqa: BLE001 - lexical tokenization has a plain-text fallback
        segmented = text
    # Preserve Vietnamese diacritics, normalize punctuation and compounds.
    return segmented.replace("_", " ").lower()


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(str(c.get("text", ""))).split() for c in self.documents]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens and any(self.corpus_tokens) else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        tokens = segment_vietnamese(query).split()
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        return [SearchResult(str(self.documents[i].get("text", "")), float(scores[i]),
                             dict(self.documents[i].get("metadata", {})), "bm25")
                for i in indices if float(scores[i]) > 0][:top_k]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
            self.backend = "qdrant_server"
        except Exception:  # noqa: BLE001 - local Qdrant is an intentional fallback
            self.client = QdrantClient(":memory:")
            self.backend = "qdrant_memory"
        self._encoder = None
        self._encoder_failed = False
        self._vector_size = EMBEDDING_DIM

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(EMBEDDING_MODEL, local_files_only=True)
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(model_path)
                self._vector_size = self._encoder.get_embedding_dimension()
            except Exception as exc:  # noqa: BLE001 - embedding/model errors use hashed vectors
                # Hashing vectors provide a local, dependency-light fallback when
                # model downloads are unavailable; BM25 remains the lexical path.
                print(f"  ⚠️  Embedding model unavailable; using hashed vectors ({exc})")
                from sklearn.feature_extraction.text import HashingVectorizer
                self._encoder = HashingVectorizer(n_features=EMBEDDING_DIM, alternate_sign=False, norm="l2")
                self._encoder_failed = True
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams
        if not chunks:
            try:
                self.client.delete_collection(collection)
            except Exception as exc:  # noqa: BLE001 - deleting an absent collection is harmless
                print(f"  ⚠️  Could not clear empty collection {collection}: {exc}")
            return
        encoder = self._get_encoder()
        texts = [str(c.get("text", "")) for c in chunks]
        if self._encoder_failed:
            vectors = encoder.transform(texts).toarray()
        else:
            vectors = encoder.encode(texts, show_progress_bar=len(texts) > 32, normalize_embeddings=True)
        size = int(vectors.shape[1])
        self.client.recreate_collection(collection_name=collection,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE))
        points = [PointStruct(id=i, vector=v.tolist(), payload={**c.get("metadata", {}), "text": texts[i]})
                  for i, (c, v) in enumerate(zip(chunks, vectors))]
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        try:
            encoder = self._get_encoder()
            if self._encoder_failed:
                query_vector = encoder.transform([query]).toarray()[0].tolist()
            else:
                query_vector = encoder.encode(query, normalize_embeddings=True).tolist()
            response = self.client.query_points(collection_name=collection, query=query_vector, limit=top_k)
            return [SearchResult(str(pt.payload.get("text", "")), float(pt.score),
                                 {k: v for k, v in pt.payload.items() if k != "text"}, "dense")
                    for pt in response.points if pt.payload]
        except Exception as exc:  # noqa: BLE001 - Qdrant errors should not break lexical search
            print(f"  ⚠️  Dense search unavailable for {collection}: {exc}")
            return []


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0 or top_k <= 0:
        return []
    fused = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in fused:
                fused[key] = {"score": 0.0, "result": result}
            fused[key]["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [SearchResult(item["result"].text, item["score"], dict(item["result"].metadata), "hybrid")
            for item in ranked]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
