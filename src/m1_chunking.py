from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from itertools import pairwise

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return []
    # Semantic embeddings are preferred; the deterministic fallback keeps this
    # strategy usable on machines that have not downloaded the model yet.
    try:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        import numpy as np
        from sentence_transformers import SentenceTransformer
        embeddings = SentenceTransformer(model_path).encode(sentences, normalize_embeddings=True)
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1) if len(sentences) > 1 else []
    except Exception as exc:  # noqa: BLE001 - model/cache failures use a local fallback
        print(f"  ⚠️  Semantic model unavailable; using token similarity ({exc})")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectors = TfidfVectorizer(ngram_range=(1, 2), analyzer="word").fit_transform(sentences)
            similarities = (vectors[:-1].multiply(vectors[1:])).sum(axis=1).A1
        except Exception:  # noqa: BLE001 - sklearn fallback is itself optional
            token_sets = [set(re.findall(r"\w+", s.lower())) for s in sentences]
            similarities = [len(a & b) / len(a | b) if a | b else 0.0
                            for a, b in pairwise(token_sets)]
    groups, current = [], [sentences[0]]
    for i, sentence in enumerate(sentences[1:]):
        if float(similarities[i]) < threshold:
            groups.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    groups.append(" ".join(current))
    return [Chunk(g, {**metadata, "chunk_index": i, "strategy": "semantic"}) for i, g in enumerate(groups)]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Split oversized paragraphs at sentence/word boundaries before grouping.
    pieces = []
    for paragraph in paragraphs:
        if len(paragraph) <= parent_size:
            pieces.append(paragraph)
            continue
        units = re.split(r"(?<=[.!?])\s+", paragraph)
        buf = ""
        for unit in units:
            while len(unit) > parent_size:
                if buf:
                    pieces.append(buf); buf = ""
                pieces.append(unit[:parent_size]); unit = unit[parent_size:]
            if buf and len(buf) + len(unit) + 1 > parent_size:
                pieces.append(buf); buf = unit
            else:
                buf = f"{buf} {unit}".strip()
        if buf:
            pieces.append(buf)
    parent_texts, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > parent_size:
            parent_texts.append(current); current = ""
        current = f"{current}\n\n{piece}".strip()
    if current:
        parent_texts.append(current)
    parents, children = [], []
    for i, parent_text in enumerate(parent_texts):
        pid = f"parent_{i}"
        parents.append(Chunk(parent_text, {**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": i}))
        # Use overlapping windows so boundaries do not lose sentence fragments.
        start = 0
        while start < len(parent_text):
            end = min(start + child_size, len(parent_text))
            if end < len(parent_text):
                boundary = parent_text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            child_text = parent_text[start:end].strip()
            if child_text:
                children.append(Chunk(child_text, {**metadata, "chunk_type": "child", "chunk_index": len(children)}, pid))
            if end >= len(parent_text):
                break
            start = max(end - min(40, child_size // 5), start + 1)
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    chunks = []
    current_header = ""
    current_lines = []
    def flush():
        body = "\n".join(current_lines).strip()
        content = "\n".join(x for x in (current_header, body) if x).strip()
        if content:
            chunks.append(Chunk(content, {**metadata, "section": current_header.lstrip("# ") or "preamble",
                                          "strategy": "structure", "chunk_index": len(chunks)}))
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            flush()
            current_header, current_lines = line.strip(), []
        else:
            current_lines.append(line)
    flush()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
