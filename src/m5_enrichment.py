from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_API_KEY, LLM_MODEL, create_llm_client


def _has_api_key() -> bool:
    key = (LLM_API_KEY or "").strip()
    return bool(key and not key.startswith("#"))


def _local_summary(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return " ".join(sentences[:2]) or text.strip()


def _local_questions(text: str, n_questions: int = 3) -> list[str]:
    sentences = [s.strip(" .") for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 8]
    return [f"Thông tin nào được nêu về: {s[:100]}?" for s in sentences[:n_questions]]


def _local_context(text: str, source: str) -> str:
    topic = next((line.strip(" #") for line in text.splitlines() if line.strip()), "nội dung tài liệu")[:100]
    return f"Trích từ {source}, nội dung về {topic}." if source else f"Đoạn trích nói về {topic}."


def _local_metadata(text: str) -> dict:
    lower = text.lower()
    category = ("it" if any(w in lower for w in ("mật khẩu", "vpn", "máy tính", "phần mềm")) else
                "finance" if any(w in lower for w in ("lương", "thuế", "tài chính", "chi phí")) else
                "hr" if any(w in lower for w in ("nhân viên", "nghỉ phép", "đào tạo", "thử việc")) else "policy")
    words = re.findall(r"[A-ZÀ-Ỹ][\wÀ-ỹ-]+", text)
    return {"topic": text.strip().splitlines()[0][:120] if text.strip() else "general",
            "entities": list(dict.fromkeys(words[:8])), "category": category, "language": "vi"}


def _chat_json(system: str, user: str, max_tokens: int = 300) -> dict:
    if not _has_api_key():
        return {}
    try:
        response = create_llm_client().chat.completions.create(
            model=LLM_MODEL, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"}, max_tokens=max_tokens)
        result = json.loads(response.choices[0].message.content or "{}")
        return result if isinstance(result, dict) else {}
    except Exception as exc:  # noqa: BLE001 - enrichment has local fallbacks
        print(f"  ⚠️  Enrichment API unavailable; using local fallback ({type(exc).__name__}, HTTP {getattr(exc, 'status_code', None)})")
        return {}


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if len(text.strip()) <= 160:
        return _local_summary(text)
    result = _chat_json("Tóm tắt ngắn gọn bằng tiếng Việt, tối đa 2 câu, không dài hơn đoạn gốc và không thêm thông tin. Trả JSON {\"summary\": \"...\"}.", text, 150)
    if result.get("summary"):
        summary = str(result["summary"]).strip()
        if len(summary) <= len(text):
            return summary
    return _local_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    result = _chat_json(f"Tạo tối đa {n_questions} câu hỏi trả lời được từ đoạn văn. Trả JSON {{\"questions\": [..]}}.", text, 200)
    questions = result.get("questions", [])
    if isinstance(questions, list) and questions:
        return [str(q).strip().lstrip("0123456789.-) ") for q in questions if str(q).strip()][:n_questions]
    return _local_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    result = _chat_json("Viết một câu tiếng Việt mô tả vị trí/chủ đề của đoạn trong tài liệu. Trả JSON {\"context\": \"...\"}.",
                        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", 100)
    context = str(result.get("context", "")).strip()
    if not context:
        context = _local_context(text, document_title)
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    result = _chat_json('Trích xuất metadata. Trả JSON {"topic":"...","entities":[],"category":"policy|hr|it|finance|general","language":"vi|en"}.', text, 150)
    if result:
        return result
    return _local_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    cache_key = hashlib.sha256(json.dumps(["combined-v2", LLM_MODEL, source, text], ensure_ascii=False).encode()).hexdigest()
    cache_path = Path(__file__).resolve().parents[1] / ".cache" / "enrichment" / f"{cache_key}.json"
    if _has_api_key() and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("_backend") == "openrouter":
                return {**cached, "_backend": "openrouter_cache"}
        except (OSError, json.JSONDecodeError):
            pass
    result = _chat_json('Phân tích đoạn và trả đúng JSON: {"summary":"...","questions":["..."],"context":"...","metadata":{"topic":"...","entities":[],"category":"policy|hr|it|finance|general","language":"vi|en"}}',
                        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", 400)
    if result:
        result["_backend"] = "openrouter"
        if _has_api_key():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return result
    # A failed combined request must not trigger four additional API calls.
    return {"summary": _local_summary(text), "questions": _local_questions(text),
            "context": _local_context(text, source), "metadata": _local_metadata(text), "_backend": "local"}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
            auto_meta = dict(auto_meta) if isinstance(auto_meta, dict) else {}
            auto_meta["enrichment_backend"] = result.get("_backend", "openrouter")
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
