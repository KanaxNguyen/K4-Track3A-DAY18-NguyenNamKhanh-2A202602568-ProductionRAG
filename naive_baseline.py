"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NAIVE_COLLECTION
from src.m1_chunking import chunk_basic, load_documents
from src.m2_search import DenseSearch
from src.m4_eval import evaluate_ragas, load_test_set, save_report


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    from config import (
        ANSWER_SYSTEM_PROMPT,
        EMBEDDING_MODEL,
        LLM_API_KEY,
        LLM_MODEL,
        create_llm_client,
    )
    llm_client = None
    if LLM_API_KEY:
        llm_client = create_llm_client()
    generation_fallbacks = 0

    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        contexts = [r.text for r in results]

        if llm_client and contexts:
            try:
                context_str = "\n\n".join(contexts)
                resp = llm_client.chat.completions.create(model=LLM_MODEL, temperature=0, max_tokens=512, messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {item['question']}"},
                ])
                answer = resp.choices[0].message.content
                if not answer:
                    raise ValueError("Empty answer from provider")
            except Exception as exc:  # noqa: BLE001 - preserve retrieval when generation fails
                print(f"  ⚠️  Generation failed: {type(exc).__name__}", flush=True)
                generation_fallbacks += 1
                answer = contexts[0]
        else:
            generation_fallbacks += 1
            answer = contexts[0] if contexts else "Không tìm thấy."

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    from src.benchmarks import corpus_fingerprint

    results["run_metadata"] = {"pipeline": "naive", "documents": len(docs), "chunks": len(chunks),
                               "dense_backend": search.backend, "embedding_model": EMBEDDING_MODEL,
                               "embedding_fallback": search._encoder_failed,
                               "llm_provider": "openrouter", "llm_model": LLM_MODEL,
                               "generation_fallbacks": generation_fallbacks,
                               "input_fingerprint": corpus_fingerprint()}
    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {results.get(m, 0):.4f}")
    save_report(results, [], path="reports/naive_baseline_report.json")
    if all(results.get(m, 0) == 0 for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]):
        print("\n💡 Lưu ý: Điểm baseline hiển thị 0.00 là bình thường khi chưa hoàn thiện M2 (Dense Search) và M4 (Eval).")
        print("   Sau khi bạn implement xong các module, hãy chạy 'python main.py' để tự động cập nhật baseline thật và so sánh.")
    print("\nDone! Now implement advanced modules and run: python main.py")
    return results


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")
