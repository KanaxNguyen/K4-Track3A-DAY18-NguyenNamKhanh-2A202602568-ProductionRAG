from __future__ import annotations

"""Production RAG Pipeline — Ghép toàn bộ M1+M2+M3+M4+M5."""

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ANSWER_SYSTEM_PROMPT, EMBEDDING_MODEL, LLM_MODEL, RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)
    from src.benchmarks import corpus_fingerprint

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        parent_texts = {parent.metadata["parent_id"]: parent.text for parent in parents}
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata,
                               "parent_id": child.parent_id,
                               "parent_text": parent_texts.get(child.parent_id, child.text)}})
    chunking_ms = (time.time() - t0) * 1000
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({chunking_ms/1000:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        enrichment_ms = (time.time() - t0) * 1000
        print(f"  ✓ Enriched {len(enriched)} chunks ({enrichment_ms/1000:.1f}s)", flush=True)
    else:
        enrichment_ms = (time.time() - t0) * 1000
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    search.index(all_chunks)
    indexing_ms = (time.time() - t0) * 1000
    print(f"  ✓ Indexed ({indexing_ms/1000:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker_ms = (time.time() - t0) * 1000
    print(f"  ✓ Reranker ready ({reranker_ms/1000:.1f}s; model loads on first query)", flush=True)

    search.pipeline_latency_ms = {"chunking": chunking_ms, "enrichment": enrichment_ms,
                                  "indexing": indexing_ms, "reranker_init": reranker_ms}
    search.run_metadata = {"pipeline": "production", "documents": len(docs), "chunks": len(all_chunks),
                           "dense_backend": search.dense.backend, "embedding_model": EMBEDDING_MODEL,
                           "embedding_fallback": search.dense._encoder_failed,
                           "enrichment_mode": "combined", "enrichment_chunks": len(enriched),
                           "enrichment_fallbacks": sum(e.auto_metadata.get("enrichment_backend") == "local"
                                                       for e in enriched),
                           "enrichment_cached_chunks": sum(e.auto_metadata.get("enrichment_backend") == "openrouter_cache"
                                                           for e in enriched),
                           "llm_provider": "openrouter", "llm_model": LLM_MODEL}
    search.run_metadata["input_fingerprint"] = corpus_fingerprint()
    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    results = search.search(query)
    retrieval_ms = (time.perf_counter() - stage_start) * 1000
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    stage_start = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    rerank_ms = (time.perf_counter() - stage_start) * 1000
    contexts = [r.metadata.get("parent_text", r.text) for r in reranked] if reranked else [
        r.metadata.get("parent_text", r.text) for r in results[:3]]
    contexts = list(dict.fromkeys(contexts))

    from config import LLM_API_KEY, LLM_MODEL, create_llm_client
    generation_start = time.perf_counter()
    reranker.last_generation_backend = "context_fallback"
    if LLM_API_KEY and contexts:
        try:
            client = create_llm_client()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model=LLM_MODEL, temperature=0, max_tokens=512, messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
            if not answer:
                raise ValueError("Empty answer from provider")
            reranker.last_generation_backend = "openrouter"
        except Exception as e:  # noqa: BLE001 - generation fallback keeps retrieval usable
            print(f"  ⚠️  LLM generation failed: {type(e).__name__}, HTTP {getattr(e, 'status_code', None)}", flush=True)
            answer = contexts[0]
    else:
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    generation_ms = (time.perf_counter() - generation_start) * 1000
    reranker.last_latency_ms = {"retrieval": retrieval_ms, "reranking": rerank_ms,
                                "generation": generation_ms,
                                "total_query": (time.perf_counter() - total_start) * 1000}
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []
    query_timings = []
    generation_fallbacks = 0
    reranking_fallbacks = 0

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        query_timings.append(dict(reranker.last_latency_ms))
        generation_fallbacks += reranker.last_generation_backend != "openrouter"
        reranking_fallbacks += reranker.last_backend != "cross_encoder"
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    results["run_metadata"] = {**search.run_metadata, "reranker_model": reranker.model_name,
                               "generation_fallbacks": generation_fallbacks,
                               "reranking_fallbacks": reranking_fallbacks}
    print(f"  ✓ RAGAS done ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    if results.get("evaluation_status", "completed") == "completed":
        failures = failure_analysis(results.get("per_question", []))
    else:
        failures = []
        print("  ⚠️  Failure ranking requires successful RAGAS scores; see manual notes in analysis/.")
    save_report(results, failures)
    stage_names = ("retrieval", "reranking", "generation", "total_query")
    latency_report = {"setup_ms": getattr(search, "pipeline_latency_ms", {}), "queries": len(query_timings),
                      "per_query_ms": query_timings,
                      "stages": {name: {"avg_ms": round(sum(row[name] for row in query_timings) /
                                                             max(1, len(query_timings)), 2),
                                         "min_ms": round(min((row[name] for row in query_timings), default=0), 2),
                                         "max_ms": round(max((row[name] for row in query_timings), default=0), 2)}
                                 for name in stage_names}}
    os.makedirs("reports", exist_ok=True)
    with open("reports/latency_report.json", "w", encoding="utf-8") as report_file:
        json.dump(latency_report, report_file, ensure_ascii=False, indent=2)
    print("Latency report saved to reports/latency_report.json")
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
