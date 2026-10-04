"""Regression checks for OpenRouter routing and enrichment failures."""

import config
import src.m5_enrichment as enrichment


def test_openrouter_client_routes_explicitly(monkeypatch):
    import openai

    received = {}

    def client(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(openai, "OpenAI", client)
    monkeypatch.setattr(config, "LLM_API_KEY", "synthetic-key")
    config.create_llm_client()
    assert received["api_key"] == "synthetic-key"
    assert received["base_url"] == "https://openrouter.ai/api/v1"


def test_overlong_llm_summary_uses_extractive_fallback(monkeypatch):
    text = "Chính sách nghỉ phép áp dụng cho nhân viên chính thức. " * 5
    monkeypatch.setattr(enrichment, "_chat_json", lambda *args: {"summary": text * 3})
    summary = enrichment.summarize_chunk(text)
    assert len(summary) <= len(text)
    assert summary == enrichment._local_summary(text)


def test_failed_combined_enrichment_attempts_one_request(monkeypatch):
    calls = []

    def failed_request(*args):
        calls.append(args)
        return {}

    monkeypatch.setattr(enrichment, "_chat_json", failed_request)
    result = enrichment._enrich_single_call("Nhân viên được nghỉ phép năm 15 ngày.", "policy.md")
    assert len(calls) == 1
    assert result["summary"]
    assert result["questions"]
    assert "policy.md" in result["context"]
    assert result["metadata"]["language"] == "vi"


def test_ragas_missing_score_is_not_reported_completed(monkeypatch):
    from types import SimpleNamespace

    import langchain_openai
    import pandas as pd
    import ragas

    from src.m4_eval import evaluate_ragas

    monkeypatch.setattr(config, "LLM_API_KEY", "synthetic-key")
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kwargs: object())
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", lambda **kwargs: object())
    frame = pd.DataFrame([{"faithfulness": 0.9, "answer_relevancy": 0.8,
                           "context_precision": 0.7, "context_recall": float("nan")}])
    monkeypatch.setattr(ragas, "evaluate", lambda *args, **kwargs: SimpleNamespace(to_pandas=lambda: frame))
    result = evaluate_ragas(["q"], ["a"], [["context"]], ["truth"])
    assert result["evaluation_status"] == "partial"
    assert result["invalid_metric_counts"]["context_recall"] == 1
    assert result["per_question"][0].context_recall is None
