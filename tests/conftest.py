"""Keep grading unit tests independent of paid API calls and account state.

Live OpenRouter and RAGAS are exercised by the full benchmark command.
"""

import pytest

import config
import src.m5_enrichment as enrichment


@pytest.fixture(autouse=True)
def disable_live_api(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    monkeypatch.setattr(enrichment, "LLM_API_KEY", "")
