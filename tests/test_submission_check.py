"""Prevent the submission checker from accepting failed tests or default scores."""

import json
from types import SimpleNamespace

import check_lab
import config
import main


def test_collection_errors_fail_test_check(monkeypatch):
    monkeypatch.setattr(check_lab.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="1 error in 0.1s", stderr="", returncode=2))
    assert check_lab.run_tests() == (0, 1, False)


def test_incomplete_ragas_report_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test_set.json").write_text(json.dumps([{"question": "q"}]))
    metrics = {k: 0.9 for k in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    report = {"aggregate": {**metrics, "evaluation_status": "partial"},
              "num_questions": 1, "per_question": [metrics]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert not check_lab.check_eval_report(str(path))
    report["aggregate"]["evaluation_status"] = "completed"
    path.write_text(json.dumps(report))
    assert check_lab.check_eval_report(str(path))
    report["per_question"][0] = {**metrics, "context_recall": None}
    path.write_text(json.dumps(report))
    assert not check_lab.check_eval_report(str(path))


def test_preflight_closes_qdrant_on_connection_error(monkeypatch):
    import pytest
    import qdrant_client

    closed = []

    def connection_error():
        raise RuntimeError("unreachable")

    client = SimpleNamespace(get_collections=connection_error, close=lambda: closed.append(True))
    monkeypatch.setattr(config, "LLM_API_KEY", "synthetic-key")
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kwargs: client)
    with pytest.raises(RuntimeError, match="unreachable"):
        main.preflight()
    assert closed == [True]
