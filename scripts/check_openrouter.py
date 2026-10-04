"""Check OpenRouter auth, chat and embeddings without logging credentials."""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_EMBEDDING_MODEL,
    LLM_MODEL,
    create_llm_client,
)


def main() -> bool:
    report = {"checked_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
              "provider": "openrouter", "authenticated": False, "chat_ok": False, "embeddings_ok": False,
              "chat_model": LLM_MODEL, "embedding_model": LLM_EMBEDDING_MODEL}
    try:
        if not LLM_API_KEY:
            raise ValueError("Missing OPENROUTER_API_KEY")
        response = httpx.get(LLM_BASE_URL + "/key", headers={"Authorization": "Bearer " + LLM_API_KEY}, timeout=20)
        report["auth_http_status"] = response.status_code
        response.raise_for_status()
        report["authenticated"] = True
        client = create_llm_client().with_options(timeout=30, max_retries=0)
        chat = client.chat.completions.create(model=LLM_MODEL, temperature=0, max_tokens=5,
                                              messages=[{"role": "user", "content": "Reply only: OK"}])
        report["chat_ok"] = bool(chat.choices[0].message.content)
        embedding = client.embeddings.create(model=LLM_EMBEDDING_MODEL, input=["API verification"],
                                              encoding_format="float")
        report["embedding_dimensions"] = len(embedding.data[0].embedding)
        report["embeddings_ok"] = report["embedding_dimensions"] > 0
    except Exception as exc:  # noqa: BLE001 - report API failures without exposing response bodies or keys
        report["error_type"] = type(exc).__name__
        report["error_http_status"] = getattr(exc, "status_code", None)
    path = Path(__file__).resolve().parents[1] / "reports" / "openrouter_check.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return all(report[k] for k in ("authenticated", "chat_ok", "embeddings_ok"))


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
