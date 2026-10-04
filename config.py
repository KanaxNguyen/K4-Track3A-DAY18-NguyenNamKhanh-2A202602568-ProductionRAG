"""Shared configuration for Lab 18."""

import os

from dotenv import load_dotenv

load_dotenv()

# --- LLM provider ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
if OPENROUTER_API_KEY in {"your_openrouter_api_key", "<your-openrouter-api-key>"}:
    OPENROUTER_API_KEY = ""

LLM_API_KEY = OPENROUTER_API_KEY
LLM_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini").strip()
LLM_EMBEDDING_MODEL = os.getenv("LLM_EMBEDDING_MODEL", "openai/text-embedding-3-small").strip()
ANSWER_SYSTEM_PROMPT = (
    "Trả lời trực tiếp, ngắn gọn bằng tiếng Việt, CHỈ dựa trên context. "
    "Ưu tiên chính sách ghi rõ đang có hiệu lực; tài liệu đã bị thay thế chỉ dùng để đối chiếu. "
    "Nếu câu hỏi cần tính toán, nêu phép tính từ các số liệu trong context. "
    "Nếu thiếu dữ kiện hoặc có mâu thuẫn chưa xác định được hiệu lực, nói rõ thay vì đoán."
)


def create_llm_client():
    """Create an OpenAI-compatible client routed through OpenRouter."""
    from openai import OpenAI

    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=60, max_retries=2)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
