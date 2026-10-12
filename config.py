"""Central settings. Override any of them with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).parent

# --- models (served by Ollama) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
# nomic-embed-text expects these prefixes. If you switch embedding models, change/clear them.
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "search_document: ")
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query: ")

# --- paths ---
KB_DIR = ROOT / "data" / "kb"
TICKETS_CSV = ROOT / "data" / "tickets.csv"
STORAGE = ROOT / "storage"
QDRANT_PATH = str(STORAGE / "qdrant")
CHUNKS_JSON = STORAGE / "chunks.json"
EMBED_CACHE = STORAGE / "embed_cache.json"
ESCALATIONS_FILE = STORAGE / "escalations.jsonl"
COLLECTION = "support"

# --- retrieval ---
CHUNK_WORDS = 150
CHUNK_OVERLAP = 25
TOP_K = 5
CANDIDATES = 20
# Refuse when the best vector similarity is below this. A starting guess: tune it with
# `python eval/run_eval.py --retrieval-only`, which prints the score gap between
# answerable and unanswerable questions.
MIN_VECTOR_SCORE = float(os.getenv("MIN_VECTOR_SCORE", "0.63"))
USE_RERANKER = os.getenv("USE_RERANKER", "0") == "1"   # needs: pip install sentence-transformers

# --- agents ---
VERIFY_ANSWERS = os.getenv("VERIFY_ANSWERS", "1") == "1"

# The grader is unreliable on 3B models (it rejects good answers). Turn on with a 7B+ model.
USE_GRADER = os.getenv("USE_GRADER", "0") == "1"

# --- access control (in production, derive the role from the login token, never from the client) ---
ROLE_ACCESS = {"customer": ["public"], "agent": ["public", "internal"]}
