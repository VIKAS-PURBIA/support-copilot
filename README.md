# Support Copilot (free, local RAG + agents)

Answers support questions from a help-center knowledge base and past tickets, **cites sources**,
says **"I don't know"** when unsure, and **escalates to a human** with a handoff note.

```
question -> router -> rewriter -> hybrid retriever -> grader -> answerer -> verifier -> answer + citations
               |                        |                |           |          |
               +---- wants a human -----+-- weak match --+-- no answer / unsupported --> escalation ticket
```

Stack: Ollama (LLM + embeddings), Qdrant (local mode), BM25, FastAPI, Streamlit. Cost: $0.

## Quick start
1. Install Python 3.10+ and Ollama (https://ollama.com/download)
2. `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`
3. `python -m venv .venv` then activate it, then `pip install -r requirements.txt`
4. `pytest -q`                        (no Ollama needed)
5. `python ingest.py`                 (builds the index from data/kb and data/tickets.csv)
6. `uvicorn app:app --port 8000`      (terminal 1)
7. `streamlit run ui.py`              (terminal 2)
8. Stop the API, then `python eval/run_eval.py --retrieval-only`, then `python eval/run_eval.py`

## Use your own data
- Put `.md` files in `data/kb/` (one article per file; `#`/`##` headings are used for chunking).
- Replace `data/tickets.csv` (columns: ticket_id, subject, question, resolution).
- Rewrite `data/eval.jsonl` with 50+ real questions, including unanswerable ones (`"expected_source": null`).
- Re-run `python ingest.py`.

## Design decisions worth explaining in interviews
- Hybrid search (BM25 + vectors, Reciprocal Rank Fusion). `eval/run_eval.py` measures all three modes.
- Access control is a metadata filter inside the vector query: customers only see `public`, agents also see `internal` tickets.
- Four independent refusal gates: retrieval score, grader, citation check, verifier. Any failure escalates.
- Retrieved text is treated as data in every prompt (prompt-injection defence); the router ignores instructions in user text.
- Every request returns a per-agent trace (latency + decisions).

## Known limits
- Local Qdrant allows one process: stop the API before `ingest.py` or the eval script.
- `role` is sent by the client for demo purposes. In production derive it from the auth token.
- `/escalations` has no auth. Add an API key before exposing it.
- `MIN_VECTOR_SCORE=0.5` is a starting guess: tune it with the eval output.
- Small local models (3B) make mistakes in JSON-mode agents; try `qwen2.5:7b` if you have 16 GB RAM
  (`CHAT_MODEL=qwen2.5:7b`).
