"""Load KB markdown + past tickets, chunk, embed, and store in Qdrant.

Run:  python ingest.py        (stop the API first: local Qdrant allows one process)
Re-running rebuilds the index, so edited and deleted documents are picked up.
Embeddings are cached by text hash, so unchanged chunks are not re-embedded.
"""
import csv
import hashlib
import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import config as cfg
import llm


def _windows(words, size, overlap):
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        yield " ".join(words[start:start + size])
        if start + size >= len(words):
            break


def load_kb():
    chunks = []
    for path in sorted(cfg.KB_DIR.glob("*.md")):
        title, heading, body, sections = path.stem, "", [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("#"):
                if body:
                    sections.append((heading, body))
                heading, body = line.lstrip("# ").strip(), []
            else:
                body.append(line)
        sections.append((heading, body))
        for heading, lines in sections:
            label = f"{title} > {heading}" if heading else title
            for piece in _windows(" ".join(lines).split(), cfg.CHUNK_WORDS, cfg.CHUNK_OVERLAP):
                chunks.append({"text": f"{label}\n{piece}", "source": path.name, "title": title,
                               "type": "kb", "access": "public"})
    return chunks


def load_tickets():
    """Expected CSV columns: ticket_id, subject, question, resolution."""
    if not cfg.TICKETS_CSV.exists():
        return []
    chunks = []
    with open(cfg.TICKETS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = f"Subject: {row['subject']}\nCustomer: {row['question']}\nResolution: {row['resolution']}"
            chunks.append({"text": text, "source": f"ticket:{row['ticket_id']}", "title": row["subject"],
                           "type": "ticket", "access": "internal"})
    return chunks


def _embed_cached(texts):
    cache = json.loads(cfg.EMBED_CACHE.read_text()) if cfg.EMBED_CACHE.exists() else {}
    keys = [hashlib.sha1((cfg.EMBED_MODEL + t).encode()).hexdigest() for t in texts]
    missing = [(k, t) for k, t in zip(keys, texts) if k not in cache]
    for i in range(0, len(missing), 32):
        batch = missing[i:i + 32]
        vectors = llm.embed([cfg.EMBED_DOC_PREFIX + t for _, t in batch])
        for (k, _), v in zip(batch, vectors):
            cache[k] = v
    cfg.EMBED_CACHE.write_text(json.dumps(cache))
    return [cache[k] for k in keys]


def build() -> int:
    cfg.STORAGE.mkdir(parents=True, exist_ok=True)
    chunks = load_kb() + load_tickets()
    if not chunks:
        raise SystemExit(f"No documents found. Put .md files in {cfg.KB_DIR}")
    for i, c in enumerate(chunks):
        c["id"] = i
    vectors = _embed_cached([c["text"] for c in chunks])

    client = QdrantClient(path=cfg.QDRANT_PATH)
    if client.collection_exists(cfg.COLLECTION):
        client.delete_collection(cfg.COLLECTION)
    client.create_collection(cfg.COLLECTION,
                             vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
    client.upsert(cfg.COLLECTION,
                  points=[PointStruct(id=c["id"], vector=v, payload=c) for c, v in zip(chunks, vectors)])
    client.close()
    cfg.CHUNKS_JSON.write_text(json.dumps(chunks))
    return len(chunks)


if __name__ == "__main__":
    print(f"Indexed {build()} chunks")
