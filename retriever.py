"""Hybrid retrieval: vector search (Qdrant) + BM25, fused with Reciprocal Rank Fusion."""
import json
import re

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from rank_bm25 import BM25Okapi

import config as cfg
import llm


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(path=cfg.QDRANT_PATH)
        self.chunks = json.loads(cfg.CHUNKS_JSON.read_text())
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._reranker = None

    def _vector(self, query, allowed, n):
        vec = llm.embed([cfg.EMBED_QUERY_PREFIX + query])[0]
        access = Filter(must=[FieldCondition(key="access", match=MatchAny(any=allowed))])
        res = self.client.query_points(cfg.COLLECTION, query=vec, limit=n,
                                       query_filter=access, with_payload=False)
        return [(p.id, p.score) for p in res.points]

    def _keyword(self, query, allowed, n):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, scores[i]) for i in ranked
                if scores[i] > 0 and self.chunks[i]["access"] in allowed][:n]

    def search(self, query, role="customer", k=None, mode="hybrid"):
        """mode: 'hybrid' | 'vector' | 'bm25' (the other two exist so you can measure the difference)."""
        k = k or cfg.TOP_K
        allowed = cfg.ROLE_ACCESS[role]          # access control happens in the query, not after
        vec = dict(self._vector(query, allowed, cfg.CANDIDATES)) if mode in ("hybrid", "vector") else {}
        kw = self._keyword(query, allowed, cfg.CANDIDATES) if mode in ("hybrid", "bm25") else []

        fused = {}
        for ranking in (list(vec), [i for i, _ in kw]):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        order = sorted(fused, key=fused.get, reverse=True)

        hits = []
        for i in order[: k * 3 if cfg.USE_RERANKER else k]:
            c = dict(self.chunks[i])
            c["vector_score"] = round(vec.get(i, 0.0), 4)
            c["rrf"] = round(fused[i], 5)
            hits.append(c)
        return self._rerank(query, hits)[:k] if cfg.USE_RERANKER else hits

    def _rerank(self, query, hits):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder("BAAI/bge-reranker-base")
        scores = self._reranker.predict([(query, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["rerank"] = float(s)
        return sorted(hits, key=lambda h: h["rerank"], reverse=True)
