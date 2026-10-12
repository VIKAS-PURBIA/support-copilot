"""Runs the whole pipeline with fake LLM + fake embeddings, so no Ollama is needed (works in CI)."""
import re

import pytest

import agents
import config as cfg
import ingest
import llm
from agents import SupportCopilot


def fake_embed(texts):
    """Deterministic bag-of-words vectors: similar wording -> similar vectors."""
    out = []
    for t in texts:
        v = [0.0] * 256
        for w in re.findall(r"\w+", t.lower()):
            v[sum(map(ord, w)) % 256] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / norm for x in v])
    return out


class FakeLLM:
    def __init__(self):
        self.grade = {"sufficient": True}
        self.answer = {"answerable": True, "answer": "Use the reset link from the login page [1]."}
        self.verdict = {"supported": True}
        self.systems_called = []

    def chat_json(self, system, user):
        self.systems_called.append(system)
        if system is agents.ROUTER_SYS:
            return {"intent": "support_question", "reason": "test"}
        if system is agents.GRADER_SYS:
            return self.grade
        if system is agents.ANSWER_SYS:
            return self.answer
        if system is agents.VERIFY_SYS:
            return self.verdict
        if system is agents.ESC_SYS:
            return {"summary": "Customer needs help.", "category": "account", "priority": "low"}
        return {}

    def chat(self, system, user, **kw):
        return "reset password"


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name, value in {
        "STORAGE": tmp_path, "QDRANT_PATH": str(tmp_path / "qdrant"),
        "CHUNKS_JSON": tmp_path / "chunks.json", "EMBED_CACHE": tmp_path / "cache.json",
        "ESCALATIONS_FILE": tmp_path / "esc.jsonl", "MIN_VECTOR_SCORE": 0.2,
        # Ensure the grader runs in pipeline tests, regardless of local/CI defaults.
        "USE_GRADER": True,
    }.items():
        monkeypatch.setattr(cfg, name, value)
    fake = FakeLLM()
    monkeypatch.setattr(llm, "embed", fake_embed)
    monkeypatch.setattr(llm, "chat_json", fake.chat_json)
    monkeypatch.setattr(llm, "chat", fake.chat)
    assert ingest.build() > 0
    return SupportCopilot(), fake


def test_hybrid_search_finds_right_article(env):
    copilot, _ = env
    hits = copilot.retriever.search("how do I reset my password", "customer")
    assert "reset-password.md" in [h["source"] for h in hits]


def test_customers_never_see_internal_tickets(env):
    copilot, _ = env
    for q in ("free plan cannot upload 3 GB video", "charged twice refund duplicate"):
        assert all(h["access"] == "public" for h in copilot.retriever.search(q, "customer"))
    agent_hits = copilot.retriever.search("cannot upload 3 GB video free plan", "agent")
    assert any(h["type"] == "ticket" for h in agent_hits)


def test_answer_has_valid_citations(env):
    copilot, _ = env
    res = copilot.ask("How do I reset my password?")
    assert res["status"] == "answered"
    assert res["citations"] and res["citations"][0]["n"] == 1


def test_escalates_when_grader_says_insufficient(env):
    copilot, fake = env
    fake.grade = {"sufficient": False}
    res = copilot.ask("How do I reset my password?")
    assert res["status"] == "escalated"
    assert cfg.ESCALATIONS_FILE.read_text().count("ESC-") == 1


def test_escalates_when_answer_has_no_citation(env):
    copilot, fake = env
    fake.answer = {"answerable": True, "answer": "Just click the button."}
    assert copilot.ask("How do I reset my password?")["status"] == "escalated"


def test_invalid_citation_numbers_are_stripped(env):
    copilot, fake = env
    fake.answer = {"answerable": True, "answer": "Click the link [1]. Also [99]."}
    res = copilot.ask("How do I reset my password?")
    assert res["status"] == "answered" and "[99]" not in res["answer"]


def test_escalates_when_verifier_finds_unsupported_claims(env):
    copilot, fake = env
    fake.verdict = {"supported": False}
    assert copilot.ask("How do I reset my password?")["status"] == "escalated"


def test_low_retrieval_confidence_escalates_before_calling_llm(env):
    copilot, fake = env
    res = copilot.ask("zzzz qqqq wwww")
    assert res["status"] == "escalated"
    assert agents.ANSWER_SYS not in fake.systems_called


def test_explicit_human_request_skips_rag(env):
    copilot, fake = env
    res = copilot.ask("I want to talk to a human")
    assert res["status"] == "escalated"
    assert agents.ANSWER_SYS not in fake.systems_called


def test_api(env, monkeypatch):
    from fastapi.testclient import TestClient

    import app as app_module
    copilot, _ = env
    monkeypatch.setattr(app_module, "SupportCopilot", lambda: copilot)
    with TestClient(app_module.app) as client:
        ok = client.post("/ask", json={"message": "How do I reset my password?"})
        assert ok.status_code == 200 and ok.json()["status"] == "answered"
        assert client.post("/ask", json={"message": "hi", "role": "hacker"}).status_code == 400
        assert client.get("/health").json() == {"ok": True}
