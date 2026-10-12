"""Thin wrappers around Ollama's REST API (plain HTTP, so no SDK version issues)."""
import json

import requests

import config as cfg


def embed(texts: list[str]) -> list[list[float]]:
    r = requests.post(f"{cfg.OLLAMA_URL}/api/embed",
                      json={"model": cfg.EMBED_MODEL, "input": texts}, timeout=300)
    r.raise_for_status()
    return r.json()["embeddings"]


def chat(system: str, user: str, json_mode: bool = False, temperature: float = 0.1) -> str:
    payload = {
        "model": cfg.CHAT_MODEL,
        "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{cfg.OLLAMA_URL}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def chat_json(system: str, user: str) -> dict:
    """Ask for JSON; return {} if the model produced something unparseable."""
    raw = chat(system, user, json_mode=True)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
