"""FastAPI backend.  Run: uvicorn app:app --port 8000"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config as cfg
from agents import SupportCopilot

state = {}


@asynccontextmanager
async def lifespan(app):
    state["copilot"] = SupportCopilot()
    yield


app = FastAPI(title="Support Copilot", lifespan=lifespan)


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    role: str = "customer"          # demo only: in production take this from the auth token
    history: list[dict] = []


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(req: AskRequest):
    if req.role not in cfg.ROLE_ACCESS:
        raise HTTPException(400, f"role must be one of {list(cfg.ROLE_ACCESS)}")
    return state["copilot"].ask(req.message, req.role, req.history)


@app.get("/escalations")        # TODO: protect with an API key before exposing anywhere
def escalations():
    if not cfg.ESCALATIONS_FILE.exists():
        return []
    lines = cfg.ESCALATIONS_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()][::-1]
