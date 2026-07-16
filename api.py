"""
FastAPI wrapper around the lead-gen agent's qualification step.

Deliberately scoped small for the first Docker pass: only /qualify (via
assess_prospect) is exposed here — not draft_email or the LangGraph
human-in-the-loop flow. Those pull in RAG/embeddings and multi-request
state across calls, which is its own follow-up once this baseline is
containerized and working.
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

import pipeline
from anthropic_backend import AnthropicClient
from ollama_backend import OllamaClient

load_dotenv(override=True)

app = FastAPI()

# Backend is chosen via env var, not hardcoded, so the same built image can
# run either way at `docker run` time without a rebuild. Anthropic is the
# default here specifically because it's a plain outbound HTTPS call — no
# dependency on reaching a service on the host machine. Ollama needs the
# container to reach the host's Ollama server (host.docker.internal), which
# is a real but deliberately deferred gap, same as your other "not yet"s.
BACKEND = os.getenv("BACKEND", "anthropic").lower()


def make_client():
    if BACKEND == "ollama":
        return OllamaClient()
    return AnthropicClient()


client = make_client()


class QualifyRequest(BaseModel):
    url: str
    name: str = ""


class QualifyResponse(BaseModel):
    url: str
    company_name: str
    website_quality_score: int
    lead_score: int
    tier: str
    observations: list[str]
    opportunities: list[str]
    summary: str


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND}


@app.post("/qualify", response_model=QualifyResponse)
def qualify(request: QualifyRequest):
    try:
        return pipeline.assess_prospect(client, request.name, request.url)
    except Exception as e:
        return pipeline.schemas.normalize({}, request.url, request.url) | {
            "summary": f"ERROR: {e}",
            "tier": "cold",
            "lead_score": 0,
        }
