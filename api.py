"""
FastAPI service for the lead-gen agent.

Turns the pipeline into a deployable HTTP API instead of a local script.

    uvicorn api:app --reload
    -> docs at http://127.0.0.1:8000/docs

Endpoints:
    GET  /health          - liveness + which backends are configured
    POST /qualify         - score ONE prospect (synchronous, ~10-30s)
    POST /jobs            - queue a BATCH, returns a job id immediately
    GET  /jobs/{job_id}   - poll batch progress / results
    GET  /jobs            - list jobs

Why two shapes: a single prospect is slow but tolerable to wait for. A batch of
20 would exceed any sensible HTTP timeout, so batches run in the background and
the client polls. That split is the normal pattern for slow work behind an API.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
import pipeline
from anthropic_backend import AnthropicClient
from ollama_backend import OllamaClient

app = FastAPI(
    title="Lead-Gen Agent API",
    description="Scrapes, qualifies, and drafts outreach for prospect websites.",
    version="1.0.0",
)


# ------------------------------------------------------------------ schemas
class Backend(str, Enum):
    ollama = "ollama"
    anthropic = "anthropic"


class QualifyRequest(BaseModel):
    url: str = Field(..., examples=["https://example.com"])
    name: str = ""
    backend: Backend = Backend.ollama
    model: Optional[str] = None
    draft: bool = Field(True, description="Also draft an email if the lead qualifies")
    threshold: int = Field(default=config.DRAFT_THRESHOLD, ge=0, le=100)


class Assessment(BaseModel):
    url: str
    company_name: str
    lead_score: int
    website_quality_score: int
    tier: str
    observations: List[str] = []
    opportunities: List[str] = []
    summary: str


class QualifyResponse(BaseModel):
    assessment: Assessment
    email: Optional[str] = None
    drafted: bool = False


class BatchRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1)
    backend: Backend = Backend.ollama
    model: Optional[str] = None
    draft: bool = True
    threshold: int = Field(default=config.DRAFT_THRESHOLD, ge=0, le=100)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    total: int
    completed: int
    created_at: str
    results: List[QualifyResponse] = []
    error: Optional[str] = None


# In-memory job store. Fine for a single-process demo; a real deployment would
# use Redis or a database so jobs survive a restart and scale past one worker.
JOBS: Dict[str, Job] = {}


# ------------------------------------------------------------------- helpers
def make_client(backend: Backend, model: Optional[str]):
    if backend == Backend.anthropic:
        return AnthropicClient(model=model)
    return OllamaClient(model=model)


def process_one(client, url: str, name: str, draft: bool, threshold: int) -> QualifyResponse:
    """One prospect through the existing pipeline functions."""
    assessment = pipeline.assess_prospect(client, name, url)

    email = None
    qualifies = (
        draft
        and assessment["lead_score"] >= threshold
        and assessment["tier"] != "review"
        and not assessment["summary"].startswith(("SCRAPE FAILED", "ERROR"))
    )
    if qualifies:
        email = pipeline.draft_email(client, assessment)

    return QualifyResponse(
        assessment=Assessment(**assessment),
        email=email,
        drafted=email is not None,
    )


def run_batch(job_id: str, req: BatchRequest):
    """Background worker for a batch job."""
    job = JOBS[job_id]
    job.status = JobStatus.running
    try:
        client = make_client(req.backend, req.model)
        for url in req.urls:
            try:
                job.results.append(
                    process_one(client, url, "", req.draft, req.threshold))
            except Exception as e:
                job.results.append(QualifyResponse(
                    assessment=Assessment(
                        url=url, company_name=url, lead_score=0,
                        website_quality_score=0, tier="cold",
                        summary=f"ERROR: {e}"),
                ))
            job.completed += 1
        job.status = JobStatus.done
    except Exception as e:
        job.status = JobStatus.failed
        job.error = str(e)


# ----------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "company": config.COMPANY_NAME,
        "default_models": {
            "ollama": config.OLLAMA_MODEL,
            "anthropic": config.ANTHROPIC_MODEL,
        },
        "rag_enabled": config.USE_RAG,
        "vector_store": config.VECTOR_STORE,
    }


@app.post("/qualify", response_model=QualifyResponse)
def qualify(req: QualifyRequest):
    """Score a single prospect (and optionally draft an email). Synchronous."""
    try:
        client = make_client(req.backend, req.model)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {e}")

    try:
        return process_one(client, req.url, req.name, req.draft, req.threshold)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Analysis failed: {e}")


@app.post("/jobs", response_model=Job, status_code=202)
def create_job(req: BatchRequest, background: BackgroundTasks):
    """Queue a batch. Returns immediately with a job id to poll."""
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = Job(
        job_id=job_id,
        status=JobStatus.queued,
        total=len(req.urls),
        completed=0,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    background.add_task(run_batch, job_id, req)
    return JOBS[job_id]


@app.get("/jobs/{job_id}", response_model=Job)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/jobs", response_model=List[Job])
def list_jobs():
    return list(JOBS.values())