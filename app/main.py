"""FastAPI backend: POST /chat plus the static chat page.

Run with:  uvicorn app.main:app --reload
"""

import logging
from dataclasses import asdict
from pathlib import Path

import openai
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import get_pool
from app.rag import answer_question

logger = logging.getLogger("optik")

app = FastAPI(title="Optik Assistant", description="RAG customer-service chatbot for an eyewear store")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    doc_id: str
    title: str
    language: str
    sample_policy: bool
    similarity: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    in_scope: bool
    sources: list[SourceOut]
    model: str
    latency_ms: int


# Plain `def` (not async): the OpenAI and psycopg calls are blocking, so FastAPI runs
# this in its thread pool instead of blocking the event loop.
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = answer_question(request.question.strip(), top_k=request.top_k)
    except openai.APIConnectionError as exc:
        logger.exception("Could not reach the model server")
        hint = " Is Ollama running?" if get_settings().llm_provider == "ollama" else ""
        raise HTTPException(status_code=503, detail=f"Could not reach the language model.{hint}") from exc
    except openai.OpenAIError as exc:
        logger.exception("OpenAI request failed")
        raise HTTPException(status_code=502, detail=f"Language model error: {exc.__class__.__name__}") from exc
    return ChatResponse(
        answer=result.answer,
        in_scope=result.in_scope,
        sources=[SourceOut(**asdict(s)) for s in result.sources],
        model=result.model,
        latency_ms=result.latency_ms,
    )


@app.get("/health")
def health() -> dict:
    with get_pool().connection() as conn:
        docs, chunks = conn.execute(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
        ).fetchone()
    settings = get_settings()
    return {"status": "ok", "provider": settings.llm_provider, "model": settings.chat_model,
            "embedding_model": settings.embedding_model, "documents": docs, "chunks": chunks}


# Mounted last so the API routes above take precedence over static files.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
