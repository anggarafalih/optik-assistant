"""LLM client and embedding helpers.

Ollama exposes an OpenAI-compatible API, so the same OpenAI SDK code talks to either
a local Ollama server or the OpenAI API; only the base URL and model names change.
"""

from functools import lru_cache

import numpy as np
from openai import OpenAI

from app.config import get_settings


@lru_cache
def get_client() -> OpenAI:
    settings = get_settings()
    if settings.llm_provider == "ollama":
        # Ollama ignores the key, but the SDK requires one. Local models can be slow to
        # load on the first request, hence the generous timeout.
        return OpenAI(base_url=settings.ollama_base_url, api_key="ollama", timeout=300)
    # Passing None lets the SDK fall back to the OPENAI_API_KEY environment variable.
    return OpenAI(api_key=settings.openai_api_key or None)


def generation_params(model: str) -> dict:
    """Sampling settings that differ by provider and model."""
    settings = get_settings()
    if settings.llm_provider == "ollama":
        # Deterministic output: small models drift between languages and break format
        # more often at Ollama's default temperature.
        return {"temperature": 0}
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        # Reasoning models reject temperature but accept a reasoning effort.
        return {"reasoning_effort": settings.reasoning_effort}
    return {"temperature": 0}


def embed_texts(texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
    settings = get_settings()
    client = get_client()
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        # The API returns items in input order; sort by index to be safe.
        for item in sorted(response.data, key=lambda d: d.index):
            vectors.append(np.array(item.embedding, dtype=np.float32))
    for v in vectors:
        if v.shape[0] != settings.embedding_dim:
            raise ValueError(
                f"{settings.embedding_model} returned {v.shape[0]}-dim vectors but EMBEDDING_DIM is "
                f"{settings.embedding_dim}. Set EMBEDDING_DIM={v.shape[0]} in .env and run "
                "`python -m app.ingest --reset`."
            )
    return vectors


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
