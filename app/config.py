from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent

# Defaults per provider, used for any model setting not given in .env.
PROVIDER_DEFAULTS = {
    # Local and free. gemma3:4b fits in 4 GB of VRAM and handles Indonesian well.
    # bge-m3 is a multilingual embedding model, so Indonesian questions can match
    # English documents and vice versa.
    "ollama": {"chat_model": "gemma3:4b", "embedding_model": "bge-m3", "embedding_dim": 1024},
    "openai": {"chat_model": "gpt-5-mini", "embedding_model": "text-embedding-3-small", "embedding_dim": 1536},
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = ""
    database_url: str = "postgresql://optik:optik@localhost:5433/optik"

    chat_model: str | None = None
    # Only used by OpenAI reasoning models (GPT-5 family). "low" keeps latency and cost
    # down for short, grounded answers; raise it if the eval shows reasoning mistakes.
    reasoning_effort: str = "low"

    embedding_model: str | None = None
    embedding_dim: int | None = None

    knowledge_base_dir: Path = ROOT_DIR / "knowledge_base"
    chunk_size: int = 200  # words
    chunk_overlap: int = 40  # words
    top_k: int = 5
    # Chunks below this cosine similarity are dropped before prompting. 0 disables the filter.
    min_similarity: float = 0.0

    @model_validator(mode="after")
    def _fill_provider_defaults(self) -> "Settings":
        for key, value in PROVIDER_DEFAULTS[self.llm_provider].items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
