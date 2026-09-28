from app.config import Settings


def test_ollama_defaults(monkeypatch):
    for var in ("CHAT_MODEL", "EMBEDDING_MODEL", "EMBEDDING_DIM"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None, llm_provider="ollama")
    assert (s.chat_model, s.embedding_model, s.embedding_dim) == ("gemma3:4b", "bge-m3", 1024)


def test_openai_defaults():
    s = Settings(_env_file=None, llm_provider="openai")
    assert (s.chat_model, s.embedding_model, s.embedding_dim) == ("gpt-5-mini", "text-embedding-3-small", 1536)


def test_explicit_values_override_provider_defaults():
    s = Settings(_env_file=None, llm_provider="ollama", chat_model="qwen2.5:7b")
    assert s.chat_model == "qwen2.5:7b"
    assert s.embedding_model == "bge-m3"
