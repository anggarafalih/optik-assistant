"""Vector similarity search over the stored chunks."""

from dataclasses import dataclass

from app.config import get_settings
from app.db import get_pool
from app.llm import embed_query


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    title: str
    language: str
    category: str
    sample_policy: bool
    chunk_index: int
    content: str
    similarity: float


# `<=>` is pgvector's cosine distance operator; cosine similarity = 1 - distance.
# Ordering by the raw distance expression lets Postgres use the HNSW index.
_SEARCH_SQL = """
SELECT c.doc_id, d.title, d.language, d.category, d.sample_policy, c.chunk_index, c.content,
       1 - (c.embedding <=> %(q)s) AS similarity
FROM chunks c
JOIN documents d ON d.id = c.doc_id
ORDER BY c.embedding <=> %(q)s
LIMIT %(k)s
"""


def retrieve(query: str, top_k: int | None = None, min_similarity: float | None = None) -> list[RetrievedChunk]:
    settings = get_settings()
    top_k = top_k or settings.top_k
    min_similarity = settings.min_similarity if min_similarity is None else min_similarity

    vector = embed_query(query)
    with get_pool().connection() as conn:
        rows = conn.execute(_SEARCH_SQL, {"q": vector, "k": top_k}).fetchall()
    chunks = [RetrievedChunk(*row) for row in rows]
    return [c for c in chunks if c.similarity >= min_similarity]
