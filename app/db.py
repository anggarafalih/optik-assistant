"""PostgreSQL + pgvector connection pool and schema."""

from functools import lru_cache

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import get_settings


def _schema(dim: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS documents (
        id            TEXT PRIMARY KEY,
        title         TEXT NOT NULL,
        language      TEXT NOT NULL,
        category      TEXT NOT NULL,
        sample_policy BOOLEAN NOT NULL DEFAULT FALSE,
        content_hash  TEXT NOT NULL,
        chunk_size    INT NOT NULL,
        chunk_overlap INT NOT NULL,
        ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id          BIGSERIAL PRIMARY KEY,
        doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        chunk_index INT NOT NULL,
        content     TEXT NOT NULL,
        embedding   vector({dim}) NOT NULL,
        UNIQUE (doc_id, chunk_index)
    );

    -- HNSW index for approximate nearest-neighbour search on cosine distance.
    -- With ~100 chunks a sequential scan is just as fast, but this is how it scales.
    CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
        ON chunks USING hnsw (embedding vector_cosine_ops);
    """


def _stored_dim(conn: psycopg.Connection) -> int | None:
    """Dimension of the existing chunks.embedding column (pgvector keeps it in atttypmod)."""
    row = conn.execute(
        """SELECT a.atttypmod FROM pg_attribute a
           WHERE a.attrelid = to_regclass('chunks') AND a.attname = 'embedding'"""
    ).fetchone()
    return row[0] if row else None


def init_schema(reset: bool = False) -> None:
    """Create the pgvector extension and tables. Safe to run repeatedly.

    Switching embedding models can change the vector dimension. An empty table is
    recreated automatically; a table with data needs an explicit reset.
    """
    settings = get_settings()
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        dim = _stored_dim(conn)
        if dim is not None and dim != settings.embedding_dim and not reset:
            if conn.execute("SELECT EXISTS (SELECT 1 FROM chunks)").fetchone()[0]:
                raise RuntimeError(
                    f"The database holds {dim}-dim embeddings but {settings.embedding_model} "
                    f"produces {settings.embedding_dim}. Run `python -m app.ingest --reset` "
                    "to drop the old embeddings and re-ingest."
                )
            reset = True
        if reset:
            conn.execute("DROP TABLE IF EXISTS chunks, documents")
        conn.execute(_schema(settings.embedding_dim))


def _configure(conn: psycopg.Connection) -> None:
    # register_vector looks up the vector type's OID, which opens a transaction.
    # The pool only accepts connections handed back idle, so close it.
    register_vector(conn)
    conn.commit()


@lru_cache
def get_pool() -> ConnectionPool:
    settings = get_settings()
    return ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=5,
        configure=_configure,
        open=True,
    )
