"""Load, chunk, embed, and store the knowledge base.

Usage:
    python -m app.ingest                     # only re-embed documents that changed
    python -m app.ingest --rebuild           # re-embed everything
    python -m app.ingest --chunk-size 300 --overlap 50
    python -m app.ingest --reset             # drop tables first (after changing embedding model)
"""

import argparse
import hashlib

from app.chunking import chunk_text
from app.config import get_settings
from app.db import get_pool, init_schema
from app.documents import Document, load_documents
from app.llm import embed_texts


def embedding_input(doc: Document, chunk: str) -> str:
    # Prefixing the title gives each chunk its document context. A chunk that only says
    # "Most people adapt within 1 to 2 weeks" otherwise doesn't say what is being adapted to.
    return f"{doc.title}\n\n{chunk}"


def content_hash(doc: Document, chunk_size: int, overlap: int, embedding_model: str) -> str:
    # Chunking settings and the embedding model are part of the hash, so changing them
    # re-embeds every document even when the text itself didn't change.
    key = "\x1f".join([doc.title, doc.language, doc.category, str(doc.sample_policy), doc.content,
                       str(chunk_size), str(overlap), embedding_model])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def ingest(chunk_size: int, overlap: int, rebuild: bool = False, reset: bool = False) -> None:
    settings = get_settings()
    docs = load_documents(settings.knowledge_base_dir)
    init_schema(reset=reset)
    pool = get_pool()

    with pool.connection() as conn:
        existing = dict(conn.execute("SELECT id, content_hash FROM documents").fetchall())

    stale = set(existing) - {d.id for d in docs}
    if stale:
        with pool.connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = ANY(%s)", (list(stale),))
        print(f"Removed {len(stale)} deleted document(s): {', '.join(sorted(stale))}")

    embedded = skipped = total_chunks = 0
    for doc in docs:
        digest = content_hash(doc, chunk_size, overlap, settings.embedding_model)
        if not rebuild and existing.get(doc.id) == digest:
            skipped += 1
            continue

        chunks = chunk_text(doc.content, chunk_size, overlap)
        vectors = embed_texts([embedding_input(doc, c) for c in chunks])

        # One transaction per document: a failure never leaves a document half-ingested.
        with pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM documents WHERE id = %s", (doc.id,))
            conn.execute(
                """INSERT INTO documents
                       (id, title, language, category, sample_policy, content_hash, chunk_size, chunk_overlap)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (doc.id, doc.title, doc.language, doc.category, doc.sample_policy, digest, chunk_size, overlap),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (doc_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                    [(doc.id, i, c, v) for i, (c, v) in enumerate(zip(chunks, vectors))],
                )
        embedded += 1
        total_chunks += len(chunks)
        print(f"  {doc.id}: {len(chunks)} chunk(s)")

    print(f"Embedded {embedded} document(s) ({total_chunks} chunks), {skipped} unchanged. "
          f"chunk_size={chunk_size}, overlap={overlap}, embeddings={settings.embedding_model}")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap, help="words of overlap")
    parser.add_argument("--rebuild", action="store_true", help="re-embed all documents")
    parser.add_argument("--reset", action="store_true",
                        help="drop and recreate the tables (needed after switching to an embedding "
                             "model with a different dimension)")
    args = parser.parse_args()
    ingest(args.chunk_size, args.overlap, args.rebuild, args.reset)


if __name__ == "__main__":
    main()
