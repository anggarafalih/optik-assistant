"""Load knowledge-base documents: Markdown files with a small front-matter header."""

from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = ("id", "title", "language")


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    language: str
    category: str
    sample_policy: bool
    content: str
    path: Path | None = None


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split `---`-delimited `key: value` front matter from the body.

    A full YAML parser is overkill for flat string fields, so this avoids the dependency.
    """
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        raise ValueError("front matter opened with '---' but never closed") from None

    meta = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"invalid front-matter line: {line!r}")
        meta[key.strip()] = value.strip()
    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def parse_document(text: str, path: Path | None = None) -> Document:
    meta, body = parse_front_matter(text)
    missing = [f for f in REQUIRED_FIELDS if not meta.get(f)]
    if missing:
        raise ValueError(f"{path or 'document'} is missing front-matter fields: {missing}")
    return Document(
        id=meta["id"],
        title=meta["title"],
        language=meta["language"],
        category=meta.get("category", "general"),
        sample_policy=meta.get("sample_policy", "false").lower() == "true",
        content=body,
        path=path,
    )


def load_documents(directory: Path) -> list[Document]:
    docs = [parse_document(p.read_text(encoding="utf-8"), p) for p in sorted(directory.glob("*.md"))]
    ids = [d.id for d in docs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate document ids: {sorted(duplicates)}")
    return docs
