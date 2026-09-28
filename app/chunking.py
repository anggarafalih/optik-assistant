"""Split document text into overlapping, paragraph-aligned chunks.

Chunks are packed from whole blocks (paragraphs, lists, tables) so a chunk rarely
cuts a sentence in half. A heading is glued to the block that follows it, so the
heading's context travels with its content. Only a single block longer than
`chunk_size` is split mid-text, using a sliding word window.
"""

import re

_HEADING = re.compile(r"^#{1,6}\s")


def _word_count(text: str) -> int:
    return len(text.split())


def split_blocks(text: str) -> list[str]:
    """Split on blank lines, merging each heading-only block into the next block."""
    raw = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    blocks: list[str] = []
    pending_heading = ""
    for block in raw:
        if _HEADING.match(block) and "\n" not in block:
            pending_heading = f"{pending_heading}\n{block}".strip()
            continue
        blocks.append(f"{pending_heading}\n{block}" if pending_heading else block)
        pending_heading = ""
    if pending_heading:
        blocks.append(pending_heading)
    return blocks


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def _overlap_tail(blocks: list[str], overlap: int) -> list[str]:
    """The trailing whole blocks of a chunk that fit within `overlap` words."""
    tail: list[str] = []
    total = 0
    for block in reversed(blocks):
        total += _word_count(block)
        if total > overlap:
            break
        tail.insert(0, block)
    return tail


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []

    for block in split_blocks(text):
        n = _word_count(block)
        if n > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current = []
            chunks.extend(_split_words(block, chunk_size, overlap))
            continue
        if current and sum(map(_word_count, current)) + n > chunk_size:
            chunks.append("\n\n".join(current))
            current = _overlap_tail(current, overlap)
            while current and sum(map(_word_count, current)) + n > chunk_size:
                current.pop(0)
        current.append(block)

    if current:
        chunks.append("\n\n".join(current))
    return chunks
