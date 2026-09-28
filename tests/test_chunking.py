import pytest

from app.chunking import chunk_text, split_blocks


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.\n\nSecond paragraph.", chunk_size=50, overlap=10) == [
        "Hello world.\n\nSecond paragraph."
    ]


def test_heading_is_glued_to_following_block():
    blocks = split_blocks("# Title\n\n## Section\n\nBody text.\n\nMore.")
    assert blocks == ["# Title\n## Section\nBody text.", "More."]


def test_chunks_respect_size_limit():
    text = "\n\n".join(words(30, f"p{i}_") for i in range(10))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunks_split_on_paragraph_boundaries():
    paragraphs = [words(30, f"p{i}_") for i in range(6)]
    chunks = chunk_text("\n\n".join(paragraphs), chunk_size=100, overlap=0)
    for chunk in chunks:
        assert all(p in paragraphs for p in chunk.split("\n\n"))


def test_overlap_repeats_trailing_paragraph():
    paragraphs = [words(30, f"p{i}_") for i in range(6)]
    chunks = chunk_text("\n\n".join(paragraphs), chunk_size=100, overlap=40)
    # Each chunk after the first starts with the last paragraph of the previous one.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split("\n\n")[0] == prev.split("\n\n")[-1]


def test_long_paragraph_uses_sliding_window():
    chunks = chunk_text(words(250), chunk_size=100, overlap=20)
    assert [len(c.split()) for c in chunks] == [100, 100, 90]
    assert chunks[1].split()[0] == "w80"  # starts 20 words before the end of chunk 1


def test_all_text_is_kept():
    text = "\n\n".join(words(45, f"p{i}_") for i in range(7))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert set(text.split()) == set(" ".join(chunks).split())


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=size, overlap=overlap)
