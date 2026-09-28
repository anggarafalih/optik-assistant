import json
from pathlib import Path

import pytest

from app.language import detect_language
from app.rag import (
    REFUSALS, build_user_message, collect_sources, is_refusal, parse_reply, strip_inline_citations,
    strip_refusal,
)
from app.retrieval import RetrievedChunk


def chunk(doc_id: str, similarity: float, index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(doc_id, f"Title {doc_id}", "en", "lenses", False, index, f"Content of {doc_id}", similarity)


def test_collect_sources_drops_ids_that_were_not_retrieved():
    sources = collect_sources(["a", "invented-id"], [chunk("a", 0.8), chunk("b", 0.7)])
    assert [s.doc_id for s in sources] == ["a"]


def test_collect_sources_one_entry_per_document_with_best_similarity():
    sources = collect_sources(["a", "b"], [chunk("a", 0.6, 0), chunk("b", 0.7), chunk("a", 0.9, 1)])
    assert [(s.doc_id, s.similarity) for s in sources] == [("a", 0.9), ("b", 0.7)]


def test_user_message_contains_sources_question_and_language():
    message = build_user_message("What is PD?", [chunk("a", 0.8)], "en")
    assert '<source id="a" title="Title a">' in message
    assert "<question>\nWhat is PD?\n</question>" in message
    assert "Reply language: English" in message
    assert "Refusal" not in message  # the refusal sentence lives in the system prompt


def test_user_message_without_sources_says_so():
    assert "(no relevant sources found)" in build_user_message("Hi", [], "id")


@pytest.mark.parametrize("reply, answer, ids", [
    ("Most people adapt in 1-2 weeks.\nSOURCES: lens-progressive", "Most people adapt in 1-2 weeks.",
     ["lens-progressive"]),
    ("Answer.\n\n**Sources:** a, b.", "Answer.", ["a", "b"]),
    ("Answer.\nSOURCES: [a]; [b]", "Answer.", ["a", "b"]),
    (f"{REFUSALS['en']}\nSOURCES: NONE", REFUSALS["en"], []),
    ("An answer with no sources line.", "An answer with no sources line.", None),
    ("Heat can crack coatings. SOURCES: lens-coatings, perawatan-kacamata", "Heat can crack coatings.",
     ["lens-coatings", "perawatan-kacamata"]),
    ("See the sources: they agree.\nMore text.", "See the sources: they agree.\nMore text.", None),
])
def test_parse_reply(reply, answer, ids):
    assert parse_reply(reply) == (answer, ids)


def test_is_refusal_detects_either_language():
    assert is_refusal(REFUSALS["id"] + " Silakan tanya staf toko.")
    assert is_refusal(REFUSALS["en"].replace("'", "’"))  # curly apostrophe
    assert not is_refusal("Most people adapt within 1 to 2 weeks.")


def test_strip_refusal_keeps_the_real_answer():
    reply = REFUSALS["en"].replace("'", "’") + " It is not okay: heat can crack the coating."
    assert strip_refusal(reply) == "It is not okay: heat can crack the coating."


@pytest.mark.parametrize("question, language", [
    ("Berapa lama adaptasi lensa progresif?", "id"),
    ("Apa arti AXIS di resep kacamata?", "id"),
    ("Jam berapa toko buka hari Minggu?", "id"),
    ("Can I sleep with my contact lenses in?", "en"),
    ("How much do progressive lenses cost?", "en"),
    ("What does 52□18 140 printed on my frame mean?", "en"),
])
def test_detect_language(question, language):
    assert detect_language(question) == language


def test_detect_language_on_every_eval_question():
    path = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        assert detect_language(q["question"]) == q["language"], q["id"]


def test_strip_inline_citations_only_removes_known_ids():
    answer, ids = strip_inline_citations("Rinse it [contact-lens-care]. See [note] too [x-1].", {"contact-lens-care"})
    assert answer == "Rinse it. See [note] too [x-1]."
    assert ids == ["contact-lens-care"]
