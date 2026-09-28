import json
from pathlib import Path

import pytest

from app.chunking import chunk_text
from app.documents import load_documents, parse_document

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "knowledge_base"


def test_parse_document_reads_front_matter():
    doc = parse_document("---\nid: a\ntitle: A: The Title\nlanguage: en\nsample_policy: true\n---\n\n# Body\n")
    assert doc.id == "a"
    assert doc.title == "A: The Title"  # only the first colon splits key from value
    assert doc.sample_policy is True
    assert doc.category == "general"
    assert doc.content == "# Body"


def test_missing_required_field_raises():
    with pytest.raises(ValueError, match="language"):
        parse_document("---\nid: a\ntitle: A\n---\nBody")


def test_knowledge_base_loads_and_is_bilingual():
    docs = load_documents(KB)
    assert 15 <= len(docs) <= 25
    languages = {d.language for d in docs}
    assert languages == {"en", "id"}


def test_sample_policies_are_labelled():
    for doc in load_documents(KB):
        if doc.category == "store-policy":
            assert doc.sample_policy, doc.id
            assert "SAMPLE DATA" in doc.content or "DATA CONTOH" in doc.content, doc.id


def test_every_document_chunks_cleanly():
    for doc in load_documents(KB):
        chunks = chunk_text(doc.content, chunk_size=200, overlap=40)
        assert chunks, doc.id
        assert all(len(c.split()) <= 200 for c in chunks), doc.id


def test_eval_questions_reference_real_documents():
    doc_ids = {d.id for d in load_documents(KB)}
    lines = (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    assert len(questions) == 30
    assert sum(not q["in_scope"] for q in questions) == 5
    assert len({q["id"] for q in questions}) == len(questions)
    for q in questions:
        assert set(q["expected_sources"]) <= doc_ids, q["id"]
        assert bool(q["expected_sources"]) == q["in_scope"], q["id"]
