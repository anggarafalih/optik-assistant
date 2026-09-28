"""Run the test set through the RAG pipeline and score it.

Usage:
    python -m eval.run_eval --label baseline
    python -m eval.run_eval --label topk-3 --top-k 3
    python -m eval.run_eval --label gpt-5 --model gpt-5

Chunk size is an ingest-time setting: re-run `python -m app.ingest --chunk-size N` first.

Scoring:
- In-scope questions are graded by an LLM judge against the expected answer
  (correct / partial / incorrect). A refusal on an in-scope question is a false refusal.
- Out-of-scope questions are correct when the assistant refuses (in_scope = false).
- Retrieval hit: at least one expected source document is among the retrieved chunks.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.config import get_settings
from app.db import get_pool
from app.llm import get_client, generation_params
from app.rag import answer_question

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# USD per 1M tokens (input, output). Prices change: check https://openai.com/api/pricing
# before quoting these numbers. Reasoning tokens are billed as output tokens.
PRICES_PER_1M = {
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}

JUDGE_PROMPT = """\
You grade answers from an eyewear store's customer-service chatbot.

Compare the chatbot answer with the reference answer and choose a verdict:
- "correct": the answer contains the key facts of the reference and states nothing that \
contradicts it.
- "partial": the answer is on track but misses a key fact from the reference, and states \
nothing wrong.
- "incorrect": the answer contradicts the reference, states wrong facts, or refuses to answer.

Important:
- Extra details, tips, or context beyond the reference are NOT a reason for "partial". Only a \
missing key fact or a wrong statement lowers the verdict.
- The answer may be in a different language from the reference, and may word the facts \
differently. Judge the meaning.
- Check each key fact of the reference carefully against the whole answer before deciding \
that it is missing.

First list the key facts of the reference and whether each appears in the answer, then give \
the verdict."""


class Verdict(BaseModel):
    reason: str
    verdict: Literal["correct", "partial", "incorrect"]


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge(question: str, expected: str, answer: str, model: str) -> Verdict:
    response = get_client().chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user",
             "content": f"Question: {question}\n\nReference answer: {expected}\n\nChatbot answer: {answer}"},
        ],
        response_format=Verdict,
        **generation_params(model),
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("judge returned no parsable verdict")
    return parsed


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if get_settings().llm_provider == "ollama":
        return 0.0  # runs locally
    prices = PRICES_PER_1M.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def chunk_config() -> str:
    with get_pool().connection() as conn:
        rows = conn.execute("SELECT DISTINCT chunk_size, chunk_overlap FROM documents").fetchall()
    return ", ".join(f"{size}/{overlap}" for size, overlap in rows) or "unknown"


def evaluate_one(q: dict, top_k: int, model: str, judge_model: str) -> dict:
    record = {"id": q["id"], "language": q["language"], "question": q["question"],
              "expected_in_scope": q["in_scope"], "expected_answer": q["expected_answer"],
              "expected_sources": q["expected_sources"]}
    try:
        result = answer_question(q["question"], top_k=top_k, model=model)
    except Exception as exc:  # record the failure and keep going
        return {**record, "error": repr(exc), "outcome": "error"}

    retrieved_ids = list(dict.fromkeys(c.doc_id for c in result.retrieved))
    cited_ids = [s.doc_id for s in result.sources]
    record.update(
        answer=result.answer,
        predicted_in_scope=result.in_scope,
        retrieved_doc_ids=retrieved_ids,
        top_similarity=round(result.retrieved[0].similarity, 4) if result.retrieved else None,
        cited_doc_ids=cited_ids,
        latency_ms=result.latency_ms,
        input_tokens=result.usage.get("input_tokens", 0),
        output_tokens=result.usage.get("output_tokens", 0),
    )

    if not q["in_scope"]:
        record["outcome"] = "correct_refusal" if not result.in_scope else "answered_out_of_scope"
        return record

    expected = set(q["expected_sources"])
    record["retrieval_hit"] = bool(expected & set(retrieved_ids))
    record["citation_hit"] = bool(expected & set(cited_ids))
    if not result.in_scope:
        record["outcome"] = "false_refusal"
        return record
    verdict = judge(q["question"], q["expected_answer"], result.answer, judge_model)
    record["judge_reason"] = verdict.reason
    record["outcome"] = verdict.verdict
    return record


def summarize(records: list[dict], model: str) -> dict:
    in_scope = [r for r in records if r["expected_in_scope"]]
    out_scope = [r for r in records if not r["expected_in_scope"]]
    ok = [r for r in records if r["outcome"] != "error"]

    def count(rows: list[dict], outcome: str) -> int:
        return sum(r["outcome"] == outcome for r in rows)

    latencies = sorted(r["latency_ms"] for r in ok)
    input_tokens = sum(r["input_tokens"] for r in ok)
    output_tokens = sum(r["output_tokens"] for r in ok)
    cost = estimate_cost(model, input_tokens, output_tokens)

    return {
        "in_scope_total": len(in_scope),
        "correct": count(in_scope, "correct"),
        "partial": count(in_scope, "partial"),
        "incorrect": count(in_scope, "incorrect"),
        "false_refusals": count(in_scope, "false_refusal"),
        "out_of_scope_total": len(out_scope),
        "correct_refusals": count(out_scope, "correct_refusal"),
        "errors": count(records, "error"),
        "retrieval_hit_rate": _ratio(sum(r.get("retrieval_hit", False) for r in in_scope), len(in_scope)),
        "citation_hit_rate": _ratio(sum(r.get("citation_hit", False) for r in in_scope), len(in_scope)),
        "latency_ms_mean": round(statistics.mean(latencies)) if latencies else None,
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
        "latency_ms_p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "est_cost_usd": round(cost, 5) if cost is not None else None,
        "est_cost_per_1k_questions_usd": round(cost / len(ok) * 1000, 2) if cost is not None and ok else None,
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render_report(run: dict) -> str:
    s, c = run["summary"], run["config"]
    lines = [
        f"# Eval run: {run['label']}",
        "",
        f"- Date: {run['timestamp']}",
        f"- Model: `{c['model']}` via {c['provider']} (reasoning effort `{c['reasoning_effort']}`, OpenAI only), judge: `{c['judge_model']}`",
        f"- Retrieval: top_k={c['top_k']}, min_similarity={c['min_similarity']}, "
        f"chunk size/overlap (words)={c['chunk_config']}, embeddings `{c['embedding_model']}`",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| In-scope answered correctly | {s['correct']}/{s['in_scope_total']} "
        f"({_pct(_ratio(s['correct'], s['in_scope_total']))}) |",
        f"| In-scope partially correct | {s['partial']} |",
        f"| In-scope incorrect | {s['incorrect']} |",
        f"| In-scope wrongly refused | {s['false_refusals']} |",
        f"| Out-of-scope correctly refused | {s['correct_refusals']}/{s['out_of_scope_total']} |",
        f"| Retrieval hit rate (expected doc in top-k) | {_pct(s['retrieval_hit_rate'])} |",
        f"| Citation hit rate (expected doc cited) | {_pct(s['citation_hit_rate'])} |",
        f"| Latency mean / p50 / p95 | {s['latency_ms_mean']} / {s['latency_ms_p50']} / {s['latency_ms_p95']} ms |",
        f"| Tokens in / out | {s['input_tokens']:,} / {s['output_tokens']:,} |",
        f"| Est. cost per 1,000 questions | "
        f"{'n/a' if s['est_cost_per_1k_questions_usd'] is None else '$' + str(s['est_cost_per_1k_questions_usd'])} |",
        f"| Errors | {s['errors']} |",
        "",
        "## Questions that were not fully correct",
        "",
    ]
    misses = [r for r in run["results"] if r["outcome"] not in ("correct", "correct_refusal")]
    if not misses:
        lines.append("None.")
    for r in misses:
        lines += [
            f"### {r['id']} ({r['outcome']}): {r['question']}",
            "",
            f"- Expected: {r['expected_answer']}",
            f"- Got: {r.get('answer', r.get('error'))}",
        ]
        if "judge_reason" in r:
            lines.append(f"- Judge: {r['judge_reason']}")
        if "retrieved_doc_ids" in r:
            lines.append(f"- Retrieved: {', '.join(r['retrieved_doc_ids'])} "
                         f"(expected: {', '.join(r['expected_sources']) or 'none'})")
        lines.append("")
    return "\n".join(lines)


def append_to_log(run: dict) -> None:
    """Keep one comparison table across runs in results/runs.md."""
    log = RESULTS_DIR / "runs.md"
    s, c = run["summary"], run["config"]
    if not log.exists():
        log.write_text(
            "# Eval runs\n\n"
            "| Date | Label | Model | Chunk | top_k | Correct | Partial | False refusals "
            "| OOS refused | Retrieval hit | p50 latency | $/1k q |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    row = (f"| {run['timestamp'][:16]} | {run['label']} | {c['model']} | {c['chunk_config']} | {c['top_k']} "
           f"| {s['correct']}/{s['in_scope_total']} | {s['partial']} | {s['false_refusals']} "
           f"| {s['correct_refusals']}/{s['out_of_scope_total']} | {_pct(s['retrieval_hit_rate'])} "
           f"| {s['latency_ms_p50']} ms | {'n/a' if s['est_cost_per_1k_questions_usd'] is None else s['est_cost_per_1k_questions_usd']} |\n")
    with log.open("a", encoding="utf-8") as f:
        f.write(row)


def main() -> None:
    # The Windows console defaults to cp1252, which can't print characters like "−" or "□".
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="short name for this run, e.g. baseline")
    parser.add_argument("--model", default=settings.chat_model)
    parser.add_argument("--judge-model", default=settings.chat_model,
                        help="model that grades answers (default: the chat model). A larger model "
                             "grades more reliably, e.g. qwen2.5:7b on Ollama.")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--limit", type=int, help="only run the first N questions (for a quick smoke test)")
    args = parser.parse_args()

    questions = load_questions(args.questions)[: args.limit]
    config = {
        "provider": settings.llm_provider,
        "model": args.model,
        "reasoning_effort": settings.reasoning_effort,
        "judge_model": args.judge_model,
        "top_k": args.top_k,
        "min_similarity": settings.min_similarity,
        "chunk_config": chunk_config(),
        "embedding_model": settings.embedding_model,
    }

    started = time.perf_counter()
    records = []
    for i, q in enumerate(questions, 1):
        record = evaluate_one(q, args.top_k, args.model, args.judge_model)
        records.append(record)
        print(f"[{i:2}/{len(questions)}] {q['id']} {record['outcome']:<22} {q['question'][:60]}")

    run = {
        "label": args.label,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "summary": summarize(records, args.model),
        "results": records,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{args.label}"
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_report(run)
    (RESULTS_DIR / f"{stem}.md").write_text(report, encoding="utf-8")
    append_to_log(run)

    print()
    print("\n".join(report.split("\n## ")[0].splitlines()[6:]))
    print(f"\nFinished in {time.perf_counter() - started:.0f}s. Report: eval/results/{stem}.md")


if __name__ == "__main__":
    main()
