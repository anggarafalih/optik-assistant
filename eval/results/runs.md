# Eval runs

| Date | Label | Model | Chunk | top_k | Correct | Partial | False refusals | OOS refused | Retrieval hit | p50 latency | $/1k q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-28T09:47 | ollama-baseline | gemma3:4b | 200/40 | 5 | 16/25 | 8 | 1 | 5/5 | 100% | 4494 ms | 0.0 |
| 2026-09-28T09:52 | ollama-v2 | gemma3:4b | 200/40 | 5 | 21/25 | 3 | 0 | 4/5 | 100% | 4708 ms | 0.0 |
| 2026-09-28T09:57 | ollama-v3 | gemma3:4b | 200/40 | 5 | 21/25 | 3 | 0 | 5/5 | 100% | 4771 ms | 0.0 |
