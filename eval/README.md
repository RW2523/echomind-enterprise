# Golden-Question Eval Harness

Regression suite for the RAG + conversation stack. A fixed set of **golden questions** with
known expected sources and facts runs against the live backend; any change to retrieval
config (RRF weights, CE floors, chunking, prompts, models) can be measured instead of
eyeballed.

## Run it

```bash
# whole suite (backend must be up; goes through the nginx proxy)
python3 eval/run_eval.py

# one set
python3 eval/run_eval.py --set health

# with LLM-as-judge grading (slower; uses local ollama)
python3 eval/run_eval.py --judge
```

Exit code 0 = all pass (CI-friendly). A JSON report lands in `eval/reports/`.

## What is scored

| Type        | Checks |
|-------------|--------|
| `retrieval` | expected docs cited, expected facts in answer, no forbidden strings, ≥1 citation, doc-precision |
| `smalltalk` | **0 citations**, no forbidden strings (e.g. Lawyer greeting must carry no disclaimer) |
| `refusal`   | setup turn seeds a topic; the refusal reply must have 0 citations and not mention the topic |
| `offcorpus` | a question the KB can't answer: 0 citations, sane general answer |

## Golden sets (`eval/golden/*.jsonl`)

One JSON object per line; `#` lines are comments. Fields: `id`, `type`, `namespace`,
`persona`, `question`, optional `setup` (prior user turns), `expect_docs` (filename
substrings), `expect_facts` (groups of alternatives — ALL groups required, ANY variant per
group), `forbid_facts`, `expect_citations` (`"0"` / `">=1"`).

**Adding questions:** every `expect_facts` variant must be a string that verifiably appears
in the source document — mine the corpus, don't guess:

```bash
docker exec echomind-backend python3 -c "import sqlite3;c=sqlite3.connect('/data/echomind.sqlite');[print(t[0][:500]) for t in c.execute(\"select text from chunks where doc_id=(select id from documents where filename like '%Formulary%') order by chunk_index limit 5\")]"
```

## When to run

- After any change to `backend/app/rag/` (retrieval, prompts, personas, thresholds)
- After swapping the chat or embedding model
- Before cutting a customer demo

Latency note: the whole suite is serial against one GPU box (~1–5 s/question). The
conversational set is fast; the retrieval sets dominate.

## Voice end-to-end test (no microphone needed)

`voice_e2e_test.py` exercises the full speech loop by synthesizing a spoken question
with Piper and feeding it into the live voice WebSocket as real 20 ms mic frames:
VAD → streaming partials → semantic endpointing → Parakeet final STT → intent routing
→ LLM/RAG → phrase-chunked TTS. It prints what the bot heard, the reply, and the
latencies that matter (asr_final / first reply text / **first audio**) measured from
end-of-speech. Runs inside the voice container:

```bash
docker cp eval/voice_e2e_test.py echomind-voice:/tmp/t.py && docker exec echomind-voice python3 /tmp/t.py
```

Reference numbers on the DGX Spark with Qwen3-30B-A3B-FP4 (TRT-LLM): casual turn
first-audio ≈ **+0.4 s** after end of speech; full RAG turn ≈ **+0.6 s** (lead
phrase) with the 30B answer streaming from ≈ +1.1 s.
