#!/usr/bin/env python3
"""Retrieval measured on its own, stage by stage (2026-11-07, external review point 2), on the demo's chat set.

    python3 eval/law/run_retrieval_eval.py --container echomind-legal-next-api --label baseline \
        [--chat eval/law/reports/<time>-chat-<label>.json] [--ce-model BAAI/bge-reranker-large --floor -4]

Run against a PRIVATE copy of the demo (its API container): the script copies itself into the container and runs there,
with the backend's code and data, then writes eval/law/reports/<time>-retrieval-<label>.json and .md on the host.

Relevance comes from the answer keys (eval/fixtures/demo/answer_key_*.jsonl, or --key-files: since 2026-11-12 also the synthetic
set in eval/fixtures/synthetic), whose `evidence` is the documents' exact
words, checked against each document when the set was made: an evidence segment (pieces joined by " | ") is found in a
passage of its own document (`source`, and the `also` entries) when the passage holds every "…"-piece of it, or, for a
segment the chunking splits, its longest piece. Each item is searched the way research searches it (the question with the
parties' roles, LEGAL_RESEARCH_PASSAGES passages), and every stage is scored:

- **dense, sparse, keyword:** each channel's own ranking; **fused:** after reciprocal-rank fusion; **reranked:** the
  cross-encoder's order before the floor; **final:** the passages search returns;
- Recall@5 and @10 (the share of an item's evidence segments found in the top k), MRR@10 and nDCG@10 (binary relevance);
- **context recall:** the share found anywhere in what the answer model is given (research's whole context: the named
  documents, parties, clauses, amendments and the rest), read up to the `sources` event; no answer is generated.

With --chat (a run_chat_keys.py report), each failed item is put at the first stage that lost its evidence: not
retrieved, ranked out by the reranker, dropped before the context, or in the context and still answered wrong or
incomplete (the answer, or the answer key's scoring). Follow-up items are left out: their last question needs the
conversation. Nothing is written to the copy's records: the research turn and its audit event are not saved.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
STAGES = ("dense", "sparse", "keyword", "fused", "reranked", "floored", "final", "hybrid", "hybrid_sparse")


def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    """Reciprocal-rank fusion of passage-id rankings; an id missing from a ranking counts as ranked last there."""
    pool = list(dict.fromkeys(i for r in rankings for i in r))
    score = {i: sum(1 / (k + (r.index(i) if i in r else len(r)) + 1) for r in rankings) for i in pool}
    return sorted(pool, key=lambda i: -score[i])


# ── Relevance ────────────────────────────────────────────────────────────────────
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("**", "").replace("\\", "")
    return " ".join(s.lower().split())


def segments(item: dict) -> list[tuple[str, list[str]]]:
    """(document title, pieces) for every evidence segment of an item."""
    out = []
    for src in [{"source": item.get("source"), "evidence": item.get("evidence")}, *(item.get("also") or [])]:
        for seg in (src.get("evidence") or "").split(" | "):
            pieces = [norm(p) for p in re.split(r"\s*(?:…|\.\.\.)\s*", seg) if norm(p)]
            if pieces:
                out.append((src.get("source") or "", pieces))
    return out


def covers(text: str, pieces: list[str], loose: bool) -> bool:
    t = norm(text)
    return max(pieces, key=len) in t if loose else all(p in t for p in pieces)


# ── Inside the container ─────────────────────────────────────────────────────────
def inside(args) -> int:
    env = dict(x.split("=", 1) for x in Path("/proc/1/environ").read_bytes().decode().split("\0") if "=" in x)
    os.environ.update({k: v for k, v in env.items() if k not in ("HOSTNAME",)})
    if args.ce_model:
        os.environ["RAG_CE_MODEL"] = args.ce_model
    sys.path.insert(0, "/repo/backend")
    import logging
    logging.disable(logging.WARNING)
    from app.core.config import settings
    if args.floor is not None:
        settings.LEGAL_RERANK_MIN_SCORE = args.floor
    if args.ce_model:
        settings.RAG_CE_MODEL = args.ce_model
    from app.core.db import get_conn
    from app.legal.documents.versions import policy_sql
    from app.legal.research import attribution
    from app.legal.research import routes as R
    from app.legal.retrieval import scope as scopemod
    from app.legal.retrieval import search as searchmod
    R._save = lambda *a, **k: None                          # nothing written to the copy's records
    R.record = lambda *a, **k: None
    timing = {"rerank": []}
    real_rerank = searchmod.rerank

    async def timed(query, hits, top_k):
        t0 = time.perf_counter()
        out = await real_rerank(query, hits, top_k)
        timing["rerank"].append(time.perf_counter() - t0)
        return out
    searchmod.rerank = timed

    items = [json.loads(line) for line in Path(args.keys).read_text().splitlines() if line.strip()]
    if args.limit:
        items = [i for i in items if not i.get("turns")][:args.limit]
    with get_conn() as conn:
        uid = conn.execute("SELECT id FROM users WHERE username='alice'").fetchone()[0]
        matters = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT number, id, org_id FROM matters")}
    texts: dict[str, tuple[str, str]] = {}

    def chunk_texts(ids):
        need = [i for i in ids if i not in texts]
        if need:
            with get_conn() as conn:
                for r in conn.execute("SELECT c.id, c.text, d.title FROM legal_chunks c JOIN legal_documents d ON d.id=c.document_id "
                                      f"WHERE c.id IN ({','.join('?' * len(need))})", need):
                    texts[r[0]] = (r[1], r[2])
        return texts

    out = []
    asyncio.run(_items(items, matters, uid, out, chunk_texts, settings, get_conn, policy_sql, attribution, R, scopemod, searchmod,
                       args.search_only))
    rr = sorted(timing["rerank"])
    Path(args.out).write_text(json.dumps({"items": out, "rerank_s": {"n": len(rr), "p50": rr[len(rr) // 2] if rr else None,
                                                                      "p95": rr[int(len(rr) * .95)] if rr else None},
                                          "ce_model": settings.RAG_CE_MODEL, "floor": settings.LEGAL_RERANK_MIN_SCORE}))
    return 0


async def _items(items, matters, uid, out, chunk_texts, settings, get_conn, policy_sql, attribution, R, scopemod, searchmod,
                 search_only=False):
    """One event loop for the whole run (the embedding client keeps its connection on it)."""
    for it in items:
        if it.get("turns") or it["matter"] not in matters:
            continue
        mid, org = matters[it["matter"]]
        segs = segments(it)
        with get_conn() as conn:                            # every passage the signed-version policy can return
            rows = conn.execute("SELECT c.id, c.text, d.title FROM legal_chunks c JOIN legal_documents d ON d.id=c.document_id "
                                "JOIN legal_document_versions v ON v.id=c.version_id "
                                f"WHERE c.matter_id=? AND c.active=1 AND {policy_sql('signed')}", (mid,)).fetchall()
        relevant: list[set] = []
        for title, pieces in segs:
            own = [r for r in rows if norm(r[2]) == norm(title)] or [r for r in rows if norm(title) in norm(r[2])] or rows
            hit = {r[0] for r in own if covers(r[1], pieces, False)} or {r[0] for r in own if covers(r[1], pieces, True)}
            relevant.append(hit)
        located = [s for s in relevant if s]
        principal = {"sub": uid, "org_id": org, "auth": "session"}
        scope = scopemod.authorize(principal, mid)
        question = it["question"]
        heads = R._heads([scope], uid)
        roles = {did: h[1] for did, h in heads.items() if h[1]}
        search_q = attribution.with_roles(question, roles)
        diag: dict = {}
        res = await searchmod.search(scope, search_q, k=settings.LEGAL_RESEARCH_PASSAGES, wide=True, diagnostics=diag)
        union = set().union(*located) if located else set()
        ce = [c for c, _ in diag.get("reranked", [])]
        diag["floored"] = [c for c, sc in diag.get("reranked", []) if sc >= settings.LEGAL_RERANK_MIN_SCORE]
        diag["hybrid"] = rrf([c for c in diag.get("fused", []) if c in set(ce)], ce)          # the reranked passages, both orders
        diag["hybrid_sparse"] = rrf([c for c in diag.get("sparse", []) if c in set(ce)], ce)
        stages = {}
        for st in STAGES:
            ranking = [c[0] if isinstance(c, (list, tuple)) else c for c in diag.get(st, [])]
            first = next((k for k, c in enumerate(ranking) if c in union), None)
            rec = {k: (sum(1 for s in located if s & set(ranking[:k])) / len(located)) if located else None for k in (5, 10)}
            dcg = sum(1 / math.log2(k + 2) for k, c in enumerate(ranking[:10]) if c in union)
            idcg = sum(1 / math.log2(k + 2) for k in range(min(len(union), 10)))
            stages[st] = {"recall@5": rec[5], "recall@10": rec[10], "mrr@10": (1 / (first + 1) if first is not None and first < 10 else 0.0)
                          if located else None, "ndcg@10": (dcg / idcg if idcg else None), "first": first,
                          "found": (sum(1 for s in located if s & set(ranking)) / len(located)) if located else None}
        final_texts = [p["text"] for p in res["passages"]]
        context = []                                        # what the answer model is given, read up to the sources event
        if search_only:
            out.append({"id": it["id"], "matter": it["matter"], "kind": it.get("kind"), "segments": len(segs), "located": len(located),
                        "stages": stages, "final_recall": None, "context_recall": None, "context_sources": 0, "context_chars": 0})
            continue

        async def read_context():
            turn = {"id": f"eval-{it['id']}", "org_id": org, "matter_id": mid, "owner_id": uid, "question": question,
                    "auth": "session", "created_at": "2026-11-07T00:00:00.000Z"}
            gen = R._answer(scope, principal, question, turn)
            try:
                async for line in gen:
                    e = json.loads(line)
                    if e.get("type") == "sources":
                        context.extend((s["document_title"], s["text"]) for s in e["citations"])
                        break
                    if e.get("type") in ("done", "error"):
                        break
            finally:
                await gen.aclose()
        await read_context()

        def found_in(passages):
            ok = []
            for (title, pieces), rel in zip(segs, relevant):
                if not rel:
                    continue
                ok.append(any(covers(t, pieces, False) or covers(t, pieces, True) for _, t in passages))
            return (sum(ok) / len(ok)) if ok else None
        ctx = found_in(context)
        fin = found_in([("", t) for t in final_texts])
        out.append({"id": it["id"], "matter": it["matter"], "kind": it.get("kind"), "segments": len(segs), "located": len(located),
                    "stages": stages, "final_recall": fin, "context_recall": ctx, "context_sources": len(context),
                    "context_chars": sum(len(t) for _, t in context)})
        print(f"{it['id']:16} {it.get('kind', ''):11} located {len(located)}/{len(segs)} final {fin} context {ctx} "
              f"({len(context)} sources)", flush=True)


# ── On the host ──────────────────────────────────────────────────────────────────
def lost_at(r: dict) -> str:
    if not r["located"]:
        return "evidence not located in the documents"
    if (r["context_recall"] or 0) >= 1:
        return "in context: answered wrong or incomplete"
    st = r["stages"]
    if (r["context_recall"] or 0) > 0:
        return "partly in context"
    if (st["reranked"]["found"] or 0) > 0:
        return "ranked out by the reranker (below the top passages or the floor)"
    if any((st[c]["found"] or 0) > 0 for c in ("dense", "sparse", "keyword", "fused")):
        return "retrieved by a channel, lost before reranking"
    return "not retrieved"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def host(args) -> int:
    keys = sorted({p for g in args.key_files.split(",") if g for p in ROOT.glob(g)}) if args.key_files \
        else sorted((ROOT / "eval" / "fixtures" / "demo").glob("answer_key_*.jsonl"))
    if not keys:
        print(f"no answer keys match {args.key_files}", file=sys.stderr)
        return 2
    tmp = "/tmp/reval"
    subprocess.run(["docker", "exec", args.container, "mkdir", "-p", tmp], check=True)
    joined = Path(f"/tmp/reval-{os.getpid()}.jsonl")
    joined.write_text("".join(k.read_text() for k in keys))
    try:
        for src, dst in ((Path(__file__), f"{tmp}/reval.py"), (joined, f"{tmp}/keys.jsonl")):
            subprocess.run(["docker", "cp", str(src), f"{args.container}:{dst}"], check=True)
    finally:
        joined.unlink(missing_ok=True)
    cmd = ["docker", "exec", "-w", "/repo/backend", args.container, "python", f"{tmp}/reval.py", "--inside", "--keys", f"{tmp}/keys.jsonl",
           "--out", f"{tmp}/out.json"] + (["--ce-model", args.ce_model] if args.ce_model else []) \
        + (["--floor", str(args.floor)] if args.floor is not None else []) + (["--search-only"] if args.search_only else []) \
        + (["--limit", str(args.limit)] if args.limit else [])
    if subprocess.run(cmd).returncode:
        return 1
    raw = json.loads(subprocess.run(["docker", "exec", args.container, "cat", f"{tmp}/out.json"], capture_output=True, text=True,
                                    check=True).stdout)
    subprocess.run(["docker", "exec", args.container, "rm", "-rf", tmp])
    rows = raw["items"]
    measured = [r for r in rows if r["located"]]
    table = {st: {m: mean(r["stages"][st][m] for r in measured) for m in ("recall@5", "recall@10", "mrr@10", "ndcg@10")}
             for st in STAGES}
    summary = {"items": len(rows), "measured": len(measured), "stages": table, "final_recall": mean(r["final_recall"] for r in measured),
               "context_recall": mean(r["context_recall"] for r in measured),
               "context_sources_mean": mean(r["context_sources"] for r in rows), "context_chars_mean": mean(r["context_chars"] for r in rows),
               "rerank_s": raw["rerank_s"], "ce_model": raw["ce_model"], "floor": raw["floor"]}
    by_kind = {}
    for r in measured:
        by_kind.setdefault(r["kind"], []).append(r)
    summary["by_kind"] = {k: {"n": len(v), "final_recall": mean(r["final_recall"] for r in v),
                              "context_recall": mean(r["context_recall"] for r in v)} for k, v in sorted(by_kind.items())}
    attribution = {}
    if args.chat:
        chat = {r["id"]: r for r in json.loads(Path(args.chat).read_text())["items"]}
        by_id = {r["id"]: r for r in rows}
        for cid, c in chat.items():
            if c["ok"]:
                continue
            label = "follow-up (not measured)" if cid not in by_id else lost_at(by_id[cid])
            attribution.setdefault(label, []).append(cid)
        summary["failed_by_stage"] = {k: len(v) for k, v in attribution.items()}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = HERE / "reports" / f"{stamp}-retrieval-{args.label}"
    out.with_suffix(".json").write_text(json.dumps({"suite": "retrieval", "label": args.label, "commit": commit, "chat": args.chat or None,
                                                    "summary": summary, "failed_by_stage": attribution, "items": rows}, indent=1))
    f = lambda x: "–" if x is None else f"{x:.3f}"
    lines = [f"# Retrieval, stage by stage ({args.label})", "",
             f"Commit {commit}; reranker {raw['ce_model']} (floor {raw['floor']}); {len(measured)} of {len(rows)} single-turn items "
             f"with located evidence. Rerank p50 {f(raw['rerank_s']['p50'])} s, p95 {f(raw['rerank_s']['p95'])} s per search.", "",
             "| Stage | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |", "|---|---|---|---|---|"]
    lines += [f"| {st} | {f(t['recall@5'])} | {f(t['recall@10'])} | {f(t['mrr@10'])} | {f(t['ndcg@10'])} |" for st, t in table.items()]
    lines += ["", f"Search's final passages hold {f(summary['final_recall'])} of the evidence; the answer model's whole context holds "
                  f"{f(summary['context_recall'])} ({summary['context_sources_mean']} sources, {summary['context_chars_mean']} characters "
                  "on average).", "", "| Kind | Items | Final recall | Context recall |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {f(v['final_recall'])} | {f(v['context_recall'])} |" for k, v in summary["by_kind"].items()]
    if attribution:
        lines += ["", f"Failed items of {Path(args.chat).name}, by the first stage that lost the evidence:", "", "| Stage | Items |", "|---|---|"]
        lines += [f"| {k} | {len(v)}: {', '.join(v)} |" for k, v in sorted(attribution.items(), key=lambda kv: -len(kv[1]))]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", default="echomind-legal-next-api", help="a private copy's API container")
    ap.add_argument("--label", default="run")
    ap.add_argument("--chat", default="", help="a run_chat_keys.py report: put each failed item at the stage that lost it")
    ap.add_argument("--ce-model", default="", help="another cross-encoder from the local cache (benchmark)")
    ap.add_argument("--floor", type=float, default=None, help="the reranker floor for that model")
    ap.add_argument("--search-only", action="store_true", help="the search stages only, not the answer model's context (fast)")
    ap.add_argument("--limit", type=int, default=0, help="the first N single-turn items only (a slow reranker's benchmark)")
    ap.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--key-files", default="", help="other answer keys: comma-separated globs from the repository root; "
                                                    "default the demo's")
    ap.add_argument("--keys", default="", help=argparse.SUPPRESS)
    ap.add_argument("--out", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.inside:
        return inside(args)
    if args.container == "echomind-legal-demo-api":
        print("refusing: run against a private copy, not the public demo", file=sys.stderr)
        return 2
    return host(args)


if __name__ == "__main__":
    sys.exit(main())
