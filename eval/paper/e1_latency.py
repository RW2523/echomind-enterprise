#!/usr/bin/env python3
"""E1 — latency decomposition. Runs INSIDE echomind-backend.

Stage timings come from instrumenting the real retrieval functions (monkey-patched wrappers
around the production code paths, not reimplementations). End-to-end T_grounded is ALSO
measured through the HTTP API so it includes serialization and queueing.
"""
import asyncio, json, os, statistics as st, sys, time
sys.path.insert(0,"/app")
OUT="/data/paper_results"; RAW=f"{OUT}/raw"
os.makedirs(RAW,exist_ok=True)
import contextvars
# Per-TASK span store. A module-global dict is contaminated by concurrent tasks: with
# concurrency 16 every task wrote into the same buckets, which produced a
# retrieval_share_of_T_grounded of 1.61 (impossible). contextvars gives each asyncio
# task its own store, so concurrent cells measure per-request cost correctly.
_SPANS: contextvars.ContextVar = contextvars.ContextVar("spans", default=None)

def _store():
    d=_SPANS.get()
    if d is None:
        d={}; _SPANS.set(d)
    return d

def rec(name,ms): _store().setdefault(name,[]).append(ms)

def instrument():
    """Wrap the real functions so we time production code, not a copy of it."""
    from app.rag import index as idx_mod
    from app.rag.index import index
    import app.rag.advanced as adv

    # ---- permission filter: time _ns_ok separately from the scan it sits inside ----
    orig_ns_ok = idx_mod._ns_ok
    def timed_ns_ok(src):
        t=time.perf_counter(); r=orig_ns_ok(src)
        _store().setdefault("_ns_accum",[0.0])[0]+=(time.perf_counter()-t)*1000
        return r
    idx_mod._ns_ok = timed_ns_ok
    adv._ns_ok = timed_ns_ok

    # ---- dense vector search ----
    orig_dense = index.search_document_only
    async def timed_dense(*a,**k):
        _store()["_ns_accum"]=[0.0]; t=time.perf_counter()
        r=await orig_dense(*a,**k)
        acc=_store().get("_ns_accum",[0.0])[0]
        rec("retrieval.vector_search",max(0.0,(time.perf_counter()-t)*1000 - acc))
        rec("retrieval.permission_filter",acc)
        return r
    index.search_document_only = timed_dense

    # ---- sparse ----
    orig_sparse = index.search_document_only_sparse
    def timed_sparse(*a,**k):
        t=time.perf_counter(); r=orig_sparse(*a,**k)
        rec("retrieval.sparse_search",(time.perf_counter()-t)*1000); return r
    index.search_document_only_sparse = timed_sparse

    # ---- embedding ----
    orig_embed = index.emb.embed
    async def timed_embed(texts):
        t=time.perf_counter(); r=await orig_embed(texts)
        rec("query.embed",(time.perf_counter()-t)*1000); return r
    index.emb.embed = timed_embed

    # ---- rerank ----
    from app.rag import reranker as rr
    orig_rr = rr.rerank_hits
    async def timed_rr(*a,**k):
        t=time.perf_counter(); r=await orig_rr(*a,**k)
        rec("retrieval.rerank",(time.perf_counter()-t)*1000); return r
    rr.rerank_hits = timed_rr
    adv._ce_rerank_hits = timed_rr

    # ---- context assembly ----
    orig_ctx = adv._build_rag_context
    async def timed_ctx(*a,**k):
        t=time.perf_counter(); r=await orig_ctx(*a,**k)
        rec("context.assemble",(time.perf_counter()-t)*1000); return r
    adv._build_rag_context = timed_ctx

def pct(v,p):
    if not v: return None
    v=sorted(v); import math
    i=min(len(v)-1,max(0,int(math.ceil(p/100*len(v)))-1)); return round(v[i],2)

async def run_cell(queries, cache_state, concurrency, n_runs):
    from app.rag.index import set_active_namespace
    import app.rag.advanced as adv
    per_run=[]
    async def one(q):
        _SPANS.set({})              # fresh per-task store
        set_active_namespace(q.get("namespace") or "")
        t0=time.perf_counter()
        try:
            res = await adv.answer(q["text"], [], persona=q.get("persona","General Assistant"),
                                   use_knowledge_base=True)
        except Exception as e:
            return {"query_id":q["query_id"],"error":str(e)[:120]}
        total=(time.perf_counter()-t0)*1000
        snap={k:sum(v) for k,v in _store().items() if not k.startswith("_")}
        snap["T_grounded"]=total
        snap["_n_citations"]=len(res.get("citations") or [])
        snap["query_id"]=q["query_id"]; snap["stratum"]=q["stratum"]
        return snap
    sel=[queries[i%len(queries)] for i in range(n_runs)]
    if concurrency==1:
        for q in sel: per_run.append(await one(q))
    else:
        for i in range(0,len(sel),concurrency):
            per_run.extend(await asyncio.gather(*[one(q) for q in sel[i:i+concurrency]]))
    ok=[r for r in per_run if "error" not in r]
    stages={}
    for name in ["query.embed","retrieval.permission_filter","retrieval.vector_search",
                 "retrieval.sparse_search","retrieval.rerank","context.assemble","T_grounded"]:
        vals=[r[name] for r in ok if r.get(name) is not None]
        if vals: stages[name]={"median_ms":round(st.median(vals),2),
                               "p95_ms":pct(vals,95),"p99_ms":pct(vals,99),"n":len(vals)}
    med=lambda k: stages.get(k,{}).get("median_ms",0) or 0
    retr = med("retrieval.permission_filter")+med("retrieval.vector_search")+ \
           med("retrieval.sparse_search")+med("retrieval.rerank")+med("context.assemble")
    share = round(retr/med("T_grounded"),4) if med("T_grounded") else None
    share_warning = None
    if share is not None and share > 1.0:
        share_warning = ("component spans exceed T_grounded — spans are contended/overlapping at this "
                         "concurrency; treat the decomposition as unreliable for this cell")
    return {"cache_state":cache_state,"concurrency":concurrency,"n":len(ok),
            "n_errors":len(per_run)-len(ok),"stages":stages,
            "retrieval_share_of_T_grounded":share,
            "share_warning":share_warning,
            "notes":"T_grounded measured in-process (no HTTP); see e1_api_endtoend for the "
                    "API-level figure that includes serialization and queueing."}, per_run

async def main():
    queries=[json.loads(l) for l in open(f"{OUT}/queries.jsonl")]
    answerable=[q for q in queries if q["stratum"] in ("S1","S2")]
    instrument()
    cells=[]; raw={}
    # warm-up
    for q in answerable[:5]:
        try: await run_cell([q],"warmup",1,1)
        except Exception: pass
    for conc in (1,4,16):
        cell,per=await run_cell(answerable,"warm",conc,30)
        cells.append(cell); raw[f"warm_c{conc}"]=per
        print(f"  warm c={conc}: T_grounded median {cell['stages'].get('T_grounded',{}).get('median_ms')}ms "
              f"retrieval_share={cell['retrieval_share_of_T_grounded']}")
    out={"status":"complete","n_runs_per_cell":30,"cells":cells,
         "_limitations":["Cold-cache cells NOT run: dropping the OS page cache requires root on the host, "
                         "which the harness does not have. All cells are warm. Reported as a gap rather "
                         "than relabelling warm runs as cold.",
                         "T_first / interaction.first_audio / bargein.acoustic_stop are voice-loop spans, "
                         "measured separately in E7 — they are not part of the text RAG path timed here.",
                         "tts.first_audio and asr.* likewise belong to the voice service (E7)."]}
    json.dump(out,open(f"{OUT}/e1_latency.json","w"),indent=2)
    json.dump(raw,open(f"{RAW}/e1_spans.json","w"),indent=2)
    print("e1_latency.json written")

asyncio.run(main())
