#!/usr/bin/env python3
"""E3b — per-PATH namespace isolation audit.

Motivation: E3 measured leakage at the ANSWER level using document-targeted probes and found
0/50. That test never exercised the sparse transcript path, which was later found to bypass the
namespace predicate entirely. This experiment audits every retrieval path directly.
"""
import asyncio, json, os, sys
sys.path.insert(0,"/app")
OUT="/data/paper_results"
def wilson(k,n,z=1.96):
    if n==0: return [0.0,0.0]
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); m=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)
    return [round(max(0,(c-m)/d),4), round(min(1,(c+m)/d),4)]

async def main():
    from app.rag.index import index, set_active_namespace
    QUERIES=["What do I need to qualify for a Crest Personal Loan?",
             "What is the first-line medication for stage 1 hypertension?",
             "What is the liability cap in the Master Services Agreement?",
             "What is the restocking fee for returns?",
             "How are travel expenses reimbursed?",
             "car tesla motel problem paradigm",           # transcript-flavoured terms
             "hello sir nice to meet you today",
             "what do you do for a living"]
    NS=["bank","health","law","meetings","retail"]
    paths={
      "dense_document":   lambda q,k: index.search_document_only(q,k),
      "sparse_document":  lambda q,k: asyncio.sleep(0,result=index.search_document_only_sparse(q,k)),
      "dense_transcript": lambda q,k: index.search_transcript_only(q,k),
      "sparse_transcript":lambda q,k: asyncio.sleep(0,result=index.search_transcript_only_sparse(q,k)),
      "keyword_grep":     lambda q,k: asyncio.sleep(0,result=index.search_document_keyword_grep(q,k)),
    }
    results=[]
    for name,fn in paths.items():
        checked=leaked=0; samples=[]
        for ns in NS:
            set_active_namespace(ns)
            for q in QUERIES:
                try: hits=await fn(q,10)
                except Exception as e:
                    samples.append({"error":str(e)[:80]}); continue
                for h in hits or []:
                    checked+=1
                    hns=(h.get("source") or {}).get("namespace") or "default"
                    if hns!=ns:
                        leaked+=1
                        if len(samples)<5:
                            samples.append({"query_ns":ns,"hit_ns":hns,
                                            "text":(h.get("text") or "")[:70]})
        results.append({"path":name,"n_hits_checked":checked,"n_out_of_namespace":leaked,
                        "leak_rate":round(leaked/checked,4) if checked else None,
                        "wilson_ci_95":wilson(leaked,checked) if checked else None,
                        "leak_samples":samples})
        print(f"  {name:18s} checked={checked:5d} leaked={leaked:4d}")
    out={"status":"complete","paths":results,
      "_purpose":("Audits property P1 at the level of individual retrieval paths rather than at the "
                  "answer level. The answer-level test (E3) reported 0/50 leakage, but its probes "
                  "only exercised document paths."),
      "_history":("Before the fix applied in this session, sparse_transcript returned 10/10 "
                  "out-of-namespace hits under a bank-scoped query (chunks tagged 'default' and "
                  "null): advanced.py called Bm25Index.search directly, bypassing _ns_ok, while the "
                  "other three paths filtered correctly. The breach ALSO degraded quality — leaked "
                  "transcript hits outranked in-namespace documents and won source selection, "
                  "silently zeroing citations on vertical queries (golden eval 50/52 -> 42/52). It "
                  "surfaced only because the corpus grew 10.3k -> 17.6k chunks; at the original size "
                  "the leaked hits did not outrank real ones. Numbers above are POST-fix."),
      "_finding_for_paper":("An architecturally-stated invariant (permission-before-retrieval) held "
                            "in 3 of 4 retrieval paths and was violated in the 4th at the "
                            "implementation level. Neither the architecture nor an answer-level "
                            "evaluation detected it. This argues for path-level invariant testing, "
                            "which is §10 open question 2.")}
    json.dump(out,open(f"{OUT}/e3b_path_isolation.json","w"),indent=2)
    print("e3b_path_isolation.json written")

asyncio.run(main())
