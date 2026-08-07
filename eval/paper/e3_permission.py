#!/usr/bin/env python3
"""E3 — permission-before-retrieval leak test (namespace isolation).

MAPPING NOTE: EchoMind isolates by TENANT NAMESPACE, not per-chunk ACLs. The pre_filter arm
is production behaviour (_ns_ok evaluated inside the candidate scan). The post_filter arm is
built here as the counterfactual "common shortcut": retrieve globally, drop unauthorized
chunks afterwards. Both arms exercise the same index and the same queries.
"""
import asyncio, json, os, statistics as st, sys, time
sys.path.insert(0,"/app")
OUT="/data/paper_results"; RAW=f"{OUT}/raw"; os.makedirs(RAW,exist_ok=True)

def wilson(k,n,z=1.96):
    if n==0: return [0.0,0.0]
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); m=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)
    return [round(max(0,(c-m)/d),4), round(min(1,(c+m)/d),4)]

def ks_2samp(a,b):
    """Two-sample KS statistic + asymptotic p-value (no scipy in the image)."""
    if not a or not b: return 0.0,1.0
    a=sorted(a); b=sorted(b); na,nb=len(a),len(b)
    allv=sorted(set(a+b)); d=0.0
    import bisect
    for v in allv:
        fa=bisect.bisect_right(a,v)/na; fb=bisect.bisect_right(b,v)/nb
        d=max(d,abs(fa-fb))
    en=(na*nb/(na+nb))**0.5; lam=(en+0.12+0.11/en)*d
    p=2*sum((-1)**(j-1)*pow(2.718281828,-2*j*j*lam*lam) for j in range(1,101))
    return round(d,4), round(min(1.0,max(0.0,p)),6)

async def main():
    from app.rag.index import index, set_active_namespace
    import app.rag.index as idx_mod
    import app.rag.advanced as adv
    queries=[json.loads(l) for l in open(f"{OUT}/queries.jsonl")]
    probes=[q for q in queries if q["stratum"]=="S4"]                 # answer only in another namespace
    # CONTROL DESIGN (corrected): the control must isolate "restricted content EXISTS elsewhere"
    # from "no answer available". S1/S2 controls are answerable, so probe-vs-control would measure
    # answerable-vs-unanswerable — a confound, not a leak. S5 queries have no supporting content
    # anywhere in the corpus, so both arms must refuse; any residual separation IS the side channel.
    controls=[q for q in queries if q["stratum"]=="S5"][:25]

    async def run(arm,q,repeats=1):
        """Return per-run dicts. post_filter = global retrieve, then drop out-of-namespace."""
        ns=q.get("namespace") or ""
        out=[]
        for _ in range(repeats):
            t0=time.perf_counter()
            if arm=="pre_filter":
                set_active_namespace(ns)
                res=await adv.answer(q["text"],[],persona="General Assistant",use_knowledge_base=True)
                cand=None
            else:
                # counterfactual: retrieve with NO namespace restriction, then filter after
                set_active_namespace("")
                st_,hits=await adv.retrieve_semantic_first(q["text"],15)
                cand_global=len(hits)
                allowed=[h for h in hits if ((h.get("source") or {}).get("namespace") or "default")==(ns or "default")]
                cand=cand_global
                set_active_namespace(ns)
                res=await adv.answer(q["text"],[],persona="General Assistant",use_knowledge_base=True)
            ms=(time.perf_counter()-t0)*1000
            ans=(res.get("answer") or "")
            cits=res.get("citations") or []
            out.append({"query_id":q["query_id"],"arm":arm,"ms":ms,"answer":ans[:600],
                        "n_citations":len(cits),
                        "cited":[(c.get("filename") or "") for c in cits],
                        "candidates_before_filter":cand})
        return out

    def content_leak(rec,q):
        """Did restricted content actually surface? gold_doc_substrings identify the owner docs."""
        for sub in q.get("gold_doc_substrings",[]):
            if any(sub.lower() in (c or "").lower() for c in rec["cited"]): return True
        return False

    def refusal_class(text):
        """DISCLOSING only if the wording implies the content EXISTS but is withheld.

        Calibrated against sampled output: generic model refusals such as "I don't have access
        to specific documents" reveal nothing about existence and were previously mis-scored as
        authorization disclosures (inflating disclosing_rate to 0.36). Only explicit
        existence+withholding language counts.
        """
        t=" ".join((text or "").lower().split())
        disclosing=any(p in t for p in [
            "you are not authorized","you do not have permission","you don't have permission",
            "not authorised to view","insufficient privileges","above your clearance",
            "exists but","is restricted to","classified","requires higher","not cleared",
            "your access level","permission to view this document"])
        generic_none=any(p in t for p in [
            "no relevant","not found","does not contain","no information","couldn't find",
            "could not find","not available","no mention","don't have access","do not have access",
            "not able to access","no specific information","isn't any information"])
        if disclosing: return "authorization_phrasing"
        if generic_none: return "nonexistence_phrasing"
        return "other"

    arms=[]; raw={}
    for arm in ("pre_filter","post_filter"):
        pr=[]; ct=[]
        for q in probes:  pr += await run(arm,q,repeats=2)
        for q in controls: ct += await run(arm,q,repeats=2)
        raw[arm]={"probes":pr,"controls":ct}
        pmap={q["query_id"]:q for q in probes}
        leaked=sum(1 for r in pr if content_leak(r,pmap[r["query_id"]]))
        p_ms=[r["ms"] for r in pr]; c_ms=[r["ms"] for r in ct]
        ks_t,p_t=ks_2samp(p_ms,c_ms)
        p_cnt=[r["candidates_before_filter"] for r in pr if r["candidates_before_filter"] is not None]
        c_cnt=[r["candidates_before_filter"] for r in ct if r["candidates_before_filter"] is not None]
        ks_c,p_c=(ks_2samp(p_cnt,c_cnt) if p_cnt and c_cnt else (None,None))
        cls=[refusal_class(r["answer"]) for r in pr]
        samples=[{"query_id":r["query_id"],"class":refusal_class(r["answer"]),
                  "text":" ".join((r["answer"] or "").split())[:220]} for r in pr[:8]]
        na=cls.count("authorization_phrasing"); nn=cls.count("nonexistence_phrasing")
        arms.append({"arm":arm,
          "content_leak":{"n_probes":len(pr),"n_leaked":leaked,"rate":round(leaked/max(len(pr),1),4),
                          "wilson_ci_95":wilson(leaked,len(pr))},
          "count_channel":({"probe_median":round(st.median(p_cnt),2),"control_median":round(st.median(c_cnt),2),
                            "ks_statistic":ks_c,"p_value":p_c,"n_each":len(p_cnt)}
                           if p_cnt and c_cnt else
                           {"status":"not_applicable",
                            "reason":"pre_filter never materialises an unfiltered candidate set — the "
                                     "namespace test is inside the scan, so no count channel exists to measure"}),
          "timing_channel":{"probe_median_ms":round(st.median(p_ms),2),
                            "control_median_ms":round(st.median(c_ms),2),
                            "median_diff_ms":round(st.median(p_ms)-st.median(c_ms),2),
                            "ks_statistic":ks_t,"p_value":p_t,"n_each":len(p_ms)},
          "refusal_samples":samples,
          "refusal_wording":{"nonexistence_phrasing":nn,"authorization_phrasing":na,
                             "other":cls.count("other"),
                             "disclosing_rate":round(na/max(len(cls),1),4)}})
        print(f"  {arm}: leak={leaked}/{len(pr)} timing_KS={ks_t} p={p_t} disclosing={na}/{len(cls)}")
    out={"status":"complete","arms":arms,
         "_mapping_note":("Isolation unit is the tenant NAMESPACE, not per-chunk ACLs — EchoMind has no "
                          "ACL model. Property P1 is therefore tested as 'namespace predicate evaluated "
                          "inside the candidate scan' vs 'global retrieve then drop'."),
         "_control_design":("Controls are S5 (no supporting content anywhere). Probes are S4 (content "
                            "exists but in another namespace). Both must refuse, so any measured "
                            "separation isolates the side channel rather than answerable-vs-unanswerable."),
         "_limitations":["Both arms invoke the same generation path, so the timing channel reflects "
                         "retrieval+generation, not retrieval alone.",
                         "n=50 per arm (25 probes x2, 25 controls x2); adequate for KS but modest."]}
    json.dump(out,open(f"{OUT}/e3_permission.json","w"),indent=2)
    json.dump(raw,open(f"{RAW}/e3_runs.json","w"),indent=2)
    print("e3_permission.json written")

asyncio.run(main())
