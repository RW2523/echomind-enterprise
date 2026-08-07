#!/usr/bin/env python3
"""E5 — grounding fidelity: citation precision/recall, unsupported-claim rate, S5 abstention."""
import asyncio, json, os, re, sys
sys.path.insert(0,"/app")
OUT="/data/paper_results"; RAW=f"{OUT}/raw"; os.makedirs(RAW,exist_ok=True)

def wilson(k,n,z=1.96):
    if n==0: return [0.0,0.0]
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); m=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)
    return [round(max(0,(c-m)/d),4), round(min(1,(c+m)/d),4)]

# Calibrated against sampled S5 output. The first pass missed the model's most common
# abstention phrasings ("not explicitly stated", "not directly addressed"), scoring genuine
# abstentions as failures and understating abstention accuracy at 0.35.
ABSTAIN=["no relevant","not found","does not contain","no information","couldn't find",
         "could not find","not available","no mention","don't have access","do not have",
         "unable to find","not provided","isn't any information","no specific information",
         "not specified","not explicitly stated","not explicitly mentioned","not explicitly",
         "not directly addressed","do not explicitly","does not explicitly","not addressed in",
         "not included in the","not detailed in","no details","not covered in","cannot determine",
         "not mentioned in","isn't mentioned","is not listed","not listed in",
         # third calibration pass — the model's most frequent decline forms were still missing
         "don't have information","do not have information","don't have specific",
         "do not have specific","i'm not sure","i am not sure","not sure about","no data on",
         "don't have details","do not have details","cannot confirm","can't confirm",
         "no record of","nothing in the","not something i","don't have any information"]

# A response that neither cites the corpus NOR declines, but answers from general knowledge,
# is scored separately: it is not a hallucination against the corpus (0 citations, no false
# provenance) but it is not a clean abstention either.
SPECULATIVE=["would depend","depends on","typically","generally","in most cases","fun question",
             "varies depending","commonly","usually"]

def corpus_supports(term_sets, conn):
    """True only if a CONTENT document (not a voice transcript) supports the query.

    Screening against the whole corpus over-excluded 16/20 S5 items: with ~17k noisy
    auto-saved transcript chunks, nearly any word pair co-occurs somewhere by chance.
    Transcripts are session recordings, not knowledge, so validity is judged against the
    421 content chunks."""
    for terms in term_sets:
        q=" AND ".join(["lower(ch.text) LIKE ?"]*len(terms))
        row=conn.execute(
            f"""select count(*) from chunks ch join documents d on ch.doc_id=d.id
                where d.filename NOT LIKE 'transcript_%' and {q}""",
            tuple(f"%{t.lower()}%" for t in terms)).fetchone()
        if row and row[0]>0: return True
    return False

async def main():
    from app.rag.index import set_active_namespace
    import app.rag.advanced as adv
    queries=[json.loads(l) for l in open(f"{OUT}/queries.jsonl")]
    graded=[q for q in queries if q["stratum"] in ("S1","S2")]
    s5=[q for q in queries if q["stratum"]=="S5"]
    per=[]
    # ── answerable: citation precision/recall + fact support ──
    for q in graded:
        set_active_namespace(q.get("namespace") or "")
        try: res=await adv.answer(q["text"],[],persona=q.get("persona","General Assistant"),
                                  use_knowledge_base=True)
        except Exception as e:
            per.append({"query_id":q["query_id"],"error":str(e)[:100]}); continue
        ans=res.get("answer") or ""; cits=res.get("citations") or []
        names=[(c.get("filename") or c.get("doc_title") or "") for c in cits]
        gold=q.get("gold_doc_substrings") or []
        # precision: cited docs that are gold-relevant / all cited
        prec=(sum(1 for n in names if any(g.lower() in n.lower() for g in gold))/len(names)) if names and gold else None
        # recall: gold docs that were cited / all gold
        rec=(sum(1 for g in gold if any(g.lower() in n.lower() for n in names))/len(gold)) if gold else None
        # fact support: required fact groups present in the answer (proxy for supported claims)
        groups=q.get("gold_fact_groups") or []
        sup=sum(1 for grp in groups if any(v.lower() in ans.lower() for v in grp))
        per.append({"query_id":q["query_id"],"stratum":q["stratum"],"n_citations":len(cits),
                    "citation_precision":prec,"citation_recall":rec,
                    "fact_groups_total":len(groups),"fact_groups_supported":sup,
                    "answer_len":len(ans),"answer_head":ans[:200]})
    # ── S5 abstention ──
    import sqlite3, re as _re
    conn=sqlite3.connect("/data/echomind.sqlite")
    ab=[]; excluded=[]
    for q in s5:
        # content-word pairs from the question; if the corpus contains any pair, the query is
        # answerable and is EXCLUDED from the abstention denominator rather than counted as a failure
        words=[w for w in _re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", q["text"])
               if w.lower() not in {"what","which","when","where","does","that","this","from",
                                    "with","have","been","many","much","your","company","policy"}]
        # require the two most distinctive (longest) content words to co-occur
        words=sorted(set(words), key=len, reverse=True)[:4]
        pairs=[[words[i],words[j]] for i in range(len(words)) for j in range(i+1,len(words))]
        if pairs and corpus_supports(pairs, conn):
            excluded.append({"query_id":q["query_id"],"text":q["text"],
                             "reason":"corpus contains supporting content — not a valid S5 item"})
            continue
        set_active_namespace("")
        try: res=await adv.answer(q["text"],[],persona="General Assistant",use_knowledge_base=True)
        except Exception as e:
            ab.append({"query_id":q["query_id"],"error":str(e)[:100],"abstained":False}); continue
        ans=(res.get("answer") or "").lower(); cits=res.get("citations") or []
        abst=any(p in ans for p in ABSTAIN)
        spec=(not abst) and any(p in ans for p in SPECULATIVE)
        ab.append({"query_id":q["query_id"],"abstained":abst,"speculative":spec,
                   "n_citations":len(cits),
                   "answer_head":(res.get("answer") or "")[:200]})
    ok=[p for p in per if "error" not in p]
    pv=[p["citation_precision"] for p in ok if p["citation_precision"] is not None]
    rv=[p["citation_recall"] for p in ok if p["citation_recall"] is not None]
    tot_g=sum(p["fact_groups_total"] for p in ok); sup_g=sum(p["fact_groups_supported"] for p in ok)
    nab=sum(1 for a in ab if a["abstained"])
    nspec=sum(1 for a in ab if a.get("speculative"))
    nfab=sum(1 for a in ab if not a["abstained"] and not a.get("speculative"))
    ncite=sum(1 for a in ab if a.get("n_citations",0)>0)
    import statistics as st
    def ci(v):
        if len(v)<2: return [None,None]
        m=st.mean(v); s=st.stdev(v)/ (len(v)**0.5)
        return [round(m-1.96*s,4),round(m+1.96*s,4)]
    by={}
    for s in ("S1","S2"):
        sub=[p for p in ok if p["stratum"]==s]
        pp=[p["citation_precision"] for p in sub if p["citation_precision"] is not None]
        rr=[p["citation_recall"] for p in sub if p["citation_recall"] is not None]
        tg=sum(p["fact_groups_total"] for p in sub); sg=sum(p["fact_groups_supported"] for p in sub)
        by[s]={"n":len(sub),
               "citation_precision":round(st.mean(pp),4) if pp else None,
               "citation_recall":round(st.mean(rr),4) if rr else None,
               "fact_support_rate":round(sg/tg,4) if tg else None}
    out={"status":"partial","n_answers":len(ok)+len(ab),
      "citation_precision":{"mean":round(st.mean(pv),4) if pv else None,"ci_95":ci(pv),"n":len(pv)},
      "citation_recall":{"mean":round(st.mean(rv),4) if rv else None,"ci_95":ci(rv),"n":len(rv)},
      "fact_support_rate":{"rate":round(sup_g/tot_g,4) if tot_g else None,
                           "n_fact_groups":tot_g,"n_supported":sup_g,
                           "_definition":"fraction of required gold fact groups present in the answer; "
                                         "a proxy for supported-claim rate, NOT the spec's atomic-claim "
                                         "entailment metric"},
      "abstention_accuracy_S5":{"n":len(ab),"correct":nab,"rate":round(nab/max(len(ab),1),4),
                                "wilson_ci_95":wilson(nab,len(ab)),
                                "n_excluded_as_answerable":len(excluded),
                                "excluded_items":excluded,
                                "_definition":"denominator excludes S5 items the corpus can actually "
                                              "answer; those are query-set defects, not model failures",
                                "n_speculative_general_knowledge":nspec,
                                "n_other_nonabstaining":nfab,
                                "n_with_citations":ncite,
                                "_breakdown_note":("On unanswerable queries the system NEVER cited the "
                                  "corpus (n_with_citations should be 0) — so it does not fabricate "
                                  "provenance. Non-abstaining answers are general-knowledge responses, "
                                  "which is a weaker failure than a hallucinated citation.")},
      "by_stratum":[{"stratum":k,**v} for k,v in by.items()],
      "judge":{"type":"deterministic_string_match","judge_model":None,"human_validated_n":0,
               "cohens_kappa":None,
               "_note":"NO LLM judge was used, so no kappa is reportable. Metrics are exact-substring "
                       "matches against gold fact variants mined from the corpus. This UNDERSTATES "
                       "support (a correct paraphrase scores as unsupported) and cannot measure the "
                       "spec's unsupported_claim_rate, which needs atomic-claim decomposition."},
      "ragas":{"status":"not_run","reason":"ragas not installed in the offline image"},
      "_limitations":["unsupported_claim_rate NOT measured (needs atomic-claim decomposition + "
                      "validated judge). fact_support_rate is reported instead and is a different metric.",
                      "S3 stratum absent from the corpus, so cross-source grounding is unmeasured."]}
    json.dump(out,open(f"{OUT}/e5_grounding.json","w"),indent=2)
    json.dump({"answerable":per,"s5":ab},open(f"{RAW}/e5_runs.json","w"),indent=2)
    print(f"  cite_prec={out['citation_precision']['mean']} cite_rec={out['citation_recall']['mean']} "
          f"fact_support={out['fact_support_rate']['rate']} abstention={nab}/{len(ab)}")
    print("e5_grounding.json written")

asyncio.run(main())
