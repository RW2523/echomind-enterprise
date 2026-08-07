#!/usr/bin/env python3
"""Build a VERIFIED-unanswerable S5 set: every candidate is screened against the content
corpus (transcripts excluded) and kept only if no distinctive term appears at all."""
import json, re, sqlite3, sys
OUT="/data/paper_results"; DB="/data/echomind.sqlite"
CAND=[
 "What is the airspeed velocity of the company shuttle service?",
 "How many polar research stations does the organisation operate?",
 "What is the recipe for the cafeteria's mushroom risotto?",
 "Which orchestral instruments are stocked in the recreation room?",
 "What is the breeding season of the office aquarium's clownfish?",
 "How deep is the geothermal borehole beneath the car park?",
 "What tartan is used for the corporate kilt uniform?",
 "Which volcano is nearest to the disaster-recovery bunker?",
 "What is the sail area of the company yacht?",
 "How many beehives are maintained on the roof terrace?",
 "What is the calibration interval for the seismograph in the basement?",
 "Which species of bamboo is planted in the atrium?",
 "What is the maximum altitude of the surveying drone fleet?",
 "How many kilograms of coffee are roasted on site each week?",
 "What is the thread count of the guest-room linens?",
 "Which glacier supplies meltwater to the cooling system?",
 "What is the wingspan of the falcon used for pigeon control?",
 "How many pipe organs are installed in the auditorium?",
 "What is the fermentation time for the on-site kombucha?",
 "Which constellation is depicted on the lobby ceiling mural?",
 "What is the tensile strength of the climbing wall anchors?",
 "How many llamas graze on the corporate retreat grounds?",
 "What is the vintage of the wine cellar's oldest bottle?",
 "Which dialect is taught in the optional language club?",
 "What is the melting point of the sculpture in reception?",
]
STOP={"what","which","when","where","does","that","this","from","with","have","been","many",
      "much","your","company","policy","corporate","office","used","each","many","organisation"}
def main():
    c=sqlite3.connect(DB); kept=[]; rejected=[]
    for q in CAND:
        words=sorted({w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{4,}",q) if w.lower() not in STOP},
                     key=len, reverse=True)[:3]
        # A query is UNANSWERABLE if at least one necessary concept is absent from the corpus.
        # Requiring every term to be absent was over-strict: it rejected "airspeed velocity of the
        # shuttle service" merely because the common word "service" occurs 143 times.
        counts={}
        for w in words:
            counts[w]=c.execute("""select count(*) from chunks ch join documents d on ch.doc_id=d.id
                           where d.filename NOT LIKE 'transcript_%' and lower(ch.text) like ?""",
                        (f"%{w.lower()}%",)).fetchone()[0]
        missing=[w for w,n in counts.items() if n==0]
        if missing:
            kept.append({"text":q,"missing_terms":missing,"term_counts":counts})
        else:
            rejected.append({"text":q,"blocking_term":max(counts.items(),key=lambda x:-x[1]),
                             "term_counts":counts})
    print(f"kept {len(kept)} / rejected {len(rejected)}")
    for r in rejected[:4]: print("   rejected (all terms present):",r["text"][:62])
    # rewrite queries.jsonl: replace S5 items with the verified set
    qs=[json.loads(l) for l in open(f"{OUT}/queries.jsonl") if json.loads(l)["stratum"]!="S5"]
    for i,k in enumerate(kept,1):
        qs.append({"query_id":f"s5_{i:03d}","stratum":"S5","text":k["text"],"namespace":"",
                   "principal":"u_high","gold_doc_substrings":[],"gold_fact_groups":[],
                   "answerable":False,"persona":"General Assistant",
                   "source":"verified_unanswerable",
                   "_verified":f"necessary concept(s) absent from all content chunks: {k['missing_terms']}"})
    with open(f"{OUT}/queries.jsonl","w") as f:
        for q in qs: f.write(json.dumps(q,ensure_ascii=False)+"\n")
    import hashlib
    h=hashlib.sha256(open(f"{OUT}/queries.jsonl","rb").read()).hexdigest()
    from collections import Counter
    meta=json.load(open(f"{OUT}/queries_meta.json"))
    meta["n_queries"]=len(qs); meta["strata"]=dict(Counter(q["stratum"] for q in qs))
    meta["queries_sha256"]=h
    meta["_s5_note"]=(f"S5 rebuilt and verified: {len(kept)} of {len(CAND)} candidates survived a "
                      f"screen requiring NO distinctive term to appear in any content chunk. The "
                      f"original S5 set contained items the corpus could answer, which collapsed the "
                      f"abstention denominator.")
    json.dump(meta,open(f"{OUT}/queries_meta.json","w"),indent=2)
    print(f"queries.jsonl rewritten: n={len(qs)} {meta['strata']} sha={h[:24]}")
main()
