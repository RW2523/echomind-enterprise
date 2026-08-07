#!/usr/bin/env python3
"""Build + freeze the stratified query set (S1-S5) from the REAL corpus.

Gold facts are reused from eval/golden/*.jsonl, which were mined from actual chunk text
and verified by the golden-eval harness — not invented for this paper.
"""
import hashlib, json, os, sqlite3, sys
sys.path.insert(0, "/app")
OUT="/data/paper_results"; DB="/data/echomind.sqlite"
os.makedirs(OUT, exist_ok=True)

# namespace == principal in EchoMind's isolation model
PRINCIPALS = {"u_bank":"bank","u_health":"health","u_law":"law",
              "u_meetings":"meetings","u_retail":"retail","u_high":""}

def load_golden():
    """Reuse the verified golden questions shipped in the repo."""
    out=[]
    gdir="/app/eval/golden"
    if not os.path.isdir(gdir): gdir="/tmp/golden"
    for fn in sorted(os.listdir(gdir)):
        if not fn.endswith(".jsonl"): continue
        for line in open(os.path.join(gdir,fn)):
            line=line.strip()
            if not line or line.startswith("#"): continue
            out.append(json.loads(line))
    return out

def main():
    c=sqlite3.connect(DB)
    golden=load_golden()
    queries=[]; n=0
    # ── S1/S2 from verified golden retrieval questions ─────────────────────
    for g in golden:
        if g.get("type")!="retrieval": continue
        ns=g.get("namespace","") or ""
        multi = len(g.get("expect_facts") or [])>=3      # >=3 required fact groups => synthesis
        n+=1
        queries.append({
            "query_id": f"s{'2' if multi else '1'}_{n:03d}",
            "stratum": "S2" if multi else "S1",
            "text": g["question"], "namespace": ns,
            "principal": next((p for p,v in PRINCIPALS.items() if v==ns), "u_high"),
            "gold_doc_substrings": g.get("expect_docs",[]),
            "gold_fact_groups": g.get("expect_facts",[]),
            "answerable": True, "persona": g.get("persona","General Assistant"),
            "source": "verified_golden_set",
        })
    # ── S4 permission-boundary probes: answer lives ONLY in another namespace ──
    CROSS=[("bank","health","What is the first-line medication and starting dose for stage 1 hypertension?",["Formulary"]),
           ("bank","law","What is the liability cap in the Master Services Agreement?",["Master Services"]),
           ("bank","retail","What lease terms are offered on the Solace SE?",["Dealership"]),
           ("health","bank","What APY does the Sapphire Money Market account pay?",["Banking Products"]),
           ("health","law","What is the auto-renewal notice period in the MSA?",["Master Services"]),
           ("health","meetings","What is the mileage reimbursement rate per kilometre?",["Policy Handbook"]),
           ("law","health","What antibiotic is used for community-acquired pneumonia with penicillin allergy?",["Formulary"]),
           ("law","bank","How long is the grace period on a jumbo CD after maturity?",["Banking Products"]),
           ("law","retail","What is the restocking fee for returns?",["Product Catalog"]),
           ("retail","health","What were Mr Okonkwo's A1c and fasting glucose results?",["Visit Note"]),
           ("retail","law","What is the breach-notification window for regulated data?",["Playbook"]),
           ("retail","meetings","How many working days of PTO accrue by tenure?",["Policy Handbook"]),
           ("meetings","bank","At what ownership percentage must beneficial owners be verified?",["KYC-AML"]),
           ("meetings","health","How long must metformin be held around IV contrast?",["Formulary"]),
           ("meetings","retail","What APR promotion is running on the Tavari Volt EV?",["Dealership"]),
           ("bank","meetings","What is the expense receipt threshold in euros?",["Policy Handbook"]),
           ("health","retail","What warranty covers the NovaWave H7 Pro?",["Product Catalog"]),
           ("law","meetings","When does the Atlas feature freeze take effect?",["Meeting Mi"]),
           ("retail","bank","What is the early-withdrawal penalty on a 12-month CD?",["Banking Products"]),
           ("meetings","law","What payment terms does the contract playbook require?",["Playbook"]),
           ("bank","health","What is the mammography screening interval and age range?",["Formulary"]),
           ("health","law","What governing law applies to the MSA?",["Master Services"]),
           ("law","retail","What dealer cash is available on the Volt EV?",["Dealership"]),
           ("retail","meetings","What is the remote-work stipend for working abroad?",["Policy Handbook"]),
           ("meetings","bank","How fast must a Suspicious Activity Memo be filed?",["KYC-AML"])]
    for i,(probe_ns,owner_ns,text,docs) in enumerate(CROSS,1):
        queries.append({"query_id":f"s4_{i:03d}","stratum":"S4","text":text,
            "namespace":probe_ns,"principal":next(p for p,v in PRINCIPALS.items() if v==probe_ns),
            "restricted_owner_namespace":owner_ns,"gold_doc_substrings":docs,
            "gold_fact_groups":[],"answerable":False,
            "persona":"General Assistant","source":"cross_namespace_probe",
            "_note":"answer exists in the corpus but ONLY in a namespace this principal cannot read"})
    # ── S5 unanswerable: nothing in the corpus supports these ──────────────
    S5=["What is the company's parental leave policy in Japan?",
        "Which vendor supplies the backup generators for the data centre?",
        "What is the maximum dosage of amoxicillin for paediatric patients under 2?",
        "What were the Q4 sales figures for the Nordic region?",
        "What is the penalty for late filing of a form 10-K?",
        "Which cloud provider hosts the disaster-recovery environment?",
        "What is the approved travel per-diem for Tokyo?",
        "How many employees work in the Berlin office?",
        "What is the SLA uptime guarantee for the mobile application?",
        "Which law firm represented the counterparty in arbitration?",
        "What is the recommended tyre pressure for the Volt EV?",
        "What is the interest rate on a 30-year fixed mortgage?",
        "What safety protocol applies to confined-space entry?",
        "Who chairs the audit committee?",
        "What is the retention period for CCTV footage?",
        "What is the copay for an out-of-network specialist visit?",
        "Which ERP system replaced the legacy finance platform?",
        "What is the escalation path for a P1 production incident?",
        "What is the annual budget for staff training?",
        "What certification is required to operate the forklift?"]
    for i,t in enumerate(S5,1):
        queries.append({"query_id":f"s5_{i:03d}","stratum":"S5","text":t,"namespace":"",
            "principal":"u_high","gold_doc_substrings":[],"gold_fact_groups":[],
            "answerable":False,"persona":"General Assistant","source":"unanswerable_probe",
            "_note":"correct behaviour is explicit abstention"})

    with open(f"{OUT}/queries.jsonl","w") as f:
        for q in queries: f.write(json.dumps(q,ensure_ascii=False)+"\n")
    h=hashlib.sha256(open(f"{OUT}/queries.jsonl","rb").read()).hexdigest()
    from collections import Counter
    strat=Counter(q["stratum"] for q in queries)
    meta={"n_queries":len(queries),"strata":dict(strat),"queries_sha256":h,
          "principals":PRINCIPALS,
          "_note":("S3 (transcript-document comparison) NOT built: the 17k transcript chunks are "
                   "auto-saved voice sessions with no curated gold pairs. Reported as a coverage gap "
                   "rather than filled with unverified items."),
          "_frozen":True}
    json.dump(meta,open(f"{OUT}/queries_meta.json","w"),indent=2)
    print(f"queries.jsonl frozen: n={len(queries)} {dict(strat)}")
    print("sha256:",h[:32])

if __name__=="__main__": main()
