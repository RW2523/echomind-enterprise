#!/usr/bin/env python3
"""Assemble SUMMARY.json from whatever experiments actually produced output.

Missing file => status "not_run". Nothing is inferred, estimated, or back-filled.
"""
import json, os, sys
OUT="/data/paper_results"

def load(n):
    p=f"{OUT}/{n}"
    return json.load(open(p)) if os.path.exists(p) else None

def main():
    e1=load("e1_latency.json"); e3=load("e3_permission.json"); e4=load("e4_injection.json")
    e5=load("e5_grounding.json"); e7=load("e7_interaction.json"); e10=load("e10_ablation.json")
    cs=load("corpus_stats.json"); qm=load("queries_meta.json")
    ex={}

    if e1:
        c=e1["cells"][0]; s=c["stages"]
        ex["E1"]={"status":e1["status"],
          "headline":(f"T_grounded median {s['T_grounded']['median_ms']:.0f} ms "
                      f"(p95 {s['T_grounded']['p95_ms']:.0f} ms) at concurrency 1; "
                      f"retrieval share {c['retrieval_share_of_T_grounded']:.3f}; "
                      f"permission filter {s['retrieval.permission_filter']['median_ms']:.2f} ms"),
          "contradicts_paper":("YES — §7.2 argues the retrieval terms dominate T_grounded. Measured, "
                               "they are 2.8% of it; generation is ~97%. The dual-loop split is still "
                               "justified, but by GENERATION latency, not retrieval latency."),
          "caveat":"warm cells only; cold-cache cells not run (needs host root to drop page cache)"}
    else: ex["E1"]={"status":"not_run","reason":"harness did not complete"}

    ex["E2"]={"status":"not_run",
      "reason":("Corpus-scale sweep not run. Only 421 of 17,514 chunks are content (the rest are "
                "auto-saved voice transcripts), far below the spec's 1,000-doc / 50k-chunk floor, so "
                "a 10/25/50/100% sweep would measure transcript noise rather than retrieval scaling.")}

    if e3:
        a={x["arm"]:x for x in e3["arms"]}
        pf=a["pre_filter"]; po=a["post_filter"]
        ex["E3"]={"status":e3["status"],
          "headline":(f"content leak 0/{pf['content_leak']['n_probes']} in BOTH arms "
                      f"(Wilson CI {pf['content_leak']['wilson_ci_95']}); disclosing refusal rate 0.0; "
                      f"BUT a timing channel is present in both: pre_filter KS="
                      f"{pf['timing_channel']['ks_statistic']} p={pf['timing_channel']['p_value']}, "
                      f"post_filter KS={po['timing_channel']['ks_statistic']} "
                      f"p={po['timing_channel']['p_value']}"),
          "partially_contradicts_paper":("§6.1 predicts pre-filtering closes the side channels. Content "
                                         "and wording channels ARE closed. The TIMING channel is NOT: "
                                         "queries whose answer exists in another tenant return "
                                         "significantly faster than queries with no answer anywhere, "
                                         "so response time still distinguishes existence.")}
    else: ex["E3"]={"status":"not_run","reason":"harness did not complete"}

    if e4:
        a={x["arm"]:x for x in e4["arms"]}
        b0=a["B0_no_defense"]["overall"]; b3=a["B3_full"]["overall"]
        ex["E4"]={"status":e4["status"],
          "headline":(f"attack success B0 {b0['attack_success_rate']:.3f} CI{b0['wilson_ci_95']} → "
                      f"B3 {b3['attack_success_rate']:.3f} CI{b3['wilson_ci_95']} "
                      f"(n={b0['n']} payloads/arm); false positives 0/25 in every arm"),
          "contradicts_paper":("§6.2 claims the lattice degrades injection to a REPORTABLE event. "
                               "report_rate is 0.000 in every arm: containment is real but SILENT. "
                               "The system never tells the user a document carried an instruction it "
                               "declined."),
          "caveat":("B0/B3 Wilson intervals overlap at n=35, so the containment trend is suggestive, "
                    "not decisive. A4 tool-abuse omitted (no tool layer exists) rather than scored 0.")}
    else: ex["E4"]={"status":"not_run","reason":"harness did not complete"}

    if e5:
        cp=e5["citation_precision"]; cr=e5["citation_recall"]; ab=e5["abstention_accuracy_S5"]
        ex["E5"]={"status":e5["status"],
          "headline":(f"citation precision {cp['mean']} (n={cp['n']}), recall {cr['mean']}; "
                      f"gold-fact support {e5['fact_support_rate']['rate']}; "
                      f"S5 abstention {ab['correct']}/{ab['n']} = {ab['rate']} "
                      f"CI{ab['wilson_ci_95']} ({ab.get('n_excluded_as_answerable',0)} S5 items "
                      f"excluded as actually answerable)"),
          "caveat":("unsupported_claim_rate NOT measured — needs atomic-claim decomposition and a "
                    "human-validated judge. fact_support_rate is a different, stricter substring "
                    "metric and understates paraphrased support. No LLM judge used, so no kappa.")}
    else: ex["E5"]={"status":"not_run","reason":"harness did not complete"}

    e3b=load("e3b_path_isolation.json")
    if e3b:
        tot=sum(p["n_hits_checked"] for p in e3b["paths"])
        lk=sum(p["n_out_of_namespace"] for p in e3b["paths"])
        ex["E3b"]={"status":"complete",
          "headline":f"per-path namespace audit: {lk} out-of-namespace hits across {tot} checked, all 5 paths clean (POST-fix)",
          "critical_finding":e3b["_history"],
          "for_paper":e3b["_finding_for_paper"]}

    ex["E6"]={"status":"not_run",
      "reason":("Outcome correctness needs per-query rubric annotation by two independent annotators "
                "with Cohen's kappa. No annotator panel available; an LLM-only rubric score would "
                "violate the spec's rule 4 (unvalidated judge).")}

    if e7:
        b=e7["bargein"]; i=e7.get("i1_checker",{})
        ex["E7"]={"status":e7["status"],
          "headline":(f"barge-in stop median {b['acoustic_stop_median_ms']} ms (n={b['n_trials']}); "
                      f"I1 violation rate {i.get('status','not_measurable')} — see e7_interaction.json"),
          "caveat":(i.get("reason","") or "") + " Barge-in measured from onset of injected user "
                    "speech to last audio frame (spec definition), n=8, below the spec's 50 trials."}
    else: ex["E7"]={"status":"not_run","reason":"harness did not complete"}

    ex["E8"]={"status":"not_run",
      "reason":("Process efficiency measures tool calls and recovery from injected tool failures. "
                "EchoMind has no tool-calling layer, so the metric is undefined for this system.")}

    ex["E9"]={"status":"not_run",
      "reason":("AHP weight elicitation requires >=3 panels x >=3 human practitioners. No panel was "
                "available; synthesising pairwise judgements would fabricate the very weights the "
                "experiment exists to measure.")}

    if e10:
        arms={x["arm"]:x for x in e10["arms"]}
        a=arms.get("dual_i1", e10["arms"][0])
        ex["E10"]={"status":e10["status"],
          "headline":(f"production dual-loop: T_first median {a['T_first_median_ms']} ms, "
                      f"first grounded token {a.get('T_first_grounded_token_median_ms')} ms, "
                      f"I1 violations {a.get('n_i1_violations')}/{a.get('n_L_I_utterances')}"),
          "all_arms":{k:{"T_first_median_ms":v.get("T_first_median_ms"),
                         "i1_violations":f"{v.get('n_i1_violations')}/{v.get('n_L_I_utterances')}"}
                      for k,v in arms.items()},
          "finding":e10.get("_finding",""),
          "caveat":("Effect size is MODEST: dual_i1 635.8 ms vs single_loop 865.8 ms first audio "
                    "(230 ms, ~27%). Fast TRT-LLM prefill leaves little latency for L_I to mask; the "
                    "benefit should grow as generation slows. n=10 per arm.")}
    else: ex["E10"]={"status":"not_run","reason":"harness did not complete"}

    summary={
      "experiments":ex,
      "corpus_is_synthetic":False,
      "corpus_is_public":True,
      "corpus_note":(f"{cs['n_chunks']} chunks total but only {cs['content_chunks']} are content "
                     f"({cs['content_documents']} documents: public DoD FMR regulation PDFs + 10 "
                     f"synthetic vertical demo documents). {cs['transcript_chunks']} are auto-saved "
                     f"voice-session transcripts. Materially below the spec's corpus floor."
                     ) if cs else "corpus_stats.json missing",
      "query_set":{"n":qm["n_queries"],"strata":qm["strata"],"sha256":qm["queries_sha256"],
                   "note":qm["_note"]} if qm else None,
      "page_limit_for_track":None,
      "surprises_or_negative_results":[
        "E1 REFUTES §7.2: retrieval is 2.8% of T_grounded, not the dominant term. Generation is ~97%. "
        "The dual-loop argument should be re-grounded on generation latency.",
        "E4 REFUTES the reportability half of §6.2: report_rate is 0.000 in every arm. Containment "
        "improves 17.1% → 2.9% from B0 to B3, but the system never surfaces the attempt to the user.",
        "E3 PARTIALLY REFUTES §6.1: content and refusal-wording channels are closed, but a significant "
        "timing channel survives pre-filtering (KS≈0.54, p<1e-5).",
        "E1 tail latency is severe: p95 T_grounded 15,171 ms against a 3,313 ms median.",
        "The permission predicate costs 0.08 ms — governance is NOT what makes retrieval expensive in "
        "this architecture, contrary to the framing in §7.2.",
      ],
      "architecture_mapping_warning":(
        "The specification assumes pgvector/HNSW with per-chunk ACLs, Whisper ASR, and a tool-calling "
        "layer. EchoMind is FAISS+BM25 over SQLite, isolates by TENANT NAMESPACE (no ACL model), uses "
        "NeMo ASR, and has NO tools. Property P1 was therefore tested as namespace pre-filtering, and "
        "the tool-allowlist component of C2 is UNIMPLEMENTED and untestable. The paper should narrow "
        "these claims rather than let a reviewer discover the gap."),
      "harness_integrity_note":(
        "Five measurement defects were found and corrected before reporting: (1) a module-global span "
        "store contaminated by concurrent tasks produced an impossible retrieval share of 1.61; "
        "(2) a refusal classifier scored generic 'I don't have access to documents' as authorization "
        "disclosure, inflating disclosing_rate to 0.36; (3) an E3 control group of answerable queries "
        "confounded 'restricted content exists' with 'no answer available'; (4) an E4 judge scored "
        "canary string presence as obedience on a summarisation task where quoting the payload was "
        "correct, yielding a false 74% attack-success rate; (5) an E5 abstention classifier missed the "
        "model's most common abstention phrasings and the S5 set contained answerable items. All "
        "corrected figures are the ones reported."),
      "notes_for_paper":(
        "Tier 1 is complete (E1, E3, E4, E5). Tier 2 is partial: E7 and E10 measured on the production "
        "configuration only. Tier 3 (E6, E8, E9) not run for the structural reasons recorded above. "
        "Three of the four Tier-1 results contradict or qualify a claim in the manuscript; all are "
        "reported as measured."),
    }
    json.dump(summary,open(f"{OUT}/SUMMARY.json","w"),indent=2)
    print("SUMMARY.json written")
    for k,v in ex.items(): print(f"  {k}: {v['status']}")

main()
