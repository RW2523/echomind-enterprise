#!/usr/bin/env python3
"""E7 (interaction quality + I1 checker) and E10 (dual-loop ablation). Runs INSIDE echomind-voice.

MAPPING: EchoMind's voice service IS a dual loop.
  L_I = the lead-phrase/backchannel path (speaks immediately, must assert nothing new)
  L_G = the RAG path (grounded answer, arrives later)
Invariant I1 = L_I must not make a novel factual assertion before its supporting increment.
"""
import asyncio, base64, io, json, os, sys, time
import numpy as np, soundfile as sf, urllib.request, websockets
OUT="/data/paper_results"; RAW=f"{OUT}/raw"
os.makedirs(OUT,exist_ok=True); os.makedirs(RAW,exist_ok=True)
WS="ws://127.0.0.1:8000/ws"

def wilson(k,n,z=1.96):
    if n==0: return [0.0,0.0]
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); m=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)
    return [round(max(0,(c-m)/d),4), round(min(1,(c+m)/d),4)]

def synth(text):
    body=json.dumps({"text":text}).encode()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/speak",body,
        {"Content-Type":"application/json"}),timeout=90)
    wav,sr=sf.read(io.BytesIO(r.read()),dtype="float32")
    if wav.ndim>1: wav=wav.mean(axis=1)
    n=int(len(wav)*16000/sr)
    return np.interp(np.linspace(0,len(wav)-1,n),np.arange(len(wav)),wav).astype(np.float32)

async def drain(ws,quiet=2.0,cap=30):
    last=time.time(); t0=time.time()
    while time.time()-t0<cap and time.time()-last<quiet:
        try:
            m=json.loads(await asyncio.wait_for(ws.recv(),timeout=0.4))
            if m.get("type")=="audio_out": last=time.time()
        except asyncio.TimeoutError: pass

async def feed(ws,audio,pace=0.016):
    pcm=(np.clip(audio,-1,1)*32767).astype(np.int16).tobytes()
    for i in range(0,len(pcm)-639,640):
        await ws.send(json.dumps({"type":"audio_frame","ts":time.time(),
                                  "pcm16_b64":base64.b64encode(pcm[i:i+640]).decode()}))
        await asyncio.sleep(pace)

async def silence(ws,frames=70,pace=0.016):
    s=base64.b64encode(b"\x00"*640).decode()
    for _ in range(frames):
        await ws.send(json.dumps({"type":"audio_frame","ts":time.time(),"pcm16_b64":s}))
        await asyncio.sleep(pace)

# ─────────── E7b: I1 utterance classifier ───────────
def classify(text):
    t=" ".join((text or "").lower().split())
    if not t: return "empty"
    if any(t.startswith(p) for p in ["let me","one moment","checking","i'll check","give me a",
                                     "sure","of course","okay","alright","got it","hold on"]) and len(t)<90:
        return "acknowledgment"
    if t.endswith("?"): return "clarifying_question"
    if any(p in t for p in ["looking that up","searching","let me look","i'm checking",
                            "pulling that up","one second"]): return "progress_meta"
    # a novel assertion carries specifics: numbers, currency, dates, named entities
    import re
    if re.search(r"\b\d+([.,]\d+)?\s*(%|percent|days?|months?|years?|mg|km|eur|usd|\$|€)\b",t) or \
       re.search(r"\b(usd|eur|\$|€)\s?\d",t) or re.search(r"\bsection\s+\d",t):
        return "novel_assertion"
    return "grounded_restatement"

async def one_turn(question, arm):
    """arm: single_loop | dual_no_i1 | dual_i1. Returns timings + utterance log."""
    audio=synth(question)
    async with websockets.connect(WS,max_size=2**23) as ws:
        await asyncio.wait_for(ws.recv(),timeout=15)
        await drain(ws)
        await feed(ws,audio); t_end=time.time(); await silence(ws)
        t_first_audio=t_final=t_first_text=None
        utts=[]; chunks=0; grounded_arrived=None
        t0=time.time()
        while time.time()-t0<120:
            try: m=json.loads(await asyncio.wait_for(ws.recv(),timeout=1.0))
            except asyncio.TimeoutError:
                if chunks and time.time()-t0>25: break
                continue
            ty=m.get("type")
            if ty=="asr_final": t_final=time.time()
            elif ty in ("assistant_phrase","assistant_text_partial","assistant_text"):
                txt=m.get("text") or ""
                if txt.strip():
                    if t_first_text is None: t_first_text=time.time()
                    utts.append({"t":time.time()-t_end,"text":txt,"cls":classify(txt),
                                 "grounded_available":grounded_arrived is not None})
                if ty=="assistant_text": grounded_arrived=time.time()
            elif ty=="audio_out":
                chunks+=1
                if t_first_audio is None: t_first_audio=time.time()
            elif ty=="event" and m.get("name")=="BACK_TO_LISTENING" and chunks: break
    d=lambda t:(round((t-t_end)*1000,1) if t else None)
    return {"arm":arm,"question":question[:60],"T_first_ms":d(t_first_audio),
            "T_asr_final_ms":d(t_final),"T_grounded_ms":d(grounded_arrived),
            "audio_chunks":chunks,"utterances":utts}

# ─────────── E7a: acoustic barge-in ───────────
async def bargein_trial(question, interrupt_after_s=2.5):
    audio=synth(question); intr=synth("Actually, stop. Tell me something else instead.")
    async with websockets.connect(WS,max_size=2**23) as ws:
        await asyncio.wait_for(ws.recv(),timeout=15); await drain(ws)
        await feed(ws,audio); await silence(ws)
        # wait until assistant is speaking
        t0=time.time(); speaking=False; last_audio=None
        while time.time()-t0<60:
            try: m=json.loads(await asyncio.wait_for(ws.recv(),timeout=0.5))
            except asyncio.TimeoutError: continue
            if m.get("type")=="audio_out":
                speaking=True; last_audio=time.time()
                if time.time()-t0>interrupt_after_s: break
        if not speaking: return None
        t_interrupt=time.time()
        await feed(ws,intr,pace=0.014)      # user speaks over the assistant
        # acoustic stop = last audio_out frame received after interruption onset
        t_last=last_audio; acked=False; t1=time.time()
        while time.time()-t1<12:
            try: m=json.loads(await asyncio.wait_for(ws.recv(),timeout=0.4))
            except asyncio.TimeoutError: break
            if m.get("type")=="audio_out": t_last=time.time()
            if m.get("type")=="cancel": acked=True
    return {"acoustic_stop_ms":round((t_last-t_interrupt)*1000,1) if t_last else None,
            "cancel_event":acked}

async def main():
    Q=["What is the liability cap in the Master Services Agreement?",
       "How are travel expenses reimbursed?",
       "What is the receipt threshold for expenses?",
       "What does the formulary say about hypertension treatment?"]
    # ---- E10: three arms ----
    runs=[]
    for arm in ("dual_i1",):     # production config; other arms need code variants (see limitations)
        for i in range(12):
            try: runs.append(await one_turn(Q[i%len(Q)],arm))
            except Exception as e: runs.append({"arm":arm,"error":str(e)[:120]})
    ok=[r for r in runs if "error" not in r and r.get("T_first_ms")]
    import statistics as st
    def pct(v,p):
        if not v: return None
        v=sorted(v); import math; return round(v[min(len(v)-1,max(0,int(math.ceil(p/100*len(v)))-1))],1)
    tf=[r["T_first_ms"] for r in ok]; tg=[r["T_grounded_ms"] for r in ok if r["T_grounded_ms"]]
    allu=[u for r in ok for u in r["utterances"]]
    viol=[u for u in allu if u["cls"]=="novel_assertion" and not u["grounded_available"]]
    from collections import Counter
    cc=Counter(u["cls"] for u in allu)
    # ---- E7a barge-in ----
    bi=[]
    for i in range(8):
        try:
            r=await bargein_trial(Q[i%len(Q)])
            if r: bi.append(r)
        except Exception: pass
    bms=[b["acoustic_stop_ms"] for b in bi if b.get("acoustic_stop_ms") is not None]
    e7={"status":"partial",
        "bargein":{"n_trials":len(bi),
                   "acoustic_stop_median_ms":round(st.median(bms),1) if bms else None,
                   "acoustic_stop_p95_ms":pct(bms,95),
                   "success_rate":round(sum(1 for b in bi if b.get("cancel_event"))/max(len(bi),1),4),
                   "wilson_ci_95":wilson(sum(1 for b in bi if b.get("cancel_event")),len(bi)),
                   "measurement":"audio_out_frame_cessation",
                   "_note":("Measured as cessation of audio_out frames on the wire — closer to the "
                            "spec's acoustic requirement than the software cancel event, but NOT a "
                            "microphone recording of the speaker; client-side buffer drain is excluded.")},
        "i1_checker":{"n_utterances":len(allu),"class_counts":dict(cc),
                      "i1_violation_rate":round(len(viol)/max(len(allu),1),4),
                      "wilson_ci_95":wilson(len(viol),len(allu)),
                      "checker_validation":{"human_labeled_n":0,"checker_accuracy":None,
                        "cohens_kappa":None,
                        "_note":"NOT human-validated. Rule-based classifier (specifics regex + opener "
                                "patterns). Per spec rule 4 this metric is therefore INDICATIVE ONLY "
                                "and must not be reported as a validated violation rate."},
                      "violations_sample":[{"t":v["t"],"text":v["text"][:120]} for v in viol[:5]]},
        "user_ratings":{"status":"not_run","reason":"no user panel available"},
        "_limitations":["n=%d turns, below the spec's >=200 utterances / >=50 barge-in trials."%len(ok),
                        "Barge-in measured on the wire, not from a recorded speaker waveform."]}
    e10={"status":"partial",
         "arms":[{"arm":"dual_i1_production","n":len(ok),
                  "T_first_median_ms":round(st.median(tf),1) if tf else None,
                  "T_first_p95_ms":pct(tf,95),
                  "T_grounded_median_ms":round(st.median(tg),1) if tg else None,
                  "T_grounded_p95_ms":pct(tg,95),
                  "i1_violation_rate":round(len(viol)/max(len(allu),1),4),
                  "decoupling_ratio":round(st.median(tg)/st.median(tf),2) if tf and tg else None}],
         "_not_run_arms":{"single_loop":"requires disabling the lead-phrase path — a code variant not "
                                        "built; reported not_run rather than simulated.",
                          "dual_no_i1":"requires removing the assertion constraint — same reason."},
         "_note":("Only the production arm was measured, so this is NOT an ablation. It establishes the "
                  "decoupling ratio T_grounded/T_first for the shipped configuration; the comparative "
                  "claim in C1 remains untested.")}
    json.dump(e7,open(f"{OUT}/e7_interaction.json","w"),indent=2)
    json.dump(e10,open(f"{OUT}/e10_ablation.json","w"),indent=2)
    json.dump({"turns":runs,"bargein":bi},open(f"{RAW}/e7_e10_runs.json","w"),indent=2)
    print(f"  T_first median={e10['arms'][0]['T_first_median_ms']}ms "
          f"T_grounded median={e10['arms'][0]['T_grounded_median_ms']}ms "
          f"ratio={e10['arms'][0]['decoupling_ratio']}")
    print(f"  utterances={len(allu)} classes={dict(cc)} i1_violations={len(viol)}")
    print(f"  bargein n={len(bi)} median={e7['bargein']['acoustic_stop_median_ms']}ms")
    print("e7_interaction.json + e10_ablation.json written")

asyncio.run(main())
