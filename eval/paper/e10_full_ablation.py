#!/usr/bin/env python3
"""E10 — dual-loop ablation across all three arms + E7b I1 checker using loop provenance.

Arms are selected by VOICE_ABLATION_MODE, which the harness sets per arm via the /set_ablation
control message (falls back to env if unavailable). Default production behaviour is dual_i1.
"""
import asyncio, base64, io, json, os, statistics as st, sys, time
import numpy as np, soundfile as sf, urllib.request, websockets
OUT="/data/paper_results"; RAW=f"{OUT}/raw"; os.makedirs(RAW,exist_ok=True)
WS="ws://127.0.0.1:8000/ws"

def wilson(k,n,z=1.96):
    if n==0: return [0.0,0.0]
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); m=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)
    return [round(max(0,(c-m)/d),4), round(min(1,(c+m)/d),4)]
def pct(v,p):
    if not v: return None
    v=sorted(v); import math
    return round(v[min(len(v)-1,max(0,int(math.ceil(p/100*len(v)))-1))],1)

def synth(text):
    body=json.dumps({"text":text}).encode()
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/speak",body,
        {"Content-Type":"application/json"}),timeout=90)
    wav,sr=sf.read(io.BytesIO(r.read()),dtype="float32")
    if wav.ndim>1: wav=wav.mean(axis=1)
    n=int(len(wav)*16000/sr)
    return np.interp(np.linspace(0,len(wav)-1,n),np.arange(len(wav)),wav).astype(np.float32)

async def drain(ws,quiet=2.0,cap=25):
    last=t0=time.time()
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
    s=base64.b64encode(b"\x00"*640).decode()
    for _ in range(70):
        await ws.send(json.dumps({"type":"audio_frame","ts":time.time(),"pcm16_b64":s}))
        await asyncio.sleep(pace)

ASSERTION_MARKERS=["is ","are ","was ","were ","costs","requires","must","cap is","rate is",
                   "about","around","approximately","i believe","i think","as far as","recall"]
import re
SPECIFIC=re.compile(r"\b\d+([.,]\d+)?\s*(%|percent|days?|months?|years?|mg|km|eur|usd|\$|€|thousand|million)\b|\b(usd|eur|\$|€)\s?\d")

def is_novel_assertion(text):
    """L_I utterance that asserts a fact (specific figure or declarative claim)."""
    t=" ".join((text or "").lower().split())
    if not t: return False
    if t.endswith("?"): return False
    if SPECIFIC.search(t): return True
    return any(m in t for m in ["i believe","i think","as far as","i recall"]) and \
           any(m in t for m in ASSERTION_MARKERS)

async def one_turn(q):
    audio=synth(q)
    async with websockets.connect(WS,max_size=2**23) as ws:
        await asyncio.wait_for(ws.recv(),timeout=15); await drain(ws)
        await feed(ws,audio); t_end=time.time()
        t_first_audio=t_grounded=None; chunks=0; li=[]; lg=0
        t0=time.time()
        while time.time()-t0<110:
            try: m=json.loads(await asyncio.wait_for(ws.recv(),timeout=1.0))
            except asyncio.TimeoutError:
                if chunks and time.time()-t0>22: break
                continue
            ty=m.get("type"); loop=m.get("loop")
            if ty=="assistant_phrase" and loop=="L_I":
                li.append({"t":round(time.time()-t_end,3),"text":m.get("text") or "",
                           "grounded_yet":t_grounded is not None})
            elif ty=="assistant_text_partial" and loop=="L_G":
                lg+=1
                if t_grounded is None: t_grounded=time.time()
            elif ty=="assistant_text" and t_grounded is None: t_grounded=time.time()
            elif ty=="audio_out":
                chunks+=1
                if t_first_audio is None: t_first_audio=time.time()
            elif ty=="event" and m.get("name")=="BACK_TO_LISTENING" and chunks: break
    d=lambda t: round((t-t_end)*1000,1) if t else None
    return {"T_first_ms":d(t_first_audio),"T_grounded_ms":d(t_grounded),
            "audio_chunks":chunks,"L_I_utterances":li,"L_G_deltas":lg}

async def main():
    Q=["What is the liability cap in the Master Services Agreement?",
       "How are travel expenses reimbursed?",
       "What is the receipt threshold for expenses?",
       "What does the formulary say about hypertension treatment?"]
    N=int(os.getenv("E10_N","10"))
    arms={}
    for arm in ("dual_i1","single_loop","dual_no_i1"):
        open("/tmp/echomind_ablation_mode","w").write(arm)   # server reads this per turn
        await asyncio.sleep(0.3)
        runs=[]
        for i in range(N):
            try: runs.append(await one_turn(Q[i%len(Q)]))
            except Exception as e: runs.append({"error":str(e)[:100]})
        ok=[r for r in runs if "error" not in r and r.get("T_first_ms")]
        tf=[r["T_first_ms"] for r in ok]; tg=[r["T_grounded_ms"] for r in ok if r["T_grounded_ms"]]
        allli=[u for r in ok for u in r["L_I_utterances"]]
        viol=[u for u in allli if not u["grounded_yet"] and is_novel_assertion(u["text"])]
        arms[arm]={"arm":arm,"n":len(ok),
          "T_first_median_ms":round(st.median(tf),1) if tf else None,"T_first_p95_ms":pct(tf,95),
          "T_grounded_median_ms":round(st.median(tg),1) if tg else None,"T_grounded_p95_ms":pct(tg,95),
          "decoupling_ratio":round(st.median(tg)/st.median(tf),2) if tf and tg else None,
          "n_L_I_utterances":len(allli),"n_i1_violations":len(viol),
          "i1_violation_rate":round(len(viol)/len(allli),4) if allli else None,
          "i1_wilson_ci_95":wilson(len(viol),len(allli)) if allli else None,
          "violation_samples":[u["text"][:100] for u in viol[:3]],
          "raw":runs}
        print(f"  {arm}: n={len(ok)} T_first={arms[arm]['T_first_median_ms']} "
              f"T_grounded={arms[arm]['T_grounded_median_ms']} "
              f"I1_viol={arms[arm]['n_i1_violations']}/{len(allli)}")
    try: os.remove("/tmp/echomind_ablation_mode")            # restore production default
    except Exception: pass
    out={"status":"complete",
         "arms":[{k:v for k,v in a.items() if k!="raw"} for a in arms.values()],
         "_predicted":("(a) single_loop: high T_first, zero violations; (b) dual_no_i1: low T_first, "
                       "nonzero violations; (c) dual_i1: low T_first AND near-zero violations."),
         "_i1_checker":{"type":"rule_based_on_loop_provenance","human_validated":False,
           "note":("Now measurable because L_I and L_G utterances carry an explicit loop tag. Only "
                   "L_I utterances emitted before the first L_G increment are candidates. The "
                   "assertion test (specific figure or hedged declarative) is rule-based and NOT "
                   "human-validated, so per spec rule 4 report it as indicative.")},
         "_limitations":[f"n={N} turns per arm, below the spec's ideal.",
                         "Ablation arms are env-gated evaluation paths; default deployment is dual_i1."]}
    json.dump(out,open(f"{OUT}/e10_ablation.json","w"),indent=2)
    json.dump({k:v["raw"] for k,v in arms.items()},open(f"{RAW}/e10_runs.json","w"),indent=2)
    print("e10_ablation.json written")

asyncio.run(main())
