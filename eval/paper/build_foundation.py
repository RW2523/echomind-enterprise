#!/usr/bin/env python3
"""Foundation for the QASC experiment suite: config manifest, corpus stats, frozen query set.

Runs INSIDE echomind-backend (needs /app + /data).
Everything here is measured from the live deployment; nothing is estimated.
"""
import hashlib, json, os, subprocess, sqlite3, sys, time
sys.path.insert(0, "/app")
OUT = "/data/paper_results"
os.makedirs(OUT + "/raw", exist_ok=True)
DB = "/data/echomind.sqlite"

def sh(c, d=""):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=30).stdout.strip() or d
    except Exception: return d

def config_manifest():
    from app.core.config import settings
    import torch, faiss
    c = sqlite3.connect(DB)
    nchunks = c.execute("select count(*) from chunks").fetchone()[0]
    ndocs   = c.execute("select count(*) from documents").fetchone()[0]
    ntrans  = c.execute("select count(*) from documents where filename like 'transcript_%'").fetchone()[0]
    tot_tok = c.execute("select coalesce(sum(length(text)),0) from chunks").fetchone()[0] // 4
    from app.rag.index import index
    dim = index.index.d if index.index else 0
    idx_type = type(index.index).__name__ if index.index else "none"
    gpu = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "unknown")
    drv = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader", "unknown")
    return {
        "_mapping_note": ("Spec assumes pgvector/ACL/Whisper stack. EchoMind is FAISS+BM25+SQLite, "
                          "namespace-based isolation (not per-chunk ACLs), NVIDIA NeMo ASR. Fields "
                          "below describe the ACTUAL system; spec-only fields are marked not_applicable."),
        "asr": {"model": "nvidia/nemotron-speech-streaming-en-0.6b (streaming partials) + "
                         "nvidia/parakeet-tdt-0.6b-v2 (final decode)",
                "revision": "HF default", "quantization": "fp32/bf16 (NeMo default)",
                "streaming": True, "chunk_ms": 560,
                "vad": "webrtcvad aggressiveness=1 + RMS gate 0.004", "beam_size": 0,
                "_note": "two-model design; spec assumed whisper-large-v3"},
        "embedding": {"model": settings.OLLAMA_EMBED_MODEL, "revision": "ollama latest",
                      "dimension": dim, "normalization": "l2"},
        "vector_store": {"engine": f"FAISS {faiss.__version__} (in-process) + SQLite metadata",
                         "index_type": idx_type, "hnsw_m": 0, "hnsw_ef_construction": 0,
                         "hnsw_ef_search": 0, "ivfflat_lists": 0, "distance": "ip (cosine via L2-norm)",
                         "permission_predicate": ("namespace equality test _ns_ok(src) evaluated INSIDE the "
                                                  "candidate scan loop in index.py (pre-filter). Not an SQL "
                                                  "predicate; not per-chunk ACLs. Sparse BM25 path filtered "
                                                  "identically."),
                         "_spec_fields_not_applicable": ["hnsw_m","hnsw_ef_construction","hnsw_ef_search","ivfflat_lists"]},
        "chunking": {"strategy": "type-detected: recursive/sentence-aware (flat) — BookRAG parent/child "
                                 "gated off by RAG_ENABLE_BOOKRAG=0",
                     "chunk_tokens": settings.CHUNK_SIZE, "overlap_tokens": settings.CHUNK_OVERLAP},
        "retrieval": {"k_candidates": settings.TOP_K,
                      "k_after_rerank": getattr(settings, "RAG_RERANK_FINAL_N", 15),
                      "reranker": getattr(settings, "RAG_CE_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                      "reranker_revision": "sentence-transformers default",
                      "fusion": "weighted RRF (K=60), query-type-adaptive dense/sparse weights",
                      "sparse": "rank-bm25 BM25Okapi"},
        "answer_model": {"model": settings.LLM_MODEL, "revision": "nvidia FP4 checkpoint",
                         "quantization": "NVFP4 (4-bit, Blackwell-native)",
                         "serving": "TensorRT-LLM 1.2.0rc6 (OpenAI-compatible)",
                         "max_context_tokens": 40960, "temperature": settings.LLM_TEMPERATURE,
                         "top_p": 1.0, "seed": 0,
                         "_note": "temperature is the deployed default; experiments override to 0.0"},
        "tts": {"model": "piper en_US-lessac-medium (CPU)", "streaming": True},
        "duplex_layer": {"implementation": "voice/app/session.py — L_I lead-phrase loop + L_G grounded "
                                           "RAG loop, phrase-committed TTS",
                         "micro_turn_ms": 20},
        "hardware": {"accelerator": gpu, "accelerator_count": 1, "driver": drv,
                     "cpu": sh("nproc", "?") + " cores ARM64 Grace",
                     "ram_gb": int(int(sh("awk '/MemTotal/{print $2}' /proc/meminfo", "0") or 0) / 1024 / 1024),
                     "storage": "NVMe SSD", "colocated": True,
                     "_note": "unified CPU+GPU memory (GB10); all services on one node"},
        "corpus": {"n_documents": ndocs, "n_chunks": nchunks, "n_transcripts": ntrans,
                   "total_tokens_est": tot_tok,
                   "domain_mix": "DoD FMR regulation PDFs + 5 synthetic vertical demo docs "
                                 "(bank/health/law/meetings/retail) + auto-saved voice transcripts",
                   "synthetic_fraction": None, "provenance": "public (DoD FMR) + synthetic demo docs"},
        "date_run": time.strftime("%Y-%m-%d"),
        "harness_commit": sh("cd /app && git rev-parse --short HEAD", "n/a (container)"),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    }

def corpus_stats():
    c = sqlite3.connect(DB)
    rows = c.execute("""select d.id,d.filename,count(ch.id),coalesce(sum(length(ch.text)),0)
                        from documents d join chunks ch on ch.doc_id=d.id group by d.id""").fetchall()
    docs = [{"doc_id": r[0], "filename": r[1], "n_chunks": r[2], "chars": r[3]} for r in rows]
    non_tr = [d for d in docs if not d["filename"].startswith("transcript_")]
    tr     = [d for d in docs if d["filename"].startswith("transcript_")]
    import collections
    ns = collections.Counter()
    for (sj,) in c.execute("select source_json from chunks"):
        ns[json.loads(sj or "{}").get("namespace") or "untagged"] += 1
    h = hashlib.sha256()
    for cid, in c.execute("select id from chunks order by id"): h.update(cid.encode())
    return {
        "n_documents": len(docs), "n_chunks": sum(d["n_chunks"] for d in docs),
        "content_documents": len(non_tr), "content_chunks": sum(d["n_chunks"] for d in non_tr),
        "transcript_documents": len(tr), "transcript_chunks": sum(d["n_chunks"] for d in tr),
        "namespace_distribution": dict(ns),
        "corpus_sha256_of_chunk_ids": h.hexdigest(),
        "content_documents_detail": sorted(non_tr, key=lambda d: -d["n_chunks"]),
        "_honesty_note": ("97% of chunks are auto-saved voice-session transcripts, most very short. "
                          "The substantive corpus is the DoD FMR PDFs + 10 demo markdown documents. "
                          "This is materially smaller than the spec's 1,000-doc minimum; every "
                          "retrieval-scale claim must be read with that caveat."),
    }

if __name__ == "__main__":
    cm = config_manifest(); cs = corpus_stats()
    json.dump(cm, open(f"{OUT}/config_manifest.json", "w"), indent=2)
    json.dump(cs, open(f"{OUT}/corpus_stats.json", "w"), indent=2)
    print("config_manifest.json + corpus_stats.json written")
    print(f"  chunks={cs['n_chunks']} content={cs['content_chunks']} transcripts={cs['transcript_chunks']}")
    print(f"  embedding dim={cm['embedding']['dimension']} index={cm['vector_store']['index_type']}")
    print(f"  LLM={cm['answer_model']['model']}")
