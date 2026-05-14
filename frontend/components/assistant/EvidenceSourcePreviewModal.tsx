import React, { useEffect, useMemo, useState } from "react";
import type { AssistantEvidence } from "../../types";
import { ICONS } from "../../constants";
import { evidenceSourceTypeBadge } from "./assistantUtils";

const API_BASE = (import.meta as { env?: Record<string, string> }).env?.VITE_API_BASE || "";

interface EvidenceSourcePreviewModalProps {
  evidence: AssistantEvidence[];
  onClose: () => void;
  initialIndex?: number;
}

function canOpenDocument(ev: AssistantEvidence): boolean {
  return Boolean(ev.doc_id) && ev.source_type !== "transcript";
}

const EvidenceSourcePreviewModal: React.FC<EvidenceSourcePreviewModalProps> = ({
  evidence,
  onClose,
  initialIndex = 0,
}) => {
  const items = useMemo(
    () => (evidence || []).filter((e) => (e.matched_text || "").trim() || (e.source_name || "").trim()),
    [evidence]
  );
  const [selectedIndex, setSelectedIndex] = useState<number | null>(() => {
    if (items.length === 0) return null;
    if (items.length === 1) return 0;
    if (initialIndex < 0) return null;
    return Math.min(Math.max(0, initialIndex), items.length - 1);
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (items.length === 0) return null;

  const selected = selectedIndex != null ? items[selectedIndex] : null;

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center p-3 sm:p-4 bg-black/75 backdrop-blur-sm"
      onClick={() => {
        if (selected && items.length > 1) setSelectedIndex(null);
        else onClose();
      }}
    >
      <div
        className="w-full max-w-2xl max-h-[88vh] flex flex-col rounded-2xl border border-white/20 bg-[#0c1220] shadow-2xl overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-4 py-3 border-b border-white/10 shrink-0 flex items-center gap-3">
          {selected && items.length > 1 ? (
            <button
              type="button"
              onClick={() => setSelectedIndex(null)}
              className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors shrink-0"
              aria-label="Back to sources"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          ) : (
            <div className="w-5 h-5 shrink-0 opacity-70">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
          )}
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm text-white truncate">
              {selected ? "Source preview" : `Referred sources · ${items.length}`}
            </h3>
            {selected && <p className="text-[11px] text-slate-500 truncate mt-0.5">{selected.source_name}</p>}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors shrink-0"
            aria-label="Close"
          >
            <ICONS.Close className="w-4 h-4" />
          </button>
        </div>

        <div className="flex-1 min-h-0 overflow-auto">
          {selected ? (
            <div className="p-4 space-y-4">
              <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3.5 space-y-2.5">
                <div className="flex items-start gap-2 flex-wrap">
                  <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500 w-14 shrink-0 pt-px">
                    Resource
                  </span>
                  <span className="text-xs text-white/90 break-all flex-1 min-w-0">{selected.source_name}</span>
                  {(() => {
                    const b = evidenceSourceTypeBadge(selected.source_type);
                    return (
                      <span className={`inline-block px-2 py-0.5 rounded-md text-[10px] font-medium ${b.cls}`}>
                        {b.label}
                      </span>
                    );
                  })()}
                </div>
                {selected.section && (
                  <div className="flex items-start gap-2">
                    <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500 w-14 shrink-0 pt-px">
                      Section
                    </span>
                    <span className="text-xs text-cyan-300/90 font-mono break-all leading-relaxed">{selected.section}</span>
                  </div>
                )}
                {selected.page != null && (
                  <div className="flex items-start gap-2">
                    <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500 w-14 shrink-0 pt-px">
                      Page
                    </span>
                    <span className="text-xs text-slate-300">{selected.page}</span>
                  </div>
                )}
                {selected.chunk_id && (
                  <div className="flex items-start gap-2">
                    <span className="text-[10px] font-medium uppercase tracking-wider text-slate-500 w-14 shrink-0 pt-px">
                      Chunk
                    </span>
                    <span className="text-xs text-slate-500 font-mono break-all">{selected.chunk_id}</span>
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-cyan-500/20 bg-cyan-950/20 p-4">
                <p className="text-[10px] font-medium uppercase tracking-wider text-cyan-600 mb-2.5">Source content</p>
                <p className="text-sm text-white/85 leading-relaxed whitespace-pre-wrap">
                  {selected.matched_text || <span className="text-slate-500 italic">No excerpt available.</span>}
                </p>
              </div>

              {canOpenDocument(selected) && (
                <a
                  href={`${API_BASE}/api/docs/${selected.doc_id}/file#page=${selected.page ?? 1}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 hover:text-white text-sm font-medium transition-colors"
                >
                  Open in document{selected.page != null ? ` — page ${selected.page}` : ""}
                </a>
              )}
            </div>
          ) : (
            <ul className="divide-y divide-white/[0.06]">
              {items.map((ev, i) => {
                const badge = evidenceSourceTypeBadge(ev.source_type);
                return (
                  <li key={`${ev.chunk_id || ev.source_name}-${i}`}>
                    <button
                      type="button"
                      onClick={() => setSelectedIndex(i)}
                      className="w-full text-left px-4 py-3.5 hover:bg-white/[0.04] transition-colors"
                    >
                      <div className="flex items-start gap-3">
                        <span className="shrink-0 w-6 h-6 rounded-full bg-white/10 flex items-center justify-center text-[10px] font-bold text-slate-400 mt-px">
                          {i + 1}
                        </span>
                        <div className="flex-1 min-w-0 space-y-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-xs font-medium text-white/90 truncate">{ev.source_name}</span>
                            <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${badge.cls}`}>
                              {badge.label}
                            </span>
                          </div>
                          {(ev.page != null || ev.section) && (
                            <p className="text-[10px] text-slate-500">
                              {ev.page != null && `Page ${ev.page}`}
                              {ev.page != null && ev.section && " · "}
                              {ev.section}
                            </p>
                          )}
                          <p className="text-[11px] text-slate-400 line-clamp-2">{ev.matched_text}</p>
                        </div>
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default EvidenceSourcePreviewModal;
