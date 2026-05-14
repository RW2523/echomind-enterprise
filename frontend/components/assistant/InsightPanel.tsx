import React, { useState } from "react";
import type { AssistantClassification, AssistantEvidence, AssistantInsight } from "../../types";
import { classificationBadgeClass, evidenceSourceTypeBadge } from "./assistantUtils";
import EvidenceSourcePreviewModal from "./EvidenceSourcePreviewModal";

interface InsightPanelProps {
  selectedInsight: AssistantInsight | null;
  showSuggestedResponse?: boolean;
  emptyHint?: string;
}

const InsightPanel: React.FC<InsightPanelProps> = ({
  selectedInsight,
  showSuggestedResponse,
  emptyHint = "Click a highlighted phrase in the transcript to see evidence and interpretation.",
}) => {
  const [previewEvidence, setPreviewEvidence] = useState<AssistantEvidence[] | null>(null);
  const [previewInitialIndex, setPreviewInitialIndex] = useState(0);

  const badge = (c: AssistantClassification) => (
    <span className={classificationBadgeClass(c)}>{c.replace("_", " ")}</span>
  );

  const openPreview = (evidence: AssistantEvidence[], index = 0) => {
    if (!evidence.length) return;
    setPreviewInitialIndex(index);
    setPreviewEvidence(evidence);
  };

  return (
    <>
      <div className="flex-1 min-h-0 overflow-y-auto p-4 text-sm">
        {!selectedInsight && <p className="text-slate-500">{emptyHint}</p>}
        {selectedInsight && (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              {badge(selectedInsight.classification)}
              <span className="text-slate-400">
                Confidence:{" "}
                <span className="text-white font-medium">{(selectedInsight.confidence * 100).toFixed(0)}%</span>
              </span>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-500 mb-1">Transcript excerpt</div>
              <p className="text-slate-200 bg-white/5 rounded-lg p-2 border border-white/10">{selectedInsight.transcript_text}</p>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-500 mb-1">Interpretation</div>
              <p className="text-slate-300">{selectedInsight.assistant_interpretation || "—"}</p>
            </div>
            <div>
              <div className="text-[10px] uppercase text-slate-500 mb-1">Suggested action</div>
              <p className="text-slate-300">{selectedInsight.suggested_action || "—"}</p>
            </div>
            {showSuggestedResponse && selectedInsight.suggested_response && (
              <div>
                <div className="text-[10px] uppercase text-slate-500 mb-1">Suggested talking point</div>
                <p className="text-cyan-200/90 bg-cyan-500/10 rounded-lg p-2 border border-cyan-500/20 italic">
                  {selectedInsight.suggested_response}
                </p>
              </div>
            )}
            <div>
              <div className="flex items-center justify-between gap-2 mb-2">
                <div className="text-[10px] uppercase text-slate-500">Referred resources</div>
                {(selectedInsight.evidence || []).length > 1 && (
                  <button
                    type="button"
                    onClick={() => openPreview(selectedInsight.evidence || [], -1)}
                    className="text-[10px] font-medium text-cyan-400 hover:text-cyan-300"
                  >
                    Preview all
                  </button>
                )}
              </div>
              <ul className="space-y-2">
                {(selectedInsight.evidence || []).map((ev, i) => {
                  const typeBadge = evidenceSourceTypeBadge(ev.source_type);
                  return (
                    <li key={`${ev.chunk_id || ev.source_name}-${i}`} className="rounded-lg border border-white/10 bg-white/5 p-2 text-xs">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-medium text-cyan-200/90 break-all">{ev.source_name}</span>
                            <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${typeBadge.cls}`}>
                              {typeBadge.label}
                            </span>
                          </div>
                          {(ev.page != null || ev.section) && (
                            <div className="text-slate-500 mt-0.5">
                              {ev.page != null && <span>Page {ev.page}</span>}
                              {ev.page != null && ev.section && " · "}
                              {ev.section && <span>{ev.section}</span>}
                            </div>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={() => openPreview(selectedInsight.evidence || [], i)}
                          className="shrink-0 rounded-md px-2 py-1 text-[10px] font-medium bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/25"
                        >
                          Preview
                        </button>
                      </div>
                      <p className="text-slate-400 mt-1 whitespace-pre-wrap line-clamp-4">{ev.matched_text}</p>
                    </li>
                  );
                })}
              </ul>
            </div>
            <p className="text-[11px] text-slate-500 border-t border-white/10 pt-3 mt-2">Verify in source before acting.</p>
          </div>
        )}
      </div>
      {previewEvidence && previewEvidence.length > 0 && (
        <EvidenceSourcePreviewModal
          evidence={previewEvidence}
          initialIndex={previewInitialIndex}
          onClose={() => setPreviewEvidence(null)}
        />
      )}
    </>
  );
};

export default InsightPanel;
