import React from "react";
import type { AssistantInsight, AssistantClassification } from "../../types";
import { buildHighlightSpans, classificationHighlightClass } from "./assistantUtils";

interface AssistantTranscriptPanelProps {
  fullTranscript: string;
  insights: AssistantInsight[];
  onHighlightClick: (insight: AssistantInsight) => void;
}

const AssistantTranscriptPanel: React.FC<AssistantTranscriptPanelProps> = ({
  fullTranscript,
  insights,
  onHighlightClick,
}) => {
  const segments = React.useMemo(() => buildHighlightSpans(fullTranscript, insights), [fullTranscript, insights]);

  return (
    <div className="flex-1 min-h-0 rounded-xl border border-white/10 bg-black/30 p-3 overflow-y-auto">
      <div className="text-sm text-slate-200 whitespace-pre-wrap leading-relaxed">
        {segments.map((seg) =>
          seg.insight ? (
            <button
              key={seg.key}
              type="button"
              onClick={() => onHighlightClick(seg.insight!)}
              className={`inline text-left rounded px-0.5 mx-0.5 cursor-pointer transition-opacity hover:opacity-90 ${classificationHighlightClass(
                seg.insight.classification as AssistantClassification
              )}`}
            >
              {seg.text}
            </button>
          ) : (
            <span key={seg.key}>{seg.text}</span>
          )
        )}
        {!fullTranscript && <span className="text-slate-500">Transcript appears here…</span>}
      </div>
    </div>
  );
};

export default AssistantTranscriptPanel;
