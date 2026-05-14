import React, { useMemo } from "react";
import type { AssistantClassification, AssistantInsight } from "../../types";

const GROUP_ORDER: { key: AssistantClassification; title: string }[] = [
  { key: "warning", title: "Warnings" },
  { key: "contradicted", title: "Contradictions" },
  { key: "related", title: "Related context" },
  { key: "supported", title: "Supported claims" },
  { key: "missing_context", title: "Missing context" },
];

interface SessionInsightsReviewProps {
  insights: AssistantInsight[];
  onSelectInsight: (insight: AssistantInsight) => void;
}

const SessionInsightsReview: React.FC<SessionInsightsReviewProps> = ({ insights, onSelectInsight }) => {
  const grouped = useMemo(() => {
    const m = new Map<AssistantClassification, AssistantInsight[]>();
    for (const g of GROUP_ORDER) m.set(g.key, []);
    for (const ins of insights) {
      const bucket = m.get(ins.classification);
      if (bucket) bucket.push(ins);
    }
    return GROUP_ORDER.map(({ key, title }) => ({
      key,
      title,
      items: m.get(key) || [],
    })).filter((g) => g.items.length > 0);
  }, [insights]);

  if (grouped.length === 0) return null;

  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-3 mb-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 mb-2">Session insights</div>
      <div className="space-y-3 max-h-56 overflow-y-auto">
        {grouped.map((g) => (
          <div key={g.key}>
            <div className="text-xs font-medium text-slate-300 mb-1">{g.title}</div>
            <ul className="space-y-1">
              {g.items.map((ins) => (
                <li key={ins.id}>
                  <button
                    type="button"
                    onClick={() => onSelectInsight(ins)}
                    className="w-full text-left text-[11px] rounded-md px-2 py-1.5 text-slate-400 hover:bg-white/10 hover:text-slate-200 border border-transparent hover:border-white/10"
                  >
                    <span className="text-slate-500">{(ins.confidence * 100).toFixed(0)}%</span>
                    <span className="mx-1 text-slate-600">·</span>
                    <span className="text-slate-300 line-clamp-2">{ins.transcript_text}</span>
                    {ins.action_status && (
                      <span className="block text-[10px] text-slate-600 mt-0.5">Status: {ins.action_status.replace(/_/g, " ")}</span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SessionInsightsReview;
