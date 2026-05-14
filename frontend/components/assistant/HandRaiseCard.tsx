import React from "react";
import type { AssistantInsight, HandRaiseAction } from "../../types";
import { classificationBadgeClass, evidenceSourceTypeBadge } from "./assistantUtils";

interface HandRaiseCardProps {
  insight: AssistantInsight;
  onAction: (action: HandRaiseAction) => void;
  /** When false, Ask Follow-up is disabled (e.g. app did not wire navigation to Knowledge Chat). */
  askFollowUpEnabled?: boolean;
  /** While TTS is loading or playing, disables Speak Now to avoid overlapping requests. */
  speakNowBusy?: boolean;
  onPreviewSource?: () => void;
}

const HandRaiseCard: React.FC<HandRaiseCardProps> = ({
  insight,
  onAction,
  askFollowUpEnabled = true,
  speakNowBusy = false,
  onPreviewSource,
}) => {
  const primaryEvidence = insight.evidence?.[0];
  const preview = (primaryEvidence?.matched_text || "").slice(0, 160);
  const reason = insight.assistant_interpretation || insight.suggested_action || "Review the match with your knowledge base.";
  const sourceBadge = primaryEvidence ? evidenceSourceTypeBadge(primaryEvidence.source_type) : null;

  return (
    <div className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-4 shadow-lg shadow-amber-500/5">
      <div className="text-xs font-semibold text-amber-200/95 tracking-wide uppercase mb-2">EchoMind has a suggestion</div>
      <div className="flex flex-wrap items-center gap-2 mb-2">
        <span className={classificationBadgeClass(insight.classification)}>{insight.classification.replace("_", " ")}</span>
        <span className="text-xs text-slate-400">{(insight.confidence * 100).toFixed(0)}% confidence</span>
      </div>
      <p className="text-sm text-slate-200 mb-2 line-clamp-3">{reason}</p>
      {primaryEvidence && (
        <div className="text-[11px] text-slate-500 border-t border-white/10 pt-2 mb-3 space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-slate-400">Referred resource:</span>
            <span className="text-slate-300 break-all">{primaryEvidence.source_name}</span>
            {sourceBadge && (
              <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${sourceBadge.cls}`}>
                {sourceBadge.label}
              </span>
            )}
          </div>
          {preview && (
            <p className="line-clamp-2">
              <span className="text-slate-400">Source preview: </span>
              {preview}
              {(primaryEvidence.matched_text?.length || 0) > 160 ? "…" : ""}
            </p>
          )}
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => onAction("view_details")}
          className="rounded-lg px-3 py-1.5 text-xs font-medium bg-cyan-500/20 text-cyan-200 border border-cyan-500/35"
        >
          View Details
        </button>
        {primaryEvidence && onPreviewSource && (
          <button
            type="button"
            onClick={onPreviewSource}
            className="rounded-lg px-3 py-1.5 text-xs font-medium bg-white/10 text-slate-200 border border-white/15"
          >
            Preview source
          </button>
        )}
        <button
          type="button"
          onClick={() => onAction("save_for_later")}
          className="rounded-lg px-3 py-1.5 text-xs font-medium bg-white/10 text-slate-200 border border-white/15"
        >
          Save for Later
        </button>
        <button
          type="button"
          onClick={() => onAction("ignore")}
          className="rounded-lg px-3 py-1.5 text-xs font-medium bg-white/5 text-slate-400 border border-white/10"
        >
          Ignore
        </button>
        <button
          type="button"
          onClick={() => onAction("ask_follow_up")}
          disabled={!askFollowUpEnabled}
          title={!askFollowUpEnabled ? "Knowledge Chat follow-up is not wired in this build" : undefined}
          className={`rounded-lg px-3 py-1.5 text-xs font-medium border ${
            askFollowUpEnabled
              ? "bg-violet-500/15 text-violet-200 border-violet-500/30"
              : "bg-slate-800/40 text-slate-500 border-white/10 cursor-not-allowed"
          }`}
        >
          Ask Follow-up
        </button>
        <button
          type="button"
          disabled={speakNowBusy}
          onClick={() => onAction("speak_now")}
          title="Speak the suggested talking point or interpretation aloud (uses your Settings voice)"
          className={`rounded-lg px-3 py-1.5 text-xs font-medium border ${
            speakNowBusy
              ? "bg-slate-700/40 text-slate-500 border-white/10 cursor-wait"
              : "bg-emerald-500/15 text-emerald-200 border-emerald-500/35 hover:bg-emerald-500/25"
          }`}
        >
          {speakNowBusy ? "Preparing / playing…" : "Speak Now"}
        </button>
      </div>
    </div>
  );
};

export default HandRaiseCard;
