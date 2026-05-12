import React from "react";

export type ModeStatusTone =
  | "neutral"
  | "listening"
  | "transcribing"
  | "thinking"
  | "speaking"
  | "suggestion"
  | "silent"
  | "muted"
  | "error";

const TONE_CLASS: Record<ModeStatusTone, string> = {
  neutral: "border-white/15 bg-white/[0.06] text-slate-300",
  listening: "border-emerald-500/35 bg-emerald-500/10 text-emerald-200",
  transcribing: "border-cyan-500/40 bg-cyan-500/10 text-cyan-200",
  thinking: "border-slate-500/35 bg-slate-600/15 text-slate-200",
  speaking: "border-amber-500/40 bg-amber-500/10 text-amber-200",
  suggestion: "border-violet-500/40 bg-violet-500/10 text-violet-200",
  silent: "border-slate-500/30 bg-slate-500/10 text-slate-200",
  muted: "border-rose-500/35 bg-rose-500/10 text-rose-200",
  error: "border-red-500/35 bg-red-500/10 text-red-200",
};

export interface ProductModeHeaderProps {
  title: string;
  tagline: string;
  /** Primary status phrase, e.g. "Listening" */
  status: string;
  statusTone?: ModeStatusTone;
  /** Extra compact chips (e.g. "Muted") */
  extraStatuses?: { label: string; tone?: ModeStatusTone }[];
  /** Session / transcript display name */
  sessionName?: string | null;
  /** Show knowledge row when true */
  showKnowledge?: boolean;
  knowledgeEnabled?: boolean;
  /** Output behavior one-liner */
  outputHint: string;
  rightSlot?: React.ReactNode;
  className?: string;
}

const ProductModeHeader: React.FC<ProductModeHeaderProps> = ({
  title,
  tagline,
  status,
  statusTone = "neutral",
  extraStatuses = [],
  sessionName,
  showKnowledge = false,
  knowledgeEnabled = false,
  outputHint,
  rightSlot,
  className = "",
}) => {
  return (
    <div
      className={`shrink-0 px-4 py-3 sm:px-5 border-b border-white/10 bg-black/[0.12] ${className}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h1 className="text-lg font-semibold text-white tracking-tight">{title}</h1>
          <p className="text-xs text-slate-500 mt-0.5 max-w-xl leading-relaxed">{tagline}</p>
          <p className="text-[11px] text-slate-500/90 mt-1.5 border-l border-cyan-500/25 pl-2">{outputHint}</p>
        </div>
        {rightSlot ? <div className="flex flex-wrap items-center gap-2 shrink-0 justify-end">{rightSlot}</div> : null}
      </div>
      <div className="mt-2.5 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-0.5 font-medium border ${TONE_CLASS[statusTone]}`}
        >
          {status}
        </span>
        {extraStatuses.map((e, i) => (
          <span
            key={`${e.label}-${i}`}
            className={`inline-flex items-center rounded-full px-2.5 py-0.5 font-medium border ${TONE_CLASS[e.tone ?? "neutral"]}`}
          >
            {e.label}
          </span>
        ))}
        {sessionName ? (
          <span className="text-slate-500">
            Session: <span className="text-slate-300 font-mono truncate max-w-[200px] inline-block align-bottom">{sessionName}</span>
          </span>
        ) : null}
        {showKnowledge ? (
          <span className={knowledgeEnabled ? "text-teal-300/90" : "text-slate-500"}>
            Knowledge base: {knowledgeEnabled ? "On" : "Off"}
          </span>
        ) : null}
      </div>
    </div>
  );
};

export default ProductModeHeader;
