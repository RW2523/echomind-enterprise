import type { AssistantClassification, AssistantEvidenceSourceType, AssistantInsight } from "../../types";

export type AnalysisUiStatus = "idle" | "listening" | "checking" | "found" | "none";

export function newSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `sess_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

export function rankInsight(ins: AssistantInsight): number {
  const cls = ins.classification;
  const base =
    cls === "contradicted" || cls === "warning"
      ? 400
      : cls === "supported"
        ? 200
        : cls === "related"
          ? 100
          : 50;
  return base + ins.confidence * 100;
}

/** Hand Raise queue: warning → contradicted → related → supported (per PRD). */
export function handRaiseTier(ins: AssistantInsight): number {
  switch (ins.classification) {
    case "warning":
      return 4;
    case "contradicted":
      return 3;
    case "related":
      return 2;
    case "supported":
      return 1;
    default:
      return 0;
  }
}

export function buildHighlightSpans(
  text: string,
  insights: AssistantInsight[]
): { text: string; insight?: AssistantInsight; key: string }[] {
  const list = insights.filter((i) => i.show_highlight && (i.transcript_text || "").trim().length >= 4);
  if (!text || list.length === 0) return [{ text, key: "all" }];

  type Span = { start: number; end: number; insight: AssistantInsight };
  const spans: Span[] = [];
  for (const ins of list) {
    const q = ins.transcript_text.trim();
    let from = 0;
    while (from < text.length) {
      const idx = text.indexOf(q, from);
      if (idx < 0) break;
      spans.push({ start: idx, end: idx + q.length, insight: ins });
      from = idx + Math.max(1, q.length);
    }
  }
  if (spans.length === 0) return [{ text, key: "all" }];

  spans.sort((a, b) => a.start - b.start || b.end - a.end);
  const merged: Span[] = [];
  for (const s of spans) {
    const last = merged[merged.length - 1];
    if (!last || s.start >= last.end) {
      merged.push({ ...s });
      continue;
    }
    if (rankInsight(s.insight) > rankInsight(last.insight)) {
      merged[merged.length - 1] = {
        start: Math.min(last.start, s.start),
        end: Math.max(last.end, s.end),
        insight: s.insight,
      };
    }
  }

  const out: { text: string; insight?: AssistantInsight; key: string }[] = [];
  let pos = 0;
  let ki = 0;
  for (const s of merged) {
    if (s.start > pos) out.push({ text: text.slice(pos, s.start), key: `p${ki++}` });
    out.push({ text: text.slice(s.start, s.end), insight: s.insight, key: `h${ki++}` });
    pos = s.end;
  }
  if (pos < text.length) out.push({ text: text.slice(pos), key: `p${ki++}` });
  return out;
}

export function classificationHighlightClass(c: AssistantClassification): string {
  switch (c) {
    case "supported":
      return "bg-emerald-500/25 text-emerald-100 border border-emerald-500/40";
    case "contradicted":
      return "bg-red-500/25 text-red-100 border border-red-500/45";
    case "warning":
      return "bg-orange-500/20 text-orange-100 border border-orange-500/40";
    case "related":
      return "bg-amber-500/20 text-amber-50 border border-amber-500/35";
    case "missing_context":
    default:
      return "bg-slate-600/20 text-slate-200 border border-dashed border-white/25";
  }
}

export function classificationBadgeClass(c: AssistantClassification): string {
  return `text-xs font-medium px-2 py-0.5 rounded-md capitalize ${classificationHighlightClass(c)}`;
}

export function evidenceSourceTypeBadge(
  sourceType: AssistantEvidenceSourceType | string
): { label: string; cls: string } {
  const map: Record<string, { label: string; cls: string }> = {
    document: { label: "Document", cls: "bg-cyan-500/15 text-cyan-300" },
    transcript: { label: "Transcript", cls: "bg-amber-500/15 text-amber-300" },
    book: { label: "Book", cls: "bg-violet-500/15 text-violet-300" },
    faq: { label: "FAQ", cls: "bg-sky-500/15 text-sky-300" },
    unknown: { label: "Source", cls: "bg-white/10 text-slate-300" },
  };
  return map[String(sourceType).toLowerCase()] ?? map.unknown;
}

export const ROLLING_CHARS = 9000;

export function assistantInsightsMetaStorageKey(mode: "silent_assistant" | "personal_assistant"): string {
  return `echomind_assistant_insights_meta_v1_${mode}`;
}
