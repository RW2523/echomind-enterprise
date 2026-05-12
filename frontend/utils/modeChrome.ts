/** Shared UI tokens for product modes (citations, evidence). */

import type { SilentFindingStatusLabel } from "../types";

/** Citation / source chip — use with `className` on buttons opening ChunkCitationModal. */
export const CITATION_CHIP_CLASS =
  "max-w-full rounded-lg border border-teal-500/25 bg-teal-950/35 px-2 py-1 text-left text-[11px] text-teal-100/90 hover:bg-teal-900/45 hover:border-teal-400/35 transition-colors truncate";

export function evidenceLabel(ev: string): string {
  switch (ev) {
    case "grounded":
      return "Grounded";
    case "partial":
      return "Partial";
    case "weak":
      return "Weak";
    case "none":
    default:
      return "No Evidence";
  }
}

/** Silent Assistant: four user-facing status labels (maps internal enum values). */
export type SilentAssistantDisplayStatus = "Supported" | "Contradicted" | "Unverified" | "Needs Review";

export function silentAssistantDisplayStatus(sl: SilentFindingStatusLabel): SilentAssistantDisplayStatus {
  if (sl === "likely_correct" || sl === "suggestion_available") return "Supported";
  if (sl === "contradicted" || sl === "possibly_wrong") return "Contradicted";
  if (sl === "unsupported") return "Unverified";
  return "Needs Review";
}

/** Merge order for overlapping highlights (higher wins). */
export function silentAssistantDisplaySeverity(sl: SilentFindingStatusLabel): number {
  switch (silentAssistantDisplayStatus(sl)) {
    case "Supported":
      return 0;
    case "Needs Review":
      return 2;
    case "Unverified":
      return 3;
    case "Contradicted":
      return 4;
    default:
      return 0;
  }
}

/** How a suggestion/finding was grounded (EchoMind: KB vs transcript; legacy values still labeled). */
export function sourceOriginLabel(origin: string): string {
  switch (origin) {
    case "rag":
    case "transcript_plus_rag":
      return "Knowledge base";
    case "transcript":
      return "Transcript";
    case "rules":
    case "rules_plus_rag":
      return "—";
    case "notes":
    case "transcript_plus_notes":
    case "notes_plus_rag":
      return "—";
    case "none":
    default:
      return "—";
  }
}

/** Silent finding list: short provenance chip (KB-only product). */
export function silentFindingSourceChip(origin: string): string {
  if (origin === "transcript_plus_rag" || origin === "rag") return "Knowledge base";
  if (origin === "transcript") return "Transcript";
  return "—";
}
