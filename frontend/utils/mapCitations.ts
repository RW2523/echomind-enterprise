import type { DocumentChunk } from "../types";

const CITATION_DEBUG = (import.meta as { env?: Record<string, string> }).env?.VITE_CITATION_DEBUG === "1";

/** Map backend RAG citation dicts to `DocumentChunk` (same shape as Knowledge Chat stream). */
export function mapCitations(citations: unknown[] | null | undefined): DocumentChunk[] {
  if (CITATION_DEBUG) {
    console.log("[Citations] mapCitations input:", { raw: citations, count: (citations || []).length });
  }
  return (citations || []).map((c: unknown, i: number) => {
    const x = c as Record<string, unknown>;
    return {
      id: `cite_${i}_${(x?.filename as string) ?? "doc"}`,
      docName: (x?.filename as string) ?? "Unknown document",
      content: (x?.snippet as string) ?? "",
      metadata: {
        section: (x?.section as string) ?? undefined,
        sectionPath: (x?.section_path as string) ?? undefined,
        pageNumber: typeof x?.page_number === "number" ? x.page_number : undefined,
        score: typeof x?.score === "number" ? x.score : undefined,
        docType: (x?.doc_type as string) ?? undefined,
        chunkIndex: typeof x?.chunk_index === "number" ? x.chunk_index : undefined,
        docId: (x?.doc_id as string) ?? undefined,
        timestamp: Date.now(),
      },
    };
  });
}
