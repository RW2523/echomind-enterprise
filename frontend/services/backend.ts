export const API_BASE = (import.meta as any).env?.VITE_API_BASE || "";

/** docs */
export async function uploadDocument(file: File): Promise<{ok:boolean; doc_id?:string; chunks?:number}> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API_BASE}/api/docs/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
  return await r.json();
}

export interface DocListItem {
  id: string;
  filename: string;
  filetype: string;
  created_at: string;
}

export async function listDocuments(): Promise<{ documents: DocListItem[] }> {
  const r = await fetch(`${API_BASE}/api/docs/list`);
  if (!r.ok) throw new Error(`list docs failed: ${r.status}`);
  return await r.json();
}

export async function deleteDocument(docId: string): Promise<{ ok: boolean; deleted: string }> {
  const r = await fetch(`${API_BASE}/api/docs/${docId}`, { method: "DELETE" });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `delete failed: ${r.status}`);
  }
  return await r.json();
}

/** Vector DB storage usage (for sidebar). */
export interface StorageUsage {
  usage_bytes: number;
  capacity_bytes: number | null;
}

export async function getStorageUsage(): Promise<StorageUsage> {
  const r = await fetch(`${API_BASE}/api/docs/usage`);
  if (!r.ok) return { usage_bytes: 0, capacity_bytes: null };
  return await r.json();
}

/** Full data preview (documents, chunks, transcripts) for Usage popover. */
export interface DataPreview {
  documents: { id: string; filename: string; filetype: string; created_at: string; meta_json: string | null }[];
  chunks: { id: string; doc_id: string; chunk_index: number; text_preview: string }[];
  transcripts: { id: string; title: string; tags: string[]; echotag: string; created_at: string; raw_length: number; polished_length: number }[];
}

export async function getDataPreview(): Promise<DataPreview> {
  const r = await fetch(`${API_BASE}/api/docs/data-preview`);
  if (!r.ok) throw new Error(`data preview failed: ${r.status}`);
  return await r.json();
}

/** Delete all data (documents, chunks, transcripts, chats, messages). */
export async function deleteAllData(): Promise<{ ok: boolean; message: string }> {
  const r = await fetch(`${API_BASE}/api/docs/delete-all`, { method: "POST" });
  if (!r.ok) throw new Error(`delete all failed: ${r.status}`);
  return await r.json();
}

/** Add sample transcript chunks for testing RAG and time-range retrieval. */
export async function addSampleTranscripts(): Promise<{ ok: boolean; added: number; items: { id: string; label: string; tags: string[] }[] }> {
  const r = await fetch(`${API_BASE}/api/docs/add-sample-transcripts`, { method: "POST" });
  if (!r.ok) throw new Error(`add sample transcripts failed: ${r.status}`);
  return await r.json();
}

/** Transcripts list (for Knowledge Chat Transcripts panel). */
export interface TranscriptListItem {
  id: string;
  title: string;
  tags: string[];
  echotag: string;
  created_at: string;
  name?: string | null;
  location?: string | null;
}

/** Default transcript name: transcript_YYYY-MM-DD_HH-MM (local time). */
export function defaultTranscriptName(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  const h = String(now.getHours()).padStart(2, "0");
  const min = String(now.getMinutes()).padStart(2, "0");
  return `transcript_${y}-${m}-${d}_${h}-${min}`;
}

export async function listTranscripts(): Promise<{ transcripts: TranscriptListItem[] }> {
  const r = await fetch(`${API_BASE}/api/transcribe/list`);
  if (!r.ok) throw new Error(`list transcripts failed: ${r.status}`);
  return await r.json();
}

/** Get a single transcript by ID with full text (for preview popup). */
export interface TranscriptDetail {
  id: string;
  title: string;
  raw_text: string;
  polished_text: string;
  tags: string[];
  echotag: string;
  created_at: string;
  name?: string | null;
  location?: string | null;
}

export async function getTranscript(transcriptId: string): Promise<TranscriptDetail> {
  const r = await fetch(`${API_BASE}/api/transcribe/transcripts/${encodeURIComponent(transcriptId)}`);
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `get transcript failed: ${r.status}`);
  }
  return await r.json();
}

/** Delete a transcript from embeddings and all stored data. */
export async function deleteTranscript(transcriptId: string): Promise<{ ok: boolean; deleted: string }> {
  const r = await fetch(`${API_BASE}/api/transcribe/transcripts/${encodeURIComponent(transcriptId)}`, { method: "DELETE" });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `delete transcript failed: ${r.status}`);
  }
  return await r.json();
}

/** chat */
export async function createChat(title: string): Promise<{chat_id: string}> {
  const r = await fetch(`${API_BASE}/api/chat/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!r.ok) throw new Error(`create chat failed: ${r.status}`);
  return await r.json();
}

export async function askChat(chatId: string, message: string): Promise<{answer: string; citations: any[]}> {
  const r = await fetch(`${API_BASE}/api/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, message }),
  });
  if (!r.ok) throw new Error(`ask failed: ${r.status}`);
  return await r.json();
}

export type AskChatStreamCallbacks = {
  onChunk: (text: string) => void;
  onDone: (result: { answer: string; citations: any[] }) => void;
  onError?: (err: Error) => void;
};

export interface SourceOptions {
  transcript: boolean;
  document: boolean;
  general: boolean;
}

export async function askChatStream(
  chatId: string,
  message: string,
  callbacks: AskChatStreamCallbacks,
  options?: { persona?: string | null; context_window?: string | null; advanced_rag?: boolean; use_knowledge_base?: boolean; source_options?: SourceOptions }
): Promise<void> {
  const r = await fetch(`${API_BASE}/api/chat/ask-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chat_id: chatId,
      message,
      persona: options?.persona ?? undefined,
      context_window: options?.context_window ?? undefined,
      advanced_rag: options?.advanced_rag ?? undefined,
      use_knowledge_base: options?.use_knowledge_base ?? undefined,
      source_options: options?.source_options ?? undefined,
    }),
  });
  if (!r.ok) {
    const err = new Error(`ask stream failed: ${r.status}`);
    callbacks.onError?.(err);
    throw err;
  }
  const reader = r.body?.getReader();
  if (!reader) {
    const err = new Error("No response body");
    callbacks.onError?.(err);
    throw err;
  }
  const dec = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        const t = line.trim();
        if (!t) continue;
        try {
          const obj = JSON.parse(t);
          if (obj.type === "chunk" && obj.text != null) callbacks.onChunk(obj.text);
          else if (obj.type === "done") callbacks.onDone({ answer: obj.answer ?? "", citations: obj.citations ?? [] });
          else if (obj.type === "error") {
            callbacks.onError?.(new Error(obj.message ?? "Stream error"));
          }
        } catch (_) {}
      }
    }
    if (buf.trim()) {
      try {
        const obj = JSON.parse(buf.trim());
        if (obj.type === "chunk" && obj.text != null) callbacks.onChunk(obj.text);
        else if (obj.type === "done") callbacks.onDone({ answer: obj.answer ?? "", citations: obj.citations ?? [] });
      } catch (_) {}
    }
  } finally {
    reader.releaseLock();
  }
}

/** Refine transcript into clear, structured notes. */
export async function refineTranscript(rawText: string): Promise<{ refined: string }> {
  const r = await fetch(`${API_BASE}/api/transcribe/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ raw_text: rawText }),
  });
  if (!r.ok) throw new Error(`refine failed: ${r.status}`);
  return await r.json();
}

/** Preview possible tags and conversation type for transcript text. */
export async function getTranscriptTags(rawText: string): Promise<{ tags: string[]; conversation_type: string }> {
  const r = await fetch(`${API_BASE}/api/transcribe/tags`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ raw_text: rawText }),
  });
  if (!r.ok) throw new Error(`tags failed: ${r.status}`);
  return await r.json();
}

export async function storeTranscript(
  rawText: string,
  options?: {
    refinedText?: string | null;
    echotag?: string | null;
    name?: string | null;
    location?: string | null;
    tags?: string[] | null;
  }
): Promise<{ transcript_id: string; title: string; name?: string | null; location?: string | null; tags: string[]; echotag: string; echodate: string; created_at: string }> {
  const r = await fetch(`${API_BASE}/api/transcribe/store`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      raw_text: rawText,
      refined_text: options?.refinedText ?? null,
      echotag: options?.echotag ?? null,
      name: options?.name ?? null,
      location: options?.location ?? null,
      tags: options?.tags ?? null,
    }),
  });
  if (!r.ok) throw new Error(`store failed: ${r.status}`);
  return await r.json();
}

export async function updateTranscript(
  transcriptId: string,
  updates: { name?: string | null; location?: string | null; tags?: string[] | null }
): Promise<{ transcript_id: string; title: string; name?: string | null; location?: string | null; tags: string[]; echotag: string; created_at: string }> {
  const r = await fetch(`${API_BASE}/api/transcribe/transcripts/${encodeURIComponent(transcriptId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(updates),
  });
  if (!r.ok) throw new Error(`update transcript failed: ${r.status}`);
  return await r.json();
}

export function transcribeWsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}${API_BASE}/api/transcribe/ws`;
}

/** Voice conversation WebSocket (proxied to voice service at /voice/ws). */
export function voiceWsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/voice/ws`;
}

/** List Piper voice ids already installed on the voice server. */
export async function getInstalledVoices(): Promise<{ voice_ids: string[] }> {
  const r = await fetch(`${location.origin}/voice/voices/installed`);
  if (!r.ok) throw new Error(`voices/installed failed: ${r.status}`);
  return r.json();
}

/** Download a Piper voice by id (e.g. en_US-lessac-medium) on the voice server. */
export async function downloadVoice(voiceId: string): Promise<{ ok: boolean; voice_id: string }> {
  const r = await fetch(`${location.origin}/voice/voices/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ voice_id: voiceId }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error((err as { detail?: string }).detail || `download failed: ${r.status}`);
  }
  return r.json();
}
