import { getActiveNamespace } from '../packs';

export const API_BASE = (import.meta as any).env?.VITE_API_BASE || "";

/** auth (Phase 0b) — cookie-based; same-origin requests carry the httpOnly session automatically. */
export interface AuthUser { id: string; username: string; role: string; tenant?: string; }

export async function getAuthConfig(): Promise<{ auth_enabled: boolean }> {
  try {
    const r = await fetch(`${API_BASE}/api/auth/config`);
    if (!r.ok) return { auth_enabled: false };
    return await r.json();
  } catch (_) { return { auth_enabled: false }; }
}

export async function getMe(): Promise<AuthUser | null> {
  try {
    const r = await fetch(`${API_BASE}/api/auth/me`);
    if (!r.ok) return null;
    const d = await r.json();
    return (d.user ?? null) as AuthUser | null;
  } catch (_) { return null; }
}

export async function login(username: string, password: string): Promise<AuthUser> {
  const r = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!r.ok) {
    let msg = "Login failed";
    try { const e = await r.json(); msg = e.detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  const d = await r.json();
  return d.user as AuthUser;
}

export async function logout(): Promise<void> {
  try { await fetch(`${API_BASE}/api/auth/logout`, { method: "POST" }); } catch (_) {}
}

export async function listUsers(): Promise<AuthUser[]> {
  const r = await fetch(`${API_BASE}/api/auth/users`);
  if (!r.ok) throw new Error(`list users failed: ${r.status}`);
  const d = await r.json();
  return (d.users ?? []) as AuthUser[];
}

export async function createUser(username: string, password: string, role: string, tenant: string = ""): Promise<AuthUser> {
  const r = await fetch(`${API_BASE}/api/auth/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, role, tenant }),
  });
  if (!r.ok) {
    let msg = "Create failed";
    try { const e = await r.json(); msg = e.detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return (await r.json()).user as AuthUser;
}

export async function deleteUserAccount(username: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/auth/users/${encodeURIComponent(username)}`, { method: "DELETE" });
  if (!r.ok) {
    let msg = "Delete failed";
    try { const e = await r.json(); msg = e.detail || msg; } catch (_) {}
    throw new Error(msg);
  }
}

export interface ActivityEvent { ts: string; username: string; role: string; method: string; path: string; status: number; ip: string; }
export async function getAudit(): Promise<ActivityEvent[]> {
  const r = await fetch(`${API_BASE}/api/auth/audit`);
  if (!r.ok) throw new Error(`audit failed: ${r.status}`);
  return ((await r.json()).events ?? []) as ActivityEvent[];
}

export interface UsageSummary { total_events: number; by_user: { username: string; events: number; last: string }[]; }
export async function getUsage(): Promise<UsageSummary> {
  const r = await fetch(`${API_BASE}/api/auth/usage`);
  if (!r.ok) throw new Error(`usage failed: ${r.status}`);
  return await r.json() as UsageSummary;
}

export interface ExportEvaluation {
  risk_level: string;
  finding_count: number;
  findings: { type: string; severity: string; preview: string }[];
  redacted_text: string;
  safe_to_export: boolean;
}
export async function evaluateExport(text: string): Promise<ExportEvaluation> {
  const r = await fetch(`${API_BASE}/api/export/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`export evaluate failed: ${r.status}`);
  return await r.json() as ExportEvaluation;
}

/** docs */
export async function uploadDocument(file: File): Promise<{ok:boolean; doc_id?:string; chunks?:number}> {
  const fd = new FormData();
  fd.append("file", file);
  const ns = getActiveNamespace();
  if (ns) fd.append("namespace", ns);
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
export const DEFAULT_CHAT_TITLE = "EchoMind Chat";

export interface ChatListItem {
  id: string;
  title: string;
  created_at: string;
}

export async function listChats(): Promise<{ chats: ChatListItem[] }> {
  const r = await fetch(`${API_BASE}/api/chat/list`);
  if (!r.ok) throw new Error(`list chats failed: ${r.status}`);
  return await r.json();
}

export interface ChatMessageFromApi {
  id: string;
  role: string;
  content: string;
  created_at: string;
}

export interface ChatDetail {
  id: string;
  title: string;
  created_at: string;
  messages: ChatMessageFromApi[];
}

export async function getChat(chatId: string): Promise<ChatDetail> {
  const r = await fetch(`${API_BASE}/api/chat/${encodeURIComponent(chatId)}`);
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `get chat failed: ${r.status}`);
  }
  return await r.json();
}

export async function deleteChat(chatId: string): Promise<{ ok: boolean; deleted: string }> {
  const r = await fetch(`${API_BASE}/api/chat/${encodeURIComponent(chatId)}`, { method: "DELETE" });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `delete chat failed: ${r.status}`);
  }
  return await r.json();
}

export async function createChat(title: string): Promise<{chat_id: string}> {
  const r = await fetch(`${API_BASE}/api/chat/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: title || DEFAULT_CHAT_TITLE }),
  });
  if (!r.ok) throw new Error(`create chat failed: ${r.status}`);
  return await r.json();
}

export async function askChat(chatId: string, message: string): Promise<{answer: string; citations: any[]}> {
  const r = await fetch(`${API_BASE}/api/chat/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, message, namespace: getActiveNamespace() || undefined }),
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
  options?: { persona?: string | null; context_window?: string | null; advanced_rag?: boolean; use_knowledge_base?: boolean; source_options?: SourceOptions; signal?: AbortSignal }
): Promise<void> {
  const r = await fetch(`${API_BASE}/api/chat/ask-stream`, {
    method: "POST",
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    signal: options?.signal,  // allow the caller to abort the stream on unmount/new-send (M20)
    body: JSON.stringify({
      chat_id: chatId,
      message,
      persona: options?.persona ?? undefined,
      context_window: options?.context_window ?? undefined,
      advanced_rag: options?.advanced_rag ?? undefined,
      use_knowledge_base: options?.use_knowledge_base ?? undefined,
      source_options: options?.source_options ?? undefined,
      namespace: getActiveNamespace() || undefined,
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
  // Track whether the server sent a terminal event. If the stream ends (or the network
  // drops) before a "done"/"error" line, we must surface an error instead of silently
  // presenting a truncated answer as if it were complete. (H9)
  let sawTerminal = false;
  const handleLine = (t: string) => {
    if (!t) return;
    try {
      const obj = JSON.parse(t);
      if (obj.type === "chunk" && obj.text != null) callbacks.onChunk(obj.text);
      else if (obj.type === "done") {
        sawTerminal = true;
        callbacks.onDone({ answer: obj.answer ?? "", citations: obj.citations ?? [] });
      } else if (obj.type === "error") {
        sawTerminal = true;
        callbacks.onError?.(new Error(obj.message ?? "Stream error"));
      }
    } catch (_) {}
  };
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) handleLine(line.trim());
    }
    if (buf.trim()) handleLine(buf.trim());
    // Clean EOF but the server never sent a terminal event -> truncated/interrupted stream.
    if (!sawTerminal) {
      callbacks.onError?.(new Error("Stream ended unexpectedly before completion"));
    }
  } catch (readErr) {
    // Reader threw mid-stream (e.g. network drop). Route to onError so the UI can recover.
    if (!sawTerminal) {
      callbacks.onError?.(readErr instanceof Error ? readErr : new Error(String(readErr)));
    }
  } finally {
    try { reader.releaseLock(); } catch (_) {}
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

/** Fetch all analysis cards for a stored transcript. */
export async function getTranscriptAnalysis(transcriptId: string): Promise<{ cards: import('../types').AnalysisCard[] }> {
  const r = await fetch(`${API_BASE}/api/transcribe/transcripts/${encodeURIComponent(transcriptId)}/analysis`);
  if (!r.ok) throw new Error(`analysis fetch failed: ${r.status}`);
  return r.json();
}

/** Fetch source chunk text for an analysis card source preview. */
export async function getChunkPreview(chunkId: string): Promise<{ chunk_id: string; text: string; doc_title: string; doc_id: string }> {
  const r = await fetch(`${API_BASE}/api/transcribe/chunks/${encodeURIComponent(chunkId)}/preview`);
  if (!r.ok) throw new Error(`chunk preview failed: ${r.status}`);
  return r.json();
}

/** Request TTS audio for a card or summary. Returns audio_b64 (WAV) or null (use browser TTS). */
export async function speakText(
  mode: 'card' | 'summary',
  options: { text?: string; cards?: import('../types').AnalysisCard[]; transcript_id?: string }
): Promise<{ audio_b64: string | null; format: string | null; text: string }> {
  const r = await fetch(`${API_BASE}/api/transcribe/speak`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode,
      text: options.text ?? null,
      cards: options.cards
        ? options.cards.map(c => ({ label: c.label, segment_text: c.segment_text, explanation: c.explanation }))
        : null,
      transcript_id: options.transcript_id ?? null,
    }),
  });
  if (!r.ok) throw new Error(`speak failed: ${r.status}`);
  return r.json();
}

// ── Boardroom API ─────────────────────────────────────────────────────────────

export async function createBoardroomSession(transcriptId?: string): Promise<{ session_id: string; status: string }> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript_id: transcriptId ?? null }),
  });
  if (!r.ok) throw new Error(`create boardroom session failed: ${r.status}`);
  return r.json();
}

export async function uploadBoardroomChunk(
  sessionId: string,
  chunkIndex: number,
  blob: Blob,
  audioFormat = 'webm'
): Promise<{ ok: boolean; chunk_count: number }> {
  const fd = new FormData();
  fd.append('file', blob, `chunk_${chunkIndex}.${audioFormat}`);
  const r = await fetch(
    `${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}/chunks?chunk_index=${chunkIndex}&audio_format=${audioFormat}`,
    { method: 'POST', body: fd }
  );
  if (!r.ok) throw new Error(`chunk upload failed: ${r.status}`);
  return r.json();
}

export async function finalizeBoardroomSession(sessionId: string): Promise<{ ok: boolean; status: string }> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}/finalize`, {
    method: 'POST',
  });
  if (!r.ok) throw new Error(`finalize failed: ${r.status}`);
  return r.json();
}

export async function analyseBoardroomSession(sessionId: string): Promise<{ ok: boolean; status: string }> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}/analyse`, {
    method: 'POST',
  });
  if (!r.ok) throw new Error(`analyse failed: ${r.status}`);
  return r.json();
}

export async function getBoardroomSession(sessionId: string): Promise<import('../types').BoardroomSession> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}`);
  if (!r.ok) throw new Error(`get boardroom session failed: ${r.status}`);
  return r.json();
}

export interface BoardroomSessionListItem {
  id: string;
  transcript_id: string | null;
  status: string;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export async function listBoardroomSessions(limit = 20): Promise<{ sessions: BoardroomSessionListItem[] }> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions?limit=${limit}`);
  if (!r.ok) throw new Error(`list boardroom sessions failed: ${r.status}`);
  return r.json();
}

export async function deleteBoardroomSession(sessionId: string): Promise<{ ok: boolean; deleted: string }> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail || `delete boardroom session failed: ${r.status}`);
  }
  return r.json();
}

export async function linkBoardroomTranscript(sessionId: string, transcriptId: string): Promise<void> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}/link-transcript`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ transcript_id: transcriptId }),
  });
  if (!r.ok) throw new Error(`link boardroom transcript failed: ${r.status}`);
}

export function boardroomExportUrl(sessionId: string, format: 'pdf' | 'pptx'): string {
  return `${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}/export?format=${format}`;
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
