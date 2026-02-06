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

export async function askChatStream(
  chatId: string,
  message: string,
  callbacks: AskChatStreamCallbacks
): Promise<void> {
  const r = await fetch(`${API_BASE}/api/chat/ask-stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ chat_id: chatId, message }),
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

/** transcription */
export async function polishTranscript(rawText: string): Promise<{polished: string}> {
  const r = await fetch(`${API_BASE}/api/transcribe/polish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ raw_text: rawText }),
  });
  if (!r.ok) throw new Error(`polish failed: ${r.status}`);
  return await r.json();
}

export async function storeTranscript(rawText: string, polishedText?: string|null): Promise<any> {
  const r = await fetch(`${API_BASE}/api/transcribe/store`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ raw_text: rawText, polished_text: polishedText ?? null }),
  });
  if (!r.ok) throw new Error(`store failed: ${r.status}`);
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
