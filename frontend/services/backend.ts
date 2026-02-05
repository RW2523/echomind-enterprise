export const API_BASE = (import.meta as any).env?.VITE_API_BASE || "";

/** docs */
export async function uploadDocument(file: File): Promise<{ok:boolean; doc_id?:string; chunks?:number}> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API_BASE}/api/docs/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`upload failed: ${r.status}`);
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
  // same-origin; nginx will proxy /api to backend if configured
  return `${proto}://${location.host}${API_BASE}/api/transcribe/ws`;
}
