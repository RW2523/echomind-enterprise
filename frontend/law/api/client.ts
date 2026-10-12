/**
 * Minimal typed client for the Law API. Same-origin only: the session cookie travels automatically
 * and services/session.ts adds the CSRF header on writes. Errors carry the server's status and its
 * human-readable `detail`, which the UI shows as-is (the server never reveals hidden matters).
 */
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const form = typeof FormData !== 'undefined' && body instanceof FormData;
  const r = await fetch(path, {
    method,
    credentials: 'same-origin',
    // A FormData body sets its own multipart Content-Type (with the boundary).
    headers: body === undefined || form ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : form ? (body as FormData) : JSON.stringify(body),
  });
  let data: any = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    const detail = typeof data?.detail === 'string' ? data.detail : r.status === 422 ? 'Please check the highlighted fields.' : 'Something went wrong.';
    throw new ApiError(r.status, detail);
  }
  return data as T;
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body ?? {}),
  patch: <T>(path: string, body: unknown) => request<T>('PATCH', path, body),
  put: <T>(path: string, body: unknown) => request<T>('PUT', path, body),
  del: <T>(path: string) => request<T>('DELETE', path),
  upload: <T>(path: string, form: FormData) => request<T>('POST', path, form),
};

/** POST that answers with NDJSON (one JSON object per line); calls onEvent for each as it arrives. */
export async function postStream(path: string, body: unknown, onEvent: (e: any) => void, signal?: AbortSignal): Promise<void> {
  const r = await fetch(path, {
    method: 'POST', credentials: 'same-origin', signal,
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!r.ok) {
    let data: any = null;
    try { data = await r.json(); } catch { /* empty */ }
    throw new ApiError(r.status, typeof data?.detail === 'string' ? data.detail : 'Something went wrong.');
  }
  const emit = (line: string) => { if (line.trim()) onEvent(JSON.parse(line)); };
  if (!r.body) { (await r.text()).split('\n').forEach(emit); return; }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf('\n')) >= 0) { emit(buf.slice(0, i)); buf = buf.slice(i + 1); }
  }
  emit(buf);
}

async function failure(r: Response): Promise<never> {
  let data: any = null;
  try { data = await r.json(); } catch { /* empty */ }
  throw new ApiError(r.status, typeof data?.detail === 'string' ? data.detail : 'Something went wrong.');
}

/** POST raw bytes, JSON back (S5.1: a spoken question as 16-bit PCM). */
export async function postBytes<T>(path: string, bytes: ArrayBuffer): Promise<T> {
  const r = await fetch(path, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/octet-stream' }, body: bytes });
  if (!r.ok) return failure(r);
  return (await r.json()) as T;
}

/** POST JSON, a file back (S5.1: an answer read aloud, as WAV). */
export async function postForBlob(path: string, body: unknown): Promise<Blob> {
  const r = await fetch(path, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) return failure(r);
  return r.blob();
}
