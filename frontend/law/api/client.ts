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

/** A refusal said once, quietly (2026-11-12, evaluation fix list item 12): the server's 403 text read as an alarm in red. */
export const NOT_YOUR_ROLE = "This isn't part of your role. Ask a managing partner or your administrator if you need it.";

/** What to tell the person about a failed request: the server's own words, a list of field problems read as one sentence
 *  (a 422 used to say only "Something went wrong."), a refusal or a busy model in plain words. */
export function failureText(status: number, data: any, retryAfter?: string | null): string {
  // the policy's general refusal only; a refusal that says why (read-only access, your own draft) keeps its words
  if (status === 403 && (typeof data?.detail !== 'string' || /^You don.t have permission to do that\.?$/.test(data.detail))) return NOT_YOUR_ROLE;
  if (status === 429) {
    const s = Number(retryAfter);
    return `Too many questions in the last minute. Try again ${Number.isFinite(s) && s > 0 ? `in ${Math.ceil(s)} seconds` : 'shortly'}.`;
  }
  if (typeof data?.detail === 'string') return data.detail;
  if (status === 422 && Array.isArray(data?.detail)) {
    const long = data.detail.find((d: any) => d?.type === 'string_too_long');
    if (long) return `That is too long: at most ${long?.ctx?.max_length ?? 'the allowed number of'} characters.`;
    const short = data.detail.find((d: any) => d?.type === 'string_too_short');
    if (short) return 'Type a question first.';
    return 'Please check what you entered.';
  }
  return status === 422 ? 'Please check the highlighted fields.' : 'Something went wrong.';
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
  if (!r.ok) throw new ApiError(r.status, failureText(r.status, data, r.headers?.get?.('Retry-After')));
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
    throw new ApiError(r.status, failureText(r.status, data, r.headers?.get?.('Retry-After')));
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
