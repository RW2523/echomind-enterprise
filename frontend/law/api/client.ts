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
  const r = await fetch(path, {
    method,
    credentials: 'same-origin',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
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
};
