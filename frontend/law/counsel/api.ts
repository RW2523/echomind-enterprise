/** Typed calls to /api/legal/counsel (backend/app/legal/counsel/routes.py). Same-origin; the CSRF header is added globally. */
import { api, ApiError, postStream } from '../api/client';
import type { AuditEvent } from '../api/types';
import type { CitationDetail, ConversationSummary, DiscrepancyStatus } from './types';

const BASE = '/api/legal/counsel';

export interface StoredTurn {
  id: string; seq: number; question: string; understood?: string | null; intent: string; status: string;
  created_at: string; finished_at?: string | null; envelope?: unknown;
}

export interface ActivityPage {
  events: AuditEvent[];
  next_before: number | null;
  in_progress: { matter: { id: string; number: string; title: string }; documents: number }[];
}

export interface AskBody { text: string; client_request_id: string; document_ids?: string[]; focus_discrepancy_id?: string }

export interface DiscrepancySummary { id: string; title: string; status: DiscrepancyStatus; kind: string; version: number; updated_at: string }

/** The answer to a component action: the updated component, or why not (with the current one on a 409). */
export interface ActionResult { ok: boolean; status: number; message: string; component?: unknown }

export const counselApi = {
  conversations: (matterId?: string) =>
    api.get<{ conversations: ConversationSummary[] }>(`${BASE}/conversations${matterId ? `?matter_id=${encodeURIComponent(matterId)}` : ''}`),
  /** A conversation about one matter, or several (the first is its own). */
  create: (matterIds: string[]) => api.post<{ conversation: { id: string; matter_id: string; matter_ids?: string[]; title: string } }>(
    `${BASE}/conversations`, matterIds.length > 1 ? { matter_ids: matterIds } : { matter_id: matterIds[0] }),
  read: (id: string) => api.get<{ conversation: { id: string; matter_id: string; matter_ids?: string[]; title: string }; turns: StoredTurn[] }>(
    `${BASE}/conversations/${encodeURIComponent(id)}`),
  ask: (id: string, body: AskBody, onEvent: (e: { event: string; data: any }) => void, signal?: AbortSignal) =>
    postStream(`${BASE}/conversations/${encodeURIComponent(id)}/turns`, body, onEvent, signal),
  citation: (id: string) => api.get<{ citation: CitationDetail }>(`${BASE}/citations/${encodeURIComponent(id)}`),
  discrepancies: (matterId: string, state: 'open' | 'resolved' | 'all' = 'open') =>
    api.get<{ discrepancies: DiscrepancySummary[] }>(`${BASE}/discrepancies?matter_id=${encodeURIComponent(matterId)}&state=${state}`),
  activity: (before?: number) => api.get<ActivityPage>(`${BASE}/activity${before ? `?before=${before}` : ''}`),
  async action(body: { component_id: string; action_id: 'review'; matter_id: string; expected_version: number;
                      parameters: Record<string, unknown>; idempotency_key: string }): Promise<ActionResult> {
    const r = await fetch(`${BASE}/actions`, {
      method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    let data: any = null;
    try { data = await r.json(); } catch { /* empty */ }
    if (r.ok) return { ok: true, status: r.status, message: '', component: data?.component };
    const message = typeof data?.detail === 'string' ? data.detail
      : r.status === 422 ? 'Choose a resolution, and say briefly why.' : 'Something went wrong.';
    return { ok: false, status: r.status, message, component: data?.component };
  },
};

export { ApiError };

/** A random id for idempotent requests (crypto.randomUUID needs a secure context; getRandomValues does not). */
export function requestId(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}
