/**
 * Board Room Mode API client.
 * Provides functions for session management, report access, WebSocket connection,
 * and PDF/PPTX export.
 */

import { API_BASE } from './backend';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface BoardRoomSession {
  id: string;
  title: string;
  location: string;
  status: 'created' | 'active' | 'completed' | 'failed';
  started_at: string | null;
  ended_at: string | null;
  duration_sec: number | null;
  speaker_count: number;
  segment_count: number;
  created_at: string;
}

export interface BoardRoomSessionDetail extends BoardRoomSession {
  raw_transcript: string;
  speaker_map: Record<string, string>;
  segments: SpeakerSegment[];
}

export interface SpeakerSegment {
  speaker_id: string;
  speaker_name: string;
  text: string;
  ts_ms: number;
  segment_index: number;
}

export interface RagEvidence {
  text: string;
  source: string;
  section: string;
  page: number | null;
  score: number;
}

export interface ActionItem {
  item: string;
  owner: string;
  priority: 'High' | 'Medium' | 'Low';
}

export interface KeyDiscussionPoint {
  topic: string;
  summary: string;
  rag_evidence: RagEvidence[];
}

export interface BoardRoomReport {
  report_id: string;
  session_id: string;
  generated_at: string;
  executive_summary: string;
  participants: { name: string; speaking_turns: number }[];
  key_discussion_points: KeyDiscussionPoint[];
  decisions: string[];
  action_items: ActionItem[];
  rag_evidence: RagEvidence[];
  contradictions: string[];
  recommendations: string[];
  topics: string[];
}

export interface ReportRow {
  report_id: string;
  session_id: string;
  status: 'pending' | 'generating' | 'ready' | 'failed' | 'not_generated';
  report: BoardRoomReport | null;
  created_at?: string;
  updated_at?: string;
}

// ── WebSocket message types ────────────────────────────────────────────────────

/** A merged speaker turn: consecutive deltas from the same speaker combined. */
export interface SpeakerTurn {
  speaker: string;
  text: string;
}

export type WsIncoming =
  | { type: 'loading' }
  // Sent when model is ready before any session has started
  | { type: 'ready'; session_id?: string; sample_rate: number }
  // Session has started (spec-aligned)
  | { type: 'session_started'; session_id: string; status: string; sample_rate?: number }
  // Periodic listening heartbeat (spec-aligned)
  | {
      type: 'listening_status';
      session_id: string;
      duration_seconds: number;
      audio_level: number;
      chunks_received: number;
      status: string;
    }
  // Live partial transcript update with merged turns
  | {
      type: 'partial';
      session_id: string;
      turns: SpeakerTurn[];
      speaker_count: number;
      duration_seconds?: number;
      audio_level?: number;
      chunks_received?: number;
    }
  // Processing pipeline status update
  | { type: 'finalizing'; session_id: string; status: string; message: string }
  // Transcription completed (spec-aligned)
  | {
      type: 'transcript_completed';
      session_id: string;
      transcript_id: string;
      speaker_count: number;
      segments: { speaker: string; start_time: number; end_time: number | null; text: string }[];
      duration_seconds?: number;
      transcription_source?: string;
      diarization_speaker_count?: number;
      fallback_used?: boolean;
    }
  // Report completed (spec-aligned)
  | {
      type: 'report_completed';
      session_id: string;
      report_id: string;
      exports: { pdf_url: string | null; pptx_url: string | null };
    }
  // Legacy final message kept for backward compatibility
  | {
      type: 'final';
      session_id: string;
      transcript: string;
      turns: SpeakerTurn[];
      speaker_map: Record<string, string>;
      duration_sec: number;
    }
  | { type: 'report_generating'; session_id: string }
  // Legacy alias kept for backward compatibility
  | { type: 'report_ready'; session_id: string; report_id: string }
  | { type: 'report_error'; session_id: string; message: string }
  | { type: 'error'; session_id?: string; message: string };

// ── REST endpoints ────────────────────────────────────────────────────────────

export async function listBoardRoomSessions(): Promise<{ sessions: BoardRoomSession[] }> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions`);
  if (!r.ok) throw new Error(`list boardroom sessions failed: ${r.status}`);
  return r.json();
}

export async function getBoardRoomSession(sessionId: string): Promise<BoardRoomSessionDetail> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}`);
  if (!r.ok) throw new Error(`get session failed: ${r.status}`);
  return r.json();
}

export async function deleteBoardRoomSession(sessionId: string): Promise<{ ok: boolean; deleted: string }> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`delete session failed: ${r.status}`);
  return r.json();
}

export async function getSessionReport(sessionId: string): Promise<ReportRow> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}/report`);
  if (!r.ok) throw new Error(`get report failed: ${r.status}`);
  return r.json();
}

export async function triggerReportGeneration(sessionId: string): Promise<{ ok: boolean; message: string }> {
  const r = await fetch(`${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}/report`, { method: 'POST' });
  if (!r.ok) throw new Error(`trigger report failed: ${r.status}`);
  return r.json();
}

export function getExportUrl(sessionId: string, format: 'pdf' | 'pptx'): string {
  return `${API_BASE}/api/boardroom/sessions/${encodeURIComponent(sessionId)}/export?format=${format}`;
}

export async function downloadExport(sessionId: string, format: 'pdf' | 'pptx'): Promise<void> {
  const url = getExportUrl(sessionId, format);
  const r = await fetch(url);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error((err as { detail?: string }).detail || `export failed: ${r.status}`);
  }
  const blob = await r.blob();
  const objUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = objUrl;
  const cd = r.headers.get('content-disposition') || '';
  const match = cd.match(/filename="([^"]+)"/);
  a.download = match ? match[1] : `boardroom_report.${format}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(objUrl);
}

// ── WebSocket URL helper ───────────────────────────────────────────────────────

export function boardroomWsUrl(): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}${API_BASE}/api/boardroom/ws`;
}

// ── SPEAKER_COLORS ─────────────────────────────────────────────────────────────
// Must match backend BOARDROOM_SPEAKER_COLORS default

export const SPEAKER_COLORS = [
  '#22d3ee', // cyan
  '#a78bfa', // violet
  '#34d399', // emerald
  '#f59e0b', // amber
  '#f87171', // red
  '#60a5fa', // blue
  '#e879f9', // fuchsia
  '#4ade80', // green
  '#fb923c', // orange
  '#94a3b8', // slate
];

export function getSpeakerColor(speakerName: string, allSpeakers: string[]): string {
  const idx = allSpeakers.indexOf(speakerName);
  return SPEAKER_COLORS[idx >= 0 ? idx % SPEAKER_COLORS.length : 0];
}
