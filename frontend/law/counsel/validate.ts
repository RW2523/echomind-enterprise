/**
 * Client-side check of the component contract (defence in depth: the server already validated it). Written by hand so
 * no dependency is added; each parser returns a typed value or throws, and parseComponent turns any failure into an
 * inert placeholder. Limits mirror backend/app/legal/counsel/schemas.py.
 */
import type {
  ActionType, AnswerState, CitationRef, ComparisonSide, DiscrepancyStatus, Envelope, EvidenceComparisonProps, NoticeProps,
  PriorityItem, PriorityListProps, ReviewEntry, SafeComponent, SourceAnswerProps, TaskProgressProps, UiAction,
} from './types';

class Invalid extends Error {}

type Raw = Record<string, unknown>;

const isObj = (v: unknown): v is Raw => typeof v === 'object' && v !== null && !Array.isArray(v);

function obj(v: unknown, what: string): Raw {
  if (!isObj(v)) throw new Invalid(`${what}: not an object`);
  return v;
}

function str(v: unknown, max: number, what: string): string {
  if (typeof v !== 'string' || v.length > max) throw new Invalid(`${what}: not a string of at most ${max}`);
  return v;
}

function optStr(v: unknown, max: number, what: string): string | undefined {
  return v === undefined || v === null ? undefined : str(v, max, what);
}

function int(v: unknown, what: string, min = 0): number {
  if (typeof v !== 'number' || !Number.isInteger(v) || v < min) throw new Invalid(`${what}: not an integer ≥ ${min}`);
  return v;
}

function optInt(v: unknown, what: string, min = 0): number | undefined {
  return v === undefined || v === null ? undefined : int(v, what, min);
}

function bool(v: unknown, what: string): boolean {
  if (typeof v !== 'boolean') throw new Invalid(`${what}: not a boolean`);
  return v;
}

function oneOf<T extends string>(v: unknown, allowed: readonly T[], what: string): T {
  if (typeof v !== 'string' || !(allowed as readonly string[]).includes(v)) throw new Invalid(`${what}: not one of ${allowed.join(', ')}`);
  return v as T;
}

function list<T>(v: unknown, max: number, what: string, each: (x: unknown, i: number) => T): T[] {
  if (v === undefined || v === null) return [];
  if (!Array.isArray(v) || v.length > max) throw new Invalid(`${what}: not a list of at most ${max}`);
  return v.map(each);
}

const IDENT = /^[A-Za-z0-9_.:-]{1,80}$/;
const ident = (v: unknown, what: string) => {
  const s = str(v, 80, what);
  if (!IDENT.test(s)) throw new Invalid(`${what}: not an identifier`);
  return s;
};

/** The workspace's own pages an action may open (same allow-list as the server). */
const PAGE = new RegExp(
  '^/m/[A-Za-z0-9._~%-]{1,96}/(?:overview|documents|compare|obligations|drafting|meetings|exports|activity'
  + '|documents/[A-Za-z0-9_-]{1,64}/read(?:\\?v=[A-Za-z0-9_-]{1,64}(?:&at=\\d{1,9}-\\d{1,9})?)?'
  + '|notes/[A-Za-z0-9_-]{1,64}|drafting/[A-Za-z0-9_-]{1,64}|meetings/[A-Za-z0-9_-]{1,64}(?:\\?tab=minutes)?'
  + '|redlines/[A-Za-z0-9_-]{1,64}|summaries/[A-Za-z0-9_-]{1,64}|exports/[A-Za-z0-9_-]{1,64}|panel(?:/[A-Za-z0-9_-]{1,64})?)$'
  + '|^/(?:reviews|matters|drafts|activity)$');

export const isWorkspacePath = (path: string | undefined): path is string => !!path && PAGE.test(path);

const ACTION_TYPES: readonly ActionType[] = ['ask', 'open_source', 'open_discrepancy', 'open_page'];

export function parseAction(v: unknown): UiAction {
  const o = obj(v, 'action');
  const a: UiAction = {
    id: str(o.id, 32, 'action.id'), label: str(o.label, 48, 'action.label'), action_type: oneOf(o.action_type, ACTION_TYPES, 'action.type'),
    target_id: o.target_id === undefined ? undefined : ident(o.target_id, 'action.target_id'),
    prompt: optStr(o.prompt, 300, 'action.prompt'),
    document_ids: list(o.document_ids, 10, 'action.document_ids', (x) => ident(x, 'action.document_id')),
    path: optStr(o.path, 300, 'action.path'),
  };
  if (a.action_type === 'ask' && !a.prompt?.trim()) throw new Invalid('ask without a prompt');
  if ((a.action_type === 'open_source' || a.action_type === 'open_discrepancy') && !a.target_id) throw new Invalid('open without a target');
  if (a.action_type === 'open_page' && !isWorkspacePath(a.path)) throw new Invalid('open_page outside the workspace');
  if (a.action_type !== 'open_page' && a.path !== undefined) throw new Invalid('only open_page carries a path');
  return a;
}

function citation(v: unknown): CitationRef {
  const o = obj(v, 'citation');
  const id = str(o.id, 40, 'citation.id');
  if (!/^cit_[0-9a-f]{32}$/.test(id)) throw new Invalid('citation.id');
  return {
    id, n: int(o.n, 'citation.n', 1), document_id: ident(o.document_id, 'citation.document_id'),
    document_title: str(o.document_title, 300, 'citation.document_title'), version_number: optInt(o.version_number, 'citation.version', 1),
    version_status: optStr(o.version_status, 32, 'citation.version_status'), page: optInt(o.page, 'citation.page', 1),
    clause: optStr(o.clause, 40, 'citation.clause'), verified: bool(o.verified, 'citation.verified'),
    removed: o.removed === undefined ? false : bool(o.removed, 'citation.removed'),
  };
}

const ANSWER_STATES: readonly AnswerState[] = ['answered', 'partial', 'unverified', 'abstained'];

function sourceAnswer(v: unknown): SourceAnswerProps {
  const o = obj(v, 'props');
  const p: SourceAnswerProps = {
    answer: str(o.answer, 8000, 'answer'), state: oneOf(o.state, ANSWER_STATES, 'state'),
    citations: list(o.citations, 30, 'citations', citation), unsupported: list(o.unsupported, 10, 'unsupported', (x) => str(x, 500, 'unsupported')),
    notes: list(o.notes, 3, 'notes', (x) => str(x, 400, 'note')), understood: optStr(o.understood, 400, 'understood'),
  };
  if (p.state === 'abstained' && p.citations.length) throw new Invalid('an abstention cites nothing');
  return p;
}

const ITEM_KINDS: readonly PriorityItem['kind'][] = ['discrepancy', 'new_documents', 'review', 'obligation', 'scan'];

function priorityList(v: unknown): PriorityListProps {
  const o = obj(v, 'props');
  return {
    summary: str(o.summary, 400, 'summary'),
    items: list(o.items, 3, 'items', (x) => {
      const i = obj(x, 'item');
      return { id: ident(i.id, 'item.id'), kind: oneOf(i.kind, ITEM_KINDS, 'item.kind'), title: str(i.title, 160, 'item.title'),
               detail: i.detail === undefined ? '' : str(i.detail, 400, 'item.detail'), action: parseAction(i.action) };
    }),
    more: o.more === undefined ? 0 : int(o.more, 'more'), more_action: o.more_action === undefined ? undefined : parseAction(o.more_action),
    empty: optStr(o.empty, 300, 'empty'),
  };
}

function side(v: unknown): ComparisonSide {
  const o = obj(v, 'side');
  const s: ComparisonSide = {
    document_id: ident(o.document_id, 'side.document_id'), document_title: str(o.document_title, 300, 'side.title'),
    version_id: ident(o.version_id, 'side.version_id'), version_number: optInt(o.version_number, 'side.version', 1),
    version_status: optStr(o.version_status, 32, 'side.status'), clause: optStr(o.clause, 40, 'side.clause'), page: optInt(o.page, 'side.page', 1),
    value: str(o.value, 300, 'side.value'), context: str(o.context, 1500, 'side.context'), value_start: int(o.value_start, 'side.value_start'),
    value_end: int(o.value_end, 'side.value_end'), reader_path: optStr(o.reader_path, 300, 'side.reader_path'),
    stale: o.stale === undefined ? false : bool(o.stale, 'side.stale'),
  };
  if (!(s.value_start <= s.value_end && s.value_end <= s.context.length)) throw new Invalid('highlight outside the context');
  if (s.reader_path !== undefined && !isWorkspacePath(s.reader_path)) throw new Invalid('reader_path outside the workspace');
  return s;
}

const STATUSES: readonly DiscrepancyStatus[] = ['potential', 'in_review', 'explained', 'confirmed', 'dismissed'];

function evidenceComparison(v: unknown): EvidenceComparisonProps {
  const o = obj(v, 'props');
  return {
    discrepancy_id: str(o.discrepancy_id, 40, 'discrepancy_id'), kind: str(o.kind, 24, 'kind'), title: str(o.title, 160, 'title'),
    status: oneOf(o.status, STATUSES, 'status'), a: side(o.a), b: side(o.b), explanation: o.explanation === undefined ? '' : str(o.explanation, 1500, 'explanation'),
    caution: str(o.caution, 400, 'caution'), method: oneOf(o.method, ['model', 'rule'] as const, 'method'),
    history: list(o.history, 20, 'history', (x): ReviewEntry => {
      const h = obj(x, 'review');
      return { to_status: oneOf(h.to_status, STATUSES, 'review.status'), by: str(h.by, 80, 'review.by'), at: str(h.at, 40, 'review.at'),
               note: optStr(h.note, 1000, 'review.note') };
    }),
    can_review: bool(o.can_review, 'can_review'),
  };
}

function taskProgress(v: unknown): TaskProgressProps {
  const o = obj(v, 'props');
  return { label: str(o.label, 200, 'label'), done: int(o.done, 'done'), total: int(o.total, 'total'),
           state: oneOf(o.state, ['running', 'done', 'canceled', 'failed'] as const, 'state') };
}

function notice(v: unknown): NoticeProps {
  const o = obj(v, 'props');
  return { tone: oneOf(o.tone, ['info', 'caution'] as const, 'tone'), text: str(o.text, 800, 'text') };
}

/** One component, or an inert placeholder when it is unknown or does not match the contract. */
export function parseComponent(v: unknown): SafeComponent {
  const o = isObj(v) ? v : {};
  const id = typeof o.id === 'string' && /^cmp_[0-9a-f]{32}$/.test(o.id) ? o.id : 'cmp_invalid';
  const version = typeof o.version === 'number' && Number.isInteger(o.version) && o.version >= 1 ? o.version : 1;
  try {
    if (id === 'cmp_invalid') throw new Invalid('component id');
    switch (o.type) {
      case 'source_answer': return { id, version, type: 'source_answer', props: sourceAnswer(o.props) };
      case 'priority_list': return { id, version, type: 'priority_list', props: priorityList(o.props) };
      case 'evidence_comparison':
        return { id, version, type: 'evidence_comparison', data_ref: str(o.data_ref, 40, 'data_ref'), props: evidenceComparison(o.props) };
      case 'task_progress': return { id, version, type: 'task_progress', props: taskProgress(o.props) };
      case 'notice': return { id, version, type: 'notice', props: notice(o.props) };
      case 'unavailable': return { id, version, type: 'unavailable', props: { text: str(obj(o.props, 'props').text, 300, 'text') } };
      default: throw new Invalid(`unknown component type ${String(o.type).slice(0, 40)}`);
    }
  } catch (e) {
    return { id, version, type: 'inert', reason: e instanceof Error ? e.message : 'invalid' };
  }
}

/** The envelope, with every component checked (at most three) and every suggested action checked; null if unusable. */
export function parseEnvelope(v: unknown): Envelope | null {
  if (!isObj(v) || v.schema_version !== '1.0' || typeof v.message_id !== 'string' || typeof v.matter_id !== 'string') return null;
  const comps = Array.isArray(v.components) ? v.components.slice(0, 3) : [];
  const actions: UiAction[] = [];
  for (const a of Array.isArray(v.suggested_actions) ? v.suggested_actions.slice(0, 4) : []) {
    try { actions.push(parseAction(a)); } catch { /* an action that does not validate is not offered */ }
  }
  return {
    schema_version: '1.0', message_id: v.message_id, matter_id: v.matter_id,
    summary: typeof v.summary === 'string' ? v.summary.slice(0, 400) : '', components: comps.map(parseComponent), suggested_actions: actions,
  };
}

export const parseTaskProgress = (v: unknown): TaskProgressProps | null => {
  try { return taskProgress(v); } catch { return null; }
};
