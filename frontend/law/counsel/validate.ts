/**
 * Client-side check of the component contract (defence in depth: the server already validated it). Written by hand so
 * no dependency is added; each parser returns a typed value or throws, and parseComponent turns any failure into an
 * inert placeholder. Limits mirror backend/app/legal/counsel/schemas.py.
 */
import type {
  ActionType, AnswerState, ApprovalItem, ApprovalRequestProps, CaseRow, CaseTableProps, ChartBar, ChecklistItem, ChecklistProps, CitationRef,
  ComparisonSide, CouncilSummaryProps, DiscrepancyStatus, DocumentPreviewProps, Envelope, EvidenceComparisonProps, GraphExcerptProps,
  GuidedQuestion, GuidedQuestionsProps, LiveTranscriptProps, MatterRef, NoticeProps, NumericChartProps,
  CalculationProps, PriorityItem, PriorityListProps, ReviewEntry, SafeComponent, SourceAnswerProps, TaskProgressProps, TimelineEvent, TimelineProps, UiAction,
  WorkingProps,
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

const MATTER_NUMBER = /^[A-Za-z0-9._~ -]{1,40}$/;
const optMatter = (v: unknown, what: string): string | undefined => {
  const s = optStr(v, 40, what);
  if (s !== undefined && !MATTER_NUMBER.test(s)) throw new Invalid(`${what}: not a matter number`);
  return s;
};

/** The workspace's own pages an action may open (same allow-list as the server). */
const PAGE = new RegExp(
  '^/m/[A-Za-z0-9._~%-]{1,96}/(?:overview|documents|compare|obligations|drafting|meetings|exports|activity'
  + '|documents/[A-Za-z0-9_-]{1,64}/read(?:\\?v=[A-Za-z0-9_-]{1,64}(?:&at=\\d{1,9}-\\d{1,9})?)?|documents/[A-Za-z0-9_-]{1,64}/definitions'
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
    matter: optMatter(o.matter, 'citation.matter'),
    sentences: o.sentences === undefined ? undefined : list(o.sentences, 8, 'citation.sentences', (x) => str(x, 600, 'citation.sentence')),
  };
}

const ANSWER_STATES: readonly AnswerState[] = ['answered', 'partial', 'unverified', 'abstained'];

function calculation(v: unknown): CalculationProps | undefined {
  if (v === undefined || v === null) return undefined;
  const o = obj(v, 'calculation');
  return {
    document_title: str(o.document_title, 300, 'calculation.document_title'),
    facts: list(o.facts, 6, 'calculation.facts', (x) => {
      const f = obj(x, 'fact');
      const path = str(f.path, 300, 'fact.path');
      if (!isWorkspacePath(path)) throw new Invalid('a fact opens the workspace\'s own reader');
      return { label: str(f.label, 60, 'fact.label'), value: str(f.value, 80, 'fact.value'), quote: str(f.quote, 300, 'fact.quote'),
               clause: optStr(f.clause, 40, 'fact.clause'), path };
    }),
    derived: list(o.derived, 4, 'calculation.derived', (x) => {
      const d = obj(x, 'derived');
      return { label: str(d.label, 80, 'derived.label'), display: str(d.display, 80, 'derived.display') };
    }),
    assumptions: list(o.assumptions, 4, 'calculation.assumptions', (x) => str(x, 400, 'assumption')),
    not_calculated: list(o.not_calculated, 3, 'calculation.not_calculated', (x) => str(x, 300, 'not_calculated')),
  };
}

function working(v: unknown): WorkingProps | undefined {
  if (v === undefined || v === null) return undefined;
  const o = obj(v, 'working');
  const w: WorkingProps = {
    steps: list(o.steps, 10, 'working.steps', (x) => {
      const s = obj(x, 'step');
      return { label: str(s.label, 120, 'step.label'), working: str(s.working, 600, 'step.working'), result: str(s.result, 80, 'step.result'),
               check: s.check === undefined ? false : bool(s.check, 'step.check') };
    }),
    figures: list(o.figures, 16, 'working.figures', (x) => {
      const f = obj(x, 'figure');
      const path = str(f.path, 300, 'figure.path');
      if (!isWorkspacePath(path)) throw new Invalid('a figure opens the workspace\'s own reader');
      return { label: str(f.label, 80, 'figure.label'), value: str(f.value, 80, 'figure.value'), quote: str(f.quote, 400, 'figure.quote'),
               document_title: str(f.document_title, 300, 'figure.document_title'), clause: optStr(f.clause, 40, 'figure.clause'), path };
    }),
    assumptions: list(o.assumptions, 3, 'working.assumptions', (x) => str(x, 400, 'assumption')),
  };
  if (!w.steps.length) throw new Invalid('a working without steps');
  return w;
}

function sourceAnswer(v: unknown): SourceAnswerProps {
  const o = obj(v, 'props');
  const p: SourceAnswerProps = {
    answer: str(o.answer, 8000, 'answer'), state: oneOf(o.state, ANSWER_STATES, 'state'),
    citations: list(o.citations, 30, 'citations', citation), unsupported: list(o.unsupported, 10, 'unsupported', (x) => str(x, 500, 'unsupported')),
    notes: list(o.notes, 3, 'notes', (x) => str(x, 400, 'note')), understood: optStr(o.understood, 400, 'understood'),
    absence: optStr(o.absence, 400, 'absence'), calculation: calculation(o.calculation), working: working(o.working),
    not_stated: o.not_stated === undefined ? undefined : bool(o.not_stated, 'not_stated'),
  };
  if (p.not_stated && p.state === 'abstained') throw new Invalid('a decline is shown as a decline');
  if ((p.calculation || p.working) && p.state === 'abstained') throw new Invalid('a decline calculates nothing');
  if (p.state === 'abstained' && p.citations.length) throw new Invalid('an abstention cites nothing');
  if (p.absence !== undefined && p.state !== 'abstained') throw new Invalid('only a decline says what is missing');
  return p;
}

const ITEM_KINDS: readonly PriorityItem['kind'][] = ['discrepancy', 'new_documents', 'review', 'obligation', 'scan', 'council'];

function priorityList(v: unknown): PriorityListProps {
  const o = obj(v, 'props');
  return {
    summary: str(o.summary, 400, 'summary'),
    items: list(o.items, 3, 'items', (x) => {
      const i = obj(x, 'item');
      return { id: ident(i.id, 'item.id'), kind: oneOf(i.kind, ITEM_KINDS, 'item.kind'), title: str(i.title, 160, 'item.title'),
               detail: i.detail === undefined ? '' : str(i.detail, 400, 'item.detail'), action: parseAction(i.action),
               matter: optMatter(i.matter, 'item.matter') };
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

const EVENT_KINDS: readonly TimelineEvent['kind'][] = ['event', 'deadline', 'document_date', 'obligation'];
const CERTAINTY: readonly TimelineEvent['certainty'][] = ['stated', 'uncertain', 'undated'];

function timeline(v: unknown): TimelineProps {
  const o = obj(v, 'props');
  return {
    summary: str(o.summary, 400, 'summary'), caution: str(o.caution, 400, 'caution'), more: o.more === undefined ? 0 : int(o.more, 'more'),
    events: list(o.events, 60, 'events', (x): TimelineEvent => {
      const e = obj(x, 'event');
      const date = optStr(e.date, 10, 'event.date');
      const certainty = oneOf(e.certainty, CERTAINTY, 'event.certainty');
      if (date !== undefined && !/^\d{4}-\d{2}-\d{2}$/.test(date)) throw new Invalid('event.date');
      if ((certainty === 'undated') !== (date === undefined)) throw new Invalid('an event has a date exactly when its date is stated');
      return {
        id: ident(e.id, 'event.id'), date, date_text: str(e.date_text, 80, 'event.date_text'), certainty,
        kind: oneOf(e.kind, EVENT_KINDS, 'event.kind'), reported: e.reported === undefined ? false : bool(e.reported, 'event.reported'),
        superseded: e.superseded === undefined ? false : bool(e.superseded, 'event.superseded'), text: str(e.text, 400, 'event.text'),
        status: optStr(e.status, 80, 'event.status'), notes: list(e.notes, 3, 'event.notes', (n) => str(n, 300, 'event.note')),
        source: side(e.source), matter: optMatter(e.matter, 'event.matter'),
      };
    }),
  };
}

function caseTable(v: unknown): CaseTableProps {
  const o = obj(v, 'props');
  const rows = list(o.rows, 10, 'rows', (x): CaseRow => {
    const r = obj(x, 'row');
    const path = str(r.path, 140, 'row.path');
    if (!isWorkspacePath(path)) throw new Invalid('row.path outside the workspace');
    const due = optStr(r.next_deadline, 10, 'row.next_deadline');
    if (due !== undefined && !/^\d{4}-\d{2}-\d{2}$/.test(due)) throw new Invalid('row.next_deadline');
    const number = str(r.number, 40, 'row.number');
    if (!MATTER_NUMBER.test(number)) throw new Invalid('row.number');
    return {
      matter_id: ident(r.matter_id, 'row.matter_id'), number, title: str(r.title, 200, 'row.title'), client: str(r.client, 200, 'row.client'),
      status: oneOf(r.status, ['open', 'closed', 'archived'] as const, 'row.status'),
      sensitivity: oneOf(r.sensitivity, ['standard', 'restricted', 'highly_restricted'] as const, 'row.sensitivity'),
      documents: int(r.documents, 'row.documents'), open_inconsistencies: optInt(r.open_inconsistencies, 'row.open'),
      next_deadline: due, next_deadline_text: optStr(r.next_deadline_text, 200, 'row.next_deadline_text'),
      deadline_state: optStr(r.deadline_state, 60, 'row.deadline_state'), last_activity: optStr(r.last_activity, 40, 'row.last_activity'), path,
    };
  });
  if (!rows.length) throw new Invalid('a table has rows');
  return { summary: str(o.summary, 400, 'summary'), rows, caution: str(o.caution, 400, 'caution') };
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

const ISO = /^\d{4}-\d{2}-\d{2}$/;
const page = (v: unknown, max: number, what: string) => {
  const p = str(v, max, what);
  if (!isWorkspacePath(p)) throw new Invalid(`${what} outside the workspace`);
  return p;
};

function checklist(v: unknown): ChecklistProps {
  const o = obj(v, 'props');
  return {
    summary: str(o.summary, 400, 'summary'), more: o.more === undefined ? 0 : int(o.more, 'more'), can_done: bool(o.can_done, 'can_done'),
    can_confirm: bool(o.can_confirm, 'can_confirm'), caution: str(o.caution, 400, 'caution'), register_path: page(o.register_path, 140, 'register_path'),
    items: list(o.items, 12, 'items', (x): ChecklistItem => {
      const i = obj(x, 'item');
      const due = optStr(i.due_date, 10, 'item.due_date');
      if (due !== undefined && !ISO.test(due)) throw new Invalid('item.due_date');
      return {
        id: ident(i.id, 'item.id'), title: str(i.title, 300, 'item.title'), party: optStr(i.party, 80, 'item.party'), when: optStr(i.when, 300, 'item.when'),
        due_date: due, status: oneOf(i.status, ['proposed', 'confirmed', 'done', 'dismissed'] as const, 'item.status'),
        overdue: i.overdue === undefined ? false : bool(i.overdue, 'item.overdue'), owner: optStr(i.owner, 80, 'item.owner'),
        document_title: str(i.document_title, 300, 'item.document_title'), clause: optStr(i.clause, 40, 'item.clause'),
        path: page(i.path, 300, 'item.path'), version: int(i.version, 'item.version', 1),
      };
    }),
  };
}

function approvalRequest(v: unknown): ApprovalRequestProps {
  const o = obj(v, 'props');
  return {
    summary: str(o.summary, 400, 'summary'), more: o.more === undefined ? 0 : int(o.more, 'more'), caution: str(o.caution, 400, 'caution'),
    items: list(o.items, 3, 'items', (x): ApprovalItem => {
      const i = obj(x, 'item');
      return {
        artifact_id: ident(i.artifact_id, 'item.artifact_id'), kind: str(i.kind, 40, 'item.kind'), title: str(i.title, 200, 'item.title'),
        version: int(i.version, 'item.version', 1), author: str(i.author, 80, 'item.author'), created_at: str(i.created_at, 40, 'item.created_at'),
        preview: list(i.preview, 4, 'item.preview', (p) => str(p, 600, 'preview')), gaps: i.gaps === undefined ? 0 : int(i.gaps, 'item.gaps'),
        consequence: str(i.consequence, 400, 'item.consequence'), path: page(i.path, 300, 'item.path'),
        state: i.state === undefined ? 'waiting' : oneOf(i.state, ['waiting', 'approved', 'changes_requested', 'changed', 'gone'] as const, 'item.state'),
        decided_by: optStr(i.decided_by, 80, 'item.decided_by'), can_decide: bool(i.can_decide, 'item.can_decide'),
      };
    }),
  };
}

const QUESTION_ID = /^[a-z][a-z0-9_]{0,31}$/;

function guidedQuestions(v: unknown): GuidedQuestionsProps {
  const o = obj(v, 'props');
  const questions = list(o.questions, 3, 'questions', (x): GuidedQuestion => {
    const q = obj(x, 'question');
    const id = str(q.id, 32, 'question.id');
    if (!QUESTION_ID.test(id)) throw new Invalid('question.id');
    const kind = oneOf(q.kind, ['choice', 'text'] as const, 'question.kind');
    const options = list(q.options, 12, 'question.options', (y) => {
      const op = obj(y, 'option');
      return { value: ident(op.value, 'option.value'), label: str(op.label, 200, 'option.label'), detail: optStr(op.detail, 200, 'option.detail') };
    });
    const dflt = optStr(q.default, 1000, 'question.default');
    if (kind === 'choice' && (!options.length || (dflt !== undefined && !options.some((op) => op.value === dflt)))) throw new Invalid('choice options');
    if (kind === 'text' && options.length) throw new Invalid('a text question has no options');
    const max = q.max_length === undefined ? 200 : int(q.max_length, 'question.max_length', 1);
    return { id, label: str(q.label, 160, 'question.label'), kind, options, default: dflt,
             required: q.required === undefined ? true : bool(q.required, 'question.required'), max_length: Math.min(max, 1000) };
  });
  if (!questions.length) throw new Invalid('a form asks something');
  return { purpose: oneOf(o.purpose, ['draft', 'panel'] as const, 'purpose'), title: str(o.title, 160, 'title'), intro: str(o.intro, 400, 'intro'),
           questions, submit_label: str(o.submit_label, 48, 'submit_label'), note: str(o.note, 400, 'note') };
}

function documentPreview(v: unknown): DocumentPreviewProps {
  const o = obj(v, 'props');
  return {
    artifact_id: ident(o.artifact_id, 'artifact_id'), title: str(o.title, 200, 'title'), template: str(o.template, 200, 'template'),
    version: int(o.version, 'version', 1), status: oneOf(o.status, ['draft', 'approved', 'superseded', 'filed'] as const, 'status'),
    gaps: int(o.gaps, 'gaps'), sources: int(o.sources, 'sources'),
    fields: list(o.fields, 12, 'fields', (x) => {
      const f = obj(x, 'field');
      return { label: str(f.label, 120, 'field.label'), value: str(f.value, 300, 'field.value'),
               status: oneOf(f.status, ['sourced', 'matter', 'system', 'entered', 'needs_input'] as const, 'field.status') };
    }),
    blocks: list(o.blocks, 8, 'blocks', (x) => {
      const b = obj(x, 'block');
      return { type: oneOf(b.type, ['heading', 'paragraph'] as const, 'block.type'), text: str(b.text, 1500, 'block.text'),
               origin: oneOf(b.origin, ['template', 'written', 'edited'] as const, 'block.origin') };
    }),
    more_blocks: o.more_blocks === undefined ? 0 : int(o.more_blocks, 'more_blocks'), review: str(o.review, 300, 'review'),
    path: page(o.path, 300, 'path'), can_edit: bool(o.can_edit, 'can_edit'),
  };
}

function councilSummary(v: unknown): CouncilSummaryProps {
  const o = obj(v, 'props');
  return {
    review_id: ident(o.review_id, 'review_id'), question: str(o.question, 2000, 'question'), area: str(o.area, 80, 'area'),
    status: oneOf(o.status, ['queued', 'running', 'done', 'needs_facts', 'failed', 'not_allowed'] as const, 'status'),
    issues: list(o.issues, 6, 'issues', (x) => {
      const i = obj(x, 'issue');
      return { issue: str(i.issue, 300, 'issue.issue'), disagreement: i.disagreement === undefined ? false : bool(i.disagreement, 'issue.disagreement'),
               positions: list(i.positions, 5, 'issue.positions', (y) => {
                 const p = obj(y, 'position');
                 return { role: str(p.role, 60, 'position.role'), text: str(p.text, 600, 'position.text') };
               }) };
    }),
    missing_facts: list(o.missing_facts, 6, 'missing_facts', (x) => str(x, 300, 'missing_fact')), approval: optStr(o.approval, 80, 'approval'),
    path: page(o.path, 300, 'path'), caution: str(o.caution, 400, 'caution'),
    conclusion: optStr(o.conclusion, 400, 'conclusion'),
    points: o.points === undefined ? [] : list(o.points, 3, 'points', (x) => str(x, 400, 'point')),
    gaps: o.gaps === undefined ? [] : list(o.gaps, 3, 'gaps', (x) => str(x, 300, 'gap')),
    outcome: o.outcome === undefined || o.outcome === null ? undefined : oneOf(o.outcome, ['answered', 'inconclusive', 'stopped'] as const, 'outcome'),
    figures_unsupported: o.figures_unsupported === undefined ? [] : list(o.figures_unsupported, 6, 'figures_unsupported', (x) => str(x, 40, 'figure')),
  };
}

function numericChart(v: unknown): NumericChartProps {
  const o = obj(v, 'props');
  return {
    summary: str(o.summary, 400, 'summary'), unit: str(o.unit, 12, 'unit'), more: o.more === undefined ? 0 : int(o.more, 'more'),
    other_units: list(o.other_units, 6, 'other_units', (x) => str(x, 12, 'unit')), caution: str(o.caution, 400, 'caution'),
    bars: list(o.bars, 24, 'bars', (x): ChartBar => {
      const b = obj(x, 'bar');
      if (typeof b.value !== 'number' || !Number.isFinite(b.value) || b.value < 0) throw new Invalid('bar.value');
      return { id: ident(b.id, 'bar.id'), shown: str(b.shown, 80, 'bar.shown'), value: b.value, document_title: str(b.document_title, 300, 'bar.document_title'),
               clause: optStr(b.clause, 40, 'bar.clause'), source: side(b.source) };
    }),
  };
}

function graphExcerpt(v: unknown): GraphExcerptProps {
  const o = obj(v, 'props');
  const nodes = list(o.nodes, 18, 'nodes', (x) => {
    const n = obj(x, 'node');
    return { id: ident(n.id, 'node.id'), label: str(n.label, 160, 'node.label'), kind: oneOf(n.kind, ['party', 'document'] as const, 'node.kind'),
             role: optStr(n.role, 40, 'node.role') };
  });
  const ids = new Set(nodes.map((n) => n.id));
  const edges = list(o.edges, 40, 'edges', (x) => {
    const e = obj(x, 'edge');
    const out = { id: ident(e.id, 'edge.id'), source: ident(e.source, 'edge.source'), target: ident(e.target, 'edge.target'),
                  label: oneOf(e.label, ['party to', 'named in'] as const, 'edge.label'), evidence: side(e.evidence) };
    if (!ids.has(out.source) || !ids.has(out.target)) throw new Invalid('an edge joins two shown nodes');
    return out;
  });
  return { summary: str(o.summary, 400, 'summary'), nodes, edges, more: o.more === undefined ? 0 : int(o.more, 'more'), caution: str(o.caution, 400, 'caution') };
}

function liveTranscript(v: unknown): LiveTranscriptProps {
  const o = obj(v, 'props');
  return {
    meeting_id: ident(o.meeting_id, 'meeting_id'), title: str(o.title, 200, 'title'),
    state: oneOf(o.state, ['recording', 'paused', 'not_recording'] as const, 'state'), consent: str(o.consent, 200, 'consent'),
    segments: list(o.segments, 12, 'segments', (x) => {
      const g = obj(x, 'segment');
      return { speaker: optStr(g.speaker, 80, 'segment.speaker'), text: str(g.text, 1000, 'segment.text'), at: optStr(g.at, 40, 'segment.at') };
    }),
    earlier: o.earlier === undefined ? 0 : int(o.earlier, 'earlier'), path: page(o.path, 300, 'path'), caution: str(o.caution, 400, 'caution'),
  };
}

/** One component, or an inert placeholder when it is unknown or does not match the contract. */
export function parseComponent(v: unknown): SafeComponent {
  const o = isObj(v) ? v : {};
  const id = typeof o.id === 'string' && /^cmp_[0-9a-f]{32}$/.test(o.id) ? o.id : 'cmp_invalid';
  const version = typeof o.version === 'number' && Number.isInteger(o.version) && o.version >= 1 ? o.version : 1;
  try {
    if (id === 'cmp_invalid') throw new Invalid('component id');
    const own = o.matter_id === undefined ? {} : { matter_id: ident(o.matter_id, 'matter_id') };    // its matter, when not the conversation's
    const base = { id, version, ...own };
    switch (o.type) {
      case 'source_answer': return { ...base, type: 'source_answer', props: sourceAnswer(o.props) };
      case 'priority_list': return { ...base, type: 'priority_list', props: priorityList(o.props) };
      case 'evidence_comparison':
        return { ...base, type: 'evidence_comparison', data_ref: str(o.data_ref, 40, 'data_ref'), props: evidenceComparison(o.props) };
      case 'timeline': return { ...base, type: 'timeline', props: timeline(o.props) };
      case 'case_table': return { ...base, type: 'case_table', props: caseTable(o.props) };
      case 'task_progress': return { ...base, type: 'task_progress', props: taskProgress(o.props) };
      case 'notice': return { ...base, type: 'notice', props: notice(o.props) };
      case 'unavailable': return { ...base, type: 'unavailable', props: { text: str(obj(o.props, 'props').text, 300, 'text') } };
      case 'checklist': return { ...base, type: 'checklist', data_ref: str(o.data_ref, 40, 'data_ref'), props: checklist(o.props) };
      case 'approval_request': return { ...base, type: 'approval_request', data_ref: str(o.data_ref, 40, 'data_ref'), props: approvalRequest(o.props) };
      case 'guided_questions': return { ...base, type: 'guided_questions', props: guidedQuestions(o.props) };
      case 'document_preview': return { ...base, type: 'document_preview', data_ref: str(o.data_ref, 40, 'data_ref'), props: documentPreview(o.props) };
      case 'council_summary': return { ...base, type: 'council_summary', data_ref: str(o.data_ref, 40, 'data_ref'), props: councilSummary(o.props) };
      case 'numeric_chart': return { ...base, type: 'numeric_chart', props: numericChart(o.props) };
      case 'graph_excerpt': return { ...base, type: 'graph_excerpt', props: graphExcerpt(o.props) };
      case 'live_transcript': return { ...base, type: 'live_transcript', data_ref: str(o.data_ref, 40, 'data_ref'), props: liveTranscript(o.props) };
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
  let matters: MatterRef[] | undefined;
  try {
    matters = list(v.matters, 5, 'matters', (x): MatterRef => {
      const m = obj(x, 'matter');
      const number = str(m.number, 40, 'matter.number');
      if (!MATTER_NUMBER.test(number)) throw new Invalid('matter.number');
      return { id: ident(m.id, 'matter.id'), number, title: str(m.title, 200, 'matter.title') };
    });
  } catch { matters = undefined; }                       // informational only: an unusable list is left out
  return {
    schema_version: '1.0', message_id: v.message_id, matter_id: v.matter_id, ...(matters?.length ? { matters } : {}),
    summary: typeof v.summary === 'string' ? v.summary.slice(0, 400) : '', components: comps.map(parseComponent), suggested_actions: actions,
  };
}

export const parseTaskProgress = (v: unknown): TaskProgressProps | null => {
  try { return taskProgress(v); } catch { return null; }
};
