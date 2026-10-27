/**
 * The trusted UI component contract v1.0 (docs/echomind-legal/03), as the browser sees it. The server validates every
 * envelope before sending it (backend/app/legal/counsel/schemas.py); validate.ts checks it again here, and anything that
 * does not match becomes an inert placeholder. Nothing in a component is ever rendered as HTML.
 */
export type AnswerState = 'answered' | 'partial' | 'unverified' | 'abstained';
export type DiscrepancyStatus = 'potential' | 'in_review' | 'explained' | 'confirmed' | 'dismissed';
export type ActionType = 'ask' | 'open_source' | 'open_discrepancy' | 'open_page';

export interface UiAction {
  id: string;
  label: string;
  action_type: ActionType;
  target_id?: string;
  prompt?: string;
  document_ids: string[];
  path?: string;
}

export interface CitationRef {
  id: string;
  n: number;
  document_id: string;
  document_title: string;
  version_number?: number;
  version_status?: string;
  page?: number;
  clause?: string;
  verified: boolean;
  removed: boolean;
}

export interface SourceAnswerProps {
  answer: string;
  state: AnswerState;
  citations: CitationRef[];
  unsupported: string[];
  notes: string[];
  understood?: string;
}

export interface PriorityItem {
  id: string;
  kind: 'discrepancy' | 'new_documents' | 'review' | 'obligation' | 'scan';
  title: string;
  detail: string;
  action: UiAction;
}

export interface PriorityListProps {
  summary: string;
  items: PriorityItem[];
  more: number;
  more_action?: UiAction;
  empty?: string;
}

export interface ComparisonSide {
  document_id: string;
  document_title: string;
  version_id: string;
  version_number?: number;
  version_status?: string;
  clause?: string;
  page?: number;
  value: string;
  context: string;
  value_start: number;
  value_end: number;
  reader_path?: string;
  stale: boolean;
}

export interface ReviewEntry { to_status: DiscrepancyStatus; by: string; at: string; note?: string }

export interface EvidenceComparisonProps {
  discrepancy_id: string;
  kind: string;
  title: string;
  status: DiscrepancyStatus;
  a: ComparisonSide;
  b: ComparisonSide;
  explanation: string;
  caution: string;
  method: 'model' | 'rule';
  history: ReviewEntry[];
  can_review: boolean;
}

export interface TimelineEvent {
  id: string;
  date?: string;                     // YYYY-MM-DD as the document states it; absent when it does not
  date_text: string;                 // the date's words, as written
  certainty: 'stated' | 'uncertain' | 'undated';
  kind: 'event' | 'deadline' | 'document_date' | 'obligation';
  reported: boolean;
  superseded: boolean;
  text: string;
  status?: string;
  notes: string[];
  source: ComparisonSide;
}

export interface TimelineProps { summary: string; events: TimelineEvent[]; more: number; caution: string }

export interface TaskProgressProps { label: string; done: number; total: number; state: 'running' | 'done' | 'canceled' | 'failed' }
export interface NoticeProps { tone: 'info' | 'caution'; text: string }
export interface UnavailableProps { text: string }

interface Base { id: string; version: number }
export type SafeComponent =
  | (Base & { type: 'source_answer'; props: SourceAnswerProps })
  | (Base & { type: 'priority_list'; props: PriorityListProps })
  | (Base & { type: 'evidence_comparison'; data_ref: string; props: EvidenceComparisonProps })
  | (Base & { type: 'timeline'; props: TimelineProps })
  | (Base & { type: 'task_progress'; props: TaskProgressProps })
  | (Base & { type: 'notice'; props: NoticeProps })
  | (Base & { type: 'unavailable'; props: UnavailableProps })
  /** Client-side only: a component that was unknown or did not validate. Rendered without any control. */
  | (Base & { type: 'inert'; reason: string });

export interface Envelope {
  schema_version: '1.0';
  message_id: string;
  matter_id: string;
  summary: string;
  components: SafeComponent[];
  suggested_actions: UiAction[];
}

/** A turn as the chat shows it: the question, then either the stream in progress or the finished envelope. */
export type TurnStatus = 'streaming' | 'completed' | 'abstained' | 'failed' | 'scope_revoked' | 'canceled';
export interface ChatTurn {
  key: string;
  id?: string;
  question: string;
  understood?: string;
  status: TurnStatus;
  text: string;
  progress?: TaskProgressProps;
  envelope?: Envelope;
  error?: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  matter: { id: string; number: string; title: string };
  turns: number;
}

export interface CitationDetail {
  id: string;
  n: number;
  matter_id: string;
  verification: 'verbatim' | 'normalized' | 'not_quoted' | 'failed';
  verified: boolean;
  document: { id: string; title: string; kind: string; status: string };
  version: { id: string; number: number; status: string; filename: string; sha256: string };
  quote: string | null;
  context: { text: string; highlight_start: number; highlight_end: number; offset: number; truncated_before: boolean; truncated_after: boolean } | null;
  clause?: string | null;
  page?: number | null;
  page_end?: number | null;
  line?: number | null;
  newer_version?: number | null;
  reader_path: string;
}
