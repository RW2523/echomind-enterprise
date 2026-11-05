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
  matter?: string;                   // in a conversation about several matters: the source's matter
  /** The answer's sentences this span supports (2026-11-05): a source quoted in several places has one citation per place. */
  sentences?: string[];
}

export interface SourceAnswerProps {
  answer: string;
  state: AnswerState;
  citations: CitationRef[];
  unsupported: string[];
  notes: string[];
  understood?: string;
  /** A decline: what the documents were found not to state, in the model's words, without a source. */
  absence?: string;
  /** Dates calculated from one document's stated words, never presented as stated (QA RAG-003). */
  calculation?: CalculationProps;
  /** The answer opens by saying the documents do not state what was asked, then cites related text. */
  not_stated?: boolean;
}

export interface CalculationFact { label: string; value: string; quote: string; clause?: string; path: string }
export interface CalculationProps {
  document_title: string;
  facts: CalculationFact[];
  derived: { label: string; display: string }[];
  assumptions: string[];
  not_calculated: string[];
}

export interface PriorityItem {
  id: string;
  kind: 'discrepancy' | 'new_documents' | 'review' | 'obligation' | 'scan' | 'council';
  title: string;
  detail: string;
  action: UiAction;
  matter?: string;
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
  matter?: string;
}

export interface TimelineProps { summary: string; events: TimelineEvent[]; more: number; caution: string }

/** One matter as its records stand (several matters side by side). */
export interface CaseRow {
  matter_id: string; number: string; title: string; client: string;
  status: 'open' | 'closed' | 'archived'; sensitivity: 'standard' | 'restricted' | 'highly_restricted';
  documents: number; open_inconsistencies?: number; next_deadline?: string; next_deadline_text?: string; deadline_state?: string;
  last_activity?: string; path: string;
}
export interface CaseTableProps { summary: string; rows: CaseRow[]; caution: string }
export interface MatterRef { id: string; number: string; title: string }

export interface TaskProgressProps { label: string; done: number; total: number; state: 'running' | 'done' | 'canceled' | 'failed' }
export interface NoticeProps { tone: 'info' | 'caution'; text: string }
export interface UnavailableProps { text: string }

// ── Cards added 2026-11-02 (docs/echomind-legal/03 registry): each rebuilt by the server from its records ──
export interface ChecklistItem {
  id: string; title: string; party?: string; when?: string; due_date?: string; status: 'proposed' | 'confirmed' | 'done' | 'dismissed';
  overdue: boolean; owner?: string; document_title: string; clause?: string; path: string; version: number;
}
export interface ChecklistProps { summary: string; items: ChecklistItem[]; more: number; can_done: boolean; can_confirm: boolean; caution: string; register_path: string }

export interface ApprovalItem {
  artifact_id: string; kind: string; title: string; version: number; author: string; created_at: string; preview: string[]; gaps: number;
  consequence: string; path: string; state: 'waiting' | 'approved' | 'changes_requested' | 'changed' | 'gone'; decided_by?: string; can_decide: boolean;
}
export interface ApprovalRequestProps { summary: string; items: ApprovalItem[]; more: number; caution: string }

export interface GuidedOption { value: string; label: string; detail?: string }
export interface GuidedQuestion {
  id: string; label: string; kind: 'choice' | 'text'; options: GuidedOption[]; default?: string; required: boolean; max_length: number;
}
export interface GuidedQuestionsProps { purpose: 'draft' | 'panel'; title: string; intro: string; questions: GuidedQuestion[]; submit_label: string; note: string }

export interface PreviewField { label: string; value: string; status: 'sourced' | 'matter' | 'system' | 'entered' | 'needs_input' }
export interface PreviewBlock { type: 'heading' | 'paragraph'; text: string; origin: 'template' | 'written' | 'edited' }
export interface DocumentPreviewProps {
  artifact_id: string; title: string; template: string; version: number; status: 'draft' | 'approved' | 'superseded' | 'filed'; gaps: number;
  sources: number; fields: PreviewField[]; blocks: PreviewBlock[]; more_blocks: number; review: string; path: string; can_edit: boolean;
}

export interface PanelIssue { issue: string; positions: { role: string; text: string }[]; disagreement: boolean }
export interface CouncilSummaryProps {
  review_id: string; question: string; area: string; status: 'queued' | 'running' | 'done' | 'needs_facts' | 'failed' | 'not_allowed';
  issues: PanelIssue[]; missing_facts: string[]; approval?: string; path: string; caution: string;
  /** The Council (2026-11-05): the chair's conclusion first, at most three points and gaps, figures no answer gave. */
  conclusion?: string; points: string[]; gaps: string[]; outcome?: 'answered' | 'inconclusive' | 'stopped'; figures_unsupported: string[];
}

export interface ChartBar { id: string; shown: string; value: number; document_title: string; clause?: string; source: ComparisonSide }
export interface NumericChartProps { summary: string; unit: string; bars: ChartBar[]; more: number; other_units: string[]; caution: string }

export interface GraphNode { id: string; label: string; kind: 'party' | 'document'; role?: string }
export interface GraphEdge { id: string; source: string; target: string; label: 'party to' | 'named in'; evidence: ComparisonSide }
export interface GraphExcerptProps { summary: string; nodes: GraphNode[]; edges: GraphEdge[]; more: number; caution: string }

export interface LiveSegment { speaker?: string; text: string; at?: string }
export interface LiveTranscriptProps {
  meeting_id: string; title: string; state: 'recording' | 'paused' | 'not_recording'; consent: string; segments: LiveSegment[]; earlier: number;
  path: string; caution: string;
}

interface Base { id: string; version: number; matter_id?: string }
export type SafeComponent =
  | (Base & { type: 'source_answer'; props: SourceAnswerProps })
  | (Base & { type: 'priority_list'; props: PriorityListProps })
  | (Base & { type: 'evidence_comparison'; data_ref: string; props: EvidenceComparisonProps })
  | (Base & { type: 'timeline'; props: TimelineProps })
  | (Base & { type: 'case_table'; props: CaseTableProps })
  | (Base & { type: 'task_progress'; props: TaskProgressProps })
  | (Base & { type: 'notice'; props: NoticeProps })
  | (Base & { type: 'unavailable'; props: UnavailableProps })
  | (Base & { type: 'checklist'; data_ref: string; props: ChecklistProps })
  | (Base & { type: 'approval_request'; data_ref: string; props: ApprovalRequestProps })
  | (Base & { type: 'guided_questions'; props: GuidedQuestionsProps })
  | (Base & { type: 'document_preview'; data_ref: string; props: DocumentPreviewProps })
  | (Base & { type: 'council_summary'; data_ref: string; props: CouncilSummaryProps })
  | (Base & { type: 'numeric_chart'; props: NumericChartProps })
  | (Base & { type: 'graph_excerpt'; props: GraphExcerptProps })
  | (Base & { type: 'live_transcript'; data_ref: string; props: LiveTranscriptProps })
  /** Client-side only: a component that was unknown or did not validate. Rendered without any control. */
  | (Base & { type: 'inert'; reason: string });

export interface Envelope {
  schema_version: '1.0';
  message_id: string;
  matter_id: string;
  matters?: MatterRef[];             // when the conversation covers several matters
  summary: string;
  components: SafeComponent[];
  suggested_actions: UiAction[];
}

/** A turn as the chat shows it: the question, then either the stream in progress or the finished envelope. */
export type TurnStatus = 'streaming' | 'completed' | 'abstained' | 'failed' | 'scope_revoked' | 'canceled';
export type FeedbackReason = 'wrong_source' | 'missed' | 'not_asked' | 'unclear' | 'other';
export interface Feedback { rating: 'up' | 'down'; reason?: FeedbackReason }

export interface ChatTurn {
  key: string;
  id?: string;
  /** The turn this one follows (its server id), null for a first question. Turns with the same parent are versions. */
  parent?: string | null;
  feedback?: Feedback;
  question: string;
  understood?: string;
  status: TurnStatus;
  text: string;
  progress?: TaskProgressProps;
  envelope?: Envelope;
  partial?: SafeComponent[];         // components shown as soon as they are ready, before the whole answer (ui.partial)
  error?: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  matter: { id: string; number: string; title: string };
  matters?: { id: string; number: string; title: string }[];
  turns: number;
  match?: string;                    // a search result: the words around the matching question
}

export interface CitationDetail {
  id: string;
  n: number;
  matter_id: string;
  verification: 'verbatim' | 'normalized' | 'not_quoted' | 'failed';
  verified: boolean;
  document: { id: string; title: string; kind: string; status: string };
  version: { id: string; number: number; status: string; filename: string; sha256: string; uploaded_at?: string | null };
  quote: string | null;
  context: { text: string; highlight_start: number; highlight_end: number; offset: number; truncated_before: boolean; truncated_after: boolean } | null;
  clause?: string | null;
  page?: number | null;
  page_end?: number | null;
  line?: number | null;
  newer_version?: number | null;
  reader_path: string;
}
