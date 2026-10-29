/** Types for the EchoMind Law API (backend/app/legal/*, backend/app/api/routes/auth.py). */
export type MatterRole = 'lead' | 'attorney' | 'paralegal' | 'reviewer' | 'read_only';
export type FirmRole = 'firm_admin' | 'managing_partner' | 'attorney' | 'paralegal' | 'reviewer' | 'records_admin' | 'read_only' | 'intake' | 'conflicts';
export type Sensitivity = 'standard' | 'restricted' | 'highly_restricted';
export type MatterStatus = 'open' | 'closed' | 'archived';

export interface Me {
  user: { id: string; username: string; role: string; roles: FirmRole[] };
  session: { auth: string; mfa_enrolled: boolean; must_change_password: boolean; idle_expires_at: number; abs_expires_at: number };
  features?: { intake: boolean; intake_links: boolean; summaries?: boolean; time_entries?: boolean; panel?: boolean; conversation?: boolean; regulatory?: boolean; translation?: boolean; demo_signin?: boolean;
    /** Where the answer model runs, from its configured address (review P1): the interface claims no more than this. */
    model_location?: 'this_server' | 'firm_network' | 'external' };
}
export interface Client { id: string; name: string; code?: string | null; status: string; version: number; }
export interface Matter {
  id: string; number: string; title: string; practice_area?: string | null; jurisdiction?: string | null;
  status: MatterStatus; sensitivity: Sensitivity; version: number; created_at: string; updated_at?: string | null;
  client: { id: string; name: string; code?: string | null }; my_role?: MatterRole | 'break_glass' | null;
  on_hold?: boolean;
}
export interface Membership {
  id: string; user_id: string; username: string; display_name: string; role: MatterRole; can_export: boolean;
  valid_from: string; valid_to?: string | null; granted_at: string;
}
export interface FirmUser { username: string; display_name: string; status: string; roles: FirmRole[]; mfa_enabled: boolean; }
export interface SessionInfo { id: string; created_at: string; last_seen_at?: string; ip?: string; user_agent?: string; mfa_verified: boolean; current: boolean; }
export interface EnrollmentLink { token: string; expires_at: number; path: string; }

export interface Wall {
  id: string; name: string; rationale: string;
  status: 'requested' | 'active' | 'release_requested' | 'released' | 'rejected';
  requested_by: string; requested_at: string; approved_by?: string | null; approved_at?: string | null;
  release_requested_by?: string | null; released_by?: string | null; released_at?: string | null;
  screened: string[]; protected: { type: 'matter' | 'client'; id: string; label: string }[];
  exceptions: { id: string; username: string; matter_id: string; reason: string; expires_at: string; approved: boolean; revoked: boolean; requested_by: string }[];
}
export interface DirectoryMatter { id: string; number: string; title: string; client: { id: string; name: string } }
export interface BreakGlassRequest {
  id: string; matter_number: string; status: 'requested' | 'active' | 'rejected' | 'ended'; minutes: number; reason: string;
  requested_at: string; expires_at?: string | null; requester?: string; decided_by?: string | null; decision_note?: string | null;
  matter?: { number: string; title: string; client: string; sensitivity: string } | null;
}

/** One audit event as the API shows it (S1.5). Identifiers, actions and outcomes only — never content. */
export interface AuditEvent {
  seq: number; ts: string; action: string; outcome: 'allow' | 'deny' | 'error'; reason?: string | null;
  actor?: { username: string; roles: string[] } | null; auth?: string | null; session_ref?: string | null; ip?: string | null;
  object_type?: string | null; object_id?: string | null;
  matter?: { id: string; number?: string; title?: string } | null;
  details?: Record<string, unknown> | null;
}
export interface AuditPage { events: AuditEvent[]; next_before: number | null; shows_matter_numbers?: boolean }
export interface AuditVerification {
  ok: boolean; events: number; head: string | null; head_seq: number | null; broken_at: number | null;
  problem: string | null; anchors_checked: number; verified_at: string;
}

/** Matter documents (S1.6). */
export type DocKind = 'contract' | 'correspondence' | 'pleading' | 'evidence' | 'memo' | 'note' | 'other';
export type DocStatus = 'draft' | 'final' | 'executed' | 'received';
export interface DocVersion {
  id: string; number: number; filename: string; mime: string; size_bytes: number; sha256: string; pages?: number | null;
  uploaded_by?: string | null; uploaded_at: string;
  index_status: 'queued' | 'indexing' | 'indexed' | 'failed' | 'superseded'; index_error?: string | null;
  chunk_count: number; indexed_at?: string | null;
  /** S2.1: the version's own status, reading progress, and what reading found (counts only). */
  status?: DocStatus | null;
  progress?: { stage: 'reading' | 'ocr' | 'indexing' | 'done' | 'failed'; page?: number; pages?: number; passages?: number } | null;
  flags?: VersionFlags;
}
export interface VersionFlags {
  hidden_text: number; instructions: number; tracked_changes: number; comments: number; ocr_pages: number; ocr_low_confidence: number;
}
export interface LawDocument {
  id: string; title: string; kind: DocKind; status: DocStatus; version: number;
  created_by?: string | null; created_at: string; updated_at?: string | null;
  current_version: DocVersion | null; findings: Record<string, number>;
  label?: PrivilegeLabel | null; label_suggested?: boolean;
  source?: { kind: 'upload' } | { kind: 'connector'; name: string | null; state: 'current' | 'removed_in_source'; path: string };
}
export interface Passage {
  chunk_id: string; text: string; score?: number | null; score_kind?: string | null;
  citation: { matter_id: string; document_id: string; document_title: string; version_id: string; version_number: number;
    page_start?: number | null; page_end?: number | null; char_start?: number | null; char_end?: number | null;
    clause?: string | null; line?: number | null; focus_start?: number | null; focus_end?: number | null };
}

/** The document reader (S2.1). Offsets are into `text`. */
export interface ReaderPage { number: number; start: number; end: number; source: 'text' | 'ocr' | 'image' | 'empty'; confidence?: number | null }
export interface ReaderClause { number: string; label: string; heading?: string | null; level: number; parent?: string | null; start: number; end: number; page?: number | null }
export interface ReaderTerm { term: string; start: number; end: number; term_start: number; term_end: number }
export type MarkKind = 'hidden_text' | 'instructions' | 'tracked_insert' | 'tracked_delete' | 'comment' | 'ocr_low_confidence';
export interface ReaderMark { kind: MarkKind; start?: number | null; end?: number | null; page?: number | null; meta: Record<string, any> }
export interface ReaderData {
  text: string; pages: ReaderPage[]; lines: number[]; clauses: ReaderClause[]; terms: ReaderTerm[]; marks: ReaderMark[];
  blocks?: { kind: string; start: number; end: number; label?: string | null; path?: string | null }[];
  engines: Record<string, string>;
  version: { id: string; number: number; filename: string; status?: DocStatus | null; index_status: DocVersion['index_status'] };
}
export interface SearchResult { passages: Passage[]; abstained: boolean }

/** Matter research (S1.7). */
export interface ResearchSource {
  n: number; document_id: string; document_title: string; version_id: string; version_number: number;
  page_start?: number | null; page_end?: number | null; char_start?: number | null; char_end?: number | null; text: string;
  clause?: string | null; line?: number | null; focus_start?: number | null; focus_end?: number | null;
  version_status?: DocStatus | null; signed?: boolean | null;
  /** The passage contains text addressed to an AI; it was withheld from the model (T08). */
  instructions?: boolean;
}
export interface ResearchTurn {
  id: string; question: string; answer: string | null; citations: ResearchSource[];
  status: 'answered' | 'partial' | 'unverified' | 'abstained' | 'scope_revoked' | 'error'; created_at: string;
  /** Sentences of the answer that cite no source, exactly as they appear in it. */
  unsupported?: string[];
  evidence?: ResearchEvidence[]; conflicts?: ResearchConflict[]; versions?: VersionPolicy | null; abstention_reason?: string | null;
  amendments?: ResearchAmendment[];
  /** A decline: what the documents were found not to state, in the model's words, without a source. */
  absence?: string | null;
}
/** S2.2: answers come from signed versions (default) or the latest drafts. */
export type VersionPolicy = 'signed' | 'current';
export type Verification = 'verbatim' | 'normalized' | 'failed' | 'not_quoted';
export interface ResearchEvidence {
  n: number; quote: string; verification: Verification; document_id: string; document_title: string; version_id: string;
  version_number: number; version_status?: DocStatus | null; signed?: boolean | null; clause?: string | null;
  page_start?: number | null; page_end?: number | null; line?: number | null; char_start?: number | null; char_end?: number | null;
  removed?: boolean;
  /** D-073: the answer quoted this word for word without a reference; the system found where and cited it. */
  matched?: boolean;
}
/** A cited clause another document of the matter replaced or changed (EchoMind Legal), found in the documents' own words. */
export interface ResearchAmendment {
  document_id: string; document_title: string; version_id: string; clause: string; n: number; how: string; cited: boolean; amendment_cited: boolean;
  by: { document_id: string; document_title: string; version_id: string; version_number?: number; clause?: string | null; signed: boolean; n: number;
        char_start?: number | null; char_end?: number | null; effective?: { kind: string; date?: string | null; words: string } | null };
}
export interface ResearchConflict {
  document_id: string; document_title: string; clause: string;
  version: { id: string; number: number; status?: DocStatus | null; text: string };
  other: { id: string; number: number; status?: DocStatus | null; text: string };
}
export interface ResearchParagraph {
  index: number; text: string; unsupported: string[]; failed: string[];
  claims: { sentence: string; state: 'supported' | 'failed' | 'unquoted'; citations: { n: number; quote: string | null; verification: Verification }[] }[];
}

/** Compare and contradictions (S2.3). Offsets are into each version's text; marks are relative to the side's text. */
export type PairKind = 'unchanged' | 'formatting' | 'changed' | 'added' | 'removed';
export type ChangeFlag = 'number' | 'obligation' | 'negation';
export interface CompareVersion {
  document_id: string; document_title: string; version_id: string; number: number; status?: DocStatus | null; filename: string;
}
export interface CompareSide {
  start: number; end: number; label?: string | null; clause?: string | null; heading?: string | null; kind: string;
  text: string; page?: number | null; line?: number | null;
}
export interface CompareQuote {
  n: number; side: 'left' | 'right'; quote: string; verification: Verification; char_start?: number | null; char_end?: number | null;
}
export interface CompareExplanation {
  text: string; status: 'answered' | 'partial' | 'unverified'; evidence: CompareQuote[]; failed: string[]; unsupported: string[];
}
export interface ComparePair {
  index: number; kind: PairKind; moved: boolean; how: 'same' | 'wording' | 'moved' | 'meaning'; similarity: number; renumbered: boolean;
  left: CompareSide | null; right: CompareSide | null; left_marks: [number, number][]; right_marks: [number, number][];
  changes: { from: string; to: string }[]; flags: ChangeFlag[]; explanation?: CompareExplanation | null;
}
export interface Comparison {
  id: string; kind: 'versions'; status: string; created_at: string; left: CompareVersion; right: CompareVersion;
  pairs: ComparePair[]; passes: string[];
  summary: { unchanged: number; formatting: number; changed: number; added: number; removed: number; moved: number; renumbered: number };
}
export interface ScanSide {
  document_id: string; document_title: string; version_id: string; version_number: number; version_status?: DocStatus | null;
  signed: boolean; clause?: string | null; page?: number | null; line?: number | null; value: string; value_start: number; value_end: number;
  sentence_start: number; sentence_end: number; context_start: number; context_end: number; context: string; parties?: string[];
}
export type FactKind = 'money' | 'period' | 'percent' | 'date' | 'law' | 'entity' | 'reference';
export interface ScanFinding {
  kind: FactKind; score: number; verdict: 'conflict' | 'no_conflict' | 'unclear'; explanation: string;
  /** "rule": a company's legal form or an agreement's date, reported as found; "model": checked in context. */
  method: 'rule' | 'model'; subject?: string;
  evidence: { n: number; quote: string; verification: Verification; char_start?: number | null; char_end?: number | null }[];
  failed: string[]; a: ScanSide; b: ScanSide; error?: boolean;
}
export interface ConflictScan {
  id: string; kind: 'scan'; status: string; created_at: string; versions: VersionPolicy;
  documents: { document_id: string; document_title: string; version_id: string; number: number; status?: DocStatus | null }[];
  findings: ScanFinding[]; rejected: ScanFinding[]; candidates: number; hidden: number;
}
export type ComparisonSummary =
  | { id: string; kind: 'versions'; status: string; created_at: string; left: CompareVersion; right: CompareVersion; summary: Comparison['summary'] }
  | { id: string; kind: 'scan'; status: string; created_at: string; versions: VersionPolicy; documents: number; conflicts: number };

/** Meetings, consent and capture (S1.8). */
export type MeetingKind = 'client_call' | 'internal' | 'negotiation' | 'interview' | 'board' | 'hearing' | 'other';
export type ParticipantRole = 'host' | 'attorney' | 'client' | 'counterparty' | 'witness' | 'expert' | 'other';
export type ConsentMethod = 'verbal_on_record' | 'written' | 'host_attestation';
export interface MeetingSummary {
  id: string; title: string; kind: MeetingKind; scheduled_at?: string | null; status: 'planned' | 'ended';
  created_at: string; ended_at?: string | null; consent: { granted: number; needed: number };
}
export interface Participant {
  id: string; name: string; role: ParticipantRole;
  consent: { state: 'granted' | 'withdrawn' | 'missing' | 'stale'; method?: ConsentMethod | null; at?: string | null; by?: string | null };
}
export interface CaptureRecord {
  id: string; started_at: string; started_by?: string | null; consent_snapshot: string; transcription: 'available' | 'unavailable';
  custody_records: number; audio_bytes: number; segments: number;
  events: { kind: 'started' | 'paused' | 'resumed' | 'stopped' | 'frame_rejected'; reason?: string | null; at: string; by?: string | null }[];
}
export interface MeetingDetail extends MeetingSummary {
  assist?: AssistState; agenda?: AgendaItem[];
  policy: { mode: string; methods: ConsentMethod[]; max_age_hours: number; statement: string; source: string };
  participants: Participant[]; captures: CaptureRecord[];
}
export interface Segment {
  id: string; capture_id?: string; seq: number; t_start_ms?: number | null; t_end_ms?: number | null; text: string;
  /** S2.4: the speech model's text as produced; `text` is after corrections. */
  raw_text?: string; speaker?: string | null; speaker_participant_id?: string | null; raw_speaker_participant_id?: string | null;
  wall_start?: string | null; wall_end?: string | null; revised?: boolean; revisions?: Revision[];
}
export interface Revision {
  id: string; kind: 'text' | 'speaker'; text?: string | null; speaker_participant_id?: string | null; reason: string;
  source: 'person' | 'vocabulary'; provenance: Record<string, any>; by?: string | null; at: string;
}
/** Boardroom legal mode (S2.6): agenda, minutes and their review (D-042). */
export interface AgendaItem { id: string; position: number; title: string; documents: { id: string; title: string }[] }
export interface MinutesSource {
  segment_id: string; line: number; time?: string | null; speaker?: string | null; agenda_item?: string | null; quote: string; verification: Verification;
}
export interface MinutesItem {
  id: string; kind: 'decision' | 'action' | 'unresolved' | 'discussion'; text: string; owner?: string | null; deadline?: string | null;
  sources: MinutesSource[]; status: 'verified' | 'not_verified' | 'added_by_person'; edited_by?: string | null; by_rule?: boolean;
}
export interface MinutesContent {
  meeting: { id: string; title: string; kind: string; date?: string | null }; attendees: { name: string; role: string }[];
  sections: { agenda_item: string | null; title: string; items: MinutesItem[] }[];
  summary: Record<string, number>; gaps: string[];
}
export interface ReviewDecision { decision: 'approved' | 'changes_requested'; content_sha256: string; reason?: string | null; by?: string | null; at: string }
/** An AI draft under review (S2.6–S2.7, D-042/D-043): fixed versions, hashed, approved by a second lawyer. */
export type ArtifactKind = 'minutes' | 'research_note' | 'legal_draft' | 'redline' | 'summary' | 'panel';
export interface ArtifactVersion<C = unknown> {
  id: string; version: number; status: 'draft' | 'approved' | 'superseded' | 'withdrawn'; content: C; content_sha256: string;
  inputs: Record<string, any>; created_by: string; created_by_name?: string | null; created_at: string; origin: 'model' | 'person';
  document_id?: string | null; decisions: ReviewDecision[]; kind?: ArtifactKind; title?: string; subject_id?: string;
}
export type MinutesVersion = ArtifactVersion<MinutesContent>;
export interface ResearchNoteContent {
  question: string; answer: string; status: 'answered' | 'partial' | 'unverified' | 'abstained'; abstention_reason?: string | null;
  versions: VersionPolicy; unsupported: string[]; evidence: ResearchEvidence[]; conflicts: ResearchConflict[];
  sources: Partial<ResearchSource>[]; asked_by: string; asked_at: string;
}
export type ResearchNote = ArtifactVersion<ResearchNoteContent>;
export interface ArtifactView<C = unknown> {
  artifact: ArtifactVersion<C>;
  versions: { id: string; version: number; status: ArtifactVersion['status']; created_by_name?: string | null; created_at: string; origin: string }[];
  can: { approve: boolean; file: boolean; author: boolean };
}
/** The reviews queue (S2.7): what waits for you, and what of yours waits for others. */
export type ReviewKind = ArtifactKind | 'export' | 'deletion' | 'hold' | 'wall' | 'emergency_access';
export interface ReviewItem {
  kind: ReviewKind; id: string; title: string; version?: number; matter: { id?: string; number: string; title: string } | null;
  by: string | null; at: string | null; path: string; state?: 'waiting' | 'changes_requested'; reason?: string | null; decided_by?: string | null;
}
export interface ReviewQueue { waiting: ReviewItem[]; mine: ReviewItem[]; count: number }

/** Silent Assist (S2.5): evidence-relative alerts during a meeting (D-015, D-041). */
export type AssistPack = 'legal_client_call' | 'legal_negotiation' | 'legal_governance';
export interface AssistAlert {
  id: string; meeting_id: string; segment_id: string; pack: AssistPack; rule: string; verdict: 'differs' | 'matches' | 'needs_review';
  severity: 'critical' | 'notice'; title: string; explanation?: string | null; confidence: number; threshold: number; method: 'rule' | 'model';
  status: 'open' | 'confirmed' | 'dismissed'; decided_by?: string | null; decided_at?: string | null; decision_reason?: string | null;
  heard_at: string; raised_at: string;
  said: { segment_id: string; start: number; end: number; quote: string };
  source?: ({ document_id?: string; document_title?: string; version_id?: string; version_number?: number; version_status?: DocStatus | null;
    signed?: boolean; clause?: string | null; page?: number | null; line?: number | null; start?: number; end?: number; quote?: string;
    segment_id?: string } & { removed?: boolean }) | null;
}
export interface AssistState { pack: AssistPack | null; chosen: string | null; label: string | null; rules: { id: string; label: string; severity: string }[] }
export interface AssistTelemetry {
  alerts: number; critical: number; per_hour: number | null; confirmed: number; dismissed: number; dismiss_rate: number | null;
  latency_p50_s: number | null; latency_max_s: number | null; threshold: number; thresholds_approved: boolean;
}
export interface CaptureMark { kind: 'speaker' | 'degraded' | 'recovered' | 'gap'; participant_id?: string | null; detail?: string | null; audio_ms?: number | null; at: string; capture_id: string }
export interface TranscriptData {
  segments: Segment[]; marks: CaptureMark[]; participants: { id: string; name: string; role: string; removed: boolean }[];
  finalised: { document_id: string; at: string; stale: boolean } | null;
}
export interface VocabularySuggestion {
  id: string; segment_id: string; start: number; end: number; heard: string; suggested: string; kind: 'participant' | 'company' | 'defined_term';
  score: number; text: string;
  source: { kind: 'participant'; participant_id: string } | { kind: 'document'; document_id: string; document_title: string; version_id: string; version_number: number };
}

/** Records (S1.9): labels, holds, deletions, retention. */
export type PrivilegeLabel = 'privileged' | 'work_product' | 'confidential' | 'not_privileged' | 'unreviewed';
export interface LabelEntry { label: PrivilegeLabel; kind: 'set' | 'suggested'; source: 'person' | 'detector'; reason_code?: string | null; by?: string | null; at: string }
export interface LabelState { current: LabelEntry | null; suggestions: LabelEntry[]; history: LabelEntry[] }
export interface DeletionReport {
  object: 'document' | 'meeting_transcript'; versions?: number; passages?: number; findings?: number; labels?: number; files?: number;
  file_bytes?: number; index_vectors?: number; research_citations?: number; comparisons?: number; segments?: number;
  revisions?: number; finalised_document?: number; kept: string[];
}
export interface Deletion {
  id: string; object_type: 'document' | 'meeting_transcript'; object_id: string; object_label: string | null; reason: string;
  origin: 'person' | 'retention'; status: 'requested' | 'purged' | 'rejected' | 'restored' | 'blocked';
  requested_by?: string | null; requested_at: string; decided_by?: string | null; decided_at?: string | null; decision_note?: string | null;
  purged_at?: string | null; report: DeletionReport;
  verification: { ok: boolean; checks: Record<string, boolean>; verified_at: string } | null;
  matter: { id: string; number: string; title: string } | null;
}
export interface Hold {
  id: string; name: string; reason: string; status: 'requested' | 'active' | 'release_requested' | 'released' | 'rejected';
  requested_by?: string | null; requested_at: string; approved_by?: string | null; approved_at?: string | null;
  release_requested_by?: string | null; released_by?: string | null; released_at?: string | null;
  matters: { id: string; number: string; title: string; client: string }[];
}
export interface RetentionPolicy {
  id: string; object_type: 'document' | 'meeting_transcript'; days_after_close: number; note?: string | null;
  status: 'draft' | 'approved' | 'retired'; drafted_by?: string | null; drafted_at: string; approved_by?: string | null; approved_at?: string | null;
}
export interface RetentionRun {
  id: string; mode: 'dry_run' | 'apply'; run_by: string; run_at: string;
  report: { candidates: number; held_skipped: number; draft_preview: number; requests_created: number };
}

/** Exports (S1.10). */
export interface PlanItem { id: string; kind: string; text: string; count: number; redact: boolean; reason: string; source: 'detector' | 'person' }
export interface LawExport {
  id: string; object_type: 'document' | 'meeting_transcript'; object_id: string; title: string; format: 'pdf' | 'docx'; purpose: string;
  status: 'draft' | 'pending_approval' | 'ready' | 'failed' | 'rejected' | 'expired';
  plan: {
    items: PlanItem[]; source: 'pdf' | 'docx' | 'text'; warnings: { image_pages: number[] };
    /** S2.7: the document was filed from an approved AI draft; the export carries that approval. */
    approval?: { kind: ArtifactKind; version: number; approved_by: string | null; approved_at: string | null; content_sha256: string };
  };
  source_sha256?: string | null;
  label?: string | null; requested_by?: string | null; requested_at: string; submitted_at?: string | null;
  decided_by?: string | null; decided_at?: string | null; decision_note?: string | null; watermark?: string | null;
  output_bytes?: number | null; output_sha256?: string | null;
  verification?: { ok: boolean; checks?: Record<string, boolean>; pages?: number; source_changed?: boolean } | null; expires_at?: string | null; downloads: number;
}

/** Connectors (S1.11). */
export interface ConnectorRun { started_at: string; finished_at?: string | null; status: 'ok' | 'partial' | 'failed'; report: Record<string, number | string | null> }
export interface Connector {
  id: string; kind: 'local_folder'; name: string; status: 'active' | 'disabled'; sync_minutes: number; stale_after_minutes: number;
  last_sync_at?: string | null; last_ok_at?: string | null; last_status?: 'ok' | 'partial' | 'failed' | null; last_error?: string | null;
  fresh: boolean; links: number; documents: number; runs: ConnectorRun[];
}
export interface ConnectorLink {
  id: string; connector_id: string; connector_name: string; remote_path: string; created_by?: string | null; created_at: string;
  connector_status: 'active' | 'disabled'; last_ok_at?: string | null; last_status?: string | null;
}

/** Evaluation runs (S2.8, D-044): what was measured, on which code, models and data. */
export interface EvaluationRun {
  id: string; suite: string; set: string; commit: string | null; models: Record<string, string>; dataset_sha256: string | null;
  summary: Record<string, unknown>; report: string; report_sha256: string; run_at: string | null; imported_at: string;
}

/** Firm templates and legal drafts (S3.1, D-045). */
export type TemplateKind = 'engagement_letter' | 'nda' | 'demand_letter' | 'memo' | 'contract_review' | 'board_minutes' | 'custom';
export type FieldSource = 'documents' | 'matter' | 'person' | 'today';
export interface TemplateField { key: string; label: string; kind: string; source: FieldSource; question?: string | null; from?: string | null }
export interface TemplateVersion {
  id: string; version: number; status: 'draft' | 'approved' | 'retired'; body: string; fields: TemplateField[]; styles: string | null;
  content_sha256: string; created_by: string | null; created_at: string; approved_by: string | null; approved_at: string | null;
}
export interface FirmTemplate { id: string; name: string; kind: TemplateKind; preset: string | null; created_by: string | null; created_at: string; versions: TemplateVersion[] }
export interface DraftSource {
  document_id: string; document_title: string; version_id: string; version_number: number; version_status?: string | null; clause?: string | null;
  quote: string; verification: Verification; char_start?: number | null; char_end?: number | null;
}
export interface DraftField {
  key: string; label: string; kind: string; source: FieldSource; value: string | null;
  status: 'sourced' | 'matter' | 'system' | 'entered' | 'needs_input'; sources: DraftSource[];
  suggestion?: { value: string; reason: string; quote?: string } | null; edited_by?: string | null;
}
export interface DraftBlock {
  type: 'heading' | 'paragraph'; level?: number; origin: 'template' | 'written' | 'edited'; text: string; raw?: string;
  instruction?: string; sources?: DraftSource[]; unsupported?: string[]; edited_by?: string | null;
}
export interface DraftContent {
  title: string; matter: { number: string; title: string; client: string }; fields: DraftField[]; blocks: DraftBlock[]; gaps: number;
  template: { id: string; name: string; kind: TemplateKind; version: number; version_id: string; content_sha256: string };
  summary: { fields: number; sourced: number; needs_input: number; written: number; unsupported: number };
}
export type LegalDraft = ArtifactVersion<DraftContent>;
export interface DraftSummary {
  id: string; draft_id: string; title: string; version: number; status: ArtifactVersion['status']; gaps: number;
  template: DraftContent['template']; created_by_name?: string | null; created_at: string; document_id?: string | null; sent_back: boolean;
}

/** Playbooks and playbook checks (S3.2, D-046). */
export interface PlaybookPosition {
  id: string; topic: string; topic_label: string; version: number; status: 'draft' | 'approved' | 'retired'; title: string; standard: string;
  fallbacks: { text: string; note: string | null }[]; guidance: string | null; content_sha256: string; created_by: string | null;
  created_at: string; approved_by: string | null; approved_at: string | null; owner_name: string | null;
}
export interface Playbook { id: string; name: string; description: string | null; jurisdiction: string | null; created_by: string | null; created_at: string; positions: PlaybookPosition[] }
export interface PlaybookProposal { clause: string; title: string | null; text: string; topic: string | null }
export type CheckClass = 'standard' | 'fallback' | 'outside' | 'not_found' | 'unclear';
export interface CheckLine {
  topic: string; title: string; topic_label: string; classification: CheckClass; matched: number | null;
  position: { id: string; version: number; standard: string; fallbacks: { text: string; note: string | null }[] };
  clause: { number: string | null; heading: string | null; start: number; end: number } | null;
  quote: { text: string; start: number; end: number; verification: Verification } | null; reason: string | null; method: 'model' | 'rule';
  decision: { decision: 'agree' | 'disagree'; reason: string | null; by: string | null; at: string } | null;
}
export interface PlaybookCheck {
  id: string; document_id: string; version_id: string; playbook: { id: string; name: string | null }; results: CheckLine[];
  summary: Record<CheckClass, number>; created_by: string | null; created_at: string; advisory: true; inputs: Record<string, any>;
}

/** Citation and quote checks (S3.4, D-048). External authorities are never verified (P-009). */
export type CitationStatus = 'resolved' | 'missing' | 'not_found' | 'accurate' | 'altered' | 'unverified';
export interface CitationItem {
  id: string; kind: 'reference' | 'quote' | 'authority'; type: string; text: string; start: number; end: number; status: CitationStatus;
  target: { document_id: string; version_id: string; title: string; clause: string | null; start: number; end: number; passage: string } | null;
  note: string | null; diff: [('eq' | 'del' | 'ins'), string][]; malformed: string[]; cited_as?: string | null;
  fields?: Record<string, string>; notes: { note: string; by: string | null; at: string }[];
}
export interface CitationCheck {
  id: string; document_id: string; version_id: string; version_number: number | null; items: CitationItem[]; engine: string;
  summary: { references: number; resolved: number; missing: number; quotes: number; accurate: number; altered: number; not_found: number; authorities: number; malformed: number };
  external_lookup: false; notice: string; created_by: string | null; created_at: string;
}

/** Redlines (S3.3, D-047): revisions to the firm's positions, written as Word tracked changes. */
export interface RedlineChange {
  id: string; topic: string; title: string; clause: { number: string | null; heading: string | null; start: number; end: number };
  original: string; revised: string; ops: [('eq' | 'del' | 'ins'), string][]; reason: string; comment: string;
  fitted: boolean; fit_note: string | null; edited_by?: string | null;
  from: { kind: 'standard' | 'fallback' | 'fix'; n: number | null; position_id: string | null; position_version: number | null; text: string; findings?: string[] };
}
export interface RedlineContent {
  title: string; document: { id: string; title: string; version_id: string; version_number: number; filename: string; mime: string };
  /** S3.5: corrections picked from a definitions check have no playbook. */
  origin?: 'definitions'; check_id: string; playbook: { id: string; name: string | null } | null; changes: RedlineChange[]; summary: { changes: number; fitted: number };
}

/** Defined terms and cross-references (S3.5, D-049). Rules only; fixes become a redline for approval. */
export type DefinitionFindingType = 'duplicate' | 'unused' | 'undefined' | 'inconsistent' | 'before_definition' | 'gap' | 'repeat' | 'broken' | 'outdated';
export interface DefinitionFix { label: string; start: number; end: number; original: string; revised: string; clause: string | null }
export interface DefinitionFinding {
  id: string; kind: 'definition' | 'numbering' | 'reference'; type: DefinitionFindingType; severity: 'high' | 'medium' | 'low';
  text: string; start: number; end: number; message: string; where?: string; places: { start: number; end: number; where?: string }[]; fixes: DefinitionFix[];
  conflicting?: boolean; notes: { note: string; by: string | null; at: string }[];
}
export interface DefinitionCheck {
  id: string; document_id: string; version_id: string; version_number: number | null; findings: DefinitionFinding[]; engine: string; advisory: true;
  summary: Record<'definitions' | DefinitionFindingType, number>; created_by: string | null; created_at: string;
}

/** Obligations and deadlines (S3.6, D-050): read by rules, proposed until a lawyer confirms. */
export type ObligationKind = 'renewal' | 'termination' | 'change_of_control' | 'consent' | 'payment' | 'notification' | 'post_termination' | 'confidentiality' | 'notice';
export type ObligationStatus = 'proposed' | 'confirmed' | 'done' | 'dismissed';
export interface Obligation {
  id: string; document_id: string; document_title: string | null; version_id: string; clause: string | null; kind: ObligationKind;
  party: string | null; action: string; when: string | null; due_date: string | null; due_note: string | null; timezone: string;
  due_at: string | null; overdue: boolean; status: ObligationStatus; owner: string | null; source: 'rules' | 'model' | 'person';
  evidence: { start: number; end: number; quote: string }; confirmed_by: string | null; confirmed_at: string | null; version: number;
  created_by: string | null; created_at: string; updated_at: string;
}
export interface ObligationEvent { action: string; changes: Record<string, [unknown, unknown]>; reason: string | null; by: string | null; at: string }

/** Due diligence (S3.7, D-051): firm checklists, runs over a matter's documents, and an issue register. */
export type DdQuestion = 'change_of_control' | 'assignment' | 'termination' | 'governing_law' | 'liability' | 'renewal';
export interface DdItem { id: string; section: string; title: string; expect: string[]; questions: DdQuestion[] }
export type DdGovernance = 'share_capital' | 'authority' | 'constitution' | 'ip_chain';
export interface DdChecklist {
  id: string; family_id: string; version: number; name: string; expected_law: string | null; items: DdItem[]; governance?: DdGovernance[]; content_sha256: string;
  status: 'draft' | 'approved' | 'retired'; created_by: string | null; created_at: string; approved_by: string | null; approved_at: string | null;
}
export interface DdIssue {
  id: string; run_id: string; item_id: string; document_id: string; document_title: string; version_id: string; kind: string;
  severity: 'high' | 'medium' | 'low'; title: string; detail: string | null; evidence: { start: number; end: number; quote: string; clause: string | null };
  status: 'preliminary' | 'verified' | 'dismissed'; reviewer: string | null; decided_at: string | null; version: number; created_at: string;
  /** S3.8: the second document a governance finding quotes (an approval, the delegation of authority, the articles). */
  related?: { document_id: string; document_title: string; version_id: string; start: number; end: number; quote: string }[];
}
export interface DdSummary { items: number; missing: number; issues: number; high: number; medium: number; low: number; preliminary: number; verified: number; dismissed: number }
export interface DdRunInfo {
  id: string; status: 'queued' | 'running' | 'done' | 'failed'; error: string | null; checklist: { id: string; name: string | null; version: number | null };
  created_by: string | null; created_at: string; finished_at: string | null; documents_read?: number | null; summary?: DdSummary;
}
export interface DdRunView {
  run: DdRunInfo; items: { item_id: string; title: string; section: string; documents: { id: string; title: string }[]; missing: boolean }[];
  missing: { item_id: string; title: string; section: string }[]; issues: DdIssue[]; summary: DdSummary;
}

// ── Intake (S4.1, D-054) ──────────────────────────────────────────────────────────
export type PartyRole = 'adverse' | 'related' | 'client_group' | 'other';
export interface MatterParty { id: string; name: string; role: PartyRole; added_by: string | null; added_at: string }
export type IntakeQuestionKind = 'text' | 'email' | 'phone' | 'choice' | 'names' | 'longtext' | 'date';
export interface IntakeQuestion {
  key: string; label: string; kind: IntakeQuestionKind; required: boolean; max?: number; max_items?: number;
  choices?: Record<string, string>; ask: string; thanks: string;
}
export interface IntakeScript { version: string; questions: IntakeQuestion[] }
export interface IntakeDisclaimer { version: string; text: string; accepted_at?: string | null; accepted?: boolean }
export type IntakeStatus = 'invited' | 'open' | 'submitted' | 'cleared' | 'conflict' | 'converted' | 'closed' | 'purged';
export interface IntakeScreening { result: 'not_run' | 'no_match' | 'possible_conflict'; message: string; decision: 'cleared' | 'conflict' | null }
export type IntakeTemplate = 'acknowledge' | 'can_help' | 'cannot_act' | 'more_info';
export interface IntakeReply {
  id: string; template: IntakeTemplate; body: string; body_sha256: string; status: 'draft' | 'released' | 'withdrawn';
  drafted_by: string | null; drafted_at: string; approved_by: string | null; approved_at: string | null;
}
export type IntakeAnswers = Record<string, string | string[]>;
export interface Intake {
  id: string; reference: string; channel: 'phone' | 'in_person' | 'link'; status: IntakeStatus; created_at: string; created_by: string | null;
  updated_at: string | null; version: number; disclaimer: IntakeDisclaimer; script: IntakeScript; answers: IntakeAnswers;
  screening: IntakeScreening; consent: { method: 'verbal' | 'written'; recorded_by: string | null; recorded_at: string } | null;
  recording: { mime: string; bytes: number; uploaded_by: string | null; uploaded_at: string } | null; replies: IntakeReply[];
  events: { action: string; actor: string | null; at: string; note?: string }[]; matter_number: string | null;
  link: { expires_at: string | null; submitted_at: string | null } | null;
}
export interface IntakeMatch { searched: string; matched: string; kind: 'client' | PartyRole; score: number; client: string | null; matter_number: string | null }
export interface IntakeView { intake: Intake; matches?: IntakeMatch[] }
export interface IntakeSummary {
  id: string; reference: string; channel: Intake['channel']; status: IntakeStatus; name: string | null; organisation: string | null;
  screening: IntakeScreening; created_at: string; updated_at: string | null;
}
export interface IntakeCan { manage: boolean; approve: boolean; conflicts: boolean; purge: boolean; convert: boolean; links: boolean }
export interface IntakeLinkPage {
  status: 'invited' | 'submitted' | 'replied'; reference: string; disclaimer: IntakeDisclaimer; script: IntakeScript; reply: { body: string } | null;
}

// ── Summaries (S4.2, D-055) ───────────────────────────────────────────────────────
export interface SummaryCite { page?: number; line?: number; clause?: string; timecode?: string | null }
export interface SummaryPoint {
  id: string; text: string; quote: string; verification: 'verbatim' | 'normalized' | 'failed'; start: number | null; end: number | null;
  cite: SummaryCite | null; figures_unsupported: string[]; origin: 'model' | 'person';
}
export interface SummaryContent {
  title: string; document: { id: string; title: string; version_id: string; version_number: number }; transcript: boolean;
  points: SummaryPoint[]; sections: number; unverified: number; figures_unsupported: number; translation: { available: boolean; reason: string };
}
export type SummaryVersion = ArtifactVersion<SummaryContent>;

// ── Time entries (S4.3, D-056) ────────────────────────────────────────────────────
export interface TimeEntry {
  id: string; work_date: string; matter: { id: string; number: string; title: string; client: string }; source: 'suggested' | 'manual';
  minutes: number; hours: string; minutes_suggested: number | null; narrative: string; narrative_suggested: string | null;
  sessions: { from: string; to: string }[]; counts: Record<string, number>; edited: boolean; status: 'draft' | 'approved' | 'discarded';
  approved_at: string | null; updated_at: string | null; version: number;
}

// ── Review panel (S4.5, D-057) ────────────────────────────────────────────────────
export type PanelRole = 'deal' | 'risk' | 'litigation' | 'specialist';
export interface PanelPerspective {
  role: PanelRole; label: string; text: string; evidence: ResearchEvidence[]; unsupported: string[]; failed: string[];
  missing_facts: string[]; abstained: boolean;
}
export interface PanelIssue { issue: string; positions: Record<PanelRole, string>; disagreement: boolean; summary: string | null; figures_unsupported: string[] }
export interface PanelContent {
  title: string; question: string; specialist: string; area: string; perspectives: PanelPerspective[]; matrix: PanelIssue[];
  missing_facts: string[]; stopped: boolean; facts_given: { question: string; answer: string }[]; notice: string; sources: number;
  proceeded_without?: { facts: string[]; by: string; reason: string; at: string };
}
export interface PanelReview {
  id: string; question: string; specialist: string; area: string | null; status: 'queued' | 'running' | 'done' | 'needs_facts' | 'failed' | 'not_allowed';
  facts: { question: string; answer: string }[]; created_by: string | null; created_at: string; finished_at: string | null; mine: boolean;
  version?: number | null; approval?: string | null; documents?: { id: string; title: string | null }[];
}
export interface PanelView {
  review: PanelReview; artifact: ArtifactVersion<PanelContent> | null;
  versions: { id: string; version: number; status: string; created_at: string }[]; can: { approve: boolean; facts: boolean; author: boolean };
}

// Regulatory updates (S4.4, D-068): public law, fetched on the host and matched to open matters.
export type RegulatoryStatus = 'new' | 'reviewed' | 'not_relevant';
export interface RegulatoryUpdate {
  id: string; source: 'legislation.gov.uk' | 'federalregister.gov'; source_name: string; external_id: string; jurisdiction: 'UK' | 'US';
  kind: string; title: string; summary: string; url: string; published: string; subjects: string[]; licence: 'OGL-3.0' | 'public-domain';
}
export interface RegulatoryFirmUpdate extends RegulatoryUpdate { matters: { id: string; number: string; title: string; status: RegulatoryStatus }[] }
export interface RegulatoryMatch {
  update: RegulatoryUpdate; terms: string[]; status: RegulatoryStatus; decided_by: string | null; decided_at: string | null; note: string | null; version: number;
}
