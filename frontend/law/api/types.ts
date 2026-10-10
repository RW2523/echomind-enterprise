/** Types for the EchoMind Law API (backend/app/legal/*, backend/app/api/routes/auth.py). */
export type MatterRole = 'lead' | 'attorney' | 'paralegal' | 'reviewer' | 'read_only';
export type FirmRole = 'firm_admin' | 'managing_partner' | 'attorney' | 'paralegal' | 'reviewer' | 'records_admin' | 'read_only';
export type Sensitivity = 'standard' | 'restricted' | 'highly_restricted';
export type MatterStatus = 'open' | 'closed' | 'archived';

export interface Me {
  user: { id: string; username: string; role: string; roles: FirmRole[] };
  session: { auth: string; mfa_enrolled: boolean; must_change_password: boolean; idle_expires_at: number; abs_expires_at: number };
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
}
/** S2.2: answers come from signed versions (default) or the latest drafts. */
export type VersionPolicy = 'signed' | 'current';
export type Verification = 'verbatim' | 'normalized' | 'failed' | 'not_quoted';
export interface ResearchEvidence {
  n: number; quote: string; verification: Verification; document_id: string; document_title: string; version_id: string;
  version_number: number; version_status?: DocStatus | null; signed?: boolean | null; clause?: string | null;
  page_start?: number | null; page_end?: number | null; line?: number | null; char_start?: number | null; char_end?: number | null;
  removed?: boolean;
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
  plan: { items: PlanItem[]; source: 'pdf' | 'docx' | 'text'; warnings: { image_pages: number[] } };
  label?: string | null; requested_by?: string | null; requested_at: string; submitted_at?: string | null;
  decided_by?: string | null; decided_at?: string | null; decision_note?: string | null; watermark?: string | null;
  output_bytes?: number | null; output_sha256?: string | null;
  verification?: { ok: boolean; checks: Record<string, boolean>; pages?: number } | null; expires_at?: string | null; downloads: number;
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
