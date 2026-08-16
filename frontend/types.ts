
export enum AppView {
  KNOWLEDGE_CHAT = 'knowledge_chat',
  TRANSCRIPTION = 'transcription',
  VOICE_CONVERSATION = 'voice_conversation',
  DOCUMENT_STUDIO = 'document_studio',
  SETTINGS = 'settings'
}

export enum PersonaType {
  TEACHER    = 'Teacher / Professor',
  FINANCIAL  = 'Financial Advisor',
  FUNNY      = 'Funny & Calming Assistant',
  LAWYER     = 'Lawyer',
  AI_EXPERT  = 'AI Expert & Manager',
  GENERAL    = 'General Assistant',
  ECHOMIND   = 'EchoMind Guide',
  CLINICAL   = 'Clinical Assistant',
  BANKING    = 'Banking Advisor',
  MEETING    = 'Meeting Facilitator',
  RETAIL     = 'Retail Advisor',
}

export interface DocumentChunk {
  id: string;
  docName: string;
  content: string;
  metadata: {
    pageNumber?: number;
    section?: string;
    /** Hierarchical section breadcrumb, e.g. "Volume 1 > Chapter 3 > 030201" */
    sectionPath?: string;
    timestamp: number;
    /** Relevance score 0–1 from retrieval pipeline */
    score?: number;
    /** Document type: "book" | "glossary" | "faq" | "transcript" etc. */
    docType?: string;
    /** Raw chunk index in the document */
    chunkIndex?: number;
    /** Backend document ID — used to construct the file-serve URL for in-browser PDF preview */
    docId?: string;
  };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: DocumentChunk[];
  timestamp: number;
}

export interface TranscriptEntry {
  id: string;
  /** Raw transcript text (live transcription uses this) */
  raw?: string;
  /** Same as raw; LiveTranscription uses rawText for new entries */
  rawText?: string;
  refined?: string;
  tags?: string[];
  timestamp: number;
  metadata?: {
    date?: string;
    time?: string;
    topic?: string;
  };
}

// ── Silent Assistant & Boardroom types ───────────────────────────────────────

export type AnalysisLabel =
  | 'Supported'
  | 'Contradicted'
  | 'Unverified'
  | 'Violating'
  | 'Risky Statement'
  | 'Relevant';

export interface SourceChunkPreview {
  chunk_id: string;
  text: string;
  doc_title?: string;
  doc_id?: string;
}

export interface AnalysisCard {
  id: string;
  /** Maps to paragraph_id from the WS segment message */
  segment_id: string;
  segment_text: string;
  label: AnalysisLabel;
  confidence: number;
  explanation: string;
  source_chunks: SourceChunkPreview[];
  created_at?: string;
}

export interface TranscriptSegment {
  paragraph_id: string;
  text: string;
  /** Set when an analysis card exists for this segment */
  label?: AnalysisLabel;
  confidence?: number;
  /** v2: role id of the speaker (from session.roles), null/undefined = unknown */
  role?: Role | null;
  /** v2: sentence split of this segment (char offsets relative to `text`) */
  sentences?: TranscriptSentence[];
}

// ── Silent Assistant v2 (scenario / sentence checks / records) ───────────────

/** Scenario ids known to the backend profiles (+ 'auto' = let the server pick). */
export type ScenarioId = 'auto' | 'customer_care' | 'legal' | 'banking' | 'general' | (string & {});

export type AnalysisMode = 'flags_only' | 'flags_and_records';

/** Role id used on the wire, e.g. 'agent' | 'caller' | 'lawyer' | 'client' | 'banker' | 'speaker_a'. */
export type Role = string;

export type TagTone =
  | 'green' | 'red' | 'yellow' | 'blue' | 'violet' | 'indigo'
  | 'teal' | 'orange' | 'amber' | 'grey' | 'cyan' | (string & {});

export type TagProof = 'quote' | 'record' | 'rule' | 'span' | 'none' | (string & {});

export interface TagSpec {
  id: string;
  label: string;
  tone: TagTone;
  proof: TagProof;
  description?: string;
}

export interface Scenario {
  id: ScenarioId;
  label: string;
  description: string;
  default_namespace: string;
  roles: { me: Role; other: Role };
  analysis_mode_default: AnalysisMode;
  tag_vocab: TagSpec[];
}

export type EvidenceKind = 'document' | 'transcript' | 'rule' | (string & {});

export interface EvidenceQuote {
  quote: string;
  chunk_id?: string | null;
  doc_id?: string | null;
  doc_title?: string | null;
  page?: number | null;
  section_path?: string | null;
  kind: EvidenceKind;
  rule_id?: string | null;
}

export interface CheckTag {
  tag: string;
  label?: string;
  tone?: TagTone;
  confidence?: number;
}

export type CheckVerdict = 'supported' | 'contradicted' | 'unverified' | null;
export type CheckKind = 'claim' | 'personal_detail' | 'commitment' | 'question' | 'filler' | (string & {});
/** Per-sentence lifecycle: pending (analysis_start) -> checked | skipped | timeout | no_tags (analysis_done). */
export type CheckStatus = 'pending' | 'checked' | 'skipped' | 'timeout' | 'no_tags';

/** Superset of AnalysisCard — the v2 `analysis` payload. Legacy fields stay populated by the server. */
export interface SentenceCheck extends AnalysisCard {
  session_id?: string;
  sentence_id: string;
  sentence_text: string;
  char_start?: number;
  char_end?: number;
  role?: Role | null;
  kind?: CheckKind;
  verdict?: CheckVerdict;
  tags: CheckTag[];
  evidence: EvidenceQuote[];
  record_ids: string[];
  searched_docs: string[];
  llm_skipped?: boolean;
  latency_ms?: number;
  /** TTS-ready phrase for the hand-raise readout */
  phrase?: string;
  status?: CheckStatus;
}

export interface DetectedEntity {
  id: string;
  kind: string;
  value: string;
  normalized?: string;
  sentence_id?: string;
  segment_id?: string;
  role?: Role | null;
  confidence?: number;
  subject_id?: string | null;
}

export type SubjectKind = 'customer' | 'client' | 'account_holder' | 'counterparty' | 'person' | (string & {});
export type SubjectStatus = 'candidate' | 'confirmed' | 'rejected';

export interface Subject {
  id: string;
  kind: SubjectKind;
  display_name: string;
  matched_fields: string[];
  entity_ids: string[];
  confidence?: number;
  status: SubjectStatus;
  records_count?: number;
}

export type RecordKind =
  | 'customer_file' | 'contract' | 'ticket' | 'previous_call' | 'matter' | 'related_case'
  | 'account' | 'product' | 'policy' | 'kyc' | 'document' | (string & {});

export interface RecordHit {
  id: string;
  subject_id?: string | null;
  entity_id?: string | null;
  sentence_id?: string;
  kind: RecordKind;
  title: string;
  doc_id?: string | null;
  doc_title?: string | null;
  page?: number | null;
  section_path?: string | null;
  quotes: { text: string; chunk_id?: string | null }[];
  score?: number;
  match?: 'exact' | 'fuzzy' | 'semantic' | (string & {});
  namespace?: string;
  source_transcript_id?: string | null;
}

export interface TranscriptSentence {
  sentence_id: string;
  text: string;
  char_start: number;
  char_end: number;
  role?: Role | null;
}

export interface SessionAck {
  session_id: string;
  scenario: ScenarioId;
  scenario_label: string;
  namespace: string;
  analysis_mode: AnalysisMode;
  kb_docs: number;
  roles: { me: Role; other: Role };
  tag_vocab: TagSpec[];
}

export type WsWarningCode = 'namespace_empty' | 'llm_slow' | 'overloaded' | 'stt_dropped' | (string & {});
export interface WsWarning { code: WsWarningCode; message: string; at: number }

export interface ScenarioSuggestion { scenario: ScenarioId; confidence: number; reason?: string }

/** Derived from checks tagged action-item / commitment / decision. */
export interface ActionItem {
  id: string;
  sentence_id: string;
  tag: 'action-item' | 'commitment' | 'decision' | (string & {});
  role?: Role | null;
  text: string;
  check: SentenceCheck;
}

/** GET /api/transcribe/transcripts/{id}/assistant */
export interface TranscriptAssistantData {
  checks: SentenceCheck[];
  entities: DetectedEntity[];
  subjects: Subject[];
  records: RecordHit[];
  segments: TranscriptSegment[];
}

export interface DiarizedSegment {
  speaker: string;
  text: string;
  start_time?: number;
  end_time?: number;
}

export interface MeetingReport {
  executive_summary?: string;
  speakers?: { speaker: string; summary: string; key_points?: string[] }[];
  key_topics?: string[];
  rag_verified_facts?: string[];
  contradictions?: string[];
  recommendations?: string[];
  overall_sentiment?: string;
  raw_transcript?: string;
}

export interface BoardroomSession {
  id: string;
  transcript_id?: string | null;
  status: 'recording' | 'processing' | 'transcribed' | 'analysing' | 'analysed' | 'error';
  chunk_count?: number;
  diarized_segments?: DiarizedSegment[] | null;
  report?: MeetingReport | null;
  created_at?: string;
  updated_at?: string;
}

/** Piper TTS voice id (en_US model name, e.g. en_US-lessac-medium). */
export type PiperVoiceId = string;

export interface AppSettings {
  /** Piper voice: e.g. en_US-lessac-medium, en_US-ryan-medium */
  voiceName: PiperVoiceId;
  contextWindow: '24h' | '48h' | '1w' | 'all';
  persona: PersonaType;
  /** When true: fast RAG (no query rewriting; single embedding + LLM). When false: full RAG (intent + query rewrite). */
  advancedRag: boolean;
  /** Voice only: when true use knowledge base (RAG); when false answer generally. */
  voiceUseKnowledgeBase: boolean;
  /** Voice: assistant name (e.g. Alex); conversation is natural, like a voice assistant. */
  voiceBotName: string;
  /** Voice: user's name (optional); assistant uses it when it fits naturally. */
  voiceUserName: string;
  /** Voice: context/role (system prompt); persisted until user clears context. */
  voiceContext: string;
  /** Voice TTS engine: 'piper' (fast, default) or 'kokoro' (more natural). */
  ttsEngine?: 'piper' | 'kokoro';
  /** Kokoro voice id (used when ttsEngine === 'kokoro'), e.g. af_heart. */
  kokoroVoice?: string;
}

/** Kokoro-82M voices (American English). */
export const KOKORO_VOICES: { id: string; label: string }[] = [
  { id: 'af_heart', label: 'Heart (F)' },
  { id: 'af_bella', label: 'Bella (F)' },
  { id: 'af_nicole', label: 'Nicole (F)' },
  { id: 'af_sarah', label: 'Sarah (F)' },
  { id: 'am_michael', label: 'Michael (M)' },
  { id: 'am_fenrir', label: 'Fenrir (M)' },
  { id: 'am_adam', label: 'Adam (M)' },
];

/** Piper English (en_US) voices available for TTS. Format: voiceKey -> label. Quality variants in id. */
export const PIPER_VOICES: { id: string; label: string }[] = [
  { id: 'en_US-amy-medium', label: 'Amy (medium)' },
  { id: 'en_US-arctic-medium', label: 'Arctic (medium)' },
  { id: 'en_US-bryce-medium', label: 'Bryce (medium)' },
  { id: 'en_US-danny-low', label: 'Danny (low)' },
  { id: 'en_US-hfc_female-medium', label: 'HFC Female (medium)' },
  { id: 'en_US-hfc_male-medium', label: 'HFC Male (medium)' },
  { id: 'en_US-joe-medium', label: 'Joe (medium)' },
  { id: 'en_US-john-medium', label: 'John (medium)' },
  { id: 'en_US-kathleen-low', label: 'Kathleen (low)' },
  { id: 'en_US-kristin-medium', label: 'Kristin (medium)' },
  { id: 'en_US-kusal-medium', label: 'Kusal (medium)' },
  { id: 'en_US-l2arctic-medium', label: 'L2 Arctic (medium)' },
  { id: 'en_US-lessac-low', label: 'Lessac (low)' },
  { id: 'en_US-lessac-medium', label: 'Lessac (medium)' },
  { id: 'en_US-lessac-high', label: 'Lessac (high)' },
  { id: 'en_US-libritts-high', label: 'LibriTTS (high)' },
  { id: 'en_US-libritts_r-medium', label: 'LibriTTS R (medium)' },
  { id: 'en_US-ljspeech-medium', label: 'LJ Speech (medium)' },
  { id: 'en_US-ljspeech-high', label: 'LJ Speech (high)' },
  { id: 'en_US-norman-medium', label: 'Norman (medium)' },
  { id: 'en_US-reza_ibrahim-medium', label: 'Reza Ibrahim (medium)' },
  { id: 'en_US-ryan-low', label: 'Ryan (low)' },
  { id: 'en_US-ryan-medium', label: 'Ryan (medium)' },
  { id: 'en_US-ryan-high', label: 'Ryan (high)' },
  { id: 'en_US-sam-medium', label: 'Sam (medium)' },
];
