
export enum AppView {
  KNOWLEDGE_CHAT = 'knowledge_chat',
  TRANSCRIPTION = 'transcription',
  SILENT_ASSISTANT = 'silent_assistant',
  PERSONAL_ASSISTANT = 'personal_assistant',
  BOARD_ROOM = 'board_room',
  VOICE_CONVERSATION = 'voice_conversation',
  SETTINGS = 'settings'
}

export type AssistantMode = 'silent_assistant' | 'personal_assistant';

export type HandRaiseAction =
  | 'view_details'
  | 'save_for_later'
  | 'ignore'
  | 'ask_follow_up'
  | 'speak_now';

export type AssistantInsightActionStatus =
  | 'ignored'
  | 'saved_for_later'
  | 'viewed'
  | 'asked_follow_up'
  | 'spoke_now';

export enum PersonaType {
  TEACHER    = 'Teacher / Professor',
  FINANCIAL  = 'Financial Advisor',
  FUNNY      = 'Funny & Calming Assistant',
  LAWYER     = 'Lawyer',
  AI_EXPERT  = 'AI Expert & Manager',
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

/** Piper TTS voice id (en_US model name, e.g. en_US-lessac-medium). */
export type PiperVoiceId = string;

export type AssistantClassification =
  | 'supported'
  | 'contradicted'
  | 'related'
  | 'missing_context'
  | 'warning';

export type AssistantPriority = 'low' | 'medium' | 'high';

export type AssistantEvidenceSourceType = 'document' | 'transcript' | 'book' | 'faq' | 'unknown';

export interface AssistantEvidence {
  source_name: string;
  source_type: AssistantEvidenceSourceType;
  doc_id?: string | null;
  chunk_id?: string | null;
  page?: number | null;
  section?: string | null;
  matched_text: string;
}

export interface AssistantInsight {
  id: string;
  transcript_text: string;
  classification: AssistantClassification;
  confidence: number;
  start_char?: number | null;
  end_char?: number | null;
  paragraph_id?: string | null;
  show_highlight: boolean;
  show_hand_raise: boolean;
  priority: AssistantPriority;
  evidence: AssistantEvidence[];
  assistant_interpretation: string;
  suggested_action: string;
  suggested_response?: string | null;
  /** Set when loaded from SQLite or after bulk-save. */
  action_status?: AssistantInsightActionStatus | null;
  created_at?: string | null;
  persisted?: boolean;
}

/** In-memory Personal Assistant session UI (no persistence). */
export interface HandRaiseSessionState {
  dismissedInsightIds: string[];
  savedForLater: AssistantInsight[];
}

export interface AssistantAnalysisScope {
  documents: boolean;
  transcripts: boolean;
  books: boolean;
  faqs: boolean;
}

export interface AssistantAnalyzeRequest {
  session_id: string;
  mode: 'silent_assistant' | 'personal_assistant';
  transcript_window: string;
  rolling_context: string;
  analysis_scope: AssistantAnalysisScope;
}

export interface AssistantAnalyzeResponse {
  session_id: string;
  mode: string;
  insights: AssistantInsight[];
}

export interface AssistantBulkSaveResponse {
  session_id: string;
  inserted: number;
  skipped: number;
  duplicate_merged: number;
  id_map: Record<string, string>;
}

export interface AssistantSessionInsightsResponse {
  session_id: string;
  insights: AssistantInsight[];
}

export interface BoardRoomSttStatus {
  available: boolean;
  model_name: string;
  loaded: boolean;
  cached?: boolean;
  using_fallback?: boolean;
  fallback_model_name?: string | null;
  load_error?: string | null;
  import_error?: string | null;
}

export interface BoardRoomKnowledgeCheck {
  claim: string;
  classification: string;
  confidence: number;
  interpretation: string;
  suggested_action: string;
  evidence: AssistantEvidence[];
}

export interface BoardRoomReport {
  report_id: string;
  session_id: string;
  title: string;
  session_name: string;
  session_location: string;
  polished_transcript: string;
  executive_summary: string;
  knowledge_checks: BoardRoomKnowledgeCheck[];
  markdown: string;
}

export type BoardRoomExportFormat = 'pdf' | 'pptx';

export interface AppSettings {
  /** Piper voice: e.g. en_US-lessac-medium, en_US-ryan-medium */
  voiceName: PiperVoiceId;
  contextWindow: '24h' | '48h' | '1w' | 'all';
  persona: PersonaType;
  model: string;
  developerMode: boolean;
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
}

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
