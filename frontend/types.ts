/**
 * EchoMind product modes: every mode listens and transcribes; behavior after transcription differs.
 * Use with `productModeToAppView` / `appViewToProductMode` when bridging navigation.
 */
export enum ProductMode {
  TRANSCRIBE = 'transcribe',
  ASSISTANT = 'assistant',
  SILENT_ASSISTANT = 'silent_assistant',
  CONVERSATION = 'conversation',
}

/**
 * Primary navigation routes. String values are stable for URLs/localStorage compatibility.
 * Transcribe maps from legacy `transcription`; Conversation from `voice_conversation`.
 */
export enum AppView {
  KNOWLEDGE_CHAT = 'knowledge_chat',
  TRANSCRIPTION = 'transcription',
  ASSISTANT = 'assistant',
  SILENT_ASSISTANT = 'silent_assistant',
  VOICE_CONVERSATION = 'voice_conversation',
  SETTINGS = 'settings',
}

/** Map primary views to the four EchoMind listening modes (null for Knowledge Chat / Settings). */
export function appViewToProductMode(view: AppView): ProductMode | null {
  switch (view) {
    case AppView.TRANSCRIPTION:
      return ProductMode.TRANSCRIBE;
    case AppView.ASSISTANT:
      return ProductMode.ASSISTANT;
    case AppView.SILENT_ASSISTANT:
      return ProductMode.SILENT_ASSISTANT;
    case AppView.VOICE_CONVERSATION:
      return ProductMode.CONVERSATION;
    default:
      return null;
  }
}

export function productModeToAppView(mode: ProductMode): AppView {
  switch (mode) {
    case ProductMode.TRANSCRIBE:
      return AppView.TRANSCRIPTION;
    case ProductMode.ASSISTANT:
      return AppView.ASSISTANT;
    case ProductMode.SILENT_ASSISTANT:
      return AppView.SILENT_ASSISTANT;
    case ProductMode.CONVERSATION:
      return AppView.VOICE_CONVERSATION;
  }
}

/** True when the Conversation product mode is active (voice duplex + existing VoiceConversation). */
export function isConversationAppView(view: AppView): boolean {
  return view === AppView.VOICE_CONVERSATION;
}

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

/** One line in Conversation Mode live transcript (voice WebSocket). */
export interface VoiceMessage {
  role: 'user' | 'assistant';
  text: string;
  /** Matches `generation_id` / `message_id` from voice WS for merging `assistant_citations`. */
  generationId?: number;
  citations?: DocumentChunk[];
}

/** Assistant Mode hand-raise card (mirrors backend `SuggestionOut`). */
export type SuggestionMode = 'ASSISTANT';

export type SuggestionCategory =
  | 'fact_check'
  | 'contradiction'
  | 'relevant_knowledge'
  | 'action_reminder'
  | 'follow_up_question'
  | 'clarification'
  | 'summary_help'
  | 'missing_context';

export type SuggestionStatus =
  | 'pending'
  | 'approved'
  | 'dismissed'
  | 'ignored'
  | 'saved'
  | 'spoken';

export type SuggestionSourceOrigin =
  | 'transcript'
  | 'rag'
  | 'rules'
  | 'rules_plus_rag'
  | 'transcript_plus_rag'
  | 'notes'
  | 'transcript_plus_notes'
  | 'notes_plus_rag'
  | 'none';

export type SuggestionEvidenceStatus = 'grounded' | 'partial' | 'weak' | 'none';

/** Citation dicts match Knowledge Chat / `mapCitations` backend keys (`filename`, `snippet`, …). */
export type SuggestionCitation = Record<string, unknown>;

/** Unified KB transcript analysis (POST /api/assistant/analyze-transcript). */
export type KbFindingLabel = 'Supported' | 'Contradicted' | 'Related' | 'Unverified' | 'Needs Review';

export interface AssistantSource {
  document_id?: string | null;
  document_name: string;
  page?: number | null;
  snippet: string;
  score: number;
}

export interface AssistantAnalysisItem {
  id: string;
  text: string;
  start_char: number;
  end_char: number;
  label: KbFindingLabel;
  confidence: number;
  evidence_status: SuggestionEvidenceStatus;
  explanation: string;
  feedback: string;
  speak_text: string;
  sources: AssistantSource[];
  persisted_id?: string | null;
}

export interface AnalyzeTranscriptResponse {
  items: AssistantAnalysisItem[];
  skipped_reason?: string | null;
}

export interface Suggestion {
  id: string;
  session_id: string;
  mode: SuggestionMode;
  title: string;
  short_text: string;
  speak_text: string;
  reason: string;
  category: SuggestionCategory;
  confidence: number;
  source_origin: SuggestionSourceOrigin;
  evidence_status: SuggestionEvidenceStatus;
  citations: SuggestionCitation[];
  created_at: string;
  status: SuggestionStatus;
  /** Transcript substring matched for KB check / highlight */
  trigger_excerpt?: string | null;
  influencing_rule_set_id?: string | null;
  influencing_rule_set_name?: string | null;
  influencing_rule_id?: string | null;
  influencing_rule_title?: string | null;
}

/** Silent Assistant correction row (mirrors backend `CorrectionFindingOut`). */
export type SilentFindingCategory =
  | 'rules_violation'
  | 'factual_inconsistency'
  | 'contradiction_with_indexed_knowledge'
  | 'unsupported_claim'
  | 'possible_misinterpretation'
  | 'useful_suggestion'
  | 'needs_verification';

export type SilentFindingStatusLabel =
  | 'likely_correct'
  | 'possibly_wrong'
  | 'unsupported'
  | 'contradicted'
  | 'needs_verification'
  | 'suggestion_available';

export type SilentFindingEvidenceStatus = 'grounded' | 'partial' | 'weak' | 'none';

export type SilentFindingUserAction =
  | 'pending'
  | 'accepted'
  | 'dismissed'
  | 'marked_unhelpful'
  | 'saved'
  | 'pinned';

export interface CorrectionFinding {
  id: string;
  session_id: string;
  transcript_segment_id?: string | null;
  turn_id?: string | null;
  original_text: string;
  highlighted_span_start: number;
  highlighted_span_end: number;
  category: SilentFindingCategory;
  status_label: SilentFindingStatusLabel;
  suggested_correction: string;
  reason: string;
  evidence_status: SilentFindingEvidenceStatus;
  confidence: number;
  source_origin: SuggestionSourceOrigin;
  citations: SuggestionCitation[];
  created_at: string;
  user_action: SilentFindingUserAction;
  influencing_rule_set_id?: string | null;
  influencing_rule_set_name?: string | null;
  influencing_rule_id?: string | null;
  influencing_rule_title?: string | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: DocumentChunk[];
  timestamp: number;
}

/** One committed paragraph/segment from live transcribe WebSocket (`segment` or `segments` on `partial`/`final`). */
export interface TranscriptSegment {
  paragraphId: string;
  text: string;
  /** When this segment was received in the browser (server WS does not send STT clock per segment). */
  receivedAt: number;
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
