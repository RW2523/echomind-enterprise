
export enum AppView {
  KNOWLEDGE_CHAT = 'knowledge_chat',
  TRANSCRIPTION = 'transcription',
  VOICE_CONVERSATION = 'voice_conversation',
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
  | 'Risky Statement';

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
