
export enum AppView {
  KNOWLEDGE_CHAT = 'knowledge_chat',
  TRANSCRIPTION = 'transcription',
  VOICE_CONVERSATION = 'voice_conversation',
  SETTINGS = 'settings'
}

export enum PersonaType {
  GENERAL = 'General Assistant',
  LAWYER = 'Corporate Lawyer',
  ACCOUNTANT = 'Financial Accountant',
  TECH_EXPERT = 'Technical Architect'
}

export interface DocumentChunk {
  id: string;
  docName: string;
  content: string;
  metadata: {
    pageNumber?: number;
    section?: string;
    timestamp: number;
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
  polished?: string;
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
