
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
  raw: string;
  polished?: string;
  tags?: string[];
  timestamp: number;
  metadata: {
    date: string;
    time: string;
    topic?: string;
  };
}

export interface AppSettings {
  voiceName: 'Zephyr' | 'Puck' | 'Charon' | 'Kore' | 'Fenrir';
  contextWindow: '24h' | '48h' | '1w' | 'all';
  persona: PersonaType;
  model: string;
  developerMode: boolean;
}
