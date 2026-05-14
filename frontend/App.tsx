import React, { useState, useCallback, useEffect } from 'react';
import { AppView, AppSettings, PersonaType } from './types';
import { ICONS, COLORS } from './constants';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import KnowledgeChat from './components/KnowledgeChat';
import LiveTranscription from './components/LiveTranscription';
import SilentAssistant from './components/SilentAssistant';
import PersonalAssistant from './components/PersonalAssistant';
import BoardRoom from './components/BoardRoom';
import VoiceConversation from './components/VoiceConversation';
import Settings from './components/Settings';
import { useVoiceConnection } from './hooks/useVoiceConnection';
import { useLiveTranscription } from './hooks/useLiveTranscription';
import { useKnowledgeChat } from './hooks/useKnowledgeChat';
import { defaultTranscriptName } from './services/backend';

const SETTINGS_KEY = "echomind_settings";

const defaultSettings: AppSettings = {
  voiceName: 'en_US-lessac-medium',
  contextWindow: 'all',
  persona: PersonaType.FINANCIAL,
  model: '-3-pro-preview',
  developerMode: false,
  advancedRag: false,
  voiceUseKnowledgeBase: false,
  voiceBotName: '',
  voiceUserName: '',
  voiceContext: '',
};

function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<AppSettings>;
      return {
        voiceName: parsed.voiceName ?? defaultSettings.voiceName,
        contextWindow: parsed.contextWindow ?? defaultSettings.contextWindow,
        persona: parsed.persona ?? defaultSettings.persona,
        model: parsed.model ?? defaultSettings.model,
        developerMode: parsed.developerMode ?? defaultSettings.developerMode,
        advancedRag: parsed.advancedRag ?? defaultSettings.advancedRag,
        voiceUseKnowledgeBase: parsed.voiceUseKnowledgeBase ?? defaultSettings.voiceUseKnowledgeBase,
        voiceBotName: parsed.voiceBotName ?? defaultSettings.voiceBotName,
        voiceUserName: parsed.voiceUserName ?? defaultSettings.voiceUserName,
        voiceContext: parsed.voiceContext ?? defaultSettings.voiceContext,
      };
    }
  } catch (_) {}
  return defaultSettings;
}

function saveSettings(s: AppSettings) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch (_) {}
}

const App: React.FC = () => {
  const [activeView, setActiveView] = useState<AppView>(AppView.KNOWLEDGE_CHAT);
  const [settings, setSettingsState] = useState<AppSettings>(() => loadSettings());
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const setSettings = useCallback((s: AppSettings) => {
    setSettingsState(s);
    saveSettings(s);
  }, []);

  // Voice connection disconnects when user switches browser tabs or navigates away from Voice.
  const voiceConnection = useVoiceConnection({ settings });

  // Live Transcription lives in App so it keeps running when switching tabs (Chat / Voice).
  // It only stops when the user clicks Stop.
  const liveTranscription = useLiveTranscription(defaultTranscriptName);

  /** Separate hook from Live Transcript so both modes do not share one WebSocket. */
  const silentAssistantTranscription = useLiveTranscription(defaultTranscriptName);

  const personalAssistantTranscription = useLiveTranscription(defaultTranscriptName);

  const boardRoomTranscription = useLiveTranscription(defaultTranscriptName, { sttProfile: "board_room" });

  // Knowledge Chat lives in App so conversation persists when switching tabs.
  const knowledgeChat = useKnowledgeChat();

  // Disconnect when user navigates away from Voice AI tab within the app
  useEffect(() => {
    if (activeView !== AppView.VOICE_CONVERSATION && voiceConnection.state.isConnected) {
      voiceConnection.disconnect();
    }
  }, [activeView, voiceConnection]);

  const renderView = () => {
    switch (activeView) {
      case AppView.KNOWLEDGE_CHAT:
        return <KnowledgeChat settings={settings} knowledgeChat={knowledgeChat} />;
      case AppView.TRANSCRIPTION:
        return <LiveTranscription liveTranscription={liveTranscription} />;
      case AppView.SILENT_ASSISTANT:
        return <SilentAssistant liveTranscription={silentAssistantTranscription} />;
      case AppView.PERSONAL_ASSISTANT:
        return (
          <PersonalAssistant
            settings={settings}
            liveTranscription={personalAssistantTranscription}
            onOpenKnowledgeChatWithDraft={(draft) => {
              setActiveView(AppView.KNOWLEDGE_CHAT);
              window.setTimeout(() => {
                window.dispatchEvent(new CustomEvent('echomind-chat-followup', { detail: { draft } }));
              }, 0);
            }}
          />
        );
      case AppView.BOARD_ROOM:
        return <BoardRoom liveTranscription={boardRoomTranscription} />;
      case AppView.VOICE_CONVERSATION:
        return (
          <VoiceConversation
            settings={settings}
            onUpdateSetting={(key, val) => setSettings({ ...settings, [key]: val })}
            voiceConnection={voiceConnection}
          />
        );
      case AppView.SETTINGS:
        return <Settings settings={settings} setSettings={setSettings} />;
      default:
        return <KnowledgeChat settings={settings} knowledgeChat={knowledgeChat} />;
    }
  };

  return (
    <div className="flex h-full w-full bg-[#05070a] text-slate-200 overflow-hidden" style={{ height: '100dvh', paddingTop: 'env(safe-area-inset-top)', paddingBottom: 'env(safe-area-inset-bottom)', paddingLeft: 'env(safe-area-inset-left)', paddingRight: 'env(safe-area-inset-right)' }}>
      {/* Dynamic Background Glows */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-cyan-500/10 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2 pointer-events-none z-0" aria-hidden />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-violet-600/10 blur-[100px] rounded-full translate-y-1/2 -translate-x-1/2 pointer-events-none z-0" aria-hidden />

      <Sidebar activeView={activeView} setActiveView={setActiveView} sidebarOpen={sidebarOpen} onCloseSidebar={() => setSidebarOpen(false)} knowledgeChat={knowledgeChat} />
      
      <main className="flex-1 flex flex-col relative z-10 border-l border-white/5 min-w-0 min-h-0 overflow-hidden">
        <Header activeView={activeView} settings={settings} onMenuClick={() => setSidebarOpen(true)} />
        <div className="flex-1 min-h-0 overflow-auto flex flex-col overscroll-contain">
          <div className="flex-1 min-h-0 px-3 py-3 sm:px-5 sm:py-5 md:px-6 md:py-5 lg:px-8 lg:py-6 flex flex-col min-w-0">
            {renderView()}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
