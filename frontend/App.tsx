
import React, { useState, useEffect } from 'react';
import { AppView, AppSettings, PersonaType } from './types';
import { ICONS, COLORS } from './constants';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import KnowledgeChat from './components/KnowledgeChat';
import LiveTranscription from './components/LiveTranscription';
import VoiceConversation from './components/VoiceConversation';
import Settings from './components/Settings';

const App: React.FC = () => {
  const [activeView, setActiveView] = useState<AppView>(AppView.KNOWLEDGE_CHAT);
  const [settings, setSettings] = useState<AppSettings>({
    voiceName: 'Zephyr',
    contextWindow: 'all',
    persona: PersonaType.GENERAL,
    model: '-3-pro-preview',
    developerMode: false
  });

  const renderView = () => {
    switch (activeView) {
      case AppView.KNOWLEDGE_CHAT:
        return <KnowledgeChat />;
      case AppView.TRANSCRIPTION:
        return <LiveTranscription />;
      case AppView.VOICE_CONVERSATION:
        return <VoiceConversation settings={settings} />;
      case AppView.SETTINGS:
        return <Settings settings={settings} setSettings={setSettings} />;
      default:
        return <KnowledgeChat />;
    }
  };

  return (
    <div className="flex h-full w-full bg-[#05070a] text-slate-200 overflow-hidden" style={{ height: '100dvh' }}>
      {/* Dynamic Background Glows */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-cyan-500/10 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2 pointer-events-none z-0"></div>
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-violet-600/10 blur-[100px] rounded-full translate-y-1/2 -translate-x-1/2 pointer-events-none z-0"></div>

      <Sidebar activeView={activeView} setActiveView={setActiveView} />
      
      <main className="flex-1 flex flex-col relative z-10 border-l border-white/5 min-w-0 min-h-0">
        <Header activeView={activeView} settings={settings} />
        <div className="flex-1 min-h-0 overflow-auto flex flex-col">
          <div className="flex-1 min-h-0 px-4 py-4 sm:px-5 sm:py-5 md:px-6 md:py-5 lg:px-8 lg:py-6 flex flex-col">
            {renderView()}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
