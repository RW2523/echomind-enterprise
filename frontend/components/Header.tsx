
import React from 'react';
import { AppView, AppSettings } from '../types';

interface HeaderProps {
  activeView: AppView;
  settings: AppSettings;
}

const Header: React.FC<HeaderProps> = ({ activeView, settings }) => {
  const getTitle = () => {
    switch (activeView) {
      case AppView.KNOWLEDGE_CHAT: return 'Knowledge Chat';
      case AppView.TRANSCRIPTION: return 'Live Transcription & Polishing';
      case AppView.VOICE_CONVERSATION: return 'Voice AI Conversation';
      case AppView.SETTINGS: return 'Platform Settings';
      default: return 'EchoMind';
    }
  };

  return (
    <header className="h-14 sm:h-16 shrink-0 border-b border-white/5 px-4 sm:px-5 md:px-6 lg:px-8 flex items-center justify-between bg-[#05070a]/50 backdrop-blur-md">
      <div>
        <h2 className="text-lg font-semibold text-white">{getTitle()}</h2>
      </div>

      <div className="flex items-center gap-3">
        <p className="text-xs font-semibold text-white">Ajace Admin</p>
        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 border border-white/10 flex items-center justify-center text-xs font-bold shadow-lg">
          AA
        </div>
      </div>
    </header>
  );
};

export default Header;
