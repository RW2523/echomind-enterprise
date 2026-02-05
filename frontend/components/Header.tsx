
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
    <header className="h-16 border-b border-white/5 px-8 flex items-center justify-between bg-[#05070a]/50 backdrop-blur-md">
      <div>
        <h2 className="text-lg font-semibold text-white">{getTitle()}</h2>
      </div>

      <div className="flex items-center gap-6">
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
          <span className="text-xs font-medium text-slate-300">Enterprise Engine: {settings.model}</span>
        </div>

        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-xs font-semibold text-white">Ajace Admin</p>
            <p className="text-[10px] text-cyan-400 uppercase">Pro Tier</p>
          </div>
          <div className="w-9 h-9 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 border border-white/10 flex items-center justify-center text-xs font-bold shadow-lg">
            AA
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
