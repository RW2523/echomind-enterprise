import React, { useState } from 'react';
import { AppView, AppSettings } from '../types';
import { ICONS } from '../constants';
import { deleteAllData } from '../services/backend';

interface HeaderProps {
  activeView: AppView;
  settings: AppSettings;
  onMenuClick?: () => void;
}

const Header: React.FC<HeaderProps> = ({ activeView, onMenuClick }) => {
  const [clearing, setClearing] = useState(false);

  const getTitle = () => {
    switch (activeView) {
      case AppView.KNOWLEDGE_CHAT: return 'Knowledge Chat';
      case AppView.TRANSCRIPTION: return 'Live Transcription & Refinement';
      case AppView.SILENT_ASSISTANT: return 'Silent Assistant';
      case AppView.PERSONAL_ASSISTANT: return 'Personal Assistant';
      case AppView.BOARD_ROOM: return 'Board Room';
      case AppView.VOICE_CONVERSATION: return 'Voice AI Conversation';
      case AppView.SETTINGS: return 'Platform Settings';
      default: return 'EchoMind';
    }
  };

  const handleClearData = async () => {
    if (!window.confirm('Clear ALL data (documents, chunks, transcripts, chats)? This cannot be undone.')) return;
    setClearing(true);
    try {
      await deleteAllData();
      window.dispatchEvent(new CustomEvent('echomind-data-cleared'));
    } catch (e) {
      alert((e as Error)?.message || 'Failed to clear data');
    } finally {
      setClearing(false);
    }
  };

  return (
    <header className="h-14 sm:h-16 shrink-0 border-b border-white/5 px-3 sm:px-5 md:px-6 lg:px-8 flex items-center justify-between gap-2 bg-[#05070a]/50 backdrop-blur-md min-h-[52px]">
      <div className="flex items-center gap-2 min-w-0 flex-1">
        {onMenuClick && (
          <button
            type="button"
            onClick={onMenuClick}
            className="md:hidden shrink-0 p-2 -ml-1 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 touch-manipulation min-h-[44px] min-w-[44px] flex items-center justify-center"
            aria-label="Open menu"
          >
            <ICONS.Menu className="w-6 h-6" />
          </button>
        )}
        <h2 className="text-base sm:text-lg font-semibold text-white truncate">{getTitle()}</h2>
      </div>

      {/* <div className="flex items-center gap-2 shrink-0">
        <button
          type="button"
          onClick={handleClearData}
          disabled={clearing}
          className="px-3 py-2.5 sm:py-1.5 rounded-lg text-sm font-medium bg-white/10 text-slate-300 hover:bg-red-500/20 hover:text-red-400 border border-white/10 disabled:opacity-50 transition-colors touch-manipulation min-h-[44px] md:min-h-0"
        >
          {clearing ? 'Clearing…' : 'Clear Data'}
        </button>
      </div> */}
    </header>
  );
};

export default Header;
