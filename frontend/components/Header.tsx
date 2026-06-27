import React from 'react';
import { AppView, AppSettings } from '../types';
import { ICONS } from '../constants';
import type { VerticalPack } from '../packs';
import type { AuthUser } from '../services/backend';

interface HeaderProps {
  activeView: AppView;
  settings?: AppSettings;  // retained for caller compatibility; not used here
  pack?: VerticalPack | null;
  user?: AuthUser | null;
  onLogout?: () => void;
  onMenuClick?: () => void;
}

const Header: React.FC<HeaderProps> = ({ activeView, pack, user, onLogout, onMenuClick }) => {
  // NOTE: the Clear-Data button (JSX commented out below) and its handler/state/import were removed
  // as dead code. Re-add deleteAllData + handler if the button is restored. (L37)
  const getTitle = () => {
    switch (activeView) {
      case AppView.KNOWLEDGE_CHAT: return 'Knowledge Chat';
      case AppView.TRANSCRIPTION: return 'Live Transcription & Refinement';
      case AppView.VOICE_CONVERSATION: return 'Voice AI Conversation';
      case AppView.DOCUMENT_STUDIO: return 'Document Studio';
      case AppView.SETTINGS: return 'Platform Settings';
      default: return 'EchoMind';
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

      {pack && (
        <div className="flex items-center gap-2 shrink-0">
          <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: pack.accent }} aria-hidden />
          <div className="text-right leading-tight hidden sm:block">
            <div className="text-sm font-semibold text-white">{pack.name}</div>
            <div className="text-[11px] text-slate-400">{pack.tagline}</div>
          </div>
        </div>
      )}

      {user && (
        <div className="flex items-center gap-2 shrink-0 ml-2">
          <span className="hidden sm:inline text-xs text-slate-400">{user.username}</span>
          <button
            type="button"
            onClick={onLogout}
            className="px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-300 bg-white/5 border border-white/10 hover:bg-white/10 hover:text-white transition-colors touch-manipulation"
          >
            Sign out
          </button>
        </div>
      )}

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
