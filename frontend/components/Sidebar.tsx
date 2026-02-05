
import React from 'react';
import { AppView } from '../types';
import { ICONS } from '../constants';

interface SidebarProps {
  activeView: AppView;
  setActiveView: (view: AppView) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeView, setActiveView }) => {
  const navItems = [
    { id: AppView.KNOWLEDGE_CHAT, label: 'Knowledge Chat', icon: ICONS.Chat },
    { id: AppView.TRANSCRIPTION, label: 'Live Transcript', icon: ICONS.Transcript },
    { id: AppView.VOICE_CONVERSATION, label: 'Voice Mode', icon: ICONS.Mic },
    { id: AppView.SETTINGS, label: 'Settings', icon: ICONS.Settings },
  ];

  return (
    <aside className="w-16 md:w-64 flex flex-col bg-[#080b14] border-r border-white/5 transition-all duration-300 h-full overflow-y-auto overflow-x-hidden">
      <div className="p-4 md:p-6 flex items-center gap-3">
        <div className="w-8 h-8 md:w-10 md:h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 shrink-0">
          <ICONS.Zap className="text-white w-5 h-5 md:w-6 md:h-6" />
        </div>
        <div className="hidden md:block">
          <h1 className="text-lg font-bold tracking-tight text-white leading-none">EchoMind</h1>
          <p className="text-[10px] text-cyan-400/80 uppercase tracking-widest font-semibold mt-1">by Ajace AI</p>
        </div>
      </div>

      <nav className="flex-1 px-2 md:px-3 mt-4 space-y-1 md:space-y-2">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveView(item.id)}
            className={`w-full flex items-center justify-center md:justify-start gap-4 p-3 md:px-4 md:py-3 rounded-xl transition-all duration-200 group ${
              activeView === item.id 
                ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-[0_0_20px_rgba(34,211,238,0.05)]' 
                : 'text-slate-400 hover:bg-white/5 hover:text-white border border-transparent'
            }`}
          >
            <item.icon className={`w-5 h-5 shrink-0 ${activeView === item.id ? 'text-cyan-400' : 'group-hover:text-white'}`} />
            <span className="hidden md:block font-medium text-sm truncate">{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="p-4 mt-auto hidden md:block">
        <div className="glass rounded-2xl p-4 border-white/5 bg-white/5">
          <p className="text-xs text-slate-500 mb-2">Usage</p>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div className="bg-cyan-400 h-full w-2/3 shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
          </div>
          <p className="text-[10px] text-slate-400 mt-2">1.2 GB of 2.0 GB Vector DB</p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
