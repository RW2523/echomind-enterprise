import React, { useState, useEffect, useCallback } from 'react';
import { AppView } from '../types';
import { ICONS } from '../constants';
import { getStorageUsage, StorageUsage } from '../services/backend';

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

interface SidebarProps {
  activeView: AppView;
  setActiveView: (view: AppView) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ activeView, setActiveView }) => {
  const [storage, setStorage] = useState<StorageUsage>({ usage_bytes: 0, capacity_bytes: null });

  const refreshUsage = useCallback(async () => {
    try {
      const u = await getStorageUsage();
      setStorage(u);
    } catch {
      setStorage({ usage_bytes: 0, capacity_bytes: null });
    }
  }, []);

  useEffect(() => {
    refreshUsage();
    const interval = setInterval(refreshUsage, 30000);
    return () => clearInterval(interval);
  }, [refreshUsage]);

  const usageBytes = storage.usage_bytes;
  const capacityBytes = storage.capacity_bytes ?? 0;
  const ratio = capacityBytes > 0 ? Math.min(1, usageBytes / capacityBytes) : 0;
  const usageStr = formatBytes(usageBytes);
  const capacityStr = capacityBytes > 0 ? formatBytes(capacityBytes) : null;

  const navItems = [
    { id: AppView.KNOWLEDGE_CHAT, label: 'Knowledge Chat', icon: ICONS.Chat },
    { id: AppView.TRANSCRIPTION, label: 'Live Transcript', icon: ICONS.Transcript },
    { id: AppView.VOICE_CONVERSATION, label: 'Conversation', icon: ICONS.Mic },
    { id: AppView.SETTINGS, label: 'Settings', icon: ICONS.Settings },
  ];

  return (
    <aside className="w-16 md:w-60 lg:w-64 flex flex-col bg-[#080b14] border-r border-white/5 transition-all duration-300 h-full overflow-y-auto overflow-x-hidden shrink-0">
      <div className="px-3 py-4 md:px-4 md:py-5 flex items-center gap-3">
        <img
          src="https://www.ajace.com/wp-content/uploads/2016/12/cropped-logo-32x32.png"
          alt="Ajace"
          className="w-8 h-8 md:w-9 md:h-9 rounded-xl object-contain shrink-0"
        />
        <div className="hidden md:block min-w-0">
          <h1 className="text-base font-bold tracking-tight text-white leading-none truncate">EchoMind</h1>
          <p className="text-[10px] text-cyan-400/80 uppercase tracking-widest font-semibold mt-0.5">by Ajace AI</p>
        </div>
      </div>

      <nav className="flex-1 px-2 md:px-3 pt-2 md:pt-4 space-y-1">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveView(item.id)}
            className={`w-full flex items-center justify-center md:justify-start gap-3 px-2 py-2.5 md:px-3 md:py-3 rounded-xl transition-all duration-200 group ${
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

      <div className="px-3 py-4 md:px-4 md:py-4 mt-auto hidden md:block border-t border-white/5">
        <div className="glass rounded-2xl p-4 border-white/5 bg-white/5">
          <p className="text-xs text-slate-500 mb-2">Usage</p>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
            <div
              className="bg-cyan-400 h-full shadow-[0_0_10px_rgba(34,211,238,0.5)] transition-all duration-500"
              style={{ width: `${ratio * 100}%` }}
            />
          </div>
          <p className="text-[10px] text-slate-400 mt-2">
            {capacityStr ? `${usageStr} of ${capacityStr} Vector DB` : `${usageStr} Vector DB`}
          </p>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
