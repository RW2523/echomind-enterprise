import React, { useState, useEffect, useCallback } from 'react';
import { AppView } from '../types';
import { ICONS } from '../constants';
import { getStorageUsage, StorageUsage } from '../services/backend';
import type { UseKnowledgeChatReturn } from '../hooks/useKnowledgeChat';

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

function formatChatDate(created_at: string): string {
  if (!created_at) return '';
  const d = new Date(created_at);
  const now = new Date();
  const sameDay = d.getDate() === now.getDate() && d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
  if (sameDay) return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  const isYesterday = d.getDate() === yesterday.getDate() && d.getMonth() === yesterday.getMonth() && d.getFullYear() === yesterday.getFullYear();
  if (isYesterday) return 'Yesterday';
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

interface SidebarProps {
  activeView: AppView;
  setActiveView: (view: AppView) => void;
  sidebarOpen?: boolean;
  onCloseSidebar?: () => void;
  knowledgeChat?: UseKnowledgeChatReturn | null;
}

const Sidebar: React.FC<SidebarProps> = ({ activeView, setActiveView, sidebarOpen = false, onCloseSidebar, knowledgeChat }) => {
  const handleNav = (view: AppView) => {
    setActiveView(view);
    onCloseSidebar?.();
  };
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

  useEffect(() => {
    const onCleared = () => { refreshUsage(); };
    window.addEventListener('echomind-data-cleared', onCleared);
    return () => window.removeEventListener('echomind-data-cleared', onCleared);
  }, [refreshUsage]);

  const usageBytes = storage.usage_bytes;
  const capacityBytes = storage.capacity_bytes ?? 0;
  const ratio = capacityBytes > 0 ? Math.min(1, usageBytes / capacityBytes) : 0;
  const usageStr = formatBytes(usageBytes);
  const capacityStr = capacityBytes > 0 ? formatBytes(capacityBytes) : null;
  const navItems = [
    { id: AppView.KNOWLEDGE_CHAT, label: 'Knowledge Chat', icon: ICONS.Chat },
    { id: AppView.TRANSCRIPTION, label: 'Live Transcript', icon: ICONS.Transcript },
    { id: AppView.VOICE_CONVERSATION, label: 'Voice Conversation', icon: ICONS.Mic },
  ];

  const sidebarContent = (
    <>
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
        <div className="flex-1 md:hidden" />
        {onCloseSidebar && (
          <button type="button" onClick={onCloseSidebar} className="md:hidden p-2 -m-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 touch-manipulation" aria-label="Close menu">
            <ICONS.Close className="w-6 h-6" />
          </button>
        )}
      </div>

      <nav className="px-2 md:px-3 pt-2 md:pt-4 space-y-1 shrink-0">
        {navItems.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => handleNav(item.id)}
            className={`w-full flex items-center justify-center md:justify-start gap-3 px-2 py-3 md:px-3 md:py-3 rounded-xl transition-all duration-200 group min-h-[44px] touch-manipulation ${
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

      {activeView === AppView.KNOWLEDGE_CHAT && knowledgeChat && (
        <div className="flex-1 min-h-0 flex flex-col px-2 md:px-3 pt-3 border-t border-white/5 overflow-hidden">
          <div className="flex items-center justify-between gap-2 shrink-0 mb-2">
            <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">Chat history</span>
            <button
              type="button"
              onClick={() => knowledgeChat.newChat()}
              className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg text-xs font-medium text-cyan-400 hover:bg-cyan-500/10 border border-cyan-500/20 transition-colors touch-manipulation"
            >
              <span className="w-3.5 h-3.5 flex items-center justify-center">+</span>
              <span className="hidden md:inline">New chat</span>
            </button>
          </div>
          <ul className="flex-1 min-h-0 overflow-y-auto space-y-0.5">
            {knowledgeChat.chats.length === 0 && (
              <li className="text-xs text-slate-500 py-2 px-2">No chats yet</li>
            )}
            {knowledgeChat.chats.map((chat) => (
              <li key={chat.id} className="group flex items-center gap-1 rounded-lg hover:bg-white/5 min-h-[36px]">
                <button
                  type="button"
                  onClick={() => knowledgeChat.selectChat(chat.id)}
                  className={`flex-1 min-w-0 text-left px-2 py-2 rounded-lg text-sm truncate touch-manipulation ${
                    knowledgeChat.chatId === chat.id
                      ? 'bg-cyan-500/15 text-cyan-300'
                      : 'text-slate-300 hover:text-white'
                  }`}
                  title={chat.title}
                >
                  <span className="block truncate">{chat.title || 'New chat'}</span>
                  <span className="block text-[10px] text-slate-500 mt-0.5">{formatChatDate(chat.created_at)}</span>
                </button>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); if (window.confirm('Delete this chat? This cannot be undone.')) knowledgeChat.deleteChat(chat.id); }}
                  className="shrink-0 p-1.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 opacity-0 group-hover:opacity-100 transition-opacity touch-manipulation"
                  aria-label="Delete chat"
                  title="Delete chat"
                >
                  <ICONS.Trash className="w-3.5 h-3.5" />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="px-3 py-4 md:px-4 md:py-4 mt-auto hidden md:block border-t border-white/5 shrink-0">
        <div className="w-full glass rounded-2xl p-4 border border-white/5 bg-white/5 text-left">
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
    </>
  );

  return (
    <>
      {/* Mobile overlay backdrop */}
      {onCloseSidebar && (
        <div
          className={`fixed inset-0 z-20 bg-black/60 backdrop-blur-sm md:hidden transition-opacity duration-200 ${sidebarOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}
          onClick={onCloseSidebar}
          onKeyDown={(e) => e.key === 'Escape' && onCloseSidebar()}
          aria-hidden
        />
      )}
      {/* Sidebar: drawer on mobile, inline on desktop */}
      <aside
        className={`
          flex flex-col bg-[#080b14] border-r border-white/5 transition-all duration-300 h-full overflow-y-auto overflow-x-hidden shrink-0
          w-[min(280px,85vw)] md:w-60 lg:w-64
          fixed md:relative inset-y-0 left-0 z-30 md:z-auto
          transform md:transform-none
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
          shadow-xl md:shadow-none
        `}
        role="navigation"
        aria-label="Main navigation"
      >
        <div className="flex-1 flex flex-col min-h-0 md:min-h-full">
          {sidebarContent}
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
