import React, { useState, useEffect, useCallback, useRef } from 'react';
import { AppView } from '../types';
import { ICONS } from '../constants';
import { getStorageUsage, StorageUsage, getDataPreview, deleteAllData, DataPreview } from '../services/backend';

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
  const [usageOpen, setUsageOpen] = useState(false);
  const [dataPreview, setDataPreview] = useState<DataPreview | null>(null);
  const [deletingAll, setDeletingAll] = useState(false);
  const usageRef = useRef<HTMLDivElement>(null);

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
    if (!usageOpen) return;
    let cancelled = false;
    getDataPreview().then((d) => { if (!cancelled) setDataPreview(d); }).catch(() => { if (!cancelled) setDataPreview(null); });
    return () => { cancelled = true; };
  }, [usageOpen]);

  useEffect(() => {
    if (!usageOpen) return;
    const close = (e: MouseEvent) => {
      if (usageRef.current && !usageRef.current.contains(e.target as Node)) setUsageOpen(false);
    };
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, [usageOpen]);

  const handleDeleteAll = useCallback(async () => {
    if (!window.confirm('Delete ALL data (documents, chunks, transcripts, chats)? This cannot be undone.')) return;
    setDeletingAll(true);
    try {
      await deleteAllData();
      setUsageOpen(false);
      await refreshUsage();
      setDataPreview(null);
    } catch (e) {
      alert((e as Error)?.message || 'Failed to delete all data');
    } finally {
      setDeletingAll(false);
    }
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

      <div className="px-3 py-4 md:px-4 md:py-4 mt-auto hidden md:block border-t border-white/5 relative" ref={usageRef}>
        <button
          type="button"
          onClick={() => setUsageOpen((o) => !o)}
          className="w-full glass rounded-2xl p-4 border border-white/5 bg-white/5 text-left hover:bg-white/10 transition-colors"
        >
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
        </button>
        {usageOpen && (
          <div className="absolute bottom-full left-2 right-2 mb-2 z-50 rounded-xl border border-white/20 bg-slate-900/98 shadow-2xl overflow-hidden max-h-[70vh] flex flex-col">
            <div className="p-3 border-b border-white/10 flex items-center justify-between shrink-0">
              <span className="text-sm font-semibold text-white">Data preview</span>
              <button type="button" onClick={() => setUsageOpen(false)} className="text-slate-400 hover:text-white p-1">✕</button>
            </div>
            <div className="overflow-auto p-3 space-y-4 text-xs">
              {dataPreview == null ? (
                <p className="text-slate-500">Loading…</p>
              ) : (
                <>
                  <div>
                    <p className="text-slate-400 font-medium mb-1">Documents ({dataPreview.documents.length})</p>
                    <div className="rounded-lg border border-white/10 overflow-hidden">
                      <table className="w-full text-left">
                        <thead><tr className="bg-white/5"><th className="px-2 py-1.5">id</th><th className="px-2 py-1.5">filename</th><th className="px-2 py-1.5">created_at</th></tr></thead>
                        <tbody>
                          {dataPreview.documents.slice(0, 50).map((d) => (
                            <tr key={d.id} className="border-t border-white/5"><td className="px-2 py-1 truncate max-w-[80px]">{d.id}</td><td className="px-2 py-1 truncate max-w-[120px]" title={d.filename}>{d.filename}</td><td className="px-2 py-1">{d.created_at?.slice(0, 19)}</td></tr>
                          ))}
                          {dataPreview.documents.length > 50 && <tr className="border-t border-white/5"><td colSpan={3} className="px-2 py-1 text-slate-500">+ {dataPreview.documents.length - 50} more</td></tr>}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <div>
                    <p className="text-slate-400 font-medium mb-1">Chunks ({dataPreview.chunks.length})</p>
                    <div className="rounded-lg border border-white/10 overflow-hidden">
                      <table className="w-full text-left">
                        <thead><tr className="bg-white/5"><th className="px-2 py-1.5">id</th><th className="px-2 py-1.5">doc_id</th><th className="px-2 py-1.5">preview</th></tr></thead>
                        <tbody>
                          {dataPreview.chunks.slice(0, 30).map((c) => (
                            <tr key={c.id} className="border-t border-white/5"><td className="px-2 py-1 truncate max-w-[70px]">{c.id}</td><td className="px-2 py-1 truncate max-w-[80px]">{c.doc_id}</td><td className="px-2 py-1 truncate max-w-[180px]" title={c.text_preview}>{c.text_preview}</td></tr>
                          ))}
                          {dataPreview.chunks.length > 30 && <tr className="border-t border-white/5"><td colSpan={3} className="px-2 py-1 text-slate-500">+ {dataPreview.chunks.length - 30} more</td></tr>}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <div>
                    <p className="text-slate-400 font-medium mb-1">Transcripts ({dataPreview.transcripts.length})</p>
                    <div className="rounded-lg border border-white/10 overflow-hidden">
                      <table className="w-full text-left">
                        <thead><tr className="bg-white/5"><th className="px-2 py-1.5">id</th><th className="px-2 py-1.5">title</th><th className="px-2 py-1.5">tags</th><th className="px-2 py-1.5">created_at</th></tr></thead>
                        <tbody>
                          {dataPreview.transcripts.slice(0, 30).map((t) => (
                            <tr key={t.id} className="border-t border-white/5"><td className="px-2 py-1 truncate max-w-[70px]">{t.id}</td><td className="px-2 py-1 truncate max-w-[100px]" title={t.title}>{t.title}</td><td className="px-2 py-1 truncate max-w-[120px]">{(t.tags || []).join(', ')}</td><td className="px-2 py-1">{t.created_at?.slice(0, 19)}</td></tr>
                          ))}
                          {dataPreview.transcripts.length > 30 && <tr className="border-t border-white/5"><td colSpan={4} className="px-2 py-1 text-slate-500">+ {dataPreview.transcripts.length - 30} more</td></tr>}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </div>
            <div className="p-3 border-t border-white/10 shrink-0 flex justify-end gap-2">
              <button type="button" onClick={() => setUsageOpen(false)} className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 text-sm">Close</button>
              <button type="button" onClick={handleDeleteAll} disabled={deletingAll} className="px-3 py-1.5 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 text-sm disabled:opacity-50">Delete all data</button>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;
