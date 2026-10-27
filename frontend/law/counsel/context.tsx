/**
 * Shell-wide state for the chat: which matter the conversation is about (the header's matter selector), and what the
 * right-hand panel shows. The panel remembers which control opened it and gives focus back on close (Esc or ×).
 */
import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useMatters } from '../api/queries';
import type { Matter } from '../api/types';
import type { ComparisonSide } from './types';

const STORE_KEY = 'echomind.counsel.matter';

function readStored(): string | null {
  try { return window.localStorage.getItem(STORE_KEY); } catch { return null; }
}
function writeStored(id: string | null) {
  try {
    if (id) window.localStorage.setItem(STORE_KEY, id);
    else window.localStorage.removeItem(STORE_KEY);
  } catch { /* private window: the choice lasts for this page only */ }
}

interface ChatMatter { matter: Matter | null; matterId: string | null; setMatterId: (id: string | null) => void }
const ChatMatterContext = React.createContext<ChatMatter>({ matter: null, matterId: null, setMatterId: () => {} });

/** The chat's matter: only one of the person's own visible matters; a remembered id that is no longer visible is dropped. */
export function ChatMatterProvider({ children }: { children: React.ReactNode }) {
  const matters = useMatters();
  const [matterId, setId] = useState<string | null>(readStored);
  const matter = useMemo(() => (matterId && matters.data?.find((m) => m.id === matterId)) || null, [matterId, matters.data]);
  useEffect(() => {
    if (matterId && matters.data && !matter) { setId(null); writeStored(null); }
  }, [matterId, matters.data, matter]);
  const setMatterId = useCallback((id: string | null) => { setId(id); writeStored(id); }, []);
  const value = useMemo(() => ({ matter, matterId: matter ? matter.id : null, setMatterId }), [matter, setMatterId]);
  return <ChatMatterContext.Provider value={value}>{children}</ChatMatterContext.Provider>;
}

export const useChatMatter = () => useContext(ChatMatterContext);

export type PanelContent =
  | { kind: 'citation'; id: string; onUse?: () => void }
  | { kind: 'side'; side: ComparisonSide; label: string };

interface Panel { content: PanelContent | null; open: (c: PanelContent, opener?: HTMLElement | null) => void; close: () => void }
const PanelContext = React.createContext<Panel>({ content: null, open: () => {}, close: () => {} });

export function PanelProvider({ children }: { children: React.ReactNode }) {
  const [content, setContent] = useState<PanelContent | null>(null);
  const opener = useRef<HTMLElement | null>(null);
  const open = useCallback((c: PanelContent, from?: HTMLElement | null) => {
    opener.current = from ?? (document.activeElement as HTMLElement | null);
    setContent(c);
  }, []);
  const close = useCallback(() => {
    setContent(null);
    const back = opener.current;
    opener.current = null;
    if (back && document.contains(back)) window.setTimeout(() => back.focus(), 0);
  }, []);
  useEffect(() => {
    if (!content) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [content, close]);
  const value = useMemo(() => ({ content, open, close }), [content, open, close]);
  return <PanelContext.Provider value={value}>{children}</PanelContext.Provider>;
}

export const usePanel = () => useContext(PanelContext);
