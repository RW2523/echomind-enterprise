/**
 * Shell-wide state for the chat: which matter the conversation is about (the header's matter selector), and what the
 * right-hand panel shows. The panel remembers which control opened it and gives focus back on close (Esc or ×).
 */
import React, { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { useMatters } from '../api/queries';
import type { Matter } from '../api/types';
import type { ComparisonSide } from './types';

const STORE_KEY = 'echomind.counsel.matter';          // the first (or only) matter: kept for pages that know one matter
const LIST_KEY = 'echomind.counsel.matters';          // every matter of the next conversation, the first being its own
export const MAX_MATTERS = 5;

function readStored(): string[] {
  try {
    const list = JSON.parse(window.localStorage.getItem(LIST_KEY) || 'null');
    if (Array.isArray(list) && list.every((x) => typeof x === 'string')) return list.slice(0, MAX_MATTERS);
    const one = window.localStorage.getItem(STORE_KEY);
    return one ? [one] : [];
  } catch { return []; }
}
function writeStored(ids: string[]) {
  try {
    if (ids.length) { window.localStorage.setItem(STORE_KEY, ids[0]); window.localStorage.setItem(LIST_KEY, JSON.stringify(ids)); }
    else { window.localStorage.removeItem(STORE_KEY); window.localStorage.removeItem(LIST_KEY); }
  } catch { /* private window: the choice lasts for this page only */ }
}

/** A highly restricted matter is only ever discussed on its own (the server refuses it in company too). */
export function withMatter(current: Matter[], m: Matter): Matter[] {
  if (current.some((x) => x.id === m.id)) return current.filter((x) => x.id !== m.id);
  if (m.sensitivity === 'highly_restricted' || current.some((x) => x.sensitivity === 'highly_restricted')) return [m];
  return [...current, m].slice(0, MAX_MATTERS);
}

interface ChatMatter {
  matter: Matter | null;                    // the conversation's own matter (the first)
  matterId: string | null;
  matters: Matter[];                        // every matter of the conversation, in order
  matterIds: string[];
  setMatterId: (id: string | null) => void;
  setMatterIds: (ids: string[]) => void;
}
const ChatMatterContext = React.createContext<ChatMatter>({ matter: null, matterId: null, matters: [], matterIds: [], setMatterId: () => {},
                                                            setMatterIds: () => {} });

/** The chat's matters: only the person's own visible matters; a remembered id that is no longer visible is dropped. */
export function ChatMatterProvider({ children }: { children: React.ReactNode }) {
  const list = useMatters();
  const [ids, setIds] = useState<string[]>(readStored);
  const matters = useMemo(() => ids.map((id) => list.data?.find((m) => m.id === id)).filter((m): m is Matter => !!m), [ids, list.data]);
  useEffect(() => {
    if (list.data && matters.length !== ids.length) { const keep = matters.map((m) => m.id); setIds(keep); writeStored(keep); }
  }, [ids, list.data, matters]);
  const setMatterIds = useCallback((next: string[]) => {
    const unique = Array.from(new Set(next)).slice(0, MAX_MATTERS);
    setIds(unique); writeStored(unique);
  }, []);
  const setMatterId = useCallback((id: string | null) => setMatterIds(id ? [id] : []), [setMatterIds]);
  const value = useMemo(() => ({ matter: matters[0] ?? null, matterId: matters[0]?.id ?? null, matters, matterIds: matters.map((m) => m.id),
                                 setMatterId, setMatterIds }), [matters, setMatterId, setMatterIds]);
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
