import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router';
import { Briefcase, CornerDownLeft, Search } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useMatters } from '../api/queries';
import { useSession } from '../app/session';
import { useCurrentMatter } from '../app/matter';
import { firmEntries, matterEntries } from '../app/nav';

/**
 * The command palette (D-065): ⌘K / Ctrl+K anywhere, or the search button in the sidebar. Type to jump to a matter,
 * one of its tools, or a firm page; Home's quick actions open it asking which matter to work on.
 */
export type PaletteAction = 'research' | 'conversation' | 'meetings' | 'drafting';
export interface PaletteIntent { action?: PaletteAction; prompt?: string }

const Ctx = React.createContext<(intent?: PaletteIntent) => void>(() => undefined);
export const useOpenPalette = () => React.useContext(Ctx);
export const isMac = () => typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);

export function PaletteProvider({ children }: { children: React.ReactNode }) {
  const [intent, setIntent] = useState<PaletteIntent | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.altKey && e.key.toLowerCase() === 'k') { e.preventDefault(); setIntent((i) => (i ? null : {})); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return (
    <Ctx.Provider value={(i) => setIntent(i ?? {})}>
      {children}
      {intent && <Palette intent={intent} onClose={() => setIntent(null)} />}
    </Ctx.Provider>
  );
}

interface Item { id: string; group: string; label: string; hint?: string; icon: LucideIcon; to: string }

function Palette({ intent, onClose }: { intent: PaletteIntent; onClose: () => void }) {
  const me = useSession();
  const matters = useMatters();
  const current = useCurrentMatter();
  const navigate = useNavigate();
  const [q, setQ] = useState('');
  const [sel, setSel] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLUListElement>(null);
  useEffect(() => { input.current?.focus(); }, []);

  const items = useMemo<Item[]>(() => {
    const open = (matters.data ?? []).filter((m) => m.status === 'open');
    const out: Item[] = [];
    if (intent.action) {
      for (const m of open) {
        const entry = matterEntries(me, m).find((e) => e.key === intent.action);
        if (entry) out.push({ id: `${m.number}:${entry.key}`, group: 'Choose a matter', label: `${m.number} · ${m.title}`, hint: m.client.name, icon: Briefcase, to: entry.to });
      }
      return out;
    }
    if (current) {
      for (const e of matterEntries(me, current)) out.push({ id: `cur:${e.key}`, group: `In ${current.number}`, label: e.label, hint: e.about, icon: e.icon, to: e.to });
    }
    for (const m of open) out.push({ id: `m:${m.number}`, group: 'Matters', label: `${m.number} · ${m.title}`, hint: m.client.name, icon: Briefcase, to: `/m/${encodeURIComponent(m.number)}/overview` });
    for (const e of firmEntries(me)) out.push({ id: `f:${e.key}`, group: 'Go to', label: e.label, icon: e.icon, to: e.to });
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (words.length) {                                    // a matter named in the search: its tools too
      for (const m of open) {
        if (current && m.number === current.number) continue;
        const name = `${m.number} ${m.title} ${m.client.name}`.toLowerCase();
        if (words.some((w) => name.includes(w))) {
          for (const e of matterEntries(me, m)) {
            if (e.key !== 'overview') out.push({ id: `${m.number}:${e.key}`, group: m.number, label: `${m.number} · ${e.label}`, hint: m.title, icon: e.icon, to: e.to });
          }
        }
      }
    }
    return out;
  }, [intent.action, matters.data, current, me, q]);

  const shown = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    const hits = words.length ? items.filter((it) => words.every((w) => `${it.label} ${it.hint ?? ''} ${it.group}`.toLowerCase().includes(w))) : items;
    return hits.slice(0, 60);
  }, [items, q]);
  useEffect(() => { setSel(0); }, [q]);
  useEffect(() => { list.current?.querySelector('[aria-selected="true"]')?.scrollIntoView?.({ block: 'nearest' }); }, [sel]);

  const go = (it?: Item) => { if (!it) return; onClose(); navigate(it.to); };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSel((s) => Math.min(s + 1, shown.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSel((s) => Math.max(s - 1, 0)); }
    else if (e.key === 'Enter') { e.preventDefault(); go(shown[sel]); }
    else if (e.key === 'Escape') { e.preventDefault(); onClose(); }
  };

  let lastGroup = '';
  return (
    <div className="lw-palette-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="lw-glass lw-palette" role="dialog" aria-modal="true" aria-label="Search or jump to">
        <div className="lw-palette-input">
          <Search aria-hidden style={{ width: 18, height: 18, color: 'var(--ink-3)' }} />
          <input ref={input} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={onKey}
            placeholder={intent.prompt ?? 'Search matters, tools and pages…'} aria-label={intent.prompt ?? 'Search or jump to'}
            role="combobox" aria-expanded="true" aria-controls="lw-palette-list" aria-autocomplete="list"
            aria-activedescendant={shown[sel] ? `lw-pal-${sel}` : undefined} />
          <span className="lw-kbd">Esc</span>
        </div>
        <ul className="lw-palette-list" id="lw-palette-list" role="listbox" ref={list} aria-label="Results">
          {shown.length === 0 && <li className="lw-meta" style={{ padding: 14 }}>Nothing matches. Try a matter number, a client or a tool.</li>}
          {shown.map((it, i) => {
            const head = it.group !== lastGroup ? it.group : null;
            lastGroup = it.group;
            const Icon = it.icon;
            return (
              <React.Fragment key={it.id}>
                {head && <li className="lw-palette-group" role="presentation">{head}</li>}
                <li id={`lw-pal-${i}`} role="option" aria-selected={i === sel} className="lw-palette-item"
                  onMouseMove={() => setSel(i)} onClick={() => go(it)}>
                  <Icon aria-hidden />
                  <span style={{ minWidth: 0, flex: 1 }}>
                    <span style={{ display: 'block', fontWeight: 550 }}>{it.label}</span>
                    {it.hint && <span className="lw-meta" style={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.hint}</span>}
                  </span>
                  {i === sel && <CornerDownLeft aria-hidden />}
                </li>
              </React.Fragment>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
