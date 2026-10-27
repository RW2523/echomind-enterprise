/**
 * The matter a conversation is about: one button and its list, in the header and again in the composer. Only the
 * person's own matters appear; leaving a page with work in progress asks first.
 * Popovers are fixed-position and need no portal (nothing above them is transformed).
 */
import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useMatch, useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { Briefcase, ChevronsUpDown, Lock } from 'lucide-react';
import { keys, useMatters } from '../api/queries';
import type { Matter } from '../api/types';
import { useCurrentMatter } from '../app/matter';
import { getBlockers } from '../app/navigationGuard';
import { Dialog } from '../ui/primitives';
import { useChatMatter } from './context';

export function usePopover() {
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  const pop = useRef<HTMLDivElement>(null);
  const [at, setAt] = useState<{ top?: number; bottom?: number; left?: number; right?: number }>({ top: 0 });
  useLayoutEffect(() => {
    if (!open) return;
    const r = button.current?.getBoundingClientRect();
    if (!r) return;
    const x = r.left > window.innerWidth / 2 ? { right: Math.max(8, window.innerWidth - r.right) } : { left: r.left };
    // Below the button, unless it sits low on the screen (the composer): then above it.
    setAt(r.bottom > window.innerHeight * 0.6 ? { bottom: window.innerHeight - r.top + 6, ...x } : { top: r.bottom + 6, ...x });
  }, [open]);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!button.current?.contains(t) && !pop.current?.contains(t)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { setOpen(false); button.current?.focus(); } };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey); };
  }, [open]);
  return { open, setOpen, button, pop, style: { top: at.top, bottom: at.bottom, left: at.left, right: at.right } as React.CSSProperties };
}

export default function MatterPicker({ variant = 'header' }: { variant?: 'header' | 'composer' }) {
  const matters = useMatters();
  const routeMatter = useCurrentMatter();
  const { matter: chatMatter, setMatterId } = useChatMatter();
  const onChat = !!useMatch('/chat/*');
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { open, setOpen, button, pop, style } = usePopover();
  const [filter, setFilter] = useState('');
  const [pending, setPending] = useState<{ target: Matter; reasons: string[] } | null>(null);
  const current = onChat ? chatMatter : routeMatter ?? chatMatter;
  const list = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return (matters.data ?? []).filter((m) => !q || `${m.number} ${m.title} ${m.client.name}`.toLowerCase().includes(q));
  }, [matters.data, filter]);

  const go = (m: Matter) => {
    setPending(null);
    setOpen(false);
    if (routeMatter && routeMatter.id !== m.id) qc.removeQueries({ queryKey: keys.matter(routeMatter.id) });
    const changed = chatMatter?.id !== m.id;
    setMatterId(m.id);
    if (onChat) { if (changed) navigate('/chat'); }         // a conversation is about one matter: a new matter is a new chat
    else if (routeMatter) navigate(`/m/${encodeURIComponent(m.number)}/overview`);
  };
  const choose = (m: Matter) => {
    if (current?.id === m.id) { setOpen(false); return; }
    const reasons = getBlockers();
    if (reasons.length) setPending({ target: m, reasons });
    else go(m);
  };

  const composer = variant === 'composer';
  return (
    <>
      <button ref={button} type="button" className={composer ? 'qc-tool' : 'qc-matter-btn'} aria-haspopup="listbox" aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        aria-label={current ? `Matter: ${current.number} ${current.title}. ${composer ? 'Change the matter' : 'Switch matter'}`
          : composer ? 'Pick the matter to ask about' : 'Choose a matter'}>
        {current && current.sensitivity !== 'standard' ? <Lock aria-label="Restricted" /> : <Briefcase className="qc-matter-ico" aria-hidden />}
        <span>{current ? <>{current.number}<span className={composer ? '' : 'qc-hide-small'}> · {current.title}</span></> : 'Choose a matter'}</span>
        <ChevronsUpDown aria-hidden />
      </button>
      {open && (
        <div ref={pop} className="qc-pop" style={style}>
          <input className="lw-input" autoFocus placeholder="Find a matter…" aria-label="Find a matter" value={filter} onChange={(e) => setFilter(e.target.value)} />
          <div className="qc-pop-label">Your matters</div>
          <div role="listbox" aria-label="Your matters" style={{ maxHeight: 320, overflowY: 'auto' }}>
            {list.map((m) => (
              <button key={m.id} type="button" role="option" aria-selected={current?.id === m.id} className="qc-pop-item" onClick={() => choose(m)}>
                <span style={{ minWidth: 0 }}>
                  <span style={{ display: 'block', fontWeight: 600 }}>{m.number} · {m.title}</span>
                  <span className="qc-meta">{m.client.name}</span>
                </span>
              </button>
            ))}
            {matters.data && list.length === 0 && <p className="qc-meta" style={{ padding: '6px 10px' }}>No matching matters.</p>}
          </div>
          <p className="qc-meta" style={{ borderTop: '1px solid var(--rule)', margin: '6px 0 0', padding: '8px 10px 2px' }}>
            Only matters you're a member of appear here. Each conversation uses one matter's documents.
          </p>
        </div>)}
      {pending && (
        <Dialog title={`Switch to ${pending.target.number}?`} onClose={() => setPending(null)}
          footer={<>
            <button type="button" className="lw-btn lw-btn-ghost" onClick={() => setPending(null)}>Stay here</button>
            <button type="button" className="lw-btn lw-btn-primary" onClick={() => go(pending.target)}>Switch matter</button>
          </>}>
          <p style={{ margin: 0 }}>Something is still in progress:</p>
          <ul style={{ margin: 0, paddingLeft: 18 }}>{pending.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
        </Dialog>
      )}
    </>
  );
}
