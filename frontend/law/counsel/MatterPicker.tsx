/**
 * The matters a conversation is about: in the header, one matter; in the composer, one or several (up to five, owner's
 * request 2026-10-28). Only the person's own matters appear; leaving a page with work in progress asks first. A highly
 * restricted matter is only ever discussed on its own.
 * Popovers are fixed-position and need no portal (nothing above them is transformed).
 */
import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useMatch, useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { Briefcase, Check, ChevronsUpDown, Lock } from 'lucide-react';
import { keys, useMatters } from '../api/queries';
import type { Matter } from '../api/types';
import { useCurrentMatter } from '../app/matter';
import { getBlockers } from '../app/navigationGuard';
import { Dialog } from '../ui/primitives';
import { MAX_MATTERS, useChatMatter, withMatter } from './context';

/** Popovers render on <body>: a glass surface (backdrop-filter) becomes the frame of fixed-position children, which would
 *  place a popover opened from the composer or the header relative to that surface and clip it (found in Chrome, 2026-10-28). */
export function Layer({ children }: { children: React.ReactNode }) {
  return createPortal(children, document.body) as React.ReactElement;
}

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
    // Keyboard (review A3): the popover sits at the end of <body>, so focus moves into it when it opens (unless something in
    // it took focus itself), leaving it closes it, and a menu answers the arrow keys.
    const first = requestAnimationFrame(() => {
      const el = pop.current;
      if (el && !el.contains(document.activeElement)) {
        (el.querySelector<HTMLElement>('[role="menuitem"], input, button, [href], [tabindex]:not([tabindex="-1"])') ?? el).focus();
      }
    });
    const inside = (t: EventTarget | null) => !!t && (button.current?.contains(t as Node) || pop.current?.contains(t as Node));
    const onDown = (e: MouseEvent) => { if (!inside(e.target)) setOpen(false); };
    const onFocus = (e: FocusEvent) => { if (!inside(e.target)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setOpen(false); button.current?.focus(); return; }
      const el = pop.current;
      if (!el || el.getAttribute('role') !== 'menu' || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) return;
      const items = [...el.querySelectorAll<HTMLElement>('[role="menuitem"]:not([disabled])')];
      if (!items.length) return;
      e.preventDefault();
      const at = items.indexOf(document.activeElement as HTMLElement);
      const next = e.key === 'Home' ? 0 : e.key === 'End' ? items.length - 1
        : e.key === 'ArrowDown' ? (at + 1) % items.length : (at - 1 + items.length) % items.length;
      items[next].focus();
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('focusin', onFocus);
    document.addEventListener('keydown', onKey);
    return () => {
      cancelAnimationFrame(first);
      document.removeEventListener('mousedown', onDown); document.removeEventListener('focusin', onFocus); document.removeEventListener('keydown', onKey);
    };
  }, [open]);
  return { open, setOpen, button, pop, style: { top: at.top, bottom: at.bottom, left: at.left, right: at.right } as React.CSSProperties };
}

export default function MatterPicker({ variant = 'header' }: { variant?: 'header' | 'composer' }) {
  const matters = useMatters();
  const routeMatter = useCurrentMatter();
  const { matter: chatMatter, matters: chosen, setMatterId, setMatterIds } = useChatMatter();
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
    if (current?.id === m.id && chosen.length <= 1) { setOpen(false); return; }
    const reasons = getBlockers();
    if (reasons.length) setPending({ target: m, reasons });
    else go(m);
  };
  /** The composer: tick matters on and off; a new set is a new conversation. */
  const toggle = (m: Matter) => {
    const next = withMatter(chosen, m);
    if (!next.length) return;                                   // a conversation is about at least one matter
    setMatterIds(next.map((x) => x.id));
    if (onChat) navigate('/chat');
  };

  const composer = variant === 'composer';
  const several = onChat && chosen.length > 1;
  const label = !current ? '' : several
    ? `${chosen.map((m) => m.number).join(', ')}. ${composer ? 'Change the matters' : 'Switch matter'}`
    : `${current.number} ${current.title}. ${composer ? 'Change the matter' : 'Switch matter'}`;
  return (
    <>
      <button ref={button} type="button" className={composer ? 'qc-tool' : 'qc-matter-btn'} aria-haspopup="dialog" aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        aria-label={current ? `${several ? 'Matters' : 'Matter'}: ${label}` : composer ? 'Pick the matter to ask about' : 'Choose a matter'}>
        {current && current.sensitivity !== 'standard' ? <Lock aria-label="Restricted" /> : <Briefcase className="qc-matter-ico" aria-hidden />}
        <span>{!current ? 'Choose a matter' : several
          ? <>{chosen[0].number}<span className="qc-matter-more"> + {chosen.length - 1} more</span></>
          : <>{current.number}<span className={composer ? '' : 'qc-hide-small'}> · {current.title}</span></>}</span>
        <ChevronsUpDown aria-hidden />
      </button>
      {open && (<Layer>
        <div ref={pop} className="qc-pop" style={style}>
          <input className="lw-input" autoFocus placeholder="Find a matter…" aria-label="Find a matter" value={filter} onChange={(e) => setFilter(e.target.value)} />
          <div className="qc-pop-label">{composer ? `Ask about one matter or several (up to ${MAX_MATTERS})` : 'Your matters'}</div>
          <div role="listbox" aria-label="Your matters" aria-multiselectable={composer || undefined} style={{ maxHeight: 320, overflowY: 'auto' }}>
            {list.map((m) => {
              const on = composer ? chosen.some((x) => x.id === m.id) : current?.id === m.id;
              const full = composer && !on && chosen.length >= MAX_MATTERS;
              return (
                <button key={m.id} type="button" role="option" aria-selected={on} className="qc-pop-item" disabled={full}
                  onClick={() => (composer ? toggle(m) : choose(m))}>
                  {composer && <span className="qc-check" data-on={on || undefined} aria-hidden>{on && <Check />}</span>}
                  <span style={{ minWidth: 0 }}>
                    <span style={{ display: 'block', fontWeight: 600 }}>
                      {m.number} · {m.title}{m.sensitivity === 'highly_restricted' && <Lock aria-label="Highly restricted" style={{ marginLeft: 6, verticalAlign: -2 }} />}
                    </span>
                    <span className="qc-meta">{m.client.name}</span>
                  </span>
                </button>
              );
            })}
            {matters.data && list.length === 0 && <p className="qc-meta" style={{ padding: '6px 10px' }}>No matching matters.</p>}
          </div>
          <p className="qc-meta" style={{ borderTop: '1px solid var(--rule)', margin: '6px 0 0', padding: '8px 10px 2px' }}>
            {composer
              ? 'Only your own matters appear. With several, every answer says which matter each source comes from; a highly restricted matter is discussed on its own.'
              : "Only matters you're a member of appear here. To ask about several at once, choose them in the question box."}
          </p>
        </div></Layer>)}
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
