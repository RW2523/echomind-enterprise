import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { ChevronsUpDown, Lock } from 'lucide-react';
import { keys, useMatters } from '../api/queries';
import type { Matter } from '../api/types';
import { useCurrentMatter } from '../app/matter';
import { getBlockers } from '../app/navigationGuard';
import { Dialog } from './primitives';

// Company-form suffixes carry no identity ("Voss-Lindqvist AB" → VL, not VA).
const SUFFIXES = new Set(['ab', 'ag', 'as', 'bv', 'co', 'corp', 'gmbh', 'inc', 'llc', 'llp', 'lp', 'ltd', 'nv', 'oy', 'plc', 'pty', 'sa', 'sas', 'spa']);
export const monogram = (name: string) => {
  const words = name.split(/[\s\-–—&,.]+/).filter((w) => w && !SUFFIXES.has(w.toLowerCase()));
  return words.slice(0, 2).map((w) => w[0]!.toUpperCase()).join('') || '·';
};

/**
 * Always-visible matter context. Lists only the person's own matters (server-filtered). Switching
 * drops the previous matter's cached data, and asks first if something is in progress (an active
 * recording, an unsaved draft) — see app/navigationGuard.ts.
 */
export default function MatterSwitcher() {
  const current = useCurrentMatter();
  const matters = useMatters();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const [pending, setPending] = useState<{ target: Matter; reasons: string[] } | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const [anchor, setAnchor] = useState<{ top: number; left: number } | null>(null);

  useEffect(() => {
    if (!open) return;
    const r = boxRef.current?.getBoundingClientRect();
    if (r) setAnchor({ top: r.bottom + 6, left: r.left });
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!boxRef.current?.contains(t) && !popRef.current?.contains(t)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey); };
  }, [open]);

  const visible = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return (matters.data ?? []).filter((m) => !q || `${m.number} ${m.title} ${m.client.name}`.toLowerCase().includes(q));
  }, [matters.data, filter]);

  const go = (target: Matter) => {
    if (current && current.id !== target.id) qc.removeQueries({ queryKey: keys.matter(current.id) });
    setPending(null);
    setOpen(false);
    navigate(`/m/${encodeURIComponent(target.number)}/overview`);
  };
  const choose = (target: Matter) => {
    if (current?.id === target.id) { setOpen(false); return; }
    const reasons = getBlockers();
    if (reasons.length) setPending({ target, reasons });
    else go(target);
  };

  return (
    <div ref={boxRef} style={{ position: 'relative' }}>
      <button type="button" className="lw-switcher" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        aria-label={current ? `Matter: ${current.number} ${current.title}. Switch matter` : 'Choose a matter'}>
        <span className={`lw-monogram${current ? '' : ' lw-muted'}`}>{current ? monogram(current.client.name) : '—'}</span>
        <span className="lw-hide-rail" style={{ minWidth: 0, flex: 1 }}>
          <span style={{ display: 'block', fontWeight: 600, fontSize: 13, lineHeight: '17px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {current ? current.client.name : 'No matter selected'}
          </span>
          <span className="lw-meta" style={{ display: 'flex', gap: 6, alignItems: 'center', minWidth: 0 }}>
            {current?.sensitivity !== 'standard' && current && <Lock size={12} aria-label="Restricted" style={{ flex: 'none' }} />}
            <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {current ? `${current.number} · ${current.title}` : 'Choose a matter'}
            </span>
          </span>
        </span>
        <ChevronsUpDown className="lw-hide-rail" size={16} style={{ color: 'var(--ink-3)' }} />
      </button>

      {open && anchor && createPortal(
        // Portalled with fixed positioning so it floats above the content, not inside the sidebar's glass.
        <div ref={popRef} className="lw-glass lw-popover" style={{ position: 'fixed', top: anchor.top, left: anchor.left }}>
          <input className="lw-input" autoFocus placeholder="Find a matter…" aria-label="Find a matter"
            value={filter} onChange={(e) => setFilter(e.target.value)} />
          <div className="lw-eyebrow" style={{ padding: '10px 10px 4px' }}>Your matters</div>
          <div role="listbox" aria-label="Your matters">
            {visible.map((m) => (
              <button key={m.id} type="button" role="option" aria-selected={current?.id === m.id} className="lw-option" onClick={() => choose(m)}>
                <span className="lw-monogram lw-muted" style={{ width: 26, height: 26, borderRadius: 8, fontSize: 10 }}>{monogram(m.client.name)}</span>
                <span style={{ minWidth: 0 }}>
                  <span style={{ display: 'block', fontWeight: 600, fontSize: 13 }}>{m.number} · {m.title}</span>
                  <span className="lw-meta">{m.client.name}</span>
                </span>
              </button>
            ))}
            {matters.data && visible.length === 0 && <p className="lw-meta" style={{ padding: '6px 10px' }}>No matching matters.</p>}
          </div>
          <p className="lw-meta" style={{ borderTop: '1px solid var(--rule)', margin: '6px 0 0', padding: '8px 10px 2px' }}>
            Only matters you're a member of appear here.
          </p>
        </div>,
        document.body,
      )}

      {pending && (
        <Dialog title={`Switch to ${pending.target.number}?`} onClose={() => setPending(null)}
          footer={<>
            <button type="button" className="lw-btn lw-btn-ghost" onClick={() => setPending(null)}>Stay here</button>
            <button type="button" className="lw-btn lw-btn-primary" onClick={() => go(pending.target)}>Switch matter</button>
          </>}>
          <p style={{ margin: 0 }}>Something is still in progress in {current?.number}:</p>
          <ul style={{ margin: 0, paddingLeft: 18 }}>{pending.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
        </Dialog>
      )}
    </div>
  );
}
