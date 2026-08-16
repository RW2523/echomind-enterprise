import React, { useEffect, useRef, useState } from 'react';
import type { EvidenceQuote } from '../types';
import { documentFileUrl } from '../services/backend';
import { copyToClipboard, sourceLine } from '../utils/silentAssistant';

interface ProofPopoverProps {
  evidence: EvidenceQuote;
  /** Extra line under the quote (e.g. the check's explanation) */
  note?: string;
  /** Show on hover of children (default) and/or toggle on click. */
  trigger?: 'hover' | 'click' | 'both';
  /** Anchor: the element the popover attaches to */
  children: React.ReactNode;
  className?: string;
  /** Preferred side; auto-flips when there's no room. */
  side?: 'top' | 'bottom';
  /** Render the wrapper as a block (for whole-paragraph anchors). */
  block?: boolean;
}

/** Small "proof" bubble: quote, doc · page · section (or a Rule badge), Copy. */
export const ProofBody: React.FC<{ evidence: EvidenceQuote; note?: string; compact?: boolean }> = ({ evidence, note, compact }) => {
  const [copied, setCopied] = useState(false);
  const isRule = evidence.kind === 'rule';
  const isTranscript = evidence.kind === 'transcript';
  const src = sourceLine(evidence.doc_title, evidence.page, evidence.section_path);
  const canOpen = !!evidence.doc_id && !isRule && !isTranscript;

  const onCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    const text = evidence.quote + (src ? ` — ${src}` : '') + (isRule && evidence.rule_id ? ` [rule ${evidence.rule_id}]` : '');
    if (await copyToClipboard(text)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    }
  };

  return (
    <div className={compact ? 'space-y-1.5' : 'space-y-2'}>
      <div className="flex items-center gap-1.5">
        {isRule ? (
          <span className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded-md bg-orange-500/20 text-orange-300 border border-orange-500/40">Rule</span>
        ) : isTranscript ? (
          <span className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/40">Transcript</span>
        ) : (
          <span className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded-md bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">Source</span>
        )}
        <span className="text-[10px] text-slate-400 truncate" title={src || evidence.rule_id || ''}>
          {isRule ? (evidence.rule_id ? `rule: ${evidence.rule_id}` : 'domain rule') : src || 'Knowledge base'}
        </span>
        <div className="ml-auto flex items-center gap-1 shrink-0">
          {canOpen && (
            <a
              href={documentFileUrl(evidence.doc_id!, evidence.page)}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="text-[10px] font-semibold text-cyan-300 hover:text-white px-1.5 py-0.5 rounded-md hover:bg-white/10"
              title="Open document"
            >
              Open
            </a>
          )}
          <button
            type="button"
            onClick={onCopy}
            className="text-[10px] font-semibold text-slate-300 hover:text-white px-1.5 py-0.5 rounded-md hover:bg-white/10"
            title="Copy quote"
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>
      <blockquote className={`border-l-2 ${isRule ? 'border-orange-400/60' : 'border-cyan-400/60'} pl-2.5 text-[12px] leading-relaxed text-white/90 italic ${compact ? 'line-clamp-4' : ''}`}>
        “{evidence.quote}”
      </blockquote>
      {note && <p className="text-[11px] text-slate-400 leading-relaxed">{note}</p>}
    </div>
  );
};

const ProofPopover: React.FC<ProofPopoverProps> = ({ evidence, note, trigger = 'both', children, className = '', side = 'top', block }) => {
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [flip, setFlip] = useState<'top' | 'bottom'>(side);
  const [alignRight, setAlignRight] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const hover = trigger === 'hover' || trigger === 'both';
  const click = trigger === 'click' || trigger === 'both';

  useEffect(() => {
    if (!open) return;
    const el = wrapRef.current;
    if (el) {
      const r = el.getBoundingClientRect();
      // flip if there is no room above (or below)
      if (side === 'top' && r.top < 190) setFlip('bottom');
      else if (side === 'bottom' && window.innerHeight - r.bottom < 190) setFlip('top');
      else setFlip(side);
      // Keep the bubble inside the nearest scroll container (or the viewport).
      let host: HTMLElement | null = el.parentElement;
      while (host && host !== document.body) {
        const ov = getComputedStyle(host).overflowX;
        if (ov === 'auto' || ov === 'scroll' || ov === 'hidden') break;
        host = host.parentElement;
      }
      const rightEdge = host && host !== document.body ? host.getBoundingClientRect().right : window.innerWidth;
      setAlignRight(r.left + 300 > rightEdge && r.right - 300 > 0);
    }
    const onDoc = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) { setOpen(false); setPinned(false); }
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { setOpen(false); setPinned(false); } };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey); };
  }, [open, side]);

  const onEnter = () => {
    if (!hover) return;
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    hoverTimer.current = setTimeout(() => setOpen(true), 180);
  };
  const onLeave = () => {
    if (!hover) return;
    if (hoverTimer.current) clearTimeout(hoverTimer.current);
    if (!pinned) hoverTimer.current = setTimeout(() => setOpen(false), 160);
  };
  const onClick = (e: React.MouseEvent) => {
    if (!click) return;
    e.stopPropagation();
    if (pinned) { setPinned(false); setOpen(false); }
    else { setPinned(true); setOpen(true); }
  };

  return (
    <span ref={wrapRef} className={`relative ${block ? 'block' : 'inline'} ${className}`} onMouseEnter={onEnter} onMouseLeave={onLeave} onClick={onClick}>
      {children}
      {open && (
        <span
          role="tooltip"
          onMouseEnter={onEnter}
          onMouseLeave={onLeave}
          onClick={(e) => e.stopPropagation()}
          className={`absolute z-[70] ${alignRight ? 'right-0' : 'left-0'} ${flip === 'top' ? 'bottom-full mb-1.5' : 'top-full mt-1.5'} w-72 max-w-[80vw] block rounded-xl border border-white/15 bg-slate-900/95 backdrop-blur shadow-2xl p-3 text-left not-italic normal-case font-normal tracking-normal cursor-default`}
        >
          <ProofBody evidence={evidence} note={note} compact />
        </span>
      )}
    </span>
  );
};

export default ProofPopover;
