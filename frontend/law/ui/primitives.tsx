import React, { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Check, Copy, X } from 'lucide-react';

export function Skeleton({ height = 16, width = '100%' }: { height?: number; width?: number | string }) {
  return <div className="lw-skeleton" style={{ height, width }} aria-hidden="true" />;
}

export type Tone = 'verified' | 'attention' | 'conflict' | 'privileged' | 'hold' | 'neutral' | 'accent';
/** Status is never colour alone: always a word, optionally an icon (DESIGN_SYSTEM §4.2). */
export function StatusPill({ tone, icon, children }: { tone: Tone; icon?: React.ReactNode; children: React.ReactNode }) {
  return <span className={`lw-pill lw-pill-${tone}`}>{icon}{children}</span>;
}

export function EmptyState({ title, children, action }: { title: string; children?: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="lw-paper lw-sheet lw-empty" style={{ textAlign: 'center', padding: '40px 24px' }}>
      <h2 className="lw-h2">{title}</h2>
      {children && <p className="lw-meta" style={{ margin: '8px auto 0', maxWidth: '52ch', fontSize: 14 }}>{children}</p>}
      {action && <div style={{ marginTop: 16 }}>{action}</div>}
    </div>
  );
}

export function ErrorNote({ children }: { children: React.ReactNode }) {
  return <p role="alert" style={{ color: 'var(--conflict)', margin: 0, fontSize: 13.5 }}>{children}</p>;
}

/** A labelled control. The hint is linked with aria-describedby, so it is announced as a description
 *  and does not become part of the field's accessible name. */
export function Field({ label, hint, children }: { label: string; hint?: string; children: (id: string) => React.ReactElement }) {
  const id = useId();
  const hintId = `${id}-hint`;
  const control = children(id);
  return (
    <div className="lw-label">
      <label htmlFor={id}>{label}</label>
      {hint ? React.cloneElement(control as React.ReactElement<any>, { 'aria-describedby': hintId }) : control}
      {hint && <span id={hintId} className="lw-hint">{hint}</span>}
    </div>
  );
}

export function CopyField({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="lw-label">
      <span>{label}</span>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <div className="lw-code" style={{ flex: 1 }} data-testid="copy-value">{value}</div>
        <button type="button" className="lw-btn lw-btn-ghost lw-btn-sm" aria-label={`Copy ${label.toLowerCase()}`}
          onClick={async () => { try { await navigator.clipboard.writeText(value); setCopied(true); } catch { /* clipboard blocked */ } }}>
          {copied ? <Check /> : <Copy />}{copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    </div>
  );
}

/** Modal sheet: glass frame, paper body (no glass-on-glass). Escape and the close button dismiss it. */
export function Dialog({ title, onClose, children, footer }: {
  title: string; onClose: () => void; children: React.ReactNode; footer?: React.ReactNode;
}) {
  const titleId = useId();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    ref.current?.querySelector<HTMLElement>('input, select, textarea, button')?.focus();
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);
  // Portalled to <body>: glass surfaces (the sidebar) create their own stacking context, so a
  // dialog opened from inside one would otherwise sit underneath the content.
  return createPortal(
    <div className="lw-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div ref={ref} className="lw-glass lw-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="lw-paper">
          <div className="lw-card-h">
            <h2 className="lw-h2" id={titleId}>{title}</h2>
            <button type="button" className="lw-btn lw-btn-sm" aria-label="Close" onClick={onClose}><X /></button>
          </div>
          <div className="lw-dialog-body">{children}</div>
          {footer && <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '0 20px 20px' }}>{footer}</div>}
        </div>
      </div>
    </div>,
    document.body,
  );
}

const ROLE_LABEL: Record<string, string> = {
  lead: 'Lead', attorney: 'Attorney', paralegal: 'Paralegal', reviewer: 'Reviewer', read_only: 'Read only',
  firm_admin: 'Firm admin', managing_partner: 'Managing partner', records_admin: 'Records admin', break_glass: 'Emergency access',
  intake: 'Intake (front desk)', conflicts: 'Conflicts team',
};
export const roleLabel = (r?: string | null) => (r ? ROLE_LABEL[r] ?? r : '');

const SENSITIVITY: Record<string, [string, Tone]> = {
  standard: ['Standard', 'neutral'], restricted: ['Restricted', 'attention'], highly_restricted: ['Highly restricted', 'conflict'],
};
export function SensitivityPill({ value }: { value: string }) {
  const [label, tone] = SENSITIVITY[value] ?? [value, 'neutral'];
  return <StatusPill tone={tone}>{label}</StatusPill>;
}
const STATUS: Record<string, [string, Tone]> = { open: ['Open', 'verified'], closed: ['Closed', 'neutral'], archived: ['Archived', 'neutral'] };
export function MatterStatusPill({ value }: { value: string }) {
  const [label, tone] = STATUS[value] ?? [value, 'neutral'];
  return <StatusPill tone={tone}>{label}</StatusPill>;
}
