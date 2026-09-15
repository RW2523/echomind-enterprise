import React from 'react';
import type { Subject } from '../types';
import { roleLabel } from '../utils/silentAssistant';

interface SubjectCardProps {
  subject: Subject;
  /** Records currently linked to this subject (overrides subject.records_count when provided). */
  recordsCount?: number;
  onConfirm?: (id: string) => void;
  onReject?: (id: string) => void;
  onClick?: (id: string) => void;
  /** One-line layout (default in the insights column): name · kind · records + Confirm / Not them. */
  compact?: boolean;
}

const KIND_ICON: Record<string, string> = {
  customer: '🧑‍💼', client: '🧑', account_holder: '🏦', counterparty: '🤝', person: '👤',
};

/** Candidate/confirmed person pulled from the spoken details (name, id, phone…). */
const SubjectCard: React.FC<SubjectCardProps> = ({ subject, recordsCount, onConfirm, onReject, onClick, compact = true }) => {
  const confirmed = subject.status === 'confirmed';
  const rejected = subject.status === 'rejected';
  const count = recordsCount ?? subject.records_count ?? 0;
  const conf = typeof subject.confidence === 'number' ? Math.round(subject.confidence <= 1 ? subject.confidence * 100 : subject.confidence) : null;
  const meta = [roleLabel(subject.kind), conf != null && !confirmed ? `${conf}% match` : null, `${count} record${count === 1 ? '' : 's'}`].filter(Boolean).join(' · ');

  if (compact) {
    return (
      <div
        onClick={onClick ? () => onClick(subject.id) : undefined}
        className={`flex items-center gap-2 rounded-lg border px-2.5 py-1.5 transition-colors ${
          rejected ? 'border-white/10 bg-white/[0.03] opacity-50'
          : confirmed ? 'border-emerald-500/25 bg-emerald-500/[0.06]'
          : 'border-white/10 bg-white/[0.04]'
        } ${onClick ? 'cursor-pointer hover:bg-white/[0.07]' : ''}`}
        title={subject.matched_fields?.length ? `Matched on: ${subject.matched_fields.join(', ')}` : undefined}
      >
        <span className="text-base leading-none shrink-0">{KIND_ICON[subject.kind] ?? '👤'}</span>
        <div className="min-w-0 flex-1 flex items-baseline gap-1.5">
          <span className="text-xs font-semibold text-white truncate">{subject.display_name}</span>
          <span className="text-[11px] text-slate-500 truncate">{meta}</span>
        </div>
        {confirmed && <span className="text-[10px] font-semibold text-emerald-300 shrink-0">Confirmed</span>}
        {!rejected && !confirmed && (onConfirm || onReject) && (
          <div className="shrink-0 flex items-center gap-1">
            {onConfirm && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onConfirm(subject.id); }}
                className="rounded-md px-2 py-1 text-[11px] font-semibold bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/25 transition-colors touch-manipulation"
              >
                Confirm
              </button>
            )}
            {onReject && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onReject(subject.id); }}
                className="rounded-md px-2 py-1 text-[11px] font-semibold text-slate-400 hover:text-white hover:bg-white/10 transition-colors touch-manipulation"
              >
                Not them
              </button>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      onClick={onClick ? () => onClick(subject.id) : undefined}
      className={`rounded-xl border p-3 transition-colors ${
        rejected ? 'border-white/10 bg-white/5 opacity-50'
        : confirmed ? 'border-emerald-500/30 bg-emerald-500/[0.08]'
        : 'border-white/10 bg-white/[0.04]'
      } ${onClick ? 'cursor-pointer hover:brightness-110' : ''}`}
    >
      <div className="flex items-start gap-2.5">
        <div className={`w-9 h-9 shrink-0 rounded-xl flex items-center justify-center text-lg ${confirmed ? 'bg-emerald-500/20' : 'bg-white/10'}`}>
          {KIND_ICON[subject.kind] ?? '👤'}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-semibold text-white truncate">{subject.display_name}</span>
            <span className={`text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded-md border ${
              confirmed ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
              : rejected ? 'bg-white/10 text-slate-400 border-white/15'
              : 'bg-white/10 text-slate-300 border-white/15'
            }`}>
              {confirmed ? 'Confirmed' : rejected ? 'Not them' : 'Candidate'}
            </span>
          </div>
          <div className="text-[11px] text-slate-400 mt-0.5">{meta}</div>
          {subject.matched_fields?.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {subject.matched_fields.map((f, i) => (
                <span key={`${f}-${i}`} className="text-[10px] rounded-md bg-white/10 border border-white/10 px-1.5 py-0.5 text-slate-200">{f}</span>
              ))}
            </div>
          )}
        </div>
      </div>
      {!rejected && !confirmed && (onConfirm || onReject) && (
        <div className="mt-2.5 flex gap-2">
          {onConfirm && (
            <button type="button" onClick={(e) => { e.stopPropagation(); onConfirm(subject.id); }} className="flex-1 rounded-lg px-3 py-1.5 text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/30 transition-colors touch-manipulation min-h-[32px]">
              Confirm
            </button>
          )}
          {onReject && (
            <button type="button" onClick={(e) => { e.stopPropagation(); onReject(subject.id); }} className="flex-1 rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-300 border border-white/10 hover:bg-white/15 transition-colors touch-manipulation min-h-[32px]">
              Not them
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default SubjectCard;
