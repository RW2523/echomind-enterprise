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
  compact?: boolean;
}

const KIND_ICON: Record<string, string> = {
  customer: '🧑‍💼', client: '🧑', account_holder: '🏦', counterparty: '🤝', person: '👤',
};

/** Candidate/confirmed person pulled from the spoken details (name, id, phone…). */
const SubjectCard: React.FC<SubjectCardProps> = ({ subject, recordsCount, onConfirm, onReject, onClick, compact }) => {
  const confirmed = subject.status === 'confirmed';
  const rejected = subject.status === 'rejected';
  const count = recordsCount ?? subject.records_count ?? 0;
  const conf = typeof subject.confidence === 'number' ? Math.round(subject.confidence <= 1 ? subject.confidence * 100 : subject.confidence) : null;

  return (
    <div
      onClick={onClick ? () => onClick(subject.id) : undefined}
      className={`rounded-xl border p-3 transition-colors ${
        rejected ? 'border-white/10 bg-white/5 opacity-50'
        : confirmed ? 'border-emerald-500/40 bg-emerald-500/10'
        : 'border-violet-500/40 bg-violet-500/10'
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
              : 'bg-violet-500/20 text-violet-300 border-violet-500/40'
            }`}>
              {confirmed ? 'Confirmed' : rejected ? 'Not them' : 'Candidate'}
            </span>
          </div>
          <div className="text-[10px] text-slate-400 mt-0.5 flex items-center gap-1.5 flex-wrap">
            <span>{roleLabel(subject.kind)}</span>
            {conf != null && <span>· {conf}% match</span>}
            <span>· {count} record{count === 1 ? '' : 's'}</span>
          </div>
          {!compact && subject.matched_fields?.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {subject.matched_fields.map((f, i) => (
                <span key={`${f}-${i}`} className="text-[10px] rounded-md bg-white/10 border border-white/10 px-1.5 py-0.5 text-slate-200">
                  {f}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
      {!rejected && !confirmed && (onConfirm || onReject) && (
        <div className="mt-2.5 flex gap-2">
          {onConfirm && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onConfirm(subject.id); }}
              className="flex-1 rounded-lg px-3 py-1.5 text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-500/30 transition-colors touch-manipulation min-h-[32px]"
            >
              Confirm
            </button>
          )}
          {onReject && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onReject(subject.id); }}
              className="flex-1 rounded-lg px-3 py-1.5 text-xs font-semibold bg-white/10 text-slate-300 border border-white/10 hover:bg-white/15 transition-colors touch-manipulation min-h-[32px]"
            >
              Not them
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default SubjectCard;
