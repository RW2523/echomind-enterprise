import React, { useEffect, useMemo, useState } from 'react';
import type { ActionItem, AnalysisCard, RecordHit, SentenceCheck, Subject, TagSpec } from '../types';
import AnalysisPanel from './AnalysisPanel';
import RecordPanel from './RecordPanel';
import ActionsPanel from './ActionsPanel';
import SubjectCard from './SubjectCard';

export type AssistantTab = 'records' | 'checks' | 'actions';

interface AssistantSidebarProps {
  cards: AnalysisCard[];
  records: RecordHit[];
  actionItems: ActionItem[];
  subjects: Subject[];
  vocab?: TagSpec[];
  analyzingSegmentIds?: Set<string>;
  selectedSegmentId: string | null;
  onSelectSegment: (id: string | null) => void;
  selectedSentenceId: string | null;
  onSelectSentence: (id: string | null) => void;
  onOpenCard: (check: SentenceCheck) => void;
  onConfirmSubject?: (id: string) => void;
  onRejectSubject?: (id: string) => void;
  /** Controlled tab (so desktop column and mobile sheet share state). */
  tab: AssistantTab;
  onTabChange: (t: AssistantTab) => void;
  /** Unread counters per tab (owned by parent via useAssistantUnread). */
  unread: Record<AssistantTab, number>;
  /** Show records tab (hidden when analysis mode is flags-only and nothing arrived). */
  showRecords?: boolean;
  namespaceEmptyHint?: string;
  onClose?: () => void;
}

/** Tracks unread counts per tab: increments when items arrive on a non-active tab. */
export function useAssistantUnread(counts: Record<AssistantTab, number>, activeTab: AssistantTab) {
  const [seen, setSeen] = useState<Record<AssistantTab, number>>({ records: 0, checks: 0, actions: 0 });
  useEffect(() => {
    setSeen((s) => (s[activeTab] === counts[activeTab] ? s : { ...s, [activeTab]: counts[activeTab] }));
  }, [activeTab, counts.records, counts.checks, counts.actions]); // eslint-disable-line react-hooks/exhaustive-deps
  const unread = useMemo<Record<AssistantTab, number>>(() => ({
    records: Math.max(0, counts.records - seen.records),
    checks: Math.max(0, counts.checks - seen.checks),
    actions: Math.max(0, counts.actions - seen.actions),
  }), [counts.records, counts.checks, counts.actions, seen]);
  const reset = () => setSeen({ records: 0, checks: 0, actions: 0 });
  return { unread, reset };
}

/**
 * Insights column: compact "Who is this?" row, then a compact tab strip
 * `Records (2) | Checks | Actions (6)` — counts only when > 0, never "+N" badges.
 */
const AssistantSidebar: React.FC<AssistantSidebarProps> = ({
  cards, records, actionItems, subjects, vocab, analyzingSegmentIds,
  selectedSegmentId, onSelectSegment, selectedSentenceId, onSelectSentence, onOpenCard,
  onConfirmSubject, onRejectSubject, tab, onTabChange, unread, showRecords = true, namespaceEmptyHint, onClose,
}) => {
  const visibleSubjects = subjects.filter((s) => s.status !== 'rejected');
  const recordsBySubject = useMemo(() => {
    const m = new Map<string, number>();
    for (const r of records) if (r.subject_id) m.set(r.subject_id, (m.get(r.subject_id) ?? 0) + 1);
    return m;
  }, [records]);

  const tabs: { id: AssistantTab; label: string }[] = [
    ...(showRecords ? [{ id: 'records' as const, label: records.length > 0 ? `Records (${records.length})` : 'Records' }] : []),
    { id: 'checks', label: 'Checks' },
    { id: 'actions', label: actionItems.length > 0 ? `Actions (${actionItems.length})` : 'Actions' },
  ];

  return (
    <div className="h-full flex flex-col min-h-0">
      {onClose && (
        <div className="shrink-0 flex items-center px-3 py-2 border-b border-white/[0.06] md:hidden">
          <span className="text-xs font-semibold text-slate-300">Insights</span>
          <button type="button" onClick={onClose} className="ml-auto p-2 rounded-lg text-slate-400 hover:text-white hover:bg-white/10" aria-label="Close">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
      )}

      {/* Who is this? — compact, one line per person */}
      {visibleSubjects.length > 0 && (
        <div className="shrink-0 px-2.5 pt-2.5 pb-2 space-y-1.5 border-b border-white/[0.06]">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-slate-500 px-0.5">
            {visibleSubjects.some((s) => s.status === 'confirmed') ? 'On the call' : 'Who is this?'}
          </div>
          {visibleSubjects.map((s) => (
            <SubjectCard
              key={s.id}
              subject={s}
              compact
              recordsCount={recordsBySubject.get(s.id) ?? s.records_count}
              onConfirm={onConfirmSubject}
              onReject={onRejectSubject}
              onClick={showRecords ? () => onTabChange('records') : undefined}
            />
          ))}
        </div>
      )}

      {/* Compact tab strip */}
      <div className="shrink-0 flex items-center gap-1 px-2 pt-2 pb-1">
        {tabs.map((t) => {
          const active = tab === t.id;
          const hasUnread = unread[t.id] > 0 && !active;
          return (
            <button
              key={t.id}
              type="button"
              onClick={() => onTabChange(t.id)}
              className={`relative px-2.5 py-1.5 rounded-lg text-xs font-semibold transition-colors touch-manipulation ${
                active ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white hover:bg-white/[0.06]'
              }`}
            >
              {t.label}
              {hasUnread && <span className="absolute top-1 right-1 w-1.5 h-1.5 rounded-full bg-cyan-400" aria-label="new items" />}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div className="flex-1 min-h-0 overflow-auto">
        {tab === 'records' && showRecords ? (
          <RecordPanel
            records={records}
            subjects={subjects}
            selectedSentenceId={selectedSentenceId}
            onSelectSentence={onSelectSentence}
            emptyHint={namespaceEmptyHint}
          />
        ) : tab === 'actions' ? (
          <ActionsPanel items={actionItems} vocab={vocab} selectedSentenceId={selectedSentenceId} onSelect={onOpenCard} />
        ) : (
          <AnalysisPanel
            cards={cards}
            vocab={vocab}
            selectedSegmentId={selectedSegmentId}
            onSelectSegment={onSelectSegment}
            selectedSentenceId={selectedSentenceId}
            onSelectSentence={onSelectSentence}
            analyzingSegmentIds={analyzingSegmentIds}
            onOpenCard={onOpenCard}
            embedded
          />
        )}
      </div>
    </div>
  );
};

export default AssistantSidebar;
