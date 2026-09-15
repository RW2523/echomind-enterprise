import React, { useState, useCallback, useRef, useMemo, useEffect } from 'react';
import type { AnalysisCard, SentenceCheck, TagSpec } from '../types';
import AnalysisCardModal, { checkGlyph, checkStyle } from './AnalysisCardModal';
import TagChip from './TagChip';
import ProofPopover from './ProofPopover';
import { useHandraise, cardPhrase } from '../hooks/useHandraise';
import {
  QUICK_FILTER_TAGS, asCheck, checkTagIds, isFlagged, isSupported, roleLabel, sortChecksProblemsFirst, sourceLine, tagSpecFor,
} from '../utils/silentAssistant';

export { cardPhrase };

interface AnalysisPanelProps {
  cards: AnalysisCard[];
  selectedSegmentId: string | null;
  onSelectSegment: (id: string | null) => void;
  analyzingSegmentIds?: Set<string>;
  /** v2: session tag vocabulary for chip colours/labels */
  vocab?: TagSpec[];
  /** v2: sentence-level selection (preferred over segment when provided) */
  selectedSentenceId?: string | null;
  onSelectSentence?: (id: string | null) => void;
  /** v2: when given, clicking a card calls this instead of opening the built-in modal */
  onOpenCard?: (card: SentenceCheck) => void;
  /** Hide the "Checks" title (when embedded in a tab strip) */
  embedded?: boolean;
}

type QuickFilter = 'all' | 'supported' | 'flagged';
const QUICK: { id: QuickFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'supported', label: 'Supported' },
  { id: 'flagged', label: 'Flagged' },
];
/** Only the most recent N checks are shown until "View all". */
const RECENT_LIMIT = 5;

/**
 * Checks tab: `All | Supported | Flagged` quick filters (+ a filter popover for the remaining tags),
 * the 5 most recent checks with problems pinned first, and a "View all N" expander.
 */
const AnalysisPanel: React.FC<AnalysisPanelProps> = ({
  cards, selectedSegmentId, onSelectSegment, analyzingSegmentIds, vocab, selectedSentenceId, onSelectSentence, onOpenCard, embedded,
}) => {
  const [expandedCard, setExpandedCard] = useState<SentenceCheck | null>(null);
  const [quick, setQuick] = useState<QuickFilter>('all');
  const [tagFilters, setTagFilters] = useState<Set<string>>(() => new Set());
  const [showAll, setShowAll] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);
  const filterRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef(new Map<string, HTMLDivElement>());
  const scrolledToRef = useRef<string | null>(null);
  const { speaking, speakCard, stop } = useHandraise();

  const isAnalyzing = (analyzingSegmentIds?.size ?? 0) > 0;

  const checks = useMemo(() => cards.map(asCheck), [cards]); // arrival order (oldest first)
  const flaggedCount = useMemo(() => checks.filter(isFlagged).length, [checks]);

  // Tags present that are NOT covered by the quick filters (ordered by vocab order) -> filter popover.
  const extraTags = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of checks) for (const id of new Set(checkTagIds(c))) if (!QUICK_FILTER_TAGS.has(id)) counts.set(id, (counts.get(id) ?? 0) + 1);
    const order = (vocab ?? []).map((t) => t.id);
    return [...counts.entries()]
      .sort((a, b) => {
        const ia = order.indexOf(a[0]), ib = order.indexOf(b[0]);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
      })
      .map(([id, n]) => ({ id, n }));
  }, [checks, vocab]);

  const filtered = useMemo(() => checks.filter((c) => {
    if (quick === 'supported' && !isSupported(c)) return false;
    if (quick === 'flagged' && !isFlagged(c)) return false;
    if (tagFilters.size > 0 && !checkTagIds(c).some((t) => tagFilters.has(t))) return false;
    return true;
  }), [checks, quick, tagFilters]);

  // Collapsed: the RECENT_LIMIT newest of the filtered list, problems pinned first within those.
  const visible = useMemo(
    () => sortChecksProblemsFirst(showAll ? filtered : filtered.slice(-RECENT_LIMIT)),
    [filtered, showAll]
  );
  const hiddenCount = Math.max(0, filtered.length - RECENT_LIMIT);

  useEffect(() => {
    if (!filterOpen) return;
    const onDoc = (e: MouseEvent) => { if (filterRef.current && !filterRef.current.contains(e.target as Node)) setFilterOpen(false); };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [filterOpen]);

  // Bidirectional highlight: a sentence picked in the transcript scrolls its card into view
  // (expanding the list / clearing filters when it is hidden).
  useEffect(() => {
    if (!selectedSentenceId) { scrolledToRef.current = null; return; }
    if (scrolledToRef.current === selectedSentenceId) return; // already brought into view; new arrivals must not re-scroll
    const exists = checks.some((c) => c.sentence_id === selectedSentenceId);
    if (!exists) return;
    const inFiltered = filtered.some((c) => c.sentence_id === selectedSentenceId);
    if (!inFiltered) { setQuick('all'); setTagFilters(new Set()); setShowAll(true); return; }
    const inVisible = visible.some((c) => c.sentence_id === selectedSentenceId);
    if (!inVisible) { setShowAll(true); return; }
    const el = cardRefs.current.get(selectedSentenceId);
    if (el) {
      scrolledToRef.current = selectedSentenceId;
      requestAnimationFrame(() => el.scrollIntoView({ block: 'nearest', behavior: 'smooth' }));
    }
  }, [selectedSentenceId, checks, filtered, visible]);

  const isSelected = useCallback((c: SentenceCheck) =>
    selectedSentenceId != null ? c.sentence_id === selectedSentenceId : c.segment_id === selectedSegmentId,
  [selectedSentenceId, selectedSegmentId]);

  const handleCardClick = useCallback((card: SentenceCheck) => {
    if (onSelectSentence) onSelectSentence(card.sentence_id);
    onSelectSegment(card.segment_id);
    if (onOpenCard) onOpenCard(card); else setExpandedCard(card);
  }, [onSelectSegment, onSelectSentence, onOpenCard]);

  const toggleTag = (id: string) => setTagFilters((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  if (cards.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-6 gap-3">
        <div className={`w-11 h-11 rounded-2xl border flex items-center justify-center transition-all duration-500 ${
          isAnalyzing ? 'bg-cyan-500/10 border-cyan-500/30 animate-pulse' : 'bg-white/5 border-white/10'
        }`}>
          {isAnalyzing ? (
            <svg className="w-5 h-5 text-cyan-400 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          ) : (
            <svg className="w-5 h-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
        </div>
        {isAnalyzing ? (
          <p className="text-xs text-cyan-400/80 max-w-[200px] leading-relaxed animate-pulse">Checking statement against your documents…</p>
        ) : (
          <p className="text-xs text-slate-500 max-w-[230px] leading-relaxed">
            Verifiable claims are checked as they are spoken — supported, wrong, contract and policy matches — each with the quote that proves it.
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col min-h-0">
      {/* Quick filters + filter popover */}
      <div className="shrink-0 flex items-center gap-1.5 px-2.5 py-1.5">
        {!embedded && <span className="text-xs font-semibold text-slate-300 mr-1">Checks</span>}
        <div className="inline-flex items-center rounded-lg border border-white/10 bg-white/[0.04] p-0.5" role="group" aria-label="Filter checks">
          {QUICK.map((q) => (
            <button
              key={q.id}
              type="button"
              onClick={() => setQuick(q.id)}
              aria-pressed={quick === q.id}
              className={`px-2 py-1 text-[11px] font-semibold rounded-md transition-colors touch-manipulation ${
                quick === q.id
                  ? q.id === 'flagged' ? 'bg-rose-500/20 text-rose-200' : 'bg-white/15 text-white'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              {q.label}
            </button>
          ))}
        </div>
        {flaggedCount > 0 && <span className="text-[11px] text-rose-300/80 tabular-nums">{flaggedCount} flagged</span>}
        {isAnalyzing && (
          <svg className="w-3 h-3 text-cyan-400 animate-spin" fill="none" viewBox="0 0 24 24" aria-label="Checking…">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
          </svg>
        )}
        <div className="ml-auto relative" ref={filterRef}>
          <button
            type="button"
            onClick={() => setFilterOpen((v) => !v)}
            disabled={extraTags.length === 0}
            aria-expanded={filterOpen}
            className={`relative p-1.5 rounded-lg transition-colors touch-manipulation disabled:opacity-30 ${
              tagFilters.size > 0 ? 'text-cyan-300 bg-cyan-500/10' : 'text-slate-400 hover:text-white hover:bg-white/10'
            }`}
            title="More filters"
            aria-label="More filters"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth={1.75} viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" d="M3 5h18M6 12h12M10 19h4" /></svg>
            {tagFilters.size > 0 && <span className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full bg-cyan-400" />}
          </button>
          {filterOpen && (
            <div className="absolute right-0 top-full mt-1 z-50 w-56 rounded-xl border border-white/15 bg-slate-900 shadow-2xl p-2">
              <div className="flex items-center px-1 pb-1.5">
                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Show only</span>
                {tagFilters.size > 0 && (
                  <button type="button" onClick={() => setTagFilters(new Set())} className="ml-auto text-[10px] font-semibold text-slate-400 hover:text-white">Clear</button>
                )}
              </div>
              <ul className="max-h-64 overflow-auto space-y-0.5">
                {extraTags.map(({ id, n }) => {
                  const spec = tagSpecFor(vocab, id);
                  const on = tagFilters.has(id);
                  return (
                    <li key={id}>
                      <label className="flex items-center gap-2 px-1.5 py-1 rounded-md hover:bg-white/[0.06] cursor-pointer">
                        <input type="checkbox" checked={on} onChange={() => toggleTag(id)} className="accent-cyan-400 w-3.5 h-3.5" />
                        <span className={`text-xs ${on ? 'text-white' : 'text-slate-300'}`}>{spec.label}</span>
                        <span className="ml-auto text-[10px] text-slate-500 tabular-nums">{n}</span>
                      </label>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Cards */}
      <div className="flex-1 min-h-0 overflow-auto px-2 pb-2 space-y-1.5">
        {visible.length === 0 && (
          <div className="text-xs text-slate-500 text-center py-6">Nothing matches this filter.</div>
        )}
        {visible.map((card) => {
          const st = checkStyle(card, vocab);
          const selected = isSelected(card);
          const first = card.evidence?.[0];
          const src = first ? sourceLine(first.doc_title, first.page, undefined) : '';
          const conf = Number.isFinite(card.confidence) ? card.confidence : 0;
          const tags = card.tags?.length ? card.tags : [];
          return (
            <div
              key={card.id}
              ref={(el) => { if (el) cardRefs.current.set(card.sentence_id, el); else cardRefs.current.delete(card.sentence_id); }}
              onClick={() => handleCardClick(card)}
              className={`group rounded-xl border px-3 py-2.5 cursor-pointer transition-colors ${st.border} ${st.bg} ${
                selected ? 'ring-1 ring-white/30' : 'hover:border-white/20'
              }`}
            >
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className={`text-[13px] font-bold leading-none shrink-0 ${st.text}`}>{checkGlyph(card)}</span>
                {tags.length ? tags.slice(0, 2).map((t) => <TagChip key={t.tag} tag={t} vocab={vocab} />) : (
                  <span className={`text-[10px] font-semibold uppercase tracking-wider ${st.text}`}>{card.label}</span>
                )}
                {tags.length > 2 && <span className="text-[10px] text-slate-500">+{tags.length - 2}</span>}
                <span className="ml-auto text-[11px] text-slate-500 tabular-nums whitespace-nowrap">
                  {card.role ? `${roleLabel(card.role)} · ` : ''}{conf.toFixed(0)}%
                </span>
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); if (speaking) stop(); else speakCard(card); }}
                  className="p-1 -mr-1 rounded-md text-slate-500 hover:text-slate-200 hover:bg-white/10 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity"
                  title={speaking ? 'Stop reading' : 'Read this check out loud'}
                  aria-label="Read out"
                >
                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
                </button>
              </div>
              <p className="mt-1.5 text-[12.5px] text-slate-100/90 leading-relaxed line-clamp-2">“{card.sentence_text || card.segment_text}”</p>
              {first ? (
                <div className="mt-1.5">
                  <ProofPopover evidence={first} note={card.explanation} trigger="hover">
                    <blockquote className={`border-l-2 ${first.kind === 'rule' ? 'border-orange-400/60' : 'border-white/20'} pl-2 text-[11.5px] leading-snug text-slate-300 italic line-clamp-2`}>
                      “{first.quote}”
                    </blockquote>
                  </ProofPopover>
                  <div className="mt-1 flex items-center gap-1.5 text-[10.5px] text-slate-500">
                    {first.kind === 'rule' ? (
                      <span className="text-[9px] font-bold uppercase tracking-widest text-orange-300/90">Rule</span>
                    ) : (
                      <svg className="w-3 h-3 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/></svg>
                    )}
                    <span className="truncate">{first.kind === 'rule' ? (first.rule_id ?? 'domain rule') : (src || 'knowledge base')}</span>
                    {card.evidence.length > 1 && <span className="shrink-0">· +{card.evidence.length - 1} more</span>}
                  </div>
                </div>
              ) : card.source_chunks.length > 0 ? (
                <div className="mt-1 text-[10.5px] text-slate-500">{card.source_chunks.length} source{card.source_chunks.length > 1 ? 's' : ''} · click to view</div>
              ) : null}
            </div>
          );
        })}

        {/* Expander — carries the count */}
        {(hiddenCount > 0 || showAll) && filtered.length > RECENT_LIMIT && (
          <button
            type="button"
            onClick={() => setShowAll((v) => !v)}
            className="w-full mt-1 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-xs font-semibold text-slate-300 hover:text-white hover:bg-white/[0.07] transition-colors touch-manipulation"
          >
            {showAll ? `Show recent ${RECENT_LIMIT}` : `View all ${filtered.length}`}
          </button>
        )}
      </div>

      {/* Expanded card modal (built-in; parents may take over via onOpenCard) */}
      {expandedCard && !onOpenCard && (
        <AnalysisCardModal card={expandedCard} vocab={vocab} onClose={() => setExpandedCard(null)} />
      )}
    </div>
  );
};

export default AnalysisPanel;
