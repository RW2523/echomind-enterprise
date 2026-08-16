import React, { useState, useCallback, useRef, useMemo } from 'react';
import type { AnalysisCard, SentenceCheck, TagSpec } from '../types';
import AnalysisCardModal, { LABEL_CONFIG, checkGlyph, checkStyle } from './AnalysisCardModal';
import TagChip from './TagChip';
import ProofPopover from './ProofPopover';
import { speakText } from '../services/backend';
import { asCheck, checkTagIds, isProblem, roleLabel, sortChecksProblemsFirst, sourceLine, tagSpecFor } from '../utils/silentAssistant';

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
  /** Hide the "Analysis" title (when embedded in a tab strip) */
  embedded?: boolean;
}

// Browser TTS fallback
function browserSpeak(text: string) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.rate = 1.0;
  utt.pitch = 1.0;
  window.speechSynthesis.speak(utt);
}

async function playAudioB64(b64: string): Promise<void> {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const ctx = new AudioContext();
  const buf = await ctx.decodeAudioData(bytes.buffer);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start(0);
  return new Promise((r) => { src.onended = () => r(); });
}

/** TTS-ready sentence for a check: server `phrase` if present, else composed from label/explanation. */
export function cardPhrase(card: AnalysisCard | SentenceCheck): string {
  const c = asCheck(card);
  if (c.phrase && c.phrase.trim()) return c.phrase.trim();
  const tagLabel = c.tags?.[0]?.label ?? c.label;
  return `This statement is ${tagLabel}. "${c.sentence_text || c.segment_text}". ${c.explanation ?? ''}`.trim();
}

const AnalysisPanel: React.FC<AnalysisPanelProps> = ({
  cards, selectedSegmentId, onSelectSegment, analyzingSegmentIds, vocab, selectedSentenceId, onSelectSentence, onOpenCard, embedded,
}) => {
  const [expandedCard, setExpandedCard] = useState<SentenceCheck | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [handraiseDrop, setHandraiseDrop] = useState(false);
  const [filter, setFilter] = useState<string | null>(null); // tag id | 'problems' | null
  const dropRef = useRef<HTMLDivElement>(null);

  const isAnalyzing = (analyzingSegmentIds?.size ?? 0) > 0;

  const checks = useMemo(() => cards.map(asCheck), [cards]);
  const sorted = useMemo(() => sortChecksProblemsFirst(checks), [checks]);
  const problemCount = useMemo(() => checks.filter(isProblem).length, [checks]);

  // Tag filter chips: tags actually present, ordered by vocab order.
  const presentTags = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of checks) for (const id of new Set(checkTagIds(c))) counts.set(id, (counts.get(id) ?? 0) + 1);
    const order = (vocab ?? []).map((t) => t.id);
    return [...counts.entries()]
      .sort((a, b) => {
        const ia = order.indexOf(a[0]), ib = order.indexOf(b[0]);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
      })
      .map(([id, n]) => ({ id, n }));
  }, [checks, vocab]);

  const visible = useMemo(() => {
    if (!filter) return sorted;
    if (filter === 'problems') return sorted.filter(isProblem);
    return sorted.filter((c) => checkTagIds(c).includes(filter));
  }, [sorted, filter]);

  const isSelected = useCallback((c: SentenceCheck) =>
    selectedSentenceId != null ? c.sentence_id === selectedSentenceId : c.segment_id === selectedSegmentId,
  [selectedSentenceId, selectedSegmentId]);

  const handleCardClick = useCallback((card: SentenceCheck) => {
    if (onSelectSentence) onSelectSentence(card.sentence_id === selectedSentenceId ? null : card.sentence_id);
    onSelectSegment(card.segment_id === selectedSegmentId ? null : card.segment_id);
    if (onOpenCard) onOpenCard(card); else setExpandedCard(card);
  }, [selectedSegmentId, selectedSentenceId, onSelectSegment, onSelectSentence, onOpenCard]);

  const handleSpeakCard = useCallback(async (card: AnalysisCard) => {
    setSpeaking(true);
    setHandraiseDrop(false);
    const text = cardPhrase(card);
    try {
      const res = await speakText('card', { text });
      if (res.audio_b64) {
        await playAudioB64(res.audio_b64);
      } else {
        browserSpeak(res.text || text);
      }
    } catch (e) {
      browserSpeak(text);
    } finally {
      setSpeaking(false);
    }
  }, []);

  const handleSpeakAll = useCallback(async () => {
    if (!cards.length) return;
    setSpeaking(true);
    setHandraiseDrop(false);
    // Problems first, and prefer server phrases when we have them.
    const list = sortChecksProblemsFirst(checks);
    try {
      const res = await speakText('summary', { cards: list });
      if (res.audio_b64) {
        await playAudioB64(res.audio_b64);
      } else {
        browserSpeak(res.text || list.map(cardPhrase).join('. '));
      }
    } catch {
      browserSpeak('Here is the analysis summary. ' + list.map(cardPhrase).join('. '));
    } finally {
      setSpeaking(false);
    }
  }, [cards, checks]);

  const stopSpeaking = useCallback(() => {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    setSpeaking(false);
  }, []);

  if (cards.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-6 gap-3">
        <div className={`w-12 h-12 rounded-2xl border flex items-center justify-center transition-all duration-500 ${
          isAnalyzing
            ? 'bg-cyan-500/10 border-cyan-500/30 animate-pulse'
            : 'bg-white/5 border-white/10'
        }`}>
          {isAnalyzing ? (
            <svg className="w-6 h-6 text-cyan-400 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
          ) : (
            <svg className="w-6 h-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
        </div>
        <div className="text-sm font-medium text-slate-400">Silent Assistant</div>
        {isAnalyzing ? (
          <p className="text-xs text-cyan-400/80 max-w-[200px] leading-relaxed animate-pulse">
            Checking statement against your documents…
          </p>
        ) : (
          <p className="text-xs text-slate-500 max-w-[220px] leading-relaxed">
            Checks appear here as you speak: what's supported, what's wrong, contract and policy matches — each with the quote that proves it.
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col min-h-0">
      {/* Panel header with handraise controls */}
      <div className="shrink-0 flex items-center gap-2 px-3 py-2 border-b border-white/10 bg-black/10">
        {!embedded && <span className="text-xs font-semibold text-slate-300">Checks</span>}
        <span className="text-xs text-slate-500 bg-white/5 rounded-full px-2 py-0.5">{cards.length}</span>
        {problemCount > 0 && (
          <span className="text-[10px] font-semibold text-rose-300 bg-rose-500/15 border border-rose-500/30 rounded-full px-2 py-0.5">
            {problemCount} problem{problemCount === 1 ? '' : 's'}
          </span>
        )}
        {isAnalyzing && (
          <span className="flex items-center gap-1 text-[10px] text-cyan-400 animate-pulse">
            <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            Analyzing…
          </span>
        )}
        <div className="ml-auto relative" ref={dropRef}>
          <button
            type="button"
            onClick={() => setHandraiseDrop((v) => !v)}
            disabled={speaking}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-xs font-semibold transition-colors touch-manipulation ${
              speaking
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'bg-white/10 text-slate-300 hover:bg-white/15 border border-white/10'
            }`}
            title="Voice readout (Handraise)"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 11.5V14m0-2.5v-6a1.5 1.5 0 113 0m-3 6a1.5 1.5 0 00-3 0v2a7.5 7.5 0 0015 0v-5a1.5 1.5 0 00-3 0m-6-3V11m0-5.5v-1a1.5 1.5 0 013 0v1m0 0V11m0-5.5a1.5 1.5 0 013 0v3m0 0V11" />
            </svg>
            {speaking ? 'Speaking…' : 'Handraise'}
            {!speaking && <svg className="w-3 h-3 ml-0.5" viewBox="0 0 20 20" fill="currentColor"><path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"/></svg>}
          </button>

          {handraiseDrop && !speaking && (
            <div className="absolute right-0 top-full mt-1 z-50 w-56 rounded-xl border border-white/15 bg-slate-800 shadow-xl overflow-hidden">
              <button
                type="button"
                onClick={handleSpeakAll}
                className="w-full flex items-center gap-2.5 px-4 py-3 text-sm text-slate-200 hover:bg-white/10 text-left transition-colors"
              >
                <svg className="w-4 h-4 text-cyan-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/></svg>
                Speak Full Summary
              </button>
              {sorted.slice(0, 5).map((card) => {
                const st = checkStyle(card, vocab);
                return (
                  <button
                    key={card.id}
                    type="button"
                    onClick={() => handleSpeakCard(card)}
                    className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-slate-300 hover:bg-white/10 text-left transition-colors border-t border-white/5"
                  >
                    <span className={`text-xs font-bold ${st.text}`}>{checkGlyph(card)}</span>
                    <span className="truncate">{(card.sentence_text || card.segment_text).slice(0, 40)}…</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
        {speaking && (
          <button
            type="button"
            onClick={stopSpeaking}
            className="p-1.5 rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
            title="Stop speaking"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
          </button>
        )}
      </div>

      {/* Filter chips */}
      {(presentTags.length > 1 || problemCount > 0) && (
        <div className="shrink-0 flex items-center gap-1 px-2 py-1.5 overflow-x-auto border-b border-white/5 [scrollbar-width:none]">
          <button
            type="button"
            onClick={() => setFilter(null)}
            className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border whitespace-nowrap ${filter === null ? 'bg-white/15 text-white border-white/30' : 'text-slate-400 border-white/10 hover:bg-white/10'}`}
          >
            All {checks.length}
          </button>
          {problemCount > 0 && (
            <button
              type="button"
              onClick={() => setFilter(filter === 'problems' ? null : 'problems')}
              className={`text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border whitespace-nowrap ${filter === 'problems' ? 'bg-rose-500/25 text-rose-200 border-rose-400/50' : 'text-rose-300 border-rose-500/30 hover:bg-rose-500/10'}`}
            >
              Problems {problemCount}
            </button>
          )}
          {presentTags.map(({ id, n }) => (
            <TagChip
              key={id}
              tag={id}
              vocab={vocab}
              active={filter === id}
              onClick={() => setFilter(filter === id ? null : id)}
              title={`${tagSpecFor(vocab, id).label}: ${n}`}
              className={filter && filter !== id ? 'opacity-60' : ''}
            />
          ))}
        </div>
      )}

      {/* Cards list */}
      <div className="flex-1 min-h-0 overflow-auto p-2 space-y-2">
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
              onClick={() => handleCardClick(card)}
              className={`rounded-xl border cursor-pointer transition-all duration-200 overflow-hidden ${st.border} ${st.bg}
                ${selected ? 'ring-2 ring-offset-1 ring-offset-slate-900 ring-white/20 shadow-lg scale-[1.01]' : 'hover:scale-[1.005] hover:shadow-md'}`}
            >
              {/* Tags + role + confidence + mic */}
              <div className="flex items-center gap-1.5 px-3 pt-2.5 pb-1 flex-wrap">
                <span className={`text-sm font-bold shrink-0 ${st.text}`}>{checkGlyph(card)}</span>
                {tags.length ? tags.slice(0, 3).map((t) => <TagChip key={t.tag} tag={t} vocab={vocab} />) : (
                  <span className={`text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full border shrink-0 ${LABEL_CONFIG[card.label]?.badge ?? ''} ${st.border}`}>
                    {card.label}
                  </span>
                )}
                {tags.length > 3 && <span className="text-[10px] text-slate-400">+{tags.length - 3}</span>}
                <div className="ml-auto flex items-center gap-1 shrink-0">
                  {card.role && <span className="text-[10px] text-slate-400 font-medium">{roleLabel(card.role)}</span>}
                  <span className="text-[10px] text-slate-400 font-medium tabular-nums">{conf.toFixed(0)}%</span>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); handleSpeakCard(card); }}
                    className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-white/10 transition-colors"
                    title="Speak this card"
                  >
                    <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"/>
                    </svg>
                  </button>
                </div>
              </div>

              {/* Confidence bar (thin) */}
              <div className="mx-3 h-0.5 rounded-full bg-white/5 overflow-hidden mb-2">
                <div className={`h-full rounded-full ${st.bar} opacity-70`} style={{ width: `${Math.max(0, Math.min(100, conf))}%` }} />
              </div>

              {/* Sentence snippet */}
              <p className="px-3 pb-2 text-xs text-slate-200 leading-relaxed line-clamp-2">
                “{card.sentence_text || card.segment_text}”
              </p>

              {/* First evidence quote */}
              {first ? (
                <div className="px-3 pb-2.5">
                  <ProofPopover evidence={first} note={card.explanation} trigger="hover">
                    <blockquote className={`border-l-2 ${first.kind === 'rule' ? 'border-orange-400/60' : 'border-cyan-400/60'} pl-2 text-[11px] leading-snug text-white/80 italic line-clamp-2`}>
                      “{first.quote}”
                    </blockquote>
                  </ProofPopover>
                  <div className="mt-1 flex items-center gap-1.5 text-[10px] text-slate-500">
                    {first.kind === 'rule' ? (
                      <span className="text-[9px] font-bold uppercase tracking-widest px-1 py-px rounded bg-orange-500/20 text-orange-300 border border-orange-500/40">Rule</span>
                    ) : (
                      <svg className="w-3 h-3 text-slate-500 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/></svg>
                    )}
                    <span className="truncate">{first.kind === 'rule' ? (first.rule_id ?? 'domain rule') : (src || 'knowledge base')}</span>
                    {card.evidence.length > 1 && <span className="shrink-0">· +{card.evidence.length - 1} more</span>}
                  </div>
                </div>
              ) : card.source_chunks.length > 0 ? (
                <div className="px-3 pb-2 flex items-center gap-1">
                  <svg className="w-3 h-3 text-slate-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/></svg>
                  <span className="text-[10px] text-slate-500">{card.source_chunks.length} source{card.source_chunks.length > 1 ? 's' : ''} · click to view</span>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      {/* Expanded card modal (built-in; parents may take over via onOpenCard) */}
      {expandedCard && !onOpenCard && (
        <AnalysisCardModal
          card={expandedCard}
          vocab={vocab}
          onClose={() => setExpandedCard(null)}
        />
      )}
    </div>
  );
};

export default AnalysisPanel;
