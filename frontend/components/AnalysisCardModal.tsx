import React, { useEffect, useRef } from 'react';
import type { AnalysisCard, AnalysisLabel } from '../types';
import { ICONS } from '../constants';

export const LABEL_CONFIG: Record<AnalysisLabel, { bg: string; border: string; text: string; badge: string; icon: string }> = {
  Supported:         { bg: 'bg-emerald-500/10', border: 'border-emerald-500/40', text: 'text-emerald-300', badge: 'bg-emerald-500/20 text-emerald-300', icon: '✓' },
  Contradicted:      { bg: 'bg-red-500/10',     border: 'border-red-500/40',     text: 'text-red-300',     badge: 'bg-red-500/20 text-red-300',     icon: '✗' },
  Unverified:        { bg: 'bg-yellow-500/10',  border: 'border-yellow-400/40',  text: 'text-yellow-300',  badge: 'bg-yellow-500/20 text-yellow-300',  icon: '?' },
  Violating:         { bg: 'bg-orange-600/10',  border: 'border-orange-500/40',  text: 'text-orange-300',  badge: 'bg-orange-500/20 text-orange-300',  icon: '⚠' },
  'Risky Statement': { bg: 'bg-amber-500/10',   border: 'border-amber-400/40',   text: 'text-amber-300',   badge: 'bg-amber-500/20 text-amber-300',   icon: '⚡' },
  Relevant:          { bg: 'bg-cyan-500/10',    border: 'border-cyan-500/40',    text: 'text-cyan-300',    badge: 'bg-cyan-500/20 text-cyan-300',    icon: 'ℹ' },
};

interface AnalysisCardModalProps {
  card: AnalysisCard;
  onClose: () => void;
}

const AnalysisCardModal: React.FC<AnalysisCardModalProps> = ({ card, onClose }) => {
  const cfg = LABEL_CONFIG[card.label] ?? LABEL_CONFIG['Unverified'];
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={(e) => { if (e.target === overlayRef.current) onClose(); }}
    >
      <div className="relative w-full max-w-2xl max-h-[90vh] flex flex-col rounded-2xl border border-white/10 bg-slate-900 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className={`shrink-0 flex items-center gap-3 px-5 py-4 border-b ${cfg.border} ${cfg.bg}`}>
          <span className={`text-xl font-bold ${cfg.text}`}>{cfg.icon}</span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className={`text-xs font-semibold uppercase tracking-wider px-2.5 py-1 rounded-full border ${cfg.badge} ${cfg.border}`}>
                {card.label}
              </span>
              <span className="text-xs text-slate-400">
                {card.confidence.toFixed(0)}% confidence
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 p-2 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
            aria-label="Close"
          >
            <ICONS.Close className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 min-h-0 overflow-auto p-5 space-y-5">
          {/* Spoken statement */}
          <div>
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Spoken Statement</div>
            <blockquote className={`rounded-xl border-l-4 ${cfg.border} pl-4 pr-3 py-3 ${cfg.bg} text-sm text-white/90 leading-relaxed italic`}>
              "{card.segment_text}"
            </blockquote>
          </div>

          {/* Confidence bar */}
          <div>
            <div className="flex justify-between text-xs text-slate-400 mb-1">
              <span>Confidence</span>
              <span className={cfg.text}>{card.confidence.toFixed(0)}%</span>
            </div>
            <div className="h-2 rounded-full bg-white/5 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${cfg.text.replace('text-', 'bg-').replace('-300', '-500')}`}
                style={{ width: `${card.confidence}%` }}
              />
            </div>
          </div>

          {/* Explanation */}
          <div>
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Why this label</div>
            <p className="text-sm text-slate-200 leading-relaxed">{card.explanation}</p>
          </div>

          {/* Source references */}
          {card.source_chunks.length > 0 && (
            <div>
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Reference Sources ({card.source_chunks.length})
              </div>
              <div className="space-y-3">
                {card.source_chunks.map((chunk, i) => (
                  <div
                    key={chunk.chunk_id || i}
                    className="rounded-xl border border-white/10 bg-white/5 p-4"
                  >
                    {chunk.doc_title && (
                      <div className="flex items-center gap-2 mb-2">
                        <ICONS.File className="w-4 h-4 text-cyan-400 shrink-0" />
                        <span className="text-xs font-medium text-cyan-400 truncate">{chunk.doc_title}</span>
                      </div>
                    )}
                    <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap">{chunk.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="shrink-0 px-5 py-3 border-t border-white/10 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-300 hover:bg-white/15 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisCardModal;
