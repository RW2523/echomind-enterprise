/**
 * Transcript renderer — speaker blocks.
 *
 *   Lawyer · 10:14:32
 *   Utterance text… (consecutive segments from the same role are merged into one block)
 *
 * Inside a block every sentence is its own span. The ONLY highlighting:
 *   • the sentence currently being checked -> pulsing cyan left rule (while `analysis_start` is pending)
 *   • flagged sentences -> thin coloured left rule (red = contradicted, dark red = violating, orange = risk)
 * No chips, underlines or checkmarks. Clicking a sentence selects it (its details show in the right panel).
 * Partial (in-progress) text renders muted inside the current block.
 */
import React, { useMemo } from 'react';
import type { CheckStatus, Role, SentenceCheck, TranscriptSegment } from '../types';
import { flagTone, formatClock, speakerLabel, type FlagTone } from '../utils/silentAssistant';

// ── Block model ───────────────────────────────────────────────────────────────

export interface SentenceUnit {
  /** sentence_id (v2) or paragraph_id (v1 segments without sentences) — the key used by checks/sentenceStatus */
  key: string;
  segmentId: string;
  /** Verbatim text between the previous unit and this one (whitespace / punctuation gaps). */
  lead: string;
  text: string;
  role: Role | null;
}

export interface SpeakerBlock {
  key: string;
  role: Role | null;
  /** Wall-clock (epoch ms) when the first utterance of the block started. */
  startedAt?: number;
  units: SentenceUnit[];
}

function segmentStartedAt(seg: TranscriptSegment): number | undefined {
  if (typeof seg.started_at === 'number') return seg.started_at;
  // Server timing is only trusted when it is unmistakably an epoch timestamp (ms since 1970).
  if (typeof seg.start_ms === 'number' && seg.start_ms > 1e12) return seg.start_ms;
  return undefined;
}

/** Merge consecutive same-role sentences (across segments) into speaker blocks. */
export function buildSpeakerBlocks(segments: TranscriptSegment[]): SpeakerBlock[] {
  const blocks: SpeakerBlock[] = [];
  const push = (unit: SentenceUnit, startedAt: number | undefined, firstOfSegment: boolean) => {
    const last = blocks[blocks.length - 1];
    if (last && last.role === unit.role) {
      // Joining a new segment onto an existing block: make sure the texts do not run together.
      if (firstOfSegment && !unit.lead) unit.lead = ' ';
      last.units.push(unit);
      return;
    }
    if (!last) unit.lead = unit.lead.trimStart();
    blocks.push({ key: unit.key, role: unit.role, startedAt, units: [unit] });
  };

  for (const seg of segments) {
    const text = seg.text ?? '';
    const at = segmentStartedAt(seg);
    const sentences = seg.sentences;
    if (sentences && sentences.length > 0) {
      let cursor = 0;
      sentences.forEach((s, i) => {
        const cs = s.char_start, ce = s.char_end;
        const valid = Number.isFinite(cs) && Number.isFinite(ce) && cs >= 0 && ce > cs && ce <= text.length && cs >= cursor;
        let lead = '';
        let sText = s.text;
        if (valid) { lead = text.slice(cursor, cs); sText = text.slice(cs, ce); cursor = ce; }
        else if (i > 0) lead = ' ';
        push({ key: s.sentence_id, segmentId: seg.paragraph_id, lead, text: sText, role: s.role ?? seg.role ?? null }, at, i === 0);
      });
      if (cursor < text.length) {
        const tail = text.slice(cursor);
        const last = blocks[blocks.length - 1];
        if (last && tail.trim()) last.units.push({ key: `${seg.paragraph_id}:tail`, segmentId: seg.paragraph_id, lead: '', text: tail, role: last.role });
      }
    } else if (text.trim()) {
      push({ key: seg.paragraph_id, segmentId: seg.paragraph_id, lead: '', text, role: seg.role ?? null }, at, true);
    }
  }
  return blocks;
}

/** Plain-text export: "Lawyer [10:14:32]: …" per block. */
export function blocksToText(blocks: SpeakerBlock[]): string {
  return blocks
    .map((b) => {
      const when = b.startedAt ? ` [${formatClock(b.startedAt)}]` : '';
      const text = b.units.map((u) => u.lead + u.text).join('').trim();
      return `${speakerLabel(b.role)}${when}: ${text}`;
    })
    .join('\n\n');
}

// ── Styling helpers ───────────────────────────────────────────────────────────

const RULE: Record<FlagTone, string> = {
  red: 'border-rose-500/80 bg-rose-500/[0.06]',
  darkred: 'border-red-800 bg-red-900/[0.14]',
  orange: 'border-orange-400/80 bg-orange-500/[0.06]',
};

/** Classes for one sentence span: neutral unless being checked (pulsing) or flagged (coloured rule). */
export function sentenceMarkClass(check?: SentenceCheck, status?: CheckStatus, selected?: boolean): string {
  const parts = ['rounded-sm transition-colors'];
  const flag = check ? flagTone(check) : null;
  if (flag) parts.push(`sentence-rule border-l-2 pl-1.5 ${RULE[flag]}`);
  else if (status === 'pending' && !check) parts.push('sentence-rule border-l-2 pl-1.5 border-cyan-400/60 animate-checking');
  if (check) parts.push('cursor-pointer hover:bg-white/[0.05]');
  if (selected) parts.push('bg-white/[0.08] ring-1 ring-white/20');
  return parts.join(' ');
}

/** Speaker label colour: "me" role cyan, "other" violet, unknown slate. */
export function roleTextClass(role: Role | null | undefined, roles?: { me: Role; other: Role }): string {
  if (!role) return 'text-slate-400';
  if (roles && role === roles.me) return 'text-cyan-300';
  if (roles && role === roles.other) return 'text-violet-300';
  return 'text-slate-300';
}

// ── Component ─────────────────────────────────────────────────────────────────

interface TranscriptBlocksProps {
  segments: TranscriptSegment[];
  checks: Record<string, SentenceCheck>;
  sentenceStatus?: Record<string, CheckStatus>;
  roles?: { me: Role; other: Role };
  selectedSentenceId: string | null;
  onSelectSentence: (id: string, check?: SentenceCheck) => void;
  /** Live partial (in-progress) text, rendered muted inside the current block. */
  partial?: string;
  /** Role the partial belongs to (the currently selected speaker); null/undefined = unknown. */
  partialRole?: Role | null;
  /** Show the HH:MM:SS start time in block headers (default true; times need `started_at`). */
  showTimes?: boolean;
  className?: string;
}

const BlockHeader: React.FC<{ role: Role | null; startedAt?: number; roles?: { me: Role; other: Role }; showTimes: boolean }> = ({ role, startedAt, roles, showTimes }) => (
  <div className="flex items-baseline gap-2 mb-1 select-none">
    <span className={`text-[12px] font-semibold tracking-wide ${roleTextClass(role, roles)}`}>{speakerLabel(role)}</span>
    {showTimes && startedAt != null && (
      <span className="text-[11px] text-slate-500 tabular-nums">· {formatClock(startedAt)}</span>
    )}
  </div>
);

export const TranscriptBlocks: React.FC<TranscriptBlocksProps> = ({
  segments, checks, sentenceStatus = {}, roles, selectedSentenceId, onSelectSentence, partial, partialRole, showTimes = true, className = '',
}) => {
  const blocks = useMemo(() => buildSpeakerBlocks(segments), [segments]);
  const partialText = (partial ?? '').trim();
  const last = blocks[blocks.length - 1];
  // Keep the live text flowing inside the last block. A detached block appears only when the
  // speaker has genuinely changed — i.e. the last block carries a role and it differs from the
  // current one. Previously any role mismatch split it off, so selecting a speaker mid-session
  // tore the live text away from the sentence it was continuing, mid-clause.
  const partialInLast =
    !!partialText && !!last &&
    (partialRole === undefined || last.role == null || last.role === (partialRole ?? null));
  const partialAsNewBlock = !!partialText && !partialInLast;

  return (
    <div className={`space-y-4 ${className}`}>
      {blocks.map((b, bi) => {
        const isLast = bi === blocks.length - 1;
        return (
          <section key={b.key} data-block-role={b.role ?? 'unknown'}>
            <BlockHeader role={b.role} startedAt={b.startedAt} roles={roles} showTimes={showTimes} />
            <p className="text-[13.5px] leading-7 text-slate-100/90 break-words">
              {b.units.map((u) => {
                const check = checks[u.key];
                const status = sentenceStatus[u.key];
                const selected = selectedSentenceId === u.key;
                const cls = sentenceMarkClass(check, status, selected);
                return (
                  <React.Fragment key={u.key}>
                    {u.lead}
                    <span
                      data-sentence-id={u.key}
                      onClick={check ? (e) => { e.stopPropagation(); onSelectSentence(u.key, check); } : undefined}
                      className={cls}
                      title={check ? `${check.tags?.map((t) => t.label ?? t.tag).join(', ') || check.label} · ${Math.round(check.confidence)}%` : undefined}
                    >
                      {u.text}
                    </span>
                  </React.Fragment>
                );
              })}
              {isLast && partialInLast && <span className="text-slate-400/80"> {partialText}</span>}
            </p>
          </section>
        );
      })}
      {partialAsNewBlock && (
        <section data-block-role={partialRole ?? 'unknown'}>
          <BlockHeader role={partialRole ?? null} roles={roles} showTimes={false} />
          <p className="text-[13.5px] leading-7 text-slate-400/80 break-words">{partialText}</p>
        </section>
      )}
    </div>
  );
};

export default TranscriptBlocks;
