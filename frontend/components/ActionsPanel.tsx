import React, { useMemo, useState } from 'react';
import type { ActionItem, SentenceCheck, TagSpec } from '../types';
import TagChip from './TagChip';
import { copyToClipboard, roleLabel } from '../utils/silentAssistant';

interface ActionsPanelProps {
  items: ActionItem[];
  vocab?: TagSpec[];
  selectedSentenceId?: string | null;
  onSelect?: (check: SentenceCheck) => void;
}

const GROUPS: { tag: string; title: string; icon: string }[] = [
  { tag: 'action-item', title: 'Action items', icon: '☑️' },
  { tag: 'commitment', title: 'Commitments', icon: '🤝' },
  { tag: 'decision', title: 'Decisions', icon: '⚖️' },
];

/** Action items / commitments / decisions captured from the conversation (who said what). */
const ActionsPanel: React.FC<ActionsPanelProps> = ({ items, vocab, selectedSentenceId, onSelect }) => {
  const [copied, setCopied] = useState(false);
  const grouped = useMemo(() => {
    const m = new Map<string, ActionItem[]>();
    for (const it of items) { if (!m.has(it.tag)) m.set(it.tag, []); m.get(it.tag)!.push(it); }
    return GROUPS.filter((g) => m.has(g.tag)).map((g) => ({ ...g, items: m.get(g.tag)! }))
      .concat([...m.keys()].filter((k) => !GROUPS.some((g) => g.tag === k)).map((k) => ({ tag: k, title: roleLabel(k), icon: '•', items: m.get(k)! })));
  }, [items]);

  const copyAll = async () => {
    const text = grouped
      .map((g) => `${g.title}\n${g.items.map((it) => `- ${it.role ? `[${roleLabel(it.role)}] ` : ''}${it.text}`).join('\n')}`)
      .join('\n\n');
    if (await copyToClipboard(text)) { setCopied(true); setTimeout(() => setCopied(false), 1200); }
  };

  if (!items.length) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-6 gap-2">
        <div className="w-12 h-12 rounded-2xl border border-white/10 bg-white/5 flex items-center justify-center text-xl">☑️</div>
        <div className="text-sm font-medium text-slate-400">No actions yet</div>
        <p className="text-xs text-slate-500 max-w-[220px] leading-relaxed">
          Promises, to-dos and decisions are captured here as they are spoken.
        </p>
      </div>
    );
  }

  return (
    <div className="p-2 space-y-3">
      <div className="flex items-center px-1">
        <span className="text-[10px] text-slate-500">{items.length} item{items.length === 1 ? '' : 's'}</span>
        <button type="button" onClick={copyAll} className="ml-auto text-[10px] font-semibold text-slate-300 hover:text-white px-2 py-1 rounded-md hover:bg-white/10">
          {copied ? 'Copied' : 'Copy all'}
        </button>
      </div>
      {grouped.map((g) => (
        <section key={g.tag}>
          <div className="flex items-center gap-1.5 px-1 mb-1.5">
            <span className="text-sm leading-none">{g.icon}</span>
            <span className="text-[10px] font-bold uppercase tracking-widest text-slate-300">{g.title}</span>
            <span className="text-[10px] text-slate-500 bg-white/5 rounded-full px-1.5">{g.items.length}</span>
          </div>
          <ul className="space-y-1.5">
            {g.items.map((it) => {
              const sel = selectedSentenceId === it.sentence_id;
              return (
                <li key={it.id}>
                  <button
                    type="button"
                    onClick={onSelect ? () => onSelect(it.check) : undefined}
                    className={`w-full text-left rounded-xl border p-2.5 transition-all ${sel ? 'border-amber-400/50 bg-amber-500/10 ring-1 ring-amber-400/30' : 'border-white/10 bg-white/5 hover:bg-white/[0.07]'}`}
                  >
                    <div className="flex items-center gap-1.5 flex-wrap mb-1">
                      <TagChip tag={it.tag} vocab={vocab} />
                      {it.role && (
                        <span className={`text-[10px] font-semibold ${it.role === it.check.role ? 'text-slate-300' : 'text-slate-400'}`}>
                          {roleLabel(it.role)}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-white/90 leading-relaxed">{it.text}</p>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
};

export default ActionsPanel;
