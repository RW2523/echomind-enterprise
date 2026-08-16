import React from 'react';
import type { CheckTag, TagSpec, TagTone } from '../types';
import { tagSpecFor, toneClasses } from '../utils/silentAssistant';

interface TagChipProps {
  /** Tag id (e.g. 'contradicted') or the inline {tag,label,tone,confidence} from a check. */
  tag: string | CheckTag;
  /** Session tag vocabulary (from `session.tag_vocab`); falls back to static defaults. */
  vocab?: TagSpec[];
  size?: 'xs' | 'sm';
  active?: boolean;
  onClick?: (e: React.MouseEvent) => void;
  showConfidence?: boolean;
  className?: string;
  title?: string;
}

/**
 * Data-driven tag chip: colours come from the tag's tone in the vocabulary
 * (green emerald, red rose, yellow amber outline, blue sky, violet, indigo, teal, orange, amber, grey slate, cyan).
 * Unknown tags render neutral.
 */
const TagChip: React.FC<TagChipProps> = ({ tag, vocab, size = 'xs', active, onClick, showConfidence, className = '', title }) => {
  const id = typeof tag === 'string' ? tag : tag.tag;
  const inline = typeof tag === 'string' ? undefined : { label: tag.label, tone: tag.tone as TagTone | undefined };
  const spec = tagSpecFor(vocab, id, inline);
  const tone = toneClasses(spec.tone);
  const conf = typeof tag === 'string' ? undefined : tag.confidence;
  const pad = size === 'sm' ? 'px-2.5 py-1 text-[11px]' : 'px-2 py-0.5 text-[10px]';
  const Comp: any = onClick ? 'button' : 'span';
  return (
    <Comp
      type={onClick ? 'button' : undefined}
      onClick={onClick}
      title={title ?? spec.description ?? spec.label}
      className={`inline-flex items-center gap-1 rounded-full border font-semibold uppercase tracking-wider whitespace-nowrap leading-none ${pad} ${tone.badge} ${
        onClick ? 'cursor-pointer hover:brightness-125 transition' : ''
      } ${active ? 'ring-1 ring-white/40' : ''} ${className}`}
    >
      {spec.label}
      {showConfidence && typeof conf === 'number' && (
        <span className="opacity-70 tabular-nums font-normal normal-case">{Math.round(conf <= 1 ? conf * 100 : conf)}%</span>
      )}
    </Comp>
  );
};

export default TagChip;
