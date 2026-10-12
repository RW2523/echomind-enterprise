import React from 'react';
import { Languages } from 'lucide-react';
import { StatusPill } from './primitives';

/** Machine translation (OPT-02, D-070): the languages offered, the label every translation carries, and its checks. */
export const LANGUAGES: [string, string][] = [
  ['en', 'English'], ['fr', 'French'], ['de', 'German'], ['es', 'Spanish'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'],
  ['pl', 'Polish'], ['zh', 'Chinese (Simplified)'], ['ja', 'Japanese'], ['ko', 'Korean'], ['ar', 'Arabic'],
];
export const RTL = new Set(['ar']);

export function TranslationLabel({ text }: { text: string }) {
  return <p role="note" className="lw-translate-label">{text}</p>;
}

export function TranslationChecks({ figures, withheld }: { figures: string[]; withheld?: boolean }) {
  if (!figures.length && !withheld) return null;
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {figures.length > 0 && (
        <StatusPill tone="attention">{`Check: ${figures.join(', ')} ${figures.length === 1 ? 'is' : 'are'} not in the translation`}</StatusPill>
      )}
      {withheld && <StatusPill tone="neutral">Text addressed to an AI was not sent for translation</StatusPill>}
    </div>
  );
}

export const TranslateIcon = Languages;
