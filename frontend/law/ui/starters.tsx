import React from 'react';

/** Starter questions for an empty Research or Conversation page (D-065): what lawyers most often ask of a matter. */
export const STARTERS = [
  'What is each party’s liability cap, and what is excluded?',
  'How can the agreement be terminated, and on what notice?',
  'What deadlines and notice periods apply?',
  'What are the payment terms?',
  'Which law governs, and how are disputes resolved?',
];

export function Starters({ onPick, label = 'Try asking' }: { onPick: (q: string) => void; label?: string }) {
  return (
    <div className="lw-starters" role="group" aria-label={label}>
      <span className="lw-meta" style={{ alignSelf: 'center' }}>{label}:</span>
      {STARTERS.map((q) => <button key={q} type="button" className="lw-starter" onClick={() => onPick(q)}>{q}</button>)}
    </div>
  );
}
