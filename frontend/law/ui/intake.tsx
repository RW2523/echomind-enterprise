import React, { useState } from 'react';
import { ArrowLeft, ArrowRight, Send } from 'lucide-react';
import type { IntakeAnswers, IntakeQuestion, IntakeScreening, IntakeScript, IntakeStatus } from '../api/types';
import { StatusPill, type Tone } from './primitives';

/** Shared by the staff intake (S4.1, D-054) and the enquirer's link page: the fixed script, never a model. */

export const emptyAnswers = (script: IntakeScript): IntakeAnswers =>
  Object.fromEntries(script.questions.map((q) => [q.key, q.kind === 'names' ? [] : '']));

const asText = (v: string | string[] | undefined) => (Array.isArray(v) ? v.join('\n') : v ?? '');

function shown(q: IntakeQuestion, v: string | string[] | undefined): string {
  if (q.kind === 'choice') return (q.choices ?? {})[asText(v)] ?? '';
  if (q.kind === 'names') return (Array.isArray(v) ? v : []).join('; ');
  return asText(v);
}

/** A first check in the browser; the server checks again and has the last word. */
export function problem(q: IntakeQuestion, v: string | string[] | undefined, all: IntakeAnswers): string {
  const text = asText(v).trim();
  if (q.required && !text) return `${q.label}: this is needed.`;
  if (q.kind === 'email' && text && !/^[^@\s]+@[^@\s]+\.[^@\s.]{2,}$/.test(text)) return 'Email: this does not look like an email address.';
  if (q.key === 'phone' && !asText(all.email).trim() && !text) return 'Give an email address or a telephone number.';
  if (q.max && q.kind !== 'names' && text.length > q.max) return `${q.label}: keep it to ${q.max} characters.`;
  if (q.kind === 'names' && Array.isArray(v) && q.max_items && v.length > q.max_items) return `${q.label}: give up to ${q.max_items} names.`;
  return '';
}

function Input({ q, value, onChange, autoFocus }: { q: IntakeQuestion; value: string | string[]; onChange: (v: string | string[]) => void; autoFocus?: boolean }) {
  const id = `intake-${q.key}`;
  if (q.kind === 'choice') {
    return (
      <select id={id} className="lw-input" value={asText(value)} autoFocus={autoFocus} onChange={(e) => onChange(e.target.value)}>
        <option value="">{q.required ? 'Choose one' : 'Prefer not to say'}</option>
        {Object.entries(q.choices ?? {}).map(([k, label]) => <option key={k} value={k}>{label}</option>)}
      </select>
    );
  }
  if (q.kind === 'names') {
    return <textarea id={id} className="lw-input" rows={3} autoFocus={autoFocus} value={asText(value)}
      onChange={(e) => onChange(e.target.value.split('\n'))} placeholder="One name per line" />;
  }
  if (q.kind === 'longtext') {
    return <textarea id={id} className="lw-input" rows={4} maxLength={q.max} autoFocus={autoFocus} value={asText(value)} onChange={(e) => onChange(e.target.value)} />;
  }
  return <input id={id} className="lw-input" autoFocus={autoFocus} maxLength={q.max}
    type={q.kind === 'email' ? 'email' : q.kind === 'phone' ? 'tel' : q.kind === 'date' ? 'date' : 'text'}
    autoComplete={q.kind === 'email' ? 'email' : q.kind === 'phone' ? 'tel' : q.key === 'your_name' ? 'name' : q.key === 'organisation' ? 'organization' : 'off'}
    value={asText(value)} onChange={(e) => onChange(e.target.value)} />;
}

const clean = (a: IntakeAnswers): IntakeAnswers =>
  Object.fromEntries(Object.entries(a).map(([k, v]) => [k, Array.isArray(v) ? v.map((x) => x.trim()).filter(Boolean) : v]));

/** Staff on the telephone or at the front desk: every question, with the words to say. */
export function ScriptForm({ script, onSubmit, busy }: { script: IntakeScript; onSubmit: (a: IntakeAnswers) => void; busy: boolean }) {
  const [answers, setAnswers] = useState<IntakeAnswers>(() => emptyAnswers(script));
  const [error, setError] = useState('');
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const first = script.questions.map((q) => problem(q, answers[q.key], answers)).find(Boolean);
    if (first) { setError(first); return; }
    setError('');
    onSubmit(clean(answers));
  };
  return (
    <form onSubmit={submit} className="lw-intake-form" aria-label="Intake questions">
      {script.questions.map((q) => (
        <div key={q.key} className="lw-intake-q">
          <label htmlFor={`intake-${q.key}`}><span className="lw-intake-say">Ask: “{q.ask}”</span><b>{q.label}{q.required ? '' : ' (optional)'}</b></label>
          <Input q={q} value={answers[q.key]} onChange={(v) => setAnswers((a) => ({ ...a, [q.key]: v }))} />
        </div>
      ))}
      {error && <p role="alert" className="lw-intake-error">{error}</p>}
      <div><button type="submit" className="lw-btn lw-btn-primary" disabled={busy}><Send />Submit answers and screen the names</button></div>
    </form>
  );
}

/** The enquirer's own page: one question at a time, each followed by the script's acknowledgement. */
export function Conversation({ script, onSubmit, busy, error }: { script: IntakeScript; onSubmit: (a: IntakeAnswers) => void; busy: boolean; error: string }) {
  const [answers, setAnswers] = useState<IntakeAnswers>(() => emptyAnswers(script));
  const [step, setStep] = useState(0);
  const [hint, setHint] = useState('');
  const qs = script.questions;
  const done = step >= qs.length;
  const next = (e: React.FormEvent) => {
    e.preventDefault();
    const q = qs[step];
    const p = problem(q, answers[q.key], answers);
    if (p) { setHint(p); return; }
    setHint('');
    setStep(step + 1);
  };
  return (
    <div className="lw-intake-chat">
      <ol className="lw-intake-turns" aria-label="Your answers so far">
        {qs.slice(0, step).map((q) => (
          <li key={q.key}>
            <p className="lw-intake-ask">{q.ask}</p>
            <p className="lw-intake-answer">{shown(q, answers[q.key]) || <i>No answer</i>}</p>
            <p className="lw-intake-ack">{q.thanks}</p>
          </li>
        ))}
      </ol>
      {!done ? (
        <form onSubmit={next} className="lw-intake-current" aria-label={`Question ${step + 1} of ${qs.length}`}>
          <label htmlFor={`intake-${qs[step].key}`} className="lw-intake-ask">{qs[step].ask}{qs[step].required ? '' : ' (optional)'}</label>
          <Input q={qs[step]} autoFocus value={answers[qs[step].key]} onChange={(v) => setAnswers((a) => ({ ...a, [qs[step].key]: v }))} />
          {hint && <p role="alert" className="lw-intake-error">{hint}</p>}
          <div className="lw-intake-nav">
            <button type="button" className="lw-btn lw-btn-ghost" disabled={step === 0} onClick={() => { setHint(''); setStep(step - 1); }}><ArrowLeft />Back</button>
            <span className="lw-meta lw-tnum">{`${step + 1} of ${qs.length}`}</span>
            <button type="submit" className="lw-btn lw-btn-primary">Next<ArrowRight /></button>
          </div>
        </form>
      ) : (
        <div className="lw-intake-current">
          <p className="lw-intake-ask">Please check your answers, then send them. You cannot change them after sending.</p>
          {error && <p role="alert" className="lw-intake-error">{error}</p>}
          <div className="lw-intake-nav">
            <button type="button" className="lw-btn lw-btn-ghost" onClick={() => setStep(qs.length - 1)}><ArrowLeft />Back</button>
            <button type="button" className="lw-btn lw-btn-primary" disabled={busy} onClick={() => onSubmit(clean(answers))}><Send />{busy ? 'Sending…' : 'Send my answers'}</button>
          </div>
        </div>
      )}
    </div>
  );
}

export function AnswersList({ script, answers }: { script: IntakeScript; answers: IntakeAnswers }) {
  return (
    <dl className="lw-intake-answers">
      {script.questions.map((q) => (
        <React.Fragment key={q.key}><dt>{q.label}</dt><dd>{shown(q, answers[q.key]) || '—'}</dd></React.Fragment>
      ))}
    </dl>
  );
}

const STATUS: Record<IntakeStatus, [string, Tone]> = {
  invited: ['Link sent', 'neutral'], open: ['Taking details', 'accent'], submitted: ['Awaiting conflicts check', 'attention'],
  cleared: ['Cleared', 'verified'], conflict: ['Conflict', 'conflict'], converted: ['Matter opened', 'verified'], closed: ['Closed', 'neutral'],
  purged: ['Purged', 'neutral'],
};
export function IntakeStatusPill({ status }: { status: IntakeStatus }) {
  const [label, tone] = STATUS[status];
  return <StatusPill tone={tone}>{label}</StatusPill>;
}

/** What front-desk staff see of a screening: whether to refer, never what matched. */
export function ScreeningPill({ s }: { s: IntakeScreening }) {
  if (s.decision === 'cleared') return <StatusPill tone="verified">Cleared by the conflicts team</StatusPill>;
  if (s.decision === 'conflict') return <StatusPill tone="conflict">Conflict confirmed</StatusPill>;
  if (s.result === 'possible_conflict') return <StatusPill tone="conflict">Refer to the conflicts team</StatusPill>;
  if (s.result === 'no_match') return <StatusPill tone="neutral">No match found</StatusPill>;
  return <StatusPill tone="neutral">Not screened</StatusPill>;
}
