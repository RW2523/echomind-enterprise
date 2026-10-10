import React from 'react';
import { useEvaluations } from '../api/queries';
import type { EvaluationRun } from '../api/types';
import { hasFirmRole, useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Skeleton, StatusPill } from '../ui/primitives';
import NotAvailable from './NotAvailable';

const SUITE: Record<string, string> = {
  research: 'Research answers', compare: 'Compare and conflicts', assist: 'Silent Assist', minutes: 'Minutes',
  transcription: 'Transcription', injection: 'Prompt injection', latency: 'Latency', load: 'Load', gate: 'Gate',
  lawyer: 'Lawyer review', drafting: 'Drafting', playbook: 'Playbook checks', redline: 'Redlines (Word)', citations: 'Citations and quotes',
  definitions: 'Definitions and references', obligations: 'Obligations and deadlines', diligence: 'Due diligence', governance: 'Governance checks',
};
// The measures a reviewer looks at first, per suite; anything else is in the report file.
const HEADLINE: Record<string, string[]> = {
  gate: ['suites', 'failed'],
  compare: ['explanations.quote_verification_rate', 'explanations.changed_with_both_sides_quoted', 'scan.planted_recall', 'scan.precision', 'explanations.errors'],
  research: ['answer_accuracy', 'abstention_accuracy', 'citation_precision', 'quote_verification_rate', 't06_leaks', 'latency_total_p95_s'],
  assist: ['recall', 'critical_recall', 'precision', 'false_alerts', 'latency_max_s'],
  minutes: ['recall', 'action_owners', 'action_deadlines', 'verified', 'invented_specifics'],
  transcription: ['wer_normalised', 'entity_accuracy', 'negation_accuracy', 'speaker_accuracy'],
  playbook: ['accuracy', 'deviations_caught', 'passed_in_error'],
  redline: ['verified', 'missed', 'libreoffice_round_trips_kept'],
  citations: ['correct', 'missing_caught', 'false_missing', 'altered_caught', 'false_alarms', 'malformed_caught', 'wrongly_flagged', 'external_lookups'],
  definitions: ['caught', 'fixes_right', 'false_findings', 'controls_clean'],
  obligations: ['found', 'timing_in_its_words', 'dates_right', 'dates_guessed', 'extra', 'controls_clean'],
  diligence: ['missing_lists_right', 'issues_found', 'extra_issues', 'controls_clean'],
  governance: ['found', 'extra', 'controls_clean'],
};
const label = (k: string) => k.replace(/_/g, ' ').replace(/\bp(50|95)\b/, 'p$1').replace(/\bs$/, '(s)');

function leaves(o: unknown, prefix = ''): [string, string][] {
  if (o === null || typeof o !== 'object' || Array.isArray(o)) return [[prefix, Array.isArray(o) ? `${o.length} items` : String(o)]];
  return Object.entries(o as Record<string, unknown>).flatMap(([k, v]) => leaves(v, prefix ? `${prefix} · ${k}` : k));
}

const at = (o: unknown, path: string): unknown =>
  path.split('.').reduce<unknown>((v, k) => (v && typeof v === 'object' ? (v as Record<string, unknown>)[k] : undefined), o);

function headline(r: EvaluationRun): [string, string][] {
  const want = HEADLINE[r.suite];
  const picked = (want ?? []).filter((k) => at(r.summary, k) !== undefined).map((k): [string, string] => [label(k.split('.').join(' · ')), String(at(r.summary, k))]);
  if (picked.length) return picked;
  return leaves(r.summary).slice(0, 6).map(([k, v]) => [label(k), v]);
}

/** Evaluation runs (S2.8): every measurement of the system, with the code, models and dataset it ran on. */
export default function Evaluations() {
  const me = useSession();
  const allowed = hasFirmRole(me, 'firm_admin', 'managing_partner', 'records_admin');
  const runs = useEvaluations(allowed);
  if (!allowed) return <NotAvailable />;
  return (
    <>
      <Toolbar title="Evaluations" />
      <div className="lw-content">
        <section className="lw-paper" aria-labelledby="ev-h">
          <div className="lw-card-h"><h2 className="lw-h2" id="ev-h">Evaluation runs</h2></div>
          <p className="lw-meta" style={{ padding: '0 20px', margin: 0 }}>Fictional test sets, run on this system. Held-out sets were never used to tune it. Each run names the code, models and dataset it measured; its full report holds every item.</p>
          {runs.isPending ? <div style={{ padding: 20, display: 'grid', gap: 10 }}><Skeleton /><Skeleton /></div>
            : runs.isError ? <div style={{ padding: 20 }}><ErrorNote>The evaluation runs could not be loaded.</ErrorNote></div>
              : runs.data.length === 0 ? <div style={{ padding: 20 }}><EmptyState title="No evaluation runs recorded">Import reports with python -m app.legal.evaluations import.</EmptyState></div>
                : (
                  <table className="lw-table" aria-label="Evaluation runs">
                    <thead><tr><th scope="col">Suite</th><th scope="col">Measures</th><th scope="col">Code and models</th><th scope="col">Run</th></tr></thead>
                    <tbody>{runs.data.map((r) => (
                      <tr key={r.id}>
                        <td style={{ whiteSpace: 'nowrap' }}><div style={{ fontWeight: 600 }}>{SUITE[r.suite] ?? r.suite}</div>
                          <StatusPill tone={r.set === 'heldout' ? 'verified' : 'neutral'}>{r.set === 'heldout' ? 'Held-out' : r.set === 'golden' ? 'Tuning set' : r.set || '—'}</StatusPill></td>
                        <td><dl className="lw-eval-measures">{headline(r).map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="lw-tnum">{v}</dd></div>)}</dl></td>
                        <td className="lw-meta"><span className="lw-mono">{r.commit ?? '—'}</span><br />{Object.values(r.models).join(' · ') || '—'}</td>
                        <td className="lw-meta" style={{ whiteSpace: 'nowrap' }}>{r.run_at ? `${r.run_at.slice(0, 4)}-${r.run_at.slice(4, 6)}-${r.run_at.slice(6, 8)} ${r.run_at.slice(9, 11)}:${r.run_at.slice(11, 13)}` : '—'}<br /><span className="lw-mono" title={r.report}>{r.report_sha256.slice(0, 10)}…</span></td>
                      </tr>
                    ))}</tbody>
                  </table>
                )}
        </section>
      </div>
    </>
  );
}
