import React, { useState } from 'react';
import { Link } from 'react-router';
import { ExternalLink } from 'lucide-react';
import { ApiError } from '../api/client';
import { useRegulatory } from '../api/queries';
import type { RegulatoryUpdate } from '../api/types';
import Toolbar from '../ui/Toolbar';
import { EmptyState, ErrorNote, Skeleton, StatusPill } from '../ui/primitives';

export const longDate = (d: string) => new Date(`${d}T12:00:00Z`).toLocaleDateString([], { day: 'numeric', month: 'long', year: 'numeric' });
export const licenceLabel = (u: RegulatoryUpdate) => u.licence === 'OGL-3.0'
  ? `Real public law · Open Government Licence v3.0 · ${u.source_name}`
  : `Real public law · ${u.source_name} · public domain`;

/** The official page, in a new tab. Public law: titles and summaries only are kept here. */
export function OfficialTitle({ u }: { u: RegulatoryUpdate }) {
  return (
    <a href={u.url} target="_blank" rel="noopener noreferrer" className="lw-reg-title">
      {u.title}<ExternalLink aria-hidden /><span className="lw-sr-only"> (opens the official page)</span>
    </a>
  );
}

export function UpdateMeta({ u }: { u: RegulatoryUpdate }) {
  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
      <StatusPill tone="accent">{u.jurisdiction}</StatusPill>
      <span className="lw-meta">{u.kind} · {longDate(u.published)}</span>
    </div>
  );
}

/** Regulatory updates (S4.4, D-068): new UK legislation and US rules, fetched daily by this server's host from the
 *  official sources and matched to open matters by jurisdiction and practice area. The matters shown are only those
 *  the person may see. */
export default function Regulatory() {
  const [jurisdiction, setJurisdiction] = useState<'' | 'UK' | 'US'>('');
  const [mine, setMine] = useState(false);
  const q = useRegulatory(jurisdiction, mine);
  return (
    <>
      <Toolbar title="Regulatory updates" />
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>New UK legislation from legislation.gov.uk and US rules and proposed rules from the Federal Register, fetched each morning by this server's host from the official sources (this application itself never goes online). Each update is matched to open matters by their jurisdiction and practice area; the words that matched are shown on the matter. Open the official page for the text.</p>
        <section className="lw-paper lw-sheet" aria-label="Filter">
          <div style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
            <div className="lw-segmented" role="group" aria-label="Jurisdiction">
              {([['', 'All'], ['UK', 'United Kingdom'], ['US', 'United States']] as const).map(([v, label]) => (
                <button key={v} type="button" aria-pressed={jurisdiction === v} onClick={() => setJurisdiction(v)}>{label}</button>
              ))}
            </div>
            <label style={{ display: 'inline-flex', gap: 8, alignItems: 'center', fontSize: 14 }}>
              <input type="checkbox" checked={mine} onChange={(e) => setMine(e.target.checked)} />Only updates that may concern my matters
            </label>
            {q.data && <span className="lw-meta lw-tnum">{`${q.data.updates.length} since ${longDate(q.data.since)}`}</span>}
          </div>
        </section>
        {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /><Skeleton /></div>
          : q.isError ? <ErrorNote>{q.error instanceof ApiError ? q.error.message : 'The updates could not be loaded.'}</ErrorNote>
            : q.data.updates.length === 0 ? (
              <EmptyState title="No updates">{mine ? 'Nothing recent matches your matters.' : 'Nothing has been imported for these dates yet. Updates arrive each morning.'}</EmptyState>
            ) : (
              <ul className="lw-reg-list" aria-label="Regulatory updates">
                {q.data.updates.map((u) => (
                  <li key={u.id} className="lw-paper lw-sheet lw-reg-item">
                    <UpdateMeta u={u} />
                    <OfficialTitle u={u} />
                    {u.summary && <p className="lw-reg-summary">{u.summary}</p>}
                    {u.matters.length > 0 && (
                      <div className="lw-reg-matters">
                        <span className="lw-meta">May concern:</span>
                        <ul className="lw-chips" aria-label="Matters it may concern">
                          {u.matters.map((m) => (
                            <li key={m.id}>
                              <Link to={`/m/${encodeURIComponent(m.number)}/regulatory`} className="lw-chip" aria-label={`${m.number} ${m.title}`}>
                                <b>{m.number}</b> {m.title}{m.status !== 'new' ? ` · ${m.status === 'reviewed' ? 'reviewed' : 'not relevant'}` : ''}
                              </Link>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    <span className="lw-meta" style={{ fontSize: 12 }}>{licenceLabel(u)}</span>
                  </li>
                ))}
              </ul>
            )}
      </div>
    </>
  );
}
