import React, { useState } from 'react';
import { Link2, Phone, Plus, UserRound } from 'lucide-react';
import { Link, useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '../api/client';
import { intakePath, useIntakes } from '../api/queries';
import type { IntakeView } from '../api/types';
import Toolbar from '../ui/Toolbar';
import { CopyField, Dialog, EmptyState, ErrorNote, Skeleton } from '../ui/primitives';
import { IntakeStatusPill, ScreeningPill } from '../ui/intake';

const CHANNEL = { phone: 'Telephone', in_person: 'In person', link: 'Online link' } as const;
const when = (iso: string) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** Enquiries from prospective clients (S4.1, D-054): taken by the front desk on the telephone or in person, or
 *  answered by the enquirer through a single-use link. Every name is screened; staff see only whether to refer. */
export default function Enquiries() {
  const q = useIntakes();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [error, setError] = useState('');
  const [link, setLink] = useState<{ url: string; expires: string; reference: string } | null>(null);
  const can = q.data?.can;
  const start = async (channel: 'phone' | 'in_person') => {
    setError('');
    try {
      const r = await api.post<IntakeView>(intakePath(), { channel });
      await qc.invalidateQueries({ queryKey: ['intakes'] });
      navigate(`/enquiries/${encodeURIComponent(r.intake.id)}`);
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The enquiry could not be started.'); }
  };
  const sendLink = async () => {
    setError('');
    try {
      const r = await api.post<IntakeView & { link: string; expires_at: string }>(`${intakePath()}/links`, {});
      setLink({ url: `${window.location.origin}${r.link}`, expires: r.expires_at, reference: r.intake.reference });
      await qc.invalidateQueries({ queryKey: ['intakes'] });
    } catch (e) { setError(e instanceof ApiError ? e.message : 'The link could not be created.'); }
  };
  return (
    <>
      <Toolbar title="Enquiries">
        {can?.manage && <button type="button" className="lw-btn lw-btn-sm" onClick={() => start('phone')}><Phone />Telephone enquiry</button>}
        {can?.manage && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={() => start('in_person')}><UserRound />In person</button>}
        {can?.manage && can.links && <button type="button" className="lw-btn lw-btn-sm lw-btn-ghost" onClick={sendLink}><Link2 />Send a link</button>}
      </Toolbar>
      <div className="lw-content">
        <p className="lw-meta" style={{ margin: 0 }}>Administrative details only, from a fixed script. Every name given is checked against the firm's clients and the parties on its matters. Front-desk staff see only whether to refer an enquiry to the conflicts team; the conflicts team sees the matches and decides. Replies go out once a lawyer approves them.</p>
        {error && <ErrorNote>{error}</ErrorNote>}
        {q.isPending ? <div className="lw-paper lw-sheet"><Skeleton /><Skeleton /></div>
          : q.isError ? <ErrorNote>{q.error instanceof ApiError && q.error.status === 403 ? 'Enquiries are for intake staff, lawyers, the conflicts team and records administrators.' : 'The enquiries could not be loaded.'}</ErrorNote>
            : q.data.intakes.length === 0 ? <EmptyState title="No enquiries yet" action={can?.manage ? <button type="button" className="lw-btn" onClick={() => start('phone')}><Plus />Take a telephone enquiry</button> : undefined}>New enquiries appear here.</EmptyState>
              : (
                <div className="lw-paper" style={{ overflowX: 'auto' }}>
                  <table className="lw-table" aria-label="Enquiries">
                    <thead><tr><th>Reference</th><th>Enquirer</th><th>How</th><th>Status</th><th>Screening</th><th>Received</th></tr></thead>
                    <tbody>
                      {q.data.intakes.map((i) => (
                        <tr key={i.id}>
                          <td className="lw-tnum"><Link to={`/enquiries/${encodeURIComponent(i.id)}`}>{i.reference}</Link></td>
                          <td>{i.name ?? <span className="lw-meta">Not answered yet</span>}{i.organisation && <div className="lw-meta">{i.organisation}</div>}</td>
                          <td>{CHANNEL[i.channel]}</td>
                          <td><IntakeStatusPill status={i.status} /></td>
                          <td><ScreeningPill s={i.screening} /></td>
                          <td className="lw-tnum">{when(i.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
      </div>
      {link && (
        <Dialog title={`Link for enquiry ${link.reference}`} onClose={() => setLink(null)}
          footer={<button type="button" className="lw-btn lw-btn-primary" onClick={() => setLink(null)}>Done</button>}>
          <p style={{ margin: '0 0 10px' }}>Send this link to the enquirer. It opens the disclaimer and the questions, works once, and expires on {new Date(link.expires).toLocaleDateString([], { dateStyle: 'long' })}. It is shown only now.</p>
          <CopyField label="Link" value={link.url} />
        </Dialog>
      )}
    </>
  );
}
