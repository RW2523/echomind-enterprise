import React from 'react';
import { Link } from 'react-router';
import { ShieldCheck } from 'lucide-react';
import { useMatters } from '../api/queries';
import { useSession } from '../app/session';
import Toolbar from '../ui/Toolbar';
import { EmptyState, MatterStatusPill, roleLabel, Skeleton, StatusPill } from '../ui/primitives';

export default function Home() {
  const me = useSession();
  const matters = useMatters();
  const open = (matters.data ?? []).filter((m) => m.status === 'open');
  return (
    <>
      <Toolbar title="Home" />
      <div className="lw-content" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(260px, 1fr)', alignItems: 'start' }}>
        <section className="lw-paper" aria-labelledby="home-matters">
          <div className="lw-card-h">
            <h2 className="lw-h2" id="home-matters">Your open matters</h2>
            <Link to="/matters" className="lw-btn lw-btn-sm">All matters</Link>
          </div>
          {matters.isPending ? <div style={{ padding: 20, display: 'grid', gap: 10 }}><Skeleton /><Skeleton /><Skeleton /></div>
            : open.length === 0 ? <p className="lw-meta" style={{ padding: 20, margin: 0, fontSize: 14 }}>You're not on any open matters. The lead on a matter can add you.</p>
              : (
                <table className="lw-table">
                  <tbody>
                    {open.map((m) => (
                      <tr key={m.id}>
                        <td style={{ width: '1%', whiteSpace: 'nowrap' }} className="lw-tnum"><Link to={`/m/${encodeURIComponent(m.number)}/overview`} style={{ color: 'var(--accent)', fontWeight: 600, textDecoration: 'none' }}>{m.number}</Link></td>
                        <td><div style={{ fontWeight: 600 }}>{m.title}</div><div className="lw-meta">{m.client.name}</div></td>
                        <td style={{ textAlign: 'right' }}><span className="lw-meta">{roleLabel(m.my_role)}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
        </section>
        <section className="lw-paper lw-sheet" aria-labelledby="home-account" style={{ display: 'grid', gap: 10 }}>
          <h2 className="lw-h2" id="home-account">Your account</h2>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 600 }}>{me.user.username}</span>
            {me.user.roles.map((r) => <StatusPill key={r} tone="neutral">{roleLabel(r)}</StatusPill>)}
          </div>
          {me.session.mfa_enrolled && <StatusPill tone="verified" icon={<ShieldCheck />}>Two-step sign-in on</StatusPill>}
          <Link to="/account/security" className="lw-btn lw-btn-ghost lw-btn-sm" style={{ justifySelf: 'start' }}>Sessions and password</Link>
        </section>
        {matters.data && matters.data.length === 0 && (
          <div style={{ gridColumn: '1 / -1' }}>
            <EmptyState title="No matters yet">Matters you're a member of will appear here.</EmptyState>
          </div>
        )}
      </div>
    </>
  );
}
