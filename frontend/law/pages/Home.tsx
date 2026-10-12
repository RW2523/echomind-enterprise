import React from 'react';
import { Link } from 'react-router';
import { BookOpenText, CalendarClock, FilePen, Inbox, Mic, ShieldCheck } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useMatters, useReviews } from '../api/queries';
import { hasFirmRole, useSession } from '../app/session';
import { hueOf, initials } from '../app/nav';
import Toolbar from '../ui/Toolbar';
import { useOpenPalette, type PaletteAction } from '../ui/Palette';
import { EmptyState, roleLabel, Skeleton, StatusPill } from '../ui/primitives';

/**
 * Home (D-065): what to do next. A greeting, the work a person starts most often (each asks which matter, through
 * the command palette), counts, their open matters as cards, and what is waiting for their decision.
 */
const greeting = (d = new Date()) => (d.getHours() < 12 ? 'Good morning' : d.getHours() < 18 ? 'Good afternoon' : 'Good evening');

export default function Home() {
  const me = useSession();
  const matters = useMatters();
  const reviews = useReviews();
  const open = (matters.data ?? []).filter((m) => m.status === 'open');
  const leading = open.filter((m) => m.my_role === 'lead').length;
  const waiting = reviews.data?.waiting ?? [];
  const lawyer = hasFirmRole(me, 'managing_partner', 'attorney', 'paralegal', 'reviewer');
  const palette = useOpenPalette();
  const name = me.user.username.charAt(0).toUpperCase() + me.user.username.slice(1);

  const quick: { action?: PaletteAction; to?: string; label: string; about: string; icon: LucideIcon; hue: number }[] = [];
  if (lawyer) {
    quick.push({ action: 'research', label: 'Ask the documents', about: 'Answers with exact quotes', icon: BookOpenText, hue: 232 });
    if (me.features?.conversation) quick.push({ action: 'conversation', label: 'Talk to a matter', about: 'Speak; hear the answer', icon: Mic, hue: 262 });
    quick.push({ action: 'meetings', label: 'Record a meeting', about: 'Live transcript, Silent Assist', icon: CalendarClock, hue: 352 });
    quick.push({ action: 'drafting', label: 'Draft a document', about: 'From firm templates', icon: FilePen, hue: 278 });
  }
  if (me.features?.intake && hasFirmRole(me, 'intake', 'conflicts', 'attorney', 'managing_partner')) {
    quick.push({ to: '/enquiries', label: 'Take an enquiry', about: 'Script, screening, conflicts', icon: Inbox, hue: 168 });
  }
  const promptFor: Record<PaletteAction, string> = {
    research: 'Ask the documents of which matter?', conversation: 'Talk to which matter?',
    meetings: 'Record a meeting on which matter?', drafting: 'Draft for which matter?',
  };

  return (
    <>
      <Toolbar title="Home" />
      <div className="lw-content">
        <div className="lw-home">
          <section className="lw-glass lw-hero" aria-labelledby="home-hello">
            <div>
              <h2 id="home-hello">{`${greeting()}, ${name}`}</h2>
              <p className="lw-hero-sub">
                {new Date().toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'long' })}
                {waiting.length > 0 ? ` · ${waiting.length} ${waiting.length === 1 ? 'item is' : 'items are'} waiting for your review` : ''}
              </p>
            </div>
            {quick.length > 0 && (
              <div className="lw-quick" role="group" aria-label="Start something">
                {quick.map((qa) => {
                  const Icon = qa.icon;
                  const body = (
                    <>
                      <span className="lw-icon-badge" style={{ ['--h' as string]: qa.hue }}><Icon aria-hidden /></span>
                      <span><strong>{qa.label}</strong><span className="lw-meta">{qa.about}</span></span>
                    </>
                  );
                  return qa.to
                    ? <Link key={qa.label} to={qa.to} className="lw-quick-btn">{body}</Link>
                    : <button key={qa.label} type="button" className="lw-quick-btn"
                        onClick={() => palette({ action: qa.action, prompt: promptFor[qa.action!] })}>{body}</button>;
                })}
              </div>
            )}
          </section>

          <div className="lw-stats" role="group" aria-label="At a glance">
            <div className="lw-paper lw-stat"><span className="lw-meta">Open matters</span><b>{matters.isPending ? '–' : open.length}</b></div>
            <div className="lw-paper lw-stat"><span className="lw-meta">You lead</span><b>{matters.isPending ? '–' : leading}</b></div>
            <div className="lw-paper lw-stat"><span className="lw-meta">Waiting for your review</span><b>{reviews.isPending ? '–' : waiting.length}</b></div>
          </div>

          <section className="lw-paper" aria-labelledby="home-matters">
            <div className="lw-card-h">
              <h2 className="lw-h2" id="home-matters">Your open matters</h2>
              <Link to="/matters" className="lw-btn lw-btn-sm lw-btn-ghost">All matters</Link>
            </div>
            {matters.isPending ? <div style={{ padding: 20, display: 'grid', gap: 10 }}><Skeleton /><Skeleton /><Skeleton /></div>
              : open.length === 0 ? <p className="lw-meta" style={{ padding: 20, margin: 0, fontSize: 14 }}>You're not on any open matters. The lead on a matter can add you.</p>
                : (
                  <div className="lw-matter-grid">
                    {open.map((m) => (
                      <Link key={m.id} to={`/m/${encodeURIComponent(m.number)}/overview`} className="lw-matter-card" aria-label={`${m.number} ${m.title}`}>
                        <span className="lw-matter-card-h">
                          <span className="lw-avatar" style={{ ['--h' as string]: hueOf(m.client.name) }} aria-hidden>{initials(m.client.name)}</span>
                          <span style={{ minWidth: 0 }}>
                            <span className="lw-meta lw-tnum" style={{ display: 'block', fontWeight: 600, color: 'var(--accent)' }}>{m.number}</span>
                            <span className="lw-matter-card-title" style={{ display: 'block' }}>{m.title}</span>
                          </span>
                        </span>
                        <span className="lw-meta">{m.client.name}{m.practice_area ? ` · ${m.practice_area}` : ''}</span>
                        <span><StatusPill tone={m.my_role === 'lead' ? 'accent' : 'neutral'}>{roleLabel(m.my_role)}</StatusPill></span>
                      </Link>
                    ))}
                  </div>
                )}
          </section>

          <div style={{ display: 'grid', gap: 18, alignContent: 'start' }}>
            <section className="lw-paper" aria-labelledby="home-waiting">
              <div className="lw-card-h"><h2 className="lw-h2" id="home-waiting">Waiting for you</h2>
                <Link to="/reviews" className="lw-btn lw-btn-sm lw-btn-ghost">Reviews</Link></div>
              {reviews.isPending ? <div style={{ padding: 16 }}><Skeleton /></div>
                : waiting.length === 0 ? <p className="lw-meta" style={{ padding: '14px 20px', margin: 0 }}>Nothing is waiting for your decision.</p>
                  : (
                    <ul className="lw-waiting">
                      {waiting.slice(0, 6).map((r) => (
                        <li key={`${r.kind}:${r.id}`}>
                          <Link to={r.path}>
                            <span style={{ fontWeight: 600 }}>{r.title}</span>
                            <span className="lw-meta">{r.matter ? `${r.matter.number} · ` : ''}{r.by ? `from ${r.by}` : ''}</span>
                          </Link>
                        </li>
                      ))}
                    </ul>
                  )}
            </section>
            <section className="lw-paper lw-sheet" aria-labelledby="home-account" style={{ display: 'grid', gap: 10, padding: '18px 20px' }}>
              <h2 className="lw-h2" id="home-account">Your account</h2>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <span className="lw-avatar" style={{ ['--h' as string]: hueOf(me.user.username), width: 30, height: 30, fontSize: 11 }} aria-hidden>{initials(me.user.username)}</span>
                <span style={{ fontWeight: 600 }}>{me.user.username}</span>
                {me.user.roles.map((r) => <StatusPill key={r} tone="neutral">{roleLabel(r)}</StatusPill>)}
              </div>
              {me.session.mfa_enrolled && <div><StatusPill tone="verified" icon={<ShieldCheck />}>Two-step sign-in on</StatusPill></div>}
              <Link to="/account/security" className="lw-btn lw-btn-ghost lw-btn-sm" style={{ justifySelf: 'start' }}>Sessions and password</Link>
            </section>
          </div>

          {matters.data && matters.data.length === 0 && (
            <div style={{ gridColumn: '1 / -1' }}>
              <EmptyState title="No matters yet">Matters you're a member of will appear here.</EmptyState>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
