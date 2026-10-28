/**
 * The EchoMind Legal shell (docs/echomind-legal/02, DESIGN.md): a rail with four destinations (Chat, Matters, Drafts, Activity)
 * and recent conversations — a matter's own tools live on its tabs; a header with the matter the chat is about, search
 * (the command palette), the privacy state, the real capture state and the person; the page; and a right-hand panel that
 * opens beside the conversation. Settings and the firm's administration pages live under the profile menu. Every existing
 * Law page keeps working inside this shell; a matter's own pages are reached from the tab strip above them.
 */
import React, { useEffect, useRef, useState } from 'react';
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Activity, Briefcase, ChevronDown, FilePen, KeyRound, Landmark, LogOut, Menu, MessageSquare, MessageSquarePlus, Mic, MicOff,
  Moon, PanelLeftClose, PanelLeftOpen, Search, ShieldCheck, Sun,
} from 'lucide-react';
import { api } from '../api/client';
import { useReviews } from '../api/queries';
import { useCapturing } from '../app/capture';
import { MatterProvider, useCurrentMatter } from '../app/matter';
import { firmEntries, initials, matterEntries } from '../app/nav';
import { setThemeMode } from '../app/preferences';
import { hasFirmRole, useSession } from '../app/session';
import { whereAnswersComeFrom } from './where';
import { isMac, PaletteProvider, useOpenPalette } from '../ui/Palette';
import { counselApi } from './api';
import { ChatMatterProvider, PanelProvider, useChatMatter, usePanel } from './context';
import MatterPicker, { Layer, usePopover } from './MatterPicker';
import SidePanel from './SidePanel';
import './counsel.css';

const RAIL_KEY = 'echomind.counsel.rail';
const ROLE_LABEL: Record<string, string> = {
  managing_partner: 'Managing partner', attorney: 'Attorney', paralegal: 'Paralegal', reviewer: 'Reviewer', firm_admin: 'Firm administrator',
  records_admin: 'Records', read_only: 'Read only', intake: 'Intake', conflicts: 'Conflicts',
};

// ── Header: privacy and capture state ──────────────────────────────────────────
function PrivacyStatus() {
  const { open, setOpen, button, pop, style } = usePopover();
  const where = whereAnswersComeFrom(useSession());
  return (
    <>
      <button ref={button} type="button" className="qc-status" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        aria-label={`${where.label}: about where answers come from`} data-state={where.external ? 'external' : undefined}>
        <ShieldCheck aria-hidden /><span className="qc-narrow-sr">{where.label}</span>
      </button>
      {open && (<Layer>
        <div ref={pop} className="qc-pop" role="dialog" aria-label={where.title} style={{ ...style, padding: 14 }}>
          <p style={{ margin: '0 0 8px', fontWeight: 600 }}>{where.title}</p>
          <p className="qc-meta" style={{ margin: 0 }}>{where.text}</p>
        </div></Layer>)}
    </>
  );
}

function RecordingStatus() {
  const capturing = useCapturing();
  return (
    <span className="qc-status" aria-live="polite" data-state={capturing ? 'recording' : 'idle'} style={{ cursor: 'default' }}
      title={capturing ? 'The microphone is in use by EchoMind in this window.' : 'EchoMind is not using the microphone in this window.'}>
      <span className="qc-dot" aria-hidden />
      {capturing ? <Mic aria-hidden /> : <MicOff aria-hidden />}
      <span className="qc-narrow-sr">{capturing ? 'Recording' : 'Not recording'}</span>
    </span>
  );
}

// ── Header: profile menu ───────────────────────────────────────────────────────
function ProfileMenu() {
  const me = useSession();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const { open, setOpen, button, pop, style } = usePopover();
  const [dark, setDark] = useState(() => document.documentElement.dataset.theme === 'dark');
  const firm = firmEntries(me).filter((e) => !['chat', 'matters', 'drafts'].includes(e.key));
  const signOut = async () => {
    try { await api.post('/api/auth/logout'); } finally { qc.clear(); navigate('/signin', { replace: true }); }
  };
  const go = (to: string) => { setOpen(false); navigate(to); };
  return (
    <>
      <button ref={button} type="button" className="qc-me" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        aria-label={`Account and settings for ${me.user.username}`}>
        <span className="qc-avatar" aria-hidden>{initials(me.user.username)}</span>
        <span className="qc-me-text qc-hide-medium" aria-hidden>
          <b>{me.user.username}</b><small>{ROLE_LABEL[me.user.roles[0] ?? me.user.role] ?? 'Member'}</small>
        </span>
      </button>
      {open && (<Layer>
        <div ref={pop} className="qc-pop" role="menu" aria-label="Account and settings" style={style}>
          <div className="qc-pop-label">{me.user.username}</div>
          <button type="button" role="menuitem" className="qc-pop-item" onClick={() => go('/account/security')}><KeyRound aria-hidden />Security</button>
          <button type="button" role="menuitem" className="qc-pop-item" onClick={() => { setThemeMode(dark ? 'light' : 'dark'); setDark(!dark); }}>
            {dark ? <Sun aria-hidden /> : <Moon aria-hidden />}{dark ? 'Light appearance' : 'Dark appearance'}
          </button>
          {firm.length > 0 && <><div className="qc-pop-sep" /><div className="qc-pop-label">Firm</div></>}
          {firm.map((e) => (
            <button key={e.key} type="button" role="menuitem" className="qc-pop-item" onClick={() => go(e.to)}><e.icon aria-hidden />{e.label}</button>
          ))}
          <div className="qc-pop-sep" />
          <button type="button" role="menuitem" className="qc-pop-item" onClick={signOut}><LogOut aria-hidden />Sign out</button>
        </div></Layer>)}
    </>
  );
}

// ── Rail ───────────────────────────────────────────────────────────────────────
function History() {
  const q = useQuery({ queryKey: ['counsel', 'conversations'], retry: false, staleTime: 10_000, queryFn: () => counselApi.conversations() });
  const items = q.data?.conversations ?? [];
  if (!items.length) return null;
  return (
    <>
      <div className="qc-rail-label qc-hide-collapsed">Recent</div>
      <nav className="qc-history qc-hide-collapsed" aria-label="Recent conversations">
        {items.slice(0, 8).map((c) => (
          <NavLink key={c.id} to={`/chat/${c.id}`} className={({ isActive }) => `qc-history-item${isActive ? ' active' : ''}`}>
            <span>{c.title}</span><small>{(c.matters?.length ?? 0) > 1 ? c.matters!.map((m) => m.number).join(' + ') : c.matter.number}</small>
          </NavLink>
        ))}
      </nav>
    </>
  );
}

/** A rail link's name is its text, kept for screen readers when the rail is collapsed (visually hidden, never removed);
 *  `label` only where the name says more than the text (Drafts' count). `title` is the tooltip. */
function RailLink({ to, icon, children, end, label, title }: { to: string; icon: React.ReactNode; children: React.ReactNode; end?: boolean; label?: string; title?: string }) {
  return (
    <NavLink to={to} end={end} className="qc-rail-link" aria-label={label} title={title}>
      {icon}<span className="qc-rail-text">{children}</span>
    </NavLink>
  );
}

/** Drafts, with how many items wait for the person's decision (the reviews queue). */
function DraftsLink() {
  const n = useReviews().data?.count ?? 0;
  return (
    <RailLink to="/drafts" icon={<FilePen aria-hidden />} title="Drafts" label={n ? `Drafts, ${n} waiting for you` : undefined}>
      Drafts{n > 0 && <span className="qc-rail-count" aria-hidden="true">{n}</span>}
    </RailLink>
  );
}

/** The command palette (Ctrl/⌘ K) that the Law pages already provide: matters, pages and actions. */
function SearchButton() {
  const open = useOpenPalette();
  const shortcut = isMac() ? '⌘K' : 'Ctrl K';
  return (
    <>
      <button type="button" className="qc-search qc-hide-small" onClick={() => open()} aria-label={`Search or jump to (${shortcut})`}>
        <Search aria-hidden /><span>Find a matter, page or action…</span><kbd className="qc-kbd">{shortcut}</kbd>
      </button>
      <button type="button" className="qc-icon-btn qc-only-small" onClick={() => open()} aria-label="Find a matter, page or action"><Search aria-hidden /></button>
    </>
  );
}

/** A matter's own pages, above them (they used to fill the sidebar). */
const MAIN_TABS = new Set(['overview', 'documents', 'research', 'drafting', 'obligations', 'meetings']);

function MatterTabs() {
  const me = useSession();
  const matter = useCurrentMatter();
  const { setMatterId } = useChatMatter();
  const navigate = useNavigate();
  const location = useLocation();
  const moreRef = useRef<HTMLDetailsElement>(null);
  useEffect(() => { moreRef.current?.removeAttribute('open'); }, [location.pathname]);
  if (!matter) return null;
  // Six everyday tabs; the rest under More (review M1). The page you are on is always shown as a tab.
  const all = matterEntries(me, matter);
  const main = all.filter((e) => MAIN_TABS.has(e.key));
  const more = all.filter((e) => !MAIN_TABS.has(e.key));
  const here = more.find((e) => location.pathname === e.to || location.pathname.startsWith(`${e.to}/`));
  return (
    <nav className="qc-tabs" aria-label="Matter">
      <button type="button" className="qc-tab" style={{ border: 'none', background: 'none', font: 'inherit', cursor: 'pointer' }}
        onClick={() => { setMatterId(matter.id); navigate('/chat'); }} aria-label="Chat about this matter">
        <MessageSquare aria-hidden /><span className="qc-hide-small">Chat about this matter</span><span className="qc-only-small">Chat</span></button>
      {main.map((e) => (
        <NavLink key={e.key} to={e.to} className={({ isActive }) => `qc-tab${isActive ? ' active' : ''}`}><e.icon aria-hidden />{e.label}</NavLink>
      ))}
      {here && <NavLink to={here.to} className="qc-tab active"><here.icon aria-hidden />{here.label}</NavLink>}
      {more.length > 0 && (
        <details className="qc-more" ref={moreRef}>
          <summary className="qc-tab">More<ChevronDown aria-hidden /></summary>
          <div className="qc-more-pop" role="group" aria-label="More of this matter">
            {more.filter((e) => e !== here).map((e) => (
              <NavLink key={e.key} to={e.to} className="qc-pop-item" onClick={() => moreRef.current?.removeAttribute('open')}>
                <e.icon aria-hidden />{e.label}
              </NavLink>
            ))}
          </div>
        </details>
      )}
    </nav>
  );
}

function ShellBody() {
  const me = useSession();
  const panel = usePanel();
  const location = useLocation();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(() => { try { return localStorage.getItem(RAIL_KEY) === 'collapsed'; } catch { return false; } });
  const [drawer, setDrawer] = useState(false);
  useEffect(() => { setDrawer(false); }, [location.pathname]);
  useEffect(() => { document.title = pageTitle(location.pathname); }, [location.pathname]);
  // The drawer on a narrow screen (review N1): out of the tab order while closed, Escape closes it, focus goes in and back.
  const narrow = useNarrow();
  const menuButton = useRef<HTMLButtonElement>(null);
  const rail = useRef<HTMLElement>(null);
  useEffect(() => {
    if (!narrow || !drawer) return;
    rail.current?.querySelector<HTMLElement>('a, button')?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { setDrawer(false); menuButton.current?.focus(); } };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [narrow, drawer]);
  const toggleRail = () => {
    setCollapsed((c) => { try { localStorage.setItem(RAIL_KEY, c ? 'open' : 'collapsed'); } catch { /* ignore */ } return !c; });
  };
  const { close } = panel;
  const previous = useRef(location.pathname);
  useEffect(() => {
    // The panel belongs to what is on screen: leaving the chat or switching between two conversations closes it. A new
    // conversation getting its id (/chat → /chat/:id, committed after the answer may already be showing) does not.
    const before = previous.current.split('/');
    const now = location.pathname.split('/');
    previous.current = location.pathname;
    if (before[1] !== now[1] || (before[1] === 'chat' && before[2] && before[2] !== now[2])) close();
  }, [location.pathname, close]);
  return (
    <div className="qc-shell" data-rail={collapsed ? 'collapsed' : 'open'} data-drawer={drawer ? 'open' : 'closed'}>
      <header className="qc-header">
        <button ref={menuButton} type="button" className="qc-icon-btn qc-only-narrow" aria-label={drawer ? 'Close menu' : 'Open menu'} aria-expanded={drawer}
          onClick={() => setDrawer((d) => !d)}><Menu aria-hidden /></button>
        <button type="button" className="qc-icon-btn qc-hide-narrow" aria-label={collapsed ? 'Expand the sidebar' : 'Collapse the sidebar'} onClick={toggleRail}>
          {collapsed ? <PanelLeftOpen aria-hidden /> : <PanelLeftClose aria-hidden />}
        </button>
        <MatterPicker />
        <SearchButton />
        <span className="qc-header-spacer" />
        <PrivacyStatus />
        <RecordingStatus />
        <ProfileMenu />
      </header>
      <aside ref={rail} className="qc-rail" aria-label="Sidebar" inert={narrow && !drawer}>
        <NavLink to="/chat" className="qc-brand" aria-label="EchoMind Legal, new chat">
          <span className="qc-brand-mark" aria-hidden><Landmark /></span>
          <span className="qc-brand-text qc-hide-collapsed" aria-hidden><b>EchoMind<em>Legal</em></b><small>Private AI for legal work</small></span>
        </NavLink>
        <nav aria-label="Workspace">
          <button type="button" className="qc-new-chat" title="New chat" onClick={() => navigate('/chat')}><MessageSquarePlus aria-hidden /><span className="qc-rail-text">New chat</span></button>
          <RailLink to="/chat" icon={<MessageSquare aria-hidden />} title="Chat">Chat</RailLink>
          <RailLink to="/matters" icon={<Briefcase aria-hidden />} title="Matters">Matters</RailLink>
          <DraftsLink />
          <RailLink to="/activity" icon={<Activity aria-hidden />} title="Activity">Activity</RailLink>
        </nav>
        <History />
      </aside>
      {drawer && <button type="button" className="qc-scrim qc-only-narrow" aria-label="Close menu" onClick={() => setDrawer(false)} />}
      <div className="qc-body">
        <main className="qc-main" id="main">
          {me.features?.demo_signin && (
            <div className="lw-demo-banner" role="status" aria-label="Demo sign-in">
              Demo sign-in is on: anyone who can open this page can enter as the demo's fictional people, without a password.
              {hasFirmRole(me, 'firm_admin') ? ' Turn it off before real data goes in.' : ' Do not enter real client information.'}
            </div>
          )}
          <MatterTabs />
          <Outlet />
        </main>
        <SidePanel />
      </div>
    </div>
  );
}

function useNarrow(): boolean {
  const query = '(max-width: 860px)';
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && !!window.matchMedia?.(query).matches);
  useEffect(() => {
    const m = window.matchMedia?.(query);
    if (!m) return;
    const on = () => setNarrow(m.matches);
    m.addEventListener?.('change', on);
    return () => m.removeEventListener?.('change', on);
  }, []);
  return narrow;
}

const PAGE: Record<string, string> = {
  chat: 'Chat', matters: 'Matters', drafts: 'Drafts', reviews: 'Drafts', activity: 'Activity', templates: 'Templates', playbooks: 'Playbooks',
  checklists: 'Checklists', enquiries: 'Enquiries', time: 'Time', regulatory: 'Regulatory updates', home: 'Home',
};
const TAB: Record<string, string> = {
  overview: 'Overview', documents: 'Documents', research: 'Research', conversation: 'Conversation', compare: 'Compare', drafting: 'Drafting',
  obligations: 'Obligations', diligence: 'Due diligence', summaries: 'Summaries', panel: 'Review panel', regulatory: 'Regulatory',
  meetings: 'Meetings', members: 'Members', exports: 'Exports', activity: 'Activity',
};

/** Each page its own title (review A4): screen readers and browser tabs say where you are. */
export function pageTitle(pathname: string): string {
  const parts = pathname.split('/').filter(Boolean).map((p) => { try { return decodeURIComponent(p); } catch { return p; } });
  let name = '';
  if (parts[0] === 'm' && parts[1]) name = `${parts[1]} · ${TAB[parts[2] ?? 'overview'] ?? 'Matter'}`;
  else if (parts[0] === 'admin') name = { users: 'Users & roles', walls: 'Ethical walls', audit: 'Audit trail', records: 'Records',
    connectors: 'Connectors', 'emergency-access': 'Emergency access', evaluations: 'Evaluations' }[parts[1] ?? ''] ?? 'Administration';
  else name = PAGE[parts[0] ?? 'chat'] ?? '';
  return name ? `${name} — EchoMind Legal` : 'EchoMind Legal';
}

export default function CounselShell() {
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.shell = 'counsel';                              // also set app-wide by LawApp; kept for pages mounted alone
  }, []);
  return (
    <MatterProvider>
      <PaletteProvider>
        <ChatMatterProvider>
          <PanelProvider>
            <ShellBody />
          </PanelProvider>
        </ChatMatterProvider>
      </PaletteProvider>
    </MatterProvider>
  );
}
