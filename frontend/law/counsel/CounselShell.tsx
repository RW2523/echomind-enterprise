/**
 * The EchoMind Legal shell (docs/echomind-legal/02): a navy rail with the workspace (Chat, Matters, Drafts, Activity), the
 * chosen matter's own pages, recent matters and recent conversations; a header with the matter the chat is about, search
 * (the command palette), the privacy state, the real capture state and the person; the page; and a right-hand panel that
 * opens beside the conversation. Settings and the firm's administration pages live under the profile menu. Every existing
 * Law page keeps working inside this shell; a matter's own pages are reached from the tab strip above them.
 */
import React, { useEffect, useRef, useState } from 'react';
import { NavLink, Outlet, useLocation, useMatch, useNavigate } from 'react-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Activity, Briefcase, FilePen, FolderOpen, KeyRound, Landmark, LogOut, Menu, MessageSquare, MessageSquarePlus, Mic, MicOff,
  Moon, PanelLeftClose, PanelLeftOpen, Search, ShieldCheck, Sun, Timer,
} from 'lucide-react';
import { api } from '../api/client';
import { useMatters, useReviews } from '../api/queries';
import { useCapturing } from '../app/capture';
import { MatterProvider, useCurrentMatter } from '../app/matter';
import { firmEntries, initials, matterEntries } from '../app/nav';
import { setThemeMode } from '../app/preferences';
import { useSession } from '../app/session';
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
/** The matter pages offered in the rail when no matter page is open (the tab strip shows them all on a matter page). */
const RAIL_MATTER_PAGES = ['overview', 'documents', 'research', 'meetings', 'conversation', 'drafting', 'obligations', 'summaries', 'panel'];

// ── Header: privacy and capture state ──────────────────────────────────────────
function PrivacyStatus() {
  const { open, setOpen, button, pop, style } = usePopover();
  return (
    <>
      <button ref={button} type="button" className="qc-status" aria-expanded={open} onClick={() => setOpen((o) => !o)} aria-label="Private: about where answers come from">
        <ShieldCheck aria-hidden /><span className="qc-narrow-sr">Private</span>
      </button>
      {open && (<Layer>
        <div ref={pop} className="qc-pop" role="dialog" aria-label="Private" style={{ ...style, padding: 14 }}>
          <p style={{ margin: '0 0 8px', fontWeight: 600 }}>Private to this appliance</p>
          <p className="qc-meta" style={{ margin: 0 }}>
            Answers are produced on this EchoMind appliance by its own model, only from the documents of the matter you chose and only
            those you may read. No cloud service is used to answer. Every question, source opened and decision is recorded in the audit trail.
          </p>
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
  const firm = firmEntries(me).filter((e) => !['home', 'matters', 'reviews'].includes(e.key));
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
/** The chosen matter's main pages, from the chat and the firm pages (on a matter page its tab strip shows them all). */
function CurrentMatter() {
  const me = useSession();
  const routeMatter = useCurrentMatter();
  const { matter } = useChatMatter();
  if (routeMatter || !matter) return null;
  const entries = matterEntries(me, matter).filter((e) => RAIL_MATTER_PAGES.includes(e.key));
  return (
    <>
      <div className="qc-rail-label qc-hide-collapsed" title={`${matter.number} · ${matter.title}`}>{matter.number}</div>
      <nav aria-label="Current matter">
        {entries.map((e) => <RailLink key={e.key} to={e.to} icon={<e.icon aria-hidden />} label={e.label}>{e.label}</RailLink>)}
      </nav>
    </>
  );
}

/** A few of the person's other matters, to open one directly (beside the chat; the Matters page lists them all). */
function RecentMatters() {
  const onChat = !!useMatch('/chat/*');
  const matters = useMatters();
  const { matter } = useChatMatter();
  const routeMatter = useCurrentMatter();
  const current = routeMatter?.id ?? matter?.id;
  const list = (matters.data ?? []).filter((m) => m.status === 'open' && m.id !== current).slice(0, 4);
  if (!onChat || !list.length) return null;
  return (
    <>
      <div className="qc-rail-label qc-hide-collapsed">Other matters</div>
      <nav className="qc-history qc-hide-collapsed" aria-label="Other matters">
        {list.map((m) => (
          <NavLink key={m.id} to={`/m/${encodeURIComponent(m.number)}/overview`} className="qc-recent-matter" title={`${m.number} · ${m.title}`}>
            <FolderOpen aria-hidden /><span>{m.title}</span>
          </NavLink>
        ))}
      </nav>
    </>
  );
}

function History() {
  const q = useQuery({ queryKey: ['counsel', 'conversations'], retry: false, staleTime: 10_000, queryFn: () => counselApi.conversations() });
  const items = q.data?.conversations ?? [];
  if (!items.length) return null;
  return (
    <>
      <div className="qc-rail-label qc-hide-collapsed">Recent conversations</div>
      <nav className="qc-history qc-hide-collapsed" aria-label="Recent conversations">
        {items.slice(0, 12).map((c) => (
          <NavLink key={c.id} to={`/chat/${c.id}`} className={({ isActive }) => `qc-history-item${isActive ? ' active' : ''}`}>
            <span>{c.title}</span><small>{(c.matters?.length ?? 0) > 1 ? c.matters!.map((m) => m.number).join(' + ') : c.matter.number}</small>
          </NavLink>
        ))}
      </nav>
    </>
  );
}

function RailLink({ to, icon, children, end, label }: { to: string; icon: React.ReactNode; children: React.ReactNode; end?: boolean; label?: string }) {
  return (
    <NavLink to={to} end={end} className="qc-rail-link" aria-label={label} title={label}>
      {icon}<span className="qc-rail-text qc-hide-collapsed">{children}</span>
    </NavLink>
  );
}

/** Drafts, with how many items wait for the person's decision (the reviews queue). */
function DraftsLink() {
  const n = useReviews().data?.count ?? 0;
  return (
    <RailLink to="/drafts" icon={<FilePen aria-hidden />} label={n ? `Drafts, ${n} waiting for you` : 'Drafts'}>
      Drafts{n > 0 && <span className="qc-rail-count" aria-hidden="true">{n}</span>}
    </RailLink>
  );
}

/** The command palette (Ctrl/⌘ K) that the Law pages already provide: matters, pages and actions. */
function SearchButton() {
  const open = useOpenPalette();
  const shortcut = isMac() ? '⌘K' : 'Ctrl K';
  return (
    <button type="button" className="qc-search qc-hide-small" onClick={() => open()} aria-label={`Search or jump to (${shortcut})`}>
      <Search aria-hidden /><span>Find a matter, page or action…</span><kbd className="qc-kbd">{shortcut}</kbd>
    </button>
  );
}

/** A matter's own pages, above them (they used to fill the sidebar). */
function MatterTabs() {
  const me = useSession();
  const matter = useCurrentMatter();
  const { setMatterId } = useChatMatter();
  const navigate = useNavigate();
  if (!matter) return null;
  return (
    <nav className="qc-tabs" aria-label="Matter">
      <button type="button" className="qc-tab" style={{ border: 'none', background: 'none', font: 'inherit', cursor: 'pointer' }}
        onClick={() => { setMatterId(matter.id); navigate('/chat'); }}><MessageSquare aria-hidden />Chat about this matter</button>
      {matterEntries(me, matter).map((e) => (
        <NavLink key={e.key} to={e.to} className={({ isActive }) => `qc-tab${isActive ? ' active' : ''}`}><e.icon aria-hidden />{e.label}</NavLink>
      ))}
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
        <button type="button" className="qc-icon-btn qc-only-narrow" aria-label={drawer ? 'Close menu' : 'Open menu'} aria-expanded={drawer}
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
      <aside className="qc-rail" aria-label="Sidebar">
        <NavLink to="/chat" className="qc-brand" aria-label="EchoMind Legal, new chat">
          <span className="qc-brand-mark" aria-hidden><Landmark /></span>
          <span className="qc-brand-text qc-hide-collapsed" aria-hidden><b>EchoMind<em>Legal</em></b><small>Private AI for legal work</small></span>
        </NavLink>
        <nav aria-label="Workspace">
          <button type="button" className="qc-new-chat" onClick={() => navigate('/chat')}><MessageSquarePlus aria-hidden /><span className="qc-hide-collapsed">New chat</span></button>
          <RailLink to="/chat" icon={<MessageSquare aria-hidden />}>Chat</RailLink>
          <RailLink to="/matters" icon={<Briefcase aria-hidden />}>Matters</RailLink>
          <DraftsLink />
          <RailLink to="/activity" icon={<Activity aria-hidden />}>Activity</RailLink>
          {me.features?.time_entries && <RailLink to="/time" icon={<Timer aria-hidden />}>Time</RailLink>}
        </nav>
        <CurrentMatter />
        <RecentMatters />
        <History />
        <div className="qc-rail-foot qc-hide-collapsed"><ShieldCheck aria-hidden />Runs on this appliance. No cloud.</div>
      </aside>
      {drawer && <button type="button" className="qc-scrim qc-only-narrow" aria-label="Close menu" onClick={() => setDrawer(false)} />}
      <div className="qc-body">
        <main className="qc-main" id="main">
          {me.features?.demo_signin && (
            <div className="lw-demo-banner" role="status" aria-label="Demo sign-in">
              Demo sign-in is on: anyone who can open this page can enter as the demo's fictional people, without a password. Turn it off before real data goes in.
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

export default function CounselShell() {
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.shell = 'counsel';                              // also set app-wide by LawApp; kept for pages mounted alone
    document.title = 'EchoMind Legal';
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
