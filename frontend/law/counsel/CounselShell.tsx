/**
 * Popovers are fixed-position in the header and need no portal (nothing above them is transformed).
 *
 * The Quiet Counsel shell (docs/echomind-legal/02): four destinations (Chat, Matters, Drafts, Activity) and recent
 * conversations in a collapsible rail; a header with the matter the chat is about, the privacy state and the real
 * capture state; the page; and a right-hand panel that opens beside the conversation. Settings and the firm's
 * administration pages live under the profile menu. Every existing Law page keeps working inside this shell; a matter's
 * own pages are reached from the tab strip above them.
 */
import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { NavLink, Outlet, useLocation, useMatch, useNavigate } from 'react-router';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Activity, Briefcase, ChevronsUpDown, FilePen, KeyRound, Lock, LogOut, Menu, MessageSquare, MessageSquarePlus, Mic, MicOff,
  Moon, PanelLeftClose, PanelLeftOpen, Search, ShieldCheck, Sun,
} from 'lucide-react';
import { api } from '../api/client';
import { keys, useMatters, useReviews } from '../api/queries';
import type { Matter } from '../api/types';
import { useCapturing } from '../app/capture';
import { MatterProvider, useCurrentMatter } from '../app/matter';
import { firmEntries, initials, matterEntries } from '../app/nav';
import { getBlockers } from '../app/navigationGuard';
import { setThemeMode } from '../app/preferences';
import { useSession } from '../app/session';
import { isMac, PaletteProvider, useOpenPalette } from '../ui/Palette';
import { Dialog } from '../ui/primitives';
import { counselApi } from './api';
import { ChatMatterProvider, PanelProvider, useChatMatter, usePanel } from './context';
import SidePanel from './SidePanel';
import './counsel.css';

const RAIL_KEY = 'echomind.counsel.rail';

function usePopover() {
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  const pop = useRef<HTMLDivElement>(null);
  const [at, setAt] = useState<{ top: number; left?: number; right?: number }>({ top: 0 });
  useLayoutEffect(() => {
    if (!open) return;
    const r = button.current?.getBoundingClientRect();
    if (r) setAt(r.left > window.innerWidth / 2 ? { top: r.bottom + 6, right: Math.max(8, window.innerWidth - r.right) } : { top: r.bottom + 6, left: r.left });
  }, [open]);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!button.current?.contains(t) && !pop.current?.contains(t)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { setOpen(false); button.current?.focus(); } };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey); };
  }, [open]);
  return { open, setOpen, button, pop, style: { top: at.top, left: at.left, right: at.right } as React.CSSProperties };
}

// ── Header: matter context ─────────────────────────────────────────────────────
function MatterPicker() {
  const matters = useMatters();
  const routeMatter = useCurrentMatter();
  const { matter: chatMatter, setMatterId } = useChatMatter();
  const onChat = !!useMatch('/chat/*');
  const navigate = useNavigate();
  const qc = useQueryClient();
  const { open, setOpen, button, pop, style } = usePopover();
  const [filter, setFilter] = useState('');
  const [pending, setPending] = useState<{ target: Matter; reasons: string[] } | null>(null);
  const current = onChat ? chatMatter : routeMatter ?? chatMatter;
  const list = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return (matters.data ?? []).filter((m) => !q || `${m.number} ${m.title} ${m.client.name}`.toLowerCase().includes(q));
  }, [matters.data, filter]);

  const go = (m: Matter) => {
    setPending(null);
    setOpen(false);
    if (routeMatter && routeMatter.id !== m.id) qc.removeQueries({ queryKey: keys.matter(routeMatter.id) });
    const changed = chatMatter?.id !== m.id;
    setMatterId(m.id);
    if (onChat) { if (changed) navigate('/chat'); }         // a conversation is about one matter: a new matter is a new chat
    else if (routeMatter) navigate(`/m/${encodeURIComponent(m.number)}/overview`);
  };
  const choose = (m: Matter) => {
    if (current?.id === m.id) { setOpen(false); return; }
    const reasons = getBlockers();
    if (reasons.length) setPending({ target: m, reasons });
    else go(m);
  };

  return (
    <>
      <button ref={button} type="button" className="qc-matter-btn" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        aria-label={current ? `Matter: ${current.number} ${current.title}. Switch matter` : 'Choose a matter'}>
        {current && current.sensitivity !== 'standard' && <Lock aria-label="Restricted" />}
        <span>{current ? <>{current.number}<span className="qc-hide-small"> · {current.title}</span></> : 'Choose a matter'}</span>
        <ChevronsUpDown aria-hidden />
      </button>
      {open && (
        <div ref={pop} className="qc-pop" style={style}>
          <input className="lw-input" autoFocus placeholder="Find a matter…" aria-label="Find a matter" value={filter} onChange={(e) => setFilter(e.target.value)} />
          <div className="qc-pop-label">Your matters</div>
          <div role="listbox" aria-label="Your matters" style={{ maxHeight: 320, overflowY: 'auto' }}>
            {list.map((m) => (
              <button key={m.id} type="button" role="option" aria-selected={current?.id === m.id} className="qc-pop-item" onClick={() => choose(m)}>
                <span style={{ minWidth: 0 }}>
                  <span style={{ display: 'block', fontWeight: 600 }}>{m.number} · {m.title}</span>
                  <span className="qc-meta">{m.client.name}</span>
                </span>
              </button>
            ))}
            {matters.data && list.length === 0 && <p className="qc-meta" style={{ padding: '6px 10px' }}>No matching matters.</p>}
          </div>
          <p className="qc-meta" style={{ borderTop: '1px solid var(--rule)', margin: '6px 0 0', padding: '8px 10px 2px' }}>
            Only matters you're a member of appear here. Each conversation uses one matter's documents.
          </p>
        </div>)}
      {pending && (
        <Dialog title={`Switch to ${pending.target.number}?`} onClose={() => setPending(null)}
          footer={<>
            <button type="button" className="lw-btn lw-btn-ghost" onClick={() => setPending(null)}>Stay here</button>
            <button type="button" className="lw-btn lw-btn-primary" onClick={() => go(pending.target)}>Switch matter</button>
          </>}>
          <p style={{ margin: 0 }}>Something is still in progress:</p>
          <ul style={{ margin: 0, paddingLeft: 18 }}>{pending.reasons.map((r) => <li key={r}>{r}</li>)}</ul>
        </Dialog>
      )}
    </>
  );
}

// ── Header: privacy and capture state ──────────────────────────────────────────
function PrivacyStatus() {
  const { open, setOpen, button, pop, style } = usePopover();
  return (
    <>
      <button ref={button} type="button" className="qc-status" aria-expanded={open} onClick={() => setOpen((o) => !o)} aria-label="Private: about where answers come from">
        <ShieldCheck aria-hidden /><span className="qc-narrow-sr">Private</span>
      </button>
      {open && (
        <div ref={pop} className="qc-pop" role="dialog" aria-label="Private" style={{ ...style, padding: 14 }}>
          <p style={{ margin: '0 0 8px', fontWeight: 600 }}>Private to this appliance</p>
          <p className="qc-meta" style={{ margin: 0 }}>
            Answers are produced on this EchoMind appliance by its own model, only from the documents of the matter you chose and only
            those you may read. No cloud service is used to answer. Every question, source opened and decision is recorded in the audit trail.
          </p>
        </div>)}
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
      <button ref={button} type="button" className="qc-icon-btn" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        aria-label={`Account and settings for ${me.user.username}`} style={{ border: '1px solid var(--rule)', fontWeight: 650, fontSize: 13 }}>
        {initials(me.user.username)}
      </button>
      {open && (
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
        </div>)}
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
        {items.slice(0, 20).map((c) => (
          <NavLink key={c.id} to={`/chat/${c.id}`} className={({ isActive }) => `qc-history-item${isActive ? ' active' : ''}`}>
            <span>{c.title}</span><small>{c.matter.number}</small>
          </NavLink>
        ))}
      </nav>
    </>
  );
}

function RailLink({ to, icon, children, end, label }: { to: string; icon: React.ReactNode; children: React.ReactNode; end?: boolean; label?: string }) {
  return <NavLink to={to} end={end} className="qc-rail-link" aria-label={label}>{icon}<span className="qc-hide-collapsed">{children}</span></NavLink>;
}

/** Drafts, with how many items wait for the person's decision (the reviews queue). */
function DraftsLink() {
  const n = useReviews().data?.count ?? 0;
  return (
    <RailLink to="/drafts" icon={<FilePen aria-hidden />} label={n ? `Drafts, ${n} waiting for you` : 'Drafts'}>
      Drafts{n > 0 && <span className="lw-nav-count" aria-hidden="true" style={{ marginLeft: 'auto' }}>{n}</span>}
    </RailLink>
  );
}

/** The command palette (Ctrl/⌘ K) that the Law pages already provide. */
function SearchButton() {
  const open = useOpenPalette();
  return (
    <button type="button" className="qc-icon-btn qc-hide-small" onClick={() => open()} aria-label={`Search or jump to (${isMac() ? '⌘K' : 'Ctrl K'})`}>
      <Search aria-hidden />
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
        <NavLink to="/chat" className="qc-brand" aria-label="EchoMind Legal, new chat"><span className="qc-brand-mark" aria-hidden>E</span><span className="qc-hide-narrow">EchoMind</span></NavLink>
        <MatterPicker />
        <span className="qc-header-spacer" />
        <SearchButton />
        <PrivacyStatus />
        <RecordingStatus />
        <ProfileMenu />
      </header>
      <nav className="qc-rail" aria-label="Workspace">
        <button type="button" className="qc-new-chat" onClick={() => navigate('/chat')}><MessageSquarePlus aria-hidden /><span className="qc-hide-collapsed">New chat</span></button>
        <RailLink to="/chat" icon={<MessageSquare aria-hidden />}>Chat</RailLink>
        <RailLink to="/matters" icon={<Briefcase aria-hidden />}>Matters</RailLink>
        <DraftsLink />
        <RailLink to="/activity" icon={<Activity aria-hidden />}>Activity</RailLink>
        <History />
      </nav>
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
    root.dataset.shell = 'counsel';
    document.title = 'EchoMind Legal';
    return () => { delete root.dataset.shell; };
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
