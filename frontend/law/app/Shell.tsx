import React from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { ClipboardCheck, KeyRound, LogOut, Moon, Search, Sun } from 'lucide-react';
import { api } from '../api/client';
import { useReviews } from '../api/queries';
import MatterSwitcher from '../ui/MatterSwitcher';
import { isMac, PaletteProvider, useOpenPalette } from '../ui/Palette';
import { useSession } from './session';
import { MatterProvider, useCurrentMatter } from './matter';
import { firmEntries, matterEntries } from './nav';
import { setThemeMode } from './preferences';

/** D-065: the light on glass follows the pointer (a CSS position on the surface under it); off with reduced motion. */
function usePointerLight() {
  React.useEffect(() => {
    if (typeof window === 'undefined' || window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;
    let frame = 0;
    const onMove = (e: PointerEvent) => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        const el = (e.target as Element | null)?.closest?.('.lw-glass, .lw-paper, .lw-tile') as HTMLElement | null;
        if (!el) return;
        const r = el.getBoundingClientRect();
        el.style.setProperty('--mx', `${e.clientX - r.left}px`);
        el.style.setProperty('--my', `${e.clientY - r.top}px`);
      });
    };
    window.addEventListener('pointermove', onMove, { passive: true });
    return () => { window.removeEventListener('pointermove', onMove); if (frame) cancelAnimationFrame(frame); };
  }, []);
}

function SearchButton() {
  const open = useOpenPalette();
  const keys = isMac() ? '⌘K' : 'Ctrl K';
  return (
    <button type="button" className="lw-search-btn" onClick={() => open()} aria-label={`Search or jump to (${keys})`}>
      <Search aria-hidden /><span>Search or jump to…</span><kbd className="lw-kbd">{keys}</kbd>
    </button>
  );
}

function NavItem({ to, icon, children }: { to: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <NavLink to={to} className="lw-nav-link">
      {icon}<span className="lw-hide-rail">{children}</span>
    </NavLink>
  );
}

/** S2.7: everything waiting for your decision, with how many. */
function ReviewsNavItem() {
  const reviews = useReviews();
  const n = reviews.data?.count ?? 0;
  return (
    <NavLink to="/reviews" className="lw-nav-link" aria-label={n ? `Reviews, ${n} waiting for you` : 'Reviews'}>
      <ClipboardCheck /><span className="lw-hide-rail">Reviews</span>{n > 0 && <span className="lw-nav-count" aria-hidden="true">{n}</span>}
    </NavLink>
  );
}

/** Glass sidebar (navigation layer) + content. Items a person cannot use are absent, not disabled. */
export default function Shell() {
  const me = useSession();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [dark, setDark] = React.useState(() => document.documentElement.dataset.theme === 'dark');

  const signOut = async () => {
    try { await api.post('/api/auth/logout'); } finally {
      qc.clear();
      navigate('/signin', { replace: true });
    }
  };
  const toggleTheme = () => {
    setThemeMode(dark ? 'light' : 'dark');
    setDark(!dark);
  };
  usePointerLight();

  return (
    <MatterProvider>
    <PaletteProvider>
    <div className="lw-shell">
      <aside className="lw-glass lw-sidebar" aria-label="Workspace navigation">
        <ShellMatterBlock />
        <div className="lw-nav-label lw-hide-rail">Firm</div>
        <nav aria-label="Firm" style={{ display: 'grid', gap: 2 }}>
          {firmEntries(me).map((e) => (e.key === 'reviews' ? <ReviewsNavItem key={e.key} />
            : <NavItem key={e.key} to={e.to} icon={<e.icon />}>{e.label}</NavItem>))}
        </nav>
        <div style={{ marginTop: 'auto', borderTop: '1px solid var(--rule)', paddingTop: 8, display: 'grid', gap: 2 }}>
          <NavItem to="/account/security" icon={<KeyRound />}>Security</NavItem>
          <button type="button" className="lw-nav-link" style={{ border: 'none', background: 'none', font: 'inherit', cursor: 'pointer' }} onClick={toggleTheme}>
            {dark ? <Sun size={16} /> : <Moon size={16} />}<span className="lw-hide-rail">{dark ? 'Light appearance' : 'Dark appearance'}</span>
          </button>
          <button type="button" className="lw-nav-link" style={{ border: 'none', background: 'none', font: 'inherit', cursor: 'pointer' }} onClick={signOut}>
            <LogOut size={16} /><span className="lw-hide-rail">Sign out <span className="lw-meta">· {me.user.username}</span></span>
          </button>
        </div>
      </aside>
      <main className="lw-main" id="main">
        <Outlet />
      </main>
    </div>
    </PaletteProvider>
    </MatterProvider>
  );
}

function ShellMatterBlock() {
  const matter = useCurrentMatter();
  const me = useSession();
  return (
    <>
      <MatterSwitcher />
      <SearchButton />
      {matter && (
        <>
          <div className="lw-nav-label lw-hide-rail">Matter</div>
          <nav aria-label="Matter" style={{ display: 'grid', gap: 2 }}>
            {matterEntries(me, matter).map((e) => <NavItem key={e.key} to={e.to} icon={<e.icon />}>{e.label}</NavItem>)}
          </nav>
        </>
      )}
    </>
  );
}
