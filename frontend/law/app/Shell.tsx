import React from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router';
import { useQueryClient } from '@tanstack/react-query';
import { Briefcase, FileText, History, Home as HomeIcon, KeyRound, LogOut, Moon, ScrollText, ShieldOff, Siren, Sun, Users, UserCog } from 'lucide-react';
import { api } from '../api/client';
import MatterSwitcher from '../ui/MatterSwitcher';
import { hasFirmRole, useSession } from './session';
import { MatterProvider, useCurrentMatter } from './matter';
import { setThemeMode } from './preferences';

function NavItem({ to, icon, children }: { to: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <NavLink to={to} className="lw-nav-link">
      {icon}<span className="lw-hide-rail">{children}</span>
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

  return (
    <MatterProvider>
    <div className="lw-shell">
      <aside className="lw-glass lw-sidebar" aria-label="Workspace navigation">
        <ShellMatterBlock />
        <div className="lw-nav-label lw-hide-rail">Firm</div>
        <nav aria-label="Firm" style={{ display: 'grid', gap: 2 }}>
          <NavItem to="/home" icon={<HomeIcon />}>Home</NavItem>
          <NavItem to="/matters" icon={<Briefcase />}>Matters</NavItem>
          {hasFirmRole(me, 'firm_admin') && <NavItem to="/admin/users" icon={<UserCog />}>Users &amp; roles</NavItem>}
          {hasFirmRole(me, 'managing_partner') && <NavItem to="/admin/walls" icon={<ShieldOff />}>Ethical walls</NavItem>}
          {hasFirmRole(me, 'firm_admin', 'managing_partner') && <NavItem to="/admin/emergency-access" icon={<Siren />}>Emergency access</NavItem>}
          {hasFirmRole(me, 'firm_admin', 'managing_partner', 'records_admin') && <NavItem to="/admin/audit" icon={<ScrollText />}>Audit trail</NavItem>}
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
    </MatterProvider>
  );
}

function ShellMatterBlock() {
  const matter = useCurrentMatter();
  return (
    <>
      <MatterSwitcher />
      {matter && (
        <>
          <div className="lw-nav-label lw-hide-rail">Matter</div>
          <nav aria-label="Matter" style={{ display: 'grid', gap: 2 }}>
            <NavItem to={`/m/${encodeURIComponent(matter.number)}/overview`} icon={<FileText />}>Overview</NavItem>
            <NavItem to={`/m/${encodeURIComponent(matter.number)}/members`} icon={<Users />}>Members</NavItem>
            {matter.my_role === 'lead' && <NavItem to={`/m/${encodeURIComponent(matter.number)}/activity`} icon={<History />}>Activity</NavItem>}
          </nav>
        </>
      )}
    </>
  );
}
