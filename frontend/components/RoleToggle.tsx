import React, { useEffect } from 'react';
import type { Role } from '../types';
import { roleLabel } from '../utils/silentAssistant';

interface RoleToggleProps {
  roles: { me: Role; other: Role };
  value: Role | null;
  onChange: (role: Role | null) => void;
  disabled?: boolean;
  /** Bind Alt+1 (me) / Alt+2 (other) / Alt+0 (unknown) while mounted (default true). */
  hotkeys?: boolean;
  className?: string;
}

/**
 * "Who is speaking" toggle: two buttons labelled with the profile's role names.
 * Click the active button again to reset to unknown.
 */
const RoleToggle: React.FC<RoleToggleProps> = ({ roles, value, onChange, disabled, hotkeys = true, className = '' }) => {
  useEffect(() => {
    if (!hotkeys || disabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (!e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key === '1') { e.preventDefault(); onChange(roles.me); }
      else if (e.key === '2') { e.preventDefault(); onChange(roles.other); }
      else if (e.key === '0') { e.preventDefault(); onChange(null); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [hotkeys, disabled, roles.me, roles.other, onChange]);

  const btn = (role: Role, label: string, hint: string, activeCls: string) => {
    const active = value === role;
    return (
      <button
        key={role}
        type="button"
        disabled={disabled}
        onClick={() => onChange(active ? null : role)}
        title={`${label} is speaking (${hint})`}
        aria-pressed={active}
        className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold rounded-lg transition-colors touch-manipulation min-h-[32px] disabled:opacity-50 ${
          active ? activeCls : 'text-slate-400 hover:text-white hover:bg-white/10'
        }`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${active ? 'bg-current animate-pulse' : 'bg-slate-600'}`} />
        {label}
        <kbd className="hidden sm:inline text-[9px] font-mono opacity-60 border border-current/30 rounded px-1">{hint}</kbd>
      </button>
    );
  };

  return (
    <div className={`inline-flex items-center gap-1 rounded-xl border border-white/10 bg-white/5 p-0.5 ${className}`} role="group" aria-label="Who is speaking">
      <span className="hidden md:inline text-[10px] uppercase tracking-wider text-slate-500 pl-2 pr-1">Speaking</span>
      {btn(roles.me, roleLabel(roles.me), 'Alt+1', 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30')}
      {btn(roles.other, roleLabel(roles.other), 'Alt+2', 'bg-violet-500/20 text-violet-300 border border-violet-500/30')}
      {value == null && <span className="text-[10px] text-slate-500 pr-2 hidden lg:inline">tap who’s talking</span>}
    </div>
  );
};

export default RoleToggle;
