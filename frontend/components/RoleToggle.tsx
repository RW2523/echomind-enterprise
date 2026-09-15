import React, { useEffect } from 'react';
import type { Role } from '../types';
import { roleLabel } from '../utils/silentAssistant';

interface RoleToggleProps {
  roles: { me: Role; other: Role };
  value: Role | null;
  onChange: (role: Role | null) => void;
  disabled?: boolean;
  /** Bind Alt+1 (me) / Alt+2 (other) / Alt+0 (auto) while mounted (default true). */
  hotkeys?: boolean;
  className?: string;
}

/**
 * "Who is speaking" segmented control: `Lawyer | Client | Auto`.
 * Auto = unknown role (sends {type:'speaker', role:null}). Keyboard shortcuts are kept
 * (Alt+1 / Alt+2 / Alt+0) but only surface as title tooltips.
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

  const options: { role: Role | null; label: string; hint: string; activeCls: string }[] = [
    { role: roles.me, label: roleLabel(roles.me), hint: 'Alt+1', activeCls: 'bg-cyan-500/20 text-cyan-200' },
    { role: roles.other, label: roleLabel(roles.other), hint: 'Alt+2', activeCls: 'bg-violet-500/20 text-violet-200' },
    { role: null, label: 'Auto', hint: 'Alt+0', activeCls: 'bg-white/15 text-white' },
  ];

  return (
    <div
      role="radiogroup"
      aria-label="Who is speaking"
      className={`inline-flex items-center rounded-xl border border-white/10 bg-white/[0.04] p-0.5 ${disabled ? 'opacity-50' : ''} ${className}`}
      title="Who is speaking now"
    >
      {options.map((o, i) => {
        const active = value === o.role;
        return (
          <React.Fragment key={o.role ?? 'auto'}>
            {i > 0 && <span className="w-px h-3.5 bg-white/10" aria-hidden />}
            <button
              type="button"
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onClick={() => onChange(o.role)}
              title={o.role ? `${o.label} is speaking · ${o.hint}` : `Speaker unknown — let the assistant decide · ${o.hint}`}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors touch-manipulation min-h-[32px] whitespace-nowrap ${
                active ? o.activeCls : 'text-slate-400 hover:text-white hover:bg-white/[0.06]'
              }`}
            >
              {o.label}
            </button>
          </React.Fragment>
        );
      })}
    </div>
  );
};

export default RoleToggle;
