import React, { useEffect, useRef, useState } from 'react';
import type { Scenario, ScenarioId } from '../types';
import { AUTO_SCENARIO, SCENARIO_ICONS, findScenario, roleLabel } from '../utils/silentAssistant';

interface ScenarioPickerProps {
  value: ScenarioId;
  onChange: (id: ScenarioId) => void;
  scenarios: Scenario[];
  /** 'tiles' = start-modal grid (Auto + 4 tiles). 'pill' = compact session-bar pill with a dropdown. */
  variant?: 'tiles' | 'pill';
  /** Server-resolved scenario id (shown in the pill when the user chose 'auto'). */
  resolved?: ScenarioId | null;
  disabled?: boolean;
  className?: string;
}

const ORDER = ['customer_care', 'legal', 'banking', 'general'];

function orderedScenarios(list: Scenario[]): Scenario[] {
  const byId = new Map(list.map((s) => [s.id, s]));
  const out: Scenario[] = [];
  for (const id of ORDER) { const s = byId.get(id); if (s) out.push(s); }
  for (const s of list) if (!ORDER.includes(s.id)) out.push(s);
  return out;
}

const ScenarioPicker: React.FC<ScenarioPickerProps> = ({ value, onChange, scenarios, variant = 'tiles', resolved, disabled, className = '' }) => {
  const list = orderedScenarios(scenarios);

  if (variant === 'pill') {
    return <ScenarioPill value={value} onChange={onChange} list={list} resolved={resolved} disabled={disabled} className={className} />;
  }

  const tiles: Scenario[] = [AUTO_SCENARIO, ...list];
  return (
    <div className={`grid grid-cols-2 sm:grid-cols-3 gap-2 ${className}`}>
      {tiles.map((s) => {
        const active = s.id === value;
        return (
          <button
            key={s.id}
            type="button"
            disabled={disabled}
            onClick={() => onChange(s.id)}
            className={`text-left rounded-xl border p-3 transition-all touch-manipulation min-h-[72px] ${
              active
                ? 'border-cyan-500/50 bg-cyan-500/10 ring-1 ring-cyan-500/30'
                : 'border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/20'
            } disabled:opacity-50`}
            title={s.description}
          >
            <div className="flex items-center gap-2">
              <span className="text-base leading-none">{SCENARIO_ICONS[s.id] ?? '💬'}</span>
              <span className={`text-sm font-semibold ${active ? 'text-cyan-300' : 'text-white'}`}>{s.label}</span>
            </div>
            <div className="mt-1 text-[11px] text-slate-400 leading-snug line-clamp-2">
              {s.id === 'auto' ? s.description : `${roleLabel(s.roles.me)} · ${roleLabel(s.roles.other)}`}
            </div>
          </button>
        );
      })}
    </div>
  );
};

const ScenarioPill: React.FC<{
  value: ScenarioId; onChange: (id: ScenarioId) => void; list: Scenario[];
  resolved?: ScenarioId | null; disabled?: boolean; className?: string;
}> = ({ value, onChange, list, resolved, disabled, className = '' }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const shownId = value === 'auto' ? (resolved ?? 'auto') : value;
  const shown = findScenario(list, shownId) ?? AUTO_SCENARIO;

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey); };
  }, [open]);

  // The pill is THE mode selector of the session bar: one accent colour, icon + label + chevron.
  return (
    <div ref={ref} className={`relative ${className}`}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="inline-flex items-center gap-2 rounded-xl border border-cyan-400/40 bg-cyan-500/15 pl-2.5 pr-2 py-1.5 text-[13px] font-semibold text-cyan-100 hover:bg-cyan-500/25 hover:border-cyan-300/60 transition-colors touch-manipulation min-h-[38px] disabled:opacity-50 shadow-[0_0_24px_-10px_rgba(34,211,238,0.6)]"
        title="Conversation type — drives who is on the call, which records are pulled and which rules apply"
      >
        <span className="text-base leading-none">{SCENARIO_ICONS[shown.id] ?? '💬'}</span>
        <span className="truncate max-w-[11rem] text-left leading-tight">
          {shown.label}
          {value === 'auto' && <span className="block text-[9px] font-bold uppercase tracking-wider text-cyan-300/70 leading-none mt-0.5">{resolved ? 'auto-detected' : 'auto-detect'}</span>}
        </span>
        <svg className={`w-3.5 h-3.5 opacity-80 transition-transform ${open ? 'rotate-180' : ''}`} viewBox="0 0 20 20" fill="currentColor"><path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" /></svg>
      </button>
      {open && (
        <div role="listbox" className="absolute left-0 top-full mt-1.5 z-50 w-72 rounded-xl border border-white/15 bg-slate-900 shadow-2xl overflow-hidden">
          <div className="px-3 pt-2.5 pb-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-500">Conversation type</div>
          {[AUTO_SCENARIO, ...list].map((s) => {
            const active = s.id === value;
            return (
              <button
                key={s.id}
                type="button"
                role="option"
                aria-selected={active}
                onClick={() => { onChange(s.id); setOpen(false); }}
                className={`w-full flex items-start gap-2.5 px-3 py-2.5 text-left hover:bg-white/[0.07] transition-colors ${active ? 'bg-cyan-500/10' : ''}`}
              >
                <span className="text-base leading-none mt-0.5">{SCENARIO_ICONS[s.id] ?? '💬'}</span>
                <span className="min-w-0 flex-1">
                  <span className={`block text-xs font-semibold ${active ? 'text-cyan-300' : 'text-white'}`}>{s.label}</span>
                  <span className="block text-[11px] text-slate-400 leading-snug line-clamp-2">
                    {s.id === 'auto' ? s.description : `${roleLabel(s.roles.me)} · ${roleLabel(s.roles.other)}`}
                  </span>
                </span>
                {active && <svg className="w-4 h-4 text-cyan-300 shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.704 5.29a1 1 0 010 1.42l-7.5 7.5a1 1 0 01-1.42 0l-3.5-3.5a1 1 0 111.42-1.42L8.5 12.09l6.79-6.8a1 1 0 011.414 0z" clipRule="evenodd" /></svg>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ScenarioPicker;
