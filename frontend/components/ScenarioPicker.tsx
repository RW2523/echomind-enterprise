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
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  return (
    <div ref={ref} className={`relative ${className}`}>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2.5 py-1.5 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/20 transition-colors touch-manipulation min-h-[32px] disabled:opacity-50"
        title="Change scenario"
      >
        <span>{SCENARIO_ICONS[shown.id] ?? '💬'}</span>
        <span className="truncate max-w-[9rem]">{shown.label}</span>
        {value === 'auto' && <span className="text-[9px] uppercase tracking-wider text-cyan-400/70">auto</span>}
        <svg className="w-3 h-3 opacity-70" viewBox="0 0 20 20" fill="currentColor"><path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" /></svg>
      </button>
      {open && (
        <div className="absolute left-0 top-full mt-1 z-50 w-64 rounded-xl border border-white/15 bg-slate-800 shadow-xl overflow-hidden">
          {[AUTO_SCENARIO, ...list].map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => { onChange(s.id); setOpen(false); }}
              className={`w-full flex items-start gap-2.5 px-3 py-2.5 text-left hover:bg-white/10 transition-colors border-b border-white/5 last:border-b-0 ${s.id === value ? 'bg-cyan-500/10' : ''}`}
            >
              <span className="text-base leading-none mt-0.5">{SCENARIO_ICONS[s.id] ?? '💬'}</span>
              <span className="min-w-0">
                <span className={`block text-xs font-semibold ${s.id === value ? 'text-cyan-300' : 'text-white'}`}>{s.label}</span>
                <span className="block text-[10px] text-slate-400 leading-snug line-clamp-2">
                  {s.id === 'auto' ? s.description : `${roleLabel(s.roles.me)} · ${roleLabel(s.roles.other)}`}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default ScenarioPicker;
