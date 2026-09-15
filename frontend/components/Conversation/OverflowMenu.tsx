import React, { useEffect, useRef, useState } from "react";
import { PersonaType } from "../../types";

export interface OverflowMenuProps {
  /** Active persona (drives the voice system prompt) */
  persona?: PersonaType;
  onPersonaChange?: (persona: PersonaType) => void;
  /** Continuous listening: only reply after the wake word */
  listenOnly?: boolean;
  onListenOnlyToggle?: () => void;
  /** Push the current persona / context to the live session */
  onApplyContext?: () => void;
  onSettingsClick?: () => void;
  isConnected: boolean;
  className?: string;
}

const PERSONAS = Object.values(PersonaType);

/** Secondary, rarely used voice controls, kept out of the main control row. */
export const OverflowMenu: React.FC<OverflowMenuProps> = ({
  persona,
  onPersonaChange,
  listenOnly = false,
  onListenOnlyToggle,
  onApplyContext,
  onSettingsClick,
  isConnected,
  className = "",
}) => {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent | TouchEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("touchstart", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("touchstart", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const hasAnything =
    !!onPersonaChange || !!onListenOnlyToggle || !!onApplyContext || !!onSettingsClick;
  if (!hasAnything) return null;

  return (
    <div ref={rootRef} className={`relative ${className}`}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="More voice options"
        title="More options"
        className={`h-11 w-11 rounded-xl flex items-center justify-center text-[18px] leading-none transition-colors touch-manipulation ${
          open
            ? "bg-white/[0.1] text-slate-200"
            : "bg-white/[0.04] text-slate-400 hover:bg-white/[0.08] hover:text-slate-200"
        }`}
      >
        <span aria-hidden>•••</span>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute bottom-full right-0 mb-2 w-[16.5rem] rounded-xl bg-[#0b1220]/95 backdrop-blur-xl border border-white/10 shadow-[0_18px_50px_-12px_rgba(0,0,0,0.7)] p-3 space-y-3 z-50"
        >
          {onPersonaChange && (
            <label className="block">
              <span className="block text-[11px] font-medium uppercase tracking-wider text-slate-500 mb-1.5">
                Persona
              </span>
              <select
                value={persona ?? ""}
                onChange={(e) => onPersonaChange(e.target.value as PersonaType)}
                className="w-full rounded-lg bg-black/40 border border-white/10 px-2.5 py-2 text-[13px] text-slate-200 outline-none focus:border-accent/50"
              >
                {PERSONAS.map((p) => (
                  <option key={p} value={p} className="bg-[#0b1220]">
                    {p}
                  </option>
                ))}
              </select>
            </label>
          )}

          {onListenOnlyToggle && (
            <button
              type="button"
              role="menuitemcheckbox"
              aria-checked={listenOnly}
              onClick={onListenOnlyToggle}
              className="w-full flex items-center justify-between gap-3 rounded-lg px-2.5 py-2 text-left text-[13px] text-slate-300 hover:bg-white/[0.06] transition-colors"
            >
              <span className="min-w-0">
                <span className="block truncate">Continuous listening</span>
                <span className="block text-[11px] leading-snug text-slate-500">
                  Reply only after the wake word
                </span>
              </span>
              <span
                className={`shrink-0 w-9 h-5 rounded-full p-0.5 transition-colors ${
                  listenOnly ? "bg-accent/70" : "bg-white/[0.12]"
                }`}
                aria-hidden
              >
                <span
                  className={`block w-4 h-4 rounded-full bg-white/90 transition-transform ${
                    listenOnly ? "translate-x-4" : ""
                  }`}
                />
              </span>
            </button>
          )}

          {onApplyContext && (
            <button
              type="button"
              role="menuitem"
              disabled={!isConnected}
              onClick={() => {
                onApplyContext();
                setOpen(false);
              }}
              className="w-full text-left rounded-lg px-2.5 py-2 text-[13px] text-slate-300 hover:bg-white/[0.06] disabled:opacity-40 disabled:hover:bg-transparent transition-colors"
            >
              Apply context
              <span className="block text-[11px] text-slate-500">
                Send persona and context to this session
              </span>
            </button>
          )}

          {onSettingsClick && (
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                onSettingsClick();
                setOpen(false);
              }}
              className="w-full text-left rounded-lg px-2.5 py-2 text-[13px] text-slate-300 hover:bg-white/[0.06] transition-colors"
            >
              Voice settings
            </button>
          )}
        </div>
      )}
    </div>
  );
};
