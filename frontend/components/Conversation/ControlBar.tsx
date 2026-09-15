import React from "react";
import { ICONS } from "../../constants";
import { OverflowMenu } from "./OverflowMenu";
import { PersonaType } from "../../types";

export interface ControlBarProps {
  isConnected: boolean;
  connecting: boolean;
  connectionError: string | null;
  micMuted: boolean;
  assistantOrb: string;
  /** Continuous listening: only respond after wake word "EchoMind" */
  listenOnly?: boolean;
  onListenOnlyToggle?: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
  onMicMutedToggle: () => void;
  onClearMemory: () => void;
  /** Secondary controls, shown in the ••• menu */
  persona?: PersonaType;
  onPersonaChange?: (persona: PersonaType) => void;
  onApplyContext?: () => void;
  onSettingsClick?: () => void;
  className?: string;
}

/** Single compact row: primary voice controls left, ••• overflow right. */
export const ControlBar: React.FC<ControlBarProps> = ({
  isConnected,
  connecting,
  connectionError,
  micMuted,
  listenOnly = false,
  onListenOnlyToggle,
  onConnect,
  onDisconnect,
  onMicMutedToggle,
  onClearMemory,
  persona,
  onPersonaChange,
  onApplyContext,
  onSettingsClick,
  className = "",
}) => {
  return (
    <div
      className={`shrink-0 border-t border-white/[0.05] px-3 sm:px-4 py-2.5 ${className}`}
      style={{ paddingBottom: "calc(0.625rem + env(safe-area-inset-bottom))" }}
    >
      {connectionError && (
        <p
          className="mb-2 text-[12px] leading-snug text-amber-400/80 text-center px-2"
          role="alert"
        >
          {connectionError}
        </p>
      )}

      {/* Primary controls sit in the horizontal centre; the ••• menu stays flush right.
          The two flex-1 rails keep the centre group optically centred at any width. */}
      <div className="flex items-center justify-center gap-2">
        <div className="flex-1 min-w-0" aria-hidden />
        <div className="flex items-center justify-center gap-2 flex-wrap">
          {!isConnected ? (
            <button
              type="button"
              onClick={onConnect}
              disabled={connecting}
              className="rounded-xl px-5 h-11 text-[13px] font-medium text-slate-900 bg-accent hover:brightness-110 active:scale-[0.98] disabled:opacity-50 transition-all duration-200 touch-manipulation"
            >
              {connecting ? "Starting…" : "Start"}
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={onMicMutedToggle}
                title={micMuted ? "Unmute microphone" : "Mute microphone"}
                aria-label={micMuted ? "Unmute microphone" : "Mute microphone"}
                aria-pressed={micMuted}
                className={`h-11 w-11 shrink-0 rounded-xl flex items-center justify-center transition-colors duration-200 touch-manipulation active:scale-[0.98] ${
                  micMuted
                    ? "bg-rose-500/15 text-rose-400/90 hover:bg-rose-500/25"
                    : "bg-accent/15 text-accent hover:bg-accent/25"
                }`}
              >
                <span className="relative inline-flex items-center justify-center w-5 h-5">
                  <ICONS.Mic className="w-5 h-5" strokeWidth={2} />
                  {micMuted && (
                    <span
                      className="absolute inset-0 flex items-center justify-center pointer-events-none"
                      aria-hidden
                    >
                      <span className="block w-6 h-0.5 bg-current rounded-full origin-center rotate-45 opacity-90" />
                    </span>
                  )}
                </span>
              </button>
              <button
                type="button"
                onClick={onClearMemory}
                className="h-11 px-3 sm:px-3.5 rounded-xl text-[13px] font-medium text-slate-400 bg-white/[0.04] hover:bg-white/[0.08] hover:text-slate-200 active:scale-[0.98] transition-colors duration-200 touch-manipulation whitespace-nowrap"
              >
                Clear<span className="hidden sm:inline"> memory</span>
              </button>
              <button
                type="button"
                onClick={onDisconnect}
                className="h-11 px-3 sm:px-3.5 rounded-xl text-[13px] font-medium text-rose-400/90 bg-rose-500/[0.1] hover:bg-rose-500/20 active:scale-[0.98] transition-colors duration-200 touch-manipulation whitespace-nowrap"
              >
                Stop
              </button>
            </>
          )}
        </div>

        <div className="flex-1 min-w-0 flex justify-end">
          <OverflowMenu
            persona={persona}
            onPersonaChange={onPersonaChange}
            listenOnly={listenOnly}
            onListenOnlyToggle={onListenOnlyToggle}
            onApplyContext={onApplyContext}
            onSettingsClick={onSettingsClick}
            isConnected={isConnected}
            className="shrink-0"
          />
        </div>
      </div>
    </div>
  );
};
