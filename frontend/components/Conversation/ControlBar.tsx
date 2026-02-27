import React from "react";
import { ICONS } from "../../constants";

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
  className?: string;
}

export const ControlBar: React.FC<ControlBarProps> = ({
  isConnected,
  connecting,
  connectionError,
  micMuted,
  assistantOrb,
  listenOnly = false,
  onListenOnlyToggle,
  onConnect,
  onDisconnect,
  onMicMutedToggle,
  onClearMemory,
  className = "",
}) => {
  return (
    <div
      className={`shrink-0 border-t border-white/[0.04] px-4 py-3 flex flex-col items-center gap-3 ${className}`}
      style={{ paddingBottom: "calc(0.75rem + env(safe-area-inset-bottom))" }}
    >
      {connectionError && (
        <p
          className="text-[13px] text-amber-400/80 text-center max-w-md px-2"
          role="alert"
        >
          {connectionError}
        </p>
      )}
      {isConnected && listenOnly && (
        <p className="text-[12px] text-teal-400/90 font-medium uppercase tracking-wider">
          Listening mode — say &quot;EchoMind&quot; or &quot;now you can speak&quot; to respond
        </p>
      )}
      <div className="flex flex-wrap items-center justify-center gap-2.5">
        {!isConnected ? (
          <button
            type="button"
            onClick={onConnect}
            disabled={connecting}
            className="rounded-2xl px-7 py-3.5 min-h-[48px] text-[15px] font-medium text-slate-900 bg-teal-400/95 hover:bg-teal-400 active:scale-[0.97] disabled:opacity-50 transition-all duration-300 touch-manipulation shadow-[0_4px_24px_-4px_rgba(20,184,166,0.25)]"
            style={{ transitionTimingFunction: "cubic-bezier(0.25, 0.1, 0.25, 1)" }}
          >
            {connecting ? "Starting…" : "Start"}
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={onMicMutedToggle}
              title={micMuted ? "Unmute mic" : "Mute mic"}
              className={`rounded-2xl p-3.5 min-h-[48px] min-w-[48px] flex items-center justify-center transition-all duration-300 touch-manipulation active:scale-[0.97] ${
                micMuted
                  ? "bg-rose-500/15 text-rose-400/90 border border-rose-500/20 hover:bg-rose-500/25"
                  : "bg-teal-500/15 text-teal-400/90 border border-teal-500/20 hover:bg-teal-500/25"
              }`}
              style={{ transitionTimingFunction: "cubic-bezier(0.25, 0.1, 0.25, 1)" }}
            >
              <span className="relative inline-flex items-center justify-center w-6 h-6">
                <ICONS.Mic className="w-5 h-5" strokeWidth={2} />
                {micMuted && (
                  <span
                    className="absolute inset-0 flex items-center justify-center pointer-events-none"
                    aria-hidden
                  >
                    <span className="block w-7 h-0.5 bg-current rounded-full origin-center rotate-45 opacity-90" />
                  </span>
                )}
              </span>
            </button>
            <button
              type="button"
              onClick={onClearMemory}
              className="rounded-2xl px-4 py-3.5 min-h-[48px] text-[15px] font-medium text-slate-500 bg-white/[0.04] border border-white/[0.06] hover:bg-white/[0.08] hover:text-slate-400 active:scale-[0.97] transition-all duration-300 touch-manipulation"
              style={{ transitionTimingFunction: "cubic-bezier(0.25, 0.1, 0.25, 1)" }}
            >
              Clear memory
            </button>
            <button
              type="button"
              onClick={onDisconnect}
              className="rounded-2xl px-4 py-3.5 min-h-[48px] text-[15px] font-medium text-rose-400/80 bg-rose-500/[0.08] border border-rose-500/15 hover:bg-rose-500/15 active:scale-[0.97] transition-all duration-300 touch-manipulation"
              style={{ transitionTimingFunction: "cubic-bezier(0.25, 0.1, 0.25, 1)" }}
            >
              Stop
            </button>
          </>
        )}
      </div>
    </div>
  );
};
