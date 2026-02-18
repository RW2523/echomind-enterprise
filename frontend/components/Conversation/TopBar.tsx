import React from "react";
import { ICONS } from "../../constants";

export interface TopBarProps {
  isConnected: boolean;
  onSettingsClick?: () => void;
  className?: string;
}

export const TopBar: React.FC<TopBarProps> = ({
  isConnected,
  onSettingsClick,
  className = "",
}) => {
  return (
    <header
      className={`flex items-center justify-between h-14 min-h-[3.25rem] px-4 shrink-0 border-b border-white/[0.04] bg-[var(--voice-bg,#0f172a)]/70 backdrop-blur-xl ${className}`}
      style={{ paddingTop: "env(safe-area-inset-top)", paddingLeft: "calc(1rem + env(safe-area-inset-left))", paddingRight: "calc(1rem + env(safe-area-inset-right))" }}
    >
      <div className="flex items-center gap-2.5 min-w-0">
        <span className="text-[19px] font-medium text-white/95 tracking-tight truncate">
          EchoMind
        </span>
        <div className="flex items-center gap-1.5 min-w-0">
          <span
            className={`inline-flex h-1.5 w-1.5 shrink-0 rounded-full transition-colors duration-400 ${
              isConnected ? "bg-emerald-500/90" : "bg-slate-500/80"
            }`}
            aria-hidden
          />
          <span className="text-[13px] text-slate-500 truncate">
            {isConnected ? "Connected" : "Tap to speak"}
          </span>
        </div>
      </div>
      {onSettingsClick && (
        <button
          type="button"
          onClick={onSettingsClick}
          className="shrink-0 p-2 rounded-2xl text-slate-500 hover:text-slate-300 hover:bg-white/[0.06] active:scale-[0.97] transition-all duration-300 touch-manipulation"
          style={{ transitionTimingFunction: "cubic-bezier(0.25, 0.1, 0.25, 1)" }}
          aria-label="Settings"
        >
          <ICONS.Settings className="w-5 h-5" strokeWidth={1.8} />
        </button>
      )}
    </header>
  );
};
