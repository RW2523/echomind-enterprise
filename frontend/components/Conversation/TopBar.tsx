import React from "react";
import { ICONS } from "../../constants";

export interface TopBarProps {
  onSettingsClick?: () => void;
  className?: string;
}

export const TopBar: React.FC<TopBarProps> = ({
  onSettingsClick,
  className = "",
}) => {
  return (
    <header
      className={`flex items-center justify-between h-12 min-h-[3rem] px-4 shrink-0 border-b border-white/[0.05] ${className}`}
      style={{
        paddingTop: "env(safe-area-inset-top)",
        paddingLeft: "calc(1rem + env(safe-area-inset-left))",
        paddingRight: "calc(1rem + env(safe-area-inset-right))",
      }}
    >
      <span className="text-[13px] font-medium uppercase tracking-wider text-slate-500 truncate">
        Voice
      </span>
      {onSettingsClick && (
        <button
          type="button"
          onClick={onSettingsClick}
          className="shrink-0 p-2 rounded-xl text-slate-500 hover:text-slate-300 hover:bg-white/[0.06] active:scale-[0.98] transition-colors duration-200 touch-manipulation"
          aria-label="Settings"
        >
          <ICONS.Settings className="w-5 h-5" strokeWidth={1.8} />
        </button>
      )}
    </header>
  );
};
