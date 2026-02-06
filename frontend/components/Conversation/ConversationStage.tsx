import React, { useState, useRef, useCallback, useLayoutEffect } from "react";
import { OrbCanvas } from "../OrbVisualizer/OrbCanvas";
import { StatusLabel } from "../UI/StatusLabel";
import type { ConversationState, OrbState } from "./ChatState";

/** Resolve CSS variable (e.g. "var(--assistant-color, #14b8a6)") to hex for canvas. */
function resolveOrbColor(element: HTMLElement | null, cssVar: string, fallbackHex: string): string {
  if (!element) return fallbackHex;
  const match = cssVar.match(/var\s*\(\s*(--[^,]+)\s*,\s*([^)]+)\s*\)/);
  if (!match) return fallbackHex;
  const [, varName, fallback] = match;
  const value = getComputedStyle(element).getPropertyValue(varName).trim();
  if (value && /^#?[0-9A-Fa-f]{6}$/.test(value)) return value.startsWith("#") ? value : `#${value}`;
  const hex = (fallback ?? fallbackHex).trim();
  return /^#?[0-9A-Fa-f]{6}$/.test(hex) ? (hex.startsWith("#") ? hex : `#${hex}`) : fallbackHex;
}

export interface ConversationStageProps {
  /** Current conversation/orb state */
  state: ConversationState;
  /** User mic analyser (for wave ring when user is speaking) */
  userAnalyser: AnalyserNode | null;
  /** Assistant playback analyser (for wave ring when assistant is speaking) */
  assistantAnalyser: AnalyserNode | null;
  /** Context / system prompt */
  contextValue: string;
  onContextChange: (value: string) => void;
  onApplyContext: () => void;
  onClearMemory: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
  /** Connection in progress (e.g. connecting...) */
  connecting?: boolean;
}

const ASSISTANT_COLOR_VAR = "var(--assistant-color, #14b8a6)";
const USER_COLOR_VAR = "var(--user-color, #94a3b8)";

export const ConversationStage: React.FC<ConversationStageProps> = ({
  state,
  userAnalyser,
  assistantAnalyser,
  contextValue,
  onContextChange,
  onApplyContext,
  onClearMemory,
  onConnect,
  onDisconnect,
  connecting = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [resolvedAssistantColor, setResolvedAssistantColor] = useState("#14b8a6");
  const [resolvedUserColor, setResolvedUserColor] = useState("#94a3b8");

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    setResolvedAssistantColor(resolveOrbColor(el, ASSISTANT_COLOR_VAR, "#14b8a6"));
    setResolvedUserColor(resolveOrbColor(el, USER_COLOR_VAR, "#94a3b8"));
  }, []);

  const userActive = state.userOrb === "listening";
  const assistantActive = state.assistantOrb === "speaking" || state.assistantOrb === "thinking";

  return (
    <div
      ref={containerRef}
      className="flex flex-col h-full min-h-0 bg-[var(--voice-bg,#0b0e14)] text-[var(--voice-text,#f1f5f9)]"
    >
      {/* Context box */}
      <div className="shrink-0 border-b border-white/10 p-4">
        <label className="block text-sm font-semibold mb-2 opacity-90">
          Context / Role (System Prompt)
        </label>
        <textarea
          value={contextValue}
          onChange={(e) => onContextChange(e.target.value)}
          placeholder="Example: You are a car dealership sales agent. Ask 1-2 questions, then recommend a car."
          className="w-full h-24 rounded-xl bg-black/30 border border-white/10 px-4 py-3 text-sm resize-y outline-none focus:border-white/30 placeholder:opacity-50"
        />
        <div className="flex flex-wrap items-center gap-2 mt-2">
          <button
            type="button"
            onClick={onApplyContext}
            disabled={!state.isConnected}
            className="rounded-lg px-4 py-2 text-sm font-medium bg-white/10 hover:bg-white/15 disabled:opacity-50 disabled:pointer-events-none transition-colors"
          >
            Apply Context
          </button>
          <button
            type="button"
            onClick={onClearMemory}
            disabled={!state.isConnected}
            className="rounded-lg px-4 py-2 text-sm font-medium bg-white/10 hover:bg-white/15 disabled:opacity-50 disabled:pointer-events-none transition-colors"
          >
            Clear Memory
          </button>
          <span className="text-xs opacity-60">
            Context is applied per session. Memory persists until you Clear Memory.
          </span>
        </div>
      </div>

      {/* Orbs row: fixed min-height so layout never jumps; same structure in all states */}
      <div className="flex-1 flex items-center justify-center gap-12 md:gap-20 min-h-[320px] py-8">
        {/* Assistant orb (left) - fixed width so no reflow */}
        <div className="flex flex-col items-center gap-3 shrink-0 w-[260px]">
          <div className="relative w-[260px] h-[260px] flex items-center justify-center">
            <OrbCanvas
              role="assistant"
              analyserNode={assistantAnalyser}
              isActive={assistantActive}
              isConnected={state.isConnected}
              orbState={state.assistantOrb}
              interruptedAt={state.interruptedAt}
              color={resolvedAssistantColor}
              size={260}
            />
          </div>
          <StatusLabel state={state.assistantOrb} role="assistant" />
          <span className="text-xs opacity-60">EchoMind</span>
        </div>

        {/* User orb (right) - fixed width so no reflow */}
        <div className="flex flex-col items-center gap-3 shrink-0 w-[200px]">
          <div className="relative w-[200px] h-[200px] flex items-center justify-center">
            <OrbCanvas
              role="user"
              analyserNode={userAnalyser}
              isActive={userActive}
              isConnected={state.isConnected}
              orbState={state.userOrb}
              interruptedAt={state.interruptedAt}
              color={resolvedUserColor}
              size={200}
            />
          </div>
          <StatusLabel state={state.userOrb} role="user" />
          <span className="text-xs opacity-60">You</span>
        </div>
      </div>

      {/* Controls */}
      <div className="shrink-0 border-t border-white/10 p-4 flex flex-wrap items-center justify-center gap-3">
        {!state.isConnected ? (
          <button
            type="button"
            onClick={onConnect}
            disabled={connecting}
            className="rounded-xl px-6 py-3 text-sm font-semibold bg-[var(--assistant-color,#00ff9c)] text-black hover:opacity-90 disabled:opacity-50 transition-opacity"
          >
            {connecting ? "Starting…" : "Start"}
          </button>
        ) : (
          <button
            type="button"
            onClick={onDisconnect}
            className="rounded-xl px-6 py-3 text-sm font-semibold bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors"
          >
            Disconnect
          </button>
        )}
      </div>
    </div>
  );
};
