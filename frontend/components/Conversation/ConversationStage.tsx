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
  /** When true, mic is muted so assistant can finish without interruption */
  micMuted?: boolean;
  onMicMutedToggle?: () => void;
  /** When true, voice uses knowledge base (RAG); when false, answers generally */
  voiceUseKnowledgeBase?: boolean;
  onVoiceUseKnowledgeBaseToggle?: () => void;
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
  micMuted = false,
  onMicMutedToggle,
  voiceUseKnowledgeBase = false,
  onVoiceUseKnowledgeBaseToggle,
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
      className="flex flex-col h-full min-h-0 bg-[var(--voice-bg,#0b0e14)] text-[var(--voice-text,#f1f5f9)] overflow-y-auto"
    >
      {/* Context box: compact so Start button stays visible */}
      <div className="shrink-0 border-b border-white/10 p-3 space-y-2">
        <div className="flex items-center justify-between gap-3 rounded-xl bg-white/5 border border-white/10 px-3 py-2">
          <div className="min-w-0">
            <span className="text-sm font-semibold opacity-90">Resource Knowledge Base</span>
            <p className="text-xs opacity-60 mt-0.5 truncate">When on, answers from your documents. When off, answers generally.</p>
          </div>
          {onVoiceUseKnowledgeBaseToggle && (
            <button
              type="button"
              onClick={onVoiceUseKnowledgeBaseToggle}
              className={`w-12 h-7 rounded-full relative transition-colors shrink-0 ${voiceUseKnowledgeBase ? "bg-teal-500" : "bg-slate-700"}`}
            >
              <div className={`absolute top-1 w-5 h-5 rounded-full bg-white shadow transition-all ${voiceUseKnowledgeBase ? "left-6" : "left-1"}`} />
            </button>
          )}
        </div>
        <label className="block text-sm font-semibold opacity-90">Context / Role (System Prompt)</label>
        <textarea
          value={contextValue}
          onChange={(e) => onContextChange(e.target.value)}
          placeholder="Example: You are a car dealership sales agent. Ask 1-2 questions, then recommend a car."
          className="w-full h-20 min-h-[4rem] rounded-xl bg-black/30 border border-white/10 px-3 py-2 text-sm resize-y outline-none focus:border-white/30 placeholder:opacity-50"
        />
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={onApplyContext}
            disabled={!state.isConnected}
            className="rounded-lg px-3 py-1.5 text-sm font-medium bg-white/10 hover:bg-white/15 disabled:opacity-50 disabled:pointer-events-none transition-colors"
          >
            Apply Context
          </button>
          <button
            type="button"
            onClick={onClearMemory}
            disabled={!state.isConnected}
            className="rounded-lg px-3 py-1.5 text-sm font-medium bg-white/10 hover:bg-white/15 disabled:opacity-50 disabled:pointer-events-none transition-colors"
          >
            Clear Memory
          </button>
          <span className="text-xs opacity-60">Context per session. Memory until Clear. Say &quot;listen to conversation&quot; to accumulate; then &quot;now you can speak&quot; or &quot;fact check&quot; to process.</span>
        </div>
      </div>

      {/* Orbs row: min-h-0 so this section shrinks and bottom controls stay visible */}
      <div className="flex-1 min-h-0 flex items-center justify-center gap-8 md:gap-16 py-4">
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
          <>
            <button
              type="button"
              onClick={onMicMutedToggle}
              title={micMuted ? "Unmute mic" : "Mute mic (let assistant finish without interruption)"}
              className={`rounded-xl px-5 py-3 text-sm font-semibold border transition-colors ${
                micMuted
                  ? "bg-amber-500/30 text-amber-300 border-amber-500/40 hover:bg-amber-500/40"
                  : "bg-white/10 text-slate-300 border-white/20 hover:bg-white/15"
              }`}
            >
              {micMuted ? "Unmute" : "Mute"}
            </button>
            <button
              type="button"
              onClick={onDisconnect}
              className="rounded-xl px-6 py-3 text-sm font-semibold bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors"
            >
              Disconnect
            </button>
          </>
        )}
      </div>
    </div>
  );
};
