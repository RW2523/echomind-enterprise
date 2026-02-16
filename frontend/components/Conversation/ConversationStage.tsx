import React, { useState, useRef, useLayoutEffect, useEffect } from "react";
import { OrbCanvas } from "../OrbVisualizer/OrbCanvas";
import { StatusLabel } from "../UI/StatusLabel";
import { ICONS } from "../../constants";
import type { ConversationState } from "./ChatState";

function useOrbSizes(): { assistant: number; user: number } {
  const [sizes, setSizes] = useState({ assistant: 180, user: 150 });
  useEffect(() => {
    const update = () => {
      const w = typeof window !== "undefined" ? window.innerWidth : 768;
      if (w >= 768) setSizes({ assistant: 260, user: 200 });
      else if (w >= 640) setSizes({ assistant: 180, user: 150 });
      else setSizes({ assistant: 140, user: 120 });
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return sizes;
}

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

export interface VoiceMessage {
  role: "user" | "assistant";
  text: string;
}

export interface ConversationStageProps {
  /** Current conversation/orb state */
  state: ConversationState;
  /** User mic analyser (for wave ring when user is speaking) */
  userAnalyser: AnalyserNode | null;
  /** Assistant playback analyser (for wave ring when assistant is speaking) */
  assistantAnalyser: AnalyserNode | null;
  /** Live transcript: what you said and what the assistant replied */
  voiceMessages?: VoiceMessage[];
  /** Current assistant reply while streaming */
  pendingAssistantText?: string;
  /** Error starting mic or connection; shown near Start button */
  connectionError?: string | null;
  onClearMemory: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
  /** Connection in progress (e.g. connecting...) */
  connecting?: boolean;
  /** When true, mic is muted so assistant can finish without interruption */
  micMuted?: boolean;
  onMicMutedToggle?: () => void;
}

const ASSISTANT_COLOR_VAR = "var(--assistant-color, #14b8a6)";
const USER_COLOR_VAR = "var(--user-color, #94a3b8)";

export const ConversationStage: React.FC<ConversationStageProps> = ({
  state,
  userAnalyser,
  assistantAnalyser,
  voiceMessages = [],
  pendingAssistantText = "",
  connectionError = null,
  onClearMemory,
  onConnect,
  onDisconnect,
  connecting = false,
  micMuted = false,
  onMicMutedToggle,
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
  const orbSizes = useOrbSizes();
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [voiceMessages, pendingAssistantText]);

  return (
    <div
      ref={containerRef}
      className="flex flex-col h-full min-h-0 bg-[var(--voice-bg,#0b0e14)] text-[var(--voice-text,#f1f5f9)] overflow-y-auto"
    >
      {/* Orbs: stacked on small screens, side-by-side on larger; responsive sizes */}
      <div className="flex-1 min-h-0 flex flex-col sm:flex-row items-center justify-center gap-4 sm:gap-6 md:gap-12 py-3 sm:py-4 overflow-auto">
        {/* Assistant orb */}
        <div className="flex flex-col items-center gap-2 sm:gap-3 shrink-0" style={{ width: orbSizes.assistant }}>
          <div className="relative flex items-center justify-center" style={{ width: orbSizes.assistant, height: orbSizes.assistant }}>
            <OrbCanvas
              role="assistant"
              analyserNode={assistantAnalyser}
              isActive={assistantActive}
              isConnected={state.isConnected}
              orbState={state.assistantOrb}
              interruptedAt={state.interruptedAt}
              color={resolvedAssistantColor}
              size={orbSizes.assistant}
            />
          </div>
          <StatusLabel state={state.assistantOrb} role="assistant" />
          <span className="text-xs opacity-60">EchoMind</span>
        </div>

        {/* User orb */}
        <div className="flex flex-col items-center gap-2 sm:gap-3 shrink-0" style={{ width: orbSizes.user }}>
          <div className="relative flex items-center justify-center" style={{ width: orbSizes.user, height: orbSizes.user }}>
            <OrbCanvas
              role="user"
              analyserNode={userAnalyser}
              isActive={userActive}
              isConnected={state.isConnected}
              orbState={state.userOrb}
              interruptedAt={state.interruptedAt}
              color={resolvedUserColor}
              size={orbSizes.user}
            />
          </div>
          <StatusLabel state={state.userOrb} role="user" />
          <span className="text-xs opacity-60">You</span>
        </div>
      </div>

      {/* Live transcript: what you said and what the assistant replied */}
      {(voiceMessages.length > 0 || pendingAssistantText) && (
        <div className="shrink-0 flex flex-col max-h-[40vh] min-h-0 border-t border-white/10">
          <div className="px-3 py-2 text-xs font-medium text-white/60 uppercase tracking-wider">
            Live transcript
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto px-3 pb-3 space-y-2">
            {voiceMessages.map((msg, i) => (
              <div
                key={i}
                className={`rounded-lg px-3 py-2 text-sm max-w-[85%] ${
                  msg.role === "user"
                    ? "ml-auto bg-[var(--user-color,#3b82f6)]/20 text-[var(--user-color,#93c5fd)]"
                    : "mr-auto bg-[var(--assistant-color,#00ff9c)]/10 text-[var(--assistant-color,#5eead4)]"
                }`}
              >
                {msg.text}
              </div>
            ))}
            {pendingAssistantText && (
              <div className="mr-auto rounded-lg px-3 py-2 text-sm max-w-[85%] bg-[var(--assistant-color,#00ff9c)]/10 text-[var(--assistant-color,#5eead4)] border border-[var(--assistant-color,#00ff9c)]/30">
                {pendingAssistantText}
                <span className="inline-block w-2 h-4 ml-0.5 bg-current animate-pulse" aria-hidden />
              </div>
            )}
            <div ref={transcriptEndRef} />
          </div>
        </div>
      )}

      {/* Controls - touch-friendly on mobile */}
      <div className="shrink-0 border-t border-white/10 p-3 sm:p-4 flex flex-col items-center gap-2 sm:gap-3">
        {connectionError && (
          <p className="text-sm text-amber-400/95 text-center max-w-md" role="alert">
            {connectionError}
          </p>
        )}
        <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3">
        {!state.isConnected ? (
          <button
            type="button"
            onClick={onConnect}
            disabled={connecting}
            className="rounded-xl px-6 py-3 min-h-[44px] text-sm font-semibold bg-[var(--assistant-color,#00ff9c)] text-black hover:opacity-90 disabled:opacity-50 transition-opacity touch-manipulation"
          >
            {connecting ? "Starting…" : "Start"}
          </button>
        ) : (
          <>
            <button
              type="button"
              onClick={onMicMutedToggle}
              title={micMuted ? "Unmute mic" : "Mute mic (let assistant finish without interruption)"}
              className={`rounded-xl p-3 min-h-[44px] min-w-[44px] border transition-colors flex items-center justify-center touch-manipulation ${
                micMuted
                  ? "bg-red-500/90 text-white border-red-400 hover:bg-red-500"
                  : "bg-emerald-500/90 text-white border-emerald-400 hover:bg-emerald-500"
              }`}
            >
              <span className="relative inline-flex items-center justify-center w-8 h-8">
                <ICONS.Mic className="w-6 h-6 stroke-[2.5]" stroke="currentColor" />
                {micMuted && (
                  <span
                    className="absolute inset-0 flex items-center justify-center pointer-events-none"
                    aria-hidden
                  >
                    <span className="block w-10 h-0.5 bg-white rounded-full origin-center rotate-45" />
                  </span>
                )}
              </span>
            </button>
            <button
              type="button"
              onClick={onClearMemory}
              disabled={!state.isConnected}
              className="rounded-xl px-4 py-3 min-h-[44px] text-sm font-medium bg-white/10 text-slate-300 border border-white/20 hover:bg-white/15 transition-colors touch-manipulation"
            >
              Clear Memory
            </button>
            <button
              type="button"
              onClick={onDisconnect}
              className="rounded-xl px-5 py-3 min-h-[44px] text-sm font-semibold bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors touch-manipulation"
            >
              Disconnect
            </button>
          </>
        )}
        </div>
      </div>
    </div>
  );
};
