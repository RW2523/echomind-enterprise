import React, { useState, useRef, useLayoutEffect, useEffect } from "react";
import { TopBar } from "./TopBar";
import { VoiceOrb } from "./VoiceOrb";
import { ControlBar } from "./ControlBar";
import type { ConversationState } from "./ChatState";

function useOrbSize(): number {
  const [size, setSize] = useState(200);
  useEffect(() => {
    const update = () => {
      const w = typeof window !== "undefined" ? window.innerWidth : 768;
      if (w >= 768) setSize(240);
      else if (w >= 640) setSize(200);
      else setSize(168);
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return size;
}

function resolveOrbColor(element: HTMLElement | null, cssVar: string, fallbackHex: string): string {
  if (!element) return fallbackHex;
  const match = cssVar.match(/var\s*\(\s*(--[^,]+)\s*,\s*([^)]+)\s*\)/);
  if (!match) return fallbackHex;
  const [, varName] = match;
  const value = getComputedStyle(element).getPropertyValue(varName).trim();
  if (value && /^#?[0-9A-Fa-f]{6}$/.test(value)) return value.startsWith("#") ? value : `#${value}`;
  const hex = fallbackHex.trim();
  return /^#?[0-9A-Fa-f]{6}$/.test(hex) ? (hex.startsWith("#") ? hex : `#${hex}`) : fallbackHex;
}

export interface VoiceMessage {
  role: "user" | "assistant";
  text: string;
}

export interface ConversationStageProps {
  state: ConversationState;
  userAnalyser: AnalyserNode | null;
  assistantAnalyser: AnalyserNode | null;
  voiceMessages?: VoiceMessage[];
  pendingAssistantText?: string;
  /** Accumulated transcript in listen-only mode (live updates until wake word) */
  listenBufferText?: string;
  connectionError?: string | null;
  onClearMemory: () => void;
  /** Continuous listening: only respond after wake word "EchoMind" */
  listenOnly?: boolean;
  onListenOnlyToggle?: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
  connecting?: boolean;
  micMuted?: boolean;
  onMicMutedToggle?: () => void;
  onSettingsClick?: () => void;
}

const ASSISTANT_COLOR_VAR = "var(--assistant-color, #14b8a6)";
const USER_COLOR_VAR = "var(--user-color, #94a3b8)";

export const ConversationStage: React.FC<ConversationStageProps> = ({
  state,
  userAnalyser,
  assistantAnalyser,
  voiceMessages = [],
  pendingAssistantText = "",
  listenBufferText = "",
  connectionError = null,
  onClearMemory,
  listenOnly = false,
  onListenOnlyToggle,
  onConnect,
  onDisconnect,
  connecting = false,
  micMuted = false,
  onMicMutedToggle,
  onSettingsClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [resolvedAssistantColor, setResolvedAssistantColor] = useState("#14b8a6");
  const [resolvedUserColor, setResolvedUserColor] = useState("#94a3b8");
  const orbSize = useOrbSize();
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    setResolvedAssistantColor(resolveOrbColor(el, ASSISTANT_COLOR_VAR, "#14b8a6"));
    setResolvedUserColor(resolveOrbColor(el, USER_COLOR_VAR, "#94a3b8"));
  }, []);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [voiceMessages, pendingAssistantText]);

  const showTranscript = voiceMessages.length > 0 || !!pendingAssistantText || listenOnly;

  return (
    <div
      ref={containerRef}
      className="flex flex-col h-full min-h-0 bg-[var(--voice-bg,#0f172a)] text-[var(--voice-text,#f1f5f9)] overflow-hidden"
    >
      <TopBar onSettingsClick={onSettingsClick} />

      <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
        <VoiceOrb
          orbState={state.assistantOrb}
          isConnected={state.isConnected}
          userOrb={state.userOrb}
          assistantAnalyser={assistantAnalyser}
          userAnalyser={userAnalyser}
          interruptedAt={state.interruptedAt}
          assistantColor={resolvedAssistantColor}
          userColor={resolvedUserColor}
          size={orbSize}
        />

        {showTranscript && (
          <div className="shrink-0 flex flex-col max-h-[36vh] min-h-0 border-t border-white/[0.04]">
            <div className="px-4 py-2.5 text-[13px] font-medium text-slate-500 uppercase tracking-wider">
              {listenOnly ? "Listening — say EchoMind when done" : "Live transcript"}
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto px-4 pb-3 space-y-1.5">
              {listenOnly && listenBufferText ? (
                <div className="rounded-2xl px-4 py-3 text-[15px] bg-white/[0.06] text-slate-300 border border-white/10 whitespace-pre-wrap break-words">
                  {listenBufferText}
                  <span className="inline-block w-2 h-4 ml-1 bg-teal-400/80 rounded-sm animate-pulse align-middle" aria-hidden />
                </div>
              ) : null}
              {!listenOnly &&
                voiceMessages.map((msg, i) => (
                  <div
                    key={i}
                    className={`rounded-2xl px-4 py-2.5 text-[15px] max-w-[85%] animate-[fadeIn_0.4s_cubic-bezier(0.25,0.1,0.25,1)] ${
                      msg.role === "user"
                        ? "ml-auto bg-white/[0.06] text-slate-400"
                        : "mr-auto bg-teal-500/[0.08] text-teal-200/90"
                    }`}
                  >
                    {msg.text}
                  </div>
                ))}
              {pendingAssistantText && (
                <div className="mr-auto rounded-2xl px-4 py-2.5 text-[15px] max-w-[85%] bg-teal-500/[0.08] text-teal-200/90 border border-teal-500/15 animate-[fadeIn_0.4s_cubic-bezier(0.25,0.1,0.25,1)]">
                  {pendingAssistantText}
                  <span className="inline-block w-2 h-4 ml-1 bg-current animate-pulse rounded-sm opacity-80" aria-hidden />
                </div>
              )}
              <div ref={transcriptEndRef} />
            </div>
          </div>
        )}
      </div>

      <ControlBar
        isConnected={state.isConnected}
        connecting={connecting}
        connectionError={connectionError}
        micMuted={micMuted}
        assistantOrb={state.assistantOrb}
        listenOnly={listenOnly}
        onListenOnlyToggle={onListenOnlyToggle}
        onConnect={onConnect}
        onDisconnect={onDisconnect}
        onMicMutedToggle={onMicMutedToggle ?? (() => {})}
        onClearMemory={onClearMemory}
      />
    </div>
  );
};
