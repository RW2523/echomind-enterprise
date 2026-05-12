import React, { useState, useRef, useLayoutEffect, useEffect } from "react";
import { TopBar } from "./TopBar";
import { VoiceOrb } from "./VoiceOrb";
import { ControlBar } from "./ControlBar";
import type { ConversationState } from "./ChatState";
import type { AppSettings, DocumentChunk, VoiceMessage } from "../../types";
import { ChunkCitationModal } from "../KnowledgeChat";
import ProductModeHeader, { type ModeStatusTone } from "../ProductModeHeader";
import { CITATION_CHIP_CLASS } from "../../utils/modeChrome";

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
  settings?: AppSettings | null;
  onInterruptAssistant?: () => void;
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
  settings = null,
  onInterruptAssistant,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [resolvedAssistantColor, setResolvedAssistantColor] = useState("#14b8a6");
  const [resolvedUserColor, setResolvedUserColor] = useState("#94a3b8");
  const orbSize = useOrbSize();
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const [citationModal, setCitationModal] = useState<DocumentChunk[] | null>(null);

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    setResolvedAssistantColor(resolveOrbColor(el, ASSISTANT_COLOR_VAR, "#14b8a6"));
    setResolvedUserColor(resolveOrbColor(el, USER_COLOR_VAR, "#94a3b8"));
  }, []);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [voiceMessages, pendingAssistantText]);

  const showTranscript =
    state.isConnected || voiceMessages.length > 0 || !!pendingAssistantText || listenOnly;

  const conversationStatus = (() => {
    if (connecting) return { text: "Connecting…", tone: "thinking" as const };
    if (!state.isConnected) return { text: "Disconnected", tone: "neutral" as const };
    if (micMuted) return { text: "Muted", tone: "muted" as const };
    if (state.assistantOrb === "speaking" || pendingAssistantText) return { text: "Speaking", tone: "speaking" as const };
    if (state.assistantOrb === "thinking") return { text: "Thinking", tone: "thinking" as const };
    if (state.userOrb === "listening" || state.assistantOrb === "listening") return { text: "Listening", tone: "listening" as const };
    if (state.interruptedAt && Date.now() - state.interruptedAt < 1800) return { text: "Interrupted", tone: "muted" as const };
    return { text: "Idle", tone: "neutral" as const };
  })();

  const extraConv: { label: string; tone?: ModeStatusTone }[] = [];
  if (state.isConnected && listenOnly) extraConv.push({ label: "Listen only", tone: "thinking" });

  return (
    <div
      ref={containerRef}
      className="flex flex-col h-full min-h-0 bg-[var(--voice-bg,#0f172a)] text-[var(--voice-text,#f1f5f9)] overflow-hidden"
    >
      {citationModal && citationModal.length > 0 ? (
        <ChunkCitationModal citations={citationModal} onClose={() => setCitationModal(null)} />
      ) : null}
      <TopBar onSettingsClick={onSettingsClick} />
      <ProductModeHeader
        className="border-white/[0.04] bg-black/20 !py-2.5"
        title="Conversation"
        tagline="Real-time voice conversation."
        status={conversationStatus.text}
        statusTone={conversationStatus.tone}
        extraStatuses={extraConv}
        sessionName={null}
        showKnowledge
        knowledgeEnabled={!!settings?.voiceUseKnowledgeBase}
        outputHint="Speech enabled — the assistant replies with voice when connected."
      />

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
              {!listenOnly &&
              state.isConnected &&
              voiceMessages.length === 0 &&
              !pendingAssistantText ? (
                <p className="text-center text-slate-500 text-sm py-8 px-2">Start speaking naturally.</p>
              ) : null}
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
                    className={`max-w-[85%] animate-[fadeIn_0.4s_cubic-bezier(0.25,0.1,0.25,1)] ${
                      msg.role === "user" ? "ml-auto" : "mr-auto"
                    }`}
                  >
                    <div
                      className={`rounded-2xl px-4 py-2.5 text-[15px] ${
                        msg.role === "user"
                          ? "bg-white/[0.06] text-slate-400"
                          : "bg-teal-500/[0.08] text-teal-200/90"
                      }`}
                    >
                      {msg.text}
                    </div>
                    {msg.role === "assistant" && msg.citations && msg.citations.length > 0 ? (
                      <div className="mt-1.5 flex flex-wrap gap-1">
                        {msg.citations.map((c) => {
                          const label =
                            (c.docName.length > 28 ? `${c.docName.slice(0, 26)}…` : c.docName) +
                            (c.metadata.pageNumber != null ? ` · p.${c.metadata.pageNumber}` : "");
                          return (
                            <button
                              key={c.id}
                              type="button"
                              onClick={() => setCitationModal(msg.citations ?? null)}
                              className={CITATION_CHIP_CLASS}
                              title={c.docName}
                            >
                              {label}
                            </button>
                          );
                        })}
                      </div>
                    ) : null}
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
        hasPendingAssistantText={!!pendingAssistantText}
        listenOnly={listenOnly}
        onListenOnlyToggle={onListenOnlyToggle}
        onConnect={onConnect}
        onDisconnect={onDisconnect}
        onMicMutedToggle={onMicMutedToggle ?? (() => {})}
        onClearMemory={onClearMemory}
        onInterrupt={onInterruptAssistant}
      />
    </div>
  );
};
