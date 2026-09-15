import React, { useState, useRef, useLayoutEffect, useEffect } from "react";
import { TopBar } from "./TopBar";
import { VoiceStatusBar } from "./VoiceStatusBar";
import { ControlBar } from "./ControlBar";
import { TranscriptPanel } from "./TranscriptPanel";
import type { ConversationState } from "./ChatState";
import { PersonaType } from "../../types";

/**
 * Compact orb size: capped by viewport height so the status strip never
 * takes more than ~20vh, and shrinks further on small screens.
 */
function useOrbSize(): number {
  const [size, setSize] = useState(96);
  useEffect(() => {
    const update = () => {
      const w = typeof window !== "undefined" ? window.innerWidth : 1024;
      const h = typeof window !== "undefined" ? window.innerHeight : 800;
      const byWidth = w >= 1024 ? 104 : w >= 640 ? 92 : 76;
      const byHeight = Math.round(h * 0.11);
      setSize(Math.max(60, Math.min(byWidth, byHeight)));
    };
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return size;
}

/** Read a CSS custom property as a #rrggbb string (accepts hex or an "r g b" triplet). */
function cssVarToHex(element: HTMLElement | null, varName: string): string | null {
  if (!element) return null;
  const raw = getComputedStyle(element).getPropertyValue(varName).trim();
  if (!raw) return null;
  if (/^#?[0-9A-Fa-f]{6}$/.test(raw)) return raw.startsWith("#") ? raw : `#${raw}`;
  const nums = raw.match(/\d{1,3}/g);
  if (nums && nums.length >= 3) {
    return (
      "#" +
      nums
        .slice(0, 3)
        .map((n) => Math.max(0, Math.min(255, Number(n))).toString(16).padStart(2, "0"))
        .join("")
    );
  }
  return null;
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
  /** Streaming partial transcript shown while user is still speaking */
  partialTranscript?: string;
  /** Latest backchannel word from assistant */
  backchannelText?: string;
  /** Secondary controls (••• menu) */
  persona?: PersonaType;
  onPersonaChange?: (persona: PersonaType) => void;
  onApplyContext?: () => void;
  /** Speaker labels in the transcript */
  userLabel?: string;
  assistantLabel?: string;
}

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
  partialTranscript = "",
  backchannelText = "",
  persona,
  onPersonaChange,
  onApplyContext,
  userLabel = "You",
  assistantLabel = "EchoMind",
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [resolvedAssistantColor, setResolvedAssistantColor] = useState("#22d3ee");
  const [resolvedUserColor, setResolvedUserColor] = useState("#94a3b8");
  const orbSize = useOrbSize();

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    setResolvedAssistantColor(
      cssVarToHex(el, "--assistant-color") ?? cssVarToHex(el, "--accent-rgb") ?? "#22d3ee"
    );
    setResolvedUserColor(cssVarToHex(el, "--user-color") ?? "#94a3b8");
  }, []);

  return (
    <div
      ref={containerRef}
      className="flex flex-col h-full min-h-0 bg-[var(--voice-bg,#0f172a)] text-[var(--voice-text,#f1f5f9)] overflow-hidden"
    >
      <TopBar onSettingsClick={onSettingsClick} />

      {/* 1 — Logo + status (top centre) */}
      <VoiceStatusBar
        assistantOrb={state.assistantOrb}
        userOrb={state.userOrb}
        isConnected={state.isConnected}
        connecting={connecting}
        micMuted={micMuted}
        listenOnly={listenOnly}
        assistantAnalyser={assistantAnalyser}
        userAnalyser={userAnalyser}
        interruptedAt={state.interruptedAt}
        assistantColor={resolvedAssistantColor}
        userColor={resolvedUserColor}
        orbSize={orbSize}
        className="border-b border-white/[0.05]"
      />

      {/* 2 — Conversation (hero, scrolls internally) */}
      <TranscriptPanel
        messages={voiceMessages}
        pendingAssistantText={pendingAssistantText}
        partialTranscript={partialTranscript}
        listenOnly={listenOnly}
        listenBufferText={listenBufferText}
        backchannelText={backchannelText}
        isConnected={state.isConnected}
        userLabel={userLabel}
        assistantLabel={assistantLabel}
      />

      {/* 3 — Controls */}
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
        persona={persona}
        onPersonaChange={onPersonaChange}
        onApplyContext={onApplyContext}
        onSettingsClick={onSettingsClick}
      />
    </div>
  );
};
