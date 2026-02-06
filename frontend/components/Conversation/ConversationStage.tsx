import React, { useState, useRef, useCallback } from "react";
import { OrbCanvas } from "../OrbVisualizer/OrbCanvas";
import { StatusLabel } from "../UI/StatusLabel";
import type { ConversationState, OrbState } from "./ChatState";

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
  /** Demo: simulate user speaking */
  onDemoUserSpeak?: () => void;
  /** Demo: simulate assistant speaking */
  onDemoAssistantSpeak?: () => void;
  /** Demo: simulate interrupt */
  onDemoInterrupt?: () => void;
  /** Show demo buttons (for testing without real connection) */
  showDemoControls?: boolean;
  /** Connection in progress (e.g. connecting...) */
  connecting?: boolean;
}

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
  onDemoUserSpeak,
  onDemoAssistantSpeak,
  onDemoInterrupt,
  showDemoControls = false,
  connecting = false,
}) => {
  const userActive = state.userOrb === "listening";
  const assistantActive = state.assistantOrb === "speaking" || state.assistantOrb === "thinking";

  return (
    <div className="flex flex-col h-full min-h-0 bg-[var(--voice-bg,#0b0e14)] text-[var(--voice-text,#f1f5f9)]">
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

      {/* Orbs row: AI left (larger), You right - main focus */}
      <div className="flex-1 flex items-center justify-center gap-12 md:gap-20 min-h-0 py-8">
        {/* Assistant orb (left, larger) */}
        <div className="flex flex-col items-center gap-3">
          <div className="relative">
            <OrbCanvas
              role="assistant"
              analyserNode={assistantAnalyser}
              isActive={assistantActive}
              isConnected={state.isConnected}
              orbState={state.assistantOrb}
              interruptedAt={state.interruptedAt}
              color="var(--assistant-color, #14b8a6)"
              size={260}
            />
          </div>
          <StatusLabel state={state.assistantOrb} role="assistant" />
          <span className="text-xs opacity-60">EchoMind</span>
        </div>

        {/* User orb (right, smaller) */}
        <div className="flex flex-col items-center gap-3">
          <div className="relative">
            <OrbCanvas
              role="user"
              analyserNode={userAnalyser}
              isActive={userActive}
              isConnected={state.isConnected}
              orbState={state.userOrb}
              interruptedAt={state.interruptedAt}
              color="var(--user-color, #94a3b8)"
              size={200}
            />
          </div>
          <StatusLabel state={state.userOrb} role="user" />
          <span className="text-xs opacity-60">You</span>
        </div>
      </div>

      {/* Intro tip: unmute and speak (below orbs, cleared on first audio or timeout) */}
      {state.showIntroTip && state.isConnected && (
        <div className="shrink-0 text-center py-1 text-sm text-[var(--voice-text,#f1f5f9)] opacity-75">
          Unmute and say something to start.
        </div>
      )}

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

        {showDemoControls && (
          <>
            <button
              type="button"
              onClick={onDemoUserSpeak}
              className="rounded-lg px-4 py-2 text-sm bg-white/10 hover:bg-white/15"
            >
              User Speak
            </button>
            <button
              type="button"
              onClick={onDemoAssistantSpeak}
              className="rounded-lg px-4 py-2 text-sm bg-white/10 hover:bg-white/15"
            >
              Assistant Speak
            </button>
            <button
              type="button"
              onClick={onDemoInterrupt}
              className="rounded-lg px-4 py-2 text-sm bg-amber-500/20 text-amber-400 border border-amber-500/30"
            >
              Interrupt
            </button>
          </>
        )}
      </div>
    </div>
  );
};
