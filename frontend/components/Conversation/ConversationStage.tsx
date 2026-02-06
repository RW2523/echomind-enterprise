import React from "react";
import { VoiceActivityBars } from "../UI/VoiceActivityBars";
import type { ConversationState, OrbState } from "./ChatState";

const STATE_LABELS: Record<OrbState, string> = {
  idle: "Idle",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking",
  interrupted: "Interrupted",
  disconnected: "Disconnected",
};

function StatusPill({ state }: { state: OrbState }) {
  const active = state === "listening" || state === "speaking" || state === "thinking";
  const disconnected = state === "disconnected";
  return (
    <span
      className={`
        inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium
        ${disconnected ? "bg-white/10 text-white/60" : active ? "bg-emerald-500/20 text-emerald-300" : "bg-white/10 text-white/80"}
      `}
      data-state={state}
    >
      {!disconnected && (
        <span
          className={`w-1.5 h-1.5 rounded-full ${active ? "bg-emerald-400 animate-pulse" : "bg-white/50"}`}
        />
      )}
      {STATE_LABELS[state]}
    </span>
  );
}

export interface ConversationStageProps {
  state: ConversationState;
  userAnalyser: AnalyserNode | null;
  assistantAnalyser: AnalyserNode | null;
  contextValue: string;
  onContextChange: (value: string) => void;
  onApplyContext: () => void;
  onClearMemory: () => void;
  onConnect: () => void;
  onDisconnect: () => void;
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
  connecting = false,
}) => {
  const userActive = state.userOrb === "listening";
  const assistantActive = state.assistantOrb === "speaking" || state.assistantOrb === "thinking";

  return (
    <div className="flex flex-col h-full min-h-0 bg-[#0c1222] text-slate-100">
      {/* Status bar */}
      <div className="shrink-0 flex items-center justify-between px-4 py-2.5 border-b border-white/10 bg-black/20">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
          Voice session
        </span>
        <span className="inline-flex items-center gap-2 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${state.isConnected ? "bg-emerald-500" : "bg-slate-500"}`}
          />
          {state.isConnected ? "Connected" : "Disconnected"}
        </span>
      </div>

      {/* Context */}
      <div className="shrink-0 border-b border-white/10 p-4 bg-black/10">
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Context / role
        </label>
        <textarea
          value={contextValue}
          onChange={(e) => onContextChange(e.target.value)}
          placeholder="e.g. You are a helpful assistant. Be concise."
          className="w-full h-20 rounded-lg bg-white/5 border border-white/10 px-3 py-2.5 text-sm text-slate-200 placeholder:text-slate-500 focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/30 outline-none resize-none transition-colors"
        />
        <div className="flex flex-wrap items-center gap-2 mt-2">
          <button
            type="button"
            onClick={onApplyContext}
            disabled={!state.isConnected}
            className="rounded-lg px-3 py-1.5 text-xs font-medium bg-white/10 text-slate-300 hover:bg-white/15 disabled:opacity-40 disabled:pointer-events-none transition-colors"
          >
            Apply context
          </button>
          <button
            type="button"
            onClick={onClearMemory}
            disabled={!state.isConnected}
            className="rounded-lg px-3 py-1.5 text-xs font-medium bg-white/10 text-slate-300 hover:bg-white/15 disabled:opacity-40 disabled:pointer-events-none transition-colors"
          >
            Clear memory
          </button>
          <span className="text-xs text-slate-500">Memory persists until cleared.</span>
        </div>
      </div>

      {/* Participants: two cards */}
      <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-2 gap-4 p-4 overflow-auto">
        {/* EchoMind card */}
        <div
          className={`
            flex flex-col rounded-xl border bg-black/20 overflow-hidden
            ${assistantActive ? "border-emerald-500/40 bg-emerald-950/20" : "border-white/10"}
          `}
        >
          <div className="shrink-0 flex items-center justify-between px-4 py-3 border-b border-white/10">
            <span className="text-sm font-semibold text-slate-200">EchoMind</span>
            <StatusPill state={state.assistantOrb} />
          </div>
          <div className="flex-1 flex flex-col items-center justify-center min-h-[120px] p-4">
            <VoiceActivityBars
              analyser={assistantAnalyser}
              isActive={assistantActive && state.isConnected}
              color="#34d399"
              className="mb-2"
            />
            <span className="text-xs text-slate-500">
              {state.isConnected ? "Assistant audio level" : "Not connected"}
            </span>
          </div>
        </div>

        {/* You card */}
        <div
          className={`
            flex flex-col rounded-xl border bg-black/20 overflow-hidden
            ${userActive ? "border-sky-500/40 bg-sky-950/20" : "border-white/10"}
          `}
        >
          <div className="shrink-0 flex items-center justify-between px-4 py-3 border-b border-white/10">
            <span className="text-sm font-semibold text-slate-200">You</span>
            <StatusPill state={state.userOrb} />
          </div>
          <div className="flex-1 flex flex-col items-center justify-center min-h-[120px] p-4">
            <VoiceActivityBars
              analyser={userAnalyser}
              isActive={userActive && state.isConnected}
              color="#38bdf8"
              className="mb-2"
            />
            <span className="text-xs text-slate-500">
              {state.isConnected ? "Microphone level" : "Not connected"}
            </span>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="shrink-0 border-t border-white/10 px-4 py-4 bg-black/20">
        <div className="flex justify-center">
          {!state.isConnected ? (
            <button
              type="button"
              onClick={onConnect}
              disabled={connecting}
              className="rounded-lg px-8 py-3 text-sm font-semibold bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg shadow-emerald-900/30"
            >
              {connecting ? "Connecting…" : "Start"}
            </button>
          ) : (
            <button
              type="button"
              onClick={onDisconnect}
              className="rounded-lg px-8 py-3 text-sm font-semibold bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25 transition-colors"
            >
              Disconnect
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
