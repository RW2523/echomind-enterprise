import React from "react";
import { VoiceOrb } from "./VoiceOrb";
import type { OrbState } from "./ChatState";

export type VoiceStatus =
  | "offline"
  | "connecting"
  | "ready"
  | "listening"
  | "thinking"
  | "speaking"
  | "muted";

export interface VoiceStatusBarProps {
  assistantOrb: OrbState;
  userOrb: OrbState;
  isConnected: boolean;
  connecting: boolean;
  micMuted: boolean;
  listenOnly: boolean;
  assistantAnalyser: AnalyserNode | null;
  userAnalyser: AnalyserNode | null;
  interruptedAt: number;
  assistantColor: string;
  userColor: string;
  orbSize: number;
  className?: string;
}

const STATUS_LABEL: Record<VoiceStatus, string> = {
  offline: "Not connected",
  connecting: "Connecting",
  ready: "Ready",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
  muted: "Microphone off",
};

export function resolveVoiceStatus(args: {
  isConnected: boolean;
  connecting: boolean;
  micMuted: boolean;
  assistantOrb: OrbState;
  userOrb: OrbState;
}): VoiceStatus {
  const { isConnected, connecting, micMuted, assistantOrb, userOrb } = args;
  if (!isConnected) return connecting ? "connecting" : "offline";
  if (assistantOrb === "speaking" || assistantOrb === "filler") return "speaking";
  if (assistantOrb === "thinking") return "thinking";
  if (micMuted) return "muted";
  if (userOrb === "listening") return "listening";
  return "ready";
}

/**
 * Compact voice-status strip: small orb + a status pill.
 * Sits between the conversation and the controls and stays well under 20vh.
 */
export const VoiceStatusBar: React.FC<VoiceStatusBarProps> = ({
  assistantOrb,
  userOrb,
  isConnected,
  connecting,
  micMuted,
  listenOnly,
  assistantAnalyser,
  userAnalyser,
  interruptedAt,
  assistantColor,
  userColor,
  orbSize,
  className = "",
}) => {
  const status = resolveVoiceStatus({ isConnected, connecting, micMuted, assistantOrb, userOrb });
  const label = STATUS_LABEL[status];

  const dotClass =
    status === "muted"
      ? "bg-rose-400/90"
      : status === "offline"
        ? "bg-slate-600"
        : "bg-accent";
  const pulse =
    status === "listening" || status === "speaking" || status === "thinking" || status === "connecting";

  return (
    <div
      className={`shrink-0 flex flex-col items-center justify-center gap-2 px-4 py-3 ${className}`}
      style={{ maxHeight: "22vh" }}
    >
      <VoiceOrb
        orbState={assistantOrb}
        isConnected={isConnected}
        userOrb={userOrb}
        assistantAnalyser={assistantAnalyser}
        userAnalyser={userAnalyser}
        interruptedAt={interruptedAt}
        assistantColor={assistantColor}
        userColor={userColor}
        size={orbSize}
      />

      <div className="min-w-0 flex flex-col items-center gap-1.5">
        <span
          className="inline-flex items-center gap-2 rounded-full bg-white/[0.05] px-3 py-1.5 text-[12px] font-medium tracking-wide text-slate-300"
          role="status"
          aria-live="polite"
        >
          <span className="relative flex w-2 h-2" aria-hidden>
            {pulse && (
              <span className={`absolute inline-flex w-full h-full rounded-full opacity-60 animate-ping ${dotClass}`} />
            )}
            <span className={`relative inline-flex w-2 h-2 rounded-full ${dotClass}`} />
          </span>
          {label}
        </span>
        {isConnected && listenOnly && (
          <span className="text-[12px] text-slate-500 truncate max-w-[52vw]">
            Say &ldquo;EchoMind&rdquo; to get a reply
          </span>
        )}
      </div>
    </div>
  );
};
