import React from "react";
import type { OrbState } from "../Conversation/ChatState";

export interface StatusLabelProps {
  state: OrbState;
  role: "user" | "assistant";
  className?: string;
}

const STATE_LABELS: Record<OrbState, string> = {
  idle: "Idle",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking",
  interrupted: "Interrupted",
  disconnected: "Disconnected",
};

export const StatusLabel: React.FC<StatusLabelProps> = ({ state, role, className = "" }) => {
  const label = STATE_LABELS[state];
  return (
    <span
      className={`inline-block text-xs font-medium opacity-80 min-w-[5.5rem] text-center ${className}`}
      data-state={state}
    >
      {label}
    </span>
  );
};
