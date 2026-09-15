import React from "react";
import { OrbCanvas } from "../OrbVisualizer/OrbCanvas";
import type { OrbState } from "./ChatState";

const ORB_CENTER_GIF = "/EchoMind_Animation.gif";

export interface VoiceOrbProps {
  orbState: OrbState;
  isConnected: boolean;
  userOrb: OrbState;
  assistantAnalyser: AnalyserNode | null;
  userAnalyser: AnalyserNode | null;
  interruptedAt: number;
  assistantColor: string;
  userColor: string;
  /** Canvas diameter in px. Compact by design — the conversation is the hero. */
  size?: number;
}

/**
 * Compact status visual. Renders at exactly `size` px (plus a thin halo ring),
 * never grows to fill its parent, and still reacts to the audio level.
 */
export const VoiceOrb: React.FC<VoiceOrbProps> = ({
  orbState,
  isConnected,
  userOrb,
  assistantAnalyser,
  userAnalyser: _userAnalyser,
  interruptedAt,
  assistantColor,
  userColor: _userColor,
  size: sizeProp,
}) => {
  const size = Math.max(56, sizeProp ?? 104);
  const ring = size + 12;

  const stateForOrb = orbState === "disconnected" ? "idle" : orbState;
  const isListening = userOrb === "listening";

  return (
    <div
      className="voice-orb-wrapper relative shrink-0 flex items-center justify-center"
      data-state={stateForOrb}
      style={{ width: ring, height: ring }}
    >
      <style>{`
        @keyframes voice-orb-breathe {
          0%, 100% { opacity: 0.85; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.01); }
        }
        @keyframes voice-orb-ring-pulse {
          0%, 100% { transform: scale(1); opacity: 0.35; }
          50% { transform: scale(1.05); opacity: 0.6; }
        }
        @keyframes voice-orb-thinking {
          0%, 100% { transform: scale(1); opacity: 0.4; }
          50% { transform: scale(1.02); opacity: 0.65; }
        }
        @keyframes voice-orb-halo {
          0%, 100% { transform: scale(1); opacity: 0.25; }
          50% { transform: scale(1.07); opacity: 0.45; }
        }
        .voice-orb-wrapper[data-state="idle"] .voice-orb-halo {
          animation: voice-orb-breathe 4s cubic-bezier(0.25, 0.1, 0.25, 1) infinite;
        }
        .voice-orb-wrapper[data-state="listening"] .voice-orb-halo {
          animation: voice-orb-ring-pulse 1.5s cubic-bezier(0.25, 0.1, 0.25, 1) infinite;
        }
        .voice-orb-wrapper[data-state="thinking"] .voice-orb-halo {
          animation: voice-orb-thinking 2.5s cubic-bezier(0.25, 0.1, 0.25, 1) infinite;
        }
        .voice-orb-wrapper[data-state="speaking"] .voice-orb-halo {
          animation: voice-orb-halo 1.6s cubic-bezier(0.25, 0.1, 0.25, 1) infinite;
        }
        .voice-orb-wrapper[data-state="filler"] .voice-orb-halo {
          animation: voice-orb-thinking 1.8s cubic-bezier(0.25, 0.1, 0.25, 1) infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .voice-orb-wrapper .voice-orb-halo { animation: none !important; }
        }
      `}</style>

      <div
        className="voice-orb-halo absolute inset-0 rounded-full border origin-center pointer-events-none"
        style={{ borderColor: `${assistantColor}33` }}
        aria-hidden
      />
      <div
        className="relative rounded-full overflow-hidden transition-shadow duration-500"
        style={{
          width: size,
          height: size,
          boxShadow: isListening
            ? `0 0 24px ${assistantColor}1f`
            : `0 0 14px ${assistantColor}0d`,
        }}
      >
        <OrbCanvas
          role="assistant"
          analyserNode={assistantAnalyser}
          isActive={orbState === "speaking" || orbState === "thinking" || orbState === "filler"}
          isConnected={isConnected}
          orbState={orbState}
          interruptedAt={interruptedAt}
          color={assistantColor}
          size={size}
        />
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div
            className="rounded-full overflow-hidden flex items-center justify-center bg-[var(--voice-bg,#0f172a)]/30"
            style={{ width: size * 0.81, height: size * 0.81 }}
          >
            <img
              src={ORB_CENTER_GIF}
              alt=""
              className="w-full h-full object-cover object-center"
              style={{ aspectRatio: "1" }}
              onError={(e) => {
                (e.target as HTMLImageElement).style.opacity = "0";
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
