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
  size?: number;
}

export const VoiceOrb: React.FC<VoiceOrbProps> = ({
  orbState,
  isConnected,
  userOrb,
  assistantAnalyser,
  userAnalyser,
  interruptedAt,
  assistantColor,
  userColor,
  size: sizeProp,
}) => {
  const size = sizeProp ?? 220;

  const stateForOrb = orbState === "disconnected" ? "idle" : orbState;
  const isListening = userOrb === "listening";

  return (
    <>
      <style>{`
        @keyframes voice-orb-breathe {
          0%, 100% { opacity: 0.92; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.008); }
        }
        @keyframes voice-orb-ring-pulse {
          0%, 100% { transform: scale(1); opacity: 0.35; }
          50% { transform: scale(1.04); opacity: 0.55; }
        }
        @keyframes voice-orb-thinking {
          0%, 100% { transform: scale(1); opacity: 0.5; filter: hue-rotate(0deg); }
          50% { transform: scale(1.015); opacity: 0.7; filter: hue-rotate(-8deg); }
        }
        @keyframes voice-orb-halo {
          0%, 100% { transform: scale(1); opacity: 0.2; }
          50% { transform: scale(1.06); opacity: 0.38; }
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
      `}</style>
      <div className="flex flex-col items-center gap-5 flex-1 min-h-0 justify-center py-5">
        <div
          className="voice-orb-wrapper relative flex items-center justify-center mx-auto"
          data-state={stateForOrb}
          style={{ width: size + 64, height: size + 64 }}
        >
          <div
            className="absolute inset-0 rounded-full pointer-events-none transition-opacity duration-500"
            style={{
              background: `radial-gradient(circle, ${assistantColor}12 0%, transparent 72%)`,
            }}
            aria-hidden
          />
          <div
            className="absolute"
            style={{
              width: size + 20,
              height: size + 20,
              top: "50%",
              left: "50%",
              marginTop: -(size + 20) / 2,
              marginLeft: -(size + 20) / 2,
            }}
            aria-hidden
          >
            <div className="voice-orb-halo rounded-full border border-teal-500/20 w-full h-full origin-center" />
          </div>
          <div
            className="relative rounded-full overflow-hidden transition-shadow duration-500"
            style={{
              width: size,
              height: size,
              boxShadow: isListening
                ? `0 0 32px ${assistantColor}18`
                : `0 0 20px ${assistantColor}0d`,
            }}
          >
            <OrbCanvas
              role="assistant"
              analyserNode={assistantAnalyser}
              isActive={orbState === "speaking" || orbState === "thinking"}
              isConnected={isConnected}
              orbState={orbState}
              interruptedAt={interruptedAt}
              color={assistantColor}
              size={size}
            />
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div
                className="rounded-full overflow-hidden flex items-center justify-center bg-[var(--voice-bg,#0f172a)]/30"
                style={{
                  width: size * 0.81,
                  height: size * 0.81,
                }}
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
      </div>
    </>
  );
};
