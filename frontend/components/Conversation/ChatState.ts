/**
 * Orb state model for conversation UI.
 * State affects glow intensity, ring thickness, animation speed, particle count.
 */
export type OrbState =
  | "idle"
  | "listening"
  | "thinking"
  | "speaking"
  | "interrupted"
  | "disconnected";

export type OrbRole = "user" | "assistant";

export interface ConversationState {
  userOrb: OrbState;
  assistantOrb: OrbState;
  isConnected: boolean;
  /** Last interruption timestamp for ripple effect */
  interruptedAt: number;
  /** Show "Unmute and speak to start" briefly after connect (cleared on first audio or timeout) */
  showIntroTip: boolean;
}

export const initialConversationState: ConversationState = {
  userOrb: "disconnected",
  assistantOrb: "disconnected",
  isConnected: false,
  interruptedAt: 0,
  showIntroTip: false,
};

export function getOrbStateParams(state: OrbState): {
  glowIntensity: number;
  ringThickness: number;
  waveAmplitude: number;
  particleCount: number;
  pulseSpeed: number;
  rotationSpeed: number;
} {
  switch (state) {
    case "idle":
      return { glowIntensity: 0.18, ringThickness: 2, waveAmplitude: 0, particleCount: 6, pulseSpeed: 0, rotationSpeed: 0.08 };
    case "listening":
      return { glowIntensity: 0.4, ringThickness: 2.5, waveAmplitude: 0.06, particleCount: 10, pulseSpeed: 0.5, rotationSpeed: 0.12 };
    case "thinking":
      return { glowIntensity: 0.3, ringThickness: 2, waveAmplitude: 0.02, particleCount: 8, pulseSpeed: 0.3, rotationSpeed: 0.1 };
    case "speaking":
      return { glowIntensity: 0.45, ringThickness: 2.5, waveAmplitude: 0.08, particleCount: 12, pulseSpeed: 0.6, rotationSpeed: 0.12 };
    case "interrupted":
      return { glowIntensity: 0.5, ringThickness: 3, waveAmplitude: 0.05, particleCount: 8, pulseSpeed: 0.5, rotationSpeed: 0.1 };
    case "disconnected":
      return { glowIntensity: 0.08, ringThickness: 1.5, waveAmplitude: 0, particleCount: 4, pulseSpeed: 0, rotationSpeed: 0 };
    default:
      return getOrbStateParams("idle");
  }
}
