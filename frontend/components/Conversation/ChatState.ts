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
}

export const initialConversationState: ConversationState = {
  userOrb: "disconnected",
  assistantOrb: "disconnected",
  isConnected: false,
  interruptedAt: 0,
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
      return { glowIntensity: 0.15, ringThickness: 2, waveAmplitude: 0.02, particleCount: 8, pulseSpeed: 0.8, rotationSpeed: 0 };
    case "listening":
      return { glowIntensity: 0.5, ringThickness: 3, waveAmplitude: 0.12, particleCount: 16, pulseSpeed: 2, rotationSpeed: 0 };
    case "thinking":
      return { glowIntensity: 0.35, ringThickness: 2.5, waveAmplitude: 0.04, particleCount: 12, pulseSpeed: 1.2, rotationSpeed: 0.15 };
    case "speaking":
      return { glowIntensity: 0.55, ringThickness: 3.5, waveAmplitude: 0.18, particleCount: 20, pulseSpeed: 2.5, rotationSpeed: 0.05 };
    case "interrupted":
      return { glowIntensity: 0.6, ringThickness: 4, waveAmplitude: 0.2, particleCount: 14, pulseSpeed: 3, rotationSpeed: 0 };
    case "disconnected":
      return { glowIntensity: 0.08, ringThickness: 1.5, waveAmplitude: 0, particleCount: 4, pulseSpeed: 0.3, rotationSpeed: 0 };
    default:
      return getOrbStateParams("idle");
  }
}
