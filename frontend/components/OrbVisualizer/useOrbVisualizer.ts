import { useRef, useEffect, useCallback } from "react";
import type { OrbState } from "../Conversation/ChatState";
import { getOrbStateParams } from "../Conversation/ChatState";
import {
  drawCenterAvatar,
  drawGlowRing,
  drawWaveRing,
  drawOrbitingParticles,
  drawInterruptionRipple,
} from "./orbEffects";

export interface OrbVisualizerParams {
  role: "user" | "assistant";
  analyserNode: AnalyserNode | null;
  isActive: boolean;
  isConnected: boolean;
  orbState: OrbState;
  avatarImage: HTMLImageElement | null;
  interruptedAt: number;
  /** CSS variable for color, e.g. #ffffff or #00ff9c */
  color: string;
  /** Canvas size (width/height). */
  size: number;
}

const SMOOTHING = 0.25;
const CONNECTION_DURATION_MS = 600;
const INTERRUPTION_RIPPLE_MS = 400;

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function useOrbVisualizer(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  params: OrbVisualizerParams
) {
  const {
    role,
    analyserNode,
    isActive,
    isConnected,
    orbState,
    avatarImage,
    interruptedAt,
    color,
    size,
  } = params;

  const timeDomainRef = useRef(new Float32Array(2048));
  const animRef = useRef<number>(0);
  const connectStartRef = useRef<number>(0);
  const interruptStartRef = useRef<number>(0);
  const prevConnectedRef = useRef<boolean>(false);

  const draw = useCallback(
    (time: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const centerX = size / 2;
      const centerY = size / 2;
      const baseRadius = (size / 2) * 0.9;

      if (analyserNode) {
        analyserNode.smoothingTimeConstant = SMOOTHING;
        analyserNode.getFloatTimeDomainData(timeDomainRef.current);
      }

      const stateParams = getOrbStateParams(orbState);

      let scale = 1;
      if (isConnected && !prevConnectedRef.current) {
        connectStartRef.current = time;
      }
      prevConnectedRef.current = isConnected;
      if (isConnected) {
        const elapsed = time - connectStartRef.current;
        const t = Math.min(1, elapsed / CONNECTION_DURATION_MS);
        scale = 0.8 + 0.2 * easeOutCubic(t);
      } else {
        scale = 0.8;
      }

      const radius = baseRadius * scale;

      let rippleProgress = 1;
      if (interruptedAt > 0) {
        if (interruptStartRef.current === 0) interruptStartRef.current = interruptedAt;
        rippleProgress = (time - interruptStartRef.current) / INTERRUPTION_RIPPLE_MS;
        if (rippleProgress >= 1) interruptStartRef.current = 0;
      }

      ctx.save();
      ctx.clearRect(0, 0, size, size);

      const glowIntensity = stateParams.glowIntensity * (isActive ? 1.2 : 0.7);
      const pulse = 1 + 0.04 * Math.sin(time * 0.002 * stateParams.pulseSpeed * 60);
      const r = radius * pulse;

      drawGlowRing(ctx, centerX, centerY, r, glowIntensity, color, stateParams.ringThickness);

      if (rippleProgress < 1) {
        drawInterruptionRipple(ctx, centerX, centerY, r, rippleProgress, color);
      }

      drawWaveRing(
        ctx,
        centerX,
        centerY,
        r,
        timeDomainRef.current,
        stateParams.waveAmplitude,
        color,
        SMOOTHING
      );

      drawOrbitingParticles(
        ctx,
        centerX,
        centerY,
        r,
        stateParams.particleCount,
        time / 1000,
        color,
        orbState
      );

      drawCenterAvatar(ctx, centerX, centerY, r, avatarImage, role, color + "40");

      ctx.restore();
    },
    [
      canvasRef,
      size,
      role,
      analyserNode,
      isActive,
      isConnected,
      orbState,
      avatarImage,
      interruptedAt,
      color,
    ]
  );

  useEffect(() => {
    let running = true;
    const loop = (time: number) => {
      if (!running) return;
      draw(time);
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => {
      running = false;
      cancelAnimationFrame(animRef.current);
    };
  }, [draw]);
}
