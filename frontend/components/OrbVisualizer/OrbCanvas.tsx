import React, { useRef, useEffect } from "react";
import type { OrbState } from "../Conversation/ChatState";
import { useOrbVisualizer } from "./useOrbVisualizer";

export interface OrbCanvasProps {
  role: "user" | "assistant";
  analyserNode: AnalyserNode | null;
  isActive: boolean;
  isConnected: boolean;
  orbState: OrbState;
  avatarImage?: HTMLImageElement | null;
  interruptedAt?: number;
  color?: string;
  className?: string;
}

const DEFAULT_COLORS = {
  user: "#e2e8f0",
  assistant: "#00ff9c",
};

export const OrbCanvas: React.FC<OrbCanvasProps> = ({
  role,
  analyserNode,
  isActive,
  isConnected,
  orbState,
  avatarImage = null,
  interruptedAt = 0,
  color,
  className = "",
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = 220;
  const fillColor = color ?? DEFAULT_COLORS[role];

  useOrbVisualizer(canvasRef, {
    role,
    analyserNode,
    isActive,
    isConnected,
    orbState,
    avatarImage: avatarImage ?? null,
    interruptedAt,
    color: fillColor,
    size,
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== size * dpr || canvas.height !== size * dpr) {
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      canvas.style.width = `${size}px`;
      canvas.style.height = `${size}px`;
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.scale(dpr, dpr);
    }
  }, [size]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className={`block ${className}`}
      style={{ width: size, height: size }}
      aria-hidden
    />
  );
};
