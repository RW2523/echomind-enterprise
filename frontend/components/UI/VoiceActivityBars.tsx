import React, { useRef, useEffect } from "react";

const BAR_COUNT = 6;
const SMOOTHING = 0.6;

export interface VoiceActivityBarsProps {
  analyser: AnalyserNode | null;
  isActive: boolean;
  color?: string;
  className?: string;
}

export const VoiceActivityBars: React.FC<VoiceActivityBarsProps> = ({
  analyser,
  isActive,
  color = "currentColor",
  className = "",
}) => {
  const barRefs = useRef<(HTMLDivElement | null)[]>([]);
  const heightsRef = useRef<number[]>(new Array(BAR_COUNT).fill(0));

  useEffect(() => {
    if (!analyser) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    let rafId: number;

    const update = () => {
      analyser.getByteFrequencyData(data);
      const step = Math.floor(data.length / BAR_COUNT);
      for (let i = 0; i < BAR_COUNT; i++) {
        const slice = data.slice(i * step, (i + 1) * step);
        let sum = 0;
        for (let j = 0; j < slice.length; j++) sum += slice[j];
        const raw = slice.length ? sum / slice.length : 0;
        const normalized = Math.min(1, raw / 128);
        heightsRef.current[i] =
          heightsRef.current[i] * SMOOTHING + normalized * (1 - SMOOTHING);
      }
      barRefs.current.forEach((el, i) => {
        if (el) {
          const h = Math.max(4, heightsRef.current[i] * 24);
          el.style.height = `${h}px`;
        }
      });
      rafId = requestAnimationFrame(update);
    };
    rafId = requestAnimationFrame(update);
    return () => cancelAnimationFrame(rafId);
  }, [analyser]);

  if (!analyser || !isActive) {
    return (
      <div
        className={`flex items-end justify-center gap-0.5 h-7 ${className}`}
        aria-hidden
      >
        {Array.from({ length: BAR_COUNT }).map((_, i) => (
          <div
            key={i}
            className="w-1 rounded-full bg-white/20 min-h-[4px] transition-all duration-300"
            style={{ height: 4 }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={`flex items-end justify-center gap-0.5 h-7 ${className}`}
      aria-hidden
    >
      {Array.from({ length: BAR_COUNT }).map((_, i) => (
        <div
          key={i}
          ref={(el) => { barRefs.current[i] = el; }}
          className="w-1 rounded-full min-h-[4px] transition-all duration-75 ease-out"
          style={{ height: 4, backgroundColor: color }}
        />
      ))}
    </div>
  );
};
