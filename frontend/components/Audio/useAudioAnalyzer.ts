import { useRef, useEffect, useState, useCallback } from "react";

const FFT_SIZE = 2048;
const SMOOTHING = 0.25;

export interface UseAudioAnalyzerOptions {
  /** Stream to analyze (e.g. microphone or playback). */
  stream: MediaStream | null;
  enabled: boolean;
}

export function useAudioAnalyzer(options: UseAudioAnalyzerOptions): AnalyserNode | null {
  const { stream, enabled } = options;
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  useEffect(() => {
    if (!stream || !enabled) {
      if (sourceRef.current && audioContextRef.current) {
        try {
          sourceRef.current.disconnect();
        } catch (_) {}
        sourceRef.current = null;
      }
      setAnalyser(null);
      return;
    }

    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    audioContextRef.current = ctx;
    const source = ctx.createMediaStreamSource(stream);
    sourceRef.current = source;

    const analyserNode = ctx.createAnalyser();
    analyserNode.fftSize = FFT_SIZE;
    analyserNode.smoothingTimeConstant = SMOOTHING;
    analyserNode.minDecibels = -90;
    analyserNode.maxDecibels = -10;
    source.connect(analyserNode);

    setAnalyser(analyserNode);

    return () => {
      try {
        source.disconnect();
      } catch (_) {}
      sourceRef.current = null;
      ctx.close().catch(() => {});
      audioContextRef.current = null;
      setAnalyser(null);
    };
  }, [stream, enabled]);

  return analyser;
}

/** Create an analyser from an AudioContext and an existing node (e.g. destination for playback). */
export function usePlaybackAnalyzer(
  audioContext: AudioContext | null,
  /** Callback that receives the destination node so you can connect your buffer source to it. */
  getDestination: (analyser: AnalyserNode) => void,
  enabled: boolean
): AnalyserNode | null {
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const destRef = useRef<AudioDestinationNode | null>(null);

  useEffect(() => {
    if (!audioContext || !enabled) {
      setAnalyser(null);
      return;
    }
    const analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = FFT_SIZE;
    analyserNode.smoothingTimeConstant = SMOOTHING;
    const dest = audioContext.destination;
    destRef.current = dest;
    analyserNode.connect(dest);
    getDestination(analyserNode);
    setAnalyser(analyserNode);
    return () => {
      try {
        analyserNode.disconnect();
      } catch (_) {}
      setAnalyser(null);
    };
  }, [audioContext, enabled]);

  return analyser;
}
