import { useCallback, useEffect, useRef, useState } from "react";
import { speakAssistantTts } from "../services/backend";

export type SpeakNowDisplayStatus = "ready" | "speaking" | "stopped" | "error";

/**
 * Personal Assistant Speak Now: fetch WAV from voice /tts/speak and play in-browser.
 * Does not use the Conversation WebSocket; safe alongside Voice Conversation when that tab is idle.
 */
export function useSpeakAssistantTts(voiceName: string) {
  const [displayStatus, setDisplayStatus] = useState<SpeakNowDisplayStatus>("ready");
  const [isFetching, setIsFetching] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const revokeUrl = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
  }, []);

  /** Stop audio and in-flight fetch without showing the "Stopped" flash (used before starting a new clip). */
  const interrupt = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    const a = audioRef.current;
    if (a) {
      a.pause();
      a.currentTime = 0;
      a.src = "";
      a.load();
    }
    audioRef.current = null;
    revokeUrl();
    setIsFetching(false);
  }, [revokeUrl]);

  const stop = useCallback(() => {
    interrupt();
    setDisplayStatus((prev) => (prev === "speaking" || prev === "error" || prev === "ready" ? "stopped" : prev));
    setTimeout(() => {
      setDisplayStatus("ready");
      setErrorMessage(null);
    }, 400);
  }, [interrupt]);

  useEffect(() => () => interrupt(), [interrupt]);

  const speak = useCallback(
    async (text: string) => {
      const trimmed = (text || "").trim();
      if (!trimmed) {
        setDisplayStatus("error");
        setErrorMessage("Nothing to speak.");
        return;
      }
      interrupt();
      setErrorMessage(null);
      setDisplayStatus("ready");
      abortRef.current = new AbortController();
      const signal = abortRef.current.signal;
      setIsFetching(true);
      try {
        const blob = await speakAssistantTts(
          { text: trimmed, voice_id: voiceName || undefined },
          signal
        );
        if (signal.aborted) return;
        revokeUrl();
        const url = URL.createObjectURL(blob);
        objectUrlRef.current = url;
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.onended = () => {
          revokeUrl();
          audioRef.current = null;
          setDisplayStatus("ready");
        };
        audio.onerror = () => {
          revokeUrl();
          audioRef.current = null;
          setDisplayStatus("error");
          setErrorMessage("Audio playback failed.");
        };
        setIsFetching(false);
        setDisplayStatus("speaking");
        await audio.play();
      } catch (e) {
        setIsFetching(false);
        if ((e as Error)?.name === "AbortError") return;
        revokeUrl();
        audioRef.current = null;
        setDisplayStatus("error");
        const msg = (e as Error)?.message || "Speak Now failed.";
        if (/Failed to fetch|NetworkError/i.test(msg)) {
          setErrorMessage("Voice service unreachable. Is the voice container running?");
        } else if (/blocked|not allowed|interact/i.test(String(msg))) {
          setErrorMessage("Playback was blocked by the browser. Try clicking Speak Now again.");
        } else {
          setErrorMessage(msg);
        }
      }
    },
    [voiceName, interrupt, revokeUrl]
  );

  const busy = isFetching || displayStatus === "speaking";

  return { speak, stop, displayStatus, errorMessage, busy };
}
