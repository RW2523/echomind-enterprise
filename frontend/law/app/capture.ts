/**
 * Whether this page is capturing audio right now, for the always-visible recording status (docs/echomind-legal/02:
 * "Not recording" must map to an actual capture-off state). The only way the Law workspace opens the microphone is
 * startMic() in app/mic.ts, which reports here when a capture starts and when it is stopped, so the status follows the
 * real MediaStream, not a page's own idea of whether it is recording.
 */
import { useSyncExternalStore } from 'react';

let active = 0;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}

export function captureStarted(): () => void {
  active += 1;
  emit();
  let done = false;
  return () => {
    if (done) return;
    done = true;
    active = Math.max(0, active - 1);
    emit();
  };
}

export const isCapturing = () => active > 0;

function subscribe(l: () => void) {
  listeners.add(l);
  return () => { listeners.delete(l); };
}

export function useCapturing(): boolean {
  return useSyncExternalStore(subscribe, isCapturing, isCapturing);
}

/** Test hook. */
export function _resetCaptureForTests() {
  active = 0;
  emit();
}
