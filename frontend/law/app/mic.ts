/**
 * Microphone → 16 kHz mono PCM16 frames (S1.8). Audio goes only to the capture socket, which the
 * server opens only after consent; nothing is recorded in the browser.
 */
export interface Mic { stop: () => void; sampleRate: number }

function downsample(input: Float32Array, from: number, to: number): Float32Array {
  if (from === to) return input;
  const ratio = from / to;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) {
    const start = Math.floor(i * ratio), end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    out[i] = end > start ? sum / (end - start) : 0;
  }
  return out;
}

function toPcm16(f32: Float32Array): ArrayBuffer {
  const buf = new ArrayBuffer(f32.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

export async function startMic(onFrame: (pcm: ArrayBuffer) => void): Promise<Mic> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = (e) => onFrame(toPcm16(downsample(e.inputBuffer.getChannelData(0), ctx.sampleRate, 16000)));
  source.connect(proc);
  proc.connect(ctx.destination);           // a ScriptProcessor only runs when connected; its output is silence
  return {
    sampleRate: 16000,
    stop: () => {
      proc.onaudioprocess = null;
      try { proc.disconnect(); source.disconnect(); } catch { /* already disconnected */ }
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
}
