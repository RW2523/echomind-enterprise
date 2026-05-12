/** Shared helpers for decoding voice `audio_out` PCM and resampling for WebAudio playback. */

export function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

export function pcm16ToFloat32(pcmBytes: Uint8Array): Float32Array {
  const view = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength);
  const out = new Float32Array(pcmBytes.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}

export function resampleLinear(input: Float32Array, srcSr: number, dstSr: number): Float32Array {
  if (srcSr === dstSr) return input;
  const ratio = dstSr / srcSr;
  const n = Math.floor(input.length * ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i / ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const w = x - i0;
    out[i] = (1 - w) * input[i0] + w * input[i1];
  }
  return out;
}

export function fadeBufferEdges(f32: Float32Array, sampleRate: number, fadeMs: number = 3): void {
  const n = Math.min(Math.floor((sampleRate * fadeMs) / 1000), Math.floor(f32.length / 2));
  if (n <= 0) return;
  for (let i = 0; i < n; i++) {
    const t = (i + 1) / (n + 1);
    f32[i] *= t;
    f32[f32.length - 1 - i] *= t;
  }
}
