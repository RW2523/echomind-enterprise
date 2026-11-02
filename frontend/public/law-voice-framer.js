// EchoMind Legal voice mode: microphone → 20 ms frames of 16 kHz mono PCM16 (320 samples), posted to the page, which
// sends them to the voice service. Ported from the original EchoMind (frontend/hooks/useVoiceConnection.ts); a static
// file because the Legal app's Content-Security-Policy allows scripts from 'self' only (no blob: modules).
class LawFramer16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = 16000 / sampleRate;
    this.acc = 0;
    this.buf = [];
    this.frameSamples = 320;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      this.acc += this.ratio;
      while (this.acc >= 1.0) { this.buf.push(ch[i]); this.acc -= 1.0; }
    }
    while (this.buf.length >= this.frameSamples) {
      const frame = this.buf.splice(0, this.frameSamples);
      const pcm16 = new Int16Array(this.frameSamples);
      for (let i = 0; i < this.frameSamples; i++) {
        const v = Math.max(-1, Math.min(1, frame[i]));
        pcm16[i] = (v * 32767) | 0;
      }
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }
    return true;
  }
}
registerProcessor('law-framer16k', LawFramer16k);
