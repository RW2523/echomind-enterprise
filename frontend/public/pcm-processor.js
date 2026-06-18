class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._frameSize = 4096;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) this._buffer.push(input[i]);
    while (this._buffer.length >= this._frameSize) {
      const frame = new Float32Array(this._buffer.splice(0, this._frameSize));
      this.port.postMessage(frame.buffer, [frame.buffer]);
    }
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
