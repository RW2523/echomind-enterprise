import io
import logging
import os
import wave

import numpy as np
import soundfile as sf
from piper.config import SynthesisConfig
from piper.voice import PiperVoice

logger = logging.getLogger(__name__)


class PiperTTS:
    def __init__(
        self,
        model_path: str,
        speaker_id: int = 0,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        use_cuda: bool = False,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model not found: {model_path}. Mount it into the container at /voices.")
        if use_cuda:
            try:
                self.voice = PiperVoice.load(model_path, use_cuda=True)
                logger.info("Piper TTS: CUDAExecutionProvider (%s)", model_path)
            except Exception as e:
                logger.warning("Piper TTS: CUDA failed (%s); falling back to CPU", e)
                self.voice = PiperVoice.load(model_path, use_cuda=False)
                logger.info("Piper TTS: CPUExecutionProvider (%s)", model_path)
        else:
            self.voice = PiperVoice.load(model_path, use_cuda=False)
            logger.info("Piper TTS: CPUExecutionProvider (%s)", model_path)
        self.speaker_id = speaker_id
        self.noise_scale = noise_scale
        self.length_scale = length_scale
        self.sr = 22050

    def synth(self, text: str) -> np.ndarray:
        # Piper 1.4+ (piper1-gpl): synthesize_wav + SynthesisConfig (replaces rhasspy synthesize kwargs).
        buf = io.BytesIO()
        wf = wave.open(buf, "wb")
        syn = SynthesisConfig(
            speaker_id=self.speaker_id,
            noise_scale=self.noise_scale,
            length_scale=self.length_scale,
        )
        try:
            self.voice.synthesize_wav(text=text, wav_file=wf, syn_config=syn)
        finally:
            wf.close()

        buf.seek(0)
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.sr = int(sr)
        # Trim the model's leading/trailing silence. Each phrase is an independent Piper run
        # that bakes in sentence-end silence + quiet edges; phrase-chunked streaming then
        # stacks those into 200-300 ms of dead air at EVERY phrase join (measured: 128 gaps,
        # 26.6 s dead air in a 198 s reply). The caller inserts a controlled short pause
        # between phrases instead, so joins sound natural rather than gappy.
        return _trim_edge_silence(audio.astype(np.float32), int(sr))


def _trim_edge_silence(
    y: np.ndarray,
    sr: int,
    threshold: float = 0.003,
    keep_head_ms: float = 25.0,
    keep_tail_ms: float = 50.0,
) -> np.ndarray:
    """Cut leading/trailing near-silence, keeping small natural margins."""
    if y.size == 0:
        return y
    win = max(1, int(sr * 0.01))  # 10 ms windows
    n = y.size // win
    if n < 3:
        return y
    rms = np.sqrt((y[: n * win].reshape(n, win) ** 2).mean(axis=1))
    voiced = np.nonzero(rms > threshold)[0]
    if voiced.size == 0:
        return y
    start = max(0, voiced[0] * win - int(sr * keep_head_ms / 1000))
    end = min(y.size, (voiced[-1] + 1) * win + int(sr * keep_tail_ms / 1000))
    return y[start:end]
