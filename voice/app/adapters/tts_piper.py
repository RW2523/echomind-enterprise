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
        return audio.astype(np.float32)
