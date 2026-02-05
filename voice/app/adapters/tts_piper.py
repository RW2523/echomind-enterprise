import io
import os
import wave
import numpy as np
import soundfile as sf
from piper.voice import PiperVoice

class PiperTTS:
    def __init__(self, model_path: str, speaker_id: int = 0, noise_scale: float = 0.667, length_scale: float = 1.0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Piper model not found: {model_path}. Mount it into the container at /voices.")
        self.voice = PiperVoice.load(model_path)
        self.speaker_id = speaker_id
        self.noise_scale = noise_scale
        self.length_scale = length_scale
        self.sr = 22050

    def synth(self, text: str) -> np.ndarray:
        # Piper expects wav_file to be a wave.Wave_write (has setframerate, etc.)
        buf = io.BytesIO()
        wf = wave.open(buf, "wb")
        try:
            self.voice.synthesize(
                text=text,
                wav_file=wf,
                noise_scale=self.noise_scale,
                length_scale=self.length_scale,
            )
        finally:
            wf.close()

        buf.seek(0)
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self.sr = int(sr)
        return audio.astype(np.float32)
