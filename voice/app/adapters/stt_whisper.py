import torch
import whisper
import numpy as np

class WhisperSTT:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe(self, audio16k_f32: np.ndarray) -> str:
        r = self.model.transcribe(audio16k_f32, fp16=torch.cuda.is_available(), temperature=0.0)
        return (r.get("text") or "").strip()
