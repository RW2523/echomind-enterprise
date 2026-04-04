"""App configuration from environment."""
import os
from typing import Optional

# Model
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "nvidia/nemotron-speech-streaming-en-0.6b")
# att_context_size: [left, right] in 80ms frames. right=1->160ms, 6->560ms (better WER), 13->1120ms (best WER, slower)
ASR_ATT_CONTEXT_RIGHT = int(os.getenv("ASR_ATT_CONTEXT_RIGHT", "6"))  # 6 = 560ms for better accuracy
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WS_PATH = os.getenv("WS_PATH", "/ws/transcribe")

# Endpointing / finalization
SILENCE_MS_BEFORE_FINAL = int(os.getenv("SILENCE_MS_BEFORE_FINAL", "500"))
MIN_PARTIAL_LENGTH_FOR_FINAL = int(os.getenv("MIN_PARTIAL_LENGTH_FOR_FINAL", "2"))

# Chunking: match model frame (560ms for right=6 = better accuracy, fewer inferences = faster)
AUDIO_CHUNK_MS = int(os.getenv("AUDIO_CHUNK_MS", "560"))
