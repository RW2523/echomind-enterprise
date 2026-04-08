# nemotron_asr

Shared package: `ASRModelAdapter` for `nvidia/nemotron-speech-streaming-en-0.6b` (NeMo) and `transcribe_utterance_float32` for offline / utterance-final decoding.

Used by:

- `backend/app/transcribe/` (live transcript WebSocket)
- `voice/app/adapters/stt_nemotron.py` (voice conversation path)

Install with NeMo ASR already present in the environment:

```bash
pip install -e ./nemotron_asr
```
