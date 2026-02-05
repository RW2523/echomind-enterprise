# Unmute Path A v5.1 — Context box + Conversation Memory

✅ Context/Role box in the browser (System prompt)  
✅ Remembers previous turns in the same session (like Unmute)  
✅ Clear Memory button  
✅ Still supports: streaming LLM -> phrase TTS -> barge-in cancel + smooth fade

## Run
```bash
bash scripts/download_voice.sh
docker build -t unmute-path-a:v5.1 .

docker run --rm -it --gpus all --network host   -v $PWD/voices:/voices   -e LLM_URL=http://127.0.0.1:11434/v1/chat/completions   -e LLM_MODEL=qwen2.5:7b-instruct   -e PIPER_MODEL=/voices/en_US-lessac-medium.onnx   unmute-path-a:v5.1
```

Open: `http://<DGX_IP>:8000`

## Notes
- Memory is per WebSocket session.
- Server keeps last ~12 turns and trims to a token budget.


## v5.3 interrupt fix
- True Unmute-like barge-in: sustained speech cancels assistant pipeline immediately while continuing to capture your utterance.
- Client fallback: BARGE_IN event stops playback immediately.
