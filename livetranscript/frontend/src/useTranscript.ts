import { useCallback, useRef, useState } from 'react'
import type { TranscriptEvent } from './types/transcript'

const SAMPLE_RATE = 16000
const CHUNK_MS = 560

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'error'
export type MicStatus = 'idle' | 'listening' | 'processing'

export interface TranscriptLine {
  id: string
  text: string
  isFinal: boolean
}

function getWsUrl(): string {
  const base = import.meta.env.VITE_WS_URL
  if (base) return base
  const { protocol, hostname } = window.location
  const wsProtocol = protocol === 'https:' ? 'wss:' : 'ws:'
  // HTTPS: use same host (nginx proxies /ws/). HTTP localhost dev: use :8000.
  const wsPort = protocol === 'https:' ? '' : (hostname === 'localhost' ? ':8000' : '')
  return `${wsProtocol}//${hostname}${wsPort}/ws/transcribe`
}

export function useTranscript() {
  const [lines, setLines] = useState<TranscriptLine[]>([])
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected')
  const [micStatus, setMicStatus] = useState<MicStatus>('idle')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<AudioWorkletNode | null>(null)
  const nextIdRef = useRef(0)

  const addLine = useCallback((text: string, isFinal: boolean) => {
    const id = `line-${nextIdRef.current++}`
    setLines((prev) => {
      if (!isFinal) {
        const withoutLastPartial = prev[prev.length - 1]?.isFinal === false
          ? prev.slice(0, -1)
          : prev
        return [...withoutLastPartial, { id, text, isFinal }]
      }
      const withoutLastPartial = prev[prev.length - 1]?.isFinal === false
        ? prev.slice(0, -1)
        : prev
      return [...withoutLastPartial, { id, text, isFinal }]
    })
  }, [])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    setConnectionStatus('connecting')
    setErrorMessage(null)
    const url = getWsUrl()
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    ws.onopen = () => setConnectionStatus('connected')
    ws.onclose = () => {
      setConnectionStatus('disconnected')
      setMicStatus('idle')
      wsRef.current = null
    }
    ws.onerror = () => {
      setConnectionStatus('error')
      setErrorMessage('WebSocket error')
    }
    ws.onmessage = (event) => {
      try {
        const data: TranscriptEvent = JSON.parse(event.data as string)
        if (data.type === 'partial' && data.text != null) {
          addLine(data.text, false)
          setMicStatus('processing')
        } else if (data.type === 'final' && data.text != null) {
          addLine(data.text, true)
          setMicStatus('listening')
        } else if (data.type === 'status') {
          if (data.detail === 'connected') setConnectionStatus('connected')
        } else if (data.type === 'error') {
          setErrorMessage(data.detail ?? 'Unknown error')
        }
      } catch {
        setErrorMessage('Invalid server message')
      }
    }
    wsRef.current = ws
  }, [addLine])

  const disconnect = useCallback(() => {
    stopMic()
    wsRef.current?.close()
    wsRef.current = null
    setConnectionStatus('disconnected')
    setMicStatus('idle')
  }, [])

  const startMic = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connect()
      await new Promise<void>((resolve) => {
        const check = () => {
          if (wsRef.current?.readyState === WebSocket.OPEN) return resolve()
          setTimeout(check, 50)
        }
        check()
      })
    }
    const ws = wsRef.current!
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE })
      const actualSampleRate = audioContext.sampleRate
      ws.send(JSON.stringify({ type: 'config', sample_rate: actualSampleRate }))
      const source = audioContext.createMediaStreamSource(stream)
      const chunkSamples = Math.floor((actualSampleRate * CHUNK_MS) / 1000)
      const script = `
        const chunkSamples = ${chunkSamples};
        class Processor extends AudioWorkletProcessor {
          constructor() {
            super();
            this.buffer = new Float32Array(chunkSamples);
            this.offset = 0;
          }
          process(inputs, outputs, parameters) {
            const input = inputs[0]?.[0];
            if (!input) return true;
            for (let i = 0; i < input.length; i++) {
              this.buffer[this.offset++] = input[i];
              if (this.offset >= chunkSamples) {
                this.port.postMessage(this.buffer.slice(0));
                this.buffer = new Float32Array(chunkSamples);
                this.offset = 0;
              }
            }
            return true;
          }
        }
        registerProcessor('processor', Processor);
      `
      const blob = new Blob([script], { type: 'application/javascript' })
      const dataUri = await new Promise<string>((res) => {
        const r = new FileReader()
        r.onloadend = () => res(r.result as string)
        r.readAsDataURL(blob)
      })
      await audioContext.audioWorklet.addModule(dataUri)
      const worklet = new AudioWorkletNode(audioContext, 'processor', { numberOfInputs: 1, numberOfOutputs: 0 })
      worklet.port.onmessage = (e: MessageEvent<Float32Array>) => {
        if (ws.readyState !== WebSocket.OPEN) return
        const float32 = e.data
        const int16 = new Int16Array(float32.length)
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]))
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
        }
        ws.send(int16.buffer)
      }
      source.connect(worklet)
      processorRef.current = worklet
      setMicStatus('listening')
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Microphone access failed')
      setMicStatus('idle')
    }
  }, [connect])

  const stopMic = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    processorRef.current = null
    setMicStatus('idle')
  }, [])

  const clearTranscript = useCallback(() => {
    setLines([])
    nextIdRef.current = 0
  }, [])

  return {
    lines,
    connectionStatus,
    micStatus,
    errorMessage,
    setErrorMessage,
    connect,
    disconnect,
    startMic,
    stopMic,
    clearTranscript,
  }
}
