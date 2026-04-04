import { useEffect, useRef } from 'react'
import type { TranscriptLine } from './useTranscript'

interface TranscriptPanelProps {
  lines: TranscriptLine[]
  onClear: () => void
}

export function TranscriptPanel({ lines, onClear }: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [lines])

  return (
    <section className="transcript-panel">
      <div className="toolbar">
        <button type="button" className="btn" onClick={onClear}>
          Clear
        </button>
      </div>
      <div className="transcript-content" ref={scrollRef}>
        {lines.map((line) => (
          <div
            key={line.id}
            className={`transcript-line ${line.isFinal ? 'final' : 'partial'}`}
          >
            {line.text}
          </div>
        ))}
      </div>
    </section>
  )
}
