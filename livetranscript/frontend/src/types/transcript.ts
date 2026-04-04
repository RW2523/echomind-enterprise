export type TranscriptEventType = 'partial' | 'final' | 'status' | 'error'

export interface TranscriptEvent {
  type: TranscriptEventType
  text?: string
  start_ms?: number
  end_ms?: number
  session_id?: string
  timestamp?: string
  detail?: string
}
