import { useTranscript } from './useTranscript'
import { TranscriptPanel } from './TranscriptPanel'

function App() {
  const {
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
  } = useTranscript()

  const isConnected = connectionStatus === 'connected'
  const isListening = micStatus === 'listening' || micStatus === 'processing'

  return (
    <div className="app">
      <header className="header">
        <h1>Live Transcript</h1>
        <div className="controls">
          <div className="status-bar">
            <span
              className={`status-dot ${connectionStatus === 'connected' ? 'connected' : ''} ${micStatus === 'listening' ? 'listening' : ''} ${micStatus === 'processing' ? 'processing' : ''} ${connectionStatus === 'error' ? 'error' : ''}`}
            />
            {connectionStatus === 'disconnected' && 'Disconnected'}
            {connectionStatus === 'connecting' && 'Connecting…'}
            {connectionStatus === 'connected' && !isListening && 'Connected'}
            {connectionStatus === 'connected' && micStatus === 'listening' && 'Listening'}
            {connectionStatus === 'connected' && micStatus === 'processing' && 'Processing'}
            {connectionStatus === 'error' && 'Error'}
          </div>
          {!isConnected ? (
            <button type="button" className="btn primary" onClick={connect}>
              Connect
            </button>
          ) : (
            <>
              <button
                type="button"
                className="btn primary"
                onClick={isListening ? stopMic : startMic}
                disabled={micStatus === 'idle' && connectionStatus !== 'connected'}
              >
                {isListening ? 'Stop mic' : 'Start mic'}
              </button>
              <button type="button" className="btn" onClick={disconnect}>
                Disconnect
              </button>
            </>
          )}
        </div>
      </header>

      {errorMessage && (
        <div className="error-message">
          {errorMessage}
          <button
            type="button"
            className="btn"
            style={{ marginLeft: '0.5rem' }}
            onClick={() => setErrorMessage(null)}
            aria-label="Dismiss"
          >
            Dismiss
          </button>
        </div>
      )}

      <TranscriptPanel lines={lines} onClear={clearTranscript} />
    </div>
  )
}

export default App
