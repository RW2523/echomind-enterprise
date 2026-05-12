import React, { useMemo, useState } from "react";
import { ICONS } from "../constants";
import { defaultTranscriptName } from "../services/backend";
import type { UseLiveTranscriptionReturn } from "../hooks/useLiveTranscription";
import type { AppSettings } from "../types";
import ProductModeHeader from "./ProductModeHeader";
import WordCloudModal from "./WordCloudModal";

function formatSegmentTime(ts: number): string {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return "";
  }
}

function formatSessionDateTime(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  const h = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${y}-${m}-${day} ${h}:${min}`;
}

interface LiveTranscriptionProps {
  liveTranscription: UseLiveTranscriptionReturn;
  settings?: AppSettings;
}

const LiveTranscription: React.FC<LiveTranscriptionProps> = ({ liveTranscription, settings }) => {
  const [showWordCloud, setShowWordCloud] = useState(false);
  const {
    fullTranscript,
    partial,
    listening,
    wsStatus,
    wsError,
    sessionName,
    sessionLocation,
    sessionStartedAt,
    customTags,
    newTagInput,
    setSessionName,
    setSessionLocation,
    setNewTagInput,
    openStartModal,
    startSession,
    handleStopAndExtractTags,
    clearAndReset,
    addTag,
    removeTag,
    micMuted,
    setMicMuted,
    showStartModal,
    modalName,
    modalLocation,
    setModalName,
    setModalLocation,
    setShowStartModal,
    applyDefault,
    transcriptSegments,
  } = liveTranscription;

  const onStartFromModal = () => {
    const name = (modalName || '').trim() || defaultTranscriptName();
    const location = (modalLocation || '').trim() || 'default';
    startSession(name, location);
  };

  const { primaryStatus, statusTone, extras } = useMemo(() => {
    const extrasLocal: { label: string; tone?: "muted" }[] = [];
    if (listening && micMuted) extrasLocal.push({ label: "Muted", tone: "muted" });
    if (wsStatus === "connecting" || wsStatus === "loading") {
      return { primaryStatus: "Connecting…", statusTone: "thinking" as const, extras: extrasLocal };
    }
    if (!listening) {
      return { primaryStatus: "Stopped", statusTone: "neutral" as const, extras: extrasLocal };
    }
    if (partial.trim().length > 0) {
      return { primaryStatus: "Transcribing", statusTone: "transcribing" as const, extras: extrasLocal };
    }
    return { primaryStatus: "Listening", statusTone: "listening" as const, extras: extrasLocal };
  }, [listening, micMuted, partial, wsStatus]);

  const rightControls = (
    <>
      <button
        type="button"
        onClick={() => listening && setMicMuted(!micMuted)}
        disabled={!listening}
        className={`relative flex items-center justify-center w-10 h-10 rounded-xl touch-manipulation transition-colors ${
          !listening ? "bg-white/5 cursor-default" : micMuted ? "bg-red-500/20 hover:bg-red-500/30" : "bg-emerald-500/20 hover:bg-emerald-500/30"
        }`}
        aria-label={listening ? (micMuted ? "Unmute mic" : "Mute mic") : "Mic"}
        title={listening ? (micMuted ? "Unmute" : "Mute") : undefined}
      >
        <ICONS.Mic className={`w-5 h-5 ${!listening ? "text-slate-400" : micMuted ? "text-red-400" : "text-emerald-400"}`} />
        {listening && !micMuted && (
          <span className="absolute inset-0 rounded-xl bg-emerald-400/20 animate-ping" style={{ animationDuration: "1.5s" }} />
        )}
      </button>
      <button
        type="button"
        onClick={() => setShowWordCloud(true)}
        className="shrink-0 p-2.5 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 touch-manipulation min-h-[44px] min-w-[44px] flex items-center justify-center"
        aria-label="Word cloud"
        title="Word cloud"
      >
        <ICONS.WordCloud className="w-5 h-5" />
      </button>
      <button
        type="button"
        onClick={clearAndReset}
        className="shrink-0 p-2.5 rounded-xl text-slate-400 hover:text-white hover:bg-white/10 touch-manipulation min-h-[44px] min-w-[44px] flex items-center justify-center"
        aria-label="Clear transcript and start new session"
        title="Clear"
      >
        <ICONS.Trash className="w-5 h-5" />
      </button>
      {wsStatus === "connecting" && <span className="text-xs text-slate-400">Connecting…</span>}
      {wsStatus === "loading" && (
        <span className="text-xs text-slate-400 max-w-[140px] sm:max-w-none">Loading STT…</span>
      )}
      {wsError && (
        <span className="text-xs text-red-400 max-w-[120px] sm:max-w-[200px] truncate" title={wsError}>
          {wsError}
        </span>
      )}
      {!listening ? (
        <button
          type="button"
          onClick={openStartModal}
          disabled={wsStatus === "connecting" || wsStatus === "loading"}
          className="rounded-xl px-4 py-2.5 min-h-[44px] text-sm font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30 disabled:opacity-50 transition-colors touch-manipulation"
        >
          Start
        </button>
      ) : (
        <button
          type="button"
          onClick={handleStopAndExtractTags}
          className="rounded-xl px-4 py-2.5 min-h-[44px] text-sm font-semibold bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors touch-manipulation"
        >
          Stop
        </button>
      )}
    </>
  );

  return (
    <div className="h-full min-h-0 flex flex-col rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
      {/* Start modal */}
      {showStartModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={() => setShowStartModal(false)}>
          <div className="rounded-2xl border border-white/20 bg-slate-900 shadow-xl max-w-md w-full p-5 space-y-4" onClick={(e) => e.stopPropagation()}>
            <div className="font-semibold text-white">Start transcription</div>
            <p className="text-sm text-slate-400">Name and location are saved with the transcript every 1 min and used in RAG (e.g. &quot;summary of last 5 mins in office&quot;).</p>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Name</label>
              <input
                type="text"
                value={modalName}
                onChange={(e) => setModalName(e.target.value)}
                placeholder="e.g. transcript_2025-02-12_14-30"
                className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Location</label>
              <input
                type="text"
                value={modalLocation}
                onChange={(e) => setModalLocation(e.target.value)}
                placeholder="e.g. default or Office"
                className="w-full rounded-lg border border-white/20 bg-white/5 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
              />
            </div>
            <div className="flex flex-wrap gap-2 pt-2">
              <button type="button" onClick={applyDefault} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-300 hover:bg-white/15">
                Default
              </button>
              <button type="button" onClick={() => setShowStartModal(false)} className="rounded-xl px-4 py-2 text-sm font-semibold bg-white/10 text-slate-400 hover:bg-white/15">
                Cancel
              </button>
              <button type="button" onClick={onStartFromModal} className="rounded-xl px-4 py-2 text-sm font-semibold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/30">
                Start
              </button>
            </div>
          </div>
        </div>
      )}

      <div className={`shrink-0 border-b transition-all duration-300 ${listening ? "border-cyan-500/25 bg-cyan-500/[0.04]" : "border-white/10"}`}>
        <ProductModeHeader
          title="Transcribe"
          tagline="Listen and write everything down."
          status={primaryStatus}
          statusTone={statusTone}
          extraStatuses={extras}
          sessionName={sessionName?.trim() || null}
          showKnowledge
          knowledgeEnabled={!!settings?.voiceUseKnowledgeBase}
          outputHint="No speech — text-only transcript. Suggestions are not generated in this mode."
          rightSlot={<div className="flex flex-wrap items-center gap-2">{rightControls}</div>}
        />
      </div>

      {/* Editable bar: name, location, date/time, custom tags - wraps on mobile */}
      {(listening || sessionName || sessionLocation || sessionStartedAt) && (
        <div className="shrink-0 px-3 sm:px-5 py-3 border-b border-white/10 bg-black/10 flex flex-wrap items-center gap-2 sm:gap-3">
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Name</span>
            <input type="text" value={sessionName} onChange={(e) => setSessionName(e.target.value)} placeholder="Transcript name" className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-2 text-sm text-white placeholder-slate-500 w-36 sm:w-48 max-w-full focus:border-cyan-500/40 focus:outline-none min-h-[40px]" />
          </div>
          <span className="text-slate-600 hidden sm:inline">|</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Location</span>
            <input type="text" value={sessionLocation} onChange={(e) => setSessionLocation(e.target.value)} placeholder="Location" className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-2 text-sm text-white placeholder-slate-500 w-24 sm:w-32 max-w-full focus:border-cyan-500/40 focus:outline-none min-h-[40px]" />
          </div>
          <span className="text-slate-600 hidden sm:inline">|</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Date & time</span>
            <span className="text-sm text-slate-300">{sessionStartedAt ? formatSessionDateTime(sessionStartedAt) : '—'}</span>
          </div>
          <span className="text-slate-600 hidden sm:inline">|</span>
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            <span className="text-xs text-slate-500 shrink-0">Tags</span>
            {customTags.map((tag) => (
              <span key={tag} className="inline-flex items-center gap-1 rounded-lg bg-white/10 border border-white/10 px-2 py-1.5 text-xs text-white/90">
                {tag}
                <button type="button" onClick={() => removeTag(tag)} className="text-slate-400 hover:text-white leading-none p-0.5 touch-manipulation min-w-[28px]" aria-label={`Remove ${tag}`}>×</button>
              </span>
            ))}
            <input type="text" value={newTagInput} onChange={(e) => setNewTagInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())} placeholder="+ Add tag" className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-2 text-sm text-white placeholder-slate-500 w-20 sm:w-24 min-w-0 max-w-full focus:border-cyan-500/40 focus:outline-none min-h-[40px]" />
            <button type="button" onClick={addTag} className="rounded-lg px-3 py-2 text-xs font-medium bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 touch-manipulation min-h-[40px]">Add</button>
          </div>
        </div>
      )}

      <div className="flex-1 min-h-0 p-4 sm:p-5 overflow-auto flex flex-col gap-4">
        <div className="rounded-2xl border border-white/10 bg-black/20 p-4 flex flex-col flex-1 min-h-0">
          <div className="text-xs font-semibold opacity-70 mb-3 shrink-0">Live transcript (auto-saved every 1 min to transcripts table + RAG; name, location &amp; time used for chat queries)</div>
          <div className="flex-1 min-h-0 text-sm whitespace-pre-wrap opacity-90 overflow-auto">
            {[fullTranscript, partial].filter(Boolean).join(" ").trim() ? (
              [fullTranscript, partial].filter(Boolean).join(" ")
            ) : (
              <span className="text-slate-500 italic">Start speaking to see live transcript.</span>
            )}
          </div>
        </div>
        {transcriptSegments.length > 0 && (
          <div className="rounded-2xl border border-white/10 bg-black/15 p-4 shrink-0 max-h-[40vh] flex flex-col min-h-0">
            <div className="text-xs font-semibold text-slate-400 mb-2 shrink-0">Committed segments ({transcriptSegments.length})</div>
            <ul className="space-y-2 overflow-y-auto text-sm min-h-0">
              {transcriptSegments.map((seg) => (
                <li key={seg.paragraphId} className="border-b border-white/5 pb-2 last:border-0 last:pb-0">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">
                    {seg.paragraphId}
                    {seg.receivedAt ? <span className="normal-case text-slate-500"> · {formatSegmentTime(seg.receivedAt)}</span> : null}
                  </div>
                  <div className="text-slate-200 whitespace-pre-wrap">{seg.text}</div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {showWordCloud && (
        <WordCloudModal
          onClose={() => setShowWordCloud(false)}
          liveText={[fullTranscript, partial].filter(Boolean).join(' ')}
          listening={listening}
        />
      )}
    </div>
  );
};

export default LiveTranscription;
