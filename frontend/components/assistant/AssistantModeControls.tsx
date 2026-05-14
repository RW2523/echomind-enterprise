import React from "react";
import { ICONS } from "../../constants";

export interface AssistantModeControlsProps {
  listening: boolean;
  micMuted: boolean;
  onMicMutedChange: (muted: boolean) => void;
  onOpenStart: () => void;
  onStop: () => void;
  onClear: () => void;
  saveTranscript?: boolean;
  onSaveTranscriptChange?: (v: boolean) => void;
  showSaveToggle?: boolean;
  disableSaveToggleWhileListening?: boolean;
}

const AssistantModeControls: React.FC<AssistantModeControlsProps> = ({
  listening,
  micMuted,
  onMicMutedChange,
  onOpenStart,
  onStop,
  onClear,
  saveTranscript,
  onSaveTranscriptChange,
  showSaveToggle = true,
  disableSaveToggleWhileListening = true,
}) => {
  return (
    <div className="flex flex-wrap items-center gap-2 mb-3">
      {!listening ? (
        <button
          type="button"
          onClick={onOpenStart}
          className="inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
        >
          <ICONS.Mic className="w-4 h-4" />
          Start listening
        </button>
      ) : (
        <>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-red-500/20 text-red-300 px-3 py-1 text-xs font-medium border border-red-500/30">
            <span className="w-2 h-2 rounded-full bg-red-400 animate-pulse" aria-hidden />
            Listening
          </span>
          <button
            type="button"
            onClick={onStop}
            className="rounded-xl px-4 py-2 text-sm font-medium bg-white/10 text-white border border-white/15"
          >
            Stop session
          </button>
        </>
      )}
      {showSaveToggle && onSaveTranscriptChange && typeof saveTranscript === "boolean" && (
        <label className="flex items-center gap-2 text-xs text-slate-400 order-last sm:order-none">
          <input
            type="checkbox"
            checked={saveTranscript}
            disabled={disableSaveToggleWhileListening && listening}
            onChange={(e) => onSaveTranscriptChange(e.target.checked)}
          />
          Save transcript
        </label>
      )}
      <label className="flex items-center gap-2 text-xs text-slate-400 ml-auto sm:ml-0">
        <input type="checkbox" checked={micMuted} onChange={(e) => onMicMutedChange(e.target.checked)} />
        Mute mic
      </label>
      <button type="button" onClick={onClear} className="text-xs text-slate-500 hover:text-slate-300 underline">
        Clear &amp; reset
      </button>
    </div>
  );
};

export default AssistantModeControls;
