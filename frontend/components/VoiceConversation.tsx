import React, { useCallback, useEffect, useRef } from "react";
import { ConversationStage } from "./Conversation/ConversationStage";
import type { UseVoiceConnectionReturn } from "../hooks/useVoiceConnection";
import type { AppSettings } from "../types";
import { PersonaType } from "../types";

interface VoiceConversationProps {
  settings?: AppSettings;
  onUpdateSetting?: (key: keyof AppSettings, val: AppSettings[keyof AppSettings]) => void;
  voiceConnection: UseVoiceConnectionReturn;
}

const VoiceConversation: React.FC<VoiceConversationProps> = ({ settings, onUpdateSetting, voiceConnection }) => {
  const {
    state,
    userAnalyser,
    assistantAnalyser,
    voiceMessages,
    pendingAssistantText,
    listenBufferText,
    applyContext,
    clearMemory,
    listenOnly,
    setListenOnly,
    connect,
    disconnect,
    connecting,
    micMuted,
    setMicMuted,
    connectionError,
  } = voiceConnection;

  const persona = settings?.persona;
  const isConnected = state.isConnected;

  const handlePersonaChange = useCallback(
    (next: PersonaType) => {
      onUpdateSetting?.("persona", next);
    },
    [onUpdateSetting]
  );

  // Push a persona change to the live session (settings update asynchronously,
  // so apply once the new persona has landed in props).
  const lastPersonaRef = useRef(persona);
  useEffect(() => {
    if (lastPersonaRef.current === persona) return;
    lastPersonaRef.current = persona;
    if (isConnected) applyContext();
  }, [persona, isConnected, applyContext]);

  return (
    <div
      className="rounded-[20px] border border-white/[0.05] overflow-hidden h-full min-h-0 flex flex-col"
      style={
        {
          boxShadow: "0 8px 40px -8px rgba(0,0,0,0.18), 0 0 0 1px rgba(255,255,255,0.03)",
          "--user-color": "#94a3b8",
          "--voice-bg": "#0f172a",
          "--voice-text": "#f1f5f9",
        } as React.CSSProperties
      }
    >
      <ConversationStage
        state={state}
        userAnalyser={userAnalyser}
        assistantAnalyser={assistantAnalyser}
        voiceMessages={voiceMessages}
        pendingAssistantText={pendingAssistantText}
        listenBufferText={listenBufferText}
        connectionError={connectionError}
        onClearMemory={clearMemory}
        listenOnly={listenOnly}
        onListenOnlyToggle={() => setListenOnly(!listenOnly)}
        onConnect={connect}
        onDisconnect={disconnect}
        connecting={connecting}
        micMuted={micMuted}
        onMicMutedToggle={() => setMicMuted(!micMuted)}
        partialTranscript={state.partialTranscript}
        backchannelText={state.backchannelText}
        persona={persona}
        onPersonaChange={onUpdateSetting ? handlePersonaChange : undefined}
        onApplyContext={applyContext}
        userLabel={(settings?.voiceUserName ?? "").trim() || "You"}
        assistantLabel={(settings?.voiceBotName ?? "").trim() || "EchoMind"}
      />
    </div>
  );
};

export default VoiceConversation;
