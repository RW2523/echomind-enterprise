import React from "react";
import { ConversationStage } from "./Conversation/ConversationStage";
import type { UseVoiceConnectionReturn } from "../hooks/useVoiceConnection";
import type { AppSettings } from "../types";

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

  return (
    <div
      className="rounded-[20px] border border-white/[0.05] overflow-hidden h-full min-h-0 flex flex-col"
      style={
        {
          boxShadow: "0 8px 40px -8px rgba(0,0,0,0.18), 0 0 0 1px rgba(255,255,255,0.03)",
          "--user-color": "#94a3b8",
          "--assistant-color": "#14b8a6",
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
      />
    </div>
  );
};

export default VoiceConversation;
