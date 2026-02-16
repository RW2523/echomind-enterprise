import React from "react";
import { ConversationStage } from "./Conversation/ConversationStage";
import { useVoiceConnection } from "../hooks/useVoiceConnection";
import type { AppSettings } from "../types";

interface VoiceConversationProps {
  settings?: AppSettings;
  onUpdateSetting?: (key: keyof AppSettings, val: AppSettings[keyof AppSettings]) => void;
}

const VoiceConversation: React.FC<VoiceConversationProps> = ({ settings, onUpdateSetting }) => {
  const {
    state,
    userAnalyser,
    assistantAnalyser,
    voiceMessages,
    pendingAssistantText,
    clearMemory,
    connect,
    disconnect,
    connecting,
    micMuted,
    setMicMuted,
    connectionError,
  } = useVoiceConnection({ settings });

  return (
    <div
      className="rounded-2xl border border-white/10 overflow-hidden h-full min-h-0 flex flex-col"
      style={
        {
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
        connectionError={connectionError}
        onClearMemory={clearMemory}
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
