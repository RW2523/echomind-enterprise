import React from "react";
import { ConversationStage } from "./Conversation/ConversationStage";
import { useVoiceConnection } from "../hooks/useVoiceConnection";
import type { AppSettings } from "../types";

interface VoiceConversationProps {
  settings?: AppSettings;
}

const VoiceConversation: React.FC<VoiceConversationProps> = () => {
  const {
    state,
    userAnalyser,
    assistantAnalyser,
    contextValue,
    setContextValue,
    applyContext,
    clearMemory,
    connect,
    disconnect,
    connecting,
    setUserOrbState,
    setAssistantOrbState,
    triggerInterrupt,
  } = useVoiceConnection();

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
        contextValue={contextValue}
        onContextChange={setContextValue}
        onApplyContext={applyContext}
        onClearMemory={clearMemory}
        onConnect={connect}
        onDisconnect={disconnect}
        connecting={connecting}
        showDemoControls={!state.isConnected}
        onDemoUserSpeak={() => {
          setUserOrbState("listening");
          setAssistantOrbState("idle");
        }}
        onDemoAssistantSpeak={() => {
          setAssistantOrbState("speaking");
          setUserOrbState("idle");
        }}
        onDemoInterrupt={triggerInterrupt}
      />
    </div>
  );
};

export default VoiceConversation;
