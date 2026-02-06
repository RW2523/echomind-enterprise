import React from "react";
import { ConversationStage } from "./Conversation/ConversationStage";
import { useVoiceConnection } from "../hooks/useVoiceConnection";

const VoiceConversation: React.FC = () => {
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
      className="rounded-2xl border border-white/10 overflow-hidden h-[80vh] flex flex-col"
      style={
        {
          "--user-color": "#e2e8f0",
          "--assistant-color": "#00ff9c",
          "--voice-bg": "#0b0e14",
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
