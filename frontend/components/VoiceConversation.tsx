import React from 'react';
import { ICONS } from '../constants';

const VoiceConversation: React.FC = () => {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden h-[80vh]">
      <div className="px-5 py-4 border-b border-white/10 flex items-center gap-2">
        <div className="opacity-80">{ICONS.wave}</div>
        <div className="font-semibold">Real-Time Conversational AI</div>
        <div className="ml-auto text-xs opacity-60">Unmute Path-A</div>
      </div>
      <iframe title="Voice" src="/voice/" className="w-full h-full" />
    </div>
  );
};

export default VoiceConversation;
