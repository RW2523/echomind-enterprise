import React, { useEffect, useRef } from "react";

export interface TranscriptMessage {
  role: "user" | "assistant";
  text: string;
}

export interface TranscriptPanelProps {
  messages: TranscriptMessage[];
  /** Assistant reply currently streaming in */
  pendingAssistantText?: string;
  /** Streaming partial transcript while the user is still speaking */
  partialTranscript?: string;
  /** Continuous-listening mode: buffer accumulates until the wake word */
  listenOnly?: boolean;
  listenBufferText?: string;
  /** Short acknowledgement flash from the assistant */
  backchannelText?: string;
  isConnected: boolean;
  /** Speaker labels (fall back to "You" / "EchoMind") */
  userLabel?: string;
  assistantLabel?: string;
  className?: string;
}

const SCROLL_STICKY_PX = 120;

/**
 * Conversation view — the hero of the voice screen.
 * Takes the majority of the vertical space and scrolls internally.
 */
export const TranscriptPanel: React.FC<TranscriptPanelProps> = ({
  messages,
  pendingAssistantText = "",
  partialTranscript = "",
  listenOnly = false,
  listenBufferText = "",
  backchannelText = "",
  isConnected,
  userLabel = "You",
  assistantLabel = "EchoMind",
  className = "",
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < SCROLL_STICKY_PX;
  };

  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !stickToBottomRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, pendingAssistantText, partialTranscript, listenBufferText, backchannelText]);

  const isEmpty =
    messages.length === 0 &&
    !pendingAssistantText &&
    !partialTranscript &&
    !listenBufferText;

  return (
    <section
      className={`flex-1 min-h-0 flex flex-col ${className}`}
      aria-label="Conversation"
    >
      <style>{`
        @keyframes em-voice-turn-in {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: none; }
        }
        .em-voice-turn { animation: em-voice-turn-in 0.28s cubic-bezier(0.25, 0.1, 0.25, 1); }
        @media (prefers-reduced-motion: reduce) {
          .em-voice-turn { animation: none; }
        }
      `}</style>

      <div className="shrink-0 px-4 sm:px-5 pt-3 pb-2">
        <span className="text-[12px] font-medium uppercase tracking-wider text-slate-500">
          Conversation
        </span>
      </div>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="flex-1 min-h-0 overflow-y-auto overscroll-contain px-4 sm:px-5 pb-4 space-y-5 sm:space-y-6"
      >
        {isEmpty && (
          <div className="h-full min-h-[120px] flex items-center justify-center px-6">
            <p className="text-[14px] leading-relaxed text-slate-500 text-center max-w-sm">
              {isConnected
                ? "Tap the microphone and start speaking."
                : "Press Start to begin a voice session."}
            </p>
          </div>
        )}

        {listenOnly && listenBufferText && (
          <Turn label={userLabel} align="user">
            <span className="whitespace-pre-wrap break-words">{listenBufferText}</span>
            <Caret />
          </Turn>
        )}

        {!listenOnly &&
          messages.map((msg, i) => (
            <Turn
              key={i}
              label={msg.role === "user" ? userLabel : assistantLabel}
              align={msg.role}
            >
              <span className="whitespace-pre-wrap break-words">{msg.text}</span>
            </Turn>
          ))}

        {!listenOnly && partialTranscript && !pendingAssistantText && (
          <Turn label={userLabel} align="user" muted>
            <span className="whitespace-pre-wrap break-words">{partialTranscript}</span>
            <Caret />
          </Turn>
        )}

        {pendingAssistantText && (
          <Turn label={assistantLabel} align="assistant">
            <span className="whitespace-pre-wrap break-words">{pendingAssistantText}</span>
            <Caret />
          </Turn>
        )}

        {backchannelText && !pendingAssistantText && (
          <p className="text-[12px] text-slate-500 pl-0.5">{backchannelText}</p>
        )}
      </div>
    </section>
  );
};

const Caret: React.FC = () => (
  <span
    className="inline-block w-[2px] h-[1em] ml-1 align-[-0.1em] bg-current opacity-70 animate-pulse rounded-sm"
    aria-hidden
  />
);

interface TurnProps {
  label: string;
  align: "user" | "assistant";
  muted?: boolean;
  children: React.ReactNode;
}

const Turn: React.FC<TurnProps> = ({ label, align, muted = false, children }) => {
  const isUser = align === "user";
  return (
    <div className={`em-voice-turn flex flex-col ${isUser ? "items-end" : "items-start"}`}>
      <span className="text-[11px] font-medium uppercase tracking-wider text-slate-500 mb-1.5 px-1">
        {label}
      </span>
      <div
        className={`max-w-[88%] sm:max-w-[78%] lg:max-w-[62ch] rounded-2xl px-4 py-3 text-[15px] leading-[1.65] ${
          isUser
            ? "bg-white/[0.06] text-slate-200"
            : "bg-accent/[0.08] text-slate-100"
        } ${muted ? "opacity-60 italic" : ""}`}
      >
        {children}
      </div>
    </div>
  );
};
