import { useState, useEffect, useCallback } from "react";
import { createChat } from "../services/backend";
import type { ChatMessage } from "../types";

export interface UseKnowledgeChatReturn {
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  chatId: string;
  clearChat: () => Promise<void>;
}

export function useKnowledgeChat(): UseKnowledgeChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatId, setChatId] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { chat_id } = await createChat("EchoMind Chat");
        if (!cancelled) setChatId(chat_id);
      } catch (e) {
        console.error(e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const clearChat = useCallback(async () => {
    setMessages([]);
    try {
      const { chat_id } = await createChat("EchoMind Chat");
      setChatId(chat_id);
    } catch (e) {
      console.error(e);
    }
  }, []);

  return {
    messages,
    setMessages,
    chatId,
    clearChat,
  };
}
