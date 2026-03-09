import { useState, useEffect, useCallback } from "react";
import { createChat, listChats, getChat, deleteChat, DEFAULT_CHAT_TITLE } from "../services/backend";
import type { ChatMessage } from "../types";
import type { ChatListItem } from "../services/backend";

export interface UseKnowledgeChatReturn {
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  chatId: string;
  /** All chats for sidebar (newest first). Refreshed on load, newChat, deleteChat, and after listing. */
  chats: ChatListItem[];
  /** Reload chat list from API. Call after sending first message if you want sidebar to show updated title. */
  loadChats: () => Promise<void>;
  /** Switch to a chat: load its messages and set as current. */
  selectChat: (id: string) => Promise<void>;
  /** Start a new chat (create + set as current, clear messages). */
  newChat: () => Promise<void>;
  /** Delete a chat from history. If it was current, switches to another or creates new. */
  deleteChat: (id: string) => Promise<void>;
  /** @deprecated Use newChat() for ChatGPT-style "new chat". clearChat() still works. */
  clearChat: () => Promise<void>;
}

function apiMessageToChatMessage(m: { id: string; role: string; content: string; created_at: string }): ChatMessage {
  const ts = m.created_at ? new Date(m.created_at).getTime() : Date.now();
  return {
    id: m.id,
    role: m.role === "assistant" ? "assistant" : "user",
    content: m.content ?? "",
    timestamp: ts,
  };
}

export function useKnowledgeChat(): UseKnowledgeChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatId, setChatId] = useState<string>("");
  const [chats, setChats] = useState<ChatListItem[]>([]);

  const loadChats = useCallback(async () => {
    try {
      const { chats: list } = await listChats();
      setChats(list ?? []);
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { chats: list } = await listChats();
        if (cancelled) return;
        setChats(list ?? []);
        if (list && list.length > 0) {
          const latest = list[0];
          const chat = await getChat(latest.id);
          if (!cancelled) {
            setChatId(chat.id);
            setMessages((chat.messages ?? []).map(apiMessageToChatMessage));
          }
        } else {
          const { chat_id } = await createChat(DEFAULT_CHAT_TITLE);
          if (!cancelled) {
            setChatId(chat_id);
            setMessages([]);
            setChats(prev => [{ id: chat_id, title: DEFAULT_CHAT_TITLE, created_at: new Date().toISOString() }, ...prev]);
          }
        }
      } catch (e) {
        console.error(e);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const selectChat = useCallback(async (id: string) => {
    if (id === "") return;
    try {
      const chat = await getChat(id);
      const msgs = (chat.messages ?? []).map(apiMessageToChatMessage);
      setChatId(chat.id);
      setMessages(msgs);
    } catch (e) {
      console.error(e);
    }
  }, []);

  const newChat = useCallback(async () => {
    try {
      const { chat_id } = await createChat(DEFAULT_CHAT_TITLE);
      setChatId(chat_id);
      setMessages([]);
      await loadChats();
    } catch (e) {
      console.error(e);
    }
  }, [loadChats]);

  const deleteChatById = useCallback(async (id: string) => {
    try {
      await deleteChat(id);
      await loadChats();
      if (chatId === id) {
        const remaining = (await listChats()).chats ?? [];
        const next = remaining.find(c => c.id !== id);
        if (next) {
          const chat = await getChat(next.id);
          setChatId(chat.id);
          setMessages((chat.messages ?? []).map(apiMessageToChatMessage));
        } else {
          const { chat_id } = await createChat(DEFAULT_CHAT_TITLE);
          setChatId(chat_id);
          setMessages([]);
        }
      }
    } catch (e) {
      console.error(e);
    }
  }, [chatId, loadChats]);

  const clearChat = useCallback(async () => {
    setMessages([]);
    try {
      const { chat_id } = await createChat(DEFAULT_CHAT_TITLE);
      setChatId(chat_id);
      await loadChats();
    } catch (e) {
      console.error(e);
    }
  }, [loadChats]);

  return {
    messages,
    setMessages,
    chatId,
    chats,
    loadChats,
    selectChat,
    newChat,
    deleteChat: deleteChatById,
    clearChat,
  };
}
