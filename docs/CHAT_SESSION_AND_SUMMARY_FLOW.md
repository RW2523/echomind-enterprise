# Chat Session and Summary Flow

## Overview

This document explains how chat sessions, `chat_id`, and conversation summary work—and how they behave on page refresh or chat clear.

---

## 1. Chat ID and Session

### Frontend (`useKnowledgeChat.ts`)

- **chat_id** is stored in React state only (not localStorage or sessionStorage).
- On mount: `createChat()` is called → new `chat_id` from backend.
- On `clearChat()`: new `createChat()` → new `chat_id`.

### Backend

- **chats** table: `id` (chat_id), `title`, `created_at`, `conversation_summary`
- **messages** table: `id`, `chat_id`, `role`, `content`, `created_at`
- All data is keyed by `chat_id`.

### When Page Is Refreshed

1. React state is lost.
2. `useEffect` runs → `createChat()` → **new** `chat_id`.
3. Old chat_id is gone; history and summary for the old chat remain in DB but are no longer used.
4. Result: user sees a **new chat** with empty history and no summary.

### When Chat Is Cleared

1. `clearChat()` → `createChat()` → **new** `chat_id`.
2. Same as refresh: new chat, empty history, no summary.

### Summary

- **Session = one chat_id.** When you refresh or clear, you get a new chat_id.
- **Data is ephemeral from the user’s perspective** per session.
- Backend stores data by chat_id; old chats are orphaned when the frontend switches to a new chat_id.

---

## 2. Conversation Summary

### When It Is Updated

After each answer, a background task runs:

```
update_conversation_summary(prev_summary, user_msg, assistant_msg)
→ stored in chats.conversation_summary for that chat_id
```

### When It Is Used

- **Before:** Summary was always included and placed first in the prompt.
- **Now:** Summary is used only when the current question looks like a **follow-up** (short, vague, or with follow-up markers like “that”, “it”, “what about”, “and”, “more”).
- **Order:** Question first, RAG context second, summary last (only when follow-up).
- **Instruction:** Summary is labeled as “Optional context from earlier (use only if directly relevant)”.

### Follow-up Detection

`_is_follow_up_question(question)` returns True when:

- Question has ≤ 4 words, or
- Question contains markers like: “that”, “it”, “this”, “what about”, “and”, “also”, “more”, “explain”, “elaborate”, “tell me more”, “go on”.

### When Not a Follow-up

- Summary is not used.
- Instead, the last 10 messages from history are used.

---

## 3. Flow Diagram

```
Page load / clear
   → createChat() → new chat_id
   → history = [], summary = None

User sends message
   → Backend: load history + summary for chat_id
   → If follow-up: include summary (at end, optional)
   → Else: use history[-10:]
   → Answer → store messages → update summary in background

Page refresh / clear
   → new chat_id
   → history = [], summary = None (same as fresh load)
```

---

## 4. Persistence Options

- **Current:** chat_id is ephemeral; refresh/clear = new chat.
- **To persist across refresh:** Store `chat_id` in `sessionStorage` in the frontend and reuse it on mount.
- **To persist across browser close:** Store `chat_id` in `localStorage`.

The current behavior matches “single session, everything gone on refresh/clear.”
