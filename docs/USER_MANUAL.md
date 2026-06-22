# EchoMind Enterprise — User Manual

*Your complete guide to the private, on-premises AI workspace.*

---

## Contents

1. [Welcome to EchoMind](#1-welcome-to-echomind)
2. [How EchoMind Works (The Big Picture)](#2-how-echomind-works-the-big-picture)
3. [Getting Started](#3-getting-started)
4. [Knowledge Chat — Ask Your Documents Anything](#4-knowledge-chat--ask-your-documents-anything)
5. [Live Transcription & the Silent Assistant](#5-live-transcription--the-silent-assistant)
6. [Boardroom — Meetings into Decisions](#6-boardroom--meetings-into-decisions)
7. [Voice Conversation — Talk to Your Knowledge](#7-voice-conversation--talk-to-your-knowledge)
8. [Document Studio — Generate Polished Documents](#8-document-studio--generate-polished-documents)
9. [Personas — Choose How EchoMind Thinks](#9-personas--choose-how-echomind-thinks)
10. [Settings & Personalization](#10-settings--personalization)
11. [Privacy, Security & Trust](#11-privacy-security--trust)
12. [Tips & Best Practices](#12-tips--best-practices)
13. [Troubleshooting & FAQ](#13-troubleshooting--faq)
14. [Glossary](#14-glossary)
15. [Appendix — For Administrators](#15-appendix--for-administrators)
16. [Appendix — Additional Concepts](#16-appendix--additional-concepts)

---

## 1. Welcome to EchoMind

EchoMind Enterprise is a complete AI workspace — chat, transcription, meetings, voice, and document creation — that runs entirely on your own hardware. No cloud, no API keys, no data leaving the building. Everything you read, say, upload, and generate stays on your machines.

### What EchoMind Enterprise is

In one breath: **EchoMind Enterprise is a private, offline AI workspace that lives on your own NVIDIA GPU servers.** Every model it uses — the language model that writes and reasons, the speech recognizer that transcribes you, the voice that talks back, the engine that searches your documents, and the one that paints images — is pre-loaded and served locally inside Docker on your hardware.

That means no outbound calls to any AI provider, no telemetry, and no internet dependency once it's installed. You can literally pull the network cable and it keeps working.

**Why it matters:** Most "AI assistants" quietly ship your prompts, files, and recordings to a vendor's cloud. EchoMind doesn't. Your contracts, board recordings, patient notes, and source code are processed where they already live.

### Who it's for

EchoMind is built for people who can't — or won't — send their data to someone else's servers:

- **Regulated industries** — finance, legal, defense, healthcare, government — where data residency and confidentiality aren't optional.
- **Privacy-conscious teams** who want modern AI without a third-party processor in the loop.
- **Air-gapped and secure environments** where there is no internet at all, by design.
- **Anyone** who simply prefers that their work stay on their own machine.

If your data cannot go to a cloud, EchoMind is the workspace that meets it where it is.

### A quick tour of the five modules

EchoMind bundles five products behind one web interface. Here's the whole workspace at a glance:

- **Knowledge Chat** — Upload your PDFs, Word docs, and slide decks, then ask questions in plain language. Answers are grounded in *your* documents and come with citations (document, section, page), so you can always check the source. Under the hood it uses hybrid search (dense vectors plus keyword BM25) with cross-encoder reranking for accuracy on large, structured documents.
- **Live Transcription + Silent Assistant** — Turns live speech into text in real time, and quietly fact-checks each statement against your knowledge base as it's spoken — labeling claims **Supported**, **Contradicted**, **Unverified**, **Violating**, or **Risky** with an explanation. Transcripts are saved into your knowledge base automatically.
- **Boardroom** — Records a full meeting, separates who said what (Speaker 1, Speaker 2…), and produces a structured AI meeting report — summary, key points, decisions, contradictions against your documents — that you can export to **PDF or PPTX**.
- **Voice Conversation** — A natural, full-duplex voice assistant you can talk over (barge-in), that fills silence with quick lead phrases and backchannels, remembers the last part of your conversation, and answers from your documents. It even has a listen-only mode and a wake word.
- **Document Studio** — Turns a topic, a chat, or your uploaded sources into a finished, professionally formatted document using one of **18 built-in templates**, complete with on-device AI-generated images, exported to **PDF and PPTX**.

All five share the same private foundation: a local language model (**Llama-3.1-8B**), local embeddings (**nomic-embed-text**), local speech recognition (**Nemotron**), a local voice (**Piper**), and a local image generator (**SDXL-Turbo**) — and a shared set of **personas** (Teacher/Professor, Financial Advisor, Lawyer, AI Expert & Manager, General Assistant, Funny & Calming, and the EchoMind Guide) that change the tone and expertise of the assistant across modules.

### How to use this manual

This manual is written to be read in order or jumped into by chapter. Each module gets its own chapter, and within it you'll find:

1. **The outcome first** — what the feature does for you and why you'd reach for it.
2. **How to use it** — concrete, click-by-click steps based on the real interface, including what you'll see at each stage.
3. **Callouts** — short, scannable notes (legend below) for the things worth knowing without slowing down.

You don't need to be technical to use EchoMind. Where deployment or administration details come up, they're flagged clearly so you can hand them to whoever runs your servers.

#### Callout legend

You'll see four kinds of highlighted notes throughout this manual:

| Callout | What it tells you |
|---|---|
| **Why it matters** | The benefit or the reason a feature exists — the payoff. |
| **Try this** | A concrete action to take right now to see the feature work. |
| **Pro tip** | A shortcut, setting, or technique that gets you a better result. |
| **Heads up** | A limit, prerequisite, or gotcha to know before it surprises you. |

### A note for whoever runs the servers

EchoMind is **private by design**: all inference is local, nothing is sent to any external AI service, there's no telemetry, and any content the AI reads from your documents is treated as untrusted and fenced off to resist prompt-injection.

**Heads up:** This build is designed to run on a **trusted, isolated network**. As a best practice, deploy it behind your own SSO, VPN, or an authenticating reverse proxy rather than exposing it directly to the open internet — the same way you'd protect any internal system that holds sensitive data. Your administrator can find the full deployment and HTTPS guidance in the project's `README.md` and the `docs/` folder.

**Try this:** Open a browser to your EchoMind address (typically `http://<your-server-ip>:3000`, or the HTTPS address your admin set up) and you'll land in the workspace. From here, the next chapter walks you through Knowledge Chat — the fastest way to feel what on-device AI can do with your own documents.

---

## 2. How EchoMind Works (The Big Picture)

Think of EchoMind as one website your team opens in a browser — except every piece of intelligence behind it runs on your own machines, in your own building. Nothing you upload, say, or ask travels to an outside company. This chapter gives you the mental model: how the whole thing fits together, and why that design quietly works in your favor.

### One app, five tools

You open **one web address** (something like `http://your-server:3000`) and land in a single app. Inside it live the five tools you'll use day to day:

| Module | What you reach for it for |
|---|---|
| **Knowledge Chat** | Ask questions across your own documents and get answers with citations. |
| **Live Transcription + Silent Assistant** | Turn live speech into text while it quietly fact-checks against your files. |
| **Boardroom** | Record a meeting and get a speaker-by-speaker transcript plus an AI report (PDF/PPTX). |
| **Voice Conversation** | Talk to a hands-free assistant that answers out loud from your knowledge base. |
| **Document Studio** | Generate a finished, formatted document — with AI images — and export it. |

**Why it matters:** one login-free door, one consistent look, no juggling separate apps or accounts. You'll learn each one in its own chapter.

### Everything runs on your own GPUs

This is the part that makes EchoMind different from cloud AI tools. Every model — the part that "thinks," "listens," "speaks," and "draws" — runs on the **NVIDIA GPU(s) inside your server**. There are **no third-party AI APIs**, no usage meters ticking somewhere, and **no telemetry phoning home**.

**Why it matters:** your documents, recordings, and conversations stay on your hardware. A meeting transcript or a sensitive contract is processed in the same building it was created in and goes nowhere else. There's no "we may use your data to improve our service" fine print, because there's no outside service involved.

### It works air-gapped

EchoMind is **offline-first**. The AI models are downloaded once during setup, then cached locally — after that the system runs **fully disconnected from the internet**.

**Why it matters:** you can deploy it in a secure, network-isolated facility and it just keeps working. No internet connection is required to chat, transcribe, record meetings, hold a voice conversation, or generate a document. If your environment loses connectivity — or never had it by design — EchoMind doesn't notice.

**Heads up:** the one-time setup needs internet to fetch and cache the models. Your administrator runs that once; everyone after that uses an air-gapped system.

### The local engines, in plain terms

You don't have to think about the machinery — but it helps to know there are a handful of specialized "engines" working behind the scenes, each good at one job and all running on your GPUs:

- **The chat brain (LLM).** This is the engine that reads your documents and writes answers, summaries, and reports in natural language. It powers the writing in Knowledge Chat, Boardroom reports, Voice answers, and Document Studio.
- **The librarian (embeddings).** When you upload a file, this engine turns its meaning into a form the system can search, so Knowledge Chat can find the right passage even when your wording doesn't match the document's exactly.
- **The ears (speech-to-text).** This engine turns spoken audio into written text — the live transcript you watch, the meeting recording you finalize, and what you say to the voice assistant.
- **The voice (text-to-speech).** This engine speaks the assistant's answers out loud in Voice Conversation, so you can have a real back-and-forth.
- **The illustrator (image generation).** This engine creates pictures on demand inside Document Studio, so a generated report or deck can include relevant visuals — all rendered on your own hardware, never pulled from the web.

**Pro tip:** because the chat brain and the librarian work together, Knowledge Chat answers are **grounded in your files and come with citations** — the document, section, and page — so you can verify any claim instead of taking the AI's word for it.

### How a request actually flows

You never see this, but here's the shape of it so the system feels predictable rather than magical:

1. **You act in the browser** — type a question, speak, upload a file, or click "generate."
2. **Your request goes to EchoMind's server** on your network. The browser only ever talks to your own machine.
3. **The right engines do the work on your GPUs** — the ears transcribe, the librarian finds relevant passages, the chat brain composes the answer, the voice speaks it, or the illustrator draws.
4. **The result streams back to your screen** — answers appear word by word, transcripts update live, and exports download straight to you.

**Why it matters:** every hop in that chain happens inside your walls. There's no step where your data is handed to someone else's cloud.

### A note for whoever sets it up

EchoMind is built to live on a **trusted, isolated network** — that's the environment its privacy promises assume. If you ever plan to make it reachable beyond a closed LAN, the right move is to place it **behind your own SSO, VPN, or an authenticating reverse proxy** first, the same way you'd protect any internal system. Treat EchoMind like the rest of your private infrastructure and the on-premises design does exactly what it's meant to: keep your data yours.

---

## 3. Getting Started

Your first few minutes with EchoMind set the tone for everything after, so this chapter walks you from a fresh browser tab to a confident lap around the whole interface. By the end you'll know exactly where to open the app, why microphone features ask for a secure connection, and what each part of the screen is for.

### Open the app

EchoMind runs on your own hardware and serves its interface straight to your browser. Open a modern browser (Chrome, Edge, or Firefox all work well) and go to one of two addresses, swapping `HOST` for your server's IP address or hostname:

- **`http://HOST:3000`** — the plain HTTP entrance.
- **`https://HOST:3443`** — the secure HTTPS entrance.

If you're sitting at the machine itself, `http://localhost:3000` works too.

**Heads up:** if your server uses the built-in self-signed certificate, the HTTPS address may show a one-time browser warning. Click **Advanced**, then **Proceed** to continue. To remove the warning for good, your administrator can set up a trusted certificate (see `docs/HTTPS_TRUSTED_CERTIFICATE.md` for a public domain, or `docs/HTTPS_LOCAL_TRUSTED.md` for local machines).

When the page loads, you land on **Knowledge Chat** by default, with the navigation sidebar on the left and a title bar across the top.

### Why microphone features need HTTPS (or localhost)

Three of EchoMind's modules listen to your microphone: **Live Transcript**, **Conversation**, and any voice-driven flow. Browsers only hand microphone access to pages served over a *secure context* — that means **HTTPS**, or **localhost** on the same machine. This is a browser rule that protects you, not an EchoMind limitation.

**What this means in practice:**

- Working at the server itself? `http://localhost:3000` is treated as secure, so the mic works.
- Connecting from another computer over the network? Use the **`https://HOST:3443`** address so the mic features can run.
- On plain `http://HOST:3000` from a remote machine, text features (Knowledge Chat, Document Studio) work fine, but the browser will block the microphone.

**Pro tip:** the first time you start Live Transcript or Conversation, your browser pops up a microphone permission prompt. Click **Allow**. If you accidentally block it, open the browser's site settings (usually a small icon in the address bar) and switch the microphone back to **Allow**.

**Privacy note:** your audio is processed entirely on your own GPU hardware. Nothing is sent to any outside service, and there is no telemetry — speech-to-text, the language model, and text-to-speech all run locally.

### A guided tour of the interface

EchoMind keeps its layout simple on purpose: a sidebar to move between tools, a header to tell you where you are, and a large main area where the work happens.

#### The left sidebar

At the top of the sidebar you'll see the **EchoMind** logo and the line **by Ajace AI**. Below it are the five tabs that make up the platform. The tab you're currently in is highlighted in cyan.

| Tab | What it's for |
|---|---|
| **Knowledge Chat** | Ask questions over your uploaded documents and transcripts and get answers with citations. |
| **Live Transcript** | Turn live speech into text in real time, with a Silent Assistant that fact-checks statements against your knowledge base. |
| **Conversation** | Hold a natural, spoken back-and-forth with the assistant — it listens, thinks, and talks back. |
| **Document Studio** | Generate a polished, formatted document from a topic, a chat, or your sources, then export to PDF and PPTX. |
| **Settings** | Choose your persona, voice, knowledge time-window, and voice-assistant preferences. |

Click any tab to switch to it. **Knowledge Chat** has a bonus: while you're in it, the sidebar grows a **Chat history** list with a **New chat** button, so you can jump between past conversations or start a fresh one. Hover a chat to reveal a trash icon for deleting it (you'll be asked to confirm).

At the very bottom sits a small **Usage** meter showing how much of your Vector DB storage your documents and transcripts are using.

**Heads up:** on a phone or narrow window the sidebar tucks away. Tap the **menu icon** (three lines) at the top-left of the header to slide it open, and tap outside it or the close icon to dismiss it.

#### The header

The bar across the top simply names where you are, so you always have your bearings. The title updates with each tab — for example, the **Live Transcript** tab reads "Live Transcription & Refinement" in the header, **Conversation** reads "Voice AI Conversation," and **Settings** reads "Platform Settings." On small screens, the header also holds the menu button that opens the sidebar.

#### The main area

Everything else is the workspace for the active tab — your chat thread, the live transcript, the voice conversation view, the document builder, or your settings. It scrolls on its own, so the sidebar and header stay put as you work.

### What to expect on first use

A couple of things make the first run feel smooth once you know them:

- **Your work stays put when you switch tabs.** A running Live Transcript keeps transcribing, and your Knowledge Chat conversation is preserved, even if you pop over to another tab. The one exception is **Conversation**: leaving that tab disconnects the voice session on purpose, so the mic isn't held open in the background.
- **An empty knowledge base is normal at first.** Until you upload documents or capture a transcript, Knowledge Chat has nothing to cite. Head to Knowledge Chat to add your first files — that's covered in the next chapter.
- **Set your preferences once.** Open **Settings** early to pick a **persona** (Teacher/Professor, Financial Advisor, Lawyer, AI Expert & Manager, General Assistant, Funny & Calming, or EchoMind Guide), a **voice** for spoken replies, and a knowledge **time window**. Your choices are saved in your browser for next time.

**Try this:** open the app, click through all five tabs once to see each workspace, then return to Settings and choose your persona. You're now ready to load your first documents.

---

## 4. Knowledge Chat — Ask Your Documents Anything

Knowledge Chat turns your private library into a conversation. Drop in your PDFs, Word files, and slide decks, then ask plain-English questions and get grounded answers — every claim backed by a real citation you can open and read for yourself. Everything runs on your own hardware, so your documents never leave the building.

### What you can do here

You ask questions; EchoMind answers from **your** material — uploaded documents *and* meeting transcripts captured elsewhere in the app. Answers stream in as they're written, and each one carries clickable **Sources** showing exactly which document, section, and page the answer came from.

**Why it matters:** This is not a chatbot guessing from the internet. It reads your corpus and tells you where it found things, so you can verify every word.

### Upload your documents

The right-hand **Resources** panel is your knowledge base. (On a phone, tap the file icon in the chat header to slide it open.)

1. Make sure the **Resources** tab is selected.
2. Use the uploader at the top: drag a file onto it, or click to browse.
3. Supported formats: **PDF**, **DOCX** (Word), and **PPTX** (PowerPoint).
4. Watch it appear in the list below once **indexing** finishes.

**What indexing does:** EchoMind extracts the text, strips repeated headers/footers, tracks page numbers, splits the document into smart overlapping chunks, and builds a searchable index. Large structured documents (think a multi-thousand-page regulation) get extra "book-aware" treatment — a section map and table-of-contents routing — so retrieval stays precise even in huge files.

**Heads up:** Scanned/image-only PDFs won't work (there's no OCR). If a file has no extractable text, you'll get a clear message asking you to re-upload a text-based version.

### Ask a question and read the answer

1. Type your question in the box at the bottom and press **Enter** (or click **Send**).
2. You'll see a **Thinking…** indicator, then the answer **streams in** token by token.
3. When it finishes, a **Sources** button appears under the answer with a count badge.

**Pro tip — write better questions.** Be specific and use the document's own vocabulary. Ask *"What liability does a certifying officer have for an improper payment?"* rather than *"tell me about payments."* Specific, well-phrased questions retrieve sharper evidence and better answers.

### Trust the citations

Click **Sources** to open the citation panel. You'll see a numbered list of the exact chunks behind the answer. Click any one to see:

- **File** — the source document's name.
- **Path / Section** — the section trail the text lives under.
- **Page** — the page number in the original.
- **Relevance** — a match percentage so you can judge strength at a glance.
- A **doc-type badge** (Book, Glossary, FAQ, Transcript) and the **retrieved text** itself.

For documents, click **View in Document** to open the original PDF in a new tab, jumped to the cited page.

**Why to trust them:** Answers are grounded in retrieved text that's shown to you verbatim. Nothing is invented — if the evidence is weak, the assistant is built to say so rather than make something up.

### Choose a persona

Your assistant's voice and lens come from the **persona** set in Settings. The active one is shown as a chip in the chat header. Options:

| Persona | Best for |
|---|---|
| Teacher / Professor | Clear explanations, analogies, step-by-step lessons |
| Financial Advisor | Regulations, compliance, budget and DoD FMR detail |
| Lawyer | Obligations, risks, structured legal reasoning |
| AI Expert & Manager | Architecture, engineering decisions, action items |
| General Assistant | Direct, everyday answers and writing help |
| Funny & Calming | Warm, easy-to-digest summaries |
| EchoMind Guide | How EchoMind itself works |

The persona changes tone and emphasis — it never changes the facts or the sources.

### Filter by time and choose your sources

**Time window.** A context filter (**24h / 48h / 1w / all**, set in Settings) limits how far back the assistant looks. Pair it with phrases like *"summarize the last 2 hours"* or *"recent transcript summary"* and EchoMind pulls only transcripts from that window. Great for "what did we decide this morning?"

**Source options.** EchoMind can draw from three places — your **documents**, your **transcripts**, and **general** knowledge for everyday questions. By default all three are on, so a single question searches your whole knowledge base at once.

### Multi-turn conversations and memory

This is a real conversation, not one-shot Q&A.

- Each chat keeps its full message history, so you can ask follow-ups like *"and what about the exceptions?"* without repeating context.
- Behind the scenes, EchoMind maintains a rolling **conversation summary** so long chats stay coherent without re-sending everything.
- Start a fresh thread anytime with the **+ (New chat)** button in the header. The first message becomes the chat's title for easy scanning later.

**Try this:** Ask a broad question, then drill down with two or three follow-ups. The assistant remembers what you've been discussing.

### Manage your knowledge base

In the **Resources** panel you can:

- **Search** your documents by name.
- **Delete** any document with the trash icon (confirm the prompt — this removes it from the index and can't be undone).
- Switch to the **Transcripts** tab to browse meeting transcripts captured by Live Transcription. Each shows its keyword **tags** and date; use the **eye icon** to preview the raw text and refined notes, or the trash icon to remove it from storage and embeddings.

**Transcripts are searchable here too.** Anything captured in Live Transcription is auto-saved into the same knowledge base, so Knowledge Chat answers from your meetings just like your documents — and cites them with a **Transcript** badge.

### How it finds answers (the short version)

EchoMind uses **hybrid search**: a meaning-based vector search plus a keyword (BM25) search, fused together so you get both conceptual and exact-term matches. Top candidates are then **reranked** by a cross-encoder that re-reads each one against your question for true relevance. For large structured documents, **book-aware retrieval** routes through the table of contents and section map, and a dedicated glossary index handles definition questions. The result: precise evidence, surfaced as the citations you can open and verify.

---

## 5. Live Transcription & the Silent Assistant

Some conversations are too important to half-remember. A compliance call, a board interview, a lecture you'll need to quote next week — **Live Transcription** turns the spoken word into a clean, searchable record as it happens, and the **Silent Assistant** sits beside it, quietly checking each claim against your own documents. Everything runs on your own hardware: the audio, the transcript, and the fact-checks never leave the building.

### Starting a live session

Open the **Live Transcription** module. You'll see the **Real-Time Transcription** header with a **Stopped** badge and a **Start** button on the right.

1. Click **Start**. The **Start transcription** dialog opens.
2. Fill in two fields:
   - **Name** — a label for the transcript (e.g. `Q3 Compliance Call`). Leave it blank and EchoMind assigns a timestamped default like `transcript_2025-02-12_14-30`.
   - **Location** — where the conversation is happening (e.g. `Office`). Blank defaults to `default`.
3. Click **Start**. (Prefer the auto-generated name? Click **Default** to fill both fields instantly, then **Start**.)
4. Your browser asks for **microphone access** — click **Allow**. This is required; if you deny it, you'll see "Microphone access denied or unavailable."

You'll briefly see **Connecting…** and **Loading STT…** while the on-device speech engine spins up, then the badge flips to a red **Live** with animated level bars and "Listening…".

**Why it matters:** Name and Location aren't just labels — they're saved with the transcript and fed into your knowledge base, so a later search like "what did we agree in the Q3 office call" can find it.

### Watching the transcript form

Speak naturally. Words appear first as faint, in-progress text (the live **partial**), then settle into solid **paragraphs** as the engine detects complete sentences and natural pauses. You don't manage any of this — punctuation and silence rules group your speech into readable blocks automatically. The view auto-scrolls so the latest line is always in sight.

**Heads up:** if a network hiccup drops the connection, EchoMind reconnects on its own (up to several attempts) and **keeps your visible transcript on screen** — you won't lose what's already been said.

### The Silent Assistant: live fact-checking

To the right of the transcript is the **Silent Assistant** panel. Each time a paragraph finalizes, it searches your **uploaded documents** (not other transcripts) and asks the local LLM to judge the statement. When it finds something worth flagging, an **analysis card** appears.

**Pro tip:** The Silent Assistant only checks against documents you've uploaded to **Knowledge Chat**. With an empty knowledge base it stays quiet — load your policies, contracts, or course material first.

Each card carries one of five labels:

| Label | Icon | Meaning |
|---|---|---|
| **Supported** | ✓ | Confirmed or strongly backed by your documents |
| **Contradicted** | ✗ | Conflicts with your documents |
| **Unverified** | ? | On-topic but not confirmable from references |
| **Violating** | ⚠ | Appears to break a rule, policy, or regulation in your docs |
| **Risky Statement** | ⚡ | Potentially dangerous, misleading, or risky per your docs |

Every card shows a **confidence** percentage and a one-to-two-sentence explanation. The Assistant deliberately stays silent on casual chatter and only surfaces a card when confidence is **60% or higher** — so you get signal, not noise. While it's thinking, a spinner and "Analyzing…" appear; matching paragraphs in the transcript are tinted and color-coded by label.

#### Opening a card and its sources

Click any card to open the full view. You'll see the exact **spoken statement** quoted, a **confidence bar**, the **"Why this label"** explanation, and **Reference Sources** — the document excerpts the judgment was based on, each headed by its file name. Click a paragraph in the transcript and its card highlights, and vice versa, so you always know which claim a card refers to. Press **Esc** or **Close** to return.

**Try this — Handraise (voice readout):** use the **Handraise** button at the top of the panel to have a card or a **full summary** read aloud — handy when you're watching the room instead of the screen.

### Auto-save to your knowledge base

You don't have to remember to save. The session **auto-stores** to your knowledge base about every **60 seconds** and again when you stop — writing the transcript to storage and embedding it so it's instantly searchable in **Knowledge Chat**. EchoMind also infers a conversation type and suggested tags as it saves.

There's no save button to hunt for — capture is automatic. When you reopen a saved transcript later (from session history, or the **Transcripts** tab in Knowledge Chat), you'll see its full raw text and, when a cleaned-up polished version is available, that too — a tidy record alongside the verbatim capture.

### Editing details, pausing, and ending

While a session runs, the **session info bar** lets you edit on the fly:

- **Name** and **Location** — type directly into the fields; changes are saved with the transcript.
- **Tags** — type a tag and press **Enter** or click **Add**; remove one with its **×**.
- **Date & time** — shown for reference.

To **pause** without ending: click the **mic button** to mute. The status reads "Muted" and no audio is sent; click again to resume. The session and transcript stay intact.

To **end**: click **Stop**. EchoMind finalizes the transcript, does a last save, and automatically extracts suggested tags from the full text. Use the **trash** icon to clear the screen and start fresh.

### Reviewing past transcripts

Click the **history (clock) icon** in the header to open your **session history**. From there you can reopen any past transcript, read its full text and any refined notes, see its name, location, tags, and date, and review the **analysis cards** that were captured live — including ones flagged before the first auto-save. You can also delete a transcript, which removes it from the knowledge base too.

### Use cases

- **Compliance calls:** load your policy documents first. As the call runs, **Violating** and **Contradicted** cards flag statements that clash with your rules — in the moment, not in a post-call review.
- **Lectures:** capture a clean, paragraph-formatted transcript while **Supported** cards confirm claims against your course readings; review it later in Knowledge Chat.
- **Interviews:** name the session after the candidate, tag it, and let it auto-save. Afterward, search across every interview by topic or by who said what.

**Bonus — Boardroom Mode:** the Start dialog has a **Boardroom Mode** toggle that records full-quality audio for a speaker-by-speaker meeting report. That workflow has its own chapter.

> **Security & admin note:** All transcription, fact-checking, and storage happen entirely on your own GPU hardware — no audio or text is sent to any third-party service. As with every EchoMind module, deploy it on a trusted network (behind your own SSO, VPN, or reverse proxy) so only authorized people on your network can reach it.

---

## 6. Boardroom — Meetings into Decisions

Boardroom turns a recorded meeting into something you can act on: a clean, speaker-by-speaker transcript and a structured AI report that tells you who said what, which claims hold up against your own documents, and what to do next. It all runs on your own hardware — no recording ever leaves the box.

### What Boardroom is for

Use **Boardroom** when you want a *finished record* of a whole meeting: a transcript that separates each voice plus a written report you can hand to people who weren't in the room. Boardroom records the full session, figures out **who spoke when** (diarization), then writes an executive-ready summary and exports it to **PDF** or **PPTX**.

**Why it matters:** a raw recording is hard to use. Boardroom gives you a labelled transcript *and* a decision-focused report — including a fact-check of meeting claims against your knowledge base — in one place.

#### Boardroom vs. Live Transcription

| | **Boardroom** | **Live Transcription** |
|---|---|---|
| Best for | A whole meeting → report after the fact | Watching a conversation as it happens |
| Output | Speaker-diarized transcript + AI report (PDF/PPTX) | Live transcript + per-statement fact-check cards |
| Speaker labels | Yes (Speaker 1/2/3…) | No |
| When you see results | After you stop and analyse | In real time, sentence by sentence |

**Pro tip:** they pair well. Run **Live Transcription** during the meeting for instant fact-check cards, then let **Boardroom** produce the polished report afterward — you can even link the two (see *Linking a live transcript*).

### Recording a meeting

1. Start a new **Boardroom** session. It opens in **Recording in Boardroom Mode** — you'll see a pulsing mic and the note *"Stop the session to upload and transcribe audio."*
2. Speak / let the meeting run. Boardroom captures audio in the browser and uploads it in chunks as you go, so a long meeting isn't held until the end.
3. **Stop** the session when the meeting ends. This finalizes the recording.

**Heads up:** Boardroom captures whatever audio your browser/mic feeds it. For multi-person meetings, place the mic so everyone is audible — clearer audio means cleaner speaker separation.

### What "diarization" means

**Diarization** is the step that answers *who said what*. Instead of one undifferentiated wall of text, Boardroom splits the audio by voice and labels each turn **Speaker 1**, **Speaker 2**, and so on. In the **Transcript** tab each speaker gets its own colour-coded card with a coloured dot, the speaker label, the timestamp, and the spoken text.

**Heads up:** speakers are identified by voice, not by name — Boardroom doesn't know real names, so it uses generic labels. You decide who "Speaker 2" is when you read or share the report.

### The processing stages (and the status you see)

After you stop, the session moves through a fixed sequence. The view shows a live **status badge** and polls for updates every 3 seconds, so you don't need to refresh.

| Stage | Status badge | What's happening |
|---|---|---|
| `recording` | (mic, "Recording in Boardroom Mode") | Capturing/uploading audio |
| `processing` | **Transcribing…** (amber) | Audio is being transcribed and diarized |
| `transcribed` | **✓ Transcribed** (green) | Transcript ready; **Analyse Meeting** appears |
| `analysing` | **Analysing…** (cyan) | AI is building the report |
| `analysed` | **✓ Report Ready** (violet) | Report done; **AI Report** tab + exports appear |

While it works you'll see a friendly progress line — *"Transcribing audio with VibeVoice-ASR…"* during processing, *"Analysing meeting with AI…"* during analysis.

### Generating the AI report

Once the badge reads **✓ Transcribed**, click **Analyse Meeting** in the header. The status switches to **Analysing…**; when it finishes you'll see **✓ Report Ready** and a new **AI Report** tab.

**What's in the report:**

- **Sentiment** — an overall read on the meeting's tone, shown as a small pill at the top.
- **Executive Summary** — a few sentences capturing the meeting at a glance.
- **Key Topics** — the main themes, as quick tags.
- **Speaker Breakdown** — a per-speaker summary plus that speaker's **key points**, colour-matched to the transcript.
- **RAG-Verified Facts** — claims from the meeting that are *supported* by your uploaded documents (green ✓).
- **Contradictions / Risks** — statements that *conflict with* your documents or raise a flag (amber ⚠).
- **Recommendations** — suggested next steps drawn from the discussion.

**Why it matters:** the **RAG-Verified Facts** and **Contradictions / Risks** sections check the meeting against *your* knowledge base — so you instantly see which decisions are grounded in your documents and which claims need a second look. Everything is generated locally; nothing is sent to an outside service.

**Pro tip:** if a report looks thin, you can re-run **Analyse Meeting** — a re-analysis keeps your existing report until the new one is ready, so you never lose what you had.

### Transcript vs. Report tabs

Once a transcript exists, two tabs appear at the top:

- **Transcript** — the full diarized conversation, speaker by speaker, with timestamps. This is your verbatim record.
- **AI Report** — the structured summary above. (This tab only appears after a report is generated.)

Switch freely between them — the **Transcript** is the evidence, the **AI Report** is the interpretation.

### Exporting to PDF or PPTX

Once the report is ready, two export buttons appear in the header:

- **PDF** — a formatted document, ideal for filing, email, or printing.
- **PPT** — a slide deck (PPTX), ideal for presenting the outcome to a group.

Click either; the file downloads automatically as `boardroom_<id>.pdf` / `.pptx`. **Both exports are built on-device** from the same report.

**Try this:** export the **PPT** for a leadership readout and the **PDF** for the meeting archive — same content, two formats.

### Linking a live transcript

If you ran **Live Transcription** during the same meeting, you can **link** that live-transcript record to your Boardroom session so the two are tied together. The link is established when transcription auto-saves, keeping your real-time notes and your post-meeting report connected as one body of work.

### A quick word on privacy

Every stage — recording, diarization, the AI report, and both exports — runs entirely on your own NVIDIA hardware. No audio, transcript, or report is sent to any third-party AI service, and there's no telemetry. **Best practice:** because this build ships without built-in login, run EchoMind on a trusted, isolated network (behind your own SSO, VPN, or reverse proxy) so only the right people can reach your meeting records.

---

## 7. Voice Conversation — Talk to Your Knowledge

Imagine asking your knowledge base a question out loud — and getting a spoken answer back, in a voice you chose, from a persona you picked, grounded in your own documents. That's Voice Conversation: a hands-free, full-duplex assistant that listens, thinks, and talks like a person who's genuinely paying attention. And like everything in EchoMind, it runs entirely on your own hardware — your voice never leaves the building.

### Starting a voice session

Voice Conversation needs two things: a working **microphone** and a **secure connection**. Browsers only hand microphone access to pages served over HTTPS (or `localhost`), so open EchoMind at `https://<your-host>:3443` rather than the plain HTTP address when you want to talk.

**How to start:**

1. Open the **Voice Conversation** module from the EchoMind sidebar.
2. Click **Connect** (the button on the conversation stage).
3. Your browser will ask for **microphone permission** — click **Allow**. You only do this once per browser.
4. Two orbs appear — yours and the assistant's. When both settle to a calm idle glow, you're live. Just start talking.

**Heads up:** If you see "Microphone access denied," re-enable the mic in your browser's site settings and click Connect again. "Could not connect to the voice service" means the voice container isn't reachable — confirm it's running and retry. EchoMind keeps your on-screen transcript intact through brief hiccups.

**Pro tip:** Use the **mute** control to silence your mic for a moment (to take a phone call, say) without ending the session, and **Stop** to disconnect fully.

### The natural conversation — what makes it feel human

These features are always on. You don't configure them; you just talk, and the assistant keeps up.

- **Barge-in — just talk over it.** Changed your mind mid-answer? Start speaking and the assistant stops *instantly* and listens. No "let me finish" awkwardness — interrupting is how you steer.
- **Instant lead phrases — no dead air.** The moment you finish a question, you'll hear a quick filler like "Let me check that…" while the real answer is being retrieved and generated. The assistant's orb shows a soft "thinking" glow so you know it's working, not stuck.
- **Backchannels.** During a long explanation from *you*, the assistant murmurs a brief "mm-hmm" or "I see" so the line never feels dead. It uses echo cancellation to keep its own voice out of your mic, so these never trigger by accident.
- **It ends your turn smartly.** Adaptive silence detection (semantic endpointing) figures out when you've actually finished a sentence versus when you're just pausing to think — so it replies faster when you're done and waits patiently when you're not.

### Listen-only mode and the wake word

Sometimes you want EchoMind in the room without it jumping into every sentence — say, during a discussion you only occasionally want to consult.

- **Listen-only mode** keeps the assistant quietly transcribing without replying. Toggle it on the stage, or just say **"start listening"** (also "just listen" / "keep listening"). You'll see the running text accumulate live.
- **The wake word** brings it back. By default that's **"EchoMind"** — say it (optionally followed by your question) and the assistant responds to that turn. Say **"stop listening"** to leave listen-only.

**Change the wake word by voice.** Say something like "rename you to Atlas" or "change the wake word to Nova." The assistant confirms first — "Do you want to change the wake word to Nova? Say yes to confirm" — so a stray phrase never silently renames it.

### Memory questions — ask about what was just said

Voice Conversation remembers a rolling **30-minute window** of the conversation, so you can interrogate the recent past out loud:

- "**What did I say in the last 5 minutes?**" (recap)
- "**Summarize the last 10 minutes.**"
- "**When did we talk about the budget?**"
- "Give me **timestamps and tags** / who said what."

**Why it matters:** In a working session you can offload remembering. Ask, and the assistant pulls the answer from its memory of the conversation — no scrolling, no notes.

### Voice commands you can speak

Beyond questions, plain spoken commands set things up on the fly:

| Say something like | What happens |
|---|---|
| "Call yourself Nova" / "Your name is Nova" | Sets the **assistant's name** immediately |
| "My name is Sam" / "Call me Sam" | Sets **your name**; it'll address you by it |
| "I'm in Berlin" / "Set timezone to CET" | Sets **location / timezone** for context |
| "Fact-check that" / "Verify it" | Re-checks the last point against your documents |
| "Clear memory" / "Forget everything" | **Wipes** the conversation memory |

You can also clear memory with the on-screen control — handy before switching topics.

### Choosing a persona, names, and voice

Open **Settings** to shape the assistant before (or during) a session. Changes apply live.

- **Persona** sets expertise and tone. Choose from **Teacher/Professor**, **Financial Advisor**, **Lawyer**, **AI Expert & Manager**, **General Assistant**, **Funny & Calming**, or **EchoMind Guide**. Each stays in its lane — the Lawyer persona, for instance, always notes that its analysis isn't formal legal advice.
- **Assistant name** and **your name** personalize the exchange. The assistant adopts the name as its single identity and weaves your name in naturally.
- **Voice** picks the synthesized speaking voice (a Piper voice such as `en_US-lessac-medium`). Pick the one that's easiest on your ears for long sessions.

### Knowledge-base mode — spoken answers from your documents

Voice Conversation is connected to your knowledge base, so questions about your uploaded documents and saved transcripts are answered straight from RAG — same retrieval and citations engine behind Knowledge Chat, just spoken aloud. Ask "What does our travel policy say about per diem?" and the reply comes from *your* corpus, not generic guesswork. Retrieved text is always treated as untrusted data, so a malicious document can't hijack the assistant.

### Etiquette tips for best results

- **One thought at a time.** Ask, pause, listen. Short, clear turns beat rambling paragraphs.
- **Interrupt freely.** Barge-in is a feature — cut in the second you have what you need.
- **Use a decent mic in a quiet-ish room.** Echo cancellation and noise suppression are on, but cleaner input means sharper transcription.
- **Name things early.** Say your name and pick a persona at the start so the whole session feels tailored.
- **Lean on memory.** When you lose the thread, just ask "what did I say a few minutes ago?"

**Security note (best practice):** Voice Conversation, like the rest of EchoMind, is built for a **trusted, isolated network** — no data ever leaves your hardware. If you plan to reach it from beyond your LAN, put it behind your own SSO, VPN, or authenticating reverse proxy first.

---

## 8. Document Studio — Generate Polished Documents

Document Studio is where a rough idea, a finished chat, or a folder of source files becomes a polished, professionally laid-out document — a technical spec, a board report, a pitch deck, a brand book — exported as a real PDF or PowerPoint, optionally illustrated with images your own GPU draws. Everything is generated on your hardware: no cloud, no upload of your content anywhere, no third-party AI service.

### What you can build, three ways to start it

Every document begins by answering one question: where does the content come from? Document Studio gives you three sources, chosen with the **From Chat / From Documents / From Brief** toggle.

- **From Chat** — turn an existing Knowledge Chat conversation into a structured deliverable (great for "write that up as a report"). Pick the conversation from the dropdown.
- **From Documents** — ground the document in files you've already uploaded to the knowledge base. Tick one or more documents in the list; a counter shows how many you've selected.
- **From Brief** — start from nothing but your own instructions. Describe the topic, audience, and key points, and the generator writes the whole document from your words alone.

**Why it matters:** the same template can produce a one-pager from a brief or a thoroughly sourced report from your documents — you choose how much grounding it gets.

### The 18 built-in templates

Pick a template from the gallery (Step 1). Each card shows its name, the **persona** it's tuned for, a small theme-colour dot, and an **images** badge if it supports illustrations. Here is the full set:

| # | Template | One-line description |
|---|---|---|
| 1 | **Technical Document** | Engineering/system-design spec: purpose, control principles, numbered process flows, activity tables, status-codes appendix. |
| 2 | **Business / Executive Report** | Executive-ready report: summary, key findings, analysis, risks, and recommendations with supporting tables. |
| 3 | **SOP / Process Document** | Standard Operating Procedure: purpose, scope, roles, step-by-step procedure, controls, and references. |
| 4 | **Training / Learning Guide** | Learning material: objectives, modules with examples, key takeaways, and review questions. |
| 5 | **Legal / Compliance Brief** | IRAC-structured legal analysis citing clauses/regulations, with obligations, risks, and a disclaimer. |
| 6 | **Meeting / Conversation Summary** | Clean recap of a chat or transcript: overview, decisions, action items, and open questions. |
| 7 | **Pitch Deck** | Investor/sales deck: problem, solution, market, product, model, traction, team, and the ask. Best as PPTX. |
| 8 | **Whitepaper / Research Report** | In-depth paper: abstract, intro, background, approach/architecture, evaluation, discussion, conclusion. |
| 9 | **Product Requirements (PRD)** | Goals & non-goals, users & use cases, numbered requirements, UX flows, metrics, and milestones. |
| 10 | **Project Proposal** | Persuasive proposal: summary, objectives, scope, approach, timeline, budget, risks, and outcomes. |
| 11 | **Case Study** | Marketing case study: background, challenge, solution, quantified results, and a closing CTA. |
| 12 | **System Architecture Document** | C4-style architecture: context, requirements, components, data flow, deployment, and trade-offs. |
| 13 | **Marketing Book / GTM Playbook** | Board-ready handbook: market, positioning, funnel, channels, campaigns, KPIs, and a 90-day plan. |
| 14 | **Promotional Flyer** | Single-page, scroll-stopping flyer: hook headline, benefit bullets, and one bold call-to-action. |
| 15 | **Brand Book / Brand Guidelines** | Editorial brand bible: story, values, voice, logo/color/type system, imagery, and governance. |
| 16 | **Marketing Campaign Plan** | Launch-ready plan: objectives, audience, big idea, channel mix, timeline, budget, KPIs, and sign-off. |
| 17 | **Social Media Playbook** | Per-platform strategy, content pillars, a 7-day calendar, engagement rules, and a metrics dashboard. |
| 18 | **Product Launch / GTM Plan** | Cross-functional launch plan: positioning, segments, pricing, timeline, enablement, metrics, and risks. |

**Pro tip:** the dot colour signals the visual theme (midnight, executive, evergreen, counsel, spotlight), so a Legal Brief comes out in warm amber and a Pitch Deck in bold magenta without any setup.

### Turning AI images on or off

Below the details fields is a **Generate with images** switch. When on, an "art-director" step plans a cover hero and per-section illustrations, then renders them on-device with **SDXL-Turbo** — no image ever leaves your machine.

- If an image backend is configured, the switch defaults **on** and shows the active backend (e.g. *Image backend: diffusers*).
- If none is configured, you'll see a note that placeholders will be used — the document still generates cleanly, just with neat placeholder frames instead of art.
- Toggle it **off** for faster, text-only output.

**Heads up:** with images enabled, SDXL-Turbo holds roughly 18 GB of GPU memory alongside your other models — plan capacity accordingly on a busy box.

### Generate, watch progress, and preview

1. Add optional **Document title** and **Organisation / footer** in Step 3, plus an optional **brief** to steer tone or emphasis (for Brief mode this field is required).
2. Click **Generate Document**. The panel switches to a live progress view with an animated **Queued → Planning → Writing → Assembling → Ready** stepper, so you can watch the pipeline plan the outline, write sections in parallel, illustrate, and assemble.
3. When it reaches **Ready**, the full document renders inline: cover image, title, theme/persona tags, headings, paragraphs, bullets, tables, flow diagrams, and callouts — exactly the structure that will export.

Every job also lands in **Recent Documents** at the bottom. Click any entry to reopen its preview; if it's still running you rejoin the live progress. Use the trash icon to delete one. Click **New Document** in the header to start over.

### Exporting to PDF and PPTX

From a finished preview, use **Export PDF** or **Export PPTX** (top-right). The file downloads in a new tab — a true reportlab PDF (cover, table of contents, themed sections, banded tables) or a 16:9 python-pptx deck (themed slides, native bullets, image slides). Re-exports are always freshly rendered, never a stale cached copy, so a regenerated job gives you the current file.

### Uploading your own template

Want output in your house style? Click **Upload template** (top-right of Step 1) and pick a **.pptx, .docx, .pdf, .md, or .txt** file (up to 25 MB). Document Studio reads the file's structure — slide titles, Word "Heading" styles, or heading-like lines — and builds a matching section blueprint, tagged **Custom** in the gallery and auto-selected for you.

**Best part:** upload a **.pptx** and the export renders generated content *into your actual deck*, preserving its master, theme, and colours. Generate as usual, then **Export PPTX** to get slides in your brand's design.

### Tips for great output

- **Be specific in the brief** — name the audience, the must-cover points, and any numbers or constraints. The generator stays strictly on-topic and won't invent facts beyond your brief and sources.
- **Match template to source:** Meeting Summary loves *From Chat*; Whitepaper and Architecture shine *From Documents*; Flyer and Brand Book work beautifully *From Brief*.
- **Set the title and org** so covers and footers read correctly in the export.
- **Pick the export to fit the format:** decks (Pitch Deck, GTM Plan) export best as PPTX; reports and briefs as PDF.
- **Privacy by design:** planning, writing, and image generation all run locally on your GPUs — your chats, files, and finished documents stay on your hardware.

---

## 9. Personas — Choose How EchoMind Thinks

A persona is the "mindset" EchoMind brings to a conversation — its expertise, tone, structure, and guardrails. The same powerful models sit underneath every time; the persona decides *how* they show up. Pick the right one and EchoMind explains like a professor, reasons like a lawyer, or cites a regulation like a compliance officer. You choose once, and your choice applies to **both Knowledge Chat and Voice Conversation**.

### Where to choose your persona

There's one place to set it, and it governs everything.

1. Open **Settings** from the left navigation.
2. Find the **Persona Configuration** section at the top — "Choose the AI persona for Knowledge Chat and Voice."
3. You'll see seven persona cards, each with an icon, a name, and a one-line description.
4. **Click a card.** The active card highlights with a colored dot. That's it — there's no Save button; the choice takes effect on your next message or voice turn.

**Why it matters:** Every persona stays grounded in *your* knowledge base. Whichever you pick, EchoMind still consults your uploaded documents and saved transcripts first and cites them — the persona only changes the voice, structure, and subject focus on top of that grounding.

**Heads up:** Each persona carries a **guardrail**. The specialists (Financial Advisor, Lawyer, AI Expert) will politely redirect questions outside their domain. If you want one assistant for anything, use **General Assistant** or **Funny & Calming**.

### The seven personas

#### 🎓 Teacher / Professor

Your patient explainer. It opens with a clear overview, then builds depth using analogies, numbered steps, and structured breakdowns — and adapts the complexity to how much you already seem to know. It even nudges you toward related concepts and follow-up questions.

**Pick it when** you're learning something new, onboarding into a dense document, or want a concept made genuinely understandable rather than just answered.

#### 💼 Financial Advisor

The strict, citation-first specialist, tuned for the **DoD Financial Management Regulation (FMR)**, government financial procedures, compliance, and audit readiness. It leads with the direct answer, then the regulatory basis, and attaches an inline citation — like `(Volume 5, Chapter 3, Section 030201, page 142)` — to every factual regulatory claim. It will not invent a section number or page; if your documents don't cover it, it says so and points to the likely Volume/Chapter.

**Pick it when** accuracy and traceability are non-negotiable: regulatory questions, compliance checks, payment procedures, audit prep. **This is also the default persona.**

#### ⚖️ Lawyer

A legal analyst that reasons in **IRAC** structure (Issue, Rule, Analysis, Conclusion). It leads with the conclusion, cites the specific statute, clause, or transcript passage from your record, flags risks, obligations, and deadlines, and always closes with a reminder that this is informational analysis, not formal legal advice.

**Pick it when** you're reviewing contracts, interpreting regulations, or want a structured risk read on something in your documents.

#### 🤖 AI Expert & Manager

A senior AI/ML engineer who also thinks like an engineering manager. It gives a direct technical recommendation, then the trade-offs and alternatives, and reaches for pseudocode or architecture sketches when useful. It mines **transcripts** especially hard — design reviews and standups often hold decisions that never made it into a formal doc.

**Pick it when** you need architecture decisions, model/tooling choices, code or system reviews, or management advice (Agile, OKRs, DORA).

#### 💬 General Assistant

The friendly all-rounder. It answers anything — questions, writing, brainstorming, summarizing — leading with a clear, direct answer and scaling length to the ask. It checks your documents and transcripts when relevant and cites them, but happily draws on broad knowledge for everything else.

**Pick it when** you want one capable assistant with no domain fence and no special formatting.

#### 😄 Funny & Calming Assistant

Same broad helpfulness as General, with a warmer, lighter touch. It opens with a friendly (sometimes witty) acknowledgment, keeps answers short and delightful, and adds calming reassurance when a topic feels stressful — but humor never replaces accuracy.

**Pick it when** the mood matters: a long day, a tense topic, or just when you'd rather your assistant feel human and easygoing.

#### 🧠 EchoMind Guide

Your in-app product expert. It explains what EchoMind is, what each feature does, where to find it, and how it all runs **fully on-device on the NVIDIA DGX Spark** — private, offline, GPU-accelerated, with nothing sent to the cloud. It's written for non-technical readers.

**Pick it when** you (or a teammate) are getting started and want to understand how EchoMind works.

### Which persona when

| You want to… | Pick |
|---|---|
| Understand a tough concept, step by step | 🎓 Teacher / Professor |
| Get cited, audit-ready FMR / compliance answers | 💼 Financial Advisor |
| Analyze contracts or legal risk (IRAC) | ⚖️ Lawyer |
| Decide architecture, models, or eng-management | 🤖 AI Expert & Manager |
| Ask anything, no domain fence | 💬 General Assistant |
| Keep it warm, light, low-stress | 😄 Funny & Calming Assistant |
| Learn how EchoMind itself works | 🧠 EchoMind Guide |

### Using personas in Voice

The persona you set in Settings travels straight into **Voice Conversation** — the spoken assistant adopts the same expertise, tone, and guardrails. Ask the Lawyer persona a contract question by voice and you'll hear IRAC-style reasoning; switch to Funny & Calming and the spoken replies relax.

**Pro tip:** Personas pair naturally with Voice's other controls. Set the **persona** and **Piper voice** in Settings, name your assistant on the fly by voice (for example, "Call yourself Nova"), and turn the **knowledge base** on for document-grounded spoken answers — together they shape both *what* it knows and *how* it sounds.

**Try this:** Keep **Financial Advisor** for precise, cited document work, then switch to **Teacher / Professor** when you want the same source material *explained*. Same knowledge base, two very different conversations — and switching is a single click in Settings.

---

## 10. Settings & Personalization

The **Settings** tab is where you make EchoMind feel like yours. One place sets the tone of every answer, the voice you hear back, how far into the past the system looks, and what lives in your knowledge base. Change a control here and the effect ripples across Knowledge Chat, Voice Conversation, Live Transcription, and Boardroom — so a minute spent here saves you repeating yourself everywhere else.

### Choosing your default persona

A **persona** is the expert EchoMind becomes when it answers you. It sets the tone, the depth, the structure of replies, and the guardrails — and your choice applies to **both Knowledge Chat and Voice Conversation** at the same time.

Under **Persona Configuration**, you'll see a grid of cards. Tap one to select it; the active card lights up with its accent color and a small dot in the corner. Your seven choices:

| Persona | Best for |
|---|---|
| **Teacher / Professor** | Learning a topic, step-by-step explanations, analogies |
| **Financial Advisor** | DoD FMR, government finance, compliance, precise section citations |
| **Lawyer** | Contracts, regulations, risk — uses IRAC structure with legal disclaimers |
| **AI Expert & Manager** | AI architecture, system design, engineering and team-leadership advice |
| **General Assistant** | Everyday questions, writing, brainstorming across any topic |
| **Funny & Calming Assistant** | A lighter, warm, witty touch for low-stress help |
| **EchoMind Guide** | Questions about EchoMind itself — how the app works and runs on-device |

**Why it matters:** the persona changes *how* answers are written, not *what* facts exist. Switch to **Lawyer** and the same document yields an IRAC-structured analysis with disclaimers; switch to **Teacher** and you get the same facts explained with analogies.

**Try this:** select **EchoMind Guide**, then ask in Knowledge Chat, "What can you do?" — you'll get a tour of the platform in plain language.

### Voice preferences (Piper TTS)

The **Voice & Audio (Piper TTS)** section controls the spoken voice you hear in **Voice Conversation**. It only affects spoken replies — typed Knowledge Chat answers are unchanged.

How to use it:

1. Open **Voice & Audio (Piper TTS)**. You'll see a row of selectable `en_US` voice chips.
2. Click a voice to make it active. The selected voice turns solid violet.
3. Voices already on your machine show a small **✓**. A voice you haven't used yet downloads **automatically** the moment you pick it — you'll see "Downloading voice…" briefly, then it becomes active.

**Heads up:** if the voice server isn't reachable yet, you'll see an amber note. You can still pick a voice — it will be fetched when the server is back. Everything is served on your own hardware; choosing a voice never reaches out to the internet at runtime.

### Context-window defaults (Retrieval Window)

Under **Knowledge Base Context**, the **Retrieval Window** decides how far back in time EchoMind looks when it searches your knowledge base for an answer. Pick one of four:

- **24h** — only the last day
- **48h** — the last two days
- **1w** — the last week
- **All Time** — everything you've stored

**Why it matters:** this is your temporal focus knob for **RAG retrieval**. If you only care about what was discussed in recent meetings, set **24h** and chat answers stay grounded in fresh transcripts. For research across your whole corpus, use **All Time**. The setting feeds Knowledge Chat and the RAG-grounded answers in Voice Conversation.

**Pro tip:** if a question about an older document comes back thin, widen the window to **All Time** — a narrow window may simply be filtering the right source out by date.

### Knowledge base & data management

A couple of these data controls live here in **Settings**; the rest live where you actually work with the content, in **Knowledge Chat**. Either way, everything — your uploaded **documents**, the **chunks** they're split into for retrieval, and your **transcripts** — lives in a single on-device database, and nothing leaves your hardware.

**Add sample transcripts (RAG testing).** In the amber **RAG Testing** card, click **Add sample transcripts**. EchoMind seeds your knowledge base with realistic meeting transcripts spanning the last 48 hours plus two fixed dates (Dec 1, 2025 and Oct 10, 2025), then confirms how many were added.

- They appear instantly in the **Transcripts** tab and become searchable in Knowledge Chat.
- **Try this:** with the samples added, ask "what happened in the last 2 hours", "pricing", "2 Dec 2025", or "10 Oct 2025" to see time-range retrieval and the Retrieval Window working together.

**Remove a document or transcript.** Day-to-day cleanup happens in **Knowledge Chat** (Chapter 4), where your **Documents** and **Transcripts** are listed. Use the **trash icon** to delete an item — EchoMind removes the file or transcript, its chunks, and its vectors from every retrieval index at once, so it disappears from Knowledge Chat and Voice answers immediately (re-uploading is the only way back). The **eye icon** previews a transcript's raw text and any polished notes first.

**Starting over completely.** A full wipe — clearing every document, transcript, chat, and index in one step — is an administrator action performed against EchoMind's on-device data volume, not a button in the app. Ask whoever runs your deployment if you need a clean slate (see the Administrator appendix).

### A note for administrators

EchoMind is **private by design**: all inference is local, nothing is sent to third-party AI services, and there's no telemetry — your documents and transcripts stay on your hardware. As a best practice, deploy EchoMind on a **trusted, isolated network**, and if you ever need to expose it more broadly, place it behind your own SSO, VPN, or authenticating reverse proxy first. That keeps the data-management controls in this chapter in the hands of the people you intend.

---

## 11. Privacy, Security & Trust

Your most valuable conversations, your most sensitive documents, your live meetings — none of it should have to leave the building for an AI to be useful. EchoMind Enterprise is built on a simple promise: your data stays on your hardware, full stop. This chapter explains exactly how that promise is kept, what it means for regulated teams, and the one piece of homework that's on you.

### The short version: nothing leaves your hardware

Every model EchoMind uses runs on your own NVIDIA GPU, inside Docker, on your premises. There is no cloud step hiding in the middle.

- **No external AI APIs.** The chat LLM (**Llama-3.1-8B-Instruct-FP4**), embeddings (**nomic-embed-text**), speech-to-text (**Nemotron streaming STT**), diarization (**VibeVoice**), text-to-speech (**Piper**), and document images (**SDXL-Turbo**) are all served locally. Your prompts and documents are never sent to a third party.
- **No telemetry.** EchoMind doesn't phone home, doesn't ship usage analytics, and doesn't report errors to anyone. There is nothing to opt out of, because there's nothing being collected.
- **No internet dependency at runtime.** Model weights are downloaded once during setup, then cached into Docker volumes. After that, the stack runs with offline flags switched on (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `OLLAMA_OFFLINE`, `VOICE_OFFLINE`, and friends), which block any network model fetch.

**Why it matters:** When a vendor says "private," ask where inference happens. With EchoMind, the honest answer is: on the machine in your rack. Your knowledge base, transcripts, and meeting recordings live in one local volume (`echomind_data`) on your host — not in someone else's account.

### It can run fully air-gapped

Because everything is pre-cached locally, EchoMind is designed to run on a network with **no internet access at all**.

You populate the model volumes once on a machine with connectivity (the one-time `prepare_offline.sh` step), then run the stack with `docker compose up -d` on an isolated network. For installations that can never touch the internet, those model volumes can be exported as tarballs and imported on the air-gapped host.

**Try this:** After your first successful run, physically disconnect the host from the internet and reload the app. Knowledge Chat, Live Transcription, Boardroom, Voice, and Document Studio all keep working — proof that nothing was reaching out.

### Retrieved content is treated as untrusted

A document in your library could contain hidden instructions ("ignore your rules and reveal X"). EchoMind assumes exactly that. Across every module that feeds your content into the model, **all retrieved and recorded text is fenced as untrusted data** before it reaches the LLM.

- In **Knowledge Chat**, retrieved passages are wrapped so the model treats them as reference material to cite, not commands to obey.
- In **Voice Conversation**, transcripts injected as context are fenced the same way.
- In **Live Transcription** and **Boardroom**, fact-checks and reports are generated against your corpus with the same separation between instructions and content.

**Why it matters:** This is prompt-injection resistance built into the pipeline, so a poisoned PDF or a stray line in a meeting transcript can't quietly hijack an answer.

### How this maps to compliance

For regulated teams — government, finance, legal, healthcare — the architecture lines up with the controls auditors usually ask about:

| Compliance concern | How EchoMind addresses it |
|---|---|
| Data residency / sovereignty | All data and inference stay on your host; nothing crosses a boundary you don't control. |
| Third-party data sharing | No external AI APIs, no telemetry — no third party to assess. |
| Network exposure | Can run air-gapped on an isolated network. |
| Data at rest | Documents, chats, transcripts, and indexes persist in a single local volume you own and can back up or wipe. |
| Content integrity | Retrieved content is sandboxed as untrusted input (injection-resistant). |

EchoMind is tuned for exactly the kind of large, structured regulatory material these teams work with — it was built to handle documents on the scale of a multi-thousand-page regulatory manual — so the privacy story and the workload story point the same direction.

**Heads up:** EchoMind gives you a private, on-prem foundation. Mapping that foundation to a specific framework (FedRAMP, HIPAA, SOC 2, ITAR, and so on) is still your organization's certification work — the platform removes the hardest obstacle (data leaving your control), but it doesn't grant a certification on its own.

### Admin best practice: deploy on a trusted network

EchoMind is designed to be installed on a **trusted, internal network** — the kind of isolated LAN where the people who can reach the server are already the people allowed to use it. On that footing, it works out of the box.

If you plan to make EchoMind reachable beyond that trusted boundary, treat access control as your responsibility to add at the edge. This is standard practice for on-prem internal tools, and it's quick to set up:

1. **Put it behind your own front door.** Place EchoMind behind your organization's **SSO, VPN, or an authenticating reverse proxy** so only authorized users reach it.
2. **Use trusted HTTPS.** Terminate TLS with a certificate your browsers already trust. EchoMind ships HTTPS on port 3443, and the guides at `docs/HTTPS_TRUSTED_CERTIFICATE.md` (production / Let's Encrypt) and `docs/HTTPS_LOCAL_TRUSTED.md` (local / mkcert) walk you through it with no browser warnings.
3. **Scope access at the proxy.** Restrict who and what can hit the service from your reverse proxy or firewall, rather than exposing the container ports directly.

**Pro tip:** Run EchoMind the way you'd run any internal knowledge system — reachable from inside, gated at the perimeter. Keep it on the trusted network, front it with the SSO/VPN you already operate, and you get the full private-AI experience with access firmly in your hands.

---

## 12. Tips & Best Practices

You already know how to run each module. This chapter is the playbook for getting *great* output instead of merely good — the small habits that turn EchoMind from a search box into a colleague. Skim it, steal the bullets, and come back when a result feels thin.

### Get sharper answers in Knowledge Chat

The quality of a RAG answer is mostly decided by your question and your filters, not by luck. Three levers do the heavy lifting.

**Ask specific, self-contained questions.** The retriever matches your words against your documents, so vague prompts pull vague chunks.

- Name the thing: "What's the **per diem rate** for lodging in the DoD FMR?" beats "tell me about travel."
- Use the document's own vocabulary. If your policy says "obligation," ask about "obligation," not "spending commitment."
- For definitions, ask a definition: "**Define** material weakness" routes straight to the glossary index.
- For one-shot lookups, start a fresh chat. Multi-turn memory is a rolling summary — great for follow-ups, but a long, drifting thread can muddy a precise new question.

**Pick the persona that matches the job.** The persona changes tone and framing, not the facts.

| Persona | Reach for it when |
|---|---|
| **Lawyer** | Compliance, contracts, careful caveats |
| **Financial Advisor** | Budgets, rates, figures from financial docs |
| **Teacher/Professor** | You want it explained, step by step |
| **AI Expert & Manager** | Technical or strategic synthesis |
| **General Assistant** | Everyday, neutral answers |
| **Funny & Calming** | Low-stakes or tense moments |
| **EchoMind Guide** | Questions about EchoMind itself |

**Narrow the source with the time-window filter.** If you only care about this week's uploads and transcripts, set the window to **24h / 48h / 1w** so the retriever stops competing with your whole archive. Switch back to **All** for historical research.

**Pro tip:** Every answer carries citations — document, section path, page. If a citation looks off, click through and check the source. The grounding is doing its job; the citation is your audit trail.

**Heads up:** If the answer says it can't find enough support, that's the **evidence gate** protecting you from a confident guess. Rephrase with the document's exact terms, or upload the source it needs.

### Make your documents index well

EchoMind handles messy files, but clean inputs retrieve better.

- **Upload text-based PDFs, DOCX, and PPTX**, not photos or scans of pages. Real, selectable text embeds far better than an image of text.
- **Keep a real structure** — headings, a table of contents, numbered sections. The "book-aware" pipeline routes through your TOC and section index, so structure literally becomes a retrieval shortcut.
- **Give files descriptive names.** "FY26-Travel-Policy.pdf" is easier to trace in citations than "scan_0007.pdf."
- **One topic per document** where you can. Splitting a 12-subject binder into focused files keeps chunks coherent.
- **Big regulatory documents are welcome.** The system is tuned for very large, structured corpora — upload the whole 7,000-page manual rather than excerpts.

### Clean transcripts and useful Silent-Assistant flags

Live Transcription is only as good as the audio it hears.

- **Use a decent mic in a quiet room.** Less crosstalk means cleaner sentence boundaries and better fact-check cards.
- **Speak in complete sentences and pause naturally.** Paragraphs finalize on punctuation and silence — a clean pause triggers the Silent Assistant sooner.
- **Read the labels for what they are.** **Supported / Contradicted / Unverified / Violating / Risky Statement** each carry an explanation and source references. *Unverified* usually means "not in your knowledge base," not "false."
- **Want richer flags? Upload the reference material first.** The Silent Assistant checks against your documents, so it can only verify what you've indexed.
- **Transcripts auto-save** (about every 60 seconds and on stop) and become searchable in Knowledge Chat. Edit or refine before relying on a transcript for the record.

### Great Boardroom reports

- **Record the whole meeting in one session** so diarization can learn each voice across the conversation.
- **Encourage one speaker at a time.** Overlapping talk is the hardest case for "Speaker 1/2/…" separation.
- **Upload the relevant docs beforehand.** The report's verified facts, contradictions, and recommendations are RAG-checked against your corpus — no source, no cross-check.
- **Let processing finish.** The UI polls every few seconds through `recording → processing → transcribed → analysing → analysed`. Export to **PDF** for reading, **PPTX** for presenting.

### Voice Conversation etiquette

- **Just interrupt.** **Barge-in** stops the assistant instantly — you don't need a wake word mid-conversation.
- **A short filler is normal.** A **lead phrase** ("Let me check that…") covers the moment while RAG runs; it isn't a stall.
- **Backchannels mean it's listening,** not interrupting — keep talking through "mm-hmm."
- **Use memory out loud:** "summarize the last 10 minutes," "what did I say in the last 5 minutes" (rolling 30-minute window). You can also set names, fact-check, recap, or clear memory by voice.
- **Switch to listen-only** when you want it present but silent.

### Best Document Studio output

The brief is the steering wheel. A vague brief yields a generic document.

- **Write a tight brief:** state the audience, the goal, and the must-cover points. "PRD for a mobile checkout redesign, for engineering, covering scope, success metrics, and rollout" outperforms "make a PRD about checkout."
- **Pick the matching template** from the 18 built-ins — or upload your own to enforce a house style.
- **Ground it in sources.** Generating *from a chat* or *from uploaded documents* keeps content factual; *from a brief* is best for net-new material.
- **Enable images when visuals add value** — pitch decks, brand books, flyers, marketing plans. Skip them for dense legal or technical text where they're decoration.
- **Heads up:** Images load the on-device SDXL-Turbo model (~18 GB GPU). Leave them off when the GPU is busy with heavy chat or voice load, and re-enable when you want the polish.
- **Pro tip:** Generation runs as a background job you can poll, and exports never serve stale files — re-export freely after a tweak.

### One admin note

EchoMind keeps everything on your hardware — no outbound API calls, no telemetry, nothing leaves the host. To keep that promise airtight, **deploy it on a trusted, isolated network**, and if you ever expose it more widely, put it behind your own SSO, VPN, or authenticating reverse proxy first.

---

## 13. Troubleshooting & FAQ

Even a well-built system has its odd moments — a microphone that won't wake up, a browser warning that looks scarier than it is, a session that blinks and reconnects. The good news: almost everything here has a quick, known fix, and your data stays exactly where it belongs the whole time. This chapter walks you through the most common questions, solution first.

### Microphone won't turn on

Live Transcription, Boardroom, and Voice Conversation all need your microphone, and browsers only hand it over under two conditions: a **secure page** and your **explicit permission**.

- **Use a secure address.** Browsers grant mic access only on **HTTPS** (`https://<HOST_IP>:3443`) or on **localhost**. If you opened EchoMind over plain `http://` from another machine, the mic button simply won't work. Switch to the HTTPS address.
- **Grant permission.** The first time you start a session, your browser shows a small "Allow microphone?" prompt — click **Allow**. If you dismissed it earlier, click the camera/mic icon in the address bar, set the microphone to **Allow**, and reload the page.
- **Pick the right input.** If your machine has several microphones (headset, webcam, built-in), check your operating system's sound settings and the browser's site permissions to confirm the right one is selected.

**Pro tip:** After changing any permission, do a hard refresh (Ctrl+Shift+R) so the page re-requests the device cleanly.

### "Your connection is not private" — the self-signed HTTPS warning

If you're using the built-in **self-signed certificate**, your browser will flag the page as untrusted. This is expected and does not mean anything is wrong — the traffic is still encrypted; the browser just can't verify the certificate against a public authority because EchoMind runs entirely on your own network.

- **To proceed now:** click **Advanced**, then **Proceed to <host> (unsafe)**. You'll reach the app normally.
- **To remove the warning for good:** install a trusted certificate.
  - **Production / server with a hostname:** get a free subdomain (e.g. DuckDNS) and a Let's Encrypt cert with `sudo certbot --nginx -d <yourname>.duckdns.org`. Full steps live in `docs/HTTPS_TRUSTED_CERTIFICATE.md`.
  - **Local development:** use **mkcert** to issue a locally trusted cert for `localhost`. Full steps in `docs/HTTPS_LOCAL_TRUSTED.md`.

**Why it matters:** Once a trusted cert is in place, the warning disappears for everyone on your network and the microphone "just works" without the secure-page hurdle.

### A voice or transcription session dropped or reconnected

Live Transcription and Voice Conversation run over long-lived WebSockets, and they're built to **heal themselves**. If your network blips or the laptop sleeps, the client automatically reconnects — and Live Transcription **preserves the transcript you can already see** so you don't lose your place.

If a session feels stuck:

1. Wait a few seconds — the auto-reconnect usually restores the stream on its own.
2. If it doesn't, **stop and start** the session again from the same tab.
3. Reload the page (Ctrl+Shift+R) if the controls stop responding.

**Heads up:** Live Transcription has a cap on simultaneous sessions. In a busy deployment, if you can't start a new one, ask whether others can close finished sessions first.

### What happens if the GPU hiccups

EchoMind shares one GPU context across its speech and language models, and a rare fatal CUDA fault can disrupt it. You don't have to fix anything: a **watchdog detects the fault, the affected service reports unhealthy, and Docker restarts it with a clean context** automatically.

- **What you'll see:** a Voice or transcription session may drop for a short moment while the service comes back.
- **What to do:** wait a few seconds, then **reconnect** by starting the session again. Knowledge Chat, Boardroom, and Document Studio resume normally once the service is healthy.

**Heads up:** Because speech-to-text holds the GPU directly, voice recovery happens via that container restart rather than instantly in place — a brief reconnect is normal, not a sign of data loss.

### Large document uploads

Knowledge Chat is tuned for big, structured files — including very large regulatory books — so sizeable PDFs, DOCX, and PPTX are welcome.

- **Give ingestion time.** Large files are parsed, chunked, embedded, and indexed across several indexes before they're searchable. A long document can take a while; let it finish.
- **Check the document list.** Open **Knowledge Chat** and confirm the file appears in your documents before querying it.
- **Uploads are validated** for type and size, so an unsupported or oversized file is rejected up front rather than failing silently.

**Try this:** Upload a big manual, wait for it to land in the list, then ask a pointed question and watch the citations appear.

### "No answer found" or insufficient-context replies

Knowledge Chat is grounded in *your* corpus and is built to **decline rather than guess**. If it says it can't find an answer, it means the retrieved evidence was too weak — an honest result, not a failure.

- **Rephrase with the document's own words.** Use specific terms, section names, or defined phrases instead of paraphrasing.
- **Confirm the source is uploaded** and finished indexing (see above).
- **Widen the time window.** If you've set a 24h/48h/1w filter, switch it to **All** so older documents are eligible.
- **Ask one thing at a time.** Break a multi-part question into focused queries.

### Exports look stale?

They won't. Every Document Studio and Boardroom export is **rendered fresh on each request** — Document Studio serves files with a no-store policy specifically so a re-export is never a cached copy. If a file looks out of date, you're likely opening a previously downloaded copy. Re-run the export and check your newest download.

### Where your data is stored

Everything stays on your hardware. All content — chats, messages, documents and their chunks, transcripts and analyses, Boardroom sessions, and Document Studio jobs — persists in the local `echomind_data` volume (`/data`) on the host: an SQLite database alongside the FAISS vector indexes and uploaded source files. Model weights live in their own local volumes. **Nothing is sent to any third-party AI service, and there is no telemetry** — no data leaves your machine.

**Admin note (best practice):** EchoMind is designed for a trusted, isolated network. If you plan to reach it beyond your LAN, place it behind your own SSO, VPN, or authenticating reverse proxy, and tighten CORS first.

---

## 14. Glossary

Every product has its own vocabulary, and EchoMind borrows a few words from speech, search, and AI. This glossary explains each one in plain language so nothing in the rest of the manual catches you off guard. Terms are listed alphabetically. Where a term names a feature you can actually click, the chapter that covers it is noted.

### A

**Air-gapped** — A machine with no connection to the internet. EchoMind is built to run this way: every model lives on your hardware, so the whole platform works with the network cable unplugged. Nothing you upload, say, or generate ever leaves your host.

### B

**Backchannel** — The little "mm-hmm" or "I see" the **Voice Conversation** assistant slips in while you're mid-sentence. It signals that it's still listening during a long turn, so the conversation never feels frozen.

**Barge-in** — The ability to talk over the assistant while it's speaking. The moment you start, it stops talking and listens — just like interrupting a person who'll politely let you finish your thought.

**BM25 / keyword search** — A classic search method that matches the exact words in your question against the exact words in your documents. It's the "sparse" half of EchoMind's search, and it shines on precise terms like a regulation number or a product code.

**Boardroom** — The EchoMind module that records a whole meeting, sorts out who said what, and writes an AI meeting report you can export as PDF or PPTX. See the Boardroom chapter.

### C

**Chunk** — A small, self-contained slice of a document (a few sentences to a few pages). EchoMind splits every file into chunks so it can find and quote the exact passage that answers your question, instead of handing back the whole document.

**Citation** — The source pointer attached to an answer in **Knowledge Chat** — the document name, section, and page a fact came from. Citations let you verify any claim against the original.

**Context window** — The amount of text an AI model can "hold in mind" at once: your question, the conversation so far, and the supporting passages it's been given. EchoMind carefully trims retrieved material to fit this window so the most relevant evidence makes the cut.

**Cross-encoder reranking** — A second-pass quality check on search results. After the first search gathers candidate passages, a specialized model re-reads each one alongside your question and reorders them so the truly relevant passages rise to the top.

### D

**Diarization** — Figuring out *who spoke when* in a recording and labeling each part ("Speaker 1," "Speaker 2," and so on). It's what turns a raw **Boardroom** recording into a readable, speaker-by-speaker transcript.

### E

**Embedding** — A way of turning a piece of text into a list of numbers that captures its meaning. Passages about similar ideas end up with similar numbers, which is how EchoMind finds relevant material even when you don't use the document's exact words. EchoMind creates embeddings with the on-device **nomic-embed-text** model.

### F

**FAISS / vector search** — The engine behind meaning-based ("dense") search. FAISS stores all your document **embeddings** and instantly finds the passages whose meaning is closest to your question, even if the wording is completely different.

### H

**Hybrid retrieval** — Running both kinds of search together and blending the results: **FAISS** for meaning plus **BM25** for exact words. You get the best of both — questions phrased loosely still land, and precise terms still match exactly.

### K

**Knowledge base (KB)** — Your private library of everything EchoMind can search: uploaded PDFs, DOCX, and PPTX files, plus saved live transcripts. **Knowledge Chat**, **Voice Conversation**, the **Silent Assistant**, and **Boardroom** all draw their facts from it.

### L

**LLM (Large Language Model)** — The AI that understands your questions and writes answers in natural language. EchoMind's LLM is **Llama-3.1-8B-Instruct**, running entirely on your GPU — no outside AI service is ever called.

### P

**Persona** — A built-in personality and tone for the assistant, which you pick from a menu. Choices include Teacher/Professor, Financial Advisor, Lawyer, AI Expert & Manager, General Assistant, Funny & Calming, and EchoMind Guide. The persona changes *how* answers are phrased, not which facts are used.

### R

**RAG (Retrieval-Augmented Generation)** — The core technique behind EchoMind's grounded answers. Before the **LLM** writes a word, EchoMind *retrieves* the most relevant passages from your **knowledge base** and feeds them in, so replies are based on your documents rather than the model's general training. This is why answers come with **citations**.

### S

**Silent Assistant** — The quiet fact-checker that runs during **Live Transcription**. As sentences form, it compares each statement to your **knowledge base** and posts a small card labeled **Supported, Contradicted, Unverified, Violating,** or **Risky Statement**, with a short explanation and sources. See the Live Transcription chapter.

**STT / speech-to-text** — Turning spoken audio into written text, also called transcription. EchoMind uses NVIDIA's **Nemotron** streaming model, which transcribes as you talk rather than waiting for you to finish.

### T

**Template** — A reusable layout and structure for a generated document in **Document Studio**. EchoMind ships 18 built-in templates (pitch deck, whitepaper, SOP, and more), and you can upload your own to match your house style. See the Document Studio chapter.

**TTS / text-to-speech** — The reverse of STT: turning written text into spoken audio. It's how the **Voice Conversation** assistant talks back, using the on-device **Piper** voice engine.

### W

**Wake word** — A spoken trigger phrase that gets the **Voice Conversation** assistant's attention, so it knows you're addressing it rather than just talking in the room. See the Voice Conversation chapter.

> **A note for whoever installs EchoMind:** the platform is private by design — all processing stays on your hardware, with no telemetry and no third-party AI APIs. As a best practice, deploy it on a trusted, isolated network, or place it behind your own SSO, VPN, or reverse proxy before opening it to a wider audience.

---

## 15. Appendix — For Administrators

If you are the person standing up EchoMind on your own hardware, this appendix is your one-page map. It covers what runs where, how the offline model works, where your data lives, what to expect from the GPU, your two HTTPS paths, and where to go in the repo when you need the full story. Everything below assumes you have shell access to the host and a working Docker setup.

### What's running: the services

EchoMind is a small set of Docker containers orchestrated by **Docker Compose**. The browser only ever talks to one of them; the rest sit behind it.

| Service | Port | GPU | What it does |
|---|---|---|---|
| **frontend** (nginx) | 3000 http / 3443 https | – | Serves the React app; reverse-proxies `/api` → backend and `/voice` → voice |
| **backend** (FastAPI) | 8000 (internal) | 1 | Document ingestion, Knowledge Chat/RAG, Live Transcription, Boardroom, Document Studio |
| **voice** (FastAPI) | 8002 (host) | 1 | The full-duplex speech-to-speech loop (STT → LLM → TTS) |
| **trtllm** (TensorRT-LLM) | 8355 (internal) | all | The chat LLM — **Llama-3.1-8B-Instruct-FP4**, OpenAI-compatible API |
| **ollama** | 11434 | all | **Embeddings only** — `nomic-embed-text`. Chat does *not* use Ollama. |

**Why it matters:** the browser reaches everything through nginx on **3000/3443**, so that's the only port your users need. The voice service is also exposed directly on **8002** (change it with `VOICE_HOST_PORT` in `.env`). The other models — STT (Nemotron), diarization (VibeVoice), TTS (Piper), and Document Studio images (SDXL-Turbo) — run *inside* the backend and voice containers, not as separate services.

### The offline / air-gap model: prepare once, then run dark

EchoMind is built to run with **no internet at runtime** — no outbound API calls, no telemetry, no model downloads. The trick is a one-time preparation step that pulls every model while you still have a connection, caches the weights into Docker volumes, then locks the door.

1. **Prepare once, with internet.** Run `./scripts/prepare_offline.sh`. This builds the images and populates the model caches (LLM, embeddings, speech models).
2. **Run forever, offline.** `docker compose up -d`. From here on the stack is air-gapped.

Offline mode is enforced by runtime flags the compose file already sets — `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `TRTLLM_SKIP_DOWNLOAD=1`, `OLLAMA_OFFLINE=1`, `VOICE_OFFLINE=1` — which block any network model fetch.

**Heads up:** the first preparation downloads gated models (Nemotron), so put a valid `HF_TOKEN` in your `.env` before running prepare. For a truly air-gapped target machine, you can export the model volumes (`trtllm_hf_cache`, `ollama_data`, `echomind_data`) as tarballs on a connected box and import them on the isolated one — see `OFFLINE_DEPLOYMENT.md`.

### Where your data lives

All user data persists in a single Docker volume, **`echomind_data`**, mounted at `/data`. Back this one volume up and you've backed up everything your users created:

- `echomind.sqlite` — chats, messages, documents, chunks, transcripts and their analyses, Boardroom sessions, Document Studio jobs and templates.
- `faiss*.index` + `*_meta.json` — the dense vector, BM25 sparse, section, and glossary indexes.
- `boardroom/<id>/` — uploaded meeting audio chunks; `uploads/` — uploaded source files; `docgen_models/` — the on-device image model cache.

Model **weights** live in separate volumes (`trtllm_hf_cache` for the LLM, `ollama_data` for embeddings). Those are reproducible from the prepare step, so your real backup priority is `echomind_data`.

### GPU expectations

EchoMind expects a Linux host with **NVIDIA GPU(s)**, current drivers, and the **NVIDIA Container Toolkit**. The models share GPU memory, so plan capacity before you turn features on.

- The **LLM** (Llama-3.1-8B in 4-bit FP4) and **STT** (Nemotron) are always resident on the GPU.
- **Document Studio images** load **SDXL-Turbo** as a singleton holding roughly **18 GB** of GPU memory while active — budget for it on top of the LLM and STT.

**GPU fault recovery:** a fatal CUDA fault can poison the shared GPU context. A watchdog detects this, the service's `/health` returns 503, and the process exits so Docker's `restart: unless-stopped` recreates it with a clean context. **Heads up:** because the STT model holds the GPU, the voice service recovers from a fatal CUDA fault *only* by restarting the container — there is no in-process recovery, and there's no separate healthcheck wired up, so watch for a flapping voice container if you see repeated audio failures.

### HTTPS: self-signed vs. trusted

Out of the box the frontend image serves HTTPS on **3443** with a **self-signed certificate** — fine for a quick start, but browsers show a warning (click **Advanced → Proceed**). For a clean, warning-free experience, pick one of two trusted paths:

- **Production (server with a hostname):** get a free subdomain (e.g. DuckDNS) and a Let's Encrypt cert via `sudo certbot --nginx -d <your-domain>`. Full steps in `docs/HTTPS_TRUSTED_CERTIFICATE.md`.
- **Local development:** use **mkcert** to issue a locally-trusted cert for `localhost`. Steps in `docs/HTTPS_LOCAL_TRUSTED.md`.

nginx is already tuned for EchoMind's traffic — streaming responses (`proxy_buffering off`) and long-lived WebSockets (`proxy_read_timeout 86400`) — so token-by-token chat and live audio survive the proxy.

#### A word on access (best practice)

EchoMind is **private by design**: all inference is local, nothing leaves your hardware, and retrieved content is fenced as untrusted data to resist prompt-injection from your corpus. This build is intended for a **trusted, isolated network**. If you ever need to reach it from outside that network, put it behind your own **SSO, VPN, or an authenticating reverse proxy**, and lock down CORS first — the same way you'd front any internal tool.

### Where to read more

This appendix is deliberately light. When you need depth, the repo has it:

- **`docs/CAPABILITIES.md`** — the full capabilities-and-architecture reference; the source of truth for everything in this manual.
- **`README.md`** — quick start, run commands, HTTPS, and FAISS-GPU notes.
- **`OFFLINE_DEPLOYMENT.md`** — exporting/importing volumes for air-gapped machines and troubleshooting.
- The **`docs/`** folder — deeper flow docs for RAG and chunking, the chat pipeline, the voice assistant and wake word, and transcript storage.

**Pro tip:** keep your `.env` (with `HF_TOKEN` and any port or RAG overrides) under your own secrets management — it's the one file that ties a fresh host back to your exact configuration.

---

## 16. Appendix — Additional Concepts

Live Transcription does more than capture words — it can show you the *shape* of a conversation at a glance. The **Word Cloud** turns every transcript you've ever saved, plus whatever is being spoken right now, into a single full-screen picture where the words people actually use loom largest.

### See What a Conversation Is Really About — the Word Cloud

Sometimes you don't want to read a transcript line by line. You want the gist: which themes dominate, which terms keep coming up, what a meeting or a stack of meetings is *really* circling around. The Word Cloud answers that in one look. It reads across **all of your saved transcripts together with the live transcript on screen**, counts how often each meaningful word appears, and draws the most frequent ones biggest.

**Why it matters:** A wall of text hides patterns. A word cloud surfaces them. Drop into a long recording and you'll spot the recurring topic, the product name that won't stop coming up, or the term a team leans on — without scrolling through a single paragraph.

#### How to open it

1. Go to **Live Transcription**.
2. In the header toolbar (top-right, next to the session-history clock and the clear/trash buttons), click the **Word Cloud** button.
3. The view opens **full-screen** with a dark canvas. The header reads **"Word Cloud"** with the subtitle **"All previous transcripts + live transcript,"** so you always know exactly what's being visualized.
4. While the cloud is loading your saved transcripts, you'll briefly see **"Loading transcripts…"**. It then renders the words.

**Heads up:** You can open the Word Cloud at any time — you don't need to be actively recording. If you've saved transcripts before, it draws from those right away. If you have nothing yet, you'll see a friendly prompt: **"No words yet. Add transcripts or start a live transcript."**

#### Reading the cloud

- **Bigger word = more frequent.** Sizes scale with how often each word appears across your transcripts, so the visual hierarchy maps directly to what's talked about most.
- **Color is for legibility, not meaning.** Words are tinted across a bright palette purely so they stay distinct against the dark background — color doesn't encode importance.
- **It's focused, not cluttered.** The cloud shows the **top 50** content words. Common filler words — *the, and, to, for, is, was,* and the like — are filtered out automatically, so only words that carry meaning make the cut. Single letters are dropped too.
- **EchoMind is always front and center.** The brand word is pinned as the largest item in the cloud, a consistent visual anchor at the heart of the view.

#### Watch it grow live

When you open the Word Cloud **while a transcription session is running**, it becomes a live picture of the conversation as it happens.

- A small **"Live — updates every 1 min"** badge appears in the header so you know the cloud is tracking the session.
- The cloud **refreshes automatically about once a minute**, folding in everything new that's been said — both finalized text and the in-progress partial line — alongside your saved history.
- Resize the browser window and the layout re-flows to fit; the words re-pack themselves into the new space.

**Try this:** Start a live session, open the Word Cloud, and leave it up on a second monitor during a meeting. As the discussion moves, the dominant terms shift and grow in near-real-time — an ambient read on where the conversation is heading.

#### Closing the view

The Word Cloud is a full-screen overlay. To return to Live Transcription, either:

- Click the **X (close)** button in the top-right of the header, or
- Click anywhere on the **dark background** outside the content.

Your transcription session keeps running underneath — closing the cloud never interrupts recording or loses anything.

#### Good to know

- **It reflects your knowledge base.** Because saved transcripts are stored locally and auto-added to your knowledge base, the Word Cloud naturally widens as you capture more sessions over time — it's a rolling portrait of everything your team has transcribed.
- **Live text counts even before it's saved.** The on-screen live transcript is blended into the cloud immediately, so you don't have to wait for an auto-save to see new words appear.
- **Everything stays on your hardware.** Like the rest of EchoMind, the Word Cloud is computed entirely in your browser from transcripts that never leave your environment — no external service builds or stores the visualization, and no data is sent off-device.

**Pro tip:** Use the Word Cloud as a fast triage step before searching. Spot the term that dominates a set of meetings, then jump to **Knowledge Chat** and ask about it directly — the same saved transcripts that feed the cloud are fully searchable there.

---

Relevant source files reviewed for accuracy:
- `/home/echomind/Documents/echomind/echomind-enterprise/frontend/components/WordCloudModal.tsx` — full-screen modal, header text ("Word Cloud" / "All previous transcripts + live transcript"), live badge ("Live — updates every 1 min"), loading/empty states, close-on-backdrop/X, combines saved DB transcripts + live text, "EchoMind" pinned largest, top 50 words, ~1-minute live refresh, resize handling.
- `/home/echomind/Documents/echomind/echomind-enterprise/frontend/utils/wordCloudUtils.ts` — tokenization, stopword filtering, min word length, top-50 frequency cap, "bigger = more frequent" basis.
- `/home/echomind/Documents/echomind/echomind-enterprise/frontend/components/LiveTranscription.tsx` (lines 283-291, 449-455) — the header toolbar Word Cloud button (aria-label/title "Word cloud") placed next to session-history and clear buttons, and how live text (`fullTranscript` + `partial`) and `listening` are passed in.
- `/home/echomind/Documents/echomind/echomind-enterprise/docs/CAPABILITIES.md` — Module 2 behavior (transcripts auto-saved to the knowledge base, fully on-prem/offline, no external calls).

---

