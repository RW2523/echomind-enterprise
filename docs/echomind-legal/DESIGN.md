# EchoMind Legal — design system

Professional software for lawyers with the presence of a good chambers: a walnut rail, parchment and paper, brass for where
you are, and the firm's own photographs of the courthouse and the desk behind the work. It earns trust through evidence and
precision; the room sets the tone and never sits behind text. Implemented in `frontend/law/counsel/counsel.css` (tokens
under `[data-shell="counsel"]`, which also re-skins the Law pages the shell hosts) and `frontend/law/counsel/courthouse.css`
(the "Courthouse" layer, owner's request 2026-11-05, which replaced the earlier "quiet, no photographs" direction).

## Principles

1. **The work is the interface.** A screen shows the matter, the question box and the next useful action. The room (the
   photographs, the walnut) frames it; every control on screen does something real.
2. **One dominant action per view.** On the chat home it is the question box; on a matter page, the page's own task.
   Specialist tools stay reachable on the matter's tabs, never all at once on the home.
3. **Evidence is the signature.** In an answer, gold is used for one job only: citation chips and the highlighted words of
   a source; wherever gold appears there, a reader can open the exact passage. Outside answers, the walnut palette also uses
   gold for where you are (the current rail item and matter tab) and the brand mark — never for a button or a status.
4. **Say what is known.** Answer states (supported, partly supported, not verified, no source), review states (draft,
   pending review, approved) and potential inconsistencies are always shown in words, never as scores or percentages.
5. **Familiar controls.** Standard buttons, tabs, inputs, lists and sortable tables. No custom metaphors.

## Navigation

- Rail (walnut, the gavel photograph faded in at its foot): the AJACE mark and "EchoMind Legal", New chat, then **Chat,
  Matters**, the current matter's **Documents, Council** (where the panel is on) and **Transcription** ("Live" while EchoMind
  records here), then **Drafts** (with what waits for you) and **Activity**; recent conversations; at the foot the current
  matter (the matter picker) and the person (the profile menu). Without a matter, the matter's links lead to Matters.
- Top bar, no band: search (the command palette, ⌘K), where answers come from, the recording state, and a bell counting
  what waits for the person's decision (it opens Drafts).
- Chat: a serif greeting by the person's clock on the home; the conversation on one paper "stage"; beside it on wide screens
  (≥ 1361px) the column **Context · Sources · Actions** — the matter, its latest documents, questions found for the Council
  (or the starters), talking by voice; the passages this conversation cites; what waits, deadlines not confirmed,
  inconsistencies open. Every line is a record and opens where it can be acted on.
- A matter's tools are its tabs: six every day (Overview, Documents, Research, Drafting, Obligations, Meetings) and the
  rest under **More**; the page you are on is always a tab. Its overview lists every tool with one line each. The chat
  home links quietly to the most used ones for the chosen matter, with one line for what waits for the person's decision
  and the next dated deadline.
- Firm and account pages (Templates, Playbooks, Time, administration, Security) live in the profile menu.

## Tokens

Two palettes share every token name (`counsel.css`); the profile menu switches between them and the choice is kept per
browser (`echomind.law.palette`).

**Walnut and gold** (default since 2026-11-02, the owner's brown/gold/black request): a black rail, warm parchment and
paper, brown for actions, gold for the current place and for evidence.

| Token | Light | Dark | Use |
|---|---|---|---|
| canvas | `#F5F0E8` | `#110E0B` | page ground |
| paper | `#FFFCF7` | `#1A1612` | cards, panels, header |
| ink / ink-2 / ink-3 | `#231912` / `#4A3B2E` / `#6B5A4A` | `#F1E8DA` / `#D3C6B3` / `#AC9D8C` | text, secondary, muted |
| rule | `#E6DCCC` | `#2F2821` | 1px borders and dividers |
| accent | `#6B4A1E` (brown, white text) | `#CFAA66` (gold, black text) | primary actions, focus, selection |
| tab-line / rail active | `#9C7536` / `#E2C27F` on `#14100C` | `#CFAA66` / `#E2C27F` | the current tab and rail item |
| gold (evidence) | `#B08D57`, text `#7A5A1E` | `#D9B776`, text `#E2C68F` | citations and highlighted source words |
| rail | `#14100C` | `#0B0907` | the navigation rail |

Contrast (WCAG AA): ink on paper 16.8:1, muted on canvas 5.8:1, accent with white 8.0:1, evidence text 6.2:1, rail text
10.5:1; dark muted 6.8:1, dark accent 8.8:1.

**Navy** (the palette before 2026-11-02):

| Token | Light | Dark | Use |
|---|---|---|---|
| canvas | `#F6F5F2` | `#16181D` | page ground |
| paper | `#FFFFFF` | `#1C1F25` | cards, panels, header |
| ink / ink-2 / ink-3 | `#1E242C` / `#3D4552` / `#5B6370` | `#ECEEF1` / `#C9CED6` / `#A3ABB8` | text, secondary, muted |
| rule | `#E4E2DD` | `#2C3038` | 1px borders and dividers |
| accent | `#1F3A5F` | `#8FB0DC` | primary actions, focus, selection |
| gold (evidence) | `#B58B3F`, text `#7A5A1E` | `#D2AE6A`, text `#DDBE80` | citations and highlighted source words only |
| state | green `#23714F`, amber `#8A5A0E`, red `#A3352C` | lighter equivalents | verified, needs review, conflict/recording |
| rail | `#172435` | `#121419` | the navigation rail |

All text pairs meet WCAG AA (values in the header of `counsel.css`). The appearance follows the operating system until a
person chooses light or dark in the profile menu.

## Type and spacing

- UI: Inter (bundled). Answers, quotes and source passages: Source Serif 4 (bundled), 16–16.5px, line-height 1.65,
  measure ≤ 68ch. Page headings 22–24px, weight 600; card titles 15px.
- Controls 34px high; radius 8px (controls) and 10px (cards); 1px borders; shadows only on popovers.
- Spacing steps 4 / 8 / 12 / 16 / 24px; more space above a heading than below it.

## Components

- **Answer card**: state in words (with icon), the answer in serif, source chips (number, matter when several, title,
  clause/page), notes on amendments or versions, "Sentences without a verified source" collapsed.
- **Source panel**: the exact passage with the cited words highlighted in gold, version and file hash, "Open full
  document", newer-version warning.
- **Comparison**: A and B side by side with the conflicting value highlighted, status in words, resolution requires a reason.
- **Chronology**: date column, a rail of dots coloured by kind, filters (kind, matter) and a month chart that filters.
- **Cases table**: sortable columns, each row opens its matter.
- **Notes**: a tinted block with an icon; no thick coloured side borders.

## Words

- Where answers come from follows the configuration (`model_location`): "On this server", "On your network" or, in the
  attention colour, "External model". Never "nothing leaves", "no cloud", "air-gapped" or "guaranteed".
- Review states: "Waiting for approval", "Changes requested", "Approved". Activity is a sentence, never an event code.

## The room

- Light: the courthouse in fog (`/law/court-light.webp`) at the top right, faded into parchment `#F4EEE4` where the work
  is; behind a matter's pages it shows only above the tabs. Dark: the gavel and books (`/law/gavel-dark.webp`), faded into
  `#0F0B08`. The rail's foot: a crop of the gavel (`/law/gavel-rail.webp`). All three are served by this server.
- Surfaces on the room are paper at 88–90% with a light blur (stage, context column, chips); text never sits on the
  photograph itself. Stage radius 20px, cards 14px, controls 10–12px.
- Type: page and greeting headings in Source Serif 4 (40px greeting, 19px brand); UI in Inter.

## Do not

Model-made scores, risk meters or confidence percentages; a "reviewed" or "verified" badge the records do not back; a button
or tab that does nothing; illustrations or stock imagery beyond the firm's two photographs; emoji; decorative motion;
external fonts or assets (the product runs offline).
