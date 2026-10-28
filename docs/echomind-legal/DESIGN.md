# EchoMind Legal — design system

Quiet, familiar professional software for lawyers. The product earns trust through evidence and precision, not through
decoration. Implemented in `frontend/law/counsel/counsel.css` (tokens under `[data-shell="counsel"]`), which also re-skins
the Law pages the shell hosts.

## Principles

1. **The work is the interface.** A screen shows the matter, the question box and the next useful action. Nothing on it is
   there to impress.
2. **One dominant action per view.** On the chat home it is the question box; on a matter page, the page's own task.
   Specialist tools stay reachable on the matter's tabs, never all at once on the home.
3. **Evidence is the signature.** Gold is used for one job only: citation chips and the highlighted words of a source.
   Wherever gold appears, a reader can open the exact passage.
4. **Say what is known.** Answer states (supported, partly supported, not verified, no source), review states (draft,
   pending review, approved) and potential inconsistencies are always shown in words, never as scores or percentages.
5. **Familiar controls.** Standard buttons, tabs, inputs, lists and sortable tables. No custom metaphors.

## Navigation

- Rail: New chat, then four destinations — **Chat, Matters, Drafts, Activity** — then recent conversations. Nothing else.
- A matter's tools (Overview, Documents, Research, Meetings, Drafting, Obligations, …) are its tabs, and its overview lists
  them with one line each. The chat home links quietly to the most used ones for the chosen matter.
- Firm and account pages (Templates, Playbooks, Time, administration, Security) live in the profile menu.

## Tokens

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

## Do not

Photographs or illustrations as decoration, glass or blur, gradient fills, eyebrow labels above headings, walls of
same-size feature cards, confidence percentages, decorative motion, emoji, more top-level navigation, external fonts or
assets (the product runs offline).
