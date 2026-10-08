# EchoMind Law — Design System & UX Specification ("Paper & Glass")

> Status: **PROPOSED (D-019)**. Approve the direction, ideally after looking at `design/prototype.html`, before slice S1.3 builds it.
> Sources: Apple's Liquid Glass guidance (WWDC25 *Meet Liquid Glass*; HIG *Materials*); spec §8 and §17; and the product owner's brief: *Apple Liquid Glass, professional, unique, usable, not overwhelming, should not feel like the usual AI.*

---

## 1. The idea in one paragraph

A law firm lives in documents, so the interface uses two materials.

- **Paper** is the content layer: documents, evidence, transcripts, drafts and tables. It is solid, calm and set in a reading serif.
- **Glass** is the navigation and control layer: sidebar, toolbars, menus, recording status. It floats over the paper, lets the work show through and never competes with it.

EchoMind's intelligence does not appear as a chatbot. It appears the way a capable associate's work appears: **findings with pinpoint sources, drafts marked "Unreviewed", and flags that explain themselves.** The user should feel in control of a precise instrument, not in conversation with a personality.

## 2. Liquid Glass, translated faithfully to the web

### 2.1 Rules we adopt from Apple's guidance

| Apple guidance | Our rule |
|---|---|
| "Liquid Glass is best reserved for the navigation layer that floats above the content." | Glass is used **only** for sidebar, toolbars, matter switcher, menus/popovers, command palette, toasts, the recording capsule and sheet chrome. **Never** for documents, tables, lists, evidence cards or form bodies. |
| "Avoid applying the material to both layers [glass-on-glass]. Use fills, transparency and vibrancy for the top elements." | Controls placed on glass use neutral fills (`--fill-1/2/3`) and vibrant text, never a second blur. |
| Regular vs clear variants. Clear only over media-rich content, with a dimming layer. | **Regular everywhere.** Clear is allowed only in the document image viewer (page scans) and always with a dimming layer. The two are never mixed in one context. |
| "Avoid tinting all your elements … do it in the content layer instead." | **One tinted control per view at most**: the primary action (e.g. *Start recording* once consent is complete, *Approve*, *Export*). Everything else is neutral glass. Colour lives in content (status pills, highlights). |
| Small elements adapt light/dark to what's behind them; large ones don't flip. | Capsules (toolbar, recording indicator) increase their shadow and opacity when content scrolls beneath, using `IntersectionObserver` and a `data-over="text"` attribute. The sidebar does not flip. |
| Accessibility: Reduce Transparency → frostier; Increase Contrast → black/white with borders; Reduce Motion → no elastic effects. | `prefers-reduced-transparency: reduce` → opaque fill (no blur). `prefers-contrast: more` and `forced-colors` → solid surfaces, 1px contrasting borders. `prefers-reduced-motion` → cross-fades only. An in-app setting mirrors each one, because browser support varies. |
| Concentric shapes: inner radius = outer radius − padding. Capsules are for standout actions on desktop; dense controls keep rounded rectangles. | Radius scale is concentric (§4.4). Capsules are reserved for the matter switcher, primary toolbar actions, the recording indicator and status pills. Dense inspector and table controls use 8px rectangles. |
| Motion: elements "lift into glass" on interaction and morph between states. | Pressed controls lift (scale 1.02, specular brightens). Popovers grow from their invoking control. The toolbar morphs between contexts (e.g. Document → Compare) with a shared-element transition. No bounce on data. |

### 2.2 Glass recipe (CSS)

```css
/* frontend/law/theme/glass.css — scoped under [data-edition="law"] */
.glass {
  --_fill: var(--glass-fill);
  background: var(--_fill);
  -webkit-backdrop-filter: blur(var(--glass-blur)) saturate(var(--glass-sat));
          backdrop-filter: blur(var(--glass-blur)) saturate(var(--glass-sat));
  border: 1px solid var(--glass-outline);
  box-shadow:
    inset 0 1px 0 var(--glass-specular),      /* top light edge  */
    inset 0 -1px 0 var(--glass-shade),        /* bottom shade    */
    var(--glass-shadow);                       /* lift off paper  */
  position: relative; isolation: isolate;
}
.glass::before {                               /* specular sheen, static; brightens on :active */
  content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
  background: linear-gradient(180deg, var(--glass-sheen) 0%, transparent 42%);
  opacity: .7; transition: opacity var(--dur-2) var(--ease-out);
}
.glass[data-over="text"] { --_fill: var(--glass-fill-strong); box-shadow: inset 0 1px 0 var(--glass-specular), var(--glass-shadow-strong); }

@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
  .glass { background: var(--glass-fallback); }
}
@media (prefers-reduced-transparency: reduce) { .glass { backdrop-filter: none; -webkit-backdrop-filter: none; background: var(--glass-fallback); } }
:root[data-transparency="reduced"] .glass { backdrop-filter: none; -webkit-backdrop-filter: none; background: var(--glass-fallback); }
@media (prefers-contrast: more) { .glass { backdrop-filter: none; background: var(--paper); border-color: var(--ink); box-shadow: none; } }
```

**Optional lensing (progressive enhancement):** small capsules only (recording indicator, primary toolbar capsule) may add an SVG `feDisplacementMap` refraction edge in Chromium. It is never used on large surfaces, is disabled under reduced motion or transparency, and the layout must look correct without it.

**Performance budget:**
- At most **3 large** glass surfaces visible (sidebar, toolbar, one popover or sheet), plus capsules.
- Never glass on a scrolling list item.
- Blur stays between 20 and 28px.
- No animated blur radius.
- Profile on the DGX Spark's browser and a typical firm laptop. Target 60 fps scroll on a 300-page document.

## 3. "Not the usual AI": principles and anti-patterns

| Do | Don't |
|---|---|
| Name actions by legal work: **Find in matter, Compare, Check citations, Draft, Review, Flag** | "Ask AI", "Chat with your docs", "Magic", "✨ Generate", "Copilot" |
| Present answers as **Findings**: a short memo on paper with numbered pinpoints `[1]` that open the exact quote | Chat bubbles, left/right message alignment, avatars, typing dots |
| Show provenance on every AI output: *Prepared from 4 sources · Unreviewed · Qwen3-30B · 14:02* | Unlabelled AI text mixed with human text |
| Show progress as factual steps: *Searching 38 documents in M-101 → Reading 7 passages → Checking quotes* | Spinners with "Thinking…", streaming caret, personality filler |
| Evidence states: **Supported · Partly supported · Not found in this matter · Conflicting sources** | "I'm confident that…", raw % scores up front, "True/False" verdicts |
| Monochrome UI with one accent; colour only for meaning | Purple-blue gradients, neon glows, animated blobs, orbs |
| Line icons (Lucide 1.5px), the firm's typography | Emoji, robot or sparkle icons, mascots |
| Honest, specific empty and error states | Cheerful copy, exclamation marks, apologies |
| "Not found in this matter's documents" as a first-class answer | Answering from general knowledge without saying so |

The existing marketing landing page (`frontend/verticals/LawExperience.tsx`) uses emoji, AI-style glows and "Ask the associate". It is replaced by the workspace for signed-in users (see §7).

## 4. Tokens

All tokens are CSS custom properties under `[data-edition="law"]`, so other verticals are unaffected. They are exposed to Tailwind through a `law` preset (`colors: { ink: 'rgb(var(--ink) / <alpha-value>)', … }`).

### 4.1 Colour: light (default)

| Token | Value | Use |
|---|---|---|
| `--canvas` | `#F2F0EB` → `#ECEEF2` (soft daylight gradient, top-left to bottom-right) | App background behind paper and glass |
| `--paper` | `#FFFFFF` | Documents, cards, tables |
| `--paper-2` | `#FAF9F6` | Alternating rows, inset areas |
| `--ink` | `#15171C` | Primary text |
| `--ink-2` | `#4A4F59` | Secondary text |
| `--ink-3` | `#7C818B` | Tertiary text, placeholders (meets 4.5:1 on `--paper` at ≥14px medium) |
| `--rule` | `rgba(21,23,28,.10)` | Hairlines |
| `--accent` | `#24467A` ("Counsel blue") | The single primary action, focus ring, links |
| `--accent-ink` | `#FFFFFF` | Text on accent |
| `--verified` | `#2E7A55` | Supported / matches source / approved |
| `--attention` | `#9A6412` | Needs review / partly supported / OCR low confidence |
| `--conflict` | `#A8382B` | Differs from source / conflicting versions / denied |
| `--privileged` | `#6A3D7C` | Privilege and work-product labels |
| `--hold` | `#1F6B74` | Legal hold |
| `--recording` | `#D1342F` | **Only** live capture state |
| `--draft` | `#6B7280` | Draft / unreviewed |
| `--glass-fill` | `rgba(255,255,255,.62)` | |
| `--glass-fill-strong` | `rgba(255,255,255,.78)` | When over text |
| `--glass-fallback` | `rgba(250,249,246,.96)` | No-blur fallback |
| `--glass-outline` | `rgba(21,23,28,.08)` | |
| `--glass-specular` | `rgba(255,255,255,.85)` | |
| `--glass-shade` | `rgba(21,23,28,.05)` | |
| `--glass-sheen` | `rgba(255,255,255,.55)` | |
| `--glass-shadow` | `0 12px 32px rgba(17,24,39,.10), 0 1px 2px rgba(17,24,39,.06)` | |
| `--glass-blur` / `--glass-sat` | `24px` / `170%` | |

### 4.2 Colour: dark

`--canvas #0E1014→#121620` · `--paper #171A20` · `--paper-2 #1C2027` · `--ink #ECEDEF` · `--ink-2 #B4B9C2` · `--ink-3 #868C97` · `--rule rgba(236,237,239,.10)` · `--accent #8DB0E6` (text on accent `#0E1014`) · `--verified #5CC392` · `--attention #E0A447` · `--conflict #EE7A6B` · `--privileged #C19BD6` · `--hold #63BCC6` · `--recording #FF5A52` · `--glass-fill rgba(30,34,42,.58)` · `--glass-fill-strong rgba(30,34,42,.76)` · `--glass-specular rgba(255,255,255,.14)` · `--glass-shadow 0 12px 32px rgba(0,0,0,.45)`.

Theme follows the system by default, with a user override (Light / Dark / System) stored per user on the server, not in `localStorage`.

**Rule:** status colour is always paired with an icon and a word (e.g. ✓ *Supported*), never colour alone.

### 4.3 Typography (self-hosted, offline)

| Role | Font | Size / line-height / weight |
|---|---|---|
| UI | Inter Variable (OFL) with `font-feature-settings: "cv11","ss01","tnum"` for numbers | Caption 12/16 · Meta 13/18 · Body 14/20 · Emphasis 14/20 600 · Section 16/22 600 · Title 20/26 600 · Page title 24/30 650 |
| Documents, findings, drafts, quotes | Source Serif 4 Variable (OFL) | Body 16.5/26 · Quote 16/26 italic-free with left rule · Doc title 28/34 600 |
| Pinpoints, clause refs, IDs, timecodes | Inter with `tnum` and `ss02`, or JetBrains Mono (OFL) at 12.5 | `MSA §8.2 · p.7 · v3 (signed)` |

Serif signals "this is a document, read carefully"; sans signals "this is the instrument". AI findings are set in serif because they are drafts of work product.

### 4.4 Shape, space, elevation

- **Spacing:** 4-pt grid (4, 8, 12, 16, 20, 24, 32, 40, 56).
- **Concentric radii:** window-inset panels 20 → inner groups 12 (20 − 8 padding) → controls 8 → chips capsule (`9999px`). The floating sidebar and toolbar are inset 12px from the window edge, with radius 20.
- **Elevation:**
  - E0: paper on canvas, `0 1px 2px rgba(0,0,0,.05)` plus a 1px rule.
  - E1: glass (above).
  - E2: popovers and sheets, with a stronger glass shadow.
  - Modal backdrop: `rgba(14,16,20,.28)`, no blur, which avoids glass-on-glass.

### 4.5 Motion

| Token | Value | Use |
|---|---|---|
| `--dur-1` / `--dur-2` / `--dur-3` | 120 / 200 / 320 ms | Hover / press / panel |
| `--ease-out` | `cubic-bezier(.2,.8,.2,1)` | Default |
| `--ease-glass` | `linear(0, .5 18%, .86 38%, 1.01 62%, 1)` | Glass morph, a slight settle with no bounce |

Rules:
- Panels slide 8–12px and fade.
- Popovers scale from 0.96 at their origin.
- Data never animates its values (no counting up).
- Reduced motion → opacity only, ≤120 ms.

## 5. Components (`frontend/law/ui/`)

| Component | Material | Notes |
|---|---|---|
| `AppSidebar` | Glass, floating | Matter switcher on top; nav groups; permission-filtered items (items you can't use are **absent**, not disabled); user menu at the bottom. Collapses to a 64px glass icon rail below 1024px. |
| `MatterSwitcher` | Glass capsule + glass popover | Shows `Client · M-101 · Title`, plus 🔒 for restricted matters. The popover has search and recent matters. Recents are drawn only from authorized matters and are server-filtered. **Confirmation** when switching with an active recording, unsaved draft or pending upload. Switching purges matter-keyed caches. |
| `Toolbar` | Glass capsule cluster | Page title on the left (sans 15/600); actions on the right. At most one tinted primary action. |
| `CommandPalette` (⌘K) | Glass | "Go to…", "Open document…", "Find in matter…". No free-form "ask AI" box here. |
| `StatusPill` | Content (solid tint 10% + text) | `Supported`, `Partly supported`, `Not found`, `Conflicting`, `Draft`, `Unreviewed`, `Approved`, `Consent needed`, `Recording`, `Paused`, `Privileged`, `Work product`, `On hold`, `OCR text`, `Connector stale` |
| `Pinpoint` | Content chip | `[1]` superscript in findings → hover preview (quote + `doc · §8.2 · p.7 · v3 signed`) → click opens the Inspector at the highlighted quote |
| `FindingSheet` | Paper | Question (sans 16/600) · factual progress steps · findings (serif) · "Sources (4)" · provenance line · actions (*Copy with citations*, *Add to draft*, *Mark reviewed*) |
| `Inspector` | Paper panel, right | Tabs: **Sources · Document · Versions · Activity**. Quote highlighted in context; page image toggle for OCR; "Open full document". |
| `DocumentViewer` | Paper | PDF.js pages with text layer; clause outline; version selector (signed / draft / superseded); highlight API (char offsets). |
| `DiffView` / `RedlineView` | Paper | Side by side and inline; insertions underlined in `--verified` at 18%; deletions struck in `--conflict` at 18%; comment gutter. |
| `ConsentPanel` | Paper card inside a glass sheet | Participant roster, per-person state, method, policy version; *Start recording* stays disabled until consent is complete, with the reason shown. |
| `RecordingCapsule` | Glass capsule, top-centre, persistent | `● Recording · 12:41 · M-101 · 3/3 consented · Pause · Stop`. Red dot only. Survives navigation within the matter. A live region announces state changes. |
| `AlertCard` (Silent Assist) | Paper | Evidence-relative title (D-015), quote, source pinpoint, confidence, *Confirm* / *Dismiss (reason)* |
| `ReviewBanner` | Paper strip | "Draft prepared by EchoMind · Unreviewed · Approve / Request changes" |
| `ApprovalDialog` | Glass sheet + paper body | Shows exactly what is approved (hash, version) and the approver identity |
| `ExportPreview` | Paper | Page thumbnails, redaction boxes with reasons, findings list, watermark preview, metadata-stripping checklist; *Approve export* (tinted) |
| `AuditTable` | Paper | Virtualized; filters (actor, action, matter, outcome, date); chain-verification badge |
| `EmptyState` / `ErrorState` / `Skeleton` | Paper | Specific, honest copy (§6) |
| `Toast` | Glass | Short, factual; no celebration |

## 6. Voice and copy

- **Tone:** precise, neutral, brief. A senior associate's register. No exclamation marks, emoji, apologies, or "As an AI".
- **Labels:**
  - Research: "Research" in the nav (spec's *Knowledge Chat*). Page action: *Find in matter*. Results: *Findings*.
  - Meetings: *Prepare · Record · Review minutes*.
  - Drafting: *Draft from template · Draft from precedent*. Every draft carries the *Unreviewed* pill until approved.
- **Example copy:**
  - Abstention: *"Not found in M-101's documents. Searched 38 documents (signed and draft). Try including draft versions or the firm library."*
  - Non-disclosing error: *"This item isn't available. It may not exist, or you may not have access."* Never "access denied to Matter M-202".
  - Consent block: *"Recording needs consent from 1 more participant: J. Okafor (client). Policy: all-party consent, v1 (awaiting counsel approval)."*
  - Staff AI notice (P-007 draft): *"EchoMind prepares drafts and findings from your matter's documents. It is not a lawyer. Verify sources and authorities before relying on them. Local hosting does not by itself guarantee privilege."*

## 7. Information architecture and screens

### 7.1 Navigation

- **Outside a matter:** *Home · Matters · Reviews (n) · Admin* (permission-filtered).
- **Inside a matter:** *Overview · Documents · Research · Meetings · Contract review · Drafting · Obligations · Diligence* (Phase 3) · *Activity*.
- **Admin:** *Users & roles · Ethical walls · Retention & holds · Audit · Connectors · System*.
- **URL scheme:** `/home`, `/matters`, `/m/:matterNo/{overview|documents|documents/:docId|research|research/:threadId|meetings|meetings/:id|review|drafting|obligations|diligence|activity}`, `/reviews`, `/admin/*`.
- An unknown or forbidden `:matterNo` renders the non-disclosing state.

### 7.2 Layout grid

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ canvas (soft daylight gradient)                                                      │
│ ┌──────────────┐  ┌──────────────────────── Toolbar (glass capsules) ─────────────┐   │
│ │ GLASS SIDEBAR│  │ Research · M-101            [Signed only ▾] [Find in matter]  │   │
│ │ ┌──────────┐ │  └───────────────────────────────────────────────────────────────┘   │
│ │ │Voss-L. ▾ │ │  ┌──────────── PAPER (content) ────────────┐ ┌── INSPECTOR (paper) ─┐│
│ │ │M-101 MSA │ │  │                                         │ │ Sources  Document  … ││
│ │ └──────────┘ │  │                                         │ │                      ││
│ │ Overview     │  │                                         │ │                      ││
│ │ Documents    │  │                                         │ │                      ││
│ │ Research   ● │  │                                         │ │                      ││
│ │ Meetings     │  │                                         │ │                      ││
│ │ …            │  │                                         │ │                      ││
│ │ ──────────── │  │                                         │ │                      ││
│ │ A. Rahman  ⋯ │  └─────────────────────────────────────────┘ └──────────────────────┘│
│ └──────────────┘                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Breakpoints:**
- ≥1280: three panes.
- 1024–1279: the Inspector becomes a right sheet.
- 768–1023: the sidebar becomes an icon rail.
- <768: single column, read-mostly. Recording start is still allowed, with the consent gate.

### 7.3 Key screens (wireframes)

**A. Home (signed in, outside a matter)**
```
Toolbar: Home                                                   [⌘K Go to…]
┌ Your matters (authorized only) ──────────────┐ ┌ Needs your review ───────────────┐
│ Voss-Lindqvist AB   M-101  MSA renewal   ●3  │ │ Draft minutes · Board 07 Oct  ⟶  │
│ Alder Finch Ltd     M-114  Series B DD       │ │ Redline · Supply Agreement v4 ⟶  │
│ …                                            │ │ Export request · M-101         ⟶  │
└──────────────────────────────────────────────┘ └──────────────────────────────────┘
┌ Upcoming obligations (authorized matters) ─────────────────────────────────────────┐
│ 14 Nov  Non-renewal notice due · MSA §3.2 · M-101 · Unconfirmed                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**B. Matter overview**
```
Toolbar: Voss-Lindqvist AB · M-101 · MSA renewal   🔒 Restricted   [Members] [⋯]
Paper: Summary (practice area, jurisdiction, status, lead) · Members (role, since) · Holds/Retention
       Recent documents · Recent meetings · Drafts awaiting review · Activity (matter-scoped)
```

**C. Documents library**
```
Toolbar: Documents · M-101          [Import ▾ (Upload · From folder · From DMS*)]  [Search in documents]
Table (paper): Title | Kind | Version (v3 signed · v4 draft) | Privilege | Source (Local/DMS) | Updated | ⋯
Row states: Indexing… / OCR text / Connector stale / On hold
Selecting a row opens the Inspector: versions timeline, labels (editable, audited), activity.
* DMS item only when configured; otherwise absent.
```

**D. Research (spec: Knowledge Chat)**
```
Toolbar: Research · M-101   Scope: [This matter ▾ + Firm library]  [Signed only ●]   [New question]
Left list (paper, narrow): previous questions in this matter (titles only)
Paper sheet:
  Q  What is the change-of-control notice period in the signed agreement?
  ✓ Searching 38 documents → Reading 6 passages → Checking quotes
  Findings  (serif)
    The signed MSA (v3) requires 30 days' written notice before a change of control [1].
    The v4 draft proposes 60 days [2]. ⚠ Conflicting versions — v3 is the signed one.
  Sources (2)   [1] MSA v3 (signed) · §14.2 · p.11 · "…no less than thirty (30) days…"
                [2] MSA v4 (draft) · §14.2 · p.11 · "…sixty (60) days…"
  Prepared from 2 sources · Unreviewed · Qwen3-30B · 14:02     [Copy with citations] [Add to draft] [Mark reviewed]
Inspector: the quote highlighted in the page; version switcher.
```

**E. Meeting: prepare → consent → record → review**
```
Toolbar: Meetings · Board meeting 07 Oct · M-101            [Prepare] [Consent] [Record] [Minutes]
Prepare: agenda, board pack (authorized docs), "Questions on the board pack" (Research scoped to pack)
Consent (paper card in glass sheet):
   A. Rahman (counsel)  ✓ verbal, on record 10:02 · J. Okafor (client) ○ pending · M. Berg (director) ✓ written
   Policy: all-party consent · v1 (awaiting counsel approval)          [Start recording] (disabled: 1 pending)
Record: RecordingCapsule (top centre) · live transcript (speaker labels, timecodes, edit with provenance)
        Right: Flags (AlertCards: "Differs from signed MSA §8.2", Confirm/Dismiss) · Actions · Records
Minutes: draft minutes (serif) with timecode pinpoints · resolutions · actions (owner/deadline) · unresolved
         ReviewBanner → Approve (secretary/attorney) → Export (via Export preview)
```

**F. Contract review / redline**
```
Toolbar: Contract review · Supply Agreement v4 vs Playbook "Vendor MSA (2026)"   [Inline | Side by side]  [Export ▾]
Paper left: clause list with deviation markers (Standard · Fallback · Outside playbook)
Paper centre: RedlineView with suggested revision, risk label, playbook source pinpoint
Inspector: playbook clause text, fallback positions, approval owner
Export ▾: "Word (tracked changes)" appears only once T10 passes; until then "HTML redline (PDF)"
```

**G. Drafting:** template explorer (approved firm templates with version) → fields form (each value sourced or `NEEDS INPUT`) → draft (serif) with source pinpoints → review → export.

**H. Citation and quote check report:** a table of each citation or quote, with its status (*Verified in matter* · *Altered quote* · *Not found* · *External authority, unverified*) and evidence.

**I. Obligations:** a table with trigger, due date (timezone), owner, status (*Extracted · Confirmed · Done*) and evidence pinpoint; calendar strip; confirm and edit with audit.

**J. Diligence register:** checklist progress, missing documents, issues with severity, sources and reviewer status; preliminary and verified kept in separate tabs.

**K. Export preview:** see the component above. A redaction is only *applied* after approval, and the result is *verified*.

**L. Admin:**
- Audit: table with chain badge "Verified to 14:05 today".
- Retention & holds: policies (dry-run results), holds (request → approve → active → release).
- Users & roles: MFA status, sessions, revoke.
- Ethical walls: wall, members, rationale, exceptions, approvals.
- Connectors: status, last sync, failures.

**M. Not available (non-disclosing):** the same layout for both "doesn't exist" and "no access".

### 7.4 States catalogue

Every data view implements six states: **loading** (skeleton, no spinner over 300 ms without text) · **empty** (specific next step) · **error** (what failed, what to do, reference ID) · **inaccessible** (non-disclosing) · **stale** (connector or index behind, with timestamp) · **partial** (e.g. OCR pending on 3 pages).

## 8. Accessibility (WCAG 2.2 AA minimum)

- **Contrast:** text on glass is measured against the **worst-case** backdrop (black text over dark content, white over light). `--glass-fill-strong` is raised automatically when the measured contrast would drop below 4.5:1.
- **Focus:** 2px `--accent` ring with a 2px `--paper` offset; never removed.
- **Live regions:** recording state, alert arrivals and export completion are announced politely. Recording start and stop are announced assertively.
- **Keyboard:**
  - `⌘K` palette; `G` then `O/D/R/M/C` to jump to matter sections; `/` search.
  - `[` / `]` previous and next source; `E` evidence inspector; `Esc` closes the topmost layer.
  - All drag interactions have keyboard equivalents.
- **Targets:** at least 24×24 px (WCAG 2.5.8), 32px in dense tables.
- Status is never conveyed by colour alone.
- Respects transparency, contrast and motion preferences, with in-app overrides (§2.1).

## 9. Implementation notes (slice S1.3)

```
frontend/law/
  theme/tokens.css · theme/glass.css · theme/tailwind-preset.cjs
  ui/ (components above, each with a *.test.tsx)
  app/LawApp.tsx (router, providers) · app/routes/* · app/auth/* · app/matter/MatterContext.tsx
  api/client.ts (typed fetch, 401→re-auth, matter-keyed query keys) · api/types.ts (generated from FastAPI OpenAPI in dev)
  dev/DesignCatalogue.tsx  (/__design, dev builds only: every component and state for review)
```

- `App.tsx` renders `<LawApp/>` **before** mounting the general-purpose hooks when `resolvePack()?.id === 'law'` and `VITE_EDITION === 'law'`. This fixes the stray-chat bug, R-12.
- Tailwind `content` must include `./law/**/*` and `./verticals/**/*` (fixes the missing-styles bug, R-12).
- Fonts: `@fontsource-variable/inter` and `@fontsource-variable/source-serif-4`, imported in `law/theme`. No CDN.
- Visual regression: Playwright screenshots of `/__design` in light, dark, reduced transparency and high contrast. These are reviewed, not auto-accepted.
