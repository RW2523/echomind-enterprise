# EchoMind — Marketing & Product Documentation

Audience-facing collateral for the EchoMind platform. Every figure quoted in these
documents is measured, not estimated; the measurement study behind them was accepted
for publication at QASC 2026.

| Document | Audience | Format |
|---|---|---|
| `EchoMind_Pitch_Deck.pptx` / `.pdf` | Prospects, execs — 13 slides | 16:9 deck, editable |
| `EchoMind_Business_Overview.pdf` | Marketing and sales | 4 pp |
| `EchoMind_Technical_Overview.pdf` | Engineering / technical leads | 5 pp |
| `EchoMind_Architecture_Flows.pdf` | Architects, technical evaluation | 7 pp, landscape diagrams |
| `EchoMind_AI_Capabilities.pdf` | HR, internal capability catalog | 9 pp, one page per capability |
| `EchoMind_Architecture_Brief.pdf` / `_v2.pdf` | Management summary | 3 pp / 9 pp |

## Regenerating

All PDFs and the PPTX are generated from source in `src/` — edit the script, not the PDF.

```bash
cd marketing_doc/src
python3 build_deck.py          # pitch deck (PDF)
python3 build_deck_pptx.py     # pitch deck (PowerPoint)
python3 build_business.py      # business & sales overview
python3 build_technical.py     # technical overview
python3 build_architecture.py  # architecture & flow diagrams
```

Requires `reportlab` (PDF) and `python-pptx` (PowerPoint). `emkit.py` holds the shared
design system — colors, typography, tables, callouts and the diagram primitives.

## House rules for edits

- US English throughout.
- No unmeasured performance claims. If a number is not in the evaluation, do not add it.
- The two known open findings (response-timing side channel; injection containment is
  silent) are stated deliberately in the deck and the sales overview. Keep them.
