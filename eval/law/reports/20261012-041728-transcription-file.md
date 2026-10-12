# EchoMind Law transcription evaluation — file

- Commit: `28b3416`
- Run: 20261012-041728

| Measure | Result |
|---|---|
| scenarios | 4 |
| wer normalised | 0.2442 |
| wer verbatim | 0.5875 |
| entity accuracy | 0.2881 |
| negation accuracy | 0.5417 |
| speaker accuracy | 0.9398 |
| missed entities | 13 |
| lost negations | 5 |

| Scenario | WER (normalised) | WER (verbatim) | Entities | Negations | Speakers | Missed | Lost negations |
|---|---|---|---|---|---|---|---|
| client-call-renewal | 0.3182 | 0.6705 | 0.2857 | 0.0 | 0.9659 | Harrowgate Systems, 2 March 2026, USD 18,500, clause 8.1, USD 400,000 | must not fall below USD 400,000, We will not accept the fee increase, I will not concede the thirty day convenience notice |
| negotiation-change-of-control | 0.1765 | 0.5072 | 0.2 | 1.0 | 1.0 | Pellucid Holdings, 15 October 2026, Clause 14.2, Harrowgate Systems | — |
| board-meeting-mandate | 0.274 | 0.589 | 0.6667 | 0.6667 | 0.8767 | Northbridge Analytics | I will not vote on item five |
| witness-interview | 0.2083 | 0.5833 | 0.0 | 0.5 | 0.9167 | USD 19,750, 1 June 2026, Harrowgate Systems | I never saw a signed amendment |
