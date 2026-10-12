# EchoMind Law transcription evaluation — file

- Commit: `a410326`
- Run: 20261012-051119

| Measure | Result |
|---|---|
| scenarios | 4 |
| wer normalised | 0.1149 |
| wer verbatim | 0.336 |
| entity accuracy | 0.4762 |
| negation accuracy | 0.8334 |
| speaker accuracy | 0.9891 |
| missed entities | 10 |
| lost negations | 2 |

| Scenario | WER (normalised) | WER (verbatim) | Entities | Negations | Speakers | Missed | Lost negations |
|---|---|---|---|---|---|---|---|
| client-call-renewal | 0.1477 | 0.4545 | 0.5714 | 0.6667 | 0.9773 | Harrowgate Systems, 2 March 2026, clause 8.1 | I will not concede the thirty day convenience notice |
| negotiation-change-of-control | 0.1471 | 0.2609 | 0.0 | 1.0 | 1.0 | Pellucid Holdings, 15 October 2026, Clause 14.2, 3 September 2026, Harrowgate Systems | — |
| board-meeting-mandate | 0.1233 | 0.2329 | 0.6667 | 0.6667 | 1.0 | USD 400,000 | the liability cap is not reduced below USD 400,000 |
| witness-interview | 0.0417 | 0.3958 | 0.6667 | 1.0 | 0.9792 | Harrowgate Systems | — |
