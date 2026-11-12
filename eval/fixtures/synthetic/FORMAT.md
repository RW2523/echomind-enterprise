# Source format of the synthetic dataset

Every document of `eval/fixtures/synthetic/` is written once as a text source, `src/<matter>/<NN>-<slug>.src.md`, and
rendered by `scripts/law/synthetic/render.py` into the file a firm would hold (Word, a typeset PDF, a scanned PDF, plain
text or a file the product cannot import). The sources are the truth: the answer keys quote them.

## Front matter

```
---
file: Supply and Installation Agreement - EXECUTED.docx   # the rendered file's name; its extension must suit `render`
title: Supply and Installation Agreement (executed)       # the title it is imported under
kind: contract          # contract | correspondence | pleading | evidence | memo | note | other
status: executed        # draft | final | executed | received
render: docx            # docx | pdf | scan | txt | md | raw | xlsx
style: contract         # contract | letter | email | minutes | report | table | policy | form | statement (layout)
font: serif             # serif | sans | mono
header: VARNHOLT / QUELLCREST - CONFIDENTIAL              # optional running header (docx, pdf, scan)
footer: HL-2023-0141 v7                                   # optional running footer; page numbers are added
letterhead: Harcourt Lyle LLP | 12 Wexford Row | London EC4A 2BD   # optional, first page of pdf/scan/docx letters
numbering: auto         # docx only: numbered clauses become Word list numbering (the numbers are not typed text)
author: J. Pell         # docx only: the author of tracked changes and comments
changes_date: 2023-02-27
scan: skew=0.6 noise=0.3 dpi=200 mode=gray   # scan only: mode gray | bw (an old black-and-white office scan)
version_of: 01-supply-agreement.src.md       # optional: imported as a new version of that earlier document
import: no              # optional, raw/xlsx only: a format the product is expected to refuse (an import test, not keyed)
---
```

## Body

- **Headings:** `# Title`, `## Heading`, `### Sub-heading`.
- **Paragraphs:** separated by a blank line. A single line break inside a paragraph is kept (addresses, signature
  blocks, email headers, verse-like lists).
- **Numbered clauses:** type the number at the start of the paragraph: `12. LIMITATION OF LIABILITY` (a heading),
  `12.1 The Supplier's total liability ...`, `(a) ...`, `(i) ...`. With `numbering: auto` the renderer turns these into
  Word list numbering, so the numbers must run in order as Word would count them (1, 2, 3; 12.1, 12.2; (a), (b); (i), (ii)),
  and a level restarts under each new parent. A line `<<<NUMBERING OFF>>>` stops auto numbering (schedules usually keep
  typed numbers); `<<<NUMBERING ON>>>` resumes it.
- **Tables:** pipe tables (`| Item | Amount |` then `|---|---|`).
- **Page break:** a line `<<<PAGE>>>`.
- **Tracked changes and comments (docx only; other formats show the accepted text):** `{++inserted++}`,
  `{--deleted--}`, `{~~old~>new~~}`, and a comment on some words: `{==the words==}{>>the comment<<}`.
- **Marks a reader sees on paper (scan and pdf):** `[[hand: Agreed - RT 14/3]]` handwriting, `[[stamp: RECEIVED
  14 MAR 2025]]` a rubber stamp, `[[sign: R. Thorne]]` a signature (Word shows `/s/ R. Thorne`).
- **raw / xlsx:** the body is the file's exact content (CSV, an .eml message); `xlsx` turns a pipe table into a sheet.
