# Synthetic legal dataset (2026-11-12)

A realistic but **entirely fictional** set of law-firm files, made to test EchoMind Legal away from its own demo data
(owner, in the 2026-11-11 evaluation report: "create a real time synthetic proper dataset and attach it and test it how
it is working"). Every company, person, address, email domain, figure and event is invented. Real statutes are named
where a real file would name them; there are no case citations. Each rendered page carries the footer "Fictional test
document - EchoMind synthetic set".

The demo's own documents (`eval/fixtures/demo/`) are clean Markdown written alongside the product. These are what a firm
actually holds: Word files with list numbering, tracked changes and comments; typeset PDFs with running headers and
footers; scanned PDFs with no text layer (skewed, speckled, stamped, signed, with handwritten amendments); long email
chains with quoted history; spreadsheets printed to PDF; superseded drafts beside executed versions; amendments that
change earlier clauses; mixed US and UK dates; and files the product cannot import.

## Matters

| Matter | What it is | Law |
|---|---|---|
| SYN-01 | Varnholt v Quellcrest: delay damages under a supply-and-installation contract for robotic cells | England & Wales |
| SYN-02 | Larkspur Health: a billing vendor's ransomware incident (contract, BAA, cyber insurance) | New York |
| SYN-03 | Ashdown & Pryce: a tenant's break under a 2016 retail lease, disputed by the landlord | England & Wales |
| SYN-04 | Okafor-Reyes v Brackenridge Analytics: retaliation claim, separation terms | California |

## Layout

- `src/<matter>/NN-*.src.md`: each document's text source (format: `FORMAT.md`); `src/<matter>/matter.json`: the matter;
  `src/<matter>/bible.md`: the author's fact sheet, with the deliberate traps and the facts no document states (for
  reviewing the keys; never loaded into the product).
- `<matter>/`: the rendered files, as imported. `text/<matter>/`: what a reader sees on each one (the keys quote it).
- `manifest.json`: matters, documents, formats, features and checksums.
- `answer_key_*.jsonl`: the questions, in the format of `eval/fixtures/demo/answer_key_*.jsonl`, written from the
  documents alone by a reader who had not seen the fact sheets or any answer from the product.

## Rebuild and run

```
python3 scripts/law/synthetic/render.py                          # src/ -> rendered files, text/, manifest.json
# what the product's own reader extracts (offline, in the backend image):
docker run --rm --network none --user 1000:1000 -v "$PWD:/repo:ro" -v "$OUT:/out" -w /repo/backend \
    -e ECHOMIND_EDITION=law echomind-law-backend:latest python3 /repo/scripts/law/synthetic/extract_check.py --out /out
# a PRIVATE copy of the demo (never the public demo), then:
python3 scripts/law/synthetic/load.py --base http://127.0.0.1:8161/api --container echomind-legal-syn-api
python3 eval/law/run_chat_keys.py --base http://127.0.0.1:8161/api --keys 'eval/fixtures/synthetic/answer_key_*.jsonl' --label syn
python3 eval/law/run_retrieval_eval.py --container echomind-legal-syn-api --key-files 'eval/fixtures/synthetic/answer_key_*.jsonl' --label syn
```
