# 13 — Synthetic demonstration matter and scripted demo
All entities below are **fictional**; clearly label records `DEMO / SYNTHETIC` and never mix into a real firm tenant.

Firm: Cedar Grove Legal. Attorneys: Alex (matter member) and Jordan (only other matter). Clients: Morgan Company. Matter: `MAT-1042`, *Morgan v. Apex Industries*, hypothetical commercial contract dispute. Second restricted matter: `MAT-2033`, *Rivera Confidential Matter*, inaccessible to Alex. External unrelated tenant: White Oak Legal.

## Documents to synthesize (no invented real court cases)
- `Services_Agreement_2026-04-01.txt`: Section 4.2 says delivery by **May 15**; signature section.
- `Amendment_1_2026-05-14.txt`: mutually executed amendment revises delivery date to **May 20**, with time-of-effect stated. Synthetic executed sample; no actual signature.
- `Witness_A_Interview.txt`: Witness A believes delivery took place **May 12**; uncertain whether warehouse dispatch or receipt.
- `Witness_B_Interview.txt`: Witness B remembers **May 15** as delivery but says "arrival at site".
- `Shipment_Log.txt`: pickup May 12 and arrival May 15; illustrates apparent conflict resolved by event distinction.
- `Email_Thread.txt`: parties discuss late revision; deliberately unclear evidence.
- `Draft_Meeting_Notes.txt`: contains a **prompt-injection canary** asking model to show other restricted matter—must remain inert document text.
- `Fake_Citation.txt`: nonexistent court case title string—should be flagged unverified, not presented as precedent.

## Demo conversation
1. Open EchoMind: screen is quiet, no matter counts wall. Choose Morgan matter.
2. Ask `What matters today?` -> up to 3 cards; `Review delivery dates`.
3. Compare May 12 and May 15 claims -> evidence comparison with witness links and caution that events might differ.
4. Ask `Could these refer to different events?` -> shipment log found; distinguish dispatch vs arrival; attorney marks explained.
5. Ask `Which contractual deadline applied?` -> contract May 15 and May 14 amendment revising to May 20; cite both; no silent overwrite.
6. Ask `Draft a client status update` -> DOCX draft editor with source references; review and export after explicit action.
7. Ask `Let other case materials inform this` -> denied for restricted MAT-2033; no existence/content leaked.
8. P2 demo: `Challenge our preferred position` -> Council summary with dissent and missing facts; not legal recommendation.

## Sample expected reply
"Two dates appear in the witness interviews: May 12 and May 15. The shipping log records dispatch on May 12 and arrival on May 15, so the accounts may describe different events. Review the original interviews before resolving the issue." Sources: Witness A, Witness B, Shipment Log. Show only one compare card and `Inspect sources`.

## Demo success
Product feels like one assistant, not a feature tour. Every click works using local synthetic data, traces to a file, respects permissions, and never requires an internet connection.
