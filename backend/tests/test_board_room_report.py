from app.board_room.report_pipeline import _markdown_report, _split_segments


def test_split_segments_empty():
    assert _split_segments("") == []


def test_split_segments_groups_sentences():
    text = "First point. Second point. Third point."
    segs = _split_segments(text, max_chars=30)
    assert len(segs) >= 2
    assert "First point." in segs[0]


def test_markdown_report_includes_sections():
    md = _markdown_report(
        title="Q1 Review",
        polished="## Minutes\n- Item one",
        summary="Leaders agreed on next steps.",
        checks=[
            {
                "claim": "Revenue grew",
                "classification": "supported",
                "interpretation": "Matches filings",
                "suggested_action": "Cite source",
                "evidence": [{"source_name": "deck.pdf", "matched_text": "Revenue up 12%"}],
            }
        ],
        session_name="Board",
        session_location="HQ",
    )
    assert "# Q1 Review" in md
    assert "Executive summary" in md
    assert "Knowledge validation" in md
    assert "Revenue grew" in md
