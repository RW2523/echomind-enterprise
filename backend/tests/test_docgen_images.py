"""Regression tests for Document Studio inline images.

Before the fix, several templates declared an "image" section kind and the renderers +
``models.normalize_document`` supported "image" blocks, but the section-writer/planner prompts
never offered the "image" block type, so inline image blocks were never generated. These tests
lock in that the writer is offered the image block (only when image generation is wanted) and
that an inline image block flows end-to-end through generation into a rendered PDF/PPTX.
"""
from __future__ import annotations

import asyncio
import io
import os
import tempfile

os.environ.setdefault("ECHOMIND_DATA_DIR", tempfile.mkdtemp(prefix="echomind_test_"))


# --- tiny helpers ----------------------------------------------------------------------------
def _png_bytes() -> bytes:
    """A real, decodable PNG so the renderers embed it instead of drawing a placeholder."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (12, 80, 160)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeLLM:
    """Records every chat() call and replies based on which subagent is calling."""

    def __init__(self, responder):
        self.calls = []          # list of `messages` lists, in call order
        self._responder = responder

    async def chat(self, messages, temperature, max_tokens):
        self.calls.append(messages)
        return self._responder(messages)

    def system_prompts(self):
        return [m[0]["content"] for m in self.calls]


# --- pure spec/hint tests --------------------------------------------------------------------
def test_block_types_spec_offers_image_only_when_enabled():
    from app.docgen import pipeline

    without = pipeline._block_types_spec(False)
    with_imgs = pipeline._block_types_spec(True)

    # Core types + footer are always present.
    for spec in (without, with_imgs):
        assert '"type":"paragraph"' in spec
        assert '"type":"table"' in spec
        assert "Do NOT emit a top-level section heading block" in spec

    assert '"type":"image"' not in without
    assert '"type":"image"' in with_imgs
    # The image block must still carry a generation prompt (what the image stage scans for).
    assert '"prompt"' in with_imgs


def test_kind_hint_has_image_entry():
    from app.docgen import pipeline

    assert "image" in pipeline._KIND_HINT
    assert "image" in pipeline._KIND_HINT["image"].lower()


# --- writer prompt gating ---------------------------------------------------------------------
def test_write_section_offers_image_block_to_writer_only_when_wanted():
    from app.docgen import pipeline, templates

    template = templates.get_template("pitch_deck")
    section = {"title": "The Problem", "level": 1, "instruction": "Make the pain vivid.",
               "kinds": ["prose", "image"]}

    def responder(_messages):
        return '[{"type":"image","prompt":"a vivid illustration of the pain","caption":"The pain"}]'

    # with_images=True -> the writer is told it may emit image blocks.
    on = _FakeLLM(responder)
    blocks_on = asyncio.run(pipeline._write_section(
        on, template, section, source_slice="src", sem=asyncio.Semaphore(1), with_images=True))
    sys_on = on.system_prompts()[0]
    assert '"type":"image"' in sys_on
    assert any(b.get("type") == "image" for b in blocks_on)

    # with_images=False -> no image block in the contract, and the "image" kind is dropped
    # from the per-section hint (so the model isn't nudged toward placeholder-only images).
    off = _FakeLLM(responder)
    asyncio.run(pipeline._write_section(
        off, template, section, source_slice="src", sem=asyncio.Semaphore(1), with_images=False))
    sys_off = off.system_prompts()[0]
    assert '"type":"image"' not in sys_off
    assert "PREFERRED BLOCK KINDS:" in off.calls[0][1]["content"]
    # the "image" kind hint must not appear when images are off
    assert "illustration" not in off.calls[0][1]["content"].lower()


# --- end-to-end: plan -> write -> image stage -> render --------------------------------------
def _run_generate_with_image(monkeypatch_setattr):
    from app.docgen import pipeline, images

    plan_json = (
        '{"title":"Acme Pitch","subtitle":"Seed Round","sections":['
        '{"title":"The Problem","level":1,"instruction":"the pain","kinds":["prose","image"]}'
        ']}'
    )
    writer_json = (
        '[{"type":"paragraph","text":"Teams waste hours on manual reporting."},'
        '{"type":"image","prompt":"a frustrated team buried in spreadsheets","caption":"The pain"}]'
    )

    def responder(messages):
        sys = messages[0]["content"]
        if "PLANNING agent" in sys:
            return plan_json
        return writer_json

    fake = _FakeLLM(responder)

    async def fake_generate_image(prompt, *, width=1024, height=576, seed=0):
        return _png_bytes()

    # Patch: deterministic LLM, an "enabled" image backend, and a real PNG generator.
    monkeypatch_setattr(pipeline, "_llm", lambda: fake)
    monkeypatch_setattr(images, "images_enabled", lambda: True)
    monkeypatch_setattr(images, "generate_image", fake_generate_image)

    return asyncio.run(pipeline.generate_document(
        template_id="pitch_deck",
        brief="Pitch for Acme, an AI reporting tool.",
        source_text="Acme automates reporting for finance teams.",
        with_images=True,
    ))


def test_generate_document_produces_inline_image_block(monkeypatch):
    doc = _run_generate_with_image(monkeypatch.setattr)

    image_blocks = [b for b in doc["blocks"] if b.get("type") == "image"]
    assert image_blocks, "expected at least one inline image block in the generated document"
    # The image stage must have filled the block with an embedded PNG data URL.
    assert any((b.get("data_url") or "").startswith("data:image/png;base64,") for b in image_blocks)
    # The per-template cover hero should also be produced.
    assert (doc.get("cover_image_data_url") or "").startswith("data:image/png;base64,")


def test_generated_inline_image_renders_to_pdf_and_pptx(monkeypatch):
    from app.docgen.render_pdf import render_pdf
    from app.docgen.render_pptx import render_pptx

    doc = _run_generate_with_image(monkeypatch.setattr)
    assert any(b.get("type") == "image" for b in doc["blocks"])  # precondition

    pdf = render_pdf(doc)
    assert isinstance(pdf, (bytes, bytearray)) and pdf[:4] == b"%PDF"

    pptx = render_pptx(doc)
    assert isinstance(pptx, (bytes, bytearray)) and pptx[:2] == b"PK"  # zip/OOXML magic


def test_images_disabled_skips_inline_image_generation(monkeypatch):
    """with_images=True but no backend -> want_images is False, so no image stage runs."""
    from app.docgen import pipeline, images

    plan_json = ('{"title":"X","subtitle":"Y","sections":['
                 '{"title":"The Problem","level":1,"instruction":"x","kinds":["prose"]}]}')
    writer_json = '[{"type":"paragraph","text":"No images here."}]'

    def responder(messages):
        return plan_json if "PLANNING agent" in messages[0]["content"] else writer_json

    fake = _FakeLLM(responder)
    monkeypatch.setattr(pipeline, "_llm", lambda: fake)
    monkeypatch.setattr(images, "images_enabled", lambda: False)

    doc = asyncio.run(pipeline.generate_document(
        template_id="pitch_deck", brief="b", source_text="s", with_images=True))

    assert "cover_image_data_url" not in doc
    # Planner system prompt must not advertise the image kind when generation is off.
    assert "image" not in fake.system_prompts()[0].split("favour (")[1].split(")")[0]
