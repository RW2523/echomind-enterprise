"""
Image generation for Document Studio.

Generates illustrative/cover images for documents so PDFs/PPTs look professional. Backend is
selected by env DOCGEN_IMAGE_BACKEND:

  comfyui  -> a local ComfyUI server (the NVIDIA build.nvidia.com/station/comfyui setup).
              COMFYUI_URL (default http://comfyui:8188). Flow: POST /prompt {prompt: <workflow>,
              client_id} -> poll GET /history/{id} -> GET /view?filename&subfolder&type=output.
              The text2image workflow is taken from COMFYUI_WORKFLOW (a JSON file with the
              placeholders {PROMPT},{NEG},{SEED},{WIDTH},{HEIGHT},{CKPT}) or a built-in SD-style
              default; the checkpoint name is COMFYUI_CKPT.
  nim      -> an NVIDIA NIM image endpoint. DOCGEN_IMAGE_NIM_URL + NVIDIA_API_KEY (Bearer).
              Tolerant parse of {artifacts:[{base64}]} / {image} / {data:[{b64_json}]}.
  diffusers-> real on-device text-to-image (e.g. SDXL-Turbo) via the diffusers library on the
              GPU. Private + offline after a one-time model download. DOCGEN_DIFFUSERS_MODEL,
              DOCGEN_DIFFUSERS_CACHE, DOCGEN_DIFFUSERS_STEPS, DOCGEN_DIFFUSERS_MAX.
  local    -> offline synthetic generator (Pillow). No GPU/network/model: renders a clean,
              prompt-tinted abstract hero so docs look designed out of the box. Not diffusion.
  none/unset -> disabled; generate_image returns None and renderers draw a tasteful placeholder.

generate_image NEVER raises — on any error it returns None so a document still renders.
Generated PNGs are cached on disk (DATA_DIR/docgen_images) keyed by a hash of prompt+size+backend.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import string
import threading
from io import BytesIO
from typing import Optional

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(settings.DATA_DIR, "docgen_images")


def backend_name() -> str:
    return (os.getenv("DOCGEN_IMAGE_BACKEND", "none") or "none").strip().lower()


def images_enabled() -> bool:
    return backend_name() in ("comfyui", "nim", "local", "diffusers")


def image_backend_status() -> dict:
    b = backend_name()
    status = {"backend": b, "enabled": b in ("comfyui", "nim", "local", "diffusers")}
    if b == "comfyui":
        status["url"] = os.getenv("COMFYUI_URL", "http://comfyui:8188")
        status["checkpoint"] = os.getenv("COMFYUI_CKPT", "")
    elif b == "nim":
        status["url"] = os.getenv("DOCGEN_IMAGE_NIM_URL", "")
        status["has_key"] = bool(os.getenv("NVIDIA_API_KEY"))
    elif b == "local":
        status["mode"] = "offline-synthetic"
    elif b == "diffusers":
        status["mode"] = "on-device"
        status["model"] = os.getenv("DOCGEN_DIFFUSERS_MODEL", "stabilityai/sdxl-turbo")
    return status


def _cache_path(key: str) -> str:
    return os.path.join(_CACHE_DIR, f"{key}.png")


def _key(prompt: str, width: int, height: int) -> str:
    raw = f"{backend_name()}|{width}x{height}|{prompt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


# ── default ComfyUI SD-style text2image workflow (API format) ────────────────
# Placeholders are substituted with the prompt/seed/size/checkpoint. Override entirely by pointing
# COMFYUI_WORKFLOW at a JSON file exported from your ComfyUI graph (Save (API Format)).
_DEFAULT_COMFY_WORKFLOW = """
{
  "3": {"class_type": "KSampler", "inputs": {
      "seed": {SEED}, "steps": 25, "cfg": 6.5, "sampler_name": "euler", "scheduler": "normal",
      "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
  "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "{CKPT}"}},
  "5": {"class_type": "EmptyLatentImage", "inputs": {"width": {WIDTH}, "height": {HEIGHT}, "batch_size": 1}},
  "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "{PROMPT}", "clip": ["4", 1]}},
  "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "{NEG}", "clip": ["4", 1]}},
  "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
  "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "docgen", "images": ["8", 0]}}
}
"""

_NEG = "text, watermark, signature, logo, blurry, low quality, distorted, deformed, jpeg artifacts"


def _json_escape(s: str) -> str:
    # escape for safe inline substitution into the workflow JSON string values
    return json.dumps(s)[1:-1]


def _build_comfy_workflow(prompt: str, width: int, height: int, seed: int) -> Optional[dict]:
    tmpl = None
    path = os.getenv("COMFYUI_WORKFLOW", "").strip()
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                tmpl = f.read()
        except Exception as e:
            logger.warning("docgen images: failed reading COMFYUI_WORKFLOW: %s", e)
    if tmpl is None:
        tmpl = _DEFAULT_COMFY_WORKFLOW
    ckpt = os.getenv("COMFYUI_CKPT", "sd_xl_base_1.0.safetensors")
    filled = (
        tmpl.replace("{PROMPT}", _json_escape(prompt))
        .replace("{NEG}", _json_escape(_NEG))
        .replace("{SEED}", str(seed))
        .replace("{WIDTH}", str(width))
        .replace("{HEIGHT}", str(height))
        .replace("{CKPT}", _json_escape(ckpt))
    )
    try:
        return json.loads(filled)
    except Exception as e:
        logger.warning("docgen images: bad ComfyUI workflow JSON after fill: %s", e)
        return None


async def _gen_comfyui(prompt: str, width: int, height: int, seed: int) -> Optional[bytes]:
    base = os.getenv("COMFYUI_URL", "http://comfyui:8188").rstrip("/")
    workflow = _build_comfy_workflow(prompt, width, height, seed)
    if not workflow:
        return None
    client_id = hashlib.sha256(f"{seed}{prompt}".encode()).hexdigest()[:16]
    timeout = float(os.getenv("DOCGEN_IMAGE_TIMEOUT", "120"))
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{base}/prompt", json={"prompt": workflow, "client_id": client_id})
        r.raise_for_status()
        pid = r.json().get("prompt_id")
        if not pid:
            return None
        # poll history
        deadline = asyncio.get_event_loop().time() + timeout
        hist = None
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(1.5)
            h = await client.get(f"{base}/history/{pid}")
            if h.status_code == 200 and h.json().get(pid):
                hist = h.json()[pid]
                break
        if not hist:
            logger.warning("docgen images: ComfyUI history timed out for %s", pid)
            return None
        # find an output image
        for node_out in (hist.get("outputs") or {}).values():
            for img in (node_out.get("images") or []):
                params = {"filename": img.get("filename"), "subfolder": img.get("subfolder", ""),
                          "type": img.get("type", "output")}
                v = await client.get(f"{base}/view", params=params)
                if v.status_code == 200 and v.content:
                    return v.content
    return None


async def _gen_nim(prompt: str, width: int, height: int, seed: int) -> Optional[bytes]:
    url = os.getenv("DOCGEN_IMAGE_NIM_URL", "").strip()
    key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not url or not key:
        return None
    payload = {
        "prompt": prompt, "negative_prompt": _NEG,
        "width": width, "height": height, "seed": seed,
        "cfg_scale": 5, "steps": 25, "samples": 1,
    }
    timeout = float(os.getenv("DOCGEN_IMAGE_TIMEOUT", "120"))
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"})
        r.raise_for_status()
        j = r.json()
    # tolerant extraction of base64 image from common NIM/SDXL/FLUX shapes
    for path in (("artifacts", 0, "base64"), ("image",), ("data", 0, "b64_json"), ("images", 0)):
        cur = j
        try:
            for p in path:
                cur = cur[p]
            if isinstance(cur, str) and cur:
                b = cur.split(",", 1)[-1] if cur.startswith("data:") else cur
                return base64.b64decode(b)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    logger.warning("docgen images: could not parse NIM image response keys=%s", list(j)[:6] if isinstance(j, dict) else type(j))
    return None


# ── local (offline) backend: synthesize a clean, professional abstract image ──
# No GPU / network / model required — useful for offline deployments and for an
# out-of-the-box "Generate with images" experience. It is NOT a diffusion model:
# it renders a tasteful, prompt-tinted abstract hero (gradient + soft geometry)
# so documents look designed rather than carrying empty placeholders. Switch to
# the comfyui / nim backends for true text-to-image generation.
_PALETTES = {
    "blue":    ((18, 32, 68), (9, 16, 38), (56, 132, 255)),
    "cyan":    ((10, 30, 46), (7, 16, 28), (34, 211, 238)),
    "teal":    ((8, 38, 40), (6, 20, 24), (20, 184, 166)),
    "emerald": ((8, 40, 28), (6, 20, 16), (16, 185, 129)),
    "green":   ((10, 38, 26), (6, 20, 15), (34, 197, 94)),
    "indigo":  ((24, 22, 60), (13, 11, 34), (99, 102, 241)),
    "violet":  ((30, 18, 56), (15, 9, 32), (139, 92, 246)),
    "purple":  ((30, 18, 56), (15, 9, 32), (168, 85, 247)),
    "magenta": ((48, 12, 40), (22, 7, 22), (232, 62, 140)),
    "pink":    ((48, 14, 40), (22, 8, 22), (244, 114, 182)),
    "amber":   ((48, 34, 10), (24, 17, 6), (245, 158, 11)),
    "gold":    ((48, 38, 12), (24, 19, 6), (234, 179, 8)),
    "orange":  ((48, 26, 12), (24, 12, 6), (249, 115, 22)),
    "warm":    ((44, 24, 16), (22, 11, 8), (249, 115, 22)),
    "red":     ((48, 16, 16), (24, 8, 8), (239, 68, 68)),
    "slate":   ((24, 30, 40), (12, 16, 24), (148, 163, 184)),
}
_DEFAULT_PALETTE = ((12, 18, 34), (7, 11, 22), (56, 189, 248))   # midnight navy + sky


def _palette_from_prompt(prompt: str):
    p = (prompt or "").lower()
    for word, pal in _PALETTES.items():
        if word in p:
            return pal
    return _DEFAULT_PALETTE


def _render_local_png(prompt: str, width: int, height: int, seed: int) -> bytes:
    """Render a professional abstract hero image (synchronous; called via executor)."""
    import random as _random
    from PIL import Image, ImageDraw, ImageFilter

    rnd = _random.Random(seed)
    top, bot, accent = _palette_from_prompt(prompt)

    # Vertical gradient background.
    grad = Image.new("RGB", (1, height))
    for y in range(height):
        t = y / max(1, height - 1)
        grad.putpixel((0, y), tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3)))
    img = grad.resize((width, height)).convert("RGBA")

    # Soft abstract geometry on a translucent layer.
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    mn = min(width, height)
    for _ in range(rnd.randint(7, 11)):
        r = rnd.randint(int(mn * 0.14), int(mn * 0.52))
        cx = rnd.randint(int(-0.1 * width), int(1.1 * width))
        cy = rnd.randint(int(-0.1 * height), int(1.1 * height))
        col = accent if rnd.random() < 0.62 else (255, 255, 255)
        if rnd.random() < 0.5:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (rnd.randint(10, 24),))
        else:
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      outline=col + (rnd.randint(34, 72),), width=max(2, r // 28))
    for _ in range(rnd.randint(2, 4)):
        x0, y0 = rnd.randint(0, width), rnd.randint(0, height)
        d.line([x0, y0, x0 + rnd.randint(-width, width), y0 + rnd.randint(-height, height)],
               fill=accent + (rnd.randint(22, 46),), width=max(1, mn // 300))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(1.0, mn / 240.0)))
    img = Image.alpha_composite(img, layer)

    # Gentle corner vignette for depth.
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).ellipse(
        [int(-width * 0.25), int(-height * 0.25), int(width * 1.25), int(height * 1.25)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=mn / 7.0))
    shade = Image.new("RGBA", (width, height), (0, 0, 0, 105))
    img = Image.composite(img, Image.alpha_composite(img, shade), mask)

    out = BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()


async def _gen_local(prompt: str, width: int, height: int, seed: int) -> Optional[bytes]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _render_local_png, prompt, width, height, seed)


# ── diffusers (local GPU text-to-image, e.g. SDXL-Turbo) ──────────────────────
# Real on-device generation: fully private, offline after a one-time model download.
# The pipeline loads once (lazily) and is reused; GPU calls are serialized by a lock.
_diff_pipe = None
_diff_lock = threading.Lock()


def _load_diffusers_pipe():
    global _diff_pipe
    if _diff_pipe is not None:
        return _diff_pipe
    with _diff_lock:
        if _diff_pipe is not None:
            return _diff_pipe
        import torch
        from diffusers import AutoPipelineForText2Image
        model = os.getenv("DOCGEN_DIFFUSERS_MODEL", "stabilityai/sdxl-turbo")
        cache = os.getenv("DOCGEN_DIFFUSERS_CACHE", os.path.join(settings.DATA_DIR, "docgen_models"))
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model, torch_dtype=torch.float16, variant="fp16", cache_dir=cache)
        except Exception:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model, torch_dtype=torch.float16, cache_dir=cache)
        pipe = pipe.to("cuda")
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        _diff_pipe = pipe
        logger.info("docgen images: diffusers pipeline loaded (%s)", model)
        return _diff_pipe


# Hard guardrails: strip text/diagram/UI requests that diffusion renders as garbage, and
# append quality/style tokens. Belt-and-suspenders behind the pipeline's art-director subagent.
_DIFF_STYLE_SUFFIX = "cinematic lighting, professional, highly detailed, sharp focus, minimal, no text"
_DIFF_REPLACE = [
    (r"\b(architecture|system|technical|technology)\s+diagrams?\b", "abstract representation of interconnected technology"),
    (r"\bflow\s*charts?\b", "abstract flowing connected forms"),
    (r"\bdiagrams?\b", "abstract representation"),
    (r"\bflowcharts?\b", "abstract flowing connected forms"),
    (r"\binfographics?\b", "abstract illustration"),
    (r"\b(schematics?|blueprints?|wireframes?)\b", "abstract structure"),
    (r"\b(charts?|graphs?|plots?)\b", "abstract network of glowing nodes"),
    (r"\bscreenshots?\b", "sleek glowing device"),
    (r"\bdashboards?\b", "glowing abstract panels"),
    (r"\b(user\s+interface|ui|interface)\b", "glowing abstract panels"),
    (r"\b(tables?|spreadsheets?)\b", "abstract grid of light"),
    (r"\blabell?ed\b", ""),
    (r"\b(text|words?|letters?|labels?|captions?|titles?|fonts?|typography|signage|logos?)\b", ""),
]


def _sanitize_for_diffusion(prompt: str) -> str:
    """Steer a prompt toward what diffusion does well: a clean, text-free, professional scene."""
    p = (prompt or "").strip()
    for pat, repl in _DIFF_REPLACE:
        p = re.sub(pat, repl, p, flags=re.I)
    p = re.sub(r"\s{2,}", " ", p).strip(" ,.;:")
    if not p:
        p = "a clean professional abstract technology illustration"
    if "no text" not in p.lower():
        p = p + ", " + _DIFF_STYLE_SUFFIX
    return p


def _render_diffusers_png(prompt: str, width: int, height: int, seed: int) -> Optional[bytes]:
    import torch
    prompt = _sanitize_for_diffusion(prompt)
    pipe = _load_diffusers_pipe()
    steps = max(1, int(os.getenv("DOCGEN_DIFFUSERS_STEPS", "4")))
    guidance = float(os.getenv("DOCGEN_DIFFUSERS_GUIDANCE", "0.0"))
    cap = max(384, int(os.getenv("DOCGEN_DIFFUSERS_MAX", "1024")))
    neg = os.getenv("DOCGEN_DIFFUSERS_NEG", _NEG)
    # Clamp to a model-friendly size (longest side <= cap, multiples of 8).
    w, h = int(width), int(height)
    if max(w, h) > cap:
        if w >= h:
            h = int(h * cap / w); w = cap
        else:
            w = int(w * cap / h); h = cap
    w = max(384, (w // 8) * 8)
    h = max(384, (h // 8) * 8)
    kwargs = dict(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
                  width=w, height=h)
    # SDXL-Turbo ignores negative prompts at guidance 0; pass it only when guided.
    if guidance and guidance > 0:
        kwargs["negative_prompt"] = neg
    with _diff_lock:
        g = torch.Generator(device="cuda").manual_seed(int(seed) & 0x7FFFFFFF)
        img = pipe(generator=g, **kwargs).images[0]
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _gen_diffusers(prompt: str, width: int, height: int, seed: int) -> Optional[bytes]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _render_diffusers_png, prompt, width, height, seed)


async def generate_image(prompt: str, *, width: int = 1024, height: int = 576, seed: int = 0) -> Optional[bytes]:
    """Generate a PNG for `prompt`. Returns bytes or None (disabled / error). Never raises."""
    prompt = (prompt or "").strip()
    if not prompt or not images_enabled():
        return None
    # clamp + round to multiples of 8 (model requirement)
    width = max(256, min(1536, (int(width) // 8) * 8))
    height = max(256, min(1536, (int(height) // 8) * 8))
    seed = int(seed) or (int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 2_000_000_000)

    key = _key(prompt, width, height)
    cp = _cache_path(key)
    try:
        if os.path.exists(cp):
            with open(cp, "rb") as f:
                return f.read()
    except Exception:
        pass

    b = backend_name()
    try:
        if b == "comfyui":
            data = await _gen_comfyui(prompt, width, height, seed)
        elif b == "nim":
            data = await _gen_nim(prompt, width, height, seed)
        elif b == "diffusers":
            data = await _gen_diffusers(prompt, width, height, seed)
        elif b == "local":
            data = await _gen_local(prompt, width, height, seed)
        else:
            data = None
    except Exception as e:
        logger.warning("docgen images: %s generation failed: %s", b, e)
        data = None
    if not data:
        return None
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(cp, "wb") as f:
            f.write(data)
    except Exception:
        pass
    return data


def data_url_png(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


# Printable characters allowed in workflow substitution (defensive; unused but handy for callers)
_SAFE = set(string.printable)
