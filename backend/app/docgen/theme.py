"""
Visual themes for Document Studio renderers.

A theme is a small palette + font choice. Templates pick a theme by name (see templates.py
"theme" key); render_pdf and render_pptx both consume get_theme(name) so a document looks
consistent and professionally designed across PDF and PPTX.

Colors are hex strings "RRGGBB" (no leading #) so both reportlab (HexColor('#'+x)) and
python-pptx (RGBColor(int(x[:2],16),...)) can use them via the helpers here.
"""
from __future__ import annotations

from typing import Dict, Tuple

# name -> palette
_THEMES: Dict[str, Dict[str, str]] = {
    # Deep slate + cyan — technical / engineering (default, matches the app)
    "midnight": {
        "primary": "0EA5B7", "primary_dark": "0E7490", "accent": "22D3EE",
        "ink": "0F172A", "body": "1E293B", "muted": "64748B",
        "bg": "FFFFFF", "panel": "F1F5F9", "panel_alt": "F8FAFC",
        "cover_bg": "0F172A", "cover_ink": "FFFFFF", "cover_accent": "22D3EE",
        "font": "Helvetica", "font_bold": "Helvetica-Bold",
    },
    # Indigo/violet — executive / business
    "executive": {
        "primary": "4F46E5", "primary_dark": "3730A3", "accent": "818CF8",
        "ink": "1E1B4B", "body": "312E81", "muted": "6B7280",
        "bg": "FFFFFF", "panel": "EEF2FF", "panel_alt": "F8FAFC",
        "cover_bg": "1E1B4B", "cover_ink": "FFFFFF", "cover_accent": "A5B4FC",
        "font": "Helvetica", "font_bold": "Helvetica-Bold",
    },
    # Teal/emerald — process / SOP / training
    "evergreen": {
        "primary": "0D9488", "primary_dark": "0F766E", "accent": "2DD4BF",
        "ink": "042F2E", "body": "134E4A", "muted": "6B7280",
        "bg": "FFFFFF", "panel": "ECFDF5", "panel_alt": "F8FAFC",
        "cover_bg": "042F2E", "cover_ink": "FFFFFF", "cover_accent": "5EEAD4",
        "font": "Helvetica", "font_bold": "Helvetica-Bold",
    },
    # Slate/amber — legal / compliance (serious, warm accent)
    "counsel": {
        "primary": "B45309", "primary_dark": "92400E", "accent": "F59E0B",
        "ink": "1C1917", "body": "292524", "muted": "78716C",
        "bg": "FFFFFF", "panel": "FEF3C7", "panel_alt": "FAFAF9",
        "cover_bg": "1C1917", "cover_ink": "FFFFFF", "cover_accent": "FBBF24",
        "font": "Times-Roman", "font_bold": "Times-Bold",
    },
    # Bold near-black + magenta — pitch deck / marketing (high contrast)
    "spotlight": {
        "primary": "DB2777", "primary_dark": "9D174D", "accent": "F472B6",
        "ink": "0A0A0A", "body": "1F2937", "muted": "6B7280",
        "bg": "FFFFFF", "panel": "FDF2F8", "panel_alt": "FAFAFA",
        "cover_bg": "0A0A0A", "cover_ink": "FFFFFF", "cover_accent": "F472B6",
        "font": "Helvetica", "font_bold": "Helvetica-Bold",
    },
}

DEFAULT_THEME = "midnight"

# callout style -> (border/accent color key, soft background hex)
CALLOUT_COLORS = {
    "info": ("3B82F6", "EFF6FF"),
    "warning": ("D97706", "FFFBEB"),
    "principle": ("0891B2", "ECFEFF"),
    "success": ("16A34A", "F0FDF4"),
}


def get_theme(name: str) -> Dict[str, str]:
    return dict(_THEMES.get(name or "", _THEMES[DEFAULT_THEME]))


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = (h or "000000").lstrip("#")
    if len(h) != 6:
        h = "000000"
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hexc(h: str) -> str:
    """Return '#RRGGBB' for reportlab HexColor."""
    return "#" + (h or "000000").lstrip("#")
