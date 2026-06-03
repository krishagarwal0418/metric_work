from __future__ import annotations

import html
import re
import unicodedata
from urllib.parse import unquote

from observation_labeler.types import NormalizedText, ObfuscationSignals

_WHITESPACE = re.compile(r"\s+")
_HTML_ENTITY = re.compile(r"&(#x?[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")
_URL_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_BASE64ISH = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
_ZERO_WIDTH_RANGES = (
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
)
_TAG_MIN = 0xE0000
_TAG_MAX = 0xE007F

_HOMOGLYPHS = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
}


def _is_invisible(ch: str) -> bool:
    return ch in _ZERO_WIDTH_RANGES or _TAG_MIN <= ord(ch) <= _TAG_MAX


def _strip_invisible(text: str) -> tuple[str, int]:
    count = sum(1 for ch in text if _is_invisible(ch))
    return "".join(ch for ch in text if not _is_invisible(ch)), count


def _fold_homoglyphs(text: str) -> tuple[str, int]:
    hits = sum(1 for ch in text if ch in _HOMOGLYPHS)
    return "".join(_HOMOGLYPHS.get(ch, ch) for ch in text), hits


def normalize_channel(content: str) -> NormalizedText:
    nfkc = unicodedata.normalize("NFKC", content)
    stripped, zero_width_count = _strip_invisible(nfkc)
    folded, homoglyph_hits = _fold_homoglyphs(stripped)

    had_html = bool(_HTML_ENTITY.search(folded))
    html_decoded = html.unescape(folded) if had_html else folded

    had_url = bool(_URL_ESCAPE.search(html_decoded))
    url_decoded = unquote(html_decoded) if had_url else html_decoded

    encoded_payload_detected = bool(_BASE64ISH.search(url_decoded))
    normalized = _WHITESPACE.sub(" ", url_decoded).strip()
    classification_text = normalized.lower()

    material = bool(
        nfkc != content
        or zero_width_count
        or homoglyph_hits
        or had_html
        or had_url
        or encoded_payload_detected
    )

    return NormalizedText(
        original=content,
        normalized=normalized,
        classification_text=classification_text,
        normalization_was_material=material,
        obfuscation_signals=ObfuscationSignals(
            zero_width_count=zero_width_count,
            homoglyph_hits=homoglyph_hits,
            had_html_entity=had_html,
            had_url_escape=had_url,
            encoded_payload_detected=encoded_payload_detected,
        ),
    )
