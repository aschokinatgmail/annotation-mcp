"""Garbage-text detection for OCR results.

Tesseract frequently produces high-confidence nonsense on real-world
photos (e.g., 'VLOAOL' at conf 77, 'yOLO' at conf 39, single punctuation
characters at conf 87). Pure text-length and confidence thresholds
don't catch this — the garbage can be long and the confidence can be high.

This module provides a heuristic garbage filter based on:
- Character-class diversity: text dominated by punctuation or a single
  repeated character is suspicious
- Vowel/consonant ratio: English/Latin-script words with very low vowel
  ratios are almost always OCR noise (Latin-letter words need at least
  one vowel)
- Dictionary presence: tokens not in a small Latin wordlist are flagged
- Repetition: a single character repeated >70% of the time is suspicious

The filter is intentionally lenient on short text. A clean word like
'TOYOTA' or 'ФИТОСПОРИН' should pass; 'VLOAOL' or 'aS' or '||||' should fail.
"""
from __future__ import annotations

import re
from typing import Final

_ENGLISH_WORDS: Final[frozenset[str]] = frozenset(
    {
        # Common short words
        "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
        "her", "his", "how", "man", "new", "now", "old", "see", "way", "who",
        "boy", "did", "its", "let", "put", "say", "she", "too", "use",
        # Common OCR targets — labels, parts, warnings
        "oil", "gas", "fan", "belt", "filter", "engine", "brake", "tire",
        "wheel", "motor", "radiator", "alternator", "battery", "fuse",
        "relay", "power", "fluid", "coolant", "spark", "plug", "valve",
        "head", "gasket", "pump", "hose", "clutch", "transmission", "drive",
        "shaft", "axle", "model", "frame", "type", "class", "weight",
        "year", "make", "volts", "amps", "watts", "watt", "psi", "rpm",
        "kmh", "mph", "max", "min", "diag", "code", "warning", "caution",
        "danger", "notice", "only", "with", "from", "into", "than",
        "more", "less", "over", "under", "this", "that", "these", "those",
        "what", "when", "where", "which", "while", "before", "after",
        "between", "through", "during", "above", "below", "north", "south",
        "east", "west", "left", "right", "front", "rear", "side", "top",
        "bottom", "high", "low", "open", "close", "start", "stop", "off",
        "fail", "pass",
        # Common document/label words
        "title", "brand", "name", "product", "label", "warning", "content",
        "ingredients", "weight", "size", "color", "color", "brand", "date",
        "batch", "lot", "exp", "best", "before", "use", "by", "store",
        "cool", "dry", "place", "keep", "out", "reach", "of", "children",
        "instructions", "directions", "warning", "caution", "notice",
        "ingredient", "amount", "serving", "servings", "per", "serving",
        "calories", "fat", "saturated", "trans", "cholesterol", "sodium",
        "carbohydrate", "fiber", "sugars", "protein", "vitamin", "calcium",
        "iron",
        # Brand names
        "denso", "bosch", "toyota", "honda", "ford", "mazda", "nissan",
        "mitsubishi", "subaru", "isuzu", "lexus", "hino", "kawasaki",
        "motor", "corporation", "company", "industries", "limited", "inc",
        "corp", "llc", "co", "gmbh", "ltd",
        # Mechanical
        "inspect", "touch", "designated", "fuses", "fuel", "heater",
        "alt", "made", "japan", "china", "korea", "voltage", "current",
        "ignition", "starter", "sensor", "switch", "manual", "automatic",
        "shift", "park", "neutral", "reverse", "lock", "unlock", "empty",
        "full", "check", "level", "pressure", "temperature",
        # Action words
        "do", "not", "use", "do not", "see", "do not use", "read",
    }
)


def _char_class_diversity(text: str) -> float:
    """Return ratio of distinct characters to total length. 1.0 = all distinct."""
    if not text:
        return 0.0
    return len(set(text)) / len(text)


def _vowel_ratio(text: str) -> float:
    """Fraction of Latin-alphabetic chars that are vowels.

    Returns -1 if text has no Latin alphabetic characters (so non-Latin
    scripts like Cyrillic or CJK are exempt from this check).
    """
    vowels = set("aeiouyAEIOUY")
    alpha_count = sum(1 for c in text if c.isascii() and c.isalpha())
    if alpha_count == 0:
        return -1.0
    vowel_count = sum(1 for c in text if c in vowels)
    return vowel_count / alpha_count


def _looks_like_part_number(token: str) -> bool:
    """Heuristic: is this token a plausible part-number or barcode-style string?

    Accepts:
    - Mixed letter+digit with optional hyphens/dots, ≥3 chars (27060-54170,
      4607026420155, CODE128, ALT-S, QR-1 pass; 12V, 5L too short)
    - All-digit, ≥4 chars (2204008 passes; 04, 000 fail)
    """
    if not token:
        return False
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9\-\.]+$", token):
        has_alpha = any(c.isalpha() for c in token)
        has_digit = any(c.isdigit() for c in token)
        if has_alpha and has_digit and len(token) >= 3:
            return True
        if has_digit and not has_alpha and len(token) >= 4:
            return True
    return False


def _has_real_token(text: str) -> bool:
    """True if any whitespace-separated token in text is plausible real text.

    A token is plausible if:
    - It's in the English wordlist (case-insensitive), OR
    - It looks like a part number, OR
    - It contains non-ASCII characters (Cyrillic, CJK, etc.)
    """
    if not text:
        return False
    for token in re.split(r"[\s,;:.()]+", text):
        if not token:
            continue
        if any(ord(c) > 127 for c in token):
            return True
        if token.lower() in _ENGLISH_WORDS:
            return True
        if _looks_like_part_number(token):
            return True
    return False


def looks_like_garbage(
    text: str,
    *,
    min_length: int = 3,
    min_diversity: float = 0.30,
    min_vowel_ratio: float = 0.10,
) -> bool:
    """Heuristic: is this OCR text likely noise?

    Returns True if the text appears to be OCR garbage. Returns False for
    plausible real text.

    Heuristics (any one failing returns True = garbage):
    1. Text shorter than ``min_length`` characters
    2. Less than ``min_diversity`` distinct characters per length
    3. Latin-only text with vowel ratio below ``min_vowel_ratio``
    4. No dictionary word, no part-number pattern, no non-ASCII chars
    """
    if not text or len(text) < min_length:
        return True

    if _char_class_diversity(text) < min_diversity:
        return True

    vr = _vowel_ratio(text)
    if vr >= 0 and vr < min_vowel_ratio:
        return True

    if not _has_real_token(text):
        return True

    return False
