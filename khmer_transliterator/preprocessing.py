from __future__ import annotations

import re

from khmerspell import khnormal


def normalize_khmer(text: str) -> str:
    """Correct character order and apply Unicode normalization to Khmer text."""
    return khnormal(text)


def clean_text(text: str, is_khmer: bool = False) -> str:
    """Remove non-target characters and normalize text.

    For English (is_khmer=False): strips everything except a-z and lowercases.
    For Khmer (is_khmer=True): keeps only Khmer Unicode block (U+1780–U+17FF) and normalizes.
    """
    text = str(text).strip()
    if is_khmer:
        text = re.sub(r"[^ក-៿]", "", text)
        text = normalize_khmer(text)
    else:
        text = re.sub(r"[^a-z]", "", text.lower())
    return text
