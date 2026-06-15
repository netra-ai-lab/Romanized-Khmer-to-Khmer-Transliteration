"""
khmer_transliterator — English-to-Khmer transliteration using a BiGRU-Attention model.

Quick start::

    from khmer_transliterator import transliterate, transliterate_with_dict

    transliterate("sokha")                  # -> 'សុខា'
    transliterate_with_dict("sokha", n=3)   # -> ['សុខា', 'សុខ', ...]

The Keras model is loaded lazily on the first transliteration call.
"""

from __future__ import annotations

from khmer_transliterator._inference import (
    Transliterator,
    transliterate,
    transliterate_top_n,
    transliterate_with_dict,
)
from khmer_transliterator.preprocessing import clean_text, normalize_khmer

__version__ = "1.0.1"
__author__ = "Darayut Nhem"

__all__ = [
    "Transliterator",
    "transliterate",
    "transliterate_top_n",
    "transliterate_with_dict",
    "clean_text",
    "normalize_khmer",
    "__version__",
]
