from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ._model import build_inference_models
from .preprocessing import clean_text, normalize_khmer

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).parent / "weights"
_MODEL_PATH  = _WEIGHTS_DIR / "bi_gru_attention_khmer_transliterator.keras"
_ASSETS_PATH = _WEIGHTS_DIR / "gru_transliteration_assets.pkl"
_DICT_PATH   = _WEIGHTS_DIR / "khmer_dictionary.txt"

# Load dictionary once at module import (38k lines, ~1 MB — fast and safe)
with open(_DICT_PATH, "r", encoding="utf-8") as _f:
    _KHMER_DICT: frozenset[str] = frozenset(line.strip() for line in _f if line.strip())


class Transliterator:
    """BiGRU-Attention transliterator. The Keras model loads lazily on first use."""

    def __init__(self) -> None:
        self._loaded = False
        self._model = None
        self._encoder = None
        self._decoder = None
        self._gru_units: int | None = None
        self._eng_tokenizer = None
        self._khm_tokenizer = None
        self._max_eng_len: int | None = None
        self._max_khm_len: int | None = None

    def _load(self) -> None:
        if self._loaded:
            return
        from tensorflow.keras.models import load_model
        self._model = load_model(str(_MODEL_PATH))
        with open(_ASSETS_PATH, "rb") as f:
            assets = pickle.load(f)
        self._eng_tokenizer = assets["eng_tokenizer"]
        self._khm_tokenizer = assets["khm_tokenizer"]
        self._max_eng_len   = assets["max_eng_len"]
        self._max_khm_len   = assets["max_khm_len"]
        self._encoder, self._decoder, self._gru_units = build_inference_models(self._model)
        self._loaded = True

    # ------------------------------------------------------------------
    # Internal beam search helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str):
        cleaned = clean_text(text, is_khmer=False)
        enc_seq = self._eng_tokenizer.texts_to_sequences([cleaned])[0]
        enc_padded = pad_sequences([enc_seq], maxlen=self._max_eng_len, padding="post")
        _, encoder_proj = self._encoder.predict(enc_padded, verbose=0)
        return encoder_proj

    def _beam_search(
        self, encoder_proj, beam_width: int, max_length: int
    ) -> list[dict]:
        start_token = self._khm_tokenizer.word_index.get("\t", 1)
        end_token   = self._khm_tokenizer.word_index.get("\n", 2)
        initial_state = np.zeros((1, self._gru_units))

        beams: list[dict] = [{"seq": [start_token], "prob": 1.0, "state": initial_state, "finished": False}]
        finished_beams: list[dict] = []

        for _ in range(max_length):
            if not beams:
                break
            new_beams: list[dict] = []
            for beam in beams:
                if beam["finished"]:
                    finished_beams.append(beam)
                    continue
                target_seq = np.array([[beam["seq"][-1]]])
                outputs, new_state = self._decoder.predict(
                    [target_seq, beam["state"], encoder_proj], verbose=0
                )
                probs = outputs[0, -1, :]
                top_indices = np.argsort(probs)[-beam_width:]
                for idx in top_indices:
                    token_prob = probs[idx]
                    if token_prob <= 1e-8:
                        continue
                    new_seq  = beam["seq"] + [int(idx)]
                    new_prob = beam["prob"] * token_prob
                    done = (int(idx) == end_token) or (len(new_seq) >= max_length)
                    nb = {"seq": new_seq, "prob": new_prob, "state": new_state, "finished": done}
                    (finished_beams if done else new_beams).append(nb)
            beams = sorted(new_beams, key=lambda x: x["prob"], reverse=True)[:beam_width]
            if len(finished_beams) >= beam_width:
                break

        return finished_beams + beams

    def _decode_beam(self, beam: dict) -> str:
        start_token = self._khm_tokenizer.word_index.get("\t", 1)
        end_token   = self._khm_tokenizer.word_index.get("\n", 2)
        chars = []
        for idx in beam["seq"]:
            if idx == start_token:
                continue
            if idx == end_token:
                break
            chars.append(self._khm_tokenizer.index_word.get(idx, ""))
        return "".join(chars)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transliterate(self, text: str, beam_width: int = 3, max_length: int = 32) -> str:
        """Return the single best Khmer transliteration for *text*."""
        if not text.strip():
            return ""
        self._load()
        encoder_proj = self._encode(text)
        all_beams = self._beam_search(encoder_proj, beam_width, max_length)
        if not all_beams:
            return ""
        best = max(all_beams, key=lambda x: x["prob"])
        return self._decode_beam(best)

    def transliterate_top_n(
        self, text: str, n: int = 3, beam_width: int = 5, max_length: int = 32
    ) -> list[str]:
        """Return the top *n* Khmer candidates for *text* (raw model output, no dict check)."""
        if not text.strip():
            return []
        self._load()
        encoder_proj = self._encode(text)
        all_beams = self._beam_search(encoder_proj, beam_width, max_length)
        if not all_beams:
            return []
        top = sorted(all_beams, key=lambda x: x["prob"], reverse=True)[:n]
        return [self._decode_beam(b) for b in top]

    def transliterate_with_dict(
        self,
        text: str,
        n: int = 5,
        beam_width: int = 5,
        max_length: int = 32,
        max_distance: int = 2,
    ) -> list[str]:
        """Return up to *n* dictionary-validated Khmer candidates for *text*.

        Step 1: generate top-n candidates via beam search.
        Step 2: normalize with khmerspell.
        Step 3: exact dictionary matches first.
        Step 4: fuzzy (Levenshtein) matches for remaining slots.
        """
        if not text.strip():
            return []
        self._load()

        import Levenshtein

        candidates = self.transliterate_top_n(text, n=n, beam_width=beam_width, max_length=max_length)
        candidates = [normalize_khmer(c) for c in candidates]
        logger.debug("Top candidates from model: %s", candidates)

        valid: list[str] = []
        used: set[str] = set()

        # Exact matches
        for cand in candidates:
            if cand in _KHMER_DICT and cand not in used:
                valid.append(cand)
                used.add(cand)

        # Fuzzy matches for remaining slots
        if len(valid) < n:
            for cand in candidates:
                if len(valid) >= n:
                    break
                best_match: str | None = None
                best_dist = float("inf")
                for word in _KHMER_DICT:
                    if abs(len(word) - len(cand)) > 2:
                        continue
                    d = Levenshtein.distance(cand, word)
                    if d <= max_distance and d < best_dist:
                        best_match = word
                        best_dist = d
                if best_match and best_match not in used:
                    valid.append(best_match)
                    used.add(best_match)

        result = list(dict.fromkeys(valid))
        logger.debug("Valid candidates after filtering: %s", result)
        return result if result else candidates


# Module-level singleton and convenience functions
_t = Transliterator()


def transliterate(text: str, beam_width: int = 3, max_length: int = 32) -> str:
    """Return the single best Khmer transliteration for *text*."""
    return _t.transliterate(text, beam_width=beam_width, max_length=max_length)


def transliterate_top_n(
    text: str, n: int = 3, beam_width: int = 5, max_length: int = 32
) -> list[str]:
    """Return the top *n* Khmer candidates for *text* (raw model output)."""
    return _t.transliterate_top_n(text, n=n, beam_width=beam_width, max_length=max_length)


def transliterate_with_dict(
    text: str,
    n: int = 5,
    beam_width: int = 5,
    max_length: int = 32,
    max_distance: int = 2,
) -> list[str]:
    """Return up to *n* dictionary-validated Khmer candidates for *text*."""
    return _t.transliterate_with_dict(
        text, n=n, beam_width=beam_width, max_length=max_length, max_distance=max_distance
    )
