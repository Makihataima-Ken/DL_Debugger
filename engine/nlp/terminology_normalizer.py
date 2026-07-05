"""
terminology_normalizer.py
=========================
Lightweight, dependency-free text normalisation for the NLP layer.

Normalisation steps (pure string ops only):
  * lowercase
  * strip punctuation to spaces
  * collapse whitespace
  * expand domain synonyms/abbreviations
"""

from __future__ import annotations

import re

from engine.nlp.training_debugging_vocabulary import SYNONYMS

_PUNCT_RE = re.compile(r"[^a-z0-9%\s]+")
_WS_RE = re.compile(r"\s+")
_CONTRACTIONS: dict[str, str] = {
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "won't": "will not",
    "can't": "can not",
    "cannot": "can not",
}


def expand_terms(text: str) -> str:
    """Lowercase *text* and expand controlled contractions/synonyms."""
    lowered = text.lower()
    for src, dst in _CONTRACTIONS.items():
        lowered = lowered.replace(src, dst)
    # Expand synonyms only as standalone tokens/phrases.
    for src, dst in SYNONYMS.items():
        lowered = re.sub(rf"\b{re.escape(src)}\b", dst, lowered)
    return lowered


def normalize(text: str) -> str:
    """Return a normalised form of *text* suitable for token utilities."""
    expanded = expand_terms(text)
    no_punct = _PUNCT_RE.sub(" ", expanded)
    collapsed = _WS_RE.sub(" ", no_punct).strip()
    return collapsed


def tokenize(text: str) -> list[str]:
    """Return whitespace tokens of the normalised text."""
    return normalize(text).split()
