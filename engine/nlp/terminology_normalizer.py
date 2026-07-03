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


def normalize(text: str) -> str:
    """Return a normalised form of *text* suitable for phrase matching."""
    lowered = text.lower()
    # Expand synonyms only as standalone tokens/phrases.
    for src, dst in SYNONYMS.items():
        lowered = re.sub(rf"\b{re.escape(src)}\b", dst, lowered)
    no_punct = _PUNCT_RE.sub(" ", lowered)
    collapsed = _WS_RE.sub(" ", no_punct).strip()
    return collapsed


def tokenize(text: str) -> list[str]:
    """Return whitespace tokens of the normalised text."""
    return normalize(text).split()
