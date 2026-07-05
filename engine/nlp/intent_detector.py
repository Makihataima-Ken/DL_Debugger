"""
intent_detector.py
==================
Coarse intent classification for routing (not diagnosis).

Returns one of: "diagnose", "explain", "help", "unknown". This only affects
CLI presentation; it never changes rule firing.
"""

from __future__ import annotations

from engine.nlp.terminology_normalizer import normalize

_EXPLAIN_CUES = ("why", "explain", "reason", "how come", "rationale")
_HELP_CUES = ("help", "usage", "how do i", "what can you")
_DIAGNOSE_CUES = (
    "loss", "accuracy", "gradient", "overfit", "underfit", "nan",
    "diverg", "converg", "attention", "relu", "tokeniz", "dataset",
)


def detect_intent(text: str) -> str:
    norm = normalize(text)
    if any(cue in norm for cue in _HELP_CUES):
        return "help"
    if any(cue in norm for cue in _DIAGNOSE_CUES):
        return "diagnose"
    if any(cue in norm for cue in _EXPLAIN_CUES):
        return "explain"
    return "unknown"
