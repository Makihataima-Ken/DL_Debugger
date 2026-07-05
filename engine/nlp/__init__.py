"""
engine.nlp
==========
Pure-Python NLP preprocessing layer.

IMPORTANT: This package performs NO diagnosis. It only converts free text into
the *names* of symptom / model-type Fact classes. All reasoning happens inside
Experta rules downstream.
"""

from engine.nlp.symptom_extractor import SymptomExtractor, ExtractionResult
from engine.nlp.intent_detector import detect_intent

__all__ = [
    "SymptomExtractor",
    "ExtractionResult",
    "detect_intent",
]
