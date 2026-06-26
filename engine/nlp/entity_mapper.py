"""
entity_mapper.py
================
Maps normalised text to Fact class *names* using the controlled vocabulary.

This module performs lexical matching only. It deliberately produces no
diagnosis: it just answers "which symptom/model-type facts does this text
mention, and with what lexical confidence".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.nlp.terminology_normalizer import normalize
from engine.nlp.training_debugging_vocabulary import (
    SYMPTOM_PHRASES,
    MODEL_TYPE_PHRASES,
)


@dataclass
class MappedEntity:
    fact_name: str
    phrase: str
    confidence: float


@dataclass
class MappingResult:
    symptoms: list[MappedEntity] = field(default_factory=list)
    model_types: list[MappedEntity] = field(default_factory=list)

    def symptom_names(self) -> list[str]:
        # de-duplicate while preserving first occurrence
        seen: dict[str, None] = {}
        for e in self.symptoms:
            seen.setdefault(e.fact_name, None)
        return list(seen.keys())

    def model_type_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.model_types:
            seen.setdefault(e.fact_name, None)
        return list(seen.keys())


def _phrase_confidence(phrase: str) -> float:
    """Longer / more specific phrases earn higher lexical confidence."""
    words = len(phrase.split())
    if words >= 3:
        return 0.95
    if words == 2:
        return 0.9
    return 0.8


def map_entities(text: str) -> MappingResult:
    """Return all symptom and model-type facts mentioned in *text*."""
    norm = " " + normalize(text) + " "
    result = MappingResult()

    for phrase, fact_name in SYMPTOM_PHRASES.items():
        if phrase in norm:
            result.symptoms.append(
                MappedEntity(fact_name, phrase, _phrase_confidence(phrase))
            )

    for phrase, fact_name in MODEL_TYPE_PHRASES.items():
        if phrase in norm:
            result.model_types.append(
                MappedEntity(fact_name, phrase, _phrase_confidence(phrase))
            )

    return result
