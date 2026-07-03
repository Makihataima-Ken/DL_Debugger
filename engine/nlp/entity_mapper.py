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
    CONTEXT_PHRASES,
)


NEGATION_SCOPE_TOKENS = 5
NEGATION_CUES: tuple[tuple[str, ...], ...] = (
    ("no", "sign", "of"),
    ("free", "of"),
    ("rather", "than"),
    ("no",),
    ("not",),
    ("without",),
)
PROTECTED_NEGATION_TOKENS = {"not"}
HEDGE_CUES: tuple[tuple[str, ...], ...] = (
    ("might", "be"),
    ("maybe",),
    ("possibly",),
    ("seems", "like"),
    ("seem", "like"),
    ("could", "be"),
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
    contexts: list[MappedEntity] = field(default_factory=list)
    negated: list[MappedEntity] = field(default_factory=list)

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

    def context_names(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.contexts:
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


def _token_matches(phrase_token: str, text_token: str) -> bool:
    """Match exact tokens, with stem-prefix support for long vocabulary stems."""
    if phrase_token == text_token:
        return True
    return len(phrase_token) >= 5 and text_token.startswith(phrase_token)


def _find_spans(tokens: list[str], phrase_tokens: list[str]) -> list[tuple[int, int]]:
    if not phrase_tokens or len(phrase_tokens) > len(tokens):
        return []

    spans: list[tuple[int, int]] = []
    width = len(phrase_tokens)
    for start in range(0, len(tokens) - width + 1):
        window = tokens[start:start + width]
        if all(
            _token_matches(phrase_token, text_token)
            for phrase_token, text_token in zip(phrase_tokens, window)
        ):
            spans.append((start, start + width))
    return spans


def _cue_positions(tokens: list[str], cues: tuple[tuple[str, ...], ...]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for cue in cues:
        width = len(cue)
        if width > len(tokens):
            continue
        for start in range(0, len(tokens) - width + 1):
            if tuple(tokens[start:start + width]) == cue:
                positions.append((start, start + width))
    return positions


def _protected_from_negation(phrase_tokens: list[str]) -> bool:
    return any(token in PROTECTED_NEGATION_TOKENS for token in phrase_tokens)


def _is_negated(
    start: int,
    phrase_tokens: list[str],
    negation_positions: list[tuple[int, int]],
) -> bool:
    if _protected_from_negation(phrase_tokens):
        return False

    for cue_start, cue_end in negation_positions:
        if cue_end <= start <= cue_end + NEGATION_SCOPE_TOKENS:
            return True
    return False


def _is_hedged(start: int, hedge_positions: list[tuple[int, int]]) -> bool:
    for _cue_start, cue_end in hedge_positions:
        if cue_end <= start <= cue_end + NEGATION_SCOPE_TOKENS:
            return True
    return False


def _confidence_for_match(
    phrase: str,
    start: int,
    hedge_positions: list[tuple[int, int]],
) -> float:
    confidence = _phrase_confidence(phrase)
    if _is_hedged(start, hedge_positions):
        confidence *= 0.75
    return round(confidence, 3)


def _phrase_tokens(phrase: str) -> list[str]:
    return normalize(phrase).split()


def _collect_matches(
    text_tokens: list[str],
    phrase_map: dict[str, str],
    negation_positions: list[tuple[int, int]],
    hedge_positions: list[tuple[int, int]],
) -> tuple[list[MappedEntity], list[MappedEntity]]:
    matches: list[MappedEntity] = []
    negated: list[MappedEntity] = []

    for phrase, fact_name in phrase_map.items():
        tokens = _phrase_tokens(phrase)
        for start, _end in _find_spans(text_tokens, tokens):
            entity = MappedEntity(
                fact_name=fact_name,
                phrase=phrase,
                confidence=_confidence_for_match(phrase, start, hedge_positions),
            )
            if _is_negated(start, tokens, negation_positions):
                negated.append(entity)
            else:
                matches.append(entity)

    return matches, negated


def map_entities(text: str) -> MappingResult:
    """Return all symptom and model-type facts mentioned in *text*."""
    tokens = normalize(text).split()
    negation_positions = _cue_positions(tokens, NEGATION_CUES)
    hedge_positions = _cue_positions(tokens, HEDGE_CUES)
    result = MappingResult()

    symptoms, negated_symptoms = _collect_matches(
        tokens,
        SYMPTOM_PHRASES,
        negation_positions,
        hedge_positions,
    )
    model_types, negated_model_types = _collect_matches(
        tokens,
        MODEL_TYPE_PHRASES,
        negation_positions,
        hedge_positions,
    )
    contexts, negated_contexts = _collect_matches(
        tokens,
        CONTEXT_PHRASES,
        negation_positions,
        hedge_positions,
    )

    result.symptoms.extend(symptoms)
    result.model_types.extend(model_types)
    result.contexts.extend(contexts)
    result.negated.extend(negated_symptoms)
    result.negated.extend(negated_model_types)
    result.negated.extend(negated_contexts)

    return result
