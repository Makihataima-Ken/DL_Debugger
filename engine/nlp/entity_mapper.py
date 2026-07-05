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
from functools import lru_cache
from typing import Literal

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from engine.nlp.terminology_normalizer import expand_terms
from engine.nlp.training_debugging_vocabulary import (
    FactNameSpec,
    SYMPTOM_PHRASES,
    MODEL_TYPE_PHRASES,
    CONTEXT_PHRASES,
)


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
CLAUSE_BOUNDARY_PUNCT = {",", ";", ":", ".", "!", "?"}
CLAUSE_BOUNDARY_WORDS = {
    "although",
    "but",
    "however",
    "though",
    "whereas",
    "yet",
}
DOMAIN_LEMMA_OVERRIDES = {
    "oscillated": "oscillate",
    "oscillates": "oscillate",
    "oscillating": "oscillate",
    "overfitting": "overfit",
    "underfitting": "underfit",
}

EntityCategory = Literal["symptoms", "model_types", "contexts"]


@dataclass(frozen=True)
class MatchMetadata:
    category: EntityCategory
    phrase: str
    fact_names: tuple[str, ...]
    phrase_order: int


@Language.component("training_debug_lemma_overrides")
def _training_debug_lemma_overrides(doc: Doc) -> Doc:
    for token in doc:
        lemma = DOMAIN_LEMMA_OVERRIDES.get(token.lower_)
        if lemma is not None:
            token.lemma_ = lemma
    return doc


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


def _load_nlp() -> Language:
    """Load the deterministic NLP pipeline used for lexical extraction."""
    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    if "training_debug_lemma_overrides" not in nlp.pipe_names:
        nlp.add_pipe("training_debug_lemma_overrides", after="lemmatizer")
    return nlp


def _fact_names(spec: FactNameSpec) -> tuple[str, ...]:
    if isinstance(spec, str):
        return (spec,)
    return tuple(spec)


def _vocabulary_sources() -> tuple[tuple[EntityCategory, dict[str, FactNameSpec]], ...]:
    return (
        ("symptoms", SYMPTOM_PHRASES),
        ("model_types", MODEL_TYPE_PHRASES),
        ("contexts", CONTEXT_PHRASES),
    )


def _validate_fact_names(phrase: str, fact_names: tuple[str, ...]) -> None:
    from engine.knowledge_engine import FACT_CLASS_REGISTRY

    unknown = [name for name in fact_names if name not in FACT_CLASS_REGISTRY]
    if unknown:
        raise ValueError(
            f"NLP vocabulary phrase '{phrase}' references unknown Fact class "
            f"name(s): {', '.join(unknown)}"
        )


@lru_cache(maxsize=1)
def _matcher_components() -> tuple[Language, PhraseMatcher, dict[int, MatchMetadata]]:
    nlp = _load_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA", validate=True)
    metadata: dict[int, MatchMetadata] = {}

    for category, phrase_map in _vocabulary_sources():
        for phrase_order, (phrase, spec) in enumerate(phrase_map.items()):
            fact_names = _fact_names(spec)
            _validate_fact_names(phrase, fact_names)
            pattern = nlp(expand_terms(phrase))
            if len(pattern) == 0:
                continue

            match_key = f"{category}:{phrase_order}"
            matcher.add(match_key, [pattern])
            metadata[nlp.vocab.strings[match_key]] = MatchMetadata(
                category=category,
                phrase=phrase,
                fact_names=fact_names,
                phrase_order=phrase_order,
            )

    return nlp, matcher, metadata


def _cue_spans(doc: Doc, cues: tuple[tuple[str, ...], ...]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    lowered = [token.lower_ for token in doc]

    for cue in cues:
        width = len(cue)
        if width > len(lowered):
            continue
        for start in range(0, len(lowered) - width + 1):
            if tuple(lowered[start:start + width]) == cue:
                positions.append((start, start + width))

    return positions


def _same_sentence(doc: Doc, left: int, right: int) -> bool:
    return doc[left].sent.start == doc[right].sent.start


def _has_clause_boundary(doc: Doc, start: int, end: int) -> bool:
    for token in doc[start:end]:
        if token.text in CLAUSE_BOUNDARY_PUNCT:
            return True
        if token.lower_ in CLAUSE_BOUNDARY_WORDS:
            return True
    return False


def _protected_from_negation(span: Span) -> bool:
    return any(token.lower_ in PROTECTED_NEGATION_TOKENS for token in span)


def _direct_dependency_negates(span: Span) -> bool:
    for token in span:
        for child in token.children:
            if child.lower_ == "not" and child.dep_ == "neg":
                return True
            if child.lower_ == "no" and child.dep_ in {"det", "advmod", "neg"}:
                return True
    return False


def _cue_applies_to_span(
    doc: Doc,
    span: Span,
    cue_positions: list[tuple[int, int]],
) -> bool:
    for cue_start, cue_end in cue_positions:
        if cue_end > span.start:
            continue
        if not _same_sentence(doc, cue_start, span.start):
            continue
        if _has_clause_boundary(doc, cue_end, span.start):
            continue
        return True
    return False


def _is_negated(span: Span, negation_positions: list[tuple[int, int]]) -> bool:
    if _protected_from_negation(span):
        return False
    return _direct_dependency_negates(span) or _cue_applies_to_span(
        span.doc,
        span,
        negation_positions,
    )


def _is_hedged(span: Span, hedge_positions: list[tuple[int, int]]) -> bool:
    return _cue_applies_to_span(span.doc, span, hedge_positions)


def _confidence_for_match(
    phrase: str,
    span: Span,
    hedge_positions: list[tuple[int, int]],
) -> float:
    confidence = _phrase_confidence(phrase)
    if _is_hedged(span, hedge_positions):
        confidence *= 0.75
    return round(confidence, 3)


def _append_entity(result: MappingResult, category: EntityCategory, entity: MappedEntity) -> None:
    if category == "symptoms":
        result.symptoms.append(entity)
    elif category == "model_types":
        result.model_types.append(entity)
    else:
        result.contexts.append(entity)


def map_entities(text: str) -> MappingResult:
    """Return all symptom and model-type facts mentioned in *text*."""
    nlp, matcher, metadata = _matcher_components()
    doc = nlp(expand_terms(text))
    negation_positions = _cue_spans(doc, NEGATION_CUES)
    hedge_positions = _cue_spans(doc, HEDGE_CUES)
    result = MappingResult()

    matches = sorted(
        matcher(doc),
        key=lambda match: (
            match[1],
            -(match[2] - match[1]),
            metadata[match[0]].phrase_order,
        ),
    )
    for match_id, start, end in matches:
        match = metadata[match_id]
        span = doc[start:end]
        negated = _is_negated(span, negation_positions)

        for fact_name in match.fact_names:
            entity = MappedEntity(
                fact_name=fact_name,
                phrase=match.phrase,
                confidence=_confidence_for_match(
                    match.phrase,
                    span,
                    hedge_positions,
                ),
            )
            if negated:
                result.negated.append(entity)
            else:
                _append_entity(result, match.category, entity)

    return result
