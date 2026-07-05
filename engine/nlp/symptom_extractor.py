"""
symptom_extractor.py
====================
Top-level NLP entry point: free text -> Fact names.

The extractor orchestrates normalise -> entity map -> intent. It returns an
ExtractionResult that the knowledge engine consumes via run_text(). It NEVER
performs diagnosis; it only proposes which facts to assert.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.nlp.entity_mapper import map_entities, MappedEntity
from engine.nlp.intent_detector import detect_intent


@dataclass
class ExtractionResult:
    text: str
    intent: str
    symptom_facts: list[str] = field(default_factory=list)
    cause_facts: list[str] = field(default_factory=list)
    model_type_facts: list[str] = field(default_factory=list)
    context_facts: list[str] = field(default_factory=list)
    evidence: list[MappedEntity] = field(default_factory=list)
    negated: list[MappedEntity] = field(default_factory=list)

    @property
    def all_fact_names(self) -> list[str]:
        """Fact names to inject into the engine."""
        return (
            list(self.symptom_facts)
            + list(self.cause_facts)
            + list(self.model_type_facts)
            + list(self.context_facts)
        )

    @property
    def lexical_confidence(self) -> float:
        """Mean lexical confidence across matched evidence (0.0 if none)."""
        if not self.evidence:
            return 0.0
        return round(sum(e.confidence for e in self.evidence) / len(self.evidence), 3)


class SymptomExtractor:
    """Stateless façade over the NLP modules."""

    def extract(self, text: str) -> ExtractionResult:
        mapping = map_entities(text)
        evidence = (
            list(mapping.symptoms)
            + list(mapping.causes)
            + list(mapping.model_types)
            + list(mapping.contexts)
        )
        return ExtractionResult(
            text=text,
            intent=detect_intent(text),
            symptom_facts=mapping.symptom_names(),
            cause_facts=mapping.cause_names(),
            model_type_facts=mapping.model_type_names(),
            context_facts=mapping.context_names(),
            evidence=evidence,
            negated=mapping.negated,
        )
