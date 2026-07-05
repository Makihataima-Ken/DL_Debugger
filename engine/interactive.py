"""
interactive.py
==============
Pure session and diff helpers for interactive "what-if" diagnosis.

This module owns presentation-layer state only. All diagnosis still delegates
to ``DebuggingKnowledgeEngine.run_scenario()`` or ``run_text()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

from data.scenario_loader import BUILTIN_SCENARIOS, get_builtin_symptoms
from engine.knowledge_engine import (
    DebuggingKnowledgeEngine,
    DiagnosisResult,
    FACT_CLASS_REGISTRY,
)


DIFF_CONFIDENCE_EPSILON = 1e-3


ConfidenceChange = tuple[float | None, float | None]


@dataclass(frozen=True)
class DiagnosisDiff:
    """Structured comparison between two DiagnosisResult objects."""

    added_causes: list[str]
    removed_causes: list[str]
    added_recommendations: list[str]
    removed_recommendations: list[str]
    added_conflicts: list[str]
    removed_conflicts: list[str]
    confidence_added: dict[str, ConfidenceChange]
    confidence_removed: dict[str, ConfidenceChange]
    confidence_changed: dict[str, tuple[float, float]]

    @property
    def has_changes(self) -> bool:
        """Return True when any visible diagnosis field changed."""
        return any(
            (
                self.added_causes,
                self.removed_causes,
                self.added_recommendations,
                self.removed_recommendations,
                self.added_conflicts,
                self.removed_conflicts,
                self.confidence_added,
                self.confidence_removed,
                self.confidence_changed,
            )
        )


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ordered_unique(fact_names: list[str]) -> dict[str, None]:
    return {name: None for name in fact_names}


def _validate_fact_names(fact_names: tuple[str, ...]) -> None:
    unknown = [name for name in fact_names if name not in FACT_CLASS_REGISTRY]
    if not unknown:
        return

    suggestions: list[str] = []
    for name in unknown:
        matches = get_close_matches(name, FACT_CLASS_REGISTRY.keys(), n=3)
        if matches:
            suggestions.append(f"{name}: {', '.join(matches)}")

    message = f"Unknown fact name(s): {', '.join(unknown)}."
    if suggestions:
        message += f" Did you mean {', '.join(suggestions)}?"
    message += " Use --list to inspect built-in scenarios or check valid fact class names."
    raise ValueError(message)


def _conflict_ids(result: DiagnosisResult) -> set[str]:
    return {
        str(conflict["rule_id"])
        for conflict in result.conflicts
    }


def diff_diagnoses(
    before: DiagnosisResult,
    after: DiagnosisResult,
    epsilon: float = DIFF_CONFIDENCE_EPSILON,
) -> DiagnosisDiff:
    """Return a pure set/dict comparison of two diagnosis results."""
    before_causes = set(before.causes)
    after_causes = set(after.causes)
    before_recommendations = set(before.recommendations)
    after_recommendations = set(after.recommendations)
    before_conflicts = _conflict_ids(before)
    after_conflicts = _conflict_ids(after)
    before_confidence = before.confidence
    after_confidence = after.confidence

    confidence_added: dict[str, ConfidenceChange] = {}
    confidence_removed: dict[str, ConfidenceChange] = {}
    confidence_changed: dict[str, tuple[float, float]] = {}

    for fact in sorted(set(before_confidence) | set(after_confidence)):
        old = before_confidence.get(fact)
        new = after_confidence.get(fact)
        if old is None and new is not None:
            confidence_added[fact] = (None, new)
        elif old is not None and new is None:
            confidence_removed[fact] = (old, None)
        elif old is not None and new is not None and abs(new - old) > epsilon:
            confidence_changed[fact] = (old, new)

    return DiagnosisDiff(
        added_causes=sorted(after_causes - before_causes),
        removed_causes=sorted(before_causes - after_causes),
        added_recommendations=sorted(after_recommendations - before_recommendations),
        removed_recommendations=sorted(before_recommendations - after_recommendations),
        added_conflicts=sorted(after_conflicts - before_conflicts),
        removed_conflicts=sorted(before_conflicts - after_conflicts),
        confidence_added=confidence_added,
        confidence_removed=confidence_removed,
        confidence_changed=confidence_changed,
    )


class WhatIfSession:
    """Mutable what-if fact set that delegates all reasoning to the engine.

    A single ``DebuggingKnowledgeEngine`` instance is reused. This is safe
    because ``run_scenario()`` and ``run_text()`` reset the engine before each
    diagnosis run.
    """

    def __init__(
        self,
        engine: DebuggingKnowledgeEngine | None = None,
    ) -> None:
        self.engine = engine or DebuggingKnowledgeEngine()
        self._facts: dict[str, None] = {}
        self.root_confidence: dict[str, float] = {}
        self.extraction: Any | None = None
        self.last_result: DiagnosisResult | None = None

    def add(self, *fact_names: str) -> list[str]:
        """Add known fact names to the session; duplicates are no-ops."""
        _validate_fact_names(fact_names)
        for name in fact_names:
            self._facts.setdefault(name, None)
            self.root_confidence.setdefault(name, 1.0)
        self.extraction = None
        return self.current_facts()

    def remove(self, *fact_names: str) -> list[str]:
        """Remove known fact names from the session; absent facts are no-ops."""
        _validate_fact_names(fact_names)
        for name in fact_names:
            self._facts.pop(name, None)
            self.root_confidence.pop(name, None)
        self.extraction = None
        return self.current_facts()

    def set_from_text(self, text: str) -> Any | None:
        """Populate session facts from NLP extraction and keep provenance."""
        result = self.engine.run_text(text)
        extraction = result.extraction
        fact_names = extraction.all_fact_names if extraction is not None else []
        active_facts = set(fact_names)
        root_confidence: dict[str, float] = {}

        if extraction is not None:
            for evidence in extraction.evidence:
                if evidence.fact_name not in active_facts:
                    continue
                root_confidence[evidence.fact_name] = max(
                    root_confidence.get(evidence.fact_name, 0.0),
                    _clamp_confidence(evidence.confidence),
                )

        self._facts = _ordered_unique(fact_names)
        self.root_confidence = {
            name: root_confidence.get(name, 1.0)
            for name in fact_names
        }
        self.extraction = extraction
        return extraction

    def load_scenario(self, name: str) -> list[str]:
        """Populate session facts from a built-in scenario name."""
        try:
            fact_names = get_builtin_symptoms(name)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from exc

        self._facts = _ordered_unique(fact_names)
        self.root_confidence = {fact_name: 1.0 for fact_name in fact_names}
        self.extraction = None
        return self.current_facts()

    def clear(self) -> None:
        """Empty facts, NLP provenance, root confidence, and baseline result."""
        self._facts.clear()
        self.root_confidence.clear()
        self.extraction = None
        self.last_result = None

    reset = clear

    def current_facts(self) -> list[str]:
        """Return the current ordered fact-name list."""
        return list(self._facts)

    def diagnose(self) -> DiagnosisResult:
        """Run the current fact set through Experta and store the baseline."""
        fact_names = self.current_facts()
        root_confidence = {
            name: self.root_confidence.get(name, 1.0)
            for name in fact_names
        }
        result = self.engine.run_scenario(
            fact_names,
            root_confidence=root_confidence,
        )
        self.last_result = result
        return result

    def what_if(self, action: str, *fact_names: str) -> DiagnosisDiff:
        """Apply an add/remove change, rerun, and diff against the baseline."""
        before = self.last_result or self.diagnose()
        normalized = action.strip().lower()
        if normalized == "add":
            self.add(*fact_names)
        elif normalized == "remove":
            self.remove(*fact_names)
        else:
            raise ValueError("what-if action must be 'add' or 'remove'.")

        after = self.diagnose()
        return diff_diagnoses(before, after)

    def available_scenarios(self) -> list[str]:
        """Return sorted built-in scenario names for presentation layers."""
        return sorted(BUILTIN_SCENARIOS)
