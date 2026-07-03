"""
test_confidence_model.py
========================
Tests for reporting-only confidence propagation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine, FACT_CLASS_REGISTRY
from models.facts import Explanation


def run(facts: list[str]):
    return DebuggingKnowledgeEngine().run_scenario(facts)


def run_declared(
    facts: list[str],
    explanations: list[Explanation],
    root_confidence: dict[str, float] | None = None,
):
    engine = DebuggingKnowledgeEngine()
    engine.reset()
    engine._injected_symptoms = set()
    engine._root_confidence_by_fact = root_confidence or {}
    for name in facts:
        engine.declare(FACT_CLASS_REGISTRY[name]())
    for explanation in explanations:
        engine.declare(explanation)
    engine.run()
    return engine.get_diagnosis()


def test_antecedent_propagation_discounts_multihop_cause():
    result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])

    assert "OverfittingObserved" in result.confidence
    assert "ModelTooComplex" in result.confidence
    assert 0.0 < result.confidence["ModelTooComplex"]
    assert result.confidence["ModelTooComplex"] < result.confidence["OverfittingObserved"]


def test_nlp_lexical_confidence_sets_root_confidence():
    baseline = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
    hedged = DebuggingKnowledgeEngine().run_text(
        "maybe high training accuracy and maybe validation accuracy is much lower"
    )

    assert hedged.extraction is not None
    assert "TrainingAccuracyHigh" in hedged.extraction.symptom_facts
    assert "ValidationAccuracyLow" in hedged.extraction.symptom_facts
    assert hedged.confidence["OverfittingObserved"] < baseline.confidence["OverfittingObserved"]


def test_multiple_supporting_rules_raise_confidence_with_noisy_or():
    first = Explanation(
        rule_id="SYN_CONF_001",
        triggered_by="SyntheticEvidenceA",
        derived="LearningRateTooHigh",
        explanation="Synthetic support for high learning rate.",
        confidence=0.5,
    )
    second = Explanation(
        rule_id="SYN_CONF_002",
        triggered_by="SyntheticEvidenceB",
        derived="LearningRateTooHigh",
        explanation="Independent synthetic support for high learning rate.",
        confidence=0.5,
    )

    single = run_declared(["LearningRateTooHigh"], [first])
    combined = run_declared(["LearningRateTooHigh"], [first, second])

    assert combined.confidence["LearningRateTooHigh"] > single.confidence["LearningRateTooHigh"]
    assert combined.confidence["LearningRateTooHigh"] == pytest.approx(0.75)


def test_suppressed_causes_and_their_filtered_recommendations_leave_confidence():
    result = run(["TrainingLossHigh", "OscillatingLoss", "SlowConvergence"])

    assert "LearningRateTooHigh" in result.causes
    assert "LearningRateTooLow" not in result.causes
    assert "LearningRateTooLow" not in result.confidence
    assert "IncreaseLearningRate" not in result.recommendations
    assert "IncreaseLearningRate" not in result.confidence


def test_cyclic_confidence_graph_terminates_with_clamped_values():
    result = run_declared(
        ["DistributionShift", "PoorGeneralization"],
        [
            Explanation(
                rule_id="SYN_CYCLE_001",
                triggered_by="DistributionShift",
                derived="PoorGeneralization",
                explanation="Synthetic cyclic support.",
                confidence=0.8,
            ),
            Explanation(
                rule_id="SYN_CYCLE_002",
                triggered_by="PoorGeneralization",
                derived="DistributionShift",
                explanation="Synthetic cyclic support.",
                confidence=0.8,
            ),
        ],
        root_confidence={"DistributionShift": 1.0},
    )

    assert "DistributionShift" in result.confidence
    assert "PoorGeneralization" in result.confidence
    assert all(0.0 <= value <= 1.0 for value in result.confidence.values())


def test_all_reported_confidences_remain_clamped():
    result = run(
        [
            "TrainingLossHigh",
            "OscillatingLoss",
            "SlowConvergence",
            "NaNLoss",
            "OptimizerIsSGD",
            "MissingMomentum",
        ]
    )

    assert result.confidence
    assert all(0.0 <= value <= 1.0 for value in result.confidence.values())
