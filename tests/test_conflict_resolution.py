"""
test_conflict_resolution.py
===========================
Tests for low-salience cause-conflict resolution meta-rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine, FACT_CLASS_REGISTRY
from models.facts import Explanation


def run(facts: list[str]):
    engine = DebuggingKnowledgeEngine()
    result = engine.run_scenario(facts)
    return engine, result


def run_declared(facts: list[str], explanations: list[Explanation] | None = None):
    engine = DebuggingKnowledgeEngine()
    engine.reset()
    engine._injected_symptoms = set()
    for name in facts:
        engine.declare(FACT_CLASS_REGISTRY[name]())
    for explanation in explanations or []:
        engine.declare(explanation)
    engine.run()
    return engine, engine.get_diagnosis()


def fact_exists(engine, fact_name: str, **fields) -> bool:
    for fact in engine.facts.values():
        if type(fact).__name__ != fact_name:
            continue
        if all(fact.get(key) == value for key, value in fields.items()):
            return True
    return False


CONFLICT_CASES = [
    {
        "rule_id": "CONFLICT_001",
        "facts": ["TrainingAccuracyHigh", "ValidationAccuracyLow", "TrainingAccuracyLow"],
        "winner": "OverfittingObserved",
        "losers": ["UnderfittingObserved"],
        "present_recs": ["ReduceModelComplexity"],
        "absent_recs": ["IncreaseModelComplexity"],
    },
    {
        "rule_id": "CONFLICT_002",
        "facts": ["TrainingLossHigh", "OscillatingLoss", "SlowConvergence"],
        "winner": "LearningRateTooHigh",
        "losers": ["LearningRateTooLow"],
        "present_recs": ["ReduceLearningRate"],
        "absent_recs": ["IncreaseLearningRate"],
    },
    {
        "rule_id": "CONFLICT_003",
        "facts": ["TrainingLossHigh", "OscillatingLoss", "SlowConvergence", "UsesDistributedTraining"],
        "winner": "LearningRateTooHigh",
        "losers": ["LearningRateNotScaled"],
        "present_recs": ["ReduceLearningRate"],
        "absent_recs": ["ScaleLearningRateByWorldSize"],
    },
    {
        "rule_id": "CONFLICT_004",
        "facts": ["TrainingLossHigh", "SlowConvergence", "OptimizerIsAdam", "AdamHighLearningRateInstability"],
        "winner": "AdamLearningRateTooHigh",
        "losers": ["LearningRateTooLow"],
        "present_recs": ["UseLearningRateWarmup"],
        "absent_recs": ["IncreaseLearningRate"],
    },
    {
        "rule_id": "CONFLICT_005",
        "facts": ["UsesDistributedTraining", "SlowConvergence", "OptimizerIsAdam", "AdamHighLearningRateInstability"],
        "winner": "AdamLearningRateTooHigh",
        "losers": ["LearningRateNotScaled"],
        "present_recs": ["UseLearningRateWarmup"],
        "absent_recs": ["ScaleLearningRateByWorldSize"],
    },
    {
        "rule_id": "CONFLICT_006",
        "facts": ["TrainingAccuracyHigh", "ValidationAccuracyLow", "TrainingAccuracyLow"],
        "winner": "ModelTooComplex",
        "losers": ["ModelTooSimple"],
        "present_recs": ["ReduceModelComplexity"],
        "absent_recs": ["IncreaseModelComplexity"],
    },
    {
        "rule_id": "CONFLICT_007",
        "facts": ["ValidationLossHigh", "SmallDataset", "DataLeakageSuspected", "ValidationAccuracyLow"],
        "winner": "ExcessiveRegularization",
        "losers": ["InsufficientRegularization"],
        "present_recs": ["RemoveDropout", "ReduceRegularization"],
        "absent_recs": ["AddDropout", "UseEarlyStopping"],
    },
    {
        "rule_id": "CONFLICT_009",
        "facts": ["NaNLoss", "OscillatingLoss", "OptimizerIsSGD", "MissingMomentum"],
        "winner": "MomentumMisconfigured",
        "losers": ["MomentumTooHigh"],
        "present_recs": ["EnableNesterovMomentum"],
        "absent_recs": ["ReduceMomentum"],
    },
]


@pytest.mark.parametrize("case", CONFLICT_CASES)
def test_conflicting_derived_causes_are_resolved(case):
    engine, result = run(case["facts"])
    conflict = next(
        conflict for conflict in result.conflicts
        if conflict["rule_id"] == case["rule_id"]
    )

    assert case["winner"] in result.causes
    assert conflict["winner"] == case["winner"]
    for loser in case["losers"]:
        assert loser not in result.causes
        assert loser in conflict["losers"]
        assert fact_exists(engine, "SuppressedCause", cause=loser)

    assert fact_exists(engine, "CauseConflict", rule_id=case["rule_id"])
    assert case["rule_id"] in [e["rule_id"] for e in result.explanations]
    exp = next(e for e in result.explanations if e["rule_id"] == case["rule_id"])
    assert 0.5 <= exp["confidence"] <= 0.95

    for recommendation in case["present_recs"]:
        assert recommendation in result.recommendations
    for recommendation in case["absent_recs"]:
        assert recommendation not in result.recommendations


def test_batch_size_conflict_resolves_directly_declared_orphaned_causes():
    engine, result = run_declared(["BatchSizeTooLarge", "BatchSizeTooSmall"])

    assert "BatchSizeTooLarge" in result.causes
    assert "BatchSizeTooSmall" not in result.causes
    assert "ReduceBatchSize" in result.recommendations
    assert "IncreaseBatchSize" not in result.recommendations
    assert fact_exists(engine, "SuppressedCause", cause="BatchSizeTooSmall")
    assert result.conflicts[0]["rule_id"] == "CONFLICT_008"
    assert "lexicographic tie-break" in result.conflicts[0]["reason"]


def test_equal_strength_tie_uses_support_count_before_name():
    explanations = [
        Explanation(
            rule_id="TEST_HIGH_A",
            triggered_by="SyntheticEvidence",
            derived="LearningRateTooHigh",
            explanation="Synthetic high-LR support.",
            confidence=0.5,
        ),
        Explanation(
            rule_id="TEST_HIGH_B",
            triggered_by="SyntheticEvidence",
            derived="LearningRateTooHigh",
            explanation="Second synthetic high-LR support.",
            confidence=0.5,
        ),
        Explanation(
            rule_id="TEST_LOW_A",
            triggered_by="SyntheticEvidence",
            derived="LearningRateTooLow",
            explanation="Synthetic low-LR support.",
            confidence=0.75,
        ),
    ]
    _, result = run_declared(
        ["LearningRateTooHigh", "LearningRateTooLow"],
        explanations=explanations,
    )

    conflict = result.conflicts[0]
    assert conflict["winner"] == "LearningRateTooHigh"
    assert "more supporting explanations" in conflict["reason"]


def test_consistent_overfitting_output_is_unchanged():
    _, result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])

    assert result.symptoms == ["TrainingAccuracyHigh", "ValidationAccuracyLow"]
    assert result.causes == [
        "InsufficientRegularization",
        "ModelTooComplex",
        "OverfittingObserved",
    ]
    assert result.recommendations == [
        "AddDropout",
        "IncreaseDatasetSize",
        "ReduceModelComplexity",
        "UseEarlyStopping",
    ]
    rule_ids = [e["rule_id"] for e in result.explanations]
    assert not any(rule_id.startswith("CONFLICT_") for rule_id in rule_ids)
    assert result.conflicts == []


def test_consistent_oscillating_loss_output_is_unchanged():
    _, result = run(["TrainingLossHigh", "OscillatingLoss"])

    assert result.symptoms == ["OscillatingLoss", "TrainingLossHigh"]
    assert result.causes == ["LearningRateTooHigh"]
    assert result.recommendations == ["ReduceLearningRate", "UseWeightClipping"]
    assert [e["rule_id"] for e in result.explanations] == [
        "OPT_004",
        "REC_001",
        "TRAIN_001",
    ]
    assert result.conflicts == []
