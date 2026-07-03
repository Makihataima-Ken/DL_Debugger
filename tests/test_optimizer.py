"""
test_optimizer.py
=================
Tests for optimizer, numerical-stability, and optimizer-context Experta rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine


def run(facts: list[str]):
    return DebuggingKnowledgeEngine().run_scenario(facts)


OPT_CASES = [
    ("OPT_001", ["GradientExplosion"], "UseWeightClipping", "recommendations"),
    ("OPT_002", ["BadWeightInitialization"], "UseProperInitialization", "recommendations"),
    ("OPT_003", ["GradientVanishing"], "AddBatchNormalization", "recommendations"),
    ("OPT_004", ["TrainingLossHigh", "OscillatingLoss"], "UseWeightClipping", "recommendations"),
    ("OPT_005", ["SlowConvergence"], "AddBatchNormalization", "recommendations"),
    ("OPT_006", ["GradientExplosion", "GradientVanishing"], "BadWeightInitialization", "causes"),
    ("OPT_007", ["NaNLoss"], "NumericalInstability", "causes"),
    ("OPT_008", ["NumericalInstability"], "UseGradientClipping", "recommendations"),
    ("OPT_009", ["NaNLoss", "OscillatingLoss"], "MomentumTooHigh", "causes"),
    ("REC_017", ["BatchSizeTooLarge"], "ReduceBatchSize", "recommendations"),
    ("REC_018", ["BatchSizeTooSmall"], "IncreaseBatchSize", "recommendations"),
    ("REC_019", ["MomentumTooHigh"], "ReduceMomentum", "recommendations"),
    ("REC_020", ["WeightDecayTooHigh"], "ReduceWeightDecay", "recommendations"),
]


OPTZ_CASES = [
    ("OPTZ_001", ["OptimizerIsAdam", "CoupledWeightDecayUsed"], "WeightDecayCoupledWithAdam", "causes"),
    ("OPTZ_002", ["OptimizerIsAdam", "WeightDecayCoupledWithAdam"], "UseAdamW", "recommendations"),
    ("OPTZ_003", ["OptimizerIsSGD", "MissingMomentum"], "MomentumMisconfigured", "causes"),
    ("OPTZ_004", ["OptimizerIsSGD", "MomentumMisconfigured"], "EnableNesterovMomentum", "recommendations"),
    ("OPTZ_005", ["OptimizerIsAdam", "AdamHighLearningRateInstability"], "AdamLearningRateTooHigh", "causes"),
    ("OPTZ_006", ["OptimizerIsAdam", "AdamLearningRateTooHigh"], "UseLearningRateWarmup", "recommendations"),
]


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), OPT_CASES)
def test_optimizer_rule_fires(rule_id, facts, derived, bucket):
    result = run(facts)
    assert derived in getattr(result, bucket)
    assert rule_id in [e["rule_id"] for e in result.explanations]
    exp = next(e for e in result.explanations if e["rule_id"] == rule_id)
    assert 0.5 <= exp["confidence"] <= 0.95


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), OPTZ_CASES)
def test_optimizer_context_rule_fires(rule_id, facts, derived, bucket):
    result = run(facts)
    assert derived in getattr(result, bucket)
    assert rule_id in [e["rule_id"] for e in result.explanations]
    exp = next(e for e in result.explanations if e["rule_id"] == rule_id)
    assert 0.5 <= exp["confidence"] <= 0.95


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), OPTZ_CASES)
def test_optimizer_context_rules_require_context(rule_id, facts, derived, bucket):
    context = "OptimizerIsAdam" if "OptimizerIsAdam" in facts else "OptimizerIsSGD"
    result = run([fact for fact in facts if fact != context])
    assert derived not in getattr(result, bucket)
    assert rule_id not in [e["rule_id"] for e in result.explanations]
