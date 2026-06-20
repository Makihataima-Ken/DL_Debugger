"""
test_training.py
================
Pytest tests for training-phase rules.

All tests use real Experta inference – no mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from engine.knowledge_engine import DebuggingKnowledgeEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(symptoms: list[str]):
    """Short-hand: create engine, run scenario, return DiagnosisResult."""
    engine = DebuggingKnowledgeEngine()
    return engine.run_scenario(symptoms)


# ---------------------------------------------------------------------------
# TRAIN_001 – oscillating high loss → LR too high
# ---------------------------------------------------------------------------

class TestTrainingRuleOscillatingLoss:
    def test_cause_derived(self):
        result = run(["TrainingLossHigh", "OscillatingLoss"])
        assert "LearningRateTooHigh" in result.causes

    def test_recommendation_derived(self):
        result = run(["TrainingLossHigh", "OscillatingLoss"])
        assert "ReduceLearningRate" in result.recommendations

    def test_explanation_present(self):
        result = run(["TrainingLossHigh", "OscillatingLoss"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TRAIN_001" in rule_ids

    def test_no_false_low_lr(self):
        """Oscillating + high loss should NOT assert LR too low."""
        result = run(["TrainingLossHigh", "OscillatingLoss"])
        assert "LearningRateTooLow" not in result.causes


# ---------------------------------------------------------------------------
# TRAIN_002 – slow convergence + high loss → LR too low
# ---------------------------------------------------------------------------

class TestTrainingRuleSlowConvergence:
    def test_cause_derived(self):
        result = run(["TrainingLossHigh", "SlowConvergence"])
        assert "LearningRateTooLow" in result.causes

    def test_recommendation_derived(self):
        result = run(["TrainingLossHigh", "SlowConvergence"])
        assert "IncreaseLearningRate" in result.recommendations

    def test_explanation_present(self):
        result = run(["TrainingLossHigh", "SlowConvergence"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TRAIN_002" in rule_ids


# ---------------------------------------------------------------------------
# TRAIN_003 / TRAIN_004 – gradient pathologies → bad init
# ---------------------------------------------------------------------------

class TestGradientExplosion:
    def test_cause_bad_init(self):
        result = run(["GradientExplosion"])
        assert "BadWeightInitialization" in result.causes

    def test_recommendation_proper_init(self):
        result = run(["GradientExplosion"])
        assert "UseProperInitialization" in result.recommendations

    def test_recommendation_weight_clipping(self):
        result = run(["GradientExplosion"])
        assert "UseWeightClipping" in result.recommendations

    def test_explanation_train_003(self):
        result = run(["GradientExplosion"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TRAIN_003" in rule_ids


class TestGradientVanishing:
    def test_cause_bad_init(self):
        result = run(["GradientVanishing"])
        assert "BadWeightInitialization" in result.causes

    def test_recommendation_batch_norm(self):
        result = run(["GradientVanishing"])
        assert "AddBatchNormalization" in result.recommendations

    def test_explanation_train_004(self):
        result = run(["GradientVanishing"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TRAIN_004" in rule_ids


# ---------------------------------------------------------------------------
# TRAIN_005 – noisy labels → poor data quality
# ---------------------------------------------------------------------------

class TestNoisyLabels:
    def test_cause_derived(self):
        result = run(["NoisyLabels"])
        assert "PoorDataQuality" in result.causes

    def test_recommendation_improve_labels(self):
        result = run(["NoisyLabels"])
        assert "ImproveLabelQuality" in result.recommendations

    def test_explanation_present(self):
        result = run(["NoisyLabels"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TRAIN_005" in rule_ids


# ---------------------------------------------------------------------------
# TRAIN_006 – class imbalance → DataImbalance cause
# ---------------------------------------------------------------------------

class TestClassImbalance:
    def test_cause_derived(self):
        result = run(["ClassImbalanceDetected"])
        assert "DataImbalance" in result.causes

    def test_recommendation_rebalance(self):
        result = run(["ClassImbalanceDetected"])
        assert "RebalanceClasses" in result.recommendations

    def test_recommendation_weighted_loss(self):
        result = run(["ClassImbalanceDetected"])
        assert "UseWeightedLoss" in result.recommendations


# ---------------------------------------------------------------------------
# TRAIN_007 / TRAIN_008 – underfitting detection
# ---------------------------------------------------------------------------

class TestUnderfitting:
    def test_small_dataset_underfitting(self):
        result = run(["TrainingAccuracyLow", "SmallDataset"])
        assert "UnderfittingObserved" in result.causes

    def test_model_too_simple_without_small_data(self):
        result = run(["TrainingAccuracyLow"])
        assert "ModelTooSimple" in result.causes

    def test_increase_model_complexity_recommended(self):
        result = run(["TrainingAccuracyLow"])
        assert "IncreaseModelComplexity" in result.recommendations


# ---------------------------------------------------------------------------
# TRAIN_012 – slow convergence alone → LR too low
# ---------------------------------------------------------------------------

class TestSlowConvergenceAlone:
    def test_cause_derived(self):
        result = run(["SlowConvergence"])
        assert "LearningRateTooLow" in result.causes

    def test_recommendation_increase_lr(self):
        result = run(["SlowConvergence"])
        assert "IncreaseLearningRate" in result.recommendations


# ---------------------------------------------------------------------------
# Explanation structure
# ---------------------------------------------------------------------------

class TestExplanationStructure:
    def test_explanation_has_all_fields(self):
        result = run(["TrainingLossHigh", "OscillatingLoss"])
        for exp in result.explanations:
            assert "rule_id" in exp
            assert "triggered_by" in exp
            assert "derived" in exp
            assert "explanation" in exp
            assert len(exp["explanation"]) > 10

    def test_explanations_are_sorted_by_rule_id(self):
        result = run(["GradientExplosion", "TrainingLossHigh", "OscillatingLoss"])
        ids = [e["rule_id"] for e in result.explanations]
        assert ids == sorted(ids)
