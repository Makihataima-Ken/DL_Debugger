"""
test_validation.py
==================
Pytest tests for validation-phase rules.

All tests use real Experta inference – no mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from engine.knowledge_engine import DebuggingKnowledgeEngine


def run(symptoms: list[str]):
    engine = DebuggingKnowledgeEngine()
    return engine.run_scenario(symptoms)


# ---------------------------------------------------------------------------
# VAL_001 – overfitting detection
# ---------------------------------------------------------------------------

class TestOverfittingDetection:
    def test_overfitting_observed(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        assert "OverfittingObserved" in result.causes

    def test_model_too_complex_derived(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        assert "ModelTooComplex" in result.causes

    def test_insufficient_regularisation_derived(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        assert "InsufficientRegularization" in result.causes

    def test_add_dropout_recommended(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        assert "AddDropout" in result.recommendations

    def test_early_stopping_recommended(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        assert "UseEarlyStopping" in result.recommendations

    def test_explanation_val_001(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "VAL_001" in rule_ids

    def test_explanation_val_002(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "VAL_002" in rule_ids

    def test_explanation_val_003(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "VAL_003" in rule_ids


# ---------------------------------------------------------------------------
# VAL_004 – underfitting detection
# ---------------------------------------------------------------------------

class TestUnderfittingDetection:
    def test_underfitting_observed(self):
        result = run(["TrainingAccuracyLow", "ValidationAccuracyLow"])
        assert "UnderfittingObserved" in result.causes

    def test_model_too_simple_derived(self):
        result = run(["TrainingAccuracyLow", "ValidationAccuracyLow"])
        assert "ModelTooSimple" in result.causes

    def test_increase_complexity_recommended(self):
        result = run(["TrainingAccuracyLow", "ValidationAccuracyLow"])
        assert "IncreaseModelComplexity" in result.recommendations

    def test_explanation_val_004(self):
        result = run(["TrainingAccuracyLow", "ValidationAccuracyLow"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "VAL_004" in rule_ids


# ---------------------------------------------------------------------------
# VAL_006 – validation loss high + small dataset
# ---------------------------------------------------------------------------

class TestHighValLossSmallDataset:
    def test_insufficient_reg_derived(self):
        result = run(["ValidationLossHigh", "SmallDataset"])
        assert "InsufficientRegularization" in result.causes

    def test_dropout_recommended(self):
        result = run(["ValidationLossHigh", "SmallDataset"])
        assert "AddDropout" in result.recommendations

    def test_early_stopping_recommended(self):
        result = run(["ValidationLossHigh", "SmallDataset"])
        assert "UseEarlyStopping" in result.recommendations

    def test_explanation_val_006(self):
        result = run(["ValidationLossHigh", "SmallDataset"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "VAL_006" in rule_ids


# ---------------------------------------------------------------------------
# Overfitting + small dataset
# ---------------------------------------------------------------------------

class TestOverfittingSmallDataset:
    def test_data_augmentation_recommended(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow", "SmallDataset"])
        assert "ApplyDataAugmentation" in result.recommendations

    def test_increase_dataset_recommended(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow", "SmallDataset"])
        assert "IncreaseDatasetSize" in result.recommendations

    def test_explanation_arch_003(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow", "SmallDataset"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "ARCH_003" in rule_ids


# ---------------------------------------------------------------------------
# Fact deduplication – no duplicate causes or recommendations
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_no_duplicate_causes(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        assert len(result.causes) == len(set(result.causes))

    def test_no_duplicate_recommendations(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        assert len(result.recommendations) == len(set(result.recommendations))


# ---------------------------------------------------------------------------
# Symptom reporting
# ---------------------------------------------------------------------------

class TestSymptomReporting:
    def test_symptoms_in_result(self):
        result = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
        assert "TrainingAccuracyHigh" in result.symptoms
        assert "ValidationAccuracyLow" in result.symptoms

    def test_symptoms_sorted(self):
        result = run(["ValidationAccuracyLow", "TrainingAccuracyHigh"])
        assert result.symptoms == sorted(result.symptoms)
