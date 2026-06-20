"""
test_testing.py
===============
Pytest tests for testing-phase rules (distribution shift, poor generalisation).

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
# TEST_001 / TEST_002 – distribution shift scenario
# ---------------------------------------------------------------------------

class TestDistributionShift:
    def test_cause_derived(self):
        result = run(["ValidationAccuracyHigh", "TestAccuracyLow"])
        assert "DistributionShift" in result.causes

    def test_poor_generalisation_derived(self):
        result = run(["ValidationAccuracyHigh", "TestAccuracyLow"])
        assert "PoorGeneralization" in result.causes

    def test_collect_domain_data_recommended(self):
        result = run(["ValidationAccuracyHigh", "TestAccuracyLow"])
        assert "CollectDomainData" in result.recommendations

    def test_augmentation_recommended(self):
        result = run(["ValidationAccuracyHigh", "TestAccuracyLow"])
        assert "ApplyDataAugmentation" in result.recommendations

    def test_inspect_pipeline_recommended(self):
        result = run(["ValidationAccuracyHigh", "TestAccuracyLow"])
        assert "InspectDataPipeline" in result.recommendations

    def test_explanation_test_001(self):
        result = run(["ValidationAccuracyHigh", "TestAccuracyLow"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TEST_001" in rule_ids

    def test_explanation_test_002(self):
        result = run(["ValidationAccuracyHigh", "TestAccuracyLow"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TEST_002" in rule_ids


# ---------------------------------------------------------------------------
# TEST_003 – data leakage + low test accuracy
# ---------------------------------------------------------------------------

class TestDataLeakage:
    def test_distribution_shift_derived(self):
        result = run(["DataLeakageSuspected", "TestAccuracyLow"])
        assert "DistributionShift" in result.causes

    def test_inspect_pipeline_recommended(self):
        result = run(["DataLeakageSuspected", "TestAccuracyLow"])
        assert "InspectDataPipeline" in result.recommendations

    def test_explanation_test_003(self):
        result = run(["DataLeakageSuspected", "TestAccuracyLow"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TEST_003" in rule_ids


# ---------------------------------------------------------------------------
# TEST_004 – class imbalance + test accuracy low
# ---------------------------------------------------------------------------

class TestClassImbalanceTest:
    def test_data_imbalance_derived(self):
        result = run(["ClassImbalanceDetected", "TestAccuracyLow"])
        assert "DataImbalance" in result.causes

    def test_rebalance_recommended(self):
        result = run(["ClassImbalanceDetected", "TestAccuracyLow"])
        assert "RebalanceClasses" in result.recommendations

    def test_weighted_loss_recommended(self):
        result = run(["ClassImbalanceDetected", "TestAccuracyLow"])
        assert "UseWeightedLoss" in result.recommendations

    def test_explanation_test_004(self):
        result = run(["ClassImbalanceDetected", "TestAccuracyLow"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TEST_004" in rule_ids


# ---------------------------------------------------------------------------
# TEST_005 – high validation loss + low test accuracy
# ---------------------------------------------------------------------------

class TestHighValLossLowTestAcc:
    def test_poor_gen_derived(self):
        result = run(["ValidationLossHigh", "TestAccuracyLow"])
        assert "PoorGeneralization" in result.causes

    def test_inspect_pipeline_recommended(self):
        result = run(["ValidationLossHigh", "TestAccuracyLow"])
        assert "InspectDataPipeline" in result.recommendations

    def test_explanation_test_005(self):
        result = run(["ValidationLossHigh", "TestAccuracyLow"])
        rule_ids = [e["rule_id"] for e in result.explanations]
        assert "TEST_005" in rule_ids


# ---------------------------------------------------------------------------
# Multi-symptom – combined distribution shift + class imbalance
# ---------------------------------------------------------------------------

class TestCombinedDistributionImbalance:
    def test_both_causes_present(self):
        result = run(["ValidationAccuracyHigh", "TestAccuracyLow", "ClassImbalanceDetected"])
        assert "DistributionShift" in result.causes
        assert "DataImbalance" in result.causes

    def test_recommendations_cover_both(self):
        result = run(["ValidationAccuracyHigh", "TestAccuracyLow", "ClassImbalanceDetected"])
        assert "CollectDomainData" in result.recommendations
        assert "RebalanceClasses" in result.recommendations


# ---------------------------------------------------------------------------
# Engine reset between runs
# ---------------------------------------------------------------------------

class TestEngineReset:
    def test_independent_runs(self):
        """Two sequential runs on the same engine instance should be independent."""
        engine = DebuggingKnowledgeEngine()

        r1 = engine.run_scenario(["ValidationAccuracyHigh", "TestAccuracyLow"])
        r2 = engine.run_scenario(["TrainingLossHigh", "OscillatingLoss"])

        # r1 should contain distribution shift; r2 should not
        assert "DistributionShift" in r1.causes
        assert "DistributionShift" not in r2.causes

        # r2 should contain high LR; r1 should not
        assert "LearningRateTooHigh" in r2.causes
        assert "LearningRateTooHigh" not in r1.causes

    def test_no_fact_bleed(self):
        """Facts from a previous run must not appear in subsequent runs."""
        engine = DebuggingKnowledgeEngine()
        engine.run_scenario(["GradientExplosion"])
        result = engine.run_scenario(["TrainingLossHigh", "SlowConvergence"])
        # GradientExplosion facts should be gone
        assert "UseWeightClipping" not in result.recommendations or \
               "GradientExplosion" not in result.symptoms
