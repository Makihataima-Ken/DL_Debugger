"""
test_distributed.py
===================
Tests for distributed and multi-GPU Experta rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine


def run(facts: list[str]):
    return DebuggingKnowledgeEngine().run_scenario(facts)


DIST_CASES = [
    (
        "DIST_001",
        ["UsesDistributedTraining", "MultiGpuThroughputLow", "GpuUnderutilization"],
        "DataLoadingBottleneck",
        "causes",
    ),
    ("DIST_002", ["UsesDistributedTraining", "BatchNormDesync"], "ImproperBatchNormSync", "causes"),
    ("DIST_003", ["UsesDistributedTraining", "GradientSyncSlow"], "GradientSyncOverhead", "causes"),
    (
        "DIST_004",
        ["UsesDistributedTraining", "PerDeviceBatchTooSmallObserved"],
        "PerDeviceBatchTooSmall",
        "causes",
    ),
    ("DIST_005", ["UsesDistributedTraining", "SlowConvergence"], "LearningRateNotScaled", "causes"),
    (
        "DIST_006",
        ["UsesDistributedTraining", "DataLoadingBottleneck"],
        "IncreaseDataLoaderWorkers",
        "recommendations",
    ),
    ("DIST_007", ["UsesDistributedTraining", "ImproperBatchNormSync"], "UseSyncBatchNorm", "recommendations"),
    (
        "DIST_008",
        ["UsesDistributedTraining", "GradientSyncOverhead"],
        "UseGradientAccumulation",
        "recommendations",
    ),
    (
        "DIST_009",
        ["UsesDistributedTraining", "LearningRateNotScaled"],
        "ScaleLearningRateByWorldSize",
        "recommendations",
    ),
    (
        "DIST_010",
        ["UsesDistributedTraining", "PerDeviceBatchTooSmall"],
        "UseGradientAccumulation",
        "recommendations",
    ),
]


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), DIST_CASES)
def test_distributed_rule_fires(rule_id, facts, derived, bucket):
    result = run(facts)
    assert derived in getattr(result, bucket)
    assert rule_id in [e["rule_id"] for e in result.explanations]
    exp = next(e for e in result.explanations if e["rule_id"] == rule_id)
    assert 0.5 <= exp["confidence"] <= 0.95


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), DIST_CASES)
def test_distributed_rules_require_context(rule_id, facts, derived, bucket):
    result = run([fact for fact in facts if fact != "UsesDistributedTraining"])
    assert derived not in getattr(result, bucket)
    assert rule_id not in [e["rule_id"] for e in result.explanations]
