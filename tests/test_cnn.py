"""
test_cnn.py
===========
Tests for expanded CNN-specific Experta rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine


def run(facts: list[str]):
    return DebuggingKnowledgeEngine().run_scenario(facts)


CNN_CASES = [
    ("CNN_004", ["ModelIsCNN", "BatchNormTrainEvalMismatch"], "BatchNormModeMismatch", "causes"),
    ("CNN_005", ["ModelIsCNN", "BatchNormModeMismatch"], "FixBatchNormMomentum", "recommendations"),
    ("CNN_006", ["ModelIsCNN", "AggressivePoolingOrStride"], "StrideTooAggressive", "causes"),
    ("CNN_007", ["ModelIsCNN", "StrideTooAggressive"], "ReduceStride", "recommendations"),
    ("CNN_008", ["ModelIsCNN", "InsufficientSpatialAugmentation"], "SpatialAugmentationMissing", "causes"),
    ("CNN_009", ["ModelIsCNN", "SpatialAugmentationMissing"], "AddSpatialAugmentation", "recommendations"),
    ("CNN_010", ["ModelIsCNN", "ChannelCollapse"], "ChannelCollapseCause", "causes"),
    ("CNN_011", ["ModelIsCNN", "ChannelCollapseCause"], "UseGlobalAveragePooling", "recommendations"),
]


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), CNN_CASES)
def test_cnn_rule_fires(rule_id, facts, derived, bucket):
    result = run(facts)
    assert derived in getattr(result, bucket)
    assert rule_id in [e["rule_id"] for e in result.explanations]
    exp = next(e for e in result.explanations if e["rule_id"] == rule_id)
    assert 0.5 <= exp["confidence"] <= 0.95


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), CNN_CASES)
def test_cnn_rules_require_context(rule_id, facts, derived, bucket):
    result = run([fact for fact in facts if fact != "ModelIsCNN"])
    assert derived not in getattr(result, bucket)
    assert rule_id not in [e["rule_id"] for e in result.explanations]
