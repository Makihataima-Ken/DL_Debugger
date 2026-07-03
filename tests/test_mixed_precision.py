"""
test_mixed_precision.py
=======================
Tests for AMP/mixed-precision Experta rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine


def run(facts: list[str]):
    return DebuggingKnowledgeEngine().run_scenario(facts)


AMP_CASES = [
    ("AMP_001", ["UsesMixedPrecision", "NaNLoss"], "Fp16RangeExceeded", "causes"),
    ("AMP_002", ["UsesMixedPrecision", "MixedPrecisionOverflow"], "Fp16RangeExceeded", "causes"),
    ("AMP_003", ["UsesMixedPrecision", "LossScaleOverflow"], "LossScalingMisconfigured", "causes"),
    ("AMP_004", ["UsesMixedPrecision", "GradientUnderflow"], "Fp16GradientUnderflow", "causes"),
    ("AMP_005", ["UsesMixedPrecision", "LossScalingMisconfigured"], "EnableDynamicLossScaling", "recommendations"),
    ("AMP_006", ["UsesMixedPrecision", "Fp16RangeExceeded"], "UseBf16", "recommendations"),
    ("AMP_007", ["UsesMixedPrecision", "Fp16GradientUnderflow"], "KeepMasterWeightsInFp32", "recommendations"),
]


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), AMP_CASES)
def test_mixed_precision_rule_fires(rule_id, facts, derived, bucket):
    result = run(facts)
    assert derived in getattr(result, bucket)
    assert rule_id in [e["rule_id"] for e in result.explanations]
    exp = next(e for e in result.explanations if e["rule_id"] == rule_id)
    assert 0.5 <= exp["confidence"] <= 0.95


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), AMP_CASES)
def test_mixed_precision_rules_require_context(rule_id, facts, derived, bucket):
    result = run([fact for fact in facts if fact != "UsesMixedPrecision"])
    assert derived not in getattr(result, bucket)
    assert rule_id not in [e["rule_id"] for e in result.explanations]
