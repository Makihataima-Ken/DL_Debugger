"""
test_transformer.py
===================
Tests for expanded Transformer-specific Experta rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine


def run(facts: list[str]):
    return DebuggingKnowledgeEngine().run_scenario(facts)


TF_CASES = [
    ("TF_005", ["ModelIsTransformer", "WarmupMissing"], "MissingLearningRateWarmup", "causes"),
    ("TF_006", ["ModelIsTransformer", "MissingLearningRateWarmup"], "AddLearningRateWarmup", "recommendations"),
    ("TF_007", ["ModelIsTransformer", "AttentionEntropyCollapse"], "AttentionEntropyCollapsed", "causes"),
    ("TF_008", ["ModelIsTransformer", "AttentionEntropyCollapsed"], "ClipAttentionLogits", "recommendations"),
    ("TF_009", ["ModelIsTransformer", "PositionalEncodingProblem"], "PositionalEncodingMisconfigured", "causes"),
    ("TF_010", ["ModelIsTransformer", "PositionalEncodingMisconfigured"], "FixPositionalEncoding", "recommendations"),
    ("TF_011", ["ModelIsTransformer", "LongSequenceMemoryBlowup"], "QuadraticAttentionMemoryBlowup", "causes"),
    ("TF_012", ["ModelIsTransformer", "QuadraticAttentionMemoryBlowup"], "UseGradientCheckpointing", "recommendations"),
    ("TF_013", ["ModelIsTransformer", "QuadraticAttentionMemoryBlowup"], "UseFlashAttention", "recommendations"),
    ("TF_014", ["ModelIsTransformer", "RepetitiveGeneration"], "DegenerateGeneration", "causes"),
    ("TF_015", ["ModelIsTransformer", "DegenerateGeneration"], "ApplyLabelSmoothing", "recommendations"),
    ("TF_016", ["ModelIsTransformer", "DegenerateGeneration"], "AdjustDecodingStrategy", "recommendations"),
    ("TF_017", ["ModelIsTransformer", "ContextLengthExceeded"], "ChunkOrTruncateInput", "recommendations"),
]


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), TF_CASES)
def test_transformer_rule_fires(rule_id, facts, derived, bucket):
    result = run(facts)
    assert derived in getattr(result, bucket)
    assert rule_id in [e["rule_id"] for e in result.explanations]
    exp = next(e for e in result.explanations if e["rule_id"] == rule_id)
    assert 0.5 <= exp["confidence"] <= 0.95


@pytest.mark.parametrize(("rule_id", "facts", "derived", "bucket"), TF_CASES)
def test_transformer_rules_require_context(rule_id, facts, derived, bucket):
    result = run([fact for fact in facts if fact != "ModelIsTransformer"])
    assert derived not in getattr(result, bucket)
    assert rule_id not in [e["rule_id"] for e in result.explanations]


def test_tokenization_issue_has_generic_fallback():
    result = run(["TokenizationIssue"])

    assert "FixTokenizer" in result.recommendations
    assert "TF_018" in [e["rule_id"] for e in result.explanations]


def test_transformer_tokenization_uses_specific_rule_over_fallback():
    result = run(["ModelIsTransformer", "TokenizationIssue"])
    rule_ids = [e["rule_id"] for e in result.explanations]

    assert "FixTokenizer" in result.recommendations
    assert "TF_002" in rule_ids
    assert "TF_018" not in rule_ids
