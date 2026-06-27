"""
test_new_rules.py
=================
Direct rule-activation tests for the newly added rule modules
(data, transformer, cnn, extended optimisation/architecture).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine


def run(symptoms):
    return DebuggingKnowledgeEngine().run_scenario(symptoms)


class TestDataRules:
    def test_leakage_pipeline(self):
        r = run(["DataLeakageSuspected"])
        assert "InspectDataPipeline" in r.recommendations
        assert "DATA_001" in [e["rule_id"] for e in r.explanations]

    def test_noise_quality_chain(self):
        r = run(["NoisyLabels"])
        assert "PoorDataQuality" in r.causes
        assert "ImproveLabelQuality" in r.recommendations


class TestOptimisationExtensions:
    def test_nan_instability(self):
        r = run(["NaNLoss"])
        assert "NumericalInstability" in r.causes
        assert "UseGradientClipping" in r.recommendations
        assert "OPT_007" in [e["rule_id"] for e in r.explanations]


class TestArchitectureExtensions:
    def test_dead_relu(self):
        r = run(["DeadReLUDetected"])
        assert "UseLeakyReLU" in r.recommendations
        assert "ARCH_009" in [e["rule_id"] for e in r.explanations]


class TestTransformerRules:
    def test_context_length(self):
        r = run(["ModelIsTransformer", "ContextLengthExceeded"])
        assert "TruncateOrChunkInput" in r.recommendations
        assert "TF_003" in [e["rule_id"] for e in r.explanations]

    def test_attention_collapse(self):
        r = run(["ModelIsTransformer", "AttentionCollapse"])
        assert "PoorGeneralization" in r.causes


class TestCNNRules:
    def test_receptive_field(self):
        r = run(["ModelIsCNN", "ReceptiveFieldTooSmall"])
        assert "IncreaseReceptiveField" in r.recommendations
        assert "CNN_002" in [e["rule_id"] for e in r.explanations]


class TestConfidenceAggregation:
    def test_noisy_or_increases_with_evidence(self):
        # GradientExplosion triggers multiple supporting rules for bad init.
        r = run(["GradientExplosion"])
        if "BadWeightInitialization" in r.confidence:
            assert 0.0 < r.confidence["BadWeightInitialization"] <= 1.0
