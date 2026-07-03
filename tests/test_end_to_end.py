"""
test_end_to_end.py
==================
End-to-end tests: natural language -> NLP -> Experta -> diagnosis.

All reasoning is exercised through real Experta inference (no mocks).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine


def diagnose(text: str):
    return DebuggingKnowledgeEngine().run_text(text)


class TestOscillatingLossEnd2End:
    def test_cause(self):
        r = diagnose("my training loss keeps oscillating and never converges")
        assert "LearningRateTooHigh" in r.causes

    def test_recommendation(self):
        r = diagnose("training loss high and loss oscillates")
        assert "ReduceLearningRate" in r.recommendations

    def test_confidence_present(self):
        r = diagnose("training loss high and loss oscillates")
        assert "LearningRateTooHigh" in r.confidence
        assert 0.0 < r.confidence["LearningRateTooHigh"] <= 1.0


class TestOverfittingEnd2End:
    def test_overfitting(self):
        r = diagnose(
            "the model reaches 99% training accuracy but validation accuracy is much lower"
        )
        assert "OverfittingObserved" in r.causes
        assert "AddDropout" in r.recommendations


class TestNaNTransformerEnd2End:
    def test_numerical_instability(self):
        r = diagnose("my transformer suddenly starts producing NaN values")
        assert "NumericalInstability" in r.causes

    def test_clipping_recommended(self):
        r = diagnose("my transformer suddenly starts producing NaN values")
        assert "UseGradientClipping" in r.recommendations


class TestTransformerGating:
    def test_tokenizer_only_for_transformer(self):
        r = diagnose("my transformer has lots of unk tokens (tokenization problem)")
        assert "FixTokenizer" in r.recommendations

    def test_tokenizer_not_fired_without_transformer(self):
        # Same tokenization symptom but no model-type mentioned.
        r = DebuggingKnowledgeEngine().run_scenario(["TokenizationIssue"])
        assert "FixTokenizer" not in r.recommendations


class TestCNNGating:
    def test_dead_relu_cnn(self):
        r = diagnose("my cnn has dead relu units")
        assert "UseLeakyReLU" in r.recommendations


class TestExtractionProvenance:
    def test_extraction_attached(self):
        r = diagnose("validation accuracy much lower than training accuracy")
        assert r.extraction is not None
        assert "ValidationAccuracyLow" in r.extraction.symptom_facts


class TestExplanationConfidenceField:
    def test_every_explanation_has_confidence(self):
        r = diagnose("training loss high and loss oscillates")
        for exp in r.explanations:
            assert "confidence" in exp
            assert 0.0 <= exp["confidence"] <= 1.0


class TestNegationEnd2End:
    def test_negated_overfitting_produces_no_diagnosis(self):
        r = diagnose("there is no overfitting")
        assert r.extraction is not None
        assert r.extraction.all_fact_names == []
        assert r.causes == []
        assert r.recommendations == []

    def test_negated_nan_produces_no_nan_diagnosis(self):
        r = diagnose("no NaNs")
        assert r.extraction is not None
        assert "NaNLoss" not in r.extraction.symptom_facts
        assert "NumericalInstability" not in r.causes


class TestExpandedVocabularyEnd2End:
    def test_loss_blows_up(self):
        r = diagnose("the loss blows up after a few batches")
        assert "LearningRateTooHigh" in r.causes
        assert "ReduceLearningRate" in r.recommendations

    def test_adam_mixed_precision_context(self):
        r = diagnose("adam with mixed precision produces NaNs")
        assert "OptimizerIsAdam" in r.extraction.context_facts
        assert "UsesMixedPrecision" in r.extraction.context_facts
        assert "Fp16RangeExceeded" in r.causes
