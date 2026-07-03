"""
test_nlp.py
===========
Tests for the NLP preprocessing layer (text -> Fact names). No diagnosis here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.nlp.terminology_normalizer import normalize
from engine.nlp.symptom_extractor import SymptomExtractor


def extract(text: str):
    return SymptomExtractor().extract(text)


class TestNormalization:
    def test_lowercase_and_punct(self):
        assert normalize("Loss!!! oscillates.") == "loss oscillates"

    def test_synonym_expansion(self):
        # "val" -> "validation", "acc" -> "accuracy"
        assert "validation accuracy" in normalize("val acc is low")


class TestSymptomExtraction:
    def test_oscillating_loss(self):
        r = extract("my training loss keeps oscillating and never converges")
        assert "OscillatingLoss" in r.symptom_facts
        assert "TrainingLossHigh" in r.symptom_facts

    def test_overfitting_from_memorizing(self):
        r = extract("the model is just memorizing the training set")
        assert "OverfittingObserved" in r.symptom_facts

    def test_validation_lower(self):
        r = extract("validation accuracy is much lower than training accuracy")
        assert "ValidationAccuracyLow" in r.symptom_facts

    def test_gradient_explosion(self):
        r = extract("my training loss is exploding")
        assert "GradientExplosion" in r.symptom_facts

    def test_nan(self):
        r = extract("my transformer suddenly starts producing NaN values")
        assert "NaNLoss" in r.symptom_facts

    def test_model_type_transformer(self):
        r = extract("my transformer produces nan")
        assert "ModelIsTransformer" in r.model_type_facts

    def test_model_type_cnn(self):
        r = extract("my cnn has dead relu units")
        assert "ModelIsCNN" in r.model_type_facts
        assert "DeadReLUDetected" in r.symptom_facts

    def test_lexical_confidence_range(self):
        r = extract("validation accuracy much lower")
        assert 0.0 < r.lexical_confidence <= 1.0

    def test_unknown_text_empty(self):
        r = extract("the weather is nice today")
        assert r.all_fact_names == []


class TestNegationHandling:
    def test_no_overfitting_suppressed(self):
        r = extract("there is no overfitting")
        assert "OverfittingObserved" not in r.symptom_facts
        assert r.all_fact_names == []
        assert [e.fact_name for e in r.negated] == ["OverfittingObserved"]

    def test_not_oscillating_suppressed(self):
        r = extract("loss is not oscillating")
        assert "OscillatingLoss" not in r.symptom_facts
        assert r.all_fact_names == []

    def test_not_exploding_suppressed(self):
        r = extract("gradients are not exploding")
        assert "GradientExplosion" not in r.symptom_facts
        assert r.all_fact_names == []

    def test_without_data_leakage_suppressed(self):
        r = extract("training without data leakage")
        assert "DataLeakageSuspected" not in r.symptom_facts
        assert r.all_fact_names == []

    def test_no_nans_suppressed(self):
        r = extract("no NaNs")
        assert "NaNLoss" not in r.symptom_facts
        assert r.all_fact_names == []

    def test_positive_loss_not_decreasing_survives(self):
        r = extract("loss not decreasing")
        assert "TrainingLossHigh" in r.symptom_facts

    def test_positive_loss_never_converges_survives(self):
        r = extract("loss never converges")
        assert "TrainingLossHigh" in r.symptom_facts

    def test_positive_early_layers_not_learning_survives(self):
        r = extract("early layers not learning")
        assert "GradientVanishing" in r.symptom_facts


class TestExpandedVocabulary:
    def test_plateau_maps_to_slow_convergence(self):
        r = extract("training has plateaued and seems stuck")
        assert "SlowConvergence" in r.symptom_facts

    def test_infinity_maps_to_nan(self):
        r = extract("the loss goes to infinity")
        assert "NaNLoss" in r.symptom_facts

    def test_blows_up_maps_to_gradient_explosion(self):
        r = extract("the loss blows up after a few batches")
        assert "GradientExplosion" in r.symptom_facts

    def test_vision_transformer_model_type(self):
        r = extract("my vision transformer has attention collapse")
        assert "ModelIsTransformer" in r.model_type_facts
        assert "AttentionCollapse" in r.symptom_facts

    def test_context_fact_extraction(self):
        r = extract("adam with mixed precision produces NaNs")
        assert "OptimizerIsAdam" in r.context_facts
        assert "UsesMixedPrecision" in r.context_facts
        assert "NaNLoss" in r.symptom_facts


class TestTokenAwareMatching:
    def test_short_key_inside_larger_word_does_not_match(self):
        r = extract("the banana analysis looks fine")
        assert "NaNLoss" not in r.symptom_facts
        assert r.all_fact_names == []

    def test_val_inside_validation_is_not_double_expanded(self):
        assert normalize("validation accuracy is low") == "validation accuracy is low"


class TestHedgedConfidence:
    def test_hedged_language_lowers_lexical_confidence(self):
        definite = extract("validation accuracy is much lower")
        hedged = extract("maybe validation accuracy is much lower")
        assert "ValidationAccuracyLow" in hedged.symptom_facts
        assert hedged.lexical_confidence < definite.lexical_confidence
