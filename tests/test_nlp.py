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
