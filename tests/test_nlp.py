"""
test_nlp.py
===========
Tests for the NLP preprocessing layer (text -> Fact names). No diagnosis here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

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

    def test_lemma_based_oscillating_loss_forms(self):
        for text in (
            "loss oscillates",
            "loss is oscillating",
            "loss oscillated wildly",
        ):
            r = extract(text)
            assert "OscillatingLoss" in r.symptom_facts

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

    def test_negation_does_not_bleed_across_clause(self):
        r = extract("no overfitting, but the loss oscillates")
        assert "OverfittingObserved" not in r.symptom_facts
        assert "OscillatingLoss" in r.symptom_facts
        assert [e.fact_name for e in r.negated] == ["OverfittingObserved"]


class TestExpandedVocabulary:
    def test_plateau_maps_to_slow_convergence(self):
        r = extract("training has plateaued and seems stuck")
        assert "SlowConvergence" in r.symptom_facts

    def test_loss_plateau_emits_both_intended_facts(self):
        r = extract("loss plateau")
        assert "TrainingLossHigh" in r.symptom_facts
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


class TestReportedPromptVocabulary:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "training loss is high and the loss keeps oscillating",
                {"TrainingLossHigh", "OscillatingLoss"},
            ),
            (
                "the model is converging very slowly and the training loss is still high",
                {"SlowConvergence", "TrainingLossHigh"},
            ),
            (
                "the early layers are barely learning and gradients look near zero",
                {"GradientVanishing"},
            ),
            (
                "training accuracy is high but validation accuracy is low",
                {"TrainingAccuracyHigh", "ValidationAccuracyLow"},
            ),
            (
                "both training accuracy and validation accuracy are low",
                {"TrainingAccuracyLow", "ValidationAccuracyLow"},
            ),
            (
                "validation loss is high and the dataset is small",
                {"ValidationLossHigh", "SmallDataset"},
            ),
            (
                "validation accuracy is high but test accuracy is low",
                {"ValidationAccuracyHigh", "TestAccuracyLow"},
            ),
            (
                "labels are noisy and training loss stays high",
                {"NoisyLabels", "TrainingLossHigh"},
            ),
            (
                "the transformer runs out of memory on long sequences",
                {"LongSequenceMemoryBlowup", "ModelIsTransformer"},
            ),
            (
                "attention entropy collapses and warmup is missing",
                {"AttentionEntropyCollapse", "WarmupMissing", "ModelIsTransformer"},
            ),
            (
                "batch norm stats are desynchronized across GPUs",
                {"BatchNormDesync", "UsesDistributedTraining"},
            ),
        ],
    )
    def test_reported_prompt_phrases_extract_expected_facts(self, text, expected):
        r = extract(text)
        assert expected <= set(r.all_fact_names)

    def test_user_stated_cause_is_separate_from_symptoms(self):
        r = extract("maybe the learning rate is too high because loss seems to oscillate")
        assert "LearningRateTooHigh" in r.cause_facts
        assert "LearningRateTooHigh" not in r.symptom_facts


class TestAdditionalReportedPromptVocabulary:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "Both training and validation accuracy are low.",
                {"TrainingAccuracyLow", "ValidationAccuracyLow"},
            ),
            (
                "Lots of unk tokens, looks like a tokenization issue.",
                {"TokenizationIssue"},
            ),
            (
                "The resnet feature maps are constant.",
                {"ModelIsCNN", "FeatureCollapse"},
            ),
            (
                "Multi gpu training throughput is low and the gpus are idle.",
                {
                    "UsesDistributedTraining",
                    "MultiGpuThroughputLow",
                    "GpuUnderutilization",
                },
            ),
            (
                "Using adam with coupled weight decay.",
                {"OptimizerIsAdam", "CoupledWeightDecayUsed"},
            ),
        ],
    )
    def test_additional_reported_prompt_phrases_extract_expected_facts(
        self,
        text,
        expected,
    ):
        r = extract(text)
        assert expected <= set(r.all_fact_names)


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

    def test_hedged_match_lowers_entity_confidence(self):
        definite = extract("loss oscillates")
        hedged = extract("might be loss oscillating")
        definite_confidence = max(
            e.confidence for e in definite.evidence
            if e.fact_name == "OscillatingLoss"
        )
        hedged_confidence = max(
            e.confidence for e in hedged.evidence
            if e.fact_name == "OscillatingLoss"
        )
        assert hedged_confidence < definite_confidence
