"""
knowledge_engine.py
===================
Central ``DebuggingKnowledgeEngine`` that aggregates all rule modules and
exposes a clean interface for running inference and collecting results.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING
from collections import defaultdict

from experta import KnowledgeEngine, DefFacts, Fact

from models.facts import (
    # Symptom facts (all importable from models.facts)
    TrainingLossHigh,
    ValidationLossHigh,
    TrainingAccuracyHigh,
    TrainingAccuracyLow,
    ValidationAccuracyHigh,
    ValidationAccuracyLow,
    TestAccuracyLow,
    OverfittingObserved,
    UnderfittingObserved,
    GradientExplosion,
    GradientVanishing,
    SlowConvergence,
    OscillatingLoss,
    ClassImbalanceDetected,
    DataLeakageSuspected,
    SmallDataset,
    LargeDataset,
    NoisyLabels,
    PoorGeneralization,
    NaNLoss,
    DeadReLUDetected,
    AttentionCollapse,
    TokenizationIssue,
    ContextLengthExceeded,
    FeatureCollapse,
    ReceptiveFieldTooSmall,
    ModelIsTransformer,
    ModelIsCNN,
    # Cause facts
    LearningRateTooHigh,
    LearningRateTooLow,
    ModelTooComplex,
    ModelTooSimple,
    InsufficientRegularization,
    ExcessiveRegularization,
    BadWeightInitialization,
    PoorDataQuality,
    DistributionShift,
    DataImbalance,
    BatchSizeTooLarge,
    BatchSizeTooSmall,
    MomentumTooHigh,
    WeightDecayTooHigh,
    NumericalInstability,
    # Recommendation facts
    ReduceLearningRate,
    IncreaseLearningRate,
    AddDropout,
    RemoveDropout,
    AddBatchNormalization,
    UseEarlyStopping,
    IncreaseDatasetSize,
    ApplyDataAugmentation,
    RebalanceClasses,
    ImproveLabelQuality,
    UseWeightClipping,
    UseProperInitialization,
    ReduceModelComplexity,
    IncreaseModelComplexity,
    UseWeightedLoss,
    InspectDataPipeline,
    CollectDomainData,
    ReduceRegularization,
    ReduceBatchSize,
    IncreaseBatchSize,
    ReduceMomentum,
    ReduceWeightDecay,
    UseLeakyReLU,
    UseGradientClipping,
    TruncateOrChunkInput,
    FixTokenizer,
    IncreaseReceptiveField,
    # XAI
    Explanation,
)

from engine.rules.training_rules import TrainingRules
from engine.rules.validation_rules import ValidationRules
from engine.rules.testing_rules import TestingRules
from engine.rules.optimization_rules import OptimizationRules
from engine.rules.architecture_rules import ArchitectureRules
from engine.rules.recommendation_rules import RecommendationRules
from engine.rules.data_rules import DataRules
from engine.rules.transformer_rules import TransformerRules
from engine.rules.cnn_rules import CNNRules


# ---------------------------------------------------------------------------
# Fact category sets (for result classification)
# ---------------------------------------------------------------------------

SYMPTOM_FACT_CLASSES: set[type] = {
    TrainingLossHigh,
    ValidationLossHigh,
    TrainingAccuracyHigh,
    TrainingAccuracyLow,
    ValidationAccuracyHigh,
    ValidationAccuracyLow,
    TestAccuracyLow,
    OverfittingObserved,
    UnderfittingObserved,
    GradientExplosion,
    GradientVanishing,
    SlowConvergence,
    OscillatingLoss,
    ClassImbalanceDetected,
    DataLeakageSuspected,
    SmallDataset,
    LargeDataset,
    NoisyLabels,
    PoorGeneralization,
    NaNLoss,
    DeadReLUDetected,
    AttentionCollapse,
    TokenizationIssue,
    ContextLengthExceeded,
    FeatureCollapse,
    ReceptiveFieldTooSmall,
    ModelIsTransformer,
    ModelIsCNN,
}

CAUSE_FACT_CLASSES: set[type] = {
    LearningRateTooHigh,
    LearningRateTooLow,
    ModelTooComplex,
    ModelTooSimple,
    InsufficientRegularization,
    ExcessiveRegularization,
    BadWeightInitialization,
    PoorDataQuality,
    DistributionShift,
    DataImbalance,
    BatchSizeTooLarge,
    BatchSizeTooSmall,
    MomentumTooHigh,
    WeightDecayTooHigh,
    NumericalInstability,
}

RECOMMENDATION_FACT_CLASSES: set[type] = {
    ReduceLearningRate,
    IncreaseLearningRate,
    AddDropout,
    RemoveDropout,
    AddBatchNormalization,
    UseEarlyStopping,
    IncreaseDatasetSize,
    ApplyDataAugmentation,
    RebalanceClasses,
    ImproveLabelQuality,
    UseWeightClipping,
    UseProperInitialization,
    ReduceModelComplexity,
    IncreaseModelComplexity,
    UseWeightedLoss,
    InspectDataPipeline,
    CollectDomainData,
    ReduceRegularization,
    ReduceBatchSize,
    IncreaseBatchSize,
    ReduceMomentum,
    ReduceWeightDecay,
    UseLeakyReLU,
    UseGradientClipping,
    TruncateOrChunkInput,
    FixTokenizer,
    IncreaseReceptiveField,
}

# Mapping: string name → Fact class (for dynamic symptom injection)
FACT_CLASS_REGISTRY: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        SYMPTOM_FACT_CLASSES
        | CAUSE_FACT_CLASSES
        | RECOMMENDATION_FACT_CLASSES
    )
}


# ---------------------------------------------------------------------------
# Composite engine
# ---------------------------------------------------------------------------

class DebuggingKnowledgeEngine(
    TrainingRules,
    ValidationRules,
    TestingRules,
    OptimizationRules,
    ArchitectureRules,
    RecommendationRules,
    DataRules,
    TransformerRules,
    CNNRules,
):
    """
    Central expert-system engine that combines all rule modules via multiple
    inheritance.

    Usage
    -----
    ::

        engine = DebuggingKnowledgeEngine()
        engine.reset()
        engine.declare(TrainingLossHigh())
        engine.declare(OscillatingLoss())
        engine.run()

        result = engine.get_diagnosis()
        print(result)
    """

    def get_diagnosis(self) -> "DiagnosisResult":
        """Collect and classify all facts asserted during inference.

        A fact is classified as a **symptom** only if it was explicitly
        injected (present in ``_injected_symptoms``).  Facts that were
        *derived* by the engine are classified as causes so that
        intermediate inferences (e.g. ``OverfittingObserved``) appear in
        the right bucket.

        Returns
        -------
        DiagnosisResult
            Structured container with symptoms, causes, recommendations,
            and the XAI explanation chain.
        """
        symptoms: list[str] = []
        causes: list[str] = []
        recommendations: list[str] = []
        explanations: list[dict[str, str]] = []
        confidence_by_fact: dict[str, list[float]] = defaultdict(list)

        for fact in self.facts.values():
            fact_type = type(fact)
            name = fact_type.__name__

            if fact_type in RECOMMENDATION_FACT_CLASSES:
                recommendations.append(name)
            elif fact_type is Explanation:
                explanations.append(
                    {
                        "rule_id": fact["rule_id"],
                        "triggered_by": fact["triggered_by"],
                        "derived": fact["derived"],
                        "explanation": fact["explanation"],
                        "confidence": float(fact.get("confidence", 0.8)),
                    }
                )
            elif name in self._injected_symptoms:
                symptoms.append(name)
            elif fact_type in SYMPTOM_FACT_CLASSES or fact_type in CAUSE_FACT_CLASSES:
                causes.append(name)

        # ---- Confidence aggregation (REPORTING ONLY; not diagnostic logic) ----
        # Each Explanation already encodes the rule-assigned confidence in its
        # derived fact. We aggregate evidence using a noisy-OR combination so
        # that multiple rules supporting the same conclusion reinforce it.
        for exp in explanations:
            confidence_by_fact[exp["derived"]].append(exp["confidence"])

        def _noisy_or(values: list[float]) -> float:
            acc = 1.0
            for v in values:
                acc *= (1.0 - max(0.0, min(1.0, v)))
            return round(1.0 - acc, 3)

        confidence = {
            fact_name: _noisy_or(vals)
            for fact_name, vals in confidence_by_fact.items()
        }

        return DiagnosisResult(
            symptoms=sorted(set(symptoms)),
            causes=sorted(set(causes)),
            recommendations=sorted(set(recommendations)),
            explanations=sorted(explanations, key=lambda x: x["rule_id"]),
            confidence=confidence,
        )

    def inject_symptoms(self, symptom_names: list[str]) -> None:
        """Declare symptom facts by class name.

        Parameters
        ----------
        symptom_names:
            List of Fact subclass names (e.g. ``["TrainingLossHigh"]``).

        Raises
        ------
        ValueError
            If a name is not found in the fact registry.
        """
        self._injected_symptoms: set[str] = set(symptom_names)
        for name in symptom_names:
            cls = FACT_CLASS_REGISTRY.get(name)
            if cls is None:
                raise ValueError(
                    f"Unknown fact class: '{name}'. "
                    f"Valid names: {sorted(FACT_CLASS_REGISTRY)}"
                )
            self.declare(cls())

    def run_scenario(self, symptom_names: list[str]) -> "DiagnosisResult":
        """Convenience: reset, inject symptoms, run, return diagnosis.

        Parameters
        ----------
        symptom_names:
            Fact class names to assert as the starting symptom set.

        Returns
        -------
        DiagnosisResult
            Final inference result.
        """
        self._injected_symptoms: set[str] = set()
        self.reset()
        self.inject_symptoms(symptom_names)
        self.run()
        return self.get_diagnosis()

    def run_text(self, text: str) -> "DiagnosisResult":
        """Natural-language entry point.

        Uses the NLP layer to convert *text* into symptom/model-type Fact
        names, then runs the standard rule-driven inference. The NLP layer
        performs NO diagnosis; all reasoning still happens in Experta rules.

        Parameters
        ----------
        text:
            Free-text description of the neural-network problem.

        Returns
        -------
        DiagnosisResult
            Final inference result, with an attached ``extraction`` record.
        """
        # Imported lazily to avoid a hard dependency for pure-symptom usage.
        from engine.nlp.symptom_extractor import SymptomExtractor

        extraction = SymptomExtractor().extract(text)
        result = self.run_scenario(extraction.all_fact_names)
        result.extraction = extraction
        return result


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class DiagnosisResult:
    """
    Immutable container for the output of one inference run.

    Attributes
    ----------
    symptoms : list[str]
        Fact names of symptoms present at the start of inference.
    causes : list[str]
        Fact names of root causes derived by the engine.
    recommendations : list[str]
        Fact names of recommended actions.
    explanations : list[dict[str, str]]
        Ordered list of XAI explanation records, each with keys
        ``rule_id``, ``triggered_by``, ``derived``, ``explanation``.
    confidence : dict[str, float]
        Aggregated confidence per derived fact (noisy-OR over rule evidence).
    """

    def __init__(
        self,
        symptoms: list[str],
        causes: list[str],
        recommendations: list[str],
        explanations: list[dict[str, str]],
        confidence: dict[str, float] | None = None,
    ) -> None:
        self.symptoms = symptoms
        self.causes = causes
        self.recommendations = recommendations
        self.explanations = explanations
        self.confidence = confidence or {}
        # Optional NLP provenance (set by run_text); None for symptom-only runs.
        self.extraction = None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DiagnosisResult("
            f"symptoms={self.symptoms}, "
            f"causes={self.causes}, "
            f"recommendations={self.recommendations})"
        )
