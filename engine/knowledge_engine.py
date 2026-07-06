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
from pathlib import Path

from utils import experta_compat  # noqa: F401
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
    MixedPrecisionOverflow,
    LossScaleOverflow,
    GradientUnderflow,
    MultiGpuThroughputLow,
    GpuUnderutilization,
    BatchNormDesync,
    GradientSyncSlow,
    PerDeviceBatchTooSmallObserved,
    CoupledWeightDecayUsed,
    MissingMomentum,
    AdamHighLearningRateInstability,
    WarmupMissing,
    AttentionEntropyCollapse,
    PositionalEncodingProblem,
    LongSequenceMemoryBlowup,
    RepetitiveGeneration,
    BatchNormTrainEvalMismatch,
    AggressivePoolingOrStride,
    InsufficientSpatialAugmentation,
    ChannelCollapse,
    # Context facts
    ModelIsTransformer,
    ModelIsCNN,
    UsesMixedPrecision,
    UsesDistributedTraining,
    OptimizerIsAdam,
    OptimizerIsSGD,
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
    LossScalingMisconfigured,
    Fp16RangeExceeded,
    Fp16GradientUnderflow,
    DataLoadingBottleneck,
    ImproperBatchNormSync,
    GradientSyncOverhead,
    LearningRateNotScaled,
    PerDeviceBatchTooSmall,
    WeightDecayCoupledWithAdam,
    MomentumMisconfigured,
    AdamLearningRateTooHigh,
    MissingLearningRateWarmup,
    AttentionEntropyCollapsed,
    PositionalEncodingMisconfigured,
    QuadraticAttentionMemoryBlowup,
    DegenerateGeneration,
    BatchNormModeMismatch,
    StrideTooAggressive,
    SpatialAugmentationMissing,
    ChannelCollapseCause,
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
    EnableDynamicLossScaling,
    UseBf16,
    KeepMasterWeightsInFp32,
    UseSyncBatchNorm,
    IncreaseDataLoaderWorkers,
    UseGradientAccumulation,
    ScaleLearningRateByWorldSize,
    UseAdamW,
    EnableNesterovMomentum,
    UseLearningRateWarmup,
    AddLearningRateWarmup,
    UseGradientCheckpointing,
    UseFlashAttention,
    ApplyLabelSmoothing,
    ClipAttentionLogits,
    ChunkOrTruncateInput,
    FixPositionalEncoding,
    AdjustDecodingStrategy,
    AddSpatialAugmentation,
    ReduceStride,
    FixBatchNormMomentum,
    UseGlobalAveragePooling,
    # Meta facts
    SuppressedCause,
    CauseConflict,
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
from engine.rules.mixed_precision_rules import MixedPrecisionRules
from engine.rules.distributed_rules import DistributedRules
from engine.rules.optimizer_rules import OptimizerInteractionRules
from engine.rules.conflict_resolution_rules import ConflictResolutionRules


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
    MixedPrecisionOverflow,
    LossScaleOverflow,
    GradientUnderflow,
    MultiGpuThroughputLow,
    GpuUnderutilization,
    BatchNormDesync,
    GradientSyncSlow,
    PerDeviceBatchTooSmallObserved,
    CoupledWeightDecayUsed,
    MissingMomentum,
    AdamHighLearningRateInstability,
    WarmupMissing,
    AttentionEntropyCollapse,
    PositionalEncodingProblem,
    LongSequenceMemoryBlowup,
    RepetitiveGeneration,
    BatchNormTrainEvalMismatch,
    AggressivePoolingOrStride,
    InsufficientSpatialAugmentation,
    ChannelCollapse,
}

# Context/model facts are injectable by class name, but are not diagnoses.
CONTEXT_FACT_CLASSES: set[type] = {
    ModelIsTransformer,
    ModelIsCNN,
    UsesMixedPrecision,
    UsesDistributedTraining,
    OptimizerIsAdam,
    OptimizerIsSGD,
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
    LossScalingMisconfigured,
    Fp16RangeExceeded,
    Fp16GradientUnderflow,
    DataLoadingBottleneck,
    ImproperBatchNormSync,
    GradientSyncOverhead,
    LearningRateNotScaled,
    PerDeviceBatchTooSmall,
    WeightDecayCoupledWithAdam,
    MomentumMisconfigured,
    AdamLearningRateTooHigh,
    MissingLearningRateWarmup,
    AttentionEntropyCollapsed,
    PositionalEncodingMisconfigured,
    QuadraticAttentionMemoryBlowup,
    DegenerateGeneration,
    BatchNormModeMismatch,
    StrideTooAggressive,
    SpatialAugmentationMissing,
    ChannelCollapseCause,
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
    EnableDynamicLossScaling,
    UseBf16,
    KeepMasterWeightsInFp32,
    UseSyncBatchNorm,
    IncreaseDataLoaderWorkers,
    UseGradientAccumulation,
    ScaleLearningRateByWorldSize,
    UseAdamW,
    EnableNesterovMomentum,
    UseLearningRateWarmup,
    AddLearningRateWarmup,
    UseGradientCheckpointing,
    UseFlashAttention,
    ApplyLabelSmoothing,
    ClipAttentionLogits,
    ChunkOrTruncateInput,
    FixPositionalEncoding,
    AdjustDecodingStrategy,
    AddSpatialAugmentation,
    ReduceStride,
    FixBatchNormMomentum,
    UseGlobalAveragePooling,
}

# Mapping: string name → Fact class (for dynamic symptom injection)
META_FACT_CLASSES: set[type] = {
    SuppressedCause,
    CauseConflict,
}

FACT_CLASS_REGISTRY: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        SYMPTOM_FACT_CLASSES
        | CONTEXT_FACT_CLASSES
        | CAUSE_FACT_CLASSES
        | RECOMMENDATION_FACT_CLASSES
        | META_FACT_CLASSES
    )
}

CONFIDENCE_CONVERGENCE_EPSILON = 1e-6
CONFIDENCE_FIXPOINT_MAX_ITERATIONS = 50


def _clamp_confidence(value: float) -> float:
    """Keep reported confidences in the public [0.0, 1.0] range."""
    return max(0.0, min(1.0, float(value)))


def _split_antecedents(triggered_by: str) -> list[str]:
    """Parse Explanation.triggered_by into clean fact names."""
    return [
        name.strip()
        for name in triggered_by.split(",")
        if name.strip()
    ]


def _noisy_or(values: list[float]) -> float:
    """Combine independent supporting contributions without exceeding 1.0."""
    acc = 1.0
    for value in values:
        acc *= 1.0 - _clamp_confidence(value)
    return _clamp_confidence(1.0 - acc)


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
    MixedPrecisionRules,
    DistributedRules,
    OptimizerInteractionRules,
    ConflictResolutionRules,
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
        conflicts: list[dict[str, str]] = []
        suppressed_causes: set[str] = set()
        injected_symptoms = getattr(self, "_injected_symptoms", set())

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
            elif fact_type is SuppressedCause:
                suppressed_causes.add(fact["cause"])
            elif fact_type is CauseConflict:
                conflicts.append(
                    {
                        "rule_id": fact["rule_id"],
                        "contending_causes": fact["contending_causes"],
                        "winner": fact["winner"],
                        "losers": fact["losers"],
                        "reason": fact["reason"],
                    }
                )
            elif name in injected_symptoms and fact_type in SYMPTOM_FACT_CLASSES:
                symptoms.append(name)
            elif fact_type in CONTEXT_FACT_CLASSES:
                symptoms.append(name)
            elif fact_type in SYMPTOM_FACT_CLASSES or fact_type in CAUSE_FACT_CLASSES:
                causes.append(name)

        causes = [
            cause for cause in causes
            if cause not in suppressed_causes
        ]
        surviving_causes = set(causes)

        recommendation_support: dict[str, list[set[str]]] = defaultdict(list)
        for exp in explanations:
            if exp["derived"] in recommendations:
                recommendation_support[exp["derived"]].append(
                    {
                        name.strip()
                        for name in exp["triggered_by"].split(",")
                        if name.strip()
                    }
                )

        filtered_recommendations: list[str] = []
        for recommendation in recommendations:
            supports = recommendation_support.get(recommendation, [])
            suppressed_only_supports = [
                bool(support & suppressed_causes)
                and not bool(support & surviving_causes)
                for support in supports
            ]
            if supports and all(suppressed_only_supports):
                continue
            filtered_recommendations.append(recommendation)
        recommendations = filtered_recommendations

        # ---- Confidence aggregation (REPORTING ONLY; not diagnostic logic) ----
        # The weighted model uses each rule's Explanation.confidence as the
        # rule weight, discounts it by the product of antecedent confidences,
        # and combines multiple supports for the same derived fact with
        # noisy-OR. It never feeds back into Experta rule firing.
        filtered_recommendation_names = set(recommendation_support) - set(recommendations)
        confidence = self._compute_reporting_confidence(
            explanations=explanations,
            suppressed_causes=suppressed_causes,
            filtered_recommendations=filtered_recommendation_names,
        )

        return DiagnosisResult(
            symptoms=sorted(set(symptoms)),
            causes=sorted(set(causes)),
            recommendations=sorted(set(recommendations)),
            explanations=sorted(explanations, key=lambda x: x["rule_id"]),
            confidence=confidence,
            conflicts=sorted(conflicts, key=lambda x: x["rule_id"]),
        )

    def _compute_reporting_confidence(
        self,
        explanations: list[dict[str, str]],
        suppressed_causes: set[str],
        filtered_recommendations: set[str],
    ) -> dict[str, float]:
        """Compute weighted, antecedent-aware confidence for reporting only."""
        excluded_facts = suppressed_causes | filtered_recommendations
        derived_to_rules: dict[str, list[tuple[float, list[str]]]] = defaultdict(list)

        for exp in explanations:
            derived = str(exp["derived"])
            if derived in excluded_facts:
                continue
            derived_to_rules[derived].append(
                (
                    _clamp_confidence(float(exp["confidence"])),
                    _split_antecedents(str(exp["triggered_by"])),
                )
            )

        if not derived_to_rules:
            return {}

        root_confidence = {
            name: _clamp_confidence(confidence)
            for name, confidence in getattr(
                self,
                "_root_confidence_by_fact",
                {},
            ).items()
        }
        values: dict[str, float] = {
            name: confidence
            for name, confidence in root_confidence.items()
        }
        for derived in derived_to_rules:
            values.setdefault(derived, 0.0)

        def antecedent_confidence(name: str, current_values: dict[str, float]) -> float:
            if name in suppressed_causes:
                return 0.0
            return current_values.get(name, root_confidence.get(name, 1.0))

        for _ in range(CONFIDENCE_FIXPOINT_MAX_ITERATIONS):
            next_values = dict(values)
            max_change = 0.0

            for derived, rule_supports in derived_to_rules.items():
                contributions: list[float] = []
                for rule_confidence, antecedents in rule_supports:
                    antecedent_product = 1.0
                    for antecedent in antecedents:
                        antecedent_product *= antecedent_confidence(
                            antecedent,
                            values,
                        )
                    contributions.append(rule_confidence * antecedent_product)

                propagated = _noisy_or(contributions)
                if derived in root_confidence:
                    propagated = max(root_confidence[derived], propagated)
                propagated = _clamp_confidence(propagated)
                max_change = max(
                    max_change,
                    abs(propagated - values.get(derived, 0.0)),
                )
                next_values[derived] = propagated

            values = next_values
            if max_change < CONFIDENCE_CONVERGENCE_EPSILON:
                break

        return {
            derived: round(_clamp_confidence(values[derived]), 3)
            for derived in sorted(derived_to_rules)
        }

    def inject_symptoms(
        self,
        symptom_names: list[str],
        root_confidence: dict[str, float] | None = None,
    ) -> None:
        """Declare symptom facts by class name.

        Parameters
        ----------
        symptom_names:
            List of Fact subclass names (e.g. ``["TrainingLossHigh"]``).
        root_confidence:
            Optional base confidence per injected fact. Missing roots default
            to 1.0, preserving deterministic scenario behavior.

        Raises
        ------
        ValueError
            If a name is not found in the fact registry.
        """
        self._injected_symptoms: set[str] = set(symptom_names)
        root_confidence = root_confidence or {}
        self._root_confidence_by_fact = {
            name: _clamp_confidence(root_confidence.get(name, 1.0))
            for name in symptom_names
        }
        for name in symptom_names:
            cls = FACT_CLASS_REGISTRY.get(name)
            if cls is None:
                raise ValueError(
                    f"Unknown fact class: '{name}'. "
                    f"Valid names: {sorted(FACT_CLASS_REGISTRY)}"
            )
            self.declare(cls())

    def run_scenario(
        self,
        symptom_names: list[str],
        root_confidence: dict[str, float] | None = None,
    ) -> "DiagnosisResult":
        """Convenience: reset, inject symptoms, run, return diagnosis.

        Parameters
        ----------
        symptom_names:
            Fact class names to assert as the starting symptom set.
        root_confidence:
            Optional base confidence for injected facts. Scenario callers that
            omit it get 1.0 for every injected fact.

        Returns
        -------
        DiagnosisResult
            Final inference result.
        """
        self.reset()
        self.inject_symptoms(symptom_names, root_confidence=root_confidence)
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
        extracted_roots = set(extraction.all_fact_names)
        root_confidence: dict[str, float] = {}
        for evidence in extraction.evidence:
            if evidence.fact_name not in extracted_roots:
                continue
            root_confidence[evidence.fact_name] = max(
                root_confidence.get(evidence.fact_name, 0.0),
                _clamp_confidence(evidence.confidence),
            )

        result = self.run_scenario(
            extraction.all_fact_names,
            root_confidence=root_confidence,
        )
        result.extraction = extraction
        return result

    def run_training_history_csv(self, path: str | Path) -> "DiagnosisResult":
        """CSV metric-history entry point.

        The metrics adapter converts observed training-history patterns into
        existing symptom Fact names and root confidences. It performs no root
        cause diagnosis; all causes and recommendations still come from the
        Experta rule set.
        """
        from engine.metrics.training_history_analyzer import (
            analyze_training_history_csv,
        )

        analysis = analyze_training_history_csv(path)
        return self._run_training_history_analysis(analysis)

    def run_training_history_text(
        self,
        csv_text: str,
        source: str = "uploaded CSV",
    ) -> "DiagnosisResult":
        """CSV content entry point for API/browser uploads."""
        from engine.metrics.training_history_analyzer import (
            analyze_training_history_text,
        )

        analysis = analyze_training_history_text(csv_text, source=source)
        return self._run_training_history_analysis(analysis)

    def _run_training_history_analysis(self, analysis) -> "DiagnosisResult":
        result = self.run_scenario(
            analysis.all_fact_names,
            root_confidence=analysis.root_confidence,
        )
        result.training_history = analysis
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
        Reporting-only confidence per derived fact. Each rule contribution is
        weighted by antecedent confidence and merged with noisy-OR.
    conflicts : list[dict[str, str]]
        Ordered list of resolved cause conflicts.
    """

    def __init__(
        self,
        symptoms: list[str],
        causes: list[str],
        recommendations: list[str],
        explanations: list[dict[str, str]],
        confidence: dict[str, float] | None = None,
        conflicts: list[dict[str, str]] | None = None,
    ) -> None:
        self.symptoms = symptoms
        self.causes = causes
        self.recommendations = recommendations
        self.explanations = explanations
        self.confidence = confidence or {}
        self.conflicts = conflicts or []
        # Optional NLP provenance (set by run_text); None for symptom-only runs.
        self.extraction = None
        # Optional metric provenance (set by run_training_history_*).
        self.training_history = None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DiagnosisResult("
            f"symptoms={self.symptoms}, "
            f"causes={self.causes}, "
            f"recommendations={self.recommendations})"
        )
