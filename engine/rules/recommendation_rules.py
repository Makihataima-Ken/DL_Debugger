"""
recommendation_rules.py
=======================
Experta rules that translate inferred *causes* into concrete *recommendations*.

Rule IDs: REC_001 - REC_020
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Causes
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
    PoorGeneralization,
    OverfittingObserved,
    BatchSizeTooLarge,
    BatchSizeTooSmall,
    MomentumTooHigh,
    WeightDecayTooHigh,
    # Recommendations
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
    # XAI
    Explanation,
)


class RecommendationRules(KnowledgeEngine):
    """Rule set: cause → recommendation translations."""

    # ------------------------------------------------------------------
    # REC_001 – LearningRateTooHigh → ReduceLearningRate
    # ------------------------------------------------------------------
    @Rule(LearningRateTooHigh(), NOT(ReduceLearningRate()))
    def rec_001_reduce_lr(self) -> None:
        """High learning rate calls for a reduction."""
        self.declare(ReduceLearningRate())
        self.declare(
            Explanation(
                rule_id="REC_001",
                triggered_by="LearningRateTooHigh",
                derived="ReduceLearningRate",
                explanation=(
                    "Reducing the learning rate (e.g., by a factor of 5–10) "
                    "is the primary action when the optimiser overshoots "
                    "and causes instability or oscillation."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_002 – LearningRateTooLow → IncreaseLearningRate
    # ------------------------------------------------------------------
    @Rule(LearningRateTooLow(), NOT(IncreaseLearningRate()))
    def rec_002_increase_lr(self) -> None:
        """Low learning rate calls for an increase."""
        self.declare(IncreaseLearningRate())
        self.declare(
            Explanation(
                rule_id="REC_002",
                triggered_by="LearningRateTooLow",
                derived="IncreaseLearningRate",
                explanation=(
                    "Increasing the learning rate, possibly combined with a "
                    "warm-up schedule, lets the optimiser take larger steps "
                    "and converge in fewer epochs."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_003 – ModelTooComplex → AddDropout
    # ------------------------------------------------------------------
    @Rule(ModelTooComplex(), NOT(AddDropout()))
    def rec_003_complex_dropout(self) -> None:
        """Complex model → add dropout as regularisation."""
        self.declare(AddDropout())
        self.declare(
            Explanation(
                rule_id="REC_003",
                triggered_by="ModelTooComplex",
                derived="AddDropout",
                explanation=(
                    "Dropout reduces effective model complexity at each "
                    "training step, combating the overfitting tendency of "
                    "over-parameterised architectures."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_004 – ModelTooComplex → UseEarlyStopping
    # ------------------------------------------------------------------
    @Rule(ModelTooComplex(), NOT(UseEarlyStopping()))
    def rec_004_complex_early_stop(self) -> None:
        """Complex model → use early stopping."""
        self.declare(UseEarlyStopping())
        self.declare(
            Explanation(
                rule_id="REC_004",
                triggered_by="ModelTooComplex",
                derived="UseEarlyStopping",
                explanation=(
                    "Early stopping halts training when validation performance "
                    "begins to degrade, preventing the model from overfitting "
                    "regardless of its capacity."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_005 – ModelTooSimple → IncreaseModelComplexity
    # ------------------------------------------------------------------
    @Rule(ModelTooSimple(), NOT(IncreaseModelComplexity()))
    def rec_005_simple_increase(self) -> None:
        """Simple model → increase model complexity."""
        self.declare(IncreaseModelComplexity())
        self.declare(
            Explanation(
                rule_id="REC_005",
                triggered_by="ModelTooSimple",
                derived="IncreaseModelComplexity",
                explanation=(
                    "Adding layers or increasing hidden dimensions gives the "
                    "model the expressiveness needed to fit complex data "
                    "distributions."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_006 – InsufficientRegularization → AddDropout
    # ------------------------------------------------------------------
    @Rule(InsufficientRegularization(), NOT(AddDropout()))
    def rec_006_no_reg_dropout(self) -> None:
        """Insufficient regularisation → add dropout."""
        self.declare(AddDropout())
        self.declare(
            Explanation(
                rule_id="REC_006",
                triggered_by="InsufficientRegularization",
                derived="AddDropout",
                explanation=(
                    "Adding dropout layers (typically 0.2–0.5) introduces "
                    "regularisation that curbs the model's tendency to "
                    "memorise training data."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_007 – InsufficientRegularization → UseEarlyStopping
    # ------------------------------------------------------------------
    @Rule(InsufficientRegularization(), NOT(UseEarlyStopping()))
    def rec_007_no_reg_early_stop(self) -> None:
        """Insufficient regularisation → add early stopping."""
        self.declare(UseEarlyStopping())
        self.declare(
            Explanation(
                rule_id="REC_007",
                triggered_by="InsufficientRegularization",
                derived="UseEarlyStopping",
                explanation=(
                    "Early stopping is a lightweight regulariser that "
                    "complements dropout and weight decay by preventing "
                    "indefinite training past the generalisation peak."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_008 – ExcessiveRegularization → RemoveDropout
    # ------------------------------------------------------------------
    @Rule(ExcessiveRegularization(), NOT(RemoveDropout()))
    def rec_008_excess_reg_remove_dropout(self) -> None:
        """Excessive regularisation → remove or reduce dropout."""
        self.declare(RemoveDropout())
        self.declare(
            Explanation(
                rule_id="REC_008",
                triggered_by="ExcessiveRegularization",
                derived="RemoveDropout",
                explanation=(
                    "Removing dropout layers or reducing their rate allows "
                    "the model to utilise its full capacity and escape "
                    "under-fitting caused by over-regularisation."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_009 – BadWeightInitialization → UseProperInitialization
    # ------------------------------------------------------------------
    @Rule(BadWeightInitialization(), NOT(UseProperInitialization()))
    def rec_009_bad_init(self) -> None:
        """Bad weight initialisation → use proper initialisation scheme."""
        self.declare(UseProperInitialization())
        self.declare(
            Explanation(
                rule_id="REC_009",
                triggered_by="BadWeightInitialization",
                derived="UseProperInitialization",
                explanation=(
                    "Replacing ad-hoc initialisation with He (ReLU networks) "
                    "or Glorot (tanh/sigmoid) ensures that gradients and "
                    "activations are well-scaled from the very first step."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_010 – PoorDataQuality → ImproveLabelQuality
    # ------------------------------------------------------------------
    @Rule(PoorDataQuality(), NOT(ImproveLabelQuality()))
    def rec_010_data_quality(self) -> None:
        """Poor data quality → improve label quality."""
        self.declare(ImproveLabelQuality())
        self.declare(
            Explanation(
                rule_id="REC_010",
                triggered_by="PoorDataQuality",
                derived="ImproveLabelQuality",
                explanation=(
                    "Auditing labels, removing ambiguous examples, and "
                    "re-annotating noisy samples directly addresses the root "
                    "cause of data-quality-driven training failures."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_011 – DataImbalance → RebalanceClasses
    # ------------------------------------------------------------------
    @Rule(DataImbalance(), NOT(RebalanceClasses()))
    def rec_011_imbalance_rebalance(self) -> None:
        """Data imbalance → rebalance classes."""
        self.declare(RebalanceClasses())
        self.declare(
            Explanation(
                rule_id="REC_011",
                triggered_by="DataImbalance",
                derived="RebalanceClasses",
                explanation=(
                    "Oversampling minority classes (SMOTE), undersampling "
                    "majority classes, or using a class-weighted loss ensures "
                    "the model treats all classes fairly during training."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_012 – DataImbalance → UseWeightedLoss
    # ------------------------------------------------------------------
    @Rule(DataImbalance(), NOT(UseWeightedLoss()))
    def rec_012_imbalance_weighted_loss(self) -> None:
        """Data imbalance → use a class-weighted loss function."""
        self.declare(UseWeightedLoss())
        self.declare(
            Explanation(
                rule_id="REC_012",
                triggered_by="DataImbalance",
                derived="UseWeightedLoss",
                explanation=(
                    "Weighting each class's contribution to the loss inversely "
                    "proportional to its frequency penalises errors on minority "
                    "classes more heavily, correcting the optimiser's bias."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_013 – DistributionShift → CollectDomainData
    # ------------------------------------------------------------------
    @Rule(DistributionShift(), NOT(CollectDomainData()))
    def rec_013_shift_domain_data(self) -> None:
        """Distribution shift → collect in-domain data."""
        self.declare(CollectDomainData())
        self.declare(
            Explanation(
                rule_id="REC_013",
                triggered_by="DistributionShift",
                derived="CollectDomainData",
                explanation=(
                    "The most direct fix for distribution shift is to collect "
                    "labelled examples from the target domain and include them "
                    "in training or fine-tuning."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_014 – DistributionShift → ApplyDataAugmentation
    # ------------------------------------------------------------------
    @Rule(DistributionShift(), NOT(ApplyDataAugmentation()))
    def rec_014_shift_augment(self) -> None:
        """Distribution shift → apply augmentation to bridge domains."""
        self.declare(ApplyDataAugmentation())
        self.declare(
            Explanation(
                rule_id="REC_014",
                triggered_by="DistributionShift",
                derived="ApplyDataAugmentation",
                explanation=(
                    "Domain-bridging augmentations (style transfer, domain "
                    "randomisation, domain-specific transforms) can help "
                    "narrow the gap between training and deployment distributions."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_015 – PoorGeneralization → InspectDataPipeline
    # ------------------------------------------------------------------
    @Rule(PoorGeneralization(), NOT(InspectDataPipeline()))
    def rec_015_poor_gen_pipeline(self) -> None:
        """Poor generalisation → inspect data pipeline for leakage or errors."""
        self.declare(InspectDataPipeline())
        self.declare(
            Explanation(
                rule_id="REC_015",
                triggered_by="PoorGeneralization",
                derived="InspectDataPipeline",
                explanation=(
                    "Poor generalisation sometimes traces back to subtle "
                    "preprocessing bugs, train/test contamination, or "
                    "incorrect normalisation statistics; auditing the pipeline "
                    "is an essential diagnostic step."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_016 – OverfittingObserved → IncreaseDatasetSize
    # ------------------------------------------------------------------
    @Rule(OverfittingObserved(), NOT(IncreaseDatasetSize()))
    def rec_016_overfit_more_data(self) -> None:
        """Overfitting → collect more training data."""
        self.declare(IncreaseDatasetSize())
        self.declare(
            Explanation(
                rule_id="REC_016",
                triggered_by="OverfittingObserved",
                derived="IncreaseDatasetSize",
                explanation=(
                    "More training data directly reduces overfitting: "
                    "a larger dataset makes memorisation harder and provides "
                    "richer generalisation signal regardless of model size."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_017 - BatchSizeTooLarge -> ReduceBatchSize
    # ------------------------------------------------------------------
    @Rule(BatchSizeTooLarge(), NOT(ReduceBatchSize()))
    def rec_017_reduce_batch_size(self) -> None:
        self.declare(ReduceBatchSize())
        self.declare(
            Explanation(
                rule_id="REC_017",
                triggered_by="BatchSizeTooLarge",
                derived="ReduceBatchSize",
                confidence=0.8,
                explanation=(
                    "Reducing an overly large batch restores gradient noise "
                    "that can help optimization and generalization."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_018 - BatchSizeTooSmall -> IncreaseBatchSize
    # ------------------------------------------------------------------
    @Rule(BatchSizeTooSmall(), NOT(IncreaseBatchSize()))
    def rec_018_increase_batch_size(self) -> None:
        self.declare(IncreaseBatchSize())
        self.declare(
            Explanation(
                rule_id="REC_018",
                triggered_by="BatchSizeTooSmall",
                derived="IncreaseBatchSize",
                confidence=0.8,
                explanation=(
                    "Increasing a too-small batch reduces gradient variance "
                    "and can stabilize noisy optimizer steps."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_019 - MomentumTooHigh -> ReduceMomentum
    # ------------------------------------------------------------------
    @Rule(MomentumTooHigh(), NOT(ReduceMomentum()))
    def rec_019_reduce_momentum(self) -> None:
        self.declare(ReduceMomentum())
        self.declare(
            Explanation(
                rule_id="REC_019",
                triggered_by="MomentumTooHigh",
                derived="ReduceMomentum",
                confidence=0.78,
                explanation=(
                    "Reducing momentum lowers accumulated velocity and helps "
                    "stop oscillatory optimizer overshoot."
                ),
            )
        )

    # ------------------------------------------------------------------
    # REC_020 - WeightDecayTooHigh -> ReduceWeightDecay
    # ------------------------------------------------------------------
    @Rule(WeightDecayTooHigh(), NOT(ReduceWeightDecay()))
    def rec_020_reduce_weight_decay(self) -> None:
        self.declare(ReduceWeightDecay())
        self.declare(
            Explanation(
                rule_id="REC_020",
                triggered_by="WeightDecayTooHigh",
                derived="ReduceWeightDecay",
                confidence=0.82,
                explanation=(
                    "Reducing excessive weight decay restores capacity that "
                    "was being suppressed by too much L2 regularization."
                ),
            )
        )
