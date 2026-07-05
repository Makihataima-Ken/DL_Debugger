"""
cnn_rules.py
============
Experta rules specific to convolutional networks. Gated on ModelIsCNN.

Rule IDs: CNN_001 - CNN_011
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    ModelIsCNN,
    FeatureCollapse,
    ReceptiveFieldTooSmall,
    DeadReLUDetected,
    PoorGeneralization,
    BatchNormTrainEvalMismatch,
    AggressivePoolingOrStride,
    InsufficientSpatialAugmentation,
    ChannelCollapse,
    BatchNormModeMismatch,
    StrideTooAggressive,
    SpatialAugmentationMissing,
    ChannelCollapseCause,
    # Recommendations
    IncreaseReceptiveField,
    UseLeakyReLU,
    FixBatchNormMomentum,
    ReduceStride,
    AddSpatialAugmentation,
    UseGlobalAveragePooling,
    # XAI
    Explanation,
)


class CNNRules(KnowledgeEngine):
    """CNN-specific diagnostics."""

    # CNN_001 - CNN + FeatureCollapse -> PoorGeneralization
    @Rule(
        ModelIsCNN(),
        FeatureCollapse(),
        NOT(PoorGeneralization()),
    )
    def cnn_001_feature_collapse(self) -> None:
        self.declare(PoorGeneralization())
        self.declare(
            Explanation(
                rule_id="CNN_001",
                triggered_by="ModelIsCNN,FeatureCollapse",
                derived="PoorGeneralization",
                confidence=0.7,
                explanation=(
                    "Collapsed/near-constant feature maps mean the network is "
                    "no longer extracting discriminative features, harming "
                    "generalisation; revisit normalisation and channel diversity."
                ),
            )
        )

    # CNN_002 - CNN + ReceptiveFieldTooSmall -> IncreaseReceptiveField
    @Rule(
        ModelIsCNN(),
        ReceptiveFieldTooSmall(),
        NOT(IncreaseReceptiveField()),
    )
    def cnn_002_receptive_field(self) -> None:
        self.declare(IncreaseReceptiveField())
        self.declare(
            Explanation(
                rule_id="CNN_002",
                triggered_by="ModelIsCNN,ReceptiveFieldTooSmall",
                derived="IncreaseReceptiveField",
                confidence=0.8,
                explanation=(
                    "A receptive field smaller than the target structures "
                    "prevents the model from integrating global context; add "
                    "depth, dilation, or pooling/striding."
                ),
            )
        )

    # CNN_003 - CNN + DeadReLUDetected -> UseLeakyReLU
    @Rule(
        ModelIsCNN(),
        DeadReLUDetected(),
        NOT(UseLeakyReLU()),
    )
    def cnn_003_dead_relu(self) -> None:
        self.declare(UseLeakyReLU())
        self.declare(
            Explanation(
                rule_id="CNN_003",
                triggered_by="ModelIsCNN,DeadReLUDetected",
                derived="UseLeakyReLU",
                confidence=0.82,
                explanation=(
                    "Dead ReLUs output zero for all inputs and stop learning; "
                    "switching to LeakyReLU/GELU/ELU (and lowering the LR) "
                    "restores gradient flow."
                ),
            )
        )

    # CNN_004 - CNN + BatchNormTrainEvalMismatch -> BatchNormModeMismatch
    @Rule(
        ModelIsCNN(),
        BatchNormTrainEvalMismatch(),
        NOT(BatchNormModeMismatch()),
    )
    def cnn_004_batch_norm_mode(self) -> None:
        self.declare(BatchNormModeMismatch())
        self.declare(
            Explanation(
                rule_id="CNN_004",
                triggered_by="ModelIsCNN,BatchNormTrainEvalMismatch",
                derived="BatchNormModeMismatch",
                confidence=0.88,
                explanation=(
                    "A train/eval behavior gap in a CNN commonly points to "
                    "BatchNorm running statistics or mode handling being wrong."
                ),
            )
        )

    # CNN_005 - CNN + BatchNormModeMismatch -> FixBatchNormMomentum
    @Rule(
        ModelIsCNN(),
        BatchNormModeMismatch(),
        NOT(FixBatchNormMomentum()),
    )
    def cnn_005_fix_batch_norm(self) -> None:
        self.declare(FixBatchNormMomentum())
        self.declare(
            Explanation(
                rule_id="CNN_005",
                triggered_by="ModelIsCNN,BatchNormModeMismatch",
                derived="FixBatchNormMomentum",
                confidence=0.86,
                explanation=(
                    "Fixing BatchNorm momentum, running-stat updates, and "
                    "train/eval mode usage directly addresses this mismatch."
                ),
            )
        )

    # CNN_006 - CNN + AggressivePoolingOrStride -> StrideTooAggressive
    @Rule(
        ModelIsCNN(),
        AggressivePoolingOrStride(),
        NOT(StrideTooAggressive()),
    )
    def cnn_006_stride_too_aggressive(self) -> None:
        self.declare(StrideTooAggressive())
        self.declare(
            Explanation(
                rule_id="CNN_006",
                triggered_by="ModelIsCNN,AggressivePoolingOrStride",
                derived="StrideTooAggressive",
                confidence=0.84,
                explanation=(
                    "Aggressive early stride or pooling discards spatial detail "
                    "before later convolutional layers can use it."
                ),
            )
        )

    # CNN_007 - CNN + StrideTooAggressive -> ReduceStride
    @Rule(
        ModelIsCNN(),
        StrideTooAggressive(),
        NOT(ReduceStride()),
    )
    def cnn_007_reduce_stride(self) -> None:
        self.declare(ReduceStride())
        self.declare(
            Explanation(
                rule_id="CNN_007",
                triggered_by="ModelIsCNN,StrideTooAggressive",
                derived="ReduceStride",
                confidence=0.83,
                explanation=(
                    "Reducing stride or replacing early pooling preserves "
                    "spatial resolution for deeper feature extraction."
                ),
            )
        )

    # CNN_008 - CNN + InsufficientSpatialAugmentation -> SpatialAugmentationMissing
    @Rule(
        ModelIsCNN(),
        InsufficientSpatialAugmentation(),
        NOT(SpatialAugmentationMissing()),
    )
    def cnn_008_spatial_aug_missing(self) -> None:
        self.declare(SpatialAugmentationMissing())
        self.declare(
            Explanation(
                rule_id="CNN_008",
                triggered_by="ModelIsCNN,InsufficientSpatialAugmentation",
                derived="SpatialAugmentationMissing",
                confidence=0.82,
                explanation=(
                    "Weak spatial augmentation leaves CNNs brittle to shifts, "
                    "crops, rotations, and scale changes in validation or test data."
                ),
            )
        )

    # CNN_009 - CNN + SpatialAugmentationMissing -> AddSpatialAugmentation
    @Rule(
        ModelIsCNN(),
        SpatialAugmentationMissing(),
        NOT(AddSpatialAugmentation()),
    )
    def cnn_009_add_spatial_aug(self) -> None:
        self.declare(AddSpatialAugmentation())
        self.declare(
            Explanation(
                rule_id="CNN_009",
                triggered_by="ModelIsCNN,SpatialAugmentationMissing",
                derived="AddSpatialAugmentation",
                confidence=0.84,
                explanation=(
                    "Spatial augmentation teaches invariance to translation, "
                    "crop, scale, and rotation changes that CNNs must handle."
                ),
            )
        )

    # CNN_010 - CNN + ChannelCollapse -> ChannelCollapseCause
    @Rule(
        ModelIsCNN(),
        ChannelCollapse(),
        NOT(ChannelCollapseCause()),
    )
    def cnn_010_channel_collapse(self) -> None:
        self.declare(ChannelCollapseCause())
        self.declare(
            Explanation(
                rule_id="CNN_010",
                triggered_by="ModelIsCNN,ChannelCollapse",
                derived="ChannelCollapseCause",
                confidence=0.83,
                explanation=(
                    "Collapsed channels indicate the convolutional stack is "
                    "learning redundant filters instead of diverse visual features."
                ),
            )
        )

    # CNN_011 - CNN + ChannelCollapseCause -> UseGlobalAveragePooling
    @Rule(
        ModelIsCNN(),
        ChannelCollapseCause(),
        NOT(UseGlobalAveragePooling()),
    )
    def cnn_011_global_average_pool(self) -> None:
        self.declare(UseGlobalAveragePooling())
        self.declare(
            Explanation(
                rule_id="CNN_011",
                triggered_by="ModelIsCNN,ChannelCollapseCause",
                derived="UseGlobalAveragePooling",
                confidence=0.7,
                explanation=(
                    "Global average pooling can reduce brittle classifier heads "
                    "and encourages channels to carry class-level evidence."
                ),
            )
        )
