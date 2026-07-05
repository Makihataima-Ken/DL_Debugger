"""
architecture_rules.py
=====================
Experta rules that infer architectural causes from observed symptoms and
derived causes.

Rule IDs: ARCH_001 – ARCH_008
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Symptoms
    OverfittingObserved,
    UnderfittingObserved,
    SmallDataset,
    LargeDataset,
    GradientVanishing,
    DeadReLUDetected,
    # Causes
    ModelTooComplex,
    ModelTooSimple,
    InsufficientRegularization,
    ExcessiveRegularization,
    # Recommendations (directly asserted by architecture rules)
    ReduceModelComplexity,
    IncreaseModelComplexity,
    AddDropout,
    AddBatchNormalization,
    ApplyDataAugmentation,
    IncreaseDatasetSize,
    ReduceRegularization,
    UseLeakyReLU,
    # XAI
    Explanation,
)


class ArchitectureRules(KnowledgeEngine):
    """Rule set covering architecture-level diagnostics and recommendations."""

    # ------------------------------------------------------------------
    # ARCH_001 – ModelTooComplex → ReduceModelComplexity
    # ------------------------------------------------------------------
    @Rule(
        ModelTooComplex(),
        NOT(ReduceModelComplexity()),
    )
    def arch_001_complex_reduce(self) -> None:
        """A model that is too complex should be simplified."""
        self.declare(ReduceModelComplexity())
        self.declare(
            Explanation(
                rule_id="ARCH_001",
                triggered_by="ModelTooComplex",
                derived="ReduceModelComplexity",
                explanation=(
                    "Reducing model depth, width, or the number of parameters "
                    "directly addresses overfitting caused by excessive model "
                    "capacity relative to the dataset size."
                ),
            )
        )

    # ------------------------------------------------------------------
    # ARCH_002 – ModelTooSimple → IncreaseModelComplexity
    # ------------------------------------------------------------------
    @Rule(
        ModelTooSimple(),
        NOT(IncreaseModelComplexity()),
    )
    def arch_002_simple_increase(self) -> None:
        """A model that is too simple should be made more expressive."""
        self.declare(IncreaseModelComplexity())
        self.declare(
            Explanation(
                rule_id="ARCH_002",
                triggered_by="ModelTooSimple",
                derived="IncreaseModelComplexity",
                explanation=(
                    "Adding layers, increasing hidden units, or adopting a "
                    "more expressive architecture gives the model the capacity "
                    "it needs to capture the target mapping."
                ),
            )
        )

    # ------------------------------------------------------------------
    # ARCH_003 – OverfittingObserved + SmallDataset → ApplyDataAugmentation
    # ------------------------------------------------------------------
    @Rule(
        OverfittingObserved(),
        SmallDataset(),
        NOT(ApplyDataAugmentation()),
    )
    def arch_003_overfit_small_augment(self) -> None:
        """Overfitting on a small dataset can be partially mitigated by
        data augmentation, which increases the effective training set size.
        """
        self.declare(ApplyDataAugmentation())
        self.declare(
            Explanation(
                rule_id="ARCH_003",
                triggered_by="OverfittingObserved,SmallDataset",
                derived="ApplyDataAugmentation",
                explanation=(
                    "When a small dataset leads to overfitting, data "
                    "augmentation creates synthetic training examples that "
                    "expose the model to a wider variety of inputs, reducing "
                    "memorisation."
                ),
            )
        )

    # ------------------------------------------------------------------
    # ARCH_004 – SmallDataset + UnderfittingObserved → IncreaseDatasetSize
    # ------------------------------------------------------------------
    @Rule(
        SmallDataset(),
        UnderfittingObserved(),
        NOT(IncreaseDatasetSize()),
    )
    def arch_004_small_underfit_data(self) -> None:
        """Underfitting on a small dataset calls for more data before
        architecture changes are made.
        """
        self.declare(IncreaseDatasetSize())
        self.declare(
            Explanation(
                rule_id="ARCH_004",
                triggered_by="SmallDataset,UnderfittingObserved",
                derived="IncreaseDatasetSize",
                explanation=(
                    "A small dataset combined with underfitting suggests the "
                    "model has not seen enough examples to learn the task. "
                    "Collecting more labelled data is the highest-leverage fix."
                ),
            )
        )

    # ------------------------------------------------------------------
    # ARCH_005 – InsufficientRegularization → AddDropout
    # ------------------------------------------------------------------
    @Rule(
        InsufficientRegularization(),
        NOT(AddDropout()),
    )
    def arch_005_no_reg_dropout(self) -> None:
        """Insufficient regularisation should be addressed by adding dropout."""
        self.declare(AddDropout())
        self.declare(
            Explanation(
                rule_id="ARCH_005",
                triggered_by="InsufficientRegularization",
                derived="AddDropout",
                explanation=(
                    "Dropout randomly deactivates neurons during training, "
                    "preventing co-adaptation and acting as an effective "
                    "regulariser to reduce overfitting."
                ),
            )
        )

    # ------------------------------------------------------------------
    # ARCH_006 – ExcessiveRegularization → ReduceRegularization
    # ------------------------------------------------------------------
    @Rule(
        ExcessiveRegularization(),
        NOT(ReduceRegularization()),
    )
    def arch_006_excess_reg_reduce(self) -> None:
        """Excessive regularisation should be reduced to restore model capacity."""
        self.declare(ReduceRegularization())
        self.declare(
            Explanation(
                rule_id="ARCH_006",
                triggered_by="ExcessiveRegularization",
                derived="ReduceRegularization",
                explanation=(
                    "Over-regularisation suppresses the model's ability to "
                    "fit the data. Lowering dropout rates or L2 penalties "
                    "restores the model's learning capacity."
                ),
            )
        )

    # ------------------------------------------------------------------
    # ARCH_007 – GradientVanishing + LargeDataset → AddBatchNormalization
    # ------------------------------------------------------------------
    @Rule(
        GradientVanishing(),
        LargeDataset(),
        NOT(AddBatchNormalization()),
    )
    def arch_007_vanish_large_bn(self) -> None:
        """On large datasets, batch normalisation is especially effective at
        stabilising gradient flow through deep architectures.
        """
        self.declare(AddBatchNormalization())
        self.declare(
            Explanation(
                rule_id="ARCH_007",
                triggered_by="GradientVanishing,LargeDataset",
                derived="AddBatchNormalization",
                explanation=(
                    "With a large dataset, batch normalisation is a scalable "
                    "and highly effective remedy for vanishing gradients: it "
                    "re-centres and re-scales activations at each layer, "
                    "preserving gradient magnitudes across depth."
                ),
            )
        )

    # ------------------------------------------------------------------
    # ARCH_008 – OverfittingObserved + LargeDataset → AddDropout
    # ------------------------------------------------------------------
    @Rule(
        OverfittingObserved(),
        LargeDataset(),
        NOT(AddDropout()),
    )
    def arch_008_overfit_large_dropout(self) -> None:
        """Even with a large dataset, overfitting can occur in very deep or
        wide models; dropout is an efficient regulariser in this regime.
        """
        self.declare(AddDropout())
        self.declare(
            Explanation(
                rule_id="ARCH_008",
                triggered_by="OverfittingObserved,LargeDataset",
                derived="AddDropout",
                explanation=(
                    "Large models can overfit even large datasets. Dropout "
                    "provides an inexpensive regularisation signal at each "
                    "forward pass without reducing the model's representational "
                    "capacity at inference time."
                ),
            )
        )

    # ------------------------------------------------------------------
    # ARCH_009 – DeadReLUDetected -> UseLeakyReLU (architecture-agnostic)
    # ------------------------------------------------------------------
    @Rule(
        DeadReLUDetected(),
        NOT(UseLeakyReLU()),
    )
    def arch_009_dead_relu(self) -> None:
        """Dead ReLUs anywhere warrant a smoother activation function."""
        self.declare(UseLeakyReLU())
        self.declare(
            Explanation(
                rule_id="ARCH_009",
                triggered_by="DeadReLUDetected",
                derived="UseLeakyReLU",
                confidence=0.78,
                explanation=(
                    "A large fraction of permanently-zero ReLU units indicates "
                    "the dying-ReLU problem; a leaky/parametric activation keeps "
                    "a small gradient alive for negative pre-activations."
                ),
            )
        )
