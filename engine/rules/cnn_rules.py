"""
cnn_rules.py
============
Experta rules specific to convolutional networks. Gated on ModelIsCNN.

Rule IDs: CNN_001 - CNN_003
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    ModelIsCNN,
    FeatureCollapse,
    ReceptiveFieldTooSmall,
    DeadReLUDetected,
    PoorGeneralization,
    # Recommendations
    IncreaseReceptiveField,
    UseLeakyReLU,
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
