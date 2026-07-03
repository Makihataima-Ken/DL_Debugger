"""
optimizer_rules.py
==================
Experta rules for optimizer-specific interactions and configuration issues.

Rule IDs: OPTZ_001 - OPTZ_006
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Context
    OptimizerIsAdam,
    OptimizerIsSGD,
    # Symptoms
    CoupledWeightDecayUsed,
    MissingMomentum,
    AdamHighLearningRateInstability,
    # Causes
    WeightDecayCoupledWithAdam,
    MomentumMisconfigured,
    AdamLearningRateTooHigh,
    # Recommendations
    UseAdamW,
    EnableNesterovMomentum,
    UseLearningRateWarmup,
    # XAI
    Explanation,
)


class OptimizerInteractionRules(KnowledgeEngine):
    """Rule set covering optimizer-specific interactions."""

    # OPTZ_001 - Adam + coupled L2 decay -> WeightDecayCoupledWithAdam
    @Rule(
        OptimizerIsAdam(),
        CoupledWeightDecayUsed(),
        NOT(WeightDecayCoupledWithAdam()),
    )
    def optz_001_adam_coupled_decay(self) -> None:
        self.declare(WeightDecayCoupledWithAdam())
        self.declare(
            Explanation(
                rule_id="OPTZ_001",
                triggered_by="OptimizerIsAdam,CoupledWeightDecayUsed",
                derived="WeightDecayCoupledWithAdam",
                confidence=0.88,
                explanation=(
                    "Coupled L2 decay interacts poorly with Adam's adaptive "
                    "updates; decoupled weight decay is the safer formulation."
                ),
            )
        )

    # OPTZ_002 - Adam + WeightDecayCoupledWithAdam -> UseAdamW
    @Rule(
        OptimizerIsAdam(),
        WeightDecayCoupledWithAdam(),
        NOT(UseAdamW()),
    )
    def optz_002_use_adamw(self) -> None:
        self.declare(UseAdamW())
        self.declare(
            Explanation(
                rule_id="OPTZ_002",
                triggered_by="OptimizerIsAdam,WeightDecayCoupledWithAdam",
                derived="UseAdamW",
                confidence=0.9,
                explanation=(
                    "AdamW decouples weight decay from adaptive moment updates, "
                    "which directly fixes Adam plus coupled L2 regularization."
                ),
            )
        )

    # OPTZ_003 - SGD + missing momentum -> MomentumMisconfigured
    @Rule(
        OptimizerIsSGD(),
        MissingMomentum(),
        NOT(MomentumMisconfigured()),
    )
    def optz_003_sgd_missing_momentum(self) -> None:
        self.declare(MomentumMisconfigured())
        self.declare(
            Explanation(
                rule_id="OPTZ_003",
                triggered_by="OptimizerIsSGD,MissingMomentum",
                derived="MomentumMisconfigured",
                confidence=0.82,
                explanation=(
                    "SGD without useful momentum often converges slowly on "
                    "deep networks because it cannot smooth noisy gradients."
                ),
            )
        )

    # OPTZ_004 - SGD + MomentumMisconfigured -> EnableNesterovMomentum
    @Rule(
        OptimizerIsSGD(),
        MomentumMisconfigured(),
        NOT(EnableNesterovMomentum()),
    )
    def optz_004_enable_nesterov(self) -> None:
        self.declare(EnableNesterovMomentum())
        self.declare(
            Explanation(
                rule_id="OPTZ_004",
                triggered_by="OptimizerIsSGD,MomentumMisconfigured",
                derived="EnableNesterovMomentum",
                confidence=0.77,
                explanation=(
                    "Nesterov momentum gives SGD a lookahead correction that "
                    "usually improves convergence on curved loss surfaces."
                ),
            )
        )

    # OPTZ_005 - Adam + high-LR instability -> AdamLearningRateTooHigh
    @Rule(
        OptimizerIsAdam(),
        AdamHighLearningRateInstability(),
        NOT(AdamLearningRateTooHigh()),
    )
    def optz_005_adam_high_lr(self) -> None:
        self.declare(AdamLearningRateTooHigh())
        self.declare(
            Explanation(
                rule_id="OPTZ_005",
                triggered_by="OptimizerIsAdam,AdamHighLearningRateInstability",
                derived="AdamLearningRateTooHigh",
                confidence=0.86,
                explanation=(
                    "Adam can still become unstable when its base learning rate "
                    "is too high, especially early in training before moments settle."
                ),
            )
        )

    # OPTZ_006 - Adam + AdamLearningRateTooHigh -> UseLearningRateWarmup
    @Rule(
        OptimizerIsAdam(),
        AdamLearningRateTooHigh(),
        NOT(UseLearningRateWarmup()),
    )
    def optz_006_warmup_adam(self) -> None:
        self.declare(UseLearningRateWarmup())
        self.declare(
            Explanation(
                rule_id="OPTZ_006",
                triggered_by="OptimizerIsAdam,AdamLearningRateTooHigh",
                derived="UseLearningRateWarmup",
                confidence=0.81,
                explanation=(
                    "Learning-rate warmup prevents Adam from taking large, "
                    "poorly calibrated updates before its moment estimates stabilize."
                ),
            )
        )
