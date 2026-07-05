"""
optimization_rules.py
=====================
Experta rules for optimizer and numerical-stability recommendations.

Rule IDs: OPT_001 - OPT_009
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Symptoms
    GradientExplosion,
    GradientVanishing,
    OscillatingLoss,
    SlowConvergence,
    NaNLoss,
    # Causes
    LearningRateTooHigh,
    LearningRateTooLow,
    BadWeightInitialization,
    MomentumTooHigh,
    NumericalInstability,
    # Recommendations
    UseWeightClipping,
    UseProperInitialization,
    AddBatchNormalization,
    UseGradientClipping,
    # XAI
    Explanation,
)


class OptimizationRules(KnowledgeEngine):
    """Rule set covering optimizer and numerical-stability fixes."""

    # ------------------------------------------------------------------
    # OPT_001 - GradientExplosion -> UseWeightClipping
    # ------------------------------------------------------------------
    @Rule(
        GradientExplosion(),
        NOT(UseWeightClipping()),
    )
    def opt_001_explosion_clipping(self) -> None:
        self.declare(UseWeightClipping())
        self.declare(
            Explanation(
                rule_id="OPT_001",
                triggered_by="GradientExplosion",
                derived="UseWeightClipping",
                confidence=0.85,
                explanation=(
                    "Exploding gradients call for clipping gradient norms or "
                    "weights so that individual optimizer steps cannot run away."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_002 - BadWeightInitialization -> UseProperInitialization
    # ------------------------------------------------------------------
    @Rule(
        BadWeightInitialization(),
        NOT(UseProperInitialization()),
        salience=1,
    )
    def opt_002_init_scheme(self) -> None:
        self.declare(UseProperInitialization())
        self.declare(
            Explanation(
                rule_id="OPT_002",
                triggered_by="BadWeightInitialization",
                derived="UseProperInitialization",
                confidence=0.84,
                explanation=(
                    "Poor initialization is best addressed by switching to an "
                    "activation-aware scheme such as He or Xavier initialization."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_003 - GradientVanishing -> AddBatchNormalization
    # ------------------------------------------------------------------
    @Rule(
        GradientVanishing(),
        NOT(AddBatchNormalization()),
    )
    def opt_003_vanishing_batch_norm(self) -> None:
        self.declare(AddBatchNormalization())
        self.declare(
            Explanation(
                rule_id="OPT_003",
                triggered_by="GradientVanishing",
                derived="AddBatchNormalization",
                confidence=0.74,
                explanation=(
                    "Batch normalization can improve gradient flow by keeping "
                    "intermediate activations in a trainable numeric range."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_004 - LearningRateTooHigh + OscillatingLoss -> UseWeightClipping
    # ------------------------------------------------------------------
    @Rule(
        LearningRateTooHigh(),
        OscillatingLoss(),
        NOT(UseWeightClipping()),
    )
    def opt_004_high_lr_clip(self) -> None:
        self.declare(UseWeightClipping())
        self.declare(
            Explanation(
                rule_id="OPT_004",
                triggered_by="LearningRateTooHigh,OscillatingLoss",
                derived="UseWeightClipping",
                confidence=0.78,
                explanation=(
                    "When a high learning rate produces oscillation, clipping "
                    "acts as a guardrail while the learning rate is retuned."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_005 - LearningRateTooLow + SlowConvergence -> AddBatchNormalization
    # ------------------------------------------------------------------
    @Rule(
        LearningRateTooLow(),
        SlowConvergence(),
        NOT(AddBatchNormalization()),
    )
    def opt_005_slow_batch_norm(self) -> None:
        self.declare(AddBatchNormalization())
        self.declare(
            Explanation(
                rule_id="OPT_005",
                triggered_by="LearningRateTooLow,SlowConvergence",
                derived="AddBatchNormalization",
                confidence=0.58,
                explanation=(
                    "Slow convergence can improve when normalization reduces "
                    "internal scale drift and makes optimization less brittle."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_006 - GradientExplosion + GradientVanishing -> BadWeightInitialization
    # ------------------------------------------------------------------
    @Rule(
        GradientExplosion(),
        GradientVanishing(),
        NOT(BadWeightInitialization()),
        salience=1,
    )
    def opt_006_mixed_gradient_pathology(self) -> None:
        self.declare(BadWeightInitialization())
        self.declare(
            Explanation(
                rule_id="OPT_006",
                triggered_by="GradientExplosion,GradientVanishing",
                derived="BadWeightInitialization",
                confidence=0.86,
                explanation=(
                    "Seeing both exploding and vanishing gradients points to "
                    "poor scale propagation through the network, often caused "
                    "by an unsuitable initialization scheme."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_007 - NaNLoss -> NumericalInstability
    # ------------------------------------------------------------------
    @Rule(
        NaNLoss(),
        NOT(NumericalInstability()),
    )
    def opt_007_nan_instability(self) -> None:
        self.declare(NumericalInstability())
        self.declare(
            Explanation(
                rule_id="OPT_007",
                triggered_by="NaNLoss",
                derived="NumericalInstability",
                confidence=0.92,
                explanation=(
                    "A loss that becomes NaN or Inf indicates numerical "
                    "overflow or invalid values propagating through training."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_008 - NumericalInstability -> UseGradientClipping
    # ------------------------------------------------------------------
    @Rule(
        NumericalInstability(),
        NOT(UseGradientClipping()),
    )
    def opt_008_instability_clip(self) -> None:
        self.declare(UseGradientClipping())
        self.declare(
            Explanation(
                rule_id="OPT_008",
                triggered_by="NumericalInstability",
                derived="UseGradientClipping",
                confidence=0.85,
                explanation=(
                    "Clipping the global gradient norm bounds update sizes and "
                    "is an immediate mitigation for NaN-producing instability."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_009 - OscillatingLoss + NumericalInstability -> MomentumTooHigh
    # ------------------------------------------------------------------
    @Rule(
        OscillatingLoss(),
        NumericalInstability(),
        NOT(MomentumTooHigh()),
    )
    def opt_009_oscillation_momentum(self) -> None:
        self.declare(MomentumTooHigh())
        self.declare(
            Explanation(
                rule_id="OPT_009",
                triggered_by="OscillatingLoss,NumericalInstability",
                derived="MomentumTooHigh",
                confidence=0.6,
                explanation=(
                    "Oscillation that coincides with instability can stem from "
                    "an over-aggressive momentum term accumulating velocity and "
                    "overshooting minima."
                ),
            )
        )
