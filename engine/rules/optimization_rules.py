"""
optimization_rules.py
=====================
Experta rules related to optimiser-level diagnostics (gradient behaviour,
learning-rate schedules, weight updates).

Rule IDs: OPT_001 – OPT_006
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Symptoms / derived symptoms
    GradientExplosion,
    GradientVanishing,
    OscillatingLoss,
    SlowConvergence,
    # Causes
    LearningRateTooHigh,
    LearningRateTooLow,
    BadWeightInitialization,
    # Recommendations (emitted directly from optimiser rules)
    UseWeightClipping,
    UseProperInitialization,
    AddBatchNormalization,
    # XAI
    Explanation,
)


class OptimizationRules(KnowledgeEngine):
    """Rule set covering optimisation-phase diagnostics."""

    # ------------------------------------------------------------------
    # OPT_001 – GradientExplosion → UseWeightClipping
    # ------------------------------------------------------------------
    @Rule(
        GradientExplosion(),
        NOT(UseWeightClipping()),
    )
    def opt_001_explosion_clipping(self) -> None:
        """Exploding gradients call for gradient / weight clipping as an
        immediate stabilisation measure.
        """
        self.declare(UseWeightClipping())
        self.declare(
            Explanation(
                rule_id="OPT_001",
                triggered_by="GradientExplosion",
                derived="UseWeightClipping",
                explanation=(
                    "Gradient clipping caps the norm of the gradient vector "
                    "at each step, preventing runaway updates that cause "
                    "exploding gradient failures."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_002 – BadWeightInitialization → UseProperInitialization
    # ------------------------------------------------------------------
    @Rule(
        BadWeightInitialization(),
        NOT(UseProperInitialization()),
    )
    def opt_002_bad_init_fix(self) -> None:
        """Poor weight initialisation should be corrected with an
        appropriate scheme (He, Xavier, Glorot).
        """
        self.declare(UseProperInitialization())
        self.declare(
            Explanation(
                rule_id="OPT_002",
                triggered_by="BadWeightInitialization",
                derived="UseProperInitialization",
                explanation=(
                    "Switching to a principled initialisation scheme "
                    "(He for ReLU networks, Glorot/Xavier for tanh/sigmoid) "
                    "ensures that activations and gradients are well-scaled "
                    "at the start of training."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_003 – GradientVanishing → AddBatchNormalization
    # ------------------------------------------------------------------
    @Rule(
        GradientVanishing(),
        NOT(AddBatchNormalization()),
    )
    def opt_003_vanishing_bn(self) -> None:
        """Batch normalisation helps mitigate vanishing gradients by keeping
        layer inputs normalised throughout the network.
        """
        self.declare(AddBatchNormalization())
        self.declare(
            Explanation(
                rule_id="OPT_003",
                triggered_by="GradientVanishing",
                derived="AddBatchNormalization",
                explanation=(
                    "Batch normalisation standardises layer inputs, reducing "
                    "internal covariate shift and preserving gradient flow "
                    "through deep networks where vanishing gradients occur."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_004 – LearningRateTooHigh + OscillatingLoss → UseWeightClipping
    # ------------------------------------------------------------------
    @Rule(
        LearningRateTooHigh(),
        OscillatingLoss(),
        NOT(UseWeightClipping()),
    )
    def opt_004_high_lr_oscillating(self) -> None:
        """When a high learning rate causes loss oscillation, temporary
        gradient clipping can stabilise training while the LR is tuned.
        """
        self.declare(UseWeightClipping())
        self.declare(
            Explanation(
                rule_id="OPT_004",
                triggered_by="LearningRateTooHigh,OscillatingLoss",
                derived="UseWeightClipping",
                explanation=(
                    "Combining a learning-rate reduction with gradient "
                    "clipping can quickly stabilise a training run that "
                    "is oscillating due to an overly aggressive learning rate."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_005 – LearningRateTooLow + SlowConvergence → AddBatchNormalization
    # ------------------------------------------------------------------
    @Rule(
        LearningRateTooLow(),
        SlowConvergence(),
        NOT(AddBatchNormalization()),
    )
    def opt_005_slow_bn(self) -> None:
        """Batch normalisation can accelerate convergence by smoothing the
        loss landscape, partially compensating for a low learning rate.
        """
        self.declare(AddBatchNormalization())
        self.declare(
            Explanation(
                rule_id="OPT_005",
                triggered_by="LearningRateTooLow,SlowConvergence",
                derived="AddBatchNormalization",
                explanation=(
                    "Batch normalisation smooths the optimisation landscape, "
                    "allowing faster convergence and making the training "
                    "less sensitive to the choice of learning rate."
                ),
            )
        )

    # ------------------------------------------------------------------
    # OPT_006 – GradientExplosion + GradientVanishing → BadWeightInitialization
    # ------------------------------------------------------------------
    @Rule(
        GradientExplosion(),
        GradientVanishing(),
        NOT(BadWeightInitialization()),
    )
    def opt_006_both_gradient_issues(self) -> None:
        """Simultaneous explosion and vanishing in different layers strongly
        implicates weight initialisation as the root cause.
        """
        self.declare(BadWeightInitialization())
        self.declare(
            Explanation(
                rule_id="OPT_006",
                triggered_by="GradientExplosion,GradientVanishing",
                derived="BadWeightInitialization",
                explanation=(
                    "Observing both exploding and vanishing gradients in "
                    "different parts of the network is a strong signal that "
                    "the weight initialisation is miscalibrated, causing "
                    "inconsistent gradient magnitudes across layers."
                ),
            )
        )
