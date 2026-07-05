"""
training_rules.py
=================
Experta rules for diagnosing problems observed during *training*.

Rule IDs: TRAIN_001 – TRAIN_012
"""

from experta import KnowledgeEngine, Rule, AS, NOT

from models.facts import (
    # Symptoms
    TrainingLossHigh,
    TrainingAccuracyLow,
    OscillatingLoss,
    SlowConvergence,
    GradientExplosion,
    GradientVanishing,
    NoisyLabels,
    SmallDataset,
    ClassImbalanceDetected,
    UnderfittingObserved,
    # Causes
    LearningRateTooHigh,
    LearningRateTooLow,
    BadWeightInitialization,
    PoorDataQuality,
    DataImbalance,
    ModelTooSimple,
    # XAI
    Explanation,
)


class TrainingRules(KnowledgeEngine):
    """Rule set covering training-phase diagnostics."""

    # ------------------------------------------------------------------
    # TRAIN_001 – Oscillating high loss → LR too high
    # ------------------------------------------------------------------
    @Rule(
        TrainingLossHigh(),
        OscillatingLoss(),
        NOT(LearningRateTooHigh()),
    )
    def train_001_lr_too_high(self) -> None:
        """High training loss combined with loss oscillation implies the
        learning rate is too large, causing the optimiser to overshoot minima.
        """
        self.declare(LearningRateTooHigh())
        self.declare(
            Explanation(
                rule_id="TRAIN_001",
                triggered_by="TrainingLossHigh,OscillatingLoss",
                derived="LearningRateTooHigh",
                explanation=(
                    "High loss combined with oscillation usually indicates "
                    "an excessively large learning rate that causes the "
                    "optimiser to repeatedly overshoot the loss minimum."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_002 – Slow convergence with high loss → LR too low
    # ------------------------------------------------------------------
    @Rule(
        TrainingLossHigh(),
        SlowConvergence(),
        NOT(LearningRateTooLow()),
    )
    def train_002_lr_too_low(self) -> None:
        """High training loss combined with very slow progress implies the
        learning rate is too small.
        """
        self.declare(LearningRateTooLow())
        self.declare(
            Explanation(
                rule_id="TRAIN_002",
                triggered_by="TrainingLossHigh,SlowConvergence",
                derived="LearningRateTooLow",
                explanation=(
                    "A high loss that decreases only very slowly is a "
                    "classic sign of a learning rate that is too small; "
                    "gradient steps are too tiny to make meaningful progress."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_003 – Gradient explosion → bad weight initialisation
    # ------------------------------------------------------------------
    @Rule(
        GradientExplosion(),
        NOT(BadWeightInitialization()),
    )
    def train_003_explosion_init(self) -> None:
        """Exploding gradients are often rooted in poor weight initialisation
        that places activations in saturating regions.
        """
        self.declare(BadWeightInitialization())
        self.declare(
            Explanation(
                rule_id="TRAIN_003",
                triggered_by="GradientExplosion",
                derived="BadWeightInitialization",
                explanation=(
                    "Exploding gradients indicate that the initial weight "
                    "scale is too large, causing activations and gradients "
                    "to grow uncontrollably through the network."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_004 – Gradient vanishing → bad weight initialisation
    # ------------------------------------------------------------------
    @Rule(
        GradientVanishing(),
        NOT(BadWeightInitialization()),
    )
    def train_004_vanishing_init(self) -> None:
        """Vanishing gradients indicate weights are initialised too small or
        that unsuitable activation functions are being used.
        """
        self.declare(BadWeightInitialization())
        self.declare(
            Explanation(
                rule_id="TRAIN_004",
                triggered_by="GradientVanishing",
                derived="BadWeightInitialization",
                explanation=(
                    "Vanishing gradients suggest that the initial weight "
                    "magnitudes are too small or activations saturate, "
                    "preventing error signals from propagating to early layers."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_005 – Noisy labels → poor data quality
    # ------------------------------------------------------------------
    @Rule(
        NoisyLabels(),
        NOT(PoorDataQuality()),
        salience=1,
    )
    def train_005_noisy_labels(self) -> None:
        """Noisy labels are a direct indicator of poor data quality."""
        self.declare(PoorDataQuality())
        self.declare(
            Explanation(
                rule_id="TRAIN_005",
                triggered_by="NoisyLabels",
                derived="PoorDataQuality",
                explanation=(
                    "Label noise directly degrades data quality; "
                    "the model receives contradictory supervision signals "
                    "that prevent it from learning a consistent decision boundary."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_006 – Class imbalance detected → DataImbalance cause
    # ------------------------------------------------------------------
    @Rule(
        ClassImbalanceDetected(),
        NOT(DataImbalance()),
    )
    def train_006_class_imbalance(self) -> None:
        """Class imbalance in the dataset is confirmed as a root cause."""
        self.declare(DataImbalance())
        self.declare(
            Explanation(
                rule_id="TRAIN_006",
                triggered_by="ClassImbalanceDetected",
                derived="DataImbalance",
                explanation=(
                    "Severe class imbalance causes the model to be biased "
                    "towards majority classes, resulting in deceptively high "
                    "overall accuracy while minority classes are ignored."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_007 – Low training accuracy + small dataset → underfitting
    # ------------------------------------------------------------------
    @Rule(
        TrainingAccuracyLow(),
        SmallDataset(),
        NOT(UnderfittingObserved()),
    )
    def train_007_underfit_small_data(self) -> None:
        """Low training accuracy on a small dataset suggests the model is
        underfitting – possibly because the dataset is too limited to train.
        """
        self.declare(UnderfittingObserved())
        self.declare(
            Explanation(
                rule_id="TRAIN_007",
                triggered_by="TrainingAccuracyLow,SmallDataset",
                derived="UnderfittingObserved",
                explanation=(
                    "When training accuracy is low even on a small dataset "
                    "the model is likely underfitting: it lacks the capacity "
                    "or training signal needed to fit the available examples."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_008 – Low training accuracy, no small dataset → ModelTooSimple
    # ------------------------------------------------------------------
    @Rule(
        TrainingAccuracyLow(),
        NOT(SmallDataset()),
        NOT(ModelTooSimple()),
    )
    def train_008_model_too_simple(self) -> None:
        """Low training accuracy without a dataset-size explanation implies
        the model architecture lacks sufficient capacity.
        """
        self.declare(ModelTooSimple())
        self.declare(
            Explanation(
                rule_id="TRAIN_008",
                triggered_by="TrainingAccuracyLow",
                derived="ModelTooSimple",
                explanation=(
                    "Low training accuracy on an adequately sized dataset "
                    "points to a model that is too simple to capture the "
                    "underlying data distribution."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_009 – UnderfittingObserved → ModelTooSimple
    # ------------------------------------------------------------------
    @Rule(
        UnderfittingObserved(),
        NOT(ModelTooSimple()),
    )
    def train_009_underfit_simple(self) -> None:
        """Underfitting implies the model is too simple for the task."""
        self.declare(ModelTooSimple())
        self.declare(
            Explanation(
                rule_id="TRAIN_009",
                triggered_by="UnderfittingObserved",
                derived="ModelTooSimple",
                explanation=(
                    "Underfitting is the canonical symptom of a model with "
                    "insufficient capacity; increasing model complexity or "
                    "depth is the primary remedy."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_010 – Gradient explosion alone → LR too high (secondary)
    # ------------------------------------------------------------------
    @Rule(
        GradientExplosion(),
        NOT(LearningRateTooHigh()),
    )
    def train_010_explosion_lr(self) -> None:
        """Gradient explosion can also indicate the learning rate is too high,
        amplifying already-large gradient signals.
        """
        self.declare(LearningRateTooHigh())
        self.declare(
            Explanation(
                rule_id="TRAIN_010",
                triggered_by="GradientExplosion",
                derived="LearningRateTooHigh",
                explanation=(
                    "An excessively large learning rate can amplify gradient "
                    "magnitudes at each step, eventually causing them to "
                    "explode; reducing the LR stabilises training."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_011 – Noisy labels + high training loss → poor data quality
    # ------------------------------------------------------------------
    @Rule(
        NoisyLabels(),
        TrainingLossHigh(),
        NOT(PoorDataQuality()),
    )
    def train_011_noisy_high_loss(self) -> None:
        """High training loss alongside noisy labels reinforces poor data quality."""
        self.declare(PoorDataQuality())
        self.declare(
            Explanation(
                rule_id="TRAIN_011",
                triggered_by="NoisyLabels,TrainingLossHigh",
                derived="PoorDataQuality",
                explanation=(
                    "When noisy labels co-occur with high training loss, "
                    "the label noise is likely the primary driver of the "
                    "high loss: the model is being penalised for correct "
                    "predictions on mislabelled examples."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_012 – Slow convergence alone → LR too low (no high loss required)
    # ------------------------------------------------------------------
    @Rule(
        SlowConvergence(),
        NOT(TrainingLossHigh()),
        NOT(LearningRateTooLow()),
    )
    def train_012_slow_no_high_loss(self) -> None:
        """Slow convergence even without flagrantly high loss suggests an
        overly small learning rate.
        """
        self.declare(LearningRateTooLow())
        self.declare(
            Explanation(
                rule_id="TRAIN_012",
                triggered_by="SlowConvergence",
                derived="LearningRateTooLow",
                explanation=(
                    "Slow convergence without obviously high loss still "
                    "indicates that the learning rate is too conservative "
                    "for the optimiser to navigate the loss landscape efficiently."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TRAIN_013 - Oscillating loss alone -> LR too high (weak fallback)
    # ------------------------------------------------------------------
    @Rule(
        OscillatingLoss(),
        NOT(TrainingLossHigh()),
        NOT(LearningRateTooHigh()),
    )
    def train_013_oscillation_lr_fallback(self) -> None:
        """Loss oscillation alone is weaker evidence for an excessive LR."""
        self.declare(LearningRateTooHigh())
        self.declare(
            Explanation(
                rule_id="TRAIN_013",
                triggered_by="OscillatingLoss",
                derived="LearningRateTooHigh",
                explanation=(
                    "Loss oscillation by itself is weaker but useful evidence "
                    "that the optimiser may be overshooting; lowering the "
                    "learning rate is a reasonable first check."
                ),
                confidence=0.55,
            )
        )
