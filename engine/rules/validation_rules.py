"""
validation_rules.py
===================
Experta rules for diagnosing problems observed during *validation*.

Rule IDs: VAL_001 – VAL_008
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Symptoms
    TrainingAccuracyHigh,
    TrainingAccuracyLow,
    ValidationAccuracyLow,
    ValidationLossHigh,
    OverfittingObserved,
    UnderfittingObserved,
    SmallDataset,
    DataLeakageSuspected,
    # Causes
    ModelTooComplex,
    ModelTooSimple,
    InsufficientRegularization,
    ExcessiveRegularization,
    # XAI
    Explanation,
)


class ValidationRules(KnowledgeEngine):
    """Rule set covering validation-phase diagnostics."""

    # ------------------------------------------------------------------
    # VAL_001 – High train acc + low val acc → OverfittingObserved
    # ------------------------------------------------------------------
    @Rule(
        TrainingAccuracyHigh(),
        ValidationAccuracyLow(),
        NOT(OverfittingObserved()),
    )
    def val_001_detect_overfitting(self) -> None:
        """The canonical overfitting signal: the model memorises training data
        but fails to generalise to the validation set.
        """
        self.declare(OverfittingObserved())
        self.declare(
            Explanation(
                rule_id="VAL_001",
                triggered_by="TrainingAccuracyHigh,ValidationAccuracyLow",
                derived="OverfittingObserved",
                explanation=(
                    "A large gap between high training accuracy and low "
                    "validation accuracy is the hallmark of overfitting: "
                    "the model has memorised training examples instead of "
                    "learning generalisable patterns."
                ),
            )
        )

    # ------------------------------------------------------------------
    # VAL_002 – OverfittingObserved → ModelTooComplex
    # ------------------------------------------------------------------
    @Rule(
        OverfittingObserved(),
        NOT(ModelTooComplex()),
    )
    def val_002_overfit_complex_model(self) -> None:
        """Overfitting commonly arises from a model that is too complex for
        the available data.
        """
        self.declare(ModelTooComplex())
        self.declare(
            Explanation(
                rule_id="VAL_002",
                triggered_by="OverfittingObserved",
                derived="ModelTooComplex",
                explanation=(
                    "Overfitting implies the model has more capacity than "
                    "the dataset can support; reducing model depth, width, "
                    "or parameter count is a primary fix."
                ),
            )
        )

    # ------------------------------------------------------------------
    # VAL_003 – OverfittingObserved → InsufficientRegularization
    # ------------------------------------------------------------------
    @Rule(
        OverfittingObserved(),
        NOT(InsufficientRegularization()),
    )
    def val_003_overfit_no_regularization(self) -> None:
        """Overfitting can also be attributed to missing or weak regularisation."""
        self.declare(InsufficientRegularization())
        self.declare(
            Explanation(
                rule_id="VAL_003",
                triggered_by="OverfittingObserved",
                derived="InsufficientRegularization",
                explanation=(
                    "Without adequate regularisation (dropout, L2 penalty, "
                    "batch normalisation), the model is free to overfit the "
                    "training set. Introducing or strengthening regularisation "
                    "typically narrows the train-val gap."
                ),
            )
        )

    # ------------------------------------------------------------------
    # VAL_004 – Low train acc + low val acc → UnderfittingObserved
    # ------------------------------------------------------------------
    @Rule(
        TrainingAccuracyLow(),
        ValidationAccuracyLow(),
        NOT(UnderfittingObserved()),
    )
    def val_004_detect_underfitting(self) -> None:
        """Both training and validation accuracy are low: the model is
        underfitting across the board.
        """
        self.declare(UnderfittingObserved())
        self.declare(
            Explanation(
                rule_id="VAL_004",
                triggered_by="TrainingAccuracyLow,ValidationAccuracyLow",
                derived="UnderfittingObserved",
                explanation=(
                    "When accuracy is low on both the training and validation "
                    "sets the model is underfitting: it has not yet learned "
                    "the task. This calls for a more complex model, better "
                    "features, or longer training."
                ),
            )
        )

    # ------------------------------------------------------------------
    # VAL_005 – UnderfittingObserved → ModelTooSimple
    # ------------------------------------------------------------------
    @Rule(
        UnderfittingObserved(),
        NOT(ModelTooSimple()),
    )
    def val_005_underfit_simple(self) -> None:
        """Underfitting implies the model architecture lacks capacity."""
        self.declare(ModelTooSimple())
        self.declare(
            Explanation(
                rule_id="VAL_005",
                triggered_by="UnderfittingObserved",
                derived="ModelTooSimple",
                explanation=(
                    "The model is underfitting, which strongly suggests it "
                    "is too simple (too few layers, too few parameters) to "
                    "capture the complexity of the target mapping."
                ),
            )
        )

    # ------------------------------------------------------------------
    # VAL_006 – ValidationLossHigh + SmallDataset → InsufficientRegularization
    # ------------------------------------------------------------------
    @Rule(
        ValidationLossHigh(),
        SmallDataset(),
        NOT(InsufficientRegularization()),
    )
    def val_006_val_loss_small_data(self) -> None:
        """High validation loss on a small dataset strongly points to
        insufficient regularisation allowing the model to overfit.
        """
        self.declare(InsufficientRegularization())
        self.declare(
            Explanation(
                rule_id="VAL_006",
                triggered_by="ValidationLossHigh,SmallDataset",
                derived="InsufficientRegularization",
                explanation=(
                    "A small dataset magnifies the risk of overfitting; "
                    "high validation loss under these conditions indicates "
                    "that regularisation (dropout, weight decay, data "
                    "augmentation) is inadequate."
                ),
            )
        )

    # ------------------------------------------------------------------
    # VAL_007 – DataLeakageSuspected + ValidationAccuracyLow →
    #           overfit signal via ExcessiveRegularization (secondary path)
    # ------------------------------------------------------------------
    @Rule(
        DataLeakageSuspected(),
        ValidationAccuracyLow(),
        NOT(ExcessiveRegularization()),
    )
    def val_007_leakage_low_val(self) -> None:
        """If data leakage is suspected yet validation accuracy is still low,
        excessive regularisation may be masking the leak while still
        hurting generalisation.
        """
        self.declare(ExcessiveRegularization())
        self.declare(
            Explanation(
                rule_id="VAL_007",
                triggered_by="DataLeakageSuspected,ValidationAccuracyLow",
                derived="ExcessiveRegularization",
                explanation=(
                    "When data leakage is present but validation accuracy "
                    "remains low, excessive regularisation may be preventing "
                    "the model from exploiting any signal at all – reducing "
                    "regularisation while fixing the leakage is advised."
                ),
            )
        )

    # ------------------------------------------------------------------
    # VAL_008 – UnderfittingObserved → ExcessiveRegularization (alternative)
    # ------------------------------------------------------------------
    @Rule(
        UnderfittingObserved(),
        NOT(ExcessiveRegularization()),
        NOT(ModelTooSimple()),
    )
    def val_008_underfit_excess_reg(self) -> None:
        """Underfitting can also be caused by regularisation that is too
        aggressive, effectively preventing the model from fitting the data.
        """
        self.declare(ExcessiveRegularization())
        self.declare(
            Explanation(
                rule_id="VAL_008",
                triggered_by="UnderfittingObserved",
                derived="ExcessiveRegularization",
                explanation=(
                    "Overly strong regularisation (very high dropout rate, "
                    "large L2 penalty) can suppress the model's ability to "
                    "fit the training data, manifesting as underfitting even "
                    "when the architecture is adequate."
                ),
            )
        )
