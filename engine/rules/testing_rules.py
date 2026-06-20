"""
testing_rules.py
================
Experta rules for diagnosing problems observed during *testing / evaluation*.

Rule IDs: TEST_001 – TEST_006
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Symptoms
    ValidationAccuracyHigh,
    ValidationLossHigh,
    TestAccuracyLow,
    DataLeakageSuspected,
    ClassImbalanceDetected,
    PoorGeneralization,
    # Causes
    DistributionShift,
    DataImbalance,
    # XAI
    Explanation,
)


class TestingRules(KnowledgeEngine):
    """Rule set covering test-phase diagnostics."""

    # ------------------------------------------------------------------
    # TEST_001 – High val acc + low test acc → DistributionShift
    # ------------------------------------------------------------------
    @Rule(
        ValidationAccuracyHigh(),
        TestAccuracyLow(),
        NOT(DistributionShift()),
    )
    def test_001_distribution_shift(self) -> None:
        """High validation accuracy with low test accuracy is the canonical
        indicator of a distribution shift between validation and test data.
        """
        self.declare(DistributionShift())
        self.declare(
            Explanation(
                rule_id="TEST_001",
                triggered_by="ValidationAccuracyHigh,TestAccuracyLow",
                derived="DistributionShift",
                explanation=(
                    "When the model performs well on validation but poorly "
                    "on the test set, the most likely explanation is a "
                    "distribution shift: the test data follows a different "
                    "statistical distribution than the validation data."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TEST_002 – DistributionShift → PoorGeneralization
    # ------------------------------------------------------------------
    @Rule(
        DistributionShift(),
        NOT(PoorGeneralization()),
    )
    def test_002_shift_poor_gen(self) -> None:
        """Distribution shift directly causes poor generalisation."""
        self.declare(PoorGeneralization())
        self.declare(
            Explanation(
                rule_id="TEST_002",
                triggered_by="DistributionShift",
                derived="PoorGeneralization",
                explanation=(
                    "A distribution shift means the model has not learned "
                    "features that transfer across domains; it generalises "
                    "poorly to any data that deviates from its training "
                    "distribution."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TEST_003 – DataLeakageSuspected + TestAccuracyLow → DistributionShift
    # ------------------------------------------------------------------
    @Rule(
        DataLeakageSuspected(),
        TestAccuracyLow(),
        NOT(DistributionShift()),
    )
    def test_003_leakage_test_drop(self) -> None:
        """If data leakage is suspected and test accuracy is still low, the
        model's inflated validation score was artificially boosted by leakage,
        masking the true distribution shift.
        """
        self.declare(DistributionShift())
        self.declare(
            Explanation(
                rule_id="TEST_003",
                triggered_by="DataLeakageSuspected,TestAccuracyLow",
                derived="DistributionShift",
                explanation=(
                    "Data leakage can inflate validation metrics, making the "
                    "drop to test accuracy appear like a distribution shift. "
                    "Eliminating leakage is necessary before a fair evaluation "
                    "of generalisation is possible."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TEST_004 – ClassImbalanceDetected + TestAccuracyLow → DataImbalance
    # ------------------------------------------------------------------
    @Rule(
        ClassImbalanceDetected(),
        TestAccuracyLow(),
        NOT(DataImbalance()),
    )
    def test_004_imbalance_test(self) -> None:
        """Class imbalance in test data can cause low aggregate accuracy
        if the model is biased toward majority classes.
        """
        self.declare(DataImbalance())
        self.declare(
            Explanation(
                rule_id="TEST_004",
                triggered_by="ClassImbalanceDetected,TestAccuracyLow",
                derived="DataImbalance",
                explanation=(
                    "Class imbalance in the test set inflates the apparent "
                    "accuracy of majority-class predictions while minority "
                    "classes are misclassified, depressing overall test "
                    "accuracy."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TEST_005 – ValidationLossHigh + TestAccuracyLow → PoorGeneralization
    # ------------------------------------------------------------------
    @Rule(
        ValidationLossHigh(),
        TestAccuracyLow(),
        NOT(PoorGeneralization()),
    )
    def test_005_val_loss_test_acc(self) -> None:
        """High validation loss alongside low test accuracy confirms the
        model cannot generalise even to in-distribution held-out data.
        """
        self.declare(PoorGeneralization())
        self.declare(
            Explanation(
                rule_id="TEST_005",
                triggered_by="ValidationLossHigh,TestAccuracyLow",
                derived="PoorGeneralization",
                explanation=(
                    "High validation loss combined with low test accuracy "
                    "indicates systematic poor generalisation: the model "
                    "struggles with any data outside the training set, "
                    "regardless of distribution."
                ),
            )
        )

    # ------------------------------------------------------------------
    # TEST_006 – PoorGeneralization alone → DistributionShift (secondary)
    # ------------------------------------------------------------------
    @Rule(
        PoorGeneralization(),
        NOT(DistributionShift()),
        NOT(ValidationAccuracyHigh()),
    )
    def test_006_poor_gen_shift(self) -> None:
        """Poor generalisation without an obvious validation-accuracy signal
        may still indicate a subtle distribution shift.
        """
        self.declare(DistributionShift())
        self.declare(
            Explanation(
                rule_id="TEST_006",
                triggered_by="PoorGeneralization",
                derived="DistributionShift",
                explanation=(
                    "Poor generalisation without clear overfitting evidence "
                    "often points to a subtle shift between training and test "
                    "distributions that was not captured by the validation set."
                ),
            )
        )
