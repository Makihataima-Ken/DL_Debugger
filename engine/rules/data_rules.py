"""
data_rules.py
=============
Experta rules dedicated to data-centric root causes: leakage, label noise,
imbalance, and distribution shift.

Rule IDs: DATA_001 - DATA_005
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Symptoms
    DataLeakageSuspected,
    ValidationAccuracyHigh,
    TestAccuracyLow,
    NoisyLabels,
    ClassImbalanceDetected,
    # Causes
    DistributionShift,
    PoorDataQuality,
    DataImbalance,
    # Recommendations
    InspectDataPipeline,
    ImproveLabelQuality,
    # XAI
    Explanation,
)


class DataRules(KnowledgeEngine):
    """Rule set covering data-quality diagnostics."""

    # DATA_001 - DataLeakageSuspected -> InspectDataPipeline
    @Rule(DataLeakageSuspected(), NOT(InspectDataPipeline()))
    def data_001_leakage_pipeline(self) -> None:
        self.declare(InspectDataPipeline())
        self.declare(
            Explanation(
                rule_id="DATA_001",
                triggered_by="DataLeakageSuspected",
                derived="InspectDataPipeline",
                confidence=0.9,
                explanation=(
                    "Suspected leakage requires auditing the split boundaries, "
                    "feature engineering, and normalisation statistics to ensure "
                    "no test information reaches training."
                ),
            )
        )

    # DATA_002 - ValidationAccuracyHigh + TestAccuracyLow -> DistributionShift
    @Rule(
        ValidationAccuracyHigh(),
        TestAccuracyLow(),
        NOT(DistributionShift()),
    )
    def data_002_shift(self) -> None:
        self.declare(DistributionShift())
        self.declare(
            Explanation(
                rule_id="DATA_002",
                triggered_by="ValidationAccuracyHigh,TestAccuracyLow",
                derived="DistributionShift",
                confidence=0.8,
                explanation=(
                    "Strong validation performance with weak test performance "
                    "is a hallmark of a train/test distribution shift."
                ),
            )
        )

    # DATA_003 - NoisyLabels -> PoorDataQuality
    @Rule(NoisyLabels(), NOT(PoorDataQuality()))
    def data_003_noise_quality(self) -> None:
        self.declare(PoorDataQuality())
        self.declare(
            Explanation(
                rule_id="DATA_003",
                triggered_by="NoisyLabels",
                derived="PoorDataQuality",
                confidence=0.88,
                explanation=(
                    "Label noise directly degrades the supervision signal and "
                    "is the primary driver of poor data quality."
                ),
            )
        )

    # DATA_004 - PoorDataQuality -> ImproveLabelQuality
    @Rule(PoorDataQuality(), NOT(ImproveLabelQuality()))
    def data_004_quality_fix(self) -> None:
        self.declare(ImproveLabelQuality())
        self.declare(
            Explanation(
                rule_id="DATA_004",
                triggered_by="PoorDataQuality",
                derived="ImproveLabelQuality",
                confidence=0.85,
                explanation=(
                    "Re-annotating ambiguous samples and removing corrupt "
                    "examples is the direct remedy for poor data quality."
                ),
            )
        )

    # DATA_005 - ClassImbalanceDetected -> DataImbalance
    @Rule(ClassImbalanceDetected(), NOT(DataImbalance()))
    def data_005_imbalance(self) -> None:
        self.declare(DataImbalance())
        self.declare(
            Explanation(
                rule_id="DATA_005",
                triggered_by="ClassImbalanceDetected",
                derived="DataImbalance",
                confidence=0.9,
                explanation=(
                    "Detected class imbalance is confirmed as a root cause that "
                    "biases the model toward majority classes."
                ),
            )
        )
