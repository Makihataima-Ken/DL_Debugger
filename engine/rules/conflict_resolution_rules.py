"""
conflict_resolution_rules.py
============================
Low-salience Experta meta-rules that resolve contradictory derived causes.

Rule IDs: CONFLICT_001 - CONFLICT_009
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    AdamLearningRateTooHigh,
    BatchSizeTooLarge,
    BatchSizeTooSmall,
    CauseConflict,
    ExcessiveRegularization,
    Explanation,
    InsufficientRegularization,
    LearningRateNotScaled,
    LearningRateTooHigh,
    LearningRateTooLow,
    ModelTooComplex,
    ModelTooSimple,
    MomentumMisconfigured,
    MomentumTooHigh,
    OverfittingObserved,
    SuppressedCause,
    UnderfittingObserved,
)


CONTRADICTORY_CAUSE_GROUPS: list[frozenset[str]] = [
    frozenset({"OverfittingObserved", "UnderfittingObserved"}),
    frozenset({"LearningRateTooHigh", "LearningRateTooLow"}),
    frozenset({"LearningRateTooHigh", "LearningRateNotScaled"}),
    frozenset({"LearningRateTooLow", "AdamLearningRateTooHigh"}),
    frozenset({"LearningRateNotScaled", "AdamLearningRateTooHigh"}),
    frozenset({"ModelTooComplex", "ModelTooSimple"}),
    frozenset({"InsufficientRegularization", "ExcessiveRegularization"}),
    frozenset({"BatchSizeTooLarge", "BatchSizeTooSmall"}),
    frozenset({"MomentumTooHigh", "MomentumMisconfigured"}),
]


class ConflictResolutionRules(KnowledgeEngine):
    """Meta-rules that suppress lower-supported contradictory causes."""

    @staticmethod
    def _noisy_or(values: list[float]) -> float:
        acc = 1.0
        for value in values:
            acc *= (1.0 - max(0.0, min(1.0, value)))
        return round(1.0 - acc, 3)

    def _evidence_values(self, cause_name: str) -> list[float]:
        return [
            float(fact.get("confidence", 0.8))
            for fact in self.facts.values()
            if type(fact) is Explanation and fact["derived"] == cause_name
        ]

    def _resolve_conflict(self, rule_id: str, contenders: tuple[str, ...]) -> None:
        evidence = {
            cause: self._evidence_values(cause)
            for cause in contenders
        }
        strengths = {
            cause: self._noisy_or(values)
            for cause, values in evidence.items()
        }
        counts = {
            cause: len(values)
            for cause, values in evidence.items()
        }

        winner = sorted(
            contenders,
            key=lambda cause: (-strengths[cause], -counts[cause], cause),
        )[0]
        losers = [cause for cause in contenders if cause != winner]

        best_loser_strength = max(strengths[cause] for cause in losers)
        best_loser_count = max(counts[cause] for cause in losers)
        if strengths[winner] > best_loser_strength:
            basis = "highest evidence strength"
        elif counts[winner] > best_loser_count:
            basis = "more supporting explanations"
        else:
            basis = "lexicographic tie-break"

        contending_causes = ",".join(contenders)
        loser_text = ",".join(losers)
        reason = (
            f"{winner} won by {basis}; strengths={strengths}; "
            f"support_counts={counts}."
        )

        self.declare(
            CauseConflict(
                rule_id=rule_id,
                contending_causes=contending_causes,
                winner=winner,
                losers=loser_text,
                reason=reason,
            )
        )
        for loser in losers:
            self.declare(
                SuppressedCause(
                    cause=loser,
                    superseded_by=winner,
                    reason=reason,
                )
            )

        confidence = max(0.5, min(0.95, strengths[winner]))
        self.declare(
            Explanation(
                rule_id=rule_id,
                triggered_by=contending_causes,
                derived="CauseConflict",
                confidence=confidence,
                explanation=(
                    f"Contradictory causes were present ({contending_causes}). "
                    f"{winner} is retained and {loser_text} is suppressed because "
                    f"it has {basis}."
                ),
            )
        )

    @Rule(
        OverfittingObserved(),
        UnderfittingObserved(),
        NOT(CauseConflict(rule_id="CONFLICT_001")),
        salience=-100,
    )
    def conflict_001_overfit_underfit(self) -> None:
        self._resolve_conflict(
            "CONFLICT_001",
            ("OverfittingObserved", "UnderfittingObserved"),
        )

    @Rule(
        LearningRateTooHigh(),
        LearningRateTooLow(),
        NOT(CauseConflict(rule_id="CONFLICT_002")),
        salience=-100,
    )
    def conflict_002_lr_high_low(self) -> None:
        self._resolve_conflict(
            "CONFLICT_002",
            ("LearningRateTooHigh", "LearningRateTooLow"),
        )

    @Rule(
        LearningRateTooHigh(),
        LearningRateNotScaled(),
        NOT(CauseConflict(rule_id="CONFLICT_003")),
        salience=-100,
    )
    def conflict_003_lr_high_not_scaled(self) -> None:
        self._resolve_conflict(
            "CONFLICT_003",
            ("LearningRateTooHigh", "LearningRateNotScaled"),
        )

    @Rule(
        LearningRateTooLow(),
        AdamLearningRateTooHigh(),
        NOT(CauseConflict(rule_id="CONFLICT_004")),
        salience=-100,
    )
    def conflict_004_lr_low_adam_high(self) -> None:
        self._resolve_conflict(
            "CONFLICT_004",
            ("LearningRateTooLow", "AdamLearningRateTooHigh"),
        )

    @Rule(
        LearningRateNotScaled(),
        AdamLearningRateTooHigh(),
        NOT(CauseConflict(rule_id="CONFLICT_005")),
        salience=-100,
    )
    def conflict_005_lr_not_scaled_adam_high(self) -> None:
        self._resolve_conflict(
            "CONFLICT_005",
            ("LearningRateNotScaled", "AdamLearningRateTooHigh"),
        )

    @Rule(
        ModelTooComplex(),
        ModelTooSimple(),
        NOT(CauseConflict(rule_id="CONFLICT_006")),
        salience=-100,
    )
    def conflict_006_model_complex_simple(self) -> None:
        self._resolve_conflict(
            "CONFLICT_006",
            ("ModelTooComplex", "ModelTooSimple"),
        )

    @Rule(
        InsufficientRegularization(),
        ExcessiveRegularization(),
        NOT(CauseConflict(rule_id="CONFLICT_007")),
        salience=-100,
    )
    def conflict_007_regularization(self) -> None:
        self._resolve_conflict(
            "CONFLICT_007",
            ("InsufficientRegularization", "ExcessiveRegularization"),
        )

    @Rule(
        BatchSizeTooLarge(),
        BatchSizeTooSmall(),
        NOT(CauseConflict(rule_id="CONFLICT_008")),
        salience=-100,
    )
    def conflict_008_batch_size(self) -> None:
        self._resolve_conflict(
            "CONFLICT_008",
            ("BatchSizeTooLarge", "BatchSizeTooSmall"),
        )

    @Rule(
        MomentumTooHigh(),
        MomentumMisconfigured(),
        NOT(CauseConflict(rule_id="CONFLICT_009")),
        salience=-100,
    )
    def conflict_009_momentum(self) -> None:
        self._resolve_conflict(
            "CONFLICT_009",
            ("MomentumTooHigh", "MomentumMisconfigured"),
        )
