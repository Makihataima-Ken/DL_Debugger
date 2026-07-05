"""
mixed_precision_rules.py
========================
Experta rules for automatic mixed-precision training issues.

Rule IDs: AMP_001 - AMP_007
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    # Context
    UsesMixedPrecision,
    # Symptoms
    NaNLoss,
    MixedPrecisionOverflow,
    LossScaleOverflow,
    GradientUnderflow,
    # Causes
    LossScalingMisconfigured,
    Fp16RangeExceeded,
    Fp16GradientUnderflow,
    # Recommendations
    EnableDynamicLossScaling,
    UseBf16,
    KeepMasterWeightsInFp32,
    # XAI
    Explanation,
)


class MixedPrecisionRules(KnowledgeEngine):
    """Rule set covering AMP/fp16 stability diagnostics."""

    # AMP_001 - AMP + NaNLoss -> Fp16RangeExceeded
    @Rule(
        UsesMixedPrecision(),
        NaNLoss(),
        NOT(Fp16RangeExceeded()),
    )
    def amp_001_nan_fp16_range(self) -> None:
        self.declare(Fp16RangeExceeded())
        self.declare(
            Explanation(
                rule_id="AMP_001",
                triggered_by="UsesMixedPrecision,NaNLoss",
                derived="Fp16RangeExceeded",
                confidence=0.83,
                explanation=(
                    "NaN or Inf loss during mixed-precision training often "
                    "means activations or gradients exceeded fp16's numeric range."
                ),
            )
        )

    # AMP_002 - AMP + MixedPrecisionOverflow -> Fp16RangeExceeded
    @Rule(
        UsesMixedPrecision(),
        MixedPrecisionOverflow(),
        NOT(Fp16RangeExceeded()),
    )
    def amp_002_overflow_fp16_range(self) -> None:
        self.declare(Fp16RangeExceeded())
        self.declare(
            Explanation(
                rule_id="AMP_002",
                triggered_by="UsesMixedPrecision,MixedPrecisionOverflow",
                derived="Fp16RangeExceeded",
                confidence=0.9,
                explanation=(
                    "Explicit overflow under reduced precision is strong "
                    "evidence that fp16 range is too narrow for the current run."
                ),
            )
        )

    # AMP_003 - AMP + LossScaleOverflow -> LossScalingMisconfigured
    @Rule(
        UsesMixedPrecision(),
        LossScaleOverflow(),
        NOT(LossScalingMisconfigured()),
    )
    def amp_003_loss_scale_overflow(self) -> None:
        self.declare(LossScalingMisconfigured())
        self.declare(
            Explanation(
                rule_id="AMP_003",
                triggered_by="UsesMixedPrecision,LossScaleOverflow",
                derived="LossScalingMisconfigured",
                confidence=0.88,
                explanation=(
                    "Repeated loss-scale overflow indicates that the scaler is "
                    "not adapting well to the magnitude of the gradients."
                ),
            )
        )

    # AMP_004 - AMP + GradientUnderflow -> Fp16GradientUnderflow
    @Rule(
        UsesMixedPrecision(),
        GradientUnderflow(),
        NOT(Fp16GradientUnderflow()),
    )
    def amp_004_gradient_underflow(self) -> None:
        self.declare(Fp16GradientUnderflow())
        self.declare(
            Explanation(
                rule_id="AMP_004",
                triggered_by="UsesMixedPrecision,GradientUnderflow",
                derived="Fp16GradientUnderflow",
                confidence=0.87,
                explanation=(
                    "Gradients that underflow to zero during AMP training are "
                    "a direct sign that fp16 precision is losing small updates."
                ),
            )
        )

    # AMP_005 - AMP + LossScalingMisconfigured -> EnableDynamicLossScaling
    @Rule(
        UsesMixedPrecision(),
        LossScalingMisconfigured(),
        NOT(EnableDynamicLossScaling()),
    )
    def amp_005_dynamic_loss_scaling(self) -> None:
        self.declare(EnableDynamicLossScaling())
        self.declare(
            Explanation(
                rule_id="AMP_005",
                triggered_by="UsesMixedPrecision,LossScalingMisconfigured",
                derived="EnableDynamicLossScaling",
                confidence=0.9,
                explanation=(
                    "Dynamic loss scaling adapts the scale during training, "
                    "reducing skipped steps caused by overflow or underflow."
                ),
            )
        )

    # AMP_006 - AMP + Fp16RangeExceeded -> UseBf16
    @Rule(
        UsesMixedPrecision(),
        Fp16RangeExceeded(),
        NOT(UseBf16()),
    )
    def amp_006_use_bf16(self) -> None:
        self.declare(UseBf16())
        self.declare(
            Explanation(
                rule_id="AMP_006",
                triggered_by="UsesMixedPrecision,Fp16RangeExceeded",
                derived="UseBf16",
                confidence=0.86,
                explanation=(
                    "bfloat16 preserves a wider exponent range than fp16, "
                    "making it a practical fix for range-driven AMP overflow."
                ),
            )
        )

    # AMP_007 - AMP + Fp16GradientUnderflow -> KeepMasterWeightsInFp32
    @Rule(
        UsesMixedPrecision(),
        Fp16GradientUnderflow(),
        NOT(KeepMasterWeightsInFp32()),
    )
    def amp_007_master_weights(self) -> None:
        self.declare(KeepMasterWeightsInFp32())
        self.declare(
            Explanation(
                rule_id="AMP_007",
                triggered_by="UsesMixedPrecision,Fp16GradientUnderflow",
                derived="KeepMasterWeightsInFp32",
                confidence=0.82,
                explanation=(
                    "Keeping optimizer master weights in fp32 preserves small "
                    "updates that would otherwise vanish in reduced precision."
                ),
            )
        )
