"""
transformer_rules.py
====================
Experta rules specific to Transformer/attention architectures. These rules are
gated on ModelIsTransformer so they only fire for the relevant model type.

Rule IDs: TF_001 - TF_004
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    ModelIsTransformer,
    AttentionCollapse,
    TokenizationIssue,
    ContextLengthExceeded,
    PoorGeneralization,
    # Recommendations
    FixTokenizer,
    TruncateOrChunkInput,
    AddBatchNormalization,
    # XAI
    Explanation,
)


class TransformerRules(KnowledgeEngine):
    """Transformer-specific diagnostics."""

    # TF_001 - Transformer + AttentionCollapse -> PoorGeneralization
    @Rule(
        ModelIsTransformer(),
        AttentionCollapse(),
        NOT(PoorGeneralization()),
    )
    def tf_001_attention_collapse(self) -> None:
        self.declare(PoorGeneralization())
        self.declare(
            Explanation(
                rule_id="TF_001",
                triggered_by="ModelIsTransformer,AttentionCollapse",
                derived="PoorGeneralization",
                confidence=0.75,
                explanation=(
                    "When attention collapses onto a single token or becomes "
                    "uniform, the model stops routing information and fails to "
                    "generalise; check temperature scaling, LR warmup, and "
                    "attention/entropy regularisation."
                ),
            )
        )

    # TF_002 - Transformer + TokenizationIssue -> FixTokenizer
    @Rule(
        ModelIsTransformer(),
        TokenizationIssue(),
        NOT(FixTokenizer()),
    )
    def tf_002_tokenizer(self) -> None:
        self.declare(FixTokenizer())
        self.declare(
            Explanation(
                rule_id="TF_002",
                triggered_by="ModelIsTransformer,TokenizationIssue",
                derived="FixTokenizer",
                confidence=0.8,
                explanation=(
                    "Excessive <unk> tokens or vocabulary mismatch corrupts the "
                    "input representation; audit tokenizer coverage and special "
                    "token handling before further training."
                ),
            )
        )

    # TF_003 - Transformer + ContextLengthExceeded -> TruncateOrChunkInput
    @Rule(
        ModelIsTransformer(),
        ContextLengthExceeded(),
        NOT(TruncateOrChunkInput()),
    )
    def tf_003_context_length(self) -> None:
        self.declare(TruncateOrChunkInput())
        self.declare(
            Explanation(
                rule_id="TF_003",
                triggered_by="ModelIsTransformer,ContextLengthExceeded",
                derived="TruncateOrChunkInput",
                confidence=0.85,
                explanation=(
                    "Inputs exceeding the maximum context length are silently "
                    "truncated or error out; chunk long sequences or adopt a "
                    "long-context attention variant."
                ),
            )
        )

    # TF_004 - Transformer + AttentionCollapse -> AddBatchNormalization (LayerNorm audit proxy)
    @Rule(
        ModelIsTransformer(),
        AttentionCollapse(),
        NOT(AddBatchNormalization()),
    )
    def tf_004_norm_audit(self) -> None:
        self.declare(AddBatchNormalization())
        self.declare(
            Explanation(
                rule_id="TF_004",
                triggered_by="ModelIsTransformer,AttentionCollapse",
                derived="AddBatchNormalization",
                confidence=0.55,
                explanation=(
                    "Attention collapse is frequently tied to mis-placed or "
                    "mis-scaled normalisation; verify pre/post-LayerNorm "
                    "placement and residual scaling."
                ),
            )
        )
