"""
transformer_rules.py
====================
Experta rules specific to Transformer/attention architectures. These rules are
gated on ModelIsTransformer so they only fire for the relevant model type.

Rule IDs: TF_001 - TF_017
"""

from experta import KnowledgeEngine, Rule, NOT

from models.facts import (
    ModelIsTransformer,
    AttentionCollapse,
    TokenizationIssue,
    ContextLengthExceeded,
    PoorGeneralization,
    WarmupMissing,
    AttentionEntropyCollapse,
    PositionalEncodingProblem,
    LongSequenceMemoryBlowup,
    RepetitiveGeneration,
    MissingLearningRateWarmup,
    AttentionEntropyCollapsed,
    PositionalEncodingMisconfigured,
    QuadraticAttentionMemoryBlowup,
    DegenerateGeneration,
    # Recommendations
    FixTokenizer,
    TruncateOrChunkInput,
    AddBatchNormalization,
    AddLearningRateWarmup,
    UseGradientCheckpointing,
    UseFlashAttention,
    ApplyLabelSmoothing,
    ClipAttentionLogits,
    ChunkOrTruncateInput,
    FixPositionalEncoding,
    AdjustDecodingStrategy,
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

    # TF_005 - Transformer + WarmupMissing -> MissingLearningRateWarmup
    @Rule(
        ModelIsTransformer(),
        WarmupMissing(),
        NOT(MissingLearningRateWarmup()),
    )
    def tf_005_missing_warmup(self) -> None:
        self.declare(MissingLearningRateWarmup())
        self.declare(
            Explanation(
                rule_id="TF_005",
                triggered_by="ModelIsTransformer,WarmupMissing",
                derived="MissingLearningRateWarmup",
                confidence=0.84,
                explanation=(
                    "Transformers are sensitive to early optimizer steps; "
                    "missing warmup is a likely cause of unstable or slow starts."
                ),
            )
        )

    # TF_006 - Transformer + MissingLearningRateWarmup -> AddLearningRateWarmup
    @Rule(
        ModelIsTransformer(),
        MissingLearningRateWarmup(),
        NOT(AddLearningRateWarmup()),
    )
    def tf_006_add_warmup(self) -> None:
        self.declare(AddLearningRateWarmup())
        self.declare(
            Explanation(
                rule_id="TF_006",
                triggered_by="ModelIsTransformer,MissingLearningRateWarmup",
                derived="AddLearningRateWarmup",
                confidence=0.86,
                explanation=(
                    "Adding a warmup schedule lets transformer optimizers ramp "
                    "up safely before full learning-rate updates are applied."
                ),
            )
        )

    # TF_007 - Transformer + AttentionEntropyCollapse -> AttentionEntropyCollapsed
    @Rule(
        ModelIsTransformer(),
        AttentionEntropyCollapse(),
        NOT(AttentionEntropyCollapsed()),
    )
    def tf_007_attention_entropy(self) -> None:
        self.declare(AttentionEntropyCollapsed())
        self.declare(
            Explanation(
                rule_id="TF_007",
                triggered_by="ModelIsTransformer,AttentionEntropyCollapse",
                derived="AttentionEntropyCollapsed",
                confidence=0.88,
                explanation=(
                    "Collapsed attention entropy means heads are no longer "
                    "distributing information usefully across the sequence."
                ),
            )
        )

    # TF_008 - Transformer + AttentionEntropyCollapsed -> ClipAttentionLogits
    @Rule(
        ModelIsTransformer(),
        AttentionEntropyCollapsed(),
        NOT(ClipAttentionLogits()),
    )
    def tf_008_clip_attention_logits(self) -> None:
        self.declare(ClipAttentionLogits())
        self.declare(
            Explanation(
                rule_id="TF_008",
                triggered_by="ModelIsTransformer,AttentionEntropyCollapsed",
                derived="ClipAttentionLogits",
                confidence=0.79,
                explanation=(
                    "Clipping or temperature-scaling attention logits can keep "
                    "attention entropy from collapsing into unusable patterns."
                ),
            )
        )

    # TF_009 - Transformer + PositionalEncodingProblem -> PositionalEncodingMisconfigured
    @Rule(
        ModelIsTransformer(),
        PositionalEncodingProblem(),
        NOT(PositionalEncodingMisconfigured()),
    )
    def tf_009_position_encoding(self) -> None:
        self.declare(PositionalEncodingMisconfigured())
        self.declare(
            Explanation(
                rule_id="TF_009",
                triggered_by="ModelIsTransformer,PositionalEncodingProblem",
                derived="PositionalEncodingMisconfigured",
                confidence=0.86,
                explanation=(
                    "Position-encoding symptoms indicate that sequence order, "
                    "offsets, or maximum length handling are misconfigured."
                ),
            )
        )

    # TF_010 - Transformer + PositionalEncodingMisconfigured -> FixPositionalEncoding
    @Rule(
        ModelIsTransformer(),
        PositionalEncodingMisconfigured(),
        NOT(FixPositionalEncoding()),
    )
    def tf_010_fix_position_encoding(self) -> None:
        self.declare(FixPositionalEncoding())
        self.declare(
            Explanation(
                rule_id="TF_010",
                triggered_by="ModelIsTransformer,PositionalEncodingMisconfigured",
                derived="FixPositionalEncoding",
                confidence=0.86,
                explanation=(
                    "Correcting positional embedding length, offsets, and "
                    "padding alignment restores coherent sequence-order signals."
                ),
            )
        )

    # TF_011 - Transformer + LongSequenceMemoryBlowup -> QuadraticAttentionMemoryBlowup
    @Rule(
        ModelIsTransformer(),
        LongSequenceMemoryBlowup(),
        NOT(QuadraticAttentionMemoryBlowup()),
    )
    def tf_011_attention_memory(self) -> None:
        self.declare(QuadraticAttentionMemoryBlowup())
        self.declare(
            Explanation(
                rule_id="TF_011",
                triggered_by="ModelIsTransformer,LongSequenceMemoryBlowup",
                derived="QuadraticAttentionMemoryBlowup",
                confidence=0.9,
                explanation=(
                    "Memory blowup on long sequences is consistent with the "
                    "quadratic activation cost of standard full attention."
                ),
            )
        )

    # TF_012 - Transformer + QuadraticAttentionMemoryBlowup -> UseGradientCheckpointing
    @Rule(
        ModelIsTransformer(),
        QuadraticAttentionMemoryBlowup(),
        NOT(UseGradientCheckpointing()),
    )
    def tf_012_gradient_checkpointing(self) -> None:
        self.declare(UseGradientCheckpointing())
        self.declare(
            Explanation(
                rule_id="TF_012",
                triggered_by="ModelIsTransformer,QuadraticAttentionMemoryBlowup",
                derived="UseGradientCheckpointing",
                confidence=0.83,
                explanation=(
                    "Gradient checkpointing trades extra compute for much lower "
                    "activation memory, which helps long-sequence transformer runs."
                ),
            )
        )

    # TF_013 - Transformer + QuadraticAttentionMemoryBlowup -> UseFlashAttention
    @Rule(
        ModelIsTransformer(),
        QuadraticAttentionMemoryBlowup(),
        NOT(UseFlashAttention()),
    )
    def tf_013_flash_attention(self) -> None:
        self.declare(UseFlashAttention())
        self.declare(
            Explanation(
                rule_id="TF_013",
                triggered_by="ModelIsTransformer,QuadraticAttentionMemoryBlowup",
                derived="UseFlashAttention",
                confidence=0.84,
                explanation=(
                    "Memory-efficient attention kernels reduce attention memory "
                    "pressure without changing the high-level model architecture."
                ),
            )
        )

    # TF_014 - Transformer + RepetitiveGeneration -> DegenerateGeneration
    @Rule(
        ModelIsTransformer(),
        RepetitiveGeneration(),
        NOT(DegenerateGeneration()),
    )
    def tf_014_degenerate_generation(self) -> None:
        self.declare(DegenerateGeneration())
        self.declare(
            Explanation(
                rule_id="TF_014",
                triggered_by="ModelIsTransformer,RepetitiveGeneration",
                derived="DegenerateGeneration",
                confidence=0.86,
                explanation=(
                    "Repetitive generation indicates that the model's decoding "
                    "distribution has collapsed into low-diversity loops."
                ),
            )
        )

    # TF_015 - Transformer + DegenerateGeneration -> ApplyLabelSmoothing
    @Rule(
        ModelIsTransformer(),
        DegenerateGeneration(),
        NOT(ApplyLabelSmoothing()),
    )
    def tf_015_label_smoothing(self) -> None:
        self.declare(ApplyLabelSmoothing())
        self.declare(
            Explanation(
                rule_id="TF_015",
                triggered_by="ModelIsTransformer,DegenerateGeneration",
                derived="ApplyLabelSmoothing",
                confidence=0.72,
                explanation=(
                    "Label smoothing can reduce overconfident next-token "
                    "distributions that contribute to repetitive generation."
                ),
            )
        )

    # TF_016 - Transformer + DegenerateGeneration -> AdjustDecodingStrategy
    @Rule(
        ModelIsTransformer(),
        DegenerateGeneration(),
        NOT(AdjustDecodingStrategy()),
    )
    def tf_016_adjust_decoding(self) -> None:
        self.declare(AdjustDecodingStrategy())
        self.declare(
            Explanation(
                rule_id="TF_016",
                triggered_by="ModelIsTransformer,DegenerateGeneration",
                derived="AdjustDecodingStrategy",
                confidence=0.8,
                explanation=(
                    "Tuning decoding temperature, top-p, repetition penalties, "
                    "or beam settings directly targets repetitive output loops."
                ),
            )
        )

    # TF_017 - Transformer + ContextLengthExceeded -> ChunkOrTruncateInput
    @Rule(
        ModelIsTransformer(),
        ContextLengthExceeded(),
        NOT(ChunkOrTruncateInput()),
    )
    def tf_017_chunk_or_truncate(self) -> None:
        self.declare(ChunkOrTruncateInput())
        self.declare(
            Explanation(
                rule_id="TF_017",
                triggered_by="ModelIsTransformer,ContextLengthExceeded",
                derived="ChunkOrTruncateInput",
                confidence=0.86,
                explanation=(
                    "Chunking or truncating inputs keeps sequence length within "
                    "the transformer's supported context window."
                ),
            )
        )
