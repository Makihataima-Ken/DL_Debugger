"""
facts.py
========
All Experta Fact classes used by the Deep Learning Debugging Expert System.

Hierarchy
---------
Fact
├── Symptom facts   – observable problems the user reports
├── Cause facts     – inferred root causes
├── Recommendation facts – actionable steps
└── Explanation facts – audit trail for XAI
"""

from experta import Fact, Field


# ---------------------------------------------------------------------------
# Symptom Facts
# ---------------------------------------------------------------------------

class TrainingLossHigh(Fact):
    """Training loss is significantly above the expected threshold."""
    pass


class ValidationLossHigh(Fact):
    """Validation loss is significantly above the expected threshold."""
    pass


class TrainingAccuracyHigh(Fact):
    """Training accuracy is high (model fits training data well)."""
    pass


class TrainingAccuracyLow(Fact):
    """Training accuracy is low – model is not fitting training data."""
    pass


class NaNLoss(Fact):
    """Loss became NaN/Inf during training (numerical blow-up)."""
    pass


class ValidationAccuracyHigh(Fact):
    """Validation accuracy is high."""
    pass


class ValidationAccuracyLow(Fact):
    """Validation accuracy is significantly below training accuracy."""
    pass


class TestAccuracyLow(Fact):
    """Test-set accuracy is substantially below validation accuracy."""
    pass


class OverfittingObserved(Fact):
    """Overfitting has been detected (high train acc, low val acc)."""
    pass


class UnderfittingObserved(Fact):
    """Underfitting has been detected (low train acc and val acc)."""
    pass


class GradientExplosion(Fact):
    """Gradients are exploding (NaN loss, very large parameter updates)."""
    pass


class GradientVanishing(Fact):
    """Gradients are vanishing (near-zero updates in early layers)."""
    pass


class SlowConvergence(Fact):
    """Model is converging but at an unusually slow rate."""
    pass


class OscillatingLoss(Fact):
    """Loss oscillates significantly between steps or epochs."""
    pass


class ClassImbalanceDetected(Fact):
    """Severe class imbalance exists in the training dataset."""
    pass


class DataLeakageSuspected(Fact):
    """Data leakage between training and test splits is suspected."""
    pass


class SmallDataset(Fact):
    """The dataset is too small to train the current model reliably."""
    pass


class LargeDataset(Fact):
    """The dataset is large (may require specific optimisation strategies)."""
    pass


class NoisyLabels(Fact):
    """Training labels contain significant noise or annotation errors."""
    pass


class PoorGeneralization(Fact):
    """Model generalises poorly to unseen data."""
    pass


class DeadReLUDetected(Fact):
    """A large fraction of ReLU units output zero for all inputs."""
    pass


class AttentionCollapse(Fact):
    """Attention weights collapse onto a single token / become uniform."""
    pass


class TokenizationIssue(Fact):
    """Suspected tokenization/vocabulary mismatch or excessive <unk>."""
    pass


class ContextLengthExceeded(Fact):
    """Inputs exceed the model's maximum context/sequence length."""
    pass


class FeatureCollapse(Fact):
    """CNN feature maps collapse to near-constant / redundant features."""
    pass


class ReceptiveFieldTooSmall(Fact):
    """Effective receptive field is too small for the target structures."""
    pass


class MixedPrecisionOverflow(Fact):
    """Overflow is observed when training with reduced precision."""
    pass


class LossScaleOverflow(Fact):
    """The AMP loss scaler repeatedly overflows or skips optimizer steps."""
    pass


class GradientUnderflow(Fact):
    """Gradients underflow to zero or become too tiny under reduced precision."""
    pass


class MultiGpuThroughputLow(Fact):
    """Multi-GPU training throughput is lower than expected."""
    pass


class GpuUnderutilization(Fact):
    """One or more GPUs spend significant time idle during training."""
    pass


class BatchNormDesync(Fact):
    """Batch-normalization statistics differ across distributed workers."""
    pass


class GradientSyncSlow(Fact):
    """Gradient synchronization is a visible distributed-training bottleneck."""
    pass


class PerDeviceBatchTooSmallObserved(Fact):
    """Each device receives too few samples for stable training."""
    pass


class CoupledWeightDecayUsed(Fact):
    """Adam-style optimization is using coupled L2 weight decay."""
    pass


class MissingMomentum(Fact):
    """SGD is configured without momentum or with negligible momentum."""
    pass


class AdamHighLearningRateInstability(Fact):
    """Adam becomes unstable at the configured learning rate."""
    pass


class WarmupMissing(Fact):
    """No learning-rate warmup is configured for a warmup-sensitive model."""
    pass


class AttentionEntropyCollapse(Fact):
    """Attention entropy collapses, producing overly sharp or uniform attention."""
    pass


class PositionalEncodingProblem(Fact):
    """Position embeddings or encodings appear misaligned with the input."""
    pass


class LongSequenceMemoryBlowup(Fact):
    """Long sequences cause transformer memory usage to grow explosively."""
    pass


class RepetitiveGeneration(Fact):
    """Generation is repetitive, degenerate, or stuck in loops."""
    pass


class BatchNormTrainEvalMismatch(Fact):
    """CNN behavior changes unexpectedly between training and evaluation modes."""
    pass


class AggressivePoolingOrStride(Fact):
    """Pooling or stride settings remove spatial detail too aggressively."""
    pass


class InsufficientSpatialAugmentation(Fact):
    """Spatial augmentation is missing or too weak for visual generalization."""
    pass


class ChannelCollapse(Fact):
    """CNN channels collapse to redundant or near-constant activations."""
    pass


# ---------------------------------------------------------------------------
# Context Facts (injectable; never diagnostic on their own)
# ---------------------------------------------------------------------------

class ModelIsTransformer(Fact):
    """The model under analysis is a Transformer/attention architecture."""
    pass


class ModelIsCNN(Fact):
    """The model under analysis is a convolutional neural network."""
    pass


class UsesMixedPrecision(Fact):
    """Training uses fp16/bfloat16 automatic mixed precision."""
    pass


class UsesDistributedTraining(Fact):
    """Training is distributed across multiple devices or workers."""
    pass


class OptimizerIsAdam(Fact):
    """The optimizer is Adam, AdamW, or an Adam-like variant."""
    pass


class OptimizerIsSGD(Fact):
    """The optimizer is SGD or an SGD-like variant."""
    pass


# ---------------------------------------------------------------------------
# Cause Facts
# ---------------------------------------------------------------------------

class LearningRateTooHigh(Fact):
    """The learning rate is too high, causing instability."""
    pass


class LearningRateTooLow(Fact):
    """The learning rate is too low, causing slow convergence."""
    pass


class ModelTooComplex(Fact):
    """The model has too many parameters relative to the data."""
    pass


class ModelTooSimple(Fact):
    """The model lacks capacity to capture the underlying patterns."""
    pass


class InsufficientRegularization(Fact):
    """Regularisation is absent or too weak."""
    pass


class ExcessiveRegularization(Fact):
    """Regularisation is too strong, preventing the model from fitting."""
    pass


class BadWeightInitialization(Fact):
    """Weight initialisation is poor, causing gradient pathologies."""
    pass


class PoorDataQuality(Fact):
    """The training data is noisy, mislabelled, or corrupted."""
    pass


class DistributionShift(Fact):
    """The test distribution differs significantly from train/val."""
    pass


class DataImbalance(Fact):
    """The dataset has a severe class imbalance."""
    pass


class BatchSizeTooLarge(Fact):
    """Batch size is so large that gradient estimates over-smooth / generalise poorly."""
    pass


class BatchSizeTooSmall(Fact):
    """Batch size is so small that gradient noise destabilises training."""
    pass


class MomentumTooHigh(Fact):
    """Momentum coefficient is too high, amplifying oscillation/overshoot."""
    pass


class WeightDecayTooHigh(Fact):
    """Weight decay (L2) is too strong, suppressing model capacity."""
    pass


class NumericalInstability(Fact):
    """Training is numerically unstable (overflow / NaN propagation)."""
    pass


class LossScalingMisconfigured(Fact):
    """AMP loss scaling is fixed too high, too low, or otherwise misconfigured."""
    pass


class Fp16RangeExceeded(Fact):
    """Values exceed the representable numeric range of fp16 training."""
    pass


class Fp16GradientUnderflow(Fact):
    """Reduced precision causes small gradients to underflow."""
    pass


class DataLoadingBottleneck(Fact):
    """Input loading or host-to-device transfer limits distributed throughput."""
    pass


class ImproperBatchNormSync(Fact):
    """Batch-normalization statistics are not synchronized correctly."""
    pass


class GradientSyncOverhead(Fact):
    """Distributed gradient synchronization dominates step time."""
    pass


class LearningRateNotScaled(Fact):
    """Learning rate was not scaled for the distributed world size."""
    pass


class PerDeviceBatchTooSmall(Fact):
    """Per-device batch size is too small for stable distributed training."""
    pass


class WeightDecayCoupledWithAdam(Fact):
    """Adam is using coupled L2 weight decay instead of decoupled AdamW decay."""
    pass


class MomentumMisconfigured(Fact):
    """Momentum is absent, too weak, or otherwise misconfigured."""
    pass


class AdamLearningRateTooHigh(Fact):
    """Adam's learning rate is high enough to cause instability."""
    pass


class MissingLearningRateWarmup(Fact):
    """A warmup-sensitive architecture is being trained without LR warmup."""
    pass


class AttentionEntropyCollapsed(Fact):
    """Attention entropy has collapsed and is harming information routing."""
    pass


class PositionalEncodingMisconfigured(Fact):
    """Positional encodings are misconfigured for sequence length or layout."""
    pass


class QuadraticAttentionMemoryBlowup(Fact):
    """Standard attention memory cost is too high for the sequence length."""
    pass


class DegenerateGeneration(Fact):
    """The model generates repetitive or low-diversity outputs."""
    pass


class BatchNormModeMismatch(Fact):
    """BatchNorm running statistics or train/eval mode handling are mismatched."""
    pass


class StrideTooAggressive(Fact):
    """Stride or pooling settings remove spatial information too early."""
    pass


class SpatialAugmentationMissing(Fact):
    """The CNN lacks sufficient spatial augmentation for invariance."""
    pass


class ChannelCollapseCause(Fact):
    """CNN channel activations have collapsed into redundant features."""
    pass


# ---------------------------------------------------------------------------
# Recommendation Facts
# ---------------------------------------------------------------------------

class ReduceLearningRate(Fact):
    """Recommendation: reduce the learning rate."""
    pass


class IncreaseLearningRate(Fact):
    """Recommendation: increase the learning rate."""
    pass


class AddDropout(Fact):
    """Recommendation: add dropout layers to combat overfitting."""
    pass


class RemoveDropout(Fact):
    """Recommendation: remove or reduce dropout (may be over-regularising)."""
    pass


class AddBatchNormalization(Fact):
    """Recommendation: add batch normalisation layers."""
    pass


class UseEarlyStopping(Fact):
    """Recommendation: use early stopping on validation loss."""
    pass


class IncreaseDatasetSize(Fact):
    """Recommendation: collect or synthesise more training data."""
    pass


class ApplyDataAugmentation(Fact):
    """Recommendation: apply data augmentation to expand effective dataset."""
    pass


class RebalanceClasses(Fact):
    """Recommendation: rebalance classes via oversampling / undersampling."""
    pass


class ImproveLabelQuality(Fact):
    """Recommendation: audit and clean training labels."""
    pass


class UseWeightClipping(Fact):
    """Recommendation: apply gradient / weight clipping."""
    pass


class UseProperInitialization(Fact):
    """Recommendation: use He / Xavier / Glorot initialisation."""
    pass


class ReduceModelComplexity(Fact):
    """Recommendation: reduce model depth or width."""
    pass


class IncreaseModelComplexity(Fact):
    """Recommendation: increase model depth or width."""
    pass


class UseWeightedLoss(Fact):
    """Recommendation: use a class-weighted loss function."""
    pass


class InspectDataPipeline(Fact):
    """Recommendation: inspect preprocessing and data pipeline for leakage."""
    pass


class CollectDomainData(Fact):
    """Recommendation: collect in-domain data that matches the test distribution."""
    pass


class ReduceRegularization(Fact):
    """Recommendation: reduce regularisation strength (L1/L2/dropout)."""
    pass


class ReduceBatchSize(Fact):
    """Recommendation: reduce the batch size."""
    pass


class IncreaseBatchSize(Fact):
    """Recommendation: increase the batch size."""
    pass


class ReduceMomentum(Fact):
    """Recommendation: lower the optimiser momentum / beta1."""
    pass


class ReduceWeightDecay(Fact):
    """Recommendation: lower the weight-decay coefficient."""
    pass


class UseLeakyReLU(Fact):
    """Recommendation: switch dead ReLUs to LeakyReLU/GELU/ELU."""
    pass


class UseGradientClipping(Fact):
    """Recommendation: clip gradients to bound numerical instability."""
    pass


class TruncateOrChunkInput(Fact):
    """Recommendation: truncate/chunk sequences to fit context length."""
    pass


class FixTokenizer(Fact):
    """Recommendation: audit tokenizer/vocabulary coverage."""
    pass


class IncreaseReceptiveField(Fact):
    """Recommendation: add depth/dilation/pooling to widen receptive field."""
    pass


class EnableDynamicLossScaling(Fact):
    """Recommendation: enable dynamic AMP loss scaling."""
    pass


class UseBf16(Fact):
    """Recommendation: use bfloat16 where hardware supports it."""
    pass


class KeepMasterWeightsInFp32(Fact):
    """Recommendation: keep optimizer master weights in fp32."""
    pass


class UseSyncBatchNorm(Fact):
    """Recommendation: synchronize BatchNorm statistics across workers."""
    pass


class IncreaseDataLoaderWorkers(Fact):
    """Recommendation: increase or tune data-loader workers/prefetching."""
    pass


class UseGradientAccumulation(Fact):
    """Recommendation: accumulate gradients across microbatches."""
    pass


class ScaleLearningRateByWorldSize(Fact):
    """Recommendation: scale learning rate for distributed world size."""
    pass


class UseAdamW(Fact):
    """Recommendation: switch Adam with L2 decay to AdamW."""
    pass


class EnableNesterovMomentum(Fact):
    """Recommendation: enable Nesterov momentum for SGD."""
    pass


class UseLearningRateWarmup(Fact):
    """Recommendation: add a learning-rate warmup schedule."""
    pass


class AddLearningRateWarmup(Fact):
    """Recommendation: add transformer-friendly learning-rate warmup."""
    pass


class UseGradientCheckpointing(Fact):
    """Recommendation: use gradient checkpointing to reduce activation memory."""
    pass


class UseFlashAttention(Fact):
    """Recommendation: use memory-efficient attention kernels."""
    pass


class ApplyLabelSmoothing(Fact):
    """Recommendation: apply label smoothing to reduce overconfidence."""
    pass


class ClipAttentionLogits(Fact):
    """Recommendation: clip or temperature-scale attention logits."""
    pass


class ChunkOrTruncateInput(Fact):
    """Recommendation: chunk or truncate long transformer inputs."""
    pass


class FixPositionalEncoding(Fact):
    """Recommendation: correct positional encoding length, offsets, or layout."""
    pass


class AdjustDecodingStrategy(Fact):
    """Recommendation: tune decoding strategy to avoid repetitive generation."""
    pass


class AddSpatialAugmentation(Fact):
    """Recommendation: add spatial image augmentation."""
    pass


class ReduceStride(Fact):
    """Recommendation: reduce early stride or pooling aggressiveness."""
    pass


class FixBatchNormMomentum(Fact):
    """Recommendation: fix BatchNorm momentum/statistics handling."""
    pass


class UseGlobalAveragePooling(Fact):
    """Recommendation: use global average pooling to stabilize CNN heads."""
    pass


# ---------------------------------------------------------------------------
# Explanation Fact
# ---------------------------------------------------------------------------

class Explanation(Fact):
    """
    Stores a single XAI audit record produced by a rule firing.

    Fields
    ------
    rule_id : str
        Unique identifier for the rule that fired.
    triggered_by : str
        Comma-separated names of the triggering facts.
    derived : str
        Name of the fact that was derived.
    explanation : str
        Human-readable rationale.
    confidence : float
        Rule-assigned confidence in the derived fact, in [0.0, 1.0].
    """

    rule_id: str = Field(str, mandatory=True)
    triggered_by: str = Field(str, mandatory=True)
    derived: str = Field(str, mandatory=True)
    explanation: str = Field(str, mandatory=True)
    confidence: float = Field(float, default=0.8)
