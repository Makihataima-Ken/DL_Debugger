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


# ---------------------------------------------------------------------------
# Model-type Facts (set by the NLP layer; never diagnostic on their own)
# ---------------------------------------------------------------------------

class ModelIsTransformer(Fact):
    """The model under analysis is a Transformer/attention architecture."""
    pass


class ModelIsCNN(Fact):
    """The model under analysis is a convolutional neural network."""
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
