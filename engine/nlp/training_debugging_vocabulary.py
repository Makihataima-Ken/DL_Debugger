"""
training_debugging_vocabulary.py
================================
Controlled vocabulary mapping natural-language phrases to Fact class names.

This is *data*, not logic: each entry is a controlled trigger phrase mapped
to the symptom or model-type Fact name(s) it should emit. The matcher in
entity_mapper.py lemmatises phrases with spaCy before lookup; no diagnostic
inference lives here.
"""

from __future__ import annotations

FactNameSpec = str | tuple[str, ...]


# phrase -> Fact class name(s). Phrases are matched by spaCy lemmas against
# the controlled vocabulary. Order does not matter; all matches are collected.
SYMPTOM_PHRASES: dict[str, FactNameSpec] = {
    # loss behaviour
    "oscillate": "OscillatingLoss",
    "loss oscillates": "OscillatingLoss",
    "loss is oscillating": "OscillatingLoss",
    "loss keeps oscillating": "OscillatingLoss",
    "loss bounces": "OscillatingLoss",
    "loss fluctuates": "OscillatingLoss",
    "loss spikes": "OscillatingLoss",
    "training loss high": "TrainingLossHigh",
    "high training loss": "TrainingLossHigh",
    "loss stays high": "TrainingLossHigh",
    "loss not decreasing": "TrainingLossHigh",
    "loss never converges": "TrainingLossHigh",
    "never converges": "TrainingLossHigh",
    "loss plateau": ("TrainingLossHigh", "SlowConvergence"),
    "loss stuck": "TrainingLossHigh",
    "stuck loss": "TrainingLossHigh",
    "validation loss high": "ValidationLossHigh",
    "high validation loss": "ValidationLossHigh",
    "val loss high": "ValidationLossHigh",
    # nan / explosion
    "nan": "NaNLoss",
    "nans": "NaNLoss",
    "not a number": "NaNLoss",
    "loss becomes inf": "NaNLoss",
    "infinite loss": "NaNLoss",
    "loss goes to infinity": "NaNLoss",
    "goes to infinity": "NaNLoss",
    "inf loss": "NaNLoss",
    "overflow": "NaNLoss",
    "explode": "GradientExplosion",
    "loss explodes": "GradientExplosion",
    "loss is exploding": "GradientExplosion",
    "loss blows up": "GradientExplosion",
    "blows up": "GradientExplosion",
    "gradient explodes": "GradientExplosion",
    "exploding gradient": "GradientExplosion",
    "gradients blow up": "GradientExplosion",
    "gradient vanishes": "GradientVanishing",
    "vanishing gradient": "GradientVanishing",
    "gradients are zero": "GradientVanishing",
    "early layers not learning": "GradientVanishing",
    # convergence
    "slow convergence": "SlowConvergence",
    "converges slowly": "SlowConvergence",
    "training is slow": "SlowConvergence",
    "barely improves": "SlowConvergence",
    "will not learn": "SlowConvergence",
    "wont learn": "SlowConvergence",
    "stuck": "SlowConvergence",
    "plateau": "SlowConvergence",
    "training plateau": "SlowConvergence",
    # accuracy
    "training accuracy high": "TrainingAccuracyHigh",
    "high training accuracy": "TrainingAccuracyHigh",
    "99% training": "TrainingAccuracyHigh",
    "training accuracy low": "TrainingAccuracyLow",
    "low training accuracy": "TrainingAccuracyLow",
    "validation accuracy low": "ValidationAccuracyLow",
    "low validation accuracy": "ValidationAccuracyLow",
    "validation accuracy much lower": "ValidationAccuracyLow",
    "validation accuracy is much lower": "ValidationAccuracyLow",
    "val accuracy lower": "ValidationAccuracyLow",
    "train validation gap": "ValidationAccuracyLow",
    "train val gap": "ValidationAccuracyLow",
    "accuracy gap": "ValidationAccuracyLow",
    "validation accuracy high": "ValidationAccuracyHigh",
    "high validation accuracy": "ValidationAccuracyHigh",
    "test accuracy low": "TestAccuracyLow",
    "low test accuracy": "TestAccuracyLow",
    "poor on test": "TestAccuracyLow",
    # generalisation / overfit phrasing
    "memorize": "OverfittingObserved",
    "memorising": "OverfittingObserved",
    "overfit": "OverfittingObserved",
    "underfit": "UnderfittingObserved",
    "generalise": "PoorGeneralization",
    "generalize": "PoorGeneralization",
    # data
    "class imbalance": "ClassImbalanceDetected",
    "imbalanced dataset": "ClassImbalanceDetected",
    "data leak": "DataLeakageSuspected",
    "data leakage": "DataLeakageSuspected",
    "split leakage": "DataLeakageSuspected",
    "train test contamination": "DataLeakageSuspected",
    "leaking data": "DataLeakageSuspected",
    "leakage": "DataLeakageSuspected",
    "noisy label": "NoisyLabels",
    "label noise": "NoisyLabels",
    "mislabel": "NoisyLabels",
    "small dataset": "SmallDataset",
    "few examples": "SmallDataset",
    "not much data": "SmallDataset",
    "little data": "SmallDataset",
    "tiny dataset": "SmallDataset",
    "large dataset": "LargeDataset",
    "millions of examples": "LargeDataset",
    # architecture-specific symptoms
    "dead relu": "DeadReLUDetected",
    "dying relu": "DeadReLUDetected",
    "units output zero": "DeadReLUDetected",
    "attention collapse": "AttentionCollapse",
    "attention is uniform": "AttentionCollapse",
    "attends to one token": "AttentionCollapse",
    "tokenize": "TokenizationIssue",
    "tokenization": "TokenizationIssue",
    "lots of unk": "TokenizationIssue",
    "out of vocabulary": "TokenizationIssue",
    "context length": "ContextLengthExceeded",
    "sequence too long": "ContextLengthExceeded",
    "max length exceeded": "ContextLengthExceeded",
    "feature collapse": "FeatureCollapse",
    "feature maps constant": "FeatureCollapse",
    "receptive field": "ReceptiveFieldTooSmall",
}

# Model-type detection phrases.
MODEL_TYPE_PHRASES: dict[str, FactNameSpec] = {
    "transformer": "ModelIsTransformer",
    "attention": "ModelIsTransformer",
    "bert": "ModelIsTransformer",
    "gpt": "ModelIsTransformer",
    "llm": "ModelIsTransformer",
    "self-attention": "ModelIsTransformer",
    "vision transformer": "ModelIsTransformer",
    "vit": "ModelIsTransformer",
    "cnn": "ModelIsCNN",
    "convolution": "ModelIsCNN",
    "conv net": "ModelIsCNN",
    "convnet": "ModelIsCNN",
    "resnet": "ModelIsCNN",
    "vgg": "ModelIsCNN",
}

CONTEXT_PHRASES: dict[str, FactNameSpec] = {
    "mixed precision": "UsesMixedPrecision",
    "automatic mixed precision": "UsesMixedPrecision",
    "amp": "UsesMixedPrecision",
    "fp16": "UsesMixedPrecision",
    "bf16": "UsesMixedPrecision",
    "distributed training": "UsesDistributedTraining",
    "multi gpu": "UsesDistributedTraining",
    "multi device": "UsesDistributedTraining",
    "ddp": "UsesDistributedTraining",
    "adam": "OptimizerIsAdam",
    "adamw": "OptimizerIsAdam",
    "sgd": "OptimizerIsSGD",
}

# Synonym normalisation applied before phrase matching.
SYNONYMS: dict[str, str] = {
    "acc": "accuracy",
    "lr": "learning rate",
    "val": "validation",
    "grads": "gradients",
    "grad": "gradient",
    "diverge": "explode",
    "diverging": "exploding",
    "blowing up": "exploding",
    "wont": "will not",
}
