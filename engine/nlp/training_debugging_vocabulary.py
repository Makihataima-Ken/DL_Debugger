"""
training_debugging_vocabulary.py
================================
Controlled vocabulary mapping natural-language phrases to Fact class names.

This is *data*, not logic: each entry is a list of trigger phrases (already
normalised: lowercase, singular-ish, hyphen-free) mapped to the symptom or
model-type Fact name it should emit. The matcher in entity_mapper.py performs
the lookup; no diagnostic inference lives here.
"""

from __future__ import annotations

# phrase -> Fact class name. Phrases are matched token-wise against the
# normalised text. Order does not matter; all matches are collected.
SYMPTOM_PHRASES: dict[str, str] = {
    # loss behaviour
    "oscillat": "OscillatingLoss",
    "loss oscillat": "OscillatingLoss",
    "loss is oscillat": "OscillatingLoss",
    "loss keeps oscillat": "OscillatingLoss",
    "loss bounces": "OscillatingLoss",
    "loss fluctuat": "OscillatingLoss",
    "loss spikes": "OscillatingLoss",
    "training loss high": "TrainingLossHigh",
    "high training loss": "TrainingLossHigh",
    "loss stays high": "TrainingLossHigh",
    "loss not decreasing": "TrainingLossHigh",
    "loss never converg": "TrainingLossHigh",
    "never converg": "TrainingLossHigh",
    "loss plateau": "TrainingLossHigh",
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
    "explod": "GradientExplosion",
    "loss explod": "GradientExplosion",
    "loss is explod": "GradientExplosion",
    "loss blows up": "GradientExplosion",
    "blows up": "GradientExplosion",
    "gradient explod": "GradientExplosion",
    "exploding gradient": "GradientExplosion",
    "gradients blow up": "GradientExplosion",
    "gradient vanish": "GradientVanishing",
    "vanishing gradient": "GradientVanishing",
    "gradients are zero": "GradientVanishing",
    "early layers not learning": "GradientVanishing",
    # convergence
    "slow converg": "SlowConvergence",
    "converges slowly": "SlowConvergence",
    "training is slow": "SlowConvergence",
    "barely improv": "SlowConvergence",
    "will not learn": "SlowConvergence",
    "wont learn": "SlowConvergence",
    "stuck": "SlowConvergence",
    "plateau": "SlowConvergence",
    "training plateau": "SlowConvergence",
    "loss plateau": "SlowConvergence",
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
    "memoriz": "OverfittingObserved",
    "memorising": "OverfittingObserved",
    "overfit": "OverfittingObserved",
    "underfit": "UnderfittingObserved",
    "generalis": "PoorGeneralization",
    "generaliz": "PoorGeneralization",
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
    "tokeniz": "TokenizationIssue",
    "lots of unk": "TokenizationIssue",
    "out of vocabulary": "TokenizationIssue",
    "context length": "ContextLengthExceeded",
    "sequence too long": "ContextLengthExceeded",
    "max length exceed": "ContextLengthExceeded",
    "feature collapse": "FeatureCollapse",
    "feature maps constant": "FeatureCollapse",
    "receptive field": "ReceptiveFieldTooSmall",
}

# Model-type detection phrases.
MODEL_TYPE_PHRASES: dict[str, str] = {
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

CONTEXT_PHRASES: dict[str, str] = {
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
    "diverg": "explod",
    "blowing up": "explod",
    "wont": "will not",
}
