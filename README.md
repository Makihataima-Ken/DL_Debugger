# Knowledge-Based Expert System for Systematic Debugging of Deep Learning Model Outputs

A hybrid **NLP-powered, explainable, knowledge-based expert system** that helps AI students and developers diagnose problems in deep learning model training, validation, and testing. Users describe problems in **plain English**; an NLP layer converts the text into Experta facts, and **all reasoning still emerges exclusively from Experta rule activation** — no procedural decision logic exists outside rules.

---

## Architecture

```
dl_debugger/
├── main.py                          # CLI entry point
│
├── engine/
│   ├── nlp/                         # NLP preprocessing (NO diagnosis)
│   │   ├── training_debugging_vocabulary.py  # phrase -> Fact-name vocabulary
│   │   ├── terminology_normalizer.py         # normalise + synonym expansion
│   │   ├── entity_mapper.py                  # text -> Fact names + lexical conf.
│   │   ├── intent_detector.py                # coarse routing intent
│   │   └── symptom_extractor.py              # façade: text -> ExtractionResult
│   ├── knowledge_engine.py          # DebuggingKnowledgeEngine + DiagnosisResult
│   └── rules/
│       ├── training_rules.py        # TRAIN_001–TRAIN_012
│       ├── validation_rules.py      # VAL_001–VAL_008
│       ├── testing_rules.py         # TEST_001–TEST_006
│       ├── optimization_rules.py    # OPT_001–OPT_009
│       ├── architecture_rules.py    # ARCH_001–ARCH_009
│       ├── recommendation_rules.py # REC_001–REC_020
│       ├── data_rules.py            # DATA_001–DATA_005
│       ├── mixed_precision_rules.py # AMP_001–AMP_007 (gated on UsesMixedPrecision)
│       ├── distributed_rules.py     # DIST_001–DIST_010 (gated on UsesDistributedTraining)
│       ├── optimizer_rules.py       # OPTZ_001–OPTZ_006 (gated on optimizer context)
│       ├── transformer_rules.py     # TF_001–TF_017 (gated on ModelIsTransformer)
│       ├── cnn_rules.py             # CNN_001–CNN_011 (gated on ModelIsCNN)
│       └── conflict_resolution_rules.py # CONFLICT_001–CONFLICT_009
│
├── models/
│   └── facts.py                     # All Experta Fact subclasses
│
├── utils/
│   └── helpers.py                   # Hashing, formatting, validation utilities
│
├── data/
│   ├── scenario_loader.py           # JSON loader + built-in scenario registry
│   └── scenarios/
│       ├── training_examples.json
│       ├── validation_examples.json
│       └── testing_examples.json
│
├── tests/
│   ├── test_training.py
│   ├── test_validation.py
│   ├── test_testing.py
│   ├── test_nlp.py                  # NLP extraction tests
│   ├── test_new_rules.py            # data/transformer/cnn/opt/arch rules
│   └── test_end_to_end.py           # text -> diagnosis pipeline
│
├── requirements.txt
└── README.md
```

### Inference Flow

```
User supplies symptom names
         │
         ▼
DebuggingKnowledgeEngine.run_scenario()
  ├── reset()            ← clears working memory
  ├── inject_symptoms()  ← declares Fact instances
  └── run()              ← fires rules until fixpoint
         │
         ▼
Rule Activation (forward-chaining, Rete algorithm)
  Training Rules  ──────────────────────────┐
  Validation Rules ─────────────────────────┤
  Testing Rules ────────────────────────────┤──► Cause Facts
  Optimization Rules ───────────────────────┤    + Explanation Facts
  Architecture Rules ───────────────────────┘
         │
         ▼
Recommendation Rules
  Cause Facts ──────────────────────────────────► Recommendation Facts
         │
         ▼
get_diagnosis()
  ├── symptoms      (injected facts)
  ├── causes        (derived symptom + cause facts)
  ├── recommendations (derived recommendation facts)
  └── explanations  (XAI audit chain, sorted by rule ID)
```

---

## Fact Hierarchy

### Symptom Facts
Observable problems the user reports:

| Fact Class | Meaning |
|---|---|
| `TrainingLossHigh` | Training loss above expected threshold |
| `ValidationLossHigh` | Validation loss above expected threshold |
| `TrainingAccuracyHigh` | Model fits training data well |
| `TrainingAccuracyLow` | Model not fitting training data |
| `ValidationAccuracyHigh` | Validation accuracy is high |
| `ValidationAccuracyLow` | Validation accuracy well below training |
| `TestAccuracyLow` | Test accuracy below validation accuracy |
| `OverfittingObserved` | High train acc, low val acc |
| `UnderfittingObserved` | Low train and val accuracy |
| `GradientExplosion` | NaN loss or very large updates |
| `GradientVanishing` | Near-zero updates in early layers |
| `SlowConvergence` | Converging at unusually slow rate |
| `OscillatingLoss` | Loss oscillates between steps |
| `ClassImbalanceDetected` | Severe class imbalance in dataset |
| `DataLeakageSuspected` | Suspected train/test contamination |
| `SmallDataset` | Dataset too small for current model |
| `LargeDataset` | Large dataset |
| `NoisyLabels` | Significant label noise |
| `PoorGeneralization` | Poor transfer to unseen data |
| `NaNLoss` | Loss becomes NaN or Inf |
| `DeadReLUDetected` | ReLU units stop activating |
| `AttentionCollapse` | Attention collapses or becomes uniform |
| `TokenizationIssue` | Tokenizer/vocabulary mismatch |
| `ContextLengthExceeded` | Input exceeds sequence/context length |
| `FeatureCollapse` | CNN features collapse or become redundant |
| `ReceptiveFieldTooSmall` | CNN receptive field is too small |
| `MixedPrecisionOverflow` | AMP/fp16 overflow observed |
| `LossScaleOverflow` | AMP loss scaler overflows or skips steps |
| `GradientUnderflow` | Gradients underflow under reduced precision |
| `MultiGpuThroughputLow` | Distributed throughput is poor |
| `GpuUnderutilization` | GPUs are idle or underused |
| `BatchNormDesync` | BatchNorm stats desynchronize across workers |
| `GradientSyncSlow` | Gradient synchronization is slow |
| `PerDeviceBatchTooSmallObserved` | Per-device batch is too small |
| `CoupledWeightDecayUsed` | Adam uses coupled L2 weight decay |
| `MissingMomentum` | SGD momentum is missing or too weak |
| `AdamHighLearningRateInstability` | Adam is unstable at the current LR |
| `WarmupMissing` | LR warmup is missing |
| `AttentionEntropyCollapse` | Attention entropy collapses |
| `PositionalEncodingProblem` | Positional encodings appear wrong |
| `LongSequenceMemoryBlowup` | Long sequences exhaust memory |
| `RepetitiveGeneration` | Generated text is repetitive or degenerate |
| `BatchNormTrainEvalMismatch` | CNN train/eval behavior diverges |
| `AggressivePoolingOrStride` | CNN stride/pooling removes too much detail |
| `InsufficientSpatialAugmentation` | Spatial augmentation is missing or weak |
| `ChannelCollapse` | CNN channels become redundant |

### Context Facts
Injectable facts that gate specialized rule modules. They are treated like input symptoms in reports and are never diagnosed causes:

| Fact Class | Meaning |
|---|---|
| `ModelIsTransformer` | Enables Transformer-specific rules |
| `ModelIsCNN` | Enables CNN-specific rules |
| `UsesMixedPrecision` | Enables AMP/mixed-precision rules |
| `UsesDistributedTraining` | Enables distributed-training rules |
| `OptimizerIsAdam` | Enables Adam-specific optimizer rules |
| `OptimizerIsSGD` | Enables SGD-specific optimizer rules |

### Cause Facts
Inferred root causes:

| Fact Class | Meaning |
|---|---|
| `LearningRateTooHigh` | LR causes overshoot / oscillation |
| `LearningRateTooLow` | LR too small for meaningful progress |
| `ModelTooComplex` | Too many parameters for available data |
| `ModelTooSimple` | Insufficient model capacity |
| `InsufficientRegularization` | Regularisation absent or too weak |
| `ExcessiveRegularization` | Regularisation suppressing learning |
| `BadWeightInitialization` | Poor initial weight scale |
| `PoorDataQuality` | Noisy, mislabelled, or corrupted data |
| `DistributionShift` | Test distribution differs from train |
| `DataImbalance` | Severe class imbalance |
| `BatchSizeTooLarge` | Batch is too large |
| `BatchSizeTooSmall` | Batch is too small |
| `MomentumTooHigh` | Momentum is too high |
| `WeightDecayTooHigh` | Weight decay is too high |
| `NumericalInstability` | Overflow, NaN, or invalid numeric values |
| `LossScalingMisconfigured` | AMP loss scaling is misconfigured |
| `Fp16RangeExceeded` | Values exceed fp16 numeric range |
| `Fp16GradientUnderflow` | fp16 causes gradients to underflow |
| `DataLoadingBottleneck` | Input pipeline limits multi-GPU throughput |
| `ImproperBatchNormSync` | BatchNorm is not synchronized correctly |
| `GradientSyncOverhead` | Distributed gradient sync dominates step time |
| `LearningRateNotScaled` | LR was not scaled for distributed world size |
| `PerDeviceBatchTooSmall` | Per-device batch is too small |
| `WeightDecayCoupledWithAdam` | Adam uses coupled weight decay |
| `MomentumMisconfigured` | SGD momentum is absent or misconfigured |
| `AdamLearningRateTooHigh` | Adam LR is too high |
| `MissingLearningRateWarmup` | Warmup-sensitive training lacks warmup |
| `AttentionEntropyCollapsed` | Attention entropy collapse harms routing |
| `PositionalEncodingMisconfigured` | Position encoding setup is wrong |
| `QuadraticAttentionMemoryBlowup` | Standard attention memory cost is too high |
| `DegenerateGeneration` | Generation distribution has collapsed |
| `BatchNormModeMismatch` | BatchNorm train/eval handling is wrong |
| `StrideTooAggressive` | CNN stride/pooling is too aggressive |
| `SpatialAugmentationMissing` | CNN spatial augmentation is insufficient |
| `ChannelCollapseCause` | CNN channels collapse into redundant features |

### Recommendation Facts
Actionable next steps:

| Fact Class | Action |
|---|---|
| `ReduceLearningRate` | Lower the LR |
| `IncreaseLearningRate` | Raise the LR |
| `AddDropout` | Add dropout layers |
| `RemoveDropout` | Remove / reduce dropout |
| `AddBatchNormalization` | Insert batch norm layers |
| `UseEarlyStopping` | Monitor val loss; stop early |
| `IncreaseDatasetSize` | Collect more labelled data |
| `ApplyDataAugmentation` | Synthesise training examples |
| `RebalanceClasses` | Oversample / undersample |
| `ImproveLabelQuality` | Audit and clean labels |
| `UseWeightClipping` | Clip gradient norms |
| `UseProperInitialization` | Use He / Xavier / Glorot init |
| `ReduceModelComplexity` | Fewer layers / units |
| `IncreaseModelComplexity` | More layers / units |
| `UseWeightedLoss` | Class-weighted loss function |
| `InspectDataPipeline` | Audit preprocessing for leaks |
| `CollectDomainData` | Gather in-domain test data |
| `ReduceRegularization` | Lower L1/L2/dropout strength |
| `ReduceBatchSize` | Reduce batch size |
| `IncreaseBatchSize` | Increase batch size |
| `ReduceMomentum` | Lower optimizer momentum |
| `ReduceWeightDecay` | Lower weight decay |
| `UseLeakyReLU` | Use LeakyReLU/GELU/ELU |
| `UseGradientClipping` | Clip gradients |
| `TruncateOrChunkInput` | Truncate or chunk long inputs |
| `FixTokenizer` | Audit tokenizer coverage and special tokens |
| `IncreaseReceptiveField` | Widen CNN receptive field |
| `EnableDynamicLossScaling` | Enable dynamic AMP loss scaling |
| `UseBf16` | Use bfloat16 where supported |
| `KeepMasterWeightsInFp32` | Keep fp32 optimizer master weights |
| `UseSyncBatchNorm` | Synchronize BatchNorm across workers |
| `IncreaseDataLoaderWorkers` | Tune data-loader workers and prefetching |
| `UseGradientAccumulation` | Accumulate gradients across microbatches |
| `ScaleLearningRateByWorldSize` | Scale LR for distributed world size |
| `UseAdamW` | Switch coupled Adam decay to AdamW |
| `EnableNesterovMomentum` | Enable Nesterov momentum for SGD |
| `UseLearningRateWarmup` | Add LR warmup |
| `AddLearningRateWarmup` | Add transformer-friendly LR warmup |
| `UseGradientCheckpointing` | Reduce activation memory with checkpointing |
| `UseFlashAttention` | Use memory-efficient attention kernels |
| `ApplyLabelSmoothing` | Apply label smoothing |
| `ClipAttentionLogits` | Clip or temperature-scale attention logits |
| `ChunkOrTruncateInput` | Chunk or truncate transformer inputs |
| `FixPositionalEncoding` | Correct positional encoding setup |
| `AdjustDecodingStrategy` | Tune decoding to avoid repetition |
| `AddSpatialAugmentation` | Add image spatial augmentation |
| `ReduceStride` | Reduce early stride/pooling |
| `FixBatchNormMomentum` | Fix BatchNorm momentum/statistics |
| `UseGlobalAveragePooling` | Use global average pooling |

### Meta Facts
Conflict-resolution facts used for transparent reporting; they are registered but not reported as causes:

| Fact Class | Meaning |
|---|---|
| `SuppressedCause` | Records a contradictory cause hidden from final diagnosis |
| `CauseConflict` | Records the winner, loser(s), and resolution reason |

---

## Rule Hierarchy

### Training Rules (TRAIN_001–TRAIN_012)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| TRAIN_001 | TrainingLossHigh ∧ OscillatingLoss | LearningRateTooHigh |
| TRAIN_002 | TrainingLossHigh ∧ SlowConvergence | LearningRateTooLow |
| TRAIN_003 | GradientExplosion | BadWeightInitialization |
| TRAIN_004 | GradientVanishing | BadWeightInitialization |
| TRAIN_005 | NoisyLabels | PoorDataQuality |
| TRAIN_006 | ClassImbalanceDetected | DataImbalance |
| TRAIN_007 | TrainingAccuracyLow ∧ SmallDataset | UnderfittingObserved |
| TRAIN_008 | TrainingAccuracyLow ∧ ¬SmallDataset | ModelTooSimple |
| TRAIN_009 | UnderfittingObserved | ModelTooSimple |
| TRAIN_010 | GradientExplosion | LearningRateTooHigh |
| TRAIN_011 | NoisyLabels ∧ TrainingLossHigh | PoorDataQuality |
| TRAIN_012 | SlowConvergence ∧ ¬TrainingLossHigh | LearningRateTooLow |

### Validation Rules (VAL_001–VAL_008)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| VAL_001 | TrainingAccuracyHigh ∧ ValidationAccuracyLow | OverfittingObserved |
| VAL_002 | OverfittingObserved | ModelTooComplex |
| VAL_003 | OverfittingObserved | InsufficientRegularization |
| VAL_004 | TrainingAccuracyLow ∧ ValidationAccuracyLow | UnderfittingObserved |
| VAL_005 | UnderfittingObserved | ModelTooSimple |
| VAL_006 | ValidationLossHigh ∧ SmallDataset | InsufficientRegularization |
| VAL_007 | DataLeakageSuspected ∧ ValidationAccuracyLow | ExcessiveRegularization |
| VAL_008 | UnderfittingObserved ∧ ¬ModelTooSimple | ExcessiveRegularization |

### Testing Rules (TEST_001–TEST_006)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| TEST_001 | ValidationAccuracyHigh ∧ TestAccuracyLow | DistributionShift |
| TEST_002 | DistributionShift | PoorGeneralization |
| TEST_003 | DataLeakageSuspected ∧ TestAccuracyLow | DistributionShift |
| TEST_004 | ClassImbalanceDetected ∧ TestAccuracyLow | DataImbalance |
| TEST_005 | ValidationLossHigh ∧ TestAccuracyLow | PoorGeneralization |
| TEST_006 | PoorGeneralization ∧ ¬ValidationAccuracyHigh | DistributionShift |

### Optimization Rules (OPT_001-OPT_009)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| OPT_001 | GradientExplosion | UseWeightClipping |
| OPT_002 | BadWeightInitialization | UseProperInitialization |
| OPT_003 | GradientVanishing | AddBatchNormalization |
| OPT_004 | LearningRateTooHigh ∧ OscillatingLoss | UseWeightClipping |
| OPT_005 | LearningRateTooLow ∧ SlowConvergence | AddBatchNormalization |
| OPT_006 | GradientExplosion ∧ GradientVanishing | BadWeightInitialization |
| OPT_007 | NaNLoss | NumericalInstability |
| OPT_008 | NumericalInstability | UseGradientClipping |
| OPT_009 | OscillatingLoss ∧ NumericalInstability | MomentumTooHigh |

### Architecture Rules (ARCH_001-ARCH_009)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| ARCH_001 | ModelTooComplex | ReduceModelComplexity |
| ARCH_002 | ModelTooSimple | IncreaseModelComplexity |
| ARCH_003 | OverfittingObserved ∧ SmallDataset | ApplyDataAugmentation |
| ARCH_004 | SmallDataset ∧ UnderfittingObserved | IncreaseDatasetSize |
| ARCH_005 | InsufficientRegularization | AddDropout |
| ARCH_006 | ExcessiveRegularization | ReduceRegularization |
| ARCH_007 | GradientVanishing ∧ LargeDataset | AddBatchNormalization |
| ARCH_008 | OverfittingObserved ∧ LargeDataset | AddDropout |
| ARCH_009 | DeadReLUDetected | UseLeakyReLU |

### Data Rules (DATA_001-DATA_005)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| DATA_001 | DataLeakageSuspected | InspectDataPipeline |
| DATA_002 | ValidationAccuracyHigh ∧ TestAccuracyLow | DistributionShift |
| DATA_003 | NoisyLabels | PoorDataQuality |
| DATA_004 | PoorDataQuality | ImproveLabelQuality |
| DATA_005 | ClassImbalanceDetected | DataImbalance |

### Mixed Precision Rules (AMP_001-AMP_007)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| AMP_001 | UsesMixedPrecision ∧ NaNLoss | Fp16RangeExceeded |
| AMP_002 | UsesMixedPrecision ∧ MixedPrecisionOverflow | Fp16RangeExceeded |
| AMP_003 | UsesMixedPrecision ∧ LossScaleOverflow | LossScalingMisconfigured |
| AMP_004 | UsesMixedPrecision ∧ GradientUnderflow | Fp16GradientUnderflow |
| AMP_005 | UsesMixedPrecision ∧ LossScalingMisconfigured | EnableDynamicLossScaling |
| AMP_006 | UsesMixedPrecision ∧ Fp16RangeExceeded | UseBf16 |
| AMP_007 | UsesMixedPrecision ∧ Fp16GradientUnderflow | KeepMasterWeightsInFp32 |

### Distributed Rules (DIST_001-DIST_010)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| DIST_001 | UsesDistributedTraining ∧ MultiGpuThroughputLow ∧ GpuUnderutilization | DataLoadingBottleneck |
| DIST_002 | UsesDistributedTraining ∧ BatchNormDesync | ImproperBatchNormSync |
| DIST_003 | UsesDistributedTraining ∧ GradientSyncSlow | GradientSyncOverhead |
| DIST_004 | UsesDistributedTraining ∧ PerDeviceBatchTooSmallObserved | PerDeviceBatchTooSmall |
| DIST_005 | UsesDistributedTraining ∧ SlowConvergence | LearningRateNotScaled |
| DIST_006 | UsesDistributedTraining ∧ DataLoadingBottleneck | IncreaseDataLoaderWorkers |
| DIST_007 | UsesDistributedTraining ∧ ImproperBatchNormSync | UseSyncBatchNorm |
| DIST_008 | UsesDistributedTraining ∧ GradientSyncOverhead | UseGradientAccumulation |
| DIST_009 | UsesDistributedTraining ∧ LearningRateNotScaled | ScaleLearningRateByWorldSize |
| DIST_010 | UsesDistributedTraining ∧ PerDeviceBatchTooSmall | UseGradientAccumulation |

### Optimizer Interaction Rules (OPTZ_001-OPTZ_006)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| OPTZ_001 | OptimizerIsAdam ∧ CoupledWeightDecayUsed | WeightDecayCoupledWithAdam |
| OPTZ_002 | OptimizerIsAdam ∧ WeightDecayCoupledWithAdam | UseAdamW |
| OPTZ_003 | OptimizerIsSGD ∧ MissingMomentum | MomentumMisconfigured |
| OPTZ_004 | OptimizerIsSGD ∧ MomentumMisconfigured | EnableNesterovMomentum |
| OPTZ_005 | OptimizerIsAdam ∧ AdamHighLearningRateInstability | AdamLearningRateTooHigh |
| OPTZ_006 | OptimizerIsAdam ∧ AdamLearningRateTooHigh | UseLearningRateWarmup |

### Transformer Rules (TF_001-TF_017)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| TF_001 | ModelIsTransformer ∧ AttentionCollapse | PoorGeneralization |
| TF_002 | ModelIsTransformer ∧ TokenizationIssue | FixTokenizer |
| TF_003 | ModelIsTransformer ∧ ContextLengthExceeded | TruncateOrChunkInput |
| TF_004 | ModelIsTransformer ∧ AttentionCollapse | AddBatchNormalization |
| TF_005 | ModelIsTransformer ∧ WarmupMissing | MissingLearningRateWarmup |
| TF_006 | ModelIsTransformer ∧ MissingLearningRateWarmup | AddLearningRateWarmup |
| TF_007 | ModelIsTransformer ∧ AttentionEntropyCollapse | AttentionEntropyCollapsed |
| TF_008 | ModelIsTransformer ∧ AttentionEntropyCollapsed | ClipAttentionLogits |
| TF_009 | ModelIsTransformer ∧ PositionalEncodingProblem | PositionalEncodingMisconfigured |
| TF_010 | ModelIsTransformer ∧ PositionalEncodingMisconfigured | FixPositionalEncoding |
| TF_011 | ModelIsTransformer ∧ LongSequenceMemoryBlowup | QuadraticAttentionMemoryBlowup |
| TF_012 | ModelIsTransformer ∧ QuadraticAttentionMemoryBlowup | UseGradientCheckpointing |
| TF_013 | ModelIsTransformer ∧ QuadraticAttentionMemoryBlowup | UseFlashAttention |
| TF_014 | ModelIsTransformer ∧ RepetitiveGeneration | DegenerateGeneration |
| TF_015 | ModelIsTransformer ∧ DegenerateGeneration | ApplyLabelSmoothing |
| TF_016 | ModelIsTransformer ∧ DegenerateGeneration | AdjustDecodingStrategy |
| TF_017 | ModelIsTransformer ∧ ContextLengthExceeded | ChunkOrTruncateInput |

### CNN Rules (CNN_001-CNN_011)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| CNN_001 | ModelIsCNN ∧ FeatureCollapse | PoorGeneralization |
| CNN_002 | ModelIsCNN ∧ ReceptiveFieldTooSmall | IncreaseReceptiveField |
| CNN_003 | ModelIsCNN ∧ DeadReLUDetected | UseLeakyReLU |
| CNN_004 | ModelIsCNN ∧ BatchNormTrainEvalMismatch | BatchNormModeMismatch |
| CNN_005 | ModelIsCNN ∧ BatchNormModeMismatch | FixBatchNormMomentum |
| CNN_006 | ModelIsCNN ∧ AggressivePoolingOrStride | StrideTooAggressive |
| CNN_007 | ModelIsCNN ∧ StrideTooAggressive | ReduceStride |
| CNN_008 | ModelIsCNN ∧ InsufficientSpatialAugmentation | SpatialAugmentationMissing |
| CNN_009 | ModelIsCNN ∧ SpatialAugmentationMissing | AddSpatialAugmentation |
| CNN_010 | ModelIsCNN ∧ ChannelCollapse | ChannelCollapseCause |
| CNN_011 | ModelIsCNN ∧ ChannelCollapseCause | UseGlobalAveragePooling |

### Recommendation Rules (REC_001-REC_020)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| REC_001 | LearningRateTooHigh | ReduceLearningRate |
| REC_002 | LearningRateTooLow | IncreaseLearningRate |
| REC_003 | ModelTooComplex | AddDropout |
| REC_004 | ModelTooComplex | UseEarlyStopping |
| REC_005 | ModelTooSimple | IncreaseModelComplexity |
| REC_006 | InsufficientRegularization | AddDropout |
| REC_007 | InsufficientRegularization | UseEarlyStopping |
| REC_008 | ExcessiveRegularization | RemoveDropout |
| REC_009 | BadWeightInitialization | UseProperInitialization |
| REC_010 | PoorDataQuality | ImproveLabelQuality |
| REC_011 | DataImbalance | RebalanceClasses |
| REC_012 | DataImbalance | UseWeightedLoss |
| REC_013 | DistributionShift | CollectDomainData |
| REC_014 | DistributionShift | ApplyDataAugmentation |
| REC_015 | PoorGeneralization | InspectDataPipeline |
| REC_016 | OverfittingObserved | IncreaseDatasetSize |
| REC_017 | BatchSizeTooLarge | ReduceBatchSize |
| REC_018 | BatchSizeTooSmall | IncreaseBatchSize |
| REC_019 | MomentumTooHigh | ReduceMomentum |
| REC_020 | WeightDecayTooHigh | ReduceWeightDecay |

---

## Conflict Resolution

Contradictory causes are resolved by low-salience Experta meta-rules (`CONFLICT_001`-`CONFLICT_009`) after ordinary diagnostic and recommendation rules have fired. The rules do not retract facts; they declare `CauseConflict` and `SuppressedCause` meta facts. `get_diagnosis()` then filters suppressed causes and recommendations supported only by suppressed causes from the reported result, while preserving the full explanation chain and noisy-OR confidence aggregation.

Resolution policy:
1. Combine each contending cause's supporting `Explanation.confidence` values with the same noisy-OR formula used in reporting.
2. Keep the cause with the highest evidence strength.
3. If evidence strength ties, keep the cause with more supporting explanations.
4. If still tied, keep the lexicographically first cause name for deterministic output.

| Rule ID | Contending causes |
|---|---|
| CONFLICT_001 | OverfittingObserved vs UnderfittingObserved |
| CONFLICT_002 | LearningRateTooHigh vs LearningRateTooLow |
| CONFLICT_003 | LearningRateTooHigh vs LearningRateNotScaled |
| CONFLICT_004 | LearningRateTooLow vs AdamLearningRateTooHigh |
| CONFLICT_005 | LearningRateNotScaled vs AdamLearningRateTooHigh |
| CONFLICT_006 | ModelTooComplex vs ModelTooSimple |
| CONFLICT_007 | InsufficientRegularization vs ExcessiveRegularization |
| CONFLICT_008 | BatchSizeTooLarge vs BatchSizeTooSmall |
| CONFLICT_009 | MomentumTooHigh vs MomentumMisconfigured |

---

## Installation

```bash
# Clone / unzip the project
pip install -r requirements.txt
```

> **Note:** Requires `frozendict==2.3.4` (overrides experta's default) for Python 3.12 compatibility.

---

## Usage

### Named Scenarios

```bash
python main.py scenario=overfitting
python main.py scenario=gradient_explosion
python main.py scenario=distribution_shift
python main.py scenario=underfitting
python main.py scenario=gradient_vanishing
python main.py scenario=noisy_labels
python main.py scenario=class_imbalance
python main.py scenario=slow_convergence
python main.py --list          # show all built-in scenarios
```

### Custom Symptom Sets

```bash
python main.py symptoms=TrainingLossHigh,OscillatingLoss
python main.py symptoms=TrainingAccuracyHigh,ValidationAccuracyLow,SmallDataset
python main.py symptoms=GradientExplosion,GradientVanishing
python main.py symptoms=UsesMixedPrecision,NaNLoss
python main.py symptoms=ModelIsTransformer,LongSequenceMemoryBlowup
```

---

## Example Executions

### Scenario 1 – Overfitting

```
Input symptoms : TrainingAccuracyHigh, ValidationAccuracyLow
Derived causes : OverfittingObserved, ModelTooComplex, InsufficientRegularization
Recommendations: AddDropout, UseEarlyStopping, ReduceModelComplexity, IncreaseDatasetSize
Rules fired    : VAL_001, VAL_002, VAL_003, ARCH_001, ARCH_005, REC_003, REC_004, REC_016
```

### Scenario 2 – Oscillating Loss (High LR)

```
Input symptoms : TrainingLossHigh, OscillatingLoss
Derived causes : LearningRateTooHigh
Recommendations: ReduceLearningRate, UseWeightClipping
Rules fired    : TRAIN_001, OPT_004, REC_001
```

### Scenario 3 – Distribution Shift

```
Input symptoms : ValidationAccuracyHigh, TestAccuracyLow
Derived causes : DistributionShift, PoorGeneralization
Recommendations: CollectDomainData, ApplyDataAugmentation, InspectDataPipeline
Rules fired    : TEST_001, TEST_002, REC_013, REC_014, REC_015
```

---

## Running Tests

```bash
cd dl_debugger
python -m pytest tests/ -v
```

All 214 tests use real Experta inference — no mocks.

Test coverage:
- Rule firing and fact derivation for all rule modules
- Context gating for Transformer, CNN, AMP, distributed, and optimizer rules
- Conflict resolution and suppressed-cause reporting filters
- Recommendation generation from every cause
- Explanation structure and audit-trail integrity
- Engine reset and fact isolation between runs
- Fact deduplication

---

## Extending the System

### Adding a New Symptom Fact

1. Add a class in `models/facts.py` inheriting from `Fact`.
2. Add it to `SYMPTOM_FACT_CLASSES` in `engine/knowledge_engine.py`.
3. Write rules that reference it in the appropriate `engine/rules/` module.

### Adding a New Rule

1. Open the relevant rule module (e.g. `training_rules.py`).
2. Decorate a method with `@Rule(...)` following the existing pattern.
3. Use `NOT(FactClass())` guards to prevent duplicate assertions.
4. Call `self.declare(Explanation(...))` with a unique `rule_id`.
5. Add a pytest test in the matching `tests/test_*.py` file.

### Adding a New Scenario

Either add to `data/scenario_loader.py`'s `BUILTIN_SCENARIOS` dict, or add a JSON entry to one of the `data/scenarios/*.json` files.

---

## Design Principles

- **Pure rule-driven inference** — zero `if/elif/else` in the diagnostic path.
- **NOT() guards** — every rule that asserts a fact guards with `NOT(FactClass())` to prevent duplicates and infinite loops.
- **XAI by default** — every rule fires an `Explanation` fact alongside the derived fact, giving a complete audit trail.
- **Injected vs derived** — the engine tracks which facts were user-supplied so `OverfittingObserved` correctly appears as a derived cause rather than a user symptom.
- **Multiple inheritance composition** — `DebuggingKnowledgeEngine` inherits from all rule classes; Experta merges all `@Rule` methods via the Rete algorithm.
