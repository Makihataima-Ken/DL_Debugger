# Knowledge-Based Expert System for Systematic Debugging of Deep Learning Model Outputs

A pure Experta rule-based expert system that helps AI students and developers diagnose problems in deep learning model training, validation, and testing. All reasoning emerges exclusively from Experta rule activation — no procedural decision logic exists outside rules.

---

## Architecture

```
dl_debugger/
├── main.py                          # CLI entry point
│
├── engine/
│   ├── knowledge_engine.py          # DebuggingKnowledgeEngine + DiagnosisResult
│   └── rules/
│       ├── training_rules.py        # TRAIN_001–TRAIN_012
│       ├── validation_rules.py      # VAL_001–VAL_008
│       ├── testing_rules.py         # TEST_001–TEST_006
│       ├── optimization_rules.py    # OPT_001–OPT_006
│       ├── architecture_rules.py    # ARCH_001–ARCH_008
│       └── recommendation_rules.py # REC_001–REC_016
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
│   └── test_testing.py
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

### Optimization Rules (OPT_001–OPT_006)
| Rule ID | Antecedents | Consequent |
|---|---|---|
| OPT_001 | GradientExplosion | UseWeightClipping |
| OPT_002 | BadWeightInitialization | UseProperInitialization |
| OPT_003 | GradientVanishing | AddBatchNormalization |
| OPT_004 | LearningRateTooHigh ∧ OscillatingLoss | UseWeightClipping |
| OPT_005 | LearningRateTooLow ∧ SlowConvergence | AddBatchNormalization |
| OPT_006 | GradientExplosion ∧ GradientVanishing | BadWeightInitialization |

### Architecture Rules (ARCH_001–ARCH_008)
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

### Recommendation Rules (REC_001–REC_016)
Translate causes directly into recommendations. Each rule fires once per run using `NOT()` guards to avoid duplicates.

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

All 71 tests use real Experta inference — no mocks.

Test coverage:
- Rule firing and fact derivation for all rule modules
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
- **Multiple inheritance composition** — `DebuggingKnowledgeEngine` inherits from all six rule classes; Experta merges all `@Rule` methods via the Rete algorithm.
