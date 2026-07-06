"""
problem_catalog.py
==================
Curated natural-language problem prompts exposed to the browser UI.

These are user-facing examples only. Diagnosis still flows through the NLP
extractor and rule engine when a prompt is selected or typed manually.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TypedDict


class ProblemOption(TypedDict):
    id: str
    category: str
    text: str


_RAW_PROBLEMS: tuple[tuple[str, str], ...] = (
    # Quick examples from the prompt.
    ("Quick Examples", "multi gpu training throughput is low and the gpus are idle"),
    ("Quick Examples", "labels are noisy and training loss stays high"),
    ("Quick Examples", "the transformer runs out of memory on long sequences"),
    ("Quick Examples", "my gradients are exploding and the loss becomes NaN"),
    ("Quick Examples", "both training and validation accuracy are low"),
    ("Quick Examples", "training accuracy is high but validation accuracy is low"),
    ("Quick Examples", "training loss is high and loss keeps oscillating"),
    # Core training.
    ("Core Training", "training loss is high and the loss keeps oscillating"),
    (
        "Core Training",
        "the model is converging very slowly and the training loss is still high",
    ),
    ("Core Training", "loss became NaN after a few epochs"),
    ("Core Training", "gradients are exploding during training"),
    (
        "Core Training",
        "the early layers are barely learning and gradients look near zero",
    ),
    # Validation / generalization.
    (
        "Validation / Generalization",
        "training accuracy is high but validation accuracy is low",
    ),
    (
        "Validation / Generalization",
        "both training accuracy and validation accuracy are low",
    ),
    (
        "Validation / Generalization",
        "validation loss is high and the dataset is small",
    ),
    (
        "Validation / Generalization",
        "the model generalizes poorly on unseen data",
    ),
    (
        "Validation / Generalization",
        "validation accuracy is high but test accuracy is low",
    ),
    # Data problems.
    ("Data Problems", "the dataset is small and the model is overfitting"),
    ("Data Problems", "labels are noisy and training loss stays high"),
    (
        "Data Problems",
        "there is severe class imbalance and test accuracy is low",
    ),
    ("Data Problems", "I suspect data leakage and validation accuracy is low"),
    (
        "Data Problems",
        "test data seems from a different distribution than training data",
    ),
    # Transformer cases.
    (
        "Transformer Cases",
        "my transformer attention collapses and outputs repetitive text",
    ),
    (
        "Transformer Cases",
        "the transformer runs out of memory on long sequences",
    ),
    ("Transformer Cases", "tokenization looks wrong and the model is a transformer"),
    ("Transformer Cases", "the transformer has positional encoding problems"),
    ("Transformer Cases", "attention entropy collapses and warmup is missing"),
    # CNN cases.
    ("CNN Cases", "my CNN features collapse and generalization is poor"),
    ("CNN Cases", "the CNN receptive field is too small"),
    (
        "CNN Cases",
        "batch norm behaves differently between train and eval in my CNN",
    ),
    (
        "CNN Cases",
        "the CNN uses aggressive pooling and loses spatial details",
    ),
    ("CNN Cases", "my CNN has dead ReLU units"),
    # Mixed precision / distributed.
    ("Mixed Precision / Distributed", "I use mixed precision and the loss becomes NaN"),
    ("Mixed Precision / Distributed", "fp16 training has gradient underflow"),
    (
        "Mixed Precision / Distributed",
        "multi GPU training has low throughput and GPUs are underutilized",
    ),
    (
        "Mixed Precision / Distributed",
        "distributed training has slow gradient synchronization",
    ),
    (
        "Mixed Precision / Distributed",
        "batch norm stats are desynchronized across GPUs",
    ),
    # Negation / robustness.
    ("Negation / Robustness", "loss oscillates but there is no slow convergence"),
    ("Negation / Robustness", "training loss is high, but no gradient explosion"),
    (
        "Negation / Robustness",
        "maybe the learning rate is too high because loss seems to oscillate",
    ),
    (
        "Negation / Robustness",
        "not overfitting, but validation accuracy is low and test accuracy is low",
    ),
    (
        "Negation / Robustness",
        "the model is free of NaN loss but gradients are vanishing",
    ),
    # Additional supported examples already covered by tests and vocabulary.
    ("Additional Examples", "my transformer has lots of unk tokens"),
    ("Additional Examples", "The resnet feature maps are constant"),
    ("Additional Examples", "Using adam with coupled weight decay"),
    ("Additional Examples", "my transformer input exceeds the context length"),
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "problem"


def get_problem_options() -> list[ProblemOption]:
    """Return stable, grouped problem prompts for the web UI."""
    counters: defaultdict[str, int] = defaultdict(int)
    options: list[ProblemOption] = []

    for category, text in _RAW_PROBLEMS:
        counters[category] += 1
        options.append(
            {
                "id": f"{_slugify(category)}-{counters[category]:02d}",
                "category": category,
                "text": text,
            }
        )

    return options


def get_problem_categories() -> list[str]:
    """Return categories in display order."""
    seen: set[str] = set()
    categories: list[str] = []
    for category, _ in _RAW_PROBLEMS:
        if category in seen:
            continue
        seen.add(category)
        categories.append(category)
    return categories
