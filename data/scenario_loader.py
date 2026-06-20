"""
scenario_loader.py
==================
Loads and validates scenario JSON files for the expert system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.helpers import validate_scenario, validate_symptom_names
from engine.knowledge_engine import FACT_CLASS_REGISTRY

SCENARIOS_DIR = Path(__file__).parent / "scenarios"

# Built-in named scenarios for CLI use
BUILTIN_SCENARIOS: dict[str, list[str]] = {
    "overfitting": ["TrainingAccuracyHigh", "ValidationAccuracyLow"],
    "overfitting_small": ["TrainingAccuracyHigh", "ValidationAccuracyLow", "SmallDataset"],
    "underfitting": ["TrainingAccuracyLow", "ValidationAccuracyLow"],
    "oscillating_loss": ["TrainingLossHigh", "OscillatingLoss"],
    "slow_convergence": ["TrainingLossHigh", "SlowConvergence"],
    "gradient_explosion": ["GradientExplosion"],
    "gradient_vanishing": ["GradientVanishing"],
    "distribution_shift": ["ValidationAccuracyHigh", "TestAccuracyLow"],
    "data_leakage": ["DataLeakageSuspected", "TestAccuracyLow"],
    "class_imbalance": ["ClassImbalanceDetected", "TestAccuracyLow"],
    "noisy_labels": ["NoisyLabels", "TrainingLossHigh"],
    "poor_generalisation": ["ValidationLossHigh", "TestAccuracyLow"],
}


def load_scenario_file(path: Path) -> list[dict[str, Any]]:
    """Load and validate a scenario JSON file.

    Parameters
    ----------
    path:
        Absolute or relative path to a JSON scenario file.

    Returns
    -------
    list[dict[str, Any]]
        List of valid scenario dicts.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file is not valid JSON or contains no valid scenarios.
    """
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw: Any = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(raw).__name__}")

    valid: list[dict[str, Any]] = []
    known = set(FACT_CLASS_REGISTRY)

    for item in raw:
        if not validate_scenario(item):
            continue
        item["symptoms"] = validate_symptom_names(item["symptoms"], known)
        valid.append(item)

    return valid


def load_all_scenarios() -> dict[str, list[dict[str, Any]]]:
    """Load all three built-in scenario files.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Mapping of category name → list of scenario dicts.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    for category in ("training", "validation", "testing"):
        path = SCENARIOS_DIR / f"{category}_examples.json"
        try:
            result[category] = load_scenario_file(path)
        except (FileNotFoundError, ValueError):
            result[category] = []
    return result


def get_builtin_symptoms(scenario_name: str) -> list[str]:
    """Return the symptom list for a built-in named scenario.

    Parameters
    ----------
    scenario_name:
        Key from :data:`BUILTIN_SCENARIOS`.

    Returns
    -------
    list[str]
        Symptom class names.

    Raises
    ------
    KeyError
        If *scenario_name* is not a recognised built-in scenario.
    """
    if scenario_name not in BUILTIN_SCENARIOS:
        available = ", ".join(sorted(BUILTIN_SCENARIOS))
        raise KeyError(
            f"Unknown scenario '{scenario_name}'. "
            f"Available: {available}"
        )
    return BUILTIN_SCENARIOS[scenario_name]
