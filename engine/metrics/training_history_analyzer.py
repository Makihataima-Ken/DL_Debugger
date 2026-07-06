"""
training_history_analyzer.py
============================
CSV training-history input adapter.

This module translates observed metric patterns into existing Fact class
names. It does not diagnose root causes directly; all diagnosis remains in
the Experta rule engine.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
    "learning_rate",
)

TAIL_WINDOW = 3
LOW_ACCURACY_THRESHOLD = 0.60
HIGH_ACCURACY_THRESHOLD = 0.85
HIGH_LOSS_THRESHOLD = 1.00
MIN_LOSS_IMPROVEMENT_FOR_PROGRESS = 0.15


@dataclass(frozen=True)
class TrainingHistoryRow:
    """One parsed CSV row of training metrics."""

    epoch: float
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float


@dataclass(frozen=True)
class MetricEvidence:
    """Evidence item explaining why a metric-derived fact was emitted."""

    fact_name: str
    metric: str
    confidence: float
    reason: str
    value: float | None = None


@dataclass
class TrainingHistoryAnalysis:
    """Metric adapter output consumed by the knowledge engine."""

    source: str
    rows: list[TrainingHistoryRow]
    columns: list[str]
    fact_names: list[str] = field(default_factory=list)
    root_confidence: dict[str, float] = field(default_factory=dict)
    evidence: list[MetricEvidence] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def all_fact_names(self) -> list[str]:
        """Fact names to inject into the engine."""
        return list(self.fact_names)


def analyze_training_history_csv(path: str | Path) -> TrainingHistoryAnalysis:
    """Read a training-history CSV file and return metric-derived facts."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training history CSV not found: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Training history path is not a file: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return _analyze_reader(csv.DictReader(handle), source=str(csv_path))


def analyze_training_history_text(
    csv_text: str,
    source: str = "uploaded CSV",
) -> TrainingHistoryAnalysis:
    """Analyze CSV content supplied as a string."""
    if not csv_text.strip():
        raise ValueError("Training history CSV content is empty.")
    return _analyze_reader(csv.DictReader(io.StringIO(csv_text)), source=source)


def _analyze_reader(
    reader: csv.DictReader,
    source: str,
) -> TrainingHistoryAnalysis:
    fieldnames = reader.fieldnames or []
    column_map = _column_map(fieldnames)
    missing = [name for name in REQUIRED_COLUMNS if name not in column_map]
    if missing:
        raise ValueError(
            "Training history CSV is missing required column(s): "
            + ", ".join(missing)
        )

    rows: list[TrainingHistoryRow] = []
    for row_number, raw in enumerate(reader, start=2):
        if not _row_has_content(raw.values()):
            continue
        rows.append(_parse_row(raw, column_map, row_number))

    if not rows:
        raise ValueError("Training history CSV contains no data rows.")

    analysis = TrainingHistoryAnalysis(
        source=source,
        rows=rows,
        columns=list(fieldnames),
        summary=_metric_summary(rows),
    )
    _derive_metric_facts(analysis)
    return analysis


def _column_map(fieldnames: Iterable[str]) -> dict[str, str]:
    return {
        _normalise_column(name): name
        for name in fieldnames
        if name is not None
    }


def _normalise_column(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _row_has_content(values: Iterable[object]) -> bool:
    return any(str(value or "").strip() for value in values)


def _parse_row(
    raw: dict[str, str],
    column_map: dict[str, str],
    row_number: int,
) -> TrainingHistoryRow:
    def value(column: str) -> float:
        key = column_map[column]
        raw_value = raw.get(key, "")
        try:
            return float(str(raw_value).strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value for '{column}' on CSV row {row_number}: "
                f"{raw_value!r}"
            ) from exc

    epoch = value("epoch")
    if not math.isfinite(epoch):
        raise ValueError(f"Invalid epoch value on CSV row {row_number}: {epoch!r}")

    return TrainingHistoryRow(
        epoch=epoch,
        train_loss=value("train_loss"),
        val_loss=value("val_loss"),
        train_accuracy=value("train_accuracy"),
        val_accuracy=value("val_accuracy"),
        learning_rate=value("learning_rate"),
    )


def _derive_metric_facts(analysis: TrainingHistoryAnalysis) -> None:
    rows = analysis.rows
    train_loss = [row.train_loss for row in rows]
    val_loss = [row.val_loss for row in rows]
    train_accuracy = [row.train_accuracy for row in rows]
    val_accuracy = [row.val_accuracy for row in rows]

    non_finite_metrics = [
        metric
        for metric, values in (
            ("train_loss", train_loss),
            ("val_loss", val_loss),
            ("train_accuracy", train_accuracy),
            ("val_accuracy", val_accuracy),
            ("learning_rate", [row.learning_rate for row in rows]),
        )
        if any(not math.isfinite(value) for value in values)
    ]
    if non_finite_metrics:
        _add_fact(
            analysis,
            "NaNLoss",
            "loss",
            0.95,
            "Non-finite metric values were found in "
            + ", ".join(non_finite_metrics)
            + ".",
        )

    train_tail = _tail_mean(train_accuracy)
    if train_tail is not None:
        if train_tail <= LOW_ACCURACY_THRESHOLD:
            _add_fact(
                analysis,
                "TrainingAccuracyLow",
                "train_accuracy",
                _threshold_confidence(
                    train_tail,
                    LOW_ACCURACY_THRESHOLD,
                    below=True,
                ),
                (
                    f"Recent training accuracy average {train_tail:.3f} is below "
                    f"{LOW_ACCURACY_THRESHOLD:.2f}."
                ),
                train_tail,
            )
        elif train_tail >= HIGH_ACCURACY_THRESHOLD:
            _add_fact(
                analysis,
                "TrainingAccuracyHigh",
                "train_accuracy",
                _threshold_confidence(
                    train_tail,
                    HIGH_ACCURACY_THRESHOLD,
                    below=False,
                ),
                (
                    f"Recent training accuracy average {train_tail:.3f} is above "
                    f"{HIGH_ACCURACY_THRESHOLD:.2f}."
                ),
                train_tail,
            )

    val_tail = _tail_mean(val_accuracy)
    if val_tail is not None:
        if val_tail <= LOW_ACCURACY_THRESHOLD:
            _add_fact(
                analysis,
                "ValidationAccuracyLow",
                "val_accuracy",
                _threshold_confidence(
                    val_tail,
                    LOW_ACCURACY_THRESHOLD,
                    below=True,
                ),
                (
                    f"Recent validation accuracy average {val_tail:.3f} is below "
                    f"{LOW_ACCURACY_THRESHOLD:.2f}."
                ),
                val_tail,
            )
        elif val_tail >= HIGH_ACCURACY_THRESHOLD:
            _add_fact(
                analysis,
                "ValidationAccuracyHigh",
                "val_accuracy",
                _threshold_confidence(
                    val_tail,
                    HIGH_ACCURACY_THRESHOLD,
                    below=False,
                ),
                (
                    f"Recent validation accuracy average {val_tail:.3f} is above "
                    f"{HIGH_ACCURACY_THRESHOLD:.2f}."
                ),
                val_tail,
            )

    train_loss_tail = _tail_mean(train_loss)
    if train_loss_tail is not None and train_loss_tail >= HIGH_LOSS_THRESHOLD:
        _add_fact(
            analysis,
            "TrainingLossHigh",
            "train_loss",
            _threshold_confidence(
                train_loss_tail,
                HIGH_LOSS_THRESHOLD,
                below=False,
            ),
            (
                f"Recent training loss average {train_loss_tail:.3f} is above "
                f"{HIGH_LOSS_THRESHOLD:.2f}."
            ),
            train_loss_tail,
        )

    val_loss_tail = _tail_mean(val_loss)
    if val_loss_tail is not None and val_loss_tail >= HIGH_LOSS_THRESHOLD:
        _add_fact(
            analysis,
            "ValidationLossHigh",
            "val_loss",
            _threshold_confidence(
                val_loss_tail,
                HIGH_LOSS_THRESHOLD,
                below=False,
            ),
            (
                f"Recent validation loss average {val_loss_tail:.3f} is above "
                f"{HIGH_LOSS_THRESHOLD:.2f}."
            ),
            val_loss_tail,
        )

    oscillation = _oscillation_signal(train_loss, "train_loss")
    val_oscillation = _oscillation_signal(val_loss, "val_loss")
    best_oscillation = max(oscillation, val_oscillation, key=lambda item: item[0])
    if best_oscillation[0] >= 2:
        changes, metric_name, confidence = best_oscillation
        _add_fact(
            analysis,
            "OscillatingLoss",
            metric_name,
            confidence,
            (
                f"{metric_name} changed direction {changes} times across epochs, "
                "indicating unstable optimization."
            ),
        )

    if "OscillatingLoss" not in analysis.fact_names:
        _maybe_add_slow_convergence(analysis, train_loss)

    _add_quality_warnings(analysis)


def _add_fact(
    analysis: TrainingHistoryAnalysis,
    fact_name: str,
    metric: str,
    confidence: float,
    reason: str,
    value: float | None = None,
) -> None:
    confidence = _clamp_confidence(confidence)
    if fact_name not in analysis.fact_names:
        analysis.fact_names.append(fact_name)
    analysis.root_confidence[fact_name] = max(
        analysis.root_confidence.get(fact_name, 0.0),
        confidence,
    )
    analysis.evidence.append(
        MetricEvidence(
            fact_name=fact_name,
            metric=metric,
            confidence=confidence,
            reason=reason,
            value=value,
        )
    )


def _metric_summary(rows: list[TrainingHistoryRow]) -> dict[str, dict[str, float]]:
    metrics = {
        "train_loss": [row.train_loss for row in rows],
        "val_loss": [row.val_loss for row in rows],
        "train_accuracy": [row.train_accuracy for row in rows],
        "val_accuracy": [row.val_accuracy for row in rows],
        "learning_rate": [row.learning_rate for row in rows],
    }
    summary: dict[str, dict[str, float]] = {
        "epochs": {
            "count": float(len(rows)),
            "first": rows[0].epoch,
            "last": rows[-1].epoch,
        }
    }

    for name, values in metrics.items():
        finite = _finite(values)
        if not finite:
            continue
        summary[name] = {
            "start": finite[0],
            "end": finite[-1],
            "min": min(finite),
            "max": max(finite),
            "mean": sum(finite) / len(finite),
            "tail_mean": sum(finite[-TAIL_WINDOW:]) / min(TAIL_WINDOW, len(finite)),
        }

    return summary


def _finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _tail_mean(values: list[float]) -> float | None:
    finite = _finite(values)
    if not finite:
        return None
    tail = finite[-TAIL_WINDOW:]
    return sum(tail) / len(tail)


def _threshold_confidence(value: float, threshold: float, below: bool) -> float:
    distance = (threshold - value) if below else (value - threshold)
    scaled = max(0.0, distance) / max(abs(threshold), 1e-9)
    return _clamp_confidence(0.65 + min(0.28, scaled * 0.55))


def _oscillation_signal(values: list[float], metric_name: str) -> tuple[int, str, float]:
    finite = _finite(values)
    if len(finite) < 4:
        return (0, metric_name, 0.0)

    mean_abs = sum(abs(value) for value in finite) / len(finite)
    epsilon = max(0.01, mean_abs * 0.03)
    signs: list[int] = []
    for previous, current in zip(finite, finite[1:]):
        delta = current - previous
        if abs(delta) <= epsilon:
            continue
        signs.append(1 if delta > 0 else -1)

    changes = sum(
        1
        for previous, current in zip(signs, signs[1:])
        if previous != current
    )
    swing = max(finite) - min(finite)
    min_swing = max(0.1, mean_abs * 0.12)
    if changes < 2 or swing < min_swing:
        return (0, metric_name, 0.0)

    confidence = _clamp_confidence(
        0.60
        + min(0.25, changes * 0.06)
        + min(0.10, swing / max(mean_abs, 1e-9) * 0.08)
    )
    return (changes, metric_name, confidence)


def _maybe_add_slow_convergence(
    analysis: TrainingHistoryAnalysis,
    train_loss: list[float],
) -> None:
    finite = _finite(train_loss)
    if len(finite) < 3:
        return

    first = finite[0]
    last = finite[-1]
    if first <= 0:
        return

    improvement = (first - last) / abs(first)
    tail = _tail_mean(train_loss)
    if tail is None:
        return

    if tail >= HIGH_LOSS_THRESHOLD * 0.75 and improvement < MIN_LOSS_IMPROVEMENT_FOR_PROGRESS:
        confidence = _clamp_confidence(
            0.62 + min(0.25, (MIN_LOSS_IMPROVEMENT_FOR_PROGRESS - improvement) * 0.9)
        )
        _add_fact(
            analysis,
            "SlowConvergence",
            "train_loss",
            confidence,
            (
                f"Training loss improved by only {improvement:.1%} from first "
                "to last epoch while remaining elevated."
            ),
            tail,
        )


def _add_quality_warnings(analysis: TrainingHistoryAnalysis) -> None:
    if len(analysis.rows) < TAIL_WINDOW:
        analysis.warnings.append(
            "Very few epochs were provided; metric-derived facts may be noisy."
        )

    if not analysis.fact_names:
        analysis.warnings.append(
            "No known diagnostic metric pattern was detected from the CSV."
        )


def _clamp_confidence(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 3)
