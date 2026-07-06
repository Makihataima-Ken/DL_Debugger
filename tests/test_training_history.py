"""
test_training_history.py
========================
Tests for CSV training-history input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.interactive import WhatIfSession
from engine.knowledge_engine import DebuggingKnowledgeEngine
from engine.metrics.training_history_analyzer import (
    analyze_training_history_csv,
    analyze_training_history_text,
)


ROOT = Path(__file__).parent.parent
SAMPLE_CSV = ROOT / "training_history.csv"


def test_training_history_analyzer_extracts_metric_facts_from_sample():
    analysis = analyze_training_history_csv(SAMPLE_CSV)

    assert analysis.all_fact_names == [
        "TrainingAccuracyLow",
        "ValidationAccuracyLow",
        "TrainingLossHigh",
        "ValidationLossHigh",
        "OscillatingLoss",
    ]
    assert analysis.root_confidence["OscillatingLoss"] > 0.8
    assert analysis.summary["epochs"]["count"] == 7.0
    assert not analysis.warnings


def test_training_history_engine_uses_existing_rule_diagnosis():
    result = DebuggingKnowledgeEngine().run_training_history_csv(SAMPLE_CSV)

    assert result.training_history is not None
    assert "LearningRateTooHigh" in result.causes
    assert "UnderfittingObserved" in result.causes
    assert "ModelTooSimple" in result.causes
    assert "ReduceLearningRate" in result.recommendations
    assert "IncreaseModelComplexity" in result.recommendations
    assert "TRAIN_001" in [item["rule_id"] for item in result.explanations]


def test_training_history_missing_required_columns_raises():
    csv_text = "epoch,train_loss,val_loss\n1,1.2,1.3\n"

    with pytest.raises(ValueError, match="missing required column"):
        analyze_training_history_text(csv_text)


def test_interactive_session_can_load_training_history_csv():
    session = WhatIfSession()

    analysis = session.set_from_training_history_csv(str(SAMPLE_CSV))
    result = session.diagnose()

    assert analysis is not None
    assert "OscillatingLoss" in session.current_facts()
    assert "LearningRateTooHigh" in result.causes
    assert result.training_history is analysis
