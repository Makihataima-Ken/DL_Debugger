"""
test_interactive.py
===================
Pure tests for the interactive what-if session layer.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.interactive import DiagnosisDiff, WhatIfSession, diff_diagnoses
from engine.knowledge_engine import DebuggingKnowledgeEngine, DiagnosisResult
from main import _run_interactive


def run(facts: list[str]) -> DiagnosisResult:
    return DebuggingKnowledgeEngine().run_scenario(facts)


def test_session_add_remove_and_validation_errors():
    session = WhatIfSession()

    session.add("TrainingLossHigh", "OscillatingLoss")
    session.add("TrainingLossHigh")
    assert session.current_facts() == ["TrainingLossHigh", "OscillatingLoss"]

    session.remove("SlowConvergence")
    assert session.current_facts() == ["TrainingLossHigh", "OscillatingLoss"]

    session.remove("TrainingLossHigh")
    assert session.current_facts() == ["OscillatingLoss"]

    with pytest.raises(ValueError, match="Unknown fact name"):
        session.add("TrainingLossHgh")


def test_session_load_scenario():
    session = WhatIfSession()

    facts = session.load_scenario("overfitting")

    assert facts == ["TrainingAccuracyHigh", "ValidationAccuracyLow"]
    assert session.current_facts() == facts

    with pytest.raises(ValueError, match="Unknown scenario"):
        session.load_scenario("does_not_exist")


def test_session_set_from_text_and_diagnose():
    session = WhatIfSession()

    extraction = session.set_from_text(
        "training loss high and loss oscillates but no slow convergence"
    )
    result = session.diagnose()

    assert extraction is not None
    assert set(session.current_facts()) == {"TrainingLossHigh", "OscillatingLoss"}
    assert "SlowConvergence" not in session.current_facts()
    assert "TrainingLossHigh" in session.root_confidence
    assert "LearningRateTooHigh" in result.causes
    assert session.last_result is result


def test_diff_diagnoses_reports_added_recommendations_and_confidence():
    before = run(["TrainingAccuracyHigh", "ValidationAccuracyLow"])
    after = run(["TrainingAccuracyHigh", "ValidationAccuracyLow", "SmallDataset"])

    diff = diff_diagnoses(before, after)

    assert isinstance(diff, DiagnosisDiff)
    assert diff.added_recommendations == ["ApplyDataAugmentation"]
    assert diff.removed_recommendations == []
    assert diff.confidence_added["ApplyDataAugmentation"] == (None, 0.64)
    assert diff.has_changes


def test_diff_diagnoses_reports_added_conflicts_by_rule_id():
    before = run(["TrainingLossHigh", "OscillatingLoss"])
    after = run(["TrainingLossHigh", "OscillatingLoss", "SlowConvergence"])

    diff = diff_diagnoses(before, after)

    assert diff.added_conflicts == ["CONFLICT_002"]
    assert diff.removed_conflicts == []


def test_what_if_flow_matches_recomputed_diff():
    session = WhatIfSession()
    session.load_scenario("overfitting")
    baseline = session.diagnose()

    diff = session.what_if("add", "SmallDataset")
    recomputed = diff_diagnoses(
        baseline,
        run(["TrainingAccuracyHigh", "ValidationAccuracyLow", "SmallDataset"]),
    )

    assert diff == recomputed
    assert session.current_facts() == [
        "TrainingAccuracyHigh",
        "ValidationAccuracyLow",
        "SmallDataset",
    ]
    assert session.last_result is not baseline


def test_repl_loop_can_be_driven_with_string_io():
    input_stream = io.StringIO(
        "scenario overfitting\n"
        "diagnose\n"
        "whatif add SmallDataset\n"
        "quit\n"
    )
    output_stream = io.StringIO()

    _run_interactive(input_stream=input_stream, output_stream=output_stream)

    output = output_stream.getvalue()
    assert "CURRENT FACTS" in output
    assert "ADDED RECOMMENDATIONS" in output
    assert "Apply Data Augmentation" in output
    assert "Goodbye." in output
