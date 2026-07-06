"""
test_api.py
===========
Tests for the presentation-only web API layer.
"""

from __future__ import annotations

import json
import sys
from http import HTTPStatus
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.problem_catalog import get_problem_options
from data.scenario_loader import get_builtin_symptoms
from engine.interactive import diff_diagnoses
from engine.knowledge_engine import DebuggingKnowledgeEngine
from web.api import (
    dispatch_api_request,
    serialize_diff,
    serialize_result,
)


def roundtrip(value):
    return json.loads(json.dumps(value))


def test_api_diagnose_matches_engine_run_scenario():
    symptoms = ["TrainingAccuracyHigh", "ValidationAccuracyLow"]
    expected = DebuggingKnowledgeEngine().run_scenario(symptoms)

    status, body = dispatch_api_request(
        "POST",
        "/api/diagnose",
        {"symptoms": symptoms},
    )

    assert status == HTTPStatus.OK
    assert body["symptoms"] == expected.symptoms
    assert body["causes"] == expected.causes
    assert body["recommendations"] == expected.recommendations
    assert body["conflicts"] == expected.conflicts


def test_api_scenario_matches_builtin_symptoms_and_engine():
    scenario_name = "oscillating_loss"
    symptoms = get_builtin_symptoms(scenario_name)
    expected = DebuggingKnowledgeEngine().run_scenario(symptoms)

    status, body = dispatch_api_request(
        "POST",
        "/api/scenario",
        {"name": scenario_name},
    )

    assert status == HTTPStatus.OK
    assert body["symptoms"] == expected.symptoms
    assert body["causes"] == expected.causes
    assert body["recommendations"] == expected.recommendations


def test_api_unknown_fact_returns_error_with_suggestion():
    status, body = dispatch_api_request(
        "POST",
        "/api/diagnose",
        {"symptoms": ["TrainingLossHgh"]},
    )

    assert status == HTTPStatus.BAD_REQUEST
    assert "Unknown fact name" in body["error"]
    assert "TrainingLossHigh" in body["error"]


def test_api_whatif_diff_matches_direct_diff():
    base = ["TrainingAccuracyHigh", "ValidationAccuracyLow"]
    added = ["SmallDataset"]
    engine = DebuggingKnowledgeEngine()
    before = engine.run_scenario(base)
    after = engine.run_scenario(base + added)
    expected_diff = serialize_diff(diff_diagnoses(before, after))

    status, body = dispatch_api_request(
        "POST",
        "/api/whatif",
        {
            "base": {"facts": base},
            "action": "add",
            "facts": added,
        },
    )

    assert status == HTTPStatus.OK
    assert body["result"]["recommendations"] == after.recommendations
    assert body["diff"] == expected_diff


def test_api_facts_and_scenarios_are_json_roundtrippable():
    facts_status, facts_body = dispatch_api_request("GET", "/api/facts")
    scenarios_status, scenarios_body = dispatch_api_request("GET", "/api/scenarios")

    assert facts_status == HTTPStatus.OK
    assert scenarios_status == HTTPStatus.OK
    assert "TrainingLossHigh" in roundtrip(facts_body)["facts"]
    assert "overfitting" in roundtrip(scenarios_body)["scenarios"]


def test_api_problem_options_are_json_roundtrippable():
    status, body = dispatch_api_request("GET", "/api/problem_options")
    payload = roundtrip(body)
    problems = payload["problems"]

    assert status == HTTPStatus.OK
    assert payload["categories"][0] == "Quick Examples"
    assert problems == get_problem_options()
    assert len({problem["id"] for problem in problems}) == len(problems)
    assert any(
        problem["text"] == "multi gpu training throughput is low and the gpus are idle"
        for problem in problems
    )


def test_serialize_diagnosis_result_is_json_roundtrippable():
    result = DebuggingKnowledgeEngine().run_text(
        "training loss high and loss oscillates but no slow convergence"
    )

    payload = roundtrip(serialize_result(result))

    assert payload["causes"] == result.causes
    assert payload["extraction"]["all_fact_names"] == result.extraction.all_fact_names
    assert payload["extraction"]["negated"]


def test_api_diagnose_text_includes_nlp_extraction_summary():
    status, body = dispatch_api_request(
        "POST",
        "/api/diagnose_text",
        {"text": "training loss high and loss oscillates"},
    )

    assert status == HTTPStatus.OK
    assert "LearningRateTooHigh" in body["causes"]
    assert body["extraction"]["symptom_facts"] == [
        "TrainingLossHigh",
        "OscillatingLoss",
    ]
    assert body["extraction"]["evidence"]


def test_api_diagnose_text_exposes_user_stated_cause_facts():
    status, body = dispatch_api_request(
        "POST",
        "/api/diagnose_text",
        {"text": "the learning rate is too high because loss oscillates"},
    )

    assert status == HTTPStatus.OK
    assert "LearningRateTooHigh" in body["causes"]
    assert body["extraction"]["cause_facts"] == ["LearningRateTooHigh"]
