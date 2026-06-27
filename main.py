"""
main.py
=======
CLI entry point for the Deep Learning Debugging Expert System.

Usage
-----
::

    python main.py --scenario overfitting
    python main.py --scenario gradient_explosion
    python main.py --scenario distribution_shift
    python main.py --file data/scenarios/training_examples.json
    python main.py --list
    python main.py --symptoms TrainingLossHigh,OscillatingLoss
    python main.py --text "my training loss keeps oscillating and never converges"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine, FACT_CLASS_REGISTRY
from data.scenario_loader import get_builtin_symptoms, BUILTIN_SCENARIOS
from utils.helpers import banner, format_section, format_explanation, normalise_fact_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_result(
    scenario_label: str,
    symptoms: list[str],
    causes: list[str],
    recommendations: list[str],
    explanations: list[dict[str, str]],
    confidence: dict[str, float] | None = None,
) -> None:
    """Pretty-print the full diagnosis to stdout."""
    print(banner(f"DL Debugging Expert System -- {scenario_label}"))

    print(format_section(
        "INPUT SYMPTOMS",
        [normalise_fact_name(s) for s in symptoms],
    ))

    conf = confidence or {}
    cause_lines = [
        f"{normalise_fact_name(c)}"
        + (f"  (confidence {conf[c]:.2f})" if c in conf else "")
        for c in causes
    ]
    print(format_section(
        "DERIVED CAUSES",
        cause_lines,
    ))

    print(format_section(
        "RECOMMENDATIONS",
        [normalise_fact_name(r) for r in recommendations],
    ))

    print(f"\n{'─' * 60}")
    print("  EXPLANATION CHAIN")
    print(f"{'─' * 60}")

    if explanations:
        for exp in explanations:
            print(
                format_explanation(
                    rule_id=exp["rule_id"],
                    triggered_by=exp["triggered_by"],
                    derived=exp["derived"],
                    explanation=exp["explanation"],
                )
            )
    else:
        print("  (no explanations generated)")

    print(f"{'═' * 60}\n")


def _execute_scenario(symptom_names: list[str], label: str) -> None:
    """Run inference on a list of symptoms and print the result."""
    engine = DebuggingKnowledgeEngine()
    result = engine.run_scenario(symptom_names)

    _print_result(
        scenario_label=label,
        symptoms=result.symptoms,
        causes=result.causes,
        recommendations=result.recommendations,
        explanations=result.explanations,
        confidence=result.confidence,
    )


def _run_scenario(name: str) -> None:
    """Look up and execute a named built-in scenario."""
    try:
        symptoms = get_builtin_symptoms(name)
    except KeyError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    _execute_scenario(symptoms, label=name)


def _run_custom(symptom_str: str) -> None:
    """Parse a comma-separated symptom list and execute inference."""
    raw_names = [s.strip() for s in symptom_str.split(",") if s.strip()]
    unknown = [n for n in raw_names if n not in FACT_CLASS_REGISTRY]
    if unknown:
        print(
            f"[ERROR] Unknown symptom(s): {', '.join(unknown)}\n"
            f"Valid fact names: {', '.join(sorted(FACT_CLASS_REGISTRY))}",
            file=sys.stderr,
        )
        sys.exit(1)

    _execute_scenario(raw_names, label="custom")


def _run_text(text: str) -> None:
    """Natural-language entry point: text -> facts -> diagnosis."""
    engine = DebuggingKnowledgeEngine()
    result = engine.run_text(text)
    extraction = result.extraction

    if extraction is not None and not extraction.all_fact_names:
        print(
            "[INFO] No known symptoms were recognised in the text.\n"
            "       Try describing loss, accuracy, gradients, or the model type.",
            file=sys.stderr,
        )

    _print_result(
        scenario_label="natural-language",
        symptoms=result.symptoms,
        causes=result.causes,
        recommendations=result.recommendations,
        explanations=result.explanations,
        confidence=result.confidence,
    )


def _list_scenarios() -> None:
    """Print all available built-in scenario names."""
    print(banner("Available Built-in Scenarios"))
    for name, symptoms in sorted(BUILTIN_SCENARIOS.items()):
        sym_str = ", ".join(normalise_fact_name(s) for s in symptoms)
        print(f"  {name:<25} → {sym_str}")
    print()


def _prompt_choose_case(cases: list[dict]) -> dict:
    """Display available cases and prompt the user to choose one."""
    print("\nAvailable cases in this scenario file:")
    for idx, case in enumerate(cases, start=1):
        description = case.get("description", "No description")
        print(f"  {idx}. {case['name']} - {description}")

    while True:
        try:
            choice = input("\nEnter the case number to run: ").strip()
            choice = int(choice)
            if 1 <= choice <= len(cases):
                return cases[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(cases)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def _run_file(path_str: str) -> None:
    """Load scenarios from a JSON file and let the user pick a case."""
    path = Path(path_str)
    if not path.exists():
        print(f"[ERROR] Scenario file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with path.open(encoding="utf-8") as fh:
            cases = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(cases, list) or not cases:
        print("[ERROR] Scenario file must contain a non-empty list of cases.", file=sys.stderr)
        sys.exit(1)

    selected_case = _prompt_choose_case(cases)
    symptoms = selected_case.get("symptoms", [])

    if not symptoms:
        print("[ERROR] Selected case has no symptoms.", file=sys.stderr)
        sys.exit(1)

    _execute_scenario(symptoms, label=selected_case["name"])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Deep Learning Debugging Expert System -- diagnose training issues using expert-system inference.",
        epilog="For more information see README.md."
    )

    parser.add_argument(
        "--scenario",
        metavar="NAME",
        help="Run a named built-in scenario (e.g. overfitting, gradient_explosion).",
    )
    parser.add_argument(
        "--symptoms",
        metavar="LIST",
        help="Comma-separated list of symptom fact names (e.g. TrainingLossHigh,OscillatingLoss).",
    )
    parser.add_argument(
        "--file",
        dest="scenario_file",
        metavar="PATH",
        help="Path to a JSON scenario file with cases to choose from.",
    )
    parser.add_argument(
        "--text",
        metavar="TEXT",
        help="Describe the problem in natural language (e.g. 'my training loss oscillates').",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_scenarios",
        help="List all available built-in scenarios and exit.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate handler."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        _list_scenarios()
        return

    if args.text:
        _run_text(args.text)
        return

    if args.scenario:
        _run_scenario(args.scenario)
        return

    if args.symptoms:
        _run_custom(args.symptoms)
        return

    if args.scenario_file:
        _run_file(args.scenario_file)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
