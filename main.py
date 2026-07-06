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
    python main.py --history-csv training_history.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

# Ensure project root is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from engine.knowledge_engine import DebuggingKnowledgeEngine, FACT_CLASS_REGISTRY
from engine.interactive import DiagnosisDiff, WhatIfSession
from data.scenario_loader import get_builtin_symptoms, BUILTIN_SCENARIOS
from utils.helpers import banner, format_section, format_explanation, normalise_fact_name
from web.api import run_server


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


def _format_fact_names(fact_names: list[str]) -> list[str]:
    """Return readable labels for fact class names."""
    return [normalise_fact_name(name) for name in fact_names]


def _print_repl_result(result, output_stream: TextIO) -> None:
    """Print a compact interactive diagnosis result."""
    print(banner("DL Debugging Expert System -- interactive"), file=output_stream)
    print(
        format_section("INPUT SYMPTOMS", _format_fact_names(result.symptoms)),
        file=output_stream,
    )

    cause_lines = [
        f"{normalise_fact_name(cause)}"
        + (
            f"  (confidence {result.confidence[cause]:.2f})"
            if cause in result.confidence
            else ""
        )
        for cause in result.causes
    ]
    print(format_section("DERIVED CAUSES", cause_lines), file=output_stream)
    print(
        format_section(
            "RECOMMENDATIONS",
            _format_fact_names(result.recommendations),
        ),
        file=output_stream,
    )

    explanation_lines = [
        (
            f"{exp['rule_id']}: {normalise_fact_name(exp['derived'])} "
            f"from {', '.join(_format_fact_names(exp['triggered_by'].split(',')))}"
        )
        for exp in result.explanations[:5]
    ]
    print(format_section("TOP EXPLANATIONS", explanation_lines), file=output_stream)

    if len(result.explanations) > 5:
        print(
            f"  ({len(result.explanations) - 5} more explanation(s) omitted)",
            file=output_stream,
        )

    conflict_lines = [
        (
            f"{conflict['rule_id']}: kept "
            f"{normalise_fact_name(conflict['winner'])}; suppressed "
            f"{', '.join(_format_fact_names(conflict['losers'].split(',')))}"
        )
        for conflict in result.conflicts
    ]
    print(format_section("CONFLICTS", conflict_lines), file=output_stream)

    confidence_lines = [
        f"{normalise_fact_name(name)}: {value:.3f}"
        for name, value in sorted(result.confidence.items())
    ]
    print(format_section("CONFIDENCE", confidence_lines), file=output_stream)


def _print_extraction(extraction, output_stream: TextIO) -> None:
    """Print NLP extraction provenance for an interactive text command."""
    if extraction is None:
        print("[INFO] No extraction provenance returned.", file=output_stream)
        return

    facts = extraction.all_fact_names
    print(format_section("EXTRACTED FACTS", _format_fact_names(facts)), file=output_stream)

    if extraction.evidence:
        evidence_lines = [
            (
                f"{normalise_fact_name(evidence.fact_name)} from "
                f"'{evidence.phrase}' (confidence {evidence.confidence:.3f})"
            )
            for evidence in extraction.evidence
        ]
        print(format_section("EVIDENCE", evidence_lines), file=output_stream)

    if extraction.negated:
        negated_lines = [
            f"{normalise_fact_name(entity.fact_name)} from '{entity.phrase}'"
            for entity in extraction.negated
        ]
        print(format_section("NEGATED TERMS", negated_lines), file=output_stream)


def _print_training_history_analysis(analysis, output_stream: TextIO) -> None:
    """Print CSV metric extraction provenance."""
    if analysis is None:
        print("[INFO] No metric analysis provenance returned.", file=output_stream)
        return

    fact_lines = [
        (
            f"{normalise_fact_name(evidence.fact_name)} from "
            f"{evidence.metric} (confidence {evidence.confidence:.3f})"
        )
        for evidence in analysis.evidence
    ]
    print(format_section("METRIC-DERIVED FACTS", fact_lines), file=output_stream)

    if analysis.warnings:
        print(format_section("CSV WARNINGS", analysis.warnings), file=output_stream)


def _print_diff(diff: DiagnosisDiff, output_stream: TextIO) -> None:
    """Print a structured what-if diff."""
    if not diff.has_changes:
        print("[INFO] No visible diagnosis changes.", file=output_stream)
        return

    print(format_section("ADDED CAUSES", _format_fact_names(diff.added_causes)), file=output_stream)
    print(format_section("REMOVED CAUSES", _format_fact_names(diff.removed_causes)), file=output_stream)
    print(
        format_section(
            "ADDED RECOMMENDATIONS",
            _format_fact_names(diff.added_recommendations),
        ),
        file=output_stream,
    )
    print(
        format_section(
            "REMOVED RECOMMENDATIONS",
            _format_fact_names(diff.removed_recommendations),
        ),
        file=output_stream,
    )
    print(format_section("ADDED CONFLICTS", diff.added_conflicts), file=output_stream)
    print(format_section("REMOVED CONFLICTS", diff.removed_conflicts), file=output_stream)

    confidence_lines: list[str] = []
    for fact, (_old, new) in diff.confidence_added.items():
        confidence_lines.append(f"+ {normalise_fact_name(fact)}: {new:.3f}")
    for fact, (old, _new) in diff.confidence_removed.items():
        confidence_lines.append(f"- {normalise_fact_name(fact)}: {old:.3f}")
    for fact, (old, new) in diff.confidence_changed.items():
        confidence_lines.append(f"~ {normalise_fact_name(fact)}: {old:.3f} -> {new:.3f}")
    print(format_section("CONFIDENCE CHANGES", confidence_lines), file=output_stream)


def _split_fact_args(raw: str) -> list[str]:
    """Parse command fact arguments, accepting spaces or commas."""
    return [
        item.strip()
        for item in raw.replace(",", " ").split()
        if item.strip()
    ]


def _read_repl_line(
    input_stream: TextIO | None,
    output_stream: TextIO,
) -> str:
    """Read one REPL line, supporting stdin or injected streams."""
    if input_stream is None:
        return input("what-if> ")

    print("what-if> ", end="", file=output_stream)
    line = input_stream.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\n")


def _print_repl_help(output_stream: TextIO) -> None:
    """Print interactive command help."""
    commands = [
        "help - show this command list",
        "facts / list - show the current fact set",
        "add <FactName...> - add facts to the current set",
        "remove <FactName...> - remove facts from the current set",
        "text <description> - replace facts from NLP extraction",
        "history <csv-path> - replace facts from training-history CSV",
        "scenario <name> - load a built-in scenario",
        "diagnose / run - run the current fact set",
        "whatif add <FactName...> - apply, rerun, and diff vs baseline",
        "whatif remove <FactName...> - remove, rerun, and diff vs baseline",
        "reset / clear - empty the session",
        "quit / exit - leave interactive mode",
    ]
    print(format_section("COMMANDS", commands), file=output_stream)


def _run_interactive(
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """Run the presentation-only what-if REPL."""
    output_stream = output_stream or sys.stdout
    session = WhatIfSession()

    print(banner("DL Debugging Expert System -- Interactive What-If"), file=output_stream)
    print("Type 'help' for commands. Type 'quit' to exit.", file=output_stream)

    while True:
        try:
            line = _read_repl_line(input_stream, output_stream)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.", file=output_stream)
            return

        stripped = line.strip()
        if not stripped:
            continue

        command, _sep, rest = stripped.partition(" ")
        command = command.lower()
        rest = rest.strip()

        try:
            if command in {"quit", "exit"}:
                print("Goodbye.", file=output_stream)
                return

            if command == "help":
                _print_repl_help(output_stream)
                continue

            if command in {"facts", "list"}:
                print(
                    format_section(
                        "CURRENT FACTS",
                        _format_fact_names(session.current_facts()),
                    ),
                    file=output_stream,
                )
                continue

            if command in {"add", "remove"}:
                fact_names = _split_fact_args(rest)
                if not fact_names:
                    print(f"[ERROR] Usage: {command} <FactName...>", file=output_stream)
                    continue
                if command == "add":
                    session.add(*fact_names)
                else:
                    session.remove(*fact_names)
                print(
                    format_section(
                        "CURRENT FACTS",
                        _format_fact_names(session.current_facts()),
                    ),
                    file=output_stream,
                )
                continue

            if command == "text":
                if not rest:
                    print("[ERROR] Usage: text <free description>", file=output_stream)
                    continue
                extraction = session.set_from_text(rest)
                _print_extraction(extraction, output_stream)
                continue

            if command == "history":
                if not rest:
                    print("[ERROR] Usage: history <csv-path>", file=output_stream)
                    continue
                analysis = session.set_from_training_history_csv(rest)
                _print_training_history_analysis(analysis, output_stream)
                continue

            if command == "scenario":
                if not rest:
                    print("[ERROR] Usage: scenario <name>", file=output_stream)
                    continue
                session.load_scenario(rest)
                print(
                    format_section(
                        "CURRENT FACTS",
                        _format_fact_names(session.current_facts()),
                    ),
                    file=output_stream,
                )
                continue

            if command in {"diagnose", "run"}:
                if not session.current_facts():
                    print(
                        "[INFO] No facts set. Use add, text, or scenario first.",
                        file=output_stream,
                    )
                    continue
                _print_repl_result(session.diagnose(), output_stream)
                continue

            if command == "whatif":
                subcommand, _sub_sep, sub_rest = rest.partition(" ")
                subcommand = subcommand.lower()
                fact_names = _split_fact_args(sub_rest)
                if subcommand not in {"add", "remove"} or not fact_names:
                    print(
                        "[ERROR] Usage: whatif add <FactName...> or whatif remove <FactName...>",
                        file=output_stream,
                    )
                    continue
                diff = session.what_if(subcommand, *fact_names)
                _print_diff(diff, output_stream)
                print(
                    format_section(
                        "CURRENT FACTS",
                        _format_fact_names(session.current_facts()),
                    ),
                    file=output_stream,
                )
                continue

            if command in {"reset", "clear"}:
                session.clear()
                print("[INFO] Session cleared.", file=output_stream)
                continue

            print(
                f"[ERROR] Unknown command: {command}. Type 'help' for commands.",
                file=output_stream,
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=output_stream)


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


def _run_history_csv(path_str: str) -> None:
    """Metric-history entry point: CSV -> facts -> diagnosis."""
    engine = DebuggingKnowledgeEngine()
    try:
        result = engine.run_training_history_csv(path_str)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    _print_result(
        scenario_label=f"training-history: {Path(path_str).name}",
        symptoms=result.symptoms,
        causes=result.causes,
        recommendations=result.recommendations,
        explanations=result.explanations,
        confidence=result.confidence,
    )
    _print_training_history_analysis(result.training_history, sys.stdout)


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
        "--history-csv",
        metavar="PATH",
        help="Diagnose a CSV training history with epoch/loss/accuracy metrics.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_scenarios",
        help="List all available built-in scenarios and exit.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch an interactive what-if session.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the browser UI and JSON API server.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for --serve (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for --serve (default: 8000).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _configure_console_encoding() -> None:
    """Prefer UTF-8 for CLI output on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate handler."""
    _configure_console_encoding()
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        _list_scenarios()
        return

    if args.interactive:
        _run_interactive()
        return

    if args.text:
        _run_text(args.text)
        return

    if args.history_csv:
        _run_history_csv(args.history_csv)
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

    if args.serve:
        run_server(host=args.host, port=args.port)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
