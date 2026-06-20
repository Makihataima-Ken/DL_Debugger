"""
helpers.py
==========
Pure utility functions for the Deep Learning Debugging Expert System.

No diagnosis logic lives here – only hashing, formatting, normalisation, and
validation helpers that support the rule engine.
"""

from __future__ import annotations

import hashlib
import textwrap
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def stable_hash(value: str) -> str:
    """Return a short, stable SHA-256 hex digest for *value*.

    Parameters
    ----------
    value:
        The string to hash.

    Returns
    -------
    str
        First 8 hex characters of the SHA-256 digest.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_fact_name(name: str) -> str:
    """Convert a fact class name to a readable label.

    ``TrainingLossHigh`` → ``Training Loss High``

    Parameters
    ----------
    name:
        CamelCase class name string.

    Returns
    -------
    str
        Space-separated words.
    """
    import re
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_scenario(scenario: dict[str, Any]) -> bool:
    """Return *True* when *scenario* contains the required keys.

    Parameters
    ----------
    scenario:
        Parsed JSON object representing a debug scenario.

    Returns
    -------
    bool
        Whether the scenario has at minimum ``name`` and ``symptoms``.
    """
    return (
        isinstance(scenario, dict)
        and "name" in scenario
        and "symptoms" in scenario
        and isinstance(scenario["symptoms"], list)
    )


def validate_symptom_names(symptoms: list[str], known: set[str]) -> list[str]:
    """Return the subset of *symptoms* that are recognised fact class names.

    Parameters
    ----------
    symptoms:
        List of symptom names from a scenario file.
    known:
        Set of valid Fact class names.

    Returns
    -------
    list[str]
        Only the names present in *known*.
    """
    return [s for s in symptoms if s in known]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_section(title: str, items: Sequence[str], width: int = 60) -> str:
    """Render a titled section with a list of items for CLI output.

    Parameters
    ----------
    title:
        Section heading.
    items:
        Lines to display under the heading.
    width:
        Total character width of the separator line.

    Returns
    -------
    str
        Formatted multi-line string.
    """
    sep = "─" * width
    lines = [f"\n{sep}", f"  {title}", sep]
    for item in items:
        lines.append(f"  • {item}")
    if not items:
        lines.append("  (none)")
    return "\n".join(lines)


def format_explanation(
    rule_id: str,
    triggered_by: str,
    derived: str,
    explanation: str,
) -> str:
    """Render a single XAI explanation block.

    Parameters
    ----------
    rule_id:
        Identifier of the firing rule.
    triggered_by:
        Comma-separated triggering fact names.
    derived:
        The fact that was derived.
    explanation:
        Human-readable rationale.

    Returns
    -------
    str
        Formatted block string.
    """
    wrapped = textwrap.fill(explanation, width=56, initial_indent="    ", subsequent_indent="    ")
    triggers = ", ".join(
        normalise_fact_name(t.strip()) for t in triggered_by.split(",")
    )
    return (
        f"\n  Rule     : {rule_id}\n"
        f"  Triggered: {triggers}\n"
        f"  Derived  : {normalise_fact_name(derived)}\n"
        f"  Rationale:\n{wrapped}\n"
    )


def banner(text: str, width: int = 60) -> str:
    """Return a simple banner string.

    Parameters
    ----------
    text:
        Banner text.
    width:
        Total width.

    Returns
    -------
    str
        Banner string.
    """
    pad = max(0, (width - len(text) - 2) // 2)
    return f"\n{'═' * width}\n{' ' * pad} {text}\n{'═' * width}"
