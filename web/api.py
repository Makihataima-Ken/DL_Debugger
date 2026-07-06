"""
api.py
======
Dependency-free HTTP API and static UI serving for the DL debugger.

This module is presentation-only: it validates request shape, delegates all
reasoning to DebuggingKnowledgeEngine or WhatIfSession, and serializes the
returned objects.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from data.problem_catalog import get_problem_categories, get_problem_options
from data.scenario_loader import BUILTIN_SCENARIOS
from engine.interactive import (
    DiagnosisDiff,
    WhatIfSession,
)
from engine.knowledge_engine import (
    DebuggingKnowledgeEngine,
    DiagnosisResult,
    FACT_CLASS_REGISTRY,
)


STATIC_DIR = Path(__file__).parent / "static"


class ApiError(Exception):
    """Expected API error with an HTTP status code."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def serialize_extraction(extraction: Any | None) -> dict[str, Any] | None:
    """Return a JSON-safe NLP extraction summary."""
    if extraction is None:
        return None

    return {
        "text": extraction.text,
        "intent": extraction.intent,
        "symptom_facts": list(extraction.symptom_facts),
        "cause_facts": list(getattr(extraction, "cause_facts", [])),
        "model_type_facts": list(extraction.model_type_facts),
        "context_facts": list(extraction.context_facts),
        "all_fact_names": list(extraction.all_fact_names),
        "lexical_confidence": extraction.lexical_confidence,
        "evidence": [
            {
                "fact_name": item.fact_name,
                "phrase": item.phrase,
                "confidence": item.confidence,
            }
            for item in extraction.evidence
        ],
        "negated": [
            {
                "fact_name": item.fact_name,
                "phrase": item.phrase,
                "confidence": item.confidence,
            }
            for item in extraction.negated
        ],
    }


def serialize_training_history(analysis: Any | None) -> dict[str, Any] | None:
    """Return a JSON-safe metric-history analysis summary."""
    if analysis is None:
        return None

    return {
        "source": analysis.source,
        "rows": len(analysis.rows),
        "columns": list(analysis.columns),
        "all_fact_names": list(analysis.all_fact_names),
        "root_confidence": dict(analysis.root_confidence),
        "summary": {
            metric: {
                key: round(float(value), 6)
                for key, value in values.items()
            }
            for metric, values in analysis.summary.items()
        },
        "warnings": list(analysis.warnings),
        "evidence": [
            {
                "fact_name": item.fact_name,
                "metric": item.metric,
                "confidence": item.confidence,
                "reason": item.reason,
                "value": item.value,
            }
            for item in analysis.evidence
        ],
    }


def serialize_result(result: DiagnosisResult) -> dict[str, Any]:
    """Return a JSON-safe DiagnosisResult dict."""
    return {
        "symptoms": list(result.symptoms),
        "causes": list(result.causes),
        "recommendations": list(result.recommendations),
        "explanations": [dict(explanation) for explanation in result.explanations],
        "confidence": dict(result.confidence),
        "conflicts": [dict(conflict) for conflict in result.conflicts],
        "extraction": serialize_extraction(result.extraction),
        "training_history": serialize_training_history(result.training_history),
    }


def serialize_diff(diff: DiagnosisDiff) -> dict[str, Any]:
    """Return a JSON-safe DiagnosisDiff dict."""
    return {
        "added_causes": list(diff.added_causes),
        "removed_causes": list(diff.removed_causes),
        "added_recommendations": list(diff.added_recommendations),
        "removed_recommendations": list(diff.removed_recommendations),
        "added_conflicts": list(diff.added_conflicts),
        "removed_conflicts": list(diff.removed_conflicts),
        "confidence_added": {
            fact: list(values)
            for fact, values in diff.confidence_added.items()
        },
        "confidence_removed": {
            fact: list(values)
            for fact, values in diff.confidence_removed.items()
        },
        "confidence_changed": {
            fact: list(values)
            for fact, values in diff.confidence_changed.items()
        },
        "has_changes": diff.has_changes,
    }


def _require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object.")
    return payload


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{key}' must be a non-empty string.")
    return value.strip()


def _require_fact_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{key}' must be a list of fact names.")
    return list(value)


def _base_facts(base: Any) -> list[str]:
    base = _require_object(base)
    facts = base.get("facts", base.get("symptoms", []))
    if not isinstance(facts, list) or not all(isinstance(item, str) for item in facts):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "'base' must contain a 'facts' or 'symptoms' list.",
        )
    return list(facts)


def _diagnose_symptoms(payload: dict[str, Any]) -> DiagnosisResult:
    session = WhatIfSession()
    session.add(*_require_fact_list(payload, "symptoms"))
    return session.diagnose()


def _diagnose_text(payload: dict[str, Any]) -> DiagnosisResult:
    return DebuggingKnowledgeEngine().run_text(_require_string(payload, "text"))


def _diagnose_history(payload: dict[str, Any]) -> DiagnosisResult:
    csv_text = payload.get("csv", payload.get("content"))
    if not isinstance(csv_text, str) or not csv_text.strip():
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "'csv' must contain non-empty training-history CSV content.",
        )
    filename = payload.get("filename", "uploaded CSV")
    if not isinstance(filename, str) or not filename.strip():
        filename = "uploaded CSV"
    return DebuggingKnowledgeEngine().run_training_history_text(
        csv_text,
        source=filename.strip(),
    )


def _diagnose_scenario(payload: dict[str, Any]) -> DiagnosisResult:
    session = WhatIfSession()
    session.load_scenario(_require_string(payload, "name"))
    return session.diagnose()


def _what_if(payload: dict[str, Any]) -> dict[str, Any]:
    session = WhatIfSession()
    session.add(*_base_facts(payload.get("base", {})))
    session.diagnose()
    action = _require_string(payload, "action").lower()
    facts = _require_fact_list(payload, "facts")
    diff = session.what_if(action, *facts)
    return {
        "result": serialize_result(session.last_result),
        "diff": serialize_diff(diff),
    }


def dispatch_api_request(
    method: str,
    path: str,
    payload: Any | None = None,
) -> tuple[int, dict[str, Any]]:
    """Dispatch one JSON API request and return (status, JSON body)."""
    method = method.upper()

    try:
        if method == "GET" and path == "/api/facts":
            return HTTPStatus.OK, {"facts": sorted(FACT_CLASS_REGISTRY)}

        if method == "GET" and path == "/api/scenarios":
            session = WhatIfSession()
            return HTTPStatus.OK, {
                "scenarios": session.available_scenarios(),
                "definitions": {
                    name: list(facts)
                    for name, facts in sorted(BUILTIN_SCENARIOS.items())
                },
            }

        if method == "GET" and path == "/api/problem_options":
            return HTTPStatus.OK, {
                "categories": get_problem_categories(),
                "problems": get_problem_options(),
            }

        if method == "POST" and path == "/api/diagnose":
            result = _diagnose_symptoms(_require_object(payload))
            return HTTPStatus.OK, serialize_result(result)

        if method == "POST" and path == "/api/diagnose_text":
            result = _diagnose_text(_require_object(payload))
            return HTTPStatus.OK, serialize_result(result)

        if method == "POST" and path == "/api/diagnose_history":
            result = _diagnose_history(_require_object(payload))
            return HTTPStatus.OK, serialize_result(result)

        if method == "POST" and path == "/api/scenario":
            result = _diagnose_scenario(_require_object(payload))
            return HTTPStatus.OK, serialize_result(result)

        if method == "POST" and path == "/api/whatif":
            return HTTPStatus.OK, _what_if(_require_object(payload))

        return HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"}
    except ApiError as exc:
        return exc.status, {"error": exc.message}
    except (KeyError, ValueError) as exc:
        return HTTPStatus.BAD_REQUEST, {"error": str(exc)}


class DLDebuggerRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving JSON API endpoints and the static browser UI."""

    server_version = "DLDebuggerHTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            status, body = dispatch_api_request("GET", path)
            self._send_json(status, body)
            return

        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
        except ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
            return

        status, body = dispatch_api_request("POST", path, payload)
        self._send_json(status, body)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep default access logs quiet during tests and CLI use."""

    def _read_json(self) -> Any:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        if content_length <= 0:
            return {}

        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"Malformed JSON: {exc.msg}",
            ) from exc

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            file_path = STATIC_DIR / "index.html"
        else:
            requested = path.lstrip("/")
            file_path = (STATIC_DIR / requested).resolve()
            if not str(file_path).startswith(str(STATIC_DIR.resolve())):
                self.send_error(HTTPStatus.NOT_FOUND)
                return

        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content = file_path.read_bytes()
        content_type = "text/html; charset=utf-8"
        if file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the stdlib HTTP server and block until interrupted."""
    server = ThreadingHTTPServer((host, port), DLDebuggerRequestHandler)
    print(f"Serving DL Debugger at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
