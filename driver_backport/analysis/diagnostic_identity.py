#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from driver_backport.core.common import normalize_relpath, stable_id


def _diagnostic_file(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    return str(path) if path.is_absolute() else normalize_relpath(text)


_RAW_LOCATION_PREFIX = re.compile(
    r"^.*?:\d+(?::\d+)?:\s*"
    r"(?:(?:fatal\s+)?error|warning|note):\s*",
    re.DOTALL,
)


def _diagnostic_message(issue: dict[str, Any]) -> str:
    message = str(issue.get("message") or "").strip()
    if message:
        return message
    raw = str(issue.get("raw") or "").strip()
    return _RAW_LOCATION_PREFIX.sub("", raw, count=1).strip()


def diagnostic_signature(issue: dict[str, Any]) -> str:
    """Location-independent identity for cross-round diagnostic comparison."""

    return stable_id(
        "diag-signature",
        _diagnostic_file(issue.get("file")),
        str(issue.get("kind") or ""),
        str(issue.get("warning_flag") or ""),
        str(issue.get("symbol") or ""),
        str(issue.get("module") or ""),
        str(issue.get("object") or ""),
        _diagnostic_message(issue),
    )


def diagnostic_occurrence_signature(issue: dict[str, Any]) -> str:
    """Exact source occurrence identity within one validation snapshot."""

    return stable_id(
        "diag-occurrence",
        diagnostic_signature(issue),
        _diagnostic_file(issue.get("file")),
        str(issue.get("line") or ""),
        str(issue.get("column") or ""),
    )


def diagnostic_event_record(issue: dict[str, Any]) -> dict[str, Any]:
    """Project one compiler issue onto the facts allowed in the B/C graph flow."""

    relative_path = (
        issue.get("file")
        or issue.get("source_file")
        or issue.get("path")
        or issue.get("relative_path")
    )
    raw = str(issue.get("raw") or issue.get("message") or "")
    signature_input = {**issue, "file": relative_path}
    record = {
        "diagnostic_event_id": issue.get("diagnostic_event_id"),
        "diagnostic_signature": str(
            issue.get("diagnostic_signature") or diagnostic_signature(signature_input)
        ),
        "diagnostic_occurrence_count": int(
            issue.get("diagnostic_occurrence_count") or 1
        ),
        "diagnostic_occurrence_event_ids": list(
            issue.get("diagnostic_occurrence_event_ids") or []
        ),
        "diagnostic_occurrences": list(
            issue.get("diagnostic_occurrences") or []
        ),
        "trigger_files": list(issue.get("trigger_files") or []),
        "kind": issue.get("kind"),
        "severity": issue.get("severity"),
        "warning_flag": issue.get("warning_flag"),
        "warning_as_error": bool(issue.get("warning_as_error")),
        "migration_contract_error": bool(
            issue.get("migration_contract_error")
        ),
        "migration_contract_warning": bool(
            issue.get("migration_contract_warning")
        ),
        "repair_relevance": issue.get("repair_relevance"),
        "_phase_index": issue.get("_phase_index"),
        "_issue_index": issue.get("_issue_index"),
        "phase_name": issue.get("phase_name"),
        "file": relative_path,
        "line": issue.get("line"),
        "column": issue.get("column"),
        "message": str(issue.get("message") or raw),
        "raw": raw,
        "diagnostic_route": issue.get("diagnostic_route"),
    }
    related_notes: list[dict[str, Any]] = []
    for note in issue.get("related_notes") or []:
        if not isinstance(note, dict):
            continue
        projected = {
            key: note.get(key)
            for key in (
                "kind",
                "file",
                "line",
                "column",
                "symbol",
                "message",
                "raw",
                "type",
            )
            if note.get(key) is not None
        }
        if projected:
            related_notes.append(projected)
    if related_notes:
        record["related_notes"] = related_notes
    compiler_suggestions: list[dict[str, str]] = []
    for suggestion in issue.get("compiler_suggestions") or []:
        if not isinstance(suggestion, dict):
            continue
        symbol = str(suggestion.get("symbol") or "").strip()
        relation = str(suggestion.get("relation") or "").strip()
        authority = str(suggestion.get("authority") or "").strip()
        if not symbol or authority != "compiler":
            continue
        compiler_suggestions.append(
            {
                "symbol": symbol,
                "relation": relation or "suggested_contract",
                "authority": authority,
            }
        )
    if compiler_suggestions:
        record["compiler_suggestions"] = compiler_suggestions
    if (
        str(issue.get("kind") or "") == "compile_missing_declaration"
        and str(issue.get("symbol") or "").strip()
    ):
        record["symbol"] = str(issue["symbol"]).strip()
        record["source_symbol_evidence"] = {
            "authority": "compiler_missing_declaration",
            "relation": "requires_visible_declaration",
        }
    elif (
        str(issue.get("kind") or "") == "compile_missing_type"
        and str(issue.get("symbol") or "").strip()
    ):
        record["symbol"] = str(issue["symbol"]).strip()
        record["source_symbol_evidence"] = {
            "authority": "compiler_missing_type",
            "relation": "requires_complete_type_definition",
        }
    elif (
        str(issue.get("kind") or "") == "compile_missing_struct_member"
        and str(issue.get("symbol") or "").strip()
        and isinstance(issue.get("member_container"), dict)
    ):
        record["symbol"] = str(issue["symbol"]).strip()
        container = issue["member_container"]
        record["member_container"] = {
            "container_kind": str(container.get("container_kind") or ""),
            "container_name": str(container.get("container_name") or ""),
        }
    graph_contract_anchor = issue.get("graph_contract_anchor")
    if isinstance(graph_contract_anchor, dict):
        anchor = {
            key: graph_contract_anchor.get(key)
            for key in (
                "schema",
                "authority",
                "tree_role",
                "file",
                "contract_key",
                "entity_kind",
                "symbol",
                "extent",
            )
            if graph_contract_anchor.get(key) not in (None, "", [], {})
        }
        symbol = str(issue.get("symbol") or "").strip()
        if anchor and symbol:
            record["symbol"] = symbol
            record["graph_contract_anchor"] = anchor
            record["graph_root_policy"] = str(
                issue.get("graph_root_policy")
                or "typed_contract"
            )
    # A locationless linker diagnostic carries its program identity through
    # the reported module/object. Source diagnostics must instead resolve the
    # entity from their code location; validator-derived symbol hints are not
    # graph evidence.
    for key in ("module", "object"):
        value = issue.get(key)
        if value is not None and str(value).strip():
            record[key] = value
    if str(issue.get("kind") or "") == "modpost_namespace_import_missing":
        namespace = str(issue.get("namespace") or "").strip()
        if namespace:
            record["namespace"] = namespace
    if (
        not relative_path
        and any(record.get(key) for key in ("module", "object"))
        and issue.get("symbol") is not None
        and str(issue.get("symbol")).strip()
    ):
        record["symbol"] = issue.get("symbol")
    return record
