#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from driver_backport.analysis.diagnostic_identity import (
    diagnostic_occurrence_signature,
    diagnostic_signature,
)
from driver_backport.core.common import normalize_relpath, stable_id


BUILD_METADATA_NAMES = {"Kconfig", "Makefile", "Kbuild"}
BUILD_PREP_ISSUE_KINDS = {
    "kconfig_symbol_missing",
    "kconfig_assignment_not_enabled",
    "kconfig_unset_not_preserved",
    "missing_expected_module",
    "missing_expected_object",
    "build_target_not_materialized",
}
MISSING_MODULE_BUILD_REASONS = {
    "config_not_enabled",
    "config_dependency_not_enabled",
    "module_target_missing_from_makefile",
    "makefile_config_not_enabled",
}
VALIDATION_ENVIRONMENT_KINDS = {
    "load_validation_missing",
    "dependency_symvers_boundary",
    "target_symvers_boundary",
}
REPAIR_ROUTES = {"code_graph_repair"}


def validation_phase_issue_pairs(
    validation_report: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (phase, issue)
        for phase in validation_report.get("phases") or []
        if isinstance(phase, dict)
        for issue in phase.get("issues") or []
        if isinstance(issue, dict)
    ]


def is_build_metadata_file(path: str) -> bool:
    normalized = normalize_relpath(path)
    return Path(normalized).name in BUILD_METADATA_NAMES or normalized.endswith(".mk")


def missing_expected_module_reason(issue: dict[str, Any]) -> str:
    diagnosis = issue.get("build_output_diagnosis")
    if isinstance(diagnosis, dict):
        return str(diagnosis.get("reason") or "")
    return ""


def is_build_prep_issue(phase: dict[str, Any], issue: dict[str, Any]) -> bool:
    kind = str(issue.get("kind") or "")
    phase_name = str(phase.get("name") or "")
    if (
        str(issue.get("repair_relevance") or "")
        == "validation_environment"
    ):
        return False
    if str(issue.get("repair_relevance") or "") == "build_preparation":
        return True
    if kind == "command_failed" and phase_name == "kconfig-sync":
        return True
    if kind == "missing_expected_module":
        return missing_expected_module_reason(issue) in MISSING_MODULE_BUILD_REASONS
    if kind in BUILD_PREP_ISSUE_KINDS:
        return True
    for key in ("file", "path", "relative_path"):
        value = str(issue.get(key) or "")
        if value and is_build_metadata_file(value):
            return True
    return False


def is_validation_environment_issue(issue: dict[str, Any]) -> bool:
    if str(issue.get("repair_relevance") or "") == "validation_environment":
        return True
    return str(issue.get("kind") or "") in VALIDATION_ENVIRONMENT_KINDS


def route_diagnostic(phase: dict[str, Any], issue: dict[str, Any]) -> tuple[str, str]:
    if is_validation_environment_issue(issue):
        return "validation_environment", "validation/build artifact issue is not a source repair"
    if is_build_prep_issue(phase, issue):
        return "build_prep", "Kconfig/Makefile/config/build-output issue belongs to preparation"

    severity = str(issue.get("severity") or "").lower()
    if (
        severity in {"note", "warning", "info"}
        and not issue.get("warning_as_error")
        and not issue.get("migration_contract_warning")
    ):
        return "ignored", "non-error diagnostic remains supporting evidence"
    if issue.get("migration_contract_warning"):
        return (
            "code_graph_repair",
            "type/ABI warning in the migrated target remains an unmet compiler contract",
        )
    return "code_graph_repair", "non-preparation diagnostic enters CodeIndex root analysis"


def iter_routed_diagnostics(validation_report: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    event_index = 0
    for phase_index, phase in enumerate(validation_report.get("phases") or [], start=1):
        if not isinstance(phase, dict):
            continue
        phase_name = str(phase.get("name") or phase.get("phase") or f"phase-{phase_index}")
        for issue_index, issue in enumerate(phase.get("issues") or [], start=1):
            if not isinstance(issue, dict):
                continue
            event_index += 1
            route, reason = route_diagnostic(phase, issue)
            normalized = dict(issue)
            normalized["_phase_index"] = phase_index
            normalized["_issue_index"] = issue_index
            normalized["phase_name"] = phase_name
            normalized["diagnostic_route"] = route
            normalized["diagnostic_route_reason"] = reason
            normalized["diagnostic_event_id"] = stable_id(
                "diag",
                str(event_index),
                phase_name,
                str(normalized.get("file") or ""),
                str(normalized.get("line") or ""),
                str(normalized.get("column") or ""),
                str(normalized.get("raw") or normalized.get("message") or ""),
            )
            diagnostics.append(normalized)
    return diagnostics


def repair_routed_diagnostics(validation_report: dict[str, Any]) -> list[dict[str, Any]]:
    repair_diagnostics = [
        item
        for item in iter_routed_diagnostics(validation_report)
        if str(item.get("diagnostic_route") or "") in REPAIR_ROUTES
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in repair_diagnostics:
        occurrence_signature = diagnostic_occurrence_signature(item)
        if occurrence_signature not in grouped:
            grouped[occurrence_signature] = []
            order.append(occurrence_signature)
        grouped[occurrence_signature].append(item)

    coalesced: list[dict[str, Any]] = []
    for occurrence_signature in order:
        occurrences = grouped[occurrence_signature]
        canonical = dict(occurrences[0])
        signature = diagnostic_signature(canonical)
        event_ids = [
            str(item.get("diagnostic_event_id") or "")
            for item in occurrences
            if item.get("diagnostic_event_id")
        ]
        occurrence_records: list[dict[str, Any]] = []
        trigger_files: list[str] = []
        seen_trigger_files: set[str] = set()
        related_notes: list[dict[str, Any]] = []
        seen_notes: set[tuple[tuple[str, str], ...]] = set()
        for item in occurrences:
            record = {
                key: item.get(key)
                for key in (
                    "diagnostic_event_id",
                    "phase_name",
                    "_phase_index",
                    "_issue_index",
                    "file",
                    "line",
                    "column",
                    "message",
                    "raw",
                    "translation_unit",
                    "trigger_file",
                    "trigger_files",
                    "include_stack",
                    "object",
                    "module",
                )
                if item.get(key) not in (None, "", [])
            }
            occurrence_records.append(record)
            raw_trigger_files = [
                item.get("translation_unit"),
                item.get("trigger_file"),
                *(item.get("trigger_files") or []),
            ]
            for value in raw_trigger_files:
                path = str(value or "").strip().replace("\\", "/")
                if path and path not in seen_trigger_files:
                    seen_trigger_files.add(path)
                    trigger_files.append(path)
            for note in item.get("related_notes") or []:
                if not isinstance(note, dict):
                    continue
                identity = tuple(
                    sorted(
                        (str(key), repr(value))
                        for key, value in note.items()
                    )
                )
                if identity in seen_notes:
                    continue
                seen_notes.add(identity)
                related_notes.append(dict(note))

        canonical["diagnostic_signature"] = signature
        canonical["diagnostic_event_id"] = stable_id(
            "diag-event",
            occurrence_signature,
        )
        canonical["diagnostic_occurrence_count"] = len(occurrences)
        canonical["diagnostic_occurrence_event_ids"] = event_ids
        canonical["diagnostic_occurrences"] = occurrence_records
        if trigger_files:
            canonical["trigger_files"] = trigger_files
        if related_notes:
            canonical["related_notes"] = related_notes
        coalesced.append(canonical)
    return coalesced


def diagnostic_route_summary(validation_report_or_issues: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics = (
        iter_routed_diagnostics(validation_report_or_issues)
        if isinstance(validation_report_or_issues, dict)
        else list(validation_report_or_issues)
    )
    routes = Counter(str(item.get("diagnostic_route") or "unknown") for item in diagnostics)
    kinds = Counter(str(item.get("kind") or "unknown") for item in diagnostics)
    build_prep_count = routes.get("build_prep", 0)
    repair_routed_count = sum(routes.get(route, 0) for route in REPAIR_ROUTES)
    validation_environment_count = routes.get("validation_environment", 0)
    return {
        "schema": "diagnostic_route_summary_v2",
        "diagnostic_count": len(diagnostics),
        "routes": dict(sorted(routes.items())),
        "issue_kinds": dict(sorted(kinds.items())),
        "build_prep_count": build_prep_count,
        "validation_environment_count": validation_environment_count,
        "code_graph_repair_count": routes.get("code_graph_repair", 0),
        "repair_routed_count": repair_routed_count,
        "has_build_prep": bool(build_prep_count),
        "has_repair_routes": bool(repair_routed_count),
        "build_prep_only": bool(build_prep_count and not repair_routed_count),
        "mixed_build_prep_and_repair": bool(build_prep_count and repair_routed_count),
    }


def build_prep_stage_decision(
    validation_report: dict[str, Any],
    *,
    validation_status: str | None = None,
) -> dict[str, Any]:
    """Return the strict A-stage gate for a validation result.

    Any build-preparation diagnostic takes precedence over source repair.  Source
    diagnostics remain in the validation artifact but cannot enter B/C until a new
    validation proves that A is closed.
    """

    summary = diagnostic_route_summary(validation_report)
    status = str(validation_status or validation_report.get("status") or "")
    repairable_status = status in {
        "needs_build_preparation",
        "needs_patching",
        "patch_apply_incomplete",
    }
    needs_build_prep = repairable_status and bool(summary.get("has_build_prep"))
    if needs_build_prep and summary.get("mixed_build_prep_and_repair"):
        mode = "strict_build_prep_before_source_repair"
    elif needs_build_prep:
        mode = "exclusive_build_prep"
    else:
        mode = "none"
    return {
        **summary,
        "schema": "build_prep_stage_decision_v2",
        "validation_status": status,
        "needs_build_prep": needs_build_prep,
        "build_prep_route_mode": mode,
        "source_repair_blocked": needs_build_prep,
        "requires_revalidation_before_source_repair": needs_build_prep,
    }
