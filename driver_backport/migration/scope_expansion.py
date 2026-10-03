#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Iterable

from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    ensure_dir,
    normalize_relpath,
    ordered_unique_relpaths as _ordered_relpaths,
    sha256_file,
    stable_id,
    utc_now,
    write_json,
)
from driver_backport.migration.transplant import apply_initial_transplant_plan
from driver_backport.scope.semantic_scope import (
    build_integration_files_for_sources,
)
from driver_backport.planning.work_unit import (
    exact_absent_declaration_insertion_owner,
    exact_graph_scope_owners,
    unsafe_automatic_scope_owner,
)


SCOPE_EXPANSION_SCHEMA = "graph_scope_expansion_v2"
SCOPE_EXPANSION_STATE_NAME = "graph_scope_expansions.json"
_ACTIVE_EVENT_STATUSES = {
    "source_applied_pending_compile_context",
    "compile_context_ready",
    "permission_ready",
    "validated",
}


def scope_expansion_state_path(state_root: Path) -> Path:
    return state_root / SCOPE_EXPANSION_STATE_NAME


def _load_scope_state(spec: CaseSpec, state_root: Path) -> dict[str, Any]:
    path = scope_expansion_state_path(state_root)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema": SCOPE_EXPANSION_SCHEMA,
            "case_id": spec.case_id,
            "created_at": utc_now(),
            "promoted_files": [],
            "events": [],
        }
    payload["schema"] = SCOPE_EXPANSION_SCHEMA
    payload["case_id"] = spec.case_id
    payload.setdefault("events", [])
    payload.setdefault("promoted_files", [])
    return payload


def _event_is_active(event: dict[str, Any]) -> bool:
    status = str(event.get("status") or "")
    return status in _ACTIVE_EVENT_STATUSES or not status


def _refresh_promoted_files(payload: dict[str, Any]) -> None:
    events = [
        event
        for event in payload.get("events") or []
        if isinstance(event, dict)
    ]
    if events:
        payload["promoted_files"] = _ordered_relpaths(
            relative_path
            for event in events
            if _event_is_active(event)
            for relative_path in event.get("promoted_files") or []
        )


def _write_scope_state(state_root: Path, payload: dict[str, Any]) -> None:
    _refresh_promoted_files(payload)
    payload["updated_at"] = utc_now()
    write_json(scope_expansion_state_path(state_root), payload)


def graph_scope_expansion_summary(
    state_root: Path,
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Project reporting facts from the sole scope-expansion state owner."""

    path = scope_expansion_state_path(state_root)
    if not path.is_file():
        return {
            "schema": "graph_scope_expansion_summary_v1",
            "scope_expansion_count": 0,
            "event_count": 0,
            "committed_event_ids": [],
            "state_path": str(path.resolve()),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCOPE_EXPANSION_SCHEMA:
        raise ValueError("unsupported graph scope expansion state schema")
    stored_case_id = str(payload.get("case_id") or "")
    if case_id is not None and stored_case_id != str(case_id):
        raise ValueError(
            "graph scope expansion state belongs to another case: "
            f"{stored_case_id} != {case_id}"
        )
    events = payload.get("events")
    if not isinstance(events, list) or any(
        not isinstance(event, dict) for event in events
    ):
        raise ValueError("graph scope expansion events must be objects")
    committed_event_ids = [
        str(event.get("event_id") or "")
        for event in events
        if str(event.get("status") or "")
        in {"compile_context_ready", "permission_ready", "validated"}
        and str(event.get("event_id") or "")
    ]
    if len(committed_event_ids) != len(set(committed_event_ids)):
        raise ValueError("graph scope expansion event ids must be unique")
    return {
        "schema": "graph_scope_expansion_summary_v1",
        "scope_expansion_count": len(committed_event_ids),
        "event_count": len(events),
        "committed_event_ids": committed_event_ids,
        "state_path": str(path.resolve()),
    }


def _promote_exact_files(spec: CaseSpec, files: Iterable[object]) -> CaseSpec:
    promoted = _ordered_relpaths(files)
    if not promoted:
        return spec
    editable_roots = _ordered_relpaths([*spec.editable_roots, *promoted])
    return replace(spec, editable_roots=editable_roots)


def load_graph_scope_expansions(spec: CaseSpec, *, state_root: Path) -> CaseSpec:
    """Restore graph-proven exact-file permissions for a resumed migration."""

    path = scope_expansion_state_path(state_root)
    if not path.is_file():
        return spec
    payload = _load_scope_state(spec, state_root)
    _refresh_promoted_files(payload)
    promoted: list[str] = []
    for value in payload.get("promoted_files") or []:
        relative_path = normalize_relpath(str(value or ""))
        if not relative_path or spec.classify_edit_root(relative_path) == "forbidden":
            continue
        if spec.resolve_old(relative_path).is_file():
            promoted.append(relative_path)
    return _promote_exact_files(spec, promoted)


def pending_graph_scope_expansions(
    spec: CaseSpec,
    *,
    state_root: Path,
) -> list[dict[str, Any]]:
    """Return source transactions that still require compile-context validation."""

    payload = _load_scope_state(spec, state_root)
    return [
        {
            "schema": SCOPE_EXPANSION_SCHEMA,
            **dict(event),
            "state_path": str(scope_expansion_state_path(state_root).resolve()),
        }
        for event in payload.get("events") or []
        if isinstance(event, dict)
        and str(event.get("status") or "")
        == "source_applied_pending_compile_context"
    ]


def pending_graph_scope_validations(
    spec: CaseSpec,
    *,
    state_root: Path,
) -> list[dict[str, Any]]:
    """Return committed scope changes that have not reached a fresh build."""

    payload = _load_scope_state(spec, state_root)
    return [
        {
            "schema": SCOPE_EXPANSION_SCHEMA,
            **dict(event),
            "state_path": str(
                scope_expansion_state_path(state_root).resolve()
            ),
        }
        for event in payload.get("events") or []
        if isinstance(event, dict)
        and str(event.get("status") or "") == "compile_context_ready"
    ]


def complete_graph_scope_validations(
    spec: CaseSpec,
    *,
    state_root: Path,
    event_ids: Iterable[object],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Close only scope events observed by the supplied fresh validation."""

    selected_ids = {str(item) for item in event_ids if str(item)}
    payload = _load_scope_state(spec, state_root)
    completed: list[str] = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id") or "")
        if (
            event_id not in selected_ids
            or str(event.get("status") or "") != "compile_context_ready"
        ):
            continue
        event.update(
            {
                "status": "validated",
                "validated_at": utc_now(),
                "validation_cycle": validation.get("cycle"),
                "validation_status": validation.get("status"),
                "validation_path": validation.get("validation_path"),
            }
        )
        completed.append(event_id)
    _write_scope_state(state_root, payload)
    return {
        "schema": "graph_scope_validation_completion_v1",
        "status": "completed" if completed else "event_not_found",
        "completed_event_ids": completed,
        "requested_event_ids": sorted(selected_ids),
        "validation_cycle": validation.get("cycle"),
        "validation_path": validation.get("validation_path"),
        "state_path": str(scope_expansion_state_path(state_root).resolve()),
    }


def reconcile_graph_scope_transactions(
    spec: CaseSpec,
    *,
    state_root: Path,
) -> dict[str, Any]:
    """Reconcile scope intents after generic open-transaction recovery."""

    payload = _load_scope_state(spec, state_root)
    manifests_by_work_unit: dict[str, list[tuple[int, Path, dict[str, Any]]]] = {}
    for manifest_path in (state_root / "transactions").glob("**/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            modified = manifest_path.stat().st_mtime_ns
        except (OSError, ValueError):
            continue
        if str(manifest.get("case_id") or "") != spec.case_id:
            continue
        work_unit_id = str(manifest.get("work_unit_id") or "")
        if work_unit_id:
            manifests_by_work_unit.setdefault(work_unit_id, []).append(
                (modified, manifest_path.resolve(), manifest)
            )

    reconciled: list[dict[str, Any]] = []
    changed = False
    for event in payload.get("events") or []:
        if (
            not isinstance(event, dict)
            or str(event.get("status") or "") != "applying"
        ):
            continue
        work_unit_id = str(event.get("transaction_work_unit_id") or "")
        manifests = sorted(manifests_by_work_unit.get(work_unit_id, []))
        if not manifests:
            event["status"] = "aborted_before_apply"
            event["reconciled_at"] = utc_now()
            event["reconciliation_reason"] = "transaction_manifest_missing"
        else:
            _modified, manifest_path, manifest = manifests[-1]
            transaction_status = str(manifest.get("status") or "open")
            event["transaction_dir"] = str(manifest_path.parent)
            event["transaction_status"] = transaction_status
            event["reconciled_at"] = utc_now()
            if transaction_status == "applied":
                event["status"] = "source_applied_pending_compile_context"
                event["source_applied_at"] = str(
                    manifest.get("closed_at") or utc_now()
                )
            else:
                event["status"] = "rolled_back"
                event["reconciliation_reason"] = (
                    f"transaction_terminal_status:{transaction_status}"
                )
        changed = True
        reconciled.append(
            {
                "event_id": event.get("event_id"),
                "transaction_work_unit_id": work_unit_id,
                "status": event.get("status"),
                "transaction_dir": event.get("transaction_dir"),
            }
        )
    if changed:
        _write_scope_state(state_root, payload)
    return {
        "schema": "graph_scope_transaction_recovery_v1",
        "status": "reconciled" if changed else "clean",
        "reconciled_count": len(reconciled),
        "events": reconciled,
        "state_path": str(scope_expansion_state_path(state_root).resolve()),
    }


def _plan_unit_id(unit: dict[str, Any]) -> str:
    return str(unit.get("id") or unit.get("work_unit_id") or "")


def _plan_root_id(unit: dict[str, Any]) -> str:
    root = dict(
        unit.get("code_graph_root_cause_cluster")
        or unit.get("root_cause_cluster")
        or {}
    )
    return str(
        (unit.get("repair_decision") or {}).get("root_node_id")
        or root.get("node_id")
        or root.get("entity_id")
        or ""
    )


def _donor_file_records(spec: CaseSpec, relative_path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for donor_index, donor_root in enumerate(spec.donor_root_paths):
        path = spec.resolve_donor(relative_path, donor_index=donor_index)
        if not path.is_file():
            continue
        digest = sha256_file(path)
        records.append(
            {
                "donor_index": donor_index,
                "donor_root": str(donor_root),
                "path": str(path),
                "sha256": digest,
                "size": path.stat().st_size,
            }
        )
    return records


def graph_scope_materialization_candidates(
    spec: CaseSpec,
    repair_plan: dict[str, Any],
    *,
    work_unit_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return exact root/provider files proven by the diagnostic graph."""

    selected_ids = {str(item) for item in work_unit_ids or set() if str(item)}
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        unit_id = _plan_unit_id(unit)
        if selected_ids and unit_id not in selected_ids and str(unit.get("work_unit_id") or "") not in selected_ids:
            continue
        root_node_id = _plan_root_id(unit)
        if not root_node_id:
            continue
        materialization_files = set(
            _ordered_relpaths(
                [
                    *list(unit.get("scope_expansion_candidates") or []),
                    *list(unit.get("escalation_candidates") or []),
                ]
            )
        )
        owner_candidates = exact_graph_scope_owners(
            unit,
            root_node_id=root_node_id,
        )
        for owner in owner_candidates:
            if not isinstance(owner, dict):
                continue
            exact_declaration_action = exact_absent_declaration_insertion_owner(
                owner
            )
            if unsafe_automatic_scope_owner(owner) and not exact_declaration_action:
                continue
            relative_path = normalize_relpath(str(owner.get("file") or ""))
            node_id = str(owner.get("node_id") or "")
            owner_root_id = str(owner.get("root_node_id") or "")
            owner_role = str(owner.get("owner_role") or "")
            selected_root_owner = bool(
                (owner_role == "root_owner" and node_id == root_node_id)
                or (
                    owner_role in {"", "diagnostic_location_owner"}
                    and node_id == root_node_id
                )
            )
            selected_declaration_owner = bool(
                owner_role == "provider_declaration_owner"
                and node_id == root_node_id
                and (
                    owner.get("old_corresponding_entity_ids")
                    or exact_declaration_action
                )
            )
            if (
                not relative_path
                or not node_id
                or owner_root_id != root_node_id
                or not (
                    selected_root_owner
                    or owner_role == "provider_definition_owner"
                    or selected_declaration_owner
                )
                or relative_path not in materialization_files
                or str(owner.get("permission") or "")
                not in {"editable", "escalation"}
                or spec.classify_edit_root(relative_path)
                != str(owner.get("permission") or "")
            ):
                continue
            target = spec.resolve_old(relative_path)
            if target.exists() and not target.is_file():
                continue
            if target.is_file() and str(owner.get("permission") or "") != "escalation":
                continue
            donor_records = _donor_file_records(spec, relative_path)
            donor_hashes = {item["sha256"] for item in donor_records}
            if not donor_records or len(donor_hashes) != 1:
                continue
            identity = (unit_id, root_node_id, relative_path)
            if identity in seen:
                continue
            seen.add(identity)
            records.append(
                {
                    "work_unit_id": unit_id,
                    "root_node_id": root_node_id,
                    "node_id": node_id,
                    "owner_role": owner_role or None,
                    "file": relative_path,
                    "action": "promote_existing_owner" if target.is_file() else "copy_missing_provider",
                    "permission": str(owner.get("permission") or ""),
                    "old_exists": target.is_file(),
                    "donor": donor_records[0],
                    "equivalent_donor_count": len(donor_records),
                    "graph_reasons": list(owner.get("reasons") or []),
                    "semantic_gap_ids": sorted(
                        {
                            str(gap_id)
                            for gap_id in unit.get(
                                "semantic_gap_ids"
                            )
                            or []
                            if str(gap_id)
                        }
                    ),
                    "required_semantic_gap_ids": sorted(
                        {
                            str(gap_id)
                            for gap_id in unit.get(
                                "required_semantic_gap_ids"
                            )
                            or []
                            if str(gap_id)
                        }
                    ),
                }
            )
    return records


def materialize_graph_scope(
    spec: CaseSpec,
    repair_plan: dict[str, Any],
    *,
    state_root: Path,
    cycle: int,
    work_unit_ids: set[str] | None = None,
    allow_escalation: bool = False,
    execution_phase: str = "compiler_repair",
) -> tuple[CaseSpec, dict[str, Any]]:
    """Apply one graph-proven scope expansion and persist it for later cycles."""

    ensure_dir(state_root)
    candidates = graph_scope_materialization_candidates(
        spec,
        repair_plan,
        work_unit_ids=work_unit_ids,
    )
    state_path = scope_expansion_state_path(state_root)
    if not candidates:
        return spec, {
            "schema": SCOPE_EXPANSION_SCHEMA,
            "status": "no_graph_proven_candidates",
            "candidate_count": 0,
            "state_path": str(state_path.resolve()),
        }

    unauthorized_escalations = _ordered_relpaths(
        candidate["file"]
        for candidate in candidates
        if candidate.get("permission") == "escalation"
    )
    if unauthorized_escalations and not allow_escalation:
        return spec, {
            "schema": SCOPE_EXPANSION_SCHEMA,
            "status": "authorization_required",
            "candidate_count": len(candidates),
            "candidates": candidates,
            "escalation_files": unauthorized_escalations,
            "state_path": str(state_path.resolve()),
        }

    promoted_files = _ordered_relpaths(item["file"] for item in candidates)
    expanded_spec = _promote_exact_files(spec, promoted_files)
    expanded_spec.validate()
    copied_source_files = _ordered_relpaths(
        item["file"]
        for item in candidates
        if item["action"] == "copy_missing_provider"
    )
    build_integration_files = build_integration_files_for_sources(
        expanded_spec,
        set(copied_source_files),
    )
    transaction_identity = stable_id(
        "scope-expansion",
        *(sorted(f"{item['work_unit_id']}:{item['file']}" for item in candidates)),
    )
    event_id = stable_id(
        "scope-expansion-event",
        spec.case_id,
        str(max(0, int(cycle))),
        transaction_identity,
        utc_now(),
    )
    copy_candidates_by_file: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate["action"] != "copy_missing_provider":
            continue
        relative_path = str(candidate["file"])
        previous = copy_candidates_by_file.get(relative_path)
        if previous is not None and (
            str(previous["donor"].get("sha256") or "")
            != str(candidate["donor"].get("sha256") or "")
        ):
            raise ValueError(
                "one structural scope checkpoint selected conflicting donor "
                f"content for {relative_path}"
            )
        copy_candidates_by_file.setdefault(relative_path, candidate)
    copy_candidates = list(copy_candidates_by_file.values())
    transplant_plan = {
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "copy_operations": [
            {
                "file": candidate["file"],
                "action": "copy_missing_provider",
                "donor_path": candidate["donor"]["path"],
                "old_path": str(expanded_spec.resolve_old(candidate["file"])),
                "reasons": [
                    "typed diagnostic graph selected this missing provider owner"
                ],
            }
            for candidate in copy_candidates
        ],
        "build_integration": [
            {
                "file": relative_path,
                "old_path": str(expanded_spec.resolve_old(relative_path)),
                "donor_path": str(expanded_spec.resolve_donor(relative_path)),
                "action": "merge_scope_build_closure",
                "strategy": "structural_build_merge",
                "related_source_files": copied_source_files,
            }
            for relative_path in build_integration_files
        ],
        "shared_candidates": [],
    }
    state_payload = _load_scope_state(spec, state_root)
    scope_event = {
        "event_id": event_id,
        "timestamp": utc_now(),
        "cycle": max(0, int(cycle)),
        "execution_phase": str(execution_phase or "compiler_repair"),
        "status": "applying",
        "transaction_work_unit_id": transaction_identity,
        "candidates": candidates,
        "promoted_files": promoted_files,
        "build_integration_files": build_integration_files,
        "editable_roots_before": list(spec.editable_roots),
    }
    state_payload.setdefault("events", []).append(scope_event)
    _write_scope_state(state_root, state_payload)
    apply_result = apply_initial_transplant_plan(
        expanded_spec,
        transplant_plan,
        state_root=state_root,
        transaction_work_unit_id=transaction_identity,
        transaction_cycle=max(0, int(cycle)),
        allow_escalation=allow_escalation,
    )
    failed_count = sum(
        int(apply_result.get(key) or 0)
        for key in (
            "blocked_copy_operations",
            "failed_copy_operations",
            "blocked_build_integrations",
            "failed_build_integrations",
        )
    )
    if failed_count:
        scope_event.update(
            {
                "status": "rolled_back",
                "transaction_dir": apply_result.get("transaction_dir"),
                "apply_result": apply_result,
                "failed_at": utc_now(),
            }
        )
        _write_scope_state(state_root, state_payload)
        raise RuntimeError(
            "graph scope source/build closure materialization did not apply atomically"
        )
    copied_files = [
        normalize_relpath(str(result.get("file") or ""))
        for result in apply_result.get("results") or []
        if result.get("status") == "copied"
        and str(result.get("file") or "")
    ]
    source_or_build_changed = bool(
        copied_files
        or int(apply_result.get("applied_copy_operations") or 0)
        or int(apply_result.get("applied_build_integrations") or 0)
    )
    scope_event.update(
        {
            "status": "source_applied_pending_compile_context",
            "source_applied_at": utc_now(),
            "transaction_dir": apply_result.get("transaction_dir"),
            "copied_files": copied_files,
            "source_or_build_changed": source_or_build_changed,
            "validation_required": source_or_build_changed,
            "apply_result": apply_result,
        }
    )
    _write_scope_state(state_root, state_payload)

    return expanded_spec, {
        "schema": SCOPE_EXPANSION_SCHEMA,
        "status": "expanded",
        "event_id": event_id,
        "execution_phase": str(execution_phase or "compiler_repair"),
        "compile_context_status": "pending",
        "candidate_count": len(candidates),
        "unique_copy_operation_count": len(copy_candidates),
        "candidates": candidates,
        "copied_files": copied_files,
        "promoted_files": promoted_files,
        "build_integration_files": build_integration_files,
        "build_closure_apply": apply_result,
        "source_or_build_changed": source_or_build_changed,
        "validation_required": source_or_build_changed,
        "transaction_dir": apply_result.get("transaction_dir"),
        "state_path": str(state_path.resolve()),
        "editable_roots_before": list(spec.editable_roots),
    }


def commit_graph_scope(
    spec: CaseSpec,
    *,
    state_root: Path,
    scope_expansion: dict[str, Any],
    compile_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Commit source scope only after compile-context materialization succeeds."""

    payload = _load_scope_state(spec, state_root)
    event_id = str(scope_expansion.get("event_id") or "")
    transaction_dir = str(scope_expansion.get("transaction_dir") or "")
    committed: list[str] = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        matches = bool(
            (event_id and str(event.get("event_id") or "") == event_id)
            or (
                transaction_dir
                and str(event.get("transaction_dir") or "") == transaction_dir
            )
        )
        if not matches:
            continue
        if str(event.get("status") or "") not in {
            "source_applied_pending_compile_context",
            "compile_context_ready",
        }:
            continue
        validation_required = bool(
            event.get("validation_required", True)
        )
        event["status"] = (
            "compile_context_ready"
            if validation_required
            else "permission_ready"
        )
        event["compile_context_ready_at"] = utc_now()
        event["compile_context"] = dict(compile_context or {})
        committed.append(str(event.get("event_id") or ""))
    _write_scope_state(state_root, payload)
    return {
        "schema": "graph_scope_commit_v1",
        "status": "committed" if committed else "event_not_found",
        "committed_event_ids": committed,
        "state_path": str(scope_expansion_state_path(state_root).resolve()),
    }
