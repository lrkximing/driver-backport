#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from driver_backport.analysis.code_index import (
    CodeIndex,
    _resolve_include_target,
    parse_include_directives,
)
from driver_backport.core.common import (
    ensure_dir,
    load_json,
    normalize_relpath,
    ordered_unique_relpaths,
    stable_id,
    utc_now,
    write_json,
)
from driver_backport.core.executor import classify_edit_permission
from driver_backport.analysis.diagnostic_identity import (
    diagnostic_event_record,
    diagnostic_signature,
)
from driver_backport.planning.diagnostic_routing import (
    build_prep_stage_decision,
    diagnostic_route_summary,
    repair_routed_diagnostics,
)
from driver_backport.planning.code_graph_scope import (
    _record_owner_candidate,
    is_shared_kernel_implementation_path,
)
from driver_backport.planning.root_binding import (
    graph_root_binding_key,
    mapped_root_alternative_ids,
)
WORK_UNIT_STATES = {
    "created",
    "analyzing",
    "planning",
    "patching",
    "applying",
    "validating",
    "succeeded",
    "failed",
    "escalated",
    "abandoned",
}

WORK_UNIT_TRANSITIONS = {
    "created": {"analyzing", "abandoned"},
    "analyzing": {"planning", "failed", "abandoned"},
    "planning": {"patching", "escalated", "failed", "abandoned"},
    "patching": {"applying", "failed", "abandoned"},
    "applying": {"validating", "failed", "abandoned"},
    "validating": {"succeeded", "failed", "patching", "abandoned"},
    "failed": {"planning", "abandoned"},
    "escalated": {"planning", "abandoned"},
    "succeeded": {"planning"},
    "abandoned": {"planning"},
}

ROOT_CAUSE_WORK_UNIT_SCHEMA = "root_cause_work_unit_v3"


def validate_work_unit_state(value: str) -> str:
    state = str(value or "created")
    if state not in WORK_UNIT_STATES:
        raise ValueError(f"invalid work unit state: {state}")
    return state


def validate_work_unit_transition(current: str, new: str) -> str:
    source = validate_work_unit_state(current)
    target = validate_work_unit_state(new)
    if target not in WORK_UNIT_TRANSITIONS.get(source, set()):
        raise ValueError(f"invalid work unit transition: {source} -> {target}")
    return target


def transition_work_unit(work_unit: dict, *, new_state: str, reason: str) -> dict:
    updated = dict(work_unit)
    current = validate_work_unit_state(str(updated.get("status") or "created"))
    target = validate_work_unit_transition(current, new_state)
    updated["status"] = target
    updated["updated_at"] = utc_now()
    updated["retry_count"] = int(updated.get("retry_count") or 0) + (1 if target == "planning" else 0)
    history = list(updated.get("history") or [])
    history.append(
        {
            "at": updated["updated_at"],
            "state": target,
            "reason": reason,
        }
    )
    updated["history"] = history
    return updated


def work_unit_path(work_units_dir: Path, work_unit_id: str) -> Path:
    return work_units_dir / f"{work_unit_id}.json"


def load_work_unit(path: Path) -> dict:
    payload = dict(load_json(path))
    if payload.get("schema") != ROOT_CAUSE_WORK_UNIT_SCHEMA:
        raise ValueError(
            f"unsupported work unit schema in {path}: "
            f"{payload.get('schema')!r}; expected {ROOT_CAUSE_WORK_UNIT_SCHEMA!r}"
        )
    payload["status"] = validate_work_unit_state(str(payload.get("status") or "created"))
    return payload


def load_work_units(work_units_dir: Path) -> dict[str, dict]:
    if not work_units_dir.exists():
        return {}
    result: dict[str, dict] = {}
    for path in sorted(work_units_dir.glob("*.json")):
        work_unit = load_work_unit(path)
        result[str(work_unit["id"])] = work_unit
    return result


def write_work_unit_file(work_unit: dict, path: Path) -> Path:
    if work_unit.get("schema") != ROOT_CAUSE_WORK_UNIT_SCHEMA:
        raise ValueError(
            "cannot persist a work unit without the current "
            f"{ROOT_CAUSE_WORK_UNIT_SCHEMA!r} schema"
        )
    validate_work_unit_state(str(work_unit.get("status") or "created"))
    write_json(path, work_unit)
    return path


def write_work_unit(work_unit: dict, *, work_units_dir: Path) -> Path:
    path = work_unit_path(ensure_dir(work_units_dir), str(work_unit["id"]))
    return write_work_unit_file(work_unit, path)


def _advance_work_unit_to_state(work_unit: dict, steps: list[tuple[str, str]]) -> dict:
    updated = dict(work_unit)
    for target, reason in steps:
        current = validate_work_unit_state(str(updated.get("status") or "created"))
        if current == target:
            continue
        updated = transition_work_unit(updated, new_state=target, reason=reason)
    return updated


def promote_work_unit_to_planning(work_unit: dict, *, reason: str) -> dict:
    current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "planning":
        return dict(work_unit)
    if current == "created":
        return _advance_work_unit_to_state(
            work_unit,
            [
                ("analyzing", "collecting evidence and causal signals"),
                ("planning", reason),
            ],
        )
    if current == "analyzing":
        return _advance_work_unit_to_state(work_unit, [("planning", reason)])
    if current == "validating":
        return _advance_work_unit_to_state(
            work_unit,
            [
                ("failed", "validation still reports this work unit"),
                ("planning", reason),
            ],
        )
    if current == "failed":
        return _advance_work_unit_to_state(work_unit, [("planning", reason)])
    if current == "escalated":
        return _advance_work_unit_to_state(work_unit, [("planning", reason)])
    if current in {"succeeded", "abandoned"}:
        return _advance_work_unit_to_state(work_unit, [("planning", reason)])
    if current in {"patching", "applying"}:
        return _advance_work_unit_to_state(
            work_unit,
            [
                ("failed", "repair execution did not complete cleanly before replanning"),
                ("planning", reason),
            ],
        )
    return dict(work_unit)


def promote_work_unit_to_patching(work_unit: dict, *, reason: str) -> dict:
    current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "patching":
        return dict(work_unit)
    if current in {"created", "analyzing", "failed", "escalated", "validating"}:
        work_unit = promote_work_unit_to_planning(work_unit, reason="preparing repair plan before patch generation")
        current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "planning":
        return _advance_work_unit_to_state(work_unit, [("patching", reason)])
    return dict(work_unit)


def promote_work_unit_to_validating(work_unit: dict, *, reason: str) -> dict:
    current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "validating":
        return dict(work_unit)
    if current in {"created", "analyzing", "failed", "escalated"}:
        work_unit = promote_work_unit_to_patching(work_unit, reason="patch generation started before validation handoff")
        current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "planning":
        work_unit = promote_work_unit_to_patching(work_unit, reason="patch generation started before validation handoff")
        current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "patching":
        return _advance_work_unit_to_state(
            work_unit,
            [
                ("applying", "applying repair patch to old tree"),
                ("validating", reason),
            ],
        )
    if current == "applying":
        return _advance_work_unit_to_state(work_unit, [("validating", reason)])
    return dict(work_unit)


def mark_work_unit_succeeded(work_unit: dict, *, reason: str) -> dict:
    current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "succeeded":
        return dict(work_unit)
    if current == "validating":
        return _advance_work_unit_to_state(work_unit, [("succeeded", reason)])
    if current in {"planning", "patching", "applying", "failed", "escalated", "analyzing", "created"}:
        work_unit = promote_work_unit_to_validating(work_unit, reason="recording validation outcome after repair")
        return _advance_work_unit_to_state(work_unit, [("succeeded", reason)])
    return dict(work_unit)


def mark_work_unit_validation_deferred(
    work_unit: dict,
    *,
    reason: str,
    coverage: dict,
) -> dict:
    updated = promote_work_unit_to_validating(
        work_unit,
        reason="repair outcome is awaiting translation-unit coverage",
    )
    updated = dict(updated)
    updated["updated_at"] = utc_now()
    history = list(updated.get("history") or [])
    history.append(
        {
            "at": updated["updated_at"],
            "state": "validating",
            "reason": reason,
        }
    )
    updated["history"] = history
    observations = list(updated.get("validation_coverage_history") or [])
    observations.append(
        {
            "at": updated["updated_at"],
            **coverage,
        }
    )
    updated["validation_coverage_history"] = observations[-16:]
    return updated


def mark_work_unit_failed(work_unit: dict, *, reason: str) -> dict:
    current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "failed":
        return dict(work_unit)
    if current == "validating":
        return _advance_work_unit_to_state(work_unit, [("failed", reason)])
    if current == "planning":
        return _advance_work_unit_to_state(
            work_unit,
            [
                ("patching", "recording failed execution without a completed validation handoff"),
                ("applying", "recording failed execution without a completed validation handoff"),
                ("validating", "recording failed execution without a completed validation handoff"),
                ("failed", reason),
            ],
        )
    if current == "patching":
        return _advance_work_unit_to_state(
            work_unit,
            [
                ("applying", "recording failed execution without a completed validation handoff"),
                ("validating", "recording failed execution without a completed validation handoff"),
                ("failed", reason),
            ],
        )
    if current == "applying":
        return _advance_work_unit_to_state(
            work_unit,
            [
                ("validating", "recording failed execution without a completed validation handoff"),
                ("failed", reason),
            ],
        )
    if current in {"created", "analyzing"}:
        work_unit = promote_work_unit_to_planning(work_unit, reason="recording failure after initial work-unit creation")
        return mark_work_unit_failed(work_unit, reason=reason)
    if current == "escalated":
        work_unit = promote_work_unit_to_planning(work_unit, reason="escalation returned unresolved and requires failure recording")
        return mark_work_unit_failed(work_unit, reason=reason)
    return dict(work_unit)


def mark_work_unit_escalated(work_unit: dict, *, reason: str) -> dict:
    current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "escalated":
        return dict(work_unit)
    if current in {"created", "analyzing", "failed", "validating"}:
        work_unit = promote_work_unit_to_planning(work_unit, reason="escalation requested during repair planning")
        current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "planning":
        return _advance_work_unit_to_state(work_unit, [("escalated", reason)])
    if current in {"patching", "applying"}:
        work_unit = mark_work_unit_failed(work_unit, reason="repair execution halted because escalation is required")
        work_unit = promote_work_unit_to_planning(work_unit, reason="escalation requested after failed execution")
        return _advance_work_unit_to_state(work_unit, [("escalated", reason)])
    return dict(work_unit)


def mark_work_unit_abandoned(work_unit: dict, *, reason: str) -> dict:
    """Close a work-unit identity that was replaced by an exact successor."""

    current = validate_work_unit_state(str(work_unit.get("status") or "created"))
    if current == "abandoned":
        return dict(work_unit)
    if current == "succeeded":
        work_unit = promote_work_unit_to_planning(
            work_unit,
            reason="reopening the historical identity before recording its exact successor",
        )
    return _advance_work_unit_to_state(work_unit, [("abandoned", reason)])


def _compact_repair_owner(owner: dict) -> dict:
    return {
        "id": owner.get("id"),
        "owner_kind": owner.get("owner_kind"),
        "owner_file": owner.get("owner_file"),
        "owner_symbol": owner.get("owner_symbol"),
        "owner_entity_id": owner.get("owner_entity_id"),
        "owner_contract_key": owner.get("owner_contract_key"),
        "owner_entity_kind": owner.get("owner_entity_kind"),
        "owner_entity_tree": owner.get("owner_entity_tree"),
        "owner_role": owner.get("owner_role"),
        "owner_entity_container": owner.get("owner_entity_container"),
        "owner_entity_extent": dict(
            owner.get("owner_entity_extent") or {}
        ),
        "owner_is_definition": owner.get("owner_is_definition"),
        "old_corresponding_entity_ids": list(
            owner.get("old_corresponding_entity_ids") or []
        ),
        "donor_only_entity": bool(owner.get("donor_only_entity")),
        "exact_absent_entity_insertion_allowed": bool(
            owner.get("exact_absent_entity_insertion_allowed")
        ),
        "exact_absent_declaration_insertion_allowed": bool(
            owner.get("exact_absent_declaration_insertion_allowed")
        ),
        "allowed_for_entity_materialization": bool(
            owner.get("allowed_for_entity_materialization")
        ),
        "entity_materialization_files": list(
            owner.get("entity_materialization_files") or []
        ),
        "driver_interface_ownership": dict(
            owner.get("driver_interface_ownership") or {}
        ),
        "exact_missing_include_materialization": dict(
            owner.get("exact_missing_include_materialization") or {}
        ),
        "reason": owner.get("reason"),
        "evidence": list(owner.get("evidence") or []),
        "allowed_as_patch_target": bool(owner.get("allowed_as_patch_target")),
        "allowed_patch_files": list(owner.get("allowed_patch_files") or []),
        "allowed_patch_permissions": dict(owner.get("allowed_patch_permissions") or {}),
        "owner_permission": owner.get("owner_permission"),
        "old_exists": bool(owner.get("old_exists")),
        "symptom_files": list(owner.get("symptom_files") or []),
        "trigger_files": list(owner.get("trigger_files") or []),
        "read_only_evidence_files": list(owner.get("read_only_evidence_files") or []),
        "diagnostic_structure": dict(owner.get("diagnostic_structure") or {}),
    }


def _proven_driver_interface_ownership(owner: dict[str, Any]) -> bool:
    """Return whether active declarations authorize donor materialization."""

    evidence = owner.get("driver_interface_ownership")
    if not isinstance(evidence, dict):
        return False
    demanded_declaration_count = int(
        evidence.get("demanded_external_declaration_count")
        if "demanded_external_declaration_count" in evidence
        else evidence.get("external_declaration_count")
        or 0
    )
    return bool(
        evidence.get("schema") == "driver_interface_header_ownership_v1"
        and evidence.get("status")
        in {"proven", "proven_provider_closure"}
        and int(evidence.get("exact_binding_count") or 0)
        == demanded_declaration_count
        and not list(evidence.get("unmatched_declarations") or [])
        and not list(evidence.get("ambiguous_declarations") or [])
    )


def exact_absent_declaration_insertion_owner(
    owner: dict[str, Any],
) -> bool:
    """Return whether one exact donor declaration may be inserted.

    This is action authority, not permission to edit the surrounding shared
    file. Callers must continue to bind the hunk to this declaration and allow
    insertion-only application.
    """

    return bool(
        owner.get("exact_absent_declaration_insertion_allowed")
        and (
            str(owner.get("owner_role") or "")
            == "provider_declaration_owner"
            or (
                str(owner.get("owner_role") or "") == "root_owner"
                and owner.get(
                    "root_declaration_from_existing_definition"
                ) is True
            )
        )
        and bool(owner.get("old_exists"))
        and str(owner.get("owner_entity_tree") or "").startswith("donor_")
        and not list(
            owner.get("old_corresponding_entity_ids")
            or owner.get("old_counterpart_ids")
            or []
        )
    )


def exact_missing_include_materialization_owner(
    owner: dict[str, Any],
) -> bool:
    """Return whether one diagnosed include authorizes one missing file."""

    evidence = owner.get("exact_missing_include_materialization")
    if not isinstance(evidence, dict):
        return False
    owner_file = normalize_relpath(
        str(owner.get("file") or owner.get("owner_file") or "")
    )
    owner_node_id = str(
        owner.get("node_id") or owner.get("owner_entity_id") or ""
    )
    return bool(
        evidence.get("schema")
        == "exact_missing_include_materialization_v1"
        and evidence.get("status") == "proven"
        and owner_file
        and owner_file
        == normalize_relpath(str(evidence.get("file") or ""))
        and owner_node_id
        and owner_node_id == str(evidence.get("root_node_id") or "")
        and str(owner.get("owner_role") or "") == "root_owner"
        and owner.get("old_exists") is False
        and str(owner.get("owner_entity_tree") or "").startswith(
            "donor_"
        )
        and bool(evidence.get("diagnostic_node_ids"))
        and bool(evidence.get("include_node_ids"))
        and bool(evidence.get("unresolved_include_node_ids"))
        and bool(evidence.get("consumer_file_node_ids"))
    )


def repair_owner_has_action_authority(owner: dict[str, Any]) -> bool:
    """Separate causal action owners from read-only attribution evidence."""

    if str(owner.get("owner_kind") or "") == "unlocalized":
        return False
    if owner.get("allowed_as_patch_target") is True:
        return True
    if (
        owner.get("allowed_for_entity_materialization") is True
        and bool(owner.get("entity_materialization_files"))
    ):
        return True
    return str(owner.get("owner_role") or "") in {
        "root_owner",
        "provider_definition_owner",
        "provider_declaration_owner",
        "consumer_integration_owner",
        "include_relation_owner",
    }


def unsafe_automatic_scope_owner(owner: dict[str, Any]) -> bool:
    """Reject donor-only shared owners from automatic scope admission.

    A matched old-tree entity in these paths remains adaptable.  What is
    unsafe is treating a donor-only shared entity (or an absent shared file)
    as a bounded provider: that silently imports release-level dependencies.
    """

    relative_path = normalize_relpath(
        str(owner.get("file") or owner.get("owner_file") or "")
    )
    if not relative_path or not is_shared_kernel_implementation_path(relative_path):
        return False
    if exact_missing_include_materialization_owner(owner):
        return False
    if _proven_driver_interface_ownership(owner):
        return False
    if not bool(owner.get("old_exists")):
        return True
    entity_tree = str(
        owner.get("owner_entity_tree") or owner.get("tree") or ""
    )
    donor_only = bool(owner.get("donor_only_entity")) or entity_tree.startswith(
        "donor_"
    )
    old_counterparts = list(
        owner.get("old_corresponding_entity_ids")
        or owner.get("old_counterpart_ids")
        or []
    )
    return donor_only and not old_counterparts


def _same_external_contract(
    declaration: dict[str, Any],
    definition: dict[str, Any],
) -> bool:
    """Match one declaration to its exact donor definition, not by name alone."""

    if (
        not definition.get("is_definition")
        or str(declaration.get("kind") or "")
        != str(definition.get("kind") or "")
        or str(declaration.get("contract_key") or "")
        != str(definition.get("contract_key") or "")
    ):
        return False
    declaration_usr = str(declaration.get("usr") or "")
    definition_usr = str(definition.get("usr") or "")
    declaration_id = str(
        declaration.get("id") or declaration.get("entity_id") or ""
    )
    definition_id = str(
        definition.get("id") or definition.get("entity_id") or ""
    )
    exact_identity = bool(
        (declaration_usr and declaration_usr == definition_usr)
        or (declaration_id and declaration_id == definition_id)
    )
    if not exact_identity:
        return False
    declaration_signature = str(
        declaration.get("signature_fingerprint") or ""
    )
    definition_signature = str(
        definition.get("signature_fingerprint") or ""
    )
    return bool(
        not declaration_signature
        or (
            definition_signature
            and declaration_signature == definition_signature
        )
    )


def _driver_interface_header_ownership(
    spec: object,
    index: CodeIndex | None,
    *,
    donor_tree: str,
    header_file: str,
) -> dict[str, Any]:
    """Prove that a missing shared-path header is owned by selected entries.

    Every active function or variable declaration that needs an out-of-header
    provider must have one exact definition in the designated donor entry
    files. Self-contained types, fields, enums, and macros do not widen this
    provider boundary.
    """

    relative_path = normalize_relpath(header_file)
    base: dict[str, Any] = {
        "schema": "driver_interface_header_ownership_v1",
        "status": "unproven",
        "donor_tree": donor_tree,
        "header_file": relative_path,
        "entry_files": [],
        "external_declaration_count": 0,
        "demanded_external_declaration_count": 0,
        "exact_binding_count": 0,
        "exact_bindings": [],
        "unmatched_declarations": [],
        "ambiguous_declarations": [],
    }
    if (
        index is None
        or not donor_tree.startswith("donor_")
        or Path(relative_path).suffix.lower() not in {".h", ".hpp"}
    ):
        base["reason"] = "not a donor header with a bounded ownership index"
        return base

    entry_files = ordered_unique_relpaths(
        str(item)
        for item in getattr(spec, "entry_files", []) or []
        if Path(str(item)).suffix.lower() in {".c", ".cc", ".cpp"}
    )
    base["entry_files"] = entry_files
    if not entry_files:
        base["reason"] = (
            "no designated donor source entry can own external declarations"
        )
        return base

    parse_reports: list[dict[str, Any]] = []
    for entry_file in entry_files:
        try:
            entry_report = index.index_file(donor_tree, entry_file)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            parse_reports.append(
                {
                    "translation_unit_file": entry_file,
                    "exists": False,
                    "parse_ok": False,
                    "parse_error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if not entry_report.get("exists"):
            continue
        try:
            parse_reports.append(
                index.index_file_from_translation_unit(
                    donor_tree,
                    relative_path,
                    entry_file,
                )
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
            parse_reports.append(
                {
                    "translation_unit_file": entry_file,
                    "exists": False,
                    "parse_ok": False,
                    "parse_error": f"{type(error).__name__}: {error}",
                }
            )
    base["header_parse_reports"] = [
        {
            "translation_unit_file": report.get("translation_unit_file"),
            "exists": bool(report.get("exists")),
            "parse_ok": bool(report.get("parse_ok")),
            "parse_error": report.get("parse_error"),
        }
        for report in parse_reports
    ]
    if not parse_reports or not any(
        report.get("exists") and report.get("parse_ok")
        for report in parse_reports
    ):
        base["reason"] = (
            "header could not be parsed under a designated donor entry context"
        )
        return base

    header_entities = list(
        index.entities_by_file.get((donor_tree, relative_path), [])
    )
    declarations_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entity in header_entities:
        if (
            str(entity.get("kind") or "") not in {"function", "variable"}
            or bool(entity.get("is_definition"))
            or not str(entity.get("contract_key") or "")
        ):
            continue
        interface = (
            entity.get("interface")
            if isinstance(entity.get("interface"), dict)
            else {}
        )
        if str(interface.get("linkage") or "").upper() == "INTERNAL":
            continue
        identity = (
            str(entity.get("id") or entity.get("entity_id") or ""),
            str(entity.get("contract_key") or ""),
            str(entity.get("signature_fingerprint") or ""),
        )
        declarations_by_identity.setdefault(identity, dict(entity))
    declarations = list(declarations_by_identity.values())
    base["external_declaration_count"] = len(declarations)
    if not declarations:
        base["reason"] = (
            "header exposes no active external provider declaration"
        )
        return base

    referenced_entity_ids: set[str] = set()
    referenced_contract_keys: set[str] = set()
    for entry_file in entry_files:
        for reference in index.references_by_file.get(
            (donor_tree, entry_file), []
        ):
            if not isinstance(reference, dict):
                continue
            referenced_entity = (
                reference.get("referenced_entity")
                if isinstance(reference.get("referenced_entity"), dict)
                else {}
            )
            referenced_entity_ids.update(
                str(value)
                for value in (
                    reference.get("referenced_entity_id"),
                    referenced_entity.get("id"),
                    referenced_entity.get("entity_id"),
                )
                if str(value or "")
            )
            referenced_contract_keys.update(
                str(value)
                for value in (
                    reference.get("referenced_contract_key"),
                    referenced_entity.get("contract_key"),
                    referenced_entity.get("key"),
                )
                if str(value or "")
            )

    def declaration_is_exercised(declaration: dict[str, Any]) -> bool:
        entity_id = str(
            declaration.get("id")
            or declaration.get("entity_id")
            or ""
        )
        contract_key = str(declaration.get("contract_key") or "")
        return bool(
            (entity_id and entity_id in referenced_entity_ids)
            or (
                contract_key
                and contract_key in referenced_contract_keys
            )
        )

    exercised_declaration_ids = {
        str(
            declaration.get("id")
            or declaration.get("entity_id")
            or ""
        )
        for declaration in declarations
        if declaration_is_exercised(declaration)
    }
    exercised_declaration_ids.discard("")
    base["exercised_external_declaration_count"] = len(
        exercised_declaration_ids
    )
    base["demanded_external_declaration_count"] = len(
        exercised_declaration_ids
    )

    entry_definitions = [
        entity
        for entry_file in entry_files
        for entity in index.entities_by_file.get(
            (donor_tree, entry_file),
            [],
        )
        if entity.get("is_definition")
    ]
    bindings: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for declaration in declarations:
        exercised = declaration_is_exercised(declaration)
        matches_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
        for definition in entry_definitions:
            if not _same_external_contract(declaration, definition):
                continue
            match_identity = (
                normalize_relpath(str(definition.get("file") or "")),
                str(
                    definition.get("id")
                    or definition.get("entity_id")
                    or ""
                ),
                str(definition.get("signature_fingerprint") or ""),
            )
            matches_by_identity.setdefault(match_identity, dict(definition))
        matches = list(matches_by_identity.values())
        declaration_record = {
            "entity_id": declaration.get("id")
            or declaration.get("entity_id"),
            "symbol": declaration.get("symbol")
            or declaration.get("name"),
            "contract_key": declaration.get("contract_key"),
            "signature_fingerprint": declaration.get(
                "signature_fingerprint"
            ),
        }
        if not matches:
            if exercised:
                unmatched.append(declaration_record)
            continue
        if len(matches) != 1:
            if exercised:
                ambiguous.append(
                    {
                        **declaration_record,
                        "candidate_definition_ids": sorted(
                            str(
                                item.get("id")
                                or item.get("entity_id")
                                or ""
                            )
                            for item in matches
                        ),
                    }
                )
            continue
        if not exercised:
            continue
        definition = matches[0]
        bindings.append(
            {
                **declaration_record,
                "definition_entity_id": definition.get("id")
                or definition.get("entity_id"),
                "definition_file": normalize_relpath(
                    str(definition.get("file") or "")
                ),
            }
        )

    base.update(
        {
            "exact_binding_count": len(bindings),
            "exact_bindings": bindings,
            "unmatched_declarations": unmatched,
            "ambiguous_declarations": ambiguous,
        }
    )
    if unmatched or ambiguous:
        base["status"] = "external_providers_outside_selected_entries"
        base["reason"] = (
            "one or more entry-referenced header declarations lack a unique "
            "exact definition in the designated donor entry files"
        )
        return base
    base["status"] = "proven"
    base["reason"] = (
        "all entry-referenced external header declarations bind to exact "
        "entry definitions or require no external provider"
    )
    return base


def _exact_missing_include_materialization_evidence(
    diagnostic_graph: dict[str, Any],
    root_node_id: str,
    spec: object,
) -> dict[str, Any]:
    """Prove one exact missing include file from typed diagnostic edges."""

    nodes = {
        str(node.get("id") or ""): node
        for node in diagnostic_graph.get("nodes") or []
        if isinstance(node, dict) and str(node.get("id") or "")
    }
    edges = [
        edge
        for edge in diagnostic_graph.get("edges") or []
        if isinstance(edge, dict)
    ]
    root = nodes.get(root_node_id) or {}
    relative_path = normalize_relpath(
        str(root.get("file") or root.get("path") or "")
    )
    base: dict[str, Any] = {
        "schema": "exact_missing_include_materialization_v1",
        "status": "unproven",
        "root_node_id": root_node_id,
        "file": relative_path,
        "diagnostic_node_ids": [],
        "include_node_ids": [],
        "unresolved_include_node_ids": [],
        "consumer_file_node_ids": [],
    }
    if (
        str(root.get("kind") or "") != "File"
        or not str(root.get("tree") or "").startswith("donor_")
        or not relative_path
    ):
        base["reason"] = "selected root is not one concrete donor file"
        return base

    unresolved_ids = {
        str(edge.get("from") or "")
        for edge in edges
        if str(edge.get("type") or "") == "candidate_resolution"
        and str(edge.get("to") or "") == root_node_id
        and str(
            (nodes.get(str(edge.get("from") or "")) or {}).get(
                "kind"
            )
            or ""
        )
        == "UnresolvedInclude"
    }
    exact_unresolved_ids = {
        unresolved_id
        for unresolved_id in unresolved_ids
        if {
            str(edge.get("to") or "")
            for edge in edges
            if str(edge.get("type") or "") == "candidate_resolution"
            and str(edge.get("from") or "") == unresolved_id
        }
        == {root_node_id}
    }
    include_ids = {
        str(edge.get("from") or "")
        for edge in edges
        if str(edge.get("type") or "") == "unresolved_dependency"
        and str(edge.get("to") or "") in exact_unresolved_ids
        and str(
            (nodes.get(str(edge.get("from") or "")) or {}).get(
                "kind"
            )
            or ""
        )
        == "Include"
        and str(
            (nodes.get(str(edge.get("from") or "")) or {}).get(
                "tree"
            )
            or ""
        )
        == "old"
        and (nodes.get(str(edge.get("from") or "")) or {}).get(
            "resolved"
        )
        is False
    }
    diagnostic_ids = {
        str(edge.get("from") or "")
        for edge in edges
        if str(edge.get("type") or "") == "diagnoses"
        and str(edge.get("to") or "") in include_ids
        and str(
            (nodes.get(str(edge.get("from") or "")) or {}).get(
                "kind"
            )
            or ""
        )
        == "Diagnostic"
    }
    consumer_ids = {
        str(edge.get("to") or "")
        for edge in edges
        if str(edge.get("type") or "") == "occurs_at"
        and str(edge.get("from") or "") in diagnostic_ids
        and str(
            (nodes.get(str(edge.get("to") or "")) or {}).get("kind")
            or ""
        )
        == "File"
        and str(
            (nodes.get(str(edge.get("to") or "")) or {}).get("tree")
            or ""
        )
        == "old"
    }
    include_source_files = {
        normalize_relpath(str((nodes.get(include_id) or {}).get("source_file") or ""))
        for include_id in include_ids
    }
    consumer_files = {
        normalize_relpath(
            str(
                (nodes.get(consumer_id) or {}).get("file")
                or (nodes.get(consumer_id) or {}).get("path")
                or ""
            )
        )
        for consumer_id in consumer_ids
    }
    if (
        not exact_unresolved_ids
        or not include_ids
        or not diagnostic_ids
        or not consumer_ids
        or not include_source_files
        or "" in include_source_files
        or include_source_files != consumer_files
    ):
        base["reason"] = "typed missing-include path is incomplete or ambiguous"
        return base
    donor_tree = str(root.get("tree") or "")
    try:
        donor_index = int(donor_tree.split("_", 1)[1])
        donor_root = Path(spec.donor_root_paths[donor_index])
        old_root = Path(spec.resolve_old(""))
        driver_root = str(spec.driver_root)
    except (AttributeError, IndexError, OSError, TypeError, ValueError):
        base["reason"] = "donor or target tree is unavailable for include validation"
        return base
    donor_file = donor_root / relative_path
    if not donor_file.is_file():
        base["reason"] = "selected donor include file is unavailable"
        return base
    missing_dependencies = sorted(
        {
            str(include.get("name") or "")
            for include in parse_include_directives(
                root=donor_root,
                driver_root=donor_root / driver_root,
                relative_path=relative_path,
                tree=donor_tree,
            )
            if include.get("target_file")
            and not _resolve_include_target(
                root=old_root,
                driver_root=old_root / driver_root,
                source_file=relative_path,
                include_name=str(include.get("name") or ""),
                delimiter=str(include.get("delimiter") or ""),
            )
        }
    )
    if missing_dependencies:
        base["reason"] = "donor include has unresolved target dependencies"
        base["missing_direct_dependencies"] = missing_dependencies
        return base
    base.update(
        {
            "status": "proven",
            "diagnostic_node_ids": sorted(diagnostic_ids),
            "include_node_ids": sorted(include_ids),
            "unresolved_include_node_ids": sorted(
                exact_unresolved_ids
            ),
            "consumer_file_node_ids": sorted(consumer_ids),
            "reason": (
                "a fresh diagnostic selects one unresolved current include "
                "relation and exactly one donor file"
            ),
        }
    )
    return base


def repair_owner_patch_files(
    repair_owners: list[dict[str, Any]],
) -> list[str]:
    return ordered_unique_relpaths(
        [
            path
            for owner in repair_owners
            if isinstance(owner, dict)
            and owner.get("allowed_as_patch_target")
            for path in owner.get("allowed_patch_files") or []
        ]
    )


def repair_owner_localization(
    repair_owners: list[dict[str, Any]],
) -> dict[str, Any]:
    targets: list[dict] = []
    allowed_fileset = repair_owner_patch_files(repair_owners)
    for owner in repair_owners:
        allowed_patch_files = ordered_unique_relpaths(
            list(owner.get("allowed_patch_files") or [])
        )
        targets.append(
            {
                "kind": owner.get("owner_kind"),
                "file": normalize_relpath(str(owner.get("owner_file") or "")),
                "symbol": owner.get("owner_symbol"),
                "allowed": bool(owner.get("allowed_as_patch_target")),
                "allowed_patch_files": allowed_patch_files,
                "reason": owner.get("reason"),
            }
        )

    if allowed_fileset:
        status = "patch_target_ready"
    elif repair_owners:
        status = "context_expansion_required"
    else:
        status = "no_patch_target"
    return {
        "schema": "work_unit_patch_target_localization_v1",
        "source": "repair_owner_localization",
        "status": status,
        "allowed_fileset": allowed_fileset,
        "target_count": len(targets),
        "targets": targets,
    }


def exact_entity_materialization_owner_ids(
    work_unit: dict[str, Any],
) -> list[str]:
    """Return exact donor definitions authorized for entity materialization."""

    unit_files = set(
        ordered_unique_relpaths(
            work_unit.get("entity_materialization_files") or []
        )
    )
    owner_ids: set[str] = set()
    for owner in work_unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        owner_id = str(owner.get("owner_entity_id") or "")
        owner_file = normalize_relpath(
            str(owner.get("owner_file") or "")
        )
        owner_files = set(
            ordered_unique_relpaths(
                owner.get("entity_materialization_files") or []
            )
        )
        if (
            owner_id
            and owner_file
            and owner.get("allowed_for_entity_materialization") is True
            and owner.get("owner_is_definition") is True
            and str(owner.get("owner_entity_tree") or "").startswith(
                "donor_"
            )
            and owner_file in owner_files
            and owner_file in unit_files
        ):
            owner_ids.add(owner_id)
    return sorted(owner_ids)


def exact_graph_scope_owners(
    work_unit: dict[str, Any],
    *,
    root_node_id: str | None = None,
) -> list[dict[str, Any]]:
    """Project exact materializable owners from the work unit's graph evidence."""

    root = dict(
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    expected_root_id = str(
        root_node_id
        or (work_unit.get("repair_decision") or {}).get("root_node_id")
        or root.get("node_id")
        or root.get("entity_id")
        or ""
    )
    scope_files = set(
        ordered_unique_relpaths(
            [
                *list(work_unit.get("scope_expansion_candidates") or []),
                *list(work_unit.get("materialization_candidates") or []),
                *list(work_unit.get("escalation_candidates") or []),
            ]
        )
    )
    missing_materialization_files = set(
        ordered_unique_relpaths(
            work_unit.get("materialization_candidates") or []
        )
    )
    if not expected_root_id or not scope_files:
        return []

    candidates: list[dict[str, Any]] = []
    for raw_owner in (work_unit.get("code_graph_scope") or {}).get(
        "owner_candidates"
    ) or []:
        if not isinstance(raw_owner, dict):
            continue
        candidates.append(dict(raw_owner))

    for owner in work_unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        structure = dict(owner.get("diagnostic_structure") or {})
        owner_file = normalize_relpath(str(owner.get("owner_file") or ""))
        owner_node_id = str(
            owner.get("owner_entity_id")
            or structure.get("owner_node_id")
            or structure.get("root_entity_id")
            or ""
        )
        owner_kind = str(owner.get("owner_kind") or "")
        canonical_owner_role = {
            "code_index_root_entity": "root_owner",
            "code_index_diagnostic_ast_owner": (
                "diagnostic_location_owner"
            ),
            "code_index_diagnostic_file": (
                "diagnostic_file_owner"
            ),
            "code_index_consumer_integration_owner": (
                "consumer_integration_owner"
            ),
            "code_index_include_relation": (
                "include_relation_owner"
            ),
            "code_index_provider_definition": (
                "provider_definition_owner"
            ),
            "code_index_provider_declaration": (
                "provider_declaration_owner"
            ),
            "code_index_repair_dependency": (
                "repair_dependency_owner"
            ),
        }.get(owner_kind, "")
        owner_role = canonical_owner_role or str(
            owner.get("owner_role") or ""
        )
        permission = str(
            owner.get("owner_permission")
            or (owner.get("allowed_patch_permissions") or {}).get(owner_file)
            or ""
        )
        if (
            not structure.get("enabled")
            or not owner_node_id
            or (
                owner.get("old_exists") is False
                and owner_file not in missing_materialization_files
            )
        ):
            continue
        candidates.append(
            {
                "file": owner_file,
                "node_id": owner_node_id,
                "root_node_id": expected_root_id,
                "permission": permission,
                "old_exists": bool(owner.get("old_exists")),
                "reasons": [str(owner.get("reason") or "")],
                "owner_id": owner.get("id"),
                "owner_kind": owner_kind,
                "owner_role": owner_role or None,
                "owner_entity_tree": owner.get("owner_entity_tree"),
                "old_corresponding_entity_ids": list(
                    owner.get("old_corresponding_entity_ids") or []
                ),
                "allowed_for_entity_materialization": bool(
                    owner.get("allowed_for_entity_materialization")
                ),
                "entity_materialization_files": list(
                    owner.get("entity_materialization_files") or []
                ),
                "driver_interface_ownership": dict(
                    owner.get("driver_interface_ownership") or {}
                ),
                "exact_missing_include_materialization": dict(
                    owner.get(
                        "exact_missing_include_materialization"
                    )
                    or {}
                ),
                "exact_absent_declaration_insertion_allowed": bool(
                    owner.get(
                        "exact_absent_declaration_insertion_allowed"
                    )
                ),
            }
        )

    exact: dict[tuple[str, str], dict[str, Any]] = {}
    for owner in candidates:
        relative_path = normalize_relpath(str(owner.get("file") or ""))
        node_id = str(owner.get("node_id") or "")
        if (
            not relative_path
            or relative_path not in scope_files
            or not node_id
            or str(owner.get("root_node_id") or "") != expected_root_id
            or (
                str(owner.get("owner_role") or "") == "root_owner"
                and node_id != expected_root_id
            )
            or str(owner.get("permission") or "")
            not in {"editable", "escalation"}
        ):
            continue
        normalized = dict(owner)
        normalized.update(
            {
                "file": relative_path,
                "node_id": node_id,
                "root_node_id": expected_root_id,
                "reasons": [
                    reason
                    for reason in dict.fromkeys(
                        str(item or "") for item in owner.get("reasons") or []
                    )
                    if reason
                ],
            }
        )
        if (
            str(normalized.get("owner_role") or "")
            == "diagnostic_location_owner"
            and node_id != expected_root_id
        ):
            continue
        if (
            unsafe_automatic_scope_owner(normalized)
            and not exact_absent_declaration_insertion_owner(normalized)
        ):
            continue
        _record_owner_candidate(exact, normalized)
    return [exact[key] for key in sorted(exact)]


def _code_index_repair_owners(
    spec: object | None,
    root: dict,
    diagnostic_graph: dict,
    *,
    ownership_index: CodeIndex | None = None,
) -> list[dict]:
    if spec is None:
        return []
    nodes_by_id = {
        str(node.get("id") or ""): node
        for node in diagnostic_graph.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    old_counterparts_by_node: dict[str, set[str]] = defaultdict(set)
    for edge in diagnostic_graph.get("edges") or []:
        if not isinstance(edge, dict) or str(edge.get("type") or "") != "same_contract_as":
            continue
        source_id = str(edge.get("from") or "")
        target_id = str(edge.get("to") or "")
        source_tree = str((nodes_by_id.get(source_id) or {}).get("tree") or "")
        target_tree = str((nodes_by_id.get(target_id) or {}).get("tree") or "")
        if source_tree.startswith("donor_") and target_tree == "old":
            old_counterparts_by_node[source_id].add(target_id)
        elif target_tree.startswith("donor_") and source_tree == "old":
            old_counterparts_by_node[target_id].add(source_id)
    candidates: dict[tuple[str, str], tuple[dict, str, str]] = {}

    def add_candidate(node_id: str, *, owner_kind: str, reason: str) -> None:
        node = nodes_by_id.get(node_id) or {}
        owner_file = normalize_relpath(str(node.get("file") or node.get("path") or ""))
        if not node_id or not owner_file:
            return
        previous = candidates.get((node_id, owner_file))
        if (
            previous
            and previous[1] == "code_index_consumer_integration_owner"
            and owner_kind in {"code_index_diagnostic_ast_owner", "code_index_diagnostic_file"}
        ):
            return
        candidates[(node_id, owner_file)] = (node, owner_kind, reason)

    root_node_ids = mapped_root_alternative_ids(root)
    if not root_node_ids:
        root_node_id = str(
            root.get("node_id") or root.get("entity_id") or ""
        )
        root_node_ids = [root_node_id] if root_node_id else []
    for root_node_id in root_node_ids:
        add_candidate(
            root_node_id,
            owner_kind="code_index_root_entity",
            reason=(
                "selected common-upstream root has an exact source owner"
            ),
        )
    diagnostic_node_ids = {
        node_id
        for node_id, node in nodes_by_id.items()
        if str(node.get("kind") or "") == "Diagnostic"
    }
    diagnosed_reference_ids = {
        str(edge.get("to") or "")
        for edge in diagnostic_graph.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("type") or "") == "diagnoses"
        and str(edge.get("from") or "") in diagnostic_node_ids
        and str(
            (
                nodes_by_id.get(str(edge.get("to") or ""))
                or {}
            ).get("kind")
            or ""
        )
        == "Reference"
    }
    for edge in diagnostic_graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source_id = str(edge.get("from") or "")
        target_id = str(edge.get("to") or "")
        edge_type = str(edge.get("type") or "")
        source_kind = str((nodes_by_id.get(source_id) or {}).get("kind") or "")
        target_node = nodes_by_id.get(target_id) or {}
        target_tree = str(target_node.get("tree") or "")
        target_kind = str(
            target_node.get("entity_kind")
            or target_node.get("kind")
            or ""
        ).lower()
        if (
            edge_type == "callback_contract_consumer"
            and source_id in root_node_ids
            and target_tree == "old"
            and target_node.get("extent")
        ):
            add_candidate(
                target_id,
                owner_kind="code_index_consumer_integration_owner",
                reason="exact current binding or direct call consumes the diagnosed callback contract",
            )
        elif (
            edge_type == "occurs_in"
            and source_kind == "Reference"
            and source_id in diagnosed_reference_ids
            and target_tree == "old"
        ):
            add_candidate(
                target_id,
                owner_kind="code_index_diagnostic_ast_owner",
                reason="diagnostic reference occurs inside this exact old-tree AST owner",
            )
        elif (
            edge_type == "diagnoses"
            and source_kind == "Diagnostic"
            and target_tree == "old"
            and target_kind not in {
                "diagnostic",
                "file",
                "reference",
                "unresolvedcontract",
                "unresolvedentity",
            }
        ):
            add_candidate(
                target_id,
                owner_kind="code_index_diagnostic_ast_owner",
                reason="diagnostic location maps directly to this old-tree AST owner",
            )
        elif edge_type == "occurs_at" and source_kind == "Diagnostic" and target_tree == "old":
            add_candidate(
                target_id,
                owner_kind="code_index_diagnostic_file",
                reason="compiler diagnostic occurs in this exact old-tree source file",
            )

    owners: list[dict] = []
    root_node_id_set = {
        str(node_id) for node_id in root_node_ids if str(node_id)
    }
    for (owner_node_id, owner_file), (node, owner_kind, reason) in sorted(candidates.items()):
        try:
            permission = classify_edit_permission(spec, owner_file)
        except (AttributeError, OSError, TypeError, ValueError):
            try:
                permission = str(spec.classify_edit_root(owner_file))
            except (AttributeError, OSError, TypeError, ValueError):
                permission = "forbidden"
        try:
            old_exists = bool(spec.resolve_old(owner_file).is_file())
        except (AttributeError, OSError, TypeError, ValueError):
            old_exists = False
        old_corresponding_entity_ids = sorted(
            old_counterparts_by_node.get(owner_node_id) or []
        )
        owner_tree = str(node.get("tree") or "")
        interface_ownership = (
            _driver_interface_header_ownership(
                spec,
                ownership_index,
                donor_tree=owner_tree,
                header_file=owner_file,
            )
            if (
                str(node.get("kind") or "") == "File"
                and owner_tree.startswith("donor_")
                and not old_exists
            )
            else {}
        )
        missing_include_evidence = (
            _exact_missing_include_materialization_evidence(
                diagnostic_graph,
                owner_node_id,
                spec,
            )
            if (
                owner_kind == "code_index_root_entity"
                and str(node.get("kind") or "") == "File"
                and owner_tree.startswith("donor_")
                and not old_exists
            )
            else {}
        )
        driver_interface_owned = _proven_driver_interface_ownership(
            {"driver_interface_ownership": interface_ownership}
        )
        exact_missing_include = bool(
            missing_include_evidence.get("status") == "proven"
        )
        owner_record = {
            "id": stable_id("owner", owner_node_id, owner_file),
            "owner_kind": owner_kind,
            "owner_file": owner_file,
            "owner_symbol": node.get("symbol") or node.get("name"),
            "owner_entity_id": owner_node_id,
            "owner_contract_key": (
                node.get("contract_key") or node.get("key")
            ),
            "owner_entity_kind": node.get("kind"),
            "owner_entity_tree": owner_tree or None,
            "owner_role": {
                "code_index_root_entity": "root_owner",
                "code_index_diagnostic_ast_owner": (
                    "diagnostic_location_owner"
                ),
                "code_index_diagnostic_file": (
                    "diagnostic_file_owner"
                ),
                "code_index_consumer_integration_owner": (
                    "consumer_integration_owner"
                ),
            }.get(owner_kind),
            "owner_entity_container": node.get("container"),
            "owner_entity_extent": dict(
                node.get("source_extent") or node.get("extent") or {}
            ),
            "owner_is_definition": bool(node.get("is_definition")),
            "old_corresponding_entity_ids": old_corresponding_entity_ids,
            "donor_only_entity": bool(
                owner_tree.startswith("donor_")
                and not old_corresponding_entity_ids
            ),
            "exact_absent_entity_insertion_allowed": False,
            "allowed_for_entity_materialization": bool(
                (driver_interface_owned or exact_missing_include)
                and permission in {"editable", "escalation"}
            ),
            "entity_materialization_files": (
                [owner_file]
                if driver_interface_owned or exact_missing_include
                else []
            ),
            "driver_interface_ownership": interface_ownership,
            "exact_missing_include_materialization": (
                missing_include_evidence
            ),
            "reason": reason,
            "evidence": [owner_node_id],
            "allowed_patch_permissions": {owner_file: permission},
            "owner_permission": permission,
            "old_exists": old_exists,
            "symptom_files": (
                [owner_file]
                if owner_kind != "code_index_root_entity"
                else []
            ),
            "trigger_files": [owner_file],
            "diagnostic_structure": {
                "enabled": True,
                "backend": "code_index_v3",
                "owner_node_id": owner_node_id,
            },
        }
        unsafe_scope = unsafe_automatic_scope_owner(owner_record)
        causal_owner = bool(
            owner_kind
            in {
                "code_index_root_entity",
                "code_index_consumer_integration_owner",
            }
            or (
                owner_kind == "code_index_diagnostic_ast_owner"
                and owner_node_id in root_node_id_set
            )
        )
        allowed = bool(
            permission == "editable"
            and old_exists
            and not unsafe_scope
            and causal_owner
        )
        owner_record["allowed_as_patch_target"] = allowed
        owner_record["allowed_patch_files"] = [owner_file] if allowed else []
        owner_record["read_only_evidence_files"] = (
            [] if allowed else [owner_file]
        )
        if unsafe_scope:
            owner_record["reason"] = (
                f"{reason}; donor-only shared kernel owner requires explicit "
                "architecture-level evidence"
            )
            owner_record["read_only_evidence_files"] = [owner_file]
        elif driver_interface_owned:
            owner_record["reason"] = (
                f"{reason}; exact active interface declarations prove this "
                "missing header is owned by the designated donor entry files"
            )
            owner_record["read_only_evidence_files"] = []
        elif exact_missing_include:
            owner_record["reason"] = (
                f"{reason}; a fresh typed missing-include path selects only "
                "this absent donor file"
            )
            owner_record["read_only_evidence_files"] = []
        owners.append(owner_record)
    return owners


def derive_work_units(
    validation_report: dict,
    *,
    spec: object | None = None,
    semantic_state: dict[str, Any] | None = None,
    semantic_contract_evidence_by_ref: (
        dict[str, dict[str, Any]] | None
    ) = None,
) -> list[dict]:
    """Derive work units from common upstream nodes in one typed dependency graph.

    CodeIndex owns parsing and work-unit semantics do not depend on pre-graph
    diagnostic categories.
    """

    build_prep_gate = build_prep_stage_decision(
        validation_report,
        validation_status=(
            "ready_for_manual_test"
            if validation_report.get("overall_ok")
            else "needs_patching"
        ),
    )
    if build_prep_gate.get("needs_build_prep"):
        return []

    routed_issues = [
        dict(issue)
        for issue in repair_routed_diagnostics(validation_report)
    ]
    if not routed_issues:
        return []
    repair_issues = [diagnostic_event_record(issue) for issue in routed_issues]

    code_index = CodeIndex(spec)
    components = (
        code_index.explanation_components(
            repair_issues,
            semantic_state=semantic_state,
            semantic_contract_evidence_by_ref=(
                semantic_contract_evidence_by_ref
            ),
        )
        if semantic_state is not None
        else code_index.explanation_components(repair_issues)
    )
    work_units: list[dict] = []
    needs_interface_ownership = any(
        str((component.get("root") or {}).get("tree") or "").startswith(
            "donor_"
        )
        and str((component.get("root") or {}).get("entity_kind") or "")
        == "File"
        and Path(
            str((component.get("root") or {}).get("file") or "")
        ).suffix.lower()
        in {".h", ".hpp"}
        for component in components
        if isinstance(component, dict)
    )
    ownership_index = (
        CodeIndex(spec, graph_projection="records")
        if spec is not None and needs_interface_ownership
        else None
    )
    for component in components:
        issues = list(component.get("issues") or [])
        root = dict(component.get("root") or {})
        title = str(root.get("title") or "unmapped_diagnostic")
        scope = str(root.get("scope") or stable_id("unmapped", repr(issues)))
        root_key = str(
            graph_root_binding_key(root)
            or stable_id("root", title, scope)
        )
        root["title"] = title
        root["scope"] = scope
        root["key"] = root_key

        diagnostic_graph = code_index.diagnostic_graph_dict(issues, selected_root=root)
        route_summary = diagnostic_route_summary(issues)
        route_summary.pop("issue_kinds", None)
        repair_owners = _code_index_repair_owners(
            spec,
            root,
            diagnostic_graph,
            ownership_index=ownership_index,
        )
        compact_repair_owners = [_compact_repair_owner(item) for item in repair_owners]
        localization = repair_owner_localization(compact_repair_owners)
        patch_targets = list(localization.get("targets") or [])

        diagnostic_signatures = sorted(
            {
                str(issue.get("diagnostic_signature") or diagnostic_signature(issue))
                for issue in issues
            }
        )
        work_unit_identity = [root_key]
        root_alternatives = [
            item for item in component.get("root_alternatives") or [] if isinstance(item, dict)
        ]
        binding_node_ids = {
            str(root.get("node_id") or root.get("entity_id") or ""),
            *[
                str(item.get("node_id") or item.get("entity_id") or "")
                for item in root_alternatives
            ],
            *[
                str(item)
                for item in component.get("common_root_node_ids") or []
            ],
        }
        semantic_binding = code_index.semantic_contract_binding(
            item for item in binding_node_ids if item
        )
        mapped_root_set = bool(
            root.get("ambiguous")
            and root_alternatives
            and all(item.get("node_id") or item.get("entity_id") for item in root_alternatives)
        )
        if root.get("unmapped") or (
            root.get("ambiguous") and not mapped_root_set
        ):
            if root.get("node_id") or root.get("entity_id"):
                work_unit_identity.extend(diagnostic_signatures)
            else:
                work_unit_identity.extend(
                    sorted(component.get("diagnostic_event_ids") or [])
                )
        work_units.append(
            {
                "schema": ROOT_CAUSE_WORK_UNIT_SCHEMA,
                "id": stable_id("wu", *work_unit_identity),
                "title": title,
                "scope": scope,
                "status": validate_work_unit_state("created"),
                "generated_at": utc_now(),
                "updated_at": utc_now(),
                "retry_count": 0,
                "diagnostic_signature_count": len(diagnostic_signatures),
                "diagnostic_signatures": diagnostic_signatures,
                "diagnostic_routes": route_summary,
                "diagnostic_graph": diagnostic_graph,
                "root_cause_cluster": root,
                "root_cause_alternatives": root_alternatives,
                "diagnostic_partition_reason": component.get("partition_reason"),
                "overlapping_root_node_ids": list(component.get("overlapping_root_node_ids") or []),
                "diagnostic_mappings": list(component.get("mapping_statuses") or []),
                "semantic_gap_ids": list(
                    semantic_binding.get("semantic_gap_ids") or []
                ),
                "semantic_contract_binding": semantic_binding,
                "repair_owner_count": len(compact_repair_owners),
                "repair_owners": compact_repair_owners,
                "localization": localization,
                "patch_targets": patch_targets,
                "patch_target": patch_targets[0] if len(patch_targets) == 1 else {},
                "decision": {
                    "mode": "expand_graph" if root.get("unmapped") else "undecided",
                    "reason": (
                        "dependency graph still contains an unmapped diagnostic"
                        if root.get("unmapped")
                        else (
                            "mapped minimal upstream root set; pending repair planning"
                            if root.get("ambiguous")
                            else "root entity mapped; pending repair planning"
                        )
                    ),
                },
                "history": [
                    {
                        "at": utc_now(),
                        "state": "created",
                        "reason": "derived from CodeIndex common-upstream graph",
                    }
                ],
                "validation": {"required": ["compile", "modpost"]},
                "source_issues": issues,
            }
        )
    root_nodes_by_unit: dict[str, set[str]] = {}
    for unit in work_units:
        root = dict(unit.get("root_cause_cluster") or {})
        root_nodes = {
            str(root.get("node_id") or root.get("entity_id") or "")
        }
        root_nodes.update(
            str(item.get("node_id") or item.get("entity_id") or "")
            for item in unit.get("root_cause_alternatives") or []
            if isinstance(item, dict)
        )
        root_nodes_by_unit[str(unit["id"])] = {
            item for item in root_nodes if item
        }

    for unit in work_units:
        unit_id = str(unit["id"])
        roots = root_nodes_by_unit[unit_id]
        upstream = code_index.upstream_dependency_nodes(roots)
        dependencies = [
            candidate_id
            for candidate_id, candidate_roots in root_nodes_by_unit.items()
            if candidate_id != unit_id
            and bool(candidate_roots & (upstream - roots))
        ]
        unit["dependency_root_node_ids"] = sorted(upstream - roots)
        unit["depends_on_work_unit_ids"] = sorted(dependencies)

    return sorted(work_units, key=lambda item: str(item.get("id") or ""))


def write_work_units(
    work_units: list[dict],
    *,
    work_units_dir: Path,
) -> list[Path]:
    return [
        write_work_unit(work_unit, work_units_dir=work_units_dir)
        for work_unit in work_units
    ]
