#!/usr/bin/env python3
from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    normalize_relpath,
    normalize_root_relative,
    ordered_unique_relpaths,
    stable_id,
    utc_now,
)
from driver_backport.core.executor import classify_edit_permission
from driver_backport.analysis.diagnostic_identity import diagnostic_event_record
from driver_backport.planning.context_expansion import (
    _needs_code_graph_scope,
    _root_mapping_requires_expansion,
    _stable_current_tree_root_anchor,
    expand_repair_context,
)
from driver_backport.planning.dependency_schedule import dependency_order
from driver_backport.planning.repair_source_pack import build_repair_source_pack
from driver_backport.planning.root_binding import (
    graph_root_binding_key,
    mapped_root_alternative_ids,
)
from driver_backport.planning.execution_phase import (
    unit_execution_phase,
)
from driver_backport.planning.work_unit import (
    exact_missing_include_materialization_owner,
    exact_graph_scope_owners,
    repair_owner_has_action_authority,
    repair_owner_localization,
    repair_owner_patch_files,
)


def _normalize_old_relative(spec: CaseSpec, value: str | None) -> str | None:
    if not value:
        return None
    return normalize_root_relative(value, root=spec.old_root_path, preferred_prefix=spec.driver_root)


def _normalize_repair_work_unit_paths(spec: CaseSpec, work_unit: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(work_unit)

    def paths(values: list[object]) -> list[str]:
        return ordered_unique_relpaths(
            [
                relative_path
                for value in values
                for relative_path in [_normalize_old_relative(spec, str(value or ""))]
                if relative_path
            ]
        )

    def provider_resolution(value: object) -> object:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        if value.get("preferred_patch_file"):
            result["preferred_patch_file"] = (
                _normalize_old_relative(spec, str(value.get("preferred_patch_file")))
                or value.get("preferred_patch_file")
            )
        for key in ("target_entities", "donor_entities"):
            entities: list[dict[str, Any]] = []
            for entity in value.get(key) or []:
                if not isinstance(entity, dict):
                    continue
                entity_item = dict(entity)
                if entity.get("file"):
                    entity_item["file"] = (
                        _normalize_old_relative(spec, str(entity.get("file")))
                        or entity.get("file")
                    )
                entities.append(entity_item)
            result[key] = entities
        return result

    normalized["provider_resolutions"] = [
        provider_resolution(item)
        for item in work_unit.get("provider_resolutions") or []
        if isinstance(item, dict)
    ]
    normalized_issues: list[dict[str, Any]] = []
    for issue in work_unit.get("source_issues") or []:
        if not isinstance(issue, dict):
            continue
        item = dict(issue)
        if issue.get("file"):
            item["file"] = _normalize_old_relative(spec, str(issue.get("file"))) or issue.get("file")
        normalized_issues.append(diagnostic_event_record(item))
    normalized["source_issues"] = normalized_issues

    normalized_owners: list[dict[str, Any]] = []
    for owner in work_unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        item = dict(owner)
        if owner.get("owner_file"):
            item["owner_file"] = (
                _normalize_old_relative(spec, str(owner.get("owner_file")))
                or owner.get("owner_file")
            )
        for key in (
            "allowed_patch_files",
            "symptom_files",
            "trigger_files",
            "read_only_evidence_files",
        ):
            item[key] = paths(list(owner.get(key) or []))
        normalized_owners.append(item)
    normalized["repair_owners"] = normalized_owners
    normalized_authorities: list[dict[str, Any]] = []
    for authority in (
        work_unit.get("repair_transaction_authorities") or []
    ):
        if not isinstance(authority, dict):
            continue
        item = dict(authority)
        if authority.get("file"):
            item["file"] = (
                _normalize_old_relative(
                    spec, str(authority.get("file"))
                )
                or authority.get("file")
            )
        normalized_authorities.append(item)
    normalized["repair_transaction_authorities"] = (
        normalized_authorities
    )
    return normalized


def _path_record(spec: CaseSpec, relative_path: str) -> dict[str, Any]:
    donor_candidates = []
    for donor_root in spec.donor_root_paths:
        donor_path = (donor_root / relative_path).resolve()
        if donor_path.exists():
            donor_candidates.append(str(donor_path))
    old_path = spec.resolve_old(relative_path)
    return {
        "file": relative_path,
        "permission": classify_edit_permission(spec, relative_path),
        "old_exists": old_path.exists(),
        "old_path": str(old_path),
        "donor_paths": donor_candidates,
        "donor_exists": bool(donor_candidates),
    }


def _ordered_add(target: list[str], seen: set[str], items: list[str], reason_map: dict[str, list[str]], reason: str) -> None:
    for item in items:
        relative_path = normalize_relpath(item)
        if relative_path in seen:
            if reason not in reason_map[relative_path]:
                reason_map[relative_path].append(reason)
            continue
        seen.add(relative_path)
        target.append(relative_path)
        reason_map[relative_path] = [reason]


def _repair_owner_patch_files(work_unit: dict[str, Any] | None) -> set[str]:
    if not work_unit:
        return set()
    return set(
        repair_owner_patch_files(
            list(work_unit.get("repair_owners") or [])
        )
    )


def _repair_owner_block_reason(owner: dict[str, Any], allowed_fileset: list[str]) -> str:
    owner_kind = str(owner.get("owner_kind") or "")
    owner_file = normalize_relpath(str(owner.get("owner_file") or ""))
    allowed_files = {normalize_relpath(str(item)) for item in allowed_fileset if str(item).strip()}
    owner_patch_files = set(
        repair_owner_patch_files([owner])
    )
    if owner_kind == "unlocalized" or not (owner_file or owner_patch_files):
        return "owner_unlocalized"
    if not bool(owner.get("allowed_as_patch_target")):
        return "owner_not_allowed_as_patch_target"
    if not owner_patch_files:
        return "owner_has_no_allowed_patch_files"
    if allowed_files and not (owner_patch_files & allowed_files):
        return "owner_file_not_in_allowed_fileset"
    return ""


def _graph_continuation_request(
    work_unit: dict[str, Any],
    *,
    reason: str,
    blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = dict(
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    scope = work_unit.get("code_graph_scope") if isinstance(work_unit.get("code_graph_scope"), dict) else {}
    continuation = scope.get("continuation") if isinstance(scope.get("continuation"), dict) else {}
    event_ids = [
        str(issue.get("diagnostic_event_id") or "")
        for issue in work_unit.get("source_issues") or []
        if isinstance(issue, dict) and issue.get("diagnostic_event_id")
    ]
    return {
        "schema": "code_graph_continuation_request_v2",
        "reason": reason,
        "diagnostic_event_ids": list(dict.fromkeys(event_ids)),
        "root_node_id": root.get("node_id") or root.get("entity_id"),
        "unresolved_frontier_node_ids": list(
            continuation.get("unresolved_frontier_node_ids") or []
        ),
        "cursor_by_tree": dict(continuation.get("cursor_by_tree") or {}),
        "cursor_by_tree_symbol": dict(
            continuation.get("cursor_by_tree_symbol") or {}
        ),
        "blockers": list(blockers or []),
    }


def _repair_localization_gate(work_unit: dict[str, Any], allowed_fileset: list[str]) -> dict[str, Any]:
    all_repair_owners = [
        owner
        for owner in work_unit.get("repair_owners") or []
        if isinstance(owner, dict)
    ]
    repair_owners = [
        owner
        for owner in all_repair_owners
        if repair_owner_has_action_authority(owner)
    ]
    allowed_file_set = {normalize_relpath(str(item)) for item in allowed_fileset if str(item).strip()}

    def ready_allowed_files(owners: list[dict[str, Any]]) -> list[str]:
        files = repair_owner_patch_files(owners)
        if not allowed_file_set:
            return files
        return [path for path in files if path in allowed_file_set]

    if not repair_owners:
        blockers = [
            {
                "reason": "no_repair_action_owner_evidence",
                "localization_reason": (
                    "CodeIndex must map the diagnostic to a causal root, "
                    "provider, declaration, integration, or include-relation "
                    "owner before patching; diagnostic locations remain "
                    "read-only evidence"
                ),
            }
        ]
        return {
            "status": "no_repair_action_owner_evidence",
            "patch_allowed": False,
            "requires_context_expansion": True,
            "ready_owner_count": 0,
            "blocked_owner_count": 0,
            "blockers": blockers,
            "context_expansion_request": _graph_continuation_request(
                work_unit,
                reason="no CodeIndex repair owner is mapped yet",
                blockers=blockers,
            ),
        }

    ready: list[dict[str, Any]] = []
    materialization_ready: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for owner in repair_owners:
        if owner.get("allowed_for_entity_materialization"):
            materialization_ready.append(owner)
            continue
        reason = _repair_owner_block_reason(owner, allowed_fileset)
        if reason:
            blockers.append(
                {
                    "owner_id": owner.get("id"),
                    "owner_kind": owner.get("owner_kind"),
                    "owner_file": owner.get("owner_file"),
                    "owner_symbol": owner.get("owner_symbol"),
                    "reason": reason,
                    "localization_reason": owner.get("reason"),
                }
            )
        else:
            ready.append(owner)

    materialization_files = ordered_unique_relpaths(
        [
            str(path)
            for owner in materialization_ready
            for path in (
                owner.get("entity_materialization_files")
                or [owner.get("owner_file")]
            )
            if str(path or "")
        ]
    )
    if not blockers and ready:
        patch_allowed_files = ready_allowed_files(ready) or list(allowed_fileset)
        return {
            "status": "patch_owner_ready",
            "patch_allowed": True,
            "requires_context_expansion": False,
            "ready_owner_count": len(ready),
            "blocked_owner_count": 0,
            "allowed_fileset": patch_allowed_files,
            "blockers": [],
            "context_expansion_request": {},
            "entity_materialization_files": materialization_files,
        }
    if not blockers and materialization_ready:
        return {
            "status": "entity_materialization_ready",
            "patch_allowed": False,
            "requires_context_expansion": False,
            "ready_owner_count": 0,
            "materialization_owner_count": len(
                materialization_ready
            ),
            "blocked_owner_count": 0,
            "allowed_fileset": [],
            "entity_materialization_files": materialization_files,
            "blockers": [],
            "context_expansion_request": {},
        }

    patch_allowed_files = ready_allowed_files(ready)
    if patch_allowed_files:
        return {
            "status": "partial_context_expansion_required",
            "patch_allowed": True,
            "requires_context_expansion": False,
            "ready_owner_count": len(ready),
            "blocked_owner_count": len(blockers),
            "allowed_fileset": patch_allowed_files,
            "blockers": blockers,
            "context_expansion_request": _graph_continuation_request(
                work_unit,
                reason=(
                    "some repair owners are patch-ready while other owners still require context expansion; "
                    "patch generation is allowed only for the ready owner files in allowed_fileset"
                ),
                blockers=blockers,
            ),
            "entity_materialization_files": materialization_files,
        }
    if materialization_ready:
        return {
            "status": "entity_materialization_ready_with_read_only_blockers",
            "patch_allowed": False,
            "requires_context_expansion": False,
            "ready_owner_count": 0,
            "materialization_owner_count": len(
                materialization_ready
            ),
            "blocked_owner_count": len(blockers),
            "allowed_fileset": [],
            "entity_materialization_files": materialization_files,
            "blockers": blockers,
            "context_expansion_request": {},
        }

    return {
        "status": (
            "partial_context_expansion_required" if ready else "context_expansion_required"
        ),
        "patch_allowed": False,
        "requires_context_expansion": True,
        "ready_owner_count": len(ready),
        "blocked_owner_count": len(blockers),
        "blockers": blockers,
        "context_expansion_request": _graph_continuation_request(
            work_unit,
            reason=(
                "one or more repair owners are unlocalized or not allowed as patch targets; "
                "do not generate a patch until CodeIndex reaches a permitted root entity"
            ),
            blockers=blockers,
        ),
    }


def _pinned_repair_files(
    candidate_files: list[str],
    work_unit: dict[str, Any] | None = None,
) -> list[str]:
    """Pin only files reached as permitted CodeIndex owners."""

    owner_files = _repair_owner_patch_files(work_unit)
    if not owner_files:
        return []
    return [
        normalized
        for relative_path in candidate_files
        for normalized in [normalize_relpath(relative_path)]
        if normalized in owner_files
    ]


def _retrieve_repair_candidates(
    spec: CaseSpec,
    work_unit: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Collect only files already justified by diagnostic or CodeIndex edges."""

    work_unit = _normalize_repair_work_unit_paths(spec, work_unit)
    candidates: list[str] = []
    seen: set[str] = set()
    reason_map: dict[str, list[str]] = {}

    owner_files = sorted(_repair_owner_patch_files(work_unit))
    _ordered_add(candidates, seen, owner_files, reason_map, "code_index_root_owner")
    entity_materialization_files = ordered_unique_relpaths(
        [
            str(item)
            for item in (
                work_unit.get("entity_materialization_files")
                or []
            )
            if str(item)
        ]
    )
    _ordered_add(
        candidates,
        seen,
        entity_materialization_files,
        reason_map,
        "reachable_donor_entity_owner",
    )

    graph_scope = work_unit.get("code_graph_scope") or {}
    graph_candidates = [
        normalize_relpath(str(value))
        for value in graph_scope.get("discovered_candidate_files") or []
        if str(value).strip()
    ]
    _ordered_add(candidates, seen, graph_candidates, reason_map, "code_index_scope_path")

    roots = [work_unit.get("root_cause_cluster") or {}]
    roots.extend(work_unit.get("root_cause_alternatives") or [])
    for root in roots:
        if not isinstance(root, dict):
            continue
        root_file = _normalize_old_relative(spec, root.get("file"))
        if root_file:
            _ordered_add(candidates, seen, [root_file], reason_map, "code_index_root_file")

    return candidates, {
        "schema": "graph_repair_candidate_retrieval_v3",
        "root_owner_files": owner_files,
        "graph_candidate_files": graph_candidates,
        "entry_seed_files": [],
        "reason_map": reason_map,
    }



def _deferred_repair_source_pack(
    *,
    allowed_fileset: list[str],
    candidate_files: list[str],
    read_only_evidence_files: list[str],
) -> dict[str, Any]:
    return {
        "schema": "repair_source_pack_deferred_v2",
        "deferred": True,
        "reason": (
            "inactive repair unit; build the full source pack only after "
            "active work-unit selection"
        ),
        "summary": {
            "allowed_edit_file_count": len(allowed_fileset),
            "candidate_file_count": len(candidate_files),
            "read_only_evidence_file_count": len(read_only_evidence_files),
            "allowed_edit_source_count": 0,
            "donor_reference_source_count": 0,
            "read_only_evidence_source_count": 0,
            "materialization": "deferred",
        },
        "allowed_edit_files": list(allowed_fileset),
        "read_only_evidence_files": list(read_only_evidence_files),
        "allowed_edit_sources": [],
        "donor_reference_sources": [],
        "read_only_evidence_sources": [],
        "localization_blockers": [],
    }


def _allowed_fileset_after_localization(
    initial_allowed_fileset: list[str],
    localization_gate: dict[str, Any],
) -> list[str]:
    gate_allowed_fileset = ordered_unique_relpaths(
        [
            str(item)
            for item in localization_gate.get("allowed_fileset") or []
            if str(item).strip()
        ]
    )
    if not localization_gate.get("requires_context_expansion"):
        return gate_allowed_fileset or initial_allowed_fileset
    return []


def _exact_consumer_adaptation_authority(
    work_unit: dict[str, Any],
    allowed_fileset: list[str],
) -> dict[str, Any]:
    """Return graph-proven consumers at an exhausted provider boundary."""

    root = dict(
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    expected_root_ids = {
        str(node_id)
        for node_id in (
            mapped_root_alternative_ids(root)
            or [root.get("node_id") or root.get("entity_id")]
        )
        if str(node_id or "")
    }
    allowed = {
        normalize_relpath(str(path))
        for path in allowed_fileset
        if str(path).strip()
    }
    authorities: list[dict[str, Any]] = []
    expected_owner_ids: set[str] = set()
    authorized_owners: dict[str, dict[str, Any]] = {}
    files: set[str] = set()
    for owner in work_unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        authority = dict(owner.get("consumer_adaptation_authority") or {})
        if not authority:
            continue
        owner_file = normalize_relpath(str(owner.get("owner_file") or ""))
        owner_id = str(owner.get("owner_entity_id") or "")
        authority_root_ids = {
            str(node_id)
            for node_id in authority.get("root_node_ids") or []
            if str(node_id)
        }
        authority_owner_ids = {
            str(node_id)
            for node_id in authority.get("consumer_owner_node_ids") or []
            if str(node_id)
        }
        if (
            authority.get("status") != "authorized"
            or authority.get("repair_authority")
            != "exhausted_provider_query_consumer"
            or authority.get("provider_resolution_complete") is not False
            or authority.get("bounded_provider_query_complete") is not True
            or not expected_root_ids
            or authority_root_ids != expected_root_ids
            or not authority_owner_ids
            or str(owner.get("owner_role") or "")
            != "consumer_integration_owner"
            or not owner_id
            or owner_id not in authority_owner_ids
        ):
            return {}
        authorities.append(authority)
        expected_owner_ids.update(authority_owner_ids)
        authorized_owners[owner_id] = owner

    if (
        not authorities
        or set(authorized_owners) != expected_owner_ids
    ):
        return {}
    for owner in authorized_owners.values():
        owner_file = normalize_relpath(str(owner.get("owner_file") or ""))
        if (
            owner.get("allowed_as_patch_target") is not True
            or not owner_file
            or owner_file not in allowed
        ):
            return {}
        files.add(owner_file)
    root_ids = {
        str(node_id)
        for authority in authorities
        for node_id in authority.get("root_node_ids") or []
        if str(node_id)
    }
    provider_symbols = {
        str(symbol)
        for authority in authorities
        for symbol in authority.get("provider_symbols") or []
        if str(symbol)
    }
    return {
        "schema": "exhausted_provider_consumer_authority_v1",
        "status": "authorized",
        "repair_authority": "exhausted_provider_query_consumer",
        "provider_resolution_complete": False,
        "bounded_provider_query_complete": True,
        "root_node_ids": sorted(root_ids),
        "consumer_owner_node_ids": sorted(expected_owner_ids),
        "provider_symbols": sorted(provider_symbols),
        "candidate_write_set": sorted(files),
    }


def exact_active_patch_postimage_boundary(
    work_unit: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    """Check the retained transaction proof behind a local patch decision."""

    if (
        decision.get("repair_authority")
        != "active_patch_transaction_postimage"
        or decision.get("mode") != "patch_root_subgraph"
        or decision.get("root_mapping_complete") is not False
    ):
        return False
    patch_ids = {
        str(item)
        for item in work_unit.get("incident_patch_ids") or []
        if str(item)
    }
    write_files = {
        normalize_relpath(str(item))
        for item in decision.get("candidate_write_set") or []
        if str(item)
    }
    return bool(patch_ids and write_files) and any(
        isinstance(item, dict)
        and item.get("schema")
        == "active_patch_transaction_postimage_authority_v1"
        and item.get("status") == "verified"
        and item.get("authority") == "active_patch_transaction_postimage"
        and item.get("evidence_authority")
        == "fresh_diagnostic_exact_retained_patch_postimage"
        and item.get("causal_relation")
        == "fresh_diagnostic_in_retained_patch_postimage"
        and normalize_relpath(str(item.get("file") or "")) in write_files
        and bool(
            patch_ids
            & {
                str(value)
                for value in item.get("patch_ids") or []
                if str(value)
            }
        )
        for item in work_unit.get("repair_transaction_authorities") or []
    )


def _graph_repair_decision(
    work_unit: dict[str, Any],
    *,
    allowed_fileset: list[str],
    requires_context_expansion: bool,
    requires_escalation: bool,
    escalation_candidates: list[str],
    scope_expansion_candidates: list[str] | None = None,
    entity_materialization_files: list[str] | None = None,
    missing_materialization_files: list[str] | None = None,
) -> dict[str, Any]:
    """Choose the next tool action from graph state and permissions only."""

    root = dict(
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    scope_continuation = dict((work_unit.get("code_graph_scope") or {}).get("continuation") or {})
    root_alternatives = [
        item
        for item in root.get("alternatives") or work_unit.get("root_cause_alternatives") or []
        if isinstance(item, dict)
    ]
    mapped_root_set = bool(mapped_root_alternative_ids(root))
    root_key = graph_root_binding_key(root)
    allowed_file_set = {
        normalize_relpath(str(item))
        for item in allowed_fileset
        if str(item).strip()
    }
    exact_relation_owners = [
        owner
        for owner in work_unit.get("repair_owners") or []
        if isinstance(owner, dict)
        and owner.get("owner_kind") == "code_index_include_relation"
        and owner.get("owner_role") == "include_relation_owner"
        and owner.get("allowed_as_patch_target") is True
        and (
            set(repair_owner_patch_files([owner]))
            & allowed_file_set
        )
    ]
    exact_relation_authority = bool(exact_relation_owners)
    unresolved_include_relation_authority = bool(
        str(root.get("entity_kind") or "") == "UnresolvedInclude"
        and len(exact_relation_owners) == 1
        and str(exact_relation_owners[0].get("include_relation_mode") or "")
        == "mutate_unresolved"
        and str(exact_relation_owners[0].get("current_include_name") or "")
        and int(exact_relation_owners[0].get("expected_line") or 0) > 0
    )
    unresolved_include_write_set = (
        [normalize_relpath(str(exact_relation_owners[0].get("owner_file") or ""))]
        if unresolved_include_relation_authority
        else []
    )
    exact_missing_include_owners = [
        owner
        for owner in work_unit.get("repair_owners") or []
        if isinstance(owner, dict)
        and exact_missing_include_materialization_owner(owner)
    ]
    exact_missing_include_authority = bool(
        exact_missing_include_owners
    )
    active_transaction_authorities = [
        item
        for item in work_unit.get("repair_transaction_authorities") or []
        if isinstance(item, dict)
        and item.get("schema")
        == "active_patch_transaction_postimage_authority_v1"
        and item.get("status") == "verified"
        and item.get("authority")
        == "active_patch_transaction_postimage"
        and item.get("evidence_authority")
        == "fresh_diagnostic_exact_retained_patch_postimage"
        and item.get("causal_relation")
        == "fresh_diagnostic_in_retained_patch_postimage"
        and normalize_relpath(str(item.get("file") or ""))
        in allowed_file_set
        and int(item.get("start_line") or 0) > 0
        and int(item.get("end_line") or 0)
        >= int(item.get("start_line") or 0)
        and bool(item.get("patch_ids"))
        and bool(
            set(str(value) for value in item.get("patch_ids") or [])
            & set(
                str(value)
                for value in work_unit.get("incident_patch_ids") or []
            )
        )
    ]
    exact_transaction_authority = bool(active_transaction_authorities)
    transaction_write_set = sorted(
        {
            normalize_relpath(str(item.get("file") or ""))
            for item in active_transaction_authorities
        }
    )
    consumer_adaptation_authority = _exact_consumer_adaptation_authority(
        work_unit,
        allowed_fileset,
    )
    exact_consumer_authority = bool(consumer_adaptation_authority)
    has_consumer_authority_evidence = any(
        isinstance(owner, dict)
        and bool(owner.get("consumer_adaptation_authority"))
        for owner in work_unit.get("repair_owners") or []
    )
    graph_scope_needed = _needs_code_graph_scope(work_unit)
    graph_root_mapping_incomplete = (
        (
            _root_mapping_requires_expansion(work_unit)
            or graph_scope_needed
            or has_consumer_authority_evidence
        )
        and not (
            exact_relation_authority
            and not unresolved_include_relation_authority
        )
        and not exact_missing_include_authority
        and not exact_consumer_authority
    )
    root_mapping_incomplete = (
        graph_root_mapping_incomplete and not exact_transaction_authority
    )
    scope_complete = bool(
        exact_transaction_authority
        or exact_missing_include_authority
        or (
            (not root_mapping_incomplete or exact_consumer_authority)
            and (not scope_continuation or scope_continuation.get("complete"))
        )
    )
    scope_expansion_candidates = list(scope_expansion_candidates or [])
    entity_materialization_files = list(
        entity_materialization_files or []
    )
    missing_materialization_files = list(
        missing_materialization_files or []
    )
    missing_owner_files = sorted(
        set(scope_expansion_candidates)
        & set(entity_materialization_files)
        & set(missing_materialization_files)
    )
    if exact_transaction_authority:
        mode = "patch_root_subgraph"
        next_action = (
            "correct only the exact surviving postimage of the applied patch "
            "that produced the fresh diagnostic"
        )
    elif exact_missing_include_authority and missing_owner_files:
        mode = "materialize_graph_scope"
        next_action = (
            "materialize only the exact donor file selected by the fresh "
            "missing-include dependency path, then compile before rebuilding "
            "the graph"
        )
    elif exact_consumer_authority and scope_complete:
        mode = "patch_root_subgraph"
        next_action = (
            "adapt only the exact diagnosed consumers after the bounded "
            "provider query reached an unresolved fixpoint"
        )
    elif unresolved_include_relation_authority:
        mode = "patch_root_subgraph"
        next_action = (
            "adapt only the exact unresolved include directive; provider "
            "resolution remains incomplete until the next build"
        )
    elif requires_context_expansion or graph_scope_needed:
        mode = "expand_graph"
        next_action = "continue CodeIndex traversal, remap diagnostics, and rebuild work units"
    elif root_mapping_incomplete:
        mode = "blocked_no_write_target"
        next_action = (
            "record the exhausted unresolved root without patching a symptom "
            "location or requesting another graph page"
        )
    elif missing_owner_files:
        mode = "materialize_graph_scope"
        next_action = (
            "atomically materialize the exact missing graph-derived owner "
            "and its build closure, then compile before rebuilding the graph"
        )
    elif entity_materialization_files:
        mode = "patch_root_subgraph"
        next_action = (
            "materialize the exact reachable donor owner as one "
            "controller-generated entity transaction"
        )
    elif exact_relation_authority and allowed_fileset:
        mode = "patch_root_subgraph"
        next_action = "generate one patch for the exact writable relation"
    elif scope_expansion_candidates:
        mode = "materialize_graph_scope"
        next_action = (
            "atomically materialize the exact graph-derived owner and its build closure, "
            "then compile before rebuilding the graph"
        )
    elif allowed_fileset:
        mode = "patch_root_subgraph"
        next_action = (
            "generate one patch for the mapped root explanation subgraph"
            if scope_complete
            else "patch the mapped root or continue its stored CodeIndex cursor"
        )
    elif requires_escalation or escalation_candidates:
        mode = "request_edit_scope"
        next_action = "request permission for the graph-derived existing owner files"
    else:
        mode = "blocked_no_write_target"
        next_action = "record the unresolved owner boundary without patching a symptom file"

    return {
        "schema": "graph_repair_decision_v2",
        "mode": mode,
        "next_action": next_action,
        "root_node_id": root.get("node_id") or root.get("entity_id"),
        "root_key": root_key or None,
        "root_alternative_node_ids": [
            item.get("node_id") or item.get("entity_id")
            for item in root_alternatives
            if isinstance(item, dict) and (item.get("node_id") or item.get("entity_id"))
        ],
        "root_shape": "mapped_root_set" if mapped_root_set else "single_root",
        "diagnostic_event_ids": list(root.get("diagnostic_event_ids") or []),
        "graph_complete": bool(
            not graph_root_mapping_incomplete and not exact_consumer_authority
        ),
        "root_mapping_complete": bool(
            not graph_root_mapping_incomplete and not exact_consumer_authority
        ),
        "scope_complete": scope_complete,
        "scope_expandable": bool(scope_continuation and not scope_continuation.get("complete")),
        "candidate_write_set": (
            transaction_write_set
            if exact_transaction_authority
            else unresolved_include_write_set
            if unresolved_include_relation_authority
            else
            list(
                consumer_adaptation_authority.get("candidate_write_set")
                or []
            )
            if exact_consumer_authority
            else []
            if requires_context_expansion or root_mapping_incomplete
            else list(allowed_fileset)
        ),
        "repair_authority": (
            "active_patch_transaction_postimage"
            if exact_transaction_authority
            else "exact_missing_include_file"
            if exact_missing_include_authority
            else "exhausted_provider_query_consumer"
            if exact_consumer_authority
            else "include_relation"
            if exact_relation_authority
            else "code_graph_root"
        ),
        "consumer_adaptation_authority": (
            consumer_adaptation_authority or None
        ),
        "graph_expansion_deferred_by_repair_authority": False,
        "escalation_files": list(escalation_candidates),
        "scope_expansion_files": list(scope_expansion_candidates),
        "entity_materialization_files": (
            entity_materialization_files
        ),
        "missing_owner_materialization_files": missing_owner_files,
        "permission_blocked": not bool(
            allowed_fileset
            or entity_materialization_files
            or scope_expansion_candidates
        ),
    }


def _exact_existing_escalation_owner_files(
    unit: dict[str, Any],
) -> list[str]:
    """Return escalation files backed by an exact, existing repair owner."""

    escalation_candidates = set(
        ordered_unique_relpaths(
            [
                str(item)
                for item in unit.get("escalation_candidates") or []
                if str(item).strip()
            ]
        )
    )
    if not escalation_candidates:
        return []
    exact_scope_candidates = set(
        ordered_unique_relpaths(
            [
                str(item)
                for item in unit.get("scope_expansion_candidates") or []
                if str(item).strip()
            ]
        )
    )
    if not exact_scope_candidates:
        return []
    decision = dict(unit.get("repair_decision") or {})
    if (
        unit.get("requires_context_expansion")
        or decision.get("root_mapping_complete") is False
    ):
        return []

    authorized: list[str] = []
    for owner in unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        owner_file = normalize_relpath(
            str(owner.get("owner_file") or "")
        )
        permission = str(
            owner.get("owner_permission")
            or (
                owner.get("allowed_patch_permissions") or {}
            ).get(owner_file)
            or ""
        )
        if (
            owner_file
            and owner_file in escalation_candidates
            and owner_file in exact_scope_candidates
            and repair_owner_has_action_authority(owner)
            and owner.get("old_exists") is True
            and str(owner.get("owner_kind") or "") != "unlocalized"
            and permission == "escalation"
        ):
            authorized.append(owner_file)
    return ordered_unique_relpaths(authorized)


def authorize_repair_plan_escalation(
    repair_plan: dict[str, Any],
    *,
    allow_escalation: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authorize only exact graph-localized escalation repair owners.

    Planning records permission-derived views in several fields.  Runtime
    authorization must update those views as one transaction so the prompt,
    source pack, and patch validator all describe the same writable set.
    """

    if not allow_escalation:
        return repair_plan, []

    authorized_plan = copy.deepcopy(repair_plan)
    records: list[dict[str, Any]] = []
    for unit in authorized_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        authorized_files = _exact_existing_escalation_owner_files(unit)
        if not authorized_files:
            continue
        authorized_set = set(authorized_files)

        repair_owners: list[dict[str, Any]] = []
        for raw_owner in unit.get("repair_owners") or []:
            if not isinstance(raw_owner, dict):
                continue
            owner = copy.deepcopy(raw_owner)
            owner_file = normalize_relpath(
                str(owner.get("owner_file") or "")
            )
            if (
                owner_file in authorized_set
                and owner.get("old_exists") is True
                and str(owner.get("owner_permission") or "") == "escalation"
                and repair_owner_has_action_authority(owner)
            ):
                owner["allowed_as_patch_target"] = True
                owner["allowed_patch_files"] = ordered_unique_relpaths(
                    [
                        *list(owner.get("allowed_patch_files") or []),
                        owner_file,
                    ]
                )
                owner["read_only_evidence_files"] = [
                    path
                    for path in ordered_unique_relpaths(
                        [
                            str(item)
                            for item in owner.get(
                                "read_only_evidence_files"
                            )
                            or []
                        ]
                    )
                    if path not in authorized_set
                ]
                owner["escalation_authorization"] = {
                    "authorized": True,
                    "authority": "runtime_allow_escalation",
                    "permission": "escalation",
                    "files": [owner_file],
                }
            repair_owners.append(owner)

        allowed_fileset = ordered_unique_relpaths(
            [
                *list(unit.get("allowed_fileset") or []),
                *authorized_files,
            ]
        )
        read_only_evidence_files = [
            path
            for path in ordered_unique_relpaths(
                [
                    str(item)
                    for item in unit.get("read_only_evidence_files") or []
                ]
            )
            if path not in authorized_set
        ]
        localization = repair_owner_localization(
            repair_owners
        )
        prior_localization = dict(unit.get("localization") or {})
        localization = {
            **{
                key: value
                for key, value in prior_localization.items()
                if key not in {"status", "allowed_fileset", "targets"}
            },
            **localization,
            "authorization": {
                "authority": "runtime_allow_escalation",
                "files": authorized_files,
            },
        }
        localization_gate = _repair_localization_gate(
            {**unit, "repair_owners": repair_owners},
            allowed_fileset,
        )
        requires_context_expansion = bool(
            unit.get("requires_context_expansion")
            or localization_gate.get("requires_context_expansion")
        )
        requires_escalation = bool(
            requires_context_expansion
            or (
                not allowed_fileset
                and unit.get("escalation_candidates")
            )
        )
        decision = _graph_repair_decision(
            {**unit, "repair_owners": repair_owners},
            allowed_fileset=allowed_fileset,
            requires_context_expansion=requires_context_expansion,
            requires_escalation=requires_escalation,
            escalation_candidates=list(
                unit.get("escalation_candidates") or []
            ),
            scope_expansion_candidates=list(
                unit.get("scope_expansion_candidates") or []
            ),
            entity_materialization_files=list(
                unit.get("entity_materialization_files") or []
            ),
            missing_materialization_files=list(
                unit.get("materialization_candidates") or []
            ),
        )
        patch_targets = list(localization.get("targets") or [])
        unit.update(
            {
                "repair_owners": repair_owners,
                "allowed_fileset": allowed_fileset,
                "read_only_evidence_files": (
                    read_only_evidence_files
                ),
                "localization": localization,
                "patch_targets": patch_targets,
                "patch_target": (
                    dict(patch_targets[0])
                    if len(patch_targets) == 1
                    else {}
                ),
                "context_expansion_request": dict(
                    localization_gate.get("context_expansion_request") or {}
                ),
                "requires_context_expansion": requires_context_expansion,
                "requires_escalation": requires_escalation,
                "repair_decision": decision,
                "escalation_authorization": {
                    "authorized": True,
                    "authority": "runtime_allow_escalation",
                    "files": authorized_files,
                },
                "source_pack_materialization": {
                    **dict(
                        unit.get("source_pack_materialization") or {}
                    ),
                    "authorization_refresh_required": True,
                },
            }
        )
        records.append(
            {
                "unit_id": unit.get("id"),
                "work_unit_id": unit.get("work_unit_id"),
                "authorized_files": authorized_files,
                "authority": "runtime_allow_escalation",
            }
        )

    return authorized_plan, records


def _build_repair_unit_plan(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    candidate_files: list[str],
    retrieval_trace: dict[str, Any],
    expand_context: bool = False,
    refresh_current_tree: bool = False,
) -> dict[str, Any]:
    work_unit = _normalize_repair_work_unit_paths(spec, work_unit)
    graph_expansion_deferred = bool(not expand_context and _needs_code_graph_scope(work_unit))
    if expand_context:
        expansion_result = expand_repair_context(
            spec,
            work_unit=work_unit,
            candidate_files=candidate_files,
            refresh_current_tree=refresh_current_tree,
        )
    else:
        expansion_result = {
            "work_unit": work_unit,
            "extra_candidate_files": [],
            "read_only_evidence_files": [],
            "report": {
                "schema": "repair_context_expansion_v3",
                "enabled": False,
                "deferred": graph_expansion_deferred,
                "reason": (
                    "graph expansion deferred until this ordered root becomes active"
                    if graph_expansion_deferred
                    else "typed diagnostic graph is complete for planning"
                ),
            },
        }
    work_unit = dict(expansion_result.get("work_unit") or work_unit)
    if not work_unit.get("stable_root_anchor"):
        stable_root_anchor = _stable_current_tree_root_anchor(
            work_unit
        )
        if stable_root_anchor:
            work_unit["stable_root_anchor"] = stable_root_anchor
    context_expansion_report = dict(expansion_result.get("report") or {})
    expanded_candidate_files = ordered_unique_relpaths(
        [
            *candidate_files,
            *list(expansion_result.get("extra_candidate_files") or []),
        ]
    )
    if expanded_candidate_files != ordered_unique_relpaths(candidate_files):
        retrieval_trace = dict(retrieval_trace)
        reason_map = {
            normalize_relpath(str(key)): list(value or [])
            for key, value in dict(retrieval_trace.get("reason_map") or {}).items()
        }
        for relative_path in expanded_candidate_files:
            if relative_path in set(ordered_unique_relpaths(candidate_files)):
                continue
            reason_map.setdefault(relative_path, [])
            if "context_expansion_candidate" not in reason_map[relative_path]:
                reason_map[relative_path].append("context_expansion_candidate")
        retrieval_trace["reason_map"] = reason_map
        retrieval_trace["context_expansion"] = {
            "enabled": bool(context_expansion_report.get("enabled")),
            "extra_candidate_files": list(expansion_result.get("extra_candidate_files") or []),
            "read_only_evidence_files": list(expansion_result.get("read_only_evidence_files") or []),
            "relocalized_owner_count": len(context_expansion_report.get("relocalized_owners") or []),
            "code_graph_scope": (context_expansion_report.get("code_graph_scope") or {}).get("summary") or {},
        }
        candidate_files = expanded_candidate_files

    path_records = [_path_record(spec, relative_path) for relative_path in candidate_files]
    escalation_candidates = [
        item["file"]
        for item in path_records
        if item["permission"] == "escalation" and item["old_exists"]
    ]
    materialization_candidates = [
        item["file"]
        for item in path_records
        if item["permission"] in {"editable", "escalation"}
        and not item["old_exists"]
        and item["donor_exists"]
    ]
    broad_scope_expansion_candidates = ordered_unique_relpaths(
        [*escalation_candidates, *materialization_candidates]
    )
    scope_owner_projection = {
        **work_unit,
        "escalation_candidates": escalation_candidates,
        "materialization_candidates": materialization_candidates,
        "scope_expansion_candidates": broad_scope_expansion_candidates,
    }
    scope_expansion_candidates = ordered_unique_relpaths(
        owner.get("file")
        for owner in exact_graph_scope_owners(scope_owner_projection)
        if isinstance(owner, dict)
    )
    pinned_files = set(_pinned_repair_files(candidate_files, work_unit))
    editable_files = [
        item["file"]
        for item in path_records
        if item["permission"] == "editable" and item["old_exists"]
    ]
    if pinned_files:
        initial_allowed_fileset = [relative_path for relative_path in editable_files if relative_path in pinned_files]
        if not initial_allowed_fileset:
            initial_allowed_fileset = editable_files
    else:
        initial_allowed_fileset = editable_files
    localization_gate = _repair_localization_gate(work_unit, initial_allowed_fileset)
    scope_expansion_set = set(scope_expansion_candidates)
    localization_blockers = [
        blocker
        for blocker in localization_gate.get("blockers") or []
        if isinstance(blocker, dict)
    ]
    if (
        localization_gate.get("requires_context_expansion")
        and localization_blockers
        and all(
            normalize_relpath(str(blocker.get("owner_file") or ""))
            in scope_expansion_set
            for blocker in localization_blockers
        )
    ):
        localization_gate = {
            **localization_gate,
            "status": "graph_scope_materialization_ready",
            "patch_allowed": False,
            "requires_context_expansion": False,
            "allowed_fileset": [],
            "context_expansion_request": {},
        }
    entity_materialization_files = ordered_unique_relpaths(
        [
            str(item)
            for item in (
                localization_gate.get(
                    "entity_materialization_files"
                )
                or work_unit.get(
                    "entity_materialization_files"
                )
                or []
            )
            if str(item).strip()
        ]
    )
    allowed_fileset = _allowed_fileset_after_localization(
        initial_allowed_fileset,
        localization_gate,
    )
    read_only_evidence_files = [
        item["file"]
        for item in path_records
        if item["file"] not in set(allowed_fileset)
    ]
    read_only_evidence_files = ordered_unique_relpaths(
        [
            *read_only_evidence_files,
            *[
                item
                for item in expansion_result.get("read_only_evidence_files") or []
                if normalize_relpath(str(item or ""))
                not in set(allowed_fileset)
            ],
        ]
    )
    forbidden_candidates = [item["file"] for item in path_records if item["permission"] == "forbidden"]

    root_cause_cluster = dict(work_unit.get("root_cause_cluster") or {})
    code_graph_root_cause_cluster = dict(work_unit.get("code_graph_root_cause_cluster") or {})
    provider_resolutions = list(work_unit.get("provider_resolutions") or [])
    repair_owners = list(work_unit.get("repair_owners") or [])
    localization = repair_owner_localization(repair_owners)
    patch_targets = list(localization.get("targets") or [])
    patch_target = (
        dict(patch_targets[0]) if len(patch_targets) == 1 else {}
    )
    repair_source_pack = _deferred_repair_source_pack(
        allowed_fileset=allowed_fileset,
        candidate_files=candidate_files,
        read_only_evidence_files=read_only_evidence_files,
    )
    effective_context_expansion_request = dict(
        localization_gate.get("context_expansion_request") or {}
    )
    requires_context_expansion = bool(
        (
            localization_gate["requires_context_expansion"]
            or graph_expansion_deferred
        )
        and _needs_code_graph_scope(work_unit)
    )
    requires_escalation = (
        requires_context_expansion
        or (bool(escalation_candidates) and not bool(allowed_fileset))
    )
    repair_decision = _graph_repair_decision(
        work_unit,
        allowed_fileset=allowed_fileset,
        requires_context_expansion=requires_context_expansion,
        requires_escalation=requires_escalation,
        escalation_candidates=escalation_candidates,
        scope_expansion_candidates=scope_expansion_candidates,
        entity_materialization_files=(
            entity_materialization_files
        ),
        missing_materialization_files=materialization_candidates,
    )
    status = "pending_agent_decision"
    if requires_context_expansion:
        status = "pending_context_expansion"

    return {
        "id": stable_id(
            "plan",
            spec.case_id,
            str(work_unit["id"]),
            "single",
        ),
        "work_unit_id": work_unit["id"],
        "unit_kind": work_unit.get("unit_kind") or "compiler_diagnostic",
        "execution_phase": unit_execution_phase(work_unit),
        "component_work_unit_ids": list(
            work_unit.get("component_work_unit_ids") or []
        ),
        "title": work_unit.get("title"),
        "scope": work_unit.get("scope"),
        "semantic_gap_ids": list(work_unit.get("semantic_gap_ids") or []),
        "required_semantic_gap_ids": list(
            work_unit.get("required_semantic_gap_ids") or []
        ),
        "pending_semantic_gap_ids": list(
            work_unit.get("pending_semantic_gap_ids") or []
        ),
        "blocked_semantic_gap_ids": list(
            work_unit.get("blocked_semantic_gap_ids") or []
        ),
        "semantic_transaction_ids": list(
            work_unit.get("semantic_transaction_ids") or []
        ),
        "incident_patch_ids": list(
            work_unit.get("incident_patch_ids") or []
        ),
        "semantic_transactions": list(
            work_unit.get("semantic_transactions") or []
        ),
        "integration_obligations": list(
            work_unit.get("integration_obligations") or []
        ),
        "required_integration_obligation_ids": list(
            work_unit.get(
                "required_integration_obligation_ids"
            )
            or []
        ),
        "integration_coverage_blockers": list(
            work_unit.get("integration_coverage_blockers") or []
        ),
        "semantic_recovery_context": dict(
            work_unit.get("semantic_recovery_context") or {}
        ),
        "semantic_replan_lineage": dict(
            work_unit.get("semantic_replan_lineage") or {}
        ),
        "semantic_contract_binding": dict(
            work_unit.get("semantic_contract_binding") or {}
        ),
        "semantic_contracts": list(
            work_unit.get("semantic_contracts") or []
        ),
        "difference_atoms": list(work_unit.get("difference_atoms") or []),
        "contract_evidence": list(work_unit.get("contract_evidence") or []),
        "demand_anchors": list(work_unit.get("demand_anchors") or []),
        "demand_anchor_selection": dict(
            work_unit.get("demand_anchor_selection") or {}
        ),
        "demand_owner_analysis": dict(
            work_unit.get("demand_owner_analysis") or {}
        ),
        "repair_loci": list(work_unit.get("repair_loci") or []),
        "status": status,
        "prior_unresolved_patch_attempts": list(work_unit.get("prior_unresolved_patch_attempts") or []),
        "repair_transaction_authorities": list(
            work_unit.get("repair_transaction_authorities") or []
        ),
        "repair_owners": repair_owners,
        "localization": localization,
        "patch_targets": patch_targets,
        "patch_target": patch_target,
        "provider_resolutions": provider_resolutions,
        "diagnostic_signature_count": int(work_unit.get("diagnostic_signature_count") or 0),
        "diagnostic_signatures": list(work_unit.get("diagnostic_signatures") or []),
        "diagnostic_routes": dict(work_unit.get("diagnostic_routes") or {}),
        "diagnostic_graph": dict(work_unit.get("diagnostic_graph") or {}),
        "diagnostic_mappings": list(work_unit.get("diagnostic_mappings") or []),
        "depends_on_work_unit_ids": list(
            work_unit.get("depends_on_work_unit_ids") or []
        ),
        "dependency_root_node_ids": list(
            work_unit.get("dependency_root_node_ids") or []
        ),
        "root_cause_cluster": root_cause_cluster,
        "root_cause_alternatives": list(work_unit.get("root_cause_alternatives") or []),
        "code_graph_root_cause_cluster": code_graph_root_cause_cluster,
        "stable_root_anchor": dict(
            work_unit.get("stable_root_anchor") or {}
        ),
        "source_issue_samples": list(work_unit.get("source_issues") or []),
        "allowed_fileset": allowed_fileset,
        "entity_materialization_files": (
            entity_materialization_files
        ),
        "read_only_evidence_files": read_only_evidence_files,
        "escalation_candidates": escalation_candidates,
        "materialization_candidates": materialization_candidates,
        "scope_expansion_candidates": scope_expansion_candidates,
        "forbidden_candidates": forbidden_candidates,
        "candidate_files": candidate_files,
        "requires_escalation": requires_escalation,
        "requires_context_expansion": requires_context_expansion,
        "context_expansion_request": effective_context_expansion_request,
        "context_expansion": context_expansion_report,
        "code_graph_scope": work_unit.get("code_graph_scope") or context_expansion_report.get("code_graph_scope") or {},
        "retrieval_trace": retrieval_trace,
        "repair_source_pack": repair_source_pack,
        "source_pack_materialization": {
            "status": "deferred",
            "reason": "source pack deferred until this work unit is selected for prompting",
        },
        "repair_decision": repair_decision,
    }


def summarize_repair_plan_units(
    work_unit_plans: list[dict[str, Any]],
    *,
    execution_phase: str | None = None,
    dependency_schedule: dict[str, Any] | None = None,
    source_pack_materialization: str | None = None,
) -> dict[str, Any]:
    """Return the canonical summary for one repair-plan unit set."""

    patchable_count = sum(
        1
        for item in work_unit_plans
        if (
            item.get("allowed_fileset")
            or item.get("entity_materialization_files")
        )
        and not item.get("requires_escalation")
    )
    summary: dict[str, Any] = {
        "execution_phase": execution_phase,
        "work_unit_count": len(work_unit_plans),
        "plans_requiring_escalation": sum(
            1 for item in work_unit_plans if item.get("requires_escalation")
        ),
        "plans_requiring_context_expansion": sum(
            1
            for item in work_unit_plans
            if item.get("requires_context_expansion")
        ),
        "plans_without_editable_files": sum(
            1 for item in work_unit_plans if not item.get("allowed_fileset")
        ),
        "entity_materialization_work_unit_count": sum(
            1
            for item in work_unit_plans
            if item.get("entity_materialization_files")
        ),
        "plans_with_complete_graph": sum(
            1
            for item in work_unit_plans
            if (item.get("repair_decision") or {}).get("graph_complete")
        ),
        "plans_with_incomplete_graph": sum(
            1
            for item in work_unit_plans
            if not (item.get("repair_decision") or {}).get("graph_complete")
        ),
        "patchable_work_unit_count": patchable_count,
        "escalation_only_work_unit_count": sum(
            1
            for item in work_unit_plans
            if not item.get("allowed_fileset")
            and not item.get("entity_materialization_files")
            and item.get("escalation_candidates")
        ),
        "scope_expansion_work_unit_count": sum(
            1
            for item in work_unit_plans
            if item.get("scope_expansion_candidates")
        ),
        "blocked_work_unit_count": sum(
            1
            for item in work_unit_plans
            if not item.get("allowed_fileset")
            and not item.get("entity_materialization_files")
            and not item.get("escalation_candidates")
            and not item.get("scope_expansion_candidates")
        ),
    }
    if source_pack_materialization is not None:
        summary["source_pack_materialization"] = source_pack_materialization
    if dependency_schedule is not None:
        summary["dependency_schedule"] = dependency_schedule
    return summary


def build_repair_plan(
    spec: CaseSpec,
    work_units: list[dict],
    *,
    progress: Callable[[str], None] | None = None,
) -> dict:
    execution_phases = {
        unit_execution_phase(work_unit)
        for work_unit in work_units
    }
    if len(execution_phases) > 1:
        raise ValueError(
            "one repair plan cannot mix execution phases: "
            + ", ".join(sorted(execution_phases))
        )
    execution_phase = (
        next(iter(execution_phases))
        if execution_phases
        else None
    )
    work_unit_plans: list[dict[str, Any]] = []
    total_units = len(work_units)
    for index, work_unit in enumerate(work_units, start=1):
        if progress and (index == 1 or index == total_units or index % 10 == 0):
            progress(
                "repair planning: "
                f"diagnostic_unit={index}/{total_units} "
                f"title={work_unit.get('title')} scope={work_unit.get('scope')}"
            )
        candidate_files, retrieval_trace = _retrieve_repair_candidates(
            spec,
            work_unit,
        )
        plan = _build_repair_unit_plan(
            spec,
            work_unit=work_unit,
            candidate_files=candidate_files,
            retrieval_trace=retrieval_trace,
            expand_context=False,
        )
        work_unit_plans.append(plan)

    work_unit_plans, dependency_schedule = dependency_order(work_unit_plans)

    return {
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "execution_phase": execution_phase,
        "summary": summarize_repair_plan_units(
            work_unit_plans,
            execution_phase=execution_phase,
            dependency_schedule=dependency_schedule,
            source_pack_materialization="deferred",
        ),
        "work_unit_plans": work_unit_plans,
    }


def _repair_work_unit_from_plan_unit(unit: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(unit.get("work_unit_id") or unit.get("id") or ""),
        "unit_kind": unit.get("unit_kind") or "compiler_diagnostic",
        "execution_phase": unit_execution_phase(unit),
        "component_work_unit_ids": list(
            unit.get("component_work_unit_ids") or []
        ),
        "title": unit.get("title"),
        "scope": unit.get("scope"),
        "semantic_gap_ids": list(unit.get("semantic_gap_ids") or []),
        "required_semantic_gap_ids": list(
            unit.get("required_semantic_gap_ids") or []
        ),
        "pending_semantic_gap_ids": list(
            unit.get("pending_semantic_gap_ids") or []
        ),
        "blocked_semantic_gap_ids": list(
            unit.get("blocked_semantic_gap_ids") or []
        ),
        "semantic_transaction_ids": list(
            unit.get("semantic_transaction_ids") or []
        ),
        "incident_patch_ids": list(
            unit.get("incident_patch_ids") or []
        ),
        "semantic_transactions": list(
            unit.get("semantic_transactions") or []
        ),
        "integration_obligations": list(
            unit.get("integration_obligations") or []
        ),
        "required_integration_obligation_ids": list(
            unit.get("required_integration_obligation_ids") or []
        ),
        "integration_coverage_blockers": list(
            unit.get("integration_coverage_blockers") or []
        ),
        "semantic_recovery_context": dict(
            unit.get("semantic_recovery_context") or {}
        ),
        "semantic_replan_lineage": dict(
            unit.get("semantic_replan_lineage") or {}
        ),
        "semantic_contract_binding": dict(
            unit.get("semantic_contract_binding") or {}
        ),
        "semantic_contracts": list(
            unit.get("semantic_contracts") or []
        ),
        "difference_atoms": list(unit.get("difference_atoms") or []),
        "contract_evidence": list(unit.get("contract_evidence") or []),
        "demand_anchors": list(unit.get("demand_anchors") or []),
        "demand_anchor_selection": dict(
            unit.get("demand_anchor_selection") or {}
        ),
        "demand_owner_analysis": dict(
            unit.get("demand_owner_analysis") or {}
        ),
        "repair_loci": list(unit.get("repair_loci") or []),
        "prior_unresolved_patch_attempts": list(unit.get("prior_unresolved_patch_attempts") or []),
        "repair_transaction_authorities": list(
            unit.get("repair_transaction_authorities") or []
        ),
        "source_issues": list(unit.get("source_issue_samples") or []),
        "repair_owners": list(unit.get("repair_owners") or []),
        "provider_resolutions": list(unit.get("provider_resolutions") or []),
        "diagnostic_signature_count": int(unit.get("diagnostic_signature_count") or 0),
        "diagnostic_signatures": list(unit.get("diagnostic_signatures") or []),
        "diagnostic_routes": dict(unit.get("diagnostic_routes") or {}),
        "diagnostic_graph": dict(unit.get("diagnostic_graph") or {}),
        "diagnostic_mappings": list(unit.get("diagnostic_mappings") or []),
        "depends_on_work_unit_ids": list(
            unit.get("depends_on_work_unit_ids") or []
        ),
        "dependency_root_node_ids": list(
            unit.get("dependency_root_node_ids") or []
        ),
        "root_cause_cluster": dict(unit.get("root_cause_cluster") or {}),
        "root_cause_alternatives": list(unit.get("root_cause_alternatives") or []),
        "code_graph_root_cause_cluster": dict(unit.get("code_graph_root_cause_cluster") or {}),
        "stable_root_anchor": dict(unit.get("stable_root_anchor") or {}),
        "localization": dict(unit.get("localization") or {}),
        "patch_targets": list(unit.get("patch_targets") or []),
        "patch_target": dict(unit.get("patch_target") or {}),
        "context_expansion": dict(unit.get("context_expansion") or {}),
        "code_graph_scope": dict(unit.get("code_graph_scope") or {}),
        "entity_materialization_files": list(
            unit.get("entity_materialization_files") or []
        ),
    }


def _unit_has_deferred_source_pack(unit: dict[str, Any]) -> bool:
    source_pack = unit.get("repair_source_pack")
    return isinstance(source_pack, dict) and bool(source_pack.get("deferred"))


def _materialize_repair_plan_unit_source_pack(
    spec: CaseSpec,
    unit: dict[str, Any],
) -> dict[str, Any]:
    materialized = copy.deepcopy(unit)
    work_unit = _repair_work_unit_from_plan_unit(materialized)
    allowed_fileset = ordered_unique_relpaths([str(item) for item in materialized.get("allowed_fileset") or []])
    read_only_evidence_files = ordered_unique_relpaths(
        [str(item) for item in materialized.get("read_only_evidence_files") or []]
    )
    repair_source_pack = build_repair_source_pack(
        spec,
        work_unit=work_unit,
        allowed_fileset=allowed_fileset,
        read_only_evidence_files=read_only_evidence_files,
    )
    materialized.update(
        {
            "repair_source_pack": repair_source_pack,
            "source_pack_materialization": {
                "status": "materialized",
                "materialized_at": utc_now(),
                "reason": "selected active repair unit required full source evidence for prompt generation",
            },
        }
    )
    return materialized


def materialize_deferred_repair_plan_units(
    spec: CaseSpec,
    repair_plan: dict[str, Any],
    *,
    work_unit_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    selected_ids = {str(item) for item in (work_unit_ids or []) if str(item)}
    result = copy.deepcopy(repair_plan)
    materialized_count = 0
    units: list[dict[str, Any]] = []
    for unit in result.get("work_unit_plans") or []:
        unit_id = str(unit.get("work_unit_id") or "")
        plan_unit_id = str(unit.get("id") or "")
        selected = not selected_ids or unit_id in selected_ids or plan_unit_id in selected_ids
        if selected and (refresh or _unit_has_deferred_source_pack(unit)):
            unit = _materialize_repair_plan_unit_source_pack(
                spec,
                unit,
            )
            materialized_count += 1
        units.append(unit)
    result["work_unit_plans"] = units
    summary = dict(result.get("summary") or {})
    summary["materialized_source_pack_count"] = int(summary.get("materialized_source_pack_count") or 0) + materialized_count
    summary["deferred_source_pack_count"] = sum(
        1 for unit in units if _unit_has_deferred_source_pack(unit)
    )
    result["summary"] = summary
    return result


def _graph_plan_unit_state(unit: dict[str, Any]) -> dict[str, Any]:
    scope = dict(unit.get("code_graph_scope") or {})
    continuation = dict(scope.get("continuation") or {})
    root = dict(
        unit.get("code_graph_root_cause_cluster")
        or unit.get("root_cause_cluster")
        or {}
    )
    return {
        "root_node_id": root.get("node_id") or root.get("entity_id"),
        "root_key": graph_root_binding_key(root) or None,
        "ambiguous": bool(root.get("ambiguous")),
        "unmapped": bool(root.get("unmapped")),
        "cursor_by_tree": dict(continuation.get("cursor_by_tree") or {}),
        "cursor_by_tree_symbol": dict(
            continuation.get("cursor_by_tree_symbol") or {}
        ),
        "pending_cursor_by_tree": dict(continuation.get("pending_cursor_by_tree") or {}),
        "pending_cursor_by_tree_symbol": dict(
            continuation.get("pending_cursor_by_tree_symbol") or {}
        ),
        "stage_by_tree_symbol": dict(
            continuation.get("stage_by_tree_symbol") or {}
        ),
        "pending_stage_by_tree_symbol": dict(
            continuation.get("pending_stage_by_tree_symbol") or {}
        ),
        "unresolved_frontier_node_ids": list(
            continuation.get("unresolved_frontier_node_ids") or []
        ),
        "complete": continuation.get("complete"),
        "node_count": int((scope.get("summary") or {}).get("node_count") or len(scope.get("nodes") or [])),
        "edge_count": int((scope.get("summary") or {}).get("edge_count") or len(scope.get("edges") or [])),
        "candidate_files": list(unit.get("candidate_files") or []),
        "allowed_fileset": list(unit.get("allowed_fileset") or []),
        "entity_materialization_files": list(
            unit.get("entity_materialization_files") or []
        ),
        "repair_mode": (unit.get("repair_decision") or {}).get("mode"),
    }


def _unit_requires_graph_continuation(unit: dict[str, Any]) -> bool:
    decision = dict(unit.get("repair_decision") or {})
    consumer_authority = dict(
        decision.get("consumer_adaptation_authority") or {}
    )
    exact_consumer_boundary = bool(
        str(decision.get("mode") or "") == "patch_root_subgraph"
        and decision.get("repair_authority")
        == "exhausted_provider_query_consumer"
        and decision.get("scope_complete") is True
        and consumer_authority.get("status") == "authorized"
        and consumer_authority.get("provider_resolution_complete") is False
        and consumer_authority.get("bounded_provider_query_complete") is True
        and consumer_authority.get("candidate_write_set")
    )
    if bool(unit.get("requires_context_expansion")):
        return True
    if str(decision.get("mode") or "") == "expand_graph":
        return True
    continuation = ((unit.get("code_graph_scope") or {}).get("continuation") or {})
    if exact_consumer_boundary:
        return bool(continuation and not continuation.get("complete"))
    if (
        decision.get("root_mapping_complete") is False
        or decision.get("scope_complete") is False
    ):
        return True
    return bool(continuation and not continuation.get("complete"))


def continue_repair_plan_graph(
    spec: CaseSpec,
    repair_plan: dict[str, Any],
    *,
    work_unit_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    refresh_current_tree: bool = False,
) -> dict[str, Any]:
    """Advance every selected incomplete CodeIndex work unit by exactly one page."""

    selected_ids = {str(item) for item in (work_unit_ids or []) if str(item)}
    rebuilt_units: list[dict[str, Any]] = []
    page_records: list[dict[str, Any]] = []
    for source_unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(source_unit, dict):
            continue
        unit_ids = {
            str(source_unit.get("id") or ""),
            str(source_unit.get("work_unit_id") or ""),
        }
        selected = not selected_ids or bool(unit_ids & selected_ids)
        if not selected or (
            not refresh_current_tree
            and not _unit_requires_graph_continuation(source_unit)
        ):
            rebuilt_units.append(source_unit)
            continue

        before = _graph_plan_unit_state(source_unit)
        work_unit = _repair_work_unit_from_plan_unit(source_unit)
        rebuilt = _build_repair_unit_plan(
            spec,
            work_unit=work_unit,
            candidate_files=ordered_unique_relpaths(
                list(source_unit.get("candidate_files") or [])
            ),
            retrieval_trace=dict(source_unit.get("retrieval_trace") or {}),
            expand_context=True,
            refresh_current_tree=refresh_current_tree,
        )
        rebuilt["id"] = source_unit.get("id")
        rebuilt["work_unit_id"] = rebuilt.get("work_unit_id") or source_unit.get("work_unit_id")
        rebuilt["prior_unresolved_patch_attempts"] = list(
            source_unit.get("prior_unresolved_patch_attempts") or []
        )
        after = _graph_plan_unit_state(rebuilt)
        progressed = before != after
        page_records.append(
            {
                "plan_unit_id": rebuilt.get("id"),
                "work_unit_id": rebuilt.get("work_unit_id"),
                "progressed": progressed,
                "current_tree_refreshed": refresh_current_tree,
                "before": before,
                "after": after,
            }
        )
        rebuilt_units.append(rebuilt)

    result = dict(repair_plan)
    result["generated_at"] = utc_now()
    result["work_unit_plans"] = rebuilt_units
    prior_summary = dict(result.get("summary") or {})
    result["summary"] = {
        **prior_summary,
        **summarize_repair_plan_units(
            rebuilt_units,
            execution_phase=(
                str(result.get("execution_phase") or "") or None
            ),
            dependency_schedule=(
                prior_summary.get("dependency_schedule")
                if isinstance(
                    prior_summary.get("dependency_schedule"),
                    dict,
                )
                else None
            ),
            source_pack_materialization=(
                str(prior_summary.get("source_pack_materialization") or "")
                or None
            ),
        ),
    }
    result["graph_continuation"] = {
        "schema": "repair_plan_graph_continuation_v1",
        "page_count": len(page_records),
        "progressed_count": sum(1 for record in page_records if record["progressed"]),
        "stalled_count": sum(1 for record in page_records if not record["progressed"]),
        "pages": page_records,
    }
    return result
