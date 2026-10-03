#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
from typing import Any, Iterable

from driver_backport.analysis.code_index import (
    CodeIndex,
    GraphBudget,
    build_conditioned_provider_records,
    canonical_entity_records,
    parse_include_directives,
)
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    normalize_relpath,
    normalize_root_relative,
    ordered_unique_relpaths as _ordered_unique,
    stable_id,
)
from driver_backport.core.executor import classify_edit_permission
from driver_backport.planning.root_binding import (
    mapped_root_alternative_ids,
)


MAX_DEFINITION_QUERY_FILES = 16
MAX_GRAPH_NODES = 20_000
MAX_GRAPH_EDGES = 60_000
MAX_EXTRA_CANDIDATES = 32
MAX_READ_ONLY_EVIDENCE = 48


def is_shared_kernel_implementation_path(relative_path: str) -> bool:
    """Return whether a path belongs to a release-wide kernel surface."""

    normalized = normalize_relpath(relative_path)
    return normalized.startswith(
        (
            "include/",
            "arch/",
            "kernel/",
            "lib/",
            "mm/",
            "fs/",
            "net/",
            "block/",
            "security/",
            "ipc/",
            "init/",
            "virt/",
        )
    )


def _has_concrete_source_extent(record: dict[str, Any]) -> bool:
    extent = record.get("extent") if isinstance(record.get("extent"), dict) else {}
    start_line = int(extent.get("start_line") or 0)
    end_line = int(extent.get("end_line") or 0)
    return bool(
        str(record.get("file") or "").strip()
        and start_line > 0
        and end_line >= start_line
    )


def _same_concrete_source_entity(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    """Return whether two parser records describe one source occurrence."""

    if not (
        _has_concrete_source_extent(left)
        and _has_concrete_source_extent(right)
    ):
        return False
    left_extent = left.get("extent") or {}
    right_extent = right.get("extent") or {}
    return (
        str(left.get("tree") or "")
        == str(right.get("tree") or "")
        and normalize_relpath(str(left.get("file") or ""))
        == normalize_relpath(str(right.get("file") or ""))
        and str(left.get("contract_key") or left.get("key") or "")
        == str(right.get("contract_key") or right.get("key") or "")
        and left.get("is_definition") is right.get("is_definition")
        and all(
            int(left_extent.get(field) or 0)
            == int(right_extent.get(field) or 0)
            for field in (
                "start_line",
                "start_column",
                "end_line",
                "end_column",
            )
        )
    )


def _unresolved_include_relations(
    graph: dict[str, Any],
    root_node_ids: Iterable[object],
) -> list[dict[str, Any]]:
    """Return exact current include relations for a selected unresolved path.

    The path is Diagnostic -> Include -> UnresolvedInclude, optionally ending
    at a selected donor File. The diagnostic must identify one current file.
    """

    nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and str(node.get("id") or "")
    }
    selected_ids = {str(node_id) for node_id in root_node_ids if str(node_id)}
    root_ids = {
        str(node_id)
        for node_id in selected_ids
        if str(node_id)
        and str((nodes.get(str(node_id)) or {}).get("kind") or "")
        == "File"
        and str((nodes.get(str(node_id)) or {}).get("tree") or "").startswith(
            "donor_"
        )
    }
    selected_unresolved_ids = {
        node_id
        for node_id in selected_ids
        if str((nodes.get(node_id) or {}).get("kind") or "")
        == "UnresolvedInclude"
    }
    if not root_ids and not selected_unresolved_ids:
        return []

    candidate_unresolved = selected_unresolved_ids | {
        str(edge.get("from") or "")
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("type") or "") == "candidate_resolution"
        and str(edge.get("to") or "") in root_ids
        and str(
            (nodes.get(str(edge.get("from") or "")) or {}).get("kind")
            or ""
        )
        == "UnresolvedInclude"
    }
    include_to_unresolved = {
        str(edge.get("from") or ""): str(edge.get("to") or "")
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("type") or "") == "unresolved_dependency"
        and str(edge.get("to") or "") in candidate_unresolved
        and str(
            (nodes.get(str(edge.get("from") or "")) or {}).get("kind")
            or ""
        )
        == "Include"
        and str(
            (nodes.get(str(edge.get("from") or "")) or {}).get("tree")
            or ""
        )
        == "old"
    }
    diagnostic_to_includes: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict) or str(edge.get("type") or "") != "diagnoses":
            continue
        diagnostic_id = str(edge.get("from") or "")
        include_id = str(edge.get("to") or "")
        if (
            include_id in include_to_unresolved
            and str((nodes.get(diagnostic_id) or {}).get("kind") or "")
            == "Diagnostic"
        ):
            diagnostic_to_includes[diagnostic_id].add(include_id)
    diagnostic_to_consumers: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        diagnostic_id = str(edge.get("from") or "")
        consumer_id = str(edge.get("to") or "")
        if (
            str(edge.get("type") or "") == "occurs_at"
            and diagnostic_id in diagnostic_to_includes
            and str((nodes.get(consumer_id) or {}).get("kind") or "")
            == "File"
            and str((nodes.get(consumer_id) or {}).get("tree") or "")
            == "old"
        ):
            diagnostic_to_consumers[diagnostic_id].add(consumer_id)

    relations: list[dict[str, Any]] = []
    for diagnostic_id, include_ids in sorted(diagnostic_to_includes.items()):
        consumers = diagnostic_to_consumers.get(diagnostic_id) or set()
        if len(consumers) != 1 or len(include_ids) != 1:
            continue
        include_id = next(iter(include_ids))
        include = nodes.get(include_id) or {}
        relations.append(
            {
                "diagnostic_node_id": diagnostic_id,
                "include_node_id": include_id,
                "unresolved_node_id": include_to_unresolved[include_id],
                "consumer_node_id": next(iter(consumers)),
                "include_name": normalize_relpath(
                    str(include.get("name") or include.get("spelling") or "")
                ),
                "line": int(include.get("line") or 0),
            }
        )
    return relations


def _proven_shared_contract_adaptation_root(
    graph: dict[str, Any],
    root_node_ids: str | Iterable[object],
    *,
    spec: object | None = None,
    provider_resolutions: Iterable[dict[str, Any]] = (),
) -> bool:
    """Return whether complete evidence selects one shared contract."""

    if isinstance(root_node_ids, str):
        selected_root_ids = [root_node_ids] if root_node_ids else []
    else:
        selected_root_ids = sorted(
            {str(node_id) for node_id in root_node_ids if str(node_id)}
        )
    if not selected_root_ids or bool(
        (graph.get("summary") or {}).get("truncated")
    ):
        return False
    nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and str(node.get("id") or "")
    }
    roots = [nodes.get(node_id) or {} for node_id in selected_root_ids]
    if any(not root for root in roots):
        return False
    root_contracts = {
        str(root.get("contract_key") or root.get("key") or "")
        for root in roots
    }
    if len(root_contracts) != 1 or not next(iter(root_contracts)):
        return False
    root_contract = next(iter(root_contracts))
    if any(
        (
            str(root.get("tree") or "") != "old"
            and not str(root.get("tree") or "").startswith("donor_")
        )
        or not (
            is_shared_kernel_implementation_path(
                str(root.get("file") or root.get("path") or "")
            )
            or (
                spec is not None
                and classify_edit_permission(
                    spec,
                    normalize_relpath(
                        str(root.get("file") or root.get("path") or "")
                    ),
                )
                != "editable"
            )
        )
        for root in roots
    ):
        return False
    for resolution in provider_resolutions:
        if not isinstance(resolution, dict):
            continue
        status = str(resolution.get("status") or "")
        entity_key = (
            "target_entities"
            if status == "old_entity_present"
            else "donor_entities"
            if status == "donor_entity_missing_in_old_unique"
            else ""
        )
        if not entity_key:
            continue
        resolved_entities = [
            entity
            for entity in resolution.get(entity_key) or []
            if isinstance(entity, dict)
            and str(entity.get("contract_key") or entity.get("key") or "")
            == root_contract
        ]
        if all(
            any(
                root_id
                == str(
                    entity.get("id")
                    or entity.get("entity_id")
                    or ""
                )
                or _same_concrete_source_entity(root, entity)
                for entity in resolved_entities
            )
            for root_id, root in zip(selected_root_ids, roots)
        ):
            return True
    return False


def _exhausted_provider_consumer_adaptation_authority(
    graph: dict[str, Any],
    root_node_ids: str | Iterable[object],
    *,
    provider_resolutions: Iterable[dict[str, Any]] = (),
    diagnostic_mappings: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Authorize exact consumers without claiming provider resolution."""

    if isinstance(root_node_ids, str):
        selected_root_ids = [root_node_ids] if root_node_ids else []
    else:
        selected_root_ids = sorted(
            {str(node_id) for node_id in root_node_ids if str(node_id)}
        )
    if not selected_root_ids or bool(
        (graph.get("summary") or {}).get("truncated")
    ):
        return {}

    nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and str(node.get("id") or "")
    }
    roots = [nodes.get(node_id) or {} for node_id in selected_root_ids]
    if any(
        not root
        or not str(root.get("tree") or "").startswith("donor_")
        or root.get("is_definition") is not False
        or not _has_concrete_source_extent(root)
        for root in roots
    ):
        return {}

    matched_resolutions: list[dict[str, Any]] = []
    for root_id, root in zip(selected_root_ids, roots):
        root_contract = str(
            root.get("contract_key") or root.get("key") or ""
        )
        if not root_contract:
            return {}
        matches: list[dict[str, Any]] = []
        for resolution in provider_resolutions:
            if not isinstance(resolution, dict):
                continue
            donor_entities = [
                entity
                for entity in resolution.get("donor_entities") or []
                if isinstance(entity, dict)
                and str(entity.get("id") or entity.get("entity_id") or "")
                == root_id
                and str(
                    entity.get("contract_key") or entity.get("key") or ""
                )
                == root_contract
                and entity.get("is_definition") is False
                and _has_concrete_source_extent(entity)
            ]
            query = dict(resolution.get("old_definition_query") or {})
            if (
                str(resolution.get("status") or "") == "entity_unresolved"
                and len(donor_entities) == 1
                and not resolution.get("target_entities")
                and int(resolution.get("old_definition_count") or 0) == 0
                and resolution.get("definition_required") is True
                and query.get("complete") is True
                and query.get("resolved") is False
                and query.get("unavailable") is False
            ):
                matches.append(resolution)
        if len(matches) != 1:
            return {}
        matched_resolutions.append(matches[0])

    component_diagnostic_ids = {
        node_id
        for node_id, node in nodes.items()
        if str(node.get("kind") or "") == "Diagnostic"
    }
    mappings = [
        mapping
        for mapping in diagnostic_mappings
        if isinstance(mapping, dict)
        and str(mapping.get("diagnostic_node_id") or "")
        in component_diagnostic_ids
    ]
    if not mappings:
        return {}
    consumer_owner_ids: set[str] = set()
    for mapping in mappings:
        owner_ids = {
            str(node_id)
            for node_id in mapping.get(
                "diagnostic_transaction_owner_node_ids"
            )
            or []
            if str(node_id)
        }
        if mapping.get("unmapped") or not owner_ids:
            return {}
        consumer_owner_ids.update(owner_ids)
    if any(
        not (nodes.get(owner_id) or {})
        or str((nodes.get(owner_id) or {}).get("tree") or "") != "old"
        or (nodes.get(owner_id) or {}).get("is_definition") is not True
        or not _has_concrete_source_extent(nodes.get(owner_id) or {})
        for owner_id in consumer_owner_ids
    ):
        return {}

    return {
        "schema": "exhausted_provider_consumer_authority_v1",
        "status": "authorized",
        "repair_authority": "exhausted_provider_query_consumer",
        "provider_resolution_complete": False,
        "bounded_provider_query_complete": True,
        "root_node_ids": selected_root_ids,
        "consumer_owner_node_ids": sorted(consumer_owner_ids),
        "provider_symbols": sorted(
            {
                str(resolution.get("symbol") or "")
                for resolution in matched_resolutions
                if str(resolution.get("symbol") or "")
            }
        ),
        "reason": (
            "the bounded available old-tree provider query is exhausted, the "
            "selected donor contract is declaration-only, and every component "
            "diagnostic maps to an exact current transaction owner"
        ),
    }


def _include_relation_owner_candidates(
    index: CodeIndex,
    graph: dict[str, Any],
    root_node_id: str,
    *,
    provider_resolutions: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return exact include-relation owners for one selected graph root."""

    if not root_node_id or bool((graph.get("summary") or {}).get("truncated")):
        return []
    nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and str(node.get("id") or "")
    }
    root = nodes.get(root_node_id) or {}
    root_contract = str(root.get("contract_key") or root.get("key") or "")
    header_files: set[str] = set()
    if (
        str(root.get("kind") or "") == "File"
        and str(root.get("tree") or "").startswith("donor_")
    ):
        root_file = normalize_relpath(
            str(root.get("file") or root.get("path") or "")
        )
        if root_file.startswith("include/") and root_file.endswith(".h"):
            header_files.add(root_file)
    if root_contract:
        for resolution in provider_resolutions:
            if not isinstance(resolution, dict):
                continue
            status = str(resolution.get("status") or "")
            entity_field = (
                "target_entities"
                if status == "old_entity_present"
                else "donor_entities"
                if status == "donor_entity_missing_in_old_unique"
                else ""
            )
            if not entity_field:
                continue
            for entity in resolution.get(entity_field) or []:
                if not isinstance(entity, dict):
                    continue
                entity_contract = str(
                    entity.get("contract_key") or entity.get("key") or ""
                )
                relative_path = normalize_relpath(
                    str(entity.get("file") or entity.get("path") or "")
                )
                if (
                    entity_contract == root_contract
                    and relative_path.startswith("include/")
                    and relative_path.endswith(".h")
                ):
                    header_files.add(relative_path)
    unresolved_root = str(root.get("kind") or "") == "UnresolvedInclude"
    if not unresolved_root and len(header_files) != 1:
        return []
    provider_header = (
        next(iter(header_files))
        if not unresolved_root and len(header_files) == 1
        else ""
    )
    include_name = provider_header[len("include/") :] if provider_header else ""
    provider_tree = str(root.get("tree") or "")
    unresolved_relations = _unresolved_include_relations(
        graph,
        [root_node_id],
    )
    header_symbols: set[str] = set()
    if unresolved_relations and provider_header:
        header_report = index.index_structural_file(
            provider_tree,
            provider_header,
        )
        header_symbols = {
            str(entity.get("symbol") or entity.get("name") or "")
            for entity in header_report.get("entities") or []
            if isinstance(entity, dict)
            and str(entity.get("symbol") or entity.get("name") or "")
            and _has_concrete_source_extent(entity)
            and (
                str(entity.get("kind") or "").lower() == "macro"
                or not CodeIndex._extent_overlaps_error_nodes(
                    dict(entity.get("extent") or {}),
                    header_report.get("error_nodes") or [],
                )
            )
        }

    records: list[dict[str, Any]] = []
    unresolved_consumer_ids: set[str] = set()
    retained_consumer_entity_ids: set[str] = set()
    for relation in unresolved_relations:
        current_include_name = normalize_relpath(
            str(relation.get("include_name") or "")
        )
        consumer_node_id = str(relation.get("consumer_node_id") or "")
        consumer = nodes.get(consumer_node_id) or {}
        consumer_file = normalize_relpath(
            str(consumer.get("file") or consumer.get("path") or "")
        )
        if (
            not current_include_name
            or int(relation.get("line") or 0) <= 0
            or not consumer_file.endswith(".c")
            or not _tree_has_file(index, "old", consumer_file)
        ):
            continue
        if unresolved_root:
            current_includes = parse_include_directives(
                root=index.root_for_tree("old"),
                driver_root=index.spec.resolve_old(index.spec.driver_root),
                relative_path=consumer_file,
                tree="old",
            )
            exact_includes = [
                include
                for include in current_includes
                if normalize_relpath(str(include.get("name") or ""))
                == current_include_name
                and int(include.get("line") or 0)
                == int(relation.get("line") or 0)
                and not include.get("resolved")
            ]
            if len(exact_includes) != 1:
                continue
        unresolved_consumer_ids.add(consumer_node_id)
        consumer_report = (
            index.index_structural_file("old", consumer_file)
            if header_symbols
            else {}
        )
        consumer_entities = canonical_entity_records(
            entity
            for entity in consumer_report.get("entities") or []
            if isinstance(entity, dict)
            and bool(entity.get("is_definition"))
            and _has_concrete_source_extent(entity)
            and header_symbols.intersection(
                str(symbol)
                for symbol in entity.get("dependency_symbols") or []
                if str(symbol)
            )
        )
        exact_consumer_ids: list[str] = []
        for entity in consumer_entities:
            entity_id = str(
                entity.get("id") or entity.get("entity_id") or ""
            )
            if not entity_id:
                continue
            exact_consumer_ids.append(entity_id)
            if entity_id in retained_consumer_entity_ids:
                continue
            retained_consumer_entity_ids.add(entity_id)
            records.append(
                {
                    **entity,
                    "file": consumer_file,
                    "node_id": entity_id,
                    "root_node_id": root_node_id,
                    "owner_kind": "code_index_consumer_integration_owner",
                    "owner_role": "consumer_integration_owner",
                    "permission": classify_edit_permission(
                        index.spec,
                        consumer_file,
                    ),
                    "old_exists": True,
                    "donor_exists": any(
                        _tree_has_file(index, tree_name, consumer_file)
                        for tree_name in index.contexts
                        if tree_name.startswith("donor_")
                    ),
                    "tree": "old",
                    "reasons": [
                        "this exact current definition depends on a contract "
                        "declared by the uniquely selected missing header"
                    ],
                }
            )
        records.append(
            {
                "file": consumer_file,
                "node_id": str(relation.get("include_node_id") or ""),
                "root_node_id": root_node_id,
                "owner_kind": "code_index_include_relation",
                "owner_role": "include_relation_owner",
                "permission": classify_edit_permission(index.spec, consumer_file),
                "old_exists": True,
                "donor_exists": any(
                    _tree_has_file(index, tree_name, consumer_file)
                    for tree_name in index.contexts
                    if tree_name.startswith("donor_")
                ),
                "tree": "old",
                "kind": "Include",
                "contract_key": f"include:{current_include_name}",
                "include_relation_mode": "mutate_unresolved",
                "current_include_name": current_include_name,
                "provider_header_file": provider_header or None,
                "expected_line": int(relation.get("line") or 0),
                "consumer_owner_node_ids": (
                    sorted(exact_consumer_ids)
                    if exact_consumer_ids
                    else [consumer_node_id]
                ),
                "reasons": [
                    "the located diagnostic selects this exact current "
                    "include relation as unresolved; provider resolution "
                    "remains separate"
                ],
            }
        )

    diagnostic_ids = {
        node_id
        for node_id, node in nodes.items()
        if str(node.get("kind") or "") == "Diagnostic"
    }
    diagnosed_reference_ids = {
        str(edge.get("to") or "")
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("type") or "") == "diagnoses"
        and str(edge.get("from") or "") in diagnostic_ids
        and str((nodes.get(str(edge.get("to") or "")) or {}).get("kind") or "")
        == "Reference"
    }
    consumer_owner_ids = {
        str(edge.get("to") or "")
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("type") or "") == "occurs_in"
        and str(edge.get("from") or "") in diagnosed_reference_ids
        and str((nodes.get(str(edge.get("to") or "")) or {}).get("tree") or "")
        == "old"
    }
    seen_files: set[str] = set()
    for owner_node_id in sorted(consumer_owner_ids - unresolved_consumer_ids):
        owner = nodes.get(owner_node_id) or {}
        consumer_file = normalize_relpath(
            str(owner.get("file") or owner.get("path") or "")
        )
        if (
            not consumer_file.endswith(".c")
            or consumer_file in seen_files
            or not _tree_has_file(index, "old", consumer_file)
        ):
            continue
        existing_includes = {
            normalize_relpath(str(include.get("name") or ""))
            for include in index.includes_by_file.get(("old", consumer_file), [])
            if isinstance(include, dict)
        }
        if include_name in existing_includes:
            continue
        seen_files.add(consumer_file)
        relation_node_id = stable_id(
            "include-relation",
            root_node_id,
            consumer_file,
            include_name,
        )
        records.append(
            {
                "file": consumer_file,
                "node_id": relation_node_id,
                "root_node_id": root_node_id,
                "owner_kind": "code_index_include_relation",
                "owner_role": "include_relation_owner",
                "permission": classify_edit_permission(index.spec, consumer_file),
                "old_exists": True,
                "donor_exists": any(
                    _tree_has_file(index, tree_name, consumer_file)
                    for tree_name in index.contexts
                    if tree_name.startswith("donor_")
                ),
                "tree": "old",
                "kind": "IncludeBlock",
                "contract_key": f"include:{include_name}",
                "include_relation_mode": "insert_provider",
                "required_include_name": include_name,
                "provider_header_file": provider_header,
                "consumer_owner_node_ids": [owner_node_id],
                "reasons": [
                    "the completed provider query resolves this consumer's "
                    "missing declaration to exactly one provider header and "
                    "the current source lacks that include relation"
                ],
            }
        )
    return records


def _normalize_old_relative(spec: CaseSpec, value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return normalize_root_relative(
        text,
        root=spec.old_root_path,
        preferred_prefix=spec.driver_root,
    ) or ""


def _work_unit_issues(work_unit: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in work_unit.get("source_issues") or []:
        if not isinstance(raw, dict):
            continue
        event_id = str(raw.get("diagnostic_event_id") or repr(raw))
        if event_id not in seen:
            seen.add(event_id)
            issues.append(dict(raw))
    return issues


def _seed_files(spec: CaseSpec, work_unit: dict[str, Any], candidate_files: list[str]) -> list[str]:
    values: list[object] = []
    root = (
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    if isinstance(root, dict):
        values.append(root.get("file"))
    for issue in _work_unit_issues(work_unit):
        values.extend(
            [
                issue.get("file"),
                issue.get("source_file"),
                issue.get("translation_unit"),
                issue.get("trigger_file"),
                *(issue.get("trigger_files") or []),
            ]
        )
    for owner in work_unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        values.append(owner.get("owner_file"))
        values.extend(owner.get("allowed_patch_files") or [])
    for resolution in work_unit.get("provider_resolutions") or []:
        if not isinstance(resolution, dict):
            continue
        values.append(resolution.get("preferred_patch_file"))
        for key in ("target_entities", "donor_entities"):
            values.extend(
                entity.get("file")
                for entity in resolution.get(key) or []
                if isinstance(entity, dict)
            )
    if not any(str(value or "").strip() for value in values):
        values.extend([*candidate_files, *spec.entry_files])
    return _ordered_unique(
        normalized
        for value in values
        for normalized in [_normalize_old_relative(spec, value)]
        if normalized
    )


def _seed_files_for_tree(
    spec: CaseSpec,
    work_unit: dict[str, Any],
    candidate_files: list[str],
    tree: str,
) -> list[str]:
    seeds = _seed_files(spec, work_unit, candidate_files)
    if tree == "old":
        return seeds
    root = (
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    selected: list[object] = []
    if isinstance(root, dict) and str(root.get("tree") or "") == tree:
        selected.append(root.get("file"))
    for owner in work_unit.get("repair_owners") or []:
        if isinstance(owner, dict) and owner.get("owner_entity_tree") == tree:
            selected.append(owner.get("owner_file"))
    for resolution in work_unit.get("provider_resolutions") or []:
        if not isinstance(resolution, dict):
            continue
        selected.extend(
            entity.get("file")
            for entity in resolution.get("donor_entities") or []
            if isinstance(entity, dict) and entity.get("tree") == tree
        )
    return _ordered_unique(
        path
        for value in selected
        for path in [_normalize_old_relative(spec, value)]
        if path and path in seeds
    )


def _normalized_contract_keys(values: Iterable[object]) -> list[str]:
    return sorted(
        {
            normalized
            for value in values
            if value is not None
            for normalized in [str(value).strip()]
            if normalized
        }
    )


def _graph_query_demand_records(
    index: CodeIndex,
    work_unit: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return unresolved symbol contracts and their demanded provider facet."""

    graph_nodes = {
        str(node_id): dict(node)
        for node_id, node in index.graph.nodes.items()
    }
    prior_graph = work_unit.get("diagnostic_graph")
    if isinstance(prior_graph, dict):
        for node in prior_graph.get("nodes") or []:
            if isinstance(node, dict) and node.get("id"):
                graph_nodes.setdefault(str(node["id"]), dict(node))

    demanded_node_ids: set[str] = set()
    diagnosed_provider_declaration_ids: set[str] = set()
    root = (
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    if isinstance(root, dict):
        demanded_node_ids.update(
            str(value)
            for value in (root.get("node_id"), root.get("entity_id"))
            if str(value or "")
        )
        demanded_node_ids.update(
            str(item.get("node_id") or item.get("entity_id") or "")
            for item in root.get("alternatives") or []
            if isinstance(item, dict)
            and (item.get("node_id") or item.get("entity_id"))
        )
    for mapping in index.diagnostic_mappings.values():
        mapping_roots = {
            str(item)
            for item in mapping.get("root_candidate_node_ids") or []
            if str(item)
        }
        demanded_node_ids.update(mapping_roots)
        if not mapping.get("unmapped"):
            for node_id in mapping.get("mapped_node_ids") or []:
                reference = graph_nodes.get(str(node_id)) or {}
                provider_id = str(reference.get("referenced_entity_id") or "")
                provider = graph_nodes.get(provider_id) or {}
                if (
                    reference.get("kind") == "Reference"
                    and reference.get("tree") == "old"
                    and provider_id in mapping_roots
                    and provider.get("tree") == "old"
                    and provider.get("kind") in {"Function", "Variable"}
                    and str(provider.get("contract_key") or provider.get("key") or "")
                    .startswith(("function:", "variable:"))
                    and provider.get("is_definition") is False
                    and _has_concrete_source_extent(provider)
                ):
                    diagnosed_provider_declaration_ids.add(provider_id)
        if mapping_roots:
            continue
        demanded_node_ids.update(
            str(item)
            for item in mapping.get("mapped_node_ids") or []
            if str(item)
            and str(
                (graph_nodes.get(str(item)) or {}).get("kind") or ""
            ).startswith("Unresolved")
        )

    def has_concrete_definition(node: dict[str, Any]) -> bool:
        contract = str(node.get("contract_key") or node.get("key") or "")
        symbol = str(
            node.get("symbol") or node.get("spelling") or node.get("name") or ""
        )
        tree = str(node.get("tree") or "")
        for candidate in index.graph.nodes.values():
            if not bool(candidate.get("is_definition")):
                continue
            if tree and str(candidate.get("tree") or "") != tree:
                continue
            candidate_contract = str(
                candidate.get("contract_key") or candidate.get("key") or ""
            )
            candidate_symbol = str(
                candidate.get("symbol")
                or candidate.get("spelling")
                or candidate.get("name")
                or ""
            )
            if contract and candidate_contract == contract:
                return True
            if not contract and symbol and candidate_symbol == symbol:
                return True
        return False

    def has_concrete_old_counterpart(node: dict[str, Any]) -> bool:
        contract = str(node.get("contract_key") or node.get("key") or "")
        symbol = str(
            node.get("symbol") or node.get("spelling") or node.get("name") or ""
        )
        for candidate in graph_nodes.values():
            if str(candidate.get("tree") or "") != "old":
                continue
            if not _has_concrete_source_extent(candidate):
                continue
            if (
                node.get("is_definition") is False
                and candidate.get("is_definition") is not False
            ):
                continue
            candidate_contract = str(
                candidate.get("contract_key") or candidate.get("key") or ""
            )
            candidate_symbol = str(
                candidate.get("symbol")
                or candidate.get("spelling")
                or candidate.get("name")
                or ""
            )
            if contract and candidate_contract == contract:
                return True
            if not contract and symbol and candidate_symbol == symbol:
                return True
        return False

    demands: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "contract_keys": set(),
            "require_definition": False,
        }
    )

    def record_demand(
        node: dict[str, Any],
        *,
        require_definition: bool,
    ) -> None:
        symbols = list(
            dict.fromkeys(
                str(node.get(field) or "").strip()
                for field in ("symbol", "spelling", "name")
                if str(node.get(field) or "").strip()
            )
        )
        provider_contracts = set(
            _normalized_contract_keys(
                node.get("provider_contract_keys") or []
            )
        )
        node_kind = str(node.get("kind") or "")
        node_contract = str(
            node.get("contract_key") or node.get("key") or ""
        )
        if (
            node_contract
            and not node_kind.startswith("Unresolved")
            and node_kind != "Reference"
        ):
            provider_contracts.add(node_contract)
        for symbol in symbols:
            demand = demands[str(symbol)]
            demand["contract_keys"].update(provider_contracts)
            demand["require_definition"] = bool(
                demand["require_definition"] or require_definition
            )
            if node_kind.startswith("Unresolved") and not provider_contracts:
                declarations = {
                    str(record.get("contract_key") or record.get("key") or "")
                    for record in index.entities_by_name.get(symbol, [])
                    if _has_concrete_source_extent(record)
                    and str(record.get("cursor_kind") or "")
                    in {"FUNCTION_DECL", "VAR_DECL"}
                    and str(record.get("contract_key") or record.get("key") or "")
                    .startswith(("function:", "variable:"))
                }
                if len(declarations) == 1:
                    demand["contract_keys"].update(declarations)
                    demand["require_definition"] = True

    for node_id in sorted(demanded_node_ids):
        node = graph_nodes.get(node_id) or {}
        kind = str(node.get("kind") or "")
        unresolved_reference = (
            kind == "Reference"
            and not node.get("referenced_entity_id")
        )
        unresolved_entity = kind.startswith("Unresolved")
        incomplete_demanded_root = (
            kind in {"Type", "Field"}
            and node.get("is_definition") is False
            and not has_concrete_definition(node)
        )
        donor_root_without_old_counterpart = (
            str(node.get("tree") or "").startswith("donor_")
            and kind
            in {
                "Function",
                "Variable",
                "Type",
                "Field",
                "Enum_Constant",
                "Macro",
                "Entity",
            }
            and not has_concrete_old_counterpart(node)
        )
        if not (
            unresolved_reference
            or unresolved_entity
            or incomplete_demanded_root
            or node_id in diagnosed_provider_declaration_ids
            or donor_root_without_old_counterpart
        ):
            continue
        record_demand(
            node,
            require_definition=bool(
                incomplete_demanded_root
                or node_id in diagnosed_provider_declaration_ids
                or (
                    donor_root_without_old_counterpart
                    and node.get("is_definition") is True
                )
            ),
        )
    resolved_symbols = {
        str(resolution.get("symbol") or "")
        for resolution in work_unit.get("provider_resolutions") or []
        if isinstance(resolution, dict)
        and str(resolution.get("status") or "")
        not in {"", "entity_unresolved"}
    }
    for owner in work_unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        interface = dict(owner.get("driver_interface_ownership") or {})
        if interface.get("status") != "external_providers_outside_selected_entries":
            continue
        for declaration in interface.get("unmatched_declarations") or []:
            if not isinstance(declaration, dict):
                continue
            symbol = str(declaration.get("symbol") or "").strip()
            if symbol and symbol not in resolved_symbols:
                contract_keys = _normalized_contract_keys(
                    [
                        declaration.get("contract_key"),
                        declaration.get("key"),
                    ]
                )
                demands[symbol]["contract_keys"].update(contract_keys)
                if any(
                    contract_key.startswith(("function:", "variable:"))
                    for contract_key in contract_keys
                ):
                    demands[symbol]["require_definition"] = True
    return {
        symbol: {
            "contract_keys": sorted(demand["contract_keys"]),
            "require_definition": bool(demand["require_definition"]),
        }
        for symbol, demand in sorted(demands.items())
        if symbol
    }


def _graph_query_demands(
    index: CodeIndex,
    work_unit: dict[str, Any],
) -> dict[str, list[str]]:
    """Return unresolved symbols with any exact demanded contract keys."""

    return {
        symbol: list(record["contract_keys"])
        for symbol, record in _graph_query_demand_records(
            index,
            work_unit,
        ).items()
    }


def _graph_query_symbols(
    index: CodeIndex,
    work_unit: dict[str, Any],
) -> list[str]:
    """Project demanded symbols for callers that do not need contract keys."""

    return list(_graph_query_demands(index, work_unit))


def _tree_names(spec: CaseSpec) -> list[str]:
    return ["old", *[f"donor_{index}" for index in range(len(spec.donor_root_paths))]]


def _definition_query_start(
    traversal_cursor: dict[str, Any],
    *,
    tree: str,
    symbol: str,
) -> int:
    nested = traversal_cursor.get(tree)
    if not isinstance(nested, dict):
        return 0
    try:
        return max(0, int(nested.get(symbol) or 0))
    except (TypeError, ValueError):
        return 0


def _definition_query_demands_with_history(
    query_demands: dict[str, list[str]],
    definition_requirements: dict[str, bool],
    prior_scope: dict[str, Any],
) -> tuple[dict[str, list[str]], dict[str, bool]]:
    """Keep exact query obligations monotonic for one candidate generation."""

    merged_demands = {
        str(symbol): list(contract_keys)
        for symbol, contract_keys in query_demands.items()
        if str(symbol)
    }
    merged_requirements = {
        str(symbol): bool(required)
        for symbol, required in definition_requirements.items()
        if str(symbol)
    }
    for tree_states in dict(
        prior_scope.get("definition_query_state") or {}
    ).values():
        if not isinstance(tree_states, dict):
            continue
        for symbol, state in tree_states.items():
            if not str(symbol) or not isinstance(state, dict):
                continue
            merged_demands.setdefault(str(symbol), []).extend(
                _normalized_contract_keys(
                    state.get("contract_keys") or []
                )
            )
            if state.get("require_definition"):
                merged_requirements[str(symbol)] = True
    return (
        {
            symbol: _normalized_contract_keys(contract_keys)
            for symbol, contract_keys in merged_demands.items()
        },
        merged_requirements,
    )


DEFINITION_QUERY_STAGES = (
    "case_compile_frontier",
    "case_dependency_frontier",
    "demand_provider_definition",
)


def _requires_external_provider_definition(
    contract_keys: Iterable[object],
    *,
    require_definition: bool,
) -> bool:
    normalized = _normalized_contract_keys(contract_keys)
    return bool(
        any(
            contract_key.startswith(("function:", "variable:"))
            for contract_key in normalized
        )
        or (require_definition and not normalized)
    )


def _definition_query_prior_state(
    prior_scope: dict[str, Any],
    *,
    tree: str,
    symbol: str,
    contract_keys: Iterable[object] = (),
    require_definition: bool = False,
) -> dict[str, Any]:
    state = (
        (prior_scope.get("definition_query_state") or {})
        .get(tree, {})
        .get(symbol, {})
    )
    if not isinstance(state, dict):
        return {}
    stage = str(state.get("stage") or "")
    if stage not in DEFINITION_QUERY_STAGES:
        return {}
    expected_contracts = _normalized_contract_keys(contract_keys)
    if sorted(state.get("contract_keys") or []) != expected_contracts:
        return {}
    if bool(state.get("require_definition")) != bool(require_definition):
        return {}
    return dict(state)


def _next_definition_query_stage(stage: str) -> str | None:
    try:
        index = DEFINITION_QUERY_STAGES.index(stage)
    except ValueError:
        return DEFINITION_QUERY_STAGES[0]
    if index + 1 >= len(DEFINITION_QUERY_STAGES):
        return None
    return DEFINITION_QUERY_STAGES[index + 1]


def _definition_query_transition(
    stage: str,
    page: dict[str, Any],
    *,
    allow_external_provider_stage: bool = True,
) -> dict[str, Any]:
    resolved = bool(page.get("resolved"))
    unavailable = bool(page.get("unavailable"))
    if resolved or unavailable:
        return {
            "complete": True,
            "resolved": resolved,
            "unavailable": unavailable,
            "stage": stage,
            "cursor": int(page.get("total_file_count") or 0),
        }
    if not bool(page.get("complete")):
        return {
            "complete": False,
            "resolved": False,
            "unavailable": False,
            "stage": stage,
            "cursor": int(page.get("next_cursor") or 0),
        }
    next_stage = _next_definition_query_stage(stage)
    if (
        next_stage == "demand_provider_definition"
        and not allow_external_provider_stage
    ):
        next_stage = None
    return {
        "complete": next_stage is None,
        "resolved": False,
        "unavailable": False,
        "stage": next_stage or stage,
        "cursor": (
            int(page.get("total_file_count") or 0)
            if next_stage is None
            else 0
        ),
    }


def _tree_has_file(index: CodeIndex, tree: str, relative_path: str) -> bool:
    root = index.root_for_tree(tree)
    return bool(root and (root / normalize_relpath(relative_path)).is_file())


def _old_corresponding_entity_ids(index: CodeIndex, donor_entity_id: str) -> set[str]:
    candidates = {
        str(edge.get("to") or "")
        for edge in index.graph.edges
        if edge.get("from") == donor_entity_id
        and edge.get("type") == "same_contract_as"
        and str(edge.get("to") or "")
        and (index.graph.nodes.get(str(edge.get("to") or "")) or {}).get("tree") == "old"
    }
    donor = index.graph.nodes.get(donor_entity_id) or {}
    if donor.get("is_definition"):
        return {
            entity_id
            for entity_id in candidates
            if bool((index.graph.nodes.get(entity_id) or {}).get("is_definition"))
        }
    return candidates


def _candidate_from_root(index: CodeIndex, node_id: str) -> dict[str, Any] | None:
    node = index.graph.nodes.get(node_id) or {}
    tree = str(node.get("tree") or "")
    if not tree.startswith("donor_"):
        return None
    relative_path = normalize_relpath(str(node.get("file") or node.get("path") or ""))
    if not relative_path:
        return None
    kind = str(node.get("kind") or "")
    contract_key = str(node.get("contract_key") or "")
    if kind == "File":
        if _tree_has_file(index, "old", relative_path):
            return None
        evidence_edge = "candidate_resolution -> donor File absent from old"
    else:
        if _old_corresponding_entity_ids(index, node_id):
            return None
        evidence_edge = "candidate_resolution -> donor Entity without old same_contract_as"
    permission = classify_edit_permission(index.spec, relative_path)
    return {
        "file": relative_path,
        "node_id": node_id,
        "tree": tree,
        "entity_kind": node.get("entity_kind") or kind,
        "contract_key": contract_key or None,
        "permission": permission,
        "old_exists": _tree_has_file(index, "old", relative_path),
        "donor_exists": True,
        "reasons": [evidence_edge],
    }


def _owner_candidates_from_root(index: CodeIndex, node_id: str) -> list[dict[str, Any]]:
    """Map one unique explanation root to concrete old-tree write owners."""

    node = index.graph.nodes.get(node_id) or {}
    kind = str(node.get("kind") or "")
    tree = str(node.get("tree") or "")
    owner_node_ids: list[str] = []
    if tree == "old":
        owner_node_ids = [node_id]
    elif tree.startswith("donor_"):
        counterparts = sorted(_old_corresponding_entity_ids(index, node_id))
        if (
            kind == "Function"
            and node.get("is_definition") is False
            and normalize_relpath(str(node.get("file") or "")).endswith(".h")
        ):
            owner_node_ids = [node_id, *counterparts]
        else:
            owner_node_ids = counterparts or [node_id]
    elif kind == "Contract":
        old_entity_id = str(node.get("old_entity_id") or "")
        if old_entity_id:
            owner_node_ids = [old_entity_id]
    elif kind == "File":
        relative_path = normalize_relpath(str(node.get("path") or node.get("file") or ""))
        if relative_path:
            old_file_id = f"file:old:{relative_path}"
            owner_node_ids = [old_file_id if old_file_id in index.graph.nodes else node_id]

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for owner_node_id in owner_node_ids:
        owner = index.graph.nodes.get(owner_node_id) or node
        relative_path = normalize_relpath(str(owner.get("file") or owner.get("path") or ""))
        if not relative_path:
            continue
        identity = (owner_node_id, relative_path)
        if identity in seen:
            continue
        seen.add(identity)
        permission = classify_edit_permission(index.spec, relative_path)
        records.append(
            {
                "file": relative_path,
                "node_id": owner_node_id,
                "root_node_id": node_id,
                "owner_role": "root_owner",
                "permission": permission,
                "old_exists": _tree_has_file(index, "old", relative_path),
                "donor_exists": any(
                    _tree_has_file(index, tree_name, relative_path)
                    for tree_name in index.contexts
                    if tree_name.startswith("donor_")
                ),
                "reasons": [
                    "unique diagnostic explanation root mapped through old-tree entity identity"
                ],
            }
        )
    return records


def _record_owner_candidate(
    candidates: dict[tuple[str, str], dict[str, Any]],
    record: dict[str, Any],
) -> None:
    identity = (record["node_id"], record["file"])
    prior = candidates.get(identity)
    if (
        prior
        and prior.get("owner_role") == "root_owner"
        and record.get("owner_role") == "diagnostic_location_owner"
    ):
        prior["reasons"] = list(dict.fromkeys([
            *prior.get("reasons", []),
            *record.get("reasons", []),
        ]))
        return
    candidates[identity] = record


def _owner_candidates_from_diagnostic_graph(
    index: CodeIndex,
    graph: dict[str, Any],
    root_node_id: str,
    *,
    root_node_ids: Iterable[object] = (),
    provider_resolutions: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Return exact old-tree AST/file owners attached to diagnostic locations."""

    nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    selected_root_ids = sorted(
        {
            *(
                str(node_id)
                for node_id in root_node_ids
                if str(node_id)
            ),
            *([root_node_id] if root_node_id else []),
        }
    )
    shared_contract_adaptation_root = _proven_shared_contract_adaptation_root(
        graph,
        selected_root_ids,
        spec=index.spec,
        provider_resolutions=provider_resolutions,
    )
    consumer_adaptation_authority = (
        _exhausted_provider_consumer_adaptation_authority(
            graph,
            selected_root_ids,
            provider_resolutions=provider_resolutions,
            diagnostic_mappings=index.diagnostic_mappings.values(),
        )
    )
    consumer_adaptation_owner_ids = set(
        consumer_adaptation_authority.get("consumer_owner_node_ids") or []
    )
    diagnostic_node_ids = {
        node_id
        for node_id, node in nodes.items()
        if str(node.get("kind") or "") == "Diagnostic"
    }
    diagnosed_reference_ids = {
        str(edge.get("to") or "")
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("type") or "") == "diagnoses"
        and str(edge.get("from") or "") in diagnostic_node_ids
        and str(
            (
                nodes.get(str(edge.get("to") or ""))
                or {}
            ).get("kind")
            or ""
        )
        == "Reference"
    }
    owner_records: dict[str, tuple[str, str]] = {}

    def record_owner(node_id: str, role: str, reason: str) -> None:
        previous = owner_records.get(node_id)
        if previous and previous[0] == "consumer_integration_owner":
            return
        owner_records[node_id] = (role, reason)

    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source_id = str(edge.get("from") or "")
        target_id = str(edge.get("to") or "")
        edge_type = str(edge.get("type") or "")
        source_kind = str((nodes.get(source_id) or {}).get("kind") or "")
        target_node = nodes.get(target_id) or {}
        target_tree = str(target_node.get("tree") or "")
        target_kind = str(
            target_node.get("entity_kind")
            or target_node.get("kind")
            or ""
        ).lower()
        if target_tree != "old":
            continue
        if (
            edge_type == "callback_contract_consumer"
            and source_id in selected_root_ids
            and _has_concrete_source_extent(target_node)
        ):
            record_owner(
                target_id, "consumer_integration_owner",
                "exact current binding or direct call consumes the diagnosed callback contract",
            )
        elif (
            edge_type == "occurs_in"
            and source_kind == "Reference"
            and source_id in diagnosed_reference_ids
        ):
            if (
                shared_contract_adaptation_root
                or target_id in consumer_adaptation_owner_ids
            ):
                record_owner(
                    target_id,
                    "consumer_integration_owner",
                    (
                        str(consumer_adaptation_authority.get("reason") or "")
                        if target_id in consumer_adaptation_owner_ids
                        else "a complete provider query proves the shared "
                        "contract's old-tree disposition and this exact AST "
                        "owner contains the diagnosed use that must adapt"
                    ),
                )
            else:
                record_owner(
                    target_id,
                    "diagnostic_location_owner",
                    "diagnostic reference occurs inside this exact old-tree AST owner",
                )
        elif (
            edge_type == "diagnoses"
            and source_kind == "Diagnostic"
            and target_kind not in {
                "diagnostic",
                "file",
                "reference",
                "unresolvedcontract",
                "unresolvedentity",
            }
        ):
            if (
                shared_contract_adaptation_root
                or target_id in consumer_adaptation_owner_ids
            ):
                record_owner(
                    target_id,
                    "consumer_integration_owner",
                    (
                        str(consumer_adaptation_authority.get("reason") or "")
                        if target_id in consumer_adaptation_owner_ids
                        else "a complete provider query proves the shared "
                        "contract's old-tree disposition and the diagnostic "
                        "maps directly to this exact old-tree AST consumer owner"
                    ),
                )
            else:
                record_owner(
                    target_id,
                    "diagnostic_location_owner",
                    "diagnostic location maps directly to this old-tree AST owner",
                )
        elif edge_type == "occurs_at" and source_kind == "Diagnostic":
            record_owner(
                target_id,
                "diagnostic_file_owner",
                "compiler diagnostic occurs in this exact old-tree source file",
            )

    records: list[dict[str, Any]] = []
    for owner_node_id, (owner_role, reason) in sorted(owner_records.items()):
        owner = nodes.get(owner_node_id) or {}
        relative_path = normalize_relpath(str(owner.get("file") or owner.get("path") or ""))
        if not relative_path:
            continue
        record = {
            "file": relative_path,
            "node_id": owner_node_id,
            "root_node_id": root_node_id or None,
            "owner_role": owner_role,
            "permission": classify_edit_permission(index.spec, relative_path),
            "old_exists": _tree_has_file(index, "old", relative_path),
            "donor_exists": any(
                _tree_has_file(index, tree_name, relative_path)
                for tree_name in index.contexts
                if tree_name.startswith("donor_")
            ),
            "reasons": [reason],
        }
        if owner_node_id in consumer_adaptation_owner_ids:
            record["consumer_adaptation_authority"] = dict(
                consumer_adaptation_authority
            )
        records.append(record)
    records.extend(
        _include_relation_owner_candidates(
            index,
            graph,
            root_node_id,
            provider_resolutions=provider_resolutions,
        )
    )
    return records


def _direct_repair_dependency_records(
    index: CodeIndex,
    graph: dict[str, Any],
    root_node_id: str,
) -> list[dict[str, Any]]:
    """Return donor implementation dependencies attached to one selected root."""

    nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    donor_owner_ids = {
        root_node_id
        for node in [nodes.get(root_node_id) or {}]
        if str(node.get("tree") or "").startswith("donor_")
    }
    donor_owner_ids.update(
        str(edge.get("to") or "")
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("type") or "") == "same_contract_as"
        and str(edge.get("from") or "") == root_node_id
        and str((nodes.get(str(edge.get("to") or "")) or {}).get("tree") or "").startswith(
            "donor_"
        )
    )

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        owner_id = str(edge.get("from") or "")
        target_id = str(edge.get("to") or "")
        if (
            str(edge.get("type") or "") != "implementation_depends_on"
            or owner_id not in donor_owner_ids
        ):
            continue
        owner = nodes.get(owner_id) or index.graph.nodes.get(owner_id) or {}
        target = nodes.get(target_id) or index.graph.nodes.get(target_id) or {}
        target_tree = str(target.get("tree") or "")
        target_file = normalize_relpath(str(target.get("file") or target.get("path") or ""))
        if not target_tree.startswith("donor_") or not target_file:
            continue
        identity = (owner_id, target_id, str(edge.get("reference_id") or ""))
        if identity in seen:
            continue
        seen.add(identity)
        old_entity_ids = sorted(_old_corresponding_entity_ids(index, target_id))
        records.append(
            {
                "root_node_id": root_node_id,
                "owner_entity_id": owner_id,
                "owner_file": normalize_relpath(
                    str(owner.get("file") or owner.get("path") or "")
                ),
                "reference_node_id": str(edge.get("reference_id") or ""),
                "use_kind": edge.get("use_kind"),
                "dependency_scope": edge.get("dependency_scope"),
                "target_entity_id": target_id,
                "target_entity": dict(target),
                "target_file": target_file,
                "target_tree": target_tree,
                "old_corresponding_entity_ids": old_entity_ids,
            }
        )
    return records


def _repair_dependency_candidate(
    index: CodeIndex,
    dependency: dict[str, Any],
) -> dict[str, Any]:
    target = dict(dependency.get("target_entity") or {})
    relative_path = normalize_relpath(str(dependency.get("target_file") or ""))
    old_entity_ids = list(dependency.get("old_corresponding_entity_ids") or [])
    target_digest = str(target.get("digest") or "")
    old_entities = [
        index.graph.nodes.get(str(entity_id)) or {}
        for entity_id in old_entity_ids
    ]
    equivalent_old_contract = bool(
        target_digest
        and old_entities
        and any(str(entity.get("digest") or "") == target_digest for entity in old_entities)
    )
    dependency_scope = str(
        dependency.get("dependency_scope") or "unresolved"
    )
    requires_edit_scope = bool(
        not equivalent_old_contract
        and (
            dependency_scope == "local_driver"
            or not old_entity_ids
        )
    )
    return {
        "file": relative_path,
        "node_id": str(dependency.get("target_entity_id") or ""),
        "root_node_id": str(dependency.get("root_node_id") or ""),
        "owner_role": "repair_dependency_owner",
        "tree": str(dependency.get("target_tree") or ""),
        "entity_kind": target.get("entity_kind") or target.get("kind"),
        "contract_key": target.get("contract_key") or target.get("key"),
        "permission": classify_edit_permission(index.spec, relative_path),
        "old_exists": _tree_has_file(index, "old", relative_path),
        "donor_exists": True,
        "old_corresponding_entity_ids": old_entity_ids,
        "dependency_scope": dependency_scope,
        "requires_edit_scope": requires_edit_scope,
        "contract_delta": (
            "missing_old_contract"
            if not old_entity_ids
            else (
                "equivalent_old_contract"
                if equivalent_old_contract
                else "changed_old_donor_contract"
            )
        ),
        "reasons": [
            (
                "selected donor implementation directly references this entity; "
                "an old counterpart exists"
                if old_entity_ids
                else (
                    "selected donor implementation directly references this entity, "
                    "which has no old same_contract_as counterpart"
                )
            )
        ],
    }


def _provider_resolutions(
    index: CodeIndex,
    symbols: list[str],
    definition_query_state: dict[str, dict[str, dict[str, Any]]],
    query_demands: dict[str, list[str]] | None = None,
    definition_requirements: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    query_demands = dict(query_demands or {})
    definition_requirements = dict(definition_requirements or {})
    resolutions: list[dict[str, Any]] = []
    for symbol in symbols:
        contract_keys = set(query_demands.get(symbol) or [])
        definition_required = bool(
            definition_requirements.get(symbol)
        )
        canonical_records = canonical_entity_records(build_conditioned_provider_records(
            record
            for record in index.entities_by_name.get(symbol, [])
            if (
                not contract_keys
                or str(
                    record.get("contract_key")
                    or record.get("key")
                    or ""
                )
                in contract_keys
            )
        ))
        old = [
            record
            for record in canonical_records
            if record.get("tree") == "old"
            and _has_concrete_source_extent(record)
        ]
        old_definitions = [record for record in old if record.get("is_definition")]
        donor = [
            record
            for record in canonical_records
            if str(record.get("tree") or "").startswith("donor_")
            and _has_concrete_source_extent(record)
        ]
        donor_definitions = [
            record
            for record in donor
            if record.get("is_definition")
            if not _old_corresponding_entity_ids(index, str(record.get("id") or ""))
        ]
        old_query_state = dict(
            (definition_query_state.get("old") or {}).get(symbol) or {}
        )
        old_absence_proved = bool(
            old_query_state.get("complete")
            and not old_query_state.get("unavailable")
            and not old_query_state.get("resolved")
        )
        if old_definitions:
            status = "old_entity_present"
            preferred = old_definitions[0] if len(old_definitions) == 1 else None
        elif (
            definition_required
            and old_absence_proved
            and len(donor_definitions) == 1
        ):
            status = "donor_entity_missing_in_old_unique"
            preferred = donor_definitions[0]
        elif (
            definition_required
            and old_absence_proved
            and len(donor_definitions) > 1
        ):
            status = "donor_entity_missing_in_old_ambiguous"
            preferred = None
        elif definition_required:
            status = "entity_unresolved"
            preferred = None
        elif old:
            status = "old_declaration_only"
            preferred = None
        elif old_absence_proved and len(donor_definitions) == 1:
            status = "donor_entity_missing_in_old_unique"
            preferred = donor_definitions[0]
        elif old_absence_proved and len(donor_definitions) > 1:
            status = "donor_entity_missing_in_old_ambiguous"
            preferred = None
        else:
            status = "entity_unresolved"
            preferred = None
        resolutions.append(
            {
                "enabled": True,
                "backend": "code_index_v3",
                "symbol": symbol,
                "contract_keys": sorted(contract_keys),
                "status": status,
                "preferred_patch_file": (
                    normalize_relpath(str(preferred.get("file") or "")) if preferred else None
                ),
                "preferred_entity_id": preferred.get("id") if preferred else None,
                "target_entities": old,
                "donor_entities": donor,
                "old_definition_query": old_query_state,
                "old_definition_count": len(old_definitions),
                "old_declaration_only": bool(old) and not old_definitions,
                "definition_required": definition_required,
                "ambiguous": (
                    (old_absence_proved and len(donor_definitions) > 1)
                    or len(old_definitions) > 1
                ),
                "reason": (
                    "old-tree presence or absence follows the completed per-tree definition query"
                    if old_query_state
                    else "old-tree definition query has not completed"
                ),
            }
        )
    return resolutions


def _bind_unique_provider_resolutions(
    index: CodeIndex,
    resolutions: Iterable[dict[str, Any]],
) -> int:
    """Attach completed unique provider queries to unresolved graph roots."""

    bound = 0
    for resolution in resolutions:
        if (
            not isinstance(resolution, dict)
            or str(resolution.get("backend") or "")
            != "code_index_v3"
        ):
            continue
        status = str(resolution.get("status") or "")
        entity_field = (
            "target_entities"
            if status == "old_entity_present"
            else "donor_entities"
            if status == "donor_entity_missing_in_old_unique"
            else ""
        )
        preferred_id = str(resolution.get("preferred_entity_id") or "")
        symbol = str(resolution.get("symbol") or "")
        preferred = [
            entity
            for entity in resolution.get(entity_field) or []
            if isinstance(entity, dict)
            and str(entity.get("id") or entity.get("entity_id") or "")
            == preferred_id
            and str(entity.get("symbol") or entity.get("name") or "")
            == symbol
        ] if entity_field and preferred_id and symbol else []
        if len(preferred) != 1 or preferred_id not in index.graph.nodes:
            continue
        unresolved_ids = [
            node_id
            for node_id, node in index.graph.nodes.items()
            if str(node.get("kind") or "")
            in {"UnresolvedContract", "UnresolvedEntity"}
            and str(
                node.get("symbol")
                or node.get("name")
                or node.get("spelling")
                or ""
            )
            == symbol
        ]
        for unresolved_id in unresolved_ids:
            index.graph.add_edge(
                unresolved_id,
                preferred_id,
                "candidate_resolution",
                budget_exempt=True,
                resolution_status=status,
                resolution_authority=(
                    "completed_unique_provider_definition_query"
                ),
            )
            bound += 1
    return bound


def _provider_owner_candidate(
    index: CodeIndex,
    resolution: dict[str, Any],
) -> dict[str, Any] | None:
    """Promote one structurally unique provider into the repair owner set."""

    status = str(resolution.get("status") or "")
    if str(resolution.get("backend") or "") != "code_index_v3":
        return None
    if status == "old_entity_present":
        tree = "old"
        definitions = [
            record
            for record in resolution.get("target_entities") or []
            if isinstance(record, dict)
            and str(record.get("tree") or "") == "old"
            and bool(record.get("is_definition"))
        ]
        if int(resolution.get("old_definition_count") or 0) != 1:
            return None
    elif status == "donor_entity_missing_in_old_unique":
        definitions = [
            record
            for record in resolution.get("donor_entities") or []
            if isinstance(record, dict)
            and str(record.get("tree") or "").startswith("donor_")
            and bool(record.get("is_definition"))
            and not _old_corresponding_entity_ids(
                index,
                str(record.get("id") or ""),
            )
        ]
        if len(definitions) != 1:
            return None
        tree = str(definitions[0].get("tree") or "")
    else:
        return None
    preferred_id = str(resolution.get("preferred_entity_id") or "")
    preferred_file = normalize_relpath(
        str(resolution.get("preferred_patch_file") or "")
    )
    if len(definitions) != 1:
        return None
    definition = definitions[0]
    definition_id = str(definition.get("id") or "")
    definition_file = normalize_relpath(str(definition.get("file") or ""))
    if (
        not preferred_id
        or not preferred_file
        or definition_id != preferred_id
        or definition_file != preferred_file
        or not _tree_has_file(index, tree, preferred_file)
    ):
        return None
    permission = classify_edit_permission(index.spec, preferred_file)
    if permission not in {"editable", "escalation"}:
        return None
    return {
        "file": preferred_file,
        "node_id": definition_id,
        "root_node_id": definition_id,
        "owner_role": "provider_definition_owner",
        "tree": tree,
        "entity_kind": definition.get("entity_kind") or definition.get("kind"),
        "contract_key": definition.get("contract_key") or definition.get("key"),
        "permission": permission,
        "old_exists": _tree_has_file(index, "old", preferred_file),
        "donor_exists": any(
            _tree_has_file(index, tree_name, preferred_file)
            for tree_name in index.contexts
            if tree_name.startswith("donor_")
        ),
        "reasons": [
            (
                "bounded CodeIndex resolution found exactly one old-tree "
                "provider definition"
                if tree == "old"
                else "bounded CodeIndex resolution found exactly one donor "
                "provider definition and proved it absent from the old tree"
            )
        ],
        "allowed_for_entity_materialization": bool(
            tree.startswith("donor_")
            and not _tree_has_file(index, "old", preferred_file)
        ),
    }


def _provider_contract_owner_candidates(
    index: CodeIndex,
    resolution: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return exact writable owners for one uniquely resolved provider contract."""

    status = str(resolution.get("status") or "")
    if (
        str(resolution.get("backend") or "") != "code_index_v3"
        or status
        not in {"old_entity_present", "donor_entity_missing_in_old_unique"}
    ):
        return []

    old_entities = [
        dict(record)
        for record in resolution.get("target_entities") or []
        if isinstance(record, dict)
        and str(record.get("tree") or "") == "old"
    ]
    selected_entities = (
        old_entities
        if status == "old_entity_present"
        else [
            dict(record)
            for record in resolution.get("donor_entities") or []
            if isinstance(record, dict)
            and str(record.get("tree") or "").startswith("donor_")
        ]
    )
    definitions = [
        record
        for record in selected_entities
        if bool(record.get("is_definition"))
        and (
            status == "old_entity_present"
            or not _old_corresponding_entity_ids(
                index,
                str(record.get("id") or ""),
            )
        )
    ]
    if len(definitions) != 1:
        return []
    definition = definitions[0]
    definition_id = str(definition.get("id") or "")
    contract_key = str(
        definition.get("contract_key") or definition.get("key") or ""
    )
    symbol = str(
        definition.get("symbol") or definition.get("name") or ""
    )
    signature_fingerprint = str(
        definition.get("signature_fingerprint") or ""
    )
    if (
        not definition_id
        or definition_id != str(resolution.get("preferred_entity_id") or "")
        or not contract_key
        or not symbol
    ):
        return []

    records: list[dict[str, Any]] = []
    definition_owner = _provider_owner_candidate(index, resolution)
    if definition_owner is not None:
        records.append(definition_owner)

    old_declaration_files: set[str] = set()
    declaration_entities = [
        *[
            record
            for record in old_entities
            if not bool(record.get("is_definition"))
        ],
        *[
            dict(record)
            for record in resolution.get("donor_entities") or []
            if isinstance(record, dict)
            and str(record.get("tree") or "").startswith("donor_")
            and not bool(record.get("is_definition"))
        ],
    ]
    for declaration in declaration_entities:
        declaration_key = str(
            declaration.get("contract_key") or declaration.get("key") or ""
        )
        declaration_symbol = str(
            declaration.get("symbol") or declaration.get("name") or ""
        )
        declaration_signature = str(
            declaration.get("signature_fingerprint") or ""
        )
        if (
            declaration_key != contract_key
            or declaration_symbol != symbol
            or (
                signature_fingerprint
                and declaration_signature
                and declaration_signature != signature_fingerprint
            )
        ):
            continue
        declaration_id = str(declaration.get("id") or "")
        declaration_file = normalize_relpath(
            str(declaration.get("file") or "")
        )
        declaration_tree = str(declaration.get("tree") or "")
        if (
            declaration_tree.startswith("donor_")
            and declaration.get("token_count") == 0
        ):
            continue
        if (
            not declaration_id
            or not declaration_file
            or not _tree_has_file(index, "old", declaration_file)
        ):
            continue
        permission = classify_edit_permission(index.spec, declaration_file)
        if permission not in {"editable", "escalation"}:
            continue
        old_counterparts = sorted(
            entity_id
            for entity_id in _old_corresponding_entity_ids(index, declaration_id)
            if not bool(
                (index.graph.nodes.get(entity_id) or {}).get("is_definition")
            )
        )
        if declaration_tree == "old":
            old_declaration_files.add(declaration_file)
        elif old_counterparts or declaration_file in old_declaration_files:
            continue
        records.append(
            {
                "file": declaration_file,
                "node_id": declaration_id,
                "root_node_id": definition_id,
                "owner_role": "provider_declaration_owner",
                "tree": declaration_tree,
                "entity_kind": (
                    declaration.get("entity_kind")
                    or declaration.get("kind")
                ),
                "contract_key": declaration_key,
                "symbol": declaration_symbol,
                "is_definition": False,
                "signature_fingerprint": declaration_signature or None,
                "permission": permission,
                "old_exists": True,
                "donor_exists": any(
                    _tree_has_file(index, tree_name, declaration_file)
                    for tree_name in index.contexts
                    if tree_name.startswith("donor_")
                ),
                "old_corresponding_entity_ids": old_counterparts,
                "exact_absent_declaration_insertion_allowed": bool(
                    declaration_tree.startswith("donor_")
                    and not old_counterparts
                ),
                "reasons": [
                    (
                        "same-contract old-tree declaration participates in "
                        "the uniquely resolved provider contract"
                        if declaration_tree == "old"
                        else "same-contract donor declaration has no old-tree "
                        "counterpart and its owner file already exists"
                    )
                ],
            }
        )
    return records


def _read_only_records(
    index: CodeIndex,
    files: Iterable[str],
    candidate_files: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative_path in _ordered_unique(files):
        if relative_path in candidate_files:
            continue
        records.append(
            {
                "file": relative_path,
                "permission": classify_edit_permission(index.spec, relative_path),
                "old_exists": _tree_has_file(index, "old", relative_path),
                "donor_exists": any(
                    _tree_has_file(index, tree, relative_path)
                    for tree in index.contexts
                    if tree.startswith("donor_")
                ),
                "reasons": ["file node lies in the indexed diagnostic explanation graph"],
            }
        )
    return records


def _retained_graph_snapshot(
    index: CodeIndex,
    *evidence_payloads: object,
) -> dict[str, Any]:
    """Persist only graph nodes that participate in a repair decision."""

    retained_ids: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, str):
            if value in index.graph.nodes:
                retained_ids.add(value)
            return
        if isinstance(value, dict):
            for nested in value.values():
                collect(nested)
            return
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                collect(nested)

    for payload in evidence_payloads:
        collect(payload)

    # A retained structural entity also needs its concrete owner file. This is
    # directional: retaining a file must not pull every entity in that file.
    for node_id in list(retained_ids):
        node = index.graph.nodes.get(node_id) or {}
        tree = str(node.get("tree") or "")
        relative_path = normalize_relpath(
            str(node.get("file") or node.get("path") or "")
        )
        file_id = f"file:{tree}:{relative_path}"
        if tree and relative_path and file_id in index.graph.nodes:
            retained_ids.add(file_id)

    nodes = [
        dict(index.graph.nodes[node_id])
        for node_id in sorted(retained_ids)
        if node_id in index.graph.nodes
    ]
    edges = [
        dict(edge)
        for edge in index.graph.edges
        if str(edge.get("from") or "") in retained_ids
        and str(edge.get("to") or "") in retained_ids
    ]
    return {
        "schema": "typed_code_dependency_graph_repair_slice_v1",
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "truncated": False,
            "truncation_reasons": [],
        },
    }


def _paginate_candidate_records(
    discovered_records: Iterable[dict[str, Any]],
    *,
    candidate_files: Iterable[str],
    prior_records: Iterable[dict[str, Any]],
    page_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return accumulated, current-page, and remaining graph candidates."""

    existing = set(_ordered_unique(candidate_files))
    accumulated_by_file = {
        normalize_relpath(str(record.get("file") or "")): dict(record)
        for record in prior_records
        if isinstance(record, dict) and normalize_relpath(str(record.get("file") or ""))
    }
    pending = [
        dict(record)
        for record in discovered_records
        if normalize_relpath(str(record.get("file") or "")) not in existing
        and normalize_relpath(str(record.get("file") or "")) not in accumulated_by_file
    ]
    page = pending[: max(0, int(page_size))]
    for record in page:
        relative_path = normalize_relpath(str(record.get("file") or ""))
        record["file"] = relative_path
        accumulated_by_file[relative_path] = record
    accumulated = [accumulated_by_file[path] for path in sorted(accumulated_by_file)]
    return accumulated, page, pending[len(page) :]


def expand_scope_from_code_graph(
    spec: CaseSpec,
    work_unit: dict[str, Any],
    candidate_files: list[str],
    *,
    traversal_cursor: dict[str, Any] | None = None,
    prior_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expand scope only through typed CodeIndex paths and resumable AST queries."""

    started_at = time.monotonic()
    issues = _work_unit_issues(work_unit)
    seeds = _seed_files(spec, work_unit, candidate_files)
    cursor_by_tree = dict(traversal_cursor or {})
    index = CodeIndex(
        spec,
        budget=GraphBudget(max_nodes=MAX_GRAPH_NODES, max_edges=MAX_GRAPH_EDGES),
    )
    prior_scope = dict(prior_scope or {})
    prior_summary = dict(prior_scope.get("summary") or {})
    index.import_graph_snapshot(
        nodes=[item for item in prior_scope.get("nodes") or [] if isinstance(item, dict)],
        edges=[item for item in prior_scope.get("edges") or [] if isinstance(item, dict)],
        truncated=bool(prior_summary.get("persisted_graph_truncated")),
        truncation_reasons=prior_summary.get("persisted_graph_truncation_reasons")
        or [],
    )
    index_ready_at = time.monotonic()

    for tree in _tree_names(spec):
        index.index_files(
            tree,
            [
                relative_path
                for relative_path in _seed_files_for_tree(
                    spec, work_unit, candidate_files, tree
                )
                if _tree_has_file(index, tree, relative_path)
            ],
        )
        index.index_kbuild_directory(tree, spec.driver_root)

    index.add_diagnostics(issues)
    query_demand_records = _graph_query_demand_records(
        index,
        work_unit,
    )
    query_demands = {
        symbol: list(record["contract_keys"])
        for symbol, record in query_demand_records.items()
    }
    definition_requirements = {
        symbol: bool(record["require_definition"])
        for symbol, record in query_demand_records.items()
    }
    query_demands, definition_requirements = (
        _definition_query_demands_with_history(
            query_demands,
            definition_requirements,
            prior_scope,
        )
    )
    symbols = list(query_demands)
    seed_index_ready_at = time.monotonic()
    bounded_candidate_reports: list[dict[str, Any]] = []
    compile_frontier_reports: dict[str, dict[str, dict[str, Any]]] = {}
    exact_provider_reports: dict[str, dict[str, Any]] = {}
    query_pages: list[dict[str, Any]] = []
    definition_query_state: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    candidate_discovery_seconds = 0.0
    definition_query_seconds = 0.0
    if symbols:
        for tree in _tree_names(spec):
            for symbol in symbols:
                prior_state = _definition_query_prior_state(
                    prior_scope,
                    tree=tree,
                    symbol=symbol,
                    contract_keys=query_demands.get(symbol) or [],
                    require_definition=bool(
                        definition_requirements.get(symbol)
                    ),
                )
                if prior_state.get("complete"):
                    definition_query_state[tree][symbol] = {
                        **prior_state,
                        "require_definition": bool(
                            definition_requirements.get(symbol)
                        ),
                    }
                    continue
                stage = str(
                    prior_state.get("stage")
                    or DEFINITION_QUERY_STAGES[0]
                )
                cursor = int(
                    prior_state.get("cursor")
                    if prior_state.get("cursor") is not None
                    else _definition_query_start(
                        cursor_by_tree,
                        tree=tree,
                        symbol=symbol,
                    )
                )
                attempted_files = list(
                    prior_state.get("attempted_files") or []
                )
                discovery_started = time.monotonic()
                provider_discovery_unavailable = False
                if stage in {
                    "case_compile_frontier",
                    "case_dependency_frontier",
                }:
                    if stage == "case_compile_frontier":
                        frontier_kind = "driver_root"
                        within_root = spec.driver_root
                        candidate_source = (
                            "case_compile_frontier_exact_token"
                        )
                    else:
                        frontier_kind = "all_reachable_includes"
                        within_root = ""
                        candidate_source = (
                            "case_dependency_frontier_exact_token"
                        )
                    frontier = compile_frontier_reports.get(tree, {}).get(
                        frontier_kind
                    )
                    if frontier is None:
                        frontier = index.compile_dependency_frontier(
                            tree,
                            within_root=within_root,
                        )
                        compile_frontier_reports.setdefault(tree, {})[
                            frontier_kind
                        ] = frontier
                    bounded = index.exact_token_candidates_in_files(
                        tree,
                        [symbol],
                        frontier.get("files") or [],
                    )
                    bounded_candidate_reports.append(bounded)
                    query_candidates = (
                        (bounded.get("candidates_by_symbol") or {}).get(
                            symbol,
                            [],
                        )
                    )
                elif stage == "demand_provider_definition":
                    candidate_source = "demand_provider_exact_token"
                    provider_report = exact_provider_reports.get(tree)
                    if provider_report is None:
                        provider_symbols = [
                            candidate_symbol
                            for candidate_symbol in symbols
                            if _requires_external_provider_definition(
                                query_demands.get(candidate_symbol) or [],
                                require_definition=bool(
                                    definition_requirements.get(
                                        candidate_symbol
                                    )
                                ),
                            )
                        ]
                        provider_report = (
                            index.exact_build_owned_definition_candidates(
                                tree,
                                provider_symbols,
                            )
                        )
                        exact_provider_reports[tree] = provider_report
                        bounded_candidate_reports.append(provider_report)
                        for provider_file in (
                            provider_report.get(
                                "build_owned_candidate_files"
                            )
                            or []
                        ):
                            index.index_kbuild_directory(
                                tree,
                                normalize_relpath(
                                    str(Path(provider_file).parent.as_posix())
                                ),
                            )
                    provider_discovery_unavailable = (
                        str(provider_report.get("status") or "")
                        != "ready"
                    )
                    query_candidates = (
                        (provider_report.get("candidates_by_symbol") or {}).get(
                            symbol,
                            [],
                        )
                    )
                else:
                    raise ValueError(
                        f"unsupported definition query stage: {stage}"
                    )
                candidate_discovery_seconds += (
                    time.monotonic() - discovery_started
                )
                query_started = time.monotonic()
                if provider_discovery_unavailable:
                    page = {
                        "schema": "code_index_query_page_v3",
                        "tree": tree,
                        "symbols": [symbol],
                        "contract_keys": list(
                            query_demands.get(symbol) or []
                        ),
                        "cursor": cursor,
                        "next_cursor": None,
                        "scanned_file_count": 0,
                        "skipped_full_parse_count": 0,
                        "total_file_count": 0,
                        "complete": True,
                        "truncated": False,
                        "unavailable": True,
                        "unresolved_reason": provider_report.get("reason"),
                        "matches": [],
                        "resolution_source": candidate_source,
                        "candidate_source": candidate_source,
                        "attempted_files": attempted_files,
                        "resolved": False,
                    }
                else:
                    page = index.query_definition_batches(
                        tree=tree,
                        symbols=[symbol],
                        cursor=cursor,
                        max_files=MAX_DEFINITION_QUERY_FILES,
                        candidate_files=query_candidates,
                        candidate_source=candidate_source,
                        previously_attempted_files=attempted_files,
                        contract_keys=query_demands.get(symbol) or [],
                        require_definition=bool(
                            definition_requirements.get(symbol)
                        ),
                        exhaustive_candidate_scan=(
                            stage == "demand_provider_definition"
                        ),
                        query_only_files=(
                            [
                                path
                                for path in query_candidates
                                if is_shared_kernel_implementation_path(path)
                            ]
                            if tree.startswith("donor_")
                            else []
                        ),
                        definition_probe_files=(
                            query_candidates
                            if tree.startswith("donor_")
                            and any(
                                str(contract).startswith("function:")
                                for contract in query_demands.get(symbol) or []
                            )
                            else []
                        ),
                    )
                definition_query_seconds += time.monotonic() - query_started
                page["query_symbol"] = symbol
                page["query_stage"] = stage
                transition = _definition_query_transition(
                    stage,
                    page,
                    allow_external_provider_stage=(
                        _requires_external_provider_definition(
                            query_demands.get(symbol) or [],
                            require_definition=bool(
                                definition_requirements.get(symbol)
                            ),
                        )
                    ),
                )
                page["stage_complete"] = bool(page.get("complete"))
                page["next_query_stage"] = transition["stage"]
                page["query_complete"] = transition["complete"]
                definition_query_state[tree][symbol] = {
                    **transition,
                    "contract_keys": list(
                        query_demands.get(symbol) or []
                    ),
                    "require_definition": bool(
                        definition_requirements.get(symbol)
                    ),
                    "attempted_files": list(
                        page.get("attempted_files") or attempted_files
                    ),
                }
                query_pages.append(page)
    definition_query_ready_at = time.monotonic()
    provider_resolutions = _provider_resolutions(
        index,
        symbols,
        definition_query_state,
        query_demands,
        definition_requirements,
    )
    _bind_unique_provider_resolutions(index, provider_resolutions)
    # Re-map after query pages add entity, counterpart, and resolution edges.
    index.add_diagnostics(issues)
    components = index.explanation_components(issues)

    explanation_graph_files: list[str] = []
    component_graphs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    explanation_nodes_by_id: dict[str, dict[str, Any]] = {}
    explanation_edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    repair_dependencies: list[dict[str, Any]] = []
    repair_dependency_alignments: list[dict[str, Any]] = []
    for component in components:
        component_graph = index.diagnostic_graph_dict(
            list(component.get("issues") or []),
            selected_root=dict(component.get("root") or {}),
        )
        component_graphs.append((component, component_graph))
        root = component.get("root") if isinstance(component.get("root"), dict) else {}
        root_node_ids = mapped_root_alternative_ids(root)
        if not root_node_ids:
            root_node_id = str(
                root.get("node_id") or root.get("entity_id") or ""
            )
            root_node_ids = [root_node_id] if root_node_id else []
        component_dependencies = [
            dependency
            for root_node_id in root_node_ids
            for dependency in _direct_repair_dependency_records(
                index,
                component_graph,
                root_node_id,
            )
        ]
        repair_dependencies.extend(component_dependencies)
        for dependency in component_dependencies:
            if dependency.get("old_corresponding_entity_ids"):
                continue
            target_file = normalize_relpath(str(dependency.get("target_file") or ""))
            donor_tree = str(dependency.get("target_tree") or "")
            if (
                not target_file
                or not _tree_has_file(index, "old", target_file)
                or not _tree_has_file(index, donor_tree, target_file)
            ):
                continue
            alignment = index._owner_entity_alignment(
                donor_tree=donor_tree,
                source_file=str(dependency.get("owner_file") or ""),
                target_entity_id=str(dependency.get("target_entity_id") or ""),
                target_file=target_file,
            )
            repair_dependency_alignments.append(
                {
                    **dependency,
                    "contract_owner_file": target_file,
                    "owner_alignment": alignment,
                }
            )
        for node in component_graph.get("nodes") or []:
            if isinstance(node, dict) and node.get("id"):
                explanation_nodes_by_id[str(node["id"])] = dict(node)
        for edge in component_graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            key = (
                str(edge.get("from") or ""),
                str(edge.get("to") or ""),
                str(edge.get("type") or ""),
            )
            explanation_edges_by_key[key] = dict(edge)
        explanation_graph_files.extend(
            normalize_relpath(str(node.get("path") or node.get("file") or ""))
            for node in component_graph.get("nodes") or []
            if isinstance(node, dict)
            and (node.get("path") or node.get("file"))
        )

    root_node_ids = {
        node_id
        for event_id in index.diagnostic_roots
        for node_id in index.upstream_root_nodes(event_id)
    }
    discovered_by_file: dict[str, dict[str, Any]] = {}
    for node_id in sorted(root_node_ids):
        candidate = _candidate_from_root(index, node_id)
        if candidate is None:
            continue
        current = discovered_by_file.get(candidate["file"])
        if current is None:
            discovered_by_file[candidate["file"]] = candidate
            continue
        current["reasons"] = sorted(set(current["reasons"] + candidate["reasons"]))

    prior_discovered_records = [
        dict(record)
        for record in prior_scope.get("discovered_candidate_records") or []
        if isinstance(record, dict) and record.get("file")
        and str(record.get("owner_role") or "")
        != "repair_dependency_owner"
    ]
    for record in prior_discovered_records:
        relative_path = normalize_relpath(str(record.get("file") or ""))
        if not relative_path:
            continue
        current = discovered_by_file.get(relative_path)
        if current is None:
            discovered_by_file[relative_path] = record
        else:
            current["reasons"] = sorted(
                set(list(current.get("reasons") or []) + list(record.get("reasons") or []))
            )
    discovered_records = [discovered_by_file[path] for path in sorted(discovered_by_file)]
    existing = set(_ordered_unique(candidate_files))
    prior_extra_candidate_records = [
        dict(record)
        for record in prior_scope.get("extra_candidate_records") or []
        if isinstance(record, dict)
        and str(record.get("owner_role") or "")
        != "repair_dependency_owner"
    ]
    extra_records, candidate_page, remaining_candidate_records = _paginate_candidate_records(
        discovered_records,
        candidate_files=candidate_files,
        prior_records=prior_extra_candidate_records,
        page_size=MAX_EXTRA_CANDIDATES,
    )
    candidate_output_truncated = bool(remaining_candidate_records)
    candidate_set = (
        {record["file"] for record in discovered_records}
        | existing
        | {record["file"] for record in extra_records}
    )

    all_read_only_records = _read_only_records(
        index,
        explanation_graph_files,
        candidate_set,
    )
    prior_read_only_records = [
        dict(record)
        for record in prior_scope.get("read_only_evidence_records") or []
        if isinstance(record, dict) and record.get("file")
    ]
    prior_read_only_files = {
        normalize_relpath(str(record.get("file") or ""))
        for record in prior_read_only_records
    }
    pending_read_only_records = [
        record
        for record in all_read_only_records
        if normalize_relpath(str(record.get("file") or "")) not in prior_read_only_files
    ]
    read_only_page = pending_read_only_records[:MAX_READ_ONLY_EVIDENCE]
    read_only_records = [*prior_read_only_records, *read_only_page]
    read_only_output_truncated = len(pending_read_only_records) > len(read_only_page)
    pending_cursor_by_tree_symbol: dict[str, dict[str, int]] = defaultdict(dict)
    cursor_by_tree_symbol: dict[str, dict[str, int]] = defaultdict(dict)
    stage_by_tree_symbol: dict[str, dict[str, str]] = defaultdict(dict)
    pending_stage_by_tree_symbol: dict[str, dict[str, str]] = defaultdict(dict)
    for tree, symbol_states in definition_query_state.items():
        for symbol, state in symbol_states.items():
            cursor_value = int(state.get("cursor") or 0)
            cursor_by_tree_symbol[tree][symbol] = cursor_value
            stage_by_tree_symbol[tree][symbol] = str(
                state.get("stage") or DEFINITION_QUERY_STAGES[0]
            )
            if not bool(state.get("complete")):
                pending_cursor_by_tree_symbol[tree][symbol] = cursor_value
                pending_stage_by_tree_symbol[tree][symbol] = str(
                    state.get("stage") or DEFINITION_QUERY_STAGES[0]
                )
    cursor_state = {
        tree: min(symbol_cursors.values()) if symbol_cursors else 0
        for tree, symbol_cursors in cursor_by_tree_symbol.items()
    }
    pending_cursor = {
        tree: min(symbol_cursors.values()) if symbol_cursors else 0
        for tree, symbol_cursors in pending_cursor_by_tree_symbol.items()
        if symbol_cursors
    }
    unavailable_trees = sorted(
        {str(page["tree"]) for page in query_pages if page.get("unavailable")}
    )
    unresolved_frontier_ids = sorted(
        node_id
        for node_id in root_node_ids
        if str((index.graph.nodes.get(node_id) or {}).get("kind") or "").startswith("Unresolved")
    )
    query_traversal_pending = any(
        not bool(state.get("complete"))
        for symbol_states in definition_query_state.values()
        for state in symbol_states.values()
    )
    traversal_resumable = bool(
        query_traversal_pending
        or candidate_output_truncated
        or read_only_output_truncated
    )
    traversal_complete = not traversal_resumable
    continuation = {
        "schema": "code_index_scope_cursor_v6",
        "complete": traversal_complete,
        "cursor_by_tree": cursor_state,
        "pending_cursor_by_tree": pending_cursor,
        "cursor_by_tree_symbol": {
            tree: dict(sorted(symbol_cursors.items()))
            for tree, symbol_cursors in sorted(cursor_by_tree_symbol.items())
        },
        "pending_cursor_by_tree_symbol": {
            tree: dict(sorted(symbol_cursors.items()))
            for tree, symbol_cursors in sorted(pending_cursor_by_tree_symbol.items())
            if symbol_cursors
        },
        "stage_by_tree_symbol": {
            tree: dict(sorted(symbol_stages.items()))
            for tree, symbol_stages in sorted(stage_by_tree_symbol.items())
        },
        "pending_stage_by_tree_symbol": {
            tree: dict(sorted(symbol_stages.items()))
            for tree, symbol_stages in sorted(
                pending_stage_by_tree_symbol.items()
            )
            if symbol_stages
        },
        "complete_trees": sorted(
            tree
            for tree in _tree_names(spec)
            if symbols
            and all(
                bool(
                    (definition_query_state.get(tree, {}).get(symbol) or {}).get(
                        "complete"
                    )
                )
                for symbol in symbols
            )
        ),
        "unresolved_frontier_node_ids": unresolved_frontier_ids,
        "terminal_unresolved_frontier_node_ids": (
            unresolved_frontier_ids if traversal_complete else []
        ),
        "expected_edge_kinds": [
            "candidate_resolution",
            "same_contract_as",
            "defined_in",
            "resolves_to",
            "implementation_depends_on",
        ],
        "unavailable_trees": unavailable_trees,
        "terminal_graph_budget_truncation": bool(
            traversal_complete and index.graph.truncated
        ),
        "remaining_candidate_files": [
            record["file"] for record in remaining_candidate_records
        ],
        "remaining_read_only_files": [
            record["file"] for record in pending_read_only_records[len(read_only_page) :]
        ],
    }

    root_clusters = [dict(component.get("root") or {}) for component in components]
    owner_candidates_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    external_interface_roots: dict[str, set[str]] = defaultdict(set)
    selected_work_unit_root = (
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    selected_work_unit_root_id = str(
        selected_work_unit_root.get("node_id")
        or selected_work_unit_root.get("entity_id")
        or ""
    ) if isinstance(selected_work_unit_root, dict) else ""
    for owner in work_unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        interface = dict(owner.get("driver_interface_ownership") or {})
        if interface.get("status") != "external_providers_outside_selected_entries":
            continue
        interface_root_id = str(
            owner.get("owner_entity_id")
            or selected_work_unit_root_id
            or ""
        )
        if not interface_root_id:
            continue
        for declaration in interface.get("unmatched_declarations") or []:
            if not isinstance(declaration, dict):
                continue
            symbol = str(declaration.get("symbol") or "").strip()
            if symbol:
                external_interface_roots[symbol].add(interface_root_id)
    for component, component_graph in component_graphs:
        root = component.get("root") if isinstance(component.get("root"), dict) else {}
        root_node_ids = mapped_root_alternative_ids(root)
        if not root_node_ids:
            root_node_id = str(
                root.get("node_id") or root.get("entity_id") or ""
            )
            root_node_ids = [root_node_id] if root_node_id else []
        root_records = [
            record
            for root_node_id in root_node_ids
            for record in _owner_candidates_from_root(
                index,
                root_node_id,
            )
        ] if not root.get("unmapped") else []
        location_records = _owner_candidates_from_diagnostic_graph(
            index,
            component_graph,
            "" if len(root_node_ids) != 1 else root_node_ids[0],
            root_node_ids=root_node_ids,
            provider_resolutions=provider_resolutions,
        )
        for record in [*root_records, *location_records]:
            if record["permission"] not in {"editable", "escalation"}:
                continue
            _record_owner_candidate(owner_candidates_by_identity, record)
    for dependency in repair_dependencies:
        record = _repair_dependency_candidate(index, dependency)
        if (
            not record.get("requires_edit_scope")
            or record["permission"] not in {"editable", "escalation"}
        ):
            continue
        owner_candidates_by_identity[(record["node_id"], record["file"])] = record
    for resolution in provider_resolutions:
        for record in _provider_contract_owner_candidates(index, resolution):
            symbol_roots = external_interface_roots.get(
                str(resolution.get("symbol") or ""),
                set(),
            )
            if len(symbol_roots) == 1:
                record = {
                    **record,
                    "root_node_id": next(iter(symbol_roots)),
                }
            owner_candidates_by_identity[(record["node_id"], record["file"])] = record
    owner_candidates = [
        owner_candidates_by_identity[key]
        for key in sorted(owner_candidates_by_identity)
    ]
    explanation_graph = {
        "schema": "diagnostic_code_dependency_graph_v3",
        "nodes": [explanation_nodes_by_id[key] for key in sorted(explanation_nodes_by_id)],
        "edges": [explanation_edges_by_key[key] for key in sorted(explanation_edges_by_key)],
        "diagnostic_mappings": list(index.diagnostic_mappings.values()),
        "summary": {
            "diagnostic_count": len(issues),
            "mapped_diagnostic_count": sum(
                not item.get("unmapped") for item in index.diagnostic_mappings.values()
            ),
            "unmapped_diagnostic_count": sum(
                bool(item.get("unmapped")) for item in index.diagnostic_mappings.values()
            ),
            "node_count": len(explanation_nodes_by_id),
            "edge_count": len(explanation_edges_by_key),
            "root_cause_cluster_count": len(root_clusters),
            "truncated": index.graph.truncated,
            "truncation_reasons": sorted(index.graph.truncation_reasons),
        },
    }
    internal_graph_payload = index.graph.to_dict()
    graph_payload = _retained_graph_snapshot(
        index,
        explanation_graph,
        list(index.diagnostic_mappings.values()),
        provider_resolutions,
        owner_candidates,
        repair_dependencies,
        repair_dependency_alignments,
        discovered_records,
        root_clusters,
    )
    completed_at = time.monotonic()
    parse_reports = list(index.parse_reports.values())
    return {
        "schema": "code_index_scope_expansion_v4",
        "enabled": True,
        "input": {
            "diagnostic_event_ids": sorted(
                str(issue.get("diagnostic_event_id") or "") for issue in issues if issue.get("diagnostic_event_id")
            ),
            "seed_files": seeds,
            "symbols": symbols,
            "candidate_files": _ordered_unique(candidate_files),
            "traversal_cursor": cursor_by_tree,
        },
        "summary": {
            "seed_file_count": len(seeds),
            "symbol_count": len(symbols),
            "query_page_count": len(query_pages),
            "definition_query_file_visit_count": sum(
                int(page.get("scanned_file_count") or 0) for page in query_pages
            ),
            "definition_query_skipped_full_parse_count": sum(
                int(page.get("skipped_full_parse_count") or 0)
                for page in query_pages
            ),
            "definition_query_exact_token_candidate_count": sum(
                int(page.get("lexical_candidate_count") or 0)
                for page in query_pages
            ),
            "bounded_frontier_search_file_count": sum(
                int(report.get("searched_file_count") or 0)
                for report in bounded_candidate_reports
            ),
            "pending_definition_query_count": sum(
                not bool(state.get("complete"))
                for symbol_states in definition_query_state.values()
                for state in symbol_states.values()
            ),
            "discovered_candidate_count": len(discovered_records),
            "extra_candidate_count": len(extra_records),
            "new_extra_candidate_count": len(candidate_page),
            "read_only_evidence_count": len(read_only_records),
            "provider_resolution_count": len(provider_resolutions),
            "direct_repair_dependency_count": len(repair_dependencies),
            "repair_dependency_alignment_count": len(repair_dependency_alignments),
            "root_cause_cluster_count": len(root_clusters),
            "truncated": traversal_resumable,
            "graph_budget_truncated": index.graph.truncated,
            "internal_graph_node_count": int(
                (internal_graph_payload.get("summary") or {}).get(
                    "node_count"
                )
                or 0
            ),
            "internal_graph_edge_count": int(
                (internal_graph_payload.get("summary") or {}).get(
                    "edge_count"
                )
                or 0
            ),
            "persisted_graph_node_count": len(graph_payload["nodes"]),
            "persisted_graph_edge_count": len(graph_payload["edges"]),
            "persisted_graph_truncated": bool(
                (graph_payload.get("summary") or {}).get("truncated")
            ),
            "persisted_graph_truncation_reasons": list(
                (graph_payload.get("summary") or {}).get(
                    "truncation_reasons"
                )
                or []
            ),
            "candidate_output_truncated": candidate_output_truncated,
            "read_only_output_truncated": read_only_output_truncated,
            "resumable": traversal_resumable,
            "complete": traversal_complete,
        },
        "timings_sec": {
            "index_restore": round(index_ready_at - started_at, 6),
            "seed_index": round(seed_index_ready_at - index_ready_at, 6),
            "candidate_discovery": round(
                candidate_discovery_seconds,
                6,
            ),
            "definition_query": round(definition_query_seconds, 6),
            "graph_assembly": round(
                completed_at - definition_query_ready_at,
                6,
            ),
            "libclang_parse_report_sum": round(
                sum(float(report.get("parse_seconds") or 0.0) for report in parse_reports),
                6,
            ),
            "total": round(completed_at - started_at, 6),
        },
        "nodes": graph_payload["nodes"],
        "edges": graph_payload["edges"],
        "diagnostic_mappings": list(index.diagnostic_mappings.values()),
        "explanation_graph": explanation_graph,
        "query_pages": query_pages,
        "bounded_candidate_batches": bounded_candidate_reports,
        "compile_dependency_frontiers": {
            tree: {
                frontier_kind: dict(report)
                for frontier_kind, report in sorted(frontiers.items())
            }
            for tree, frontiers in sorted(compile_frontier_reports.items())
        },
        "definition_query_state": {
            tree: {
                symbol: dict(state)
                for symbol, state in sorted(symbol_states.items())
            }
            for tree, symbol_states in sorted(
                definition_query_state.items()
            )
        },
        "continuation": continuation,
        "discovered_candidate_files": [record["file"] for record in discovered_records],
        "discovered_candidate_records": discovered_records,
        "extra_candidate_files": [record["file"] for record in extra_records],
        "extra_candidate_records": extra_records,
        "read_only_evidence_files": [record["file"] for record in read_only_records],
        "read_only_evidence_records": read_only_records,
        "owner_candidates": owner_candidates,
        "repair_dependencies": repair_dependencies,
        "repair_dependency_alignments": repair_dependency_alignments,
        "provider_resolutions": provider_resolutions,
        "root_cause_clusters": root_clusters,
        "root_cause_sources": [
            {
                "diagnostic_event_ids": list(component.get("diagnostic_event_ids") or []),
                "root_node_ids": list(component.get("common_root_node_ids") or []),
            }
            for component in components
        ],
    }
