#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths as _ordered_unique,
    stable_id,
)
from driver_backport.planning.code_graph_scope import expand_scope_from_code_graph
from driver_backport.planning.root_binding import bind_graph_root_identity
from driver_backport.planning.work_unit import (
    exact_absent_declaration_insertion_owner,
    exact_missing_include_materialization_owner,
    repair_owner_has_action_authority,
    repair_owner_patch_files,
    unsafe_automatic_scope_owner,
)


def _owner_patch_ready(owner: dict[str, Any]) -> bool:
    """Patchability is a graph/permission fact, never a confidence threshold."""

    return (
        bool(owner.get("owner_file"))
        and bool(owner.get("allowed_as_patch_target"))
        and str(owner.get("owner_kind") or "") != "unlocalized"
    )


def graph_proven_editable_diagnostic_owner_files(
    work_unit: dict[str, Any],
) -> list[str]:
    """Return exact writable AST owners proven by diagnostic attribution."""

    selected_root_ids = {
        str(node_id)
        for node_id in work_unit.get("dependency_root_node_ids") or []
        if str(node_id)
    }
    for field in (
        "code_graph_root_cause_cluster",
        "root_cause_cluster",
    ):
        root = work_unit.get(field)
        if not isinstance(root, dict):
            continue
        for candidate in [
            root,
            *[
                item
                for item in root.get("alternatives") or []
                if isinstance(item, dict)
            ],
        ]:
            node_id = str(
                candidate.get("node_id")
                or candidate.get("entity_id")
                or ""
            )
            if node_id:
                selected_root_ids.add(node_id)

    graph = (
        work_unit.get("diagnostic_graph")
        if isinstance(work_unit.get("diagnostic_graph"), dict)
        else {}
    )
    nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    edges = [
        edge
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
    ]
    diagnostic_ids = {
        node_id
        for node_id, node in nodes.items()
        if str(node.get("kind") or "") == "Diagnostic"
    }
    exact_reference_ids = {
        str(edge.get("to") or "")
        for edge in edges
        if str(edge.get("from") or "") in diagnostic_ids
        and str(edge.get("type") or "") == "diagnoses"
        and str(
            (nodes.get(str(edge.get("to") or "")) or {}).get("kind") or ""
        )
        == "Reference"
    }
    graph_owner_ids: set[str] = set()
    for edge in edges:
        source_id = str(edge.get("from") or "")
        target_id = str(edge.get("to") or "")
        edge_type = str(edge.get("type") or "")
        target_tree = str((nodes.get(target_id) or {}).get("tree") or "")
        if target_tree != "old":
            continue
        if source_id in exact_reference_ids and edge_type == "occurs_in":
            graph_owner_ids.add(target_id)
        elif source_id in diagnostic_ids and edge_type in {
            "diagnoses",
            "occurs_at",
        }:
            graph_owner_ids.add(target_id)
    # Repair plans persist exact CodeIndex mappings even when the bulky graph
    # is moved to content-addressed evidence.  These IDs are produced from the
    # diagnostic source range and must still match a writable AST owner below.
    for mapping in work_unit.get("diagnostic_mappings") or []:
        if (
            not isinstance(mapping, dict)
            or mapping.get("unmapped")
            or str(mapping.get("status") or "").startswith("unresolved_")
            or not str(mapping.get("diagnostic_event_id") or "")
        ):
            continue
        graph_owner_ids.update(
            str(node_id)
            for node_id in mapping.get("diagnostic_owner_node_ids") or []
            if str(node_id)
        )
    if not graph_owner_ids:
        return []

    files: list[str] = []
    for owner in work_unit.get("repair_owners") or []:
        if (
            not isinstance(owner, dict)
            or not _owner_patch_ready(owner)
            or str(owner.get("owner_kind") or "")
            != "code_index_diagnostic_ast_owner"
        ):
            continue
        structure = (
            owner.get("diagnostic_structure")
            if isinstance(owner.get("diagnostic_structure"), dict)
            else {}
        )
        owner_node_id = str(
            structure.get("owner_node_id")
            or structure.get("root_entity_id")
            or owner.get("owner_entity_id")
            or ""
        )
        if (
            owner_node_id in graph_owner_ids
            and owner_node_id in selected_root_ids
        ):
            files.extend(repair_owner_patch_files([owner]))
    return _ordered_unique(files)


def _has_graph_proven_editable_diagnostic_locus(
    work_unit: dict[str, Any],
) -> bool:
    """Return whether a diagnostic reaches an exact writable old-tree locus."""

    return bool(
        graph_proven_editable_diagnostic_owner_files(work_unit)
    )


def _root_mapping_requires_expansion(
    work_unit: dict[str, Any],
) -> bool:
    """Keep unresolved roots incomplete even when traversal is exhausted."""

    code_graph_root = work_unit.get("code_graph_root_cause_cluster")
    has_code_graph_root = isinstance(code_graph_root, dict) and bool(
        code_graph_root
    )
    root = (
        code_graph_root
        if has_code_graph_root
        else work_unit.get("root_cause_cluster") or {}
    )
    if isinstance(root, dict):
        alternatives = [
            item
            for item in root.get("alternatives") or []
            if isinstance(item, dict)
        ]
        candidates = alternatives or [root]
        mapped_root_set = bool(
            root.get("ambiguous")
            and alternatives
            and all(
                item.get("node_id") or item.get("entity_id")
                for item in alternatives
            )
        )
        if root.get("unmapped") or (
            root.get("ambiguous") and not mapped_root_set
        ):
            return True
        for candidate in candidates:
            node_id = str(
                candidate.get("node_id")
                or candidate.get("entity_id")
                or ""
            )
            entity_kind = str(
                candidate.get("entity_kind")
                or candidate.get("kind")
                or ""
            ).lower()
            if (
                candidate.get("unmapped")
                or entity_kind.startswith("unresolved")
                or node_id.startswith("unresolved-")
            ):
                return True

    if has_code_graph_root:
        return False
    return any(
        item.get("unmapped")
        or str(item.get("status") or "").startswith("unresolved_")
        for item in work_unit.get("diagnostic_mappings") or []
        if isinstance(item, dict)
    )


def _needs_context_expansion(work_unit: dict[str, Any]) -> bool:
    scope = work_unit.get("code_graph_scope") or {}
    continuation = scope.get("continuation") or {}
    summary = scope.get("summary") or {}
    graph_exhausted = bool(
        scope
        and continuation.get("complete") is True
        and not summary.get("truncated")
        and not summary.get("graph_budget_truncated")
    )
    if _root_mapping_requires_expansion(work_unit):
        return not graph_exhausted
    if _has_graph_proven_editable_diagnostic_locus(work_unit):
        return False
    root = (
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    all_owners = [
        owner
        for owner in work_unit.get("repair_owners") or []
        if isinstance(owner, dict)
    ]
    action_owners = [
        owner
        for owner in all_owners
        if repair_owner_has_action_authority(owner)
    ]
    owners = action_owners or all_owners
    if any(_owner_patch_ready(owner) for owner in owners):
        return False
    if any(
        owner.get("allowed_for_entity_materialization")
        and owner.get("entity_materialization_files")
        for owner in owners
    ):
        return False
    blocked_owners = [owner for owner in owners if not _owner_patch_ready(owner)]
    if not blocked_owners:
        return False
    root_id = str(
        root.get("node_id") or root.get("entity_id") or ""
    ) if isinstance(root, dict) else ""
    if graph_exhausted:
        return False
    materializable_files = {
        normalize_relpath(str(candidate.get("file") or ""))
        for candidate in scope.get("owner_candidates") or []
        if isinstance(candidate, dict)
        and root_id
        and str(candidate.get("root_node_id") or "") == root_id
        and str(candidate.get("node_id") or "")
        and str(candidate.get("permission") or "") in {"editable", "escalation"}
        and (
            candidate.get("old_exists") is False
            or str(candidate.get("permission") or "") == "escalation"
        )
    }
    blocked_files = {
        normalize_relpath(str(owner.get("owner_file") or ""))
        for owner in blocked_owners
        if str(owner.get("owner_file") or "")
    }
    if (
        blocked_files
        and blocked_files <= materializable_files
        and (not continuation or continuation.get("complete"))
    ):
        return False
    return True


def _root_requires_definition_lookup(
    work_unit: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
) -> bool:
    """Return true when a selected graph root is only a declaration."""

    root = (
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    if not isinstance(root, dict):
        return False
    candidates = [
        root,
        *(item for item in root.get("alternatives") or [] if isinstance(item, dict)),
    ]
    for candidate in candidates:
        node_id = str(candidate.get("node_id") or candidate.get("entity_id") or "")
        node = nodes.get(node_id) or candidate
        if (
            node.get("is_definition") is False
            and any(node.get(key) for key in ("symbol", "spelling", "name"))
        ):
            return True
    return False


def _external_interface_symbols_needing_resolution(
    work_unit: dict[str, Any],
) -> list[str]:
    resolved = {
        str(item.get("symbol") or "")
        for item in work_unit.get("provider_resolutions") or []
        if isinstance(item, dict)
        and str(item.get("status") or "")
        not in {"", "entity_unresolved"}
    }
    symbols: list[str] = []
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
            if symbol and symbol not in resolved:
                symbols.append(symbol)
    return list(dict.fromkeys(symbols))


def _close_external_interface_provider_boundary(
    owner: dict[str, Any],
    provider_resolutions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify a missing interface after every provider is unique."""

    interface = dict(owner.get("driver_interface_ownership") or {})
    if interface.get("status") != "external_providers_outside_selected_entries":
        return dict(owner)
    declarations = [
        dict(item)
        for item in interface.get("unmatched_declarations") or []
        if isinstance(item, dict)
    ]
    resolutions_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for resolution in provider_resolutions:
        symbol = str(resolution.get("symbol") or "").strip()
        if symbol:
            resolutions_by_symbol.setdefault(symbol, []).append(resolution)

    resolved_bindings: list[dict[str, Any]] = []
    for declaration in declarations:
        symbol = str(declaration.get("symbol") or "").strip()
        matches = resolutions_by_symbol.get(symbol) or []
        if not symbol or len(matches) != 1:
            return dict(owner)
        resolution = matches[0]
        status = str(resolution.get("status") or "")
        entity_key = (
            "target_entities"
            if status == "old_entity_present"
            else "donor_entities"
            if status == "donor_entity_missing_in_old_unique"
            else ""
        )
        preferred_id = str(resolution.get("preferred_entity_id") or "")
        candidates = [
            dict(item)
            for item in resolution.get(entity_key) or []
            if isinstance(item, dict)
            and bool(item.get("is_definition"))
            and str(item.get("id") or "") == preferred_id
        ] if entity_key and preferred_id else []
        if len(candidates) != 1:
            return dict(owner)
        provider = candidates[0]
        if str(provider.get("contract_key") or provider.get("key") or "") != str(
            declaration.get("contract_key") or ""
        ):
            return dict(owner)
        declaration_signature = str(
            declaration.get("signature_fingerprint") or ""
        )
        provider_signature = str(provider.get("signature_fingerprint") or "")
        if (
            declaration_signature
            and provider_signature
            and declaration_signature != provider_signature
        ):
            return dict(owner)
        resolved_bindings.append(
            {
                "entity_id": declaration.get("entity_id"),
                "symbol": symbol,
                "contract_key": declaration.get("contract_key"),
                "signature_fingerprint": declaration_signature or None,
                "definition_entity_id": preferred_id,
                "definition_file": normalize_relpath(
                    str(provider.get("file") or "")
                ),
                "provider_status": status,
            }
        )

    if not declarations or len(resolved_bindings) != len(declarations):
        return dict(owner)
    existing_bindings = [
        dict(item)
        for item in interface.get("exact_bindings") or []
        if isinstance(item, dict)
    ]
    bindings_by_declaration = {
        str(binding.get("entity_id") or ""): binding
        for binding in [*existing_bindings, *resolved_bindings]
        if str(binding.get("entity_id") or "")
    }
    bindings = list(bindings_by_declaration.values())
    demanded_declaration_count = int(
        interface.get("demanded_external_declaration_count")
        if "demanded_external_declaration_count" in interface
        else interface.get("external_declaration_count")
        or 0
    )
    if len(bindings) != demanded_declaration_count:
        return dict(owner)
    provider_statuses = {
        str(binding.get("provider_status") or "")
        for binding in resolved_bindings
        if str(binding.get("provider_status") or "")
    }
    updated = dict(owner)
    updated_interface = {
        **interface,
        "status": "proven_provider_closure",
        "exact_binding_count": len(bindings),
        "exact_bindings": bindings,
        "provider_composition": sorted(provider_statuses),
        "unmatched_declarations": [],
        "ambiguous_declarations": [],
        "reason": (
            "every entry-exercised external declaration resolves to one exact "
            "selected-entry, target, or uniquely absent donor provider"
        ),
    }
    updated["driver_interface_ownership"] = updated_interface
    materializable = bool(
        updated.get("old_exists") is False
        and str(updated.get("owner_permission") or "")
        in {"editable", "escalation"}
    )
    updated["allowed_for_entity_materialization"] = materializable
    updated["entity_materialization_files"] = (
        [normalize_relpath(str(updated.get("owner_file") or ""))]
        if materializable
        else []
    )
    return updated


def _needs_code_graph_scope(work_unit: dict[str, Any]) -> bool:
    """Use graph state, not diagnostic category names, to decide expansion."""

    editable_diagnostic_locus = _has_graph_proven_editable_diagnostic_locus(
        work_unit
    )
    if _needs_context_expansion(work_unit):
        return True
    if _external_interface_symbols_needing_resolution(work_unit):
        return True
    scope = work_unit.get("code_graph_scope") or {}
    if scope:
        continuation = scope.get("continuation") or {}
        if continuation and not continuation.get("complete"):
            return True
        if bool((scope.get("summary") or {}).get("truncated")):
            return True
        if bool(
            (scope.get("summary") or {}).get(
                "graph_budget_truncated"
            )
        ):
            return True
        return False
    graph_summary = ((work_unit.get("diagnostic_graph") or {}).get("summary") or {})
    if (
        int(graph_summary.get("unmapped_diagnostic_count") or 0) > 0
        and not editable_diagnostic_locus
    ):
        return True
    graph = work_unit.get("diagnostic_graph") or {}
    nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    if _root_requires_definition_lookup(work_unit, nodes):
        return True
    root = (
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    selected_root_ids = {
        str(value)
        for value in (
            root.get("node_id") if isinstance(root, dict) else None,
            root.get("entity_id") if isinstance(root, dict) else None,
        )
        if str(value or "")
    }
    if isinstance(root, dict):
        selected_root_ids.update(
            str(item.get("node_id") or item.get("entity_id") or "")
            for item in root.get("alternatives") or []
            if isinstance(item, dict)
            and (item.get("node_id") or item.get("entity_id"))
        )
    correspondence_sources = {
        str(edge.get("from") or "")
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
        and edge.get("type") == "same_contract_as"
        and str(
            (nodes.get(str(edge.get("to") or "")) or {}).get(
                "tree"
            )
            or ""
        )
        == "old"
    }
    donor_contract_nodes = {
        node_id
        for node_id, node in nodes.items()
        if node_id in selected_root_ids
        and str(node.get("tree") or "").startswith("donor_")
        and str(node.get("kind") or "")
        in {
            "Function",
            "Variable",
            "Type",
            "Field",
            "Enum_Constant",
            "Macro",
            "Entity",
        }
        and bool((node.get("extent") or {}).get("start_line"))
    }
    if donor_contract_nodes - correspondence_sources:
        return True
    return bool(
        (work_unit.get("decision") or {}).get("mode")
        == "expand_graph"
    )


def _existing_cursor(work_unit: dict[str, Any]) -> dict[str, Any]:
    continuation = ((work_unit.get("code_graph_scope") or {}).get("continuation") or {})
    if not continuation:
        return {}
    if str(continuation.get("schema") or "") != "code_index_scope_cursor_v6":
        raise ValueError("code graph continuation does not use the current schema")
    raw = continuation.get("cursor_by_tree_symbol") or {}
    return {
        str(tree): {
            str(symbol): int(cursor)
            for symbol, cursor in value.items()
            if str(symbol) and str(cursor).isdigit()
        }
        for tree, value in raw.items()
        if str(tree) and isinstance(value, dict)
    }


def _scope_without_current_tree(
    scope: dict[str, Any],
) -> dict[str, Any]:
    """Retain donor graph evidence while invalidating the mutable old tree."""

    nodes = [
        dict(node)
        for node in scope.get("nodes") or []
        if isinstance(node, dict)
        and str(node.get("tree") or "").startswith("donor_")
    ]
    node_ids = {str(node.get("id") or "") for node in nodes if node.get("id")}
    edges = [
        dict(edge)
        for edge in scope.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("from") or "") in node_ids
        and str(edge.get("to") or "") in node_ids
    ]
    definition_query_state = {
        str(tree): {
            str(symbol): dict(state)
            for symbol, state in dict(symbol_states or {}).items()
            if isinstance(state, dict)
        }
        for tree, symbol_states in dict(
            scope.get("definition_query_state") or {}
        ).items()
        if str(tree).startswith("donor_")
    }
    return {
        "schema": scope.get("schema"),
        "input": {
            "symbols": list((scope.get("input") or {}).get("symbols") or []),
        },
        "summary": {
            "graph_budget_truncated": False,
            "truncation_reasons": [],
        },
        "nodes": nodes,
        "edges": edges,
        "definition_query_state": definition_query_state,
    }


def _merge_refreshed_donor_scope(
    refreshed: dict[str, Any],
    retained: dict[str, Any],
) -> dict[str, Any]:
    """Keep immutable donor contracts that a current-tree refresh cannot replace."""

    merged = dict(refreshed)
    refreshed_nodes = {
        str(node.get("id") or ""): dict(node)
        for node in refreshed.get("nodes") or []
        if isinstance(node, dict) and str(node.get("id") or "")
    }
    retained_nodes = {
        str(node.get("id") or ""): dict(node)
        for node in retained.get("nodes") or []
        if isinstance(node, dict)
        and str(node.get("tree") or "").startswith("donor_")
        and str(node.get("id") or "")
    }
    nodes = {**retained_nodes, **refreshed_nodes}
    merged["nodes"] = [nodes[node_id] for node_id in sorted(nodes)]

    edges: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for edge in [
        *list(retained.get("edges") or []),
        *list(refreshed.get("edges") or []),
    ]:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        if source not in nodes or target not in nodes:
            continue
        identity = (
            source,
            target,
            str(edge.get("type") or ""),
            str(edge.get("reference_id") or ""),
        )
        edges[identity] = dict(edge)
    merged["edges"] = [edges[key] for key in sorted(edges)]

    explanation = dict(refreshed.get("explanation_graph") or {})
    explanation_nodes = {
        str(node.get("id") or ""): dict(node)
        for node in explanation.get("nodes") or []
        if isinstance(node, dict) and str(node.get("id") or "")
    }
    explanation_nodes = {**retained_nodes, **explanation_nodes}
    explanation["nodes"] = [
        explanation_nodes[node_id]
        for node_id in sorted(explanation_nodes)
    ]
    explanation_edges: dict[
        tuple[str, str, str, str], dict[str, Any]
    ] = {}
    for edge in [
        *list(retained.get("edges") or []),
        *list(explanation.get("edges") or []),
    ]:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        if source not in explanation_nodes or target not in explanation_nodes:
            continue
        identity = (
            source,
            target,
            str(edge.get("type") or ""),
            str(edge.get("reference_id") or ""),
        )
        explanation_edges[identity] = dict(edge)
    explanation["edges"] = [
        explanation_edges[key] for key in sorted(explanation_edges)
    ]
    merged["explanation_graph"] = explanation
    return merged


def _stable_current_tree_root_anchor(
    work_unit: dict[str, Any],
) -> dict[str, Any]:
    """Capture a complete logical root identity, never a source line."""

    existing = work_unit.get("stable_root_anchor")
    if isinstance(existing, dict) and (
        existing.get("contract_key") or existing.get("members")
    ):
        return dict(existing)

    graph = (
        work_unit.get("diagnostic_graph")
        if isinstance(work_unit.get("diagnostic_graph"), dict)
        else {}
    )
    nodes = {
        str(node.get("id") or ""): dict(node)
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    edges = [
        dict(edge)
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
    ]
    root = (
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    if not isinstance(root, dict) or root.get("unmapped"):
        return {}

    alternatives = [
        item
        for item in root.get("alternatives") or []
        if isinstance(item, dict)
    ]
    root_records = alternatives if root.get("ambiguous") else [root]
    if not root_records or any(
        not str(item.get("node_id") or item.get("entity_id") or "")
        for item in root_records
    ):
        return {}

    def anchor_member(
        record: dict[str, Any],
        *,
        original_node_ids: Iterable[str],
    ) -> dict[str, Any]:
        relative_path = normalize_relpath(
            str(record.get("file") or record.get("path") or "")
        )
        contract_key = str(
            record.get("contract_key") or record.get("key") or ""
        )
        entity_kind = str(
            record.get("entity_kind") or record.get("kind") or ""
        ).lower()
        tree = str(record.get("tree") or "")
        if (
            (tree != "old" and not tree.startswith("donor_"))
            or not relative_path
            or not contract_key
            or not entity_kind
            or entity_kind.startswith("unresolved")
        ):
            return {}
        return {
            "tree": tree,
            "file": relative_path,
            "contract_key": contract_key,
            "entity_kind": entity_kind,
            "symbol": (
                record.get("symbol")
                or record.get("name")
                or record.get("spelling")
            ),
            "is_definition": record.get("is_definition"),
            "original_node_ids": sorted(
                {str(item) for item in original_node_ids if str(item)}
            ),
        }

    def identity(member: dict[str, Any]) -> tuple[object, ...]:
        return (
            member.get("tree"),
            member.get("file"),
            member.get("contract_key"),
            member.get("entity_kind"),
            member.get("symbol"),
            member.get("is_definition"),
        )

    logical_members: dict[tuple[object, ...], dict[str, Any]] = {}
    for root_record in root_records:
        root_id = str(
            root_record.get("node_id")
            or root_record.get("entity_id")
            or ""
        )
        node = dict(nodes.get(root_id) or root_record)
        member = anchor_member(node, original_node_ids=[root_id])
        if not member:
            return {}

        if str(member.get("tree") or "").startswith("donor_"):
            old_members: dict[tuple[object, ...], dict[str, Any]] = {}
            old_ids = {
                candidate_id
                for edge in edges
                if str(edge.get("type") or "") == "same_contract_as"
                for candidate_id in (
                    str(edge.get("to") or "")
                    if str(edge.get("from") or "") == root_id
                    else str(edge.get("from") or "")
                    if str(edge.get("to") or "") == root_id
                    else "",
                )
                if candidate_id
                and str((nodes.get(candidate_id) or {}).get("tree") or "")
                == "old"
                and str(
                    (nodes.get(candidate_id) or {}).get("contract_key")
                    or (nodes.get(candidate_id) or {}).get("key")
                    or ""
                )
                == str(member.get("contract_key") or "")
            }
            for old_id in old_ids:
                old_member = anchor_member(
                    dict(nodes.get(old_id) or {}),
                    original_node_ids=[root_id, old_id],
                )
                if old_member:
                    old_members[identity(old_member)] = old_member
            if len(old_members) > 1:
                return {}
            if old_members:
                member = next(iter(old_members.values()))

        member_identity = identity(member)
        prior = logical_members.get(member_identity)
        if prior is None:
            logical_members[member_identity] = member
        else:
            prior["original_node_ids"] = sorted(
                {
                    *list(prior.get("original_node_ids") or []),
                    *list(member.get("original_node_ids") or []),
                }
            )

    members = [
        logical_members[key]
        for key in sorted(logical_members, key=lambda item: repr(item))
    ]
    if not members:
        return {}
    anchor = {
        "schema": "stable_current_tree_root_anchor_v2",
        "root_shape": (
            "single_logical_root" if len(members) == 1 else "mapped_root_set"
        ),
        "members": members,
    }
    if len(members) == 1:
        anchor.update(members[0])
        anchor["original_node_id"] = next(
            iter(members[0].get("original_node_ids") or []),
            None,
        )
    return anchor


def _graph_scope_node_maps(
    graph_scope: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes = {
        str(node.get("id") or ""): dict(node)
        for node in graph_scope.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    edges = [
        dict(edge)
        for edge in graph_scope.get("edges") or []
        if isinstance(edge, dict)
    ]
    return nodes, edges


def _old_counterpart_ids(
    node_id: str,
    *,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    declaration_facet: bool = False,
) -> list[str]:
    counterparts: set[str] = set()
    for edge in edges:
        if str(edge.get("type") or "") != "same_contract_as":
            continue
        source_id = str(edge.get("from") or "")
        target_id = str(edge.get("to") or "")
        other_id = (
            target_id
            if source_id == node_id
            else source_id
            if target_id == node_id
            else ""
        )
        other = nodes.get(other_id) or {}
        if not other_id or str(other.get("tree") or "") != "old":
            continue
        if declaration_facet and (
            other.get("is_definition") is True
            or other.get("token_count") == 0
        ):
            continue
        counterparts.add(other_id)
    return sorted(
        counterparts
    )


def _graph_owner(
    candidate: dict[str, Any],
    *,
    graph_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_scope = graph_scope or {}
    nodes, edges = _graph_scope_node_maps(graph_scope)
    node_id = str(candidate.get("node_id") or "")
    node = dict(nodes.get(node_id) or {})
    descriptor = {**node, **candidate}
    relative_path = normalize_relpath(
        str(
            descriptor.get("file")
            or descriptor.get("path")
            or ""
        )
    )
    permission = str(candidate.get("permission") or "forbidden")
    owner_kind_hint = str(candidate.get("owner_kind") or "")
    owner_role = {
        "code_index_root_entity": "root_owner",
        "code_index_diagnostic_ast_owner": (
            "diagnostic_location_owner"
        ),
        "code_index_diagnostic_file": "diagnostic_file_owner",
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
        "code_index_repair_dependency": "repair_dependency_owner",
    }.get(owner_kind_hint, str(candidate.get("owner_role") or ""))
    old_exists = bool(candidate.get("old_exists"))
    owner_symbol = (
        descriptor.get("symbol")
        or descriptor.get("name")
        or descriptor.get("spelling")
    )
    contract_key = str(
        descriptor.get("contract_key")
        or descriptor.get("key")
        or ""
    )
    entity_kind = str(
        descriptor.get("entity_kind")
        or descriptor.get("kind")
        or ""
    )
    entity_tree = str(descriptor.get("tree") or "")
    old_counterpart_candidates = {
        str(item)
        for item in candidate.get("old_corresponding_entity_ids") or []
        if str(item)
    } | set(_old_counterpart_ids(node_id, nodes=nodes, edges=edges))
    root_declaration_from_existing_definition = bool(
        owner_role == "root_owner"
        and node_id
        and entity_tree.startswith("donor_")
        and entity_kind.lower() == "function"
        and descriptor.get("is_definition") is False
        and Path(relative_path).suffix.lower() == ".h"
        and any(
            (nodes.get(counterpart) or {}).get("tree") == "old"
            and (nodes.get(counterpart) or {}).get("is_definition") is True
            for counterpart in old_counterpart_candidates
        )
    )
    declaration_facet = bool(
        owner_role == "provider_declaration_owner"
        or root_declaration_from_existing_definition
    )
    candidate_counterparts = {
        str(item)
        for item in candidate.get("old_corresponding_entity_ids") or []
        if str(item)
        and not (
            declaration_facet
            and (nodes.get(str(item)) or {}).get("tree") == "old"
            and (
                (nodes.get(str(item)) or {}).get("is_definition") is True
                or (nodes.get(str(item)) or {}).get("token_count") == 0
            )
        )
    }
    old_counterparts = sorted(
        candidate_counterparts
        | set(
            _old_counterpart_ids(
                node_id,
                nodes=nodes,
                edges=edges,
                declaration_facet=declaration_facet,
            )
        )
    )
    proven_missing_donor_ids = {
        str(resolution.get("preferred_entity_id") or "")
        for resolution in graph_scope.get("provider_resolutions") or []
        if isinstance(resolution, dict)
        and str(resolution.get("status") or "")
        == "donor_entity_missing_in_old_unique"
        and str(resolution.get("preferred_entity_id") or "")
    }
    donor_only_entity = bool(
        entity_tree.startswith("donor_")
        and contract_key
        and entity_kind.lower()
        not in {
            "",
            "file",
            "diagnostic",
            "reference",
            "unresolvedentity",
            "unresolvedcontract",
        }
        and not old_counterparts
        and node_id in proven_missing_donor_ids
    )
    donor_entity = entity_tree.startswith("donor_") and entity_kind.lower() != "file"
    exact_absent_declaration_insertion = bool(
        (
            owner_role == "provider_declaration_owner"
            and candidate.get("exact_absent_declaration_insertion_allowed")
            or root_declaration_from_existing_definition
        )
        and donor_entity
        and contract_key
        and not old_counterparts
    )
    selected_root_owner = bool(
        owner_role == "root_owner"
        or (
            owner_role in {"", "diagnostic_location_owner"}
            and node_id == str(candidate.get("root_node_id") or "")
        )
    )
    exact_mutation_owner = bool(
        selected_root_owner
        or owner_role == "provider_definition_owner"
        or owner_role == "provider_declaration_owner"
        or owner_role == "consumer_integration_owner"
        or owner_role == "include_relation_owner"
    )
    automatic_owner = {
        "file": relative_path,
        "old_exists": old_exists,
        "owner_entity_tree": entity_tree,
        "old_corresponding_entity_ids": old_counterparts,
        "donor_only_entity": donor_only_entity,
        "driver_interface_ownership": dict(
            candidate.get("driver_interface_ownership") or {}
        ),
        "owner_role": owner_role,
        "root_declaration_from_existing_definition": (
            root_declaration_from_existing_definition
        ),
        "exact_absent_declaration_insertion_allowed": (
            exact_absent_declaration_insertion
        ),
        "exact_missing_include_materialization": dict(
            candidate.get("exact_missing_include_materialization") or {}
        ),
        "node_id": node_id,
    }
    unsafe_automatic_owner = unsafe_automatic_scope_owner(
        automatic_owner
    )
    exact_donor_definition_materialization = bool(
        donor_only_entity
        and descriptor.get("is_definition") is True
        and permission in {"editable", "escalation"}
        and old_exists
    )
    exact_declaration_action = exact_absent_declaration_insertion_owner(
        automatic_owner
    )
    exact_missing_include_action = (
        exact_missing_include_materialization_owner(automatic_owner)
    )
    allowed = bool(
        permission == "editable"
        and old_exists
        and exact_mutation_owner
        and not exact_donor_definition_materialization
        and (
            exact_declaration_action
            or (
                not unsafe_automatic_owner
                and (
                    not donor_entity
                    or bool(old_counterparts)
                )
            )
        )
    )
    owner_kind = {
        "root_owner": "code_index_root_entity",
        "diagnostic_location_owner": "code_index_diagnostic_ast_owner",
        "diagnostic_file_owner": "code_index_diagnostic_file",
        "consumer_integration_owner": (
            "code_index_consumer_integration_owner"
        ),
        "include_relation_owner": (
            "code_index_include_relation"
        ),
        "repair_dependency_owner": "code_index_repair_dependency",
        "provider_definition_owner": "code_index_provider_definition",
        "provider_declaration_owner": "code_index_provider_declaration",
    }.get(owner_role, "code_index_root_entity")
    reason = {
        "provider_definition_owner": (
            "bounded CodeIndex resolution found exactly one old-tree provider definition"
        ),
        "provider_declaration_owner": (
            "bounded CodeIndex resolution found an exact declaration in the same provider contract"
        ),
        "repair_dependency_owner": (
            "scope expansion reached this file through a typed implementation dependency"
        ),
        "consumer_integration_owner": (
            str(
                (
                    candidate.get("consumer_adaptation_authority") or {}
                ).get("reason")
                or "typed unresolved-include path selected this exact old-tree "
                "consumer as the bounded compatibility adaptation owner"
            )
        ),
        "include_relation_owner": (
            "the completed diagnostic graph or provider query selected this "
            "consumer's exact include relation as the bounded mutation owner"
        ),
    }.get(
        owner_role,
        "scope expansion reached this file through typed dependency edges",
    )
    if donor_only_entity:
        reason = (
            "CodeIndex selected this exact donor entity and proved that the "
            "current old tree has no same_contract_as entity; its existing "
            "owner file remains the bounded write target"
        )
    driver_interface_ownership = dict(
        candidate.get("driver_interface_ownership") or {}
    )
    allowed_for_entity_materialization = bool(
        exact_donor_definition_materialization
        or exact_missing_include_action
        or (
            candidate.get("allowed_for_entity_materialization")
            and not old_exists
            and not unsafe_automatic_owner
            and permission in {"editable", "escalation"}
        )
    )
    owner_entity_extent = dict(
        descriptor.get("source_extent")
        or descriptor.get("extent")
        or {}
    )
    return {
        "id": stable_id("owner", node_id, relative_path),
        "owner_kind": owner_kind,
        "owner_role": owner_role or None,
        "root_declaration_from_existing_definition": (
            root_declaration_from_existing_definition
        ),
        "owner_file": relative_path,
        "owner_symbol": owner_symbol,
        "owner_entity_id": node_id or None,
        "owner_contract_key": contract_key or None,
        "owner_entity_kind": entity_kind or None,
        "owner_entity_tree": entity_tree or None,
        "owner_entity_container": (
            dict(descriptor.get("container") or {})
            if isinstance(descriptor.get("container"), dict)
            else None
        ),
        "owner_entity_extent": owner_entity_extent or None,
        "owner_is_definition": descriptor.get("is_definition"),
        "old_corresponding_entity_ids": old_counterparts,
        "donor_only_entity": donor_only_entity,
        "exact_absent_entity_insertion_allowed": (
            exact_donor_definition_materialization
        ),
        "exact_absent_declaration_insertion_allowed": (
            exact_declaration_action
        ),
        "allowed_for_entity_materialization": (
            allowed_for_entity_materialization
        ),
        "entity_materialization_files": (
            [relative_path] if allowed_for_entity_materialization else []
        ),
        "driver_interface_ownership": driver_interface_ownership,
        "exact_missing_include_materialization": dict(
            candidate.get("exact_missing_include_materialization") or {}
        ),
        "include_relation_mode": candidate.get("include_relation_mode"),
        "current_include_name": candidate.get("current_include_name"),
        "required_include_name": candidate.get("required_include_name"),
        "expected_line": candidate.get("expected_line"),
        "provider_header_file": candidate.get("provider_header_file"),
        "consumer_owner_node_ids": list(
            candidate.get("consumer_owner_node_ids") or []
        ),
        "consumer_adaptation_authority": dict(
            candidate.get("consumer_adaptation_authority") or {}
        ),
        "allowed_as_patch_target": allowed,
        "allowed_patch_files": [relative_path] if allowed else [],
        "allowed_patch_permissions": {relative_path: permission},
        "owner_permission": permission,
        "old_exists": old_exists,
        "reason": reason,
        "evidence": [node_id, *list(candidate.get("reasons") or [])],
        "read_only_evidence_files": (
            []
            if allowed or allowed_for_entity_materialization
            else [relative_path]
        ),
        "diagnostic_structure": {
            "enabled": True,
            "backend": "code_index_v3",
            "root_entity_id": node_id,
            "contract_key": contract_key or None,
            "entity_kind": entity_kind or None,
            "entity_tree": entity_tree or None,
            "container": (
                dict(descriptor.get("container") or {})
                if isinstance(descriptor.get("container"), dict)
                else None
            ),
        },
    }


def _graph_root_candidate_files(graph_scope: dict[str, Any]) -> list[str]:
    """Return exact old-tree owners that may enter permission authorization."""

    candidates = [
        candidate
        for candidate in graph_scope.get("owner_candidates") or []
        if isinstance(candidate, dict)
        and str(candidate.get("permission") or "") in {"editable", "escalation"}
        and bool(candidate.get("old_exists"))
        and bool(str(candidate.get("node_id") or ""))
        and bool(str(candidate.get("root_node_id") or ""))
    ]
    root_owners = [
        candidate
        for candidate in candidates
        if str(candidate.get("owner_role") or "") == "root_owner"
        or (
            not candidate.get("owner_role")
            and str(candidate.get("node_id") or "")
            == str(candidate.get("root_node_id") or "")
        )
    ]
    provider_owners = [
        candidate
        for candidate in candidates
        if str(candidate.get("owner_role") or "")
        in {"provider_definition_owner", "provider_declaration_owner"}
    ]
    consumer_owners = [
        candidate
        for candidate in candidates
        if str(candidate.get("owner_role") or "")
        == "consumer_integration_owner"
    ]
    diagnostic_owners = [
        candidate
        for candidate in candidates
        if str(candidate.get("owner_role") or "")
        == "diagnostic_location_owner"
    ]
    selected = [
        *root_owners,
        *provider_owners,
        *consumer_owners,
        *diagnostic_owners,
    ]
    owners = [
        _graph_owner(candidate, graph_scope=graph_scope)
        for candidate in selected
    ]
    return _ordered_unique(
        owner.get("owner_file")
        for owner in owners
        if owner.get("old_exists") is True
        and str(owner.get("owner_permission") or "")
        in {"editable", "escalation"}
        and (
            not unsafe_automatic_scope_owner(owner)
            or exact_absent_declaration_insertion_owner(owner)
        )
        and (
            owner.get("allowed_as_patch_target")
            or str(owner.get("owner_permission") or "")
            == "escalation"
        )
        and (
            not str(owner.get("owner_entity_tree") or "").startswith(
                "donor_"
            )
            or bool(owner.get("old_corresponding_entity_ids"))
            or bool(owner.get("donor_only_entity"))
            or bool(
                owner.get(
                    "exact_absent_declaration_insertion_allowed"
                )
            )
        )
    )


def _dedupe_owner_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records:
        identity = tuple(
            str(record.get(key) or "")
            for key in ("owner_kind", "owner_file", "owner_symbol")
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(dict(record))
    return result


def expand_repair_context(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    candidate_files: list[str],
    refresh_current_tree: bool = False,
) -> dict[str, Any]:
    """Run one resumable CodeIndex scope-expansion page for a work unit."""

    if not refresh_current_tree and not _needs_code_graph_scope(work_unit):
        return {
            "work_unit": work_unit,
            "extra_candidate_files": [],
            "read_only_evidence_files": [],
            "report": {
                "schema": "repair_context_expansion_v3",
                "enabled": False,
                "reason": "typed diagnostic graph is complete and has a permitted root owner",
            },
        }

    graph_work_unit = dict(work_unit)
    stable_root_anchor = _stable_current_tree_root_anchor(work_unit)
    if stable_root_anchor:
        graph_work_unit["stable_root_anchor"] = stable_root_anchor
    if stable_root_anchor:
        graph_work_unit["source_issues"] = [
            (
                dict(issue)
                if str(issue.get("graph_root_policy") or "")
                == "typed_contract"
                else {
                    **dict(issue),
                    "stable_root_anchor": stable_root_anchor,
                }
            )
            for issue in work_unit.get("source_issues") or []
            if isinstance(issue, dict)
        ]
    prior_scope = (
        dict(work_unit.get("code_graph_scope") or {})
        if isinstance(work_unit.get("code_graph_scope"), dict)
        else {}
    )
    retained_donor_scope: dict[str, Any] = {}
    traversal_cursor = _existing_cursor(work_unit)
    if refresh_current_tree:
        retained_donor_scope = _scope_without_current_tree(prior_scope)
        prior_scope = retained_donor_scope
        traversal_cursor.pop("old", None)
    graph_scope = expand_scope_from_code_graph(
        spec,
        graph_work_unit,
        _ordered_unique(candidate_files),
        traversal_cursor=traversal_cursor,
        prior_scope=prior_scope,
    )
    if refresh_current_tree:
        graph_scope = _merge_refreshed_donor_scope(
            graph_scope,
            retained_donor_scope,
        )

    resolutions = [
        dict(item)
        for item in graph_scope.get("provider_resolutions") or []
        if isinstance(item, dict)
    ]
    retained_owners = [] if refresh_current_tree else [
        dict(item)
        for item in work_unit.get("repair_owners") or []
        if isinstance(item, dict)
    ]
    retained_interfaces = {
        (
            str(owner.get("owner_entity_id") or ""),
            normalize_relpath(str(owner.get("owner_file") or "")),
        ): dict(owner.get("driver_interface_ownership") or {})
        for owner in retained_owners
        if owner.get("driver_interface_ownership")
    }
    retained_missing_include_evidence = {
        (
            str(owner.get("owner_entity_id") or ""),
            normalize_relpath(str(owner.get("owner_file") or "")),
        ): dict(owner.get("exact_missing_include_materialization") or {})
        for owner in retained_owners
        if owner.get("exact_missing_include_materialization")
    }
    graph_owners = [
        _graph_owner(
            candidate,
            graph_scope=graph_scope,
        )
        for candidate in graph_scope.get("owner_candidates") or []
        if isinstance(candidate, dict) and candidate.get("file")
    ]
    for owner in graph_owners:
        identity = (
            str(owner.get("owner_entity_id") or ""),
            normalize_relpath(str(owner.get("owner_file") or "")),
        )
        if identity in retained_interfaces:
            owner["driver_interface_ownership"] = retained_interfaces[identity]
        if identity in retained_missing_include_evidence:
            owner["exact_missing_include_materialization"] = (
                retained_missing_include_evidence[identity]
            )
            if (
                exact_missing_include_materialization_owner(owner)
                and str(owner.get("owner_permission") or "")
                in {"editable", "escalation"}
            ):
                owner["allowed_for_entity_materialization"] = True
                owner["entity_materialization_files"] = [
                    normalize_relpath(str(owner.get("owner_file") or ""))
                ]
                owner["read_only_evidence_files"] = []
    owners = _dedupe_owner_records(
        [
            *graph_owners,
            *retained_owners,
        ],
    )
    owners = [
        _close_external_interface_provider_boundary(
            owner,
            resolutions,
        )
        for owner in owners
    ]
    graph_root_candidate_files = _graph_root_candidate_files(graph_scope)
    materialization_owner_files = _ordered_unique(
        owner.get("owner_file")
        for owner in owners
        if owner.get("allowed_for_entity_materialization")
        and owner.get("owner_file")
    )
    extra_files = _ordered_unique(
        [
            *list(graph_scope.get("extra_candidate_files") or []),
            *graph_root_candidate_files,
            *materialization_owner_files,
        ]
    )
    read_only_files = _ordered_unique(
        path
        for path in graph_scope.get("read_only_evidence_files") or []
        if normalize_relpath(str(path or ""))
        not in set([*graph_root_candidate_files, *materialization_owner_files])
    )
    clusters = [item for item in graph_scope.get("root_cause_clusters") or [] if isinstance(item, dict)]
    if len(clusters) == 1:
        graph_root = bind_graph_root_identity(clusters[0])
    elif clusters:
        graph_root = bind_graph_root_identity(
            {
                "title": "code_dependency_root_set",
                "root_shape": "mapped_root_set",
                "ambiguous": True,
                "alternatives": clusters,
                "source": "code_index_common_upstream_partition_v3",
            }
        )
    else:
        graph_root = {
            "title": "unmapped_diagnostic",
            "scope": stable_id("unmapped", str(work_unit.get("id") or "")),
            "unmapped": True,
            "source": "code_index_common_upstream_partition_v3",
        }

    report = {
        "schema": "repair_context_expansion_v3",
        "enabled": True,
        "input": {
            "traversal_cursor": traversal_cursor,
            "current_tree_refreshed": refresh_current_tree,
        },
        "code_graph_scope": {
            "schema": graph_scope.get("schema"),
            "summary": graph_scope.get("summary") or {},
            "continuation": graph_scope.get("continuation") or {},
            "query_pages": graph_scope.get("query_pages") or [],
            "diagnostic_mappings": graph_scope.get("diagnostic_mappings") or [],
            "explanation_graph": graph_scope.get("explanation_graph") or {},
            "discovered_candidate_files": graph_scope.get("discovered_candidate_files") or [],
            "extra_candidate_files": extra_files,
            "read_only_evidence_files": read_only_files,
            "owner_candidates": graph_scope.get("owner_candidates") or [],
            "root_candidate_files": graph_root_candidate_files,
            "repair_dependencies": graph_scope.get("repair_dependencies") or [],
            "repair_dependency_alignments": graph_scope.get("repair_dependency_alignments") or [],
            "provider_resolutions": graph_scope.get("provider_resolutions") or [],
            "root_cause_clusters": clusters,
        },
        "applied_provider_resolutions": resolutions,
        "relocalized_owners": owners,
        "extra_candidate_files": extra_files,
        "read_only_evidence_files": read_only_files,
        "continuation": graph_scope.get("continuation") or {},
    }
    expanded_work_unit = {
        **graph_work_unit,
        "repair_owners": owners,
        "repair_owner_count": len(owners),
        "provider_resolutions": resolutions,
        "diagnostic_mappings": list(graph_scope.get("diagnostic_mappings") or []),
        "diagnostic_graph": {
            **dict(graph_scope.get("explanation_graph") or {}),
            "summary": {
                **dict((graph_scope.get("explanation_graph") or {}).get("summary") or {}),
                "selected_root_cause_key": graph_root.get("key"),
            },
        },
        "context_expansion": report,
        "code_graph_scope": graph_scope,
        "code_graph_root_cause_cluster": graph_root,
        "stable_root_anchor": stable_root_anchor or None,
    }
    if not expanded_work_unit["stable_root_anchor"]:
        expanded_work_unit["stable_root_anchor"] = (
            _stable_current_tree_root_anchor(expanded_work_unit) or None
        )
    return {
        "work_unit": expanded_work_unit,
        "extra_candidate_files": extra_files,
        "read_only_evidence_files": read_only_files,
        "report": report,
    }
