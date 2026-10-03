#!/usr/bin/env python3
from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from driver_backport.core.case_spec import CaseSpec
from driver_backport.analysis.code_index import (
    CodeIndex,
    parse_structural_entity_file,
    structural_entity_bundle_alignments,
    structural_entity_insertion_plan,
    structural_entity_neighbor_alignment,
    structural_error_crosses_insertion_boundary,
)
from driver_backport.analysis.entity_resolution import (
    case_aware_entity_report,
    exact_entity_provenance_match,
)
from driver_backport.analysis.conditional_contract import (
    apply_conditional_variant_insertions,
    conditional_variant_insertions,
)
from driver_backport.analysis.diagnostic_identity import diagnostic_event_record
from driver_backport.analysis.entity_dependency_closure import (
    EntityDependencyClosureError,
    expand_entity_materialization_dependency_closure,
    prove_entity_materialization_dependency_closure,
)
from driver_backport.analysis.export_index import parse_export_contracts
from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths,
    stable_id,
    utc_now,
)
from driver_backport.planning.apply_protocol import (
    extract_json_object,
    immutable_diff_body_digest,
)
from driver_backport.planning.diagnostic_routing import diagnostic_route_summary
from driver_backport.planning.prompt_evidence import (
    assemble_prompt_evidence_view,
    compact_prompt_references,
    prompt_contract_evidence as _prompt_contract_evidence,
    prompt_entity_identity as _prompt_entity_identity,
    render_prompt_payload,
    source_evidence_record,
)
from driver_backport.planning.response_apply import (
    build_patch_conflict_evidence,
    touched_files_from_diff,
)
from driver_backport.planning.root_binding import (
    graph_root_binding_key,
    mapped_root_alternative_ids,
)
from driver_backport.planning.planner import exact_active_patch_postimage_boundary
from driver_backport.planning.execution_phase import (
    EXECUTION_PHASE_SEMANTIC_CLOSURE,
    unit_execution_phase,
)
from driver_backport.planning.work_unit import (
    exact_entity_materialization_owner_ids,
    exact_graph_scope_owners,
)


AGENTIC_ACTIONS = {
    "inspect",
    "expand_scope",
    "materialize_entity_from_donor",
    "apply_patch",
    "needs_escalation",
    "stop",
}
AGENTIC_INSPECT_MAX_LINES = 400
BUILD_METADATA_NAMES = {"Kconfig", "Makefile", "Kbuild"}


def _source_run_has_unaccounted_structural_error(
    report: dict[str, Any],
    *,
    start_line: int,
    end_line: int,
    entities: list[dict[str, Any]],
    dependency_closure: dict[str, Any] | None = None,
) -> bool:
    """Return whether a source-run error lacks exact entity-level proof."""

    proof = dependency_closure or {}
    semantic = proof.get("semantic_parse") or {}
    semantic_roots = (
        proof.get("semantic_root_entities") or []
        if proof.get("status") in {
            "proven", "admissible_with_post_materialization_provider_obligations"
        }
        and semantic.get("available") is True
        and not semantic.get("selected_root_inactive")
        and not proof.get("hard_unresolved")
        else []
    )
    for node in report.get("error_nodes") or []:
        if not isinstance(node, dict):
            continue
        error_extent = (
            node.get("extent")
            if isinstance(node.get("extent"), dict)
            else {}
        )
        error_start_line = int(error_extent.get("start_line") or 0)
        error_end_line = int(
            error_extent.get("end_line") or error_start_line
        )
        if (
            error_start_line <= 0
            or error_end_line <= 0
            or error_start_line > end_line
            or error_end_line < start_line
        ):
            continue
        error_start_column = int(error_extent.get("start_column") or 0)
        error_end_column = int(error_extent.get("end_column") or 0)
        if error_start_column <= 0 or error_end_column <= 0:
            return True
        error_start = (error_start_line, error_start_column)
        error_end = (error_end_line, error_end_column)
        accounted = any(
            entity.get("dependency_analysis_complete") is True
            and (
                int((entity.get("extent") or {}).get("start_line") or 0),
                int((entity.get("extent") or {}).get("start_column") or 0),
            )
            <= error_start
            and error_end
            <= (
                int((entity.get("extent") or {}).get("end_line") or 0),
                int((entity.get("extent") or {}).get("end_column") or 0),
            )
            for entity in entities
            if isinstance(entity.get("extent"), dict)
        )
        if not accounted:
            # Semantic proof may account for interior parser recovery, never
            # a different entity or an uncertain copy boundary.
            accounted = any(
                root.get("extent") == entity["extent"]
                and normalize_relpath(str(root.get("file") or ""))
                == normalize_relpath(str(entity.get("file") or ""))
                and root.get("kind") == _structural_entity_kind(entity)
                and root.get("contract_key") == entity.get("contract_key")
                and bool(root.get("is_definition"))
                == bool(entity.get("is_definition"))
                and (
                    int(entity["extent"].get("start_line") or 0),
                    int(entity["extent"].get("start_column") or 0),
                ) < error_start
                and error_end < (
                    int(entity["extent"].get("end_line") or 0),
                    int(entity["extent"].get("end_column") or 0),
                )
                for entity in entities
                if isinstance(entity.get("extent"), dict)
                and all(int(entity["extent"].get(key) or 0) > 0 for key in (
                    "start_line", "start_column", "end_line", "end_column"
                ))
                for root in semantic_roots
                if isinstance(root, dict)
            )
        if not accounted:
            return True
    return False
C_TRANSLATION_UNIT_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}


class EntityRepresentationAdaptationRequired(ValueError):
    def __init__(self, authority: dict[str, Any]) -> None:
        self.authority = authority
        super().__init__(
            "graph-selected donor entity is generated by a top-level macro "
            "declaration and requires a target-representation adaptation"
        )


class EntitySourceBoundaryError(ValueError):
    def __init__(self, context: dict[str, Any]) -> None:
        self.boundary_context = context
        super().__init__(
            "entity materialization structural anchors do not "
            "prove a current-tree source boundary"
        )


def agentic_materialization_rejection(
    error: ValueError, *, action: dict[str, Any], step: int, static: bool = False,
) -> dict[str, Any]:
    record = {
        "kind": ("static_entity_materialization_rejected" if static
                 else "entity_materialization_rejected"),
        "step": step, "error": str(error), "action": action,
    }
    for attribute, key in (
        ("proof", "dependency_closure"),
        ("boundary_context", "source_boundary_context"),
    ):
        value = getattr(error, attribute, None)
        if isinstance(value, dict):
            record[key] = value
    return record


def _extract_json_payload(text: str) -> dict[str, Any]:
    payload = extract_json_object(text)
    if payload is not None:
        return payload
    raise ValueError("agentic repair response does not contain a JSON object")


def parse_agentic_action(text: str) -> dict[str, Any]:
    payload = _extract_json_payload(text)
    if not payload.get("action"):
        wrappers = [key for key in AGENTIC_ACTIONS if isinstance(payload.get(key), dict)]
        if len(wrappers) == 1:
            wrapper_action = wrappers[0]
            nested = dict(payload[wrapper_action])
            nested.setdefault("action", wrapper_action)
            if wrapper_action == "apply_patch" and "patch" not in nested:
                nested = {"action": wrapper_action, "patch": nested}
            payload = nested
    action = str(payload.get("action") or "").strip()
    if action not in AGENTIC_ACTIONS:
        raise ValueError(f"unsupported agentic repair action: {action or '<empty>'}")
    payload["action"] = action
    allowed_top_level = {
        "inspect": {"action", "rationale", "inspect"},
        "expand_scope": {"action", "rationale", "scope_expansion"},
        "materialize_entity_from_donor": {
            "action",
            "rationale",
            "decision",
            "entity",
        },
        "apply_patch": {"action", "rationale", "decision", "patch_plan", "patch"},
        "needs_escalation": {"action", "rationale", "missing_context"},
        "stop": {"action", "rationale", "reason"},
    }[action]
    unexpected_top_level = sorted(set(payload) - allowed_top_level)
    if unexpected_top_level:
        raise ValueError(
            f"{action} action contains unexpected fields: "
            + ", ".join(unexpected_top_level)
        )
    if action == "inspect":
        request = payload.get("inspect")
        if not isinstance(request, dict):
            raise ValueError("inspect action requires inspect object")
        unexpected = sorted(set(request) - {"files", "entities", "logs"})
        if unexpected:
            raise ValueError(
                "inspect accepts only exact graph entities, file ranges, and logs, "
                "unexpected fields: "
                + ", ".join(unexpected)
            )
        for index, file_request in enumerate(request.get("files") or []):
            if not isinstance(file_request, dict):
                raise ValueError(f"inspect.files[{index}] must be an exact range object")
            unexpected_file_fields = sorted(
                set(file_request) - {"tree", "path", "startline", "endline"}
            )
            if unexpected_file_fields:
                raise ValueError(
                    f"inspect.files[{index}] contains unexpected fields: "
                    + ", ".join(unexpected_file_fields)
                )
            if not str(file_request.get("path") or "").strip():
                raise ValueError(f"inspect.files[{index}].path is required")
            try:
                startline = int(file_request.get("startline"))
                endline = int(file_request.get("endline"))
            except (TypeError, ValueError):
                raise ValueError(
                    f"inspect.files[{index}] requires integer startline and endline"
                ) from None
            if startline <= 0 or endline < startline:
                raise ValueError(f"inspect.files[{index}] has an invalid line range")
            if endline - startline + 1 > AGENTIC_INSPECT_MAX_LINES:
                raise ValueError(
                    f"inspect.files[{index}] exceeds the {AGENTIC_INSPECT_MAX_LINES}-line limit"
                )
        for index, entity_request in enumerate(request.get("entities") or []):
            if not isinstance(entity_request, dict):
                raise ValueError(f"inspect.entities[{index}] must be an exact entity object")
            unexpected_entity_fields = sorted(
                set(entity_request) - {"id", "tree"}
            )
            if unexpected_entity_fields:
                raise ValueError(
                    f"inspect.entities[{index}] contains unexpected fields: "
                    + ", ".join(unexpected_entity_fields)
                )
            if not str(entity_request.get("id") or "").strip():
                raise ValueError(f"inspect.entities[{index}].id is required")
            tree = str(entity_request.get("tree") or "").strip().lower()
            if tree and tree not in {"old", "target", "donor", "new", "both"}:
                raise ValueError(
                    f"inspect.entities[{index}].tree must be old, donor, or both"
                )
    if action == "apply_patch":
        patch = payload.get("patch")
        if not isinstance(patch, dict):
            raise ValueError("apply_patch action requires patch object")
        if patch.get("format") != "unified_diff":
            raise ValueError("apply_patch patch.format must be unified_diff")
        if not (patch.get("content") or patch.get("content_lines")):
            raise ValueError("apply_patch action requires patch.content or patch.content_lines")
    if action == "materialize_entity_from_donor":
        decision = payload.get("decision")
        if not isinstance(decision, dict):
            raise ValueError(
                "materialize_entity_from_donor action requires decision object"
            )
        unexpected_decision = sorted(
            set(decision)
            - {
                "mode",
                "work_unit_id",
                "root_node_id",
                "root_key",
                "addressed_semantic_gap_ids",
                "materialization_authority",
            }
        )
        if unexpected_decision:
            raise ValueError(
                "materialize_entity_from_donor decision contains unexpected fields: "
                + ", ".join(unexpected_decision)
            )
        if decision.get("mode") != "patch_root_subgraph":
            raise ValueError(
                "materialize_entity_from_donor decision.mode must be "
                "patch_root_subgraph"
            )
        materialization_authority = str(
            decision.get("materialization_authority") or ""
        )
        if materialization_authority and materialization_authority not in {
            "exact_required_demand_anchor_consumer",
            "unique_graph_root_contract",
        }:
            raise ValueError(
                "materialize_entity_from_donor "
                "decision.materialization_authority is not a supported "
                "controller authority"
            )
        if not str(decision.get("work_unit_id") or "").strip():
            raise ValueError(
                "materialize_entity_from_donor requires decision.work_unit_id"
            )
        entity = payload.get("entity")
        if not isinstance(entity, dict):
            raise ValueError(
                "materialize_entity_from_donor action requires entity object"
            )
        unexpected_entity = sorted(set(entity) - {"id"})
        if unexpected_entity:
            raise ValueError(
                "materialize_entity_from_donor accepts only an exact entity id, "
                "unexpected fields: "
                + ", ".join(unexpected_entity)
            )
        if not str(entity.get("id") or "").strip():
            raise ValueError(
                "materialize_entity_from_donor requires entity.id"
            )
    if action == "expand_scope":
        request = payload.get("scope_expansion")
        if not isinstance(request, dict):
            raise ValueError("expand_scope action requires scope_expansion object")
        if not str(request.get("work_unit_id") or "").strip():
            raise ValueError("expand_scope action requires scope_expansion.work_unit_id")
        if not str(request.get("root_node_id") or "").strip():
            raise ValueError("expand_scope action requires scope_expansion.root_node_id")
        unexpected = sorted(set(request) - {"work_unit_id", "root_node_id"})
        if unexpected:
            raise ValueError(
                "expand_scope accepts only stored graph identity, unexpected fields: "
                + ", ".join(unexpected)
            )
    if action == "needs_escalation":
        missing_context = payload.get("missing_context")
        if not isinstance(missing_context, dict):
            raise ValueError("needs_escalation action requires missing_context object")
        unexpected = sorted(
            set(missing_context) - {"work_unit_id", "continuation", "reason"}
        )
        if unexpected:
            raise ValueError(
                "needs_escalation cannot request guessed files or symbols, unexpected fields: "
                + ", ".join(unexpected)
            )
        if not str(missing_context.get("work_unit_id") or "").strip():
            raise ValueError("needs_escalation requires missing_context.work_unit_id")
    return payload


def graph_scope_expansion_request(
    action: dict[str, Any],
    repair_plan: dict[str, Any],
) -> dict[str, str]:
    """Bind scope expansion to either a graph cursor or an exact graph owner."""

    if action.get("action") != "expand_scope":
        raise ValueError("graph scope expansion requires expand_scope action")
    request = action.get("scope_expansion")
    if not isinstance(request, dict):
        raise ValueError("scope_expansion object is missing")
    unexpected = sorted(set(request) - {"work_unit_id", "root_node_id"})
    if unexpected:
        raise ValueError(
            "scope expansion accepts only work_unit_id and root_node_id: "
            + ", ".join(unexpected)
        )
    requested_unit_id = str(request.get("work_unit_id") or "")
    requested_root_id = str(request.get("root_node_id") or "")
    matching = [
        unit
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and requested_unit_id
        in {str(unit.get("id") or ""), str(unit.get("work_unit_id") or "")}
    ]
    if len(matching) != 1:
        raise ValueError("scope_expansion.work_unit_id must identify exactly one graph work unit")
    unit = matching[0]
    root = dict(
        unit.get("code_graph_root_cause_cluster")
        or unit.get("root_cause_cluster")
        or {}
    )
    actual_root_id = str(
        (unit.get("repair_decision") or {}).get("root_node_id")
        or root.get("node_id")
        or root.get("entity_id")
        or ""
    )
    if not actual_root_id or requested_root_id != actual_root_id:
        raise ValueError("scope_expansion.root_node_id does not match the selected graph root")
    continuation = dict((unit.get("code_graph_scope") or {}).get("continuation") or {})
    mode = "continue_graph" if continuation and not continuation.get("complete") else ""
    if not mode and exact_graph_scope_owners(
        unit,
        root_node_id=actual_root_id,
    ):
        mode = "materialize_graph_owner"
    if not mode:
        raise ValueError(
            "selected graph work unit has neither a pending CodeIndex continuation "
            "nor an exact graph-derived escalation owner"
        )
    return {
        "plan_unit_id": str(unit.get("id") or unit.get("work_unit_id") or ""),
        "mode": mode,
    }


def graph_escalation_target(
    action: dict[str, Any],
    repair_plan: dict[str, Any],
) -> str:
    """Bind an escalation request to exactly one current graph work unit."""

    if action.get("action") != "needs_escalation":
        raise ValueError("graph escalation requires needs_escalation action")
    missing_context = action.get("missing_context")
    if not isinstance(missing_context, dict):
        raise ValueError("missing_context object is missing")
    requested_unit_id = str(missing_context.get("work_unit_id") or "")
    matching = [
        unit
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and requested_unit_id
        in {str(unit.get("id") or ""), str(unit.get("work_unit_id") or "")}
    ]
    if len(matching) != 1:
        raise ValueError("missing_context.work_unit_id must identify exactly one graph work unit")
    unit = matching[0]
    continuation = dict(
        (unit.get("code_graph_scope") or {}).get("continuation")
        or {}
    )
    authorization = dict(
        unit.get("escalation_authorization") or {}
    )
    authorized_files = set(
        ordered_unique_relpaths(authorization.get("files") or [])
    )
    unresolved_escalation_files = set(
        ordered_unique_relpaths(
            [
                *list(unit.get("escalation_candidates") or []),
                *list(unit.get("materialization_candidates") or []),
            ]
        )
    ) - authorized_files
    unresolved_boundary = bool(
        unit.get("requires_context_expansion")
        or (continuation and not continuation.get("complete"))
        or unresolved_escalation_files
    )
    if not unresolved_boundary:
        raise ValueError(
            "the selected graph work unit has no unresolved graph or "
            "permission boundary; its exact escalation owners are already "
            "authorized"
        )
    return str(unit.get("id") or unit.get("work_unit_id") or "")


def _normalized_tree(value: object) -> str:
    tree = str(value or "old").strip().lower()
    if tree in {"target", "old"}:
        return "old"
    if tree == "new" or tree == "donor" or tree.startswith("donor_"):
        return "donor"
    return tree


def _unit_inspectable_ranges(unit: dict[str, Any]) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    source_pack = unit.get("repair_source_pack")
    if isinstance(source_pack, dict):
        for key, default_tree in (
            ("allowed_edit_sources", "old"),
            ("donor_reference_sources", "donor"),
            ("read_only_evidence_sources", "old"),
        ):
            for record in source_pack.get(key) or []:
                if not isinstance(record, dict):
                    continue
                relative_path = normalize_relpath(str(record.get("file") or ""))
                start = int(record.get("line_start") or 0)
                end = int(record.get("line_end") or start)
                if relative_path and start > 0 and end >= start:
                    ranges.append(
                        {
                            "tree": _normalized_tree(record.get("tree") or default_tree),
                            "path": relative_path,
                            "line_start": start,
                            "line_end": end,
                            "source": record.get("source_kind") or key,
                            "entity_id": (
                                record.get("code_graph_node_id")
                                or record.get("entity_id")
                                or record.get("node_id")
                            ),
                        }
                    )

    diagnostic_graph = unit.get("diagnostic_graph")
    if isinstance(diagnostic_graph, dict):
        for node in diagnostic_graph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            extent = node.get("extent") if isinstance(node.get("extent"), dict) else {}
            relative_path = normalize_relpath(str(node.get("file") or node.get("path") or ""))
            start = int(extent.get("start_line") or node.get("line") or 0)
            end = int(extent.get("end_line") or start)
            if relative_path and start > 0 and end >= start:
                ranges.append(
                    {
                        "tree": _normalized_tree(node.get("tree") or "old"),
                        "path": relative_path,
                        "line_start": start,
                        "line_end": end,
                        "source": "code_index_node_extent",
                        "entity_id": node.get("id"),
                    }
                )

    for issue in unit.get("source_issue_samples") or []:
        if not isinstance(issue, dict):
            continue
        relative_path = normalize_relpath(
            str(issue.get("file") or issue.get("path") or issue.get("relative_path") or "")
        )
        line = _issue_line_number(issue)
        if relative_path and line > 0:
            ranges.append(
                {
                    "tree": "old",
                    "path": relative_path,
                    "line_start": line,
                    "line_end": line,
                    "source": "diagnostic_location",
                    "diagnostic_event_id": issue.get("diagnostic_event_id"),
                }
            )

    unique: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for record in ranges:
        identity = (
            str(record["tree"]),
            str(record["path"]),
            int(record["line_start"]),
            int(record["line_end"]),
        )
        unique.setdefault(identity, record)
    return [unique[key] for key in sorted(unique)]


def _structural_entity_kind(record: dict[str, Any]) -> str:
    kind = str(
        record.get("entity_kind")
        or record.get("provider_kind")
        or record.get("kind")
        or ""
    ).strip().lower()
    aliases = {
        "function_decl": "function",
        "cxx_method": "function",
        "var_decl": "variable",
        "struct_decl": "type",
        "union_decl": "type",
        "enum_decl": "type",
        "typedef_decl": "type",
        "field_decl": "field",
        "enum_constant_decl": "enum_constant",
        "macro_definition": "macro",
    }
    return aliases.get(kind, kind)


def _unit_inspectable_entities(unit: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose only structural entities already selected for this work unit."""

    node_by_id: dict[str, dict[str, Any]] = {}
    for graph_key in ("diagnostic_graph", "code_graph_scope"):
        graph = unit.get(graph_key)
        if not isinstance(graph, dict):
            continue
        for node in graph.get("nodes") or []:
            if isinstance(node, dict) and node.get("id"):
                node_by_id[str(node["id"])] = node

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    source_pack = unit.get("repair_source_pack")
    if isinstance(source_pack, dict):
        for key in (
            "allowed_edit_sources",
            "donor_reference_sources",
            "read_only_evidence_sources",
        ):
            for source in source_pack.get(key) or []:
                if not isinstance(source, dict):
                    continue
                graph_id = str(
                    source.get("materialized_entity_id")
                    or source.get("code_graph_node_id")
                    or source.get("entity_id")
                    or source.get("node_id")
                    or ""
                )
                candidates.append((source, node_by_id.get(graph_id) or {}))

    diagnostic_graph = unit.get("diagnostic_graph")
    if isinstance(diagnostic_graph, dict):
        for node in diagnostic_graph.get("nodes") or []:
            if isinstance(node, dict):
                candidates.append((node, node))

    exact_materialization_ids = set(
        exact_entity_materialization_owner_ids(unit)
    )
    for owner in unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        owner_entity_id = str(owner.get("owner_entity_id") or "")
        if owner_entity_id not in exact_materialization_ids:
            continue
        candidates.append(
            (
                {
                    "entity_id": owner_entity_id,
                    "tree": owner.get("owner_entity_tree"),
                    "file": owner.get("owner_file"),
                    "entity_kind": owner.get("owner_entity_kind"),
                    "symbol": owner.get("owner_symbol"),
                    "contract_key": owner.get("owner_contract_key"),
                    "is_definition": owner.get("owner_is_definition"),
                    "extent": owner.get("owner_entity_extent"),
                },
                node_by_id.get(owner_entity_id) or {},
            )
        )

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    non_entities = {
        "",
        "diagnostic",
        "reference",
        "file",
        "contract",
        "contractowner",
        "translationunit",
    }
    for source, graph_node in candidates:
        merged = {**source, **graph_node}
        tree = _normalized_tree(merged.get("tree") or source.get("tree"))
        relative_path = normalize_relpath(
            str(
                merged.get("file")
                or merged.get("path")
                or source.get("file")
                or ""
            )
        )
        kind = _structural_entity_kind(merged)
        symbol = str(
            merged.get("symbol")
            or merged.get("name")
            or source.get("symbol")
            or ""
        ).strip()
        contract = str(
            merged.get("contract_key")
            or merged.get("key")
            or source.get("contract_key")
            or ""
        ).strip()
        if (
            tree not in {"old", "donor"}
            or not relative_path
            or kind in non_entities
            or not (contract or (kind and symbol))
        ):
            continue
        extent = merged.get("extent") if isinstance(merged.get("extent"), dict) else {}
        start = int(extent.get("start_line") or source.get("line_start") or 0)
        end = int(extent.get("end_line") or source.get("line_end") or start)
        graph_entity_id = str(
            merged.get("entity_id")
            or merged.get("id")
            or source.get("materialized_entity_id")
            or source.get("code_graph_node_id")
            or ""
        )
        if graph_entity_id:
            identity = ("graph_entity", tree, graph_entity_id)
        else:
            identity = (
                "structural_entity",
                tree,
                relative_path,
                contract,
                kind,
                symbol,
                str(start),
                str(end),
                str(merged.get("is_definition")),
            )
        if identity in seen:
            continue
        seen.add(identity)
        records.append(
            {
                "id": stable_id(
                    "agentic-entity-query",
                    *identity,
                ),
                "graph_entity_id": graph_entity_id or None,
                "tree": tree,
                "path": relative_path,
                "kind": kind,
                "symbol": symbol or None,
                "contract_key": contract or None,
                "is_definition": (
                    bool(merged.get("is_definition"))
                    if merged.get("is_definition") is not None
                    else None
                ),
                "line_start": start or None,
                "line_end": max(start, end) if start else None,
            }
        )
    return sorted(
        records,
        key=lambda item: (
            str(item.get("tree") or ""),
            str(item.get("path") or ""),
            int(item.get("line_start") or 0),
            str(item.get("contract_key") or ""),
        ),
    )


def _module_visibility_demands(unit: dict[str, Any]) -> list[dict[str, Any]]:
    """Return parser-produced cross-module symbol demands for one graph unit."""

    demands: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in unit.get("source_issue_samples") or []:
        if not isinstance(issue, dict):
            continue
        event = diagnostic_event_record(issue)
        symbol = str(event.get("symbol") or "").strip()
        module = normalize_relpath(str(event.get("module") or ""))
        if not symbol or not module:
            continue
        identity = (
            symbol,
            module,
            str(event.get("diagnostic_signature") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        demands.append(
            {
                "symbol": symbol,
                "consumer_module": module,
                "diagnostic_event_id": event.get("diagnostic_event_id"),
                "diagnostic_signature": event.get("diagnostic_signature"),
            }
        )
    return demands


def _unit_preferred_translation_units(
    spec: CaseSpec,
    unit: dict[str, Any],
) -> list[str]:
    """Retain real source translation units carried by diagnostic evidence."""

    roots = [
        Path(root).resolve()
        for root in [
            getattr(spec, "old_root_path", None),
            getattr(spec, "primary_donor_root", None),
            *(getattr(spec, "donor_root_paths", None) or []),
        ]
        if root
    ]

    def relative_source(raw_path: object) -> str:
        text = str(raw_path or "").strip()
        if not text:
            return ""
        path = Path(text)
        if path.is_absolute():
            resolved = path.resolve()
            for root in roots:
                try:
                    relative = resolved.relative_to(root).as_posix()
                except ValueError:
                    continue
                break
            else:
                return ""
        else:
            relative = normalize_relpath(text)
        if Path(relative).suffix.lower() not in C_TRANSLATION_UNIT_SUFFIXES:
            return ""
        try:
            exists = bool(
                spec.resolve_old(relative).is_file()
                or spec.resolve_donor(relative).is_file()
            )
        except (AttributeError, OSError):
            exists = False
        return relative if exists else ""

    candidates: list[object] = []
    for issue in unit.get("source_issue_samples") or []:
        if not isinstance(issue, dict):
            continue
        candidates.append(issue.get("file"))
        candidates.extend(issue.get("trigger_files") or [])
        for occurrence in issue.get("diagnostic_occurrences") or []:
            if not isinstance(occurrence, dict):
                continue
            candidates.append(occurrence.get("file"))
            candidates.extend(occurrence.get("trigger_files") or [])
    return ordered_unique_relpaths(
        relative
        for raw_path in candidates
        if (relative := relative_source(raw_path))
    )


def _semantic_entity_gap_bindings(
    unit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Bind exact provider/consumer entities to their semantic obligations."""

    active_gap_ids = {
        str(item)
        for item in unit.get("semantic_gap_ids") or []
        if str(item)
    }
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_binding(
        *,
        entity_id: object,
        contract_identity: object,
        role: str,
        evidence_id: object = None,
    ) -> None:
        normalized_entity_id = str(entity_id or "")
        normalized_contract_identity = str(contract_identity or "")
        if not normalized_entity_id or not normalized_contract_identity:
            return
        gap_id = stable_id(
            "semantic-contract-gap",
            normalized_contract_identity,
        )
        if active_gap_ids and gap_id not in active_gap_ids:
            return
        identity = (gap_id, normalized_entity_id, role)
        if identity in seen:
            return
        seen.add(identity)
        records.append(
            {
                "semantic_gap_id": gap_id,
                "entity_id": normalized_entity_id,
                "role": role,
                "contract_identity": normalized_contract_identity,
                **(
                    {"evidence_id": str(evidence_id)}
                    if str(evidence_id or "")
                    else {}
                ),
            }
        )

    for atom in unit.get("difference_atoms") or []:
        if not isinstance(atom, dict):
            continue
        contract_identity = atom.get("contract_identity")
        add_binding(
            entity_id=atom.get("old_entity_id"),
            contract_identity=contract_identity,
            role="old_provider",
            evidence_id=atom.get("id"),
        )
        add_binding(
            entity_id=atom.get("donor_entity_id"),
            contract_identity=contract_identity,
            role="donor_provider",
            evidence_id=atom.get("id"),
        )

    for anchor in unit.get("demand_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        owner = (
            anchor.get("owner_entity")
            if isinstance(anchor.get("owner_entity"), dict)
            else {}
        )
        add_binding(
            entity_id=(
                anchor.get("owner_entity_id")
                or owner.get("id")
                or owner.get("entity_id")
            ),
            contract_identity=anchor.get("contract_identity"),
            role="demand_anchor_consumer",
            evidence_id=anchor.get("id"),
        )

    return sorted(
        records,
        key=lambda item: (
            str(item.get("semantic_gap_id") or ""),
            str(item.get("entity_id") or ""),
            str(item.get("role") or ""),
        ),
    )


def _bound_semantic_gap_ids_for_entity(
    graph_unit: dict[str, Any],
    descriptor: dict[str, Any],
) -> list[str]:
    graph_entity_id = str(descriptor.get("graph_entity_id") or "")
    if not graph_entity_id:
        return []
    available = {
        str(item)
        for item in graph_unit.get("semantic_gap_ids") or []
        if str(item)
    }
    return sorted(
        {
            str(binding.get("semantic_gap_id") or "")
            for binding in graph_unit.get(
                "semantic_entity_gap_bindings"
            )
            or []
            if isinstance(binding, dict)
            and str(binding.get("entity_id") or "")
            == graph_entity_id
            and str(binding.get("semantic_gap_id") or "")
            in available
        }
    )


def build_agentic_scope(
    spec: CaseSpec,
    repair_plan: dict[str, Any],
    *,
    allow_escalation: bool,
) -> dict[str, Any]:
    graph_units: list[dict[str, Any]] = []
    execution_phases: set[str] = set()
    editable: list[str] = []
    escalation: list[str] = []
    inspectable: list[str] = []
    inspectable_ranges: list[dict[str, Any]] = []
    inspectable_entities: list[dict[str, Any]] = []
    explicit_read_only: list[str] = []
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        execution_phase = unit_execution_phase(unit)
        execution_phases.add(execution_phase)
        decision = dict(unit.get("repair_decision") or {})
        root = dict(
            unit.get("code_graph_root_cause_cluster")
            or unit.get("root_cause_cluster")
            or {}
        )
        root_node_id = str(
            decision.get("root_node_id")
            or root.get("node_id")
            or root.get("entity_id")
            or ""
        )
        root_key = str(
            decision.get("root_key")
            or graph_root_binding_key(root)
            or ""
        )
        root_alternative_node_ids = [
            str(item)
            for item in (
                decision.get("root_alternative_node_ids")
                or mapped_root_alternative_ids(root)
            )
            if str(item)
        ]
        mapped_root_set = bool(
            root_alternative_node_ids
            and not root_node_id
            and root_key
        )
        root_binding_id = root_node_id or root_key
        root_mapping_complete = bool(
            decision.get("root_mapping_complete")
            if "root_mapping_complete" in decision
            else decision.get("graph_complete")
            if "graph_complete" in decision
            else root_node_id and not root.get("ambiguous") and not root.get("unmapped")
        )
        continuation = dict((unit.get("code_graph_scope") or {}).get("continuation") or {})
        scope_complete = bool(not continuation or continuation.get("complete"))
        repair_authority = str(
            decision.get("repair_authority") or "code_graph_root"
        )
        consumer_adaptation_authority = dict(
            decision.get("consumer_adaptation_authority") or {}
        )
        exact_consumer_boundary = bool(
            repair_authority == "exhausted_provider_query_consumer"
            and root_mapping_complete is False
            and consumer_adaptation_authority.get("status") == "authorized"
            and consumer_adaptation_authority.get(
                "provider_resolution_complete"
            )
            is False
            and consumer_adaptation_authority.get(
                "bounded_provider_query_complete"
            )
            is True
            and consumer_adaptation_authority.get("candidate_write_set")
        )
        exact_postimage_boundary = exact_active_patch_postimage_boundary(
            unit,
            decision,
        )
        exact_include_owners = [
            owner
            for owner in unit.get("repair_owners") or []
            if isinstance(owner, dict)
            and owner.get("owner_kind") == "code_index_include_relation"
            and owner.get("owner_role") == "include_relation_owner"
            and owner.get("allowed_as_patch_target") is True
            and owner.get("include_relation_mode") == "mutate_unresolved"
        ]
        exact_unresolved_include_boundary = bool(
            repair_authority == "include_relation"
            and str(root.get("entity_kind") or "") == "UnresolvedInclude"
            and root_mapping_complete is False
            and len(exact_include_owners) == 1
            and ordered_unique_relpaths(
                list(decision.get("candidate_write_set") or [])
            )
            == [
                normalize_relpath(
                    str(exact_include_owners[0].get("owner_file") or "")
                )
            ]
        )
        unit_editable = [
            path
            for path in ordered_unique_relpaths(list(unit.get("allowed_fileset") or []))
            if spec.classify_edit_root(path) == "editable"
        ]
        if exact_unresolved_include_boundary:
            unit_editable = [
                path
                for path in unit_editable
                if path in decision.get("candidate_write_set") or []
            ]
        unit_escalation = [
            path
            for path in ordered_unique_relpaths(list(unit.get("escalation_candidates") or []))
            if allow_escalation
            and spec.classify_edit_root(path) == "escalation"
            and spec.resolve_old(path).is_file()
        ]
        unit_materialization_files = [
            path
            for path in ordered_unique_relpaths(
                list(
                    unit.get(
                        "entity_materialization_files"
                    )
                    or []
                )
            )
            if spec.classify_edit_root(path) == "editable"
            and spec.resolve_old(path).is_file()
            and spec.resolve_donor(path).is_file()
        ]
        if exact_unresolved_include_boundary:
            unit_escalation = []
            unit_materialization_files = []
        unit_ranges = _unit_inspectable_ranges(unit)
        unit_entities = _unit_inspectable_entities(unit)
        module_visibility_demands = _module_visibility_demands(unit)
        writable_unit_files = (
            set(unit_editable)
            | set(unit_escalation)
            | set(unit_materialization_files)
        )
        semantic_consumer_entity_ids = {
            str(
                (anchor.get("owner_entity") or {}).get("id")
                or (anchor.get("owner_entity") or {}).get(
                    "entity_id"
                )
                or ""
            )
            for anchor in unit.get("demand_anchors") or []
            if isinstance(anchor, dict)
            and isinstance(anchor.get("owner_entity"), dict)
            and str(
                (anchor.get("owner_entity") or {}).get("id")
                or ""
            )
        }
        exact_materialization_entity_ids = set(
            exact_entity_materialization_owner_ids(unit)
        )
        semantic_difference_dimensions = sorted(
            {
                str(atom.get("dimension") or "")
                for atom in unit.get("difference_atoms") or []
                if isinstance(atom, dict)
                and str(atom.get("dimension") or "")
            }
        )
        semantic_entity_gap_bindings = (
            _semantic_entity_gap_bindings(unit)
        )
        materializable_entity_ids = [
            str(record["id"])
            for record in unit_entities
            if record.get("is_definition") is True
            and str(record.get("tree") or "") == "donor"
            and normalize_relpath(str(record.get("path") or ""))
            in writable_unit_files
            and str(record.get("graph_entity_id") or "")
            in exact_materialization_entity_ids
        ]
        scope_materializable = bool(
            materializable_entity_ids
            or exact_graph_scope_owners(
                unit,
                root_node_id=root_node_id,
            )
        )
        materialization_action_ready = bool(
            unit_materialization_files and materializable_entity_ids
        )
        patch_ready = bool(
            root_binding_id
            and (
                root_mapping_complete
                or exact_consumer_boundary
                or exact_postimage_boundary
                or exact_unresolved_include_boundary
            )
            and (scope_complete or exact_unresolved_include_boundary)
            and (
                unit_editable
                or unit_escalation
                or materialization_action_ready
            )
        )
        if patch_ready:
            editable.extend(unit_editable)
            escalation.extend(unit_escalation)
        unit_paths = ordered_unique_relpaths(
            [record.get("path") for record in unit_ranges if isinstance(record, dict)]
        )
        explicit_read_only.extend(
            ordered_unique_relpaths(
                [
                    *list(unit.get("read_only_evidence_files") or []),
                ]
            )
        )
        inspectable_ranges.extend(unit_ranges)
        inspectable_entities.extend(unit_entities)
        inspectable.extend(unit_paths)
        graph_units.append(
            {
                "plan_unit_id": str(unit.get("id") or ""),
                "work_unit_id": str(unit.get("work_unit_id") or unit.get("id") or ""),
                "unit_kind": str(
                    unit.get("unit_kind") or "compiler_diagnostic"
                ),
                "execution_phase": execution_phase,
                "component_work_unit_ids": list(
                    unit.get("component_work_unit_ids") or []
                ),
                "semantic_gap_ids": list(unit.get("semantic_gap_ids") or []),
                "required_semantic_gap_ids": list(
                    unit.get("required_semantic_gap_ids") or []
                ),
                "pending_semantic_gap_ids": list(
                    unit.get("pending_semantic_gap_ids") or []
                ),
                "semantic_transaction_ids": list(
                    unit.get("semantic_transaction_ids") or []
                ),
                "incident_patch_ids": list(
                    unit.get("incident_patch_ids") or []
                ),
                "required_integration_obligation_ids": list(
                    unit.get(
                        "required_integration_obligation_ids"
                    )
                    or []
                ),
                "integration_obligations": list(
                    unit.get("integration_obligations") or []
                ),
                "integration_coverage_blockers": list(
                    unit.get("integration_coverage_blockers") or []
                ),
                "semantic_consumer_entity_ids": sorted(
                    semantic_consumer_entity_ids
                ),
                "semantic_entity_gap_bindings": (
                    semantic_entity_gap_bindings
                ),
                "semantic_difference_dimensions": (
                    semantic_difference_dimensions
                ),
                "root_node_id": root_node_id or None,
                "root_key": root_key or None,
                "root_binding_id": root_binding_id or None,
                "root_binding_kind": (
                    "mapped_root_set"
                    if mapped_root_set
                    else "node"
                    if root_node_id
                    else "unbound"
                ),
                "root_alternative_node_ids": root_alternative_node_ids,
                "repair_authority": repair_authority,
                "consumer_adaptation_authority": (
                    consumer_adaptation_authority or None
                ),
                "candidate_write_set": ordered_unique_relpaths(
                    list(decision.get("candidate_write_set") or [])
                ),
                "diagnostic_event_ids": list(
                    decision.get("diagnostic_event_ids")
                    or root.get("diagnostic_event_ids")
                    or []
                ),
                "graph_complete": root_mapping_complete,
                "root_mapping_complete": root_mapping_complete,
                "scope_complete": scope_complete,
                "scope_expandable": bool(continuation and not continuation.get("complete")),
                "scope_materializable": scope_materializable,
                "patch_ready": patch_ready,
                "editable_files": unit_editable if patch_ready else [],
                "escalation_files": unit_escalation if patch_ready else [],
                "entity_materialization_files": (
                    unit_materialization_files
                    if patch_ready
                    else []
                ),
                "inspectable_files": unit_paths,
                "inspectable_ranges": unit_ranges,
                "inspectable_entity_ids": [
                    record["id"] for record in unit_entities
                ],
                "materializable_entity_ids": materializable_entity_ids,
                "module_visibility_demands": module_visibility_demands,
                "preferred_translation_units": (
                    _unit_preferred_translation_units(spec, unit)
                ),
                "continuation": continuation,
            }
        )
    editable = ordered_unique_relpaths(editable)
    escalation = ordered_unique_relpaths(escalation)
    inspectable = ordered_unique_relpaths(inspectable)
    writable = set(editable) | set(escalation)
    read_only = [
        path
        for path in ordered_unique_relpaths(
            [*explicit_read_only, *inspectable]
        )
        if path not in writable
    ]
    unique_entities = {
        str(record["id"]): record
        for record in inspectable_entities
        if isinstance(record, dict) and record.get("id")
    }
    if len(execution_phases) > 1:
        raise ValueError(
            "agentic scope cannot mix execution phases: "
            + ", ".join(sorted(execution_phases))
        )
    execution_phase = (
        next(iter(execution_phases))
        if execution_phases
        else None
    )
    return {
        "schema": "agentic_repair_scope_v5",
        "execution_phase": execution_phase,
        "editable_files": editable,
        "escalation_files": escalation,
        "read_only_evidence_files": read_only,
        "inspectable_files": inspectable,
        "inspectable_ranges": inspectable_ranges,
        "inspectable_entities": [
            unique_entities[key] for key in sorted(unique_entities)
        ],
        "graph_units": graph_units,
        "allow_escalation": allow_escalation,
    }


def _compact_validation(
    validation_report: dict[str, Any],
) -> dict[str, Any]:
    route_summary = dict(
        validation_report.get("diagnostic_routes")
        or diagnostic_route_summary(validation_report)
    )
    route_summary.pop("issue_kinds", None)
    return {
        "status": validation_report.get("status"),
        "overall_ok": validation_report.get("overall_ok"),
        "stop_reason": validation_report.get("stop_reason"),
        "summary": validation_report.get("summary") or {},
        "diagnostic_routes": route_summary,
    }


def _issue_line_number(issue: dict[str, Any]) -> int:
    try:
        return int(issue.get("line") or issue.get("line_start") or 0)
    except (TypeError, ValueError):
        return 0


def _iter_validation_issues(
    validation_report: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()

    def add(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        path = raw.get("file") or raw.get("path") or raw.get("relative_path")
        line = _issue_line_number(raw)
        message = str(raw.get("message") or raw.get("raw") or "")
        key = (str(path or ""), line, message)
        if not key[0] or key in seen:
            return
        seen.add(key)
        issues.append(raw)

    for item in validation_report.get("issues") or []:
        add(item)
    for phase in validation_report.get("phases") or []:
        if not isinstance(phase, dict):
            continue
        for item in phase.get("issues") or []:
            add(item)
    return issues


def _issue_relpath(spec: CaseSpec, value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(spec.old_root_path.resolve()).as_posix()
        except ValueError:
            return None
    return normalize_relpath(text)


def _issue_column_number(issue: dict[str, Any]) -> int:
    try:
        return int(issue.get("column") or issue.get("column_start") or 0)
    except (TypeError, ValueError):
        return 0


def _issue_identity(spec: CaseSpec, issue: dict[str, Any]) -> tuple[str, str, int, int, str]:
    event_id = str(issue.get("diagnostic_event_id") or "")
    relpath = _issue_relpath(
        spec,
        issue.get("file") or issue.get("path") or issue.get("relative_path"),
    ) or ""
    return (
        event_id,
        relpath,
        _issue_line_number(issue),
        _issue_column_number(issue),
        str(issue.get("message") or issue.get("raw") or ""),
    )


def _repair_plan_issue_records(repair_plan: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        for issue in unit.get("source_issue_samples") or []:
            if isinstance(issue, dict):
                records.append((unit, issue))
    return records


def _diagnostic_bindings(
    spec: CaseSpec,
    issue: dict[str, Any],
    repair_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    identity = _issue_identity(spec, issue)
    event_id = identity[0]
    bindings: list[dict[str, Any]] = []
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        samples = [item for item in unit.get("source_issue_samples") or [] if isinstance(item, dict)]
        sample_matches = any(
            (event_id and str(sample.get("diagnostic_event_id") or "") == event_id)
            or (not event_id and _issue_identity(spec, sample) == identity)
            for sample in samples
        )
        mappings = [
            dict(mapping)
            for mapping in unit.get("diagnostic_mappings") or []
            if isinstance(mapping, dict)
            and event_id
            and str(mapping.get("diagnostic_event_id") or "") == event_id
        ]
        root = dict(
            unit.get("code_graph_root_cause_cluster")
            or unit.get("root_cause_cluster")
            or {}
        )
        root_events = {
            str(item)
            for item in root.get("diagnostic_event_ids") or []
            if str(item)
        }
        if not sample_matches and not mappings and (not event_id or event_id not in root_events):
            continue
        decision = dict(unit.get("repair_decision") or {})
        bindings.append(
            {
                "plan_unit_id": unit.get("id"),
                "work_unit_id": unit.get("work_unit_id") or unit.get("id"),
                "root_node_id": decision.get("root_node_id") or root.get("node_id") or root.get("entity_id"),
                "root_key": (
                    decision.get("root_key")
                    or graph_root_binding_key(root)
                    or None
                ),
                "graph_complete": decision.get("graph_complete"),
                "diagnostic_mappings": mappings,
            }
        )
    return bindings


def build_current_diagnostic_context(
    spec: CaseSpec,
    validation_report: dict[str, Any],
    *,
    repair_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    issues = _iter_validation_issues(validation_report)
    if repair_plan:
        seen = {_issue_identity(spec, item) for item in issues}
        for _, item in _repair_plan_issue_records(repair_plan):
            key = _issue_identity(spec, item)
            if key not in seen:
                seen.add(key)
                issues.append(item)
    for issue in issues:
        relpath = _issue_relpath(spec, issue.get("file") or issue.get("path") or issue.get("relative_path"))
        if not relpath:
            continue
        line = max(1, _issue_line_number(issue))
        contexts.append(
            {
                "diagnostic_event_id": issue.get("diagnostic_event_id"),
                "file": relpath,
                "line": line,
                "column": _issue_column_number(issue) or None,
                "message": str(
                    issue.get("message") or issue.get("raw") or ""
                ),
                "raw": str(issue.get("raw") or ""),
                "diagnostic_route": issue.get("diagnostic_route"),
                "code_graph_bindings": (
                    _diagnostic_bindings(spec, issue, repair_plan)
                    if repair_plan
                    else []
                ),
            }
        )

    return contexts


def _selected_current_diagnostic_context(
    spec: CaseSpec,
    validation_report: dict[str, Any],
    repair_plan: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Return diagnostics with an exact binding to the selected plan unit."""

    contexts = build_current_diagnostic_context(
        spec,
        validation_report,
        repair_plan=repair_plan,
    )
    selected = [
        context
        for context in contexts
        if context.get("code_graph_bindings")
    ]
    if contexts and not selected:
        raise ValueError(
            "selected repair plan has no exact current diagnostic binding"
        )
    return selected, len(contexts)


def _inspect_file_key(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    tree = str(record.get("tree") or "").strip() or "unknown"
    path = normalize_relpath(str(record.get("path") or ""))
    if not path:
        return None
    return f"{tree}:{path}"


def _inspect_entity_key(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    entity_id = str(
        record.get("requested_entity_id")
        or record.get("id")
        or ""
    ).strip()
    if not entity_id:
        return None
    tree = str(record.get("tree") or "").strip() or "unknown"
    return f"{tree}:{entity_id}"


def summarize_agentic_evidence_log(
    evidence_log: list[dict[str, Any]],
) -> dict[str, Any]:
    inspected_files: list[str] = []
    inspected_file_ranges: list[dict[str, Any]] = []
    inspected_entities: list[str] = []
    inspected_entity_requests: list[dict[str, Any]] = []
    rejected_inspects = 0

    def add_unique(target: list[str], value: str) -> None:
        if value and value not in target:
            target.append(value)

    for entry in evidence_log:
        if entry.get("kind") == "redundant_inspect_rejected":
            rejected_inspects += 1
        evidence = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
        for record in evidence.get("files") or []:
            key = _inspect_file_key(record)
            if key:
                start = record.get("line_start")
                end = record.get("line_end")
                suffix = f":{start}-{end}" if start and end else ""
                add_unique(inspected_files, f"{key}{suffix}")
                inspected_file_ranges.append(
                    {
                        "tree": record.get("tree"),
                        "path": record.get("path"),
                        "line_start": record.get("line_start"),
                        "line_end": record.get("line_end"),
                    }
                )
        for record in evidence.get("entities") or []:
            key = _inspect_entity_key(record)
            if not key:
                continue
            add_unique(inspected_entities, key)
            inspected_entity_requests.append(
                {
                    "tree": record.get("tree"),
                    "id": (
                        record.get("requested_entity_id")
                        or record.get("id")
                    ),
                    "status": record.get("status"),
                }
            )

    return {
        "inspect_result_count": sum(1 for entry in evidence_log if entry.get("kind") == "inspect_result"),
        "scope_expansion_result_count": sum(
            1 for entry in evidence_log if entry.get("kind") == "graph_scope_expansion_result"
        ),
        "redundant_inspect_rejected_count": rejected_inspects,
        "blocked_apply_count": sum(
            1
            for entry in evidence_log
            if entry.get("kind")
            in {
                "apply_rejected_missing_requested_evidence",
                "invalid_graph_patch_decision",
                "apply_without_validation_worthy_change",
                "validated_regressive_patch_rejected",
            }
        ),
        "inspected_files": inspected_files,
        "inspected_file_ranges": inspected_file_ranges,
        "inspected_entities": inspected_entities,
        "inspected_entity_requests": inspected_entity_requests,
    }


def agentic_inspect_redundancy(action: dict[str, Any], evidence_log: list[dict[str, Any]]) -> dict[str, Any]:
    if action.get("action") != "inspect":
        return {"redundant": False}
    summary = summarize_agentic_evidence_log(evidence_log)
    seen_ranges: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for record in summary["inspected_file_ranges"]:
        key = (str(record.get("tree") or ""), normalize_relpath(str(record.get("path") or "")))
        start = int(record.get("line_start") or 1)
        end = int(record.get("line_end") or start)
        seen_ranges.setdefault(key, []).append((start, end))
    seen_entities = {
        (
            str(record.get("tree") or ""),
            str(record.get("id") or ""),
        )
        for record in summary["inspected_entity_requests"]
        if record.get("id")
    }
    request = action.get("inspect") if isinstance(action.get("inspect"), dict) else {}
    requested_ranges: list[dict[str, Any]] = []
    for raw in request.get("files") or []:
        normalized = _normalize_file_request(raw)
        if normalized is None:
            continue
        trees = [normalized["tree"]]
        if normalized["tree"] == "both":
            trees = ["old", "donor"]
        for tree in trees:
            requested_ranges.append({**normalized, "tree": tree})
    requested_count = len(requested_ranges)
    requested_entities: list[dict[str, str]] = []
    for raw in request.get("entities") or []:
        if not isinstance(raw, dict):
            continue
        entity_id = str(raw.get("id") or "").strip()
        tree = _normalized_tree(raw.get("tree") or "")
        trees = [tree] if tree in {"old", "donor"} else ["old", "donor"]
        for tree_name in trees:
            requested_entities.append({"tree": tree_name, "id": entity_id})
    requested_count += len(requested_entities)
    if requested_count == 0:
        return {"redundant": False}
    missing_ranges = [
        record
        for record in requested_ranges
        if not any(
            seen_start <= int(record["startline"]) and seen_end >= int(record["endline"])
            for seen_start, seen_end in seen_ranges.get((record["tree"], record["path"]), [])
        )
    ]
    missing_count = len(missing_ranges)
    missing_entities = [
        record
        for record in requested_entities
        if (record["tree"], record["id"]) not in seen_entities
    ]
    missing_count += len(missing_entities)
    return {
        "redundant": missing_count == 0,
        "requested_count": requested_count,
        "already_seen_count": requested_count - missing_count,
        "missing_file_ranges": missing_ranges,
        "missing_entities": missing_entities,
    }


def _agentic_apply_action_identity(
    action: dict[str, Any],
) -> tuple[str, str]:
    patch = action.get("patch") if isinstance(action.get("patch"), dict) else {}
    content = patch.get("content")
    if content is None and isinstance(patch.get("content_lines"), list):
        content = "\n".join(str(line) for line in patch["content_lines"])
    normalized_content = "\n".join(
        line.rstrip()
        for line in str(content or "").replace("\r\n", "\n").splitlines()
    )
    decision = (
        action.get("decision")
        if isinstance(action.get("decision"), dict)
        else {}
    )
    patch_body_digest = immutable_diff_body_digest(normalized_content)
    work_unit_id = str(decision.get("work_unit_id") or "")
    root_node_id = str(decision.get("root_node_id") or "")
    root_key = str(decision.get("root_key") or "")
    if work_unit_id and (root_node_id or root_key) and patch_body_digest:
        payload = {
            "mode": decision.get("mode"),
            "work_unit_id": work_unit_id,
            "root_node_id": root_node_id,
            "root_key": root_key,
            "addressed_semantic_gap_ids": sorted(
                str(item)
                for item in decision.get("addressed_semantic_gap_ids") or []
                if str(item)
            ),
            "touched_files": sorted(
                touched_files_from_diff(normalized_content)
            ),
            "immutable_patch_body_digest": patch_body_digest,
        }
        return (
            stable_id(
                "agentic-semantic-edit",
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ),
            "graph_bound_patch_body",
        )
    payload = {
        "decision": decision,
        "patch_format": patch.get("format"),
        "patch_content": normalized_content,
    }
    return (
        stable_id(
            "agentic-apply-action",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        ),
        "exact_response",
    )


def _agentic_apply_action_fingerprint(action: dict[str, Any]) -> str:
    return _agentic_apply_action_identity(action)[0]


def agentic_source_evidence_fingerprint(plan_unit: dict[str, Any]) -> str:
    source_pack = (
        plan_unit.get("repair_source_pack")
        if isinstance(plan_unit.get("repair_source_pack"), dict)
        else {}
    )
    records: list[dict[str, Any]] = []
    for role in (
        "allowed_edit_sources",
        "donor_reference_sources",
        "read_only_evidence_sources",
    ):
        for record in source_pack.get(role) or []:
            if not isinstance(record, dict):
                continue
            records.append(
                {
                    "role": role,
                    "tree": record.get("tree"),
                    "file": record.get("file"),
                    "entity_id": record.get("entity_id")
                    or record.get("code_graph_node_id")
                    or record.get("node_id"),
                    "line_start": record.get("line_start"),
                    "line_end": record.get("line_end"),
                    "text": record.get("text"),
                }
            )
    return stable_id(
        "agentic-source-evidence",
        json.dumps(records, ensure_ascii=False, sort_keys=True),
    )


def _agentic_evidence_state_fingerprint(evidence_log: list[dict[str, Any]]) -> str:
    state_records = []
    for entry in evidence_log:
        kind = str(entry.get("kind") or "")
        if kind == "inspect_result":
            state_records.append(
                {
                    "kind": kind,
                    "evidence": entry.get("evidence") or {},
                }
            )
        elif kind == "graph_scope_expansion_result":
            state_records.append(
                {
                    "kind": kind,
                    "plan_unit_id": entry.get("plan_unit_id"),
                    "progressed": entry.get("progressed"),
                    "continuation": entry.get("continuation") or {},
                }
            )
        elif kind == "controller_recovery_result":
            state_records.append(
                {
                    "kind": kind,
                    "status": entry.get("status"),
                    "recovery_kind": entry.get("recovery_kind"),
                    "action_fingerprint": entry.get("action_fingerprint"),
                    "required_next_action": entry.get("required_next_action"),
                    "patch_conflict_evidence": entry.get(
                        "patch_conflict_evidence"
                    )
                    or {},
                    "typed_apply_failures": list(
                        entry.get("typed_apply_failures") or []
                    ),
                }
            )
        elif kind == "validated_regressive_patch_rejected":
            prior_outcome = (
                entry.get("prior_outcome")
                if isinstance(entry.get("prior_outcome"), dict)
                else {}
            )
            state_records.append(
                {
                    "kind": kind,
                    "candidate_id": (
                        (entry.get("candidate") or {}).get("candidate_id")
                        if isinstance(entry.get("candidate"), dict)
                        else None
                    ),
                    "prior_outcome_id": prior_outcome.get("outcome_id"),
                    "validation_status": prior_outcome.get(
                        "validation_status"
                    ),
                }
            )
    return stable_id(
        "agentic-evidence-state",
        json.dumps(state_records, ensure_ascii=False, sort_keys=True),
    )


def agentic_blocked_apply_metadata(
    action: dict[str, Any],
    evidence_log: list[dict[str, Any]],
) -> dict[str, str]:
    action_fingerprint, fingerprint_kind = (
        _agentic_apply_action_identity(action)
    )
    return {
        "action_fingerprint": action_fingerprint,
        "semantic_action_id": action_fingerprint,
        "action_fingerprint_kind": fingerprint_kind,
        "evidence_state_fingerprint": _agentic_evidence_state_fingerprint(
            evidence_log
        ),
    }


def _agentic_action_root_binding(action: dict[str, Any]) -> dict[str, str]:
    decision = (
        action.get("decision")
        if isinstance(action.get("decision"), dict)
        else {}
    )
    return {
        "work_unit_id": str(decision.get("work_unit_id") or ""),
        "root_node_id": str(decision.get("root_node_id") or ""),
        "root_key": str(decision.get("root_key") or ""),
    }


def _agentic_missing_evidence_blocker_fingerprint(
    blocker: dict[str, Any],
) -> str:
    action = blocker.get("action") if isinstance(blocker.get("action"), dict) else {}
    payload = {
        "kind": blocker.get("kind"),
        "root_binding": _agentic_action_root_binding(action),
        "evidence_gate": blocker.get("evidence_gate") or {},
    }
    return stable_id(
        "agentic-blocker-state",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )


def agentic_apply_redundancy(
    action: dict[str, Any],
    evidence_log: list[dict[str, Any]],
) -> dict[str, Any]:
    if action.get("action") != "apply_patch":
        return {"redundant": False}
    metadata = agentic_blocked_apply_metadata(action, evidence_log)
    action_binding = _agentic_action_root_binding(action)
    previous_missing_evidence_index = next(
        (
            index
            for index in range(len(evidence_log) - 1, -1, -1)
            if evidence_log[index].get("kind")
            == "apply_rejected_missing_requested_evidence"
            and (
                not any(action_binding.values())
                or not any(
                    _agentic_action_root_binding(
                        evidence_log[index].get("action") or {}
                    ).values()
                )
                or _agentic_action_root_binding(
                    evidence_log[index].get("action") or {}
                )
                == action_binding
            )
        ),
        None,
    )
    if previous_missing_evidence_index is not None:
        previous_blocker = evidence_log[previous_missing_evidence_index]
        later_evidence = evidence_log[previous_missing_evidence_index + 1 :]
        blocker_state_changed = any(
            entry.get("kind") == "inspect_result"
            or (
                entry.get("kind") == "graph_scope_expansion_result"
                and entry.get("progressed")
            )
            for entry in later_evidence
        )
        if not blocker_state_changed:
            return {
                "redundant": True,
                **metadata,
                "reason": "apply_blocked_by_unchanged_missing_evidence",
                "previous_blocker_kind": previous_blocker.get("kind"),
                "blocker_state_fingerprint": (
                    _agentic_missing_evidence_blocker_fingerprint(
                        previous_blocker
                    )
                ),
                "previous_step": previous_blocker.get("step"),
            }
    previous_recovery_index = next(
        (
            index
            for index in range(len(evidence_log) - 1, -1, -1)
            if evidence_log[index].get("kind") == "controller_recovery_result"
            and evidence_log[index].get("action_fingerprint")
            == metadata["action_fingerprint"]
        ),
        None,
    )
    if previous_recovery_index is not None:
        previous_recovery = evidence_log[previous_recovery_index]
        later_evidence = evidence_log[previous_recovery_index + 1 :]
        recovery_state_changed = any(
            entry.get("kind") == "inspect_result"
            or (
                entry.get("kind") == "graph_scope_expansion_result"
                and entry.get("progressed")
            )
            for entry in later_evidence
        )
        if not recovery_state_changed:
            return {
                "redundant": True,
                **metadata,
                "reason": "same_action_replayed_after_controller_recovery",
                "previous_blocker_kind": previous_recovery.get(
                    "previous_blocker_kind"
                ),
                "previous_recovery_kind": previous_recovery.get(
                    "recovery_kind"
                ),
                "previous_recovery_status": previous_recovery.get("status"),
            }
    blocker_kinds = {
        "apply_rejected_missing_requested_evidence",
        "invalid_graph_patch_decision",
        "apply_without_validation_worthy_change",
    }
    previous = None
    for offset, entry in enumerate(reversed(evidence_log)):
        if (
            entry.get("kind") not in blocker_kinds
            or entry.get("action_fingerprint")
            != metadata["action_fingerprint"]
        ):
            continue
        if entry.get("kind") == "apply_without_validation_worthy_change":
            if any(
                later.get("source_or_build_changed") is True
                or later.get("candidate_changed") is True
                for later in evidence_log[len(evidence_log) - offset :]
            ):
                continue
        elif (
            entry.get("evidence_state_fingerprint")
            != metadata["evidence_state_fingerprint"]
        ):
            continue
        previous = entry
        break
    return {
        "redundant": previous is not None,
        **metadata,
        "previous_blocker_kind": (
            previous.get("kind") if isinstance(previous, dict) else None
        ),
        "previous_step": (
            previous.get("step") if isinstance(previous, dict) else None
        ),
    }


def agentic_prior_apply_redundancy(
    action: dict[str, Any],
    repair_plan: dict[str, Any],
    evidence_log: list[dict[str, Any]],
) -> dict[str, Any]:
    if action.get("action") != "apply_patch":
        return {"redundant": False}
    if any(
        entry.get("kind") in {"inspect_result", "graph_scope_expansion_result"}
        for entry in evidence_log
    ):
        return {
            "redundant": False,
            "reason": "current_unit_has_new_dynamic_evidence",
        }

    action_fingerprint = _agentic_apply_action_fingerprint(action)
    decision = action.get("decision") if isinstance(action.get("decision"), dict) else {}
    requested_unit_id = str(decision.get("work_unit_id") or "")
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        unit_ids = {
            str(unit.get("id") or ""),
            str(unit.get("work_unit_id") or ""),
        }
        if requested_unit_id and requested_unit_id not in unit_ids:
            continue
        current_source_fingerprint = agentic_source_evidence_fingerprint(unit)
        for attempt in reversed(
            list(unit.get("prior_unresolved_patch_attempts") or [])
        ):
            if not isinstance(attempt, dict):
                continue
            if attempt.get("action_fingerprint") != action_fingerprint:
                continue
            if (
                attempt.get("diagnostic_component_match") != "exact"
                or attempt.get("source_evidence_fingerprint")
                != current_source_fingerprint
            ):
                continue
            return {
                "redundant": True,
                "reason": "identical_prior_round_action_with_unchanged_state",
                "action_fingerprint": action_fingerprint,
                "source_evidence_fingerprint": current_source_fingerprint,
                "prior_cycle": attempt.get("cycle"),
                "prior_blocker_reason": attempt.get("blocker_reason"),
                "previous_blocker_kind": (
                    attempt.get("blocked_apply_kind")
                    or attempt.get("blocker_kind")
                ),
                "previous_recovery_kind": attempt.get("recovery_kind"),
                "previous_recovery_status": attempt.get("recovery_status"),
            }
    return {
        "redundant": False,
        "action_fingerprint": action_fingerprint,
    }


def agentic_patch_text(action: dict[str, Any]) -> str:
    patch = action.get("patch") if isinstance(action.get("patch"), dict) else {}
    if patch.get("content") is not None:
        return str(patch.get("content") or "")
    if isinstance(patch.get("content_lines"), list):
        return "\n".join(str(line) for line in patch["content_lines"]) + "\n"
    return ""


def _latest_typed_apply_failures(
    evidence_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project the latest apply failures without losing their typed owner."""

    for entry in reversed(evidence_log):
        apply_summary = (
            entry.get("apply_summary")
            if isinstance(entry.get("apply_summary"), dict)
            else {}
        )
        if not apply_summary:
            continue
        failures: list[dict[str, Any]] = []
        for result in apply_summary.get("results") or []:
            if not isinstance(result, dict):
                continue
            failure = (
                result.get("apply_failure")
                if isinstance(result.get("apply_failure"), dict)
                else {}
            )
            route = (
                result.get("invalid_route")
                if isinstance(result.get("invalid_route"), dict)
                else {}
            )
            if not failure:
                continue
            details = (
                failure.get("details")
                if isinstance(failure.get("details"), dict)
                else {}
            )
            projected_details = {
                key: details.get(key)
                for key in (
                    "checker",
                    "provider_status",
                    "provider_reason",
                    "provider_blocking_findings",
                )
                if details.get(key) not in (None, "", [], {})
            }
            structural = details.get("structural_postcondition")
            if (
                failure.get("stage") == "structural_postcondition"
                and isinstance(structural, dict)
            ):
                regressions = [
                    {
                        "file": item.get("file"),
                        "after_error_nodes": list(
                            item.get("after_error_nodes") or []
                        ),
                    }
                    for item in structural.get("regressions") or []
                    if isinstance(item, dict)
                ]
                if regressions:
                    projected_details["structural_regressions"] = regressions
            failures.append(
                {
                    key: value
                    for key, value in {
                        "status": result.get("status"),
                        "original_status": result.get("original_status"),
                        "error": result.get("error"),
                        "touched_files": list(
                            result.get("touched_files") or []
                        ),
                        "stage": failure.get("stage"),
                        "recoverability": failure.get("recoverability"),
                        "finding_kind": failure.get("finding_kind"),
                        "root_binding": failure.get("root_binding"),
                        "details": projected_details,
                        "route": route.get("route"),
                        "required_transition": route.get(
                            "required_transition"
                        ),
                        "retry_requires_state_change": route.get(
                            "retry_requires_state_change"
                        ),
                    }.items()
                    if value not in (None, "", [], {})
                }
            )
        if failures:
            return failures
    return []


def _typed_apply_blocker_state_fingerprint(
    typed_apply_failures: list[dict[str, Any]],
    *,
    repair_plan: dict[str, Any],
    evidence_log: list[dict[str, Any]],
) -> str:
    """Identify unchanged repair authority independently of patch wording."""

    def canonical_records(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_content = {
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            ): record
            for record in records
        }
        return [by_content[key] for key in sorted(by_content)]

    def compact_findings(details: dict[str, Any]) -> list[dict[str, Any]]:
        return canonical_records(
            [
                {
                    key: finding.get(key)
                    for key in (
                        "kind",
                        "file",
                        "contract_key",
                        "owner_contract_key",
                        "checker",
                        "status",
                    )
                    if finding.get(key) not in (None, "", [], {})
                }
                for finding in details.get("provider_blocking_findings") or []
                if isinstance(finding, dict)
            ]
        )

    failure_state = canonical_records(
        [
            {
                "stage": failure.get("stage"),
                "recoverability": failure.get("recoverability"),
                "finding_kind": failure.get("finding_kind"),
                "root_binding": failure.get("root_binding") or {},
                "route": failure.get("route"),
                "required_transition": failure.get("required_transition"),
                "retry_requires_state_change": failure.get(
                    "retry_requires_state_change"
                ),
                "checker": (failure.get("details") or {}).get("checker"),
                "provider_status": (failure.get("details") or {}).get(
                    "provider_status"
                ),
                "provider_reason": (failure.get("details") or {}).get(
                    "provider_reason"
                ),
                "blocking_findings": compact_findings(
                    failure.get("details") or {}
                ),
            }
            for failure in typed_apply_failures
        ]
    )
    actionable_evidence = canonical_records(
        [
            {
                "kind": entry.get("kind"),
                "evidence": entry.get("evidence") or {},
                "plan_unit_id": entry.get("plan_unit_id"),
                "progressed": entry.get("progressed"),
                "continuation": entry.get("continuation") or {},
            }
            for entry in evidence_log
            if entry.get("kind") == "inspect_result"
            or (
                entry.get("kind") == "graph_scope_expansion_result"
                and entry.get("progressed")
            )
        ]
    )
    plan_unit = dict((repair_plan.get("work_unit_plans") or [{}])[0])
    return stable_id(
        "typed-apply-blocker-state",
        json.dumps(
            {
                "failures": failure_state,
                "source_evidence_fingerprint": (
                    agentic_source_evidence_fingerprint(plan_unit)
                ),
                "actionable_evidence": actionable_evidence,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def collect_agentic_recovery_evidence(
    spec: CaseSpec,
    *,
    action: dict[str, Any],
    repair_plan: dict[str, Any],
    scope: dict[str, Any],
    evidence_log: list[dict[str, Any]],
    redundancy: dict[str, Any],
) -> dict[str, Any]:
    """Run one bounded static recovery before a blocked unit is deferred."""

    action_fingerprint = str(
        redundancy.get("action_fingerprint")
        or _agentic_apply_action_fingerprint(action)
    )
    typed_apply_failures = _latest_typed_apply_failures(evidence_log)
    blocker_state_fingerprint = str(
        redundancy.get("blocker_state_fingerprint") or ""
    )
    if typed_apply_failures:
        blocker_state_fingerprint = _typed_apply_blocker_state_fingerprint(
            typed_apply_failures,
            repair_plan=repair_plan,
            evidence_log=evidence_log,
        )
    if any(
        entry.get("kind") == "controller_recovery_result"
        and (
            (
                blocker_state_fingerprint
                and entry.get("blocker_state_fingerprint")
                == blocker_state_fingerprint
            )
            or (
                not blocker_state_fingerprint
                and entry.get("action_fingerprint") == action_fingerprint
            )
        )
        for entry in evidence_log
    ):
        return {
            "schema": "agentic_controller_recovery_v1",
            "status": "recovery_exhausted",
            "action_fingerprint": action_fingerprint,
            "blocker_state_fingerprint": blocker_state_fingerprint or None,
            "reason": (
                "the bounded controller recovery for this unchanged blocker "
                "state has already been consumed without new evidence"
            ),
        }

    units = [
        unit
        for unit in scope.get("graph_units") or []
        if isinstance(unit, dict)
    ]
    graph_unit = units[0] if len(units) == 1 else {}
    previous_kind = str(redundancy.get("previous_blocker_kind") or "")
    typed_failure_stages = sorted(
        {
            str(item.get("stage") or "")
            for item in typed_apply_failures
            if str(item.get("stage") or "")
        }
    )
    effective_previous_kind = (
        typed_failure_stages[0]
        if len(typed_failure_stages) == 1
        else previous_kind
    )
    base: dict[str, Any] = {
        "schema": "agentic_controller_recovery_v1",
        "action_fingerprint": action_fingerprint,
        "blocker_state_fingerprint": blocker_state_fingerprint or None,
        "previous_blocker_kind": effective_previous_kind or None,
        "failure_origin_kind": previous_kind or None,
        "typed_apply_failures": typed_apply_failures,
        "source_evidence_fingerprint": (
            agentic_source_evidence_fingerprint(
                dict((repair_plan.get("work_unit_plans") or [{}])[0])
            )
        ),
    }

    if previous_kind == "invalid_graph_patch_decision" and graph_unit:
        corrected_action = dict(action)
        corrected_action["decision"] = {
            "mode": "patch_root_subgraph",
            "work_unit_id": (
                graph_unit.get("work_unit_id")
                or graph_unit.get("plan_unit_id")
            ),
            "root_node_id": graph_unit.get("root_node_id"),
            "root_key": graph_unit.get("root_key"),
        }
        return {
            **base,
            "status": "recovered_action",
            "recovery_kind": "canonical_graph_root_rebinding",
            "corrected_action": corrected_action,
        }

    if typed_apply_failures and typed_failure_stages == ["structural_rebase"]:
        diff_text = agentic_patch_text(action)
        conflict = build_patch_conflict_evidence(
            spec,
            diff_text=diff_text,
            touched_files=touched_files_from_diff(diff_text),
        )
        if conflict.get("hunk_count"):
            return {
                **base,
                "status": "recovered_evidence",
                "recovery_kind": "current_tree_patch_conflict_inspection",
                "patch_conflict_evidence": conflict,
                "required_next_action": (
                    "use the current source evidence to produce a different "
                    "applicable patch, inspect a new graph entity, or escalate"
                ),
            }

    if typed_apply_failures:
        required_transitions = list(
            dict.fromkeys(
                str(item.get("required_transition") or "")
                for item in typed_apply_failures
                if str(item.get("required_transition") or "")
            )
        )
        return {
            **base,
            "status": "recovered_evidence",
            "recovery_kind": "typed_apply_failure_replan",
            "required_next_action": {
                "transitions": required_transitions,
                "instruction": (
                    "repair the exact typed apply failure; do not reinterpret "
                    "it as a generic patch conflict"
                ),
                "allowed": [
                    "inspect a previously unseen graph-authorized provider or consumer entity named by the failure evidence",
                    "advance the exact graph frontier when the required contract owner is outside the current graph page",
                    "submit one atomic patch that satisfies every blocking finding at its touched locus",
                    "return needs_escalation with the unresolved graph edge",
                ],
            },
        }

    if (
        previous_kind == "apply_rejected_missing_requested_evidence"
        and graph_unit.get("scope_expandable")
        and graph_unit.get("root_node_id")
    ):
        return {
            **base,
            "status": "recovered_action",
            "recovery_kind": "advance_existing_graph_frontier",
            "corrected_action": {
                "action": "expand_scope",
                "rationale": (
                    "requested structural evidence is unavailable in the "
                    "current graph page; advance the exact stored frontier"
                ),
                "scope_expansion": {
                    "work_unit_id": (
                        graph_unit.get("work_unit_id")
                        or graph_unit.get("plan_unit_id")
                    ),
                    "root_node_id": graph_unit.get("root_node_id"),
                },
            },
        }

    if previous_kind == "apply_rejected_missing_requested_evidence":
        return {
            **base,
            "status": "recovered_evidence",
            "recovery_kind": "missing_evidence_action_replan",
            "reason": (
                "the requested evidence is unavailable in the current graph "
                "page and no deterministic continuation is exposed"
            ),
            "required_next_action": {
                "prohibited": (
                    "do not submit any apply_patch until inspect or graph scope "
                    "changes the missing-evidence state"
                ),
                "allowed": [
                    "inspect a previously unseen graph-authorized entity or range",
                    "return needs_escalation with the exact missing graph edge",
                ],
            },
        }

    return {
        **base,
        "status": "recovered_evidence",
        "recovery_kind": "alternative_action_replan",
        "reason": (
            "no deterministic source or graph transition can solve this blocker, "
            "so the controller requires one different evidence-producing or edit action"
        ),
        "required_next_action": {
            "prohibited": (
                "do not submit the same apply_patch action fingerprint again"
            ),
            "allowed": [
                "inspect a previously unseen graph-authorized entity or range",
                "advance the exact graph frontier when continuation is available",
                "submit a materially different patch for the same root",
                "return needs_escalation with the unresolved evidence gap",
            ],
        },
    }


def _compact_agentic_graph_units(scope: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for unit in scope.get("graph_units") or []:
        if not isinstance(unit, dict):
            continue
        event_ids = [str(item) for item in unit.get("diagnostic_event_ids") or [] if str(item)]
        compact.append(
            {
                key: value
                for key, value in {
                    "plan_unit_id": unit.get("plan_unit_id"),
                    "work_unit_id": unit.get("work_unit_id"),
                    "unit_kind": unit.get("unit_kind"),
                    "execution_phase": unit.get("execution_phase"),
                    "component_work_unit_ids": list(
                        unit.get("component_work_unit_ids") or []
                    ),
                    "semantic_gap_ids": list(unit.get("semantic_gap_ids") or []),
                    "required_semantic_gap_ids": list(
                        unit.get("required_semantic_gap_ids") or []
                    ),
                    "pending_semantic_gap_ids": list(
                        unit.get("pending_semantic_gap_ids") or []
                    ),
                    "semantic_transaction_ids": list(
                        unit.get("semantic_transaction_ids") or []
                    ),
                    "incident_patch_ids": list(
                        unit.get("incident_patch_ids") or []
                    ),
                    "required_integration_obligation_ids": list(
                        unit.get(
                            "required_integration_obligation_ids"
                        )
                        or []
                    ),
                    "semantic_consumer_entity_ids": list(
                        unit.get("semantic_consumer_entity_ids")
                        or []
                    ),
                    "root_node_id": unit.get("root_node_id"),
                    "root_key": unit.get("root_key"),
                    "root_binding_kind": unit.get("root_binding_kind"),
                    "root_alternative_node_ids": list(
                        unit.get("root_alternative_node_ids") or []
                    ),
                    "repair_authority": unit.get("repair_authority"),
                    "consumer_adaptation_authority": unit.get(
                        "consumer_adaptation_authority"
                    ),
                    "candidate_write_set": list(
                        unit.get("candidate_write_set") or []
                    ),
                    "patch_ready": unit.get("patch_ready"),
                    "scope_complete": unit.get("scope_complete"),
                    "scope_expandable": unit.get("scope_expandable"),
                    "scope_materializable": unit.get("scope_materializable"),
                    "editable_files": list(unit.get("editable_files") or []),
                    "escalation_files": list(unit.get("escalation_files") or []),
                    "materializable_entity_ids": list(
                        unit.get("materializable_entity_ids") or []
                    ),
                    "diagnostic_event_count": len(event_ids),
                    "diagnostic_event_ids": event_ids,
                    "continuation": unit.get("continuation") or {},
                }.items()
                if value not in (None, "", [], {})
            }
        )
    return compact


def _causal_graph_node_view(node: dict[str, Any]) -> dict[str, Any]:
    extent = node.get("extent") if isinstance(node.get("extent"), dict) else {}
    return {
        key: value
        for key, value in {
            "id": node.get("id"),
            "kind": node.get("kind"),
            "tree": node.get("tree"),
            "file": node.get("file") or node.get("path"),
            "symbol": node.get("symbol") or node.get("name"),
            "contract_key": node.get("contract_key") or node.get("key"),
            "line": node.get("line"),
            "extent": (
                {
                    "start_line": extent.get("start_line"),
                    "end_line": extent.get("end_line"),
                }
                if extent
                else None
            ),
            "diagnostic_event_id": node.get("diagnostic_event_id"),
        }.items()
        if value not in (None, "", {})
    }


def _causal_graph_edge_endpoints(
    edge: dict[str, Any],
) -> tuple[str, str]:
    source = str(edge.get("from") or edge.get("source") or "")
    target = str(edge.get("to") or edge.get("target") or "")
    return source, target


def _causal_graph_edge_view(edge: dict[str, Any]) -> dict[str, Any]:
    source, target = _causal_graph_edge_endpoints(edge)
    return {
        key: value
        for key, value in {
            "id": edge.get("id"),
            "from": source,
            "to": target,
            "type": edge.get("type") or edge.get("kind"),
            "provenance": edge.get("provenance"),
        }.items()
        if value not in (None, "", {})
    }


def _selected_causal_graph(repair_plan: dict[str, Any]) -> dict[str, Any]:
    """Render only diagnostic-to-owner paths for the selected work units."""

    unit_views: list[dict[str, Any]] = []
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        nodes_by_id: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for graph_key in ("diagnostic_graph", "code_graph_scope"):
            graph = unit.get(graph_key)
            if not isinstance(graph, dict):
                continue
            for node in graph.get("nodes") or []:
                if isinstance(node, dict) and node.get("id"):
                    nodes_by_id.setdefault(str(node["id"]), node)
            for edge in graph.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                source, target = _causal_graph_edge_endpoints(edge)
                if not source or not target:
                    continue
                identity = (
                    source,
                    target,
                    str(edge.get("type") or edge.get("kind") or ""),
                )
                if identity in seen_edges:
                    continue
                seen_edges.add(identity)
                edges.append(edge)

        decision = (
            unit.get("repair_decision")
            if isinstance(unit.get("repair_decision"), dict)
            else {}
        )
        root = (
            unit.get("code_graph_root_cause_cluster")
            if isinstance(unit.get("code_graph_root_cause_cluster"), dict)
            else unit.get("root_cause_cluster")
            if isinstance(unit.get("root_cause_cluster"), dict)
            else {}
        )
        target_ids = {
            str(value)
            for value in (
                decision.get("root_node_id"),
                root.get("node_id"),
                root.get("entity_id"),
            )
            if str(value or "")
        }
        for owner in unit.get("repair_owners") or []:
            if not isinstance(owner, dict):
                continue
            for key in (
                "root_node_id",
                "node_id",
                "entity_id",
                "graph_entity_id",
            ):
                if str(owner.get(key) or ""):
                    target_ids.add(str(owner[key]))

        diagnostic_event_ids = {
            str(item)
            for item in (
                list(decision.get("diagnostic_event_ids") or [])
                + list(root.get("diagnostic_event_ids") or [])
            )
            if str(item)
        }
        diagnostic_event_ids.update(
            str(item.get("diagnostic_event_id"))
            for item in unit.get("source_issue_samples") or []
            if isinstance(item, dict) and item.get("diagnostic_event_id")
        )
        diagnostic_ids = {
            node_id
            for node_id, node in nodes_by_id.items()
            if str(node.get("kind") or "").strip().lower() == "diagnostic"
            or node_id in diagnostic_event_ids
            or str(node.get("diagnostic_event_id") or "")
            in diagnostic_event_ids
        }

        adjacency: dict[str, list[tuple[str, int]]] = {}
        for edge_index, edge in enumerate(edges):
            source, target = _causal_graph_edge_endpoints(edge)
            adjacency.setdefault(source, []).append((target, edge_index))
            adjacency.setdefault(target, []).append((source, edge_index))

        present_targets = target_ids & set(nodes_by_id)
        selected_node_ids = set(present_targets)
        selected_edge_indexes: set[int] = set()
        unresolved_diagnostics: list[str] = []
        for diagnostic_id in sorted(diagnostic_ids):
            selected_node_ids.add(diagnostic_id)
            if not present_targets:
                unresolved_diagnostics.append(diagnostic_id)
                continue
            queue = [diagnostic_id]
            cursor = 0
            predecessor: dict[str, tuple[str, int] | None] = {
                diagnostic_id: None
            }
            reached_target: str | None = None
            while cursor < len(queue):
                current = queue[cursor]
                cursor += 1
                if current in present_targets:
                    reached_target = current
                    break
                for neighbor, edge_index in adjacency.get(current, []):
                    if neighbor in predecessor:
                        continue
                    predecessor[neighbor] = (current, edge_index)
                    queue.append(neighbor)
            if reached_target is None:
                unresolved_diagnostics.append(diagnostic_id)
                continue
            current = reached_target
            selected_node_ids.add(current)
            while predecessor[current] is not None:
                previous, edge_index = predecessor[current]  # type: ignore[misc]
                selected_edge_indexes.add(edge_index)
                selected_node_ids.add(previous)
                current = previous

        aligned_edge_indexes = {
            edge_index
            for edge_index, edge in enumerate(edges)
            if str(edge.get("type") or edge.get("kind") or "")
            == "same_contract_as"
            and (
                _causal_graph_edge_endpoints(edge)[0]
                in selected_node_ids
                or _causal_graph_edge_endpoints(edge)[1]
                in selected_node_ids
            )
        }
        for edge_index in aligned_edge_indexes:
            source, target = _causal_graph_edge_endpoints(edges[edge_index])
            selected_edge_indexes.add(edge_index)
            selected_node_ids.update((source, target))

        if not diagnostic_ids:
            selected_node_ids.update(present_targets)
        selected_nodes = [
            _causal_graph_node_view(nodes_by_id[node_id])
            for node_id in sorted(selected_node_ids)
            if node_id in nodes_by_id
        ]
        selected_edges = [
            _causal_graph_edge_view(edges[index])
            for index in sorted(selected_edge_indexes)
        ]
        graph_available = bool(nodes_by_id)
        unit_views.append(
            {
                "plan_unit_id": unit.get("id"),
                "work_unit_id": unit.get("work_unit_id"),
                "root_node_ids": sorted(target_ids),
                "diagnostic_event_ids": sorted(diagnostic_event_ids),
                "nodes": selected_nodes,
                "edges": selected_edges,
                "causal_path_complete": bool(
                    graph_available
                    and (not diagnostic_ids or not unresolved_diagnostics)
                    and (not target_ids or present_targets)
                ),
                "unresolved_diagnostic_node_ids": unresolved_diagnostics,
                "source_graph_node_count": len(nodes_by_id),
                "source_graph_edge_count": len(edges),
            }
        )
    return {
        "schema": "selected_causal_graph_v1",
        "units": unit_views,
    }


_SEMANTIC_OBLIGATION_FIELDS = (
    "id",
    "work_unit_id",
    "unit_kind",
    "component_work_unit_ids",
    "title",
    "scope",
    "semantic_gap_ids",
    "required_semantic_gap_ids",
    "pending_semantic_gap_ids",
    "blocked_semantic_gap_ids",
    "available_semantic_gap_ids",
    "semantic_transaction_ids",
    "incident_patch_ids",
    "semantic_transactions",
    "integration_obligations",
    "required_integration_obligation_ids",
    "integration_coverage_blockers",
    "semantic_recovery_context",
    "semantic_replan_lineage",
    "semantic_contract_binding",
    "semantic_contracts",
    "difference_atoms",
    "difference_atom_ids",
    "contract_evidence",
    "contract_evidence_ids",
    "contract_identities",
    "demand_anchors",
    "demand_anchor_selection",
    "repair_loci",
    "dependency_evidence",
    "depends_on_work_unit_ids",
    "dependency_root_node_ids",
    "repair_decision",
    "repair_owners",
    "provider_resolutions",
    "diagnostic_mappings",
    "root_cause_cluster",
    "root_cause_alternatives",
    "code_graph_root_cause_cluster",
    "localization",
    "patch_targets",
    "validation",
)


def _prompt_root_binding(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    alternatives = [
        item
        for item in value.get("alternatives") or []
        if isinstance(item, dict)
    ]
    return {
        key: field
        for key, field in {
            "id": value.get("id"),
            "key": value.get("key"),
            "node_id": value.get("node_id"),
            "entity_id": value.get("entity_id"),
            "title": value.get("title"),
            "scope": value.get("scope"),
            "root_shape": value.get("root_shape"),
            "ambiguous": value.get("ambiguous"),
            "unmapped": value.get("unmapped"),
            "semantic_gap_ids": value.get(
                "semantic_gap_ids"
            ),
            "diagnostic_event_ids": value.get(
                "diagnostic_event_ids"
            ),
            "alternative_count": len(alternatives),
            "alternative_node_ids": [
                str(
                    item.get("node_id")
                    or item.get("entity_id")
                    or ""
                )
                for item in alternatives
                if str(
                    item.get("node_id")
                    or item.get("entity_id")
                    or ""
                )
            ],
        }.items()
        if field not in (None, "", [], {})
    }


def _prompt_integration_obligation(
    value: object,
) -> dict[str, Any]:
    """Project one exact incoming edge without duplicating graph records."""

    if not isinstance(value, dict):
        return {}
    alternatives = [
        {
            key: field
            for key, field in {
                "donor_reference_id": item.get("donor_reference_id"),
                "source_file": item.get("source_file"),
                "translation_unit_file": item.get(
                    "translation_unit_file"
                ),
                "line": item.get("line"),
                "column": item.get("column"),
                "cursor_kind": item.get("cursor_kind"),
                "use_kind": item.get("use_kind"),
                "owner_contract_key": item.get(
                    "owner_contract_key"
                ),
                "target_contract_key": item.get(
                    "target_contract_key"
                ),
            }.items()
            if field not in (None, "", [], {})
        }
        for item in value.get("alternatives") or []
        if isinstance(item, dict)
    ]
    current_matches = [
        {
            key: field
            for key, field in {
                "reference_id": item.get("reference_id"),
                "source_file": item.get("source_file"),
                "line": item.get("line"),
                "column": item.get("column"),
                "owner_contract_key": item.get(
                    "owner_contract_key"
                ),
                "target_contract_key": item.get(
                    "target_contract_key"
                ),
            }.items()
            if field not in (None, "", [], {})
        }
        for item in value.get("current_matches") or []
        if isinstance(item, dict)
    ]
    return {
        key: field
        for key, field in {
            "id": value.get("id"),
            "status": value.get("status"),
            "target_contract_key": value.get(
                "target_contract_key"
            ),
            "target_symbol": value.get("target_symbol"),
            "target_kind": value.get("target_kind"),
            "target_file": value.get("target_file"),
            "target_entity": _prompt_entity_identity(
                value.get("target_entity")
            ),
            "selected_owner_contract_key": value.get(
                "selected_owner_contract_key"
            ),
            "selected_owner_entity": _prompt_entity_identity(
                value.get("selected_owner_entity")
            ),
            "selection_basis": value.get("selection_basis"),
            "alternatives": alternatives,
            "current_matches": current_matches,
            "validation_reason": value.get("validation_reason"),
        }.items()
        if field not in (None, "", [], {})
    }


def _prompt_integration_blocker(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: field
        for key, field in {
            "id": value.get("id"),
            "status": value.get("status"),
            "reason": value.get("reason"),
            "tree": value.get("tree"),
            "file": value.get("file"),
            "seed_contract_key": value.get(
                "seed_contract_key"
            ),
            "seed_contract_keys": value.get(
                "seed_contract_keys"
            ),
            "seed_file": value.get("seed_file"),
            "parse_error": value.get("parse_error"),
            "donor_match_count": value.get("donor_match_count"),
            "candidate_files": [
                str(item)
                for item in value.get("candidate_files") or []
                if str(item)
            ],
        }.items()
        if field not in (None, "", [], {})
    }


def _prompt_semantic_transaction(
    value: object,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    obligations = [
        item
        for item in value.get("integration_obligations") or []
        if isinstance(item, dict)
    ]
    blockers = [
        item
        for item in value.get("integration_coverage_blockers") or []
        if isinstance(item, dict)
    ]
    return {
        key: field
        for key, field in {
            "transaction_id": value.get("transaction_id"),
            "source_work_unit_id": value.get(
                "source_work_unit_id"
            ),
            "gap_ids": value.get("gap_ids"),
            "primary_gap_ids": value.get("primary_gap_ids"),
            "collateral_baseline_gap_ids": value.get(
                "collateral_baseline_gap_ids"
            ),
            "touched_files": value.get("touched_files"),
            "touched_loci": value.get("touched_loci"),
            "status": value.get("status"),
            "integration_obligation_ids": [
                str(item.get("id"))
                for item in obligations
                if str(item.get("id") or "")
            ],
            "integration_coverage_blocker_ids": [
                str(item.get("id"))
                for item in blockers
                if str(item.get("id") or "")
            ],
        }.items()
        if field not in (None, "", [], {})
    }


def _prompt_semantic_recovery_context(
    value: object,
) -> dict[str, Any]:
    """Project transaction authority without duplicating unrelated gap sets."""

    if not isinstance(value, dict):
        return {}
    exact_bindings = [
        dict(item)
        for item in value.get("exact_bindings") or []
        if isinstance(item, dict)
    ]
    exact_gap_ids = {
        str(item.get("semantic_gap_id") or "")
        for item in exact_bindings
        if str(item.get("semantic_gap_id") or "")
    }
    transaction_file_bindings: list[dict[str, Any]] = []
    for raw_binding in value.get("transaction_file_bindings") or []:
        if not isinstance(raw_binding, dict):
            continue
        binding = {
            key: raw_binding.get(key)
            for key in (
                "file",
                "patch_ids",
                "source_loci",
                "authority",
                "binding_role",
            )
            if raw_binding.get(key) not in (None, "", [], {})
        }
        relevant_gap_ids = [
            str(gap_id)
            for gap_id in raw_binding.get("gap_ids") or []
            if str(gap_id) in exact_gap_ids
        ]
        if relevant_gap_ids:
            binding["exact_gap_ids"] = relevant_gap_ids
        binding["postimage_regions"] = [
            {
                **{
                    key: region.get(key)
                    for key in (
                        "file",
                        "start_line",
                        "end_line",
                        "region_kind",
                        "patch_ids",
                    )
                    if region.get(key) not in (None, "", [], {})
                },
                **(
                    {"exact_gap_ids": region_gap_ids}
                    if (
                        region_gap_ids := [
                            str(gap_id)
                            for gap_id in region.get("gap_ids") or []
                            if str(gap_id) in exact_gap_ids
                        ]
                    )
                    else {}
                ),
            }
            for region in raw_binding.get("postimage_regions") or []
            if isinstance(region, dict)
        ]
        transaction_file_bindings.append(binding)
    return {
        key: field
        for key, field in {
            "schema": value.get("schema"),
            "batch_id": value.get("batch_id"),
            "batch_status": value.get("batch_status"),
            "incident_patch_ids": value.get("incident_patch_ids"),
            "exact_incident_patch_ids": value.get(
                "exact_incident_patch_ids"
            ),
            "semantic_transaction_ids": value.get(
                "semantic_transaction_ids"
            ),
            "exact_bindings": exact_bindings,
            "transaction_file_bindings": transaction_file_bindings,
        }.items()
        if field not in (None, "", [], {})
    }


def _prompt_nonempty_record(
    value: object,
    *,
    excluded_fields: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: field
        for key, field in value.items()
        if key not in excluded_fields
        and field not in (None, "", [], {})
    }


def _prompt_repair_owner(value: object) -> dict[str, Any]:
    """Keep repair authority once, without its duplicated graph descriptor."""

    return _prompt_nonempty_record(
        value,
        excluded_fields={"diagnostic_structure"},
    )


def _prompt_localization(value: object) -> dict[str, Any]:
    """Keep localization status while patch_targets carries the exact targets."""

    return _prompt_nonempty_record(
        value,
        excluded_fields={"targets"},
    )


def _prompt_patch_target(value: object) -> dict[str, Any]:
    return _prompt_nonempty_record(value)


def _prompt_obligation_unit(
    unit: dict[str, Any],
) -> dict[str, Any]:
    projected = {
        key: unit[key]
        for key in _SEMANTIC_OBLIGATION_FIELDS
        if key in unit
    }
    projected["semantic_transactions"] = [
        _prompt_semantic_transaction(item)
        for item in unit.get("semantic_transactions") or []
        if isinstance(item, dict)
    ]
    projected["semantic_recovery_context"] = (
        _prompt_semantic_recovery_context(
            unit.get("semantic_recovery_context")
        )
    )
    projected["integration_obligations"] = [
        _prompt_integration_obligation(item)
        for item in unit.get("integration_obligations") or []
        if isinstance(item, dict)
    ]
    projected["integration_coverage_blockers"] = [
        _prompt_integration_blocker(item)
        for item in unit.get("integration_coverage_blockers") or []
        if isinstance(item, dict)
    ]
    projected["contract_evidence"] = [
        _prompt_contract_evidence(item)
        for item in unit.get("contract_evidence") or []
        if isinstance(item, dict)
    ]
    projected["demand_anchors"] = compact_prompt_references(
        unit.get("demand_anchors") or []
    )
    projected["repair_owners"] = [
        _prompt_repair_owner(item)
        for item in unit.get("repair_owners") or []
        if isinstance(item, dict)
    ]
    projected["localization"] = _prompt_localization(
        unit.get("localization")
    )
    projected["patch_targets"] = [
        _prompt_patch_target(item)
        for item in unit.get("patch_targets") or []
        if isinstance(item, dict)
    ]
    projected["root_cause_cluster"] = _prompt_root_binding(
        unit.get("root_cause_cluster")
    )
    projected["code_graph_root_cause_cluster"] = (
        _prompt_root_binding(
            unit.get("code_graph_root_cause_cluster")
        )
    )
    root_alternatives = [
        item
        for item in unit.get("root_cause_alternatives") or []
        if isinstance(item, dict)
    ]
    projected["root_cause_alternatives"] = {
        "count": len(root_alternatives),
        "node_ids": [
            str(
                item.get("node_id")
                or item.get("entity_id")
                or ""
            )
            for item in root_alternatives
            if str(
                item.get("node_id")
                or item.get("entity_id")
                or ""
            )
        ],
    }
    projected["provider_resolutions"] = [
        {
            key: field
            for key, field in {
                "backend": item.get("backend"),
                "symbol": item.get("symbol"),
                "contract_key": item.get("contract_key"),
                "status": item.get("status"),
                "preferred_patch_file": item.get(
                    "preferred_patch_file"
                ),
                "preferred_entity_id": item.get(
                    "preferred_entity_id"
                ),
                "target_entity_ids": [
                    str(entity.get("id") or "")
                    for entity in item.get("target_entities") or []
                    if isinstance(entity, dict)
                    and str(entity.get("id") or "")
                ],
                "donor_entity_ids": [
                    str(entity.get("id") or "")
                    for entity in item.get("donor_entities") or []
                    if isinstance(entity, dict)
                    and str(entity.get("id") or "")
                ],
                "reason": item.get("reason"),
            }.items()
            if field not in (None, "", [], {})
        }
        for item in unit.get("provider_resolutions") or []
        if isinstance(item, dict)
    ]
    return {
        key: value
        for key, value in projected.items()
        if value not in (None, "", [], {})
    }


def _selected_obligation_evidence(
    repair_plan: dict[str, Any],
) -> dict[str, Any]:
    """Project the exact decision frontier without duplicating audit payloads."""

    units: list[dict[str, Any]] = []
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        units.append(_prompt_obligation_unit(unit))
    return {
        "schema": "semantic_obligation_evidence_v2",
        "projection": (
            "exact unresolved relation incidence; complete aligned-reference "
            "records remain content-addressed in the repair-plan artifact"
        ),
        "units": units,
    }


def _prompt_source_records(
    repair_plan: dict[str, Any],
    evidence_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        source_pack = (
            unit.get("repair_source_pack")
            if isinstance(unit.get("repair_source_pack"), dict)
            else {}
        )
        for section, role in (
            ("allowed_edit_sources", "old_edit_source"),
            ("donor_reference_sources", "donor_reference_source"),
            ("read_only_evidence_sources", "read_only_source"),
        ):
            for raw in source_pack.get(section) or []:
                if not isinstance(raw, dict):
                    continue
                record = source_evidence_record(
                    raw,
                    role=role,
                )
                if record is not None:
                    records.append(record)

    def add_dynamic_source(
        raw: dict[str, Any],
        *,
        tree: object,
        role: str,
        symbol: object = None,
    ) -> None:
        source = dict(raw)
        source.setdefault("tree", tree)
        source.setdefault("symbol", symbol)
        record = source_evidence_record(
            source,
            role=role,
        )
        if record is not None:
            records.append(record)

    for entry in evidence_log:
        boundary = entry.get("source_boundary_context") or {}
        if isinstance(boundary, dict):
            for source in boundary.get("files") or []:
                if isinstance(source, dict):
                    add_dynamic_source(
                        {**source, "edit_permission": False},
                        tree="old", role="read_only_source",
                    )
        if entry.get("kind") != "inspect_result":
            continue
        evidence = (
            entry.get("evidence")
            if isinstance(entry.get("evidence"), dict)
            else {}
        )
        for file_record in evidence.get("files") or []:
            if isinstance(file_record, dict):
                add_dynamic_source(
                    file_record,
                    tree=file_record.get("tree"),
                    role="dynamic_file_inspection",
                )
        for entity in evidence.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            source = entity.get("source")
            if isinstance(source, dict):
                add_dynamic_source(
                    source,
                    tree=entity.get("tree"),
                    role="dynamic_entity_inspection",
                    symbol=entity.get("symbol"),
                )
            for variant in entity.get("variants") or []:
                if not isinstance(variant, dict):
                    continue
                variant_source = variant.get("source")
                if isinstance(variant_source, dict):
                    add_dynamic_source(
                        variant_source,
                        tree=variant.get("tree") or entity.get("tree"),
                        role="dynamic_entity_variant_inspection",
                        symbol=variant.get("symbol") or entity.get("symbol"),
                    )
    return records


def latest_agentic_failure_delta(
    evidence_log: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for entry in reversed(evidence_log):
        kind = str(entry.get("kind") or "")
        if kind in {"", "inspect_result", "graph_scope_expansion_result"}:
            continue
        compact: dict[str, Any] = {
            key: entry.get(key)
            for key in (
                "kind",
                "step",
                "status",
                "reason",
                "error",
                "recovery_kind",
                "recovery_status",
                "previous_blocker_kind",
                "action_fingerprint",
                "evidence_state_fingerprint",
                "failure_origin_kind",
                "typed_apply_failures",
                "required_next_action",
                "patch_conflict_evidence",
            )
            if entry.get(key) not in (None, "", [], {})
        }
        proof = entry.get("dependency_closure")
        if isinstance(proof, dict):
            compact["materialization_dependency_blocker"] = {
                **{key: proof.get(key) for key in (
                    "status", "reason", "path", "contract_key",
                )},
                "unresolved": [
                    {key: item.get(key) for key in (
                        "source", "status", "reason", "symbol", "kind",
                        "contract_key", "provider_file", "provider_entity_id",
                        "provider_identity_complete", "resolution_basis",
                        "old_candidate_count", "donor_candidate_count",
                    ) if key in item}
                    for item in proof.get("unresolved") or []
                    if isinstance(item, dict)
                ],
            }
        boundary = entry.get("source_boundary_context")
        if isinstance(boundary, dict):
            compact["source_boundary_context"] = {
                key: boundary.get(key) for key in (
                    "path", "contract_key", "old_neighbor_contracts", "old_interval",
                ) if key in boundary
            }
        action = entry.get("action")
        if isinstance(action, dict):
            compact["root_binding"] = _agentic_action_root_binding(action)
            compact["action_kind"] = action.get("action")
        for nested_key in ("redundancy", "recovery"):
            nested = (
                entry.get(nested_key)
                if isinstance(entry.get(nested_key), dict)
                else {}
            )
            if nested:
                compact[nested_key] = {
                    key: nested.get(key)
                    for key in (
                        "status",
                        "reason",
                        "recovery_kind",
                        "previous_blocker_kind",
                        "action_fingerprint",
                        "evidence_state_fingerprint",
                        "source_evidence_fingerprint",
                    )
                    if nested.get(key) not in (None, "", [], {})
                }
        evidence_gate = (
            entry.get("evidence_gate")
            if isinstance(entry.get("evidence_gate"), dict)
            else {}
        )
        if evidence_gate:
            compact["evidence_gate"] = evidence_gate
        apply_summary = (
            entry.get("apply_summary")
            if isinstance(entry.get("apply_summary"), dict)
            else {}
        )
        if apply_summary:
            compact["apply_summary"] = {
                key: apply_summary.get(key)
                for key in (
                    "status",
                    "applied_count",
                    "incomplete_count",
                    "needs_escalation_count",
                )
                if apply_summary.get(key) not in (None, "", [], {})
            }
            compact["apply_failures"] = [
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "error",
                        "reason",
                        "apply_failure",
                        "invalid_route",
                    )
                    if result.get(key) not in (None, "", [], {})
                }
                for result in apply_summary.get("results") or []
                if isinstance(result, dict)
            ]
        return compact
    return None


def _optional_agentic_prompt_records(
    repair_plan: dict[str, Any],
    evidence_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "id": "agentic-evidence-summary",
            "kind": "evidence_summary",
            "value": evidence_summary,
        }
    ]
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        attempts = [
            attempt
            for attempt in unit.get("prior_unresolved_patch_attempts") or []
            if isinstance(attempt, dict)
        ]
        for index, attempt in enumerate(attempts[-2:]):
            records.append(
                {
                    "id": stable_id(
                        "prior-agentic-attempt",
                        str(unit.get("id") or unit.get("work_unit_id") or ""),
                        str(len(attempts) - len(attempts[-2:]) + index),
                    ),
                    "kind": "prior_unresolved_patch_attempt",
                    "plan_unit_id": unit.get("id"),
                    "work_unit_id": unit.get("work_unit_id"),
                    "value": {
                        key: attempt.get(key)
                        for key in (
                            "source",
                            "cycle",
                            "status",
                            "agent_step",
                            "blocker_reason",
                            "blocker_kind",
                            "blocked_apply_kind",
                            "recovery_kind",
                            "recovery_status",
                            "diagnostic_component_match",
                            "touched_files",
                            "patch_delta_summary",
                            "diagnostic_snapshot",
                            "unresolved_diagnostic_signatures",
                            "resolved_diagnostic_signatures_since_attempt",
                            "new_diagnostic_signatures_since_attempt",
                            "note",
                        )
                        if attempt.get(key) not in (None, "", [], {})
                    },
                }
            )
    return records


def build_agentic_prompt(
    spec: CaseSpec,
    *,
    validation_report: dict[str, Any],
    repair_plan: dict[str, Any],
    scope: dict[str, Any],
    evidence_log: list[dict[str, Any]],
    repair_round: int,
    step: int,
    max_steps: int | None = None,
) -> str:
    evidence_summary = summarize_agentic_evidence_log(evidence_log)
    remaining_steps = max(0, int(max_steps or step) - step) if max_steps else None
    current_diagnostic_context, diagnostic_context_input_count = (
        _selected_current_diagnostic_context(
            spec,
            validation_report,
            repair_plan,
        )
    )
    control_and_root_scope = {
        "case": {
            "case_id": spec.case_id,
            "driver_root": spec.driver_root,
            "entry_files": spec.entry_files,
            "expected_modules": spec.expected_modules,
            "expected_objects": spec.expected_objects,
        },
        "repair_round": repair_round,
        "agent_step": step,
        "remaining_agent_steps_after_this": remaining_steps,
        "agentic_control": {
            "inspect_results_available": evidence_summary["inspect_result_count"],
            "scope_expansions_completed": evidence_summary["scope_expansion_result_count"],
            "instruction": (
                "If inspect_results_available is greater than 0, prefer apply_patch, needs_escalation, or stop. "
                "Another inspect is valid only for a graph-authorized file range not already covered in evidence_summary."
            ),
        },
        "edit_scope": {
            "editable_files": scope.get("editable_files") or [],
            "escalation_files": scope.get("escalation_files") or [],
            "read_only_evidence_files": scope.get("read_only_evidence_files") or [],
            "inspectable_files": scope.get("inspectable_files") or [],
            "graph_units": _compact_agentic_graph_units(scope),
        },
    }
    current_diagnostics = {
        "validation": _compact_validation(validation_report),
        "diagnostic_graph_bindings": current_diagnostic_context,
    }
    evidence_view = assemble_prompt_evidence_view(
        control_state=control_and_root_scope,
        diagnostics=current_diagnostics,
        causal_graph=_selected_causal_graph(repair_plan),
        obligations=_selected_obligation_evidence(repair_plan),
        source_records=_prompt_source_records(repair_plan, evidence_log),
        latest_failure_delta=latest_agentic_failure_delta(evidence_log),
        inspectable_ranges=[
            item
            for item in scope.get("inspectable_ranges") or []
            if isinstance(item, dict)
        ],
        inspectable_entities=[
            item
            for item in scope.get("inspectable_entities") or []
            if isinstance(item, dict)
        ],
        optional_records=_optional_agentic_prompt_records(
            repair_plan,
            evidence_summary,
        ),
    )
    evidence_view["metrics"]["diagnostic_context_input_count"] = (
        diagnostic_context_input_count
    )
    evidence_view["metrics"]["diagnostic_context_rendered_count"] = len(
        current_diagnostic_context
    )
    scope["prompt_evidence_policy"] = dict(
        evidence_view.get("evidence_policy") or {}
    )
    scope["prompt_evidence_metrics"] = dict(
        evidence_view.get("metrics") or {}
    )
    scope["prompt_evidence_schema"] = evidence_view.get("schema")
    schema = {
        "inspect": {
            "action": "inspect",
            "rationale": "why more evidence is needed",
            "inspect": {
                "entities": [
                    {
                        "id": "exact id from inspectable_entities",
                        "tree": "old|donor|both",
                    }
                ],
                "files": [
                    {"tree": "old|donor|both", "path": "drivers/...", "startline": 1, "endline": 120}
                ],
                "logs": True,
            },
        },
        "expand_scope": {
            "action": "expand_scope",
            "rationale": "why the stored graph frontier must advance before patching",
            "scope_expansion": {
                "work_unit_id": "exact id from edit_scope.graph_units",
                "root_node_id": "exact root id from the same graph unit",
            },
        },
        "materialize_entity_from_donor": {
            "action": "materialize_entity_from_donor",
            "rationale": (
                "why synchronizing this complete graph-selected donor contract "
                "is safer than constructing a partial hunk"
            ),
            "decision": {
                "mode": "patch_root_subgraph",
                "work_unit_id": "exact id from edit_scope.graph_units",
                "root_node_id": (
                    "exact CodeIndex root id, or null for a mapped root set"
                ),
                "root_key": "exact root_key from the same graph unit",
            },
            "entity": {
                "id": (
                    "exact id listed by the selected graph unit's "
                    "materializable_entity_ids"
                )
            },
        },
        "apply_patch": {
            "action": "apply_patch",
            "rationale": "root cause and why this patch is the next validation-worthy step",
            "decision": {
                "mode": "patch_root_subgraph",
                "work_unit_id": "exact id from edit_scope.graph_units",
                "root_node_id": "exact CodeIndex root id, or null for a mapped root set",
                "root_key": "exact root_key from the same graph unit",
                "addressed_semantic_gap_ids": [
                    "exact gap ids this patch is intended to satisfy"
                ],
            },
            "patch_plan": {
                "strategy": "minimal compatibility edit",
                "affected_files": ["drivers/..."],
                "expected_outcome": "which compile/link error should disappear",
            },
            "patch": {
                "format": "unified_diff",
                "content": "--- a/path\\n+++ b/path\\n@@ ...",
            },
        },
        "needs_escalation": {
            "action": "needs_escalation",
            "rationale": "why editable scope is insufficient or evidence is contradictory",
            "missing_context": {
                "work_unit_id": "exact graph work unit id",
                "continuation": {},
                "reason": "graph is incomplete, ambiguous, or permission blocked",
            },
        },
        "stop": {
            "action": "stop",
            "rationale": "why no further graph-authorized action is useful in this run",
            "reason": "terminal controller reason",
        },
    }
    rendered_evidence_view, transport_instruction, transport_metrics = (
        render_prompt_payload(evidence_view)
    )
    scope["prompt_evidence_metrics"].update(
        {
            "transport_content_store_applied": transport_metrics["applied"],
            "transport_content_record_count": transport_metrics["record_count"],
            "transport_original_serialized_bytes": transport_metrics[
                "original_serialized_bytes"
            ],
            "transport_encoded_serialized_bytes": transport_metrics[
                "encoded_serialized_bytes"
            ],
            "transport_saved_serialized_bytes": transport_metrics[
                "saved_serialized_bytes"
            ],
        }
    )
    prompt_text = "\n".join(
        [
            "You are an autonomous Linux driver backport repair controller.",
            "The CodeIndex work-unit partition and its root binding are binding repair ownership facts.",
            "First request missing evidence when needed; only produce a patch after old-tree context and donor evidence are sufficient.",
            "All required evidence in this atomic work unit is present losslessly. Navigation entries are exact handles bound to this root and may be retrieved with inspect.",
            "Build-preparation diagnostics are handled by the build/config preflight tool, not by source repair patches.",
            "Scope expansion and source compatibility may be handled in the same round when the code graph gives them a common root.",
            "Select exactly one root-mapped work unit. For a single root copy work_unit_id, root_node_id, and root_key; for mapped_root_set copy work_unit_id and root_key with root_node_id=null. Scope expansion is only valid for a non-null single root_node_id.",
            "Use expand_scope only when graph_units marks scope_expandable or scope_materializable; the controller advances CodeIndex or materializes its exact graph owner without accepting guessed files or symbols.",
            "Patch the mapped common-upstream root subgraph; do not choose an owner from diagnostic wording or patch only a symptom line.",
            "When a donor-only missing include/provider root is read-only and the graph marks an exact old-tree consumer_integration_owner, adapt the missing include/use compatibility subgraph in that consumer. This owner is graph-proven integration scope, not a symptom-line exception; do not stop merely because the donor shared file is unwritable.",
            "Never invent a same-named compatibility macro, wrapper, or stub merely to clear a missing-provider diagnostic. A replacement provider is valid only when exact evidence supplies its complete behavior and every direct dependency is realizable in the old tree; otherwise adapt all graph-bound consumers to existing target primitives, preserve acquisition/release and every exit path, and delete any obsolete shim.",
            "Files listed in edit_scope.escalation_files are explicitly writable in this run because the user authorized their exact graph-localized repair owners. Do not return needs_escalation for permission to edit those files.",
            "A semantic-repair unit must use a patch or explicit escalation; it cannot be downgraded to resolved. Compile-valid semantic review runs through a separate one-call evidence protocol before this repair controller is entered.",
            "A compiler unit with required_semantic_gap_ids is recovery work for contracts already reviewed as change-required; pending_semantic_gap_ids identifies the subset with an applied transaction awaiting validation. Repair the diagnostic while preserving or completing those exact contract obligations. Deleting or reverting the migrated entity is not a valid resolution unless the same patch supplies an alternative implementation that satisfies the bound contract; compile success alone never closes a required semantic gap.",
            "Prefer inspect.entities for exact current definitions and types. Entity ids must come from inspectable_entities; the controller re-resolves their current structural extent after prior edits.",
            "Use materialize_entity_from_donor for a graph-selected donor definition whose exact contract is absent from the current old tree. The controller inserts that complete donor entity only at a unique order-preserving structural gap. For an existing old-tree provider, this tool may only add an exact graph-proven linkage/export contract when old and donor implementations are already identical; it never overwrites an existing implementation. Otherwise adapt the consumer or submit an explicit evidence-backed apply_patch.",
            "The controller, not the model, resolves current extents or insertion anchors and generates the unified diff; never guess a file, symbol, placement, or line range for materialize_entity_from_donor.",
            "An entity inspection with status resolved_variants contains distinct preprocessor-context implementations of one contract. Preserve every relevant branch and use each variant's old counterpart alignment; do not collapse the family to one unconditional stub.",
            "Inspect file ranges only when explicitly listed in inspectable_ranges. Symbol or literal full-tree searches are unavailable.",
            "Do not patch Kconfig, Makefile, or Kbuild from this source-compatibility controller.",
            "Do not repeat an inspect range already covered by the evidence_summary record.",
            "Read prior_unresolved_patch_attempt records before patching. When the same diagnostic component remains open, do not replay the same edit; account for the recorded residual and changed diagnostic signatures.",
            "For a typed apply failure, perform invalid_route.required_transition before retrying. outer_unit_replan is not permission to skip the unit: inspect, expand the bound graph, or produce a distinct graph-grounded edit that satisfies that transition.",
            "When prior inspect evidence exists and the remaining step budget is low, choose apply_patch, needs_escalation, or stop instead of another inspect.",
            "Only files explicitly listed in the selected graph unit's editable_files or escalation_files are writable, and only when patch_ready=true. read_only_evidence_files and inspectable_files are evidence only; graph reachability or inspection never grants patch authority.",
            "Return exactly one JSON object. No prose outside JSON.",
            "",
            "Allowed actions and JSON schemas:",
            json.dumps(
                schema,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "",
            "Prompt evidence view. Each source body is rendered at most once; "
            "inspectable ranges are [tree, path, startline, endline] and "
            "inspectable entities are [id, tree, path, kind, symbol, "
            "contract_key, is_definition]:",
            *(
                ["Prompt transport encoding: " + transport_instruction]
                if transport_instruction
                else []
            ),
            rendered_evidence_view,
        ]
    )
    scope["prompt_evidence_metrics"]["rendered_prompt_chars"] = len(
        prompt_text
    )
    scope["prompt_evidence_metrics"]["rendered_prompt_bytes"] = len(
        prompt_text.encode("utf-8")
    )
    return prompt_text


def _safe_read_lines(path: Path) -> list[str] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


def _render_file_excerpt(root: Path, relative_path: str, *, startline: int, endline: int) -> dict[str, Any]:
    path = root / normalize_relpath(relative_path)
    lines = _safe_read_lines(path)
    if lines is None:
        return {"path": relative_path, "exists": False}
    start = max(1, int(startline or 1))
    end = min(len(lines), max(start, int(endline or min(len(lines), start + 120))))
    return {
        "path": relative_path,
        "exists": True,
        "line_start": start,
        "line_end": end,
        "text": "\n".join(f"{line_no}: {lines[line_no - 1]}" for line_no in range(start, end + 1)),
    }


def _file_roots_for_tree(spec: CaseSpec, tree: str) -> list[tuple[str, Path]]:
    if tree == "old":
        return [("old", spec.old_root_path)]
    if tree == "donor":
        return [("donor", spec.primary_donor_root)]
    return [("old", spec.old_root_path), ("donor", spec.primary_donor_root)]


def _normalize_file_request(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        return {"tree": "both", "path": raw, "startline": 1, "endline": 160}
    if not isinstance(raw, dict):
        return None
    path = str(raw.get("path") or raw.get("file") or "").strip()
    if not path:
        return None
    try:
        start = max(1, int(raw.get("startline") or raw.get("start") or 1))
        requested_end = int(raw.get("endline") or raw.get("end") or (start + 159))
    except (TypeError, ValueError):
        return None
    end = max(start, requested_end)
    end = min(end, start + AGENTIC_INSPECT_MAX_LINES - 1)
    return {
        "tree": str(raw.get("tree") or "both").strip().lower(),
        "path": normalize_relpath(path),
        "startline": start,
        "endline": end,
    }


def _inspect_range_intersections(
    scope: dict[str, Any],
    *,
    tree: str,
    relative_path: str,
    startline: int,
    endline: int,
) -> list[tuple[int, int]]:
    normalized_tree = _normalized_tree(tree)
    normalized_path = normalize_relpath(relative_path)
    intervals = sorted(
        (
            int(record.get("line_start") or 0),
            int(record.get("line_end") or 0),
        )
        for record in scope.get("inspectable_ranges") or []
        if isinstance(record, dict)
        and _normalized_tree(record.get("tree")) == normalized_tree
        and normalize_relpath(str(record.get("path") or "")) == normalized_path
        and int(record.get("line_start") or 0) > 0
        and int(record.get("line_end") or 0) >= int(record.get("line_start") or 0)
    )
    intersections: list[tuple[int, int]] = []
    for interval_start, interval_end in intervals:
        clipped_start = max(startline, interval_start)
        clipped_end = min(endline, interval_end)
        if clipped_end < clipped_start:
            continue
        if intersections and clipped_start <= intersections[-1][1] + 1:
            intersections[-1] = (
                intersections[-1][0],
                max(intersections[-1][1], clipped_end),
            )
        else:
            intersections.append((clipped_start, clipped_end))
    return intersections


def _normalize_entity_request(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    entity_id = str(raw.get("id") or "").strip()
    if not entity_id:
        return None
    tree = str(raw.get("tree") or "").strip().lower()
    if tree in {"target", "old"}:
        tree = "old"
    elif tree in {"new", "donor"}:
        tree = "donor"
    elif tree != "both":
        tree = ""
    return {"id": entity_id, "tree": tree}


def _entity_matches(
    report: dict[str, Any],
    descriptor: dict[str, Any],
) -> list[dict[str, Any]]:
    entities = [
        item for item in report.get("entities") or [] if isinstance(item, dict)
    ]
    contract = str(descriptor.get("contract_key") or "").strip()
    if contract:
        matches = [
            item
            for item in entities
            if str(item.get("contract_key") or item.get("key") or "").strip()
            == contract
        ]
    else:
        kind = _structural_entity_kind(descriptor)
        symbol = str(descriptor.get("symbol") or "").strip()
        matches = [
            item
            for item in entities
            if _structural_entity_kind(item) == kind
            and str(item.get("symbol") or item.get("name") or "").strip()
            == symbol
        ]
    if descriptor.get("is_definition") is not None:
        expected_definition = bool(descriptor.get("is_definition"))
        matches = [
            item
            for item in matches
            if bool(item.get("is_definition")) == expected_definition
        ]
    return matches


def _macro_generated_representation_authority(
    report: dict[str, Any],
    descriptor: dict[str, Any],
) -> dict[str, Any] | None:
    descriptor_extent = (
        descriptor.get("extent")
        if isinstance(descriptor.get("extent"), dict)
        else {}
    )
    descriptor_start_line = int(
        descriptor_extent.get("start_line")
        or descriptor.get("line_start")
        or 0
    )
    descriptor_end_line = int(
        descriptor_extent.get("end_line")
        or descriptor.get("line_end")
        or descriptor_start_line
    )
    descriptor_start_column = int(
        descriptor_extent.get("start_column") or 0
    )
    descriptor_end_column = int(
        descriptor_extent.get("end_column") or 0
    )
    precise_columns = bool(
        descriptor_start_column > 0 and descriptor_end_column > 0
    )
    symbol = str(descriptor.get("symbol") or "").strip()
    contract = str(descriptor.get("contract_key") or "").strip()
    owner_file = normalize_relpath(str(descriptor.get("path") or ""))
    if (
        not symbol
        or not contract
        or not owner_file
        or descriptor_start_line <= 0
        or descriptor_end_line < descriptor_start_line
    ):
        return None

    matches: list[dict[str, Any]] = []
    for anchor in report.get("source_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        anchor_extent = (
            anchor.get("extent")
            if isinstance(anchor.get("extent"), dict)
            else {}
        )
        anchor_start = (
            int(anchor_extent.get("start_line") or 0),
            int(anchor_extent.get("start_column") or 0),
        )
        anchor_end = (
            int(anchor_extent.get("end_line") or 0),
            int(anchor_extent.get("end_column") or 0),
        )
        if (
            str(anchor.get("kind") or "")
            != "top_level_macro_declaration"
            or normalize_relpath(str(anchor.get("file") or ""))
            != owner_file
            or symbol
            not in {
                str(item)
                for item in anchor.get("source_identifiers") or []
            }
            or anchor_start[0] <= 0
            or not (
                (
                    anchor_start
                    <= (
                        descriptor_start_line,
                        descriptor_start_column,
                    )
                    and (
                        descriptor_end_line,
                        descriptor_end_column,
                    )
                    <= anchor_end
                )
                if precise_columns
                else (
                    anchor_start[0] <= descriptor_start_line
                    and descriptor_end_line <= anchor_end[0]
                )
            )
        ):
            continue
        matches.append(anchor)
    if len(matches) != 1:
        return None
    anchor = matches[0]
    return {
        "schema": "target_representation_adaptation_authority_v1",
        "status": "authorized",
        "contract_key": contract,
        "entity_kind": _structural_entity_kind(descriptor),
        "owner_file": owner_file,
        "donor_entity_id": str(
            descriptor.get("graph_entity_id")
            or descriptor.get("entity_id")
            or descriptor.get("id")
            or ""
        ),
        "source_anchor_id": str(anchor.get("id") or ""),
        "source_anchor_kind": str(anchor.get("kind") or ""),
        "source_anchor_symbol": str(anchor.get("symbol") or ""),
        "source_anchor_extent": dict(anchor.get("extent") or {}),
        "reason": "exact_donor_root_is_generated_by_macro_declaration",
    }


def _unique_structural_definition(
    matches: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if len(matches) == 1:
        return matches[0]
    definitions = [
        match
        for match in matches
        if match.get("is_definition") is True
    ]
    return definitions[0] if len(definitions) == 1 else None


def _conditional_variant_family(
    matches: list[dict[str, Any]],
    descriptor: dict[str, Any],
) -> bool:
    if len(matches) < 2:
        return False
    first = matches[0]
    expected_identity = (
        _structural_entity_kind(descriptor) or _structural_entity_kind(first),
        str(
            descriptor.get("symbol")
            or first.get("symbol")
            or first.get("name")
            or ""
        ).strip(),
        str(
            descriptor.get("contract_key")
            or first.get("contract_key")
            or first.get("key")
            or ""
        ).strip(),
    )
    identities = {
        (
            _structural_entity_kind(match),
            str(match.get("symbol") or match.get("name") or "").strip(),
            str(match.get("contract_key") or match.get("key") or "").strip(),
        )
        for match in matches
    }
    contexts = [
        tuple(str(item) for item in match.get("conditional_context") or [])
        for match in matches
    ]
    ordered_contexts = sorted(contexts, key=lambda item: (len(item), item))
    base_context = ordered_contexts[0] if ordered_contexts else ()
    base_count = sum(context == base_context for context in ordered_contexts)
    structurally_alternative = bool(base_context) and base_count == 1 and all(
        context == base_context
        or (
            len(context) > len(base_context)
            and context[: len(base_context)] == base_context
            and context[len(base_context)].startswith(
                ("preproc_else:", "preproc_elif:")
            )
        )
        for context in ordered_contexts
    )
    return (
        identities == {expected_identity}
        and len(set(contexts)) == len(contexts)
        and structurally_alternative
    )


def _old_alignment_sources(
    spec: CaseSpec,
    *,
    relative_path: str,
    alignment: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    counterpart = alignment.get("old_counterpart")
    if isinstance(counterpart, dict):
        candidates.append(("old_counterpart", counterpart))
    for role in ("predecessor", "successor"):
        pair = alignment.get(role)
        if isinstance(pair, dict) and isinstance(pair.get("old"), dict):
            candidates.append((f"old_{role}", pair["old"]))

    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for role, entity in candidates:
        extent = entity.get("extent") if isinstance(entity.get("extent"), dict) else {}
        start = int(extent.get("start_line") or 0)
        end = int(extent.get("end_line") or start)
        identity = (start, end)
        if start <= 0 or identity in seen:
            continue
        seen.add(identity)
        records.append(
            {
                "role": role,
                "entity": entity,
                "source": _render_file_excerpt(
                    spec.old_root_path,
                    relative_path,
                    startline=start,
                    endline=max(start, end),
                ),
            }
        )
    return records


def _resolved_entity_record(
    spec: CaseSpec,
    *,
    root: Path,
    relative_path: str,
    tree: str,
    match: dict[str, Any],
    donor_report: dict[str, Any],
    old_report: dict[str, Any] | None,
) -> dict[str, Any]:
    extent = match.get("extent") if isinstance(match.get("extent"), dict) else {}
    start = int(extent.get("start_line") or 0)
    end = int(extent.get("end_line") or start)
    record: dict[str, Any] = {
        "current_entity_id": match.get("id"),
        "current_extent": extent,
        "conditional_context": list(match.get("conditional_context") or []),
        "source": _render_file_excerpt(
            root,
            relative_path,
            startline=start,
            endline=max(start, end),
        ),
    }
    if tree != "donor" or old_report is None or not old_report.get("parse_ok"):
        return record
    alignment = structural_entity_neighbor_alignment(
        target_entity_id=str(match.get("id") or ""),
        donor_entities=donor_report.get("entities") or [],
        old_entities=old_report.get("entities") or [],
    )
    record["old_counterpart_alignment"] = alignment
    record["old_anchor_sources"] = _old_alignment_sources(
        spec,
        relative_path=relative_path,
        alignment=alignment,
    )
    return record


def _collect_entity_evidence(
    spec: CaseSpec,
    *,
    request: dict[str, str],
    scope: dict[str, Any],
    code_index: CodeIndex,
    report_cache: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    descriptor = next(
        (
            item
            for item in scope.get("inspectable_entities") or []
            if isinstance(item, dict) and str(item.get("id") or "") == request["id"]
        ),
        None,
    )
    if descriptor is None:
        return [
            {
                "status": "rejected_unknown_entity_id",
                "requested_entity_id": request["id"],
                "tree": request.get("tree") or None,
            }
        ]
    requested_tree = request.get("tree") or str(descriptor.get("tree") or "")
    trees = [requested_tree] if requested_tree in {"old", "donor"} else ["old", "donor"]
    records: list[dict[str, Any]] = []
    relative_path = normalize_relpath(str(descriptor.get("path") or ""))
    for tree in trees:
        root = spec.old_root_path if tree == "old" else spec.primary_donor_root
        path = root / relative_path
        base = {
            "requested_entity_id": request["id"],
            "tree": tree,
            "path": relative_path,
            "kind": descriptor.get("kind"),
            "symbol": descriptor.get("symbol"),
            "contract_key": descriptor.get("contract_key"),
        }
        if not path.is_file():
            records.append({**base, "status": "resolved_file_absent"})
            continue
        report = case_aware_entity_report(
            spec,
            relative_path=relative_path,
            tree=tree,
            index=code_index,
            report_cache=report_cache,
        )
        if not report.get("parse_ok"):
            records.append(
                {
                    **base,
                    "status": "unavailable_entity_parse",
                    "parse_error": report.get("parse_error"),
                    "parse_exception": report.get("parse_exception"),
                    "backend": report.get("backend"),
                    "compile_aware": bool(
                        report.get("compile_aware")
                    ),
                    "compile_aware_failure": report.get(
                        "compile_aware_failure"
                    ),
                }
            )
            continue
        match_descriptor = dict(descriptor)
        if report.get("compile_aware") and _structural_entity_kind(
            descriptor
        ) in {
            "function",
            "variable",
        }:
            # A stale structural declaration is not an owner definition. The
            # active case TU is authoritative for the current implementation.
            match_descriptor["is_definition"] = True
        matches = _entity_matches(report, match_descriptor)
        old_report: dict[str, Any] | None = None
        if tree == "donor":
            old_report = case_aware_entity_report(
                spec,
                relative_path=relative_path,
                tree="old",
                index=code_index,
                report_cache=report_cache,
            )
        if not matches:
            records.append(
                {
                    **base,
                    "status": (
                        "resolved_entity_absent"
                        if report.get(
                            "owner_presence_authoritative"
                        )
                        else "entity_absence_unproven"
                    ),
                    "match_count": 0,
                    "backend": report.get("backend"),
                    "compile_aware": bool(
                        report.get("compile_aware")
                    ),
                    "owner_presence_authoritative": bool(
                        report.get(
                            "owner_presence_authoritative"
                        )
                    ),
                }
            )
            continue
        if len(matches) != 1:
            if _conditional_variant_family(matches, descriptor):
                records.append(
                    {
                        **base,
                        "status": "resolved_variants",
                        "match_count": len(matches),
                        "backend": report.get("backend"),
                        "compile_aware": bool(
                            report.get("compile_aware")
                        ),
                        "variant_semantics": (
                            "mutually separated preprocessor-context implementations "
                            "of the same structural contract"
                        ),
                        "variants": [
                            _resolved_entity_record(
                                spec,
                                root=root,
                                relative_path=relative_path,
                                tree=tree,
                                match=match,
                                donor_report=report,
                                old_report=old_report,
                            )
                            for match in matches
                        ],
                    }
                )
                continue
            records.append(
                {
                    **base,
                    "status": "unresolved_ambiguous_entity",
                    "match_count": len(matches),
                    "backend": report.get("backend"),
                    "compile_aware": bool(
                        report.get("compile_aware")
                    ),
                    "candidate_entity_ids": [
                        str(item.get("id") or "") for item in matches
                    ],
                }
            )
            continue
        match = matches[0]
        records.append(
            {
                **base,
                "status": "resolved",
                "match_count": 1,
                "backend": report.get("backend"),
                "compile_aware": bool(report.get("compile_aware")),
                "owner_presence_authoritative": bool(
                    report.get("owner_presence_authoritative")
                ),
                **_resolved_entity_record(
                    spec,
                    root=root,
                    relative_path=relative_path,
                    tree=tree,
                    match=match,
                    donor_report=report,
                    old_report=old_report,
                ),
            }
        )
    return records


def collect_agentic_evidence(
    spec: CaseSpec,
    *,
    action: dict[str, Any],
    scope: dict[str, Any],
    validation_report: dict[str, Any],
) -> dict[str, Any]:
    request = action.get("inspect") if isinstance(action.get("inspect"), dict) else {}
    evidence: dict[str, Any] = {
        "schema": "agentic_repair_evidence_v3",
        "requested_at": utc_now(),
        "rationale": action.get("rationale"),
        "files": [],
        "entities": [],
    }
    code_index = CodeIndex(spec, graph_projection="records")
    entity_report_cache: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    for raw_entity in request.get("entities") or []:
        entity_request = _normalize_entity_request(raw_entity)
        if entity_request is None:
            evidence["entities"].append(
                {
                    "status": "rejected_invalid_entity_request",
                    "request": raw_entity,
                }
            )
            continue
        evidence["entities"].extend(
            _collect_entity_evidence(
                spec,
                request=entity_request,
                scope=scope,
                code_index=code_index,
                report_cache=entity_report_cache,
            )
        )
    for raw_file in request.get("files") or []:
        file_request = _normalize_file_request(raw_file)
        if file_request is None:
            evidence["files"].append({"status": "rejected_invalid_file_request", "request": raw_file})
            continue
        tree = file_request["tree"]
        if tree not in {"old", "target", "donor", "new", "both"}:
            tree = "both"
        normalized_tree = "old" if tree == "target" else "donor" if tree == "new" else tree
        for tree_name, root in _file_roots_for_tree(spec, normalized_tree):
            intersections = _inspect_range_intersections(
                scope,
                tree=tree_name,
                relative_path=file_request["path"],
                startline=file_request["startline"],
                endline=file_request["endline"],
            )
            fully_authorized = intersections == [
                (file_request["startline"], file_request["endline"])
            ]
            # A range based on donor line numbers may extend past one edge of
            # the corresponding old entity. Return the exact graph-authorized
            # intersection instead of poisoning every later action in the
            # unit. Internal gaps remain rejected.
            edge_clipped = len(intersections) == 1 and not fully_authorized
            if not fully_authorized and not edge_clipped:
                evidence["files"].append(
                    {
                        "status": "rejected_outside_code_graph_range",
                        "tree": tree_name,
                        "path": file_request["path"],
                        "line_start": file_request["startline"],
                        "line_end": file_request["endline"],
                    }
                )
                continue
            for actual_start, actual_end in intersections:
                record = {
                    "tree": tree_name,
                    **_render_file_excerpt(
                        root,
                        file_request["path"],
                        startline=actual_start,
                        endline=actual_end,
                    ),
                }
                if edge_clipped:
                    record.update(
                        {
                            "status": "resolved_graph_authorized_intersection",
                            "requested_line_start": file_request["startline"],
                            "requested_line_end": file_request["endline"],
                            "coverage_complete": False,
                        }
                    )
                evidence["files"].append(record)

    if request.get("logs"):
        evidence["validation"] = _compact_validation(validation_report)
    return evidence


def agentic_evidence_ready_for_apply(
    evidence_log: list[dict[str, Any]],
    *,
    spec: CaseSpec | None = None,
    action: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require requested evidence or an independently grounded current-tree patch."""

    latest = next(
        (
            entry
            for entry in reversed(evidence_log)
            if entry.get("kind") == "inspect_result"
            and isinstance(entry.get("evidence"), dict)
        ),
        None,
    )
    if latest is None:
        return {"ready": True, "reason": "no_inspect_requested"}
    evidence = dict(latest.get("evidence") or {})
    unresolved: list[dict[str, Any]] = []
    for record in evidence.get("entities") or []:
        if not isinstance(record, dict):
            continue
        status = str(record.get("status") or "")
        if status and not status.startswith("resolved"):
            unresolved.append(record)
    for record in evidence.get("files") or []:
        if not isinstance(record, dict):
            continue
        status = str(record.get("status") or "")
        if status.startswith("rejected_") or record.get("exists") is False:
            unresolved.append(record)
    if unresolved and spec is not None and action is not None and scope is not None:
        diff_text = agentic_patch_text(action)
        touched_files = {
            normalize_relpath(path)
            for path in touched_files_from_diff(diff_text)
            if normalize_relpath(path)
        }
        authorized_files = {
            normalize_relpath(path)
            for path in [
                *(scope.get("editable_files") or []),
                *(scope.get("escalation_files") or []),
            ]
            if normalize_relpath(path)
        }
        if touched_files and touched_files <= authorized_files:
            grounding = build_patch_conflict_evidence(
                spec,
                diff_text=diff_text,
                touched_files=sorted(touched_files),
            )
            grounded_hunks = [
                hunk
                for hunk in grounding.get("hunks") or []
                if hunk.get("hunk")
                and hunk.get("static_relocation_possible")
            ]
            grounded_files = {
                normalize_relpath(str(hunk.get("file") or ""))
                for hunk in grounded_hunks
                if normalize_relpath(str(hunk.get("file") or ""))
            }
            all_hunks_grounded = (
                bool(grounded_hunks)
                and len(grounded_hunks)
                == int(grounding.get("hunk_count") or 0)
                and not int(grounding.get("omitted_hunk_count") or 0)
                and grounded_files == touched_files
            )
            if all_hunks_grounded:
                return {
                    "ready": True,
                    "reason": "patch_grounded_in_current_authorized_tree",
                    "unresolved_count": len(unresolved),
                    "unresolved": unresolved,
                    "grounded_hunk_count": len(grounded_hunks),
                    "grounded_files": sorted(grounded_files),
                }
    return {
        "ready": not unresolved,
        "reason": (
            "latest_requested_evidence_available"
            if not unresolved
            else "latest_requested_evidence_unavailable"
        ),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
    }


def _repair_source_pack_with_dynamic_evidence(
    source_pack: dict[str, Any],
    *,
    evidence_log: list[dict[str, Any]],
    allowed_files: set[str],
) -> dict[str, Any]:
    merged = dict(source_pack)
    sections = {
        key: [
            dict(record)
            for record in source_pack.get(key) or []
            if isinstance(record, dict)
        ]
        for key in (
            "allowed_edit_sources",
            "donor_reference_sources",
            "read_only_evidence_sources",
        )
    }
    dynamic_records: list[tuple[str, dict[str, Any]]] = []

    def add_source(
        *,
        tree: str,
        source: dict[str, Any],
        symbol: object = None,
        source_kind: str,
    ) -> None:
        path = normalize_relpath(
            str(source.get("path") or source.get("file") or "")
        )
        text = str(source.get("text") or "")
        if not path or not text:
            return
        normalized_tree = "donor" if tree in {"donor", "new"} else "old"
        section = (
            "donor_reference_sources"
            if normalized_tree == "donor"
            else "allowed_edit_sources"
            if path in allowed_files
            else "read_only_evidence_sources"
        )
        dynamic_records.append(
            (
                section,
                {
                    "tree": normalized_tree,
                    "file": path,
                    "role": "dynamic_inspection_evidence",
                    "source_kind": source_kind,
                    "symbol": symbol,
                    "line_start": source.get("line_start"),
                    "line_end": source.get("line_end"),
                    "exists": source.get("exists", True),
                    "text": text,
                    "reason": (
                        "exact evidence collected by the agentic inspect tool "
                        "before this apply attempt"
                    ),
                },
            )
        )

    for entry in reversed(evidence_log):
        if entry.get("kind") != "inspect_result":
            continue
        evidence = (
            entry.get("evidence")
            if isinstance(entry.get("evidence"), dict)
            else {}
        )
        for entity in evidence.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            tree = str(entity.get("tree") or "")
            source = entity.get("source")
            if isinstance(source, dict):
                add_source(
                    tree=tree,
                    source=source,
                    symbol=entity.get("symbol"),
                    source_kind="dynamic_entity_inspection",
                )
            for variant in entity.get("variants") or []:
                if not isinstance(variant, dict):
                    continue
                variant_source = variant.get("source")
                if isinstance(variant_source, dict):
                    add_source(
                        tree=str(variant.get("tree") or tree),
                        source=variant_source,
                        symbol=variant.get("symbol") or entity.get("symbol"),
                        source_kind="dynamic_entity_variant_inspection",
                    )
        for file_record in evidence.get("files") or []:
            if not isinstance(file_record, dict):
                continue
            add_source(
                tree=str(file_record.get("tree") or ""),
                source=file_record,
                source_kind="dynamic_file_inspection",
            )

    seen = {
        (
            str(record.get("tree") or ""),
            normalize_relpath(str(record.get("file") or "")),
            record.get("line_start"),
            record.get("line_end"),
            str(record.get("text") or ""),
        )
        for records in sections.values()
        for record in records
    }
    selected_dynamic = {key: [] for key in sections}
    added = 0
    for section, record in dynamic_records:
        identity = (
            str(record.get("tree") or ""),
            normalize_relpath(str(record.get("file") or "")),
            record.get("line_start"),
            record.get("line_end"),
            str(record.get("text") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        selected_dynamic[section].append(record)
        added += 1
    for section, records in selected_dynamic.items():
        sections[section] = [*records, *sections[section]]
    merged.update(sections)
    summary = dict(merged.get("summary") or {})
    summary["dynamic_inspection_source_count"] = added
    merged["summary"] = summary
    return merged


def _graph_proven_donor_export(
    *,
    scope: dict[str, Any],
    graph_unit: dict[str, Any],
    relative_path: str,
    symbol: str,
    donor_text: str,
) -> dict[str, Any] | None:
    """Return one export contract attached to the selected provider by CodeIndex."""

    visibility_demands = [
        dict(item)
        for item in graph_unit.get("module_visibility_demands") or []
        if isinstance(item, dict)
        and str(item.get("symbol") or "") == symbol
    ]
    if not visibility_demands:
        return None
    proof_ranges = [
        item
        for item in graph_unit.get("inspectable_ranges") or []
        if isinstance(item, dict)
        and str(item.get("tree") or "") == "donor"
        and normalize_relpath(str(item.get("path") or "")) == relative_path
        and str(item.get("source") or "")
        == "donor_external_provider_contract_reference"
    ]
    if not proof_ranges:
        return None
    scope_ranges = {
        (
            str(item.get("tree") or ""),
            normalize_relpath(str(item.get("path") or "")),
            int(item.get("line_start") or 0),
            int(item.get("line_end") or item.get("line_start") or 0),
            str(item.get("source") or ""),
        )
        for item in scope.get("inspectable_ranges") or []
        if isinstance(item, dict)
    }
    verified_ranges = [
        item
        for item in proof_ranges
        if (
            "donor",
            relative_path,
            int(item.get("line_start") or 0),
            int(item.get("line_end") or item.get("line_start") or 0),
            "donor_external_provider_contract_reference",
        )
        in scope_ranges
    ]
    exports = [
        record
        for record in parse_export_contracts(
            donor_text,
            file=relative_path,
            tree="donor",
        )
        if str(record.get("symbol") or "") == symbol
        and any(
            int(source_range.get("line_start") or 0)
            <= int(record.get("line") or 0)
            <= int(
                source_range.get("line_end")
                or source_range.get("line_start")
                or 0
            )
            for source_range in verified_ranges
        )
    ]
    if not exports:
        return None
    identities = {
        (
            str(record.get("macro") or ""),
            str(record.get("namespace_or_modules") or ""),
            str(record.get("raw") or ""),
            int(record.get("line") or 0),
        )
        for record in exports
    }
    if len(identities) != 1:
        raise ValueError(
            "entity materialization has ambiguous graph-proven donor export contracts"
        )
    return {
        **exports[0],
        "module_visibility_demands": visibility_demands,
    }


def _synchronize_graph_proven_export(
    *,
    scope: dict[str, Any],
    graph_unit: dict[str, Any],
    relative_path: str,
    symbol: str,
    old_text: str,
    donor_text: str,
    materialized_lines: list[str],
    entity_end_index: int,
    donor_entity_end_line: int,
) -> tuple[list[str], dict[str, Any] | None]:
    donor_export = _graph_proven_donor_export(
        scope=scope,
        graph_unit=graph_unit,
        relative_path=relative_path,
        symbol=symbol,
        donor_text=donor_text,
    )
    if donor_export is None:
        return materialized_lines, None

    old_exports = [
        record
        for record in parse_export_contracts(
            old_text,
            file=relative_path,
            tree="old",
        )
        if str(record.get("symbol") or "") == symbol
    ]
    donor_shape = (
        str(donor_export.get("macro") or ""),
        str(donor_export.get("namespace_or_modules") or ""),
    )
    if len(old_exports) > 1:
        raise ValueError(
            "entity materialization found ambiguous old export contracts"
        )
    if old_exports:
        old_shape = (
            str(old_exports[0].get("macro") or ""),
            str(old_exports[0].get("namespace_or_modules") or ""),
        )
        if old_shape != donor_shape:
            raise ValueError(
                "entity materialization cannot replace a different old export "
                "without an explicit patch"
            )
        return materialized_lines, {
            "status": "already_satisfied",
            "old": old_exports[0],
            "donor": donor_export,
        }

    donor_lines = donor_text.splitlines(keepends=True)
    export_start = int(donor_export.get("line") or 0)
    export_end = int(donor_export.get("end_line") or export_start)
    if (
        export_start <= donor_entity_end_line
        or export_end != export_start
        or export_end > len(donor_lines)
    ):
        raise ValueError(
            "graph-proven donor export must be one complete line after its "
            "provider entity"
        )
    between = donor_lines[donor_entity_end_line : export_start - 1]
    if any(line.strip() for line in between):
        raise ValueError(
            "graph-proven donor export is separated from its provider by "
            "another source contract"
        )
    export_line = donor_lines[export_start - 1]
    if export_line.strip() != str(donor_export.get("raw") or "").strip():
        raise ValueError(
            "graph-proven donor export does not own one complete source line"
        )
    append_lines = [*between, export_line]
    if append_lines and not append_lines[-1].endswith(("\n", "\r")):
        append_lines[-1] += "\n"
    if entity_end_index < 0 or entity_end_index > len(materialized_lines):
        raise ValueError(
            "entity materialization export insertion point is invalid"
        )
    updated = [
        *materialized_lines[:entity_end_index],
        *append_lines,
        *materialized_lines[entity_end_index:],
    ]
    materialized_exports = [
        record
        for record in parse_export_contracts(
            "".join(updated),
            file=relative_path,
            tree="old_materialized",
        )
        if str(record.get("symbol") or "") == symbol
    ]
    if {
        (
            str(record.get("macro") or ""),
            str(record.get("namespace_or_modules") or ""),
        )
        for record in materialized_exports
    } != {donor_shape}:
        raise ValueError(
            "materialized source does not satisfy the exact donor export contract"
        )
    return updated, {
        "status": "inserted",
        "old": None,
        "donor": donor_export,
        "insertion_index": entity_end_index,
        "inserted_line_count": len(append_lines),
        "module_visibility_demands": list(
            donor_export.get("module_visibility_demands") or []
        ),
    }


def unique_agentic_entity_materialization_request(
    scope: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the only graph-proven entity tool request, if one exists."""

    graph_units = [
        unit
        for unit in scope.get("graph_units") or []
        if isinstance(unit, dict) and unit.get("patch_ready")
    ]
    if len(graph_units) != 1:
        return None
    graph_unit = graph_units[0]
    entity_ids = [
        str(item)
        for item in graph_unit.get("materializable_entity_ids") or []
        if str(item)
    ]
    all_descriptors = [
        item
        for item in scope.get("inspectable_entities") or []
        if isinstance(item, dict)
        and str(item.get("id") or "") in set(entity_ids)
        and str(item.get("tree") or "") == "donor"
        and str(item.get("contract_key") or "").strip()
        and normalize_relpath(str(item.get("path") or ""))
    ]
    semantic_consumer_entity_ids = {
        str(item)
        for item in graph_unit.get(
            "semantic_consumer_entity_ids"
        )
        or []
        if str(item)
    }
    recovery_consumer_descriptors = [
        item
        for item in all_descriptors
        if graph_unit.get("required_semantic_gap_ids")
        and (
            graph_unit.get(
                "required_integration_obligation_ids"
            )
            or "integration_reference"
            in {
                str(dimension)
                for dimension in graph_unit.get(
                    "semantic_difference_dimensions"
                )
                or []
            }
        )
        and str(item.get("graph_entity_id") or "")
        in semantic_consumer_entity_ids
    ]
    if recovery_consumer_descriptors:
        descriptors = recovery_consumer_descriptors
        materialization_authority = (
            "exact_required_demand_anchor_consumer"
        )
    else:
        if len(entity_ids) != 1:
            return None
        descriptors = all_descriptors
        materialization_authority = (
            "unique_graph_root_contract"
        )
    if len(descriptors) != 1:
        return None
    entity_id = str(descriptors[0].get("id") or "")

    selected_id = str(
        graph_unit.get("work_unit_id")
        or graph_unit.get("plan_unit_id")
        or ""
    )
    if not selected_id:
        return None
    decision: dict[str, Any] = {
        "mode": "patch_root_subgraph",
        "work_unit_id": selected_id,
    }
    root_node_id = str(graph_unit.get("root_node_id") or "")
    root_key = str(graph_unit.get("root_key") or "")
    if root_node_id:
        decision["root_node_id"] = root_node_id
    if root_key:
        decision["root_key"] = root_key
    if graph_unit.get("semantic_gap_ids"):
        addressed_gap_ids = _bound_semantic_gap_ids_for_entity(
            graph_unit,
            descriptors[0],
        )
        if (
            materialization_authority
            == "exact_required_demand_anchor_consumer"
            and not addressed_gap_ids
        ):
            return None
        if addressed_gap_ids:
            decision["addressed_semantic_gap_ids"] = (
                addressed_gap_ids
            )
    return {
        "action": "materialize_entity_from_donor",
        "rationale": (
            (
                "The pending semantic transaction has one exact DemandAnchor "
                "consumer, so the controller can restore that complete donor "
                "consumer before repairing compatibility diagnostics."
            )
            if materialization_authority
            == "exact_required_demand_anchor_consumer"
            else (
                "The code graph selects one patch-ready root and one exact "
                "materializable contract, so the controller can synchronize "
                "that entity without a model choice."
            )
        ),
        "decision": {
            **decision,
            "materialization_authority": (
                materialization_authority
            ),
        },
        "entity": {"id": entity_id},
    }


def build_exact_entity_materialization_action(
    spec: CaseSpec,
    *,
    work_unit_id: str,
    descriptor: dict[str, Any],
    editable_files: list[object],
    escalation_files: list[object] | None = None,
    semantic_gap_ids: list[object] | None = None,
    root_key: str | None = None,
    rationale: str = "",
    materialization_authority: str = (
        "exact_donor_entity_with_proven_target_absence"
    ),
    module_visibility_demands: list[dict[str, Any]] | None = None,
    inspectable_ranges: list[dict[str, Any]] | None = None,
    preferred_translation_units: list[object] | None = None,
) -> dict[str, Any]:
    """Adapt one exact donor entity to the canonical graph materializer."""

    unit_id = str(work_unit_id or "").strip()
    entity_id = str(
        descriptor.get("id") or descriptor.get("entity_id") or ""
    ).strip()
    relative_path = normalize_relpath(
        str(descriptor.get("path") or descriptor.get("file") or "")
    )
    gap_ids = list(
        dict.fromkeys(
            str(item) for item in semantic_gap_ids or [] if str(item)
        )
    )
    binding_key = str(root_key or "").strip() or stable_id(
        "exact-entity-materialization-root",
        unit_id,
        entity_id,
    )
    ranges = list(inspectable_ranges or [])
    graph_unit = {
        "work_unit_id": unit_id,
        "plan_unit_id": unit_id,
        "patch_ready": True,
        "root_node_id": "",
        "root_key": binding_key,
        "materializable_entity_ids": [entity_id],
        "editable_files": ordered_unique_relpaths(editable_files),
        "escalation_files": ordered_unique_relpaths(
            escalation_files or []
        ),
        "entity_materialization_files": [relative_path],
        "semantic_gap_ids": gap_ids,
        "semantic_entity_gap_bindings": [],
        "module_visibility_demands": list(
            module_visibility_demands or []
        ),
        "inspectable_ranges": ranges,
        "preferred_translation_units": ordered_unique_relpaths(
            preferred_translation_units or []
        ),
    }
    scope = {
        "graph_units": [graph_unit],
        "inspectable_entities": [
            {
                **descriptor,
                "id": entity_id,
                "tree": "donor",
                "path": relative_path,
            }
        ],
        "inspectable_ranges": ranges,
    }
    action = unique_agentic_entity_materialization_request(scope)
    if action is None:
        raise ValueError(
            "exact descriptor did not produce one materialization request"
        )
    action["rationale"] = str(rationale)
    action["decision"].update(
        {
            "addressed_semantic_gap_ids": gap_ids,
            "materialization_authority": str(materialization_authority),
        }
    )
    return build_agentic_entity_materialization_action(
        spec,
        action=action,
        scope=scope,
    )


def build_agentic_entity_materialization_action(
    spec: CaseSpec,
    *,
    action: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, Any]:
    """Synchronize one graph-selected contract from donor into the current tree."""

    if action.get("action") != "materialize_entity_from_donor":
        raise ValueError(
            "entity materialization requires materialize_entity_from_donor action"
        )
    decision = dict(action.get("decision") or {})
    selected_id = str(decision.get("work_unit_id") or "")
    graph_units = [
        unit
        for unit in scope.get("graph_units") or []
        if isinstance(unit, dict)
        and selected_id
        in {
            str(unit.get("plan_unit_id") or ""),
            str(unit.get("work_unit_id") or ""),
        }
    ]
    if len(graph_units) != 1:
        raise ValueError(
            "entity materialization work_unit_id must identify exactly one graph unit"
        )
    graph_unit = graph_units[0]
    if not graph_unit.get("patch_ready"):
        raise ValueError("selected graph unit is not patch ready")

    expected_root_id = str(graph_unit.get("root_node_id") or "")
    expected_root_key = str(graph_unit.get("root_key") or "")
    selected_root_id = str(decision.get("root_node_id") or "")
    selected_root_key = str(decision.get("root_key") or "")
    if expected_root_id and selected_root_id != expected_root_id:
        raise ValueError(
            "entity materialization root_node_id does not match the selected graph unit"
        )
    if not expected_root_id and selected_root_id:
        raise ValueError(
            "mapped root-set entity materialization must not provide root_node_id"
        )
    if expected_root_key and selected_root_key != expected_root_key:
        raise ValueError(
            "entity materialization root_key does not match the selected graph unit"
        )

    requested_entity_id = str((action.get("entity") or {}).get("id") or "")
    if requested_entity_id not in {
        str(item) for item in graph_unit.get("materializable_entity_ids") or []
    }:
        raise ValueError(
            "entity materialization id is not a root definition attached to "
            "the selected graph unit"
        )
    descriptors = [
        item
        for item in scope.get("inspectable_entities") or []
        if isinstance(item, dict)
        and str(item.get("id") or "") == requested_entity_id
    ]
    if len(descriptors) != 1:
        raise ValueError(
            "entity materialization id must resolve to exactly one scope entity"
        )
    descriptor = descriptors[0]
    available_semantic_gap_ids = {
        str(item)
        for item in graph_unit.get("semantic_gap_ids") or []
        if str(item)
    }
    bound_semantic_gap_ids = set(
        _bound_semantic_gap_ids_for_entity(
            graph_unit,
            descriptor,
        )
    )
    addressed_semantic_gap_ids = {
        str(item)
        for item in decision.get("addressed_semantic_gap_ids") or []
        if str(item)
    }
    unknown_semantic_gap_ids = (
        addressed_semantic_gap_ids - available_semantic_gap_ids
    )
    if unknown_semantic_gap_ids:
        raise ValueError(
            "entity materialization references semantic gaps outside "
            "the selected graph unit: "
            + ", ".join(sorted(unknown_semantic_gap_ids))
        )
    if (
        graph_unit.get("semantic_entity_gap_bindings")
        and addressed_semantic_gap_ids - bound_semantic_gap_ids
    ):
        raise ValueError(
            "entity materialization may only address semantic gaps "
            "exactly bound to the selected provider or DemandAnchor "
            "consumer entity"
        )
    if not addressed_semantic_gap_ids and bound_semantic_gap_ids:
        decision["addressed_semantic_gap_ids"] = sorted(
            bound_semantic_gap_ids
        )
    if str(descriptor.get("tree") or "") not in {"old", "donor"}:
        raise ValueError(
            "entity materialization requires an old- or donor-tree entity id"
        )
    contract = str(descriptor.get("contract_key") or "").strip()
    relative_path = normalize_relpath(str(descriptor.get("path") or ""))
    entity_kind = _structural_entity_kind(descriptor)
    if not contract or not relative_path or not entity_kind:
        raise ValueError(
            "entity materialization requires a typed structural contract and owner file"
        )
    permitted = {
        normalize_relpath(str(path))
        for path in [
            *(graph_unit.get("editable_files") or []),
            *(graph_unit.get("escalation_files") or []),
            *(
                graph_unit.get(
                    "entity_materialization_files"
                )
                or []
            ),
        ]
        if normalize_relpath(str(path))
    }
    if relative_path not in permitted:
        raise ValueError(
            "entity materialization owner file is outside the selected graph write set"
        )
    preferred_translation_units = ordered_unique_relpaths(
        [
            *(graph_unit.get("preferred_translation_units") or []),
            *(descriptor.get("preferred_translation_units") or []),
        ]
    )

    old_report = parse_structural_entity_file(
        root=spec.old_root_path,
        relative_path=relative_path,
        tree="old",
    )
    donor_report = parse_structural_entity_file(
        root=spec.primary_donor_root,
        relative_path=relative_path,
        tree="donor",
    )
    if not old_report.get("parse_ok") or not donor_report.get("parse_ok"):
        raise ValueError(
            "entity materialization requires parseable old and donor owner files"
    )
    old_matches = _entity_matches(old_report, descriptor)
    donor_matches = _entity_matches(donor_report, descriptor)
    family_descriptor = {
        key: value for key, value in descriptor.items()
        if key != "is_definition"
    }
    family_matches = _entity_matches(donor_report, family_descriptor)
    conditional_family = _conditional_variant_family(
        family_matches, descriptor
    )
    if conditional_family:
        donor_matches = family_matches
        old_matches = _entity_matches(old_report, family_descriptor)
    conditional_plan = None
    if conditional_family:
        old_path = spec.resolve_old(relative_path)
        donor_path = spec.resolve_donor(relative_path)
        if not old_path.is_file() or not donor_path.is_file():
            raise ValueError("entity materialization owner file is missing")
        old_lines = old_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines(keepends=True)
        donor_lines = donor_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines(keepends=True)
        conditional_plan = conditional_variant_insertions(
            donor_lines=donor_lines,
            old_lines=old_lines,
            target_entities=donor_matches,
            donor_entities=[
                entity
                for entity in donor_report.get("entities") or []
                if isinstance(entity, dict)
            ],
            old_entities=[
                entity
                for entity in old_report.get("entities") or []
                if isinstance(entity, dict)
            ],
        )
        if conditional_plan is None:
            raise ValueError(
                "conditional entity variants do not map to exact old branch anchors"
            )
    if conditional_plan is not None:
        missing_ids = {
            str(item.get("entity_id") or "")
            for item in conditional_plan.get("insertions") or []
            if isinstance(item, dict)
        }
        missing_variants = [
            item for item in donor_matches
            if str(item.get("id") or "") in missing_ids
        ]
        if len(missing_variants) != len(missing_ids):
            raise ValueError(
                "conditional entity insertion plan does not identify exact donor variants"
            )
        dependency_closure = prove_entity_materialization_dependency_closure(
            spec,
            relative_path=relative_path,
            contract_key=contract,
            old_structural_report=old_report,
            donor_structural_report=donor_report,
            donor_root_entities=missing_variants,
            preferred_translation_units=preferred_translation_units,
        )
        if dependency_closure.get("status") != "proven":
            raise EntityDependencyClosureError(dependency_closure)
        visibility_demands = [
            item
            for item in graph_unit.get("module_visibility_demands") or []
            if isinstance(item, dict)
            and str(item.get("symbol") or "")
            == str(descriptor.get("symbol") or "")
        ]
        if visibility_demands:
            raise ValueError(
                "conditional entity materialization cannot place a shared "
                "linkage contract at one branch-local definition"
            )
        materialized_lines = apply_conditional_variant_insertions(
            donor_lines=donor_lines,
            old_lines=old_lines,
            insertion_plan=conditional_plan,
        )
        patch_text = "".join(
            difflib.unified_diff(
                old_lines,
                materialized_lines,
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                n=3,
            )
        )
        if not patch_text:
            raise ValueError(
                "conditional entity materialization did not produce a patch"
            )
        strategy = "insert_conditional_donor_entity_family_at_branch_anchors"
        return {
            "schema": "graph_entity_materialization_v2",
            "status": "resolved",
            "contract_key": contract,
            "entity_kind": entity_kind,
            "path": relative_path,
            "strategy": strategy,
            "old_entity_id": None,
            "donor_entity_ids": [
                str(entity.get("id") or "") for entity in donor_matches
            ],
            "materialized_donor_entity_ids": sorted(missing_ids),
            "conditional_insertion_plan": conditional_plan,
            "linkage_contract": None,
            "dependency_closure": dependency_closure,
            "action": {
                "action": "apply_patch",
                "rationale": str(action.get("rationale") or ""),
                "decision": decision,
                "patch_plan": {
                    "strategy": strategy,
                    "affected_files": [relative_path],
                    "expected_outcome": (
                        "validate every mutually exclusive donor contract "
                        "variant in the corresponding current-tree branch"
                    ),
                },
                "patch": {
                    "format": "unified_diff",
                    "content": patch_text,
                },
            },
            "evidence": {
                "schema": "agentic_repair_evidence_v4",
                "entities": [
                    {
                        "requested_entity_id": requested_entity_id,
                        "tree": "old",
                        "path": relative_path,
                        "kind": entity_kind,
                        "contract_key": contract,
                        "status": (
                            "resolved_entity_partially_present"
                            if old_matches else "resolved_entity_absent"
                        ),
                        "current_entity_id": (
                            old_matches[0].get("id") if len(old_matches) == 1 else None
                        ),
                        "current_extent": (
                            old_matches[0].get("extent")
                            if len(old_matches) == 1 else None
                        ),
                    },
                    *[
                        {
                            "requested_entity_id": requested_entity_id,
                            "tree": "donor",
                            "path": relative_path,
                            "kind": entity_kind,
                            "contract_key": contract,
                            "status": "resolved_variant",
                            "current_entity_id": entity.get("id"),
                            "current_extent": entity.get("extent"),
                            "conditional_context": list(
                                entity.get("conditional_context") or []
                            ),
                        }
                        for entity in donor_matches
                    ],
                ],
                "linkage_contracts": [],
                "dependency_closure": dependency_closure,
                "files": [],
            },
        }
    donor_entity = exact_entity_provenance_match(
        descriptor,
        donor_matches,
        tree=str(donor_report.get("tree") or "donor"),
    ) or _unique_structural_definition(donor_matches)
    if donor_entity is None:
        representation_authority = (
            _macro_generated_representation_authority(
                donor_report,
                descriptor,
            )
        )
        if representation_authority is not None:
            raise EntityRepresentationAdaptationRequired(
                representation_authority
            )
        raise ValueError(
            "entity materialization contract is absent or ambiguous in donor"
        )
    if (
        _structural_entity_kind(donor_entity) != entity_kind
        or str(donor_entity.get("contract_key") or "") != contract
    ):
        raise ValueError(
            "entity materialization donor structural contract does not match"
        )

    old_path = spec.resolve_old(relative_path)
    donor_path = spec.resolve_donor(relative_path)
    if not old_path.is_file() or not donor_path.is_file():
        raise ValueError("entity materialization owner file is missing")
    old_lines = old_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines(keepends=True)
    donor_lines = donor_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines(keepends=True)
    old_text = "".join(old_lines)
    donor_text = "".join(donor_lines)

    old_entity = _unique_structural_definition(old_matches)
    if old_matches and old_entity is None:
        raise ValueError(
            "entity materialization contract is ambiguous in the current old tree"
        )
    donor_extent = dict(donor_entity.get("extent") or {})
    donor_start = int(donor_extent.get("start_line") or 0)
    donor_end = int(donor_extent.get("end_line") or donor_start)
    if donor_start <= 0 or donor_end < donor_start:
        raise ValueError(
            "entity materialization requires an exact donor source extent"
        )
    dependency_closure: dict[str, Any] | None = None
    dependency_expansion: dict[str, Any] | None = None
    post_materialization_provider_obligations: list[
        dict[str, Any]
    ] = []
    materialization_entities = [donor_entity]
    materialization_extent = dict(donor_extent)
    materialization_extents = [dict(donor_extent)]
    bundle_alignment_set: dict[str, Any] | None = None
    if old_entity is None:
        dependency_expansion = (
            expand_entity_materialization_dependency_closure(
                spec,
                relative_path=relative_path,
                contract_key=contract,
                old_structural_report=old_report,
                donor_structural_report=donor_report,
                donor_root_entities=[donor_entity],
                preferred_translation_units=(
                    preferred_translation_units
                ),
            )
        )
        dependency_closure = dependency_expansion.get(
            "dependency_closure"
        )
        if dependency_expansion.get("status") not in {
            "proven",
            (
                "admissible_with_post_materialization_"
                "provider_obligations"
            ),
        }:
            proof = dict(
                dependency_closure
                if isinstance(dependency_closure, dict)
                else dependency_expansion
            )
            proof["dependency_expansion"] = dependency_expansion
            raise EntityDependencyClosureError(proof)
        post_materialization_provider_obligations = [
            dict(item)
            for item in dependency_expansion.get(
                "post_materialization_provider_obligations"
            )
            or []
            if isinstance(item, dict)
        ]
        materialization_ids = {
            str(item)
            for item in dependency_expansion.get(
                "donor_root_entity_ids"
            )
            or []
            if str(item)
        }
        materialization_entities = [
            entity
            for entity in donor_report.get("entities") or []
            if isinstance(entity, dict)
            and str(entity.get("id") or entity.get("entity_id") or "")
            in materialization_ids
        ]
        if len(materialization_entities) != len(materialization_ids):
            raise ValueError(
                "dependency closure does not map to exact donor structural entities"
            )
        bundle_alignment_set = structural_entity_bundle_alignments(
            target_entity_ids=materialization_ids,
            donor_entities=[
                *(donor_report.get("entities") or []),
                *(donor_report.get("source_anchors") or []),
            ],
            old_entities=[
                *(old_report.get("entities") or []),
                *(old_report.get("source_anchors") or []),
            ],
        )
        if bundle_alignment_set.get("status") != "aligned":
            proof = dict(dependency_closure or {})
            proof.update(
                {
                    "status": "unproven",
                    "reason": (
                        "same_owner_dependency_runs_have_no_atomic_source_alignment"
                    ),
                    "bundle_alignment_set": bundle_alignment_set,
                    "dependency_expansion": dependency_expansion,
                }
            )
            raise EntityDependencyClosureError(proof)
        bundle_alignments = list(
            bundle_alignment_set.get("alignments") or []
        )
        root_entity_id = str(
            donor_entity.get("id")
            or donor_entity.get("entity_id")
            or ""
        )
        root_alignments = [
            alignment
            for alignment in bundle_alignments
            if root_entity_id
            in {
                str(item)
                for item in alignment.get("target_entity_ids") or []
            }
        ]
        if len(root_alignments) != 1:
            raise ValueError(
                "dependency closure does not identify one root source run"
            )
        materialization_extent = dict(
            (root_alignments[0].get("bundle") or {}).get(
                "extent"
            )
            or {}
        )
        materialization_extents = [
            dict((alignment.get("bundle") or {}).get("extent") or {})
            for alignment in bundle_alignments
        ]
    else:
        bundle_alignments = []
    if (
        donor_start <= 0
        or donor_end < donor_start
        or donor_end > len(donor_lines)
    ):
        raise ValueError(
            "entity materialization donor extent exceeds its owner file"
        )
    donor_entity_lines = donor_lines[donor_start - 1 : donor_end]
    if donor_entity_lines and not donor_entity_lines[-1].endswith(("\n", "\r")):
        donor_entity_lines[-1] += "\n"

    alignment: dict[str, Any] | None = None
    if old_entity is not None:
        if (
            _structural_entity_kind(old_entity) != entity_kind
            or str(old_entity.get("contract_key") or "") != contract
        ):
            raise ValueError(
                "entity materialization old structural contract does not match"
            )
        old_extent = dict(old_entity.get("extent") or {})
        old_start = int(old_extent.get("start_line") or 0)
        old_end = int(old_extent.get("end_line") or old_start)
        if old_start <= 0 or old_end < old_start or old_end > len(old_lines):
            raise ValueError(
                "entity materialization requires an exact current source extent"
            )
        current_entity_lines = old_lines[old_start - 1 : old_end]
        entity_changed = current_entity_lines != donor_entity_lines
        if entity_changed:
            raise ValueError(
                "automatic donor entity materialization cannot overwrite an "
                "existing old-tree implementation; use an explicit "
                "compatibility patch"
            )
        materialized_lines = list(old_lines)
        entity_end_index = old_end
        strategy = "same_contract_entity_already_equivalent"
        old_status = "already_equivalent"
        placement = {
            "old_extent": old_extent,
            "insertion_index": None,
        }
    else:
        insertion_operations: list[dict[str, Any]] = []
        entities_by_id = {
            str(
                entity.get("id")
                or entity.get("entity_id")
                or ""
            ): entity
            for entity in materialization_entities
        }
        root_entity_id = str(
            donor_entity.get("id")
            or donor_entity.get("entity_id")
            or ""
        )
        for run_alignment in bundle_alignments:
            run_extent = dict(
                (run_alignment.get("bundle") or {}).get("extent")
                or {}
            )
            run_start = int(run_extent.get("start_line") or 0)
            run_end = int(run_extent.get("end_line") or run_start)
            run_ids = [
                str(item)
                for item in run_alignment.get("target_entity_ids") or []
                if str(item)
            ]
            run_entities = [
                entities_by_id[entity_id]
                for entity_id in run_ids
                if entity_id in entities_by_id
            ]
            if (
                len(run_entities) != len(run_ids)
                or run_start <= 0
                or run_end < run_start
                or run_end > len(donor_lines)
            ):
                raise ValueError(
                    "entity materialization source run is incomplete"
                )
            if _source_run_has_unaccounted_structural_error(
                donor_report,
                start_line=run_start,
                end_line=run_end,
                entities=run_entities,
                dependency_closure=dependency_closure,
            ):
                raise ValueError(
                    "entity materialization donor parser error overlaps "
                    "a dependency source run"
                )
            first_entity = min(
                run_entities,
                key=lambda entity: int(
                    (entity.get("extent") or {}).get("start_line")
                    or 0
                ),
            )
            last_entity = max(
                run_entities,
                key=lambda entity: int(
                    (entity.get("extent") or {}).get("end_line")
                    or 0
                ),
            )
            first_extent = first_entity.get("extent") or {}
            last_extent = last_entity.get("extent") or {}
            first_source_line = donor_lines[
                run_start - 1
            ].rstrip("\r\n")
            last_source_line = donor_lines[
                run_end - 1
            ].rstrip("\r\n")
            prefix = first_source_line[
                : max(
                    0,
                    int(first_extent.get("start_column") or 0)
                    - 1,
                )
            ]
            suffix = (
                ""
                if last_entity.get("ends_at_line_boundary")
                else last_source_line[
                    max(
                        0,
                        int(last_extent.get("end_column") or 0)
                        - 1,
                    ) :
                ]
            )
            allowed_suffix = {""}
            if _structural_entity_kind(last_entity) == "enum_constant":
                allowed_suffix.add(",")
            if _structural_entity_kind(last_entity) == "type":
                allowed_suffix.add(";")
            if prefix.strip() or suffix.strip() not in allowed_suffix:
                raise ValueError(
                    "entity materialization dependency source run does "
                    "not own complete source lines"
                )
            insertion_plan = structural_entity_insertion_plan(
                alignment=run_alignment,
                donor_lines=donor_lines,
                old_lines=old_lines,
            )
            if insertion_plan is None:
                predecessor_old = (run_alignment.get("predecessor") or {}).get("old") or {}
                successor_old = (run_alignment.get("successor") or {}).get("old") or {}
                start = int((predecessor_old.get("extent") or {}).get("end_line") or 0) + 1
                end = int((successor_old.get("extent") or {}).get("start_line") or 0) - 1
                context: dict[str, Any] = {
                    "path": relative_path, "contract_key": contract,
                    "old_neighbor_contracts": [
                        {key: entity.get(key) for key in ("id", "contract_key", "kind", "extent")}
                        for entity in run_alignment.get("old_between") or []
                        if isinstance(entity, dict)
                    ],
                    "files": [],
                }
                if predecessor_old and successor_old and 0 < start <= end <= len(old_lines):
                    context["old_interval"] = {"line_start": start, "line_end": end}
                    context["files"] = [{
                        "tree": "old", "path": relative_path,
                        "line_start": start, "line_end": end,
                        "text": "".join(old_lines[start - 1:end]),
                        "source_kind": "bounded_materialization_old_interval",
                        "edit_permission": False,
                    }]
                raise EntitySourceBoundaryError(context)
            insertion_index = int(
                insertion_plan["insertion_index"]
            )
            if structural_error_crosses_insertion_boundary(
                old_report,
                insertion_index=insertion_index,
            ):
                raise ValueError(
                    "entity materialization old parser error crosses a "
                    "dependency insertion boundary"
                )
            leading_trivia_start = int(
                insertion_plan.get("donor_leading_trivia_start_line")
                or run_start
            )
            if leading_trivia_start < 1 or leading_trivia_start > run_start:
                raise ValueError(
                    "entity materialization leading trivia exceeds the dependency source run"
                )
            leading_trivia_line_count = run_start - leading_trivia_start
            run_lines = donor_lines[leading_trivia_start - 1 : run_end]
            if run_lines and not run_lines[-1].endswith(("\n", "\r")):
                run_lines[-1] += "\n"
            insertion_operations.append(
                {
                    "target_entity_ids": run_ids,
                    "extent": run_extent,
                    "alignment": run_alignment,
                    "insertion_plan": insertion_plan,
                    "insertion_index": insertion_index,
                    "leading_trivia_line_count": leading_trivia_line_count,
                    "lines": run_lines,
                }
            )

        insertion_indexes = [
            int(operation["insertion_index"])
            for operation in insertion_operations
        ]
        if len(insertion_indexes) != len(set(insertion_indexes)):
            raise ValueError(
                "disconnected dependency runs map to the same insertion boundary"
            )
        materialized_lines = list(old_lines)
        for operation in sorted(
            insertion_operations,
            key=lambda item: int(item["insertion_index"]),
            reverse=True,
        ):
            insertion_index = int(operation["insertion_index"])
            materialized_lines[
                insertion_index:insertion_index
            ] = list(operation["lines"])
        root_operations = [
            operation
            for operation in insertion_operations
            if root_entity_id in operation["target_entity_ids"]
        ]
        if len(root_operations) != 1:
            raise ValueError(
                "materialized dependency runs do not contain one root entity"
            )
        root_operation = root_operations[0]
        root_insertion_index = int(
            root_operation["insertion_index"]
        )
        preceding_inserted_line_count = sum(
            len(operation["lines"])
            for operation in insertion_operations
            if int(operation["insertion_index"])
            < root_insertion_index
        )
        root_run_start = int(
            (root_operation.get("extent") or {}).get(
                "start_line"
            )
            or 0
        )
        entity_changed = True
        entity_end_index = (
            root_insertion_index
            + preceding_inserted_line_count
            + int(donor_extent.get("end_line") or 0)
            - root_run_start
            + 1
            + int(root_operation.get("leading_trivia_line_count") or 0)
        )
        strategy = (
            "insert_exact_donor_dependency_bundles_at_structural_gaps"
            if len(insertion_operations) > 1
            else "insert_exact_donor_dependency_bundle_at_structural_gap"
            if len(materialization_entities) > 1
            else "insert_exact_donor_entity_at_structural_gap"
        )
        old_extent = None
        old_status = "resolved_entity_absent"
        alignment = root_operation["alignment"]
        insertion_plan = root_operation["insertion_plan"]
        placement = {
            "old_extent": None,
            "insertion_index": root_insertion_index,
            "insertion_line": min(
                len(old_lines),
                root_insertion_index + 1,
            ),
            "alignment": alignment,
            "insertion_plan": insertion_plan,
            "bundle_alignment_set": bundle_alignment_set,
            "insertion_operations": [
                {
                    key: operation[key]
                    for key in (
                        "target_entity_ids",
                        "extent",
                        "alignment",
                        "insertion_plan",
                        "insertion_index",
                        "leading_trivia_line_count",
                    )
                }
                for operation in sorted(
                    insertion_operations,
                    key=lambda item: int(item["insertion_index"]),
                )
            ],
        }

    if entity_changed and dependency_closure is None:
        dependency_closure = prove_entity_materialization_dependency_closure(
            spec,
            relative_path=relative_path,
            contract_key=contract,
            old_structural_report=old_report,
            donor_structural_report=donor_report,
            donor_root_entities=[donor_entity],
        )
        if dependency_closure.get("status") not in {
            "proven",
            (
                "admissible_with_post_materialization_"
                "provider_obligations"
            ),
        }:
            raise EntityDependencyClosureError(dependency_closure)
        post_materialization_provider_obligations = [
            dict(item)
            for item in dependency_closure.get(
                "post_materialization_provider_obligations"
            )
            or []
            if isinstance(item, dict)
        ]

    materialized_lines, linkage_contract = _synchronize_graph_proven_export(
        scope=scope,
        graph_unit=graph_unit,
        relative_path=relative_path,
        symbol=str(donor_entity.get("symbol") or descriptor.get("symbol") or ""),
        old_text=old_text,
        donor_text=donor_text,
        materialized_lines=materialized_lines,
        entity_end_index=entity_end_index,
        donor_entity_end_line=int(
            donor_extent.get("end_line") or 0
        ),
    )
    linkage_changed = bool(
        linkage_contract and linkage_contract.get("status") == "inserted"
    )
    if not entity_changed and not linkage_changed:
        raise ValueError("entity materialization would not change the current tree")
    if linkage_changed:
        strategy = (
            "synchronize_graph_proven_linkage_contract"
            if not entity_changed
            else f"{strategy}_with_graph_linkage_contract"
        )

    patch_text = "".join(
        difflib.unified_diff(
            old_lines,
            materialized_lines,
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
            n=3,
        )
    )
    if not patch_text:
        raise ValueError("entity materialization did not produce a patch")

    return {
        "schema": "graph_entity_materialization_v1",
        "status": "resolved",
        "contract_key": contract,
        "entity_kind": entity_kind,
        "path": relative_path,
        "strategy": strategy,
        "old_entity_id": old_entity.get("id") if old_entity else None,
        "donor_entity_id": donor_entity.get("id"),
        "materialized_entity_ids": [
            str(entity.get("id") or entity.get("entity_id") or "")
            for entity in sorted(
                materialization_entities,
                key=lambda item: int(
                    (item.get("extent") or {}).get("start_line") or 0
                ),
            )
        ],
        "materialized_contract_keys": [
            str(entity.get("contract_key") or entity.get("key") or "")
            for entity in sorted(
                materialization_entities,
                key=lambda item: int(
                    (item.get("extent") or {}).get("start_line") or 0
                ),
            )
        ],
        "linkage_contract": linkage_contract,
        "dependency_closure": dependency_closure,
        "dependency_expansion": dependency_expansion,
        "post_materialization_provider_obligations": (
            post_materialization_provider_obligations
        ),
        **placement,
        "donor_extent": donor_extent,
        "materialization_extent": materialization_extent,
        "materialization_extents": materialization_extents,
        "action": {
            "action": "apply_patch",
            "rationale": str(action.get("rationale") or ""),
            "decision": decision,
            "patch_plan": {
                "strategy": strategy,
                "affected_files": [relative_path],
                "expected_outcome": (
                    "validate the complete donor entity contract in the current "
                    "old-tree dependency context"
                ),
                "post_materialization_provider_obligations": (
                    post_materialization_provider_obligations
                ),
            },
            "patch": {
                "format": "unified_diff",
                "content": patch_text,
            },
        },
        "evidence": {
            "schema": "agentic_repair_evidence_v4",
            "entities": [
                {
                    "requested_entity_id": requested_entity_id,
                    "tree": "old",
                    "path": relative_path,
                    "kind": entity_kind,
                    "contract_key": contract,
                    "status": old_status,
                    "current_entity_id": (
                        old_entity.get("id") if old_entity else None
                    ),
                    "current_extent": old_extent,
                },
                {
                    "requested_entity_id": requested_entity_id,
                    "tree": "donor",
                    "path": relative_path,
                    "kind": entity_kind,
                    "contract_key": contract,
                    "status": "resolved",
                    "current_entity_id": donor_entity.get("id"),
                    "current_extent": donor_extent,
                },
            ],
            "linkage_contracts": (
                [linkage_contract] if linkage_contract is not None else []
            ),
            "dependency_closure": dependency_closure,
            "dependency_expansion": dependency_expansion,
            "post_materialization_provider_obligations": (
                post_materialization_provider_obligations
            ),
            "files": [],
        },
    }


def build_agentic_apply_plan(
    spec: CaseSpec,
    *,
    action: dict[str, Any],
    repair_plan: dict[str, Any],
    scope: dict[str, Any],
    repair_round: int,
    step: int,
    evidence_log: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    decision = dict(action.get("decision") or {})
    selected_id = str(decision.get("work_unit_id") or "")
    selected_root_id = str(decision.get("root_node_id") or "")
    selected_root_key = str(decision.get("root_key") or "")
    if decision.get("mode") != "patch_root_subgraph":
        raise ValueError("agentic patch decision.mode must be patch_root_subgraph")
    if not selected_id or not (selected_root_id or selected_root_key):
        raise ValueError("agentic patch requires exact work_unit_id and graph root binding")
    matching_units = [
        unit
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and selected_id in {str(unit.get("id") or ""), str(unit.get("work_unit_id") or "")}
    ]
    if len(matching_units) != 1:
        raise ValueError("agentic patch work_unit_id must identify exactly one repair-plan unit")
    source_unit = matching_units[0]
    semantic_phase = (
        unit_execution_phase(source_unit)
        == EXECUTION_PHASE_SEMANTIC_CLOSURE
    )
    available_semantic_gap_ids = {
        str(item)
        for item in source_unit.get("semantic_gap_ids") or []
        if str(item)
    }
    addressed_semantic_gap_ids = {
        str(item)
        for item in decision.get("addressed_semantic_gap_ids") or []
        if str(item)
    }
    unknown_semantic_gap_ids = (
        addressed_semantic_gap_ids - available_semantic_gap_ids
    )
    if unknown_semantic_gap_ids:
        raise ValueError(
            "agentic patch claims semantic gaps outside the selected work unit: "
            + ", ".join(sorted(unknown_semantic_gap_ids))
        )
    if available_semantic_gap_ids and not addressed_semantic_gap_ids:
        raise ValueError(
            "semantic-bearing patch must identify "
            "addressed_semantic_gap_ids"
        )
    semantic_contracts = [
        dict(item)
        for item in source_unit.get("semantic_contracts") or []
        if isinstance(item, dict)
    ]
    contracts_by_gap_id = {
        str(item.get("gap_id") or ""): item
        for item in semantic_contracts
        if str(item.get("gap_id") or "")
    }
    partial_semantic_patch = bool(
        semantic_phase
        and addressed_semantic_gap_ids
        and addressed_semantic_gap_ids
        != available_semantic_gap_ids
    )
    if partial_semantic_patch:
        missing_contract_bindings = (
            addressed_semantic_gap_ids - set(contracts_by_gap_id)
        )
        if missing_contract_bindings:
            raise ValueError(
                "partial semantic patch lacks exact gap-to-contract "
                "bindings: "
                + ", ".join(sorted(missing_contract_bindings))
            )
    selected_semantic_contracts = [
        item
        for item in semantic_contracts
        if (
            not partial_semantic_patch
            or str(item.get("gap_id") or "")
            in addressed_semantic_gap_ids
        )
    ]
    selected_evidence_ids = {
        str(item.get("contract_evidence_id") or "")
        for item in selected_semantic_contracts
        if str(item.get("contract_evidence_id") or "")
    }
    graph_unit = next(
        (
            unit
            for unit in scope.get("graph_units") or []
            if isinstance(unit, dict)
            and selected_id in {str(unit.get("plan_unit_id") or ""), str(unit.get("work_unit_id") or "")}
        ),
        None,
    )
    if graph_unit is None or not graph_unit.get("patch_ready"):
        raise ValueError("selected work unit is not graph-complete with a permitted patch target")
    expected_root_id = str(graph_unit.get("root_node_id") or "")
    expected_root_key = str(graph_unit.get("root_key") or "")
    if expected_root_id:
        if selected_root_id != expected_root_id:
            raise ValueError("agentic patch root_node_id does not match the selected CodeIndex work unit")
    elif selected_root_id:
        raise ValueError("mapped root-set patch must bind root_key with root_node_id=null")
    if expected_root_key and selected_root_key and selected_root_key != expected_root_key:
        raise ValueError("agentic patch root_key does not match the selected CodeIndex work unit")
    if not expected_root_id and selected_root_key != expected_root_key:
        raise ValueError("agentic root-set patch requires the exact root_key")
    if not expected_root_id and not expected_root_key:
        raise ValueError("selected CodeIndex work unit has no stable graph root binding")

    materialization_authority = str(
        decision.get("materialization_authority") or ""
    )
    controller_entity_materialization = (
        materialization_authority
        in {
            "exact_required_demand_anchor_consumer",
            "unique_graph_root_contract",
        }
        or action.get("_controller_provenance")
        == "deterministic_graph_entity_materialization"
    )
    raw_provider_obligations = [
        *list(
            source_unit.get(
                "post_materialization_provider_obligations"
            )
            or []
        ),
        *list(
            (action.get("patch_plan") or {}).get(
                "post_materialization_provider_obligations"
            )
            or []
        ),
    ]
    provider_obligations_authorized = bool(
        controller_entity_materialization
        or source_unit.get("semantic_recovery_context")
    )
    provider_obligations_by_contract = {
        (
            str(item.get("symbol") or ""),
            str(item.get("contract_key") or ""),
            normalize_relpath(str(item.get("provider_file") or "")),
        ): {
            key: item.get(key)
            for key in (
                "source",
                "symbol",
                "contract_key",
                "provider_file",
                "status",
                "reason",
            )
            if item.get(key) not in (None, "")
        }
        for item in raw_provider_obligations
        if provider_obligations_authorized
        and isinstance(item, dict)
        and item.get("source")
        in {"semantic_reference", "structural_reference"}
        and str(item.get("symbol") or "")
        and str(item.get("contract_key") or "")
        and normalize_relpath(str(item.get("provider_file") or ""))
    }
    post_materialization_provider_obligations = [
        provider_obligations_by_contract[key]
        for key in sorted(provider_obligations_by_contract)
    ]
    selected_repair_owners = [
        dict(owner)
        for owner in source_unit.get("repair_owners") or []
        if isinstance(owner, dict)
        and (
            not partial_semantic_patch
            or bool(
                {
                    str(item)
                    for item in owner.get("semantic_gap_ids") or []
                    if str(item)
                }
                & addressed_semantic_gap_ids
            )
        )
    ]
    if (
        partial_semantic_patch
        and source_unit.get("repair_owners")
        and not selected_repair_owners
    ):
        raise ValueError(
            "partial semantic patch has no exact repair owner bound "
            "to its addressed gap ids"
        )
    selected_owner_files = {
        normalize_relpath(str(path))
        for owner in selected_repair_owners
        for path in [
            owner.get("owner_file"),
            *list(owner.get("allowed_patch_files") or []),
            *list(
                owner.get("entity_materialization_files") or []
            ),
        ]
        if normalize_relpath(str(path or ""))
    }
    entity_materialization_files = ordered_unique_relpaths(
        list(
            graph_unit.get(
                "entity_materialization_files"
            )
            or []
        )
    )
    if partial_semantic_patch:
        entity_materialization_files = [
            path
            for path in entity_materialization_files
            if path in selected_owner_files
        ]
    allowed_fileset = ordered_unique_relpaths(
        [
            *list(graph_unit.get("editable_files") or []),
            *(
                entity_materialization_files
                if controller_entity_materialization
                else []
            ),
        ]
    )
    if partial_semantic_patch:
        allowed_fileset = [
            path
            for path in allowed_fileset
            if path in selected_owner_files
        ]
    escalation_candidates = ordered_unique_relpaths(list(graph_unit.get("escalation_files") or []))
    if partial_semantic_patch:
        escalation_candidates = [
            path
            for path in escalation_candidates
            if path in selected_owner_files
        ]
    permitted = set(allowed_fileset) | set(escalation_candidates)
    patch = dict(action.get("patch") or {})
    patch_text = str(patch.get("content") or "\n".join(patch.get("content_lines") or []))
    touched_files = touched_files_from_diff(patch_text)
    if not touched_files:
        raise ValueError("agentic patch does not contain a touched file")
    outside = [path for path in touched_files if path not in permitted]
    if outside:
        raise ValueError(
            "agentic patch touches files outside the selected root subgraph: "
            + ", ".join(outside)
        )
    build_metadata = [path for path in touched_files if Path(path).name in BUILD_METADATA_NAMES]
    if build_metadata:
        raise ValueError(
            "source compatibility repair cannot patch build metadata: "
            + ", ".join(build_metadata)
        )

    unit_id = stable_id(
        "agentic-repair",
        spec.case_id,
        str(source_unit.get("id") or source_unit.get("work_unit_id") or ""),
        str(repair_round),
        str(step),
    )
    unit = {
        "id": unit_id,
        "work_unit_id": source_unit.get("work_unit_id") or source_unit.get("id"),
        "source_plan_unit_id": source_unit.get("id"),
        "title": f"agentic validation repair round {repair_round} step {step}",
        "scope": source_unit.get("scope") or f"code-graph-root:{selected_root_id or selected_root_key}",
        "unit_kind": "code_graph_root_repair_unit",
        "execution_phase": unit_execution_phase(source_unit),
        "candidate_files": ordered_unique_relpaths([*allowed_fileset, *escalation_candidates]),
        "allowed_fileset": allowed_fileset,
        "escalation_candidates": escalation_candidates,
        "entity_materialization_files": (
            entity_materialization_files
        ),
        "controller_entity_materialization": (
            controller_entity_materialization
        ),
        "materialization_authority": (
            materialization_authority or None
        ),
        "target_representation_adaptation_authority": dict(
            graph_unit.get(
                "target_representation_adaptation_authority"
            )
            or {}
        ),
        "post_materialization_provider_obligations": (
            post_materialization_provider_obligations
        ),
        "repair_transaction_authorities": list(
            source_unit.get("repair_transaction_authorities") or []
        ),
        "repair_owners": selected_repair_owners,
        "read_only_evidence_files": ordered_unique_relpaths(
            list(source_unit.get("read_only_evidence_files") or [])
        ),
        "diagnostic_signature_count": int(
            source_unit.get("diagnostic_signature_count")
            or len(source_unit.get("diagnostic_signatures") or [])
        ),
        "diagnostic_signatures": list(source_unit.get("diagnostic_signatures") or []),
        "source_issue_samples": [
            diagnostic_event_record(item)
            for item in source_unit.get("source_issue_samples") or []
            if isinstance(item, dict)
        ],
        "diagnostic_routes": dict(source_unit.get("diagnostic_routes") or {}),
        "diagnostic_mappings": list(source_unit.get("diagnostic_mappings") or []),
        "semantic_gap_ids": sorted(addressed_semantic_gap_ids),
        "available_semantic_gap_ids": sorted(available_semantic_gap_ids),
        "semantic_replan_lineage": dict(
            source_unit.get("semantic_replan_lineage") or {}
        ),
        "integration_obligations": list(
            source_unit.get("integration_obligations") or []
        ),
        "required_integration_obligation_ids": list(
            source_unit.get(
                "required_integration_obligation_ids"
            )
            or []
        ),
        "integration_coverage_blockers": list(
            source_unit.get("integration_coverage_blockers") or []
        ),
        "root_cause_cluster": dict(source_unit.get("root_cause_cluster") or {}),
        "code_graph_root_cause_cluster": dict(
            source_unit.get("code_graph_root_cause_cluster") or {}
        ),
        "diagnostic_graph": dict(source_unit.get("diagnostic_graph") or {}),
        "code_graph_scope": dict(source_unit.get("code_graph_scope") or {}),
        "repair_source_pack": _repair_source_pack_with_dynamic_evidence(
            dict(source_unit.get("repair_source_pack") or {}),
            evidence_log=list(evidence_log or []),
            allowed_files=set(allowed_fileset),
        ),
        "repair_decision": {
            "schema": "graph_repair_decision_v2",
            "mode": "patch_root_subgraph",
            "root_node_id": selected_root_id or None,
            "root_key": expected_root_key or None,
            "root_alternative_node_ids": list(graph_unit.get("root_alternative_node_ids") or []),
            "diagnostic_event_ids": list(graph_unit.get("diagnostic_event_ids") or []),
            "graph_complete": bool(
                graph_unit.get("root_mapping_complete") is True
            ),
            "root_mapping_complete": bool(
                graph_unit.get("root_mapping_complete") is True
            ),
            "scope_complete": bool(
                graph_unit.get("scope_complete") is True
            ),
            "repair_authority": graph_unit.get("repair_authority"),
            "consumer_adaptation_authority": dict(
                graph_unit.get("consumer_adaptation_authority") or {}
            ),
            "candidate_write_set": allowed_fileset,
            "entity_materialization_files": (
                entity_materialization_files
            ),
            "escalation_files": escalation_candidates,
            "permission_blocked": False,
        },
    }
    if semantic_phase:
        def selected_evidence_records(
            field: str,
        ) -> list[dict[str, Any]]:
            records = [
                dict(item)
                for item in source_unit.get(field) or []
                if isinstance(item, dict)
            ]
            if not partial_semantic_patch:
                return records
            return [
                item
                for item in records
                if str(
                    item.get("contract_evidence_id")
                    or (
                        item.get("id")
                        if field == "contract_evidence"
                        else ""
                    )
                    or ""
                )
                in selected_evidence_ids
            ]

        selected_contract_evidence = (
            selected_evidence_records("contract_evidence")
        )
        unit.update(
            {
                "semantic_contracts": selected_semantic_contracts,
                "contract_evidence": selected_contract_evidence,
                "contract_evidence_ids": sorted(
                    str(item.get("id") or "")
                    for item in selected_contract_evidence
                    if str(item.get("id") or "")
                ),
                "difference_atoms": selected_evidence_records(
                    "difference_atoms"
                ),
                "demand_anchors": selected_evidence_records(
                    "demand_anchors"
                ),
                "repair_loci": selected_evidence_records(
                    "repair_loci"
                ),
                "repair_owners": selected_repair_owners,
                "demand_anchor_selection": dict(
                    source_unit.get("demand_anchor_selection")
                    or {}
                ),
                "required_semantic_gap_ids": sorted(
                    addressed_semantic_gap_ids
                    & {
                        str(item)
                        for item in source_unit.get(
                            "required_semantic_gap_ids"
                        )
                        or []
                        if str(item)
                    }
                ),
                "pending_semantic_gap_ids": sorted(
                    addressed_semantic_gap_ids
                    & {
                        str(item)
                        for item in source_unit.get(
                            "pending_semantic_gap_ids"
                        )
                        or []
                        if str(item)
                    }
                ),
                "blocked_semantic_gap_ids": sorted(
                    addressed_semantic_gap_ids
                    & {
                        str(item)
                        for item in source_unit.get(
                            "blocked_semantic_gap_ids"
                        )
                        or []
                        if str(item)
                    }
                ),
            }
        )
        unit["difference_atom_ids"] = sorted(
            str(item.get("id") or "")
            for item in unit["difference_atoms"]
            if str(item.get("id") or "")
        )
        semantic_binding = dict(
            source_unit.get("semantic_contract_binding") or {}
        )
        if semantic_binding:
            semantic_binding["semantic_gap_ids"] = sorted(
                addressed_semantic_gap_ids
            )
            unit["semantic_contract_binding"] = semantic_binding
        localization = dict(source_unit.get("localization") or {})
        localization["allowed_fileset"] = list(allowed_fileset)
        localization["targets"] = [
            dict(item)
            for item in localization.get("targets") or []
            if isinstance(item, dict)
            and (
                not partial_semantic_patch
                or normalize_relpath(str(item.get("file") or ""))
                in selected_owner_files
            )
        ]
        unit["localization"] = localization
        unit["patch_targets"] = list(localization["targets"])
        unit["patch_target"] = (
            dict(unit["patch_targets"][0])
            if len(unit["patch_targets"]) == 1
            else {}
        )
    return {
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "summary": {
            "work_unit_count": 1,
            "agentic_repair": True,
            "selected_source_plan_unit_id": source_unit.get("id"),
            "selected_root_node_id": selected_root_id or None,
            "selected_root_key": expected_root_key or None,
        },
        "work_unit_plans": [unit],
    }
