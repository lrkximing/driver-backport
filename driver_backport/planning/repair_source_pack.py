#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from driver_backport.analysis.code_index import (
    STRUCTURAL_ALIGNMENT_RESOLVED_STATUSES,
    parse_module_metadata_calls,
    parse_structural_entity_file,
    structural_entity_neighbor_alignment,
)
from driver_backport.analysis.conditional_contract import (
    conditional_variant_family as _conditional_variant_family,
)
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    normalize_relpath,
    normalize_root_relative,
    ordered_unique_relpaths,
    stable_id,
)
from driver_backport.core.executor import classify_edit_permission


def _text(value: object) -> str:
    return str(value or "").strip()


def _normalize_old_relative(spec: CaseSpec, value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    return normalize_root_relative(text, root=spec.old_root_path, preferred_prefix=spec.driver_root) or ""


def _ordered_relpaths(spec: CaseSpec, values: list[object]) -> list[str]:
    return ordered_unique_relpaths(
        _normalize_old_relative(spec, value) for value in values
    )


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _render_line_range(lines: list[str], *, start_line: int, end_line: int) -> str:
    if not lines:
        return ""
    start = max(1, start_line)
    end = min(len(lines), max(start, end_line))
    return "\n".join(f"{line_no}: {lines[line_no - 1]}" for line_no in range(start, end + 1))


def _extent_source_record(
    spec: CaseSpec,
    *,
    tree: str,
    relative_path: str,
    role: str,
    edit_permission: str,
    reason: str,
    line_start: int,
    line_end: int,
    symbol: str = "",
    provider_kind: str = "",
    source_kind: str = "semantic_extent",
    boundary_context_lines: int = 0,
) -> dict[str, Any]:
    if tree == "donor":
        path = spec.resolve_donor(relative_path)
    else:
        path = spec.resolve_old(relative_path)
    lines = _read_lines(path)
    if lines and line_start > 0:
        start = max(1, min(len(lines), line_start))
        end = max(start, min(len(lines), line_end if line_end > 0 else line_start))
        context_start = max(1, start - max(0, boundary_context_lines))
        context_end = min(
            len(lines),
            end + max(0, boundary_context_lines),
        )
        text = _render_line_range(
            lines,
            start_line=context_start,
            end_line=context_end,
        )
    else:
        start = None
        end = None
        context_start = None
        context_end = None
        text = ""
    return {
        "id": stable_id("src", tree, role, relative_path, str(line_start), str(line_end), symbol),
        "tree": tree,
        "file": relative_path,
        "role": role,
        "edit_permission": edit_permission,
        "reason": reason,
        "source_kind": source_kind,
        "exists": path.exists(),
        "line_start": start,
        "line_end": end,
        "context_line_start": context_start,
        "context_line_end": context_end,
        "line_count": len(lines),
        "symbol": symbol or None,
        "provider_kind": provider_kind or None,
        "text": text,
    }


def _diagnostic_line_map(spec: CaseSpec, work_unit: dict[str, Any]) -> dict[str, list[int]]:
    by_file: dict[str, list[int]] = {}

    def add_location(file_value: object, line_value: object) -> None:
        relative_path = _normalize_old_relative(spec, file_value)
        if not relative_path:
            return
        try:
            line_no = int(line_value or 0)
        except (TypeError, ValueError):
            return
        if line_no <= 0:
            return
        lines = by_file.setdefault(relative_path, [])
        if line_no not in lines:
            lines.append(line_no)

    for issue in [
        *list(work_unit.get("source_issues") or []),
        *list(work_unit.get("source_issue_samples") or []),
    ]:
        if isinstance(issue, dict):
            add_location(issue.get("file"), issue.get("line"))
    return {path: sorted(lines) for path, lines in by_file.items()}


def _module_import_relation_source_records(
    spec: CaseSpec,
    work_unit: dict[str, Any],
    *,
    relative_path: str,
) -> list[dict[str, Any]]:
    issues = {
        str(issue.get("diagnostic_event_id") or ""): issue
        for issue in [
            *list(work_unit.get("source_issues") or []),
            *list(work_unit.get("source_issue_samples") or []),
        ]
        if isinstance(issue, dict)
        and issue.get("kind") == "modpost_namespace_import_missing"
    }
    path = spec.resolve_old(relative_path)
    if not path.is_file():
        return []
    calls = parse_module_metadata_calls(
        path.read_text(encoding="utf-8", errors="ignore")
    )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for mapping in work_unit.get("diagnostic_mappings") or []:
        if not isinstance(mapping, dict) or mapping.get("status") != "mapped_module_namespace_import_relation":
            continue
        issue = issues.get(str(mapping.get("diagnostic_event_id") or ""))
        evidence = mapping.get("symbol_evidence") or {}
        if not isinstance(evidence, dict) or issue is None:
            continue
        namespace = str(evidence.get("namespace") or "")
        if (
            not namespace
            or namespace != str(issue.get("namespace") or "")
            or _normalize_old_relative(spec, evidence.get("module_import_owner_file")) != relative_path
        ):
            continue
        imports = [
            call for call in calls
            if call["name"] == "MODULE_IMPORT_NS" and call["namespace"] == namespace
        ]
        if len(imports) == 1:
            call = imports[0]
        elif not imports:
            licenses = [call for call in calls if call["name"] == "MODULE_LICENSE"]
            if len(licenses) != 1:
                continue
            call = licenses[0]
        else:
            continue
        key = f"{call['line']}:{call['end_line']}"
        if key in seen:
            continue
        seen.add(key)
        records.append(
            _extent_source_record(
                spec,
                tree="old",
                relative_path=relative_path,
                role="allowed_edit_module_import_relation",
                edit_permission="editable",
                reason="current module metadata call at the mapped import relation",
                line_start=call["line"],
                line_end=call["end_line"],
                symbol=namespace,
                provider_kind=call["name"],
                source_kind="module_import_relation_extent",
            )
        )
    return records


def _included_macro_contract_source_record(
    spec: CaseSpec,
    *,
    consumer_file: str,
    macro_name: str,
) -> dict[str, Any] | None:
    root = spec.old_root_path.resolve()
    source = spec.resolve_old(consumer_file)
    if not source.is_file():
        return None
    include_pattern = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
    definition_pattern = re.compile(
        r'^\s*#\s*define\s+' + re.escape(macro_name) + r'\s*\('
    )
    matches: list[tuple[str, int, int]] = []
    for line in _read_lines(source):
        include = include_pattern.match(line)
        if include is None:
            continue
        name = include.group(2)
        candidates = (
            [source.parent / name, root / "include" / name]
            if include.group(1) == '"'
            else [root / "include" / name]
        )
        for candidate in candidates:
            candidate = candidate.resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                continue
            relative = candidate.relative_to(root).as_posix()
            lines = _read_lines(candidate)
            for index, header_line in enumerate(lines):
                if not definition_pattern.match(header_line):
                    continue
                end = index
                while end < len(lines) - 1 and lines[end].rstrip().endswith("\\"):
                    end += 1
                matches.append((relative, index + 1, end + 1))
    matches = sorted(set(matches))
    if len(matches) != 1:
        return None
    relative_path, line_start, line_end = matches[0]
    return _extent_source_record(
        spec,
        tree="old",
        relative_path=relative_path,
        role="read_only_macro_contract",
        edit_permission="read_only",
        reason="exact macro definition in a header directly included by the mapped consumer",
        line_start=line_start,
        line_end=line_end,
        symbol=macro_name,
        source_kind="included_macro_contract_extent",
    )


def _diagnostic_source_record(
    spec: CaseSpec,
    *,
    relative_path: str,
    line_numbers: list[int],
    role: str,
    reason: str,
    edit_permission: str = "read_only",
) -> dict[str, Any]:
    path = spec.resolve_old(relative_path)
    lines = _read_lines(path)
    ranges: list[tuple[int, int]] = []
    for line_no in sorted(set(line_numbers)):
        if not lines:
            continue
        current = max(1, min(len(lines), line_no))
        start = current
        end = current
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))

    text_parts = [
        _render_line_range(lines, start_line=start, end_line=end)
        for start, end in ranges
    ]
    return {
        "id": stable_id("src", "old", role, relative_path, ",".join(str(item) for item in line_numbers)),
        "tree": "old",
        "file": relative_path,
        "role": role,
        "edit_permission": edit_permission,
        "reason": reason,
        "source_kind": "diagnostic_line_context",
        "exists": path.exists(),
        "line_numbers": sorted(set(line_numbers)),
        "line_count": len(lines),
        "text": "\n...\n".join(part for part in text_parts if part),
    }


def _obligation_bound_node_ids(
    work_unit: dict[str, Any],
) -> set[str]:
    """Return graph identities that are direct members of this obligation.

    A mapped root set may contain many valid navigation candidates.  Those
    candidates are not all required source evidence.  This frontier contains
    only persisted provider, repair-locus, demand-anchor, diagnostic, and
    explicitly selected dependency identities.
    """

    selected: set[str] = {
        str(item)
        for item in work_unit.get("dependency_root_node_ids") or []
        if str(item)
    }

    def add_identity(record: dict[str, Any] | None) -> None:
        if not isinstance(record, dict):
            return
        for key in (
            "id",
            "entity_id",
            "node_id",
            "graph_entity_id",
            "old_entity_id",
            "donor_entity_id",
            "owner_entity_id",
            "referenced_entity_id",
            "reference_id",
            "preferred_entity_id",
            "root_node_id",
        ):
            value = str(record.get(key) or "")
            if value:
                selected.add(value)

    for atom in work_unit.get("difference_atoms") or []:
        add_identity(atom if isinstance(atom, dict) else None)
    for evidence in work_unit.get("contract_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        add_identity(evidence)
        for key in (
            "old_entity",
            "donor_entity",
            "consumer_reference",
        ):
            nested = evidence.get(key)
            add_identity(nested if isinstance(nested, dict) else None)
        for key in (
            "old_declarations",
            "donor_declarations",
            "old_exports",
            "donor_exports",
        ):
            for nested in evidence.get(key) or []:
                add_identity(
                    nested if isinstance(nested, dict) else None
                )
    for anchor in work_unit.get("demand_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        add_identity(anchor)
        add_identity(
            anchor.get("owner_entity")
            if isinstance(anchor.get("owner_entity"), dict)
            else None
        )
        add_identity(
            anchor.get("referenced_entity")
            if isinstance(anchor.get("referenced_entity"), dict)
            else None
        )
    for locus in work_unit.get("repair_loci") or []:
        add_identity(locus if isinstance(locus, dict) else None)
    for owner in work_unit.get("repair_owners") or []:
        add_identity(owner if isinstance(owner, dict) else None)
        if isinstance(owner, dict):
            selected.update(
                str(item)
                for item in owner.get("evidence") or []
                if str(item)
            )
    for issue in [
        *list(work_unit.get("source_issues") or []),
        *list(work_unit.get("source_issue_samples") or []),
    ]:
        if not isinstance(issue, dict):
            continue
        add_identity(issue)
        event_id = str(issue.get("diagnostic_event_id") or "")
        if event_id:
            selected.add(event_id)
            selected.add(f"diagnostic:{event_id}")
    return selected


def _code_graph_nodes(work_unit: dict[str, Any]) -> list[dict[str, Any]]:
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    graphs = [work_unit.get("diagnostic_graph")]
    context = work_unit.get("context_expansion")
    if isinstance(context, dict):
        scope = context.get("code_graph_scope")
        if isinstance(scope, dict):
            graphs.append(scope.get("explanation_graph"))
    for graph in graphs:
        if not isinstance(graph, dict):
            continue
        for node in graph.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "")
            if node_id and node_id not in nodes_by_id:
                nodes_by_id[node_id] = dict(node)
        edges.extend(
            dict(edge)
            for edge in graph.get("edges") or []
            if isinstance(edge, dict) and edge.get("from") and edge.get("to")
        )
    root = dict(
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    root_id = str(root.get("node_id") or root.get("entity_id") or "")
    event_ids = {
        str(event_id)
        for event_id in root.get("diagnostic_event_ids") or []
        if str(event_id)
    }
    diagnostic_ids = {
        node_id
        for node_id, node in nodes_by_id.items()
        if str(node.get("kind") or "") == "Diagnostic"
        and str(node.get("diagnostic_event_id") or "") in event_ids
    }
    diagnostic_ids.update(f"diagnostic:{event_id}" for event_id in event_ids)
    adjacency: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    for edge in edges:
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        adjacency.setdefault(source, set()).add(target)
        reverse.setdefault(target, set()).add(source)

    def reachable(starts: set[str], graph: dict[str, set[str]]) -> set[str]:
        result: set[str] = set()
        pending = list(starts)
        while pending:
            node_id = pending.pop()
            if node_id in result:
                continue
            result.add(node_id)
            pending.extend(graph.get(node_id, set()))
        return result

    selected_ids: set[str] = set()
    if root_id:
        from_diagnostics = reachable(diagnostic_ids, adjacency)
        to_root = reachable({root_id}, reverse)
        selected_ids = from_diagnostics & to_root
        selected_ids.add(root_id)
        selected_ids.update(
            _obligation_bound_node_ids(work_unit) & set(nodes_by_id)
        )
        donor_owner_ids = {
            root_id
            for root in [nodes_by_id.get(root_id) or {}]
            if str(root.get("tree") or "").startswith("donor_")
        }
        donor_owner_ids.update(
            str(edge.get("to") or "")
            for edge in edges
            if str(edge.get("type") or "") == "same_contract_as"
            and str(edge.get("from") or "") == root_id
            and str((nodes_by_id.get(str(edge.get("to") or "")) or {}).get("tree") or "").startswith(
                "donor_"
            )
        )
        dependency_target_ids: set[str] = set()
        dependency_reference_ids: set[str] = set()
        for edge in edges:
            if (
                str(edge.get("type") or "") != "implementation_depends_on"
                or str(edge.get("from") or "") not in donor_owner_ids
            ):
                continue
            dependency_target_ids.add(str(edge.get("to") or ""))
            reference_id = str(edge.get("reference_id") or "")
            if reference_id:
                dependency_reference_ids.add(reference_id)
        selected_ids.update(donor_owner_ids)
        selected_ids.update(dependency_target_ids)
        selected_ids.update(dependency_reference_ids)
        for edge in edges:
            if str(edge.get("type") or "") != "same_contract_as":
                continue
            source = str(edge.get("from") or "")
            target = str(edge.get("to") or "")
            if source in dependency_target_ids or target in dependency_target_ids:
                selected_ids.update([source, target])
    else:
        selected_ids = (
            _obligation_bound_node_ids(work_unit)
            & set(nodes_by_id)
        )
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if str(edge.get("type") or "") != "same_contract_as":
                continue
            source = str(edge.get("from") or "")
            target = str(edge.get("to") or "")
            if source in selected_ids and target in nodes_by_id:
                if target not in selected_ids:
                    selected_ids.add(target)
                    changed = True
            elif target in selected_ids and source in nodes_by_id:
                if source not in selected_ids:
                    selected_ids.add(source)
                    changed = True
    nodes = [
        node
        for node_id, node in nodes_by_id.items()
        if node_id in selected_ids
    ]
    return sorted(
        nodes,
        key=lambda node: (
            0 if str(node.get("id") or "") == root_id else 1,
            str(node.get("tree") or ""),
            normalize_relpath(
                str(node.get("file") or node.get("path") or node.get("source_file") or "")
            ),
            int((node.get("extent") or {}).get("start_line") or node.get("line") or 0),
            str(node.get("id") or ""),
        ),
    )


def _code_graph_source_records(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    tree: str,
    relative_path: str,
    role: str,
    edit_permission: str,
    reason: str,
    source_kind: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    expected_trees = {"old"} if tree == "old" else {"donor", "donor_0"}
    for node in _code_graph_nodes(work_unit):
        node_tree = str(node.get("tree") or "")
        if tree == "donor" and node_tree.startswith("donor_"):
            node_tree = "donor"
        node_file = normalize_relpath(
            str(node.get("file") or node.get("path") or node.get("source_file") or "")
        )
        extent = node.get("extent") if isinstance(node.get("extent"), dict) else {}
        line_start = int(extent.get("start_line") or node.get("line") or 0)
        line_end = int(extent.get("end_line") or line_start)
        if node_tree not in expected_trees or node_file != relative_path or line_start <= 0:
            continue
        records.append(
            _extent_source_record(
                spec,
                tree=tree,
                relative_path=relative_path,
                role=role,
                edit_permission=edit_permission,
                reason=reason,
                line_start=line_start,
                line_end=line_end,
                symbol=_text(node.get("symbol") or node.get("name")),
                provider_kind=_text(node.get("entity_kind") or node.get("kind")),
                source_kind=source_kind,
                boundary_context_lines=3,
            )
        )
        records[-1]["code_graph_node_id"] = node.get("id")
    return records


def _repair_owner_source_records(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    relative_path: str,
) -> list[dict[str, Any]]:
    """Resolve exact writable owners against the current candidate tree."""

    owners = [
        owner
        for owner in work_unit.get("repair_owners") or []
        if isinstance(owner, dict)
        and bool(owner.get("allowed_as_patch_target"))
        and normalize_relpath(str(owner.get("owner_file") or ""))
        == relative_path
    ]
    if not owners:
        return []
    report = parse_structural_entity_file(
        root=Path(spec.old_root_path),
        relative_path=relative_path,
        tree="old",
    )
    if not report.get("parse_ok"):
        return []
    entities = [
        entity
        for entity in report.get("entities") or []
        if isinstance(entity, dict)
    ]
    records: list[dict[str, Any]] = []
    seen_contracts: set[tuple[str, str]] = set()
    for owner in owners:
        contract_key = _text(owner.get("owner_contract_key"))
        owner_kind = _text(owner.get("owner_entity_kind")).lower()
        owner_symbol = _text(owner.get("owner_symbol"))
        matches = [
            entity
            for entity in entities
            if contract_key
            and _text(entity.get("contract_key") or entity.get("key"))
            == contract_key
            and (
                not owner_kind
                or _text(entity.get("kind") or entity.get("entity_kind")).lower()
                == owner_kind
            )
        ]
        if not matches and owner_kind and owner_symbol:
            matches = [
                entity
                for entity in entities
                if _text(entity.get("kind") or entity.get("entity_kind")).lower()
                == owner_kind
                and _text(entity.get("symbol") or entity.get("name"))
                == owner_symbol
            ]
        if len(matches) != 1:
            continue
        entity = matches[0]
        resolved_contract = _text(
            entity.get("contract_key") or entity.get("key")
        )
        identity = (owner_kind, resolved_contract or contract_key)
        if identity in seen_contracts:
            continue
        seen_contracts.add(identity)
        line_start, line_end = _entity_extent_lines(entity)
        if line_start <= 0:
            continue
        record = _extent_source_record(
            spec,
            tree="old",
            relative_path=relative_path,
            role="allowed_edit_current_repair_owner",
            edit_permission="editable",
            reason=(
                "exact writable repair owner rebound against the current "
                "candidate tree"
            ),
            line_start=line_start,
            line_end=line_end,
            symbol=owner_symbol
            or _text(entity.get("symbol") or entity.get("name")),
            provider_kind=owner_kind,
            source_kind="old_current_repair_owner_extent",
            boundary_context_lines=3,
        )
        record["repair_owner_id"] = owner.get("id")
        record["entity_id"] = owner.get("owner_entity_id")
        record["contract_key"] = resolved_contract or contract_key
        record["owner_contract_key"] = resolved_contract or contract_key
        records.append(record)
    return records


def _include_relation_source_records(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    relative_path: str,
) -> list[dict[str, Any]]:
    from driver_backport.planning.response_apply import (
        _current_include_relation_boundary,
    )

    records: list[dict[str, Any]] = []
    for owner in work_unit.get("repair_owners") or []:
        if (
            not isinstance(owner, dict)
            or owner.get("owner_role") != "include_relation_owner"
            or owner.get("allowed_as_patch_target") is not True
            or _normalize_old_relative(spec, owner.get("owner_file"))
            != relative_path
        ):
            continue
        mode = _text(owner.get("include_relation_mode"))
        current_name = _text(owner.get("current_include_name"))
        required_name = _text(owner.get("required_include_name"))
        boundary = _current_include_relation_boundary(
            spec,
            relative_path=relative_path,
            relation_mode=mode,
            current_include_name=current_name,
            required_include_name=required_name,
            expected_line=int(owner.get("expected_line") or 0),
        )
        if boundary.get("status") != "resolved":
            continue
        records.append(
            _extent_source_record(
                spec,
                tree="old",
                relative_path=relative_path,
                role="allowed_edit_include_relation_locus",
                edit_permission="editable",
                reason="current source at the exact authorized include relation",
                line_start=int(boundary["start_line"]),
                line_end=int(boundary["end_line"]),
                symbol=required_name or current_name,
                provider_kind="include_relation",
                source_kind="old_include_relation_locus",
            )
        )
    return records


def _code_graph_enclosing_container_source_records(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    tree: str,
    relative_path: str,
    edit_permission: str,
) -> list[dict[str, Any]]:
    """Expose the complete current container for graph-selected members.

    A donor-only field or enumerator cannot be inserted safely from its
    isolated declaration extent.  The editable evidence must be the current
    old-tree struct/union/enum, while the donor container remains read-only
    reference evidence.  Container identity comes from the parsed graph
    record; no symbol-name or source-text guessing is involved.
    """

    member_nodes = []
    for node in _code_graph_nodes(work_unit):
        node_tree = str(node.get("tree") or "")
        if not node_tree.startswith("donor"):
            continue
        node_file = normalize_relpath(
            str(
                node.get("file")
                or node.get("path")
                or node.get("source_file")
                or ""
            )
        )
        container = (
            dict(node.get("container") or {})
            if isinstance(node.get("container"), dict)
            else {}
        )
        if (
            node_file == relative_path
            and str(node.get("kind") or node.get("entity_kind") or "")
            in {"field", "enum_constant"}
            and str(container.get("container_kind") or "")
            in {"struct", "union", "enum"}
            and str(container.get("container_name") or "")
        ):
            member_nodes.append((node, container))
    if not member_nodes:
        return []

    root = (
        Path(spec.old_root_path)
        if tree == "old"
        else next(
            (
                Path(candidate)
                for candidate in spec.donor_root_paths
                if (Path(candidate) / relative_path).is_file()
            ),
            None,
        )
    )
    if root is None:
        return []
    report = parse_structural_entity_file(
        root=root,
        relative_path=relative_path,
        tree="old" if tree == "old" else "donor_0",
    )
    if not report.get("parse_ok"):
        return []

    records: list[dict[str, Any]] = []
    seen_containers: set[tuple[str, str]] = set()
    entities = [
        entity
        for entity in report.get("entities") or []
        if isinstance(entity, dict)
    ]
    for _node, container in member_nodes:
        container_kind = str(container["container_kind"])
        container_name = str(container["container_name"])
        identity = (container_kind, container_name)
        if identity in seen_containers:
            continue
        seen_containers.add(identity)
        expected_cursor_kind = (
            f"TREE_SITTER_{container_kind.upper()}_SPECIFIER"
        )
        matches = [
            entity
            for entity in entities
            if str(entity.get("kind") or entity.get("entity_kind") or "")
            == "type"
            and str(entity.get("symbol") or entity.get("name") or "")
            == container_name
            and str(entity.get("cursor_kind") or "")
            == expected_cursor_kind
            and bool(entity.get("is_definition"))
        ]
        if len(matches) != 1:
            continue
        entity = matches[0]
        extent = (
            dict(entity.get("extent") or {})
            if isinstance(entity.get("extent"), dict)
            else {}
        )
        line_start = int(extent.get("start_line") or 0)
        line_end = int(extent.get("end_line") or line_start)
        if line_start <= 0:
            continue
        record = _extent_source_record(
            spec,
            tree=tree,
            relative_path=relative_path,
            role=(
                "allowed_edit_enclosing_contract_container"
                if tree == "old"
                else "donor_enclosing_contract_container"
            ),
            edit_permission=edit_permission,
            reason=(
                "complete current structural container for graph-selected "
                "donor member insertion"
                if tree == "old"
                else (
                    "complete donor structural container for graph-selected "
                    "member reference"
                )
            ),
            line_start=line_start,
            line_end=line_end,
            symbol=container_name,
            provider_kind=container_kind,
            source_kind=(
                "old_enclosing_contract_container"
                if tree == "old"
                else "donor_enclosing_contract_container"
            ),
        )
        record["container_contract"] = {
            "container_kind": container_kind,
            "container_name": container_name,
            "entity_id": entity.get("id") or entity.get("entity_id"),
            "contract_key": entity.get("contract_key")
            or entity.get("key"),
        }
        records.append(record)
    return records


def _external_provider_contract_source_records(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Materialize donor references to a provider outside its definition extent."""

    symbols = {
        _text(issue.get("symbol"))
        for issue in work_unit.get("source_issues") or []
        if isinstance(issue, dict)
        and not _text(issue.get("file") or issue.get("source_file"))
        and _text(issue.get("symbol"))
    }
    if not symbols:
        return []
    graphs = [work_unit.get("diagnostic_graph")]
    context = work_unit.get("context_expansion")
    if isinstance(context, dict):
        scope = context.get("code_graph_scope")
        if isinstance(scope, dict):
            graphs.append(scope.get("explanation_graph"))
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for graph in graphs:
        if not isinstance(graph, dict):
            continue
        for node in graph.get("nodes") or []:
            if isinstance(node, dict) and node.get("id"):
                nodes[str(node["id"])] = dict(node)
        edges.extend(
            dict(edge)
            for edge in graph.get("edges") or []
            if isinstance(edge, dict)
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for edge in edges:
        if str(edge.get("type") or "") != "implementation_depends_on":
            continue
        owner = nodes.get(str(edge.get("from") or "")) or {}
        target = nodes.get(str(edge.get("to") or "")) or {}
        reference_id = str(edge.get("reference_id") or "")
        reference = nodes.get(reference_id) or {}
        symbol = _text(reference.get("symbol"))
        if (
            symbol not in symbols
            or symbol != _text(owner.get("symbol"))
            or symbol != _text(target.get("symbol"))
            or not str(reference.get("tree") or "").startswith("donor_")
        ):
            continue
        owner_extent = owner.get("extent") if isinstance(owner.get("extent"), dict) else {}
        reference_extent = (
            reference.get("extent") if isinstance(reference.get("extent"), dict) else {}
        )
        owner_start = int(owner_extent.get("start_line") or 0)
        owner_end = int(owner_extent.get("end_line") or 0)
        reference_start = int(reference_extent.get("start_line") or 0)
        reference_end = int(reference_extent.get("end_line") or reference_start)
        if (
            reference_start <= 0
            or not owner_start
            or not owner_end
            or owner_start <= reference_start <= owner_end
        ):
            continue
        relative_path = normalize_relpath(
            str(
                reference.get("file")
                or reference.get("path")
                or reference.get("source_file")
                or ""
            )
        )
        if not relative_path:
            continue
        record = _extent_source_record(
            spec,
            tree="donor",
            relative_path=relative_path,
            role="donor_external_provider_contract_reference",
            edit_permission="read_only",
            reason=(
                "selected provider is referenced outside its definition extent; "
                "this exact graph location is part of its linkage/API contract"
            ),
            line_start=reference_start,
            line_end=reference_end,
            symbol=symbol,
            provider_kind=_text(owner.get("entity_kind") or owner.get("kind")),
            source_kind="donor_external_provider_contract_reference",
        )
        record["code_graph_node_id"] = reference_id
        _add_unique_record(records, record, seen)
    return records


def _contract_owner_files(work_unit: dict[str, Any]) -> set[str]:
    files = {
        normalize_relpath(str(node.get("file") or node.get("path") or ""))
        for node in _code_graph_nodes(work_unit)
        if str(node.get("kind") or "") == "ContractOwner"
        and str(node.get("file") or node.get("path") or "").strip()
    }
    for mapping in work_unit.get("diagnostic_mappings") or []:
        if not isinstance(mapping, dict):
            continue
        for evidence in [
            *list(mapping.get("donor_position_evidence") or []),
            *list(mapping.get("root_owner_alignment_evidence") or []),
        ]:
            if not isinstance(evidence, dict):
                continue
            owner_file = normalize_relpath(str(evidence.get("contract_owner_file") or ""))
            if owner_file:
                files.add(owner_file)
    scope = work_unit.get("code_graph_scope")
    if not isinstance(scope, dict):
        scope = {}
    for evidence in scope.get("repair_dependency_alignments") or []:
        if not isinstance(evidence, dict):
            continue
        owner_file = normalize_relpath(str(evidence.get("contract_owner_file") or ""))
        if owner_file:
            files.add(owner_file)
    return files


def _owner_alignment_evidence(
    work_unit: dict[str, Any],
    *,
    relative_path: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for mapping in work_unit.get("diagnostic_mappings") or []:
        if not isinstance(mapping, dict):
            continue
        for evidence in [
            *list(mapping.get("donor_position_evidence") or []),
            *list(mapping.get("root_owner_alignment_evidence") or []),
        ]:
            if not isinstance(evidence, dict):
                continue
            owner_file = normalize_relpath(str(evidence.get("contract_owner_file") or ""))
            alignment = evidence.get("owner_alignment")
            if owner_file != relative_path or not isinstance(alignment, dict):
                continue
            key = (str(evidence.get("target_entity_id") or ""), str(alignment.get("status") or ""))
            if key in seen:
                continue
            seen.add(key)
            records.append(alignment)
    scope = work_unit.get("code_graph_scope")
    if not isinstance(scope, dict):
        scope = {}
    for evidence in scope.get("repair_dependency_alignments") or []:
        if not isinstance(evidence, dict):
            continue
        owner_file = normalize_relpath(str(evidence.get("contract_owner_file") or ""))
        alignment = evidence.get("owner_alignment")
        if owner_file != relative_path or not isinstance(alignment, dict):
            continue
        key = (
            str(evidence.get("target_entity_id") or ""),
            str(alignment.get("status") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(alignment)
    return records


def _owner_alignment_source_records(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    relative_path: str,
    tree: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    donor_roots = list(spec.donor_root_paths)
    donor_root = next(
        (root for root in donor_roots if (root / relative_path).is_file()),
        donor_roots[0] if donor_roots else None,
    )
    old_report = parse_structural_entity_file(
        root=Path(spec.old_root_path),
        relative_path=relative_path,
        tree="old",
    )
    donor_report = (
        parse_structural_entity_file(
            root=donor_root,
            relative_path=relative_path,
            tree="donor_0",
        )
        if donor_root is not None
        else {"parse_ok": False, "parse_error": "donor_root_unavailable", "entities": []}
    )

    def exact_matches(report: dict[str, Any], entity: dict[str, Any]) -> list[dict[str, Any]]:
        current_entities = [
            item for item in report.get("entities") or [] if isinstance(item, dict)
        ]
        contract = _text(entity.get("contract_key") or entity.get("key"))
        if contract:
            return [
                item
                for item in current_entities
                if _text(item.get("contract_key") or item.get("key")) == contract
            ]
        expected_kind = _text(entity.get("kind") or entity.get("entity_kind")).lower()
        expected_symbol = _text(entity.get("symbol") or entity.get("name"))
        return [
            item
            for item in current_entities
            if _text(item.get("kind") or item.get("entity_kind")).lower() == expected_kind
            and _text(item.get("symbol") or item.get("name")) == expected_symbol
        ]

    for cycle_alignment in _owner_alignment_evidence(work_unit, relative_path=relative_path):
        alignment = cycle_alignment
        if old_report.get("parse_ok") and donor_report.get("parse_ok"):
            cycle_target = cycle_alignment.get("target")
            if not isinstance(cycle_target, dict) and _text(
                cycle_alignment.get("graph_target_contract_key")
            ):
                contract_key = _text(cycle_alignment.get("graph_target_contract_key"))
                entity_kind, _, symbol = contract_key.partition(":")
                cycle_target = {
                    "contract_key": contract_key,
                    "key": contract_key,
                    "kind": entity_kind,
                    "symbol": symbol,
                }
            target_matches = (
                exact_matches(donor_report, cycle_target)
                if isinstance(cycle_target, dict)
                else []
            )
            variant_family = _conditional_variant_family(
                donor_lines=_read_lines(spec.resolve_donor(relative_path)),
                old_lines=_read_lines(spec.resolve_old(relative_path)),
                target_entities=target_matches,
            )
            if variant_family is not None:
                line_start = int(
                    variant_family[
                        "old_start_line" if tree == "old" else "donor_start_line"
                    ]
                )
                line_end = int(
                    variant_family[
                        "old_end_line" if tree == "old" else "donor_end_line"
                    ]
                )
                record = _extent_source_record(
                    spec,
                    tree=tree,
                    relative_path=relative_path,
                    role=(
                        "allowed_edit_conditional_contract_context"
                        if tree == "old"
                        else "donor_conditional_contract_reference"
                    ),
                    edit_permission="editable" if tree == "old" else "read_only",
                    reason=(
                        "same contract has mutually exclusive preprocessor-branch variants; "
                        "the complete corresponding conditional block is the structural unit"
                    ),
                    line_start=line_start,
                    line_end=line_end,
                    symbol=_text((cycle_target or {}).get("symbol") or (cycle_target or {}).get("name")),
                    provider_kind=_text((cycle_target or {}).get("kind") or (cycle_target or {}).get("entity_kind")),
                    source_kind=(
                        "old_conditional_contract_variant_set"
                        if tree == "old"
                        else "donor_conditional_contract_variant_set"
                    ),
                )
                record["conditional_contract"] = variant_family
                record["alignment_position"] = "conditional_variant_set"
                record["alignment_status"] = "corresponding_preprocessor_conditional"
                record["structural_extent_status"] = "current_tree_conditional_block"
                records.append(record)
                continue
            if len(target_matches) != 1:
                blockers.append(
                    {
                        "owner_kind": "structural_alignment_target_relocation",
                        "owner_file": relative_path,
                        "tree": "donor",
                        "match_count": len(target_matches),
                        "reason": (
                            "the donor target structural contract is missing or ambiguous during "
                            "active-unit source-pack materialization"
                        ),
                    }
                )
                continue
            alignment = structural_entity_neighbor_alignment(
                target_entity_id=str(
                    target_matches[0].get("id") or target_matches[0].get("entity_id") or ""
                ),
                donor_entities=donor_report.get("entities") or [],
                old_entities=old_report.get("entities") or [],
            )
            if (
                alignment.get("status")
                not in STRUCTURAL_ALIGNMENT_RESOLVED_STATUSES
            ):
                blockers.append(
                    {
                        "owner_kind": "structural_alignment_refresh",
                        "owner_file": relative_path,
                        "tree": tree,
                        "alignment_status": alignment.get("status"),
                        "reason": alignment.get("reason")
                        or "current old/donor structural sequences do not provide a safe insertion anchor",
                    }
                )
                continue
        if tree == "old":
            candidates = [
                ("counterpart", alignment.get("old_counterpart")),
                ("predecessor", (alignment.get("predecessor") or {}).get("old")),
                ("successor", (alignment.get("successor") or {}).get("old")),
                *[
                    (f"between_{index:02d}", entity)
                    for index, entity in enumerate(alignment.get("old_between") or [], start=1)
                ],
            ]
        else:
            donor_between = [
                entity
                for entity in alignment.get("donor_between") or []
                if isinstance(entity, dict)
            ]
            candidates = [
                *(
                    [
                        (f"between_{index:02d}", entity)
                        for index, entity in enumerate(donor_between, start=1)
                    ]
                    if donor_between
                    else [("target", alignment.get("target"))]
                ),
                ("predecessor", (alignment.get("predecessor") or {}).get("donor")),
                ("successor", (alignment.get("successor") or {}).get("donor")),
            ]
        cycle_candidates = {
            "counterpart": cycle_alignment.get("old_counterpart"),
            "target": cycle_alignment.get("target"),
            "predecessor": (cycle_alignment.get("predecessor") or {}).get(
                "old" if tree == "old" else "donor"
            ),
            "successor": (cycle_alignment.get("successor") or {}).get(
                "old" if tree == "old" else "donor"
            ),
        }
        structural_report = old_report if tree == "old" else donor_report
        alignment_record_start = len(records)
        for position, entity in candidates:
            if not isinstance(entity, dict):
                continue
            source_entity = entity
            relocation_status = "cycle_graph_extent"
            if structural_report.get("parse_ok"):
                matches = exact_matches(structural_report, entity)
                contract = _text(entity.get("contract_key") or entity.get("key"))
                relocation_identity = contract or (
                    f"{_text(entity.get('kind') or entity.get('entity_kind')).lower()}:"
                    f"{_text(entity.get('symbol') or entity.get('name'))}"
                )
                if len(matches) != 1:
                    blockers.append(
                        {
                            "owner_kind": "structural_alignment_anchor_relocation",
                            "owner_file": relative_path,
                            "tree": tree,
                            "position": position,
                            "contract_key": contract or None,
                            "structural_identity": relocation_identity,
                            "match_count": len(matches),
                            "reason": (
                                "current-tree structural anchor is missing or ambiguous after earlier "
                                "unit patches; a stale cycle-start line extent was not used"
                            ),
                        }
                    )
                    continue
                source_entity = matches[0]
                relocation_status = "current_tree_exact_contract"
            else:
                relocation_status = str(
                    (structural_report or {}).get("parse_error")
                    or "current_tree_structural_parser_unavailable"
                )
            extent = (
                source_entity.get("extent")
                if isinstance(source_entity.get("extent"), dict)
                else {}
            )
            line_start = int(extent.get("start_line") or 0)
            line_end = int(extent.get("end_line") or line_start)
            if line_start <= 0:
                continue
            record = _extent_source_record(
                spec,
                tree=tree,
                relative_path=relative_path,
                role=(
                    (
                        "allowed_edit_structural_gap_entity"
                        if position.startswith("between_")
                        else "allowed_edit_structural_owner_anchor"
                    )
                    if tree == "old"
                    else (
                        "donor_structural_gap_entity"
                        if position.startswith("between_")
                        else "donor_structural_owner_reference"
                    )
                ),
                edit_permission="editable" if tree == "old" else "read_only",
                reason=(
                    f"{position} entity from ordered old/donor structural owner alignment; "
                    f"alignment_status={alignment.get('status')}"
                ),
                line_start=line_start,
                line_end=line_end,
                symbol=_text(source_entity.get("symbol") or source_entity.get("name")),
                provider_kind=_text(source_entity.get("kind") or source_entity.get("entity_kind")),
                source_kind=(
                    (
                        "old_structural_owner_between_entity"
                        if position.startswith("between_")
                        else "old_structural_owner_alignment_anchor"
                    )
                    if tree == "old"
                    else (
                        "donor_structural_owner_between_entity"
                        if position.startswith("between_")
                        else "donor_structural_owner_alignment_entity"
                    )
                ),
            )
            cycle_entity = cycle_candidates.get(position)
            if cycle_entity is None and position.startswith("between_"):
                cycle_target = cycle_candidates.get("target")
                if isinstance(cycle_target, dict) and _text(
                    cycle_target.get("contract_key") or cycle_target.get("key")
                ) == _text(entity.get("contract_key") or entity.get("key")):
                    cycle_entity = cycle_target
            record["code_graph_node_id"] = (
                (cycle_entity or {}).get("id")
                or (cycle_entity or {}).get("entity_id")
                or entity.get("id")
                or entity.get("entity_id")
            )
            record["materialized_entity_id"] = (
                source_entity.get("id") or source_entity.get("entity_id")
            )
            record["alignment_position"] = position
            record["alignment_status"] = alignment.get("status")
            record["structural_extent_status"] = relocation_status
            records.append(record)
        if tree == "old" and not alignment.get("old_between"):
            current_anchors = {
                record.get("alignment_position"): record
                for record in records[alignment_record_start:]
                if record.get("structural_extent_status")
                == "current_tree_exact_contract"
            }
            predecessor = current_anchors.get("predecessor")
            successor = current_anchors.get("successor")
            if predecessor and successor:
                gap_start = int(predecessor["line_end"]) + 1
                gap_end = int(successor["line_start"]) - 1
                intervening_entity = any(
                    int((entity.get("extent") or {}).get("start_line") or 0)
                    <= gap_end
                    and int((entity.get("extent") or {}).get("end_line") or 0)
                    >= gap_start
                    for entity in old_report.get("entities") or []
                    if isinstance(entity, dict)
                )
                if gap_start <= gap_end and not intervening_entity:
                    gap_record = _extent_source_record(
                        spec,
                        tree="old",
                        relative_path=relative_path,
                        role="old_structural_inter_owner_gap",
                        edit_permission="read_only",
                        reason=(
                            "exact current-tree text between uniquely "
                            "aligned structural neighbors"
                        ),
                        line_start=gap_start,
                        line_end=gap_end,
                        source_kind=(
                            "old_structural_inter_owner_gap_context"
                        ),
                    )
                    gap_record["alignment_status"] = alignment.get("status")
                    records.append(gap_record)
    return records, blockers


def _provider_records_from_resolution(resolution: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ("donor_entities", "target_entities"):
        for entity in resolution.get(key) or []:
            if isinstance(entity, dict):
                records.append(entity)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    for record in records:
        key = (
            _text(record.get("tree")),
            _text(record.get("kind")),
            normalize_relpath(_text(record.get("file"))),
            int((record.get("extent") or {}).get("start_line") or 0),
            int((record.get("extent") or {}).get("end_line") or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def _entity_extent_lines(record: dict[str, Any]) -> tuple[int, int]:
    extent = record.get("extent") if isinstance(record.get("extent"), dict) else {}
    start = int(extent.get("start_line") or record.get("line_start") or 0)
    end = int(extent.get("end_line") or record.get("line_end") or start)
    return start, max(start, end)


def _provider_resolutions(work_unit: dict[str, Any]) -> list[dict[str, Any]]:
    resolutions: list[dict[str, Any]] = []
    for resolution in work_unit.get("provider_resolutions") or []:
        if isinstance(resolution, dict):
            resolutions.append(resolution)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for resolution in resolutions:
        key = (
            _text(resolution.get("backend")),
            _text(resolution.get("symbol")),
            _text(resolution.get("status")),
            _text(resolution.get("preferred_entity_id")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolution)
    return deduped


def _owner_read_only_files(spec: CaseSpec, work_unit: dict[str, Any]) -> list[str]:
    values: list[object] = []
    for owner in work_unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        values.extend(owner.get("read_only_evidence_files") or [])
        values.extend(owner.get("symptom_files") or [])
        values.extend(owner.get("trigger_files") or [])
    return _ordered_relpaths(spec, values)


def _materializable_provider_neighborhood_source_records(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    relative_path: str,
) -> list[dict[str, Any]]:
    owners = [
        owner
        for owner in work_unit.get("repair_owners") or []
        if isinstance(owner, dict)
        and (
            owner.get("allowed_for_entity_materialization") is True
            or owner.get("exact_absent_entity_insertion_allowed") is True
        )
        and _normalize_old_relative(spec, owner.get("owner_file")) == relative_path
        and _text(owner.get("owner_contract_key"))
    ]
    if not owners:
        return []
    old_report = parse_structural_entity_file(
        root=Path(spec.old_root_path),
        relative_path=relative_path,
        tree="old",
    )
    if not old_report.get("parse_ok"):
        return []
    donor_roots = list(spec.donor_root_paths)
    records: list[dict[str, Any]] = []
    for owner in owners:
        tree = _text(owner.get("owner_entity_tree"))
        if not tree.startswith("donor_"):
            continue
        try:
            donor_index = int(tree.partition("_")[2])
        except ValueError:
            continue
        if donor_index < 0 or donor_index >= len(donor_roots):
            continue
        donor_root = donor_roots[donor_index]
        donor_report = parse_structural_entity_file(
            root=donor_root,
            relative_path=relative_path,
            tree=tree,
        )
        if not donor_report.get("parse_ok"):
            continue
        donor_matches = [
            entity
            for entity in donor_report.get("entities") or []
            if entity.get("contract_key") == owner.get("owner_contract_key")
            and _text(entity.get("kind")).lower()
            == _text(owner.get("owner_entity_kind")).lower()
            and entity.get("is_definition") is True
        ]
        if len(donor_matches) != 1:
            continue
        alignment = structural_entity_neighbor_alignment(
            target_entity_id=_text(donor_matches[0].get("id")),
            donor_entities=donor_report.get("entities") or [],
            old_entities=old_report.get("entities") or [],
        )
        if (
            alignment.get("status") not in STRUCTURAL_ALIGNMENT_RESOLVED_STATUSES
            or not alignment.get("conditional_context_aligned")
        ):
            continue
        neighborhood = list(alignment.get("old_between") or [])
        if owner.get("exact_absent_entity_insertion_allowed") is True:
            neighborhood.extend(
                anchor for side in ("predecessor", "successor")
                if (anchor := (alignment.get(side) or {}).get("old"))
            )
        for entity in neighborhood:
            extent = entity.get("extent") or {}
            start = int(extent.get("start_line") or 0)
            end = int(extent.get("end_line") or 0)
            if start <= 0 or end < start:
                continue
            records.append(
                _extent_source_record(
                    spec,
                    tree="old",
                    relative_path=relative_path,
                    role="old_materializable_provider_neighborhood",
                    edit_permission="read_only",
                    reason=(
                        "current old-tree entity between the same structural "
                        "neighbors as a materializable donor provider; "
                        "compatibility is not established"
                    ),
                    line_start=start,
                    line_end=end,
                    symbol=_text(entity.get("symbol")),
                    provider_kind=_text(entity.get("kind")),
                    source_kind="old_structural_provider_neighborhood",
                    boundary_context_lines=3,
                )
            )
    return records


def _diagnostic_role_for_file(relative_path: str, work_unit: dict[str, Any]) -> str:
    symptom = set()
    trigger = set()
    for owner in work_unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        symptom.update(normalize_relpath(str(item)) for item in owner.get("symptom_files") or [])
        trigger.update(normalize_relpath(str(item)) for item in owner.get("trigger_files") or [])
    if relative_path in symptom and relative_path in trigger:
        return "diagnostic_symptom_and_trigger"
    if relative_path in symptom:
        return "diagnostic_symptom"
    if relative_path in trigger:
        return "diagnostic_trigger"
    return "read_only_context"


def _add_unique_record(records: list[dict[str, Any]], record: dict[str, Any], seen: set[str]) -> None:
    key = str(record.get("id") or "")
    if not key:
        key = "|".join(
            str(record.get(field) or "")
            for field in ("tree", "file", "role", "source_kind", "line_start", "line_end", "symbol")
        )
    entity_ids = list(
        dict.fromkeys(
            str(item)
            for item in [
                record.get("entity_id"),
                record.get("code_graph_node_id"),
                record.get("node_id"),
                record.get("materialized_entity_id"),
                *list(record.get("entity_ids") or []),
            ]
            if str(item or "")
        )
    )
    if entity_ids:
        key = f"{key}|entities={'|'.join(entity_ids)}"
    if key in seen:
        return
    seen.add(key)
    records.append(record)


def _missing_include_contract_source_records(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    allowed_files: list[str],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Expose a missing header's contract and its complete consumer uses.

    A missing-include diagnostic is attached to the include directive, but
    repairing only that line is not enough evidence for an API migration.  For
    the exact donor ``File`` root selected by the diagnostic graph, retain the
    complete donor header as read-only contract evidence and every old-tree
    structural entity that uses one of the header's declared identifiers.
    This keeps the evidence root-bounded while showing the whole consumer
    function/initializer that an adaptation can affect.
    """

    root = dict(
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    root_tree = str(root.get("tree") or "")
    root_kind = str(root.get("entity_kind") or root.get("kind") or "")
    relative_path = normalize_relpath(
        str(root.get("file") or root.get("path") or root.get("scope") or "")
    )
    mappings = [
        mapping
        for mapping in work_unit.get("diagnostic_mappings") or []
        if isinstance(mapping, dict)
    ]
    if not mappings:
        graph = work_unit.get("diagnostic_graph")
        if isinstance(graph, dict):
            mappings = [
                mapping
                for mapping in graph.get("diagnostic_mappings") or []
                if isinstance(mapping, dict)
            ]
    if (
        not root_tree.startswith("donor")
        or root_kind != "File"
        or not relative_path.endswith(".h")
        or spec.resolve_old(relative_path).exists()
        or not any(
            str(mapping.get("status") or "")
            == "mapped_unresolved_include"
            for mapping in mappings
        )
    ):
        return [], {}

    donor_root = next(
        (
            Path(candidate)
            for candidate in spec.donor_root_paths
            if (Path(candidate) / relative_path).is_file()
        ),
        None,
    )
    if donor_root is None:
        return [], {}

    donor_report = parse_structural_entity_file(
        root=donor_root,
        relative_path=relative_path,
        tree="donor_0",
    )
    header_entities = [
        entity
        for entity in donor_report.get("entities") or []
        if isinstance(entity, dict)
        and str(entity.get("symbol") or entity.get("name") or "")
    ]
    header_symbols = {
        str(entity.get("symbol") or entity.get("name") or "")
        for entity in header_entities
    }

    consumers_by_file: dict[str, list[dict[str, Any]]] = {}
    referenced_symbols: set[str] = set()
    for consumer_file in allowed_files:
        consumer_report = parse_structural_entity_file(
            root=Path(spec.old_root_path),
            relative_path=consumer_file,
            tree="old",
        )
        for entity in consumer_report.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            matched_symbols = sorted(
                header_symbols
                & {
                    str(symbol)
                    for symbol in entity.get("dependency_symbols") or []
                    if str(symbol)
                }
            )
            if not matched_symbols:
                continue
            extent = (
                dict(entity.get("extent") or {})
                if isinstance(entity.get("extent"), dict)
                else {}
            )
            line_start = int(extent.get("start_line") or 0)
            line_end = int(extent.get("end_line") or line_start)
            if line_start <= 0:
                continue
            referenced_symbols.update(matched_symbols)
            record = _extent_source_record(
                spec,
                tree="old",
                relative_path=consumer_file,
                role="allowed_edit_missing_include_consumer_contract",
                edit_permission="editable",
                reason=(
                    "complete consumer entity using contracts from the "
                    f"missing header {relative_path}: "
                    + ", ".join(matched_symbols)
                ),
                line_start=line_start,
                line_end=line_end,
                symbol=_text(entity.get("symbol") or entity.get("name")),
                provider_kind=_text(
                    entity.get("kind") or entity.get("entity_kind")
                ),
                source_kind="old_missing_include_consumer_entity",
                boundary_context_lines=3,
            )
            record["matched_header_symbols"] = matched_symbols
            record["structural_entity_id"] = entity.get("id")
            consumers_by_file.setdefault(consumer_file, []).append(record)

    line_count = len(_read_lines(donor_root / relative_path))
    donor_record = _extent_source_record(
        spec,
        tree="donor",
        relative_path=relative_path,
        role="donor_missing_include_contract",
        edit_permission="read_only",
        reason=(
            "complete contract owner for the graph-selected missing header; "
            + (
                "consumer references: " + ", ".join(sorted(referenced_symbols))
                if referenced_symbols
                else (
                    "no direct declared identifier from this header is used "
                    "by the graph-authorized consumer entities"
                )
            )
        ),
        line_start=1,
        line_end=max(1, line_count),
        symbol=relative_path,
        provider_kind="File",
        source_kind="donor_missing_include_contract_file",
    )
    donor_record.update(
        {
            "declared_symbols": sorted(header_symbols),
            "referenced_symbols": sorted(referenced_symbols),
            "consumer_entity_count": sum(
                len(records) for records in consumers_by_file.values()
            ),
        }
    )
    return [donor_record], consumers_by_file


def build_repair_source_pack(
    spec: CaseSpec,
    *,
    work_unit: dict[str, Any],
    allowed_fileset: list[str],
    read_only_evidence_files: list[str],
) -> dict[str, Any]:
    allowed = [
        path
        for path in _ordered_relpaths(spec, allowed_fileset)
        if classify_edit_permission(spec, path) == "editable"
    ]
    allowed_set = set(allowed)
    graph_nodes = _code_graph_nodes(work_unit)
    graph_old_files = {
        normalize_relpath(
            str(
                node.get("file")
                or node.get("path")
                or node.get("source_file")
                or ""
            )
        )
        for node in graph_nodes
        if str(node.get("tree") or "") == "old"
    }
    graph_donor_files = {
        normalize_relpath(
            str(
                node.get("file")
                or node.get("path")
                or node.get("source_file")
                or ""
            )
        )
        for node in graph_nodes
        if str(node.get("tree") or "").startswith("donor")
    }
    graph_old_files.discard("")
    graph_donor_files.discard("")
    provider_resolutions = [
        (resolution, _provider_records_from_resolution(resolution))
        for resolution in _provider_resolutions(work_unit)
    ]
    old_provider_files: list[str] = []
    for _resolution, providers in provider_resolutions:
        for provider in providers:
            if _text(provider.get("tree")) not in {"old", "target"}:
                continue
            provider_path = _normalize_old_relative(spec, provider.get("file"))
            if provider_path not in allowed_set:
                old_provider_files.append(provider_path)
    read_only = [
        path
        for path in _ordered_relpaths(
            spec,
            [
                *read_only_evidence_files,
                *_owner_read_only_files(spec, work_unit),
                *old_provider_files,
                *graph_old_files,
            ],
        )
        if path not in allowed_set
    ]

    allowed_records: list[dict[str, Any]] = []
    donor_records: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    seen_allowed: set[str] = set()
    seen_donor: set[str] = set()
    seen_evidence: set[str] = set()
    missing_graph_extent_files: list[str] = []
    structural_alignment_blockers: list[dict[str, Any]] = []

    (
        missing_include_donor_records,
        missing_include_consumers_by_file,
    ) = _missing_include_contract_source_records(
        spec,
        work_unit=work_unit,
        allowed_files=allowed,
    )
    for record in missing_include_donor_records:
        _add_unique_record(donor_records, record, seen_donor)

    for record in _external_provider_contract_source_records(
        spec,
        work_unit=work_unit,
    ):
        _add_unique_record(donor_records, record, seen_donor)

    diagnostic_lines = _diagnostic_line_map(spec, work_unit)
    contract_owner_files = _contract_owner_files(work_unit)

    for relative_path in allowed:
        before_count = len(allowed_records)
        import_records = _module_import_relation_source_records(
            spec, work_unit, relative_path=relative_path,
        )
        for record in import_records:
            _add_unique_record(allowed_records, record, seen_allowed)
            if record.get("provider_kind") != "MODULE_IMPORT_NS":
                continue
            definition = _included_macro_contract_source_record(
                spec,
                consumer_file=relative_path,
                macro_name=str(record["provider_kind"]),
            )
            if definition and definition["file"] not in allowed_set:
                _add_unique_record(evidence_records, definition, seen_evidence)
        for record in missing_include_consumers_by_file.get(
            relative_path, []
        ):
            _add_unique_record(allowed_records, record, seen_allowed)
        for record in _include_relation_source_records(
            spec,
            work_unit=work_unit,
            relative_path=relative_path,
        ):
            _add_unique_record(allowed_records, record, seen_allowed)
        for record in _repair_owner_source_records(
            spec,
            work_unit=work_unit,
            relative_path=relative_path,
        ):
            _add_unique_record(allowed_records, record, seen_allowed)
        for record in _materializable_provider_neighborhood_source_records(
            spec, work_unit=work_unit, relative_path=relative_path,
        ):
            _add_unique_record(evidence_records, record, seen_evidence)
        if relative_path in contract_owner_files:
            owner_records, owner_blockers = _owner_alignment_source_records(
                spec,
                work_unit=work_unit,
                relative_path=relative_path,
                tree="old",
            )
            structural_alignment_blockers.extend(owner_blockers)
            for record in owner_records:
                if record.get("edit_permission") == "read_only":
                    _add_unique_record(
                        evidence_records, record, seen_evidence
                    )
                else:
                    _add_unique_record(
                        allowed_records, record, seen_allowed
                    )
        for record in _code_graph_enclosing_container_source_records(
            spec,
            work_unit=work_unit,
            tree="old",
            relative_path=relative_path,
            edit_permission="editable",
        ):
            _add_unique_record(
                allowed_records,
                record,
                seen_allowed,
            )
        for record in _code_graph_source_records(
            spec,
            work_unit=work_unit,
            tree="old",
            relative_path=relative_path,
            role="allowed_edit_code_graph_context",
            edit_permission="editable",
            reason="current old-tree source at an exact CodeIndex entity extent; diffs may touch this file",
            source_kind="old_code_graph_entity_extent",
        ):
            _add_unique_record(allowed_records, record, seen_allowed)

        for resolution, providers in provider_resolutions:
            symbol = _text(resolution.get("symbol"))
            provider_kind = ""
            for provider in providers:
                provider_path = _normalize_old_relative(spec, provider.get("file"))
                if provider_path != relative_path:
                    continue
                tree = _text(provider.get("tree"))
                provider_kind = _text(provider.get("entity_kind") or provider.get("kind"))
                line_start, line_end = _entity_extent_lines(provider)
                if line_start <= 0:
                    continue
                if tree in {"old", "target"}:
                    _add_unique_record(
                        allowed_records,
                        _extent_source_record(
                            spec,
                            tree="old",
                            relative_path=relative_path,
                            role="allowed_edit_provider_context",
                            edit_permission="editable",
                            reason=f"current old-tree provider context for {provider_kind or 'provider'} {symbol or provider.get('symbol') or ''}".strip(),
                            line_start=line_start,
                            line_end=line_end,
                            symbol=symbol or _text(provider.get("symbol")),
                            provider_kind=provider_kind or _text(provider.get("kind")),
                            source_kind="old_provider_semantic_extent",
                        ),
                        seen_allowed,
                    )
        if relative_path in diagnostic_lines:
            _add_unique_record(
                allowed_records,
                _diagnostic_source_record(
                    spec,
                    relative_path=relative_path,
                    line_numbers=diagnostic_lines[relative_path],
                    role="allowed_edit_diagnostic_line",
                    edit_permission="editable",
                    reason="exact old-tree source lines reported by the compiler",
                ),
                seen_allowed,
            )
        if len(allowed_records) == before_count:
            exact_absent_declaration_owner = any(
                isinstance(owner, dict)
                and bool(owner.get("allowed_as_patch_target"))
                and bool(owner.get("exact_absent_declaration_insertion_allowed"))
                and _normalize_old_relative(spec, owner.get("owner_file"))
                == relative_path
                for owner in work_unit.get("repair_owners") or []
            )
            if exact_absent_declaration_owner:
                current_lines = _read_lines(spec.resolve_old(relative_path))
                if current_lines:
                    _add_unique_record(
                        allowed_records,
                        _extent_source_record(
                            spec,
                            tree="old",
                            relative_path=relative_path,
                            role="allowed_edit_absent_declaration_context",
                            edit_permission="editable",
                            reason=(
                                "current target file context for an exact "
                                "absent declaration insertion"
                            ),
                            line_start=1,
                            line_end=len(current_lines),
                            source_kind="old_absent_declaration_file_context",
                        ),
                        seen_allowed,
                    )
        if len(allowed_records) == before_count and not any(
            record.get("file") == relative_path
            and record.get("role") == "old_materializable_provider_neighborhood"
            for record in evidence_records
        ):
            missing_graph_extent_files.append(relative_path)

    for resolution, providers in provider_resolutions:
        symbol = _text(resolution.get("symbol"))
        provider_kind = ""
        for provider in providers:
            tree = _text(provider.get("tree"))
            if not tree.startswith("donor"):
                continue
            relative_path = _normalize_old_relative(spec, provider.get("file"))
            if not relative_path:
                continue
            provider_kind = _text(provider.get("entity_kind") or provider.get("kind"))
            line_start, line_end = _entity_extent_lines(provider)
            if line_start <= 0:
                continue
            _add_unique_record(
                donor_records,
                _extent_source_record(
                    spec,
                    tree="donor",
                    relative_path=relative_path,
                    role="donor_provider_reference",
                    edit_permission="read_only",
                    reason=f"provider-use reference for {provider_kind or 'provider'} {symbol or provider.get('symbol') or ''}".strip(),
                    line_start=line_start,
                    line_end=line_end,
                    symbol=symbol or _text(provider.get("symbol")),
                    provider_kind=provider_kind or _text(provider.get("kind")),
                    source_kind="provider_semantic_extent",
                ),
                seen_donor,
            )
        for provider in providers:
            tree = _text(provider.get("tree"))
            if tree not in {"old", "target"}:
                continue
            relative_path = _normalize_old_relative(spec, provider.get("file"))
            if not relative_path or relative_path in allowed_set:
                continue
            provider_kind = _text(provider.get("entity_kind") or provider.get("kind"))
            line_start, line_end = _entity_extent_lines(provider)
            if line_start <= 0:
                continue
            _add_unique_record(
                evidence_records,
                _extent_source_record(
                    spec,
                    tree="old",
                    relative_path=relative_path,
                    role="old_provider_compatibility_constraint",
                    edit_permission="read_only",
                    reason=(
                        f"existing old-tree implementation constrains the compatible declaration for "
                        f"{provider_kind or 'provider'} {symbol or provider.get('symbol') or ''}"
                    ).strip(),
                    line_start=line_start,
                    line_end=line_end,
                    symbol=symbol or _text(provider.get("symbol")),
                    provider_kind=provider_kind or _text(provider.get("kind")),
                    source_kind="old_provider_compatibility_extent",
                ),
                seen_evidence,
            )
    for relative_path in allowed:
        owner_records, owner_blockers = _owner_alignment_source_records(
            spec,
            work_unit=work_unit,
            relative_path=relative_path,
            tree="donor",
        )
        structural_alignment_blockers.extend(owner_blockers)
        for record in owner_records:
            _add_unique_record(donor_records, record, seen_donor)
        if spec.resolve_donor(relative_path).exists():
            for record in (
                _code_graph_enclosing_container_source_records(
                    spec,
                    work_unit=work_unit,
                    tree="donor",
                    relative_path=relative_path,
                    edit_permission="read_only",
                )
            ):
                _add_unique_record(
                    donor_records,
                    record,
                    seen_donor,
                )
    for relative_path in _ordered_relpaths(
        spec,
        [*allowed, *graph_donor_files],
    ):
        for record in _code_graph_source_records(
            spec,
            work_unit=work_unit,
            tree="donor",
            relative_path=relative_path,
            role="donor_counterpart_code_graph_context",
            edit_permission="read_only",
            reason=(
                "donor source at an exact CodeIndex entity extent; use as reference only"
            ),
            source_kind="donor_code_graph_entity_extent",
        ):
            _add_unique_record(donor_records, record, seen_donor)
    for relative_path in read_only:
        for record in _materializable_provider_neighborhood_source_records(
            spec,
            work_unit=work_unit,
            relative_path=relative_path,
        ):
            _add_unique_record(evidence_records, record, seen_evidence)
        for record in _code_graph_source_records(
            spec,
            work_unit=work_unit,
            tree="old",
            relative_path=relative_path,
            role="read_only_code_graph_context",
            edit_permission="read_only",
            reason=(
                "current old-tree source at an exact CodeIndex entity extent; "
                "read-only evidence, not an automatic patch target"
            ),
            source_kind="read_only_code_graph_entity_extent",
        ):
            _add_unique_record(evidence_records, record, seen_evidence)
        line_numbers = diagnostic_lines.get(relative_path) or []
        if line_numbers:
            _add_unique_record(
                evidence_records,
                _diagnostic_source_record(
                    spec,
                    relative_path=relative_path,
                    line_numbers=line_numbers,
                    role=_diagnostic_role_for_file(relative_path, work_unit),
                    reason="exact old-tree source lines reported by the compiler; read-only evidence",
                ),
                seen_evidence,
            )

    unresolved_owners = [
        owner
        for owner in work_unit.get("repair_owners") or []
        if isinstance(owner, dict)
        and not bool(owner.get("allowed_as_patch_target"))
    ]
    return {
        "schema": "repair_source_pack_v5",
        "policy": {
            "purpose": "expose exact obligation-bound graph ranges while keeping patch permissions explicit",
            "allowed_edit_sources": "exact old-tree CodeIndex extents or compiler diagnostic lines; only these files may appear in the diff",
            "donor_reference_sources": "exact donor CodeIndex/provider extents; reference-only",
            "read_only_evidence_sources": "exact old-tree graph extents or diagnostic lines; never automatic patch targets",
            "evidence_retention": (
                "all records incident to provider, repair-locus, unresolved "
                "demand-anchor, diagnostic, or explicit dependency identities "
                "are retained; root-set alternatives remain inspectable "
                "navigation records"
            ),
        },
        "summary": {
            "allowed_edit_source_count": len(allowed_records),
            "donor_reference_source_count": len(donor_records),
            "read_only_evidence_source_count": len(evidence_records),
            "unresolved_owner_count": len(unresolved_owners),
            "missing_graph_extent_file_count": len(set(missing_graph_extent_files)),
            "missing_include_contract_count": len(
                missing_include_donor_records
            ),
            "missing_include_consumer_entity_count": sum(
                len(records)
                for records in missing_include_consumers_by_file.values()
            ),
        },
        "allowed_edit_files": allowed,
        "read_only_evidence_files": read_only,
        "allowed_edit_sources": allowed_records,
        "donor_reference_sources": donor_records,
        "read_only_evidence_sources": evidence_records,
        "localization_blockers": [
            {
                "owner_kind": owner.get("owner_kind"),
                "owner_file": owner.get("owner_file"),
                "reason": owner.get("reason"),
            }
            for owner in unresolved_owners
        ]
        + [
            {
                "owner_kind": "code_index_extent_missing",
                "owner_file": relative_path,
                "reason": "graph-selected file has no exact diagnostic or CodeIndex entity extent",
            }
            for relative_path in sorted(set(missing_graph_extent_files))
        ]
        + structural_alignment_blockers,
    }
