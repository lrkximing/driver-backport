#!/usr/bin/env python3
from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Iterable

from driver_backport.core.common import normalize_relpath, sha256_file, stable_id


SOURCE_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp", ".S", ".s"}
BUILD_NAMES = {"Makefile", "Kbuild", "Kconfig"}
ATOMIC_MERGE_EDGE_TYPES = {
    "same_semantic_decision",
    "same_atomic_edit",
    "overlapping_write_region",
}


def _tracked(path: Path) -> bool:
    return path.suffix in SOURCE_SUFFIXES or path.name in BUILD_NAMES


def _relative_files(root: Path, driver_root: str) -> dict[str, Path]:
    base = (root.resolve() / normalize_relpath(driver_root)).resolve()
    if not base.is_dir():
        return {}
    return {
        path.resolve().relative_to(root.resolve()).as_posix(): path.resolve()
        for path in base.rglob("*")
        if path.is_file() and _tracked(path)
    }


def build_file_change_manifest(
    *,
    old_root: Path,
    donor_root: Path,
    driver_root: str,
    originally_missing: Iterable[object] = (),
) -> dict[str, Any]:
    old_files = _relative_files(old_root, driver_root)
    donor_files = _relative_files(donor_root, driver_root)
    copied = {normalize_relpath(str(item or "")) for item in originally_missing if str(item or "").strip()}
    records: list[dict[str, Any]] = []
    for relative_path in sorted(set(old_files) | set(donor_files) | copied):
        old_path = old_files.get(relative_path)
        donor_path = donor_files.get(relative_path)
        if relative_path in copied and donor_path is not None:
            presence = "donor_only_transplanted"
        elif old_path is None:
            presence = "donor_only"
        elif donor_path is None:
            presence = "old_only"
        else:
            presence = (
                "shared_identical"
                if sha256_file(old_path) == sha256_file(donor_path)
                else "shared_changed"
            )
        records.append(
            {
                "file": relative_path,
                "presence": presence,
                "old_exists": old_path is not None,
                "donor_exists": donor_path is not None,
                "old_sha256": sha256_file(old_path) if old_path is not None else None,
                "donor_sha256": sha256_file(donor_path) if donor_path is not None else None,
            }
        )
    counts: dict[str, int] = {}
    for record in records:
        key = str(record["presence"])
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": "cross_version_file_change_manifest_v1",
        "driver_root": normalize_relpath(driver_root),
        "files": records,
        "summary": {
            "file_count": len(records),
            "presence_counts": counts,
        },
    }


def structured_line_hunks(
    *,
    old_text: str,
    donor_text: str,
    file: str,
) -> list[dict[str, Any]]:
    old_lines = old_text.splitlines()
    donor_lines = donor_text.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=donor_lines, autojunk=False)
    hunks: list[dict[str, Any]] = []
    for tag, old_start, old_end, donor_start, donor_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunk_id = stable_id(
            "hunk",
            normalize_relpath(file),
            tag,
            str(old_start),
            str(old_end),
            str(donor_start),
            str(donor_end),
        )
        hunks.append(
            {
                "id": hunk_id,
                "file": normalize_relpath(file),
                "operation": tag,
                "old_start_line": old_start + 1,
                "old_end_line": old_end,
                "donor_start_line": donor_start + 1,
                "donor_end_line": donor_end,
                "old_lines": old_lines[old_start:old_end],
                "donor_lines": donor_lines[donor_start:donor_end],
            }
        )
    return hunks


def hunks_overlapping_extent(
    hunks: Iterable[dict[str, Any]],
    *,
    tree: str,
    extent: dict[str, Any] | None,
) -> list[str]:
    extent = dict(extent or {})
    start = int(extent.get("start_line") or 0)
    end = int(extent.get("end_line") or start)
    if start <= 0:
        return []
    prefix = "donor" if tree.startswith("donor") else "old"
    result: list[str] = []
    for hunk in hunks:
        hunk_start = int(hunk.get(f"{prefix}_start_line") or 0)
        hunk_end = int(hunk.get(f"{prefix}_end_line") or hunk_start)
        if hunk_start <= end and start <= max(hunk_start, hunk_end):
            result.append(str(hunk.get("id") or ""))
    return [item for item in result if item]


def build_change_atoms(
    *,
    contract_identity: str,
    comparison: dict[str, Any],
    old_entity: dict[str, Any] | None,
    donor_entity: dict[str, Any] | None,
    raw_hunk_ids: Iterable[object] = (),
    contract_evidence_id: str = "",
    semantic_requirement: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    dimensions = (
        "consumer_compile_binding",
        "name",
        "signature",
        "implementation",
        "interface",
        "declaration",
        "structural_source",
        "export",
        "module_boundary",
        "integration_reference",
    )
    decision_dimensions = {
        str(item)
        for item in [
            *list(
                (semantic_requirement or {}).get(
                    "required_dimensions"
                )
                or []
            ),
            *list(
                (semantic_requirement or {}).get(
                    "evidence_dimensions"
                )
                or []
            ),
        ]
        if str(item)
    }
    atoms: list[dict[str, Any]] = []
    for dimension in dimensions:
        if decision_dimensions and dimension not in decision_dimensions:
            continue
        relation = str(comparison.get(dimension) or "")
        if relation in {
            "",
            "equivalent",
            "satisfied",
            "same_module",
            "not_required",
            "unavailable",
            "donor_contract_missing",
        }:
            continue
        operation = "update"
        if relation in {"missing_in_old", "donor_only", "required_missing"}:
            operation = "insert"
        elif relation in {"missing_in_donor", "old_only"}:
            operation = "delete"
        atoms.append(
            {
                "id": stable_id("change-atom", contract_identity, dimension, relation),
                "contract_evidence_id": contract_evidence_id or None,
                "contract_identity": contract_identity,
                "dimension": dimension,
                "operation": operation,
                "relation": relation,
                "symbol": (donor_entity or old_entity or {}).get("symbol"),
                "entity_kind": (donor_entity or old_entity or {}).get("kind"),
                "semantic_requirement": dict(semantic_requirement or {}),
                "old_entity_id": (old_entity or {}).get("id"),
                "donor_entity_id": (donor_entity or {}).get("id"),
                "old_file": (old_entity or {}).get("file"),
                "donor_file": (donor_entity or {}).get("file"),
                "old_extent": dict((old_entity or {}).get("extent") or {}),
                "donor_extent": dict((donor_entity or {}).get("extent") or {}),
                "raw_hunk_ids": (
                    []
                    if dimension == "consumer_compile_binding"
                    else sorted(
                        {
                            str(item)
                            for item in raw_hunk_ids
                            if str(item)
                        }
                    )
                ),
            }
        )
    return atoms


def _compact_entity(entity: object) -> dict[str, Any]:
    if not isinstance(entity, dict):
        return {}
    return {
        key: entity.get(key)
        for key in (
            "id",
            "tree",
            "file",
            "symbol",
            "kind",
            "cursor_kind",
            "contract_key",
            "is_definition",
            "extent",
            "container",
        )
        if entity.get(key) not in (None, "", {})
    }


def _demand_anchor(
    reference: dict[str, Any],
    *,
    contract_evidence_id: str,
    contract_identity: str,
) -> dict[str, Any]:
    reference_id = str(reference.get("id") or "")
    anchor_id = stable_id(
        "initial-demand",
        contract_identity,
        reference_id,
        str(reference.get("source_file") or ""),
        str(reference.get("line") or 0),
        str(reference.get("column") or 0),
        str(reference.get("use_kind") or ""),
    )
    return {
        "id": anchor_id,
        "kind": "DemandAnchor",
        "contract_evidence_id": contract_evidence_id,
        "contract_identity": contract_identity,
        "reference_id": reference_id or None,
        "source_file": normalize_relpath(
            str(reference.get("source_file") or "")
        ),
        "line": int(reference.get("line") or 0),
        "column": int(reference.get("column") or 0),
        "extent": dict(reference.get("extent") or {}),
        "symbol": reference.get("symbol"),
        "entity_kind": reference.get("kind"),
        "use_kind": reference.get("use_kind"),
        "cursor_kind": reference.get("cursor_kind"),
        "dependency_scope": reference.get("dependency_scope"),
        "contract_key": reference.get("contract_key"),
        "owner_entity": _compact_entity(reference.get("owner_entity")),
        "referenced_entity": _compact_entity(
            reference.get("referenced_entity")
        ),
    }


def _repair_loci(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_id = str(evidence.get("id") or "")
    contract_identity = str(evidence.get("contract_identity") or "")
    integration_hunks = [
        dict(item)
        for item in evidence.get("integration_hunks") or []
        if isinstance(item, dict)
    ]
    if integration_hunks:
        return [
            {
                "id": stable_id(
                    "initial-repair-locus",
                    contract_identity,
                    str(hunk.get("id") or ""),
                ),
                "kind": "RepairLocus",
                "contract_evidence_id": evidence_id,
                "contract_identity": contract_identity,
                "file": normalize_relpath(str(hunk.get("file") or "")),
                "symbol": evidence.get("symbol"),
                "entity_kind": "integration_hunk",
                "target_presence": "shared_file",
                "old_extent": {
                    "start_line": int(hunk.get("old_start_line") or 0),
                    "end_line": int(hunk.get("old_end_line") or 0),
                },
                "donor_extent": {
                    "start_line": int(hunk.get("donor_start_line") or 0),
                    "end_line": int(hunk.get("donor_end_line") or 0),
                },
                "raw_hunk_ids": [
                    str(hunk.get("id") or "")
                ] if hunk.get("id") else [],
            }
            for hunk in integration_hunks
            if normalize_relpath(str(hunk.get("file") or ""))
        ]

    old_entity = (
        evidence.get("old_entity")
        if isinstance(evidence.get("old_entity"), dict)
        else {}
    )
    donor_entity = (
        evidence.get("donor_entity")
        if isinstance(evidence.get("donor_entity"), dict)
        else {}
    )
    entity = old_entity or donor_entity
    relative_path = normalize_relpath(str(entity.get("file") or ""))
    if not relative_path:
        return []
    contract_key = str(
        entity.get("contract_key")
        or entity.get("key")
        or evidence.get("symbol")
        or ""
    )
    return [
        {
            "id": stable_id(
                "initial-repair-locus",
                contract_identity,
                relative_path,
                contract_key,
            ),
            "kind": "RepairLocus",
            "contract_evidence_id": evidence_id,
            "contract_identity": contract_identity,
            "file": relative_path,
            "symbol": entity.get("symbol") or evidence.get("symbol"),
            "entity_kind": entity.get("kind"),
            "contract_key": contract_key,
            "container": dict(entity.get("container") or {}),
            "target_presence": (
                "existing_old_entity"
                if old_entity
                else "missing_old_entity"
            ),
            "old_entity_id": old_entity.get("id"),
            "donor_entity_id": donor_entity.get("id"),
            "old_extent": dict(old_entity.get("extent") or {}),
            "donor_extent": dict(donor_entity.get("extent") or {}),
            "raw_hunk_ids": sorted(
                {
                    str(item)
                    for item in evidence.get("raw_hunk_ids") or []
                    if str(item)
                }
            ),
        }
    ]


def _extent_overlap(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_start = int(left.get("start_line") or 0)
    left_end = int(left.get("end_line") or left_start)
    right_start = int(right.get("start_line") or 0)
    right_end = int(right.get("end_line") or right_start)
    return bool(
        left_start > 0
        and right_start > 0
        and left_start <= max(right_start, right_end)
        and right_start <= max(left_start, left_end)
    )


def build_initial_difference_graph(
    *,
    contract_evidence: Iterable[dict[str, Any]],
    change_atoms: Iterable[dict[str, Any]],
    raw_hunks: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    evidence_records = [
        dict(item)
        for item in contract_evidence
        if isinstance(item, dict) and item.get("id")
    ]
    atoms = [
        dict(item)
        for item in change_atoms
        if isinstance(item, dict) and item.get("id")
    ]
    evidence_by_id = {
        str(item["id"]): item for item in evidence_records
    }
    evidence_by_identity = {
        str(item.get("contract_identity") or ""): item
        for item in evidence_records
        if item.get("contract_identity")
    }
    atoms_by_evidence: dict[str, list[dict[str, Any]]] = {}
    for atom in atoms:
        evidence_id = str(atom.get("contract_evidence_id") or "")
        if not evidence_id:
            identity = str(atom.get("contract_identity") or "")
            evidence_id = str(
                (evidence_by_identity.get(identity) or {}).get("id") or ""
            )
            if evidence_id:
                atom["contract_evidence_id"] = evidence_id
        atoms_by_evidence.setdefault(evidence_id, []).append(atom)
    atom_ids_by_evidence = {
        evidence_id: sorted(
            str(item.get("id") or "")
            for item in evidence_atoms
            if item.get("id")
        )
        for evidence_id, evidence_atoms in atoms_by_evidence.items()
    }

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node: dict[str, Any]) -> None:
        node_id = str(node.get("id") or "")
        if not node_id or node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append(node)

    def add_edge(source: str, target: str, edge_type: str) -> None:
        key = (source, target, edge_type)
        if not source or not target or key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"from": source, "to": target, "type": edge_type})

    loci_by_evidence: dict[str, list[dict[str, Any]]] = {}
    anchors_by_evidence: dict[str, list[dict[str, Any]]] = {}
    for evidence in evidence_records:
        evidence_id = str(evidence.get("id") or "")
        contract_identity = str(
            evidence.get("contract_identity") or ""
        )
        add_node(
            {
                "id": evidence_id,
                "kind": "ContractEvidence",
                "contract_identity": contract_identity,
                "symbol": evidence.get("symbol"),
                "entity_kind": (
                    (evidence.get("donor_entity") or {}).get("kind")
                    if isinstance(evidence.get("donor_entity"), dict)
                    else None
                ),
                "resolution": evidence.get("resolution"),
                "semantic_requirement": dict(
                    evidence.get("semantic_requirement") or {}
                ),
            }
        )

        references = [
            dict(item)
            for item in evidence.get("consumer_references") or []
            if isinstance(item, dict)
        ]
        if not references and isinstance(
            evidence.get("consumer_reference"), dict
        ):
            references = [dict(evidence["consumer_reference"])]
        anchors = [
            _demand_anchor(
                reference,
                contract_evidence_id=evidence_id,
                contract_identity=contract_identity,
            )
            for reference in references
        ]
        anchors_by_evidence[evidence_id] = anchors
        for anchor in anchors:
            add_node(anchor)
            add_edge(
                str(anchor["id"]),
                evidence_id,
                "demands_contract",
            )

        loci = _repair_loci(evidence)
        loci_by_evidence[evidence_id] = loci
        for locus in loci:
            add_node(locus)
            add_edge(evidence_id, str(locus["id"]), "located_at")

    for atom in atoms:
        atom_id = str(atom.get("id") or "")
        evidence_id = str(atom.get("contract_evidence_id") or "")
        add_node({"kind": "DifferenceAtom", **atom})
        if evidence_id:
            add_edge(evidence_id, atom_id, "described_by")
            for anchor in anchors_by_evidence.get(evidence_id, []):
                add_edge(str(anchor["id"]), atom_id, "requires_delta")
            for locus in loci_by_evidence.get(evidence_id, []):
                add_edge(atom_id, str(locus["id"]), "repaired_at")

    for evidence_id, ordered_atom_ids in atom_ids_by_evidence.items():
        for left, right in zip(ordered_atom_ids, ordered_atom_ids[1:]):
            add_edge(left, right, "same_semantic_decision")
        evidence = evidence_by_id.get(evidence_id) or {}
        dependency_ids = [
            str(item)
            for item in evidence.get("depends_on_evidence_ids") or []
            if str(item)
        ]
        if not ordered_atom_ids:
            continue
        for dependency_id in dependency_ids:
            dependency_atoms = atom_ids_by_evidence.get(dependency_id) or []
            if dependency_atoms:
                add_edge(ordered_atom_ids[0], dependency_atoms[0], "requires")

    hunk_by_id = {
        str(item.get("id") or ""): dict(item)
        for item in raw_hunks
        if isinstance(item, dict) and item.get("id")
    }
    atomic_hunk_groups: dict[
        tuple[str, str, str, str],
        list[str],
    ] = {}
    for evidence in evidence_records:
        if str(evidence.get("resolution") or "") != "requires_llm":
            continue
        donor_entity = (
            evidence.get("donor_entity")
            if isinstance(evidence.get("donor_entity"), dict)
            else {}
        )
        container = (
            donor_entity.get("container")
            if isinstance(donor_entity.get("container"), dict)
            else {}
        )
        relative_path = normalize_relpath(str(donor_entity.get("file") or ""))
        container_kind = str(container.get("container_kind") or "")
        container_name = str(container.get("container_name") or "")
        if not relative_path or not container_kind or not container_name:
            continue
        for hunk_id in evidence.get("raw_hunk_ids") or []:
            hunk = hunk_by_id.get(str(hunk_id))
            if (
                hunk is None
                or normalize_relpath(str(hunk.get("file") or ""))
                != relative_path
            ):
                continue
            atomic_hunk_groups.setdefault(
                (
                    relative_path,
                    container_kind,
                    container_name,
                    str(hunk_id),
                ),
                [],
            ).extend(
                str(item.get("id") or "")
                for item in atoms_by_evidence.get(
                    str(evidence.get("id") or ""),
                    [],
                )
                if item.get("id")
            )
    for atom_ids in atomic_hunk_groups.values():
        ordered = sorted(set(atom_ids))
        for left, right in zip(ordered, ordered[1:]):
            add_edge(left, right, "same_atomic_edit")

    locus_items = [
        (evidence_id, locus)
        for evidence_id, loci in loci_by_evidence.items()
        for locus in loci
        if str(locus.get("target_presence") or "") == "existing_old_entity"
    ]
    for index, (left_evidence_id, left_locus) in enumerate(locus_items):
        left_file = normalize_relpath(str(left_locus.get("file") or ""))
        for right_evidence_id, right_locus in locus_items[index + 1 :]:
            if (
                left_evidence_id == right_evidence_id
                or left_file
                != normalize_relpath(str(right_locus.get("file") or ""))
                or not _extent_overlap(
                    dict(left_locus.get("old_extent") or {}),
                    dict(right_locus.get("old_extent") or {}),
                )
            ):
                continue
            left_atoms = atom_ids_by_evidence.get(left_evidence_id) or []
            right_atoms = atom_ids_by_evidence.get(right_evidence_id) or []
            if left_atoms and right_atoms:
                add_edge(
                    left_atoms[0],
                    right_atoms[0],
                    "overlapping_write_region",
                )

    # A reverse-integration observation and the provider contract it exposes
    # can describe the same donor edit even when the old side has no extent
    # (for example, a newly inserted function).  Merge only when their typed
    # donor extents overlap and they share the exact diff hunk; sharing a file
    # or symbol alone is insufficient.
    donor_locus_items = [
        (evidence_id, locus)
        for evidence_id, loci in loci_by_evidence.items()
        for locus in loci
        if dict(locus.get("donor_extent") or {}).get("start_line")
    ]
    for index, (left_evidence_id, left_locus) in enumerate(
        donor_locus_items
    ):
        left_file = normalize_relpath(str(left_locus.get("file") or ""))
        for right_evidence_id, right_locus in donor_locus_items[index + 1 :]:
            if (
                left_evidence_id == right_evidence_id
                or left_file
                != normalize_relpath(str(right_locus.get("file") or ""))
                or "integration_hunk"
                not in {
                    str(left_locus.get("entity_kind") or ""),
                    str(right_locus.get("entity_kind") or ""),
                }
                or not _extent_overlap(
                    dict(left_locus.get("donor_extent") or {}),
                    dict(right_locus.get("donor_extent") or {}),
                )
                or not (
                    set(left_locus.get("raw_hunk_ids") or [])
                    & set(right_locus.get("raw_hunk_ids") or [])
                )
            ):
                continue
            left_atoms = atom_ids_by_evidence.get(left_evidence_id) or []
            right_atoms = atom_ids_by_evidence.get(right_evidence_id) or []
            if left_atoms and right_atoms:
                add_edge(
                    left_atoms[0],
                    right_atoms[0],
                    "overlapping_write_region",
                )

    edge_counts: dict[str, int] = {}
    for edge in edges:
        edge_type = str(edge.get("type") or "")
        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
    return {
        "schema": "initial_difference_graph_v3",
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "difference_atom_count": len(atoms),
            "demand_anchor_count": sum(
                len(items) for items in anchors_by_evidence.values()
            ),
            "repair_locus_count": sum(
                len(items) for items in loci_by_evidence.values()
            ),
            "contract_evidence_count": len(evidence_records),
            "atomic_merge_edge_count": sum(
                count
                for edge_type, count in edge_counts.items()
                if edge_type in ATOMIC_MERGE_EDGE_TYPES
            ),
            "dependency_edge_count": edge_counts.get("requires", 0),
            "edge_type_counts": dict(sorted(edge_counts.items())),
        },
    }
