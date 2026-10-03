#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from driver_backport.analysis.change_graph import (
    ATOMIC_MERGE_EDGE_TYPES,
    build_initial_difference_graph,
)
from driver_backport.analysis.code_index import (
    CodeIndex,
    parse_structural_entity_file,
)
from driver_backport.analysis.conditional_contract import (
    conditional_variant_family,
)
from driver_backport.analysis.entity_resolution import (
    resolve_entity_in_case_tree,
)
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths as _ordered_unique,
    stable_id,
    utc_now,
)
from driver_backport.planning.execution_phase import (
    EXECUTION_PHASE_INITIAL_COMPATIBILITY,
)
from driver_backport.planning.dependency_schedule import dependency_order


INITIAL_CONSUMER_EVIDENCE_LIMIT = 8
INITIAL_COMPATIBILITY_PLAN_SCHEMA = "initial_compatibility_plan_v4"
INITIAL_COMPATIBILITY_FRONTIER_REVISION = (
    "canonical_contract_transaction_units_v21"
)


def _excerpt(
    root: Path,
    *,
    relative_path: str,
    extent: dict[str, Any] | None,
    context: int = 2,
) -> tuple[str, int, int, bool]:
    path = (root / normalize_relpath(relative_path)).resolve()
    if not path.is_file():
        return "", 0, 0, False
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    extent = dict(extent or {})
    start = max(1, int(extent.get("start_line") or 1))
    end = max(start, int(extent.get("end_line") or start))
    selected_start = max(1, start - context)
    selected_end = min(len(lines), end + context)
    return "\n".join(lines[selected_start - 1 : selected_end]), selected_start, selected_end, True


def _source_record(
    root: Path,
    *,
    tree: str,
    entity: dict[str, Any] | None,
    role: str,
    reason: str,
) -> dict[str, Any] | None:
    if not entity:
        return None
    relative_path = normalize_relpath(str(entity.get("file") or ""))
    if not relative_path:
        return None
    text, start, end, exists = _excerpt(
        root,
        relative_path=relative_path,
        extent=entity.get("extent"),
    )
    return {
        "tree": tree,
        "file": relative_path,
        "role": role,
        "source_kind": "contract_entity",
        "symbol": entity.get("symbol"),
        "line_start": start or None,
        "line_end": end or None,
        "exists": exists,
        "reason": reason,
        "text": text,
    }


def _line_source_record(
    root: Path,
    *,
    tree: str,
    relative_path: str,
    line: int,
    role: str,
    reason: str,
) -> dict[str, Any]:
    text, start, end, exists = _excerpt(
        root,
        relative_path=relative_path,
        extent={"start_line": line, "end_line": line},
        context=2,
    )
    return {
        "tree": tree,
        "file": normalize_relpath(relative_path),
        "role": role,
        "source_kind": "contract_line",
        "line_start": start or None,
        "line_end": end or None,
        "exists": exists,
        "reason": reason,
        "text": text,
    }


def _range_source_record(
    root: Path,
    *,
    tree: str,
    relative_path: str,
    start_line: int,
    end_line: int,
    role: str,
    reason: str,
) -> dict[str, Any]:
    text, start, end, exists = _excerpt(
        root,
        relative_path=relative_path,
        extent={
            "start_line": max(1, int(start_line or 1)),
            "end_line": max(1, int(end_line or start_line or 1)),
        },
        context=3,
    )
    return {
        "tree": tree,
        "file": normalize_relpath(relative_path),
        "role": role,
        "source_kind": "contract_diff_hunk",
        "line_start": start or None,
        "line_end": end or None,
        "exists": exists,
        "reason": reason,
        "text": text,
    }


def _conditional_variant_source_records(
    spec: CaseSpec,
    donor_entity: dict[str, Any] | None,
    *,
    file_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not donor_entity:
        return None
    relative_path = normalize_relpath(str(donor_entity.get("file") or ""))
    contract = str(
        donor_entity.get("contract_key") or donor_entity.get("key") or ""
    )
    if not relative_path or not contract:
        return None
    old_path = spec.resolve_old(relative_path)
    donor_path = spec.resolve_donor(relative_path)
    if not old_path.is_file() or not donor_path.is_file():
        return None
    cache = file_cache if file_cache is not None else {}
    cached = cache.get(relative_path)
    if cached is None:
        try:
            cached = {
                "old_report": parse_structural_entity_file(
                    root=spec.old_root_path,
                    relative_path=relative_path,
                    tree="old",
                ),
                "donor_report": parse_structural_entity_file(
                    root=spec.primary_donor_root,
                    relative_path=relative_path,
                    tree="donor_0",
                ),
                "old_lines": old_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines(),
                "donor_lines": donor_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines(),
            }
        except (AttributeError, OSError, RuntimeError, ValueError):
            return None
        cache[relative_path] = cached
    old_report = dict(cached.get("old_report") or {})
    donor_report = dict(cached.get("donor_report") or {})
    if not old_report.get("parse_ok") or not donor_report.get("parse_ok"):
        return None
    donor_matches = [
        item
        for item in donor_report.get("entities") or []
        if isinstance(item, dict)
        and str(item.get("contract_key") or item.get("key") or "")
        == contract
    ]
    old_lines = list(cached.get("old_lines") or [])
    donor_lines = list(cached.get("donor_lines") or [])
    family = conditional_variant_family(
        donor_lines=donor_lines,
        old_lines=old_lines,
        target_entities=donor_matches,
    )
    if family is None:
        return None
    reason = (
        "the same contract has mutually exclusive preprocessor variants; "
        "the complete corresponding conditional block is one migration unit"
    )
    old_record = _range_source_record(
        spec.old_root_path,
        tree="old",
        relative_path=relative_path,
        start_line=int(family["old_start_line"]),
        end_line=int(family["old_end_line"]),
        role="current_conditional_contract_variant_set",
        reason=reason,
    )
    old_record.update(
        {
            "source_kind": "old_conditional_contract_variant_set",
            "symbol": donor_entity.get("symbol"),
            "contract_key": contract,
            "conditional_contract": family,
        }
    )
    donor_record = _range_source_record(
        spec.primary_donor_root,
        tree="donor_0",
        relative_path=relative_path,
        start_line=int(family["donor_start_line"]),
        end_line=int(family["donor_end_line"]),
        role="donor_conditional_contract_variant_set",
        reason=reason,
    )
    donor_record.update(
        {
            "source_kind": "donor_conditional_contract_variant_set",
            "symbol": donor_entity.get("symbol"),
            "contract_key": contract,
            "conditional_contract": family,
        }
    )
    return old_record, donor_record


def _candidate_edit_files(delta: dict[str, Any]) -> list[str]:
    files: list[object] = []
    for key in ("old_entity", "donor_entity"):
        entity = delta.get(key)
        if isinstance(entity, dict):
            files.append(entity.get("file"))
    for key in ("old_declarations", "donor_declarations"):
        for entity in delta.get(key) or []:
            if isinstance(entity, dict):
                files.append(entity.get("file"))
    references = list(delta.get("consumer_references") or [])
    if not references and isinstance(delta.get("consumer_reference"), dict):
        references = [delta["consumer_reference"]]
    for reference in references:
        if isinstance(reference, dict):
            files.append(reference.get("source_file"))
    for export in delta.get("old_exports") or []:
        if isinstance(export, dict):
            files.append(export.get("file"))
    return _ordered_unique(files)


def _bounded_consumer_reference_evidence(
    references: list[object],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for item in references:
        if not isinstance(item, dict):
            continue
        key = (
            normalize_relpath(str(item.get("source_file") or "")),
            int(item.get("line") or 0),
            int(item.get("column") or 0),
            str(item.get("owner_entity_id") or ""),
            str(item.get("referenced_entity_id") or ""),
            str(item.get("use_kind") or item.get("kind") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= INITIAL_CONSUMER_EVIDENCE_LIMIT:
            break
    return selected


def _contract_evidence(delta: dict[str, Any]) -> dict[str, Any]:
    references = [
        dict(item)
        for item in delta.get("consumer_references") or []
        if isinstance(item, dict)
    ]
    return {
        key: value
        for key, value in {
            "id": delta.get("id"),
            "contract_identity": delta.get("contract_identity"),
            "kind": delta.get("kind"),
            "symbol": delta.get("symbol"),
            "dependency_scope": delta.get("dependency_scope"),
            "semantic_requirement": dict(
                delta.get("semantic_requirement") or {}
            ),
            "comparison": dict(delta.get("comparison") or {}),
            "resolution": delta.get("resolution"),
            "old_entity": (
                dict(delta["old_entity"])
                if isinstance(delta.get("old_entity"), dict)
                else None
            ),
            "donor_entity": (
                dict(delta["donor_entity"])
                if isinstance(delta.get("donor_entity"), dict)
                else None
            ),
            "alignment": {
                key: (delta.get("alignment") or {}).get(key)
                for key in (
                    "status",
                    "reason",
                    "old_candidate_count",
                    "missing_reference_count",
                )
                if (delta.get("alignment") or {}).get(key)
                not in (None, "", [], {})
            },
            "old_declarations": [
                dict(item)
                for item in delta.get("old_declarations") or []
                if isinstance(item, dict)
            ],
            "donor_declarations": [
                dict(item)
                for item in delta.get("donor_declarations") or []
                if isinstance(item, dict)
            ],
            "old_exports": [
                dict(item)
                for item in delta.get("old_exports") or []
                if isinstance(item, dict)
            ],
            "donor_exports": [
                dict(item)
                for item in delta.get("donor_exports") or []
                if isinstance(item, dict)
            ],
            "consumer_references": references,
            "consumer_reference_count": len(references),
            "consumer_modules": list(delta.get("consumer_modules") or []),
            "provider_modules": list(delta.get("provider_modules") or []),
            "target_validation_files": list(
                delta.get("target_validation_files") or []
            ),
            "raw_hunk_ids": list(delta.get("raw_hunk_ids") or []),
            "integration_hunks": [
                dict(item)
                for item in delta.get("integration_hunks") or []
                if isinstance(item, dict)
            ],
            "change_atom_ids": list(delta.get("change_atom_ids") or []),
            "depends_on_evidence_ids": list(
                delta.get("depends_on_evidence_ids") or []
            ),
            "static_import_ready": bool(delta.get("static_import_ready")),
            "post_materialization_validation": (
                dict(delta["post_materialization_validation"])
                if isinstance(
                    delta.get("post_materialization_validation"),
                    dict,
                )
                else None
            ),
            "old_provider_ambiguous": bool(
                delta.get("old_provider_ambiguous")
            ),
        }.items()
        if value not in (None, "", [], {})
    }


def _difference_components(
    graph: dict[str, Any],
) -> tuple[
    list[list[str]],
    dict[str, dict[str, Any]],
    dict[str, set[str]],
]:
    atoms = {
        str(node.get("id") or ""): dict(node)
        for node in graph.get("nodes") or []
        if isinstance(node, dict)
        and str(node.get("kind") or "") == "DifferenceAtom"
        and str(node.get("id") or "")
    }
    parent = {atom_id: atom_id for atom_id in atoms}

    def find(atom_id: str) -> str:
        while parent[atom_id] != atom_id:
            parent[atom_id] = parent[parent[atom_id]]
            atom_id = parent[atom_id]
        return atom_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    dependency_edges: list[tuple[str, str]] = []
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        edge_type = str(edge.get("type") or "")
        if source not in atoms or target not in atoms:
            continue
        if edge_type in ATOMIC_MERGE_EDGE_TYPES:
            union(source, target)
        elif edge_type == "requires":
            dependency_edges.append((source, target))

    members_by_root: dict[str, list[str]] = {}
    for atom_id in sorted(atoms):
        members_by_root.setdefault(find(atom_id), []).append(atom_id)
    components = sorted(
        members_by_root.values(),
        key=lambda members: tuple(members),
    )
    component_by_atom = {
        atom_id: component_index
        for component_index, members in enumerate(components)
        for atom_id in members
    }
    component_dependencies: dict[str, set[str]] = {
        str(index): set() for index in range(len(components))
    }
    for source, target in dependency_edges:
        owner = component_by_atom[source]
        dependency = component_by_atom[target]
        if owner != dependency:
            component_dependencies[str(owner)].add(str(dependency))
    return components, atoms, component_dependencies


def _source_pack(
    spec: CaseSpec,
    delta: dict[str, Any],
    allowed_files: set[str],
    *,
    conditional_source_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    conditional_source_cache = (
        conditional_source_cache
        if conditional_source_cache is not None
        else {}
    )
    old_records: list[dict[str, Any]] = []
    donor_records: list[dict[str, Any]] = []
    read_only_records: list[dict[str, Any]] = []

    old_entity = delta.get("old_entity") if isinstance(delta.get("old_entity"), dict) else None
    donor_entity = delta.get("donor_entity") if isinstance(delta.get("donor_entity"), dict) else None
    integration_hunks = [
        hunk
        for hunk in delta.get("integration_hunks") or []
        if isinstance(hunk, dict)
    ]
    conditional_variant_records = (
        _conditional_variant_source_records(
            spec,
            donor_entity,
            file_cache=conditional_source_cache,
        )
        if not integration_hunks
        else None
    )
    if str(delta.get("kind") or "") == "reverse_integration" and integration_hunks:
        for hunk in integration_hunks:
            relative_path = normalize_relpath(str(hunk.get("file") or ""))
            old_record = _range_source_record(
                spec.old_root_path,
                tree="old",
                relative_path=relative_path,
                start_line=int(hunk.get("old_start_line") or 1),
                end_line=int(
                    hunk.get("old_end_line")
                    or hunk.get("old_start_line")
                    or 1
                ),
                role="current_integration_hunk",
                reason="old-tree lines corresponding to the exact donor contract-use hunk",
            )
            (old_records if relative_path in allowed_files else read_only_records).append(
                old_record
            )
            donor_records.append(
                _range_source_record(
                    spec.primary_donor_root,
                    tree="donor_0",
                    relative_path=relative_path,
                    start_line=int(hunk.get("donor_start_line") or 1),
                    end_line=int(
                        hunk.get("donor_end_line")
                        or hunk.get("donor_start_line")
                        or 1
                    ),
                    role="donor_integration_hunk",
                    reason="exact donor hunk containing the new driver's typed contract use",
                )
            )
    else:
        record = (
            conditional_variant_records[0]
            if conditional_variant_records is not None
            else _source_record(
                spec.old_root_path,
                tree="old",
                entity=old_entity,
                role="current_contract_owner",
                reason="current old-tree entity at a graph-proven repair endpoint",
            )
        )
        if record:
            (old_records if record["file"] in allowed_files else read_only_records).append(record)
    for declaration in delta.get("old_declarations") or []:
        record = _source_record(
            spec.old_root_path,
            tree="old",
            entity=declaration,
            role="current_declaration",
            reason="old declaration participating in the same provider contract",
        )
        if record:
            (old_records if record["file"] in allowed_files else read_only_records).append(record)

    if not integration_hunks:
        donor_record = (
            conditional_variant_records[1]
            if conditional_variant_records is not None
            else _source_record(
                spec.primary_donor_root,
                tree="donor_0",
                entity=donor_entity,
                role="donor_contract_reference",
                reason="donor entity that satisfies the new driver's contract",
            )
        )
        if donor_record:
            donor_records.append(donor_record)
    for declaration in delta.get("donor_declarations") or []:
        record = _source_record(
            spec.primary_donor_root,
            tree="donor_0",
            entity=declaration,
            role="donor_declaration_reference",
            reason="donor declaration belonging to the same contract",
        )
        if record:
            donor_records.append(record)
    for export in delta.get("donor_exports") or []:
        if not isinstance(export, dict):
            continue
        donor_records.append(
            _line_source_record(
                spec.primary_donor_root,
                tree="donor_0",
                relative_path=str(export.get("file") or ""),
                line=int(export.get("line") or 1),
                role="donor_export_reference",
                reason="exact donor export macro and namespace contract",
            )
        )

    references = list(delta.get("consumer_references") or [])
    if not references and isinstance(delta.get("consumer_reference"), dict):
        references = [delta["consumer_reference"]]
    selected_references = _bounded_consumer_reference_evidence(references)
    for reference in selected_references:
        if not isinstance(reference, dict):
            continue
        consumer_file = normalize_relpath(str(reference.get("source_file") or ""))
        if consumer_file:
            consumer = _line_source_record(
                spec.old_root_path,
                tree="old",
                relative_path=consumer_file,
                line=int(reference.get("line") or 1),
                role="consumer_reference",
                reason="call/type/registration site that requires this contract",
            )
            (old_records if consumer_file in allowed_files else read_only_records).append(consumer)

    return {
        "summary": {
            "contract_evidence": {
                "id": delta.get("id"),
                "contract_identity": delta.get("contract_identity"),
                "kind": delta.get("kind"),
                "symbol": delta.get("symbol"),
                "comparison": delta.get("comparison"),
                "semantic_requirement": delta.get("semantic_requirement"),
                "resolution": delta.get("resolution"),
                "consumer_modules": delta.get("consumer_modules"),
                "provider_modules": delta.get("provider_modules"),
                "change_atom_ids": delta.get("change_atom_ids"),
                "raw_hunk_ids": delta.get("raw_hunk_ids"),
                "integration_hunk_count": len(integration_hunks),
                "consumer_reference_count": len(references),
                "included_consumer_reference_count": len(selected_references),
                "omitted_consumer_reference_count": max(
                    0, len(references) - len(selected_references)
                ),
                "consumer_reference_files": _ordered_unique(
                    reference.get("source_file")
                    for reference in references
                    if isinstance(reference, dict)
                ),
                "old_consumer_impact_status": delta.get("old_consumer_impact_status"),
                "target_validation_files": list(
                    delta.get("target_validation_files") or []
                ),
            }
        },
        "policy": {
            "unit_basis": "demand-projected cross-version differences",
            "scope_expansion": "only through explicit code dependencies",
            "shared_file_copy": "forbidden",
            "no_patch": "allowed when every demanded difference is already satisfied",
            "existing_old_consumers": "validated by the configured target compile after the initial pass",
        },
        "localization_blockers": [],
        "allowed_edit_sources": old_records,
        "donor_reference_sources": donor_records,
        "read_only_evidence_sources": read_only_records,
    }


def _record_identity(record: dict[str, Any]) -> tuple[object, ...]:
    return (
        record.get("tree"),
        normalize_relpath(str(record.get("file") or "")),
        record.get("source_kind"),
        record.get("role"),
        record.get("symbol"),
        record.get("line_start"),
        record.get("line_end"),
    )


def _merge_source_packs(
    packs: Iterable[dict[str, Any]],
    *,
    difference_atoms: list[dict[str, Any]],
    demand_anchors: list[dict[str, Any]],
    repair_loci: list[dict[str, Any]],
    contract_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    pack_list = [dict(pack) for pack in packs if isinstance(pack, dict)]

    def merge_records(key: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[tuple[object, ...]] = set()
        for pack in pack_list:
            for record in pack.get(key) or []:
                if not isinstance(record, dict):
                    continue
                identity = _record_identity(record)
                if identity in seen:
                    continue
                seen.add(identity)
                records.append(dict(record))
        return records

    blockers: list[dict[str, Any]] = []
    seen_blockers: set[str] = set()
    for pack in pack_list:
        for blocker in pack.get("localization_blockers") or []:
            if not isinstance(blocker, dict):
                continue
            blocker_id = stable_id("initial-localization-blocker", repr(sorted(blocker.items())))
            if blocker_id in seen_blockers:
                continue
            seen_blockers.add(blocker_id)
            blockers.append(dict(blocker))
    return {
        "summary": {
            "difference_atom_ids": [
                str(item.get("id") or "")
                for item in difference_atoms
                if item.get("id")
            ],
            "contract_evidence": [
                (pack.get("summary") or {}).get("contract_evidence")
                for pack in pack_list
                if isinstance(
                    (pack.get("summary") or {}).get("contract_evidence"),
                    dict,
                )
            ],
            "contract_evidence_ids": [
                str(item.get("id") or "")
                for item in contract_evidence
                if item.get("id")
            ],
            "demand_anchor_count": len(demand_anchors),
            "repair_locus_count": len(repair_loci),
        },
        "policy": {
            "unit_basis": "one atomic connected component of DifferenceAtom nodes",
            "merge_edges": sorted(ATOMIC_MERGE_EDGE_TYPES),
            "dependency_edges": "schedule units but never merge them",
            "shared_file_copy": "forbidden",
            "validation": "one configured target compile after the complete initial pass",
        },
        "localization_blockers": blockers,
        "allowed_edit_sources": merge_records("allowed_edit_sources"),
        "donor_reference_sources": merge_records("donor_reference_sources"),
        "read_only_evidence_sources": merge_records(
            "read_only_evidence_sources"
        ),
    }


def unit_contract_evidence(unit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in unit.get("contract_evidence") or []
        if isinstance(item, dict)
    ]


def _typed_reference_dependencies(
    evidence_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    """Recover unit ordering from exact donor owner-to-provider references."""

    evidence_by_entity_id: dict[str, set[str]] = {}
    evidence_by_contract_locus: dict[
        tuple[str, str, str, str],
        set[str],
    ] = {}
    for evidence_id, evidence in evidence_by_id.items():
        entities = [
            evidence.get("donor_entity"),
            *list(evidence.get("donor_declarations") or []),
        ]
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("id") or "")
            if entity_id:
                evidence_by_entity_id.setdefault(entity_id, set()).add(
                    evidence_id
                )
            contract_key = str(
                entity.get("contract_key") or entity.get("key") or ""
            )
            relative_path = normalize_relpath(
                str(entity.get("file") or "")
            )
            entity_kind = str(
                entity.get("kind") or entity.get("entity_kind") or ""
            )
            tree = str(entity.get("tree") or "donor_0")
            if contract_key and relative_path and entity_kind:
                evidence_by_contract_locus.setdefault(
                    (tree, relative_path, entity_kind, contract_key),
                    set(),
                ).add(evidence_id)

    dependencies: dict[str, set[str]] = {
        evidence_id: set() for evidence_id in evidence_by_id
    }
    proofs: list[dict[str, Any]] = []
    seen_proofs: set[tuple[str, str, str]] = set()
    for provider_evidence_id, evidence in evidence_by_id.items():
        references = [
            item
            for item in evidence.get("consumer_references") or []
            if isinstance(item, dict)
        ]
        if not references and isinstance(
            evidence.get("consumer_reference"),
            dict,
        ):
            references = [evidence["consumer_reference"]]
        for reference in references:
            owner = (
                reference.get("owner_entity")
                if isinstance(reference.get("owner_entity"), dict)
                else {}
            )
            owner_id = str(
                owner.get("id")
                or reference.get("owner_entity_id")
                or ""
            )
            candidates = set(
                evidence_by_entity_id.get(owner_id, set())
                if owner_id
                else set()
            )
            resolution = "exact_owner_entity_id"
            if not candidates:
                contract_key = str(
                    owner.get("contract_key") or owner.get("key") or ""
                )
                relative_path = normalize_relpath(
                    str(
                        owner.get("file")
                        or reference.get("source_file")
                        or ""
                    )
                )
                entity_kind = str(
                    owner.get("kind")
                    or owner.get("entity_kind")
                    or ""
                )
                tree = str(
                    owner.get("tree")
                    or reference.get("tree")
                    or "donor_0"
                )
                if contract_key and relative_path and entity_kind:
                    candidates = set(
                        evidence_by_contract_locus.get(
                            (
                                tree,
                                relative_path,
                                entity_kind,
                                contract_key,
                            ),
                            set(),
                        )
                    )
                    resolution = "unique_owner_contract_locus"
            if len(candidates) != 1:
                continue
            consumer_evidence_id = next(iter(candidates))
            if consumer_evidence_id == provider_evidence_id:
                continue
            dependencies[consumer_evidence_id].add(provider_evidence_id)
            proof_key = (
                consumer_evidence_id,
                provider_evidence_id,
                str(reference.get("id") or ""),
            )
            if proof_key in seen_proofs:
                continue
            seen_proofs.add(proof_key)
            proofs.append(
                {
                    "schema": "typed_initial_unit_dependency_v1",
                    "consumer_evidence_id": consumer_evidence_id,
                    "provider_evidence_id": provider_evidence_id,
                    "reference_id": reference.get("id"),
                    "source_file": normalize_relpath(
                        str(reference.get("source_file") or "")
                    ),
                    "line": int(reference.get("line") or 0),
                    "column": int(reference.get("column") or 0),
                    "owner_entity_id": owner_id or None,
                    "provider_contract_identity": evidence.get(
                        "contract_identity"
                    ),
                    "resolution": resolution,
                }
            )
    return dependencies, proofs


def _component_mode(evidence: list[dict[str, Any]]) -> str:
    modes = {
        str(item.get("resolution") or "")
        for item in evidence
        if str(item.get("resolution") or "")
    }
    if "requires_llm" in modes:
        return "requires_llm"
    if modes == {"post_materialization_validation_only"}:
        return "post_materialization_validation_only"
    deterministic = {mode for mode in modes if mode.startswith("deterministic_")}
    if len(deterministic) == 1 and len(modes) == 1:
        return next(iter(deterministic))
    # One atomic edit cannot safely be split between different deterministic
    # operations.  The combined component therefore becomes one model decision.
    return "requires_llm"


def _donor_source_order(
    unit: dict[str, Any],
) -> tuple[str, str, str, int, int, str] | None:
    """Return one graph-derived donor declaration position for scheduling."""

    entities = [
        evidence.get("donor_entity")
        for evidence in unit.get("contract_evidence") or []
        if isinstance(evidence, dict)
        and isinstance(evidence.get("donor_entity"), dict)
    ]
    identities = {
        (
            normalize_relpath(str(entity.get("file") or "")),
            str(
                (
                    entity.get("container")
                    if isinstance(entity.get("container"), dict)
                    else {}
                ).get("container_kind")
                or ""
            ),
            str(
                (
                    entity.get("container")
                    if isinstance(entity.get("container"), dict)
                    else {}
                ).get("container_name")
                or ""
            ),
            int((entity.get("extent") or {}).get("start_line") or 0),
            int((entity.get("extent") or {}).get("start_column") or 0),
            str(entity.get("contract_key") or entity.get("key") or ""),
        )
        for entity in entities
    }
    if len(identities) != 1:
        return None
    position = next(iter(identities))
    if not position[0] or position[3] <= 0:
        return None
    return position


def _order_units_by_donor_source(
    units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use structural source order as the stable tie-breaker before topology."""

    original_position = {
        str(unit.get("id") or ""): index
        for index, unit in enumerate(units)
    }
    for unit in units:
        position = _donor_source_order(unit)
        if position is None:
            continue
        unit["structural_execution_order"] = {
            "schema": "initial_structural_execution_order_v1",
            "basis": "donor_declaration_source_order",
            "owner_file": position[0],
            "container_kind": position[1] or None,
            "container_name": position[2] or None,
            "start_line": position[3],
            "start_column": position[4],
            "contract_key": position[5],
        }

    return sorted(
        units,
        key=lambda unit: (
            0 if _donor_source_order(unit) is not None else 1,
            _donor_source_order(unit)
            or (
                "",
                "",
                "",
                0,
                0,
                "",
            ),
            original_position.get(str(unit.get("id") or ""), 0),
        ),
    )


def _unit_translation_units(unit: dict[str, Any]) -> set[str]:
    source_suffixes = {".c", ".cc", ".cpp", ".cxx"}
    candidates = [
        *[
            anchor.get("source_file")
            for anchor in unit.get("demand_anchors") or []
            if isinstance(anchor, dict)
        ],
        *[
            reference.get("translation_unit_file")
            or reference.get("source_file")
            for evidence in unit.get("contract_evidence") or []
            if isinstance(evidence, dict)
            for reference in evidence.get("consumer_references") or []
            if isinstance(reference, dict)
        ],
    ]
    return {
        normalize_relpath(str(path))
        for path in candidates
        if str(path)
        and Path(normalize_relpath(str(path))).suffix.lower()
        in source_suffixes
    }


def _apply_translation_unit_analysis_boundaries(
    units: list[dict[str, Any]],
    target_translation_unit_validity: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Defer units whose target-side comparison used an invalid exact TU."""

    invalid_by_translation_unit = {
        normalize_relpath(str(record.get("translation_unit_file") or "")): (
            dict(record)
        )
        for record in target_translation_unit_validity
        if isinstance(record, dict)
        and str(record.get("status") or "")
        == "invalid_active_unresolved_include"
        and normalize_relpath(
            str(record.get("translation_unit_file") or "")
        )
    }
    boundaries: list[dict[str, Any]] = []
    for unit in units:
        affected_translation_units = sorted(
            _unit_translation_units(unit) & set(invalid_by_translation_unit)
        )
        if not affected_translation_units:
            continue
        affected_anchor_ids = sorted(
            {
                str(anchor.get("id") or "")
                for anchor in unit.get("demand_anchors") or []
                if isinstance(anchor, dict)
                and normalize_relpath(str(anchor.get("source_file") or ""))
                in affected_translation_units
                and str(anchor.get("id") or "")
            }
        )
        boundary = {
            "schema": "initial_translation_unit_analysis_boundary_v1",
            "reason": (
                "target-side contract comparison was derived from an exact "
                "translation unit with an active unresolved include"
            ),
            "invalidated_translation_units": affected_translation_units,
            "affected_demand_anchor_ids": affected_anchor_ids,
            "target_validity_evidence": [
                invalid_by_translation_unit[path]
                for path in affected_translation_units
            ],
            "required_transition": (
                "compile_current_target_then_rebuild_diagnostic_code_graph_"
                "and_preserve_unconfirmed_demands"
            ),
        }
        unit["requires_context_expansion"] = True
        unit["analysis_validity_boundary"] = boundary
        decision = dict(unit.get("repair_decision") or {})
        decision["analysis_validity_boundary"] = boundary
        unit["repair_decision"] = decision
        source_pack = dict(unit.get("repair_source_pack") or {})
        blockers = list(source_pack.get("localization_blockers") or [])
        blockers.append(boundary)
        source_pack["localization_blockers"] = blockers
        unit["repair_source_pack"] = source_pack
        boundaries.append(
            {
                "work_unit_id": str(
                    unit.get("id") or unit.get("work_unit_id") or ""
                ),
                **boundary,
            }
        )
    return units, boundaries


def build_initial_compatibility_plan(
    spec: CaseSpec,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    units: list[dict[str, Any]] = []
    conditional_source_cache: dict[str, dict[str, Any]] = {}
    raw_hunks_by_id = {
        str(item.get("id") or ""): dict(item)
        for item in analysis.get("raw_hunks") or []
        if isinstance(item, dict) and item.get("id")
    }
    evidence_by_id = {
        str(item.get("id") or ""): _contract_evidence(item)
        for item in analysis.get("contract_evidence") or []
        if isinstance(item, dict) and item.get("id")
    }
    graph = (
        dict(analysis.get("initial_difference_graph") or {})
        if (analysis.get("initial_difference_graph") or {}).get("schema")
        == "initial_difference_graph_v3"
        else build_initial_difference_graph(
            contract_evidence=list(evidence_by_id.values()),
            change_atoms=analysis.get("change_atoms") or [],
            raw_hunks=analysis.get("raw_hunks") or [],
        )
    )
    components, atom_by_id, component_dependencies = _difference_components(
        graph
    )
    typed_evidence_dependencies, typed_dependency_proofs = (
        _typed_reference_dependencies(evidence_by_id)
    )
    component_by_evidence: dict[str, int] = {}
    for component_index, atom_ids in enumerate(components):
        for atom_id in atom_ids:
            evidence_id = str(
                (atom_by_id.get(atom_id) or {}).get(
                    "contract_evidence_id"
                )
                or ""
            )
            if evidence_id:
                component_by_evidence[evidence_id] = component_index
    typed_proofs_by_component: dict[int, list[dict[str, Any]]] = {}
    for consumer_evidence_id, provider_evidence_ids in (
        typed_evidence_dependencies.items()
    ):
        consumer_component = component_by_evidence.get(
            consumer_evidence_id
        )
        if consumer_component is None:
            continue
        for provider_evidence_id in provider_evidence_ids:
            provider_component = component_by_evidence.get(
                provider_evidence_id
            )
            if (
                provider_component is None
                or provider_component == consumer_component
            ):
                continue
            component_dependencies[str(consumer_component)].add(
                str(provider_component)
            )
    for proof in typed_dependency_proofs:
        consumer_component = component_by_evidence.get(
            str(proof.get("consumer_evidence_id") or "")
        )
        provider_component = component_by_evidence.get(
            str(proof.get("provider_evidence_id") or "")
        )
        if (
            consumer_component is None
            or provider_component is None
            or consumer_component == provider_component
        ):
            continue
        typed_proofs_by_component.setdefault(
            consumer_component,
            [],
        ).append(proof)
    nodes = [
        dict(item)
        for item in graph.get("nodes") or []
        if isinstance(item, dict)
    ]
    anchors_by_evidence: dict[str, list[dict[str, Any]]] = {}
    loci_by_evidence: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        evidence_id = str(node.get("contract_evidence_id") or "")
        if str(node.get("kind") or "") == "DemandAnchor":
            anchors_by_evidence.setdefault(evidence_id, []).append(node)
        elif str(node.get("kind") or "") == "RepairLocus":
            loci_by_evidence.setdefault(evidence_id, []).append(node)

    component_unit_ids: dict[str, str] = {}
    for component_index, atom_ids in enumerate(components):
        difference_atoms = [
            atom_by_id[atom_id]
            for atom_id in atom_ids
            if atom_id in atom_by_id
        ]
        evidence_ids = sorted(
            {
                str(atom.get("contract_evidence_id") or "")
                for atom in difference_atoms
                if (
                    str(atom.get("contract_evidence_id") or "")
                    in evidence_by_id
                )
            }
        )
        evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in evidence_ids
        ]
        if not evidence:
            continue
        resolution = _component_mode(evidence)
        contract_identities = _ordered_unique(
            item.get("contract_identity") for item in evidence
        )
        demand_anchors = [
            dict(anchor)
            for evidence_id in evidence_ids
            for anchor in anchors_by_evidence.get(evidence_id, [])
        ]
        repair_loci = [
            dict(locus)
            for evidence_id in evidence_ids
            for locus in loci_by_evidence.get(evidence_id, [])
        ]
        for locus in repair_loci:
            relative_path = normalize_relpath(str(locus.get("file") or ""))
            locus["raw_hunks"] = [
                raw_hunk
                for hunk_id in locus.get("raw_hunk_ids") or []
                for raw_hunk in [raw_hunks_by_id.get(str(hunk_id))]
                if isinstance(raw_hunk, dict)
                and normalize_relpath(str(raw_hunk.get("file") or ""))
                == relative_path
            ]
        candidate_files = _ordered_unique(
            relative_path
            for item in evidence
            for relative_path in _candidate_edit_files(item)
        )
        editable: list[str] = []
        escalation: list[str] = []
        forbidden: list[str] = []
        missing_old_candidates: list[str] = []
        for relative_path in candidate_files:
            classification = spec.classify_edit_root(relative_path)
            old_exists = spec.resolve_old(relative_path).is_file()
            if not old_exists:
                missing_old_candidates.append(relative_path)
            elif classification == "editable":
                editable.append(relative_path)
            elif classification == "escalation":
                escalation.append(relative_path)
            else:
                forbidden.append(relative_path)
        allowed_set = set(editable)
        unit_id = stable_id("initial-difference-unit", *atom_ids)
        component_unit_ids[str(component_index)] = unit_id
        requires_context = bool(
            not editable
            and resolution
            != "post_materialization_validation_only"
        )
        source_pack = _merge_source_packs(
            (
                _source_pack(
                    spec,
                    item,
                    allowed_set,
                    conditional_source_cache=conditional_source_cache,
                )
                for item in evidence
            ),
            difference_atoms=difference_atoms,
            demand_anchors=demand_anchors,
            repair_loci=repair_loci,
            contract_evidence=evidence,
        )
        if missing_old_candidates and not editable:
            source_pack["localization_blockers"].append(
                {
                    "reason": "candidate contract owner is absent from the old tree",
                    "resolution": resolution,
                    "candidate_files": candidate_files,
                    "missing_old_candidates": missing_old_candidates,
                    "required_action": "atomic_graph_scope_expansion",
                }
            )
        elif requires_context:
            source_pack["localization_blockers"].append(
                {
                    "reason": (
                        "no editable repair endpoint is available for the "
                        "demanded difference"
                    ),
                    "resolution": resolution,
                    "candidate_files": candidate_files,
                }
            )
        units.append(
            {
                "id": unit_id,
                "work_unit_id": unit_id,
                "title": "initial compatibility: " + ", ".join(
                    _ordered_unique(item.get("symbol") for item in evidence)[:3]
                    or contract_identities[:3]
                ),
                "scope": ",".join(contract_identities),
                "unit_kind": "initial_compatibility",
                "execution_phase": (
                    EXECUTION_PHASE_INITIAL_COMPATIBILITY
                ),
                "allowed_fileset": editable,
                "analysis_read_files": _ordered_unique(
                    [
                        *candidate_files,
                        *[
                            item.get("source_file")
                            for item in demand_anchors
                        ],
                    ]
                ),
                "escalation_candidates": escalation,
                "forbidden_candidates": forbidden,
                "missing_old_candidates": missing_old_candidates,
                "read_only_evidence_files": _ordered_unique(
                    [
                        record.get("file")
                        for record in source_pack.get("read_only_evidence_sources") or []
                    ]
                ),
                "candidate_files": candidate_files,
                "requires_escalation": bool(escalation and not editable),
                "requires_context_expansion": requires_context,
                "contract_evidence": evidence,
                "contract_evidence_ids": evidence_ids,
                "contract_identities": contract_identities,
                "dependency_evidence": list(
                    typed_proofs_by_component.get(component_index, [])
                ),
                "difference_atoms": difference_atoms,
                "difference_atom_ids": atom_ids,
                "demand_anchors": demand_anchors,
                "repair_loci": repair_loci,
                "repair_decision": {
                    "schema": "initial_difference_decision_v1",
                    "mode": resolution,
                    "difference_atom_ids": atom_ids,
                    "repair_locus_ids": [
                        str(item.get("id") or "")
                        for item in repair_loci
                        if item.get("id")
                    ],
                    "candidate_write_set": editable,
                    "reasoning": [
                        "the unit is an atomic connected component of demanded cross-version differences",
                        "only semantic-decision, atomic-edit, or overlapping-write edges merge atoms",
                        "requires edges only order independently repairable units",
                        "shared donor files may not be copied wholesale",
                    ],
                },
                "repair_source_pack": source_pack,
                "localization": {
                    "anchor_kind": "DifferenceAtomSet",
                    "difference_atom_ids": atom_ids,
                    "demand_anchor_ids": [
                        str(item.get("id") or "")
                        for item in demand_anchors
                        if item.get("id")
                    ],
                    "repair_locus_ids": [
                        str(item.get("id") or "")
                        for item in repair_loci
                        if item.get("id")
                    ],
                },
                "patch_targets": [
                    {
                        "file": relative_path,
                        "authorization": "editable",
                        "difference_atom_ids": atom_ids,
                        "repair_locus_ids": [
                            str(item.get("id") or "")
                            for item in repair_loci
                            if normalize_relpath(
                                str(item.get("file") or "")
                            )
                            == relative_path
                            and item.get("id")
                        ],
                    }
                    for relative_path in editable
                ],
            }
        )
    units, analysis_validity_boundaries = (
        _apply_translation_unit_analysis_boundaries(
            units,
            analysis.get("target_translation_unit_validity") or [],
        )
    )
    units = _order_units_by_donor_source(units)
    unit_by_id = {str(unit.get("id") or ""): unit for unit in units}
    for component_index, unit_id in component_unit_ids.items():
        unit = unit_by_id.get(unit_id)
        if unit is None:
            continue
        unit["depends_on_work_unit_ids"] = sorted(
            component_unit_ids[dependency]
            for dependency in component_dependencies.get(
                component_index, set()
            )
            if dependency in component_unit_ids
        )
    units, dependency_schedule = dependency_order(units)
    return {
        "schema": INITIAL_COMPATIBILITY_PLAN_SCHEMA,
        "difference_unit_revision": (
            INITIAL_COMPATIBILITY_FRONTIER_REVISION
        ),
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "execution_phase": EXECUTION_PHASE_INITIAL_COMPATIBILITY,
        "source_analysis_schema": analysis.get("schema"),
        "work_unit_plans": units,
        "summary": {
            "execution_phase": EXECUTION_PHASE_INITIAL_COMPATIBILITY,
            "work_unit_count": len(units),
            "difference_atom_count": sum(
                len(unit.get("difference_atoms") or []) for unit in units
            ),
            "contract_evidence_count": len(
                {
                    evidence_id
                    for unit in units
                    for evidence_id in unit.get("contract_evidence_ids") or []
                }
            ),
            "multi_atom_unit_count": sum(
                len(unit.get("difference_atoms") or []) > 1
                for unit in units
            ),
            "typed_reference_dependency_count": len(
                {
                    (
                        str(proof.get("consumer_evidence_id") or ""),
                        str(proof.get("provider_evidence_id") or ""),
                    )
                    for proofs in typed_proofs_by_component.values()
                    for proof in proofs
                    if str(proof.get("consumer_evidence_id") or "")
                    and str(proof.get("provider_evidence_id") or "")
                }
            ),
            "deterministic_unit_count": sum(
                str((unit.get("repair_decision") or {}).get("mode") or "").startswith("deterministic_")
                and not unit.get("requires_context_expansion")
                for unit in units
            ),
            "llm_unit_count": sum(
                (unit.get("repair_decision") or {}).get("mode") == "requires_llm"
                and not unit.get("requires_context_expansion")
                for unit in units
            ),
            "post_materialization_validation_unit_count": sum(
                (unit.get("repair_decision") or {}).get("mode")
                == "post_materialization_validation_only"
                for unit in units
            ),
            "blocked_unit_count": sum(bool(unit.get("requires_context_expansion")) for unit in units),
            "analysis_validity_boundary_count": len(
                analysis_validity_boundaries
            ),
            "analysis_validity_boundaries": analysis_validity_boundaries,
            "invalid_target_translation_unit_count": len(
                {
                    translation_unit
                    for boundary in analysis_validity_boundaries
                    for translation_unit in boundary.get(
                        "invalidated_translation_units"
                    )
                    or []
                }
            ),
            "dependency_schedule": dependency_schedule,
            "execution_order_basis": (
                "dependency_topology_then_donor_declaration_source_order"
            ),
        },
    }


def _refresh_entity_in_current_tree(
    spec: CaseSpec,
    entity: dict[str, Any],
    *,
    code_index: CodeIndex,
    report_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    relative_path = normalize_relpath(str(entity.get("file") or ""))
    contract = str(entity.get("contract_key") or entity.get("key") or "")
    kind = str(entity.get("kind") or entity.get("entity_kind") or "")
    if not relative_path or not contract:
        return dict(entity)
    resolution = resolve_entity_in_case_tree(
        spec,
        descriptor=entity,
        relative_path=relative_path,
        tree="old",
        require_definition=bool(
            entity.get("is_definition") is True
            or kind in {"function", "variable"}
        ),
        index=code_index,
        report_cache=report_cache,
    )
    current_resolution = {
        key: resolution.get(key)
        for key in (
            "schema",
            "status",
            "tree",
            "file",
            "contract_key",
            "entity_kind",
            "backend",
            "compile_aware",
            "owner_presence_authoritative",
            "parse_ok",
            "parse_error",
            "match_count",
        )
    }
    if resolution.get("status") != "resolved":
        unresolved = dict(entity)
        unresolved["current_resolution"] = current_resolution
        unresolved["current_entity_id"] = None
        unresolved["current_extent"] = None
        return unresolved
    current = dict(resolution["entity"])
    current["current_resolution"] = current_resolution
    current["current_entity_id"] = current.get("id")
    current["current_extent"] = dict(current.get("extent") or {})
    previous_id = str(entity.get("id") or "")
    current_id = str(current.get("id") or "")
    aliases = {
        str(item)
        for item in [
            *list(entity.get("evidence_alias_ids") or []),
            *list(current.get("evidence_alias_ids") or []),
            entity.get("replaces_entity_id"),
            previous_id,
        ]
        if str(item or "")
    }
    if aliases:
        current["evidence_alias_ids"] = sorted(aliases)
    if previous_id and current_id and previous_id != current_id:
        current["replaces_entity_id"] = previous_id
    return current


def _refresh_reference_in_current_tree(
    spec: CaseSpec,
    reference: dict[str, Any],
    *,
    code_index: CodeIndex,
    report_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    refreshed = dict(reference)
    owner = (
        reference.get("owner_entity")
        if isinstance(reference.get("owner_entity"), dict)
        else None
    )
    if owner is None:
        return refreshed
    current_owner = _refresh_entity_in_current_tree(
        spec,
        owner,
        code_index=code_index,
        report_cache=report_cache,
    )
    owner_resolution = (
        current_owner.get("current_resolution")
        if isinstance(
            current_owner.get("current_resolution"),
            dict,
        )
        else {}
    )
    refreshed["owner_current_resolution"] = dict(
        owner_resolution
    )
    if owner_resolution.get("status") != "resolved":
        refreshed["owner_entity"] = current_owner
        refreshed["owner_entity_id"] = None
        return refreshed
    old_start = int((owner.get("extent") or {}).get("start_line") or 0)
    current_start = int(
        (current_owner.get("extent") or {}).get("start_line") or 0
    )
    offset = current_start - old_start if old_start and current_start else 0
    refreshed["owner_entity"] = current_owner
    refreshed["owner_entity_id"] = current_owner.get("id")
    if offset:
        line = int(reference.get("line") or 0)
        if line:
            refreshed["line"] = line + offset
        extent = dict(reference.get("extent") or {})
        if int(extent.get("start_line") or 0):
            extent["start_line"] = int(extent["start_line"]) + offset
        if int(extent.get("end_line") or 0):
            extent["end_line"] = int(extent["end_line"]) + offset
        if extent:
            refreshed["extent"] = extent
    return refreshed


def refresh_initial_unit_from_current_tree(
    spec: CaseSpec,
    unit: dict[str, Any],
    *,
    changed_files: Iterable[object] = (),
) -> dict[str, Any]:
    """Refresh source extents locally after earlier initial units changed files.

    The semantic difference graph stays immutable for the initial pass.  Only
    current-tree entities and source excerpts are re-read; no provider search,
    graph expansion, or cross-version analysis is repeated.
    """

    refreshed = deepcopy(unit)
    touched = {
        normalize_relpath(str(item or ""))
        for item in changed_files
        if normalize_relpath(str(item or ""))
    }
    current_index = CodeIndex(
        spec,
        graph_projection="records",
    )
    report_cache: dict[tuple[str, str], dict[str, Any]] = {}
    evidence_records = unit_contract_evidence(refreshed)
    for evidence in evidence_records:
        old_entity = evidence.get("old_entity")
        if isinstance(old_entity, dict):
            relative_path = normalize_relpath(
                str(old_entity.get("file") or "")
            )
            if not touched or relative_path in touched:
                evidence["old_entity"] = _refresh_entity_in_current_tree(
                    spec,
                    old_entity,
                    code_index=current_index,
                    report_cache=report_cache,
                )
        declarations: list[dict[str, Any]] = []
        for declaration in evidence.get("old_declarations") or []:
            if not isinstance(declaration, dict):
                continue
            relative_path = normalize_relpath(
                str(declaration.get("file") or "")
            )
            declarations.append(
                _refresh_entity_in_current_tree(
                    spec,
                    declaration,
                    code_index=current_index,
                    report_cache=report_cache,
                )
                if not touched or relative_path in touched
                else dict(declaration)
            )
        evidence["old_declarations"] = declarations
        evidence["consumer_references"] = [
            _refresh_reference_in_current_tree(
                spec,
                reference,
                code_index=current_index,
                report_cache=report_cache,
            )
            if isinstance(reference, dict)
            and (
                not touched
                or normalize_relpath(
                    str(reference.get("source_file") or "")
                )
                in touched
            )
            else dict(reference)
            for reference in evidence.get("consumer_references") or []
            if isinstance(reference, dict)
        ]
    refreshed["contract_evidence"] = evidence_records

    evidence_by_id = {
        str(item.get("id") or ""): item for item in evidence_records
    }
    for locus in refreshed.get("repair_loci") or []:
        if not isinstance(locus, dict):
            continue
        evidence = evidence_by_id.get(
            str(locus.get("contract_evidence_id") or "")
        )
        current = (evidence or {}).get("old_entity")
        if isinstance(current, dict):
            locus["old_entity_id"] = current.get("id")
            locus["old_extent"] = dict(current.get("extent") or {})
    refreshed_references = {
        str(reference.get("id") or ""): reference
        for evidence in evidence_records
        for reference in evidence.get("consumer_references") or []
        if isinstance(reference, dict) and reference.get("id")
    }
    for anchor in refreshed.get("demand_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        reference = refreshed_references.get(
            str(anchor.get("reference_id") or "")
        )
        if reference is None:
            continue
        for key in (
            "source_file",
            "line",
            "column",
            "extent",
            "owner_entity",
            "owner_entity_id",
            "owner_current_resolution",
        ):
            if key in reference:
                anchor[key] = deepcopy(reference[key])

    allowed_set = {
        normalize_relpath(str(item or ""))
        for item in refreshed.get("allowed_fileset") or []
        if normalize_relpath(str(item or ""))
    }
    refreshed["repair_source_pack"] = _merge_source_packs(
        (
            _source_pack(spec, evidence, allowed_set)
            for evidence in evidence_records
        ),
        difference_atoms=[
            dict(item)
            for item in refreshed.get("difference_atoms") or []
            if isinstance(item, dict)
        ],
        demand_anchors=[
            dict(item)
            for item in refreshed.get("demand_anchors") or []
            if isinstance(item, dict)
        ],
        repair_loci=[
            dict(item)
            for item in refreshed.get("repair_loci") or []
            if isinstance(item, dict)
        ],
        contract_evidence=evidence_records,
    )
    refreshed["local_refresh"] = {
        "schema": "initial_current_tree_local_refresh_v1",
        "files": sorted(
            {
                relative_path
                for _tree, relative_path in report_cache
            }
        ),
        "entity_resolution": [
            {
                "tree": tree,
                "file": relative_path,
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
            for (tree, relative_path), report in sorted(
                report_cache.items()
            )
        ],
        "graph_rebuilt": False,
    }
    return refreshed
