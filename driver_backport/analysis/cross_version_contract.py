#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque
import filecmp
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable

from driver_backport.analysis.change_graph import (
    build_initial_difference_graph,
    build_change_atoms,
    build_file_change_manifest,
    hunks_overlapping_extent,
    structured_line_hunks,
)
from driver_backport.analysis.conditional_contract import (
    conditional_variant_insertions,
)
from driver_backport.analysis.code_index import (
    CodeIndex,
    STRUCTURAL_ALIGNMENT_INSERTION_STATUSES,
    canonical_entity_records,
    parse_structural_entity_file,
    reference_crosses_entity_boundary,
    structural_entity_neighbor_alignment,
)
from driver_backport.analysis.entity_fingerprint import (
    compare_entity_fingerprints,
    entity_is_internally_linked,
)
from driver_backport.analysis.export_index import (
    export_contract_relation,
    index_export_contracts,
    module_ownership_for_sources,
)
from driver_backport.analysis.kbuild_index import KbuildIndex
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths as _ordered_unique,
    stable_id,
    utc_now,
)


SOURCE_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp"}
INITIAL_COMPATIBILITY_ANALYSIS_SCHEMA = "initial_compatibility_analysis_v1"
INITIAL_COMPATIBILITY_ANALYSIS_REVISION = (
    "canonical_contract_transaction_frontier_v44"
)
STATIC_IMPORT_ENTITY_KINDS = {"macro", "field", "enum_constant", "function"}
DIRECT_PROVIDER_ENTITY_KINDS = {
    "macro",
    "field",
    "enum_constant",
    "function",
    "type",
    "variable",
}


def _target_relative_diagnostic_file(
    raw_file: object,
    *,
    old_root: Path,
) -> str | None:
    text = str(raw_file or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        try:
            return normalize_relpath(
                path.resolve().relative_to(old_root.resolve()).as_posix()
            )
        except ValueError:
            return None
    try:
        return normalize_relpath(text)
    except ValueError:
        return None


def _unresolved_include_candidate_paths(
    include: dict[str, Any],
    *,
    driver_root: str,
) -> list[str]:
    name = normalize_relpath(str(include.get("name") or ""))
    source_file = normalize_relpath(
        str(include.get("source_file") or "")
    )
    if not name or not source_file:
        return []
    candidates = [
        f"include/{name}",
        f"include/uapi/{name}",
        name,
    ]
    if str(include.get("delimiter") or "") == '"':
        candidates.extend(
            [
                (Path(source_file).parent / name).as_posix(),
                (Path(driver_root) / name).as_posix(),
            ]
        )
    return _ordered_unique(candidates)


def _target_translation_unit_validity(
    index: CodeIndex,
    spec: CaseSpec,
) -> list[dict[str, Any]]:
    """Return build-conditioned target TUs invalidated by active includes.

    A textual unresolved include is not enough: the include line must also be
    the locus of a fatal diagnostic from the exact target compile context.
    """

    records_by_translation_unit: dict[str, dict[str, Any]] = {}
    for (tree, _relative_path), report in index.parse_reports.items():
        if tree != "old" or not isinstance(report, dict):
            continue
        fatal_loci = {
            (relative_file, int(location.get("line") or 0))
            for diagnostic in report.get("diagnostics") or []
            if isinstance(diagnostic, dict)
            and int(diagnostic.get("severity") or 0) >= 4
            and isinstance(diagnostic.get("location"), dict)
            for location in [dict(diagnostic.get("location") or {})]
            for relative_file in [
                _target_relative_diagnostic_file(
                    location.get("file"),
                    old_root=spec.old_root_path,
                )
            ]
            if relative_file and int(location.get("line") or 0) > 0
        }
        if not fatal_loci:
            continue
        translation_unit = normalize_relpath(
            str(
                report.get("translation_unit_file")
                or report.get("file")
                or ""
            )
        )
        if not translation_unit:
            continue
        for include in report.get("includes") or []:
            if not isinstance(include, dict) or include.get("resolved"):
                continue
            source_file = normalize_relpath(
                str(include.get("source_file") or report.get("file") or "")
            )
            line = int(include.get("line") or 0)
            if (source_file, line) not in fatal_loci:
                continue
            candidate_paths = _unresolved_include_candidate_paths(
                include,
                driver_root=spec.driver_root,
            )
            if not candidate_paths:
                continue
            record = records_by_translation_unit.setdefault(
                translation_unit,
                {
                    "schema": "target_translation_unit_validity_v1",
                    "tree": "old",
                    "translation_unit_file": translation_unit,
                    "status": "invalid_active_unresolved_include",
                    "active_unresolved_includes": [],
                },
            )
            record["active_unresolved_includes"].append(
                {
                    "source_file": source_file,
                    "line": line,
                    "name": include.get("name"),
                    "delimiter": include.get("delimiter"),
                    "candidate_paths": candidate_paths,
                    "evidence": (
                        "unresolved_include_at_fatal_target_diagnostic_locus"
                    ),
                }
            )
    return [
        records_by_translation_unit[key]
        for key in sorted(records_by_translation_unit)
    ]


def _reachable_demand_ledger(
    evaluated_contracts: Iterable[dict[str, Any]],
    *,
    root_contract_identities: Iterable[object],
    contract_dependencies: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Retain every projected demand reachable from a designated entry.

    Resolution controls the next phase for a demand; it does not decide
    whether the demand remains in the ledger. Satisfied, compile-validation-
    only, and evidence-blocked records all survive until the compile-valid
    current tree is evaluated.
    """

    records = [dict(item) for item in evaluated_contracts]
    by_identity = {
        str(record.get("contract_identity") or ""): record
        for record in records
        if str(record.get("contract_identity") or "")
    }
    dependencies: dict[str, set[str]] = defaultdict(set)
    for parent_identity, dependency_identity in contract_dependencies:
        if (
            parent_identity in by_identity
            and dependency_identity in by_identity
        ):
            dependencies[parent_identity].add(dependency_identity)

    reachable = {
        str(identity)
        for identity in root_contract_identities
        if str(identity) in by_identity
    }
    queue = deque(sorted(reachable))
    while queue:
        parent_identity = queue.popleft()
        for dependency_identity in sorted(
            dependencies.get(parent_identity, set())
        ):
            if dependency_identity in reachable:
                continue
            reachable.add(dependency_identity)
            queue.append(dependency_identity)

    ledger = [
        record
        for record in records
        if str(record.get("contract_identity") or "") in reachable
    ]
    ledger_ids = {
        str(record.get("id") or "")
        for record in ledger
        if str(record.get("id") or "")
    }
    for record in ledger:
        record["depends_on_evidence_ids"] = sorted(
            {
                str(dependency_id)
                for dependency_id in record.get(
                    "depends_on_evidence_ids"
                )
                or []
                if str(dependency_id) in ledger_ids
            }
        )
    return ledger


def _reachable_actionable_contract_evidence(
    evaluated_contracts: Iterable[dict[str, Any]],
    *,
    root_contract_identities: Iterable[object],
    contract_dependencies: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Return the current executable frontier without losing the ledger.

    A descendant cannot become an initial repair task through a satisfied,
    validation-only, or blocked parent. It remains in the demand ledger and
    can be promoted after fresh target-build evidence changes that parent.
    """

    records = [dict(item) for item in evaluated_contracts]
    actionable = {
        str(record.get("contract_identity") or ""): record
        for record in records
        if str(record.get("contract_identity") or "")
        and (
            record.get("resolution") == "requires_llm"
            or str(record.get("resolution") or "").startswith(
                "deterministic_"
            )
        )
    }
    dependencies: dict[str, set[str]] = defaultdict(set)
    for parent_identity, dependency_identity in contract_dependencies:
        if (
            parent_identity in actionable
            and dependency_identity in actionable
        ):
            dependencies[parent_identity].add(dependency_identity)

    reachable = {
        str(identity)
        for identity in root_contract_identities
        if str(identity) in actionable
    }
    queue = deque(sorted(reachable))
    while queue:
        parent_identity = queue.popleft()
        for dependency_identity in sorted(
            dependencies.get(parent_identity, set())
        ):
            if dependency_identity in reachable:
                continue
            reachable.add(dependency_identity)
            queue.append(dependency_identity)

    selected = [
        record
        for record in records
        if str(record.get("contract_identity") or "") in reachable
    ]
    selected_ids = {
        str(record.get("id") or "")
        for record in selected
        if str(record.get("id") or "")
    }
    for record in selected:
        record["depends_on_evidence_ids"] = sorted(
            {
                str(dependency_id)
                for dependency_id in record.get(
                    "depends_on_evidence_ids"
                )
                or []
                if str(dependency_id) in selected_ids
            }
        )
    return selected


def _records_with_unique_ids(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        record = dict(raw)
        identity = str(record.get("id") or record.get("entity_id") or "")
        if not identity:
            identity = stable_id("anonymous-record", json.dumps(record, sort_keys=True))
        if identity in seen:
            continue
        seen.add(identity)
        result.append(record)
    return result


def _static_import_entity_is_supported(entity: dict[str, Any]) -> bool:
    kind = str(entity.get("kind") or "")
    if kind != "function":
        return kind in STATIC_IMPORT_ENTITY_KINDS
    interface = (
        entity.get("interface")
        if isinstance(entity.get("interface"), dict)
        else {}
    )
    return bool(
        entity.get("is_definition")
        and Path(str(entity.get("file") or "")).suffix.lower() in {".h", ".hpp"}
        and (
            str(interface.get("storage_class") or "") == "STATIC"
            or str(interface.get("linkage") or "") == "INTERNAL"
        )
    )


def _static_import_alignment_is_supported(
    alignment: dict[str, Any],
    *,
    donor_entity: dict[str, Any],
) -> bool:
    if (
        str(alignment.get("status") or "")
        not in STRUCTURAL_ALIGNMENT_INSERTION_STATUSES
        or not alignment.get("conditional_context_aligned")
    ):
        return False
    old_between = [
        item
        for item in alignment.get("old_between") or []
        if isinstance(item, dict)
    ]
    if not old_between:
        return True
    if str(donor_entity.get("kind") or "") != "function":
        return False
    target_contract = str(donor_entity.get("contract_key") or "")
    return bool(
        target_contract
        and not alignment.get("verified_rename_candidate_ids")
        and all(
            str(item.get("contract_key") or "") != target_contract
            for item in old_between
        )
    )


def _conditional_static_import_plan(
    spec: CaseSpec,
    donor_entity: dict[str, Any],
) -> dict[str, Any] | None:
    """Prove that duplicate donor definitions form an insertable branch family."""

    relative_path = normalize_relpath(str(donor_entity.get("file") or ""))
    contract_key = str(
        donor_entity.get("contract_key") or donor_entity.get("key") or ""
    )
    entity_kind = str(donor_entity.get("kind") or "")
    if (
        not relative_path
        or not contract_key
        or not spec.resolve_old(relative_path).is_file()
        or not spec.resolve_donor(relative_path).is_file()
    ):
        return None
    donor_report = parse_structural_entity_file(
        root=spec.primary_donor_root,
        relative_path=relative_path,
        tree="donor_0",
    )
    old_report = parse_structural_entity_file(
        root=spec.old_root_path,
        relative_path=relative_path,
        tree="old",
    )
    donor_matches = [
        entity
        for entity in donor_report.get("entities") or []
        if str(entity.get("contract_key") or entity.get("key") or "")
        == contract_key
        and str(entity.get("kind") or "") == entity_kind
    ]
    if len(donor_matches) < 2:
        return None
    donor_lines = spec.resolve_donor(relative_path).read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines(keepends=True)
    old_lines = spec.resolve_old(relative_path).read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines(keepends=True)
    return conditional_variant_insertions(
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


def _exact_old_source_entity(
    spec: CaseSpec,
    donor_entity: dict[str, Any],
    *,
    structural_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    relative_path = normalize_relpath(str(donor_entity.get("file") or ""))
    extent = (
        donor_entity.get("extent")
        if isinstance(donor_entity.get("extent"), dict)
        else {}
    )
    start_line = int(extent.get("start_line") or 0)
    end_line = int(extent.get("end_line") or start_line)
    if not relative_path or start_line <= 0 or end_line < start_line:
        return None
    donor_path = spec.resolve_donor(relative_path)
    old_path = spec.resolve_old(relative_path)
    if not donor_path.is_file() or not old_path.is_file():
        return None
    donor_lines = donor_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines(keepends=True)
    if end_line > len(donor_lines):
        return None
    source = "".join(donor_lines[start_line - 1 : end_line])
    if not source.strip():
        return None
    old_text = old_path.read_text(encoding="utf-8", errors="ignore")
    match_count = old_text.count(source)
    exact_source_status = "unique_exact_entity_source_in_same_owner_file"
    if match_count == 1:
        offset = old_text.index(source)
        old_start_line = old_text.count("\n", 0, offset) + 1
        old_end_line = old_start_line + source.count("\n")
        if source.endswith(("\n", "\r")):
            old_end_line = max(old_start_line, old_end_line - 1)
    else:
        cache = structural_cache if structural_cache is not None else {}
        report = cache.get(relative_path)
        if report is None:
            report = parse_structural_entity_file(
                root=spec.old_root_path,
                relative_path=relative_path,
                tree="old",
            )
            cache[relative_path] = report
        donor_contract = str(donor_entity.get("contract_key") or "")
        donor_kind = str(donor_entity.get("kind") or "")
        donor_is_definition = bool(donor_entity.get("is_definition"))
        old_lines = old_text.splitlines(keepends=True)
        structural_matches: list[dict[str, Any]] = []
        for candidate in report.get("entities") or []:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("contract_key") or "") != donor_contract:
                continue
            if str(candidate.get("kind") or "") != donor_kind:
                continue
            if donor_is_definition and not bool(candidate.get("is_definition")):
                continue
            candidate_extent = dict(candidate.get("extent") or {})
            candidate_start = int(candidate_extent.get("start_line") or 0)
            candidate_end = int(
                candidate_extent.get("end_line") or candidate_start
            )
            if (
                candidate_start <= 0
                or candidate_end < candidate_start
                or candidate_end > len(old_lines)
            ):
                continue
            candidate_source = "".join(
                old_lines[candidate_start - 1 : candidate_end]
            )
            if candidate_source == source:
                structural_matches.append(candidate)
        if len(structural_matches) != 1:
            return None
        structural_match = structural_matches[0]
        structural_extent = dict(structural_match.get("extent") or {})
        old_start_line = int(structural_extent.get("start_line") or 0)
        old_end_line = int(
            structural_extent.get("end_line") or old_start_line
        )
        exact_source_status = (
            "unique_exact_structural_contract_in_same_owner_file"
        )
    entity_id = stable_id(
        "entity",
        "old",
        relative_path,
        str(donor_entity.get("contract_key") or ""),
        str(old_start_line),
        "exact-source",
    )
    return {
        **donor_entity,
        "id": entity_id,
        "entity_id": entity_id,
        "tree": "old",
        "extent": {
            **extent,
            "start_line": old_start_line,
            "end_line": old_end_line,
        },
        "exact_source_match": {
            "status": exact_source_status,
            "file": relative_path,
        },
    }


def _source_like(relative_path: str) -> bool:
    return Path(relative_path).suffix.lower() in SOURCE_SUFFIXES


def reference_is_source_contract(reference: dict[str, Any]) -> bool:
    """Return whether the recorded use is explicit in donor source."""

    evidence = (
        reference.get("source_spelling_evidence")
        if isinstance(reference.get("source_spelling_evidence"), dict)
        else {}
    )
    return (
        str(evidence.get("status") or "")
        != "different_identifier_at_source_location"
    )


def _group_forward_references(
    references: Iterable[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group consumers only when the typed graph resolves the same target entity."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for reference in references:
        record = dict(reference)
        target = (
            record.get("referenced_entity")
            if isinstance(record.get("referenced_entity"), dict)
            else {}
        )
        target_id = str(target.get("id") or record.get("referenced_entity_id") or "")
        reference_id = str(record.get("id") or "")
        key = (
            str(record.get("tree") or ""),
            str(record.get("dependency_scope") or ""),
            target_id or f"unresolved-reference:{reference_id}",
        )
        groups.setdefault(key, []).append(record)
    return list(groups.values())


def _compact_demand_entity(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        key: value.get(key)
        for key in (
            "id",
            "entity_id",
            "tree",
            "file",
            "name",
            "symbol",
            "kind",
            "entity_kind",
            "cursor_kind",
            "contract_key",
            "key",
            "usr",
            "is_definition",
            "extent",
            "container",
            "digest",
            "signature_fingerprint",
            "implementation_fingerprint",
            "interface_fingerprint",
            "syntax_fingerprint",
            "source_fingerprint",
            "interface",
            "storage_class",
            "linkage",
            "conditional_context",
        )
        if value.get(key) not in (None, "", [], {})
    }


def _compact_demand_reference(reference: dict[str, Any]) -> dict[str, Any]:

    compact = {
        key: reference.get(key)
        for key in (
            "id",
            "tree",
            "source_file",
            "symbol",
            "kind",
            "use_kind",
            "cursor_kind",
            "line",
            "column",
            "extent",
            "referenced_file",
            "referenced_kind",
            "dependency_scope",
            "container",
            "contract_key",
            "translation_unit_file",
            "source_spelling_evidence",
            "ast_path",
            "owner_resolution",
            "reference_context_fingerprint",
            "reference_context_depth",
        )
        if reference.get(key) not in (None, "", {})
    }
    owner = _compact_demand_entity(reference.get("owner_entity"))
    target = _compact_demand_entity(reference.get("referenced_entity"))
    if owner is not None:
        compact["owner_entity_id"] = owner.get("id")
        compact["owner_entity"] = owner
    if target is not None:
        compact["referenced_entity_id"] = target.get("id")
        compact["referenced_entity"] = target
    return compact


def _compact_semantic_demand_entity(
    value: object,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        key: value.get(key)
        for key in (
            "id",
            "entity_id",
            "tree",
            "file",
            "name",
            "symbol",
            "kind",
            "entity_kind",
            "contract_key",
            "key",
            "usr",
            "is_definition",
            "extent",
            "conditional_context",
        )
        if value.get(key) not in (None, "", [], {})
    }


def _compact_semantic_demand_reference(
    reference: dict[str, Any],
) -> dict[str, Any]:
    compact = {
        key: reference.get(key)
        for key in (
            "id",
            "tree",
            "source_file",
            "symbol",
            "kind",
            "use_kind",
            "cursor_kind",
            "line",
            "column",
            "extent",
            "referenced_file",
            "referenced_kind",
            "dependency_scope",
            "container",
            "contract_key",
            "translation_unit_file",
            "source_spelling_evidence",
            "ast_path",
            "owner_resolution",
            "reference_context_fingerprint",
            "reference_context_depth",
            "reference_use_contract",
            "reference_use_fingerprint",
        )
        if reference.get(key) not in (None, "", [], {})
    }
    owner = _compact_semantic_demand_entity(
        reference.get("owner_entity")
    )
    target = _compact_semantic_demand_entity(
        reference.get("referenced_entity")
    )
    if owner is not None:
        compact["owner_entity"] = owner
    if target is not None:
        compact["referenced_entity"] = target
    return compact


def compact_demand_ledger_record(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Keep the complete demand contract without duplicating parser state.

    The full source-analysis record is persisted separately by the workflow's
    content store. This descriptor contains every locator and requirement used
    by compile-valid semantic evaluation plus the initial disposition.
    """

    comparison = (
        evidence.get("observed_comparison")
        if isinstance(evidence.get("observed_comparison"), dict)
        else evidence.get("comparison")
        if isinstance(evidence.get("comparison"), dict)
        else {}
    )
    scalar_relations = {
        str(key): value
        for key, value in comparison.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    alignment = (
        evidence.get("alignment")
        if isinstance(evidence.get("alignment"), dict)
        else {}
    )
    compact_alignment = {
        str(key): value
        for key, value in alignment.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    references = [
        _compact_semantic_demand_reference(item)
        for item in evidence.get("consumer_references") or []
        if isinstance(item, dict)
    ]
    if not references and isinstance(
        evidence.get("consumer_reference"), dict
    ):
        references = [
            _compact_semantic_demand_reference(
                evidence["consumer_reference"]
            )
        ]

    compact = {
        key: evidence.get(key)
        for key in (
            "id",
            "contract_identity",
            "kind",
            "symbol",
            "dependency_scope",
            "semantic_requirement",
            "resolution",
            "initial_action_disposition",
            "depends_on_evidence_ids",
            "blocking_evidence_ids",
        )
        if evidence.get(key) not in (None, "", [], {})
    }
    compact["old_entity"] = _compact_semantic_demand_entity(
        evidence.get("old_entity")
    ) or {}
    compact["donor_entity"] = _compact_semantic_demand_entity(
        evidence.get("donor_entity")
    ) or {}
    compact["old_declarations"] = [
        descriptor
        for item in evidence.get("old_declarations") or []
        if (
            descriptor := _compact_semantic_demand_entity(item)
        ) is not None
    ]
    compact["donor_declarations"] = [
        descriptor
        for item in evidence.get("donor_declarations") or []
        if (
            descriptor := _compact_semantic_demand_entity(item)
        ) is not None
    ]
    compact["old_exports"] = [
        dict(item)
        for item in evidence.get("old_exports") or []
        if isinstance(item, dict)
    ]
    compact["donor_exports"] = [
        dict(item)
        for item in evidence.get("donor_exports") or []
        if isinstance(item, dict)
    ]
    compact["consumer_references"] = references
    if scalar_relations:
        compact["initial_observed_relations"] = scalar_relations
    if compact_alignment:
        compact["initial_alignment_summary"] = compact_alignment
    return compact


def _semantic_requirement(
    references: Iterable[dict[str, Any]],
    donor_entity: dict[str, Any] | None,
    *,
    integration_reference_required: bool | None = None,
) -> dict[str, Any]:
    reference_records = [
        item for item in references if isinstance(item, dict)
    ]
    use_kinds = sorted(
        {
            str(item.get("use_kind") or "")
            for item in reference_records
            if str(item.get("use_kind") or "")
        }
    )
    entity_kind = str((donor_entity or {}).get("kind") or "")
    dependency_scopes = sorted(
        {
            str(item.get("dependency_scope") or "")
            for item in reference_records
            if str(item.get("dependency_scope") or "")
        }
    )
    local_implementation_required = "local_driver" in dependency_scopes
    if entity_kind == "type":
        mode = "type_identity"
        dimensions = ["name"]
    elif entity_kind == "field":
        mode = "field_contract"
        dimensions = [
            "name",
            "structural_source",
        ]
    elif entity_kind == "macro":
        mode = "macro_value"
        dimensions = [
            "name",
            "structural_source",
        ]
    elif entity_kind == "enum_constant":
        mode = "enum_constant_value"
        dimensions = [
            "name",
            "structural_source",
        ]
    elif entity_kind == "variable":
        mode = f"{entity_kind}_value"
        dimensions = [
            "name",
            "signature",
            "interface",
            "declaration",
            "export",
            "module_boundary",
        ]
        if local_implementation_required:
            dimensions.append("implementation")
    else:
        mode = "callable_contract" if entity_kind == "function" else "entity_contract"
        dimensions = [
            "name",
            "signature",
            "interface",
            "declaration",
            "export",
            "module_boundary",
        ]
        if local_implementation_required:
            dimensions.append("implementation")
    if (
        reference_records
        and (
            integration_reference_required
            if integration_reference_required is not None
            else True
        )
    ):
        dimensions.append("integration_reference")
    return {
        "schema": "initial_semantic_requirement_v1",
        "mode": mode,
        "entity_kind": entity_kind,
        "use_kinds": use_kinds,
        "dependency_scopes": dependency_scopes,
        "local_implementation_required": local_implementation_required,
        "required_dimensions": list(dict.fromkeys(dimensions)),
        "demand_count": len(reference_records),
    }


def canonical_semantic_requirement(
    requirement: dict[str, Any],
) -> dict[str, Any]:
    """Normalize persisted contracts to dimensions their evidence can prove.

    Early semantic artifacts represented macros, fields, enum constants, and
    complete type layouts with callable-oriented signature/interface fields.
    A structural parser deliberately does not synthesize those callable
    fingerprints.  The complete definition source is the native contract for
    these entities.  Target-kernel providers are binding evidence for their
    consumers, not donor implementations to transplant into the target tree.
    """

    result = dict(requirement)
    entity_kind = str(result.get("entity_kind") or "").lower()
    mode = str(result.get("mode") or "")
    requested = [
        str(item)
        for item in result.get("required_dimensions") or []
        if str(item)
    ]
    integration_required = bool(
        "integration_reference" in requested
        or int(result.get("demand_count") or 0) > 0
    )
    dependency_scopes = {
        str(item)
        for item in result.get("dependency_scopes") or []
        if str(item)
    }
    use_kinds = {
        str(item)
        for item in result.get("use_kinds") or []
        if str(item)
    }

    if dependency_scopes == {"kernel_tree"}:
        # A backport consumes the target kernel's provider contract.  It does
        # not migrate that provider's donor implementation into the target.
        # A compile-valid consumer proves target binding; the separate
        # integration dimension still proves that the migrated driver kept
        # the relevant use edge.
        result["mode"] = "target_provider_binding"
        dimensions = ["consumer_compile_binding"]
        result.pop("evidence_dimensions", None)
    elif entity_kind == "function" and "local_driver" in dependency_scopes:
        # Direct calls retain their projected callable signature. For an
        # indirect target binding (for example a callback-table assignment),
        # the target declaration and successful consumer compile are the
        # authoritative compatibility contract; requiring the donor signature
        # as a second contract can repeatedly invalidate a compile-valid bind.
        # Function bodies remain covered by recursively projected demands.
        result["mode"] = "local_provider_target_binding"
        dimensions = [
            dimension
            for dimension in requested
            if dimension != "implementation"
        ]
        indirect_target_binding = bool(
            use_kinds and "call" not in use_kinds
        )
        if indirect_target_binding:
            dimensions = [
                dimension
                for dimension in dimensions
                if dimension != "signature"
            ]
        dimensions.append("consumer_compile_binding")
        result["signature_evidence_model"] = (
            "target_indirect_use_compile_binding"
            if indirect_target_binding
            else "projected_direct_call_contract_and_target_compile_binding"
        )
        result["implementation_evidence_model"] = (
            "transitive_demand_ledger"
        )
        result.pop("evidence_dimensions", None)
    elif mode == "type_identity":
        dimensions = ["name"]
    elif entity_kind in {
        "type",
        "field",
        "macro",
        "enum_constant",
    }:
        dimensions = ["name", "structural_source"]
    else:
        dimensions = list(requested)

    if integration_required:
        dimensions.append("integration_reference")
    result["required_dimensions"] = list(dict.fromkeys(dimensions))
    result["dimension_model"] = "target_binding_contract_v4"
    return result


def _baseline_old_root(spec: CaseSpec) -> Path | None:
    """Return the unique clean-old root mapped to the mutable target tree."""

    current_root = Path(spec.old_root_path).resolve()
    candidates: set[Path] = set()
    for old_prefix, new_prefix in spec.compile_path_maps:
        baseline = Path(str(old_prefix)).expanduser().resolve()
        mapped_current = Path(str(new_prefix)).expanduser().resolve()
        if (
            mapped_current == current_root
            and baseline != current_root
            and baseline.is_dir()
        ):
            candidates.add(baseline)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _baseline_provider_provenance(
    spec: CaseSpec,
    donor_entity: dict[str, Any] | None,
    *,
    structural_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Prove whether a demanded provider predates the migration.

    This provenance controls provider materialization decisions.  It does not
    discharge the separate obligation to retain each projected consumer use.
    """

    entity = donor_entity or {}
    relative_path = normalize_relpath(str(entity.get("file") or ""))
    contract_key = str(
        entity.get("contract_key") or entity.get("key") or ""
    )
    entity_kind = str(entity.get("kind") or "")
    symbol = str(entity.get("symbol") or entity.get("name") or "")
    baseline_root = _baseline_old_root(spec)
    base = {
        "schema": "baseline_provider_provenance_v1",
        "provider_file": relative_path or None,
        "provider_contract": contract_key or None,
        "provider_kind": entity_kind or None,
        "provider_symbol": symbol or None,
        "baseline_root": (
            str(baseline_root) if baseline_root is not None else None
        ),
    }
    if (
        baseline_root is None
        or not relative_path
        or not contract_key
        or not entity_kind
    ):
        return {
            **base,
            "status": "baseline_provider_provenance_unavailable",
        }
    baseline_path = baseline_root / relative_path
    if not baseline_path.is_file():
        return {
            **base,
            "status": "baseline_provider_file_absent",
            "provider_introduced_by_migration": True,
        }

    current_path = Path(spec.old_root_path).resolve() / relative_path
    if current_path.is_file():
        try:
            baseline_matches_current = filecmp.cmp(
                baseline_path,
                current_path,
                shallow=False,
            )
        except OSError:
            baseline_matches_current = False
        if baseline_matches_current:
            return {
                **base,
                "status": "baseline_provider_file_matches_current",
                "provider_introduced_by_migration": False,
                "proof": "byte_identical_baseline_and_current_provider_file",
            }

    cache_key = (str(baseline_root), relative_path)
    report = structural_cache.get(cache_key)
    if report is None:
        report = parse_structural_entity_file(
            root=baseline_root,
            relative_path=relative_path,
            tree="baseline_old",
        )
        structural_cache[cache_key] = report
    records = [
        item
        for item in report.get("entities") or []
        if isinstance(item, dict)
        and str(item.get("kind") or "") == entity_kind
    ]
    exact_matches = [
        item
        for item in records
        if str(item.get("contract_key") or item.get("key") or "")
        == contract_key
        and (
            not bool(entity.get("is_definition"))
            or bool(item.get("is_definition"))
        )
    ]
    if exact_matches:
        return {
            **base,
            "status": "baseline_provider_present",
            "provider_introduced_by_migration": False,
            "match_count": len(exact_matches),
            "baseline_entity_ids": sorted(
                str(item.get("id") or item.get("entity_id") or "")
                for item in exact_matches
                if str(item.get("id") or item.get("entity_id") or "")
            ),
        }

    same_symbol_matches = [
        item
        for item in records
        if symbol
        and str(item.get("symbol") or item.get("name") or "") == symbol
    ]
    if same_symbol_matches:
        return {
            **base,
            "status": "baseline_provider_identity_ambiguous",
            "provider_introduced_by_migration": None,
            "candidate_count": len(same_symbol_matches),
        }
    return {
        **base,
        "status": "baseline_provider_not_observed",
        "provider_introduced_by_migration": None,
        "reason": (
            "structural parsing did not observe the exact provider, but parser "
            "non-observation is not proof that an entity was absent"
        ),
    }


def _integration_reference_is_required(
    references: Iterable[dict[str, Any]],
) -> bool:
    return any(isinstance(item, dict) for item in references)


def _structural_source_relation(
    old_entity: dict[str, Any] | None,
    donor_entity: dict[str, Any] | None,
    structural_counterpart: dict[str, Any],
    structural_target: dict[str, Any],
) -> str:
    old_entity = old_entity or {}
    donor_entity = donor_entity or {}
    for field in ("source_fingerprint", "digest"):
        old_fingerprint = str(
            old_entity.get(field)
            or structural_counterpart.get(field)
            or ""
        )
        donor_fingerprint = str(
            donor_entity.get(field)
            or structural_target.get(field)
            or ""
        )
        if old_fingerprint and donor_fingerprint:
            return (
                "equivalent"
                if old_fingerprint == donor_fingerprint
                else "different"
            )
    return "unavailable"


_SOURCE_LEVEL_CONTRACT_DIMENSIONS = {
    "name",
    "signature",
    "implementation",
    "interface",
}


def _with_structural_source_evidence(
    requirement: dict[str, Any],
    *,
    old_entity: dict[str, Any] | None,
    donor_entity: dict[str, Any] | None,
    observed_comparison: dict[str, Any],
    structural_source_relation: str,
) -> dict[str, Any]:
    """Expose a complete changed definition when field fingerprints are absent."""

    result = dict(requirement)
    old = old_entity or {}
    donor = donor_entity or {}
    old_contract = str(
        old.get("contract_key") or old.get("key") or ""
    )
    donor_contract = str(
        donor.get("contract_key") or donor.get("key") or ""
    )
    if not (
        structural_source_relation == "different"
        and bool(old.get("is_definition"))
        and bool(donor.get("is_definition"))
        and str(old.get("kind") or "")
        == str(donor.get("kind") or "")
        and old_contract
        and old_contract == donor_contract
    ):
        return result

    required = {
        str(item)
        for item in result.get("required_dimensions") or []
        if str(item)
    }
    incompatible = {
        dimension
        for dimension in required
        if str(observed_comparison.get(dimension) or "")
        in {
            "different",
            "missing_in_old",
            "required_missing",
            "donor_only",
            "donor_contract_missing",
            "old_only",
        }
    }
    unresolved = {
        dimension
        for dimension in required
        if str(observed_comparison.get(dimension) or "unavailable")
        in {"", "unknown", "unavailable"}
    }
    if incompatible or not unresolved:
        return result
    if not unresolved.issubset(_SOURCE_LEVEL_CONTRACT_DIMENSIONS):
        return result

    result["evidence_dimensions"] = sorted(
        {
            str(item)
            for item in result.get("evidence_dimensions") or []
            if str(item)
        }
        | {"structural_source"}
    )
    return result


def _project_comparison_for_requirement(
    comparison: dict[str, Any],
    requirement: dict[str, Any],
) -> dict[str, Any]:
    projected = dict(comparison)
    decision_dimensions = {
        str(item)
        for item in [
            *list(requirement.get("required_dimensions") or []),
            *list(requirement.get("evidence_dimensions") or []),
        ]
        if str(item)
    }
    if "consumer_compile_binding" in decision_dimensions:
        projected["consumer_compile_binding"] = (
            "pending_compile_validation"
        )
    for dimension in (
        "name",
        "signature",
        "implementation",
        "interface",
        "declaration",
        "structural_source",
        "export",
        "module_boundary",
        "integration_reference",
    ):
        if dimension not in decision_dimensions:
            projected[dimension] = "not_required"
    projected["semantic_requirement"] = dict(requirement)
    return projected


def _reference_site_hunk_ids(
    hunks: Iterable[dict[str, Any]],
    reference: dict[str, Any],
) -> list[str]:
    extent = dict(reference.get("extent") or {})
    line = int(reference.get("line") or 0)
    if line > 0 and int(extent.get("start_line") or 0) <= 0:
        extent = {"start_line": line, "end_line": line}
    return hunks_overlapping_extent(
        hunks,
        tree=str(reference.get("tree") or "donor_0"),
        extent=extent,
    )


def _references_owned_by_entity(
    index: CodeIndex,
    *,
    tree: str,
    entity: dict[str, Any],
    preferred_translation_units: Iterable[object] = (),
) -> list[dict[str, Any]]:
    """Return typed dependencies occurring inside one provider entity."""
    relative_path = normalize_relpath(str(entity.get("file") or ""))
    entity_id = str(entity.get("id") or "")
    extent = dict(entity.get("extent") or {})
    start_line = int(extent.get("start_line") or 0)
    end_line = int(extent.get("end_line") or start_line)
    if not relative_path or not start_line:
        return []
    index.index_file(tree, relative_path)
    for translation_unit in _ordered_unique(preferred_translation_units):
        if Path(translation_unit).suffix.lower() not in {".c", ".cc", ".cpp"}:
            continue
        index.index_file_from_translation_unit(
            tree,
            relative_path,
            translation_unit,
        )

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reference in index.references_by_file.get((tree, relative_path), []):
        target = (
            reference.get("referenced_entity")
            if isinstance(reference.get("referenced_entity"), dict)
            else {}
        )
        if entity_id and str(target.get("id") or "") == entity_id:
            continue
        owner = (
            reference.get("owner_entity")
            if isinstance(reference.get("owner_entity"), dict)
            else {}
        )
        line = int(reference.get("line") or 0)
        if not (
            (entity_id and str(owner.get("id") or "") == entity_id)
            or start_line <= line <= end_line
        ):
            continue
        if not reference_crosses_entity_boundary(reference):
            continue
        identity = str(reference.get("id") or "")
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        result.append(reference)
    return result


def _typed_dependency_references_in_hunks(
    index: CodeIndex,
    *,
    tree: str,
    relative_path: str,
    owner_entity_id: str,
    hunks: Iterable[dict[str, Any]],
    hunk_ids: Iterable[object],
    exclude_reference_ids: Iterable[object] = (),
) -> list[dict[str, Any]]:
    """Return exact cross-entity dependencies introduced by selected hunks."""

    selected_hunk_ids = {str(item) for item in hunk_ids if str(item)}
    excluded = {str(item) for item in exclude_reference_ids if str(item)}
    if not relative_path or not owner_entity_id or not selected_hunk_ids:
        return []
    index.index_file(tree, relative_path)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reference in index.references_by_file.get((tree, relative_path), []):
        reference_id = str(reference.get("id") or "")
        if reference_id in excluded:
            continue
        owner = (
            reference.get("owner_entity")
            if isinstance(reference.get("owner_entity"), dict)
            else {}
        )
        if str(owner.get("id") or reference.get("owner_entity_id") or "") != owner_entity_id:
            continue
        if not reference_crosses_entity_boundary(reference):
            continue
        if not (
            set(_reference_site_hunk_ids(hunks, reference))
            & selected_hunk_ids
        ):
            continue
        identity = reference_id or stable_id(
            "hunk-dependency-reference",
            tree,
            relative_path,
            str(reference.get("line") or 0),
            str(reference.get("column") or 0),
            str(reference.get("contract_key") or reference.get("symbol") or ""),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(reference)
    return result


def _direct_donor_target_is_usable(
    donor_target: dict[str, Any] | None,
    *,
    dependency_scope: str,
) -> bool:
    if not donor_target:
        return False
    has_source_extent = bool(
        normalize_relpath(str(donor_target.get("file") or ""))
        and int((donor_target.get("extent") or {}).get("start_line") or 0) > 0
    )
    if not has_source_extent:
        return False
    kind = str(donor_target.get("kind") or "")
    if (
        dependency_scope != "kernel_tree"
        and kind == "function"
        and not bool(donor_target.get("is_definition"))
    ):
        return False
    return bool(
        dependency_scope == "kernel_tree"
        or kind in DIRECT_PROVIDER_ENTITY_KINDS
    )


def _missing_provider_needs_dependency_closure(
    *,
    old_entity: dict[str, Any] | None,
    donor_entity: dict[str, Any] | None,
    resolution: str,
    provider_frontier_files: Iterable[object],
) -> bool:
    provider_file = normalize_relpath(
        str((donor_entity or {}).get("file") or "")
    )
    frontier = set(_ordered_unique(provider_frontier_files))
    return bool(
        old_entity is None
        and donor_entity
        and donor_entity.get("id")
        and bool(donor_entity.get("is_definition"))
        and provider_file in frontier
        and (
            resolution == "requires_llm"
            or resolution.startswith("deterministic_")
        )
    )


def _externally_addressable_seed_entities(
    entities_by_file: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    def strictly_contains(
        outer: dict[str, Any],
        inner: dict[str, Any],
    ) -> bool:
        outer_start = (
            int(outer.get("start_line") or 0),
            int(outer.get("start_column") or 0),
        )
        outer_end = (
            int(outer.get("end_line") or 0),
            int(outer.get("end_column") or 0),
        )
        inner_start = (
            int(inner.get("start_line") or 0),
            int(inner.get("start_column") or 0),
        )
        inner_end = (
            int(inner.get("end_line") or 0),
            int(inner.get("end_column") or 0),
        )
        return bool(
            outer_start[0] > 0
            and inner_start[0] > 0
            and outer_start <= inner_start
            and inner_end <= outer_end
            and (outer_start, outer_end)
            != (inner_start, inner_end)
        )

    result: list[dict[str, Any]] = []
    for entities in entities_by_file.values():
        function_extents = [
            dict(entity.get("extent") or {})
            for entity in entities
            if str(entity.get("kind") or "") == "function"
            and bool(entity.get("is_definition"))
        ]
        for entity in entities:
            if not entity.get("symbol"):
                continue
            kind = str(entity.get("kind") or "")
            if (
                kind in {"function", "variable"}
                and entity_is_internally_linked(entity)
            ):
                continue
            if str(entity.get("cursor_kind") or "") == "VAR_DECL":
                extent = dict(entity.get("extent") or {})
                start = int(extent.get("start_line") or 0)
                end = int(extent.get("end_line") or start)
                if any(
                    int(owner.get("start_line") or 0)
                    <= start
                    <= end
                    <= int(owner.get("end_line") or 0)
                    for owner in function_extents
                    if start > 0 and int(owner.get("start_line") or 0) > 0
                ):
                    continue
            if kind == "type" and not bool(entity.get("is_definition")):
                extent = dict(entity.get("extent") or {})
                entity_id = str(entity.get("id") or "")
                if any(
                    str(owner.get("id") or "") != entity_id
                    and strictly_contains(
                        dict(owner.get("extent") or {}),
                        extent,
                    )
                    for owner in entities
                ):
                    continue
            result.append(entity)
    return result


def _aligned_old_reference(
    index: CodeIndex,
    donor_reference: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    source_file = normalize_relpath(str(donor_reference.get("source_file") or ""))
    if not source_file:
        return None, {
            "status": "unavailable",
            "equivalence_proven": False,
            "reason": "donor reference has no source file",
        }
    index.index_file("old", source_file)
    old_references = index.references_by_file.get(("old", source_file), [])
    donor_target = (
        donor_reference.get("referenced_entity")
        if isinstance(donor_reference.get("referenced_entity"), dict)
        else {}
    )
    donor_target_contract = str(donor_target.get("contract_key") or "")
    donor_target_kind = str(donor_target.get("kind") or "")
    donor_owner = (
        donor_reference.get("owner_entity")
        if isinstance(donor_reference.get("owner_entity"), dict)
        else {}
    )
    donor_owner_contract = str(
        donor_owner.get("contract_key") or ""
    )
    donor_use_kind = str(donor_reference.get("use_kind") or "")

    def target_matches(
        candidates: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            candidate
            for candidate in candidates
            if isinstance(candidate.get("referenced_entity"), dict)
            and str((candidate.get("referenced_entity") or {}).get("contract_key") or "")
            == donor_target_contract
            and str((candidate.get("referenced_entity") or {}).get("kind") or "")
            == donor_target_kind
            and str(
                (candidate.get("owner_entity") or {}).get(
                    "contract_key"
                )
                or ""
            )
            == donor_owner_contract
            and str(candidate.get("use_kind") or "")
            == donor_use_kind
        ]

    context_fingerprint = str(
        donor_reference.get("reference_context_fingerprint") or ""
    )
    exact_context_matches = target_matches(
        reference
        for reference in old_references
        if context_fingerprint
        and str(
            reference.get("reference_context_fingerprint") or ""
        )
        == context_fingerprint
    )
    if exact_context_matches:
        selected = min(
            exact_context_matches,
            key=lambda item: (
                int(item.get("line") or 0),
                int(item.get("column") or 0),
                str(item.get("id") or ""),
            ),
        )
        return selected, {
            "status": (
                "exact_ast_reference_context"
                if donor_owner_contract
                and int(
                    donor_reference.get(
                        "reference_context_depth"
                    )
                    or 0
                )
                > 0
                else "exact_typed_file_reference_edge"
            ),
            "equivalence_proven": True,
            "source_file": source_file,
            "reference_context_fingerprint": context_fingerprint,
            "owner_contract": donor_owner_contract,
            "referenced_contract": donor_target_contract,
            "candidate_count": len(exact_context_matches),
        }

    owner_target_matches = target_matches(old_references)
    selected_owner_target = (
        min(
            owner_target_matches,
            key=lambda item: (
                int(item.get("line") or 0),
                int(item.get("column") or 0),
                str(item.get("id") or ""),
            ),
        )
        if owner_target_matches
        else None
    )
    if selected_owner_target is not None and donor_use_kind != "call":
        return selected_owner_target, {
            "status": (
                "typed_owner_target_reference_edge"
                if donor_owner_contract
                else "typed_file_target_reference_edge"
            ),
            "equivalence_proven": True,
            "source_file": source_file,
            "owner_contract": donor_owner_contract,
            "referenced_contract": donor_target_contract,
            "candidate_count": len(owner_target_matches),
        }
    if selected_owner_target is not None:
        return selected_owner_target, {
            "status": (
                "typed_owner_target_match_without_exact_ast_context"
            ),
            "equivalence_proven": False,
            "source_file": source_file,
            "owner_contract": donor_owner_contract,
            "referenced_contract": donor_target_contract,
            "candidate_count": len(owner_target_matches),
            "reason": (
                "the same owner calls the same contract, but its AST "
                "position and control context are not proven equivalent"
            ),
        }
    return None, {
        "status": "unavailable",
        "equivalence_proven": False,
        "reason": (
            "consumer reference context is missing or ambiguous in the "
            "old tree"
        ),
        "exact_context_candidate_count": len(
            exact_context_matches
        ),
        "owner_target_candidate_count": len(
            owner_target_matches
        ),
    }


def _direct_migration_seed_files(
    spec: CaseSpec,
    copied_files: Iterable[object],
) -> list[str]:
    return _ordered_unique([*spec.entry_files, *copied_files])


def _forward_references_from_demand_roots(
    index: CodeIndex,
    *,
    tree: str,
    demand_seed_files: Iterable[object],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for relative_path in _ordered_unique(demand_seed_files):
        references.extend(
            reference
            for reference in index.references_by_file.get(
                (tree, relative_path), []
            )
            if reference_crosses_entity_boundary(reference)
        )
    return references


def _case_compile_source_frontier(
    index: CodeIndex,
    *,
    tree: str,
) -> dict[str, Any]:
    files = _ordered_unique(index.compile_command_files(tree))
    return {
        "schema": "case_compile_source_frontier_v1",
        "tree": tree,
        "status": "ready" if files else "empty",
        "files": files,
        "summary": {"file_count": len(files)},
    }


def _case_compile_provider_frontier(
    index: CodeIndex,
    *,
    tree: str,
) -> dict[str, Any]:
    compile_source_frontier = _case_compile_source_frontier(
        index,
        tree=tree,
    )
    compile_sources = _ordered_unique(
        compile_source_frontier.get("files") or []
    )
    dependency_frontier = index.compile_dependency_frontier(
        tree,
        within_root="",
    )
    dependency_files = (
        dependency_frontier.get("files") or []
        if str(dependency_frontier.get("status") or "") == "ready"
        else []
    )
    files = _ordered_unique([*compile_sources, *dependency_files])
    return {
        "schema": "case_compile_provider_frontier_v1",
        "tree": tree,
        "status": "ready" if files else "empty",
        "files": files,
        "dependency_frontier_status": dependency_frontier.get("status"),
        "dependency_frontier_reason": dependency_frontier.get("reason"),
        "summary": {
            "compile_source_file_count": len(compile_sources),
            "dependency_file_count": len(
                _ordered_unique(dependency_files)
            ),
            "file_count": len(files),
        },
    }


def _tree_entities(index: CodeIndex, tree: str, symbol: str) -> list[dict[str, Any]]:
    return canonical_entity_records(
        record
        for record in index.entities_by_name.get(symbol, [])
        if str(record.get("tree") or "") == tree
    )


def _query_symbol_in_frontier(
    index: CodeIndex,
    *,
    tree: str,
    symbol: str,
    candidate_files: Iterable[object],
    cache: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
    force_frontier_scan: bool = False,
) -> list[dict[str, Any]]:
    cache_key = (tree, symbol)
    if (
        not force_frontier_scan
        and cache is not None
        and cache_key in cache
    ):
        return cache[cache_key]

    existing = _tree_entities(index, tree, symbol)
    existing_definitions = [
        record
        for record in existing
        if bool(record.get("is_definition"))
        and int((record.get("extent") or {}).get("start_line") or 0) > 0
    ]
    if len(existing_definitions) == 1 and not force_frontier_scan:
        if cache is not None:
            cache[cache_key] = existing
        return existing

    frontier = _ordered_unique(candidate_files)
    candidate_report = index.exact_token_candidates_in_files(
        tree,
        [symbol],
        frontier,
    )
    matched_files = _ordered_unique(
        (candidate_report.get("candidates_by_symbol") or {}).get(
            symbol,
            [],
        )
    )
    for relative_path in matched_files:
        index.index_file(tree, relative_path)
        file_definitions = [
            record
            for record in _tree_entities(index, tree, symbol)
            if bool(record.get("is_definition"))
            and normalize_relpath(str(record.get("file") or ""))
            == relative_path
            and int((record.get("extent") or {}).get("start_line") or 0) > 0
        ]
        if not file_definitions:
            index.materialize_structural_definition_candidates(
                tree,
                relative_path,
                [symbol],
            )
    records = _tree_entities(index, tree, symbol)
    if cache is not None:
        cache[cache_key] = records
    return records


def _function_like_macro_source(
    index: CodeIndex,
    entity: dict[str, Any],
) -> bool:
    if str(entity.get("kind") or "") != "macro":
        return False
    spec = getattr(index, "spec", None)
    resolve_donor = getattr(spec, "resolve_donor", None)
    relative_path = normalize_relpath(str(entity.get("file") or ""))
    symbol = str(entity.get("symbol") or "")
    extent = entity.get("extent") if isinstance(entity.get("extent"), dict) else {}
    start_line = int(extent.get("start_line") or 0)
    end_line = int(extent.get("end_line") or start_line)
    if (
        not callable(resolve_donor)
        or not relative_path
        or not symbol
        or start_line <= 0
        or end_line < start_line
    ):
        return False
    source_path = resolve_donor(relative_path)
    if not source_path.is_file():
        return False
    lines = source_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines()
    if end_line > len(lines):
        return False
    source = "\n".join(lines[start_line - 1 : end_line])
    return bool(
        re.search(
            rf"^[ \t]*#[ \t]*define[ \t]+{re.escape(symbol)}\(",
            source,
            flags=re.MULTILINE,
        )
    )


def _target_provider_in_frontier(
    index: CodeIndex,
    *,
    tree: str,
    donor_entity: dict[str, Any],
    candidate_files: Iterable[object],
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    """Resolve a target provider moved or represented differently by version.

    Candidate discovery is bounded by the case's real compile/include
    frontier. Exact contracts are preferred. A representation change is
    accepted only for a function-like macro backed by one callable target, or
    for one field with the same concrete type fingerprint. Identifier uses and
    ambiguous candidates are never promoted to providers.
    """

    symbol = str(donor_entity.get("symbol") or "")
    contract_key = str(donor_entity.get("contract_key") or "")
    entity_kind = str(donor_entity.get("kind") or "")
    if not symbol or not contract_key or not entity_kind:
        return None, []

    records = _query_symbol_in_frontier(
        index,
        tree=tree,
        symbol=symbol,
        candidate_files=candidate_files,
        cache=None,
        force_frontier_scan=True,
    )
    require_definition = bool(donor_entity.get("is_definition"))
    exact_matches = canonical_entity_records(
        record
        for record in records
        if str(record.get("contract_key") or "") == contract_key
        and str(record.get("kind") or "") == entity_kind
        and (
            not require_definition
            or bool(record.get("is_definition"))
        )
        and int((record.get("extent") or {}).get("start_line") or 0)
        > 0
    )
    if exact_matches:
        return (
            exact_matches[0] if len(exact_matches) == 1 else None
        ), exact_matches

    concrete_records = canonical_entity_records(
        record
        for record in records
        if str(record.get("symbol") or "") == symbol
        and int((record.get("extent") or {}).get("start_line") or 0)
        > 0
    )
    compatible_matches: list[dict[str, Any]] = []
    if entity_kind == "macro" and _function_like_macro_source(
        index,
        donor_entity,
    ):
        compatible_matches = [
            record
            for record in concrete_records
            if str(record.get("kind") or "") == "function"
        ]
    elif entity_kind == "field":
        signature_fingerprint = str(
            donor_entity.get("signature_fingerprint") or ""
        )
        if signature_fingerprint:
            compatible_matches = [
                record
                for record in concrete_records
                if str(record.get("kind") or "") == "field"
                and str(record.get("signature_fingerprint") or "")
                == signature_fingerprint
            ]
    return (
        compatible_matches[0]
        if len(compatible_matches) == 1
        else None
    ), compatible_matches


def _unique_definition(
    records: Iterable[dict[str, Any]],
    *,
    preferred_file: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    definitions = [
        record
        for record in canonical_entity_records(records)
        if bool(record.get("is_definition"))
        and int((record.get("extent") or {}).get("start_line") or 0) > 0
    ]
    if preferred_file:
        preferred = [
            record
            for record in definitions
            if normalize_relpath(str(record.get("file") or "")) == normalize_relpath(preferred_file)
        ]
        if len(preferred) == 1:
            return preferred[0], definitions
    if len(definitions) == 1:
        return definitions[0], definitions
    return None, definitions


def _definition_backed_provider(
    index: CodeIndex,
    *,
    tree: str,
    symbol: str,
    endpoint: dict[str, Any] | None,
    preferred_file: str | None,
    candidate_files: Iterable[object],
    cache: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve a declaration endpoint to its unique provider definition.

    A reference endpoint often denotes a declaration in a header.  That node
    is useful consumer evidence, but an implementation-bearing contract must
    be compared at its definition and must not be hidden by the declaration.
    The resolution uses exact symbol/USR records within the case compile
    frontier and remains unresolved when more than one definition survives.
    """

    queried = _query_symbol_in_frontier(
        index,
        tree=tree,
        symbol=symbol,
        candidate_files=candidate_files,
        cache=cache,
    )
    records = canonical_entity_records(
        [
            *([endpoint] if isinstance(endpoint, dict) else []),
            *queried,
        ]
    )
    definitions = [
        record
        for record in records
        if bool(record.get("is_definition"))
        and int((record.get("extent") or {}).get("start_line") or 0) > 0
    ]

    endpoint_usr = str((endpoint or {}).get("usr") or "")
    if endpoint_usr:
        usr_matches = [
            record
            for record in definitions
            if str(record.get("usr") or "") == endpoint_usr
        ]
        if len(usr_matches) == 1:
            return usr_matches[0], records, definitions

    definition, _ = _unique_definition(
        definitions,
        preferred_file=preferred_file,
    )
    return definition, records, definitions


def _matching_declarations(
    records: Iterable[dict[str, Any]],
    definition: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if definition is None:
        return []
    expected = str(definition.get("signature_fingerprint") or "")
    return [
        record
        for record in canonical_entity_records(records)
        if not bool(record.get("is_definition"))
        and (
            not expected
            or str(record.get("signature_fingerprint") or "") == expected
        )
    ]


def _same_source_entity_proof(
    old_entity: dict[str, Any] | None,
    donor_entity: dict[str, Any] | None,
    identical_source_files: set[str],
) -> dict[str, Any] | None:
    if old_entity is None or donor_entity is None:
        return None
    old_file = normalize_relpath(str(old_entity.get("file") or ""))
    donor_file = normalize_relpath(str(donor_entity.get("file") or ""))
    if not old_file or old_file != donor_file or old_file not in identical_source_files:
        return None
    if str(old_entity.get("kind") or "") != str(donor_entity.get("kind") or ""):
        return None
    if str(old_entity.get("contract_key") or "") != str(donor_entity.get("contract_key") or ""):
        return None
    return {
        "status": "exact_source_file_entity",
        "file": old_file,
        "proof": "old and donor source SHA-256 values are identical for the same typed entity",
    }


def _declaration_relation(
    *,
    old_declarations: list[dict[str, Any]],
    donor_declarations: list[dict[str, Any]],
    identical_source_files: set[str],
    dependency_scope: str,
) -> tuple[str, dict[str, Any] | None]:
    if dependency_scope == "kernel_tree":
        return "not_required", None
    donor_files = {
        normalize_relpath(str(declaration.get("file") or ""))
        for declaration in donor_declarations
        if declaration.get("file")
    }
    if (
        old_declarations
        and donor_declarations
        and donor_files
        and donor_files <= identical_source_files
    ):
        return (
            "equivalent",
            {
                "status": "exact_declaration_source_files",
                "files": sorted(donor_files),
                "proof": "donor declaration source files exist with identical SHA-256 values in old",
            },
        )
    if old_declarations and donor_declarations:
        old_fingerprints = {
            str(item.get("signature_fingerprint") or "")
            for item in old_declarations
            if str(item.get("signature_fingerprint") or "")
        }
        donor_fingerprints = {
            str(item.get("signature_fingerprint") or "")
            for item in donor_declarations
            if str(item.get("signature_fingerprint") or "")
        }
        if old_fingerprints and donor_fingerprints:
            return (
                "equivalent"
                if old_fingerprints == donor_fingerprints
                else "different",
                None,
            )
        return "unavailable", None
    if donor_declarations and not old_declarations:
        return "missing_in_old", None
    return "unavailable", None


def _structural_rename_candidate(
    index: CodeIndex,
    *,
    donor_entity: dict[str, Any],
    old_candidate_files: Iterable[object],
    query_cache: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    donor_file = normalize_relpath(str(donor_entity.get("file") or ""))
    if not donor_file:
        return None, {"status": "unavailable", "reason": "donor provider has no owner file"}
    donor_report = index.index_structural_file("donor_0", donor_file)
    old_report = index.index_structural_file("old", donor_file)
    if not donor_report.get("parse_ok") or not old_report.get("parse_ok"):
        return None, {
            "status": "unavailable",
            "reason": "old/donor structural owner parsing failed",
            "old_parse_error": old_report.get("parse_error"),
            "donor_parse_error": donor_report.get("parse_error"),
        }
    contract = str(donor_entity.get("contract_key") or "")
    kind = str(donor_entity.get("kind") or "")
    donor_targets = [
        record
        for record in donor_report.get("entities") or []
        if str(record.get("contract_key") or "") == contract
        and str(record.get("kind") or "") == kind
    ]
    if len(donor_targets) != 1:
        return None, {
            "status": "unavailable",
            "reason": "donor provider does not map to one structural entity",
            "match_count": len(donor_targets),
        }
    alignment = structural_entity_neighbor_alignment(
        target_entity_id=str(donor_targets[0].get("id") or ""),
        donor_entities=donor_report.get("entities") or [],
        old_entities=old_report.get("entities") or [],
    )
    rename_candidates: list[dict[str, Any]] = []
    replacement_candidates: list[dict[str, Any]] = []
    donor_between = [
        item
        for item in alignment.get("donor_between") or []
        if isinstance(item, dict)
    ]
    old_between = [
        item
        for item in alignment.get("old_between") or []
        if isinstance(item, dict)
    ]
    exact_bounded_slot = bool(
        alignment.get("predecessor")
        and alignment.get("successor")
        and len(donor_between) == 1
        and len(old_between) == 1
        and str(donor_between[0].get("id") or "")
        == str(donor_targets[0].get("id") or "")
    )
    for structural in alignment.get("old_between") or []:
        if str(structural.get("kind") or "") != kind:
            continue
        symbol = str(structural.get("symbol") or "")
        if not symbol:
            continue
        records = _query_symbol_in_frontier(
            index,
            tree="old",
            symbol=symbol,
            candidate_files=old_candidate_files,
            cache=query_cache,
        )
        definition, _all_definitions = _unique_definition(records, preferred_file=donor_file)
        if definition is None:
            continue
        comparison = compare_entity_fingerprints(definition, donor_entity)
        if (
            comparison.get("signature") == "equivalent"
            and comparison.get("implementation") == "equivalent"
        ):
            rename_candidates.append(definition)
        elif (
            exact_bounded_slot
            and comparison.get("signature") == "equivalent"
            and str(old_between[0].get("id") or "")
            == str(structural.get("id") or "")
        ):
            # This is correspondence, not equivalence: the two definitions
            # occupy the only entity slot between the same unique anchors and
            # expose compatible types.  Their implementation difference remains
            # a demanded LLM decision in one unit.
            replacement_candidates.append(definition)
    alignment["verified_rename_candidate_ids"] = [
        str(record.get("id") or "") for record in rename_candidates
    ]
    alignment["verified_replacement_candidate_ids"] = [
        str(record.get("id") or "") for record in replacement_candidates
    ]
    if len(rename_candidates) == 1:
        alignment["status"] = "verified_structural_rename"
        return rename_candidates[0], alignment
    if rename_candidates:
        alignment["status"] = "ambiguous_verified_structural_rename"
        return None, alignment
    if len(replacement_candidates) == 1:
        alignment["status"] = "verified_structural_replacement"
        alignment["proof"] = (
            "one signature-compatible old and donor entity occupy the unique "
            "slot between the same ordered structural anchors"
        )
        return replacement_candidates[0], alignment
    if replacement_candidates:
        alignment["status"] = "ambiguous_structural_replacement"
    return None, alignment


def _structural_old_counterpart(
    alignment: dict[str, Any],
) -> dict[str, Any] | None:
    counterpart = (
        alignment.get("old_counterpart")
        if isinstance(alignment.get("old_counterpart"), dict)
        else None
    )
    if counterpart is None:
        return None
    return {
        **counterpart,
        "tree": "old",
        "is_definition": True,
        "backend": "tree_sitter_c_structural_entities_v2",
    }


class _BuildContractLookup:
    def __init__(self, spec: CaseSpec) -> None:
        self.spec = spec
        self._module_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._module_source_cache: dict[tuple[str, str], list[str]] = {}
        self._kbuild_graph_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._export_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def _root(self, tree: str) -> Path:
        return self.spec.old_root_path if tree == "old" else self.spec.primary_donor_root

    def modules(self, tree: str, source_file: str) -> list[dict[str, Any]]:
        source_file = normalize_relpath(source_file)
        key = (tree, source_file)
        if key not in self._module_cache:
            ownership = module_ownership_for_sources(
                self._root(tree),
                [source_file],
                tree=tree,
            )
            self._module_cache[key] = list(ownership.get(source_file) or [])
        return self._module_cache[key]

    def module_sources(self, tree: str, source_file: str) -> list[str]:
        source_file = normalize_relpath(source_file)
        key = (tree, source_file)
        if key in self._module_source_cache:
            return list(self._module_source_cache[key])
        directory = normalize_relpath(Path(source_file).parent.as_posix())
        graph_key = (tree, directory)
        if graph_key not in self._kbuild_graph_cache:
            self._kbuild_graph_cache[graph_key] = KbuildIndex.from_directory(
                self._root(tree), directory
            ).build_contract_graph()
        graph = self._kbuild_graph_cache[graph_key]
        module_to_sources = dict(graph.get("module_to_sources") or {})
        owning_modules = [
            module
            for module, sources in module_to_sources.items()
            if source_file in _ordered_unique(sources)
        ]
        selected_modules = owning_modules or list(module_to_sources)
        sources = _ordered_unique(
            source
            for module in selected_modules
            for source in module_to_sources.get(module) or []
            if _source_like(str(source or ""))
        )
        self._module_source_cache[key] = sources
        return list(sources)

    def exports(self, tree: str, source_file: str, symbol: str) -> list[dict[str, Any]]:
        source_file = normalize_relpath(source_file)
        key = (tree, source_file)
        if key not in self._export_cache:
            indexed = index_export_contracts(self._root(tree), [source_file], tree=tree)
            for indexed_symbol, records in indexed.items():
                self._export_cache[(tree, f"{source_file}:{indexed_symbol}")] = records
            self._export_cache[key] = []
        return list(self._export_cache.get((tree, f"{source_file}:{symbol}")) or [])


def _module_names(records: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({normalize_relpath(str(record.get("module") or "")) for record in records if record.get("module")})


def _target_module_boundary(
    *,
    dependency_scope: str,
    donor_consumer_modules: Iterable[object],
    old_consumer_modules: Iterable[object],
    donor_provider_modules: Iterable[object],
    old_provider_modules: Iterable[object],
    provider_internal_linkage: bool = False,
) -> dict[str, Any]:
    """Resolve module visibility in the tree that will actually be compiled."""

    donor_consumers = _ordered_unique(donor_consumer_modules)
    old_consumers = _ordered_unique(old_consumer_modules)
    donor_providers = _ordered_unique(donor_provider_modules)
    old_providers = _ordered_unique(old_provider_modules)
    target_consumers = old_consumers or donor_consumers
    target_providers = old_providers or donor_providers
    if dependency_scope == "kernel_tree" or provider_internal_linkage:
        relation = "not_required"
    elif target_consumers and target_providers:
        relation = (
            "same_module"
            if set(target_consumers) <= set(target_providers)
            else "cross_module"
        )
    else:
        relation = "unknown"
    return {
        "relation": relation,
        "target_consumer_modules": target_consumers,
        "target_provider_modules": target_providers,
        "consumer_evidence_tree": (
            "old_target_kbuild"
            if old_consumers
            else "donor_kbuild"
            if donor_consumers
            else "unavailable"
        ),
        "provider_evidence_tree": (
            "old_target_kbuild"
            if old_providers
            else "donor_kbuild"
            if donor_providers
            else "unavailable"
        ),
        "relation_authority": (
            "provider_entity_internal_linkage"
            if provider_internal_linkage
            else "kbuild_module_ownership"
        ),
    }


def _old_consumer_impact(
    build_lookup: _BuildContractLookup,
    old_entity: dict[str, Any] | None,
    *,
    exclude_files: Iterable[object] = (),
) -> dict[str, Any]:
    if not old_entity or not old_entity.get("id") or not old_entity.get("file"):
        return {
            "status": "unavailable",
            "source_files": [],
            "references": [],
        }
    excluded = set(_ordered_unique(exclude_files))
    source_files = [
        source
        for source in build_lookup.module_sources("old", str(old_entity["file"]))
        if source not in excluded
    ]
    return {
        "status": "deferred_to_configured_module_validation",
        "source_files": source_files,
        "references": [],
    }


def _export_relation(
    *,
    old_exports: list[dict[str, Any]],
    donor_exports: list[dict[str, Any]],
    module_boundary: str,
) -> str:
    return export_contract_relation(
        old_exports,
        donor_exports,
        required=module_boundary == "cross_module",
    )


def _resolution_for_evidence(
    *,
    old_entity: dict[str, Any] | None,
    donor_entity: dict[str, Any] | None,
    comparison: dict[str, Any],
    semantic_requirement: dict[str, Any],
    old_declarations: list[dict[str, Any]],
    donor_exports: list[dict[str, Any]],
    old_provider_ambiguous: bool,
    static_import_ready: bool,
    structural_source_relation: str,
    dependency_scope: str = "local_driver",
    integration_reference_provenance: dict[str, Any] | None = None,
) -> str:
    required_dimensions = {
        str(item)
        for item in semantic_requirement.get("required_dimensions") or []
        if str(item)
    }
    if (
        dependency_scope != "kernel_tree"
        and str((donor_entity or {}).get("kind") or "") == "function"
        and not bool((old_entity or {}).get("is_definition"))
    ):
        old_entity = None
    if _post_materialization_integration_validation_only(
        old_entity=old_entity,
        donor_entity=donor_entity,
        comparison=comparison,
        semantic_requirement=semantic_requirement,
        integration_reference_provenance=(
            integration_reference_provenance
        ),
    ):
        return "post_materialization_validation_only"
    if old_entity is None:
        if old_provider_ambiguous:
            return "blocked_ambiguous_old_provider"
        if donor_entity is None:
            return "blocked_missing_evidence"
        if static_import_ready:
            # A consumer edge cannot resolve in the target graph before its
            # provider exists. Exact materialization is the action; integration
            # is a postcondition revalidated against the resulting tree.
            return "deterministic_import_missing_contract"
        return "requires_llm"
    signature = str(comparison.get("signature") or "unavailable")
    implementation = str(comparison.get("implementation") or "unavailable")
    declaration = str(comparison.get("declaration") or "unavailable")
    export = str(comparison.get("export") or "not_required")
    module_boundary = str(comparison.get("module_boundary") or "unknown")
    projected_structural_source = str(
        comparison.get("structural_source") or "unavailable"
    )
    exact_structural_contract = projected_structural_source == "equivalent"
    if (
        (signature == "equivalent" or exact_structural_contract)
        and (implementation == "equivalent" or exact_structural_contract)
        and comparison.get("export") == "missing_in_old"
        and module_boundary == "cross_module"
        and not entity_is_internally_linked(old_entity)
        and bool(old_declarations)
        and bool(donor_exports)
    ):
        return "deterministic_export_only"
    if "consumer_compile_binding" in required_dimensions:
        # Compilation is the authority for target binding.  A missing use edge
        # remains in the semantic ledger, but it is not a provider repair
        # before a compile-valid current tree exists.
        pending_validation = {
            dimension: str(comparison.get(dimension) or "unavailable")
            for dimension in required_dimensions
            if dimension
            not in {"consumer_compile_binding", "integration_reference"}
        }
        if any(
            relation
            in {
                "different",
                "missing_in_old",
                "required_missing",
                "donor_only",
                "donor_contract_missing",
                "old_only",
            }
            for relation in pending_validation.values()
        ):
            return "requires_llm"
        if any(
            relation in {"", "unknown", "unavailable"}
            for relation in pending_validation.values()
        ):
            return "blocked_missing_evidence"
        return "post_materialization_validation_only"
    if required_dimensions:
        relations = {
            dimension: str(comparison.get(dimension) or "unavailable")
            for dimension in required_dimensions
        }
        incompatible = {
            dimension: relation
            for dimension, relation in relations.items()
            if relation
            in {
                "different",
                "missing_in_old",
                "required_missing",
                "donor_only",
                "donor_contract_missing",
                "old_only",
            }
        }
        if incompatible:
            return "requires_llm"

        unresolved = {
            dimension: relation
            for dimension, relation in relations.items()
            if relation in {"", "unknown", "unavailable"}
        }
        unresolved = {
            dimension: relation
            for dimension, relation in unresolved.items()
            if not (
                exact_structural_contract
                and dimension
                in {"name", "signature", "implementation", "interface"}
            )
            and not (
                dimension == "declaration"
                and relation == "unavailable"
                and old_entity is not None
            )
        }
        if unresolved:
            evidence_dimensions = {
                str(item)
                for item in semantic_requirement.get(
                    "evidence_dimensions"
                )
                or []
                if str(item)
            }
            if (
                structural_source_relation == "different"
                and "structural_source" in evidence_dimensions
                and set(unresolved).issubset(
                    _SOURCE_LEVEL_CONTRACT_DIMENSIONS
                )
            ):
                return "requires_llm"
            return "blocked_missing_evidence"

        satisfied_relations = {
            "equivalent",
            "not_required",
            "same_module",
            "cross_module",
        }
        if all(
            relation in satisfied_relations
            for relation in relations.values()
        ):
            return "satisfied"

    relevant = (signature, implementation, declaration, export)
    if (
        module_boundary == "same_module"
        and all(value in {"equivalent", "not_required", "unavailable"} for value in relevant)
    ):
        return "satisfied"
    if (
        module_boundary == "not_required"
        and (signature == "equivalent" or exact_structural_contract)
        and declaration in {"equivalent", "not_required", "unavailable"}
    ):
        return "satisfied"
    if (
        module_boundary == "cross_module"
        and export == "equivalent"
        and (signature == "equivalent" or exact_structural_contract)
        and (
            implementation in {"equivalent", "unavailable"}
            or exact_structural_contract
        )
        and declaration in {"equivalent", "unavailable"}
    ):
        return "satisfied"
    if (
        module_boundary == "unknown"
        and (signature == "equivalent" or exact_structural_contract)
        and (
            implementation in {"equivalent", "unavailable"}
            or exact_structural_contract
        )
    ):
        return "blocked_missing_evidence"
    return "requires_llm"


def _post_materialization_integration_validation_only(
    *,
    old_entity: dict[str, Any] | None,
    donor_entity: dict[str, Any] | None,
    comparison: dict[str, Any],
    semantic_requirement: dict[str, Any],
    integration_reference_provenance: dict[str, Any] | None,
) -> bool:
    """Defer an isolated use-edge gap until current-tree validation.

    Provider compatibility and consumer integration have different owners.
    When every provider dimension is already satisfied, a missing source-side
    use is retained in the demand ledger and re-evaluated after compilation;
    it must not turn the provider into a speculative initial repair target.
    """

    del donor_entity, integration_reference_provenance
    if old_entity is None:
        return False
    required_dimensions = {
        str(item)
        for item in semantic_requirement.get("required_dimensions") or []
        if str(item)
    }
    if not (
        "integration_reference" in required_dimensions
        and str(comparison.get("integration_reference") or "")
        == "missing_in_old"
    ):
        return False

    satisfied_relations = {
        "equivalent",
        "not_required",
        "same_module",
        "cross_module",
    }
    for dimension in required_dimensions - {
        "consumer_compile_binding",
        "integration_reference",
    }:
        relation = str(comparison.get(dimension) or "unavailable")
        if relation in satisfied_relations:
            continue
        if (
            dimension == "declaration"
            and relation == "unavailable"
            and old_entity is not None
        ):
            continue
        return False
    return True


def _post_materialization_validation_requirement(
    *,
    old_entity: dict[str, Any] | None,
    donor_entity: dict[str, Any] | None,
    consumer_references: Iterable[dict[str, Any]],
    comparison: dict[str, Any],
    semantic_requirement: dict[str, Any],
    static_import_ready: bool,
    integration_reference_provenance: dict[str, Any] | None = None,
    unproven_consumer_reference_ids: Iterable[object] = (),
) -> dict[str, Any] | None:
    required_dimensions = {
        str(item)
        for item in semantic_requirement.get("required_dimensions") or []
        if str(item)
    }
    validation_only = _post_materialization_integration_validation_only(
        old_entity=old_entity,
        donor_entity=donor_entity,
        comparison=comparison,
        semantic_requirement=semantic_requirement,
        integration_reference_provenance=(
            integration_reference_provenance
        ),
    )
    if not (
        "integration_reference" in required_dimensions
        and (
            (old_entity is None and static_import_ready)
            or validation_only
        )
    ):
        return None
    unproven_ids = {
        str(item)
        for item in unproven_consumer_reference_ids
        if str(item)
    }
    selected_references = [
        reference
        for reference in consumer_references
        if isinstance(reference, dict)
        and (
            not unproven_ids
            or str(reference.get("id") or "") in unproven_ids
        )
    ]
    provenance = (
        dict(integration_reference_provenance)
        if isinstance(integration_reference_provenance, dict)
        else {}
    )
    return {
        "schema": "post_materialization_validation_requirement_v1",
        "phase": "after_structural_materialization",
        "required_dimensions": ["integration_reference"],
        "target_contract_key": str(
            (donor_entity or {}).get("contract_key")
            or (donor_entity or {}).get("key")
            or ""
        )
        or None,
        "consumer_reference_ids": sorted(
            {
                str(reference.get("id") or "")
                for reference in selected_references
                if isinstance(reference, dict)
                and str(reference.get("id") or "")
            }
        ),
        "consumer_references": [
            _compact_demand_reference(reference)
            for reference in selected_references
        ],
        "pre_materialization_observation": {
            "integration_reference": str(
                comparison.get("integration_reference") or "unavailable"
            ),
        },
        "materialization_provenance": provenance or None,
        "validation_authority": (
            "typed_current_tree_code_graph_and_compile_validation"
        ),
    }


def _raw_hunks_for_file(
    cache: dict[str, list[dict[str, Any]]],
    *,
    spec: CaseSpec,
    relative_path: str,
) -> list[dict[str, Any]]:
    relative_path = normalize_relpath(relative_path)
    if relative_path in cache:
        return cache[relative_path]
    old_path = spec.resolve_old(relative_path)
    donor_path = spec.resolve_donor(relative_path)
    if not old_path.is_file() or not donor_path.is_file():
        cache[relative_path] = []
    else:
        cache[relative_path] = structured_line_hunks(
            old_text=old_path.read_text(encoding="utf-8", errors="ignore"),
            donor_text=donor_path.read_text(encoding="utf-8", errors="ignore"),
            file=relative_path,
        )
    return cache[relative_path]


def analyze_initial_compatibility(
    spec: CaseSpec,
    *,
    initial_copy_set: Iterable[object],
) -> dict[str, Any]:
    started_at = time.monotonic()
    copied_files = _ordered_unique(initial_copy_set)
    donor_tree = "donor_0"
    file_manifest = build_file_change_manifest(
        old_root=spec.old_root_path,
        donor_root=spec.primary_donor_root,
        driver_root=spec.driver_root,
        originally_missing=copied_files,
    )
    identical_source_files = {
        normalize_relpath(str(record.get("file") or ""))
        for record in file_manifest.get("files") or []
        if record.get("old_sha256")
        and record.get("old_sha256") == record.get("donor_sha256")
    }
    copied_exact_files = {
        normalize_relpath(str(record.get("file") or ""))
        for record in file_manifest.get("files") or []
        if record.get("presence") == "donor_only_transplanted"
        and normalize_relpath(str(record.get("file") or "")) in identical_source_files
    }
    manifest_completed_at = time.monotonic()
    changed_shared_files = [
        str(record["file"])
        for record in file_manifest.get("files") or []
        if record.get("presence") == "shared_changed" and _source_like(str(record.get("file") or ""))
    ]
    explicit_seed_files = _direct_migration_seed_files(spec, copied_files)
    analysis_seed_files = list(explicit_seed_files)

    # Initial compatibility decisions consume complete parser records and
    # build their own contract/change-atom graph. Mirroring the same records
    # into a second occurrence graph only duplicates evidence.
    index = CodeIndex(spec, graph_projection="records")
    index.index_files(donor_tree, analysis_seed_files)
    index.index_files("old", analysis_seed_files)
    reverse_build_frontier = index.compile_dependency_frontier(
        donor_tree,
        within_root=spec.driver_root,
    )
    seed_index_completed_at = time.monotonic()
    donor_definition_frontier = _case_compile_source_frontier(
        index,
        tree=donor_tree,
    )
    old_definition_frontier = _case_compile_source_frontier(
        index,
        tree="old",
    )
    old_provider_frontier = _case_compile_provider_frontier(
        index,
        tree="old",
    )

    seed_entities = _externally_addressable_seed_entities(
        {
            relative_path: index.entities_by_file.get((donor_tree, relative_path), [])
            for relative_path in explicit_seed_files
        }
    )
    seed_entity_ids = {str(entity.get("id") or "") for entity in seed_entities}
    donor_seed_parse_blockers = []
    for relative_path in analysis_seed_files:
        if Path(relative_path).suffix.lower() not in {".c", ".cc", ".cpp"}:
            continue
        report = index.parse_reports.get((donor_tree, relative_path)) or {}
        fatal_diagnostics = [
            diagnostic
            for diagnostic in report.get("diagnostics") or []
            if int(diagnostic.get("severity") or 0) >= 4
        ]
        if not bool(report.get("parse_ok")) or fatal_diagnostics:
            blocker = {
                "tree": donor_tree,
                "file": relative_path,
                "parse_error": report.get("parse_error"),
                "parse_exception": report.get("parse_exception"),
                "fatal_diagnostics": fatal_diagnostics,
            }
            donor_seed_parse_blockers.append(blocker)
    preparation_blockers = list(donor_seed_parse_blockers)
    if str(reverse_build_frontier.get("status") or "") != "ready":
        preparation_blockers.append(
            {
                "tree": donor_tree,
                "stage": "reverse_integration_build_frontier",
                "reason": reverse_build_frontier.get("reason")
                or "compile_dependency_frontier_unavailable",
                "frontier": reverse_build_frontier,
            }
        )
    if preparation_blockers:
        completed_at = time.monotonic()
        return {
            "schema": INITIAL_COMPATIBILITY_ANALYSIS_SCHEMA,
            "analysis_revision": INITIAL_COMPATIBILITY_ANALYSIS_REVISION,
            "case_id": spec.case_id,
            "generated_at": utc_now(),
            "status": "preparation_incomplete",
            "preparation_complete": False,
            "preparation_blockers": preparation_blockers,
            "demand_ledger": [],
            "contract_evidence": [],
            "change_atoms": [],
            "raw_hunks": [],
            "initial_difference_graph": build_initial_difference_graph(
                contract_evidence=[],
                change_atoms=[],
            ),
            "summary": {
                "preparation_complete": False,
                "preparation_blocker_count": len(preparation_blockers),
                "seed_entity_count": len(seed_entities),
                "evaluated_contract_count": 0,
                "demand_ledger_count": 0,
                "contract_evidence_count": 0,
                "post_materialization_validation_count": 0,
                "satisfied_count": 0,
                "deterministic_count": 0,
                "requires_llm_count": 0,
                "blocked_count": 0,
            },
            "timings_sec": {
                "file_manifest": round(manifest_completed_at - started_at, 6),
                "seed_index": round(seed_index_completed_at - manifest_completed_at, 6),
                "reverse_candidate_index": 0.0,
                "contract_resolution": 0.0,
                "total": round(completed_at - started_at, 6),
            },
        }
    build_frontier_files = set(
        _ordered_unique(reverse_build_frontier.get("files") or [])
    )
    seed_symbols = sorted(
        {
            str(entity.get("symbol") or "")
            for entity in seed_entities
            if str(entity.get("symbol") or "")
        }
    )
    reverse_search_frontier = sorted(
        build_frontier_files.intersection(changed_shared_files)
    )
    exact_candidates = index.exact_token_candidates_in_files(
        donor_tree,
        seed_symbols,
        reverse_search_frontier,
    )
    reverse_integration_candidate_files = _ordered_unique(
        relative_path
        for files in dict(
            exact_candidates.get("candidates_by_symbol") or {}
        ).values()
        for relative_path in files
        if normalize_relpath(str(relative_path))
        not in set(explicit_seed_files)
    )
    reverse_probe_reports = index.query_reference_files_to_entities(
        donor_tree,
        reverse_integration_candidate_files,
        seed_entity_ids,
    )
    for report in reverse_probe_reports:
        index.add_reference_probe_report(report)
    reverse_integration_files = _ordered_unique(
        report.get("file")
        for report in reverse_probe_reports
        if report.get("references")
    )
    index.index_files("old", reverse_integration_files)
    reverse_index_completed_at = time.monotonic()
    build_lookup = _BuildContractLookup(spec)
    hunk_cache: dict[str, list[dict[str, Any]]] = {}
    contract_evidence: list[dict[str, Any]] = []
    atoms: list[dict[str, Any]] = []
    seen_contracts: set[str] = set()
    symbol_query_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    definition_frontier_files = {
        donor_tree: _ordered_unique(
            donor_definition_frontier.get("files") or []
        ),
        "old": _ordered_unique(old_definition_frontier.get("files") or []),
    }
    old_provider_frontier_files = _ordered_unique(
        old_provider_frontier.get("files") or []
    )
    exact_source_structural_cache: dict[str, dict[str, Any]] = {}
    baseline_structural_cache: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    queue: deque[dict[str, Any]] = deque()
    queued_reference_tasks: set[tuple[str, str]] = set()
    expanded_missing_provider_ids: set[str] = set()
    discovered_contract_dependencies: set[tuple[str, str]] = set()
    root_contract_identities: set[str] = set()
    transitive_reference_count = 0

    def enqueue_references(
        references: Iterable[dict[str, Any]],
        *,
        direction: str,
        discovered_from_contract: str = "",
    ) -> None:
        selected: list[dict[str, Any]] = []
        for reference in references:
            if not isinstance(reference, dict):
                continue
            record = dict(reference)
            if not reference_is_source_contract(record):
                continue
            selected.append(record)
        if not selected:
            return
        reference = selected[0]
        reference_id = str(
            reference.get("id")
            or stable_id(
                "initial-reference",
                direction,
                str(reference.get("source_file") or ""),
                str(reference.get("line") or 0),
                str(reference.get("column") or 0),
                str(reference.get("contract_key") or reference.get("symbol") or ""),
            )
        )
        group_identity = stable_id(
            "initial-reference-group",
            direction,
            *sorted(
                str(item.get("id") or "")
                for item in selected
                if str(item.get("id") or "")
            ),
        )
        key = (direction, group_identity or reference_id)
        if key in queued_reference_tasks:
            return
        queued_reference_tasks.add(key)
        queue.append(
            {
                "reference": reference,
                "references": selected,
                "direction": direction,
                "discovered_from_contract": discovered_from_contract,
            }
        )

    initial_forward_references = _forward_references_from_demand_roots(
        index,
        tree=donor_tree,
        demand_seed_files=analysis_seed_files,
    )
    for references in _group_forward_references(initial_forward_references):
        enqueue_references(references, direction="forward")

    reverse_reference_groups: dict[
        tuple[str, str, tuple[str, ...]],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for relative_path in reverse_integration_files:
        for reference in index.references_by_file.get((donor_tree, relative_path), []):
            target = (
                reference.get("referenced_entity")
                if isinstance(reference.get("referenced_entity"), dict)
                else None
            )
            if target is None or str(target.get("id") or "") not in seed_entity_ids:
                continue
            owner = reference.get("owner_entity") if isinstance(reference.get("owner_entity"), dict) else None
            if owner is None or str(owner.get("id") or "") in seed_entity_ids:
                continue
            owner_file = normalize_relpath(
                str(owner.get("file") or reference.get("source_file") or "")
            )
            hunks = _raw_hunks_for_file(
                hunk_cache,
                spec=spec,
                relative_path=owner_file,
            )
            site_hunk_ids = tuple(
                sorted(_reference_site_hunk_ids(hunks, reference))
            )
            group_key = (
                str(owner.get("contract_key") or owner.get("id") or ""),
                owner_file,
                site_hunk_ids
                or (f"reference:{str(reference.get('id') or '')}",),
            )
            reverse_reference_groups[group_key].append(reference)
    for references in reverse_reference_groups.values():
        enqueue_references(
            references,
            direction="reverse_integration",
        )

    while queue:
        task = queue.popleft()
        reference = dict(task.get("reference") or {})
        task_references = [
            dict(item)
            for item in task.get("references") or [reference]
            if isinstance(item, dict)
        ]
        symbol = str(reference.get("symbol") or "").strip()
        if not symbol:
            continue
        direction = str(task.get("direction") or "forward")
        discovered_from_contract = str(task.get("discovered_from_contract") or "")
        consumer_file = normalize_relpath(str(reference.get("source_file") or ""))
        consumer_files = _ordered_unique(
            normalize_relpath(str(item.get("source_file") or ""))
            for item in task_references
            if str(item.get("source_file") or "")
        )
        consumer_modules = sorted(
            {
                module
                for path in consumer_files
                for module in _module_names(build_lookup.modules(donor_tree, path))
            }
        )
        old_consumer_modules = sorted(
            {
                module
                for path in consumer_files
                for module in _module_names(build_lookup.modules("old", path))
            }
        )

        if direction == "reverse_integration":
            donor_owner = reference.get("owner_entity") if isinstance(reference.get("owner_entity"), dict) else None
            if donor_owner is None:
                continue
            owner_contract = str(donor_owner.get("contract_key") or donor_owner.get("id") or "")
            reference_ids = sorted(
                str(item.get("id") or "")
                for item in task_references
                if str(item.get("id") or "")
            )
            contract_identity = (
                f"integration:{owner_contract}:sites:"
                f"{stable_id('integration-sites', *reference_ids)}"
            )
            if not discovered_from_contract:
                root_contract_identities.add(contract_identity)
            if contract_identity in seen_contracts:
                continue
            seen_contracts.add(contract_identity)
            owner_file = normalize_relpath(str(donor_owner.get("file") or consumer_file))
            index.index_file("old", owner_file)
            old_candidates = [
                entity
                for entity in index.entities_by_file.get(("old", owner_file), [])
                if str(entity.get("contract_key") or "") == owner_contract
                and bool(entity.get("is_definition")) == bool(donor_owner.get("is_definition"))
            ]
            old_owner = old_candidates[0] if len(old_candidates) == 1 else None
            if not old_candidates:
                continue
            reference_alignments: list[dict[str, Any]] = []
            missing_references: list[dict[str, Any]] = []
            for donor_reference in task_references:
                aligned_reference, alignment_record = _aligned_old_reference(
                    index,
                    donor_reference,
                )
                reference_alignments.append(
                    {
                        "donor_reference_id": donor_reference.get("id"),
                        "donor_reference": donor_reference,
                        "old_reference": aligned_reference,
                        "alignment": alignment_record,
                    }
                )
                if not bool(
                    alignment_record.get("equivalence_proven")
                ):
                    missing_references.append(donor_reference)
            integration_relation = (
                "equivalent"
                if not missing_references
                else "missing_in_old"
            )
            comparison = {
                "name": (
                    "equivalent"
                    if old_owner and old_owner.get("symbol") == donor_owner.get("symbol")
                    else "missing_in_old"
                ),
                "declaration": "not_required",
                "export": "not_required",
                "module_boundary": "not_required",
                "integration_reference": integration_relation,
            }
            hunks = _raw_hunks_for_file(hunk_cache, spec=spec, relative_path=owner_file)
            owner_hunk_ids = hunks_overlapping_extent(
                hunks,
                tree=donor_tree,
                extent=(
                    donor_owner.get("extent")
                    if isinstance(donor_owner.get("extent"), dict)
                    else None
                ),
            )
            raw_hunk_ids = sorted(
                {
                    hunk_id
                    for hunk_id in [
                        *owner_hunk_ids,
                        *(
                            hunk_id
                            for donor_reference in task_references
                            for hunk_id in _reference_site_hunk_ids(
                                hunks,
                                donor_reference,
                            )
                        ),
                    ]
                }
            )
            resolution = (
                "satisfied"
                if old_owner is not None and not missing_references
                else "requires_llm"
                if raw_hunk_ids
                else "blocked_missing_evidence"
            )
            integration_hunks = [
                hunk
                for hunk in hunks
                if str(hunk.get("id") or "") in set(raw_hunk_ids)
            ]
            # Dependencies inside a donor hunk become migration demands only
            # when that hunk remains a candidate for application.  Expanding
            # an already-equivalent or blocked integration change leaves
            # child contracts whose parent will never enter the plan.
            direct_dependency_references = (
                _typed_dependency_references_in_hunks(
                    index,
                    tree=donor_tree,
                    relative_path=owner_file,
                    owner_entity_id=str(donor_owner.get("id") or ""),
                    hunks=hunks,
                    hunk_ids=raw_hunk_ids,
                    exclude_reference_ids=reference_ids,
                )
                if resolution == "requires_llm"
                else []
            )
            evidence_id = stable_id(
                "contract-evidence",
                contract_identity,
            )
            semantic_requirement = {
                "schema": "initial_semantic_requirement_v1",
                "mode": "incoming_integration_reference",
                "entity_kind": str(donor_owner.get("kind") or ""),
                "use_kinds": sorted(
                    {
                        str(item.get("use_kind") or "")
                        for item in task_references
                        if str(item.get("use_kind") or "")
                    }
                ),
                "required_dimensions": ["integration_reference"],
                "demand_count": len(task_references),
            }
            evidence_record = {
                "id": evidence_id,
                "contract_identity": contract_identity,
                "kind": "reverse_integration",
                "symbol": symbol,
                "consumer_reference": _compact_demand_reference(reference),
                "consumer_references": [
                    _compact_demand_reference(item)
                    for item in task_references
                ],
                "consumer_modules": consumer_modules,
                "old_entity": old_owner,
                "donor_entity": donor_owner,
                "comparison": comparison,
                "observed_comparison": dict(comparison),
                "semantic_requirement": semantic_requirement,
                "resolution": resolution,
                "alignment": {
                    "status": "exact_contract" if old_owner else "missing_or_ambiguous_old_owner",
                    "old_candidate_count": len(old_candidates),
                    "reference_alignments": reference_alignments,
                    "missing_reference_count": len(missing_references),
                },
                "old_declarations": [],
                "donor_declarations": [],
                "old_exports": [],
                "donor_exports": [],
                "raw_hunk_ids": raw_hunk_ids,
                "integration_hunks": integration_hunks,
                "direct_dependency_references": direct_dependency_references,
                "depends_on_evidence_ids": [],
            }
            evidence_atoms = build_change_atoms(
                contract_identity=contract_identity,
                comparison=comparison,
                old_entity=old_owner,
                donor_entity=donor_owner,
                raw_hunk_ids=raw_hunk_ids,
                contract_evidence_id=evidence_id,
                semantic_requirement=semantic_requirement,
            )
            evidence_record["change_atom_ids"] = [
                str(item["id"]) for item in evidence_atoms
            ]
            atoms.extend(evidence_atoms)
            contract_evidence.append(evidence_record)
            for dependency_reference in direct_dependency_references:
                before = len(queued_reference_tasks)
                enqueue_references(
                    [dependency_reference],
                    direction="forward",
                    discovered_from_contract=contract_identity,
                )
                transitive_reference_count += int(
                    len(queued_reference_tasks) > before
                )
            continue

        dependency_scope = str(reference.get("dependency_scope") or "")
        donor_target = (
            reference.get("referenced_entity")
            if isinstance(reference.get("referenced_entity"), dict)
            else None
        )
        preferred_file = normalize_relpath(
            str((donor_target or {}).get("file") or reference.get("referenced_file") or "")
        )
        aligned_reference_records = [
            (item, *_aligned_old_reference(index, item))
            for item in task_references
        ]
        aligned_old_reference = aligned_reference_records[0][1]
        reference_alignment = aligned_reference_records[0][2]
        consumer_reference_alignments = [
            {
                "donor_reference_id": item.get("id"),
                "old_reference_id": (old_reference or {}).get("id"),
                "alignment": alignment_record,
            }
            for item, old_reference, alignment_record in aligned_reference_records
        ]
        unproven_consumer_reference_ids = sorted(
            {
                str(item.get("id") or "")
                for item, _old_reference, alignment_record
                in aligned_reference_records
                if str(item.get("id") or "")
                and not bool(
                    alignment_record.get("equivalence_proven")
                )
            }
        )
        integration_reference_relation = (
            "equivalent"
            if task_references
            and not unproven_consumer_reference_ids
            else "missing_in_old"
            if task_references
            else "not_required"
        )
        aligned_old_targets = canonical_entity_records(
            old_reference.get("referenced_entity")
            for _item, old_reference, _alignment in aligned_reference_records
            if isinstance(old_reference, dict)
            and isinstance(old_reference.get("referenced_entity"), dict)
        )
        aligned_old_target = (
            aligned_old_reference.get("referenced_entity")
            if isinstance(aligned_old_reference, dict)
            and isinstance(aligned_old_reference.get("referenced_entity"), dict)
            else None
        )

        alignment: dict[str, Any] = {}
        old_provider_ambiguous = False
        direct_donor_target = _direct_donor_target_is_usable(
            donor_target,
            dependency_scope=dependency_scope,
        )
        definition_backed_donor_records: list[dict[str, Any]] | None = None
        donor_kind = str((donor_target or {}).get("kind") or "")
        definition_backed_kind = (
            dependency_scope != "kernel_tree"
            and donor_kind in {"function", "variable"}
        )
        if (
            direct_donor_target
            and definition_backed_kind
        ):
            resolved_donor, resolved_records, _resolved_definitions = (
                _definition_backed_provider(
                    index,
                    tree=donor_tree,
                    symbol=symbol,
                    endpoint=donor_target,
                    preferred_file=normalize_relpath(
                        str((donor_target or {}).get("file") or "")
                    )
                    or None,
                    candidate_files=definition_frontier_files[donor_tree],
                    cache=symbol_query_cache,
                )
            )
            if (
                resolved_donor is None
                and dependency_scope != "kernel_tree"
            ):
                direct_donor_target = False
            elif resolved_donor is not None:
                donor_target = resolved_donor
                definition_backed_donor_records = resolved_records
        if direct_donor_target:
            donor_entity = donor_target
            donor_records = definition_backed_donor_records or [donor_entity]
            provider_definition_contract = (
                str(donor_entity.get("kind") or "")
                in {"function", "variable"}
                and bool(donor_entity.get("is_definition"))
            )
            target_file = normalize_relpath(str(donor_entity.get("file") or ""))
            old_provider_lookup = ""
            if provider_definition_contract:
                old_entity, old_records, old_definitions = (
                    _definition_backed_provider(
                        index,
                        tree="old",
                        symbol=symbol,
                        endpoint=aligned_old_target,
                        preferred_file=target_file or None,
                        candidate_files=definition_frontier_files["old"],
                        cache=symbol_query_cache,
                    )
                )
                old_matches = old_definitions
                old_provider_ambiguous = bool(
                    len(old_definitions) > 1
                    and not bool((old_entity or {}).get("is_definition"))
                )
                old_provider_lookup = "definition_backed_provider"
            elif aligned_old_target is not None:
                old_matches = [aligned_old_target]
                old_entity = aligned_old_target
                old_records = old_matches
                old_provider_lookup = "aligned_consumer_reference"
            else:
                exact_old_entity = (
                    _exact_old_source_entity(
                        spec,
                        donor_entity,
                        structural_cache=exact_source_structural_cache,
                    )
                    if dependency_scope == "kernel_tree"
                    else None
                )
                if exact_old_entity is not None:
                    old_matches = [exact_old_entity]
                    old_entity = exact_old_entity
                    old_provider_lookup = (
                        "exact_source_in_same_owner_file"
                    )
                else:
                    index.index_file("old", target_file)
                    old_matches = [
                        entity
                        for entity in index.entities_by_file.get(("old", target_file), [])
                        if str(entity.get("contract_key") or "")
                        == str(donor_entity.get("contract_key") or "")
                    ]
                    old_entity = old_matches[0] if len(old_matches) == 1 else None
                    old_provider_lookup = (
                        "exact_contract_in_same_owner_file"
                        if old_entity is not None
                        else "unresolved_in_same_owner_file"
                    )
                    if not old_matches and dependency_scope == "kernel_tree":
                        old_entity, old_matches = (
                            _target_provider_in_frontier(
                                index,
                                tree="old",
                                donor_entity=donor_entity,
                                candidate_files=old_provider_frontier_files,
                            )
                        )
                        if old_entity is None:
                            old_provider_lookup = (
                                "unresolved_in_target_compile_dependency_frontier"
                            )
                        elif (
                            str(old_entity.get("kind") or "")
                            == str(donor_entity.get("kind") or "")
                            and str(old_entity.get("contract_key") or "")
                            == str(donor_entity.get("contract_key") or "")
                        ):
                            old_provider_lookup = (
                                "exact_contract_in_target_compile_dependency_frontier"
                            )
                        else:
                            old_provider_lookup = (
                                "compatible_target_representation_in_compile_dependency_frontier"
                            )
                old_provider_ambiguous = len(old_matches) > 1
                old_records = old_matches
            if (
                provider_definition_contract
                and bool((old_entity or {}).get("is_definition"))
            ):
                alignment_status = (
                    "definition_backed_consumer_reference_contract"
                )
            elif (
                provider_definition_contract
                and aligned_old_target is not None
            ):
                alignment_status = (
                    "aligned_consumer_declaration_without_provider_definition"
                )
            elif aligned_old_target is not None:
                alignment_status = "aligned_consumer_reference_contract"
            elif (old_entity or {}).get("exact_source_match"):
                alignment_status = (
                    "unique_exact_entity_source_in_same_owner_file"
                )
            elif old_provider_lookup == (
                "exact_contract_in_target_compile_dependency_frontier"
            ):
                alignment_status = (
                    "exact_reference_contract_in_target_compile_dependency_frontier"
                )
            elif old_provider_lookup == (
                "compatible_target_representation_in_compile_dependency_frontier"
            ):
                alignment_status = (
                    "compatible_target_representation_in_compile_dependency_frontier"
                )
            elif old_entity is not None:
                alignment_status = "exact_reference_contract"
            else:
                alignment_status = "missing_reference_contract"
            alignment = {
                "status": alignment_status,
                "old_candidate_count": len(old_matches),
                "old_declaration_candidate_count": int(
                    aligned_old_target is not None
                    and not bool(aligned_old_target.get("is_definition"))
                ),
                "consumer_reference_alignment": reference_alignment,
                "consumer_reference_alignments": consumer_reference_alignments,
                "old_provider_lookup": old_provider_lookup,
            }
            if old_entity is None and not old_matches:
                _rename, structural_alignment = _structural_rename_candidate(
                    index,
                    donor_entity=donor_entity,
                    old_candidate_files=definition_frontier_files["old"],
                    query_cache=symbol_query_cache,
                )
                if structural_alignment.get("status") != "unavailable":
                    alignment = {
                        **alignment,
                        "structural_alignment": structural_alignment,
                    }
                    structural_counterpart = _structural_old_counterpart(
                        structural_alignment
                    )
                    if structural_counterpart is not None:
                        old_entity = structural_counterpart
                        old_records = [old_entity]
        else:
            queried_local_provider = bool(
                dependency_scope != "kernel_tree"
                and str((donor_target or {}).get("kind") or "")
                in {"function", "variable"}
            )
            if queried_local_provider:
                donor_entity, donor_records, donor_definitions = (
                    _definition_backed_provider(
                        index,
                        tree=donor_tree,
                        symbol=symbol,
                        endpoint=donor_target,
                        preferred_file=preferred_file or None,
                        candidate_files=definition_frontier_files[donor_tree],
                        cache=symbol_query_cache,
                    )
                )
            else:
                donor_records = _query_symbol_in_frontier(
                    index,
                    tree=donor_tree,
                    symbol=symbol,
                    candidate_files=definition_frontier_files[donor_tree],
                    cache=symbol_query_cache,
                )
                donor_entity, donor_definitions = _unique_definition(
                    donor_records,
                    preferred_file=preferred_file or None,
                )
            if donor_entity is None:
                contract_identity = f"unresolved:{reference.get('contract_key') or symbol}:{consumer_file}"
                if contract_identity in seen_contracts:
                    continue
                seen_contracts.add(contract_identity)
                evidence_id = stable_id(
                    "contract-evidence",
                    contract_identity,
                )
                contract_evidence.append(
                    {
                        "id": evidence_id,
                        "contract_identity": contract_identity,
                        "kind": "unresolved_provider",
                        "symbol": symbol,
                        "consumer_reference": reference,
                        "consumer_modules": consumer_modules,
                        "old_entity": None,
                        "donor_entity": None,
                        "comparison": {"provider": "ambiguous_or_missing_in_donor"},
                        "resolution": "blocked_missing_evidence",
                        "alignment": {"donor_definition_count": len(donor_definitions)},
                        "change_atom_ids": [],
                        "raw_hunk_ids": [],
                        "depends_on_evidence_ids": [],
                    }
                )
                continue
            local_definition_contract = bool(
                dependency_scope != "kernel_tree"
                and str(donor_entity.get("kind") or "")
                in {"function", "variable"}
            )
            if local_definition_contract:
                old_entity, old_records, old_definitions = (
                    _definition_backed_provider(
                        index,
                        tree="old",
                        symbol=symbol,
                        endpoint=aligned_old_target,
                        preferred_file=normalize_relpath(
                            str(donor_entity.get("file") or "")
                        )
                        or None,
                        candidate_files=definition_frontier_files["old"],
                        cache=symbol_query_cache,
                    )
                )
            else:
                old_records = _query_symbol_in_frontier(
                    index,
                    tree="old",
                    symbol=symbol,
                    candidate_files=definition_frontier_files["old"],
                    cache=symbol_query_cache,
                )
                if aligned_old_target is not None:
                    old_records = canonical_entity_records(
                        [aligned_old_target, *old_records]
                    )
                old_entity, old_definitions = _unique_definition(
                    old_records,
                    preferred_file=normalize_relpath(
                        str(donor_entity.get("file") or "")
                    )
                    or None,
                )
            old_provider_ambiguous = len(old_definitions) > 1
            if aligned_old_target is not None:
                alignment = {
                    "status": (
                        "aligned_consumer_provider_definition"
                        if old_entity is not None
                        else "aligned_consumer_declaration_without_provider_definition"
                    ),
                    "consumer_reference_alignment": reference_alignment,
                    "consumer_reference_alignments": consumer_reference_alignments,
                    "old_definition_count": len(old_definitions),
                }
            donor_owner_file = normalize_relpath(str(donor_entity.get("file") or ""))
            if old_entity is None and donor_owner_file in copied_exact_files:
                old_entity = {
                    **donor_entity,
                    "id": stable_id(
                        "entity",
                        "old",
                        str(donor_entity.get("usr") or donor_owner_file),
                        str(donor_entity.get("contract_key") or ""),
                        str((donor_entity.get("extent") or {}).get("start_line") or 0),
                    ),
                    "tree": "old",
                    "transplanted_exact_file_proof": True,
                }
                old_records = [old_entity, *old_records]
                alignment = {
                    "status": "exact_transplanted_file_contract",
                    "proof": "old and donor file SHA-256 values are identical after transplant",
                }
            if old_entity is None and not old_definitions:
                old_entity, alignment = _structural_rename_candidate(
                    index,
                    donor_entity=donor_entity,
                    old_candidate_files=definition_frontier_files["old"],
                    query_cache=symbol_query_cache,
                )
                if old_entity is not None:
                    old_provider_ambiguous = False
                    old_records = _query_symbol_in_frontier(
                        index,
                        tree="old",
                        symbol=str(old_entity.get("symbol") or ""),
                        candidate_files=definition_frontier_files["old"],
                        cache=symbol_query_cache,
                    )
                else:
                    structural_counterpart = _structural_old_counterpart(alignment)
                    if structural_counterpart is not None:
                        old_entity = structural_counterpart
                        old_records = [old_entity]
            if not alignment:
                alignment = {
                    "status": "exact_contract" if old_entity else "missing_or_ambiguous_old_provider",
                    "old_definition_count": len(old_definitions),
                }

        if donor_entity is None:
            continue
        donor_file = normalize_relpath(str(donor_entity.get("file") or ""))
        old_file = normalize_relpath(str((old_entity or {}).get("file") or donor_file))
        provider_modules = _module_names(build_lookup.modules(donor_tree, donor_file))
        old_provider_modules = _module_names(build_lookup.modules("old", old_file))
        module_boundary_evidence = _target_module_boundary(
            dependency_scope=dependency_scope,
            donor_consumer_modules=consumer_modules,
            old_consumer_modules=old_consumer_modules,
            donor_provider_modules=provider_modules,
            old_provider_modules=old_provider_modules,
            provider_internal_linkage=(
                entity_is_internally_linked(donor_entity)
            ),
        )
        module_boundary = str(module_boundary_evidence["relation"])

        old_declarations = _matching_declarations(old_records, old_entity)
        donor_declarations = _matching_declarations(donor_records, donor_entity)
        for aligned_target in aligned_old_targets:
            if (
                not bool(aligned_target.get("is_definition"))
                and all(
                    str(item.get("id") or "") != str(aligned_target.get("id") or "")
                    for item in old_declarations
                )
            ):
                old_declarations.append(aligned_target)
        if (
            donor_target is not None
            and not bool(donor_target.get("is_definition"))
            and all(
                str(item.get("id") or "") != str(donor_target.get("id") or "")
                for item in donor_declarations
            )
        ):
            donor_declarations.append(donor_target)
        old_exports = build_lookup.exports("old", old_file, str(old_entity.get("symbol") if old_entity else symbol))
        donor_exports = build_lookup.exports(donor_tree, donor_file, str(donor_entity.get("symbol") or symbol))
        fingerprint = compare_entity_fingerprints(old_entity, donor_entity)
        exact_source_proof = _same_source_entity_proof(
            old_entity,
            donor_entity,
            identical_source_files,
        )
        if exact_source_proof is not None:
            fingerprint["exact_source_proof"] = exact_source_proof
            for dimension in ("implementation", "interface"):
                if fingerprint.get(dimension) == "unavailable":
                    fingerprint[dimension] = "equivalent"
        declaration_relation, declaration_proof = _declaration_relation(
            old_declarations=old_declarations,
            donor_declarations=donor_declarations,
            identical_source_files=identical_source_files,
            dependency_scope=dependency_scope,
        )
        comparison = {
            "name": (
                "equivalent"
                if old_entity and old_entity.get("symbol") == donor_entity.get("symbol")
                else ("different" if old_entity else "missing_in_old")
            ),
            **fingerprint,
            "declaration": declaration_relation,
            "module_boundary": module_boundary,
            "module_boundary_evidence": module_boundary_evidence,
        }
        if declaration_proof is not None:
            comparison["declaration_proof"] = declaration_proof
        comparison["export"] = _export_relation(
            old_exports=old_exports,
            donor_exports=donor_exports,
            module_boundary=module_boundary,
        )
        comparison["integration_reference"] = (
            integration_reference_relation
        )
        baseline_provider_provenance = (
            _baseline_provider_provenance(
                spec,
                donor_entity,
                structural_cache=baseline_structural_cache,
            )
        )
        integration_reference_required = (
            _integration_reference_is_required(
                task_references,
            )
        )
        # A gap keeps the same identity while the target tree changes.  Old
        # provider state and module integration are observations of that gap,
        # not part of the contract being migrated.
        contract_identity = ":".join(
            [
                "provider",
                str(donor_entity.get("kind") or "entity"),
                str(donor_entity.get("contract_key") or symbol),
                "owner",
                donor_file or "<unknown-provider-file>",
            ]
        )
        if not discovered_from_contract:
            root_contract_identities.add(contract_identity)
        if contract_identity in seen_contracts:
            if discovered_from_contract:
                discovered_contract_dependencies.add(
                    (discovered_from_contract, contract_identity)
                )
            existing = next(
                (
                    item
                    for item in contract_evidence
                    if str(item.get("contract_identity") or "") == contract_identity
                ),
                None,
            )
            if existing is not None:
                combined_consumer_modules = sorted(
                    set(existing.get("consumer_modules") or [])
                    | set(consumer_modules)
                )
                combined_old_consumer_modules = sorted(
                    set(existing.get("old_consumer_modules") or [])
                    | set(old_consumer_modules)
                )
                combined_old_declarations = canonical_entity_records(
                    [
                        *(existing.get("old_declarations") or []),
                        *old_declarations,
                    ]
                )
                combined_donor_declarations = canonical_entity_records(
                    [
                        *(existing.get("donor_declarations") or []),
                        *donor_declarations,
                    ]
                )
                combined_old_exports = _records_with_unique_ids(
                    [
                        *(existing.get("old_exports") or []),
                        *old_exports,
                    ]
                )
                combined_donor_exports = _records_with_unique_ids(
                    [
                        *(existing.get("donor_exports") or []),
                        *donor_exports,
                    ]
                )
                combined_boundary_evidence = _target_module_boundary(
                    dependency_scope=str(
                        existing.get("dependency_scope") or dependency_scope
                    ),
                    donor_consumer_modules=combined_consumer_modules,
                    old_consumer_modules=combined_old_consumer_modules,
                    donor_provider_modules=existing.get("provider_modules") or [],
                    old_provider_modules=existing.get("old_provider_modules") or [],
                    provider_internal_linkage=entity_is_internally_linked(
                        (
                            existing.get("donor_entity")
                            if isinstance(
                                existing.get("donor_entity"),
                                dict,
                            )
                            else donor_entity
                        )
                    ),
                )
                combined_observed_comparison = dict(
                    existing.get("observed_comparison")
                    or existing.get("comparison")
                    or {}
                )
                declaration_relation, declaration_proof = _declaration_relation(
                    old_declarations=combined_old_declarations,
                    donor_declarations=combined_donor_declarations,
                    identical_source_files=identical_source_files,
                    dependency_scope=str(
                        existing.get("dependency_scope") or dependency_scope
                    ),
                )
                combined_observed_comparison["declaration"] = (
                    declaration_relation
                )
                if declaration_proof is None:
                    combined_observed_comparison.pop(
                        "declaration_proof",
                        None,
                    )
                else:
                    combined_observed_comparison["declaration_proof"] = (
                        declaration_proof
                    )
                combined_observed_comparison["module_boundary"] = str(
                    combined_boundary_evidence["relation"]
                )
                combined_observed_comparison[
                    "module_boundary_evidence"
                ] = (
                    combined_boundary_evidence
                )
                combined_observed_comparison["export"] = _export_relation(
                    old_exports=combined_old_exports,
                    donor_exports=combined_donor_exports,
                    module_boundary=str(combined_boundary_evidence["relation"]),
                )
                combined_references = [
                    *[
                        item
                        for item in existing.get("consumer_references") or []
                        if isinstance(item, dict)
                    ],
                    *task_references,
                ]
                combined_unproven_reference_ids = sorted(
                    {
                        *[
                            str(item)
                            for item in existing.get(
                                "unproven_consumer_reference_ids"
                            )
                            or []
                            if str(item)
                        ],
                        *unproven_consumer_reference_ids,
                    }
                )
                combined_observed_comparison[
                    "integration_reference"
                ] = (
                    "equivalent"
                    if combined_references
                    and not combined_unproven_reference_ids
                    else "missing_in_old"
                    if combined_references
                    else "not_required"
                )
                existing_integration_required = (
                    "integration_reference"
                    in {
                        str(item)
                        for item in (
                            existing.get("semantic_requirement")
                            or {}
                        ).get("required_dimensions")
                        or []
                    }
                )
                semantic_requirement = canonical_semantic_requirement(
                    _semantic_requirement(
                        combined_references,
                        (
                            existing.get("donor_entity")
                            if isinstance(existing.get("donor_entity"), dict)
                            else None
                        ),
                        integration_reference_required=(
                            existing_integration_required
                            or integration_reference_required
                        ),
                    )
                )
                combined_structural_source_relation = str(
                    combined_observed_comparison.get(
                        "structural_source"
                    )
                    or "unavailable"
                )
                semantic_requirement = (
                    _with_structural_source_evidence(
                        semantic_requirement,
                        old_entity=(
                            existing.get("old_entity")
                            if isinstance(
                                existing.get("old_entity"),
                                dict,
                            )
                            else None
                        ),
                        donor_entity=(
                            existing.get("donor_entity")
                            if isinstance(
                                existing.get("donor_entity"),
                                dict,
                            )
                            else None
                        ),
                        observed_comparison=(
                            combined_observed_comparison
                        ),
                        structural_source_relation=(
                            combined_structural_source_relation
                        ),
                    )
                )
                combined_comparison = _project_comparison_for_requirement(
                    combined_observed_comparison,
                    semantic_requirement,
                )
                combined_integration_provenance = (
                    dict(
                        existing.get(
                            "integration_reference_provenance"
                        )
                        or {}
                    )
                    or baseline_provider_provenance
                )
                combined_alignment = dict(existing.get("alignment") or {})
                combined_alignment["consumer_reference_alignments"] = (
                    _records_with_unique_ids(
                        [
                            *(
                                combined_alignment.get(
                                    "consumer_reference_alignments"
                                )
                                or []
                            ),
                            *(
                                alignment.get("consumer_reference_alignments")
                                or []
                            ),
                        ]
                    )
                )
                combined_resolution = _resolution_for_evidence(
                    old_entity=existing.get("old_entity"),
                    donor_entity=existing.get("donor_entity"),
                    comparison=combined_comparison,
                    semantic_requirement=semantic_requirement,
                    old_declarations=combined_old_declarations,
                    donor_exports=combined_donor_exports,
                    old_provider_ambiguous=bool(
                        existing.get("old_provider_ambiguous")
                        or old_provider_ambiguous
                    ),
                    static_import_ready=bool(
                        existing.get("static_import_ready") and static_import_ready
                    ),
                    structural_source_relation=str(
                        combined_structural_source_relation
                    ),
                    dependency_scope=str(
                        existing.get("dependency_scope")
                        or dependency_scope
                    ),
                    integration_reference_provenance=(
                        combined_integration_provenance
                    ),
                )
                combined_post_materialization_validation = (
                    _post_materialization_validation_requirement(
                        old_entity=existing.get("old_entity"),
                        donor_entity=existing.get("donor_entity"),
                        consumer_references=combined_references,
                        comparison=combined_comparison,
                        semantic_requirement=semantic_requirement,
                        static_import_ready=bool(
                            existing.get("static_import_ready")
                            and static_import_ready
                        ),
                        integration_reference_provenance=(
                            combined_integration_provenance
                        ),
                        unproven_consumer_reference_ids=(
                            combined_unproven_reference_ids
                        ),
                    )
                )
                existing["consumer_references"] = _records_with_unique_ids(
                    [
                        *(existing.get("consumer_references") or []),
                        *[
                            _compact_demand_reference(item)
                            for item in task_references
                        ],
                    ]
                )
                existing["consumer_modules"] = combined_consumer_modules
                existing["old_consumer_modules"] = combined_old_consumer_modules
                existing["target_consumer_modules"] = list(
                    combined_boundary_evidence["target_consumer_modules"]
                )
                existing["comparison"] = combined_comparison
                existing["observed_comparison"] = (
                    combined_observed_comparison
                )
                existing["semantic_requirement"] = semantic_requirement
                existing["integration_reference_provenance"] = (
                    combined_integration_provenance
                )
                existing[
                    "unproven_consumer_reference_ids"
                ] = combined_unproven_reference_ids
                existing["resolution"] = combined_resolution
                existing["post_materialization_validation"] = (
                    combined_post_materialization_validation
                )
                existing["alignment"] = combined_alignment
                existing["old_declarations"] = combined_old_declarations
                existing["donor_declarations"] = combined_donor_declarations
                existing["old_exports"] = combined_old_exports
                existing["donor_exports"] = combined_donor_exports
                existing["old_provider_ambiguous"] = bool(
                    existing.get("old_provider_ambiguous")
                    or old_provider_ambiguous
                )
                atoms[:] = [
                    atom
                    for atom in atoms
                    if str(atom.get("contract_identity") or "")
                    != contract_identity
                ]
                merged_atoms = build_change_atoms(
                    contract_identity=contract_identity,
                    comparison=combined_comparison,
                    old_entity=existing.get("old_entity"),
                    donor_entity=existing.get("donor_entity"),
                    raw_hunk_ids=existing.get("raw_hunk_ids") or [],
                    contract_evidence_id=str(existing.get("id") or ""),
                    semantic_requirement=semantic_requirement,
                )
                existing["change_atom_ids"] = [
                    str(item["id"]) for item in merged_atoms
                ]
                atoms.extend(merged_atoms)
            continue
        seen_contracts.add(contract_identity)
        if discovered_from_contract:
            discovered_contract_dependencies.add(
                (discovered_from_contract, contract_identity)
            )
        old_consumer_impact = _old_consumer_impact(
            build_lookup,
            old_entity,
            exclude_files=copied_files,
        )
        comparison["old_consumer_impact"] = str(
            old_consumer_impact.get("status")
            or "deferred_to_configured_module_validation"
        )
        structural_alignment = (
            alignment.get("structural_alignment")
            if isinstance(alignment.get("structural_alignment"), dict)
            else alignment
        )
        structural_target = (
            structural_alignment.get("target")
            if isinstance(structural_alignment, dict)
            and isinstance(structural_alignment.get("target"), dict)
            else {}
        )
        structural_counterpart = (
            structural_alignment.get("old_counterpart")
            if isinstance(structural_alignment, dict)
            and isinstance(structural_alignment.get("old_counterpart"), dict)
            else {}
        )
        structural_source_relation = _structural_source_relation(
            old_entity,
            donor_entity,
            structural_counterpart,
            structural_target,
        )
        comparison["structural_source"] = structural_source_relation
        consumer_reference_alignments: list[
            dict[str, Any]
        ] = []
        unproven_consumer_reference_ids: list[str] = []
        for donor_reference in task_references:
            aligned_reference, alignment_record = (
                _aligned_old_reference(
                    index,
                    donor_reference,
                )
            )
            consumer_reference_alignments.append(
                {
                    "donor_reference_id": (
                        donor_reference.get("id")
                    ),
                    "donor_reference": (
                        _compact_demand_reference(
                            donor_reference
                        )
                    ),
                    "old_reference": (
                        _compact_demand_reference(
                            aligned_reference
                        )
                        if aligned_reference is not None
                        else None
                    ),
                    "alignment": alignment_record,
                }
            )
            if not bool(
                alignment_record.get("equivalence_proven")
            ):
                unproven_consumer_reference_ids.append(
                    str(donor_reference.get("id") or "")
                )
        comparison["integration_reference"] = (
            "equivalent"
            if task_references
            and not unproven_consumer_reference_ids
            else "missing_in_old"
            if task_references
            else "not_required"
        )
        observed_comparison = dict(comparison)
        semantic_requirement = canonical_semantic_requirement(
            _semantic_requirement(
                task_references,
                donor_entity,
                integration_reference_required=(
                    integration_reference_required
                ),
            )
        )
        semantic_requirement = _with_structural_source_evidence(
            semantic_requirement,
            old_entity=old_entity,
            donor_entity=donor_entity,
            observed_comparison=observed_comparison,
            structural_source_relation=structural_source_relation,
        )
        comparison = _project_comparison_for_requirement(
            observed_comparison,
            semantic_requirement,
        )
        static_import_common_ready = bool(
            old_entity is None
            and not old_provider_ambiguous
            and _static_import_entity_is_supported(donor_entity)
            and donor_file
            and spec.resolve_old(donor_file).is_file()
            and spec.resolve_donor(donor_file).is_file()
            and spec.classify_edit_root(donor_file) == "editable"
        )
        conditional_static_import = (
            _conditional_static_import_plan(spec, donor_entity)
            if static_import_common_ready
            and not _static_import_alignment_is_supported(
                structural_alignment,
                donor_entity=donor_entity,
            )
            else None
        )
        static_import_ready = bool(
            static_import_common_ready
            and (
                _static_import_alignment_is_supported(
                    structural_alignment,
                    donor_entity=donor_entity,
                )
                or conditional_static_import is not None
            )
        )
        post_materialization_validation = (
            _post_materialization_validation_requirement(
                old_entity=old_entity,
                donor_entity=donor_entity,
                consumer_references=task_references,
                comparison=comparison,
                semantic_requirement=semantic_requirement,
                static_import_ready=static_import_ready,
                integration_reference_provenance=(
                    baseline_provider_provenance
                ),
                unproven_consumer_reference_ids=(
                    unproven_consumer_reference_ids
                ),
            )
        )
        resolution = _resolution_for_evidence(
            old_entity=old_entity,
            donor_entity=donor_entity,
            comparison=comparison,
            semantic_requirement=semantic_requirement,
            old_declarations=old_declarations,
            donor_exports=donor_exports,
            old_provider_ambiguous=old_provider_ambiguous,
            static_import_ready=static_import_ready,
            structural_source_relation=structural_source_relation,
            dependency_scope=dependency_scope,
            integration_reference_provenance=(
                baseline_provider_provenance
            ),
        )
        hunks = _raw_hunks_for_file(hunk_cache, spec=spec, relative_path=donor_file)
        raw_hunk_ids = sorted(
            set(
                hunks_overlapping_extent(
                    hunks,
                    tree=donor_tree,
                    extent=donor_entity.get("extent"),
                )
                + hunks_overlapping_extent(
                    hunks,
                    tree="old",
                    extent=(old_entity or {}).get("extent"),
                )
            )
        )
        evidence_id = stable_id(
            "contract-evidence",
            contract_identity,
        )
        evidence_record = {
            "id": evidence_id,
            "contract_identity": contract_identity,
            "kind": "provider_contract",
            "symbol": symbol,
            "dependency_scope": dependency_scope,
            "consumer_reference": _compact_demand_reference(reference),
            "consumer_references": [
                _compact_demand_reference(item)
                for item in task_references
            ],
            "consumer_modules": consumer_modules,
            "old_consumer_modules": old_consumer_modules,
            "target_consumer_modules": module_boundary_evidence["target_consumer_modules"],
            "old_consumer_impact_status": old_consumer_impact.get("status"),
            "target_validation_files": list(
                old_consumer_impact.get("source_files") or []
            ),
            "provider_modules": provider_modules,
            "old_provider_modules": old_provider_modules,
            "target_provider_modules": module_boundary_evidence["target_provider_modules"],
            "old_entity": old_entity,
            "donor_entity": donor_entity,
            "comparison": comparison,
            "observed_comparison": observed_comparison,
            "semantic_requirement": semantic_requirement,
            "integration_reference_provenance": (
                baseline_provider_provenance
            ),
            "consumer_reference_alignments": (
                consumer_reference_alignments
            ),
            "unproven_consumer_reference_ids": sorted(
                {
                    item
                    for item in unproven_consumer_reference_ids
                    if item
                }
            ),
            "resolution": resolution,
            "old_provider_ambiguous": old_provider_ambiguous,
            "static_import_ready": static_import_ready,
            "post_materialization_validation": (
                post_materialization_validation
            ),
            "conditional_static_import": conditional_static_import,
            "alignment": alignment,
            "old_declarations": old_declarations,
            "donor_declarations": donor_declarations,
            "old_exports": old_exports,
            "donor_exports": donor_exports,
            "raw_hunk_ids": raw_hunk_ids,
            "depends_on_evidence_ids": [],
            "discovered_from_contract_identity": discovered_from_contract or None,
        }
        evidence_atoms = build_change_atoms(
            contract_identity=contract_identity,
            comparison=comparison,
            old_entity=old_entity,
            donor_entity=donor_entity,
            raw_hunk_ids=raw_hunk_ids,
            contract_evidence_id=evidence_id,
            semantic_requirement=semantic_requirement,
        )
        evidence_record["change_atom_ids"] = [
            str(item["id"]) for item in evidence_atoms
        ]
        atoms.extend(evidence_atoms)
        contract_evidence.append(evidence_record)
        donor_entity_id = str(donor_entity.get("id") or "")
        if (
            _missing_provider_needs_dependency_closure(
                old_entity=old_entity,
                donor_entity=donor_entity,
                resolution=resolution,
                provider_frontier_files=definition_frontier_files[
                    donor_tree
                ],
            )
            and donor_entity_id not in expanded_missing_provider_ids
        ):
            expanded_missing_provider_ids.add(donor_entity_id)
            for dependency_reference in _references_owned_by_entity(
                index,
                tree=donor_tree,
                entity=donor_entity,
                preferred_translation_units=[
                    reference.get("translation_unit_file")
                    or reference.get("source_file")
                    for reference in task_references
                ],
            ):
                before = len(queued_reference_tasks)
                enqueue_references(
                    [dependency_reference],
                    direction="forward",
                    discovered_from_contract=contract_identity,
                )
                transitive_reference_count += int(
                    len(queued_reference_tasks) > before
                )

    evaluated_contracts = list(contract_evidence)
    evaluated_by_identity = {
        str(evidence.get("contract_identity") or ""): evidence
        for evidence in evaluated_contracts
    }
    for parent_identity, dependency_identity in discovered_contract_dependencies:
        parent = evaluated_by_identity.get(parent_identity)
        dependency = evaluated_by_identity.get(dependency_identity)
        if parent is None or dependency is None:
            continue
        parent["depends_on_evidence_ids"] = sorted(
            set(parent.get("depends_on_evidence_ids") or [])
            | {str(dependency.get("id") or "")}
        )
    evidence_by_id = {
        str(evidence.get("id") or ""): evidence
        for evidence in evaluated_contracts
        if str(evidence.get("id") or "")
    }
    changed = True
    while changed:
        changed = False
        for evidence in evaluated_contracts:
            blocking_dependencies = [
                dependency_id
                for dependency_id in evidence.get(
                    "depends_on_evidence_ids"
                )
                or []
                if str(
                    (evidence_by_id.get(str(dependency_id)) or {}).get(
                        "resolution"
                    )
                    or ""
                ).startswith("blocked")
            ]
            if not blocking_dependencies or str(
                evidence.get("resolution") or ""
            ).startswith("blocked"):
                continue
            evidence["resolution"] = "blocked_dependency"
            evidence["blocking_evidence_ids"] = sorted(
                blocking_dependencies
            )
            changed = True
    demand_ledger = _reachable_demand_ledger(
        evaluated_contracts,
        root_contract_identities=root_contract_identities,
        contract_dependencies=discovered_contract_dependencies,
    )
    contract_evidence = _reachable_actionable_contract_evidence(
        evaluated_contracts,
        root_contract_identities=root_contract_identities,
        contract_dependencies=discovered_contract_dependencies,
    )
    scheduled_evidence_ids = {
        str(evidence.get("id") or "")
        for evidence in contract_evidence
        if str(evidence.get("id") or "")
    }
    for evidence in demand_ledger:
        evidence_id = str(evidence.get("id") or "")
        resolution = str(evidence.get("resolution") or "")
        if evidence_id in scheduled_evidence_ids:
            disposition = "scheduled_initial_repair"
        elif resolution == "post_materialization_validation_only":
            disposition = "compile_validation_only"
        elif resolution == "satisfied":
            disposition = "no_initial_change"
        elif resolution.startswith("blocked"):
            disposition = "blocked_pending_fresh_evidence"
        elif (
            resolution == "requires_llm"
            or resolution.startswith("deterministic_")
        ):
            disposition = "deferred_behind_non_actionable_ancestor"
        else:
            disposition = "no_initial_action"
        evidence["initial_action_disposition"] = disposition
    for evidence in contract_evidence:
        evidence["initial_action_disposition"] = (
            "scheduled_initial_repair"
        )
    satisfied_count = sum(
        evidence.get("resolution") == "satisfied"
        for evidence in demand_ledger
    )
    blocked_count = sum(
        str(evidence.get("resolution") or "").startswith("blocked")
        for evidence in demand_ledger
    )
    reachable_actionable_count = len(contract_evidence)
    evaluated_actionable_count = sum(
        evidence.get("resolution") == "requires_llm"
        or str(evidence.get("resolution") or "").startswith(
            "deterministic_"
        )
        for evidence in evaluated_contracts
    )
    pruned_unreachable_actionable_count = (
        evaluated_actionable_count - reachable_actionable_count
    )
    evidence_contracts = {
        str(evidence.get("contract_identity") or "")
        for evidence in contract_evidence
    }
    atoms = [
        atom
        for atom in atoms
        if str(atom.get("contract_identity") or "") in evidence_contracts
    ]
    raw_hunks = [
        hunk
        for relative_path in sorted(hunk_cache)
        for hunk in hunk_cache[relative_path]
    ]
    graph = build_initial_difference_graph(
        contract_evidence=contract_evidence,
        change_atoms=atoms,
        raw_hunks=raw_hunks,
    )
    parse_reports = [
        *index.parse_reports.values(),
        *index.reference_probe_reports.values(),
    ]
    target_translation_unit_validity = (
        _target_translation_unit_validity(index, spec)
    )
    parser_seconds = sum(
        float(report.get("parse_seconds") or 0.0)
        for report in parse_reports
        if isinstance(report, dict)
    )
    completed_at = time.monotonic()
    return {
        "schema": INITIAL_COMPATIBILITY_ANALYSIS_SCHEMA,
        "analysis_revision": INITIAL_COMPATIBILITY_ANALYSIS_REVISION,
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "status": "analyzed",
        "preparation_complete": True,
        "preparation_blockers": [],
        "target_translation_unit_validity": (
            target_translation_unit_validity
        ),
        "demand_ledger": demand_ledger,
        "contract_evidence": contract_evidence,
        "change_atoms": atoms,
        "raw_hunks": raw_hunks,
        "initial_difference_graph": graph,
        "summary": {
            "preparation_complete": True,
            "preparation_blocker_count": 0,
            "invalid_target_translation_unit_count": len(
                target_translation_unit_validity
            ),
            "seed_entity_count": len(seed_entities),
            "evaluated_contract_count": len(evaluated_contracts),
            "demand_ledger_count": len(demand_ledger),
            "pruned_unreachable_evaluated_count": (
                len(evaluated_contracts) - len(demand_ledger)
            ),
            "contract_evidence_count": len(contract_evidence),
            "pruned_unreachable_actionable_count": (
                pruned_unreachable_actionable_count
            ),
            "satisfied_count": satisfied_count,
            "deterministic_count": sum(
                str(evidence.get("resolution") or "").startswith(
                    "deterministic_"
                )
                for evidence in contract_evidence
            ),
            "requires_llm_count": sum(
                evidence.get("resolution") == "requires_llm"
                for evidence in contract_evidence
            ),
            "post_materialization_validation_count": sum(
                evidence.get("resolution")
                == "post_materialization_validation_only"
                for evidence in demand_ledger
            ),
            "deferred_reachable_actionable_count": sum(
                evidence.get("initial_action_disposition")
                == "deferred_behind_non_actionable_ancestor"
                for evidence in demand_ledger
            ),
            "blocked_count": blocked_count,
            "expanded_missing_provider_count": len(expanded_missing_provider_ids),
            "transitive_reference_count": transitive_reference_count,
            "indexed_parse_report_count": len(parse_reports),
            "successful_parse_report_count": sum(
                bool(report.get("parse_ok"))
                for report in parse_reports
                if isinstance(report, dict)
            ),
        },
        "timings_sec": {
            "file_manifest": round(manifest_completed_at - started_at, 6),
            "seed_index": round(seed_index_completed_at - manifest_completed_at, 6),
            "reverse_candidate_index": round(reverse_index_completed_at - seed_index_completed_at, 6),
            "contract_resolution": round(completed_at - reverse_index_completed_at, 6),
            "libclang_parse_total": round(parser_seconds, 6),
            "total": round(completed_at - started_at, 6),
        },
    }
