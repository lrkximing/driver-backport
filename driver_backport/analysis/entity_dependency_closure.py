#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from driver_backport.analysis.code_index import CodeIndex, GraphBudget, entity_record_rank
from driver_backport.analysis.conditional_contract import conditional_variant_family
from driver_backport.analysis.entity_fingerprint import compare_entity_fingerprints
from driver_backport.analysis.entity_resolution import (
    entity_contract_key,
    entity_kind,
)
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import normalize_relpath


_INTERFACE_BOUNDARY_KINDS = {"function", "variable"}
_DEFINITION_BOUNDARY_KINDS = {
    "macro",
    "type",
    "field",
    "enum_constant",
}
_C_TRANSLATION_UNIT_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}


def _context_translation_units(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            relative_path
            for value in values
            if (
                relative_path := normalize_relpath(
                    str(value or "")
                )
            )
            and Path(relative_path).suffix.lower()
            in _C_TRANSLATION_UNIT_SUFFIXES
        )
    )


class EntityDependencyClosureError(ValueError):
    def __init__(self, proof: dict[str, Any]) -> None:
        self.proof = proof
        unresolved = list(proof.get("unresolved") or [])
        details = ", ".join(
            str(item.get("contract_key") or item.get("symbol") or item.get("reason") or "")
            for item in unresolved[:5]
            if isinstance(item, dict)
        )
        suffix = f": {details}" if details else ""
        super().__init__(
            "entity materialization dependency closure is unproven"
            f" ({proof.get('reason') or 'missing evidence'}){suffix}"
        )


def _extent(record: dict[str, Any]) -> tuple[int, int]:
    extent = record.get("extent") if isinstance(record.get("extent"), dict) else {}
    start = int(extent.get("start_line") or 0)
    end = int(extent.get("end_line") or start)
    return start, end


def _strongest(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [dict(record) for record in records if isinstance(record, dict)]
    if not candidates:
        return []
    best_rank = max(entity_record_rank(record) for record in candidates)
    return [
        record
        for record in candidates
        if entity_record_rank(record) == best_rank
    ]


def _same_source_extent(
    record: dict[str, Any],
    *,
    relative_path: str,
    root_extents: Iterable[tuple[int, int]],
) -> bool:
    if normalize_relpath(str(record.get("file") or "")) != relative_path:
        return False
    start, end = _extent(record)
    return start > 0 and any(
        root_start <= start <= end <= root_end
        for root_start, root_end in root_extents
    )


def _diagnostic_overlaps_roots(
    report: dict[str, Any],
    *,
    root: Path,
    relative_path: str,
    root_extents: Iterable[tuple[int, int]],
) -> bool:
    expected = (root / relative_path).resolve()
    for diagnostic in report.get("diagnostics") or []:
        if int(diagnostic.get("severity") or 0) < 3:
            continue
        location = (
            diagnostic.get("location")
            if isinstance(diagnostic.get("location"), dict)
            else {}
        )
        try:
            diagnostic_path = Path(str(location.get("file") or "")).resolve()
        except (OSError, RuntimeError):
            continue
        line = int(location.get("line") or 0)
        if diagnostic_path == expected and any(
            start <= line <= end for start, end in root_extents
        ):
            return True
    return False


def _provider_relation(
    donor: dict[str, Any],
    old_candidates: Iterable[dict[str, Any]],
    *,
    interface_only: bool = False,
) -> dict[str, Any]:
    donor_kind = entity_kind(donor)
    donor_contract = entity_contract_key(donor)
    matching = _strongest(
        candidate
        for candidate in old_candidates
        if entity_kind(candidate) == donor_kind
        and entity_contract_key(candidate) == donor_contract
    )
    comparisons = [
        {
            "old_entity_id": candidate.get("id") or candidate.get("entity_id"),
            "old_file": candidate.get("file"),
            "comparison": compare_entity_fingerprints(candidate, donor),
            "old_digest": candidate.get("digest"),
            "donor_digest": donor.get("digest"),
        }
        for candidate in matching
    ]
    compatible: list[dict[str, Any]] = []
    for candidate, comparison in zip(matching, comparisons):
        relation = comparison["comparison"]
        if (
            donor_kind in _INTERFACE_BOUNDARY_KINDS
            or donor_kind == "type"
            and interface_only
        ):
            accepted = (
                relation.get("signature") == "equivalent"
                and relation.get("interface") == "equivalent"
            )
        elif donor_kind in _DEFINITION_BOUNDARY_KINDS:
            accepted = bool(
                donor.get("digest")
                and candidate.get("digest")
                and donor.get("digest") == candidate.get("digest")
            )
        else:
            accepted = False
        if accepted:
            compatible.append(comparison)
    return {
        "status": (
            "compatible"
            if compatible and len(compatible) == len(comparisons)
            else "missing"
            if not matching
            else "incompatible_or_ambiguous"
        ),
        "kind": donor_kind,
        "contract_key": donor_contract,
        "candidate_count": len(matching),
        "compatible_candidate_count": len(compatible),
        "comparisons": comparisons,
    }


def _reference_is_function_interface_type(
    reference: dict[str, Any],
) -> bool:
    """Return whether a type reference belongs to the copied function ABI."""

    if str(reference.get("use_kind") or "") != "typeref":
        return False
    path_kinds = [
        str(item.get("cursor_kind") or "")
        for item in reference.get("ast_path") or []
        if isinstance(item, dict)
    ]
    return bool(
        path_kinds
        and path_kinds[0] == "FUNCTION_DECL"
        and "COMPOUND_STMT" not in path_kinds
    )


def _reference_is_internal_local_declaration(
    reference: dict[str, Any],
    *,
    donor_root: Path,
    relative_path: str,
    root_extents: Iterable[tuple[int, int]],
) -> bool:
    """Recognize exact parameter/local declarations inside the copied entity."""

    if str(reference.get("referenced_kind") or "") not in {
        "PARM_DECL",
        "VAR_DECL",
    }:
        return False
    location = (
        reference.get("referenced_location")
        if isinstance(reference.get("referenced_location"), dict)
        else {}
    )
    raw_file = str(
        location.get("file")
        or reference.get("referenced_file")
        or ""
    )
    try:
        path = Path(raw_file)
        referenced_file = (
            normalize_relpath(
                path.resolve().relative_to(donor_root.resolve()).as_posix()
            )
            if path.is_absolute()
            else normalize_relpath(raw_file)
        )
    except (OSError, ValueError):
        return False
    line = int(location.get("line") or 0)
    return bool(
        referenced_file == relative_path
        and line > 0
        and any(start <= line <= end for start, end in root_extents)
    )


def _semantic_root_entities(
    report: dict[str, Any],
    *,
    relative_path: str,
    structural_roots: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for structural_root in structural_roots:
        contract_key = entity_contract_key(structural_root)
        structural_start, structural_end = _extent(structural_root)
        exact = [
            record
            for record in report.get("entities") or []
            if isinstance(record, dict)
            and entity_contract_key(record) == contract_key
            and normalize_relpath(str(record.get("file") or ""))
            == relative_path
            and bool(record.get("is_definition"))
            == bool(structural_root.get("is_definition"))
            and max(_extent(record)[0], structural_start)
            <= min(_extent(record)[1], structural_end)
        ]
        strongest = _strongest(exact)
        if len(strongest) != 1:
            return []
        record = strongest[0]
        identity = str(record.get("id") or record.get("entity_id") or "")
        if not identity:
            return []
        selected[identity] = record
    return list(selected.values())


def _structural_dependency_entity(
    dependency: dict[str, Any],
    *,
    relative_path: str,
    donor_entities: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    target_contract = entity_contract_key(dependency)
    target_kind = entity_kind(dependency)
    target_start, target_end = _extent(dependency)
    matches = [
        record
        for record in donor_entities
        if isinstance(record, dict)
        and normalize_relpath(str(record.get("file") or ""))
        == relative_path
        and entity_contract_key(record) == target_contract
        and entity_kind(record) == target_kind
        and bool(record.get("is_definition"))
        and (
            target_start <= 0
            or target_end < target_start
            or max(_extent(record)[0], target_start)
            <= min(_extent(record)[1], target_end)
        )
    ]
    strongest = _strongest(matches)
    return strongest[0] if len(strongest) == 1 else None


def _structural_dependency_symbols(
    entities: Iterable[dict[str, Any]],
) -> tuple[set[str], bool]:
    records = [record for record in entities if isinstance(record, dict)]
    complete = bool(records) and all(
        record.get("dependency_analysis_complete") is True
        for record in records
    )
    return {
        str(symbol)
        for record in records
        for symbol in record.get("dependency_symbols") or []
        if str(symbol)
    }, complete


def _structural_provider_relation(
    symbol: str,
    *,
    old_entities: Iterable[dict[str, Any]],
    donor_entities: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    donor_matches = [
        record
        for record in donor_entities
        if isinstance(record, dict)
        and str(record.get("symbol") or record.get("name") or "") == symbol
        and bool(record.get("is_definition"))
    ]
    old_matches = [
        record
        for record in old_entities
        if isinstance(record, dict)
        and str(record.get("symbol") or record.get("name") or "") == symbol
        and bool(record.get("is_definition"))
    ]
    compatible_pairs = [
        (old, donor)
        for old in old_matches
        for donor in donor_matches
        if entity_kind(old) == entity_kind(donor)
        and entity_contract_key(old) == entity_contract_key(donor)
        and old.get("source_fingerprint")
        and old.get("source_fingerprint") == donor.get("source_fingerprint")
    ]
    old_type_references = [
        record
        for record in old_entities
        if isinstance(record, dict)
        and record.get("dependency_analysis_complete") is True
        and symbol in set(record.get("dependency_type_symbols") or [])
    ]
    donor_type_references = [
        record
        for record in donor_entities
        if isinstance(record, dict)
        and record.get("dependency_analysis_complete") is True
        and symbol in set(record.get("dependency_type_symbols") or [])
    ]
    existing_owner_type_reference = bool(
        not old_matches
        and not donor_matches
        and old_type_references
        and donor_type_references
    )
    definition_compatible = bool(
        len(compatible_pairs) == 1
        and len(old_matches) == 1
        and len(donor_matches) == 1
    )
    return {
        "status": (
            "compatible"
            if definition_compatible or existing_owner_type_reference
            else "missing"
            if not old_matches or not donor_matches
            else "incompatible_or_ambiguous"
        ),
        "symbol": symbol,
        "old_candidate_count": len(old_matches),
        "donor_candidate_count": len(donor_matches),
        "compatible_pair_count": len(compatible_pairs),
        "existing_owner_type_reference": existing_owner_type_reference,
        "old_type_reference_count": len(old_type_references),
        "donor_type_reference_count": len(donor_type_references),
        "resolution_basis": (
            "same_owner_existing_type_reference"
            if existing_owner_type_reference
            else "definition_fingerprint"
        ),
        "old_contract_keys": sorted({entity_contract_key(record) for record in old_matches}),
        "donor_contract_keys": sorted({entity_contract_key(record) for record in donor_matches}),
    }


def _unique_strongest_symbol_matches(
    records: Iterable[dict[str, Any]],
    *,
    symbol: str,
    unscoped_lookup: bool = False,
) -> list[dict[str, Any]]:
    unique: dict[
        tuple[str, str, str, str, tuple[int, int]],
        dict[str, Any],
    ] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or str(record.get("symbol") or record.get("name") or "")
            != symbol
        ):
            continue
        interface = record.get("interface") or {}
        if not isinstance(interface, dict):
            interface = {}
        if (
            unscoped_lookup
            and entity_kind(record) == "variable"
            and str(interface.get("linkage") or record.get("linkage") or "").upper()
            == "NO_LINKAGE"
        ):
            # A token search across translation units cannot bind a block local.
            continue
        identity = (
            str(record.get("id") or record.get("entity_id") or ""),
            entity_kind(record),
            entity_contract_key(record),
            normalize_relpath(str(record.get("file") or "")),
            _extent(record),
        )
        unique.setdefault(identity, record)
    return _strongest(unique.values())


def _build_conditioned_structural_provider_relations(
    index: CodeIndex,
    symbols: Iterable[str],
    *,
    old_entities: Iterable[dict[str, Any]],
    donor_entities: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve structural-only dependencies in each realized build frontier."""

    requested = sorted({str(symbol) for symbol in symbols if str(symbol)})
    relations = {
        symbol: _structural_provider_relation(
            symbol,
            old_entities=old_entities,
            donor_entities=donor_entities,
        )
        for symbol in requested
    }
    unresolved_symbols = [
        symbol
        for symbol in requested
        if relations[symbol].get("status") != "compatible"
    ]
    if not unresolved_symbols:
        return [relations[symbol] for symbol in requested]

    query_reports: dict[str, dict[str, dict[str, Any]]] = {}
    for tree in ("old", "donor_0"):
        frontier = index.compile_dependency_frontier(tree, within_root="")
        if str(frontier.get("status") or "") != "ready":
            query_reports[tree] = {
                symbol: {
                    "complete": False,
                    "unavailable": True,
                    "unresolved_reason": (
                        frontier.get("reason")
                        or frontier.get("status")
                        or "compile_dependency_frontier_unavailable"
                    ),
                    "matches": [],
                }
                for symbol in unresolved_symbols
            }
            continue
        exact = index.exact_token_candidates_in_files(
            tree,
            unresolved_symbols,
            frontier.get("files") or [],
        )
        query_reports[tree] = {
            symbol: index.query_definition_batches(
                tree=tree,
                symbols=[symbol],
                candidate_files=(
                    exact.get("candidates_by_symbol") or {}
                ).get(symbol, []),
                candidate_source=(
                    "case_dependency_frontier_exact_token"
                ),
                require_definition=False,
                exhaustive_candidate_scan=True,
            )
            for symbol in unresolved_symbols
        }

    for symbol in unresolved_symbols:
        fallback = relations[symbol]
        old_query = query_reports["old"][symbol]
        donor_query = query_reports["donor_0"][symbol]
        query_complete = bool(
            old_query.get("complete")
            and not old_query.get("unavailable")
            and donor_query.get("complete")
            and not donor_query.get("unavailable")
        )
        if not query_complete:
            relations[symbol] = {
                **fallback,
                "status": "unresolved",
                "provider_identity_complete": False,
                "resolution_basis": (
                    "build_conditioned_provider_query_incomplete"
                ),
                "old_query_complete": bool(old_query.get("complete")),
                "donor_query_complete": bool(donor_query.get("complete")),
                "old_query_error": old_query.get("unresolved_reason"),
                "donor_query_error": donor_query.get("unresolved_reason"),
            }
            continue
        old_matches = _unique_strongest_symbol_matches(
            old_query.get("matches") or [],
            symbol=symbol,
            unscoped_lookup=True,
        )
        donor_matches = _unique_strongest_symbol_matches(
            donor_query.get("matches") or [],
            symbol=symbol,
            unscoped_lookup=True,
        )
        if len(donor_matches) != 1 or len(old_matches) > 1:
            relations[symbol] = {
                **fallback,
                "status": (
                    "provider_identity_absent"
                    if not donor_matches
                    else "provider_identity_ambiguous"
                ),
                "provider_identity_complete": False,
                "resolution_basis": (
                    "build_conditioned_exact_provider_query"
                ),
                "old_candidate_count": len(old_matches),
                "donor_candidate_count": len(donor_matches),
            }
            continue
        donor = donor_matches[0]
        relation = _provider_relation(donor, old_matches)
        relations[symbol] = {
            "symbol": symbol,
            **relation,
            "provider_identity_complete": True,
            "provider_file": normalize_relpath(
                str(donor.get("file") or "")
            ),
            "provider_entity_id": str(
                donor.get("id") or donor.get("entity_id") or ""
            ),
            "resolution_basis": "build_conditioned_exact_provider_query",
            "old_query_complete": True,
            "donor_query_complete": True,
        }
    return [relations[symbol] for symbol in requested]


def _inactive_conditional_root_family(
    spec: CaseSpec,
    *,
    relative_path: str,
    structural_root: dict[str, Any],
    donor_structural_entities: Iterable[dict[str, Any]],
    semantic_entities: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    """Recognize an exact root absent only from the donor's active branch."""

    context = structural_root.get("conditional_context") or []
    if not isinstance(context, list) or len(context) < 2:
        return None
    contract = entity_contract_key(structural_root)
    kind = entity_kind(structural_root)
    variants = [
        entity
        for entity in donor_structural_entities
        if isinstance(entity, dict)
        and normalize_relpath(str(entity.get("file") or "")) == relative_path
        and entity_contract_key(entity) == contract
        and entity_kind(entity) == kind
    ]
    root_extent = _extent(structural_root)
    if len(variants) < 2 or len(
        [
            entity for entity in variants
            if _extent(entity) == root_extent
            and bool(entity.get("is_definition"))
            == bool(structural_root.get("is_definition"))
            and entity.get("conditional_context") == context
        ]
    ) != 1:
        return None
    try:
        donor_lines = spec.resolve_donor(relative_path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines(keepends=True)
        old_lines = spec.resolve_old(relative_path).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines(keepends=True)
    except OSError:
        return None
    family = conditional_variant_family(
        donor_lines=donor_lines,
        old_lines=old_lines,
        target_entities=variants,
    )
    if family is None:
        return None
    active_siblings = [
        entity
        for entity in semantic_entities
        if isinstance(entity, dict)
        and normalize_relpath(str(entity.get("file") or "")) == relative_path
        and entity_contract_key(entity) == contract
        and entity_kind(entity) == kind
        and any(
            _extent(entity)[0] <= _extent(variant)[1]
            and _extent(variant)[0] <= _extent(entity)[1]
            for variant in variants
            if _extent(variant) != root_extent
        )
    ]
    if len(_strongest(active_siblings)) != 1:
        return None
    if any(
        _extent(entity)[0] <= root_extent[1]
        and root_extent[0] <= _extent(entity)[1]
        for entity in semantic_entities
        if isinstance(entity, dict)
        and normalize_relpath(str(entity.get("file") or "")) == relative_path
        and entity_contract_key(entity) == contract
        and entity_kind(entity) == kind
    ):
        return None
    return family


def _exact_type_dependency_relation(
    index: CodeIndex,
    symbol: str,
    *,
    interface_only: bool = False,
) -> dict[str, Any]:
    """Replace same-owner type-name inference with two exact providers."""

    providers: dict[str, dict[str, Any]] = {}
    for tree in ("old", "donor_0"):
        frontier = index.compile_dependency_frontier(tree, within_root="")
        if frontier.get("status") != "ready":
            return {
                "symbol": symbol,
                "status": "unresolved",
                "reason": "compile_frontier_unavailable",
            }
        candidates = index.exact_token_candidates_in_files(
            tree, [symbol], frontier.get("files") or []
        )
        query = index.query_definition_batches(
            tree=tree,
            symbols=[symbol],
            candidate_files=(candidates.get("candidates_by_symbol") or {}).get(symbol, []),
            candidate_source="case_dependency_frontier_exact_token",
            require_definition=False,
            exhaustive_candidate_scan=True,
        )
        if not query.get("complete") or query.get("unavailable"):
            return {"symbol": symbol, "status": "unresolved", "reason": "provider_query_incomplete"}
        exact = _unique_strongest_symbol_matches(
            (
                item for item in query.get("matches") or []
                if entity_kind(item) == "type"
                and entity_contract_key(item) == f"type:{symbol}"
                and item.get("is_definition") is True
            ),
            symbol=symbol,
        )
        if len(exact) != 1:
            return {"symbol": symbol, "status": "unresolved", "reason": "type_provider_not_unique"}
        providers[tree] = exact[0]
    return {
        "symbol": symbol,
        "provider_identity_complete": True,
        "resolution_basis": "exact_build_conditioned_type_definitions",
        "required_facet": "identity" if interface_only else "layout",
        **_provider_relation(
            providers["donor_0"],
            [providers["old"]],
            interface_only=interface_only,
        ),
    }


def _generated_configuration_macro_evidence(
    spec: CaseSpec,
    reference: dict[str, Any],
    *,
    relative_path: str,
) -> dict[str, Any] | None:
    symbol = str(reference.get("symbol") or "")
    if (
        not symbol
        or str(reference.get("use_kind") or "") != "macro"
        or str(reference.get("referenced_kind") or "") != "MACRO_DEFINITION"
        or str(reference.get("contract_key") or "") != f"macro:{symbol}"
        or spec.build_dir_path is None
        or not spec.donor_compile_commands
    ):
        return None
    from driver_backport.scope.compile_commands import CompileCommands

    try:
        database = CompileCommands.from_file(
            Path(spec.donor_compile_commands[0])
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    expected_file = (spec.primary_donor_root / relative_path).resolve()
    matching = [
        entry for entry in database.entries
        if entry.file.resolve() == expected_file
    ]
    contextual = matching or database.entries
    donor_build_dirs = {entry.directory for entry in contextual}
    if len(donor_build_dirs) != 1:
        return None
    donor_build_dir = next(iter(donor_build_dirs))

    def definition(build_dir: Path) -> str | None:
        header = build_dir / "include/generated/autoconf.h"
        try:
            lines = header.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return None
        values = [
            parts[2]
            for line in lines
            if len(parts := line.strip().split(None, 2)) == 3
            and parts[0] == "#define"
            and parts[1] == symbol
        ]
        return values[0] if len(values) == 1 else None

    old_value = definition(spec.build_dir_path)
    donor_value = definition(donor_build_dir)
    if old_value is None or old_value != donor_value:
        return None
    return {
        "source": "generated_configuration",
        "symbol": symbol,
        "contract_key": f"macro:{symbol}",
        "status": "compatible",
        "definition": old_value,
        "old_header": str(spec.build_dir_path / "include/generated/autoconf.h"),
        "donor_header": str(donor_build_dir / "include/generated/autoconf.h"),
    }


def prove_entity_materialization_dependency_closure(
    spec: CaseSpec,
    *,
    relative_path: str,
    contract_key: str,
    old_structural_report: dict[str, Any],
    donor_structural_report: dict[str, Any],
    donor_root_entities: Iterable[dict[str, Any]],
    preferred_translation_units: Iterable[object] = (),
    _index: CodeIndex | None = None,
    _report_cache: dict[tuple[str, str, bool], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove that a donor entity can execute against current-tree providers.

    The copied entity is the transaction boundary. References contained in
    that boundary are internal. Exact same-owner dependencies join the same
    transaction. Exact cross-owner dependencies may remain open for the next
    target build and demand-closure pass; they are recorded as obligations,
    never treated as satisfied. Missing provider identity or build evidence is
    a failed proof, not implicit success.
    """

    relative_path = normalize_relpath(relative_path)
    context_translation_units = _context_translation_units(
        [relative_path, *preferred_translation_units]
    )
    structural_roots = [
        dict(record)
        for record in donor_root_entities
        if isinstance(record, dict)
    ]
    root_extents = {_extent(record) for record in structural_roots}
    root_extents = {
        (start, end)
        for start, end in root_extents
        if start > 0 and end >= start
    }
    structural_symbols, structural_complete = _structural_dependency_symbols(
        structural_roots
    )
    result: dict[str, Any] = {
        "schema": "entity_materialization_dependency_closure_v1",
        "status": "unproven",
        "contract_key": contract_key,
        "path": relative_path,
        "root_extents": [
            {"start_line": start, "end_line": end}
            for start, end in sorted(root_extents)
        ],
        "structural_dependency_symbols": sorted(structural_symbols),
        "structural_dependency_analysis_complete": structural_complete,
        "obligations": [],
        "unresolved": [],
    }
    if not root_extents:
        result["reason"] = "structural_dependency_inventory_incomplete"
        return result
    if structural_complete and not structural_symbols:
        result.update(
            {
                "status": "proven",
                "reason": (
                    "complete_structural_inventory_has_no_dependencies"
                ),
                "semantic_parse": {
                    "available": False,
                    "not_required": True,
                },
            }
        )
        return result

    index = _index or CodeIndex(
        spec,
        budget=GraphBudget(max_nodes=2_000, max_edges=6_000),
    )
    report_cache = (
        _report_cache if _report_cache is not None else {}
    )

    def indexed_report(
        tree: str,
        path: str,
        *,
        detailed_processing: bool,
    ) -> dict[str, Any]:
        key = (tree, normalize_relpath(path), detailed_processing)
        cached = report_cache.get(key)
        if cached is not None:
            return cached
        if not detailed_processing:
            detailed = report_cache.get((tree, key[1], True))
            if detailed is not None:
                return detailed
        report = index.index_file(
            tree,
            key[1],
            detailed_processing=detailed_processing,
        )
        report_cache[key] = report
        return report

    donor_tree = "donor_0"
    donor_report = indexed_report(
        donor_tree,
        relative_path,
        detailed_processing=True,
    )
    contextual_translation_unit = ""
    if (
        not donor_report.get("parse_ok")
        or _diagnostic_overlaps_roots(
            donor_report,
            root=spec.primary_donor_root,
            relative_path=relative_path,
            root_extents=root_extents,
        )
    ):
        contextual_indexer = getattr(
            index,
            "index_file_from_translation_unit",
            None,
        )
        if callable(contextual_indexer):
            for translation_unit in context_translation_units:
                candidate = contextual_indexer(
                    donor_tree,
                    relative_path,
                    translation_unit,
                    detailed_processing=True,
                )
                if (
                    candidate.get("parse_ok")
                    and len(
                        _semantic_root_entities(
                            candidate,
                            relative_path=relative_path,
                            structural_roots=structural_roots,
                        )
                    )
                    == len(structural_roots)
                    and not _diagnostic_overlaps_roots(
                        candidate,
                        root=spec.primary_donor_root,
                        relative_path=relative_path,
                        root_extents=root_extents,
                    )
                ):
                    donor_report = candidate
                    contextual_translation_unit = translation_unit
                    break
    if not donor_report.get("parse_ok"):
        if not structural_complete:
            result["reason"] = "structural_dependency_inventory_incomplete"
            result["semantic_parse"] = {
                "available": False,
                "parse_error": donor_report.get("parse_error"),
            }
            return result
        structural_obligations = [
            {"source": "structural_reference", **relation}
            for relation in _build_conditioned_structural_provider_relations(
                index,
                structural_symbols,
                old_entities=old_structural_report.get("entities") or [],
                donor_entities=donor_structural_report.get("entities") or [],
            )
        ]
        structural_unresolved = [
            obligation
            for obligation in structural_obligations
            if obligation.get("status") != "compatible"
        ]
        result["obligations"] = structural_obligations
        result["unresolved"] = structural_unresolved
        if not structural_symbols:
            result.update(
                {
                    "status": "proven",
                    "reason": "complete_structural_inventory_has_no_dependencies",
                    "semantic_parse": {
                        "available": False,
                        "parse_error": donor_report.get("parse_error"),
                    },
                }
            )
        elif not structural_unresolved:
            result.update(
                {
                    "status": "proven",
                    "reason": (
                        "complete_structural_inventory_has_exact_same_owner_providers"
                    ),
                    "semantic_parse": {
                        "available": False,
                        "parse_error": donor_report.get("parse_error"),
                    },
                }
            )
        else:
            result["reason"] = (
                "donor_semantic_parse_unavailable_and_structural_closure_unproven"
            )
            result["semantic_parse"] = {
                "available": False,
                "parse_error": donor_report.get("parse_error"),
            }
        return result

    semantic_roots = _semantic_root_entities(
        donor_report,
        relative_path=relative_path,
        structural_roots=structural_roots,
    )
    if len(semantic_roots) != len(structural_roots):
        if not structural_complete:
            result["reason"] = "structural_dependency_inventory_incomplete"
            result["semantic_root_count"] = len(semantic_roots)
            result["expected_semantic_root_count"] = len(structural_roots)
            return result
        inactive_family = (
            _inactive_conditional_root_family(
                spec,
                relative_path=relative_path,
                structural_root=structural_roots[0],
                donor_structural_entities=donor_structural_report.get("entities") or [],
                semantic_entities=donor_report.get("entities") or [],
            )
            if len(structural_roots) == 1
            and old_structural_report.get("parse_ok") is True
            and donor_structural_report.get("parse_ok") is True
            else None
        )
        if inactive_family is not None:
            relations = _build_conditioned_structural_provider_relations(
                index,
                structural_symbols,
                old_entities=old_structural_report.get("entities") or [],
                donor_entities=donor_structural_report.get("entities") or [],
            )
            type_facets = structural_roots[0].get("dependency_type_facets") or {}
            if not isinstance(type_facets, dict):
                type_facets = {}
            relations = [
                _exact_type_dependency_relation(
                    index,
                    str(relation["symbol"]),
                    interface_only=(
                        type_facets.get(relation.get("symbol")) == "identity"
                    ),
                )
                if relation.get("resolution_basis") == "same_owner_existing_type_reference"
                or type_facets.get(relation.get("symbol")) == "identity"
                else relation
                for relation in relations
            ]
            obligations = [
                {"source": "inactive_branch_structural_reference", **relation}
                for relation in relations
            ]
            unresolved = [
                item for item in obligations
                if item.get("status") != "compatible"
            ]
            result["obligations"] = obligations
            result["unresolved"] = unresolved
            result["hard_unresolved"] = unresolved
            result["conditional_variant_family"] = inactive_family
            result["semantic_parse"] = {
                "available": True,
                "selected_root_inactive": True,
                "translation_unit_file": contextual_translation_unit or None,
            }
            result["status"] = "unproven" if unresolved else "proven"
            result["reason"] = (
                "inactive_conditional_dependency_closure_unproven"
                if unresolved
                else "inactive_conditional_structural_dependencies_proven"
            )
            return result
        result["reason"] = "donor_semantic_root_absent_or_ambiguous"
        result["semantic_root_count"] = len(semantic_roots)
        result["expected_semantic_root_count"] = len(structural_roots)
        return result
    if _diagnostic_overlaps_roots(
        donor_report,
        root=spec.primary_donor_root,
        relative_path=relative_path,
        root_extents=root_extents,
    ):
        result["reason"] = "donor_semantic_parse_error_overlaps_entity"
        return result
    result["dependency_evidence_basis"] = (
        "active_semantic_root_with_partial_structural_inventory"
        if not structural_complete
        else "active_semantic_and_structural_inventory"
    )

    root_entity_ids = {
        str(root.get("id") or root.get("entity_id") or "")
        for root in semantic_roots
    }
    root_contracts = {
        entity_contract_key(root)
        for root in structural_roots
        if entity_contract_key(root)
    }
    semantic_references = [
        reference
        for reference in donor_report.get("references") or []
        if isinstance(reference, dict)
        and (
            str(reference.get("owner_entity_id") or "") in root_entity_ids
            or (
                entity_contract_key(reference.get("owner_entity") or {})
                in root_contracts
                and any(
                    start <= int(reference.get("line") or 0) <= end
                    for start, end in root_extents
                )
            )
        )
    ]
    old_owner_report = indexed_report(
        "old",
        relative_path,
        detailed_processing=True,
    )
    semantically_accounted_symbols: set[str] = set()
    obligations: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    materializable_dependencies: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, str, str]] = set()
    for reference in semantic_references:
        symbol = str(reference.get("symbol") or "")
        if symbol:
            semantically_accounted_symbols.add(symbol)
        target = (
            reference.get("referenced_entity")
            if isinstance(reference.get("referenced_entity"), dict)
            else {}
        )
        if not target:
            if _reference_is_internal_local_declaration(
                reference,
                donor_root=spec.primary_donor_root,
                relative_path=relative_path,
                root_extents=root_extents,
            ):
                obligations.append(
                    {
                        "source": "semantic_reference",
                        "symbol": symbol,
                        "contract_key": reference.get("contract_key"),
                        "status": "internal_local_declaration",
                        "referenced_kind": reference.get(
                            "referenced_kind"
                        ),
                        "line": reference.get("line"),
                    }
                )
                continue
            generated_config = _generated_configuration_macro_evidence(
                spec,
                reference,
                relative_path=relative_path,
            )
            if generated_config is not None:
                obligations.append(generated_config)
                continue
            unresolved.append(
                {
                    "source": "semantic_reference",
                    "symbol": symbol,
                    "contract_key": reference.get("contract_key"),
                    "reason": "referenced_entity_unresolved",
                    "line": reference.get("line"),
                }
            )
            continue
        if _same_source_extent(
            target,
            relative_path=relative_path,
            root_extents=root_extents,
        ):
            obligations.append(
                {
                    "source": "semantic_reference",
                    "symbol": symbol,
                    "contract_key": entity_contract_key(target),
                    "status": "internal_to_materialized_entity",
                }
            )
            continue
        target_file = normalize_relpath(str(target.get("file") or ""))
        target_contract = entity_contract_key(target)
        identity = (target_file, target_contract, entity_kind(target))
        if identity in seen_targets:
            continue
        seen_targets.add(identity)
        if not target_file or not target_contract:
            unresolved.append(
                {
                    "source": "semantic_reference",
                    "symbol": symbol,
                    "reason": "provider_identity_incomplete",
                    "line": reference.get("line"),
                }
            )
            continue
        old_provider_report = indexed_report(
            "old",
            target_file,
            detailed_processing=False,
        )
        old_candidates = list(
            old_provider_report.get("entities") or []
            if old_provider_report.get("parse_ok")
            else []
        )
        provider_resolution_basis = (
            "direct_owner_parse"
            if old_provider_report.get("parse_ok")
            else ""
        )
        focused_provider_reports: list[dict[str, Any]] = []
        focused_indexer = getattr(
            index,
            "index_file_from_translation_unit",
            None,
        )
        direct_contract_found = any(
            entity_kind(candidate) == entity_kind(target)
            and entity_contract_key(candidate) == target_contract
            for candidate in old_candidates
            if isinstance(candidate, dict)
        )
        if (
            not direct_contract_found
            and callable(focused_indexer)
        ):
            for translation_unit in context_translation_units:
                focused_key = (
                    "old",
                    f"{target_file}\0{translation_unit}",
                    False,
                )
                focused_provider_report = report_cache.get(focused_key)
                if focused_provider_report is None:
                    focused_provider_report = focused_indexer(
                        "old",
                        target_file,
                        translation_unit,
                        detailed_processing=False,
                    )
                    report_cache[focused_key] = focused_provider_report
                focused_provider_reports.append(focused_provider_report)
                if focused_provider_report.get("parse_ok"):
                    old_candidates.extend(
                        focused_provider_report.get("entities") or []
                    )
            if any(
                entity_kind(candidate) == entity_kind(target)
                and entity_contract_key(candidate) == target_contract
                for candidate in old_candidates
                if isinstance(candidate, dict)
            ):
                provider_resolution_basis = (
                    "provider_owner_through_diagnostic_translation_unit"
                )
        if not old_candidates:
            owner_context_candidates = [
                entity
                for entity in old_owner_report.get("entities") or []
                if isinstance(entity, dict)
                and normalize_relpath(
                    str(entity.get("file") or "")
                )
                == target_file
                and entity_contract_key(entity) == target_contract
                and entity_kind(entity) == entity_kind(target)
            ]
            owner_context_candidates.extend(
                referenced
                for reference in old_owner_report.get("references") or []
                if isinstance(reference, dict)
                and isinstance(
                    reference.get("referenced_entity"),
                    dict,
                )
                and (
                    referenced := reference["referenced_entity"]
                )
                and normalize_relpath(
                    str(referenced.get("file") or "")
                )
                == target_file
                and entity_contract_key(referenced) == target_contract
                and entity_kind(referenced) == entity_kind(target)
            )
            old_candidates.extend(owner_context_candidates)
            if owner_context_candidates:
                provider_resolution_basis = (
                    "existing_reference_in_owner_translation_unit"
                )
        old_candidates = list(
            {
                str(
                    candidate.get("id")
                    or candidate.get("entity_id")
                    or (
                        f"{entity_kind(candidate)}:"
                        f"{entity_contract_key(candidate)}:"
                        f"{candidate.get('file')}:"
                        f"{_extent(candidate)}"
                    )
                ): candidate
                for candidate in old_candidates
                if isinstance(candidate, dict)
            }.values()
        )
        provider_parse_available = bool(
            old_provider_report.get("parse_ok")
            or any(
                report.get("parse_ok")
                for report in focused_provider_reports
            )
            or old_candidates
        )
        if not old_candidates and not provider_parse_available:
            unresolved.append(
                {
                    "source": "semantic_reference",
                    "symbol": symbol,
                    "contract_key": target_contract,
                    "provider_file": target_file,
                    "reason": "old_provider_parse_unavailable",
                    "parse_error": old_provider_report.get("parse_error"),
                    "focused_parse_errors": [
                        report.get("parse_error")
                        for report in focused_provider_reports
                        if report.get("parse_error")
                    ],
                }
            )
            continue
        relation = _provider_relation(
            target,
            old_candidates,
            interface_only=(
                entity_kind(target) == "type"
                and _reference_is_function_interface_type(reference)
            ),
        )
        obligation = {
            "source": "semantic_reference",
            "symbol": symbol,
            "provider_file": target_file,
            "provider_resolution_basis": provider_resolution_basis,
            **relation,
        }
        if (
            relation["status"] == "missing"
            and target_file == relative_path
            and target.get("is_definition") is True
        ):
            obligation["status"] = (
                "same_owner_dependency_materialization_required"
            )
            materializable_dependencies.append(
                {
                    key: target.get(key)
                    for key in (
                        "id",
                        "entity_id",
                        "tree",
                        "file",
                        "kind",
                        "entity_kind",
                        "symbol",
                        "contract_key",
                        "is_definition",
                        "extent",
                        "conditional_context",
                    )
                    if target.get(key) not in (None, "", [], {})
                }
            )
        obligations.append(obligation)
        if obligation["status"] not in {
            "compatible",
            "same_owner_dependency_materialization_required",
        }:
            unresolved.append(obligation)

    structural_only_symbols = structural_symbols - semantically_accounted_symbols
    old_structural_entities = old_structural_report.get("entities") or []
    donor_structural_entities = donor_structural_report.get("entities") or []
    for relation in _build_conditioned_structural_provider_relations(
        index,
        structural_only_symbols,
        old_entities=old_structural_entities,
        donor_entities=donor_structural_entities,
    ):
        obligation = {
            "source": "structural_reference",
            **relation,
        }
        obligations.append(obligation)
        if relation["status"] != "compatible":
            unresolved.append(obligation)

    result["obligations"] = obligations
    result["unresolved"] = unresolved
    result["materializable_dependencies"] = (
        materializable_dependencies
    )
    result["semantic_reference_count"] = len(semantic_references)
    result["semantic_parse"] = {
        "available": True,
        "parse_seconds": donor_report.get("parse_seconds"),
        "translation_unit_file": contextual_translation_unit or None,
        "context": (
            "build_conditioned_translation_unit"
            if contextual_translation_unit
            else "owner_file"
        ),
    }
    result["semantic_root_entities"] = [
        {
            "file": relative_path,
            "kind": entity_kind(root),
            "contract_key": entity_contract_key(root),
            "is_definition": bool(root.get("is_definition")),
            "extent": dict(root.get("extent") or {}),
        }
        for root in semantic_roots
    ]
    post_materialization_obligations = [
        obligation
        for obligation in unresolved
        if obligation.get("source")
        in {"semantic_reference", "structural_reference"}
        and obligation.get("status")
        in {"missing", "incompatible_or_ambiguous"}
        and str(obligation.get("symbol") or "")
        and str(obligation.get("contract_key") or "")
        and str(obligation.get("provider_file") or "")
        and (
            obligation.get("source") == "semantic_reference"
            or obligation.get("provider_identity_complete") is True
        )
    ]
    hard_unresolved = [
        obligation
        for obligation in unresolved
        if obligation not in post_materialization_obligations
    ]
    result["post_materialization_provider_obligations"] = (
        post_materialization_obligations
    )
    result["hard_unresolved"] = hard_unresolved
    if materializable_dependencies:
        result["status"] = "requires_dependency_materialization"
        result["reason"] = (
            "same_owner_dependency_materialization_required"
        )
        return result
    if unresolved and not hard_unresolved:
        result["status"] = (
            "admissible_with_post_materialization_provider_obligations"
        )
        result["reason"] = (
            "exact_external_provider_differences_require_iterative_validation"
        )
        return result
    if unresolved:
        result["reason"] = "dependency_contract_closure_unproven"
        return result
    result["status"] = "proven"
    result["reason"] = "all_dependency_boundaries_compatible"
    return result


def expand_entity_materialization_dependency_closure(
    spec: CaseSpec,
    *,
    relative_path: str,
    contract_key: str,
    old_structural_report: dict[str, Any],
    donor_structural_report: dict[str, Any],
    donor_root_entities: Iterable[dict[str, Any]],
    preferred_translation_units: Iterable[object] = (),
) -> dict[str, Any]:
    """Expand one materialization root to its finite same-owner dependency closure.

    Expansion follows exact semantic reference targets. It may only add a
    donor definition from the same owner file when the old tree has no
    compatible provider. The worklist is bounded by the finite set of parsed
    structural definitions in that file.
    """

    relative_path = normalize_relpath(relative_path)
    donor_entities = [
        dict(record)
        for record in donor_structural_report.get("entities") or []
        if isinstance(record, dict)
    ]
    roots: dict[str, dict[str, Any]] = {}
    for root in donor_root_entities:
        if not isinstance(root, dict):
            continue
        root_id = str(root.get("id") or root.get("entity_id") or "")
        if root_id:
            roots[root_id] = dict(root)
    result: dict[str, Any] = {
        "schema": "entity_materialization_dependency_expansion_v1",
        "status": "unproven",
        "contract_key": contract_key,
        "path": relative_path,
        "initial_root_entity_ids": sorted(roots),
        "expansion_rounds": [],
    }
    if not roots:
        result["reason"] = "materialization_root_absent"
        return result

    index = CodeIndex(
        spec,
        budget=GraphBudget(max_nodes=2_000, max_edges=6_000),
    )
    report_cache: dict[tuple[str, str, bool], dict[str, Any]] = {}
    finite_definition_count = sum(
        1
        for record in donor_entities
        if bool(record.get("is_definition"))
        and normalize_relpath(str(record.get("file") or ""))
        == relative_path
    )
    while len(result["expansion_rounds"]) <= finite_definition_count:
        ordered_roots = sorted(
            roots.values(),
            key=lambda record: (
                _extent(record)[0],
                _extent(record)[1],
                entity_contract_key(record),
            ),
        )
        proof = prove_entity_materialization_dependency_closure(
            spec,
            relative_path=relative_path,
            contract_key=contract_key,
            old_structural_report=old_structural_report,
            donor_structural_report=donor_structural_report,
            donor_root_entities=ordered_roots,
            preferred_translation_units=preferred_translation_units,
            _index=index,
            _report_cache=report_cache,
        )
        round_record: dict[str, Any] = {
            "root_entity_ids": [
                str(root.get("id") or root.get("entity_id") or "")
                for root in ordered_roots
            ],
            "root_contract_keys": [
                entity_contract_key(root) for root in ordered_roots
            ],
            "proof_status": proof.get("status"),
            "proof_reason": proof.get("reason"),
            "added_dependencies": [],
        }
        result["expansion_rounds"].append(round_record)
        if proof.get("status") == "proven":
            result.update(
                {
                    "status": "proven",
                    "reason": "finite_same_owner_dependency_closure_proven",
                    "donor_root_entity_ids": round_record[
                        "root_entity_ids"
                    ],
                    "donor_root_contract_keys": round_record[
                        "root_contract_keys"
                    ],
                    "dependency_closure": proof,
                }
            )
            return result
        if (
            proof.get("status")
            == "admissible_with_post_materialization_provider_obligations"
        ):
            result.update(
                {
                    "status": (
                        "admissible_with_post_materialization_"
                        "provider_obligations"
                    ),
                    "reason": (
                        "finite_same_owner_dependency_closure_requires_"
                        "iterative_provider_repair"
                    ),
                    "donor_root_entity_ids": round_record[
                        "root_entity_ids"
                    ],
                    "donor_root_contract_keys": round_record[
                        "root_contract_keys"
                    ],
                    "dependency_closure": proof,
                    "post_materialization_provider_obligations": list(
                        proof.get(
                            "post_materialization_provider_obligations"
                        )
                        or []
                    ),
                }
            )
            return result
        if proof.get("status") != "requires_dependency_materialization":
            result["reason"] = "dependency_closure_unproven"
            result["dependency_closure"] = proof
            return result

        additions: list[dict[str, Any]] = []
        for dependency in proof.get("materializable_dependencies") or []:
            if not isinstance(dependency, dict):
                continue
            structural = _structural_dependency_entity(
                dependency,
                relative_path=relative_path,
                donor_entities=donor_entities,
            )
            if structural is None:
                result["reason"] = (
                    "same_owner_dependency_has_no_unique_structural_definition"
                )
                result["dependency_closure"] = proof
                result["unresolved_dependency"] = dependency
                return result
            structural_id = str(
                structural.get("id")
                or structural.get("entity_id")
                or ""
            )
            if structural_id not in roots:
                additions.append(structural)
        if not additions:
            result["reason"] = "dependency_expansion_made_no_progress"
            result["dependency_closure"] = proof
            return result
        for addition in additions:
            addition_id = str(
                addition.get("id")
                or addition.get("entity_id")
                or ""
            )
            roots[addition_id] = addition
            round_record["added_dependencies"].append(
                {
                    "entity_id": addition_id,
                    "contract_key": entity_contract_key(addition),
                    "extent": addition.get("extent"),
                }
            )

    result["reason"] = "dependency_expansion_exceeded_finite_owner_definitions"
    return result
