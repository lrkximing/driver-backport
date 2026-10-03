#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

from driver_backport.analysis.code_index import (
    CodeIndex,
    parse_error_diagnostic_count,
    parse_structural_entity_file,
    structural_error_overlaps_extent,
)
from driver_backport.core.common import normalize_relpath


_COMPILE_AWARE_SUFFIXES = {".c", ".cc", ".cpp"}


def normalize_code_tree(tree: object) -> str:
    value = str(tree or "").strip().lower()
    if value in {"old", "target"}:
        return "old"
    if value in {"donor", "new", "donor_0"}:
        return "donor_0"
    return value


def entity_contract_key(entity: dict[str, Any]) -> str:
    return str(entity.get("contract_key") or entity.get("key") or "")


def entity_kind(entity: dict[str, Any]) -> str:
    return str(entity.get("kind") or entity.get("entity_kind") or "").lower()


def _entity_extent_identity(
    entity: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    extent = entity.get("extent")
    if not isinstance(extent, dict):
        return None
    try:
        identity = tuple(
            int(extent.get(field) or 0)
            for field in (
                "start_line",
                "start_column",
                "end_line",
                "end_column",
            )
        )
    except (TypeError, ValueError):
        return None
    start_line, start_column, end_line, end_column = identity
    if (
        start_line <= 0
        or start_column <= 0
        or end_line < start_line
        or end_column <= 0
    ):
        return None
    return identity


def exact_entity_provenance_match(
    descriptor: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    tree: str,
) -> dict[str, Any] | None:
    """Resolve one candidate from immutable same-tree source provenance.

    A contract may legitimately occur more than once in one file, for
    example as both a C tag declaration and a later type reference.  The
    originating descriptor identifies the exact occurrence.  Its location is
    authoritative only in the tree that produced it; donor coordinates must
    never be used to choose an old-tree candidate (or vice versa).
    """

    normalized_tree = normalize_code_tree(tree)
    descriptor_tree = normalize_code_tree(descriptor.get("tree"))
    if not descriptor_tree or descriptor_tree != normalized_tree:
        return None

    descriptor_file = normalize_relpath(
        str(descriptor.get("file") or descriptor.get("path") or "")
    )
    if not descriptor_file:
        return None
    same_origin = [
        candidate
        for candidate in candidates
        if normalize_code_tree(candidate.get("tree")) == normalized_tree
        and normalize_relpath(
            str(candidate.get("file") or candidate.get("path") or "")
        )
        == descriptor_file
    ]
    if not same_origin:
        return None

    descriptor_ids = {
        str(descriptor.get(key) or "")
        for key in ("id", "entity_id")
        if str(descriptor.get(key) or "")
    }
    if descriptor_ids:
        identity_matches = [
            candidate
            for candidate in same_origin
            if descriptor_ids
            & {
                str(candidate.get(key) or "")
                for key in ("id", "entity_id")
                if str(candidate.get(key) or "")
            }
        ]
        if len(identity_matches) == 1:
            return dict(identity_matches[0])

    descriptor_extent = _entity_extent_identity(descriptor)
    if descriptor_extent is None:
        return None
    extent_matches = [
        candidate
        for candidate in same_origin
        if _entity_extent_identity(candidate) == descriptor_extent
    ]
    return dict(extent_matches[0]) if len(extent_matches) == 1 else None


def _structural_source_extent_resolution(
    spec: object,
    *,
    entity: dict[str, Any],
    relative_path: str,
    tree: str,
    report_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve a complete source-edit extent for one semantic entity.

    Compile-aware parsing remains authoritative for entity presence and
    identity. During a broken intermediate build, however, libclang can still
    return a definition whose extent ends at the first invalid token. A
    structural parser may supply the complete declaration boundary only when
    it resolves the same exact contract unambiguously and that boundary does
    not overlap a structural parse error.
    """

    normalized_tree = normalize_code_tree(tree)
    root = (
        Path(getattr(spec, "old_root_path"))
        if normalized_tree == "old"
        else Path(getattr(spec, "donor_root_paths")[0])
        if normalized_tree == "donor_0"
        else None
    )
    contract = entity_contract_key(entity)
    kind = entity_kind(entity)
    base = {
        "schema": "structural_source_extent_resolution_v1",
        "tree": normalized_tree,
        "file": relative_path,
        "contract_key": contract or None,
        "entity_kind": kind or None,
        "semantic_entity_id": (
            entity.get("id") or entity.get("entity_id")
        ),
        "semantic_extent": dict(entity.get("extent") or {}),
    }
    if root is None or not contract or not kind:
        return {
            **base,
            "status": "unavailable",
            "reason": "tree_or_exact_contract_unavailable",
        }

    structural_cache_key = (
        f"structural_source_extent:{normalized_tree}",
        relative_path,
    )
    structural_report = (
        report_cache.get(structural_cache_key)
        if report_cache is not None
        else None
    )
    if structural_report is None:
        structural_report = parse_structural_entity_file(
            root=root,
            relative_path=relative_path,
            tree=normalized_tree,
        )
        if report_cache is not None:
            report_cache[structural_cache_key] = structural_report
    if not structural_report.get("parse_ok"):
        return {
            **base,
            "status": "unavailable",
            "reason": "structural_parse_unavailable",
            "backend": structural_report.get("backend"),
            "parse_error": structural_report.get("parse_error"),
        }

    definition_state = entity.get("is_definition")
    candidates = [
        candidate
        for candidate in structural_report.get("entities") or []
        if isinstance(candidate, dict)
        and entity_contract_key(candidate) == contract
        and entity_kind(candidate) == kind
        and (
            not isinstance(definition_state, bool)
            or candidate.get("is_definition") is definition_state
        )
    ]
    semantic_extent = entity.get("extent") or {}
    try:
        semantic_start = (
            int(semantic_extent.get("start_line") or 0),
            int(semantic_extent.get("start_column") or 0),
        )
    except (TypeError, ValueError):
        semantic_start = (0, 0)
    same_start = [
        candidate
        for candidate in candidates
        if semantic_start[0] > 0
        and semantic_start[1] > 0
        and (
            int((candidate.get("extent") or {}).get("start_line") or 0),
            int((candidate.get("extent") or {}).get("start_column") or 0),
        )
        == semantic_start
    ]
    if len(same_start) == 1:
        selected = same_start[0]
        match_basis = "exact_contract_kind_definition_and_start"
    elif len(same_start) > 1:
        return {
            **base,
            "status": "ambiguous",
            "reason": "multiple_structural_entities_share_semantic_start",
            "match_count": len(same_start),
            "backend": structural_report.get("backend"),
        }
    elif len(candidates) == 1:
        selected = candidates[0]
        match_basis = "unique_exact_contract_kind_and_definition"
    else:
        return {
            **base,
            "status": "ambiguous" if candidates else "unavailable",
            "reason": (
                "multiple_exact_structural_contract_matches"
                if candidates
                else "exact_structural_contract_absent"
            ),
            "match_count": len(candidates),
            "backend": structural_report.get("backend"),
        }

    source_extent = dict(selected.get("extent") or {})
    extent_identity = _entity_extent_identity(selected)
    if extent_identity is None:
        return {
            **base,
            "status": "unavailable",
            "reason": "structural_extent_invalid",
            "backend": structural_report.get("backend"),
        }
    start_line, _start_column, end_line, _end_column = extent_identity
    if structural_error_overlaps_extent(
        structural_report,
        start_line=start_line,
        end_line=end_line,
    ):
        return {
            **base,
            "status": "bounded_with_internal_errors",
            "reason": (
                "the unique exact structural contract still provides its "
                "complete edit boundary, but internal syntax errors prevent "
                "using it as semantic-validity evidence"
            ),
            "match_basis": match_basis,
            "match_count": 1,
            "backend": structural_report.get("backend"),
            "structural_entity_id": (
                selected.get("id") or selected.get("entity_id")
            ),
            "source_extent": source_extent,
            "edit_boundary_only": True,
        }
    return {
        **base,
        "status": "proven",
        "reason": "exact_structural_contract_has_complete_error_free_extent",
        "match_basis": match_basis,
        "match_count": 1,
        "backend": structural_report.get("backend"),
        "structural_entity_id": (
            selected.get("id") or selected.get("entity_id")
        ),
        "source_extent": source_extent,
    }


def case_aware_entity_report(
    spec: object,
    *,
    relative_path: str,
    tree: str,
    index: CodeIndex | None = None,
    report_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parse one owner file in the case's active compile configuration.

    A source translation unit is authoritative only when it can be parsed
    through the case compile command. Structural parsing remains available as
    evidence when that parse is unavailable, but callers can distinguish the
    fallback from a proof that an entity is active or absent.
    """

    relative_path = normalize_relpath(relative_path)
    normalized_tree = normalize_code_tree(tree)
    cache_key = (normalized_tree, relative_path)
    if report_cache is not None and cache_key in report_cache:
        return report_cache[cache_key]

    def inventory_is_error_free(report: dict[str, Any]) -> bool:
        if not report.get("parse_ok") or report.get("has_error_nodes"):
            return False
        persisted_error_count = report.get("error_diagnostic_count")
        if (
            report.get("restored_from_graph_snapshot")
            and persisted_error_count is None
        ):
            return False
        error_count = (
            int(persisted_error_count)
            if persisted_error_count is not None
            else parse_error_diagnostic_count(
                report.get("diagnostics") or []
            )
        )
        return error_count == 0

    root = (
        Path(getattr(spec, "old_root_path"))
        if normalized_tree == "old"
        else Path(getattr(spec, "donor_root_paths")[0])
        if normalized_tree == "donor_0"
        else None
    )
    if root is None:
        report = {
            "file": relative_path,
            "tree": normalized_tree,
            "exists": False,
            "parse_ok": False,
            "parse_error": "unknown_tree",
            "entities": [],
            "references": [],
            "backend": "unavailable",
            "compile_aware": False,
            "owner_presence_authoritative": False,
        }
    elif Path(relative_path).suffix.lower() in _COMPILE_AWARE_SUFFIXES:
        code_index = index or CodeIndex(spec, graph_projection="records")
        semantic_report = code_index.index_file(
            normalized_tree,
            relative_path,
            detailed_processing=True,
        )
        if semantic_report.get("parse_ok"):
            report = {
                **semantic_report,
                "backend": "libclang_case_compile_context",
                "compile_aware": True,
                "owner_presence_authoritative": (
                    inventory_is_error_free(semantic_report)
                ),
                "compile_context": dict(
                    code_index.context_reports.get(normalized_tree) or {}
                ),
            }
        else:
            structural_report = parse_structural_entity_file(
                root=root,
                relative_path=relative_path,
                tree=normalized_tree,
            )
            report = {
                **structural_report,
                "backend": str(
                    structural_report.get("backend")
                    or "tree_sitter_c_structural_entities"
                ),
                "compile_aware": False,
                "owner_presence_authoritative": False,
                "compile_aware_failure": {
                    "parse_error": semantic_report.get("parse_error"),
                    "parse_exception": semantic_report.get(
                        "parse_exception"
                    ),
                    "diagnostics": list(
                        semantic_report.get("diagnostics") or []
                    ),
                },
                "compile_context": dict(
                    code_index.context_reports.get(normalized_tree) or {}
                ),
            }
    else:
        structural_report = parse_structural_entity_file(
            root=root,
            relative_path=relative_path,
            tree=normalized_tree,
        )
        report = {
            **structural_report,
            "compile_aware": False,
            "owner_presence_authoritative": (
                inventory_is_error_free(structural_report)
            ),
        }

    if report_cache is not None:
        report_cache[cache_key] = report
    return report


def resolve_entity_in_case_tree(
    spec: object,
    *,
    descriptor: dict[str, Any],
    relative_path: str | None = None,
    tree: str,
    require_definition: bool = True,
    index: CodeIndex | None = None,
    report_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve an exact entity contract and preserve absence uncertainty."""

    path = normalize_relpath(
        relative_path or str(descriptor.get("file") or descriptor.get("path") or "")
    )
    contract = entity_contract_key(descriptor)
    kind = entity_kind(descriptor)
    symbol = str(descriptor.get("symbol") or descriptor.get("name") or "")
    source_call_anchor = bool(
        descriptor.get("is_source_anchor")
        and kind == "top_level_call_expression"
    )
    if source_call_anchor:
        normalized_tree = normalize_code_tree(tree)
        root = (
            Path(getattr(spec, "old_root_path"))
            if normalized_tree == "old"
            else Path(getattr(spec, "donor_root_paths")[0])
            if normalized_tree == "donor_0"
            else None
        )
        report = (
            parse_structural_entity_file(
                root=root, relative_path=path, tree=normalized_tree,
            )
            if root is not None
            else {"exists": False, "parse_ok": False}
        )
    else:
        report = case_aware_entity_report(
            spec,
            relative_path=path,
            tree=tree,
            index=index,
            report_cache=report_cache,
        )
    base = {
        "schema": "case_entity_resolution_v1",
        "tree": normalize_code_tree(tree),
        "file": path,
        "contract_key": contract or None,
        "entity_kind": kind or None,
        "symbol": symbol or None,
        "backend": report.get("backend"),
        "compile_aware": bool(report.get("compile_aware")),
        "owner_presence_authoritative": bool(
            report.get("owner_presence_authoritative")
        ),
        "parse_ok": bool(report.get("parse_ok")),
        "parse_error": report.get("parse_error"),
        "compile_aware_failure": report.get("compile_aware_failure"),
    }
    if not path or not report.get("exists", True):
        return {
            **base,
            "status": "file_absent",
            "match_count": 0,
            "matches": [],
            "entity": None,
        }
    if not report.get("parse_ok"):
        return {
            **base,
            "status": "parse_unavailable",
            "match_count": 0,
            "matches": [],
            "entity": None,
        }

    entities = [
        item
        for item in report.get(
            "source_anchors" if source_call_anchor else "entities"
        ) or []
        if isinstance(item, dict)
    ]
    if contract:
        matches = [
            item
            for item in entities
            if entity_contract_key(item) == contract
        ]
    else:
        matches = [
            item
            for item in entities
            if (not kind or entity_kind(item) == kind)
            and (not symbol or str(item.get("symbol") or item.get("name") or "") == symbol)
        ]
    if kind:
        matches = [item for item in matches if entity_kind(item) == kind]
    if require_definition and not source_call_anchor:
        matches = [
            item for item in matches if item.get("is_definition") is True
        ]
    elif not source_call_anchor and descriptor.get("is_definition") is False:
        matches = [
            item for item in matches if item.get("is_definition") is False
        ]

    contract_match_count = len(matches)
    resolution_basis = ""
    if len(matches) > 1:
        exact_match = exact_entity_provenance_match(
            descriptor,
            matches,
            tree=tree,
        )
        if exact_match is not None:
            matches = [exact_match]
            resolution_basis = "exact_same_tree_descriptor_provenance"

    if len(matches) == 1:
        status = "resolved"
        entity = dict(matches[0])
        if not resolution_basis:
            resolution_basis = "unique_contract_match"
    elif matches:
        status = "ambiguous"
        entity = None
        resolution_basis = "multiple_contract_matches"
    elif report.get("owner_presence_authoritative"):
        status = "entity_absent"
        entity = None
        resolution_basis = "authoritative_owner_absence"
    else:
        status = "absence_unproven"
        entity = None
        resolution_basis = "owner_absence_not_authoritative"
    source_extent_resolution = None
    if status == "resolved" and entity is not None and report.get(
        "compile_aware"
    ):
        source_extent_resolution = _structural_source_extent_resolution(
            spec,
            entity=entity,
            relative_path=path,
            tree=tree,
            report_cache=report_cache,
        )
        if source_extent_resolution.get("status") in {
            "proven",
            "bounded_with_internal_errors",
        }:
            entity["source_extent"] = dict(
                source_extent_resolution.get("source_extent") or {}
            )
            entity["source_extent_authority"] = {
                key: value
                for key, value in source_extent_resolution.items()
                if key not in {"semantic_extent", "source_extent"}
            }
    return {
        **base,
        "status": status,
        "match_count": len(matches),
        "contract_match_count": contract_match_count,
        "resolution_basis": resolution_basis,
        "matches": [dict(item) for item in matches],
        "entity": entity,
        "source_extent_resolution": source_extent_resolution,
    }
