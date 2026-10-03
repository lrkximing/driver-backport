from __future__ import annotations

import difflib
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from driver_backport.analysis.code_index import (
    STRUCTURAL_ALIGNMENT_INSERTION_STATUSES,
    CodeIndex,
    _extent_contains_extent,
    parse_structural_entity_file,
    structural_entity_neighbor_alignment,
    structural_error_crosses_insertion_boundary,
    structural_error_overlaps_extent,
)
from driver_backport.analysis.entity_resolution import (
    exact_entity_provenance_match,
    resolve_entity_in_case_tree,
)
from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_strings as _ordered_strings,
    stable_id,
)
from driver_backport.migration.owner_key import (
    semantic_open_mutation_dimensions,
    semantic_owner_mutation_dimensions,
)
from driver_backport.migration.semantic_closure_evaluator import (
    integration_reference_target_representation_agnostic,
    reference_shape,
)
from driver_backport.migration.semantic_closure_store import (
    load_semantic_closure_state,
    load_semantic_record,
    load_initial_analysis_artifact,
)
from driver_backport.planning.agentic_repair import (
    build_exact_entity_materialization_action,
)
from driver_backport.planning.sanity_check import (
    unified_diff_changed_ranges,
)


SEMANTIC_TOOL_EVIDENCE_SCHEMA = "semantic_closure_tool_evidence_v1"

def _compact_descriptor(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: deepcopy(value.get(key))
        for key in (
            "id",
            "entity_id",
            "tree",
            "file",
            "path",
            "symbol",
            "name",
            "kind",
            "entity_kind",
            "cursor_kind",
            "contract_key",
            "key",
            "container",
            "is_definition",
            "extent",
            "syntax_fingerprint",
            "source_fingerprint",
            "signature_fingerprint",
            "interface_fingerprint",
            "implementation_fingerprint",
        )
        if value.get(key) not in (None, "", [], {})
    }


def _compact_resolution(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value.get(key))
        for key in (
            "status",
            "tree",
            "file",
            "contract_key",
            "entity_kind",
            "symbol",
            "backend",
            "compile_aware",
            "owner_presence_authoritative",
            "parse_ok",
            "parse_error",
            "match_count",
            "contract_match_count",
            "resolution_basis",
        )
        if value.get(key) not in (None, "", [], {})
    }


def _merge_unique(target: dict[str, Any], key: str, values: Iterable[object]) -> None:
    target[key] = _ordered_strings(
        [
            *list(target.get(key) or []),
            *list(values),
        ]
    )


def _drop_dominated_edit_handles(
    handles: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Remove regions whose write authority is fully covered by another.

    Dominance is structural: the containing region must be in the same file
    and carry every semantic gap authorized by the contained region.  The
    removed region's provenance remains attached to the surviving handle.
    """

    retained = {key: deepcopy(value) for key, value in handles.items()}
    ordered = sorted(
        retained,
        key=lambda handle_id: (
            str(retained[handle_id].get("file") or ""),
            int(retained[handle_id].get("start_line") or 0),
            -int(retained[handle_id].get("end_line") or 0),
            handle_id,
        ),
    )
    removed: set[str] = set()
    for inner_id in ordered:
        if inner_id in removed:
            continue
        inner = retained[inner_id]
        inner_gaps = set(_ordered_strings(inner.get("gap_ids") or []))
        for outer_id in ordered:
            if outer_id == inner_id or outer_id in removed:
                continue
            outer = retained[outer_id]
            if (
                str(outer.get("file") or "")
                != str(inner.get("file") or "")
                or int(outer.get("start_line") or 0)
                > int(inner.get("start_line") or 0)
                or int(outer.get("end_line") or 0)
                < int(inner.get("end_line") or 0)
            ):
                continue
            outer_gaps = set(_ordered_strings(outer.get("gap_ids") or []))
            if not inner_gaps or not inner_gaps <= outer_gaps:
                continue
            if (
                int(outer.get("start_line") or 0)
                == int(inner.get("start_line") or 0)
                and int(outer.get("end_line") or 0)
                == int(inner.get("end_line") or 0)
                and outer_id > inner_id
            ):
                continue
            for key in (
                "authorities",
                "owner_keys",
                "source_owner_keys",
                "gap_ids",
                "repair_dimensions",
                "read_handle_ids",
            ):
                _merge_unique(outer, key, inner.get(key) or [])
            outer.setdefault("subsumed_regions", []).append(
                {
                    "edit_handle_id": inner_id,
                    "start_line": int(inner.get("start_line") or 0),
                    "end_line": int(inner.get("end_line") or 0),
                    "source_sha256": str(
                        inner.get("source_sha256") or ""
                    ),
                }
            )
            removed.add(inner_id)
            break
    return {
        handle_id: retained[handle_id]
        for handle_id in ordered
        if handle_id not in removed
    }


def _descriptor_extent(
    descriptor: dict[str, Any],
) -> tuple[int, int] | None:
    extent = (
        descriptor.get("extent")
        if isinstance(descriptor.get("extent"), dict)
        else {}
    )
    start = int(extent.get("start_line") or 0)
    end = int(extent.get("end_line") or start)
    if start <= 0 or end < start:
        return None
    return start, end


def _descriptor_contract_identity(
    descriptor: dict[str, Any],
) -> tuple[str, str]:
    return (
        str(
            descriptor.get("kind")
            or descriptor.get("entity_kind")
            or ""
        ),
        str(
            descriptor.get("contract_key")
            or descriptor.get("key")
            or ""
        ),
    )


def _unique_structural_entity(
    report: dict[str, Any],
    descriptor: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve one descriptor against a structural owner-file report."""

    expected_kind, expected_contract = _descriptor_contract_identity(
        descriptor
    )
    if not expected_kind or not expected_contract:
        return None
    expected_definition = descriptor.get("is_definition")
    matches = [
        entity
        for entity in report.get("entities") or []
        if isinstance(entity, dict)
        and _descriptor_contract_identity(entity)
        == (expected_kind, expected_contract)
        and (
            expected_definition is None
            or bool(entity.get("is_definition"))
            is bool(expected_definition)
        )
    ]
    if len(matches) == 1:
        return matches[0]
    return exact_entity_provenance_match(
        descriptor,
        matches,
        tree=str(report.get("tree") or descriptor.get("tree") or ""),
    )


def _unique_containing_source_anchor(
    report: dict[str, Any],
    descriptor: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve a generated descriptor to one exact file-scope syntax owner."""

    descriptor_extent = (
        descriptor.get("source_extent")
        if isinstance(descriptor.get("source_extent"), dict)
        else descriptor.get("extent")
        if isinstance(descriptor.get("extent"), dict)
        else {}
    )
    matches = [
        anchor
        for anchor in report.get("source_anchors") or []
        if isinstance(anchor, dict)
        and str(anchor.get("kind") or "")
        == "top_level_call_expression"
        and _extent_contains_extent(
            dict(anchor.get("extent") or {}),
            descriptor_extent,
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _unique_outermost_structural_owner(
    report: dict[str, Any],
    anchor: dict[str, Any],
) -> dict[str, Any] | None:
    """Promote a nested neighbor to its unique complete source owner."""

    anchor_extent = (
        anchor.get("extent")
        if isinstance(anchor.get("extent"), dict)
        else {}
    )
    anchor_context = tuple(
        str(item) for item in anchor.get("conditional_context") or []
    )

    containing_by_extent: dict[
        tuple[int, int, int, int], dict[str, Any]
    ] = {}
    for entity in report.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        extent = dict(entity.get("extent") or {})
        extent_key = tuple(
            int(extent.get(field) or 0)
            for field in (
                "start_line",
                "start_column",
                "end_line",
                "end_column",
            )
        )
        if (
            extent == anchor_extent
            or tuple(
                str(item)
                for item in entity.get("conditional_context") or []
            )
            != anchor_context
            or not _extent_contains_extent(extent, anchor_extent)
        ):
            continue
        containing_by_extent.setdefault(extent_key, entity)
    if not containing_by_extent:
        return dict(anchor)

    outermost = [
        (extent, entity)
        for extent, entity in containing_by_extent.items()
        if not any(
            other_extent != extent
            and _extent_contains_extent(
                dict(containing_by_extent[other_extent].get("extent") or {}),
                dict(entity.get("extent") or {}),
            )
            for other_extent in containing_by_extent
        )
    ]
    return dict(outermost[0][1]) if len(outermost) == 1 else None


def _structural_insertion_anchor(
    *,
    old_root: Path,
    donor_root: Path,
    descriptor: dict[str, Any],
    owner_key: str,
    gap_ids: Iterable[object],
    report_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Authorize a missing entity insertion at an exact structural boundary."""

    relative_path = normalize_relpath(
        str(descriptor.get("file") or descriptor.get("path") or "")
    )
    base = {
        "status": "unavailable",
        "owner_key": owner_key,
        "file": relative_path,
        "gap_ids": _ordered_strings(gap_ids),
        "target_descriptor": _compact_descriptor(descriptor),
    }
    if not relative_path:
        return {**base, "reason": "missing_entity_owner_file_absent"}

    reports: dict[str, dict[str, Any]] = {}
    for tree, root in (("old", old_root), ("donor", donor_root)):
        cache_key = (tree, relative_path)
        report = report_cache.get(cache_key)
        if report is None:
            report = parse_structural_entity_file(
                root=root,
                relative_path=relative_path,
                tree=tree,
            )
            report_cache[cache_key] = report
        reports[tree] = report
        if not report.get("parse_ok"):
            return {
                **base,
                "reason": "owner_file_structural_parse_unavailable",
                "tree": tree,
                "parse_error": report.get("parse_error"),
            }

    target = _unique_structural_entity(reports["donor"], descriptor)
    target_resolution = "exact_contract_entity"
    if target is None:
        target = _unique_containing_source_anchor(
            reports["donor"],
            descriptor,
        )
        target_resolution = "exact_containing_source_anchor"
    if target is None:
        return {
            **base,
            "reason": "donor_target_contract_absent_or_ambiguous",
        }
    target_extent = _descriptor_extent(target)
    if target_extent is None or structural_error_overlaps_extent(
        reports["donor"],
        start_line=target_extent[0],
        end_line=target_extent[1],
    ):
        return {
            **base,
            "reason": "donor_target_extent_is_not_structurally_complete",
        }

    alignment = structural_entity_neighbor_alignment(
        target_entity_id=str(
            target.get("id") or target.get("entity_id") or ""
        ),
        donor_entities=[
            *(reports["donor"].get("entities") or []),
            *(reports["donor"].get("source_anchors") or []),
        ],
        old_entities=[
            *(reports["old"].get("entities") or []),
            *(reports["old"].get("source_anchors") or []),
        ],
        include_declarations=target.get("is_definition") is False,
    )
    if (
        alignment.get("status")
        not in STRUCTURAL_ALIGNMENT_INSERTION_STATUSES
        or not alignment.get("conditional_context_aligned")
    ):
        return {
            **base,
            "reason": "missing_entity_has_no_exact_structural_boundary",
            "alignment": alignment,
        }

    predecessor = alignment.get("predecessor")
    successor = alignment.get("successor")
    if isinstance(predecessor, dict) and isinstance(
        predecessor.get("old"), dict
    ):
        edit_mode = "insert_after_anchor"
        anchor = dict(predecessor["old"])
        anchor_extent = _descriptor_extent(anchor)
        boundary_line = anchor_extent[1] if anchor_extent else 0
        insertion_index = boundary_line
    elif isinstance(successor, dict) and isinstance(
        successor.get("old"), dict
    ):
        edit_mode = "insert_before_anchor"
        anchor = dict(successor["old"])
        anchor_extent = _descriptor_extent(anchor)
        boundary_line = anchor_extent[0] if anchor_extent else 0
        insertion_index = boundary_line - 1
    else:
        return {
            **base,
            "reason": "structural_alignment_has_no_old_anchor",
            "alignment": alignment,
        }

    if target_resolution == "exact_containing_source_anchor":
        anchor = _unique_outermost_structural_owner(
            reports["old"],
            anchor,
        )
        if anchor is None:
            return {
                **base,
                "reason": "old_structural_anchor_owner_is_ambiguous",
                "alignment": alignment,
            }
        anchor_extent = _descriptor_extent(anchor)
        if edit_mode == "insert_after_anchor":
            boundary_line = anchor_extent[1] if anchor_extent else 0
            insertion_index = boundary_line
        else:
            boundary_line = anchor_extent[0] if anchor_extent else 0
            insertion_index = boundary_line - 1

    if (
        boundary_line <= 0
        or normalize_relpath(str(anchor.get("file") or ""))
        != relative_path
        or structural_error_crosses_insertion_boundary(
            reports["old"],
            insertion_index=insertion_index,
        )
    ):
        return {
            **base,
            "reason": "old_structural_anchor_is_not_a_safe_boundary",
            "alignment": alignment,
        }
    return {
        **base,
        "status": "resolved",
        "authority": "exact_missing_entity_structural_insertion_anchor",
        "edit_mode": edit_mode,
        "start_line": boundary_line,
        "end_line": boundary_line,
        "insertion_index": insertion_index,
        "target_resolution": target_resolution,
        "anchor_resolution": (
            "unique_outermost_structural_owner"
            if target_resolution == "exact_containing_source_anchor"
            else "aligned_structural_neighbor"
        ),
        "target_descriptor": _compact_descriptor(target),
        "anchor_descriptor": _compact_descriptor(anchor),
        "alignment": alignment,
    }


def _source_lines(
    root: Path,
    *,
    relative_path: str,
    start_line: int,
    end_line: int,
) -> dict[str, Any]:
    path = (root / normalize_relpath(relative_path)).resolve()
    if not path.is_file():
        return {
            "status": "file_absent",
            "file": normalize_relpath(relative_path),
            "start_line": int(start_line),
            "end_line": int(end_line),
            "lines": [],
        }
    source = path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    source_lines = source.splitlines(keepends=True)
    display_lines = source.splitlines()
    if (
        int(start_line) <= 0
        or int(end_line) < int(start_line)
        or int(end_line) > len(source_lines)
    ):
        return {
            "status": "range_outside_file",
            "file": normalize_relpath(relative_path),
            "start_line": int(start_line),
            "end_line": int(end_line),
            "line_count": len(source_lines),
            "lines": [],
        }
    region = "".join(
        source_lines[int(start_line) - 1 : int(end_line)]
    )
    return {
        "status": "resolved",
        "file": normalize_relpath(relative_path),
        "start_line": int(start_line),
        "end_line": int(end_line),
        "source": region,
        "source_sha256": hashlib.sha256(
            region.encode("utf-8")
        ).hexdigest(),
        "lines": [
            {
                "line": number,
                "text": display_lines[number - 1],
            }
            for number in range(int(start_line), int(end_line) + 1)
        ],
    }


def _deduplicate_ranges(
    ranges: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Merge structurally identical ranges without widening authority."""

    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, values in ranges.items():
        merged: dict[tuple[int, int, str], dict[str, Any]] = {}
        for raw in values:
            if not isinstance(raw, dict):
                continue
            start = int(raw.get("start_line") or 0)
            end = int(raw.get("end_line") or start)
            authority = str(raw.get("authority") or "")
            if start <= 0 or end < start or not authority:
                continue
            identity = (start, end, authority)
            record = merged.setdefault(
                identity,
                {
                    "start_line": start,
                    "end_line": end,
                    "authority": authority,
                    "owner_keys": [],
                    "gap_ids": [],
                    "entity_ids": [],
                    "hunk_ids": [],
                    "patch_ids": [],
                },
            )
            for source_key, target_key in (
                ("owner_key", "owner_keys"),
                ("entity_id", "entity_ids"),
                ("hunk_id", "hunk_ids"),
                ("patch_id", "patch_ids"),
            ):
                value = str(raw.get(source_key) or "")
                if value and value not in record[target_key]:
                    record[target_key].append(value)
            record["gap_ids"] = _ordered_strings(
                [
                    *record["gap_ids"],
                    *list(raw.get("gap_ids") or []),
                ]
            )
        result[key] = [
            merged[identity]
            for identity in sorted(merged)
        ]
    return result


def _source_edit_diff(
    root: Path,
    *,
    edits: Iterable[dict[str, Any]],
) -> str:
    """Construct one atomic diff from exact replacements and insertions.

    A structural insertion handle includes its unchanged anchor in the
    replacement text.  That anchor may also be the first or last line of an
    independently authorized replacement.  Compose that shared boundary
    deterministically; reject overlaps whose position in the replacement
    postimage cannot be proven from the advertised ranges.
    """

    def insertion_order(item: dict[str, Any]) -> tuple[int, str, str]:
        target_starts = [
            int(
                (
                    (target.get("descriptor") or {}).get("extent")
                    or {}
                ).get("start_line")
                or 0
            )
            for target in item.get("insertion_targets") or []
            if isinstance(target, dict)
        ]
        positive_starts = [value for value in target_starts if value > 0]
        return (
            min(positive_starts) if positive_starts else 0,
            str(item.get("file") or ""),
            str(item.get("id") or ""),
        )

    def overlap_error(
        relative_path: str,
        first: dict[str, Any],
        second: dict[str, Any],
    ) -> ValueError:
        def describe(item: dict[str, Any]) -> str:
            return (
                f"{str(item.get('id') or '<unknown>')}"
                f"[{int(item.get('start_line') or 0)}:"
                f"{int(item.get('end_line') or 0)} "
                f"{str(item.get('edit_mode') or 'replace')}]"
            )

        return ValueError(
            "source edit handles overlap without a composable exact "
            f"boundary in {relative_path}: {describe(first)} conflicts "
            f"with {describe(second)}"
        )

    def inserted_text(item: dict[str, Any]) -> str:
        anchor = str(item.get("source") or "")
        replacement = str(item.get("replacement_source") or "")
        mode = str(item.get("edit_mode") or "")
        if mode == "insert_after_anchor":
            return replacement[len(anchor) :]
        if mode == "insert_before_anchor":
            return replacement[: len(replacement) - len(anchor)]
        raise ValueError(f"unsupported structural insertion mode: {mode}")

    def canonical_file_edits(
        relative_path: str,
        file_edits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        replacements = sorted(
            [
                deepcopy(item)
                for item in file_edits
                if not str(item.get("edit_mode") or "").startswith(
                    "insert_"
                )
            ],
            key=lambda item: (
                int(item["start_line"]),
                int(item["end_line"]),
                str(item.get("id") or ""),
            ),
        )
        for previous, current in zip(replacements, replacements[1:]):
            if int(current["start_line"]) <= int(previous["end_line"]):
                raise overlap_error(relative_path, previous, current)

        insertions = sorted(
            [
                deepcopy(item)
                for item in file_edits
                if str(item.get("edit_mode") or "").startswith(
                    "insert_"
                )
            ],
            key=lambda item: (
                int(item["start_line"]),
                int(item["end_line"]),
                insertion_order(item),
            ),
        )
        standalone: dict[tuple[int, int], list[dict[str, Any]]] = {}
        prefixes: dict[str, list[dict[str, Any]]] = {}
        suffixes: dict[str, list[dict[str, Any]]] = {}
        for insertion in insertions:
            start = int(insertion["start_line"])
            end = int(insertion["end_line"])
            overlapping = [
                replacement
                for replacement in replacements
                if max(start, int(replacement["start_line"]))
                <= min(end, int(replacement["end_line"]))
            ]
            if not overlapping:
                standalone.setdefault((start, end), []).append(insertion)
                continue
            if len(overlapping) != 1:
                raise overlap_error(
                    relative_path,
                    overlapping[0],
                    insertion,
                )
            replacement = overlapping[0]
            replacement_id = str(replacement.get("id") or "")
            mode = str(insertion.get("edit_mode") or "")
            contained = (
                start >= int(replacement["start_line"])
                and end <= int(replacement["end_line"])
            )
            if (
                contained
                and mode == "insert_before_anchor"
                and start == int(replacement["start_line"])
            ):
                prefixes.setdefault(replacement_id, []).append(insertion)
            elif (
                contained
                and mode == "insert_after_anchor"
                and end == int(replacement["end_line"])
            ):
                suffixes.setdefault(replacement_id, []).append(insertion)
            else:
                raise overlap_error(relative_path, replacement, insertion)

        canonical: list[dict[str, Any]] = []
        for replacement in replacements:
            replacement_id = str(replacement.get("id") or "")
            prefix = "".join(
                inserted_text(item)
                for item in sorted(
                    prefixes.get(replacement_id, []),
                    key=insertion_order,
                )
            )
            suffix = "".join(
                inserted_text(item)
                for item in sorted(
                    suffixes.get(replacement_id, []),
                    key=insertion_order,
                )
            )
            replacement["replacement_source"] = (
                prefix
                + str(replacement.get("replacement_source") or "")
                + suffix
            )
            canonical.append(replacement)

        for (_start, _end), grouped in sorted(standalone.items()):
            ordered_group = sorted(grouped, key=insertion_order)
            anchor = str(ordered_group[0].get("source") or "")
            if any(
                str(item.get("source") or "") != anchor
                for item in ordered_group[1:]
            ):
                raise overlap_error(
                    relative_path,
                    ordered_group[0],
                    ordered_group[1],
                )
            before = "".join(
                inserted_text(item)
                for item in ordered_group
                if item.get("edit_mode") == "insert_before_anchor"
            )
            after = "".join(
                inserted_text(item)
                for item in ordered_group
                if item.get("edit_mode") == "insert_after_anchor"
            )
            combined = deepcopy(ordered_group[0])
            combined["replacement_source"] = before + anchor + after
            canonical.append(combined)

        canonical.sort(
            key=lambda item: (
                int(item["start_line"]),
                int(item["end_line"]),
                str(item.get("id") or ""),
            )
        )
        for previous, current in zip(canonical, canonical[1:]):
            if int(current["start_line"]) <= int(previous["end_line"]):
                raise overlap_error(relative_path, previous, current)
        return canonical

    by_file: dict[str, list[dict[str, Any]]] = {}
    for edit in edits:
        relative_path = normalize_relpath(str(edit.get("file") or ""))
        if relative_path:
            by_file.setdefault(relative_path, []).append(dict(edit))
    diff_lines: list[str] = []
    for relative_path in sorted(by_file):
        path = (root / relative_path).resolve()
        original = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        original_lines = original.splitlines(keepends=True)
        updated_lines = list(original_lines)
        ordered = canonical_file_edits(
            relative_path,
            by_file[relative_path],
        )
        for item in reversed(ordered):
            start = int(item["start_line"])
            end = int(item["end_line"])
            replacement = str(item.get("replacement_source") or "")
            replaced_lines = original_lines[start - 1 : end]
            if replacement and replaced_lines:
                original_ending = (
                    "\r\n"
                    if replaced_lines[-1].endswith("\r\n")
                    else "\n"
                    if replaced_lines[-1].endswith("\n")
                    else ""
                )
                if original_ending and not replacement.endswith(
                    ("\n", "\r")
                ):
                    replacement += original_ending
            updated_lines[start - 1 : end] = replacement.splitlines(
                keepends=True
            )
        if updated_lines == original_lines:
            continue
        diff_lines.extend(
            difflib.unified_diff(
                original_lines,
                updated_lines,
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                n=3,
            )
        )
    return "".join(diff_lines)


def _descriptor_identity(
    descriptor: dict[str, Any],
) -> tuple[str, str, str, str]:
    return (
        normalize_relpath(
            str(descriptor.get("file") or descriptor.get("path") or "")
        ),
        str(
            descriptor.get("contract_key")
            or descriptor.get("key")
            or ""
        ),
        str(
            descriptor.get("kind")
            or descriptor.get("entity_kind")
            or ""
        ),
        str(descriptor.get("symbol") or descriptor.get("name") or ""),
    )


def _same_entity_contract(
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    left_identity = _descriptor_identity(left)
    right_identity = _descriptor_identity(right)
    if not left_identity[0] or not left_identity[1]:
        return False
    if left_identity[:2] != right_identity[:2]:
        return False
    return all(
        not left_value
        or not right_value
        or left_value == right_value
        for left_value, right_value in zip(
            left_identity[2:],
            right_identity[2:],
        )
    )


def _exact_definition_source_proof(
    spec: object,
    *,
    current_resolution: dict[str, Any],
    donor_resolution: dict[str, Any],
) -> dict[str, Any]:
    """Prove equality from the exact current and donor definition extents."""

    current = (
        current_resolution.get("entity")
        if isinstance(current_resolution.get("entity"), dict)
        else {}
    )
    donor = (
        donor_resolution.get("entity")
        if isinstance(donor_resolution.get("entity"), dict)
        else {}
    )
    if (
        current_resolution.get("status") != "resolved"
        or donor_resolution.get("status") != "resolved"
        or not _same_entity_contract(current, donor)
        or current.get("is_definition") is False
        or donor.get("is_definition") is False
    ):
        return {"status": "not_proven"}
    current_extent = _descriptor_extent(current)
    donor_extent = _descriptor_extent(donor)
    current_file = _descriptor_identity(current)[0]
    donor_file = _descriptor_identity(donor)[0]
    if (
        not current_extent
        or not donor_extent
        or not current_file
        or not donor_file
    ):
        return {"status": "not_proven"}
    current_source = _source_lines(
        Path(getattr(spec, "old_root_path")),
        relative_path=current_file,
        start_line=current_extent[0],
        end_line=current_extent[1],
    )
    donor_source = _source_lines(
        Path(getattr(spec, "primary_donor_root")),
        relative_path=donor_file,
        start_line=donor_extent[0],
        end_line=donor_extent[1],
    )
    if (
        current_source.get("status") != "resolved"
        or donor_source.get("status") != "resolved"
    ):
        return {
            "status": "not_proven",
            "current_source_status": current_source.get("status"),
            "donor_source_status": donor_source.get("status"),
        }
    current_lines = [
        str(item.get("text") or "")
        for item in current_source.get("lines") or []
        if isinstance(item, dict)
    ]
    donor_lines = [
        str(item.get("text") or "")
        for item in donor_source.get("lines") or []
        if isinstance(item, dict)
    ]
    return {
        "status": (
            "equivalent"
            if current_lines == donor_lines
            else "different"
        ),
        "current_file": current_file,
        "current_extent": {
            "start_line": current_extent[0],
            "end_line": current_extent[1],
        },
        "donor_file": donor_file,
        "donor_extent": {
            "start_line": donor_extent[0],
            "end_line": donor_extent[1],
        },
    }


def _exact_export_obligation(
    gaps: Iterable[dict[str, Any]],
    *,
    owner_descriptor: dict[str, Any],
) -> dict[str, Any] | None:
    """Return one unambiguous export-only obligation for this provider."""

    demands: list[dict[str, Any]] = []
    exports: dict[
        tuple[str, int, int, str, str, str],
        dict[str, Any],
    ] = {}
    for gap in gaps:
        evidence = (
            gap.get("contract_evidence")
            if isinstance(gap.get("contract_evidence"), dict)
            else {}
        )
        evaluation = (
            gap.get("compile_valid_evaluation")
            if isinstance(
                gap.get("compile_valid_evaluation"),
                dict,
            )
            else {}
        )
        donor_entity = (
            evidence.get("donor_entity")
            if isinstance(evidence.get("donor_entity"), dict)
            else {}
        )
        unresolved_dimensions = {
            str(item)
            for item in evaluation.get("unresolved_dimensions") or []
            if str(item)
        }
        if (
            not _same_entity_contract(owner_descriptor, donor_entity)
            or evaluation.get("status") != "unresolved"
            or unresolved_dimensions != {"export"}
            or (evaluation.get("relations") or {}).get("export")
            != "missing_in_old"
        ):
            continue
        symbol = str(
            evidence.get("symbol")
            or donor_entity.get("symbol")
            or donor_entity.get("name")
            or ""
        )
        donor_file = _descriptor_identity(donor_entity)[0]
        donor_exports = [
            dict(record)
            for record in evidence.get("donor_exports") or []
            if isinstance(record, dict)
            and str(record.get("symbol") or "") == symbol
            and normalize_relpath(str(record.get("file") or ""))
            == donor_file
            and int(record.get("line") or 0) > 0
            and int(record.get("end_line") or record.get("line") or 0)
            == int(record.get("line") or 0)
            and str(record.get("macro") or "")
            and str(record.get("raw") or "").strip()
        ]
        if len(donor_exports) != 1:
            continue
        donor_export = donor_exports[0]
        identity = (
            donor_file,
            int(donor_export["line"]),
            int(donor_export.get("end_line") or donor_export["line"]),
            str(donor_export.get("macro") or ""),
            str(donor_export.get("namespace_or_modules") or ""),
            str(donor_export.get("raw") or ""),
        )
        exports[identity] = donor_export
        demands.append(
            {
                "symbol": symbol,
                "gap_id": str(gap.get("id") or ""),
                "contract_identity": str(
                    gap.get("contract_identity") or ""
                ),
                "contract_evidence_ref": str(
                    gap.get("contract_evidence_ref") or ""
                ),
                "authority": (
                    "compile_valid_exact_export_relation"
                ),
            }
        )
    if not demands or len(exports) != 1:
        return None
    donor_export = next(iter(exports.values()))
    export_range = {
        "tree": "donor",
        "path": normalize_relpath(str(donor_export.get("file") or "")),
        "line_start": int(donor_export.get("line") or 0),
        "line_end": int(
            donor_export.get("end_line")
            or donor_export.get("line")
            or 0
        ),
        "source": "donor_external_provider_contract_reference",
        "entity_id": str(donor_export.get("id") or ""),
        "contract_key": str(donor_export.get("contract_key") or ""),
    }
    return {
        "donor_export": donor_export,
        "module_visibility_demands": demands,
        "inspectable_ranges": [export_range],
        "gap_ids": _ordered_strings(
            demand.get("gap_id") for demand in demands
        ),
    }


class SemanticClosureEvidenceTools:
    """Exact, bounded source and dependency tools for semantic closure."""

    def __init__(self, spec: object, *, state_root: Path) -> None:
        self.spec = spec
        self.state_root = state_root.resolve()
        self._analysis: dict[str, Any] | None = None
        self._hunks_by_id: dict[str, dict[str, Any]] | None = None
        self._catalog_resolver_index: CodeIndex | None = None

    def _state(self) -> dict[str, Any]:
        state = load_semantic_closure_state(
            self.state_root,
            case_id=str(getattr(self.spec, "case_id")),
        )
        if state is None:
            raise ValueError("semantic closure state is not initialized")
        return state

    def _initial_analysis(self) -> dict[str, Any]:
        if self._analysis is None:
            self._analysis = load_initial_analysis_artifact(
                self.state_root,
                self._state(),
            )
        return self._analysis

    def _raw_hunks(self) -> dict[str, dict[str, Any]]:
        if self._hunks_by_id is None:
            state = self._state()
            hunk_refs = state.get("raw_hunk_refs")
            if isinstance(hunk_refs, dict):
                self._hunks_by_id = {}
                for hunk_id, reference in hunk_refs.items():
                    record = load_semantic_record(
                        self.state_root,
                        str(reference),
                    )
                    if (
                        not isinstance(record, dict)
                        or str(record.get("id") or "") != str(hunk_id)
                    ):
                        raise ValueError(
                            "semantic raw hunk reference has an identity "
                            f"mismatch: {hunk_id}"
                        )
                    self._hunks_by_id[str(hunk_id)] = dict(record)
            else:
                self._hunks_by_id = {
                    str(item.get("id") or ""): dict(item)
                    for item in self._initial_analysis().get("raw_hunks")
                    or []
                    if isinstance(item, dict)
                    and str(item.get("id") or "")
                }
        return self._hunks_by_id

    def _resolve_descriptor(
        self,
        descriptor: dict[str, Any],
        *,
        tree: str,
        report_cache: dict[tuple[str, str], dict[str, Any]],
    ) -> dict[str, Any]:
        relative_path = normalize_relpath(
            str(
                descriptor.get("file")
                or descriptor.get("path")
                or ""
            )
        )
        if not relative_path or not descriptor:
            return {
                "status": "descriptor_unavailable",
                "tree": tree,
                "file": relative_path,
                "entity": None,
            }
        return resolve_entity_in_case_tree(
            self.spec,
            descriptor=descriptor,
            relative_path=relative_path,
            tree=tree,
            index=(
                self._catalog_resolver_index
                or CodeIndex(self.spec, graph_projection="records")
            ),
            require_definition=(
                descriptor.get("is_definition") is not False
            ),
            report_cache=report_cache,
        )

    def build_catalog(
        self,
        context: dict[str, Any],
        *,
        include_write_actions: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build prompt evidence and an internal exact-authorization catalog."""

        work_unit = (
            context.get("work_unit")
            if isinstance(context.get("work_unit"), dict)
            else {}
        )
        gaps = [
            dict(item)
            for item in context.get("gaps") or []
            if isinstance(item, dict)
        ]
        bindings: dict[str, dict[str, Any]] = {}

        def add_binding(
            raw: object,
            *,
            semantic_gap_id: str | None = None,
        ) -> None:
            if not isinstance(raw, dict):
                return
            owner_key = str(raw.get("owner_key") or "")
            if not owner_key:
                return
            record = bindings.setdefault(owner_key, deepcopy(raw))
            record["semantic_gap_ids"] = _ordered_strings(
                [
                    *list(record.get("semantic_gap_ids") or []),
                    *list(raw.get("semantic_gap_ids") or []),
                    *([semantic_gap_id] if semantic_gap_id else []),
                ]
            )

        for raw in work_unit.get("owner_bindings") or []:
            add_binding(raw)
        for gap in gaps:
            gap_id = str(gap.get("id") or "")
            for raw in gap.get("owner_bindings") or []:
                add_binding(raw, semantic_gap_id=gap_id or None)

        open_dimensions_by_gap = {
            str(gap.get("id") or ""): semantic_open_mutation_dimensions(gap)
            for gap in gaps
            if str(gap.get("id") or "")
        }
        mutation_targets_by_owner: dict[
            str,
            dict[str, list[str]],
        ] = {}
        for owner_key, binding in bindings.items():
            targets: dict[str, list[str]] = {}
            for gap_id in _ordered_strings(
                binding.get("semantic_gap_ids") or []
            ):
                dimensions = semantic_owner_mutation_dimensions(
                    binding,
                    open_dimensions_by_gap.get(gap_id, []),
                )
                if dimensions:
                    targets[gap_id] = dimensions
            mutation_targets_by_owner[owner_key] = targets

        requested_hunk_ids = _ordered_strings(
            hunk_id
            for gap in gaps
            for hunk_id in gap.get("raw_hunk_ids") or []
        )
        gap_ids_by_hunk: dict[str, list[str]] = {}
        for gap in gaps:
            gap_id = str(gap.get("id") or "")
            for hunk_id in gap.get("raw_hunk_ids") or []:
                normalized_hunk_id = str(hunk_id or "")
                if normalized_hunk_id and gap_id:
                    gap_ids_by_hunk[normalized_hunk_id] = (
                        _ordered_strings(
                            [
                                *gap_ids_by_hunk.get(
                                    normalized_hunk_id,
                                    [],
                                ),
                                gap_id,
                            ]
                        )
                    )
        all_hunks = self._raw_hunks() if gaps else {}
        missing_hunks = [
            hunk_id
            for hunk_id in requested_hunk_ids
            if hunk_id not in all_hunks
        ]
        if missing_hunks:
            raise ValueError(
                "semantic gaps reference missing immutable hunks: "
                + ", ".join(missing_hunks)
            )

        report_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._catalog_resolver_index = CodeIndex(
            self.spec,
            graph_projection="records",
        )
        integration_relations: dict[str, dict[str, Any]] = {}
        for gap in gaps:
            gap_id = str(gap.get("id") or "")
            evaluation = (
                gap.get("compile_valid_evaluation")
                if isinstance(gap.get("compile_valid_evaluation"), dict)
                else {}
            )
            integration = (
                (evaluation.get("observations") or {}).get(
                    "integration_reference"
                )
                if isinstance(evaluation.get("observations"), dict)
                else {}
            )
            if not gap_id or not isinstance(integration, dict):
                continue
            relation = str(
                (evaluation.get("relations") or {}).get(
                    "integration_reference"
                )
                or "unavailable"
            )
            integration_relations[gap_id] = {
                "relation": relation,
                "observation": integration,
                "missing_shapes": {
                    tuple(str(value) for value in item.get("shape") or [])
                    for item in integration.get("missing_reference_shapes")
                    or []
                    if isinstance(item, dict)
                },
            }
        old_root = Path(getattr(self.spec, "old_root_path"))
        donor_root = Path(getattr(self.spec, "primary_donor_root"))
        entity_catalog: dict[str, dict[str, Any]] = {}
        ranges: dict[tuple[str, str], list[dict[str, Any]]] = {}
        owner_records: list[dict[str, Any]] = []
        materializable: list[dict[str, Any]] = []
        missing_entity_anchor_candidates: list[dict[str, Any]] = []
        structural_report_cache: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        for owner_key, binding in bindings.items():
            demand_reference_relation: dict[str, Any] = {}
            if str(binding.get("role") or "") == "demand_owner":
                demand = (
                    binding.get("demand_reference")
                    if isinstance(binding.get("demand_reference"), dict)
                    else {}
                )
                relevant_gap_ids = _ordered_strings(
                    binding.get("semantic_gap_ids") or []
                )
                evidence_by_gap = {
                    str(gap.get("id") or ""): (
                        gap.get("contract_evidence")
                        if isinstance(gap.get("contract_evidence"), dict)
                        else {}
                    )
                    for gap in gaps
                }
                representation_agnostic = any(
                    integration_reference_target_representation_agnostic(
                        evidence_by_gap.get(gap_id, {})
                    )
                    for gap_id in relevant_gap_ids
                )
                demand_shape = tuple(
                    str(value)
                    for value in reference_shape(
                        {
                            **demand,
                            "owner_entity": binding.get(
                                "donor_descriptor"
                            )
                            or binding.get("current_descriptor")
                            or {},
                        },
                        target_representation_agnostic=(
                            representation_agnostic
                        ),
                    )
                )
                unresolved_for_owner = [
                    gap_id
                    for gap_id in relevant_gap_ids
                    if demand_shape
                    in (
                        integration_relations.get(gap_id) or {}
                    ).get("missing_shapes", set())
                ]
                evaluated_gap_ids = [
                    gap_id
                    for gap_id in relevant_gap_ids
                    if gap_id in integration_relations
                ]
                relation_unavailable = any(
                    (integration_relations.get(gap_id) or {}).get(
                        "relation"
                    )
                    == "unavailable"
                    for gap_id in evaluated_gap_ids
                )
                exact_shapes_available = all(
                    bool(
                        (
                            integration_relations.get(gap_id) or {}
                        ).get("observation", {}).get(
                            "current_reference_shapes"
                        )
                        or (
                            integration_relations.get(gap_id) or {}
                        ).get("observation", {}).get(
                            "missing_reference_shapes"
                        )
                    )
                    for gap_id in evaluated_gap_ids
                )
                relation_unavailable = (
                    relation_unavailable
                    or not evaluated_gap_ids
                    or not exact_shapes_available
                )
                demand_reference_relation = {
                    "status": (
                        "missing_in_old"
                        if unresolved_for_owner
                        else "unavailable"
                        if relation_unavailable
                        else "already_satisfied"
                        if evaluated_gap_ids
                        else "not_evaluated"
                    ),
                    "shape": list(demand_shape),
                    "evaluated_gap_ids": evaluated_gap_ids,
                    "unresolved_gap_ids": unresolved_for_owner,
                    "authority": "typed_current_donor_reference_shape",
                }
                if (
                    include_write_actions
                    and evaluated_gap_ids
                    and not relation_unavailable
                ):
                    refined_targets: dict[str, list[str]] = {}
                    for gap_id, dimensions in (
                        mutation_targets_by_owner.get(owner_key, {}).items()
                    ):
                        remaining = [
                            dimension
                            for dimension in dimensions
                            if dimension != "integration_reference"
                            or gap_id in unresolved_for_owner
                        ]
                        if remaining:
                            refined_targets[gap_id] = remaining
                    mutation_targets_by_owner[owner_key] = refined_targets
            current_descriptor = _compact_descriptor(
                binding.get("current_descriptor")
            )
            donor_descriptor = _compact_descriptor(
                binding.get("donor_descriptor")
            )
            current_resolution = self._resolve_descriptor(
                current_descriptor or donor_descriptor,
                tree="old",
                report_cache=report_cache,
            )
            donor_resolution = self._resolve_descriptor(
                donor_descriptor or current_descriptor,
                tree="donor",
                report_cache=report_cache,
            )
            owner_records.append(
                {
                    "owner_key": owner_key,
                    "role": binding.get("role"),
                    "file": binding.get("file"),
                    "contract_key": binding.get("contract_key"),
                    "symbol": binding.get("symbol"),
                    "entity_kind": binding.get("entity_kind"),
                    "target_presence": binding.get("target_presence"),
                    "exact_node_refs": list(
                        binding.get("exact_node_refs") or []
                    ),
                    "semantic_gap_ids": list(
                        binding.get("semantic_gap_ids") or []
                    ),
                    "current_resolution": _compact_resolution(
                        current_resolution
                    ),
                    "donor_resolution": _compact_resolution(
                        donor_resolution
                    ),
                    "demand_reference": deepcopy(
                        binding.get("demand_reference") or {}
                    ),
                    "demand_reference_relation": deepcopy(
                        demand_reference_relation
                    ),
                    "mutation_targets": [
                        {
                            "gap_id": gap_id,
                            "dimensions": dimensions,
                        }
                        for gap_id, dimensions in sorted(
                            mutation_targets_by_owner.get(
                                owner_key,
                                {},
                            ).items()
                        )
                    ],
                }
            )
            for tree, descriptor, resolution in (
                ("old", current_descriptor, current_resolution),
                ("donor", donor_descriptor, donor_resolution),
            ):
                if (
                    include_write_actions
                    and demand_reference_relation.get("status")
                    == "already_satisfied"
                ):
                    continue
                resolved = (
                    resolution.get("entity")
                    if resolution.get("status") == "resolved"
                    and isinstance(resolution.get("entity"), dict)
                    else {}
                )
                selected = _compact_descriptor(resolved)
                if not selected:
                    continue
                extent = _descriptor_extent(selected)
                relative_path = normalize_relpath(
                    str(
                        selected.get("file")
                        or selected.get("path")
                        or ""
                    )
                )
                catalog_id = stable_id(
                    "semantic-tool-entity",
                    tree,
                    str(
                        selected.get("contract_key")
                        or selected.get("key")
                        or ""
                    ),
                    normalize_relpath(
                        str(
                            selected.get("file")
                            or selected.get("path")
                            or ""
                        )
                    ),
                    str(extent or ""),
                )
                record = entity_catalog.setdefault(
                    catalog_id,
                    {
                        "id": catalog_id,
                        "owner_keys": [],
                        "tree": tree,
                        "descriptor": selected,
                    },
                )
                if owner_key not in record["owner_keys"]:
                    record["owner_keys"].append(owner_key)
                if extent and relative_path:
                    ranges.setdefault(
                        (tree, relative_path),
                        [],
                    ).append(
                        {
                            "start_line": extent[0],
                            "end_line": extent[1],
                            "authority": "exact_owner_entity",
                            "entity_id": catalog_id,
                            "owner_key": owner_key,
                            "gap_ids": list(
                                binding.get("semantic_gap_ids") or []
                            ),
                        }
                    )
            if (
                current_resolution.get("status") == "entity_absent"
                and donor_resolution.get("status") == "resolved"
            ):
                donor_entries = [
                    item
                    for item in entity_catalog.values()
                    if owner_key in item.get("owner_keys", [])
                    and item["tree"] == "donor"
                ]
                if len(donor_entries) == 1:
                    authorized_gap_ids = sorted(
                        mutation_targets_by_owner.get(owner_key, {})
                    )
                    candidate = {
                        "entity_id": donor_entries[0]["id"],
                        "owner_key": owner_key,
                        "descriptor": deepcopy(
                            donor_entries[0]["descriptor"]
                        ),
                        "authority": (
                            "exact_donor_entity_with_proven_target_absence"
                        ),
                        "mode": "missing_entity",
                        "module_visibility_demands": [],
                        "inspectable_ranges": [],
                        "authorized_gap_ids": authorized_gap_ids,
                    }
                    if authorized_gap_ids:
                        materializable.append(candidate)
                        missing_entity_anchor_candidates.append(
                            {
                                **deepcopy(candidate),
                                "gap_ids": authorized_gap_ids,
                            }
                        )
            elif (
                current_resolution.get("status") == "resolved"
                and donor_resolution.get("status") == "resolved"
            ):
                donor_entries = [
                    item
                    for item in entity_catalog.values()
                    if owner_key in item.get("owner_keys", [])
                    and item["tree"] == "donor"
                ]
                donor_entity = (
                    donor_resolution.get("entity")
                    if isinstance(
                        donor_resolution.get("entity"),
                        dict,
                    )
                    else donor_descriptor
                )
                export_obligation = _exact_export_obligation(
                    gaps,
                    owner_descriptor=donor_entity,
                )
                source_proof = _exact_definition_source_proof(
                    self.spec,
                    current_resolution=current_resolution,
                    donor_resolution=donor_resolution,
                )
                if (
                    len(donor_entries) == 1
                    and export_obligation is not None
                    and source_proof.get("status") == "equivalent"
                ):
                    candidate = {
                            "entity_id": donor_entries[0]["id"],
                            "owner_key": owner_key,
                            "descriptor": deepcopy(
                                donor_entries[0]["descriptor"]
                            ),
                            "authority": (
                                "exact_donor_export_with_equivalent_provider"
                            ),
                            "mode": "existing_equivalent_provider_export",
                            "source_equivalence": source_proof,
                            "module_visibility_demands": deepcopy(
                                export_obligation[
                                    "module_visibility_demands"
                                ]
                            ),
                            "inspectable_ranges": deepcopy(
                                export_obligation[
                                    "inspectable_ranges"
                                ]
                            ),
                            "authorized_gap_ids": list(
                                export_obligation["gap_ids"]
                            ),
                        }
                    mutation_gap_ids = set(
                        mutation_targets_by_owner.get(owner_key, {})
                    )
                    candidate["authorized_gap_ids"] = sorted(
                        mutation_gap_ids.intersection(
                            candidate["authorized_gap_ids"]
                        )
                    )
                    if candidate["authorized_gap_ids"]:
                        materializable.append(candidate)

        insertion_anchor_specs: list[dict[str, Any]] = []
        unavailable_insertion_anchors: list[dict[str, Any]] = []
        for candidate in (
            missing_entity_anchor_candidates
            if include_write_actions
            else []
        ):
            anchor = _structural_insertion_anchor(
                old_root=old_root,
                donor_root=donor_root,
                descriptor=dict(candidate.get("descriptor") or {}),
                owner_key=str(candidate.get("owner_key") or ""),
                gap_ids=candidate.get("gap_ids") or [],
                report_cache=structural_report_cache,
            )
            if anchor.get("status") == "resolved":
                insertion_anchor_specs.append(anchor)
            else:
                unavailable_insertion_anchors.append(anchor)

        executable_materializations: list[dict[str, Any]] = []
        unavailable_materializations: list[dict[str, Any]] = []
        for candidate in materializable if include_write_actions else []:
            preflight_gap_ids = _ordered_strings(
                candidate.get("authorized_gap_ids") or []
            )
            try:
                preflight = self._materialize_selected(
                    context,
                    selected=candidate,
                    entity_id=str(candidate["entity_id"]),
                    gap_ids=preflight_gap_ids,
                    rationale="static structural preflight",
                )
            except (OSError, ValueError) as exc:
                unavailable_materializations.append(
                    {
                        "entity_id": candidate.get("entity_id"),
                        "owner_key": candidate.get("owner_key"),
                        "mode": candidate.get("mode"),
                        "status": "preflight_failed",
                        "reason": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                continue
            handle_id = stable_id(
                "semantic-materialization-handle",
                str(candidate.get("entity_id") or ""),
                str(candidate.get("owner_key") or ""),
                hashlib.sha256(
                    str(preflight.get("diff_text") or "").encode("utf-8")
                ).hexdigest(),
            )
            executable_materializations.append(
                {
                    **candidate,
                    "handle_id": handle_id,
                    "preflight_result": preflight,
                }
            )
        materializable = executable_materializations

        for item in materializable:
            for source_range in item.get("inspectable_ranges") or []:
                if not isinstance(source_range, dict):
                    continue
                tree = str(source_range.get("tree") or "")
                relative_path = normalize_relpath(
                    str(
                        source_range.get("path")
                        or source_range.get("file")
                        or ""
                    )
                )
                start_line = int(
                    source_range.get("line_start")
                    or source_range.get("start_line")
                    or 0
                )
                end_line = int(
                    source_range.get("line_end")
                    or source_range.get("end_line")
                    or start_line
                )
                if (
                    tree not in {"old", "donor"}
                    or not relative_path
                    or start_line <= 0
                    or end_line < start_line
                ):
                    continue
                ranges.setdefault((tree, relative_path), []).append(
                    {
                        "start_line": start_line,
                        "end_line": end_line,
                        "authority": (
                            "exact_donor_export_contract"
                        ),
                        "entity_id": source_range.get("entity_id"),
                        "contract_key": source_range.get(
                            "contract_key"
                        ),
                    }
                )

        hunk_catalog: dict[str, dict[str, Any]] = {}
        hunk_records: list[dict[str, Any]] = []
        for hunk_id in requested_hunk_ids:
            hunk = dict(all_hunks[hunk_id])
            relative_path = normalize_relpath(
                str(hunk.get("file") or "")
            )
            record = {
                "id": hunk_id,
                "file": relative_path,
                "operation": hunk.get("operation"),
                "old_start_line": hunk.get("old_start_line"),
                "old_end_line": hunk.get("old_end_line"),
                "donor_start_line": hunk.get("donor_start_line"),
                "donor_end_line": hunk.get("donor_end_line"),
                "old_line_count": len(hunk.get("old_lines") or []),
                "donor_line_count": len(
                    hunk.get("donor_lines") or []
                ),
            }
            hunk_catalog[hunk_id] = hunk
            hunk_records.append(record)
            old_start = int(hunk.get("old_start_line") or 0)
            old_end = int(hunk.get("old_end_line") or old_start)
            if old_start > 0 and old_end >= old_start:
                ranges.setdefault(("old", relative_path), []).append(
                    {
                        "start_line": old_start,
                        "end_line": old_end,
                        "authority": "immutable_initial_hunk",
                        "hunk_id": hunk_id,
                        "gap_ids": list(
                            gap_ids_by_hunk.get(hunk_id, [])
                        ),
                    }
                )
            donor_start = int(hunk.get("donor_start_line") or 0)
            donor_end = int(
                hunk.get("donor_end_line") or donor_start
            )
            if donor_start > 0 and donor_end >= donor_start:
                ranges.setdefault(("donor", relative_path), []).append(
                    {
                        "start_line": donor_start,
                        "end_line": donor_end,
                        "authority": "immutable_initial_hunk",
                        "hunk_id": hunk_id,
                        "gap_ids": list(
                            gap_ids_by_hunk.get(hunk_id, [])
                        ),
                    }
                )

        ranges = _deduplicate_ranges(ranges)
        read_handles: dict[str, dict[str, Any]] = {}
        source_span_handles: dict[tuple[str, str, int, int], str] = {}
        for hunk_id, hunk in hunk_catalog.items():
            handle_id = stable_id(
                "semantic-read-hunk",
                hunk_id,
            )
            read_handles[handle_id] = {
                "id": handle_id,
                "kind": "hunk",
                "hunk_id": hunk_id,
                "record": deepcopy(hunk),
            }
        for entity_id, entity in entity_catalog.items():
            descriptor = dict(entity.get("descriptor") or {})
            extent = _descriptor_extent(descriptor)
            relative_path = normalize_relpath(
                str(
                    descriptor.get("file")
                    or descriptor.get("path")
                    or ""
                )
            )
            if not extent or not relative_path:
                continue
            handle_id = stable_id(
                "semantic-read-entity",
                entity_id,
            )
            tree = str(entity.get("tree") or "")
            span = (tree, relative_path, extent[0], extent[1])
            existing_handle_id = source_span_handles.get(span)
            if existing_handle_id is None:
                read_handles[handle_id] = {
                    "id": handle_id,
                    "kind": "entity",
                    "tree": tree,
                    "file": relative_path,
                    "start_line": extent[0],
                    "end_line": extent[1],
                    "entity_id": entity_id,
                    "entity_ids": [entity_id],
                    "authorities": ["exact_owner_entity"],
                    "owner_keys": list(entity.get("owner_keys") or []),
                    "source_owner_keys": _ordered_strings(
                        source_owner_key
                        for owner_key in entity.get("owner_keys") or []
                        for source_owner_key in (
                            bindings.get(str(owner_key), {}).get(
                                "source_owner_keys"
                            )
                            or []
                        )
                    ),
                    "gap_ids": _ordered_strings(
                        gap_id
                        for owner_key in entity.get("owner_keys") or []
                        for gap_id in (
                            bindings.get(str(owner_key), {}).get(
                                "semantic_gap_ids"
                            )
                            or []
                        )
                    ),
                    "descriptor": descriptor,
                }
                source_span_handles[span] = handle_id
                continue
            existing = read_handles[existing_handle_id]
            _merge_unique(existing, "entity_ids", [entity_id])
            _merge_unique(
                existing,
                "owner_keys",
                entity.get("owner_keys") or [],
            )
            _merge_unique(
                existing,
                "source_owner_keys",
                (
                    source_owner_key
                    for owner_key in entity.get("owner_keys") or []
                    for source_owner_key in (
                        bindings.get(str(owner_key), {}).get(
                            "source_owner_keys"
                        )
                        or []
                    )
                ),
            )
            _merge_unique(
                existing,
                "gap_ids",
                [
                    gap_id
                    for owner_key in entity.get("owner_keys") or []
                    for gap_id in (
                        bindings.get(str(owner_key), {}).get(
                            "semantic_gap_ids"
                        )
                        or []
                    )
                ],
            )
        for (tree, relative_path), values in ranges.items():
            for item in values:
                span = (
                    tree,
                    relative_path,
                    int(item["start_line"]),
                    int(item["end_line"]),
                )
                handle_id = source_span_handles.get(span)
                if handle_id is None:
                    handle_id = stable_id(
                        "semantic-read-range",
                        tree,
                        relative_path,
                        str(item["start_line"]),
                        str(item["end_line"]),
                    )
                    read_handles[handle_id] = {
                        "id": handle_id,
                        "kind": "range",
                        "tree": tree,
                        "file": relative_path,
                        "start_line": int(item["start_line"]),
                        "end_line": int(item["end_line"]),
                        "authorities": [],
                        "owner_keys": [],
                        "gap_ids": [],
                    }
                    source_span_handles[span] = handle_id
                record = read_handles[handle_id]
                _merge_unique(
                    record,
                    "authorities",
                    [item.get("authority")],
                )
                _merge_unique(
                    record,
                    "owner_keys",
                    item.get("owner_keys") or [],
                )
                _merge_unique(
                    record,
                    "gap_ids",
                    item.get("gap_ids") or [],
                )

        edit_handles: dict[str, dict[str, Any]] = {}
        for item in read_handles.values() if include_write_actions else []:
            if item.get("kind") == "hunk" or item.get("tree") != "old":
                continue
            relative_path = normalize_relpath(
                str(item.get("file") or "")
            )
            if (
                not relative_path
                or Path(relative_path).name
                in {"Kconfig", "Makefile", "Kbuild"}
                or str(
                    getattr(self.spec, "classify_edit_root")(
                        relative_path
                    )
                )
                not in {"editable", "escalation"}
            ):
                continue
            target_dimensions_by_gap: dict[str, list[str]] = {}
            for owner_key in item.get("owner_keys") or []:
                for gap_id, dimensions in mutation_targets_by_owner.get(
                    str(owner_key),
                    {},
                ).items():
                    target_dimensions_by_gap[gap_id] = _ordered_strings(
                        [
                            *target_dimensions_by_gap.get(gap_id, []),
                            *dimensions,
                        ]
                    )
            if not target_dimensions_by_gap:
                continue
            authorized_gap_ids = sorted(target_dimensions_by_gap)
            repair_dimensions = _ordered_strings(
                dimension
                for dimensions in target_dimensions_by_gap.values()
                for dimension in dimensions
            )
            source = _source_lines(
                old_root,
                relative_path=relative_path,
                start_line=int(item["start_line"]),
                end_line=int(item["end_line"]),
            )
            if source.get("status") != "resolved":
                continue
            handle_id = stable_id(
                "semantic-edit-region",
                relative_path,
                str(item["start_line"]),
                str(item["end_line"]),
                str(source["source_sha256"]),
            )
            record = edit_handles.setdefault(
                handle_id,
                {
                    "id": handle_id,
                    "file": relative_path,
                    "start_line": int(item["start_line"]),
                    "end_line": int(item["end_line"]),
                    "source_sha256": source["source_sha256"],
                    "source": source["source"],
                    "authorities": [],
                    "owner_keys": [],
                    "source_owner_keys": [],
                    "gap_ids": authorized_gap_ids,
                    "repair_dimensions": repair_dimensions,
                    "read_handle_ids": [],
                },
            )
            _merge_unique(
                record,
                "authorities",
                [
                    *list(item.get("authorities") or []),
                    item.get("authority"),
                ],
            )
            for owner_key in item.get("owner_keys") or []:
                if owner_key not in record["owner_keys"]:
                    record["owner_keys"].append(owner_key)
            _merge_unique(
                record,
                "source_owner_keys",
                item.get("source_owner_keys") or [],
            )
            record["gap_ids"] = _ordered_strings(
                [
                    *record["gap_ids"],
                    *authorized_gap_ids,
                ]
            )
            _merge_unique(
                record,
                "repair_dimensions",
                repair_dimensions,
            )
            if item["id"] not in record["read_handle_ids"]:
                record["read_handle_ids"].append(item["id"])

        edit_handles = _drop_dominated_edit_handles(edit_handles)
        for anchor in insertion_anchor_specs:
            relative_path = normalize_relpath(
                str(anchor.get("file") or "")
            )
            if (
                not relative_path
                or Path(relative_path).name
                in {"Kconfig", "Makefile", "Kbuild"}
                or str(
                    getattr(self.spec, "classify_edit_root")(
                        relative_path
                    )
                )
                not in {"editable", "escalation"}
            ):
                unavailable_insertion_anchors.append(
                    {
                        **deepcopy(anchor),
                        "status": "unavailable",
                        "reason": "structural_anchor_is_outside_write_scope",
                    }
                )
                continue
            source = _source_lines(
                old_root,
                relative_path=relative_path,
                start_line=int(anchor["start_line"]),
                end_line=int(anchor["end_line"]),
            )
            if source.get("status") != "resolved":
                unavailable_insertion_anchors.append(
                    {
                        **deepcopy(anchor),
                        "status": "unavailable",
                        "reason": "structural_anchor_source_is_unavailable",
                    }
                )
                continue
            edit_mode = str(anchor.get("edit_mode") or "")
            anchor_owner_key = str(anchor.get("owner_key") or "")
            target_dimensions_by_gap = {
                gap_id: dimensions
                for gap_id, dimensions in mutation_targets_by_owner.get(
                    anchor_owner_key,
                    {},
                ).items()
                if gap_id in set(
                    _ordered_strings(anchor.get("gap_ids") or [])
                )
            }
            repair_dimensions = _ordered_strings(
                dimension
                for dimensions in target_dimensions_by_gap.values()
                for dimension in dimensions
            )
            handle_id = stable_id(
                "semantic-structural-insertion-region",
                relative_path,
                str(anchor["start_line"]),
                str(anchor["end_line"]),
                edit_mode,
                str(source["source_sha256"]),
            )
            record = edit_handles.setdefault(
                handle_id,
                {
                    "id": handle_id,
                    "file": relative_path,
                    "start_line": int(anchor["start_line"]),
                    "end_line": int(anchor["end_line"]),
                    "source_sha256": source["source_sha256"],
                    "source": source["source"],
                    "edit_mode": edit_mode,
                    "authorities": [],
                    "owner_keys": [],
                    "gap_ids": [],
                    "repair_dimensions": repair_dimensions,
                    "read_handle_ids": [],
                    "insertion_targets": [],
                },
            )
            _merge_unique(
                record,
                "authorities",
                [anchor.get("authority")],
            )
            _merge_unique(
                record,
                "owner_keys",
                [anchor.get("owner_key")],
            )
            _merge_unique(
                record,
                "gap_ids",
                anchor.get("gap_ids") or [],
            )
            _merge_unique(
                record,
                "repair_dimensions",
                repair_dimensions,
            )
            target = {
                "owner_key": str(anchor.get("owner_key") or ""),
                "gap_ids": _ordered_strings(
                    anchor.get("gap_ids") or []
                ),
                "descriptor": deepcopy(
                    anchor.get("target_descriptor") or {}
                ),
                "anchor_descriptor": deepcopy(
                    anchor.get("anchor_descriptor") or {}
                ),
                "alignment": deepcopy(anchor.get("alignment") or {}),
            }
            target_identity = (
                target["owner_key"],
                _descriptor_contract_identity(target["descriptor"]),
            )
            if target_identity not in {
                (
                    str(item.get("owner_key") or ""),
                    _descriptor_contract_identity(
                        dict(item.get("descriptor") or {})
                    ),
                )
                for item in record["insertion_targets"]
                if isinstance(item, dict)
            }:
                record["insertion_targets"].append(target)

        source_records: dict[str, dict[str, Any]] = {}
        requested_gap_id_set = {
            str(gap.get("id") or "")
            for gap in gaps
            if str(gap.get("id") or "")
        }

        def add_source_record(
            *,
            tree: str,
            relative_path: str,
            start_line: int,
            end_line: int,
            source: dict[str, Any],
            read_handle_ids: Iterable[object] = (),
            edit_handle_ids: Iterable[object] = (),
            owner_keys: Iterable[object] = (),
            gap_ids: Iterable[object] = (),
        ) -> str | None:
            if source.get("status") != "resolved":
                return None
            record_id = stable_id(
                "semantic-source-record",
                tree,
                relative_path,
                str(start_line),
                str(end_line),
                str(source.get("source_sha256") or ""),
            )
            record = source_records.setdefault(
                record_id,
                {
                    "id": record_id,
                    "tree": tree,
                    "file": relative_path,
                    "start_line": int(start_line),
                    "end_line": int(end_line),
                    "source_sha256": str(
                        source.get("source_sha256") or ""
                    ),
                    "source": str(source.get("source") or ""),
                    "read_handle_ids": [],
                    "edit_handle_ids": [],
                    "owner_keys": [],
                    "gap_ids": [],
                },
            )
            for key, values in (
                ("read_handle_ids", read_handle_ids),
                ("edit_handle_ids", edit_handle_ids),
                ("owner_keys", owner_keys),
                ("gap_ids", gap_ids),
            ):
                _merge_unique(record, key, values)
            return record_id

        for handle_id, item in edit_handles.items():
            source_record_id = add_source_record(
                tree="old",
                relative_path=str(item["file"]),
                start_line=int(item["start_line"]),
                end_line=int(item["end_line"]),
                source={
                    "status": "resolved",
                    "source_sha256": item.get("source_sha256"),
                    "source": item.get("source"),
                },
                read_handle_ids=item.get("read_handle_ids") or [],
                edit_handle_ids=[handle_id],
                owner_keys=item.get("owner_keys") or [],
                gap_ids=item.get("gap_ids") or [],
            )
            if source_record_id:
                item["source_record_id"] = source_record_id
                for read_handle_id in item.get("read_handle_ids") or []:
                    read_handle = read_handles.get(str(read_handle_id))
                    if isinstance(read_handle, dict):
                        read_handle["source_record_id"] = source_record_id

        tree_roots = {
            "old": old_root,
            "donor": Path(getattr(self.spec, "primary_donor_root")),
        }
        for handle_id, item in read_handles.items():
            if item.get("kind") != "entity":
                continue
            direct_gap_ids = set(
                _ordered_strings(item.get("gap_ids") or [])
            )
            if not direct_gap_ids.intersection(requested_gap_id_set):
                continue
            tree = str(item.get("tree") or "")
            root = tree_roots.get(tree)
            if root is None:
                continue
            source_record_id = add_source_record(
                tree=tree,
                relative_path=str(item["file"]),
                start_line=int(item["start_line"]),
                end_line=int(item["end_line"]),
                source=_source_lines(
                    root,
                    relative_path=str(item["file"]),
                    start_line=int(item["start_line"]),
                    end_line=int(item["end_line"]),
                ),
                read_handle_ids=[handle_id],
                owner_keys=item.get("owner_keys") or [],
                gap_ids=item.get("gap_ids") or [],
            )
            if source_record_id:
                item["source_record_id"] = source_record_id

        for item in edit_handles.values():
            if not str(item.get("edit_mode") or "").startswith(
                "insert_"
            ):
                continue
            owner_keys = set(_ordered_strings(item.get("owner_keys") or []))
            item["donor_source_record_ids"] = _ordered_strings(
                record_id
                for record_id, record in source_records.items()
                if record.get("tree") == "donor"
                and owner_keys.intersection(
                    _ordered_strings(record.get("owner_keys") or [])
                )
            )

        evidence = {
            "schema": SEMANTIC_TOOL_EVIDENCE_SCHEMA,
            "work_unit_id": str(
                work_unit.get("id")
                or work_unit.get("work_unit_id")
                or ""
            ),
            "owners": owner_records,
            "hunks": hunk_records,
            "executable_materializations": [
                {
                    key: deepcopy(item.get(key))
                    for key in (
                        "handle_id",
                        "entity_id",
                        "owner_key",
                        "descriptor",
                        "authority",
                        "mode",
                        "authorized_gap_ids",
                    )
                    if item.get(key) not in (None, "", [], {})
                }
                for item in materializable
            ],
            "unavailable_materializations": unavailable_materializations,
            "unavailable_insertion_anchors": (
                unavailable_insertion_anchors
            ),
            "mutation_targets": [
                {
                    "gap_id": gap_id,
                    "open_dimensions": dimensions,
                    "owner_keys": _ordered_strings(
                        owner_key
                        for owner_key, targets in (
                            mutation_targets_by_owner.items()
                        )
                        if gap_id in targets
                    ),
                    "edit_handle_ids": _ordered_strings(
                        handle_id
                        for handle_id, handle in edit_handles.items()
                        if gap_id in (handle.get("gap_ids") or [])
                    ),
                    "materialization_handle_ids": _ordered_strings(
                        item.get("handle_id")
                        for item in materializable
                        if gap_id
                        in (item.get("authorized_gap_ids") or [])
                    ),
                }
                for gap_id, dimensions in sorted(
                    open_dimensions_by_gap.items()
                )
            ],
            "source_records": [
                deepcopy(source_records[record_id])
                for record_id in sorted(source_records)
            ],
            "read_handles": [
                {
                    key: deepcopy(value)
                    for key, value in read_handles[handle_id].items()
                    if key != "record"
                }
                for handle_id in sorted(read_handles)
            ],
            "edit_handles": [
                {
                    key: deepcopy(value)
                    for key, value in edit_handles[handle_id].items()
                    if key != "source"
                }
                for handle_id in sorted(edit_handles)
            ],
        }
        internal = {
            "bindings": bindings,
            "read_handles": read_handles,
            "edit_handles": edit_handles,
            "materializable": {
                str(item["handle_id"]): item
                for item in materializable
            },
        }
        return evidence, internal

    def inspect(
        self,
        *,
        handle_ids: Iterable[object],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for handle_id in _ordered_strings(handle_ids):
            request = catalog["read_handles"].get(handle_id)
            if not isinstance(request, dict):
                raise ValueError(
                    "inspect handle is not advertised for this exact unit: "
                    + handle_id
                )
            kind = str(request.get("kind") or "")
            if kind == "hunk":
                hunk_id = str(request.get("hunk_id") or "")
                hunk = request.get("record")
                if not isinstance(hunk, dict):
                    raise ValueError(
                        f"inspect hunk record is unavailable: {hunk_id}"
                    )
                results.append(
                    {
                        "kind": "hunk",
                        "handle_id": handle_id,
                        "hunk_id": hunk_id,
                        "status": "resolved",
                        "record": deepcopy(hunk),
                    }
                )
                continue
            if kind not in {"entity", "range"}:
                raise ValueError(
                    f"unsupported advertised inspect handle kind: {kind}"
                )
            tree = str(request.get("tree") or "")
            relative_path = normalize_relpath(
                str(request.get("file") or "")
            )
            start = int(request.get("start_line") or 0)
            end = int(request.get("end_line") or 0)
            root = (
                Path(getattr(self.spec, "old_root_path"))
                if tree == "old"
                else Path(getattr(self.spec, "primary_donor_root"))
            )
            results.append(
                {
                    "kind": kind,
                    "handle_id": handle_id,
                    "tree": tree,
                    "file": relative_path,
                    "entity_id": request.get("entity_id"),
                    "descriptor": deepcopy(
                        request.get("descriptor") or {}
                    ),
                    "source": _source_lines(
                        root,
                        relative_path=relative_path,
                        start_line=start,
                        end_line=end,
                    ),
                }
            )
        return {
            "schema": "semantic_closure_inspection_v1",
            "status": "resolved",
            "results": results,
        }

    def build_source_edit_patch(
        self,
        *,
        source_edits: Iterable[dict[str, Any]],
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate exact source fingerprints and construct an atomic diff."""

        normalized: list[dict[str, Any]] = []
        seen_handles: set[str] = set()
        old_root = Path(getattr(self.spec, "old_root_path"))
        for raw in source_edits:
            if not isinstance(raw, dict):
                raise ValueError("source edit must be an object")
            handle_id = str(raw.get("edit_handle_id") or "")
            if not handle_id or handle_id in seen_handles:
                raise ValueError(
                    "source edit handles must be non-empty and unique"
                )
            seen_handles.add(handle_id)
            advertised = catalog["edit_handles"].get(handle_id)
            if not isinstance(advertised, dict):
                raise ValueError(
                    "source edit handle is not advertised for this exact unit"
                )
            expected = str(
                raw.get("expected_source_sha256") or ""
            )
            if expected != str(advertised.get("source_sha256") or ""):
                raise ValueError(
                    "source edit fingerprint differs from the advertised "
                    "current source"
                )
            current = _source_lines(
                old_root,
                relative_path=str(advertised["file"]),
                start_line=int(advertised["start_line"]),
                end_line=int(advertised["end_line"]),
            )
            if (
                current.get("status") != "resolved"
                or current.get("source_sha256") != expected
            ):
                raise ValueError(
                    "source edit fingerprint is stale in the current tree"
                )
            replacement_source = str(
                raw.get("replacement_source") or ""
            )
            edit_mode = str(advertised.get("edit_mode") or "")
            anchor_source = str(advertised.get("source") or "")
            if edit_mode == "insert_after_anchor" and (
                replacement_source == anchor_source
                or not replacement_source.startswith(anchor_source)
            ):
                raise ValueError(
                    "insert-after source edit must preserve the exact anchor "
                    "as its unchanged prefix"
                )
            if edit_mode == "insert_before_anchor" and (
                replacement_source == anchor_source
                or not replacement_source.endswith(anchor_source)
            ):
                raise ValueError(
                    "insert-before source edit must preserve the exact anchor "
                    "as its unchanged suffix"
                )
            if edit_mode not in {
                "",
                "insert_after_anchor",
                "insert_before_anchor",
            }:
                raise ValueError(
                    "source edit handle has an unsupported edit mode"
                )
            normalized.append(
                {
                    **deepcopy(advertised),
                    "replacement_source": replacement_source,
                }
            )
        diff_text = _source_edit_diff(
            old_root,
            edits=normalized,
        )
        return {
            "schema": "semantic_structured_source_patch_v1",
            "status": (
                "patch" if diff_text else "already_satisfied"
            ),
            "diff_text": diff_text,
            "direct_gap_ids": _ordered_strings(
                gap_id
                for item in normalized
                for gap_id in item.get("gap_ids") or []
            ),
            "edit_regions": [
                {
                    "authority": "exact_structured_source_edit",
                    "file": str(item["file"]),
                    "start_line": int(item["start_line"]),
                    "end_line": int(item["end_line"]),
                    "source_sha256": str(item["source_sha256"]),
                    "source_authorities": list(
                        item.get("authorities") or []
                    ),
                    "direct_gap_ids": _ordered_strings(
                        item.get("gap_ids") or []
                    ),
                    "owner_keys": _ordered_strings(
                        item.get("owner_keys") or []
                    ),
                    "source_owner_keys": _ordered_strings(
                        item.get("source_owner_keys") or []
                    ),
                    "edit_handle_id": str(item["id"]),
                    "edit_mode": str(item.get("edit_mode") or "replace"),
                }
                for item in normalized
            ],
        }

    def _direct_reference_edges(
        self,
        *,
        index: CodeIndex,
        tree: str,
        descriptor: dict[str, Any],
    ) -> list[dict[str, Any]]:
        relative_path = normalize_relpath(
            str(
                descriptor.get("file")
                or descriptor.get("path")
                or ""
            )
        )
        if not relative_path:
            return []
        report = index.index_file(
            "old" if tree == "old" else "donor_0",
            relative_path,
        )
        resolution = resolve_entity_in_case_tree(
            self.spec,
            descriptor=descriptor,
            relative_path=relative_path,
            tree=tree,
            require_definition=(
                descriptor.get("is_definition") is not False
            ),
            index=index,
        )
        owner = (
            resolution.get("entity")
            if isinstance(resolution.get("entity"), dict)
            else {}
        )
        owner_id = str(
            owner.get("id") or owner.get("entity_id") or ""
        )
        owner_contract = str(
            owner.get("contract_key") or owner.get("key") or ""
        )
        if not owner_id:
            return []
        edges: list[dict[str, Any]] = []
        for reference in report.get("references") or []:
            if not isinstance(reference, dict):
                continue
            reference_owner = (
                reference.get("owner_entity")
                if isinstance(reference.get("owner_entity"), dict)
                else {}
            )
            reference_owner_id = str(
                reference.get("owner_entity_id")
                or reference_owner.get("id")
                or ""
            )
            reference_owner_contract = str(
                reference_owner.get("contract_key")
                or reference_owner.get("key")
                or ""
            )
            if (
                reference_owner_id != owner_id
                and not (
                    owner_contract
                    and reference_owner_contract == owner_contract
                )
            ):
                continue
            target = (
                reference.get("referenced_entity")
                if isinstance(
                    reference.get("referenced_entity"),
                    dict,
                )
                else {}
            )
            if not target:
                continue
            edges.append(
                {
                    "id": stable_id(
                        "semantic-direct-dependency",
                        str(reference.get("id") or ""),
                        owner_id,
                        str(
                            target.get("id")
                            or target.get("entity_id")
                            or ""
                        ),
                    ),
                    "tree": tree,
                    "direction": "outgoing",
                    "reference": {
                        key: deepcopy(reference.get(key))
                        for key in (
                            "id",
                            "source_file",
                            "line",
                            "column",
                            "extent",
                            "symbol",
                            "kind",
                            "use_kind",
                            "contract_key",
                        )
                        if reference.get(key) not in (
                            None,
                            "",
                            [],
                            {},
                        )
                    },
                    "owner_entity": _compact_descriptor(owner),
                    "referenced_entity": _compact_descriptor(target),
                }
            )
        return edges

    def expand_dependencies(
        self,
        *,
        owner_keys: Iterable[object],
        direction: str,
        dependency_edge_ids: Iterable[object] = (),
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        """List or bind direct typed edges from exact selected owner nodes.

        Listing is read-only. It returns compact, stable edge ids without
        widening the work unit. A later request may name a subset of those
        exact ids; only that subset becomes owner bindings.
        """

        selected = _ordered_strings(owner_keys)
        unknown = sorted(set(selected) - set(catalog["bindings"]))
        if unknown:
            raise ValueError(
                "dependency expansion references unknown owner keys: "
                + ", ".join(unknown)
            )
        if direction not in {"outgoing", "incoming", "both"}:
            raise ValueError("unsupported dependency direction")
        requested_edge_ids = _ordered_strings(dependency_edge_ids)
        index = self._catalog_resolver_index
        if index is None:
            raise ValueError(
                "dependency expansion requires the current evidence catalog index"
            )
        edges: list[dict[str, Any]] = []
        for owner_key in selected:
            binding = catalog["bindings"][owner_key]
            if direction in {"outgoing", "both"}:
                for tree, descriptor in (
                    (
                        "old",
                        binding.get("current_descriptor")
                        or binding.get("donor_descriptor")
                        or {},
                    ),
                    (
                        "donor",
                        binding.get("donor_descriptor")
                        or binding.get("current_descriptor")
                        or {},
                    ),
                ):
                    if isinstance(descriptor, dict) and descriptor:
                        for edge in self._direct_reference_edges(
                                index=index,
                                tree=tree,
                                descriptor=descriptor,
                        ):
                            edges.append(
                                {
                                    **edge,
                                    "selected_owner_key": owner_key,
                                }
                            )
            if direction in {"incoming", "both"}:
                demand = binding.get("demand_reference")
                owner_descriptor = (
                    binding.get("current_descriptor")
                    or binding.get("donor_descriptor")
                    or {}
                )
                if isinstance(demand, dict) and demand:
                    edges.append(
                        {
                            "id": stable_id(
                                "semantic-direct-demand-edge",
                                owner_key,
                                json.dumps(
                                    demand,
                                    ensure_ascii=True,
                                    sort_keys=True,
                                ),
                            ),
                            "tree": "donor",
                            "direction": "incoming",
                            "selected_owner_key": owner_key,
                            "reference": deepcopy(demand),
                            "referenced_entity": _compact_descriptor(
                                owner_descriptor
                            ),
                        }
                    )

        unique_edges: dict[str, dict[str, Any]] = {
            str(edge["id"]): edge
            for edge in edges
            if str(edge.get("id") or "")
        }
        unknown_edge_ids = sorted(
            set(requested_edge_ids) - set(unique_edges)
        )
        if unknown_edge_ids:
            raise ValueError(
                "dependency binding references edges that are not direct "
                "neighbors of the selected owners: "
                + ", ".join(unknown_edge_ids)
            )
        selected_edges = (
            {
                edge_id: unique_edges[edge_id]
                for edge_id in requested_edge_ids
            }
            if requested_edge_ids
            else {}
        )

        def compact_edge(edge: dict[str, Any]) -> dict[str, Any]:
            reference = (
                edge.get("reference")
                if isinstance(edge.get("reference"), dict)
                else {}
            )
            target = (
                edge.get("referenced_entity")
                if isinstance(edge.get("referenced_entity"), dict)
                else {}
            )
            return {
                "id": str(edge.get("id") or ""),
                "selected_owner_key": str(
                    edge.get("selected_owner_key") or ""
                ),
                "tree": str(edge.get("tree") or ""),
                "direction": str(edge.get("direction") or ""),
                "reference": {
                    key: deepcopy(reference.get(key))
                    for key in (
                        "source_file",
                        "line",
                        "column",
                        "symbol",
                        "use_kind",
                        "contract_key",
                    )
                    if reference.get(key) not in (None, "", [], {})
                },
                "target": {
                    key: deepcopy(target.get(key))
                    for key in (
                        "file",
                        "symbol",
                        "name",
                        "kind",
                        "contract_key",
                        "key",
                        "is_definition",
                    )
                    if target.get(key) not in (None, "", [], {})
                },
            }

        if not requested_edge_ids:
            return {
                "schema": "semantic_direct_dependency_expansion_v2",
                "status": (
                    "listed"
                    if unique_edges
                    else "no_direct_edges_resolved"
                ),
                "selected_owner_keys": selected,
                "direction": direction,
                "dependency_edge_ids": sorted(unique_edges),
                "dependency_edges": [
                    compact_edge(unique_edges[key])
                    for key in sorted(unique_edges)
                ],
                "owner_bindings": [],
            }

        owner_bindings: dict[str, dict[str, Any]] = {}
        for edge in selected_edges.values():
            target = (
                edge.get("referenced_entity")
                if isinstance(edge.get("referenced_entity"), dict)
                else {}
            )
            relative_path = normalize_relpath(
                str(target.get("file") or target.get("path") or "")
            )
            contract_key = str(
                target.get("contract_key")
                or target.get("key")
                or ""
            )
            if not relative_path or not contract_key:
                continue
            permission = str(
                getattr(self.spec, "classify_edit_root")(
                    relative_path
                )
            )
            if permission not in {"editable", "escalation"}:
                continue
            tree = str(edge.get("tree") or "")
            binding = {
                "owner_key": stable_id(
                    "semantic-direct-owner",
                    str(edge["id"]),
                    contract_key,
                    relative_path,
                ),
                "role": "direct_dependency_owner",
                "authority": "typed_direct_dependency_edge",
                "file": relative_path,
                "contract_key": contract_key,
                "symbol": target.get("symbol") or target.get("name"),
                "entity_kind": (
                    target.get("kind")
                    or target.get("entity_kind")
                ),
                "exact_node_refs": [str(edge["id"])],
                "target_presence": (
                    "existing_dependency"
                    if tree == "old"
                    else "donor_dependency"
                ),
                "current_descriptor": deepcopy(target),
                "donor_descriptor": deepcopy(target),
                "edit_permission": permission,
                "semantic_gap_ids": _ordered_strings(
                    (
                        catalog["bindings"][
                            str(edge.get("selected_owner_key") or "")
                        ].get("semantic_gap_ids")
                        or []
                    )
                ),
            }
            owner_bindings.setdefault(
                str(binding["owner_key"]),
                binding,
            )
        return {
            "schema": "semantic_direct_dependency_expansion_v2",
            "status": "expanded",
            "selected_owner_keys": selected,
            "direction": direction,
            "dependency_edge_ids": requested_edge_ids,
            "dependency_edges": [
                compact_edge(selected_edges[key])
                for key in requested_edge_ids
            ],
            "owner_bindings": [
                deepcopy(owner_bindings[key])
                for key in sorted(owner_bindings)
            ],
            "expanded_gap_ids": _ordered_strings(
                gap_id
                for binding in owner_bindings.values()
                for gap_id in binding.get("semantic_gap_ids") or []
            ),
        }

    def _materialize_selected(
        self,
        context: dict[str, Any],
        *,
        selected: dict[str, Any],
        entity_id: str,
        gap_ids: Iterable[object],
        rationale: str,
    ) -> dict[str, Any]:
        """Run the generic structural materializer for one proven absence."""

        requested_gap_ids = _ordered_strings(gap_ids)
        authorized_gap_ids = {
            str(item)
            for item in selected.get("authorized_gap_ids") or []
            if str(item)
        }
        if not authorized_gap_ids or (
            not requested_gap_ids
            or not set(requested_gap_ids) <= authorized_gap_ids
        ):
            raise ValueError(
                "static materialization may only claim the exact gaps "
                "authorized by its current-tree evidence"
            )
        work_unit = dict(context.get("work_unit") or {})
        owner_key = str(selected.get("owner_key") or "")
        descriptor = dict(selected.get("descriptor") or {})
        relative_path = normalize_relpath(
            str(descriptor.get("file") or descriptor.get("path") or "")
        )
        graph_root_key = stable_id(
            "semantic-materialization-root",
            str(work_unit.get("id") or work_unit.get("work_unit_id") or ""),
            owner_key,
        )
        result = build_exact_entity_materialization_action(
            self.spec,
            work_unit_id=str(
                work_unit.get("id")
                or work_unit.get("work_unit_id")
                or ""
            ),
            descriptor={
                **descriptor,
                "id": str(entity_id),
                "entity_kind": (
                    descriptor.get("kind")
                    or descriptor.get("entity_kind")
                ),
                "graph_entity_id": (
                    descriptor.get("id")
                    or descriptor.get("entity_id")
                ),
            },
            editable_files=[relative_path],
            semantic_gap_ids=requested_gap_ids,
            root_key=graph_root_key,
            rationale=str(rationale),
            materialization_authority=str(
                selected.get("authority")
                or "exact_donor_entity_with_proven_target_absence"
            ),
            module_visibility_demands=list(
                selected.get("module_visibility_demands") or []
            ),
            inspectable_ranges=list(
                selected.get("inspectable_ranges") or []
            ),
        )
        generated_action = (
            result.get("action")
            if isinstance(result.get("action"), dict)
            else {}
        )
        patch = (
            generated_action.get("patch")
            if isinstance(generated_action.get("patch"), dict)
            else {}
        )
        diff_text = str(patch.get("content") or "")
        if not diff_text:
            raise ValueError(
                "static entity materialization produced no patch"
            )
        linkage_contract = (
            result.get("linkage_contract")
            if isinstance(result.get("linkage_contract"), dict)
            else {}
        )
        changed_ranges = unified_diff_changed_ranges(diff_text)
        insertion_authorizations = [
            {
                "authority": (
                    "static_structural_insertion_alignment"
                ),
                "file": relative_path,
                "old_line": int(changed.get("old_start") or 0),
                "owner_key": owner_key,
                "materialization_entity_id": str(entity_id),
                "alignment": deepcopy(
                    result.get("alignment")
                    or result.get("bundle_alignment_set")
                    or result.get("conditional_insertion_plan")
                    or linkage_contract
                    or {}
                ),
            }
            for changed in changed_ranges
            if normalize_relpath(str(changed.get("file") or ""))
            == relative_path
            and int(changed.get("old_count") or 0) == 0
        ]
        return {
            "gap_ids": requested_gap_ids,
            "diff_text": diff_text,
            "insertion_authorizations": insertion_authorizations,
        }

    def materialize_entity(
        self,
        *,
        materialization_handle_id: str,
        gap_ids: Iterable[object],
        rationale: str,
        catalog: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a fresh, preflighted structural materialization patch."""

        selected = catalog["materializable"].get(
            str(materialization_handle_id)
        )
        if not isinstance(selected, dict):
            raise ValueError(
                "materialization handle is not advertised as executable "
                "for this exact unit"
            )
        requested_gap_ids = _ordered_strings(gap_ids)
        authorized_gap_ids = {
            str(item)
            for item in selected.get("authorized_gap_ids") or []
            if str(item)
        }
        if not authorized_gap_ids or (
            not requested_gap_ids
            or not set(requested_gap_ids) <= authorized_gap_ids
        ):
            raise ValueError(
                "static materialization may only claim the exact gaps "
                "authorized by current-tree evidence"
            )
        preflight = selected.get("preflight_result")
        if not isinstance(preflight, dict):
            raise ValueError(
                "advertised materialization has no successful preflight"
            )
        result = deepcopy(preflight)
        result["gap_ids"] = requested_gap_ids
        result["rationale"] = str(rationale)
        return result
