from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Iterable

from driver_backport.analysis.code_index import (
    CodeIndex,
)
from driver_backport.analysis.entity_resolution import (
    resolve_entity_in_case_tree,
)
from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths as _ordered_paths,
)
from driver_backport.planning.sanity_check import (
    unified_diff_changed_ranges,
)


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    ordered = sorted(
        (max(1, int(start)), max(1, int(end)))
        for start, end in intervals
        if int(start) > 0 and int(end) >= int(start)
    )
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], end),
            )
        else:
            merged.append((start, end))
    return merged


def resolve_current_owner_ranges(
    spec: object,
    *,
    work_unit: dict[str, Any],
) -> dict[str, Any]:
    """Resolve every exact owner again in the current target tree."""

    code_index = CodeIndex(
        spec,
        graph_projection="records",
    )
    report_cache: dict[tuple[str, str], dict[str, Any]] = {}
    intervals_by_file: dict[str, list[tuple[int, int]]] = {}
    resolutions: list[dict[str, Any]] = []
    for binding in work_unit.get("owner_bindings") or []:
        if not isinstance(binding, dict):
            continue
        descriptor = (
            binding.get("current_descriptor")
            if isinstance(binding.get("current_descriptor"), dict)
            and binding.get("current_descriptor")
            else binding.get("donor_descriptor")
            if isinstance(binding.get("donor_descriptor"), dict)
            else {}
        )
        relative_path = normalize_relpath(
            str(
                descriptor.get("file")
                or binding.get("file")
                or ""
            )
        )
        if not relative_path or not descriptor:
            resolutions.append(
                {
                    "owner_key": binding.get("owner_key"),
                    "status": "descriptor_unavailable",
                    "file": relative_path,
                }
            )
            continue
        require_definition = descriptor.get("is_definition") is not False
        resolution = resolve_entity_in_case_tree(
            spec,
            descriptor=descriptor,
            relative_path=relative_path,
            tree="old",
            require_definition=require_definition,
            index=code_index,
            report_cache=report_cache,
        )
        entity = (
            resolution.get("entity")
            if isinstance(resolution.get("entity"), dict)
            else {}
        )
        extent = (
            entity.get("extent")
            if isinstance(entity.get("extent"), dict)
            else {}
        )
        start = int(extent.get("start_line") or 0)
        end = int(extent.get("end_line") or start)
        if resolution.get("status") == "resolved" and start > 0:
            intervals_by_file.setdefault(
                relative_path,
                [],
            ).append((start, max(start, end)))
        resolutions.append(
            {
                "owner_key": binding.get("owner_key"),
                "role": binding.get("role"),
                "file": relative_path,
                "status": resolution.get("status"),
                "contract_key": resolution.get("contract_key"),
                "entity_kind": resolution.get("entity_kind"),
                "symbol": resolution.get("symbol"),
                "current_entity": {
                    key: deepcopy(entity.get(key))
                    for key in (
                        "id",
                        "file",
                        "contract_key",
                        "kind",
                        "symbol",
                        "extent",
                    )
                    if entity.get(key) not in (None, "", [], {})
                },
            }
        )
    return {
        "schema": "semantic_current_owner_ranges_v1",
        "work_unit_id": str(
            work_unit.get("id")
            or work_unit.get("work_unit_id")
            or ""
        ),
        "ranges_by_file": {
            relative_path: [
                {"start_line": start, "end_line": end}
                for start, end in _merge_intervals(intervals)
            ]
            for relative_path, intervals in sorted(
                intervals_by_file.items()
            )
        },
        "resolutions": resolutions,
    }


def _range_is_covered(
    *,
    start: int,
    end: int,
    intervals: Iterable[dict[str, Any]],
) -> bool:
    remaining_start = int(start)
    target_end = int(end)
    for interval in intervals:
        interval_start = int(interval.get("start_line") or 0)
        interval_end = int(interval.get("end_line") or 0)
        if interval_end < remaining_start:
            continue
        if interval_start > remaining_start:
            return False
        remaining_start = max(remaining_start, interval_end + 1)
        if remaining_start > target_end:
            return True
    return remaining_start > target_end


def _authorized_insertion_points(
    authorizations: Iterable[dict[str, Any]],
) -> set[tuple[str, int]]:
    result: set[tuple[str, int]] = set()
    for authorization in authorizations:
        if not isinstance(authorization, dict):
            continue
        if (
            authorization.get("authority")
            != "static_structural_insertion_alignment"
        ):
            continue
        relative_path = normalize_relpath(
            str(authorization.get("file") or "")
        )
        line = int(authorization.get("old_line") or 0)
        if relative_path and line > 0:
            result.add((relative_path, line))
    return result


def _exact_source_edit_intervals(
    authorizations: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = {}
    for authorization in authorizations:
        if not isinstance(authorization, dict) or (
            authorization.get("authority")
            != "exact_structured_source_edit"
        ):
            continue
        relative_path = normalize_relpath(
            str(authorization.get("file") or "")
        )
        start = int(authorization.get("start_line") or 0)
        end = int(authorization.get("end_line") or start)
        if relative_path and start > 0 and end >= start:
            intervals.setdefault(relative_path, []).append(
                (start, end)
            )
    return {
        relative_path: [
            {"start_line": start, "end_line": end}
            for start, end in _merge_intervals(values)
        ]
        for relative_path, values in intervals.items()
    }


def _source_edit_covers_insertion(
    *,
    line: int,
    intervals: Iterable[dict[str, Any]],
) -> bool:
    """Return whether an insertion is produced by an exact replacement.

    Replacing lines N..M may insert immediately before N, within that range,
    or immediately after M.  The final point is represented as M + 1 in a
    unified diff even though the replacement authority ends at M.
    """

    return any(
        int(interval.get("start_line") or 0)
        <= int(line)
        <= int(interval.get("end_line") or 0) + 1
        for interval in intervals
    )


def validate_semantic_patch_boundary(
    spec: object,
    *,
    work_unit: dict[str, Any],
    diff_text: str,
    insertion_authorizations: Iterable[dict[str, Any]] = (),
    source_edit_authorizations: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Authorize changed blocks against the union of current exact owners."""

    changed_ranges = unified_diff_changed_ranges(diff_text)
    if not changed_ranges:
        return {
            "schema": "semantic_patch_boundary_v1",
            "status": "denied",
            "reason": "patch_has_no_changed_ranges",
            "changed_ranges": [],
        }
    owners = resolve_current_owner_ranges(
        spec,
        work_unit=work_unit,
    )
    allowed_files = set(
        _ordered_paths(work_unit.get("allowed_fileset") or [])
    )
    owner_ranges = dict(owners.get("ranges_by_file") or {})
    insertion_points = _authorized_insertion_points(
        insertion_authorizations
    )
    source_edit_ranges = _exact_source_edit_intervals(
        source_edit_authorizations
    )
    diff_sha256 = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    verified_structured_files = {
        normalize_relpath(str(item.get("file") or ""))
        for item in source_edit_authorizations
        if isinstance(item, dict)
        and item.get("authority") == "exact_structured_source_edit"
        and str(item.get("verified_diff_sha256") or "") == diff_sha256
        and normalize_relpath(str(item.get("file") or ""))
    }
    structured_transformation_verified = bool(
        verified_structured_files
    )
    decisions: list[dict[str, Any]] = []
    for changed in changed_ranges:
        relative_path = normalize_relpath(
            str(changed.get("file") or "")
        )
        old_start = int(changed.get("old_start") or 0)
        old_end = int(changed.get("old_end") or old_start)
        old_count = int(changed.get("old_count") or 0)
        if relative_path not in allowed_files:
            authorized = False
            reason = "file_outside_work_unit"
        elif (
            structured_transformation_verified
            and relative_path in verified_structured_files
        ):
            authorized = True
            reason = "inside_verified_structured_source_transformation"
        elif old_count == 0:
            inside_owner = _range_is_covered(
                start=old_start,
                end=old_start,
                intervals=owner_ranges.get(relative_path) or [],
            )
            inside_source_edit = _source_edit_covers_insertion(
                line=old_start,
                intervals=(
                    source_edit_ranges.get(relative_path) or []
                ),
            )
            authorized = (
                inside_owner
                or inside_source_edit
                or (relative_path, old_start) in insertion_points
            )
            reason = (
                "inside_current_owner"
                if inside_owner
                else "inside_exact_structured_source_edit"
                if inside_source_edit
                else "static_structural_insertion_alignment"
                if authorized
                else "insertion_has_no_exact_owner_or_static_alignment"
            )
        else:
            inside_owner = _range_is_covered(
                start=old_start,
                end=old_end,
                intervals=owner_ranges.get(relative_path) or [],
            )
            inside_source_edit = _range_is_covered(
                start=old_start,
                end=old_end,
                intervals=source_edit_ranges.get(relative_path) or [],
            )
            authorized = (
                inside_owner
                or inside_source_edit
            )
            reason = (
                "inside_current_owner_union"
                if inside_owner
                else "inside_exact_structured_source_edit"
                if inside_source_edit
                else "changed_lines_escape_current_owner_union"
            )
        decisions.append(
            {
                **deepcopy(changed),
                "authorized": authorized,
                "reason": reason,
            }
        )
    denied = [
        decision
        for decision in decisions
        if not decision["authorized"]
    ]
    return {
        "schema": "semantic_patch_boundary_v1",
        "status": "authorized" if not denied else "denied",
        "work_unit_id": owners.get("work_unit_id"),
        "changed_ranges": decisions,
        "denied_changed_ranges": denied,
        "owner_resolution": owners,
        "insertion_authorizations": [
            deepcopy(item)
            for item in insertion_authorizations
            if isinstance(item, dict)
        ],
        "source_edit_authorizations": [
            deepcopy(item)
            for item in source_edit_authorizations
            if isinstance(item, dict)
        ],
    }
