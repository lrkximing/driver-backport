from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_strings as _ordered_strings,
)


def _authorized_changes(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, dict) or raw.get("authorized") is False:
            continue
        relative_path = normalize_relpath(str(raw.get("file") or ""))
        old_start = int(raw.get("old_start") or 0)
        old_count = int(raw.get("old_count") or 0)
        new_start = int(raw.get("new_start") or 0)
        new_count = int(raw.get("new_count") or 0)
        if (
            not relative_path
            or old_start < 0
            or new_start < 0
            or (old_count <= 0 and new_count <= 0)
        ):
            continue
        change = {
            "file": relative_path,
            "old_start": old_start,
            "old_end": (
                old_start + old_count - 1
                if old_count
                else old_start
            ),
            "old_count": old_count,
            "new_start": new_start,
            "new_end": (
                new_start + new_count - 1
                if new_count
                else new_start
            ),
            "new_count": new_count,
        }
        if "direct_gap_ids" in raw:
            change["direct_gap_ids"] = _ordered_strings(
                raw.get("direct_gap_ids") or []
            )
        changes.append(change)
    return sorted(
        changes,
        key=lambda item: (
            item["file"],
            item["old_start"],
            item["new_start"],
        ),
    )


def active_patch_regions_intersecting(
    regions: Iterable[dict[str, Any]],
    *,
    file: str,
    start_line: int,
    end_line: int | None = None,
) -> list[dict[str, Any]]:
    """Return exact active patch regions intersecting one source locus."""

    relative_path = normalize_relpath(str(file or ""))
    start = int(start_line or 0)
    end = int(end_line if end_line is not None else start)
    if not relative_path or start <= 0 or end < start:
        return []
    matches: list[dict[str, Any]] = []
    for raw in regions:
        if not isinstance(raw, dict):
            continue
        if normalize_relpath(str(raw.get("file") or "")) != relative_path:
            continue
        region_start = int(raw.get("start_line") or 0)
        region_end = int(raw.get("end_line") or region_start)
        if region_start <= 0 or region_end < region_start:
            continue
        if not (region_end < start or region_start > end):
            matches.append(deepcopy(raw))
    return matches


def active_patch_ids_intersecting_changes(
    regions: Iterable[dict[str, Any]],
    changes: Iterable[dict[str, Any]],
) -> list[str]:
    """Return patch identities whose current regions a new hunk replaces."""

    patch_ids: list[str] = []
    for change in _authorized_changes(changes):
        start = int(change["old_start"])
        end = int(change["old_end"])
        for region in active_patch_regions_intersecting(
            regions,
            file=str(change["file"]),
            start_line=start,
            end_line=end,
        ):
            patch_ids.extend(region.get("patch_ids") or [])
    return _ordered_strings(patch_ids)


def _line_after_changes(
    line: int,
    changes: Iterable[dict[str, Any]],
) -> int:
    offset = 0
    for change in changes:
        old_start = int(change["old_start"])
        old_count = int(change["old_count"])
        if old_count == 0:
            if old_start <= line:
                offset += int(change["new_count"])
        elif int(change["old_end"]) < line:
            offset += int(change["new_count"]) - old_count
    return max(1, int(line) + offset)


def _point_after_changes(
    line: int,
    changes: Iterable[dict[str, Any]],
) -> int:
    offset = 0
    for change in changes:
        old_start = int(change["old_start"])
        old_count = int(change["old_count"])
        if old_count == 0:
            if old_start <= line:
                offset += int(change["new_count"])
            continue
        if int(change["old_end"]) < line:
            offset += int(change["new_count"]) - old_count
        elif old_start <= line <= int(change["old_end"]):
            return max(1, int(change["new_start"]))
    return max(1, int(line) + offset)


def _surviving_segments(
    start: int,
    end: int,
    changes: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    cursor = int(start)
    segments: list[tuple[int, int]] = []
    for change in changes:
        old_start = int(change["old_start"])
        old_count = int(change["old_count"])
        old_end = int(change["old_end"])
        if old_count == 0:
            if cursor < old_start <= end:
                segments.append((cursor, old_start - 1))
                cursor = old_start
            continue
        if old_end < cursor:
            continue
        if old_start > end:
            break
        if cursor < old_start:
            segments.append((cursor, min(end, old_start - 1)))
        cursor = max(cursor, old_end + 1)
        if cursor > end:
            break
    if cursor <= end:
        segments.append((cursor, end))
    return [
        (
            _line_after_changes(segment_start, changes),
            _line_after_changes(segment_end, changes),
        )
        for segment_start, segment_end in segments
    ]


def _merge_regions(
    regions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    region_records = [
        deepcopy(region)
        for region in regions
        if isinstance(region, dict)
    ]
    line_regions = sorted(
        (
            region
            for region in region_records
            if region.get("region_kind") == "line_range"
        ),
        key=lambda item: (
            str(item.get("file") or ""),
            int(item.get("start_line") or 0),
            int(item.get("end_line") or 0),
        ),
    )
    merged: list[dict[str, Any]] = []
    for region in line_regions:
        region["patch_ids"] = _ordered_strings(
            region.get("patch_ids") or []
        )
        region["gap_ids"] = _ordered_strings(
            region.get("gap_ids") or []
        )
        region["direct_gap_ids"] = _ordered_strings(
            (
                region.get("direct_gap_ids")
                if "direct_gap_ids" in region
                else region.get("gap_ids")
            )
            or []
        )
        same_provenance = bool(merged) and all(
            merged[-1].get(field) == region.get(field)
            for field in (
                "authority",
                "patch_ids",
                "gap_ids",
                "direct_gap_ids",
            )
        )
        if (
            same_provenance
            and merged[-1]["file"] == region["file"]
            and int(region["start_line"])
            <= int(merged[-1]["end_line"]) + 1
        ):
            merged[-1]["end_line"] = max(
                int(merged[-1]["end_line"]),
                int(region["end_line"]),
            )
        else:
            merged.append(region)

    points: dict[tuple[str, int], dict[str, Any]] = {}
    for raw in region_records:
        if raw.get("region_kind") != "insertion_point":
            continue
        region = deepcopy(raw)
        key = (
            normalize_relpath(str(region.get("file") or "")),
            int(region.get("start_line") or 0),
        )
        if not key[0] or key[1] <= 0:
            continue
        existing = points.setdefault(
            key,
            {
                **region,
                "file": key[0],
                "start_line": key[1],
                "end_line": key[1],
                "patch_ids": [],
                "gap_ids": [],
                "direct_gap_ids": [],
            },
        )
        existing["patch_ids"] = _ordered_strings(
            [
                *(existing.get("patch_ids") or []),
                *(region.get("patch_ids") or []),
            ]
        )
        existing["gap_ids"] = _ordered_strings(
            [
                *(existing.get("gap_ids") or []),
                *(region.get("gap_ids") or []),
            ]
        )
        existing["direct_gap_ids"] = _ordered_strings(
            [
                *(existing.get("direct_gap_ids") or []),
                *(region.get("direct_gap_ids") or []),
            ]
        )
    return [*merged, *[points[key] for key in sorted(points)]]


def rebase_active_patch_regions(
    existing_regions: Iterable[dict[str, Any]],
    *,
    changed_ranges: Iterable[dict[str, Any]],
    patch_id: str,
    gap_ids: Iterable[object],
) -> list[dict[str, Any]]:
    """Map all active patch ownership into the latest source coordinates."""

    primary_gap_ids = _ordered_strings(gap_ids)
    changes = _authorized_changes(changed_ranges)
    by_file: dict[str, list[dict[str, Any]]] = {}
    for change in changes:
        by_file.setdefault(change["file"], []).append(change)

    lineage_by_change: dict[
        tuple[str, int, int, int, int],
        dict[str, list[str]],
    ] = {}
    existing_records = [
        deepcopy(region)
        for region in existing_regions
        if isinstance(region, dict)
    ]
    for change in changes:
        old_start = int(change["old_start"])
        old_end = int(change["old_end"])
        inherited = {
            "patch_ids": [],
            "gap_ids": [],
        }
        for region in existing_records:
            if normalize_relpath(
                str(region.get("file") or "")
            ) != change["file"]:
                continue
            region_start = int(region.get("start_line") or 0)
            region_end = int(
                region.get("end_line") or region_start
            )
            if region.get("region_kind") == "insertion_point":
                overlaps = old_start <= region_start <= old_end
            elif int(change["old_count"]) == 0:
                overlaps = region_start <= old_start <= region_end
            else:
                overlaps = not (
                    region_end < old_start or region_start > old_end
                )
            if not overlaps:
                continue
            inherited["patch_ids"] = _ordered_strings(
                [
                    *inherited["patch_ids"],
                    *(region.get("patch_ids") or []),
                ]
            )
            inherited["gap_ids"] = _ordered_strings(
                [
                    *inherited["gap_ids"],
                    *(region.get("gap_ids") or []),
                ]
            )
        lineage_by_change[
            (
                change["file"],
                old_start,
                int(change["old_count"]),
                int(change["new_start"]),
                int(change["new_count"]),
            )
        ] = inherited

    rebased: list[dict[str, Any]] = []
    for region in existing_records:
        relative_path = normalize_relpath(str(region.get("file") or ""))
        start = int(region.get("start_line") or 0)
        end = int(region.get("end_line") or start)
        if not relative_path or start <= 0 or end < start:
            continue
        file_changes = by_file.get(relative_path) or []
        if region.get("region_kind") == "insertion_point":
            line = _point_after_changes(start, file_changes)
            rebased.append(
                {
                    **region,
                    "file": relative_path,
                    "start_line": line,
                    "end_line": line,
                }
            )
            continue
        for segment_start, segment_end in _surviving_segments(
            start,
            end,
            file_changes,
        ):
            rebased.append(
                {
                    **region,
                    "file": relative_path,
                    "region_kind": "line_range",
                    "start_line": segment_start,
                    "end_line": segment_end,
                }
            )

    for change in changes:
        inherited = lineage_by_change[
            (
                change["file"],
                int(change["old_start"]),
                int(change["old_count"]),
                int(change["new_start"]),
                int(change["new_count"]),
            )
        ]
        direct_gap_ids = (
            list(change.get("direct_gap_ids") or [])
            if "direct_gap_ids" in change
            else primary_gap_ids
        )
        provenance = {
            "patch_ids": _ordered_strings(
                [*inherited["patch_ids"], str(patch_id)]
            ),
            "gap_ids": _ordered_strings(
                [*inherited["gap_ids"], *primary_gap_ids]
            ),
            "direct_gap_ids": _ordered_strings(
                direct_gap_ids
            ),
        }
        if int(change["new_count"]) > 0:
            rebased.append(
                {
                    "authority": "active_semantic_patch_postimage",
                    "region_kind": "line_range",
                    "file": change["file"],
                    "start_line": max(1, int(change["new_start"])),
                    "end_line": max(1, int(change["new_end"])),
                    **provenance,
                }
            )
        else:
            rebased.append(
                {
                    "authority": "active_semantic_patch_postimage",
                    "region_kind": "insertion_point",
                    "file": change["file"],
                    "start_line": max(1, int(change["new_start"])),
                    "end_line": max(1, int(change["new_start"])),
                    **provenance,
                    "direct_gap_ids": _ordered_strings(
                        direct_gap_ids
                    ),
                }
            )
    return _merge_regions(rebased)
