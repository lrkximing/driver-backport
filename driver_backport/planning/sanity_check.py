#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any

from driver_backport.core.common import normalize_relpath


HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


def normalize_diff_path(path: str) -> str:
    text = path.strip().split("\t", 1)[0]
    if text.startswith(("a/", "b/")):
        text = text[2:]
    return normalize_relpath(text)


def unified_diff_hunk_ranges(diff_text: str) -> list[dict[str, Any]]:
    """Return old- and new-file locations for each unified-diff hunk."""

    ranges: list[dict[str, Any]] = []
    current_file = ""
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            current_file = normalize_diff_path(line[4:])
            continue
        match = HUNK_HEADER_RE.match(line)
        if not match or not current_file:
            continue
        old_start = int(match.group("old_start") or 0)
        old_count = int(match.group("old_count") or 1)
        new_start = int(match.group("new_start") or 0)
        new_count = int(match.group("new_count") or 1)
        ranges.append(
            {
                "file": current_file,
                "old_start": old_start,
                "old_end": old_start + max(old_count, 1) - 1,
                "old_count": old_count,
                "new_start": new_start,
                "new_end": new_start + max(new_count, 1) - 1,
                "new_count": new_count,
            }
        )
    return ranges


def unified_diff_changed_ranges(
    diff_text: str,
    *,
    include_content: bool = False,
) -> list[dict[str, Any]]:
    """Return only contiguous changed-line spans from a unified diff.

    Hunk headers include unchanged context.  That context is useful for patch
    overlap checks, but it is not evidence that a semantic contract was
    modified.  This parser keeps the old and new coordinate spaces separate
    and records zero-length sides for pure insertions or deletions.
    """

    ranges: list[dict[str, Any]] = []
    old_file = ""
    new_file = ""
    current_file = ""
    old_line = 0
    new_line = 0
    in_hunk = False
    block_old_start = 0
    block_new_start = 0
    block_old_count = 0
    block_new_count = 0
    block_old_lines: list[str] = []
    block_new_lines: list[str] = []

    def flush_block() -> None:
        nonlocal block_old_start
        nonlocal block_new_start
        nonlocal block_old_count
        nonlocal block_new_count
        nonlocal block_old_lines
        nonlocal block_new_lines
        if not current_file or not (block_old_count or block_new_count):
            block_old_start = 0
            block_new_start = 0
            block_old_count = 0
            block_new_count = 0
            block_old_lines = []
            block_new_lines = []
            return
        record = {
            "file": current_file,
            "old_start": block_old_start,
            "old_end": (
                block_old_start + block_old_count - 1
                if block_old_count
                else block_old_start
            ),
            "old_count": block_old_count,
            "new_start": block_new_start,
            "new_end": (
                block_new_start + block_new_count - 1
                if block_new_count
                else block_new_start
            ),
            "new_count": block_new_count,
            "range_kind": "exact_changed_lines",
            "change_kind": (
                "replacement"
                if block_old_count and block_new_count
                else "deletion"
                if block_old_count
                else "insertion"
            ),
        }
        if include_content:
            record["old_lines"] = list(block_old_lines)
            record["new_lines"] = list(block_new_lines)
        ranges.append(record)
        block_old_start = 0
        block_new_start = 0
        block_old_count = 0
        block_new_count = 0
        block_old_lines = []
        block_new_lines = []

    for line in diff_text.splitlines():
        if line.startswith("--- "):
            flush_block()
            old_file = normalize_diff_path(line[4:])
            in_hunk = False
            continue
        if line.startswith("+++ "):
            flush_block()
            new_file = normalize_diff_path(line[4:])
            current_file = (
                new_file
                if new_file not in {"", "dev/null"}
                else old_file
            )
            in_hunk = False
            continue
        match = HUNK_HEADER_RE.match(line)
        if match:
            flush_block()
            old_line = int(match.group("old_start") or 0)
            new_line = int(match.group("new_start") or 0)
            in_hunk = bool(current_file)
            continue
        if not in_hunk:
            continue
        if line.startswith("\\"):
            continue
        marker = line[:1]
        if marker == " ":
            flush_block()
            old_line += 1
            new_line += 1
            continue
        if marker not in {"+", "-"}:
            flush_block()
            in_hunk = False
            continue
        if not (block_old_count or block_new_count):
            block_old_start = old_line
            block_new_start = new_line
        if marker == "-":
            block_old_count += 1
            block_old_lines.append(line[1:])
            old_line += 1
        else:
            block_new_count += 1
            block_new_lines.append(line[1:])
            new_line += 1
    flush_block()
    return ranges


def build_sanity_report(
    *,
    candidates: list[dict[str, Any]],
    transaction_preview: dict[str, Any],
) -> dict[str, Any]:
    """Report only transaction and unified-diff conflicts.

    C-language compatibility is validated by the CodeIndex/clang contract
    checker and the real build. This layer must not infer semantics from source
    text or special-case a compiler diagnostic family.
    """

    warnings: list[dict[str, Any]] = []

    for group in transaction_preview.get("transaction_groups") or []:
        if int(group.get("candidate_count") or 0) > 1:
            warnings.append(
                {
                    "checker": "SameOwnerMultiPatchChecker",
                    "severity": "warning",
                    "owner": group.get("owner"),
                    "phase": group.get("phase"),
                    "candidate_count": group.get("candidate_count"),
                    "message": "one graph owner has multiple patch candidates",
                }
            )

    for conflict in transaction_preview.get("conflicts") or []:
        if conflict.get("kind") == "file_touched_by_multiple_transaction_groups":
            warnings.append(
                {
                    "checker": "SameFileMultiOwnerChecker",
                    "severity": "warning",
                    "file": conflict.get("file"),
                    "groups": conflict.get("groups"),
                    "message": "one file is assigned to multiple graph-root transactions",
                }
            )

    hunk_ranges: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        diff_text = str(candidate.get("diff_text") or "")
        if not diff_text:
            continue
        unit_id = str(candidate.get("unit_id") or candidate.get("index") or "")
        for hunk in unified_diff_hunk_ranges(diff_text):
            hunk_ranges.setdefault(str(hunk["file"]), []).append(
                {"unit_id": unit_id, **hunk}
            )
    for relative_path, ranges in hunk_ranges.items():
        ordered = sorted(
            ranges,
            key=lambda item: (int(item["old_start"]), int(item["old_end"])),
        )
        for left, right in zip(ordered, ordered[1:]):
            if int(right["old_start"]) <= int(left["old_end"]):
                warnings.append(
                    {
                        "checker": "PatchOverlapChecker",
                        "severity": "warning",
                        "file": relative_path,
                        "left": left,
                        "right": right,
                        "message": "patch candidates overlap in the same old-file hunk range",
                    }
                )

    by_checker: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for warning in warnings:
        checker = str(warning.get("checker") or "unknown")
        severity = str(warning.get("severity") or "warning")
        by_checker[checker] = by_checker.get(checker, 0) + 1
        by_severity[severity] = by_severity.get(severity, 0) + 1

    return {
        "schema": "patch_sanity_report_v2",
        "summary": {
            "warning_count": len(warnings),
            "by_checker": by_checker,
            "by_severity": by_severity,
        },
        "warnings": warnings,
    }
