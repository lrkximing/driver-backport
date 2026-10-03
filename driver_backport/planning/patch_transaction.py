#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths,
    stable_id,
    utc_now,
)
from driver_backport.migration.owner_key import is_build_path, owner_key_for_unit


PHASE_ORDER = {
    "build_preparation": 10,
    "code_graph_repair": 20,
    "unknown": 30,
}


def classify_patch_phase(touched_files: list[str]) -> str:
    touched = ordered_unique_relpaths(touched_files)
    if any(is_build_path(item) for item in touched):
        return "build_preparation"
    return "code_graph_repair"


def build_patch_transaction_preview(
    *,
    case_id: str,
    source_plan_path: str | None,
    response_dir: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        unit = dict(candidate.get("unit") or {})
        touched_files = ordered_unique_relpaths([str(item) for item in candidate.get("touched_files") or []])
        action = str(candidate.get("action") or "")
        phase = str(candidate.get("phase") or "")
        if not phase and action == "apply_patch":
            phase = classify_patch_phase(touched_files)
        elif not phase:
            phase = "unknown"
        owner = str(candidate.get("owner") or "")
        if not owner:
            owner = owner_key_for_unit(unit=unit, touched_files=touched_files)
        patch_signature = str(candidate.get("patch_signature") or "")
        patch_transaction_id = (
            stable_id("patch-tx", patch_signature)
            if patch_signature
            else stable_id("response-tx", str(candidate.get("unit_id") or ""), str(index))
        )
        normalized_candidates.append(
            {
                **{key: value for key, value in candidate.items() if key not in {"unit", "diff_text"}},
                "patch_transaction_id": patch_transaction_id,
                "index": index,
                "phase": phase,
                "phase_order": PHASE_ORDER.get(phase, PHASE_ORDER["unknown"]),
                "owner": owner,
                "touched_files": touched_files,
                "unit": {
                    "id": unit.get("id"),
                    "work_unit_id": unit.get("work_unit_id"),
                    "title": unit.get("title"),
                    "scope": unit.get("scope"),
                    "unit_kind": unit.get("unit_kind"),
                },
            }
        )

    groups_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in normalized_candidates:
        key = (str(candidate["phase"]), str(candidate["owner"]))
        group = groups_by_key.setdefault(
            key,
            {
                "id": stable_id("ptx", case_id, "repair", key[0], key[1]),
                "phase": key[0],
                "phase_order": candidate["phase_order"],
                "owner": key[1],
                "candidates": [],
                "touched_files": [],
            },
        )
        group["candidates"].append(candidate)
        group["touched_files"] = ordered_unique_relpaths(
            [*group.get("touched_files", []), *candidate.get("touched_files", [])]
        )

    groups = sorted(
        groups_by_key.values(),
        key=lambda item: (int(item.get("phase_order") or 0), str(item.get("owner") or "")),
    )
    for group in groups:
        group["candidate_count"] = len(group["candidates"])

    file_to_groups: dict[str, list[str]] = {}
    for group in groups:
        for relative_path in group.get("touched_files") or []:
            file_to_groups.setdefault(normalize_relpath(relative_path), []).append(str(group["id"]))

    conflicts: list[dict[str, Any]] = []
    for relative_path, group_ids in sorted(file_to_groups.items()):
        unique_group_ids = sorted(set(group_ids))
        if len(unique_group_ids) > 1:
            conflicts.append(
                {
                    "kind": "file_touched_by_multiple_transaction_groups",
                    "file": relative_path,
                    "groups": unique_group_ids,
                }
            )

    duplicate_signatures: dict[str, list[str]] = {}
    for candidate in normalized_candidates:
        signature = str(candidate.get("patch_signature") or "")
        if signature:
            duplicate_signatures.setdefault(signature, []).append(str(candidate.get("unit_id") or ""))
    for signature, unit_ids in sorted(duplicate_signatures.items()):
        unique_unit_ids = sorted(set(unit_ids))
        if len(unique_unit_ids) > 1:
            conflicts.append(
                {
                    "kind": "duplicate_patch_signature",
                    "patch_signature": signature,
                    "unit_ids": unique_unit_ids,
                }
            )

    status_counts: dict[str, int] = {}
    owner_counts: dict[str, int] = {}
    for candidate in normalized_candidates:
        status = str(candidate.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        owner = str(candidate.get("owner") or "unknown")
        owner_counts[owner] = owner_counts.get(owner, 0) + 1
    same_file_multi_owner_count = sum(
        1
        for conflict in conflicts
        if conflict.get("kind") == "file_touched_by_multiple_transaction_groups"
    )
    duplicate_owner_count = sum(1 for count in owner_counts.values() if count > 1)

    return {
        "schema": "patch_transaction_preview_v1",
        "case_id": case_id,
        "plan_kind": "repair",
        "source_plan_path": source_plan_path,
        "response_dir": response_dir,
        "generated_at": utc_now(),
        "summary": {
            "candidate_count": len(normalized_candidates),
            "owner_count": len(owner_counts),
            "duplicate_owner_count": duplicate_owner_count,
            "multi_owner_same_file_count": same_file_multi_owner_count,
            "transaction_group_count": len(groups),
            "conflict_count": len(conflicts),
            "candidate_status_counts": status_counts,
            "phase_counts": {
                phase: sum(1 for item in normalized_candidates if item.get("phase") == phase)
                for phase in sorted({str(item.get("phase")) for item in normalized_candidates})
            },
            "owner_distribution": dict(sorted(owner_counts.items(), key=lambda item: (-item[1], item[0]))[:32]),
        },
        "candidates": normalized_candidates,
        "transaction_groups": groups,
        "conflicts": conflicts,
    }
