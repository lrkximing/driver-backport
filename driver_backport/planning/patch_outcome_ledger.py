#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    normalize_relpath,
    stable_id,
    stable_json_sha256,
    utc_now,
)
from driver_backport.core.record_log import (
    append_jsonl_records,
    read_jsonl_records,
)


PATCH_OUTCOME_LEDGER_NAME = "patch_outcome_ledger.jsonl"
UNPRODUCTIVE_VALIDATION_STATUSES = {
    "unchanged",
    "changed",
    "regressed",
    "regressed_at_patch_locus",
}
PRODUCTIVE_VALIDATION_STATUSES = {"progressed", "resolved"}


def patch_outcome_ledger_path(state_root: Path) -> Path:
    return state_root / PATCH_OUTCOME_LEDGER_NAME


def _file_preimage(spec: CaseSpec, relative_path: str) -> dict[str, Any]:
    relative_path = normalize_relpath(relative_path)
    path = spec.resolve_old(relative_path)
    if not path.exists():
        return {
            "path": relative_path,
            "exists": False,
            "kind": "missing",
            "sha256": None,
        }
    if not path.is_file():
        return {
            "path": relative_path,
            "exists": True,
            "kind": "non_file",
            "sha256": None,
        }
    try:
        content = path.read_bytes()
    except OSError as exc:
        return {
            "path": relative_path,
            "exists": True,
            "kind": "unreadable",
            "sha256": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    return {
        "path": relative_path,
        "exists": True,
        "kind": "file",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _normalized_source_delta(diff_text: str) -> dict[str, Any]:
    """Return context-independent forward and inverse source deltas."""

    by_file: dict[str, dict[str, Any]] = {}
    current_file = ""
    pending_old_file = ""
    in_hunk = False

    def normalized_line(value: str) -> str:
        # Source whitespace can be semantically relevant in preprocessor
        # directives and continued macros.  Exact content is deliberately
        # retained so the safety ledger cannot reject a merely similar patch
        # as the inverse of an earlier productive one.
        return value.rstrip("\r")

    for raw_line in str(diff_text or "").splitlines():
        if raw_line.startswith("--- "):
            candidate = raw_line[4:].strip().split("\t", 1)[0]
            pending_old_file = (
                ""
                if candidate == "/dev/null"
                else normalize_relpath(
                    candidate[2:]
                    if candidate.startswith("a/")
                    else candidate
                )
            )
            current_file = ""
            in_hunk = False
            continue
        if raw_line.startswith("+++ "):
            candidate = raw_line[4:].strip().split("\t", 1)[0]
            if candidate == "/dev/null":
                current_file = pending_old_file
            else:
                current_file = normalize_relpath(
                    candidate[2:]
                    if candidate.startswith("b/")
                    else candidate
                )
            in_hunk = False
            continue
        if raw_line.startswith("@@"):
            in_hunk = bool(current_file)
            continue
        if not in_hunk or not current_file or not raw_line:
            continue
        prefix = raw_line[0]
        if prefix not in {"+", "-"}:
            continue
        record = by_file.setdefault(
            current_file,
            {"file": current_file, "removed": [], "added": []},
        )
        record["added" if prefix == "+" else "removed"].append(
            normalized_line(raw_line[1:])
        )

    forward = [by_file[path] for path in sorted(by_file)]
    inverse = [
        {
            "file": item["file"],
            "removed": list(item["added"]),
            "added": list(item["removed"]),
        }
        for item in forward
    ]
    return {
        "forward": forward,
        "inverse": inverse,
        "forward_fingerprint": stable_json_sha256(forward),
        "inverse_fingerprint": stable_json_sha256(inverse),
    }


def build_patch_outcome_candidate(
    spec: CaseSpec,
    *,
    patch_signature: str,
    touched_files: Iterable[object],
    root_binding: dict[str, Any],
    source_evidence_fingerprint: str,
    diagnostic_signatures: Iterable[object],
    diff_text: str = "",
) -> dict[str, Any]:
    files = sorted(
        {
            normalize_relpath(str(path))
            for path in touched_files
            if normalize_relpath(str(path))
        }
    )
    preimage = [_file_preimage(spec, path) for path in files]
    normalized_root = {
        "work_unit_id": str(root_binding.get("work_unit_id") or ""),
        "root_node_id": str(root_binding.get("root_node_id") or ""),
        "root_key": str(root_binding.get("root_key") or ""),
    }
    normalized_diagnostics = sorted(
        {str(signature) for signature in diagnostic_signatures if str(signature)}
    )
    preimage_fingerprint = stable_json_sha256(preimage)
    diagnostic_fingerprint = stable_json_sha256(normalized_diagnostics)
    source_delta = _normalized_source_delta(diff_text)
    candidate_id = stable_id(
        "patch-outcome-candidate",
        patch_signature,
        preimage_fingerprint,
        json.dumps(normalized_root, ensure_ascii=True, sort_keys=True),
        diagnostic_fingerprint,
    )
    return {
        "schema": "patch_outcome_candidate_v3",
        "candidate_id": candidate_id,
        "patch_signature": patch_signature,
        "root_binding": normalized_root,
        "touched_files": files,
        "preimage_fingerprint": preimage_fingerprint,
        "preimage": preimage,
        "source_evidence_fingerprint": source_evidence_fingerprint,
        "diagnostic_fingerprint": diagnostic_fingerprint,
        "diagnostic_signatures": normalized_diagnostics,
        "source_delta": source_delta["forward"],
        "inverse_source_delta": source_delta["inverse"],
        "source_delta_fingerprint": source_delta[
            "forward_fingerprint"
        ],
        "inverse_source_delta_fingerprint": source_delta[
            "inverse_fingerprint"
        ],
    }


def rejected_patch_outcome(
    state_root: Path,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    candidate_id = str(candidate.get("candidate_id") or "")
    if not candidate_id:
        return None
    records = list(
        reversed(read_jsonl_records(patch_outcome_ledger_path(state_root)))
    )
    for record in records:
        if str(record.get("candidate_id") or "") != candidate_id:
            continue
        status = str(record.get("validation_status") or "")
        if status in UNPRODUCTIVE_VALIDATION_STATUSES:
            return {**record, "rejection_kind": "repeated_unproductive_patch"}
        return None

    candidate_delta = str(
        candidate.get("source_delta_fingerprint") or ""
    )
    if not candidate_delta:
        return None
    candidate_files = set(candidate.get("touched_files") or [])
    candidate_diagnostics = set(
        candidate.get("diagnostic_signatures") or []
    )
    candidate_root = dict(candidate.get("root_binding") or {})
    for record in records:
        if str(record.get("validation_status") or "") not in (
            PRODUCTIVE_VALIDATION_STATUSES
        ):
            continue
        prior = (
            record.get("candidate")
            if isinstance(record.get("candidate"), dict)
            else {}
        )
        if candidate_delta != str(
            prior.get("inverse_source_delta_fingerprint") or ""
        ):
            continue
        if candidate_files != set(prior.get("touched_files") or []):
            continue
        prior_root = dict(prior.get("root_binding") or {})
        roots_align = bool(
            candidate_root.get("root_key")
            and candidate_root.get("root_key") == prior_root.get("root_key")
        ) or bool(
            candidate_root.get("root_node_id")
            and candidate_root.get("root_node_id")
            == prior_root.get("root_node_id")
        )
        if not roots_align:
            continue
        prior_diagnostics = set(
            prior.get("diagnostic_signatures") or []
        )
        # Missing diagnostic provenance is not authority to suppress a
        # source change, even when its text happens to invert an older delta.
        if not candidate_diagnostics or not prior_diagnostics:
            continue
        if not (candidate_diagnostics & prior_diagnostics):
            continue
        return {
            **record,
            "rejection_kind": "inverse_of_productive_patch",
            "inverse_match": {
                "source_delta_fingerprint": candidate_delta,
                "prior_inverse_source_delta_fingerprint": prior.get(
                    "inverse_source_delta_fingerprint"
                ),
                "aligned_touched_files": sorted(candidate_files),
            },
        }
    return None


def _target_statuses(validation_delta: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("target_work_unit_id") or ""): str(
            item.get("status") or ""
        )
        for item in validation_delta.get("target_results") or []
        if isinstance(item, dict) and item.get("target_work_unit_id")
    }


def record_round_patch_outcomes(
    state_root: Path,
    *,
    repair_round: int,
    applied_units: Iterable[dict[str, Any]],
    validation_delta: dict[str, Any],
    candidate_state: dict[str, Any],
) -> dict[str, Any]:
    path = patch_outcome_ledger_path(state_root)
    status_by_work_unit = _target_statuses(validation_delta)
    global_status = str(validation_delta.get("status") or "")
    retained_work_units = {
        str(item.get("work_unit_id") or "")
        for item in candidate_state.get("retained") or []
        if isinstance(item, dict) and item.get("work_unit_id")
    }
    existing_outcome_ids = {
        str(record.get("outcome_id") or "")
        for record in read_jsonl_records(path)
        if record.get("outcome_id")
    }
    records: list[dict[str, Any]] = []
    for applied_unit in applied_units:
        if not isinstance(applied_unit, dict):
            continue
        work_unit_id = str(applied_unit.get("work_unit_id") or "")
        validation_status = status_by_work_unit.get(work_unit_id)
        if validation_status is None and not status_by_work_unit:
            validation_status = global_status
        disposition = (
            str(candidate_state.get("status") or "retained")
            if work_unit_id in retained_work_units
            else "committed"
        )
        for result in (applied_unit.get("apply_result") or {}).get("results") or []:
            if not isinstance(result, dict):
                continue
            candidate = (
                result.get("patch_outcome_candidate")
                if isinstance(result.get("patch_outcome_candidate"), dict)
                else {}
            )
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                continue
            outcome_id = stable_id(
                "patch-outcome",
                candidate_id,
                str(repair_round),
                str(validation_status or ""),
                disposition,
            )
            if outcome_id in existing_outcome_ids:
                continue
            existing_outcome_ids.add(outcome_id)
            records.append(
                {
                    "schema": "patch_outcome_record_v1",
                    "outcome_id": outcome_id,
                    "candidate_id": candidate_id,
                    "recorded_at": utc_now(),
                    "repair_round": repair_round,
                    "plan_unit_id": applied_unit.get("plan_unit_id"),
                    "work_unit_id": work_unit_id,
                    "validation_status": validation_status,
                    "transaction_disposition": disposition,
                    "patch_signature": result.get("patch_signature"),
                    "patch_path": result.get("patch_path"),
                    "transaction_dir": result.get("transaction_dir"),
                    "candidate": candidate,
                }
            )
    if records:
        append_jsonl_records(path, records)
    return {
        "schema": "patch_outcome_ledger_update_v1",
        "ledger_path": str(path.resolve()),
        "recorded_count": len(records),
        "records": records,
    }
