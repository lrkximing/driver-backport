#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from driver_backport.core.common import normalize_relpath, stable_id


APPLY_FAILURE_SCHEMA = "apply_failure_v1"

_JSON_OBJECT_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

APPLY_FAILURE_STAGES = {
    "response_envelope",
    "unified_diff_structure",
    "target_policy",
    "current_tree_preflight",
    "structural_rebase",
    "provider_contract",
    "transaction_apply",
    "structural_postcondition",
    "semantic_transaction_postcondition",
}

APPLY_FAILURE_RECOVERABILITY = {
    "deterministic_repair",
    "outer_unit_replan",
    "terminal_infrastructure",
}

APPLY_FAILURE_RESULT_STATUSES = frozenset(
    {
        "invalid_response",
        "disallowed_targets",
        "apply_failed",
        "apply_conflict",
    }
)
APPLY_INCOMPLETE_RESULT_STATUSES = (
    APPLY_FAILURE_RESULT_STATUSES
    | {"missing_response", "rerun_flow_required"}
)

class ResponseInterpretationError(ValueError):
    """A response failure whose stage is known at the parser branch."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        finding_kind: str,
        immutable_patch_text: str | None = None,
    ) -> None:
        super().__init__(message)
        if stage not in APPLY_FAILURE_STAGES:
            raise ValueError(f"invalid response interpretation stage: {stage}")
        self.stage = stage
        self.finding_kind = str(finding_kind or "unspecified")
        self.immutable_patch_text = immutable_patch_text


def extract_json_object(
    text: str,
    *,
    required_key: str | None = None,
) -> dict[str, Any] | None:
    """Extract one JSON object without repairing model output."""

    source = str(text or "")
    candidates = [source.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in _JSON_OBJECT_BLOCK_RE.finditer(source)
    )
    decoder = json.JSONDecoder(strict=False)
    for candidate in candidates:
        if not candidate:
            continue
        for offset, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                value, _unused = decoder.raw_decode(candidate[offset:])
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            if required_key and not value.get(required_key):
                continue
            return value
    return None


def immutable_diff_body_digest(diff_text: str) -> str | None:
    """Digest file identity and complete hunk bodies while ignoring ranges."""

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_hunk: list[str] | None = None
    lines = str(diff_text or "").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            line.startswith("--- ")
            and index + 1 < len(lines)
            and lines[index + 1].startswith("+++ ")
        ):
            old_path = line[4:].strip().split("\t", 1)[0]
            new_path = lines[index + 1][4:].strip().split("\t", 1)[0]
            selected_path = new_path if new_path != "/dev/null" else old_path
            if selected_path.startswith(("a/", "b/")):
                selected_path = selected_path[2:]
            current = {
                "file": normalize_relpath(selected_path),
                "hunks": [],
            }
            records.append(current)
            current_hunk = None
            index += 2
            continue
        if current is not None and line.startswith("@@"):
            current_hunk = []
            current["hunks"].append(current_hunk)
            index += 1
            continue
        if current_hunk is not None:
            if line.startswith("diff --git ") or (
                line.startswith("--- ")
                and index + 1 < len(lines)
                and lines[index + 1].startswith("+++ ")
            ):
                current_hunk = None
                continue
            if not line or line[0] not in {" ", "+", "-", "\\"}:
                return None
            current_hunk.append(line)
        index += 1
    if not records or any(
        not record.get("file")
        or not record.get("hunks")
        or any(not hunk for hunk in record.get("hunks") or [])
        for record in records
    ):
        return None
    return hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def patch_delta_summary(lines: list[str]) -> dict[str, Any]:
    added = [
        line[1:]
        for line in lines
        if line.startswith("+") and not line.startswith("+++")
    ]
    removed = [
        line[1:]
        for line in lines
        if line.startswith("-") and not line.startswith("---")
    ]
    return {
        "direction_signature": stable_id(
            "patch-direction",
            *sorted(f"+{line}" for line in added),
            *sorted(f"-{line}" for line in removed),
        ),
        "added_line_count": len(added),
        "removed_line_count": len(removed),
    }


def build_apply_failure(
    *,
    stage: str,
    recoverability: str,
    finding_kind: str,
    unit_id: str,
    work_unit_id: str | None = None,
    root_binding: dict[str, Any] | None = None,
    semantic_action_id: str | None = None,
    source_preimage_digest: str | None = None,
    evidence_set_digest: str | None = None,
    diagnostic_set_digest: str | None = None,
    patch_signature: str | None = None,
    deterministic_attempts: list[dict[str, Any]] | None = None,
    evidence_ids: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage not in APPLY_FAILURE_STAGES:
        raise ValueError(f"unsupported apply failure stage: {stage}")
    if recoverability not in APPLY_FAILURE_RECOVERABILITY:
        raise ValueError(
            f"unsupported apply failure recoverability: {recoverability}"
        )
    normalized_details = dict(details or {})
    identity_payload = {
        "stage": stage,
        "finding_kind": str(finding_kind or "unspecified"),
        "unit_id": str(unit_id or ""),
        "work_unit_id": str(work_unit_id or ""),
        "root_binding": dict(root_binding or {}),
        "semantic_action_id": str(semantic_action_id or ""),
        "source_preimage_digest": str(source_preimage_digest or ""),
        "evidence_set_digest": str(evidence_set_digest or ""),
        "diagnostic_set_digest": str(diagnostic_set_digest or ""),
        "patch_signature": str(patch_signature or ""),
        "details": normalized_details,
    }
    return {
        "schema": APPLY_FAILURE_SCHEMA,
        "failure_id": stable_id(
            "apply-failure",
            json.dumps(identity_payload, ensure_ascii=True, sort_keys=True),
        ),
        "stage": stage,
        "recoverability": recoverability,
        "finding_kind": str(finding_kind or "unspecified"),
        "unit_id": str(unit_id or ""),
        "work_unit_id": str(work_unit_id or unit_id or ""),
        "root_binding": dict(root_binding or {}),
        "semantic_action_id": semantic_action_id,
        "source_preimage_digest": source_preimage_digest,
        "evidence_set_digest": evidence_set_digest,
        "diagnostic_set_digest": diagnostic_set_digest,
        "patch_signature": patch_signature,
        "deterministic_attempts": list(deterministic_attempts or []),
        "evidence_ids": list(dict.fromkeys(evidence_ids or [])),
        "details": normalized_details,
    }


def apply_failure_route(
    failure: dict[str, Any],
) -> dict[str, Any]:
    stage = str(failure.get("stage") or "")
    recoverability = str(failure.get("recoverability") or "")
    finding_kind = str(failure.get("finding_kind") or "")

    if stage not in APPLY_FAILURE_STAGES:
        return {
            "route": "terminal_infrastructure",
            "reason": "apply failure has no recognized source-stage identity",
            "required_transition": "repair_apply_failure_instrumentation",
            "retry_requires_state_change": True,
        }

    if recoverability == "terminal_infrastructure":
        return {
            "route": "terminal_infrastructure",
            "reason": (
                "the deterministic apply/checker infrastructure failed; "
                "another semantic model call cannot repair it"
            ),
            "required_transition": "repair_deterministic_apply_infrastructure",
            "retry_requires_state_change": True,
        }
    if stage == "provider_contract" and finding_kind == "checker_infrastructure_failure":
        return {
            "route": "terminal_infrastructure",
            "reason": "provider checker was unavailable rather than finding a source violation",
            "required_transition": "restore_provider_contract_checker",
            "retry_requires_state_change": True,
        }
    required_transition = {
        "response_envelope": "regenerate_outer_unit_response_envelope",
        "unified_diff_structure": "regenerate_outer_unit_patch_transaction",
        "target_policy": "rebind_patch_to_graph_authorized_write_set",
        "current_tree_preflight": (
            "refresh_current_entity_extents_and_replan_patch"
        ),
        "structural_rebase": (
            "acquire_new_structural_evidence_or_choose_a_distinct_edit"
        ),
        "provider_contract": (
            "repair_or_expand_the_graph_bound_provider_contract"
        ),
        "transaction_apply": "restore_transaction_engine_before_replanning",
        "structural_postcondition": (
            "replan_the_atomic_unit_to_satisfy_structural_postconditions"
        ),
        "semantic_transaction_postcondition": (
            "replan_the_atomic_unit_to_satisfy_incoming_contract_obligations"
        ),
    }[stage]
    return {
        "route": "outer_unit_replan",
        "reason": (
            "the failure requires graph evidence or a distinct semantic action "
            "from the outer work-unit controller"
        ),
        "required_transition": required_transition,
        "retry_requires_state_change": True,
    }


def summarize_apply_failures(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, int] = {}
    by_recoverability: dict[str, int] = {}
    by_finding: dict[str, int] = {}
    unclassified = 0
    for result in results:
        failure = result.get("apply_failure")
        if not isinstance(failure, dict):
            if str(result.get("status") or "") in {
                "invalid_response",
                "disallowed_targets",
                "apply_failed",
                "apply_conflict",
                "rerun_flow_required",
            }:
                unclassified += 1
            continue
        stage = str(failure.get("stage") or "unclassified")
        recoverability = str(
            failure.get("recoverability") or "unclassified"
        )
        finding = str(failure.get("finding_kind") or "unclassified")
        by_stage[stage] = by_stage.get(stage, 0) + 1
        by_recoverability[recoverability] = (
            by_recoverability.get(recoverability, 0) + 1
        )
        by_finding[finding] = by_finding.get(finding, 0) + 1
    return {
        "schema": "apply_failure_metrics_v1",
        "failure_count": sum(by_stage.values()),
        "unclassified_failure_count": unclassified,
        "by_stage": by_stage,
        "by_recoverability": by_recoverability,
        "by_finding_kind": by_finding,
    }


def summarize_apply_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return canonical status counts for every response-apply caller."""

    return {
        "unit_count": len(results),
        "applied_count": sum(
            1 for item in results if item.get("status") == "applied"
        ),
        "already_applied_count": sum(
            1
            for item in results
            if item.get("status")
            in {"already_applied", "already_applied_duplicate"}
        ),
        "duplicate_patch_skipped_count": sum(
            1
            for item in results
            if item.get("status") == "already_applied_duplicate"
        ),
        "no_patch_count": sum(
            1 for item in results if item.get("status") == "no_patch"
        ),
        "needs_escalation_count": sum(
            1
            for item in results
            if item.get("status") == "needs_escalation"
        ),
        "rerun_flow_required_count": sum(
            1
            for item in results
            if item.get("status") == "rerun_flow_required"
        ),
        "incomplete_count": sum(
            1
            for item in results
            if str(item.get("status") or "")
            in APPLY_INCOMPLETE_RESULT_STATUSES
        ),
        "apply_failure_metrics": summarize_apply_failures(results),
    }
