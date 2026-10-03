from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_strings as _ordered_unique_strings,
    stable_id,
    utc_now,
)
from driver_backport.migration.semantic_closure_regions import (
    active_patch_regions_intersecting,
    active_patch_ids_intersecting_changes,
    rebase_active_patch_regions,
)
from driver_backport.migration.owner_key import (
    semantic_observation_owner_key,
    semantic_open_mutation_dimensions,
)
SEMANTIC_CLOSURE_STATE_SCHEMA = "semantic_closure_state_v4"
SEMANTIC_GAP_SNAPSHOT_SCHEMA = "semantic_gap_snapshot_v1"
SEMANTIC_NO_CHANGE_PROOF_SCHEMA = "semantic_no_change_proof_v1"
SEMANTIC_DECISION_EPOCH_SCHEMA = "semantic_decision_epoch_v1"
SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION = (
    "frozen-gap-evidence-bound-v1"
)
SEMANTIC_DECISION_POLICY_REVISION = (
    "target-behavior-proof-manifest-v4"
)
SEMANTIC_COMPILE_RECOVERY_POLICY_REVISION = (
    "semantic_compile_recovery_policy_v3"
)
SEMANTIC_OWNER_BINDING_REVISION = "entity-contract-edge-v2"

PENDING_DECISION = "pending_decision"
CHANGE_REQUIRED = "change_required"
PATCH_PENDING_COMPILE = "patch_pending_compile"
CLOSED_NO_CHANGE = "closed_no_change"
CLOSED_MODIFIED = "closed_modified"

OPEN_GAP_STATUSES = {
    PENDING_DECISION,
    CHANGE_REQUIRED,
    PATCH_PENDING_COMPILE,
}
TERMINAL_GAP_STATUSES = {
    CLOSED_NO_CHANGE,
    CLOSED_MODIFIED,
}
GAP_STATUSES = OPEN_GAP_STATUSES | TERMINAL_GAP_STATUSES

BATCH_OPEN = "open"
BATCH_COMPILE_FAILED = "compile_failed"
BATCH_COMMITTED = "committed"
BATCH_ROLLED_BACK = "rolled_back"
BATCH_STATUSES = {
    BATCH_OPEN,
    BATCH_COMPILE_FAILED,
    BATCH_COMMITTED,
    BATCH_ROLLED_BACK,
}

ROUND_RUNNING = "running"
ROUND_VALIDATED = "validated"
ROUND_COMPILE_FAILED = "compile_failed"
ROUND_NO_PROGRESS = "no_progress"
ROUND_BUDGET_EXHAUSTED = "budget_exhausted"
ROUND_STATUSES = {
    ROUND_RUNNING,
    ROUND_VALIDATED,
    ROUND_COMPILE_FAILED,
    ROUND_NO_PROGRESS,
    ROUND_BUDGET_EXHAUSTED,
}

SEMANTIC_GAP_SWEEP = "semantic_gap_sweep"
CANONICAL_COMPILER_RECOVERY = "canonical_compiler_recovery"
ROUND_EXECUTION_MODES = {
    SEMANTIC_GAP_SWEEP,
    CANONICAL_COMPILER_RECOVERY,
}


def new_semantic_closure_state(
    *,
    case_id: str,
    source_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = {
        "schema": SEMANTIC_CLOSURE_STATE_SCHEMA,
        "case_id": str(case_id),
        "generation": 0,
        "source_analysis": deepcopy(source_analysis or {}),
        "gaps": {},
        "batches": {},
        "active_batch_id": None,
        "rounds": [],
        "pending_repair_sweep": None,
        "repair_resumptions": [],
        "owner_binding_revision": SEMANTIC_OWNER_BINDING_REVISION,
        "history": [],
    }
    validate_semantic_closure_state(state)
    return state


def semantic_gap_id(contract_identity: str) -> str:
    identity = str(contract_identity or "").strip()
    if not identity:
        raise ValueError("semantic gap requires a stable contract identity")
    return stable_id("semantic-gap", identity)


def _descriptor_extent(
    descriptor: object,
) -> tuple[str, int, int] | None:
    if not isinstance(descriptor, dict):
        return None
    relative_path = normalize_relpath(
        str(descriptor.get("file") or descriptor.get("path") or "")
    )
    extent = (
        descriptor.get("extent")
        if isinstance(descriptor.get("extent"), dict)
        else {}
    )
    start = int(extent.get("start_line") or 0)
    end = int(extent.get("end_line") or start)
    if not relative_path or start <= 0 or end < start:
        return None
    return relative_path, start, end


def _changed_range_extent(
    changed: object,
) -> tuple[str, int, int] | None:
    if not isinstance(changed, dict):
        return None
    relative_path = normalize_relpath(str(changed.get("file") or ""))
    start = int(
        changed.get("new_start")
        or changed.get("start_line")
        or 0
    )
    count = int(changed.get("new_count") or 0)
    end = int(
        changed.get("new_end")
        or changed.get("end_line")
        or (start + count - 1 if count > 0 else start)
    )
    if not relative_path or start <= 0 or end < start:
        return None
    return relative_path, start, end


_CONTRACT_IMPACT_OWNER_ROLES = {
    "repair_locus",
    "contract_entity",
    "contract_declaration",
}


def _contract_impact_bindings(
    gap: dict[str, Any],
    *,
    include_integration_consumers: bool = True,
) -> list[dict[str, Any]]:
    """Return exact entities whose edits can change this contract.

    A consumer is part of the contract only for an open integration-reference
    obligation. Other dimensions remain provider-local, so a shared read-only
    consumer cannot make unrelated gaps affect each other.
    """

    roles = set(_CONTRACT_IMPACT_OWNER_ROLES)
    if (
        include_integration_consumers
        and "integration_reference" in semantic_open_mutation_dimensions(gap)
    ):
        roles.add("demand_owner")
    return [
        binding
        for binding in gap.get("owner_bindings") or []
        if isinstance(binding, dict)
        and str(binding.get("role") or "") in roles
    ]


def _binding_identity_keys(binding: dict[str, Any]) -> set[str]:
    descriptor = next(
        (
            value
            for value in (
                binding.get("current_descriptor"),
                binding.get("donor_descriptor"),
            )
            if isinstance(value, dict) and value
        ),
        {},
    )
    relative_path = normalize_relpath(
        str(
            descriptor.get("file")
            or descriptor.get("path")
            or binding.get("file")
            or ""
        )
    )
    kind = str(
        descriptor.get("kind")
        or descriptor.get("entity_kind")
        or binding.get("entity_kind")
        or ""
    )
    contract_key = str(
        descriptor.get("contract_key")
        or descriptor.get("key")
        or binding.get("contract_key")
        or ""
    )
    keys = {str(binding.get("owner_key") or "")}
    if relative_path and kind and contract_key:
        keys.add(
            stable_id(
                "semantic-impact-entity",
                relative_path,
                kind,
                contract_key,
                (
                    "definition"
                    if descriptor.get("is_definition") is True
                    else "declaration"
                    if descriptor.get("is_definition") is False
                    else "entity"
                ),
            )
        )
    return {key for key in keys if key}


def _structurally_affected_gap_ids(
    state: dict[str, Any],
    *,
    primary_gap_ids: Iterable[object],
    changed_ranges: Iterable[dict[str, Any]],
) -> list[str]:
    """Find contracts whose own exact entities intersect surviving edits."""

    primary = set(_ordered_unique_strings(primary_gap_ids))
    changed_range_records = [
        changed
        for changed in changed_ranges
        if isinstance(changed, dict)
    ]
    direct_gap_ids = {
        str(gap_id)
        for changed in changed_range_records
        for gap_id in changed.get("direct_gap_ids") or []
        if str(gap_id)
    }
    direct_owner_keys = {
        str(owner_key)
        for changed in changed_range_records
        for owner_key in changed.get("direct_owner_keys") or []
        if str(owner_key)
    }
    all_bindings = [
        binding
        for gap in (state.get("gaps") or {}).values()
        if isinstance(gap, dict)
        for binding in gap.get("owner_bindings") or []
        if isinstance(binding, dict)
    ]
    direct_owner_identities = {
        identity
        for binding in all_bindings
        if str(binding.get("owner_key") or "") in direct_owner_keys
        for identity in _binding_identity_keys(binding)
    }
    primary_owner_identities = {
        identity
        for gap_id in primary
        for binding in _contract_impact_bindings(
            ((state.get("gaps") or {}).get(gap_id) or {}),
            include_integration_consumers=False,
        )
        for identity in _binding_identity_keys(binding)
    }
    changed_extents = [
        extent
        for changed in changed_range_records
        if (extent := _changed_range_extent(changed)) is not None
    ]
    affected: list[str] = []
    for gap_id, gap in (state.get("gaps") or {}).items():
        if (
            str(gap_id) in primary
            or not isinstance(gap, dict)
        ):
            continue
        bindings = _contract_impact_bindings(gap)
        owner_identities = {
            identity
            for item in bindings
            for identity in _binding_identity_keys(item)
        }
        same_owner = bool(
            primary_owner_identities & owner_identities
        )
        direct_lineage = (
            str(gap_id) in direct_gap_ids
            or bool(direct_owner_identities & owner_identities)
        )
        overlaps_change = False
        for binding in bindings:
            descriptor = (
                binding.get("current_descriptor")
                if isinstance(binding.get("current_descriptor"), dict)
                else binding
            )
            extent = _descriptor_extent(descriptor)
            if extent is None:
                continue
            path, start, end = extent
            if any(
                changed_path == path
                and max(start, changed_start)
                <= min(end, changed_end)
                for changed_path, changed_start, changed_end
                in changed_extents
            ):
                overlaps_change = True
                break
        if (
            direct_lineage
            or same_owner
            or overlaps_change
        ):
            affected.append(str(gap_id))
    return affected


def _semantic_gap_set_fingerprint(
    gaps: dict[str, Any],
) -> str:
    descriptors = [
        "\x1f".join(
            (
                str(gap_id),
                str(gap.get("contract_identity") or ""),
                str(gap.get("contract_evidence_ref") or ""),
            )
        )
        for gap_id, gap in sorted(gaps.items())
        if isinstance(gap, dict)
    ]
    return stable_id("semantic-gap-set", *descriptors)


def _current_contract_evaluation(
    gap: dict[str, Any],
) -> dict[str, Any] | None:
    for field in (
        "post_batch_contract_evaluation",
        "compile_valid_handoff_evaluation",
    ):
        evaluation = gap.get(field)
        if (
            isinstance(evaluation, dict)
            and str(evaluation.get("evaluation_ref") or "").strip()
        ):
            return evaluation
    return None


def _latest_satisfied_no_change_evaluation(
    gap: dict[str, Any],
) -> dict[str, Any] | None:
    evaluation = _current_contract_evaluation(gap)
    return (
        evaluation
        if evaluation is not None
        and evaluation.get("status") == "satisfied"
        else None
    )


def _exact_no_change_proof(
    state: dict[str, Any],
    *,
    gap_id: str,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    snapshot = state.get("semantic_gap_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("semantic no-change proof requires a frozen snapshot")
    required_dimensions = _ordered_unique_strings(
        evaluation.get("required_dimensions")
        or (
            (state["gaps"][gap_id].get(
                "compile_valid_handoff_evaluation"
            ) or {}).get("required_dimensions")
        )
        or []
    )
    return {
        "schema": SEMANTIC_NO_CHANGE_PROOF_SCHEMA,
        "policy_revision": SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION,
        "snapshot_id": str(snapshot.get("id") or ""),
        "gap_id": str(gap_id),
        "basis": "exact_current_tree_satisfied",
        "evaluation_ref": str(evaluation.get("evaluation_ref") or ""),
        "checkpoint_ref": str(evaluation.get("checkpoint_ref") or ""),
        "covered_dimensions": required_dimensions,
        "evidence_ids": _ordered_unique_strings(
            [
                str(
                    state["gaps"][gap_id].get(
                        "contract_evidence_ref"
                    )
                    or ""
                ),
                str(evaluation.get("evaluation_ref") or ""),
            ]
        ),
        "validated_at": utc_now(),
    }


def activate_semantic_gap_snapshot(
    state: dict[str, Any],
    *,
    checkpoint_ref: str,
    evaluator_revision: str,
) -> dict[str, Any]:
    """Freeze the compile-valid gap set and enforce evidence-bound closure."""

    validate_semantic_closure_state(state)
    checkpoint = str(checkpoint_ref or "").strip()
    revision = str(evaluator_revision or "").strip()
    if not checkpoint or not revision:
        raise ValueError(
            "semantic gap snapshot requires checkpoint and evaluator revision"
        )
    gaps = state.get("gaps") or {}
    gap_ids = sorted(str(gap_id) for gap_id in gaps)
    fingerprint = _semantic_gap_set_fingerprint(gaps)
    snapshot = state.get("semantic_gap_snapshot")
    created = False
    if snapshot is None:
        snapshot = {
            "schema": SEMANTIC_GAP_SNAPSHOT_SCHEMA,
            "id": stable_id(
                "semantic-gap-snapshot",
                str(state.get("case_id") or ""),
                checkpoint,
                fingerprint,
            ),
            "checkpoint_ref": checkpoint,
            "evaluator_revision": revision,
            "gap_ids": gap_ids,
            "gap_set_fingerprint": fingerprint,
            "proof_policy_revision": (
                SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION
            ),
            "frozen_at": utc_now(),
        }
        state["semantic_gap_snapshot"] = snapshot
        created = True
    elif not isinstance(snapshot, dict):
        raise ValueError("semantic gap snapshot must be an object")
    elif (
        snapshot.get("schema") != SEMANTIC_GAP_SNAPSHOT_SCHEMA
        or list(snapshot.get("gap_ids") or []) != gap_ids
        or str(snapshot.get("gap_set_fingerprint") or "") != fingerprint
    ):
        raise ValueError("frozen semantic gap set has changed")

    proven_gap_ids: list[str] = []
    reopened_gap_ids: list[str] = []
    policy_reopened_gap_ids: list[str] = []
    for gap_id, gap in gaps.items():
        if (
            not isinstance(gap, dict)
            or gap.get("status") != CHANGE_REQUIRED
        ):
            continue
        decision = gap.get("decision")
        if not (
            isinstance(decision, dict)
            and str(decision.get("work_unit_id") or "")
            and str(decision.get("policy_revision") or "")
            != SEMANTIC_DECISION_POLICY_REVISION
        ):
            continue
        gap["decision"] = None
        gap["status"] = PENDING_DECISION
        reopened_gap_ids.append(str(gap_id))
        policy_reopened_gap_ids.append(str(gap_id))
    for gap_id, gap in gaps.items():
        if not isinstance(gap, dict) or gap.get("status") != CLOSED_NO_CHANGE:
            continue
        decision = gap.get("decision")
        existing_proof = (
            decision.get("no_change_proof")
            if isinstance(decision, dict)
            and isinstance(decision.get("no_change_proof"), dict)
            else {}
        )
        if (
            existing_proof.get("schema")
            == SEMANTIC_NO_CHANGE_PROOF_SCHEMA
            and existing_proof.get("policy_revision")
            == SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION
            and existing_proof.get("snapshot_id") == snapshot.get("id")
        ):
            continue
        evaluation = _latest_satisfied_no_change_evaluation(gap)
        if evaluation is not None:
            gap.setdefault("decision", {})["no_change_proof"] = (
                _exact_no_change_proof(
                    state,
                    gap_id=str(gap_id),
                    evaluation=evaluation,
                )
            )
            proven_gap_ids.append(str(gap_id))
            continue
        gap["decision"] = None
        gap["status"] = PENDING_DECISION
        reopened_gap_ids.append(str(gap_id))

    changed = created or bool(proven_gap_ids) or bool(reopened_gap_ids)
    if changed:
        if reopened_gap_ids:
            existing_epoch = state.get("pending_decision_epoch")
            if existing_epoch is not None:
                raise ValueError(
                    "cannot start proof-policy review while another "
                    "decision epoch is pending"
                )
            decision_epoch_source = (
                "decision_policy_refresh"
                if policy_reopened_gap_ids
                else "compile_valid_proof_review"
            )
            state["pending_decision_epoch"] = {
                "schema": SEMANTIC_DECISION_EPOCH_SCHEMA,
                "id": stable_id(
                    "semantic-decision-epoch",
                    str(snapshot.get("id") or ""),
                    SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION,
                    SEMANTIC_DECISION_POLICY_REVISION,
                    *reopened_gap_ids,
                ),
                "source": decision_epoch_source,
                "snapshot_id": str(snapshot.get("id") or ""),
                "gap_ids": list(reopened_gap_ids),
                "created_at": utc_now(),
            }
            _invalidate_running_gap_sweep_barriers(
                state,
                reopened_gap_ids=reopened_gap_ids,
                reason=decision_epoch_source,
            )
        _touch_state(
            state,
            event="semantic_gap_snapshot_activated",
            payload={
                "snapshot_id": str(snapshot.get("id") or ""),
                "created": created,
                "proven_gap_ids": proven_gap_ids,
                "reopened_gap_ids": reopened_gap_ids,
                "policy_reopened_gap_ids": policy_reopened_gap_ids,
                "decision_policy_revision": (
                    SEMANTIC_DECISION_POLICY_REVISION
                ),
                "proof_policy_revision": (
                    SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION
                ),
            },
        )
    validate_semantic_closure_state(state)
    return {
        "status": "activated" if changed else "unchanged",
        "snapshot_id": str(snapshot.get("id") or ""),
        "proven_gap_ids": proven_gap_ids,
        "reopened_gap_ids": reopened_gap_ids,
    }


def _append_history(record: dict[str, Any], event: dict[str, Any]) -> None:
    history = list(record.get("history") or [])
    history.append({"at": utc_now(), **deepcopy(event)})
    record["history"] = history


def _touch_state(
    state: dict[str, Any],
    *,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    state["generation"] = int(state.get("generation") or 0) + 1
    _append_history(
        state,
        {
            "event": event,
            **deepcopy(payload or {}),
        },
    )


def _invalidate_running_gap_sweep_barriers(
    state: dict[str, Any],
    *,
    reopened_gap_ids: Iterable[object],
    reason: str,
) -> dict[str, Any] | None:
    """Invalidate stale barriers when current evidence reopens decisions."""

    reopened = _ordered_unique_strings(reopened_gap_ids)
    if not reopened:
        return None
    running_sweeps = [
        record
        for record in state.get("rounds") or []
        if isinstance(record, dict)
        and record.get("status") == ROUND_RUNNING
        and record.get("execution_mode") == SEMANTIC_GAP_SWEEP
    ]
    if len(running_sweeps) > 1:
        raise ValueError("multiple semantic gap sweeps are running")
    if not running_sweeps:
        return None
    round_record = running_sweeps[0]
    invalidated_fields = [
        field
        for field in (
            "decision_sweep_complete",
            "decision_snapshot_gap_ids",
            "mutation_sweep_complete",
            "mutation_snapshot_gap_ids",
            "attempted_gap_ids",
            "unresolved_gap_ids",
        )
        if field in round_record
    ]
    for field in invalidated_fields:
        round_record.pop(field, None)
    event = {
        "event": "semantic_gap_sweep_barriers_invalidated",
        "reason": str(reason),
        "reopened_gap_ids": reopened,
        "invalidated_fields": invalidated_fields,
    }
    _touch_state(
        state,
        event="semantic_gap_sweep_barriers_invalidated",
        payload={
            "round": int(round_record.get("number") or 0),
            **{key: value for key, value in event.items() if key != "event"},
        },
    )
    return {
        "round": int(round_record.get("number") or 0),
        "reopened_gap_ids": reopened,
        "invalidated_fields": invalidated_fields,
    }


def reopen_gaps_for_authoritative_noop(
    state: dict[str, Any],
    *,
    gap_ids: Iterable[object],
    evidence_refs: Iterable[object],
    round_number: int,
) -> dict[str, Any]:
    """Return a non-mutating repair result to the decision evidence loop.

    ``already_satisfied`` is not closure proof, but it is new authoritative
    evidence that the mutation tool found no source delta for its exact edit
    handles.  Re-review each such gap once per immutable evidence set instead
    of exhausting the mutation budget on repeated no-ops.
    """

    snapshot = state.get("semantic_gap_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError(
            "authoritative no-op refresh requires a frozen gap snapshot"
        )
    refs = _ordered_unique_strings(evidence_refs)
    if not refs:
        raise ValueError("authoritative no-op refresh requires evidence")
    requested = _ordered_unique_strings(gap_ids)
    refreshed: list[str] = []
    for gap_id in requested:
        gap = (state.get("gaps") or {}).get(gap_id)
        if not isinstance(gap, dict) or gap.get("status") != CHANGE_REQUIRED:
            continue
        prior_refs = _ordered_unique_strings(
            gap.get("authoritative_noop_evidence_refs") or []
        )
        added_refs = [reference for reference in refs if reference not in prior_refs]
        if not added_refs:
            continue
        gap["authoritative_noop_evidence_refs"] = _ordered_unique_strings(
            [*prior_refs, *added_refs]
        )
        gap["superseded_decision"] = deepcopy(gap.get("decision"))
        gap["decision"] = None
        gap["status"] = PENDING_DECISION
        refreshed.append(gap_id)
    if not refreshed:
        return {
            "status": "unchanged",
            "gap_ids": [],
            "evidence_refs": refs,
        }
    if state.get("pending_decision_epoch") is not None:
        raise ValueError(
            "cannot start authoritative no-op review while another "
            "decision epoch is pending"
        )
    state["pending_decision_epoch"] = {
        "schema": SEMANTIC_DECISION_EPOCH_SCHEMA,
        "id": stable_id(
            "semantic-decision-epoch",
            str(snapshot.get("id") or ""),
            "authoritative-noop-refresh",
            *refs,
            *refreshed,
        ),
        "source": "authoritative_noop_refresh",
        "snapshot_id": str(snapshot.get("id") or ""),
        "gap_ids": refreshed,
        "evidence_refs": refs,
        "round": int(round_number),
        "created_at": utc_now(),
    }
    invalidation = _invalidate_running_gap_sweep_barriers(
        state,
        reopened_gap_ids=refreshed,
        reason="authoritative_noop_refresh",
    )
    _touch_state(
        state,
        event="authoritative_noop_decision_refresh",
        payload={
            "round": int(round_number),
            "gap_ids": refreshed,
            "evidence_refs": refs,
        },
    )
    validate_semantic_closure_state(state)
    return {
        "status": "refreshed",
        "gap_ids": refreshed,
        "evidence_refs": refs,
        "barrier_invalidation": invalidation,
    }


def register_difference_gaps(
    state: dict[str, Any],
    *,
    gaps: Iterable[dict[str, Any]],
    discovery_id: str,
    source: str,
) -> dict[str, Any]:
    """Register projected demands for compile-valid current-tree closure.

    A demand may start with a repair delta, a pending compile binding, an
    already-satisfied source comparison, or blocked source evidence. Its
    compile-valid evaluation, rather than the initial disposition, determines
    whether it closes.
    """

    validate_semantic_closure_state(state)
    if state.get("semantic_gap_snapshot") is not None:
        raise ValueError(
            "cannot register or merge semantic gaps after the compile-valid "
            "gap snapshot is frozen"
        )
    added: list[str] = []
    merged: list[str] = []
    for raw_gap in gaps:
        if not isinstance(raw_gap, dict):
            raise ValueError("semantic gap registration requires objects")
        contract_identity = str(
            raw_gap.get("contract_identity")
            or raw_gap.get("contract_key")
            or ""
        ).strip()
        gap_id = str(raw_gap.get("id") or "").strip() or semantic_gap_id(
            contract_identity
        )
        if not contract_identity:
            raise ValueError(f"semantic gap {gap_id} has no contract identity")

        existing = (state.get("gaps") or {}).get(gap_id)
        if existing is None:
            gap = {
                "id": gap_id,
                "contract_identity": contract_identity,
                "status": PENDING_DECISION,
                "raw_hunk_ids": _ordered_unique_strings(
                    raw_gap.get("raw_hunk_ids") or []
                ),
                "source_work_unit_ids": _ordered_unique_strings(
                    raw_gap.get("source_work_unit_ids") or []
                ),
                "owner_bindings": deepcopy(
                    raw_gap.get("owner_bindings") or []
                ),
                "contract_evidence_ref": str(
                    raw_gap.get("contract_evidence_ref") or ""
                )
                or None,
                "decision": None,
                "patch_ids": [],
            }
            state["gaps"][gap_id] = gap
            added.append(gap_id)
            continue

        if str(existing.get("contract_identity") or "") != contract_identity:
            raise ValueError(
                f"semantic gap identity collision for {gap_id}"
            )
        existing["raw_hunk_ids"] = _ordered_unique_strings(
            [
                *(existing.get("raw_hunk_ids") or []),
                *(raw_gap.get("raw_hunk_ids") or []),
            ]
        )
        existing["source_work_unit_ids"] = _ordered_unique_strings(
            [
                *(existing.get("source_work_unit_ids") or []),
                *(raw_gap.get("source_work_unit_ids") or []),
            ]
        )
        existing["owner_bindings"] = _merge_records(
            existing.get("owner_bindings") or [],
            raw_gap.get("owner_bindings") or [],
        )
        incoming_contract_evidence_ref = str(
            raw_gap.get("contract_evidence_ref") or ""
        ).strip()
        existing_contract_evidence_ref = str(
            existing.get("contract_evidence_ref") or ""
        ).strip()
        if (
            incoming_contract_evidence_ref
            and existing_contract_evidence_ref
            and incoming_contract_evidence_ref
            != existing_contract_evidence_ref
        ):
            raise ValueError(
                f"semantic gap {gap_id} has conflicting contract evidence"
            )
        if incoming_contract_evidence_ref:
            existing["contract_evidence_ref"] = (
                incoming_contract_evidence_ref
            )
        merged.append(gap_id)

    if added or merged:
        _touch_state(
            state,
            event="difference_gaps_registered",
            payload={
                "discovery_id": str(discovery_id),
                "added_gap_ids": added,
                "merged_gap_ids": merged,
            },
        )
    validate_semantic_closure_state(state)
    return {
        "added_gap_ids": added,
        "merged_gap_ids": merged,
        "gap_count": len(state["gaps"]),
    }


def normalize_semantic_owner_bindings(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Coalesce reference observations onto their exact mutation relation."""

    validate_semantic_closure_state(state)
    if state.get("owner_binding_revision") == SEMANTIC_OWNER_BINDING_REVISION:
        return {"status": "unchanged", "changed_gap_ids": []}
    changed_gap_ids: list[str] = []
    removed_owner_count = 0
    for gap_id, gap in (state.get("gaps") or {}).items():
        if not isinstance(gap, dict):
            continue
        bindings: dict[str, dict[str, Any]] = {}
        for raw in gap.get("owner_bindings") or []:
            if not isinstance(raw, dict):
                continue
            binding = deepcopy(raw)
            revised_key = semantic_observation_owner_key(binding)
            if revised_key:
                binding["owner_key"] = revised_key
            owner_key = str(binding.get("owner_key") or "")
            if not owner_key:
                continue
            existing = bindings.get(owner_key)
            if existing is None:
                bindings[owner_key] = binding
                continue
            existing["exact_node_refs"] = _ordered_unique_strings(
                [
                    *list(existing.get("exact_node_refs") or []),
                    *list(binding.get("exact_node_refs") or []),
                ]
            )
            existing["semantic_gap_ids"] = _ordered_unique_strings(
                [
                    *list(existing.get("semantic_gap_ids") or []),
                    *list(binding.get("semantic_gap_ids") or []),
                ]
            )
        previous_count = len(gap.get("owner_bindings") or [])
        normalized = list(bindings.values())
        if normalized != list(gap.get("owner_bindings") or []):
            gap["owner_bindings"] = normalized
            changed_gap_ids.append(str(gap_id))
            removed_owner_count += max(0, previous_count - len(normalized))
    state["owner_binding_revision"] = SEMANTIC_OWNER_BINDING_REVISION
    _touch_state(
        state,
        event="semantic_owner_bindings_normalized",
        payload={
            "revision": SEMANTIC_OWNER_BINDING_REVISION,
            "changed_gap_ids": changed_gap_ids,
            "removed_owner_count": removed_owner_count,
        },
    )
    validate_semantic_closure_state(state)
    return {
        "status": "normalized",
        "changed_gap_ids": changed_gap_ids,
        "removed_owner_count": removed_owner_count,
    }


def record_initial_unit_outcomes_snapshot(
    state: dict[str, Any],
    *,
    record_ref: str,
    unit_count: int,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the immutable final B0 outcome snapshot exactly once."""

    validate_semantic_closure_state(state)
    normalized_ref = str(record_ref or "").strip()
    if not normalized_ref:
        raise ValueError("initial unit outcomes snapshot needs a record ref")
    existing_ref = str(state.get("initial_unit_outcomes_ref") or "")
    if existing_ref:
        if existing_ref != normalized_ref:
            raise ValueError(
                "semantic initial unit outcomes snapshot is immutable"
            )
        return {
            "status": "already_recorded",
            "record_ref": existing_ref,
            "unit_count": int(unit_count),
        }
    state["initial_unit_outcomes_ref"] = normalized_ref
    _touch_state(
        state,
        event="initial_unit_outcomes_snapshotted",
        payload={
            "record_ref": normalized_ref,
            "unit_count": int(unit_count),
            "summary": deepcopy(summary or {}),
        },
    )
    return {
        "status": "recorded",
        "record_ref": normalized_ref,
        "unit_count": int(unit_count),
    }


def _record_key(record: object) -> str:
    if not isinstance(record, dict):
        return repr(record)
    for key in (
        "id",
        "contract_identity",
        "contract_key",
        "owner_key",
        "entity_id",
    ):
        value = str(record.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return repr(sorted(record.items()))


def _merge_records(
    first: Iterable[object],
    second: Iterable[object],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in [*list(first), *list(second)]:
        if not isinstance(record, dict):
            continue
        merged.setdefault(_record_key(record), deepcopy(record))
    return list(merged.values())


def _validate_no_change_proof(
    state: dict[str, Any],
    *,
    gap_id: str,
    proof: object,
) -> dict[str, Any] | None:
    snapshot = state.get("semantic_gap_snapshot")
    if not isinstance(snapshot, dict) or snapshot.get(
        "proof_policy_revision"
    ) != SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION:
        return None
    if not isinstance(proof, dict):
        raise ValueError(
            f"no-change decision requires a validated proof: {gap_id}"
        )
    required_fields = {
        "schema",
        "policy_revision",
        "snapshot_id",
        "gap_id",
        "basis",
        "evaluation_ref",
        "covered_dimensions",
        "evidence_ids",
        "validated_at",
    }
    missing = sorted(required_fields - set(proof))
    if missing:
        raise ValueError(
            f"no-change proof is incomplete for {gap_id}: {missing}"
        )
    if (
        proof.get("schema") != SEMANTIC_NO_CHANGE_PROOF_SCHEMA
        or proof.get("policy_revision")
        != SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION
        or str(proof.get("snapshot_id") or "")
        != str(snapshot.get("id") or "")
        or str(proof.get("gap_id") or "") != str(gap_id)
    ):
        raise ValueError(
            f"no-change proof does not belong to the frozen gap: {gap_id}"
        )
    gap = state["gaps"][gap_id]
    evaluation = _current_contract_evaluation(gap) or {}
    evaluation_ref = str(proof.get("evaluation_ref") or "")
    current_evaluation_ref = str(
        evaluation.get("evaluation_ref") or ""
    )
    if not evaluation_ref or evaluation_ref != current_evaluation_ref:
        raise ValueError(
            f"no-change proof uses stale evaluation evidence: {gap_id}"
        )
    evidence_ids = _ordered_unique_strings(proof.get("evidence_ids") or [])
    mandatory_evidence = {
        str(gap.get("contract_evidence_ref") or ""),
        evaluation_ref,
    } - {""}
    if not mandatory_evidence <= set(evidence_ids):
        raise ValueError(
            f"no-change proof lacks contract/evaluation evidence: {gap_id}"
        )
    basis = str(proof.get("basis") or "")
    covered_dimensions = set(
        _ordered_unique_strings(proof.get("covered_dimensions") or [])
    )
    required_dimensions = set(
        _ordered_unique_strings(evaluation.get("required_dimensions") or [])
    )
    open_dimensions = set(
        _ordered_unique_strings(
            [
                *(evaluation.get("unresolved_dimensions") or []),
                *(evaluation.get("unavailable_dimensions") or []),
            ]
        )
    )
    if basis == "exact_current_tree_satisfied":
        current = _latest_satisfied_no_change_evaluation(gap)
        if current is None or str(current.get("evaluation_ref") or "") != (
            evaluation_ref
        ):
            raise ValueError(
                f"exact no-change proof is not evaluator-backed: {gap_id}"
            )
        if covered_dimensions != required_dimensions:
            raise ValueError(
                f"exact no-change proof has incomplete dimensions: {gap_id}"
            )
    elif basis == "evidence_bound_target_judgment":
        if evaluation_ref != current_evaluation_ref:
            raise ValueError(
                f"semantic no-change proof must bind the current evaluation: "
                f"{gap_id}"
            )
        if (
            not open_dimensions
            or not open_dimensions <= covered_dimensions
            or not covered_dimensions <= required_dimensions
        ):
            raise ValueError(
                f"semantic no-change proof does not cover open dimensions: "
                f"{gap_id}"
            )
        if proof.get("validation_status") != "validated":
            raise ValueError(
                f"semantic no-change proof was not validated: {gap_id}"
            )
    else:
        raise ValueError(
            f"unsupported no-change proof basis for {gap_id}: {basis}"
        )
    return deepcopy(proof)


def record_gap_decisions(
    state: dict[str, Any],
    *,
    work_unit_id: str,
    expected_gap_ids: Iterable[object],
    decisions: Iterable[dict[str, Any]],
    round_number: int,
) -> dict[str, Any]:
    """Record one explicit decision for every gap in a review unit."""

    validate_semantic_closure_state(state)
    expected = _ordered_unique_strings(expected_gap_ids)
    decision_by_gap: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("semantic decisions must be objects")
        gap_id = str(decision.get("gap_id") or "").strip()
        if not gap_id:
            raise ValueError("semantic decision has no gap_id")
        if gap_id in decision_by_gap:
            raise ValueError(f"duplicate semantic decision for {gap_id}")
        decision_by_gap[gap_id] = dict(decision)
    if set(decision_by_gap) != set(expected):
        missing = sorted(set(expected) - set(decision_by_gap))
        unexpected = sorted(set(decision_by_gap) - set(expected))
        raise ValueError(
            "semantic decision set must exactly match the work unit: "
            f"missing={missing} unexpected={unexpected}"
        )

    prepared: dict[str, dict[str, Any]] = {}
    for gap_id in expected:
        gap = state["gaps"].get(gap_id)
        if not isinstance(gap, dict):
            raise ValueError(f"unknown semantic gap: {gap_id}")
        status = str(gap.get("status") or "")
        if status in TERMINAL_GAP_STATUSES:
            raise ValueError(
                f"terminal semantic gap cannot be decided again: {gap_id}"
            )
        if status == PATCH_PENDING_COMPILE:
            raise ValueError(
                f"compile-pending semantic gap cannot be reviewed: {gap_id}"
            )
        decision = decision_by_gap[gap_id]
        disposition = str(decision.get("disposition") or "").strip()
        if disposition not in {
            "no_change_required",
            "change_required",
        }:
            raise ValueError(
                f"unsupported semantic disposition for {gap_id}: "
                f"{disposition or '<missing>'}"
            )
        rationale = str(decision.get("rationale") or "").strip()
        if not rationale:
            raise ValueError(
                f"semantic decision for {gap_id} requires a rationale"
            )
        validated_proof = (
            _validate_no_change_proof(
                state,
                gap_id=gap_id,
                proof=decision.get("no_change_proof"),
            )
            if disposition == "no_change_required"
            else None
        )
        prepared[gap_id] = {
            "decision": decision,
            "disposition": disposition,
            "rationale": rationale,
            "validated_proof": validated_proof,
        }

    no_change: list[str] = []
    change_required: list[str] = []
    for gap_id in expected:
        gap = state["gaps"][gap_id]
        item = prepared[gap_id]
        decision = item["decision"]
        disposition = str(item["disposition"])
        gap["decision"] = {
            "disposition": disposition,
            "rationale": str(item["rationale"]),
            "evidence_refs": _ordered_unique_strings(
                decision.get("evidence_refs") or []
            ),
            "work_unit_id": str(work_unit_id),
            "policy_revision": SEMANTIC_DECISION_POLICY_REVISION,
            "round": int(round_number),
            "at": utc_now(),
        }
        if disposition == "no_change_required":
            validated_proof = item["validated_proof"]
            if validated_proof is not None:
                gap["decision"]["no_change_proof"] = validated_proof
            gap["status"] = CLOSED_NO_CHANGE
            no_change.append(gap_id)
        else:
            gap["status"] = CHANGE_REQUIRED
            change_required.append(gap_id)
    decision_epoch = state.get("pending_decision_epoch")
    if isinstance(decision_epoch, dict):
        remaining_epoch_gap_ids = [
            gap_id
            for gap_id in _ordered_unique_strings(
                decision_epoch.get("gap_ids") or []
            )
            if ((state.get("gaps") or {}).get(gap_id) or {}).get("status")
            == PENDING_DECISION
        ]
        if remaining_epoch_gap_ids:
            decision_epoch["gap_ids"] = remaining_epoch_gap_ids
        else:
            state["pending_decision_epoch"] = None

    _touch_state(
        state,
        event="gap_decisions_recorded",
        payload={
            "work_unit_id": str(work_unit_id),
            "round": int(round_number),
            "no_change_gap_ids": no_change,
            "change_required_gap_ids": change_required,
        },
    )
    validate_semantic_closure_state(state)
    return {
        "no_change_gap_ids": no_change,
        "change_required_gap_ids": change_required,
    }


def record_gap_review_failure(
    state: dict[str, Any],
    *,
    work_unit_id: str,
    gap_ids: Iterable[object],
    evidence_ref: str,
    error: str,
    round_number: int,
) -> dict[str, Any]:
    """Keep a failed review observable without inventing a decision."""

    validate_semantic_closure_state(state)
    selected_gap_ids = _ordered_unique_strings(gap_ids)
    reference = str(evidence_ref or "").strip()
    message = str(error or "").strip()
    if not selected_gap_ids or not reference or not message:
        raise ValueError(
            "semantic review failure requires gaps, evidence, and error"
        )
    for gap_id in selected_gap_ids:
        gap = state["gaps"].get(gap_id)
        if not isinstance(gap, dict):
            raise ValueError(f"unknown semantic gap: {gap_id}")
        if gap.get("status") != PENDING_DECISION:
            raise ValueError(
                "review failure can only be attached to pending gaps: "
                f"{gap_id}={gap.get('status')}"
            )
        failure = {
            "work_unit_id": str(work_unit_id),
            "round": int(round_number),
            "evidence_ref": reference,
            "error": message,
            "at": utc_now(),
        }
        gap.setdefault("review_failures", []).append(failure)
    _touch_state(
        state,
        event="gap_review_failed",
        payload={
            "work_unit_id": str(work_unit_id),
            "round": int(round_number),
            "gap_ids": selected_gap_ids,
            "evidence_ref": reference,
            "error": message,
        },
    )
    validate_semantic_closure_state(state)
    return {
        "status": "recorded",
        "gap_ids": selected_gap_ids,
        "evidence_ref": reference,
    }


def record_exact_owner_expansion(
    state: dict[str, Any],
    *,
    work_unit_id: str,
    gap_ids: Iterable[object],
    owner_bindings: Iterable[dict[str, Any]],
    evidence_ref: str,
    round_number: int,
    superseded_owner_keys: Iterable[object] = (),
) -> dict[str, Any]:
    """Attach only statically proven direct dependency owners to open gaps."""

    validate_semantic_closure_state(state)
    selected_gap_ids = _ordered_unique_strings(gap_ids)
    if not selected_gap_ids:
        raise ValueError("semantic owner expansion requires exact gap ids")
    reference = str(evidence_ref or "").strip()
    if not reference:
        raise ValueError("semantic owner expansion requires evidence")
    superseded_keys = set(
        _ordered_unique_strings(superseded_owner_keys)
    )
    normalized_bindings: list[dict[str, Any]] = []
    seen_owner_keys: set[str] = set()
    for raw in owner_bindings:
        if not isinstance(raw, dict):
            raise ValueError("semantic owner binding must be an object")
        binding = deepcopy(raw)
        owner_key = str(binding.get("owner_key") or "").strip()
        relative_path = str(binding.get("file") or "").strip()
        authority = str(binding.get("authority") or "").strip()
        descriptor = (
            binding.get("current_descriptor")
            if isinstance(binding.get("current_descriptor"), dict)
            else {}
        )
        donor_descriptor = (
            binding.get("donor_descriptor")
            if isinstance(binding.get("donor_descriptor"), dict)
            else {}
        )
        exact_node_refs = _ordered_unique_strings(
            binding.get("exact_node_refs") or []
        )
        if (
            not owner_key
            or owner_key in seen_owner_keys
            or not relative_path
            or authority
            not in {
                "typed_direct_dependency_edge",
                "typed_diagnostic_provider_resolution",
                "typed_diagnostic_consumer_resolution",
            }
            or not exact_node_refs
            or not (descriptor or donor_descriptor)
        ):
            raise ValueError(
                "semantic owner expansion requires a unique owner key, "
                "file, exact typed edge or diagnostic-provider relation, "
                "and entity descriptor"
            )
        seen_owner_keys.add(owner_key)
        binding["exact_node_refs"] = exact_node_refs
        normalized_bindings.append(binding)

    def descriptor_identity(
        descriptor: object,
    ) -> tuple[object, ...] | None:
        if not isinstance(descriptor, dict) or not descriptor:
            return None
        container = (
            descriptor.get("container")
            if isinstance(descriptor.get("container"), dict)
            else {}
        )
        relative_path = normalize_relpath(
            str(descriptor.get("file") or descriptor.get("path") or "")
        )
        entity_kind = str(
            descriptor.get("kind")
            or descriptor.get("entity_kind")
            or ""
        ).strip()
        contract_key = str(
            descriptor.get("contract_key")
            or descriptor.get("key")
            or ""
        ).strip()
        symbol = str(
            descriptor.get("symbol") or descriptor.get("name") or ""
        ).strip()
        is_definition = descriptor.get("is_definition")
        definition_kind = (
            "definition"
            if is_definition is True
            else "declaration"
            if is_definition is False
            else "entity"
        )
        identity = (
            relative_path,
            entity_kind,
            contract_key,
            symbol,
            str(container.get("container_kind") or "").strip(),
            str(container.get("container_name") or "").strip(),
            definition_kind,
        )
        return identity if any(identity[:-1]) else None

    def normalized_binding_field(
        binding: dict[str, Any],
        field: str,
    ) -> str:
        value = str(binding.get(field) or "").strip()
        return normalize_relpath(value) if field == "file" else value

    def merge_same_owner_binding(
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge new provenance for one stable owner, rejecting real drift."""

        identity_fields = (
            "owner_key",
            "role",
            "authority",
            "file",
            "contract_key",
            "symbol",
            "entity_kind",
            "edit_permission",
        )
        for field in identity_fields:
            previous = normalized_binding_field(existing, field)
            current = normalized_binding_field(incoming, field)
            if previous and current and previous != current:
                raise ValueError(
                    "semantic owner key maps to conflicting bindings: "
                    f"{incoming.get('owner_key')} ({field})"
                )
        for field in ("current_descriptor", "donor_descriptor"):
            previous_identity = descriptor_identity(existing.get(field))
            current_identity = descriptor_identity(incoming.get(field))
            if (
                previous_identity is not None
                and current_identity is not None
                and previous_identity != current_identity
            ):
                raise ValueError(
                    "semantic owner key maps to conflicting bindings: "
                    f"{incoming.get('owner_key')} ({field})"
                )

        merged = deepcopy(existing)
        for key, value in incoming.items():
            if key in {"exact_node_refs", "semantic_gap_ids"}:
                continue
            if value not in (None, "", [], {}):
                if key in {"current_descriptor", "donor_descriptor"}:
                    # Keep entity locations current while the stable contract
                    # identity above prevents an unrelated owner replacement.
                    merged[key] = deepcopy(value)
                elif merged.get(key) in (None, "", [], {}):
                    merged[key] = deepcopy(value)
        merged["exact_node_refs"] = _ordered_unique_strings(
            [
                *(existing.get("exact_node_refs") or []),
                *(incoming.get("exact_node_refs") or []),
            ]
        )
        merged["semantic_gap_ids"] = _ordered_unique_strings(
            [
                *(existing.get("semantic_gap_ids") or []),
                *(incoming.get("semantic_gap_ids") or []),
            ]
        )
        return merged

    attached: dict[str, list[str]] = {}
    merged_by_gap: dict[str, list[str]] = {}
    removed_by_gap: dict[str, list[str]] = {}
    staged_gaps: dict[str, dict[str, Any]] = {}
    for gap_id in selected_gap_ids:
        stored_gap = state["gaps"].get(gap_id)
        if not isinstance(stored_gap, dict):
            raise ValueError(f"unknown semantic gap: {gap_id}")
        if stored_gap.get("status") in TERMINAL_GAP_STATUSES:
            raise ValueError(
                f"terminal semantic gap cannot expand scope: {gap_id}"
            )
        gap = deepcopy(stored_gap)
        removed_owner_keys = sorted(
            {
                str(item.get("owner_key") or "")
                for item in gap.get("owner_bindings") or []
                if isinstance(item, dict)
                and str(item.get("owner_key") or "")
                in superseded_keys
            }
        )
        if removed_owner_keys:
            gap["owner_bindings"] = [
                item
                for item in gap.get("owner_bindings") or []
                if not isinstance(item, dict)
                or str(item.get("owner_key") or "")
                not in superseded_keys
            ]
        existing_by_key = {
            str(item.get("owner_key") or ""): item
            for item in gap.get("owner_bindings") or []
            if isinstance(item, dict)
            and str(item.get("owner_key") or "")
        }
        added: list[str] = []
        merged_owner_keys: list[str] = []
        for binding in normalized_bindings:
            owner_key = str(binding["owner_key"])
            existing = existing_by_key.get(owner_key)
            if existing is not None:
                merged = merge_same_owner_binding(existing, binding)
                if merged != existing:
                    existing.clear()
                    existing.update(merged)
                    merged_owner_keys.append(owner_key)
                continue
            gap.setdefault("owner_bindings", []).append(
                deepcopy(binding)
            )
            existing_by_key[owner_key] = binding
            added.append(owner_key)
        attached[gap_id] = added
        merged_by_gap[gap_id] = merged_owner_keys
        removed_by_gap[gap_id] = removed_owner_keys
        staged_gaps[gap_id] = gap
    for gap_id, gap in staged_gaps.items():
        state["gaps"][gap_id] = gap
    _touch_state(
        state,
        event="exact_owner_scope_expanded",
        payload={
            "work_unit_id": str(work_unit_id),
            "round": int(round_number),
            "gap_ids": selected_gap_ids,
            "owner_keys": sorted(seen_owner_keys),
            "attached_owner_keys_by_gap": deepcopy(attached),
            "merged_owner_keys_by_gap": deepcopy(merged_by_gap),
            "superseded_owner_keys_by_gap": deepcopy(removed_by_gap),
            "evidence_ref": reference,
        },
    )
    validate_semantic_closure_state(state)
    return {
        "status": "recorded",
        "gap_ids": selected_gap_ids,
        "owner_keys": sorted(seen_owner_keys),
        "attached_owner_keys_by_gap": attached,
        "merged_owner_keys_by_gap": merged_by_gap,
        "superseded_owner_keys_by_gap": removed_by_gap,
    }


def record_compile_valid_handoff_evaluations(
    state: dict[str, Any],
    *,
    evaluations: Iterable[dict[str, Any]],
    checkpoint_ref: str,
    refresh: bool = False,
) -> dict[str, Any]:
    """Record one exact current-tree evaluation for each pending contract.

    The handoff is observational.  A proven-satisfied contract closes without
    model review; a proven difference or unavailable evidence remains pending.
    Neither outcome is allowed to invent a change-required decision.
    """

    validate_semantic_closure_state(state)
    if not str(checkpoint_ref or "").strip():
        raise ValueError("compile-valid handoff requires a checkpoint ref")
    refreshing = bool(refresh)
    if refreshing:
        if state.get("active_batch_id"):
            raise ValueError(
                "compile-valid refresh cannot inspect an unvalidated batch"
            )
        if not isinstance(state.get("semantic_gap_snapshot"), dict) or not (
            isinstance(state.get("compile_valid_semantic_handoff"), dict)
        ):
            raise ValueError(
                "compile-valid refresh requires an existing frozen handoff"
            )

    closed: list[str] = []
    unresolved: list[str] = []
    unavailable: list[str] = []
    superseded_decisions: list[str] = []
    seen: set[str] = set()
    for raw_evaluation in evaluations:
        if not isinstance(raw_evaluation, dict):
            raise ValueError("static semantic evaluation must be an object")
        gap_id = str(raw_evaluation.get("gap_id") or "").strip()
        if not gap_id or gap_id in seen:
            raise ValueError(
                f"compile-valid handoff has an invalid duplicate gap: {gap_id}"
            )
        seen.add(gap_id)
        gap = state["gaps"].get(gap_id)
        if not isinstance(gap, dict):
            raise ValueError(f"unknown semantic gap: {gap_id}")
        status = str(raw_evaluation.get("status") or "")
        if status not in {
            "satisfied",
            "satisfied_modified",
            "unresolved",
            "evidence_unavailable",
        }:
            raise ValueError(
                f"unsupported compile-valid evaluation status: {status}"
            )
        if refreshing:
            prior_decision = gap.get("decision")
            if isinstance(prior_decision, dict):
                superseded_decisions.append(gap_id)
            gap.pop("post_batch_contract_evaluation", None)
            gap["status"] = PENDING_DECISION
            gap["decision"] = None
        else:
            if gap.get("status") in TERMINAL_GAP_STATUSES:
                raise ValueError(
                    "compile-valid handoff cannot reevaluate a terminal "
                    f"contract: {gap_id}"
                )
            if gap.get("status") != PENDING_DECISION:
                raise ValueError(
                    "compile-valid handoff cannot override a semantic "
                    f"decision: {gap_id}={gap.get('status')}"
                )
        evidence_ref = str(
            raw_evaluation.get("evaluation_ref") or ""
        ).strip()
        if not evidence_ref:
            raise ValueError(
                f"compile-valid evaluation has no evidence ref: {gap_id}"
            )
        gap["compile_valid_handoff_evaluation"] = {
            "status": status,
            "checkpoint_ref": str(checkpoint_ref),
            "evaluation_ref": evidence_ref,
            "required_dimensions": _ordered_unique_strings(
                raw_evaluation.get("required_dimensions") or []
            ),
            "relations": deepcopy(raw_evaluation.get("relations") or {}),
            "unresolved_dimensions": _ordered_unique_strings(
                raw_evaluation.get("unresolved_dimensions") or []
            ),
            "unavailable_dimensions": _ordered_unique_strings(
                raw_evaluation.get("unavailable_dimensions") or []
            ),
            "at": utc_now(),
        }
        if status == "unresolved":
            unresolved.append(gap_id)
            continue
        if status == "evidence_unavailable":
            unavailable.append(gap_id)
            continue
        modified = status == "satisfied_modified"
        gap["status"] = (
            CLOSED_MODIFIED if modified else CLOSED_NO_CHANGE
        )
        gap["decision"] = {
            "disposition": (
                "modified" if modified else "no_change_required"
            ),
            "rationale": (
                "the accepted initial contract change remains represented "
                "in the compile-valid current tree"
                if modified
                else (
                    "the compile-valid current tree satisfies every required "
                    "dimension of this exact contract"
                )
            ),
            "authority": "exact_current_tree_contract_evaluation",
            "checkpoint_ref": str(checkpoint_ref),
            "evidence_refs": [evidence_ref],
            "at": utc_now(),
        }
        if not modified and isinstance(
            state.get("semantic_gap_snapshot"),
            dict,
        ):
            gap["decision"]["no_change_proof"] = _exact_no_change_proof(
                state,
                gap_id=gap_id,
                evaluation=gap["compile_valid_handoff_evaluation"],
            )
        closed.append(gap_id)
    if refreshing:
        decision_epoch = state.get("pending_decision_epoch")
        remaining: list[str] = []
        if isinstance(decision_epoch, dict):
            remaining = [
                gap_id
                for gap_id in _ordered_unique_strings(
                    decision_epoch.get("gap_ids") or []
                )
                if gap_id not in seen
            ]
            if remaining:
                decision_epoch["gap_ids"] = remaining
            else:
                state["pending_decision_epoch"] = None
        handoff = state["compile_valid_semantic_handoff"]
        snapshot = state["semantic_gap_snapshot"]
        previous_checkpoint = str(handoff.get("checkpoint_ref") or "")
        refreshed_pending = _ordered_unique_strings(
            [*unresolved, *unavailable]
        )
        epoch_gap_ids = _ordered_unique_strings(
            [*remaining, *refreshed_pending]
        )
        if epoch_gap_ids:
            state["pending_decision_epoch"] = {
                "schema": SEMANTIC_DECISION_EPOCH_SCHEMA,
                "id": stable_id(
                    "semantic-decision-epoch",
                    str(snapshot.get("id") or ""),
                    "compile-valid-checkpoint-refresh",
                    str(checkpoint_ref),
                    *epoch_gap_ids,
                ),
                "source": "compile_valid_checkpoint_refresh",
                "snapshot_id": str(snapshot.get("id") or ""),
                "checkpoint_ref": str(checkpoint_ref),
                "gap_ids": epoch_gap_ids,
                "created_at": utc_now(),
            }
        _invalidate_running_gap_sweep_barriers(
            state,
            reopened_gap_ids=refreshed_pending,
            reason="compile_valid_checkpoint_refresh",
        )
        handoff["checkpoint_ref"] = str(checkpoint_ref)
        snapshot["checkpoint_ref"] = str(checkpoint_ref)
    if seen or refreshing:
        _touch_state(
            state,
            event=(
                "compile_valid_handoff_refreshed"
                if refreshing
                else "compile_valid_handoff_evaluated"
            ),
            payload={
                "checkpoint_ref": str(checkpoint_ref),
                **(
                    {"from_checkpoint_ref": previous_checkpoint}
                    if refreshing
                    else {}
                ),
                "evaluated_gap_ids": sorted(seen),
                "closed_gap_ids": closed,
                "unresolved_gap_ids": unresolved,
                "evidence_unavailable_gap_ids": unavailable,
                "superseded_decision_gap_ids": superseded_decisions,
            },
        )
    validate_semantic_closure_state(state)
    return {
        "status": "refreshed" if refreshing else "recorded",
        "evaluated_gap_ids": sorted(seen),
        "closed_gap_ids": closed,
        "unresolved_gap_ids": unresolved,
        "evidence_unavailable_gap_ids": unavailable,
        "superseded_decision_gap_ids": superseded_decisions,
    }


def begin_or_resume_patch_batch(
    state: dict[str, Any],
    *,
    round_number: int,
    preimage_ref: str,
) -> dict[str, Any]:
    """Return the one uncommitted semantic source batch.

    A compile failure keeps this batch active for the next semantic round.
    A replan or a new execution directory cannot create another batch.
    """

    validate_semantic_closure_state(state)
    active_id = str(state.get("active_batch_id") or "")
    if active_id:
        batch = state["batches"].get(active_id)
        if not isinstance(batch, dict):
            raise ValueError("active semantic batch is missing")
        rounds = list(batch.get("rounds") or [])
        if int(round_number) not in rounds:
            rounds.append(int(round_number))
            batch["rounds"] = rounds
            _touch_state(
                state,
                event="patch_batch_resumed",
                payload={
                    "batch_id": active_id,
                    "round": int(round_number),
                },
            )
        validate_semantic_closure_state(state)
        return batch

    batch_id = stable_id(
        "semantic-batch",
        state.get("case_id"),
        str(int(round_number)),
        str(preimage_ref),
    )
    if batch_id in state["batches"]:
        raise ValueError(f"semantic batch id already exists: {batch_id}")
    batch = {
        "id": batch_id,
        "status": BATCH_OPEN,
        "preimage_ref": str(preimage_ref),
        "rounds": [int(round_number)],
        "patches": [],
        "primary_gap_ids": [],
        "affected_gap_ids": [],
        "compile_validations": [],
    }
    state["batches"][batch_id] = batch
    state["active_batch_id"] = batch_id
    _touch_state(
        state,
        event="patch_batch_started",
        payload={
            "batch_id": batch_id,
            "round": int(round_number),
        },
    )
    validate_semantic_closure_state(state)
    return batch


def record_batch_patch(
    state: dict[str, Any],
    *,
    batch_id: str,
    patch_id: str,
    patch_signature: str,
    gap_ids: Iterable[object],
    touched_files: Iterable[object],
    work_unit_id: str,
    kind: str = "semantic_repair",
    authorized_changed_ranges: Iterable[dict[str, Any]] = (),
    boundary_ref: str | None = None,
    patch_evidence_ref: str | None = None,
    transaction_ref: str | None = None,
    depends_on_patch_ids: Iterable[object] = (),
    provider_obligations: Iterable[dict[str, Any]] = (),
    structural_contract_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_semantic_closure_state(state)
    if str(state.get("active_batch_id") or "") != str(batch_id):
        raise ValueError("patch must belong to the active semantic batch")
    batch = state["batches"].get(str(batch_id))
    if not isinstance(batch, dict):
        raise ValueError(f"unknown semantic batch: {batch_id}")
    if batch.get("status") not in {BATCH_OPEN, BATCH_COMPILE_FAILED}:
        raise ValueError("cannot append to a terminal semantic batch")
    claimed_gap_ids = _ordered_unique_strings(gap_ids)
    if kind == "semantic_repair" and not claimed_gap_ids:
        raise ValueError("semantic repair patch requires exact gap ids")
    if kind not in {"semantic_repair", "compile_recovery"}:
        raise ValueError(f"unsupported semantic patch kind: {kind}")

    signature = str(patch_signature or "").strip()
    if not signature:
        raise ValueError("semantic patch requires a stable signature")
    for existing in batch.get("patches") or []:
        if str(existing.get("patch_signature") or "") == signature:
            return {
                "status": "duplicate",
                "batch_id": str(batch_id),
                "patch_id": str(existing.get("id") or ""),
            }

    for gap_id in claimed_gap_ids:
        gap = state["gaps"].get(gap_id)
        if not isinstance(gap, dict):
            raise ValueError(f"unknown semantic gap: {gap_id}")
        if gap.get("status") not in {
            CHANGE_REQUIRED,
            PATCH_PENDING_COMPILE,
        }:
            raise ValueError(
                f"semantic patch cannot claim gap {gap_id} in "
                f"status {gap.get('status')}"
            )

    authorized_ranges = [
        deepcopy(item)
        for item in authorized_changed_ranges
        if isinstance(item, dict)
        and item.get("authorized") is not False
    ]
    structured_direct_gap_ids = _ordered_unique_strings(
        gap_id
        for changed in authorized_ranges
        for gap_id in changed.get("direct_gap_ids") or []
        if str(gap_id) in state["gaps"]
        and (
            kind != "semantic_repair"
            or state["gaps"][str(gap_id)].get("status")
            in {CHANGE_REQUIRED, PATCH_PENDING_COMPILE}
        )
    )
    primary_gap_ids = _ordered_unique_strings(
        [
            *claimed_gap_ids,
            *(
                structured_direct_gap_ids
                if kind == "semantic_repair"
                else []
            ),
        ]
    )
    for gap_id in primary_gap_ids:
        gap = state["gaps"].get(gap_id)
        if not isinstance(gap, dict):
            raise ValueError(f"unknown semantic gap: {gap_id}")
        if gap.get("status") not in {
            CHANGE_REQUIRED,
            PATCH_PENDING_COMPILE,
        }:
            raise ValueError(
                f"semantic patch cannot directly cover gap {gap_id} in "
                f"status {gap.get('status')}"
            )

    affected_gap_ids = _structurally_affected_gap_ids(
        state,
        primary_gap_ids=primary_gap_ids,
        changed_ranges=authorized_ranges,
    )
    existing_patch_ids = {
        str(existing.get("id") or "")
        for existing in batch.get("patches") or []
        if isinstance(existing, dict)
        and str(existing.get("id") or "")
    }
    explicit_dependencies = _ordered_unique_strings(
        depends_on_patch_ids
    )
    unknown_dependencies = sorted(
        set(explicit_dependencies) - existing_patch_ids
    )
    if unknown_dependencies:
        raise ValueError(
            "semantic patch dependency references unknown or future "
            "transactions: " + ", ".join(unknown_dependencies)
        )
    overlap_dependencies = active_patch_ids_intersecting_changes(
        batch.get("active_patch_regions") or [],
        authorized_ranges,
    )
    affected_gap_set = set(affected_gap_ids)
    causal_gap_dependencies = (
        _ordered_unique_strings(
            str(existing.get("id") or "")
            for existing in batch.get("patches") or []
            if isinstance(existing, dict)
            and str(existing.get("id") or "")
            and affected_gap_set.intersection(
                {
                    str(gap_id)
                    for gap_id in existing.get("gap_ids") or []
                    if str(gap_id)
                }
            )
        )
        if kind == "compile_recovery"
        else []
    )
    patch_dependencies = _ordered_unique_strings(
        [
            *explicit_dependencies,
            *overlap_dependencies,
            *causal_gap_dependencies,
        ]
    )
    exact_provider_obligations = [
        deepcopy(item)
        for item in provider_obligations
        if isinstance(item, dict)
        and str(item.get("id") or "")
        and str(item.get("target_entity_id") or "")
        and str(item.get("target_contract_key") or "")
    ]
    contract_delta = (
        structural_contract_delta
        if isinstance(structural_contract_delta, dict)
        else {}
    )
    exact_structural_contract_delta = {
        disposition: [
            deepcopy(item)
            for item in contract_delta.get(disposition) or []
            if isinstance(item, dict)
            and str(item.get("file") or "")
            and str(item.get("contract_key") or "")
            and str(item.get("kind") or "")
            and str(item.get("symbol") or "")
        ]
        for disposition in ("removed", "added")
    }
    patch = {
        "id": str(patch_id),
        "patch_signature": signature,
        "gap_ids": primary_gap_ids,
        "claimed_gap_ids": claimed_gap_ids,
        "structured_direct_gap_ids": structured_direct_gap_ids,
        "affected_gap_ids": affected_gap_ids,
        "touched_files": _ordered_unique_strings(touched_files),
        "work_unit_id": str(work_unit_id),
        "kind": str(kind),
        "authorized_changed_ranges": authorized_ranges,
        "boundary_ref": str(boundary_ref or "") or None,
        "patch_evidence_ref": str(patch_evidence_ref or "") or None,
        "transaction_ref": str(transaction_ref or "") or None,
        "depends_on_patch_ids": patch_dependencies,
        "provider_obligations": exact_provider_obligations,
        "structural_contract_delta": exact_structural_contract_delta,
        "at": utc_now(),
    }
    batch.setdefault("patches", []).append(patch)
    batch["active_patch_regions"] = rebase_active_patch_regions(
        batch.get("active_patch_regions") or [],
        changed_ranges=patch["authorized_changed_ranges"],
        patch_id=str(patch_id),
        gap_ids=primary_gap_ids,
    )
    batch["primary_gap_ids"] = _ordered_unique_strings(
        [
            *(batch.get("primary_gap_ids") or []),
            *primary_gap_ids,
        ]
    )
    batch["affected_gap_ids"] = _ordered_unique_strings(
        [
            *(batch.get("affected_gap_ids") or []),
            *affected_gap_ids,
        ]
    )
    # A semantic patch opens a new unvalidated mutation batch.  A compiler
    # recovery patch, however, belongs to the batch whose last validation
    # already failed; it remains compile_failed until the next validation.
    batch["status"] = (
        BATCH_OPEN
        if kind == "semantic_repair"
        else BATCH_COMPILE_FAILED
    )
    for gap_id in primary_gap_ids:
        gap = state["gaps"][gap_id]
        gap["status"] = PATCH_PENDING_COMPILE
        gap["patch_ids"] = _ordered_unique_strings(
            [*(gap.get("patch_ids") or []), str(patch_id)]
        )
    for gap_id in affected_gap_ids:
        gap = state["gaps"][gap_id]
        if gap.get("status") in {
            CHANGE_REQUIRED,
            PATCH_PENDING_COMPILE,
        }:
            gap["status"] = PATCH_PENDING_COMPILE
            gap["patch_ids"] = _ordered_unique_strings(
                [*(gap.get("patch_ids") or []), str(patch_id)]
            )
            gap["indirect_patch_ids"] = _ordered_unique_strings(
                [
                    *(gap.get("indirect_patch_ids") or []),
                    str(patch_id),
                ]
            )
    _touch_state(
        state,
        event="batch_patch_recorded",
        payload={
            "batch_id": str(batch_id),
            "patch_id": str(patch_id),
            "gap_ids": primary_gap_ids,
            "claimed_gap_ids": claimed_gap_ids,
            "affected_gap_ids": affected_gap_ids,
            "kind": str(kind),
            "depends_on_patch_ids": patch_dependencies,
        },
    )
    validate_semantic_closure_state(state)
    return {
        "status": "recorded",
        "batch_id": str(batch_id),
        "patch_id": str(patch_id),
        "gap_ids": primary_gap_ids,
        "affected_gap_ids": affected_gap_ids,
        "depends_on_patch_ids": patch_dependencies,
    }


def record_batch_compile_validation(
    state: dict[str, Any],
    *,
    batch_id: str,
    compile_ok: bool,
    validation_ref: str,
    checkpoint_ref: str | None = None,
    round_number: int,
    gap_evaluations: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_semantic_closure_state(state)
    if str(state.get("active_batch_id") or "") != str(batch_id):
        raise ValueError("compile validation must target the active batch")
    batch = state["batches"].get(str(batch_id))
    if not isinstance(batch, dict):
        raise ValueError(f"unknown semantic batch: {batch_id}")
    if compile_ok and gap_evaluations is None:
        raise ValueError(
            "compile-valid semantic batch requires fresh current-tree "
            "contract evaluations"
        )
    compile_checkpoint_ref = str(
        checkpoint_ref or validation_ref or ""
    ).strip()
    if not compile_checkpoint_ref:
        raise ValueError("semantic compile validation has no checkpoint ref")
    validation = {
        "ref": str(validation_ref),
        "checkpoint_ref": compile_checkpoint_ref,
        "compile_ok": bool(compile_ok),
        "round": int(round_number),
        "at": utc_now(),
    }
    batch.setdefault("compile_validations", []).append(validation)
    handoff: object = None
    previous_checkpoint = ""
    if compile_ok:
        primary_ids = _ordered_unique_strings(
            batch.get("primary_gap_ids") or []
        )
        affected_ids = _ordered_unique_strings(
            gap_id
            for gap_id in batch.get("affected_gap_ids") or []
            if str(gap_id) not in set(primary_ids)
        )
        evaluation_gap_ids = _ordered_unique_strings(
            [*primary_ids, *affected_ids]
        )
        evaluation_by_gap = {
            str(item.get("gap_id") or ""): dict(item)
            for item in gap_evaluations or []
            if isinstance(item, dict)
            and str(item.get("gap_id") or "")
        }
        if set(evaluation_by_gap) != {
            str(gap_id)
            for gap_id in evaluation_gap_ids
        }:
            raise ValueError(
                "compile-valid semantic batch requires one current-tree "
                "evaluation per primary or structurally affected gap"
            )
        closed_gap_ids: list[str] = []
        reopened_gap_ids: list[str] = []
        reopened_decision_gap_ids: list[str] = []
        no_change_gap_ids: list[str] = []
        affected_closed_gap_ids: list[str] = []
        for gap_id in evaluation_gap_ids:
            gap = state["gaps"].get(str(gap_id))
            if not isinstance(gap, dict):
                raise ValueError(
                    f"semantic batch references unknown gap: {gap_id}"
                )
            is_primary = str(gap_id) in set(primary_ids)
            if is_primary and gap.get("status") != PATCH_PENDING_COMPILE:
                continue
            prior_status = str(gap.get("status") or "")
            evaluation = evaluation_by_gap.get(str(gap_id))
            evaluation_status = str(
                (evaluation or {}).get("status") or ""
            )
            if evaluation is not None:
                evaluation_ref = str(
                    evaluation.get("evaluation_ref") or ""
                )
                if not evaluation_ref:
                    raise ValueError(
                        "post-batch semantic evaluation has no evidence ref: "
                        f"{gap_id}"
                    )
                gap["post_batch_contract_evaluation"] = {
                    "status": evaluation_status,
                    "evaluation_ref": evaluation_ref,
                    "checkpoint_ref": compile_checkpoint_ref,
                    "required_dimensions": _ordered_unique_strings(
                        evaluation.get("required_dimensions")
                        or (
                            gap.get("compile_valid_handoff_evaluation")
                            or {}
                        ).get("required_dimensions")
                        or []
                    ),
                    "relations": deepcopy(
                        evaluation.get("relations") or {}
                    ),
                    "unresolved_dimensions": _ordered_unique_strings(
                        evaluation.get("unresolved_dimensions") or []
                    ),
                    "unavailable_dimensions": _ordered_unique_strings(
                        evaluation.get("unavailable_dimensions") or []
                    ),
                    "at": utc_now(),
                }
            if evaluation_status == "satisfied_modified":
                if is_primary or prior_status == CLOSED_NO_CHANGE:
                    gap["decision"] = {
                        "disposition": "modify",
                        "rationale": (
                            "a surviving patch changed the exact contract "
                            "entity and fresh current-tree evaluation proves "
                            "the modified contract is satisfied"
                        ),
                        "authority": (
                            "post_batch_current_tree_contract_evaluation"
                        ),
                        "checkpoint_ref": compile_checkpoint_ref,
                        "evidence_refs": [
                            str(evaluation.get("evaluation_ref") or "")
                        ],
                        "at": utc_now(),
                    }
                gap["status"] = CLOSED_MODIFIED
                closed_gap_ids.append(str(gap_id))
                if not is_primary:
                    affected_closed_gap_ids.append(str(gap_id))
            elif evaluation_status == "satisfied":
                gap["status"] = CLOSED_NO_CHANGE
                gap["decision"] = {
                    "disposition": "no_change_required",
                    "rationale": (
                        "the post-batch current tree satisfies the exact "
                        "contract without a surviving direct semantic change"
                    ),
                    "authority": (
                        "post_batch_current_tree_contract_evaluation"
                    ),
                    "checkpoint_ref": compile_checkpoint_ref,
                    "evidence_refs": [
                        str(evaluation.get("evaluation_ref") or "")
                    ],
                    "at": utc_now(),
                }
                if isinstance(state.get("semantic_gap_snapshot"), dict):
                    gap["decision"]["no_change_proof"] = (
                        _exact_no_change_proof(
                            state,
                            gap_id=str(gap_id),
                            evaluation=gap[
                                "post_batch_contract_evaluation"
                            ],
                        )
                    )
                closed_gap_ids.append(str(gap_id))
                no_change_gap_ids.append(str(gap_id))
                if not is_primary:
                    affected_closed_gap_ids.append(str(gap_id))
            elif evaluation_status in {
                "unresolved",
                "evidence_unavailable",
            }:
                # A fresh unresolved observation invalidates the decision
                # whose patch produced this tree; it does not itself prove
                # that another source mutation is required.
                gap["decision"] = None
                gap["status"] = PENDING_DECISION
                reopened_decision_gap_ids.append(str(gap_id))
                reopened_gap_ids.append(str(gap_id))
            else:
                raise ValueError(
                    "unsupported post-batch semantic evaluation status: "
                    f"{evaluation_status}"
                )
        if reopened_decision_gap_ids:
            snapshot = state.get("semantic_gap_snapshot")
            if not isinstance(snapshot, dict):
                raise ValueError(
                    "post-batch decision refresh requires a frozen gap "
                    "snapshot"
                )
            existing_epoch = state.get("pending_decision_epoch")
            epoch_gap_ids = _ordered_unique_strings(
                [
                    *(
                        existing_epoch.get("gap_ids") or []
                        if isinstance(existing_epoch, dict)
                        else []
                    ),
                    *reopened_decision_gap_ids,
                ]
            )
            state["pending_decision_epoch"] = {
                "schema": SEMANTIC_DECISION_EPOCH_SCHEMA,
                "id": stable_id(
                    "semantic-decision-epoch",
                    str(snapshot.get("id") or ""),
                    "post-batch-checkpoint-refresh",
                    compile_checkpoint_ref,
                    *epoch_gap_ids,
                ),
                "source": "post_batch_contract_refresh",
                "snapshot_id": str(snapshot.get("id") or ""),
                "checkpoint_ref": compile_checkpoint_ref,
                "gap_ids": epoch_gap_ids,
                "created_at": utc_now(),
            }
            _invalidate_running_gap_sweep_barriers(
                state,
                reopened_gap_ids=reopened_decision_gap_ids,
                reason="post_batch_compile_valid_checkpoint_refresh",
            )
        remaining_gap_ids = _ordered_unique_strings(
            gap_id
            for gap_id, gap in state["gaps"].items()
            if isinstance(gap, dict)
            and gap.get("status") in {
                PENDING_DECISION,
                CHANGE_REQUIRED,
            }
        )
        if remaining_gap_ids:
            _queue_repair_resumption(
                state,
                batch_id=str(batch_id),
                evidence_ref=str(validation_ref),
                reopened_gap_ids=remaining_gap_ids,
                source="post_compile_semantic_evaluation",
            )
        batch["status"] = BATCH_COMMITTED
        state["active_batch_id"] = None
        handoff = state.get("compile_valid_semantic_handoff")
        snapshot = state.get("semantic_gap_snapshot")
        if isinstance(handoff, dict) and handoff.get("status") != "complete":
            raise ValueError(
                "semantic batch commit requires a complete handoff"
            )
        if isinstance(handoff, dict):
            previous_checkpoint = str(handoff.get("checkpoint_ref") or "")
            handoff["checkpoint_ref"] = compile_checkpoint_ref
            if isinstance(snapshot, dict):
                snapshot["checkpoint_ref"] = compile_checkpoint_ref
        outcome = {
            "status": BATCH_COMMITTED,
            "closed_gap_ids": closed_gap_ids,
            "no_change_gap_ids": no_change_gap_ids,
            "reopened_gap_ids": reopened_gap_ids,
            "remaining_gap_ids": remaining_gap_ids,
            "affected_closed_gap_ids": affected_closed_gap_ids,
        }
    else:
        batch["status"] = BATCH_COMPILE_FAILED
        outcome = {
            "status": BATCH_COMPILE_FAILED,
            "closed_gap_ids": [],
        }
    _touch_state(
        state,
        event="patch_batch_compile_validated",
        payload={
            "batch_id": str(batch_id),
            **outcome,
            "validation_ref": str(validation_ref),
            "checkpoint_ref": compile_checkpoint_ref,
            **(
                {"from_checkpoint_ref": previous_checkpoint}
                if compile_ok and isinstance(handoff, dict)
                else {}
            ),
            "round": int(round_number),
        },
    )
    validate_semantic_closure_state(state)
    return {
        "batch_id": str(batch_id),
        **outcome,
    }


def rollback_active_batch(
    state: dict[str, Any],
    *,
    rollback_ref: str,
    reason: str,
) -> dict[str, Any]:
    """Update ledger state after the source batch was atomically rolled back."""

    validate_semantic_closure_state(state)
    batch_id = str(state.get("active_batch_id") or "")
    if not batch_id:
        return {"status": "not_required", "reopened_gap_ids": []}
    batch = state["batches"].get(batch_id)
    if not isinstance(batch, dict):
        raise ValueError("active semantic batch is missing")
    batch_patch_ids = {
        str(patch.get("id") or "")
        for patch in batch.get("patches") or []
        if isinstance(patch, dict) and str(patch.get("id") or "")
    }
    reopened: list[str] = []
    for gap_id in _ordered_unique_strings(
        [
            *(batch.get("primary_gap_ids") or []),
            *(batch.get("affected_gap_ids") or []),
        ]
    ):
        gap = state["gaps"].get(str(gap_id))
        if not isinstance(gap, dict):
            continue
        if gap.get("status") == PATCH_PENDING_COMPILE:
            gap["patch_ids"] = [
                patch_id
                for patch_id in _ordered_unique_strings(
                    gap.get("patch_ids") or []
                )
                if patch_id not in batch_patch_ids
            ]
            gap["indirect_patch_ids"] = [
                patch_id
                for patch_id in _ordered_unique_strings(
                    gap.get("indirect_patch_ids") or []
                )
                if patch_id not in batch_patch_ids
            ]
            gap["status"] = CHANGE_REQUIRED
            reopened.append(str(gap_id))
    batch["status"] = BATCH_ROLLED_BACK
    batch["rollback"] = {
        "ref": str(rollback_ref),
        "reason": str(reason),
        "at": utc_now(),
    }
    if reopened:
        _queue_repair_resumption(
            state,
            batch_id=batch_id,
            evidence_ref=str(rollback_ref),
            reopened_gap_ids=reopened,
            source="patch_batch_rollback",
        )
    state["active_batch_id"] = None
    _touch_state(
        state,
        event="active_patch_batch_rolled_back",
        payload={
            "batch_id": batch_id,
            "rollback_ref": str(rollback_ref),
            "reason": str(reason),
            "reopened_gap_ids": reopened,
        },
    )
    validate_semantic_closure_state(state)
    return {
        "status": BATCH_ROLLED_BACK,
        "batch_id": batch_id,
        "reopened_gap_ids": reopened,
        "repair_resumption": deepcopy(
            state.get("pending_repair_sweep")
        ),
    }


def reconcile_active_batch_affected_gaps(
    state: dict[str, Any],
    *,
    exact_edit_regions: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Recompute indirect contract lineage for a resumable active batch.

    This upgrades only durable, already-authorized patch records. It never
    creates a patch or closes a gap; newly linked open gaps become compile
    pending and must pass the same unified compile and fresh evaluation as
    direct gaps.
    """

    validate_semantic_closure_state(state)
    batch_id = str(state.get("active_batch_id") or "")
    if not batch_id:
        return {"status": "not_required", "affected_gap_ids": []}
    batch = state["batches"].get(batch_id)
    if not isinstance(batch, dict) or batch.get("status") != BATCH_OPEN:
        return {"status": "not_required", "affected_gap_ids": []}

    newly_affected: list[str] = []
    patch_by_id = {
        str(patch.get("id") or ""): patch
        for patch in batch.get("patches") or []
        if isinstance(patch, dict) and str(patch.get("id") or "")
    }

    def attribute_gap_to_patches(
        gap_id: str,
        patch_ids: Iterable[object],
    ) -> None:
        gap = state["gaps"].get(str(gap_id))
        exact_patch_ids = _ordered_unique_strings(
            patch_id
            for patch_id in patch_ids
            if str(patch_id) in patch_by_id
        )
        if (
            not isinstance(gap, dict)
            or not exact_patch_ids
            or gap.get("status") not in {
                CHANGE_REQUIRED,
                PATCH_PENDING_COMPILE,
            }
        ):
            return
        was_change_required = gap.get("status") == CHANGE_REQUIRED
        gap["status"] = PATCH_PENDING_COMPILE
        gap["patch_ids"] = _ordered_unique_strings(
            [*(gap.get("patch_ids") or []), *exact_patch_ids]
        )
        gap["indirect_patch_ids"] = _ordered_unique_strings(
            [
                *(gap.get("indirect_patch_ids") or []),
                *exact_patch_ids,
            ]
        )
        for patch_id in exact_patch_ids:
            patch = patch_by_id[patch_id]
            patch["affected_gap_ids"] = _ordered_unique_strings(
                [*(patch.get("affected_gap_ids") or []), str(gap_id)]
            )
        if was_change_required:
            newly_affected.append(str(gap_id))

    for patch in batch.get("patches") or []:
        if not isinstance(patch, dict):
            continue
        affected = _structurally_affected_gap_ids(
            state,
            primary_gap_ids=patch.get("gap_ids") or [],
            changed_ranges=patch.get("authorized_changed_ranges") or [],
        )
        patch["affected_gap_ids"] = _ordered_unique_strings(
            [*(patch.get("affected_gap_ids") or []), *affected]
        )
        for gap_id in affected:
            attribute_gap_to_patches(
                gap_id,
                [str(patch.get("id") or "")],
            )

    # A structured edit can become a no-op after an earlier patch in the
    # same batch changed that exact source region.  Attribute it only when
    # the current edit handle intersects a surviving patch postimage; the
    # model's claim or file co-location alone is not sufficient.
    for region in exact_edit_regions:
        if not isinstance(region, dict):
            continue
        relative_path = normalize_relpath(str(region.get("file") or ""))
        start_line = int(region.get("start_line") or 0)
        end_line = int(region.get("end_line") or start_line)
        direct_gap_ids = _ordered_unique_strings(
            region.get("direct_gap_ids") or []
        )
        if (
            not relative_path
            or start_line <= 0
            or end_line < start_line
            or not direct_gap_ids
            or not str(region.get("source_sha256") or "")
        ):
            continue
        intersecting_patch_ids = _ordered_unique_strings(
            patch_id
            for active_region in active_patch_regions_intersecting(
                batch.get("active_patch_regions") or [],
                file=relative_path,
                start_line=start_line,
                end_line=end_line,
            )
            for patch_id in active_region.get("patch_ids") or []
        )
        for gap_id in direct_gap_ids:
            attribute_gap_to_patches(gap_id, intersecting_patch_ids)
    batch["affected_gap_ids"] = _ordered_unique_strings(
        gap_id
        for patch in batch.get("patches") or []
        if isinstance(patch, dict)
        for gap_id in patch.get("affected_gap_ids") or []
        if str(gap_id) not in set(batch.get("primary_gap_ids") or [])
    )
    if newly_affected:
        _touch_state(
            state,
            event="active_batch_affected_gaps_reconciled",
            payload={
                "batch_id": batch_id,
                "affected_gap_ids": _ordered_unique_strings(newly_affected),
                "authority": (
                    "exact_contract_or_structured_edit_postimage_incidence"
                ),
            },
        )
    validate_semantic_closure_state(state)
    return {
        "status": "reconciled" if newly_affected else "unchanged",
        "batch_id": batch_id,
        "affected_gap_ids": _ordered_unique_strings(newly_affected),
    }


def _new_repair_resumption(
    *,
    batch_id: str,
    evidence_ref: str,
    reopened_gap_ids: Iterable[object],
    created_at: str,
    source: str,
) -> dict[str, Any]:
    reopened = _ordered_unique_strings(reopened_gap_ids)
    if not reopened:
        raise ValueError("semantic repair resumption requires reopened gaps")
    return {
        "schema": "semantic_repair_resumption_v1",
        "id": stable_id(
            "semantic-repair-resumption",
            str(batch_id),
            str(evidence_ref),
            *reopened,
        ),
        "source": str(source),
        "source_batch_id": str(batch_id),
        "evidence_ref": str(evidence_ref),
        "reopened_gap_ids": reopened,
        "status": "pending",
        "created_at": created_at,
        **(
            {"rollback_ref": str(evidence_ref)}
            if source == "patch_batch_rollback"
            else {}
        ),
    }


def _queue_repair_resumption(
    state: dict[str, Any],
    *,
    batch_id: str,
    evidence_ref: str,
    reopened_gap_ids: Iterable[object],
    source: str,
) -> dict[str, Any]:
    reopened = _ordered_unique_strings(reopened_gap_ids)
    if not reopened:
        raise ValueError("semantic repair resumption requires reopened gaps")
    created_at = utc_now()
    cause = {
        "source": str(source),
        "source_batch_id": str(batch_id),
        "evidence_ref": str(evidence_ref),
        "at": created_at,
    }
    pending = state.get("pending_repair_sweep")
    if not isinstance(pending, dict):
        pending = _new_repair_resumption(
            batch_id=batch_id,
            evidence_ref=evidence_ref,
            reopened_gap_ids=reopened,
            created_at=created_at,
            source=source,
        )
        pending["causes"] = [cause]
        state["pending_repair_sweep"] = pending
        return deepcopy(pending)
    if pending.get("status") != "pending":
        raise ValueError(
            "semantic repair resumption queue contains a non-pending item"
        )
    pending["reopened_gap_ids"] = _ordered_unique_strings(
        [*list(pending.get("reopened_gap_ids") or []), *reopened]
    )
    causes = [
        dict(item)
        for item in pending.get("causes") or []
        if isinstance(item, dict)
    ]
    if not causes:
        causes.append(
            {
                "source": str(pending.get("source") or ""),
                "source_batch_id": str(
                    pending.get("source_batch_id") or ""
                ),
                "evidence_ref": str(
                    pending.get("evidence_ref")
                    or pending.get("rollback_ref")
                    or ""
                ),
                "at": str(pending.get("created_at") or ""),
            }
        )
    cause_identity = (
        cause["source"],
        cause["source_batch_id"],
        cause["evidence_ref"],
    )
    if cause_identity not in {
        (
            str(item.get("source") or ""),
            str(item.get("source_batch_id") or ""),
            str(item.get("evidence_ref") or ""),
        )
        for item in causes
    }:
        causes.append(cause)
    pending["causes"] = causes
    return deepcopy(pending)


def completed_semantic_recovery_rounds(state: dict[str, Any]) -> int:
    """Count completed compiler-recovery rounds in the semantic ledger."""

    return sum(
        1
        for item in state.get("rounds") or []
        if isinstance(item, dict)
        and item.get("status") != ROUND_RUNNING
        and item.get("execution_mode") == CANONICAL_COMPILER_RECOVERY
    )


def semantic_repair_resume_context(
    state: dict[str, Any],
    *,
    max_recovery_rounds: int,
    max_gap_sweeps: int | None = None,
) -> dict[str, Any]:
    """Describe whether an isolated compile regression may be repaired again."""

    validate_semantic_closure_state(state)
    pending = state.get("pending_repair_sweep")
    if not isinstance(pending, dict):
        return {"status": "not_required"}
    reopened_gap_ids = _ordered_unique_strings(
        pending.get("reopened_gap_ids") or []
    )
    remaining_gap_ids = [
        gap_id
        for gap_id in reopened_gap_ids
        if ((state.get("gaps") or {}).get(gap_id) or {}).get("status")
        in {PENDING_DECISION, CHANGE_REQUIRED}
    ]
    completed_recovery_rounds = completed_semantic_recovery_rounds(state)
    limit = max(0, int(max_recovery_rounds))
    remaining_rounds = max(0, limit - completed_recovery_rounds)
    gap_sweep_limit = max(
        0,
        int(
            max_gap_sweeps
            if max_gap_sweeps is not None
            else max_recovery_rounds
        ),
    )
    completed_gap_sweeps = sum(
        1
        for item in state.get("rounds") or []
        if isinstance(item, dict)
        and item.get("status") != ROUND_RUNNING
        and item.get("execution_mode") == SEMANTIC_GAP_SWEEP
        and semantic_gap_sweep_consumed_budget(state, item)
    )
    remaining_gap_sweeps = max(
        0,
        gap_sweep_limit - completed_gap_sweeps,
    )
    if not remaining_gap_ids:
        status = "no_longer_required"
    elif state.get("active_batch_id"):
        status = "active_batch_present"
    elif remaining_gap_sweeps <= 0:
        status = "gap_sweep_budget_exhausted"
    else:
        status = "ready"
    return {
        "status": status,
        "resumption_id": str(pending.get("id") or ""),
        "source": str(pending.get("source") or ""),
        "source_batch_id": str(pending.get("source_batch_id") or ""),
        "evidence_ref": str(pending.get("evidence_ref") or ""),
        "rollback_ref": str(pending.get("rollback_ref") or ""),
        "reopened_gap_ids": reopened_gap_ids,
        "remaining_gap_ids": remaining_gap_ids,
        "completed_recovery_rounds": completed_recovery_rounds,
        "max_recovery_rounds": limit,
        "remaining_recovery_rounds": remaining_rounds,
        "compile_recovery_available": remaining_rounds > 0,
        "completed_gap_sweeps": completed_gap_sweeps,
        "max_gap_sweeps": gap_sweep_limit,
        "remaining_gap_sweeps": remaining_gap_sweeps,
    }


def consume_pending_repair_sweep(
    state: dict[str, Any],
    *,
    round_number: int,
) -> dict[str, Any]:
    """Bind one provenance-backed repair resumption to its new gap sweep."""

    validate_semantic_closure_state(state)
    pending = state.get("pending_repair_sweep")
    if not isinstance(pending, dict):
        raise ValueError("semantic repair resumption is not pending")
    record = {
        **deepcopy(pending),
        "status": "started",
        "round": int(round_number),
        "started_at": utc_now(),
    }
    state.setdefault("repair_resumptions", []).append(record)
    state["pending_repair_sweep"] = None
    _touch_state(
        state,
        event="semantic_repair_resumption_started",
        payload={
            "resumption_id": str(record.get("id") or ""),
            "round": int(round_number),
            "reopened_gap_ids": list(
                record.get("reopened_gap_ids") or []
            ),
        },
    )
    validate_semantic_closure_state(state)
    return record


def dismiss_pending_repair_sweep(
    state: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    """Close a resumption whose reopened gaps were satisfied indirectly."""

    validate_semantic_closure_state(state)
    pending = state.get("pending_repair_sweep")
    if not isinstance(pending, dict):
        raise ValueError("semantic repair resumption is not pending")
    record = {
        **deepcopy(pending),
        "status": "no_longer_required",
        "reason": str(reason),
        "resolved_at": utc_now(),
    }
    state.setdefault("repair_resumptions", []).append(record)
    state["pending_repair_sweep"] = None
    _touch_state(
        state,
        event="semantic_repair_resumption_dismissed",
        payload={
            "resumption_id": str(record.get("id") or ""),
            "reason": str(reason),
        },
    )
    validate_semantic_closure_state(state)
    return record


def begin_semantic_round(
    state: dict[str, Any],
    *,
    round_number: int,
    max_rounds: int,
    execution_mode: str = CANONICAL_COMPILER_RECOVERY,
    phase_round: int | None = None,
) -> dict[str, Any]:
    validate_semantic_closure_state(state)
    number = int(round_number)
    limit = int(max_rounds)
    budget_index = int(phase_round or number)
    mode = str(execution_mode or "")
    if mode not in ROUND_EXECUTION_MODES:
        raise ValueError(f"unsupported semantic execution mode: {mode}")
    if (
        number <= 0
        or limit <= 0
        or budget_index <= 0
        or budget_index > limit
    ):
        raise ValueError(
            f"semantic phase round {budget_index} exceeds configured "
            f"budget {limit}"
        )
    existing = _round_record(state, number)
    if existing is not None:
        if existing.get("status") != ROUND_RUNNING:
            raise ValueError(f"semantic round {number} is already terminal")
        if (
            int(existing.get("max_rounds") or 0) != limit
            or int(existing.get("phase_round") or 0) != budget_index
            or str(existing.get("execution_mode") or "") != mode
        ):
            raise ValueError(
                f"semantic round {number} cannot resume with a different "
                "execution mode or phase budget"
            )
        return existing
    if state.get("rounds") and number != (
        max(int(item.get("number") or 0) for item in state["rounds"]) + 1
    ):
        raise ValueError("semantic rounds must be created sequentially")
    record = {
        "number": number,
        "max_rounds": limit,
        "phase_round": budget_index,
        "execution_mode": mode,
        "status": ROUND_RUNNING,
        "units": {},
        **(
            {
                "execution_policy_revision": (
                    SEMANTIC_COMPILE_RECOVERY_POLICY_REVISION
                )
            }
            if mode == CANONICAL_COMPILER_RECOVERY
            else {}
        ),
    }
    state["rounds"].append(record)
    _touch_state(
        state,
        event="semantic_round_started",
        payload={
            "round": number,
            "phase_round": budget_index,
            "max_rounds": limit,
            "execution_mode": mode,
        },
    )
    validate_semantic_closure_state(state)
    return record


def _round_record(
    state: dict[str, Any],
    round_number: int,
) -> dict[str, Any] | None:
    matches = [
        item
        for item in state.get("rounds") or []
        if int(item.get("number") or 0) == int(round_number)
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate semantic round: {round_number}")
    return matches[0] if matches else None


def _unit_actions_in_evidence_state(
    unit: dict[str, Any],
    *,
    evidence_state_signature: str,
) -> list[dict[str, Any]]:
    actions = [
        item
        for field in ("attempts", "evidence_actions")
        for item in unit.get(field) or []
        if isinstance(item, dict)
    ]
    signature = str(evidence_state_signature or "").strip()
    if not signature:
        raise ValueError("semantic unit action requires an evidence fingerprint")
    return [
        item
        for item in actions
        if str(item.get("evidence_state_signature") or "") == signature
    ]


def semantic_unit_action_count(
    state: dict[str, Any],
    *,
    round_number: int,
    work_unit_id: str,
    evidence_state_signature: str,
) -> int:
    """Count model actions consumed under one exact evidence fingerprint."""

    round_record = _round_record(state, int(round_number))
    if not isinstance(round_record, dict):
        return 0
    unit = (round_record.get("units") or {}).get(str(work_unit_id))
    if not isinstance(unit, dict):
        return 0
    return len(
        _unit_actions_in_evidence_state(
            unit,
            evidence_state_signature=evidence_state_signature,
        )
    )


def _record_unit_action(
    state: dict[str, Any],
    *,
    round_number: int,
    work_unit_id: str,
    action_signature: str,
    outcome: str,
    max_steps: int,
    evidence_refs: Iterable[object] = (),
    evidence_state_signature: str,
    action_kind: str,
) -> dict[str, Any]:
    validate_semantic_closure_state(state)
    round_record = _round_record(state, int(round_number))
    if round_record is None or round_record.get("status") != ROUND_RUNNING:
        raise ValueError("unit action requires an active semantic round")
    if action_kind not in {"attempt", "evidence_action"}:
        raise ValueError(f"unsupported semantic unit action kind: {action_kind}")
    unit_id = str(work_unit_id or "").strip()
    if not unit_id:
        raise ValueError("semantic unit action requires a stable work_unit_id")
    step_limit = max(1, int(max_steps))
    unit = round_record.setdefault("units", {}).setdefault(
        unit_id,
        {
            "work_unit_id": unit_id,
            "step_limit": step_limit,
            "attempts": [],
            "evidence_actions": [],
            "replan_evidence_refs": [],
        },
    )
    if int(unit.get("step_limit") or 0) != step_limit:
        raise ValueError("logical semantic unit step limit cannot change")
    evidence_signature = str(evidence_state_signature or "").strip()
    if not evidence_signature:
        raise ValueError("semantic unit action requires an evidence fingerprint")
    actions_in_evidence_state = _unit_actions_in_evidence_state(
        unit,
        evidence_state_signature=evidence_signature,
    )
    consumed_in_state = len(actions_in_evidence_state)
    if consumed_in_state >= step_limit:
        raise ValueError(
            f"semantic unit {unit_id} exhausted its {step_limit} model "
            "actions for the current evidence state"
        )
    signature = str(action_signature or "").strip()
    if not signature:
        raise ValueError("semantic unit action requires a stable signature")
    duplicate = any(
        str(item.get("action_signature") or "") == signature
        for item in actions_in_evidence_state
    )
    collection = "attempts" if action_kind == "attempt" else "evidence_actions"
    ordinal = "step" if action_kind == "attempt" else "evidence_action"
    actions = list(unit.get(collection) or [])
    action = {
        ordinal: len(actions) + 1,
        "action_signature": signature,
        "outcome": str(outcome),
        "duplicate": duplicate,
        "evidence_refs": _ordered_unique_strings(evidence_refs),
        "evidence_state_signature": evidence_signature,
        "at": utc_now(),
    }
    actions.append(action)
    unit[collection] = actions
    event_name = (
        "unit_attempt_recorded"
        if action_kind == "attempt"
        else "unit_evidence_action_recorded"
    )
    _touch_state(
        state,
        event=f"semantic_{event_name}",
        payload={
            "round": int(round_number),
            "work_unit_id": unit_id,
            ordinal: action[ordinal],
            "duplicate": duplicate,
            "outcome": str(outcome),
        },
    )
    validate_semantic_closure_state(state)
    return {
        **action,
        "remaining_steps": step_limit - consumed_in_state - 1,
    }


def record_unit_attempt(
    state: dict[str, Any],
    *,
    round_number: int,
    work_unit_id: str,
    action_signature: str,
    outcome: str,
    max_steps: int,
    evidence_refs: Iterable[object] = (),
    evidence_state_signature: str,
) -> dict[str, Any]:
    """Consume one patch/review step in an exact evidence state."""

    return _record_unit_action(
        state,
        round_number=round_number,
        work_unit_id=work_unit_id,
        action_signature=action_signature,
        outcome=outcome,
        max_steps=max_steps,
        evidence_refs=evidence_refs,
        evidence_state_signature=evidence_state_signature,
        action_kind="attempt",
    )


def record_unit_evidence_action(
    state: dict[str, Any],
    *,
    round_number: int,
    work_unit_id: str,
    action_signature: str,
    outcome: str,
    max_steps: int,
    evidence_refs: Iterable[object] = (),
    evidence_state_signature: str,
) -> dict[str, Any]:
    """Record one graph/tool action in the current evidence-state budget."""

    return _record_unit_action(
        state,
        round_number=round_number,
        work_unit_id=work_unit_id,
        action_signature=action_signature,
        outcome=outcome,
        max_steps=max_steps,
        evidence_refs=evidence_refs,
        evidence_state_signature=evidence_state_signature,
        action_kind="evidence_action",
    )


def record_unit_replan_evidence(
    state: dict[str, Any],
    *,
    round_number: int,
    work_unit_id: str,
    evidence_refs: Iterable[object],
) -> dict[str, Any]:
    """Attach graph evidence without creating a new logical unit budget."""

    validate_semantic_closure_state(state)
    round_record = _round_record(state, int(round_number))
    if round_record is None or round_record.get("status") != ROUND_RUNNING:
        raise ValueError("unit replan requires an active semantic round")
    unit = (round_record.get("units") or {}).get(str(work_unit_id))
    if not isinstance(unit, dict):
        raise ValueError("unit replan requires an existing logical unit")
    refs = _ordered_unique_strings(
        [
            *(unit.get("replan_evidence_refs") or []),
            *list(evidence_refs),
        ]
    )
    unit["replan_evidence_refs"] = refs
    _touch_state(
        state,
        event="semantic_unit_replan_evidence_recorded",
        payload={
            "round": int(round_number),
            "work_unit_id": str(work_unit_id),
            "evidence_refs": refs,
        },
    )
    validate_semantic_closure_state(state)
    return {
        "work_unit_id": str(work_unit_id),
        "attempt_count": len(unit.get("attempts") or []),
        "remaining_steps": (
            int(unit.get("step_limit") or 0)
            - len(unit.get("attempts") or [])
            - len(unit.get("evidence_actions") or [])
        ),
        "replan_evidence_refs": refs,
    }


def finish_semantic_round(
    state: dict[str, Any],
    *,
    round_number: int,
    status: str,
) -> dict[str, Any]:
    validate_semantic_closure_state(state)
    if status not in {
        ROUND_VALIDATED,
        ROUND_COMPILE_FAILED,
        ROUND_NO_PROGRESS,
        ROUND_BUDGET_EXHAUSTED,
    }:
        raise ValueError(f"unsupported semantic round outcome: {status}")
    round_record = _round_record(state, int(round_number))
    if round_record is None or round_record.get("status") != ROUND_RUNNING:
        raise ValueError("semantic round is not active")
    if round_record.get("execution_mode") == SEMANTIC_GAP_SWEEP:
        round_record["budget_consumed"] = semantic_gap_sweep_consumed_budget(
            state,
            round_record,
        )
    round_record["status"] = status
    _touch_state(
        state,
        event="semantic_round_finished",
        payload={"round": int(round_number), "status": status},
    )
    validate_semantic_closure_state(state)
    return round_record


def semantic_gap_sweep_consumed_budget(
    state: dict[str, Any],
    round_record: dict[str, Any],
) -> bool:
    """Return whether a gap sweep performed durable semantic work.

    Decision records and mutation attempts are the budget authority. Merely
    carrying stale completion barriers across a checkpoint refresh is not.
    """

    if round_record.get("execution_mode") != SEMANTIC_GAP_SWEEP:
        return False
    number = int(round_record.get("number") or 0)
    if number <= 0:
        return False
    if any(
        isinstance(unit, dict)
        and (
            unit.get("attempts")
            or unit.get("evidence_actions")
            or unit.get("replan_evidence_refs")
        )
        for unit in (round_record.get("units") or {}).values()
    ):
        return True
    if (
        round_record.get("mutation_snapshot_gap_ids")
        or round_record.get("attempted_gap_ids")
    ):
        return True
    return any(
        isinstance(gap, dict)
        and isinstance(gap.get("decision"), dict)
        and int(gap["decision"].get("round") or 0) == number
        for gap in (state.get("gaps") or {}).values()
    )


def semantic_closure_summary(
    state: dict[str, Any],
) -> dict[str, Any]:
    validate_semantic_closure_state(state)
    statuses: dict[str, int] = {status: 0 for status in GAP_STATUSES}
    for gap in state.get("gaps") or {}:
        status = str(state["gaps"][gap].get("status") or "")
        statuses[status] = statuses.get(status, 0) + 1
    open_count = sum(statuses.get(status, 0) for status in OPEN_GAP_STATUSES)
    terminal_count = sum(
        statuses.get(status, 0) for status in TERMINAL_GAP_STATUSES
    )
    evidence_unavailable_gap_ids = sorted(
        str(gap_id)
        for gap_id, gap in (state.get("gaps") or {}).items()
        if isinstance(gap, dict)
        and gap.get("status") == PENDING_DECISION
        and (
            gap.get("compile_valid_handoff_evaluation") or {}
        ).get("status")
        == "evidence_unavailable"
    )
    # Evidence-unavailable is an observation from the static handoff, not a
    # reason to remove a gap from the decision queue.  The review phase must
    # distinguish a proven target absence from a failed evidence acquisition.
    reviewable_gap_count = statuses.get(PENDING_DECISION, 0)
    active_batch_id = str(state.get("active_batch_id") or "")
    if statuses.get(PENDING_DECISION, 0):
        closure_status = "semantic_decisions_pending"
    elif statuses.get(PATCH_PENDING_COMPILE, 0) or active_batch_id:
        closure_status = "semantic_compile_validation_pending"
    elif statuses.get(CHANGE_REQUIRED, 0):
        closure_status = "semantic_repairs_pending"
    else:
        closure_status = "semantic_migration_closed"
    return {
        "schema": SEMANTIC_CLOSURE_STATE_SCHEMA,
        "gap_count": len(state.get("gaps") or {}),
        "open_gap_count": open_count,
        "terminal_gap_count": terminal_count,
        "reviewable_gap_count": reviewable_gap_count,
        "evidence_unavailable_gap_count": len(
            evidence_unavailable_gap_ids
        ),
        "evidence_unavailable_gap_ids": evidence_unavailable_gap_ids,
        "status_counts": {
            key: value for key, value in statuses.items() if value
        },
        "active_batch_id": active_batch_id or None,
        "semantic_rounds_completed": completed_semantic_recovery_rounds(
            state
        ),
        "gap_sweeps_completed": sum(
            1
            for item in state.get("rounds") or []
            if item.get("status") != ROUND_RUNNING
            and item.get("execution_mode") == SEMANTIC_GAP_SWEEP
            and semantic_gap_sweep_consumed_budget(state, item)
        ),
        "repair_resumption_count": len(
            state.get("repair_resumptions") or []
        ),
        "pending_repair_resumption_id": str(
            (state.get("pending_repair_sweep") or {}).get("id") or ""
        )
        or None,
        "semantic_gap_snapshot_id": str(
            (state.get("semantic_gap_snapshot") or {}).get("id") or ""
        )
        or None,
        "proof_policy_revision": str(
            (state.get("semantic_gap_snapshot") or {}).get(
                "proof_policy_revision"
            )
            or ""
        )
        or None,
        "decision_policy_revision": SEMANTIC_DECISION_POLICY_REVISION,
        "pending_decision_epoch_id": str(
            (state.get("pending_decision_epoch") or {}).get("id") or ""
        )
        or None,
        "closure_status": closure_status,
        "static_semantic_closed": (
            closure_status == "semantic_migration_closed"
        ),
    }


def validate_semantic_closure_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict):
        raise ValueError("semantic closure state must be an object")
    if state.get("schema") != SEMANTIC_CLOSURE_STATE_SCHEMA:
        raise ValueError("unsupported semantic closure state schema")
    if not str(state.get("case_id") or "").strip():
        raise ValueError("semantic closure state has no case_id")
    for mapping_name in ("source_unit_refs", "raw_hunk_refs"):
        mapping = state.get(mapping_name, {})
        if not isinstance(mapping, dict) or any(
            not str(identity or "").strip()
            or not str(reference or "").strip()
            for identity, reference in mapping.items()
        ):
            raise ValueError(
                f"semantic closure {mapping_name} must map stable ids to "
                "record references"
            )
    if "initial_unit_outcomes_ref" in state and not str(
        state.get("initial_unit_outcomes_ref") or ""
    ).strip():
        raise ValueError(
            "semantic closure initial_unit_outcomes_ref is empty"
        )
    gaps = state.get("gaps")
    batches = state.get("batches")
    rounds = state.get("rounds")
    if not isinstance(gaps, dict):
        raise ValueError("semantic closure gaps must be an object")
    if not isinstance(batches, dict):
        raise ValueError("semantic closure batches must be an object")
    if not isinstance(rounds, list):
        raise ValueError("semantic closure rounds must be a list")
    snapshot = state.get("semantic_gap_snapshot")
    if snapshot is not None:
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema") != SEMANTIC_GAP_SNAPSHOT_SCHEMA
            or not str(snapshot.get("id") or "")
            or not str(snapshot.get("checkpoint_ref") or "")
            or not str(snapshot.get("evaluator_revision") or "")
            or snapshot.get("proof_policy_revision")
            != SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION
        ):
            raise ValueError("semantic gap snapshot is invalid")
        expected_gap_ids = sorted(str(gap_id) for gap_id in gaps)
        if list(snapshot.get("gap_ids") or []) != expected_gap_ids:
            raise ValueError("frozen semantic gap ids have changed")
        if str(snapshot.get("gap_set_fingerprint") or "") != (
            _semantic_gap_set_fingerprint(gaps)
        ):
            raise ValueError("frozen semantic gap fingerprint has changed")
    decision_epoch = state.get("pending_decision_epoch")
    if decision_epoch is not None:
        if (
            not isinstance(decision_epoch, dict)
            or decision_epoch.get("schema")
            != SEMANTIC_DECISION_EPOCH_SCHEMA
            or decision_epoch.get("source")
            not in {
                "compile_valid_proof_review",
                "compile_valid_checkpoint_refresh",
                "decision_policy_refresh",
                "post_batch_contract_refresh",
                "authoritative_noop_refresh",
            }
            or not str(decision_epoch.get("id") or "")
            or not isinstance(snapshot, dict)
            or str(decision_epoch.get("snapshot_id") or "")
            != str(snapshot.get("id") or "")
        ):
            raise ValueError("semantic decision epoch is invalid")
        epoch_gap_ids = _ordered_unique_strings(
            decision_epoch.get("gap_ids") or []
        )
        if not epoch_gap_ids or any(
            gap_id not in gaps
            or gaps[gap_id].get("status") != PENDING_DECISION
            for gap_id in epoch_gap_ids
        ):
            raise ValueError(
                "semantic decision epoch must contain pending frozen gaps"
            )
    repair_resumptions = state.get("repair_resumptions", [])
    if not isinstance(repair_resumptions, list):
        raise ValueError(
            "semantic closure repair resumptions must be a list"
        )
    resumption_ids: set[str] = set()

    def validate_resumption_provenance(
        resumption: dict[str, Any],
    ) -> None:
        if (
            resumption.get("schema")
            != "semantic_repair_resumption_v1"
        ):
            raise ValueError(
                "semantic repair resumption has an invalid schema"
            )
        if resumption.get("source") not in {
            "patch_batch_rollback",
            "post_compile_semantic_evaluation",
        }:
            raise ValueError(
                "semantic repair resumption has an invalid source"
            )
        source_batch_id = str(
            resumption.get("source_batch_id") or ""
        )
        if source_batch_id not in batches:
            raise ValueError(
                "semantic repair resumption references an unknown batch"
            )
        if not str(
            resumption.get("evidence_ref")
            or resumption.get("rollback_ref")
            or ""
        ):
            raise ValueError(
                "semantic repair resumption has no transition evidence"
            )
        reopened_gap_ids = _ordered_unique_strings(
            resumption.get("reopened_gap_ids") or []
        )
        if not reopened_gap_ids:
            raise ValueError(
                "semantic repair resumption has no reopened gaps"
            )
        if any(gap_id not in gaps for gap_id in reopened_gap_ids):
            raise ValueError(
                "semantic repair resumption references unknown gaps"
            )
        for cause in resumption.get("causes") or []:
            if not isinstance(cause, dict):
                raise ValueError(
                    "semantic repair resumption cause must be an object"
                )
            if cause.get("source") not in {
                "patch_batch_rollback",
                "post_compile_semantic_evaluation",
            }:
                raise ValueError(
                    "semantic repair resumption cause has an invalid source"
                )
            if str(cause.get("source_batch_id") or "") not in batches:
                raise ValueError(
                    "semantic repair resumption cause references an "
                    "unknown batch"
                )
            if not str(cause.get("evidence_ref") or ""):
                raise ValueError(
                    "semantic repair resumption cause has no evidence"
                )

    for resumption in repair_resumptions:
        if not isinstance(resumption, dict):
            raise ValueError(
                "semantic closure repair resumption must be an object"
            )
        resumption_id = str(resumption.get("id") or "")
        if not resumption_id or resumption_id in resumption_ids:
            raise ValueError(
                "semantic closure repair resumption ids must be unique"
            )
        validate_resumption_provenance(resumption)
        if resumption.get("status") not in {
            "started",
            "no_longer_required",
        }:
            raise ValueError(
                "consumed semantic repair resumption has an invalid status"
            )
        if (
            resumption.get("status") == "started"
            and int(resumption.get("round") or 0) <= 0
        ):
            raise ValueError(
                "semantic repair resumption has no target round"
            )
        resumption_ids.add(resumption_id)

    pending_repair_sweep = state.get("pending_repair_sweep")
    if pending_repair_sweep is not None:
        if not isinstance(pending_repair_sweep, dict):
            raise ValueError(
                "pending semantic repair sweep must be an object"
            )
        pending_id = str(pending_repair_sweep.get("id") or "")
        if not pending_id or pending_id in resumption_ids:
            raise ValueError(
                "pending semantic repair sweep has an invalid identity"
            )
        if pending_repair_sweep.get("status") != "pending":
            raise ValueError(
                "pending semantic repair sweep has an invalid status"
            )
        validate_resumption_provenance(pending_repair_sweep)

    for gap_id, gap in gaps.items():
        if not isinstance(gap, dict) or str(gap.get("id") or "") != str(
            gap_id
        ):
            raise ValueError(f"invalid semantic gap record: {gap_id}")
        if gap.get("status") not in GAP_STATUSES:
            raise ValueError(
                f"unsupported semantic gap status for {gap_id}: "
                f"{gap.get('status')}"
            )
        if not str(gap.get("contract_identity") or "").strip():
            raise ValueError(f"semantic gap has no contract identity: {gap_id}")
        if gap.get("status") == CLOSED_NO_CHANGE:
            disposition = str(
                (gap.get("decision") or {}).get("disposition") or ""
            )
            if disposition != "no_change_required":
                raise ValueError(
                    f"no-change gap has no explicit decision: {gap_id}"
                )
            if (
                isinstance(snapshot, dict)
                and snapshot.get("proof_policy_revision")
                == SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION
            ):
                _validate_no_change_proof(
                    state,
                    gap_id=str(gap_id),
                    proof=(gap.get("decision") or {}).get(
                        "no_change_proof"
                    ),
                )
        if gap.get("status") in {
            CHANGE_REQUIRED,
            PATCH_PENDING_COMPILE,
        }:
            disposition = str(
                (gap.get("decision") or {}).get("disposition") or ""
            )
            if disposition not in {
                "change_required",
                "modify",
            }:
                raise ValueError(
                    f"repair gap has no explicit change decision: {gap_id}"
                )
        if (
            gap.get("status") == PATCH_PENDING_COMPILE
            and not gap.get("patch_ids")
        ):
            raise ValueError(
                f"compile-pending gap has no patch: {gap_id}"
            )

    active_batch_id = str(state.get("active_batch_id") or "")
    active_batches = []
    for batch_id, batch in batches.items():
        if not isinstance(batch, dict) or str(batch.get("id") or "") != str(
            batch_id
        ):
            raise ValueError(f"invalid semantic batch record: {batch_id}")
        if batch.get("status") not in BATCH_STATUSES:
            raise ValueError(
                f"unsupported semantic batch status: {batch.get('status')}"
            )
        if batch.get("status") in {BATCH_OPEN, BATCH_COMPILE_FAILED}:
            active_batches.append(str(batch_id))
        for gap_id in batch.get("primary_gap_ids") or []:
            if str(gap_id) not in gaps:
                raise ValueError(
                    f"semantic batch {batch_id} references unknown gap "
                    f"{gap_id}"
                )
        patch_records = [
            patch
            for patch in batch.get("patches") or []
            if isinstance(patch, dict)
        ]
        patch_ids = [
            str(patch.get("id") or "") for patch in patch_records
        ]
        if any(not patch_id for patch_id in patch_ids) or len(
            patch_ids
        ) != len(set(patch_ids)):
            raise ValueError(
                f"semantic batch {batch_id} patch ids must be unique"
            )
        known_patch_ids = set(patch_ids)
        for patch in patch_records:
            patch_id = str(patch["id"])
            dependencies = _ordered_unique_strings(
                patch.get("depends_on_patch_ids") or []
            )
            invalid_dependencies = sorted(
                (set(dependencies) - known_patch_ids) | (
                    {patch_id} if patch_id in dependencies else set()
                )
            )
            if invalid_dependencies:
                raise ValueError(
                    f"semantic patch {patch_id} has invalid transaction "
                    "dependencies: " + ", ".join(invalid_dependencies)
                )
    if len(active_batches) > 1:
        raise ValueError("only one semantic patch batch may be active")
    if active_batch_id:
        if active_batches != [active_batch_id]:
            raise ValueError("active semantic batch pointer is inconsistent")
    elif active_batches:
        raise ValueError("active semantic batch has no state pointer")

    round_numbers: list[int] = []
    running_round_numbers: list[int] = []
    for round_record in rounds:
        if not isinstance(round_record, dict):
            raise ValueError("semantic round record must be an object")
        number = int(round_record.get("number") or 0)
        if number <= 0 or number in round_numbers:
            raise ValueError("semantic round numbers must be unique and positive")
        round_numbers.append(number)
        if round_record.get("status") not in ROUND_STATUSES:
            raise ValueError(
                f"unsupported semantic round status: "
                f"{round_record.get('status')}"
            )
        execution_mode = str(round_record.get("execution_mode") or "")
        if execution_mode not in ROUND_EXECUTION_MODES:
            raise ValueError(
                "semantic round has an unsupported execution mode: "
                f"{execution_mode or '<missing>'}"
            )
        if (
            execution_mode == CANONICAL_COMPILER_RECOVERY
            and str(round_record.get("execution_policy_revision") or "")
            != SEMANTIC_COMPILE_RECOVERY_POLICY_REVISION
        ):
            raise ValueError(
                "compiler recovery round does not use the current policy"
            )
        phase_round = int(round_record.get("phase_round") or 0)
        max_rounds = int(round_record.get("max_rounds") or 0)
        if phase_round <= 0 or max_rounds <= 0 or phase_round > max_rounds:
            raise ValueError("semantic round phase budget is inconsistent")
        if execution_mode == SEMANTIC_GAP_SWEEP and (
            phase_round != 1 or max_rounds != 1
        ):
            raise ValueError(
                "the semantic gap sweep must be one mutation phase, "
                "independent of compiler-recovery rounds"
            )
        if (
            execution_mode == SEMANTIC_GAP_SWEEP
            and round_record.get("status") != ROUND_RUNNING
            and "budget_consumed" in round_record
        ):
            budget_consumed = round_record.get("budget_consumed")
            if not isinstance(budget_consumed, bool):
                raise ValueError(
                    "semantic gap sweep budget marker must be boolean"
                )
            if budget_consumed != semantic_gap_sweep_consumed_budget(
                state,
                round_record,
            ):
                raise ValueError(
                    "semantic gap sweep budget marker disagrees with "
                    "durable work evidence"
                )
        if (
            execution_mode == SEMANTIC_GAP_SWEEP
            and round_record.get("status") == ROUND_RUNNING
            and round_record.get("decision_sweep_complete")
            and any(
                isinstance(gap, dict)
                and gap.get("status") == PENDING_DECISION
                for gap in gaps.values()
            )
        ):
            raise ValueError(
                "a completed semantic decision sweep retains pending gaps"
            )
        if round_record.get("status") == ROUND_RUNNING:
            running_round_numbers.append(number)
        for unit_id, unit in (round_record.get("units") or {}).items():
            if str(unit.get("work_unit_id") or "") != str(unit_id):
                raise ValueError("semantic logical unit identity is inconsistent")
            attempts = list(unit.get("attempts") or [])
            evidence_actions = list(unit.get("evidence_actions") or [])
            step_limit = int(unit.get("step_limit") or 0)
            budgeted_actions = [*attempts, *evidence_actions]
            action_counts_by_evidence: dict[str, int] = {}
            for action in budgeted_actions:
                if not isinstance(action, dict):
                    raise ValueError("semantic logical unit action is invalid")
                signature = str(action.get("evidence_state_signature") or "")
                if not signature:
                    raise ValueError(
                        "semantic logical unit action has no evidence fingerprint"
                    )
                action_counts_by_evidence[signature] = (
                    action_counts_by_evidence.get(signature, 0) + 1
                )
            max_action_count = max(
                action_counts_by_evidence.values(),
                default=0,
            )
            if step_limit <= 0 or max_action_count > step_limit:
                raise ValueError("semantic logical unit exceeded step budget")
            expected_steps = list(range(1, len(attempts) + 1))
            actual_steps = [
                int(item.get("step") or 0) for item in attempts
            ]
            if actual_steps != expected_steps:
                raise ValueError(
                    "semantic logical unit steps must be contiguous"
                )
    if round_numbers != sorted(round_numbers):
        raise ValueError("semantic rounds must be stored in order")
    if len(running_round_numbers) > 1:
        raise ValueError("only one semantic round may be running")
    if (
        running_round_numbers
        and running_round_numbers[0] != round_numbers[-1]
    ):
        raise ValueError("only the latest semantic round may be running")
    if active_batch_id:
        active_batch = batches[active_batch_id]
        if (
            active_batch.get("status") == BATCH_OPEN
            and not any(
                int(round_record.get("number") or 0)
                in running_round_numbers
                and round_record.get("execution_mode")
                == SEMANTIC_GAP_SWEEP
                for round_record in rounds
            )
        ):
            raise ValueError(
                "an unvalidated semantic batch requires a running gap sweep"
            )
