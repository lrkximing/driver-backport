from __future__ import annotations

from typing import Any, Iterable

from driver_backport.planning.execution_phase import (
    EXECUTION_PHASE_COMPILER_REPAIR,
    EXECUTION_PHASE_SEMANTIC_CLOSURE,
    validate_execution_phase,
)


WORKFLOW_RESUME_SCHEMA = "migration_workflow_resume_v1"

RESUMABLE_CASE_STATES = {
    "transplant_applied_pending_validation",
    "repair_applied_pending_validation",
    "patch_apply_incomplete",
    "needs_patching",
    "ready_for_manual_test",
}


def migration_state_is_resumable(state: dict[str, Any]) -> bool:
    status = str(state.get("status") or "")
    if status in RESUMABLE_CASE_STATES:
        return True
    if status == "needs_build_preparation":
        return bool(state.get("last_transplant_plan_path"))
    return status == "planning" and bool(
        state.get("last_transplant_plan_path")
        or state.get("last_validation_path")
    )


def project_workflow_resume(
    *,
    cycle_state: dict[str, Any],
    semantic_resume: dict[str, Any] | None,
    pending_manifest_phases: Iterable[str] = (),
) -> dict[str, Any]:
    """Select one resume action from existing durable state.

    This function writes no state and classifies no diagnostics.  It only
    enforces ordering between phase state, semantic ledger state, and repair
    transaction journals.
    """

    phase = (
        EXECUTION_PHASE_SEMANTIC_CLOSURE
        if semantic_resume is not None
        else EXECUTION_PHASE_COMPILER_REPAIR
    )
    case_status = str(cycle_state.get("status") or "")
    pending_phases = {
        validate_execution_phase(value) for value in pending_manifest_phases
    }
    resumable = migration_state_is_resumable(cycle_state)

    if not resumable and (semantic_resume is not None or pending_phases):
        raise ValueError(
            "durable resume state conflicts with a non-resumable "
            f"cycle state: {case_status or '<missing>'}"
        )

    if not resumable:
        action = "start_new_migration"
        after_recovery = None
    elif case_status == "transplant_applied_pending_validation":
        if semantic_resume is not None or pending_phases:
            raise ValueError(
                "post-transplant state conflicts with later repair state"
            )
        action = "resume_after_transplant"
        after_recovery = None
    elif semantic_resume is not None:
        semantic_action = str(
            semantic_resume.get("next_action") or ""
        )
        if not semantic_action:
            raise ValueError("semantic resume state has no next action")
        if pending_phases:
            if semantic_action != "resume_compile_recovery":
                raise ValueError(
                    "an interrupted compiler round exists outside semantic "
                    "compile recovery"
                )
            if pending_phases != {EXECUTION_PHASE_SEMANTIC_CLOSURE}:
                raise ValueError(
                    "semantic repair journals have a mismatched phase"
                )
            action = "recover_interrupted_repair_round"
            after_recovery = "resume_semantic_state"
        else:
            action = "resume_semantic_state"
            after_recovery = None
    elif pending_phases:
        if pending_phases != {phase}:
            raise ValueError(
                "repair journals do not match the active execution phase"
            )
        action = "recover_interrupted_repair_round"
        after_recovery = "continue_from_validation"
    else:
        action = "validate_current_tree"
        after_recovery = None

    semantic_action = str(
        (semantic_resume or {}).get("next_action") or ""
    ) or None
    return {
        "schema": WORKFLOW_RESUME_SCHEMA,
        "action": action,
        "after_recovery": after_recovery,
        "case_status": case_status,
        "execution_phase": phase,
        "semantic_action": semantic_action,
        "pending_manifest_phases": sorted(pending_phases),
        "resumable": resumable,
    }
