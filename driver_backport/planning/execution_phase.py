#!/usr/bin/env python3
from __future__ import annotations

from typing import Any


EXECUTION_PHASE_INITIAL_COMPATIBILITY = "initial_compatibility"
EXECUTION_PHASE_COMPILER_REPAIR = "compiler_repair"
EXECUTION_PHASE_SEMANTIC_CLOSURE = "semantic_closure"
EXECUTION_PHASES = {
    EXECUTION_PHASE_INITIAL_COMPATIBILITY,
    EXECUTION_PHASE_COMPILER_REPAIR,
    EXECUTION_PHASE_SEMANTIC_CLOSURE,
}

_SEMANTIC_CONTROL_FIELDS = (
    "semantic_gap_ids",
    "required_semantic_gap_ids",
    "pending_semantic_gap_ids",
    "blocked_semantic_gap_ids",
    "semantic_transaction_ids",
    "semantic_transactions",
    "semantic_recovery_context",
)

def validate_execution_phase(value: object) -> str:
    phase = str(value or "").strip()
    if phase not in EXECUTION_PHASES:
        raise ValueError(f"unknown repair execution phase: {phase or '<missing>'}")
    return phase


def has_semantic_control(unit: dict[str, Any]) -> bool:
    return any(bool(unit.get(field)) for field in _SEMANTIC_CONTROL_FIELDS)


def unit_execution_phase(
    unit: dict[str, Any],
    *,
    default: str = EXECUTION_PHASE_COMPILER_REPAIR,
) -> str:
    raw_phase = unit.get("execution_phase")
    if raw_phase:
        phase = validate_execution_phase(raw_phase)
    else:
        phase = validate_execution_phase(default)
    if (
        phase != EXECUTION_PHASE_SEMANTIC_CLOSURE
        and has_semantic_control(unit)
    ):
        raise ValueError(
            "semantic control requires execution_phase=semantic_closure"
        )
    return phase


def bind_execution_phase(
    unit: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    bound_phase = validate_execution_phase(phase)
    existing_phase = str(unit.get("execution_phase") or "").strip()
    if existing_phase and validate_execution_phase(existing_phase) != bound_phase:
        raise ValueError(
            "work unit cannot cross repair execution phases: "
            f"{existing_phase} -> {bound_phase}"
        )
    bound = dict(unit)
    bound["execution_phase"] = bound_phase
    unit_execution_phase(bound)
    return bound


def semantic_controls_enabled(unit: dict[str, Any]) -> bool:
    return (
        unit_execution_phase(unit)
        == EXECUTION_PHASE_SEMANTIC_CLOSURE
        and has_semantic_control(unit)
    )


def repair_plan_execution_phase(
    repair_plan: dict[str, Any],
) -> str:
    """Return the one phase bound to a repair plan and all of its units."""

    top_level_phase = str(repair_plan.get("execution_phase") or "").strip()
    unit_phases = {
        unit_execution_phase(
            unit,
            default=top_level_phase or EXECUTION_PHASE_COMPILER_REPAIR,
        )
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
    }
    if len(unit_phases) > 1:
        raise ValueError(
            "one repair plan cannot mix execution phases: "
            + ", ".join(sorted(unit_phases))
        )
    phase = (
        next(iter(unit_phases))
        if unit_phases
        else validate_execution_phase(
            top_level_phase or EXECUTION_PHASE_COMPILER_REPAIR
        )
    )
    if top_level_phase and validate_execution_phase(top_level_phase) != phase:
        raise ValueError(
            "repair-plan execution phase does not match its work units"
        )
    return phase


def repair_phase_budget_status(
    *,
    execution_phase: str,
    compile_rounds_completed: int,
    semantic_rounds_completed: int,
    max_compile_rounds: int,
    max_semantic_rounds: int,
) -> dict[str, Any]:
    """Project validated manifests onto the budget owned by one phase."""

    phase = validate_execution_phase(execution_phase)
    if phase == EXECUTION_PHASE_COMPILER_REPAIR:
        completed = max(0, int(compile_rounds_completed))
        limit = max(0, int(max_compile_rounds))
        budget_name = "compile_repair"
    elif phase == EXECUTION_PHASE_SEMANTIC_CLOSURE:
        completed = max(0, int(semantic_rounds_completed))
        limit = max(0, int(max_semantic_rounds))
        budget_name = "semantic_repair"
    else:
        raise ValueError(
            f"repair budget is undefined for execution phase: {phase}"
        )
    return {
        "execution_phase": phase,
        "budget_name": budget_name,
        "completed": completed,
        "limit": limit,
        "remaining": max(0, limit - completed),
        "available": completed < limit,
    }
