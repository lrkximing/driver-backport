#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from driver_backport.core.common import load_json, utc_now, write_json


CASE_STATES = {
    "initialized",
    "planning",
    "needs_build_preparation",
    "needs_transplant",
    "transplant_apply_failed",
    "transplant_apply_blocked",
    "transplant_applied_pending_validation",
    "repair_applied_pending_validation",
    "patch_apply_incomplete",
    "needs_patching",
    "ready_for_manual_test",
}

CASE_TRANSITIONS = {
    "initialized": {"planning"},
    "planning": {
        "needs_build_preparation",
        "needs_transplant",
        "needs_patching",
        "ready_for_manual_test",
        "patch_apply_incomplete",
    },
    "needs_build_preparation": {"planning"},
    "needs_transplant": {
        "needs_build_preparation",
        "transplant_applied_pending_validation",
        "transplant_apply_failed",
        "transplant_apply_blocked",
        "patch_apply_incomplete",
        "planning",
    },
    "transplant_applied_pending_validation": {
        "patch_apply_incomplete",
        "planning",
    },
    "transplant_apply_failed": {"planning"},
    "transplant_apply_blocked": {"planning"},
    "needs_patching": {
        "repair_applied_pending_validation",
        "patch_apply_incomplete",
        "ready_for_manual_test",
        "planning",
    },
    "repair_applied_pending_validation": {"needs_patching", "planning"},
    "patch_apply_incomplete": {
        "transplant_applied_pending_validation",
        "repair_applied_pending_validation",
        "needs_patching",
        "planning",
    },
    "ready_for_manual_test": {"planning"},
}

def _validate_case_state(value: str) -> str:
    state = str(value or "initialized")
    if state not in CASE_STATES:
        raise ValueError(f"invalid case state: {state}")
    return state


def _validate_case_transition(current: str, new: str) -> str:
    target = _validate_case_state(new)
    source = _validate_case_state(current)
    if source == target:
        return target
    allowed = CASE_TRANSITIONS.get(source, set())
    if target not in allowed:
        raise ValueError(f"invalid case state transition: {source} -> {target}")
    return target


@dataclass
class CycleState:
    case_id: str
    cycle: int = 0
    status: str = "initialized"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    current_cycle_dir: str | None = None
    last_transplant_plan_path: str | None = None
    last_validation_path: str | None = None
    open_work_units: list[str] = field(default_factory=list)
    pending_validation_work_units: list[str] = field(default_factory=list)

    @classmethod
    def initial(cls, case_id: str) -> "CycleState":
        return cls(case_id=case_id)

    @classmethod
    def from_file(cls, path: Path) -> "CycleState":
        raw = load_json(path)
        return cls(
            case_id=str(raw["case_id"]),
            cycle=int(raw.get("cycle") or 0),
            status=_validate_case_state(str(raw.get("status") or "initialized")),
            created_at=str(raw.get("created_at") or utc_now()),
            updated_at=str(raw.get("updated_at") or utc_now()),
            current_cycle_dir=raw.get("current_cycle_dir"),
            last_transplant_plan_path=raw.get("last_transplant_plan_path"),
            last_validation_path=raw.get("last_validation_path"),
            open_work_units=list(raw.get("open_work_units") or []),
            pending_validation_work_units=list(
                raw.get("pending_validation_work_units") or []
            ),
        )

    def begin_cycle(self, cycle_dir: Path) -> int:
        self.cycle += 1
        self.status = _validate_case_transition(self.status, "planning")
        self.current_cycle_dir = str(cycle_dir.resolve())
        self.updated_at = utc_now()
        return self.cycle

    def finish_cycle(
        self,
        *,
        status: str,
        transplant_plan_path: Path | None = None,
        validation_path: Path | None = None,
        open_work_units: list[str] | None = None,
        pending_validation_work_units: list[str] | None = None,
    ) -> None:
        self.status = _validate_case_transition(self.status, status)
        self.last_transplant_plan_path = (
            str(transplant_plan_path.resolve()) if transplant_plan_path else self.last_transplant_plan_path
        )
        self.last_validation_path = str(validation_path.resolve()) if validation_path else self.last_validation_path
        if open_work_units is not None:
            self.open_work_units = list(dict.fromkeys(open_work_units))
        if pending_validation_work_units is not None:
            self.pending_validation_work_units = list(
                dict.fromkeys(pending_validation_work_units)
            )
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "cycle": self.cycle,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_cycle_dir": self.current_cycle_dir,
            "last_transplant_plan_path": self.last_transplant_plan_path,
            "last_validation_path": self.last_validation_path,
            "open_work_units": list(self.open_work_units),
            "pending_validation_work_units": list(
                self.pending_validation_work_units
            ),
        }

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())
