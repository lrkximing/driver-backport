from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from driver_backport.core.common import write_json
from driver_backport.core.content_store import (
    load_content_record,
    put_content_record,
)


PLAN_CONTENT_REFS_KEY = "content_record_refs"
PLAN_CONTENT_FIELDS = (
    "diagnostic_graph",
    "code_graph_scope",
    "context_expansion",
    "repair_source_pack",
)


def _stored_plan(
    state_root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    stored = dict(plan)
    stored_units: list[Any] = []
    for raw_unit in plan.get("work_unit_plans") or []:
        if not isinstance(raw_unit, dict):
            stored_units.append(raw_unit)
            continue
        unit = dict(raw_unit)
        references = dict(unit.get(PLAN_CONTENT_REFS_KEY) or {})
        for field in PLAN_CONTENT_FIELDS:
            if field not in unit:
                continue
            if unit[field] in (None, [], {}):
                references.pop(field, None)
                continue
            references[field] = put_content_record(
                state_root,
                kind=f"repair-plan-{field}",
                record=unit.pop(field),
            )
        if references:
            unit[PLAN_CONTENT_REFS_KEY] = references
        stored_units.append(unit)
    stored["work_unit_plans"] = stored_units
    return stored


def write_repair_plan(
    path: Path,
    plan: dict[str, Any],
    *,
    state_root: Path,
) -> None:
    """Write a repair plan while interning immutable evidence subrecords."""

    write_json(path, _stored_plan(state_root, plan))


def load_repair_plan(
    path: Path,
    *,
    state_root: Path,
) -> dict[str, Any]:
    """Load a full plan from either legacy inline JSON or content refs."""

    plan = json.loads(path.read_text(encoding="utf-8"))
    units: list[Any] = []
    for raw_unit in plan.get("work_unit_plans") or []:
        if not isinstance(raw_unit, dict):
            units.append(raw_unit)
            continue
        unit = dict(raw_unit)
        references = unit.pop(PLAN_CONTENT_REFS_KEY, {})
        if not isinstance(references, dict):
            raise ValueError(
                f"repair plan has invalid content refs: {path}"
            )
        for field, reference in references.items():
            if field not in PLAN_CONTENT_FIELDS:
                raise ValueError(
                    f"repair plan has unsupported content field: {field}"
                )
            unit[field] = load_content_record(
                state_root,
                str(reference),
            )
        units.append(unit)
    plan["work_unit_plans"] = units
    return plan
