#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Iterable

from driver_backport.core.common import normalize_relpath


def work_unit_id(unit: dict[str, Any]) -> str:
    return str(unit.get("work_unit_id") or unit.get("id") or "")


def work_unit_dependencies(unit: dict[str, Any]) -> set[str]:
    return {
        str(item)
        for item in unit.get("depends_on_work_unit_ids") or []
        if str(item)
    }


def declared_write_files(unit: dict[str, Any]) -> set[str]:
    files: list[object] = [
        *list(unit.get("allowed_fileset") or []),
        *list(unit.get("escalation_candidates") or []),
        *list(unit.get("scope_expansion_candidates") or []),
        *list(unit.get("entity_materialization_files") or []),
    ]
    for target in unit.get("patch_targets") or []:
        if isinstance(target, dict):
            files.append(target.get("file"))
    patch_target = unit.get("patch_target")
    if isinstance(patch_target, dict):
        files.append(patch_target.get("file"))
    return {
        normalize_relpath(str(item or ""))
        for item in files
        if normalize_relpath(str(item or ""))
    }


def _dependency_components(
    units: Iterable[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, set[str]],
    list[list[str]],
    dict[str, int],
]:
    ordered_units = [unit for unit in units if work_unit_id(unit)]
    unit_by_id = {work_unit_id(unit): unit for unit in ordered_units}
    position = {
        work_unit_id(unit): index
        for index, unit in enumerate(ordered_units)
    }
    dependencies = {
        unit_id: work_unit_dependencies(unit) & set(unit_by_id)
        for unit_id, unit in unit_by_id.items()
    }

    index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(unit_id: str) -> None:
        nonlocal index
        indexes[unit_id] = index
        lowlinks[unit_id] = index
        index += 1
        stack.append(unit_id)
        on_stack.add(unit_id)

        for dependency_id in sorted(
            dependencies.get(unit_id, set()),
            key=lambda item: (position.get(item, 0), item),
        ):
            if dependency_id not in indexes:
                visit(dependency_id)
                lowlinks[unit_id] = min(
                    lowlinks[unit_id],
                    lowlinks[dependency_id],
                )
            elif dependency_id in on_stack:
                lowlinks[unit_id] = min(
                    lowlinks[unit_id],
                    indexes[dependency_id],
                )

        if lowlinks[unit_id] != indexes[unit_id]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == unit_id:
                break
        component.sort(key=lambda item: (position.get(item, 0), item))
        components.append(component)

    for unit_id in sorted(unit_by_id, key=lambda item: (position[item], item)):
        if unit_id not in indexes:
            visit(unit_id)
    components.sort(
        key=lambda component: min(
            (position.get(unit_id, 0), unit_id)
            for unit_id in component
        )
    )
    return unit_by_id, dependencies, components, position


def dependency_order(
    units: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unit_by_id, dependencies, components, position = _dependency_components(units)
    component_by_unit = {
        unit_id: component_index
        for component_index, component in enumerate(components)
        for unit_id in component
    }
    component_dependencies: dict[int, set[int]] = defaultdict(set)
    component_dependents: dict[int, set[int]] = defaultdict(set)
    for unit_id, dependency_ids in dependencies.items():
        owner_component = component_by_unit[unit_id]
        for dependency_id in dependency_ids:
            dependency_component = component_by_unit[dependency_id]
            if dependency_component == owner_component:
                continue
            component_dependencies[owner_component].add(dependency_component)
            component_dependents[dependency_component].add(owner_component)

    component_key = {
        component_index: min(
            (position.get(unit_id, 0), unit_id)
            for unit_id in component
        )
        for component_index, component in enumerate(components)
    }
    pending_dependencies = {
        component_index: set(component_dependencies.get(component_index, set()))
        for component_index in range(len(components))
    }
    ready = sorted(
        (
            component_index
            for component_index, pending in pending_dependencies.items()
            if not pending
        ),
        key=lambda item: component_key[item],
    )
    component_order: list[int] = []
    while ready:
        component_index = ready.pop(0)
        component_order.append(component_index)
        for dependent in sorted(
            component_dependents.get(component_index, set()),
            key=lambda item: component_key[item],
        ):
            pending_dependencies[dependent].discard(component_index)
            if not pending_dependencies[dependent] and dependent not in component_order and dependent not in ready:
                ready.append(dependent)
        ready.sort(key=lambda item: component_key[item])

    if len(component_order) != len(components):
        raise RuntimeError("dependency SCC condensation must be acyclic")
    ordered_ids = [
        unit_id
        for component_index in component_order
        for unit_id in components[component_index]
    ]
    cyclic_components = [
        component
        for component in components
        if len(component) > 1
        or any(unit_id in dependencies.get(unit_id, set()) for unit_id in component)
    ]
    return [unit_by_id[unit_id] for unit_id in ordered_ids], {
        "schema": "dependency_schedule_v1",
        "unit_count": len(unit_by_id),
        "dependency_edge_count": sum(len(items) for items in dependencies.values()),
        "component_count": len(components),
        "cyclic_component_count": len(cyclic_components),
        "cyclic_components": cyclic_components,
        "ordered_work_unit_ids": ordered_ids,
    }


def select_dependency_ready_frontier(
    units: Iterable[dict[str, Any]],
    *,
    eligible: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Select every eligible SCC whose external predecessors are absent."""

    ordered, schedule = dependency_order(units)
    unit_by_id, dependencies, components, _position = (
        _dependency_components(ordered)
    )
    component_by_unit = {
        unit_id: component_index
        for component_index, component in enumerate(components)
        for unit_id in component
    }
    component_dependencies: dict[int, set[int]] = defaultdict(set)
    for unit_id, dependency_ids in dependencies.items():
        owner_component = component_by_unit[unit_id]
        for dependency_id in dependency_ids:
            dependency_component = component_by_unit[dependency_id]
            if dependency_component != owner_component:
                component_dependencies[owner_component].add(
                    dependency_component
                )

    eligible_ids = {
        unit_id
        for unit_id, unit in unit_by_id.items()
        if eligible is None or eligible(unit)
    }
    ready_components = {
        component_index
        for component_index, component in enumerate(components)
        if all(unit_id in eligible_ids for unit_id in component)
        and not component_dependencies.get(component_index)
    }
    selected_ids = {
        unit_id
        for component_index in ready_components
        for unit_id in components[component_index]
    }
    selected = [
        unit for unit in ordered if work_unit_id(unit) in selected_ids
    ]
    selected_ids = {work_unit_id(unit) for unit in selected}
    deferred_units = [
        unit for unit in ordered if work_unit_id(unit) not in selected_ids
    ]
    selected_files = {
        path
        for unit in selected
        for path in declared_write_files(unit)
    }
    return {
        "schema": "dependency_ready_frontier_v1",
        "selected_units": selected,
        "selected_work_unit_ids": [
            work_unit_id(unit) for unit in selected
        ],
        "selected_write_files": sorted(selected_files),
        "deferred": [
            {
                "work_unit_id": work_unit_id(unit),
                "reason": (
                    "not_eligible_for_current_stage"
                    if work_unit_id(unit) not in eligible_ids
                    else "awaiting_fresh_validation_of_dependencies"
                ),
                "blocking_work_unit_ids": sorted(
                    dependency_id
                    for dependency_id in dependencies.get(
                        work_unit_id(unit), set()
                    )
                    if component_by_unit.get(dependency_id)
                    != component_by_unit.get(work_unit_id(unit))
                ),
                "declared_write_files": sorted(
                    declared_write_files(unit)
                ),
            }
            for unit in deferred_units
        ],
        "schedule": schedule,
        "summary": {
            "selected_count": len(selected),
            "deferred_count": len(deferred_units),
            "write_file_count": len(selected_files),
            "validation_after_frontier": True,
            "scc_atomic": True,
        },
    }
