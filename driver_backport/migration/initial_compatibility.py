#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from driver_backport.analysis.code_index import (
    parse_structural_entity_file,
)
from driver_backport.analysis.conditional_contract import (
    conditional_variant_family,
)
from driver_backport.analysis.cross_version_contract import (
    INITIAL_COMPATIBILITY_ANALYSIS_SCHEMA,
    INITIAL_COMPATIBILITY_ANALYSIS_REVISION,
    _static_import_entity_is_supported,
    analyze_initial_compatibility,
    compact_demand_ledger_record,
)
from driver_backport.analysis.export_index import index_export_contracts
from driver_backport.analysis.entity_integration import (
    INTEGRATION_ANALYSIS_REVISION,
    discover_integration_obligations,
    enforce_post_materialization_validation_requirement,
)
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    ensure_dir,
    normalize_relpath,
    stable_id,
    utc_now,
    write_json,
)
from driver_backport.core.content_store import put_content_record
from driver_backport.core.executor import (
    backup_file,
    finalize_transaction,
    rollback_transaction,
    start_transaction,
)
from driver_backport.llm.model_util import model_api_error_metadata
from driver_backport.planning.initial_compatibility_unit import (
    INITIAL_COMPATIBILITY_FRONTIER_REVISION,
    INITIAL_COMPATIBILITY_PLAN_SCHEMA,
    build_initial_compatibility_plan,
    refresh_initial_unit_from_current_tree,
    unit_contract_evidence,
)
from driver_backport.planning.plan_store import (
    load_repair_plan,
    write_repair_plan,
)
from driver_backport.planning.dependency_schedule import (
    dependency_order,
    declared_write_files,
    work_unit_dependencies,
    work_unit_id,
)
from driver_backport.planning.prompting import (
    build_initial_compatibility_prompt,
)
from driver_backport.planning.agentic_repair import (
    build_exact_entity_materialization_action,
)
from driver_backport.planning.execution_phase import (
    EXECUTION_PHASE_INITIAL_COMPATIBILITY,
)
from driver_backport.planning.response_apply import (
    apply_plan_responses,
    apply_single_generated_diff,
)
ModelRunner = Callable[[str], tuple[str, dict[str, Any]]]
INITIAL_PREPARATION_SCHEMA = "initial_compatibility_preparation_v3"
INITIAL_INCREMENTAL_STATE_SCHEMA = (
    "initial_compatibility_incremental_state_v9"
)
INITIAL_ATTEMPTS_SCHEMA = "initial_compatibility_unit_attempts_v6"
INITIAL_UNIT_REPLAN_LIMIT = 1
INITIAL_RETRY_DISPOSITIONS = {
    "completed",
    "retry_pending",
    "exhausted",
    "not_retryable",
}


def _cycle_number(cycle_dir: Path) -> int:
    name = cycle_dir.resolve().name
    if name.startswith("cycle-") and name.split("-", 1)[1].isdigit():
        return int(name.split("-", 1)[1])
    return 0


def _unit_mode(unit: dict[str, Any]) -> str:
    return str((unit.get("repair_decision") or {}).get("mode") or "")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_cycle_repair_plan(
    cycle_dir: Path,
    path: Path,
) -> dict[str, Any]:
    resolved_cycle = cycle_dir.resolve()
    if resolved_cycle.parent.name != "cycles":
        raise ValueError(
            f"initial compatibility cycle is outside state_root/cycles: {cycle_dir}"
        )
    if not path.is_file():
        return {}
    return load_repair_plan(
        path,
        state_root=resolved_cycle.parent.parent,
    )


def _analysis_is_current(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("schema") == INITIAL_COMPATIBILITY_ANALYSIS_SCHEMA
        and str(payload.get("analysis_revision") or "")
        == INITIAL_COMPATIBILITY_ANALYSIS_REVISION
    )


def _plan_is_current(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("schema") == INITIAL_COMPATIBILITY_PLAN_SCHEMA
        and payload.get("source_analysis_schema")
        == INITIAL_COMPATIBILITY_ANALYSIS_SCHEMA
        and payload.get("difference_unit_revision")
        == INITIAL_COMPATIBILITY_FRONTIER_REVISION
    )


def _attempt_payload_is_current(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("schema") == INITIAL_ATTEMPTS_SCHEMA
        and payload.get("analysis_revision")
        == INITIAL_COMPATIBILITY_ANALYSIS_REVISION
        and payload.get("difference_unit_revision")
        == INITIAL_COMPATIBILITY_FRONTIER_REVISION
        and payload.get("plan_schema")
        == INITIAL_COMPATIBILITY_PLAN_SCHEMA
        and all(
            isinstance(attempt, dict)
            and attempt.get("retry_disposition")
            in INITIAL_RETRY_DISPOSITIONS
            for attempt in payload.get("attempts") or []
        )
    )


def _load_prepared_artifacts(
    output_dir: Path,
    *,
    case_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = _load_json_object(output_dir / "analysis.json")
    plan = _load_cycle_repair_plan(
        output_dir.parent,
        output_dir / "initial_compatibility_plan.json",
    )
    preparation = _load_json_object(
        output_dir / "preparation_result.json"
    )
    if (
        not _analysis_is_current(analysis)
        or not _plan_is_current(plan)
        or str(analysis.get("case_id") or "") != case_id
        or str(plan.get("case_id") or "") != case_id
        or preparation.get("schema") != INITIAL_PREPARATION_SCHEMA
        or str(preparation.get("case_id") or "") != case_id
        or str(preparation.get("status") or "") != "prepared"
    ):
        raise RuntimeError(
            "current initial compatibility preparation artifacts are required "
            "before unit execution"
        )
    return analysis, plan


def _load_incremental_state(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    return (
        payload
        if (
            payload.get("schema") == INITIAL_INCREMENTAL_STATE_SCHEMA
            and payload.get("difference_unit_revision")
            == INITIAL_COMPATIBILITY_FRONTIER_REVISION
        )
        else {}
    )


def _source_changes_from_apply_results(
    apply_results: Iterable[dict[str, Any]],
) -> set[str]:
    changed: set[str] = set()
    for result in apply_results:
        if str(result.get("status") or "") != "applied":
            continue
        changed.update(
            normalize_relpath(str(path))
            for path in result.get("touched_files") or []
            if str(path).strip()
        )
        if result.get("file"):
            changed.add(normalize_relpath(str(result["file"])))
    return changed


def _unit_editable_source_files(
    unit: dict[str, Any],
    *,
    allow_escalation: bool,
) -> set[str]:
    values = [
        *(unit.get("allowed_fileset") or []),
        *(unit.get("candidate_files") or []),
    ]
    if allow_escalation:
        values.extend(unit.get("escalation_candidates") or [])
    return {
        normalize_relpath(str(path))
        for path in values
        if str(path).strip()
        and Path(str(path)).suffix.lower() in {".c", ".h", ".cc", ".cpp", ".hpp"}
    }


def _unit_authorized_files(
    unit: dict[str, Any],
    *,
    allow_escalation: bool,
) -> set[str]:
    files = list(unit.get("allowed_fileset") or [])
    if allow_escalation:
        files.extend(unit.get("escalation_candidates") or [])
    return {
        normalize_relpath(str(path))
        for path in files
        if str(path).strip()
    }


def _structural_uniqueness_counts(report: dict[str, Any]) -> dict[tuple[Any, ...], int]:
    counts: dict[tuple[Any, ...], int] = {}
    for entity in report.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        kind = str(entity.get("kind") or "")
        is_definition = bool(entity.get("is_definition"))
        if kind not in {"field", "enum_constant"} and not (
            kind in {"function", "type", "variable"}
            and is_definition
        ):
            continue
        container = entity.get("container")
        container_key = (
            str(container.get("container_kind") or ""),
            str(container.get("container_name") or ""),
        ) if isinstance(container, dict) else ("", "")
        key = (
            kind,
            str(entity.get("contract_key") or entity.get("key") or ""),
            is_definition,
            container_key,
            tuple(str(item) for item in entity.get("conditional_context") or []),
        )
        counts[key] = counts.get(key, 0) + 1
    return counts


def _structural_error_profile(report: dict[str, Any]) -> Counter[tuple[Any, ...]]:
    return Counter(
        (
            str(node.get("node_type") or ""),
            bool(node.get("is_error")),
            bool(node.get("is_missing")),
        )
        for node in report.get("error_nodes") or []
        if isinstance(node, dict)
    )


def _changed_line_ranges(
    changed_ranges: Iterable[dict[str, Any]],
    *,
    side: str,
) -> dict[str, list[tuple[int, int]]]:
    by_file: dict[str, list[tuple[int, int]]] = {}
    for changed in changed_ranges:
        if not isinstance(changed, dict):
            continue
        relative_path = normalize_relpath(str(changed.get("file") or ""))
        start = int(
            changed.get(f"{side}_start")
            or changed.get("start_line")
            or 0
        )
        count = int(changed.get(f"{side}_count") or 0)
        if count <= 0:
            continue
        end = int(
            changed.get(f"{side}_end")
            or changed.get("end_line")
            or (start + count - 1 if count > 0 else start)
        )
        if relative_path and start > 0 and end >= start:
            by_file.setdefault(relative_path, []).append((start, end))
    return by_file


def _structural_contract_inventory(
    report: dict[str, Any],
    *,
    relative_path: str,
    ranges: Iterable[tuple[int, int]],
) -> list[dict[str, Any]]:
    selected_ranges = list(ranges)
    if not selected_ranges:
        return []
    records: list[dict[str, Any]] = []
    for entity in report.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        extent = (
            entity.get("extent")
            if isinstance(entity.get("extent"), dict)
            else {}
        )
        start = int(extent.get("start_line") or 0)
        end = int(extent.get("end_line") or start)
        if start <= 0 or not any(
            max(start, range_start) <= min(end, range_end)
            for range_start, range_end in selected_ranges
        ):
            continue
        container = (
            dict(entity.get("container") or {})
            if isinstance(entity.get("container"), dict)
            else {}
        )
        records.append(
            {
                "file": relative_path,
                "contract_key": str(
                    entity.get("contract_key")
                    or entity.get("key")
                    or ""
                ),
                "kind": str(
                    entity.get("kind")
                    or entity.get("entity_kind")
                    or ""
                ),
                "symbol": str(
                    entity.get("symbol")
                    or entity.get("name")
                    or ""
                ),
                "is_definition": bool(entity.get("is_definition")),
                "container": {
                    "container_kind": str(
                        container.get("container_kind") or ""
                    ),
                    "container_name": str(
                        container.get("container_name") or ""
                    ),
                },
                "conditional_context": [
                    str(item)
                    for item in entity.get("conditional_context") or []
                ],
                "extent": {
                    "start_line": start,
                    "end_line": end,
                },
            }
        )
    return records


def _structural_contract_identity(
    record: dict[str, Any],
) -> tuple[Any, ...]:
    container = (
        record.get("container")
        if isinstance(record.get("container"), dict)
        else {}
    )
    return (
        str(record.get("contract_key") or ""),
        str(record.get("kind") or ""),
        str(record.get("symbol") or ""),
        bool(record.get("is_definition")),
        str(container.get("container_kind") or ""),
        str(container.get("container_name") or ""),
        tuple(
            str(item)
            for item in record.get("conditional_context") or []
        ),
    )


def _contract_inventory_delta(
    before: Iterable[dict[str, Any]],
    after: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    before_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    after_by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in before:
        before_by_identity.setdefault(
            _structural_contract_identity(record), []
        ).append(dict(record))
    for record in after:
        after_by_identity.setdefault(
            _structural_contract_identity(record), []
        ).append(dict(record))

    removed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    for identity in sorted(
        set(before_by_identity) | set(after_by_identity),
        key=repr,
    ):
        before_records = before_by_identity.get(identity) or []
        after_records = after_by_identity.get(identity) or []
        if len(before_records) > len(after_records):
            removed.extend(before_records[len(after_records) :])
        elif len(after_records) > len(before_records):
            added.extend(after_records[len(before_records) :])
    return {"removed": removed, "added": added}


def _capture_structural_baseline(
    spec: CaseSpec,
    relative_paths: Iterable[object],
    *,
    changed_ranges: Iterable[dict[str, Any]] = (),
) -> dict[str, dict[str, Any]]:
    baseline: dict[str, dict[str, Any]] = {}
    old_ranges = _changed_line_ranges(
        changed_ranges,
        side="old",
    )
    for value in relative_paths:
        relative_path = normalize_relpath(str(value or ""))
        if not relative_path:
            continue
        try:
            report = parse_structural_entity_file(
                root=spec.old_root_path,
                relative_path=relative_path,
                tree="old",
            )
        except (AttributeError, OSError, ValueError):
            continue
        baseline[relative_path] = {
            "parse_ok": bool(report.get("parse_ok")),
            "has_error_nodes": bool(report.get("has_error_nodes")),
            "error_node_count": len(report.get("error_nodes") or []),
            "error_nodes": [
                dict(node)
                for node in report.get("error_nodes") or []
                if isinstance(node, dict)
            ],
            "uniqueness_counts": _structural_uniqueness_counts(report),
            "contract_inventory": _structural_contract_inventory(
                report,
                relative_path=relative_path,
                ranges=old_ranges.get(relative_path) or [],
            ),
        }
    return baseline


def _postcondition_is_unconfirmed(
    record: dict[str, Any],
    *,
    changed_file_set: set[str],
) -> bool:
    kind = str(record.get("kind") or "")
    if kind == "expected_contract_not_realized":
        return True
    if kind == "expected_conditional_contract_variants_not_realized":
        return str(record.get("operation") or "insert") == "insert"
    return bool(
        kind == "structural_baseline_unavailable"
        and normalize_relpath(str(record.get("file") or ""))
        not in changed_file_set
    )


def _validate_structural_postconditions(
    spec: CaseSpec,
    *,
    changed_files: Iterable[object],
    baseline: dict[str, dict[str, Any]],
    unit: dict[str, Any] | None = None,
    changed_ranges: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    evidence_records = (
        unit_contract_evidence(unit)
        if isinstance(unit, dict)
        else []
    )
    changed_file_set = {
        normalize_relpath(str(value or ""))
        for value in changed_files
        if normalize_relpath(str(value or ""))
    }
    changed_range_records = [
        dict(item)
        for item in changed_ranges
        if isinstance(item, dict)
    ]
    new_ranges = _changed_line_ranges(
        changed_range_records,
        side="new",
    )
    contract_delta = {"removed": [], "added": []}
    expected_contracts: list[dict[str, Any]] = []
    for delta in evidence_records:
        if str(delta.get("kind") or "") == "reverse_integration":
            continue
        old_entity = (
            delta.get("old_entity")
            if isinstance(delta.get("old_entity"), dict)
            else None
        )
        donor_entity = (
            delta.get("donor_entity")
            if isinstance(delta.get("donor_entity"), dict)
            else None
        )
        old_provider_missing = bool(
            old_entity is None
            or (
                str((donor_entity or {}).get("kind") or "") == "function"
                and not bool((old_entity or {}).get("is_definition"))
            )
        )
        if donor_entity is None:
            continue
        if old_provider_missing:
            expected_contracts.append(
                {
                    "contract_identity": delta.get("contract_identity"),
                    "operation": "insert",
                    "entity": donor_entity,
                    "target_file": normalize_relpath(
                        str(donor_entity.get("file") or "")
                    ),
                }
            )
            continue
        old_file = normalize_relpath(str(old_entity.get("file") or ""))
        if old_file and old_file in changed_file_set:
            expected_contracts.append(
                {
                    "contract_identity": delta.get("contract_identity"),
                    "operation": "update",
                    "entity": old_entity,
                    "donor_entity": donor_entity,
                    "target_file": old_file,
                }
            )
    source_pack = (
        unit.get("repair_source_pack")
        if isinstance(unit, dict)
        and isinstance(unit.get("repair_source_pack"), dict)
        else {}
    )
    expected_contract_files = {
        normalize_relpath(str(record.get("target_file") or ""))
        for record in expected_contracts
    }
    checked_file_set = changed_file_set | {
        path for path in expected_contract_files if path
    }
    for relative_path in sorted(checked_file_set - set(baseline)):
        violations.append(
            {
                "file": relative_path,
                "kind": "structural_baseline_unavailable",
                "reason": (
                    "a changed or required contract file cannot be validated "
                    "without a pre-apply structural baseline"
                ),
            }
        )
    for relative_path in sorted(checked_file_set):
        if not relative_path or relative_path not in baseline:
            continue
        before = baseline[relative_path]
        after = parse_structural_entity_file(
            root=spec.old_root_path,
            relative_path=relative_path,
            tree="old",
        )
        file_violations: list[dict[str, Any]] = []
        if before.get("parse_ok") and not after.get("parse_ok"):
            file_violations.append(
                {
                    "kind": "structural_parse_regression",
                    "parse_error": after.get("parse_error"),
                }
            )
        if not before.get("has_error_nodes") and after.get("has_error_nodes"):
            file_violations.append({"kind": "new_structural_error_nodes"})
        before_error_count = int(before.get("error_node_count") or 0)
        after_error_count = len(after.get("error_nodes") or [])
        if after_error_count > before_error_count:
            file_violations.append(
                {
                    "kind": "increased_structural_error_node_count",
                    "before_count": before_error_count,
                    "after_count": after_error_count,
                }
            )
        before_error_nodes = [
            node
            for node in before.get("error_nodes") or []
            if isinstance(node, dict)
        ]
        after_error_nodes = [
            node
            for node in after.get("error_nodes") or []
            if isinstance(node, dict)
        ]
        for changed in changed_range_records:
            if normalize_relpath(str(changed.get("file") or "")) != relative_path:
                continue
            old_start = int(changed.get("old_start") or 0)
            old_count = int(changed.get("old_count") or 0)
            old_end = old_start + max(0, old_count - 1)
            new_start = int(changed.get("new_start") or 0)
            new_count = int(changed.get("new_count") or 0)
            new_end = new_start + max(0, new_count - 1)

            def errors_in_range(
                nodes: Iterable[dict[str, Any]],
                *,
                start: int,
                end: int,
            ) -> list[dict[str, Any]]:
                return [
                    node
                    for node in nodes
                    if (
                        (extent := node.get("extent"))
                        and isinstance(extent, dict)
                        and int(extent.get("start_line") or 0) > 0
                        and max(
                            start,
                            int(extent.get("start_line") or 0),
                        )
                        <= min(
                            end,
                            int(
                                extent.get("end_line")
                                or extent.get("start_line")
                                or 0
                            ),
                        )
                    )
                ]

            before_local = errors_in_range(
                before_error_nodes,
                start=old_start,
                end=old_end,
            ) if old_start > 0 else []
            after_local = errors_in_range(
                after_error_nodes,
                start=new_start,
                end=new_end,
            ) if new_start > 0 else []
            before_profile = _structural_error_profile(
                {"error_nodes": before_local}
            )
            after_profile = _structural_error_profile(
                {"error_nodes": after_local}
            )
            introduced = after_profile - before_profile
            if introduced:
                file_violations.append(
                    {
                        "kind": "new_structural_error_in_changed_range",
                        "old_range": {
                            "start_line": old_start,
                            "end_line": old_end,
                        },
                        "new_range": {
                            "start_line": new_start,
                            "end_line": new_end,
                        },
                        "introduced_error_profile": [
                            {
                                "node_type": identity[0],
                                "is_error": identity[1],
                                "is_missing": identity[2],
                                "count": count,
                            }
                            for identity, count in introduced.items()
                        ],
                    }
                )
        file_contract_delta = _contract_inventory_delta(
            before.get("contract_inventory") or [],
            _structural_contract_inventory(
                after,
                relative_path=relative_path,
                ranges=new_ranges.get(relative_path) or [],
            ),
        )
        contract_delta["removed"].extend(
            file_contract_delta["removed"]
        )
        contract_delta["added"].extend(
            file_contract_delta["added"]
        )
        before_counts = dict(before.get("uniqueness_counts") or {})
        after_counts = _structural_uniqueness_counts(after)
        for key, after_count in after_counts.items():
            before_count = int(before_counts.get(key) or 0)
            if after_count <= 1 or after_count <= before_count:
                continue
            file_violations.append(
                {
                    "kind": "new_duplicate_structural_contract",
                    "entity_kind": key[0],
                    "contract_key": key[1],
                    "container": list(key[3]),
                    "conditional_context": list(key[4]),
                    "before_count": before_count,
                    "after_count": after_count,
                }
            )
        for expected_contract in expected_contracts:
            expected_entity = dict(expected_contract.get("entity") or {})
            expected_file = normalize_relpath(
                str(expected_contract.get("target_file") or "")
            )
            expected_operation = str(
                expected_contract.get("operation") or "insert"
            )
            expected_kind = str(expected_entity.get("kind") or "")
            expected_symbol = str(expected_entity.get("symbol") or "")
            expected_container = (
                expected_entity.get("container")
                if isinstance(expected_entity.get("container"), dict)
                else {}
            )
            if (
                relative_path != expected_file
                or not expected_kind
                or not expected_symbol
            ):
                continue
            conditional_expectation = next(
                (
                    record.get("conditional_contract")
                    for record in source_pack.get(
                        "donor_reference_sources"
                    )
                    or []
                    if isinstance(record, dict)
                    and record.get("source_kind")
                    == "donor_conditional_contract_variant_set"
                    and str(record.get("symbol") or "")
                    == expected_symbol
                    and isinstance(
                        record.get("conditional_contract"),
                        dict,
                    )
                ),
                None,
            )
            realized_candidates = [
                entity
                for entity in after.get("entities") or []
                if isinstance(entity, dict)
                and str(entity.get("kind") or "") == expected_kind
                and str(entity.get("symbol") or "") == expected_symbol
                and (
                    not expected_container
                    or (
                        isinstance(entity.get("container"), dict)
                        and str((entity.get("container") or {}).get("container_kind") or "")
                        == str(expected_container.get("container_kind") or "")
                        and str((entity.get("container") or {}).get("container_name") or "")
                        == str(expected_container.get("container_name") or "")
                    )
                )
            ]
            if conditional_expectation is not None:
                realized = realized_candidates
                expected_variant_count = int(
                    conditional_expectation.get("variant_count") or 0
                )
                donor_path = spec.resolve_donor(relative_path)
                old_path = spec.resolve_old(relative_path)
                actual_family = conditional_variant_family(
                    donor_lines=(
                        old_path.read_text(
                            encoding="utf-8",
                            errors="ignore",
                        ).splitlines()
                        if old_path.is_file()
                        else []
                    ),
                    old_lines=(
                        donor_path.read_text(
                            encoding="utf-8",
                            errors="ignore",
                        ).splitlines()
                        if donor_path.is_file()
                        else []
                    ),
                    target_entities=realized,
                )
                expected_roles = sorted(
                    str(record.get("role") or "")
                    for record in conditional_expectation.get("variants") or []
                    if isinstance(record, dict) and record.get("role")
                )
                realized_roles = sorted(
                    "definition"
                    if bool(entity.get("is_definition"))
                    else "declaration"
                    for entity in realized
                )
                if (
                    len(realized) != expected_variant_count
                    or actual_family is None
                    or int(actual_family.get("variant_count") or 0)
                    != expected_variant_count
                    or str(actual_family.get("signature") or "")
                    != str(conditional_expectation.get("signature") or "")
                    or (
                        expected_roles
                        and realized_roles != expected_roles
                    )
                ):
                    file_violations.append(
                        {
                            "kind": (
                                "expected_conditional_contract_variants_not_realized"
                            ),
                            "operation": expected_operation,
                            "entity_kind": expected_kind,
                            "symbol": expected_symbol,
                            "contract_identity": expected_contract.get(
                                "contract_identity"
                            ),
                            "expected_variant_count": expected_variant_count,
                            "realized_variant_count": len(realized),
                            "expected_variant_roles": expected_roles,
                            "realized_variant_roles": realized_roles,
                            "expected_signature": conditional_expectation.get(
                                "signature"
                            ),
                            "realized_family": actual_family,
                        }
                    )
            else:
                realized = [
                    entity
                    for entity in realized_candidates
                    if (
                        expected_kind not in {"function", "type"}
                        or bool(entity.get("is_definition"))
                        == (
                            bool(expected_entity.get("is_definition"))
                            if expected_operation == "update"
                            else True
                        )
                    )
                ]
            if conditional_expectation is None and len(realized) != 1:
                file_violations.append(
                    {
                        "kind": (
                            "updated_contract_not_preserved"
                            if expected_operation == "update"
                            else "expected_contract_not_realized"
                        ),
                        "operation": expected_operation,
                        "entity_kind": expected_kind,
                        "symbol": expected_symbol,
                        "contract_identity": expected_contract.get(
                            "contract_identity"
                        ),
                        "container": {
                            "container_kind": expected_container.get("container_kind"),
                            "container_name": expected_container.get("container_name"),
                        }
                        if expected_container
                        else None,
                        "match_count": len(realized),
                    }
                )
        reports.append(
            {
                "file": relative_path,
                "parse_ok": bool(after.get("parse_ok")),
                "has_error_nodes": bool(after.get("has_error_nodes")),
                "violations": file_violations,
            }
        )
        violations.extend(
            {"file": relative_path, **violation}
            for violation in file_violations
        )
    unconfirmed_obligations = [
        record
        for record in violations
        if _postcondition_is_unconfirmed(
            record,
            changed_file_set=changed_file_set,
        )
    ]
    hard_violations = [
        record
        for record in violations
        if not _postcondition_is_unconfirmed(
            record,
            changed_file_set=changed_file_set,
        )
    ]
    for report in reports:
        report_file = normalize_relpath(str(report.get("file") or ""))
        report_records = [
            {"file": report_file, **record}
            for record in report.get("violations") or []
            if isinstance(record, dict)
        ]
        report["violations"] = [
            {key: value for key, value in record.items() if key != "file"}
            for record in report_records
            if not _postcondition_is_unconfirmed(
                record,
                changed_file_set=changed_file_set,
            )
        ]
        report["unconfirmed_obligations"] = [
            {key: value for key, value in record.items() if key != "file"}
            for record in report_records
            if _postcondition_is_unconfirmed(
                record,
                changed_file_set=changed_file_set,
            )
        ]
    status = (
        "failed"
        if hard_violations
        else "unconfirmed"
        if unconfirmed_obligations
        else "passed"
        if changed_file_set or expected_contracts
        else "not_evaluated"
    )
    return {
        "status": status,
        "reason": (
            "no source change and no structurally checkable contract"
            if status == "not_evaluated"
            else (
                "patch integrity passed but demand realization remains "
                "unconfirmed"
                if status == "unconfirmed"
                else None
            )
        ),
        "checked_files": sorted(
            path for path in checked_file_set if path in baseline
        ),
        "reports": reports,
        "violations": hard_violations,
        "unconfirmed_obligations": unconfirmed_obligations,
        "contract_delta": contract_delta,
    }


def _rollback_postcondition_failures(
    spec: CaseSpec,
    apply_results: list[dict[str, Any]],
    postcondition: dict[str, Any],
) -> list[dict[str, Any]]:
    if postcondition.get("status") != "failed":
        return apply_results
    rolled_back: dict[str, dict[str, Any]] = {}
    transaction_paths = [
        str(result.get("transaction_dir") or "")
        for result in apply_results
        if str(result.get("status") or "") == "applied"
        and str(result.get("transaction_dir") or "")
    ]
    for raw_path in reversed(transaction_paths):
        if raw_path in rolled_back:
            continue
        rolled_back[raw_path] = rollback_transaction(spec, Path(raw_path))
    return [
        {
            **result,
            **(
                {
                    "status": "postcondition_failed",
                    "postcondition": postcondition,
                    "rollback": rolled_back.get(str(result.get("transaction_dir") or "")),
                }
                if str(result.get("status") or "") == "applied"
                else {}
            ),
        }
        for result in apply_results
    ]


_ACCEPTED_INITIAL_APPLY_STATUSES = {
    "applied",
    "already_applied",
    "already_applied_duplicate",
    "already_satisfied",
}

def _initial_attempt_statuses(attempt: dict[str, Any]) -> set[str]:
    return {
        str(result.get("status") or "")
        for result in attempt.get("apply_results") or []
        if isinstance(result, dict)
    }


def _initial_attempt_completed(
    attempt: dict[str, Any],
) -> bool:
    statuses = _initial_attempt_statuses(attempt)
    if not statuses or not statuses <= _ACCEPTED_INITIAL_APPLY_STATUSES:
        return False
    if statuses <= {
        "already_satisfied",
        "already_applied",
        "already_applied_duplicate",
    }:
        return True
    postcondition = attempt.get("structural_postcondition")
    postcondition_status = str(
        (postcondition or {}).get("status")
        if isinstance(postcondition, dict)
        else ""
    )
    return postcondition_status in {
        "passed",
        "enforced_by_deterministic_contract_tool",
    }


def _initial_attempt_applied_pending_confirmation(
    attempt: dict[str, Any],
) -> bool:
    statuses = _initial_attempt_statuses(attempt)
    if (
        not statuses
        or not statuses <= _ACCEPTED_INITIAL_APPLY_STATUSES
        or "applied" not in statuses
    ):
        return False
    postcondition = attempt.get("structural_postcondition")
    return bool(
        isinstance(postcondition, dict)
        and str(postcondition.get("status") or "") == "unconfirmed"
    )


def _initial_attempt_replanable(attempt: dict[str, Any]) -> bool:
    """Return whether exact apply evidence requests a fresh unit plan."""

    failures = [
        result.get("apply_failure")
        for result in attempt.get("apply_results") or []
        if isinstance(result, dict)
        and isinstance(result.get("apply_failure"), dict)
    ]
    if failures and all(
        str(failure.get("recoverability") or "")
        == "outer_unit_replan"
        for failure in failures
    ):
        return True
    return bool(
        str(
            (attempt.get("structural_postcondition") or {}).get("status")
            if isinstance(attempt.get("structural_postcondition"), dict)
            else ""
        )
        == "failed"
    )


def _initial_attempt_terminal(attempt: dict[str, Any]) -> bool:
    if _initial_attempt_completed(attempt):
        return True
    disposition = str(attempt.get("retry_disposition") or "")
    if disposition not in INITIAL_RETRY_DISPOSITIONS:
        raise ValueError(
            "current initial compatibility attempt has an invalid retry "
            "disposition"
        )
    return disposition in {"exhausted", "not_retryable"}


def _initial_retry_prompt_record(
    attempt: dict[str, Any],
) -> dict[str, Any]:
    apply_results = [
        result
        for result in attempt.get("apply_results") or []
        if isinstance(result, dict)
    ]
    touched_files = sorted(
        {
            normalize_relpath(str(path))
            for result in apply_results
            for path in result.get("touched_files") or []
            if normalize_relpath(str(path))
        }
    )
    patch_paths = [
        str(result.get("patch_path") or "")
        for result in apply_results
        if str(result.get("patch_path") or "")
    ]
    tree_change = (
        "rolled_back"
        if any(
            isinstance(result.get("rollback"), dict)
            for result in apply_results
        )
        else "not_applied"
    )
    return {
        "source": "initial_compatibility_atomic_apply",
        "work_unit_id": attempt.get("unit_id"),
        "attempt_number": int(attempt.get("attempt_number") or 1),
        "status": "atomic_apply_rejected",
        "tree_change": tree_change,
        "touched_files": touched_files,
        "patch_path": patch_paths[-1] if patch_paths else None,
        "apply_failures": [
            dict(result.get("apply_failure") or {})
            for result in apply_results
            if isinstance(result.get("apply_failure"), dict)
        ],
        "note": (
            "Regenerate the complete atomic patch from the current allowed "
            "source preimage. Preserve every demanded contract dimension; "
            "do not replay stale hunk context."
        ),
    }


def _contract_identity(unit: dict[str, Any]) -> str:
    evidence = unit_contract_evidence(unit)
    return str((evidence[0] if evidence else {}).get("contract_identity") or "")


def _unit_contract_identities(unit: dict[str, Any]) -> list[str]:
    identities = [
        str(item)
        for item in unit.get("contract_identities") or []
        if str(item)
    ]
    if not identities:
        identities = [
            str(delta.get("contract_identity") or "")
            for delta in unit_contract_evidence(unit)
            if str(delta.get("contract_identity") or "")
        ]
    if not identities and _contract_identity(unit):
        identities = [_contract_identity(unit)]
    return list(dict.fromkeys(identities))


def _write_incremental_state(
    path: Path,
    *,
    pending_changed_files: set[str],
    validation_required: bool = False,
    active_batch: dict[str, Any] | None = None,
    queue_work_unit_ids: Iterable[object] | None = None,
    resolved_work_unit_ids: Iterable[object] | None = None,
    deferred_work_units: dict[str, Any] | None = None,
    validation_history: list[dict[str, Any]] | None = None,
    refresh_generation: int | None = None,
) -> None:
    previous = _load_incremental_state(path)
    payload = {
        **previous,
        "schema": INITIAL_INCREMENTAL_STATE_SCHEMA,
        "difference_unit_revision": (
            INITIAL_COMPATIBILITY_FRONTIER_REVISION
        ),
        "updated_at": utc_now(),
        "pending_changed_files": sorted(pending_changed_files),
        "validation_required": bool(validation_required),
        "active_batch": dict(active_batch or {}),
    }
    if queue_work_unit_ids is not None:
        payload["queue_work_unit_ids"] = sorted(
            {str(item) for item in queue_work_unit_ids if str(item)}
        )
    if resolved_work_unit_ids is not None:
        payload["resolved_work_unit_ids"] = sorted(
            {str(item) for item in resolved_work_unit_ids if str(item)}
        )
    if deferred_work_units is not None:
        payload["deferred_work_units"] = dict(deferred_work_units)
    if validation_history is not None:
        payload["validation_history"] = list(validation_history)
    if refresh_generation is not None:
        payload["refresh_generation"] = max(0, int(refresh_generation))
    write_json(
        path,
        payload,
    )


def _restore_incremental_state(
    path: Path,
    *,
    attempts: list[dict[str, Any]],
) -> set[str]:
    stored = _load_incremental_state(path)
    if stored:
        return {
            normalize_relpath(str(item))
            for item in stored.get("pending_changed_files") or []
            if str(item).strip()
        }

    pending_changed_files: set[str] = set()
    for attempt in attempts:
        pending_changed_files.update(
            normalize_relpath(str(item))
            for item in attempt.get("changed_files") or []
            if str(item).strip()
        )
    return pending_changed_files


def _finalized_inflight_transaction_files(
    state_root: Path,
    *,
    case_id: str,
    work_unit_id: str,
) -> set[str]:
    changed_files: set[str] = set()
    for manifest_path in (state_root / "transactions").glob("**/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            str(manifest.get("case_id") or "") != case_id
            or str(manifest.get("work_unit_id") or "") != work_unit_id
            or str(manifest.get("status") or "") != "applied"
        ):
            continue
        changed_files.update(
            normalize_relpath(str(entry.get("relative_path") or ""))
            for entry in manifest.get("entries") or []
            if str(entry.get("type") or "") == "backup"
            and str(entry.get("relative_path") or "")
        )
    return {path for path in changed_files if path}


def _append_exact_export_to_current_provider(
    spec: CaseSpec,
    *,
    state_root: Path,
    cycle_dir: Path,
    unit_id: str,
    current_entity: dict[str, Any],
    donor_export: dict[str, Any],
    authorized_files: set[str],
) -> dict[str, Any]:
    symbol = str(current_entity.get("symbol") or "")
    relative_path = normalize_relpath(str(current_entity.get("file") or ""))
    if relative_path not in authorized_files:
        return {
            "unit_id": unit_id,
            "status": "blocked",
            "reason": "export owner is outside the authorized write set",
            "file": relative_path,
        }
    if (
        not symbol
        or not relative_path
        or normalize_relpath(str(donor_export.get("file") or "")) != relative_path
    ):
        return {
            "unit_id": unit_id,
            "status": "blocked",
            "reason": "current definition and donor export do not have one source owner",
        }

    current_exports = index_export_contracts(
        spec.old_root_path,
        [relative_path],
        tree="old",
    ).get(symbol) or []
    donor_shape = (
        str(donor_export.get("macro") or ""),
        str(donor_export.get("namespace_or_modules") or ""),
    )
    if any(
        (
            str(record.get("macro") or ""),
            str(record.get("namespace_or_modules") or ""),
        )
        == donor_shape
        for record in current_exports
    ):
        return {
            "unit_id": unit_id,
            "status": "already_satisfied",
            "file": relative_path,
            "symbol": symbol,
        }

    report = parse_structural_entity_file(
        root=spec.old_root_path,
        relative_path=relative_path,
        tree="old",
    )
    expected_contract = str(
        current_entity.get("contract_key") or current_entity.get("key") or ""
    )
    current_definitions = [
        entity
        for entity in report.get("entities") or []
        if isinstance(entity, dict)
        if bool(entity.get("is_definition"))
        and str(entity.get("symbol") or "") == symbol
        and str(entity.get("kind") or "")
        == str(current_entity.get("kind") or "")
        and (
            not expected_contract
            or str(entity.get("contract_key") or entity.get("key") or "")
            == expected_contract
        )
    ]
    if len(current_definitions) != 1:
        return {
            "unit_id": unit_id,
            "status": "blocked",
            "reason": "current old-tree definition is no longer unique",
            "definition_count": len(current_definitions),
            "file": relative_path,
            "symbol": symbol,
            "definition_query": {
                "backend": report.get("backend"),
                "parse_ok": report.get("parse_ok"),
                "parse_error": report.get("parse_error"),
            },
        }
    end_line = int((current_definitions[0].get("extent") or {}).get("end_line") or 0)
    target = spec.resolve_old(relative_path)
    lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
    if end_line <= 0 or end_line > len(lines):
        return {
            "unit_id": unit_id,
            "status": "blocked",
            "reason": "current definition extent is invalid",
            "file": relative_path,
            "symbol": symbol,
            "end_line": end_line,
        }
    raw_export = str(donor_export.get("raw") or "").strip()
    if not raw_export:
        return {
            "unit_id": unit_id,
            "status": "blocked",
            "reason": "donor export record has no exact source text",
        }

    transaction = start_transaction(
        state_root,
        case_id=spec.case_id,
        cycle=_cycle_number(cycle_dir),
        work_unit_id=stable_id("exact-export", unit_id, symbol),
    )
    backup_file(transaction, spec, relative_path)
    updated = [*lines[:end_line], raw_export, *lines[end_line:]]
    target.write_text("\n".join(updated) + "\n", encoding="utf-8")
    verified = index_export_contracts(
        spec.old_root_path,
        [relative_path],
        tree="old",
    ).get(symbol) or []
    verified_shape = any(
        (
            str(record.get("macro") or ""),
            str(record.get("namespace_or_modules") or ""),
        )
        == donor_shape
        for record in verified
    )
    if not verified_shape:
        finalize_transaction(
            transaction,
            status="failed",
            notes=["deterministic export postcondition did not hold"],
        )
        rollback = rollback_transaction(spec, transaction.root)
        return {
            "unit_id": unit_id,
            "status": "rolled_back",
            "reason": "deterministic export postcondition did not hold",
            "transaction_dir": str(transaction.root.resolve()),
            "rollback": rollback,
        }
    finalize_transaction(
        transaction,
        status="applied",
        notes=[
            "initial compatibility deterministic export",
            f"copied exact donor export contract for {symbol}",
        ],
    )
    return {
        "unit_id": unit_id,
        "status": "applied",
        "file": relative_path,
        "symbol": symbol,
        "export": raw_export,
        "transaction_dir": str(transaction.root.resolve()),
    }


def _append_export_contract(
    spec: CaseSpec,
    *,
    state_root: Path,
    cycle_dir: Path,
    unit: dict[str, Any],
    allow_escalation: bool,
) -> dict[str, Any]:
    evidence = unit_contract_evidence(unit)
    delta = evidence[0] if len(evidence) == 1 else {}
    old_entity = (
        delta.get("old_entity")
        if isinstance(delta.get("old_entity"), dict)
        else None
    )
    donor_exports = [
        record
        for record in delta.get("donor_exports") or []
        if isinstance(record, dict)
    ]
    if old_entity is None or len(donor_exports) != 1:
        return {
            "unit_id": unit.get("id"),
            "status": "blocked",
            "reason": (
                "deterministic export requires one old definition and one "
                "exact donor export"
            ),
        }
    return _append_exact_export_to_current_provider(
        spec,
        state_root=state_root,
        cycle_dir=cycle_dir,
        unit_id=str(unit.get("id") or "initial-export"),
        current_entity=old_entity,
        donor_export=donor_exports[0],
        authorized_files=_unit_authorized_files(
            unit,
            allow_escalation=allow_escalation,
        ),
    )


def _structural_matches(
    report: dict[str, Any],
    entity: dict[str, Any],
) -> list[dict[str, Any]]:
    contract = str(entity.get("contract_key") or entity.get("key") or "")
    kind = str(entity.get("kind") or entity.get("entity_kind") or "")
    return [
        record
        for record in report.get("entities") or []
        if str(record.get("contract_key") or record.get("key") or "") == contract
        and str(record.get("kind") or record.get("entity_kind") or "") == kind
    ]


def _finalize_static_materialization_integration(
    spec: CaseSpec,
    attempts: Iterable[dict[str, Any]],
    *,
    validation_units: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    validation_candidates: list[dict[str, Any]] = []
    seeds_by_contract: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        for result in attempt.get("apply_results") or []:
            if (
                not isinstance(result, dict)
                or str(result.get("status") or "") != "applied"
                or not isinstance(
                    result.get("post_materialization_seed"),
                    dict,
                )
            ):
                continue
            seed = dict(result["post_materialization_seed"])
            contract_key = str(seed.get("contract_key") or "")
            if not contract_key:
                continue
            candidates.append((result, seed))
            seeds_by_contract.setdefault(contract_key, seed)

    for unit in validation_units:
        if (
            not isinstance(unit, dict)
            or _unit_mode(unit)
            != "post_materialization_validation_only"
        ):
            continue
        unit_id = work_unit_id(unit)
        for evidence in unit_contract_evidence(unit):
            if (
                str(evidence.get("resolution") or "")
                != "post_materialization_validation_only"
                or not isinstance(
                    evidence.get("post_materialization_validation"),
                    dict,
                )
                or not isinstance(evidence.get("donor_entity"), dict)
            ):
                continue
            seed = dict(evidence["donor_entity"])
            contract_key = str(
                seed.get("contract_key") or seed.get("key") or ""
            )
            if not contract_key:
                continue
            seed["contract_key"] = contract_key
            validation_candidates.append(
                {
                    "source_work_unit_id": unit_id,
                    "contract_identity": str(
                        evidence.get("contract_identity") or ""
                    ),
                    "seed": seed,
                    "validation_requirement": dict(
                        evidence["post_materialization_validation"]
                    ),
                }
            )
            seeds_by_contract.setdefault(contract_key, seed)

    if not candidates and not validation_candidates:
        return {
            "schema": "initial_static_integration_batch_v1",
            "status": "not_required",
            "transaction_count": 0,
            "validation_unit_count": 0,
            "seed_contract_count": 0,
            "validation_records": [],
        }

    validation_requirements_by_digest: dict[
        str, dict[str, Any]
    ] = {}
    for result, _seed in candidates:
        integration_report = (
            result.get("post_patch_integration")
            if isinstance(result.get("post_patch_integration"), dict)
            else {}
        )
        requirement = (
            result.get("post_materialization_validation")
            or integration_report.get("validation_requirement")
        )
        if isinstance(requirement, dict):
            normalized = dict(requirement)
            validation_requirements_by_digest.setdefault(
                json.dumps(normalized, sort_keys=True),
                normalized,
            )
    for candidate in validation_candidates:
        requirement = candidate.get("validation_requirement")
        if isinstance(requirement, dict):
            normalized = dict(requirement)
            validation_requirements_by_digest.setdefault(
                json.dumps(normalized, sort_keys=True),
                normalized,
            )

    try:
        batch_report = discover_integration_obligations(
            spec,
            seed_entities=seeds_by_contract.values(),
            validation_requirements=(
                validation_requirements_by_digest.values()
            ),
        )
    except Exception as exc:
        batch_report = {
            "schema": "post_materialization_integration_report_v1",
            "status": "unavailable",
            "obligations": [],
            "coverage_blockers": [
                {
                    "reason": (
                        "post_materialization_integration_batch_failed"
                    ),
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "seed_contract_keys": sorted(seeds_by_contract),
                }
            ],
            "summary": {
                "seed_count": len(seeds_by_contract),
                "obligation_count": 0,
                "coverage_blocker_count": 1,
            },
        }

    obligations = [
        dict(item)
        for item in batch_report.get("obligations") or []
        if isinstance(item, dict)
    ]
    blockers = [
        dict(item)
        for item in batch_report.get("coverage_blockers") or []
        if isinstance(item, dict)
    ]

    def report_for_seed(
        seed: dict[str, Any],
        validation_requirement: dict[str, Any],
        *,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        contract_key = str(seed.get("contract_key") or "")
        result_obligations = [
            item
            for item in obligations
            if str(item.get("target_contract_key") or "")
            == contract_key
        ]
        result_blockers = [
            item
            for item in blockers
            if (
                not str(item.get("seed_contract_key") or "")
                or str(item.get("seed_contract_key") or "")
                == contract_key
            )
            and (
                not item.get("seed_contract_keys")
                or contract_key
                in {
                    str(value)
                    for value in item.get("seed_contract_keys") or []
                    if str(value)
                }
            )
        ]
        report = {
            "schema": "post_patch_integration_analysis_v1",
            "analysis_revision": INTEGRATION_ANALYSIS_REVISION,
            "status": (
                "partially_available"
                if result_obligations and result_blockers
                else "unavailable"
                if result_blockers
                else "ready"
                if result_obligations
                else "not_required"
            ),
            "revalidation_request": {
                "schema": (
                    "post_patch_integration_revalidation_request_v1"
                ),
                **dict(request or {}),
            },
            "obligations": result_obligations,
            "coverage_blockers": result_blockers,
            "frontier": batch_report.get("frontier"),
            "termination": batch_report.get("termination"),
            "batch": {
                "schema": "initial_static_integration_batch_v1",
                "seed_contract_count": len(seeds_by_contract),
            },
            "summary": {
                "seed_count": 1,
                "obligation_count": len(result_obligations),
                "unresolved_count": sum(
                    str(item.get("status") or "") != "satisfied"
                    for item in result_obligations
                ),
                "coverage_blocker_count": len(result_blockers),
            },
        }
        return enforce_post_materialization_validation_requirement(
            report,
            validation_requirement,
        )

    for result, seed in candidates:
        relative_path = normalize_relpath(str(result.get("file") or ""))
        transaction_dir = str(result.get("transaction_dir") or "")
        request = {
            "transaction_dir": transaction_dir or None,
            "touched_files": [relative_path] if relative_path else [],
        }
        prior_report = (
            result.get("post_patch_integration")
            if isinstance(result.get("post_patch_integration"), dict)
            else {}
        )
        validation_requirement = (
            dict(result["post_materialization_validation"])
            if isinstance(
                result.get("post_materialization_validation"),
                dict,
            )
            else dict(prior_report.get("validation_requirement") or {})
        )
        result["post_patch_integration"] = report_for_seed(
            seed,
            validation_requirement,
            request=request,
        )
        result.pop("post_materialization_validation", None)

    validation_records = [
        {
            "schema": (
                "initial_post_materialization_validation_record_v1"
            ),
            "source_work_unit_id": candidate[
                "source_work_unit_id"
            ],
            "contract_identity": candidate["contract_identity"],
            "seed": candidate["seed"],
            "validation_requirement": candidate[
                "validation_requirement"
            ],
            "integration_report": report_for_seed(
                candidate["seed"],
                candidate["validation_requirement"],
                request={
                    "source_work_unit_id": candidate[
                        "source_work_unit_id"
                    ],
                    "touched_files": [
                        normalize_relpath(
                            str(candidate["seed"].get("file") or "")
                        )
                    ],
                },
            ),
        }
        for candidate in validation_candidates
    ]

    return {
        "schema": "initial_static_integration_batch_v1",
        "status": str(batch_report.get("status") or "unavailable"),
        "transaction_count": len(candidates),
        "validation_unit_count": len(validation_records),
        "seed_contract_count": len(seeds_by_contract),
        "obligation_count": len(obligations),
        "coverage_blocker_count": len(blockers),
        "validation_records": validation_records,
    }


def _insert_missing_structural_contract(
    spec: CaseSpec,
    *,
    state_root: Path,
    cycle_dir: Path,
    unit: dict[str, Any],
    allow_escalation: bool,
) -> dict[str, Any]:
    """Materialize one donor-only B0 contract through the shared graph tool."""

    unit_id = str(unit.get("id") or "initial-static-import")
    evidence = unit_contract_evidence(unit)
    delta = evidence[0] if len(evidence) == 1 else {}
    donor_entity = (
        delta.get("donor_entity")
        if isinstance(delta.get("donor_entity"), dict)
        else None
    )
    contract_identity = str(delta.get("contract_identity") or "")

    def blocked(reason: str, **details: Any) -> dict[str, Any]:
        return {
            "unit_id": unit_id,
            "contract_identity": contract_identity,
            "status": "blocked",
            "reason": reason,
            **details,
        }

    if donor_entity is None or delta.get("old_entity") is not None:
        return blocked(
            "static import requires one donor entity and no old counterpart"
        )

    kind = str(donor_entity.get("kind") or "")
    relative_path = normalize_relpath(str(donor_entity.get("file") or ""))
    contract_key = str(
        donor_entity.get("contract_key")
        or donor_entity.get("key")
        or ""
    )
    entity_id = str(
        donor_entity.get("id")
        or donor_entity.get("entity_id")
        or stable_id(
            "initial-materialization-entity",
            relative_path,
            kind,
            contract_key,
        )
    )
    authorized_files = _unit_authorized_files(
        unit,
        allow_escalation=allow_escalation,
    )
    if relative_path not in authorized_files:
        return blocked(
            "contract owner is outside the authorized write set",
            file=relative_path,
        )
    if (
        not relative_path
        or not contract_key
        or not _static_import_entity_is_supported(donor_entity)
    ):
        return blocked(
            "entity is not a typed declarative static-import contract",
            file=relative_path,
        )
    if spec.classify_edit_root(relative_path) != "editable":
        return blocked("contract owner is not editable", file=relative_path)

    old_report = parse_structural_entity_file(
        root=spec.old_root_path,
        relative_path=relative_path,
        tree="old",
    )
    old_matches = _structural_matches(old_report, donor_entity)
    if len(old_matches) == 1:
        return {
            "unit_id": unit_id,
            "contract_identity": contract_identity,
            "status": "already_satisfied",
            "file": relative_path,
            "symbol": donor_entity.get("symbol"),
        }
    if not old_report.get("parse_ok") or old_matches:
        return blocked(
            (
                "current structural owner is unparseable or has an "
                "ambiguous old counterpart"
            ),
            file=relative_path,
            old_match_count=len(old_matches),
            old_parse_ok=bool(old_report.get("parse_ok")),
        )

    root_key = contract_identity or contract_key
    descriptor = {
        **donor_entity,
        "id": entity_id,
        "tree": "donor",
        "path": relative_path,
        "file": relative_path,
        "entity_kind": kind,
        "contract_key": contract_key,
    }
    preferred_translation_units = list(
        dict.fromkeys(
            normalize_relpath(
                str(
                    reference.get("translation_unit_file")
                    or reference.get("source_file")
                    or ""
                )
            )
            for reference in delta.get("consumer_references") or []
            if isinstance(reference, dict)
            and normalize_relpath(
                str(
                    reference.get("translation_unit_file")
                    or reference.get("source_file")
                    or ""
                )
            )
        )
    )
    try:
        materialization = build_exact_entity_materialization_action(
            spec,
            work_unit_id=unit_id,
            descriptor=descriptor,
            editable_files=list(unit.get("allowed_fileset") or []),
            escalation_files=(
                list(unit.get("escalation_candidates") or [])
                if allow_escalation
                else []
            ),
            root_key=root_key,
            rationale=(
                "materialize the exact donor-only initial compatibility "
                "contract"
            ),
            materialization_authority="unique_graph_root_contract",
            preferred_translation_units=preferred_translation_units,
        )
    except ValueError as exc:
        proof = getattr(exc, "proof", None)
        return blocked(
            str(exc),
            file=relative_path,
            **(
                {"dependency_closure": proof}
                if isinstance(proof, dict)
                else {}
            ),
        )

    generated_action = dict(materialization.get("action") or {})
    diff_text = str(
        (generated_action.get("patch") or {}).get("content") or ""
    )
    if not diff_text:
        return blocked(
            "shared materialization tool produced no patch",
            file=relative_path,
        )
    provider_obligations = [
        dict(item)
        for item in materialization.get(
            "post_materialization_provider_obligations"
        )
        or []
        if isinstance(item, dict)
    ]
    apply_result = apply_single_generated_diff(
        spec,
        state_root=state_root,
        artifact_dir=(
            cycle_dir
            / "initial_deterministic_apply"
            / unit_id
        ),
        plan_unit={
            **unit,
            "id": unit_id,
            "work_unit_id": unit_id,
            "unit_kind": "initial_compatibility",
            "execution_phase": (
                EXECUTION_PHASE_INITIAL_COMPATIBILITY
            ),
            "controller_entity_materialization": True,
            "entity_materialization_files": [relative_path],
            "post_materialization_provider_obligations": (
                provider_obligations
            ),
        },
        diff_text=diff_text,
        allow_escalation=allow_escalation,
    )
    if str(apply_result.get("status") or "") != "applied":
        return blocked(
            (
                "shared materialization patch was rejected by the "
                "canonical apply protocol"
            ),
            file=relative_path,
            apply_status=apply_result.get("status"),
            apply_failure=apply_result.get("apply_failure"),
            patch_path=apply_result.get("patch_path"),
            touched_files=list(
                apply_result.get("touched_files") or []
            ),
        )

    verified_report = parse_structural_entity_file(
        root=spec.old_root_path,
        relative_path=relative_path,
        tree="old",
    )
    verified_matches = _structural_matches(
        verified_report,
        donor_entity,
    )
    expected_match_count = max(
        1,
        len(materialization.get("donor_entity_ids") or []),
    )
    new_parse_errors = (
        _structural_error_profile(verified_report)
        - _structural_error_profile(old_report)
    )
    if (
        not verified_report.get("parse_ok")
        or len(verified_matches) != expected_match_count
        or bool(new_parse_errors)
    ):
        transaction_dir = Path(
            str(apply_result.get("transaction_dir") or "")
        )
        rollback = (
            rollback_transaction(spec, transaction_dir)
            if str(transaction_dir) not in {"", "."}
            else {"status": "transaction_missing"}
        )
        return {
            "unit_id": unit_id,
            "contract_identity": contract_identity,
            "status": "rolled_back",
            "reason": (
                "shared materialization structural postcondition did not hold"
            ),
            "file": relative_path,
            "expected_match_count": expected_match_count,
            "verified_match_count": len(verified_matches),
            "rollback": rollback,
        }

    transaction_dir = Path(str(apply_result["transaction_dir"]))
    validation_requirement = delta.get(
        "post_materialization_validation"
    )
    return {
        "unit_id": unit_id,
        "contract_identity": contract_identity,
        "status": "applied",
        "file": relative_path,
        "symbol": donor_entity.get("symbol"),
        "entity_kind": kind,
        "variant_count": expected_match_count,
        "materialization_strategy": materialization.get("strategy"),
        "transaction_dir": str(transaction_dir.resolve()),
        "post_materialization_validation": validation_requirement,
        "post_materialization_seed": {
            "schema": "post_patch_integration_seed_v1",
            "contract_key": contract_key,
            "symbol": donor_entity.get("symbol"),
            "kind": kind,
            "file": relative_path,
        },
    }


def _apply_deterministic_units(
    spec: CaseSpec,
    *,
    state_root: Path,
    cycle_dir: Path,
    plan: dict[str, Any],
    allow_escalation: bool,
    completed_work_unit_ids: Iterable[object] = (),
    known_work_unit_ids: Iterable[object] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    completed_unit_ids = {
        str(item) for item in completed_work_unit_ids if str(item)
    }
    known_unit_ids = (
        {str(item) for item in known_work_unit_ids if str(item)}
        if known_work_unit_ids is not None
        else {
            work_unit_id(unit)
            for unit in plan.get("work_unit_plans") or []
            if isinstance(unit, dict) and work_unit_id(unit)
        }
    )
    for unit in plan.get("work_unit_plans") or []:
        mode = _unit_mode(unit)
        if not mode.startswith("deterministic_"):
            continue
        unresolved_dependencies = sorted(
            (work_unit_dependencies(unit) & known_unit_ids)
            - completed_unit_ids
        )
        if unresolved_dependencies:
            results.append(
                {
                    "unit_id": work_unit_id(unit),
                    "contract_identity": _contract_identity(unit),
                    "status": "deferred_dependency",
                    "unresolved_work_unit_ids": unresolved_dependencies,
                }
            )
            continue
        if mode == "deterministic_export_only":
            result = _append_export_contract(
                spec,
                state_root=state_root,
                cycle_dir=cycle_dir,
                unit=unit,
                allow_escalation=allow_escalation,
            )
        elif mode == "deterministic_import_missing_contract":
            result = _insert_missing_structural_contract(
                spec,
                state_root=state_root,
                cycle_dir=cycle_dir,
                unit=unit,
                allow_escalation=allow_escalation,
            )
        else:
            continue
        results.append(result)
        if str(result.get("status") or "") in {
            "applied",
            "already_satisfied",
        }:
            completed_unit_ids.add(work_unit_id(unit))
    return {
        "schema": "initial_deterministic_apply_v1",
        "results": results,
        "summary": {
            "unit_count": len(results),
            "applied_count": sum(result.get("status") == "applied" for result in results),
            "already_satisfied_count": sum(
                result.get("status") == "already_satisfied" for result in results
            ),
            "blocked_count": sum(
                result.get("status") in {"blocked", "rolled_back"} for result in results
            ),
            "deferred_dependency_count": sum(
                result.get("status") == "deferred_dependency"
                for result in results
            ),
        },
    }


def _write_analysis_artifacts(
    output_dir: Path,
    *,
    state_root: Path,
    analysis: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    stored_analysis = {
        key: value
        for key, value in analysis.items()
        if key != "demand_ledger"
    }
    stored_ledger: list[dict[str, Any]] = []
    for raw_evidence in analysis.get("demand_ledger") or []:
        if not isinstance(raw_evidence, dict):
            raise ValueError("demand ledger records must be objects")
        evidence_ref = put_content_record(
            state_root,
            kind="initial-demand-evidence",
            record=raw_evidence,
        )
        stored_ledger.append(
            {
                **compact_demand_ledger_record(raw_evidence),
                "source_evidence_ref": evidence_ref,
            }
        )
    stored_analysis["demand_ledger"] = stored_ledger
    write_json(output_dir / "analysis.json", stored_analysis)
    write_repair_plan(
        output_dir / "initial_compatibility_plan.json",
        plan,
        state_root=state_root,
    )


def _plan_units_by_id(
    plan: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        work_unit_id(unit): dict(unit)
        for unit in plan.get("work_unit_plans") or []
        if isinstance(unit, dict) and work_unit_id(unit)
    }


def _queue_units_from_plan(plan: dict[str, Any]) -> set[str]:
    """Track every repair candidate, including structurally blocked ones.

    Context-blocked units must remain in the one-shot B0 queue so the first
    target compile can explicitly settle them as invalidated observations.
    Execution still filters them out until the missing structural context is
    available; omitting them here would instead make them disappear from the
    workflow state before that compile boundary.
    """

    return {
        work_unit_id(unit)
        for unit in plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and work_unit_id(unit)
        and (
            _unit_mode(unit) == "requires_llm"
            or _unit_mode(unit).startswith("deterministic_")
        )
    }


def _pending_queue_work_unit_ids(
    units_by_id: dict[str, dict[str, Any]],
    *,
    queue_work_unit_ids: Iterable[object],
    resolved_work_unit_ids: Iterable[object],
    deferred_work_unit_ids: Iterable[object],
    attempted_work_unit_ids: Iterable[object],
) -> set[str]:
    queue = {str(item) for item in queue_work_unit_ids if str(item)}
    resolved = {
        str(item) for item in resolved_work_unit_ids if str(item)
    }
    deferred = {
        str(item) for item in deferred_work_unit_ids if str(item)
    }
    attempted = {
        str(item) for item in attempted_work_unit_ids if str(item)
    }
    return {
        unit_id
        for unit_id in queue
        if unit_id in units_by_id
        and unit_id not in resolved
        and unit_id not in deferred
        and unit_id not in attempted
    }


def _unresolved_queue_work_unit_ids(
    units_by_id: dict[str, dict[str, Any]],
    *,
    queue_work_unit_ids: Iterable[object],
    settled_work_unit_ids: Iterable[object],
) -> set[str]:
    queue = {str(item) for item in queue_work_unit_ids if str(item)}
    settled = {
        str(item) for item in settled_work_unit_ids if str(item)
    }
    return {
        unit_id
        for unit_id in queue
        if unit_id in units_by_id and unit_id not in settled
    }


def _contract_identities_for_unit_ids(
    units_by_id: dict[str, dict[str, Any]],
    unit_ids: Iterable[object],
) -> list[str]:
    return sorted(
        {
            identity
            for unit_id in unit_ids
            for identity in _unit_contract_identities(
                units_by_id.get(str(unit_id)) or {}
            )
            if identity
        }
    )


def _attempts_from_path(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_object(path)
    if not _attempt_payload_is_current(payload):
        return []
    return [
        dict(item)
        for item in payload.get("attempts") or []
        if isinstance(item, dict)
    ]


def initial_compatibility_unit_outcomes(
    cycle_dir: Path,
) -> dict[str, Any]:
    """Return auditable final outcomes from the one-shot initial pass.

    This is the public projection boundary used by semantic closure.  It
    exposes only accepted unit outcomes and immutable patch/transaction
    locators; merely attempting, deferring, or producing ``no_patch`` never
    counts as handling a semantic contract.
    """

    output_dir = cycle_dir.resolve() / "initial_compatibility"
    analysis = _load_json_object(output_dir / "analysis.json")
    plan = _load_cycle_repair_plan(
        cycle_dir,
        output_dir / "initial_compatibility_plan.json"
    )
    attempts_payload = _load_json_object(
        output_dir / "llm_attempts.json"
    )
    units = _plan_units_by_id(plan)
    attempts_current = _attempt_payload_is_current(attempts_payload)
    queue_status = initial_compatibility_queue_status(cycle_dir)
    queue_finalized_without_attempts = bool(
        not int(queue_status.get("pending_work_unit_count") or 0)
        and int(queue_status.get("refresh_generation") or 0) > 0
        and str(queue_status.get("status") or "")
        in {"completed", "completed_with_deferred"}
    )
    artifacts_available = (
        _analysis_is_current(analysis)
        and _plan_is_current(plan)
        and (
            not units
            or attempts_current
            or queue_finalized_without_attempts
        )
    )
    if not artifacts_available:
        return {
            "schema": "initial_compatibility_unit_outcomes_v1",
            "status": "artifacts_unavailable",
            "unit_outcomes": {},
        }

    attempts = {
        str(item.get("unit_id") or ""): dict(item)
        for item in attempts_payload.get("attempts") or []
        if isinstance(item, dict)
        and str(item.get("unit_id") or "")
    }
    deferred_units = {
        str(unit_id): dict(record)
        for unit_id, record in dict(
            queue_status.get("deferred_work_units") or {}
        ).items()
        if str(unit_id) and isinstance(record, dict)
    }
    resolved_units = {
        str(unit_id)
        for unit_id in queue_status.get("resolved_work_unit_ids") or []
        if str(unit_id)
    }
    outcomes: dict[str, dict[str, Any]] = {}
    for unit_id, unit in sorted(units.items()):
        attempt = attempts.get(unit_id)
        if not isinstance(attempt, dict):
            if _unit_mode(unit) == "post_materialization_validation_only":
                outcomes[unit_id] = {
                    "unit_id": unit_id,
                    "status": "validation_only",
                    "contract_identities": _unit_contract_identities(unit),
                    "reason": (
                        "unit_requires_no_source_edit_and_is_closed_by_the_"
                        "compile_valid_contract_evaluator"
                    ),
                    "outcome_source": "initial_validation_unit",
                }
                continue
            deferred = deferred_units.get(unit_id) or {}
            outcomes[unit_id] = {
                "unit_id": unit_id,
                "status": "unresolved",
                "contract_identities": _unit_contract_identities(unit),
                "reason": (
                    deferred.get("reason")
                    or (
                        "initial_unit_was_marked_resolved_without_an_"
                        "immutable_attempt_artifact"
                        if unit_id in resolved_units
                        else "initial_unit_has_no_completed_attempt"
                    )
                ),
                "outcome_source": (
                    "finalized_initial_queue"
                    if queue_finalized_without_attempts
                    else "initial_attempt_artifact"
                ),
            }
            continue

        statuses = sorted(_initial_attempt_statuses(attempt))
        completed = _initial_attempt_completed(attempt)
        apply_results = [
            dict(item)
            for item in attempt.get("apply_results") or []
            if isinstance(item, dict)
        ]
        transaction_dirs = sorted(
            {
                str(item.get("transaction_dir") or "")
                for item in apply_results
                if str(item.get("transaction_dir") or "")
            }
        )
        patch_paths: set[str] = {
            str(item.get("patch_path") or "")
            for item in apply_results
            if str(item.get("patch_path") or "")
        }
        transaction_statuses: list[dict[str, Any]] = []
        for transaction_dir in transaction_dirs:
            manifest_path = Path(transaction_dir) / "manifest.json"
            manifest = _load_json_object(manifest_path)
            transaction_statuses.append(
                {
                    "transaction_dir": transaction_dir,
                    "manifest_path": str(manifest_path),
                    "status": str(manifest.get("status") or ""),
                }
            )
            if str(manifest.get("patch_path") or ""):
                patch_paths.add(str(manifest["patch_path"]))

        disposition = "unresolved"
        if completed:
            disposition = (
                "already_satisfied"
                if set(statuses) == {"already_satisfied"}
                else "modified"
            )
        outcomes[unit_id] = {
            "unit_id": unit_id,
            "status": disposition,
            "contract_identities": _unit_contract_identities(unit),
            "apply_statuses": statuses,
            "patch_paths": sorted(path for path in patch_paths if path),
            "patch_signatures": sorted(
                {
                    str(item.get("patch_signature") or "")
                    for item in apply_results
                    if str(item.get("patch_signature") or "")
                }
            ),
            "transaction_statuses": transaction_statuses,
            "touched_files": sorted(
                _source_changes_from_apply_results(apply_results)
            ),
            "structural_postcondition": dict(
                attempt.get("structural_postcondition") or {}
            ),
            **(
                {}
                if completed
                else {
                    "reason": (
                        "initial_attempt_did_not_complete_with_an_accepted_"
                        "atomic_outcome"
                    )
                }
            ),
        }

    counts = Counter(
        str(item.get("status") or "")
        for item in outcomes.values()
    )
    return {
        "schema": "initial_compatibility_unit_outcomes_v1",
        "status": "resolved",
        "artifact_revisions": {
            "analysis_revision": str(
                attempts_payload.get("analysis_revision")
                or analysis.get("analysis_revision")
                or ""
            ),
            "difference_unit_revision": str(
                attempts_payload.get("difference_unit_revision")
                or plan.get("difference_unit_revision")
                or ""
            ),
            "plan_schema": str(
                attempts_payload.get("plan_schema")
                or plan.get("schema")
                or ""
            ),
        },
        "unit_outcomes": outcomes,
        "summary": {
            "unit_count": len(outcomes),
            "modified_unit_count": int(counts["modified"]),
            "already_satisfied_unit_count": int(
                counts["already_satisfied"]
            ),
            "validation_only_unit_count": int(
                counts["validation_only"]
            ),
            "unresolved_unit_count": int(counts["unresolved"]),
            "finalized_queue_outcome_count": sum(
                1
                for outcome in outcomes.values()
                if outcome.get("outcome_source")
                == "finalized_initial_queue"
            ),
        },
    }


def finalize_initial_static_integration_batch(
    spec: CaseSpec,
    *,
    cycle_dir: Path,
) -> dict[str, Any]:
    attempts_path = (
        cycle_dir.resolve()
        / "initial_compatibility"
        / "llm_attempts.json"
    )
    payload = _load_json_object(attempts_path)
    plan = _load_cycle_repair_plan(
        cycle_dir,
        cycle_dir.resolve()
        / "initial_compatibility"
        / "initial_compatibility_plan.json"
    )
    attempts = [
        dict(item)
        for item in (
            payload.get("attempts") or []
            if _attempt_payload_is_current(payload)
            else []
        )
        if isinstance(item, dict)
    ]
    summary = _finalize_static_materialization_integration(
        spec,
        attempts,
        validation_units=plan.get("work_unit_plans") or [],
    )
    if not _attempt_payload_is_current(payload):
        payload = {
            "schema": INITIAL_ATTEMPTS_SCHEMA,
            "analysis_revision": INITIAL_COMPATIBILITY_ANALYSIS_REVISION,
            "difference_unit_revision": (
                INITIAL_COMPATIBILITY_FRONTIER_REVISION
            ),
            "plan_schema": INITIAL_COMPATIBILITY_PLAN_SCHEMA,
        }
    payload["attempts"] = attempts
    payload["static_integration_batch"] = summary
    write_json(attempts_path, payload)
    return summary


def initial_compatibility_queue_status(
    cycle_dir: Path,
) -> dict[str, Any]:
    output_dir = cycle_dir.resolve() / "initial_compatibility"
    plan = _load_cycle_repair_plan(
        cycle_dir,
        output_dir / "initial_compatibility_plan.json",
    )
    state = _load_incremental_state(
        output_dir / "incremental_state.json"
    )
    attempts_path = output_dir / "llm_attempts.json"
    attempts_payload = _load_json_object(attempts_path)
    attempts = _attempts_from_path(attempts_path)
    static_integration_batch = (
        dict(attempts_payload.get("static_integration_batch") or {})
        if _attempt_payload_is_current(attempts_payload)
        and isinstance(
            attempts_payload.get("static_integration_batch"),
            dict,
        )
        else {}
    )
    units_by_id = _plan_units_by_id(plan)
    queue_units = {
        str(item)
        for item in (
            state.get("queue_work_unit_ids")
            or _queue_units_from_plan(plan)
        )
        if str(item)
    }
    resolved = {
        str(item)
        for item in state.get("resolved_work_unit_ids") or []
        if str(item)
    }
    deferred = {
        str(unit_id): dict(record) if isinstance(record, dict) else {
            "reason": str(record)
        }
        for unit_id, record in dict(
            state.get("deferred_work_units") or {}
        ).items()
        if str(unit_id)
    }
    attempted = {
        str(attempt.get("unit_id") or "")
        for attempt in attempts
        if str(attempt.get("unit_id") or "")
        and _initial_attempt_terminal(attempt)
    }
    completed_attempts = {
        str(attempt.get("unit_id") or "")
        for attempt in attempts
        if _initial_attempt_completed(attempt)
        and str(attempt.get("unit_id") or "")
    }
    applied_pending_confirmation = {
        str(attempt.get("unit_id") or "")
        for attempt in attempts
        if _initial_attempt_applied_pending_confirmation(attempt)
        and str(attempt.get("unit_id") or "")
    }
    apply_failed = (
        attempted - completed_attempts - applied_pending_confirmation
    )
    pending = _pending_queue_work_unit_ids(
        units_by_id,
        queue_work_unit_ids=queue_units,
        resolved_work_unit_ids=resolved,
        deferred_work_unit_ids=deferred,
        attempted_work_unit_ids=attempted,
    )
    unresolved = _unresolved_queue_work_unit_ids(
        units_by_id,
        queue_work_unit_ids=queue_units,
        settled_work_unit_ids=resolved | completed_attempts,
    )
    pending_context_blocked = {
        unit_id
        for unit_id in pending
        if bool(
            (units_by_id.get(unit_id) or {}).get(
                "requires_context_expansion"
            )
        )
    }
    pending_actionable = pending - pending_context_blocked
    orphaned = {
        unit_id
        for unit_id in queue_units
        if unit_id not in units_by_id
        and unit_id not in resolved
        and unit_id not in deferred
    }
    validation_requested = bool(
        state.get("validation_required")
        or state.get("pending_changed_files")
        or (
            attempted
            and not state.get("validation_history")
        )
    )
    validation_required = validation_requested and not pending
    queued_contracts = _contract_identities_for_unit_ids(
        units_by_id, queue_units
    )
    pending_contracts = _contract_identities_for_unit_ids(
        units_by_id, pending
    )
    unresolved_contracts = _contract_identities_for_unit_ids(
        units_by_id, unresolved
    )
    resolved_contracts = _contract_identities_for_unit_ids(
        units_by_id, resolved
    )
    attempted_contracts = _contract_identities_for_unit_ids(
        units_by_id, attempted
    )
    apply_failed_contracts = _contract_identities_for_unit_ids(
        units_by_id, apply_failed
    )
    applied_pending_confirmation_contracts = (
        _contract_identities_for_unit_ids(
            units_by_id,
            applied_pending_confirmation,
        )
    )
    awaiting_validation = (
        attempted - resolved - set(deferred)
        if validation_required
        else set()
    )
    return {
        "schema": "initial_compatibility_queue_status_v5",
        "status": (
            "pending_validation"
            if validation_required
            else "pending"
            if pending
            else "completed_with_deferred"
            if deferred or apply_failed or orphaned
            else "completed"
        ),
        "validation_required": validation_required,
        "pending_changed_files": list(state.get("pending_changed_files") or []),
        "post_materialization_validation_records": list(
            static_integration_batch.get("validation_records") or []
        ),
        "queue_work_unit_count": len(queue_units),
        "queue_work_unit_ids": sorted(queue_units),
        "pending_work_unit_count": len(pending),
        "pending_work_unit_ids": sorted(pending),
        "unresolved_work_unit_count": len(unresolved),
        "unresolved_work_unit_ids": sorted(unresolved),
        "pending_context_blocked_work_unit_count": len(
            pending_context_blocked
        ),
        "pending_context_blocked_work_unit_ids": sorted(
            pending_context_blocked
        ),
        "pending_actionable_work_unit_count": len(pending_actionable),
        "pending_actionable_work_unit_ids": sorted(pending_actionable),
        "resolved_work_unit_count": len(resolved),
        "resolved_work_unit_ids": sorted(resolved),
        "deferred_work_unit_count": len(deferred),
        "deferred_work_units": deferred,
        "attempted_work_unit_count": len(attempted),
        "attempted_work_unit_ids": sorted(attempted),
        "completed_initial_attempt_work_unit_ids": sorted(
            completed_attempts
        ),
        "applied_pending_compile_confirmation_work_unit_count": len(
            applied_pending_confirmation
        ),
        "applied_pending_compile_confirmation_work_unit_ids": sorted(
            applied_pending_confirmation
        ),
        "apply_failed_work_unit_ids": sorted(apply_failed),
        "awaiting_validation_work_unit_ids": sorted(
            awaiting_validation
        ),
        "orphaned_work_unit_ids": sorted(orphaned),
        "queue_contract_count": len(queued_contracts),
        "queue_contract_identities": queued_contracts,
        "pending_contract_count": len(pending_contracts),
        "pending_contract_identities": pending_contracts,
        "unresolved_contract_count": len(unresolved_contracts),
        "unresolved_contract_identities": unresolved_contracts,
        "resolved_contract_count": len(resolved_contracts),
        "resolved_contract_identities": resolved_contracts,
        "attempted_contract_identities": attempted_contracts,
        "applied_pending_compile_confirmation_contract_identities": (
            applied_pending_confirmation_contracts
        ),
        "apply_failed_contract_identities": apply_failed_contracts,
        "refresh_generation": int(state.get("refresh_generation") or 0),
        "incremental_state_path": str(
            (output_dir / "incremental_state.json").resolve()
        ),
    }


def finalize_initial_compatibility_pass(
    *,
    cycle_dir: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Close the one-shot B0 pass without reusing its units in B/C."""

    output_dir = cycle_dir.resolve() / "initial_compatibility"
    incremental_state_path = output_dir / "incremental_state.json"
    state = _load_incremental_state(incremental_state_path)
    plan = _load_cycle_repair_plan(
        cycle_dir,
        output_dir / "initial_compatibility_plan.json"
    )
    queue_before = initial_compatibility_queue_status(cycle_dir)
    units_by_id = _plan_units_by_id(plan)
    attempts = _attempts_from_path(output_dir / "llm_attempts.json")
    attempts_by_unit = {
        str(attempt.get("unit_id") or ""): attempt
        for attempt in attempts
        if str(attempt.get("unit_id") or "")
    }
    queue_units = set(queue_before.get("queue_work_unit_ids") or [])
    resolved = set(queue_before.get("resolved_work_unit_ids") or [])
    deferred = dict(queue_before.get("deferred_work_units") or {})
    successful_units = {
        unit_id
        for unit_id, attempt in attempts_by_unit.items()
        if _initial_attempt_completed(attempt)
    }
    resolved.update(successful_units)
    unresolved_units = queue_units - resolved
    for unit_id in unresolved_units:
        attempt = attempts_by_unit.get(unit_id) or {}
        statuses = sorted(_initial_attempt_statuses(attempt))
        previous = deferred.get(unit_id) or {}
        unit = units_by_id.get(unit_id) or {}
        analysis_boundary = (
            dict(unit.get("analysis_validity_boundary") or {})
            if isinstance(unit.get("analysis_validity_boundary"), dict)
            else {}
        )
        deferred[unit_id] = {
            "status": "unresolved_observation",
            "reason": (
                "target_translation_unit_analysis_deferred_to_fresh_evidence"
                if analysis_boundary
                else previous.get("reason")
                or "initial_unit_did_not_produce_a_statically_validated_change"
            ),
            "apply_statuses": statuses,
            "contract_identities": _unit_contract_identities(
                unit
            ),
            "reuse_in_compiler_repair": False,
            "successor_evidence": (
                "fresh_compiler_diagnostic_code_graph_or_demand_closure"
                if analysis_boundary
                else "fresh_compiler_diagnostic_code_graph"
            ),
            "demand_ledger_retained": bool(analysis_boundary),
            "analysis_validity_boundary": analysis_boundary or None,
        }
    deferred = {
        unit_id: record
        for unit_id, record in deferred.items()
        if unit_id in unresolved_units
    }
    validation_ok = bool(validation.get("overall_ok"))

    generation = int(state.get("refresh_generation") or 0) + 1
    validation_record = {
        "generation": generation,
        "validated_at": utc_now(),
        "validation_status": validation.get("status"),
        "validation_overall_ok": validation_ok,
        "validation_path": validation.get("validation_path"),
        "cycle": validation.get("cycle"),
        "changed_files": sorted(
            {
                normalize_relpath(str(item))
                for item in state.get("pending_changed_files") or []
                if normalize_relpath(str(item))
            }
        ),
        "resolved_work_unit_ids": sorted(resolved),
        "unresolved_observation_work_unit_ids": sorted(
            unresolved_units
        ),
        "next_stage": (
            "complete"
            if validation_ok
            else "fresh_compiler_diagnostic_code_graph"
        ),
        "b0_units_reused": False,
        "graph_refresh_performed": False,
    }
    validation_history = [
        *[
            dict(item)
            for item in state.get("validation_history") or []
            if isinstance(item, dict)
        ],
        validation_record,
    ]
    history_path = ensure_dir(
        output_dir / "validation_history"
    ) / f"validation-{generation:04d}.json"
    write_json(history_path, validation_record)
    _write_incremental_state(
        incremental_state_path,
        pending_changed_files=set(),
        validation_required=False,
        active_batch={},
        queue_work_unit_ids=queue_units,
        resolved_work_unit_ids=resolved,
        deferred_work_units=deferred,
        validation_history=validation_history,
        refresh_generation=generation,
    )
    return {
        "schema": "initial_compatibility_pass_result_v3",
        "status": (
            "compile_validated"
            if validation_ok
            else "completed_with_compile_errors"
        ),
        "validation_record_path": str(history_path.resolve()),
        "resolved_work_unit_count": len(resolved),
        "resolved_work_unit_ids": sorted(resolved),
        "unresolved_observation_work_unit_count": len(
            unresolved_units
        ),
        "unresolved_observation_work_unit_ids": sorted(
            unresolved_units
        ),
        "b0_units_reused": False,
        "queue_before": queue_before,
        "queue": initial_compatibility_queue_status(cycle_dir),
    }


def prepare_initial_compatibility(
    spec: CaseSpec,
    *,
    state_root: Path,
    cycle_dir: Path,
    initial_copy_set: Iterable[object],
) -> dict[str, Any]:
    output_dir = ensure_dir(cycle_dir / "initial_compatibility")
    copied = [normalize_relpath(str(item)) for item in initial_copy_set if str(item).strip()]
    existing_analysis = _load_json_object(output_dir / "analysis.json")
    attempts_path = output_dir / "llm_attempts.json"
    existing_preparation = _load_json_object(
        output_dir / "preparation_result.json"
    )
    completed_preparation_artifact = bool(
        existing_preparation.get("schema")
        == INITIAL_PREPARATION_SCHEMA
        and str(existing_preparation.get("case_id") or "") == spec.case_id
        and str(existing_preparation.get("status") or "") == "prepared"
        and [
            normalize_relpath(str(item))
            for item in existing_preparation.get("initial_copy_set") or []
            if str(item).strip()
        ]
        == copied
    )
    attempts_payload = _load_json_object(attempts_path)
    current_attempt_artifact = _attempt_payload_is_current(
        attempts_payload
    )
    resume_existing = bool(
        (current_attempt_artifact or completed_preparation_artifact)
        and _analysis_is_current(existing_analysis)
    )
    analysis = (
        existing_analysis
        if resume_existing
        else analyze_initial_compatibility(spec, initial_copy_set=copied)
    )
    plan = build_initial_compatibility_plan(spec, analysis)
    _write_analysis_artifacts(
        output_dir,
        state_root=state_root,
        analysis=analysis,
        plan=plan,
    )

    result = {
        "schema": INITIAL_PREPARATION_SCHEMA,
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "status": (
            "prepared"
            if bool(analysis.get("preparation_complete", True))
            else "preparation_incomplete"
        ),
        "output_dir": str(output_dir.resolve()),
        "analysis_path": str((output_dir / "analysis.json").resolve()),
        "plan_path": str((output_dir / "initial_compatibility_plan.json").resolve()),
        "initial_copy_set": copied,
        "analysis_summary": analysis.get("summary") or {},
        "preparation_blockers": list(
            analysis.get("preparation_blockers") or []
        ),
        "plan_summary": plan.get("summary") or {},
        "resumed_from_existing_graph": resume_existing,
        "analysis_pass_count": 0 if resume_existing else 1,
        "source_changes_applied": False,
        "execution_policy": {
            "deterministic_and_model_units": "one_topological_initial_pass",
            "intermediate_compile": False,
            "cross_version_graph_reanalysis": False,
            "current_tree_refresh": "local_source_extents_only",
        },
    }
    write_json(output_dir / "preparation_result.json", result)
    return result


def execute_initial_llm_units(
    spec: CaseSpec,
    *,
    state_root: Path,
    cycle_dir: Path,
    model_runner: ModelRunner,
    allow_escalation: bool,
) -> dict[str, Any]:
    output_dir = ensure_dir(cycle_dir / "initial_compatibility")
    response_dir = ensure_dir(output_dir / "responses")
    attempts_path = output_dir / "llm_attempts.json"
    attempts = _attempts_from_path(attempts_path)
    max_attempts_per_unit = 1 + INITIAL_UNIT_REPLAN_LIMIT
    attempts_by_unit = {
        str(item.get("unit_id") or ""): item
        for item in attempts
        if str(item.get("unit_id") or "")
    }
    attempted_units = {
        str(item.get("unit_id") or "")
        for item in attempts
        if str(item.get("unit_id") or "")
        and _initial_attempt_terminal(item)
    }
    dependency_completed_units = {
        str(item.get("unit_id") or "")
        for item in attempts
        if _initial_attempt_completed(item)
        and str(item.get("unit_id") or "")
    }
    _analysis, plan = _load_prepared_artifacts(
        output_dir,
        case_id=spec.case_id,
    )

    incremental_state_path = output_dir / "incremental_state.json"
    pending_changed_files = _restore_incremental_state(
        incremental_state_path,
        attempts=attempts,
    )
    stored_incremental = _load_incremental_state(
        incremental_state_path
    )
    queue_units = {
        str(item)
        for item in (
            stored_incremental.get("queue_work_unit_ids")
            or _queue_units_from_plan(plan)
        )
        if str(item)
    }
    resolved_units = {
        str(item)
        for item in stored_incremental.get("resolved_work_unit_ids") or []
        if str(item)
    }
    deferred_units = {
        str(unit_id): dict(record) if isinstance(record, dict) else {
            "reason": str(record)
        }
        for unit_id, record in dict(
            stored_incremental.get("deferred_work_units") or {}
        ).items()
        if str(unit_id)
    }
    active_batch = (
        dict(stored_incremental.get("active_batch") or {})
        if isinstance(stored_incremental.get("active_batch"), dict)
        else {}
    )
    inflight_unit_id = str(active_batch.get("inflight_unit_id") or "")
    if inflight_unit_id:
        recovered_files = _finalized_inflight_transaction_files(
            state_root,
            case_id=spec.case_id,
            work_unit_id=inflight_unit_id,
        )
        if recovered_files:
            pending_changed_files.update(recovered_files)
            active_batch["recovered_inflight_transaction"] = {
                "unit_id": inflight_unit_id,
                "contract_identity": active_batch.get(
                    "inflight_contract_identity"
                ),
                "contract_identities": list(
                    active_batch.get("inflight_contract_identities") or []
                ),
                "changed_files": sorted(recovered_files),
                "recovered_at": utc_now(),
            }
        active_batch.pop("inflight_unit_id", None)
        active_batch.pop("inflight_contract_identity", None)
        active_batch.pop("inflight_contract_identities", None)

    pending_unit_ids = _pending_queue_work_unit_ids(
        _plan_units_by_id(plan),
        queue_work_unit_ids=queue_units,
        resolved_work_unit_ids=resolved_units,
        deferred_work_unit_ids=deferred_units,
        attempted_work_unit_ids=attempted_units,
    )
    validation_required = bool(
        stored_incremental.get("validation_required")
        or pending_changed_files
        or (
            dependency_completed_units
            and not stored_incremental.get("validation_history")
        )
    ) and not pending_unit_ids

    _write_incremental_state(
        incremental_state_path,
        pending_changed_files=pending_changed_files,
        validation_required=validation_required,
        active_batch=active_batch,
        queue_work_unit_ids=queue_units,
        resolved_work_unit_ids=resolved_units,
        deferred_work_units=deferred_units,
    )
    batch_records: list[dict[str, Any]] = []

    def persist_attempts() -> None:
        write_json(
            attempts_path,
            {
                "schema": INITIAL_ATTEMPTS_SCHEMA,
                "analysis_revision": INITIAL_COMPATIBILITY_ANALYSIS_REVISION,
                "difference_unit_revision": (
                    INITIAL_COMPATIBILITY_FRONTIER_REVISION
                ),
                "plan_schema": INITIAL_COMPATIBILITY_PLAN_SCHEMA,
                "attempts": attempts,
            },
        )

    def persist_state() -> None:
        _write_incremental_state(
            incremental_state_path,
            pending_changed_files=pending_changed_files,
            validation_required=validation_required,
            active_batch=active_batch,
            queue_work_unit_ids=queue_units,
            resolved_work_unit_ids=resolved_units,
            deferred_work_units=deferred_units,
        )

    def unit_is_pending(unit: dict[str, Any]) -> bool:
        unit_id = work_unit_id(unit)
        return bool(
            unit_id
            and unit_id in queue_units
            and unit_id not in resolved_units
            and unit_id not in deferred_units
            and unit_id not in attempted_units
        )

    while not validation_required:
        candidates = [
            unit
            for unit in plan.get("work_unit_plans") or []
            if (
                _unit_mode(unit) == "requires_llm"
                or _unit_mode(unit).startswith("deterministic_")
            )
            and not unit.get("requires_context_expansion")
            and unit_is_pending(unit)
        ]
        if not candidates:
            active_batch = {}
            pending_unit_ids = _pending_queue_work_unit_ids(
                _plan_units_by_id(plan),
                queue_work_unit_ids=queue_units,
                resolved_work_unit_ids=resolved_units,
                deferred_work_unit_ids=deferred_units,
                attempted_work_unit_ids=attempted_units,
            )
            validation_required = bool(
                pending_changed_files or attempted_units
            ) and not pending_unit_ids
            break

        active_unit_ids = {
            str(item)
            for item in active_batch.get("selected_work_unit_ids") or []
            if str(item)
        }
        if active_unit_ids:
            selected_units = [
                unit
                for unit in candidates
                if work_unit_id(unit) in active_unit_ids
            ]
            active_batch["summary"] = {
                **dict(active_batch.get("summary") or {}),
                "resumed": True,
            }
        else:
            selected_units, schedule = dependency_order(candidates)
            selected_write_files = sorted(
                {
                    path
                    for unit in selected_units
                    for path in declared_write_files(unit)
                }
            )
            active_batch = {
                "schema": "initial_compatibility_topological_pass_chunk_v2",
                "started_at": utc_now(),
                "selected_work_unit_ids": [
                    work_unit_id(unit) for unit in selected_units
                ],
                "contract_identities": sorted(
                    {
                        identity
                        for unit in selected_units
                        for identity in _unit_contract_identities(unit)
                    }
                ),
                "selected_write_files": selected_write_files,
                "deferred": [],
                "schedule": schedule,
                "summary": {
                    "selected_count": len(selected_units),
                    "deferred_count": 0,
                    "write_file_count": len(selected_write_files),
                    "topological_sequential_execution": True,
                    "compile_diagnostics_involved": False,
                },
            }
        if not selected_units:
            active_batch = {}
            if active_unit_ids:
                persist_state()
                continue
            break

        batch_changed_files: set[str] = set(
            normalize_relpath(str(item))
            for item in active_batch.get("changed_files") or []
            if str(item)
        )
        completed_units = {
            str(item)
            for item in active_batch.get("completed_work_unit_ids") or []
            if str(item)
        }
        dependency_completed_units.update(completed_units)
        file_revisions: Counter[str] = Counter(
            {
                relative_path: 1
                for relative_path in batch_changed_files
            }
        )
        deterministic_retry_history: dict[
            str, list[dict[str, Any]]
        ] = {}
        deterministic_waiting: dict[str, dict[str, Any]] = {}
        unit_queue: list[
            tuple[
                dict[str, Any],
                bool,
                dict[str, Any] | None,
            ]
        ] = [
            (unit, False, None)
            for unit in selected_units
        ]
        while unit_queue or deterministic_waiting:
            if not unit_queue:
                retriable_ids = [
                    unit_id
                    for unit_id, record in deterministic_waiting.items()
                    if any(
                        file_revisions[relative_path]
                        > int(
                            (
                                record.get("observed_file_revisions")
                                or {}
                            ).get(relative_path)
                            or 0
                        )
                        for relative_path in record.get("context_files") or []
                    )
                    or bool(
                        set(
                            record.get("unresolved_work_unit_ids") or []
                        )
                        & dependency_completed_units
                    )
                ]
                if retriable_ids:
                    unit_queue = [
                        (
                            dict(deterministic_waiting.pop(unit_id)["unit"]),
                            False,
                            None,
                        )
                        for unit_id in retriable_ids
                    ]
                    continue

                active_batch["deterministic_fixed_point"] = {
                    "schema": (
                        "initial_deterministic_fixed_point_v1"
                    ),
                    "status": "stable",
                    "remaining_work_unit_ids": list(
                        deterministic_waiting
                    ),
                    "remaining_count": len(
                        deterministic_waiting
                    ),
                    "retry_count": sum(
                        max(0, len(history) - 1)
                        for history in (
                            deterministic_retry_history.values()
                        )
                    ),
                    "fallback": None,
                    "blocked_before_compile": True,
                }
                unit_queue = [
                    (
                        dict(record["unit"]),
                        True,
                        dict(record["last_result"]),
                    )
                    for record in deterministic_waiting.values()
                ]
                deterministic_waiting.clear()

            unit, force_deterministic_fallback, cached_deterministic = (
                unit_queue.pop(0)
            )
            unit_id = work_unit_id(unit)
            contract_identity = _contract_identity(unit)
            member_contract_identities = _unit_contract_identities(unit)
            if (
                not unit_is_pending(unit)
                or unit_id in completed_units
            ):
                continue
            unresolved_dependencies = sorted(
                (work_unit_dependencies(unit) & queue_units)
                - dependency_completed_units
            )
            waiting_dependencies = [
                dependency_id
                for dependency_id in unresolved_dependencies
                if dependency_id in deterministic_waiting
            ]
            if waiting_dependencies:
                # A dependent unit may not run on the assumption that a
                # blocked deterministic prerequisite succeeded.  Resolve the
                # prerequisite through its model fallback first, then revisit
                # the dependent against the resulting current tree.
                unit_queue.insert(
                    0,
                    (unit, force_deterministic_fallback, cached_deterministic),
                )
                for dependency_id in reversed(waiting_dependencies):
                    record = deterministic_waiting.pop(dependency_id)
                    unit_queue.insert(
                        0,
                        (
                            dict(record["unit"]),
                            True,
                            dict(record["last_result"]),
                        ),
                    )
                continue
            if unresolved_dependencies:
                deferred_units[unit_id] = {
                    "status": "unresolved_observation",
                    "reason": "initial_dependency_not_statically_satisfied",
                    "unresolved_work_unit_ids": unresolved_dependencies,
                    "reuse_in_compiler_repair": False,
                }
                active_batch.setdefault(
                    "dependency_blocked_work_units",
                    [],
                ).append(
                    {
                        "work_unit_id": unit_id,
                        "unresolved_work_unit_ids": unresolved_dependencies,
                    }
                )
                persist_state()
                continue
            current_tree_context_files = {
                normalize_relpath(str(item))
                for item in [
                    *list(unit.get("analysis_read_files") or []),
                    *list(declared_write_files(unit)),
                ]
                if normalize_relpath(str(item))
            }
            overlapping_changes = (
                batch_changed_files & current_tree_context_files
            )
            prior_attempt = attempts_by_unit.get(unit_id)
            refresh_files = (
                current_tree_context_files
                if isinstance(prior_attempt, dict)
                and not _initial_attempt_terminal(prior_attempt)
                else overlapping_changes
            )
            if refresh_files:
                unit = refresh_initial_unit_from_current_tree(
                    spec,
                    unit,
                    changed_files=refresh_files,
                )
                active_batch.setdefault("unit_context_refreshes", []).append(
                    {
                        "schema": "initial_current_tree_local_refresh_v1",
                        "contract_identity": contract_identity,
                        "contract_identities": member_contract_identities,
                        "files": sorted(refresh_files),
                        "graph_rebuilt": False,
                    }
                )
            if isinstance(prior_attempt, dict):
                unit["prior_unresolved_patch_attempts"] = [
                    _initial_retry_prompt_record(prior_attempt)
                ]

            unit_attempt_number = (
                int((prior_attempt or {}).get("attempt_number") or 0) + 1
            )
            single_plan = {
                **plan,
                "work_unit_plans": [unit],
                "summary": {
                    **dict(plan.get("summary") or {}),
                    "work_unit_count": 1,
                    "selected_initial_unit_id": unit_id,
                    "initial_pass_chunk": active_batch,
                },
            }
            plan_path = output_dir / f"{unit_id}.plan.json"
            write_repair_plan(
                plan_path,
                single_plan,
                state_root=state_root,
            )
            active_batch["inflight_unit_id"] = unit_id
            active_batch["inflight_contract_identity"] = contract_identity
            active_batch["inflight_contract_identities"] = list(
                member_contract_identities
            )
            persist_state()

            prompt_path: Path | None = None
            response_path: Path | None = None
            deterministic_fallback: dict[str, Any] | None = None
            mode = _unit_mode(unit)
            if mode.startswith("deterministic_"):
                deterministic = (
                    cached_deterministic
                    if cached_deterministic is not None
                    else _apply_deterministic_units(
                        spec,
                        state_root=state_root,
                        cycle_dir=cycle_dir,
                        plan=single_plan,
                        allow_escalation=allow_escalation,
                        completed_work_unit_ids=(
                            dependency_completed_units
                        ),
                        known_work_unit_ids=queue_units,
                    )
                )
                apply_results = list(deterministic.get("results") or [])
                metadata = {
                    "api_mode": "deterministic_static_tool",
                    "model": None,
                }
                structural_postcondition = {
                    "status": "enforced_by_deterministic_contract_tool",
                }
                deterministic_summary = dict(
                    deterministic.get("summary") or {}
                )
                blocked_count = int(
                    deterministic_summary.get("blocked_count") or 0
                )
                deferred_dependency_count = int(
                    deterministic_summary.get(
                        "deferred_dependency_count"
                    )
                    or 0
                )
                if (
                    blocked_count or deferred_dependency_count
                ) and not force_deterministic_fallback:
                    deterministic_retry_history.setdefault(
                        unit_id,
                        [],
                    ).append(deterministic)
                    unresolved_work_unit_ids = sorted(
                        {
                            str(dependency_id)
                            for result in apply_results
                            for dependency_id in (
                                result.get(
                                    "unresolved_work_unit_ids"
                                )
                                or []
                            )
                            if str(dependency_id)
                        }
                    )
                    deterministic_waiting[unit_id] = {
                        "unit": unit,
                        "last_result": deterministic,
                        "context_files": sorted(
                            current_tree_context_files
                        ),
                        "observed_file_revisions": {
                            relative_path: file_revisions[
                                relative_path
                            ]
                            for relative_path in (
                                current_tree_context_files
                            )
                        },
                        "unresolved_work_unit_ids": (
                            unresolved_work_unit_ids
                        ),
                    }
                    active_batch.setdefault(
                        "deterministic_fixed_point_waits",
                        [],
                    ).append(
                        {
                            "work_unit_id": unit_id,
                            "contract_identities": (
                                member_contract_identities
                            ),
                            "blocked_count": blocked_count,
                            "deferred_dependency_count": (
                                deferred_dependency_count
                            ),
                            "context_files": sorted(
                                current_tree_context_files
                            ),
                            "unresolved_work_unit_ids": (
                                unresolved_work_unit_ids
                            ),
                            "wait_generation": len(
                                deterministic_retry_history[
                                    unit_id
                                ]
                            ),
                        }
                    )
                    active_batch.pop("inflight_unit_id", None)
                    active_batch.pop(
                        "inflight_contract_identity",
                        None,
                    )
                    active_batch.pop(
                        "inflight_contract_identities",
                        None,
                    )
                    persist_state()
                    continue
                if blocked_count or deferred_dependency_count:
                    deterministic_fallback = deterministic

            if not mode.startswith("deterministic_"):
                prompt = build_initial_compatibility_prompt(
                    spec,
                    unit=unit,
                )
                prompt_path = output_dir / (
                    f"{unit_id}.attempt-{unit_attempt_number:04d}.prompt.txt"
                )
                prompt_path.write_text(prompt, encoding="utf-8")
                try:
                    response_text, metadata = model_runner(prompt)
                except Exception as exc:
                    model_error_path = response_dir / (
                        f"{unit_id}.attempt-{unit_attempt_number:04d}.model_error.json"
                    )
                    write_json(
                        model_error_path,
                        {
                            "schema": "initial_compatibility_model_error_v1",
                            "generated_at": utc_now(),
                            "unit_id": unit_id,
                            "contract_identity": contract_identity,
                            "contract_identities": list(
                                member_contract_identities
                            ),
                            "attempt_number": unit_attempt_number,
                            "prompt_path": str(prompt_path.resolve()),
                            "error": str(exc),
                            "model_api_error": model_api_error_metadata(exc),
                        },
                    )
                    active_batch["completed_work_unit_ids"] = sorted(
                        completed_units
                    )
                    active_batch["changed_files"] = sorted(
                        batch_changed_files
                    )
                    persist_state()
                    raise
                response_path = response_dir / (
                    f"{unit_id}.attempt-{unit_attempt_number:04d}.response.txt"
                )
                response_path.write_text(response_text, encoding="utf-8")
                canonical_response_path = (
                    response_dir / f"{unit_id}.response.txt"
                )
                canonical_response_path.write_text(
                    response_text,
                    encoding="utf-8",
                )
                write_json(
                    response_dir
                    / f"{unit_id}.attempt-{unit_attempt_number:04d}.response.meta.json",
                    metadata,
                )
                structural_baseline = _capture_structural_baseline(
                    spec,
                    _unit_editable_source_files(
                        unit,
                        allow_escalation=allow_escalation,
                    ),
                )
                apply_results = apply_plan_responses(
                    spec,
                    single_plan,
                    response_dir=response_dir,
                    state_root=state_root,
                    allow_escalation=allow_escalation,
                    plan_path=plan_path,
                )
                changed_files = _source_changes_from_apply_results(
                    apply_results
                )
                structural_postcondition = (
                    _validate_structural_postconditions(
                        spec,
                        changed_files=changed_files,
                        baseline=structural_baseline,
                        unit=unit,
                    )
                )
                apply_results = _rollback_postcondition_failures(
                    spec,
                    apply_results,
                    structural_postcondition,
                )

            changed_files = _source_changes_from_apply_results(apply_results)
            completed_attempt = _initial_attempt_completed(
                {
                    "apply_results": apply_results,
                    "structural_postcondition": structural_postcondition,
                }
            )
            if changed_files:
                batch_changed_files.update(changed_files)
                pending_changed_files.update(changed_files)
                file_revisions.update(changed_files)
            provisional_attempt = {
                "unit_id": unit_id,
                "prompt_path": (
                    str(prompt_path.resolve()) if prompt_path else None
                ),
                "response_path": (
                    str(response_path.resolve()) if response_path else None
                ),
                "model_call": metadata,
                "apply_results": apply_results,
                "attempt_number": unit_attempt_number,
                "changed_files": sorted(changed_files),
                "structural_postcondition": structural_postcondition,
                "deterministic_fallback": deterministic_fallback,
                "deterministic_retry": (
                    {
                        "schema": (
                            "initial_deterministic_retry_history_v1"
                        ),
                        "prior_attempt_count": len(
                            deterministic_retry_history[unit_id]
                        ),
                        "prior_results": [
                            {
                                "summary": dict(
                                    prior.get("summary") or {}
                                ),
                                "results": [
                                    {
                                        key: result.get(key)
                                        for key in (
                                            "status",
                                            "reason",
                                            "file",
                                            "contract_identity",
                                            "unresolved_work_unit_ids",
                                        )
                                        if result.get(key)
                                        not in (None, "", [], {})
                                    }
                                    for result in (
                                        prior.get("results") or []
                                    )
                                    if isinstance(result, dict)
                                ],
                            }
                            for prior in (
                                deterministic_retry_history[
                                    unit_id
                                ]
                            )
                        ],
                    }
                    if unit_id in deterministic_retry_history
                    else None
                ),
            }
            replanable = _initial_attempt_replanable(provisional_attempt)
            retry_pending = bool(
                not completed_attempt
                and replanable
                and unit_attempt_number < max_attempts_per_unit
            )
            attempt = {
                **provisional_attempt,
                "retry_disposition": (
                    "completed"
                    if completed_attempt
                    else "retry_pending"
                    if retry_pending
                    else "exhausted"
                    if replanable
                    else "not_retryable"
                ),
                "prior_attempts": [
                    *list((prior_attempt or {}).get("prior_attempts") or []),
                    *(
                        [_initial_retry_prompt_record(prior_attempt)]
                        if isinstance(prior_attempt, dict)
                        else []
                    ),
                ],
            }
            if isinstance(prior_attempt, dict):
                attempts[attempts.index(prior_attempt)] = attempt
            else:
                attempts.append(attempt)
            attempts_by_unit[unit_id] = attempt
            if completed_attempt:
                completed_units.add(unit_id)
                dependency_completed_units.add(unit_id)
                attempted_units.add(unit_id)
            elif retry_pending:
                unit_queue.insert(0, (unit, False, None))
            else:
                attempted_units.add(unit_id)
            active_batch["completed_work_unit_ids"] = sorted(
                completed_units
            )
            active_batch["changed_files"] = sorted(batch_changed_files)
            active_batch.pop("inflight_unit_id", None)
            active_batch.pop("inflight_contract_identity", None)
            active_batch.pop("inflight_contract_identities", None)
            persist_attempts()
            persist_state()

        active_batch["completed_at"] = utc_now()
        active_batch["changed_files"] = sorted(batch_changed_files)
        batch_records.append(dict(active_batch))
        active_batch = {}

    static_integration_batch = (
        _finalize_static_materialization_integration(
            spec,
            attempts,
            validation_units=plan.get("work_unit_plans") or [],
        )
    )
    persist_attempts()
    persist_state()
    queue = initial_compatibility_queue_status(cycle_dir)
    pending_work_unit_count = int(
        queue.get("pending_work_unit_count") or 0
    )
    result = {
        "schema": "initial_compatibility_llm_execution_v3",
        "case_id": spec.case_id,
        "status": (
            "initial_pass_incomplete"
            if pending_work_unit_count > 0
            else
            "initial_pass_applied_pending_validation"
            if validation_required
            else "completed"
        ),
        "attempt_count": len(attempts),
        "execution_attempt_count": sum(
            int(attempt.get("attempt_number") or 1)
            for attempt in attempts
        ),
        "unit_replan_limit": INITIAL_UNIT_REPLAN_LIMIT,
        "applied_count": sum(
            result.get("status") == "applied"
            for attempt in attempts
            for result in attempt.get("apply_results") or []
        ),
        "no_patch_count": sum(
            result.get("status") == "no_patch"
            for attempt in attempts
            for result in attempt.get("apply_results") or []
        ),
        "needs_escalation_count": sum(
            result.get("status") == "needs_escalation"
            for attempt in attempts
            for result in attempt.get("apply_results") or []
        ),
        "validation_required": validation_required,
        "pending_changed_files": sorted(pending_changed_files),
        "static_integration_batch": static_integration_batch,
        "post_materialization_validation_records": list(
            static_integration_batch.get("validation_records") or []
        ),
        "initial_pass_chunks": batch_records,
        "active_batch": active_batch,
        "initial_pass_policy": (
            "all_topological_chunks_then_one_compile"
        ),
        "intermediate_compile": False,
        "apply_failed_contracts": list(
            queue.get("apply_failed_contract_identities") or []
        ),
        "applied_pending_compile_confirmation_contracts": list(
            queue.get(
                "applied_pending_compile_confirmation_contract_identities"
            )
            or []
        ),
        "apply_failed_work_unit_ids": list(
            queue.get("apply_failed_work_unit_ids") or []
        ),
        "applied_pending_compile_confirmation_work_unit_ids": list(
            queue.get(
                "applied_pending_compile_confirmation_work_unit_ids"
            )
            or []
        ),
        "pending_work_unit_count": pending_work_unit_count,
        "queue": queue,
        "incremental_state_path": str(incremental_state_path.resolve()),
        "attempts_path": str(attempts_path.resolve()),
    }
    write_json(output_dir / "llm_execution_result.json", result)
    return result
