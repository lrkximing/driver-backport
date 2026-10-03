#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import replace
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from driver_backport.validation.build_utils import find_make_binary, parse_kconfig_assignment
from driver_backport.analysis.code_index import semantic_graph_gap_ids
from driver_backport.analysis.diagnostic_identity import diagnostic_signature
from driver_backport.analysis.kbuild_index import KbuildIndex
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    cycle_hint_from_path,
    ensure_dir,
    load_json,
    normalize_relpath,
    normalize_root_relative,
    slugify,
    stable_id,
    utc_now,
    write_json,
)
from driver_backport.core.record_log import (
    ROUND_MANIFEST_COLLECTIONS,
    append_event,
    append_jsonl_records,
    append_round_manifest_collections,
    pending_repair_round_manifests,
    round_manifest_collections,
    round_manifest_event_log_path,
    validated_repair_round_count,
    validated_repair_round_counts,
)
from driver_backport.core.cycle_state import CycleState
from driver_backport.core.dataset_cases import materialize_dataset_cases, parse_row_selector
from driver_backport.core.executor import rollback_transaction
from driver_backport.llm.model_util import (
    gpt_api_with_metadata,
    has_model_config,
    model_api_error_metadata,
)
from driver_backport.planning.planner import (
    authorize_repair_plan_escalation,
    build_repair_plan,
    continue_repair_plan_graph,
    exact_active_patch_postimage_boundary,
    materialize_deferred_repair_plan_units,
    summarize_repair_plan_units,
)
from driver_backport.planning.plan_store import (
    load_repair_plan,
    write_repair_plan,
)
from driver_backport.planning.agentic_repair import (
    EntityRepresentationAdaptationRequired,
    agentic_apply_redundancy,
    agentic_blocked_apply_metadata,
    agentic_evidence_ready_for_apply,
    agentic_inspect_redundancy,
    agentic_materialization_rejection,
    agentic_prior_apply_redundancy,
    agentic_source_evidence_fingerprint,
    agentic_patch_text,
    build_agentic_apply_plan,
    build_agentic_entity_materialization_action,
    build_agentic_prompt,
    build_agentic_scope,
    collect_agentic_recovery_evidence,
    collect_agentic_evidence,
    graph_escalation_target,
    graph_scope_expansion_request,
    latest_agentic_failure_delta,
    parse_agentic_action,
    unique_agentic_entity_materialization_request,
)
from driver_backport.planning.build_config_preflight import run_build_config_preflight
from driver_backport.planning.diagnostic_routing import (
    build_prep_stage_decision,
    diagnostic_route_summary,
    validation_phase_issue_pairs,
)
from driver_backport.planning.prompting import (
    repair_plan_units,
)
from driver_backport.scope.compile_context_materialization import (
    materialize_transplanted_compile_context,
)
from driver_backport.scope.semantic_scope import (
    build_semantic_migration_scope,
)
from driver_backport.planning.response_apply import (
    apply_plan_responses,
    canonical_patch_signature,
    touched_files_from_diff,
)
from driver_backport.planning.root_binding import graph_root_binding_key
from driver_backport.planning.apply_protocol import (
    APPLY_FAILURE_RESULT_STATUSES,
    APPLY_INCOMPLETE_RESULT_STATUSES,
    apply_failure_route,
    patch_delta_summary,
    summarize_apply_results,
)
from driver_backport.planning.patch_outcome_ledger import (
    build_patch_outcome_candidate,
    record_round_patch_outcomes,
    rejected_patch_outcome,
)
from driver_backport.planning.sanity_check import (
    unified_diff_changed_ranges,
    unified_diff_hunk_ranges,
)
from driver_backport.planning.dependency_schedule import (
    select_dependency_ready_frontier,
)
from driver_backport.migration.transplant import (
    apply_initial_transplant_plan,
    build_initial_transplant_plan,
    verify_transplant_prerequisites,
)
from driver_backport.migration.scope_expansion import (
    complete_graph_scope_validations,
    commit_graph_scope,
    graph_scope_expansion_summary,
    load_graph_scope_expansions,
    materialize_graph_scope,
    pending_graph_scope_expansions,
    pending_graph_scope_validations,
    reconcile_graph_scope_transactions,
)
from driver_backport.migration.initial_compatibility import (
    execute_initial_llm_units,
    finalize_initial_compatibility_pass,
    finalize_initial_static_integration_batch,
    initial_compatibility_queue_status,
    prepare_initial_compatibility,
)
from driver_backport.migration.exact_export_closure import (
    materialize_ready_exact_exports,
)
from driver_backport.migration.semantic_closure_controller import (
    SemanticCompileRecoveryLifecycle,
    reconcile_active_semantic_batch_transaction,
    run_semantic_closure_controller,
    semantic_closure_resume_context,
)
from driver_backport.migration.semantic_closure_evaluator import (
    COMPILE_VALID_HANDOFF_EVALUATOR_REVISION,
    evaluate_compile_valid_semantic_handoff,
)
from driver_backport.migration.semantic_closure_runtime import (
    DefaultSemanticClosureRuntime,
)
from driver_backport.migration.semantic_compile_recovery import (
    bind_semantic_compile_recovery_work_units,
)
from driver_backport.migration.workflow_state import (
    migration_state_is_resumable,
    project_workflow_resume,
)
from driver_backport.migration.semantic_closure_state import (
    semantic_closure_summary as isolated_semantic_closure_summary,
)
from driver_backport.migration.semantic_closure_store import (
    initialize_semantic_closure_from_artifacts,
    load_semantic_closure_state,
    load_semantic_record,
    snapshot_initial_compatibility_outcomes,
)
from driver_backport.validation.validator import (
    run_validation,
    sync_build_kconfig,
)
from driver_backport.validation.compile_coverage import (
    diagnostic_files_resolution_coverage,
    merged_validation_translation_unit_coverage,
)
from driver_backport.planning.work_unit import (
    derive_work_units,
    exact_entity_materialization_owner_ids,
    load_work_units,
    mark_work_unit_abandoned,
    mark_work_unit_escalated,
    mark_work_unit_failed,
    mark_work_unit_succeeded,
    mark_work_unit_validation_deferred,
    promote_work_unit_to_planning,
    promote_work_unit_to_validating,
    write_work_unit,
    write_work_units,
)
from driver_backport.planning.execution_phase import (
    EXECUTION_PHASE_COMPILER_REPAIR,
    EXECUTION_PHASE_SEMANTIC_CLOSURE,
    bind_execution_phase,
    repair_phase_budget_status,
    repair_plan_execution_phase,
    validate_execution_phase,
)


DEFAULT_STATE_BASE = Path(__file__).resolve().parents[1] / "state"
CANONICAL_REPAIR_CONTROLLER_RESULT_SCHEMA = (
    "canonical_repair_controller_result_v1"
)

def state_root_for(spec: CaseSpec, override: Path | None) -> Path:
    if override:
        return override.resolve()
    return spec.state_root_path(DEFAULT_STATE_BASE.resolve())


def _effective_semantic_summary(
    state_root: Path,
    *,
    case_id: str,
) -> dict[str, Any]:
    scope_summary = graph_scope_expansion_summary(
        state_root,
        case_id=case_id,
    )
    isolated_state = load_semantic_closure_state(
        state_root,
        case_id=case_id,
    )
    if isolated_state is not None:
        return {
            **isolated_semantic_closure_summary(isolated_state),
            "scope_expansion_count": scope_summary[
                "scope_expansion_count"
            ],
        }
    return {
        "enabled": False,
        "closure_status": "not_initialized",
        "static_semantic_closed": False,
        "open_gap_count": 0,
        "scope_expansion_count": scope_summary[
            "scope_expansion_count"
        ],
    }


def state_file_for(state_root: Path) -> Path:
    return state_root / "cycle_state.json"


def journal_file_for(state_root: Path) -> Path:
    return state_root / "journal.jsonl"


def _initial_copy_set_from_cycle(cycle_dir: Path) -> list[str]:
    cycle_dir = cycle_dir.resolve()
    for name in ("frontier.pre_transplant.json", "frontier.json"):
        path = cycle_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        values = [
            normalize_relpath(str(item))
            for item in payload.get("initial_copy_set") or []
            if str(item).strip()
        ]
        if values:
            return list(dict.fromkeys(values))
    plan_path = cycle_dir / "initial_transplant_plan.json"
    if not plan_path.exists():
        return []
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return list(
        dict.fromkeys(
            normalize_relpath(str(item.get("file") or ""))
            for item in plan.get("copy_operations") or []
            if str(item.get("file") or "").strip()
        )
    )


def progress_stage_log_file_for(state_root: Path) -> Path:
    return state_root / "migration_stage_events.jsonl"




def _recover_open_transactions(
    spec: CaseSpec,
    *,
    state_root: Path,
) -> dict[str, Any]:
    """Rollback orphaned transactions while retaining an owned semantic batch."""

    try:
        semantic_batch = reconcile_active_semantic_batch_transaction(
            spec,
            state_root=state_root,
        )
    except (OSError, ValueError, KeyError) as exc:
        semantic_batch = {
            "status": "failed",
            "reason": "semantic_batch_transaction_reconciliation_failed",
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    if semantic_batch.get("status") == "failed":
        return {
            "schema": "open_transaction_recovery_v2",
            "status": "failed",
            "open_transaction_count": 0,
            "protected_open_transaction_count": 0,
            "rolled_back_count": 0,
            "rollbacks": [],
            "errors": [],
            "invalid_manifests": [],
            "semantic_batch": semantic_batch,
        }
    protected_transactions = {
        str(Path(reference).resolve())
        for reference in [semantic_batch.get("transaction_ref")]
        if str(reference or "")
    }

    open_transactions: list[tuple[str, Path]] = []
    protected_open_transactions: list[str] = []
    invalid_manifests: list[dict[str, str]] = []
    for manifest_path in (state_root / "transactions").glob("**/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            invalid_manifests.append(
                {
                    "manifest_path": str(manifest_path.resolve()),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue
        if str(manifest.get("case_id") or "") != spec.case_id:
            continue
        if str(manifest.get("status") or "open") != "open":
            continue
        transaction_dir = manifest_path.parent.resolve()
        if str(transaction_dir) in protected_transactions:
            protected_open_transactions.append(str(transaction_dir))
            continue
        open_transactions.append(
            (str(manifest.get("created_at") or ""), transaction_dir)
        )

    rollbacks: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for _created_at, transaction_dir in sorted(open_transactions, reverse=True):
        try:
            rollbacks.append(rollback_transaction(spec, transaction_dir))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(
                {
                    "transaction_dir": str(transaction_dir),
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
    return {
        "schema": "open_transaction_recovery_v2",
        "status": (
            "failed"
            if errors
            else "rolled_back"
            if rollbacks
            else "clean"
        ),
        "open_transaction_count": len(open_transactions),
        "protected_open_transaction_count": len(
            protected_open_transactions
        ),
        "protected_open_transactions": protected_open_transactions,
        "rolled_back_count": len(rollbacks),
        "rollbacks": rollbacks,
        "errors": errors,
        "invalid_manifests": invalid_manifests,
        "semantic_batch": semantic_batch,
    }


def _applied_transaction_result_for_work_unit(
    state_root: Path,
    *,
    work_unit_id: str,
) -> dict[str, Any] | None:
    results: list[dict[str, Any]] = []
    for manifest_path in (state_root / "transactions").glob("**/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        status = str(manifest.get("status") or "")
        if (
            str(manifest.get("work_unit_id") or "") != work_unit_id
            or status != "applied"
        ):
            continue
        results.append(
            {
                "unit_id": work_unit_id,
                "status": status,
                "transaction_dir": str(manifest_path.parent.resolve()),
                "patch_signature": manifest.get("patch_signature"),
                "patch_path": manifest.get("patch_path"),
                "touched_files": list(manifest.get("touched_files") or []),
                **(
                    {
                        "patch_outcome_candidate": dict(
                            manifest["patch_outcome_candidate"]
                        )
                    }
                    if isinstance(
                        manifest.get("patch_outcome_candidate"),
                        dict,
                    )
                    else {}
                ),
            }
        )
    if not results:
        return None
    return {
        "status": "applied",
        "applied_count": len(results),
        "results": results,
        "recovered_from_transaction_manifests": True,
    }



def _new_run_id(spec: CaseSpec) -> str:
    timestamp = utc_now().replace(":", "").replace("+", "z")
    return f"{spec.case_id}-{timestamp}"


def _emit_progress(message: str) -> None:
    print(f"[{utc_now()}] {message}", file=sys.stderr, flush=True)


def _write_migration_progress(
    *,
    spec: CaseSpec,
    state_root: Path,
    run_id: str,
    started_at: str,
    stages: list[dict],
    status: str,
    current_stage: str | None = None,
    stop_reason: str | None = None,
    repair_rounds_completed: int = 0,
    extra: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "case_id": spec.case_id,
        "run_id": run_id,
        "started_at": started_at,
        "updated_at": utc_now(),
        "status": status,
        "current_stage": current_stage,
        "stop_reason": stop_reason,
        "repair_rounds_completed": repair_rounds_completed,
        "stage_count": len(stages),
        "state_root": str(state_root.resolve()),
        "cycle_state_path": str(state_file_for(state_root).resolve()),
        "journal_path": str(journal_file_for(state_root).resolve()),
        "stage_event_log_path": str(
            progress_stage_log_file_for(state_root).resolve()
        ),
        "stages": list(stages),
    }
    if extra:
        payload.update(extra)
    path = state_root / "migration_progress.json"
    write_json(path, payload)
    return str(path.resolve())


def _record_migration_stage(
    *,
    spec: CaseSpec,
    state_root: Path,
    run_id: str,
    started_at: str,
    stages: list[dict],
    stage: str,
    result: dict,
    repair_rounds_completed: int = 0,
) -> dict:
    sequence = len(stages) + 1
    record = {
        "schema": "migration_stage_event_v1",
        "case_id": spec.case_id,
        "run_id": run_id,
        "sequence": sequence,
        "stage": stage,
        "completed_at": utc_now(),
        "result": result,
    }
    append_jsonl_records(
        progress_stage_log_file_for(state_root),
        [record],
    )
    stages.append(
        {
            "stage": stage,
            "completed_at": record["completed_at"],
            "sequence": sequence,
            "result_status": result.get("status"),
            "overall_ok": result.get("overall_ok"),
        }
    )
    _emit_progress(f"completed stage: {stage}")
    _write_migration_progress(
        spec=spec,
        state_root=state_root,
        run_id=run_id,
        started_at=started_at,
        stages=stages,
        status="running",
        current_stage=stage,
        repair_rounds_completed=repair_rounds_completed,
    )
    return record


def _validation_issue_signature(issue: dict[str, Any]) -> str:
    existing = issue.get("diagnostic_signature")
    if existing not in (None, "", [], {}):
        return str(existing)
    return diagnostic_signature(issue)


def _validation_diagnostic_signatures(validation: dict[str, Any]) -> set[str]:
    signatures: set[str] = set()
    for _phase, issue in validation_phase_issue_pairs(validation):
        signature = _validation_issue_signature(issue)
        if signature.strip("|"):
            signatures.add(signature)
    return signatures


def _validation_diagnostics_by_signature(
    validation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        _validation_issue_signature(issue): issue
        for _phase, issue in validation_phase_issue_pairs(validation)
        if _validation_issue_signature(issue).strip("|")
    }


def _classify_validation_delta(
    previous_validation: dict[str, Any],
    current_validation: dict[str, Any],
    *,
    spec: CaseSpec | None = None,
) -> dict[str, Any]:
    previous = _validation_diagnostic_signatures(previous_validation)
    current = _validation_diagnostic_signatures(current_validation)
    candidate_resolved = previous - current
    resolved = set(candidate_resolved)
    unverified_resolved: set[str] = set()
    resolution_coverage: dict[str, dict[str, Any]] = {}
    if spec is not None and candidate_resolved:
        coverage = merged_validation_translation_unit_coverage(
            current_validation
        )
        previous_by_signature = _validation_diagnostics_by_signature(
            previous_validation
        )
        resolved = set()
        for signature in candidate_resolved:
            issue = previous_by_signature.get(signature) or {}
            evidence = diagnostic_files_resolution_coverage(
                [
                    issue.get("file")
                    or issue.get("source_file")
                    or issue.get("path")
                ],
                coverage=coverage,
                old_root=spec.old_root_path,
                preferred_prefix=spec.driver_root,
            )
            resolution_coverage[signature] = evidence
            if evidence.get("covered"):
                resolved.add(signature)
            else:
                unverified_resolved.add(signature)
    new = current - previous
    persisted = previous & current
    if previous and not current:
        status = "resolved" if not unverified_resolved else "coverage_deferred"
    elif current == previous:
        status = "unchanged"
    elif new and not resolved:
        status = (
            "changed_with_unverified_frontier"
            if unverified_resolved
            else "regressed"
        )
    elif resolved:
        status = "progressed"
    elif unverified_resolved:
        status = "coverage_deferred"
    else:
        status = "changed"
    return {
        "schema": "validation_delta_v3",
        "basis": "diagnostic_event_set_with_compile_coverage",
        "status": status,
        "previous_overall_ok": bool(
            previous_validation.get("overall_ok")
        ),
        "current_overall_ok": bool(
            current_validation.get("overall_ok")
        ),
        "compile_regressed_from_valid_checkpoint": bool(
            previous_validation.get("overall_ok")
            and not current_validation.get("overall_ok")
        ),
        "frontier_advanced": bool(resolved and new),
        "previous_signature_count": len(previous),
        "current_signature_count": len(current),
        "persisted_signature_count": len(persisted),
        "resolved_signature_count": len(resolved),
        "unverified_resolved_signature_count": len(unverified_resolved),
        "new_signature_count": len(new),
        "persisted_signatures": sorted(persisted)[:24],
        "resolved_signatures": sorted(resolved)[:24],
        "unverified_resolved_signatures": sorted(unverified_resolved)[:24],
        "new_signatures": sorted(new)[:24],
        "resolution_coverage": {
            signature: resolution_coverage[signature]
            for signature in sorted(resolution_coverage)[:24]
        },
    }


def _unmasked_apply_diagnostics(
    apply_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return diagnostics proven to pre-exist behind an earlier fatal error.

    Provider-contract validation parses both the preimage and the candidate
    translation unit.  Its ``unmasked_translation_unit_diagnostics`` records
    are therefore stronger evidence than source-line proximity: they identify
    errors that become visible after an upstream fatal diagnostic disappears,
    but were not introduced by the patch.  Preserve only exact projections to
    unchanged source so later exact-build attribution cannot mistake these
    downstream-frontier observations for patch-locus regressions.
    """

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for result in apply_result.get("results") or []:
        if not isinstance(result, dict):
            continue
        report = result.get("provider_contract_report")
        if not isinstance(report, dict):
            continue
        for diagnostic in report.get(
            "unmasked_translation_unit_diagnostics"
        ) or []:
            if not isinstance(diagnostic, dict):
                continue
            spelling = str(diagnostic.get("spelling") or "")
            for location in diagnostic.get("current_locations") or []:
                if not isinstance(location, dict) or str(
                    location.get("projection") or ""
                ) != "exact_unique_unchanged_diff_block":
                    continue
                relative_path = normalize_relpath(
                    str(location.get("file") or diagnostic.get("file") or "")
                )
                try:
                    line = int(
                        location.get("patched_line")
                        or location.get("line")
                        or 0
                    )
                except (TypeError, ValueError):
                    line = 0
                if not relative_path or line <= 0:
                    continue
                identity = (relative_path, line, spelling)
                if identity in seen:
                    continue
                seen.add(identity)
                records.append(
                    {
                        "file": relative_path,
                        "line": line,
                        "spelling": spelling,
                        "projection": (
                            "exact_unique_unchanged_diff_block"
                        ),
                        "source": "provider_contract_preimage_comparison",
                    }
                )
    return records


def _classify_target_root_validation_delta(
    validation_delta: dict[str, Any],
    *,
    validation_target: dict[str, Any] | None,
    fresh_work_units: list[dict[str, Any]],
    deferred_work_unit_ids: set[str] | None = None,
    spec: CaseSpec | None = None,
) -> dict[str, Any]:
    """Refine patch validation against the exact CodeIndex root transaction.

    A successful repair of an upstream root can reveal a larger downstream
    diagnostic frontier.  Total diagnostic count therefore cannot decide
    whether the patch regressed.  The apply plan carries the stable source
    work-unit identity, so compare that root's diagnostic set with the freshly
    rebuilt graph instead.
    """

    result = dict(validation_delta)
    if not validation_target:
        return result

    materialized_work_unit_id = str(
        validation_target.get("work_unit_id") or ""
    )
    target_work_unit_id = str(
        validation_target.get("origin_work_unit_id")
        or materialized_work_unit_id
    )
    if not target_work_unit_id:
        return result

    current_by_id = {
        str(unit.get("id") or ""): unit
        for unit in fresh_work_units
        if isinstance(unit, dict) and unit.get("id")
    }
    decision = dict(
        validation_target.get("origin_repair_decision")
        or validation_target.get("repair_decision")
        or {}
    )
    target_root_node_id = str(decision.get("root_node_id") or "") or None
    target_root_key = str(decision.get("root_key") or "") or None
    previous_target_signatures = {
        str(signature)
        for signature in (
            validation_target.get("origin_diagnostic_signatures")
            or validation_target.get("diagnostic_signatures")
            or []
        )
        if str(signature).strip("|")
    }
    current_target_units: list[dict[str, Any]] = []
    seen_current_ids: set[str] = set()

    def add_current_target(unit: dict[str, Any] | None) -> None:
        if not isinstance(unit, dict):
            return
        unit_id = str(unit.get("id") or "")
        identity = unit_id or f"anonymous-{id(unit)}"
        if identity in seen_current_ids:
            return
        seen_current_ids.add(identity)
        current_target_units.append(unit)

    add_current_target(current_by_id.get(target_work_unit_id))
    if materialized_work_unit_id != target_work_unit_id:
        add_current_target(current_by_id.get(materialized_work_unit_id))
    for unit in fresh_work_units:
        if not isinstance(unit, dict):
            continue
        current_signatures = {
            str(signature)
            for signature in unit.get("diagnostic_signatures") or []
            if str(signature).strip("|")
        }
        if (
            previous_target_signatures & current_signatures
        ):
            add_current_target(unit)
    current_target_signatures = {
        str(signature)
        for unit in current_target_units
        for signature in unit.get("diagnostic_signatures") or []
        if str(signature).strip("|")
    }
    current_target = current_target_units[0] if current_target_units else None
    deferred_ids = set(deferred_work_unit_ids or set())
    target_validation_deferred = bool(
        not current_target_units
        and {
            target_work_unit_id,
            materialized_work_unit_id,
        }
        & deferred_ids
    )
    pre_patch_validation_signatures = {
        str(signature)
        for signature in validation_target.get(
            "pre_patch_validation_signatures"
        )
        or previous_target_signatures
        if str(signature).strip("|")
    }

    def canonical_old_path(value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if spec is not None:
            relative = normalize_root_relative(
                text,
                root=spec.old_root_path,
                preferred_prefix=spec.driver_root,
            )
            if relative:
                return relative
        return normalize_relpath(text)

    loci_by_file: dict[str, list[dict[str, Any]]] = {}
    for locus in validation_target.get("applied_patch_loci") or []:
        if not isinstance(locus, dict):
            continue
        relative_path = canonical_old_path(locus.get("file"))
        if relative_path:
            loci_by_file.setdefault(relative_path, []).append(locus)
    patch_locus_regressions: list[dict[str, Any]] = []
    unmasked_frontier_diagnostics: list[dict[str, Any]] = []
    prevalidated_unmasked = [
        record
        for record in validation_target.get(
            "prevalidated_unmasked_diagnostics"
        )
        or []
        if isinstance(record, dict)
    ]
    seen_locus_regressions: set[tuple[str, str, int]] = set()
    for unit in fresh_work_units:
        if not isinstance(unit, dict):
            continue
        issues = [
            issue
            for issue in [
                *list(unit.get("source_issues") or []),
                *list(unit.get("source_issue_samples") or []),
            ]
            if isinstance(issue, dict)
        ]
        for issue in issues:
            signature = _validation_issue_signature(issue)
            if not signature or signature in pre_patch_validation_signatures:
                continue
            relative_path = canonical_old_path(
                issue.get("file")
                or issue.get("source_file")
                or issue.get("path")
            )
            try:
                line = int(issue.get("line") or 0)
            except (TypeError, ValueError):
                line = 0
            matching_locus = next(
                (
                    locus
                    for locus in loci_by_file.get(relative_path, [])
                    if int(locus.get("new_start") or 0)
                    <= line
                    <= int(locus.get("new_end") or 0)
                ),
                None,
            )
            identity = (signature, relative_path, line)
            if matching_locus is None or identity in seen_locus_regressions:
                continue
            unmasked_evidence = next(
                (
                    record
                    for record in prevalidated_unmasked
                    if canonical_old_path(record.get("file"))
                    == relative_path
                    and int(record.get("line") or 0) == line
                    and str(record.get("projection") or "")
                    == "exact_unique_unchanged_diff_block"
                ),
                None,
            )
            if unmasked_evidence is not None:
                unmasked_frontier_diagnostics.append(
                    {
                        "diagnostic_signature": signature,
                        "file": relative_path,
                        "line": line,
                        "message": issue.get("message"),
                        "work_unit_id": unit.get("id"),
                        "evidence": unmasked_evidence,
                    }
                )
                continue
            seen_locus_regressions.add(identity)
            patch_locus_regressions.append(
                {
                    "diagnostic_signature": signature,
                    "file": relative_path,
                    "line": line,
                    "message": issue.get("message"),
                    "work_unit_id": unit.get("id"),
                    "patch_locus": matching_locus,
                }
            )

    result.update(
        {
            "schema": "target_root_validation_delta_v1",
            "basis": "target_code_graph_root",
            "target_work_unit_id": target_work_unit_id,
            "target_materialized_work_unit_id": (
                materialized_work_unit_id
                if materialized_work_unit_id != target_work_unit_id
                else None
            ),
            "target_rebound_work_unit_ids": [
                str(unit.get("id") or "")
                for unit in current_target_units
                if str(unit.get("id") or "")
                not in {target_work_unit_id, materialized_work_unit_id}
            ],
            "target_root_node_id": target_root_node_id,
            "target_root_key": target_root_key,
            "target_root_closed": (
                current_target is None and not target_validation_deferred
            ),
            "target_validation_deferred": target_validation_deferred,
            "patch_locus_regression_count": len(
                patch_locus_regressions
            ),
            "patch_locus_regressions": patch_locus_regressions[:24],
            "unmasked_frontier_diagnostic_count": len(
                unmasked_frontier_diagnostics
            ),
            "unmasked_frontier_diagnostics": (
                unmasked_frontier_diagnostics[:24]
            ),
        }
    )
    if current_target is None and target_validation_deferred:
        result["status"] = "validation_deferred"
        result["patch_locus_causal_status"] = "unassessed_root_deferred"
        result["frontier_advanced"] = False
        result["target_previous_signature_count"] = len(
            previous_target_signatures
        )
        result["target_current_signature_count"] = 0
        result["target_persisted_signature_count"] = 0
        result["target_resolved_signature_count"] = 0
        result["target_unobserved_signature_count"] = len(
            previous_target_signatures
        )
        result["target_new_signature_count"] = 0
        return result
    if current_target is None:
        result["status"] = (
            "resolved" if not fresh_work_units else "progressed"
        )
        result["patch_locus_causal_status"] = (
            "unbound_fresh_observation"
            if patch_locus_regressions
            else "not_observed"
        )
        result["frontier_advanced"] = bool(fresh_work_units)
        result["target_previous_signature_count"] = len(previous_target_signatures)
        result["target_current_signature_count"] = 0
        result["target_persisted_signature_count"] = 0
        result["target_resolved_signature_count"] = len(previous_target_signatures)
        result["target_new_signature_count"] = 0
        return result

    target_resolved = previous_target_signatures - current_target_signatures
    target_new = current_target_signatures - previous_target_signatures
    target_persisted = previous_target_signatures & current_target_signatures
    if current_target_signatures == previous_target_signatures:
        status = "unchanged"
    elif target_new and not target_resolved:
        status = "regressed"
    elif target_resolved and not target_new:
        status = "progressed"
    else:
        # The same root still exists, so replacing some of its diagnostics with
        # new ones is not a downstream-frontier advance.  A root advance is
        # recognized above only when this root disappears and fresh work units
        # expose the next dependency frontier.
        status = "changed"
    patch_locus_signatures = {
        str(item.get("diagnostic_signature") or "")
        for item in patch_locus_regressions
        if str(item.get("diagnostic_signature") or "")
    }
    patch_locus_regression = bool(
        status == "regressed"
        and previous_target_signatures
        and target_new
        and not target_resolved
        and patch_locus_signatures & target_new
    )
    if patch_locus_regression:
        status = "regressed_at_patch_locus"
    result.update(
        {
            "status": status,
            "patch_locus_causal_status": (
                "same_root_new_patch_locus_diagnostic"
                if patch_locus_regression
                else "observed_without_same_root_regression"
                if patch_locus_regressions
                else "not_observed"
            ),
            "frontier_advanced": False,
            "target_previous_signature_count": len(previous_target_signatures),
            "target_current_signature_count": len(current_target_signatures),
            "target_persisted_signature_count": len(target_persisted),
            "target_resolved_signature_count": len(target_resolved),
            "target_new_signature_count": len(target_new),
            "target_persisted_signatures": sorted(target_persisted)[:24],
            "target_resolved_signatures": sorted(target_resolved)[:24],
            "target_new_signatures": sorted(target_new)[:24],
        }
    )
    return result


def _classify_round_root_validation_delta(
    validation_delta: dict[str, Any],
    *,
    validation_targets: list[dict[str, Any]],
    fresh_work_units: list[dict[str, Any]],
    deferred_work_unit_ids: set[str] | None = None,
    spec: CaseSpec | None = None,
) -> dict[str, Any]:
    target_results = [
        _classify_target_root_validation_delta(
            validation_delta,
            validation_target=target,
            fresh_work_units=fresh_work_units,
            deferred_work_unit_ids=deferred_work_unit_ids,
            spec=spec,
        )
        for target in validation_targets
        if isinstance(target, dict) and target.get("work_unit_id")
    ]
    compile_checkpoint_regressed = bool(
        validation_delta.get(
            "compile_regressed_from_valid_checkpoint"
        )
        or (
            any(
                bool(target.get("pre_patch_overall_ok"))
                for target in validation_targets
                if isinstance(target, dict)
            )
            and validation_delta.get("current_overall_ok") is False
            and "current_overall_ok" in validation_delta
        )
    )
    if compile_checkpoint_regressed:
        target_results = [
            {
                **item,
                "pre_compile_recovery_status": item.get("status"),
                "status": (
                    "regressed_at_patch_locus"
                    if str(item.get("status") or "")
                    == "regressed_at_patch_locus"
                    else "pending_compile_recovery"
                ),
                "frontier_advanced": False,
                "target_root_closed": False,
                "target_validation_deferred": False,
                "checkpoint_regression": True,
                "required_transition": (
                    "derive compiler diagnostics against the retained "
                    "candidate tree and continue the dependency component"
                ),
            }
            for item in target_results
        ]
    closed = [item for item in target_results if item.get("target_root_closed")]
    deferred = [
        item for item in target_results if item.get("target_validation_deferred")
    ]
    persisted = [
        item
        for item in target_results
        if not item.get("target_root_closed")
        and not item.get("target_validation_deferred")
    ]
    assessable = [
        item
        for item in target_results
        if not item.get("target_validation_deferred")
    ]
    productive_statuses = {"resolved", "progressed"}
    productive = [
        item
        for item in assessable
        if str(item.get("status") or "") in productive_statuses
    ]
    unproductive = [
        item
        for item in assessable
        if str(item.get("status") or "") not in productive_statuses
    ]
    statuses = {str(item.get("status") or "") for item in target_results}
    if compile_checkpoint_regressed:
        round_status = "compile_recovery_required"
    elif not assessable:
        round_status = "validation_deferred"
    elif len(closed) == len(target_results) and not fresh_work_units:
        round_status = "resolved"
    elif productive and unproductive:
        round_status = "mixed_root_progress"
    elif productive:
        round_status = "progressed"
    elif statuses & {"regressed", "regressed_at_patch_locus"}:
        round_status = "diagnostics_regressed"
    elif statuses == {"unchanged"}:
        round_status = "unchanged"
    else:
        round_status = "diagnostics_changed"
    result = dict(validation_delta)
    result.update(
        {
            "schema": "validation_delta_v3",
            "basis": "ordered_graph_root_round",
            "status": round_status,
            "round_target_count": len(target_results),
            "round_closed_target_count": len(closed),
            "round_persisted_target_count": len(persisted),
            "round_deferred_target_count": len(deferred),
            "round_assessable_target_count": len(assessable),
            "round_productive_target_count": len(productive),
            "round_unproductive_target_count": len(unproductive),
            "round_candidate_state": (
                "retained_for_compile_recovery"
                if compile_checkpoint_regressed
                else "retained_with_exact_root_progress"
                if productive
                else "validation_deferred"
                if not assessable
                else "retained_with_open_diagnostics"
            ),
            "round_closed_work_unit_ids": [item.get("target_work_unit_id") for item in closed],
            "round_persisted_work_unit_ids": [item.get("target_work_unit_id") for item in persisted],
            "round_deferred_work_unit_ids": [
                item.get("target_work_unit_id") for item in deferred
            ],
            "round_productive_work_unit_ids": [
                item.get("target_work_unit_id") for item in productive
            ],
            "round_unproductive_work_unit_ids": [
                item.get("target_work_unit_id") for item in unproductive
            ],
            "target_results": target_results,
        }
    )
    return result


def _local_plan_path_for_apply_summary(apply_path: Path, apply_summary: dict[str, Any]) -> Path | None:
    raw_path = str(apply_summary.get("source_plan_path") or apply_summary.get("plan_path") or "").strip()
    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate.resolve()
        cycle_dir = next((parent for parent in apply_path.parents if parent.name.startswith("cycle-")), None)
        if cycle_dir is not None:
            localized = cycle_dir / candidate.name
            if localized.exists():
                return localized.resolve()
    cycle_dir = next((parent for parent in apply_path.parents if parent.name.startswith("cycle-")), None)
    if cycle_dir is None:
        return None
    candidate = cycle_dir / "repair_plan.json"
    if candidate.exists():
        return candidate.resolve()
    return None


def _patch_excerpt_from_apply_result(result: dict[str, Any]) -> list[str]:
    return _patch_lines_from_apply_result(result)[:80]


def _patch_lines_from_apply_result(result: dict[str, Any]) -> list[str]:
    payload = dict(result.get("payload") or {})
    patch = dict(payload.get("patch") or {})
    content_lines = [str(line) for line in patch.get("content_lines") or []]
    if content_lines:
        return content_lines
    patch_path = str(result.get("patch_path") or "").strip()
    if not patch_path:
        return []
    path = Path(patch_path)
    if path.exists():
        try:
            return path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []
    return []


def _applied_patch_loci(
    apply_result: dict[str, Any],
) -> list[dict[str, Any]]:
    loci: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for result in apply_result.get("results") or []:
        if not isinstance(result, dict) or str(result.get("status") or "") not in {
            "applied",
            "already_applied",
            "already_applied_duplicate",
        }:
            continue
        diff_text = "\n".join(_patch_lines_from_apply_result(result))
        for hunk in unified_diff_hunk_ranges(diff_text):
            identity = (
                str(hunk.get("file") or ""),
                int(hunk.get("old_start") or 0),
                int(hunk.get("old_end") or 0),
                int(hunk.get("new_start") or 0),
                int(hunk.get("new_end") or 0),
            )
            if identity in seen:
                continue
            seen.add(identity)
            loci.append(hunk)
    return loci


def _applied_changed_files(
    apply_result: dict[str, Any],
) -> list[str]:
    """Return files changed by applied transactions, never rejected actions."""

    changed: list[str] = []
    for result in apply_result.get("results") or []:
        if (
            not isinstance(result, dict)
            or str(result.get("status") or "") != "applied"
        ):
            continue
        diff_text = "\n".join(_patch_lines_from_apply_result(result))
        diff_files = [
            normalize_relpath(str(hunk.get("file") or ""))
            for hunk in unified_diff_hunk_ranges(diff_text)
            if str(hunk.get("file") or "")
        ]
        changed.extend(
            diff_files
            or [
                normalize_relpath(str(path))
                for path in result.get("touched_files") or []
                if str(path)
            ]
        )
    return list(dict.fromkeys(path for path in changed if path))


def _applied_added_contract_definitions(
    apply_result: dict[str, Any],
) -> list[dict[str, str]]:
    """Return exact contracts introduced by applied candidate transactions."""

    contracts: dict[tuple[str, str, str], dict[str, str]] = {}
    for result in apply_result.get("results") or []:
        if (
            not isinstance(result, dict)
            or str(result.get("status") or "") != "applied"
        ):
            continue
        reports = [result.get("provider_contract_report")]
        patch_rebase = result.get("patch_rebase")
        if isinstance(patch_rebase, dict):
            reports.append(
                patch_rebase.get("rebased_provider_contract_report")
            )
        for report in reports:
            if not isinstance(report, dict):
                continue
            for record in report.get("added_contract_definitions") or []:
                if not isinstance(record, dict):
                    continue
                owner_file = normalize_relpath(
                    str(record.get("file") or "")
                )
                entity_kind = str(record.get("kind") or "").lower()
                contract_key = str(
                    record.get("key")
                    or record.get("contract_key")
                    or ""
                )
                identity = (owner_file, entity_kind, contract_key)
                if not all(identity):
                    continue
                contracts[identity] = {
                    "file": owner_file,
                    "entity_kind": entity_kind,
                    "contract_key": contract_key,
                }
    return [contracts[key] for key in sorted(contracts)]


def _validation_target_with_apply_evidence(
    validation_target: dict[str, Any],
    *,
    apply_result: dict[str, Any] | None,
    baseline_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    target = dict(validation_target)
    if baseline_validation is not None:
        target["pre_patch_overall_ok"] = bool(
            baseline_validation.get("overall_ok")
        )
        target["pre_patch_validation_signatures"] = sorted(
            _validation_diagnostic_signatures(baseline_validation)
        )
    if apply_result is not None:
        target["applied_patch_loci"] = _applied_patch_loci(apply_result)
        target["prevalidated_unmasked_diagnostics"] = (
            _unmasked_apply_diagnostics(apply_result)
        )
    return target


def _validation_target_for_plan_unit(
    repair_plan: dict[str, Any],
    *,
    selected_plan_unit_id: str,
) -> dict[str, Any] | None:
    matches = [
        unit
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and str(unit.get("id") or "") == selected_plan_unit_id
    ]
    if len(matches) != 1:
        return None
    unit = matches[0]
    work_unit_id = str(unit.get("work_unit_id") or unit.get("id") or "")
    if not work_unit_id:
        return None
    diagnostic_signatures = list(
        unit.get("diagnostic_signatures") or []
    )
    decision = dict(unit.get("repair_decision") or {})
    compact_decision = {
        key: decision.get(key)
        for key in (
            "mode",
            "root_node_id",
            "root_key",
        )
        if decision.get(key) not in (None, "", [], {})
    }
    recovery_context = dict(
        unit.get("semantic_recovery_context") or {}
    )
    compact_recovery_context = {
        key: copy.deepcopy(recovery_context.get(key))
        for key in (
            "schema",
            "batch_id",
            "batch_status",
            "incident_patch_ids",
            "semantic_transaction_ids",
            "exact_bindings",
            "transaction_file_bindings",
        )
        if recovery_context.get(key) not in (None, "", [], {})
    }
    return {
        "schema": "root_validation_target_v1",
        "id": str(unit.get("id") or selected_plan_unit_id),
        "work_unit_id": work_unit_id,
        "origin_work_unit_id": work_unit_id,
        "unit_kind": unit.get("unit_kind"),
        "execution_phase": unit.get("execution_phase"),
        "semantic_gap_ids": list(unit.get("semantic_gap_ids") or []),
        "required_semantic_gap_ids": list(
            unit.get("required_semantic_gap_ids") or []
        ),
        "pending_semantic_gap_ids": list(
            unit.get("pending_semantic_gap_ids") or []
        ),
        "incident_patch_ids": list(
            unit.get("incident_patch_ids") or []
        ),
        "semantic_transaction_ids": list(
            unit.get("semantic_transaction_ids") or []
        ),
        "semantic_recovery_context": compact_recovery_context,
        "diagnostic_signatures": diagnostic_signatures,
        "origin_diagnostic_signatures": diagnostic_signatures,
        "repair_decision": compact_decision,
        "origin_repair_decision": compact_decision,
    }


def _patch_outcome_candidate_for_action(
    spec: CaseSpec,
    *,
    action: dict[str, Any],
    repair_plan: dict[str, Any],
) -> dict[str, Any] | None:
    if action.get("action") != "apply_patch":
        return None
    patch_text = agentic_patch_text(action)
    touched_files = touched_files_from_diff(patch_text)
    if not patch_text.strip() or not touched_files:
        return None
    decision = (
        action.get("decision")
        if isinstance(action.get("decision"), dict)
        else {}
    )
    requested_unit_id = str(decision.get("work_unit_id") or "")
    matching_units = [
        unit
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and requested_unit_id
        in {
            str(unit.get("id") or ""),
            str(unit.get("work_unit_id") or ""),
        }
    ]
    if len(matching_units) != 1:
        return None
    plan_unit = matching_units[0]
    return build_patch_outcome_candidate(
        spec,
        patch_signature=canonical_patch_signature(patch_text),
        touched_files=touched_files,
        root_binding={
            "work_unit_id": requested_unit_id,
            "root_node_id": decision.get("root_node_id"),
            "root_key": decision.get("root_key"),
        },
        source_evidence_fingerprint=agentic_source_evidence_fingerprint(
            plan_unit
        ),
        diagnostic_signatures=plan_unit.get("diagnostic_signatures") or [],
        diff_text=patch_text,
    )


def _merge_round_assessment_target(
    origin_target: dict[str, Any] | None,
    applied_target: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if origin_target is None and applied_target is None:
        return None
    merged = copy.deepcopy(applied_target or origin_target or {})
    if origin_target is not None:
        origin_work_unit_id = str(
            origin_target.get("origin_work_unit_id")
            or origin_target.get("work_unit_id")
            or ""
        )
        merged["origin_work_unit_id"] = origin_work_unit_id
        merged["origin_diagnostic_signatures"] = list(
            origin_target.get("origin_diagnostic_signatures")
            or origin_target.get("diagnostic_signatures")
            or []
        )
        merged["origin_repair_decision"] = dict(
            origin_target.get("origin_repair_decision")
            or origin_target.get("repair_decision")
            or {}
        )
        for field in (
            "execution_phase",
            "semantic_gap_ids",
            "required_semantic_gap_ids",
            "pending_semantic_gap_ids",
            "incident_patch_ids",
            "semantic_transaction_ids",
            "semantic_recovery_context",
        ):
            if merged.get(field) in (None, "", [], {}) and (
                origin_target.get(field) not in (None, "", [], {})
            ):
                merged[field] = copy.deepcopy(
                    origin_target[field]
                )
    merged["repair_applied"] = applied_target is not None
    return merged


def _semantic_lifecycle_dependencies(
    assessment_target: dict[str, Any] | None,
) -> list[str]:
    """Return only active semantic-batch transaction dependencies."""

    return sorted(
        {
            str(patch_id)
            for patch_id in (
                (assessment_target or {}).get("semantic_transaction_ids")
                or []
            )
            if str(patch_id)
        }
    )


def _compact_previous_unresolved_attempt(
    *,
    apply_path: Path,
    result: dict[str, Any],
    work_unit_id: str,
    plan_unit: dict[str, Any],
) -> dict[str, Any]:
    status = str(result.get("status") or "")
    already_present = status in {"already_applied", "already_applied_duplicate"}
    compact: dict[str, Any] = {
        "source": "previous_applied_patch_still_unresolved",
        "cycle": cycle_hint_from_path(apply_path),
        "work_unit_id": work_unit_id,
        "plan_unit_id": result.get("unit_id"),
        "status": result.get("status"),
        "tree_change": "already_present" if already_present else "applied",
        "validation_relation": "work_unit_remains_open_after_patch",
        "source_evidence_fingerprint": agentic_source_evidence_fingerprint(
            plan_unit
        ),
        "touched_files": list(result.get("touched_files") or []),
        "patch_path": result.get("patch_path"),
        "patch_signature": result.get("patch_signature"),
        "patch_outcome_candidate_id": (
            (result.get("patch_outcome_candidate") or {}).get(
                "candidate_id"
            )
            if isinstance(
                result.get("patch_outcome_candidate"), dict
            )
            else None
        ),
        "transaction_dir": result.get("transaction_dir"),
        "note": "The exact patch and diagnostic snapshots are retained to prevent an identical retry.",
    }
    excerpt = _patch_excerpt_from_apply_result(result)
    if excerpt:
        compact["patch_excerpt_lines"] = excerpt
    patch_lines = _patch_lines_from_apply_result(result)
    if patch_lines:
        compact["patch_delta_summary"] = patch_delta_summary(patch_lines)
    compact["diagnostic_snapshot"] = _diagnostic_snapshot(plan_unit)
    compact["repair_evidence_snapshot"] = _repair_evidence_snapshot(
        plan_unit
    )
    return {key: value for key, value in compact.items() if value not in (None, "", [], {})}


def _compact_apply_failure_payload(
    failure: dict[str, Any],
) -> dict[str, Any]:
    compact = {
        key: failure.get(key)
        for key in (
            "schema",
            "failure_id",
            "stage",
            "recoverability",
            "finding_kind",
            "unit_id",
            "work_unit_id",
            "root_binding",
            "semantic_action_id",
            "source_preimage_digest",
            "evidence_set_digest",
            "diagnostic_set_digest",
            "patch_signature",
            "evidence_ids",
            "details",
        )
        if failure.get(key) not in (None, "", [], {})
    }
    deterministic_attempts = []
    for attempt in failure.get("deterministic_attempts") or []:
        if not isinstance(attempt, dict):
            continue
        deterministic_attempts.append(
            {
                key: attempt.get(key)
                for key in (
                    "kind",
                    "succeeded",
                    "checker",
                    "status",
                )
                if attempt.get(key) not in (None, "", [], {})
            }
        )
    if deterministic_attempts:
        compact["deterministic_attempts"] = deterministic_attempts
    return compact


def _compact_routed_apply_failure(
    result: dict[str, Any],
) -> dict[str, Any]:
    compact = {
        key: result.get(key)
        for key in (
            "unit_id",
            "work_unit_id",
            "status",
            "original_status",
            "error",
            "reason",
            "touched_files",
        )
        if result.get(key) not in (None, "", [], {})
    }
    failure = (
        result.get("apply_failure")
        if isinstance(result.get("apply_failure"), dict)
        else {}
    )
    if failure:
        compact["apply_failure"] = _compact_apply_failure_payload(failure)
    invalid_route = (
        result.get("invalid_route")
        if isinstance(result.get("invalid_route"), dict)
        else {}
    )
    if invalid_route:
        compact["invalid_route"] = {
            key: invalid_route.get(key)
            for key in (
                "route",
                "reason",
                "required_transition",
                "retry_requires_state_change",
            )
            if invalid_route.get(key) not in (None, "", [], {})
        }
    provider_report = (
        result.get("provider_contract_report")
        if isinstance(result.get("provider_contract_report"), dict)
        else {}
    )
    if provider_report:
        compact["provider_contract"] = {
            key: provider_report.get(key)
            for key in (
                "status",
                "reason",
                "blocking_findings",
            )
            if provider_report.get(key) not in (None, "", [], {})
        }
    return compact


def _compact_previous_deferred_unit_attempt(
    *,
    manifest_path: Path,
    outcome: dict[str, Any],
    work_unit_id: str,
    plan_unit: dict[str, Any],
) -> dict[str, Any]:
    blocker = outcome.get("blocker") if isinstance(outcome.get("blocker"), dict) else {}
    redundancy = (
        blocker.get("redundancy")
        if isinstance(blocker.get("redundancy"), dict)
        else {}
    )
    recovery = (
        blocker.get("recovery")
        if isinstance(blocker.get("recovery"), dict)
        else {}
    )
    apply_failure = (
        outcome.get("apply_failure")
        if isinstance(outcome.get("apply_failure"), dict)
        else {}
    )
    apply_summary = (
        apply_failure.get("apply_summary")
        if isinstance(apply_failure.get("apply_summary"), dict)
        else {}
    )
    compact_apply_failure: dict[str, Any] = {}
    if apply_failure:
        compact_apply_failure = {
            key: apply_failure.get(key)
            for key in (
                "kind",
                "step",
                "action_fingerprint",
                "evidence_state_fingerprint",
                "source_evidence_fingerprint",
            )
            if apply_failure.get(key) not in (None, "", [], {})
        }
        if apply_summary:
            compact_apply_failure["apply_summary"] = {
                "status": apply_summary.get("status"),
                "applied_count": apply_summary.get("applied_count"),
                "incomplete_count": apply_summary.get("incomplete_count"),
                "needs_escalation_count": apply_summary.get(
                    "needs_escalation_count"
                ),
                "routes": list(apply_summary.get("routes") or []),
                "results": [
                    _compact_routed_apply_failure(result)
                    for result in apply_summary.get("results") or []
                    if isinstance(result, dict)
                ],
            }
    compact = {
        "source": "previous_agentic_unit_deferred",
        "cycle": cycle_hint_from_path(manifest_path),
        "work_unit_id": work_unit_id,
        "plan_unit_id": outcome.get("plan_unit_id"),
        "status": outcome.get("status"),
        "agent_step": outcome.get("step"),
        "blocker_reason": outcome.get("reason"),
        "blocker_kind": blocker.get("kind"),
        "blocked_apply_kind": (
            recovery.get("previous_blocker_kind")
            or redundancy.get("previous_blocker_kind")
        ),
        "recovery_kind": recovery.get("recovery_kind"),
        "recovery_status": recovery.get("status"),
        "action_fingerprint": (
            redundancy.get("action_fingerprint")
            or blocker.get("action_fingerprint")
        ),
        "evidence_state_fingerprint": (
            redundancy.get("evidence_state_fingerprint")
            or blocker.get("evidence_state_fingerprint")
        ),
        "validation_relation": "work_unit_remains_open_without_a_validated_patch",
        "source_evidence_fingerprint": (
            blocker.get("source_evidence_fingerprint")
            or agentic_source_evidence_fingerprint(plan_unit)
        ),
        "apply_failure": compact_apply_failure,
        "diagnostic_snapshot": _diagnostic_snapshot(plan_unit),
        "repair_evidence_snapshot": _repair_evidence_snapshot(
            plan_unit
        ),
        "note": (
            "This graph-root unit was deferred without a validation-worthy edit. "
            "Do not repeat the same blocked action unless source, diagnostics, "
            "inspection evidence, or graph scope changed."
        ),
    }
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _diagnostic_snapshot(plan_unit: dict[str, Any]) -> dict[str, Any]:
    issues = [
        issue
        for issue in plan_unit.get("source_issue_samples") or plan_unit.get("source_issues") or []
        if isinstance(issue, dict)
    ]
    signatures: list[str] = []
    for issue in issues:
        signature = _validation_issue_signature(issue)
        if signature and signature not in signatures:
            signatures.append(signature)
    root = dict(
        plan_unit.get("code_graph_root_cause_cluster")
        or plan_unit.get("root_cause_cluster")
        or {}
    )
    return {
        "root_node_id": root.get("node_id") or root.get("entity_id"),
        "root_key": graph_root_binding_key(root) or None,
        "signature_count": len(signatures),
        "signatures": signatures[:24],
    }


def _current_writable_source_fingerprint(
    plan_unit: dict[str, Any],
    *,
    spec: CaseSpec | None,
) -> dict[str, Any]:
    paths = sorted(
        {
            normalize_relpath(str(item or ""))
            for item in [
                *list(plan_unit.get("allowed_fileset") or []),
                *list(
                    (plan_unit.get("repair_decision") or {}).get(
                        "candidate_write_set"
                    )
                    or []
                ),
            ]
            if normalize_relpath(str(item or ""))
        }
    )
    records: list[dict[str, Any]] = []
    if spec is not None:
        for relative_path in paths:
            path = spec.resolve_old(relative_path)
            records.append(
                {
                    "file": relative_path,
                    "exists": path.is_file(),
                    "sha256": (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        if path.is_file()
                        else None
                    ),
                }
            )
    return {
        "files": records,
        "fingerprint": (
            stable_id(
                "repair-writable-source-preimage",
                json.dumps(records, ensure_ascii=False, sort_keys=True),
            )
            if records
            else agentic_source_evidence_fingerprint(plan_unit)
        ),
        "authority": (
            "current_worktree_bytes"
            if records
            else "materialized_repair_source_pack"
        ),
    }


def _repair_failure_feedback_classes(
    plan_unit: dict[str, Any],
) -> list[str]:
    classes: set[str] = set()
    for attempt in plan_unit.get("prior_unresolved_patch_attempts") or []:
        if not isinstance(attempt, dict):
            continue
        for field in (
            "status",
            "blocker_kind",
            "blocked_apply_kind",
            "recovery_kind",
            "recovery_status",
            "validation_relation",
        ):
            value = str(attempt.get(field) or "")
            if value:
                classes.add(f"{field}:{value}")
        failure = (
            attempt.get("apply_failure")
            if isinstance(attempt.get("apply_failure"), dict)
            else {}
        )
        for field in ("status", "route", "reason", "invalid_route"):
            value = failure.get(field)
            if isinstance(value, dict):
                value = value.get("route") or value.get("kind")
            text = str(value or "")
            if text:
                classes.add(f"apply_failure.{field}:{text}")
    return sorted(classes)


def _repair_evidence_snapshot(
    plan_unit: dict[str, Any],
    *,
    spec: CaseSpec | None = None,
) -> dict[str, Any]:
    """Capture the current root, source preimage, and typed failure feedback."""

    decision = dict(plan_unit.get("repair_decision") or {})
    diagnostic = _diagnostic_snapshot(plan_unit)
    return {
        "root_key": decision.get("root_key") or diagnostic.get("root_key"),
        "diagnostic_signatures": sorted(
            str(item)
            for item in diagnostic.get("signatures") or []
            if str(item)
        ),
        "repair_mode": str(decision.get("mode") or ""),
        "root_mapping_complete": bool(
            decision.get("root_mapping_complete")
        ),
        "graph_complete": bool(decision.get("graph_complete")),
        "scope_complete": bool(decision.get("scope_complete")),
        "permission_blocked": bool(decision.get("permission_blocked")),
        "candidate_write_set": sorted(
            str(item)
            for item in decision.get("candidate_write_set") or []
            if str(item)
        ),
        "depends_on_work_unit_ids": sorted(
            str(item)
            for item in plan_unit.get("depends_on_work_unit_ids") or []
            if str(item)
        ),
        "incident_patch_ids": sorted(
            str(item)
            for item in plan_unit.get("incident_patch_ids") or []
            if str(item)
        ),
        "source_evidence_fingerprint": (
            agentic_source_evidence_fingerprint(plan_unit)
        ),
        "writable_source_preimage": _current_writable_source_fingerprint(
            plan_unit,
            spec=spec,
        ),
        "failure_feedback_classes": _repair_failure_feedback_classes(
            plan_unit
        ),
    }


def _repair_evidence_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Project fields that remain comparable across persisted controller versions."""

    writable = (
        snapshot.get("writable_source_preimage")
        if isinstance(snapshot.get("writable_source_preimage"), dict)
        else {}
    )
    source_fingerprint = str(
        snapshot.get("source_evidence_fingerprint") or ""
    )
    if not source_fingerprint and writable.get("authority") == (
        "materialized_repair_source_pack"
    ):
        source_fingerprint = str(writable.get("fingerprint") or "")
    return {
        key: copy.deepcopy(snapshot.get(key))
        for key in (
            "root_key",
            "diagnostic_signatures",
            "repair_mode",
            "root_mapping_complete",
            "graph_complete",
            "scope_complete",
            "permission_blocked",
            "candidate_write_set",
            "depends_on_work_unit_ids",
            "incident_patch_ids",
            "failure_feedback_classes",
        )
    } | {"source_evidence_fingerprint": source_fingerprint}


def _diagnostic_signatures_from_snapshot(snapshot: dict[str, Any]) -> set[str]:
    return {
        str(item)
        for item in snapshot.get("signatures") or []
        if str(item)
    }


def _work_unit_diagnostic_signatures(work_unit: dict[str, Any]) -> set[str]:
    return _diagnostic_signatures_from_snapshot(_diagnostic_snapshot(work_unit))


_TYPED_CONTRACT_PREFIXES = {
    "enum_constant",
    "export",
    "field",
    "function",
    "macro",
    "type",
    "variable",
}


def _typed_contract_keys(value: Any) -> set[str]:
    """Collect explicit semantic contract keys, never free-form symbols."""

    keys: set[str] = set()
    if isinstance(value, dict):
        for field, child in value.items():
            if field in {"contract_key", "removed_contract_key"}:
                candidate = str(child or "").strip()
                if (
                    ":" in candidate
                    and candidate.split(":", 1)[0]
                    in _TYPED_CONTRACT_PREFIXES
                ):
                    keys.add(candidate)
            keys.update(_typed_contract_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_typed_contract_keys(child))
    return keys


def _initial_unit_primary_contract_keys(
    plan_unit: dict[str, Any],
) -> set[str]:
    """Return provider contracts owned by an initial difference unit."""

    keys: set[str] = set()
    provider_fields = (
        "old_entity",
        "donor_entity",
        "old_declarations",
        "donor_declarations",
        "old_exports",
        "donor_exports",
    )
    for evidence in plan_unit.get("contract_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        for field in provider_fields:
            keys.update(_typed_contract_keys(evidence.get(field)))
    return keys


def _work_unit_root_contract_keys(
    work_unit: dict[str, Any],
) -> set[str]:
    """Return only contracts in the unit's selected responsible root set."""

    code_graph_root = work_unit.get("code_graph_root_cause_cluster")
    root = (
        code_graph_root
        if isinstance(code_graph_root, dict) and code_graph_root
        else work_unit.get("root_cause_cluster")
    )
    keys = _typed_contract_keys(root)
    for alternative in work_unit.get("root_cause_alternatives") or []:
        if isinstance(alternative, dict):
            keys.update(_typed_contract_keys(alternative))
    return keys


def _localized_initial_patch_lines(
    initial_dir: Path,
    result: dict[str, Any],
) -> list[str]:
    lines = _patch_lines_from_apply_result(result)
    if lines:
        return lines
    patch_name = Path(str(result.get("patch_path") or "")).name
    candidates = [
        initial_dir / "responses" / "patches" / patch_name,
        initial_dir / "patches" / patch_name,
    ]
    for candidate in candidates:
        if not patch_name or not candidate.is_file():
            continue
        try:
            return candidate.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except OSError:
            continue
    return []


def _initial_contract_attempts(
    cycle_dirs: Iterable[Path],
    *,
    state_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Index initial-stage patch lineage by exact typed provider contract."""

    attempts_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, str, str]] = set()
    for cycle_dir in cycle_dirs:
        initial_dir = cycle_dir / "initial_compatibility"
        attempts_path = initial_dir / "llm_attempts.json"
        if not attempts_path.is_file():
            continue
        try:
            payload = json.loads(attempts_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for attempt in payload.get("attempts") or []:
            if not isinstance(attempt, dict):
                continue
            unit_id = str(attempt.get("unit_id") or "")
            plan_path = initial_dir / f"{unit_id}.plan.json"
            try:
                plan = load_repair_plan(
                    plan_path,
                    state_root=state_root,
                )
            except (OSError, ValueError):
                continue
            plan_units = [
                item
                for item in plan.get("work_unit_plans") or []
                if isinstance(item, dict)
                and str(item.get("id") or "") == unit_id
            ]
            if len(plan_units) != 1:
                continue
            contract_keys = _initial_unit_primary_contract_keys(plan_units[0])
            if not contract_keys:
                continue
            for result in attempt.get("apply_results") or []:
                if not isinstance(result, dict):
                    continue
                status = str(result.get("status") or "")
                patch_lines = _localized_initial_patch_lines(
                    initial_dir,
                    result,
                )
                patch_signature = str(result.get("patch_signature") or "")
                identity = (
                    unit_id,
                    int(attempt.get("attempt_number") or 0),
                    status,
                    patch_signature,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                compact = {
                    "source": "initial_compatibility_contract_attempt",
                    "cycle": cycle_hint_from_path(cycle_dir),
                    "initial_unit_id": unit_id,
                    "attempt_number": attempt.get("attempt_number"),
                    "status": status,
                    "contract_keys": sorted(contract_keys),
                    "touched_files": list(result.get("touched_files") or []),
                    "patch_path": result.get("patch_path"),
                    "patch_signature": patch_signature or None,
                    "transaction_dir": result.get("transaction_dir"),
                    "error": result.get("error"),
                    "validation_failure": result.get("validation_failure"),
                    "patch_rebase": result.get("patch_rebase"),
                    "validation_relation": (
                        "typed provider contract remains open after the "
                        "initial compatibility attempt"
                    ),
                    "note": (
                        "Use this as contract-level lineage. Re-evaluate it "
                        "against the current tree; never replay the old hunk "
                        "without current static proof."
                    ),
                }
                if patch_lines:
                    compact["patch_excerpt_lines"] = patch_lines[:80]
                    compact["patch_delta_summary"] = patch_delta_summary(
                        patch_lines
                    )
                    compact["changed_ranges"] = (
                        unified_diff_changed_ranges(
                            "\n".join(patch_lines) + "\n",
                            include_content=True,
                        )
                    )
                compact = {
                    key: value
                    for key, value in compact.items()
                    if value not in (None, "", [], {})
                }
                for contract_key in contract_keys:
                    attempts_by_contract[contract_key].append(compact)
    return dict(attempts_by_contract)


def _work_unit_diagnostic_loci(
    work_unit: dict[str, Any],
    *,
    old_root: Path | None = None,
) -> list[dict[str, Any]]:
    loci: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for issue in (
        work_unit.get("source_issue_samples")
        or work_unit.get("source_issues")
        or []
    ):
        if not isinstance(issue, dict):
            continue
        signature = _validation_issue_signature(issue)
        locations = [
            issue,
            *[
                note
                for note in issue.get("related_notes") or []
                if isinstance(note, dict)
            ],
        ]
        for location in locations:
            raw_path = str(location.get("file") or "")
            relative_path = (
                normalize_root_relative(raw_path, root=old_root)
                if old_root is not None
                else normalize_relpath(raw_path)
            )
            line = int(location.get("line") or 0)
            identity = (relative_path, line, signature)
            if (
                not relative_path
                or line <= 0
                or identity in seen
            ):
                continue
            seen.add(identity)
            loci.append(
                {
                    "file": relative_path,
                    "line": line,
                    "diagnostic_signature": signature or None,
                }
            )
    return loci


def _initial_patch_locus_attempts(
    work_unit: dict[str, Any],
    initial_attempts: Iterable[dict[str, Any]],
    *,
    old_root: Path,
) -> list[dict[str, Any]]:
    """Bind first-compile diagnostics to exact surviving initial hunks."""

    matches: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int, str]] = set()
    for locus in _work_unit_diagnostic_loci(
        work_unit,
        old_root=old_root,
    ):
        relative_path = str(locus["file"])
        line = int(locus["line"])
        for attempt in initial_attempts:
            if str(attempt.get("status") or "") not in {
                "applied",
                "already_applied",
                "already_applied_duplicate",
            }:
                continue
            for changed in attempt.get("changed_ranges") or []:
                if not isinstance(changed, dict):
                    continue
                changed_file = normalize_relpath(
                    str(changed.get("file") or "")
                )
                new_count = int(changed.get("new_count") or 0)
                start_line = int(changed.get("new_start") or 0)
                end_line = int(changed.get("new_end") or start_line)
                expected_lines = [
                    str(value)
                    for value in changed.get("new_lines") or []
                ]
                if (
                    changed_file != relative_path
                    or new_count <= 0
                    or len(expected_lines) != new_count
                    or not (start_line <= line <= end_line)
                ):
                    continue
                target = old_root / changed_file
                if not target.is_file():
                    continue
                try:
                    current_lines = target.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines()
                except OSError:
                    continue
                if current_lines[start_line - 1 : end_line] != expected_lines:
                    continue
                patch_id = str(
                    attempt.get("patch_signature")
                    or Path(
                        str(attempt.get("transaction_dir") or "")
                    ).name
                    or attempt.get("initial_unit_id")
                    or ""
                )
                identity = (
                    str(attempt.get("initial_unit_id") or ""),
                    changed_file,
                    start_line,
                    end_line,
                    str(locus.get("diagnostic_signature") or ""),
                )
                if identity in seen:
                    continue
                seen.add(identity)
                matches.append(
                    {
                        **copy.deepcopy(attempt),
                        "source": "initial_patch_locus_regression",
                        "incident_patch_ids": [patch_id] if patch_id else [],
                        "incident_diagnostic_signature": locus.get(
                            "diagnostic_signature"
                        ),
                        "patch_locus": {
                            **dict(changed),
                            "file": changed_file,
                            "start_line": start_line,
                            "end_line": end_line,
                        },
                        "validation_relation": (
                            "first target compile diagnostic intersects the "
                            "exact postimage of a surviving initial patch"
                        ),
                        "note": (
                            "This is read-only history for the surviving "
                            "initial change. Current demand and graph-root "
                            "evidence determine the repair target."
                        ),
                    }
                )
    return matches


def _retain_current_unresolved_attempt_context(
    attempt: dict[str, Any],
    *,
    current_diagnostic_signatures: set[str],
) -> dict[str, Any] | None:
    compact = dict(attempt)
    if compact.get("source") == "previous_patch_locus_regression":
        incident_signature = str(
            compact.get("incident_diagnostic_signature") or ""
        )
        if (
            incident_signature
            and incident_signature
            not in current_diagnostic_signatures
        ):
            return None
        compact["diagnostic_component_match"] = (
            "exact_patch_locus_successor"
        )
        return compact
    previous_diagnostic_signatures = _diagnostic_signatures_from_snapshot(
        dict(attempt.get("diagnostic_snapshot") or {})
    )
    if not current_diagnostic_signatures or not previous_diagnostic_signatures:
        compact["diagnostic_component_match"] = "unknown"
        compact["note"] = (
            f"{str(compact.get('note') or '').strip()} "
            "This historical attempt lacks diagnostic-node identity and is retained as context only."
        ).strip()
        return compact

    unresolved_overlap = current_diagnostic_signatures & previous_diagnostic_signatures
    if not unresolved_overlap:
        compact["diagnostic_component_match"] = "same_graph_root_changed_diagnostics"
        compact["resolved_diagnostic_signatures_since_attempt"] = sorted(
            previous_diagnostic_signatures
        )[:24]
        compact["new_diagnostic_signatures_since_attempt"] = sorted(
            current_diagnostic_signatures
        )[:24]
        compact.pop("patch_excerpt_lines", None)
        compact["patch_excerpt_omitted_reason"] = (
            "the graph root persisted but its exact compiler diagnostics changed after this attempt"
        )
        compact["note"] = (
            "The same CodeIndex root remains open with a different diagnostic set. Retain the exact "
            "patch delta as history, but do not replay its old hunk body."
        )
        return compact
    resolved_since_attempt = previous_diagnostic_signatures - current_diagnostic_signatures
    new_since_attempt = current_diagnostic_signatures - previous_diagnostic_signatures
    same_diagnostic_set = current_diagnostic_signatures == previous_diagnostic_signatures
    compact["unresolved_diagnostic_signatures"] = sorted(unresolved_overlap)[:24]
    compact["resolved_diagnostic_signatures_since_attempt"] = sorted(resolved_since_attempt)[:24]
    compact["new_diagnostic_signatures_since_attempt"] = sorted(new_since_attempt)[:24]
    compact["diagnostic_component_match"] = "exact" if same_diagnostic_set else "partial"

    if not same_diagnostic_set:
        if resolved_since_attempt:
            compact.pop("patch_excerpt_lines", None)
            compact["patch_excerpt_omitted_reason"] = (
                "the unresolved diagnostic-node set changed after this attempt; omit the old patch body so "
                "already-resolved edits do not pollute the next repair prompt"
            )
            compact["note"] = (
                "This attempt overlaps the current graph component, but some previous diagnostic nodes "
                "were resolved. Treat only the listed residual diagnostic overlap as still open."
            )
        else:
            compact["note"] = (
                "This attempt overlaps the current graph component and added new diagnostic nodes without "
                "resolving prior ones. The exact old and new diagnostic signatures are recorded above."
            )
    return compact


def _active_patch_postimage_correction_authority(
    work_unit: dict[str, Any],
    attempts: Iterable[dict[str, Any]],
    *,
    old_root: Path,
) -> dict[str, Any]:
    """Authorize correction only inside an exact surviving patch postimage."""

    root = dict(
        work_unit.get("code_graph_root_cause_cluster")
        or work_unit.get("root_cause_cluster")
        or {}
    )
    if not (
        bool(root.get("unmapped"))
        or str(root.get("entity_kind") or "") == "UnresolvedContract"
    ):
        return work_unit

    mapped_owner_ids = {
        str(owner_id)
        for mapping in work_unit.get("diagnostic_mappings") or []
        if isinstance(mapping, dict)
        and mapping.get("unmapped") is not True
        for owner_id in mapping.get("diagnostic_owner_node_ids") or []
        if str(owner_id)
    }
    diagnostic_loci = _work_unit_diagnostic_loci(
        work_unit,
        old_root=old_root,
    )
    repair_owners = [
        item
        for item in work_unit.get("repair_owners") or []
        if isinstance(item, dict)
    ]

    verified: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for attempt in attempts:
        if (
            str(attempt.get("source") or "")
            != "previous_patch_locus_regression"
            or str(attempt.get("diagnostic_component_match") or "")
            != "exact_patch_locus_successor"
            or str(attempt.get("status") or "")
            not in {"applied", "already_applied", "already_applied_duplicate"}
        ):
            continue
        locus = dict(attempt.get("patch_locus") or {})
        relative_path = normalize_relpath(str(locus.get("file") or ""))
        start_line = int(locus.get("new_start") or locus.get("start_line") or 0)
        new_count = int(locus.get("new_count") or 0)
        expected_lines = [str(line) for line in locus.get("new_lines") or []]
        end_line = int(
            locus.get("new_end")
            or locus.get("end_line")
            or (start_line + new_count - 1)
        )
        patch_ids = [
            str(patch_id)
            for patch_id in attempt.get("incident_patch_ids") or []
            if str(patch_id)
        ]
        incident_signature = str(
            attempt.get("incident_diagnostic_signature") or ""
        )
        if (
            not relative_path
            or start_line <= 0
            or new_count <= 0
            or end_line != start_line + new_count - 1
            or not patch_ids
            or not incident_signature
        ):
            continue
        matching_loci = [
            item
            for item in diagnostic_loci
            if item.get("file") == relative_path
            and item.get("diagnostic_signature") == incident_signature
        ]
        if len(matching_loci) != 1:
            continue
        diagnostic_line = int(matching_loci[0]["line"])
        if not expected_lines:
            patch_path = Path(str(attempt.get("patch_path") or ""))
            try:
                patch_text = patch_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, ValueError):
                patch_text = ""
            matching_ranges = [
                item
                for item in unified_diff_changed_ranges(
                    patch_text,
                    include_content=True,
                )
                if normalize_relpath(str(item.get("file") or ""))
                == relative_path
                and int(item.get("new_count") or 0) > 0
                and start_line <= int(item.get("new_start") or 0)
                and int(item.get("new_end") or 0) <= end_line
            ]
            if len(matching_ranges) == 1:
                start_line = int(matching_ranges[0]["new_start"])
                new_count = int(matching_ranges[0]["new_count"])
                end_line = int(matching_ranges[0]["new_end"])
                expected_lines = [
                    str(line)
                    for line in matching_ranges[0].get("new_lines") or []
                ]
        if len(expected_lines) != new_count:
            continue
        target = old_root / relative_path
        if not target.is_file():
            continue
        try:
            current_lines = target.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError:
            continue
        surviving_starts = [
            index + 1
            for index in range(len(current_lines) - new_count + 1)
            if current_lines[index : index + new_count] == expected_lines
        ]
        if len(surviving_starts) != 1:
            continue
        start_line = surviving_starts[0]
        end_line = start_line + new_count - 1
        if not (start_line <= diagnostic_line <= end_line):
            continue

        matching_owners: list[dict[str, Any]] = []
        for owner in repair_owners:
            owner_id = str(
                owner.get("owner_entity_id")
                or (owner.get("diagnostic_structure") or {}).get(
                    "owner_node_id"
                )
                or ""
            )
            extent = dict(owner.get("owner_entity_extent") or {})
            owner_start = int(extent.get("start_line") or 0)
            owner_end = int(extent.get("end_line") or owner_start)
            if (
                str(owner.get("owner_kind") or "")
                == "code_index_diagnostic_ast_owner"
                and normalize_relpath(str(owner.get("owner_file") or ""))
                == relative_path
                and owner.get("old_exists") is True
                and owner_id
                and owner_id in mapped_owner_ids
                and owner_start <= diagnostic_line <= owner_end
            ):
                matching_owners.append(owner)
        if len(matching_owners) != 1:
            continue
        verified.append(
            (
                attempt,
                {
                    **locus,
                    "new_start": start_line,
                    "new_end": end_line,
                    "new_count": new_count,
                    "new_lines": expected_lines,
                },
                matching_owners[0],
            )
        )

    if len(verified) != 1:
        return work_unit

    attempt, locus, owner = verified[0]
    relative_path = normalize_relpath(str(locus.get("file") or ""))
    start_line = int(locus.get("new_start") or locus.get("start_line") or 0)
    new_count = int(locus.get("new_count") or 0)
    end_line = start_line + new_count - 1
    patch_ids = sorted(
        {
            str(patch_id)
            for patch_id in attempt.get("incident_patch_ids") or []
            if str(patch_id)
        }
    )
    owner_id = str(
        owner.get("owner_entity_id")
        or (owner.get("diagnostic_structure") or {}).get("owner_node_id")
        or ""
    )
    authority_id = stable_id(
        "active-patch-postimage-authority",
        relative_path,
        str(start_line),
        str(end_line),
        *patch_ids,
        str(attempt.get("incident_diagnostic_signature") or ""),
        owner_id,
    )
    authority = {
        "id": authority_id,
        "schema": "active_patch_transaction_postimage_authority_v1",
        "status": "verified",
        "authority": "active_patch_transaction_postimage",
        "evidence_authority": (
            "fresh_diagnostic_exact_retained_patch_postimage"
        ),
        "file": relative_path,
        "start_line": start_line,
        "end_line": end_line,
        "patch_ids": patch_ids,
        "diagnostic_signature": attempt.get(
            "incident_diagnostic_signature"
        ),
        "diagnostic_owner_node_id": owner_id,
        "causal_relation": "fresh_diagnostic_in_retained_patch_postimage",
    }
    updated_owners: list[dict[str, Any]] = []
    for candidate in repair_owners:
        candidate_id = str(
            candidate.get("owner_entity_id")
            or (candidate.get("diagnostic_structure") or {}).get(
                "owner_node_id"
            )
            or ""
        )
        if candidate_id != owner_id:
            updated_owners.append(candidate)
            continue
        reason = str(candidate.get("reason") or "").strip()
        authority_reason = (
            "fresh diagnostic is inside the exact surviving postimage of "
            "an applied patch transaction"
        )
        updated_owners.append(
            {
                **candidate,
                "owner_role": "patch_transaction_postimage_owner",
                "allowed_as_patch_target": True,
                "allowed_patch_files": [relative_path],
                "read_only_evidence_files": [],
                "reason": (
                    reason
                    if authority_reason in reason
                    else f"{reason}; {authority_reason}".strip("; ")
                ),
            }
        )
    return {
        **work_unit,
        "incident_patch_ids": sorted(
            {
                *[
                    str(item)
                    for item in work_unit.get("incident_patch_ids") or []
                    if str(item)
                ],
                *patch_ids,
            }
        ),
        "repair_transaction_authorities": [
            *[
                item
                for item in work_unit.get("repair_transaction_authorities")
                or []
                if isinstance(item, dict)
                and str(item.get("id") or "") != authority_id
            ],
            authority,
        ],
        "repair_owners": updated_owners,
    }


def _attach_previous_unresolved_patch_attempts(
    work_units: list[dict[str, Any]],
    *,
    spec: CaseSpec,
    previous_cycle_dir: Path | None,
    state_root: Path,
    current_validation_delta: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if previous_cycle_dir is None or not previous_cycle_dir.exists():
        return work_units, 0
    attempts_by_work_unit: dict[str, list[dict[str, Any]]] = {}

    def attach_patch_locus_regressions(
        validation_delta: dict[str, Any],
    ) -> None:
        for target in validation_delta.get("target_results") or []:
            if not isinstance(target, dict):
                continue
            origin_work_unit_id = str(
                target.get("target_work_unit_id") or ""
            )
            origin_attempts = list(
                attempts_by_work_unit.get(origin_work_unit_id) or []
            )
            if not origin_attempts:
                continue
            for regression in target.get("patch_locus_regressions") or []:
                if not isinstance(regression, dict):
                    continue
                rebound_work_unit_id = str(
                    regression.get("work_unit_id") or ""
                )
                patch_locus = (
                    dict(regression.get("patch_locus") or {})
                    if isinstance(regression.get("patch_locus"), dict)
                    else {}
                )
                relative_path = normalize_relpath(
                    str(
                        patch_locus.get("file")
                        or regression.get("file")
                        or ""
                    )
                )
                if not rebound_work_unit_id or not relative_path:
                    continue
                incident = next(
                    (
                        attempt
                        for attempt in reversed(origin_attempts)
                        if relative_path
                        in {
                            normalize_relpath(str(path))
                            for path in attempt.get("touched_files") or []
                        }
                    ),
                    None,
                )
                if incident is None:
                    continue
                incident_patch_id = str(
                    incident.get("patch_outcome_candidate_id")
                    or incident.get("patch_signature")
                    or Path(str(incident.get("transaction_dir") or "")).name
                )
                attempts_by_work_unit.setdefault(
                    rebound_work_unit_id, []
                ).append(
                    {
                        **copy.deepcopy(incident),
                        "source": "previous_patch_locus_regression",
                        "origin_work_unit_id": origin_work_unit_id,
                        "work_unit_id": rebound_work_unit_id,
                        "incident_patch_ids": (
                            [incident_patch_id] if incident_patch_id else []
                        ),
                        "incident_diagnostic_signature": str(
                            regression.get("diagnostic_signature") or ""
                        ),
                        "patch_locus": {
                            **patch_locus,
                            "file": relative_path,
                        },
                        "validation_relation": (
                            "new diagnostic intersects the exact retained "
                            "patch postimage"
                        ),
                        "note": (
                            "This is read-only history linking the fresh "
                            "diagnostic to a retained change. Current graph "
                            "roots remain the only repair authority."
                        ),
                    }
                )

    current_cycle = cycle_hint_from_path(previous_cycle_dir)
    cycle_dirs = [
        path
        for path in sorted(previous_cycle_dir.parent.glob("cycle-*"))
        if path.is_dir()
        and (not current_cycle or 0 < cycle_hint_from_path(path) <= current_cycle)
    ]
    if previous_cycle_dir not in cycle_dirs:
        cycle_dirs.append(previous_cycle_dir)

    for cycle_dir in sorted(cycle_dirs, key=cycle_hint_from_path):
        for apply_path in sorted(cycle_dir.glob("**/apply_results.json")):
            try:
                apply_summary = json.loads(apply_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            plan_path = _local_plan_path_for_apply_summary(apply_path, apply_summary)
            if plan_path is None:
                continue
            try:
                plan = load_repair_plan(
                    plan_path,
                    state_root=state_root,
                )
            except (OSError, ValueError):
                continue
            plan_units = repair_plan_units(plan)
            plan_units_by_id = {
                str(item.get("id") or ""): dict(item)
                for item in plan_units
                if str(item.get("id") or "")
            }
            for result in apply_summary.get("results") or []:
                if not isinstance(result, dict):
                    continue
                if str(result.get("status") or "") not in {
                    "applied",
                    "already_applied",
                    "already_applied_duplicate",
                }:
                    continue
                plan_unit_id = str(result.get("unit_id") or "")
                plan_unit = plan_units_by_id.get(plan_unit_id) or {}
                work_unit_id = str(plan_unit.get("work_unit_id") or plan_unit_id)
                if not work_unit_id:
                    continue
                attempts_by_work_unit.setdefault(work_unit_id, []).append(
                    _compact_previous_unresolved_attempt(
                        apply_path=apply_path,
                        result=result,
                        work_unit_id=work_unit_id,
                        plan_unit=plan_unit,
                    )
                )
        for manifest_path in sorted(cycle_dir.glob("**/round_manifest.json")):
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                continue
            source_plan_path = Path(
                str(manifest.get("source_repair_plan_path") or "")
            )
            if not source_plan_path.is_file():
                continue
            try:
                source_plan = load_repair_plan(
                    source_plan_path,
                    state_root=state_root,
                )
            except (OSError, ValueError):
                continue
            plan_units_by_id = {
                str(item.get("id") or ""): dict(item)
                for item in source_plan.get("work_unit_plans") or []
                if isinstance(item, dict) and item.get("id")
            }
            validation_delta = (
                manifest.get("validation_delta")
                if isinstance(
                    manifest.get("validation_delta"), dict
                )
                else {}
            )
            attach_patch_locus_regressions(validation_delta)
            for outcome in round_manifest_collections(
                manifest_path,
                manifest,
                state_root=state_root,
                fields=["unit_outcomes"],
            )["unit_outcomes"]:
                if not isinstance(outcome, dict):
                    continue
                if str(outcome.get("status") or "") != "deferred":
                    continue
                plan_unit_id = str(outcome.get("plan_unit_id") or "")
                plan_unit = plan_units_by_id.get(plan_unit_id) or {}
                work_unit_id = str(
                    plan_unit.get("work_unit_id") or plan_unit_id
                )
                if not work_unit_id:
                    continue
                attempts_by_work_unit.setdefault(work_unit_id, []).append(
                    _compact_previous_deferred_unit_attempt(
                        manifest_path=manifest_path,
                        outcome=outcome,
                        work_unit_id=work_unit_id,
                        plan_unit=plan_unit,
                    )
                )

    if isinstance(current_validation_delta, dict):
        attach_patch_locus_regressions(current_validation_delta)

    attempts_by_contract = _initial_contract_attempts(
        cycle_dirs,
        state_root=state_root,
    )
    if (
        not attempts_by_work_unit
        and not attempts_by_contract
    ):
        return work_units, 0
    initial_attempt_catalog: list[dict[str, Any]] = []
    seen_initial_catalog_entries: set[tuple[str, int, str, str]] = set()
    for contract_attempts in attempts_by_contract.values():
        for attempt in contract_attempts:
            identity = (
                str(attempt.get("initial_unit_id") or ""),
                int(attempt.get("attempt_number") or 0),
                str(attempt.get("status") or ""),
                str(attempt.get("patch_signature") or ""),
            )
            if identity in seen_initial_catalog_entries:
                continue
            seen_initial_catalog_entries.add(identity)
            initial_attempt_catalog.append(attempt)

    updated: list[dict[str, Any]] = []
    attached_count = 0
    for work_unit in work_units:
        work_unit_id = str(work_unit.get("id") or "")
        current_diagnostic_signatures = _work_unit_diagnostic_signatures(work_unit)
        attempts: list[dict[str, Any]] = []
        for attempt in attempts_by_work_unit.get(work_unit_id) or []:
            retained = _retain_current_unresolved_attempt_context(
                attempt,
                current_diagnostic_signatures=current_diagnostic_signatures,
            )
            if retained is not None:
                attempts.append(retained)
        current_contract_keys = _work_unit_root_contract_keys(work_unit)
        initial_attempts: list[dict[str, Any]] = []
        seen_initial_attempts: set[
            tuple[str, int, str, str]
        ] = set()
        for contract_key in sorted(current_contract_keys):
            for attempt in attempts_by_contract.get(contract_key) or []:
                identity = (
                    str(attempt.get("initial_unit_id") or ""),
                    int(attempt.get("attempt_number") or 0),
                    str(attempt.get("status") or ""),
                    str(attempt.get("patch_signature") or ""),
                )
                if identity in seen_initial_attempts:
                    continue
                seen_initial_attempts.add(identity)
                initial_attempts.append(
                    {
                        **attempt,
                        "matched_contract_keys": sorted(
                            current_contract_keys
                            & set(attempt.get("contract_keys") or [])
                        ),
                    }
                )
        initial_locus_attempts = _initial_patch_locus_attempts(
            work_unit,
            initial_attempt_catalog,
            old_root=spec.old_root_path,
        )
        if initial_locus_attempts:
            attempts.extend(initial_locus_attempts)
        if attempts:
            work_unit = dict(work_unit)
            work_unit["prior_unresolved_patch_attempts"] = [
                *list(work_unit.get("prior_unresolved_patch_attempts") or []),
                *attempts[-2:],
            ]
            work_unit = _active_patch_postimage_correction_authority(
                work_unit,
                attempts,
                old_root=spec.old_root_path,
            )
            attached_count += min(len(attempts), 2)
        if initial_attempts:
            work_unit = dict(work_unit)
            work_unit["prior_unresolved_patch_attempts"] = [
                *list(work_unit.get("prior_unresolved_patch_attempts") or []),
                initial_attempts[-1],
            ]
            attached_count += 1
        updated.append(work_unit)
    return updated, attached_count


def work_unit_registry_dir_for(state_root: Path) -> Path:
    return ensure_dir(state_root / "work_units")


def _merge_work_unit_context(existing: dict | None, fresh: dict) -> dict:
    if not existing:
        return dict(fresh)

    updated = dict(existing)
    for key in (
        "title",
        "scope",
        "decision",
        "validation",
        "source_issues",
        "repair_owners",
        "provider_resolutions",
        "diagnostic_signature_count",
        "diagnostic_signatures",
        "diagnostic_routes",
        "diagnostic_graph",
        "diagnostic_mappings",
        "diagnostic_partition_reason",
        "overlapping_root_node_ids",
        "root_cause_cluster",
        "root_cause_alternatives",
        "code_graph_root_cause_cluster",
        "depends_on_work_unit_ids",
        "dependency_root_node_ids",
        "localization",
        "patch_targets",
        "patch_target",
        "prior_unresolved_patch_attempts",
        "unit_kind",
        "execution_phase",
        "component_work_unit_ids",
        "semantic_gap_ids",
        "required_semantic_gap_ids",
        "pending_semantic_gap_ids",
        "blocked_semantic_gap_ids",
        "semantic_transaction_ids",
        "incident_patch_ids",
        "semantic_transactions",
        "integration_obligations",
        "required_integration_obligation_ids",
        "integration_coverage_blockers",
        "semantic_recovery_context",
        "semantic_contract_binding",
        "difference_atoms",
        "difference_atom_ids",
        "contract_evidence",
        "contract_evidence_ids",
        "demand_anchors",
        "demand_anchor_selection",
        "repair_loci",
    ):
        updated[key] = fresh.get(key)
    updated["last_seen_at"] = utc_now()
    return updated


def _work_unit_diagnostic_files(work_unit: dict[str, Any]) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for issue in work_unit.get("source_issues") or []:
        if not isinstance(issue, dict):
            continue
        value = str(
            issue.get("file")
            or issue.get("source_file")
            or issue.get("path")
            or ""
        ).strip()
        if value and value not in seen:
            seen.add(value)
            files.append(value)
    for node in (work_unit.get("diagnostic_graph") or {}).get("nodes") or []:
        if not isinstance(node, dict) or str(node.get("kind") or "") != "Diagnostic":
            continue
        value = str(node.get("file") or node.get("path") or "").strip()
        if value and value not in seen:
            seen.add(value)
            files.append(value)
    return files


def _reconcile_work_units_after_validation(
    *,
    spec: CaseSpec,
    state_root: Path,
    previous_open_ids: list[str],
    fresh_work_units: list[dict],
    validation: dict[str, Any],
    validation_ok: bool,
) -> tuple[list[dict], list[str], list[str]]:
    registry_dir = work_unit_registry_dir_for(state_root)
    registry = load_work_units(registry_dir)
    fresh_by_id = {str(item["id"]): dict(item) for item in fresh_work_units}
    closed_ids: list[str] = []
    deferred_ids: list[str] = []
    compile_coverage = merged_validation_translation_unit_coverage(
        validation
    )

    def exact_successors(existing: dict[str, Any]) -> list[dict[str, Any]]:
        def diagnostic_identities(unit: dict[str, Any]) -> set[str]:
            return {
                str(item)
                for item in unit.get("diagnostic_signatures") or []
                if str(item)
            } | _work_unit_diagnostic_signatures(unit)

        existing_gap_ids = {
            str(item)
            for item in existing.get("semantic_gap_ids") or []
            if str(item)
        }
        existing_signatures = diagnostic_identities(existing)
        candidates: list[dict[str, Any]] = []
        for fresh in fresh_work_units:
            fresh_id = str(fresh.get("id") or "")
            if fresh_id == str(existing.get("id") or ""):
                continue
            fresh_gap_ids = {
                str(item)
                for item in fresh.get("semantic_gap_ids") or []
                if str(item)
            }
            fresh_signatures = diagnostic_identities(fresh)
            if (
                existing_gap_ids
                and fresh_gap_ids
                and existing_gap_ids & fresh_gap_ids
            ) or (
                existing_signatures
                and fresh_signatures
                and existing_signatures & fresh_signatures
            ):
                candidates.append(fresh)
        return candidates

    for work_unit_id in previous_open_ids:
        existing = registry.get(work_unit_id)
        if existing is None:
            continue
        successors = exact_successors(existing)
        if successors:
            successor_ids = sorted(
                str(item.get("id") or "") for item in successors
            )
            updated = mark_work_unit_abandoned(
                existing,
                reason=(
                    "superseded by exact semantic-gap or diagnostic identities: "
                    + ", ".join(successor_ids)
                ),
            )
            closed_ids.append(work_unit_id)
        elif work_unit_id in fresh_by_id:
            updated = mark_work_unit_failed(
                existing,
                reason=(
                    "validation passed but persistent semantic obligations "
                    "still require an explicit disposition"
                    if validation_ok
                    else "validation still reports this work unit"
                ),
            )
        elif validation_ok:
            updated = mark_work_unit_succeeded(
                existing,
                reason="validation passed and this work unit is resolved",
            )
            closed_ids.append(work_unit_id)
        else:
            resolution_coverage = diagnostic_files_resolution_coverage(
                _work_unit_diagnostic_files(existing),
                coverage=compile_coverage,
                old_root=spec.old_root_path,
                preferred_prefix=spec.driver_root,
            )
            if resolution_coverage.get("covered"):
                updated = mark_work_unit_succeeded(
                    existing,
                    reason=(
                        "this work unit disappeared after every diagnostic "
                        "translation unit produced a current object"
                    ),
                )
                closed_ids.append(work_unit_id)
            else:
                updated = mark_work_unit_validation_deferred(
                    existing,
                    reason=(
                        "diagnostic absence is not resolution evidence because "
                        "its translation unit was not successfully covered"
                    ),
                    coverage=resolution_coverage,
                )
                deferred_ids.append(work_unit_id)
        registry[work_unit_id] = updated
        write_work_unit(updated, work_units_dir=registry_dir)

    current_units: list[dict] = []
    for fresh in fresh_work_units:
        work_unit_id = str(fresh["id"])
        merged = _merge_work_unit_context(registry.get(work_unit_id), fresh)
        planned = promote_work_unit_to_planning(
            merged,
            reason="repair plan generated from current validation cycle",
        )
        current_units.append(planned)

    return current_units, closed_ids, deferred_ids


def _update_repair_work_units_for_apply_results(
    *,
    state_root: Path,
    plan: dict,
    results: list[dict],
    no_patch_means_escalation: bool = False,
) -> None:
    registry_dir = work_unit_registry_dir_for(state_root)
    registry = load_work_units(registry_dir)
    plan_map = {str(item["id"]): str(item.get("work_unit_id") or "") for item in plan.get("work_unit_plans") or []}

    for item in results:
        work_unit_id = plan_map.get(str(item.get("unit_id") or ""))
        if not work_unit_id:
            continue
        work_unit = registry.get(work_unit_id)
        if work_unit is None:
            continue

        result_status = str(item.get("status") or "")
        if result_status in {
            "applied",
            "already_applied",
            "already_applied_duplicate",
        }:
            updated = promote_work_unit_to_validating(
                work_unit,
                reason="repair execution completed and is awaiting validation",
            )
        elif result_status == "no_patch":
            if no_patch_means_escalation:
                updated = mark_work_unit_escalated(
                    work_unit,
                    reason="repair response produced no patch for a validation-derived work unit",
                )
            else:
                updated = promote_work_unit_to_validating(
                    work_unit,
                    reason="repair execution completed and is awaiting validation",
                )
        elif result_status == "needs_escalation":
            updated = mark_work_unit_escalated(
                work_unit,
                reason="repair result explicitly requested escalation review",
            )
        else:
            updated = mark_work_unit_failed(
                work_unit,
                reason=f"repair apply status is {result_status}",
            )
        write_work_unit(updated, work_units_dir=registry_dir)


def _repair_plan_summary_counts(repair_plan: dict) -> dict[str, int]:
    work_unit_plans = list(repair_plan.get("work_unit_plans") or [])
    summary = summarize_repair_plan_units(work_unit_plans)
    return {
        "work_unit_count": int(summary.get("work_unit_count") or 0),
        "patchable_work_unit_count": int(summary.get("patchable_work_unit_count") or 0),
        "escalation_only_work_unit_count": int(summary.get("escalation_only_work_unit_count") or 0),
        "scope_expansion_work_unit_count": int(
            summary.get("scope_expansion_work_unit_count") or 0
        ),
        "blocked_work_unit_count": int(summary.get("blocked_work_unit_count") or 0),
    }


def _repair_status_from_plan(
    *,
    validation_ok: bool,
    repair_plan: dict,
) -> str:
    if validation_ok:
        return "ready_for_manual_test"
    counts = _repair_plan_summary_counts(repair_plan)
    if counts["patchable_work_unit_count"] > 0 or counts["work_unit_count"] > 0:
        return "needs_patching"
    return "patch_apply_incomplete"


def load_or_init_state(spec: CaseSpec, state_root: Path) -> CycleState:
    state_file = state_file_for(state_root)
    if state_file.exists():
        return CycleState.from_file(state_file)
    state = CycleState.initial(spec.case_id)
    state.save(state_file)
    return state


def run_cycle(spec: CaseSpec, *, state_root: Path) -> dict:
    ensure_dir(state_root)
    state = load_or_init_state(spec, state_root)
    next_cycle_dir = ensure_dir(state_root / "cycles" / f"cycle-{state.cycle + 1:04d}")
    cycle_number = state.begin_cycle(next_cycle_dir)
    state.save(state_file_for(state_root))

    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_number,
        phase="cycle",
        message="started cycle",
        payload={"cycle_dir": str(next_cycle_dir.resolve())},
    )

    frontier_path = next_cycle_dir / "frontier.json"
    transplant_plan_path = next_cycle_dir / "initial_transplant_plan.json"

    frontier_started = time.time()
    _emit_progress(f"validation cycle {cycle_number}: rebuilding candidate frontier")
    frontier, frontier_preparation_failed, frontier_error = _safe_discover_candidate_frontier(
        spec,
        reason="initial semantic scope preparation failed",
    )
    frontier_duration = round(time.time() - frontier_started, 3)
    if frontier_preparation_failed:
        _emit_progress(
            f"validation cycle {cycle_number}: semantic scope preparation failed "
            f"duration={frontier_duration}s error={(frontier_error or '')[:200]}"
        )
    else:
        _emit_progress(
            f"validation cycle {cycle_number}: candidate frontier done "
            f"duration={frontier_duration}s"
        )
    write_json(frontier_path, frontier)
    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_number,
        phase="frontier",
        message=(
            "semantic scope preparation failed before transplant"
            if frontier_preparation_failed
            else "built candidate frontier"
        ),
        payload={"frontier_path": str(frontier_path.resolve()), "summary": frontier.get("summary", {})},
        level="warning" if frontier_preparation_failed else "info",
    )

    if frontier_preparation_failed:
        state.finish_cycle(
            status="needs_build_preparation",
            open_work_units=[],
        )
        state.save(state_file_for(state_root))
        return {
            "case_id": spec.case_id,
            "cycle": cycle_number,
            "status": "needs_build_preparation",
            "compile_ok": False,
            "overall_ok": False,
            "cycle_dir": str(next_cycle_dir.resolve()),
            "frontier_path": str(frontier_path.resolve()),
            "transplant_plan_path": None,
            "failed_stage": "semantic_scope_preparation",
            "error": frontier_error,
        }

    transplant_plan = build_initial_transplant_plan(spec, frontier)
    write_json(transplant_plan_path, transplant_plan)
    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_number,
        phase="transplant",
        message="built initial transplant plan",
        payload={
            "transplant_plan_path": str(transplant_plan_path.resolve()),
            "summary": transplant_plan.get("summary", {}),
        },
    )

    state.finish_cycle(
        status="needs_transplant",
        transplant_plan_path=transplant_plan_path,
        open_work_units=[],
    )
    state.save(state_file_for(state_root))

    return {
        "case_id": spec.case_id,
        "cycle": cycle_number,
        "status": "needs_transplant",
        "cycle_dir": str(next_cycle_dir.resolve()),
        "frontier_path": str(frontier_path.resolve()),
        "transplant_plan_path": str(transplant_plan_path.resolve()),
    }


def _verify_transplant_prerequisites_before_validation(
    spec: CaseSpec,
    *,
    state: CycleState,
) -> dict:
    if not state.last_transplant_plan_path:
        return {
            "status": "skipped",
            "reason": "missing_transplant_plan_path",
        }
    transplant_plan_path = Path(state.last_transplant_plan_path).resolve()
    if not transplant_plan_path.exists():
        return {
            "status": "skipped",
            "reason": "transplant_plan_path_missing",
            "transplant_plan_path": str(transplant_plan_path),
        }
    try:
        transplant_plan = json.loads(transplant_plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "status": "failed",
            "reason": "transplant_plan_unreadable",
            "transplant_plan_path": str(transplant_plan_path),
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    return verify_transplant_prerequisites(spec, transplant_plan)


def _empty_candidate_frontier(
    spec: CaseSpec,
    *,
    reason: str,
    error: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "preparation_failed": bool(error),
        "skipped": not bool(error),
        "reason": reason,
        "initial_copy_set": 0,
        "build_integration_set": 0,
    }
    if error:
        summary["error"] = error[:4000]
    return {
        "schema": "semantic_migration_scope_unavailable_v1",
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "summary": summary,
        "entry_frontiers": {},
        "initial_copy_set": [],
        "initial_shared_candidates": [],
        "build_integration_set": [],
        "candidate_frontier": [],
    }


def _safe_discover_candidate_frontier(
    spec: CaseSpec,
    *,
    reason: str,
) -> tuple[dict[str, Any], bool, str | None]:
    try:
        return build_semantic_migration_scope(spec), False, None
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        return _empty_candidate_frontier(spec, reason=reason, error=error), True, error


def _derive_diagnostic_work_units(
    spec: CaseSpec,
    *,
    validation: dict[str, Any],
    semantic_state: dict[str, Any] | None = None,
    semantic_state_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Build work units from one graph, optionally with semantic evidence."""

    if semantic_state is None:
        return derive_work_units(validation, spec=spec)
    graph_gap_ids = semantic_graph_gap_ids(semantic_state)
    evidence_refs = sorted(
        {
            str(gap.get("contract_evidence_ref") or "")
            for gap_id, gap in (semantic_state.get("gaps") or {}).items()
            if isinstance(gap, dict)
            and str(gap_id) in graph_gap_ids
            and str(gap.get("contract_evidence_ref") or "")
        }
    )
    if evidence_refs and semantic_state_root is None:
        raise ValueError(
            "semantic diagnostic graph requires its contract-evidence store"
        )
    contract_evidence_by_ref: dict[str, dict[str, Any]] = {}
    for evidence_ref in evidence_refs:
        record = load_semantic_record(
            Path(semantic_state_root),
            evidence_ref,
        )
        if not isinstance(record, dict):
            raise ValueError(
                "semantic contract evidence is not an object: "
                f"{evidence_ref}"
            )
        contract_evidence_by_ref[evidence_ref] = record
    return derive_work_units(
        validation,
        spec=spec,
        semantic_state=semantic_state,
        semantic_contract_evidence_by_ref=contract_evidence_by_ref,
    )


def run_post_validation_cycle(
    spec: CaseSpec,
    *,
    state_root: Path,
    validation_targets: list[dict[str, Any]] | None = None,
    validation_assessment_targets: list[dict[str, Any]] | None = None,
    execution_phase: str = EXECUTION_PHASE_COMPILER_REPAIR,
) -> dict:
    active_execution_phase = validate_execution_phase(execution_phase)
    ensure_dir(state_root)
    state = load_or_init_state(spec, state_root)
    previous_cycle_dir = Path(state.current_cycle_dir).resolve() if state.current_cycle_dir else None
    previous_validation_path = (
        Path(state.last_validation_path).resolve()
        if state.last_validation_path and Path(state.last_validation_path).exists()
        else None
    )
    previous_open_work_unit_ids = list(state.open_work_units)
    previous_pending_validation_work_unit_ids = list(
        state.pending_validation_work_units
    )
    previous_tracked_work_unit_ids = list(
        dict.fromkeys(
            [
                *previous_open_work_unit_ids,
                *previous_pending_validation_work_unit_ids,
            ]
        )
    )
    next_cycle_dir = ensure_dir(state_root / "cycles" / f"cycle-{state.cycle + 1:04d}")
    cycle_number = state.begin_cycle(next_cycle_dir)
    state.save(state_file_for(state_root))

    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_number,
        phase="cycle",
        message="started target compile validation cycle",
        payload={"cycle_dir": str(next_cycle_dir.resolve())},
    )

    validation_path = next_cycle_dir / "validation.json"
    repair_plan_path = next_cycle_dir / "repair_plan.json"
    cycle_started = time.time()
    post_validation_timings: dict[str, float] = {}

    guard_started = time.time()
    build_guard = _verify_transplant_prerequisites_before_validation(
        spec,
        state=state,
    )
    post_validation_timings["transplant_prerequisite_guard_sec"] = round(time.time() - guard_started, 3)
    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_number,
        phase="transplant-prerequisite-guard",
        message="ensured transplant prerequisites before validation",
        payload=build_guard,
        level="info" if str(build_guard.get("status") or "") in {"ok", "skipped"} else "warning",
    )
    if str(build_guard.get("status") or "") in {"failed", "partial"}:
        state.finish_cycle(status="needs_build_preparation")
        state.save(state_file_for(state_root))
        return {
            "case_id": spec.case_id,
            "cycle": cycle_number,
            "status": "needs_build_preparation",
            "execution_phase": active_execution_phase,
            "compile_ok": False,
            "overall_ok": False,
            "cycle_dir": str(next_cycle_dir.resolve()),
            "validation_path": None,
            "repair_plan_path": None,
            "summary": {
                "failed_phase": "transplant-prerequisite-guard",
                "issue_kinds": {"transplant_prerequisite_guard_failed": 1},
            },
            "transplant_prerequisite_guard": build_guard,
        }

    validation_started = time.time()
    _emit_progress(f"validation cycle {cycle_number}: running build/load validation")
    validation = run_validation(
        spec,
        artifacts_dir=next_cycle_dir / "validation_artifacts",
        progress=_emit_progress,
    )
    post_validation_timings["validation_wall_sec"] = round(time.time() - validation_started, 3)
    post_validation_timings["validation_phase_reported_sec"] = round(
        sum(float(phase.get("duration_sec") or 0.0) for phase in validation.get("phases") or []),
        3,
    )
    write_json(validation_path, validation)
    validation_delta: dict[str, Any] | None = None
    if previous_validation_path is not None:
        try:
            previous_validation = json.loads(previous_validation_path.read_text(encoding="utf-8"))
            validation_delta = {
                **_classify_validation_delta(
                    previous_validation,
                    validation,
                    spec=spec,
                ),
                "previous_validation_path": str(previous_validation_path),
                "current_validation_path": str(validation_path.resolve()),
            }
        except (OSError, ValueError):
            validation_delta = {
                "schema": "validation_delta_v3",
                "status": "unavailable",
                "previous_validation_path": str(previous_validation_path),
                "current_validation_path": str(validation_path.resolve()),
            }
    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_number,
        phase="validate",
        message="completed compile-first validation",
        payload={
            "validation_path": str(validation_path.resolve()),
            "summary": validation.get("summary", {}),
            "diagnostic_routes": diagnostic_route_summary(validation),
            **({"validation_delta": validation_delta} if validation_delta else {}),
        },
        level="info" if validation.get("overall_ok") else "warning",
    )

    build_prep_gate = build_prep_stage_decision(
        validation,
        validation_status=("ready_for_manual_test" if validation.get("overall_ok") else "needs_patching"),
    )
    if build_prep_gate.get("needs_build_prep"):
        post_validation_timings["total_post_validation_cycle_sec"] = round(
            time.time() - cycle_started,
            3,
        )
        state.finish_cycle(
            status="needs_build_preparation",
            validation_path=validation_path,
            open_work_units=previous_open_work_unit_ids,
            pending_validation_work_units=(
                previous_pending_validation_work_unit_ids
            ),
        )
        state.save(state_file_for(state_root))
        append_event(
            journal_file_for(state_root),
            case_id=spec.case_id,
            cycle=cycle_number,
            phase="build-preparation-gate",
            message="blocked source work-unit construction until A-stage preparation is revalidated",
            payload=build_prep_gate,
            level="warning",
        )
        return {
            "case_id": spec.case_id,
            "cycle": cycle_number,
            "status": "needs_build_preparation",
            "execution_phase": active_execution_phase,
            "cycle_dir": str(next_cycle_dir.resolve()),
            "validation_path": str(validation_path.resolve()),
            "repair_plan_path": None,
            "open_work_units": list(previous_open_work_unit_ids),
            "pending_validation_work_units": list(
                previous_pending_validation_work_unit_ids
            ),
            "failed_stage": "build_preparation",
            "diagnostic_routes": build_prep_gate,
            "post_validation_timings": post_validation_timings,
        }

    derive_started = time.time()
    _emit_progress(f"validation cycle {cycle_number}: deriving work units from validation report")
    semantic_state: dict[str, Any] | None = None
    if active_execution_phase == EXECUTION_PHASE_SEMANTIC_CLOSURE:
        semantic_state = load_semantic_closure_state(
            state_root,
            case_id=spec.case_id,
        )
        if semantic_state is None:
            raise ValueError(
                "semantic compiler recovery has no persistent ledger"
            )
    fresh_work_units = _derive_diagnostic_work_units(
        spec,
        validation=validation,
        semantic_state=semantic_state,
        semantic_state_root=(
            state_root if semantic_state is not None else None
        ),
    )
    semantic_recovery_binding: dict[str, Any] = {}
    if (
        active_execution_phase == EXECUTION_PHASE_SEMANTIC_CLOSURE
        and fresh_work_units
    ):
        assert semantic_state is not None
        binding = bind_semantic_compile_recovery_work_units(
            spec,
            state=semantic_state,
            diagnostic_work_units=fresh_work_units,
        )
        fresh_work_units = list(binding.get("work_units") or [])
        semantic_recovery_binding = {
            "binding": dict(binding.get("summary") or {}),
        }
    _emit_progress(
        f"validation cycle {cycle_number}: derived {len(fresh_work_units)} fresh work units "
        f"duration={round(time.time() - derive_started, 3)}s"
    )
    post_validation_timings["derive_work_units_sec"] = round(time.time() - derive_started, 3)

    reconcile_started = time.time()
    _emit_progress(f"validation cycle {cycle_number}: reconciling work-unit registry")
    current_fresh_work_units = [
        bind_execution_phase(work_unit, active_execution_phase)
        for work_unit in fresh_work_units
    ]
    (
        work_units,
        closed_work_unit_ids,
        deferred_validation_work_unit_ids,
    ) = _reconcile_work_units_after_validation(
        spec=spec,
        state_root=state_root,
        previous_open_ids=previous_tracked_work_unit_ids,
        fresh_work_units=current_fresh_work_units,
        validation=validation,
        validation_ok=bool(validation.get("overall_ok")),
    )
    registry_dir = work_unit_registry_dir_for(state_root)
    if validation_delta is not None:
        deferred_set = set(deferred_validation_work_unit_ids)
        round_targets = [
            target
            for target in (
                validation_assessment_targets
                if validation_assessment_targets is not None
                else validation_targets
                or []
            )
            if isinstance(target, dict)
        ]
        if round_targets:
            validation_delta = _classify_round_root_validation_delta(
                validation_delta,
                validation_targets=round_targets,
                fresh_work_units=current_fresh_work_units,
                deferred_work_unit_ids=deferred_set,
                spec=spec,
            )
    if not validation.get("overall_ok"):
        work_units, previous_attempt_attachment_count = (
            _attach_previous_unresolved_patch_attempts(
                work_units,
                spec=spec,
                previous_cycle_dir=previous_cycle_dir,
                state_root=state_root,
                current_validation_delta=validation_delta,
            )
        )
    else:
        previous_attempt_attachment_count = 0
    _emit_progress(
        f"validation cycle {cycle_number}: reconciled {len(work_units)} open work units "
        f"closed={len(closed_work_unit_ids)} "
        f"coverage_deferred={len(deferred_validation_work_unit_ids)} "
        f"duration={round(time.time() - reconcile_started, 3)}s"
    )
    post_validation_timings["reconcile_work_units_sec"] = round(time.time() - reconcile_started, 3)

    registry_dir = work_unit_registry_dir_for(state_root)
    write_units_started = time.time()
    _emit_progress(f"validation cycle {cycle_number}: writing work-unit snapshots")
    written_work_units = write_work_units(
        work_units,
        work_units_dir=registry_dir,
    )
    _emit_progress(
        f"validation cycle {cycle_number}: wrote {len(written_work_units)} work-unit snapshots "
        f"duration={round(time.time() - write_units_started, 3)}s"
    )
    post_validation_timings["write_work_units_sec"] = round(time.time() - write_units_started, 3)
    open_work_unit_ids = [item["id"] for item in work_units]
    plan_started = time.time()
    _emit_progress(
        f"validation cycle {cycle_number}: building repair plan "
        f"planned={len(work_units)}"
    )
    repair_plan = build_repair_plan(
        spec,
        work_units,
        progress=_emit_progress,
    )
    _emit_progress(
        f"validation cycle {cycle_number}: repair plan done "
        f"plans={len(repair_plan.get('work_unit_plans') or [])} "
        "source_packs=deferred "
        f"duration={round(time.time() - plan_started, 3)}s"
    )
    post_validation_timings["build_repair_plan_sec"] = round(time.time() - plan_started, 3)
    write_plan_started = time.time()
    write_repair_plan(
        repair_plan_path,
        repair_plan,
        state_root=state_root,
    )
    post_validation_timings["write_repair_plan_sec"] = round(time.time() - write_plan_started, 3)
    post_validation_timings["total_post_validation_cycle_sec"] = round(time.time() - cycle_started, 3)

    status = _repair_status_from_plan(
        validation_ok=bool(validation.get("overall_ok")),
        repair_plan=repair_plan,
    )
    if status == "needs_patching" and not (repair_plan.get("work_unit_plans") or []):
        status = "patch_apply_incomplete"
    state.finish_cycle(
        status=status,
        validation_path=validation_path,
        open_work_units=open_work_unit_ids,
        pending_validation_work_units=deferred_validation_work_unit_ids,
    )
    state.save(state_file_for(state_root))

    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_number,
        phase="work-unit",
        message="derived work units from validation report",
        payload={
            "work_unit_count": len(work_units),
            "registry_dir": str(registry_dir.resolve()),
            "open_work_units": open_work_unit_ids,
            "closed_work_units": closed_work_unit_ids,
            "pending_validation_work_units": (
                deferred_validation_work_unit_ids
            ),
            "previous_unresolved_patch_attempt_attachment_count": previous_attempt_attachment_count,
            "execution_phase": active_execution_phase,
            **(
                {"semantic_recovery_binding": semantic_recovery_binding}
                if semantic_recovery_binding
                else {}
            ),
            **({"validation_delta": validation_delta} if validation_delta else {}),
            "timings": post_validation_timings,
        },
        level="info" if not work_units else "warning",
    )
    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_number,
        phase="plan",
        message="built repair plan from work units",
        payload={
            "repair_plan_path": str(repair_plan_path.resolve()),
            "summary": repair_plan.get("summary", {}),
            "patchable_work_unit_count": _repair_plan_summary_counts(repair_plan)["patchable_work_unit_count"],
            **({"validation_delta": validation_delta} if validation_delta else {}),
            "timings": post_validation_timings,
        },
        level="info" if not repair_plan["summary"].get("plans_requiring_escalation") else "warning",
    )
    return {
        "case_id": spec.case_id,
        "cycle": cycle_number,
        "status": status,
        "compile_ok": bool(validation.get("overall_ok")),
        "overall_ok": bool(validation.get("overall_ok")),
        "cycle_dir": str(next_cycle_dir.resolve()),
        "validation_path": str(validation_path.resolve()),
        "repair_plan_path": str(repair_plan_path.resolve()),
        "work_unit_paths": [str(path.resolve()) for path in written_work_units],
        "open_work_units": open_work_unit_ids,
        "pending_validation_work_units": (
            deferred_validation_work_unit_ids
        ),
        "diagnostic_routes": diagnostic_route_summary(validation),
        "execution_phase": active_execution_phase,
        **(
            {"semantic_recovery_binding": semantic_recovery_binding}
            if semantic_recovery_binding
            else {}
        ),
        **({"validation_delta": validation_delta} if validation_delta else {}),
        "post_validation_timings": post_validation_timings,
    }


def show_status(spec: CaseSpec, *, state_root: Path) -> dict:
    state = load_or_init_state(spec, state_root)
    payload = state.to_dict()
    payload["state_root"] = str(state_root.resolve())
    return payload


def _snapshot_pre_transplant_cycle_artifact(path: Path) -> Path | None:
    if not path.exists():
        return None
    snapshot_path = path.with_name(
        f"{path.stem}.pre_transplant{path.suffix}"
    )
    if not snapshot_path.exists():
        snapshot_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return snapshot_path


def _refresh_post_transplant_cycle_artifacts(
    spec: CaseSpec,
    *,
    cycle_dir: Path,
) -> tuple[Path, dict]:
    frontier_path = cycle_dir / "frontier.json"

    _snapshot_pre_transplant_cycle_artifact(frontier_path)

    frontier, _preparation_failed, _error = _safe_discover_candidate_frontier(
        spec,
        reason="post-transplant semantic scope preparation failed",
    )
    write_json(frontier_path, frontier)
    return frontier_path, frontier


def apply_transplant_only(spec: CaseSpec, *, state_root: Path, plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    result = apply_initial_transplant_plan(spec, plan, state_root=state_root, plan_path=plan_path)

    state = load_or_init_state(spec, state_root)
    status = "transplant_applied_pending_validation"
    if result["failed_copy_operations"] or result["failed_build_integrations"]:
        status = "transplant_apply_failed"
    elif result["blocked_copy_operations"] or result["blocked_build_integrations"]:
        status = "transplant_apply_blocked"
    frontier_path = None
    if status == "transplant_applied_pending_validation":
        frontier_path, frontier = _refresh_post_transplant_cycle_artifacts(
            spec,
            cycle_dir=plan_path.resolve().parent,
        )
        pre_frontier_path = frontier_path.with_name(
            f"{frontier_path.stem}.pre_transplant{frontier_path.suffix}"
        )
        if pre_frontier_path.exists():
            result["pre_transplant_frontier_path"] = str(pre_frontier_path.resolve())
        result["refreshed_frontier_path"] = str(frontier_path.resolve())
        result["refreshed_frontier_summary"] = frontier.get("summary", {})
        if bool((frontier.get("summary") or {}).get("preparation_failed")):
            status = "needs_build_preparation"
    state.finish_cycle(
        status=status,
        transplant_plan_path=plan_path,
    )
    state.save(state_file_for(state_root))

    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_hint_from_path(plan_path),
        phase="transplant-apply",
        message="applied initial transplant copy operations",
        payload={
            "plan_path": str(plan_path.resolve()),
            "applied": result["applied_copy_operations"],
            "blocked": result["blocked_copy_operations"],
            "failed": result["failed_copy_operations"],
            "applied_build_integrations": result["applied_build_integrations"],
            "blocked_build_integrations": result["blocked_build_integrations"],
            "failed_build_integrations": result["failed_build_integrations"],
            "explicit_kconfig_symbol_verification": result.get("explicit_kconfig_symbol_verification"),
            "results": result.get("results") or [],
            "transaction_dir": result["transaction_dir"],
        },
        level=(
            "warning"
            if (
                result["failed_copy_operations"]
                or result["blocked_copy_operations"]
                or result["failed_build_integrations"]
                or result["blocked_build_integrations"]
            )
            else "info"
        ),
    )
    if status == "transplant_applied_pending_validation":
        append_event(
            journal_file_for(state_root),
            case_id=spec.case_id,
            cycle=cycle_hint_from_path(plan_path),
            phase="frontier-refresh",
            message="rebuilt candidate frontier after transplant",
            payload={
                "frontier_path": result["refreshed_frontier_path"],
                "frontier_summary": result["refreshed_frontier_summary"],
            },
        )
    return result



def _status_from_apply_results(results: list[dict]) -> str:
    if any(
        item["status"] in APPLY_INCOMPLETE_RESULT_STATUSES
        for item in results
    ):
        return "patch_apply_incomplete"
    if any(
        str(item.get("status") or "") == "applied"
        for item in results
        if isinstance(item, dict)
    ):
        return "repair_applied_pending_validation"
    # already_applied/no_patch/escalation results do not create a new target
    # postimage.  Only an earlier, genuinely unvalidated apply may preserve
    # the pending-validation state through _preserve_pending_validation_status.
    return "needs_patching"


def _preserve_pending_validation_status(
    *,
    current_status: str,
    computed_status: str,
    summary: dict[str, Any],
) -> str:
    if (
        current_status == "repair_applied_pending_validation"
        and computed_status in {"needs_patching", "patch_apply_incomplete"}
    ):
        summary["preserved_pending_validation_status"] = {
            "previous_status": current_status,
            "computed_status": computed_status,
            "reason": (
                "a previous repair apply in this cycle already changed the tree; "
                "later retry-only failures must not downgrade the case before validation"
            ),
        }
        return "repair_applied_pending_validation"
    return computed_status


def _coherent_terminal_case_status(
    *,
    requested_status: str,
    persisted_status: str,
    final_compile_validated: bool,
) -> str:
    """Prevent a stale compile-valid baseline from describing an invalid tree."""

    if requested_status != "ready_for_manual_test" or final_compile_validated:
        return requested_status
    if persisted_status and persisted_status != "ready_for_manual_test":
        return persisted_status
    return "needs_patching"


def _apply_summary_has_effect(summary: dict) -> bool:
    return int(summary.get("applied_count") or 0) > 0


def _already_present_patch_can_join_round_validation(
    summary: dict[str, Any],
    *,
    has_unvalidated_round_changes: bool,
) -> bool:
    if not has_unvalidated_round_changes:
        return False
    statuses = [
        str(result.get("status") or "")
        for result in summary.get("results") or []
        if isinstance(result, dict)
    ]
    return bool(
        statuses
        and all(
            status in {"already_applied", "already_applied_duplicate"}
            for status in statuses
        )
    )


def _apply_plan_responses(
    spec: CaseSpec,
    *,
    state_root: Path,
    plan_path: Path,
    response_dir: Path,
    allow_escalation: bool,
    output: Path | None,
) -> dict:
    selected_plan_path = plan_path.resolve()
    plan = load_repair_plan(
        selected_plan_path,
        state_root=state_root,
    )
    units = repair_plan_units(plan)
    results = apply_plan_responses(
        spec,
        plan,
        response_dir=response_dir,
        state_root=state_root,
        allow_escalation=allow_escalation,
        plan_path=selected_plan_path,
    )
    summary = {
        "case_id": spec.case_id,
        "plan_kind": "repair",
        "plan_path": str(selected_plan_path.resolve()),
        "source_plan_path": str(plan_path.resolve()),
        "response_dir": str(response_dir.resolve()),
        "patch_transaction_preview_path": str((response_dir / "patch_transaction_preview.json").resolve()),
        "allow_escalation": allow_escalation,
        **summarize_apply_results(results),
        "unit_count": len(units),
        "results": results,
    }
    repair_no_effect = (
        summary["applied_count"] == 0
        and (
            summary["no_patch_count"] > 0
            or summary["already_applied_count"] > 0
        )
        and summary["incomplete_count"] == 0
        and summary["needs_escalation_count"] == 0
    )
    summary["repair_no_effect"] = repair_no_effect
    output_path = output.resolve() if output else (response_dir / "apply_results.json").resolve()
    write_json(output_path, summary)

    state = load_or_init_state(spec, state_root)
    _update_repair_work_units_for_apply_results(
        state_root=state_root,
        plan=plan,
        results=results,
        no_patch_means_escalation=repair_no_effect,
    )

    status = _status_from_apply_results(results)
    if status != "patch_apply_incomplete":
        if summary["applied_count"] == 0 and (
            repair_no_effect or summary["needs_escalation_count"] > 0
        ):
            status = "needs_patching"
    status = _preserve_pending_validation_status(
        current_status=state.status,
        computed_status=status,
        summary=summary,
    )
    state.finish_cycle(status=status)
    state.save(state_file_for(state_root))

    append_event(
        journal_file_for(state_root),
        case_id=spec.case_id,
        cycle=cycle_hint_from_path(plan_path),
        phase="response-apply",
        message="applied plan responses",
        payload={
            "plan_path": str(selected_plan_path.resolve()),
            "source_plan_path": str(plan_path.resolve()),
            "response_dir": str(response_dir.resolve()),
            "output_path": str(output_path),
            "status": status,
            "applied_count": summary["applied_count"],
            "no_patch_count": summary["no_patch_count"],
            "repair_no_effect": repair_no_effect,
            "needs_escalation_count": summary["needs_escalation_count"],
            "incomplete_count": summary["incomplete_count"],
            "apply_failure_metrics": summary["apply_failure_metrics"],
        },
        level="warning" if status == "patch_apply_incomplete" else "info",
    )
    summary["status"] = status
    write_json(output_path, summary)
    return summary


def _route_agentic_apply_failures(
    apply_summary: dict[str, Any],
) -> dict[str, Any]:
    """Return apply failures to the owning work-unit loop with typed evidence."""

    results: list[dict[str, Any]] = []
    for raw_result in apply_summary.get("results") or []:
        result = dict(raw_result)
        if str(result.get("status") or "") not in APPLY_FAILURE_RESULT_STATUSES:
            results.append(result)
            continue
        failure = result.get("apply_failure")
        route = (
            apply_failure_route(failure)
            if isinstance(failure, dict)
            else {
                "route": "terminal_infrastructure",
                "reason": "apply result violated the typed failure protocol",
                "required_transition": "repair_apply_failure_instrumentation",
                "retry_requires_state_change": True,
            }
        )
        results.append(
            {
                **result,
                "original_status": result.get("status"),
                "status": "rerun_flow_required",
                "invalid_route": route,
            }
        )

    routed = {
        **apply_summary,
        **summarize_apply_results(results),
        "results": results,
    }
    routed["invalid_route_summary"] = {
        route: sum(
            1
            for result in results
            if (result.get("invalid_route") or {}).get("route") == route
        )
        for route in ("outer_unit_replan", "terminal_infrastructure")
    }
    routed["routed_failure_without_validation_worthy_change"] = bool(
        not _apply_summary_has_effect(routed)
        and int(routed.get("rerun_flow_required_count") or 0) > 0
    )
    status = _status_from_apply_results(results)
    if routed["needs_escalation_count"] and not _apply_summary_has_effect(routed):
        status = "needs_patching"
    routed["status"] = _preserve_pending_validation_status(
        current_status=str(apply_summary.get("status") or ""),
        computed_status=status,
        summary=routed,
    )
    return routed


def _compact_agentic_apply_failures(
    apply_summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    compact: list[dict[str, Any]] = []
    routes: list[str] = []
    for result in apply_summary.get("results") or []:
        if not isinstance(result, dict):
            continue
        invalid_route = (
            result.get("invalid_route")
            if isinstance(result.get("invalid_route"), dict)
            else {}
        )
        route = str(invalid_route.get("route") or "")
        if route:
            routes.append(route)
        compact.append(_compact_routed_apply_failure(result))
    return compact, list(dict.fromkeys(routes))


def _terminal_agentic_apply_route(routes: list[str]) -> str | None:
    return (
        "terminal_infrastructure"
        if "terminal_infrastructure" in routes
        else None
    )



def _validation_blocks_auto_repair(validation_result: dict) -> bool:
    routes = validation_result.get("diagnostic_routes") or {}
    return bool(
        int(routes.get("validation_environment_count") or 0)
        and not int(routes.get("repair_routed_count") or 0)
        and not int(routes.get("build_prep_count") or 0)
    )


def _retain_round_candidates(
    *,
    validation_delta: dict[str, Any],
    applied_units: list[dict[str, Any]],
) -> dict[str, Any]:
    unproductive_statuses = {
        "unchanged",
        "regressed",
        "regressed_at_patch_locus",
        "compile_regressed_from_valid_checkpoint",
        "pending_compile_recovery",
        "changed",
        "diagnostics_regressed",
    }
    status_by_work_unit = {
        str(item.get("target_work_unit_id") or ""): str(
            item.get("status") or ""
        )
        for item in validation_delta.get("target_results") or []
        if isinstance(item, dict) and item.get("target_work_unit_id")
    }
    global_status = str(validation_delta.get("status") or "")
    records: list[dict[str, Any]] = []
    unproductive_indexes: set[int] = set()
    for index, record in enumerate(applied_units):
        work_unit_id = str(record.get("work_unit_id") or "")
        target_status = status_by_work_unit.get(work_unit_id)
        if target_status is None and not status_by_work_unit:
            target_status = global_status
        apply_results = [
            result
            for result in (record.get("apply_result") or {}).get(
                "results"
            )
            or []
            if isinstance(result, dict)
        ]
        touched_files = {
            normalize_relpath(str(path))
            for result in apply_results
            if str(result.get("status") or "")
            in {"applied", "already_applied"}
            for path in result.get("touched_files") or []
            if normalize_relpath(str(path))
        }
        transaction_dirs = sorted(
            {
                str(result.get("transaction_dir") or "")
                for result in apply_results
                if str(result.get("transaction_dir") or "")
            }
        )
        patch_outcome_candidate_ids = sorted(
            {
                str(
                    (result.get("patch_outcome_candidate") or {}).get(
                        "candidate_id"
                    )
                    or ""
                )
                for result in apply_results
                if isinstance(
                    result.get("patch_outcome_candidate"), dict
                )
                and str(
                    (result.get("patch_outcome_candidate") or {}).get(
                        "candidate_id"
                    )
                    or ""
                )
            }
        )
        records.append(
            {
                "plan_unit_id": record.get("plan_unit_id"),
                "work_unit_id": work_unit_id,
                "validation_status": target_status,
                "touched_files": sorted(touched_files),
                "transaction_dirs": transaction_dirs,
                "patch_outcome_candidate_ids": (
                    patch_outcome_candidate_ids
                ),
            }
        )
        if target_status in unproductive_statuses:
            unproductive_indexes.add(index)

    retained = [
        {
            **record,
            "reason": (
                "transaction retained until compiler recovery and exact "
                "semantic contract revalidation complete"
            ),
            "required_transition": (
                "continue from the current candidate tree using freshly "
                "derived diagnostic code-graph work units"
            ),
        }
        for record in records
    ]
    return {
        "schema": "round_candidate_retention_state_v1",
        "status": (
            "retained_pending_repair"
            if retained
            and validation_delta.get("current_overall_ok") is False
            else "retained_pending_semantic_validation"
            if retained
            else "no_applied_candidates"
        ),
        "policy": "retain_applied_candidates_for_fresh_validation",
        "unproductive_count": len(unproductive_indexes),
        "retained_count": len(retained),
        "retained": retained,
    }


def _graph_incomplete_plan_unit_ids(repair_plan: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for unit in repair_plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        decision = dict(unit.get("repair_decision") or {})
        continuation = ((unit.get("code_graph_scope") or {}).get("continuation") or {})
        incomplete = not _graph_plan_unit_patch_ready(unit) and (
            bool(unit.get("requires_context_expansion"))
            or str(decision.get("mode") or "") == "expand_graph"
            or decision.get("root_mapping_complete") is False
            or decision.get("scope_complete") is False
            or bool(continuation and not continuation.get("complete"))
        )
        if incomplete:
            result.append(str(unit.get("id") or unit.get("work_unit_id") or ""))
    return [item for item in result if item]


def _graph_expansion_should_continue(
    repair_plan: dict[str, Any],
    *,
    selected_plan_unit_id: str | None,
    continuation: dict[str, Any] | None = None,
) -> bool:
    if not selected_plan_unit_id:
        return False
    if continuation is not None:
        selected_pages = [
            page
            for page in continuation.get("pages") or []
            if isinstance(page, dict)
            and str(page.get("plan_unit_id") or "")
            == selected_plan_unit_id
        ]
        if any(
            (page.get("after") or {}).get("complete") is True
            for page in selected_pages
        ):
            return False
        if int(continuation.get("progressed_count") or 0) == 0:
            return False
    return selected_plan_unit_id in set(
        _graph_incomplete_plan_unit_ids(repair_plan)
    )


def _initial_analysis_boundary_compile_transition(
    queue_status: dict[str, Any],
    plan_summary: dict[str, Any],
) -> dict[str, Any] | None:
    pending_count = int(
        queue_status.get("pending_work_unit_count") or 0
    )
    blocked_count = int(
        queue_status.get("pending_context_blocked_work_unit_count")
        or plan_summary.get("blocked_unit_count")
        or 0
    )
    actionable_count = int(
        queue_status.get("pending_actionable_work_unit_count") or 0
    )
    if (
        pending_count <= 0
        or blocked_count != pending_count
        or actionable_count != 0
    ):
        return None
    return {
        "schema": "initial_analysis_boundary_transition_v1",
        "status": "compile_required",
        "reason": (
            "all remaining initial units depend on target translation units "
            "with build-confirmed active unresolved includes"
        ),
        "pending_work_unit_count": pending_count,
        "pending_context_blocked_work_unit_ids": list(
            queue_status.get("pending_context_blocked_work_unit_ids") or []
        ),
        "analysis_validity_boundaries": list(
            plan_summary.get("analysis_validity_boundaries") or []
        ),
        "required_transition": (
            "compile_current_target_then_rebuild_diagnostic_code_graph_"
            "and_preserve_unconfirmed_demands"
        ),
        "source_repair_blocked": False,
    }


def _initial_queue_compile_gate(
    queue_status: dict[str, Any],
    plan_summary: dict[str, Any],
) -> dict[str, Any]:
    """Permit compile only for a drained queue or an exact analysis boundary."""

    pending_count = int(
        queue_status.get("pending_work_unit_count") or 0
    )
    if pending_count <= 0:
        return {
            "schema": "initial_queue_compile_gate_v1",
            "status": "ready",
            "pending_work_unit_count": 0,
        }
    boundary = _initial_analysis_boundary_compile_transition(
        queue_status,
        plan_summary,
    )
    if boundary is not None:
        return boundary
    return {
        "schema": "initial_queue_compile_gate_v1",
        "status": "blocked",
        "reason": (
            "pending initial units do not have a valid compile-boundary "
            "transition"
        ),
        "pending_work_unit_count": pending_count,
        "pending_actionable_work_unit_count": int(
            queue_status.get("pending_actionable_work_unit_count") or 0
        ),
        "pending_context_blocked_work_unit_count": int(
            queue_status.get(
                "pending_context_blocked_work_unit_count"
            )
            or 0
        ),
        "pending_work_unit_ids": list(
            queue_status.get("pending_work_unit_ids") or []
        ),
        "source_repair_blocked": True,
    }


def _initial_api_failure_queue_evidence(
    cycle_dir: Path,
) -> dict[str, Any]:
    try:
        queue = initial_compatibility_queue_status(cycle_dir)
    except (OSError, ValueError, KeyError) as exc:
        return {
            "initial_compatibility_queue": {
                "status": "snapshot_failed",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        }
    return {
        "initial_compatibility_queue": queue,
        "initial_attempted_work_unit_count": int(
            queue.get("attempted_work_unit_count") or 0
        ),
        "initial_resolved_work_unit_count": int(
            queue.get("resolved_work_unit_count") or 0
        ),
        "initial_unresolved_work_unit_count": int(
            queue.get("unresolved_work_unit_count") or 0
        ),
        "initial_pending_work_unit_count": int(
            queue.get("pending_work_unit_count") or 0
        ),
    }






def _graph_plan_unit_patch_ready(unit: dict[str, Any]) -> bool:
    decision = dict(unit.get("repair_decision") or {})
    consumer_authority = dict(
        decision.get("consumer_adaptation_authority") or {}
    )
    exact_consumer_boundary = bool(
        decision.get("repair_authority")
        == "exhausted_provider_query_consumer"
        and decision.get("root_mapping_complete") is False
        and consumer_authority.get("status") == "authorized"
        and consumer_authority.get("provider_resolution_complete") is False
        and consumer_authority.get("bounded_provider_query_complete") is True
        and consumer_authority.get("candidate_write_set")
    )
    exact_postimage_boundary = exact_active_patch_postimage_boundary(
        unit,
        decision,
    )
    return bool(
        str(decision.get("mode") or "") == "patch_root_subgraph"
        and (
            decision.get("root_mapping_complete") is True
            or exact_consumer_boundary
            or exact_postimage_boundary
        )
        and decision.get("scope_complete") is True
        and (
            unit.get("allowed_fileset")
            or unit.get("escalation_candidates")
            or exact_entity_materialization_owner_ids(unit)
        )
    )


def _next_graph_plan_unit_id(repair_plan: dict[str, Any]) -> str | None:
    """Prefer executable work within the dependency-ready plan frontier."""

    incomplete = set(_graph_incomplete_plan_unit_ids(repair_plan))
    units = [
        unit
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
    ]
    for unit in units:
        unit_id = str(unit.get("id") or "")
        if unit_id and _graph_plan_unit_patch_ready(unit):
            return unit_id
    for unit in units:
        unit_id = str(unit.get("id") or "")
        mode = str((unit.get("repair_decision") or {}).get("mode") or "")
        if (
            unit_id
            and mode == "materialize_graph_scope"
            and unit.get("scope_expansion_candidates")
        ):
            return unit_id
    for unit in units:
        unit_id = str(unit.get("id") or "")
        if unit_id and unit_id in incomplete:
            return unit_id
    return None


def _select_single_root_repair_plan(
    repair_plan: dict[str, Any],
    *,
    selected_id: str,
) -> dict[str, Any]:
    original_units = [unit for unit in repair_plan.get("work_unit_plans") or [] if isinstance(unit, dict)]
    selected_units = [unit for unit in original_units if str(unit.get("id") or "") == selected_id]
    if len(selected_units) != 1:
        raise ValueError(f"active graph root must identify exactly one repair-plan unit: {selected_id}")
    filtered = dict(repair_plan)
    filtered["work_unit_plans"] = copy.deepcopy(selected_units)
    summary = dict(filtered.get("summary") or {})
    summary["work_unit_count"] = len(selected_units)
    summary["active_graph_root"] = {
        "selection": "first_stable_root_in_plan_order",
        "selected_plan_unit_id": selected_id,
        "deferred_root_count": max(0, len(original_units) - 1),
    }
    filtered["summary"] = summary
    return filtered


def _select_validation_batch_repair_plan(
    repair_plan: dict[str, Any],
    *,
    spec: CaseSpec | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    units = [
        unit
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
    ]

    def evidence_repeated(unit: dict[str, Any]) -> bool:
        current = _repair_evidence_snapshot(unit, spec=spec)
        current_identity = _repair_evidence_identity(current)
        for attempt in unit.get("prior_unresolved_patch_attempts") or []:
            if not isinstance(attempt, dict):
                continue
            previous = attempt.get("repair_evidence_snapshot")
            if (
                isinstance(previous, dict)
                and _repair_evidence_identity(previous)
                == current_identity
            ):
                return True
        return False

    def action_ready(unit: dict[str, Any]) -> bool:
        mode = str((unit.get("repair_decision") or {}).get("mode") or "")
        if mode == "patch_root_subgraph":
            return _graph_plan_unit_patch_ready(unit)
        if mode == "materialize_graph_scope":
            return bool(unit.get("scope_expansion_candidates"))
        if mode == "expand_graph":
            return True
        return False

    original_position = {
        str(unit.get("work_unit_id") or unit.get("id") or ""): index
        for index, unit in enumerate(units)
    }

    def fairness_key(unit: dict[str, Any]) -> tuple[int, int, int, str]:
        attempts = [
            attempt
            for attempt in unit.get("prior_unresolved_patch_attempts") or []
            if isinstance(attempt, dict)
        ]
        cycles = [
            int(attempt.get("cycle") or 0)
            for attempt in attempts
            if str(attempt.get("cycle") or "").isdigit()
        ]
        unit_id = str(unit.get("work_unit_id") or unit.get("id") or "")
        return (
            len(attempts),
            max(cycles, default=0),
            original_position.get(unit_id, 0),
            unit_id,
        )

    fair_units = sorted(units, key=fairness_key)
    unit_by_work_id = {
        str(unit.get("work_unit_id") or unit.get("id") or ""): unit
        for unit in fair_units
        if str(unit.get("work_unit_id") or unit.get("id") or "")
    }
    action_ready_ids = {
        unit_id
        for unit_id, unit in unit_by_work_id.items()
        if action_ready(unit)
    }
    selection = select_dependency_ready_frontier(
        fair_units,
        eligible=lambda unit: str(
            unit.get("work_unit_id") or unit.get("id") or ""
        )
        in action_ready_ids,
    )
    selected_work_unit_ids = {
        str(item)
        for item in selection.get("selected_work_unit_ids") or []
        if str(item)
    }
    repeated_evidence_ids = sorted(
        unit_id
        for unit_id, unit in unit_by_work_id.items()
        if action_ready(unit) and evidence_repeated(unit)
    )
    selection["repair_evidence_novelty"] = {
        "schema": "repair_evidence_novelty_v2",
        "policy": (
            "evidence history informs fair scheduling; only an identical "
            "action against an identical evidence version is rejected"
        ),
        "untried_work_unit_ids": sorted(
            selected_work_unit_ids - set(repeated_evidence_ids)
        ),
        "repeated_evidence_work_unit_ids": repeated_evidence_ids,
        "exhausted_work_unit_ids": [],
        "dependency_blocked_work_unit_ids": sorted(
            action_ready_ids - selected_work_unit_ids
        ),
        "fair_ordered_work_unit_ids": [
            str(unit.get("work_unit_id") or unit.get("id") or "")
            for unit in fair_units
        ],
    }

    selected_units = list(selection.get("selected_units") or [])
    deferred_units = [unit for unit in units if unit not in selected_units]

    selected_plan = dict(repair_plan)
    selected_plan["work_unit_plans"] = list(selected_units)
    selected_summary = dict(selected_plan.get("summary") or {})
    selected_summary.update(
        {
            "work_unit_count": len(selected_units),
            "validation_batch": {
                key: value
                for key, value in selection.items()
                if key != "selected_units"
            },
        }
    )
    selected_plan["summary"] = selected_summary

    deferred_plan = dict(repair_plan)
    deferred_plan["work_unit_plans"] = list(deferred_units)
    deferred_summary = dict(deferred_plan.get("summary") or {})
    deferred_summary["work_unit_count"] = len(deferred_units)
    deferred_summary["deferred_by_validation_batch"] = True
    deferred_plan["summary"] = deferred_summary
    return selected_plan, deferred_plan, selection


def _repair_plan_without_unit(
    repair_plan: dict[str, Any],
    *,
    completed_plan_unit_id: str,
    deferred_plan_unit_ids: Iterable[str] = (),
) -> dict[str, Any]:
    deferred_ids = {
        str(unit_id) for unit_id in deferred_plan_unit_ids if str(unit_id)
    }
    removed_ids = {str(completed_plan_unit_id), *deferred_ids}
    remaining = [
        unit
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and str(unit.get("id") or "") not in removed_ids
    ]
    result = dict(repair_plan)
    result["work_unit_plans"] = remaining
    summary = dict(result.get("summary") or {})
    summary["work_unit_count"] = len(remaining)
    summary["round_remaining_work_unit_count"] = len(remaining)
    summary["deferred_until_replan_plan_unit_ids"] = sorted(
        deferred_ids
    )
    result["summary"] = summary
    return result


def _repair_plan_root_keys(unit: dict[str, Any]) -> set[str]:
    keys: set[str] = set()

    def add_binding(binding: dict[str, Any]) -> None:
        root_key = str(
            binding.get("root_key") or binding.get("key") or ""
        )
        node_id = str(
            binding.get("root_node_id")
            or binding.get("node_id")
            or binding.get("entity_id")
            or ""
        )
        contract_key = str(
            binding.get("root_contract_key")
            or binding.get("contract_key")
            or ""
        )
        if root_key:
            keys.add(f"root:{root_key}")
        if node_id:
            keys.add(f"node:{node_id}")
        if contract_key:
            keys.add(f"contract:{contract_key}")

    add_binding(unit)
    decision = unit.get("repair_decision")
    if isinstance(decision, dict):
        add_binding(decision)
        keys.update(
            f"node:{node_id}"
            for node_id in decision.get("root_alternative_node_ids") or []
            if str(node_id)
        )
    for field in (
        "code_graph_root_cause_cluster",
        "root_cause_cluster",
    ):
        root = unit.get(field)
        if isinstance(root, dict):
            add_binding(root)
    for root in unit.get("root_cause_alternatives") or []:
        if isinstance(root, dict):
            add_binding(root)
    for owner in unit.get("repair_owners") or []:
        if not isinstance(owner, dict) or str(
            owner.get("owner_role") or ""
        ) != "root_owner":
            continue
        contract_key = str(owner.get("owner_contract_key") or "")
        entity_id = str(owner.get("owner_entity_id") or "")
        if contract_key:
            keys.add(f"contract:{contract_key}")
        if entity_id:
            keys.add(f"entity:{entity_id}")
    return keys


def _round_root_overlap_after_apply(
    repair_plan: dict[str, Any],
    *,
    completed_plan_unit_id: str,
    changed_files: Iterable[str] = (),
    added_contract_definitions: Iterable[dict[str, Any]] = (),
    old_root: Path | None = None,
) -> dict[str, Any]:
    """Defer shared roots and diagnostics that cannot be rebound safely."""

    def has_exact_stable_root_anchor(unit: dict[str, Any]) -> bool:
        anchor = unit.get("stable_root_anchor")
        if not isinstance(anchor, dict):
            return False
        members = [
            item
            for item in anchor.get("members") or []
            if isinstance(item, dict)
        ] or [anchor]
        return bool(members) and all(
            (
                str(member.get("tree") or "") == "old"
                or str(member.get("tree") or "").startswith("donor_")
            )
            and normalize_relpath(str(member.get("file") or ""))
            and str(member.get("contract_key") or "")
            and str(member.get("entity_kind") or "")
            for member in members
        )

    def has_use_liveness_diagnostic(unit: dict[str, Any]) -> bool:
        return any(
            str(issue.get("warning_flag") or "").startswith("unused")
            for issue in [
                *list(unit.get("source_issue_samples") or []),
                *list(unit.get("source_issues") or []),
            ]
            if isinstance(issue, dict)
        )

    units = [
        unit
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
    ]
    completed = next(
        (
            unit
            for unit in units
            if str(unit.get("id") or "")
            == str(completed_plan_unit_id)
        ),
        None,
    )
    completed_root_keys = _repair_plan_root_keys(completed or {})
    normalized_changed_files = {
        normalize_relpath(str(path))
        for path in changed_files
        if normalize_relpath(str(path))
    }
    added_contracts = {
        (
            normalize_relpath(str(record.get("file") or "")),
            str(
                record.get("entity_kind")
                or record.get("kind")
                or ""
            ).lower(),
            str(
                record.get("contract_key")
                or record.get("key")
                or ""
            ),
        )
        for record in added_contract_definitions
        if isinstance(record, dict)
    }
    added_contracts = {
        identity for identity in added_contracts if all(identity)
    }
    conflicting_units: list[dict[str, Any]] = []
    stale_location_units: list[dict[str, Any]] = []
    rebindable_location_units: list[dict[str, Any]] = []
    candidate_realized_units: list[dict[str, Any]] = []
    for unit in units:
        if str(unit.get("id") or "") == str(completed_plan_unit_id):
            continue
        shared_roots = sorted(
            completed_root_keys & _repair_plan_root_keys(unit)
        )
        if shared_roots:
            conflicting_units.append(
                {
                    "plan_unit_id": unit.get("id"),
                    "work_unit_id": unit.get("work_unit_id"),
                    "shared_root_keys": shared_roots,
                }
            )
        if added_contracts and has_exact_stable_root_anchor(unit):
            anchor = dict(unit.get("stable_root_anchor") or {})
            members = [
                item
                for item in anchor.get("members") or []
                if isinstance(item, dict)
            ] or [anchor]
            member_contracts = {
                (
                    normalize_relpath(str(member.get("file") or "")),
                    str(member.get("entity_kind") or "").lower(),
                    str(member.get("contract_key") or ""),
                )
                for member in members
            }
            if member_contracts and member_contracts <= added_contracts:
                candidate_realized_units.append(
                    {
                        "plan_unit_id": unit.get("id"),
                        "work_unit_id": unit.get("work_unit_id"),
                        "stable_root_contracts": [
                            {
                                "file": owner_file,
                                "entity_kind": entity_kind,
                                "contract_key": contract_key,
                            }
                            for owner_file, entity_kind, contract_key in sorted(
                                member_contracts
                            )
                        ],
                        "disposition": "pending_round_validation",
                    }
                )
        stale_files = sorted(
            normalized_changed_files
            & {
                str(locus.get("file") or "")
                for locus in _work_unit_diagnostic_loci(
                    unit,
                    old_root=old_root,
                )
                if str(locus.get("file") or "")
            }
        )
        if stale_files:
            record = {
                "plan_unit_id": unit.get("id"),
                "work_unit_id": unit.get("work_unit_id"),
                "stale_diagnostic_files": stale_files,
            }
            if has_use_liveness_diagnostic(unit):
                stale_location_units.append(
                    {
                        **record,
                        "reason": "use_liveness_requires_fresh_validation",
                    }
                )
            elif has_exact_stable_root_anchor(unit):
                rebindable_location_units.append(
                    {
                        **record,
                        "stable_root_anchor": dict(
                            unit.get("stable_root_anchor") or {}
                        ),
                    }
                )
            else:
                stale_location_units.append(record)
    deferred_plan_unit_ids = {
        str(item.get("plan_unit_id") or "")
        for item in [
            *conflicting_units,
            *stale_location_units,
            *candidate_realized_units,
        ]
        if str(item.get("plan_unit_id") or "")
    }
    deferred_work_unit_ids = {
        str(unit.get("work_unit_id") or unit.get("id") or "")
        for unit in units
        if str(unit.get("id") or "") in deferred_plan_unit_ids
    }
    completed_work_unit_id = str(
        (completed or {}).get("work_unit_id")
        or (completed or {}).get("id")
        or ""
    )
    changed_work_unit_ids = {
        *deferred_work_unit_ids,
        *([completed_work_unit_id] if completed_work_unit_id else []),
    }
    changed = True
    while changed:
        changed = False
        for unit in units:
            plan_unit_id = str(unit.get("id") or "")
            if (
                not plan_unit_id
                or plan_unit_id == str(completed_plan_unit_id)
                or plan_unit_id in deferred_plan_unit_ids
            ):
                continue
            dependencies = {
                str(item)
                for item in unit.get("depends_on_work_unit_ids") or []
                if str(item)
            }
            if not dependencies.intersection(changed_work_unit_ids):
                continue
            deferred_plan_unit_ids.add(plan_unit_id)
            work_unit_id = str(
                unit.get("work_unit_id") or plan_unit_id
            )
            if work_unit_id:
                deferred_work_unit_ids.add(work_unit_id)
                changed_work_unit_ids.add(work_unit_id)
            changed = True
    return {
        "requires_validation_replan": bool(deferred_plan_unit_ids),
        "completed_root_keys": sorted(completed_root_keys),
        "applied_changed_files": sorted(normalized_changed_files),
        "conflicting_units": conflicting_units,
        "stale_location_units": stale_location_units,
        "rebindable_location_units": rebindable_location_units,
        "candidate_realized_units": candidate_realized_units,
        "deferred_until_replan_plan_unit_ids": sorted(
            deferred_plan_unit_ids
        ),
        "independent_plan_unit_ids": [
            str(unit.get("id") or "")
            for unit in units
            if str(unit.get("id") or "")
            not in {
                str(completed_plan_unit_id),
                *deferred_plan_unit_ids,
            }
        ],
    }


def _deferred_scope_materialization_outcome(
    scope_expansion: dict[str, Any],
    *,
    result_path: Path,
) -> dict[str, Any]:
    status = str(scope_expansion.get("status") or "invariant_failed")
    return {
        "status": "deferred",
        "reason": f"graph_scope_expansion_{status}",
        "scope_expansion_path": str(result_path.resolve()),
        "scope_expansion": scope_expansion,
    }


def _scope_expansion_changes_source_or_build(
    scope_expansion: dict[str, Any],
) -> bool:
    explicit = scope_expansion.get("source_or_build_changed")
    if isinstance(explicit, bool):
        return explicit
    build_apply = dict(scope_expansion.get("build_closure_apply") or {})
    return bool(
        scope_expansion.get("copied_files")
        or int(build_apply.get("applied_copy_operations") or 0)
        or int(build_apply.get("applied_build_integrations") or 0)
    )


def _scope_change_round_policy(
    scope_expansion: dict[str, Any],
) -> dict[str, Any]:
    """Require fresh validation before any plan observes a changed scope."""

    source_or_build_changed = (
        _scope_expansion_changes_source_or_build(scope_expansion)
    )
    return {
        "schema": "source_change_validation_barrier_v1",
        "source_or_build_changed": source_or_build_changed,
        "force_round_close": source_or_build_changed,
        "validation_timing": (
            "before_repair_plan_refresh"
            if source_or_build_changed
            else "not_required"
        ),
        "stale_plan_policy": (
            "discard_after_validation"
            if source_or_build_changed
            else "unchanged"
        ),
    }


def _validation_observes_repair_postimage(
    validation: dict[str, Any],
) -> bool:
    """Return whether validation reached the changed target build graph."""

    status = str(validation.get("status") or "")
    validation_path = str(validation.get("validation_path") or "").strip()
    if not validation_path:
        return False
    if status == "ready_for_manual_test":
        return True
    if status != "needs_patching":
        return False
    return not _validation_blocks_auto_repair(validation)


def _dependency_ready_scope_checkpoint_plan(
    repair_plan: dict[str, Any],
    *,
    spec: CaseSpec | None = None,
) -> dict[str, Any]:
    """Select graph-proven scope work before consuming a repair round."""

    ready_plan, _pending_plan, selection = (
        _select_validation_batch_repair_plan(repair_plan, spec=spec)
    )
    checkpoint_units = [
        unit
        for unit in ready_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and str((unit.get("repair_decision") or {}).get("mode") or "")
        == "materialize_graph_scope"
        and bool(unit.get("scope_expansion_candidates"))
    ]
    checkpoint_ids = [
        str(unit.get("id") or unit.get("work_unit_id") or "")
        for unit in checkpoint_units
        if str(unit.get("id") or unit.get("work_unit_id") or "")
    ]
    checkpoint_plan = dict(repair_plan)
    checkpoint_plan["work_unit_plans"] = copy.deepcopy(checkpoint_units)
    summary = dict(checkpoint_plan.get("summary") or {})
    summary.update(
        {
            "work_unit_count": len(checkpoint_units),
            "structural_scope_checkpoint": True,
            "dependency_selection": {
                key: copy.deepcopy(value)
                for key, value in selection.items()
                if key != "selected_units"
            },
        }
    )
    checkpoint_plan["summary"] = summary
    all_ids = [
        str(unit.get("id") or unit.get("work_unit_id") or "")
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and str(unit.get("id") or unit.get("work_unit_id") or "")
    ]
    checkpoint_id_set = set(checkpoint_ids)
    return {
        "schema": "dependency_ready_scope_checkpoint_plan_v1",
        "required": bool(checkpoint_ids),
        "work_unit_ids": checkpoint_ids,
        "deferred_work_unit_ids": [
            unit_id for unit_id in all_ids if unit_id not in checkpoint_id_set
        ],
        "repair_plan": checkpoint_plan,
    }


def _scope_checkpoint_reconciliation(
    before_plan: dict[str, Any],
    after_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Show which pre-checkpoint diagnostics survived the fresh build."""

    def identities(plan: dict[str, Any] | None) -> tuple[set[str], set[str]]:
        unit_ids: set[str] = set()
        signatures: set[str] = set()
        for unit in (plan or {}).get("work_unit_plans") or []:
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("work_unit_id") or unit.get("id") or "")
            if unit_id:
                unit_ids.add(unit_id)
            signatures.update(
                str(signature)
                for signature in unit.get("diagnostic_signatures") or []
                if str(signature)
            )
        return unit_ids, signatures

    before_ids, before_signatures = identities(before_plan)
    after_ids, after_signatures = identities(after_plan)
    return {
        "schema": "scope_checkpoint_diagnostic_reconciliation_v1",
        "before_work_unit_ids": sorted(before_ids),
        "after_work_unit_ids": sorted(after_ids),
        "reobserved_work_unit_ids": sorted(before_ids & after_ids),
        "resolved_or_reattributed_work_unit_ids": sorted(
            before_ids - after_ids
        ),
        "new_work_unit_ids": sorted(after_ids - before_ids),
        "before_diagnostic_signature_count": len(before_signatures),
        "after_diagnostic_signature_count": len(after_signatures),
        "reobserved_diagnostic_signature_count": len(
            before_signatures & after_signatures
        ),
        "resolved_diagnostic_signature_count": len(
            before_signatures - after_signatures
        ),
        "new_diagnostic_signature_count": len(
            after_signatures - before_signatures
        ),
    }


def _model_failure_disposition(
    metadata: dict[str, Any],
    *,
    consecutive_failures: int,
    max_consecutive_failures: int,
) -> dict[str, Any]:
    category = str(metadata.get("category") or "")
    if category == "context_window_exceeded":
        return {
            "retryable": False,
            "consecutive_failures": max(1, int(consecutive_failures)),
            "max_consecutive_failures": max(
                1, int(max_consecutive_failures)
            ),
            "action": "repartition_graph_work_unit_before_resume",
            "required_transition": (
                "split_the_selected_unit_only_at_independent_code_graph_"
                "components_and_rebuild_lossless_prompt_evidence"
            ),
            "stop_after_validation": True,
            "reason": "model_context_window_exceeded",
        }
    retryable = bool(metadata.get("retryable"))
    threshold = max(1, int(max_consecutive_failures))
    stop_after_validation = (
        not retryable or consecutive_failures >= threshold
    )
    return {
        "retryable": retryable,
        "consecutive_failures": max(1, int(consecutive_failures)),
        "max_consecutive_failures": threshold,
        "action": (
            "validate_accumulated_patches_and_stop"
            if stop_after_validation
            else "defer_unit_and_continue"
        ),
        "stop_after_validation": stop_after_validation,
        "reason": (
            "non_retryable_model_api_failure"
            if not retryable
            else "consecutive_retryable_model_api_failures"
            if stop_after_validation
            else "isolated_retryable_model_api_failure"
        ),
    }


def _retryable_model_round_terminal(
    unit_outcomes: Iterable[dict[str, Any]],
    *,
    repair_round: int,
    persisted_outcomes: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Preserve retryable provider failures when an empty round closes."""

    failures = [
        dict(outcome)
        for outcome in [*persisted_outcomes, *unit_outcomes]
        if isinstance(outcome, dict)
        and str(outcome.get("reason") or "")
        == "isolated_retryable_model_api_failure"
        and isinstance(outcome.get("model_error"), dict)
        and (outcome.get("model_error") or {}).get("retryable") is True
    ]
    if not failures:
        return {}
    latest = failures[-1]
    return {
        "stop_reason": f"agentic_model_api_unavailable_round_{repair_round}",
        "extra": {
            "model_error": dict(latest.get("model_error") or {}),
            "model_error_path": latest.get("model_error_path"),
            "model_failure_disposition": dict(
                (latest.get("model_error") or {}).get("disposition") or {}
            ),
            "retryable_model_failure_count": len(failures),
            "round_unit_outcomes": failures,
        },
    }


def _repair_plan_unit_ids(repair_plan: dict[str, Any] | None) -> list[str]:
    if not repair_plan:
        return []
    return [
        str(unit.get("id") or unit.get("work_unit_id") or "")
        for unit in repair_plan.get("work_unit_plans") or []
        if isinstance(unit, dict)
        and str(unit.get("id") or unit.get("work_unit_id") or "")
    ]












def auto_migrate(
    spec: CaseSpec,
    *,
    state_root: Path,
    max_repair_rounds: int,
    allow_escalation: bool,
    model: str | None,
    max_completion_tokens: int,
    max_build_config_preflight_attempts: int = 2,
    max_semantic_repair_rounds: int = 3,
) -> dict:
    max_repair_rounds = max(0, int(max_repair_rounds))
    max_semantic_repair_rounds = max(
        0,
        int(max_semantic_repair_rounds),
    )
    spec = load_graph_scope_expansions(spec, state_root=state_root)
    stages: list[dict] = []
    started_at = utc_now()
    run_id = _new_run_id(spec)
    _write_migration_progress(
        spec=spec,
        state_root=state_root,
        run_id=run_id,
        started_at=started_at,
        stages=stages,
        status="running",
        current_stage="start",
    )

    def record(stage: str, result: dict, *, rounds: int = 0) -> None:
        _record_migration_stage(
            spec=spec,
            state_root=state_root,
            run_id=run_id,
            started_at=started_at,
            stages=stages,
            stage=stage,
            result=result,
            repair_rounds_completed=rounds,
        )

    def announce(stage: str, *, rounds: int = 0) -> None:
        round_text = f" round={rounds}" if rounds else ""
        _emit_progress(f"starting stage: {stage}{round_text}")
        _write_migration_progress(
            spec=spec,
            state_root=state_root,
            run_id=run_id,
            started_at=started_at,
            stages=stages,
            status="running",
            current_stage=stage,
            repair_rounds_completed=rounds,
        )

    def finish(
        *,
        status: str,
        stop_reason: str,
        repair_rounds_completed: int,
        extra: dict[str, Any] | None = None,
    ) -> dict:
        final_status = "completed" if stop_reason == "validated" else "stopped"
        final_extra = dict(extra or {})
        semantic_summary = _effective_semantic_summary(
            state_root,
            case_id=spec.case_id,
        )
        final_extra.setdefault("semantic_summary", semantic_summary)
        persisted_state = load_or_init_state(spec, state_root)
        final_compile_validated = False
        latest_validation_path = str(
            persisted_state.last_validation_path or ""
        )
        if latest_validation_path and Path(latest_validation_path).is_file():
            try:
                latest_validation = json.loads(
                    Path(latest_validation_path).read_text(encoding="utf-8")
                )
                final_compile_validated = bool(
                    latest_validation.get("overall_ok")
                )
            except (OSError, ValueError):
                final_compile_validated = False
        effective_final_compile_validated = bool(
            final_extra.get(
                "final_compile_validated",
                final_compile_validated,
            )
        )
        status = _coherent_terminal_case_status(
            requested_status=status,
            persisted_status=str(persisted_state.status or ""),
            final_compile_validated=effective_final_compile_validated,
        )
        round_counts = validated_repair_round_counts(state_root)
        semantic_rounds_completed = max(
            int(
                round_counts[
                    "semantic_repair_rounds_completed"
                ]
            ),
            int(
                semantic_summary.get(
                    "semantic_rounds_completed"
                )
                or 0
            ),
        )
        total_repair_rounds = max(
            int(repair_rounds_completed),
            int(round_counts["total_repair_rounds_completed"]),
            int(round_counts["compile_repair_rounds_completed"])
            + semantic_rounds_completed,
        )
        semantic_state = load_semantic_closure_state(
            state_root,
            case_id=spec.case_id,
        )
        compile_handoff_recorded = bool(
            isinstance(semantic_state, dict)
            and isinstance(
                semantic_state.get("compile_valid_semantic_handoff"),
                dict,
            )
            and semantic_state["compile_valid_semantic_handoff"].get(
                "status"
            )
            == "complete"
        )
        compile_repair_validated = bool(
            final_extra.get(
                "compile_repair_validated",
                compile_handoff_recorded
                or effective_final_compile_validated,
            )
        )
        semantic_complete = (
            semantic_summary.get("closure_status")
            == "semantic_migration_closed"
        )
        effective_semantic_complete = bool(
            final_extra.get("semantic_complete", semantic_complete)
        )
        final_extra.setdefault(
            "final_compile_validated",
            effective_final_compile_validated,
        )
        final_extra.setdefault(
            "compile_repair_validated",
            compile_repair_validated,
        )
        final_extra.setdefault(
            "compile_repair_rounds_completed",
            round_counts["compile_repair_rounds_completed"],
        )
        final_extra.setdefault(
            "semantic_repair_rounds_completed",
            semantic_rounds_completed,
        )
        final_extra.setdefault(
            "total_repair_rounds_completed",
            total_repair_rounds,
        )
        final_extra.setdefault(
            "compile_repair_round_limit",
            max_repair_rounds,
        )
        final_extra.setdefault(
            "semantic_repair_round_limit",
            max_semantic_repair_rounds,
        )
        final_extra["semantic_complete"] = (
            effective_semantic_complete
        )
        final_extra["migration_validation_tier"] = (
            "compile_valid_semantic_complete"
            if (
                effective_final_compile_validated
                and effective_semantic_complete
            )
            else "compile_valid_semantic_incomplete"
            if effective_final_compile_validated
            else "compile_not_validated"
        )
        progress_path = _write_migration_progress(
            spec=spec,
            state_root=state_root,
            run_id=run_id,
            started_at=started_at,
            stages=stages,
            status=final_status,
            current_stage=stages[-1]["stage"] if stages else "start",
            stop_reason=stop_reason,
            repair_rounds_completed=total_repair_rounds,
            extra=final_extra,
        )
        _emit_progress(
            f"finished migration: status={status} stop_reason={stop_reason} "
            f"compile_repair_rounds_completed="
            f"{round_counts['compile_repair_rounds_completed']} "
            f"semantic_repair_rounds_completed="
            f"{round_counts['semantic_repair_rounds_completed']}"
        )
        return {
            **final_extra,
            "case_id": spec.case_id,
            "status": status,
            "stop_reason": stop_reason,
            "repair_rounds_completed": total_repair_rounds,
            "progress_path": progress_path,
            "stage_event_log_path": str(
                progress_stage_log_file_for(state_root).resolve()
            ),
            "stages": list(stages),
        }

    def ensure_model_ready(
        *,
        repair_rounds_completed: int = 0,
    ) -> dict | None:
        if not has_model_config():
            final_state = show_status(spec, state_root=state_root)
            return finish(
                status=final_state["status"],
                stop_reason="model_config_missing",
                repair_rounds_completed=repair_rounds_completed,
                extra={"error": "OPENAI_API_KEY is not set; cannot run model prompts"},
            )
        return None

    def _run_agentic_model_step(prompt_text: str) -> tuple[str, dict[str, Any]]:
        model_call = gpt_api_with_metadata(
            prompt_text,
            model=model,
            max_completion_tokens=max_completion_tokens,
            max_retries=None,
        )
        response_text = str(model_call.pop("text", "") or "")
        model_call["prompt_size_bytes"] = len(prompt_text.encode("utf-8"))
        return response_text, model_call

    def initialize_isolated_semantic_ledger(
        cycle_dir: Path,
    ) -> dict[str, Any]:
        existing = load_semantic_closure_state(
            state_root,
            case_id=spec.case_id,
        )
        if existing is not None:
            return {
                "status": "already_initialized",
                "summary": isolated_semantic_closure_summary(
                    existing
                ),
            }
        output_dir = (
            cycle_dir.resolve() / "initial_compatibility"
        )
        analysis_path = output_dir / "analysis.json"
        plan_path = (
            output_dir / "initial_compatibility_plan.json"
        )
        if not analysis_path.is_file() or not plan_path.is_file():
            return {
                "status": "artifacts_unavailable",
                "analysis_path": str(analysis_path),
                "plan_path": str(plan_path),
            }
        try:
            analysis = json.loads(
                analysis_path.read_text(encoding="utf-8")
            )
            plan = load_repair_plan(
                plan_path,
                state_root=state_root,
            )
            return initialize_semantic_closure_from_artifacts(
                state_root=state_root,
                case_id=spec.case_id,
                analysis=analysis,
                plan=plan,
                analysis_path=analysis_path,
                plan_path=plan_path,
            )
        except (OSError, ValueError, KeyError) as exc:
            return {
                "status": "initialization_failed",
                "error": f"{exc.__class__.__name__}: {exc}",
                "analysis_path": str(analysis_path),
                "plan_path": str(plan_path),
            }

    def initial_compatibility_cycle_from_state() -> Path | None:
        """Resolve the sole B0 artifact owner from durable cycle state."""

        persisted = load_or_init_state(spec, state_root)
        raw_plan_path = str(
            getattr(persisted, "last_transplant_plan_path", None) or ""
        ).strip()
        if not raw_plan_path:
            return None
        plan_path = Path(raw_plan_path).resolve()
        cycle_dir = plan_path.parent
        output_dir = cycle_dir / "initial_compatibility"
        if not (
            plan_path.is_file()
            and (output_dir / "analysis.json").is_file()
            and (output_dir / "initial_compatibility_plan.json").is_file()
        ):
            return None
        return cycle_dir

    def canonical_semantic_validation(
        request: dict[str, Any],
    ) -> dict[str, Any]:
        ledger = load_semantic_closure_state(
            state_root,
            case_id=spec.case_id,
        )
        if ledger is None:
            raise ValueError(
                "semantic validation requires its persistent ledger"
            )
        patches = [
            item
            for item in request.get("patches") or []
            if isinstance(item, dict)
        ]
        batch_gap_ids = {
            str(gap_id)
            for patch in patches
            for key in ("gap_ids", "affected_gap_ids")
            for gap_id in patch.get(key) or []
            if str(gap_id)
        }
        validation_sources = {
            normalize_relpath(str(path))
            for patch in patches
            for path in patch.get("touched_files") or []
            if Path(str(path)).suffix.lower()
            in {".c", ".cc", ".cpp", ".cxx", ".s"}
        }
        for gap_id, gap in ledger["gaps"].items():
            if str(gap_id) not in batch_gap_ids:
                continue
            for binding in gap.get("owner_bindings") or []:
                if not isinstance(binding, dict):
                    continue
                source = normalize_relpath(str(binding.get("file") or ""))
                if (
                    Path(source).suffix.lower()
                    in {".c", ".cc", ".cpp", ".cxx", ".s"}
                    and spec.resolve_old(source).is_file()
                ):
                    validation_sources.add(source)

        kbuild_closure = KbuildIndex.from_directory(
            spec.old_root_path,
            spec.driver_root,
        ).assignment_closure_for_sources(sorted(validation_sources))
        semantic_modules = [
            str(item)
            for item in kbuild_closure.get("expected_modules") or []
        ]
        compile_spec = replace(
            spec,
            load_phases=[],
            require_load_validation=False,
            expected_modules=list(
                dict.fromkeys(
                    [*spec.expected_modules, *semantic_modules]
                )
            ),
        )
        validation = run_post_validation_cycle(
            compile_spec,
            state_root=state_root,
            execution_phase=EXECUTION_PHASE_SEMANTIC_CLOSURE,
        )
        return {
            **validation,
            "semantic_validation_scope": {
                "schema": "semantic_validation_kbuild_scope_v1",
                "batch_id": str(request.get("batch_id") or ""),
                "gap_ids": sorted(batch_gap_ids),
                "source_files": sorted(validation_sources),
                "expected_modules": list(compile_spec.expected_modules),
                "added_expected_modules": [
                    module
                    for module in semantic_modules
                    if module not in spec.expected_modules
                ],
                "kbuild_assignment_ids": list(
                    kbuild_closure.get("assignment_ids") or []
                ),
            },
        }

    def pending_semantic_resume() -> dict[str, Any] | None:
        return semantic_closure_resume_context(
            state_root,
            case_id=spec.case_id,
        )

    def finish_after_compile_validation(
        validation: dict[str, Any],
        *,
        completed_rounds: int,
        initial_cycle_dir: Path | None = None,
        completion_extra: dict[str, Any] | None = None,
        semantic_handoff_already_recorded: bool = False,
    ) -> dict[str, Any]:
        if validation.get("status") != "ready_for_manual_test":
            raise ValueError(
                "isolated semantic closure requires compile validation"
            )
        completion_context = dict(completion_extra or {})
        completion_context.setdefault("compile_repair_validated", True)

        def semantic_extra(
            values: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                **completion_context,
                **values,
            }

        ledger = load_semantic_closure_state(
            state_root,
            case_id=spec.case_id,
        )
        initialization: dict[str, Any] | None = None
        source_cycle_dir = (
            initial_cycle_dir or initial_compatibility_cycle_from_state()
        )
        if ledger is None and source_cycle_dir is not None:
            initialization = initialize_isolated_semantic_ledger(
                source_cycle_dir
            )
            record(
                "isolated_semantic_ledger_initialized",
                initialization,
                rounds=completed_rounds,
            )
            ledger = load_semantic_closure_state(
                state_root,
                case_id=spec.case_id,
            )
        if ledger is None:
            initialization_status = str(
                (initialization or {}).get("status") or ""
            )
            return finish(
                status=str(validation["status"]),
                stop_reason=(
                    "semantic_source_analysis_incomplete"
                    if initialization_status
                    == "source_analysis_incomplete"
                    else "semantic_closure_artifacts_unavailable"
                ),
                repair_rounds_completed=completed_rounds,
                extra=semantic_extra({
                    "final_compile_validated": True,
                    "semantic_complete": False,
                    "semantic_closure": {
                        "status": (
                            initialization_status
                            or "not_initialized"
                        ),
                        "initialization": initialization or {},
                    },
                }),
            )

        if source_cycle_dir is not None:
            try:
                outcome_snapshot = snapshot_initial_compatibility_outcomes(
                    state_root=state_root,
                    case_id=spec.case_id,
                    cycle_dir=source_cycle_dir,
                )
            except (OSError, ValueError, KeyError) as exc:
                outcome_snapshot = {
                    "status": "snapshot_failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            record(
                "initial_compatibility_outcomes_snapshotted",
                outcome_snapshot,
                rounds=completed_rounds,
            )

        checkpoint_ref = str(validation.get("validation_path") or "").strip()
        if not checkpoint_ref:
            return finish(
                status=str(validation["status"]),
                stop_reason="semantic_compile_valid_checkpoint_missing",
                repair_rounds_completed=completed_rounds,
                extra=semantic_extra({
                    "final_compile_validated": True,
                    "semantic_complete": False,
                }),
            )
        announce(
            "isolated_semantic_compile_valid_handoff",
            rounds=completed_rounds,
        )
        if semantic_handoff_already_recorded:
            recorded_handoff = ledger.get(
                "compile_valid_semantic_handoff"
            )
            handoff = (
                copy.deepcopy(recorded_handoff)
                if isinstance(recorded_handoff, dict)
                else {
                    "status": "failed",
                    "error": (
                        "the semantic batch validation was recorded, but "
                        "the ledger has no compile-valid handoff"
                    ),
                }
            )
        else:
            try:
                handoff = evaluate_compile_valid_semantic_handoff(
                    spec,
                    state_root=state_root,
                    checkpoint_ref=checkpoint_ref,
                )
            except Exception as exc:
                handoff = {
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
        record(
            "isolated_semantic_compile_valid_handoff",
            handoff,
            rounds=completed_rounds,
        )
        if handoff.get("status") == "failed":
            return finish(
                status=str(validation["status"]),
                stop_reason="semantic_handoff_failed",
                repair_rounds_completed=completed_rounds,
                extra=semantic_extra({
                    "final_compile_validated": True,
                    "semantic_complete": False,
                    "semantic_handoff": handoff,
                }),
            )
        if handoff.get("status") == "stale_handoff":
            return finish(
                status=str(validation["status"]),
                stop_reason="semantic_handoff_revision_stale",
                repair_rounds_completed=completed_rounds,
                extra=semantic_extra({
                    "final_compile_validated": True,
                    "semantic_complete": False,
                    "semantic_handoff": handoff,
                    "semantic_closure": {
                        "status": "fresh_ledger_required",
                        "expected_evaluator_revision": (
                            COMPILE_VALID_HANDOFF_EVALUATOR_REVISION
                        ),
                    },
                }),
            )
        handoff_summary = (
            isolated_semantic_closure_summary(ledger)
            if semantic_handoff_already_recorded
            else dict(handoff.get("summary") or {})
        )
        if handoff_summary.get("static_semantic_closed"):
            return finish(
                status=str(validation["status"]),
                stop_reason="validated",
                repair_rounds_completed=completed_rounds,
                extra=semantic_extra({
                    "semantic_handoff": handoff,
                    "semantic_closure": {
                        "status": "semantic_closed",
                        "summary": handoff_summary,
                    },
                }),
            )
        if max_semantic_repair_rounds <= 0:
            return finish(
                status=str(validation["status"]),
                stop_reason="semantic_closure_budget_disabled",
                repair_rounds_completed=completed_rounds,
                extra=semantic_extra({
                    "final_compile_validated": True,
                    "semantic_complete": False,
                    "semantic_handoff": handoff,
                    "semantic_closure": {
                        "status": "budget_disabled",
                        "round_limit": 0,
                        "summary": handoff_summary,
                    },
                }),
            )
        model_failure = ensure_model_ready(
            repair_rounds_completed=completed_rounds,
        )
        if model_failure is not None:
            return model_failure

        announce(
            "isolated_semantic_closure",
            rounds=completed_rounds,
        )
        runtime = DefaultSemanticClosureRuntime(
            spec,
            state_root=state_root,
            model=model,
            max_completion_tokens=max_completion_tokens,
            validation_call=canonical_semantic_validation,
            progress=_emit_progress,
        )
        try:
            closure = run_semantic_closure_controller(
                spec,
                state_root=state_root,
                runtime=runtime,
                max_rounds=max_semantic_repair_rounds,
            )
        except Exception as exc:
            closure = {
                "status": "controller_failed",
                "compile_validated": False,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        record(
            "isolated_semantic_closure",
            closure,
            rounds=completed_rounds,
        )
        closure_summary = closure.get("summary") or {}
        semantic_rounds = int(
            closure_summary["semantic_rounds_completed"]
            if "semantic_rounds_completed" in closure_summary
            else len(closure.get("rounds") or [])
        )
        closure_compile_validated = bool(
            closure.get("compile_validated", False)
        )
        if closure.get("status") == "compile_recovery_required":
            return continue_semantic_compile_recovery(
                dict(closure.get("validation") or {}),
                completed_rounds=completed_rounds,
                semantic_handoff=handoff,
                semantic_closure=closure,
            )
        if closure.get("status") == "semantic_closed":
            return finish(
                status=str(validation["status"]),
                stop_reason="validated",
                repair_rounds_completed=(
                    completed_rounds + semantic_rounds
                ),
                extra=semantic_extra({
                    "semantic_handoff": handoff,
                    "semantic_closure": closure,
                    "semantic_repair_rounds_completed": (
                        semantic_rounds
                    ),
                }),
            )
        if closure.get("status") == "semantic_runtime_unavailable":
            runtime_failure = dict(closure.get("runtime_failure") or {})
            return finish(
                status=str(validation["status"]),
                stop_reason="semantic_model_runtime_unavailable",
                repair_rounds_completed=(
                    completed_rounds + semantic_rounds
                ),
                extra=semantic_extra({
                    "final_compile_validated": (
                        closure_compile_validated
                    ),
                    "semantic_complete": False,
                    "semantic_handoff": handoff,
                    "semantic_closure": closure,
                    "semantic_repair_rounds_completed": semantic_rounds,
                    "infrastructure_failure": {
                        "stage": "semantic_closure",
                        "kind": "model_runtime_unavailable",
                        "retryable": bool(
                            runtime_failure.get("retryable")
                        ),
                        "runtime_failure": runtime_failure,
                        "resume_route": dict(
                            closure.get("resume_route") or {}
                        ),
                    },
                }),
            )
        return finish(
            status=str(validation["status"]),
            stop_reason="semantic_closure_incomplete",
            repair_rounds_completed=(
                completed_rounds + semantic_rounds
            ),
            extra=semantic_extra({
                "final_compile_validated": (
                    closure_compile_validated
                ),
                "semantic_complete": False,
                "semantic_handoff": handoff,
                "semantic_closure": closure,
                "semantic_repair_rounds_completed": semantic_rounds,
            }),
        )

    def run_initial_compatibility_before_compile(cycle_dir: Path) -> dict[str, Any] | None:
        initial_copy_set = _initial_copy_set_from_cycle(cycle_dir)
        announce("initial_compile_context_materialization")
        try:
            compile_context = materialize_transplanted_compile_context(spec.to_dict())
        except Exception as exc:
            compile_context = {
                "status": "failed",
                "stop_reason": "compile_context_materialization_exception",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        record("initial_compile_context_materialization", compile_context)
        if compile_context.get("status") == "failed":
            final_state = show_status(spec, state_root=state_root)
            return finish(
                status=final_state["status"],
                stop_reason=str(
                    compile_context.get("stop_reason")
                    or "compile_context_materialization_failed"
                ),
                repair_rounds_completed=0,
                extra={
                    "initial_compile_context_materialization": compile_context,
                    "source_repair_blocked": True,
                },
            )
        announce("initial_compatibility_analysis")
        try:
            prepared = prepare_initial_compatibility(
                spec,
                state_root=state_root,
                cycle_dir=cycle_dir,
                initial_copy_set=initial_copy_set,
            )
        except Exception as exc:
            incomplete = {
                "status": "analysis_incomplete",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "initial_copy_set": initial_copy_set,
                "next_stage": "continue_to_first_target_compile",
            }
            record("initial_compatibility_analysis", incomplete)
            _emit_progress(
                "initial compatibility analysis incomplete; continuing to "
                "the first target compile: "
                f"{exc.__class__.__name__}: {str(exc)[:240]}"
            )
            return None
        record("initial_compatibility_analysis", prepared)

        if not bool(
            (prepared.get("analysis_summary") or {}).get(
                "preparation_complete",
                True,
            )
        ):
            incomplete = {
                "status": "incomplete",
                "reason": (
                    "the initial compatibility graph is incomplete; available "
                    "artifacts are retained and compiler repair will provide "
                    "the next source evidence"
                ),
                "preparation_blockers": list(
                    prepared.get("preparation_blockers") or []
                ),
                "preparation_blocker_count": int(
                    (prepared.get("analysis_summary") or {}).get(
                        "preparation_blocker_count"
                    )
                    or 0
                ),
                "source_repair_blocked": False,
                "next_stage": "continue_to_first_target_compile",
            }
            record(
                "initial_compatibility_analysis_incomplete",
                incomplete,
            )

        queue_before_execution = initial_compatibility_queue_status(
            cycle_dir,
        )
        recovered_post_materialization_validations = list(
            queue_before_execution.get(
                "post_materialization_validation_records"
            )
            or []
        )
        if recovered_post_materialization_validations:
            record(
                "initial_post_materialization_validation_recovery",
                {
                    "status": "recorded_without_semantic_control",
                    "validation_records": (
                        recovered_post_materialization_validations
                    ),
                },
            )
        pending_before_execution = int(
            queue_before_execution.get("pending_work_unit_count") or 0
        )
        if (
            queue_before_execution.get("validation_required")
            and pending_before_execution <= 0
        ):
            integration_batch = (
                finalize_initial_static_integration_batch(
                    spec,
                    cycle_dir=cycle_dir,
                )
            )
            record(
                "initial_static_integration_batch_recovery",
                integration_batch,
            )
            post_materialization_validations = list(
                integration_batch.get("validation_records") or []
            )
            if post_materialization_validations:
                record(
                    "initial_post_materialization_validation_recovery",
                    {
                        "status": "recorded_without_semantic_control",
                        "validation_records": (
                            post_materialization_validations
                        ),
                    },
                )
            queue_before_execution = (
                initial_compatibility_queue_status(cycle_dir)
            )
            record(
                "initial_compatibility_pending_validation_recovery",
                {
                    "status": "compile_required_before_model",
                    "queue": queue_before_execution,
                },
            )
            return None
        if (
            int(queue_before_execution.get("queue_work_unit_count") or 0) > 0
            and pending_before_execution <= 0
        ):
            record(
                "initial_compatibility_queue_complete",
                queue_before_execution,
            )
            return None

        plan_summary = dict(prepared.get("plan_summary") or {})
        llm_unit_count = int(plan_summary.get("llm_unit_count") or 0)
        deterministic_unit_count = int(
            plan_summary.get("deterministic_unit_count") or 0
        )
        validation_only_unit_count = int(
            plan_summary.get(
                "post_materialization_validation_unit_count"
            )
            or 0
        )
        actionable_unit_count = llm_unit_count + deterministic_unit_count
        if actionable_unit_count <= 0:
            if (
                actionable_unit_count <= 0
                and validation_only_unit_count > 0
            ):
                integration_batch = (
                    finalize_initial_static_integration_batch(
                        spec,
                        cycle_dir=cycle_dir,
                    )
                )
                record(
                    "initial_static_integration_batch",
                    integration_batch,
                )
                post_materialization_validations = list(
                    integration_batch.get("validation_records") or []
                )
                if post_materialization_validations:
                    record(
                        "initial_post_materialization_validation",
                        {
                            "status": (
                                "recorded_without_semantic_control"
                            ),
                            "validation_records": (
                                post_materialization_validations
                            ),
                        },
                    )
                queue_before_execution = (
                    initial_compatibility_queue_status(cycle_dir)
                )
            record(
                "initial_compatibility_llm",
                {
                    "status": "skipped",
                    "reason": "no_actionable_initial_units",
                    "llm_unit_count": llm_unit_count,
                    "deterministic_unit_count": deterministic_unit_count,
                    "post_materialization_validation_unit_count": (
                        validation_only_unit_count
                    ),
                },
            )
            compile_gate = _initial_queue_compile_gate(
                queue_before_execution,
                plan_summary,
            )
            if compile_gate.get("status") == "compile_required":
                record(
                    "initial_compatibility_analysis_compile_boundary",
                    compile_gate,
                )
                return None
            if compile_gate.get("status") == "ready":
                return None
            record("initial_compatibility_compile_gate_blocked", compile_gate)
            final_state = show_status(spec, state_root=state_root)
            return finish(
                status=final_state["status"],
                stop_reason="initial_compatibility_queue_incomplete",
                repair_rounds_completed=0,
                extra={
                    "initial_compatibility_compile_gate": compile_gate,
                    "source_repair_blocked": True,
                    "resumable_initial_compatibility": True,
                },
            )

        if llm_unit_count > 0:
            model_failure = ensure_model_ready()
            if model_failure is not None:
                return model_failure
        announce("initial_compatibility_llm")
        try:
            execution = execute_initial_llm_units(
                spec,
                state_root=state_root,
                cycle_dir=cycle_dir,
                model_runner=_run_agentic_model_step,
                allow_escalation=allow_escalation,
            )
        except Exception as exc:
            error_metadata = model_api_error_metadata(exc)
            error_category = str(
                error_metadata.get("category") or "unexpected_runtime_error"
            )
            if bool(error_metadata.get("retryable")):
                stop_reason = (
                    "initial_compatibility_model_api_retry_exhausted"
                )
            elif error_category != "unexpected_runtime_error":
                stop_reason = "initial_compatibility_model_api_failed"
            else:
                stop_reason = "initial_compatibility_execution_failed"
            failure = {
                "status": "stopped_before_compile",
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "model_api_error": error_metadata,
                "queue_preserved_for_resume": True,
                "next_stage": "resume_initial_compatibility",
                **_initial_api_failure_queue_evidence(cycle_dir),
            }
            record("initial_compatibility_llm", failure)
            final_state = show_status(spec, state_root=state_root)
            return finish(
                status=final_state["status"],
                stop_reason=stop_reason,
                repair_rounds_completed=0,
                extra={
                    "initial_compatibility_llm": failure,
                    "source_repair_blocked": True,
                    "resumable_initial_compatibility": True,
                },
            )
        record("initial_compatibility_llm", execution)
        post_materialization_validations = list(
            execution.get(
                "post_materialization_validation_records"
            )
            or []
        )
        if post_materialization_validations:
            record(
                "initial_post_materialization_validation",
                {
                    "status": "recorded_without_semantic_control",
                    "validation_records": (
                        post_materialization_validations
                    ),
                },
            )
        queue_after_execution = dict(execution.get("queue") or {})
        remaining_initial_units = int(
            queue_after_execution.get("pending_work_unit_count")
            or execution.get("pending_work_unit_count")
            or 0
        )
        if remaining_initial_units > 0:
            compile_gate = _initial_queue_compile_gate(
                queue_after_execution,
                plan_summary,
            )
            if compile_gate.get("status") == "compile_required":
                record(
                    "initial_compatibility_analysis_compile_boundary",
                    compile_gate,
                )
                return None
            incomplete = {
                "status": "stopped_before_compile",
                "reason": (
                    "initial compatibility execution returned with pending "
                    "work units"
                ),
                "pending_work_unit_count": remaining_initial_units,
                "compile_gate": compile_gate,
                "queue_preserved_for_resume": True,
                "next_stage": "resume_initial_compatibility",
            }
            record(
                "initial_compatibility_execution_incomplete",
                incomplete,
            )
            final_state = show_status(spec, state_root=state_root)
            return finish(
                status=final_state["status"],
                stop_reason="initial_compatibility_queue_incomplete",
                repair_rounds_completed=0,
                extra={
                    "initial_compatibility_llm": incomplete,
                    "source_repair_blocked": True,
                    "resumable_initial_compatibility": True,
                },
            )
        run_exact_export_contract_closure(
            cycle_dir=cycle_dir,
            rounds=0,
        )
        return None

    def finalize_initial_compatibility_after_compile(
        *,
        initial_cycle_dir: Path,
        current_validation: dict[str, Any],
        completed_rounds: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if _build_prep_route_summary(current_validation).get(
            "needs_build_prep"
        ):
            build_prep_route = _reroute_build_prep_if_needed(
                current_validation,
                repair_rounds_completed=completed_rounds,
            )
            current_validation = dict(
                build_prep_route.get("validation") or current_validation
            )
            if build_prep_route.get("status") == "failed":
                final_state = show_status(spec, state_root=state_root)
                return (
                    finish(
                        status=final_state["status"],
                        stop_reason=str(
                            build_prep_route.get("stop_reason")
                            or "build_config_preflight_failed"
                        ),
                        repair_rounds_completed=completed_rounds,
                        extra=dict(build_prep_route.get("extra") or {}),
                    ),
                    current_validation,
                )
        if _validation_blocks_auto_repair(current_validation):
            return None, current_validation

        handoff = finalize_initial_compatibility_pass(
            cycle_dir=initial_cycle_dir,
            validation=current_validation,
        )
        record(
            "initial_compatibility_pass_finalized",
            handoff,
            rounds=completed_rounds,
        )
        return None, current_validation

    def pending_initial_compatibility_cycle() -> Path | None:
        cycle_dir = initial_compatibility_cycle_from_state()
        if cycle_dir is None:
            return None
        queue = initial_compatibility_queue_status(cycle_dir)
        if queue.get("validation_required") or int(
            queue.get("pending_work_unit_count") or 0
        ) > 0:
            return cycle_dir
        return None

    def run_exact_export_contract_closure(
        *,
        cycle_dir: Path,
        rounds: int,
    ) -> dict[str, Any]:
        source_cycle_dir = initial_compatibility_cycle_from_state()
        plan_path = (
            source_cycle_dir
            / "initial_compatibility"
            / "initial_compatibility_plan.json"
            if source_cycle_dir is not None
            else None
        )
        if plan_path is None or not plan_path.is_file():
            return {
                "schema": "exact_cross_module_export_closure_v1",
                "status": "not_available",
                "reason": "initial compatibility plan is unavailable",
                "candidate_count": 0,
                "applied_count": 0,
                "validation_targets": [],
            }
        plan = load_repair_plan(plan_path, state_root=state_root)
        closure = materialize_ready_exact_exports(
            spec,
            state_root=state_root,
            cycle_dir=cycle_dir,
            plan=plan,
            allow_escalation=allow_escalation,
        )
        closure["source_plan_path"] = str(plan_path.resolve())
        closure_path = cycle_dir / "exact_export_contract_closure.json"
        write_json(closure_path, closure)
        closure["artifact_path"] = str(closure_path.resolve())
        record(
            "exact_export_contract_closure",
            closure,
            rounds=rounds,
        )
        return closure

    def refresh_compile_context_after_scope_expansion(
        *,
        repair_rounds_completed: int,
        scope_expansion: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal spec
        kconfig_files = [
            normalize_relpath(str(path))
            for path in scope_expansion.get("build_integration_files") or []
            if str(path).strip()
            and Path(str(path)).name.startswith("Kconfig")
        ]
        if kconfig_files:
            event_id = str(scope_expansion.get("event_id") or "scope-event")
            kconfig_artifacts_dir = ensure_dir(
                state_root
                / "scope_compile_context"
                / slugify(event_id)
                / "kconfig"
            )
            try:
                kconfig_sync = sync_build_kconfig(
                    spec,
                    artifacts_dir=kconfig_artifacts_dir,
                    force=True,
                )
            except Exception as exc:
                kconfig_sync = {
                    "name": "kconfig-sync",
                    "ok": False,
                    "returncode": 1,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            record(
                "scope_expansion_kconfig_sync",
                {
                    **dict(kconfig_sync or {}),
                    "scope_expansion_event_id": scope_expansion.get(
                        "event_id"
                    ),
                    "changed_kconfig_files": kconfig_files,
                },
                rounds=repair_rounds_completed,
            )
            if not kconfig_sync or not kconfig_sync.get("ok"):
                return {
                    "status": "failed",
                    "stop_reason": "scope_expansion_kconfig_sync_failed",
                    "extra": {
                        "scope_expansion": scope_expansion,
                        "kconfig_sync": kconfig_sync,
                        "source_repair_blocked": True,
                        "candidate_changes_preserved": True,
                        "pending_scope_event_id": scope_expansion.get(
                            "event_id"
                        ),
                    },
                }
        announce(
            "scope_expansion_compile_context_materialization",
            rounds=repair_rounds_completed,
        )
        try:
            compile_context = materialize_transplanted_compile_context(
                spec.to_dict()
            )
        except Exception as exc:
            compile_context = {
                "status": "failed",
                "stop_reason": "compile_context_materialization_exception",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        record(
            "scope_expansion_compile_context_materialization",
            {
                **compile_context,
                "scope_expansion_transaction_dir": scope_expansion.get(
                    "transaction_dir"
                ),
                "scope_expansion_build_integration_files": list(
                    scope_expansion.get("build_integration_files") or []
                ),
            },
            rounds=repair_rounds_completed,
        )
        if compile_context.get("status") != "failed":
            try:
                scope_commit = commit_graph_scope(
                    spec,
                    state_root=state_root,
                    scope_expansion=scope_expansion,
                    compile_context=compile_context,
                )
            except (OSError, ValueError, KeyError) as exc:
                scope_commit = {
                    "schema": "graph_scope_commit_v1",
                    "status": "failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            record(
                "scope_expansion_compile_context_commit",
                scope_commit,
                rounds=repair_rounds_completed,
            )
            if scope_commit.get("status") == "committed":
                return {
                    "status": "committed",
                    "compile_context": compile_context,
                    "scope_commit": scope_commit,
                }
            compile_context = {
                **compile_context,
                "status": "failed",
                "stop_reason": "scope_expansion_state_commit_failed",
                "scope_commit": scope_commit,
            }
        return {
            "status": "failed",
            "stop_reason": str(
                compile_context.get("stop_reason")
                or "scope_expansion_compile_context_materialization_failed"
            ),
            "extra": {
                "scope_expansion": scope_expansion,
                "compile_context": compile_context,
                "source_repair_blocked": True,
                "candidate_changes_preserved": True,
                "pending_scope_event_id": scope_expansion.get(
                    "event_id"
                ),
            },
        }

    build_prep_reroute_attempts = 0

    def _full_validation_for_routing(current_validation: dict) -> dict[str, Any]:
        if current_validation.get("phases"):
            return current_validation
        validation_path = str(current_validation.get("validation_path") or "").strip()
        if validation_path:
            try:
                return json.loads(Path(validation_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return current_validation
        return current_validation

    def _build_prep_route_summary(current_validation: dict) -> dict[str, Any]:
        return build_prep_stage_decision(
            _full_validation_for_routing(current_validation),
            validation_status=str(current_validation.get("status") or ""),
        )

    def _validation_can_enter_repair_loop(current_validation: dict) -> bool:
        if current_validation.get("status") == "needs_build_preparation":
            return True
        if current_validation.get("status") == "needs_patching":
            return True
        if current_validation.get("status") == "patch_apply_incomplete":
            return bool(_build_prep_route_summary(current_validation).get("needs_build_prep"))
        return False

    def _reroute_build_prep_if_needed(
        current_validation: dict,
        *,
        repair_rounds_completed: int,
        execution_phase: str = EXECUTION_PHASE_COMPILER_REPAIR,
    ) -> dict[str, Any]:
        nonlocal build_prep_reroute_attempts
        active_execution_phase = validate_execution_phase(execution_phase)
        route_summary = _build_prep_route_summary(current_validation)
        if not route_summary.get("needs_build_prep"):
            return {
                "status": "not_required",
                "validation": current_validation,
            }
        max_reroutes = max(1, int(max_build_config_preflight_attempts)) + 1
        if build_prep_reroute_attempts >= max_reroutes:
            return {
                "status": "failed",
                "stop_reason": "build_config_preflight_reroute_limit_reached",
                "validation": current_validation,
                "extra": {
                    "diagnostic_routes": route_summary,
                    "build_prep_reroute_attempts": build_prep_reroute_attempts,
                    "max_build_prep_reroutes": max_reroutes,
                },
            }
        build_prep_reroute_attempts += 1
        announce("build_config_preflight_from_validation", rounds=repair_rounds_completed)
        build_config = run_build_config_preflight(
            spec,
            state_root=state_root,
            model=model,
            max_completion_tokens=max_completion_tokens,
            allow_escalation=allow_escalation,
            max_attempts=max_build_config_preflight_attempts,
            progress=_emit_progress,
        )
        record(
            "build_config_preflight_from_validation",
            {
                **build_config,
                "triggering_diagnostic_routes": route_summary,
            },
            rounds=repair_rounds_completed,
        )
        if not build_config.get("ok"):
            record(
                "build_config_preflight_failed_from_validation",
                {
                    "reason": (
                        "build/config preflight did not close A-stage diagnostics; "
                        "B/C source repair remains blocked until environment preparation succeeds"
                    ),
                    "build_config_preflight": build_config,
                    "diagnostic_routes": route_summary,
                },
                rounds=repair_rounds_completed,
            )
            return {
                "status": "failed",
                "stop_reason": "build_config_preflight_failed",
                "validation": current_validation,
                "extra": {
                    "diagnostic_routes": route_summary,
                    "build_config_preflight": build_config,
                    "source_repair_blocked": True,
                    "environment_failure": bool(
                        build_config.get("environment_failure")
                    ),
                },
            }

        announce("validate_after_build_config_preflight", rounds=repair_rounds_completed)
        refreshed_validation = run_post_validation_cycle(
            spec,
            state_root=state_root,
            execution_phase=active_execution_phase,
        )
        record(
            "validate_after_build_config_preflight",
            {
                **refreshed_validation,
                "previous_diagnostic_routes": route_summary,
                "new_diagnostic_routes": diagnostic_route_summary(
                    _full_validation_for_routing(refreshed_validation)
                ),
            },
            rounds=repair_rounds_completed,
        )
        return {
            "status": "rerouted",
            "validation": refreshed_validation,
            "execution_phase": active_execution_phase,
        }

    def continue_agentic_repair_from_validation(
        current_validation: dict,
        *,
        completed_rounds: int = 0,
        repair_lifecycle: SemanticCompileRecoveryLifecycle | None = None,
    ) -> dict:
        nonlocal spec
        durable_round_counts = validated_repair_round_counts(
            state_root
        )
        validated_compile_repair_rounds_completed = int(
            durable_round_counts["compile_repair_rounds_completed"]
        )
        validated_semantic_repair_rounds_completed = int(
            durable_round_counts["semantic_repair_rounds_completed"]
        )
        validated_repair_rounds_completed = (
            validated_compile_repair_rounds_completed
            + validated_semantic_repair_rounds_completed
        )
        repair_rounds_completed = validated_repair_rounds_completed
        if int(completed_rounds) != validated_repair_rounds_completed:
            record(
                "repair_round_count_reconciled",
                {
                    "requested_completed_rounds": int(completed_rounds),
                    **durable_round_counts,
                    "authority": "validated_round_manifests",
                },
                rounds=validated_repair_rounds_completed,
            )
        max_agent_steps = max(1, int(os.environ.get("DRIVER_BACKPORT_AGENTIC_MAX_STEPS", "4")))
        max_consecutive_model_failures = max(
            1,
            int(
                os.environ.get(
                    "DRIVER_BACKPORT_MAX_CONSECUTIVE_MODEL_FAILURES",
                    "2",
                )
            ),
        )
        consecutive_model_failures = 0
        round_plan: dict[str, Any] | None = None
        round_pending_plan: dict[str, Any] | None = None
        round_execution_phase = (
            repair_lifecycle.execution_phase
            if repair_lifecycle is not None
            else EXECUTION_PHASE_COMPILER_REPAIR
        )
        round_source_plan_path: Path | None = None
        round_root_dir: Path | None = None
        round_unit_sequence = 0
        round_apply_results: list[dict[str, Any]] = []
        round_validation_targets: list[dict[str, Any]] = []
        round_assessment_targets: list[dict[str, Any]] = []
        round_unit_outcomes: list[dict[str, Any]] = []
        round_applied_units: list[dict[str, Any]] = []
        round_batch_records: list[dict[str, Any]] = []
        round_scope_changes: list[dict[str, Any]] = []
        round_manifest_flushed_counts = {
            field: 0 for field in ROUND_MANIFEST_COLLECTIONS
        }
        round_manifest_streamed_counts = {
            "unit_outcomes": 0,
            "apply_results": 0,
        }
        selected_round_assessment_target: dict[str, Any] | None = None

        def complete_repair_controller(
            validation: dict[str, Any],
            *,
            stop_reason: str,
            extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Return repair facts; only the owning phase may transition."""

            compile_validated = (
                validation.get("status") == "ready_for_manual_test"
            )
            result = {
                "schema": CANONICAL_REPAIR_CONTROLLER_RESULT_SCHEMA,
                "outcome": (
                    "compile_validated"
                    if compile_validated
                    else "stopped"
                ),
                "status": str(validation.get("status") or ""),
                "stop_reason": str(stop_reason),
                "compile_validated": compile_validated,
                "validation": copy.deepcopy(validation),
                "compile_repair_rounds_completed": (
                    validated_compile_repair_rounds_completed
                ),
                "semantic_repair_rounds_completed": (
                    validated_semantic_repair_rounds_completed
                ),
                "repair_rounds_completed": (
                    validated_repair_rounds_completed
                ),
                "active_repair_budget": active_phase_budget(),
                **dict(extra or {}),
            }
            if repair_lifecycle is not None:
                return result
            if compile_validated:
                return finish_after_compile_validation(
                    validation,
                    completed_rounds=(
                        validated_compile_repair_rounds_completed
                    ),
                    completion_extra=dict(extra or {}),
                )
            final_state = show_status(spec, state_root=state_root)
            return finish(
                status=final_state["status"],
                stop_reason=str(stop_reason),
                repair_rounds_completed=(
                    validated_repair_rounds_completed
                ),
                extra={
                    "active_repair_budget": active_phase_budget(),
                    **dict(extra or {}),
                },
            )

        def persist_round_manifest(
            manifest_path: Path,
            *,
            status: str,
            remaining_unit_count: int,
            extra: dict[str, Any] | None = None,
        ) -> None:
            collections = {
                "validation_targets": round_validation_targets,
                "assessment_targets": round_assessment_targets,
                "applied_units": round_applied_units,
                "unit_outcomes": round_unit_outcomes,
                "apply_results": round_apply_results,
                "validation_batches": round_batch_records,
                "scope_changes": round_scope_changes,
            }
            pending_collections = {
                field: list(records)[
                    round_manifest_flushed_counts.get(field, 0) :
                ]
                for field, records in collections.items()
                if len(records)
                > round_manifest_flushed_counts.get(field, 0)
            }
            append_round_manifest_collections(
                manifest_path,
                pending_collections,
                state_root=state_root,
            )
            for field, records in collections.items():
                if field in round_manifest_streamed_counts:
                    round_manifest_streamed_counts[field] += len(
                        pending_collections.get(field) or []
                    )
                    records.clear()
                    round_manifest_flushed_counts[field] = 0
                else:
                    round_manifest_flushed_counts[field] = len(records)

            manifest_extra = dict(extra or {})
            inflight_target = manifest_extra.pop(
                "inflight_validation_target",
                None,
            )
            if isinstance(inflight_target, dict):
                inflight_target_path = (
                    manifest_path.parent
                    / "inflight_validation_target.json"
                )
                write_json(inflight_target_path, inflight_target)
                manifest_extra["inflight_validation_target_path"] = str(
                    inflight_target_path.resolve()
                )
            payload = {
                "schema": "ordered_graph_root_repair_round_v6",
                "round": repair_rounds_completed,
                "invocation_id": run_id,
                "execution_phase": round_execution_phase,
                "status": status,
                "processed_unit_count": (
                    round_manifest_streamed_counts["unit_outcomes"]
                ),
                "remaining_unit_count": max(0, int(remaining_unit_count)),
                "source_repair_plan_path": (
                    str(round_source_plan_path.resolve())
                    if round_source_plan_path is not None
                    else None
                ),
                "remaining_plan_unit_ids": _repair_plan_unit_ids(round_plan),
                "deferred_plan_unit_ids": _repair_plan_unit_ids(
                    round_pending_plan
                ),
                "validation_batch_policy": (
                    "complete_dependency_ready_frontier"
                ),
                "record_event_log_path": str(
                    round_manifest_event_log_path(
                        manifest_path
                    ).resolve()
                ),
                "validation_target_count": len(round_validation_targets),
                "assessment_target_count": len(round_assessment_targets),
                "applied_unit_count": len(round_applied_units),
                "unit_outcome_count": (
                    round_manifest_streamed_counts["unit_outcomes"]
                ),
                "apply_result_count": (
                    round_manifest_streamed_counts["apply_results"]
                ),
                "validation_batch_count": len(round_batch_records),
                "scope_change_count": len(round_scope_changes),
            }
            payload.update(manifest_extra)
            write_json(manifest_path, payload)

        def register_round_scope_change(
            *,
            full_plan: dict[str, Any],
            selected_plan_unit_id: str,
            scope_expansion: dict[str, Any],
            result_path: Path,
        ) -> None:
            """Record source-changing compiler scope work for round validation."""

            if not _scope_expansion_changes_source_or_build(scope_expansion):
                return
            event_id = str(scope_expansion.get("event_id") or "")
            if event_id and any(
                str(item.get("event_id") or "") == event_id
                for item in round_scope_changes
            ):
                return
            scope_changed_files = list(
                dict.fromkeys(
                    [
                        *list(scope_expansion.get("copied_files") or []),
                        *list(
                            scope_expansion.get("build_integration_files")
                            or []
                        ),
                    ]
                )
            )
            target = _validation_target_for_plan_unit(
                full_plan,
                selected_plan_unit_id=selected_plan_unit_id,
            )
            if target is not None:
                target = _validation_target_with_apply_evidence(
                    target,
                    apply_result=None,
                    baseline_validation=current_validation,
                )
                target = (
                    _merge_round_assessment_target(
                        selected_round_assessment_target,
                        target,
                    )
                    or target
                )
                target["repair_applied"] = True
                target["scope_expansion_event_id"] = event_id or None
                target["scope_expansion_path"] = str(result_path.resolve())
                round_validation_targets.append(target)
                round_assessment_targets.append(target)
            lifecycle_dependencies = _semantic_lifecycle_dependencies(target)
            scope_change_record = {
                "event_id": event_id or None,
                "plan_unit_id": selected_plan_unit_id,
                "result_path": str(result_path.resolve()),
                "transaction_dir": scope_expansion.get(
                    "transaction_dir"
                ),
                "copied_files": list(
                    scope_expansion.get("copied_files") or []
                ),
                "build_integration_files": list(
                    scope_expansion.get("build_integration_files") or []
                ),
                "validation": "required_before_replan",
            }
            if repair_lifecycle is not None:
                scope_change_record["repair_lifecycle_transaction"] = (
                    repair_lifecycle.record_transaction(
                        transaction_ref=str(
                            scope_expansion.get("transaction_dir") or ""
                        ),
                        work_unit_id=str(selected_plan_unit_id),
                        touched_files=[
                            str(path)
                            for path in scope_changed_files
                            if str(path)
                        ],
                        changed_ranges=[],
                        evidence={
                            "schema": (
                                "canonical_compiler_scope_transaction_v1"
                            ),
                            "controller_round": repair_rounds_completed,
                            "plan_unit_id": selected_plan_unit_id,
                            "scope_expansion": scope_expansion,
                            "result_path": str(result_path.resolve()),
                        },
                        depends_on_patch_ids=lifecycle_dependencies,
                    )
                )
            round_scope_changes.append(scope_change_record)

        def complete_scope_context_pending(
            *,
            manifest_path: Path,
            full_plan: dict[str, Any],
            selected_plan_unit_id: str,
            scope_expansion: dict[str, Any],
            compile_context_result: dict[str, Any],
        ) -> dict[str, Any]:
            """Close this invocation while durable scope awaits context."""

            stop_reason = str(
                compile_context_result.get("stop_reason")
                or "scope_expansion_compile_context_failed"
            )
            round_unit_outcomes.append(
                {
                    "sequence": round_unit_sequence,
                    "plan_unit_id": selected_plan_unit_id,
                    "status": "scope_expanded_pending_compile_context",
                    "reason": stop_reason,
                    "scope_expansion_event_id": scope_expansion.get(
                        "event_id"
                    ),
                    "repair_budget_consumed": False,
                }
            )
            persist_round_manifest(
                manifest_path,
                status="scope_change_pending_compile_context",
                remaining_unit_count=len(
                    full_plan.get("work_unit_plans") or []
                ),
                extra={
                    "repair_budget_consumed": False,
                    "scope_expansion_event_id": scope_expansion.get(
                        "event_id"
                    ),
                    "stop_reason": stop_reason,
                },
            )
            return complete_repair_controller(
                current_validation,
                stop_reason=stop_reason,
                extra=dict(compile_context_result.get("extra") or {}),
            )

        def advance_after_unit(
            *,
            full_plan: dict[str, Any],
            selected_plan_unit_id: str,
            outcome: dict[str, Any],
            apply_result: dict[str, Any] | None = None,
            validation_target: dict[str, Any] | None = None,
            force_round_close: bool = False,
            deferred_plan_unit_ids: Iterable[str] = (),
            terminal_stop_reason: str | None = None,
            terminal_extra: dict[str, Any] | None = None,
        ) -> dict[str, Any] | None:
            nonlocal round_plan
            nonlocal round_pending_plan
            nonlocal current_validation
            nonlocal validated_repair_rounds_completed
            nonlocal validated_compile_repair_rounds_completed
            nonlocal validated_semantic_repair_rounds_completed
            nonlocal round_apply_results
            nonlocal round_validation_targets
            nonlocal round_assessment_targets
            nonlocal round_unit_outcomes
            nonlocal round_applied_units
            nonlocal round_batch_records
            nonlocal round_scope_changes

            round_unit_outcomes.append(
                {
                    "sequence": round_unit_sequence,
                    "plan_unit_id": selected_plan_unit_id,
                    **outcome,
                }
            )
            assessment_target = _merge_round_assessment_target(
                selected_round_assessment_target,
                validation_target,
            )
            lifecycle_dependencies = _semantic_lifecycle_dependencies(
                assessment_target
            )
            if apply_result is not None:
                round_apply_results.append(apply_result)
                if repair_lifecycle is not None:
                    lifecycle_transactions = []
                    for applied_result in (
                        apply_result.get("results") or []
                    ):
                        if not isinstance(applied_result, dict) or str(
                            applied_result.get("status") or ""
                        ) != "applied":
                            continue
                        diff_text = "\n".join(
                            _patch_lines_from_apply_result(applied_result)
                        )
                        lifecycle_transactions.append(
                            repair_lifecycle.record_transaction(
                                transaction_ref=str(
                                    applied_result.get(
                                        "transaction_dir"
                                    )
                                    or ""
                                ),
                                work_unit_id=str(
                                    applied_result.get("unit_id")
                                    or selected_plan_unit_id
                                    or ""
                                ),
                                touched_files=[
                                    str(path)
                                    for path in applied_result.get(
                                        "touched_files"
                                    )
                                    or []
                                    if str(path)
                                ],
                                changed_ranges=(
                                    unified_diff_changed_ranges(
                                        diff_text
                                    )
                                    if diff_text
                                    else []
                                ),
                                evidence={
                                    "schema": (
                                        "canonical_compiler_repair_"
                                        "transaction_v1"
                                    ),
                                    "controller_round": (
                                        repair_rounds_completed
                                    ),
                                    "plan_unit_id": (
                                        selected_plan_unit_id
                                    ),
                                    "apply_result": applied_result,
                                },
                                depends_on_patch_ids=(
                                    lifecycle_dependencies
                                ),
                            )
                        )
                    round_unit_outcomes[-1][
                        "repair_lifecycle_transactions"
                    ] = lifecycle_transactions
            if assessment_target is not None:
                round_assessment_targets.append(assessment_target)
            if validation_target is not None:
                applied_assessment_target = (
                    assessment_target or validation_target
                )
                round_validation_targets.append(applied_assessment_target)
                round_applied_units.append(
                    {
                        "plan_unit_id": selected_plan_unit_id,
                        "work_unit_id": str(
                            applied_assessment_target.get(
                                "origin_work_unit_id"
                            )
                            or applied_assessment_target.get("work_unit_id")
                            or applied_assessment_target.get("id")
                            or ""
                        ),
                        "assessment_target": copy.deepcopy(
                            applied_assessment_target
                        ),
                        "apply_result": apply_result or {},
                    }
                )

            remaining = _repair_plan_without_unit(
                full_plan,
                completed_plan_unit_id=selected_plan_unit_id,
                deferred_plan_unit_ids=deferred_plan_unit_ids,
            )
            root_dir = round_root_dir or Path(current_validation["cycle_dir"]).resolve()
            manifest_path = root_dir / "round_manifest.json"

            if remaining.get("work_unit_plans") and not force_round_close:
                round_plan = remaining
                persist_round_manifest(
                    manifest_path,
                    status="patch_generation_in_progress",
                    remaining_unit_count=len(
                        remaining.get("work_unit_plans") or []
                    ),
                )
                return None

            round_plan = remaining if force_round_close else None
            if (
                not force_round_close
                and not round_validation_targets
                and round_pending_plan is not None
                and round_pending_plan.get("work_unit_plans")
            ):
                next_plan, next_pending, selection = (
                    _select_validation_batch_repair_plan(
                        round_pending_plan,
                        spec=spec,
                    )
                )
                if next_plan.get("work_unit_plans"):
                    round_plan = next_plan
                    round_pending_plan = next_pending
                    round_batch_records.append(
                        {
                            "selected_work_unit_ids": list(
                                selection.get("selected_work_unit_ids") or []
                            ),
                            "deferred_count": int(
                                (selection.get("summary") or {}).get(
                                    "deferred_count"
                                )
                                or 0
                            ),
                            "reason": "previous_batch_made_no_source_change",
                        }
                    )
                    persist_round_manifest(
                        manifest_path,
                        status="patch_generation_in_progress",
                        remaining_unit_count=len(
                            next_plan.get("work_unit_plans") or []
                        ),
                        extra={
                            "deferred_unit_count": len(
                                next_pending.get("work_unit_plans") or []
                            ),
                        },
                    )
                    return None
            closed_remaining_unit_count = (
                len(remaining.get("work_unit_plans") or [])
                + len(
                    (round_pending_plan or {}).get("work_unit_plans")
                    or []
                )
                if force_round_close
                else 0
            )
            exact_export_closure: dict[str, Any] | None = None
            if (
                round_execution_phase
                == EXECUTION_PHASE_COMPILER_REPAIR
                and not round_scope_changes
            ):
                exact_export_closure = run_exact_export_contract_closure(
                    cycle_dir=root_dir,
                    rounds=repair_rounds_completed,
                )
                for target in exact_export_closure.get(
                    "validation_targets"
                ) or []:
                    target = _validation_target_with_apply_evidence(
                        target,
                        apply_result=None,
                        baseline_validation=current_validation,
                    )
                    round_validation_targets.append(target)
                    round_assessment_targets.append(target)
                round_batch_records.append(
                    {
                        "kind": "exact_export_contract_closure",
                        "artifact_path": exact_export_closure.get(
                            "artifact_path"
                        ),
                        "candidate_count": exact_export_closure.get(
                            "candidate_count"
                        ),
                        "applied_count": exact_export_closure.get(
                            "applied_count"
                        ),
                    }
                )
            if not round_validation_targets and not terminal_stop_reason:
                retryable_terminal = _retryable_model_round_terminal(
                    round_unit_outcomes,
                    repair_round=repair_rounds_completed,
                    persisted_outcomes=round_manifest_collections(
                        manifest_path,
                        state_root=state_root,
                        fields=("unit_outcomes",),
                    )["unit_outcomes"],
                )
                if retryable_terminal:
                    terminal_stop_reason = str(
                        retryable_terminal["stop_reason"]
                    )
                    terminal_extra = {
                        **(terminal_extra or {}),
                        **dict(retryable_terminal.get("extra") or {}),
                    }
            persist_round_manifest(
                manifest_path,
                status=(
                    "pending_combined_validation"
                    if round_validation_targets
                    else "no_applied_patch"
                ),
                remaining_unit_count=closed_remaining_unit_count,
                extra=(
                    {
                        "forced_round_close": True,
                        "terminal_stop_reason": terminal_stop_reason,
                        "terminal_extra": terminal_extra or {},
                    }
                    if force_round_close or terminal_stop_reason
                    else (
                        {
                            "exact_export_contract_closure": {
                                "artifact_path": exact_export_closure.get(
                                    "artifact_path"
                                ),
                                "candidate_count": exact_export_closure.get(
                                    "candidate_count"
                                ),
                                "applied_count": exact_export_closure.get(
                                    "applied_count"
                                ),
                            }
                        }
                        if exact_export_closure is not None
                        else None
                    )
                ),
            )
            if not round_validation_targets:
                lifecycle_terminal = (
                    repair_lifecycle.finish_without_validation(
                        reason=(
                            terminal_stop_reason
                            or "canonical compiler round produced no "
                            "validation-worthy source change"
                        ),
                        outcome_kind=(
                            "terminal_infrastructure_failure"
                            if terminal_stop_reason
                            else "no_validation_worthy_change"
                        ),
                    )
                    if repair_lifecycle is not None
                    else None
                )
                return complete_repair_controller(
                    current_validation,
                    stop_reason=(
                        terminal_stop_reason
                        or f"agentic_round_{repair_rounds_completed}_no_validation_worthy_change"
                    ),
                    extra={
                        "round_manifest_path": str(manifest_path.resolve()),
                        **(
                            {"repair_lifecycle": lifecycle_terminal}
                            if lifecycle_terminal is not None
                            else {}
                        ),
                        **(terminal_extra or {}),
                    },
                )

            announce(
                "agentic_validate_after_ordered_root_round",
                rounds=repair_rounds_completed,
            )
            current_validation = run_post_validation_cycle(
                spec,
                state_root=state_root,
                validation_targets=list(round_validation_targets),
                validation_assessment_targets=list(
                    round_assessment_targets
                ),
                execution_phase=round_execution_phase,
            )
            record(
                "agentic_validate_after_ordered_root_round",
                current_validation,
                rounds=repair_rounds_completed,
            )
            post_round_build_prep_route = _reroute_build_prep_if_needed(
                current_validation,
                repair_rounds_completed=(
                    validated_repair_rounds_completed
                ),
                execution_phase=round_execution_phase,
            )
            current_validation = dict(
                post_round_build_prep_route.get("validation")
                or current_validation
            )
            postimage_observed = bool(
                post_round_build_prep_route.get("status") != "failed"
                and _validation_observes_repair_postimage(
                    current_validation
                )
            )
            round_scope_event_ids = [
                str(item.get("event_id") or "")
                for item in round_scope_changes
                if str(item.get("event_id") or "")
            ]
            scope_validation_completion = (
                complete_graph_scope_validations(
                    spec,
                    state_root=state_root,
                    event_ids=round_scope_event_ids,
                    validation=current_validation,
                )
                if round_scope_event_ids and postimage_observed
                else None
            )
            if not postimage_observed:
                persist_round_manifest(
                    manifest_path,
                    status="validation_blocked_pending_retry",
                    remaining_unit_count=closed_remaining_unit_count,
                    extra={
                        "validation_path": current_validation.get(
                            "validation_path"
                        ),
                        "validation_delta": current_validation.get(
                            "validation_delta"
                        ),
                        "build_preparation_route": (
                            post_round_build_prep_route
                        ),
                        "repair_budget_consumed": False,
                        "scope_event_ids_pending_validation": (
                            round_scope_event_ids
                        ),
                        **(
                            {
                                "forced_round_close": True,
                                "terminal_stop_reason": terminal_stop_reason,
                                "terminal_extra": terminal_extra or {},
                            }
                            if force_round_close
                            else {}
                        ),
                    },
                )
                return complete_repair_controller(
                    current_validation,
                    stop_reason=str(
                        post_round_build_prep_route.get("stop_reason")
                        or f"repair_round_{repair_rounds_completed}_"
                        "validation_pending_retry"
                    ),
                    extra={
                        "round_manifest_path": str(
                            manifest_path.resolve()
                        ),
                        "repair_budget_consumed": False,
                        "round_validation_pending_retry": True,
                        "resumable_repair_round": True,
                    },
                )
            candidate_state = _retain_round_candidates(
                validation_delta=dict(
                    current_validation.get("validation_delta") or {}
                ),
                applied_units=round_applied_units,
            )
            patch_outcome_update = record_round_patch_outcomes(
                state_root,
                repair_round=repair_rounds_completed,
                applied_units=round_applied_units,
                validation_delta=dict(
                    current_validation.get("validation_delta") or {}
                ),
                candidate_state=candidate_state,
            )
            record(
                "agentic_patch_outcome_ledger_updated",
                patch_outcome_update,
                rounds=repair_rounds_completed,
            )

            lifecycle_validation = (
                repair_lifecycle.record_validation(current_validation)
                if repair_lifecycle is not None
                else None
            )
            structural_checkpoint_only = bool(
                round_scope_changes and not round_applied_units
            )
            persist_round_manifest(
                manifest_path,
                status="validated",
                remaining_unit_count=closed_remaining_unit_count,
                extra={
                    "validation_path": current_validation.get("validation_path"),
                    "validation_delta": current_validation.get("validation_delta"),
                    "candidate_retention_state": candidate_state,
                    "repair_budget_consumed": (
                        not structural_checkpoint_only
                    ),
                    **(
                        {"structural_scope_checkpoint": True}
                        if structural_checkpoint_only
                        else {}
                    ),
                    **(
                        {
                            "scope_validation_completion": (
                                scope_validation_completion
                            )
                        }
                        if scope_validation_completion is not None
                        else {}
                    ),
                    **(
                        {"repair_lifecycle": lifecycle_validation}
                        if lifecycle_validation is not None
                        else {}
                    ),
                    **(
                        {
                            "forced_round_close": True,
                            "terminal_stop_reason": terminal_stop_reason,
                            "terminal_extra": terminal_extra or {},
                        }
                        if force_round_close
                        else {}
                    ),
                },
            )
            durable_counts = validated_repair_round_counts(state_root)
            validated_compile_repair_rounds_completed = int(
                durable_counts["compile_repair_rounds_completed"]
            )
            validated_semantic_repair_rounds_completed = int(
                durable_counts["semantic_repair_rounds_completed"]
            )
            validated_repair_rounds_completed = int(
                durable_counts["total_repair_rounds_completed"]
            )
            if current_validation["status"] == "ready_for_manual_test":
                return complete_repair_controller(
                    current_validation,
                    stop_reason="compile_validated",
                    extra={
                        "round_manifest_path": str(
                            manifest_path.resolve()
                        ),
                        **(
                            {"repair_lifecycle": lifecycle_validation}
                            if lifecycle_validation is not None
                            else {}
                        ),
                    },
                )
            if _validation_blocks_auto_repair(current_validation):
                return complete_repair_controller(
                    current_validation,
                    stop_reason=f"validation_environment_blocked_round_{repair_rounds_completed}",
                    extra={
                        "round_manifest_path": str(manifest_path.resolve()),
                        "environment_failure": True,
                        **(
                            {"repair_lifecycle": lifecycle_validation}
                            if lifecycle_validation is not None
                            else {}
                        ),
                    },
                )
            if terminal_stop_reason:
                return complete_repair_controller(
                    current_validation,
                    stop_reason=terminal_stop_reason,
                    extra={
                        "round_manifest_path": str(manifest_path.resolve()),
                        **(
                            {"repair_lifecycle": lifecycle_validation}
                            if lifecycle_validation is not None
                            else {}
                        ),
                        **(terminal_extra or {}),
                    },
                )
            if force_round_close:
                # Remaining units were planned against the pre-apply tree.
                # Rebuild them from the freshly validated candidate instead
                # of carrying mutable ranges and owner bindings forward.
                round_plan = None
                round_pending_plan = None
            return None

        def active_phase_budget() -> dict[str, Any]:
            if repair_lifecycle is not None:
                return repair_lifecycle.budget_status()
            base_budget = repair_phase_budget_status(
                execution_phase=EXECUTION_PHASE_COMPILER_REPAIR,
                compile_rounds_completed=(
                    validated_compile_repair_rounds_completed
                ),
                semantic_rounds_completed=(
                    validated_semantic_repair_rounds_completed
                ),
                max_compile_rounds=max_repair_rounds,
                max_semantic_rounds=(
                    max_semantic_repair_rounds
                ),
            )
            return base_budget

        if (
            repair_lifecycle is None
            and current_validation.get("status") == "needs_patching"
        ):
            pre_round_export_closure = run_exact_export_contract_closure(
                cycle_dir=Path(current_validation["cycle_dir"]).resolve(),
                rounds=validated_repair_rounds_completed,
            )
            if int(pre_round_export_closure.get("applied_count") or 0) > 0:
                current_validation = run_post_validation_cycle(
                    spec,
                    state_root=state_root,
                    validation_targets=list(
                        pre_round_export_closure.get(
                            "validation_targets"
                        )
                        or []
                    ),
                    execution_phase=EXECUTION_PHASE_COMPILER_REPAIR,
                )
                record(
                    "validate_after_exact_export_contract_closure",
                    current_validation,
                    rounds=validated_repair_rounds_completed,
                )
                if current_validation.get("status") == (
                    "ready_for_manual_test"
                ):
                    return complete_repair_controller(
                        current_validation,
                        stop_reason="compile_validated",
                        extra={
                            "exact_export_contract_closure": (
                                pre_round_export_closure
                            ),
                        },
                    )

        while _validation_can_enter_repair_loop(current_validation):
            build_prep_route = _reroute_build_prep_if_needed(
                current_validation,
                repair_rounds_completed=validated_repair_rounds_completed,
                execution_phase=round_execution_phase,
            )
            current_validation = dict(
                build_prep_route.get("validation") or current_validation
            )
            if build_prep_route.get("status") == "failed":
                return complete_repair_controller(
                    current_validation,
                    stop_reason=str(
                        build_prep_route.get("stop_reason")
                        or "build_config_preflight_failed"
                    ),
                    extra=dict(build_prep_route.get("extra") or {}),
                )
            if current_validation["status"] != "needs_patching":
                break
            cycle_dir = Path(current_validation["cycle_dir"]).resolve()
            if round_plan is None and repair_lifecycle is None:
                checkpoint_source_path = Path(
                    current_validation["repair_plan_path"]
                ).resolve()
                checkpoint_source_plan = load_repair_plan(
                    checkpoint_source_path,
                    state_root=state_root,
                )
                checkpoint = _dependency_ready_scope_checkpoint_plan(
                    checkpoint_source_plan,
                    spec=spec,
                )
                checkpoint_ids = set(
                    checkpoint.get("work_unit_ids") or []
                )
                if checkpoint_ids:
                    checkpoint_plan = dict(
                        checkpoint.get("repair_plan") or {}
                    )
                    checkpoint_phase = repair_plan_execution_phase(
                        checkpoint_plan
                    )
                    if checkpoint_phase != EXECUTION_PHASE_COMPILER_REPAIR:
                        raise ValueError(
                            "pre-repair structural scope checkpoint must "
                            "belong to compiler repair"
                        )
                    expanded_spec, scope_expansion = materialize_graph_scope(
                        spec,
                        checkpoint_plan,
                        state_root=state_root,
                        cycle=int(current_validation.get("cycle") or 0),
                        work_unit_ids=checkpoint_ids,
                        allow_escalation=allow_escalation,
                        execution_phase=(
                            EXECUTION_PHASE_COMPILER_REPAIR
                        ),
                    )
                    checkpoint_dir = ensure_dir(
                        cycle_dir / "structural_scope_checkpoint"
                    )
                    scope_expansion_path = (
                        checkpoint_dir / "scope_expansion.json"
                    )
                    write_json(scope_expansion_path, scope_expansion)
                    if scope_expansion.get("status") != "expanded":
                        checkpoint_failure = {
                            **checkpoint,
                            "status": "blocked_before_repair_round",
                            "scope_expansion_status": (
                                scope_expansion.get("status")
                            ),
                            "scope_expansion_path": str(
                                scope_expansion_path.resolve()
                            ),
                            "scope_expansion": scope_expansion,
                            "repair_budget_consumed": False,
                            "reason": (
                                "a planner-selected structural dependency "
                                "could not be materialized; ordinary repair "
                                "must not run against the incomplete graph"
                            ),
                        }
                        record(
                            "structural_scope_checkpoint_blocked",
                            checkpoint_failure,
                            rounds=validated_repair_rounds_completed,
                        )
                        return complete_repair_controller(
                            current_validation,
                            stop_reason=(
                                "scope_expansion_authorization_required"
                                if scope_expansion.get("status")
                                == "authorization_required"
                                else (
                                    "structural_scope_checkpoint_"
                                    "invariant_failed"
                                )
                            ),
                            extra={
                                "structural_scope_checkpoint": (
                                    checkpoint_failure
                                )
                            },
                        )
                    if scope_expansion.get("status") == "expanded":
                        spec = expanded_spec
                        compile_context_result = (
                            refresh_compile_context_after_scope_expansion(
                                repair_rounds_completed=(
                                    validated_repair_rounds_completed
                                ),
                                scope_expansion=scope_expansion,
                            )
                        )
                        if compile_context_result.get("status") == "failed":
                            return complete_repair_controller(
                                current_validation,
                                stop_reason=str(
                                    compile_context_result.get(
                                        "stop_reason"
                                    )
                                    or (
                                        "scope_expansion_compile_context_"
                                        "failed"
                                    )
                                ),
                                extra=dict(
                                    compile_context_result.get("extra") or {}
                                ),
                            )
                    if (
                        scope_expansion.get("status") == "expanded"
                        and not _scope_expansion_changes_source_or_build(
                            scope_expansion
                        )
                    ):
                        refreshed_plan = continue_repair_plan_graph(
                            spec,
                            checkpoint_source_plan,
                            work_unit_ids=checkpoint_ids,
                            refresh_current_tree=True,
                        )
                        refreshed_path = (
                            checkpoint_dir / "permission-refreshed-plan.json"
                        )
                        refreshed_plan["source_plan_path"] = str(
                            checkpoint_source_path
                        )
                        write_repair_plan(
                            refreshed_path,
                            refreshed_plan,
                            state_root=state_root,
                        )
                        current_validation = {
                            **current_validation,
                            "repair_plan_path": str(
                                refreshed_path.resolve()
                            ),
                        }
                        record(
                            "structural_scope_permission_checkpoint",
                            {
                                **checkpoint,
                                "status": "plan_refreshed",
                                "scope_expansion_path": str(
                                    scope_expansion_path.resolve()
                                ),
                                "refreshed_plan_path": str(
                                    refreshed_path.resolve()
                                ),
                                "repair_budget_consumed": False,
                                "validation_required": False,
                            },
                            rounds=validated_repair_rounds_completed,
                        )
                        continue
                    if (
                        scope_expansion.get("status") == "expanded"
                        and _scope_expansion_changes_source_or_build(
                            scope_expansion
                        )
                    ):
                        checkpoint_targets: list[dict[str, Any]] = []
                        for checkpoint_id in sorted(checkpoint_ids):
                            target = _validation_target_for_plan_unit(
                                checkpoint_source_plan,
                                selected_plan_unit_id=checkpoint_id,
                            )
                            if target is None:
                                continue
                            target = _validation_target_with_apply_evidence(
                                target,
                                apply_result=None,
                                baseline_validation=current_validation,
                            )
                            target.update(
                                {
                                    "repair_applied": True,
                                    "structural_scope_checkpoint": True,
                                    "scope_expansion_event_id": (
                                        scope_expansion.get("event_id")
                                    ),
                                    "scope_expansion_path": str(
                                        scope_expansion_path.resolve()
                                    ),
                                }
                            )
                            checkpoint_targets.append(target)
                        checkpoint_before = {
                            **checkpoint,
                            "status": "pending_fresh_validation",
                            "source_repair_plan_path": str(
                                checkpoint_source_path
                            ),
                            "scope_expansion_path": str(
                                scope_expansion_path.resolve()
                            ),
                            "scope_expansion": scope_expansion,
                            "repair_budget_consumed": False,
                            "deferred_unit_policy": (
                                "reobserve_all_unrepaired_diagnostics_from_"
                                "fresh_build_before_first_repair_round"
                            ),
                        }
                        checkpoint_path = checkpoint_dir / "checkpoint.json"
                        write_json(checkpoint_path, checkpoint_before)
                        announce(
                            "validate_after_structural_scope_checkpoint",
                            rounds=validated_repair_rounds_completed,
                        )
                        current_validation = run_post_validation_cycle(
                            spec,
                            state_root=state_root,
                            validation_targets=checkpoint_targets,
                            validation_assessment_targets=(
                                checkpoint_targets
                            ),
                            execution_phase=EXECUTION_PHASE_COMPILER_REPAIR,
                        )
                        checkpoint_build_prep_route = (
                            _reroute_build_prep_if_needed(
                                current_validation,
                                repair_rounds_completed=(
                                    validated_repair_rounds_completed
                                ),
                                execution_phase=(
                                    EXECUTION_PHASE_COMPILER_REPAIR
                                ),
                            )
                        )
                        current_validation = dict(
                            checkpoint_build_prep_route.get("validation")
                            or current_validation
                        )
                        checkpoint_postimage_observed = (
                            checkpoint_build_prep_route.get("status")
                            != "failed"
                            and _validation_observes_repair_postimage(
                                current_validation
                            )
                        )
                        if not checkpoint_postimage_observed:
                            checkpoint_pending = {
                                **checkpoint_before,
                                "status": "validation_blocked_pending_retry",
                                "validation_path": current_validation.get(
                                    "validation_path"
                                ),
                                "post_validation_status": (
                                    current_validation.get("status")
                                ),
                                "build_preparation_route": (
                                    checkpoint_build_prep_route
                                ),
                                "scope_event_closed": False,
                                "repair_budget_consumed": False,
                            }
                            write_json(checkpoint_path, checkpoint_pending)
                            record(
                                "structural_scope_checkpoint_validation_"
                                "blocked",
                                {
                                    **checkpoint_pending,
                                    "checkpoint_path": str(
                                        checkpoint_path.resolve()
                                    ),
                                },
                                rounds=(
                                    validated_repair_rounds_completed
                                ),
                            )
                            return complete_repair_controller(
                                current_validation,
                                stop_reason=(
                                    str(
                                        checkpoint_build_prep_route.get(
                                            "stop_reason"
                                        )
                                        or "structural_scope_checkpoint_"
                                        "validation_pending_retry"
                                    )
                                ),
                                extra={
                                    "structural_scope_checkpoint_path": str(
                                        checkpoint_path.resolve()
                                    ),
                                    "repair_budget_consumed": False,
                                    "resumable_scope_validation": True,
                                },
                            )
                        scope_validation_completion = (
                            complete_graph_scope_validations(
                                spec,
                                state_root=state_root,
                                event_ids=[
                                    scope_expansion.get("event_id")
                                ],
                                validation=current_validation,
                            )
                        )
                        after_plan: dict[str, Any] | None = None
                        after_plan_path = str(
                            current_validation.get("repair_plan_path") or ""
                        )
                        if after_plan_path:
                            after_plan = load_repair_plan(
                                Path(after_plan_path).resolve(),
                                state_root=state_root,
                            )
                        reconciliation = _scope_checkpoint_reconciliation(
                            checkpoint_source_plan,
                            after_plan,
                        )
                        checkpoint_result = {
                            **checkpoint_before,
                            "status": "validated",
                            "validation_path": current_validation.get(
                                "validation_path"
                            ),
                            "post_validation_status": (
                                current_validation.get("status")
                            ),
                            "post_repair_plan_path": after_plan_path or None,
                            "diagnostic_reconciliation": reconciliation,
                            "scope_validation_completion": (
                                scope_validation_completion
                            ),
                        }
                        write_json(checkpoint_path, checkpoint_result)
                        record(
                            "validate_after_structural_scope_checkpoint",
                            {
                                **checkpoint_result,
                                "checkpoint_path": str(
                                    checkpoint_path.resolve()
                                ),
                            },
                            rounds=validated_repair_rounds_completed,
                        )
                        if current_validation.get("status") == (
                            "ready_for_manual_test"
                        ):
                            return complete_repair_controller(
                                current_validation,
                                stop_reason="compile_validated",
                                extra={
                                    "structural_scope_checkpoint_path": str(
                                        checkpoint_path.resolve()
                                    ),
                                },
                            )
                        if _validation_blocks_auto_repair(
                            current_validation
                        ):
                            return complete_repair_controller(
                                current_validation,
                                stop_reason=(
                                    "validation_environment_blocked_after_"
                                    "structural_scope_checkpoint"
                                ),
                                extra={
                                    "environment_failure": True,
                                    "structural_scope_checkpoint_path": str(
                                        checkpoint_path.resolve()
                                    ),
                                },
                            )
                        continue
            if not bool(active_phase_budget()["available"]):
                break
            if round_plan is None:
                repair_rounds_completed = (
                    validated_compile_repair_rounds_completed
                    + int(
                        repair_lifecycle.budget_status()["completed"]
                    )
                    + 1
                    if repair_lifecycle is not None
                    else validated_repair_rounds_completed + 1
                )
                repair_plan_path = Path(current_validation["repair_plan_path"]).resolve()
                round_source_plan_path = repair_plan_path
                source_round_plan = load_repair_plan(
                    repair_plan_path,
                    state_root=state_root,
                )
                round_execution_phase = (
                    repair_plan_execution_phase(source_round_plan)
                )
                active_execution_phase = (
                    repair_lifecycle.execution_phase
                    if repair_lifecycle is not None
                    else EXECUTION_PHASE_COMPILER_REPAIR
                )
                if active_execution_phase != round_execution_phase:
                    raise ValueError(
                        "repair plan phase does not match the active "
                        "controller phase: "
                        f"{round_execution_phase} != "
                        f"{active_execution_phase}"
                    )
                round_plan, round_pending_plan, batch_selection = (
                    _select_validation_batch_repair_plan(
                        source_round_plan,
                        spec=spec,
                    )
                )
                round_root_dir = ensure_dir(
                    cycle_dir
                    / (
                        f"agentic_repair_round_{repair_rounds_completed}_"
                        f"{run_id}"
                    )
                )
                round_unit_sequence = 0
                round_apply_results = []
                round_validation_targets = []
                round_assessment_targets = []
                round_unit_outcomes = []
                round_applied_units = []
                round_batch_records = [
                    {
                        "selected_work_unit_ids": list(
                            batch_selection.get("selected_work_unit_ids") or []
                        ),
                        "selected_write_files": list(
                            batch_selection.get("selected_write_files") or []
                        ),
                        "deferred_count": int(
                            (batch_selection.get("summary") or {}).get(
                                "deferred_count"
                            )
                            or 0
                        ),
                        "reason": (
                            "dependency_ready_sequential_frontier_with_"
                            "current_tree_context_refresh"
                        ),
                    }
                ]
                round_scope_changes = []
                round_manifest_flushed_counts = {
                    field: 0 for field in ROUND_MANIFEST_COLLECTIONS
                }
                round_manifest_streamed_counts = {
                    "unit_outcomes": 0,
                    "apply_results": 0,
                }
                if repair_lifecycle is not None:
                    lifecycle_round = repair_lifecycle.begin_round(
                        controller_round=repair_rounds_completed,
                    )
                    if lifecycle_round.get("status") == (
                        "budget_exhausted"
                    ):
                        raise ValueError(
                            "repair lifecycle exhausted its budget before "
                            "the canonical compiler round started"
                        )
                    round_batch_records.append(
                        {
                            "repair_lifecycle": lifecycle_round,
                        }
                    )
            else:
                repair_plan_path = (
                    round_source_plan_path
                    or Path(current_validation["repair_plan_path"]).resolve()
                )
            repair_plan = round_plan
            full_round_plan = repair_plan
            round_unit_sequence += 1
            round_dir = ensure_dir(
                (round_root_dir or cycle_dir)
                / f"unit-{round_unit_sequence:04d}"
            )
            graph_expansion_records: list[dict[str, Any]] = []
            selected_plan_unit_id = _next_graph_plan_unit_id(repair_plan)
            selected_round_assessment_target = (
                _validation_target_for_plan_unit(
                    repair_plan,
                    selected_plan_unit_id=selected_plan_unit_id,
                )
                if selected_plan_unit_id
                else None
            )
            if selected_plan_unit_id and round_validation_targets:
                repair_plan = continue_repair_plan_graph(
                    spec,
                    repair_plan,
                    work_unit_ids={selected_plan_unit_id},
                    refresh_current_tree=True,
                )
                refresh_path = (
                    round_dir / "current-tree-graph-refresh.repair_plan.json"
                )
                repair_plan["source_plan_path"] = str(repair_plan_path)
                write_repair_plan(
                    refresh_path,
                    repair_plan,
                    state_root=state_root,
                )
                refresh_record = {
                    "schema": "selected_unit_current_tree_graph_refresh_v1",
                    "selected_plan_unit_id": selected_plan_unit_id,
                    "source_plan_path": str(repair_plan_path),
                    "refreshed_plan_path": str(refresh_path.resolve()),
                    "prior_applied_unit_count": len(round_validation_targets),
                    "graph_continuation": dict(
                        repair_plan.get("graph_continuation") or {}
                    ),
                }
                graph_expansion_records.append(refresh_record)
                record(
                    "agentic_refresh_selected_unit_current_tree_graph",
                    refresh_record,
                    rounds=repair_rounds_completed,
                )
                repair_plan_path = refresh_path
            if _graph_expansion_should_continue(
                repair_plan,
                selected_plan_unit_id=selected_plan_unit_id,
            ):
                graph_page = 0
                while _graph_expansion_should_continue(
                    repair_plan,
                    selected_plan_unit_id=selected_plan_unit_id,
                ):
                    graph_page += 1
                    expanded_plan = continue_repair_plan_graph(
                        spec,
                        repair_plan,
                        work_unit_ids={selected_plan_unit_id},
                    )
                    continuation = dict(expanded_plan.get("graph_continuation") or {})
                    expanded_path = round_dir / f"graph-page-{graph_page:02d}.repair_plan.json"
                    expanded_plan["source_plan_path"] = str(repair_plan_path)
                    write_repair_plan(
                        expanded_path,
                        expanded_plan,
                        state_root=state_root,
                    )
                    page_record = {
                        "page": graph_page,
                        "source_plan_path": str(repair_plan_path),
                        "expanded_plan_path": str(expanded_path.resolve()),
                        "requested_work_unit_ids": [selected_plan_unit_id],
                        "continuation": continuation,
                    }
                    graph_expansion_records.append(page_record)
                    record(
                        f"agentic_expand_graph_page_{graph_page}",
                        page_record,
                        rounds=repair_rounds_completed,
                    )
                    repair_plan = expanded_plan
                    repair_plan_path = expanded_path
                    selected_patch_id = next(
                        (
                            str(unit.get("id") or "")
                            for unit in repair_plan.get(
                                "work_unit_plans"
                            )
                            or []
                            if isinstance(unit, dict)
                            and str(unit.get("id") or "")
                            == selected_plan_unit_id
                            and _graph_plan_unit_patch_ready(unit)
                        ),
                        None,
                    )
                    if selected_patch_id or not _graph_expansion_should_continue(
                        repair_plan,
                        selected_plan_unit_id=selected_plan_unit_id,
                        continuation=continuation,
                    ):
                        break

            if selected_plan_unit_id is None:
                evidence_novelty = dict(
                    (
                        (repair_plan.get("summary") or {}).get(
                            "validation_batch"
                        )
                        or {}
                    ).get("repair_evidence_novelty")
                    or {}
                )
                evidence_exhausted = bool(
                    evidence_novelty.get("exhausted_work_unit_ids")
                    or evidence_novelty.get(
                        "dependency_blocked_work_unit_ids"
                    )
                )
                invariant = {
                    "schema": "dependency_schedule_invariant_v1",
                    "status": (
                        "repair_evidence_exhausted"
                        if evidence_exhausted
                        else "no_actionable_frontier"
                    ),
                    "repair_plan_path": str(repair_plan_path),
                    "work_unit_count": len(
                        repair_plan.get("work_unit_plans") or []
                    ),
                    "graph_expansion_records": graph_expansion_records,
                    "repair_evidence_novelty": evidence_novelty,
                }
                write_json(
                    round_dir / "dependency_schedule_invariant.json",
                    invariant,
                )
                return complete_repair_controller(
                    current_validation,
                    stop_reason=(
                        "repair_evidence_exhausted"
                        if evidence_exhausted
                        else "dependency_schedule_no_actionable_frontier"
                    ),
                    extra={"dependency_schedule": invariant},
                )
            full_round_plan = repair_plan
            repair_plan = _select_single_root_repair_plan(
                repair_plan,
                selected_id=selected_plan_unit_id,
            )
            repair_plan, escalation_authorizations = (
                authorize_repair_plan_escalation(
                    repair_plan,
                    allow_escalation=allow_escalation,
                )
            )
            repair_plan = materialize_deferred_repair_plan_units(
                spec,
                repair_plan,
                work_unit_ids={selected_plan_unit_id},
                refresh=True,
            )
            active_plan_path = round_dir / "active_root.repair_plan.json"
            repair_plan["source_plan_path"] = str(repair_plan_path)
            write_repair_plan(
                active_plan_path,
                repair_plan,
                state_root=state_root,
            )
            repair_plan_path = active_plan_path
            if escalation_authorizations:
                record(
                    "agentic_repair_escalation_authorized",
                    {
                        "repair_plan_path": str(
                            active_plan_path.resolve()
                        ),
                        "selected_plan_unit_id": (
                            selected_plan_unit_id
                        ),
                        "authorizations": (
                            escalation_authorizations
                        ),
                    },
                    rounds=repair_rounds_completed,
                )
            selected_mode = str(
                (
                    (repair_plan.get("work_unit_plans") or [{}])[0].get(
                        "repair_decision"
                    )
                    or {}
                ).get("mode")
                or ""
            )
            if (
                selected_plan_unit_id
                and selected_mode == "materialize_graph_scope"
            ):
                expanded_spec, scope_expansion = materialize_graph_scope(
                    spec,
                    repair_plan,
                    state_root=state_root,
                    cycle=int(current_validation.get("cycle") or 0),
                    work_unit_ids={selected_plan_unit_id},
                    allow_escalation=allow_escalation,
                    execution_phase=round_execution_phase,
                )
                if scope_expansion.get("status") == "expanded":
                    spec = expanded_spec
                    scope_expansion_path = (
                        round_dir / "automatic_graph_scope_expansion.json"
                    )
                    write_json(scope_expansion_path, scope_expansion)
                    record(
                        f"automatic_graph_scope_expansion_round_{repair_rounds_completed}",
                        {
                            **scope_expansion,
                            "result_path": str(scope_expansion_path.resolve()),
                        },
                        rounds=repair_rounds_completed,
                    )
                    scope_change_policy = _scope_change_round_policy(
                        scope_expansion
                    )
                    source_or_build_changed = bool(
                        scope_change_policy["source_or_build_changed"]
                    )
                    if source_or_build_changed:
                        register_round_scope_change(
                            full_plan=full_round_plan,
                            selected_plan_unit_id=selected_plan_unit_id,
                            scope_expansion=scope_expansion,
                            result_path=scope_expansion_path,
                        )
                    compile_context_result = (
                        refresh_compile_context_after_scope_expansion(
                            repair_rounds_completed=repair_rounds_completed,
                            scope_expansion=scope_expansion,
                        )
                    )
                    if compile_context_result.get("status") == "failed":
                        return complete_scope_context_pending(
                            manifest_path=round_dir / "round_manifest.json",
                            full_plan=full_round_plan,
                            selected_plan_unit_id=selected_plan_unit_id,
                            scope_expansion=scope_expansion,
                            compile_context_result=compile_context_result,
                        )
                    if scope_change_policy["force_round_close"]:
                        terminal = advance_after_unit(
                            full_plan=full_round_plan,
                            selected_plan_unit_id=(
                                selected_plan_unit_id
                            ),
                            outcome={
                                "status": (
                                    "scope_expanded_pending_fresh_"
                                    "validation"
                                ),
                                "reason": (
                                    "source_change_validation_barrier"
                                ),
                                "scope_expansion_path": str(
                                    scope_expansion_path.resolve()
                                ),
                                "scope_change_policy": (
                                    scope_change_policy
                                ),
                            },
                            force_round_close=True,
                        )
                        if terminal is not None:
                            return terminal
                        continue
                    refreshed_full_plan = continue_repair_plan_graph(
                        spec,
                        full_round_plan,
                        work_unit_ids={selected_plan_unit_id},
                        refresh_current_tree=True,
                    )
                    refreshed_full_plan = (
                        materialize_deferred_repair_plan_units(
                            spec,
                            refreshed_full_plan,
                            work_unit_ids={selected_plan_unit_id},
                            refresh=True,
                        )
                    )
                    refreshed_path = (
                        round_dir
                        / "post-scope-current-tree.repair_plan.json"
                    )
                    refreshed_full_plan["source_plan_path"] = str(
                        repair_plan_path
                    )
                    write_repair_plan(
                        refreshed_path,
                        refreshed_full_plan,
                        state_root=state_root,
                    )
                    refreshed_selected = _select_single_root_repair_plan(
                        refreshed_full_plan,
                        selected_id=selected_plan_unit_id,
                    )
                    refreshed_mode = str(
                        (
                            (
                                refreshed_selected.get(
                                    "work_unit_plans"
                                )
                                or [{}]
                            )[0].get("repair_decision")
                            or {}
                        ).get("mode")
                        or ""
                    )
                    if refreshed_mode != "materialize_graph_scope":
                        round_plan = refreshed_full_plan
                        persist_round_manifest(
                            (
                                round_root_dir or cycle_dir
                            )
                            / "round_manifest.json",
                            status="scope_expanded_without_validation",
                            remaining_unit_count=len(
                                refreshed_full_plan.get(
                                    "work_unit_plans"
                                )
                                or []
                            ),
                            extra={
                                "scope_expansion_path": str(
                                    scope_expansion_path.resolve()
                                ),
                                "refreshed_plan_path": str(
                                    refreshed_path.resolve()
                                ),
                                "selected_plan_unit_id": (
                                    selected_plan_unit_id
                                ),
                                "source_or_build_changed": (
                                    source_or_build_changed
                                ),
                                "validation": (
                                    "deferred_to_round_end"
                                    if source_or_build_changed
                                    else "not_required"
                                ),
                            },
                        )
                        continue
                    terminal = advance_after_unit(
                        full_plan=refreshed_full_plan,
                        selected_plan_unit_id=selected_plan_unit_id,
                        outcome={
                            "status": "deferred",
                            "reason": (
                                "graph_scope_expanded_but_selected_unit_"
                                "remained_unmaterializable"
                            ),
                            "scope_expansion_path": str(
                                scope_expansion_path.resolve()
                            ),
                            "refreshed_plan_path": str(
                                refreshed_path.resolve()
                            ),
                            "source_or_build_changed": (
                                source_or_build_changed
                            ),
                        },
                    )
                    if terminal is not None:
                        return terminal
                    continue
                scope_expansion_path = (
                    round_dir / "automatic_graph_scope_expansion.json"
                )
                write_json(scope_expansion_path, scope_expansion)
                terminal = advance_after_unit(
                    full_plan=full_round_plan,
                    selected_plan_unit_id=selected_plan_unit_id,
                    outcome=_deferred_scope_materialization_outcome(
                        scope_expansion,
                        result_path=scope_expansion_path,
                    ),
                )
                if terminal is not None:
                    return terminal
                continue
            scope = build_agentic_scope(
                spec,
                repair_plan,
                allow_escalation=allow_escalation,
            )
            write_json(round_dir / "agentic_scope.json", scope)
            record(
                f"agentic_repair_scope_round_{repair_rounds_completed}",
                {
                    "repair_plan_path": str(repair_plan_path),
                    "round_dir": str(round_dir.resolve()),
                    "editable_file_count": len(scope.get("editable_files") or []),
                    "escalation_file_count": len(scope.get("escalation_files") or []),
                    "read_only_evidence_file_count": len(scope.get("read_only_evidence_files") or []),
                    "inspectable_range_count": len(scope.get("inspectable_ranges") or []),
                    "selected_plan_unit_id": selected_plan_unit_id,
                    "graph_expansion_pages": graph_expansion_records,
                    "max_agent_steps": max_agent_steps,
                    "execution_mode": "iterative_graph_repair",
                },
                rounds=repair_rounds_completed,
            )
            static_materialization_request = (
                unique_agentic_entity_materialization_request(scope)
            )
            if (
                not scope.get("editable_files")
                and not (
                    allow_escalation
                    and scope.get("escalation_files")
                )
                and static_materialization_request is None
            ):
                terminal = advance_after_unit(
                    full_plan=full_round_plan,
                    selected_plan_unit_id=str(selected_plan_unit_id or ""),
                    outcome={
                        "status": "deferred",
                        "reason": (
                            "code_graph_scope_unresolved"
                            if _graph_incomplete_plan_unit_ids(repair_plan)
                            else "no_editable_scope"
                        ),
                        "agentic_scope_path": str((round_dir / "agentic_scope.json").resolve()),
                        "graph_expansion_pages": graph_expansion_records,
                    },
                )
                if terminal is not None:
                    return terminal
                continue

            static_materialization: dict[str, Any] | None = None
            static_materialization_rejection: dict[str, Any] | None = None
            if static_materialization_request is not None:
                try:
                    static_materialization = (
                        build_agentic_entity_materialization_action(
                            spec,
                            action=static_materialization_request,
                            scope=scope,
                        )
                    )
                except ValueError as exc:
                    representation_authority = (
                        exc.authority
                        if isinstance(
                            exc,
                            EntityRepresentationAdaptationRequired,
                        )
                        else None
                    )
                    if representation_authority is not None:
                        selected_unit_id = str(
                            (
                                static_materialization_request.get(
                                    "decision"
                                )
                                or {}
                            ).get("work_unit_id")
                            or ""
                        )
                        for graph_unit in scope.get("graph_units") or []:
                            if not isinstance(graph_unit, dict):
                                continue
                            if selected_unit_id not in {
                                str(graph_unit.get("plan_unit_id") or ""),
                                str(graph_unit.get("work_unit_id") or ""),
                            }:
                                continue
                            graph_unit[
                                "target_representation_adaptation_authority"
                            ] = representation_authority
                        write_json(round_dir / "agentic_scope.json", scope)
                    static_materialization_rejection = {
                        **agentic_materialization_rejection(
                            exc, action=static_materialization_request, step=0, static=True,
                        ),
                        **(
                            {
                                "target_representation_adaptation_authority": (
                                    representation_authority
                                )
                            }
                            if representation_authority is not None
                            else {}
                        ),
                    }
                    write_json(
                        round_dir
                        / "static_entity_materialization_rejection.json",
                        static_materialization_rejection,
                    )
                else:
                    write_json(
                        round_dir / "static_entity_materialization_request.json",
                        static_materialization_request,
                    )
                    write_json(
                        round_dir / "static_entity_materialization.json",
                        static_materialization,
                    )

            evidence_log: list[dict[str, Any]] = []
            if static_materialization_rejection is not None:
                evidence_log.append(
                    static_materialization_rejection
                )
            step_apply_failure_seen = False
            last_apply_failure_evidence: dict[str, Any] | None = None
            pending_controller_action: dict[str, Any] | None = None
            step = 0
            unit_agent_step_limit = max_agent_steps
            while (
                step < unit_agent_step_limit
                or pending_controller_action is not None
            ):
                step += 1
                patch_outcome_candidate: dict[str, Any] | None = None
                announce(
                    f"agentic_repair_step_{step}",
                    rounds=repair_rounds_completed,
                )
                use_static_materialization = bool(
                    step == 1
                    and static_materialization_request is not None
                    and static_materialization is not None
                )
                controller_action = pending_controller_action
                pending_controller_action = None
                deterministic_action = (
                    controller_action
                    or (
                        static_materialization_request
                        if use_static_materialization
                        else None
                    )
                )
                prompt_text = (
                    json.dumps(
                        {
                            "execution_mode": (
                                "deterministic_controller_recovery"
                                if controller_action is not None
                                else "deterministic_graph_entity_materialization"
                            ),
                            "request": deterministic_action,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    if deterministic_action is not None
                    else build_agentic_prompt(
                        spec,
                        validation_report=current_validation,
                        repair_plan=repair_plan,
                        scope=scope,
                        evidence_log=evidence_log,
                        repair_round=repair_rounds_completed,
                        step=step,
                        max_steps=(
                            unit_agent_step_limit
                        ),
                    )
                )
                prompt_path = round_dir / f"step-{step:02d}.prompt.txt"
                response_path = round_dir / f"step-{step:02d}.response.txt"
                metadata_path = round_dir / f"step-{step:02d}.response.meta.json"
                prompt_evidence_metadata = (
                    {
                        "schema": scope.get("prompt_evidence_schema"),
                        "evidence_policy": dict(
                            scope.get("prompt_evidence_policy") or {}
                        ),
                        "metrics": dict(
                            scope.get("prompt_evidence_metrics") or {}
                        ),
                    }
                    if deterministic_action is None
                    else {
                        "schema": "deterministic_graph_tool_v1",
                        "evidence_policy": {
                            "required_evidence": "not_applicable"
                        },
                        "metrics": {},
                    }
                )
                write_json(
                    round_dir
                    / f"step-{step:02d}.prompt_evidence_metrics.json",
                    prompt_evidence_metadata,
                )
                prompt_path.write_text(prompt_text, encoding="utf-8")
                try:
                    if deterministic_action is not None:
                        response_text = json.dumps(
                            deterministic_action,
                            ensure_ascii=False,
                        )
                        model_call = {
                            "model": None,
                            "api_mode": (
                                "deterministic_controller_recovery"
                                if controller_action is not None
                                else "deterministic_graph_tool"
                            ),
                            "response_id": None,
                            "retry_count": 0,
                            "usage": {
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0,
                            },
                            "prompt_size_bytes": len(
                                prompt_text.encode("utf-8")
                            ),
                        }
                    else:
                        response_text, model_call = (
                            _run_agentic_model_step(prompt_text)
                        )
                except Exception as exc:  # pragma: no cover - runtime/model dependent.
                    error_metadata = model_api_error_metadata(exc)
                    consecutive_model_failures += 1
                    disposition = _model_failure_disposition(
                        error_metadata,
                        consecutive_failures=consecutive_model_failures,
                        max_consecutive_failures=(
                            max_consecutive_model_failures
                        ),
                    )
                    model_error_path = round_dir / f"step-{step:02d}.model_error.json"
                    model_error = {
                        "schema": "agentic_model_error_v1",
                        "round": repair_rounds_completed,
                        "step": step,
                        "error": str(exc),
                        **error_metadata,
                        "disposition": disposition,
                        "prompt_path": str(prompt_path.resolve()),
                    }
                    write_json(model_error_path, model_error)
                    record(
                        f"agentic_model_step_failed_{step}",
                        {**model_error, "model_error_path": str(model_error_path.resolve())},
                        rounds=repair_rounds_completed,
                    )
                    stop_reason = (
                        f"agentic_model_api_unavailable_round_"
                        f"{repair_rounds_completed}"
                        if disposition["stop_after_validation"]
                        else None
                    )
                    terminal = advance_after_unit(
                        full_plan=full_round_plan,
                        selected_plan_unit_id=str(
                            selected_plan_unit_id or ""
                        ),
                        outcome={
                            "status": "deferred",
                            "reason": disposition["reason"],
                            "step": step,
                            "model_error": model_error,
                            "model_error_path": str(model_error_path.resolve()),
                        },
                        force_round_close=bool(
                            disposition["stop_after_validation"]
                        ),
                        terminal_stop_reason=stop_reason,
                        terminal_extra={
                            "model_error": model_error,
                            "model_error_path": str(
                                model_error_path.resolve()
                            ),
                            "model_failure_disposition": disposition,
                        },
                    )
                    if terminal is not None:
                        return terminal
                    break
                consecutive_model_failures = 0
                response_path.write_text(response_text, encoding="utf-8")
                model_call["prompt_evidence"] = prompt_evidence_metadata
                write_json(metadata_path, model_call)
                try:
                    action = parse_agentic_action(response_text)
                except Exception as exc:
                    parse_error = {
                        "error": str(exc),
                        "response_path": str(response_path.resolve()),
                    }
                    parse_error_path = (
                        round_dir / f"step-{step:02d}.parse_error.json"
                    )
                    write_json(parse_error_path, parse_error)
                    evidence_log.append(
                        {
                            "kind": "invalid_agentic_response",
                            "step": step,
                            "error": str(exc),
                            "response_excerpt": response_text[:4000],
                        }
                    )
                    continue
                if controller_action is not None:
                    action["_controller_provenance"] = (
                        "typed_apply_failure_recovery"
                    )
                write_json(round_dir / f"step-{step:02d}.action.json", action)
                record(
                    f"agentic_repair_step_{step}",
                    {
                        "prompt_path": str(prompt_path.resolve()),
                        "response_path": str(response_path.resolve()),
                        "metadata_path": str(metadata_path.resolve()),
                        "action": action.get("action"),
                        "rationale": action.get("rationale"),
                    },
                    rounds=repair_rounds_completed,
                )
                if action["action"] == "materialize_entity_from_donor":
                    try:
                        materialization = (
                            static_materialization
                            if use_static_materialization
                            and static_materialization is not None
                            else build_agentic_entity_materialization_action(
                                spec,
                                action=action,
                                scope=scope,
                            )
                        )
                    except ValueError as exc:
                        rejection = agentic_materialization_rejection(
                            exc, action=action, step=step,
                        )
                        write_json(
                            round_dir
                            / f"step-{step:02d}.entity_materialization_rejection.json",
                            rejection,
                        )
                        evidence_log.append(rejection)
                        continue
                    if not use_static_materialization:
                        write_json(
                            round_dir
                            / f"step-{step:02d}.entity_materialization.json",
                            materialization,
                        )
                    evidence_log.append(
                        {
                            "kind": "inspect_result",
                            "step": step,
                            "source": "graph_entity_materialization",
                            "evidence": materialization["evidence"],
                        }
                    )
                    action = dict(materialization["action"])
                    action["_controller_provenance"] = (
                        "deterministic_graph_entity_materialization"
                    )
                    write_json(
                        round_dir
                        / f"step-{step:02d}.entity_materialization_action.json",
                        action,
                    )

                if action["action"] == "apply_patch":
                    apply_redundancy = agentic_apply_redundancy(
                        action,
                        evidence_log,
                    )
                    if not apply_redundancy.get("redundant"):
                        apply_redundancy = agentic_prior_apply_redundancy(
                            action,
                            repair_plan,
                            evidence_log,
                        )
                    if apply_redundancy.get("redundant"):
                        recovery = collect_agentic_recovery_evidence(
                            spec,
                            action=action,
                            repair_plan=repair_plan,
                            scope=scope,
                            evidence_log=evidence_log,
                            redundancy=apply_redundancy,
                        )
                        recovery_record = {
                            "kind": "controller_recovery_result",
                            "step": step,
                            "trigger": "redundant_apply_action",
                            **recovery,
                        }
                        write_json(
                            round_dir
                            / f"step-{step:02d}.repeated-apply-recovery.json",
                            recovery_record,
                        )
                        evidence_log.append(recovery_record)
                        if recovery.get("status") == "recovered_action":
                            action = dict(
                                recovery.get("corrected_action") or {}
                            )
                            write_json(
                                round_dir
                                / f"step-{step:02d}.recovered_action.json",
                                action,
                            )
                        elif recovery.get("status") == "recovered_evidence":
                            continue
                        else:
                            rejection = {
                                "kind": "repeated_blocked_apply_rejected",
                                "step": step,
                                "reason": (
                                    "the identical patch action remained blocked "
                                    "after bounded static recovery"
                                ),
                                "redundancy": apply_redundancy,
                                "recovery": recovery,
                                "source_evidence_fingerprint": (
                                    agentic_source_evidence_fingerprint(
                                        dict(
                                            (
                                                repair_plan.get(
                                                    "work_unit_plans"
                                                )
                                                or [{}]
                                            )[0]
                                        )
                                    )
                                ),
                                "action": action,
                            }
                            write_json(
                                round_dir
                                / f"step-{step:02d}.repeated_blocked_apply.json",
                                rejection,
                            )
                            evidence_log.append(rejection)
                            if recovery.get("status") == "recovery_exhausted":
                                terminal = advance_after_unit(
                                    full_plan=full_round_plan,
                                    selected_plan_unit_id=str(
                                        selected_plan_unit_id or ""
                                    ),
                                    outcome={
                                        "status": "deferred",
                                        "reason": (
                                            "unchanged_redundant_apply_blocker"
                                        ),
                                        "step": step,
                                        "unit_dir": str(round_dir.resolve()),
                                        "recovery": recovery_record,
                                        "rejection": rejection,
                                    },
                                )
                                if terminal is not None:
                                    return terminal
                                break
                            continue

                    patch_outcome_candidate = (
                        _patch_outcome_candidate_for_action(
                            spec,
                            action=action,
                            repair_plan=repair_plan,
                        )
                    )
                    if patch_outcome_candidate is not None:
                        prior_rejection = rejected_patch_outcome(
                            state_root,
                            patch_outcome_candidate,
                        )
                        if prior_rejection is not None:
                            inverse_rejection = (
                                prior_rejection.get("rejection_kind")
                                == "inverse_of_productive_patch"
                            )
                            rejection = {
                                "kind": "validated_regressive_patch_rejected",
                                "step": step,
                                "reason": (
                                    (
                                        "this patch reverses a source delta "
                                        "that already advanced the same "
                                        "diagnostic root"
                                    )
                                    if inverse_rejection
                                    else (
                                        "the same patch was already validated "
                                        "as unproductive against the same "
                                        "source, diagnostic, and graph-root "
                                        "state"
                                    )
                                ),
                                "candidate": patch_outcome_candidate,
                                "prior_outcome": prior_rejection,
                                "required_next_action": (
                                    "produce a different patch from the recorded "
                                    "validation evidence, inspect unresolved graph "
                                    "dependencies, or expand scope"
                                ),
                                "action": action,
                            }
                            write_json(
                                round_dir
                                / f"step-{step:02d}.validated_patch_rejection.json",
                                rejection,
                            )
                            evidence_log.append(rejection)
                            continue

                if action["action"] == "inspect":
                    redundancy = agentic_inspect_redundancy(action, evidence_log)
                    if step > 1 and redundancy.get("redundant"):
                        rejection = {
                            "kind": "redundant_inspect_rejected",
                            "step": step,
                            "reason": (
                                "inspect request only repeated graph-authorized file ranges already present "
                                "in evidence_log; next action should be apply_patch, needs_escalation, or stop"
                            ),
                            "redundancy": redundancy,
                            "action": action,
                        }
                        write_json(round_dir / f"step-{step:02d}.redundant_inspect.json", rejection)
                        evidence_log.append(rejection)
                        continue
                    evidence = collect_agentic_evidence(
                        spec,
                        action=action,
                        scope=scope,
                        validation_report=current_validation,
                    )
                    evidence_path = round_dir / f"step-{step:02d}.evidence.json"
                    write_json(evidence_path, evidence)
                    evidence_log.append(
                        {
                            "kind": "inspect_result",
                            "step": step,
                            "evidence_path": str(evidence_path.resolve()),
                            "summary": {
                                "file_records": len(evidence.get("files") or []),
                                "truncated": bool(evidence.get("truncated")),
                            },
                            "evidence": evidence,
                        }
                    )
                    continue

                if action["action"] == "expand_scope":
                    try:
                        expansion_request = graph_scope_expansion_request(
                            action,
                            repair_plan,
                        )
                        target_plan_unit_id = expansion_request["plan_unit_id"]
                        if expansion_request["mode"] == "continue_graph":
                            expanded_plan = continue_repair_plan_graph(
                                spec,
                                repair_plan,
                                work_unit_ids={target_plan_unit_id},
                            )
                        else:
                            expanded_spec, scope_expansion = materialize_graph_scope(
                                spec,
                                repair_plan,
                                state_root=state_root,
                                cycle=int(current_validation.get("cycle") or 0),
                                work_unit_ids={target_plan_unit_id},
                                allow_escalation=allow_escalation,
                                execution_phase=round_execution_phase,
                            )
                            if scope_expansion.get("status") != "expanded":
                                raise ValueError(
                                    "the graph owner could not be materialized from the current donor trees"
                                )
                            spec = expanded_spec
                            materialization_path = (
                                round_dir
                                / f"step-{step:02d}.scope_materialization.json"
                            )
                            write_json(materialization_path, scope_expansion)
                            record(
                                f"agentic_materialize_graph_scope_step_{step}",
                                {
                                    **scope_expansion,
                                    "result_path": str(
                                        materialization_path.resolve()
                                    ),
                                },
                                rounds=repair_rounds_completed,
                            )
                            scope_change_policy = (
                                _scope_change_round_policy(scope_expansion)
                            )
                            source_or_build_changed = bool(
                                scope_change_policy["source_or_build_changed"]
                            )
                            if source_or_build_changed:
                                register_round_scope_change(
                                    full_plan=full_round_plan,
                                    selected_plan_unit_id=(
                                        target_plan_unit_id
                                    ),
                                    scope_expansion=scope_expansion,
                                    result_path=materialization_path,
                                )
                            compile_context_result = (
                                refresh_compile_context_after_scope_expansion(
                                    repair_rounds_completed=repair_rounds_completed,
                                    scope_expansion=scope_expansion,
                                )
                            )
                            if compile_context_result.get("status") == "failed":
                                return complete_scope_context_pending(
                                    manifest_path=(
                                        round_dir / "round_manifest.json"
                                    ),
                                    full_plan=full_round_plan,
                                    selected_plan_unit_id=(
                                        target_plan_unit_id
                                    ),
                                    scope_expansion=scope_expansion,
                                    compile_context_result=(
                                        compile_context_result
                                    ),
                                )
                            if scope_change_policy[
                                "force_round_close"
                            ]:
                                terminal = advance_after_unit(
                                    full_plan=full_round_plan,
                                    selected_plan_unit_id=(
                                        target_plan_unit_id
                                    ),
                                    outcome={
                                        "status": (
                                            "scope_expanded_pending_"
                                            "fresh_validation"
                                        ),
                                        "reason": (
                                            "source_change_validation_"
                                            "barrier"
                                        ),
                                        "step": step,
                                        "scope_expansion_path": str(
                                            materialization_path.resolve()
                                        ),
                                        "scope_change_policy": (
                                            scope_change_policy
                                        ),
                                    },
                                    force_round_close=True,
                                )
                                if terminal is not None:
                                    return terminal
                                break
                            expanded_plan = continue_repair_plan_graph(
                                spec,
                                full_round_plan,
                                work_unit_ids={target_plan_unit_id},
                                refresh_current_tree=True,
                            )
                            (
                                expanded_plan,
                                materialized_escalation_authorizations,
                            ) = authorize_repair_plan_escalation(
                                expanded_plan,
                                allow_escalation=allow_escalation,
                            )
                            expanded_plan = (
                                materialize_deferred_repair_plan_units(
                                    spec,
                                    expanded_plan,
                                    work_unit_ids={target_plan_unit_id},
                                    refresh=True,
                                )
                            )
                            expanded_path = (
                                round_dir
                                / f"step-{step:02d}.expanded_repair_plan.json"
                            )
                            expanded_plan["source_plan_path"] = str(
                                repair_plan_path
                            )
                            write_repair_plan(
                                expanded_path,
                                expanded_plan,
                                state_root=state_root,
                            )
                            repair_plan = _select_single_root_repair_plan(
                                expanded_plan,
                                selected_id=target_plan_unit_id,
                            )
                            repair_plan_path = expanded_path
                            scope = build_agentic_scope(
                                spec,
                                repair_plan,
                                allow_escalation=allow_escalation,
                            )
                            write_json(
                                round_dir
                                / f"step-{step:02d}.expanded_scope.json",
                                scope,
                            )
                            expansion_evidence = {
                                "kind": (
                                    "graph_scope_materialization_result"
                                ),
                                "step": step,
                                "plan_unit_id": target_plan_unit_id,
                                "repair_plan_path": str(
                                    expanded_path.resolve()
                                ),
                                "scope_expansion_path": str(
                                    materialization_path.resolve()
                                ),
                                "validation": (
                                    "deferred_to_round_end"
                                    if source_or_build_changed
                                    else "not_required"
                                ),
                                "graph_units": (
                                    scope.get("graph_units") or []
                                ),
                                "escalation_authorizations": (
                                    materialized_escalation_authorizations
                                ),
                            }
                            evidence_log.append(expansion_evidence)
                            record(
                                f"agentic_refresh_scope_step_{step}",
                                expansion_evidence,
                                rounds=repair_rounds_completed,
                            )
                    except (OSError, ValueError) as exc:
                        rejection = {
                            "kind": "invalid_graph_scope_expansion",
                            "step": step,
                            "error": str(exc),
                            "action": action,
                        }
                        write_json(
                            round_dir / f"step-{step:02d}.scope_expansion_rejection.json",
                            rejection,
                        )
                        evidence_log.append(rejection)
                        continue

                    if expansion_request["mode"] != "continue_graph":
                        continue

                    continuation_result = dict(expanded_plan.get("graph_continuation") or {})
                    expanded_plan, step_escalation_authorizations = (
                        authorize_repair_plan_escalation(
                            expanded_plan,
                            allow_escalation=allow_escalation,
                        )
                    )
                    expanded_plan = materialize_deferred_repair_plan_units(
                        spec,
                        expanded_plan,
                        work_unit_ids={target_plan_unit_id},
                        refresh=bool(
                            step_escalation_authorizations
                        ),
                    )
                    expanded_path = round_dir / f"step-{step:02d}.expanded_repair_plan.json"
                    expanded_plan["source_plan_path"] = str(repair_plan_path)
                    write_repair_plan(
                        expanded_path,
                        expanded_plan,
                        state_root=state_root,
                    )
                    repair_plan = expanded_plan
                    repair_plan_path = expanded_path
                    scope = build_agentic_scope(
                        spec,
                        repair_plan,
                        allow_escalation=allow_escalation,
                    )
                    write_json(round_dir / f"step-{step:02d}.expanded_scope.json", scope)
                    expansion_evidence = {
                        "kind": "graph_scope_expansion_result",
                        "step": step,
                        "plan_unit_id": target_plan_unit_id,
                        "repair_plan_path": str(expanded_path.resolve()),
                        "progressed": bool(continuation_result.get("progressed_count")),
                        "continuation": continuation_result,
                        "graph_units": scope.get("graph_units") or [],
                        "escalation_authorizations": (
                            step_escalation_authorizations
                        ),
                    }
                    evidence_log.append(expansion_evidence)
                    record(
                        f"agentic_expand_scope_step_{step}",
                        expansion_evidence,
                        rounds=repair_rounds_completed,
                    )
                    continue

                if action["action"] == "stop":
                    terminal = advance_after_unit(
                        full_plan=full_round_plan,
                        selected_plan_unit_id=str(selected_plan_unit_id or ""),
                        outcome={
                            "status": "deferred",
                            "reason": "agent_stopped_unit",
                            "step": step,
                            "action": action,
                        },
                    )
                    if terminal is not None:
                        return terminal
                    break

                if action["action"] == "needs_escalation":
                    try:
                        graph_escalation_target(action, repair_plan)
                    except ValueError as exc:
                        rejection = {
                            "kind": "invalid_graph_escalation",
                            "step": step,
                            "error": str(exc),
                            "action": action,
                        }
                        write_json(
                            round_dir / f"step-{step:02d}.escalation_rejection.json",
                            rejection,
                        )
                        evidence_log.append(rejection)
                        continue
                    terminal = advance_after_unit(
                        full_plan=full_round_plan,
                        selected_plan_unit_id=str(selected_plan_unit_id or ""),
                        outcome={
                            "status": "deferred",
                            "reason": "agent_needs_escalation",
                            "step": step,
                            "action": action,
                            "unit_dir": str(round_dir.resolve()),
                        },
                    )
                    if terminal is not None:
                        return terminal
                    break

                evidence_gate = agentic_evidence_ready_for_apply(
                    evidence_log,
                    spec=spec,
                    action=action,
                    scope=scope,
                )
                if not evidence_gate.get("ready"):
                    rejection = {
                        "kind": "apply_rejected_missing_requested_evidence",
                        "step": step,
                        "evidence_gate": evidence_gate,
                        "action": action,
                        **agentic_blocked_apply_metadata(
                            action,
                            evidence_log,
                        ),
                    }
                    write_json(
                        round_dir / f"step-{step:02d}.evidence_gate_rejection.json",
                        rejection,
                    )
                    evidence_log.append(rejection)
                    continue

                try:
                    agentic_plan = build_agentic_apply_plan(
                        spec,
                        action=action,
                        repair_plan=repair_plan,
                        scope=scope,
                        repair_round=repair_rounds_completed,
                        step=step,
                        evidence_log=evidence_log,
                    )
                    if patch_outcome_candidate is not None:
                        agentic_plan["work_unit_plans"][0][
                            "patch_outcome_candidate"
                        ] = patch_outcome_candidate
                except ValueError as exc:
                    rejection = {
                        "kind": "invalid_graph_patch_decision",
                        "step": step,
                        "error": str(exc),
                        "action": action,
                        **agentic_blocked_apply_metadata(
                            action,
                            evidence_log,
                        ),
                    }
                    write_json(round_dir / f"step-{step:02d}.graph_patch_rejection.json", rejection)
                    evidence_log.append(rejection)
                    continue
                agentic_plan_path = round_dir / f"step-{step:02d}.apply_plan.json"
                write_repair_plan(
                    agentic_plan_path,
                    agentic_plan,
                    state_root=state_root,
                )
                unit_id = str(agentic_plan["work_unit_plans"][0]["id"])
                inflight_validation_target = _validation_target_with_apply_evidence(
                    dict(agentic_plan["work_unit_plans"][0]),
                    apply_result=None,
                    baseline_validation=current_validation,
                )
                inflight_validation_target = (
                    _merge_round_assessment_target(
                        selected_round_assessment_target,
                        inflight_validation_target,
                    )
                    or inflight_validation_target
                )
                inflight_validation_target["repair_applied"] = False
                response_dir = ensure_dir(round_dir / f"step-{step:02d}.responses")
                response_action = {
                    key: value
                    for key, value in action.items()
                    if not str(key).startswith("_controller_")
                }
                (response_dir / f"{unit_id}.response.txt").write_text(
                    json.dumps(response_action, ensure_ascii=False, indent=2)
                    + "\n",
                    encoding="utf-8",
                )
                inflight_manifest_path = (
                    (round_root_dir or cycle_dir) / "round_manifest.json"
                )
                persist_round_manifest(
                    inflight_manifest_path,
                    status="applying_unit",
                    remaining_unit_count=len(
                        full_round_plan.get("work_unit_plans") or []
                    ),
                    extra={
                        "inflight_plan_unit_id": selected_plan_unit_id,
                        "inflight_agentic_unit_id": unit_id,
                        "inflight_apply_plan_path": str(
                            agentic_plan_path.resolve()
                        ),
                        "inflight_validation_target": inflight_validation_target,
                    },
                )
                apply_result = _route_agentic_apply_failures(
                    _apply_plan_responses(
                        spec,
                        state_root=state_root,
                        plan_path=agentic_plan_path,
                        response_dir=response_dir,
                        allow_escalation=allow_escalation,
                        output=None,
                    )
                )
                record(
                    f"agentic_apply_step_{step}",
                    apply_result,
                    rounds=repair_rounds_completed,
                )
                if _apply_summary_has_effect(apply_result):
                    root_overlap = _round_root_overlap_after_apply(
                        full_round_plan,
                        completed_plan_unit_id=str(
                            selected_plan_unit_id or ""
                        ),
                        changed_files=_applied_changed_files(apply_result),
                        added_contract_definitions=(
                            _applied_added_contract_definitions(apply_result)
                        ),
                        old_root=spec.old_root_path,
                    )
                    completed_validation_target = _validation_target_with_apply_evidence(
                        inflight_validation_target,
                        apply_result=apply_result,
                        baseline_validation=current_validation,
                    )
                    terminal = advance_after_unit(
                        full_plan=full_round_plan,
                        selected_plan_unit_id=str(selected_plan_unit_id or ""),
                        outcome={
                            "status": "patch_applied_pending_round_validation",
                            "step": step,
                            "unit_dir": str(round_dir.resolve()),
                            "apply_plan_path": str(agentic_plan_path.resolve()),
                            "transaction_dirs": [
                                item.get("transaction_dir")
                                for item in apply_result.get("results") or []
                                if item.get("transaction_dir")
                            ],
                            "root_overlap": root_overlap,
                        },
                        apply_result=apply_result,
                        validation_target=completed_validation_target,
                        deferred_plan_unit_ids=list(
                            root_overlap.get(
                                "deferred_until_replan_plan_unit_ids"
                            )
                            or []
                        ),
                    )
                    if terminal is not None:
                        return terminal
                    break

                already_present_pending_validation = (
                    _already_present_patch_can_join_round_validation(
                        apply_result,
                        has_unvalidated_round_changes=bool(
                            round_validation_targets
                        ),
                    )
                )
                if already_present_pending_validation:
                    completed_validation_target = (
                        _validation_target_with_apply_evidence(
                            inflight_validation_target,
                            apply_result=apply_result,
                            baseline_validation=current_validation,
                        )
                    )
                    completed_validation_target["repair_applied"] = False
                    completed_validation_target[
                        "validation_reason"
                    ] = "postimage_present_after_unvalidated_round_changes"
                    terminal = advance_after_unit(
                        full_plan=full_round_plan,
                        selected_plan_unit_id=str(
                            selected_plan_unit_id or ""
                        ),
                        outcome={
                            "status": (
                                "patch_postimage_present_pending_"
                                "round_validation"
                            ),
                            "step": step,
                            "unit_dir": str(round_dir.resolve()),
                            "apply_plan_path": str(
                                agentic_plan_path.resolve()
                            ),
                        },
                        apply_result=apply_result,
                        validation_target=completed_validation_target,
                    )
                    if terminal is not None:
                        return terminal
                    break

                step_apply_failure_seen = True
                compact_failures, apply_failure_routes = (
                    _compact_agentic_apply_failures(apply_result)
                )
                failure_evidence = {
                    "kind": "apply_without_validation_worthy_change",
                    "step": step,
                    **agentic_blocked_apply_metadata(
                        action,
                        evidence_log,
                    ),
                    "apply_summary": {
                        "status": apply_result.get("status"),
                        "applied_count": apply_result.get("applied_count"),
                        "incomplete_count": apply_result.get(
                            "incomplete_count"
                        ),
                        "needs_escalation_count": apply_result.get(
                            "needs_escalation_count"
                        ),
                        "routes": apply_failure_routes,
                        "results": compact_failures,
                    },
                }
                evidence_log.append(failure_evidence)
                last_apply_failure_evidence = failure_evidence
                if "outer_unit_replan" in apply_failure_routes:
                    recovery = collect_agentic_recovery_evidence(
                        spec,
                        action=action,
                        repair_plan=repair_plan,
                        scope=scope,
                        evidence_log=evidence_log,
                        redundancy={
                            **agentic_blocked_apply_metadata(
                                action,
                                evidence_log,
                            ),
                            "previous_blocker_kind": (
                                "apply_without_validation_worthy_change"
                            ),
                        },
                    )
                    recovery_record = {
                        "kind": "controller_recovery_result",
                        "step": step,
                        "trigger": "typed_outer_unit_replan_route",
                        **recovery,
                    }
                    write_json(
                        round_dir
                        / f"step-{step:02d}."
                        "outer-unit-replan-recovery.json",
                        recovery_record,
                    )
                    evidence_log.append(recovery_record)
                    if recovery.get("status") == "recovered_action":
                        pending_controller_action = dict(
                            recovery.get("corrected_action") or {}
                        )
                        write_json(
                            round_dir
                            / f"step-{step:02d}.recovered_action.json",
                            pending_controller_action,
                        )
                    elif recovery.get("status") == "recovery_exhausted":
                        terminal = advance_after_unit(
                            full_plan=full_round_plan,
                            selected_plan_unit_id=str(
                                selected_plan_unit_id or ""
                            ),
                            outcome={
                                "status": "deferred",
                                "reason": (
                                    "unchanged_typed_apply_blocker"
                                ),
                                "step": step,
                                "unit_dir": str(round_dir.resolve()),
                                "apply_plan_path": str(
                                    agentic_plan_path.resolve()
                                ),
                                "apply_failure": failure_evidence,
                                "recovery": recovery_record,
                            },
                        )
                        if terminal is not None:
                            return terminal
                        break
                terminal_apply_route = _terminal_agentic_apply_route(
                    apply_failure_routes
                )
                if terminal_apply_route is not None:
                    terminal = advance_after_unit(
                        full_plan=full_round_plan,
                        selected_plan_unit_id=str(
                            selected_plan_unit_id or ""
                        ),
                        outcome={
                            "status": "deferred",
                            "reason": terminal_apply_route,
                            "step": step,
                            "unit_dir": str(round_dir.resolve()),
                            "apply_plan_path": str(
                                agentic_plan_path.resolve()
                            ),
                            "apply_failure": failure_evidence,
                        },
                        force_round_close=True,
                        terminal_stop_reason=(
                            "agentic_apply_infrastructure_failure_round_"
                            f"{repair_rounds_completed}"
                        ),
                        terminal_extra={
                            "apply_failure": failure_evidence,
                        },
                    )
                    if terminal is not None:
                        return terminal
                    break
            else:
                deferred_evidence = latest_agentic_failure_delta(
                    evidence_log
                )
                terminal = advance_after_unit(
                    full_plan=full_round_plan,
                    selected_plan_unit_id=str(selected_plan_unit_id or ""),
                    outcome={
                        "status": "deferred",
                        "reason": (
                            "apply_without_validation_worthy_change"
                            if step_apply_failure_seen
                            else "max_steps_without_patch"
                        ),
                        "unit_dir": str(round_dir.resolve()),
                        **(
                            {"blocker": deferred_evidence}
                            if deferred_evidence is not None
                            else {}
                        ),
                        **(
                            {
                                "apply_failure": (
                                    last_apply_failure_evidence
                                )
                            }
                            if last_apply_failure_evidence is not None
                            else {}
                        ),
                    },
                )
                if terminal is not None:
                    return terminal

        final_budget = active_phase_budget()
        if current_validation["status"] == "ready_for_manual_test":
            return complete_repair_controller(
                current_validation,
                stop_reason="compile_validated",
            )
        stop_reason = (
            current_validation["status"]
            if current_validation["status"] != "needs_patching"
            else "max_semantic_repair_rounds_reached"
            if repair_lifecycle is not None
            else "max_compile_repair_rounds_reached"
        )
        return complete_repair_controller(
            current_validation,
            stop_reason=stop_reason,
            extra={"active_repair_budget": final_budget},
        )


    def continue_semantic_compile_recovery(
        recovery_validation: dict[str, Any],
        *,
        completed_rounds: int,
        semantic_handoff: dict[str, Any] | None = None,
        semantic_closure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        handoff_context = dict(semantic_handoff or {})
        closure_context = dict(semantic_closure or {})

        def durable_case_status() -> str:
            return str(load_or_init_state(spec, state_root).status)

        if (
            recovery_validation.get("status") != "needs_patching"
            or not recovery_validation.get("repair_plan_path")
        ):
            return finish(
                status=durable_case_status(),
                stop_reason="semantic_compile_recovery_handoff_invalid",
                repair_rounds_completed=completed_rounds,
                extra={
                    "final_compile_validated": False,
                    "semantic_complete": False,
                    "semantic_handoff": handoff_context,
                    "semantic_closure": closure_context,
                    "recovery_validation": recovery_validation,
                },
            )

        lifecycle = SemanticCompileRecoveryLifecycle(
            spec,
            state_root=state_root,
            max_rounds=max_semantic_repair_rounds,
        )
        announce(
            "canonical_semantic_compile_recovery",
            rounds=completed_rounds,
        )
        recovery_budget_before = lifecycle.budget_status()
        recovery_result = continue_agentic_repair_from_validation(
            recovery_validation,
            completed_rounds=(
                validated_repair_round_count(state_root)
            ),
            repair_lifecycle=lifecycle,
        )
        if recovery_result.get("schema") != (
            CANONICAL_REPAIR_CONTROLLER_RESULT_SCHEMA
        ):
            return finish(
                status=durable_case_status(),
                stop_reason=(
                    "semantic_compile_recovery_result_invalid"
                ),
                repair_rounds_completed=(
                    validated_repair_round_count(state_root)
                ),
                extra={
                    "compile_repair_validated": True,
                    "final_compile_validated": False,
                    "semantic_complete": False,
                    "compile_recovery_result": recovery_result,
                },
            )
        settlement = lifecycle.settle_controller_exit(recovery_result)
        record(
            "canonical_semantic_compile_recovery_settlement",
            {
                "recovery_result": recovery_result,
                "settlement": settlement,
            },
            rounds=validated_repair_round_count(state_root),
        )

        def invalid_settlement() -> dict[str, Any]:
            return finish(
                status=durable_case_status(),
                stop_reason=(
                    "semantic_compile_recovery_state_transition_invalid"
                ),
                repair_rounds_completed=(
                    validated_repair_round_count(state_root)
                ),
                extra={
                    "compile_repair_validated": True,
                    "final_compile_validated": False,
                    "semantic_complete": False,
                    "compile_recovery_result": recovery_result,
                    "compile_recovery_settlement": settlement,
                },
            )

        settlement_status = str(settlement.get("status") or "")
        active_batch_remains = lifecycle.has_active_batch()
        if settlement_status == "batch_resolved":
            if active_batch_remains or not recovery_result.get(
                "compile_validated"
            ):
                return invalid_settlement()
            recovered_validation = dict(
                recovery_result.get("validation") or {}
            )
            return finish_after_compile_validation(
                recovered_validation,
                completed_rounds=(
                    validated_repair_round_count(state_root)
                ),
                completion_extra={
                    "compile_recovery_result": recovery_result,
                    "compile_recovery_settlement": settlement,
                },
                semantic_handoff_already_recorded=True,
            )
        if settlement_status == "active_batch_preserved":
            if not active_batch_remains:
                return invalid_settlement()
            if settlement.get("reason") == "recovery_budget_remaining":
                budget_after = dict(settlement.get("budget") or {})
                if int(budget_after.get("completed") or 0) <= int(
                    recovery_budget_before.get("completed") or 0
                ):
                    return invalid_settlement()
                next_validation = dict(
                    recovery_result.get("validation") or {}
                )
                if (
                    next_validation.get("status") != "needs_patching"
                    or not next_validation.get("repair_plan_path")
                ):
                    return invalid_settlement()
                return continue_semantic_compile_recovery(
                    next_validation,
                    completed_rounds=(
                        validated_repair_round_count(state_root)
                    ),
                    semantic_handoff=handoff_context,
                    semantic_closure=closure_context,
                )
            ledger = load_semantic_closure_state(
                state_root,
                case_id=spec.case_id,
            )
            closure_summary = (
                isolated_semantic_closure_summary(ledger)
                if ledger is not None
                else {}
            )
            return finish(
                status=durable_case_status(),
                stop_reason="semantic_compile_recovery_paused",
                repair_rounds_completed=(
                    validated_repair_round_count(state_root)
                ),
                extra={
                    "compile_repair_validated": True,
                    "final_compile_validated": False,
                    "semantic_complete": False,
                    "semantic_handoff": handoff_context,
                    "semantic_closure": {
                        "status": "compile_recovery_paused",
                        "summary": closure_summary,
                    },
                    "semantic_resume_allowed": True,
                    "compile_recovery_result": recovery_result,
                    "compile_recovery_settlement": settlement,
                },
            )
        if settlement_status != (
            "budget_exhausted_candidate_retained"
        ):
            return invalid_settlement()
        if not active_batch_remains:
            return invalid_settlement()
        ledger = load_semantic_closure_state(
            state_root,
            case_id=spec.case_id,
        )
        closure_summary = (
            isolated_semantic_closure_summary(ledger)
            if ledger is not None
            else {}
        )
        return finish(
            status=durable_case_status(),
            stop_reason="semantic_compile_recovery_budget_exhausted",
            repair_rounds_completed=(
                validated_repair_round_count(state_root)
            ),
            extra={
                "compile_repair_validated": True,
                "final_compile_validated": False,
                "semantic_complete": False,
                "semantic_handoff": handoff_context,
                "semantic_closure": {
                    "status": "compile_recovery_budget_exhausted",
                    "summary": closure_summary,
                },
                "candidate_retained": True,
                "semantic_resume_allowed": False,
                "resume_requires_budget_increase": True,
                "compile_recovery_result": recovery_result,
                "compile_recovery_settlement": settlement,
            },
        )

    def prepare_build_configuration(
        *,
        completed_rounds: int,
    ) -> dict[str, Any] | None:
        announce(
            "build_config_static_preflight",
            rounds=completed_rounds,
        )
        build_config = run_build_config_preflight(
            spec,
            state_root=state_root,
            model=model,
            max_completion_tokens=max_completion_tokens,
            allow_escalation=allow_escalation,
            max_attempts=0,
            progress=_emit_progress,
        )
        record(
            "build_config_static_preflight",
            build_config,
            rounds=completed_rounds,
        )
        if (
            not build_config.get("ok")
            and max_build_config_preflight_attempts > 0
        ):
            model_failure = ensure_model_ready(
                repair_rounds_completed=completed_rounds,
            )
            if model_failure is not None:
                return model_failure
            announce(
                "build_config_preflight",
                rounds=completed_rounds,
            )
            build_config = run_build_config_preflight(
                spec,
                state_root=state_root,
                model=model,
                max_completion_tokens=max_completion_tokens,
                allow_escalation=allow_escalation,
                max_attempts=max_build_config_preflight_attempts,
                progress=_emit_progress,
            )
            record(
                "build_config_preflight",
                build_config,
                rounds=completed_rounds,
            )
        if build_config.get("ok"):
            return None
        final_state = show_status(spec, state_root=state_root)
        return finish(
            status=final_state["status"],
            stop_reason="build_config_preflight_failed",
            repair_rounds_completed=completed_rounds,
            extra={
                "build_config_preflight": build_config,
                "source_repair_blocked": True,
                "environment_failure": bool(
                    build_config.get("environment_failure")
                ),
            },
        )

    def continue_from_validation(
        validation: dict[str, Any],
        *,
        completed_rounds: int,
        initial_cycle_dir: Path | None = None,
        environment_stop_reason: str = "validation_environment_blocked",
    ) -> dict[str, Any]:
        if initial_cycle_dir is not None:
            initial_terminal, validation = (
                finalize_initial_compatibility_after_compile(
                    initial_cycle_dir=initial_cycle_dir,
                    current_validation=validation,
                    completed_rounds=completed_rounds,
                )
            )
            if initial_terminal is not None:
                return initial_terminal
        elif _build_prep_route_summary(validation).get(
            "needs_build_prep"
        ):
            build_prep_route = _reroute_build_prep_if_needed(
                validation,
                repair_rounds_completed=completed_rounds,
                execution_phase=str(
                    validation.get("execution_phase")
                    or EXECUTION_PHASE_COMPILER_REPAIR
                ),
            )
            validation = dict(
                build_prep_route.get("validation") or validation
            )
            if build_prep_route.get("status") == "failed":
                final_state = show_status(spec, state_root=state_root)
                return finish(
                    status=final_state["status"],
                    stop_reason=str(
                        build_prep_route.get("stop_reason")
                        or "build_config_preflight_failed"
                    ),
                    repair_rounds_completed=completed_rounds,
                    extra=dict(build_prep_route.get("extra") or {}),
                )
        if validation["status"] == "ready_for_manual_test":
            return finish_after_compile_validation(
                validation,
                completed_rounds=int(
                    validated_repair_round_counts(state_root)[
                        "compile_repair_rounds_completed"
                    ]
                ),
                initial_cycle_dir=initial_cycle_dir,
            )
        if _validation_blocks_auto_repair(validation):
            final_state = show_status(spec, state_root=state_root)
            return finish(
                status=final_state["status"],
                stop_reason=environment_stop_reason,
                repair_rounds_completed=completed_rounds,
            )
        model_failure = ensure_model_ready(
            repair_rounds_completed=completed_rounds,
        )
        if model_failure is not None:
            return model_failure
        return continue_agentic_repair_from_validation(
            validation,
            completed_rounds=completed_rounds,
        )

    def continue_validation_after_transplant(resume_state: dict) -> dict:
        cycle_dir = Path(str(resume_state.get("current_cycle_dir") or "")).resolve()
        if not cycle_dir.exists():
            final_state = show_status(spec, state_root=state_root)
            return finish(
                status=final_state["status"],
                stop_reason="resume_transplant_cycle_dir_missing",
                repair_rounds_completed=0,
            )

        preparation_failure = prepare_build_configuration(
            completed_rounds=0,
        )
        if preparation_failure is not None:
            return preparation_failure

        initial_failure = run_initial_compatibility_before_compile(cycle_dir)
        if initial_failure is not None:
            return initial_failure

        record(
            "first_target_compile_boundary",
            {
                "reason": (
                    "the one-shot initial compatibility pass completed; the target compiler now validates "
                    "deterministic and model-backed demanded-difference transactions"
                ),
            },
        )

        announce("validate_after_transplant")
        validation = run_post_validation_cycle(
            spec,
            state_root=state_root,
        )
        record("validate_after_transplant", validation)
        return continue_from_validation(
            validation,
            completed_rounds=0,
            initial_cycle_dir=cycle_dir,
            environment_stop_reason=(
                "validation_environment_blocked_after_transplant"
            ),
        )

    def continue_from_semantic_resume(
        semantic_resume: dict[str, Any],
        *,
        resume_state: dict[str, Any],
        completed_rounds: int,
    ) -> dict[str, Any]:
        next_action = str(semantic_resume.get("next_action") or "")
        record(
            "resume_semantic_state",
            semantic_resume,
            rounds=completed_rounds,
        )
        if next_action == "semantic_handoff_stale":
            return finish(
                status=str(resume_state["status"]),
                stop_reason="semantic_handoff_revision_stale",
                repair_rounds_completed=completed_rounds,
                extra={
                    "semantic_resume_route": semantic_resume,
                    "compile_repair_validated": True,
                    "semantic_complete": False,
                },
            )
        if next_action in {
            "resume_semantic_controller",
            "semantic_closed",
        }:
            if semantic_resume.get("model_required"):
                model_failure = ensure_model_ready(
                    repair_rounds_completed=completed_rounds,
                )
                if model_failure is not None:
                    return model_failure
            current_checkpoint = str(
                resume_state.get("last_validation_path") or ""
            ).strip()
            if not current_checkpoint:
                return finish(
                    status=str(resume_state["status"]),
                    stop_reason=(
                        "semantic_compile_valid_checkpoint_missing"
                    ),
                    repair_rounds_completed=completed_rounds,
                    extra={
                        "semantic_resume_route": semantic_resume,
                        "compile_repair_validated": True,
                        "semantic_complete": False,
                    },
                )
            return finish_after_compile_validation(
                {
                    "status": "ready_for_manual_test",
                    "compile_ok": True,
                    "overall_ok": True,
                    "validation_path": current_checkpoint,
                    "semantic_resume": True,
                },
                completed_rounds=int(
                    validated_repair_round_counts(state_root)[
                        "compile_repair_rounds_completed"
                    ]
                ),
                completion_extra={
                    "semantic_resume_route": semantic_resume,
                },
            )
        if next_action == "resume_compile_recovery":
            model_failure = ensure_model_ready(
                repair_rounds_completed=completed_rounds,
            )
            if model_failure is not None:
                return model_failure
            recovery_validation = dict(
                semantic_resume.get("validation") or {}
            )
            if not recovery_validation:
                return finish(
                    status=str(resume_state["status"]),
                    stop_reason=(
                        "semantic_compile_recovery_validation_missing"
                    ),
                    repair_rounds_completed=completed_rounds,
                    extra={
                        "semantic_resume_route": semantic_resume,
                        "compile_repair_validated": True,
                    },
                )
            return continue_semantic_compile_recovery(
                recovery_validation,
                completed_rounds=completed_rounds,
                semantic_handoff=dict(
                    semantic_resume.get("semantic_handoff") or {}
                ),
                semantic_closure={
                    "status": "compile_recovery_required",
                    "recovery_context": {
                        key: value
                        for key, value in semantic_resume.items()
                        if key
                        not in {
                            "semantic_handoff",
                            "semantic_summary",
                            "validation",
                        }
                    },
                    "summary": dict(
                        semantic_resume.get("semantic_summary") or {}
                    ),
                },
            )
        return finish(
            status=str(resume_state["status"]),
            stop_reason="semantic_state_incomplete",
            repair_rounds_completed=completed_rounds,
            extra={
                "semantic_resume_route": semantic_resume,
                "compile_repair_validated": True,
                "semantic_complete": False,
            },
        )

    def recover_interrupted_repair_round(
        *,
        completed_rounds: int,
        execution_phase: str,
        pending_manifests: list[tuple[Path, dict[str, Any]]],
        repair_lifecycle: SemanticCompileRecoveryLifecycle | None = None,
    ) -> tuple[dict[str, Any], int] | None:
        if not pending_manifests:
            return None
        active_execution_phase = validate_execution_phase(
            execution_phase
        )
        manifest_phases = {
            validate_execution_phase(
                manifest.get("execution_phase")
                or EXECUTION_PHASE_COMPILER_REPAIR
            )
            for _manifest_path, manifest in pending_manifests
        }
        if manifest_phases != {active_execution_phase}:
            raise ValueError(
                "interrupted repair manifests do not match the active "
                "execution phase: "
                + ", ".join(sorted(manifest_phases))
            )

        validation_targets: list[dict[str, Any]] = []
        assessment_targets: list[dict[str, Any]] = []
        applied_units: list[dict[str, Any]] = []
        degraded_manifests: list[str] = []
        for manifest_path, manifest in pending_manifests:
            manifest_records = round_manifest_collections(
                manifest_path,
                manifest,
                state_root=state_root,
                fields=(
                    "validation_targets",
                    "assessment_targets",
                    "applied_units",
                ),
            )
            manifest_validation_targets = manifest_records[
                "validation_targets"
            ]
            manifest_assessment_targets = manifest_records[
                "assessment_targets"
            ]
            manifest_applied_units = manifest_records["applied_units"]
            validation_targets.extend(
                dict(item)
                for item in manifest_validation_targets
                if isinstance(item, dict)
            )
            assessment_targets.extend(
                dict(item)
                for item in (
                    manifest_assessment_targets
                    or manifest_validation_targets
                    or []
                )
                if isinstance(item, dict)
            )
            applied_units.extend(
                dict(item)
                for item in manifest_applied_units
                if isinstance(item, dict)
            )
            inflight_target_path = Path(
                str(
                    manifest.get("inflight_validation_target_path")
                    or ""
                )
            )
            if (
                inflight_target_path.is_file()
            ):
                try:
                    loaded_inflight_target = json.loads(
                        inflight_target_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    loaded_inflight_target = None
                if isinstance(loaded_inflight_target, dict):
                    inflight_target = loaded_inflight_target
                else:
                    inflight_target = None
            else:
                inflight_target = None
            inflight_unit_id = str(
                manifest.get("inflight_agentic_unit_id") or ""
            )
            if isinstance(inflight_target, dict) and inflight_unit_id:
                recovered_apply = _applied_transaction_result_for_work_unit(
                    state_root,
                    work_unit_id=inflight_unit_id,
                )
                if recovered_apply is not None:
                    target = _validation_target_with_apply_evidence(
                        dict(inflight_target),
                        apply_result=recovered_apply,
                        baseline_validation=None,
                    )
                    validation_targets.append(target)
                    assessment_targets.append(target)
                    applied_units.append(
                        {
                            "plan_unit_id": manifest.get(
                                "inflight_plan_unit_id"
                            ),
                            "work_unit_id": str(
                                target.get("work_unit_id")
                                or target.get("id")
                                or ""
                            ),
                            "assessment_target": copy.deepcopy(target),
                            "apply_result": recovered_apply,
                        }
                    )
            if (
                int(manifest.get("validation_target_count") or 0)
                and not manifest_validation_targets
            ):
                degraded_manifests.append(str(manifest_path))

        targets_by_id: dict[str, dict[str, Any]] = {}
        for target in validation_targets:
            identity = str(
                target.get("work_unit_id") or target.get("id") or ""
            )
            if identity:
                targets_by_id[identity] = target
        validation_targets = list(targets_by_id.values())
        assessment_by_id: dict[str, dict[str, Any]] = {}
        for target in assessment_targets:
            identity = str(
                target.get("origin_work_unit_id")
                or target.get("work_unit_id")
                or target.get("id")
                or ""
            )
            if identity:
                assessment_by_id[identity] = target
        assessment_targets = list(assessment_by_id.values())

        has_durable_apply = bool(
            validation_targets
            or applied_units
            or degraded_manifests
        )
        if not has_durable_apply:
            for manifest_path, manifest in pending_manifests:
                write_json(
                    manifest_path,
                    {
                        **manifest,
                        "schema": "ordered_graph_root_repair_round_v6",
                        "status": "aborted_before_apply",
                        "recovered_at": utc_now(),
                        "recovery_reason": (
                            "no finalized applied transaction exists for the "
                            "inflight unit"
                        ),
                    },
                )
            return None

        announce(
            "resume_pending_combined_validation",
            rounds=completed_rounds,
        )
        validation = run_post_validation_cycle(
            spec,
            state_root=state_root,
            validation_targets=validation_targets,
            validation_assessment_targets=assessment_targets,
            execution_phase=active_execution_phase,
        )
        recovery_build_prep_route = _reroute_build_prep_if_needed(
            validation,
            repair_rounds_completed=completed_rounds,
            execution_phase=active_execution_phase,
        )
        validation = dict(
            recovery_build_prep_route.get("validation") or validation
        )
        postimage_observed = bool(
            recovery_build_prep_route.get("status") != "failed"
            and _validation_observes_repair_postimage(validation)
        )
        recovered_rounds = completed_rounds
        recovery = {
            "schema": "interrupted_round_recovery_v1",
            "pending_manifest_paths": [
                str(path) for path, _manifest in pending_manifests
            ],
            "validation_target_count": len(validation_targets),
            "assessment_target_count": len(assessment_targets),
            "applied_unit_count": len(applied_units),
            "degraded_manifest_paths": degraded_manifests,
            "validation_path": validation.get("validation_path"),
            "validation_delta": validation.get("validation_delta"),
            "postimage_observed": postimage_observed,
            "build_preparation_route": recovery_build_prep_route,
            "repair_budget_consumed": postimage_observed,
        }
        if not postimage_observed:
            for manifest_path, manifest in pending_manifests:
                write_json(
                    manifest_path,
                    {
                        **manifest,
                        "schema": "ordered_graph_root_repair_round_v6",
                        "status": "validation_blocked_pending_retry",
                        "recovery_attempted_at": utc_now(),
                        "recovery": recovery,
                        "post_recovery_validation_path": validation.get(
                            "validation_path"
                        ),
                    },
                )
            record(
                "resume_pending_combined_validation_blocked",
                recovery,
                rounds=completed_rounds,
            )
            return validation, completed_rounds
        candidate_state = _retain_round_candidates(
            validation_delta=dict(validation.get("validation_delta") or {}),
            applied_units=applied_units,
        )
        recovered_rounds = completed_rounds + 1
        patch_outcome_update = record_round_patch_outcomes(
            state_root,
            repair_round=recovered_rounds,
            applied_units=applied_units,
            validation_delta=dict(validation.get("validation_delta") or {}),
            candidate_state=candidate_state,
        )
        record(
            "resume_patch_outcome_ledger_updated",
            patch_outcome_update,
            rounds=recovered_rounds,
        )
        final_status = "validated"

        lifecycle_validation = (
            repair_lifecycle.record_validation(validation)
            if repair_lifecycle is not None
            else None
        )
        if lifecycle_validation is not None:
            recovery["repair_lifecycle"] = lifecycle_validation
        record(
            "resume_pending_combined_validation",
            recovery,
            rounds=recovered_rounds,
        )

        last_manifest_index = len(pending_manifests) - 1
        for manifest_index, (manifest_path, manifest) in enumerate(
            pending_manifests
        ):
            write_json(
                manifest_path,
                {
                    **manifest,
                    "schema": "ordered_graph_root_repair_round_v6",
                    "status": (
                        final_status
                        if manifest_index == last_manifest_index
                        else "recovered_by_combined_validation"
                    ),
                    "recovered_at": utc_now(),
                    "recovery": recovery,
                    "candidate_retention_state": (
                        candidate_state
                    ),
                    "post_recovery_validation_path": validation.get(
                        "validation_path"
                    ),
                },
            )
        return validation, recovered_rounds

    resume_state = show_status(spec, state_root=state_root)
    semantic_resume = pending_semantic_resume()
    pending_round_manifests = pending_repair_round_manifests(state_root)
    resume_projection = project_workflow_resume(
        cycle_state=resume_state,
        semantic_resume=semantic_resume,
        pending_manifest_phases=[
            str(
                manifest.get("execution_phase")
                or EXECUTION_PHASE_COMPILER_REPAIR
            )
            for _path, manifest in pending_round_manifests
        ],
    )
    resume_existing_migration = bool(resume_projection["resumable"])
    completed_repair_rounds = (
        validated_repair_round_count(state_root) if resume_existing_migration else 0
    )
    record(
        "workflow_resume_projection",
        resume_projection,
        rounds=completed_repair_rounds,
    )

    open_transaction_recovery = _recover_open_transactions(
        spec,
        state_root=state_root,
    )
    if open_transaction_recovery["status"] != "clean":
        record(
            "recover_open_transactions",
            open_transaction_recovery,
            rounds=completed_repair_rounds,
        )
    if open_transaction_recovery["status"] == "failed":
        final_state = show_status(spec, state_root=state_root)
        return finish(
            status=final_state["status"],
            stop_reason="open_transaction_recovery_failed",
            repair_rounds_completed=completed_repair_rounds,
            extra={"open_transaction_recovery": open_transaction_recovery},
        )

    scope_transaction_recovery = reconcile_graph_scope_transactions(
        spec,
        state_root=state_root,
    )
    if scope_transaction_recovery["status"] != "clean":
        record(
            "recover_graph_scope_transactions",
            scope_transaction_recovery,
            rounds=completed_repair_rounds,
        )
    spec = load_graph_scope_expansions(spec, state_root=state_root)

    pending_scope_expansions = pending_graph_scope_expansions(
        spec,
        state_root=state_root,
    )
    for pending_scope_expansion in pending_scope_expansions:
        record(
            "resume_scope_expansion_compile_context",
            {
                "event_id": pending_scope_expansion.get("event_id"),
                "transaction_dir": pending_scope_expansion.get(
                    "transaction_dir"
                ),
                "reason": (
                    "source/build scope transaction was durable but its "
                    "compile context was not yet committed"
                ),
            },
            rounds=completed_repair_rounds,
        )
        scope_context_result = refresh_compile_context_after_scope_expansion(
            repair_rounds_completed=completed_repair_rounds,
            scope_expansion=pending_scope_expansion,
        )
        if scope_context_result.get("status") == "failed":
            final_state = show_status(spec, state_root=state_root)
            return finish(
                status=final_state["status"],
                stop_reason=str(
                    scope_context_result.get("stop_reason")
                    or "scope_expansion_compile_context_failed"
                ),
                repair_rounds_completed=completed_repair_rounds,
                extra=dict(scope_context_result.get("extra") or {}),
            )

    pending_scope_validation_events = pending_graph_scope_validations(
        spec,
        state_root=state_root,
    )
    pending_scope_validation_event_ids = [
        str(event.get("event_id") or "")
        for event in pending_scope_validation_events
        if str(event.get("event_id") or "")
    ]
    if pending_scope_validation_event_ids:
        record(
            "pending_graph_scope_validation_recovered",
            {
                "status": "fresh_validation_required",
                "event_ids": pending_scope_validation_event_ids,
                "reason": (
                    "scope source and compile context are committed but no "
                    "fresh target validation has observed their postimage"
                ),
            },
            rounds=completed_repair_rounds,
        )

    resume_execution_phase = str(resume_projection["execution_phase"])

    resume_action = str(resume_projection["action"])
    if (
        resume_action == "resume_semantic_state"
        and pending_scope_validation_event_ids
    ):
        semantic_scope_phases = {
            str(
                event.get("execution_phase")
                or EXECUTION_PHASE_SEMANTIC_CLOSURE
            )
            for event in pending_scope_validation_events
        }
        if semantic_scope_phases != {EXECUTION_PHASE_SEMANTIC_CLOSURE}:
            raise ValueError(
                "pending semantic scope validation has mixed execution "
                "phases: " + ", ".join(sorted(semantic_scope_phases))
            )
        announce(
            "resume_semantic_scope_postimage_validation",
            rounds=completed_repair_rounds,
        )
        semantic_scope_validation = run_post_validation_cycle(
            spec,
            state_root=state_root,
            execution_phase=EXECUTION_PHASE_SEMANTIC_CLOSURE,
        )
        semantic_scope_route = _reroute_build_prep_if_needed(
            semantic_scope_validation,
            repair_rounds_completed=completed_repair_rounds,
            execution_phase=EXECUTION_PHASE_SEMANTIC_CLOSURE,
        )
        semantic_scope_validation = dict(
            semantic_scope_route.get("validation")
            or semantic_scope_validation
        )
        if not _validation_observes_repair_postimage(
            semantic_scope_validation
        ):
            return finish(
                status=str(resume_state["status"]),
                stop_reason=str(
                    semantic_scope_route.get("stop_reason")
                    or "semantic_scope_validation_pending_retry"
                ),
                repair_rounds_completed=completed_repair_rounds,
                extra={
                    "semantic_complete": False,
                    "repair_budget_consumed": False,
                    "resumable_scope_validation": True,
                    "pending_scope_event_ids": (
                        pending_scope_validation_event_ids
                    ),
                },
            )
        scope_validation_recovery = complete_graph_scope_validations(
            spec,
            state_root=state_root,
            event_ids=pending_scope_validation_event_ids,
            validation=semantic_scope_validation,
        )
        record(
            "resume_semantic_scope_postimage_validation",
            scope_validation_recovery,
            rounds=completed_repair_rounds,
        )
        if semantic_scope_validation.get("status") == (
            "ready_for_manual_test"
        ):
            return finish_after_compile_validation(
                semantic_scope_validation,
                completed_rounds=completed_repair_rounds,
                completion_extra={
                    "semantic_scope_validation_recovery": (
                        scope_validation_recovery
                    )
                },
                semantic_handoff_already_recorded=True,
            )
        assert semantic_resume is not None
        return continue_semantic_compile_recovery(
            semantic_scope_validation,
            completed_rounds=completed_repair_rounds,
            semantic_handoff=dict(
                semantic_resume.get("semantic_handoff") or {}
            ),
            semantic_closure={
                "status": "compile_recovery_required",
                "recovery_context": {
                    "reason": "resumed_pending_semantic_scope_validation",
                    "scope_event_ids": (
                        pending_scope_validation_event_ids
                    ),
                },
                "summary": dict(
                    semantic_resume.get("semantic_summary") or {}
                ),
            },
        )
    if resume_action == "resume_after_transplant":
        announce("resume_after_transplant_apply")
        record(
            "resume_after_transplant_apply",
            {
                "status": resume_state["status"],
                "reason": "transplant already applied; compiling the current worktree instead of restarting transplant",
                "current_cycle_dir": resume_state.get("current_cycle_dir"),
            },
        )
        return continue_validation_after_transplant(resume_state)

    if resume_action == "resume_semantic_state":
        announce("resume_from_existing_state")
        record(
            "resume_from_existing_state",
            {
                "status": resume_state["status"],
                "reason": (
                    "existing state is past transplant; validating current worktree instead of "
                    "restarting transplant from the baseline"
                ),
                "last_validation_path": resume_state.get("last_validation_path"),
                "completed_repair_rounds": completed_repair_rounds,
            },
            rounds=completed_repair_rounds,
        )
        assert semantic_resume is not None
        return continue_from_semantic_resume(
            semantic_resume,
            resume_state=resume_state,
            completed_rounds=completed_repair_rounds,
        )

    if resume_action == "recover_interrupted_repair_round":
        recovery_lifecycle = (
            SemanticCompileRecoveryLifecycle(
                spec,
                state_root=state_root,
                max_rounds=max_semantic_repair_rounds,
            )
            if resume_projection["after_recovery"]
            == "resume_semantic_state"
            else None
        )
        interrupted_round = recover_interrupted_repair_round(
            completed_rounds=completed_repair_rounds,
            execution_phase=resume_execution_phase,
            pending_manifests=pending_round_manifests,
            repair_lifecycle=recovery_lifecycle,
        )
        if interrupted_round is not None:
            validation, completed_repair_rounds = interrupted_round
            if (
                pending_scope_validation_event_ids
                and _validation_observes_repair_postimage(validation)
            ):
                scope_validation_recovery = (
                    complete_graph_scope_validations(
                        spec,
                        state_root=state_root,
                        event_ids=pending_scope_validation_event_ids,
                        validation=validation,
                    )
                )
                record(
                    "resume_graph_scope_validation_completed",
                    scope_validation_recovery,
                    rounds=completed_repair_rounds,
                )
        if resume_projection["after_recovery"] == "resume_semantic_state":
            resumed_semantic = pending_semantic_resume()
            if resumed_semantic is None:
                raise ValueError(
                    "semantic round recovery lost its compile-valid ledger"
                )
            return continue_from_semantic_resume(
                resumed_semantic,
                resume_state=show_status(spec, state_root=state_root),
                completed_rounds=completed_repair_rounds,
            )

        if interrupted_round is not None:
            initial_cycle_dir = pending_initial_compatibility_cycle()
            return continue_from_validation(
                validation,
                completed_rounds=completed_repair_rounds,
                initial_cycle_dir=initial_cycle_dir,
            )
        resume_action = "validate_current_tree"

    if resume_action == "validate_current_tree":
        announce("resume_from_existing_state")
        record(
            "resume_from_existing_state",
            {
                "status": resume_state["status"],
                "last_validation_path": resume_state.get(
                    "last_validation_path"
                ),
                "completed_repair_rounds": completed_repair_rounds,
            },
            rounds=completed_repair_rounds,
        )
        announce("validate_resume_state", rounds=completed_repair_rounds)
        validation = run_post_validation_cycle(
            spec,
            state_root=state_root,
            execution_phase=resume_execution_phase,
        )
        resume_build_prep_route = _reroute_build_prep_if_needed(
            validation,
            repair_rounds_completed=completed_repair_rounds,
            execution_phase=resume_execution_phase,
        )
        validation = dict(
            resume_build_prep_route.get("validation") or validation
        )
        if pending_scope_validation_event_ids:
            if _validation_observes_repair_postimage(validation):
                scope_validation_recovery = (
                    complete_graph_scope_validations(
                        spec,
                        state_root=state_root,
                        event_ids=pending_scope_validation_event_ids,
                        validation=validation,
                    )
                )
                record(
                    "resume_graph_scope_validation_completed",
                    scope_validation_recovery,
                    rounds=completed_repair_rounds,
                )
            else:
                record(
                    "resume_graph_scope_validation_still_pending",
                    {
                        "status": "validation_blocked_pending_retry",
                        "event_ids": (
                            pending_scope_validation_event_ids
                        ),
                        "validation_status": validation.get("status"),
                        "validation_path": validation.get(
                            "validation_path"
                        ),
                    },
                    rounds=completed_repair_rounds,
                )
        record(
            "validate_resume_state",
            validation,
            rounds=completed_repair_rounds,
        )
        initial_cycle_dir = pending_initial_compatibility_cycle()
        return continue_from_validation(
            validation,
            completed_rounds=completed_repair_rounds,
            initial_cycle_dir=initial_cycle_dir,
        )

    if resume_action != "start_new_migration":
        raise ValueError(f"unsupported workflow resume action: {resume_action}")

    announce("run_cycle")
    initial = run_cycle(spec, state_root=state_root)
    record("run_cycle", initial)
    if initial.get("status") != "needs_transplant" or not initial.get("transplant_plan_path"):
        final_state = show_status(spec, state_root=state_root)
        return finish(
            status=final_state["status"],
            stop_reason="semantic_scope_preparation_failed",
            repair_rounds_completed=0,
            extra={"initial_cycle": initial, "source_repair_blocked": True},
        )

    announce("apply_transplant")
    transplant = apply_transplant_only(spec, state_root=state_root, plan_path=Path(initial["transplant_plan_path"]))
    record("apply_transplant", transplant)
    if transplant["failed_copy_operations"] or transplant["blocked_copy_operations"]:
        final_state = show_status(spec, state_root=state_root)
        return finish(status=final_state["status"], stop_reason="transplant_copy_not_clean", repair_rounds_completed=0)
    if transplant["failed_build_integrations"] or transplant["blocked_build_integrations"]:
        final_state = show_status(spec, state_root=state_root)
        return finish(status=final_state["status"], stop_reason="build_integration_not_clean", repair_rounds_completed=0)

    return continue_validation_after_transplant(
        {
            "current_cycle_dir": str(
                Path(str(initial["transplant_plan_path"])).resolve().parent
            )
        }
    )


def _add_auto_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=str)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=5,
        help=(
            "validated compiler-repair round limit before the first "
            "compile-valid boundary"
        ),
    )
    parser.add_argument(
        "--max-semantic-repair-rounds",
        type=int,
        default=3,
        help=(
            "canonical compiler-recovery round limit after one complete "
            "semantic gap sweep and its unified compile"
        ),
    )
    parser.add_argument("--allow-escalation", action="store_true")
    parser.add_argument(
        "--build-config-preflight-attempts",
        type=int,
        default=2,
        help="maximum LLM repair attempts for Kconfig/Makefile/.config preparation before target compilation",
    )
def _add_auto_migrate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("case_spec", type=Path)
    _add_auto_runtime_args(parser)


def _add_dataset_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rows", type=str, help="CSV row numbers/ranges, for example: 2,5-8")
    parser.add_argument("--type", dest="types", action="append", help="filter Type column by substring")
    parser.add_argument("--name", dest="names", action="append", help="filter Name column by substring")
    parser.add_argument("--path", dest="paths", action="append", help="filter Path column by substring")
    parser.add_argument("--limit", type=int, help="maximum selected dataset entries")


def _add_dataset_case_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", type=Path, help="dataset .csv with a Path column")
    parser.add_argument("--base-case", type=Path, help="case spec template providing roots/build env")
    parser.add_argument("--old-root", type=Path, help="old target kernel root used when --base-case is omitted")
    parser.add_argument(
        "--donor-root",
        dest="donor_roots",
        action="append",
        type=Path,
        help="donor kernel root used when --base-case is omitted; repeat for multiple donors",
    )
    parser.add_argument("-o", "--output-dir", type=Path)
    parser.add_argument("--dataset-workspace", type=Path, help="dataset artifact root; cases default to this directory")
    parser.add_argument(
        "--compile-commands-matrix-root",
        type=Path,
        help="root containing cases/compile_commands_dataset_<label>/<case_id>.json",
    )
    parser.add_argument("--compile-commands-label", type=str, help="target old-tree compile_commands label, for example v5.10")
    parser.add_argument("--donor-compile-commands-label", type=str, default="donor-v6.19")
    parser.add_argument("--case-prefix", type=str, default="dataset")
    parser.add_argument("--state-root-base", type=Path)
    parser.add_argument("--build-dir-base", type=Path)
    parser.add_argument("--old-root-work", type=Path, help="rewrite generated case old_root to this shared work tree")
    parser.add_argument("--copy-validation", action="store_true", help="keep validation/load settings from base case")
    parser.add_argument("--no-infer-kconfig", action="store_true")
    parser.add_argument("--no-infer-expected-modules", action="store_true")
    _add_dataset_selection_args(parser)


def _dataset_base_case_from_args(
    args: argparse.Namespace,
    *,
    compile_commands_matrix_root: Path | None,
    compile_commands_label: str | None,
) -> dict[str, Any]:
    base_case_path = args.base_case
    if base_case_path:
        return json.loads(base_case_path.resolve().read_text(encoding="utf-8"))

    old_root = args.old_root
    donor_roots = [Path(item).expanduser() for item in args.donor_roots]
    if not old_root or not donor_roots:
        raise ValueError("--base-case is required unless --old-root and at least one --donor-root are provided")
    if compile_commands_matrix_root is None or not compile_commands_label:
        raise ValueError(
            "--compile-commands-label plus --dataset-workspace or --compile-commands-matrix-root "
            "are required when --base-case is omitted"
        )

    return {
        "case_id": "dataset-template",
        "old_root": str(Path(old_root).expanduser()),
        "donor_roots": [str(item) for item in donor_roots],
        "driver_root": "drivers",
        "entry_files": [],
        "editable_roots": [],
        "escalation_edit_roots": ["drivers", "include"],
        "forbidden_roots": ["arch", "kernel", "mm"],
        "external_edit_policy": "allow-listed-escalation",
        "compile_commands": str(Path(compile_commands_matrix_root).expanduser() / "compile_commands_placeholder.json"),
        "donor_compile_commands": [
            str(Path(compile_commands_matrix_root).expanduser() / "donor_compile_commands_placeholder.json")
        ],
        "require_load_validation": False,
        "load_phases": [],
        "build_phases": [],
        "dataset_generation": {"base_case_source": "cli-old-root-donor-root"},
    }


def _materialize_dataset_cases_from_args(args: argparse.Namespace) -> dict[str, Any]:
    dataset_workspace_arg = args.dataset_workspace
    dataset_workspace = dataset_workspace_arg.expanduser() if dataset_workspace_arg else None
    output_dir_arg = args.output_dir
    compile_commands_label = args.compile_commands_label
    donor_compile_commands_label = args.donor_compile_commands_label
    if output_dir_arg is not None:
        output_dir = ensure_dir(output_dir_arg)
    elif dataset_workspace is not None:
        label = compile_commands_label or "unbound"
        output_dir = ensure_dir(dataset_workspace / "cases" / slugify(label))
    else:
        raise ValueError("--output-dir is required unless --dataset-workspace is set")
    compile_commands_matrix_root = args.compile_commands_matrix_root
    if compile_commands_matrix_root is None and dataset_workspace is not None and compile_commands_label:
        compile_commands_matrix_root = dataset_workspace / "compile_commands"
    base_case = _dataset_base_case_from_args(
        args,
        compile_commands_matrix_root=compile_commands_matrix_root,
        compile_commands_label=compile_commands_label,
    )
    if args.base_build_dir is not None:
        base_case["target_compile_command_build_dirs"] = [
            str(args.base_build_dir.expanduser().resolve())
        ]
    manifest = materialize_dataset_cases(
        dataset_path=args.dataset,
        base_case=base_case,
        output_dir=output_dir,
        rows=parse_row_selector(args.rows),
        types=args.types,
        names=args.names,
        paths=args.paths,
        limit=args.limit,
        case_prefix=args.case_prefix,
        state_root_base=args.state_root_base,
        build_dir_base=args.build_dir_base,
        old_root_work=args.old_root_work,
        copy_validation=args.copy_validation,
        infer_kconfig=not args.no_infer_kconfig,
        infer_expected_modules=not args.no_infer_expected_modules,
        compile_commands_matrix_root=compile_commands_matrix_root,
        compile_commands_label=compile_commands_label,
        donor_compile_commands_label=donor_compile_commands_label,
    )
    return manifest


def _raw_case_state_root(raw: dict[str, Any]) -> Path:
    if raw.get("state_root"):
        return Path(str(raw["state_root"])).expanduser().resolve()
    case_id = str(raw.get("case_id") or slugify(str(raw.get("driver_root") or "driver-backport")))
    return (DEFAULT_STATE_BASE.resolve() / slugify(case_id)).resolve()


def _write_dataset_environment_report(report: dict[str, Any]) -> None:
    report["updated_at"] = utc_now()
    write_json(Path(str(report["environment_path"])), report)


def _dataset_case_resume_guard(state_root: Path) -> dict[str, Any]:
    """Protect a durable migration worktree from dataset-level reset flags."""
    inspected: list[str] = []
    for state_path in (
        state_root / "cycle_state.json",
        state_root / "migration_result.json",
    ):
        if not state_path.is_file():
            continue
        inspected.append(str(state_path.resolve()))
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "invalid",
                "resumable": False,
                "state_path": str(state_path.resolve()),
                "inspected_paths": inspected,
                "error": str(exc),
            }
        if not isinstance(payload, dict):
            return {
                "status": "invalid",
                "resumable": False,
                "state_path": str(state_path.resolve()),
                "inspected_paths": inspected,
                "error": "persisted migration state is not a JSON object",
            }
        if migration_state_is_resumable(payload):
            return {
                "status": "resumable",
                "resumable": True,
                "state_path": str(state_path.resolve()),
                "state_status": str(payload.get("status") or ""),
                "inspected_paths": inspected,
            }
    return {
        "status": "fresh",
        "resumable": False,
        "state_path": None,
        "state_status": None,
        "inspected_paths": inspected,
    }


def _environment_failure_result(
    *,
    raw: dict[str, Any],
    case_path: Path,
    state_root: Path,
    environment_path: Path,
    stop_reason: str,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": raw.get("case_id"),
        "case_spec_path": str(case_path.resolve()),
        "state_root": str(state_root.resolve()),
        "environment_path": str(environment_path.resolve()),
        "status": "environment_setup_failed",
        "stop_reason": stop_reason,
        "repair_rounds_completed": 0,
    }
    if error:
        result["error"] = error
    return result


def _case_requires_module_config(raw: dict[str, Any]) -> bool:
    if any(str(item).strip() for item in raw.get("expected_modules") or []):
        return True
    for item in raw.get("build_kconfig") or []:
        try:
            _name, value = parse_kconfig_assignment(str(item))
        except ValueError:
            continue
        if value == "m":
            return True
    return False


def _config_modules_enabled(config_path: Path) -> bool:
    if not config_path.exists():
        return False
    for line in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == "CONFIG_MODULES=y":
            return True
    return False


def _ensure_config_modules_enabled(config_path: Path) -> bool:
    text = config_path.read_text(encoding="utf-8", errors="replace") if config_path.exists() else ""
    lines = text.splitlines()
    found = False
    changed = False
    rewritten: list[str] = []
    for line in lines:
        stripped = line.strip()
        if line.startswith("CONFIG_MODULES=") or stripped == "# CONFIG_MODULES is not set":
            found = True
            if line != "CONFIG_MODULES=y":
                rewritten.append("CONFIG_MODULES=y")
                changed = True
            else:
                rewritten.append(line)
            continue
        rewritten.append(line)
    if not found:
        rewritten.append("CONFIG_MODULES=y")
        changed = True
    if changed:
        config_path.write_text("\n".join(rewritten).rstrip() + "\n", encoding="utf-8")
    return changed


def _process_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


def _run_dataset_config_make(
    *,
    raw: dict[str, Any],
    old_root: Path,
    build_dir: Path | None,
    state_root: Path,
    target: str,
) -> dict[str, Any]:
    command: list[str] = [str(raw.get("make_bin") or "make")]
    if build_dir is not None:
        command.append(f"O={build_dir}")
    command.append(target)
    log_path = ensure_dir(state_root / "dataset_environment_artifacts") / f"{target}.log"
    try:
        make_bin = find_make_binary(str(raw.get("make_bin") or "") or None)
        command[0] = make_bin
        env = os.environ.copy()
        build_env = raw.get("build_env") or {}
        if isinstance(build_env, dict):
            env.update({str(key): str(value) for key, value in build_env.items()})
        process = subprocess.run(
            command,
            cwd=old_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=1800,
        )
        returncode: int | None = process.returncode
        stdout = process.stdout
        stderr = process.stderr
        ok = process.returncode == 0
        error = None
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = _process_text(exc.stdout)
        stderr = _process_text(exc.stderr)
        ok = False
        error = f"make {target} timed out"
    except Exception as exc:
        returncode = None
        stdout = ""
        stderr = str(exc)
        ok = False
        error = str(exc)

    log_path.write_text(
        "\n".join(
            [
                f"$ {' '.join(command)}",
                f"cwd: {old_root}",
                f"returncode: {returncode}",
                "",
                "stdout:",
                stdout or "",
                "",
                "stderr:",
                stderr or "",
            ]
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    record: dict[str, Any] = {
        "target": target,
        "command": command,
        "cwd": str(old_root),
        "returncode": returncode,
        "ok": ok,
        "log_path": str(log_path.resolve()),
    }
    if error:
        record["error"] = error
    return record


def _reset_tree_from_baseline(*, baseline_old_root: Path, old_root: Path) -> None:
    baseline_root = baseline_old_root.resolve()
    ensure_dir(old_root.parent)
    if old_root.is_symlink() or (old_root.exists() and not old_root.is_dir()):
        old_root.unlink()
    ensure_dir(old_root)
    work_root = old_root.resolve()
    if baseline_root == work_root:
        raise ValueError("baseline_old_root and old_root must be different when resetting")
    command = [
        "rsync",
        "--archive",
        "--delete",
        "--checksum",
        f"{baseline_root}/",
        f"{work_root}/",
    ]
    subprocess.run(command, check=True, timeout=1800)


def _prepare_dataset_case_environment(
    *,
    raw: dict[str, Any],
    case_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    state_root = _raw_case_state_root(raw)
    environment_path = state_root / "dataset_environment.json"
    report: dict[str, Any] = {
        "case_id": raw.get("case_id"),
        "case_spec_path": str(case_path.resolve()),
        "state_root": str(state_root.resolve()),
        "environment_path": str(environment_path.resolve()),
        "started_at": utc_now(),
        "status": "ok",
        "stop_reason": None,
        "actions": [],
        "commands": [],
        "old_root": raw.get("old_root"),
        "build_dir": raw.get("build_dir"),
        "base_build_config": str(args.base_build_config.expanduser()) if args.base_build_config else None,
        "base_build_dir": str(args.base_build_dir.expanduser()) if args.base_build_dir else None,
        "base_module_symvers": str(args.base_module_symvers.expanduser()) if args.base_module_symvers else None,
        "baseline_old_root": str(args.baseline_old_root.expanduser()) if args.baseline_old_root else None,
        "reset_old_root": bool(args.reset_old_root),
        "reset_build_dir": bool(args.reset_build_dir),
        "no_init_build_dir": bool(args.no_init_build_dir),
        "require_module_symvers": bool(args.require_module_symvers),
        "allow_missing_module_symvers": bool(args.allow_missing_module_symvers),
    }
    ensure_dir(state_root)
    resume_guard = _dataset_case_resume_guard(state_root)
    report["resume_guard"] = resume_guard

    def fail(stop_reason: str, error: str | None = None) -> dict[str, Any]:
        report["status"] = "failed"
        report["stop_reason"] = stop_reason
        if error:
            report["error"] = error
        _write_dataset_environment_report(report)
        return report

    if resume_guard["status"] == "invalid":
        return fail(
            "invalid_resume_state",
            str(resume_guard.get("error") or "persisted migration state is invalid"),
        )
    resume_existing = bool(resume_guard["resumable"])

    raw_old_root = str(raw.get("old_root") or "").strip()
    if not raw_old_root:
        return fail("missing_old_root", "case old_root is empty")
    old_root = Path(raw_old_root).expanduser()
    baseline_old_root = args.baseline_old_root.expanduser() if args.baseline_old_root else None

    if resume_existing and not old_root.exists():
        return fail(
            "resumable_old_root_missing",
            (
                "persisted migration state requires the existing old_root; "
                f"refusing to replace it from baseline: {old_root}"
            ),
        )
    if args.reset_old_root and resume_existing:
        report["actions"].append(
            {
                "name": "preserve_resumable_old_root",
                "path": str(old_root.resolve()),
                "requested_action": "reset_old_root",
                "state_path": resume_guard["state_path"],
            }
        )
    elif args.reset_old_root:
        if baseline_old_root is None or not baseline_old_root.exists():
            return fail("missing_baseline_old_root", "reset-old-root requires an existing baseline-old-root")
        try:
            _reset_tree_from_baseline(baseline_old_root=baseline_old_root, old_root=old_root)
        except Exception as exc:
            return fail("old_root_reset_failed", str(exc))
        report["actions"].append(
            {
                "name": "reset_old_root",
                "strategy": "content_verified_mirror",
                "from": str(baseline_old_root.resolve()),
                "to": str(old_root.resolve()),
            }
        )
    elif not old_root.exists():
        if baseline_old_root is None:
            return fail("missing_old_root", f"old_root does not exist: {old_root}")
        if not baseline_old_root.exists():
            return fail("missing_baseline_old_root", f"baseline_old_root does not exist: {baseline_old_root}")
        try:
            _reset_tree_from_baseline(baseline_old_root=baseline_old_root, old_root=old_root)
        except Exception as exc:
            return fail("old_root_reset_failed", str(exc))
        report["actions"].append(
            {
                "name": "seed_old_root",
                "strategy": "content_verified_mirror",
                "from": str(baseline_old_root.resolve()),
                "to": str(old_root.resolve()),
            }
        )

    if not old_root.exists():
        return fail("missing_old_root", f"old_root does not exist after preparation: {old_root}")
    if not old_root.is_dir():
        return fail("missing_old_root", f"old_root is not a directory: {old_root}")

    raw_driver_root = str(raw.get("driver_root") or "").strip()
    if raw_driver_root:
        try:
            driver_root = normalize_relpath(raw_driver_root)
        except Exception:
            driver_root = raw_driver_root.strip("/")
        driver_root_path = old_root / driver_root
        if not driver_root_path.exists():
            try:
                driver_root_path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                return fail("driver_root_create_failed", str(exc))
            report["actions"].append(
                {
                    "name": "create_missing_driver_root",
                    "path": str(driver_root_path.resolve()),
                }
            )

    build_dir = Path(str(raw["build_dir"])).expanduser() if raw.get("build_dir") else None
    config_dir = build_dir if build_dir is not None else old_root
    config_path = config_dir / ".config"
    if (
        args.reset_build_dir
        and resume_existing
        and build_dir is not None
        and build_dir.exists()
    ):
        report["actions"].append(
            {
                "name": "preserve_resumable_build_dir",
                "path": str(build_dir.resolve()),
                "requested_action": "reset_build_dir",
                "state_path": resume_guard["state_path"],
            }
        )
    elif args.reset_build_dir and build_dir is not None and build_dir.exists():
        build_dir_resolved = build_dir.resolve()
        old_root_resolved = old_root.resolve()
        if build_dir_resolved == old_root_resolved or old_root_resolved.is_relative_to(build_dir_resolved):
            return fail("reset_build_dir_refused", f"refusing to reset build_dir that contains old_root: {build_dir}")
        try:
            if build_dir.is_dir():
                shutil.rmtree(build_dir)
            else:
                build_dir.unlink()
        except Exception as exc:
            return fail("reset_build_dir_failed", str(exc))
        report["actions"].append({"name": "reset_build_dir", "path": str(build_dir.resolve())})
    ensure_dir(config_dir)

    if not config_path.exists():
        copied_config = False
        if args.base_build_config is not None:
            source_config = args.base_build_config.expanduser()
            if not source_config.is_file():
                return fail("missing_base_config", f"base build config does not exist: {source_config}")
            try:
                shutil.copy2(source_config, config_path)
            except Exception as exc:
                return fail("config_copy_failed", str(exc))
            copied_config = True
            report["actions"].append(
                {
                    "name": "copy_base_build_config",
                    "from": str(source_config.resolve()),
                    "to": str(config_path.resolve()),
                }
            )
        elif args.base_build_dir is not None:
            source_config = args.base_build_dir.expanduser() / ".config"
            if not source_config.is_file():
                return fail("missing_base_config", f"base build dir .config does not exist: {source_config}")
            try:
                shutil.copy2(source_config, config_path)
            except Exception as exc:
                return fail("config_copy_failed", str(exc))
            copied_config = True
            report["actions"].append(
                {
                    "name": "copy_base_build_dir_config",
                    "from": str(source_config.resolve()),
                    "to": str(config_path.resolve()),
                }
            )
        elif not args.no_init_build_dir:
            record = _run_dataset_config_make(
                raw=raw,
                old_root=old_root,
                build_dir=build_dir,
                state_root=state_root,
                target="defconfig",
            )
            report["commands"].append(record)
            if not record.get("ok"):
                return fail("defconfig_failed", str(record.get("error") or "make defconfig failed"))
        else:
            return fail("missing_base_config", f"build config does not exist and init is disabled: {config_path}")
        if (copied_config or not args.no_init_build_dir) and not config_path.exists():
            return fail("defconfig_failed", f".config was not created: {config_path}")

    modules_required = _case_requires_module_config(raw)
    report["modules_required"] = modules_required
    report["config_path"] = str(config_path.resolve())
    case_payload_changed = False
    if modules_required:
        changed = _ensure_config_modules_enabled(config_path)
        report["actions"].append(
            {
                "name": "ensure_config_modules",
                "path": str(config_path.resolve()),
                "changed": changed,
            }
        )
        record = _run_dataset_config_make(
            raw=raw,
            old_root=old_root,
            build_dir=build_dir,
            state_root=state_root,
            target="olddefconfig",
        )
        report["commands"].append(record)
        if not record.get("ok"):
            return fail("olddefconfig_failed", str(record.get("error") or "make olddefconfig failed"))
        if not _config_modules_enabled(config_path):
            return fail("olddefconfig_failed", "CONFIG_MODULES is not enabled after olddefconfig")

        module_symvers_path = config_dir / "Module.symvers"
        source_module_symvers: Path | None = None
        if args.base_module_symvers is not None:
            source_module_symvers = args.base_module_symvers.expanduser()
            if not source_module_symvers.is_file():
                return fail("missing_module_symvers", f"base Module.symvers does not exist: {source_module_symvers}")
        elif args.base_build_dir is not None:
            candidate = args.base_build_dir.expanduser() / "Module.symvers"
            if candidate.is_file():
                source_module_symvers = candidate

        if source_module_symvers is not None:
            try:
                shutil.copy2(source_module_symvers, module_symvers_path)
            except Exception as exc:
                return fail("module_symvers_copy_failed", str(exc))
            report["actions"].append(
                {
                    "name": "copy_module_symvers",
                    "from": str(source_module_symvers.resolve()),
                    "to": str(module_symvers_path.resolve()),
                }
            )

        report["module_symvers_path"] = str(module_symvers_path.resolve())
        report["module_symvers_exists"] = module_symvers_path.is_file()
        if args.allow_missing_module_symvers and not module_symvers_path.is_file():
            build_env = raw.get("build_env") if isinstance(raw.get("build_env"), dict) else {}
            build_env = {str(key): str(value) for key, value in dict(build_env).items()}
            if build_env.get("KBUILD_MODPOST_WARN") != "1":
                build_env["KBUILD_MODPOST_WARN"] = "1"
                raw["build_env"] = build_env
                case_payload_changed = True
            report["actions"].append(
                {
                    "name": "allow_missing_module_symvers",
                    "effect": "set KBUILD_MODPOST_WARN=1 for build phases",
                }
            )
        elif args.require_module_symvers and not module_symvers_path.is_file():
            return fail(
                "missing_module_symvers",
                (
                    f"Module.symvers does not exist: {module_symvers_path}; "
                    "provide --base-module-symvers or --base-build-dir with Module.symvers"
                ),
            )

    if case_payload_changed:
        write_json(case_path, raw)
        report["actions"].append(
            {
                "name": "rewrite_case_build_env",
                "path": str(case_path.resolve()),
            }
        )

    missing_old_entries = [
        normalize_relpath(str(entry))
        for entry in raw.get("entry_files") or []
        if str(entry).strip() and not (old_root / normalize_relpath(str(entry))).is_file()
    ]
    if missing_old_entries:
        try:
            compile_context = materialize_transplanted_compile_context(raw)
        except Exception as exc:
            return fail("compile_context_materialization_failed", str(exc))
        report["compile_context_materialization"] = compile_context
        report["actions"].append(
            {
                "name": "materialize_transplanted_compile_context",
                **compile_context,
            }
        )
        if compile_context.get("status") == "failed":
            return fail(
                str(compile_context.get("stop_reason") or "compile_context_materialization_failed"),
                json.dumps(compile_context.get("unresolved") or [], ensure_ascii=False),
            )

    _write_dataset_environment_report(report)
    return report


def run_dataset_migrate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.all and not (
        args.rows or args.types or args.names or args.paths or args.limit
    ):
        raise ValueError("dataset-migrate requires a selector (--rows/--type/--name/--path/--limit) or --all")
    if args.dataset_workspace and not args.compile_commands_label:
        raise ValueError("dataset-migrate with --dataset-workspace requires --compile-commands-label")
    manifest = _materialize_dataset_cases_from_args(args)
    if args.dry_run:
        manifest["dry_run"] = True
        return manifest

    results: list[dict[str, Any]] = []
    for item in manifest.get("cases") or []:
        case_path = Path(str(item.get("case_spec_path") or ""))
        raw_case = json.loads(case_path.resolve().read_text(encoding="utf-8"))
        environment = _prepare_dataset_case_environment(raw=raw_case, case_path=case_path, args=args)
        environment_path = Path(str(environment["environment_path"]))
        if environment.get("status") != "ok":
            result_item = _environment_failure_result(
                raw=raw_case,
                case_path=case_path,
                state_root=Path(str(environment["state_root"])),
                environment_path=environment_path,
                stop_reason=str(environment.get("stop_reason") or "environment_setup_failed"),
                error=str(environment.get("error") or ""),
            )
            results.append(result_item)
            if args.stop_on_failure:
                break
            continue

        try:
            spec = CaseSpec.from_file(case_path)
        except Exception as exc:
            environment["status"] = "failed"
            environment["stop_reason"] = "case_spec_validation_failed"
            environment["error"] = str(exc)
            _write_dataset_environment_report(environment)
            result_item = _environment_failure_result(
                raw=raw_case,
                case_path=case_path,
                state_root=Path(str(environment["state_root"])),
                environment_path=environment_path,
                stop_reason="case_spec_validation_failed",
                error=str(exc),
            )
            results.append(result_item)
            if args.stop_on_failure:
                break
            continue

        case_state_root = state_root_for(spec, None)
        _emit_progress(f"dataset migration start: case_id={spec.case_id} state_root={case_state_root}")
        result = auto_migrate(
            spec,
            state_root=case_state_root,
            max_repair_rounds=args.max_repair_rounds,
            allow_escalation=args.allow_escalation,
            model=args.model,
            max_completion_tokens=args.max_completion_tokens,
            max_build_config_preflight_attempts=args.build_config_preflight_attempts,
            max_semantic_repair_rounds=args.max_semantic_repair_rounds,
        )
        result_path = case_state_root / "migration_result.json"
        write_json(result_path, result)
        results.append(
            {
                "case_id": spec.case_id,
                "case_spec_path": str(case_path.resolve()),
                "state_root": str(case_state_root.resolve()),
                "result_path": str(result_path.resolve()),
                "environment_path": str(environment_path.resolve()),
                "status": result.get("status"),
                "stop_reason": result.get("stop_reason"),
                "repair_rounds_completed": result.get("repair_rounds_completed"),
                "compile_repair_validated": result.get(
                    "compile_repair_validated"
                ),
                "compile_repair_rounds_completed": result.get(
                    "compile_repair_rounds_completed"
                ),
                "semantic_repair_rounds_completed": result.get(
                    "semantic_repair_rounds_completed"
                ),
                "final_compile_validated": result.get(
                    "final_compile_validated"
                ),
                "semantic_complete": result.get(
                    "semantic_complete"
                ),
            }
        )
        if args.stop_on_failure and str(result.get("stop_reason") or "") != "validated":
            break

    completed = [item for item in results if str(item.get("stop_reason") or "") == "validated"]
    stopped = [item for item in results if str(item.get("stop_reason") or "") != "validated"]
    output_dir = Path(str(manifest.get("output_dir") or args.output_dir)).expanduser()
    summary = {
        "dataset_path": manifest.get("dataset_path"),
        "manifest_path": str((output_dir / "manifest.json").resolve()),
        "selected_count": manifest.get("selected_count"),
        "completed_count": len(completed),
        "stopped_count": len(stopped),
        "results": results,
        "batch_note": (
            "dataset-migrate prepares old_root and build_dir before CaseSpec validation; "
            "old_root/build_dir reset is opt-in via --reset-old-root and --reset-build-dir"
        ),
    }
    write_json(output_dir / "dataset_migration_results.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentic driver backport migration harness."
    )
    parser.add_argument("--state-root", type=Path, help="override the state root directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_migrate_parser = subparsers.add_parser(
        "dataset-migrate",
        help="generate dataset case specs and run auto migration for the selected entries",
    )
    _add_dataset_case_args(dataset_migrate_parser)
    dataset_migrate_parser.add_argument("--all", action="store_true", help="run all dataset entries")
    dataset_migrate_parser.add_argument("--dry-run", action="store_true", help="only generate cases and manifest")
    dataset_migrate_parser.add_argument("--stop-on-failure", action="store_true")
    dataset_migrate_parser.add_argument("--baseline-old-root", type=Path, help="clean old kernel source used to seed old_root")
    dataset_migrate_parser.add_argument(
        "--reset-old-root",
        action="store_true",
        help="restore old_root to the content-verified baseline before each case",
    )
    dataset_migrate_parser.add_argument("--base-build-config", type=Path, help="template .config for each case build_dir")
    dataset_migrate_parser.add_argument("--base-build-dir", type=Path, help="build directory whose .config is used as template")
    dataset_migrate_parser.add_argument("--base-module-symvers", type=Path, help="template Module.symvers copied into each case build_dir")
    dataset_migrate_parser.add_argument(
        "--require-module-symvers",
        action="store_true",
        help="fail environment setup for module cases when Module.symvers is unavailable",
    )
    dataset_migrate_parser.add_argument(
        "--allow-missing-module-symvers",
        action="store_true",
        help="set KBUILD_MODPOST_WARN=1 when Module.symvers is unavailable; useful for compile-only iteration, not final load proof",
    )
    dataset_migrate_parser.add_argument("--reset-build-dir", action="store_true", help="clear each case build_dir before initialization")
    dataset_migrate_parser.add_argument("--no-init-build-dir", action="store_true", help="do not run make defconfig when .config is missing")
    _add_auto_runtime_args(dataset_migrate_parser)

    auto_parser = subparsers.add_parser(
        "auto-migrate",
        help="run transplant, compile validation, and bounded graph-rooted repair rounds end-to-end",
    )
    _add_auto_migrate_args(auto_parser)

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "dataset-migrate":
        result = run_dataset_migrate(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("dry_run") or int(result.get("stopped_count") or 0) == 0 else 1

    spec = CaseSpec.from_file(args.case_spec)
    state_root = state_root_for(spec, args.state_root)
    result = auto_migrate(
        spec,
        state_root=state_root,
        max_repair_rounds=args.max_repair_rounds,
        allow_escalation=args.allow_escalation,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        max_build_config_preflight_attempts=args.build_config_preflight_attempts,
        max_semantic_repair_rounds=args.max_semantic_repair_rounds,
    )
    result_path = state_root / "migration_result.json"
    write_json(result_path, result)
    print(
        "migration result: "
        f"status={result.get('status')} "
        f"stop_reason={result.get('stop_reason')} "
        f"compile_repair_rounds_completed="
        f"{result.get('compile_repair_rounds_completed')} "
        f"semantic_repair_rounds_completed="
        f"{result.get('semantic_repair_rounds_completed')} "
        f"compile_repair_validated="
        f"{result.get('compile_repair_validated')} "
        f"final_compile_validated="
        f"{result.get('final_compile_validated')} "
        f"progress={result.get('progress_path')} "
        f"result={result_path.resolve()}"
    )
    return 0 if str(result.get("stop_reason") or "") == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
