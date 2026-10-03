from __future__ import annotations

import hashlib
import json
import difflib
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Protocol

from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_strings as _ordered_strings,
    stable_id,
    stable_json_sha256,
    utc_now,
)
from driver_backport.core.content_store import parse_content_record_ref
from driver_backport.core.executor import (
    backup_file_once,
    finalize_transaction,
    load_transaction,
    rollback_transaction,
    run_unified_diff,
    start_transaction,
)
from driver_backport.analysis.conditional_contract import (
    preprocessor_conditional_integrity,
    preprocessor_conditional_regressions,
)
from driver_backport.analysis.code_index import (
    CodeIndex,
    TYPED_DIAGNOSTIC_PROVIDER_RESOLUTION_REVISION,
)
from driver_backport.analysis.provider_contract import (
    PROVIDER_CONTRACT_POLICY_REVISION,
    diagnostic_identifier_evidence,
)
from driver_backport.llm.model_util import (
    ModelAPIError,
    model_api_error_metadata,
)
from driver_backport.migration.semantic_closure_evaluator import (
    COMPILE_VALID_HANDOFF_EVALUATOR_REVISION,
    evaluate_active_batch_semantic_contracts,
)
from driver_backport.migration.owner_key import (
    semantic_mutable_owner_key,
)
from driver_backport.migration.initial_compatibility import (
    _capture_structural_baseline,
    _validate_structural_postconditions,
)
from driver_backport.migration.semantic_closure_patch import (
    validate_semantic_patch_boundary,
)
from driver_backport.migration.semantic_closure_projection import (
    materialize_semantic_work_units,
)
from driver_backport.migration.semantic_closure_state import (
    BATCH_COMPILE_FAILED,
    BATCH_OPEN,
    BATCH_ROLLED_BACK,
    CANONICAL_COMPILER_RECOVERY,
    CHANGE_REQUIRED,
    PATCH_PENDING_COMPILE,
    PENDING_DECISION,
    ROUND_BUDGET_EXHAUSTED,
    ROUND_COMPILE_FAILED,
    ROUND_NO_PROGRESS,
    ROUND_RUNNING,
    ROUND_VALIDATED,
    SEMANTIC_GAP_SWEEP,
    SEMANTIC_COMPILE_RECOVERY_POLICY_REVISION,
    SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION,
    SEMANTIC_NO_CHANGE_PROOF_SCHEMA,
    activate_semantic_gap_snapshot,
    begin_or_resume_patch_batch,
    begin_semantic_round,
    completed_semantic_recovery_rounds,
    consume_pending_repair_sweep,
    dismiss_pending_repair_sweep,
    finish_semantic_round,
    record_batch_compile_validation,
    record_batch_patch,
    record_exact_owner_expansion,
    record_gap_decisions,
    record_gap_review_failure,
    record_unit_attempt,
    record_unit_evidence_action,
    record_unit_replan_evidence,
    reopen_gaps_for_authoritative_noop,
    reconcile_active_batch_affected_gaps,
    rollback_active_batch,
    semantic_closure_summary,
    semantic_repair_resume_context,
    normalize_semantic_owner_bindings,
    semantic_unit_action_count,
)
from driver_backport.migration.semantic_closure_store import (
    load_semantic_closure_state,
    load_semantic_record,
    put_semantic_record,
    semantic_record_kind,
    source_units_from_state,
    write_semantic_closure_state,
)
from driver_backport.planning.response_apply import (
    apply_single_generated_diff,
    touched_files_from_diff,
)
from driver_backport.planning.dependency_schedule import dependency_order
from driver_backport.planning.execution_phase import (
    EXECUTION_PHASE_SEMANTIC_CLOSURE,
)


SEMANTIC_CLOSURE_CONTROLLER_SCHEMA = "semantic_closure_controller_v1"
SEMANTIC_ACTION_EXECUTION_POLICY_REVISION = (
    "semantic_action_execution_policy_v5"
)
SEMANTIC_UNIT_ACTION_LIMIT = 3
SEMANTIC_REVIEW_ATTEMPT_LIMIT = 3


def _conditional_integrity_reports(
    spec: object,
    relative_paths: list[str],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for relative_path in _ordered_strings(relative_paths):
        path = Path(getattr(spec, "resolve_old")(relative_path))
        if path.suffix.lower() not in {
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hh",
            ".hpp",
            ".inc",
            ".s",
        }:
            continue
        lines = (
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
            if path.is_file()
            else []
        )
        reports[relative_path] = preprocessor_conditional_integrity(
            lines
        )
    return reports


class SemanticClosureRuntime(Protocol):
    """Model, graph-tool, and canonical compile adapter."""

    def review(self, request: dict[str, Any]) -> dict[str, Any]:
        ...

    def repair(self, request: dict[str, Any]) -> dict[str, Any]:
        ...

    def validate(self, request: dict[str, Any]) -> dict[str, Any]:
        ...


def _compact_entity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: deepcopy(value.get(key))
        for key in (
            "id",
            "tree",
            "file",
            "contract_key",
            "key",
            "kind",
            "symbol",
            "is_definition",
            "extent",
            "signature_fingerprint",
            "implementation_fingerprint",
            "interface_fingerprint",
            "syntax_fingerprint",
            "source_fingerprint",
            "digest",
        )
        if value.get(key) not in (None, "", [], {})
    }


def _compact_reference(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: deepcopy(value.get(key))
        for key in (
            "id",
            "source_file",
            "line",
            "column",
            "extent",
            "symbol",
            "kind",
            "use_kind",
            "contract_key",
            "translation_unit_file",
        )
        if value.get(key) not in (None, "", [], {})
    } | {
        "owner_entity": _compact_entity(value.get("owner_entity")),
        "referenced_entity": _compact_entity(
            value.get("referenced_entity")
        ),
    }


def _compact_contract_evidence(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: deepcopy(evidence.get(key))
        for key in (
            "id",
            "contract_identity",
            "kind",
            "symbol",
            "dependency_scope",
            "semantic_requirement",
            "comparison",
            "resolution",
            "alignment",
        )
        if evidence.get(key) not in (None, "", [], {})
    } | {
        "old_entity": _compact_entity(evidence.get("old_entity")),
        "donor_entity": _compact_entity(evidence.get("donor_entity")),
        "old_declarations": [
            _compact_entity(item)
            for item in evidence.get("old_declarations") or []
            if isinstance(item, dict)
        ],
        "donor_declarations": [
            _compact_entity(item)
            for item in evidence.get("donor_declarations") or []
            if isinstance(item, dict)
        ],
        "old_exports": [
            deepcopy(item)
            for item in evidence.get("old_exports") or []
            if isinstance(item, dict)
        ],
        "donor_exports": [
            deepcopy(item)
            for item in evidence.get("donor_exports") or []
            if isinstance(item, dict)
        ],
        "consumer_references": [
            _compact_reference(item)
            for item in evidence.get("consumer_references") or []
            if isinstance(item, dict)
        ],
    }


def _compact_evaluation(
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    compact = {
        key: deepcopy(evaluation.get(key))
        for key in (
            "schema",
            "gap_id",
            "contract_identity",
            "checkpoint_ref",
            "status",
            "required_dimensions",
            "relations",
            "unresolved_dimensions",
            "unavailable_dimensions",
        )
        if evaluation.get(key) not in (None, "", [], {})
    }
    observations = (
        evaluation.get("observations")
        if isinstance(evaluation.get("observations"), dict)
        else {}
    )
    integration = (
        observations.get("integration_reference")
        if isinstance(observations.get("integration_reference"), dict)
        else {}
    )
    if integration:
        compact["observations"] = {
            "integration_reference": {
                key: deepcopy(integration.get(key))
                for key in (
                    "reference_locator_shapes",
                    "donor_reference_shapes",
                    "current_reference_shapes",
                    "missing_reference_shapes",
                    "target_representation_agnostic",
                    "comparison_limit",
                    "reason",
                )
                if integration.get(key) not in (None, "", [], {})
            }
        }
    return compact


def _work_unit_context(
    state_root: Path,
    state: dict[str, Any],
    work_unit: dict[str, Any],
) -> dict[str, Any]:
    gaps: list[dict[str, Any]] = []
    for gap_id in work_unit.get("gap_ids") or []:
        gap = (state.get("gaps") or {}).get(str(gap_id))
        if not isinstance(gap, dict):
            raise ValueError(
                f"semantic work unit references unknown gap: {gap_id}"
            )
        evidence_ref = str(
            gap.get("contract_evidence_ref") or ""
        )
        evidence = load_semantic_record(state_root, evidence_ref)
        evaluation_summary = next(
            (
                evaluation
                for evaluation in (
                    gap.get("post_batch_contract_evaluation"),
                    gap.get("compile_valid_handoff_evaluation"),
                )
                if isinstance(evaluation, dict)
                and str(evaluation.get("evaluation_ref") or "")
            ),
            {},
        )
        evaluation_ref = str(
            evaluation_summary.get("evaluation_ref") or ""
        )
        evaluation = (
            load_semantic_record(state_root, evaluation_ref)
            if evaluation_ref
            else {}
        )
        gaps.append(
            {
                "id": str(gap_id),
                "contract_identity": gap.get("contract_identity"),
                "status": gap.get("status"),
                "raw_hunk_ids": deepcopy(
                    gap.get("raw_hunk_ids") or []
                ),
                "decision": deepcopy(gap.get("decision") or {}),
                "review_failures": deepcopy(
                    gap.get("review_failures") or []
                ),
                "authoritative_noop_evidence_refs": deepcopy(
                    gap.get("authoritative_noop_evidence_refs") or []
                ),
                "owner_bindings": deepcopy(
                    gap.get("owner_bindings") or []
                ),
                "contract_evidence_ref": evidence_ref,
                "contract_evidence": _compact_contract_evidence(
                    evidence
                    if isinstance(evidence, dict)
                    else {}
                ),
                "compile_valid_evaluation_ref": (
                    evaluation_ref or None
                ),
                "compile_valid_evaluation": _compact_evaluation(
                    evaluation
                    if isinstance(evaluation, dict)
                    else {}
                ),
            }
        )
    return {
        "schema": "semantic_closure_unit_context_v1",
        "work_unit": deepcopy(work_unit),
        "gaps": gaps,
    }


def _refresh_work_unit_scope_from_state(
    state: dict[str, Any],
    work_unit: dict[str, Any],
) -> dict[str, Any]:
    """Refresh exact owners after a graph-backed scope expansion."""

    gap_ids = _ordered_strings(
        list(work_unit.get("gap_ids") or [])
    )
    bindings: dict[str, dict[str, Any]] = {}
    for binding in work_unit.get("owner_bindings") or []:
        if (
            isinstance(binding, dict)
            and str(binding.get("owner_key") or "")
            and str(binding.get("authority") or "")
            == "typed_direct_dependency_edge"
        ):
            bindings[str(binding["owner_key"])] = deepcopy(binding)
    base_allowed_fileset_present = "base_allowed_fileset" in work_unit
    allowed_files = _ordered_strings(
        list(
            work_unit.get("base_allowed_fileset")
            if base_allowed_fileset_present
            else work_unit.get("allowed_fileset")
            or []
        )
    )
    for binding in bindings.values():
        relative_path = str(binding.get("file") or "")
        if relative_path and relative_path not in allowed_files:
            allowed_files.append(relative_path)
    authoritative_gap_seen = False
    for gap_id in gap_ids:
        gap = (state.get("gaps") or {}).get(gap_id)
        if not isinstance(gap, dict):
            continue
        authoritative_gap_seen = True
        for binding in gap.get("owner_bindings") or []:
            if not isinstance(binding, dict):
                continue
            owner_key = str(binding.get("owner_key") or "")
            if owner_key:
                previous = bindings.get(owner_key, {})
                refreshed = {**previous, **deepcopy(binding)}
                for field in ("exact_node_refs", "semantic_gap_ids"):
                    refreshed[field] = _ordered_strings(
                        [
                            *list(previous.get(field) or []),
                            *list(binding.get(field) or []),
                        ]
                    )
                bindings[owner_key] = refreshed
            relative_path = str(binding.get("file") or "")
            if relative_path and relative_path not in allowed_files:
                allowed_files.append(relative_path)
    if not authoritative_gap_seen:
        bindings = {
            str(item.get("owner_key") or ""): deepcopy(item)
            for item in work_unit.get("owner_bindings") or []
            if isinstance(item, dict)
            and str(item.get("owner_key") or "")
        }
    return {
        **work_unit,
        "owner_bindings": list(bindings.values()),
        "allowed_fileset": allowed_files,
    }


def _dependency_order(
    work_units: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered, schedule = dependency_order(work_units)
    if int(schedule.get("cyclic_component_count") or 0):
        raise ValueError(
            "semantic closure work-unit dependency graph has a cycle"
        )
    return ordered


def _active_batch(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    batch_id = str(state.get("active_batch_id") or "")
    if not batch_id:
        return None
    batch = (state.get("batches") or {}).get(batch_id)
    if not isinstance(batch, dict):
        raise ValueError("active semantic batch is missing")
    return batch


def semantic_compile_recovery_context(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the exact failed-batch handoff needed by compiler recovery."""

    batch = _active_batch(state)
    if batch is None or batch.get("status") != BATCH_COMPILE_FAILED:
        return None
    validations = [
        item
        for item in batch.get("compile_validations") or []
        if isinstance(item, dict)
    ]
    return {
        "schema": "semantic_compile_recovery_context_v1",
        "batch_id": str(batch["id"]),
        "batch_status": str(batch["status"]),
        "compile_validation_ref": (
            validations[-1].get("ref") if validations else None
        ),
        "compile_failure_round": (
            validations[-1].get("round") if validations else None
        ),
        "gap_ids": sorted(
            {
                str(gap_id)
                for field in ("primary_gap_ids", "affected_gap_ids")
                for gap_id in batch.get(field) or []
                if str(gap_id)
            }
        ),
    }


def semantic_closure_resume_route(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Derive the only valid next semantic action from durable state.

    This is a state projection, not an error classifier.  In particular, an
    open mutation batch must return to the semantic controller before any
    caller is allowed to run a generic validation cycle.
    """

    summary = semantic_closure_summary(state)
    batch = _active_batch(state)
    pending_decision_gap_ids = _gap_ids_with_status(
        state,
        PENDING_DECISION,
    )
    change_required_gap_ids = _gap_ids_with_status(
        state,
        CHANGE_REQUIRED,
    )
    patch_pending_gap_ids = _gap_ids_with_status(
        state,
        PATCH_PENDING_COMPILE,
    )
    running_round = next(
        (
            item
            for item in reversed(state.get("rounds") or [])
            if isinstance(item, dict)
            and item.get("status") == ROUND_RUNNING
        ),
        None,
    )

    if batch is not None and batch.get("status") == BATCH_COMPILE_FAILED:
        next_action = "resume_compile_recovery"
        stage = "compile_recovery"
    elif (
        batch is not None
        or running_round is not None
        or pending_decision_gap_ids
        or change_required_gap_ids
        or patch_pending_gap_ids
        or state.get("pending_repair_sweep") is not None
    ):
        next_action = "resume_semantic_controller"
        if pending_decision_gap_ids:
            stage = "decision"
        elif change_required_gap_ids:
            stage = "mutation"
        else:
            stage = "unified_validation"
    elif summary["static_semantic_closed"]:
        next_action = "semantic_closed"
        stage = "complete"
    else:
        next_action = "semantic_incomplete"
        stage = "terminal_incomplete"

    return {
        "schema": "semantic_closure_resume_route_v1",
        "next_action": next_action,
        "stage": stage,
        "model_required": bool(
            pending_decision_gap_ids or change_required_gap_ids
        ),
        "active_batch_id": (
            str(batch.get("id") or "") if batch is not None else None
        ),
        "active_batch_status": (
            str(batch.get("status") or "")
            if batch is not None
            else None
        ),
        "running_round": (
            int(running_round.get("number") or 0)
            if running_round is not None
            else None
        ),
        "pending_decision_gap_ids": pending_decision_gap_ids,
        "change_required_gap_ids": change_required_gap_ids,
        "patch_pending_gap_ids": patch_pending_gap_ids,
        "summary": summary,
    }


def _write_state(
    state_root: Path,
    state: dict[str, Any],
) -> None:
    write_semantic_closure_state(state_root, state)


def _store_runtime_result(
    state_root: Path,
    *,
    kind: str,
    result: dict[str, Any],
) -> str:
    return put_semantic_record(
        state_root,
        kind=kind,
        record=result,
    )


def _model_runtime_failure(
    exc: ModelAPIError,
    *,
    operation: str,
    round_number: int,
    work_unit_id: str,
) -> dict[str, Any]:
    """Preserve a provider failure without turning it into repair evidence."""

    metadata = model_api_error_metadata(exc)
    return {
        "schema": "semantic_model_runtime_failure_v1",
        "status": "runtime_unavailable",
        "operation": str(operation),
        "round": int(round_number),
        "work_unit_id": str(work_unit_id),
        "error": f"{exc.__class__.__name__}: {exc}",
        "error_type": exc.__class__.__name__,
        "model_api_error": metadata,
        "retryable": bool(metadata.get("retryable")),
        "at": utc_now(),
    }


def _persisted_patch_diff_for_replay(
    state_root: Path,
    state: dict[str, Any],
    patch: dict[str, Any],
) -> tuple[str, str]:
    kind = str(patch.get("kind") or "")
    if kind == "semantic_repair":
        work_unit_id = str(patch.get("work_unit_id") or "")
        direct_ref = str(patch.get("patch_evidence_ref") or "")
        if direct_ref and semantic_record_kind(direct_ref) != (
            "semantic-repair-action"
        ):
            raise ValueError(
                "semantic batch replay patch has an invalid action record"
            )
        action_refs = (
            {direct_ref}
            if direct_ref
            else {
                str(reference)
                for round_record in state.get("rounds") or []
                if isinstance(round_record, dict)
                for unit_id, unit in (round_record.get("units") or {}).items()
                if str(unit_id) == work_unit_id and isinstance(unit, dict)
                for attempt in unit.get("attempts") or []
                if isinstance(attempt, dict)
                and attempt.get("outcome") == "applied"
                for reference in attempt.get("evidence_refs") or []
                if semantic_record_kind(str(reference))
                == "semantic-repair-action"
            }
        )
        if len(action_refs) != 1:
            raise ValueError(
                "semantic batch replay requires one applied action record "
                f"for {work_unit_id!r}, found {len(action_refs)}"
            )
        action_ref = next(iter(action_refs))
        action = load_semantic_record(state_root, action_ref)
        diff_text = str(action.get("diff_text") or "")
        if not diff_text:
            raise ValueError(
                "semantic batch replay action has no immutable diff"
            )
        claimed_gap_ids = set(
            patch.get("claimed_gap_ids") or patch.get("gap_ids") or []
        )
        if set(action.get("gap_ids") or []) != claimed_gap_ids:
            raise ValueError(
                "semantic batch replay gap lineage is inconsistent"
            )
        touched_files = set(touched_files_from_diff(diff_text))
        if touched_files != set(patch.get("touched_files") or []):
            raise ValueError(
                "semantic batch replay touched-file lineage is inconsistent"
            )
        expected_signature = str(
            action.get("patch_signature")
            or stable_id(
                "semantic-patch",
                "semantic_repair",
                stable_json_sha256(diff_text),
                *sorted(claimed_gap_ids),
            )
        )
        if expected_signature != str(patch.get("patch_signature") or ""):
            raise ValueError(
                "semantic batch replay patch signature is inconsistent"
            )
        boundary_ref = str(patch.get("boundary_ref") or "")
        if not boundary_ref:
            raise ValueError(
                "semantic batch replay has no authorization boundary"
            )
        boundary = load_semantic_record(state_root, boundary_ref)
        if str(boundary.get("status") or "") != "authorized":
            raise ValueError(
                "semantic batch replay authorization is not valid"
            )
        boundary_files = {
            str(changed.get("file") or "")
            for changed in boundary.get("changed_ranges") or []
            if isinstance(changed, dict) and str(changed.get("file") or "")
        }
        if boundary_files != touched_files:
            raise ValueError(
                "semantic batch replay authorization lineage is inconsistent"
            )
        return diff_text, action_ref

    if kind == "compile_recovery":
        transaction_ref = str(patch.get("transaction_ref") or "")
        if not transaction_ref:
            raise ValueError(
                "compiler recovery replay has no transaction reference"
            )
        manifest = load_transaction(Path(transaction_ref)).manifest()
        patch_path = Path(str(manifest.get("patch_path") or ""))
        if not patch_path.is_file():
            raise ValueError(
                "compiler recovery replay has no immutable patch artifact"
            )
        return patch_path.read_text(encoding="utf-8"), str(
            patch_path.resolve()
        )

    raise ValueError(
        f"semantic batch replay does not support patch kind {kind!r}"
    )


def _ensure_transaction_replay_patch(
    spec: object,
    transaction_ref: Path,
    *,
    touched_files: list[str],
) -> Path | None:
    """Persist an exact text diff when a transaction has only preimages."""

    transaction = load_transaction(transaction_ref.resolve())
    manifest = transaction.manifest()
    existing = Path(str(manifest.get("patch_path") or ""))
    if existing.is_file():
        return existing.resolve()

    backups = {
        normalize_relpath(str(entry.get("relative_path") or "")): entry
        for entry in manifest.get("entries") or []
        if isinstance(entry, dict)
        and str(entry.get("type") or "") == "backup"
        and normalize_relpath(
            str(entry.get("relative_path") or "")
        )
    }
    normalized_files = sorted(
        {
            normalize_relpath(str(path))
            for path in touched_files
            if normalize_relpath(str(path or ""))
        }
    )
    missing_preimages = sorted(set(normalized_files) - set(backups))
    if missing_preimages:
        raise ValueError(
            "compile recovery transaction has no preimage for: "
            + ", ".join(missing_preimages)
        )

    diff_lines: list[str] = []
    for relative_path in normalized_files:
        entry = backups[relative_path]
        existed_before = bool(entry.get("existed"))
        backup_path = Path(str(entry.get("backup") or ""))
        target_path = Path(getattr(spec, "resolve_old")(relative_path))
        if existed_before and not backup_path.is_file():
            raise FileNotFoundError(
                "compile recovery preimage is missing for "
                f"{relative_path}: {backup_path}"
            )
        before = (
            backup_path.read_text(
                encoding="utf-8",
                errors="surrogateescape",
            ).splitlines()
            if existed_before
            else []
        )
        exists_after = target_path.is_file()
        after = (
            target_path.read_text(
                encoding="utf-8",
                errors="surrogateescape",
            ).splitlines()
            if exists_after
            else []
        )
        if before == after and existed_before == exists_after:
            continue
        diff_lines.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=(
                    f"a/{relative_path}"
                    if existed_before
                    else "/dev/null"
                ),
                tofile=(
                    f"b/{relative_path}"
                    if exists_after
                    else "/dev/null"
                ),
                lineterm="",
                n=3,
            )
        )
    if not diff_lines:
        return None
    diff_text = "\n".join(diff_lines) + "\n"
    patch_path = transaction.root / "semantic-compile-replay.patch"
    patch_path.write_text(
        diff_text,
        encoding="utf-8",
        errors="surrogateescape",
    )
    manifest["patch_path"] = str(patch_path.resolve())
    manifest.setdefault(
        "patch_signature",
        stable_id(
            "semantic-compile-replay-patch",
            stable_json_sha256(diff_text),
        ),
    )
    transaction.save_manifest(manifest)
    return patch_path.resolve()


def reconcile_active_semantic_batch_transaction(
    spec: object,
    *,
    state_root: Path,
) -> dict[str, Any]:
    """Retain or exactly replay the transaction owned by an active batch."""

    state = load_semantic_closure_state(
        state_root,
        case_id=str(getattr(spec, "case_id")),
    )
    if state is None:
        return {"status": "not_required", "reason": "no_semantic_state"}
    batch = _active_batch(state)
    if batch is None:
        return {"status": "not_required", "reason": "no_active_batch"}
    batch_id = str(batch["id"])
    batch_status = str(batch.get("status") or "")
    preimage_ref = Path(str(batch.get("preimage_ref") or ""))
    if not preimage_ref.is_dir():
        return {
            "status": "failed",
            "reason": "active_batch_preimage_transaction_missing",
            "batch_id": batch_id,
            "batch_status": batch_status,
            "preimage_ref": str(preimage_ref),
        }
    transaction = load_transaction(preimage_ref)
    manifest = transaction.manifest()
    transaction_status = str(manifest.get("status") or "open")
    if transaction_status == "open":
        return {
            "status": "retained",
            "reason": "active_semantic_batch_owns_open_transaction",
            "batch_id": batch_id,
            "batch_status": batch_status,
            "transaction_ref": str(preimage_ref.resolve()),
        }
    if transaction_status != "rolled_back":
        return {
            "status": "failed",
            "reason": "active_batch_transaction_has_invalid_status",
            "batch_id": batch_id,
            "batch_status": batch_status,
            "transaction_ref": str(preimage_ref.resolve()),
            "transaction_status": transaction_status,
        }

    backup_entries = {
        str(entry.get("relative_path") or ""): entry
        for entry in manifest.get("entries") or []
        if isinstance(entry, dict)
        and entry.get("type") == "backup"
        and str(entry.get("relative_path") or "")
    }
    touched_files = sorted(
        {
            str(relative_path)
            for patch in batch.get("patches") or []
            if isinstance(patch, dict)
            for relative_path in patch.get("touched_files") or []
            if str(relative_path)
        }
    )
    if set(touched_files) != set(backup_entries):
        return {
            "status": "failed",
            "reason": "active_batch_preimage_file_set_mismatch",
            "batch_id": batch_id,
            "batch_status": batch_status,
            "touched_files": touched_files,
            "preimage_files": sorted(backup_entries),
        }
    for relative_path, entry in backup_entries.items():
        target = Path(getattr(spec, "resolve_old")(relative_path))
        existed = bool(entry.get("existed"))
        if target.exists() != existed:
            return {
                "status": "failed",
                "reason": "current_tree_is_not_active_batch_preimage",
                "file": relative_path,
            }
        if not existed:
            continue
        backup = Path(str(entry.get("backup") or ""))
        if not backup.is_file() or target.read_bytes() != backup.read_bytes():
            return {
                "status": "failed",
                "reason": "current_tree_is_not_active_batch_preimage",
                "file": relative_path,
            }

    replay_patches: list[dict[str, Any]] = []
    try:
        for patch in batch.get("patches") or []:
            if not isinstance(patch, dict):
                continue
            diff_text, evidence_ref = _persisted_patch_diff_for_replay(
                state_root,
                state,
                patch,
            )
            replay_patches.append(
                {
                    "patch_id": str(patch.get("id") or ""),
                    "diff_text": diff_text,
                    "evidence_ref": evidence_ref,
                }
            )
    except (OSError, ValueError, KeyError) as exc:
        return {
            "status": "failed",
            "reason": "active_batch_replay_evidence_invalid",
            "error": f"{exc.__class__.__name__}: {exc}",
        }

    replay_transaction = start_transaction(
        state_root,
        case_id=str(getattr(spec, "case_id")),
        cycle=max(
            [
                int(item.get("number") or 0)
                for item in state.get("rounds") or []
                if isinstance(item, dict)
            ]
            or [1]
        ),
        work_unit_id=stable_id(
            "semantic-batch-postimage-replay",
            batch_id,
            str(preimage_ref.resolve()),
        ),
    )
    for relative_path in touched_files:
        backup_file_once(replay_transaction, spec, relative_path)
    replay_results: list[dict[str, Any]] = []
    for replay_patch in replay_patches:
        returncode, stdout, stderr = run_unified_diff(
            Path(getattr(spec, "old_root_path")),
            str(replay_patch["diff_text"]),
        )
        replay_results.append(
            {
                "patch_id": replay_patch["patch_id"],
                "evidence_ref": replay_patch["evidence_ref"],
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        if returncode != 0:
            rollback = rollback_transaction(spec, replay_transaction.root)
            return {
                "status": "failed",
                "reason": "active_batch_postimage_replay_apply_failed",
                "failed_patch_id": replay_patch["patch_id"],
                "replay_results": replay_results,
                "rollback": rollback,
            }

    replay_record = {
        "schema": "semantic_active_batch_postimage_replay_v1",
        "status": "restored",
        "batch_id": batch_id,
        "batch_status": batch_status,
        "old_preimage_ref": str(preimage_ref.resolve()),
        "new_preimage_ref": str(replay_transaction.root.resolve()),
        "patch_ids": [item["patch_id"] for item in replay_results],
        "evidence_refs": [item["evidence_ref"] for item in replay_results],
        "replayed_at": utc_now(),
    }
    replay_ref = _store_runtime_result(
        state_root,
        kind="semantic-active-batch-postimage-replay",
        result=replay_record,
    )
    batch["preimage_ref"] = str(replay_transaction.root.resolve())
    batch.setdefault("history", []).append(
        {
            "at": utc_now(),
            "event": "active_batch_postimage_replayed",
            "replay_ref": replay_ref,
            "old_preimage_ref": str(preimage_ref.resolve()),
            "new_preimage_ref": str(replay_transaction.root.resolve()),
        }
    )
    _write_state(state_root, state)
    return {
        **replay_record,
        "replay_ref": replay_ref,
        "transaction_ref": str(replay_transaction.root.resolve()),
    }


def _expanded_dependency_edge_ids(value: object) -> set[str]:
    if isinstance(value, list):
        return {
            edge_id
            for item in value
            for edge_id in _expanded_dependency_edge_ids(item)
        }
    if not isinstance(value, dict):
        return set()
    selected = set()
    if (
        value.get("schema")
        == "semantic_direct_dependency_expansion_v2"
        and value.get("status") == "expanded"
    ):
        selected.update(
            str(edge_id)
            for edge_id in value.get("dependency_edge_ids") or []
            if str(edge_id)
        )
    for item in value.values():
        selected.update(_expanded_dependency_edge_ids(item))
    return selected


def _model_relevant_record(
    value: object,
    *,
    consumed_dependency_edge_ids: set[str] | None = None,
) -> object:
    """Remove transport duplication while preserving exact repair evidence."""

    if isinstance(value, list):
        return [
            _model_relevant_record(
                item,
                consumed_dependency_edge_ids=(
                    consumed_dependency_edge_ids
                ),
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return deepcopy(value)
    if (
        value.get("schema")
        == "semantic_direct_dependency_expansion_v2"
        and value.get("status") == "listed"
        and consumed_dependency_edge_ids
    ):
        all_edges = [
            item
            for item in value.get("dependency_edges") or []
            if isinstance(item, dict)
        ]
        retained_edges = [
            item
            for item in all_edges
            if str(item.get("id") or "")
            in consumed_dependency_edge_ids
        ]
        retained_ids = [
            str(item.get("id") or "")
            for item in retained_edges
            if str(item.get("id") or "")
        ]
        return {
            "schema": value.get("schema"),
            "status": value.get("status"),
            "selected_owner_keys": deepcopy(
                value.get("selected_owner_keys") or []
            ),
            "direction": value.get("direction"),
            "dependency_edge_ids": retained_ids,
            "dependency_edges": deepcopy(retained_edges),
            "listed_dependency_edge_count": len(all_edges),
            "replayed_dependency_edge_count": len(retained_edges),
            "complete_record": "available_by_evidence_ref",
        }
    return {
        str(key): _model_relevant_record(
            item,
            consumed_dependency_edge_ids=consumed_dependency_edge_ids,
        )
        for key, item in value.items()
        if str(key)
        not in {
            "model_call",
            "response_text",
        }
    }


def _previous_unit_outcomes(
    state_root: Path,
    state: dict[str, Any],
    *,
    work_unit_id: str,
) -> list[dict[str, Any]]:
    """Recover durable prior actions so a later round does not start blind."""

    raw_outcomes: list[dict[str, Any]] = []
    for round_record in state.get("rounds") or []:
        if not isinstance(round_record, dict):
            continue
        unit = (round_record.get("units") or {}).get(work_unit_id)
        if not isinstance(unit, dict):
            continue
        actions = [
            *[
                {**item, "budget_kind": "semantic_step"}
                for item in unit.get("attempts") or []
                if isinstance(item, dict)
            ],
            *[
                {**item, "budget_kind": "evidence_action"}
                for item in unit.get("evidence_actions") or []
                if isinstance(item, dict)
            ],
        ]
        actions.sort(key=lambda item: str(item.get("at") or ""))
        for attempt in actions:
            if not isinstance(attempt, dict):
                continue
            evidence: list[dict[str, Any]] = []
            for reference in attempt.get("evidence_refs") or []:
                try:
                    record = load_semantic_record(
                        state_root,
                        str(reference),
                    )
                except (FileNotFoundError, ValueError):
                    evidence.append(
                        {
                            "ref": str(reference),
                            "status": "evidence_record_unavailable",
                        }
                    )
                    continue
                evidence.append(
                    {
                        "ref": str(reference),
                        "record": record,
                    }
                )
            raw_outcomes.append(
                {
                    "round": int(round_record.get("number") or 0),
                    "step": int(attempt.get("step") or 0),
                    "evidence_action": int(
                        attempt.get("evidence_action") or 0
                    ),
                    "budget_kind": str(
                        attempt.get("budget_kind") or ""
                    ),
                    "action_signature": str(
                        attempt.get("action_signature") or ""
                    ),
                    "outcome": str(attempt.get("outcome") or ""),
                    "duplicate": bool(attempt.get("duplicate")),
                    "evidence": evidence,
                }
            )
    consumed_dependency_edge_ids = {
        edge_id
        for outcome in raw_outcomes
        for evidence in outcome.get("evidence") or []
        for edge_id in _expanded_dependency_edge_ids(
            evidence.get("record")
        )
    }
    outcomes = [
        {
            **outcome,
            "evidence": [
                {
                    **evidence,
                    **(
                        {
                            "record": _model_relevant_record(
                                evidence.get("record"),
                                consumed_dependency_edge_ids=(
                                    consumed_dependency_edge_ids
                                ),
                            )
                        }
                        if "record" in evidence
                        else {}
                    ),
                }
                for evidence in outcome.get("evidence") or []
            ],
        }
        for outcome in raw_outcomes
    ]
    for batch in (state.get("batches") or {}).values():
        if not isinstance(batch, dict):
            continue
        rollback = batch.get("rollback")
        if (
            batch.get("status") != BATCH_ROLLED_BACK
            or not isinstance(rollback, dict)
        ):
            continue
        rollback_ref = str(rollback.get("ref") or "")
        for patch_record in batch.get("patches") or []:
            if not isinstance(patch_record, dict) or str(
                patch_record.get("work_unit_id") or ""
            ) != str(work_unit_id):
                continue
            patch_id = str(patch_record.get("id") or "")
            outcomes.append(
                {
                    "round": 0,
                    "step": 0,
                    "evidence_action": 0,
                    "budget_kind": "compile_validation",
                    "action_signature": stable_id(
                        "semantic-full-batch-rollback-outcome",
                        patch_id,
                        rollback_ref,
                    ),
                    "outcome": "batch_rolled_back_after_compile_failure",
                    "duplicate": False,
                    "evidence": [
                        {
                            "ref": rollback_ref,
                            "record": {
                                "schema": (
                                    "semantic_batch_compile_regression_"
                                    "evidence_v1"
                                ),
                                "status": "compile_invalid_batch_rolled_back",
                                "batch_id": str(batch.get("id") or ""),
                                "patch_id": patch_id,
                                "patch_signature": str(
                                    patch_record.get("patch_signature") or ""
                                ),
                                "gap_ids": list(
                                    patch_record.get("gap_ids") or []
                                ),
                                "touched_files": list(
                                    patch_record.get("touched_files") or []
                                ),
                                "rollback_ref": rollback_ref,
                                "reason": str(rollback.get("reason") or ""),
                                "compile_validation_refs": [
                                    str(validation.get("ref") or "")
                                    for validation in batch.get(
                                        "compile_validations"
                                    )
                                    or []
                                    if isinstance(validation, dict)
                                    and str(validation.get("ref") or "")
                                ],
                                "interpretation": (
                                    "the batch failed as a whole; this does "
                                    "not prove that this patch is invalid in "
                                    "an individual transaction"
                                ),
                                "required_next_action": (
                                    "use the failed compile checkpoint and "
                                    "interacting patch lineage before "
                                    "proposing a replacement transaction"
                                ),
                            },
                        }
                    ],
                }
            )
    return outcomes


def _semantic_evidence_state_signature(
    work_unit: dict[str, Any],
) -> str:
    """Fingerprint the executable frontier and its acceptance semantics."""

    bindings = []
    for raw in work_unit.get("owner_bindings") or []:
        if not isinstance(raw, dict):
            continue
        descriptors = []
        for key in ("current_descriptor", "donor_descriptor"):
            descriptor = raw.get(key)
            if not isinstance(descriptor, dict) or not descriptor:
                continue
            descriptors.append(
                {
                    "tree": descriptor.get("tree"),
                    "file": descriptor.get("file")
                    or descriptor.get("path"),
                    "kind": descriptor.get("kind")
                    or descriptor.get("entity_kind"),
                    "contract_key": descriptor.get("contract_key")
                    or descriptor.get("key"),
                    "symbol": descriptor.get("symbol")
                    or descriptor.get("name"),
                    "container": deepcopy(
                        descriptor.get("container") or {}
                    ),
                    "is_definition": descriptor.get("is_definition"),
                }
            )
        bindings.append(
            {
                "owner_key": str(raw.get("owner_key") or ""),
                "role": str(raw.get("role") or ""),
                "file": str(raw.get("file") or ""),
                "authority": str(raw.get("authority") or ""),
                "contract_key": str(raw.get("contract_key") or ""),
                "entity_kind": str(raw.get("entity_kind") or ""),
                "edit_permission": str(
                    raw.get("edit_permission") or ""
                ),
                "descriptors": descriptors,
            }
        )
    return stable_id(
        "semantic-unit-evidence-state",
        stable_json_sha256(
            {
                "work_unit_id": str(
                    work_unit.get("id")
                    or work_unit.get("work_unit_id")
                    or ""
                ),
                "gap_ids": sorted(
                    _ordered_strings(
                        list(work_unit.get("gap_ids") or [])
                    )
                ),
                "execution_policy": {
                    "semantic_action": (
                        SEMANTIC_ACTION_EXECUTION_POLICY_REVISION
                    ),
                    "provider_contract": (
                        PROVIDER_CONTRACT_POLICY_REVISION
                    ),
                    "diagnostic_provider_resolution": (
                        TYPED_DIAGNOSTIC_PROVIDER_RESOLUTION_REVISION
                    ),
                },
                "owner_bindings": sorted(
                    bindings,
                    key=lambda item: (
                        item["owner_key"],
                        item["file"],
                        item["authority"],
                    ),
                ),
            }
        ),
    )


def _semantic_action_signature(
    *,
    work_unit_id: str,
    result: dict[str, Any],
) -> str:
    """Identify the logical action, independent of prose and API metadata."""

    action = str(result.get("action") or result.get("status") or "")
    payload: dict[str, Any] = {
        "work_unit_id": str(work_unit_id),
        "kind": "semantic_repair",
        "action": action,
    }
    if action == "replace_source" or result.get("status") == "patch":
        payload.update(
            {
                "gap_ids": sorted(
                    _ordered_strings(list(result.get("gap_ids") or []))
                ),
                "diff_sha256": stable_json_sha256(
                    str(result.get("diff_text") or "")
                ),
            }
        )
    elif action == "inspect":
        payload["inspect_handle_ids"] = sorted(
            _ordered_strings(
                list(result.get("inspect_handle_ids") or [])
            )
        )
    elif action == "expand_dependencies":
        payload.update(
            {
                "owner_keys": sorted(
                    _ordered_strings(
                        list(result.get("owner_keys") or [])
                    )
                ),
                "direction": str(result.get("direction") or ""),
                "dependency_edge_ids": sorted(
                    _ordered_strings(
                        list(result.get("dependency_edge_ids") or [])
                    )
                ),
            }
        )
    elif action == "materialize_entity_from_donor":
        payload.update(
            {
                "materialization_handle_id": str(
                    result.get("materialization_handle_id") or ""
                ),
                "gap_ids": sorted(
                    _ordered_strings(list(result.get("gap_ids") or []))
                ),
            }
        )
    elif result.get("status") == "evidence_added":
        payload["evidence_digest"] = stable_json_sha256(
            _model_relevant_record(result.get("evidence") or {})
        )
    return stable_id(
        "semantic-repair-action",
        stable_json_sha256(payload),
    )


def _semantic_action_settlement(
    state: dict[str, Any],
    work_unit: dict[str, Any],
    attempt: dict[str, Any],
    *,
    evidence_changed: bool,
) -> str | None:
    remaining = any(
        ((state.get("gaps") or {}).get(str(gap_id)) or {}).get("status")
        == CHANGE_REQUIRED
        for gap_id in work_unit.get("gap_ids") or []
    )
    if not remaining:
        return "unit_complete"
    if evidence_changed:
        return "evidence_changed"
    if attempt.get("duplicate"):
        return "duplicate_action"
    return None


def _validate_review_decisions(
    state: dict[str, Any],
    *,
    context: dict[str, Any],
    decisions: list[dict[str, Any]],
    prompt_evidence_ids: Iterable[object],
    prompt_source_records: Iterable[object] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind no-change judgments to the frozen graph evidence or fail closed."""

    snapshot = state.get("semantic_gap_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("semantic review requires a frozen gap snapshot")
    context_by_gap = {
        str(item.get("id") or ""): item
        for item in context.get("gaps") or []
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    normalized: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    raw_allowed_evidence_ids = set(
        _ordered_strings(prompt_evidence_ids)
    )
    source_records = [
        dict(record)
        for record in prompt_source_records
        if isinstance(record, dict)
    ]
    for raw_decision in decisions:
        decision = dict(raw_decision)
        decision["evidence_ids"] = [
            reference
            for reference in _ordered_strings(
                decision.get("evidence_ids") or []
            )
            if not (
                reference.startswith("record-")
                and len(reference) == len("record-000000")
                and reference.removeprefix("record-").isdigit()
            )
        ]
        gap_id = str(decision.get("gap_id") or "")
        gap_context = context_by_gap.get(gap_id) or {}
        evaluation = gap_context.get("compile_valid_evaluation") or {}
        integration_relation = str(
            (evaluation.get("relations") or {}).get(
                "integration_reference"
            )
            or ""
        )
        if (
            decision.get("disposition") == "change_required"
            and integration_relation
            == "target_alternative_review_required"
        ):
            claimed_evidence_ids = set(
                _ordered_strings(decision.get("evidence_ids") or [])
            )
            contract_ref = str(
                gap_context.get("contract_evidence_ref") or ""
            )
            evaluation_ref = str(
                gap_context.get("compile_valid_evaluation_ref") or ""
            )
            mandatory_ids = {contract_ref, evaluation_ref} - {""}
            gap_source_records = [
                record
                for record in source_records
                if gap_id in _ordered_strings(record.get("gap_ids") or [])
            ]
            current_source_ids = {
                str(record.get("id") or "")
                for record in gap_source_records
                if str(record.get("tree") or "") == "old"
                and str(record.get("id") or "")
            }
            donor_source_ids = {
                str(record.get("id") or "")
                for record in gap_source_records
                if str(record.get("tree") or "").startswith("donor")
                and str(record.get("id") or "")
            }
            reasons: list[str] = []
            if not claimed_evidence_ids <= raw_allowed_evidence_ids:
                reasons.append("unknown_evidence_ids")
            if not mandatory_ids <= claimed_evidence_ids:
                reasons.append("missing_contract_or_evaluation_evidence")
            if not current_source_ids.intersection(claimed_evidence_ids):
                reasons.append("missing_exact_current_source_evidence")
            if not donor_source_ids.intersection(claimed_evidence_ids):
                reasons.append("missing_exact_donor_source_evidence")
            if reasons:
                missing_mandatory_ids = sorted(
                    mandatory_ids - claimed_evidence_ids
                )
                rejections.append(
                    {
                        "gap_id": gap_id,
                        "claimed_basis": "change_required",
                        "reasons": reasons,
                        "claimed_evidence_ids": sorted(
                            claimed_evidence_ids
                        ),
                        "required_current_source_ids": sorted(
                            current_source_ids
                        ),
                        "required_donor_source_ids": sorted(
                            donor_source_ids
                        ),
                        "required_evidence_ids": sorted(mandatory_ids),
                        "missing_evidence_ids": missing_mandatory_ids,
                        "rejection_policy": "retry_review",
                    }
                )
                normalized.append(decision)
                continue
            normalized.append(
                {
                    **decision,
                    "change_required_proof": {
                        "schema": "semantic_change_required_proof_v1",
                        "basis": "exact_current_donor_behavior_comparison",
                        "gap_id": gap_id,
                        "evaluation_ref": evaluation_ref,
                        "current_source_evidence_ids": sorted(
                            current_source_ids.intersection(
                                claimed_evidence_ids
                            )
                        ),
                        "donor_source_evidence_ids": sorted(
                            donor_source_ids.intersection(
                                claimed_evidence_ids
                            )
                        ),
                        "validated_at": utc_now(),
                    },
                }
            )
            continue
        if decision.get("disposition") != "no_change_required":
            normalized.append(decision)
            continue
        required_dimensions = set(
            _ordered_strings(evaluation.get("required_dimensions") or [])
        )
        open_dimensions = set(
            _ordered_strings(
                [
                    *(evaluation.get("unresolved_dimensions") or []),
                    *(evaluation.get("unavailable_dimensions") or []),
                ]
            )
        )
        covered_dimensions = set(
            _ordered_strings(decision.get("covered_dimensions") or [])
        )
        raw_claimed_evidence_ids = set(
            _ordered_strings(decision.get("evidence_ids") or [])
        )
        contract_ref = str(
            gap_context.get("contract_evidence_ref") or ""
        )
        evaluation_ref = str(
            gap_context.get("compile_valid_evaluation_ref") or ""
        )
        mandatory_ids = {contract_ref, evaluation_ref} - {""}
        evidence_aliases: dict[str, str] = {}
        named_mandatory_refs = {
            "contract_evidence_ref": contract_ref,
            "compile_valid_evaluation_ref": evaluation_ref,
        }
        for field_name, reference in named_mandatory_refs.items():
            if not reference:
                continue
            try:
                kind, digest = parse_content_record_ref(reference)
            except ValueError:
                continue
            if reference in raw_allowed_evidence_ids:
                # The prompt exposes both the field label and the immutable
                # content-record value. Models may echo either the raw value,
                # its compact kind:digest form, or the exact labeled token.
                # Bind aliases only to a record already present in the frozen
                # prompt evidence set; never accept a prefix by itself.
                evidence_aliases[f"{kind}:{digest}"] = reference
                evidence_aliases[
                    f"{field_name}:{reference}"
                ] = reference
        claimed_evidence_ids = {
            evidence_aliases.get(reference, reference)
            for reference in raw_claimed_evidence_ids
        }
        allowed_evidence_ids = {
            evidence_aliases.get(reference, reference)
            for reference in raw_allowed_evidence_ids
        }
        closure_basis = str(decision.get("closure_basis") or "")
        reasons: list[str] = []
        if not claimed_evidence_ids <= allowed_evidence_ids:
            reasons.append("unknown_evidence_ids")
        if not mandatory_ids <= claimed_evidence_ids:
            reasons.append("missing_contract_or_evaluation_evidence")
        if closure_basis == "exact_current_tree_satisfied":
            if evaluation.get("status") != "satisfied":
                reasons.append("current_graph_not_satisfied")
            if open_dimensions:
                reasons.append("evaluation_has_open_dimensions")
            if covered_dimensions != required_dimensions:
                reasons.append("incomplete_required_dimension_coverage")
            proof_basis = "exact_current_tree_satisfied"
        elif closure_basis == "target_contract_not_required":
            if not open_dimensions:
                reasons.append("no_open_dimension_to_waive")
            if not open_dimensions <= covered_dimensions:
                reasons.append("incomplete_open_dimension_coverage")
            if not covered_dimensions <= required_dimensions:
                reasons.append("unknown_dimension_coverage")
            proof_basis = "evidence_bound_target_judgment"
        else:
            reasons.append("unsupported_closure_basis")
            proof_basis = ""
        if reasons:
            missing_mandatory_ids = sorted(
                mandatory_ids - claimed_evidence_ids
            )
            unknown_evidence_ids = sorted(
                reference
                for reference in raw_claimed_evidence_ids
                if evidence_aliases.get(reference, reference)
                not in allowed_evidence_ids
            )
            rejections.append(
                {
                    "gap_id": gap_id,
                    "claimed_basis": closure_basis,
                    "reasons": reasons,
                    "claimed_evidence_ids": sorted(
                        raw_claimed_evidence_ids
                    ),
                    "unknown_evidence_ids": unknown_evidence_ids,
                    "required_evidence_ids": sorted(mandatory_ids),
                    "missing_evidence_ids": missing_mandatory_ids,
                    "allowed_evidence_count": len(
                        raw_allowed_evidence_ids
                    ),
                    "allowed_evidence_digest": stable_json_sha256(
                        sorted(raw_allowed_evidence_ids)
                    ),
                    "open_dimensions": sorted(open_dimensions),
                    "rejection_policy": "retry_review",
                }
            )
            normalized.append(decision)
            continue
        normalized.append(
            {
                **decision,
                "evidence_ids": sorted(claimed_evidence_ids),
                "no_change_proof": {
                    "schema": SEMANTIC_NO_CHANGE_PROOF_SCHEMA,
                    "policy_revision": (
                        SEMANTIC_NO_CHANGE_PROOF_POLICY_REVISION
                    ),
                    "snapshot_id": str(snapshot.get("id") or ""),
                    "gap_id": gap_id,
                    "basis": proof_basis,
                    "evaluation_ref": evaluation_ref,
                    "checkpoint_ref": str(
                        evaluation.get("checkpoint_ref") or ""
                    ),
                    "covered_dimensions": sorted(covered_dimensions),
                    "evidence_ids": sorted(claimed_evidence_ids),
                    "validation_status": "validated",
                    "validated_at": utc_now(),
                },
            }
        )
    return normalized, rejections


def _review_pending_gaps_once(
    runtime: SemanticClosureRuntime,
    *,
    state_root: Path,
    state: dict[str, Any],
    work_unit: dict[str, Any],
    round_number: int,
) -> dict[str, Any]:
    all_pending_gap_ids = [
        str(gap_id)
        for gap_id in work_unit.get("gap_ids") or []
        if (
            (state.get("gaps") or {}).get(str(gap_id)) or {}
        ).get("status")
        == PENDING_DECISION
    ]
    evidence_unavailable_gap_ids = [
        gap_id
        for gap_id in all_pending_gap_ids
        if (
            (
                (state.get("gaps") or {}).get(gap_id) or {}
            ).get("compile_valid_handoff_evaluation")
            or {}
        ).get("status")
        == "evidence_unavailable"
    ]
    pending_gap_ids = list(all_pending_gap_ids)
    if not pending_gap_ids:
        return {
            "status": "not_required",
            "decided_gap_ids": [],
            "evidence_unavailable_gap_ids": (
                evidence_unavailable_gap_ids
            ),
        }
    context = _work_unit_context(
        state_root,
        state,
        {
            **work_unit,
            "gap_ids": pending_gap_ids,
        },
    )
    try:
        result = runtime.review(
            {
                "schema": "semantic_review_request_v1",
                "round": int(round_number),
                "work_unit_id": str(work_unit["id"]),
                "expected_gap_ids": pending_gap_ids,
                "context": context,
            }
        )
    except ModelAPIError as exc:
        failure = _model_runtime_failure(
            exc,
            operation="semantic_review",
            round_number=round_number,
            work_unit_id=str(work_unit["id"]),
        )
        failure_ref = _store_runtime_result(
            state_root,
            kind="semantic-model-runtime-failure",
            result=failure,
        )
        return {
            **failure,
            "runtime_failure_ref": failure_ref,
            "decided_gap_ids": [],
            "pending_gap_ids": pending_gap_ids,
            "evidence_unavailable_gap_ids": (
                evidence_unavailable_gap_ids
            ),
        }
    except Exception as exc:
        result = {
            "status": "review_failed",
            "failure_kind": "controller_exception",
            "retryable": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "decisions": [],
        }
    if not isinstance(result, dict):
        result = {
            "status": "review_failed",
            "failure_kind": "invalid_runtime_result",
            "retryable": False,
            "error": "semantic review runtime returned no object",
            "decisions": [],
        }
    evidence_ref = _store_runtime_result(
        state_root,
        kind="semantic-review",
        result=result,
    )
    if result.get("status") == "review_failed":
        failure_error = str(
            result.get("error") or "semantic review failed"
        )
        transition = record_gap_review_failure(
            state,
            work_unit_id=str(work_unit["id"]),
            gap_ids=pending_gap_ids,
            evidence_ref=evidence_ref,
            error=failure_error,
            round_number=round_number,
        )
        _write_state(state_root, state)
        return {
            **transition,
            "status": "review_failed",
            "review_ref": evidence_ref,
            "failure_kind": str(result.get("failure_kind") or ""),
            "retryable": bool(result.get("retryable")),
            "error": failure_error,
            "decided_gap_ids": [],
            "evidence_unavailable_gap_ids": (
                evidence_unavailable_gap_ids
            ),
        }
    raw_decisions = [
        {
            **dict(decision),
            "evidence_refs": _ordered_strings(
                [
                    *list(decision.get("evidence_refs") or []),
                    evidence_ref,
                ]
            ),
        }
        for decision in result.get("decisions") or []
        if isinstance(decision, dict)
    ]
    decisions, proof_rejections = _validate_review_decisions(
        state,
        context=context,
        decisions=raw_decisions,
        prompt_evidence_ids=result.get("prompt_evidence_ids") or [],
        prompt_source_records=result.get("prompt_source_records") or [],
    )
    retry_review_rejections = [
        rejection
        for rejection in proof_rejections
        if rejection.get("rejection_policy") == "retry_review"
    ]
    if retry_review_rejections:
        rejection_feedback = json.dumps(
            [
                {
                    key: deepcopy(rejection.get(key))
                    for key in (
                        "gap_id",
                        "reasons",
                        "unknown_evidence_ids",
                        "missing_evidence_ids",
                    )
                    if rejection.get(key) not in (None, "", [], {})
                }
                for rejection in retry_review_rejections
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        failure_error = (
            "semantic review proof rejected; retry with exact prompt "
            f"evidence: {rejection_feedback}"
        )
        proof_validation_ref = put_semantic_record(
            state_root,
            kind="semantic-review-proof-validation",
            record={
                "schema": "semantic_review_proof_validation_v1",
                "work_unit_id": str(work_unit["id"]),
                "review_ref": evidence_ref,
                "snapshot_id": str(
                    (state.get("semantic_gap_snapshot") or {}).get("id")
                    or ""
                ),
                "rejections": retry_review_rejections,
                "validated_at": utc_now(),
            },
        )
        transition = record_gap_review_failure(
            state,
            work_unit_id=str(work_unit["id"]),
            gap_ids=pending_gap_ids,
            evidence_ref=proof_validation_ref,
            error=failure_error,
            round_number=round_number,
        )
        _write_state(state_root, state)
        return {
            **transition,
            "status": "review_failed",
            "review_ref": evidence_ref,
            "failure_kind": "proof_validation_rejected",
            "retryable": True,
            "error": failure_error,
            "proof_validation_ref": proof_validation_ref,
            "proof_rejections": proof_rejections,
            "decided_gap_ids": [],
            "pending_gap_ids": pending_gap_ids,
            "evidence_unavailable_gap_ids": (
                evidence_unavailable_gap_ids
            ),
        }
    proof_validation_ref = put_semantic_record(
        state_root,
        kind="semantic-review-proof-validation",
        record={
            "schema": "semantic_review_proof_validation_v1",
            "work_unit_id": str(work_unit["id"]),
            "review_ref": evidence_ref,
            "snapshot_id": str(
                (state.get("semantic_gap_snapshot") or {}).get("id") or ""
            ),
            "validated_no_change_gap_ids": sorted(
                str(decision.get("gap_id") or "")
                for decision in decisions
                if isinstance(decision.get("no_change_proof"), dict)
            ),
            "change_required_gap_ids": sorted(
                str(decision.get("gap_id") or "")
                for decision in decisions
                if decision.get("disposition") == "change_required"
            ),
            "rejections": proof_rejections,
            "validated_at": utc_now(),
        },
    )
    for decision in decisions:
        decision["evidence_refs"] = _ordered_strings(
            [
                *(decision.get("evidence_refs") or []),
                proof_validation_ref,
            ]
        )
        proof = decision.get("no_change_proof")
        if isinstance(proof, dict):
            proof["evidence_ids"] = _ordered_strings(
                [*(proof.get("evidence_ids") or []), proof_validation_ref]
            )
    transition = record_gap_decisions(
        state,
        work_unit_id=str(work_unit["id"]),
        expected_gap_ids=pending_gap_ids,
        decisions=decisions,
        round_number=round_number,
    )
    _write_state(state_root, state)
    return {
        "status": "decided",
        "review_ref": evidence_ref,
        "decided_gap_ids": pending_gap_ids,
        "evidence_unavailable_gap_ids": evidence_unavailable_gap_ids,
        "proof_validation_ref": proof_validation_ref,
        "proof_rejections": proof_rejections,
        **transition,
    }


def _review_pending_gaps(
    runtime: SemanticClosureRuntime,
    *,
    state_root: Path,
    state: dict[str, Any],
    work_unit: dict[str, Any],
    round_number: int,
) -> dict[str, Any]:
    """Retry only recoverable model review failures for this pending unit."""

    result: dict[str, Any] = {
        "status": "review_failed",
        "retryable": False,
        "error": "semantic review was not attempted",
    }
    attempt_count = 0
    for attempt_number in range(1, SEMANTIC_REVIEW_ATTEMPT_LIMIT + 1):
        attempt_count = attempt_number
        result = _review_pending_gaps_once(
            runtime,
            state_root=state_root,
            state=state,
            work_unit=work_unit,
            round_number=round_number,
        )
        if not (
            result.get("status") == "review_failed"
            and result.get("retryable") is True
        ):
            break
    retry_exhausted = bool(
        result.get("status") == "review_failed"
        and result.get("retryable") is True
        and attempt_count >= SEMANTIC_REVIEW_ATTEMPT_LIMIT
    )
    return {
        **result,
        "review_attempt_count": attempt_count,
        "review_attempt_limit": SEMANTIC_REVIEW_ATTEMPT_LIMIT,
        "retry_exhausted": retry_exhausted,
    }


def _ensure_patch_batch(
    spec: object,
    *,
    state_root: Path,
    state: dict[str, Any],
    round_number: int,
) -> tuple[dict[str, Any], Any, bool]:
    batch = _active_batch(state)
    if batch is not None:
        transaction = load_transaction(
            Path(str(batch.get("preimage_ref") or ""))
        )
        return batch, transaction, False
    transaction = start_transaction(
        state_root,
        case_id=str(getattr(spec, "case_id")),
        cycle=int(round_number),
        work_unit_id=stable_id(
            "semantic-closure-batch",
            str(getattr(spec, "case_id")),
            str(round_number),
            str(state.get("generation") or 0),
        ),
    )
    batch = begin_or_resume_patch_batch(
        state,
        round_number=round_number,
        preimage_ref=str(transaction.root.resolve()),
    )
    _write_state(state_root, state)
    return batch, transaction, True


def _discard_empty_new_batch(
    spec: object,
    *,
    state_root: Path,
    state: dict[str, Any],
    transaction: Any,
) -> None:
    rollback = rollback_transaction(spec, transaction.root)
    rollback_ref = _store_runtime_result(
        state_root,
        kind="semantic-batch-rollback",
        result=rollback,
    )
    rollback_active_batch(
        state,
        rollback_ref=rollback_ref,
        reason="first patch in a new semantic batch was not applied",
    )
    _write_state(state_root, state)


def _current_source_region_sha256(
    spec: object,
    *,
    relative_path: str,
    start_line: int,
    end_line: int,
) -> str:
    path_parts = Path(relative_path).parts
    if not relative_path or ".." in path_parts:
        return ""
    resolver = getattr(spec, "resolve_old", None)
    if callable(resolver):
        path = Path(resolver(relative_path))
    else:
        root = Path(str(getattr(spec, "old_root_path", "")))
        path = (root / relative_path).resolve()
    if not path.is_file():
        return ""
    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines(keepends=True)
    if (
        start_line <= 0
        or end_line < start_line
        or end_line > len(lines)
    ):
        return ""
    source = "".join(lines[start_line - 1 : end_line])
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _verify_structured_source_edit_authorizations(
    spec: object,
    *,
    work_unit: dict[str, Any],
    candidate: dict[str, Any],
    claimed_gap_ids: list[str],
) -> dict[str, Any]:
    """Verify provenance before exact edit regions reach the boundary gate."""

    structured = candidate.get("structured_source_patch")
    if not isinstance(structured, dict):
        return {
            "schema": "semantic_source_edit_authorization_v1",
            "status": "not_applicable",
            "authorizations": [],
        }

    def denied(reason: str, **details: object) -> dict[str, Any]:
        return {
            "schema": "semantic_source_edit_authorization_v1",
            "status": "denied",
            "reason": reason,
            "authorizations": [],
            **details,
        }

    if (
        structured.get("schema")
        != "semantic_structured_source_patch_v1"
        or structured.get("status") != "patch"
    ):
        return denied("invalid_structured_source_patch_envelope")
    if str(structured.get("diff_text") or "") != str(
        candidate.get("diff_text") or ""
    ):
        return denied("structured_patch_diff_mismatch")

    raw_regions = structured.get("edit_regions")
    if not isinstance(raw_regions, list) or not raw_regions:
        return denied("structured_patch_has_no_edit_regions")
    claimed = set(_ordered_strings(claimed_gap_ids))
    work_unit_gap_ids = set(
        _ordered_strings(list(work_unit.get("gap_ids") or []))
    )
    work_unit_owner_keys = {
        owner_key
        for binding in work_unit.get("owner_bindings") or []
        if isinstance(binding, dict)
        for owner_key in (
            str(binding.get("owner_key") or ""),
            str(semantic_mutable_owner_key(binding) or ""),
        )
        if owner_key
    }
    work_unit_source_owner_keys = {
        str(owner_key)
        for binding in work_unit.get("owner_bindings") or []
        if isinstance(binding, dict)
        for owner_key in binding.get("source_owner_keys") or []
        if str(owner_key)
    }
    verified_diff_sha256 = hashlib.sha256(
        str(candidate.get("diff_text") or "").encode("utf-8")
    ).hexdigest()
    authorizations: list[dict[str, Any]] = []
    covered_gap_ids: set[str] = set()
    seen_handles: set[str] = set()
    for raw in raw_regions:
        if not isinstance(raw, dict) or raw.get(
            "authority"
        ) != "exact_structured_source_edit":
            return denied("invalid_structured_edit_region")
        relative_path = normalize_relpath(
            str(raw.get("file") or "")
        )
        start_line = int(raw.get("start_line") or 0)
        end_line = int(raw.get("end_line") or start_line)
        source_sha256 = str(raw.get("source_sha256") or "")
        handle_id = str(raw.get("edit_handle_id") or "")
        region_gap_ids = set(
            _ordered_strings(list(raw.get("direct_gap_ids") or []))
        )
        region_owner_keys = set(
            _ordered_strings(list(raw.get("owner_keys") or []))
        )
        region_source_owner_keys = set(
            _ordered_strings(
                list(raw.get("source_owner_keys") or [])
            )
        )
        if (
            not relative_path
            or start_line <= 0
            or end_line < start_line
            or not source_sha256
            or not handle_id
            or handle_id in seen_handles
        ):
            return denied("invalid_structured_edit_region_identity")
        seen_handles.add(handle_id)
        current_sha256 = _current_source_region_sha256(
            spec,
            relative_path=relative_path,
            start_line=start_line,
            end_line=end_line,
        )
        if current_sha256 != source_sha256:
            return denied(
                "structured_edit_source_is_stale",
                edit_handle_id=handle_id,
                file=relative_path,
                start_line=start_line,
                end_line=end_line,
            )
        if (
            not region_gap_ids
            or not region_gap_ids <= work_unit_gap_ids
        ):
            return denied(
                "structured_edit_gap_lineage_mismatch",
                edit_handle_id=handle_id,
                region_gap_ids=sorted(region_gap_ids),
                work_unit_gap_ids=sorted(work_unit_gap_ids),
                claimed_gap_ids=sorted(claimed),
            )
        if (
            region_owner_keys
            and not region_owner_keys <= work_unit_owner_keys
        ):
            return denied(
                "structured_edit_owner_lineage_mismatch",
                edit_handle_id=handle_id,
                region_owner_keys=sorted(region_owner_keys),
                work_unit_owner_keys=sorted(work_unit_owner_keys),
            )
        if (
            region_source_owner_keys
            and not region_source_owner_keys
            <= work_unit_source_owner_keys
        ):
            return denied(
                "structured_edit_source_owner_lineage_mismatch",
                edit_handle_id=handle_id,
                region_source_owner_keys=sorted(
                    region_source_owner_keys
                ),
                work_unit_source_owner_keys=sorted(
                    work_unit_source_owner_keys
                ),
            )
        covered_gap_ids.update(region_gap_ids)
        authorizations.append(
            {
                **deepcopy(raw),
                "verified_diff_sha256": verified_diff_sha256,
            }
        )

    declared_gap_ids = set(
        _ordered_strings(
            list(structured.get("direct_gap_ids") or [])
        )
    )
    if declared_gap_ids != covered_gap_ids:
        return denied(
            "structured_patch_gap_summary_mismatch",
            declared_gap_ids=sorted(declared_gap_ids),
            covered_gap_ids=sorted(covered_gap_ids),
        )
    if not claimed <= covered_gap_ids:
        return denied(
            "structured_patch_does_not_cover_claimed_gaps",
            claimed_gap_ids=sorted(claimed),
            covered_gap_ids=sorted(covered_gap_ids),
        )
    return {
        "schema": "semantic_source_edit_authorization_v1",
        "status": "authorized",
        "authorizations": authorizations,
        "edit_handle_ids": sorted(seen_handles),
        "covered_gap_ids": sorted(covered_gap_ids),
        "verified_diff_sha256": verified_diff_sha256,
    }


def _annotate_patch_gap_lineage(
    boundary: dict[str, Any],
    *,
    candidate: dict[str, Any],
    claimed_gap_ids: list[str],
) -> dict[str, Any]:
    """Map each changed hunk to the exact structured edit that produced it."""

    structured = (
        candidate.get("structured_source_patch")
        if isinstance(candidate.get("structured_source_patch"), dict)
        else {}
    )
    edit_regions = [
        item
        for item in structured.get("edit_regions") or []
        if isinstance(item, dict)
    ]
    annotated = deepcopy(boundary)
    for changed in annotated.get("changed_ranges") or []:
        if not isinstance(changed, dict):
            continue
        if not edit_regions:
            changed["direct_gap_ids"] = list(claimed_gap_ids)
            continue
        relative_path = str(changed.get("file") or "")
        old_start = int(changed.get("old_start") or 0)
        old_count = int(changed.get("old_count") or 0)
        old_end = old_start + max(0, old_count - 1)
        direct_gap_ids: list[str] = []
        direct_owner_keys: list[str] = []
        for region in edit_regions:
            if str(region.get("file") or "") != relative_path:
                continue
            region_start = int(region.get("start_line") or 0)
            region_end = int(region.get("end_line") or region_start)
            if old_count == 0:
                overlaps = region_start <= old_start <= region_end + 1
            else:
                overlaps = max(old_start, region_start) <= min(
                    old_end,
                    region_end,
                )
            if overlaps:
                direct_gap_ids.extend(
                    str(gap_id)
                    for gap_id in region.get("direct_gap_ids") or []
                )
                direct_owner_keys.extend(
                    str(owner_key)
                    for owner_key in region.get("owner_keys") or []
                )
                direct_owner_keys.extend(
                    str(owner_key)
                    for owner_key in region.get("source_owner_keys") or []
                )
        changed["direct_gap_ids"] = _ordered_strings(direct_gap_ids)
        changed["direct_owner_keys"] = _ordered_strings(
            direct_owner_keys
        )
    return annotated


def _provider_obligations_from_work_unit(
    work_unit: dict[str, Any],
) -> list[dict[str, Any]]:
    """Persist exact donor dependencies whose contract is absent from old."""

    scope = (
        work_unit.get("code_graph_scope")
        if isinstance(work_unit.get("code_graph_scope"), dict)
        else {}
    )
    obligations: dict[str, dict[str, Any]] = {}
    for dependency in [
        *list(scope.get("repair_dependencies") or []),
        *list(work_unit.get("repair_dependencies") or []),
    ]:
        if not isinstance(dependency, dict):
            continue
        target = (
            dict(dependency.get("target_entity") or {})
            if isinstance(dependency.get("target_entity"), dict)
            else {}
        )
        target_entity_id = str(
            dependency.get("target_entity_id")
            or target.get("id")
            or ""
        )
        target_contract_key = str(
            target.get("contract_key") or target.get("key") or ""
        )
        target_symbol = str(target.get("symbol") or "")
        owner_entity_id = str(
            dependency.get("owner_entity_id") or ""
        )
        reference_id = str(
            dependency.get("reference_node_id") or ""
        )
        if (
            not target_entity_id
            or not target_contract_key
            or not owner_entity_id
            or dependency.get("old_corresponding_entity_ids")
        ):
            continue
        obligation_id = stable_id(
            "provider-materialization-obligation",
            owner_entity_id,
            target_entity_id,
            reference_id,
        )
        obligations[obligation_id] = {
            "schema": "provider_materialization_obligation_v1",
            "id": obligation_id,
            "status": "pending_old_contract",
            "owner_entity_id": owner_entity_id,
            "owner_file": normalize_relpath(
                str(dependency.get("owner_file") or "")
            ),
            "reference_id": reference_id or None,
            "use_kind": dependency.get("use_kind"),
            "target_entity_id": target_entity_id,
            "target_contract_key": target_contract_key,
            "target_file": normalize_relpath(
                str(
                    dependency.get("target_file")
                    or target.get("file")
                    or ""
                )
            ),
            "target_entity": _compact_entity(target),
            "source": "semantic_reference",
            "symbol": target_symbol or None,
            "contract_key": target_contract_key,
            "provider_file": normalize_relpath(
                str(
                    dependency.get("target_file")
                    or target.get("file")
                    or ""
                )
            ),
            "authority": "typed_donor_implementation_dependency",
        }
    return [obligations[key] for key in sorted(obligations)]


def _provider_contract_diagnostic_issues(
    report: dict[str, Any],
    *,
    provider_contract_ref: str,
) -> list[dict[str, Any]]:
    """Project apply-checker findings into exact CodeIndex diagnostics."""

    issues: list[dict[str, Any]] = []
    for finding_index, raw in enumerate(
        report.get("blocking_findings") or []
    ):
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind") or "")
        trigger_files: list[str] = []
        patched_locations: list[dict[str, Any]] = []
        if kind == "patch_locus_diagnostic":
            identifier_evidence = diagnostic_identifier_evidence(
                raw
            )
            identifiers = [
                item["identifier"] for item in identifier_evidence
            ]
            relative_path = normalize_relpath(
                str(raw.get("file") or "")
            )
            patched_locations = [
                dict(item)
                for item in raw.get("locations") or []
                if isinstance(item, dict)
            ]
            locations = [
                dict(item)
                for item in raw.get("current_locations") or []
                if isinstance(item, dict)
            ]
            trigger_files = _ordered_strings(
                raw.get("translation_unit_files") or []
            )
            message = str(raw.get("spelling") or kind)
        elif kind == "orphaned_patch_reference":
            identifiers = _ordered_strings([raw.get("symbol")])
            identifier_evidence = [
                {
                    "identifier": identifier,
                    "source": "orphaned_reference_symbol",
                }
                for identifier in identifiers
            ]
            relative_path = normalize_relpath(
                str(raw.get("source_file") or "")
            )
            locations = [
                dict(item)
                for item in raw.get("locations") or []
                if isinstance(item, dict)
            ]
            message = str(
                raw.get("spelling")
                or "patched source still references a removed provider"
            )
        else:
            continue
        evidence_by_identifier = {
            str(item.get("identifier") or ""): dict(item)
            for item in identifier_evidence
            if isinstance(item, dict)
        }
        for identifier in identifiers:
            for location_index, location in enumerate(locations or [{}]):
                issues.append(
                    {
                        "diagnostic_event_id": stable_id(
                            "provider-contract-diagnostic",
                            provider_contract_ref,
                            str(finding_index),
                            identifier,
                            str(location_index),
                            str(location.get("line") or 0),
                            str(location.get("column") or 0),
                        ),
                        "diagnostic_route": "provider_contract",
                        "file": relative_path,
                        "line": int(location.get("line") or 0),
                        "column": int(location.get("column") or 0),
                        "location_provenance": str(
                            location.get("projection") or ""
                        ),
                        "patched_locations": patched_locations,
                        "trigger_files": trigger_files,
                        "symbol": identifier,
                        "diagnostic_subject_evidence": (
                            evidence_by_identifier.get(identifier) or {}
                        ),
                        "message": message,
                        "raw": message,
                    }
                )
    return issues


def _expand_provider_contract_evidence(
    runtime: SemanticClosureRuntime,
    spec: object,
    *,
    state_root: Path,
    state: dict[str, Any],
    work_unit: dict[str, Any],
    gap_ids: list[str],
    provider_contract_report: dict[str, Any],
    provider_contract_ref: str,
    round_number: int,
    superseded_owner_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Bind unique provider roots back into the same semantic work unit."""

    issues = _provider_contract_diagnostic_issues(
        provider_contract_report,
        provider_contract_ref=provider_contract_ref,
    )
    if not issues:
        return {
            "schema": "typed_diagnostic_provider_expansion_v1",
            "status": "no_provider_diagnostics",
            "resolver_revision": (
                TYPED_DIAGNOSTIC_PROVIDER_RESOLUTION_REVISION
            ),
            "owner_bindings": [],
            "provider_contract_ref": provider_contract_ref,
        }
    index = getattr(runtime, "_semantic_provider_code_index", None)
    if not isinstance(index, CodeIndex):
        index = CodeIndex(spec, graph_projection="records")
        try:
            setattr(runtime, "_semantic_provider_code_index", index)
        except (AttributeError, TypeError):
            pass
    index.add_semantic_gap_ledger(state)
    expansion = index.exact_provider_owner_bindings_for_diagnostics(
        issues,
        semantic_gap_ids=gap_ids,
    )
    expansion = {
        **expansion,
        "work_unit_id": str(
            work_unit.get("id")
            or work_unit.get("work_unit_id")
            or ""
        ),
        "provider_contract_ref": provider_contract_ref,
    }
    expansion["owner_bindings"] = [
        {
            **dict(item),
            "resolver_revision": (
                TYPED_DIAGNOSTIC_PROVIDER_RESOLUTION_REVISION
            ),
            "provider_contract_ref": provider_contract_ref,
        }
        for item in expansion.get("owner_bindings") or []
        if isinstance(item, dict)
    ]
    expansion_ref = _store_runtime_result(
        state_root,
        kind="semantic-provider-diagnostic-expansion",
        result=expansion,
    )
    bindings = [
        dict(item)
        for item in expansion.get("owner_bindings") or []
        if isinstance(item, dict)
    ]
    new_bindings = [
        binding
        for binding in bindings
        if any(
            str(binding.get("owner_key") or "")
            not in {
                str(existing.get("owner_key") or "")
                for existing in (
                    (state.get("gaps") or {}).get(gap_id) or {}
                ).get("owner_bindings")
                or []
                if isinstance(existing, dict)
            }
            for gap_id in gap_ids
        )
    ]
    attachment: dict[str, Any] | None = None
    if new_bindings or superseded_owner_keys:
        attachment = record_exact_owner_expansion(
            state,
            work_unit_id=str(
                work_unit.get("id")
                or work_unit.get("work_unit_id")
                or ""
            ),
            gap_ids=gap_ids,
            owner_bindings=new_bindings,
            evidence_ref=expansion_ref,
            round_number=round_number,
            superseded_owner_keys=(superseded_owner_keys or []),
        )
    return {
        **expansion,
        "expansion_ref": expansion_ref,
        "new_owner_keys": _ordered_strings(
            binding.get("owner_key") for binding in new_bindings
        ),
        "attachment": attachment,
    }


def _replay_provider_contract_expansions(
    runtime: SemanticClosureRuntime,
    spec: object,
    *,
    state_root: Path,
    state: dict[str, Any],
    work_unit: dict[str, Any],
    round_number: int,
) -> list[dict[str, Any]]:
    """Recover durable provider evidence before issuing another model call."""

    unit_id = str(
        work_unit.get("id") or work_unit.get("work_unit_id") or ""
    )
    evidence_refs: list[str] = []
    for round_record in state.get("rounds") or []:
        if not isinstance(round_record, dict):
            continue
        unit = (round_record.get("units") or {}).get(unit_id)
        if not isinstance(unit, dict):
            continue
        for action in [
            *list(unit.get("attempts") or []),
            *list(unit.get("evidence_actions") or []),
        ]:
            if not isinstance(action, dict):
                continue
            evidence_refs.extend(
                str(reference)
                for reference in action.get("evidence_refs") or []
                if str(reference)
            )
    records: dict[str, dict[str, Any]] = {}
    completed_provider_refs: set[str] = set()
    current_resolver_owner_keys: set[str] = set()
    stale_owner_keys_by_provider_ref: dict[str, set[str]] = {}
    for reference in _ordered_strings(evidence_refs):
        try:
            record = load_semantic_record(state_root, reference)
        except (FileNotFoundError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        records[reference] = record
        provider_ref = str(record.get("provider_contract_ref") or "")
        if (
            provider_ref
            and record.get("schema")
            == "typed_diagnostic_provider_expansion_v1"
            and record.get("status") == "expanded"
            and record.get("owner_bindings")
        ):
            if (
                record.get("resolver_revision")
                == TYPED_DIAGNOSTIC_PROVIDER_RESOLUTION_REVISION
            ):
                completed_provider_refs.add(provider_ref)
                current_resolver_owner_keys.update(
                    str(item.get("owner_key") or "")
                    for item in record.get("owner_bindings") or []
                    if isinstance(item, dict)
                    and str(item.get("owner_key") or "")
                )
            else:
                stale_owner_keys_by_provider_ref.setdefault(
                    provider_ref,
                    set(),
                ).update(
                    str(item.get("owner_key") or "")
                    for item in record.get("owner_bindings") or []
                    if isinstance(item, dict)
                    and str(item.get("owner_key") or "")
                )

    results: list[dict[str, Any]] = []
    for reference, record in records.items():
        if (
            reference in completed_provider_refs
            or not record.get("blocking_findings")
        ):
            continue
        results.append(
            _expand_provider_contract_evidence(
                runtime,
                spec,
                state_root=state_root,
                state=state,
                work_unit=work_unit,
                gap_ids=_ordered_strings(
                    list(work_unit.get("gap_ids") or [])
                ),
                provider_contract_report=record,
                provider_contract_ref=reference,
                round_number=round_number,
                superseded_owner_keys=sorted(
                    stale_owner_keys_by_provider_ref.get(
                        reference,
                        set(),
                    )
                    - current_resolver_owner_keys
                ),
            )
        )
    return results


def _apply_semantic_patch_with_common_protocol(
    spec: object,
    *,
    state_root: Path,
    work_unit: dict[str, Any],
    diff_text: str,
    provider_obligations: list[dict[str, Any]],
    patch_signature: str,
    round_number: int,
) -> dict[str, Any]:
    """Apply one authorized semantic diff through the canonical B/C path."""

    source_work_unit_id = str(work_unit["id"])
    apply_unit_id = stable_id(
        "semantic-common-apply",
        source_work_unit_id,
        patch_signature,
        str(round_number),
    )
    apply_dir = (
        state_root
        / "semantic_closure"
        / "common_apply"
        / f"cycle-{round_number:04d}"
        / apply_unit_id
    )
    result = apply_single_generated_diff(
        spec,
        state_root=state_root,
        artifact_dir=apply_dir,
        plan_unit={
            "id": apply_unit_id,
            "work_unit_id": source_work_unit_id,
            "unit_kind": "semantic_gap_repair",
            "execution_phase": EXECUTION_PHASE_SEMANTIC_CLOSURE,
            "allowed_fileset": list(
                work_unit.get("allowed_fileset") or []
            ),
            "post_materialization_provider_obligations": deepcopy(
                provider_obligations
            ),
        },
        diff_text=diff_text,
        allow_escalation=False,
    )
    canonical_status = str(result.get("status") or "")
    return {
        **result,
        "status": (
            "applied"
            if canonical_status == "applied"
            else "failed"
        ),
        "reason": (
            None
            if canonical_status == "applied"
            else canonical_status or "canonical_apply_failed"
        ),
        "canonical_apply_status": canonical_status,
    }


def _apply_patch_candidate(
    runtime: SemanticClosureRuntime,
    spec: object,
    *,
    state_root: Path,
    state: dict[str, Any],
    work_unit: dict[str, Any],
    candidate: dict[str, Any],
    round_number: int,
) -> dict[str, Any]:
    diff_text = str(candidate.get("diff_text") or "")
    gap_ids = _ordered_strings(
        list(candidate.get("gap_ids") or [])
    )
    allowed_gap_ids = {
        str(gap_id)
        for gap_id in work_unit.get("gap_ids") or []
        if (
            (state.get("gaps") or {}).get(str(gap_id))
            or {}
        ).get("status")
        == CHANGE_REQUIRED
    }
    if not gap_ids or not set(gap_ids) <= allowed_gap_ids:
        raise ValueError(
            "semantic patch must claim only change-required gaps "
            "from its exact work unit"
        )

    source_edit_verification = (
        _verify_structured_source_edit_authorizations(
            spec,
            work_unit=work_unit,
            candidate=candidate,
            claimed_gap_ids=gap_ids,
        )
    )
    if source_edit_verification.get("status") == "denied":
        verification_ref = _store_runtime_result(
            state_root,
            kind="semantic-source-edit-authorization",
            result=source_edit_verification,
        )
        return {
            "status": "source_edit_authorization_denied",
            "reason": str(
                source_edit_verification.get("reason") or ""
            ),
            "source_edit_verification": (
                _model_relevant_record(source_edit_verification)
            ),
            "source_edit_verification_ref": verification_ref,
            "evidence_refs": [verification_ref],
        }

    boundary = validate_semantic_patch_boundary(
        spec,
        work_unit=work_unit,
        diff_text=diff_text,
        insertion_authorizations=(
            candidate.get("insertion_authorizations") or []
        ),
        source_edit_authorizations=(
            source_edit_verification.get("authorizations") or []
        ),
    )
    boundary["source_edit_verification"] = deepcopy(
        source_edit_verification
    )
    boundary = _annotate_patch_gap_lineage(
        boundary,
        candidate=candidate,
        claimed_gap_ids=gap_ids,
    )
    boundary_ref = _store_runtime_result(
        state_root,
        kind="semantic-patch-boundary",
        result=boundary,
    )
    evidence_refs = [boundary_ref]
    if boundary.get("status") != "authorized":
        return {
            "status": "boundary_denied",
            "reason": (
                "the complete structured source action is outside its exact "
                "authorized graph region; no partial projection was applied"
            ),
            "boundary_ref": boundary_ref,
            "boundary": _model_relevant_record(boundary),
            "evidence_refs": evidence_refs,
        }
    realized_gap_ids = {
        str(gap_id)
        for changed in boundary.get("changed_ranges") or []
        if isinstance(changed, dict)
        for gap_id in changed.get("direct_gap_ids") or []
        if str(gap_id)
    }
    if not set(gap_ids) <= realized_gap_ids:
        return {
            "status": "gap_lineage_unproven",
            "reason": (
                "the applied structured edit has no changed hunk for every "
                "claimed semantic gap"
            ),
            "claimed_gap_ids": gap_ids,
            "realized_gap_ids": sorted(realized_gap_ids),
            "boundary_ref": boundary_ref,
            "evidence_refs": evidence_refs,
        }

    patch_signature = str(
        candidate.get("patch_signature")
        or stable_id(
            "semantic-patch",
            "semantic_repair",
            stable_json_sha256(diff_text),
            *gap_ids,
        )
    )
    existing_batch = _active_batch(state)
    duplicate = next(
        (
            patch
            for patch in (existing_batch or {}).get("patches") or []
            if str(patch.get("patch_signature") or "")
            == patch_signature
        ),
        None,
    )
    if duplicate is not None:
        return {
            "status": "duplicate_patch",
            "patch_signature": patch_signature,
            "existing_patch_id": duplicate.get("id"),
            "reason": (
                "the active source batch already contains this exact "
                "patch; the unit must use its prior outcome evidence to "
                "produce a different repair"
            ),
            "evidence_refs": evidence_refs,
        }
    touched_files = _ordered_strings(
        [
            str(item.get("file") or "")
            for item in boundary.get("changed_ranges") or []
            if isinstance(item, dict)
        ]
    )
    provider_obligations = _provider_obligations_from_work_unit(
        work_unit
    )
    conditional_integrity_before = _conditional_integrity_reports(
        spec,
        touched_files,
    )
    structural_baseline = _capture_structural_baseline(
        spec,
        touched_files,
        changed_ranges=boundary.get("changed_ranges") or [],
    )
    structural_contract_validation_enabled = any(
        bool(report.get("parse_ok"))
        for report in structural_baseline.values()
        if isinstance(report, dict)
    )
    batch, outer_transaction, new_batch = _ensure_patch_batch(
        spec,
        state_root=state_root,
        state=state,
        round_number=round_number,
    )
    for relative_path in touched_files:
        backup_file_once(
            outer_transaction,
            spec,
            relative_path,
        )

    applied_transaction_ref: Path | None = None

    def reject_after_rollback(result: dict[str, Any]) -> dict[str, Any]:
        if applied_transaction_ref is not None:
            result["rollback"] = rollback_transaction(
                spec,
                applied_transaction_ref,
            )
        if new_batch and not batch.get("patches"):
            _discard_empty_new_batch(
                spec,
                state_root=state_root,
                state=state,
                transaction=outer_transaction,
            )
        return result

    try:
        apply_result = _apply_semantic_patch_with_common_protocol(
            spec,
            state_root=state_root,
            work_unit=work_unit,
            diff_text=diff_text,
            provider_obligations=provider_obligations,
            patch_signature=patch_signature,
            round_number=round_number,
        )
    except Exception as exc:
        apply_result = {
            "status": "failed",
            "reason": "canonical_apply_exception",
            "error": f"{exc.__class__.__name__}: {exc}",
            "touched_files": touched_files,
        }
    if not isinstance(apply_result, dict):
        apply_result = {
            "status": "failed",
            "reason": "patch runtime returned no object",
        }
    apply_ref = _store_runtime_result(
        state_root,
        kind="semantic-patch-apply",
        result=apply_result,
    )
    if apply_result.get("status") != "applied":
        common_result = apply_result
        apply_failure = (
            common_result.get("apply_failure")
            if isinstance(common_result.get("apply_failure"), dict)
            else {}
        )
        provider_contract_report = (
            common_result.get("provider_contract_report")
            if isinstance(
                common_result.get("provider_contract_report"),
                dict,
            )
            else None
        )
        if (
            apply_failure.get("stage") == "provider_contract"
            and provider_contract_report is not None
        ):
            provider_contract_ref = _store_runtime_result(
                state_root,
                kind="semantic-patch-provider-contract",
                result=provider_contract_report,
            )
            evidence_refs.append(provider_contract_ref)
            provider_expansion = _expand_provider_contract_evidence(
                runtime,
                spec,
                state_root=state_root,
                state=state,
                work_unit=work_unit,
                gap_ids=gap_ids,
                provider_contract_report=provider_contract_report,
                provider_contract_ref=provider_contract_ref,
                round_number=round_number,
            )
            expansion_ref = str(
                provider_expansion.get("expansion_ref") or ""
            )
            if expansion_ref:
                evidence_refs.append(expansion_ref)
            return reject_after_rollback({
                "status": "provider_contract_denied",
                "reason": str(
                    common_result.get("error")
                    or provider_contract_report.get("reason")
                    or "provider contract validation rejected the patch"
                ),
                "provider_contract_ref": provider_contract_ref,
                "provider_contract": _model_relevant_record(
                    provider_contract_report
                ),
                "provider_expansion": _model_relevant_record(
                    provider_expansion
                ),
                "apply_ref": apply_ref,
                "boundary_ref": boundary_ref,
                "evidence_refs": [*evidence_refs, apply_ref],
            })
        return reject_after_rollback({
            "status": "apply_failed",
            "apply_ref": apply_ref,
            "boundary_ref": boundary_ref,
            "apply_result": _model_relevant_record(apply_result),
            "evidence_refs": [*evidence_refs, apply_ref],
        })
    raw_transaction_ref = str(
        apply_result.get("transaction_dir") or ""
    )
    if not raw_transaction_ref:
        return reject_after_rollback({
            "status": "apply_contract_failed",
            "reason": (
                "canonical apply reported success without a transaction"
            ),
            "apply_ref": apply_ref,
            "boundary_ref": boundary_ref,
            "apply_result": _model_relevant_record(apply_result),
            "evidence_refs": [*evidence_refs, apply_ref],
        })
    applied_transaction_ref = Path(raw_transaction_ref).resolve()
    reported_files = set(
        _ordered_strings(
            list(apply_result.get("touched_files") or touched_files)
        )
    )
    if reported_files != set(touched_files):
        return reject_after_rollback({
            "status": "apply_contract_failed",
            "reason": (
                "patch runtime touched a file set different from the "
                "authorized unified diff"
            ),
            "apply_ref": apply_ref,
            "boundary_ref": boundary_ref,
            "authorized_touched_files": touched_files,
            "reported_touched_files": sorted(reported_files),
            "apply_result": _model_relevant_record(apply_result),
            "evidence_refs": [*evidence_refs, apply_ref],
        })
    conditional_integrity_after = _conditional_integrity_reports(
        spec,
        touched_files,
    )
    conditional_regressions = preprocessor_conditional_regressions(
        conditional_integrity_before,
        conditional_integrity_after,
    )
    if conditional_regressions:
        structural_integrity = {
            "schema": "semantic_patch_structural_integrity_v1",
            "status": "regressed",
            "reason": (
                "the patch introduced an invalid C preprocessor "
                "conditional structure"
            ),
            "files": touched_files,
            "regressions": conditional_regressions,
        }
        structural_ref = _store_runtime_result(
            state_root,
            kind="semantic-patch-structural-integrity",
            result=structural_integrity,
        )
        return reject_after_rollback({
            "status": "structural_integrity_denied",
            "reason": structural_integrity["reason"],
            "structural_integrity_ref": structural_ref,
            "structural_integrity": _model_relevant_record(
                structural_integrity
            ),
            "apply_result": _model_relevant_record(apply_result),
            "evidence_refs": [
                *evidence_refs,
                apply_ref,
                structural_ref,
            ],
        })
    structural_postcondition: dict[str, Any] | None = None
    if structural_contract_validation_enabled:
        structural_postcondition = _validate_structural_postconditions(
            spec,
            changed_files=touched_files,
            baseline=structural_baseline,
            unit=work_unit,
            changed_ranges=boundary.get("changed_ranges") or [],
        )
        structural_postcondition_ref = _store_runtime_result(
            state_root,
            kind="semantic-patch-structural-postcondition",
            result=structural_postcondition,
        )
        evidence_refs.append(structural_postcondition_ref)
        if structural_postcondition.get("status") == "failed":
            return reject_after_rollback({
                "status": "structural_postcondition_denied",
                "reason": (
                    "the patch did not realize its exact donor contract in "
                    "the donor structural container"
                ),
                "structural_postcondition_ref": (
                    structural_postcondition_ref
                ),
                "structural_postcondition": _model_relevant_record(
                    structural_postcondition
                ),
                "apply_result": _model_relevant_record(apply_result),
                "evidence_refs": [*evidence_refs, apply_ref],
            })
    transition = record_batch_patch(
        state,
        batch_id=str(batch["id"]),
        patch_id=str(
            candidate.get("patch_id")
            or stable_id(
                "semantic-patch-record",
                patch_signature,
                str(round_number),
            )
        ),
        patch_signature=patch_signature,
        gap_ids=gap_ids,
        touched_files=touched_files,
        work_unit_id=str(work_unit["id"]),
        kind="semantic_repair",
        authorized_changed_ranges=boundary.get(
            "changed_ranges"
        )
        or [],
        boundary_ref=boundary_ref,
        patch_evidence_ref=str(
            candidate.get("_semantic_action_ref") or ""
        )
        or None,
        transaction_ref=str(applied_transaction_ref),
        provider_obligations=provider_obligations,
        structural_contract_delta=(
            (structural_postcondition or {}).get("contract_delta")
            or {}
        ),
    )
    try:
        delattr(runtime, "_semantic_provider_code_index")
    except AttributeError:
        pass
    _write_state(state_root, state)
    return {
        "status": "applied",
        "apply_ref": apply_ref,
        "boundary_ref": boundary_ref,
        "batch_id": str(batch["id"]),
        "gap_ids": gap_ids,
        "touched_files": touched_files,
        **(
            {"structural_postcondition": structural_postcondition}
            if structural_postcondition is not None
            else {}
        ),
        "transition": transition,
        "evidence_refs": [*evidence_refs, apply_ref],
    }


def _repair_one_unit(
    runtime: SemanticClosureRuntime,
    spec: object,
    *,
    state_root: Path,
    state: dict[str, Any],
    work_unit: dict[str, Any],
    round_number: int,
) -> dict[str, Any]:
    unit_id = str(
        work_unit.get("id")
        or work_unit.get("work_unit_id")
        or ""
    )
    if not unit_id:
        raise ValueError("semantic repair unit has no stable identity")
    work_unit = {
        **work_unit,
        "id": unit_id,
        "work_unit_id": unit_id,
    }
    work_unit = _refresh_work_unit_scope_from_state(state, work_unit)
    prior_outcomes = _previous_unit_outcomes(
        state_root,
        state,
        work_unit_id=unit_id,
    )
    replayed_provider_expansions = _replay_provider_contract_expansions(
        runtime,
        spec,
        state_root=state_root,
        state=state,
        work_unit=work_unit,
        round_number=round_number,
    )
    if replayed_provider_expansions:
        _write_state(state_root, state)
    replay_changed_partition = any(
        item.get("new_owner_keys")
        for item in replayed_provider_expansions
        if isinstance(item, dict)
    )
    if replay_changed_partition:
        return {
            "work_unit_id": unit_id,
            "kind": "semantic_repair",
            "status": "evidence_repartition_required",
            "prior_outcome_count": len(prior_outcomes),
            "replayed_provider_expansions": [
                _model_relevant_record(item)
                for item in replayed_provider_expansions
            ],
            "outcomes": [],
            "patch_applied": False,
            "evidence_changed": True,
        }
    outcomes: list[dict[str, Any]] = []
    prompt_outcomes = list(prior_outcomes)
    evidence_changed = False
    step_limit = SEMANTIC_UNIT_ACTION_LIMIT
    while True:
        work_unit = _refresh_work_unit_scope_from_state(
            state,
            work_unit,
        )
        evidence_state_signature = _semantic_evidence_state_signature(
            work_unit
        )
        consumed_steps = semantic_unit_action_count(
            state,
            round_number=round_number,
            work_unit_id=unit_id,
            evidence_state_signature=evidence_state_signature,
        )
        if consumed_steps >= step_limit:
            break
        context = _work_unit_context(
            state_root,
            state,
            work_unit,
        )
        try:
            result = runtime.repair(
                {
                    "schema": "semantic_repair_request_v1",
                    "round": int(round_number),
                    "kind": "semantic_repair",
                    "work_unit_id": unit_id,
                    "context": context,
                    "previous_outcomes": deepcopy(prompt_outcomes),
                }
            )
        except ModelAPIError as exc:
            failure = _model_runtime_failure(
                exc,
                operation="semantic_repair",
                round_number=round_number,
                work_unit_id=unit_id,
            )
            failure_ref = _store_runtime_result(
                state_root,
                kind="semantic-model-runtime-failure",
                result=failure,
            )
            return {
                "work_unit_id": unit_id,
                "kind": "semantic_repair",
                "status": "runtime_unavailable",
                "prior_outcome_count": len(prior_outcomes),
                "outcomes": outcomes,
                "patch_applied": False,
                "runtime_failure": failure,
                "runtime_failure_ref": failure_ref,
            }
        if not isinstance(result, dict):
            result = {
                "status": "blocked",
                "reason": "repair runtime returned no object",
            }
        result_ref = _store_runtime_result(
            state_root,
            kind="semantic-repair-action",
            result=result,
        )
        action_evidence_refs = [result_ref]
        if result.get("status") == "runtime_unavailable":
            return {
                "work_unit_id": unit_id,
                "kind": "semantic_repair",
                "status": "runtime_unavailable",
                "prior_outcome_count": len(prior_outcomes),
                "outcomes": outcomes,
                "patch_applied": False,
                "runtime_failure": _model_relevant_record(result),
                "runtime_failure_ref": result_ref,
            }
        action_signature = str(
            result.get("action_signature")
            or _semantic_action_signature(
                work_unit_id=unit_id,
                result=result,
            )
        )
        status = str(result.get("status") or "blocked")
        if status == "already_satisfied":
            satisfied_gap_ids = _ordered_strings(
                list(result.get("gap_ids") or [])
            )
            change_required_gap_ids = {
                str(gap_id)
                for gap_id in work_unit.get("gap_ids") or []
                if (
                    (state.get("gaps") or {}).get(str(gap_id)) or {}
                ).get("status")
                == CHANGE_REQUIRED
            }
            if (
                not satisfied_gap_ids
                or not set(satisfied_gap_ids)
                <= change_required_gap_ids
            ):
                raise ValueError(
                    "already-satisfied source edits must name only "
                    "change-required gaps from the exact work unit"
                )
            structured_patch = (
                result.get("structured_source_patch")
                if isinstance(
                    result.get("structured_source_patch"), dict
                )
                else {}
            )
            reconciliation = reconcile_active_batch_affected_gaps(
                state,
                exact_edit_regions=(
                    structured_patch.get("edit_regions") or []
                ),
            )
            reconciliation_ref = _store_runtime_result(
                state_root,
                kind="semantic-active-batch-noop-reconciliation",
                result=reconciliation,
            )
            action_evidence_refs.append(reconciliation_ref)
            indirectly_pending = all(
                (
                    (state.get("gaps") or {}).get(gap_id) or {}
                ).get("status")
                == PATCH_PENDING_COMPILE
                for gap_id in satisfied_gap_ids
            )
            if indirectly_pending:
                # The no-op itself is not proof.  Exact overlap between its
                # current-source edit handle and a surviving batch postimage
                # attributes the gap to that earlier patch; unified compile
                # and fresh contract evaluation still decide closure.
                status = "indirect_patch_pending_compile"
                evidence_changed = True
            else:
                # Store a content-addressed fact whose identity depends on the
                # exact current edit handles and source preimages, not on a
                # model response id or rationale.  Repeating the same no-op
                # against unchanged source must not manufacture new decision
                # evidence or consume another semantic sweep.
                noop_fact_ref = _store_runtime_result(
                    state_root,
                    kind="semantic-authoritative-noop-fact",
                    result={
                        "schema": "semantic_authoritative_noop_fact_v1",
                        "work_unit_id": unit_id,
                        "gap_ids": satisfied_gap_ids,
                        "structured_status": structured_patch.get(
                            "status"
                        ),
                        "edit_regions": [
                            {
                                key: deepcopy(region.get(key))
                                for key in (
                                    "edit_handle_id",
                                    "file",
                                    "start_line",
                                    "end_line",
                                    "source_sha256",
                                    "edit_mode",
                                    "direct_gap_ids",
                                    "owner_keys",
                                )
                                if region.get(key)
                                not in (None, "", [], {})
                            }
                            for region in (
                                structured_patch.get("edit_regions") or []
                            )
                            if isinstance(region, dict)
                        ],
                        "reconciliation": reconciliation,
                        "evidence_state_signature": (
                            evidence_state_signature
                        ),
                    },
                )
                action_evidence_refs.append(noop_fact_ref)
                noop_refresh = reopen_gaps_for_authoritative_noop(
                    state,
                    gap_ids=satisfied_gap_ids,
                    evidence_refs=[noop_fact_ref],
                    round_number=round_number,
                )
                action_evidence_refs.extend(
                    reference
                    for reference in noop_refresh.get(
                        "evidence_refs"
                    )
                    or []
                    if reference not in action_evidence_refs
                )
                if noop_refresh.get("status") == "refreshed":
                    status = "decision_refresh_required"
                    evidence_changed = True
        if status == "patch":
            apply_result = _apply_patch_candidate(
                runtime,
                spec,
                state_root=state_root,
                state=state,
                work_unit=work_unit,
                candidate={
                    **result,
                    "_semantic_action_ref": result_ref,
                },
                round_number=round_number,
            )
            outcome = str(apply_result.get("status") or "")
            evidence_refs = [
                result_ref,
                *list(apply_result.get("evidence_refs") or []),
            ]
            attempt = record_unit_attempt(
                state,
                round_number=round_number,
                work_unit_id=unit_id,
                action_signature=action_signature,
                outcome=outcome,
                max_steps=SEMANTIC_UNIT_ACTION_LIMIT,
                evidence_refs=evidence_refs,
                evidence_state_signature=evidence_state_signature,
            )
            consumed_steps += 1
            _write_state(state_root, state)
            outcomes.append(
                {
                    "result_ref": result_ref,
                    "attempt": attempt,
                    "apply": apply_result,
                }
            )
            prompt_outcomes.append(
                {
                    "round": int(round_number),
                    "step": int(attempt.get("step") or 0),
                    "action_signature": action_signature,
                    "outcome": outcome,
                    "duplicate": bool(attempt.get("duplicate")),
                    "evidence": [
                        {
                            "ref": result_ref,
                            "record": _model_relevant_record(result),
                        },
                        {
                            "record": _model_relevant_record(
                                apply_result
                            ),
                        },
                    ],
                }
            )
            provider_expansion = (
                apply_result.get("provider_expansion")
                if isinstance(
                    apply_result.get("provider_expansion"),
                    dict,
                )
                else {}
            )
            if provider_expansion.get("new_owner_keys"):
                evidence_changed = True
            if _semantic_action_settlement(
                state,
                work_unit,
                attempt,
                evidence_changed=evidence_changed,
            ):
                break
            continue
        evidence_action = status in {
            "evidence_added",
            "tool_request_rejected",
            "protocol_invalid",
        }
        if evidence_action:
            attempt = record_unit_evidence_action(
                state,
                round_number=round_number,
                work_unit_id=unit_id,
                action_signature=action_signature,
                outcome=status,
                max_steps=SEMANTIC_UNIT_ACTION_LIMIT,
                evidence_refs=action_evidence_refs,
                evidence_state_signature=evidence_state_signature,
            )
            consumed_steps += 1
        else:
            attempt = record_unit_attempt(
                state,
                round_number=round_number,
                work_unit_id=unit_id,
                action_signature=action_signature,
                outcome=status,
                max_steps=SEMANTIC_UNIT_ACTION_LIMIT,
                evidence_refs=action_evidence_refs,
                evidence_state_signature=evidence_state_signature,
            )
            consumed_steps += 1
        if status == "evidence_added":
            expanded_bindings = [
                dict(item)
                for item in result.get("owner_bindings") or []
                if isinstance(item, dict)
            ]
            if expanded_bindings:
                existing_owner_keys = {
                    str(item.get("owner_key") or "")
                    for item in work_unit.get("owner_bindings") or []
                    if isinstance(item, dict)
                    and str(item.get("owner_key") or "")
                }
                expanded_owner_keys = {
                    str(item.get("owner_key") or "")
                    for item in expanded_bindings
                    if str(item.get("owner_key") or "")
                }
                evidence_changed = bool(
                    expanded_owner_keys - existing_owner_keys
                )
                expanded_files = _ordered_strings(
                    [
                        *list(work_unit.get("allowed_fileset") or []),
                        *[
                            item.get("file")
                            for item in expanded_bindings
                        ],
                    ]
                )
                existing_bindings = {
                    str(item.get("owner_key") or ""): deepcopy(item)
                    for item in work_unit.get("owner_bindings") or []
                    if isinstance(item, dict)
                    and str(item.get("owner_key") or "")
                }
                for binding in expanded_bindings:
                    owner_key = str(binding.get("owner_key") or "")
                    if owner_key:
                        existing_bindings.setdefault(
                            owner_key,
                            deepcopy(binding),
                        )
                work_unit = {
                    **work_unit,
                    "owner_bindings": list(existing_bindings.values()),
                    "allowed_fileset": expanded_files,
                }
                expanded_gap_ids = _ordered_strings(
                    list(result.get("expanded_gap_ids") or [])
                )
                if not expanded_gap_ids or not set(
                    expanded_gap_ids
                ) <= set(work_unit.get("gap_ids") or []):
                    raise ValueError(
                        "dependency expansion has no exact work-unit "
                        "gap provenance"
                    )
                record_exact_owner_expansion(
                    state,
                    work_unit_id=unit_id,
                    gap_ids=expanded_gap_ids,
                    owner_bindings=expanded_bindings,
                    evidence_ref=result_ref,
                    round_number=round_number,
                )
        if evidence_action:
            record_unit_replan_evidence(
                state,
                round_number=round_number,
                work_unit_id=unit_id,
                evidence_refs=[
                    result_ref,
                    *list(result.get("evidence_refs") or []),
                ],
            )
        _write_state(state_root, state)
        outcomes.append(
            {
                "result_ref": result_ref,
                "attempt": attempt,
                "result": result,
            }
        )
        prompt_outcomes.append(
            {
                "round": int(round_number),
                "step": int(attempt.get("step") or 0),
                "evidence_action": int(
                    attempt.get("evidence_action") or 0
                ),
                "action_signature": action_signature,
                "outcome": status,
                "duplicate": bool(attempt.get("duplicate")),
                "evidence": [
                    {
                        "ref": result_ref,
                        "record": _model_relevant_record(result),
                    },
                    *[
                        {"ref": reference}
                        for reference in action_evidence_refs[1:]
                    ],
                ],
            }
        )
        if _semantic_action_settlement(
            state,
            work_unit,
            attempt,
            evidence_changed=evidence_changed,
        ):
            break
    return {
        "work_unit_id": unit_id,
        "kind": "semantic_repair",
        "prior_outcome_count": len(prior_outcomes),
        "replayed_provider_expansions": [
            _model_relevant_record(item)
            for item in replayed_provider_expansions
        ],
        "outcomes": outcomes,
        "patch_applied": any(
            (outcome.get("apply") or {}).get("status") == "applied"
            for outcome in outcomes
        ),
        "evidence_changed": evidence_changed,
    }


def semantic_closure_resume_context(
    state_root: Path,
    *,
    case_id: str,
) -> dict[str, Any] | None:
    """Load the semantic ledger and materialize its exact resume context.

    A ledger created during initial compatibility is not resumable until the
    compile-valid handoff is durable.  Once that barrier exists, this module
    owns all interpretation of semantic state and record references.
    """

    state = load_semantic_closure_state(state_root, case_id=case_id)
    if state is None:
        return None
    handoff = state.get("compile_valid_semantic_handoff")
    if not (
        isinstance(handoff, dict)
        and handoff.get("status") == "complete"
    ):
        return None
    actual_revision = str(handoff.get("evaluator_revision") or "")
    summary = semantic_closure_summary(state)
    if actual_revision != COMPILE_VALID_HANDOFF_EVALUATOR_REVISION:
        return {
            "schema": "semantic_closure_resume_context_v1",
            "next_action": "semantic_handoff_stale",
            "stage": "handoff",
            "model_required": False,
            "semantic_handoff": deepcopy(handoff),
            "semantic_summary": summary,
            "expected_evaluator_revision": (
                COMPILE_VALID_HANDOFF_EVALUATOR_REVISION
            ),
            "actual_evaluator_revision": actual_revision or "unversioned",
        }

    context = {
        **semantic_closure_resume_route(state),
        "schema": "semantic_closure_resume_context_v1",
        "semantic_handoff": deepcopy(handoff),
        "semantic_summary": summary,
    }
    if context["next_action"] != "resume_compile_recovery":
        return context

    recovery = semantic_compile_recovery_context(state)
    if recovery is None:
        raise ValueError(
            "semantic resume route lost its compile-recovery context"
        )
    validation_ref = str(recovery.get("compile_validation_ref") or "")
    validation = (
        load_semantic_record(state_root, validation_ref)
        if validation_ref
        else {}
    )
    if not isinstance(validation, dict):
        raise ValueError(
            "semantic compile-recovery validation is not an object"
        )
    return {
        **context,
        **recovery,
        "validation": validation,
    }


def _last_batch_validation(
    state_root: Path,
    batch: dict[str, Any],
) -> dict[str, Any]:
    validations = list(batch.get("compile_validations") or [])
    if not validations:
        return {}
    reference = str(validations[-1].get("ref") or "")
    result = load_semantic_record(state_root, reference)
    return result if isinstance(result, dict) else {}


def _batch_transaction_refs(
    batch: dict[str, Any],
) -> list[Path]:
    """Return child transactions in apply order and the outer preimage last."""

    references = [
        str(patch.get("transaction_ref") or "")
        for patch in batch.get("patches") or []
        if isinstance(patch, dict)
        and str(patch.get("transaction_ref") or "")
    ]
    references.append(str(batch.get("preimage_ref") or ""))
    return [
        Path(reference).resolve()
        for reference in dict.fromkeys(references)
        if reference
    ]


def _finalize_batch_transactions(batch: dict[str, Any]) -> None:
    for transaction_ref in _batch_transaction_refs(batch):
        finalize_transaction(
            load_transaction(transaction_ref),
            status="compile_validated",
        )


def _canonical_compile_outcome(
    validation: dict[str, Any],
) -> bool:
    """Read an explicit compile outcome from a raw or persisted wrapper."""

    for key in ("compile_ok", "overall_ok"):
        if key not in validation:
            continue
        value = validation.get(key)
        if not isinstance(value, bool):
            raise ValueError(
                f"canonical validation field {key!r} is not boolean"
            )
        return value

    validation_path = str(validation.get("validation_path") or "")
    if validation_path:
        payload = json.loads(
            Path(validation_path).resolve().read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("overall_ok"),
            bool,
        ):
            raise ValueError(
                "persisted validation report has no boolean overall_ok"
            )
        return bool(payload["overall_ok"])

    raise ValueError(
        "canonical validation has no explicit compile outcome or report"
    )


def _record_active_batch_validation(
    spec: object,
    *,
    state_root: Path,
    state: dict[str, Any],
    validation: dict[str, Any],
    round_number: int,
) -> dict[str, Any]:
    """Persist one canonical compile result for the active semantic batch."""

    batch = _active_batch(state)
    if batch is None:
        return {
            "status": "not_required",
            "compile_ok": True,
        }
    compile_ok = _canonical_compile_outcome(validation)
    validation_record = {
        **deepcopy(validation),
        "compile_ok": compile_ok,
        "batch_id": str(batch["id"]),
        "round": int(round_number),
        "validated_at": utc_now(),
    }
    validation_ref = _store_runtime_result(
        state_root,
        kind="semantic-compile-validation",
        result=validation_record,
    )
    checkpoint_ref = str(
        validation.get("validation_path") or validation_ref
    )
    semantic_evaluation = (
        evaluate_active_batch_semantic_contracts(
            spec,
            state_root=state_root,
            state=state,
            batch=batch,
            checkpoint_ref=checkpoint_ref,
        )
        if compile_ok
        else None
    )
    transition = record_batch_compile_validation(
        state,
        batch_id=str(batch["id"]),
        compile_ok=compile_ok,
        validation_ref=validation_ref,
        checkpoint_ref=checkpoint_ref,
        round_number=round_number,
        gap_evaluations=(
            semantic_evaluation.get("evaluations") or []
            if semantic_evaluation is not None
            else None
        ),
    )
    if compile_ok:
        _finalize_batch_transactions(batch)
    _write_state(state_root, state)
    return {
        "status": (
            "compile_validated"
            if compile_ok
            else "compile_failed"
        ),
        "compile_ok": compile_ok,
        "validation_ref": validation_ref,
        "validation": validation_record,
        "semantic_contract_evaluation": semantic_evaluation,
        "transition": transition,
    }


def _validate_active_batch(
    runtime: SemanticClosureRuntime,
    spec: object,
    *,
    state_root: Path,
    state: dict[str, Any],
    round_number: int,
) -> dict[str, Any]:
    batch = _active_batch(state)
    if batch is None:
        return {
            "status": "not_required",
            "compile_ok": True,
        }
    try:
        validation = runtime.validate(
            {
                "schema": "semantic_batch_validation_request_v1",
                "round": int(round_number),
                "batch_id": str(batch["id"]),
                "patches": deepcopy(batch.get("patches") or []),
            }
        )
    except Exception as exc:
        failure = {
            "schema": "semantic_compile_validation_unavailable_v1",
            "status": "validation_unavailable",
            "compile_ok": False,
            "batch_id": str(batch["id"]),
            "round": int(round_number),
            "error": f"{exc.__class__.__name__}: {exc}",
            "error_type": exc.__class__.__name__,
            "at": utc_now(),
        }
        failure_ref = _store_runtime_result(
            state_root,
            kind="semantic-compile-validation-unavailable",
            result=failure,
        )
        return {
            **failure,
            "validation_ref": failure_ref,
        }
    if not isinstance(validation, dict):
        failure = {
            "schema": "semantic_compile_validation_unavailable_v1",
            "status": "validation_unavailable",
            "compile_ok": False,
            "batch_id": str(batch["id"]),
            "round": int(round_number),
            "error": "semantic validation runtime returned no object",
            "error_type": "invalid_runtime_result",
            "at": utc_now(),
        }
        failure_ref = _store_runtime_result(
            state_root,
            kind="semantic-compile-validation-unavailable",
            result=failure,
        )
        return {
            **failure,
            "validation_ref": failure_ref,
        }
    return _record_active_batch_validation(
        spec,
        state_root=state_root,
        state=state,
        validation=validation,
        round_number=round_number,
    )


class SemanticCompileRecoveryLifecycle:
    """Bind the canonical compiler-repair loop to one semantic patch batch."""

    execution_phase = "semantic_closure"

    def __init__(
        self,
        spec: object,
        *,
        state_root: Path,
        max_rounds: int,
    ) -> None:
        self.spec = spec
        self.state_root = state_root.resolve()
        self.max_rounds = max(1, int(max_rounds))
        state = self._state()
        batch = _active_batch(state)
        if batch is None or batch.get("status") != BATCH_COMPILE_FAILED:
            raise ValueError(
                "compiler recovery lifecycle requires one failed batch"
            )
        self.batch_id = str(batch["id"])

    def _state(self) -> dict[str, Any]:
        state = load_semantic_closure_state(
            self.state_root,
            case_id=str(getattr(self.spec, "case_id")),
        )
        if state is None:
            raise ValueError("semantic compile recovery has no ledger")
        return state

    @staticmethod
    def _running_round(state: dict[str, Any]) -> dict[str, Any] | None:
        running = [
            item
            for item in state.get("rounds") or []
            if isinstance(item, dict)
            and item.get("status") == ROUND_RUNNING
            and item.get("execution_mode") == CANONICAL_COMPILER_RECOVERY
        ]
        if len(running) > 1:
            raise ValueError("semantic ledger has multiple running rounds")
        return running[0] if running else None

    def budget_status(self) -> dict[str, Any]:
        state = self._state()
        running = self._running_round(state)
        completed = completed_semantic_recovery_rounds(state)
        return {
            "execution_phase": self.execution_phase,
            "budget_name": "semantic_repair",
            "completed": completed,
            "limit": self.max_rounds,
            "remaining": max(0, self.max_rounds - completed),
            "available": completed < self.max_rounds,
            "execution_policy_revision": (
                SEMANTIC_COMPILE_RECOVERY_POLICY_REVISION
            ),
            "running_round": (
                int(running.get("phase_round") or 0)
                if running is not None
                else None
            ),
        }

    def begin_round(
        self,
        *,
        controller_round: int,
    ) -> dict[str, Any]:
        state = self._state()
        running = self._running_round(state)
        if running is not None:
            pause_reason = running.pop("pause_reason", None)
            paused_at = running.pop("paused_at", None)
            was_paused = bool(pause_reason or paused_at)
            if was_paused:
                running["resumed_at"] = utc_now()
                _write_state(self.state_root, state)
            return {
                "status": "resumed",
                "semantic_round": int(running.get("number") or 0),
                "recovery_round": int(
                    running.get("phase_round") or 0
                ),
            }
        completed = completed_semantic_recovery_rounds(state)
        recovery_round = completed + 1
        number = max(
            [
                int(item.get("number") or 0)
                for item in state.get("rounds") or []
                if isinstance(item, dict)
            ]
            or [0]
        ) + 1
        if recovery_round > self.max_rounds:
            return {
                "status": "budget_exhausted",
                "semantic_round": number,
                "recovery_round": recovery_round,
            }
        record = begin_semantic_round(
            state,
            round_number=number,
            max_rounds=self.max_rounds,
            execution_mode=CANONICAL_COMPILER_RECOVERY,
            phase_round=recovery_round,
        )
        record["controller_round"] = int(controller_round)
        _write_state(self.state_root, state)
        return {
            "status": "started",
            "semantic_round": number,
            "recovery_round": recovery_round,
        }

    def record_transaction(
        self,
        *,
        transaction_ref: str,
        work_unit_id: str,
        touched_files: list[str],
        changed_ranges: list[dict[str, Any]],
        evidence: dict[str, Any],
        depends_on_patch_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        reference = str(transaction_ref or "").strip()
        if not reference:
            return {"status": "not_recorded", "reason": "no_transaction"}
        state = self._state()
        batch = _active_batch(state)
        if batch is None or str(batch.get("id") or "") != self.batch_id:
            raise ValueError("compiler recovery has no active semantic batch")
        transaction = load_transaction(Path(reference))
        replay_patch = _ensure_transaction_replay_patch(
            self.spec,
            Path(reference),
            touched_files=touched_files,
        )
        if replay_patch is None:
            return {
                "status": "not_recorded",
                "reason": "transaction_has_no_source_delta",
                "transaction_ref": str(Path(reference).resolve()),
            }
        manifest = transaction.manifest()
        evidence_ref = _store_runtime_result(
            self.state_root,
            kind="semantic-external-compile-recovery-transaction",
            result=evidence,
        )
        signature = str(manifest.get("patch_signature") or "") or stable_id(
            "semantic-compile-recovery-transaction",
            str(batch["id"]),
            str(Path(reference).resolve()),
            *touched_files,
        )
        patch_id = stable_id(
            "semantic-compile-recovery-patch",
            str(batch["id"]),
            signature,
            str(Path(reference).resolve()),
        )
        transition = record_batch_patch(
            state,
            batch_id=str(batch["id"]),
            patch_id=patch_id,
            patch_signature=signature,
            gap_ids=[],
            touched_files=touched_files,
            work_unit_id=str(work_unit_id),
            kind="compile_recovery",
            authorized_changed_ranges=changed_ranges,
            boundary_ref=evidence_ref,
            patch_evidence_ref=evidence_ref,
            transaction_ref=str(Path(reference).resolve()),
            depends_on_patch_ids=(depends_on_patch_ids or []),
        )
        _write_state(self.state_root, state)
        return {
            **transition,
            "transaction_ref": str(Path(reference).resolve()),
            "replay_patch_path": str(replay_patch),
            "evidence_ref": evidence_ref,
        }

    def record_validation(
        self,
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._state()
        running = self._running_round(state)
        if running is None:
            raise ValueError("compile recovery validation has no active round")
        round_number = int(running.get("number") or 0)
        result = _record_active_batch_validation(
            self.spec,
            state_root=self.state_root,
            state=state,
            validation=validation,
            round_number=round_number,
        )
        finish_semantic_round(
            state,
            round_number=round_number,
            status=(
                ROUND_VALIDATED
                if result.get("compile_ok")
                else ROUND_COMPILE_FAILED
            ),
        )
        _write_state(self.state_root, state)
        return result

    def finish_without_validation(
        self,
        *,
        reason: str,
        outcome_kind: str = "no_validation_worthy_change",
    ) -> dict[str, Any]:
        state = self._state()
        running = self._running_round(state)
        if running is None:
            return {"status": "not_required"}
        round_number = int(running.get("number") or 0)
        recovery_round = int(running.get("phase_round") or 0)
        infrastructure_failure = (
            outcome_kind == "terminal_infrastructure_failure"
        )
        if infrastructure_failure:
            # Transport and environment failures do not constitute one
            # attempted semantic repair round.  Keep this exact round open so
            # resume continues from its durable evidence and batch state.
            running["pause_reason"] = str(reason)
            running["paused_at"] = utc_now()
            remaining_rounds = max(
                0,
                self.max_rounds - recovery_round + 1,
            )
        else:
            running.pop("pause_reason", None)
            running.pop("paused_at", None)
            running["terminal_reason"] = str(reason)
            running["terminal_kind"] = str(outcome_kind)
            remaining_rounds = max(0, self.max_rounds - recovery_round)
            finish_semantic_round(
                state,
                round_number=round_number,
                status=(
                    ROUND_BUDGET_EXHAUSTED
                    if recovery_round >= self.max_rounds
                    else ROUND_NO_PROGRESS
                ),
            )
        _write_state(self.state_root, state)
        resume_allowed = bool(
            infrastructure_failure or remaining_rounds > 0
        )
        return {
            "status": "recorded",
            "semantic_round": round_number,
            "recovery_round": recovery_round,
            "reason": str(reason),
            "outcome_kind": str(outcome_kind),
            "remaining_rounds": remaining_rounds,
            "resume_allowed": resume_allowed,
            "resume_reason": (
                "infrastructure failure does not authorize rollback"
                if infrastructure_failure
                else "semantic recovery budget remains"
                if remaining_rounds > 0
                else "semantic recovery budget is exhausted"
            ),
        }

    def settle_controller_exit(
        self,
        controller_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve one controller return using the semantic ledger only."""

        state = self._state()
        batch = _active_batch(state)
        budget = self.budget_status()
        if batch is None or str(batch.get("id") or "") != self.batch_id:
            return {
                "status": "batch_resolved",
                "resume_allowed": False,
                "budget": budget,
            }

        lifecycle_result = (
            controller_result.get("repair_lifecycle")
            if isinstance(
                controller_result.get("repair_lifecycle"), dict
            )
            else {}
        )
        infrastructure_failure = bool(
            controller_result.get("environment_failure")
            or controller_result.get("infrastructure_failure")
            or lifecycle_result.get("outcome_kind")
            == "terminal_infrastructure_failure"
        )
        validation_pending_retry = bool(
            controller_result.get("round_validation_pending_retry")
        )
        if validation_pending_retry:
            return {
                "status": "active_batch_preserved",
                "resume_allowed": True,
                "reason": "validation_pending_retry",
                "batch_id": self.batch_id,
                "budget": budget,
            }
        if infrastructure_failure or budget["available"]:
            return {
                "status": "active_batch_preserved",
                "resume_allowed": True,
                "reason": (
                    "infrastructure_failure"
                    if infrastructure_failure
                    else "recovery_budget_remaining"
                ),
                "batch_id": self.batch_id,
                "budget": budget,
            }

        reason = (
            "semantic compiler recovery exhausted its automatic budget; "
            "the current candidate and its fresh compile diagnostics are "
            "retained"
        )
        return {
            "status": "budget_exhausted_candidate_retained",
            "resume_allowed": False,
            "reason": reason,
            "batch_id": self.batch_id,
            "budget": budget,
            "candidate_retained": True,
            "required_transition": (
                "resume from the current candidate with a larger explicit "
                "semantic repair budget or inspect the retained diagnostics"
            ),
        }

    def has_active_batch(self) -> bool:
        batch = _active_batch(self._state())
        return bool(
            batch is not None
            and str(batch.get("id") or "") == self.batch_id
        )


def _semantic_progress_snapshot(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Capture durable semantic progress, excluding audit-only history."""

    gaps = state.get("gaps") or {}
    batches = state.get("batches") or {}
    successful_evidence_actions = sorted(
        str(action.get("action_signature") or "")
        for round_record in state.get("rounds") or []
        if isinstance(round_record, dict)
        for unit in (round_record.get("units") or {}).values()
        if isinstance(unit, dict)
        for action in unit.get("evidence_actions") or []
        if isinstance(action, dict)
        and action.get("outcome") == "evidence_added"
        and not action.get("duplicate")
    )
    return {
        "gap_statuses": sorted(
            (str(gap_id), str(gap.get("status") or ""))
            for gap_id, gap in gaps.items()
            if isinstance(gap, dict)
        ),
        "owner_binding_count": sum(
            len(gap.get("owner_bindings") or [])
            for gap in gaps.values()
            if isinstance(gap, dict)
        ),
        "patch_count": sum(
            len(batch.get("patches") or [])
            for batch in batches.values()
            if isinstance(batch, dict)
        ),
        "active_batch_id": str(state.get("active_batch_id") or ""),
        "successful_evidence_actions": successful_evidence_actions,
    }


def _controller_result(
    state_root: Path,
    state: dict[str, Any],
    *,
    status: str,
    compile_validated: bool,
    max_rounds: int,
    rounds: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "schema": SEMANTIC_CLOSURE_CONTROLLER_SCHEMA,
        "status": status,
        "compile_validated": bool(compile_validated),
        "round_limit": int(max_rounds),
        "rounds": rounds,
        **fields,
        "summary": (
            summary
            if summary is not None
            else semantic_closure_summary(state)
        ),
        "state_path": str(
            state_root.resolve() / "semantic_closure" / "state.json"
        ),
    }


def _compile_recovery_handoff(
    *,
    state_root: Path,
    state: dict[str, Any],
    max_rounds: int,
    rounds: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    batch = _active_batch(state)
    if batch is None or batch.get("status") != BATCH_COMPILE_FAILED:
        raise ValueError("compile recovery handoff requires a failed batch")
    completed = completed_semantic_recovery_rounds(state)
    return _controller_result(
        state_root,
        state,
        status="compile_recovery_required",
        compile_validated=False,
        max_rounds=max_rounds,
        rounds=rounds,
        remaining_rounds=max(0, int(max_rounds) - completed),
        execution_policy_revision=(
            SEMANTIC_COMPILE_RECOVERY_POLICY_REVISION
        ),
        batch_id=str(batch["id"]),
        validation=deepcopy(validation),
        required_transition=(
            "resume the canonical compiler-repair controller on the current "
            "candidate tree and bind every surviving child transaction to "
            "the active semantic batch"
        ),
    )


def _gap_ids_with_status(
    state: dict[str, Any],
    status: str,
) -> list[str]:
    return sorted(
        str(gap_id)
        for gap_id, gap in (state.get("gaps") or {}).items()
        if isinstance(gap, dict) and gap.get("status") == status
    )


def _semantic_unit_budget_snapshot(
    state: dict[str, Any],
    *,
    source_units: dict[str, dict[str, Any]],
    round_number: int,
) -> list[dict[str, Any]]:
    """Project current unresolved components onto one round's action state."""

    budgets: list[dict[str, Any]] = []
    round_record = next(
        (
            item
            for item in state.get("rounds") or []
            if isinstance(item, dict)
            and int(item.get("number") or 0) == int(round_number)
        ),
        {},
    )
    for unresolved_unit in materialize_semantic_work_units(
        state,
        source_units=source_units,
        statuses=(CHANGE_REQUIRED,),
    ):
        refreshed_unit = _refresh_work_unit_scope_from_state(
            state,
            unresolved_unit,
        )
        unit_id = str(
            refreshed_unit.get("id")
            or refreshed_unit.get("work_unit_id")
            or ""
        )
        evidence_signature = _semantic_evidence_state_signature(
            refreshed_unit
        )
        unit_state = (
            (round_record.get("units") or {}).get(unit_id) or {}
            if isinstance(round_record, dict)
            else {}
        )
        evidence_state_actions = [
            action
            for field in ("attempts", "evidence_actions")
            for action in unit_state.get(field) or []
            if isinstance(action, dict)
            and str(action.get("evidence_state_signature") or "")
            == evidence_signature
        ]
        action_signatures = [
            str(action.get("action_signature") or "")
            for action in evidence_state_actions
        ]
        repeated_action = len(action_signatures) != len(
            set(action_signatures)
        )
        budgets.append(
            {
                "work_unit_id": unit_id,
                "gap_ids": _ordered_strings(
                    list(refreshed_unit.get("gap_ids") or [])
                ),
                "evidence_state_signature": evidence_signature,
                "consumed": semantic_unit_action_count(
                    state,
                    round_number=round_number,
                    work_unit_id=unit_id,
                    evidence_state_signature=evidence_signature,
                ),
                "limit": SEMANTIC_UNIT_ACTION_LIMIT,
                "progress_exhausted": repeated_action,
            }
        )
    return budgets


def _pause_semantic_sweep(
    *,
    state_root: Path,
    state: dict[str, Any],
    max_rounds: int,
    rounds: list[dict[str, Any]],
    round_result: dict[str, Any],
    reason: str,
    pending_gap_ids: list[str] | None = None,
    runtime_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one resumable sweep pause without consuming its round."""

    if reason == "decision_incomplete":
        status = "semantic_decision_incomplete"
        compile_validated = True
        round_fields = {
            "mode": "semantic_decision",
            "validation": {
                "status": status,
                "compile_ok": True,
                "reason": (
                    "the compile-valid gap snapshot still has gaps without "
                    "an explicit decision"
                ),
            },
        }
        result_fields = {
            "pending_gap_ids": list(pending_gap_ids or []),
        }
        required_transition = (
            "resume the same decision sweep for only the pending gaps; "
            "source mutation remains forbidden until the decision barrier "
            "is complete"
        )
    elif reason == "runtime_unavailable":
        if runtime_failure is None:
            raise ValueError("runtime pause requires failure evidence")
        status = "semantic_runtime_unavailable"
        compile_validated = _active_batch(state) is None
        round_fields = {"runtime_failure": runtime_failure}
        result_fields = {
            "runtime_failure": runtime_failure,
            "resume_route": semantic_closure_resume_route(state),
        }
        required_transition = (
            "resume the same semantic work unit after the model provider "
            "becomes available; do not compile or roll back the partial "
            "mutation batch"
        )
    else:
        raise ValueError(f"unsupported semantic sweep pause reason: {reason}")

    summary = semantic_closure_summary(state)
    round_result.update(deepcopy(round_fields))
    round_result["status"] = ROUND_RUNNING
    round_result["summary"] = summary
    rounds.append(round_result)
    _write_state(state_root, state)
    return _controller_result(
        state_root,
        state,
        status=status,
        compile_validated=compile_validated,
        max_rounds=max_rounds,
        rounds=rounds,
        summary=summary,
        required_transition=required_transition,
        **deepcopy(result_fields),
    )


def _run_semantic_closure_controller(
    spec: object,
    *,
    state_root: Path,
    runtime: SemanticClosureRuntime,
    max_rounds: int = 3,
) -> dict[str, Any]:
    """Process one gap snapshot, then compile once and hand failures to B/C."""

    recovery_limit = max(0, int(max_rounds))
    if recovery_limit <= 0:
        raise ValueError("semantic compile recovery needs a positive budget")
    state = load_semantic_closure_state(
        state_root,
        case_id=str(getattr(spec, "case_id")),
    )
    if state is None:
        return {
            "schema": SEMANTIC_CLOSURE_CONTROLLER_SCHEMA,
            "status": "not_initialized",
        }
    owner_binding_transition = normalize_semantic_owner_bindings(state)
    if owner_binding_transition.get("status") != "unchanged":
        _write_state(state_root, state)
    handoff = state.get("compile_valid_semantic_handoff")
    if not (
        isinstance(handoff, dict)
        and handoff.get("status") == "complete"
        and handoff.get("evaluator_revision")
        == COMPILE_VALID_HANDOFF_EVALUATOR_REVISION
    ):
        raise ValueError(
            "semantic controller requires a complete current-revision "
            "compile-valid handoff"
        )
    snapshot_transition = activate_semantic_gap_snapshot(
        state,
        checkpoint_ref=str(handoff.get("checkpoint_ref") or ""),
        evaluator_revision=COMPILE_VALID_HANDOFF_EVALUATOR_REVISION,
    )
    if (
        snapshot_transition.get("status") != "unchanged"
        or snapshot_transition.get("proven_gap_ids")
        or snapshot_transition.get("reopened_gap_ids")
    ):
        _write_state(state_root, state)

    affected_reconciliation = reconcile_active_batch_affected_gaps(state)
    if affected_reconciliation.get("status") == "reconciled":
        _write_state(state_root, state)

    active_batch = _active_batch(state)
    if (
        active_batch is not None
        and active_batch.get("status") == BATCH_COMPILE_FAILED
    ):
        return _compile_recovery_handoff(
            state_root=state_root,
            state=state,
            max_rounds=recovery_limit,
            rounds=[],
            validation=_last_batch_validation(
                state_root,
                active_batch,
            ),
        )

    running_round = next(
        (
            item
            for item in reversed(state.get("rounds") or [])
            if isinstance(item, dict)
            and item.get("status") == ROUND_RUNNING
        ),
        None,
    )
    completed_sweep = next(
        (
            item
            for item in reversed(state.get("rounds") or [])
            if isinstance(item, dict)
            and item.get("execution_mode") == SEMANTIC_GAP_SWEEP
            and item.get("status") != ROUND_RUNNING
        ),
        None,
    )
    source_units = source_units_from_state(state_root, state)
    repair_resumption = semantic_repair_resume_context(
        state,
        max_recovery_rounds=recovery_limit,
    )
    pending_decisions_after_activation = _gap_ids_with_status(
        state,
        PENDING_DECISION,
    )
    decision_epoch = state.get("pending_decision_epoch")
    decision_refresh_pending = bool(
        isinstance(decision_epoch, dict)
        and decision_epoch.get("source") in {
            "compile_valid_proof_review",
            "compile_valid_checkpoint_refresh",
            "decision_policy_refresh",
            "authoritative_noop_refresh",
        }
        and pending_decisions_after_activation
    )
    if (
        completed_sweep is not None
        and running_round is None
        and not decision_refresh_pending
    ):
        if repair_resumption.get("status") == "no_longer_required":
            dismiss_pending_repair_sweep(
                state,
                reason=(
                    "surviving compile-valid transactions already satisfied "
                    "every gap reopened by batch rollback"
                ),
            )
            _write_state(state_root, state)
            repair_resumption = {"status": "not_required"}
        if repair_resumption.get("status") != "ready":
            summary = semantic_closure_summary(state)
            return _controller_result(
                state_root,
                state,
                status=(
                    "semantic_closed"
                    if summary["static_semantic_closed"]
                    else "semantic_incomplete"
                ),
                compile_validated=True,
                max_rounds=recovery_limit,
                rounds=[],
                repair_resumption=repair_resumption,
                summary=summary,
            )

    initial_summary = semantic_closure_summary(state)
    if initial_summary["static_semantic_closed"] and active_batch is None:
        return _controller_result(
            state_root,
            state,
            status="semantic_closed",
            compile_validated=True,
            max_rounds=recovery_limit,
            rounds=[],
            summary=initial_summary,
        )

    round_result_resumption: dict[str, Any] | None = None
    if running_round is not None:
        if running_round.get("execution_mode") != SEMANTIC_GAP_SWEEP:
            raise ValueError(
                "semantic gap controller cannot consume a running "
                "canonical compiler-recovery round"
            )
        if (
            int(running_round.get("phase_round") or 0) != 1
            or int(running_round.get("max_rounds") or 0) != 1
        ):
            raise ValueError(
                "running semantic gap sweep has an invalid phase budget"
            )
        round_number = int(running_round.get("number") or 0)
        round_record = running_round
    else:
        round_number = max(
            [
                int(item.get("number") or 0)
                for item in state.get("rounds") or []
                if isinstance(item, dict)
            ]
            or [0]
        ) + 1
        round_record = begin_semantic_round(
            state,
            round_number=round_number,
            max_rounds=1,
            execution_mode=SEMANTIC_GAP_SWEEP,
            phase_round=1,
        )
        pending_repair_sweep = isinstance(
            state.get("pending_repair_sweep"),
            dict,
        )
        if (
            repair_resumption.get("status") == "ready"
            or (decision_refresh_pending and pending_repair_sweep)
        ):
            round_result_resumption = consume_pending_repair_sweep(
                state,
                round_number=round_number,
            )
    _write_state(state_root, state)

    rounds: list[dict[str, Any]] = []
    round_result: dict[str, Any] = {
        "round": round_number,
        "mode": "semantic_gap_sweep",
        "reviews": [],
        "repairs": [],
    }
    if running_round is None and round_result_resumption is not None:
        round_result["repair_resumption"] = round_result_resumption
    progress_before = _semantic_progress_snapshot(state)
    active_batch = _active_batch(state)
    if active_batch is not None and active_batch.get("status") != BATCH_OPEN:
        raise ValueError("gap sweep has an invalid active batch")

    if not round_record.get("decision_sweep_complete"):
        if active_batch is not None:
            raise ValueError(
                "source mutation started before the decision sweep completed"
            )
        decision_snapshot_gap_ids = _gap_ids_with_status(
            state,
            PENDING_DECISION,
        )
        decision_units = _dependency_order(
            materialize_semantic_work_units(
                state,
                source_units=source_units,
                statuses=(PENDING_DECISION,),
            )
        )
        review_runtime_failure: dict[str, Any] | None = None
        for work_unit in decision_units:
            review = _review_pending_gaps(
                runtime,
                state_root=state_root,
                state=state,
                work_unit=work_unit,
                round_number=round_number,
            )
            round_result["reviews"].append(review)
            if review.get("status") == "runtime_unavailable":
                review_runtime_failure = review
                break
        if review_runtime_failure is not None:
            return _pause_semantic_sweep(
                state_root=state_root,
                state=state,
                max_rounds=recovery_limit,
                rounds=rounds,
                round_result=round_result,
                reason="runtime_unavailable",
                runtime_failure=review_runtime_failure,
            )
        pending_after_review = _gap_ids_with_status(
            state,
            PENDING_DECISION,
        )
        round_result["decision_barrier"] = {
            "snapshot_generation": int(state.get("generation") or 0),
            "snapshot_gap_ids": decision_snapshot_gap_ids,
            "pending_gap_ids": pending_after_review,
            "complete": not pending_after_review,
        }
        if pending_after_review:
            return _pause_semantic_sweep(
                state_root=state_root,
                state=state,
                max_rounds=recovery_limit,
                rounds=rounds,
                round_result=round_result,
                reason="decision_incomplete",
                pending_gap_ids=pending_after_review,
            )
        round_record["decision_sweep_complete"] = True
        round_record["decision_snapshot_gap_ids"] = (
            decision_snapshot_gap_ids
        )
        _write_state(state_root, state)
    else:
        round_result["decision_barrier"] = {
            "snapshot_gap_ids": list(
                round_record.get("decision_snapshot_gap_ids") or []
            ),
            "pending_gap_ids": [],
            "complete": True,
            "resumed": True,
        }

    if not round_record.get("mutation_sweep_complete"):
        mutation_snapshot_gap_ids = _gap_ids_with_status(
            state,
            CHANGE_REQUIRED,
        )
        attempted_gap_ids: list[str] = []
        repair_runtime_failure: dict[str, Any] | None = None
        while True:
            remaining_before_pass = _gap_ids_with_status(
                state,
                CHANGE_REQUIRED,
            )
            if not remaining_before_pass:
                break
            repair_units = _dependency_order(
                materialize_semantic_work_units(
                    state,
                    source_units=source_units,
                    statuses=(CHANGE_REQUIRED,),
                )
            )
            accepted_patch = False
            evidence_repartition_required = False
            for work_unit in repair_units:
                current_change_required = [
                    str(gap_id)
                    for gap_id in work_unit.get("gap_ids") or []
                    if (
                        (state.get("gaps") or {}).get(str(gap_id))
                        or {}
                    ).get("status")
                    == CHANGE_REQUIRED
                ]
                if not current_change_required:
                    continue
                attempted_gap_ids.extend(current_change_required)
                repair_unit = {
                    **work_unit,
                    "gap_ids": current_change_required,
                    "gap_statuses": {
                        gap_id: CHANGE_REQUIRED
                        for gap_id in current_change_required
                    },
                    "unit_kind": "semantic_gap_repair",
                }
                repair = _repair_one_unit(
                    runtime,
                    spec,
                    state_root=state_root,
                    state=state,
                    work_unit=repair_unit,
                    round_number=round_number,
                )
                round_result["repairs"].append(repair)
                if repair.get("status") == "runtime_unavailable":
                    repair_runtime_failure = dict(
                        repair.get("runtime_failure") or repair
                    )
                    break
                if repair.get("patch_applied"):
                    accepted_patch = True
                    break
                if repair.get("evidence_changed"):
                    evidence_repartition_required = True
                    break
            if repair_runtime_failure is not None:
                break
            remaining_after_pass = _gap_ids_with_status(
                state,
                CHANGE_REQUIRED,
            )
            if (
                accepted_patch
                or evidence_repartition_required
                or remaining_after_pass != remaining_before_pass
            ):
                # A surviving mutation or a newly resolved exact owner can
                # change source locations and dependency components. Rebuild
                # from durable state instead of continuing a stale partition.
                continue
            unit_budgets = _semantic_unit_budget_snapshot(
                state,
                source_units=source_units,
                round_number=round_number,
            )
            if any(
                int(item["consumed"]) < int(item["limit"])
                and not bool(item.get("progress_exhausted"))
                for item in unit_budgets
            ):
                # A non-mutating model response is feedback, not settlement.
                # Finish every current evidence-state budget before deciding
                # that the mutation barrier has stalled.
                continue
            break
        if repair_runtime_failure is not None:
            return _pause_semantic_sweep(
                state_root=state_root,
                state=state,
                max_rounds=recovery_limit,
                rounds=rounds,
                round_result=round_result,
                reason="runtime_unavailable",
                runtime_failure=repair_runtime_failure,
            )
        remaining_change_required = _gap_ids_with_status(
            state,
            CHANGE_REQUIRED,
        )
        round_record["mutation_snapshot_gap_ids"] = _ordered_strings(
            [
                *list(
                    round_record.get("mutation_snapshot_gap_ids") or []
                ),
                *mutation_snapshot_gap_ids,
            ]
        )
        round_record["attempted_gap_ids"] = _ordered_strings(
            [
                *list(round_record.get("attempted_gap_ids") or []),
                *attempted_gap_ids,
            ]
        )
        round_record["unresolved_gap_ids"] = remaining_change_required
        round_record["mutation_sweep_complete"] = not (
            remaining_change_required
        )
        round_result["mutation_barrier"] = {
            "complete": not remaining_change_required,
            "all_units_attempted": set(mutation_snapshot_gap_ids)
            <= set(attempted_gap_ids),
            "snapshot_gap_ids": list(
                round_record["mutation_snapshot_gap_ids"]
            ),
            "attempted_gap_ids": list(
                round_record["attempted_gap_ids"]
            ),
            "unresolved_gap_ids": remaining_change_required,
            "accepted_patch_gap_ids": _gap_ids_with_status(
                state,
                PATCH_PENDING_COMPILE,
            ),
            "active_batch_id": str(
                state.get("active_batch_id") or ""
            )
            or None,
        }
        _write_state(state_root, state)
        if remaining_change_required:
            unit_budgets = _semantic_unit_budget_snapshot(
                state,
                source_units=source_units,
                round_number=round_number,
            )
            all_unit_budgets_exhausted = bool(unit_budgets) and all(
                int(item["consumed"]) >= int(item["limit"])
                or bool(item.get("progress_exhausted"))
                for item in unit_budgets
            )
            terminal_round_status = (
                ROUND_NO_PROGRESS
                if not unit_budgets
                else ROUND_BUDGET_EXHAUSTED
                if all_unit_budgets_exhausted
                else ROUND_RUNNING
            )
            active_batch = _active_batch(state)
            if (
                terminal_round_status != ROUND_RUNNING
                and active_batch is not None
            ):
                # The accepted transactions are already atomic. Validate
                # their aggregate before spending another semantic sweep;
                # unresolved gaps remain open and resume from that new
                # compile-valid checkpoint.
                round_record["mutation_sweep_complete"] = True
                round_result["mutation_barrier"][
                    "batch_ready_for_validation"
                ] = True
                _write_state(state_root, state)
            else:
                if terminal_round_status != ROUND_RUNNING:
                    finish_semantic_round(
                        state,
                        round_number=round_number,
                        status=terminal_round_status,
                    )
                    _write_state(state_root, state)
                round_result["validation"] = {
                    "status": "semantic_mutation_incomplete",
                    "compile_ok": active_batch is None,
                    "reason": (
                        "no accepted semantic patch is ready for unified "
                        "compile validation"
                    ),
                }
                round_result["summary"] = semantic_closure_summary(state)
                round_result["status"] = terminal_round_status
                round_result["unit_budgets"] = unit_budgets
                rounds.append(round_result)
                repair_resumption = semantic_repair_resume_context(
                    state,
                    max_recovery_rounds=recovery_limit,
                    max_gap_sweeps=recovery_limit,
                )
                resume_allowed = bool(
                    terminal_round_status == ROUND_RUNNING
                    or repair_resumption.get("status") == "ready"
                )
                return _controller_result(
                    state_root,
                    state,
                    status="semantic_mutation_incomplete",
                    compile_validated=active_batch is None,
                    max_rounds=recovery_limit,
                    rounds=rounds,
                    unresolved_gap_ids=remaining_change_required,
                    resume_allowed=resume_allowed,
                    repair_resumption=repair_resumption,
                    required_transition=(
                        "continue only the unresolved gap units in the same "
                        "sweep"
                        if terminal_round_status == ROUND_RUNNING
                        else "start the next provenance-backed mutation "
                        "sweep from the current compile-valid checkpoint"
                        if resume_allowed
                        else "the semantic mutation sweep is terminal because "
                        "no evidence-state action budget remains; preserve the "
                        "last compile-valid tree and report the unresolved gaps"
                    ),
                )
    else:
        round_result["mutation_barrier"] = {
            "complete": True,
            "all_units_attempted": True,
            "snapshot_gap_ids": list(
                round_record.get("mutation_snapshot_gap_ids") or []
            ),
            "attempted_gap_ids": list(
                round_record.get("attempted_gap_ids") or []
            ),
            "unresolved_gap_ids": list(
                round_record.get("unresolved_gap_ids") or []
            ),
            "accepted_patch_gap_ids": _gap_ids_with_status(
                state,
                PATCH_PENDING_COMPILE,
            ),
            "active_batch_id": str(
                state.get("active_batch_id") or ""
            )
            or None,
            "resumed_for_validation": True,
        }

    if _active_batch(state) is not None:
        validation = _validate_active_batch(
            runtime,
            spec,
            state_root=state_root,
            state=state,
            round_number=round_number,
        )
    else:
        validation = {
            "status": "compile_checkpoint_unchanged",
            "compile_ok": True,
        }
    round_result["validation"] = validation
    round_result["summary"] = semantic_closure_summary(state)
    progress_after = _semantic_progress_snapshot(state)
    round_result["semantic_progress"] = {
        "made_progress": progress_after != progress_before,
        "terminal_gap_delta": (
            int(round_result["summary"].get("terminal_gap_count") or 0)
            - int(initial_summary.get("terminal_gap_count") or 0)
        ),
        "open_gap_delta": (
            int(round_result["summary"].get("open_gap_count") or 0)
            - int(initial_summary.get("open_gap_count") or 0)
        ),
    }

    if validation.get("status") == "validation_unavailable":
        round_result["status"] = ROUND_RUNNING
        rounds.append(round_result)
        _write_state(state_root, state)
        return _controller_result(
            state_root,
            state,
            status="semantic_validation_incomplete",
            compile_validated=False,
            max_rounds=recovery_limit,
            rounds=rounds,
            unresolved_gap_ids=list(
                round_record.get("unresolved_gap_ids") or []
            ),
            required_transition=(
                "resume only the unified compile validation; do not repeat "
                "gap decisions or semantic mutation"
            ),
        )

    round_status = (
        ROUND_COMPILE_FAILED
        if validation.get("status") == "compile_failed"
        else ROUND_VALIDATED
    )
    finish_semantic_round(
        state,
        round_number=round_number,
        status=round_status,
    )
    _write_state(state_root, state)
    round_result["status"] = round_status
    round_result["summary"] = semantic_closure_summary(state)
    rounds.append(round_result)
    if validation.get("status") == "compile_failed":
        return _compile_recovery_handoff(
            state_root=state_root,
            state=state,
            max_rounds=recovery_limit,
            rounds=rounds,
            validation=dict(validation.get("validation") or {}),
        )

    summary = semantic_closure_summary(state)
    next_resumption = semantic_repair_resume_context(
        state,
        max_recovery_rounds=recovery_limit,
        max_gap_sweeps=recovery_limit,
    )
    pending_epoch = (
        state.get("pending_decision_epoch")
        if isinstance(state.get("pending_decision_epoch"), dict)
        else {}
    )
    authoritative_noop_review_pending = bool(
        pending_epoch.get("source") == "authoritative_noop_refresh"
        and _gap_ids_with_status(state, PENDING_DECISION)
    )
    completed_gap_sweeps = sum(
        1
        for item in state.get("rounds") or []
        if isinstance(item, dict)
        and item.get("execution_mode") == SEMANTIC_GAP_SWEEP
        and item.get("status") != ROUND_RUNNING
    )
    if (
        not summary["static_semantic_closed"]
        and (
            next_resumption.get("status") == "ready"
            or (
                authoritative_noop_review_pending
                and completed_gap_sweeps < recovery_limit
            )
        )
    ):
        resumed = _run_semantic_closure_controller(
            spec,
            state_root=state_root,
            runtime=runtime,
            max_rounds=recovery_limit,
        )
        return {
            **resumed,
            "rounds": [
                *rounds,
                *list(resumed.get("rounds") or []),
            ],
            "resumed_after_compile_validation": next_resumption,
            "resumed_for_authoritative_noop_review": (
                authoritative_noop_review_pending
            ),
        }
    return _controller_result(
        state_root,
        state,
        status=(
            "semantic_closed"
            if summary["static_semantic_closed"]
            else "semantic_incomplete"
        ),
        compile_validated=True,
        max_rounds=recovery_limit,
        rounds=rounds,
        unresolved_gap_ids=_gap_ids_with_status(
            state,
            CHANGE_REQUIRED,
        ),
        repair_resumption=next_resumption,
        summary=summary,
    )


def run_semantic_closure_controller(
    spec: object,
    *,
    state_root: Path,
    runtime: SemanticClosureRuntime,
    max_rounds: int = 3,
) -> dict[str, Any]:
    """Run semantic closure while preserving every accepted candidate edit."""

    try:
        return _run_semantic_closure_controller(
            spec,
            state_root=state_root,
            runtime=runtime,
            max_rounds=max_rounds,
        )
    except Exception as exc:
        state = load_semantic_closure_state(
            state_root,
            case_id=str(getattr(spec, "case_id")),
        )
        if state is not None:
            running = [
                item
                for item in state.get("rounds") or []
                if isinstance(item, dict)
                and item.get("status") == ROUND_RUNNING
            ]
            if len(running) == 1:
                running[0].setdefault(
                    "controller_failures",
                    [],
                ).append(
                    {
                        "at": utc_now(),
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "state_preserved": True,
                    }
                )
                _write_state(state_root, state)
            summary = semantic_closure_summary(state)
            active_batch_remains = _active_batch(state) is not None
            resume_route = semantic_closure_resume_route(state)
        else:
            summary = {}
            active_batch_remains = False
            resume_route = {}
        return {
            "schema": SEMANTIC_CLOSURE_CONTROLLER_SCHEMA,
            "status": "controller_failed",
            "compile_validated": not active_batch_remains,
            "error": f"{exc.__class__.__name__}: {exc}",
            "state_preserved": state is not None,
            "resume_route": resume_route,
            "summary": summary,
            "state_path": str(
                (
                    state_root.resolve()
                    / "semantic_closure"
                    / "state.json"
                )
            ),
        }
