from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from driver_backport.core.common import (
    normalize_relpath,
    normalize_root_relative,
    ordered_unique_strings as _ordered_strings,
    stable_id,
)
from driver_backport.migration.semantic_closure_state import (
    BATCH_COMPILE_FAILED,
    BATCH_OPEN,
    PATCH_PENDING_COMPILE,
)


def _active_batch(state: dict[str, Any]) -> dict[str, Any]:
    batch_id = str(state.get("active_batch_id") or "")
    batch = (state.get("batches") or {}).get(batch_id)
    if not batch_id or not isinstance(batch, dict):
        raise ValueError(
            "semantic compile recovery requires one active patch batch"
        )
    if batch.get("status") not in {BATCH_OPEN, BATCH_COMPILE_FAILED}:
        raise ValueError(
            "semantic compile recovery batch is not mutable"
        )
    return batch


def _relative_path(spec: object, value: object) -> str | None:
    return normalize_root_relative(
        str(value or ""),
        root=Path(getattr(spec, "old_root_path")),
        preferred_prefix=str(getattr(spec, "driver_root", "") or ""),
    )


def _mapping_transaction_evidence(
    unit: dict[str, Any],
    *,
    known_patch_ids: set[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Read the exact transaction incidence already produced by CodeIndex."""

    patch_ids: list[str] = []
    evidence: list[dict[str, Any]] = []
    seen_evidence: set[tuple[object, ...]] = set()
    for mapping in unit.get("diagnostic_mappings") or []:
        if not isinstance(mapping, dict):
            continue
        patch_ids.extend(
            str(item)
            for item in mapping.get("semantic_transaction_ids") or []
            if str(item) in known_patch_ids
        )
        for raw in mapping.get("semantic_transaction_evidence") or []:
            if not isinstance(raw, dict):
                continue
            file_path = normalize_relpath(str(raw.get("file") or ""))
            start_line = int(raw.get("start_line") or raw.get("line") or 0)
            end_line = int(raw.get("end_line") or start_line)
            exact_patch_ids = _ordered_strings(
                item
                for item in raw.get("patch_ids") or []
                if str(item) in known_patch_ids
            )
            if (
                not file_path
                or start_line <= 0
                or end_line < start_line
                or not exact_patch_ids
            ):
                continue
            exact_gap_ids = _ordered_strings(raw.get("gap_ids") or [])
            identity = (
                file_path,
                start_line,
                end_line,
                tuple(exact_patch_ids),
                tuple(exact_gap_ids),
            )
            if identity in seen_evidence:
                continue
            seen_evidence.add(identity)
            evidence.append(
                {
                    "file": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "patch_ids": exact_patch_ids,
                    "gap_ids": exact_gap_ids,
                    "authority": "typed_code_graph_patch_incidence",
                }
            )
            patch_ids.extend(exact_patch_ids)
    return _ordered_strings(patch_ids), evidence


def _transaction_file_bindings(
    evidence: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_file: dict[str, dict[str, Any]] = {}
    for record in evidence:
        relative_path = normalize_relpath(str(record.get("file") or ""))
        if not relative_path:
            continue
        binding = by_file.setdefault(
            relative_path,
            {
                "file": relative_path,
                "patch_ids": [],
                "gap_ids": [],
                "source_loci": [],
                "postimage_regions": [],
                "authority": "active_patch_transaction_postimage",
                "evidence_authority": "typed_code_graph_patch_incidence",
                "binding_role": (
                    "transaction_provenance_and_write_authority"
                ),
            },
        )
        binding["patch_ids"] = _ordered_strings(
            [*binding["patch_ids"], *list(record.get("patch_ids") or [])]
        )
        binding["gap_ids"] = _ordered_strings(
            [*binding["gap_ids"], *list(record.get("gap_ids") or [])]
        )
        region = {
            "file": relative_path,
            "start_line": int(record.get("start_line") or 0),
            "end_line": int(record.get("end_line") or 0),
            "region_kind": "line_range",
            "patch_ids": list(record.get("patch_ids") or []),
            "gap_ids": list(record.get("gap_ids") or []),
        }
        source_locus = {
            "file": relative_path,
            "start_line": region["start_line"],
            "end_line": region["end_line"],
            "kind": "diagnostic_intersects_active_patch_hunk",
        }
        if source_locus not in binding["source_loci"]:
            binding["source_loci"].append(source_locus)
        if region not in binding["postimage_regions"]:
            binding["postimage_regions"].append(region)
    return [by_file[path] for path in sorted(by_file)]


def _semantic_transactions(
    patch_ids: Iterable[object],
    *,
    patches_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    transactions: list[dict[str, Any]] = []
    for patch_id in _ordered_strings(patch_ids):
        patch = patches_by_id.get(patch_id)
        if patch is None:
            continue
        transactions.append(
            {
                "id": patch_id,
                "transaction_id": patch_id,
                "kind": str(patch.get("kind") or ""),
                "gap_ids": list(patch.get("gap_ids") or []),
                "touched_files": list(patch.get("touched_files") or []),
                "transaction_ref": patch.get("transaction_ref"),
                "provider_obligations": [
                    dict(item)
                    for item in patch.get("provider_obligations") or []
                    if isinstance(item, dict)
                ],
            }
        )
    return transactions


def bind_semantic_compile_recovery_work_units(
    spec: object,
    *,
    state: dict[str, Any],
    diagnostic_work_units: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project CodeIndex semantic incidence onto canonical B/C work units.

    CodeIndex has already mapped diagnostics, partitioned common-upstream
    roots, and joined exact contract/patch nodes. This adapter only converts
    those typed graph facts into the fields consumed by the ordinary repair
    controller; it does not search source, infer providers, or repartition
    work units.
    """

    batch = _active_batch(state)
    batch_id = str(batch["id"])
    patches_by_id = {
        str(patch.get("id") or ""): dict(patch)
        for patch in batch.get("patches") or []
        if isinstance(patch, dict) and str(patch.get("id") or "")
    }
    known_patch_ids = set(patches_by_id)
    all_pending_gap_ids = {
        str(gap_id)
        for gap_id, gap in (state.get("gaps") or {}).items()
        if isinstance(gap, dict)
        and gap.get("status") == PATCH_PENDING_COMPILE
    }
    pending_gap_ids = {
        str(gap_id)
        for patch in patches_by_id.values()
        for gap_id in patch.get("gap_ids") or []
        if str(gap_id) in all_pending_gap_ids
    }
    direct_patch_ids_by_gap: dict[str, list[str]] = {}
    for gap_id in pending_gap_ids:
        direct_patch_ids_by_gap[gap_id] = _ordered_strings(
            patch_id
            for patch_id, patch in patches_by_id.items()
            if gap_id in {str(item) for item in patch.get("gap_ids") or []}
        )

    original_to_semantic_id: dict[str, str] = {}
    projected: list[dict[str, Any]] = []
    bound_unit_count = 0
    exact_binding_count = 0
    transaction_file_binding_count = 0

    for raw_unit in diagnostic_work_units:
        unit = deepcopy(raw_unit)
        original_id = str(unit.get("id") or "")
        if not original_id:
            raise ValueError(
                "semantic compiler recovery work unit has no identity"
            )
        semantic_id = stable_id(
            "semantic-recovery-wu",
            batch_id,
            original_id,
        )
        original_to_semantic_id[original_id] = semantic_id

        contract_binding = dict(
            unit.get("semantic_contract_binding") or {}
        )
        graph_gap_ids = {
            str(item)
            for item in [
                *list(unit.get("semantic_gap_ids") or []),
                *list(contract_binding.get("semantic_gap_ids") or []),
            ]
            if str(item) in pending_gap_ids
        }
        incident_patch_ids, incidence = _mapping_transaction_evidence(
            unit,
            known_patch_ids=known_patch_ids,
        )
        incidence_gap_ids = {
            str(item)
            for record in incidence
            for item in record.get("gap_ids") or []
            if str(item) in pending_gap_ids
        }
        required_gap_ids = sorted(graph_gap_ids | incidence_gap_ids)
        contract_patch_ids = _ordered_strings(
            patch_id
            for gap_id in required_gap_ids
            for patch_id in direct_patch_ids_by_gap.get(gap_id) or []
        )
        transaction_ids = _ordered_strings(
            [*incident_patch_ids, *contract_patch_ids]
        )
        transactions = _semantic_transactions(
            transaction_ids,
            patches_by_id=patches_by_id,
        )
        file_bindings = _transaction_file_bindings(incidence)
        repair_authorities = [
            {
                **dict(region),
                "authority": "active_patch_transaction_postimage",
                "evidence_authority": (
                    "typed_code_graph_patch_incidence"
                ),
                "batch_id": batch_id,
            }
            for binding in file_bindings
            for region in binding.get("postimage_regions") or []
        ]

        repair_owners = deepcopy(unit.get("repair_owners") or [])
        existing_owner_files = {
            _relative_path(spec, owner.get("owner_file"))
            for owner in repair_owners
            if isinstance(owner, dict)
            and owner.get("allowed_as_patch_target")
        }
        for binding in file_bindings:
            relative_path = str(binding["file"])
            if relative_path in existing_owner_files:
                continue
            repair_owners.append(
                {
                    "id": stable_id(
                        "semantic-transaction-file-owner",
                        batch_id,
                        relative_path,
                        *list(binding.get("patch_ids") or []),
                    ),
                    "owner_kind": "active_semantic_transaction_file",
                    "owner_file": relative_path,
                    "owner_symbol": None,
                    "reason": (
                        "CodeIndex mapped this compiler diagnostic to an "
                        "exact active semantic patch region"
                    ),
                    "evidence": list(binding.get("patch_ids") or []),
                    "allowed_as_patch_target": True,
                    "allowed_patch_files": [relative_path],
                    "allowed_patch_permissions": {
                        relative_path: (
                            "active_semantic_transaction_recovery"
                        )
                    },
                    "owner_permission": (
                        "active_semantic_transaction_recovery"
                    ),
                    "old_exists": (
                        Path(getattr(spec, "old_root_path"))
                        / relative_path
                    ).exists(),
                    "symptom_files": [relative_path],
                    "trigger_files": [relative_path],
                    "read_only_evidence_files": [],
                    "diagnostic_structure": {
                        "enabled": True,
                        "backend": "typed_code_graph_patch_incidence",
                        "root_entity_id": None,
                        "owner_node_id": None,
                    },
                }
            )

        provider_obligations_by_id = {
            str(item.get("id") or ""): dict(item)
            for transaction in transactions
            for item in transaction.get("provider_obligations") or []
            if isinstance(item, dict) and str(item.get("id") or "")
        }
        exact_bindings = [
            {
                "semantic_gap_id": gap_id,
                "binding_types": ["typed_code_graph_exact_contract"],
                "incident_patch_ids": [
                    patch_id
                    for patch_id in incident_patch_ids
                    if gap_id
                    in {
                        str(item)
                        for item in patches_by_id[patch_id].get("gap_ids")
                        or []
                    }
                ],
                "semantic_patch_ids": list(
                    direct_patch_ids_by_gap.get(gap_id) or []
                ),
                "source_loci": [
                    {
                        "file": record["file"],
                        "start_line": record["start_line"],
                        "end_line": record["end_line"],
                        "kind": "diagnostic_intersects_active_patch_hunk",
                    }
                    for record in incidence
                    if gap_id in set(record.get("gap_ids") or [])
                ],
                "contract_keys": list(
                    contract_binding.get("contract_identities") or []
                ),
                "node_ids": list(
                    contract_binding.get("contract_node_ids") or []
                ),
            }
            for gap_id in required_gap_ids
        ]

        unit.update(
            {
                "id": semantic_id,
                "compiler_work_unit_id": original_id,
                "semantic_gap_ids": required_gap_ids,
                "required_semantic_gap_ids": required_gap_ids,
                "pending_semantic_gap_ids": required_gap_ids,
                "blocked_semantic_gap_ids": [],
                "incident_patch_ids": incident_patch_ids,
                "semantic_transaction_ids": transaction_ids,
                "semantic_transactions": transactions,
                "post_materialization_provider_obligations": [
                    provider_obligations_by_id[key]
                    for key in sorted(provider_obligations_by_id)
                ],
                "repair_transaction_authorities": repair_authorities,
                "repair_owners": repair_owners,
                "semantic_recovery_context": {
                    "schema": "semantic_compile_recovery_context_v4",
                    "batch_id": batch_id,
                    "batch_status": str(batch.get("status") or ""),
                    "incident_patch_ids": incident_patch_ids,
                    "exact_incident_patch_ids": incident_patch_ids,
                    "semantic_transaction_ids": transaction_ids,
                    "exact_bindings": exact_bindings,
                    "transaction_file_bindings": file_bindings,
                },
                "semantic_contract_binding": {
                    **contract_binding,
                    "authority": "typed_code_graph_exact_contract",
                    "semantic_gap_ids": required_gap_ids,
                },
            }
        )
        if required_gap_ids or incident_patch_ids:
            bound_unit_count += 1
            exact_binding_count += len(exact_bindings)
            transaction_file_binding_count += len(file_bindings)
        projected.append(unit)

    for unit in projected:
        unit["depends_on_work_unit_ids"] = sorted(
            {
                original_to_semantic_id.get(str(item), str(item))
                for item in unit.get("depends_on_work_unit_ids") or []
                if str(item)
            }
        )
    return {
        "work_units": projected,
        "summary": {
            "schema": "semantic_compile_recovery_binding_v3",
            "batch_id": batch_id,
            "diagnostic_work_unit_count": len(diagnostic_work_units),
            "output_work_unit_count": len(projected),
            "partition_authority": "diagnostic_code_graph_preserved",
            "projection_authority": "typed_code_graph_exact_contract",
            "bound_work_unit_count": bound_unit_count,
            "unbound_work_unit_count": (
                len(diagnostic_work_units) - bound_unit_count
            ),
            "exact_binding_count": exact_binding_count,
            "transaction_file_binding_count": (
                transaction_file_binding_count
            ),
            "pending_semantic_gap_count": len(pending_gap_ids),
        },
    }
