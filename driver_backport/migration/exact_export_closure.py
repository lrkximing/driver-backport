from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import stable_id, utc_now
from driver_backport.migration.initial_compatibility import (
    _append_exact_export_to_current_provider,
    _unit_authorized_files,
)
from driver_backport.migration.semantic_closure_evaluator import (
    ExactCurrentTreeEvidence,
    evaluate_exact_cross_module_export_obligation,
)
from driver_backport.planning.dependency_schedule import work_unit_id
from driver_backport.planning.initial_compatibility_unit import (
    unit_contract_evidence,
)


def materialize_ready_exact_exports(
    spec: CaseSpec,
    *,
    state_root: Path,
    cycle_dir: Path,
    plan: dict[str, Any],
    allow_escalation: bool,
) -> dict[str, Any]:
    """Close exact cross-module export obligations against the current tree."""

    collector = ExactCurrentTreeEvidence(spec)
    results: list[dict[str, Any]] = []
    validation_targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for unit in plan.get("work_unit_plans") or []:
        if not isinstance(unit, dict):
            continue
        unit_id = work_unit_id(unit)
        authorized_files = _unit_authorized_files(
            unit,
            allow_escalation=allow_escalation,
        )
        for evidence in unit_contract_evidence(unit):
            requirement = evidence.get("semantic_requirement") or {}
            if "export" not in set(
                requirement.get("required_dimensions") or []
            ):
                continue
            donor_exports = [
                record
                for record in evidence.get("donor_exports") or []
                if isinstance(record, dict)
            ]
            if len(donor_exports) != 1:
                continue
            donor_export = donor_exports[0]
            identity = (
                str(evidence.get("id") or ""),
                str(donor_export.get("file") or ""),
                str(donor_export.get("macro") or ""),
                str(donor_export.get("raw") or ""),
            )
            if identity in seen:
                continue
            seen.add(identity)
            evaluation = evaluate_exact_cross_module_export_obligation(
                collector,
                evidence,
            )
            record = {
                "unit_id": unit_id,
                "contract_evidence_id": evidence.get("id"),
                "contract_identity": evidence.get("contract_identity"),
                "status": evaluation.get("status"),
                "reason": evaluation.get("reason"),
                "relations": dict(evaluation.get("relations") or {}),
            }
            if evaluation.get("status") == "ready":
                applied = _append_exact_export_to_current_provider(
                    spec,
                    state_root=state_root,
                    cycle_dir=cycle_dir,
                    unit_id=unit_id,
                    current_entity=dict(evaluation["current_entity"]),
                    donor_export=dict(evaluation["donor_export"]),
                    authorized_files=authorized_files,
                )
                record.update(applied)
                if applied.get("status") == "applied":
                    validation_targets.append(
                        {
                            "schema": "root_validation_target_v1",
                            "id": stable_id(
                                "exact-export-validation",
                                unit_id,
                                str(applied.get("symbol") or ""),
                            ),
                            "work_unit_id": unit_id,
                            "origin_work_unit_id": unit_id,
                            "unit_kind": "exact_export_contract_closure",
                            "execution_phase": "compiler_repair",
                            "diagnostic_signatures": [],
                            "origin_diagnostic_signatures": [],
                            "repair_decision": {
                                "mode": "exact_static_export_contract",
                            },
                            "origin_repair_decision": {
                                "mode": "exact_static_export_contract",
                            },
                        }
                    )
            results.append(record)
    counts = Counter(str(result.get("status") or "") for result in results)
    return {
        "schema": "exact_cross_module_export_closure_v1",
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "candidate_count": len(results),
        "applied_count": int(counts["applied"]),
        "already_satisfied_count": int(counts["already_satisfied"]),
        "blocked_count": int(counts["blocked"]),
        "changed_files": sorted(
            {
                str(result.get("file") or "")
                for result in results
                if result.get("status") == "applied"
                and str(result.get("file") or "")
            }
        ),
        "validation_targets": validation_targets,
        "results": results,
    }
