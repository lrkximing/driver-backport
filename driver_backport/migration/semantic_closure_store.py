from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from driver_backport.core.common import (
    sha256_file,
    stable_json_sha256,
    write_json,
)
from driver_backport.core.content_store import (
    CONTENT_RECORD_REF_PREFIX,
    load_content_record,
    parse_content_record_ref,
    put_content_record,
)
from driver_backport.migration.semantic_closure_projection import (
    project_initial_semantic_differences,
)
from driver_backport.migration.semantic_closure_state import (
    new_semantic_closure_state,
    record_initial_unit_outcomes_snapshot,
    register_difference_gaps,
    semantic_closure_summary,
    validate_semantic_closure_state,
)


SEMANTIC_CLOSURE_DIR_NAME = "semantic_closure"
SEMANTIC_CLOSURE_STATE_NAME = "state.json"
SEMANTIC_CLOSURE_RECORD_REF_PREFIX = "semantic-record"


def semantic_closure_dir(state_root: Path) -> Path:
    return state_root.resolve() / SEMANTIC_CLOSURE_DIR_NAME


def semantic_closure_state_path(state_root: Path) -> Path:
    return semantic_closure_dir(state_root) / SEMANTIC_CLOSURE_STATE_NAME


def _artifact_descriptor(
    state_root: Path,
    *,
    kind: str,
    path: Path | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "kind": str(kind),
            "record_ref": put_semantic_record(
                state_root,
                kind=kind,
                record=payload,
            ),
        }
    resolved = path.resolve()
    try:
        locator = str(resolved.relative_to(state_root.resolve()))
    except ValueError:
        return {
            "kind": str(kind),
            "record_ref": put_semantic_record(
                state_root,
                kind=kind,
                record=payload,
            ),
            "source_path": str(resolved),
            "source_sha256": sha256_file(resolved),
            "source_size_bytes": resolved.stat().st_size,
        }
    return {
        "kind": str(kind),
        "location_kind": "state_root_relative",
        "path": locator,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _artifact_path(
    state_root: Path,
    descriptor: dict[str, Any],
) -> Path | None:
    if descriptor.get("record_ref"):
        return None
    raw_path = str(descriptor.get("path") or "")
    if not raw_path:
        raise ValueError("semantic source artifact has no path")
    if descriptor.get("location_kind") == "state_root_relative":
        return (state_root.resolve() / raw_path).resolve()
    return Path(raw_path).resolve()


def put_semantic_record(
    state_root: Path,
    *,
    kind: str,
    record: Any,
) -> str:
    """Persist immutable evidence and return a content-addressed reference."""

    return put_content_record(
        state_root,
        kind=f"semantic-{kind}",
        record=record,
    )


def _parse_legacy_record_ref(reference: str) -> tuple[str, str]:
    parts = str(reference or "").split(":")
    if (
        len(parts) != 3
        or parts[0] != SEMANTIC_CLOSURE_RECORD_REF_PREFIX
    ):
        raise ValueError(f"invalid semantic record reference: {reference}")
    kind = str(parts[1])
    if not kind or not all(
        character.isalnum() or character == "-"
        for character in kind
    ):
        raise ValueError(f"invalid semantic record kind: {reference}")
    digest = str(parts[2])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            f"invalid semantic record digest: {reference}"
        )
    return kind, digest


def load_semantic_record(
    state_root: Path,
    reference: str,
) -> Any:
    if str(reference).startswith(f"{CONTENT_RECORD_REF_PREFIX}:"):
        return load_content_record(state_root, reference)

    kind, digest = _parse_legacy_record_ref(reference)
    path = (
        semantic_closure_dir(state_root)
        / "records"
        / kind
        / f"{digest}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"semantic record does not exist: {reference}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if stable_json_sha256(payload) != digest:
        raise ValueError(
            f"semantic record content does not match {reference}"
        )
    return payload


def semantic_record_kind(reference: str) -> str:
    """Return the semantic kind for current and legacy record references."""

    if str(reference).startswith(f"{CONTENT_RECORD_REF_PREFIX}:"):
        kind, _digest = parse_content_record_ref(reference)
        prefix = "semantic-"
        if not kind.startswith(prefix):
            raise ValueError(
                f"content record is not semantic evidence: {reference}"
            )
        return kind[len(prefix) :]
    kind, _digest = _parse_legacy_record_ref(reference)
    return kind


def write_semantic_closure_state(
    state_root: Path,
    state: dict[str, Any],
) -> Path:
    validate_semantic_closure_state(state)
    path = semantic_closure_state_path(state_root)
    write_json(path, state)
    return path


def load_semantic_closure_state(
    state_root: Path,
    *,
    case_id: str | None = None,
) -> dict[str, Any] | None:
    path = semantic_closure_state_path(state_root)
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    validate_semantic_closure_state(state)
    if case_id is not None and str(state.get("case_id") or "") != str(
        case_id
    ):
        raise ValueError(
            "semantic closure state belongs to another case: "
            f"{state.get('case_id')} != {case_id}"
        )
    return state


def source_units_from_state(
    state_root: Path,
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for unit_id, reference in (
        state.get("source_unit_refs") or {}
    ).items():
        record = load_semantic_record(
            state_root,
            str(reference),
        )
        if not isinstance(record, dict):
            raise ValueError(
                f"semantic source unit is not an object: {unit_id}"
            )
        record_id = str(
            record.get("id") or record.get("work_unit_id") or ""
        )
        if record_id != str(unit_id):
            raise ValueError(
                f"semantic source unit identity mismatch: {unit_id}"
            )
        result[str(unit_id)] = record
    return result


def load_initial_analysis_artifact(
    state_root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Load the immutable initial analysis after verifying its identity."""

    key = "analysis_artifact"
    descriptor = (
        (state.get("source_analysis") or {}).get(key)
        if isinstance(state.get("source_analysis"), dict)
        else None
    )
    if not isinstance(descriptor, dict):
        raise ValueError(
            f"semantic source artifact descriptor is missing: {key}"
        )
    if descriptor.get("record_ref"):
        payload = load_semantic_record(
            state_root,
            str(descriptor["record_ref"]),
        )
    else:
        path = _artifact_path(state_root, descriptor)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"semantic source artifact is missing: {key}"
            )
        if sha256_file(path) != str(descriptor.get("sha256") or ""):
            raise ValueError(
                f"semantic source artifact digest mismatch: {key}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"semantic source artifact is not an object: {key}"
        )
    return payload


def assert_semantic_source_artifacts_integrity(
    state_root: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Verify immutable analysis/plan descriptors without loading big JSON."""

    record_refs: list[str] = []
    source_analysis = state.get("source_analysis") or {}
    for key in ("analysis_artifact", "plan_artifact"):
        descriptor = source_analysis.get(key)
        if not isinstance(descriptor, dict):
            raise ValueError(
                f"semantic source artifact descriptor is missing: {key}"
            )
        record_ref = str(descriptor.get("record_ref") or "")
        if record_ref:
            load_semantic_record(state_root, record_ref)
            record_refs.append(record_ref)
            continue
        path = _artifact_path(state_root, descriptor)
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"semantic source artifact is missing: {key}"
            )
        if sha256_file(path) != str(descriptor.get("sha256") or ""):
            raise ValueError(
                f"semantic source artifact digest mismatch: {key}"
            )
    manifest_ref = str(
        source_analysis.get("demand_evidence_manifest_ref") or ""
    )
    checked_demand_records = 0
    if manifest_ref:
        manifest = load_semantic_record(state_root, manifest_ref)
        if not isinstance(manifest, dict):
            raise ValueError("initial demand evidence manifest is invalid")
        for item in manifest.get("records") or []:
            if not isinstance(item, dict):
                raise ValueError(
                    "initial demand evidence manifest records must be objects"
                )
            evidence_ref = str(item.get("source_evidence_ref") or "")
            evidence = load_content_record(state_root, evidence_ref)
            if not isinstance(evidence, dict) or (
                str(evidence.get("id") or "")
                != str(item.get("id") or "")
                or str(evidence.get("contract_identity") or "")
                != str(item.get("contract_identity") or "")
            ):
                raise ValueError(
                    "initial demand evidence manifest identity mismatch"
                )
            checked_demand_records += 1
        record_refs.append(manifest_ref)
    return {
        "status": "ok",
        "checked_artifact_count": 2,
        "checked_demand_evidence_count": checked_demand_records,
        "record_refs": record_refs,
    }


def initialize_semantic_closure_from_artifacts(
    *,
    state_root: Path,
    case_id: str,
    analysis: dict[str, Any],
    plan: dict[str, Any],
    analysis_path: Path | None = None,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    """Create the ledger from one immutable initial difference snapshot."""

    existing = load_semantic_closure_state(
        state_root,
        case_id=case_id,
    )
    if existing is not None:
        return {
            "status": "already_initialized",
            "state_path": str(
                semantic_closure_state_path(state_root)
            ),
            "summary": semantic_closure_summary(existing),
        }

    if not bool(analysis.get("preparation_complete", True)):
        return {
            "status": "source_analysis_incomplete",
            "analysis_path": str(analysis_path) if analysis_path else None,
            "plan_path": str(plan_path) if plan_path else None,
            "preparation_blockers": [
                deepcopy(item)
                for item in analysis.get("preparation_blockers") or []
                if isinstance(item, dict)
            ],
            "summary": {
                "static_semantic_closed": False,
                "source_analysis_complete": False,
            },
        }

    demand_evidence_manifest_records: list[dict[str, str]] = []
    for item in analysis.get("demand_ledger") or []:
        if not isinstance(item, dict):
            raise ValueError("demand ledger records must be objects")
        evidence_ref = str(item.get("source_evidence_ref") or "")
        if not evidence_ref:
            continue
        evidence = load_content_record(state_root, evidence_ref)
        if not isinstance(evidence, dict) or (
            str(evidence.get("id") or "")
            != str(item.get("id") or "")
            or str(evidence.get("contract_identity") or "")
            != str(item.get("contract_identity") or "")
        ):
            raise ValueError(
                "persisted demand evidence does not match its ledger entry"
            )
        demand_evidence_manifest_records.append(
            {
                "id": str(item.get("id") or ""),
                "contract_identity": str(
                    item.get("contract_identity") or ""
                ),
                "source_evidence_ref": evidence_ref,
            }
        )
    demand_evidence_manifest_ref = (
        put_semantic_record(
            state_root,
            kind="initial-demand-evidence-manifest",
            record={
                "schema": "initial_demand_evidence_manifest_v1",
                "records": demand_evidence_manifest_records,
            },
        )
        if demand_evidence_manifest_records
        else None
    )

    projection = project_initial_semantic_differences(
        analysis,
        plan,
    )
    analysis_artifact = _artifact_descriptor(
        state_root,
        kind="initial-analysis",
        path=analysis_path,
        payload=analysis,
    )
    plan_artifact = _artifact_descriptor(
        state_root,
        kind="initial-plan",
        path=plan_path,
        payload=plan,
    )
    state = new_semantic_closure_state(
        case_id=case_id,
        source_analysis={
            "schema": analysis.get("schema"),
            "analysis_revision": analysis.get("analysis_revision"),
            "analysis_artifact": analysis_artifact,
            "plan_artifact": plan_artifact,
            "demand_evidence_manifest_ref": (
                demand_evidence_manifest_ref
            ),
            "demand_evidence_record_count": len(
                demand_evidence_manifest_records
            ),
        },
    )
    state["source_unit_refs"] = {
        unit_id: put_semantic_record(
            state_root,
            kind="difference-unit",
            record=unit,
        )
        for unit_id, unit in (
            projection.get("source_units") or {}
        ).items()
    }
    required_hunk_ids = sorted(
        {
            str(hunk_id)
            for gap in projection.get("gaps") or []
            if isinstance(gap, dict)
            for hunk_id in gap.get("raw_hunk_ids") or []
            if str(hunk_id)
        }
    )
    raw_hunks_by_id = {
        str(item.get("id") or ""): item
        for item in analysis.get("raw_hunks") or []
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    missing_hunk_ids = sorted(
        set(required_hunk_ids) - set(raw_hunks_by_id)
    )
    if missing_hunk_ids:
        raise ValueError(
            "semantic projection references missing raw hunks: "
            + ", ".join(missing_hunk_ids)
        )
    state["raw_hunk_refs"] = {
        hunk_id: put_semantic_record(
            state_root,
            kind="raw-hunk",
            record=raw_hunks_by_id[hunk_id],
        )
        for hunk_id in required_hunk_ids
    }

    gap_records: list[dict[str, Any]] = []
    for gap in projection.get("gaps") or []:
        if not isinstance(gap, dict):
            continue
        evidence_ref = put_semantic_record(
            state_root,
            kind="contract-evidence",
            record=gap.get("contract_evidence") or {},
        )
        gap_records.append(
            {
                key: deepcopy(gap.get(key))
                for key in (
                    "id",
                    "contract_identity",
                    "raw_hunk_ids",
                    "source_work_unit_ids",
                    "owner_bindings",
                )
                if gap.get(key) not in (None, "", [], {})
            }
            | {"contract_evidence_ref": evidence_ref}
        )
    registration = register_difference_gaps(
        state,
        gaps=gap_records,
        discovery_id="initial-compatibility",
        source="initial_compatibility",
    )
    state_path = write_semantic_closure_state(
        state_root,
        state,
    )
    return {
        "status": "initialized",
        "state_path": str(state_path),
        "analysis_artifact": analysis_artifact,
        "plan_artifact": plan_artifact,
        "registration": registration,
        "summary": semantic_closure_summary(state),
    }


def snapshot_initial_compatibility_outcomes(
    *,
    state_root: Path,
    case_id: str,
    cycle_dir: Path,
) -> dict[str, Any]:
    """Freeze final B0 outcomes after its queue and first compile complete."""

    state = load_semantic_closure_state(
        state_root,
        case_id=case_id,
    )
    if state is None:
        return {"status": "semantic_ledger_not_initialized"}
    assert_semantic_source_artifacts_integrity(state_root, state)
    existing_ref = str(state.get("initial_unit_outcomes_ref") or "")
    if existing_ref:
        snapshot = load_semantic_record(state_root, existing_ref)
        if not isinstance(snapshot, dict):
            raise ValueError("initial unit outcome snapshot is not an object")
        return {
            "status": "already_snapshotted",
            "record_ref": existing_ref,
            "unit_count": len(snapshot.get("unit_outcomes") or {}),
            "summary": deepcopy(snapshot.get("summary") or {}),
        }

    from driver_backport.migration.initial_compatibility import (
        initial_compatibility_unit_outcomes,
    )

    outcome_snapshot = initial_compatibility_unit_outcomes(
        cycle_dir.resolve(),
    )
    if outcome_snapshot.get("status") != "resolved":
        return {
            "status": "outcomes_unavailable",
            "source_status": str(
                outcome_snapshot.get("status") or "unknown"
            ),
            "cycle_dir": str(cycle_dir.resolve()),
        }

    snapshotted_outcomes: dict[str, dict[str, Any]] = {}
    for unit_id, raw_outcome in (
        outcome_snapshot.get("unit_outcomes") or {}
    ).items():
        if not isinstance(raw_outcome, dict):
            continue
        outcome = deepcopy(raw_outcome)
        patch_record_refs: list[str] = []
        for raw_path in outcome.get("patch_paths") or []:
            patch_path = Path(str(raw_path)).resolve()
            if not patch_path.is_file():
                continue
            patch_record_refs.append(
                put_semantic_record(
                    state_root,
                    kind="initial-patch",
                    record={
                        "source_path": str(patch_path),
                        "sha256": sha256_file(patch_path),
                        "diff_text": patch_path.read_text(
                            encoding="utf-8",
                            errors="replace",
                        ),
                    },
                )
            )
        outcome["patch_record_refs"] = patch_record_refs
        snapshotted_outcomes[str(unit_id)] = outcome
    immutable_snapshot = {
        **deepcopy(outcome_snapshot),
        "unit_outcomes": snapshotted_outcomes,
    }
    record_ref = put_semantic_record(
        state_root,
        kind="initial-unit-outcomes",
        record=immutable_snapshot,
    )
    transition = record_initial_unit_outcomes_snapshot(
        state,
        record_ref=record_ref,
        unit_count=len(snapshotted_outcomes),
        summary=dict(immutable_snapshot.get("summary") or {}),
    )
    state_path = write_semantic_closure_state(state_root, state)
    return {
        "status": "snapshotted",
        "record_ref": record_ref,
        "state_path": str(state_path),
        "unit_count": len(snapshotted_outcomes),
        "summary": deepcopy(immutable_snapshot.get("summary") or {}),
        "transition": transition,
    }
