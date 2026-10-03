#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from driver_backport.core.common import ensure_dir, utc_now
from driver_backport.core.content_store import (
    load_content_record,
    put_content_record,
)


ROUND_MANIFEST_COLLECTIONS = (
    "validation_targets",
    "assessment_targets",
    "applied_units",
    "unit_outcomes",
    "apply_results",
    "validation_batches",
    "scope_changes",
    "semantic_transactions",
)

VALIDATED_REPAIR_ROUND_STATUSES = {
    "validated",
}
PENDING_REPAIR_ROUND_STATUSES = {
    "applying_unit",
    "patch_generation_in_progress",
    "pending_combined_validation",
    "validation_blocked_pending_retry",
}


def append_jsonl_records(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Append complete JSON events with one flush for the whole batch."""

    ensure_dir(path.parent)
    encoded = b"".join(
        (
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for record in records
        if isinstance(record, dict)
    )
    if not encoded:
        return
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def append_event(
    journal_path: Path,
    *,
    case_id: str,
    cycle: int,
    phase: str,
    message: str,
    payload: dict[str, Any] | None = None,
    level: str = "info",
) -> None:
    """Append one durable migration journal event."""

    append_jsonl_records(
        journal_path,
        [{
            "timestamp": utc_now(),
            "case_id": case_id,
            "cycle": cycle,
            "phase": phase,
            "level": level,
            "message": message,
            "payload": payload or {},
        }],
    )


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read complete JSON-object records and ignore a torn trailing append."""

    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    lines = path.read_bytes().splitlines()
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, ValueError):
            if index == len(lines) - 1:
                break
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def round_manifest_event_log_path(
    manifest_path: Path,
    manifest: dict[str, Any] | None = None,
) -> Path:
    configured = str(
        (manifest or {}).get("record_event_log_path") or ""
    ).strip()
    if configured:
        configured_path = Path(configured)
        if configured_path.is_absolute():
            return configured_path.resolve()
        return (manifest_path.parent / configured_path).resolve()
    return (manifest_path.parent / "round_manifest.events.jsonl").resolve()


def append_round_manifest_collections(
    manifest_path: Path,
    collections: dict[str, list[dict[str, Any]]],
    *,
    state_root: Path,
) -> str | None:
    normalized = {
        field: [dict(item) for item in records if isinstance(item, dict)]
        for field, records in collections.items()
        if field in ROUND_MANIFEST_COLLECTIONS and records
    }
    normalized = {
        field: records
        for field, records in normalized.items()
        if records
    }
    if not normalized:
        return None
    path = round_manifest_event_log_path(manifest_path)
    collection_record_refs = {
        field: [
            put_content_record(
                state_root,
                kind="round-manifest-record",
                record=record,
            )
            for record in records
        ]
        for field, records in normalized.items()
    }
    append_jsonl_records(
        path,
        [{
            "schema": "round_manifest_collection_event_v2",
            "recorded_at": utc_now(),
            "collection_record_refs": collection_record_refs,
        }],
    )
    return str(path)


def round_manifest_collections(
    manifest_path: Path,
    manifest: dict[str, Any] | None = None,
    *,
    state_root: Path,
    fields: Iterable[str] = ROUND_MANIFEST_COLLECTIONS,
) -> dict[str, list[dict[str, Any]]]:
    """Load several collections with one event-log scan."""

    if manifest is None:
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )

    selected = tuple(dict.fromkeys(str(field) for field in fields))
    unsupported = [
        field
        for field in selected
        if field not in ROUND_MANIFEST_COLLECTIONS
    ]
    if unsupported:
        raise ValueError(
            "unsupported round manifest collections: "
            + ", ".join(unsupported)
        )
    records = {field: [] for field in selected}
    for event in read_jsonl_records(
        round_manifest_event_log_path(manifest_path, manifest)
    ):
        reference_collections = event.get("collection_record_refs")
        if isinstance(reference_collections, dict):
            for field in selected:
                for reference in reference_collections.get(field) or []:
                    record = load_content_record(
                        state_root,
                        str(reference),
                    )
                    if not isinstance(record, dict):
                        raise ValueError(
                            "round manifest content record is not an "
                            f"object: {reference}"
                        )
                    records[field].append(record)
        collections = event.get("collections")
        if not isinstance(collections, dict):
            continue
        for field in selected:
            records[field].extend(
                dict(item)
                for item in collections.get(field) or []
                if isinstance(item, dict)
            )
    return records


def pending_repair_round_manifests(
    state_root: Path,
) -> list[tuple[Path, dict[str, Any]]]:
    """Return durable repair rounds that stopped before combined validation."""

    pending: list[tuple[int, str, Path, dict[str, Any]]] = []
    for manifest_path in state_root.glob(
        "cycles/cycle-*/agentic_repair_round_*/round_manifest.json"
    ):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            modified = manifest_path.stat().st_mtime_ns
        except (OSError, ValueError):
            continue
        if str(manifest.get("status") or "") not in (
            PENDING_REPAIR_ROUND_STATUSES
        ):
            continue
        if not (
            manifest.get("inflight_validation_target_path")
            or int(manifest.get("validation_target_count") or 0)
        ):
            continue
        pending.append(
            (
                modified,
                str(manifest_path),
                manifest_path.resolve(),
                manifest,
            )
        )
    return [
        (manifest_path, manifest)
        for _modified, _key, manifest_path, manifest in sorted(pending)
    ]


def validated_repair_round_counts(state_root: Path) -> dict[str, int]:
    """Project durable round manifests onto phase-specific budgets."""

    counts = {"compiler_repair": 0, "semantic_closure": 0}
    for _manifest_path, manifest in validated_repair_round_manifests(
        state_root
    ):
        phase = str(
            manifest.get("execution_phase") or "compiler_repair"
        )
        if phase not in counts:
            raise ValueError(
                "validated repair-round manifest has an unsupported "
                f"execution phase: {phase}"
            )
        counts[phase] += 1
    return {
        "compile_repair_rounds_completed": counts["compiler_repair"],
        "semantic_repair_rounds_completed": counts["semantic_closure"],
        "total_repair_rounds_completed": sum(counts.values()),
    }


def validated_repair_round_manifests(
    state_root: Path,
    *,
    execution_phase: str | None = None,
) -> list[tuple[Path, dict[str, Any]]]:
    """Return validated manifests in durable cycle/round order."""

    records: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(
        state_root.glob(
            "cycles/cycle-*/agentic_repair_round_*/round_manifest.json"
        )
    ):
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            continue
        if str(manifest.get("status") or "") not in (
            VALIDATED_REPAIR_ROUND_STATUSES
        ):
            continue
        if manifest.get("repair_budget_consumed") is False:
            continue
        phase = str(
            manifest.get("execution_phase") or "compiler_repair"
        )
        if execution_phase is not None and phase != execution_phase:
            continue
        records.append((manifest_path.resolve(), manifest))
    return records


def validated_repair_round_count(state_root: Path) -> int:
    return validated_repair_round_counts(state_root)[
        "total_repair_rounds_completed"
    ]
