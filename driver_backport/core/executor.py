#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import ensure_dir, normalize_relpath, utc_now, write_json


MAX_TRANSACTION_ATTEMPTS = 999


@dataclass
class Transaction:
    case_id: str
    cycle: int
    work_unit_id: str
    root: Path
    manifest_path: Path

    @property
    def backups_dir(self) -> Path:
        return self.root / "backups"

    def manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            "case_id": self.case_id,
            "cycle": self.cycle,
            "work_unit_id": self.work_unit_id,
            "created_at": utc_now(),
            "status": "open",
            "entries": [],
        }

    def save_manifest(self, payload: dict[str, Any]) -> None:
        write_json(self.manifest_path, payload)


def ensure_edit_allowed(spec: CaseSpec, relative_path: str) -> Path:
    normalized = normalize_relpath(relative_path)
    target = spec.resolve_old(normalized)
    classification = spec.classify_edit_root(normalized)
    if classification == "forbidden":
        raise PermissionError(f"path is outside allowed edit roots: {normalized}")
    return target


def classify_edit_permission(spec: CaseSpec, relative_path: str) -> str:
    return spec.classify_edit_root(relative_path)


def patch_strip_level(diff_text: str) -> int:
    for line in diff_text.splitlines():
        if line.startswith(("+++ ", "--- ")):
            path = line[4:].strip().split("\t", 1)[0]
            if path.startswith(("a/", "b/")):
                return 1
    return 0


def run_unified_diff(
    root: Path,
    diff_text: str,
    *,
    dry_run: bool = False,
    reverse: bool = False,
) -> tuple[int, str, str]:
    """Check or apply one complete diff with exact-context Git semantics."""

    command = [
        "git",
        "-C",
        str(root.resolve()),
        "apply",
        "--whitespace=nowarn",
        f"-p{patch_strip_level(diff_text)}",
    ]
    if dry_run:
        command.append("--check")
    if reverse:
        command.append("--reverse")
    process = subprocess.run(
        command,
        input=diff_text,
        capture_output=True,
        text=True,
    )
    return process.returncode, process.stdout, process.stderr


def start_transaction(state_root: Path, *, case_id: str, cycle: int, work_unit_id: str) -> Transaction:
    transaction_parent = ensure_dir(state_root / "transactions" / f"cycle-{cycle:04d}")
    transaction_root = transaction_parent / work_unit_id
    manifest_path = transaction_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(existing.get("status") or "open") != "open":
            for index in range(1, MAX_TRANSACTION_ATTEMPTS + 1):
                candidate = transaction_parent / f"{work_unit_id}-attempt-{index:03d}"
                if not (candidate / "manifest.json").exists():
                    transaction_root = candidate
                    break
            else:
                raise RuntimeError(
                    "transaction attempt namespace exhausted for "
                    f"cycle={cycle} work_unit={work_unit_id}"
                )
    transaction_root = ensure_dir(transaction_root)
    manifest_path = transaction_root / "manifest.json"
    transaction = Transaction(
        case_id=case_id,
        cycle=cycle,
        work_unit_id=work_unit_id,
        root=transaction_root,
        manifest_path=manifest_path,
    )
    payload = transaction.manifest()
    transaction.save_manifest(payload)
    return transaction


def backup_file(transaction: Transaction, spec: CaseSpec, relative_path: str) -> Path:
    target = ensure_edit_allowed(spec, relative_path)
    backup_path = transaction.backups_dir / normalize_relpath(relative_path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    if target.exists():
        backup_path.write_bytes(target.read_bytes())

    payload = transaction.manifest()
    payload["entries"].append(
        {
            "type": "backup",
            "relative_path": normalize_relpath(relative_path),
            "target": str(target),
            "backup": str(backup_path.resolve()),
            "existed": existed,
        }
    )
    transaction.save_manifest(payload)
    return backup_path


def backup_file_once(
    transaction: Transaction,
    spec: CaseSpec,
    relative_path: str,
) -> Path:
    """Preserve the first preimage of a file for a multi-patch transaction."""

    normalized = normalize_relpath(relative_path)
    payload = transaction.manifest()
    for entry in payload.get("entries") or []:
        if (
            str(entry.get("type") or "") == "backup"
            and normalize_relpath(
                str(entry.get("relative_path") or "")
            )
            == normalized
        ):
            backup = Path(str(entry.get("backup") or "")).resolve()
            if bool(entry.get("existed")) and not backup.is_file():
                raise FileNotFoundError(
                    "transaction preimage is missing for "
                    f"{normalized}: {backup}"
                )
            return backup
    return backup_file(transaction, spec, normalized)


def finalize_transaction(transaction: Transaction, *, status: str, notes: list[str] | None = None) -> None:
    payload = transaction.manifest()
    payload["status"] = status
    payload["closed_at"] = utc_now()
    if notes:
        payload["notes"] = list(notes)
    transaction.save_manifest(payload)


def load_transaction(transaction_root: Path) -> Transaction:
    manifest_path = transaction_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"transaction manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Transaction(
        case_id=str(payload["case_id"]),
        cycle=int(payload.get("cycle") or 0),
        work_unit_id=str(payload.get("work_unit_id") or transaction_root.name),
        root=transaction_root.resolve(),
        manifest_path=manifest_path.resolve(),
    )


def rollback_transaction(spec: CaseSpec, transaction_root: Path) -> dict[str, Any]:
    transaction = load_transaction(transaction_root)
    payload = transaction.manifest()
    restored: list[dict[str, Any]] = []
    seen: set[str] = set()

    backup_entries = [
        entry for entry in list(payload.get("entries") or [])
        if str(entry.get("type") or "") == "backup"
    ]
    for entry in reversed(backup_entries):
        relative_path = normalize_relpath(str(entry["relative_path"]))
        if relative_path in seen:
            continue
        seen.add(relative_path)

        target = spec.resolve_old(relative_path)
        backup = Path(str(entry["backup"])).resolve()
        existed = bool(entry.get("existed", backup.exists()))
        target.parent.mkdir(parents=True, exist_ok=True)

        if backup.exists():
            target.write_bytes(backup.read_bytes())
            restored.append(
                {
                    "file": relative_path,
                    "status": "restored_from_backup",
                    "target": str(target),
                }
            )
            continue

        if not existed and target.exists():
            os.remove(target)
            restored.append(
                {
                    "file": relative_path,
                    "status": "removed_created_file",
                    "target": str(target),
                }
            )
            continue

        restored.append(
            {
                "file": relative_path,
                "status": "no_backup_available",
                "target": str(target),
            }
        )

    payload["rollback"] = {
        "rolled_back_at": utc_now(),
        "restored_count": sum(1 for item in restored if item["status"] in {"restored_from_backup", "removed_created_file"}),
        "results": restored,
    }
    payload["status"] = "rolled_back"
    transaction.save_manifest(payload)
    return {
        "case_id": transaction.case_id,
        "cycle": transaction.cycle,
        "work_unit_id": transaction.work_unit_id,
        "transaction_dir": str(transaction.root.resolve()),
        "restored_count": payload["rollback"]["restored_count"],
        "results": restored,
    }
