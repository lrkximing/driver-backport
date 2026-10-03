#!/usr/bin/env python3
from __future__ import annotations

import shutil
import re
from pathlib import Path
from typing import Any

from driver_backport.analysis.kbuild_index import KbuildIndex
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    cycle_hint_from_path,
    ordered_unique_strings as _ordered_unique,
    utc_now,
)
from driver_backport.core.executor import backup_file, finalize_transaction, rollback_transaction, start_transaction
from driver_backport.validation.kconfig_sync import (
    available_kconfig_symbols,
    clear_kconfig_caches,
    dependency_expr_requirements,
    kconfig_index,
)


_MAKEFILE_NAMES = {"Makefile", "Kbuild"}
_KCONFIG_BLOCK_START_RE = re.compile(r"^(config|menuconfig)\s+([A-Za-z0-9_]+)\b")
_KCONFIG_TOP_RE = re.compile(
    r"^(config|menuconfig|choice|endchoice|menu|endmenu|if|endif|comment|source|rsource|osource|orsource|mainmenu)\b"
)
_KCONFIG_SOURCE_RE = re.compile(r'^\s*(?:source|rsource|osource|orsource)\s+"?([^"\s]+)"?')
_BUILD_KCONFIG_ASSIGNMENT_RE = re.compile(r"^(?:CONFIG_)?([A-Z0-9_]+)\s*=\s*([ym])\b")


def _entry_reasons(spec: CaseSpec, relative_path: str) -> list[str]:
    return ["entry_file"] if relative_path in spec.entry_files else []


def build_initial_transplant_plan(spec: CaseSpec, frontier: dict) -> dict:
    donor_root = spec.primary_donor_root
    old_root = spec.old_root_path

    copy_operations: list[dict] = []
    for relative_path in frontier.get("initial_copy_set") or []:
        donor_path = spec.resolve_donor(relative_path)
        old_path = spec.resolve_old(relative_path)
        action = "copy_missing" if not old_path.exists() else "overwrite_from_donor"
        copy_operations.append(
            {
                "file": relative_path,
                "action": action,
                "donor_path": str(donor_path),
                "old_path": str(old_path),
                "reasons": _entry_reasons(spec, relative_path),
            }
        )

    build_integration: list[dict] = []
    related_source_files = _ordered_unique(
        str(item).strip()
        for item in (
            list(frontier.get("initial_copy_set") or [])
            + list(spec.entry_files)
        )
    )
    for relative_path in frontier.get("build_integration_set") or []:
        build_integration.append(
            {
                "file": relative_path,
                "old_path": str(spec.resolve_old(relative_path)),
                "donor_path": str(spec.resolve_donor(relative_path)),
                "action": "merge_build_integration",
                "strategy": "structural_build_merge",
                "related_source_files": related_source_files,
            }
        )

    shared_candidates: list[dict] = []
    for relative_path in frontier.get("initial_shared_candidates") or []:
        shared_candidates.append(
            {
                "file": relative_path,
                "old_path": str(spec.resolve_old(relative_path)),
                "donor_path": str(spec.resolve_donor(relative_path)),
                "action": "retain_as_graph_scope_seed",
                "reasons": _entry_reasons(spec, relative_path),
            }
        )

    return {
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "old_root": str(old_root),
        "donor_root": str(donor_root),
        "entry_files": list(spec.entry_files),
        "copy_operations": copy_operations,
        "build_integration": build_integration,
        "shared_candidates": shared_candidates,
        "summary": {
            "copy_operations": len(copy_operations),
            "build_integration_files": len(build_integration),
            "shared_candidates": len(shared_candidates),
        },
    }
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _root_relative(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def _explicit_build_kconfig_symbols(assignments: list[str]) -> set[str]:
    symbols: set[str] = set()
    for assignment in assignments:
        match = _BUILD_KCONFIG_ASSIGNMENT_RE.match(str(assignment).strip())
        if not match:
            continue
        symbols.add(match.group(1))
    return symbols


def _available_kconfig_symbols(root: Path) -> set[str]:
    clear_kconfig_caches()
    return set(available_kconfig_symbols(root))


def _extract_related_makefile_lines(
    donor_text: str,
    old_text: str,
    related_files: list[str],
    *,
    build_directory: str,
    build_file: str,
) -> tuple[list[str], set[str], dict[str, Any]]:
    donor_index = KbuildIndex.from_text(
        donor_text,
        directory=build_directory,
        file=build_file,
    )
    old_index = KbuildIndex.from_text(
        old_text,
        directory=build_directory,
        file=build_file,
    )
    closure = donor_index.assignment_closure_for_sources(related_files)
    old_items_by_owner: dict[tuple[str, str], set[str]] = {}
    for record in old_index.assignments:
        if record.relation in {"config_membership", "fixed_membership"}:
            owner = ("top_membership", "")
        elif record.relation == "module_membership" and record.module:
            owner = ("module_membership", record.module)
        else:
            owner = ("lhs", record.lhs)
        old_items_by_owner.setdefault(owner, set()).update(record.object_items)

    selected_lines: list[str] = []
    for record_payload in closure["assignments"]:
        lhs = str(record_payload.get("lhs") or "")
        relation = str(record_payload.get("relation") or "")
        module = str(record_payload.get("module") or "")
        if relation in {"config_membership", "fixed_membership"}:
            owner = ("top_membership", "")
        elif relation == "module_membership" and module:
            owner = ("module_membership", module)
        else:
            owner = ("lhs", lhs)
        donor_items = [str(item) for item in record_payload.get("object_items") or []]
        missing_items = [
            item
            for item in donor_items
            if item not in old_items_by_owner.get(owner, set())
        ]
        if not lhs or not missing_items:
            continue
        selected_lines.append(f"{lhs} += {' '.join(missing_items)}")
        old_items_by_owner.setdefault(owner, set()).update(missing_items)

    config_symbols = {
        str(config).removeprefix("CONFIG_")
        for config in closure.get("configs") or []
    }
    return selected_lines, config_symbols, closure


def _merge_makefile_lines(
    old_text: str,
    donor_text: str,
    related_files: list[str],
    *,
    build_directory: str,
    build_file: str,
) -> tuple[str, dict[str, Any]]:
    extras, config_symbols, closure = _extract_related_makefile_lines(
        donor_text,
        old_text,
        related_files,
        build_directory=build_directory,
        build_file=build_file,
    )

    if not extras:
        merged_text = old_text if old_text.endswith("\n") or not old_text else old_text + "\n"
        return merged_text, {
            "merge_action": "unchanged",
            "added_lines": 0,
            "target_config_symbols": sorted(config_symbols),
            "selection_backend": "kbuild_assignment_closure_v1",
            "kbuild_closure": closure,
        }

    merged_lines = list(old_text.splitlines())
    if merged_lines and merged_lines[-1].strip() and extras[0].strip():
        merged_lines.append("")
    merged_lines.extend(extras)
    return "\n".join(merged_lines).rstrip("\n") + "\n", {
        "merge_action": "merged_related_makefile_lines",
        "added_lines": sum(1 for line in extras if line.strip()),
        "target_config_symbols": sorted(config_symbols),
        "selection_backend": "kbuild_assignment_closure_v1",
        "kbuild_closure": closure,
    }


def _kconfig_blocks(lines: list[str]) -> list[dict[str, Any]]:
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = _KCONFIG_BLOCK_START_RE.match(line)
        if match:
            starts.append((index, match.group(2)))

    blocks: list[dict[str, Any]] = []
    for offset, (start, symbol) in enumerate(starts):
        end = len(lines)
        scan = start + 1
        while scan < len(lines):
            if _KCONFIG_TOP_RE.match(lines[scan]):
                end = scan
                break
            scan += 1
        blocks.append(
            {
                "symbol": symbol,
                "start": start,
                "end": end,
                "lines": lines[start:end],
                "order": offset,
            }
        )
    return blocks


def _extract_makefile_config_symbols_for_kconfig(
    donor_kconfig_path: Path,
    related_files: list[str],
    *,
    donor_root: Path,
) -> set[str]:
    directory = donor_kconfig_path.parent.resolve().relative_to(donor_root.resolve()).as_posix()
    closure = KbuildIndex.from_directory(
        donor_root,
        directory,
    ).assignment_closure_for_sources(related_files)
    return {
        str(config).removeprefix("CONFIG_")
        for config in closure.get("configs") or []
    }


def _extract_related_kconfig_source_lines(
    donor_lines: list[str],
    old_lines: list[str],
    related_files: list[str],
    *,
    donor_root: Path,
    target_symbols: list[str],
) -> list[str]:
    old_existing = {line.rstrip() for line in old_lines if line.strip()}
    related_paths = [Path(str(item)) for item in related_files if str(item).strip()]
    ancestor_kconfigs = {
        (parent / "Kconfig").as_posix()
        for related_path in related_paths
        for parent in [related_path.parent, *related_path.parents]
        if parent.as_posix() not in {"", "."}
    }
    selected: list[str] = []
    for line in donor_lines:
        match = _KCONFIG_SOURCE_RE.match(line)
        if not match:
            continue
        source_path = Path(match.group(1))
        source_relative = source_path.as_posix()
        if "$" in source_relative:
            continue
        contains_target_symbol = False
        resolved_source = donor_root / source_path
        if target_symbols and resolved_source.is_file():
            source_symbols = {
                str(block["symbol"])
                for block in _kconfig_blocks(_read_text(resolved_source).splitlines())
            }
            contains_target_symbol = bool(source_symbols & set(target_symbols))
        if source_relative not in ancestor_kconfigs and not contains_target_symbol:
            continue
        normalized = line.rstrip()
        if normalized.strip() and normalized not in old_existing:
            old_existing.add(normalized)
            selected.append(normalized)
    return selected


def _target_kconfig_symbols(
    donor_kconfig_path: Path,
    donor_blocks: list[dict[str, Any]],
    old_symbols: set[str],
    related_files: list[str],
    *,
    donor_root: Path,
    explicit_configs: set[str],
) -> list[str]:
    donor_by_symbol = {str(block["symbol"]): block for block in donor_blocks}
    targets = set(explicit_configs)
    targets.update(
        _extract_makefile_config_symbols_for_kconfig(
            donor_kconfig_path,
            related_files,
            donor_root=donor_root,
        )
    )
    index = kconfig_index((str(donor_root.resolve()),))

    queue = list(targets)
    selected: set[str] = set()
    while queue:
        symbol = queue.pop(0)
        if symbol not in donor_by_symbol or symbol in selected:
            continue
        selected.add(symbol)
        if symbol in old_symbols:
            continue
        entry = index.get(symbol) or {}
        refs = {str(item) for item in entry.get("selects") or []}
        for expr in [
            *list(entry.get("visibility_depends") or []),
            *list(entry.get("depends") or []),
        ]:
            refs.update(
                dependency_expr_requirements(str(expr))["required_symbols"]
            )
        for ref in sorted(refs):
            if ref in donor_by_symbol and ref not in selected and ref not in old_symbols:
                queue.append(ref)

    return [
        str(block["symbol"])
        for block in donor_blocks
        if str(block["symbol"]) in selected and str(block["symbol"]) not in old_symbols
    ]


def _kconfig_insert_index(lines: list[str]) -> int:
    index = len(lines)
    while index > 0 and not lines[index - 1].strip():
        index -= 1

    trailer_start = index
    while trailer_start > 0 and re.match(r"^(endif|endmenu)\b", lines[trailer_start - 1]):
        trailer_start -= 1
        while trailer_start > 0 and not lines[trailer_start - 1].strip():
            trailer_start -= 1

    return trailer_start if trailer_start != index else index


def _merge_kconfig_blocks(
    old_text: str,
    donor_text: str,
    donor_path: Path,
    related_files: list[str],
    *,
    donor_root: Path,
    explicit_configs: set[str],
) -> tuple[str, dict[str, Any]]:
    old_lines = old_text.splitlines()
    donor_lines = donor_text.splitlines()
    old_blocks = _kconfig_blocks(old_lines)
    donor_blocks = _kconfig_blocks(donor_lines)
    old_symbols = {str(block["symbol"]) for block in old_blocks}
    donor_by_symbol = {str(block["symbol"]): block for block in donor_blocks}
    target_symbols = _target_kconfig_symbols(
        donor_path,
        donor_blocks,
        old_symbols,
        related_files,
        donor_root=donor_root,
        explicit_configs=explicit_configs,
    )
    source_lines = _extract_related_kconfig_source_lines(
        donor_lines,
        old_lines,
        related_files,
        donor_root=donor_root,
        target_symbols=target_symbols,
    )

    if not target_symbols and not source_lines:
        merged_text = old_text if old_text.endswith("\n") or not old_text else old_text + "\n"
        return merged_text, {
            "merge_action": "unchanged",
            "added_blocks": 0,
            "added_lines": 0,
            "target_config_symbols": [],
            "selection_backend": "canonical_kconfig_and_kbuild_closure_v1",
        }

    insert_blocks: list[str] = []
    insert_blocks.extend(source_lines)
    for symbol in target_symbols:
        block = donor_by_symbol[symbol]
        block_text = "\n".join(str(line).rstrip() for line in block["lines"]).rstrip()
        if block_text:
            insert_blocks.append(block_text)

    insert_index = _kconfig_insert_index(old_lines)
    before = old_lines[:insert_index]
    after = old_lines[insert_index:]
    merged_lines = list(before)
    if merged_lines and merged_lines[-1].strip():
        merged_lines.append("")
    merged_lines.extend("\n\n".join(insert_blocks).splitlines())
    if after:
        if merged_lines and merged_lines[-1].strip():
            merged_lines.append("")
        merged_lines.extend(after)

    return "\n".join(merged_lines).rstrip("\n") + "\n", {
        "merge_action": "merged_missing_kconfig_blocks",
        "added_blocks": len(target_symbols),
        "added_lines": len(source_lines),
        "target_config_symbols": target_symbols,
        "selection_backend": "canonical_kconfig_and_kbuild_closure_v1",
    }


def _merge_unique_build_lines(old_text: str, donor_text: str) -> tuple[str, dict[str, Any]]:
    old_lines = old_text.splitlines()
    donor_lines = donor_text.splitlines()
    existing_nonblank = {line.rstrip() for line in old_lines if line.strip()}
    extras: list[str] = []

    for line in donor_lines:
        normalized = line.rstrip()
        if not normalized.strip():
            if extras and extras[-1] != "":
                extras.append("")
            continue
        if normalized not in existing_nonblank:
            existing_nonblank.add(normalized)
            extras.append(normalized)

    while extras and extras[-1] == "":
        extras.pop()

    if not extras:
        merged_text = old_text if old_text.endswith("\n") or not old_text else old_text + "\n"
        return merged_text, {"merge_action": "unchanged", "added_lines": 0}

    merged_lines = list(old_lines)
    if merged_lines and merged_lines[-1].strip() and extras[0].strip():
        merged_lines.append("")
    merged_lines.extend(extras)
    return "\n".join(merged_lines).rstrip("\n") + "\n", {
        "merge_action": "merged_missing_lines",
        "added_lines": sum(1 for line in extras if line.strip()),
    }


def _apply_build_integration(
    spec: CaseSpec,
    transaction,
    op: dict[str, Any],
    *,
    allow_escalation: bool,
) -> dict[str, Any]:
    relative_path = str(op["file"])
    donor_path = Path(str(op["donor_path"])).resolve()
    target_path = spec.resolve_old(relative_path)

    if not donor_path.exists():
        return {
            "kind": "build_integration",
            "file": relative_path,
            "status": "missing_donor",
            "donor_path": str(donor_path),
        }

    merged_text: str | None = None
    merge_meta: dict[str, Any] | None = None
    if target_path.exists():
        old_text = _read_text(target_path)
        donor_text = _read_text(donor_path)
        related_files = [
            str(item)
            for item in op.get("related_source_files") or list(spec.entry_files)
            if str(item).strip()
        ]
        if donor_path.name.startswith("Kconfig"):
            merged_text, merge_meta = _merge_kconfig_blocks(
                old_text,
                donor_text,
                donor_path,
                related_files,
                donor_root=spec.primary_donor_root,
                explicit_configs=_explicit_build_kconfig_symbols(spec.build_kconfig),
            )
        elif donor_path.name in _MAKEFILE_NAMES or donor_path.suffix == ".mk":
            merged_text, merge_meta = _merge_makefile_lines(
                old_text,
                donor_text,
                related_files,
                build_directory=Path(relative_path).parent.as_posix(),
                build_file=relative_path,
            )
        else:
            merged_text, merge_meta = _merge_unique_build_lines(old_text, donor_text)
        if merge_meta["merge_action"] == "unchanged":
            return {
                "kind": "build_integration",
                "file": relative_path,
                "status": "already_integrated",
                "strategy": str(op.get("strategy") or ""),
                "donor_path": str(donor_path),
                "target_path": str(target_path),
                "added_lines": 0,
                "added_blocks": 0,
            }

    permission = spec.classify_edit_root(relative_path)
    if permission not in {"editable", "escalation"} or (
        permission == "escalation" and not allow_escalation
    ):
        return {
            "kind": "build_integration",
            "file": relative_path,
            "status": "blocked_by_edit_policy",
            "permission": permission,
        }

    backup_file(transaction, spec, relative_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if merged_text is None or merge_meta is None:
        shutil.copyfile(donor_path, target_path)
        return {
            "kind": "build_integration",
            "file": relative_path,
            "status": "copied_build_file",
            "strategy": str(op.get("strategy") or ""),
            "donor_path": str(donor_path),
            "target_path": str(target_path),
        }

    target_path.write_text(merged_text, encoding="utf-8")
    return {
        "kind": "build_integration",
        "file": relative_path,
        "status": merge_meta["merge_action"],
        "strategy": str(op.get("strategy") or ""),
        "donor_path": str(donor_path),
        "target_path": str(target_path),
        "added_lines": merge_meta.get("added_lines", 0),
        "added_blocks": merge_meta.get("added_blocks", 0),
        "target_config_symbols": merge_meta.get("target_config_symbols", []),
        "selection_backend": merge_meta.get("selection_backend"),
        "kbuild_closure": merge_meta.get("kbuild_closure"),
    }


def _apply_build_integration_operations(
    spec: CaseSpec,
    transaction,
    operations: list[dict[str, Any]],
    *,
    allow_escalation: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for operation in operations:
        try:
            result = _apply_build_integration(
                spec,
                transaction,
                operation,
                allow_escalation=allow_escalation,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            result = {
                "kind": "build_integration",
                "file": str(operation.get("file") or ""),
                "status": "build_merge_failed",
                "error": str(exc),
            }
        results.append(result)
    return results


def _build_integration_result_counts(
    results: list[dict[str, Any]],
) -> tuple[int, int, int]:
    blocked = sum(
        result.get("status") == "blocked_by_edit_policy"
        for result in results
    )
    failed = sum(
        result.get("status") in {"missing_donor", "build_merge_failed"}
        for result in results
    )
    return len(results) - blocked - failed, blocked, failed


def _find_donor_kconfig_symbol_files(spec: CaseSpec, missing_symbols: set[str]) -> list[tuple[Path, Path]]:
    matches: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for root_text in spec.donor_root_paths:
        donor_root = Path(root_text).resolve()
        if not donor_root.exists():
            continue
        index = kconfig_index((str(donor_root),))
        paths = {
            Path(path).resolve()
            for symbol in missing_symbols
            for path in (index.get(symbol) or {}).get("defined_in") or []
        }
        for path in sorted(paths):
            relative = _root_relative(path, donor_root)
            if not relative or relative in seen:
                continue
            seen.add(relative)
            matches.append((donor_root, path))
    return matches


def _donor_kconfig_source_link_files(donor_root: Path, target_relative: str) -> list[Path]:
    target = target_relative.rstrip("/")
    matches: list[Path] = []
    for path in sorted(donor_root.rglob("Kconfig*")):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.as_posix().endswith("/" + target):
            continue
        for line in _read_text(path).splitlines():
            match = _KCONFIG_SOURCE_RE.match(line)
            if match and match.group(1).rstrip("/") == target:
                matches.append(path)
                break
    return matches


def _ancestor_makefiles_for_build_directory(donor_root: Path, directory_relative: str) -> list[Path]:
    directory = Path(directory_relative)
    matches: list[Path] = []
    parent = directory.parent
    while str(parent) not in {"", "."}:
        for name in sorted(_MAKEFILE_NAMES):
            path = donor_root / parent / name
            if path.exists():
                matches.append(path)
        parent = parent.parent
    return matches


def _dedupe_build_ops(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for op in ops:
        file_name = str(op.get("file") or "")
        if not file_name or file_name in seen:
            continue
        seen.add(file_name)
        result.append(op)
    return result


def _supplement_missing_explicit_kconfig_symbols(
    spec: CaseSpec,
    transaction,
    *,
    missing_symbols: set[str],
    related_files: list[str],
) -> dict[str, Any]:
    donor_symbol_files = _find_donor_kconfig_symbol_files(spec, missing_symbols)
    ops: list[dict[str, Any]] = []

    for donor_root, donor_kconfig in donor_symbol_files:
        relative = _root_relative(donor_kconfig, donor_root)
        if not relative:
            continue
        donor_dir = Path(relative).parent.as_posix()
        ops.append(
            {
                "file": relative,
                "donor_path": str(donor_kconfig),
                "strategy": "explicit_kconfig_symbol_recovery",
                "related_source_files": related_files,
            }
        )

        for name in sorted(_MAKEFILE_NAMES):
            makefile = donor_kconfig.parent / name
            if makefile.exists():
                makefile_relative = _root_relative(makefile, donor_root)
                if makefile_relative:
                    ops.append(
                        {
                            "file": makefile_relative,
                            "donor_path": str(makefile),
                            "strategy": "explicit_kconfig_symbol_recovery",
                            "related_source_files": related_files,
                        }
                    )

        for link_file in _donor_kconfig_source_link_files(donor_root, relative):
            link_relative = _root_relative(link_file, donor_root)
            if link_relative:
                ops.append(
                    {
                        "file": link_relative,
                        "donor_path": str(link_file),
                        "strategy": "explicit_kconfig_symbol_recovery",
                        "related_source_files": related_files,
                    }
                )

        if donor_dir and donor_dir != ".":
            for makefile in _ancestor_makefiles_for_build_directory(donor_root, donor_dir):
                makefile_relative = _root_relative(makefile, donor_root)
                if makefile_relative:
                    ops.append(
                        {
                            "file": makefile_relative,
                            "donor_path": str(makefile),
                            "strategy": "explicit_kconfig_symbol_recovery",
                            "related_source_files": related_files,
                        }
                    )

    results: list[dict[str, Any]] = []
    for op in _dedupe_build_ops(ops):
        results.append(_apply_build_integration(spec, transaction, op))

    available_after = _available_kconfig_symbols(spec.old_root_path)
    missing_after = sorted(symbol for symbol in missing_symbols if symbol not in available_after)
    return {
        "kind": "explicit_kconfig_symbol_recovery",
        "status": "complete" if not missing_after else "missing_after_recovery",
        "requested_symbols": sorted(missing_symbols),
        "missing_after": missing_after,
        "donor_symbol_files": [
            str(path.resolve())
            for _, path in donor_symbol_files
        ],
        "results": results,
    }


def _verify_and_recover_explicit_kconfig_symbols(
    spec: CaseSpec,
    transaction,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    requested_symbols = _explicit_build_kconfig_symbols(spec.build_kconfig)
    if not requested_symbols:
        return {
            "kind": "explicit_kconfig_symbol_verification",
            "status": "skipped",
            "requested_symbols": [],
            "missing_before_recovery": [],
            "missing_after_recovery": [],
            "recovery": None,
        }

    available_before = _available_kconfig_symbols(spec.old_root_path)
    missing_before = sorted(symbol for symbol in requested_symbols if symbol not in available_before)
    if not missing_before:
        return {
            "kind": "explicit_kconfig_symbol_verification",
            "status": "complete",
            "requested_symbols": sorted(requested_symbols),
            "missing_before_recovery": [],
            "missing_after_recovery": [],
            "recovery": None,
        }

    related_files = _ordered_unique(
        str(item).strip()
        for item in (
            list(plan.get("initial_copy_set") or [])
            + [
                str(op.get("file") or "")
                for op in plan.get("copy_operations") or []
            ]
            + list(spec.entry_files)
        )
    )
    recovery = _supplement_missing_explicit_kconfig_symbols(
        spec,
        transaction,
        missing_symbols=set(missing_before),
        related_files=related_files,
    )
    missing_after = list(recovery.get("missing_after") or [])
    return {
        "kind": "explicit_kconfig_symbol_verification",
        "status": "complete" if not missing_after else "missing_after_recovery",
        "requested_symbols": sorted(requested_symbols),
        "missing_before_recovery": missing_before,
        "missing_after_recovery": missing_after,
        "recovery": recovery,
    }


def apply_initial_transplant_plan(
    spec: CaseSpec,
    plan: dict[str, Any],
    *,
    state_root: Path,
    plan_path: Path | None = None,
    transaction_work_unit_id: str = "initial-transplant",
    transaction_cycle: int | None = None,
    allow_escalation: bool = True,
) -> dict[str, Any]:
    cycle_hint = (
        int(transaction_cycle)
        if transaction_cycle is not None
        else cycle_hint_from_path(plan_path)
    )
    transaction = start_transaction(
        state_root,
        case_id=spec.case_id,
        cycle=cycle_hint,
        work_unit_id=transaction_work_unit_id,
    )

    results: list[dict[str, Any]] = []
    applied_count = 0
    blocked_count = 0
    failed_count = 0
    build_applied_count = 0
    build_blocked_count = 0
    build_failed_count = 0

    for op in plan.get("copy_operations") or []:
        relative_path = str(op["file"])
        donor_path = Path(str(op["donor_path"])).resolve()
        target_path = spec.resolve_old(relative_path)
        permission = spec.classify_edit_root(relative_path)

        if permission != "editable":
            blocked_count += 1
            results.append(
                {
                    "file": relative_path,
                    "status": "blocked_by_edit_policy",
                    "permission": permission,
                }
            )
            continue

        if not donor_path.exists():
            failed_count += 1
            results.append(
                {
                    "file": relative_path,
                    "status": "missing_donor",
                    "donor_path": str(donor_path),
                }
            )
            continue

        try:
            backup_file(transaction, spec, relative_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(donor_path, target_path)
        except Exception as exc:  # pragma: no cover - defensive path
            failed_count += 1
            results.append(
                {
                    "file": relative_path,
                    "status": "copy_failed",
                    "error": str(exc),
                }
            )
            continue

        applied_count += 1
        results.append(
            {
                "file": relative_path,
                "status": "copied",
                "action": op.get("action"),
                "donor_path": str(donor_path),
                "target_path": str(target_path),
            }
        )

    build_results = _apply_build_integration_operations(
        spec,
        transaction,
        list(plan.get("build_integration") or []),
        allow_escalation=allow_escalation,
    )
    results.extend(build_results)

    kconfig_verification = _verify_and_recover_explicit_kconfig_symbols(
        spec,
        transaction,
        plan=plan,
    )
    recovery = kconfig_verification.get("recovery") or {}
    recovery_results = list(recovery.get("results") or [])
    results.extend(recovery_results)
    results.append(kconfig_verification)
    (
        build_applied_count,
        build_blocked_count,
        build_failed_count,
    ) = _build_integration_result_counts(
        [*build_results, *recovery_results]
    )
    if kconfig_verification["status"] == "missing_after_recovery":
        build_failed_count += 1

    transaction_status = "applied"
    if failed_count or build_failed_count:
        transaction_status = "failed"
    elif blocked_count or build_blocked_count:
        transaction_status = "partial"
    finalize_transaction(transaction, status=transaction_status)
    rollback = None
    if transaction_status != "applied":
        rollback = rollback_transaction(spec, transaction.root)

    return {
        "case_id": spec.case_id,
        "applied_copy_operations": applied_count,
        "blocked_copy_operations": blocked_count,
        "failed_copy_operations": failed_count,
        "applied_build_integrations": build_applied_count,
        "blocked_build_integrations": build_blocked_count,
        "failed_build_integrations": build_failed_count,
        "shared_candidates_pending": len(plan.get("shared_candidates") or []),
        "explicit_kconfig_symbol_verification": kconfig_verification,
        "transaction_dir": str(transaction.root.resolve()),
        "rollback": rollback,
        "results": results,
    }


def verify_transplant_prerequisites(
    spec: CaseSpec,
    plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verify transplant prerequisites without changing the target tree.

    Initial Transplant and A-stage preparation own source-copy and build-file
    mutations.  Validation, compiler repair, and semantic recovery may only
    observe whether those prerequisites still hold.
    """

    plan = dict(plan or {})
    copy_ops = list(plan.get("copy_operations") or [])
    build_ops = list(plan.get("build_integration") or [])
    requested_symbols = sorted(
        _explicit_build_kconfig_symbols(spec.build_kconfig)
    )
    if not copy_ops and not build_ops and not requested_symbols:
        return {
            "schema": "transplant_prerequisite_verification_v1",
            "case_id": spec.case_id,
            "status": "skipped",
            "reason": "no_copy_build_integration_or_explicit_kconfig_symbols",
            "present_copy_operations": 0,
            "missing_copy_operations": 0,
            "integrated_build_files": 0,
            "missing_build_integrations": 0,
            "explicit_kconfig_symbol_verification": {
                "kind": "explicit_kconfig_symbol_verification",
                "status": "skipped",
                "requested_symbols": requested_symbols,
            },
            "results": [],
        }

    results: list[dict[str, Any]] = []
    copy_present_count = 0
    copy_missing_count = 0
    build_integrated_count = 0
    build_missing_count = 0

    for op in copy_ops:
        relative_path = str(op.get("file") or "")
        target_path = spec.resolve_old(relative_path)
        result = {
            "kind": "copy_operation",
            "file": relative_path,
            "status": "present" if target_path.is_file() else "missing_target",
            "target_path": str(target_path),
        }
        results.append(result)
        if result["status"] == "present":
            copy_present_count += 1
        else:
            copy_missing_count += 1

    for op in build_ops:
        relative_path = str(op.get("file") or "")
        donor_path = Path(str(op.get("donor_path") or "")).resolve()
        target_path = spec.resolve_old(relative_path)
        result: dict[str, Any] = {
            "kind": "build_integration",
            "file": relative_path,
            "donor_path": str(donor_path),
            "target_path": str(target_path),
        }
        if not donor_path.is_file():
            result["status"] = "missing_donor"
        elif not target_path.is_file():
            result["status"] = "missing_target"
        else:
            current_text = _read_text(target_path)
            donor_text = _read_text(donor_path)
            related_files = [
                str(item)
                for item in op.get("related_source_files") or list(spec.entry_files)
                if str(item).strip()
            ]
            if donor_path.name.startswith("Kconfig"):
                _candidate_text, merge_meta = _merge_kconfig_blocks(
                    current_text,
                    donor_text,
                    donor_path,
                    related_files,
                    donor_root=spec.primary_donor_root,
                    explicit_configs=_explicit_build_kconfig_symbols(spec.build_kconfig),
                )
            elif donor_path.name in _MAKEFILE_NAMES or donor_path.suffix == ".mk":
                _candidate_text, merge_meta = _merge_makefile_lines(
                    current_text,
                    donor_text,
                    related_files,
                    build_directory=Path(relative_path).parent.as_posix(),
                    build_file=relative_path,
                )
            else:
                _candidate_text, merge_meta = _merge_unique_build_lines(
                    current_text,
                    donor_text,
                )
            result["status"] = (
                "integrated"
                if merge_meta.get("merge_action") == "unchanged"
                else "missing_integration"
            )
            result["verification"] = merge_meta
        results.append(result)
        if result["status"] == "integrated":
            build_integrated_count += 1
        else:
            build_missing_count += 1

    available_symbols = _available_kconfig_symbols(spec.old_root_path)
    missing_symbols = sorted(
        symbol for symbol in requested_symbols if symbol not in available_symbols
    )
    kconfig_verification = {
        "kind": "explicit_kconfig_symbol_verification",
        "status": "complete" if not missing_symbols else "missing",
        "requested_symbols": requested_symbols,
        "missing_symbols": missing_symbols,
    }
    results.append(kconfig_verification)
    failed = bool(
        copy_missing_count
        or build_missing_count
        or missing_symbols
    )

    return {
        "schema": "transplant_prerequisite_verification_v1",
        "case_id": spec.case_id,
        "status": "failed" if failed else "ok",
        "present_copy_operations": copy_present_count,
        "missing_copy_operations": copy_missing_count,
        "integrated_build_files": build_integrated_count,
        "missing_build_integrations": build_missing_count,
        "explicit_kconfig_symbol_verification": kconfig_verification,
        "results": results,
    }
