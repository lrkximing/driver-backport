#!/usr/bin/env python3
from __future__ import annotations

import copy
import csv
import re
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from driver_backport.analysis.kbuild_index import KbuildIndex
from driver_backport.core.case_spec import parse_path_map_item
from driver_backport.core.common import normalize_relpath, slugify
from driver_backport.scope.compile_commands import CompileCommands
from driver_backport.validation.build_utils import kconfig_symbol_name
from driver_backport.validation.kconfig_groups import (
    DEFAULT_CONTEXTS,
    KconfigConfigClass,
    classify_kconfig_assignment_for_roots,
)
from driver_backport.validation.kconfig_sync import kconfig_index


@dataclass(frozen=True)
class DatasetDriverEntry:
    row_number: int
    type: str
    name: str
    path: str
    case_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_number": self.row_number,
            "type": self.type,
            "name": self.name,
            "path": self.path,
            "case_id": self.case_id,
        }


@dataclass(frozen=True)
class MakefileInference:
    configs: list[str]
    expected_modules: list[str]
    expected_objects: list[str]
    module_objects: list[str]
    makefile_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "configs": list(self.configs),
            "expected_modules": list(self.expected_modules),
            "expected_objects": list(self.expected_objects),
            "module_objects": list(self.module_objects),
            "makefile_path": self.makefile_path,
        }


def dataset_case_compile_commands_path(matrix_root: Path, *, label: str, case_id: str) -> Path:
    stem = f"compile_commands_dataset_{label}"
    return Path(matrix_root).expanduser() / "cases" / stem / f"{slugify(case_id)}.json"


def _read_csv_rows(path: Path) -> list[tuple[int, list[str]]]:
    rows: list[tuple[int, list[str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.reader(handle), start=1):
            rows.append((index, [str(item or "").strip() for item in row]))
    return rows


def _header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def read_dataset_entries(path: Path) -> list[DatasetDriverEntry]:
    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"unsupported dataset format: {path.suffix}; expected .csv"
        )
    rows = _read_csv_rows(path)

    header_row_index = None
    headers: dict[str, int] = {}
    for index, (_, values) in enumerate(rows):
        keys = {_header_key(value): column for column, value in enumerate(values)}
        if "path" in keys:
            header_row_index = index
            headers = keys
            break
    if header_row_index is None:
        raise ValueError("dataset is missing a Path column")

    entries: list[DatasetDriverEntry] = []
    for row_number, values in rows[header_row_index + 1 :]:
        path_column = headers["path"]
        raw_path = values[path_column].strip() if path_column < len(values) else ""
        if not raw_path:
            continue
        type_column = headers.get("type")
        name_column = headers.get("name")
        case_id_column = headers.get("caseid")
        if case_id_column is None:
            case_id_column = headers.get("case")
        entries.append(
            DatasetDriverEntry(
                row_number=row_number,
                type=values[type_column].strip() if type_column is not None and type_column < len(values) else "",
                name=values[name_column].strip() if name_column is not None and name_column < len(values) else "",
                path=raw_path.replace("\\", "/").strip(),
                case_id=values[case_id_column].strip()
                if case_id_column is not None and case_id_column < len(values)
                else "",
            )
        )
    return entries


def parse_row_selector(value: str | None) -> set[int] | None:
    if not value:
        return None
    rows: set[int] = set()
    for part in re.split(r"[,\s]+", value.strip()):
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                start, end = end, start
            rows.update(range(start, end + 1))
        else:
            rows.add(int(part))
    return rows


def filter_dataset_entries(
    entries: list[DatasetDriverEntry],
    *,
    rows: set[int] | None = None,
    types: list[str] | None = None,
    names: list[str] | None = None,
    paths: list[str] | None = None,
    limit: int | None = None,
) -> list[DatasetDriverEntry]:
    def matches_any(value: str, needles: list[str] | None) -> bool:
        if not needles:
            return True
        lowered = value.lower()
        return any(needle.lower() in lowered for needle in needles if needle)

    selected: list[DatasetDriverEntry] = []
    for entry in entries:
        if rows is not None and entry.row_number not in rows:
            continue
        if not matches_any(entry.type, types):
            continue
        if not matches_any(entry.name, names):
            continue
        if not matches_any(entry.path, paths):
            continue
        selected.append(entry)
        if limit is not None and limit > 0 and len(selected) >= limit:
            break
    return selected


def _case_entry_file(entry_path: str, *, old_root: str, donor_roots: list[str]) -> str:
    raw = entry_path.strip().replace("\\", "/")
    path = Path(raw).expanduser()
    if path.is_absolute():
        roots = [Path(old_root).expanduser(), *[Path(item).expanduser() for item in donor_roots]]
        for root in roots:
            try:
                return normalize_relpath(path.resolve().relative_to(root.resolve()).as_posix())
            except ValueError:
                continue
        raise ValueError(f"dataset path is absolute but outside old_root/donor_roots: {entry_path}")
    return normalize_relpath(raw)


def _entry_driver_root(entry_file: str) -> str:
    parent = PurePosixPath(entry_file).parent.as_posix()
    if parent in {"", "."}:
        raise ValueError(f"entry file must be under a driver directory: {entry_file}")
    return normalize_relpath(parent)


def _candidate_driver_roots(entry_file: str) -> list[str]:
    parent = PurePosixPath(entry_file).parent
    candidates: list[str] = []
    while parent.as_posix() not in {"", "."}:
        candidate = normalize_relpath(parent.as_posix())
        candidates.append(candidate)
        if candidate == "drivers":
            break
        parent = parent.parent
    return candidates


def derive_driver_root_for_entry(
    *,
    old_root: str,
    donor_roots: list[str],
    entry_file: str,
) -> str:
    roots = [Path(old_root).expanduser(), *[Path(item).expanduser() for item in donor_roots]]
    candidates = _candidate_driver_roots(entry_file)
    for candidate in candidates:
        if any((root / candidate / "Makefile").exists() for root in roots):
            return candidate
    return _entry_driver_root(entry_file)


def _default_escalation_roots(driver_root: str) -> list[str]:
    parts = [part for part in driver_root.split("/") if part]
    roots: list[str] = []
    if len(parts) >= 2 and parts[0] == "drivers":
        roots.append("/".join(parts[:2]))
        if len(parts) >= 3 and parts[1] == "net":
            roots.append("net")
    roots.append("include")
    return list(dict.fromkeys(root for root in roots if root and root != driver_root))


def dataset_case_id(entry: DatasetDriverEntry, *, case_prefix: str = "dataset") -> str:
    if entry.case_id:
        return slugify(entry.case_id)
    label = entry.name or PurePosixPath(entry.path).stem
    return slugify(f"{case_prefix}-{entry.row_number:03d}-{label}")


def _kconfig_symbol_type(root: Path, symbol: str) -> str | None:
    entry = kconfig_index((str(root.resolve()),)).get(symbol) or {}
    raw_type = str(entry.get("type") or "")
    return raw_type or None


def _kconfig_assignment_for_symbol(roots: list[Path], config: str) -> str:
    symbol = config.removeprefix("CONFIG_")
    for root in roots:
        symbol_type = _kconfig_symbol_type(root, symbol)
        if symbol_type:
            value = "y" if symbol_type == "bool" else "m"
            return f"{config}={value}"
    return f"{config}=m"


def _assignment_values(assignments: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = str(assignment).partition("=")
        if separator and name.strip():
            values[name.strip()] = value.strip()
    return values


def _artifact_paths_for_kbuild_closure(
    *,
    driver_root: str,
    closure: dict[str, Any],
    config_assignments: list[str],
    kconfig_roots: list[Path],
) -> tuple[list[str], list[str]]:
    """Resolve Kbuild targets to module or built-in artifacts from Kconfig values."""

    values = _assignment_values(config_assignments)
    selected_assignments = [
        item for item in closure.get("assignments") or [] if isinstance(item, dict)
    ]
    config_index = kconfig_index(tuple(str(root.resolve()) for root in kconfig_roots))
    expected_modules: list[str] = []
    expected_objects: list[str] = []
    for target in closure.get("module_targets") or []:
        normalized_target = normalize_relpath(str(target))
        controlling_configs = [
            str(item.get("config") or "")
            for item in selected_assignments
            if item.get("relation") == "config_membership"
            and normalized_target in {
                normalize_relpath(str(value)) for value in item.get("object_items") or []
            }
            and str(item.get("config") or "")
        ]
        target_values = {values.get(config) for config in controlling_configs}
        promptless = any(
            (config_index.get(config.removeprefix("CONFIG_")) or {}).get("prompt") is False
            for config in controlling_configs
            if config.removeprefix("CONFIG_") in config_index
        )
        artifact_base = PurePosixPath(driver_root) / normalized_target
        if "m" in target_values and not promptless:
            expected_modules.append(
                normalize_relpath(artifact_base.with_suffix(".ko").as_posix())
            )
        else:
            # bool=y and fixed obj-y entries are compiled as in-tree objects, not modules.
            expected_objects.append(normalize_relpath(artifact_base.as_posix()))
    return list(dict.fromkeys(expected_modules)), list(dict.fromkeys(expected_objects))


def infer_makefile_contract(
    *,
    old_root: str,
    donor_roots: list[str],
    entry_file: str,
    driver_root: str | None = None,
) -> MakefileInference:
    driver_root = driver_root or derive_driver_root_for_entry(
        old_root=old_root,
        donor_roots=donor_roots,
        entry_file=entry_file,
    )
    roots = [Path(old_root).expanduser(), *[Path(item).expanduser() for item in donor_roots]]

    for root in roots:
        index = KbuildIndex.from_directory(root, driver_root)
        if not index.assignments:
            continue
        closure = index.assignment_closure_for_sources([entry_file])
        if closure["configs"] or closure["expected_modules"]:
            selected_assignments = list(closure.get("assignments") or [])
            makefile_path = (
                str(root / selected_assignments[0]["file"])
                if selected_assignments
                else None
            )
            configs = [
                _kconfig_assignment_for_symbol(roots, config)
                for config in closure["configs"]
            ]
            expected_modules, expected_objects = _artifact_paths_for_kbuild_closure(
                driver_root=driver_root,
                closure=closure,
                config_assignments=configs,
                kconfig_roots=roots,
            )
            return MakefileInference(
                configs=configs,
                expected_modules=expected_modules,
                expected_objects=expected_objects,
                module_objects=list(closure["module_targets"] or closure["needed_objects"]),
                makefile_path=makefile_path,
            )

    return MakefileInference(
        configs=[],
        expected_modules=[],
        expected_objects=[],
        module_objects=[],
        makefile_path=None,
    )


def _choose_kconfig_context_label(classes: list[KconfigConfigClass]) -> str:
    labels = [item.context_label for item in classes]
    non_native = sorted({label for label in labels if label != "native"})
    if len(non_native) == 1:
        return non_native[0]
    if len(non_native) > 1:
        return "ambiguous[" + ",".join(non_native) + "]"
    return "native"


def _kconfig_class_report(item: KconfigConfigClass) -> dict[str, Any]:
    return {
        "assignment": item.assignment,
        "symbol": item.symbol,
        "context_label": item.context_label,
        "arch_symbols": list(item.arch_symbols),
        "compile_test_reachable": item.compile_test_reachable,
        "reason": item.reason,
        "dependency_symbol_count": len(item.dependency_symbols),
        "unresolved_expr_count": len(item.unresolved_exprs),
    }


def _infer_kconfig_setup(
    *,
    old_root: str,
    donor_roots: list[str],
    assignments: list[str],
    preferred_context: str | None = None,
) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    requested = [str(item).strip() for item in assignments if str(item).strip()]
    roots = [Path(old_root).expanduser(), *[Path(item).expanduser() for item in donor_roots]]
    if not requested:
        return [], {}, {
            "status": "skipped",
            "reason": "no_build_kconfig",
            "roots": [str(root) for root in roots],
        }
    try:
        classes = [
            classify_kconfig_assignment_for_roots(
                roots,
                assignment,
                preferred_context=preferred_context,
            )
            for assignment in requested
        ]
    except Exception as exc:
        return requested, {}, {
            "status": "failed",
            "reason": "kconfig_classification_failed",
            "error": str(exc),
            "assignments": requested,
            "roots": [str(root) for root in roots],
        }

    effective = list(requested)
    if (
        "COMPILE_TEST" not in {kconfig_symbol_name(item) for item in requested}
        and any(
            item.compile_test_reachable
            for item in classes
        )
    ):
        effective.insert(0, "CONFIG_COMPILE_TEST=y")
    effective = list(dict.fromkeys(effective))
    label = _choose_kconfig_context_label(classes)
    context = DEFAULT_CONTEXTS.get(label)
    if context is None:
        return effective, {}, {
            "status": "failed",
            "reason": "ambiguous_kconfig_build_context",
            "context_label": label,
            "assignments": effective,
            "roots": [str(root) for root in roots],
            "config_classes": [_kconfig_class_report(item) for item in classes],
        }
    return effective, dict(context), {
        "status": "ok",
        "context_label": label,
        "build_env": dict(context),
        "assignments": effective,
        "roots": [str(root) for root in roots],
        "config_classes": [_kconfig_class_report(item) for item in classes],
        "preferred_context": preferred_context,
    }


def _compile_commands_architecture(
    path: str | Path | None, *, reject_mixed: bool = False
) -> str | None:
    if not path:
        return None
    database_path = Path(path).expanduser()
    if not database_path.is_file():
        return None
    architectures = CompileCommands.from_file(database_path).architecture_contexts()
    if reject_mixed and len(architectures) > 1:
        raise ValueError(
            "target compilation database contains mixed architectures: "
            + ", ".join(architectures)
        )
    return architectures[0] if len(architectures) == 1 else None


def _replace_module_assignment(command: str, value: str) -> tuple[str, bool]:
    pattern = re.compile(
        r"(?P<prefix>(?:^|\s))M=(?P<quote>['\"]?)(?P<value>[^'\"\s]+)(?P=quote)"
    )
    replaced = False

    def repl(match: re.Match[str]) -> str:
        nonlocal replaced
        replaced = True
        return f"{match.group('prefix')}M={value}"

    return pattern.sub(repl, command, count=1), replaced


def _rewrite_build_phase(
    phase: dict[str, Any],
    *,
    new_driver_root: str,
    old_build_dir: str | None,
    new_build_dir: str | None,
    entry_label: str,
) -> dict[str, Any]:
    rewritten = copy.deepcopy(phase)
    command = str(rewritten.get("command") or "")
    if old_build_dir and new_build_dir:
        command = command.replace(old_build_dir, new_build_dir)
    command, has_m = _replace_module_assignment(command, new_driver_root)
    if not has_m:
        command = command.rstrip() + f" M={new_driver_root} modules"
    rewritten["command"] = command
    rewritten["name"] = f"build-{slugify(entry_label)}"
    return rewritten


def _rewrite_build_phase_for_objects(
    phase: dict[str, Any],
    *,
    expected_objects: list[str],
    entry_label: str,
) -> dict[str, Any]:
    """Turn a generated module build into exact in-tree object targets."""

    rewritten = copy.deepcopy(phase)
    tokens = shlex.split(str(rewritten.get("command") or "make"))
    tokens = [
        token
        for token in tokens
        if token != "modules" and not token.startswith("M=")
    ]
    tokens.extend(path for path in expected_objects if path not in tokens)
    rewritten["command"] = shlex.join(tokens)
    rewritten["name"] = f"build-{slugify(entry_label)}-built-in"
    return rewritten


def _rewrite_path_maps(
    raw_maps: Any,
    *,
    old_build_dir: str | None,
    new_build_dir: str | None,
) -> Any:
    if not old_build_dir or not new_build_dir or not raw_maps:
        return copy.deepcopy(raw_maps)
    rewritten = copy.deepcopy(raw_maps)
    if isinstance(rewritten, list):
        for item in rewritten:
            if isinstance(item, dict) and str(item.get("to") or item.get("new") or item.get("target") or "") == old_build_dir:
                if "to" in item:
                    item["to"] = new_build_dir
                elif "new" in item:
                    item["new"] = new_build_dir
                elif "target" in item:
                    item["target"] = new_build_dir
            elif isinstance(item, str):
                # Strings are uncommon in generated specs; leave them intact unless the exact path appears.
                index = rewritten.index(item)
                rewritten[index] = item.replace(old_build_dir, new_build_dir)
    return rewritten


def _ensure_path_map(raw_maps: Any, *, old_root: str, new_root: str) -> list[Any]:
    old_norm = str(Path(old_root).expanduser()).rstrip("/")
    new_norm = str(Path(new_root).expanduser()).rstrip("/")
    if old_norm == new_norm:
        if isinstance(raw_maps, list):
            return copy.deepcopy(raw_maps)
        if raw_maps:
            return [copy.deepcopy(raw_maps)]
        return []

    if isinstance(raw_maps, list):
        maps = copy.deepcopy(raw_maps)
    elif raw_maps:
        maps = [copy.deepcopy(raw_maps)]
    else:
        maps = []
    for item in maps:
        if parse_path_map_item(item) == (old_norm, new_norm):
            return maps
    maps.append({"from": old_norm, "to": new_norm})
    return maps


def _driver_root_presence(*, old_root: str, donor_roots: list[str], driver_root: str, entry_file: str) -> dict[str, Any]:
    old_tree = Path(old_root).expanduser()
    donor_trees = [Path(item).expanduser() for item in donor_roots]
    old_driver_root = old_tree / driver_root
    old_entry_file = old_tree / entry_file
    donor_driver_roots = [root / driver_root for root in donor_trees]
    donor_entry_files = [root / entry_file for root in donor_trees]
    donor_root_exists = any(path.exists() for path in donor_driver_roots)
    old_root_exists = old_driver_root.exists()
    if donor_root_exists and not old_root_exists:
        mode = "new_driver_root_backport"
    elif old_root_exists:
        mode = "existing_driver_root_backport"
    else:
        mode = "unresolved_driver_root_backport"
    return {
        "migration_mode": mode,
        "old_driver_root_exists": old_root_exists,
        "donor_driver_root_exists": donor_root_exists,
        "old_entry_file_exists": old_entry_file.exists(),
        "donor_entry_file_exists": any(path.exists() for path in donor_entry_files),
    }


def build_dataset_case_payload(
    entry: DatasetDriverEntry,
    base_case: dict[str, Any],
    *,
    case_prefix: str = "dataset",
    state_root_base: Path | None = None,
    build_dir_base: Path | None = None,
    old_root_work: Path | None = None,
    copy_validation: bool = False,
    infer_kconfig: bool = True,
    infer_expected_modules: bool = True,
    compile_commands: Path | None = None,
    donor_compile_commands: list[Path] | None = None,
) -> dict[str, Any]:
    donor_roots_raw = base_case.get("donor_roots")
    if donor_roots_raw is None and base_case.get("donor_root"):
        donor_roots_raw = [base_case["donor_root"]]
    donor_roots = [str(item) for item in donor_roots_raw or []]
    base_old_root = str(base_case.get("old_root") or "")
    work_old_root = str(old_root_work.expanduser()) if old_root_work is not None else base_old_root
    entry_file = _case_entry_file(entry.path, old_root=base_old_root, donor_roots=donor_roots)
    driver_root = derive_driver_root_for_entry(old_root=base_old_root, donor_roots=donor_roots, entry_file=entry_file)
    root_presence = _driver_root_presence(
        old_root=base_old_root,
        donor_roots=donor_roots,
        driver_root=driver_root,
        entry_file=entry_file,
    )
    case_id = dataset_case_id(entry, case_prefix=case_prefix)

    payload = copy.deepcopy(base_case)
    base_build_dir = str(base_case.get("build_dir") or "") or None
    new_build_dir = None
    if build_dir_base is not None:
        new_build_dir = str((build_dir_base / case_id).expanduser())
    elif base_build_dir:
        new_build_dir = str((Path(base_build_dir).expanduser().parent / case_id))

    inference = (
        infer_makefile_contract(
            old_root=base_old_root,
            donor_roots=donor_roots,
            entry_file=entry_file,
            driver_root=driver_root,
        )
        if infer_kconfig or infer_expected_modules
        else MakefileInference(
            configs=[],
            expected_modules=[],
            expected_objects=[],
            module_objects=[],
            makefile_path=None,
        )
    )

    payload["case_id"] = case_id
    payload["old_root"] = work_old_root
    payload["driver_root"] = driver_root
    payload["migration_mode"] = root_presence["migration_mode"]
    payload["entry_files"] = [entry_file]
    payload["editable_roots"] = [driver_root]
    payload["escalation_edit_roots"] = _default_escalation_roots(driver_root)
    payload["state_root"] = str((state_root_base / case_id).expanduser()) if state_root_base is not None else None
    if compile_commands is not None:
        payload["compile_commands"] = str(compile_commands.expanduser().resolve())
    if donor_compile_commands is not None:
        payload["donor_compile_commands"] = [
            str(path.expanduser().resolve()) for path in donor_compile_commands
        ]
    if new_build_dir:
        payload["build_dir"] = new_build_dir

    label = entry.name or PurePosixPath(entry_file).stem
    payload["build_phases"] = [
        _rewrite_build_phase(
            dict(phase),
            new_driver_root=driver_root,
            old_build_dir=base_build_dir,
            new_build_dir=new_build_dir,
            entry_label=label,
        )
        for phase in payload.get("build_phases") or []
        if isinstance(phase, dict)
    ]
    if not payload.get("build_phases"):
        command = f"make"
        if new_build_dir:
            command += f" O={new_build_dir}"
        command += f" M={driver_root} modules"
        payload["build_phases"] = [{"name": f"build-{slugify(label)}", "command": command}]
    if inference.expected_objects:
        object_phase = _rewrite_build_phase_for_objects(
            payload["build_phases"][0],
            expected_objects=inference.expected_objects,
            entry_label=label,
        )
        if inference.expected_modules:
            payload["build_phases"].append(object_phase)
        else:
            payload["build_phases"] = [object_phase]

    payload["compile_path_maps"] = _rewrite_path_maps(
        payload.get("compile_path_maps") or payload.get("path_maps") or [],
        old_build_dir=base_build_dir,
        new_build_dir=new_build_dir,
    )
    if old_root_work is not None:
        payload["compile_path_maps"] = _ensure_path_map(
            payload["compile_path_maps"],
            old_root=base_old_root,
            new_root=work_old_root,
        )

    inferred_configs = list(inference.configs) if infer_kconfig else []
    payload["build_kconfig_exclusive_prefixes"] = []
    payload["build_kconfig_unset"] = []
    explicit_context = str(
        dict(payload.get("build_env") or {}).get("ARCH") or ""
    )
    target_architecture = _compile_commands_architecture(
        payload.get("compile_commands"), reject_mixed=True
    )
    if (
        target_architecture
        and explicit_context
        and explicit_context != target_architecture
    ):
        raise ValueError(
            "target compilation architecture conflicts with build_env ARCH: "
            f"{target_architecture} != {explicit_context}"
        )
    target_context = (
        "native" if target_architecture == "x86" else target_architecture
    )
    donor_context = next(
        (
            "native" if context == "x86" else context
            for context in (
                _compile_commands_architecture(path)
                for path in payload.get("donor_compile_commands") or []
            )
            if context
        ),
        None,
    )
    preferred_context = (
        explicit_context
        if explicit_context in DEFAULT_CONTEXTS
        else target_context or donor_context
    )
    (
        payload["build_kconfig"],
        inferred_build_env,
        inferred_build_env_report,
    ) = _infer_kconfig_setup(
        old_root=base_old_root,
        donor_roots=donor_roots,
        assignments=inferred_configs,
        preferred_context=preferred_context,
    )
    inferred_architecture = str(inferred_build_env.get("ARCH") or "")
    if (
        target_architecture
        and inferred_architecture
        and inferred_architecture != target_architecture
    ):
        raise ValueError(
            "Kconfig architecture conflicts with target compilation context: "
            f"{inferred_architecture} != {target_architecture}"
        )
    if inferred_build_env:
        existing_build_env = {
            str(key): str(value)
            for key, value in dict(payload.get("build_env") or {}).items()
        }
        payload["build_env"] = {**inferred_build_env, **existing_build_env}
        if existing_build_env:
            inferred_build_env_report["merged_build_env"] = dict(payload["build_env"])
    if target_architecture and target_architecture != "x86":
        payload["build_env"] = {
            **dict(payload.get("build_env") or {}),
            "ARCH": target_architecture,
        }
        inferred_build_env_report["target_compile_architecture"] = target_architecture
        inferred_build_env_report["merged_build_env"] = dict(payload["build_env"])

    if copy_validation:
        if infer_expected_modules:
            existing = [str(item) for item in payload.get("expected_modules") or [] if str(item).strip()]
            existing.extend(item for item in inference.expected_modules if item not in existing)
            payload["expected_modules"] = existing
            existing_objects = [
                str(item) for item in payload.get("expected_objects") or [] if str(item).strip()
            ]
            existing_objects.extend(
                item for item in inference.expected_objects if item not in existing_objects
            )
            payload["expected_objects"] = existing_objects
    else:
        payload["expected_modules"] = list(inference.expected_modules) if infer_expected_modules else []
        payload["expected_objects"] = list(inference.expected_objects) if infer_expected_modules else []
        payload["require_load_validation"] = False
        payload["load_phases"] = []

    payload["dataset_entry"] = entry.to_dict()
    payload["dataset_generation"] = {
        "case_prefix": case_prefix,
        "migration_mode": root_presence["migration_mode"],
        "driver_root_presence": root_presence,
        "inferred_makefile_contract": inference.to_dict(),
        "copy_validation": copy_validation,
        "infer_kconfig": infer_kconfig,
        "infer_expected_modules": infer_expected_modules,
        "base_old_root": base_old_root,
        "old_root_work": work_old_root if old_root_work is not None else None,
        "inferred_build_env": inferred_build_env_report,
    }
    return payload


def materialize_dataset_cases(
    *,
    dataset_path: Path,
    base_case: dict[str, Any],
    output_dir: Path,
    rows: set[int] | None = None,
    types: list[str] | None = None,
    names: list[str] | None = None,
    paths: list[str] | None = None,
    limit: int | None = None,
    case_prefix: str = "dataset",
    state_root_base: Path | None = None,
    build_dir_base: Path | None = None,
    old_root_work: Path | None = None,
    copy_validation: bool = False,
    infer_kconfig: bool = True,
    infer_expected_modules: bool = True,
    compile_commands_matrix_root: Path | None = None,
    compile_commands_label: str | None = None,
    donor_compile_commands_label: str = "donor-v6.19",
) -> dict[str, Any]:
    from driver_backport.core.common import ensure_dir, write_json

    entries = read_dataset_entries(dataset_path)
    selected = filter_dataset_entries(
        entries,
        rows=rows,
        types=types,
        names=names,
        paths=paths,
        limit=limit,
    )
    cases_dir = ensure_dir(output_dir / "cases")
    resolved_state_root_base = state_root_base if state_root_base is not None else output_dir / "state"
    case_items: list[dict[str, Any]] = []
    for entry in selected:
        case_id = dataset_case_id(entry, case_prefix=case_prefix)
        old_compile_commands: Path | None = None
        donor_compile_commands: Path | None = None
        if compile_commands_matrix_root is not None:
            if not compile_commands_label:
                raise ValueError("compile_commands_label is required when compile_commands_matrix_root is set")
            old_compile_commands = dataset_case_compile_commands_path(
                compile_commands_matrix_root,
                label=compile_commands_label,
                case_id=case_id,
            )
            donor_compile_commands = dataset_case_compile_commands_path(
                compile_commands_matrix_root,
                label=donor_compile_commands_label,
                case_id=case_id,
            )
        payload = build_dataset_case_payload(
            entry,
            base_case,
            case_prefix=case_prefix,
            state_root_base=resolved_state_root_base,
            build_dir_base=build_dir_base,
            old_root_work=old_root_work,
            copy_validation=copy_validation,
            infer_kconfig=infer_kconfig,
            infer_expected_modules=infer_expected_modules,
            compile_commands=old_compile_commands,
            donor_compile_commands=(
                [donor_compile_commands]
                if donor_compile_commands is not None
                else None
            ),
        )
        compile_commands_source: dict[str, Any] | None = None
        if compile_commands_matrix_root is not None:
            compile_commands_source = {
                "matrix_root": str(Path(compile_commands_matrix_root).expanduser().resolve()),
                "compile_commands_label": compile_commands_label,
                "donor_compile_commands_label": donor_compile_commands_label,
                "compile_commands": payload["compile_commands"],
                "donor_compile_commands": list(payload["donor_compile_commands"]),
            }
            generation = dict(payload.get("dataset_generation") or {})
            generation["compile_commands_source"] = compile_commands_source
            payload["dataset_generation"] = generation
        case_path = cases_dir / f"{payload['case_id']}.case.json"
        write_json(case_path, payload)
        case_item = {
            "case_id": payload["case_id"],
            "case_spec_path": str(case_path.resolve()),
            "state_root": payload.get("state_root"),
            "driver_root": payload["driver_root"],
            "migration_mode": payload.get("migration_mode"),
            "entry_files": payload["entry_files"],
            "compile_commands": payload.get("compile_commands"),
            "donor_compile_commands": payload.get("donor_compile_commands") or [],
            "build_kconfig": payload.get("build_kconfig") or [],
            "expected_modules": payload.get("expected_modules") or [],
            "dataset_entry": entry.to_dict(),
            "dataset_generation": payload.get("dataset_generation") or {},
        }
        if compile_commands_source is not None:
            case_item["compile_commands_source"] = compile_commands_source
        case_items.append(case_item)

    manifest = {
        "dataset_path": str(dataset_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "total_dataset_entries": len(entries),
        "selected_count": len(case_items),
        "cases": case_items,
    }
    if compile_commands_matrix_root is not None:
        manifest["compile_commands_source"] = {
            "matrix_root": str(Path(compile_commands_matrix_root).expanduser().resolve()),
            "compile_commands_label": compile_commands_label,
            "donor_compile_commands_label": donor_compile_commands_label,
        }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
