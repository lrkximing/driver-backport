#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from driver_backport.analysis.kbuild_index import KbuildIndex
from driver_backport.analysis.export_index import parse_export_contracts
from driver_backport.validation.build_utils import (
    case_build_env,
    find_make_binary,
    olddefconfig_command,
    parse_kconfig_assignment,
    scripts_config_command,
)
from driver_backport.validation.kconfig_sync import (
    available_kconfig_symbols as shared_available_kconfig_symbols,
    clear_kconfig_caches,
    expand_kconfig_assignments as shared_expand_kconfig_assignments,
    kconfig_dependency_closure as shared_kconfig_dependency_closure,
    kconfig_index as shared_kconfig_index,
    partition_promptless_explicit_mismatches,
    read_config_value,
)
from driver_backport.core.case_spec import CaseSpec, CommandSpec
from driver_backport.core.common import (
    append_unique_string as _append_unique,
    ensure_dir,
    normalize_relpath,
    tri_and,
    tri_not,
    tri_or,
    utc_now,
)
from driver_backport.validation.compile_coverage import (
    TRANSLATION_UNIT_SUFFIXES,
    collect_translation_unit_coverage,
)
from driver_backport.scope.compile_context_materialization import (
    _compile_command_from_verbose_output,
)


MODPOST_UNDEFINED_RE = re.compile(r'(?:ERROR|WARNING): modpost: "([^"]+)" \[([^\]]+)\] undefined!')
MODPOST_NAMESPACE_IMPORT_RE = re.compile(
    r"(?:ERROR|WARNING): modpost: module (?P<module>\S+) uses symbol "
    r"(?P<symbol>[A-Za-z_][A-Za-z0-9_]*) from namespace "
    r"(?P<namespace>[^,\s]+), but does not import it\."
)
KCONFIG_RECURSIVE_DEPENDENCY_RE = re.compile(
    r"^(?P<file>[^:\n]+):(?P<line>\d+):error: recursive dependency detected!\s*$",
    re.MULTILINE,
)
KCONFIG_DEPENDENCY_STEP_RE = re.compile(
    r"^(?P<file>[^:\n]+):(?P<line>\d+):\s*symbol\s+"
    r"(?P<symbol>[A-Za-z0-9_]+)\s+(?P<relation>.+?)\s*$",
    re.MULTILINE,
)
MODULE_SYMVERS_MISSING_RE = re.compile(r"Module\.symvers is missing", re.IGNORECASE)
UNKNOWN_SYMBOL_RE = re.compile(r"Unknown symbol ([A-Za-z0-9_]+) \(err")
UNDECLARED_RE = re.compile(r"error: [‘'`](\w+)[’'`] undeclared")
NO_MEMBER_RE = re.compile(r"error: .* has no member named [‘'`](\w+)[’'`]")
DUPLICATE_EXPORT_RE = re.compile(r"__export_symbol_([A-Za-z0-9_]+).*already defined")
MODPOST_EXPORTED_TWICE_RE = re.compile(
    r"modpost:\s+[^:\n]+:\s+['\"]([A-Za-z0-9_]+)['\"] exported twice\.\s+"
    r"Previous export was in ([^\s]+)",
    re.IGNORECASE,
)
GENERIC_ERROR_RE = re.compile(r"^(.*?):(\d+)(?::(\d+))?: (?:fatal )?error: (.*)$", re.MULTILINE)
GENERIC_WARNING_RE = re.compile(
    r"^(.*?):(\d+)(?::(\d+))?: warning: (.*)$",
    re.MULTILINE,
)
UNDECLARED_SYMBOL_RE = re.compile(r"[‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`] undeclared")
NO_MEMBER_SYMBOL_RE = re.compile(r"has no member named [‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`]")
NO_MEMBER_CONTAINER_RE = re.compile(
    r"^[‘'`](?:(?:const|volatile)\s+)*(struct|union) ([A-Za-z_][A-Za-z0-9_]*)[’'`] "
    r"has no member named [‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`]"
)
CONFLICTING_TYPES_SYMBOL_RE = re.compile(r"conflicting types for [‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`]")
IMPLICIT_DECL_SYMBOL_RE = re.compile(r"implicit declaration of function [‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`]")
DEFAULT_INT_DECL_SYMBOL_RE = re.compile(
    r"type defaults to .+ in declaration of [‘'`]"
    r"([A-Za-z_][A-Za-z0-9_]*)[’'`]"
)
DID_YOU_ME_SYMBOL_RE = re.compile(
    r"\bdid you mean\s+[‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`]?\??",
    re.IGNORECASE,
)
PREVIOUS_DECL_NOTE_RE = re.compile(
    r"^previous declaration of "
    r"[‘'`](?P<symbol>[A-Za-z_][A-Za-z0-9_]*)[’'`] "
    r"with type (?P<type>.+?)\s*$"
)
UNKNOWN_TYPE_SYMBOL_RE = re.compile(r"unknown type name [‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`]")
UNDEFINED_TAGGED_TYPE_SYMBOL_RE = re.compile(
    r"(?:invalid use of undefined type|dereferencing pointer to incomplete type) "
    r"[‘'`](?:const\s+)?(?:struct|union|enum)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)[’'`]",
    re.IGNORECASE,
)
WERROR_FLAG_RE = re.compile(r"\[-Werror(?:=(?P<flag>[^\]]+))?\]", re.IGNORECASE)
MIGRATION_CONTRACT_WARNING_RE = re.compile(
    r"(?:"
    r"returning .+ from a function with return type .+ makes |"
    r"(?:assignment|initialization) to .+ from .+ makes |"
    r"incompatible pointer type|"
    r"passing argument .+ from incompatible pointer type|"
    r"cast between incompatible function types"
    r")",
    re.IGNORECASE,
)
UNUSED_VARIABLE_SYMBOL_RE = re.compile(r"unused variable [‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`]")
UNUSED_PARAMETER_SYMBOL_RE = re.compile(r"unused parameter [‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`]")
UNUSED_BUT_SET_SYMBOL_RE = re.compile(r"(?:variable|parameter) [‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`] set but not used")
DEFINED_NOT_USED_SYMBOL_RE = re.compile(r"[‘'`]([A-Za-z_][A-Za-z0-9_]*)[’'`] defined but not used")
MISSING_HEADER_RE = re.compile(
    r"(?:fatal error:\s*)?(?P<header>[A-Za-z0-9_./+-]+\.h):\s+No such file or directory|"
    r"[‘'`](?P<quoted>[^‘'`]+\.h)[’'`]\s+file not found",
    re.IGNORECASE,
)
NO_RULE_TARGET_RE = re.compile(
    r"No rule to make target [‘'`](?P<target>[^’'`]+)[’'`]"
    r"(?:,\s+needed by [‘'`](?P<needed_by>[^’'`]+)[’'`])?",
    re.IGNORECASE,
)
KCONFIG_TRUE_VALUES = {"y", "m"}
EXPORT_SYMBOL_RE_TEMPLATE = r"\b(EXPORT_SYMBOL(?:_GPL|_NS_GPL|_NS)?)\s*\(\s*{symbol}\b"
EXPORT_SCAN_SUFFIXES = {".c", ".h"}
EXPORT_SCAN_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
}


def _phase_cwd(spec: CaseSpec, phase: CommandSpec) -> Path:
    if not phase.cwd:
        return spec.old_root_path
    cwd = Path(phase.cwd)
    if cwd.is_absolute():
        return cwd.resolve()
    return (spec.old_root_path / phase.cwd).resolve()


def _phase_command(spec: CaseSpec, phase: CommandSpec) -> str:
    command = str(phase.command)
    if spec.make_bin:
        stripped = command.lstrip()
        prefix = command[: len(command) - len(stripped)]
        if stripped == "make":
            command = prefix + shlex.quote(str(spec.make_bin))
        elif stripped.startswith("make "):
            command = prefix + shlex.quote(str(spec.make_bin)) + stripped[4:]
    command = _narrow_module_build_command(
        command,
        directory=spec.driver_root,
        modules=spec.expected_modules,
    )
    return _inject_direct_object_kconfig_assignments(command, spec)


def _inject_direct_object_kconfig_assignments(
    command: str,
    spec: CaseSpec,
) -> str:
    """Bind hidden library objects to the exact make invocation.

    Promptless Kconfig library symbols may be selected only by a platform
    driver and therefore cannot be persisted in ``.config``.  An explicitly
    declared exact-object validation target still needs the same Kbuild
    variable on the make command line so its ``obj-$(CONFIG_...)`` rule is
    materialized.  This changes build context only; it does not edit Kconfig.
    """

    if not spec.expected_objects or not spec.build_kconfig:
        return command
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    if (
        not tokens
        or Path(tokens[0]).name not in {"make", "gmake"}
        or any(token in {"&&", "||", ";", "|"} for token in tokens)
    ):
        return command
    expected_targets = {
        normalize_relpath(str(path))
        for path in spec.expected_objects
        if str(path)
    }
    command_targets = {
        normalize_relpath(token)
        for token in tokens[1:]
        if token
        and not token.startswith("-")
        and "=" not in token
    }
    if not expected_targets.intersection(command_targets):
        return command
    existing_names = {
        token.split("=", 1)[0]
        for token in tokens
        if re.fullmatch(r"CONFIG_[A-Za-z0-9_]+=.*", token)
    }
    additions: list[str] = []
    for assignment in spec.build_kconfig:
        try:
            name, value = parse_kconfig_assignment(assignment)
        except ValueError:
            continue
        if name in existing_names:
            continue
        additions.append(f"{name}={value}")
        existing_names.add(name)
    if not additions:
        return command
    return command.rstrip() + " " + " ".join(
        shlex.quote(assignment) for assignment in additions
    )


def _module_targets_for_directory(
    directory: str,
    modules: list[object],
) -> list[str]:
    base = Path(normalize_relpath(directory))
    targets: list[str] = []
    for raw_module in modules:
        module_text = normalize_relpath(str(raw_module))
        if not module_text:
            return []
        if not module_text.endswith(".ko"):
            module_text += ".ko"
        module = Path(module_text)
        if module.parent == Path("."):
            module = base / module
        try:
            relative = module.relative_to(base)
        except ValueError:
            return []
        if relative.suffix != ".ko" or relative == Path("."):
            return []
        target = relative.as_posix()
        if target not in targets:
            targets.append(target)
    return targets


def _narrow_module_build_command(
    command: str,
    *,
    directory: str,
    modules: list[object],
) -> str:
    """Replace a broad Kbuild directory build with declared module targets."""

    targets = _module_targets_for_directory(directory, modules)
    if not targets:
        return command
    try:
        tokens = shlex.split(command)
    except ValueError:
        return command
    if not tokens or Path(tokens[0]).name not in {"make", "gmake"}:
        return command
    expected_m = normalize_relpath(directory)
    m_tokens = [token for token in tokens if token.startswith("M=")]
    if len(m_tokens) != 1 or normalize_relpath(m_tokens[0][2:]) != expected_m:
        return command
    module_indexes = [
        index for index, token in enumerate(tokens) if token == "modules"
    ]
    if len(module_indexes) != 1:
        return command
    index = module_indexes[0]
    tokens[index : index + 1] = targets
    return shlex.join(tokens)


def _format_command(args: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in args)


def _tail_text(text: str, *, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _kconfig_olddefconfig_failure_issues(output: str) -> list[dict[str, Any]]:
    """Preserve structured Kconfig failures emitted by olddefconfig.

    Kconfig reports recursive dependencies as a location followed by a
    symbol/relation chain.  Collapsing that record into ``command_failed``
    hides both the metadata owner and the fact that a source edit is needed.
    """

    recursive = KCONFIG_RECURSIVE_DEPENDENCY_RE.search(output)
    if recursive is None:
        return []
    steps = [
        {
            "file": normalize_relpath(match.group("file")),
            "line": int(match.group("line")),
            "symbol": match.group("symbol"),
            "relation": match.group("relation").strip(),
        }
        for match in KCONFIG_DEPENDENCY_STEP_RE.finditer(output)
    ]
    primary_file = normalize_relpath(recursive.group("file"))
    related_files = list(
        dict.fromkeys(
            [primary_file, *[step["file"] for step in steps]]
        )
    )
    return [
        {
            "kind": "kconfig_recursive_dependency",
            "severity": "error",
            "file": primary_file,
            "line": int(recursive.group("line")),
            "phase_group": "build",
            "symbols": list(
                dict.fromkeys(step["symbol"] for step in steps)
            ),
            "dependency_chain": steps,
            "related_files": related_files,
            "raw": output,
        }
    ]


def _snapshot_config(config_path: Path, artifacts_dir: Path) -> tuple[Path, Path, bool]:
    backup_path = artifacts_dir / "config.before-kconfig-sync"
    existed = config_path.exists()
    if existed:
        shutil.copyfile(config_path, backup_path)
    return config_path, backup_path, existed


def _restore_config(config_path: Path, backup_path: Path, existed: bool) -> None:
    if existed and backup_path.exists():
        shutil.copyfile(backup_path, config_path)
        return
    if not existed and config_path.exists():
        os.remove(config_path)


def _verify_kconfig_assignments(
    spec: CaseSpec,
    assignments: list[str],
    expansion_records: list[dict[str, Any]],
) -> list[dict[str, str | None]]:
    explicit_symbols = {
        str(record.get("symbol"))
        for record in expansion_records
        if "explicit build_kconfig" in set(record.get("reasons") or [])
    }
    mismatches: list[dict[str, str | None]] = []
    for assignment in assignments:
        name, expected = parse_kconfig_assignment(assignment)
        symbol = name.removeprefix("CONFIG_")
        if symbol not in explicit_symbols:
            continue
        actual = read_config_value(spec.build_config_path, name)
        if actual != expected and not (expected == "m" and actual == "y"):
            mismatches.append(
                {
                    "name": name,
                    "expected": expected,
                    "actual": actual,
                }
            )
    return mismatches


def _split_gcc_location(prefix: str) -> tuple[str, int, int] | None:
    parts = prefix.rsplit(":", 2)
    if len(parts) != 3:
        return None
    file_path, line_text, column_text = parts
    if not file_path or not line_text.isdigit():
        return None
    column = int(column_text) if column_text.isdigit() else 0
    return file_path, int(line_text), column


def _split_gcc_location_flexible(prefix: str) -> tuple[str, int, int] | None:
    text = str(prefix or "").strip().rstrip(",")
    while text.endswith(":"):
        text = text[:-1].rstrip()
    parts = text.rsplit(":", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
        return parts[0], int(parts[1]), int(parts[2])
    parts = text.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1]), 0
    return None


def _gcc_include_stack_between(text: str, start: int, end: int) -> list[dict[str, Any]]:
    stack: list[dict[str, Any]] = []
    for raw_line in text[start:end].splitlines():
        stripped = raw_line.strip()
        payload = ""
        if stripped.startswith("In file included from "):
            payload = stripped[len("In file included from ") :]
        elif stripped.startswith("from "):
            payload = stripped[len("from ") :]
        if not payload:
            continue
        location = _split_gcc_location_flexible(payload)
        if location is None:
            continue
        file_path, line, column = location
        stack.append(
            {
                "file": file_path,
                "line": line,
                "column": column,
                "raw": raw_line.strip(),
            }
        )
    return stack


def _trigger_files_from_include_stack(include_stack: list[dict[str, Any]]) -> list[str]:
    trigger_files: list[str] = []
    for frame in include_stack:
        file_path = str(frame.get("file") or "")
        if Path(file_path).suffix not in {".c", ".cc", ".cpp", ".cxx"}:
            continue
        _append_unique(trigger_files, file_path)
    return trigger_files


def _quoted_identifier(text: str) -> str | None:
    quote_pairs = [("‘", "’"), ("'", "'"), ("`", "`")]
    for start_quote, end_quote in quote_pairs:
        start = text.find(start_quote)
        if start < 0:
            continue
        end = text.find(end_quote, start + 1)
        if end > start:
            value = text[start + 1 : end].strip()
            return value or None
    return None


def _previous_declaration_fields(note_text: str) -> dict[str, str] | None:
    match = PREVIOUS_DECL_NOTE_RE.fullmatch(note_text.strip())
    if match is None:
        return None
    type_spelling = match.group("type").strip()
    normalized_type = type_spelling
    type_suffix = ""
    quote_pairs = {"‘": "’", "'": "'", "`": "`"}
    closing_quote = quote_pairs.get(type_spelling[:1])
    if closing_quote:
        closing_index = type_spelling.find(closing_quote, 1)
        if closing_index > 0:
            normalized_type = type_spelling[1:closing_index]
            type_suffix = type_spelling[closing_index + 1 :].strip()
    fields = {
        "symbol": match.group("symbol"),
        "type": normalized_type,
        "type_spelling": type_spelling,
    }
    if type_suffix:
        fields["type_suffix"] = type_suffix
    return fields


def _gcc_notes_between(text: str, start: int, end: int) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for raw_line in text[start:end].splitlines():
        if ": note: " not in raw_line:
            continue
        prefix, note_text = raw_line.split(": note: ", 1)
        location = _split_gcc_location(prefix)
        if location is None:
            continue
        file_path, line, column = location
        note = {
            "kind": "compiler_note",
            "file": file_path,
            "line": line,
            "column": column,
            "message": note_text.strip(),
            "raw": raw_line.strip(),
        }
        if "in expansion of macro" in note_text or "in definition of macro" in note_text:
            note["kind"] = "macro_expansion_note"
            symbol = _quoted_identifier(note_text)
            if symbol:
                note["symbol"] = symbol
        previous_declaration = _previous_declaration_fields(note_text)
        if previous_declaration is not None:
            note["kind"] = "previous_declaration"
            note.update(previous_declaration)
        notes.append(note)
    return notes


def _spec_kconfig_index(spec: CaseSpec) -> dict[str, dict[str, Any]]:
    roots = (str(spec.old_root_path),)
    return shared_kconfig_index(roots)


def _normalize_kconfig_symbol_name(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("# ") and text.endswith(" is not set"):
        text = text[2 : -len(" is not set")]
    name = text.split("=", 1)[0].strip()
    return name.removeprefix("CONFIG_")


def _normalize_kconfig_unset_assignment(value: str) -> str | None:
    symbol = _normalize_kconfig_symbol_name(value)
    if not symbol:
        return None
    return f"CONFIG_{symbol}=n"


def _normalize_kconfig_prefix(value: str) -> str:
    return str(value or "").strip().removeprefix("CONFIG_")


def _symbol_defined_under_driver(spec: CaseSpec, entry: dict[str, Any]) -> bool:
    driver_roots = [(spec.old_root_path / spec.driver_root).resolve()]
    for item in entry.get("defined_in") or []:
        try:
            path = Path(str(item)).resolve()
        except OSError:
            continue
        if any(path == root or path.is_relative_to(root) for root in driver_roots):
            return True
    return False


def _exclusive_kconfig_unset_assignments(
    spec: CaseSpec,
    *,
    available_symbols: frozenset[str],
    enabled_symbols: set[str],
) -> list[str]:
    prefixes = [
        _normalize_kconfig_prefix(item)
        for item in spec.build_kconfig_exclusive_prefixes
        if _normalize_kconfig_prefix(item)
    ]
    if not prefixes:
        return []

    index = _spec_kconfig_index(spec)
    assignments: list[str] = []
    for symbol, entry in sorted(index.items()):
        if symbol not in available_symbols:
            continue
        if symbol in enabled_symbols:
            continue
        if not any(symbol == prefix or symbol.startswith(prefix) for prefix in prefixes):
            continue
        if not _symbol_defined_under_driver(spec, entry):
            continue
        assignments.append(f"CONFIG_{symbol}=n")
    return assignments


def _verify_kconfig_unsets(spec: CaseSpec, assignments: list[str]) -> list[dict[str, str | None]]:
    mismatches: list[dict[str, str | None]] = []
    for assignment in assignments:
        name, expected = parse_kconfig_assignment(assignment)
        actual = read_config_value(spec.build_config_path, name)
        if expected != "n":
            continue
        if actual not in {None, "n"}:
            mismatches.append(
                {
                    "name": name,
                    "expected": expected,
                    "actual": actual,
                }
            )
    return mismatches


def sync_build_kconfig(
    spec: CaseSpec,
    *,
    artifacts_dir: Path,
    force: bool = False,
) -> dict | None:
    started = time.time()

    def duration() -> float:
        return round(time.time() - started, 3)

    requested_assignments = [item for item in spec.build_kconfig if item]
    requested_unsets = [item for item in spec.build_kconfig_unset if item]
    exclusive_prefixes = [item for item in spec.build_kconfig_exclusive_prefixes if item]
    if (
        not force
        and not requested_assignments
        and not requested_unsets
        and not exclusive_prefixes
    ):
        return None
    clear_kconfig_caches()
    assignments, expansion_records = shared_expand_kconfig_assignments(
        config_path=spec.build_config_path,
        assignments=requested_assignments,
        kconfig_roots=[spec.old_root_path],
    )
    explicit_symbols = {
        str(record.get("symbol"))
        for record in expansion_records
        if "explicit build_kconfig" in set(record.get("reasons") or [])
    }
    unresolved_dependency_expressions = [
        {"symbol": str(record.get("symbol")), **item}
        for record in expansion_records
        for item in record.get("unresolved_depends") or []
    ]
    enabled_symbols = {
        parse_kconfig_assignment(assignment)[0].removeprefix("CONFIG_")
        for assignment in assignments
    }

    ensure_dir(artifacts_dir)
    if spec.build_dir_path is not None:
        ensure_dir(spec.build_dir_path)
    scripts_config = spec.old_root_path / "scripts" / "config"
    available_symbols = shared_available_kconfig_symbols(spec.old_root_path)
    applied: list[str] = []
    disabled: list[str] = []
    skipped: list[str] = []
    skipped_explicit: list[str] = []
    skipped_derived: list[str] = []
    skipped_disable: list[str] = []
    sync_log = artifacts_dir / "kconfig-sync.log"
    lines: list[str] = []

    def result(
        *,
        command: str,
        returncode: int,
        issues: list[dict[str, Any]],
        include_assignment_state: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": "kconfig-sync",
            "command": command,
            "cwd": str(spec.old_root_path),
            "returncode": returncode,
            "ok": returncode == 0,
            "duration_sec": duration(),
            "log_path": str(sync_log.resolve()),
            "issues": issues,
        }
        if include_assignment_state:
            payload.update(
                {
                    "requested_assignments": requested_assignments,
                    "applied_assignments": applied,
                    "disabled_assignments": disabled,
                    "skipped_assignments": skipped,
                    "skipped_explicit_assignments": skipped_explicit,
                    "skipped_derived_assignments": skipped_derived,
                    "skipped_disable_assignments": skipped_disable,
                    "expanded_assignments": assignments,
                    "expansion_records": expansion_records,
                    "unresolved_dependency_expressions": (
                        unresolved_dependency_expressions
                    ),
                }
            )
        payload.update(extra)
        return payload

    disable_candidates: list[str] = []
    for item in requested_unsets:
        normalized = _normalize_kconfig_unset_assignment(item)
        if normalized:
            _append_unique(disable_candidates, normalized)
    for item in _exclusive_kconfig_unset_assignments(
        spec,
        available_symbols=available_symbols,
        enabled_symbols=enabled_symbols,
    ):
        _append_unique(disable_candidates, item)
    disable_assignments: list[str] = []
    for assignment in disable_candidates:
        name, _value = parse_kconfig_assignment(assignment)
        symbol = name.removeprefix("CONFIG_")
        if symbol in enabled_symbols:
            skipped_disable.append(assignment)
            continue
        _append_unique(disable_assignments, assignment)
    if disable_assignments:
        lines.append("disabled Kconfig assignments before enabling requested symbols:")
        lines.extend(f"  {assignment}" for assignment in disable_assignments)
    if skipped_disable:
        lines.append("skipped Kconfig disable assignments because they are required by build_kconfig:")
        lines.extend(f"  {assignment}" for assignment in skipped_disable)
    lines.append("Kconfig parser: shared_source_visibility_dependencies")
    lines.append("Kconfig roots:")
    for root in [spec.old_root_path]:
        lines.append(f"  {root}")
    if expansion_records:
        lines.append("expanded Kconfig assignments:")
        lines.extend(
            f"  {record['assignment']} ({'; '.join(record.get('reasons') or [])})"
            for record in expansion_records
        )
    env = case_build_env(spec)
    config_path, config_backup_path, config_existed = _snapshot_config(spec.build_config_path, artifacts_dir)

    def run_scripts_config(assignment: str) -> subprocess.CompletedProcess[str]:
        name, value = parse_kconfig_assignment(assignment)
        process = subprocess.run(
            scripts_config_command(
                scripts_config,
                name,
                value,
                config_path=(
                    spec.build_config_path
                    if spec.build_dir_path is not None
                    else None
                ),
            ),
            cwd=spec.old_root_path,
            env=env,
            text=True,
            capture_output=True,
        )
        lines.append(f"$ {' '.join(process.args)}")
        if process.stdout:
            lines.append(process.stdout.rstrip())
        if process.stderr:
            lines.append(process.stderr.rstrip())
        return process

    if not scripts_config.exists():
        sync_log.write_text("missing scripts/config\n", encoding="utf-8")
        return result(
            command="scripts/config + olddefconfig",
            returncode=1,
            issues=[
                {
                    "kind": "command_failed",
                    "severity": "error",
                    "raw": "kconfig sync failed: missing scripts/config",
                }
            ],
        )

    for assignment in disable_assignments:
        name, _value = parse_kconfig_assignment(assignment)
        symbol = name.removeprefix("CONFIG_")
        if symbol not in available_symbols:
            skipped_disable.append(assignment)
            lines.append(f"skip disable {assignment} (symbol not present)")
            continue
        process = run_scripts_config(assignment)
        if process.returncode != 0:
            sync_log.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            _restore_config(config_path, config_backup_path, config_existed)
            return result(
                command="scripts/config + olddefconfig",
                returncode=process.returncode,
                issues=[
                    {
                        "kind": "command_failed",
                        "severity": "error",
                        "raw": f"kconfig sync failed while disabling {assignment}",
                    }
                ],
            )
        disabled.append(assignment)

    for assignment in assignments:
        name, _value = parse_kconfig_assignment(assignment)
        symbol = name.removeprefix("CONFIG_")
        if symbol not in available_symbols:
            skipped.append(assignment)
            if symbol in explicit_symbols:
                skipped_explicit.append(assignment)
            else:
                skipped_derived.append(assignment)
            lines.append(f"skip {assignment} (symbol not present)")
            continue
        process = run_scripts_config(assignment)
        if process.returncode != 0:
            sync_log.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            _restore_config(config_path, config_backup_path, config_existed)
            return result(
                command="scripts/config + olddefconfig",
                returncode=process.returncode,
                issues=[
                    {
                        "kind": "command_failed",
                        "severity": "error",
                        "raw": f"kconfig sync failed for {assignment}",
                    }
                ],
            )
        applied.append(assignment)

    if skipped_explicit:
        sync_log.write_text("\n".join(lines).rstrip() + ("\n" if lines else ""), encoding="utf-8")
        return result(
            command="scripts/config + olddefconfig",
            returncode=1,
            issues=[
                {
                    "kind": "kconfig_symbol_missing",
                    "severity": "error",
                    "file": f"{spec.driver_root}/Kconfig",
                    "phase_group": "build",
                    "raw": "requested Kconfig symbols are missing from the target tree: "
                    + ", ".join(skipped_explicit),
                }
            ],
            include_assignment_state=True,
        )

    olddefconfig_args = olddefconfig_command(spec)
    process = subprocess.run(
        olddefconfig_args,
        cwd=spec.old_root_path,
        env=env,
        text=True,
        capture_output=True,
    )
    lines.append(f"$ {_format_command(olddefconfig_args)}")
    if process.stdout:
        lines.append(process.stdout.rstrip())
    if process.stderr:
        lines.append(process.stderr.rstrip())
    sync_log.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    if process.returncode != 0:
        full_output = "\n".join(lines)
        tail = _tail_text(full_output, max_lines=16)
        structured_issues = _kconfig_olddefconfig_failure_issues(
            full_output
        )
        _restore_config(config_path, config_backup_path, config_existed)
        return result(
            command=_format_command(olddefconfig_args),
            returncode=process.returncode,
            issues=structured_issues or [
                {
                    "kind": "command_failed",
                    "severity": "error",
                    "raw": (
                        "kconfig sync failed during olddefconfig"
                        + (f"\n{tail}" if tail else "")
                    ),
                }
            ],
        )

    config_after_path = artifacts_dir / "config.after-kconfig-sync"
    if spec.build_config_path.exists():
        shutil.copyfile(spec.build_config_path, config_after_path)
    requested_config_names = [parse_kconfig_assignment(item)[0] for item in requested_assignments]
    dependency_closure = shared_kconfig_dependency_closure(
        config_path=spec.build_config_path,
        seed_configs=requested_config_names,
        kconfig_roots=[spec.old_root_path],
    )
    if dependency_closure.get("symbols"):
        lines.append("final Kconfig dependency closure:")
        lines.extend(
            "  "
            f"{record['config']}={record['value']} "
            f"({record['reason']}; selects={','.join(record.get('selects') or []) or '-'})"
            for record in dependency_closure.get("symbols") or []
        )
    if dependency_closure.get("blocked_dependency_expressions"):
        lines.append("blocked Kconfig dependency expressions:")
        lines.extend(
            "  "
            f"{item['symbol']} depends on {item['expr']} "
            f"with {item['symbol_values']}"
            for item in dependency_closure.get("blocked_dependency_expressions") or []
        )
    mismatches = _verify_kconfig_assignments(spec, applied, expansion_records)
    mismatches, direct_object_required = partition_promptless_explicit_mismatches(
        mismatches,
        expansion_records,
        allow_for_exact_objects=bool(spec.expected_objects),
    )
    if direct_object_required:
        lines.append("promptless assignments delegated to exact object validation:")
        lines.extend(
            f"  {item['name']}: expected={item['expected']} actual={item['actual']}"
            for item in direct_object_required
        )
    if mismatches:
        mismatch_lines = [
            f"{item['name']}: expected={item['expected']} actual={item['actual']}"
            for item in mismatches
        ]
        lines.extend(["final Kconfig mismatches:", *mismatch_lines])
        sync_log.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return result(
            command=_format_command(olddefconfig_args),
            returncode=1,
            issues=[
                {
                    "kind": "kconfig_assignment_not_enabled",
                    "severity": "error",
                    "file": f"{spec.driver_root}/Kconfig",
                    "phase_group": "build",
                    "raw": "requested Kconfig assignments were not preserved by olddefconfig: "
                    + "; ".join(mismatch_lines),
                }
            ],
            include_assignment_state=True,
            dependency_closure=dependency_closure,
            mismatched_assignments=mismatches,
            config_after_path=str(config_after_path.resolve()) if config_after_path.exists() else None,
        )

    unset_mismatches = _verify_kconfig_unsets(spec, disabled)
    if unset_mismatches:
        mismatch_lines = [
            f"{item['name']}: expected={item['expected']} actual={item['actual']}"
            for item in unset_mismatches
        ]
        lines.extend(["final disabled Kconfig mismatches:", *mismatch_lines])
        sync_log.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return result(
            command=_format_command(olddefconfig_args),
            returncode=1,
            issues=[
                {
                    "kind": "kconfig_unset_not_preserved",
                    "severity": "error",
                    "file": f"{spec.driver_root}/Kconfig",
                    "phase_group": "build",
                    "raw": "disabled Kconfig assignments were re-enabled by olddefconfig: "
                    + "; ".join(mismatch_lines),
                }
            ],
            include_assignment_state=True,
            dependency_closure=dependency_closure,
            mismatched_unsets=unset_mismatches,
            config_after_path=str(config_after_path.resolve()) if config_after_path.exists() else None,
        )

    sync_log.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return result(
        command=_format_command(olddefconfig_args),
        returncode=0,
        issues=[],
        include_assignment_state=True,
        dependency_closure=dependency_closure,
        direct_object_required_assignments=direct_object_required,
        config_after_path=str(config_after_path.resolve()) if config_after_path.exists() else None,
    )


def _expected_module_roots(spec: CaseSpec) -> list[Path]:
    roots: list[Path] = []
    if spec.build_dir_path is not None:
        roots.append(spec.build_dir_path)
    roots.append(spec.old_root_path)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(root.resolve())
    return unique


def _expected_module_candidates(spec: CaseSpec, relative_path: str) -> list[Path]:
    return [(root / relative_path).resolve() for root in _expected_module_roots(spec)]


def _makefile_module_records(
    spec: CaseSpec,
    module_stem: str,
    *,
    build_directory: str | None = None,
) -> dict[str, Any]:
    directory = normalize_relpath(build_directory or spec.driver_root)
    index = KbuildIndex.from_directory(spec.old_root_path, directory)
    makefile = spec.old_root_path / directory / "Makefile"
    if not index.assignments:
        return {
            "makefile": str(makefile),
            "exists": False,
            "obj_membership": [],
            "object_definitions": [],
        }

    module_object = f"{module_stem}.o"
    obj_membership = [
        {
            "line": record.start_line,
            "config": record.config,
            "text": "\n".join(record.lines),
            "config_value": read_config_value(
                spec.build_config_path,
                str(record.config),
            ),
            "assignment_id": record.id,
        }
        for record in index.top_assignments
        if record.config and module_object in record.object_items
    ]
    object_definitions = [
        {
            "line": record.start_line,
            "text": "\n".join(record.lines),
            "objects": " ".join(record.object_items),
            "assignment_id": record.id,
        }
        for record in index.module_assignments.get(module_stem, [])
    ]

    return {
        "makefile": str(makefile),
        "exists": True,
        "obj_membership": obj_membership,
        "object_definitions": object_definitions,
        "kbuild_graph": index.to_graph(tree="old"),
    }


def _diagnose_missing_expected_module(spec: CaseSpec, relative_path: str) -> dict[str, Any]:
    module_stem = Path(relative_path).stem
    build_directory = normalize_relpath(Path(relative_path).parent.as_posix())
    makefile = _makefile_module_records(
        spec,
        module_stem,
        build_directory=build_directory,
    )
    membership = list(makefile.get("obj_membership") or [])
    duplicate_makefile_membership = len(membership) > 1
    enabled_membership = [
        item for item in membership if str(item.get("config_value") or "") in KCONFIG_TRUE_VALUES
    ]
    seed_configs: list[str] = []
    for config_name in [str(item.get("config") or "") for item in membership]:
        if config_name and config_name not in seed_configs:
            seed_configs.append(config_name)
    kconfig_closure = shared_kconfig_dependency_closure(
        config_path=spec.build_config_path,
        seed_configs=seed_configs,
        kconfig_roots=[spec.old_root_path],
        follow_selects=False,
    )
    seed_set = set(seed_configs)
    missing_dependency_symbols = [
        item
        for item in kconfig_closure.get("missing_or_disabled_symbols") or []
        if item.get("config") not in seed_set
    ]
    dependency_blocked = bool(kconfig_closure.get("blocked_dependency_expressions")) or bool(
        missing_dependency_symbols
    )

    if not membership:
        reason = "module_target_missing_from_makefile"
    elif not enabled_membership:
        reason = "makefile_config_not_enabled"
    elif dependency_blocked:
        reason = "config_dependency_not_enabled"
    else:
        reason = "artifact_missing_after_enabled_makefile_target"

    return {
        "module_stem": module_stem,
        "guessed_config": None,
        "guessed_config_value": None,
        "makefile": makefile,
        "duplicate_makefile_membership": duplicate_makefile_membership,
        "kconfig_closure": kconfig_closure,
        "missing_dependency_symbols": missing_dependency_symbols,
        "reason": reason,
    }


def _check_expected_modules(spec: CaseSpec, *, artifacts_dir: Path) -> dict | None:
    started = time.time()
    expected = [item for item in spec.expected_modules if item]
    if not expected:
        return None

    ensure_dir(artifacts_dir)
    log_path = artifacts_dir / "expected-modules.log"
    found: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for relative_path in expected:
        candidates = _expected_module_candidates(spec, relative_path)
        diagnosis = _diagnose_missing_expected_module(spec, relative_path)
        item = {
            "module": Path(relative_path).name,
            "relative_path": relative_path,
            "candidate_paths": [str(path) for path in candidates],
            "diagnosis": diagnosis,
        }
        found_path = next(
            (path for path in candidates if path.exists() and path.is_file() and path.stat().st_size > 0),
            None,
        )
        if found_path is not None:
            found.append({**item, "path": str(found_path)})
            continue
        missing.append(item)

    nearby_modules: list[dict[str, str]] = []
    for root in _expected_module_roots(spec):
        if not root.exists():
            continue
        for path in sorted(root.glob(f"{spec.driver_root}/**/*.ko")):
            if not path.is_file():
                continue
            nearby_modules.append(
                {
                    "root": str(root),
                    "relative_path": str(path.relative_to(root)),
                    "path": str(path),
                }
            )

    lines = [
        "search_roots:",
        *[f"  {root}" for root in _expected_module_roots(spec)],
        "expected:",
        *[f"  {item}" for item in expected],
        "found:",
        *[f"  {item['relative_path']} -> {item['path']}" for item in found],
        "missing:",
        *[f"  {item['relative_path']} -> {', '.join(item['candidate_paths'])}" for item in missing],
        "missing_diagnosis:",
        *[
            "  "
            f"{item['relative_path']}: reason={item['diagnosis']['reason']} "
            f"config={item['diagnosis']['guessed_config']}:{item['diagnosis']['guessed_config_value']} "
            f"makefile_membership={len(item['diagnosis']['makefile'].get('obj_membership') or [])} "
            f"duplicate_membership={item['diagnosis'].get('duplicate_makefile_membership')} "
            f"missing_deps="
            f"{','.join(dep.get('config') or dep.get('symbol') or '?' for dep in item['diagnosis'].get('missing_dependency_symbols') or []) or '-'}"
            for item in missing
        ],
        "missing_kconfig_closure:",
        *[
            "  "
            f"{item['relative_path']} -> "
            + "; ".join(
                f"{record['config']}={record['value']} ({record['reason']})"
                for record in (item["diagnosis"].get("kconfig_closure") or {}).get("symbols", [])
            )
            for item in missing
        ],
        "nearby_modules:",
        *[f"  {item['relative_path']} -> {item['path']}" for item in nearby_modules],
    ]
    log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    issues = [
        {
            "kind": "missing_expected_module",
            "severity": "error",
            "module": item["module"],
            "file": f"{spec.driver_root}/Makefile",
            "expected_module_path": item["relative_path"],
            "candidate_paths": item["candidate_paths"],
            "build_output_diagnosis": item["diagnosis"],
            "nearby_modules": [module["relative_path"] for module in nearby_modules],
            "phase_group": "build",
            "raw": f"expected module was not produced: {item['relative_path']}",
        }
        for item in missing
    ]
    return {
        "name": "expected-modules",
        "command": "check expected module artifacts",
        "cwd": str(_expected_module_roots(spec)[0]),
        "returncode": 0 if not missing else 1,
        "ok": not missing,
        "duration_sec": round(time.time() - started, 3),
        "log_path": str(log_path.resolve()),
        "issues": issues,
        "expected_modules": expected,
        "found_modules": found,
        "missing_modules": missing,
        "nearby_modules": nearby_modules,
    }


def _expected_object_candidates(spec: CaseSpec, relative_path: str) -> list[Path]:
    relative = normalize_relpath(relative_path)
    candidates: list[Path] = []
    for root in _expected_module_roots(spec):
        candidate = root / relative
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _check_expected_objects(spec: CaseSpec, *, artifacts_dir: Path) -> dict | None:
    started = time.time()
    expected = [item for item in spec.expected_objects if item]
    if not expected:
        return None

    ensure_dir(artifacts_dir)
    log_path = artifacts_dir / "expected-objects.log"
    found: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for relative_path in expected:
        candidates = _expected_object_candidates(spec, relative_path)
        found_path = next(
            (path for path in candidates if path.is_file() and path.stat().st_size > 0),
            None,
        )
        record = {
            "relative_path": relative_path,
            "candidate_paths": [str(path) for path in candidates],
        }
        if found_path is None:
            missing.append(record)
        else:
            found.append({**record, "path": str(found_path)})

    log_path.write_text(
        "\n".join(
            [
                "search_roots:",
                *[f"  {root}" for root in _expected_module_roots(spec)],
                "expected:",
                *[f"  {item}" for item in expected],
                "found:",
                *[f"  {item['relative_path']} -> {item['path']}" for item in found],
                "missing:",
                *[
                    f"  {item['relative_path']} -> {', '.join(item['candidate_paths'])}"
                    for item in missing
                ],
            ]
        ).rstrip()
        + "\n",
        encoding="utf-8",
    )
    issues = [
        {
            "kind": "missing_expected_object",
            "severity": "error",
            "file": f"{spec.driver_root}/Makefile",
            "expected_object_path": item["relative_path"],
            "candidate_paths": item["candidate_paths"],
            "phase_group": "build",
            "raw": f"expected built-in object was not produced: {item['relative_path']}",
        }
        for item in missing
    ]
    return {
        "name": "expected-objects",
        "command": "check expected built-in object artifacts",
        "cwd": str(_expected_module_roots(spec)[0]),
        "returncode": 0 if not missing else 1,
        "ok": not missing,
        "duration_sec": round(time.time() - started, 3),
        "log_path": str(log_path.resolve()),
        "issues": issues,
        "expected_objects": expected,
        "found_objects": found,
        "missing_objects": missing,
    }


def extract_issues(text: str) -> list[dict]:
    issues: list[dict] = []
    compile_issues_by_symbol: dict[str, list[dict]] = {}
    undeclared_symbols = set(UNDECLARED_RE.findall(text))
    no_member_symbols = set(NO_MEMBER_RE.findall(text))
    file_scoped_undeclared_symbols: set[str] = set()
    file_scoped_no_member_symbols: set[str] = set()
    module_symvers_missing = bool(MODULE_SYMVERS_MISSING_RE.search(text))

    if module_symvers_missing:
        issues.append(
            {
                "kind": "module_symvers_missing",
                "severity": "error",
                "raw": (
                    "Module.symvers is missing; modpost unresolved-symbol diagnostics "
                    "are not reliable repair evidence"
                ),
                "suppressed_issue_kinds": ["modpost_undefined_symbol"],
            }
        )
    else:
        for symbol, module in MODPOST_UNDEFINED_RE.findall(text):
            issues.append(
                {
                    "kind": "modpost_undefined_symbol",
                    "symbol": symbol,
                    "module": module,
                    "severity": "error",
                    "raw": f'modpost undefined symbol {symbol} in {module}',
                }
            )

    for match in MODPOST_NAMESPACE_IMPORT_RE.finditer(text):
        issues.append(
            {
                "kind": "modpost_namespace_import_missing",
                "symbol": match.group("symbol"),
                "module": match.group("module"),
                "namespace": match.group("namespace"),
                "severity": "error",
                "raw": match.group(0),
            }
        )

    for symbol in UNKNOWN_SYMBOL_RE.findall(text):
        issues.append(
            {
                "kind": "module_unknown_symbol",
                "symbol": symbol,
                "severity": "error",
                "raw": f"unknown symbol {symbol}",
            }
        )

    for symbol in DUPLICATE_EXPORT_RE.findall(text):
        issues.append(
            {
                "kind": "duplicate_export_symbol",
                "symbol": symbol,
                "severity": "error",
                "raw": f"duplicate export symbol {symbol}",
            }
        )

    for symbol, previous_provider in MODPOST_EXPORTED_TWICE_RE.findall(text):
        issues.append(
            {
                "kind": "duplicate_export_symbol",
                "symbol": symbol,
                "previous_provider": previous_provider,
                "severity": "error",
                "raw": (
                    f"duplicate export symbol {symbol}; previous provider "
                    f"was {previous_provider}"
                ),
            }
        )

    for match in NO_RULE_TARGET_RE.finditer(text):
        target = (match.group("target") or "").strip()
        needed_by = (match.group("needed_by") or "").strip()
        issue = {
            "kind": "make_no_rule_to_make_target",
            "severity": "error",
            "target": target,
            "needed_by": needed_by,
            "raw": match.group(0),
        }
        if target and "/" in target:
            issue["file"] = target
        if needed_by:
            issue["related_files"] = [needed_by]
        issues.append(issue)

    generic_error_matches = list(GENERIC_ERROR_RE.finditer(text))
    for index, match in enumerate(generic_error_matches):
        file_path, line, column, message = match.groups()
        stripped_message = message.strip()
        issue_kind = "compile_error"
        symbol_match = None
        undeclared_match = UNDECLARED_SYMBOL_RE.search(stripped_message)
        no_member_match = NO_MEMBER_SYMBOL_RE.search(stripped_message)
        missing_header_match = MISSING_HEADER_RE.search(stripped_message)
        if undeclared_match:
            issue_kind = "compile_undeclared_identifier"
            symbol_match = undeclared_match
            file_scoped_undeclared_symbols.add(symbol_match.group(1))
        elif no_member_match:
            issue_kind = "compile_missing_struct_member"
            symbol_match = no_member_match
            file_scoped_no_member_symbols.add(symbol_match.group(1))
        elif missing_header_match:
            issue_kind = "compile_missing_header"
        else:
            symbol_match = UNKNOWN_TYPE_SYMBOL_RE.search(stripped_message)
            if not symbol_match:
                symbol_match = UNDEFINED_TAGGED_TYPE_SYMBOL_RE.search(
                    stripped_message
                )
            if symbol_match:
                issue_kind = "compile_missing_type"
            if not symbol_match:
                symbol_match = CONFLICTING_TYPES_SYMBOL_RE.search(stripped_message)
            if not symbol_match:
                symbol_match = IMPLICIT_DECL_SYMBOL_RE.search(stripped_message)
                if symbol_match:
                    issue_kind = "compile_missing_declaration"
            if not symbol_match:
                symbol_match = DEFAULT_INT_DECL_SYMBOL_RE.search(
                    stripped_message
                )
                if symbol_match:
                    issue_kind = "compile_missing_declaration"
            if not symbol_match:
                for warning_symbol_re in (
                    UNUSED_VARIABLE_SYMBOL_RE,
                    UNUSED_PARAMETER_SYMBOL_RE,
                    UNUSED_BUT_SET_SYMBOL_RE,
                    DEFINED_NOT_USED_SYMBOL_RE,
                ):
                    symbol_match = warning_symbol_re.search(stripped_message)
                    if symbol_match:
                        break

        issue = {
            "kind": issue_kind,
            "file": file_path,
            "line": int(line or 0),
            "column": int(column or 0),
            "message": stripped_message,
            "severity": "error",
            "raw": f"{file_path}:{line}:{column}: {stripped_message}",
        }
        if MIGRATION_CONTRACT_WARNING_RE.search(stripped_message):
            issue["migration_contract_error"] = True
            issue["repair_relevance"] = "migration_contract"
        warning_as_error_match = WERROR_FLAG_RE.search(stripped_message)
        if warning_as_error_match:
            issue["warning_as_error"] = True
            warning_flag = (warning_as_error_match.group("flag") or "").strip()
            if warning_flag:
                issue["warning_flag"] = warning_flag
        context_start = generic_error_matches[index - 1].end() if index > 0 else 0
        include_stack = _gcc_include_stack_between(text, context_start, match.start())
        if include_stack:
            issue["include_stack"] = include_stack
            trigger_files = _trigger_files_from_include_stack(include_stack)
            if trigger_files:
                issue["trigger_files"] = trigger_files
        if symbol_match:
            symbol = symbol_match.group(1)
            issue["symbol"] = symbol
            compile_issues_by_symbol.setdefault(symbol, []).append(issue)
            if issue_kind == "compile_missing_struct_member":
                container_match = NO_MEMBER_CONTAINER_RE.match(stripped_message)
                if container_match and container_match.group(3) == symbol:
                    issue["member_container"] = {
                        "container_kind": container_match.group(1),
                        "container_name": container_match.group(2),
                    }
        elif missing_header_match:
            header = (missing_header_match.group("header") or missing_header_match.group("quoted") or "").strip()
            if header:
                issue["header"] = header
                issue["symbol"] = header
        suggested_symbols = list(
            dict.fromkeys(DID_YOU_ME_SYMBOL_RE.findall(stripped_message))
        )
        if suggested_symbols:
            issue["compiler_suggestions"] = [
                {
                    "symbol": symbol,
                    "relation": "did_you_mean",
                    "authority": "compiler",
                }
                for symbol in suggested_symbols
            ]
        note_end = generic_error_matches[index + 1].start() if index + 1 < len(generic_error_matches) else len(text)
        related_notes = _gcc_notes_between(text, match.end(), note_end)
        if related_notes:
            issue["related_notes"] = related_notes
            related_files = list(issue.get("related_files") or [])
            for note in related_notes:
                if note.get("file"):
                    _append_unique(related_files, str(note["file"]))
            if related_files:
                issue["related_files"] = related_files
            raw = str(issue.get("raw") or "")
            for note in related_notes:
                note_raw = str(note.get("raw") or "")
                if note_raw and note_raw not in raw:
                    raw = (raw + "\n" + note_raw).strip()
            issue["raw"] = raw
        issues.append(issue)

    # GCC may leave ABI/type failures at warning severity while another error
    # makes the translation unit fail.  Dropping them from the fresh work set
    # lets a repair round close one diagnostic while retaining a known-invalid
    # contract at the same compiler frontier.  Preserve all warnings as
    # evidence and route the type/ABI subset as migration repair observations.
    for match in GENERIC_WARNING_RE.finditer(text):
        file_path, line, column, message = match.groups()
        stripped_message = message.strip()
        warning_as_error_match = WERROR_FLAG_RE.search(stripped_message)
        migration_contract_warning = bool(
            MIGRATION_CONTRACT_WARNING_RE.search(stripped_message)
        )
        issue = {
            "kind": (
                "compile_contract_warning"
                if migration_contract_warning
                else "compile_warning"
            ),
            "file": file_path,
            "line": int(line or 0),
            "column": int(column or 0),
            "message": stripped_message,
            "severity": "warning",
            "raw": match.group(0),
        }
        if warning_as_error_match:
            issue["warning_as_error"] = True
            warning_flag = (
                warning_as_error_match.group("flag") or ""
            ).strip()
            if warning_flag:
                issue["warning_flag"] = warning_flag
        if migration_contract_warning:
            issue["migration_contract_warning"] = True
            issue["repair_relevance"] = "migration_contract"
        issues.append(issue)

    for symbol in sorted(undeclared_symbols - file_scoped_undeclared_symbols):
        issues.append(
            {
                "kind": "compile_undeclared_identifier",
                "symbol": symbol,
                "severity": "error",
                "raw": f"undeclared identifier {symbol}",
            }
        )

    for symbol in sorted(no_member_symbols - file_scoped_no_member_symbols):
        issues.append(
            {
                "kind": "compile_missing_struct_member",
                "symbol": symbol,
                "severity": "error",
                "raw": f"missing struct member {symbol}",
            }
        )

    for raw_line in text.splitlines():
        if ": note: " not in raw_line:
            continue
        prefix, note_text = raw_line.split(": note: ", 1)
        location = _split_gcc_location(prefix)
        previous_declaration = _previous_declaration_fields(note_text)
        if location is None or previous_declaration is None:
            continue
        file_path, line, column = location
        symbol = previous_declaration["symbol"]
        note = {
            "kind": "previous_declaration",
            "file": file_path,
            "line": line,
            "column": column,
            "raw": raw_line.strip(),
            **previous_declaration,
        }
        for issue in compile_issues_by_symbol.get(symbol, []):
            related_files = list(issue.get("related_files") or [])
            _append_unique(related_files, file_path)
            issue["related_files"] = related_files
            related_notes = list(issue.get("related_notes") or [])
            note_identity = (
                note["kind"],
                note["file"],
                note["line"],
                note["column"],
                note["symbol"],
            )
            if not any(
                (
                    existing.get("kind"),
                    existing.get("file"),
                    int(existing.get("line") or 0),
                    int(existing.get("column") or 0),
                    existing.get("symbol"),
                )
                == note_identity
                for existing in related_notes
                if isinstance(existing, dict)
            ):
                related_notes.append(note)
            issue["related_notes"] = related_notes
            raw = str(issue.get("raw") or "")
            if note["raw"] not in raw:
                issue["raw"] = (raw + "\n" + note["raw"]).strip()

    return issues


def _enrich_issue(issue: dict[str, Any], *, phase_group: str) -> dict[str, Any]:
    enriched = dict(issue)
    enriched["phase_group"] = phase_group
    return enriched


def _enable_make_keep_going(env: dict[str, str]) -> dict[str, str]:
    updated = dict(env)
    makeflags = str(updated.get("MAKEFLAGS") or "").strip()
    tokens = makeflags.split()
    if "-k" not in tokens and "--keep-going" not in tokens:
        tokens.append("--keep-going")
    updated["MAKEFLAGS"] = " ".join(tokens)
    return updated


def _probe_exported_provider_objects(
    spec: CaseSpec,
    issues: list[dict[str, Any]],
    *,
    artifacts_dir: Path,
    env: dict[str, str],
    timeout_sec: int | float | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify source exports against the realized Kbuild configuration."""

    build_dir = spec.build_dir_path
    sources = sorted({
        normalize_relpath(str(export.get("file") or ""))
        for issue in issues
        for export in (issue.get("provider_resolution") or {}).get("exports") or []
        if isinstance(export, dict)
    })
    probes: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    builtin_selection: subprocess.CompletedProcess[str] | None = None
    selection_command: list[str] = []
    selection_log: Path | None = None
    for index, source in enumerate(sources, start=1):
        probe: dict[str, Any] = {"source": source}
        probes.append(probe)
        if build_dir is None or not source.endswith(".c"):
            probe.update(status="unavailable", reason="provider_build_context_missing")
            continue
        try:
            if not spec.resolve_old(source).is_file():
                probe.update(status="unavailable", reason="provider_source_missing")
                continue
        except ValueError:
            probe.update(status="unavailable", reason="provider_source_outside_tree")
            continue
        directory = normalize_relpath(str(Path(source).parent))
        graph = KbuildIndex.from_directory(
            spec.old_root_path, directory
        ).build_contract_graph()
        configs = (graph.get("source_to_configs") or {}).get(source) or []
        active_modes = sorted({
            "y" if config == "y" else read_config_value(
                spec.build_config_path,
                config if str(config).startswith("CONFIG_") else f"CONFIG_{config}",
            )
            for config in configs
        } & {"y", "m"})
        if not active_modes and not configs:
            if builtin_selection is None:
                selection_command = [
                    spec.make_bin or "make", "-C", str(spec.old_root_path),
                    f"O={build_dir}", "-n", "V=1", "vmlinux",
                ]
                builtin_selection = subprocess.run(
                    selection_command, cwd=spec.old_root_path, env=env,
                    text=True, capture_output=True, timeout=timeout_sec,
                )
                selection_log = artifacts_dir / "provider-builtin-selection.log"
                selection_log.write_text(
                    builtin_selection.stdout + builtin_selection.stderr,
                    encoding="utf-8",
                )
            selection = _compile_command_from_verbose_output(
                builtin_selection.stdout,
                source_file=source, build_dir=build_dir,
                old_root=spec.old_root_path,
                make_command=selection_command,
                returncode=builtin_selection.returncode,
            )
            probe["builtin_selection"] = {
                "source_reached": selection is not None,
                "returncode": builtin_selection.returncode,
                "log_path": str(selection_log),
                "compile_context_origin": (
                    selection.get("driver_backport_context_origin")
                    if selection is not None else None
                ),
            }
            if selection is not None:
                active_modes = ["y"]
        if not active_modes:
            probe.update(
                status="unavailable",
                reason="provider_not_proven_active_by_kbuild",
                configs=list(configs),
            )
            continue
        target = str(Path(source).with_suffix(".o"))
        command = [
            spec.make_bin or "make", "-C", str(spec.old_root_path),
            f"O={build_dir}", "-j1", target,
        ]
        started = time.time()
        result = subprocess.run(
            command,
            cwd=spec.old_root_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        output = result.stdout + result.stderr
        log_path = artifacts_dir / f"provider-object-{index}.log"
        log_path.write_text(output, encoding="utf-8")
        probe.update(
            status="passed" if result.returncode == 0 else "failed",
            command=command,
            returncode=result.returncode,
            duration_sec=round(time.time() - started, 3),
            log_path=str(log_path.resolve()),
            configs=list(configs),
            active_modes=active_modes,
        )
        if result.returncode != 0:
            found = extract_issues(output)
            diagnostics.extend(found or [{
                "kind": "command_failed",
                "severity": "error",
                "file": source,
                "raw": f"provider object {target} failed to compile: "
                + _tail_text(output, max_lines=20),
            }])
    return probes, diagnostics


def _build_current_builtin_provider_exports(
    spec: CaseSpec,
    issues: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    *,
    artifacts_dir: Path,
    env: dict[str, str],
    timeout_sec: int | float | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not probes or any(
        probe.get("status") != "passed" or probe.get("active_modes") != ["y"]
        for probe in probes
    ):
        return None, []
    build_dir = spec.build_dir_path
    if build_dir is None:
        return None, []
    symbols = {
        str(issue.get("symbol") or "").strip()
        for issue in issues
        if issue.get("kind") == "modpost_undefined_symbol"
    }
    if not symbols or "" in symbols:
        return None, []
    symvers = build_dir / "Module.symvers"
    witnessed = {
        record["symbol"]
        for record in _module_symvers_records(symvers)
        if record["module"] == "vmlinux"
    }
    missing = sorted(symbols - witnessed)
    if not missing:
        return {"status": "already_witnessed", "symbols": sorted(symbols)}, []

    command = [spec.make_bin or "make", "-C", str(spec.old_root_path),
               f"O={build_dir}", "-j1", "vmlinux"]
    started = time.time()
    result = subprocess.run(
        command,
        cwd=spec.old_root_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    output = result.stdout + result.stderr
    log_path = artifacts_dir / "provider-vmlinux.log"
    log_path.write_text(output, encoding="utf-8")
    record: dict[str, Any] = {
        "status": "failed" if result.returncode else "missing_witness",
        "command": command,
        "returncode": result.returncode,
        "duration_sec": round(time.time() - started, 3),
        "log_path": str(log_path.resolve()),
        "requested_symbols": missing,
    }
    if result.returncode:
        diagnostics = extract_issues(output) or [{
            "kind": "command_failed",
            "severity": "error",
            "raw": "current built-in provider build failed: "
            + _tail_text(output, max_lines=20),
        }]
        return record, diagnostics
    builtin_symvers = build_dir / "vmlinux.symvers"
    if builtin_symvers.is_file():
        builtin_lines, skipped_lines = _normalized_module_symvers_lines(
            builtin_symvers.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        builtin_lines = [
            line for line in builtin_lines
            if _module_symvers_module_path(line) == "vmlinux"
        ]
        if builtin_lines:
            existing_lines, _ = _normalized_module_symvers_lines(
                symvers.read_text(encoding="utf-8", errors="replace").splitlines()
                if symvers.is_file() else []
            )
            # Current built-in exports replace stale CRCs; module witnesses stay intact.
            _rewrite_module_symvers(symvers, [
                *[line for line in existing_lines
                  if _module_symvers_module_path(line) != "vmlinux"],
                *builtin_lines,
            ])
            record["builtin_export_table"] = {
                "source": str(builtin_symvers.resolve()),
                "target": str(symvers.resolve()),
                "builtin_export_count": len(builtin_lines),
                "skipped_lines": skipped_lines,
            }
    witnessed = {
        item["symbol"]
        for item in _module_symvers_records(symvers)
        if item["module"] == "vmlinux"
    }
    remaining = sorted(symbols - witnessed)
    record["missing_witnesses"] = remaining
    if not remaining:
        record["status"] = "ready"
    return record, []


def _run_phase(
    spec: CaseSpec,
    phase: CommandSpec,
    *,
    artifacts_dir: Path,
    collect_compile_coverage: bool = False,
) -> dict:
    ensure_dir(artifacts_dir)
    started = time.time()
    cwd = _phase_cwd(spec, phase)
    command = _phase_command(spec, phase)
    env = case_build_env(spec)
    env.update(phase.env)
    if collect_compile_coverage:
        env = _enable_make_keep_going(env)
    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=phase.timeout_sec,
    )
    output = result.stdout + result.stderr
    log_path = artifacts_dir / f"{phase.name}.log"
    log_path.write_text(output, encoding="utf-8")

    issues = _annotate_exported_modpost_issues(spec, extract_issues(output))
    modpost_warn_retry: dict[str, Any] | None = None
    provider_probes: list[dict[str, Any]] = []
    builtin_export_build: dict[str, Any] | None = None
    if result.returncode != 0 and _only_target_exported_modpost_issues(issues):
        provider_probes, provider_diagnostics = _probe_exported_provider_objects(
            spec,
            issues,
            artifacts_dir=artifacts_dir,
            env=env,
            timeout_sec=phase.timeout_sec,
        )
        issues.extend(provider_diagnostics)
        if not provider_diagnostics:
            builtin_export_build, builtin_diagnostics = (
                _build_current_builtin_provider_exports(
                    spec, issues, provider_probes,
                    artifacts_dir=artifacts_dir,
                    env=env,
                    timeout_sec=phase.timeout_sec,
                )
            )
            issues.extend(builtin_diagnostics)
            if builtin_export_build and builtin_export_build["status"] == "ready":
                retry = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=phase.timeout_sec,
                )
                output += "\n\n--- strict retry after current vmlinux build ---\n"
                output += retry.stdout + retry.stderr
                log_path.write_text(output, encoding="utf-8")
                result = retry
                issues = _annotate_exported_modpost_issues(
                    spec, extract_issues(retry.stdout + retry.stderr)
                )
    if (
        result.returncode != 0
        and builtin_export_build is None
        and provider_probes
        and all(probe.get("status") == "passed" for probe in provider_probes)
        and _only_target_exported_modpost_issues(issues)
    ):
        retry_env = dict(env)
        retry_env.setdefault("KBUILD_MODPOST_WARN", "1")
        retry_started = time.time()
        retry = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=retry_env,
            text=True,
            capture_output=True,
            timeout=phase.timeout_sec,
        )
        retry_duration = round(time.time() - retry_started, 3)
        retry_output = retry.stdout + retry.stderr
        output = (
            output
            + "\n\n--- retry with KBUILD_MODPOST_WARN=1 for exported target-tree symbols ---\n"
            + retry_output
        )
        log_path.write_text(output, encoding="utf-8")
        retry_issues = _annotate_exported_modpost_issues(spec, extract_issues(retry_output))
        if retry.returncode == 0 and _only_target_exported_modpost_issues(retry_issues):
            issues = _mark_target_symvers_boundary_issues(retry_issues)
        else:
            issues = retry_issues or issues
        modpost_warn_retry = {
            "enabled": True,
            "strict_returncode": result.returncode,
            "returncode": retry.returncode,
            "duration_sec": retry_duration,
            "KBUILD_MODPOST_WARN": retry_env.get("KBUILD_MODPOST_WARN"),
            "reason": "strict modpost undefined symbols have exported providers in the target tree",
        }
    if result.returncode != 0 and not issues:
        tail = _tail_text(output, max_lines=20)
        issues.append(
            {
                "kind": "command_failed",
                "severity": "error",
                "raw": f"{phase.name} exited with {result.returncode}" + (f"\n{tail}" if tail else ""),
            }
        )

    translation_unit_coverage = (
        collect_translation_unit_coverage(spec, issues=issues)
        if collect_compile_coverage
        else None
    )
    return {
        "name": phase.name,
        "command": command,
        "cwd": str(cwd),
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "duration_sec": round(time.time() - started, 3),
        "log_path": str(log_path.resolve()),
        "issues": issues,
        "modpost_warn_retry": modpost_warn_retry,
        "provider_object_probes": provider_probes,
        "builtin_provider_build": builtin_export_build,
        "build_execution": {
            "keep_going": collect_compile_coverage,
            "MAKEFLAGS": env.get("MAKEFLAGS") if collect_compile_coverage else None,
        },
        "translation_unit_coverage": translation_unit_coverage,
    }


def _build_materialization_probe_result(
    spec: CaseSpec,
    phase_result: dict[str, Any],
) -> dict[str, Any]:
    """Require build preparation to reach a target compiler or artifact."""

    result = dict(phase_result)
    coverage = (
        result.get("translation_unit_coverage")
        if isinstance(result.get("translation_unit_coverage"), dict)
        else {}
    )
    observed_translation_units = {
        normalize_relpath(str(path))
        for key in (
            "succeeded_translation_units",
            "failed_translation_units",
        )
        for path in coverage.get(key) or []
        if str(path)
    }
    driver_prefix = normalize_relpath(spec.driver_root).rstrip("/") + "/"
    entry_files = {
        normalize_relpath(str(path)) for path in spec.entry_files if str(path)
    }
    relevant_translation_units = sorted(
        path
        for path in observed_translation_units
        if path in entry_files or path.startswith(driver_prefix)
    )
    def current_source_path(value: object) -> str:
        path = Path(str(value or ""))
        if path.is_absolute():
            try:
                path = path.relative_to(spec.old_root_path)
            except ValueError:
                return ""
        return normalize_relpath(path)

    issue_translation_units = sorted(
        {
            source
            for issue in result.get("issues") or []
            if isinstance(issue, dict)
            and str(issue.get("kind") or "").startswith("compile_")
            for value in [issue.get("file"), *(issue.get("trigger_files") or [])]
            if (source := current_source_path(value))
            and Path(source).suffix in TRANSLATION_UNIT_SUFFIXES
            and (source in entry_files or source.startswith(driver_prefix))
        }
    )
    materialized_objects = sorted(
        str(candidate)
        for relative_path in spec.expected_objects
        for candidate in _expected_object_candidates(spec, relative_path)
        if candidate.is_file() and candidate.stat().st_size > 0
    )
    observed = bool(
        relevant_translation_units
        or issue_translation_units
        or materialized_objects
    )
    result["build_materialization_probe"] = {
        "schema": "build_materialization_probe_v1",
        "observed": observed,
        "relevant_translation_units": relevant_translation_units,
        "issue_translation_units": issue_translation_units,
        "materialized_objects": materialized_objects,
        "authority": (
            "compile_database_coverage_or_exact_object_artifact"
        ),
    }
    if observed:
        return result
    result["ok"] = False
    if int(result.get("returncode") or 0) == 0:
        result["returncode"] = 1
    issues = [
        dict(issue)
        for issue in result.get("issues") or []
        if isinstance(issue, dict)
    ]
    issues.append(
        {
            "kind": "build_target_not_materialized",
            "severity": "error",
            "repair_relevance": "build_preparation",
            "file": f"{spec.driver_root}/Makefile",
            "phase_group": "build",
            "raw": (
                "build preparation did not observe a compiler invocation for "
                "the declared driver target or a non-empty expected object"
            ),
        }
    )
    result["issues"] = issues
    return result


def _run_make_maintenance_step(
    spec: CaseSpec,
    *,
    artifacts_dir: Path,
    name: str,
    targets: list[str],
    parse_issues: bool,
    failure_label: str,
) -> dict:
    ensure_dir(artifacts_dir)
    make_bin = find_make_binary(spec.make_bin)
    command = [make_bin]
    if spec.build_dir_path is not None:
        command.append(f"O={spec.build_dir_path}")
    command.extend(targets)
    started = time.time()
    result = subprocess.run(
        command,
        cwd=spec.old_root_path,
        env=case_build_env(spec),
        text=True,
        capture_output=True,
    )
    duration = round(time.time() - started, 3)
    output = result.stdout + result.stderr
    log_path = artifacts_dir / f"{name}.log"
    log_path.write_text(output, encoding="utf-8")
    issues = extract_issues(output) if parse_issues else []
    if result.returncode != 0 and not issues:
        tail = _tail_text(output, max_lines=20)
        issues.append(
            {
                "kind": "command_failed",
                "severity": "error",
                "raw": f"{failure_label} failed" + (f"\n{tail}" if tail else ""),
            }
        )
    return {
        "name": name,
        "command": _format_command(command),
        "cwd": str(spec.old_root_path),
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "duration_sec": duration,
        "log_path": str(log_path.resolve()),
        "issues": issues,
    }


def _run_module_clean(spec: CaseSpec, *, artifacts_dir: Path) -> dict | None:
    if not spec.expected_modules and not spec.expected_objects:
        return None
    return _run_make_maintenance_step(
        spec,
        artifacts_dir=artifacts_dir,
        name="module-clean",
        targets=[f"M={spec.driver_root}", "clean"],
        parse_issues=False,
        failure_label="module clean",
    )


def _makefile_uses_config_for_objects(makefile: Path, config_name: str) -> bool:
    try:
        lines = makefile.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    config_ref = f"$({config_name})"
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or config_ref not in stripped:
            continue
        match = re.match(rf"^\s*obj-\$\({re.escape(config_name)}\)\s*(?::=|\+=|=)\s*(.*)$", line)
        if not match:
            continue
        targets = match.group(1).split("#", 1)[0].split()
        if any(target.endswith((".o", ".a")) for target in targets):
            return True
    return False


def _prune_ancestor_module_dirs(directories: list[str]) -> list[str]:
    pruned: list[str] = []
    for directory in directories:
        prefix = directory.rstrip("/") + "/"
        if any(other != directory and other.startswith(prefix) for other in directories):
            continue
        pruned.append(directory)
    return pruned


def _normalize_module_symvers_line(line: str) -> str | None:
    columns = line.rstrip("\r\n").split("\t")
    if len(columns) == 4:
        crc, symbol, module, export = columns
        namespace = ""
    elif len(columns) >= 5:
        crc, symbol, module, export, namespace = columns[:5]
        if any(part for part in columns[5:]):
            return None
    else:
        return None
    crc = crc.strip()
    symbol = symbol.strip()
    module = module.strip()
    export = export.strip()
    namespace = namespace.strip()
    if len(namespace) >= 2 and namespace[0] == namespace[-1] == '"':
        namespace = namespace[1:-1]
    if not re.fullmatch(r"0x[0-9A-Fa-f]+", crc):
        return None
    if not symbol or not module:
        return None
    if export not in {"EXPORT_SYMBOL", "EXPORT_SYMBOL_GPL"}:
        return None
    return "\t".join([crc, symbol, module, export, namespace])


def _normalized_module_symvers_lines(
    raw_lines: list[str],
) -> tuple[list[str], int]:
    normalized_lines: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for line in raw_lines:
        if not line.strip():
            continue
        normalized = _normalize_module_symvers_line(line)
        if normalized is None:
            skipped += 1
            continue
        if normalized not in seen:
            seen.add(normalized)
            normalized_lines.append(normalized)
    return normalized_lines, skipped


def _module_symvers_module_path(line: str) -> str:
    columns = line.split("\t")
    return columns[2] if len(columns) >= 3 else ""


def _rewrite_module_symvers(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _prune_driver_root_module_symvers(
    spec: CaseSpec,
    *,
    artifacts_dir: Path,
    name: str = "module-symvers-prune",
    expected_modules_only: bool = False,
) -> dict | None:
    if not spec.expected_modules:
        return None
    target = (spec.build_dir_path or spec.old_root_path) / "Module.symvers"
    ensure_dir(artifacts_dir)
    log_path = artifacts_dir / "module-symvers-prune.log"
    driver_root = normalize_relpath(spec.driver_root).rstrip("/")
    if not target.is_file():
        log_path.write_text(f"Module.symvers not present: {target}\n", encoding="utf-8")
        return {
            "name": name,
            "command": "sanitize Module.symvers before target module build",
            "cwd": str(spec.old_root_path),
            "returncode": 0,
            "ok": True,
            "duration_sec": 0.0,
            "log_path": str(log_path.resolve()),
            "issues": [],
            "module_symvers": {"target": str(target.resolve()), "reason": "missing"},
        }

    raw_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    deduped, skipped_lines = _normalized_module_symvers_lines(raw_lines)
    prefix = driver_root + "/"
    expected_owners = {
        normalize_relpath(module).removesuffix(".ko")
        for module in spec.expected_modules
    }
    kept = [
        line
        for line in deduped
        if not (
            _module_symvers_module_path(line) in expected_owners
            if expected_modules_only
            else (
                _module_symvers_module_path(line) == driver_root
                or _module_symvers_module_path(line).startswith(prefix)
            )
        )
    ]
    removed = len(deduped) - len(kept)
    changed = raw_lines != kept
    if changed:
        _rewrite_module_symvers(target, kept)
    log_path.write_text(
        "\n".join(
            [
                f"target={target}",
                f"driver_root={driver_root}",
                f"expected_modules_only={expected_modules_only}",
                f"expected_owners={','.join(sorted(expected_owners))}",
                f"line_count_before={len(raw_lines)}",
                f"deduped_line_count={len(deduped)}",
                f"removed_driver_root_lines={removed}",
                f"line_count_after={len(kept)}",
                f"skipped_invalid_lines={skipped_lines}",
                f"changed={changed}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "name": name,
        "command": "sanitize Module.symvers before target module build",
        "cwd": str(spec.old_root_path),
        "returncode": 0,
        "ok": True,
        "duration_sec": 0.0,
        "log_path": str(log_path.resolve()),
        "issues": [],
        "module_symvers": {
            "target": str(target.resolve()),
            "line_count_before": len(raw_lines),
            "line_count_after": len(kept),
            "removed_driver_root_lines": removed,
            "skipped_invalid_lines": skipped_lines,
            "changed": changed,
        },
    }


def _old_root_relative_path(spec: CaseSpec, path: Path) -> str | None:
    try:
        return normalize_relpath(str(path.resolve().relative_to(spec.old_root_path.resolve())))
    except Exception:
        return None


def _exported_symbol_records(
    root_text: str,
    symbol: str,
) -> tuple[tuple[str, int, str], ...]:
    root = Path(root_text)
    symbol = str(symbol or "").strip()
    if not root.is_dir() or not symbol:
        return ()
    needle = symbol.encode("utf-8")
    records: list[tuple[str, int, str]] = []
    root_resolved = root.resolve()
    for current, dirs, files in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name not in EXPORT_SCAN_SKIP_DIRS and not name.startswith(".")
        ]
        current_path = Path(current)
        for name in files:
            if Path(name).suffix not in EXPORT_SCAN_SUFFIXES:
                continue
            path = current_path / name
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if needle not in data:
                continue
            text = data.decode("utf-8", errors="ignore")
            try:
                relative = normalize_relpath(
                    str(path.resolve().relative_to(root_resolved))
                )
            except Exception:
                relative = normalize_relpath(str(path))
            for record in parse_export_contracts(
                text,
                file=relative,
                tree="old",
            ):
                if str(record.get("symbol") or "") == symbol:
                    records.append(
                        (
                            relative,
                            int(record.get("line") or 0),
                            str(record.get("macro") or ""),
                        )
                    )
    return tuple(records)


def _target_export_records_for_symbol(spec: CaseSpec, symbol: str) -> list[dict[str, Any]]:
    return [
        {
            "file": file,
            "line": line,
            "export": export,
        }
        for file, line, export in _exported_symbol_records(str(spec.old_root_path.resolve()), symbol)
    ]


def _annotate_exported_modpost_issues(
    spec: CaseSpec,
    issues: list[dict[str, Any]],
    *,
    dependency_module_dir: str | None = None,
    dependency_build_ok: bool = False,
) -> list[dict[str, Any]]:
    active_symvers = (spec.build_dir_path or spec.old_root_path) / "Module.symvers"
    vmlinux_exports = (
        {
            record["symbol"]: record
            for record in _module_symvers_records(active_symvers)
            if record["module"] == "vmlinux"
        }
        if dependency_module_dir
        and dependency_build_ok
        and any(
            str(issue.get("kind") or "") == "modpost_undefined_symbol"
            for issue in issues
        )
        else {}
    )
    annotated: list[dict[str, Any]] = []
    for issue in issues:
        item = dict(issue)
        if str(item.get("kind") or "") != "modpost_undefined_symbol":
            annotated.append(item)
            continue
        symbol = str(item.get("symbol") or "").strip()
        records = _target_export_records_for_symbol(spec, symbol)
        symvers_witness = vmlinux_exports.get(symbol)
        if not records and not symvers_witness:
            annotated.append(item)
            continue
        provider_resolution = {
            "status": "exported_in_target_tree",
            "symbol": symbol,
            "target_root": str(spec.old_root_path.resolve()),
            "exports": records[:8],
            "export_count": len(records),
        }
        if symvers_witness:
            provider_resolution["module_symvers_witness"] = {
                "path": str(active_symvers.resolve()),
                "symbol": symbol,
                "module": "vmlinux",
                "export": symvers_witness["export"],
            }
        item["provider_resolution"] = provider_resolution
        if dependency_module_dir:
            item["original_kind"] = item.get("kind")
            item["kind"] = "dependency_symvers_boundary"
            item["severity"] = "warning"
            item["dependency_module_dir"] = dependency_module_dir
            item["repair_relevance"] = "validation_environment"
            item["raw"] = (
                f"{item.get('raw')}; provider {symbol} is exported in the target tree, "
                "so this dependency prebuild needs provider Module.symvers/build closure "
                "instead of a driver source repair"
            )
        annotated.append(item)
    return annotated


def _only_target_exported_modpost_issues(issues: list[dict[str, Any]]) -> bool:
    if not issues:
        return False
    exported_boundary_found = False
    for issue in issues:
        if str(issue.get("kind") or "") == "modpost_undefined_symbol":
            resolution = issue.get("provider_resolution")
            if not isinstance(resolution, dict):
                return False
            if resolution.get("status") != "exported_in_target_tree":
                return False
            exported_boundary_found = True
            continue
        if str(issue.get("severity") or "").lower() not in {
            "warning",
            "note",
            "info",
        }:
            return False
    return exported_boundary_found


def _mark_target_symvers_boundary_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    marked: list[dict[str, Any]] = []
    for issue in issues:
        item = dict(issue)
        if (
            str(item.get("kind") or "") == "modpost_undefined_symbol"
            and isinstance(item.get("provider_resolution"), dict)
            and item["provider_resolution"].get("status") == "exported_in_target_tree"
        ):
            item["original_kind"] = item.get("kind")
            item["kind"] = "target_symvers_boundary"
            item["severity"] = "warning"
            item["repair_relevance"] = "validation_environment"
            symbol = str(item.get("symbol") or "").strip()
            item["raw"] = (
                f"{item.get('raw')}; provider {symbol} is exported in the target tree, "
                "so the target module build needs provider Module.symvers/build closure "
                "instead of a driver source repair"
            )
        marked.append(item)
    return marked


def _only_dependency_symvers_boundary_issues(issues: list[dict[str, Any]]) -> bool:
    if not issues:
        return False
    return all(str(issue.get("kind") or "") == "dependency_symvers_boundary" for issue in issues)


def _candidate_makefiles_for_symbol(spec: CaseSpec, record: dict[str, Any]) -> list[Path]:
    """Return Kbuild files structurally anchored by the symbol definition.

    A missing Kconfig definition is not permission to search every Makefile in
    the target kernel. Such a search is expensive and can associate the symbol
    with an unrelated textual occurrence. The caller leaves the provider
    unmapped until later validation supplies an exact build owner.
    """

    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not resolved.is_file():
            return
        seen.add(resolved)
        candidates.append(resolved)

    old_root = spec.old_root_path.resolve()
    for defined in record.get("defined_in") or []:
        path = Path(str(defined))
        try:
            resolved = path.resolve()
            resolved.relative_to(old_root)
        except Exception:
            continue
        directory = resolved.parent
        while True:
            add(directory / "Makefile")
            add(directory / "Kbuild")
            if directory == old_root or directory.parent == directory:
                break
            directory = directory.parent
    return candidates


def _symbol_module_build_dirs(spec: CaseSpec, record: dict[str, Any]) -> list[str]:
    symbol = str(record.get("symbol") or record.get("config") or "").removeprefix("CONFIG_")
    if not symbol:
        return []
    config_name = f"CONFIG_{symbol}"
    dirs: list[str] = []
    seen: set[str] = set()
    for makefile in _candidate_makefiles_for_symbol(spec, record):
        if not _makefile_uses_config_for_objects(makefile, config_name):
            continue
        relative = _old_root_relative_path(spec, makefile.parent)
        if not relative or relative == "." or relative in seen:
            continue
        seen.add(relative)
        dirs.append(relative)
    return _prune_ancestor_module_dirs(dirs)


def _dependency_module_build_plans(spec: CaseSpec, sync_result: dict | None) -> list[dict[str, Any]]:
    if not sync_result:
        return []
    closure = sync_result.get("dependency_closure")
    if not isinstance(closure, dict):
        return []
    seed_configs = {
        str(item)
        for item in closure.get("seed_configs") or []
        if str(item)
    }
    driver_root = normalize_relpath(spec.driver_root)
    plans_by_dir: dict[str, dict[str, Any]] = {}
    for record in closure.get("symbols") or []:
        if not isinstance(record, dict) or str(record.get("value") or "") != "m":
            continue
        config = str(record.get("config") or "")
        if not config:
            symbol = str(record.get("symbol") or "").removeprefix("CONFIG_")
            config = f"CONFIG_{symbol}" if symbol else ""
        if config in seed_configs:
            continue
        for directory in _symbol_module_build_dirs(spec, record):
            plan = plans_by_dir.setdefault(
                directory,
                {
                    "directory": directory,
                    "configs": [],
                    "sources": [],
                    "modules": [],
                    "module_sources": {},
                    "mapping_complete": True,
                },
            )
            if config and config not in plan["configs"]:
                plan["configs"].append(config)

            kbuild = KbuildIndex.from_directory(
                spec.old_root_path,
                directory,
            )
            graph = kbuild.build_contract_graph()
            config_name = config.removeprefix("CONFIG_")
            sources = [
                normalize_relpath(str(source))
                for source in (graph.get("source_to_configs") or {})
                if config_name in set((graph.get("source_to_configs") or {}).get(source) or [])
            ]
            modules = sorted({
                str(Path(target).with_suffix(""))
                for assignment in kbuild.top_assignments
                if assignment.config == config
                for target in assignment.object_items
                if target.endswith(".o")
            })
            if not sources or not modules:
                plan["mapping_complete"] = False
            for source in sources:
                if source not in plan["sources"]:
                    plan["sources"].append(source)
            for module in modules:
                if module not in plan["modules"]:
                    plan["modules"].append(module)
                module_sources = plan["module_sources"].setdefault(module, [])
                for source in (graph.get("module_to_sources") or {}).get(module) or []:
                    normalized_source = normalize_relpath(str(source))
                    if normalized_source not in module_sources:
                        module_sources.append(normalized_source)

    def sort_key(directory: str) -> tuple[int, int, str]:
        related_to_driver = directory == driver_root or driver_root.startswith(directory.rstrip("/") + "/")
        return (1 if related_to_driver else 0, len(Path(directory).parts), directory)

    plans = [plans_by_dir[directory] for directory in sorted(plans_by_dir, key=sort_key)]
    for plan in plans:
        plan["configs"].sort()
        plan["sources"].sort()
        plan["modules"].sort()
        plan["module_sources"] = {
            module: sorted(sources)
            for module, sources in sorted(plan["module_sources"].items())
        }
    return plans


def _module_symvers_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    records: list[dict[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line in lines:
        normalized = _normalize_module_symvers_line(line)
        if normalized is None:
            continue
        crc, symbol, module, export, namespace = normalized.split("\t")
        records.append(
            {
                "crc": crc,
                "symbol": symbol,
                "module": module,
                "export": export,
                "namespace": namespace,
            }
        )
    return records


def _config_preprocessor_macros(config_path: Path | None, *, module_build: bool) -> dict[str, str]:
    macros: dict[str, str] = {"MODULE": "1"} if module_build else {}
    if config_path is None or not config_path.is_file():
        return macros
    try:
        lines = config_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return macros
    for line in lines:
        unset = re.fullmatch(r"#\s+(CONFIG_[A-Za-z0-9_]+)\s+is not set", line.strip())
        if unset:
            macros.pop(unset.group(1), None)
            macros.pop(f"{unset.group(1)}_MODULE", None)
            continue
        assignment = re.fullmatch(r"(CONFIG_[A-Za-z0-9_]+)=(.*)", line.strip())
        if assignment is None:
            continue
        name, value = assignment.groups()
        if value == "m":
            macros.pop(name, None)
            macros[f"{name}_MODULE"] = "1"
        elif value == "n":
            macros.pop(name, None)
            macros.pop(f"{name}_MODULE", None)
        else:
            macros[name] = value
            macros.pop(f"{name}_MODULE", None)
    return macros


def _macro_defined(name: str, macros: dict[str, str]) -> bool | None:
    if name in macros:
        return True
    if name == "MODULE" or name.startswith("CONFIG_"):
        return False
    return None


def _eval_cpp_condition(expression: str, macros: dict[str, str]) -> bool | None:
    expr = expression.split("//", 1)[0].strip()

    def replace_defined(match: re.Match[str]) -> str:
        value = _macro_defined(match.group(1) or match.group(2), macros)
        return "1" if value is True else "0" if value is False else "UNKNOWN"

    expr = re.sub(
        r"\bdefined\s*(?:\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)|\s+([A-Za-z_][A-Za-z0-9_]*))",
        replace_defined,
        expr,
    )

    def replace_config_predicate(match: re.Match[str]) -> str:
        predicate, name = match.groups()
        direct = _macro_defined(name, macros)
        module = _macro_defined(f"{name}_MODULE", macros)
        if predicate == "IS_BUILTIN":
            value = direct
        elif predicate == "IS_REACHABLE":
            value = tri_or(direct, tri_and(module, _macro_defined("MODULE", macros)))
        else:
            value = tri_or(direct, module)
        return "1" if value is True else "0" if value is False else "UNKNOWN"

    expr = re.sub(
        r"\b(IS_ENABLED|IS_REACHABLE|IS_BUILTIN)\s*\(\s*(CONFIG_[A-Za-z0-9_]+)\s*\)",
        replace_config_predicate,
        expr,
    )
    token_re = re.compile(r"UNKNOWN|0[xX][0-9A-Fa-f]+|\d+|&&|\|\||!|\(|\)|[A-Za-z_][A-Za-z0-9_]*")
    tokens = token_re.findall(expr)
    if "".join(tokens) != re.sub(r"\s+", "", expr):
        return None
    position = 0

    def primary() -> bool | None:
        nonlocal position
        if position >= len(tokens):
            return None
        token = tokens[position]
        if token == "!":
            position += 1
            return tri_not(primary())
        if token == "(":
            position += 1
            value = parse_or()
            if position >= len(tokens) or tokens[position] != ")":
                return None
            position += 1
            return value
        position += 1
        if token == "UNKNOWN":
            return None
        if re.fullmatch(r"0[xX][0-9A-Fa-f]+|\d+", token):
            try:
                return int(token, 0) != 0
            except ValueError:
                return None
        defined = _macro_defined(token, macros)
        if defined is not True:
            return defined
        value = str(macros.get(token) or "1").strip()
        try:
            return int(value, 0) != 0
        except ValueError:
            return True

    def parse_and() -> bool | None:
        nonlocal position
        value = primary()
        while position < len(tokens) and tokens[position] == "&&":
            position += 1
            value = tri_and(value, primary())
        return value

    def parse_or() -> bool | None:
        nonlocal position
        value = parse_and()
        while position < len(tokens) and tokens[position] == "||":
            position += 1
            value = tri_or(value, parse_and())
        return value

    result = parse_or()
    return result if position == len(tokens) else None


def _source_export_analysis(
    root: Path,
    sources: list[str],
    *,
    config_path: Path | None,
    module_build: bool,
) -> dict[str, Any]:
    base_macros = _config_preprocessor_macros(config_path, module_build=module_build)
    source_exports: dict[str, list[str]] = {}
    inactive_exports: dict[str, list[str]] = {}
    unresolved_exports: dict[str, list[str]] = {}
    unreadable_sources: list[str] = []
    for source in sources:
        path = root / normalize_relpath(source)
        if not path.is_file():
            source_exports[source] = []
            unreadable_sources.append(source)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            source_exports[source] = []
            unreadable_sources.append(source)
            continue
        macros = dict(base_macros)
        statuses: list[bool | None] = []
        current: bool | None = True
        frames: list[dict[str, bool | None]] = []
        for line in text.splitlines(keepends=True):
            statuses.append(current)
            directive = re.match(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)$", line)
            if directive is None:
                continue
            keyword, argument = directive.groups()
            argument = argument.strip()
            if keyword in {"if", "ifdef", "ifndef"}:
                if keyword == "if":
                    condition = _eval_cpp_condition(argument, macros)
                else:
                    name = argument.split()[0] if argument else ""
                    condition = _macro_defined(name, macros) if name else None
                    if keyword == "ifndef":
                        condition = tri_not(condition)
                frames.append({"parent": current, "taken": condition})
                current = tri_and(current, condition)
            elif keyword == "elif" and frames:
                frame = frames[-1]
                condition = _eval_cpp_condition(argument, macros)
                branch = tri_and(tri_not(frame["taken"]), condition)
                current = tri_and(frame["parent"], branch)
                frame["taken"] = tri_or(frame["taken"], condition)
            elif keyword == "else" and frames:
                frame = frames[-1]
                current = tri_and(frame["parent"], tri_not(frame["taken"]))
                frame["taken"] = True
            elif keyword == "endif" and frames:
                current = frames.pop()["parent"]
            elif keyword in {"define", "undef"} and current is True:
                name_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\b(?:\s+(.*))?", argument)
                if name_match:
                    name, value = name_match.groups()
                    if keyword == "define":
                        macros[name] = str(value or "1").strip()
                    else:
                        macros.pop(name, None)

        active: set[str] = set()
        inactive: set[str] = set()
        unresolved: set[str] = set()
        for export in parse_export_contracts(text, file=source, tree="old"):
            line_index = int(export["line"]) - 1
            status = statuses[line_index] if line_index < len(statuses) else None
            target = active if status is True else inactive if status is False else unresolved
            target.add(str(export["symbol"]))
        source_exports[source] = sorted(active)
        if inactive:
            inactive_exports[source] = sorted(inactive)
        if unresolved:
            unresolved_exports[source] = sorted(unresolved)
    return {
        "source_exports": source_exports,
        "inactive_source_exports": inactive_exports,
        "unresolved_source_exports": unresolved_exports,
        "unreadable_sources": unreadable_sources,
        "preprocessor_complete": not unreadable_sources and not unresolved_exports,
    }


def _provider_symvers_satisfaction(
    spec: CaseSpec,
    plan: dict[str, Any],
    top_module_symvers: Path,
) -> dict[str, Any]:
    sources = [normalize_relpath(str(item)) for item in plan.get("sources") or []]
    export_analysis = _source_export_analysis(
        spec.old_root_path,
        sources,
        config_path=spec.build_config_path,
        module_build=bool(spec.expected_modules),
    )
    source_exports = export_analysis["source_exports"]
    exported_symbols = sorted(
        {
            symbol
            for symbols in source_exports.values()
            for symbol in symbols
        }
    )
    records = _module_symvers_records(top_module_symvers)
    active_symvers_symbols = {record["symbol"] for record in records}
    symbols_by_owner: dict[str, set[str]] = {}
    for record in records:
        symbols_by_owner.setdefault(record["module"], set()).add(record["symbol"])
    vmlinux_symbols = {
        record["symbol"]
        for record in records
        if record.get("module") == "vmlinux"
    }
    modules = [str(item) for item in plan.get("modules") or []]
    module_sources = {
        str(module): [normalize_relpath(str(source)) for source in module_source_list]
        for module, module_source_list in (plan.get("module_sources") or {}).items()
        if isinstance(module_source_list, list)
    }
    if len(modules) == 1 and modules[0] not in module_sources:
        module_sources[modules[0]] = sources
    module_exported_symbols = {
        module: sorted(
            {
                symbol
                for source in module_sources.get(module, [])
                for symbol in source_exports.get(source, [])
            }
        )
        for module in modules
    }
    module_symvers_owners = {
        module: [
            normalize_relpath(f"{plan['directory']}/{target.removesuffix('.ko')}")
            for target in _module_targets_for_directory(str(plan["directory"]), [module])
        ]
        for module in modules
    }
    exporting_modules = [
        module for module in modules if module_exported_symbols.get(module)
    ]
    symvers_witnesses_by_module = {
        module: [
            symbol
            for symbol in module_exported_symbols[module]
            if any(
                symbol in symbols_by_owner.get(owner, set())
                for owner in module_symvers_owners[module]
            )
        ]
        for module in exporting_modules
    }
    modules_without_symvers_witness = [
        module
        for module in exporting_modules
        if set(symvers_witnesses_by_module[module]) != set(module_exported_symbols[module])
    ]
    mapping_complete = (
        bool(plan.get("mapping_complete"))
        and bool(sources)
        and bool(export_analysis["preprocessor_complete"])
        and all(module_sources.get(module) for module in modules)
        and all(len(module_symvers_owners[module]) == 1 for module in modules)
    )
    modules_requiring_build = (
        modules_without_symvers_witness
        if mapping_complete
        else modules
    )
    no_exported_provider_contract = bool(
        mapping_complete and modules and not exporting_modules
    )
    satisfied = bool(
        no_exported_provider_contract
        or (mapping_complete and not modules_requiring_build)
    )
    return {
        "status": (
            "no_exported_provider_contract"
            if no_exported_provider_contract
            else "satisfied_by_active_symvers"
            if satisfied
            else "build_required"
        ),
        "satisfied": satisfied,
        "mapping_complete": mapping_complete,
        "module_symvers": str(top_module_symvers.resolve()),
        "configs": list(plan.get("configs") or []),
        "modules": modules,
        "module_sources": module_sources,
        "module_exported_symbols": module_exported_symbols,
        "module_symvers_owners": module_symvers_owners,
        "exporting_modules": exporting_modules,
        "symvers_witnesses_by_module": symvers_witnesses_by_module,
        "modules_without_symvers_witness": modules_without_symvers_witness,
        "modules_requiring_build": modules_requiring_build,
        "sources": sources,
        "source_exports": source_exports,
        "inactive_source_exports": export_analysis["inactive_source_exports"],
        "unresolved_source_exports": export_analysis["unresolved_source_exports"],
        "unreadable_sources": export_analysis["unreadable_sources"],
        "preprocessor_complete": export_analysis["preprocessor_complete"],
        "exported_symbols": exported_symbols,
        "active_symvers_matched_symbols": [
            symbol for symbol in exported_symbols
            if symbol in active_symvers_symbols
        ],
        "vmlinux_matched_symbols": [
            symbol for symbol in exported_symbols
            if symbol in vmlinux_symbols
        ],
        "missing_from_active_symvers": sorted(
            set(exported_symbols)
            - {symbol for witnesses in symvers_witnesses_by_module.values() for symbol in witnesses}
        ),
    }


def _merge_module_symvers(
    target: Path,
    source: Path,
    *,
    allowed_modules: set[str],
) -> dict[str, Any]:
    if not source.is_file():
        return {"source": str(source), "merged": False, "reason": "missing_source"}
    try:
        raw_source_lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"source": str(source), "merged": False, "reason": str(exc)}
    source_lines, skipped_lines = _normalized_module_symvers_lines(
        raw_source_lines
    )
    source_lines = [
        line for line in source_lines
        if _module_symvers_module_path(line) in allowed_modules
    ]
    if not source_lines:
        reason = "no_matching_module_exports" if raw_source_lines else "empty_source"
        return {"source": str(source), "merged": False, "reason": reason, "skipped_lines": skipped_lines}

    existing_lines: list[str] = []
    target_needs_rewrite = False
    if target.is_file():
        raw_target_lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        existing_lines, _ = (
            _normalized_module_symvers_lines(raw_target_lines)
        )
        target_needs_rewrite = raw_target_lines != existing_lines
    seen = set(existing_lines)
    added = [line for line in source_lines if line not in seen]
    if not added:
        if target_needs_rewrite:
            ensure_dir(target.parent)
            _rewrite_module_symvers(target, existing_lines)
            return {
                "source": str(source),
                "merged": False,
                "reason": "sanitized_existing",
                "added_lines": 0,
                "skipped_lines": skipped_lines,
            }
        return {
            "source": str(source),
            "merged": False,
            "reason": "already_present",
            "added_lines": 0,
            "skipped_lines": skipped_lines,
        }
    ensure_dir(target.parent)
    _rewrite_module_symvers(target, [*existing_lines, *added])
    return {"source": str(source), "merged": True, "added_lines": len(added), "skipped_lines": skipped_lines}


def _run_dependency_module_builds(
    spec: CaseSpec,
    sync_result: dict | None,
    *,
    artifacts_dir: Path,
) -> list[dict]:
    if not spec.expected_modules:
        return []
    build_plans = _dependency_module_build_plans(spec, sync_result)
    if not build_plans:
        return []
    ensure_dir(artifacts_dir)
    results: list[dict] = []
    make_bin = find_make_binary(spec.make_bin)
    top_module_symvers = (spec.build_dir_path or spec.old_root_path) / "Module.symvers"
    for plan in build_plans:
        directory = str(plan["directory"])
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", directory).strip("-") or "dependency"
        provider_satisfaction = _provider_symvers_satisfaction(
            spec,
            plan,
            top_module_symvers,
        )
        log_path = artifacts_dir / f"dependency-modules-{safe_name}.log"
        if provider_satisfaction["satisfied"]:
            log_path.write_text(
                "dependency provider build skipped: every exporting Kbuild module has "
                "an exact export witness in the active Module.symvers\n"
                f"directory={directory}\n"
                f"configs={','.join(provider_satisfaction['configs'])}\n"
                f"sources={','.join(provider_satisfaction['sources'])}\n"
                f"witnesses={json.dumps(provider_satisfaction['symvers_witnesses_by_module'], sort_keys=True)}\n"
                f"exports_not_in_active_symvers={','.join(provider_satisfaction['missing_from_active_symvers'])}\n",
                encoding="utf-8",
            )
            results.append(
                {
                    "name": f"dependency-modules-{safe_name}",
                    "command": "reuse provider exports from active Module.symvers",
                    "cwd": str(spec.old_root_path),
                    "returncode": 0,
                    "ok": True,
                    "duration_sec": 0.0,
                    "log_path": str(log_path.resolve()),
                    "issues": [],
                    "dependency_module_dir": directory,
                    "dependency_build_plan": plan,
                    "skipped_build": True,
                    "symvers_boundary_only": True,
                    "provider_symvers_satisfaction": provider_satisfaction,
                    "module_symvers": {
                        "target": str(top_module_symvers.resolve()),
                        "merge_results": [],
                    },
                }
            )
            continue
        command = [make_bin]
        if spec.build_dir_path is not None:
            command.append(f"O={spec.build_dir_path}")
        modules_requiring_build = list(
            provider_satisfaction.get("modules_requiring_build") or []
        )
        exact_targets = _module_targets_for_directory(
            directory,
            modules_requiring_build,
        )
        if not exact_targets:
            issue = {
                "kind": "dependency_module_target_unresolved",
                "severity": "error",
                "repair_relevance": "validation_environment",
                "dependency_module_dir": directory,
                "raw": (
                    "dependency Kbuild plan did not resolve an exact module "
                    f"target for {directory}"
                ),
            }
            log_path.write_text(issue["raw"] + "\n", encoding="utf-8")
            results.append(
                {
                    "name": f"dependency-modules-{safe_name}",
                    "command": "resolve exact dependency module targets",
                    "cwd": str(spec.old_root_path),
                    "returncode": 1,
                    "ok": False,
                    "duration_sec": 0.0,
                    "log_path": str(log_path.resolve()),
                    "issues": [issue],
                    "dependency_module_dir": directory,
                    "dependency_build_plan": plan,
                    "exact_module_targets": [],
                    "skipped_build": True,
                    "provider_symvers_satisfaction": provider_satisfaction,
                }
            )
            continue
        command.extend(
            normalize_relpath(f"{directory}/{target}")
            for target in exact_targets
        )
        env = _enable_make_keep_going(case_build_env(spec))
        env.setdefault("KBUILD_MODPOST_WARN", "1")
        started = time.time()
        result = subprocess.run(
            command,
            cwd=spec.old_root_path,
            env=env,
            text=True,
            capture_output=True,
        )
        duration = round(time.time() - started, 3)
        output = result.stdout + result.stderr
        log_path.write_text(output, encoding="utf-8")
        issues = _annotate_exported_modpost_issues(
            spec,
            extract_issues(output),
            dependency_module_dir=directory,
            dependency_build_ok=result.returncode == 0,
        )
        issues = [
            {
                **issue,
                "dependency_module_dir": directory,
            }
            for issue in issues
        ]
        if result.returncode != 0 and not issues:
            tail = _tail_text(output, max_lines=20)
            issues.append(
                {
                    "kind": "command_failed",
                    "severity": "error",
                    "repair_relevance": "validation_environment",
                    "raw": f"dependency module build failed for {directory}" + (f"\n{tail}" if tail else ""),
                }
            )
        symvers_boundary_only = _only_dependency_symvers_boundary_issues(issues)
        expected_owners = {
            owner
            for module in modules_requiring_build
            for owner in provider_satisfaction["module_symvers_owners"].get(module, [])
        }
        symvers_candidates = [
            (spec.build_dir_path or spec.old_root_path) / "modules-only.symvers",
        ]
        if spec.build_dir_path is not None:
            symvers_candidates.append(spec.build_dir_path / directory / "Module.symvers")
        symvers_candidates.append(spec.old_root_path / directory / "Module.symvers")
        merge_results = (
            [
                _merge_module_symvers(
                    top_module_symvers,
                    candidate,
                    allowed_modules=expected_owners,
                )
                for candidate in dict.fromkeys(symvers_candidates)
                if candidate != top_module_symvers
            ]
            if result.returncode == 0 else []
        )
        results.append(
            {
                "name": f"dependency-modules-{safe_name}",
                "command": _format_command(command),
                "cwd": str(spec.old_root_path),
                "returncode": result.returncode,
                "ok": result.returncode == 0,
                "duration_sec": duration,
                "log_path": str(log_path.resolve()),
                "issues": issues,
                "dependency_module_dir": directory,
                "dependency_build_plan": plan,
                "exact_module_targets": exact_targets,
                "skipped_build": False,
                "provider_symvers_satisfaction": provider_satisfaction,
                "dependency_build_env": {
                    "KBUILD_MODPOST_WARN": env.get("KBUILD_MODPOST_WARN"),
                    "MAKEFLAGS": env.get("MAKEFLAGS"),
                },
                "symvers_boundary_only": symvers_boundary_only,
                "module_symvers": {
                    "target": str(top_module_symvers.resolve()),
                    "merge_results": merge_results,
                },
            }
        )
    return results


def _run_modules_prepare(spec: CaseSpec, *, artifacts_dir: Path) -> dict | None:
    if not spec.expected_modules:
        return None
    return _run_make_maintenance_step(
        spec,
        artifacts_dir=artifacts_dir,
        name="modules-prepare",
        targets=["prepare", "modules_prepare"],
        parse_issues=True,
        failure_label="modules_prepare",
    )


def run_validation(
    spec: CaseSpec,
    *,
    artifacts_dir: Path,
    build_preparation_only: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    phase_results: list[dict] = []

    def notify(message: str) -> None:
        if progress:
            progress(message)

    def notify_done(result: dict) -> None:
        notify(
            "validation phase done: "
            f"{result.get('name')} ok={bool(result.get('ok'))} "
            f"returncode={result.get('returncode')} "
            f"duration={result.get('duration_sec')}s "
            f"log={result.get('log_path')}"
        )

    def report(overall_ok: bool) -> dict:
        return {
            "case_id": spec.case_id,
            "generated_at": utc_now(),
            "overall_ok": overall_ok,
            "phases": phase_results,
            "summary": _summarize_results(phase_results),
        }

    def append_result(
        result: dict | None,
        *,
        phase_group: str | None = None,
    ) -> dict | None:
        if result is None:
            return None
        notify_done(result)
        if phase_group is not None:
            result["issues"] = [
                _enrich_issue(issue, phase_group=phase_group)
                for issue in result.get("issues") or []
            ]
        phase_results.append(result)
        return None if result["ok"] else report(False)

    for group_name, phases in (("build", spec.build_phases), ("load", spec.load_phases)):
        group_dir = ensure_dir(artifacts_dir / group_name)
        if group_name == "build":
            if spec.build_kconfig:
                notify(f"validation phase start: kconfig-sync log={group_dir / 'kconfig-sync.log'}")
            sync_result = sync_build_kconfig(spec, artifacts_dir=group_dir)
            failure = append_result(sync_result)
            if failure is not None:
                return failure
            if spec.expected_modules:
                notify(f"validation phase start: modules-prepare log={group_dir / 'modules-prepare.log'}")
            prepare_result = _run_modules_prepare(spec, artifacts_dir=group_dir)
            failure = append_result(
                prepare_result,
                phase_group=group_name,
            )
            if failure is not None:
                return failure
            if spec.expected_modules or spec.expected_objects:
                notify(f"validation phase start: module-clean log={group_dir / 'module-clean.log'}")
            clean_result = _run_module_clean(spec, artifacts_dir=group_dir)
            failure = append_result(clean_result)
            if failure is not None:
                return failure
            pre_dependency_symvers_prune = _prune_driver_root_module_symvers(
                spec,
                artifacts_dir=group_dir,
                name="module-symvers-prune-pre-dependencies",
            )
            failure = append_result(pre_dependency_symvers_prune)
            if failure is not None:
                return failure
            dependency_results = _run_dependency_module_builds(
                spec,
                sync_result,
                artifacts_dir=group_dir,
            )
            for dependency_result in dependency_results:
                failure = append_result(
                    dependency_result,
                    phase_group=group_name,
                )
                if failure is not None:
                    return failure
            symvers_prune_result = _prune_driver_root_module_symvers(
                spec,
                artifacts_dir=group_dir,
                name="module-symvers-prune-post-dependencies",
                expected_modules_only=True,
            )
            failure = append_result(symvers_prune_result)
            if failure is not None:
                return failure
            if build_preparation_only and not phases:
                missing_phase = {
                    "name": "build-materialization-probe",
                    "command": "<missing build phase>",
                    "cwd": str(spec.old_root_path),
                    "returncode": 1,
                    "ok": False,
                    "duration_sec": 0.0,
                    "log_path": "",
                    "issues": [
                        {
                            "kind": "build_target_not_materialized",
                            "severity": "error",
                            "repair_relevance": "build_preparation",
                            "file": f"{spec.driver_root}/Makefile",
                            "raw": "case has no build phase for target materialization",
                        }
                    ],
                }
                append_result(missing_phase, phase_group=group_name)
                return report(False)
        for phase in phases:
            notify(
                "validation phase start: "
                f"{phase.name} command={_phase_command(spec, phase)} "
                f"log={group_dir / f'{phase.name}.log'}"
            )
            phase_result = _run_phase(
                spec,
                phase,
                artifacts_dir=group_dir,
                collect_compile_coverage=group_name == "build",
            )
            if group_name == "build" and build_preparation_only:
                phase_result = _build_materialization_probe_result(
                    spec,
                    phase_result,
                )
            failure = append_result(
                phase_result,
                phase_group=group_name,
            )
            if failure is not None:
                return failure
            if group_name == "build" and build_preparation_only:
                return report(True)
        if group_name == "build":
            if spec.expected_modules:
                notify(f"validation phase start: expected-modules log={group_dir / 'expected-modules.log'}")
            module_result = _check_expected_modules(spec, artifacts_dir=group_dir)
            failure = append_result(module_result)
            if failure is not None:
                return failure
            if spec.expected_objects:
                notify(f"validation phase start: expected-objects log={group_dir / 'expected-objects.log'}")
            object_result = _check_expected_objects(spec, artifacts_dir=group_dir)
            failure = append_result(object_result)
            if failure is not None:
                return failure
    return report(all(item["ok"] for item in phase_results))


def _summarize_results(results: list[dict]) -> dict:
    issue_kinds = Counter()
    failed_phase = None
    for result in results:
        if not result["ok"] and failed_phase is None:
            failed_phase = result["name"]
        issue_kinds.update(issue["kind"] for issue in result["issues"])
    return {
        "phase_count": len(results),
        "failed_phase": failed_phase,
        "issue_kinds": dict(issue_kinds),
    }
