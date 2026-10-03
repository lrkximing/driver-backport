#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable

from driver_backport.core.case_spec import parse_path_maps
from driver_backport.core.common import ensure_dir, normalize_relpath
from driver_backport.scope.compile_commands import CompileCommands
from driver_backport.validation.build_utils import find_make_binary


_KBUILD_COMPILE_COMMAND_RE = re.compile(
    r"^(?:saved)?cmd_[^ ]*\.o := "
    r"(?P<command_prefix>.* )(?P<source_file>[^ ]*\.[cS]) *(?:;|$)"
)


def _load_commands(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"compile_commands must contain a JSON list: {path}")
    return [dict(item) for item in payload if isinstance(item, dict)]


def _relative_source(
    raw_file: object,
    roots: Iterable[Path],
    *,
    path_maps: Iterable[tuple[str, str]] = (),
) -> str | None:
    text = str(raw_file or "").strip()
    if not text:
        return None
    text = _replace_paths(text, list(path_maps))
    path = Path(text)
    if not path.is_absolute():
        try:
            return normalize_relpath(text)
        except ValueError:
            return None
    for root in roots:
        try:
            return normalize_relpath(path.resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            continue
    return None


def _replace_paths(value: str, replacements: list[tuple[str, str]]) -> str:
    result = str(value)
    for old, new in replacements:
        if old:
            result = result.replace(old, new)
    return result


def _drop_missing_target_forced_includes(
    arguments: list[str],
    *,
    old_root: Path,
) -> tuple[list[str], list[str]]:
    """Remove donor-only forced headers that do not exist in the target tree."""

    retained: list[str] = []
    omitted: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument != "-include" or index + 1 >= len(arguments):
            retained.append(argument)
            index += 1
            continue
        include = arguments[index + 1]
        include_path = Path(include)
        try:
            under_target = (
                include_path.is_absolute()
                and include_path.is_relative_to(old_root)
            )
        except ValueError:
            under_target = False
        if under_target and not include_path.is_file():
            omitted.append(include)
        else:
            retained.extend((argument, include))
        index += 2
    return retained, omitted


def _normalize_target_command(
    command: dict[str, Any],
    *,
    old_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    normalized = dict(command)
    omitted: list[str] = []
    if isinstance(command.get("arguments"), list):
        normalized["arguments"], argument_omitted = (
            _drop_missing_target_forced_includes(
                [str(argument) for argument in command.get("arguments") or []],
                old_root=old_root,
            )
        )
        omitted.extend(argument_omitted)
    if command.get("command") is not None:
        command_arguments, command_omitted = (
            _drop_missing_target_forced_includes(
                shlex.split(str(command["command"])),
                old_root=old_root,
            )
        )
        normalized["command"] = shlex.join(command_arguments)
        omitted.extend(command_omitted)
    return normalized, sorted(set(omitted))


def _rewrite_exact_donor_command(
    command: dict[str, Any],
    *,
    source_file: str,
    donor_root: Path,
    old_root: Path,
    build_dir: Path,
    donor_compile_commands: Path,
) -> dict[str, Any]:
    donor_directory = str(command.get("directory") or "")
    raw_donor_file = str(command.get("file") or "")
    entry_suffix = "/" + source_file
    command_donor_root = (
        raw_donor_file[: -len(entry_suffix)]
        if raw_donor_file.endswith(entry_suffix)
        else ""
    )
    replacements = [
        (str(donor_root.resolve()), str(old_root.resolve())),
        (command_donor_root, str(old_root.resolve())),
        (donor_directory, str(build_dir.resolve())),
    ]
    rewritten = dict(command)
    if isinstance(command.get("arguments"), list):
        rewritten["arguments"] = [
            _replace_paths(str(argument), replacements)
            for argument in command.get("arguments") or []
        ]
    if command.get("command") is not None:
        rewritten["command"] = _replace_paths(
            str(command["command"]), replacements
        )
    rewritten, omitted_forced_includes = _normalize_target_command(
        rewritten,
        old_root=old_root.resolve(),
    )
    rewritten["directory"] = str(build_dir.resolve())
    rewritten["file"] = str((old_root / source_file).resolve())
    rewritten["driver_backport_context_origin"] = {
        "schema": "transplanted_compile_context_v3",
        "kind": "donor_source_command_rebased_to_target_context",
        "source_file": source_file,
        "donor_root": str(donor_root.resolve()),
        "old_root": str(old_root.resolve()),
        "donor_compile_commands": str(donor_compile_commands.resolve()),
        "omitted_missing_target_forced_includes": sorted(
            set(omitted_forced_includes)
        ),
    }
    return rewritten


def exact_target_kbuild_command(
    *,
    source_file: str,
    build_dir: Path,
    old_root: Path,
    target_build_dir: Path,
) -> dict[str, Any] | None:
    source_path = Path(source_file)
    object_path = source_path.with_suffix(".o")
    command_path = (
        target_build_dir
        / object_path.parent
        / f".{object_path.name}.cmd"
    )
    if not command_path.is_file():
        return None
    try:
        first_line = command_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()[0]
    except (OSError, IndexError):
        return None
    match = _KBUILD_COMPILE_COMMAND_RE.match(first_line)
    if match is None:
        return None
    source_suffix = "/" + source_file
    raw_source = Path(match.group("source_file"))
    raw_source_text = str(raw_source)
    if not raw_source_text.endswith(source_suffix):
        try:
            relative_raw_source = normalize_relpath(raw_source_text)
        except ValueError:
            return None
        if relative_raw_source != source_file:
            return None
    source_candidates = (
        [raw_source.resolve()]
        if raw_source.is_absolute()
        else [
            (target_build_dir / raw_source).resolve(),
            (old_root / raw_source).resolve(),
            (old_root / source_file).resolve(),
        ]
    )
    realized_source = next(
        (
            candidate
            for candidate in source_candidates
            if candidate.is_file()
            and str(candidate).endswith(source_suffix)
        ),
        None,
    )
    if realized_source is None:
        return None
    source_root = Path(
        str(realized_source)[: -len(source_suffix)]
    ).resolve()
    if not (source_root / source_file).is_file():
        return None

    replacements = [
        (str(source_root), str(old_root.resolve())),
        (str(target_build_dir.resolve()), str(build_dir.resolve())),
    ]
    command = _replace_paths(
        match.group("command_prefix") + raw_source_text,
        replacements,
    ).replace("$(pound)", "#")
    rewritten, omitted_forced_includes = _normalize_target_command(
        {
            "directory": str(build_dir.resolve()),
            "file": str((old_root / source_file).resolve()),
            "command": command,
        },
        old_root=old_root.resolve(),
    )
    rewritten["driver_backport_context_origin"] = {
        "schema": "transplanted_compile_context_v3",
        "kind": "exact_target_kbuild_command_rebased_to_work_tree",
        "source_file": source_file,
        "source_root": str(source_root),
        "old_root": str(old_root.resolve()),
        "target_build_dir": str(target_build_dir.resolve()),
        "target_kbuild_command": str(command_path.resolve()),
        "omitted_missing_target_forced_includes": sorted(
            set(omitted_forced_includes)
        ),
    }
    return rewritten


def _compile_command_from_verbose_output(
    output: str,
    *,
    source_file: str,
    build_dir: Path,
    old_root: Path,
    make_command: list[str],
    returncode: int,
) -> dict[str, Any] | None:
    source_suffix = "/" + source_file
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if "-c" not in tokens:
            continue
        source_tokens = [
            token
            for token in tokens
            if token == source_file or token.endswith(source_suffix)
        ]
        if not source_tokens:
            continue
        normalized, omitted_forced_includes = _normalize_target_command(
            {
                "directory": str(build_dir.resolve()),
                "file": str((old_root / source_file).resolve()),
                "command": line,
            },
            old_root=old_root.resolve(),
        )
        normalized["driver_backport_context_origin"] = {
            "schema": "transplanted_compile_context_v4",
            "kind": "exact_target_kbuild_verbose_command",
            "source_file": source_file,
            "old_root": str(old_root.resolve()),
            "build_dir": str(build_dir.resolve()),
            "make_command": shlex.join(make_command),
            "make_returncode": returncode,
            "omitted_missing_target_forced_includes": sorted(
                set(omitted_forced_includes)
            ),
        }
        return normalized
    return None


def _capture_target_kbuild_command(
    raw_case: dict[str, Any],
    *,
    source_file: str,
    build_dir: Path,
    old_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    object_target = Path(source_file).with_suffix(".o").as_posix()
    make_command = [find_make_binary(raw_case.get("make_bin"))]
    if build_dir.resolve() != old_root.resolve():
        make_command.append(f"O={build_dir.resolve()}")
    make_command.extend(["V=1", object_target])
    existing_assignments = {
        token.split("=", 1)[0]
        for token in make_command
        if token.startswith("CONFIG_") and "=" in token
    }
    for raw_assignment in raw_case.get("build_kconfig") or []:
        assignment = str(raw_assignment).strip()
        if not assignment.startswith("CONFIG_") or "=" not in assignment:
            continue
        name = assignment.split("=", 1)[0]
        if name in existing_assignments:
            continue
        make_command.append(assignment)
        existing_assignments.add(name)

    env = os.environ.copy()
    env.update(
        {
            str(key): str(value)
            for key, value in (raw_case.get("build_env") or {}).items()
        }
    )
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    if build_dir.resolve() != old_root.resolve():
        env.setdefault("KCONFIG_CONFIG", str(build_dir / ".config"))
    timeout = max(
        1,
        int(raw_case.get("compile_context_capture_timeout_sec") or 180),
    )
    try:
        process = subprocess.run(
            make_command,
            cwd=old_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        output = process.stdout + process.stderr
        returncode = int(process.returncode)
        error = None
    except subprocess.TimeoutExpired as exc:
        output = str(exc.stdout or "") + str(exc.stderr or "")
        returncode = 124
        error = f"compile_context_capture_timeout_after_{timeout}s"
    except OSError as exc:
        output = ""
        returncode = 127
        error = f"{exc.__class__.__name__}: {exc}"

    command = _target_kbuild_command_for_source(
        source_file=source_file,
        build_dir=build_dir,
        old_root=old_root,
        target_compile_command_build_dirs=[],
    )
    authority = "generated_kbuild_cmd_record"
    if command is None:
        command = _compile_command_from_verbose_output(
            output,
            source_file=source_file,
            build_dir=build_dir,
            old_root=old_root,
            make_command=make_command,
            returncode=returncode,
        )
        authority = "observed_verbose_compiler_invocation"
    lines = [line for line in output.splitlines() if line.strip()]
    return command, {
        "schema": "target_compile_command_capture_v1",
        "source_file": source_file,
        "object_target": object_target,
        "command": shlex.join(make_command),
        "returncode": returncode,
        "compiler_invocation_observed": command is not None,
        "authority": authority if command is not None else None,
        "error": error,
        "output_tail": lines[-40:],
    }


def _target_kbuild_command_for_source(
    *,
    source_file: str,
    build_dir: Path,
    old_root: Path,
    target_compile_command_build_dirs: list[Path],
) -> dict[str, Any] | None:
    current = exact_target_kbuild_command(
        source_file=source_file,
        build_dir=build_dir,
        old_root=old_root,
        target_build_dir=build_dir,
    )
    if current is not None:
        return current
    matches = [
        command
        for target_build_dir in target_compile_command_build_dirs
        for command in [
            exact_target_kbuild_command(
                source_file=source_file,
                build_dir=build_dir,
                old_root=old_root,
                target_build_dir=target_build_dir,
            )
        ]
        if command is not None
    ]
    return matches[0] if len(matches) == 1 else None


def _is_under(relative_path: str, root: str) -> bool:
    relative_path = normalize_relpath(relative_path)
    root = normalize_relpath(root)
    return relative_path == root or relative_path.startswith(root.rstrip("/") + "/")


def materialize_transplanted_compile_context(raw_case: dict[str, Any]) -> dict[str, Any]:
    """Materialize exact per-source compile contexts for the transplanted driver.

    Donor-copied sources retain donor command provenance, while target-existing
    providers use their exact target Kbuild commands.  Both are rebased into the
    active work tree and build directory before CodeIndex parses them.
    """

    target_value = str(raw_case.get("compile_commands") or "").strip()
    if not target_value:
        return {"status": "failed", "stop_reason": "missing_compile_commands_path"}
    target = Path(target_value).expanduser().resolve()
    old_root = Path(str(raw_case.get("old_root") or "")).expanduser().resolve()
    build_dir = Path(str(raw_case.get("build_dir") or old_root)).expanduser().resolve()
    donor_roots = [
        Path(str(item)).expanduser().resolve()
        for item in raw_case.get("donor_roots") or []
        if str(item).strip()
    ]
    donor_command_paths = [
        Path(str(item)).expanduser().resolve()
        for item in raw_case.get("donor_compile_commands") or []
        if str(item).strip()
    ]
    target_compile_command_build_dirs = [
        Path(str(item)).expanduser().resolve()
        for item in raw_case.get("target_compile_command_build_dirs") or []
        if str(item).strip()
    ]
    entry_files = [
        normalize_relpath(str(item))
        for item in raw_case.get("entry_files") or []
        if str(item).strip()
    ]
    driver_root = normalize_relpath(
        str(raw_case.get("driver_root") or (Path(entry_files[0]).parent if entry_files else ""))
    )
    exact_scope_sources = {
        normalize_relpath(str(item))
        for item in raw_case.get("editable_roots") or []
        if Path(normalize_relpath(str(item))).suffix.lower()
        in {".c", ".cc", ".cpp", ".s"}
    }
    required_source_contexts = {*entry_files, *exact_scope_sources}
    old_path_maps = parse_path_maps(
        raw_case.get("compile_path_maps") or raw_case.get("path_maps")
    )
    donor_path_maps = parse_path_maps(raw_case.get("donor_compile_path_maps"))
    if not donor_roots or not donor_command_paths or not entry_files:
        return {
            "status": "failed",
            "stop_reason": "missing_donor_compile_context_contract",
        }

    existing = _load_commands(target)
    current_commands_by_source: dict[str, dict[str, Any]] = {}
    normalized_existing_forced_includes: dict[str, list[str]] = {}
    for command in existing:
        relative = _relative_source(
            command.get("file"),
            [old_root],
            path_maps=old_path_maps,
        )
        if relative and (
            (old_root / relative).is_file()
            or relative in required_source_contexts
        ):
            normalized_command, omitted = _normalize_target_command(
                command,
                old_root=old_root,
            )
            if omitted:
                origin = dict(
                    normalized_command.get(
                        "driver_backport_context_origin"
                    )
                    or {}
                )
                origin.update(
                    {
                        "schema": "transplanted_compile_context_v3",
                        "kind": (
                            origin.get("kind")
                            or "target_compile_command_normalized"
                        ),
                        "omitted_missing_target_forced_includes": sorted(
                            set(
                                list(
                                    origin.get(
                                        "omitted_missing_target_forced_includes"
                                    )
                                    or []
                                )
                                + omitted
                            )
                        ),
                    }
                )
                normalized_command[
                    "driver_backport_context_origin"
                ] = origin
                normalized_existing_forced_includes[relative] = omitted
            current_commands_by_source[relative] = normalized_command

    donor_databases = [
        (path, _load_commands(path))
        for path in donor_command_paths
        if path.is_file()
    ]
    donor_matches_by_source: dict[str, list[tuple[Path, Path, dict[str, Any]]]] = {}
    for database_path, commands in donor_databases:
        for command in commands:
            for donor_root in donor_roots:
                relative = _relative_source(
                    command.get("file"),
                    [donor_root],
                    path_maps=donor_path_maps,
                )
                if not relative or not (
                    _is_under(relative, driver_root)
                    or relative in exact_scope_sources
                ):
                    continue
                donor_matches_by_source.setdefault(relative, []).append(
                    (database_path, donor_root, command)
                )

    desired_sources = sorted(
        {
            *required_source_contexts,
            *(
                source
                for source in donor_matches_by_source
                if (old_root / source).is_file()
            ),
        }
    )
    materialized_by_source: dict[str, dict[str, Any]] = {}
    target_materialized_sources: list[str] = []
    donor_materialized_sources: list[str] = []
    unresolved: list[dict[str, Any]] = []
    ambiguous_supporting_sources: list[dict[str, Any]] = []
    target_command_capture_attempts: list[dict[str, Any]] = []
    for source_file in desired_sources:
        if source_file in current_commands_by_source:
            continue
        target_command = _target_kbuild_command_for_source(
            source_file=source_file,
            build_dir=build_dir,
            old_root=old_root,
            target_compile_command_build_dirs=(
                target_compile_command_build_dirs
            ),
        )
        if target_command is not None:
            materialized_by_source[source_file] = target_command
            target_materialized_sources.append(source_file)
            continue
        if (
            (old_root / source_file).is_file()
            and bool(
                raw_case.get(
                    "capture_missing_target_compile_commands",
                    True,
                )
            )
        ):
            target_command, capture = _capture_target_kbuild_command(
                raw_case,
                source_file=source_file,
                build_dir=build_dir,
                old_root=old_root,
            )
            target_command_capture_attempts.append(capture)
            if target_command is not None:
                materialized_by_source[source_file] = target_command
                target_materialized_sources.append(source_file)
                continue
        matches = donor_matches_by_source.get(source_file, [])
        if len(matches) != 1:
            record = {
                "source_file": source_file,
                "exact_target_command_count": 0,
                "exact_donor_command_count": len(matches),
            }
            if (
                source_file in entry_files
                or source_file in exact_scope_sources
            ):
                unresolved.append(record)
            else:
                ambiguous_supporting_sources.append(record)
            continue
        database_path, donor_root, command = matches[0]
        materialized_by_source[source_file] = _rewrite_exact_donor_command(
            command,
            source_file=source_file,
            donor_root=donor_root,
            old_root=old_root,
            build_dir=build_dir,
            donor_compile_commands=database_path,
        )
        donor_materialized_sources.append(source_file)

    if unresolved:
        return {
            "status": "failed",
            "stop_reason": "exact_source_compile_context_unresolved",
            "target": str(target),
            "unresolved": unresolved,
            "target_command_capture_attempts": (
                target_command_capture_attempts
            ),
        }
    final_commands_by_source = {
        **current_commands_by_source,
        **materialized_by_source,
    }
    existing_architectures = CompileCommands.from_records(
        existing
    ).architecture_contexts()
    final_architectures = CompileCommands.from_records(
        list(final_commands_by_source.values())
    ).architecture_contexts()
    if (
        len(existing_architectures) > 1
        or (
            len(existing_architectures) == 1
            and final_architectures != existing_architectures
        )
    ):
        return {
            "status": "failed",
            "stop_reason": "compile_context_architecture_mismatch",
            "target": str(target),
            "existing_architectures": existing_architectures,
            "materialized_architectures": final_architectures,
        }
    ensure_dir(target.parent)
    target.write_text(
        json.dumps(
            [final_commands_by_source[source] for source in sorted(final_commands_by_source)],
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    materialized_sources = sorted(materialized_by_source)
    materialized_entries = [source for source in entry_files if source in materialized_by_source]
    return {
        "status": (
            "materialized"
            if materialized_sources
            else (
                "normalized"
                if normalized_existing_forced_includes
                else "unchanged"
            )
        ),
        "target": str(target),
        "existing_command_count": len(existing),
        "current_command_count": len(current_commands_by_source),
        "normalized_existing_forced_include_count": len(
            normalized_existing_forced_includes
        ),
        "normalized_existing_forced_includes": (
            normalized_existing_forced_includes
        ),
        "materialized_command_count": len(materialized_sources),
        "materialized_entry_files": materialized_entries,
        "materialized_supporting_files": [
            source for source in materialized_sources if source not in set(entry_files)
        ],
        "materialized_source_files": materialized_sources,
        "target_kbuild_materialized_source_files": sorted(
            target_materialized_sources
        ),
        "target_command_capture_attempts": target_command_capture_attempts,
        "donor_materialized_source_files": sorted(
            donor_materialized_sources
        ),
        "exact_scope_source_files": sorted(exact_scope_sources),
        "ambiguous_supporting_sources": ambiguous_supporting_sources,
        "final_command_count": len(final_commands_by_source),
        "build_dir": str(build_dir),
    }
