#!/usr/bin/env python3
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Iterable

from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import normalize_root_relative


TRANSLATION_UNIT_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".s", ".S"}


def _compile_arguments(entry: dict[str, Any]) -> list[str]:
    arguments = entry.get("arguments")
    if isinstance(arguments, list):
        return [str(item) for item in arguments]
    command = str(entry.get("command") or "")
    if not command:
        return []
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _compile_output_argument(entry: dict[str, Any]) -> str | None:
    explicit = str(entry.get("output") or "").strip()
    if explicit:
        return explicit
    arguments = _compile_arguments(entry)
    for index, argument in enumerate(arguments):
        if argument == "-o" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("-o") and len(argument) > 2:
            return argument[2:]
    return None


def _resolve_compile_path(value: object, *, directory: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = directory / path
    return path.resolve()


def _mapped_compile_path(
    path: Path,
    *,
    path_maps: Iterable[tuple[object, object]],
) -> Path:
    resolved = path.resolve()
    mappings: list[tuple[Path, Path]] = []
    for old_prefix, new_prefix in path_maps:
        old_path = Path(str(old_prefix)).expanduser().resolve()
        new_path = Path(str(new_prefix)).expanduser().resolve()
        mappings.append((old_path, new_path))
    for old_path, new_path in sorted(
        mappings,
        key=lambda item: len(item[0].parts),
        reverse=True,
    ):
        try:
            return (new_path / resolved.relative_to(old_path)).resolve()
        except ValueError:
            continue
    return resolved


def _compile_command_records(spec: CaseSpec) -> list[dict[str, Any]]:
    compile_commands = str(spec.compile_commands or "").strip()
    if not compile_commands:
        return []
    compile_commands_path = Path(compile_commands).expanduser().resolve()
    try:
        payload = json.loads(compile_commands_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []

    old_root = spec.old_root_path
    build_dir_value = spec.build_dir_path
    build_dir = Path(build_dir_value).resolve() if build_dir_value else None
    path_maps = list(spec.compile_path_maps)
    preferred_prefix = spec.driver_root
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        directory = _resolve_compile_path(entry.get("directory"), directory=old_root)
        if directory is None:
            directory = old_root
        source_path = _resolve_compile_path(entry.get("file"), directory=directory)
        output_value = _compile_output_argument(entry)
        output_path = _resolve_compile_path(output_value, directory=directory)
        if source_path is None or output_path is None:
            continue
        source_path = _mapped_compile_path(
            source_path,
            path_maps=path_maps,
        )
        if build_dir is not None and output_value:
            raw_output = Path(output_value).expanduser()
            if raw_output.is_absolute():
                try:
                    output_path = (
                        build_dir / output_path.relative_to(directory)
                    ).resolve()
                except ValueError:
                    output_path = _mapped_compile_path(
                        output_path,
                        path_maps=path_maps,
                    )
            else:
                output_path = (build_dir / raw_output).resolve()
        relative_source = normalize_root_relative(
            source_path,
            root=old_root,
            preferred_prefix=preferred_prefix,
        )
        if not relative_source or source_path.suffix not in TRANSLATION_UNIT_SUFFIXES:
            continue
        identity = (relative_source, str(output_path))
        if identity in seen:
            continue
        seen.add(identity)
        records.append(
            {
                "source": relative_source,
                "source_path": str(source_path),
                "output_path": str(output_path),
            }
        )
    return records


def _normalized_issue_translation_units(
    issues: Iterable[dict[str, Any]],
    *,
    old_root: Path,
    preferred_prefix: str,
) -> set[str]:
    files: set[str] = set()
    for issue in issues:
        relative = normalize_root_relative(
            str(
                issue.get("file")
                or issue.get("source_file")
                or issue.get("path")
                or ""
            ),
            root=old_root,
            preferred_prefix=preferred_prefix,
        )
        if relative and Path(relative).suffix in TRANSLATION_UNIT_SUFFIXES:
            files.add(relative)
    return files


def collect_translation_unit_coverage(
    spec: CaseSpec,
    *,
    issues: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Observe successful compiler outputs without parsing human build progress.

    The compile database provides the source-to-object relation.  A diagnostic
    identifies failed translation units, while a non-empty object is positive
    evidence that the corresponding compiler invocation completed.
    """

    old_root = spec.old_root_path
    preferred_prefix = spec.driver_root
    records = _compile_command_records(spec)
    failed_files = _normalized_issue_translation_units(
        issues,
        old_root=old_root,
        preferred_prefix=preferred_prefix,
    )

    artifacts: list[dict[str, Any]] = []
    statuses_by_source: dict[str, set[str]] = {}
    for record in records:
        source = str(record["source"])
        source_path = Path(str(record["source_path"]))
        output_path = Path(str(record["output_path"]))
        output_exists = output_path.is_file()
        output_size = output_path.stat().st_size if output_exists else 0
        source_mtime_ns = source_path.stat().st_mtime_ns if source_path.is_file() else None
        output_mtime_ns = output_path.stat().st_mtime_ns if output_exists else None
        artifact_current = bool(
            output_exists
            and output_size > 0
            and (
                source_mtime_ns is None
                or output_mtime_ns is None
                or output_mtime_ns >= source_mtime_ns
            )
        )
        status = (
            "failed"
            if source in failed_files
            else "succeeded"
            if artifact_current
            else "unobserved"
        )
        statuses_by_source.setdefault(source, set()).add(status)
        artifacts.append(
            {
                **record,
                "status": status,
                "output_exists": output_exists,
                "output_size": output_size,
                "artifact_current_for_source": artifact_current,
            }
        )

    succeeded: set[str] = set()
    failed: set[str] = set()
    unobserved: set[str] = set()
    for source, statuses in statuses_by_source.items():
        if "failed" in statuses:
            failed.add(source)
        elif "succeeded" in statuses:
            succeeded.add(source)
        else:
            unobserved.add(source)
    all_sources = set(statuses_by_source)
    return {
        "schema": "translation_unit_coverage_v1",
        "source": "compile_commands_output_artifacts",
        "available": bool(records),
        "compile_commands": str(Path(spec.compile_commands).expanduser().resolve())
        if spec.compile_commands
        else None,
        "entry_count": len(records),
        "translation_unit_count": len(all_sources),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "unobserved_count": len(unobserved),
        "all_translation_units_succeeded": bool(all_sources)
        and all_sources == succeeded,
        "succeeded_translation_units": sorted(succeeded),
        "failed_translation_units": sorted(failed),
        "unobserved_translation_units": sorted(unobserved),
        "artifacts": artifacts,
    }


def merged_validation_translation_unit_coverage(
    validation: dict[str, Any],
) -> dict[str, Any]:
    phase_coverages = [
        dict(phase.get("translation_unit_coverage") or {})
        for phase in validation.get("phases") or []
        if isinstance(phase, dict)
        and (phase.get("translation_unit_coverage") or {}).get("available")
    ]
    if not phase_coverages:
        return {
            "schema": "translation_unit_coverage_v1",
            "source": "validation_phase_union",
            "available": False,
            "translation_unit_count": 0,
            "succeeded_translation_units": [],
            "failed_translation_units": [],
            "unobserved_translation_units": [],
            "all_translation_units_succeeded": False,
        }

    all_sources: set[str] = set()
    succeeded: set[str] = set()
    failed: set[str] = set()
    for coverage in phase_coverages:
        all_sources.update(str(item) for item in coverage.get("succeeded_translation_units") or [])
        all_sources.update(str(item) for item in coverage.get("failed_translation_units") or [])
        all_sources.update(str(item) for item in coverage.get("unobserved_translation_units") or [])
        succeeded.update(str(item) for item in coverage.get("succeeded_translation_units") or [])
        failed.update(str(item) for item in coverage.get("failed_translation_units") or [])
    failed -= succeeded
    unobserved = all_sources - succeeded - failed
    return {
        "schema": "translation_unit_coverage_v1",
        "source": "validation_phase_union",
        "available": True,
        "phase_count": len(phase_coverages),
        "translation_unit_count": len(all_sources),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "unobserved_count": len(unobserved),
        "all_translation_units_succeeded": bool(all_sources)
        and all_sources == succeeded,
        "succeeded_translation_units": sorted(succeeded),
        "failed_translation_units": sorted(failed),
        "unobserved_translation_units": sorted(unobserved),
    }


def diagnostic_files_resolution_coverage(
    diagnostic_files: Iterable[object],
    *,
    coverage: dict[str, Any],
    old_root: Path,
    preferred_prefix: str,
) -> dict[str, Any]:
    normalized = {
        relative
        for value in diagnostic_files
        if (
            relative := normalize_root_relative(
                str(value or ""),
                root=old_root,
                preferred_prefix=preferred_prefix,
            )
        )
    }
    translation_units = {
        item
        for item in normalized
        if Path(item).suffix in TRANSLATION_UNIT_SUFFIXES
    }
    succeeded = {
        str(item)
        for item in coverage.get("succeeded_translation_units") or []
    }
    available = bool(coverage.get("available"))
    if translation_units:
        covered = available and translation_units <= succeeded
        reason = (
            "every diagnostic translation unit produced a current object"
            if covered
            else "at least one diagnostic translation unit was not observed as successfully compiled"
        )
    else:
        covered = available and bool(coverage.get("all_translation_units_succeeded"))
        reason = (
            "all compile-database translation units succeeded, covering header or linker diagnostics"
            if covered
            else "non-translation-unit diagnostics require complete compile-database coverage"
        )
    return {
        "schema": "diagnostic_resolution_coverage_v1",
        "covered": covered,
        "reason": reason,
        "diagnostic_files": sorted(normalized),
        "diagnostic_translation_units": sorted(translation_units),
        "succeeded_translation_units": sorted(
            translation_units & succeeded
        ),
        "coverage_available": available,
        "all_translation_units_succeeded": bool(
            coverage.get("all_translation_units_succeeded")
        ),
    }
