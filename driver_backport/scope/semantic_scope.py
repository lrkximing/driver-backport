#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from driver_backport.analysis.code_index import create_clang_context, parse_source_file
from driver_backport.analysis.kbuild_index import KbuildIndex
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import normalize_relpath, utc_now
from driver_backport.scope.libclang_index import ClangCompileContext
from driver_backport.validation.kconfig_sync import (
    dependency_expr_requirements,
    kconfig_index,
    kconfig_source_edges,
)
from driver_backport.validation.build_utils import kconfig_symbol_name


SOURCE_SUFFIXES = {".c", ".h"}
BUILD_FILES = ("Makefile", "Kconfig", "Kbuild")


def _driver_prefix(spec: CaseSpec) -> str:
    return spec.driver_root.rstrip("/") + "/"


def _under_driver(relative_path: str, spec: CaseSpec) -> bool:
    return normalize_relpath(relative_path).startswith(_driver_prefix(spec))


def _source_like(relative_path: str) -> bool:
    return Path(relative_path).suffix in SOURCE_SUFFIXES


def _build_like(relative_path: str) -> bool:
    return Path(relative_path).name in BUILD_FILES or Path(relative_path).suffix == ".mk"



def _analyze_source_file(
    *,
    context: ClangCompileContext | None,
    root: Path,
    driver_root: Path,
    relative_path: str,
) -> dict[str, Any]:
    return parse_source_file(
        context,
        root.resolve(),
        driver_root.resolve(),
        relative_path,
        detailed_processing=True,
        tree=f"semantic:{root.resolve()}",
    )


def _compress_local_reference_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for edge in edges:
        key = (
            normalize_relpath(str(edge.get("entry_file") or "")),
            normalize_relpath(str(edge.get("from") or "")),
            normalize_relpath(str(edge.get("to") or "")),
            str(edge.get("symbol") or ""),
            str(edge.get("symbol_kind") or ""),
            str(edge.get("use_kind") or ""),
        )
        item = grouped.get(key)
        if item is None:
            item = dict(edge)
            item["reference_count"] = 0
            item["locations"] = []
            grouped[key] = item
        item["reference_count"] = int(item.get("reference_count") or 0) + 1
        if len(item["locations"]) < 16:
            item["locations"].append(
                {
                    "line": edge.get("line"),
                    "column": edge.get("column"),
                    "use_kind": edge.get("use_kind"),
                }
            )
    return sorted(grouped.values(), key=lambda item: (str(item.get("from") or ""), str(item.get("to") or ""), str(item.get("symbol") or "")))


def _compress_external_references(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            normalize_relpath(str(record.get("entry_file") or "")),
            normalize_relpath(str(record.get("from") or "")),
            str(record.get("symbol") or ""),
            str(record.get("symbol_kind") or ""),
            str(record.get("use_kind") or ""),
            str(record.get("dependency_scope") or ""),
        )
        item = grouped.get(key)
        if item is None:
            item = dict(record)
            item["reference_count"] = 0
            item["locations"] = []
            grouped[key] = item
        item["reference_count"] = int(item.get("reference_count") or 0) + 1
        if len(item["locations"]) < 16:
            item["locations"].append(
                {
                    "line": record.get("line"),
                    "column": record.get("column"),
                    "referenced_file": record.get("referenced_file"),
                    "use_kind": record.get("use_kind"),
                }
            )
    return sorted(grouped.values(), key=lambda item: (str(item.get("from") or ""), str(item.get("dependency_scope") or ""), str(item.get("symbol") or "")))


def _compress_file_dependency_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for edge in edges:
        source = normalize_relpath(str(edge.get("from") or ""))
        target = normalize_relpath(str(edge.get("to") or ""))
        entry_file = normalize_relpath(str(edge.get("entry_file") or ""))
        if not source or not target or source == target:
            continue
        key = (entry_file, source, target)
        item = grouped.get(key)
        if item is None:
            item = {
                "relation": "local_file_dependency",
                "from": source,
                "to": target,
                "entry_file": entry_file,
                "reference_count": 0,
                "symbols": [],
                "symbol_examples": [],
                "reason": "donor AST references cross this driver file boundary",
            }
            grouped[key] = item
        item["reference_count"] = int(item.get("reference_count") or 0) + int(edge.get("reference_count") or 1)
        symbol = str(edge.get("symbol") or "")
        if symbol and symbol not in item["symbols"]:
            item["symbols"].append(symbol)
            if len(item["symbol_examples"]) < 16:
                item["symbol_examples"].append(
                    {
                        "symbol": symbol,
                        "symbol_kind": edge.get("symbol_kind"),
                        "use_kind": edge.get("use_kind"),
                    }
                )
    for item in grouped.values():
        item["symbol_count"] = len(item["symbols"])
    return sorted(grouped.values(), key=lambda item: (str(item.get("from") or ""), str(item.get("to") or "")))


def _build_graph(root: Path, driver_root: str) -> dict[str, Any]:
    return KbuildIndex.from_directory(root, driver_root).build_contract_graph()


def _kconfig_graph(root: Path, driver_root: str) -> dict[str, Any]:
    root = root.resolve()
    local_root = (root / driver_root).resolve()
    index = kconfig_index((str(root),))
    local_entries = {
        symbol: entry
        for symbol, entry in index.items()
        if any(
            Path(path).resolve().is_relative_to(local_root)
            for path in entry.get("defined_in") or []
        )
    }
    return {
        "configs": sorted(local_entries),
        "selects": {
            symbol: sorted({str(item) for item in entry.get("selects") or []})
            for symbol, entry in local_entries.items()
        },
        "depends": {
            symbol: sorted(
                {
                    *[str(item) for item in entry.get("visibility_depends") or []],
                    *[str(item) for item in entry.get("depends") or []],
                }
            )
            for symbol, entry in local_entries.items()
        },
    }


def _required_config_symbols(expr: str, known_configs: set[str]) -> list[str]:
    requirements = dependency_expr_requirements(
        expr,
    )
    return [
        symbol
        for symbol in requirements["required_symbols"]
        if symbol in known_configs
    ]


def _config_closure(seed_configs: set[str], selects: dict[str, list[str]], depends: dict[str, list[str]] | None = None) -> set[str]:
    closure: set[str] = set()
    queue: deque[str] = deque(sorted(seed_configs))
    depends = depends or {}
    known_configs = set(selects) | set(depends)
    for values in selects.values():
        known_configs.update(values)
    for config_name in seed_configs:
        known_configs.add(config_name)
    while queue:
        config_name = queue.popleft()
        if config_name in closure:
            continue
        closure.add(config_name)
        for selected in selects.get(config_name, []):
            if selected not in closure:
                queue.append(selected)
        for expr in depends.get(config_name, []):
            for required in _required_config_symbols(str(expr), known_configs):
                if required not in closure:
                    queue.append(required)
    return closure


def _entry_build_scope(
    spec: CaseSpec,
    donor_build_graph: dict[str, Any],
    old_build_graph: dict[str, Any],
    donor_kconfig: dict[str, Any],
    old_kconfig: dict[str, Any],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    donor_source_to_configs: dict[str, list[str]] = donor_build_graph.get("source_to_configs") or {}
    donor_config_to_modules: dict[str, list[str]] = donor_build_graph.get("config_to_modules") or {}
    donor_module_to_sources: dict[str, list[str]] = donor_build_graph.get("module_to_sources") or {}
    old_configs = set(old_kconfig.get("configs") or [])
    old_config_to_modules: dict[str, list[str]] = old_build_graph.get("config_to_modules") or {}

    seed_configs = {
        symbol
        for item in spec.build_kconfig
        if (symbol := kconfig_symbol_name(item))
    }
    for entry_file in spec.entry_files:
        seed_configs.update(donor_source_to_configs.get(normalize_relpath(entry_file), []))

    closure = _config_closure(seed_configs, donor_kconfig.get("selects") or {}, donor_kconfig.get("depends") or {})
    selected_files: dict[str, set[str]] = defaultdict(set)
    selected_configs: dict[str, list[str]] = {}

    for config_name in sorted(closure):
        config_missing = config_name not in old_configs or config_name not in old_config_to_modules
        if not config_missing:
            continue
        selected_configs[config_name] = donor_config_to_modules.get(config_name, [])
        for module_name in donor_config_to_modules.get(config_name, []):
            for source in donor_module_to_sources.get(module_name, []):
                source = normalize_relpath(source)
                if not spec.resolve_donor(source).exists():
                    continue
                selected_files[source].add(f"selected by donor build config CONFIG_{config_name}")

    return selected_files, {
        "seed_configs": sorted(seed_configs),
        "selected_config_closure": sorted(closure),
        "donor_only_selected_configs": selected_configs,
    }


def _ancestor_directories(relative_path: str) -> list[str]:
    current = Path(normalize_relpath(relative_path)).parent
    directories: list[str] = []
    while str(current) not in {"", "."}:
        directories.append(normalize_relpath(current.as_posix()))
        current = current.parent
    return directories


def build_integration_files_for_sources(
    spec: CaseSpec,
    scoped_files: set[str],
) -> list[str]:
    """Find build metadata only through exact Kbuild and Kconfig graph edges."""

    donor_root = spec.primary_donor_root
    normalized_sources = {
        normalize_relpath(path)
        for path in scoped_files
        if Path(path).suffix in {".c", ".cc", ".cpp", ".S", ".s"}
        and (donor_root / normalize_relpath(path)).is_file()
    }
    if not normalized_sources:
        return []
    directories = {
        directory
        for source in normalized_sources
        for directory in _ancestor_directories(source)
    }
    build_files: set[str] = set()
    config_symbols: set[str] = set()
    for directory in sorted(directories, key=lambda item: (-len(Path(item).parts), item)):
        index = KbuildIndex.from_directory(donor_root, directory)
        closure = index.assignment_closure_for_sources(normalized_sources)
        build_files.update(
            normalize_relpath(str(record.get("file") or ""))
            for record in closure.get("assignments") or []
            if str(record.get("file") or "").strip()
        )
        config_symbols.update(
            str(symbol).removeprefix("CONFIG_")
            for symbol in closure.get("configs") or []
        )
        if directory in {Path(source).parent.as_posix() for source in normalized_sources}:
            build_files.update(
                record.file
                for record in index.assignments
                if record.relation == "unresolved_dynamic_lhs" or record.unresolved_items
            )

    config_index = kconfig_index((str(donor_root),))
    kconfig_files = {
        normalize_relpath(Path(path).resolve().relative_to(donor_root).as_posix())
        for symbol in config_symbols
        for path in (config_index.get(symbol) or {}).get("defined_in") or []
        if Path(path).resolve().is_relative_to(donor_root)
    }
    local_kconfig = normalize_relpath(f"{spec.driver_root}/Kconfig")
    if (donor_root / local_kconfig).is_file():
        kconfig_files.add(local_kconfig)

    parents_by_child: dict[str, set[str]] = defaultdict(set)
    for parent, child, _line in kconfig_source_edges(str(donor_root)):
        parents_by_child[normalize_relpath(child)].add(normalize_relpath(parent))
    queue: deque[str] = deque(sorted(kconfig_files))
    while queue:
        child = queue.popleft()
        for parent in sorted(parents_by_child.get(child, set())):
            if parent not in kconfig_files:
                kconfig_files.add(parent)
                queue.append(parent)
    build_files.update(kconfig_files)
    return sorted(item for item in build_files if item)


def _file_presence_record(spec: CaseSpec, relative_path: str, reasons: list[str]) -> dict[str, Any]:
    donor_path = spec.resolve_donor(relative_path)
    old_path = spec.resolve_old(relative_path)
    donor_exists = donor_path.exists() and donor_path.is_file()
    old_exists = old_path.exists() and old_path.is_file()
    if donor_exists and not old_exists:
        presence = "donor_only"
        action = "copy" if not _build_like(relative_path) else "build_integration"
    elif old_exists and not donor_exists:
        presence = "old_only"
        action = "context_old_only"
    elif donor_exists and old_exists:
        presence = "shared"
        action = "retain_as_graph_scope_seed"
    else:
        presence = "missing_in_both"
        action = "invalid"
    return {
        "file": relative_path,
        "presence": presence,
        "scope_action": action,
        "old_exists": old_exists,
        "donor_exists": donor_exists,
        "reasons": sorted(set(reason for reason in reasons if reason)),
    }


def build_semantic_migration_scope(spec: CaseSpec) -> dict[str, Any]:
    donor_root = spec.primary_donor_root
    old_root = spec.old_root_path
    donor_driver_root = (donor_root / spec.driver_root).resolve()
    old_driver_root = (old_root / spec.driver_root).resolve()
    max_candidates = spec.retrieval_tuning.max_candidate_files
    max_rounds = max(1, spec.retrieval_tuning.max_rounds)

    donor_context, donor_context_summary = create_clang_context(
        spec.donor_compile_commands[0] if spec.donor_compile_commands else None,
        donor_root,
        path_maps=spec.donor_compile_path_maps,
        make_bin=spec.make_bin,
        build_env=dict(spec.build_env or {}),
    )
    old_context, old_context_summary = create_clang_context(
        spec.compile_commands,
        old_root,
        path_maps=spec.compile_path_maps,
        make_bin=spec.make_bin,
        build_env=dict(spec.build_env or {}),
        candidate_build_dir=spec.build_dir_path,
    )
    if donor_context is None:
        raise RuntimeError(
            "donor semantic scope requires a usable donor compile_commands/libclang context; "
            f"reason={donor_context_summary.get('reason')}"
        )

    donor_build_graph = _build_graph(donor_root, spec.driver_root)
    old_build_graph = _build_graph(old_root, spec.driver_root)
    donor_build_records = list(donor_build_graph.get("records") or [])
    old_build_records = list(old_build_graph.get("records") or [])
    donor_kconfig = _kconfig_graph(donor_root, spec.driver_root)
    old_kconfig = _kconfig_graph(old_root, spec.driver_root)
    build_scope_files, build_scope_summary = _entry_build_scope(
        spec,
        donor_build_graph,
        old_build_graph,
        donor_kconfig,
        old_kconfig,
    )

    analysis_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def analyze(tree: str, relative_path: str) -> dict[str, Any]:
        key = (tree, normalize_relpath(relative_path))
        cached = analysis_cache.get(key)
        if cached is not None:
            return cached
        if tree == "donor":
            result = _analyze_source_file(
                context=donor_context,
                root=donor_root,
                driver_root=donor_driver_root,
                relative_path=relative_path,
            )
        else:
            result = _analyze_source_file(
                context=old_context,
                root=old_root,
                driver_root=old_driver_root,
                relative_path=relative_path,
            )
        analysis_cache[key] = result
        return result

    candidate_reasons: dict[str, set[str]] = defaultdict(set)
    candidate_files: set[str] = set()
    include_edges: list[dict[str, Any]] = []
    reference_edges: list[dict[str, Any]] = []
    external_references: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    budget_events: list[dict[str, Any]] = []
    entry_frontiers: dict[str, dict[str, Any]] = {}

    queue: deque[tuple[str, int, str]] = deque()
    for entry_file in spec.entry_files:
        normalized = normalize_relpath(entry_file)
        queue.append((normalized, 0, normalized))
        candidate_reasons[normalized].add("entry_file")

    queued: set[str] = {normalize_relpath(entry) for entry in spec.entry_files}
    default_entry = normalize_relpath(spec.entry_files[0])
    for relative_path, reasons in sorted(build_scope_files.items()):
        normalized = normalize_relpath(relative_path)
        if not _under_driver(normalized, spec):
            continue
        candidate_reasons[normalized].update(reasons)
        if normalized not in queued:
            queue.append((normalized, 1, default_entry))
            queued.add(normalized)
    analyzed: set[str] = set()

    while queue and len(candidate_files) < max_candidates:
        relative_path, depth, entry_file = queue.popleft()
        relative_path = normalize_relpath(relative_path)
        if relative_path in analyzed or not _under_driver(relative_path, spec):
            continue
        analyzed.add(relative_path)
        candidate_files.add(relative_path)
        donor_analysis = analyze("donor", relative_path)
        if donor_analysis.get("parse_error") and donor_analysis.get("parse_error") != "not_c_or_header":
            parse_errors.append({"tree": "donor", "file": relative_path, "reason": str(donor_analysis.get("parse_error"))})

        for edge in donor_analysis.get("local_includes") or []:
            include_edges.append(edge)
            included = normalize_relpath(str(edge.get("to") or ""))
            if not _under_driver(included, spec):
                continue
            candidate_reasons[included].add(f"locally included from {relative_path}:{edge.get('line')}")
            if included not in queued and depth + 1 <= max_rounds:
                queue.append((included, depth + 1, entry_file))
                queued.add(included)
            elif included not in queued and depth + 1 > max_rounds:
                budget_events.append(
                    {
                        "kind": "max_rounds_exceeded",
                        "from": relative_path,
                        "to": included,
                        "depth": depth + 1,
                        "max_rounds": max_rounds,
                    }
                )

        for ref in donor_analysis.get("references") or []:
            referenced_file = normalize_relpath(str(ref.get("referenced_file") or ""))
            if ref.get("dependency_scope") == "local_driver" and referenced_file and _under_driver(referenced_file, spec):
                if referenced_file == relative_path:
                    continue
                candidate_reasons[referenced_file].add(
                    f"{ref.get('kind')} {ref.get('symbol')} referenced from {relative_path}:{ref.get('line')}"
                )
                reference_edges.append(
                    {
                        "relation": "local_symbol_reference",
                        "from": relative_path,
                        "to": referenced_file,
                        "symbol": ref.get("symbol"),
                        "symbol_kind": ref.get("kind"),
                        "use_kind": ref.get("use_kind"),
                        "line": ref.get("line"),
                        "column": ref.get("column"),
                        "entry_file": entry_file,
                        "reason": "donor AST reference inside driver scope",
                    }
                )
                donor_ref_path = spec.resolve_donor(referenced_file)
                should_expand = donor_ref_path.exists() and donor_ref_path.is_file()
                if should_expand and referenced_file not in queued and len(candidate_files) + len(queued) < max_candidates:
                    if depth + 1 <= max_rounds:
                        queue.append((referenced_file, depth + 1, entry_file))
                        queued.add(referenced_file)
                    else:
                        budget_events.append(
                            {
                                "kind": "max_rounds_exceeded",
                                "from": relative_path,
                                "to": referenced_file,
                                "depth": depth + 1,
                                "max_rounds": max_rounds,
                            }
                        )
                elif should_expand and referenced_file not in queued:
                    budget_events.append(
                        {
                            "kind": "max_candidate_files_exceeded",
                            "from": relative_path,
                            "to": referenced_file,
                            "candidate_files": len(candidate_files),
                            "queued_files": len(queued),
                            "max_candidate_files": max_candidates,
                        }
                    )
            elif ref.get("dependency_scope") in {"kernel_tree", "external", "unresolved"}:
                external_references.append(
                    {
                        "from": relative_path,
                        "symbol": ref.get("symbol"),
                        "symbol_kind": ref.get("kind"),
                        "use_kind": ref.get("use_kind"),
                        "line": ref.get("line"),
                        "column": ref.get("column"),
                        "referenced_file": referenced_file or None,
                        "dependency_scope": ref.get("dependency_scope"),
                        "entry_file": entry_file,
                    }
                )

    if queue:
        budget_events.append(
            {
                "kind": "candidate_queue_not_exhausted",
                "remaining_queue": len(queue),
                "candidate_files": len(candidate_files),
                "max_candidate_files": max_candidates,
            }
        )
    pending_scope_files = sorted(
        {
            normalize_relpath(str(relative_path))
            for relative_path, _depth, _entry_file in queue
            if relative_path and _under_driver(normalize_relpath(str(relative_path)), spec)
        }
        | {
            normalize_relpath(str(item.get("to") or ""))
            for item in budget_events
            if item.get("to")
            and _under_driver(normalize_relpath(str(item.get("to") or "")), spec)
        }
    )
    scope_complete = not budget_events and not pending_scope_files
    continuation = {
        "schema": "semantic_scope_seed_continuation_v1",
        "complete": scope_complete,
        "pending_files": pending_scope_files,
        "budget_events": budget_events,
        "continuation_mode": "compile_diagnostic_code_graph",
        "reason": (
            "initial donor dependency seed is complete within the configured bounds"
            if scope_complete
            else "continue scope from exact compile diagnostics through CodeIndex"
        ),
    }

    for relative_path in sorted(candidate_files):
        if _source_like(relative_path):
            old_analysis = analyze("old", relative_path)
            if old_analysis.get("parse_error") and old_analysis.get("parse_error") not in {"file_missing", "not_c_or_header"}:
                parse_errors.append({"tree": "old", "file": relative_path, "reason": str(old_analysis.get("parse_error"))})

    file_records = [
        _file_presence_record(spec, relative_path, sorted(candidate_reasons.get(relative_path) or []))
        for relative_path in sorted(candidate_files)
    ]
    initial_copy_set = sorted(
        item["file"]
        for item in file_records
        if item["presence"] == "donor_only" and not _build_like(item["file"])
    )
    initial_shared_candidates = sorted(
        item["file"]
        for item in file_records
        if item["presence"] in {"shared", "old_only"}
    )
    build_integration_set = build_integration_files_for_sources(
        spec,
        set(candidate_files) | set(initial_copy_set),
    )

    local_reference_edges = _compress_local_reference_edges(reference_edges)
    file_dependency_edges = _compress_file_dependency_edges(local_reference_edges)
    external_reference_records = _compress_external_references(external_references)

    entry_to_files: dict[str, set[str]] = defaultdict(set)
    entry_to_refs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    entry_to_external: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in file_dependency_edges:
        entry = normalize_relpath(str(edge.get("entry_file") or ""))
        entry_to_files[entry].add(normalize_relpath(str(edge.get("from") or "")))
        entry_to_files[entry].add(normalize_relpath(str(edge.get("to") or "")))
        entry_to_refs[entry].append(edge)
    for item in external_reference_records:
        entry = normalize_relpath(str(item.get("entry_file") or ""))
        entry_to_external[entry].append(item)
    for edge in include_edges:
        for entry in spec.entry_files:
            entry_to_files[normalize_relpath(entry)].add(normalize_relpath(str(edge.get("from") or "")))
            entry_to_files[normalize_relpath(entry)].add(normalize_relpath(str(edge.get("to") or "")))

    for entry_file in spec.entry_files:
        entry = normalize_relpath(entry_file)
        files = sorted(path for path in entry_to_files.get(entry, set()) if path in candidate_files)
        if entry in candidate_files and entry not in files:
            files.insert(0, entry)
        entry_frontiers[entry] = {
            "entry_file": entry,
            "candidate_files": files,
            "copy_seed_files": [path for path in files if path in set(initial_copy_set)],
            "shared_candidates": [path for path in files if path in set(initial_shared_candidates)],
            "local_reference_edges": entry_to_refs.get(entry, []),
            "external_references": entry_to_external.get(entry, []),
            "include_edges": [
                edge
                for edge in include_edges
                if normalize_relpath(str(edge.get("from") or "")) in set(files)
                or normalize_relpath(str(edge.get("to") or "")) in set(files)
            ],
            "parse_errors": [item for item in parse_errors if item.get("file") in set(files)],
        }

    scope_graph = {
        "nodes": file_records,
        "edges": sorted(file_dependency_edges + include_edges, key=lambda item: (str(item.get("from") or ""), str(item.get("to") or ""), str(item.get("symbol") or ""))),
        "symbol_reference_edges": local_reference_edges,
        "external_references": external_reference_records,
        "raw_reference_counts": {
            "local": len(reference_edges),
            "external": len(external_references),
            "local_symbol_references": len(local_reference_edges),
            "external_symbol_references": len(external_reference_records),
        },
        "build_records": {
            "donor": donor_build_records,
            "old": old_build_records,
        },
        "build_graph": {
            "donor": donor_build_graph,
            "old": old_build_graph,
            "donor_kconfig": donor_kconfig,
            "old_kconfig": old_kconfig,
            "entry_build_scope": build_scope_summary,
        },
        "context": {
            "donor": donor_context_summary,
            "old": old_context_summary,
            "max_rounds": max_rounds,
            "max_candidate_files": max_candidates,
        },
        "continuation": continuation,
    }

    summary = {
        "entry_frontiers": len(entry_frontiers),
        "candidate_files": len(candidate_files),
        "copy_files": len(initial_copy_set),
        "shared_candidates": len(initial_shared_candidates),
        "build_integration_files": len(build_integration_set),
        "build_scope_files": len(build_scope_files),
        "local_reference_edges": len(file_dependency_edges),
        "unique_local_symbol_references": len(local_reference_edges),
        "external_references": len(external_reference_records),
        "raw_local_reference_edges": len(reference_edges),
        "raw_external_references": len(external_references),
        "parse_errors": len(parse_errors),
        "complete": scope_complete,
        "pending_scope_files": len(pending_scope_files),
        "budget_event_count": len(budget_events),
        "frontier_backend": "semantic_dependency_seed_v4",
    }

    return {
        "schema": "semantic_migration_scope_v4",
        "case_id": spec.case_id,
        "generated_at": utc_now(),
        "driver_root": spec.driver_root,
        "primary_donor_root": str(donor_root),
        "old_root": str(old_root),
        "entry_files": list(spec.entry_files),
        "entry_frontiers": entry_frontiers,
        "context_files": sorted(candidate_files),
        "candidate_frontier": sorted(candidate_files),
        "scope_decisions": file_records,
        "scope_graph": scope_graph,
        "initial_copy_set": initial_copy_set,
        "initial_shared_candidates": initial_shared_candidates,
        "build_integration_set": build_integration_set,
        "parse_errors": parse_errors,
        "continuation": continuation,
        "summary": summary,
    }
