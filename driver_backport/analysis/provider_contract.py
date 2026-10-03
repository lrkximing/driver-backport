#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import difflib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from driver_backport.analysis.code_index import (
    SOURCE_SUFFIXES,
    parse_source_file,
    parse_structural_entity_file,
)
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import normalize_relpath
from driver_backport.core.executor import run_unified_diff
from driver_backport.scope.libclang_index import ClangCompileContext, is_under, relative_to_root, resolved_path


PROVIDER_KINDS = {"type", "field", "enum_constant"}
STRUCTURAL_PROVIDER_KINDS = {"macro"}
REFERENCE_KINDS = {"field", "type", "function", "variable", "enum_constant", "macro"}
TRANSLATION_UNIT_SUFFIXES = {".c", ".cc", ".cpp"}
PROVIDER_CONTRACT_POLICY_REVISION = "provider_contract_policy_v15"

_OLD_PROVIDER_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}

_C_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_C_IDENTIFIERS_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUOTED_IDENTIFIER_PATTERNS = (
    re.compile(r"'(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)'"),
    re.compile(r"`(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)`"),
    re.compile(r"‘(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)’"),
)


_PREPROCESSOR_CONDITIONAL_RE = re.compile(
    r"^\s*#\s*(?P<directive>if|ifdef|ifndef|elif|else|endif)\b"
)


def _preprocessor_structure_findings(
    relative_path: str,
    text: str,
) -> list[dict[str, Any]]:
    """Validate conditional-directive nesting in one complete source file."""

    stack: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    in_block_comment = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        visible: list[str] = []
        index = 0
        while index < len(raw_line):
            if in_block_comment:
                end = raw_line.find("*/", index)
                if end < 0:
                    index = len(raw_line)
                    continue
                in_block_comment = False
                index = end + 2
                continue
            block = raw_line.find("/*", index)
            single = raw_line.find("//", index)
            stops = [value for value in (block, single) if value >= 0]
            if not stops:
                visible.append(raw_line[index:])
                break
            stop = min(stops)
            visible.append(raw_line[index:stop])
            if stop == single:
                break
            in_block_comment = True
            index = stop + 2
        match = _PREPROCESSOR_CONDITIONAL_RE.match("".join(visible))
        if match is None:
            continue
        directive = match.group("directive")
        if directive in {"if", "ifdef", "ifndef"}:
            stack.append(
                {
                    "directive": directive,
                    "line": line_number,
                    "else_seen": False,
                }
            )
            continue
        if directive == "endif":
            if stack:
                stack.pop()
            else:
                findings.append(
                    {
                        "kind": "unmatched_endif",
                        "file": normalize_relpath(relative_path),
                        "line": line_number,
                    }
                )
            continue
        if not stack:
            findings.append(
                {
                    "kind": f"unmatched_{directive}",
                    "file": normalize_relpath(relative_path),
                    "line": line_number,
                }
            )
            continue
        if directive == "else":
            if stack[-1]["else_seen"]:
                findings.append(
                    {
                        "kind": "duplicate_else",
                        "file": normalize_relpath(relative_path),
                        "line": line_number,
                        "opening_line": stack[-1]["line"],
                    }
                )
            stack[-1]["else_seen"] = True
        elif stack[-1]["else_seen"]:
            findings.append(
                {
                    "kind": "elif_after_else",
                    "file": normalize_relpath(relative_path),
                    "line": line_number,
                    "opening_line": stack[-1]["line"],
                }
            )
    findings.extend(
        {
            "kind": "missing_endif",
            "file": normalize_relpath(relative_path),
            "line": int(entry["line"]),
            "opening_directive": entry["directive"],
        }
        for entry in stack
    )
    return findings


def _ordered_identifier_evidence(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        identifier = str(raw.get("identifier") or "").strip()
        if (
            not _C_IDENTIFIER_RE.fullmatch(identifier)
            or identifier in seen
        ):
            continue
        seen.add(identifier)
        ordered.append(
            {
                "identifier": identifier,
                "source": str(raw.get("source") or "unspecified"),
            }
        )
    return ordered


def _identifiers_in_source_range(
    root: Path,
    *,
    relative_path: str,
    source_range: dict[str, Any],
) -> list[str]:
    start = (
        source_range.get("start")
        if isinstance(source_range.get("start"), dict)
        else {}
    )
    end = (
        source_range.get("end")
        if isinstance(source_range.get("end"), dict)
        else {}
    )
    start_line = int(start.get("line") or 0)
    start_column = int(start.get("column") or 0)
    end_line = int(end.get("line") or 0)
    end_column = int(end.get("column") or 0)
    if (
        start_line <= 0
        or start_column <= 0
        or end_line < start_line
        or end_column <= 0
    ):
        return []

    start_path = resolved_path(start.get("file"))
    end_path = resolved_path(end.get("file"))
    root_path = root.resolve()
    if (
        start_path is None
        or end_path is None
        or start_path != end_path
        or not is_under(start_path, root_path)
    ):
        source_path = root_path / normalize_relpath(relative_path)
    else:
        source_path = start_path
    if not source_path.is_file():
        return []
    try:
        lines = source_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines(keepends=True)
    except OSError:
        return []
    if end_line > len(lines):
        return []

    selected = lines[start_line - 1 : end_line]
    if not selected:
        return []
    selected[0] = selected[0][start_column - 1 :]
    if len(selected) == 1:
        width = max(0, end_column - start_column)
        selected[0] = selected[0][:width]
    else:
        selected[-1] = selected[-1][: max(0, end_column - 1)]
    return _C_IDENTIFIERS_RE.findall("".join(selected))


def diagnostic_identifier_evidence(
    diagnostic: dict[str, Any],
    *,
    root: Path | None = None,
    relative_path: str = "",
) -> list[dict[str, Any]]:
    """Return the compiler's strongest available diagnostic subjects.

    The result is a query seed, not edit authority. CodeIndex must still
    prove a unique typed provider before the repair scope can expand.
    """

    supplied = [
        dict(item)
        for item in diagnostic.get("identifier_evidence") or []
        if isinstance(item, dict)
    ]
    structured_subjects = [
        item
        for item in supplied
        if str(item.get("source") or "")
        == "compiler_diagnostic_subject"
    ]

    quoted: list[dict[str, Any]] = [
        item
        for item in supplied
        if str(item.get("source") or "")
        == "compiler_spelling_quoted_identifier"
    ]
    spelling = str(diagnostic.get("spelling") or "")
    for pattern in _QUOTED_IDENTIFIER_PATTERNS:
        quoted.extend(
            {
                "identifier": match.group("identifier"),
                "source": "compiler_spelling_quoted_identifier",
            }
            for match in pattern.finditer(spelling)
        )

    ranged: list[dict[str, Any]] = [
        item
        for item in supplied
        if str(item.get("source") or "")
        == "compiler_diagnostic_range"
    ]
    if root is not None:
        for source_range in diagnostic.get("ranges") or []:
            if not isinstance(source_range, dict):
                continue
            ranged.extend(
                {
                    "identifier": identifier,
                    "source": "compiler_diagnostic_range",
                }
                for identifier in _identifiers_in_source_range(
                    root,
                    relative_path=relative_path,
                    source_range=source_range,
                )
            )

    unclassified: list[dict[str, Any]] = [
        item
        for item in supplied
        if str(item.get("source") or "")
        not in {
            "compiler_diagnostic_subject",
            "compiler_spelling_quoted_identifier",
            "compiler_diagnostic_range",
            "compiler_location_token",
        }
    ]
    unclassified.extend(
        {
            "identifier": identifier,
            "source": "diagnostic_identifier_field",
        }
        for identifier in [
            diagnostic.get("identifier"),
            *list(diagnostic.get("identifiers") or []),
        ]
        if str(identifier or "").strip()
    )

    located: list[dict[str, Any]] = [
        item
        for item in supplied
        if str(item.get("source") or "")
        == "compiler_location_token"
    ]
    if root is not None:
        location = (
            diagnostic.get("location")
            if isinstance(diagnostic.get("location"), dict)
            else {}
        )
        identifier = _identifier_at_diagnostic_location(
            root,
            relative_path=relative_path,
            location=location,
        )
        if identifier:
            located.append(
                {
                    "identifier": identifier,
                    "source": "compiler_location_token",
                }
            )

    for tier in (
        structured_subjects,
        quoted,
        ranged,
        unclassified,
        located,
    ):
        normalized = _ordered_identifier_evidence(tier)
        if normalized:
            return normalized
    return []


def _source_like(relative_path: str) -> bool:
    return Path(normalize_relpath(relative_path)).suffix in SOURCE_SUFFIXES


def _changed_line_ranges(
    before: list[str],
    after: list[str],
) -> list[tuple[int, int]]:
    """Return exact new-tree line ranges changed between two file snapshots."""

    ranges: list[tuple[int, int]] = []
    matcher = difflib.SequenceMatcher(
        None,
        before,
        after,
        autojunk=False,
    )
    for tag, _old_start, _old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if new_start < new_end:
            start = new_start + 1
            end = new_end
        else:
            start = max(1, min(len(after), new_start + 1))
            end = start
        if ranges and start <= ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def _unchanged_new_to_old_lines(
    before: list[str],
    after: list[str],
) -> dict[int, int]:
    """Map post-patch lines to exact unchanged lines in the current tree."""

    projection: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(
        None,
        before,
        after,
        autojunk=False,
    )
    def occurrence_starts(
        values: list[str],
        block: list[str],
    ) -> list[int]:
        if not block or len(block) > len(values):
            return []
        starts: list[int] = []
        first = block[0]
        width = len(block)
        for index, value in enumerate(values):
            if value != first or values[index : index + width] != block:
                continue
            starts.append(index)
            if len(starts) > 1:
                break
        return starts

    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag != "equal":
            continue
        block = after[new_start:new_end]
        if (
            occurrence_starts(before, block) != [old_start]
            or occurrence_starts(after, block) != [new_start]
            or old_end - old_start != new_end - new_start
        ):
            continue
        for offset in range(new_end - new_start):
            projection[new_start + offset + 1] = old_start + offset + 1
    return projection


def _project_diagnostic_locations_to_current(
    records: list[dict[str, Any]],
    *,
    relative_path: str,
    line_projection: dict[int, int],
    projection_kind: str = "exact_unique_unchanged_diff_block",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve overlay locations and prove their exact current-tree peers."""

    patched_locations: list[dict[str, Any]] = []
    current_locations: list[dict[str, Any]] = []
    patched_seen: set[tuple[str, int, int]] = set()
    current_seen: set[tuple[str, int, int]] = set()
    normalized_path = normalize_relpath(relative_path)
    for record in records:
        raw = (
            record.get("location")
            if isinstance(record.get("location"), dict)
            else {}
        )
        patched_line = int(raw.get("line") or 0)
        column = int(raw.get("column") or 0)
        if patched_line <= 0:
            continue
        patched_key = (normalized_path, patched_line, column)
        if patched_key not in patched_seen:
            patched_seen.add(patched_key)
            patched_locations.append(
                {
                    "file": normalized_path,
                    "line": patched_line,
                    "column": column,
                }
            )
        current_line = int(line_projection.get(patched_line) or 0)
        if current_line <= 0:
            continue
        current_key = (normalized_path, current_line, column)
        if current_key in current_seen:
            continue
        current_seen.add(current_key)
        current_locations.append(
            {
                "file": normalized_path,
                "line": current_line,
                "column": column,
                    "projection": projection_kind,
                "patched_line": patched_line,
            }
        )
    return patched_locations, current_locations


def _diagnostic_overlaps_patch_change(
    diagnostic: dict[str, Any],
    patch_change_ranges: dict[str, list[tuple[int, int]]],
) -> bool:
    relative_path = normalize_relpath(str(diagnostic.get("file") or ""))
    ranges = patch_change_ranges.get(relative_path) or []
    if not ranges:
        return False
    locations = [
        location
        for location in diagnostic.get("locations") or []
        if isinstance(location, dict)
    ]
    unchanged_locations = [
        location
        for location in diagnostic.get("current_locations") or []
        if isinstance(location, dict)
        and str(location.get("projection") or "").startswith("exact_")
    ]
    for location in locations:
        line = int(location.get("line") or 0)
        column = int(location.get("column") or 0)
        projected_unchanged = any(
            int(current.get("patched_line") or 0) == line
            and (
                column <= 0
                or int(current.get("column") or 0) <= 0
                or int(current.get("column") or 0) == column
            )
            for current in unchanged_locations
        )
        if (
            line > 0
            and not projected_unchanged
            and any(start <= line <= end for start, end in ranges)
        ):
            return True
    return False


_OVERLAY_LEAF = "__overlay_writable_leaf__"


def _overlay_path_trie(relative_paths: list[str]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for raw_path in relative_paths:
        normalized = normalize_relpath(raw_path)
        parts = Path(normalized).parts
        if not parts or ".." in parts:
            raise ValueError(f"invalid overlay path: {raw_path}")
        node = root
        for part in parts:
            node = node.setdefault(part, {})
        node[_OVERLAY_LEAF] = True
    return root


def _materialize_overlay_directory(
    source: Path,
    destination: Path,
    writable_trie: dict[str, Any],
    *,
    relative_prefix: Path = Path(),
) -> tuple[list[str], list[str]]:
    """Build a copy-on-write symlink farm for exact writable paths.

    Every ancestor of a patched file is a real directory. Untouched siblings
    remain symlinks into the immutable old tree, while patched leaves are
    regular copies. This lets `patch` modify global headers without copying the
    complete kernel tree or traversing a writable-path symlink.
    """

    destination.mkdir(parents=True, exist_ok=True)
    symlinked: list[str] = []
    copied: list[str] = []
    source_children = (
        {child.name: child for child in source.iterdir()}
        if source.is_dir()
        else {}
    )
    requested_names = {
        name for name in writable_trie if name != _OVERLAY_LEAF
    }
    for name in sorted(set(source_children) | requested_names):
        child_source = source_children.get(name)
        child_destination = destination / name
        child_trie = dict(writable_trie.get(name) or {})
        relative_path = relative_prefix / name
        if name not in requested_names:
            if child_source is None:
                continue
            child_destination.symlink_to(
                child_source,
                target_is_directory=child_source.is_dir(),
            )
            symlinked.append(relative_path.as_posix())
            continue

        nested_names = {
            child for child in child_trie if child != _OVERLAY_LEAF
        }
        is_leaf = bool(child_trie.get(_OVERLAY_LEAF))
        if nested_names:
            if child_source is not None and not child_source.is_dir():
                raise ValueError(
                    f"overlay writable ancestor is not a directory: {relative_path}"
                )
            nested_symlinked, nested_copied = _materialize_overlay_directory(
                child_source or Path("/__driver_backport_missing_overlay_dir__"),
                child_destination,
                child_trie,
                relative_prefix=relative_path,
            )
            symlinked.extend(nested_symlinked)
            copied.extend(nested_copied)
            continue

        if not is_leaf:
            continue
        child_destination.parent.mkdir(parents=True, exist_ok=True)
        if child_source is not None and child_source.is_file():
            shutil.copy2(child_source, child_destination, follow_symlinks=True)
            copied.append(relative_path.as_posix())

    return symlinked, copied


def _build_overlay_root(
    spec: CaseSpec,
    overlay_root: Path,
    *,
    writable_paths: list[str],
) -> dict[str, Any]:
    old_root = spec.old_root_path.resolve()
    writable = sorted(
        dict.fromkeys(normalize_relpath(path) for path in writable_paths)
    )
    symlinked, copied = _materialize_overlay_directory(
        old_root,
        overlay_root,
        _overlay_path_trie(writable),
    )
    return {
        "schema": "copy_on_write_kernel_overlay_v1",
        "writable_paths": writable,
        "symlinked_count": len(symlinked),
        "copied_count": len(copied),
        "symlinked": symlinked,
        "copied": copied,
    }


def _context_for_root(spec: CaseSpec, root: Path) -> tuple[ClangCompileContext | None, dict[str, Any]]:
    if not spec.compile_commands:
        return None, {"enabled": False, "reason": "missing_compile_commands"}
    root = root.resolve()
    path_maps: list[tuple[str, str]] = [(str(spec.old_root_path), str(root))]
    for old_prefix, new_prefix in spec.compile_path_maps:
        remapped_new = str(new_prefix)
        try:
            new_path = Path(str(new_prefix)).expanduser().resolve()
            old_tree = spec.old_root_path
            if new_path == old_tree:
                remapped_new = str(root)
            elif new_path.is_relative_to(old_tree):
                remapped_new = str(root / new_path.relative_to(old_tree))
        except OSError:
            remapped_new = str(new_prefix)
        path_maps.append((str(old_prefix), remapped_new))
    try:
        context = ClangCompileContext(
            Path(spec.compile_commands).resolve(),
            root,
            path_maps=path_maps,
            make_bin=spec.make_bin,
            build_env=dict(spec.build_env or {}),
        )
    except Exception as exc:
        return None, {
            "enabled": False,
            "reason": f"context_init_failed:{exc.__class__.__name__}",
            "error": str(exc),
            "path_maps": [{"from": old, "to": new} for old, new in path_maps],
        }
    return context, {
        "enabled": True,
        "compile_commands": str(Path(spec.compile_commands).resolve()),
        "root": str(root),
        "path_maps": [{"from": old, "to": new} for old, new in path_maps],
    }


def _source_files_under(root: Path, relative_root: str) -> list[str]:
    base = root / normalize_relpath(relative_root)
    if not base.exists():
        return []
    return sorted(
        normalize_relpath(path.resolve().relative_to(root.resolve()).as_posix())
        for path in base.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def _parse_files(
    *,
    context: ClangCompileContext | None,
    root: Path,
    driver_root: str,
    files: list[str],
) -> dict[str, Any]:
    driver_abs = root / normalize_relpath(driver_root)
    parsed_files: list[dict[str, Any]] = []
    for relative_path in files:
        if not _source_like(relative_path):
            continue
        report = parse_source_file(
            context,
            root.resolve(),
            driver_abs.resolve(),
            relative_path,
            detailed_processing=True,
        )
        if any(
            "too many errors emitted" in str(item.get("spelling") or "").lower()
            for item in report.get("diagnostics") or []
            if isinstance(item, dict)
        ):
            report = parse_source_file(
                context,
                root.resolve(),
                driver_abs.resolve(),
                relative_path,
                detailed_processing=True,
                exhaustive_diagnostics=True,
            )
        diagnostics: list[dict[str, Any]] = []
        for raw_diagnostic in report.get("diagnostics") or []:
            diagnostic = dict(raw_diagnostic)
            location = (
                dict(diagnostic.get("location") or {})
                if isinstance(diagnostic.get("location"), dict)
                else {}
            )
            diagnostic["identifier"] = (
                _identifier_at_diagnostic_location(
                    root,
                    relative_path=relative_path,
                    location=location,
                )
            )
            diagnostics.append(diagnostic)
        report["diagnostics"] = diagnostics
        parsed_files.append(report)
    return {
        "files": parsed_files,
        "parse_ok_count": sum(1 for item in parsed_files if item.get("parse_ok")),
        "parse_failed_count": sum(1 for item in parsed_files if not item.get("parse_ok")),
    }


def _lost_touched_translation_unit_parse_coverage(
    old_parsed: dict[str, Any],
    patched_parsed: dict[str, Any],
) -> list[dict[str, str]]:
    old_by_file = {
        normalize_relpath(str(item.get("file") or "")): item
        for item in old_parsed.get("files") or []
        if isinstance(item, dict)
    }
    return [
        {
            "file": relative_path,
            "parse_error": str(item.get("parse_error") or "unknown"),
        }
        for item in patched_parsed.get("files") or []
        if isinstance(item, dict)
        and (relative_path := normalize_relpath(str(item.get("file") or "")))
        and Path(relative_path).suffix.lower() in TRANSLATION_UNIT_SUFFIXES
        and old_by_file.get(relative_path, {}).get("parse_ok") is True
        and item.get("parse_ok") is not True
    ]


def _provider_records(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        record
        for record in _definition_records(parsed)
        if str(record.get("kind") or "") in PROVIDER_KINDS
        and str(record.get("key") or "")
    ]


def _structural_contract_records(
    root: Path,
    files: list[str],
) -> list[dict[str, Any]]:
    """Return declarations and definitions from complete source files."""

    return [
        dict(record)
        for relative_path in files
        for report in [
            parse_structural_entity_file(
                root=root,
                relative_path=relative_path,
                tree="provider_contract",
            )
        ]
        if report.get("parse_ok")
        for record in report.get("definitions") or []
        if str(record.get("key") or record.get("contract_key") or "")
    ]


def _new_duplicate_aggregate_fields(
    old_source: bytes,
    patched_source: bytes,
    *,
    relative_path: str,
) -> list[dict[str, Any]] | None:
    """Compare direct members of uniquely aligned physical C aggregates."""

    try:
        from tree_sitter_languages import get_parser

        parser = get_parser("c")
    except Exception:
        return None

    def aggregate_members(source: bytes) -> dict[tuple[tuple[str, str], ...], list[dict[str, Any]]]:
        aggregates: dict[
            tuple[tuple[str, str], ...], list[dict[str, Any]]
        ] = defaultdict(list)
        root = parser.parse(source).root_node

        def segment(node: Any) -> tuple[str, str] | None:
            kind = node.type.removesuffix("_specifier")
            tags = [
                source[child.start_byte:child.end_byte].decode("utf-8", "ignore")
                for child in node.children
                if child.type == "type_identifier"
            ]
            if len(tags) == 1:
                return kind, tags[0]
            parent = node.parent
            if parent is None or parent.type != "field_declaration":
                return None
            members = [
                source[child.start_byte:child.end_byte].decode("utf-8", "ignore")
                for child in parent.children
                if child.type == "field_identifier"
            ]
            return (kind, members[0]) if len(members) == 1 else None

        def aggregate_key(node: Any) -> tuple[tuple[str, str], ...] | None:
            parts: list[tuple[str, str]] = []
            current = node
            while current is not None:
                if current.type in {"struct_specifier", "union_specifier"}:
                    part = segment(current)
                    if part is None:
                        return None
                    parts.append(part)
                current = current.parent
            return tuple(reversed(parts)) if parts else None

        nodes = [root]
        while nodes:
            node = nodes.pop()
            nodes.extend(reversed(node.children))
            if node.type not in {"struct_specifier", "union_specifier"}:
                continue
            body = next(
                (
                    child
                    for child in node.children
                    if child.type == "field_declaration_list"
                ),
                None,
            )
            key = aggregate_key(node)
            if body is None or key is None or node.has_error:
                continue
            pending = [body]
            fields: dict[str, list[int]] = defaultdict(list)
            conditional = False
            while pending:
                child = pending.pop()
                if child is not body and child.type == "field_declaration_list":
                    continue
                if child is not node and child.type in {
                    "struct_specifier",
                    "union_specifier",
                }:
                    continue
                if child.type.startswith("preproc_"):
                    conditional = True
                    break
                if child.type == "field_identifier":
                    name = source[child.start_byte:child.end_byte].decode(
                        "utf-8", "ignore"
                    )
                    fields[name].append(child.start_point[0] + 1)
                pending.extend(reversed(child.children))
            if not conditional:
                aggregates[key].append({"fields": fields})
        return aggregates

    before = aggregate_members(old_source)
    after = aggregate_members(patched_source)
    findings: list[dict[str, Any]] = []
    for key, candidates in sorted(after.items()):
        if len(candidates) != 1 or len(before.get(key, [])) > 1:
            continue
        old_fields = before[key][0]["fields"] if before.get(key) else {}
        for name, lines in sorted(candidates[0]["fields"].items()):
            old_count = len(old_fields.get(name, []))
            if len(lines) > 1 and len(lines) > old_count:
                findings.append(
                    {
                        "kind": "new_duplicate_aggregate_field",
                        "file": relative_path,
                        "aggregate": [list(part) for part in key],
                        "symbol": name,
                        "old_count": old_count,
                        "patched_count": len(lines),
                        "lines": lines,
                    }
                )
    return findings


def _orphaned_structural_static_calls(
    *,
    old_records: list[dict[str, Any]],
    patched_records: list[dict[str, Any]],
    patched_root: Path,
    source_files: list[str],
) -> list[dict[str, Any]] | None:
    """Find direct calls stranded by removal of a file-local definition."""

    source_set = {
        normalize_relpath(path)
        for path in source_files
        if Path(path).suffix.lower() == ".c"
    }
    patched_definitions = {
        (
            normalize_relpath(str(record.get("file") or "")),
            str(record.get("contract_key") or record.get("key") or ""),
        )
        for record in patched_records
        if record.get("is_definition")
        and str(record.get("kind") or "") == "function"
    }
    removed_by_file: dict[str, dict[str, str]] = defaultdict(dict)
    for record in old_records:
        relative_path = normalize_relpath(str(record.get("file") or ""))
        contract = str(record.get("contract_key") or record.get("key") or "")
        symbol = str(record.get("symbol") or "")
        if (
            relative_path in source_set
            and record.get("is_definition")
            and str(record.get("kind") or "") == "function"
            and str(record.get("linkage") or "") == "INTERNAL"
            and contract
            and symbol
            and (relative_path, contract) not in patched_definitions
        ):
            removed_by_file[relative_path][symbol] = contract
    if not removed_by_file:
        return []

    try:
        from tree_sitter_languages import get_parser

        parser = get_parser("c")
    except Exception:
        return None

    findings: list[dict[str, Any]] = []
    for relative_path, removed in sorted(removed_by_file.items()):
        path = patched_root / relative_path
        if not path.is_file():
            continue
        try:
            source = path.read_bytes()
            tree = parser.parse(source)
        except Exception:
            return None
        owners = [
            record
            for record in patched_records
            if normalize_relpath(str(record.get("file") or ""))
            == relative_path
            and str(record.get("kind") or "") == "function"
            and record.get("is_definition")
        ]
        calls: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        pending = [tree.root_node]
        while pending:
            node = pending.pop()
            if str(node.type) == "call_expression":
                callee = node.child_by_field_name("function")
                if callee is not None and str(callee.type) == "identifier":
                    symbol = source[callee.start_byte : callee.end_byte].decode(
                        "utf-8", errors="ignore"
                    )
                    if symbol in removed:
                        line = int(callee.start_point[0]) + 1
                        column = int(callee.start_point[1]) + 1
                        matching_owners = [
                            owner
                            for owner in owners
                            if symbol in (owner.get("dependency_symbols") or [])
                            and int((owner.get("extent") or {}).get("start_line") or 0)
                            <= line
                            <= int((owner.get("extent") or {}).get("end_line") or 0)
                        ]
                        if len(matching_owners) == 1:
                            calls[symbol].append(
                                (
                                    line,
                                    column,
                                    str(matching_owners[0].get("contract_key") or ""),
                                )
                            )
            pending.extend(reversed(node.children))
        for symbol, locations in sorted(calls.items()):
            findings.append(
                {
                    "source_file": relative_path,
                    "symbol": symbol,
                    "removed_contract_key": removed[symbol],
                    "removed_count": 1,
                    "locations": [
                        f"{relative_path}:{line}:{column}"
                        for line, column, _owner in locations[:12]
                    ],
                    "evidence_kind": "surviving_structural_direct_call",
                    "caller_contract_keys": sorted(
                        {_owner for _line, _column, _owner in locations if _owner}
                    ),
                }
            )
    return findings


def _split_function_contracts(
    old_records: list[dict[str, Any]],
    patched_records: list[dict[str, Any]],
    *,
    source_files: list[str],
) -> list[dict[str, Any]]:
    """Find patches that split one previously consistent function contract."""

    source_file_set = {
        normalize_relpath(path) for path in source_files
    }

    def grouped(
        records: list[dict[str, Any]],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            if str(record.get("kind") or "") != "function":
                continue
            relative_path = normalize_relpath(str(record.get("file") or ""))
            contract_key = str(
                record.get("key") or record.get("contract_key") or ""
            )
            fingerprint = str(record.get("declarator_fingerprint") or "")
            if (
                relative_path not in source_file_set
                or not contract_key
                or not fingerprint
            ):
                continue
            result[(relative_path, contract_key)].append(record)
        return result

    old_groups = grouped(old_records)
    patched_groups = grouped(patched_records)
    findings: list[dict[str, Any]] = []
    for identity, old_group in sorted(old_groups.items()):
        patched_group = patched_groups.get(identity) or []
        old_fingerprints = {
            str(record.get("declarator_fingerprint") or "")
            for record in old_group
        }
        patched_fingerprints = {
            str(record.get("declarator_fingerprint") or "")
            for record in patched_group
        }
        if len(old_fingerprints) != 1 or len(patched_fingerprints) <= 1:
            continue
        findings.append(
            {
                "kind": "split_function_contract",
                "file": identity[0],
                "contract_key": identity[1],
                "symbol": str(
                    (patched_group[0] if patched_group else old_group[0]).get(
                        "symbol"
                    )
                    or ""
                ),
                "old_declarator_fingerprints": sorted(old_fingerprints),
                "patched_declarator_fingerprints": sorted(
                    patched_fingerprints
                ),
                "patched_locations": [
                    _format_record_location(record)
                    for record in patched_group
                ],
                "conditional_contexts": [
                    list(record.get("conditional_context") or [])
                    for record in patched_group
                ],
            }
        )
    return findings


def _added_provider_definitions(
    old_records: list[dict[str, Any]],
    patched_records: list[dict[str, Any]],
    *,
    source_files: list[str],
) -> list[dict[str, Any]]:
    old_counts = _provider_counts_by_file(old_records)
    patched_counts = _provider_counts_by_file(patched_records)
    patched_by_file_and_key: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for record in patched_records:
        relative_path = normalize_relpath(str(record.get("file") or ""))
        key = str(record.get("key") or record.get("contract_key") or "")
        if relative_path and key:
            patched_by_file_and_key[(relative_path, key)].append(record)

    added: list[dict[str, Any]] = []
    for relative_path in source_files:
        for key, patched_count in sorted(
            patched_counts.get(relative_path, Counter()).items()
        ):
            old_count = int(
                old_counts.get(relative_path, Counter()).get(key, 0)
            )
            if int(patched_count) <= old_count:
                continue
            records = patched_by_file_and_key.get((relative_path, key), [])
            representative = records[0] if records else {}
            added.append(
                {
                    "file": relative_path,
                    "key": key,
                    "kind": str(representative.get("kind") or ""),
                    "symbol": str(
                        representative.get("symbol")
                        or representative.get("name")
                        or ""
                    ),
                    "added_count": int(patched_count) - old_count,
                    "dependency_symbols": list(
                        representative.get("dependency_symbols") or []
                    ),
                    "dependency_analysis_complete": bool(
                        representative.get("dependency_analysis_complete")
                    ),
                }
            )
    return added


def _added_contract_definitions(
    old_parsed: dict[str, Any],
    patched_parsed: dict[str, Any],
    *,
    old_structural_records: list[dict[str, Any]],
    patched_structural_records: list[dict[str, Any]],
    source_files: list[str],
) -> list[dict[str, Any]]:
    """Report added definitions without conflating AST and structural views."""

    added = _added_provider_definitions(
        _definition_records(old_parsed),
        _definition_records(patched_parsed),
        source_files=source_files,
    )
    added.extend(
        _added_provider_definitions(
            old_structural_records,
            patched_structural_records,
            source_files=source_files,
        )
    )
    by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in added:
        identity = (
            normalize_relpath(str(item.get("file") or "")),
            str(item.get("key") or ""),
            str(item.get("kind") or "").lower(),
        )
        existing = by_identity.get(identity)
        if existing is None:
            by_identity[identity] = dict(item)
            continue
        existing["added_count"] = max(
            int(existing.get("added_count") or 0),
            int(item.get("added_count") or 0),
        )
        existing["dependency_symbols"] = sorted(
            {
                str(symbol)
                for symbol in [
                    *list(existing.get("dependency_symbols") or []),
                    *list(item.get("dependency_symbols") or []),
                ]
                if str(symbol)
            }
        )
        existing["dependency_analysis_complete"] = bool(
            existing.get("dependency_analysis_complete")
            or item.get("dependency_analysis_complete")
        )
    return [by_identity[key] for key in sorted(by_identity)]


def _definition_records(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(record)
        for file_report in parsed.get("files") or []
        for record in file_report.get("definitions") or []
        if record.get("is_definition")
        and str(record.get("key") or record.get("contract_key") or "")
    ]


def _provider_counts_by_file(records: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        grouped[normalize_relpath(str(record.get("file") or ""))][str(record.get("key") or "")] += 1
    return grouped


def _provider_records_by_key(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("key") or "")].append(record)
    return dict(grouped)


def _reference_signature(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        normalize_relpath(str(record.get("source_file") or "")),
        str(record.get("kind") or ""),
        str(record.get("symbol") or ""),
        str(record.get("use_kind") or ""),
        str(record.get("dependency_scope") or ""),
    )


def _diagnostic_file(diagnostic: dict[str, Any], root: Path, fallback: str) -> str:
    location = diagnostic.get("location") or {}
    file_path = resolved_path(location.get("file"))
    if file_path is not None and is_under(file_path, root.resolve()):
        return normalize_relpath(relative_to_root(file_path, root.resolve()))
    return normalize_relpath(fallback)


def _identifier_at_diagnostic_location(
    root: Path,
    *,
    relative_path: str,
    location: dict[str, Any],
) -> str | None:
    """Read the exact C identifier selected by a compiler source location."""

    line_number = int(location.get("line") or 0)
    column_number = int(location.get("column") or 0)
    if line_number <= 0 or column_number <= 0:
        return None

    location_path = resolved_path(location.get("file"))
    if location_path is None or not is_under(
        location_path,
        root.resolve(),
    ):
        location_path = root / normalize_relpath(relative_path)
    if not location_path.is_file():
        return None
    try:
        lines = location_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
    except OSError:
        return None
    if line_number > len(lines):
        return None

    line = lines[line_number - 1]
    index = column_number - 1
    if index >= len(line):
        return None

    def identifier_character(character: str) -> bool:
        return character == "_" or character.isalnum()

    if not identifier_character(line[index]):
        if index == 0 or not identifier_character(line[index - 1]):
            return None
        index -= 1
    start = index
    while start > 0 and identifier_character(line[start - 1]):
        start -= 1
    end = index + 1
    while end < len(line) and identifier_character(line[end]):
        end += 1
    identifier = line[start:end]
    if not identifier or identifier[0].isdigit():
        return None
    return identifier


def _severe_diagnostic_records(parsed: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_report in parsed.get("files") or []:
        fallback_file = normalize_relpath(str(file_report.get("file") or ""))
        translation_unit_file = normalize_relpath(
            str(file_report.get("translation_unit_file") or fallback_file)
        )
        for diagnostic in file_report.get("diagnostics") or []:
            severity = int(diagnostic.get("severity") or 0)
            if severity < 3:
                continue
            relative_path = _diagnostic_file(
                diagnostic,
                root,
                fallback_file,
            )
            location = (
                dict(diagnostic.get("location") or {})
                if isinstance(diagnostic.get("location"), dict)
                else {}
            )
            identifier_evidence = diagnostic_identifier_evidence(
                diagnostic,
                root=root,
                relative_path=relative_path,
            )
            records.append(
                {
                    "file": relative_path,
                    "translation_unit_file": translation_unit_file,
                    "severity": severity,
                    "spelling": str(diagnostic.get("spelling") or ""),
                    "location": location,
                    "ranges": [
                        dict(item)
                        for item in diagnostic.get("ranges") or []
                        if isinstance(item, dict)
                    ],
                    "identifier_evidence": identifier_evidence,
                    "identifiers": [
                        item["identifier"]
                        for item in identifier_evidence
                    ],
                }
            )
    return records


def _diagnostic_signature(
    record: dict[str, Any],
) -> tuple[str, str, int, str]:
    return (
        normalize_relpath(str(record.get("file") or "")),
        normalize_relpath(
            str(record.get("translation_unit_file") or "")
        ),
        int(record.get("severity") or 0),
        str(record.get("spelling") or ""),
    )


def _unresolved_reference_records(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_report in parsed.get("files") or []:
        if not file_report.get("parse_ok"):
            continue
        for record in file_report.get("references") or []:
            if str(record.get("kind") or "") not in REFERENCE_KINDS:
                continue
            if str(record.get("dependency_scope") or "") != "unresolved":
                continue
            symbol = str(record.get("symbol") or "")
            if not symbol:
                continue
            records.append(dict(record))
    return records


def _old_cache_key(spec: CaseSpec) -> tuple[Any, ...]:
    compile_commands = Path(spec.compile_commands).resolve() if spec.compile_commands else None
    compile_mtime = compile_commands.stat().st_mtime_ns if compile_commands and compile_commands.exists() else 0
    source_snapshot = []
    for relative_path in _source_files_under(
        spec.old_root_path,
        spec.driver_root,
    ):
        path = spec.old_root_path / relative_path
        try:
            stat = path.stat()
        except OSError:
            continue
        source_snapshot.append(
            (relative_path, int(stat.st_mtime_ns), int(stat.st_size))
        )
    return (
        str(spec.old_root_path),
        normalize_relpath(spec.driver_root),
        str(compile_commands) if compile_commands else "",
        compile_mtime,
        tuple(spec.compile_path_maps),
        tuple(source_snapshot),
    )


def _old_provider_baseline(spec: CaseSpec) -> dict[str, Any]:
    key = _old_cache_key(spec)
    cached = _OLD_PROVIDER_CACHE.get(key)
    if cached is not None:
        return cached

    context, context_summary = _context_for_root(spec, spec.old_root_path)
    files = _source_files_under(spec.old_root_path, spec.driver_root)
    parsed = _parse_files(
        context=context,
        root=spec.old_root_path,
        driver_root=spec.driver_root,
        files=files,
    )
    records = _provider_records(parsed)
    provider_by_key = _provider_records_by_key(records)
    result = {
        "context": context_summary,
        "file_count": len(files),
        "parse_ok_count": parsed.get("parse_ok_count"),
        "parse_failed_count": parsed.get("parse_failed_count"),
        "provider_counts": Counter(record["key"] for record in records),
        "provider_counts_by_file": _provider_counts_by_file(records),
        "provider_records_by_key": provider_by_key,
    }
    cache_prefix = key[:-1]
    for stale_key in list(_OLD_PROVIDER_CACHE):
        if stale_key[:-1] == cache_prefix:
            _OLD_PROVIDER_CACHE.pop(stale_key, None)
    _OLD_PROVIDER_CACHE[key] = result
    return result


def _format_record_location(record: dict[str, Any]) -> str:
    extent = record.get("extent") or {}
    return (
        f"{normalize_relpath(str(record.get('file') or ''))}:"
        f"{int(extent.get('start_line') or 0)} {record.get('key')}"
    )


def _diagnostic_finding(
    diagnostic: dict[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    finding = {
        "kind": kind,
        "file": diagnostic.get("file"),
        "spelling": diagnostic.get("spelling"),
        "identifiers": list(diagnostic.get("identifiers") or []),
    }
    if diagnostic.get("identifier_evidence"):
        finding["identifier_evidence"] = deepcopy(
            diagnostic.get("identifier_evidence") or []
        )
    return finding


def _provider_contract_decision(
    report: dict[str, Any],
    *,
    expected_provider_obligations: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """Separate structural blockers from non-authoritative parse observations."""

    duplicate_providers = list(report.get("duplicate_providers") or [])
    split_function_contracts = list(
        report.get("split_function_contracts") or []
    )
    new_diagnostics = list(report.get("new_diagnostics") or [])
    new_unresolved = list(report.get("new_unresolved_references") or [])
    orphaned_patch_references = list(
        report.get("orphaned_patch_references") or []
    )
    non_authoritative_diagnostic_files = {
        normalize_relpath(str(path))
        for path in report.get("non_authoritative_diagnostic_files") or []
        if str(path).strip()
    }
    touched_source_files = {
        normalize_relpath(str(path))
        for path in report.get("touched_source_files") or []
        if str(path).strip()
    }
    old_fatal_translation_units = {
        normalize_relpath(str(path))
        for path in report.get("old_fatal_translation_units") or []
        if str(path).strip()
    }
    patch_change_ranges = {
        normalize_relpath(str(path)): [
            (int(item[0]), int(item[1]))
            for item in ranges
            if isinstance(item, (list, tuple)) and len(item) == 2
        ]
        for path, ranges in (
            report.get("patch_change_ranges") or {}
        ).items()
    }
    expected_symbols = {
        str(item.get("symbol") or "")
        for item in expected_provider_obligations or []
        if isinstance(item, dict)
        and item.get("source") == "semantic_reference"
        and str(item.get("symbol") or "")
        and str(item.get("contract_key") or "")
        and str(item.get("provider_file") or "")
    }
    added_provider_symbols = {
        str(item.get("symbol") or "")
        for item in report.get("added_provider_definitions") or []
        if isinstance(item, dict) and str(item.get("symbol") or "")
    }
    expected_unresolved_references = [
        item
        for item in new_unresolved
        if str(item.get("symbol") or "") in expected_symbols
    ]

    def diagnostic_is_authoritative(
        diagnostic: dict[str, Any],
    ) -> bool:
        translation_units = [
            normalize_relpath(str(path))
            for path in diagnostic.get("translation_unit_files") or []
            if str(path).strip()
        ]
        return any(
            Path(path).suffix.lower() in TRANSLATION_UNIT_SUFFIXES
            for path in translation_units
        )

    def diagnostic_matches_expected_reference(
        diagnostic: dict[str, Any],
    ) -> bool:
        diagnostic_identifiers = {
            str(identifier)
            for identifier in [
                diagnostic.get("identifier"),
                *list(diagnostic.get("identifiers") or []),
            ]
            if str(identifier or "")
        }
        if diagnostic_identifiers and diagnostic_identifiers <= expected_symbols:
            return True
        diagnostic_file = normalize_relpath(
            str(diagnostic.get("file") or "")
        )
        diagnostic_locations = [
            location
            for location in diagnostic.get("locations") or []
            if isinstance(location, dict)
        ]
        if not diagnostic_locations:
            return False
        for diagnostic_location in diagnostic_locations:
            location_matches = False
            for reference in expected_unresolved_references:
                if (
                    normalize_relpath(
                        str(reference.get("source_file") or "")
                    )
                    != diagnostic_file
                ):
                    continue
                diagnostic_line = int(
                    diagnostic_location.get("line") or 0
                )
                diagnostic_column = int(
                    diagnostic_location.get("column") or 0
                )
                for reference_location in (
                    reference.get("location_records") or []
                ):
                    if not isinstance(reference_location, dict):
                        continue
                    reference_line = int(
                        reference_location.get("line") or 0
                    )
                    reference_column = int(
                        reference_location.get("column") or 0
                    )
                    if (
                        diagnostic_line > 0
                        and diagnostic_line == reference_line
                        and (
                            diagnostic_column <= 0
                            or reference_column <= 0
                            or diagnostic_column == reference_column
                        )
                    ):
                        location_matches = True
                        break
                if location_matches:
                    break
            if not location_matches:
                return False
        return True

    def diagnostic_matches_added_provider(
        diagnostic: dict[str, Any],
    ) -> bool:
        identifiers = {
            str(identifier)
            for identifier in [
                diagnostic.get("identifier"),
                *list(diagnostic.get("identifiers") or []),
            ]
            if str(identifier or "")
        }
        return bool(identifiers and identifiers <= added_provider_symbols)

    all_patch_locus_diagnostics = [
        item
        for item in new_diagnostics
        if _diagnostic_overlaps_patch_change(
            item,
            patch_change_ranges,
        )
    ]
    # A real translation unit makes the diagnostic authoritative. Diagnostics
    # at the changed locus are directly patch-caused. A new error elsewhere in
    # a touched source is also a blocker when the baseline TU had no fatal parse
    # stop; only a previously fatal TU can leave such a diagnostic unproven.
    authoritative_new_diagnostics = [
        item
        for item in new_diagnostics
        if diagnostic_is_authoritative(item)
    ]
    authoritative_patch_locus_diagnostics = [
        item
        for item in all_patch_locus_diagnostics
        if diagnostic_is_authoritative(item)
    ]
    expected_provider_diagnostics = [
        item
        for item in authoritative_new_diagnostics
        if diagnostic_matches_expected_reference(item)
    ]
    added_provider_diagnostics = [
        item
        for item in authoritative_new_diagnostics
        if diagnostic_matches_added_provider(item)
    ]
    patch_locus_diagnostics = [
        item
        for item in authoritative_patch_locus_diagnostics
        if item not in expected_provider_diagnostics
        and item not in added_provider_diagnostics
    ]
    touched_source_diagnostics = [
        item
        for item in authoritative_new_diagnostics
        if item not in authoritative_patch_locus_diagnostics
        and item not in expected_provider_diagnostics
        and item not in added_provider_diagnostics
        and normalize_relpath(str(item.get("file") or ""))
        in touched_source_files
        and any(
            normalize_relpath(str(path))
            not in old_fatal_translation_units
            for path in item.get("translation_unit_files") or []
            if str(path).strip()
        )
    ]
    unmasked_translation_unit_diagnostics = [
        item
        for item in authoritative_new_diagnostics
        if item not in authoritative_patch_locus_diagnostics
        and item not in expected_provider_diagnostics
        and item not in added_provider_diagnostics
        and item not in touched_source_diagnostics
    ]
    blocking_findings = [
        dict(item)
        for item in split_function_contracts
    ]
    blocking_findings.extend(
        {
            "kind": "duplicate_provider",
            "key": item.get("key"),
            "file": item.get("file"),
            "existing_locations": item.get("existing_locations"),
        }
        for item in duplicate_providers
    )
    blocking_findings.extend(
        {
            **_diagnostic_finding(
                item,
                kind="added_provider_consumer_diagnostic",
            ),
            "severity": item.get("severity"),
            "locations": item.get("locations"),
            "current_locations": item.get("current_locations"),
            "translation_unit_files": list(
                item.get("translation_unit_files") or []
            ),
        }
        for item in added_provider_diagnostics
    )
    blocking_findings.extend(
        {
            **_diagnostic_finding(
                item,
                kind="patch_locus_diagnostic",
            ),
            "severity": item.get("severity"),
            "locations": item.get("locations"),
            "current_locations": item.get("current_locations"),
            "translation_unit_files": list(
                item.get("translation_unit_files") or []
            ),
        }
        for item in patch_locus_diagnostics
    )
    blocking_findings.extend(
        {
            **_diagnostic_finding(
                item,
                kind="new_touched_source_diagnostic",
            ),
            "severity": item.get("severity"),
            "locations": item.get("locations"),
            "current_locations": item.get("current_locations"),
            "translation_unit_files": list(
                item.get("translation_unit_files") or []
            ),
        }
        for item in touched_source_diagnostics
    )
    blocking_findings.extend(
        {
            "kind": "orphaned_patch_reference",
            "source_file": item.get("source_file"),
            "symbol": item.get("symbol"),
            "removed_contract_key": item.get("removed_contract_key"),
            "locations": item.get("locations"),
        }
        for item in orphaned_patch_references
    )
    advisory_findings = [
        _diagnostic_finding(
            item,
            kind=(
                "standalone_header_ast_diagnostic"
                if normalize_relpath(str(item.get("file") or ""))
                in non_authoritative_diagnostic_files
                else "partial_ast_diagnostic"
            ),
        )
        for item in new_diagnostics
        if item not in patch_locus_diagnostics
        and item not in touched_source_diagnostics
        and item not in unmasked_translation_unit_diagnostics
        and item not in expected_provider_diagnostics
        and item not in added_provider_diagnostics
    ]
    advisory_findings.extend(
        {
            **_diagnostic_finding(
                item,
                kind="unmasked_translation_unit_diagnostic",
            ),
            "severity": item.get("severity"),
            "locations": item.get("locations"),
            "current_locations": item.get("current_locations"),
            "translation_unit_files": list(
                item.get("translation_unit_files") or []
            ),
        }
        for item in unmasked_translation_unit_diagnostics
    )
    advisory_findings.extend(
        {
            **_diagnostic_finding(
                item,
                kind=(
                    "expected_post_materialization_provider_diagnostic"
                ),
            ),
            "locations": item.get("locations"),
        }
        for item in expected_provider_diagnostics
    )
    advisory_findings.extend(
        {
            "kind": "partial_ast_unresolved_reference",
            "source_file": item.get("source_file"),
            "symbol": item.get("symbol"),
            "reference_kind": item.get("kind"),
        }
        for item in new_unresolved
    )
    report = {
        **report,
        "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
        "patch_locus_diagnostics": patch_locus_diagnostics,
        "new_touched_source_diagnostics": touched_source_diagnostics,
        "unmasked_translation_unit_diagnostics": (
            unmasked_translation_unit_diagnostics
        ),
        "expected_post_materialization_provider_diagnostics": (
            expected_provider_diagnostics
        ),
        "added_provider_consumer_diagnostics": (
            added_provider_diagnostics
        ),
        "expected_post_materialization_provider_obligations": (
            list(expected_provider_obligations or [])
        ),
        "blocking_findings": blocking_findings,
        "advisory_findings": advisory_findings,
        "decision_basis": {
            "hard_blockers": [
                "split_previously_consistent_function_contract",
                "duplicate_provider",
                "new_error_or_fatal_at_patch_change",
                "new_error_in_touched_source_without_old_fatal_parse",
                "new_error_at_added_provider_consumer",
                "orphaned_reference_to_removed_local_contract",
            ],
            "advisory_only": [
                "partial_ast_diagnostic",
                "standalone_header_ast_diagnostic",
                "partial_ast_unresolved_reference",
                "unmasked_translation_unit_diagnostic",
                (
                    "exact_graph_bound_post_materialization_"
                    "provider_diagnostic"
                ),
            ],
            "authoritative_validation": "round_end_kernel_build",
        },
        "status": (
            "rejected"
            if blocking_findings
            else "checked_with_advisories"
            if advisory_findings
            else "checked"
        ),
    }
    if split_function_contracts:
        details = "; ".join(
            (
                f"{item.get('file')}: {item.get('contract_key')} has "
                "divergent declarations or conditional definitions"
            )
            for item in split_function_contracts[:5]
        )
        return (
            False,
            f"provider_contract_split_function_contract: {details}",
            report,
        )
    if duplicate_providers:
        details = "; ".join(
            f"{item['key']} added in {item['file']} but already exists at {item['existing_locations']}"
            for item in duplicate_providers[:5]
        )
        return False, f"provider_contract_duplicate_provider: {details}", report
    if patch_locus_diagnostics:
        details = "; ".join(
            f"{item.get('file')}: {item.get('spelling')}"
            for item in patch_locus_diagnostics[:5]
        )
        return False, f"provider_contract_patch_locus_diagnostic: {details}", report
    if touched_source_diagnostics:
        details = "; ".join(
            f"{item.get('file')}: {item.get('spelling')}"
            for item in touched_source_diagnostics[:5]
        )
        return (
            False,
            f"provider_contract_new_touched_source_diagnostic: {details}",
            report,
        )
    if added_provider_diagnostics:
        details = "; ".join(
            f"{item.get('file')}: {item.get('spelling')}"
            for item in added_provider_diagnostics[:5]
        )
        return (
            False,
            f"provider_contract_incomplete_provider_closure: {details}",
            report,
        )
    if orphaned_patch_references:
        details = "; ".join(
            (
                f"{item.get('source_file')}: {item.get('symbol')} still "
                f"references removed {item.get('removed_contract_key')}"
            )
            for item in orphaned_patch_references[:5]
        )
        return False, f"provider_contract_incomplete_reference_closure: {details}", report
    return True, None, report


def validate_provider_contract_patch(
    spec: CaseSpec,
    *,
    diff_text: str,
    touched_files: list[str],
    expected_provider_obligations: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    source_touched = [normalize_relpath(path) for path in touched_files if _source_like(path)]
    if not source_touched:
        return True, None, {
            "status": "skipped",
            "reason": "no_source_like_touched_files",
            "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
        }
    if not spec.compile_commands:
        reason = "missing_compile_commands"
        return False, f"provider_contract_validation_unavailable: {reason}", {
            "status": "unavailable",
            "reason": reason,
            "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
        }

    old_touched_context, old_touched_context_summary = _context_for_root(spec, spec.old_root_path)
    if old_touched_context is None:
        reason = "clang_context_unavailable"
        return False, f"provider_contract_validation_unavailable: {reason}", {
            "status": "unavailable",
            "reason": reason,
            "context": old_touched_context_summary,
            "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
        }
    parse_files = sorted(dict.fromkeys(source_touched))
    old_touched = _parse_files(
        context=old_touched_context,
        root=spec.old_root_path,
        driver_root=spec.driver_root,
        files=parse_files,
    )
    old_structural_contract_records = _structural_contract_records(
        spec.old_root_path,
        parse_files,
    )
    old_structural_definition_records = [
        record
        for record in old_structural_contract_records
        if record.get("is_definition")
    ]
    old_structural_provider_records = [
        record
        for record in old_structural_definition_records
        if str(record.get("kind") or "")
        in STRUCTURAL_PROVIDER_KINDS
    ]
    patched_parse_root: Path | None = None
    patched_structural_provider_records: list[dict[str, Any]] = []
    patched_structural_definition_records: list[dict[str, Any]] = []
    patched_structural_contract_records: list[dict[str, Any]] = []
    structural_orphaned_calls: list[dict[str, Any]] = []
    patch_change_ranges: dict[str, list[tuple[int, int]]] = {}
    current_line_projections: dict[str, dict[int, int]] = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        overlay_root = Path(temp_dir).resolve() / "kernel"
        overlay_root.mkdir(parents=True, exist_ok=True)
        patched_parse_root = overlay_root
        try:
            overlay_summary = _build_overlay_root(
                spec,
                overlay_root,
                writable_paths=source_touched,
            )
        except Exception as exc:
            reason = f"overlay_failed:{exc.__class__.__name__}"
            return False, f"provider_contract_validation_unavailable: {reason}", {
                "status": "unavailable",
                "reason": reason,
                "error": str(exc),
                "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
            }
        returncode, stdout, stderr = run_unified_diff(
            overlay_root,
            diff_text,
        )
        if returncode != 0:
            reason = "patch_apply_failed_in_overlay"
            return False, f"provider_contract_validation_unavailable: {reason}", {
                "status": "unavailable",
                "reason": reason,
                "stdout": stdout,
                "stderr": stderr,
                "overlay": overlay_summary,
                "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
            }

        preprocessor_findings: list[dict[str, Any]] = []
        for relative_path in parse_files:
            patched_path = overlay_root / relative_path
            if not patched_path.is_file():
                continue
            preprocessor_findings.extend(
                _preprocessor_structure_findings(
                    relative_path,
                    patched_path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    ),
                )
            )
        if preprocessor_findings:
            details = "; ".join(
                (
                    f"{item.get('file')}:{item.get('line')}:"
                    f"{item.get('kind')}"
                )
                for item in preprocessor_findings[:8]
            )
            return (
                False,
                f"provider_contract_preprocessor_structure: {details}",
                {
                    "status": "rejected",
                    "policy_revision": (
                        PROVIDER_CONTRACT_POLICY_REVISION
                    ),
                    "blocking_findings": preprocessor_findings,
                    "decision_basis": {
                        "hard_blockers": [
                            "unmatched_preprocessor_conditional",
                            "duplicate_else",
                            "elif_after_else",
                            "missing_endif",
                        ],
                        "authoritative_validation": (
                            "full_patched_source_file"
                        ),
                    },
                    "overlay": overlay_summary,
                },
            )

        patched_context, patched_context_summary = _context_for_root(spec, overlay_root)
        if patched_context is None:
            reason = "patched_clang_context_unavailable"
            return False, f"provider_contract_validation_unavailable: {reason}", {
                "status": "unavailable",
                "reason": reason,
                "context": patched_context_summary,
                "overlay": overlay_summary,
                "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
            }
        patched_touched = _parse_files(
            context=patched_context,
            root=overlay_root,
            driver_root=spec.driver_root,
            files=parse_files,
        )
        lost_parse_coverage = _lost_touched_translation_unit_parse_coverage(
            old_touched, patched_touched
        )
        if lost_parse_coverage:
            return False, "provider_contract_patched_parse_unavailable", {
                "status": "unavailable",
                "reason": "patched_translation_unit_parse_coverage_lost",
                "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
                "lost_parse_coverage": lost_parse_coverage,
                "overlay": overlay_summary,
            }
        patched_structural_contract_records = _structural_contract_records(
            overlay_root,
            parse_files,
        )
        old_field_counts = Counter(
            (
                str(record.get("file") or ""),
                str(record.get("contract_key") or record.get("key") or ""),
            )
            for record in old_structural_contract_records
            if record.get("kind") == "field" and record.get("is_definition")
        )
        patched_field_counts = Counter(
            (
                str(record.get("file") or ""),
                str(record.get("contract_key") or record.get("key") or ""),
            )
            for record in patched_structural_contract_records
            if record.get("kind") == "field" and record.get("is_definition")
        )
        duplicate_candidate_files = {
            relative_path
            for (relative_path, key), count in patched_field_counts.items()
            if key and count > 1 and count > old_field_counts[(relative_path, key)]
        }
        duplicate_field_findings: list[dict[str, Any]] = []
        for relative_path in sorted(duplicate_candidate_files):
            old_path = spec.old_root_path / relative_path
            patched_path = overlay_root / relative_path
            findings = _new_duplicate_aggregate_fields(
                old_path.read_bytes() if old_path.is_file() else b"",
                patched_path.read_bytes(),
                relative_path=relative_path,
            )
            if findings is None:
                reason = "aggregate_field_validation_unavailable"
                return False, f"provider_contract_validation_unavailable: {reason}", {
                    "status": "unavailable",
                    "reason": reason,
                    "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
                }
            duplicate_field_findings.extend(findings)
        if duplicate_field_findings:
            return False, "provider_contract_duplicate_aggregate_field", {
                "status": "rejected",
                "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
                "blocking_findings": duplicate_field_findings,
                "decision_basis": {
                    "hard_blockers": ["new_duplicate_aggregate_field"],
                    "authoritative_validation": "old_and_patched_physical_aggregate",
                },
                "overlay": overlay_summary,
            }
        structural_orphaned_calls_result = _orphaned_structural_static_calls(
            old_records=old_structural_contract_records,
            patched_records=patched_structural_contract_records,
            patched_root=overlay_root,
            source_files=parse_files,
        )
        if structural_orphaned_calls_result is None:
            reason = "structural_static_call_validation_unavailable"
            return False, f"provider_contract_validation_unavailable: {reason}", {
                "status": "unavailable",
                "reason": reason,
                "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
            }
        structural_orphaned_calls = structural_orphaned_calls_result
        patched_structural_definition_records = [
            record
            for record in patched_structural_contract_records
            if record.get("is_definition")
        ]
        patched_structural_provider_records = [
            record
            for record in patched_structural_definition_records
            if str(record.get("kind") or "")
            in STRUCTURAL_PROVIDER_KINDS
        ]
        for relative_path in parse_files:
            old_path = spec.old_root_path / relative_path
            patched_path = overlay_root / relative_path
            if not old_path.is_file() or not patched_path.is_file():
                continue
            old_lines = old_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
            patched_lines = patched_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
            patch_change_ranges[relative_path] = _changed_line_ranges(
                old_lines,
                patched_lines,
            )
            current_line_projections[relative_path] = (
                _unchanged_new_to_old_lines(old_lines, patched_lines)
            )

    old_touched_records = _provider_records(old_touched)
    patched_touched_records = _provider_records(patched_touched)
    old_touched_counts_by_file = _provider_counts_by_file(old_touched_records)
    patched_touched_counts_by_file = _provider_counts_by_file(patched_touched_records)
    patched_records_by_key = _provider_records_by_key(patched_touched_records)

    added_provider_definitions = _added_provider_definitions(
        old_touched_records,
        patched_touched_records,
        source_files=source_touched,
    )
    added_provider_definitions.extend(
        _added_provider_definitions(
            old_structural_provider_records,
            patched_structural_provider_records,
            source_files=source_touched,
        )
    )
    added_provider_definitions = list(
        {
            (
                str(item.get("file") or ""),
                str(item.get("key") or ""),
            ): item
            for item in added_provider_definitions
        }.values()
    )
    added_contract_definitions = _added_contract_definitions(
        old_touched,
        patched_touched,
        old_structural_records=(
            old_structural_definition_records
        ),
        patched_structural_records=(
            patched_structural_definition_records
        ),
        source_files=source_touched,
    )
    split_function_contracts = _split_function_contracts(
        old_structural_contract_records,
        patched_structural_contract_records,
        source_files=source_touched,
    )

    added_provider_keys: set[str] = set()
    for relative_path in source_touched:
        old_file_counts = old_touched_counts_by_file.get(relative_path, Counter())
        patched_file_counts = patched_touched_counts_by_file.get(relative_path, Counter())
        for key, patched_count in sorted(patched_file_counts.items()):
            if int(patched_count) > int(old_file_counts.get(key, 0)):
                added_provider_keys.add(key)

    old_baseline: dict[str, Any] | None = None
    if added_provider_keys:
        try:
            old_baseline = _old_provider_baseline(spec)
        except Exception as exc:
            reason = f"old_provider_baseline_failed:{exc.__class__.__name__}"
            return False, f"provider_contract_validation_unavailable: {reason}", {
                "status": "unavailable",
                "reason": reason,
                "error": str(exc),
                "policy_revision": PROVIDER_CONTRACT_POLICY_REVISION,
            }
    old_global_counts: Counter[str] = (old_baseline or {}).get("provider_counts") or Counter()
    old_records_by_key: dict[str, list[dict[str, Any]]] = (old_baseline or {}).get("provider_records_by_key") or {}

    duplicate_providers: list[dict[str, Any]] = []
    for relative_path in source_touched:
        old_file_counts = old_touched_counts_by_file.get(relative_path, Counter())
        patched_file_counts = patched_touched_counts_by_file.get(relative_path, Counter())
        for key, patched_count in sorted(patched_file_counts.items()):
            old_file_count = int(old_file_counts.get(key, 0))
            old_global_count = int(old_global_counts.get(key, 0))
            if patched_count <= old_file_count:
                continue
            if old_global_count <= 0:
                continue
            duplicate_providers.append(
                {
                    "key": key,
                    "file": relative_path,
                    "old_global_count": old_global_count,
                    "old_file_count": old_file_count,
                    "patched_file_count": int(patched_count),
                    "existing_locations": [
                        _format_record_location(record)
                        for record in old_records_by_key.get(key, [])[:6]
                    ],
                    "patched_locations": [
                        _format_record_location(record)
                        for record in patched_records_by_key.get(key, [])[:6]
                    ],
                }
            )

    old_unresolved = Counter(_reference_signature(record) for record in _unresolved_reference_records(old_touched))
    patched_unresolved_records = _unresolved_reference_records(patched_touched)
    patched_unresolved = Counter(_reference_signature(record) for record in patched_unresolved_records)
    new_unresolved: list[dict[str, Any]] = []
    for signature, count in sorted(patched_unresolved.items()):
        old_count = int(old_unresolved.get(signature, 0))
        if count <= old_count:
            continue
        examples = [
            record
            for record in patched_unresolved_records
            if _reference_signature(record) == signature
        ][:4]
        new_unresolved.append(
            {
                "source_file": signature[0],
                "kind": signature[1],
                "symbol": signature[2],
                "use_kind": signature[3],
                "dependency_scope": signature[4],
                "old_count": old_count,
                "patched_count": int(count),
                "locations": [
                    f"{record.get('source_file')}:{record.get('line')}:{record.get('column')}"
                    for record in examples
                ],
                "location_records": [
                    {
                        "file": normalize_relpath(
                            str(record.get("source_file") or "")
                        ),
                        "line": int(record.get("line") or 0),
                        "column": int(record.get("column") or 0),
                    }
                    for record in examples
                ],
            }
        )

    old_definitions = _definition_records(old_touched)
    patched_definitions = _definition_records(patched_touched)
    old_definition_counts = Counter(
        (
            normalize_relpath(str(record.get("file") or "")),
            str(record.get("key") or record.get("contract_key") or ""),
            str(record.get("symbol") or record.get("name") or ""),
        )
        for record in old_definitions
    )
    patched_definition_counts = Counter(
        (
            normalize_relpath(str(record.get("file") or "")),
            str(record.get("key") or record.get("contract_key") or ""),
            str(record.get("symbol") or record.get("name") or ""),
        )
        for record in patched_definitions
    )
    removed_local_contracts = [
        {
            "file": identity[0],
            "contract_key": identity[1],
            "symbol": identity[2],
            "removed_count": old_count
            - int(patched_definition_counts.get(identity, 0)),
        }
        for identity, old_count in sorted(old_definition_counts.items())
        if old_count > int(patched_definition_counts.get(identity, 0))
        and Path(identity[0]).suffix.lower() in TRANSLATION_UNIT_SUFFIXES
    ]
    orphaned_patch_references: list[dict[str, Any]] = []
    for removed in removed_local_contracts:
        matching_unresolved = [
            reference
            for reference in new_unresolved
            if normalize_relpath(str(reference.get("source_file") or ""))
            == removed["file"]
            and str(reference.get("symbol") or "") == removed["symbol"]
        ]
        if not matching_unresolved:
            continue
        orphaned_patch_references.append(
            {
                "source_file": removed["file"],
                "symbol": removed["symbol"],
                "removed_contract_key": removed["contract_key"],
                "removed_count": removed["removed_count"],
                "locations": [
                    location
                    for reference in matching_unresolved
                    for location in reference.get("locations") or []
                ][:12],
            }
        )
    already_orphaned = {
        (item["source_file"], item["symbol"])
        for item in orphaned_patch_references
    }
    orphaned_patch_references.extend(
        item
        for item in structural_orphaned_calls
        if (item["source_file"], item["symbol"]) not in already_orphaned
    )

    old_diagnostic_records = _severe_diagnostic_records(
        old_touched,
        spec.old_root_path,
    )
    old_diagnostics = Counter(
        _diagnostic_signature(record)
        for record in old_diagnostic_records
    )
    old_fatal_translation_units = sorted(
        {
            normalize_relpath(
                str(record.get("translation_unit_file") or "")
            )
            for record in old_diagnostic_records
            if int(record.get("severity") or 0) >= 4
            and str(record.get("translation_unit_file") or "").strip()
        }
    )
    patched_diagnostic_records = _severe_diagnostic_records(
        patched_touched,
        patched_parse_root or spec.old_root_path,
    )
    patched_diagnostics = Counter(_diagnostic_signature(record) for record in patched_diagnostic_records)
    new_diagnostics: list[dict[str, Any]] = []
    for signature, count in sorted(patched_diagnostics.items()):
        old_count = int(old_diagnostics.get(signature, 0))
        if count <= old_count:
            continue
        examples = [
            record
            for record in patched_diagnostic_records
            if _diagnostic_signature(record) == signature
        ][:4]
        identifier_evidence = _ordered_identifier_evidence(
            [
                dict(item)
                for record in examples
                for item in record.get("identifier_evidence") or []
                if isinstance(item, dict)
            ]
        )
        diagnostic_file = signature[0]
        line_projection = current_line_projections.get(
            diagnostic_file,
            {},
        )
        projection_kind = "exact_unique_unchanged_diff_block"
        if (
            not line_projection
            and diagnostic_file not in source_touched
            and (spec.old_root_path / diagnostic_file).is_file()
        ):
            line_projection = {
                int((record.get("location") or {}).get("line") or 0): int(
                    (record.get("location") or {}).get("line") or 0
                )
                for record in examples
                if int((record.get("location") or {}).get("line") or 0)
                > 0
            }
            projection_kind = "exact_unchanged_overlay_file"
        patched_locations, current_locations = (
            _project_diagnostic_locations_to_current(
                examples,
                relative_path=diagnostic_file,
                line_projection=line_projection,
                projection_kind=projection_kind,
            )
        )
        new_diagnostics.append(
            {
                "file": signature[0],
                "translation_unit_files": [signature[1]],
                "severity": signature[2],
                "spelling": signature[3],
                "old_count": old_count,
                "patched_count": int(count),
                "locations": patched_locations,
                "current_locations": current_locations,
                "identifier_evidence": identifier_evidence,
                "identifiers": [
                    item["identifier"] for item in identifier_evidence
                ],
            }
        )

    report = {
        "status": "checked",
        "touched_source_files": source_touched,
        "old_fatal_translation_units": old_fatal_translation_units,
        "parsed_source_files": parse_files,
        "patch_change_ranges": patch_change_ranges,
        "old_global": (
            {
                "context": old_baseline.get("context"),
                "file_count": old_baseline.get("file_count"),
                "parse_ok_count": old_baseline.get("parse_ok_count"),
                "parse_failed_count": old_baseline.get("parse_failed_count"),
            }
            if old_baseline
            else {
                "status": "not_built",
                "reason": "no new provider definitions in touched files",
            }
        ),
        "old_touched": {
            "context": old_touched_context_summary,
            "parse_ok_count": old_touched.get("parse_ok_count"),
            "parse_failed_count": old_touched.get("parse_failed_count"),
        },
        "patched_touched": {
            "context": patched_context_summary,
            "parse_ok_count": patched_touched.get("parse_ok_count"),
            "parse_failed_count": patched_touched.get("parse_failed_count"),
        },
        "added_provider_definitions": added_provider_definitions,
        "added_contract_definitions": added_contract_definitions,
        "split_function_contracts": split_function_contracts,
        "duplicate_providers": duplicate_providers,
        "new_diagnostics": new_diagnostics,
        "new_unresolved_references": new_unresolved,
        "removed_local_contracts": removed_local_contracts,
        "orphaned_patch_references": orphaned_patch_references,
        "non_authoritative_diagnostic_files": [
            relative_path
            for relative_path in parse_files
            if Path(relative_path).suffix.lower()
            not in TRANSLATION_UNIT_SUFFIXES
        ],
    }
    return _provider_contract_decision(
        report,
        expected_provider_obligations=(
            expected_provider_obligations
        ),
    )
