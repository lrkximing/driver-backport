#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Iterable

from driver_backport.analysis.kbuild_index import KbuildIndex
from driver_backport.core.common import normalize_relpath, stable_id


EXPORT_INVOCATION_RE = re.compile(
    r"\b(?P<macro>EXPORT_(?:TRACEPOINT_)?SYMBOL[A-Z0-9_]*)\s*"
    r"\(\s*(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\s*,\s*(?P<argument>[^)]+?))?\s*\)\s*;?",
    re.MULTILINE,
)


def _mask_comments(text: str) -> str:
    result = list(text)
    index = 0
    state = "code"
    while index < len(text):
        current = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "code" and current == "/" and following == "/":
            result[index] = result[index + 1] = " "
            index += 2
            state = "line_comment"
            continue
        if state == "code" and current == "/" and following == "*":
            result[index] = result[index + 1] = " "
            index += 2
            state = "block_comment"
            continue
        if state == "code" and current == '"':
            state = "string"
        elif state == "code" and current == "'":
            state = "character"
        elif state == "string" and current == '"' and (index == 0 or text[index - 1] != "\\"):
            state = "code"
        elif state == "character" and current == "'" and (index == 0 or text[index - 1] != "\\"):
            state = "code"
        elif state == "line_comment":
            if current == "\n":
                state = "code"
            else:
                result[index] = " "
        elif state == "block_comment":
            if current == "*" and following == "/":
                result[index] = result[index + 1] = " "
                index += 2
                state = "code"
                continue
            if current != "\n":
                result[index] = " "
        index += 1
    return "".join(result)


def parse_export_contracts(text: str, *, file: str, tree: str) -> list[dict[str, Any]]:
    masked = _mask_comments(text)
    records: list[dict[str, Any]] = []
    relative_path = normalize_relpath(file)
    for match in EXPORT_INVOCATION_RE.finditer(masked):
        macro = match.group("macro")
        symbol = match.group("symbol")
        argument = str(match.group("argument") or "").strip()
        line = text.count("\n", 0, match.start()) + 1
        end_line = text.count("\n", 0, match.end()) + 1
        raw = text[match.start() : match.end()].strip()
        records.append(
            {
                "id": stable_id("export", tree, relative_path, symbol, macro, str(line)),
                "tree": tree,
                "file": relative_path,
                "line": line,
                "end_line": end_line,
                "symbol": symbol,
                "macro": macro,
                "namespace_or_modules": argument or None,
                "raw": raw,
                "contract_key": f"export:{symbol}",
            }
        )
    return records


def export_contract_relation(
    current_exports: Iterable[dict[str, Any]],
    donor_exports: Iterable[dict[str, Any]],
    *,
    required: bool,
) -> str:
    """Compare one demanded export facet with a shared relation vocabulary."""

    if not required:
        return "not_required"
    current = [item for item in current_exports if isinstance(item, dict)]
    donor = [item for item in donor_exports if isinstance(item, dict)]
    if not donor:
        return "donor_contract_missing"
    if not current:
        return "missing_in_old"
    current_shapes = {
        (
            str(item.get("macro") or ""),
            str(item.get("namespace_or_modules") or ""),
        )
        for item in current
    }
    donor_shapes = {
        (
            str(item.get("macro") or ""),
            str(item.get("namespace_or_modules") or ""),
        )
        for item in donor
    }
    return "equivalent" if current_shapes == donor_shapes else "different"


def index_export_contracts(
    root: Path,
    files: Iterable[object],
    *,
    tree: str,
) -> dict[str, list[dict[str, Any]]]:
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root = root.resolve()
    for value in sorted({normalize_relpath(str(item or "")) for item in files if str(item or "").strip()}):
        path = (root / value).resolve()
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for record in parse_export_contracts(text, file=value, tree=tree):
            by_symbol[str(record["symbol"])].append(record)
    return dict(by_symbol)


def module_ownership_for_sources(
    root: Path,
    source_files: Iterable[object],
    *,
    tree: str,
) -> dict[str, list[dict[str, Any]]]:
    root = root.resolve()
    sources = sorted(
        {
            normalize_relpath(str(item or ""))
            for item in source_files
            if str(item or "").strip()
        }
    )
    by_directory: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        by_directory[normalize_relpath(str(Path(source).parent.as_posix()))].append(source)

    ownership: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for directory, directory_sources in sorted(by_directory.items()):
        index = KbuildIndex.from_directory(root, directory)
        contract_graph = index.build_contract_graph()
        module_to_sources = dict(contract_graph.get("module_to_sources") or {})
        source_to_configs = dict(contract_graph.get("source_to_configs") or {})
        for module, members in module_to_sources.items():
            module_path = normalize_relpath(f"{directory}/{module}")
            for source in members:
                normalized_source = normalize_relpath(source)
                if normalized_source not in directory_sources:
                    continue
                ownership[normalized_source].append(
                    {
                        "id": stable_id("module-owner", tree, normalized_source, module_path),
                        "tree": tree,
                        "source": normalized_source,
                        "module": module_path,
                        "configs": list(source_to_configs.get(normalized_source) or []),
                    }
                )
    return {
        source: sorted(records, key=lambda item: (str(item.get("module") or ""), str(item.get("id") or "")))
        for source, records in ownership.items()
    }
