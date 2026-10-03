#!/usr/bin/env python3
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from driver_backport.core.common import (
    append_unique_string as _append_unique,
    tri_and,
    tri_or,
)
from driver_backport.validation.build_utils import (
    parse_kconfig_assignment,
)


KCONFIG_SYMBOL_RE = re.compile(r"^\s*(?:menuconfig|config)\s+([A-Za-z0-9_]+)\b", re.MULTILINE)
KCONFIG_ENTRY_RE = re.compile(r"^\s*(?:menuconfig|config)\s+([A-Za-z0-9_]+)\b")
KCONFIG_TOP_IF_RE = re.compile(r"^if\s+(.+)$")
KCONFIG_TOP_ENDIF_RE = re.compile(r"^endif\b")
KCONFIG_TOP_MENU_RE = re.compile(r"^menu\b")
KCONFIG_TOP_ENDMENU_RE = re.compile(r"^endmenu\b")
KCONFIG_VISIBLE_RE = re.compile(r"^visible\s+if\s+(.+)$")
KCONFIG_SOURCE_RE = re.compile(r"^\s*(?:o?r?source|source)\s+[\"']?([^\"'\s]+)[\"']?")
KCONFIG_DEPENDS_RE = re.compile(r"^\s*depends\s+on\s+(.+)$")
KCONFIG_SELECT_RE = re.compile(r"^\s*select\s+([A-Za-z0-9_]+)(?:\s+if\s+(.+))?")
KCONFIG_TYPE_RE = re.compile(r"^\s*(bool|boolean|tristate)\b")
KCONFIG_PROMPT_RE = re.compile(r"^\s*prompt\s+['\"]")
KCONFIG_HELP_RE = re.compile(r"^\s*(?:help|---help---)\s*$")
KCONFIG_EXPR_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
KCONFIG_EXPR_COMPARISON_RE = re.compile(r"(?<![<>=!])(?:!=|=|<=|>=|<|>)(?![=])")
KCONFIG_EXPR_OPERATORS = {"AND", "OR", "NOT"}
KCONFIG_TRUE_VALUES = {"y", "m"}
KCONFIG_BACKUP_SUFFIXES = {".orig", ".bak", ".backup", ".rej"}


def _append_unique_dict(items: list[dict[str, Any]], value: dict[str, Any]) -> None:
    if value not in items:
        items.append(value)


def _pop_context(context_stack: list[dict[str, Any]], kind: str) -> None:
    for index in range(len(context_stack) - 1, -1, -1):
        if context_stack[index].get("kind") == kind:
            del context_stack[index]
            return


def read_config_value(config_path: Path, name: str) -> str | None:
    if not config_path.exists():
        return None
    disabled = f"# {name} is not set"
    prefix = f"{name}="
    for line in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped == disabled:
            return "n"
        if stripped.startswith(prefix):
            return stripped.split("=", 1)[1].strip()
    return None


def _strip_kconfig_inline_comment(text: str) -> str:
    return text.split("#", 1)[0].strip()


def _merge_kconfig_entry(index: dict[str, dict[str, Any]], entry: dict[str, Any] | None) -> None:
    if not entry:
        return
    symbol = str(entry["symbol"])
    merged = index.setdefault(
        symbol,
        {
            "symbol": symbol,
            "type": None,
            "prompt": False,
            "depends": [],
            "visibility_depends": [],
            "selects": [],
            "select_relations": [],
            "defined_in": [],
        },
    )
    if entry.get("type"):
        merged["type"] = entry["type"]
    merged["prompt"] = bool(merged.get("prompt") or entry.get("prompt"))
    for value in entry.get("depends") or []:
        _append_unique(merged["depends"], str(value))
    for value in entry.get("visibility_depends") or []:
        _append_unique(merged["visibility_depends"], str(value))
    for value in entry.get("selects") or []:
        _append_unique(merged["selects"], str(value))
    for value in entry.get("select_relations") or []:
        _append_unique_dict(merged["select_relations"], dict(value))
    for value in entry.get("defined_in") or []:
        _append_unique(merged["defined_in"], str(value))


def _parse_kconfig_file(path: Path, inherited_depends: list[str] | None = None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    context_stack: list[dict[str, Any]] = []
    help_indent: int | None = None

    def active_depends() -> list[str]:
        values: list[str] = []
        for expr in inherited_depends or []:
            _append_unique(values, str(expr))
        for context in context_stack:
            for expr in context.get("depends") or []:
                _append_unique(values, str(expr))
        return values

    def active_visibility_depends() -> list[str]:
        values: list[str] = []
        for context in context_stack:
            for expr in context.get("visibility_depends") or []:
                _append_unique(values, str(expr))
        return values

    def close_current() -> None:
        nonlocal current
        _merge_kconfig_entry(index, current)
        current = None

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        indent = len(raw_line.expandtabs(8)) - len(raw_line.expandtabs(8).lstrip())
        if help_indent is not None:
            if not stripped:
                continue
            if indent > help_indent:
                continue
            help_indent = None
        if not stripped or stripped.startswith("#"):
            continue

        if not raw_line[:1].isspace():
            if_match = KCONFIG_TOP_IF_RE.match(stripped)
            if if_match:
                close_current()
                context_stack.append({"kind": "if", "depends": [if_match.group(1).strip()]})
                continue
            if KCONFIG_TOP_ENDIF_RE.match(stripped):
                close_current()
                _pop_context(context_stack, "if")
                continue
            if KCONFIG_TOP_MENU_RE.match(stripped):
                close_current()
                context_stack.append(
                    {
                        "kind": "menu",
                        "depends": [],
                        "visibility_depends": [],
                    }
                )
                continue
            if KCONFIG_TOP_ENDMENU_RE.match(stripped):
                close_current()
                _pop_context(context_stack, "menu")
                continue

        visible_match = KCONFIG_VISIBLE_RE.match(stripped)
        if visible_match and context_stack:
            _append_unique(
                context_stack[-1].setdefault("visibility_depends", []),
                _strip_kconfig_inline_comment(visible_match.group(1)),
            )
            continue

        entry_match = KCONFIG_ENTRY_RE.match(line)
        if entry_match:
            close_current()
            symbol = entry_match.group(1)
            current = {
                "symbol": symbol,
                "type": None,
                "prompt": False,
                "depends": active_depends(),
                "visibility_depends": [
                    *list(inherited_depends or []),
                    *active_visibility_depends(),
                ],
                "selects": [],
                "select_relations": [],
                "defined_in": [str(path)],
            }
            continue

        type_match = KCONFIG_TYPE_RE.match(line)
        if current is not None and type_match and not current.get("type"):
            current["type"] = "bool" if type_match.group(1) == "boolean" else type_match.group(1)
            type_suffix = line[type_match.end() :].strip()
            if type_suffix.startswith(('"', "'")):
                current["prompt"] = True
            continue

        if current is not None and KCONFIG_PROMPT_RE.match(line):
            current["prompt"] = True
            continue

        if current is not None and KCONFIG_HELP_RE.match(line):
            help_indent = indent
            continue

        depends_match = KCONFIG_DEPENDS_RE.match(line)
        if depends_match:
            if current is not None:
                _append_unique(current["depends"], _strip_kconfig_inline_comment(depends_match.group(1)))
            elif context_stack:
                _append_unique(context_stack[-1]["depends"], _strip_kconfig_inline_comment(depends_match.group(1)))
            continue

        select_match = KCONFIG_SELECT_RE.match(line)
        if current is not None and select_match:
            selected_symbol = select_match.group(1).strip()
            condition = (
                _strip_kconfig_inline_comment(select_match.group(2))
                if select_match.group(2)
                else None
            )
            _append_unique(current["selects"], selected_symbol)
            _append_unique_dict(
                current["select_relations"],
                {"symbol": selected_symbol, "if": condition},
            )

    _merge_kconfig_entry(index, current)
    return index


def _resolve_kconfig_source(root: Path, current_path: Path, source_name: str) -> Path | None:
    if "$" in source_name:
        return None
    source_path = Path(source_name)
    if source_path.is_absolute():
        return source_path if source_path.exists() else None
    root_relative = root / source_path
    if root_relative.exists():
        return root_relative
    local_relative = current_path.parent / source_path
    if local_relative.exists():
        return local_relative
    return None


def _collect_sourced_file_depends(root: Path) -> dict[Path, list[list[str]]]:
    source_edges: dict[Path, list[tuple[Path, list[str]]]] = {}

    def append_context(items: list[list[str]], depends: list[str]) -> bool:
        if not depends:
            return False
        context = list(dict.fromkeys(depends))
        if context in items:
            return False
        items.append(context)
        return True

    def append_edge(parent: Path, child: Path, depends: list[str]) -> None:
        edges = source_edges.setdefault(parent.resolve(), [])
        item = (child.resolve(), list(dict.fromkeys(depends)))
        if item not in edges:
            edges.append(item)

    for path in sorted(root.rglob("Kconfig*")):
        if not _is_kconfig_source_file(path):
            continue
        context_stack: list[dict[str, Any]] = []
        current_entry = False

        def active_depends() -> list[str]:
            values: list[str] = []
            for context in context_stack:
                for expr in context.get("depends") or []:
                    _append_unique(values, str(expr))
            return values

        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw_line[:1].isspace():
                entry_match = KCONFIG_ENTRY_RE.match(line)
                if entry_match:
                    current_entry = True
                    continue
                if_match = KCONFIG_TOP_IF_RE.match(stripped)
                if if_match:
                    current_entry = False
                    context_stack.append({"kind": "if", "depends": [if_match.group(1).strip()]})
                    continue
                if KCONFIG_TOP_ENDIF_RE.match(stripped):
                    current_entry = False
                    _pop_context(context_stack, "if")
                    continue
                if KCONFIG_TOP_MENU_RE.match(stripped):
                    current_entry = False
                    context_stack.append({"kind": "menu", "depends": []})
                    continue
                if KCONFIG_TOP_ENDMENU_RE.match(stripped):
                    current_entry = False
                    _pop_context(context_stack, "menu")
                    continue
            depends_match = KCONFIG_DEPENDS_RE.match(line)
            if depends_match and context_stack and not current_entry:
                _append_unique(context_stack[-1]["depends"], _strip_kconfig_inline_comment(depends_match.group(1)))
                continue
            source_match = KCONFIG_SOURCE_RE.match(line)
            if source_match:
                current_entry = False
                resolved = _resolve_kconfig_source(root, path, source_match.group(1).strip())
                if resolved and _is_kconfig_source_file(resolved):
                    try:
                        resolved.resolve().relative_to(path.parent.resolve())
                    except ValueError:
                        continue
                    append_edge(path, resolved, active_depends())

    inherited: dict[Path, list[list[str]]] = {path.resolve(): [] for path in source_edges}

    changed = True
    while changed:
        changed = False
        for parent, edges in source_edges.items():
            parent_contexts = inherited.get(parent, [])
            base_contexts = [[]]
            base_contexts.extend(parent_contexts)
            for child, edge_depends in edges:
                child_contexts = inherited.setdefault(child, [])
                for parent_depends in base_contexts:
                    combined: list[str] = []
                    for expr in parent_depends:
                        _append_unique(combined, str(expr))
                    for expr in edge_depends:
                        _append_unique(combined, str(expr))
                    if append_context(child_contexts, combined):
                        changed = True

    return inherited

def _is_kconfig_source_file(path: Path) -> bool:
    return path.is_file() and path.name.startswith("Kconfig") and path.suffix not in KCONFIG_BACKUP_SUFFIXES


@lru_cache(maxsize=16)
def kconfig_source_edges(root_name: str) -> tuple[tuple[str, str, int], ...]:
    """Return exact, resolved Kconfig source edges inside one kernel tree."""

    root = Path(root_name).resolve()
    if not root.exists():
        return ()
    edges: list[tuple[str, str, int]] = []
    seen: set[tuple[str, str, int]] = set()
    for parent in sorted(root.rglob("Kconfig*")):
        if not _is_kconfig_source_file(parent):
            continue
        for line_number, raw_line in enumerate(
            parent.read_text(encoding="utf-8", errors="ignore").splitlines(),
            start=1,
        ):
            match = KCONFIG_SOURCE_RE.match(raw_line)
            if match is None:
                continue
            child = _resolve_kconfig_source(root, parent, match.group(1).strip())
            if child is None or not _is_kconfig_source_file(child):
                continue
            try:
                parent_relative = parent.resolve().relative_to(root).as_posix()
                child_relative = child.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            edge = (parent_relative, child_relative, line_number)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
    return tuple(edges)


@lru_cache(maxsize=16)
def kconfig_index(root_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for root_name in root_names:
        root = Path(root_name)
        if not root.exists():
            continue
        inherited_depends = _collect_sourced_file_depends(root)
        for path in sorted(root.rglob("Kconfig*")):
            if not _is_kconfig_source_file(path):
                continue
            parse_contexts = [None, *inherited_depends.get(path.resolve(), [])]
            for inherited in parse_contexts:
                for entry in _parse_kconfig_file(path, inherited_depends=inherited).values():
                    _merge_kconfig_entry(index, entry)
    return index


@lru_cache(maxsize=16)
def available_kconfig_symbols(root: Path) -> frozenset[str]:
    symbols: set[str] = set()
    for path in root.rglob("Kconfig*"):
        if not _is_kconfig_source_file(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        symbols.update(KCONFIG_SYMBOL_RE.findall(text))
    return frozenset(symbols)


def clear_kconfig_caches() -> None:
    kconfig_index.cache_clear()
    kconfig_source_edges.cache_clear()
    available_kconfig_symbols.cache_clear()


def _outer_parens_wrap(text: str) -> bool:
    if not (text.startswith("(") and text.endswith(")")):
        return False
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
        if depth < 0:
            return False
    return depth == 0


def _strip_outer_parens(text: str) -> str:
    stripped = text.strip()
    while _outer_parens_wrap(stripped):
        stripped = stripped[1:-1].strip()
    return stripped


def _word_operator_at(text: str, index: int, operator: str) -> bool:
    end = index + len(operator)
    if text[index:end].upper() != operator:
        return False
    before = text[index - 1] if index > 0 else " "
    after = text[end] if end < len(text) else " "
    return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")


def _split_top_level(expr: str, operators: tuple[str, ...]) -> list[str]:
    text = expr.strip()
    parts: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0:
            matched = None
            for operator in operators:
                if operator in {"&&", "||"}:
                    if text.startswith(operator, index):
                        matched = operator
                        break
                elif _word_operator_at(text, index, operator):
                    matched = operator
                    break
            if matched:
                part = text[start:index].strip()
                if part:
                    parts.append(part)
                index += len(matched)
                start = index
                continue
        index += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts or ([text] if text else [])


def _has_top_level_operator(expr: str, operators: tuple[str, ...]) -> bool:
    return len(_split_top_level(expr, operators)) > 1


def _expr_tokens(expr: str) -> list[str]:
    result: list[str] = []
    for match in KCONFIG_EXPR_TOKEN_RE.finditer(expr):
        token = match.group(0)
        token_upper = token.upper()
        if token_upper in KCONFIG_EXPR_OPERATORS or token.lower() in {"y", "m", "n"}:
            continue
        prefix = expr[max(0, match.start() - 3) : match.start()]
        if "!" in prefix:
            continue
        _append_unique(result, token)
    return result


def _merge_assignment_options(
    left: list[tuple[str, str]],
    right: list[tuple[str, str]],
) -> list[tuple[str, str]] | None:
    values = dict(left)
    for symbol, value in right:
        previous = values.get(symbol)
        if previous is not None and previous != value:
            return None
        values[symbol] = value
    return list(values.items())


def _dependency_expr_assignment_options(
    expr: str,
    *,
    desired: bool = True,
    required_level: str = "y",
) -> list[list[tuple[str, str]]] | None:
    """Return assignment alternatives for the supported Kconfig expression subset."""

    text = _strip_outer_parens(expr)
    if not text:
        return None

    operator = ("||", "OR") if desired else ("&&", "AND")
    if _has_top_level_operator(text, operator):
        alternatives: list[list[tuple[str, str]]] = []
        for part in _split_top_level(text, operator):
            options = _dependency_expr_assignment_options(
                part,
                desired=desired,
                required_level=required_level,
            )
            if options is None:
                return None
            alternatives.extend(options)
        return alternatives

    operator = ("&&", "AND") if desired else ("||", "OR")
    if _has_top_level_operator(text, operator):
        combined: list[list[tuple[str, str]]] = [[]]
        for part in _split_top_level(text, operator):
            options = _dependency_expr_assignment_options(
                part,
                desired=desired,
                required_level=required_level,
            )
            if options is None:
                return None
            next_combined: list[list[tuple[str, str]]] = []
            for existing in combined:
                for option in options:
                    merged = _merge_assignment_options(existing, option)
                    if merged is not None and merged not in next_combined:
                        next_combined.append(merged)
            combined = next_combined
        return combined

    if text.startswith("!"):
        return _dependency_expr_assignment_options(
            text[1:],
            desired=not desired,
            required_level=required_level,
        )

    comparison = re.fullmatch(
        r"([A-Za-z][A-Za-z0-9_]*)\s*(=|!=)\s*([ymn])",
        text,
        re.IGNORECASE,
    )
    if comparison:
        symbol, comparison_operator, value = comparison.groups()
        equality_required = desired == (comparison_operator == "=")
        if not equality_required:
            return None
        return [[(symbol, value.lower())]]

    lowered = text.lower()
    if lowered in {"y", "m", "n"}:
        truth = lowered in {"y", "m"}
        return [[]] if truth == desired else []
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", text):
        return [[(text, required_level if desired else "n")]]
    return None


def _dependency_expr_requirements(
    expr: str,
) -> dict[str, Any]:
    """Project deterministic positive dependencies from the canonical parser."""

    options = _dependency_expr_assignment_options(expr)
    if options is None:
        return {
            "required_symbols": [],
            "unresolved_terms": [
                {"term": _strip_outer_parens(expr), "reason": "unsupported_expression"}
            ],
        }
    unique_options: list[list[tuple[str, str]]] = []
    for option in options:
        if option not in unique_options:
            unique_options.append(option)
    if len(unique_options) != 1:
        return {
            "required_symbols": [],
            "unresolved_terms": [
                {"term": _strip_outer_parens(expr), "reason": "alternative"}
            ],
        }
    required: list[str] = []
    unresolved: list[dict[str, str]] = []
    for symbol, value in unique_options[0]:
        if value in {"y", "m"}:
            _append_unique(required, symbol)
        else:
            unresolved.append(
                {
                    "term": f"{symbol}={value}",
                    "reason": "non_positive_assignment",
                }
            )
    return {
        "required_symbols": required,
        "unresolved_terms": unresolved,
    }


def _required_symbols_from_expr(expr: str) -> list[str]:
    return list(_dependency_expr_requirements(expr)["required_symbols"])


def dependency_expr_requirements(
    expr: str,
) -> dict[str, Any]:
    """Return only dependencies that are deterministic under Kconfig semantics."""

    return _dependency_expr_requirements(expr)


def expression_symbols(expr: str, index: dict[str, dict[str, Any]]) -> list[str]:
    """Return referenced symbols known to the canonical Kconfig index."""

    return [symbol for symbol in _expr_tokens(expr) if symbol in index]


def expression_identifiers(expr: str) -> list[str]:
    """Return identifiers parsed from a Kconfig expression, including built-ins."""

    return _expr_tokens(expr)


def expression_symbol_values(
    config_path: Path,
    expr: str,
    index: dict[str, dict[str, Any]],
) -> dict[str, str | None]:
    return {
        symbol: read_config_value(config_path, f"CONFIG_{symbol}")
        for symbol in expression_symbols(expr, index)
    }


def dependency_expr_satisfied(
    config_path: Path,
    expr: str,
    index: dict[str, dict[str, Any]],
    *,
    symbol_values: dict[str, str] | None = None,
) -> bool | None:
    """Evaluate the boolean subset of Kconfig expressions against one .config.

    Comparisons and unknown symbols remain explicitly indeterminate instead of
    being guessed true or false.
    """

    if KCONFIG_EXPR_COMPARISON_RE.search(expr):
        return None
    tokens = re.findall(r"&&|\|\||!|\(|\)|[A-Za-z][A-Za-z0-9_]*", expr)
    if not tokens:
        return None
    position = 0

    def token_value(token: str) -> bool | None:
        lowered = token.lower()
        if lowered in {"y", "m"}:
            return True
        if lowered == "n":
            return False
        if token.upper() in KCONFIG_EXPR_OPERATORS or token not in index:
            return None
        if symbol_values is not None and token in symbol_values:
            return symbol_values[token] in KCONFIG_TRUE_VALUES
        return read_config_value(config_path, f"CONFIG_{token}") in KCONFIG_TRUE_VALUES

    def parse_primary() -> bool | None:
        nonlocal position
        if position >= len(tokens):
            return None
        token = tokens[position]
        if token == "(":
            position += 1
            value = parse_or()
            if position < len(tokens) and tokens[position] == ")":
                position += 1
            return value
        if token == "!":
            position += 1
            value = parse_primary()
            return None if value is None else not value
        if token == ")":
            return None
        position += 1
        return token_value(token)

    def parse_and() -> bool | None:
        nonlocal position
        value = parse_primary()
        while position < len(tokens) and tokens[position] == "&&":
            position += 1
            value = tri_and(value, parse_primary())
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


def kconfig_dependency_closure(
    *,
    config_path: Path,
    seed_configs: list[str],
    kconfig_roots: list[Path],
    follow_selects: bool = True,
) -> dict[str, Any]:
    """Build one dependency closure from the canonical Kconfig index."""

    index = kconfig_index(tuple(str(path.resolve()) for path in kconfig_roots))
    queue: list[tuple[str, str]] = []
    seen: set[str] = set()
    symbols: list[dict[str, Any]] = []
    blocked_exprs: list[dict[str, Any]] = []
    for config_name in seed_configs:
        symbol = str(config_name).removeprefix("CONFIG_")
        if symbol:
            queue.append((symbol, "seed"))

    while queue:
        symbol, reason = queue.pop(0)
        if symbol in seen:
            continue
        seen.add(symbol)
        entry = index.get(symbol) or {}
        value = read_config_value(config_path, f"CONFIG_{symbol}")
        depends: list[dict[str, Any]] = []
        dependency_exprs = [
            *[("visibility depends on", str(expr)) for expr in entry.get("visibility_depends") or []],
            *[("depends on", str(expr)) for expr in entry.get("depends") or []],
        ]
        for relation, expr_text in dependency_exprs:
            required_symbols = _required_symbols_from_expr(expr_text)
            status = dependency_expr_satisfied(config_path, expr_text, index)
            values = expression_symbol_values(config_path, expr_text, index)
            depends.append(
                {
                    "expr": expr_text,
                    "required_symbols": required_symbols,
                    "symbol_values": values,
                    "satisfied": status,
                }
            )
            if status is False:
                blocked_exprs.append(
                    {"symbol": symbol, "expr": expr_text, "symbol_values": values}
                )
            for dependency in required_symbols:
                queue.append((dependency, f"{symbol} {relation} {expr_text}"))

        select_relations = _select_relations(entry)
        active_select_relations = [
            relation
            for relation in select_relations
            if not relation.get("if")
            or dependency_expr_satisfied(
                config_path,
                str(relation["if"]),
                index,
            ) is True
        ]
        selects = [str(item["symbol"]) for item in active_select_relations]
        if follow_selects:
            queue.extend((selected, f"{symbol} selects {selected}") for selected in selects)
        symbols.append(
            {
                "symbol": symbol,
                "config": f"CONFIG_{symbol}",
                "value": value,
                "type": entry.get("type"),
                "defined_in": entry.get("defined_in", []),
                "reason": reason,
                "depends": depends,
                "selects": selects,
                "select_relations": select_relations,
                "defined": bool(entry),
            }
        )

    return {
        "seed_configs": seed_configs,
        "follow_selects": follow_selects,
        "symbols": symbols,
        "missing_or_disabled_symbols": [
            item for item in symbols if item["value"] not in KCONFIG_TRUE_VALUES
        ],
        "blocked_dependency_expressions": blocked_exprs,
        "undefined_symbols": [item for item in symbols if not item.get("defined")],
    }


def _selected_kconfig_value(
    *,
    symbol: str,
    requested_value: str,
    index: dict[str, dict[str, Any]],
) -> str:
    entry_type = str((index.get(symbol) or {}).get("type") or "")
    if entry_type == "tristate" and requested_value in {"m", "y"}:
        return requested_value
    return "y"


def _normalize_assignment_value(symbol: str, value: str, index: dict[str, dict[str, Any]]) -> str:
    entry_type = str((index.get(symbol) or {}).get("type") or "")
    if value == "m" and entry_type in {"bool", "boolean"}:
        return "y"
    return value


def _assignment_satisfied(expected: str | None, actual: str | None) -> bool:
    if actual == expected:
        return True
    return expected == "m" and actual == "y"


def _select_relations(entry: dict[str, Any]) -> list[dict[str, str | None]]:
    relations = [
        {
            "symbol": str(item.get("symbol") or ""),
            "if": str(item.get("if")) if item.get("if") else None,
        }
        for item in entry.get("select_relations") or []
        if str(item.get("symbol") or "")
    ]
    if relations:
        return relations
    return [
        {"symbol": str(symbol), "if": None}
        for symbol in entry.get("selects") or []
        if str(symbol)
    ]


def expand_kconfig_assignments(
    *,
    config_path: Path,
    assignments: list[str],
    kconfig_roots: list[Path],
) -> tuple[list[str], list[dict[str, Any]]]:
    index = kconfig_index(tuple(str(path.resolve()) for path in kconfig_roots))
    values: dict[str, str] = {}
    reasons: dict[str, list[str]] = {}
    unresolved_depends: dict[str, list[dict[str, str]]] = {}
    queue: list[str] = []

    def add(symbol: str, value: str, reason: str) -> None:
        value = _normalize_assignment_value(symbol, value, index)
        if symbol in values:
            _append_unique(reasons.setdefault(symbol, []), reason)
            current = values[symbol]
            strength = {"n": 0, "m": 1, "y": 2}
            if current in strength and value in strength and strength[value] > strength[current]:
                values[symbol] = value
                if symbol not in queue:
                    queue.append(symbol)
            return
        values[symbol] = value
        reasons[symbol] = [reason]
        queue.append(symbol)

    def add_dependency(
        *,
        symbol: str,
        parent: str,
        expr: str,
        relation: str,
        required_value: str,
    ) -> None:
        actual = values.get(symbol) or read_config_value(
            config_path,
            f"CONFIG_{symbol}",
        )
        reason = f"{parent} {relation} {expr}"
        if _assignment_satisfied(required_value, actual):
            return
        # A promptless symbol cannot necessarily retain a direct user value,
        # but it still participates in the dependency graph.  Queue it so its
        # own visibility/dependency chain is materialized; olddefconfig can
        # then realize it through defaults/selects once those prompted parents
        # are enabled.  Final assignment verification remains fail-closed when
        # the hidden symbol cannot actually become active.
        add(symbol, required_value, reason)

    def add_expression_dependencies(
        *,
        parent: str,
        requested_value: str,
        expr: str,
        relation: str,
    ) -> None:
        options = _dependency_expr_assignment_options(
            expr,
            required_level=requested_value if requested_value in {"m", "y"} else "y",
        )
        if options is None:
            _append_unique_dict(
                unresolved_depends.setdefault(parent, []),
                {
                    "expr": expr,
                    "term": expr,
                    "reason": "unsupported_expression",
                },
            )
            return

        changed_options: list[list[tuple[str, str]]] = []
        for option in options:
            changes = [
                (symbol, value)
                for symbol, value in option
                if not _assignment_satisfied(
                    value,
                    values.get(symbol)
                    or read_config_value(config_path, f"CONFIG_{symbol}")
                    or "n",
                )
            ]
            if not changes:
                return
            if changes not in changed_options:
                changed_options.append(changes)

        if not changed_options:
            _append_unique_dict(
                unresolved_depends.setdefault(parent, []),
                {"expr": expr, "term": expr, "reason": "unsatisfiable"},
            )
            return

        if len(changed_options) > 1:
            _append_unique_dict(
                unresolved_depends.setdefault(parent, []),
                {"expr": expr, "term": expr, "reason": "alternative"},
            )
            return

        for symbol, value in changed_options[0]:
            add_dependency(
                symbol=symbol,
                parent=parent,
                expr=expr,
                relation=relation,
                required_value=value,
            )

    for assignment in assignments:
        name, value = parse_kconfig_assignment(assignment)
        add(name.removeprefix("CONFIG_"), value, "explicit build_kconfig")

    while queue:
        symbol = queue.pop(0)
        requested_value = values[symbol]
        entry = index.get(symbol) or {}
        for expr in entry.get("visibility_depends") or []:
            expr_text = str(expr)
            add_expression_dependencies(
                parent=symbol,
                requested_value=requested_value,
                expr=expr_text,
                relation="visibility depends on",
            )
        for expr in entry.get("depends") or []:
            expr_text = str(expr)
            add_expression_dependencies(
                parent=symbol,
                requested_value=requested_value,
                expr=expr_text,
                relation="depends on",
            )
        for relation in _select_relations(entry):
            selected_symbol = str(relation["symbol"])
            condition = str(relation.get("if") or "")
            if condition and dependency_expr_satisfied(
                config_path,
                condition,
                index,
                symbol_values=values,
            ) is not True:
                continue
            add(
                selected_symbol,
                _selected_kconfig_value(
                    symbol=selected_symbol,
                    requested_value=requested_value,
                    index=index,
                ),
                f"{symbol} selects {selected_symbol}"
                + (f" if {condition}" if condition else ""),
            )

    expanded = [f"CONFIG_{symbol}={value}" for symbol, value in values.items()]
    records = [
        {
            "symbol": symbol,
            "assignment": f"CONFIG_{symbol}={value}",
            "value": value,
            "type": (index.get(symbol) or {}).get("type"),
            "prompt": bool((index.get(symbol) or {}).get("prompt")),
            "reasons": reasons.get(symbol, []),
            "defined_in": (index.get(symbol) or {}).get("defined_in", []),
            "unresolved_depends": unresolved_depends.get(symbol, []),
        }
        for symbol, value in values.items()
    ]
    return expanded, records


def partition_promptless_explicit_mismatches(
    mismatches: list[dict[str, str | None]],
    expansion_records: list[dict[str, Any]],
    *,
    allow_for_exact_objects: bool,
) -> tuple[list[dict[str, str | None]], list[dict[str, str | None]]]:
    """Separate real config failures from hidden symbols handled by exact object builds."""
    if not allow_for_exact_objects:
        return list(mismatches), []
    records_by_symbol = {
        str(record.get("symbol") or ""): record
        for record in expansion_records
    }
    fatal: list[dict[str, str | None]] = []
    direct_object_required: list[dict[str, str | None]] = []
    for mismatch in mismatches:
        symbol = str(mismatch.get("name") or "").removeprefix("CONFIG_")
        record = records_by_symbol.get(symbol) or {}
        explicit = "explicit build_kconfig" in set(record.get("reasons") or [])
        if explicit and record and not bool(record.get("prompt")):
            direct_object_required.append(mismatch)
        else:
            fatal.append(mismatch)
    return fatal, direct_object_required
