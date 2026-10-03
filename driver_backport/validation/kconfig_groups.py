#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from driver_backport.validation.build_utils import parse_kconfig_assignment
from driver_backport.validation.kconfig_sync import (
    dependency_expr_requirements,
    expression_identifiers,
    kconfig_index,
)
DEFAULT_CONTEXTS = {
    "native": {},
    "arm64": {"ARCH": "arm64"},
    "arm": {"ARCH": "arm"},
    "mips": {"ARCH": "mips"},
    "riscv": {"ARCH": "riscv"},
    "powerpc": {"ARCH": "powerpc"},
}


@dataclass(frozen=True)
class KconfigConfigClass:
    assignment: str
    symbol: str
    context_label: str
    dependency_symbols: tuple[str, ...]
    arch_symbols: tuple[str, ...]
    compile_test_reachable: bool
    unresolved_exprs: tuple[str, ...]
    reason: str


def _contexts_for_symbol_definition(
    roots: list[Path],
    index: dict[str, dict[str, Any]],
    symbol: str,
) -> tuple[str, ...]:
    entry = index.get(symbol) or {}
    contexts: set[str] = set()
    for raw_path in entry.get("defined_in") or []:
        path = Path(str(raw_path)).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        for root in roots:
            try:
                relative = resolved.relative_to(root.expanduser().resolve())
            except ValueError:
                continue
            parts = relative.parts
            if len(parts) >= 2 and parts[0] == "arch" and parts[1] in DEFAULT_CONTEXTS:
                contexts.add(parts[1])
    return tuple(sorted(contexts))


def _walk_dependency_symbols(
    roots: list[Path],
    index: dict[str, dict[str, Any]],
    symbol: str,
) -> tuple[list[str], list[str], list[str]]:
    queue = [symbol]
    seen: set[str] = set()
    dependencies: list[str] = []
    arch_symbols: list[str] = []
    unresolved: list[str] = []
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        entry = index.get(current) or {}
        for expr in entry.get("depends") or []:
            expr_text = str(expr)
            tokens = expression_identifiers(expr_text)
            requirements = dependency_expr_requirements(expr_text)
            if requirements.get("unresolved_terms"):
                if expr_text not in unresolved:
                    unresolved.append(expr_text)
            for token in tokens:
                if token not in dependencies:
                    dependencies.append(token)
                if _contexts_for_symbol_definition(roots, index, token) and token not in arch_symbols:
                    arch_symbols.append(token)
                if token in index and token not in seen:
                    queue.append(token)
        for selected in entry.get("selects") or []:
            selected_symbol = str(selected)
            if selected_symbol not in dependencies:
                dependencies.append(selected_symbol)
            if _contexts_for_symbol_definition(roots, index, selected_symbol) and selected_symbol not in arch_symbols:
                arch_symbols.append(selected_symbol)
            if selected_symbol in index and selected_symbol not in seen:
                queue.append(selected_symbol)
    return dependencies, arch_symbols, unresolved


def _direct_dependency_symbols(
    index: dict[str, dict[str, Any]],
    symbol: str,
) -> tuple[list[str], list[str], list[str]]:
    entry = index.get(symbol) or {}
    dependencies: list[str] = []
    deterministic: list[str] = []
    unresolved: list[str] = []
    for expr in entry.get("depends") or []:
        expr_text = str(expr)
        requirements = dependency_expr_requirements(expr_text)
        if requirements.get("unresolved_terms"):
            if expr_text not in unresolved:
                unresolved.append(expr_text)
        for token in expression_identifiers(expr_text):
            if token not in dependencies:
                dependencies.append(token)
        for token in requirements.get("required_symbols") or []:
            if token in index and token not in deterministic:
                deterministic.append(token)
    return dependencies, deterministic, unresolved


def _compile_test_is_standalone_or_term(expr: str) -> bool:
    for term in re.split(r"\|\|", expr):
        cleaned = term.strip()
        cleaned = cleaned.strip("() ")
        if cleaned == "COMPILE_TEST":
            return True
    return False


def classify_kconfig_assignment_for_roots(
    roots: list[Path],
    assignment: str,
    *,
    preferred_context: str | None = None,
) -> KconfigConfigClass:
    name, _value = parse_kconfig_assignment(assignment)
    symbol = name.removeprefix("CONFIG_")
    index = kconfig_index(
        tuple(str(root.expanduser().resolve()) for root in roots)
    )
    dependencies, _arch_symbols, unresolved = _walk_dependency_symbols(roots, index, symbol)
    direct_dependencies, deterministic_dependencies, direct_unresolved = _direct_dependency_symbols(index, symbol)
    deterministic_arch_symbols = [
        token
        for token in deterministic_dependencies
        if _contexts_for_symbol_definition(roots, index, token)
    ]
    alternative_arch_symbols = [
        token
        for token in direct_dependencies
        if token not in deterministic_dependencies
        and _contexts_for_symbol_definition(roots, index, token)
    ]
    context_arch_symbols = list(dict.fromkeys([*deterministic_arch_symbols, *alternative_arch_symbols]))
    deterministic_contexts = sorted(
        {
            context
            for arch_symbol in deterministic_arch_symbols
            for context in _contexts_for_symbol_definition(
                roots,
                index,
                arch_symbol,
            )
        }
    )
    alternative_contexts = sorted(
        {
            context
            for arch_symbol in alternative_arch_symbols
            for context in _contexts_for_symbol_definition(
                roots,
                index,
                arch_symbol,
            )
        }
    )

    compile_test_reachable = "COMPILE_TEST" in direct_dependencies
    standalone_compile_test = any(
        _compile_test_is_standalone_or_term(expr)
        for expr in direct_unresolved
    )
    if compile_test_reachable and standalone_compile_test:
        label = "native"
        reason = "standalone_compile_test_reaches_arch_dependency"
    elif preferred_context in deterministic_contexts:
        label = str(preferred_context)
        reason = "compile_context_resolves_arch_dependency:" + ",".join(
            deterministic_arch_symbols
        )
    elif len(deterministic_contexts) == 1:
        label = deterministic_contexts[0]
        reason = "deterministic_arch_dependency:" + ",".join(deterministic_arch_symbols)
    elif len(deterministic_contexts) > 1:
        label = "ambiguous[" + ",".join(deterministic_contexts) + "]"
        reason = "conflicting_deterministic_arch_dependencies:" + ",".join(deterministic_arch_symbols)
    elif preferred_context in alternative_contexts:
        label = str(preferred_context)
        reason = "compile_context_resolves_arch_alternative:" + ",".join(
            alternative_arch_symbols
        )
    elif alternative_contexts:
        label = "ambiguous[" + ",".join(["native", *alternative_contexts]) + "]"
        reason = "unresolved_arch_alternative:" + ",".join(alternative_arch_symbols)
    else:
        label = "native"
        reason = "no_deterministic_arch_dependency"

    return KconfigConfigClass(
        assignment=assignment,
        symbol=symbol,
        context_label=label,
        dependency_symbols=tuple(dependencies),
        arch_symbols=tuple(context_arch_symbols),
        compile_test_reachable=compile_test_reachable,
        unresolved_exprs=tuple(dict.fromkeys([*direct_unresolved, *unresolved])),
        reason=reason,
    )
