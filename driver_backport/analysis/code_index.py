#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from driver_backport.core.common import (
    int_or_zero as _int_or_zero,
    normalize_relpath,
    normalize_root_relative,
    ordered_unique_relpaths as _ordered_unique,
    stable_id,
)
from driver_backport.analysis.conditional_contract import (
    conditional_variant_insertions,
)
from driver_backport.analysis.kbuild_index import KbuildIndex
from driver_backport.analysis.entity_fingerprint import (
    compare_entity_fingerprints,
    cursor_fingerprints,
    cursor_signature_payload,
)
from driver_backport.scope.libclang_index import (
    ClangCompileContext,
    is_under,
    location_dict,
    relative_to_root,
    resolved_path,
)


SOURCE_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp"}
TYPED_DIAGNOSTIC_PROVIDER_RESOLUTION_REVISION = (
    "typed_diagnostic_provider_resolution_v5"
)


def parse_error_diagnostic_count(
    diagnostics: Iterable[object],
) -> int:
    count = 0
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        severity = diagnostic.get("severity")
        try:
            is_error = int(severity or 0) >= 3
        except (TypeError, ValueError):
            is_error = str(severity or "").strip().lower() in {
                "error",
                "fatal",
            }
        if is_error:
            count += 1
    return count


def _normalized_contract_keys(values: Iterable[object]) -> set[str]:
    return {
        normalized
        for value in values
        if value is not None
        for normalized in [str(value).strip()]
        if normalized
    }


_DONOR_PARSE_CACHE_MAX_RECORDS = max(
    0,
    int(os.environ.get("DRIVER_BACKPORT_DONOR_PARSE_CACHE_RECORDS", "60000")),
)
_STRUCTURAL_PARSE_CACHE_MAX_ENTRIES = max(
    0,
    int(os.environ.get("DRIVER_BACKPORT_STRUCTURAL_PARSE_CACHE_ENTRIES", "512")),
)
_DONOR_PARSE_CACHE: OrderedDict[tuple[object, ...], tuple[dict[str, Any], int]] = (
    OrderedDict()
)
_DONOR_PARSE_CACHE_RECORDS = 0
_STRUCTURAL_PARSE_CACHE: OrderedDict[
    tuple[str, str, str, str],
    dict[str, Any],
] = OrderedDict()


def _active_semantic_graph_batch(
    state: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    active_batch_id = str(state.get("active_batch_id") or "")
    active_batch = (
        (state.get("batches") or {}).get(active_batch_id)
        if active_batch_id
        and isinstance(state.get("batches") or {}, dict)
        else None
    )
    return (
        active_batch_id,
        active_batch if isinstance(active_batch, dict) else None,
    )


def semantic_graph_gap_ids(state: dict[str, Any]) -> set[str]:
    """Return the exact semantic gaps relevant to the current graph."""

    gaps = state.get("gaps") or {}
    all_gap_ids = (
        {str(gap_id) for gap_id in gaps}
        if isinstance(gaps, dict)
        else set()
    )
    _, active_batch = _active_semantic_graph_batch(state)
    if not isinstance(active_batch, dict):
        return all_gap_ids

    active_gap_ids: set[str] = set()
    for record in [active_batch, *(active_batch.get("patches") or [])]:
        if not isinstance(record, dict):
            continue
        for field in ("primary_gap_ids", "affected_gap_ids", "gap_ids"):
            active_gap_ids.update(
                str(gap_id)
                for gap_id in record.get(field) or []
                if str(gap_id)
            )
    return active_gap_ids


STRUCTURAL_ALIGNMENT_INSERTION_STATUSES = frozenset(
    {
        "bounded_by_common_neighbors",
        "anchored_after_common_predecessor",
        "anchored_before_common_successor",
        "anchored_after_adjacent_common_sibling",
        "anchored_before_adjacent_common_sibling",
    }
)
STRUCTURAL_ALIGNMENT_RESOLVED_STATUSES = frozenset(
    {
        "existing_contract",
        "conditional_variant_family",
        *STRUCTURAL_ALIGNMENT_INSERTION_STATUSES,
    }
)


def _symbol_diagnostic_root_node_ids(
    *,
    exact_source_contract: bool,
    exact_dependency_provider_ids: Iterable[str],
    selected_provider_ids: Iterable[str],
    consumer_owner_ids: Iterable[str],
) -> list[str]:
    """Select capability providers before diagnostic consumer locations."""

    if exact_source_contract:
        return _ordered_unique(exact_dependency_provider_ids)
    providers = _ordered_unique(selected_provider_ids)
    if providers:
        return providers
    return _ordered_unique(consumer_owner_ids)


DEFINITION_CURSOR_KINDS = {
    "FUNCTION_DECL",
    "CXX_METHOD",
    "CONSTRUCTOR",
    "DESTRUCTOR",
    "VAR_DECL",
    "STRUCT_DECL",
    "UNION_DECL",
    "ENUM_DECL",
    "TYPEDEF_DECL",
    "FIELD_DECL",
    "ENUM_CONSTANT_DECL",
    "MACRO_DEFINITION",
}
REFERENCE_CURSOR_KINDS = {
    "CALL_EXPR",
    "DECL_REF_EXPR",
    "MEMBER_REF",
    "MEMBER_REF_EXPR",
    "TYPE_REF",
    "MACRO_INSTANTIATION",
}
OWNER_CURSOR_KINDS = {
    "FUNCTION_DECL",
    "CXX_METHOD",
    "CONSTRUCTOR",
    "DESTRUCTOR",
    "VAR_DECL",
    "STRUCT_DECL",
    "UNION_DECL",
    "ENUM_DECL",
    "TYPEDEF_DECL",
    "FIELD_DECL",
    "ENUM_CONSTANT_DECL",
    "MACRO_DEFINITION",
}
INCLUDE_DIRECTIVE_RE = re.compile(
    r'^\s*#\s*include\s+(?P<delimiter>[<"])(?P<name>[^>"]+)[>"]'
)
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
C_KEYWORDS = frozenset(
    {
        "_Alignas",
        "_Alignof",
        "_Atomic",
        "_Bool",
        "_Complex",
        "_Generic",
        "_Imaginary",
        "_Noreturn",
        "_Static_assert",
        "_Thread_local",
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "register",
        "restrict",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
    }
)
_C_COMMENT_OR_LITERAL_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
    re.DOTALL,
)
_UNRESOLVED_PROVIDER_NODE_KINDS = frozenset(
    {"UnresolvedContract", "UnresolvedEntity", "UnresolvedInclude"}
)


def _c_identifiers_outside_comments_and_literals(source: str) -> set[str]:
    """Return lexical C identifiers without comment or literal contents."""

    visible_source = _C_COMMENT_OR_LITERAL_RE.sub(" ", source)
    return {
        match.group(0)
        for match in IDENTIFIER_RE.finditer(visible_source)
        if match.group(0) not in C_KEYWORDS
        and not (
            match.start() > 0
            and visible_source[match.start() - 1].isdigit()
        )
    }


def _nested_macro_token_only_literals(
    replacement: str,
    definitions: dict[str, tuple[list[str], str]],
) -> set[str]:
    """Find literal arguments that a uniquely bound nested macro only pastes."""

    visible = _C_COMMENT_OR_LITERAL_RE.sub(
        lambda match: " " * len(match.group(0)), replacement
    )
    tokens = list(IDENTIFIER_RE.finditer(visible))
    token_only_positions: set[tuple[int, int]] = set()
    for token in tokens:
        definition = definitions.get(token.group(0))
        if definition is None:
            continue
        formals, body = definition
        after_name = token.end()
        while after_name < len(visible) and visible[after_name].isspace():
            after_name += 1
        parsed = _balanced_call_arguments(visible, after_name)
        if parsed is None:
            continue
        _, argument_spans = parsed
        body_visible = _C_COMMENT_OR_LITERAL_RE.sub(
            lambda match: " " * len(match.group(0)), body
        )
        token_only_formals: dict[str, bool] = {}
        for formal in formals:
            occurrences = [
                occurrence
                for occurrence in IDENTIFIER_RE.finditer(body_visible)
                if occurrence.group(0) == formal
            ]
            token_only_formals[formal] = all(
                body_visible[: occurrence.start()].rstrip().endswith("##")
                or body_visible[occurrence.end() :].lstrip().startswith("##")
                or body_visible[: occurrence.start()].rstrip().endswith("#")
                for occurrence in occurrences
            )
        splices = [
            index
            for index, (start, end) in enumerate(argument_spans)
            if visible[start:end].strip() == "##__VA_ARGS__"
        ]
        if len(splices) > 1:
            continue
        for index, (start, end) in enumerate(argument_spans):
            argument = visible[start:end].strip()
            if not IDENTIFIER_RE.fullmatch(argument):
                continue
            if splices and index > splices[0]:
                possible = range(index - 1, len(formals))
            else:
                possible = (index,)
            bound = [
                formals[position]
                for position in possible
                if position < len(formals)
            ]
            if not bound or any(
                not token_only_formals[formal] for formal in bound
            ):
                continue
            argument_start = start + visible[start:end].index(argument)
            token_only_positions.add(
                (argument_start, argument_start + len(argument))
            )
    positions_by_name: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for token in tokens:
        positions_by_name[token.group(0)].add(
            (token.start(), token.end())
        )
    return {
        name
        for name, positions in positions_by_name.items()
        if positions <= token_only_positions
    }


_UNMAPPED_ROOT_NODE_KINDS = frozenset(
    {"UnresolvedEntity", "UnresolvedInclude"}
)


def _identifier_at_source_position(
    root: Path,
    relative_path: str,
    line: int,
    column: int,
) -> dict[str, Any] | None:
    """Return the lexical C identifier containing one compiler display column."""

    position = _source_position_at_display_column(root, relative_path, line, column)
    if position is None:
        return None
    offset = int(position["source_column"]) - 1
    for match in IDENTIFIER_RE.finditer(str(position["source_line"])):
        if match.start() <= offset < match.end():
            if match.group(0) in C_KEYWORDS:
                return None
            return {
                "identifier": match.group(0),
                "line": line,
                "start_column": match.start() + 1,
                "end_column": match.end() + 1,
                "display_start_column": position["display_columns"][match.start()],
                "display_end_column": position["display_columns"][match.end()],
            }
    return None


def _source_position_at_display_column(
    root: Path,
    relative_path: str,
    line: int,
    column: int,
) -> dict[str, Any] | None:
    """Translate a compiler display column to a one-based source column."""

    if line <= 0 or column <= 0:
        return None
    path = (root / normalize_relpath(relative_path)).resolve()
    if not path.is_file() or not is_under(path, root.resolve()):
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    if line > len(lines):
        return None
    source_line = lines[line - 1]

    display_columns = [1]
    display_column = 1
    for character in source_line:
        if character == "\t":
            display_column = ((display_column - 1) // 8 + 1) * 8 + 1
        else:
            display_column += 1
        display_columns.append(display_column)
    for offset, character in enumerate(source_line):
        if display_columns[offset] <= column < display_columns[offset + 1]:
            return {
                "line": line,
                "source_column": offset + 1,
                "display_column": column,
                "character": character,
                "source_line": source_line,
                "display_columns": display_columns,
            }
    return None


def _operator_reference_records(
    references: Iterable[dict[str, Any]],
    *,
    line: int,
    source_line: str,
    source_column: int,
) -> list[dict[str, Any]]:
    """Resolve a diagnostic on a C operator to the operator's semantic operand."""

    offset = source_column - 1
    if offset < 0 or offset >= len(source_line):
        return []
    punctuator = next(
        (
            candidate
            for candidate in (
                "<<=", ">>=", "->", "++", "--", "&&", "||", "==", "!=", "<=", ">=",
                "<<", ">>", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
                ".", "[", "(", "+", "-", "*", "/", "%", "<", ">", "&", "^", "|", "=",
            )
            if source_line.startswith(candidate, offset)
        ),
        source_line[offset],
    )
    line_references = [
        reference
        for reference in references
        if int(reference.get("line") or 0) == line
        and reference.get("referenced_entity_id")
    ]
    if punctuator in {"->", "."}:
        candidates = [
            reference
            for reference in line_references
            if int(reference.get("column") or 0) > source_column
            and CodeIndex._contains_position(
                reference.get("extent") or {},
                line,
                source_column,
            )
        ]
        if not candidates:
            return []
        nearest_column = min(int(reference.get("column") or 0) for reference in candidates)
        return [
            reference
            for reference in candidates
            if int(reference.get("column") or 0) == nearest_column
        ]
    if punctuator in {"[", "("}:
        candidates = [
            reference
            for reference in line_references
            if int(reference.get("column") or 0) < source_column
            and int((reference.get("extent") or {}).get("end_line") or 0) == line
            and int((reference.get("extent") or {}).get("end_column") or 0) == source_column
        ]
        if not candidates:
            return []
        latest_column = max(int(reference.get("column") or 0) for reference in candidates)
        return [
            reference
            for reference in candidates
            if int(reference.get("column") or 0) == latest_column
        ]
    if punctuator in {
        "+", "-", "*", "/", "%", "<", ">", "&", "^", "|", "=",
        "<<", ">>", "&&", "||", "==", "!=", "<=", ">=",
        "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=", "<<=", ">>=",
    }:
        left = [
            reference
            for reference in line_references
            if int(reference.get("column") or 0) < source_column
            and int((reference.get("extent") or {}).get("end_line") or 0) == line
            and int((reference.get("extent") or {}).get("end_column") or 0) <= source_column
        ]
        right_start = source_column + len(punctuator)
        right = [
            reference
            for reference in line_references
            if int(reference.get("column") or 0) >= right_start
        ]
        selected: list[dict[str, Any]] = []
        if left:
            nearest_end = max(
                int((reference.get("extent") or {}).get("end_column") or 0)
                for reference in left
            )
            nearest_left = [
                reference
                for reference in left
                if int((reference.get("extent") or {}).get("end_column") or 0) == nearest_end
            ]
            latest_column = max(int(reference.get("column") or 0) for reference in nearest_left)
            selected.extend(
                reference
                for reference in nearest_left
                if int(reference.get("column") or 0) == latest_column
            )
        if right:
            nearest_column = min(int(reference.get("column") or 0) for reference in right)
            selected.extend(
                reference
                for reference in right
                if int(reference.get("column") or 0) == nearest_column
            )
        return list({str(reference.get("id") or id(reference)): reference for reference in selected}.values())
    return []


def _exact_reference_records_at_position(
    root: Path,
    relative_path: str,
    line: int,
    display_column: int,
    references: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Map a compiler position to AST references without same-line fallback."""

    records = list(references)
    token = _identifier_at_source_position(
        root,
        relative_path,
        line,
        display_column,
    )
    position = _source_position_at_display_column(
        root,
        relative_path,
        line,
        display_column,
    )
    if token is not None:
        selected = [
            reference
            for reference in records
            if str(reference.get("symbol") or "") == str(token["identifier"])
            and int(reference.get("line") or 0) == line
            and int(reference.get("column") or 0) == int(token["start_column"])
            and CodeIndex._contains_position(
                reference.get("extent") or {},
                line,
                int(token["start_column"]),
            )
        ]
        # Enclosing-reference recovery exists for diagnostics positioned on a
        # resolved base of an indirect member call (for example, ``chip`` in
        # ``chip->ops->callback(...)``).  An exact unresolved identifier is
        # already the most specific failing contract; replacing it with an
        # enclosing resolved call would attribute a missing nested callee to
        # its outer expression.
        selected_has_concrete_targets = bool(selected) and all(
            isinstance(reference.get("referenced_entity"), dict)
            and _entity_has_source_extent(
                reference.get("referenced_entity") or {}
            )
            for reference in selected
        )
        enclosing = (
            _outermost_enclosing_references(records, selected)
            if selected_has_concrete_targets
            else []
        )
        if enclosing:
            return {
                "mapping_kind": "enclosing_typed_reference_expression",
                "references": enclosing,
                "identifier": token["identifier"],
                "source_column": int(token["start_column"]),
                "position": position,
            }
        return {
            "mapping_kind": "lexical_identifier",
            "references": selected,
            "identifier": token["identifier"],
            "source_column": int(token["start_column"]),
            "position": position,
        }
    if position is not None:
        return {
            "mapping_kind": "operator_operand",
            "references": _operator_reference_records(
                records,
                line=line,
                source_line=str(position["source_line"]),
                source_column=int(position["source_column"]),
            ),
            "identifier": None,
            "source_column": int(position["source_column"]),
            "position": position,
        }

    # Parser-backed tests and persisted graph pages may not retain source text.
    # An exact parser column is still structural evidence; no other same-line
    # references are admitted.
    return {
        "mapping_kind": "exact_parser_column",
        "references": [
            reference
            for reference in records
            if int(reference.get("line") or 0) == line
            and int(reference.get("column") or 0) == display_column
        ],
        "identifier": None,
        "source_column": display_column,
        "position": None,
    }


def _extent_contains_extent(
    outer: dict[str, Any],
    inner: dict[str, Any],
) -> bool:
    outer_start = (
        int(outer.get("start_line") or 0),
        int(outer.get("start_column") or 0),
    )
    outer_end = (
        int(outer.get("end_line") or 0),
        int(outer.get("end_column") or 0),
    )
    inner_start = (
        int(inner.get("start_line") or 0),
        int(inner.get("start_column") or 0),
    )
    inner_end = (
        int(inner.get("end_line") or 0),
        int(inner.get("end_column") or 0),
    )
    if not all((*outer_start, *outer_end, *inner_start, *inner_end)):
        return False
    return outer_start <= inner_start and outer_end >= inner_end


def _outermost_enclosing_references(
    records: Iterable[dict[str, Any]],
    selected: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_records = list(selected)
    if not selected_records:
        return []
    selected_ids = {str(record.get("id") or "") for record in selected_records}
    enclosing = [
        record
        for record in records
        if str(record.get("id") or "") not in selected_ids
        and any(
            _extent_contains_extent(
                dict(record.get("extent") or {}),
                dict(selected_record.get("extent") or {}),
            )
            for selected_record in selected_records
        )
        and isinstance(record.get("referenced_entity"), dict)
        and _entity_has_source_extent(record.get("referenced_entity") or {})
    ]
    return [
        record
        for record in enclosing
        if not any(
            str(other.get("id") or "") != str(record.get("id") or "")
            and _extent_contains_extent(
                dict(other.get("extent") or {}),
                dict(record.get("extent") or {}),
            )
            and dict(other.get("extent") or {}) != dict(record.get("extent") or {})
            for other in enclosing
        )
    ]


_ENTITY_GRAPH_FIELDS = (
    "tree",
    "key",
    "contract_key",
    "symbol",
    "name",
    "cursor_kind",
    "displayname",
    "usr",
    "container",
    "is_definition",
    "location",
    "file",
    "extent",
    "digest",
    "token_count",
    "signature_fingerprint",
    "implementation_fingerprint",
    "interface_fingerprint",
    "interface",
    "storage_class",
    "linkage",
    "conditional_context",
    "conditional_depth",
    "is_source_anchor",
    "source_fingerprint",
    "syntax_fingerprint",
)

_REFERENCE_GRAPH_FIELDS = (
    "reference_id",
    "tree",
    "source_file",
    "symbol",
    "symbol_kind",
    "use_kind",
    "cursor_kind",
    "line",
    "column",
    "extent",
    "owner_entity_id",
    "lexical_owner_entity_ids",
    "referenced_entity_id",
    "referenced_file",
    "referenced_kind",
    "referenced_location",
    "dependency_scope",
    "container",
    "contract_key",
    "source_spelling_evidence",
    "ast_path",
    "owner_resolution",
    "reference_context_fingerprint",
    "reference_context_depth",
    "reference_use_contract",
    "reference_use_fingerprint",
)

_INCLUDE_GRAPH_FIELDS = (
    "tree",
    "source_file",
    "name",
    "delimiter",
    "line",
    "target_file",
    "resolved",
)


def _compact_graph_fields(
    record: dict[str, Any],
    fields: Iterable[str],
) -> dict[str, Any]:
    return {
        key: record[key]
        for key in fields
        if key in record and record[key] not in (None, "", [])
    }


def _entity_graph_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Project a full AST entity onto the stable dependency-graph contract."""

    payload = _compact_graph_fields(record, _ENTITY_GRAPH_FIELDS)
    if record.get("kind"):
        payload["entity_kind"] = record["kind"]
    return payload


def _reference_graph_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Keep endpoint IDs on graph references; endpoint records live on entity nodes."""

    payload = _compact_graph_fields(record, _REFERENCE_GRAPH_FIELDS)
    for entity_id_key, entity_key in (
        ("owner_entity_id", "owner_entity"),
        ("referenced_entity_id", "referenced_entity"),
    ):
        entity = record.get(entity_key)
        if not payload.get(entity_id_key) and isinstance(entity, dict) and entity.get("id"):
            payload[entity_id_key] = entity["id"]
    if record.get("kind"):
        payload["entity_kind"] = record["kind"]
    return payload


def _include_graph_payload(record: dict[str, Any]) -> dict[str, Any]:
    return _compact_graph_fields(record, _INCLUDE_GRAPH_FIELDS)


def _entity_has_source_extent(record: dict[str, Any]) -> bool:
    extent = record.get("extent") if isinstance(record.get("extent"), dict) else {}
    return bool(
        str(record.get("file") or "").strip()
        and int(extent.get("start_line") or 0) > 0
        and int(extent.get("end_line") or 0) > 0
    )


def entity_record_rank(
    record: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    """Rank alternate AST records for one stable semantic entity identity."""

    extent = record.get("extent") if isinstance(record.get("extent"), dict) else {}
    start_line = int(extent.get("start_line") or 0)
    start_column = int(extent.get("start_column") or 0)
    end_line = int(extent.get("end_line") or 0)
    end_column = int(extent.get("end_column") or 0)
    concrete = bool(str(record.get("file") or "").strip() and start_line > 0 and end_line > 0)
    line_span = max(0, end_line - start_line) if concrete else 0
    column_span = max(0, end_column - start_column) if concrete and line_span == 0 else 0
    semantic_detail = bool(
        record.get("digest")
        or record.get("signature_fingerprint")
        or record.get("implementation_fingerprint")
        or record.get("interface_fingerprint")
    )
    return (
        int(bool(record.get("is_definition"))),
        int(concrete),
        line_span,
        column_span,
        int(semantic_detail),
    )


def canonical_entity_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest parser record for each concrete source entity."""

    selected: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in records:
        record = dict(raw)
        record_id = str(record.get("id") or record.get("entity_id") or "")
        if not record_id:
            continue
        extent = (
            record.get("extent")
            if isinstance(record.get("extent"), dict)
            else {}
        )
        source_identity = (
            str(record.get("tree") or ""),
            normalize_relpath(str(record.get("file") or "")),
            str(record.get("contract_key") or record.get("key") or ""),
            str(record.get("is_definition")),
            int(extent.get("start_line") or 0),
            int(extent.get("start_column") or 0),
            int(extent.get("end_line") or 0),
            int(extent.get("end_column") or 0),
        )
        has_concrete_source_identity = bool(
            source_identity[0]
            and source_identity[1]
            and source_identity[2]
            and source_identity[4] > 0
            and source_identity[6] > 0
        )
        entity_id = (
            repr(source_identity)
            if has_concrete_source_identity
            else record_id
        )
        current = selected.get(entity_id)
        if current is None:
            selected[entity_id] = record
            order.append(entity_id)
        elif entity_record_rank(record) > entity_record_rank(current):
            selected[entity_id] = record
    return [selected[entity_id] for entity_id in order]


def build_conditioned_provider_records(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Exclude conditional structural definitions contradicted by active AST linkage."""

    selected = list(records)
    external_declarations = {
        (
            str(record.get("tree") or ""),
            normalize_relpath(str(record.get("file") or "")),
            str(record.get("contract_key") or record.get("key") or ""),
        )
        for record in selected
        if not record.get("is_definition")
        and str(record.get("cursor_kind") or "") in {"FUNCTION_DECL", "VAR_DECL"}
        and str((record.get("interface") or {}).get("linkage") or "").upper()
        == "EXTERNAL"
    }
    return [
        record
        for record in selected
        if not (
            record.get("is_definition")
            and str(record.get("cursor_kind") or "").startswith("TREE_SITTER_")
            and bool(record.get("conditional_context"))
            and str(record.get("linkage") or "").upper() == "INTERNAL"
            and (
                str(record.get("tree") or ""),
                normalize_relpath(str(record.get("file") or "")),
                str(record.get("contract_key") or record.get("key") or ""),
            )
            in external_declarations
        )
    ]


def _compact_alignment_entity(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    extent = record.get("extent") if isinstance(record.get("extent"), dict) else {}
    return {
        "id": record.get("id") or record.get("entity_id"),
        "tree": record.get("tree"),
        "file": normalize_relpath(str(record.get("file") or "")),
        "kind": record.get("kind") or record.get("entity_kind"),
        "symbol": record.get("symbol") or record.get("name"),
        "contract_key": record.get("contract_key") or record.get("key"),
        "conditional_depth": max(0, int(record.get("conditional_depth") or 0)),
        "conditional_context": list(record.get("conditional_context") or []),
        "source_fingerprint": record.get("source_fingerprint"),
        "extent": {
            "start_line": int(extent.get("start_line") or 0),
            "start_column": int(extent.get("start_column") or 0),
            "end_line": int(extent.get("end_line") or 0),
            "end_column": int(extent.get("end_column") or 0),
        },
    }


def _entity_container_identity(
    entity: dict[str, Any],
) -> tuple[str, str]:
    container = (
        entity.get("container")
        if isinstance(entity.get("container"), dict)
        else {}
    )
    return (
        str(container.get("container_kind") or ""),
        str(container.get("container_name") or ""),
    )


def structural_entity_neighbor_alignment(
    *,
    target_entity_id: str,
    donor_entities: Iterable[dict[str, Any]],
    old_entities: Iterable[dict[str, Any]],
    include_declarations: bool = False,
) -> dict[str, Any]:
    """Align a donor-only entity to old source through ordered common contracts.

    The result contains exact semantic predecessor/successor anchors.  It does not
    infer an insertion point from names, diagnostics, or fixed line windows.
    """

    def ordered(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            [
                record
                for record in canonical_entity_records(records)
                if (
                    include_declarations
                    or
                    record.get("is_definition") is not False
                    or record.get("is_source_anchor") is True
                )
            ],
            key=lambda item: (
                int((item.get("extent") or {}).get("start_line") or 0),
                int((item.get("extent") or {}).get("start_column") or 0),
                int((item.get("extent") or {}).get("end_line") or 0),
                str(item.get("id") or ""),
            ),
        )

    donor = ordered(donor_entities)
    old = ordered(old_entities)
    target_indexes = [
        index
        for index, entity in enumerate(donor)
        if str(entity.get("id") or entity.get("entity_id") or "") == target_entity_id
    ]
    result: dict[str, Any] = {
        "schema": "structural_entity_neighbor_alignment_v1",
        "status": "unavailable",
        "target": None,
        "old_counterpart": None,
        "predecessor": None,
        "successor": None,
        "donor_between": [],
        "old_between": [],
    }
    if len(target_indexes) != 1:
        result["reason"] = "target entity is not uniquely present in the donor owner sequence"
        return result

    target_index = target_indexes[0]
    target = donor[target_index]
    # File scope is a real lexical scope too.  Without this filter, a macro or
    # field nested in the preceding declaration can become the predecessor of
    # a donor-only top-level entity, authorizing an insertion inside the wrong
    # declaration.
    target_container_identity = _entity_container_identity(target)
    result["container_identity"] = list(target_container_identity)
    donor = [
        entity
        for entity in donor
        if _entity_container_identity(entity) == target_container_identity
    ]
    old = [
        entity
        for entity in old
        if _entity_container_identity(entity) == target_container_identity
    ]
    target_indexes = [
        index
        for index, entity in enumerate(donor)
        if str(entity.get("id") or entity.get("entity_id") or "")
        == target_entity_id
    ]
    if len(target_indexes) != 1:
        result["reason"] = (
            "target entity is not unique within its declaring container"
        )
        return result
    target_index = target_indexes[0]
    target = donor[target_index]
    target_context = tuple(str(item) for item in target.get("conditional_context") or [])

    def same_conditional_context(entity: dict[str, Any]) -> bool:
        return tuple(
            str(item) for item in entity.get("conditional_context") or []
        ) == target_context

    donor = [entity for entity in donor if same_conditional_context(entity)]
    old = [entity for entity in old if same_conditional_context(entity)]
    target_indexes = [
        index
        for index, entity in enumerate(donor)
        if str(entity.get("id") or entity.get("entity_id") or "") == target_entity_id
    ]
    if len(target_indexes) != 1:
        result["reason"] = "target entity is not unique within its conditional branch"
        return result
    target_index = target_indexes[0]
    target = donor[target_index]
    result["conditional_context"] = list(target_context)
    result["conditional_context_aligned"] = True
    result["target"] = _compact_alignment_entity(target)

    def identity(entity: dict[str, Any]) -> tuple[str, str]:
        return (
            str(entity.get("kind") or entity.get("entity_kind") or ""),
            str(entity.get("contract_key") or entity.get("key") or ""),
        )

    old_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entity in old:
        key = identity(entity)
        if all(key):
            old_by_identity[key].append(entity)
    donor_identity_counts = Counter(
        identity(entity)
        for entity in donor
        if all(identity(entity))
    )

    target_matches = old_by_identity.get(identity(target), [])
    if len(target_matches) == 1:
        result.update(
            {
                "status": "existing_contract",
                "old_counterpart": _compact_alignment_entity(target_matches[0]),
                "reason": "the donor root has one exact structural contract counterpart in old",
            }
        )
        return result

    def common_pair(entity: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
        key = identity(entity)
        if donor_identity_counts.get(key) != 1:
            return None
        matches = old_by_identity.get(identity(entity), [])
        if len(matches) != 1:
            return None
        return entity, matches[0]

    target_extent = target.get("extent") or {}
    target_start = int(target_extent.get("start_line") or 0)
    target_end = int(target_extent.get("end_line") or target_start)
    target_kind = identity(target)[0]

    def adjacent_common(
        candidate_index: int,
        *,
        before: bool,
    ) -> tuple[int, int, tuple[dict[str, Any], dict[str, Any]]] | None:
        if candidate_index < 0 or candidate_index >= len(donor):
            return None
        candidate = donor[candidate_index]
        if identity(candidate)[0] != target_kind:
            return None
        candidate_extent = candidate.get("extent") or {}
        candidate_start = int(candidate_extent.get("start_line") or 0)
        candidate_end = int(
            candidate_extent.get("end_line") or candidate_start
        )
        source_adjacent = (
            candidate_end + 1 == target_start
            if before
            else target_end + 1 == candidate_start
        )
        if not source_adjacent:
            return None
        pair = common_pair(candidate)
        if pair is None:
            return None
        return candidate_index, old.index(pair[1]), pair

    adjacent_predecessor = adjacent_common(target_index - 1, before=True)
    adjacent_successor = adjacent_common(target_index + 1, before=False)
    adjacent_order_consistent = (
        adjacent_predecessor is None
        or adjacent_successor is None
        or adjacent_predecessor[1] < adjacent_successor[1]
    )
    if adjacent_order_consistent and (
        adjacent_predecessor or adjacent_successor
    ):
        if adjacent_predecessor:
            predecessor = adjacent_predecessor[2]
            result["predecessor"] = {
                "donor": _compact_alignment_entity(predecessor[0]),
                "old": _compact_alignment_entity(predecessor[1]),
            }
        if adjacent_successor:
            successor = adjacent_successor[2]
            result["successor"] = {
                "donor": _compact_alignment_entity(successor[0]),
                "old": _compact_alignment_entity(successor[1]),
            }

        old_anchors_are_adjacent = bool(
            adjacent_predecessor
            and adjacent_successor
            and adjacent_successor[1] == adjacent_predecessor[1] + 1
        )
        if adjacent_predecessor and adjacent_successor:
            result["old_between"] = [
                _compact_alignment_entity(entity)
                for entity in old[
                    adjacent_predecessor[1] + 1 : adjacent_successor[1]
                ]
            ]
        if old_anchors_are_adjacent:
            result["status"] = "bounded_by_common_neighbors"
            result["reason"] = (
                "target lies between adjacent same-kind structural contracts"
            )
            result["anchor_selection"] = (
                "adjacent_same_kind_common_contract_pair"
            )
        elif adjacent_predecessor:
            result["status"] = "anchored_after_adjacent_common_sibling"
            result["reason"] = (
                "target immediately follows a unique same-kind structural "
                "contract in the donor sequence"
            )
            result["anchor_selection"] = (
                "adjacent_same_kind_common_predecessor"
            )
        else:
            result["status"] = "anchored_before_adjacent_common_sibling"
            result["reason"] = (
                "target immediately precedes a unique same-kind structural "
                "contract in the donor sequence"
            )
            result["anchor_selection"] = (
                "adjacent_same_kind_common_successor"
            )
        result["adjacent_sibling_alignment"] = True
        result["donor_between"] = [_compact_alignment_entity(target)]
        return result

    predecessors = [
        (index, old.index(pair[1]), pair)
        for index in range(target_index - 1, -1, -1)
        if (pair := common_pair(donor[index])) is not None
    ]
    successors = [
        (index, old.index(pair[1]), pair)
        for index in range(target_index + 1, len(donor))
        if (pair := common_pair(donor[index])) is not None
    ]
    ordered_pairs = [
        (predecessor, successor)
        for predecessor in predecessors
        for successor in successors
        if predecessor[1] < successor[1]
    ]
    selected_pair = min(
        ordered_pairs,
        key=lambda pair: (
            (target_index - pair[0][0]) + (pair[1][0] - target_index),
            pair[1][1] - pair[0][1],
            -pair[0][0],
            pair[1][0],
        ),
        default=None,
    )
    if selected_pair is not None:
        predecessor_record, successor_record = selected_pair
        predecessor = predecessor_record[2]
        successor = successor_record[2]
        predecessor_donor_index = predecessor_record[0]
        successor_donor_index = successor_record[0]
        predecessor_old_index = predecessor_record[1]
        successor_old_index = successor_record[1]
        result["donor_between"] = [
            _compact_alignment_entity(entity)
            for entity in donor[predecessor_donor_index + 1 : successor_donor_index]
        ]
        result["old_between"] = [
            _compact_alignment_entity(entity)
            for entity in old[predecessor_old_index + 1 : successor_old_index]
        ]
        result["anchor_selection"] = "nearest_order_consistent_common_contract_pair"
    elif predecessors and successors:
        result["status"] = "non_monotonic_common_neighbors"
        result["reason"] = (
            "no common predecessor/successor pair preserves donor order in old source"
        )
        return result
    else:
        predecessor = predecessors[0][2] if predecessors else None
        successor = successors[0][2] if successors else None
        if predecessor or successor:
            result["anchor_selection"] = "nearest_unique_one_sided_common_contract"
        if predecessor:
            predecessor_record = predecessors[0]
            result["donor_between"] = [
                _compact_alignment_entity(entity)
                for entity in donor[predecessor_record[0] + 1 :]
            ]
            result["old_between"] = [
                _compact_alignment_entity(entity)
                for entity in old[predecessor_record[1] + 1 :]
            ]
        elif successor:
            successor_record = successors[0]
            result["donor_between"] = [
                _compact_alignment_entity(entity)
                for entity in donor[: successor_record[0]]
            ]
            result["old_between"] = [
                _compact_alignment_entity(entity)
                for entity in old[: successor_record[1]]
            ]

    if predecessor:
        result["predecessor"] = {
            "donor": _compact_alignment_entity(predecessor[0]),
            "old": _compact_alignment_entity(predecessor[1]),
        }
    if successor:
        result["successor"] = {
            "donor": _compact_alignment_entity(successor[0]),
            "old": _compact_alignment_entity(successor[1]),
        }
    if predecessor and successor:
        result["status"] = "bounded_by_common_neighbors"
        result["reason"] = "target lies between two order-preserving structural contracts"
    elif predecessor:
        result["status"] = "anchored_after_common_predecessor"
        result["reason"] = "target follows the last available common structural contract"
    elif successor:
        result["status"] = "anchored_before_common_successor"
        result["reason"] = "target precedes the first available common structural contract"
    else:
        result["reason"] = "owner files have no unique common structural contract around the target"
    return result


def structural_entity_bundle_alignment(
    *,
    target_entity_ids: Iterable[str],
    donor_entities: Iterable[dict[str, Any]],
    old_entities: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Collapse one exact donor dependency run into an alignable transaction.

    A bundle is admitted only when every selected definition has the same
    owner container and conditional branch, and no unselected sibling
    definition intersects the source envelope. The result therefore describes
    one source interval, not a set of unrelated edits from the owner file.
    """

    requested_ids = {
        str(entity_id)
        for entity_id in target_entity_ids
        if str(entity_id)
    }
    result: dict[str, Any] = {
        "schema": "structural_entity_bundle_alignment_v1",
        "status": "unavailable",
        "target_entity_ids": sorted(requested_ids),
        "bundle": None,
    }
    if not requested_ids:
        result["reason"] = "bundle_has_no_target_entities"
        return result

    donor = canonical_entity_records(donor_entities)
    selected = [
        entity
        for entity in donor
        if str(entity.get("id") or entity.get("entity_id") or "")
        in requested_ids
    ]
    selected_ids = {
        str(entity.get("id") or entity.get("entity_id") or "")
        for entity in selected
    }
    if selected_ids != requested_ids:
        result["reason"] = "bundle_target_is_absent_or_ambiguous"
        return result

    contexts = {
        tuple(str(item) for item in entity.get("conditional_context") or [])
        for entity in selected
    }
    containers = {_entity_container_identity(entity) for entity in selected}
    files = {
        normalize_relpath(str(entity.get("file") or ""))
        for entity in selected
    }
    if len(contexts) != 1 or len(containers) != 1 or len(files) != 1:
        result["reason"] = (
            "bundle_crosses_owner_container_or_conditional_branch"
        )
        return result

    selected = sorted(
        selected,
        key=lambda entity: (
            int((entity.get("extent") or {}).get("start_line") or 0),
            int((entity.get("extent") or {}).get("start_column") or 0),
            int((entity.get("extent") or {}).get("end_line") or 0),
        ),
    )
    extents = [
        (
            int((entity.get("extent") or {}).get("start_line") or 0),
            int((entity.get("extent") or {}).get("end_line") or 0),
        )
        for entity in selected
    ]
    if any(start <= 0 or end < start for start, end in extents):
        result["reason"] = "bundle_has_invalid_source_extent"
        return result
    if any(
        current_start <= previous_end
        for (_previous_start, previous_end), (current_start, _current_end)
        in zip(extents, extents[1:])
    ):
        result["reason"] = "bundle_target_extents_overlap"
        return result

    envelope_start = extents[0][0]
    envelope_end = extents[-1][1]
    context = next(iter(contexts))
    container = next(iter(containers))
    peers = [
        entity
        for entity in donor
        if str(entity.get("id") or entity.get("entity_id") or "")
        not in requested_ids
        and entity.get("is_definition") is True
        and _entity_container_identity(entity) == container
        and tuple(
            str(item) for item in entity.get("conditional_context") or []
        )
        == context
        and int((entity.get("extent") or {}).get("start_line") or 0)
        <= envelope_end
        and int((entity.get("extent") or {}).get("end_line") or 0)
        >= envelope_start
    ]
    if peers:
        result["reason"] = (
            "bundle_envelope_contains_unselected_sibling_definition"
        )
        result["intersecting_sibling_entity_ids"] = sorted(
            str(entity.get("id") or entity.get("entity_id") or "")
            for entity in peers
        )
        return result

    first = selected[0]
    last = selected[-1]
    first_extent = first.get("extent") or {}
    last_extent = last.get("extent") or {}
    synthetic_id = stable_id(
        "structural-materialization-bundle",
        normalize_relpath(str(first.get("file") or "")),
        *[
            str(entity.get("id") or entity.get("entity_id") or "")
            for entity in selected
        ],
    )
    synthetic = {
        **dict(first),
        "id": synthetic_id,
        "entity_id": synthetic_id,
        "key": f"bundle:{synthetic_id}",
        "contract_key": f"bundle:{synthetic_id}",
        "symbol": "<dependency-closure>",
        "name": "<dependency-closure>",
        "source_fingerprint": stable_id(
            "structural-materialization-bundle-source",
            *[
                str(entity.get("source_fingerprint") or "")
                for entity in selected
            ],
        ),
        "extent": {
            "start_line": envelope_start,
            "start_column": int(first_extent.get("start_column") or 0),
            "end_line": envelope_end,
            "end_column": int(last_extent.get("end_column") or 0),
        },
        "starts_at_line_start": bool(
            first.get("starts_at_line_start")
            or int(first_extent.get("start_column") or 0) == 1
        ),
        "ends_at_line_boundary": bool(
            last.get("ends_at_line_boundary")
        ),
    }
    collapsed_donor = [
        entity
        for entity in donor
        if str(entity.get("id") or entity.get("entity_id") or "")
        not in requested_ids
    ]
    collapsed_donor.append(synthetic)
    alignment = structural_entity_neighbor_alignment(
        target_entity_id=synthetic_id,
        donor_entities=collapsed_donor,
        old_entities=old_entities,
    )
    alignment["schema"] = "structural_entity_bundle_alignment_v1"
    alignment["target_entity_ids"] = [
        str(entity.get("id") or entity.get("entity_id") or "")
        for entity in selected
    ]
    alignment["bundle"] = {
        "synthetic_entity_id": synthetic_id,
        "contract_keys": [
            str(entity.get("contract_key") or entity.get("key") or "")
            for entity in selected
        ],
        "entities": [
            _compact_alignment_entity(entity) for entity in selected
        ],
        "extent": dict(synthetic["extent"]),
        "continuous_structural_run": True,
        "intersecting_unselected_siblings": [],
    }
    return alignment


def structural_entity_bundle_alignments(
    *,
    target_entity_ids: Iterable[str],
    donor_entities: Iterable[dict[str, Any]],
    old_entities: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Partition a dependency closure into exact structural source runs."""

    requested_ids = {
        str(entity_id)
        for entity_id in target_entity_ids
        if str(entity_id)
    }
    result: dict[str, Any] = {
        "schema": "structural_entity_bundle_alignments_v1",
        "status": "unavailable",
        "target_entity_ids": sorted(requested_ids),
        "alignments": [],
    }
    donor = [
        entity
        for entity in canonical_entity_records(donor_entities)
        if entity.get("is_definition") is True
    ]
    donor.sort(
        key=lambda entity: (
            normalize_relpath(str(entity.get("file") or "")),
            int((entity.get("extent") or {}).get("start_line") or 0),
            int((entity.get("extent") or {}).get("start_column") or 0),
            str(entity.get("id") or entity.get("entity_id") or ""),
        )
    )
    selected_ids = {
        str(entity.get("id") or entity.get("entity_id") or "")
        for entity in donor
        if str(entity.get("id") or entity.get("entity_id") or "")
        in requested_ids
    }
    if selected_ids != requested_ids:
        result["reason"] = "bundle_target_is_absent_or_not_a_definition"
        result["missing_target_entity_ids"] = sorted(
            requested_ids - selected_ids
        )
        return result

    def owner_identity(
        entity: dict[str, Any],
    ) -> tuple[str, str, str, tuple[str, ...]]:
        container = (
            entity.get("container")
            if isinstance(entity.get("container"), dict)
            else {}
        )
        return (
            normalize_relpath(str(entity.get("file") or "")),
            str(container.get("container_kind") or ""),
            str(container.get("container_name") or ""),
            tuple(
                str(item)
                for item in entity.get("conditional_context") or []
            ),
        )

    runs: list[list[str]] = []
    current_run: list[str] = []
    current_owner: tuple[str, str, str, tuple[str, ...]] | None = None
    for entity in donor:
        entity_id = str(
            entity.get("id") or entity.get("entity_id") or ""
        )
        owner = owner_identity(entity)
        if entity_id not in requested_ids:
            if current_run:
                runs.append(current_run)
                current_run = []
                current_owner = None
            continue
        if current_run and owner != current_owner:
            runs.append(current_run)
            current_run = []
        current_run.append(entity_id)
        current_owner = owner
    if current_run:
        runs.append(current_run)

    alignments: list[dict[str, Any]] = []
    for run in runs:
        alignment = structural_entity_bundle_alignment(
            target_entity_ids=run,
            donor_entities=donor_entities,
            old_entities=old_entities,
        )
        alignments.append(alignment)
        if (
            str(alignment.get("status") or "")
            not in STRUCTURAL_ALIGNMENT_INSERTION_STATUSES
            or not alignment.get("conditional_context_aligned")
        ):
            result["reason"] = (
                "dependency_run_has_no_atomic_source_alignment"
            )
            result["alignments"] = alignments
            return result

    result.update(
        {
            "status": "aligned",
            "reason": "all_dependency_runs_have_structural_anchors",
            "alignments": alignments,
            "bundle_count": len(alignments),
        }
    )
    return result


def structural_error_overlaps_extent(
    report: dict[str, Any],
    *,
    start_line: int,
    end_line: int | None = None,
) -> bool:
    """Return whether a structural parser error intersects a source extent."""

    end_line = max(start_line, int(end_line or start_line))
    return any(
        int((node.get("extent") or {}).get("start_line") or 0) <= end_line
        and int((node.get("extent") or {}).get("end_line") or 0) >= start_line
        for node in report.get("error_nodes") or []
        if isinstance(node, dict)
        and int((node.get("extent") or {}).get("start_line") or 0) > 0
        and int((node.get("extent") or {}).get("end_line") or 0) > 0
    )


def structural_error_crosses_insertion_boundary(
    report: dict[str, Any],
    *,
    insertion_index: int,
) -> bool:
    """Return whether an error spans both sides of a full-line insertion.

    ``insertion_index`` is the number of existing source lines before the
    insertion. An error wholly inside the predecessor or successor is not
    evidence that inserting between them is unsafe.
    """

    if insertion_index < 0:
        return True
    return any(
        int((node.get("extent") or {}).get("start_line") or 0)
        <= insertion_index
        and int((node.get("extent") or {}).get("end_line") or 0)
        >= insertion_index + 1
        for node in report.get("error_nodes") or []
        if isinstance(node, dict)
        and int((node.get("extent") or {}).get("start_line") or 0) > 0
        and int((node.get("extent") or {}).get("end_line") or 0) > 0
    )


def _comment_only_c_source(source: str) -> bool:
    if not source.strip():
        return False
    try:
        from tree_sitter_languages import get_parser

        data = source.encode("utf-8")
        root = get_parser("c").parse(data).root_node
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False
    if root.has_error or not root.children:
        return False
    offset = 0
    for child in root.children:
        if child.type != "comment" or data[offset : child.start_byte].strip():
            return False
        offset = child.end_byte
    return not data[offset:].strip()


def structural_entity_insertion_plan(
    *,
    alignment: dict[str, Any],
    donor_lines: list[str],
    old_lines: list[str],
) -> dict[str, Any] | None:
    """Prove one full-line insertion boundary from common source anchors.

    Semantic neighbor alignment identifies common entities. This second step
    aligns the exact source gap between those entities, including comments and
    blank lines. A one-sided anchor must align its source gap, or prove an
    otherwise empty top-level EOF. If both sides prove a boundary, they must
    agree.
    """

    if (
        str(alignment.get("status") or "")
        not in STRUCTURAL_ALIGNMENT_INSERTION_STATUSES
        or not alignment.get("conditional_context_aligned")
    ):
        return None
    target = alignment.get("target") or {}
    target_extent = target.get("extent") or {}
    donor_start = int(target_extent.get("start_line") or 0)
    donor_end = int(target_extent.get("end_line") or donor_start)
    if (
        donor_start <= 0
        or donor_end < donor_start
        or donor_end > len(donor_lines)
    ):
        return None

    predecessor = alignment.get("predecessor") or {}
    successor = alignment.get("successor") or {}
    predecessor_donor = predecessor.get("donor") or {}
    predecessor_old = predecessor.get("old") or {}
    successor_donor = successor.get("donor") or {}
    successor_old = successor.get("old") or {}

    old_floor = (
        int((predecessor_old.get("extent") or {}).get("end_line") or 0)
        if predecessor_old
        else 0
    )
    old_ceiling = (
        int((successor_old.get("extent") or {}).get("start_line") or 0) - 1
        if successor_old
        else len(old_lines)
    )
    if (
        old_floor < 0
        or old_ceiling < old_floor
        or old_ceiling > len(old_lines)
    ):
        return None
    old_gap = old_lines[old_floor:old_ceiling]
    candidates: list[dict[str, Any]] = []

    if predecessor_donor and predecessor_old:
        donor_predecessor_end = int(
            (predecessor_donor.get("extent") or {}).get("end_line") or 0
        )
        if 0 < donor_predecessor_end < donor_start:
            donor_before = donor_lines[
                donor_predecessor_end : donor_start - 1
            ]
            if old_gap[: len(donor_before)] == donor_before:
                candidates.append(
                    {
                        "side": "predecessor",
                        "insertion_index": old_floor + len(donor_before),
                        "matched_gap_lines": len(donor_before),
                    }
                )
            else:
                matching_prefix = 0
                for donor_line, old_line in zip(donor_before, old_gap):
                    if donor_line != old_line:
                        break
                    matching_prefix += 1
                trailing_donor = donor_before[matching_prefix:]
                if (
                    matching_prefix
                    and trailing_donor
                    and _comment_only_c_source("".join(trailing_donor))
                ):
                    candidates.append(
                        {
                            "side": "predecessor",
                            "insertion_index": old_floor + matching_prefix,
                            "matched_gap_lines": matching_prefix,
                            "basis": "leading_comment_source_gap_alignment",
                            "donor_leading_trivia_start_line": (
                                donor_predecessor_end + matching_prefix + 1
                            ),
                        }
                    )
            if not candidates and (
                donor_before
                and all(not line.strip() for line in donor_before)
                and all(not line.strip() for line in old_gap)
            ):
                candidates.append(
                    {
                        "side": "predecessor",
                        "insertion_index": old_ceiling,
                        "matched_gap_lines": len(old_gap),
                        "basis": "whitespace_source_gap_alignment",
                    }
                )
        elif donor_predecessor_end == donor_start - 1:
            candidates.append(
                {
                    "side": "predecessor",
                    "insertion_index": old_floor,
                    "matched_gap_lines": 0,
                }
            )

    if successor_donor and successor_old:
        donor_successor_start = int(
            (successor_donor.get("extent") or {}).get("start_line") or 0
        )
        if donor_successor_start > donor_end:
            donor_after = donor_lines[donor_end : donor_successor_start - 1]
            if (
                len(donor_after) <= len(old_gap)
                and old_gap[len(old_gap) - len(donor_after) :] == donor_after
            ):
                candidates.append(
                    {
                        "side": "successor",
                        "insertion_index": old_ceiling - len(donor_after),
                        "matched_gap_lines": len(donor_after),
                    }
                )
            elif (
                donor_after
                and all(not line.strip() for line in donor_after)
                and all(not line.strip() for line in old_gap)
            ):
                candidates.append(
                    {
                        "side": "successor",
                        "insertion_index": old_floor,
                        "matched_gap_lines": len(old_gap),
                        "basis": "whitespace_source_gap_alignment",
                    }
                )
        elif donor_successor_start == donor_end + 1:
            candidates.append(
                {
                    "side": "successor",
                    "insertion_index": old_ceiling,
                    "matched_gap_lines": 0,
                }
            )

    if (
        not candidates
        and predecessor_donor and predecessor_old
        and not successor_donor and not successor_old
        and alignment.get("container_identity") == ["", ""]
        and alignment.get("conditional_context") == []
        and not target.get("conditional_depth")
        and not alignment.get("old_between")
        and (not "".join(old_gap).strip()
             or _comment_only_c_source("".join(old_gap)))
        and 0 < old_floor <= len(old_lines)
        and old_lines[-1].endswith(("\n", "\r"))
    ):
        anchor_end_column = int(
            (predecessor_old.get("extent") or {}).get("end_column") or 0
        )
        anchor_line = old_lines[old_floor - 1].rstrip("\r\n")
        if (
            0 < anchor_end_column <= len(anchor_line) + 1
            and not anchor_line[anchor_end_column - 1:].strip()
        ):
            candidates.append({
                "side": "predecessor_and_eof",
                "insertion_index": len(old_lines),
                "matched_gap_lines": len(old_gap),
                "basis": "top_level_eof_boundary",
            })

    insertion_indexes = {
        int(candidate["insertion_index"]) for candidate in candidates
    }
    boundary_basis = "exact_source_gap_alignment"
    leading_comment = None
    if len(insertion_indexes) == 1:
        insertion_index = insertion_indexes.pop()
        leading_comment = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("basis") == "leading_comment_source_gap_alignment"
            ),
            None,
        )
        if leading_comment is not None:
            boundary_basis = "leading_comment_source_gap_alignment"
        elif any(
            candidate.get("basis") == "whitespace_source_gap_alignment"
            for candidate in candidates
        ):
            boundary_basis = "whitespace_source_gap_alignment"
        elif any(
            candidate.get("basis") == "top_level_eof_boundary"
            for candidate in candidates
        ):
            boundary_basis = "top_level_eof_boundary"
    else:
        old_between = [
            item
            for item in alignment.get("old_between") or []
            if isinstance(item, dict)
        ]
        if (
            not predecessor_donor
            or not predecessor_old
            or not successor_donor
            or not successor_old
        ):
            return None
        insertion_index = old_floor
        if old_between:
            if str(target.get("kind") or "") != "macro":
                return None
            boundary_basis = "dominating_macro_partial_order"
            proof = (
                "the macro is inserted immediately after the exact common "
                "predecessor, before every current-gap entity and the exact "
                "common successor"
            )
        else:
            boundary_basis = "empty_structural_gap_partial_order"
            proof = (
                "target is ordered between two unique common contracts "
                "and the current structural gap contains no entity"
            )
        candidates = [
            {
                "side": "ordered_common_anchor_pair",
                "insertion_index": insertion_index,
                "matched_gap_lines": 0,
                "proof": proof,
            }
        ]
    if insertion_index < old_floor or insertion_index > old_ceiling:
        return None
    return {
        "schema": "structural_entity_insertion_plan_v2",
        "status": "proven",
        "insertion_index": insertion_index,
        "insertion_line": insertion_index + 1,
        "boundary_basis": boundary_basis,
        "source_gap_exact": boundary_basis == "exact_source_gap_alignment",
        "old_gap": {
            "start_index": old_floor,
            "end_index": old_ceiling,
            "line_count": len(old_gap),
        },
        "anchor_proofs": candidates,
        **(
            {
                "donor_leading_trivia_start_line": int(
                    leading_comment["donor_leading_trivia_start_line"]
                )
            }
            if leading_comment is not None
            else {}
        ),
        "alignment": alignment,
    }


def create_clang_context(
    compile_commands_path: str | Path | None,
    root: Path,
    *,
    path_maps: list[tuple[str, str]] | None = None,
    make_bin: str | None = None,
    build_env: dict[str, str] | None = None,
    candidate_build_dir: Path | None = None,
) -> tuple[ClangCompileContext | None, dict[str, Any]]:
    """Create the only supported libclang context used by analysis/planning code."""

    if not compile_commands_path:
        return None, {
            "enabled": False,
            "backend": "code_index_v3",
            "reason": "missing_compile_commands",
            "root": str(root.resolve()),
        }
    maps = list(path_maps or [])
    try:
        context = ClangCompileContext(
            Path(compile_commands_path).resolve(),
            root.resolve(),
            path_maps=maps,
            make_bin=make_bin,
            build_env=build_env,
            candidate_build_dir=candidate_build_dir,
        )
    except Exception as exc:
        return None, {
            "enabled": False,
            "backend": "code_index_v3",
            "reason": f"context_init_failed:{exc.__class__.__name__}",
            "error": str(exc),
            "compile_commands": str(Path(compile_commands_path).resolve()),
            "root": str(root.resolve()),
            "path_maps": [{"from": old, "to": new} for old, new in maps],
        }
    return context, {
        "enabled": True,
        "backend": "code_index_v3",
        "compile_commands": str(Path(compile_commands_path).resolve()),
        "root": str(root.resolve()),
        "path_maps": [{"from": old, "to": new} for old, new in maps],
        "candidate_build_dir": (
            str(candidate_build_dir.resolve())
            if candidate_build_dir is not None
            else None
        ),
    }


def _source_like(relative_path: str) -> bool:
    return Path(relative_path).suffix in SOURCE_SUFFIXES


def _cursor_kind(cursor: Any | None) -> str:
    if cursor is None:
        return ""
    return str(getattr(getattr(cursor, "kind", None), "name", "") or "")


def _cursor_name(cursor: Any | None) -> str:
    if cursor is None:
        return ""
    for candidate in (
        getattr(cursor, "spelling", None),
        getattr(cursor, "displayname", None),
        getattr(getattr(cursor, "referenced", None), "spelling", None),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _cursor_usr(cursor: Any | None) -> str:
    if cursor is None or not hasattr(cursor, "get_usr"):
        return ""
    try:
        return str(cursor.get_usr() or "")
    except Exception:
        return ""


def _cursor_tokens(cursor: Any) -> list[str]:
    try:
        return [str(token.spelling) for token in cursor.get_tokens()]
    except Exception:
        return []


def _token_digest(tokens: list[str], fallback: str) -> tuple[str, int]:
    import hashlib

    payload = " ".join(tokens or [fallback]).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest(), len(tokens)


def _cursor_extent(cursor: Any | None) -> dict[str, int]:
    extent = getattr(cursor, "extent", None) if cursor is not None else None
    if extent is None:
        return {"start_line": 0, "start_column": 0, "end_line": 0, "end_column": 0}
    return {
        "start_line": int(getattr(extent.start, "line", 0) or 0),
        "start_column": int(getattr(extent.start, "column", 0) or 0),
        "end_line": int(getattr(extent.end, "line", 0) or 0),
        "end_column": int(getattr(extent.end, "column", 0) or 0),
    }


def _semantic_kind(cursor: Any | None) -> str:
    kind = _cursor_kind(cursor)
    if kind in {"FUNCTION_DECL", "CXX_METHOD", "CONSTRUCTOR", "DESTRUCTOR"}:
        return "function"
    if kind == "VAR_DECL":
        return "variable"
    if kind in {"STRUCT_DECL", "UNION_DECL", "ENUM_DECL", "TYPEDEF_DECL"}:
        return "type"
    if kind == "FIELD_DECL":
        return "field"
    if kind == "ENUM_CONSTANT_DECL":
        return "enum_constant"
    if kind == "MACRO_DEFINITION":
        return "macro"
    return "symbol"


def _use_kind(cursor: Any) -> str:
    kind = _cursor_kind(cursor)
    if kind == "CALL_EXPR":
        return "call"
    if kind in {"MEMBER_REF", "MEMBER_REF_EXPR"}:
        return "memberref"
    if kind == "TYPE_REF":
        return "typeref"
    if kind == "MACRO_INSTANTIATION":
        return "macro"
    return "declref"


def _semantic_parent(cursor: Any | None) -> Any | None:
    for relation in ("semantic_parent", "lexical_parent"):
        parent = getattr(cursor, relation, None) if cursor is not None else None
        while parent is not None and _cursor_kind(parent) not in OWNER_CURSOR_KINDS:
            next_parent = getattr(parent, relation, None)
            if next_parent is parent:
                break
            parent = next_parent
        if parent is not None and _cursor_kind(parent) in OWNER_CURSOR_KINDS:
            return parent
    return None


def _container_record(cursor: Any | None) -> dict[str, str] | None:
    parent = getattr(cursor, "semantic_parent", None) if cursor is not None else None
    parent_kind = _cursor_kind(parent)
    if parent_kind not in {"STRUCT_DECL", "UNION_DECL", "ENUM_DECL"}:
        return None
    name = _cursor_name(parent)
    if not name:
        return None
    return {
        "container_kind": {
            "STRUCT_DECL": "struct",
            "UNION_DECL": "union",
            "ENUM_DECL": "enum",
        }[parent_kind],
        "container_name": name,
    }


def contract_key(
    semantic_kind: str,
    symbol: str,
    *,
    container: dict[str, str] | None = None,
) -> str:
    if semantic_kind in {"field", "enum_constant"} and container:
        return (
            f"{semantic_kind}:{container.get('container_kind') or ''}:"
            f"{container.get('container_name') or ''}:{symbol}"
        )
    return f"{semantic_kind}:{symbol}"


def reference_endpoint_contract(
    *,
    cursor_kind: str,
    raw_symbol: str,
    semantic_kind: str,
    container: dict[str, str] | None,
    target_entity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Use the typed endpoint identity instead of the reference spelling."""

    if isinstance(target_entity, dict):
        target_symbol = str(target_entity.get("symbol") or "").strip()
        target_kind = str(target_entity.get("kind") or "").strip()
        target_contract = str(
            target_entity.get("contract_key")
            or target_entity.get("key")
            or ""
        ).strip()
        target_container = (
            dict(target_entity.get("container") or {})
            if isinstance(target_entity.get("container"), dict)
            else None
        )
        if target_symbol and target_kind:
            return {
                "symbol": target_symbol,
                "kind": target_kind,
                "container": target_container,
                "contract_key": target_contract
                or contract_key(
                    target_kind,
                    target_symbol,
                    container=target_container,
                ),
            }

    symbol = str(raw_symbol or "").strip()
    if cursor_kind == "TYPE_REF":
        tokens = symbol.split()
        if len(tokens) == 2 and tokens[0] in {"struct", "union", "enum"}:
            symbol = tokens[1]
    return {
        "symbol": symbol,
        "kind": semantic_kind,
        "container": container,
        "contract_key": contract_key(
            semantic_kind,
            symbol,
            container=container,
        ),
    }


def reference_provider_contract_keys(
    reference: dict[str, Any],
) -> tuple[str, ...]:
    """Return exact provider contracts permitted by the C use site.

    An unresolved call is represented by libclang as a function endpoint even
    when preprocessing would have resolved the same spelling to a function-like
    macro. The use site therefore proves a callable lookup, not that the
    provider must be a FUNCTION_DECL. Preserve both exact provider identities
    so the dependency graph can bind the call to whichever entity is present.
    """

    symbol = str(reference.get("symbol") or "").strip()
    primary = str(reference.get("contract_key") or "").strip()
    keys = [primary] if primary else []
    if symbol and str(reference.get("use_kind") or "") == "call":
        keys.extend(
            contract_key(kind, symbol)
            for kind in ("function", "macro")
        )
    return tuple(_ordered_unique(keys))


def reference_crosses_entity_boundary(
    reference: dict[str, Any],
) -> bool:
    """Return whether one typed use depends on a distinct source entity."""

    target = reference.get("referenced_entity")
    owner = reference.get("owner_entity")
    if not isinstance(target, dict) or not target.get("id"):
        return False
    if not isinstance(owner, dict) or not owner.get("id"):
        return True
    if str(target.get("id")) == str(owner.get("id")):
        return False
    if normalize_relpath(str(target.get("file") or "")) != normalize_relpath(
        str(owner.get("file") or "")
    ):
        return True
    target_extent = dict(target.get("extent") or {})
    owner_extent = dict(owner.get("extent") or {})
    target_start = int(target_extent.get("start_line") or 0)
    target_end = int(target_extent.get("end_line") or target_start)
    owner_start = int(owner_extent.get("start_line") or 0)
    owner_end = int(owner_extent.get("end_line") or owner_start)
    if (
        owner_start
        and target_start
        and owner_start <= target_start <= target_end <= owner_end
    ):
        return False
    return True


def source_location_spelling_evidence(
    symbol: str,
    source_lines: list[bytes],
    *,
    line: int,
    column: int,
) -> dict[str, Any]:
    if line <= 0 or column <= 0 or line > len(source_lines):
        return {
            "status": "unavailable",
            "target_symbol": symbol,
            "source_identifier": None,
            "line": line,
            "column": column,
        }

    source_line = source_lines[line - 1]
    byte_offset = column - 1
    source_identifier = None
    for match in re.finditer(rb"[A-Za-z_][A-Za-z0-9_]*", source_line):
        if match.start() <= byte_offset < match.end():
            source_identifier = match.group(0).decode("ascii")
            break

    if source_identifier is None:
        status = "unavailable"
    elif source_identifier == symbol:
        status = "target_spelled_at_source_location"
    else:
        status = "different_identifier_at_source_location"
    return {
        "status": status,
        "target_symbol": symbol,
        "source_identifier": source_identifier,
        "line": line,
        "column": column,
    }


def _balanced_call_arguments(
    text: str,
    open_index: int,
) -> tuple[int, list[tuple[int, int]]] | None:
    """Return a call's closing parenthesis and top-level argument spans."""

    if open_index < 0 or open_index >= len(text) or text[open_index] != "(":
        return None
    stack = ["("]
    argument_start = open_index + 1
    argument_spans: list[tuple[int, int]] = []
    index = open_index + 1
    quote = ""
    escaped = False
    line_comment = False
    block_comment = False
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char == "/" and next_char == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            index += 2
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        if char in "([{":
            stack.append(char)
            index += 1
            continue
        if char in ")]}":
            expected = {")": "(", "]": "[", "}": "{"}[char]
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
            if not stack:
                argument_spans.append((argument_start, index))
                return index, argument_spans
            index += 1
            continue
        if char == "," and len(stack) == 1:
            argument_spans.append((argument_start, index))
            argument_start = index + 1
        index += 1
    return None


def parse_module_metadata_calls(source: str) -> list[dict[str, Any]]:
    """Locate simple, complete module metadata calls outside C comments/literals."""

    visible = _C_COMMENT_OR_LITERAL_RE.sub(
        lambda match: "".join("\n" if char == "\n" else " " for char in match.group()),
        source,
    )
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"\bMODULE_(?:IMPORT_NS|LICENSE)\s*\(", visible):
        open_index = visible.find("(", match.start(), match.end())
        parsed = _balanced_call_arguments(source, open_index)
        if parsed is None or len(parsed[1]) != 1:
            continue
        close_index, spans = parsed
        end_index = close_index + 1
        while end_index < len(source) and source[end_index].isspace():
            end_index += 1
        if end_index >= len(source) or source[end_index] != ";":
            continue
        argument = source[spans[0][0] : spans[0][1]].strip()
        if not re.fullmatch(r'"[A-Za-z_][A-Za-z0-9_]*"|[A-Za-z_][A-Za-z0-9_]*', argument):
            continue
        name = visible[match.start() : open_index].strip()
        calls.append(
            {
                "name": name,
                "argument": argument,
                "namespace": argument.strip('"') if name == "MODULE_IMPORT_NS" else None,
                "line": source.count("\n", 0, match.start()) + 1,
                "end_line": source.count("\n", 0, end_index) + 1,
                "raw": source[match.start() : end_index + 1],
            }
        )
    return calls


def _macro_invocation_syntax(
    root: Path,
    relative_path: str,
    *,
    line: int,
    column: int,
) -> dict[str, Any] | None:
    """Slice the smallest call-like source expression at a macro note site."""

    source_path = root / normalize_relpath(relative_path)
    if line <= 0 or column <= 0 or not source_path.is_file():
        return None
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)
    if line > len(lines):
        return None
    line_start = sum(len(value) for value in lines[: line - 1])
    location_offset = min(
        len(text),
        line_start + max(0, column - 1),
    )
    source_line_end = line_start + len(lines[line - 1])
    candidates: list[dict[str, Any]] = []
    for match in re.finditer(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        text[line_start:source_line_end],
    ):
        identifier_start = line_start + match.start()
        cursor = line_start + match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "(":
            continue
        parsed = _balanced_call_arguments(text, cursor)
        if parsed is None:
            continue
        close_index, argument_spans = parsed
        if not (
            identifier_start <= location_offset <= close_index
            or identifier_start <= location_offset <= cursor
        ):
            continue
        candidates.append(
            {
                "name": match.group(0),
                "start_offset": identifier_start,
                "end_offset": close_index + 1,
                "open_offset": cursor,
                "argument_spans": argument_spans,
            }
        )
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda item: (
            int(item["end_offset"]) - int(item["start_offset"]),
            int(item["start_offset"]),
        ),
    )
    arguments: list[dict[str, Any]] = []
    type_tags: set[tuple[str, str]] = set()
    for index, (start, end) in enumerate(selected["argument_spans"]):
        argument_text = text[start:end]
        identifiers = [
            match.group(0)
            for match in re.finditer(
                r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                argument_text,
            )
        ]
        tags = [
            (match.group(1), match.group(2))
            for match in re.finditer(
                r"\b(struct|union|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                argument_text,
            )
        ]
        type_tags.update(tags)
        arguments.append(
            {
                "index": index,
                "text": argument_text.strip(),
                "identifiers": identifiers,
                "type_tags": [
                    {"kind": kind, "name": name} for kind, name in tags
                ],
            }
        )
    selected["arguments"] = arguments
    selected["type_tags"] = [
        {"kind": kind, "name": name} for kind, name in sorted(type_tags)
    ]
    selected.pop("argument_spans", None)
    return selected


def _macro_pasted_identifiers(
    normalized_tokens: Iterable[object],
    arguments: Iterable[dict[str, Any]],
) -> set[str]:
    """Resolve simple parameter token-pastes from a parsed macro invocation."""

    tokens = [str(token) for token in normalized_tokens]
    values = list(arguments)
    if len(tokens) < 6 or tokens[:2] != ["<self>", "("]:
        return set()
    try:
        close = tokens.index(")", 2)
    except ValueError:
        return set()
    parameter_tokens = tokens[2:close]
    if not parameter_tokens:
        return set()
    if any(token != "," for token in parameter_tokens[1::2]):
        return set()
    parameters = parameter_tokens[::2]
    if (
        len(parameters) != len(values)
        or any(not IDENTIFIER_RE.fullmatch(parameter) for parameter in parameters)
    ):
        return set()
    substitutions = {
        parameter: str(value.get("text") or "").strip()
        for parameter, value in zip(parameters, values)
    }
    if any(
        not IDENTIFIER_RE.fullmatch(value)
        for value in substitutions.values()
    ):
        return set()
    body = tokens[close + 1 :]
    products: set[str] = set()
    for index in range(1, len(body) - 1):
        if body[index] != "##":
            continue
        left, right = body[index - 1], body[index + 1]
        if left not in substitutions and right not in substitutions:
            continue
        product = substitutions.get(left, left) + substitutions.get(right, right)
        if IDENTIFIER_RE.fullmatch(product):
            products.add(product)
    return products


def entity_node_id(
    *,
    tree: str,
    usr: str,
    file: str,
    cursor_kind: str,
    symbol: str,
    extent: dict[str, int],
    is_definition: bool,
) -> str:
    identity_parts = [
        usr,
        normalize_relpath(file),
        cursor_kind,
        symbol,
        "definition" if is_definition else "declaration",
    ]
    if not usr:
        identity_parts.extend(
            [
                str(int(extent.get("start_line") or 0)),
                str(int(extent.get("start_column") or 0)),
                str(int(extent.get("end_line") or 0)),
                str(int(extent.get("end_column") or 0)),
            ]
        )
    identity = ":".join(identity_parts)
    return stable_id("entity", tree, identity)


def _cursor_file(
    cursor: Any | None,
    *,
    path_cache: dict[str, Path | None] | None = None,
) -> Path | None:
    location = getattr(cursor, "location", None) if cursor is not None else None
    value = getattr(location, "file", None)
    if value is None:
        return None
    raw_path = str(getattr(value, "name", "") or value)
    if path_cache is not None and raw_path in path_cache:
        return path_cache[raw_path]
    result = resolved_path(raw_path)
    if path_cache is not None:
        path_cache[raw_path] = result
    return result


def _entity_cache_key(
    cursor: Any,
    root: Path,
    tree: str,
    *,
    file_path: Path | None,
    cursor_kind: str,
    symbol: str,
    usr: str,
) -> tuple[str, ...] | None:
    if file_path is None or not is_under(file_path, root):
        return None
    location = getattr(cursor, "location", None)
    return (
        tree,
        usr,
        normalize_relpath(relative_to_root(file_path, root)),
        cursor_kind,
        symbol,
        str(int(getattr(location, "line", 0) or 0)),
        str(int(getattr(location, "column", 0) or 0)),
    )


def _cursor_is_definition(cursor: Any, cursor_kind: str) -> bool:
    if cursor_kind in {"MACRO_DEFINITION", "FIELD_DECL", "ENUM_CONSTANT_DECL"}:
        return True
    try:
        if cursor.is_definition():
            return True
        if cursor_kind != "VAR_DECL":
            return False
        if _cursor_kind(cursor.semantic_parent) != "TRANSLATION_UNIT":
            return False
        if cursor.storage_class.name not in {"NONE", "STATIC"}:
            return False
        return cursor.get_definition() is None
    except Exception:
        return False


def _entity_record(
    cursor: Any,
    root: Path,
    tree: str,
    *,
    cache: dict[tuple[str, ...], dict[str, Any] | None] | None = None,
    include_fingerprints: bool = True,
    file_path_cache: dict[str, Path | None] | None = None,
    cursor_kind_hint: str | None = None,
    file_path_hint: Path | None = None,
) -> dict[str, Any] | None:
    cursor_kind = (
        cursor_kind_hint
        if cursor_kind_hint is not None
        else _cursor_kind(cursor)
    )
    if cursor_kind not in DEFINITION_CURSOR_KINDS:
        return None
    symbol = _cursor_name(cursor)
    file_path = (
        file_path_hint
        if file_path_hint is not None
        else _cursor_file(cursor, path_cache=file_path_cache)
    )
    usr = _cursor_usr(cursor)
    cache_key = _entity_cache_key(
        cursor,
        root,
        tree,
        file_path=file_path,
        cursor_kind=cursor_kind,
        symbol=symbol,
        usr=usr,
    )
    if cache is not None and cache_key is not None and cache_key in cache:
        return cache[cache_key]
    if not symbol or file_path is None or not is_under(file_path, root):
        if cache is not None and cache_key is not None:
            cache[cache_key] = None
        return None
    relative_path = normalize_relpath(relative_to_root(file_path, root))
    semantic_kind = _semantic_kind(cursor)
    container = _container_record(cursor)
    extent = _cursor_extent(cursor)
    tokens = _cursor_tokens(cursor)
    digest, token_count = _token_digest(tokens, f"{cursor_kind}:{symbol}")
    fingerprints = cursor_fingerprints(cursor, symbol=symbol) if include_fingerprints else {}
    is_definition = _cursor_is_definition(cursor, cursor_kind)
    node_id = entity_node_id(
        tree=tree,
        usr=usr,
        file=relative_path,
        cursor_kind=cursor_kind,
        symbol=symbol,
        extent=extent,
        is_definition=is_definition,
    )
    record = {
        "id": node_id,
        "entity_id": node_id,
        "tree": tree,
        "key": contract_key(semantic_kind, symbol, container=container),
        "contract_key": contract_key(semantic_kind, symbol, container=container),
        "symbol": symbol,
        "name": symbol,
        "kind": semantic_kind,
        "cursor_kind": cursor_kind,
        "displayname": str(getattr(cursor, "displayname", "") or symbol),
        "usr": usr,
        "container": container,
        "is_definition": is_definition,
        "location": location_dict(getattr(cursor, "location", None)),
        "file": relative_path,
        "extent": extent,
        "digest": digest,
        "token_count": token_count,
        **fingerprints,
    }
    if cache is not None and cache_key is not None:
        cache[cache_key] = record
    return record


def _referenced_cursor(cursor: Any) -> Any | None:
    referenced = getattr(cursor, "referenced", None)
    if referenced is not None:
        return referenced
    try:
        return cursor.get_definition()
    except Exception:
        return None


def _reference_record(
    cursor: Any,
    *,
    root: Path,
    driver_root: Path,
    tree: str,
    owner_cursor: Any | None = None,
    entity_cache: dict[tuple[str, ...], dict[str, Any] | None] | None = None,
    reference_target_entity_ids: set[str] | None = None,
    source_lines: list[bytes] | None = None,
    file_path_cache: dict[str, Path | None] | None = None,
    cursor_kind_hint: str | None = None,
    source_file_hint: Path | None = None,
    source_relative_hint: str | None = None,
    ast_path: tuple[tuple[str, int], ...] = (),
) -> dict[str, Any] | None:
    cursor_kind = (
        cursor_kind_hint
        if cursor_kind_hint is not None
        else _cursor_kind(cursor)
    )
    if cursor_kind not in REFERENCE_CURSOR_KINDS:
        return None
    source_file = (
        source_file_hint
        if source_file_hint is not None
        else _cursor_file(cursor, path_cache=file_path_cache)
    )
    symbol = _cursor_name(cursor)
    if source_file is None or not is_under(source_file, root) or not symbol:
        return None
    source_relative = (
        normalize_relpath(source_relative_hint)
        if source_relative_hint is not None
        else normalize_relpath(relative_to_root(source_file, root))
    )
    location = getattr(cursor, "location", None)
    line = int(getattr(location, "line", 0) or 0)
    column = int(getattr(location, "column", 0) or 0)
    referenced = _referenced_cursor(cursor)
    referenced_file = _cursor_file(
        referenced,
        path_cache=file_path_cache,
    )
    referenced_relative = ""
    dependency_scope = "unresolved"
    target_entity = None
    if referenced_file is not None and is_under(referenced_file, root):
        referenced_relative = normalize_relpath(relative_to_root(referenced_file, root))
        dependency_scope = "local_driver" if is_under(referenced_file, driver_root) else "kernel_tree"
        if reference_target_entity_ids is None:
            target_entity = _entity_record(
                referenced,
                root,
                tree,
                cache=entity_cache,
                file_path_cache=file_path_cache,
            )
        else:
            identity = _entity_record(
                referenced,
                root,
                tree,
                include_fingerprints=False,
                file_path_cache=file_path_cache,
            )
            if identity is None or str(identity.get("id") or "") not in reference_target_entity_ids:
                return None
            target_entity = _entity_record(
                referenced,
                root,
                tree,
                cache=entity_cache,
                file_path_cache=file_path_cache,
            )
    elif referenced_file is not None:
        dependency_scope = "external"
    if reference_target_entity_ids is not None and target_entity is None:
        return None
    semantic_kind = _semantic_kind(referenced)
    referenced_kind = _cursor_kind(referenced)
    if semantic_kind == "symbol":
        semantic_kind = {
            "CALL_EXPR": "function",
            "MEMBER_REF": "field",
            "MEMBER_REF_EXPR": "field",
            "TYPE_REF": "type",
            "MACRO_INSTANTIATION": "macro",
        }.get(cursor_kind, "symbol")
    container = _container_record(referenced)
    endpoint_contract = reference_endpoint_contract(
        cursor_kind=_cursor_kind(cursor),
        raw_symbol=symbol,
        semantic_kind=semantic_kind,
        container=container,
        target_entity=target_entity,
    )
    symbol = str(endpoint_contract["symbol"])
    semantic_kind = str(endpoint_contract["kind"])
    container = endpoint_contract.get("container")
    spelling_evidence = source_location_spelling_evidence(
        symbol,
        source_lines or [],
        line=line,
        column=column,
    )
    resolved_owner = owner_cursor or _semantic_parent(cursor)
    owner = _entity_record(
        resolved_owner,
        root,
        tree,
        cache=entity_cache,
        file_path_cache=file_path_cache,
        file_path_hint=source_file if owner_cursor is not None else None,
    )
    compact_ast_path = [
        {
            "cursor_kind": str(kind),
            "child_index": int(child_index),
        }
        for kind, child_index in ast_path
    ]
    owner_resolution = {
        "status": (
            "cursor_semantic_owner"
            if owner is not None
            else "unresolved"
        ),
        "candidate_count": int(owner is not None),
    }
    reference_context_fingerprint = _reference_context_fingerprint(
        owner=owner,
        target=target_entity,
        cursor_kind=cursor_kind,
        use_kind=str(_use_kind(cursor)),
        ast_path=compact_ast_path,
    )
    reference_use_contract = _reference_use_contract(
        cursor,
        target=target_entity,
        cursor_kind=cursor_kind,
        use_kind=str(_use_kind(cursor)),
    )
    reference_id = stable_id(
        "reference",
        tree,
        source_relative,
        str(line),
        str(column),
        cursor_kind,
        str((target_entity or {}).get("id") or symbol),
    )
    return {
        "id": reference_id,
        "reference_id": reference_id,
        "tree": tree,
        "source_file": source_relative,
        "symbol": symbol,
        "kind": semantic_kind,
        "symbol_kind": semantic_kind,
        "use_kind": _use_kind(cursor),
        "cursor_kind": cursor_kind,
        "line": line,
        "column": column,
        "extent": _cursor_extent(cursor),
        "owner_entity_id": (owner or {}).get("id"),
        "owner_entity": owner,
        "referenced_entity_id": (target_entity or {}).get("id"),
        "referenced_entity": target_entity,
        "referenced_file": referenced_relative or None,
        "referenced_kind": referenced_kind,
        "referenced_location": location_dict(getattr(referenced, "location", None)) if referenced else None,
        "dependency_scope": dependency_scope,
        "container": container,
        "contract_key": endpoint_contract["contract_key"],
        "source_spelling_evidence": spelling_evidence,
        "ast_path": compact_ast_path,
        "owner_resolution": owner_resolution,
        "reference_context_fingerprint": (
            reference_context_fingerprint
        ),
        "reference_context_depth": len(compact_ast_path),
        "reference_use_contract": reference_use_contract,
        "reference_use_fingerprint": _reference_use_fingerprint(
            reference_use_contract
        ),
    }


def _reference_use_contract(
    cursor: Any,
    *,
    target: dict[str, Any] | None,
    cursor_kind: str,
    use_kind: str,
) -> dict[str, Any]:
    """Describe the typed shape of one dependency edge use site.

    The graph location fingerprint intentionally ignores expression payloads so
    equivalent owners can be aligned across line movement. This companion
    contract captures the type-level use shape needed to distinguish, for
    example, a stale call from a call adapted to a changed provider signature.
    """

    signature = cursor_signature_payload(cursor)
    target_signature = (
        (target or {}).get("signature")
        if isinstance((target or {}).get("signature"), dict)
        else {}
    )
    call_argument_types = [
        str(item)
        for item in signature.get("argument_types") or []
    ]
    target_argument_types = [
        str(item)
        for item in target_signature.get("argument_types") or []
    ]
    target_signature_available = bool(target_signature)
    target_variadic = bool(target_signature.get("variadic"))
    is_call = str(use_kind or "") == "call"
    arity_compatible: bool | None = None
    if is_call and target_signature_available:
        arity_compatible = bool(
            len(call_argument_types) >= len(target_argument_types)
            if target_variadic
            else len(call_argument_types)
            == len(target_argument_types)
        )
    return {
        "cursor_kind": str(cursor_kind or ""),
        "use_kind": str(use_kind or ""),
        "canonical_type": str(signature.get("canonical_type") or ""),
        "result_type": str(signature.get("result_type") or ""),
        "argument_types": call_argument_types,
        "argument_count": len(call_argument_types),
        "variadic": bool(signature.get("variadic")),
        "target_contract_key": str(
            (target or {}).get("contract_key")
            or (target or {}).get("key")
            or ""
        ),
        "target_signature_fingerprint": str(
            (target or {}).get("signature_fingerprint") or ""
        ),
        "target_interface_fingerprint": str(
            (target or {}).get("interface_fingerprint") or ""
        ),
        "target_argument_types": target_argument_types,
        "target_argument_count": len(target_argument_types),
        "target_signature_available": target_signature_available,
        "target_variadic": target_variadic,
        "arity_compatible": arity_compatible,
    }


def _reference_use_fingerprint(contract: dict[str, Any]) -> str:
    encoded = json.dumps(
        contract,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reference_context_fingerprint(
    *,
    owner: dict[str, Any] | None,
    target: dict[str, Any] | None,
    cursor_kind: str,
    use_kind: str,
    ast_path: Iterable[dict[str, Any]],
) -> str:
    return stable_id(
        "reference-context",
        str((owner or {}).get("contract_key") or ""),
        str((target or {}).get("contract_key") or ""),
        str(cursor_kind or ""),
        str(use_kind or ""),
        *[
            (
                f"{str(item.get('cursor_kind') or '')}:"
                f"{int(item.get('child_index') or 0)}"
            )
            for item in ast_path
            if isinstance(item, dict)
        ],
    )


def _initializer_branch_key(
    reference: dict[str, Any],
) -> tuple[tuple[str, int], ...]:
    """Return the exact designated-initializer branch containing a use."""

    path = _reference_ast_path(reference)
    initializer_indexes = [
        index
        for index, (kind, _) in enumerate(path)
        if kind == "INIT_LIST_EXPR"
    ]
    if not initializer_indexes:
        return ()
    index = initializer_indexes[-1]
    # The first child below INIT_LIST_EXPR identifies one initializer slot;
    # references in another callback field therefore cannot become siblings.
    if index + 1 >= len(path):
        return ()
    return path[: index + 2]


def _reference_ast_path(
    reference: dict[str, Any],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (
            str(item.get("cursor_kind") or ""),
            int(item.get("child_index") or 0),
        )
        for item in reference.get("ast_path") or []
        if isinstance(item, dict)
    )


def _local_return_owner_binding(
    exact_references: Iterable[dict[str, Any]],
    file_references: Iterable[dict[str, Any]],
    *,
    source_root: Path,
) -> dict[str, Any]:
    """Keep a resolved parameter in a return expression on its function owner."""

    selected = list(exact_references)
    if len(selected) != 1:
        return {}
    reference = selected[0]
    owner = reference.get("owner_entity") or {}
    declaration = reference.get("referenced_location") or {}
    if (
        str(reference.get("referenced_kind") or "") != "PARM_DECL"
        or isinstance(reference.get("referenced_entity"), dict)
        or str(owner.get("kind") or "").lower() != "function"
        or owner.get("is_definition") is not True
        or not str((owner.get("signature") or {}).get("result_type") or "")
        or str((owner.get("signature") or {}).get("result_type") or "") == "void"
        or not str(reference.get("owner_entity_id") or "")
        or str(reference.get("source_file") or "") != str(owner.get("file") or "")
        or str(reference.get("referenced_file") or "") != str(owner.get("file") or "")
        or not declaration.get("file")
        or Path(str(declaration["file"])).resolve()
        != (source_root / str(owner.get("file") or "")).resolve()
    ):
        return {}
    extent = owner.get("extent") or {}
    declaration_line = int(declaration.get("line") or 0)
    if not (
        int(extent.get("start_line") or 0) <= declaration_line
        <= int(reference.get("line") or 0)
        <= int(extent.get("end_line") or 0)
    ):
        return {}
    path = _reference_ast_path(reference)
    if not any(kind == "RETURN_STMT" for kind, _ in path):
        return {}
    enclosing = [
        item
        for item in file_references
        if str(item.get("cursor_kind") or "") == "MEMBER_REF_EXPR"
        and str(item.get("owner_entity_id") or "")
        == str(reference.get("owner_entity_id") or "")
        and int(item.get("line") or 0) == int(reference.get("line") or 0)
        and isinstance(item.get("referenced_entity"), dict)
        and _entity_has_source_extent(item.get("referenced_entity") or {})
        and (
            member_path := _reference_ast_path(item)
        )
        and len(member_path) < len(path)
        and path[:len(member_path)] == member_path
    ]
    if len(enclosing) != 1:
        return {}
    return {
        "owner": owner,
        "local_reference_id": reference.get("id"),
        "member_reference_id": enclosing[0].get("id"),
        "member_entity_id": (enclosing[0].get("referenced_entity") or {}).get("id"),
    }


def _binary_operator_branch_key(
    reference: dict[str, Any],
) -> tuple[tuple[tuple[str, int], ...], int | None]:
    """Return one binary-expression identity and this use's operand index."""

    path = _reference_ast_path(reference)
    binary_indexes = [
        index
        for index, (kind, _) in enumerate(path)
        if kind == "BINARY_OPERATOR"
    ]
    if not binary_indexes:
        return (), None
    index = binary_indexes[-1]
    if index + 1 >= len(path):
        return (), None
    return path[: index + 1], path[index + 1][1]


def _call_expression_branch_key(
    reference: dict[str, Any],
) -> tuple[tuple[tuple[str, int], ...], int | None]:
    """Return the innermost call identity and this use's operand index."""

    path = _reference_ast_path(reference)
    call_indexes = [
        index
        for index, (kind, _) in enumerate(path)
        if kind == "CALL_EXPR"
    ]
    if not call_indexes:
        return (), None
    index = call_indexes[-1]
    if index + 1 >= len(path):
        return (), None
    return path[: index + 1], path[index + 1][1]


def _call_argument_mismatch_binding(
    exact_references: Iterable[dict[str, Any]],
    file_references: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve one incompatible call argument to its exact current owner."""

    def argument_endpoint(reference: dict[str, Any]) -> bool:
        target = reference.get("referenced_entity") or {}
        if isinstance(target, dict) and _entity_has_source_extent(target):
            return True
        owner = reference.get("owner_entity") or {}
        location = reference.get("referenced_location") or {}
        extent = owner.get("extent") or {}
        return bool(
            reference.get("referenced_kind") == "PARM_DECL"
            and reference.get("tree") == "old"
            and owner.get("tree") == "old"
            and str(owner.get("kind") or "").lower() == "function"
            and _entity_has_source_extent(owner)
            and normalize_relpath(str(reference.get("referenced_file") or ""))
            == normalize_relpath(str(owner.get("file") or ""))
            and int(extent.get("start_line") or 0)
            <= int(location.get("line") or 0)
            <= int(extent.get("end_line") or 0)
            and int(location.get("column") or 0) > 0
        )

    exact_arguments = []
    for reference in exact_references:
        if not isinstance(reference, dict):
            continue
        branch, operand = _call_expression_branch_key(reference)
        owner_id = str(reference.get("owner_entity_id") or "")
        if (
            owner_id
            and branch
            and operand is not None
            and operand > 0
            and argument_endpoint(reference)
        ):
            exact_arguments.append((reference, owner_id, branch))
    branches = {
        (owner_id, branch)
        for _, owner_id, branch in exact_arguments
    }
    if len(branches) != 1:
        return {}
    owner_id, branch = next(iter(branches))
    branch_references = [
        reference
        for reference in file_references
        if isinstance(reference, dict)
        and str(reference.get("owner_entity_id") or "") == owner_id
        and _call_expression_branch_key(reference)[0] == branch
    ]
    callee_references = []
    argument_references = []
    for reference in branch_references:
        _, operand = _call_expression_branch_key(reference)
        target = (
            reference.get("referenced_entity")
            if isinstance(reference.get("referenced_entity"), dict)
            else {}
        )
        if (
            operand == 0
            and str(reference.get("kind") or "").lower() == "function"
            and target
            and _entity_has_source_extent(target)
        ):
            callee_references.append(reference)
        elif operand is not None and operand > 0 and argument_endpoint(reference):
            argument_references.append(reference)
    callee_ids = {
        str(
            reference.get("referenced_entity_id")
            or (reference.get("referenced_entity") or {}).get("id")
            or ""
        )
        for reference in callee_references
    }
    callee_ids.discard("")
    exact_argument_ids = {
        str(
            reference.get("referenced_entity_id")
            or (reference.get("referenced_entity") or {}).get("id")
            or ""
        )
        for reference, _, _ in exact_arguments
        if _entity_has_source_extent(
            reference.get("referenced_entity") or {}
        )
    }
    exact_argument_ids.discard("")
    argument_ids = {
        str(
            reference.get("referenced_entity_id")
            or (reference.get("referenced_entity") or {}).get("id")
            or ""
        )
        for reference in argument_references
    }
    argument_ids.discard("")
    if (
        len(callee_ids) != 1
        or not exact_arguments
        or not exact_argument_ids.issubset(argument_ids)
        or any(reference not in argument_references for reference, _, _ in exact_arguments)
    ):
        return {}

    endpoint_references = [*callee_references, *argument_references]
    owner_ids = {
        str(reference.get("owner_entity_id") or "")
        for reference in endpoint_references
    }
    owner_ids.discard("")
    if len(owner_ids) != 1:
        return {}
    owner_id = next(iter(owner_ids))
    owners = canonical_entity_records(
        reference.get("owner_entity")
        for reference in endpoint_references
        if isinstance(reference.get("owner_entity"), dict)
        and str(reference.get("owner_entity_id") or "") == owner_id
    )
    function_owners = [
        owner
        for owner in owners
        if str(owner.get("tree") or "") == "old"
        and str(owner.get("kind") or "").lower() == "function"
        and _entity_has_source_extent(owner)
    ]
    if len(function_owners) != 1:
        return {}
    return {
        "status": "mapped_call_argument_binding_owner",
        "authority": "exact_typed_call_argument_binding",
        "owner": function_owners[0],
        "owner_entity_id": owner_id,
        "callee_references": callee_references,
        "argument_references": argument_references,
        "exact_argument_entity_ids": sorted(exact_argument_ids),
        "call_branch": [
            {"cursor_kind": cursor_kind, "child_index": child_index}
            for cursor_kind, child_index in branch
        ],
    }


def _source_offset(
    text: str,
    *,
    line: int,
    column: int,
) -> int | None:
    if line < 1 or column < 1:
        return None
    lines = text.splitlines(keepends=True)
    if line > len(lines):
        return None
    line_text = lines[line - 1]
    column_offset = column - 1
    if column_offset > len(line_text):
        return None
    return sum(len(item) for item in lines[: line - 1]) + column_offset


def _references_form_simple_assignment(
    source_root: Path | None,
    field_reference: dict[str, Any],
    provider_reference: dict[str, Any],
) -> bool:
    """Verify the exact source operator between typed callback operands."""

    if source_root is None:
        return False
    source_root = source_root.resolve()
    field_file = normalize_relpath(
        str(field_reference.get("source_file") or "")
    )
    provider_file = normalize_relpath(
        str(provider_reference.get("source_file") or "")
    )
    if not field_file or field_file != provider_file:
        return False
    source_path = (source_root / field_file).resolve()
    if not source_path.is_file() or not is_under(source_path, source_root):
        return False
    field_extent = dict(field_reference.get("extent") or {})
    provider_extent = dict(provider_reference.get("extent") or {})
    try:
        text = source_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    field_end = _source_offset(
        text,
        line=int(field_extent.get("end_line") or 0),
        column=int(field_extent.get("end_column") or 0),
    )
    provider_start = _source_offset(
        text,
        line=int(provider_extent.get("start_line") or 0),
        column=int(provider_extent.get("start_column") or 0),
    )
    if (
        field_end is None
        or provider_start is None
        or field_end > provider_start
    ):
        return False
    return "".join(text[field_end:provider_start].split()) == "="


def _callback_binding_relationship(
    source_root: Path | None,
    field_reference: dict[str, Any],
    provider_reference: dict[str, Any],
) -> tuple[str, tuple[tuple[str, int], ...]] | None:
    """Return the exact structural relation joining a field and provider."""

    field_owner = str(field_reference.get("owner_entity_id") or "")
    provider_owner = str(provider_reference.get("owner_entity_id") or "")
    if not field_owner or field_owner != provider_owner:
        return None

    field_initializer = _initializer_branch_key(field_reference)
    provider_initializer = _initializer_branch_key(provider_reference)
    if field_initializer and field_initializer == provider_initializer:
        return "initializer", field_initializer

    field_branch, field_operand = _binary_operator_branch_key(field_reference)
    provider_branch, provider_operand = _binary_operator_branch_key(
        provider_reference
    )
    if (
        field_branch
        and field_branch == provider_branch
        and field_operand == 0
        and provider_operand == 1
        and _references_form_simple_assignment(
            source_root,
            field_reference,
            provider_reference,
        )
    ):
        return "assignment", field_branch
    return None


def _callback_binding_provider_references(
    exact_references: Iterable[dict[str, Any]],
    file_references: Iterable[dict[str, Any]],
    *,
    source_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve callback providers from one exact typed binding branch.

    A diagnostic on ``.field = provider`` may be positioned on either side of
    the assignment.  A shared owner plus either one initializer slot or a
    source-verified binary assignment provides structural authority for
    recovering the other side; symbol spelling and source-line proximity do
    not.
    """

    providers: dict[str, dict[str, Any]] = {}
    candidates = [
        dict(reference)
        for reference in file_references
        if isinstance(reference, dict)
    ]
    for exact_reference in exact_references:
        if not isinstance(exact_reference, dict):
            continue
        exact_kind = str(exact_reference.get("kind") or "").lower()
        owner_id = str(exact_reference.get("owner_entity_id") or "")
        initializer_branch = _initializer_branch_key(exact_reference)
        assignment_branch, exact_operand = _binary_operator_branch_key(
            exact_reference
        )
        if not owner_id or (
            not initializer_branch and not assignment_branch
        ):
            continue
        if initializer_branch:
            branch_references = [
                candidate
                for candidate in candidates
                if str(candidate.get("owner_entity_id") or "") == owner_id
                and _initializer_branch_key(candidate)
                == initializer_branch
            ]
            field_references = [
                candidate
                for candidate in branch_references
                if str(candidate.get("kind") or "").lower() == "field"
            ]
            provider_references = [
                candidate
                for candidate in branch_references
                if str(candidate.get("kind") or "").lower()
                == "function"
            ]
        else:
            branch_references = [
                candidate
                for candidate in candidates
                if str(candidate.get("owner_entity_id") or "") == owner_id
                and _binary_operator_branch_key(candidate)[0]
                == assignment_branch
            ]
            field_references = [
                candidate
                for candidate in branch_references
                if str(candidate.get("kind") or "").lower() == "field"
                and _binary_operator_branch_key(candidate)[1] == 0
            ]
            provider_references = [
                candidate
                for candidate in branch_references
                if str(candidate.get("kind") or "").lower()
                == "function"
                and _binary_operator_branch_key(candidate)[1] == 1
                and any(
                    _callback_binding_relationship(
                        source_root,
                        field_reference,
                        candidate,
                    ) is not None
                    for field_reference in field_references
                )
            ]
        if exact_kind == "field":
            if assignment_branch and exact_operand != 0:
                continue
            matches = [
                candidate
                for candidate in provider_references
                if _entity_has_source_extent(
                    candidate.get("referenced_entity")
                    if isinstance(
                        candidate.get("referenced_entity"), dict
                    )
                    else {}
                )
            ]
        elif exact_kind == "function":
            if assignment_branch and exact_operand != 1:
                continue
            field_ids = {
                str(
                    candidate.get("referenced_entity_id")
                    or candidate.get("symbol")
                    or "callback-field"
                )
                for candidate in field_references
            }
            target = (
                exact_reference.get("referenced_entity")
                if isinstance(
                    exact_reference.get("referenced_entity"), dict
                )
                else {}
            )
            matches = (
                [exact_reference]
                if len(field_ids) == 1
                and _entity_has_source_extent(target)
                and (
                    initializer_branch
                    or exact_reference in provider_references
                )
                else []
            )
        else:
            continue
        target_ids = {
            str(
                candidate.get("referenced_entity_id")
                or (candidate.get("referenced_entity") or {}).get("id")
                or ""
            )
            for candidate in matches
        }
        target_ids.discard("")
        if len(target_ids) != 1:
            continue
        for candidate in matches:
            target_id = str(
                candidate.get("referenced_entity_id")
                or (candidate.get("referenced_entity") or {}).get("id")
                or ""
            )
            if target_id in target_ids:
                providers[target_id] = candidate
    return [providers[key] for key in sorted(providers)]


def _diagnostic_causal_root_frontier(
    root_nodes: Iterable[str],
    transaction_owner_node_ids: Iterable[str],
    *,
    exact_binding_provider_node_ids: Iterable[str] = (),
) -> set[str]:
    """Keep a containing transaction owner only as a last-resort root.

    Once an exact provider or contract has been identified, the enclosing
    function or initializer remains ownership evidence but is not causal.  If
    it entered the root frontier, traversing its other dependencies would
    merge unrelated diagnostics that merely share a source container.

    For a uniquely resolved ``.field = provider`` initializer branch, the
    existing target field defines the contract that the imported provider must
    satisfy.  Keep that field as contract evidence, but do not expose it as a
    competing write root.  Missing or ambiguous callback bindings have no
    exact provider frontier and retain the ordinary root set.
    """

    exact_binding_providers = {
        str(node_id)
        for node_id in exact_binding_provider_node_ids
        if str(node_id)
    }
    if exact_binding_providers:
        return exact_binding_providers
    roots = {str(node_id) for node_id in root_nodes if str(node_id)}
    if roots:
        return roots
    return {
        str(node_id)
        for node_id in transaction_owner_node_ids
        if str(node_id)
    }


def _resolve_reference_ownership(
    *,
    entities: Iterable[dict[str, Any]],
    references: Iterable[dict[str, Any]],
) -> None:
    """Project implementation and precise lexical ownership from AST extents.

    Detailed-processing cursors such as macro instantiations can be emitted
    directly under the translation unit even when their source position is
    inside a function or variable initializer.  The cursor parent is then
    unavailable, but parsed entity extents still provide a structural
    ownership relation.  The unique outermost entity recovers a missing
    implementation owner.  Every reference also records its innermost lexical
    owners so a diagnostic rooted at a field or declaration can retain the
    dependencies that occur inside that exact entity.
    """

    entities_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        relative_path = normalize_relpath(
            str(entity.get("file") or "")
        )
        if (
            relative_path
            and str(entity.get("cursor_kind") or "")
            in OWNER_CURSOR_KINDS
            and _entity_has_source_extent(entity)
        ):
            entities_by_file[relative_path].append(entity)

    for reference in references:
        if not isinstance(reference, dict):
            continue
        relative_path = normalize_relpath(
            str(reference.get("source_file") or "")
        )
        reference_extent = dict(reference.get("extent") or {})
        containing = [
            entity
            for entity in entities_by_file.get(relative_path, [])
            if _extent_contains_extent(
                dict(entity.get("extent") or {}),
                reference_extent,
            )
        ]
        innermost = [
            entity
            for entity in containing
            if not any(
                str(other.get("id") or "")
                != str(entity.get("id") or "")
                and _extent_contains_extent(
                    dict(entity.get("extent") or {}),
                    dict(other.get("extent") or {}),
                )
                and dict(other.get("extent") or {})
                != dict(entity.get("extent") or {})
                for other in containing
            )
        ]
        reference["lexical_owner_entity_ids"] = sorted(
            {
                str(entity.get("id") or "")
                for entity in innermost
                if str(entity.get("id") or "")
            }
        )
        if reference.get("owner_entity_id"):
            continue
        outermost = [
            entity
            for entity in containing
            if not any(
                str(other.get("id") or "")
                != str(entity.get("id") or "")
                and _extent_contains_extent(
                    dict(other.get("extent") or {}),
                    dict(entity.get("extent") or {}),
                )
                and dict(other.get("extent") or {})
                != dict(entity.get("extent") or {})
                for other in containing
            )
        ]
        unique_by_id = {
            str(entity.get("id") or ""): entity
            for entity in outermost
            if str(entity.get("id") or "")
        }
        if len(unique_by_id) != 1:
            reference["owner_resolution"] = {
                "status": (
                    "ambiguous_lexical_extent"
                    if unique_by_id
                    else "no_containing_entity"
                ),
                "candidate_count": len(unique_by_id),
                "candidate_entity_ids": sorted(unique_by_id),
            }
            continue
        owner = next(iter(unique_by_id.values()))
        reference["owner_entity_id"] = owner["id"]
        reference["owner_entity"] = owner
        reference["owner_resolution"] = {
            "status": "unique_lexical_extent",
            "candidate_count": 1,
            "candidate_entity_ids": [owner["id"]],
        }
        reference["reference_context_fingerprint"] = (
            _reference_context_fingerprint(
                owner=owner,
                target=(
                    reference.get("referenced_entity")
                    if isinstance(
                        reference.get("referenced_entity"),
                        dict,
                    )
                    else None
                ),
                cursor_kind=str(
                    reference.get("cursor_kind") or ""
                ),
                use_kind=str(reference.get("use_kind") or ""),
                ast_path=[
                    dict(item)
                    for item in reference.get("ast_path") or []
                    if isinstance(item, dict)
                ],
            )
        )


def _walk_main_file_cursors(
    cursor: Any,
    main_file: Path,
    implementation_owner: Any | None = None,
    *,
    file_path_cache: dict[str, Path | None] | None = None,
    ast_path: tuple[tuple[str, int], ...] = (),
):
    for child_index, child in enumerate(cursor.get_children()):
        location = _cursor_file(
            child,
            path_cache=file_path_cache,
        )
        if location != main_file:
            continue
        child_owner = implementation_owner
        child_kind = _cursor_kind(child)
        if child_owner is None and child_kind in OWNER_CURSOR_KINDS:
            child_owner = child
            child_ast_path = ((child_kind, 0),)
        elif child_owner is not None:
            child_ast_path = (
                *ast_path,
                (child_kind, child_index),
            )
        else:
            child_ast_path = ast_path
        yield child, child_owner, child_kind, child_ast_path
        yield from _walk_main_file_cursors(
            child,
            main_file,
            child_owner,
            file_path_cache=file_path_cache,
            ast_path=child_ast_path,
        )


def _resolve_include_target(
    *,
    root: Path,
    driver_root: Path,
    source_file: str,
    include_name: str,
    delimiter: str,
) -> str | None:
    source_parent = Path(source_file).parent
    candidates: list[Path] = []
    if delimiter == '"':
        candidates.extend(
            [
                root / source_parent / include_name,
                driver_root / include_name,
            ]
        )
    candidates.extend(
        [
            root / "include" / include_name,
            root / "include/uapi" / include_name,
            root / include_name,
        ]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and is_under(candidate, root):
            return normalize_relpath(relative_to_root(candidate, root))
    return None


def parse_include_directives(
    *,
    root: Path,
    driver_root: Path,
    relative_path: str,
    tree: str,
) -> list[dict[str, Any]]:
    path = (root / normalize_relpath(relative_path)).resolve()
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        match = INCLUDE_DIRECTIVE_RE.match(line)
        if match is None:
            continue
        name = normalize_relpath(match.group("name"))
        delimiter = match.group("delimiter")
        target = _resolve_include_target(
            root=root,
            driver_root=driver_root,
            source_file=relative_path,
            include_name=name,
            delimiter=delimiter,
        )
        records.append(
            {
                "id": stable_id("include", tree, normalize_relpath(relative_path), str(line_number), name),
                "tree": tree,
                "source_file": normalize_relpath(relative_path),
                "name": name,
                "delimiter": delimiter,
                "line": line_number,
                "target_file": target,
                "resolved": bool(target),
            }
        )
    return records


def _structural_cache_get(
    key: tuple[str, str, str, str],
) -> dict[str, Any] | None:
    cached = _STRUCTURAL_PARSE_CACHE.get(key)
    if cached is None:
        return None
    _STRUCTURAL_PARSE_CACHE.move_to_end(key)
    return {
        **cached,
        "cache": "process_structural_content_lru",
    }


def _structural_cache_put(
    key: tuple[str, str, str, str],
    report: dict[str, Any],
) -> None:
    if _STRUCTURAL_PARSE_CACHE_MAX_ENTRIES <= 0:
        return
    _STRUCTURAL_PARSE_CACHE[key] = report
    _STRUCTURAL_PARSE_CACHE.move_to_end(key)
    while len(_STRUCTURAL_PARSE_CACHE) > _STRUCTURAL_PARSE_CACHE_MAX_ENTRIES:
        _STRUCTURAL_PARSE_CACHE.popitem(last=False)


def require_structural_c_parser() -> dict[str, str]:
    """Fail unless the structural C parser is installed and executable."""

    try:
        from tree_sitter_languages import get_parser
    except ImportError as exc:
        raise RuntimeError(
            "tree_sitter_languages is required for structural C analysis"
        ) from exc
    try:
        tree = get_parser("c").parse(b"int driver_backport_preflight;")
    except Exception as exc:
        raise RuntimeError(
            "tree_sitter_languages C parser is not executable: "
            f"{exc.__class__.__name__}: {exc}"
        ) from exc
    root = getattr(tree, "root_node", None)
    if root is None or bool(getattr(root, "has_error", True)):
        raise RuntimeError(
            "tree_sitter_languages C parser returned an invalid syntax tree"
        )
    return {
        "status": "available",
        "backend": "tree_sitter_c_structural_entities_v2",
    }


def parse_structural_entity_file(
    *,
    root: Path,
    relative_path: str,
    tree: str,
) -> dict[str, Any]:
    """Index source-owned entities without preprocessing the full kernel TU."""

    relative_path = normalize_relpath(relative_path)
    path = (root / relative_path).resolve()
    base = {
        "file": relative_path,
        "tree": tree,
        "definitions": [],
        "entities": [],
        "source_anchors": [],
        "references": [],
        "includes": [],
        "local_includes": [],
        "text_includes": [],
        "diagnostics": [],
        "error_nodes": [],
        "backend": "tree_sitter_c_structural_entities_v2",
    }
    if not path.is_file():
        return {**base, "exists": False, "parse_ok": False, "parse_error": "file_missing"}
    try:
        from tree_sitter_languages import get_parser
    except ImportError:
        return {
            **base,
            "exists": True,
            "parse_ok": False,
            "parse_error": "tree_sitter_c_unavailable",
        }
    source = path.read_bytes()
    cache_key = (
        str(root.resolve()),
        relative_path,
        tree,
        hashlib.sha256(source).hexdigest(),
    )
    cached_report = _structural_cache_get(cache_key)
    if cached_report is not None:
        return cached_report
    try:
        structural_parser = get_parser("c")
        parsed = structural_parser.parse(source)
    except Exception as exc:
        return {
            **base,
            "exists": True,
            "parse_ok": False,
            "parse_error": f"tree_sitter_parse_failed:{exc.__class__.__name__}",
            "parse_exception": str(exc),
        }

    def point_value(point: object, field: str, index: int) -> int:
        value = getattr(point, field, None)
        if value is not None:
            return int(value)
        return int(point[index])

    def node_text(node: Any | None) -> str:
        if node is None:
            return ""
        return source[int(node.start_byte) : int(node.end_byte)].decode("utf-8", errors="ignore").strip()

    def syntax_fingerprint(node: Any) -> str:
        """Hash C syntax tokens without formatting or comments."""

        tokens: list[str] = []
        pending = [node]
        while pending:
            current = pending.pop()
            if str(current.type) == "comment":
                continue
            children = list(getattr(current, "children", []) or [])
            if children:
                pending.extend(reversed(children))
                continue
            token = node_text(current)
            if token:
                tokens.append(token)
        return stable_id("structural-syntax", "\x1f".join(tokens))

    def descendants(node: Any):
        pending = list(reversed(node.named_children))
        while pending:
            child = pending.pop()
            yield child
            pending.extend(reversed(child.named_children))

    def macro_context(node: Any) -> tuple[tuple[str, int], ...]:
        conditionals: list[tuple[str, int]] = []
        parent = node.parent
        while parent is not None:
            if str(parent.type) in {
                "preproc_if", "preproc_ifdef", "preproc_ifndef",
                "preproc_elif", "preproc_else",
            }:
                conditionals.append(
                    (str(parent.type), int(parent.start_byte))
                )
            parent = parent.parent
        return tuple(reversed(conditionals))

    macro_definitions: dict[
        tuple[tuple[str, int], ...],
        dict[str, tuple[list[str], str]],
    ] = defaultdict(dict)
    macro_definition_offsets: dict[
        tuple[tuple[str, int], ...], dict[str, int]
    ] = defaultdict(dict)
    ambiguous_macros: dict[
        tuple[tuple[str, int], ...], set[str]
    ] = defaultdict(set)
    macro_counts: Counter[str] = Counter()
    undefined_macros: set[str] = set()
    for macro_node in descendants(parsed.root_node):
        macro_type = str(macro_node.type)
        if macro_type == "preproc_call":
            directive = next(
                (
                    node_text(child)
                    for child in macro_node.named_children
                    if str(child.type) == "preproc_directive"
                ),
                "",
            )
            argument = node_text(macro_node.child_by_field_name("argument"))
            if directive == "#undef" and IDENTIFIER_RE.fullmatch(argument):
                undefined_macros.add(argument)
            continue
        if macro_type not in {"preproc_def", "preproc_function_def"}:
            continue
        context = macro_context(macro_node)
        definitions = macro_definitions[context]
        name = node_text(macro_node.child_by_field_name("name"))
        macro_counts[name] += 1
        if macro_type != "preproc_function_def":
            continue
        parameters = node_text(macro_node.child_by_field_name("parameters"))
        value = node_text(macro_node.child_by_field_name("value"))
        if not name or name in ambiguous_macros[context]:
            continue
        if name in definitions:
            definitions.pop(name)
            ambiguous_macros[context].add(name)
            continue
        if not (parameters.startswith("(") and parameters.endswith(")")):
            continue
        formals = [
            "__VA_ARGS__" if item.strip() == "..." else item.strip()
            for item in parameters[1:-1].split(",")
        ]
        if all(IDENTIFIER_RE.fullmatch(formal) for formal in formals):
            definitions[name] = (formals, value)
            macro_definition_offsets[context][name] = int(
                macro_node.start_byte
            )

    def first_named(node: Any | None, kinds: set[str]) -> Any | None:
        if node is None:
            return None
        if str(node.type) in kinds:
            return node
        return next((child for child in descendants(node) if str(child.type) in kinds), None)

    def has_descendant(node: Any, kind: str) -> bool:
        return str(node.type) == kind or any(str(child.type) == kind for child in descendants(node))

    def top_level_variable_is_definition(node: Any) -> bool:
        """Apply C file-scope object-definition semantics to one declaration."""

        if has_descendant(node, "init_declarator"):
            return True
        storage_classes = {
            node_text(child)
            for child in node.named_children
            if str(child.type) == "storage_class_specifier"
        }
        return "extern" not in storage_classes

    entities: list[dict[str, Any]] = []
    source_anchors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    error_nodes: list[dict[str, Any]] = []
    structural_error_nodes: list[Any] = []

    def collect_error_nodes(node: Any) -> None:
        pending = [node]
        while pending:
            current = pending.pop()
            is_error = bool(
                getattr(current, "is_error", False)
            ) or str(current.type) == "ERROR"
            is_missing = bool(getattr(current, "is_missing", False))
            if is_error or is_missing:
                structural_error_nodes.append(current)
                start_row = point_value(current.start_point, "row", 0)
                end_row = point_value(current.end_point, "row", 0)
                error_nodes.append(
                    {
                        "node_type": str(current.type),
                        "is_error": is_error,
                        "is_missing": is_missing,
                        "extent": {
                            "start_line": start_row + 1,
                            "start_column": point_value(
                                current.start_point,
                                "column",
                                1,
                            )
                            + 1,
                            "end_line": end_row + 1,
                            "end_column": point_value(
                                current.end_point,
                                "column",
                                1,
                            )
                            + 1,
                        },
                    }
                )
            pending.extend(reversed(current.children))

    def is_gnu_named_variadic_parameter_error(
        error: Any,
        *,
        parameters: Any | None,
    ) -> bool:
        """Recognize Tree-sitter recovery for a GNU ``name...`` parameter."""

        if (
            parameters is None
            or str(error.type) != "ERROR"
            or bool(getattr(error, "is_missing", False))
            or int(error.start_byte) < int(parameters.start_byte)
            or int(error.end_byte) > int(parameters.end_byte)
            or source[int(error.start_byte) : int(error.end_byte)] != b"..."
        ):
            return False
        preceding_identifiers = [
            candidate
            for candidate in [parameters, *descendants(parameters)]
            if str(candidate.type) == "identifier"
            and int(candidate.end_byte) <= int(error.start_byte)
        ]
        if not preceding_identifiers:
            return False
        parameter_name = max(
            preceding_identifiers,
            key=lambda candidate: int(candidate.end_byte),
        )
        between = source[
            int(parameter_name.end_byte) : int(error.start_byte)
        ]
        after = source[int(error.end_byte) : int(parameters.end_byte)]
        return not between.strip() and bool(re.fullmatch(rb"\s*\)", after))

    def dependency_symbols(
        node: Any,
        *,
        own_symbol: str,
    ) -> tuple[list[str], list[str], dict[str, str], bool]:
        """Return syntax-owned external identifier uses for one entity.

        This is deliberately a structural inventory rather than name-based
        classification.  Identifiers introduced by the entity itself,
        parameters, local declarations, fields, and enumerators are removed
        from the use set.  Remaining names still require semantic resolution
        before they can be treated as dependency contracts.
        """

        declared = {own_symbol}
        node_type = str(node.type)
        parameters = None
        if node_type in {"preproc_def", "preproc_function_def"}:
            parameters = node.child_by_field_name("parameters")
            if parameters is not None:
                declared.update(IDENTIFIER_RE.findall(node_text(parameters)))
                if any(str(child.type) == "..." for child in parameters.children):
                    # The C99 ellipsis binds an implicit replacement-list argument.
                    declared.add("__VA_ARGS__")
            value = node.child_by_field_name("value")
            search_root = value or node
        else:
            search_root = node
        overlapping_errors = [
            error
            for error in structural_error_nodes
            if int(node.start_byte) < int(error.end_byte)
            and int(error.start_byte) < int(node.end_byte)
        ]
        recognized_variadic_errors = [
            error
            for error in overlapping_errors
            if node_type == "preproc_function_def"
            and is_gnu_named_variadic_parameter_error(
                error,
                parameters=parameters,
            )
        ]
        complete = len(recognized_variadic_errors) == len(overlapping_errors)
        if complete and recognized_variadic_errors:
            declared.update(
                macro_replacement_declaration_names(search_root)
            )
        if node_type not in {"preproc_def", "preproc_function_def"}:
            for candidate in [search_root, *descendants(search_root)]:
                if str(candidate.type) in {
                    "parameter_declaration",
                    "declaration",
                    "field_declaration",
                }:
                    declared.update(
                        node_text(name)
                        for name in declaration_names(candidate)
                        if node_text(name)
                    )
                elif str(candidate.type) == "enumerator":
                    name = candidate.child_by_field_name("name") or first_named(
                        candidate,
                        {"identifier"},
                    )
                    if name is not None and node_text(name):
                        declared.add(node_text(name))

        uses: set[str] = set()
        type_uses: set[str] = set()
        type_facets: dict[str, str] = {}
        body = (
            node.child_by_field_name("body")
            if node_type == "function_definition"
            else None
        )
        body_identifiers = {
            node_text(item)
            for item in descendants(body)
            if str(item.type) == "identifier"
        } if body is not None else set()
        has_function_declarator = first_named(
            node, {"function_declarator"}
        ) is not None
        if node_type in {"preproc_def", "preproc_function_def"}:
            uses.update(
                _c_identifiers_outside_comments_and_literals(
                    node_text(search_root)
                )
                - declared
            )
        structural_identifier_kinds = {
            "identifier",
            "type_identifier",
        }
        candidates = [search_root, *descendants(search_root)]
        for candidate in candidates:
            if str(candidate.type) not in structural_identifier_kinds:
                continue
            name = node_text(candidate)
            if name and name not in declared:
                uses.add(name)
                if str(candidate.type) == "type_identifier":
                    type_uses.add(name)
                    facet = "layout"
                    parent = candidate.parent
                    while parent is not None and parent != node:
                        if str(parent.type) == "parameter_declaration":
                            declarator = parent.child_by_field_name("declarator")
                            parameter_name = first_named(
                                declarator, {"identifier"}
                            )
                            if (
                                has_function_declarator
                                and declarator is not None
                                and str(declarator.type) == "pointer_declarator"
                                and int(candidate.end_byte)
                                <= int(declarator.start_byte)
                                and (
                                    parameter_name is None
                                    or node_text(parameter_name)
                                    not in body_identifiers
                                )
                            ):
                                facet = "identity"
                            break
                        parent = parent.parent
                    if type_facets.get(name) != "layout":
                        type_facets[name] = facet
        if node_type in {"preproc_def", "preproc_function_def"}:
            context = macro_context(node)
            token_only = _nested_macro_token_only_literals(
                node_text(search_root),
                {
                    name: definition
                    for name, definition in macro_definitions.get(
                        context, {}
                    ).items()
                    if macro_counts[name] == 1
                    and name not in undefined_macros
                    and macro_definition_offsets[context][name]
                    < int(node.start_byte)
                },
            )
            uses.difference_update(token_only)
            type_uses.difference_update(token_only)
            for name in token_only:
                type_facets.pop(name, None)
        return sorted(uses), sorted(type_uses), type_facets, complete

    def source_range(node: Any) -> dict[str, Any]:
        start_row = point_value(node.start_point, "row", 0)
        end_row = point_value(node.end_point, "row", 0)
        raw_end_column = point_value(node.end_point, "column", 1)
        start_line = start_row + 1
        start_column = point_value(node.start_point, "column", 1) + 1
        end_line = end_row + 1
        end_column = raw_end_column + 1
        ends_at_line_boundary = raw_end_column == 0 and end_row > start_row
        if ends_at_line_boundary:
            end_line = end_row
            end_column = 1
        return {
            "start_line": start_line,
            "start_column": start_column,
            "end_line": end_line,
            "end_column": end_column,
            "starts_at_line_start": start_column == 1,
            "ends_at_line_boundary": ends_at_line_boundary,
        }

    def add_entity(
        *,
        kind: str,
        symbol: str,
        node: Any,
        declarator_node: Any | None = None,
        container: tuple[str, str] | None = None,
        is_definition: bool = True,
        conditional_context: tuple[str, ...] = (),
    ) -> None:
        symbol = symbol.strip()
        if not symbol:
            return
        contract = contract_key(
            kind,
            symbol,
            container=(
                {"container_kind": container[0], "container_name": container[1]}
                if container
                else None
            ),
        )
        source_extent = source_range(node)
        start_line = int(source_extent["start_line"])
        start_column = int(source_extent["start_column"])
        identity = (kind, contract, start_line, start_column)
        if identity in seen:
            return
        seen.add(identity)
        (
            external_symbols,
            external_type_symbols,
            dependency_type_facets,
            dependency_analysis_complete,
        ) = dependency_symbols(node, own_symbol=symbol)
        storage_classes = {
            node_text(candidate).strip().upper()
            for candidate in [
                node,
                *list(getattr(node, "named_children", []) or []),
            ]
            if str(candidate.type) == "storage_class_specifier"
            and node_text(candidate).strip()
        }
        storage_class = (
            "STATIC"
            if "STATIC" in storage_classes
            else next(iter(sorted(storage_classes)), "")
        )
        entity_id = stable_id(
            "structural-entity",
            tree,
            relative_path,
            contract,
            str(start_line),
            str(start_column),
        )
        source_fingerprint = stable_id("structural-source", node_text(node))
        contract_declarator = (
            declarator_node
            if kind == "function" and declarator_node is not None
            else first_named(node, {"function_declarator"})
            if kind == "function"
            else None
        )
        body = (
            node.child_by_field_name("body")
            if kind == "function" and str(node.type) == "function_definition"
            else None
        )
        body_start = source_range(body) if body is not None else None
        declaration_extent = (
            {
                "start_line": start_line,
                "start_column": start_column,
                "end_line": body_start["start_line"],
                "end_column": body_start["start_column"],
            }
            if body_start is not None
            else None
        )
        entities.append(
            {
                "id": entity_id,
                "entity_id": entity_id,
                "tree": tree,
                "key": contract,
                "contract_key": contract,
                "symbol": symbol,
                "name": symbol,
                "kind": kind,
                "cursor_kind": f"TREE_SITTER_{str(node.type).upper()}",
                "is_definition": is_definition,
                "file": relative_path,
                "container": (
                    {"container_kind": container[0], "container_name": container[1]}
                    if container
                    else None
                ),
                "conditional_depth": len(conditional_context),
                "conditional_context": list(conditional_context),
                "starts_at_line_start": source_extent[
                    "starts_at_line_start"
                ],
                "ends_at_line_boundary": source_extent[
                    "ends_at_line_boundary"
                ],
                "extent": {
                    "start_line": start_line,
                    "start_column": start_column,
                    "end_line": source_extent["end_line"],
                    "end_column": source_extent["end_column"],
                },
                **(
                    {"declaration_extent": declaration_extent}
                    if declaration_extent is not None
                    else {}
                ),
                "backend": "tree_sitter_c_structural_entities_v3",
                "source_fingerprint": source_fingerprint,
                "syntax_fingerprint": syntax_fingerprint(node),
                "declarator_fingerprint": (
                    syntax_fingerprint(contract_declarator)
                    if contract_declarator is not None
                    else ""
                ),
                "storage_class": storage_class,
                "linkage": (
                    "INTERNAL" if storage_class == "STATIC" else ""
                ),
                "dependency_symbols": external_symbols,
                "dependency_type_symbols": external_type_symbols,
                "dependency_type_facets": dependency_type_facets,
                "dependency_analysis_complete": dependency_analysis_complete,
            }
        )

    def add_source_anchor(
        *,
        kind: str,
        symbol: str,
        node: Any,
        identity_symbol: str = "",
        conditional_context: tuple[str, ...] = (),
    ) -> None:
        """Record syntax-owned placement evidence outside the code graph."""

        symbol = symbol.strip()
        if not symbol:
            return
        source_extent = source_range(node)
        source_identifiers = sorted(
            {
                node_text(candidate)
                for candidate in [node, *descendants(node)]
                if str(candidate.type)
                in {"identifier", "type_identifier", "field_identifier"}
                and node_text(candidate)
            }
        )
        start_line = int(source_extent["start_line"])
        start_column = int(source_extent["start_column"])
        contract = contract_key(kind, identity_symbol or symbol)
        anchor_id = stable_id(
            "structural-source-anchor",
            tree,
            relative_path,
            contract,
            str(start_line),
            str(start_column),
        )
        source_anchors.append(
            {
                "id": anchor_id,
                "entity_id": anchor_id,
                "tree": tree,
                "key": contract,
                "contract_key": contract,
                "symbol": symbol,
                "name": symbol,
                "kind": kind,
                "cursor_kind": f"TREE_SITTER_{str(node.type).upper()}",
                "is_definition": False,
                "is_source_anchor": True,
                "file": relative_path,
                "container": None,
                "conditional_depth": len(conditional_context),
                "conditional_context": list(conditional_context),
                "starts_at_line_start": source_extent[
                    "starts_at_line_start"
                ],
                "ends_at_line_boundary": source_extent[
                    "ends_at_line_boundary"
                ],
                "extent": {
                    "start_line": start_line,
                    "start_column": start_column,
                    "end_line": source_extent["end_line"],
                    "end_column": source_extent["end_column"],
                },
                "backend": "tree_sitter_c_structural_source_anchor_v1",
                "source_fingerprint": stable_id(
                    "structural-source-anchor-source",
                    node_text(node),
                ),
                "syntax_fingerprint": syntax_fingerprint(node),
                "source_identifiers": source_identifiers,
            }
        )

    def declaration_names(
        node: Any,
        *,
        content: bytes = source,
    ) -> list[Any]:
        allowed_names = (
            {"field_identifier"}
            if str(node.type) == "field_declaration"
            else {"identifier", "field_identifier", "type_identifier"}
        )
        declarators: list[Any] = []
        recovery_declarators = [
            child
            for child in node.named_children
            if str(child.type) == "ERROR"
            and first_named(child, allowed_names) is not None
        ]
        children_by_field_name = getattr(
            node,
            "children_by_field_name",
            None,
        )
        if callable(children_by_field_name):
            declarators.extend(children_by_field_name("declarator"))
        if not declarators:
            declarator = node.child_by_field_name("declarator")
            if declarator is not None:
                declarators = [declarator]
        if not declarators:
            field_name_for_child = getattr(
                node,
                "field_name_for_child",
                None,
            )
            if callable(field_name_for_child):
                for index, child in enumerate(node.children):
                    if field_name_for_child(index) == "declarator":
                        declarators.append(child)
        declarators = [*recovery_declarators, *declarators]
        candidates_with_roots: list[tuple[Any, Any]] = []
        seen_declarators: set[tuple[int, int]] = set()
        seen_positions: set[tuple[int, int]] = set()
        for declarator in declarators:
            declarator_position = (
                int(declarator.start_byte),
                int(declarator.end_byte),
            )
            if declarator_position in seen_declarators:
                continue
            seen_declarators.add(declarator_position)
            name = first_named(declarator, allowed_names)
            if name is None:
                continue
            position = (int(name.start_byte), int(name.end_byte))
            if position in seen_positions:
                continue
            seen_positions.add(position)
            candidates_with_roots.append((declarator, name))
        candidates_with_roots.sort(
            key=lambda item: (
                int(item[0].start_byte),
                int(item[0].end_byte),
            )
        )
        selected: list[tuple[Any, Any]] = []
        for declarator, name in candidates_with_roots:
            if selected:
                previous = selected[-1][0]
                between = content[
                    int(previous.end_byte) : int(declarator.start_byte)
                ]
                if (
                    int(declarator.start_byte) >= int(previous.end_byte)
                    and b"," not in between
                ):
                    # Error recovery can parse a trailing source annotation as
                    # another declarator. Without a declaration separator it
                    # belongs to the preceding recovered declarator.
                    continue
            selected.append((declarator, name))
        return [name for _, name in selected]

    def macro_replacement_declaration_names(node: Any) -> set[str]:
        """Parse one GNU macro replacement list as an isolated C statement."""

        replacement = source[int(node.start_byte) : int(node.end_byte)]
        replacement = re.sub(rb"\\\r?\n", b"\n", replacement)
        prefix = b"void driver_backport_macro_body(void) {\n"
        wrapped = prefix + replacement + b";\n}\n"
        parsed_replacement = structural_parser.parse(wrapped)
        if parsed_replacement.root_node.has_error:
            return set()

        def wrapped_text(candidate: Any) -> str:
            return wrapped[
                int(candidate.start_byte) : int(candidate.end_byte)
            ].decode("utf-8", errors="ignore").strip()

        names: set[str] = set()
        for candidate in [
            parsed_replacement.root_node,
            *descendants(parsed_replacement.root_node),
        ]:
            if str(candidate.type) in {
                "declaration",
                "field_declaration",
            }:
                names.update(
                    wrapped_text(name)
                    for name in declaration_names(candidate, content=wrapped)
                    if wrapped_text(name)
                )
            elif str(candidate.type) == "enumerator":
                name = candidate.child_by_field_name("name") or first_named(
                    candidate,
                    {"identifier"},
                )
                if name is not None and wrapped_text(name):
                    names.add(wrapped_text(name))
        return names

    def function_declaration_entities(
        node: Any,
    ) -> list[tuple[Any, Any]]:
        """Bind an annotated function declaration to its real declarator."""

        declarators: list[Any] = []
        children_by_field_name = getattr(
            node,
            "children_by_field_name",
            None,
        )
        if callable(children_by_field_name):
            declarators.extend(children_by_field_name("declarator"))
        if not declarators:
            declarator = node.child_by_field_name("declarator")
            if declarator is not None:
                declarators.append(declarator)
        result: list[tuple[Any, Any]] = []
        seen_positions: set[tuple[int, int]] = set()
        for declarator in declarators:
            function_declarator = first_named(
                declarator,
                {"function_declarator"},
            )
            if function_declarator is None:
                continue
            name_root = function_declarator.child_by_field_name(
                "declarator"
            )
            name = first_named(name_root, {"identifier"})
            if name is None:
                continue
            position = (int(name.start_byte), int(name.end_byte))
            if position in seen_positions:
                continue
            seen_positions.add(position)
            result.append((name, function_declarator))
        return result

    def walk(
        node: Any,
        *,
        container: tuple[str, str] | None = None,
        inside_function: bool = False,
        parent_type: str = "",
        conditional_context: tuple[str, ...] = (),
    ) -> tuple[tuple[str, str] | None, bool, tuple[str, ...]]:
        node_type = str(node.type)
        next_container = container
        next_inside_function = inside_function
        conditional_node_types = {
            "preproc_if",
            "preproc_ifdef",
            "preproc_ifndef",
            "preproc_elif",
            "preproc_else",
        }
        conditional_marker = ""
        if node_type in conditional_node_types:
            first_line = node_text(node).splitlines()[0] if node_text(node) else ""
            conditional_marker = " ".join(first_line.split())
        next_conditional_context = (
            (*conditional_context, f"{node_type}:{conditional_marker}")
            if conditional_marker
            else conditional_context
        )

        if node_type in {"preproc_def", "preproc_function_def"}:
            name = node.child_by_field_name("name")
            add_entity(
                kind="macro",
                symbol=node_text(name),
                node=node,
                container=container,
                conditional_context=conditional_context,
            )
        elif node_type == "preproc_include":
            path_node = (
                node.child_by_field_name("path")
                or first_named(
                    node,
                    {"string_literal", "system_lib_string"},
                )
            )
            add_source_anchor(
                kind="include",
                symbol=node_text(path_node),
                node=node,
                conditional_context=conditional_context,
            )
        elif (
            node_type == "expression_statement"
            and not inside_function
            and not node.has_error
        ):
            call = first_named(node, {"call_expression"})
            callee = (
                call.child_by_field_name("function")
                if call is not None
                else None
            )
            if call is not None and callee is None:
                callee = first_named(
                    call,
                    {
                        "identifier",
                        "field_expression",
                        "parenthesized_expression",
                    },
                )
            call_syntax = syntax_fingerprint(node) if call is not None else ""
            add_source_anchor(
                kind="top_level_call_expression",
                symbol=node_text(callee),
                identity_symbol=call_syntax,
                node=node,
                conditional_context=conditional_context,
            )
        elif node_type in {"struct_specifier", "union_specifier", "enum_specifier"}:
            name = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            symbol = node_text(name)
            container_kind = {
                "struct_specifier": "struct",
                "union_specifier": "union",
                "enum_specifier": "enum",
            }[node_type]
            if symbol:
                add_entity(
                    kind="type",
                    symbol=symbol,
                    node=node,
                    is_definition=body is not None,
                    conditional_context=conditional_context,
                )
                if body is not None:
                    next_container = (container_kind, symbol)
        elif node_type == "type_definition":
            for name in declaration_names(node):
                add_entity(
                    kind="type",
                    symbol=node_text(name),
                    node=node,
                    conditional_context=conditional_context,
                )
        elif node_type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            name = first_named(declarator, {"identifier"})
            add_entity(
                kind="function",
                symbol=node_text(name),
                node=node,
                declarator_node=first_named(
                    declarator,
                    {"function_declarator"},
                ),
                conditional_context=conditional_context,
            )
            next_inside_function = True
        elif node_type == "declaration" and not inside_function and parent_type != "type_definition":
            macro_type = first_named(node, {"macro_type_specifier"})
            if macro_type is not None:
                macro_name = first_named(macro_type, {"identifier"})
                add_source_anchor(
                    kind="top_level_macro_declaration",
                    symbol=node_text(macro_name),
                    identity_symbol=syntax_fingerprint(node),
                    node=node,
                    conditional_context=conditional_context,
                )
            kind = "function" if has_descendant(node, "function_declarator") else "variable"
            declarations = (
                function_declaration_entities(node)
                if kind == "function"
                else [(name, None) for name in declaration_names(node)]
            )
            for name, declarator in declarations:
                add_entity(
                    kind=kind,
                    symbol=node_text(name),
                    node=node,
                    declarator_node=declarator,
                    is_definition=(
                        False
                        if kind == "function"
                        else top_level_variable_is_definition(node)
                    ),
                    conditional_context=conditional_context,
                )
        elif node_type == "field_declaration" and container:
            for name in declaration_names(node):
                add_entity(
                    kind="field",
                    symbol=node_text(name),
                    node=node,
                    container=container,
                    conditional_context=conditional_context,
                )
        elif node_type == "enumerator":
            name = node.child_by_field_name("name") or first_named(node, {"identifier"})
            add_entity(
                kind="enum_constant",
                symbol=node_text(name),
                node=node,
                container=container,
                conditional_context=conditional_context,
            )

        return (
            next_container,
            next_inside_function,
            next_conditional_context,
        )

    collect_error_nodes(parsed.root_node)
    pending_walk: list[
        tuple[
            Any,
            tuple[str, str] | None,
            bool,
            str,
            tuple[str, ...],
        ]
    ] = [(parsed.root_node, None, False, "", ())]
    while pending_walk:
        (
            node,
            container,
            inside_function,
            parent_type,
            conditional_context,
        ) = pending_walk.pop()
        (
            next_container,
            next_inside_function,
            next_conditional_context,
        ) = walk(
            node,
            container=container,
            inside_function=inside_function,
            parent_type=parent_type,
            conditional_context=conditional_context,
        )
        pending_walk.extend(
            (
                child,
                next_container,
                next_inside_function,
                str(node.type),
                next_conditional_context,
            )
            for child in reversed(node.named_children)
        )
    entities.sort(
        key=lambda item: (
            int((item.get("extent") or {}).get("start_line") or 0),
            int((item.get("extent") or {}).get("start_column") or 0),
            str(item.get("contract_key") or ""),
        )
    )
    source_anchors.sort(
        key=lambda item: (
            int((item.get("extent") or {}).get("start_line") or 0),
            int((item.get("extent") or {}).get("start_column") or 0),
            str(item.get("contract_key") or ""),
        )
    )
    report = {
        **base,
        "exists": True,
        "parse_ok": True,
        "parse_error": None,
        "definitions": entities,
        "entities": entities,
        "source_anchors": source_anchors,
        "has_error_nodes": bool(parsed.root_node.has_error),
        "error_nodes": error_nodes,
    }
    _structural_cache_put(cache_key, report)
    return report


_STRUCTURAL_C_TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"'
    r"|'(?:\\.|[^'\\])*'"
    r"|[A-Za-z_][A-Za-z0-9_]*"
    r"|0[xX][0-9A-Fa-f]+"
    r"|\d+(?:\.\d+)?"
    r"|==|!=|<=|>=|->|<<|>>|\+\+|--|&&|\|\|"
    r"|[^\s]"
)


def structural_rename_invariant_fingerprint(
    *,
    root: Path,
    relative_path: str,
    entity: dict[str, Any],
) -> str | None:
    """Fingerprint one structural entity while abstracting its own identifier."""

    source_path = root / normalize_relpath(relative_path)
    extent = dict(entity.get("extent") or {})
    start_line = int(extent.get("start_line") or 0)
    end_line = int(extent.get("end_line") or start_line)
    start_column = int(extent.get("start_column") or 1)
    end_column = int(extent.get("end_column") or 0)
    if (
        not source_path.is_file()
        or start_line <= 0
        or end_line < start_line
    ):
        return None
    lines = source_path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines(keepends=True)
    if end_line > len(lines):
        return None
    selected = list(lines[start_line - 1 : end_line])
    if not selected:
        return None
    selected[0] = selected[0][max(0, start_column - 1) :]
    if end_column > 0:
        selected[-1] = selected[-1][: max(0, end_column - 1)]
    tokens = _STRUCTURAL_C_TOKEN_RE.findall("".join(selected))
    symbol = str(entity.get("symbol") or entity.get("name") or "")
    normalized = [
        "<entity-name>" if token == symbol else token
        for token in tokens
    ]
    kind = str(entity.get("kind") or "")
    if kind == "function" and "{" in normalized:
        body_start = normalized.index("{")
        declaration = [
            token
            for token in normalized[:body_start]
            if token not in {"static", "extern", "inline"}
        ]
        normalized = [*declaration, *normalized[body_start:]]
    container = (
        dict(entity.get("container") or {})
        if isinstance(entity.get("container"), dict)
        else {}
    )
    return stable_id(
        "structural-rename-invariant",
        kind,
        str(container.get("container_kind") or ""),
        str(container.get("container_name") or ""),
        "\x1f".join(normalized),
    )


def _path_stat_signature(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size)


def _parser_errors_cover_identifiers(
    source_lines: list[str],
    error_nodes: Iterable[dict[str, Any]],
    symbols: Iterable[str],
) -> bool:
    """Return whether an exact identifier occurs inside a parser-error span."""

    wanted = {str(symbol) for symbol in symbols if str(symbol)}
    if not wanted:
        return False
    for error in error_nodes:
        if not isinstance(error, dict):
            continue
        extent = error.get("extent") or {}
        start = max(1, int(extent.get("start_line") or 0))
        end = min(len(source_lines), int(extent.get("end_line") or 0))
        for line_number in range(start, end + 1):
            if wanted.intersection(
                IDENTIFIER_RE.findall(source_lines[line_number - 1])
            ):
                return True
    return False


def _donor_parse_cache_key(
    *,
    context: ClangCompileContext | None,
    root: Path,
    driver_root: Path,
    relative_path: str,
    translation_unit_relative_path: str,
    tree: str,
    detailed_processing: bool,
    exhaustive_diagnostics: bool,
    reference_target_entity_ids: set[str] | None,
) -> tuple[object, ...] | None:
    if (
        not tree.startswith("donor_")
        or context is None
        or _DONOR_PARSE_CACHE_MAX_RECORDS <= 0
    ):
        return None
    source_signature = _path_stat_signature(root / relative_path)
    translation_unit_signature = _path_stat_signature(
        root / translation_unit_relative_path
    )
    compile_commands_signature = _path_stat_signature(
        context.compile_commands_path
    )
    if (
        source_signature is None
        or translation_unit_signature is None
        or compile_commands_signature is None
    ):
        return None
    return (
        str(root.resolve()),
        str(driver_root.resolve()),
        tree,
        relative_path,
        translation_unit_relative_path,
        bool(detailed_processing),
        bool(exhaustive_diagnostics),
        tuple(sorted(reference_target_entity_ids or set())),
        source_signature,
        translation_unit_signature,
        compile_commands_signature,
        tuple(tuple(item) for item in context.path_maps),
    )


def _donor_parse_cache_get(
    key: tuple[object, ...] | None,
) -> dict[str, Any] | None:
    if key is None:
        return None
    cached = _DONOR_PARSE_CACHE.get(key)
    if cached is None:
        return None
    _DONOR_PARSE_CACHE.move_to_end(key)
    report, _weight = cached
    return {
        **report,
        "parse_seconds": 0.0,
        "cache": "process_donor_ast_lru",
    }


def _donor_parse_cache_put(
    key: tuple[object, ...] | None,
    report: dict[str, Any],
) -> None:
    global _DONOR_PARSE_CACHE_RECORDS
    if key is None:
        return
    weight = max(
        1,
        len(report.get("entities") or [])
        + len(report.get("references") or [])
        + len(report.get("includes") or []),
    )
    previous = _DONOR_PARSE_CACHE.pop(key, None)
    if previous is not None:
        _DONOR_PARSE_CACHE_RECORDS -= previous[1]
    _DONOR_PARSE_CACHE[key] = (report, weight)
    _DONOR_PARSE_CACHE_RECORDS += weight
    while (
        _DONOR_PARSE_CACHE
        and _DONOR_PARSE_CACHE_RECORDS > _DONOR_PARSE_CACHE_MAX_RECORDS
    ):
        _old_key, (_old_report, old_weight) = _DONOR_PARSE_CACHE.popitem(
            last=False
        )
        _DONOR_PARSE_CACHE_RECORDS -= old_weight


def parse_source_file(
    context: ClangCompileContext | None,
    root: Path,
    driver_root: Path,
    relative_path: str,
    *,
    detailed_processing: bool,
    exhaustive_diagnostics: bool = False,
    tree: str = "unknown",
    reference_target_entity_ids: set[str] | None = None,
    translation_unit_relative_path: str | None = None,
) -> dict[str, Any]:
    """Parse one translation unit into the canonical CodeIndex record format."""

    relative_path = normalize_relpath(relative_path)
    translation_unit_relative_path = normalize_relpath(
        translation_unit_relative_path or relative_path
    )
    path = (root / relative_path).resolve()
    translation_unit_path = (root / translation_unit_relative_path).resolve()
    includes = parse_include_directives(
        root=root.resolve(),
        driver_root=driver_root.resolve(),
        relative_path=relative_path,
        tree=tree,
    )
    base = {
        "file": relative_path,
        "tree": tree,
        "translation_unit_file": translation_unit_relative_path,
        "definitions": [],
        "entities": [],
        "references": [],
        "includes": includes,
        "local_includes": [
            {"from": relative_path, "to": item["target_file"], "line": item["line"]}
            for item in includes
            if item.get("target_file")
        ],
        "text_includes": sorted({str(item["name"]) for item in includes}),
        "diagnostics": [],
    }
    if not path.is_file():
        return {**base, "exists": False, "parse_ok": False, "parse_error": "file_missing"}
    if not translation_unit_path.is_file():
        return {
            **base,
            "exists": True,
            "parse_ok": False,
            "parse_error": "translation_unit_missing",
        }
    if not _source_like(relative_path):
        return {**base, "exists": True, "parse_ok": False, "parse_error": "not_source_like"}
    if context is None:
        return {**base, "exists": True, "parse_ok": False, "parse_error": "missing_clang_context"}
    cache_key = _donor_parse_cache_key(
        context=context,
        root=root,
        driver_root=driver_root,
        relative_path=relative_path,
        translation_unit_relative_path=translation_unit_relative_path,
        tree=tree,
        detailed_processing=detailed_processing,
        exhaustive_diagnostics=exhaustive_diagnostics,
        reference_target_entity_ids=reference_target_entity_ids,
    )
    cached_report = _donor_parse_cache_get(cache_key)
    if cached_report is not None:
        return cached_report
    try:
        parsed = context.parse(
            translation_unit_path,
            detailed_processing=detailed_processing,
            skip_function_bodies=False,
            exhaustive_diagnostics=exhaustive_diagnostics,
        )
    except Exception as exc:
        return {
            **base,
            "exists": True,
            "parse_ok": False,
            "parse_error": f"parse_failed:{exc.__class__.__name__}",
            "parse_exception": str(exc),
        }

    root_resolved = root.resolve()
    driver_root_resolved = driver_root.resolve()
    focus_file = path.resolve()
    entities: list[dict[str, Any]] = []
    ownership_entities: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    seen_references: set[str] = set()
    entity_cache: dict[tuple[str, ...], dict[str, Any] | None] = {}
    file_path_cache: dict[str, Path | None] = {
        str(focus_file): focus_file,
    }
    try:
        source_lines = path.read_bytes().splitlines()
    except OSError:
        source_lines = []
    for (
        cursor,
        implementation_owner,
        cursor_kind,
        ast_path,
    ) in _walk_main_file_cursors(
        parsed.translation_unit.cursor,
        focus_file,
        file_path_cache=file_path_cache,
    ):
        ownership_entity = (
            _entity_record(
                cursor,
                root_resolved,
                tree,
                cache=entity_cache,
                file_path_cache=file_path_cache,
                cursor_kind_hint=cursor_kind,
                file_path_hint=focus_file,
            )
            if cursor_kind in OWNER_CURSOR_KINDS
            else None
        )
        if ownership_entity is not None:
            ownership_entities.append(ownership_entity)
        if reference_target_entity_ids is None:
            entity = ownership_entity or _entity_record(
                cursor,
                root_resolved,
                tree,
                cache=entity_cache,
                file_path_cache=file_path_cache,
                cursor_kind_hint=cursor_kind,
                file_path_hint=focus_file,
            )
            if entity is not None:
                entities.append(entity)
        reference = _reference_record(
            cursor,
            root=root_resolved,
            driver_root=driver_root_resolved,
            tree=tree,
            owner_cursor=implementation_owner,
            entity_cache=entity_cache,
            reference_target_entity_ids=reference_target_entity_ids,
            source_lines=source_lines,
            file_path_cache=file_path_cache,
            cursor_kind_hint=cursor_kind,
            source_file_hint=focus_file,
            source_relative_hint=relative_path,
            ast_path=ast_path,
        )
        if reference is not None and reference["id"] not in seen_references:
            reference["translation_unit_file"] = translation_unit_relative_path
            seen_references.add(reference["id"])
            references.append(reference)

    ownership_entities = canonical_entity_records(ownership_entities)
    entities = canonical_entity_records(entities)
    _resolve_reference_ownership(
        entities=ownership_entities,
        references=references,
    )
    entities.sort(
        key=lambda item: (
            str(item.get("file") or ""),
            int((item.get("extent") or {}).get("start_line") or 0),
            int((item.get("extent") or {}).get("start_column") or 0),
            str(item.get("id") or ""),
        )
    )
    references.sort(
        key=lambda item: (
            str(item.get("source_file") or ""),
            int(item.get("line") or 0),
            int(item.get("column") or 0),
            str(item.get("id") or ""),
        )
    )
    report = {
        **base,
        "exists": True,
        "parse_ok": True,
        "parse_error": None,
        "diagnostics": parsed.diagnostics,
        "error_diagnostic_count": parse_error_diagnostic_count(
            parsed.diagnostics
        ),
        "compile_context_origin": parsed.compile_context_origin,
        "definitions": entities,
        "entities": entities,
        "references": references,
        "parse_seconds": parsed.parse_seconds,
    }
    _donor_parse_cache_put(cache_key, report)
    return report


@dataclass(frozen=True)
class GraphBudget:
    max_nodes: int = 20_000
    max_edges: int = 60_000


class TypedCodeGraph:
    def __init__(self, *, budget: GraphBudget | None = None) -> None:
        self.budget = budget or GraphBudget()
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self._edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._budget_exempt_node_ids: set[str] = set()
        self._budget_exempt_edge_keys: set[tuple[str, str, str]] = set()
        self.truncated = False
        self.truncation_reasons: set[str] = set()

    def add_node(
        self,
        graph_node_id: str,
        kind: str,
        *,
        budget_exempt: bool = False,
        **payload: Any,
    ) -> bool:
        if budget_exempt:
            payload.setdefault("overlay_class", "diagnostic_causal")
        if graph_node_id in self.nodes:
            existing = self.nodes[graph_node_id]
            incoming = {key: value for key, value in payload.items() if value not in (None, "", [])}
            if "is_definition" in existing or "is_definition" in incoming:
                if entity_record_rank(incoming) > entity_record_rank(existing):
                    existing.update(incoming)
                    existing["kind"] = kind
                else:
                    for key, value in incoming.items():
                        if existing.get(key) in (None, "", []):
                            existing[key] = value
            else:
                existing.update(incoming)
            if budget_exempt:
                self._budget_exempt_node_ids.add(graph_node_id)
                existing["budget_exempt"] = True
            existing["id"] = graph_node_id
            existing["kind"] = kind
            return True
        budgeted_node_count = len(self.nodes) - len(self._budget_exempt_node_ids)
        if budgeted_node_count >= self.budget.max_nodes and not budget_exempt:
            self.truncated = True
            self.truncation_reasons.add("node_budget_exhausted")
            return False
        self.nodes[graph_node_id] = {
            **payload,
            "id": graph_node_id,
            "kind": kind,
            **({"budget_exempt": True} if budget_exempt else {}),
        }
        if budget_exempt:
            self._budget_exempt_node_ids.add(graph_node_id)
        return True

    def add_edge(
        self,
        graph_source: str,
        graph_target: str,
        graph_edge_type: str,
        *,
        budget_exempt: bool = False,
        **payload: Any,
    ) -> bool:
        if budget_exempt:
            payload.setdefault("overlay_class", "diagnostic_causal")
        key = (graph_source, graph_target, graph_edge_type)
        if key in self._edge_keys:
            if budget_exempt:
                existing = self._edges_by_key.get(key)
                if existing is not None:
                    existing.update(payload)
                    existing["from"] = graph_source
                    existing["to"] = graph_target
                    existing["type"] = graph_edge_type
                    existing["budget_exempt"] = True
                self._budget_exempt_edge_keys.add(key)
            return True
        budgeted_edge_count = len(self.edges) - len(self._budget_exempt_edge_keys)
        if budgeted_edge_count >= self.budget.max_edges and not budget_exempt:
            self.truncated = True
            self.truncation_reasons.add("edge_budget_exhausted")
            return False
        self._edge_keys.add(key)
        edge = {
            **payload,
            "from": graph_source,
            "to": graph_target,
            "type": graph_edge_type,
            **({"budget_exempt": True} if budget_exempt else {}),
        }
        self.edges.append(edge)
        self._edges_by_key[key] = edge
        if budget_exempt:
            self._budget_exempt_edge_keys.add(key)
        return True

    def to_dict(self) -> dict[str, Any]:
        overlay_node_counts: dict[str, int] = defaultdict(int)
        overlay_edge_counts: dict[str, int] = defaultdict(int)
        for node_id in self._budget_exempt_node_ids:
            overlay_node_counts[
                str((self.nodes.get(node_id) or {}).get("overlay_class") or "diagnostic_causal")
            ] += 1
        for edge in self.edges:
            key = (str(edge.get("from") or ""), str(edge.get("to") or ""), str(edge.get("type") or ""))
            if key in self._budget_exempt_edge_keys:
                overlay_edge_counts[str(edge.get("overlay_class") or "diagnostic_causal")] += 1
        return {
            "schema": "typed_code_dependency_graph_v6",
            "nodes": list(self.nodes.values()),
            "edges": list(self.edges),
            "summary": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "budgeted_node_count": len(self.nodes) - len(self._budget_exempt_node_ids),
                "budgeted_edge_count": len(self.edges) - len(self._budget_exempt_edge_keys),
                "actionable_overlay_node_count": len(self._budget_exempt_node_ids),
                "actionable_overlay_edge_count": len(self._budget_exempt_edge_keys),
                "diagnostic_overlay_node_count": overlay_node_counts.get("diagnostic_causal", 0),
                "diagnostic_overlay_edge_count": overlay_edge_counts.get("diagnostic_causal", 0),
                "query_overlay_node_count": overlay_node_counts.get("query_resolution", 0),
                "query_overlay_edge_count": overlay_edge_counts.get("query_resolution", 0),
                "overlay_node_counts": dict(sorted(overlay_node_counts.items())),
                "overlay_edge_counts": dict(sorted(overlay_edge_counts.items())),
                "truncated": self.truncated,
                "truncation_reasons": sorted(self.truncation_reasons),
            },
        }


def _diagnostic_mapping_result(
    status: str,
    *,
    root_node_ids: Iterable[str] = (),
    mapped_node_ids: Iterable[str] = (),
    evidence: Any = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "root_node_ids": list(root_node_ids),
        "mapped_node_ids": list(mapped_node_ids),
        "evidence": evidence,
    }


class CodeIndex:
    """Incremental old/donor typed code index used by scope and repair planning."""

    def __init__(
        self,
        spec: object | None = None,
        *,
        budget: GraphBudget | None = None,
        graph_projection: str = "occurrence",
    ) -> None:
        if graph_projection not in {"occurrence", "records"}:
            raise ValueError(
                "graph_projection must be 'occurrence' or 'records'"
            )
        self.spec = spec
        self.graph = TypedCodeGraph(budget=budget)
        self.graph_projection = graph_projection
        self.contexts: dict[str, ClangCompileContext | None] = {}
        self.context_reports: dict[str, dict[str, Any]] = {}
        self.parse_reports: dict[tuple[str, str], dict[str, Any]] = {}
        self.focused_parse_reports: dict[
            tuple[str, str, str],
            dict[str, Any],
        ] = {}
        self.reference_probe_reports: dict[
            tuple[str, str, tuple[str, ...]],
            dict[str, Any],
        ] = {}
        self._identifier_tokens_by_file: dict[
            tuple[str, str],
            tuple[tuple[str, int, int], frozenset[str]],
        ] = {}
        self._compile_dependency_frontier_cache: dict[
            tuple[str, str],
            dict[str, Any],
        ] = {}
        self.structural_reports: dict[tuple[str, str], dict[str, Any]] = {}
        self.entities_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.entities_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.entities_by_file: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.references_by_file: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.includes_by_file: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.semantic_contract_nodes_by_key: dict[str, set[str]] = defaultdict(set)
        self.semantic_contract_nodes_by_entity_id: dict[str, set[str]] = defaultdict(set)
        self.diagnostic_roots: dict[str, set[str]] = defaultdict(set)
        self.diagnostic_mappings: dict[str, dict[str, Any]] = {}
        self.semantic_transaction_nodes_by_patch_id: dict[str, str] = {}
        self._init_contexts()

    @staticmethod
    def _exact_contract_keys(record: dict[str, Any]) -> set[str]:
        return {
            str(value)
            for value in (record.get("contract_key"), record.get("key"))
            if str(value or "")
        }

    def _link_entity_to_semantic_contracts(
        self,
        entity_id: str,
        contract_keys: Iterable[object],
    ) -> None:
        if not entity_id or entity_id not in self.graph.nodes:
            return
        contract_ids = set(
            self.semantic_contract_nodes_by_entity_id.get(entity_id, set())
        )
        if not contract_ids:
            for contract_key in {
                str(item) for item in contract_keys if str(item)
            }:
                keyed_nodes = self.semantic_contract_nodes_by_key.get(
                    contract_key,
                    set(),
                )
                if len(keyed_nodes) == 1:
                    contract_ids.update(keyed_nodes)
        for contract_id in sorted(contract_ids):
            self.graph.add_edge(
                entity_id,
                contract_id,
                "instance_of_contract",
                budget_exempt=True,
                overlay_class="semantic_ledger",
            )

    def add_semantic_gap_ledger(
        self,
        state: dict[str, Any],
        *,
        contract_evidence_by_ref: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Overlay exact semantic contracts without rebuilding the program graph."""

        retained_entities: dict[str, dict[str, Any]] = {}
        added_gap_ids: set[str] = set()
        resolved_evidence_refs: set[str] = set()
        unresolved_evidence_refs: set[str] = set()
        external_evidence = contract_evidence_by_ref or {}
        active_batch_id, active_batch = _active_semantic_graph_batch(state)
        graph_gap_ids = semantic_graph_gap_ids(state)
        for gap_id, raw_gap in sorted(dict(state.get("gaps") or {}).items()):
            if not isinstance(raw_gap, dict):
                continue
            if str(gap_id) not in graph_gap_ids:
                continue
            evidence_ref = str(raw_gap.get("contract_evidence_ref") or "")
            referenced_evidence = external_evidence.get(evidence_ref)
            evidence = (
                referenced_evidence
                if isinstance(referenced_evidence, dict)
                else raw_gap.get("contract_evidence_summary")
                if isinstance(
                    raw_gap.get("contract_evidence_summary"),
                    dict,
                )
                else raw_gap.get("contract_evidence")
                if isinstance(raw_gap.get("contract_evidence"), dict)
                else {}
            )
            if evidence_ref:
                if isinstance(referenced_evidence, dict):
                    resolved_evidence_refs.add(evidence_ref)
                else:
                    unresolved_evidence_refs.add(evidence_ref)
            old_entity = (
                evidence.get("old_entity")
                if isinstance(evidence.get("old_entity"), dict)
                else {}
            )
            donor_entity = (
                evidence.get("donor_entity")
                if isinstance(evidence.get("donor_entity"), dict)
                else {}
            )
            canonical_entity = donor_entity or old_entity
            exact_keys = self._exact_contract_keys(old_entity)
            exact_keys.update(self._exact_contract_keys(donor_entity))
            if not canonical_entity or not exact_keys:
                continue
            contract_identity = str(
                raw_gap.get("contract_identity")
                or evidence.get("contract_identity")
                or ""
            )
            canonical_key = str(
                canonical_entity.get("contract_key")
                or canonical_entity.get("key")
                or ""
            )
            contract_id = stable_id(
                "semantic-ledger-contract",
                str(canonical_entity.get("kind") or "entity"),
                canonical_key,
                normalize_relpath(str(canonical_entity.get("file") or "")),
            )
            existing = self.graph.nodes.get(contract_id) or {}
            semantic_gap_ids = sorted(
                {
                    *[str(item) for item in existing.get("semantic_gap_ids") or []],
                    str(gap_id),
                }
            )
            contract_identities = sorted(
                {
                    *[
                        str(item)
                        for item in existing.get("contract_identities") or []
                        if str(item)
                    ],
                    *([contract_identity] if contract_identity else []),
                }
            )
            self.graph.add_node(
                contract_id,
                "Contract",
                budget_exempt=True,
                overlay_class="semantic_ledger",
                contract_key=canonical_key,
                contract_keys=sorted(exact_keys),
                contract_identities=contract_identities,
                semantic_gap_ids=semantic_gap_ids,
                symbol=canonical_entity.get("symbol"),
                entity_kind=canonical_entity.get("kind"),
                file=canonical_entity.get("file"),
                semantic_statuses=sorted(
                    {
                        *[
                            str(item)
                            for item in existing.get("semantic_statuses") or []
                            if str(item)
                        ],
                        str(raw_gap.get("status") or "open"),
                    }
                ),
            )
            for key in exact_keys:
                self.semantic_contract_nodes_by_key[key].add(contract_id)
            added_gap_ids.add(str(gap_id))
            for entity in (old_entity, donor_entity):
                entity_id = str(
                    entity.get("id") or entity.get("entity_id") or ""
                )
                if entity_id:
                    self.semantic_contract_nodes_by_entity_id[entity_id].add(
                        contract_id
                    )
                    retained_entities[entity_id] = dict(entity)

        for record in retained_entities.values():
            symbol = str(record.get("symbol") or record.get("name") or "")
            if symbol and not any(
                str(item.get("id") or item.get("entity_id") or "")
                == str(record.get("id") or record.get("entity_id") or "")
                for item in self.entities_by_name[symbol]
            ):
                self.entities_by_name[symbol].append(record)
            for contract_key in self._exact_contract_keys(record):
                if not any(
                    str(item.get("id") or item.get("entity_id") or "")
                    == str(record.get("id") or record.get("entity_id") or "")
                    for item in self.entities_by_contract[contract_key]
                ):
                    self.entities_by_contract[contract_key].append(record)
            self._retain_diagnostic_entity_node(record)

        transaction_node_ids: list[str] = []
        if isinstance(active_batch, dict):
            contract_nodes_by_gap: dict[str, set[str]] = defaultdict(set)
            for node_id, node in self.graph.nodes.items():
                if str(node.get("kind") or "") != "Contract":
                    continue
                for gap_id in node.get("semantic_gap_ids") or []:
                    if str(gap_id):
                        contract_nodes_by_gap[str(gap_id)].add(node_id)
            active_patches = [
                dict(raw_patch)
                for raw_patch in active_batch.get("patches") or []
                if isinstance(raw_patch, dict)
                and str(raw_patch.get("id") or "")
            ]
            for raw_patch in active_patches:
                patch_id = str(raw_patch.get("id") or "")
                transaction_id = stable_id(
                    "semantic-patch-transaction",
                    active_batch_id,
                    patch_id,
                )
                gap_ids = sorted(
                    {
                        str(item)
                        for item in raw_patch.get("gap_ids") or []
                        if str(item)
                    }
                )
                self.graph.add_node(
                    transaction_id,
                    "PatchTransaction",
                    budget_exempt=True,
                    overlay_class="semantic_ledger",
                    batch_id=active_batch_id,
                    patch_id=patch_id,
                    patch_kind=raw_patch.get("kind"),
                    semantic_gap_ids=gap_ids,
                    depends_on_patch_ids=sorted(
                        {
                            str(item)
                            for item in raw_patch.get(
                                "depends_on_patch_ids"
                            )
                            or []
                            if str(item)
                        }
                    ),
                    touched_files=sorted(
                        {
                            normalize_relpath(str(item))
                            for item in raw_patch.get("touched_files") or []
                            if str(item)
                        }
                    ),
                )
                self.semantic_transaction_nodes_by_patch_id[
                    patch_id
                ] = transaction_id
                transaction_node_ids.append(transaction_id)
                for gap_id in gap_ids:
                    for contract_id in sorted(
                        contract_nodes_by_gap.get(gap_id) or set()
                    ):
                        self.graph.add_edge(
                            contract_id,
                            transaction_id,
                            "part_of_patch_transaction",
                            budget_exempt=True,
                            overlay_class="semantic_ledger",
                            semantic_gap_id=gap_id,
                            patch_id=patch_id,
                        )
            for raw_patch in active_patches:
                patch_id = str(raw_patch.get("id") or "")
                transaction_id = self.semantic_transaction_nodes_by_patch_id[
                    patch_id
                ]
                for dependency_patch_id in sorted(
                    {
                        str(item)
                        for item in raw_patch.get("depends_on_patch_ids") or []
                        if str(item)
                        in self.semantic_transaction_nodes_by_patch_id
                    }
                ):
                    self.graph.add_edge(
                        transaction_id,
                        self.semantic_transaction_nodes_by_patch_id[
                            dependency_patch_id
                        ],
                        "depends_on_patch_transaction",
                        budget_exempt=True,
                        overlay_class="semantic_ledger",
                        patch_id=patch_id,
                        dependency_patch_id=dependency_patch_id,
                    )
        return {
            "semantic_gap_ids": sorted(added_gap_ids),
            "contract_node_ids": sorted(
                {
                    node_id
                    for node_ids in self.semantic_contract_nodes_by_key.values()
                    for node_id in node_ids
                }
            ),
            "patch_transaction_node_ids": sorted(
                set(transaction_node_ids)
            ),
            "resolved_contract_evidence_refs": sorted(
                resolved_evidence_refs
            ),
            "unresolved_contract_evidence_refs": sorted(
                unresolved_evidence_refs
            ),
        }

    def bind_diagnostics_to_semantic_transactions(
        self,
        issues: Iterable[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Connect diagnostics to active patch transactions by exact loci.

        Contract edges retained by ``add_semantic_gap_ledger`` cover exact
        provider/consumer identities.  This method adds the complementary
        postimage-hunk incidence needed when a patch introduced an entity that
        was not present in the pre-patch graph.  Same-file proximity alone is
        deliberately insufficient.
        """

        active_batch_id = str(state.get("active_batch_id") or "")
        active_batch = (
            (state.get("batches") or {}).get(active_batch_id)
            if active_batch_id
            else None
        )
        if not isinstance(active_batch, dict):
            return {
                "batch_id": active_batch_id or None,
                "bound_diagnostic_count": 0,
                "binding_count": 0,
            }
        regions = [
            dict(item)
            for item in active_batch.get("active_patch_regions") or []
            if isinstance(item, dict)
        ]
        bound_events: set[str] = set()
        binding_count = 0
        for issue in issues:
            event_id = str(
                issue.get("diagnostic_event_id")
                or stable_id("diag", repr(issue))
            )
            mapping = self.diagnostic_mappings.get(event_id) or {}
            raw_locations = [
                {
                    "file": self._normalize_issue_file(issue),
                    "line": _int_or_zero(issue.get("line")),
                },
                *[
                    dict(item)
                    for item in mapping.get("diagnostic_location_chain")
                    or []
                    if isinstance(item, dict)
                ],
                *[
                    dict(item)
                    for item in mapping.get("related_location_evidence")
                    or []
                    if isinstance(item, dict)
                ],
            ]
            matched_patch_ids: set[str] = set()
            evidence: list[dict[str, Any]] = []
            seen_loci: set[tuple[str, int]] = set()
            for location in raw_locations:
                relative_path = normalize_relpath(
                    str(
                        location.get("file")
                        or location.get("source_file")
                        or ""
                    )
                )
                line = _int_or_zero(location.get("line"))
                locus = (relative_path, line)
                if not relative_path or line <= 0 or locus in seen_loci:
                    continue
                seen_loci.add(locus)
                for region in regions:
                    if normalize_relpath(
                        str(region.get("file") or "")
                    ) != relative_path:
                        continue
                    start_line = _int_or_zero(region.get("start_line"))
                    end_line = _int_or_zero(
                        region.get("end_line") or start_line
                    )
                    if start_line <= line <= end_line:
                        patch_ids = sorted(
                            {
                                str(item)
                                for item in region.get("patch_ids") or []
                                if str(item)
                                in self.semantic_transaction_nodes_by_patch_id
                            }
                        )
                        if not patch_ids:
                            continue
                        matched_patch_ids.update(patch_ids)
                        evidence.append(
                            {
                                "file": relative_path,
                                "line": line,
                                "start_line": start_line,
                                "end_line": end_line,
                                "patch_ids": patch_ids,
                                "gap_ids": sorted(
                                    {
                                        str(item)
                                        for item in region.get("gap_ids")
                                        or []
                                        if str(item)
                                    }
                                ),
                            }
                        )
            if not matched_patch_ids:
                continue
            roots = set(self.diagnostic_roots.get(event_id) or set())
            for patch_id in sorted(matched_patch_ids):
                transaction_id = self.semantic_transaction_nodes_by_patch_id[
                    patch_id
                ]
                if roots:
                    for root_id in sorted(roots):
                        self.graph.add_edge(
                            root_id,
                            transaction_id,
                            "part_of_patch_transaction",
                            budget_exempt=True,
                            overlay_class="semantic_ledger",
                            diagnostic_event_id=event_id,
                            patch_id=patch_id,
                        )
                else:
                    self.diagnostic_roots[event_id].add(transaction_id)
                binding_count += 1
            mapping["semantic_transaction_evidence"] = evidence
            mapping["semantic_transaction_ids"] = sorted(
                matched_patch_ids
            )
            bound_events.add(event_id)
        return {
            "batch_id": active_batch_id,
            "bound_diagnostic_count": len(bound_events),
            "binding_count": binding_count,
        }

    def _init_contexts(self) -> None:
        if self.spec is None:
            return
        old_root = Path(getattr(self.spec, "old_root_path"))
        old_context, old_report = create_clang_context(
            getattr(self.spec, "compile_commands", None),
            old_root,
            path_maps=list(getattr(self.spec, "compile_path_maps", []) or []),
            make_bin=getattr(self.spec, "make_bin", None),
            build_env=dict(getattr(self.spec, "build_env", {}) or {}),
            candidate_build_dir=getattr(self.spec, "build_dir_path", None),
        )
        self.contexts["old"] = old_context
        self.context_reports["old"] = old_report
        donor_roots = list(getattr(self.spec, "donor_root_paths", []) or [])
        donor_commands = list(getattr(self.spec, "donor_compile_commands", []) or [])
        donor_maps = list(getattr(self.spec, "donor_compile_path_maps", []) or [])
        for index, root in enumerate(donor_roots):
            tree = f"donor_{index}"
            compile_commands = donor_commands[index] if index < len(donor_commands) else None
            context, report = create_clang_context(
                compile_commands,
                Path(root),
                path_maps=donor_maps,
                make_bin=getattr(self.spec, "make_bin", None),
                build_env=dict(getattr(self.spec, "build_env", {}) or {}),
            )
            self.contexts[tree] = context
            self.context_reports[tree] = report

    def import_graph_snapshot(
        self,
        *,
        nodes: Iterable[dict[str, Any]],
        edges: Iterable[dict[str, Any]],
        truncated: bool = False,
        truncation_reasons: Iterable[object] = (),
    ) -> None:
        """Restore an earlier CodeIndex page before continuing graph traversal.

        Scope expansion is paged over compile-command translation units.  A cursor
        without the graph accumulated before that cursor is not resumable: later
        pages could forget an entity found on an earlier page.  This importer
        restores stable entity identities and typed edges without reparsing source.
        """

        entity_records_by_id: dict[str, dict[str, Any]] = {
            str(record.get("id") or ""): record
            for records in self.entities_by_name.values()
            for record in records
            if record.get("id")
        }
        imported_entities = set(entity_records_by_id)
        pending_references: list[tuple[str, dict[str, Any]]] = []
        for raw_node in nodes:
            if not isinstance(raw_node, dict):
                continue
            node = dict(raw_node)
            node_id = str(node.pop("id", "") or "")
            kind = str(node.pop("kind", "") or "")
            budget_exempt = bool(node.pop("budget_exempt", False))
            if not node_id or not kind:
                continue
            self.graph.add_node(
                node_id,
                kind,
                budget_exempt=budget_exempt,
                **node,
            )
            tree = str(node.get("tree") or "")
            if kind == "Reference":
                pending_references.append((node_id, node))
                continue
            if kind == "Include":
                source_file = normalize_relpath(
                    str(node.get("source_file") or node.get("from") or "")
                )
                if tree and source_file:
                    include = {"id": node_id, **node}
                    self.includes_by_file[(tree, source_file)].append(include)
                continue
            symbol = str(node.get("symbol") or node.get("name") or "")
            entity_kind = str(node.get("entity_kind") or "")
            relative_path = normalize_relpath(str(node.get("file") or ""))
            if (
                node_id in imported_entities
                or not tree
                or not symbol
                or not relative_path
                or entity_kind not in {"function", "variable", "type", "field", "enum_constant", "macro"}
            ):
                continue
            record = {
                "id": node_id,
                "entity_id": node_id,
                "tree": tree,
                "symbol": symbol,
                "name": symbol,
                "kind": entity_kind,
                "file": relative_path,
                **node,
            }
            entity_records_by_id[node_id] = record
            imported_entities.add(node_id)
            self.entities_by_name[symbol].append(record)
            contract = str(record.get("contract_key") or record.get("key") or "")
            if contract:
                self.entities_by_contract[contract].append(record)
            self.entities_by_file[(tree, relative_path)].append(record)

        for raw_edge in edges:
            if not isinstance(raw_edge, dict):
                continue
            edge = dict(raw_edge)
            source = str(edge.pop("from", "") or "")
            target = str(edge.pop("to", "") or "")
            edge_type = str(edge.pop("type", "") or "")
            budget_exempt = bool(edge.pop("budget_exempt", False))
            if source and target and edge_type:
                self.graph.add_edge(
                    source,
                    target,
                    edge_type,
                    budget_exempt=budget_exempt,
                    **edge,
                )

        for node_id, node in pending_references:
            tree = str(node.get("tree") or "")
            source_file = normalize_relpath(str(node.get("source_file") or ""))
            if not tree or not source_file:
                continue
            reference = {"id": node_id, **node}
            for entity_id_key, entity_key in (
                ("owner_entity_id", "owner_entity"),
                ("referenced_entity_id", "referenced_entity"),
            ):
                if isinstance(reference.get(entity_key), dict):
                    continue
                entity_id = str(reference.get(entity_id_key) or "")
                entity = entity_records_by_id.get(entity_id)
                if entity is not None:
                    reference[entity_key] = entity
            existing_ids = {
                str(item.get("id") or "")
                for item in self.references_by_file[(tree, source_file)]
            }
            if node_id not in existing_ids:
                self.references_by_file[(tree, source_file)].append(reference)
        if truncated:
            self.graph.truncated = True
        self.graph.truncation_reasons.update(
            str(reason) for reason in truncation_reasons if str(reason)
        )

    def root_for_tree(self, tree: str) -> Path | None:
        if self.spec is None:
            return None
        if tree == "old":
            return Path(getattr(self.spec, "old_root_path"))
        if tree.startswith("donor_"):
            try:
                return Path(getattr(self.spec, "donor_root_paths")[int(tree.split("_", 1)[1])])
            except (AttributeError, IndexError, TypeError, ValueError):
                return None
        return None

    def _baseline_old_root(self) -> Path | None:
        if self.spec is None:
            return None
        current_root = Path(getattr(self.spec, "old_root_path")).resolve()
        candidates: set[Path] = set()
        for old_prefix, new_prefix in (
            getattr(self.spec, "compile_path_maps", []) or []
        ):
            baseline = Path(str(old_prefix)).expanduser().resolve()
            mapped_current = Path(str(new_prefix)).expanduser().resolve()
            if (
                mapped_current == current_root
                and baseline != current_root
                and baseline.is_dir()
            ):
                candidates.add(baseline)
        return next(iter(candidates)) if len(candidates) == 1 else None

    def driver_root_for_tree(self, tree: str) -> Path | None:
        root = self.root_for_tree(tree)
        if root is None:
            return None
        return (root / normalize_relpath(getattr(self.spec, "driver_root", ""))).resolve()

    def _map_baseline_contract_successor(
        self,
        *,
        diagnostic_id: str,
        relative_path: str,
        symbol: str,
    ) -> dict[str, Any]:
        """Recover a removed/renamed old contract through its structural successor."""

        baseline_root = self._baseline_old_root()
        current_root = self.root_for_tree("old")
        if (
            baseline_root is None
            or current_root is None
            or not symbol
            or not (baseline_root / relative_path).is_file()
            or not (current_root / relative_path).is_file()
        ):
            return _diagnostic_mapping_result(
                "baseline_contract_unavailable"
            )
        baseline_report = parse_structural_entity_file(
            root=baseline_root,
            relative_path=relative_path,
            tree="baseline_old",
        )
        current_report = parse_structural_entity_file(
            root=current_root,
            relative_path=relative_path,
            tree="old",
        )
        baseline_targets = [
            entity
            for entity in baseline_report.get("entities") or []
            if str(entity.get("symbol") or entity.get("name") or "")
            == symbol
            and entity.get("is_definition")
        ]
        if not baseline_targets:
            return _diagnostic_mapping_result("baseline_contract_absent")

        matches: list[dict[str, Any]] = []
        match_evidence: list[dict[str, Any]] = []
        for target in baseline_targets:
            alignment = structural_entity_neighbor_alignment(
                target_entity_id=str(target.get("id") or ""),
                donor_entities=baseline_report.get("entities") or [],
                old_entities=current_report.get("entities") or [],
            )
            if alignment.get("status") == "existing_contract":
                continue
            baseline_fingerprint = structural_rename_invariant_fingerprint(
                root=baseline_root,
                relative_path=relative_path,
                entity=target,
            )
            if not baseline_fingerprint:
                continue
            predecessor = (alignment.get("predecessor") or {}).get("old") or {}
            successor = (alignment.get("successor") or {}).get("old") or {}
            lower_bound = int(
                (predecessor.get("extent") or {}).get("end_line") or 0
            )
            upper_bound = int(
                (successor.get("extent") or {}).get("start_line")
                or 2**31
            )
            target_context = tuple(
                str(item)
                for item in target.get("conditional_context") or []
            )
            bounded_current_entities = [
                candidate
                for candidate in current_report.get("entities") or []
                if int(
                    (candidate.get("extent") or {}).get("start_line")
                    or 0
                )
                > lower_bound
                and int(
                    (candidate.get("extent") or {}).get("end_line")
                    or 0
                )
                < upper_bound
                and tuple(
                    str(item)
                    for item in candidate.get("conditional_context") or []
                )
                == target_context
            ]
            candidates = [
                candidate
                for candidate in canonical_entity_records(
                    [
                        *list(alignment.get("old_between") or []),
                        *bounded_current_entities,
                    ]
                )
                if str(candidate.get("kind") or "")
                == str(target.get("kind") or "")
                and candidate.get("is_definition")
                and structural_rename_invariant_fingerprint(
                    root=current_root,
                    relative_path=relative_path,
                    entity=candidate,
                )
                == baseline_fingerprint
            ]
            for candidate in candidates:
                matches.append(candidate)
                match_evidence.append(
                    {
                        "baseline_entity_id": target.get("id"),
                        "baseline_contract_key": target.get("contract_key"),
                        "baseline_symbol": symbol,
                        "current_entity_id": candidate.get("id"),
                        "current_contract_key": candidate.get("contract_key"),
                        "alignment_status": alignment.get("status"),
                        "anchor_selection": alignment.get(
                            "anchor_selection"
                        ),
                        "rename_invariant_fingerprint": (
                            baseline_fingerprint
                        ),
                    }
                )
        unique_matches = canonical_entity_records(matches)
        if len(unique_matches) != 1:
            return _diagnostic_mapping_result(
                (
                    "baseline_contract_successor_ambiguous"
                    if unique_matches
                    else "baseline_contract_successor_absent"
                ),
                evidence={
                    "baseline_root": str(baseline_root),
                    "candidate_count": len(unique_matches),
                    "matches": match_evidence,
                },
            )

        successor = unique_matches[0]
        successor_id = self._retain_diagnostic_entity_node(successor)
        if not successor_id:
            return _diagnostic_mapping_result(
                "baseline_contract_successor_not_retained"
            )
        provenance_id = stable_id(
            "historical-contract",
            str(baseline_root),
            relative_path,
            symbol,
        )
        self.graph.add_node(
            provenance_id,
            "HistoricalContract",
            budget_exempt=True,
            tree="baseline_old",
            file=relative_path,
            symbol=symbol,
            baseline_root=str(baseline_root),
        )
        self.graph.add_edge(
            diagnostic_id,
            provenance_id,
            "diagnoses_historical_contract",
            budget_exempt=True,
        )
        self.graph.add_edge(
            provenance_id,
            successor_id,
            "evolved_to_structural_successor",
            budget_exempt=True,
        )
        return _diagnostic_mapping_result(
            "mapped_baseline_contract_successor",
            root_node_ids=[successor_id],
            mapped_node_ids=[provenance_id, successor_id],
            evidence={
                "baseline_root": str(baseline_root),
                "mapping_kind": "structural_rename_invariant_successor",
                "matches": [
                    item
                    for item in match_evidence
                    if str(item.get("current_entity_id") or "")
                    == str(successor.get("id") or "")
                ],
            },
        )

    def index_file(self, tree: str, relative_path: str, *, detailed_processing: bool = True) -> dict[str, Any]:
        relative_path = normalize_relpath(relative_path)
        key = (tree, relative_path)
        if key in self.parse_reports:
            return self.parse_reports[key]
        restored_file = self.graph.nodes.get(
            f"file:{tree}:{relative_path}"
        ) or {}
        if (
            not self.graph.truncated
            and bool(restored_file.get("full_parse_attempted"))
        ):
            report = {
                "file": relative_path,
                "tree": tree,
                "exists": bool(restored_file.get("exists")),
                "parse_ok": bool(restored_file.get("parse_ok")),
                "parse_error": restored_file.get("parse_error"),
                "parse_exception": restored_file.get("parse_exception"),
                "error_diagnostic_count": restored_file.get(
                    "error_diagnostic_count"
                ),
                "compile_context_origin": restored_file.get(
                    "compile_context_origin"
                ),
                "definitions": list(
                    self.entities_by_file.get((tree, relative_path), [])
                ),
                "entities": list(
                    self.entities_by_file.get((tree, relative_path), [])
                ),
                "references": list(
                    self.references_by_file.get((tree, relative_path), [])
                ),
                "includes": list(
                    self.includes_by_file.get((tree, relative_path), [])
                ),
                "parse_seconds": 0.0,
                "restored_from_graph_snapshot": True,
            }
            self.parse_reports[key] = report
            return report
        root = self.root_for_tree(tree)
        driver_root = self.driver_root_for_tree(tree)
        if root is None or driver_root is None:
            report = {
                "file": relative_path,
                "tree": tree,
                "exists": False,
                "parse_ok": False,
                "parse_error": "unknown_tree",
                "definitions": [],
                "entities": [],
                "references": [],
                "includes": [],
            }
        else:
            report = parse_source_file(
                self.contexts.get(tree),
                root.resolve(),
                driver_root,
                relative_path,
                detailed_processing=detailed_processing,
                tree=tree,
            )
        return self.add_parse_report(report)

    def index_file_from_translation_unit(
        self,
        tree: str,
        relative_path: str,
        translation_unit_relative_path: str,
        *,
        detailed_processing: bool = True,
    ) -> dict[str, Any]:
        """Index a focus file through one real case translation unit.

        A header parsed on its own can miss the configured preprocessor state or
        have no compile-command entry. Parsing a known consumer translation unit
        and retaining only cursors located in the focus file gives the entity
        body its actual case configuration.
        """

        relative_path = normalize_relpath(relative_path)
        translation_unit_relative_path = normalize_relpath(
            translation_unit_relative_path
        )
        key = (tree, relative_path, translation_unit_relative_path)
        existing = self.focused_parse_reports.get(key)
        if existing is not None:
            return existing

        root = self.root_for_tree(tree)
        driver_root = self.driver_root_for_tree(tree)
        if root is None or driver_root is None:
            report = {
                "file": relative_path,
                "tree": tree,
                "translation_unit_file": translation_unit_relative_path,
                "exists": False,
                "parse_ok": False,
                "parse_error": "unknown_tree",
                "definitions": [],
                "entities": [],
                "references": [],
                "includes": [],
            }
        else:
            report = parse_source_file(
                self.contexts.get(tree),
                root.resolve(),
                driver_root,
                relative_path,
                detailed_processing=detailed_processing,
                tree=tree,
                translation_unit_relative_path=translation_unit_relative_path,
            )
        self.focused_parse_reports[key] = report
        self._merge_parse_report(report)
        return report

    def index_structural_file(
        self,
        tree: str,
        relative_path: str,
    ) -> dict[str, Any]:
        """Index one owner file with the bounded tree-sitter C backend."""

        relative_path = normalize_relpath(relative_path)
        key = (tree, relative_path)
        existing = self.structural_reports.get(key)
        if existing is not None:
            return existing
        root = self.root_for_tree(tree)
        if root is None:
            report = {
                "file": relative_path,
                "tree": tree,
                "exists": False,
                "parse_ok": False,
                "parse_error": "unknown_tree",
                "entities": [],
            }
        else:
            report = parse_structural_entity_file(
                root=root.resolve(),
                relative_path=relative_path,
                tree=tree,
            )
        self.structural_reports[key] = report
        return report

    @staticmethod
    def _extent_overlaps_error_nodes(
        extent: dict[str, Any],
        error_nodes: Iterable[dict[str, Any]],
    ) -> bool:
        start = int(extent.get("start_line") or 0)
        end = int(extent.get("end_line") or start)
        if start <= 0:
            return True
        for node in error_nodes:
            error_extent = node.get("extent") if isinstance(node.get("extent"), dict) else {}
            error_start = int(error_extent.get("start_line") or 0)
            error_end = int(error_extent.get("end_line") or error_start)
            if error_start > 0 and start <= error_end and error_start <= end:
                return True
        return False

    def materialize_structural_definition_candidates(
        self,
        tree: str,
        relative_path: str,
        symbols: Iterable[object],
        *,
        contract_keys: Iterable[object] = (),
    ) -> list[dict[str, Any]]:
        """Materialize exact tree-sitter definitions when a TU cannot be parsed.

        The lexical candidate relation only selects files containing the exact C
        identifier.  A structural definition is accepted only when its own
        extent does not overlap a tree-sitter error node.  These queried entities
        form a small actionable overlay and therefore remain available even when
        the background graph projection has reached its budget.
        """

        selected = self.structural_definition_candidates(
            tree,
            relative_path,
            symbols,
            contract_keys=contract_keys,
        )
        return self.retain_query_definition_records(selected)

    def retain_query_definition_records(
        self,
        records: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Retain exact query results outside the background graph budget."""

        selected = canonical_entity_records(
            record
            for record in records
            if bool(record.get("is_definition"))
            and _entity_has_source_extent(record)
            and record.get("tree")
            and record.get("file")
        )
        if not selected:
            return []
        touched_contracts: set[str] = set()
        known_ids_by_file: dict[tuple[str, str], set[str]] = {}
        for entity in selected:
            tree = str(entity.get("tree") or "")
            relative_path = normalize_relpath(str(entity.get("file") or ""))
            entity_id = str(entity.get("id") or entity.get("entity_id") or "")
            if not tree or not relative_path or not entity_id:
                continue
            file_key = (tree, relative_path)
            known_file_ids = known_ids_by_file.setdefault(
                file_key,
                {
                    str(existing.get("id") or "")
                    for existing in self.entities_by_file[file_key]
                },
            )
            file_id = f"file:{tree}:{relative_path}"
            self.graph.add_node(
                file_id,
                "File",
                budget_exempt=True,
                overlay_class="query_resolution",
                tree=tree,
                path=relative_path,
                exists=True,
                query_definition_owner=True,
            )
            self.graph.add_node(
                entity_id,
                str(entity.get("kind") or "Entity").title(),
                budget_exempt=True,
                overlay_class="query_resolution",
                **_entity_graph_payload(entity),
            )
            self.graph.add_edge(
                file_id,
                entity_id,
                "contains",
                budget_exempt=True,
                overlay_class="query_resolution",
            )
            self.graph.add_edge(
                entity_id,
                file_id,
                "defined_in",
                budget_exempt=True,
                overlay_class="query_resolution",
            )
            if entity_id not in known_file_ids:
                self.entities_by_file[file_key].append(entity)
                known_file_ids.add(entity_id)
            symbol = str(entity.get("symbol") or "")
            if all(
                str(existing.get("id") or "") != entity_id
                for existing in self.entities_by_name[symbol]
            ):
                self.entities_by_name[symbol].append(entity)
            contract = str(entity.get("contract_key") or entity.get("key") or "")
            if contract:
                if all(
                    str(existing.get("id") or "") != entity_id
                    for existing in self.entities_by_contract[contract]
                ):
                    self.entities_by_contract[contract].append(entity)
                touched_contracts.add(contract)
        self._link_corresponding_contracts(
            touched_contracts,
            budget_exempt=True,
        )
        return canonical_entity_records(selected)

    def structural_definition_candidates(
        self,
        tree: str,
        relative_path: str,
        symbols: Iterable[object],
        *,
        contract_keys: Iterable[object] = (),
    ) -> list[dict[str, Any]]:
        """Return exact, structurally valid definitions from one candidate file."""

        wanted = {str(symbol) for symbol in symbols if str(symbol)}
        wanted_contracts = _normalized_contract_keys(contract_keys)
        if not wanted:
            return []
        report = self.index_structural_file(tree, relative_path)
        if not report.get("parse_ok"):
            return []
        return [
            dict(entity)
            for entity in report.get("entities") or []
            if bool(entity.get("is_definition"))
            and str(entity.get("symbol") or "") in wanted
            and (
                not wanted_contracts
                or str(
                    entity.get("contract_key")
                    or entity.get("key")
                    or ""
                )
                in wanted_contracts
            )
            and not self._extent_overlaps_error_nodes(
                (
                    entity["declaration_extent"]
                    if str(entity.get("kind") or "") == "function"
                    and isinstance(entity.get("declaration_extent"), dict)
                    else entity.get("extent")
                    if isinstance(entity.get("extent"), dict)
                    else {}
                ),
                report.get("error_nodes") or [],
            )
        ]

    def _owner_entity_alignment(
        self,
        *,
        donor_tree: str,
        source_file: str,
        target_entity_id: str,
        target_file: str,
    ) -> dict[str, Any]:
        old_root = self.root_for_tree("old")
        donor_root = self.root_for_tree(donor_tree)
        if (
            old_root is None
            or donor_root is None
            or not (old_root / target_file).is_file()
            or not (donor_root / target_file).is_file()
        ):
            return {
                "schema": "structural_entity_neighbor_alignment_v1",
                "status": "unavailable",
                "reason": "old/donor owner file is missing",
            }
        donor_report = self.index_structural_file(donor_tree, target_file)
        old_report = self.index_structural_file("old", target_file)
        if not donor_report.get("parse_ok") or not old_report.get("parse_ok"):
            return {
                "schema": "structural_entity_neighbor_alignment_v1",
                "status": "unavailable",
                "reason": "bounded structural parser could not index both owner files",
                "donor_parse_error": donor_report.get("parse_error"),
                "old_parse_error": old_report.get("parse_error"),
            }
        target_node = self.graph.nodes.get(target_entity_id) or {}
        target_contract = str(target_node.get("contract_key") or target_node.get("key") or "")
        target_kind = str(target_node.get("entity_kind") or target_node.get("kind") or "").lower()
        donor_entities = [
            entity
            for entity in donor_report.get("entities") or []
            if isinstance(entity, dict)
        ]
        target_matches = [
            entity
            for entity in donor_entities
            if str(entity.get("contract_key") or entity.get("key") or "") == target_contract
            and str(entity.get("kind") or entity.get("entity_kind") or "").lower() == target_kind
        ]
        if len(target_matches) > 1:
            try:
                donor_lines = (donor_root / target_file).read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines(keepends=True)
                old_lines = (old_root / target_file).read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines(keepends=True)
            except OSError:
                conditional_plan = None
            else:
                conditional_plan = conditional_variant_insertions(
                    donor_lines=donor_lines,
                    old_lines=old_lines,
                    target_entities=target_matches,
                    donor_entities=donor_entities,
                    old_entities=[
                        entity
                        for entity in old_report.get("entities") or []
                        if isinstance(entity, dict)
                    ],
                )
            if conditional_plan is not None:
                return {
                    "schema": "conditional_contract_owner_alignment_v1",
                    "status": "conditional_variant_family",
                    "reason": (
                        "all duplicate donor contracts map to distinct branches "
                        "of one matching old preprocessor block"
                    ),
                    "graph_target_entity_id": target_entity_id,
                    "graph_target_contract_key": target_contract,
                    "source_file": source_file,
                    "target_file": target_file,
                    "target_entity_ids": [
                        str(entity.get("id") or entity.get("entity_id") or "")
                        for entity in target_matches
                    ],
                    "conditional_insertion_plan": conditional_plan,
                }
        if len(target_matches) != 1:
            return {
                "schema": "structural_entity_neighbor_alignment_v1",
                "status": "unavailable",
                "reason": "graph target does not map to one structural owner entity",
                "graph_target_entity_id": target_entity_id,
                "graph_target_contract_key": target_contract,
                "structural_target_match_count": len(target_matches),
            }
        target_entity = target_matches[0]
        old_entities = [
            entity
            for entity in old_report.get("entities") or []
            if isinstance(entity, dict)
        ]
        alignment = structural_entity_neighbor_alignment(
            target_entity_id=str(target_entity.get("id") or ""),
            donor_entities=donor_entities,
            old_entities=old_entities,
        )
        alignment["graph_target_entity_id"] = target_entity_id
        alignment["source_file"] = source_file
        return alignment

    def _diagnostic_root_owner_alignments(
        self,
        *,
        source_file: str,
        root_node_ids: Iterable[object],
        existing_evidence: Iterable[dict[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Align every concrete donor root regardless of its mapping path."""

        evidence = [
            dict(item)
            for item in existing_evidence
            if isinstance(item, dict)
        ]
        aligned_ids = {
            str(
                item.get("target_entity_id")
                or (item.get("owner_alignment") or {}).get(
                    "graph_target_entity_id"
                )
                or ""
            )
            for item in evidence
        }
        roots = {str(item) for item in root_node_ids if str(item)}
        candidates = set(roots)
        candidates.update(
            str(edge.get("to") or "")
            for edge in self.graph.edges
            if str(edge.get("from") or "") in roots
            and str(edge.get("type") or "")
            in {"candidate_resolution", "resolves_to"}
        )
        for entity_id in sorted(candidates):
            if not entity_id or entity_id in aligned_ids:
                continue
            node = self.graph.nodes.get(entity_id) or {}
            tree = str(node.get("tree") or "")
            target_file = normalize_relpath(str(node.get("file") or ""))
            if (
                not tree.startswith("donor_")
                or not target_file
                or not _entity_has_source_extent(node)
            ):
                continue
            evidence.append(
                {
                    "tree": tree,
                    "mapping_kind": "selected_donor_root_owner",
                    "target_entity_id": entity_id,
                    "contract_owner_file": target_file,
                    "owner_alignment": self._owner_entity_alignment(
                        donor_tree=tree,
                        source_file=source_file,
                        target_entity_id=entity_id,
                        target_file=target_file,
                    ),
                }
            )
            aligned_ids.add(entity_id)
        return evidence

    def index_files(self, tree: str, relative_paths: Iterable[object]) -> list[dict[str, Any]]:
        return [self.index_file(tree, path) for path in _ordered_unique(relative_paths)]

    def query_reference_files_to_entities(
        self,
        tree: str,
        relative_paths: Iterable[object],
        target_entity_ids: Iterable[object],
    ) -> list[dict[str, Any]]:
        """Parse bounded files and return only typed references to selected entities."""
        target_ids = {str(item) for item in target_entity_ids if str(item)}
        if not target_ids:
            return []
        root = self.root_for_tree(tree)
        driver_root = self.driver_root_for_tree(tree)
        if root is None or driver_root is None:
            return []
        reports: list[dict[str, Any]] = []
        target_key = tuple(sorted(target_ids))
        for relative_path in _ordered_unique(relative_paths):
            full_report = self.parse_reports.get((tree, relative_path))
            if full_report is not None:
                reports.append(
                    {
                        **full_report,
                        "references": [
                            reference
                            for reference in full_report.get("references") or []
                            if str(reference.get("referenced_entity_id") or "")
                            in target_ids
                        ],
                        "parse_seconds": 0.0,
                        "parse_scope": "reference_target_probe",
                        "reference_target_entity_ids": list(target_key),
                        "probe_reused_from": "full_parse_report",
                    }
                )
                continue
            probe_key = (tree, relative_path, target_key)
            cached = self.reference_probe_reports.get(probe_key)
            if cached is not None:
                reports.append(
                    {
                        **cached,
                        "parse_seconds": 0.0,
                        "probe_reused_from": "reference_probe_cache",
                    }
                )
                continue
            report = parse_source_file(
                self.contexts.get(tree),
                root.resolve(),
                driver_root.resolve(),
                relative_path,
                detailed_processing=False,
                tree=tree,
                reference_target_entity_ids=target_ids,
            )
            report = {
                **report,
                "parse_scope": "reference_target_probe",
                "reference_target_entity_ids": list(target_key),
            }
            self.reference_probe_reports[probe_key] = report
            reports.append(report)
        return reports

    def index_kbuild_directory(self, tree: str, directory: str) -> dict[str, Any]:
        root = self.root_for_tree(tree)
        if root is None:
            return {
                "schema": "typed_kbuild_graph_v1",
                "nodes": [],
                "edges": [],
                "summary": {"unavailable": True, "reason": "unknown_tree"},
            }
        kbuild_graph = KbuildIndex.from_directory(root, directory).to_graph(tree=tree)
        for node in kbuild_graph.get("nodes") or []:
            payload = dict(node)
            node_id = str(payload.pop("id"))
            kind = str(payload.pop("kind"))
            self.graph.add_node(node_id, kind, **payload)
        for edge in kbuild_graph.get("edges") or []:
            payload = dict(edge)
            source = str(payload.pop("from"))
            target = str(payload.pop("to"))
            edge_type = str(payload.pop("type"))
            self.graph.add_edge(source, target, edge_type, **payload)
        return kbuild_graph

    def add_parse_report(self, report: dict[str, Any]) -> dict[str, Any]:
        """Merge one canonical parser result, primarily for persistent/index backends."""

        normalized = dict(report)
        normalized["tree"] = str(normalized.get("tree") or "unknown")
        normalized["file"] = normalize_relpath(str(normalized.get("file") or ""))
        key = (normalized["tree"], normalized["file"])
        existing = self.parse_reports.get(key)
        if existing is not None:
            return existing
        self.parse_reports[key] = normalized
        self._merge_parse_report(normalized)
        return normalized

    def add_reference_probe_report(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge a target-filtered reference probe without marking the TU complete."""

        normalized = dict(report)
        normalized["tree"] = str(normalized.get("tree") or "unknown")
        normalized["file"] = normalize_relpath(str(normalized.get("file") or ""))
        target_ids = tuple(
            sorted(
                str(item)
                for item in normalized.get("reference_target_entity_ids") or []
                if str(item)
            )
        )
        normalized["parse_scope"] = "reference_target_probe"
        normalized["reference_target_entity_ids"] = list(target_ids)
        self.reference_probe_reports[
            (normalized["tree"], normalized["file"], target_ids)
        ] = normalized
        self._merge_parse_report(normalized)
        return normalized

    def _merge_parse_report(self, report: dict[str, Any]) -> None:
        tree = str(report.get("tree") or "unknown")
        relative_path = normalize_relpath(str(report.get("file") or ""))
        materialize_graph = self.graph_projection == "occurrence"
        touched_contracts: set[str] = set()
        file_id = f"file:{tree}:{relative_path}"
        if materialize_graph:
            error_diagnostic_count = int(
                report.get("error_diagnostic_count")
                if report.get("error_diagnostic_count") is not None
                else parse_error_diagnostic_count(
                    report.get("diagnostics") or []
                )
            )
            self.graph.add_node(
                file_id,
                "File",
                tree=tree,
                path=relative_path,
                exists=bool(report.get("exists")),
                parse_ok=bool(report.get("parse_ok")),
                parse_error=report.get("parse_error"),
                parse_exception=report.get("parse_exception"),
                error_diagnostic_count=error_diagnostic_count,
                compile_context_origin=report.get(
                    "compile_context_origin"
                ),
                **(
                    {"full_parse_attempted": True}
                    if str(report.get("parse_scope") or "")
                    != "reference_target_probe"
                    else {}
                ),
            )
        file_entity_ids = {
            str(record.get("id") or "")
            for record in self.entities_by_file[(tree, relative_path)]
        }
        file_reference_ids = {
            str(record.get("id") or "")
            for record in self.references_by_file[(tree, relative_path)]
        }
        file_include_ids = {
            str(record.get("id") or "")
            for record in self.includes_by_file[(tree, relative_path)]
        }
        for entity in report.get("entities") or report.get("definitions") or []:
            entity = dict(entity)
            entity_id = str(entity.get("id") or entity.get("entity_id") or "")
            if not entity_id:
                continue
            if materialize_graph:
                self.graph.add_node(
                    entity_id,
                    str(entity.get("kind") or "Entity").title(),
                    **_entity_graph_payload(entity),
                )
                self.graph.add_edge(file_id, entity_id, "contains")
                self.graph.add_edge(
                    entity_id,
                    file_id,
                    (
                        "defined_in"
                        if entity.get("is_definition")
                        else "declared_in"
                    ),
                )
            if entity_id not in file_entity_ids:
                self.entities_by_file[(tree, relative_path)].append(entity)
                file_entity_ids.add(entity_id)
            symbol = str(entity.get("symbol") or "")
            if all(
                str(existing.get("id") or "") != entity_id
                for existing in self.entities_by_name[symbol]
            ):
                self.entities_by_name[symbol].append(entity)
            contract_key = str(entity.get("contract_key") or entity.get("key") or "")
            if all(
                str(existing.get("id") or "") != entity_id
                for existing in self.entities_by_contract[contract_key]
            ):
                self.entities_by_contract[contract_key].append(entity)
            if contract_key:
                touched_contracts.add(contract_key)
        file_entities = self.entities_by_file[(tree, relative_path)]
        type_entities_by_name = {
            str(entity.get("symbol") or ""): entity
            for entity in file_entities
            if str(entity.get("kind") or "") == "type" and entity.get("symbol")
        }
        for entity in file_entities:
            container = entity.get("container") if isinstance(entity.get("container"), dict) else {}
            parent = type_entities_by_name.get(str(container.get("container_name") or ""))
            if (
                materialize_graph
                and parent
                and entity.get("id")
                and parent.get("id")
            ):
                self.graph.add_edge(str(entity["id"]), str(parent["id"]), "member_of")
        for reference in report.get("references") or []:
            reference = dict(reference)
            reference_id = str(reference.get("id") or reference.get("reference_id") or "")
            if not reference_id:
                continue
            if materialize_graph:
                self.graph.add_node(
                    reference_id,
                    "Reference",
                    **_reference_graph_payload(reference),
                )
                self.graph.add_edge(file_id, reference_id, "contains")
            owner = reference.get("owner_entity")
            owner_records = {
                str(entity.get("id") or ""): entity
                for entity in file_entities
                if str(entity.get("id") or "")
            }
            if isinstance(owner, dict) and owner.get("id"):
                owner_records[str(owner["id"])] = owner
            owner_ids = {
                str(reference.get("owner_entity_id") or ""),
                *(
                    str(owner_id)
                    for owner_id in reference.get(
                        "lexical_owner_entity_ids"
                    )
                    or []
                ),
            }
            owner_ids.discard("")
            if materialize_graph:
                for owner_id in sorted(owner_ids):
                    owner_record = owner_records.get(owner_id)
                    if owner_record is not None:
                        self.graph.add_node(
                            owner_id,
                            str(
                                owner_record.get("kind") or "Entity"
                            ).title(),
                            **_entity_graph_payload(owner_record),
                        )
                    if owner_id in self.graph.nodes:
                        self.graph.add_edge(
                            reference_id,
                            owner_id,
                            "occurs_in",
                        )
            target = reference.get("referenced_entity")
            if isinstance(target, dict) and target.get("id"):
                target_id = str(target["id"])
                if materialize_graph:
                    self.graph.add_node(
                        target_id,
                        str(target.get("kind") or "Entity").title(),
                        **_entity_graph_payload(target),
                    )
                    self.graph.add_edge(
                        reference_id,
                        target_id,
                        str(reference.get("use_kind") or "references"),
                    )
                if _entity_has_source_extent(target):
                    target_symbol = str(target.get("symbol") or "")
                    if all(
                        str(existing.get("id") or "") != target_id
                        for existing in self.entities_by_name[target_symbol]
                    ):
                        self.entities_by_name[target_symbol].append(target)
                    contract_key = str(target.get("contract_key") or target.get("key") or "")
                    if all(
                        str(existing.get("id") or "") != target_id
                        for existing in self.entities_by_contract[contract_key]
                    ):
                        self.entities_by_contract[contract_key].append(target)
                    if contract_key:
                        touched_contracts.add(contract_key)
            if reference_id not in file_reference_ids:
                self.references_by_file[(tree, relative_path)].append(reference)
                file_reference_ids.add(reference_id)
        for include in report.get("includes") or []:
            include = dict(include)
            include_id = str(include.get("id") or "")
            if not include_id:
                continue
            target = normalize_relpath(str(include.get("target_file") or ""))
            if materialize_graph:
                self.graph.add_node(
                    include_id,
                    "Include",
                    **_include_graph_payload(include),
                )
                self.graph.add_edge(file_id, include_id, "includes")
            if target:
                target_id = f"file:{tree}:{target}"
                if materialize_graph:
                    self.graph.add_node(
                        target_id,
                        "File",
                        tree=tree,
                        path=target,
                        exists=True,
                    )
                    self.graph.add_edge(include_id, target_id, "resolves_to")
            if include_id not in file_include_ids:
                self.includes_by_file[(tree, relative_path)].append(include)
                file_include_ids.add(include_id)
        if materialize_graph:
            self._link_corresponding_contracts(touched_contracts)

    def _link_corresponding_contracts(
        self,
        contract_keys: Iterable[object] | None = None,
        *,
        budget_exempt: bool = False,
    ) -> None:
        keys = (
            sorted(str(key) for key in contract_keys if str(key))
            if contract_keys is not None
            else sorted(self.entities_by_contract)
        )
        for contract_key in keys:
            records = canonical_entity_records(self.entities_by_contract.get(contract_key, []))
            for record in records:
                entity_id = str(record.get("id") or record.get("entity_id") or "")
                if entity_id and entity_id in self.graph.nodes:
                    self._link_entity_to_semantic_contracts(
                        entity_id,
                        self._exact_contract_keys(record),
                    )
            old_records = [
                record
                for record in records
                if record.get("tree") == "old" and _entity_has_source_extent(record)
            ]
            donor_records = [
                record
                for record in records
                if str(record.get("tree") or "").startswith("donor_")
                and _entity_has_source_extent(record)
            ]
            for old in old_records:
                for donor in donor_records:
                    old_usr = str(old.get("usr") or "")
                    donor_usr = str(donor.get("usr") or "")
                    same_identity = bool(old_usr and donor_usr and old_usr == donor_usr)
                    if not same_identity:
                        same_identity = (
                            normalize_relpath(str(old.get("file") or ""))
                            == normalize_relpath(str(donor.get("file") or ""))
                            and str(old.get("kind") or "") == str(donor.get("kind") or "")
                            and str(old.get("contract_key") or "") == str(donor.get("contract_key") or "")
                        )
                    if not same_identity:
                        continue
                    old_id = str(old.get("id") or "")
                    donor_id = str(donor.get("id") or "")
                    if old_id and donor_id:
                        overlay_payload = (
                            {"overlay_class": "query_resolution"}
                            if budget_exempt
                            else {}
                        )
                        self.graph.add_edge(
                            old_id,
                            donor_id,
                            "same_contract_as",
                            budget_exempt=budget_exempt,
                            **overlay_payload,
                        )
                        self.graph.add_edge(
                            donor_id,
                            old_id,
                            "same_contract_as",
                            budget_exempt=budget_exempt,
                            **overlay_payload,
                        )
                        contract_id = stable_id(
                            "contract",
                            old_usr or normalize_relpath(str(old.get("file") or "")),
                            str(old.get("contract_key") or ""),
                        )
                        self.graph.add_node(
                            contract_id,
                            "Contract",
                            budget_exempt=budget_exempt,
                            **(
                                {"overlay_class": "query_resolution"}
                                if budget_exempt
                                else {}
                            ),
                            contract_key=old.get("contract_key"),
                            symbol=old.get("symbol"),
                            entity_kind=old.get("kind"),
                            old_entity_id=old_id,
                            donor_entity_id=donor_id,
                            file=old.get("file"),
                        )
                        self.graph.add_edge(
                            old_id,
                            contract_id,
                            "instance_of_contract",
                            budget_exempt=budget_exempt,
                            **(
                                {"overlay_class": "query_resolution"}
                                if budget_exempt
                                else {}
                            ),
                        )
                        self.graph.add_edge(
                            donor_id,
                            contract_id,
                            "instance_of_contract",
                            budget_exempt=budget_exempt,
                            **(
                                {"overlay_class": "query_resolution"}
                                if budget_exempt
                                else {}
                            ),
                        )

    def compile_command_files(self, tree: str) -> list[str]:
        context = self.contexts.get(tree)
        root = self.root_for_tree(tree)
        if context is None or root is None:
            return []
        return context.compile_files(root=root)

    def compile_dependency_frontier(
        self,
        tree: str,
        *,
        within_root: str | None = None,
    ) -> dict[str, Any]:
        """Return files reachable from this case's real compilation units.

        The compilation database is already configuration-specific.  Its source
        entries are therefore the structural boundary for a case; recursively
        following resolved includes inside ``within_root`` adds headers without
        admitting unrelated sibling modules from the same directory.
        """

        root = self.root_for_tree(tree)
        driver_root = self.driver_root_for_tree(tree)
        context = self.contexts.get(tree)
        scope_root = normalize_relpath(
            within_root
            if within_root is not None
            else str(getattr(self.spec, "driver_root", "") or "")
        )
        cache_key = (tree, scope_root)
        cached = self._compile_dependency_frontier_cache.get(cache_key)
        if cached is not None:
            return {
                **cached,
                "cache": "code_index_instance",
            }
        if root is None or driver_root is None or context is None:
            result = {
                "schema": "compile_dependency_frontier_v1",
                "tree": tree,
                "status": "unavailable",
                "reason": (self.context_reports.get(tree) or {}).get(
                    "reason",
                    "compile_context_unavailable",
                ),
                "scope_root": scope_root,
                "compile_source_files": [],
                "include_files": [],
                "files": [],
            }
            self._compile_dependency_frontier_cache[cache_key] = result
            return dict(result)

        def in_scope(relative_path: str) -> bool:
            normalized = normalize_relpath(relative_path)
            return bool(
                normalized
                and (
                    not scope_root
                    or normalized == scope_root
                    or normalized.startswith(scope_root.rstrip("/") + "/")
                )
            )

        compile_sources = [
            relative_path
            for relative_path in self.compile_command_files(tree)
            if in_scope(relative_path)
            and (root / normalize_relpath(relative_path)).is_file()
        ]
        queue = list(_ordered_unique(compile_sources))
        visited: set[str] = set()
        include_files: list[str] = []
        unresolved_include_count = 0
        while queue:
            relative_path = queue.pop(0)
            if relative_path in visited:
                continue
            visited.add(relative_path)
            includes = parse_include_directives(
                root=root.resolve(),
                driver_root=driver_root.resolve(),
                relative_path=relative_path,
                tree=tree,
            )
            for include in includes:
                target = normalize_relpath(str(include.get("target_file") or ""))
                if not target:
                    unresolved_include_count += 1
                    continue
                if not in_scope(target) or target in visited:
                    continue
                if not (root / target).is_file() or not _source_like(target):
                    continue
                if target not in include_files:
                    include_files.append(target)
                queue.append(target)

        files = _ordered_unique([*compile_sources, *include_files])
        result = {
            "schema": "compile_dependency_frontier_v1",
            "tree": tree,
            "status": "ready" if compile_sources else "empty",
            "reason": None if compile_sources else "no_compile_sources_in_scope",
            "scope_root": scope_root,
            "compile_source_files": compile_sources,
            "include_files": include_files,
            "files": files,
            "summary": {
                "compile_source_file_count": len(compile_sources),
                "include_file_count": len(include_files),
                "file_count": len(files),
                "unresolved_include_count": unresolved_include_count,
            },
        }
        self._compile_dependency_frontier_cache[cache_key] = result
        return dict(result)

    def exact_token_candidates_in_files(
        self,
        tree: str,
        symbols: Iterable[str],
        relative_paths: Iterable[object],
    ) -> dict[str, Any]:
        """Find exact identifier spellings inside an explicit graph frontier.

        This is candidate discovery only. A file becomes a provider endpoint
        only after the existing AST/structural definition validation succeeds.
        """

        root = self.root_for_tree(tree)
        requested = sorted(
            {
                str(symbol).strip()
                for symbol in symbols
                if IDENTIFIER_RE.fullmatch(str(symbol).strip())
            }
        )
        candidates_by_symbol = {symbol: [] for symbol in requested}
        searched = 0
        tokenized_file_count = 0
        cached_token_file_count = 0
        if root is not None:
            wanted = set(requested)
            for relative_path in _ordered_unique(relative_paths):
                path = (root / relative_path).resolve()
                if (
                    not path.is_file()
                    or not is_under(path, root.resolve())
                    or path.suffix not in SOURCE_SUFFIXES
                ):
                    continue
                searched += 1
                signature = _path_stat_signature(path)
                if signature is None:
                    continue
                cache_key = (tree, relative_path)
                cached = self._identifier_tokens_by_file.get(cache_key)
                if cached is not None and cached[0] == signature:
                    tokens = cached[1]
                    cached_token_file_count += 1
                else:
                    try:
                        source = path.read_text(
                            encoding="utf-8",
                            errors="ignore",
                        )
                    except OSError:
                        continue
                    tokens = frozenset(IDENTIFIER_RE.findall(source))
                    self._identifier_tokens_by_file[cache_key] = (
                        signature,
                        tokens,
                    )
                    tokenized_file_count += 1
                present = wanted.intersection(tokens)
                for symbol in present:
                    candidates_by_symbol[symbol].append(relative_path)
        for symbol in candidates_by_symbol:
            candidates_by_symbol[symbol] = _ordered_unique(
                candidates_by_symbol[symbol]
            )
        return {
            "schema": "bounded_exact_token_candidates_v1",
            "tree": tree,
            "symbols": requested,
            "searched_file_count": searched,
            "tokenized_file_count": tokenized_file_count,
            "cached_token_file_count": cached_token_file_count,
            "candidate_relation_count": sum(
                len(files) for files in candidates_by_symbol.values()
            ),
            "candidates_by_symbol": candidates_by_symbol,
        }

    def exact_build_owned_definition_candidates(
        self,
        tree: str,
        symbols: Iterable[str],
    ) -> dict[str, Any]:
        """Find demanded implementation candidates without admitting siblings.

        The compile/include frontier cannot contain a provider translation unit
        that is built as another object.  This query searches the tree only for
        the exact demanded identifiers, then keeps implementation files whose
        local Kbuild closure contains an assignment for that source.  Matches
        remain read-only candidates until ``query_definition_batches`` verifies
        a concrete definition.
        """

        root = self.root_for_tree(tree)
        requested = sorted(
            {
                str(symbol).strip()
                for symbol in symbols
                if IDENTIFIER_RE.fullmatch(str(symbol).strip())
            }
        )
        empty_candidates = {symbol: [] for symbol in requested}
        if root is None:
            return {
                "schema": "exact_build_owned_definition_candidates_v1",
                "tree": tree,
                "status": "unavailable",
                "reason": "unknown_tree",
                "symbols": requested,
                "candidates_by_symbol": empty_candidates,
            }
        if not requested:
            return {
                "schema": "exact_build_owned_definition_candidates_v1",
                "tree": tree,
                "status": "ready",
                "reason": None,
                "symbols": [],
                "lexical_candidate_files": [],
                "build_owned_candidate_files": [],
                "rejected_candidate_files": [],
                "kbuild_assignment_ids_by_file": {},
                "candidates_by_symbol": {},
            }

        command = [
            "rg",
            "--files-with-matches",
            "--word-regexp",
            "--no-messages",
            "--glob",
            "*.c",
            "--glob",
            "*.cc",
            "--glob",
            "*.cpp",
        ]
        for symbol in requested:
            command.extend(["-e", symbol])
        command.append(".")
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "schema": "exact_build_owned_definition_candidates_v1",
                "tree": tree,
                "status": "unavailable",
                "reason": type(exc).__name__,
                "symbols": requested,
                "candidates_by_symbol": empty_candidates,
            }
        if completed.returncode not in {0, 1}:
            return {
                "schema": "exact_build_owned_definition_candidates_v1",
                "tree": tree,
                "status": "unavailable",
                "reason": f"rg_exit_{completed.returncode}",
                "symbols": requested,
                "candidates_by_symbol": empty_candidates,
            }

        lexical_candidates = _ordered_unique(
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip()
        )
        rejected_candidates: list[dict[str, Any]] = []
        structurally_eligible_candidates: list[str] = []
        structurally_uncertain_candidates: list[str] = []
        defined_symbols_by_file: dict[str, set[str]] = {}
        for relative_path in lexical_candidates:
            report = self.index_structural_file(tree, relative_path)
            structural_definitions = self.structural_definition_candidates(
                tree,
                relative_path,
                requested,
            )
            defined_symbols = {
                str(entity.get("symbol") or "")
                for entity in structural_definitions
                if str(entity.get("symbol") or "")
            }
            if defined_symbols:
                structurally_eligible_candidates.append(relative_path)
                defined_symbols_by_file[relative_path] = defined_symbols
                continue
            parser_uncertain = not report.get("parse_ok")
            if not parser_uncertain and report.get("error_nodes"):
                try:
                    source_lines = (
                        (root / relative_path)
                        .read_text(encoding="utf-8", errors="ignore")
                        .splitlines()
                    )
                except OSError:
                    parser_uncertain = True
                else:
                    parser_uncertain = _parser_errors_cover_identifiers(
                        source_lines,
                        report.get("error_nodes") or [],
                        requested,
                    )
            if parser_uncertain:
                structurally_eligible_candidates.append(relative_path)
                structurally_uncertain_candidates.append(relative_path)
                continue
            rejected_candidates.append(
                {
                    "file": relative_path,
                    "reason": "no_structural_definition_for_demanded_symbol",
                }
            )

        build_owned_candidates: list[str] = []
        assignment_ids_by_file: dict[str, list[str]] = {}
        for relative_path in structurally_eligible_candidates:
            directory = normalize_relpath(
                str(Path(relative_path).parent.as_posix())
            )
            try:
                closure = KbuildIndex.from_directory(
                    root,
                    directory,
                ).assignment_closure_for_sources([relative_path])
            except OSError as exc:
                rejected_candidates.append(
                    {
                        "file": relative_path,
                        "reason": type(exc).__name__,
                    }
                )
                continue
            assignment_ids = sorted(
                str(item)
                for item in closure.get("assignment_ids") or []
                if str(item)
            )
            if not assignment_ids:
                rejected_candidates.append(
                    {
                        "file": relative_path,
                        "reason": "no_kbuild_assignment_for_source",
                    }
                )
                continue
            build_owned_candidates.append(relative_path)
            assignment_ids_by_file[relative_path] = assignment_ids

        exact = self.exact_token_candidates_in_files(
            tree,
            requested,
            build_owned_candidates,
        )
        candidates_by_symbol = {
            symbol: [
                relative_path
                for relative_path in (
                    exact.get("candidates_by_symbol") or {}
                ).get(symbol, [])
                if (
                    relative_path in structurally_uncertain_candidates
                    or symbol
                    in defined_symbols_by_file.get(relative_path, set())
                )
            ]
            for symbol in requested
        }
        return {
            "schema": "exact_build_owned_definition_candidates_v1",
            "tree": tree,
            "status": "ready",
            "reason": None,
            "symbols": requested,
            "lexical_candidate_files": lexical_candidates,
            "structurally_eligible_candidate_files": (
                structurally_eligible_candidates
            ),
            "structurally_uncertain_candidate_files": (
                structurally_uncertain_candidates
            ),
            "build_owned_candidate_files": build_owned_candidates,
            "rejected_candidate_files": rejected_candidates,
            "kbuild_assignment_ids_by_file": assignment_ids_by_file,
            "candidates_by_symbol": candidates_by_symbol,
            "summary": {
                "lexical_candidate_file_count": len(lexical_candidates),
                "build_owned_candidate_file_count": len(
                    build_owned_candidates
                ),
                "structurally_eligible_candidate_file_count": len(
                    structurally_eligible_candidates
                ),
                "structurally_uncertain_candidate_file_count": len(
                    structurally_uncertain_candidates
                ),
                "rejected_candidate_file_count": len(rejected_candidates),
                "candidate_relation_count": int(
                    sum(len(files) for files in candidates_by_symbol.values())
                ),
            },
        }

    def _normalize_issue_file(self, issue: dict[str, Any]) -> str:
        for key in ("file", "source_file", "path", "relative_path"):
            value = str(issue.get(key) or "").strip()
            if not value:
                continue
            if self.spec is None:
                return normalize_relpath(value)
            normalized = normalize_root_relative(
                value,
                root=Path(getattr(self.spec, "old_root_path")),
                preferred_prefix=getattr(self.spec, "driver_root", None),
            )
            if normalized:
                return normalize_relpath(normalized)
        return ""

    @staticmethod
    def _contains_position(extent: dict[str, Any], line: int, column: int) -> bool:
        start_line = int(extent.get("start_line") or 0)
        end_line = int(extent.get("end_line") or 0)
        if not line or not start_line or not end_line or line < start_line or line > end_line:
            return False
        if column and line == start_line and column < int(extent.get("start_column") or 0):
            return False
        if column and line == end_line and int(extent.get("end_column") or 0) and column > int(extent.get("end_column") or 0):
            return False
        return True

    @staticmethod
    def _extent_order(record: dict[str, Any]) -> tuple[int, int, int, str]:
        extent = record.get("extent") or {}
        line_span = max(0, int(extent.get("end_line") or 0) - int(extent.get("start_line") or 0))
        column_span = (
            max(0, int(extent.get("end_column") or 0) - int(extent.get("start_column") or 0))
            if line_span == 0
            else 0
        )
        return line_span, column_span, int(extent.get("start_line") or 0), str(record.get("id") or "")

    def _retain_diagnostic_file_node(self, tree: str, relative_path: str) -> str:
        """Materialize one file endpoint in the non-droppable diagnostic overlay."""

        relative_path = normalize_relpath(relative_path)
        if not tree or not relative_path:
            return ""
        report = self.parse_reports.get((tree, relative_path)) or {}
        root = self.root_for_tree(tree)
        exists = bool(report.get("exists"))
        if not report and root is not None:
            exists = (root / relative_path).is_file()
        file_id = f"file:{tree}:{relative_path}"
        self.graph.add_node(
            file_id,
            "File",
            budget_exempt=True,
            tree=tree,
            path=relative_path,
            exists=exists,
            parse_ok=report.get("parse_ok"),
            parse_error=report.get("parse_error"),
            parse_exception=report.get("parse_exception"),
            compile_context_origin=report.get(
                "compile_context_origin"
            ),
        )
        return file_id

    def _retain_diagnostic_entity_node(self, record: dict[str, Any]) -> str:
        """Materialize exactly one AST entity used on a diagnostic causal path."""

        entity_id = str(record.get("id") or record.get("entity_id") or "")
        if not entity_id:
            return ""
        self.graph.add_node(
            entity_id,
            str(record.get("kind") or "Entity").title(),
            budget_exempt=True,
            **_entity_graph_payload(record),
        )
        tree = str(record.get("tree") or "")
        relative_path = normalize_relpath(str(record.get("file") or ""))
        file_id = self._retain_diagnostic_file_node(tree, relative_path)
        if file_id:
            self.graph.add_edge(
                entity_id,
                file_id,
                "defined_in" if record.get("is_definition") else "declared_in",
                budget_exempt=True,
            )
        self._link_entity_to_semantic_contracts(
            entity_id,
            self._exact_contract_keys(record),
        )
        return entity_id

    def _retain_diagnostic_reference_node(self, record: dict[str, Any]) -> str:
        """Materialize one reference and only its concrete AST endpoints."""

        reference_id = str(record.get("id") or record.get("reference_id") or "")
        if not reference_id:
            return ""
        self.graph.add_node(
            reference_id,
            "Reference",
            budget_exempt=True,
            **_reference_graph_payload(record),
        )
        tree = str(record.get("tree") or "")
        source_file = normalize_relpath(str(record.get("source_file") or ""))
        file_id = self._retain_diagnostic_file_node(tree, source_file)
        if file_id:
            self.graph.add_edge(
                file_id,
                reference_id,
                "contains",
                budget_exempt=True,
            )

        owner = record.get("owner_entity")
        owner_id = str(record.get("owner_entity_id") or "")
        if isinstance(owner, dict):
            owner_id = self._retain_diagnostic_entity_node(owner) or owner_id
        if owner_id and owner_id in self.graph.nodes:
            self.graph.add_edge(
                reference_id,
                owner_id,
                "occurs_in",
                budget_exempt=True,
            )

        target = record.get("referenced_entity")
        target_id = str(record.get("referenced_entity_id") or "")
        if isinstance(target, dict):
            target_id = self._retain_diagnostic_entity_node(target) or target_id
        if target_id and target_id in self.graph.nodes:
            self.graph.add_edge(
                reference_id,
                target_id,
                str(record.get("use_kind") or "references"),
                budget_exempt=True,
            )
        return reference_id

    def _retain_diagnostic_include_node(self, record: dict[str, Any]) -> str:
        """Materialize one include edge used by a compiler diagnostic."""

        include_id = str(record.get("id") or "")
        if not include_id:
            return ""
        self.graph.add_node(
            include_id,
            "Include",
            budget_exempt=True,
            **_include_graph_payload(record),
        )
        tree = str(record.get("tree") or "old")
        source_file = normalize_relpath(str(record.get("source_file") or ""))
        source_id = self._retain_diagnostic_file_node(tree, source_file)
        if source_id:
            self.graph.add_edge(
                source_id,
                include_id,
                "includes",
                budget_exempt=True,
            )
        target_file = normalize_relpath(str(record.get("target_file") or ""))
        target_id = self._retain_diagnostic_file_node(tree, target_file)
        if target_id:
            self.graph.add_edge(
                include_id,
                target_id,
                "resolves_to",
                budget_exempt=True,
            )
        return include_id

    def _map_donor_reference_at_position(
        self,
        *,
        diagnostic_id: str,
        relative_path: str,
        line: int,
        column: int,
    ) -> dict[str, Any]:
        """Resolve an old diagnostic through the exact corresponding donor use."""

        empty = {"root_node_ids": [], "mapped_node_ids": [], "evidence": []}
        old_root = self.root_for_tree("old")
        if old_root is None:
            return empty
        old_token = _identifier_at_source_position(old_root, relative_path, line, column)
        old_position = _source_position_at_display_column(
            old_root,
            relative_path,
            line,
            column,
        )
        if old_token is None and old_position is None:
            return empty

        root_node_ids: set[str] = set()
        mapped_node_ids: list[str] = []
        evidence: list[dict[str, Any]] = []
        donor_trees = sorted(name for name in self.contexts if name.startswith("donor_"))
        for tree in donor_trees:
            donor_root = self.root_for_tree(tree)
            if donor_root is None:
                continue
            donor_token = _identifier_at_source_position(
                donor_root,
                relative_path,
                line,
                column,
            )
            references = self.references_by_file.get((tree, relative_path), [])
            mapping_kind = "lexical_identifier"
            if old_token is not None:
                if donor_token != old_token:
                    continue
                selected_references = [
                    reference
                    for reference in references
                    if str(reference.get("symbol") or "") == str(old_token["identifier"])
                    and int(reference.get("line") or 0) == line
                    and int(reference.get("column") or 0) == int(old_token["start_column"])
                    and self._contains_position(
                        reference.get("extent") or {},
                        line,
                        int(old_token["start_column"]),
                    )
                ]
            else:
                donor_position = _source_position_at_display_column(
                    donor_root,
                    relative_path,
                    line,
                    column,
                )
                if (
                    donor_token is not None
                    or donor_position is None
                    or old_position is None
                    or int(donor_position["source_column"])
                    != int(old_position["source_column"])
                    or str(donor_position["source_line"])
                    != str(old_position["source_line"])
                    or str(donor_position["character"])
                    != str(old_position["character"])
                ):
                    continue
                mapping_kind = "operator_operand"
                selected_references = _operator_reference_records(
                    references,
                    line=line,
                    source_line=str(old_position["source_line"]),
                    source_column=int(old_position["source_column"]),
                )

            for reference in selected_references:
                reference_id = self._retain_diagnostic_reference_node(reference)
                target_record = (
                    reference.get("referenced_entity")
                    if isinstance(reference.get("referenced_entity"), dict)
                    else {}
                )
                target_id = str(
                    reference.get("referenced_entity_id")
                    or target_record.get("id")
                    or ""
                )
                target = self.graph.nodes.get(target_id) or {}
                target_file = normalize_relpath(
                    str(target.get("file") or reference.get("referenced_file") or "")
                )
                if (
                    not reference_id
                    or not target_id
                    or target_id not in self.graph.nodes
                    or str(target.get("tree") or "") != tree
                    or not target_file
                ):
                    continue

                owner_id = stable_id("contract-owner", tree, target_file)
                self.graph.add_node(
                    owner_id,
                    "ContractOwner",
                    budget_exempt=True,
                    tree=tree,
                    file=target_file,
                    path=target_file,
                    owner_kind="declaration_or_definition_file",
                    contract_key=f"file:{target_file}",
                )
                self.graph.add_edge(
                    diagnostic_id,
                    reference_id,
                    "diagnoses",
                    budget_exempt=True,
                )
                self.graph.add_edge(
                    target_id,
                    owner_id,
                    "owned_by_contract_file",
                    budget_exempt=True,
                )
                mapped_node_ids.extend([reference_id, target_id, owner_id])
                # The declaration/definition entity is the causal repair root.
                # Its containing file remains an ownership/context node, not a
                # root that can merge unrelated entities from the same header.
                root_node_ids.add(target_id)
                owner_alignment = self._owner_entity_alignment(
                    donor_tree=tree,
                    source_file=relative_path,
                    target_entity_id=target_id,
                    target_file=target_file,
                )
                evidence.append(
                    {
                        "tree": tree,
                        "source_file": relative_path,
                        "mapping_kind": mapping_kind,
                        "identifier": (
                            old_token["identifier"]
                            if old_token is not None
                            else reference.get("symbol")
                        ),
                        "line": line,
                        "start_column": (
                            old_token["start_column"]
                            if old_token is not None
                            else old_position["source_column"]
                        ),
                        "display_start_column": (
                            old_token["display_start_column"]
                            if old_token is not None
                            else column
                        ),
                        "reference_node_id": reference_id,
                        "target_entity_id": target_id,
                        "contract_owner_node_id": owner_id,
                        "contract_owner_file": target_file,
                        "owner_alignment": owner_alignment,
                    }
                )
        return {
            "root_node_ids": sorted(root_node_ids),
            "mapped_node_ids": _ordered_unique(mapped_node_ids),
            "evidence": evidence,
            "identifier": old_token["identifier"] if old_token is not None else None,
        }

    def _diagnostic_module_nodes(
        self,
        module: str,
        *,
        tree: str | None = None,
    ) -> list[str]:
        """Resolve a reported module path against typed Kbuild Module nodes."""

        normalized = normalize_relpath(module)
        if not normalized:
            return []
        module_name = Path(normalized).name
        for suffix in (".ko", ".o"):
            if module_name.endswith(suffix):
                module_name = module_name[: -len(suffix)]
                break
        exact_path_matches: list[str] = []
        name_matches: list[str] = []
        for node_id, node in self.graph.nodes.items():
            if str(node.get("kind") or "") != "Module":
                continue
            if tree is not None and str(node.get("tree") or "") != tree:
                continue
            candidate_name = str(node.get("module") or "")
            if candidate_name != module_name:
                continue
            name_matches.append(node_id)
            directory = normalize_relpath(str(node.get("directory") or ""))
            candidate_path = normalize_relpath(f"{directory}/{candidate_name}.ko")
            if candidate_path == normalized:
                exact_path_matches.append(node_id)
        return sorted(exact_path_matches or name_matches)

    def _module_source_files(
        self,
        module_node_ids: Iterable[str],
        *,
        tree: str,
    ) -> list[str]:
        module_ids = {str(node_id) for node_id in module_node_ids if str(node_id)}
        object_ids = {
            str(edge.get("from") or "")
            for edge in self.graph.edges
            if str(edge.get("type") or "") == "links_into"
            and str(edge.get("to") or "") in module_ids
        }
        file_ids = {
            str(edge.get("from") or "")
            for edge in self.graph.edges
            if str(edge.get("type") or "") == "compiles_to"
            and str(edge.get("to") or "") in object_ids
        }
        return _ordered_unique(
            (self.graph.nodes.get(file_id) or {}).get("path")
            or (self.graph.nodes.get(file_id) or {}).get("file")
            for file_id in sorted(file_ids)
            if str((self.graph.nodes.get(file_id) or {}).get("tree") or "") == tree
        )

    def _map_generated_export_origin(
        self,
        *,
        diagnostic_id: str,
        relative_path: str,
        line: int,
    ) -> dict[str, Any]:
        """Resolve one generated symbol-table row to its unique source call."""

        if not relative_path.endswith(".mod.c") or line <= 0:
            return {}
        output_root = getattr(self.spec, "build_dir_path", None) or self.root_for_tree("old")
        old_root = self.root_for_tree("old")
        if output_root is None or old_root is None:
            return {}
        generated = output_root / relative_path
        if not generated.is_file():
            return {}
        lines = generated.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line > len(lines):
            return {}
        row = lines[line - 1]
        call = re.match(r"^\s*KSYMTAB_[A-Z0-9_]+\s*\(", row)
        if call is None:
            return {}
        parsed = _balanced_call_arguments(row, row.find("(", call.start(), call.end()))
        if parsed is None or len(parsed[1]) < 2 or row[parsed[0] + 1 :].strip() != ";":
            return {}
        symbol = row[parsed[1][0][0] : parsed[1][0][1]].strip()
        if IDENTIFIER_RE.fullmatch(symbol) is None:
            return {}

        module = relative_path[: -len(".mod.c")] + ".ko"
        self.index_kbuild_directory("old", normalize_relpath(str(Path(module).parent)))
        module_ids = self._diagnostic_module_nodes(module, tree="old")
        if len(module_ids) != 1:
            return {}
        module_node = self.graph.nodes.get(module_ids[0]) or {}
        exact_module_path = normalize_relpath(
            f"{module_node.get('directory') or ''}/{module_node.get('module') or ''}.ko"
        )
        if exact_module_path != module:
            return {}
        sources = self._module_source_files(module_ids, tree="old")
        matches: list[dict[str, Any]] = []
        for source_file in sources:
            path = old_root / source_file
            if not path.is_file():
                continue
            report = parse_structural_entity_file(
                root=old_root, relative_path=source_file, tree="old"
            )
            if not report.get("parse_ok"):
                return {}
            source_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for anchor in report.get("source_anchors") or []:
                if (
                    not isinstance(anchor, dict)
                    or anchor.get("kind") != "top_level_call_expression"
                    or str(anchor.get("symbol") or "") not in {
                        "EXPORT_SYMBOL", "EXPORT_SYMBOL_GPL",
                        "EXPORT_SYMBOL_NS", "EXPORT_SYMBOL_NS_GPL",
                    }
                    or symbol not in (anchor.get("source_identifiers") or [])
                ):
                    continue
                extent = anchor.get("extent") or {}
                start = int(extent.get("start_line") or 0)
                end = int(extent.get("end_line") or 0)
                if (
                    start <= 0 or end < start or end > len(source_lines)
                    or structural_error_overlaps_extent(
                        report, start_line=start, end_line=end,
                    )
                ):
                    continue
                source_call = "\n".join(source_lines[start - 1 : end])
                open_paren = source_call.find("(")
                source_args = _balanced_call_arguments(source_call, open_paren)
                if source_args is None or not source_args[1]:
                    continue
                first_arg = source_call[source_args[1][0][0] : source_args[1][0][1]].strip()
                if first_arg == symbol:
                    matches.append(anchor)
        if len(matches) != 1:
            return {}
        anchor = matches[0]
        anchor_id = self._retain_diagnostic_entity_node(anchor)
        if not anchor_id:
            return {}
        self.graph.add_edge(
            diagnostic_id, anchor_id, "diagnoses", budget_exempt=True,
            mapping_kind="generated_export_source_call",
            generated_file=relative_path, generated_line=line,
            exported_symbol=symbol,
        )
        return {
            "status": "mapped_generated_export_source_call",
            "root_node_ids": [anchor_id],
            "mapped_node_ids": [anchor_id, *module_ids],
            "source_file": str(anchor.get("file") or ""),
            "source_extent": dict(anchor.get("extent") or {}),
            "generated_file": relative_path,
            "generated_line": line,
            "exported_symbol": symbol,
        }

    @staticmethod
    def _minimal_structural_dependency_owners(
        report: dict[str, Any],
        *,
        symbol: str,
        line: int | None = None,
        column: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return leaf entities with one complete structural dependency."""

        error_nodes = list(report.get("error_nodes") or [])
        candidates = canonical_entity_records(
            entity
            for entity in report.get("entities") or []
            if isinstance(entity, dict)
            and bool(entity.get("dependency_analysis_complete"))
            and symbol
            in {
                str(item)
                for item in entity.get("dependency_symbols") or []
                if str(item)
            }
            and (
                line is None
                or column is None
                or CodeIndex._contains_position(
                    dict(entity.get("extent") or {}),
                    line,
                    column,
                )
            )
            and _entity_has_source_extent(entity)
            and not CodeIndex._extent_overlaps_error_nodes(
                (
                    entity.get("extent")
                    if isinstance(entity.get("extent"), dict)
                    else {}
                ),
                error_nodes,
            )
        )

        def extent_identity(
            entity: dict[str, Any],
        ) -> tuple[int, int, int, int]:
            extent = dict(entity.get("extent") or {})
            return (
                int(extent.get("start_line") or 0),
                int(extent.get("start_column") or 0),
                int(extent.get("end_line") or 0),
                int(extent.get("end_column") or 0),
            )

        def strictly_contains(
            outer: dict[str, Any],
            inner: dict[str, Any],
        ) -> bool:
            outer_start_line, outer_start_column, outer_end_line, outer_end_column = (
                extent_identity(outer)
            )
            inner_start_line, inner_start_column, inner_end_line, inner_end_column = (
                extent_identity(inner)
            )
            outer_start = (outer_start_line, outer_start_column)
            outer_end = (outer_end_line, outer_end_column)
            inner_start = (inner_start_line, inner_start_column)
            inner_end = (inner_end_line, inner_end_column)
            return (
                outer_start <= inner_start
                and inner_end <= outer_end
                and (outer_start, outer_end) != (inner_start, inner_end)
            )

        return [
            candidate
            for candidate in candidates
            if not any(
                strictly_contains(candidate, other)
                for other in candidates
                if other is not candidate
            )
        ]

    def _retain_exact_dependency_owner_counterparts(
        self,
        owner: dict[str, Any],
    ) -> list[str]:
        """Retain exact donor counterparts for one old diagnostic owner."""

        relative_path = normalize_relpath(str(owner.get("file") or ""))
        contract = str(owner.get("contract_key") or owner.get("key") or "")
        kind = str(owner.get("kind") or owner.get("entity_kind") or "")
        definition_state = owner.get("is_definition")
        if not relative_path or not contract or not kind:
            return []

        retained: list[str] = []
        old_id = self._retain_diagnostic_entity_node(owner)
        for tree in sorted(
            candidate
            for candidate in self.contexts
            if candidate.startswith("donor_")
        ):
            semantic_matches = canonical_entity_records(
                entity
                for entity in self.entities_by_contract.get(contract, [])
                if str(entity.get("tree") or "") == tree
                and str(entity.get("kind") or entity.get("entity_kind") or "")
                == kind
                and (
                    not isinstance(definition_state, bool)
                    or entity.get("is_definition") is definition_state
                )
                and _entity_has_source_extent(entity)
            )
            if len(semantic_matches) == 1:
                donor_id = self._retain_diagnostic_entity_node(
                    semantic_matches[0]
                )
                if old_id and donor_id:
                    edge_payload = {
                        "budget_exempt": True,
                        "overlay_class": "diagnostic_dependency_owner",
                    }
                    self.graph.add_edge(
                        old_id,
                        donor_id,
                        "same_contract_as",
                        **edge_payload,
                    )
                    self.graph.add_edge(
                        donor_id,
                        old_id,
                        "same_contract_as",
                        **edge_payload,
                    )
                    retained.append(donor_id)
                    continue
            root = self.root_for_tree(tree)
            if root is None or not (root / relative_path).is_file():
                continue
            report = self.index_structural_file(tree, relative_path)
            matches = canonical_entity_records(
                entity
                for entity in report.get("entities") or []
                if isinstance(entity, dict)
                and str(entity.get("contract_key") or entity.get("key") or "")
                == contract
                and str(entity.get("kind") or entity.get("entity_kind") or "")
                == kind
                and (
                    not isinstance(definition_state, bool)
                    or entity.get("is_definition") is definition_state
                )
                and _entity_has_source_extent(entity)
                and not self._extent_overlaps_error_nodes(
                    (
                        entity.get("extent")
                        if isinstance(entity.get("extent"), dict)
                        else {}
                    ),
                    report.get("error_nodes") or [],
                )
            )
            if len(matches) != 1:
                continue
            donor_id = self._retain_diagnostic_entity_node(matches[0])
            if not old_id or not donor_id:
                continue
            edge_payload = {
                "budget_exempt": True,
                "overlay_class": "diagnostic_dependency_owner",
            }
            self.graph.add_edge(
                old_id,
                donor_id,
                "same_contract_as",
                **edge_payload,
            )
            self.graph.add_edge(
                donor_id,
                old_id,
                "same_contract_as",
                **edge_payload,
            )
            retained.append(donor_id)
        return retained

    def _counterpart_directly_calls(
        self,
        counterpart_id: str,
        symbol: str,
    ) -> bool:
        counterpart = self.graph.nodes.get(counterpart_id) or {}
        tree = str(counterpart.get("tree") or "")
        relative_path = normalize_relpath(str(counterpart.get("file") or ""))
        contract = str(counterpart.get("contract_key") or "")
        root = self.root_for_tree(tree)
        if (
            not tree.startswith("donor_")
            or not relative_path
            or not contract
            or not counterpart.get("is_definition")
            or str(counterpart.get("entity_kind") or "").lower() != "function"
            or root is None
        ):
            return False
        report = self.index_structural_file(tree, relative_path)
        owners = [
            entity
            for entity in report.get("entities") or []
            if str(entity.get("contract_key") or "") == contract
            and str(entity.get("kind") or "") == "function"
            and entity.get("is_definition")
            and symbol in (entity.get("dependency_symbols") or [])
            and not self._extent_overlaps_error_nodes(
                entity.get("extent") or {}, report.get("error_nodes") or []
            )
        ]
        if len(owners) != 1:
            return False
        extent = owners[0].get("extent") or {}
        try:
            from tree_sitter_languages import get_parser

            source = (root / relative_path).read_bytes()
            parsed = get_parser("c").parse(source)
        except (ImportError, OSError):
            return False
        pending = [parsed.root_node]
        while pending:
            node = pending.pop()
            if node.type == "call_expression":
                callee = node.child_by_field_name("function")
                if (
                    callee is not None
                    and callee.type == "identifier"
                    and source[callee.start_byte : callee.end_byte].decode(
                        "utf-8", errors="ignore"
                    ) == symbol
                    and int(extent.get("start_line") or 0)
                    <= callee.start_point[0] + 1
                    <= int(extent.get("end_line") or 0)
                ):
                    return True
            pending.extend(reversed(node.children))
        return False

    def _map_exact_build_owned_call_provider(
        self,
        *,
        unresolved_id: str,
        counterpart_ids: Iterable[str],
        symbol: str,
    ) -> dict[str, Any]:
        callers = [
            counterpart_id
            for counterpart_id in _ordered_unique(counterpart_ids)
            if self._counterpart_directly_calls(counterpart_id, symbol)
        ]
        if len(callers) != 1:
            return {"status": "structural_call_not_unique", "provider_ids": [], "queries": []}

        queries: list[dict[str, Any]] = []
        definitions_by_tree: dict[str, list[dict[str, Any]]] = {}
        trees = ["old", *sorted(tree for tree in self.contexts if tree.startswith("donor_"))]
        for tree in _ordered_unique(trees):
            candidates = self.exact_build_owned_definition_candidates(tree, [symbol])
            files = list((candidates.get("candidates_by_symbol") or {}).get(symbol) or [])
            query_evidence: dict[str, Any] = {
                "tree": tree,
                "candidate_source": "exact_build_owned_structural_call",
                "candidate_files": files,
                "candidate_status": candidates.get("status"),
                "kbuild_assignment_ids_by_file": candidates.get("kbuild_assignment_ids_by_file") or {},
            }
            queries.append(query_evidence)
            if candidates.get("status") != "ready":
                return {"status": "candidate_search_unavailable", "provider_ids": [], "queries": queries}
            if not files:
                definitions_by_tree[tree] = []
                continue
            query = self.query_definition_batches(
                tree=tree,
                symbols=[symbol],
                candidate_files=files,
                candidate_source="demand_provider_exact_token",
                max_files=len(files),
                require_definition=True,
                exhaustive_candidate_scan=True,
            )
            query_evidence["query_complete"] = query.get("complete")
            query_evidence["query_unavailable"] = query.get("unavailable")
            if not query.get("complete") or query.get("unavailable"):
                return {"status": "definition_query_incomplete", "provider_ids": [], "queries": queries}
            definitions = canonical_entity_records(
                entity
                for entity in query.get("matches") or []
                if str(entity.get("symbol") or "") == symbol
                and str(entity.get("kind") or "").lower() == "function"
                and entity.get("is_definition")
                and _entity_has_source_extent(entity)
            )
            query_evidence["definition_ids"] = [str(entity.get("id")) for entity in definitions]
            if len(definitions) > 1:
                return {"status": "ambiguous_build_owned_definitions", "provider_ids": [], "queries": queries}
            definitions_by_tree[tree] = definitions

        selected = definitions_by_tree.get("old") or [
            entity
            for tree in trees
            if tree.startswith("donor_")
            for entity in definitions_by_tree.get(tree, [])
        ]
        if len(selected) != 1:
            return {"status": "provider_absent_or_ambiguous", "provider_ids": [], "queries": queries}
        provider_id = self._retain_diagnostic_entity_node(selected[0])
        if not provider_id:
            return {"status": "provider_node_unavailable", "provider_ids": [], "queries": queries}
        self.graph.add_edge(
            callers[0],
            provider_id,
            "requires_provider",
            budget_exempt=True,
            mapping_kind="exact_structural_call_build_owned_definition",
            symbol=symbol,
        )
        self.graph.add_edge(
            unresolved_id,
            provider_id,
            "candidate_resolution",
            budget_exempt=True,
            resolution_source="exact_structural_call_build_owned_definition",
        )
        return {"status": "mapped_build_owned_call_provider", "provider_ids": [provider_id], "queries": queries}

    def _map_reached_donor_contract_consumers(
        self,
        *,
        diagnostic_id: str,
        root_node_ids: Iterable[str],
        diagnostic_file: str,
    ) -> dict[str, Any]:
        """Complete consumers of one donor-only contract in the build frontier."""

        root_ids = sorted(
            {str(node_id) for node_id in root_node_ids if str(node_id)}
        )
        if len(root_ids) != 1:
            return {"status": "not_unique_donor_contract", "owner_node_ids": []}
        root_id = root_ids[0]
        root_node = self.graph.nodes.get(root_id) or {}
        root_tree = str(root_node.get("tree") or "")
        symbol = str(
            root_node.get("symbol") or root_node.get("name") or ""
        ).strip()
        contract = str(
            root_node.get("contract_key") or root_node.get("key") or ""
        )
        entity_kind = str(
            root_node.get("entity_kind") or root_node.get("kind") or ""
        ).lower()
        if (
            not root_tree.startswith("donor_")
            or not IDENTIFIER_RE.fullmatch(symbol)
            or not contract
            or not entity_kind
        ):
            return {"status": "not_donor_contract", "owner_node_ids": []}

        known_current = canonical_entity_records(
            entity
            for entity in self.entities_by_contract.get(contract, [])
            if str(entity.get("tree") or "") == "old"
            and str(
                entity.get("kind") or entity.get("entity_kind") or ""
            ).lower()
            == entity_kind
            and _entity_has_source_extent(entity)
        )
        if known_current:
            return {
                "status": "current_counterpart_present",
                "owner_node_ids": [],
            }

        frontier = self.compile_dependency_frontier("old", within_root="")
        if str(frontier.get("status") or "") != "ready":
            return {
                "status": "compile_frontier_unavailable",
                "owner_node_ids": [],
                "frontier_status": frontier.get("status"),
            }
        candidate_frontier = _ordered_unique(
            [*(frontier.get("files") or []), diagnostic_file]
        )
        token_candidates = self.exact_token_candidates_in_files(
            "old",
            [symbol],
            candidate_frontier,
        )
        candidate_files = _ordered_unique(
            (token_candidates.get("candidates_by_symbol") or {}).get(
                symbol,
                [],
            )
        )
        reports = [
            self.index_structural_file("old", relative_path)
            for relative_path in candidate_files
        ]
        current_counterparts = canonical_entity_records(
            entity
            for report in reports
            if bool(report.get("parse_ok"))
            for entity in report.get("entities") or []
            if isinstance(entity, dict)
            and str(entity.get("contract_key") or entity.get("key") or "")
            == contract
            and str(
                entity.get("kind") or entity.get("entity_kind") or ""
            ).lower()
            == entity_kind
            and _entity_has_source_extent(entity)
        )
        if current_counterparts:
            return {
                "status": "current_counterpart_present",
                "owner_node_ids": [],
                "candidate_files": candidate_files,
            }

        owners = canonical_entity_records(
            owner
            for report in reports
            if bool(report.get("parse_ok"))
            for owner in self._minimal_structural_dependency_owners(
                report,
                symbol=symbol,
            )
            if bool(owner.get("is_definition"))
        )
        owner_node_ids: list[str] = []
        for owner in owners:
            owner_id = self._retain_diagnostic_entity_node(owner)
            if not owner_id:
                continue
            owner_node_ids.append(owner_id)
            self.graph.add_edge(
                diagnostic_id,
                owner_id,
                "diagnoses",
                budget_exempt=True,
                mapping_kind="build_frontier_structural_dependency_owner",
            )
            self.graph.add_edge(
                owner_id,
                root_id,
                "diagnostic_contract_dependency",
                budget_exempt=True,
                mapping_kind="exact_structural_dependency",
                symbol=symbol,
            )
        return {
            "status": (
                "mapped_reached_donor_contract_consumers"
                if owner_node_ids
                else "no_complete_structural_consumers"
            ),
            "authority": "build_frontier_exact_structural_dependency",
            "root_node_id": root_id,
            "symbol": symbol,
            "frontier_file_count": len(frontier.get("files") or []),
            "candidate_files": candidate_files,
            "owner_node_ids": _ordered_unique(owner_node_ids),
        }

    def _map_symbol_diagnostic(
        self,
        *,
        diagnostic_id: str,
        issue: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach a locationless diagnostic to module uses and provider entities."""

        symbol = str(issue.get("symbol") or "").strip()
        if not symbol:
            return {
                "root_node_ids": [],
                "mapped_node_ids": [],
                "module_node_ids": [],
                "module_source_files": [],
                "status": None,
                "unresolved": False,
            }
        module = normalize_relpath(str(issue.get("module") or ""))
        source_contract_evidence = (
            dict(issue.get("_source_contract_evidence") or {})
            if isinstance(issue.get("_source_contract_evidence"), dict)
            else {}
        )
        contract_source_file = normalize_relpath(
            str(source_contract_evidence.get("source_file") or "")
        )
        contract_namespace = str(
            source_contract_evidence.get("namespace") or ""
        ).strip()
        contract_resolution_domain = str(
            source_contract_evidence.get("resolution_domain") or ""
        ).strip()
        exact_source_contract = bool(
            source_contract_evidence.get("authority")
            == "exact_source_position"
            and contract_source_file
            and contract_namespace
            and contract_resolution_domain
            and str(source_contract_evidence.get("identifier") or "") == symbol
        )
        if exact_source_contract:
            resolution_domain = contract_resolution_domain
            unresolved_id = stable_id(
                "unresolved-contract",
                "old",
                resolution_domain,
                contract_namespace,
                symbol,
            )
            unresolved_kind = "UnresolvedContract"
            unresolved_contract_key = f"{contract_namespace}:{symbol}"
            unresolved_status = "unresolved_exact_source_contract"
        else:
            resolution_domain = module or None
            unresolved_id = stable_id("unresolved-symbol", symbol, module)
            unresolved_kind = "UnresolvedEntity"
            unresolved_contract_key = symbol
            unresolved_status = "unresolved_symbol_provider"
        self.graph.add_node(
            unresolved_id,
            unresolved_kind,
            budget_exempt=True,
            spelling=symbol,
            symbol=symbol,
            contract_key=unresolved_contract_key,
            status=unresolved_status,
            reported_module=module or None,
            resolution_domain=resolution_domain,
            source_contract_evidence=(
                source_contract_evidence if exact_source_contract else None
            ),
        )
        self.graph.add_edge(
            diagnostic_id,
            unresolved_id,
            "diagnoses",
            budget_exempt=True,
        )

        module_node_ids = self._diagnostic_module_nodes(module, tree="old")
        if module and not module_node_ids:
            module_directory = normalize_relpath(str(Path(module).parent))
            if module_directory:
                self.index_kbuild_directory("old", module_directory)
                module_node_ids = self._diagnostic_module_nodes(
                    module,
                    tree="old",
                )
        for module_node_id in module_node_ids:
            self.graph.add_edge(
                diagnostic_id,
                module_node_id,
                "reported_for_module",
                budget_exempt=True,
            )
        module_source_files = self._module_source_files(
            module_node_ids,
            tree="old",
        )
        if module and not module_source_files:
            module_directory = normalize_relpath(str(Path(module).parent))
            old_root = self.root_for_tree("old")
            for donor_tree in sorted(
                tree
                for tree in self.contexts
                if tree.startswith("donor_")
            ):
                self.index_kbuild_directory(donor_tree, module_directory)
                donor_module_node_ids = self._diagnostic_module_nodes(
                    module,
                    tree=donor_tree,
                )
                donor_source_files = self._module_source_files(
                    donor_module_node_ids,
                    tree=donor_tree,
                )
                projected_source_files = [
                    relative_path
                    for relative_path in donor_source_files
                    if old_root is not None
                    and (old_root / relative_path).is_file()
                ]
                if not projected_source_files:
                    continue
                module_node_ids = donor_module_node_ids
                module_source_files = projected_source_files
                break
        for relative_path in module_source_files:
            self.index_file("old", relative_path)

        mapped_node_ids = [unresolved_id, *module_node_ids]
        if str(issue.get("kind") or "") == "modpost_namespace_import_missing":
            namespace = str(issue.get("namespace") or "").strip()
            old_root = self.root_for_tree("old")
            source_calls: dict[str, list[dict[str, Any]]] = {}
            if old_root is not None and len(module_node_ids) == 1 and namespace:
                for relative_path in module_source_files:
                    path = old_root / relative_path
                    if path.is_file():
                        source_calls[relative_path] = parse_module_metadata_calls(
                            path.read_text(encoding="utf-8", errors="ignore")
                        )
            imports = [
                (relative_path, call)
                for relative_path, calls in source_calls.items()
                for call in calls
                if call["name"] == "MODULE_IMPORT_NS"
                and call["namespace"] == namespace
            ]
            if len(imports) == 1:
                consumer_file = imports[0][0]
            elif not imports and len(source_calls) == 1:
                consumer_file = next(iter(source_calls))
                licenses = [
                    call
                    for call in source_calls[consumer_file]
                    if call["name"] == "MODULE_LICENSE"
                ]
                if len(licenses) != 1:
                    consumer_file = ""
            else:
                consumer_file = ""
            if consumer_file:
                consumer_id = self._retain_diagnostic_file_node(
                    "old", consumer_file
                )
                self.graph.add_edge(
                    diagnostic_id,
                    consumer_id,
                    "requires_module_namespace_import",
                    budget_exempt=True,
                    namespace=namespace,
                    module=module,
                )
                return {
                    "root_node_ids": [consumer_id],
                    "mapped_node_ids": [*mapped_node_ids, consumer_id],
                    "module_node_ids": module_node_ids,
                    "module_source_files": module_source_files,
                    "status": "mapped_module_namespace_import_relation",
                    "unresolved": False,
                    "symbol": symbol,
                    "module": module or None,
                    "namespace": namespace,
                    "module_import_owner_file": consumer_file,
                }
            return {
                "root_node_ids": [unresolved_id],
                "mapped_node_ids": mapped_node_ids,
                "module_node_ids": module_node_ids,
                "module_source_files": module_source_files,
                "status": "unresolved_module_namespace_import_owner",
                "unresolved": True,
                "symbol": symbol,
                "module": module or None,
                "namespace": namespace,
            }
        exact_dependency_owner_ids: list[str] = []
        exact_dependency_counterpart_ids: list[str] = []
        exact_dependency_provider_ids: list[str] = []
        structural_call_provider: dict[str, Any] = {}
        if exact_source_contract:
            structural_report = self.index_structural_file(
                "old",
                contract_source_file,
            )
            dependency_owners = self._minimal_structural_dependency_owners(
                structural_report,
                symbol=symbol,
                line=int(source_contract_evidence.get("line") or 0),
                column=int(source_contract_evidence.get("column") or 0),
            )
            for owner in dependency_owners:
                owner_contract = str(
                    owner.get("contract_key") or owner.get("key") or ""
                )
                owner_kind = str(
                    owner.get("kind") or owner.get("entity_kind") or ""
                )
                semantic_owners = canonical_entity_records(
                    entity
                    for entity in self.entities_by_file.get(
                        ("old", contract_source_file),
                        [],
                    )
                    if str(
                        entity.get("contract_key")
                        or entity.get("key")
                        or ""
                    )
                    == owner_contract
                    and str(
                        entity.get("kind")
                        or entity.get("entity_kind")
                        or ""
                    )
                    == owner_kind
                    and bool(entity.get("is_definition"))
                    and _entity_has_source_extent(entity)
                )
                selected_owner = (
                    semantic_owners[0]
                    if len(semantic_owners) == 1
                    else owner
                )
                owner_id = self._retain_diagnostic_entity_node(
                    selected_owner
                )
                if not owner_id:
                    continue
                self.graph.add_edge(
                    diagnostic_id,
                    owner_id,
                    "diagnoses",
                    budget_exempt=True,
                    mapping_kind=(
                        "exact_source_contract_dependency_owner"
                    ),
                )
                self.graph.add_edge(
                    owner_id,
                    unresolved_id,
                    "diagnostic_contract_dependency",
                    budget_exempt=True,
                    contract_key=unresolved_contract_key,
                )
                exact_dependency_owner_ids.append(owner_id)
                exact_dependency_counterpart_ids.extend(
                    self._retain_exact_dependency_owner_counterparts(
                        selected_owner
                    )
                )
            mapped_node_ids.extend(exact_dependency_owner_ids)
            mapped_node_ids.extend(
                exact_dependency_counterpart_ids
            )
            counterpart_reports: dict[tuple[str, str], dict[str, Any]] = {}
            for counterpart_id in _ordered_unique(
                exact_dependency_counterpart_ids
            ):
                counterpart = self.graph.nodes.get(counterpart_id) or {}
                counterpart_tree = str(counterpart.get("tree") or "")
                counterpart_file = normalize_relpath(
                    str(counterpart.get("file") or "")
                )
                if not counterpart_tree or not counterpart_file:
                    continue
                key = (counterpart_tree, counterpart_file)
                if key not in counterpart_reports:
                    root = self.root_for_tree(counterpart_tree)
                    driver_root = self.driver_root_for_tree(counterpart_tree)
                    counterpart_reports[key] = (
                        parse_source_file(
                            self.contexts.get(counterpart_tree),
                            root.resolve(),
                            driver_root,
                            counterpart_file,
                            detailed_processing=True,
                            tree=counterpart_tree,
                        )
                        if root is not None and driver_root is not None
                        else {}
                    )
                report = counterpart_reports[key]
                for reference in (
                    report.get("references") or []
                    if report.get("parse_ok")
                    else []
                ):
                    reference_owner = (
                        reference.get("owner_entity")
                        if isinstance(reference.get("owner_entity"), dict)
                        else {}
                    )
                    owner_extent = reference_owner.get("extent") or {}
                    counterpart_extent = counterpart.get("extent") or {}
                    if (
                        str(reference.get("symbol") or "") != symbol
                        or str(
                            reference.get("owner_entity_id")
                            or (
                                reference.get("owner_entity")
                                if isinstance(
                                    reference.get("owner_entity"),
                                    dict,
                                )
                                else {}
                            ).get("id")
                            or ""
                        )
                        != counterpart_id
                        and not (
                            str(reference_owner.get("contract_key") or "")
                            == str(counterpart.get("contract_key") or "")
                            and str(reference_owner.get("kind") or "").lower()
                            == str(counterpart.get("entity_kind") or "").lower()
                            and all(
                                int(owner_extent.get(field) or 0)
                                == int(counterpart_extent.get(field) or 0)
                                for field in (
                                    "start_line",
                                    "start_column",
                                    "end_line",
                                    "end_column",
                                )
                            )
                            and int(owner_extent.get("start_line") or 0) > 0
                        )
                    ):
                        continue
                    target = (
                        reference.get("referenced_entity")
                        if isinstance(
                            reference.get("referenced_entity"),
                            dict,
                        )
                        else {}
                    )
                    target_id = self._retain_diagnostic_entity_node(
                        target
                    )
                    if not target_id or not _entity_has_source_extent(target):
                        continue
                    self.graph.add_edge(
                        unresolved_id,
                        target_id,
                        "candidate_resolution",
                        budget_exempt=True,
                        resolution_source=(
                            "exact_donor_owner_typed_reference"
                        ),
                    )
                    exact_dependency_provider_ids.append(target_id)
            mapped_node_ids.extend(exact_dependency_provider_ids)
            if not exact_dependency_provider_ids and exact_dependency_counterpart_ids:
                structural_call_provider = self._map_exact_build_owned_call_provider(
                    unresolved_id=unresolved_id,
                    counterpart_ids=exact_dependency_counterpart_ids,
                    symbol=symbol,
                )
                exact_dependency_provider_ids.extend(
                    structural_call_provider.get("provider_ids") or []
                )
                mapped_node_ids.extend(exact_dependency_provider_ids)
        consumer_owner_ids: list[str] = []
        for relative_path in module_source_files:
            for reference in self.references_by_file.get(("old", relative_path), []):
                if str(reference.get("symbol") or "") != symbol:
                    continue
                reference_id = self._retain_diagnostic_reference_node(reference)
                if not reference_id:
                    continue
                self.graph.add_edge(
                    diagnostic_id,
                    reference_id,
                    "diagnoses",
                    budget_exempt=True,
                )
                mapped_node_ids.append(reference_id)
                owner = (
                    reference.get("owner_entity")
                    if isinstance(reference.get("owner_entity"), dict)
                    else {}
                )
                owner_id = str(
                    owner.get("id")
                    or owner.get("entity_id")
                    or reference.get("owner_entity_id")
                    or ""
                )
                if owner_id and owner_id in self.graph.nodes:
                    consumer_owner_ids.append(owner_id)

        provider_definition_queries: list[dict[str, Any]] = []
        provider_definition_queries.extend(
            structural_call_provider.get("queries") or []
        )
        query_trees = (
            []
            if exact_source_contract
            else [
                tree
                for tree in ["old", *sorted(self.contexts)]
                if tree == "old" or tree.startswith("donor_")
            ]
        )
        for tree in _ordered_unique(query_trees):
            dependency_frontier = self.compile_dependency_frontier(
                tree,
                within_root="",
            )
            frontier_files = _ordered_unique(
                dependency_frontier.get("files") or []
            )
            if not frontier_files:
                continue
            candidates = self.exact_token_candidates_in_files(
                tree,
                [symbol],
                frontier_files,
            )
            candidate_files = _ordered_unique(
                (candidates.get("candidates_by_symbol") or {}).get(symbol, [])
            )
            if not candidate_files:
                provider_definition_queries.append(
                    {
                        "tree": tree,
                        "frontier_file_count": len(frontier_files),
                        "candidate_files": [],
                        "resolved": False,
                    }
                )
                continue
            query = self.query_definition_batches(
                tree=tree,
                symbols=[symbol],
                candidate_files=candidate_files,
                candidate_source=(
                    "case_dependency_frontier_exact_token"
                ),
                max_files=len(candidate_files),
            )
            provider_definition_queries.append(
                {
                    "tree": tree,
                    "frontier_file_count": len(frontier_files),
                    "candidate_files": candidate_files,
                    "resolved": bool(query.get("resolved")),
                    "resolution_source": query.get("resolution_source"),
                    "matches": list(query.get("matches") or []),
                }
            )

        indexed_providers = canonical_entity_records(
            self.entities_by_name.get(symbol, [])
        )
        declaration_evidence = issue.get("source_symbol_evidence") or {}
        missing_declaration = (
            issue.get("kind") == "compile_missing_declaration"
            and declaration_evidence.get("authority")
            == "compiler_missing_declaration"
            and declaration_evidence.get("relation")
            == "requires_visible_declaration"
        )
        structural_declarations: list[dict[str, Any]] = []
        diagnosed_source_file = (
            self._normalize_issue_file(issue) if missing_declaration else ""
        )
        if missing_declaration and not any(
            provider.get("is_definition") is False
            and provider.get("token_count") != 0
            and _entity_has_source_extent(provider)
            and (
                str(provider.get("tree") or "").startswith("donor_")
                and Path(str(provider.get("file") or "")).suffix.lower()
                in {".h", ".hpp"}
                or not diagnosed_source_file
            )
            for provider in indexed_providers
        ):
            if diagnosed_source_file:
                self.index_file("old", diagnosed_source_file)
            direct_headers = {
                normalize_relpath(str(include.get("target_file") or ""))
                for source_file in _ordered_unique([
                    *module_source_files,
                    diagnosed_source_file,
                ])
                for include in self.includes_by_file.get(("old", source_file), [])
                if isinstance(include, dict)
                and normalize_relpath(str(include.get("target_file") or ""))
                .startswith("include/")
            }
            for tree in sorted(self.contexts):
                if not str(tree).startswith("donor_"):
                    continue
                direct_candidates = self.exact_token_candidates_in_files(
                    tree,
                    [symbol],
                    direct_headers,
                )
                candidate_files = _ordered_unique([
                    *[
                        path
                        for query_record in provider_definition_queries
                        if str(query_record.get("tree") or "") == tree
                        for path in query_record.get("candidate_files") or []
                    ],
                    *((direct_candidates.get("candidates_by_symbol") or {}).get(symbol) or []),
                ])
                for candidate_file in candidate_files:
                    if Path(candidate_file).suffix.lower() not in {".h", ".hpp"}:
                        continue
                    report = self.index_structural_file(tree, candidate_file)
                    if not report.get("parse_ok"):
                        continue
                    structural_declarations.extend(
                        entity
                        for entity in report.get("entities") or []
                        if entity.get("is_definition") is False
                        and str(entity.get("symbol") or "") == symbol
                        and str(entity.get("contract_key") or "")
                        in {f"function:{symbol}", f"variable:{symbol}"}
                        and str(entity.get("cursor_kind") or "")
                        == "TREE_SITTER_DECLARATION"
                        and _entity_has_source_extent(entity)
                        and not self._extent_overlaps_error_nodes(
                            dict(entity.get("extent") or {}),
                            report.get("error_nodes") or [],
                        )
                    )
        providers = canonical_entity_records(
            [*indexed_providers, *structural_declarations]
        )
        definitions = [
            provider
            for provider in providers
            if provider.get("is_definition") and _entity_has_source_extent(provider)
        ]
        old_definitions = [provider for provider in definitions if provider.get("tree") == "old"]
        donor_definitions = [
            provider
            for provider in definitions
            if str(provider.get("tree") or "").startswith("donor_")
        ]
        selected_providers = (
            []
            if exact_source_contract
            else old_definitions or donor_definitions
        )
        if missing_declaration:
            declarations = canonical_entity_records(build_conditioned_provider_records(
                provider for provider in providers
                if str(provider.get("tree") or "").startswith("donor_")
                and provider.get("is_definition") is False
                and provider.get("cursor_kind")
                in {"FUNCTION_DECL", "VAR_DECL", "TREE_SITTER_DECLARATION"}
                and _entity_has_source_extent(provider)
                and provider.get("token_count") != 0
            ))
            if declarations:
                selected_providers = declarations
        selected_provider_ids: list[str] = []
        provider_owner_evidence: list[dict[str, Any]] = []
        for provider in selected_providers:
            provider_id = self._retain_diagnostic_entity_node(provider)
            if not provider_id:
                continue
            self.graph.add_edge(
                unresolved_id,
                provider_id,
                "candidate_resolution",
                budget_exempt=True,
            )
            mapped_node_ids.append(provider_id)
            selected_provider_ids.append(provider_id)
            provider_tree = str(provider.get("tree") or "")
            provider_file = normalize_relpath(
                str(provider.get("file") or "")
            )
            old_root = self.root_for_tree("old")
            donor_root = self.root_for_tree(provider_tree)
            if (
                provider_tree.startswith("donor_")
                and provider_file
                and old_root is not None
                and donor_root is not None
                and (old_root / provider_file).is_file()
                and (donor_root / provider_file).is_file()
            ):
                provider_owner_evidence.append(
                    {
                        "tree": provider_tree,
                        "mapping_kind": "symbol_provider_identity",
                        "target_entity_id": provider_id,
                        "contract_owner_file": provider_file,
                        "owner_alignment": self._owner_entity_alignment(
                            donor_tree=provider_tree,
                            source_file=self._normalize_issue_file(issue),
                            target_entity_id=provider_id,
                            target_file=provider_file,
                        ),
                    }
                )

        root_node_ids = _symbol_diagnostic_root_node_ids(
            exact_source_contract=(exact_source_contract and not missing_declaration),
            exact_dependency_provider_ids=(
                exact_dependency_provider_ids
            ),
            selected_provider_ids=selected_provider_ids,
            consumer_owner_ids=consumer_owner_ids,
        )

        unresolved = not root_node_ids
        return {
            "root_node_ids": sorted(set(root_node_ids)) if root_node_ids else [unresolved_id],
            "mapped_node_ids": _ordered_unique(mapped_node_ids),
            "module_node_ids": module_node_ids,
            "module_source_files": module_source_files,
            "status": (
                "mapped_exact_source_dependency_provider"
                if exact_source_contract and root_node_ids
                else "mapped_unresolved_contract"
                if exact_source_contract
                else "mapped_symbol_consumer"
                if consumer_owner_ids and not selected_providers
                else "unresolved_symbol_provider"
                if unresolved
                else "mapped_symbol_provider"
            ),
            "unresolved": unresolved,
            "exact_contract_mapped": exact_source_contract,
            "unresolved_contract_node_id": (
                unresolved_id if exact_source_contract else None
            ),
            "symbol": symbol,
            "module": module or None,
            "provider_owner_evidence": provider_owner_evidence,
            "provider_definition_queries": provider_definition_queries,
            "exact_dependency_owner_ids": _ordered_unique(
                exact_dependency_owner_ids
            ),
            "exact_dependency_counterpart_ids": _ordered_unique(
                exact_dependency_counterpart_ids
            ),
            "exact_dependency_provider_ids": _ordered_unique(
                exact_dependency_provider_ids
            ),
        }

    def _map_graph_contract_anchor(
        self,
        *,
        diagnostic_id: str,
        anchor: dict[str, Any],
    ) -> dict[str, Any]:
        """Map controller-produced contract evidence to exact AST entities."""

        if (
            str(anchor.get("schema") or "")
            != "typed_apply_graph_contract_anchor_v1"
            or str(anchor.get("authority") or "")
            != "deterministic_apply_checker"
        ):
            return {
                "status": "unsupported_contract_anchor",
                "root_node_ids": [],
                "mapped_node_ids": [],
            }
        relative_path = normalize_relpath(
            str(anchor.get("file") or "")
        )
        contract_key = str(
            anchor.get("contract_key") or ""
        ).strip()
        symbol = str(anchor.get("symbol") or "").strip()
        entity_kind = str(
            anchor.get("entity_kind") or ""
        ).strip().lower()
        tree_role = str(
            anchor.get("tree_role") or "donor"
        ).strip()
        if not relative_path or not (contract_key or symbol):
            return {
                "status": "contract_anchor_incomplete",
                "root_node_ids": [],
                "mapped_node_ids": [],
            }

        if tree_role == "donor":
            trees = sorted(
                tree
                for tree in self.contexts
                if tree.startswith("donor_")
            )
        elif tree_role == "old":
            trees = ["old"]
        else:
            return {
                "status": "contract_anchor_tree_role_invalid",
                "root_node_ids": [],
                "mapped_node_ids": [],
            }
        for tree in trees:
            root = self.root_for_tree(tree)
            if root is not None and (root / relative_path).is_file():
                self.index_file(tree, relative_path)

        extent = (
            dict(anchor.get("extent") or {})
            if isinstance(anchor.get("extent"), dict)
            else {}
        )
        start_line = _int_or_zero(extent.get("start_line"))
        end_line = _int_or_zero(extent.get("end_line"))
        candidates = [
            entity
            for tree in trees
            for entity in self.entities_by_file.get(
                (tree, relative_path),
                [],
            )
            if (
                not contract_key
                or str(
                    entity.get("contract_key")
                    or entity.get("key")
                    or ""
                )
                == contract_key
            )
            and (
                not symbol
                or str(entity.get("symbol") or "") == symbol
            )
            and (
                not entity_kind
                or str(
                    entity.get("entity_kind")
                    or entity.get("kind")
                    or ""
                ).lower()
                == entity_kind
            )
            and (
                not start_line
                or int(
                    (entity.get("extent") or {}).get(
                        "start_line"
                    )
                    or 0
                )
                == start_line
            )
            and (
                not end_line
                or int(
                    (entity.get("extent") or {}).get(
                        "end_line"
                    )
                    or 0
                )
                == end_line
            )
        ]
        candidates = canonical_entity_records(candidates)
        root_node_ids: list[str] = []
        for entity in candidates:
            entity_id = self._retain_diagnostic_entity_node(
                entity
            )
            if not entity_id:
                continue
            self.graph.add_edge(
                diagnostic_id,
                entity_id,
                "diagnoses_typed_contract",
                budget_exempt=True,
                authority="deterministic_apply_checker",
            )
            root_node_ids.append(entity_id)
        return {
            "status": (
                "mapped_unique_typed_contract"
                if len(root_node_ids) == 1
                else "mapped_typed_contract_set"
                if root_node_ids
                else "typed_contract_not_found"
            ),
            "root_node_ids": _ordered_unique(root_node_ids),
            "mapped_node_ids": _ordered_unique(root_node_ids),
            "anchor": dict(anchor),
        }

    def _map_related_diagnostic_locations(
        self,
        *,
        diagnostic_id: str,
        location_chain: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        """Map typed compiler-note locations into the same causal graph."""

        mapped_node_ids: list[str] = []
        root_node_ids: list[str] = []
        evidence: list[dict[str, Any]] = []
        old_root = self.root_for_tree("old")
        if old_root is None:
            return {
                "mapped_node_ids": [],
                "root_node_ids": [],
                "evidence": [],
            }

        seen_locations: set[tuple[str, int, int, str]] = set()
        for site in location_chain:
            relation = str(site.get("kind") or "")
            if relation != "previous_declaration":
                continue
            relative_path = normalize_relpath(str(site.get("file") or ""))
            line = _int_or_zero(site.get("line"))
            column = _int_or_zero(site.get("column"))
            identity = (relative_path, line, column, relation)
            if (
                not relative_path
                or identity in seen_locations
                or not (old_root / relative_path).is_file()
            ):
                continue
            seen_locations.add(identity)
            self.index_file("old", relative_path)

            file_id = self._retain_diagnostic_file_node("old", relative_path)
            if file_id:
                self.graph.add_edge(
                    diagnostic_id,
                    file_id,
                    "occurs_at",
                    budget_exempt=True,
                    diagnostic_relation=relation,
                )
                mapped_node_ids.append(file_id)

            position = _source_position_at_display_column(
                old_root,
                relative_path,
                line,
                column,
            )
            semantic_column = int(
                (position or {}).get("source_column") or column
            )
            containing = [
                entity
                for entity in self.entities_by_file.get(("old", relative_path), [])
                if self._contains_position(
                    entity.get("extent") or {},
                    line,
                    semantic_column,
                )
            ]
            note_symbol = str(site.get("symbol") or "").strip()
            symbol_matches = [
                entity
                for entity in containing
                if note_symbol
                and str(entity.get("symbol") or entity.get("name") or "")
                == note_symbol
            ]
            ordered = sorted(symbol_matches or containing, key=self._extent_order)
            if ordered:
                minimum_extent = self._extent_order(ordered[0])[:3]
                ordered = [
                    entity
                    for entity in ordered
                    if self._extent_order(entity)[:3] == minimum_extent
                ]

            site_root_ids: list[str] = []
            for entity in ordered:
                entity_id = self._retain_diagnostic_entity_node(entity)
                if not entity_id:
                    continue
                self.graph.add_edge(
                    diagnostic_id,
                    entity_id,
                    "diagnoses",
                    budget_exempt=True,
                    diagnostic_relation=relation,
                )
                mapped_node_ids.append(entity_id)
                root_node_ids.append(entity_id)
                site_root_ids.append(entity_id)
            evidence.append(
                {
                    "kind": relation,
                    "file": relative_path,
                    "line": line,
                    "column": column,
                    "symbol": note_symbol or None,
                    "mapped_root_node_ids": site_root_ids,
                }
            )

        return {
            "mapped_node_ids": _ordered_unique(mapped_node_ids),
            "root_node_ids": _ordered_unique(root_node_ids),
            "evidence": evidence,
        }

    def _map_macro_pasted_provider(
        self,
        *,
        diagnostic_id: str,
        relative_path: str,
        line: int,
        column: int,
        diagnostic_owner_node_ids: set[str],
    ) -> dict[str, Any]:
        """Follow an exact macro invocation to one absent generated provider."""

        old_root = self.root_for_tree("old")
        if old_root is None:
            return {}
        variable_owners = [
            node
            for owner_id in diagnostic_owner_node_ids
            if (node := self.graph.nodes.get(owner_id))
            and str(node.get("kind") or "").lower() == "variable"
            and str(node.get("tree") or "") == "old"
            and self._contains_position(node.get("extent") or {}, line, column)
        ]
        if not variable_owners:
            return {}
        old_includes = {
            normalize_relpath(str(item.get("target_file") or ""))
            for item in self.includes_by_file.get(("old", relative_path), [])
            if item.get("target_file")
        }
        matches: list[tuple[dict[str, Any], dict[str, Any], str, str]] = []
        for reference in self.references_by_file.get(("old", relative_path), []):
            if (
                str(reference.get("cursor_kind") or "")
                != "MACRO_INSTANTIATION"
                or int(reference.get("line") or 0) != line
            ):
                continue
            extent = reference.get("extent") or {}
            start_column = int(extent.get("start_column") or 0)
            end_line = int(extent.get("end_line") or 0)
            end_column = int(extent.get("end_column") or 0)
            if not any(
                self._contains_position(owner.get("extent") or {}, line, start_column)
                and self._contains_position(owner.get("extent") or {}, end_line, end_column)
                for owner in variable_owners
            ):
                continue
            macro = reference.get("referenced_entity") or {}
            if not isinstance(macro, dict) or str(macro.get("kind") or "") != "macro":
                continue
            syntax = _macro_invocation_syntax(
                old_root,
                relative_path,
                line=line,
                column=int(reference.get("column") or 0),
            )
            if syntax is None or syntax.get("name") != reference.get("symbol"):
                continue
            generated_names = _macro_pasted_identifiers(
                (macro.get("implementation") or {}).get("normalized_tokens") or [],
                syntax.get("arguments") or [],
            )
            if not generated_names:
                continue
            argument_names = {
                str(argument.get("text") or "").strip()
                for argument in syntax.get("arguments") or []
                if isinstance(argument, dict)
                and IDENTIFIER_RE.fullmatch(str(argument.get("text") or "").strip())
            }
            for tree in sorted(
                name for name in self.contexts if name.startswith("donor_")
            ):
                self.index_file(tree, relative_path)
                donor_includes = {
                    normalize_relpath(str(item.get("target_file") or ""))
                    for item in self.includes_by_file.get((tree, relative_path), [])
                    if item.get("target_file")
                }
                included_in_both = sorted(old_includes & donor_includes)
                if not included_in_both or not argument_names:
                    continue
                old_explicit_providers = self.exact_token_candidates_in_files(
                    "old", generated_names, sorted(old_includes)
                ).get("candidates_by_symbol") or {}
                argument_candidates = self.exact_token_candidates_in_files(
                    tree, argument_names, included_in_both
                ).get("candidates_by_symbol") or {}
                candidate_files = sorted({
                    path
                    for paths in argument_candidates.values()
                    for path in paths
                })
                old_providers: set[str] = set()
                donor_providers: list[tuple[dict[str, Any], str]] = []
                incomplete_parse = False
                for provider_file in candidate_files:
                    donor_report = self.index_file_from_translation_unit(
                        tree, provider_file, relative_path
                    )
                    old_report = self.index_file_from_translation_unit(
                        "old", provider_file, relative_path
                    )
                    if not donor_report.get("parse_ok") or not old_report.get("parse_ok"):
                        incomplete_parse = True
                        continue
                    old_providers.update(
                        str(entity.get("symbol") or "")
                        for entity in old_report.get("entities") or []
                        if bool(entity.get("is_definition"))
                        and str(entity.get("symbol") or "") in generated_names
                    )
                    donor_providers.extend(
                        (entity, provider_file)
                        for entity in donor_report.get("entities") or []
                        if bool(entity.get("is_definition"))
                        and str(entity.get("symbol") or "") in generated_names
                        and _entity_has_source_extent(entity)
                    )
                if incomplete_parse:
                    continue
                for generated_name in sorted(generated_names):
                    if (
                        generated_name in old_providers
                        or old_explicit_providers.get(generated_name)
                    ):
                        continue
                    for entity, provider_file in donor_providers:
                        if str(entity.get("symbol") or "") == generated_name:
                            matches.append(
                                (reference, entity, generated_name, provider_file)
                            )
        if len(matches) != 1:
            return {}
        reference, provider, generated_name, provider_file = matches[0]
        reference_id = self._retain_diagnostic_reference_node(reference)
        provider_id = self._retain_diagnostic_entity_node(provider)
        if not reference_id or not provider_id:
            return {}
        self.graph.add_edge(
            diagnostic_id,
            reference_id,
            "diagnoses_macro_generated_provider",
            budget_exempt=True,
        )
        self.graph.add_edge(
            reference_id,
            provider_id,
            "macro_token_paste_requires_provider",
            budget_exempt=True,
            generated_symbol=generated_name,
        )
        return {
            "status": "mapped_macro_pasted_provider",
            "root_node_ids": [provider_id],
            "mapped_node_ids": [reference_id, provider_id],
            "evidence": {
                "macro_reference_id": reference_id,
                "generated_symbol": generated_name,
                "donor_provider_file": provider_file,
                "donor_provider_entity_id": provider_id,
                "old_provider_absent": True,
            },
        }

    def _map_macro_argument_contracts(
        self,
        *,
        diagnostic_id: str,
        relative_path: str,
        line: int,
        column: int,
    ) -> dict[str, Any]:
        """Map typed macro arguments to exact structural provider contracts."""

        old_root = self.root_for_tree("old")
        if old_root is None:
            return _diagnostic_mapping_result("unavailable")
        syntax = _macro_invocation_syntax(
            old_root,
            relative_path,
            line=line,
            column=column,
        )
        if syntax is None:
            return _diagnostic_mapping_result(
                "no_call_expression_at_macro_site"
            )
        type_tags = {
            (str(item.get("kind") or ""), str(item.get("name") or ""))
            for item in syntax.get("type_tags") or []
            if isinstance(item, dict)
        }
        argument_identifiers = {
            str(identifier)
            for argument in syntax.get("arguments") or []
            if isinstance(argument, dict)
            for identifier in argument.get("identifiers") or []
            if str(identifier)
        }
        candidates: list[dict[str, Any]] = []
        for identifier in sorted(argument_identifiers):
            for entity in self.entities_by_name.get(identifier, []):
                if (
                    str(entity.get("kind") or "") != "field"
                    or not entity.get("is_definition")
                    or not _entity_has_source_extent(entity)
                ):
                    continue
                container = (
                    dict(entity.get("container") or {})
                    if isinstance(entity.get("container"), dict)
                    else {}
                )
                container_identity = (
                    str(container.get("container_kind") or ""),
                    str(container.get("container_name") or ""),
                )
                if not all(container_identity):
                    contract_parts = str(
                        entity.get("contract_key") or entity.get("key") or ""
                    ).split(":", 3)
                    if len(contract_parts) == 4 and contract_parts[0] == "field":
                        container_identity = (
                            contract_parts[1],
                            contract_parts[2],
                        )
                if container_identity in type_tags:
                    candidates.append(entity)
        candidates_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(
            list
        )
        for entity in canonical_entity_records(candidates):
            contract = str(
                entity.get("contract_key") or entity.get("key") or ""
            )
            if contract:
                candidates_by_contract[contract].append(entity)

        candidate_evidence: list[dict[str, Any]] = []
        selected_candidates: list[dict[str, Any]] = []
        for contract, records in sorted(candidates_by_contract.items()):
            old_records = [
                record
                for record in records
                if str(record.get("tree") or "") == "old"
            ]
            donor_records = [
                record
                for record in records
                if str(record.get("tree") or "").startswith("donor_")
            ]
            selected_for_contract: list[dict[str, Any]] = []
            selection_reason = "old_contract_equivalent"
            if donor_records and not old_records:
                selected_for_contract = donor_records
                selection_reason = "donor_contract_absent_in_old"
            elif donor_records and old_records:
                old_fingerprints = {
                    str(
                        record.get("interface_fingerprint")
                        or record.get("signature_fingerprint")
                        or record.get("source_fingerprint")
                        or ""
                    )
                    for record in old_records
                } - {""}
                changed_donor_records = [
                    record
                    for record in donor_records
                    if str(
                        record.get("interface_fingerprint")
                        or record.get("signature_fingerprint")
                        or record.get("source_fingerprint")
                        or ""
                    )
                    and str(
                        record.get("interface_fingerprint")
                        or record.get("signature_fingerprint")
                        or record.get("source_fingerprint")
                        or ""
                    )
                    not in old_fingerprints
                ]
                if old_fingerprints and changed_donor_records:
                    selected_for_contract = changed_donor_records
                    selection_reason = "donor_contract_differs_from_old"
            selected_candidates.extend(selected_for_contract)
            candidate_evidence.append(
                {
                    "contract_key": contract,
                    "old_entity_ids": [
                        record.get("id") for record in old_records
                    ],
                    "donor_entity_ids": [
                        record.get("id") for record in donor_records
                    ],
                    "selected_entity_ids": [
                        record.get("id")
                        for record in selected_for_contract
                    ],
                    "selection_reason": selection_reason,
                }
            )
        unique_candidates = canonical_entity_records(selected_candidates)
        if not unique_candidates:
            return _diagnostic_mapping_result(
                "no_missing_or_changed_typed_field_contract",
                evidence={
                    "mapping_kind": "macro_argument_syntax",
                    "invocation": syntax,
                    "candidate_contracts": candidate_evidence,
                },
            )

        invocation_id = stable_id(
            "macro-invocation",
            "old",
            relative_path,
            str(line),
            str(column),
            str(syntax.get("start_offset") or 0),
            str(syntax.get("end_offset") or 0),
        )
        self.graph.add_node(
            invocation_id,
            "MacroInvocation",
            budget_exempt=True,
            tree="old",
            file=relative_path,
            line=line,
            column=column,
            name=syntax.get("name"),
            argument_count=len(syntax.get("arguments") or []),
        )
        self.graph.add_edge(
            diagnostic_id,
            invocation_id,
            "diagnoses_macro_invocation",
            budget_exempt=True,
        )
        mapped_node_ids = [invocation_id]
        root_node_ids: list[str] = []
        for entity in unique_candidates:
            entity_id = self._retain_diagnostic_entity_node(entity)
            if not entity_id:
                continue
            argument_indexes = [
                int(argument.get("index") or 0)
                for argument in syntax.get("arguments") or []
                if isinstance(argument, dict)
                and str(entity.get("symbol") or "") in {
                    str(identifier)
                    for identifier in argument.get("identifiers") or []
                }
            ]
            self.graph.add_edge(
                invocation_id,
                entity_id,
                "argument_requires_structural_contract",
                budget_exempt=True,
                argument_indexes=argument_indexes,
                contract_key=entity.get("contract_key"),
            )
            mapped_node_ids.append(entity_id)
            root_node_ids.append(entity_id)
        return _diagnostic_mapping_result(
            (
                "mapped_typed_macro_argument_contract"
                if root_node_ids
                else "no_retained_typed_macro_argument_contract"
            ),
            mapped_node_ids=_ordered_unique(mapped_node_ids),
            root_node_ids=_ordered_unique(root_node_ids),
            evidence={
                "mapping_kind": "typed_macro_argument_contract",
                "invocation": syntax,
                "selected_contracts": [
                    {
                        "id": entity.get("id"),
                        "tree": entity.get("tree"),
                        "file": entity.get("file"),
                        "contract_key": entity.get("contract_key"),
                        "container": entity.get("container"),
                    }
                    for entity in unique_candidates
                ],
                "candidate_contracts": candidate_evidence,
            },
        )

    def _map_compiler_suggestions(
        self,
        *,
        diagnostic_id: str,
        suggestions: Iterable[object],
    ) -> dict[str, Any]:
        """Map compiler-authored symbol alternatives to indexed old-tree entities."""

        mapped_node_ids: list[str] = []
        root_node_ids: list[str] = []
        evidence: list[dict[str, Any]] = []
        seen_symbols: set[str] = set()
        for raw_suggestion in suggestions:
            if not isinstance(raw_suggestion, dict):
                continue
            symbol = str(raw_suggestion.get("symbol") or "").strip()
            authority = str(raw_suggestion.get("authority") or "").strip()
            relation = str(
                raw_suggestion.get("relation") or "suggested_contract"
            ).strip()
            if not symbol or authority != "compiler" or symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            candidates = canonical_entity_records(
                record
                for record in self.entities_by_name.get(symbol, [])
                if str(record.get("tree") or "") == "old"
                and _entity_has_source_extent(record)
            )
            definitions = [
                record for record in candidates if bool(record.get("is_definition"))
            ]
            selected = definitions or candidates
            selected_ids: list[str] = []
            for record in selected:
                node_id = self._retain_diagnostic_entity_node(record)
                if not node_id:
                    continue
                self.graph.add_edge(
                    diagnostic_id,
                    node_id,
                    "compiler_suggests_contract",
                    budget_exempt=True,
                    authority=authority,
                    relation=relation,
                    symbol=symbol,
                )
                selected_ids.append(node_id)
                mapped_node_ids.append(node_id)
            unique = len(selected_ids) == 1
            if unique:
                root_node_ids.extend(selected_ids)
            evidence.append(
                {
                    "symbol": symbol,
                    "authority": authority,
                    "relation": relation,
                    "status": (
                        "mapped_unique_old_definition"
                        if unique and definitions
                        else "mapped_unique_old_declaration"
                        if unique
                        else "ambiguous_old_contract"
                        if selected_ids
                        else "not_present_in_indexed_old_tree"
                    ),
                    "candidate_node_ids": selected_ids,
                }
            )
        return _diagnostic_mapping_result(
            (
                "mapped_compiler_suggested_contract"
                if root_node_ids
                else "compiler_suggestion_not_uniquely_mapped"
                if evidence
                else "no_compiler_suggestion"
            ),
            mapped_node_ids=_ordered_unique(mapped_node_ids),
            root_node_ids=_ordered_unique(root_node_ids),
            evidence=evidence,
        )

    @staticmethod
    def _structural_extent_strictly_contains(
        outer: dict[str, Any],
        inner: dict[str, Any],
    ) -> bool:
        """Return whether one parsed declaration physically encloses another."""

        outer_extent = (
            outer.get("extent")
            if isinstance(outer.get("extent"), dict)
            else {}
        )
        inner_extent = (
            inner.get("extent")
            if isinstance(inner.get("extent"), dict)
            else {}
        )
        outer_start = int(outer_extent.get("start_line") or 0)
        outer_end = int(outer_extent.get("end_line") or 0)
        inner_start = int(inner_extent.get("start_line") or 0)
        inner_end = int(inner_extent.get("end_line") or 0)
        return bool(
            outer_start > 0
            and outer_end >= outer_start
            and inner_start > 0
            and inner_end >= inner_start
            and outer_start <= inner_start
            and inner_end <= outer_end
            and (outer_start < inner_start or inner_end < outer_end)
        )

    @staticmethod
    def _structural_container_identity(
        entity: dict[str, Any] | None,
    ) -> tuple[str, str] | None:
        if not isinstance(entity, dict):
            return None
        kind = str(entity.get("kind") or entity.get("entity_kind") or "")
        symbol = str(entity.get("symbol") or entity.get("name") or "")
        return (kind, symbol) if kind and symbol else None

    def _structural_physical_parent(
        self,
        *,
        entities: Iterable[dict[str, Any]],
        subject: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Find the innermost declaration whose source extent encloses subject."""

        candidates = [
            entity
            for entity in entities
            if isinstance(entity, dict)
            and bool(entity.get("is_definition"))
            and self._structural_extent_strictly_contains(entity, subject)
        ]
        if not candidates:
            return None
        return min(candidates, key=self._extent_order)

    def _map_missing_member_container(
        self,
        *,
        diagnostic_id: str,
        issue: dict[str, Any],
        relative_path: str,
        line: int,
        column: int,
    ) -> dict[str, Any]:
        """Attribute an exact missing field to a current type, not its use site."""

        if str(issue.get("kind") or "") != "compile_missing_struct_member":
            return {}
        container = (
            issue.get("member_container")
            if isinstance(issue.get("member_container"), dict)
            else {}
        )
        tag = str(container.get("container_kind") or "")
        name = str(container.get("container_name") or "")
        field = str(issue.get("symbol") or "")
        old_root = self.root_for_tree("old")
        if (
            tag not in {"struct", "union"}
            or not IDENTIFIER_RE.fullmatch(name)
            or not IDENTIFIER_RE.fullmatch(field)
            or old_root is None
            or not relative_path
            or line <= 0
            or column <= 0
        ):
            return {}
        try:
            source_lines = (old_root / relative_path).read_bytes().splitlines(
                keepends=True
            )
        except OSError:
            return {}
        if source_location_spelling_evidence(
            field, source_lines, line=line, column=column
        ).get("status") != "target_spelled_at_source_location":
            return {}

        donor_candidates: list[
            tuple[str, dict[str, Any], dict[str, Any]]
        ] = []
        field_contract = contract_key(
            "field",
            field,
            container={"container_kind": tag, "container_name": name},
        )

        def unresolved(reason: str) -> dict[str, Any]:
            node_id = stable_id(
                "unresolved-member-container", "old", relative_path,
                tag, name, field
            )
            self.graph.add_node(
                node_id,
                "UnresolvedContract",
                budget_exempt=True,
                contract_key=field_contract,
                symbol=field,
                container={"container_kind": tag, "container_name": name},
                status=reason,
            )
            self.graph.add_edge(
                diagnostic_id, node_id, "diagnoses", budget_exempt=True
            )
            return {
                "status": "unresolved_missing_member_container",
                "reason": reason,
                "root_node_ids": [node_id],
                "mapped_node_ids": [node_id],
                "donor_field_contract": field_contract,
                "unresolved": True,
            }

        cursor_kind = f"TREE_SITTER_{tag.upper()}_SPECIFIER"
        for donor_tree in sorted(
            tree for tree in self.contexts if tree.startswith("donor_")
        ):
            frontier = self.compile_dependency_frontier(
                donor_tree, within_root=""
            )
            if frontier.get("status") != "ready":
                continue
            candidates = self.exact_token_candidates_in_files(
                donor_tree, [name, field], frontier.get("files") or []
            ).get("candidates_by_symbol") or {}
            candidate_files = set(candidates.get(name) or []) & set(
                candidates.get(field) or []
            )
            for header_file in sorted(candidate_files):
                donor_report = self.index_structural_file(
                    donor_tree, header_file
                )
                if not donor_report.get("parse_ok"):
                    continue
                donor_entities = donor_report.get("entities") or []
                donor_types = [
                    entity
                    for entity in donor_entities
                    if entity.get("kind") == "type"
                    and entity.get("symbol") == name
                    and entity.get("cursor_kind") == cursor_kind
                    and entity.get("is_definition")
                ]
                donor_fields = [
                    entity
                    for entity in donor_entities
                    if entity.get("kind") == "field"
                    and entity.get("contract_key") == field_contract
                    and entity.get("is_definition")
                ]
                if len(donor_types) != 1 or len(donor_fields) != 1:
                    continue
                if not self._structural_extent_strictly_contains(
                    donor_types[0], donor_fields[0]
                ):
                    continue
                donor_candidates.append(
                    (header_file, donor_types[0], donor_fields[0])
                )
        if len(donor_candidates) != 1:
            return unresolved("donor_field_provider_not_unique")

        header_file, donor_type, donor_field = donor_candidates[0]
        old_report = self.index_structural_file("old", header_file)
        if not old_report.get("parse_ok"):
            return unresolved("current_container_unavailable")
        old_entities = old_report.get("entities") or []
        old_types = [
            entity
            for entity in old_entities
            if entity.get("kind") == "type"
            and entity.get("symbol") == name
            and entity.get("cursor_kind") == cursor_kind
            and entity.get("is_definition")
        ]
        if len(old_types) != 1 or any(
            entity.get("contract_key") == field_contract
            for entity in old_entities
        ):
            return unresolved("current_container_not_unique_or_field_present")
        old_type = old_types[0]
        old_id = self._retain_diagnostic_entity_node(old_type)
        donor_id = self._retain_diagnostic_entity_node(donor_type)
        field_id = self._retain_diagnostic_entity_node(donor_field)
        if not old_id or not donor_id or not field_id:
            return {}
        self.graph.add_edge(
            diagnostic_id,
            old_id,
            "diagnoses",
            budget_exempt=True,
            mapping_kind="exact_missing_member_container",
        )
        self.graph.add_edge(
            old_id, donor_id, "same_contract_as", budget_exempt=True
        )
        self.graph.add_edge(
            field_id, donor_id, "member_of", budget_exempt=True
        )
        return {
            "status": "mapped_missing_member_container",
            "root_node_ids": [old_id],
            "mapped_node_ids": [old_id, donor_id, field_id],
            "old_type_entity_id": old_id,
            "donor_type_entity_id": donor_id,
            "donor_field_entity_id": field_id,
            "donor_field_contract": field_contract,
        }

    def _map_structural_container_mismatch(
        self,
        *,
        diagnostic_id: str,
        relative_path: str,
        line: int,
        column: int,
    ) -> dict[str, Any]:
        """Map a diagnostic to its enclosing declaration when placement diverges.

        A compiler can report a nested or later declaration even though the
        migration fault is the earlier declaration boundary that still encloses
        it.  The existing structural parser already records declaration extents
        for both trees, so prove that relationship by comparing the physical
        parent of the diagnosed declaration with its unique donor counterpart.
        This is intentionally inactive when either side is ambiguous.
        """

        unavailable = _diagnostic_mapping_result(
            "structural_container_relation_unavailable"
        )
        if not relative_path or line <= 0:
            return unavailable

        old_report = self.index_structural_file("old", relative_path)
        if not old_report.get("parse_ok"):
            return unavailable
        old_entities = [
            dict(entity)
            for entity in old_report.get("entities") or []
            if isinstance(entity, dict)
            and bool(entity.get("is_definition"))
            and self._contains_position(
                entity.get("extent") or {}, line, column
            )
        ]
        if not old_entities:
            return unavailable

        donor_tree = next(
            (
                tree
                for tree in sorted(self.contexts)
                if tree.startswith("donor_")
            ),
            "",
        )
        if not donor_tree:
            return unavailable
        donor_report = self.index_structural_file(donor_tree, relative_path)
        if not donor_report.get("parse_ok"):
            return unavailable
        donor_entities = [
            dict(entity)
            for entity in donor_report.get("entities") or []
            if isinstance(entity, dict)
            and bool(entity.get("is_definition"))
        ]

        for subject in sorted(old_entities, key=self._extent_order):
            subject_kind = str(subject.get("kind") or "")
            subject_symbol = str(
                subject.get("symbol") or subject.get("name") or ""
            )
            if not subject_kind or not subject_symbol:
                continue
            old_parent = self._structural_physical_parent(
                entities=old_entities,
                subject=subject,
            )
            if old_parent is None:
                continue
            donor_matches = [
                entity
                for entity in donor_entities
                if str(entity.get("kind") or "") == subject_kind
                and str(entity.get("symbol") or entity.get("name") or "")
                == subject_symbol
                and tuple(entity.get("conditional_context") or [])
                == tuple(subject.get("conditional_context") or [])
            ]
            if len(donor_matches) != 1:
                continue
            donor_parent = self._structural_physical_parent(
                entities=donor_entities,
                subject=donor_matches[0],
            )
            if self._structural_container_identity(
                old_parent
            ) == self._structural_container_identity(donor_parent):
                continue

            subject_id = self._retain_diagnostic_entity_node(subject)
            parent_id = self._retain_diagnostic_entity_node(old_parent)
            if not subject_id or not parent_id:
                continue
            self.graph.add_edge(
                diagnostic_id,
                subject_id,
                "diagnoses",
                budget_exempt=True,
                mapping_kind="cross_version_structural_container_mismatch",
            )
            self.graph.add_edge(
                subject_id,
                parent_id,
                "member_of",
                budget_exempt=True,
                mapping_kind="physical_source_containment",
            )
            return _diagnostic_mapping_result(
                "mapped_structural_container_mismatch",
                root_node_ids=[parent_id],
                mapped_node_ids=[subject_id, parent_id],
                evidence={
                    "mapping_kind": "cross_version_structural_container_mismatch",
                    "old_subject": subject,
                    "old_physical_parent": old_parent,
                    "donor_subject": donor_matches[0],
                    "donor_physical_parent": donor_parent,
                },
            )
        return unavailable

    def _map_missing_object_prerequisite(
        self,
        *,
        diagnostic_id: str,
        issue: dict[str, Any],
        relative_path: str,
    ) -> dict[str, Any]:
        """Map one missing object through exact old and donor Kbuild nodes."""

        if (
            str(issue.get("kind") or "")
            != "make_no_rule_to_make_target"
            or Path(relative_path).suffix != ".o"
        ):
            return {}

        object_path = normalize_relpath(relative_path)
        parent = Path(object_path).parent
        directories: list[str] = []
        while True:
            directory = normalize_relpath(
                "" if str(parent) == "." else parent.as_posix()
            )
            directories.append(directory)
            if str(parent) == ".":
                break
            parent = parent.parent

        for tree in self.contexts:
            for directory in directories:
                self.index_kbuild_directory(tree, directory)

        def matching_objects(tree: str) -> list[str]:
            matches: list[str] = []
            for node_id, node in self.graph.nodes.items():
                if (
                    str(node.get("kind") or "") != "Object"
                    or str(node.get("tree") or "") != tree
                ):
                    continue
                candidate_path = normalize_relpath(
                    str(
                        Path(str(node.get("directory") or ""))
                        / str(node.get("object") or "")
                    )
                )
                if candidate_path == object_path:
                    matches.append(str(node_id))
            return sorted(set(matches))

        old_objects = matching_objects("old")
        if len(old_objects) != 1:
            return {
                "status": "object_relation_unresolved",
                "object_path": object_path,
                "old_object_node_ids": old_objects,
                "donor_source_node_ids": [],
            }

        donor_sources: set[str] = set()
        donor_object_ids: set[str] = set()
        for tree in self.contexts:
            if not tree.startswith("donor_"):
                continue
            for object_id in matching_objects(tree):
                donor_object_ids.add(object_id)
                for edge in self.graph.edges:
                    if (
                        str(edge.get("type") or "") != "compiles_to"
                        or str(edge.get("to") or "") != object_id
                    ):
                        continue
                    source_id = str(edge.get("from") or "")
                    source = self.graph.nodes.get(source_id) or {}
                    source_path = normalize_relpath(
                        str(source.get("path") or source.get("file") or "")
                    )
                    root = self.root_for_tree(tree)
                    if (
                        str(source.get("kind") or "") == "File"
                        and source_path
                        and root is not None
                        and (root / source_path).is_file()
                    ):
                        donor_sources.add(source_id)

        if not donor_sources:
            return {
                "status": "object_relation_unresolved",
                "object_path": object_path,
                "old_object_node_ids": old_objects,
                "donor_object_node_ids": sorted(donor_object_ids),
                "donor_source_node_ids": [],
            }

        old_object_id = old_objects[0]
        for source_id in sorted(donor_sources):
            self.graph.add_edge(
                old_object_id,
                source_id,
                "candidate_resolution",
                budget_exempt=True,
                mapping_kind="same_kbuild_object_prerequisite",
                object_path=object_path,
            )
        self.graph.add_edge(
            diagnostic_id,
            old_object_id,
            "diagnoses_build_relation",
            budget_exempt=True,
            object_path=object_path,
        )
        return {
            "status": "mapped_missing_object_source",
            "object_path": object_path,
            "old_object_node_ids": old_objects,
            "donor_object_node_ids": sorted(donor_object_ids),
            "donor_source_node_ids": sorted(donor_sources),
            "root_node_ids": [old_object_id],
            "mapped_node_ids": [old_object_id, *sorted(donor_sources)],
        }

    def add_diagnostic(self, issue: dict[str, Any]) -> dict[str, Any]:
        event_id = str(issue.get("diagnostic_event_id") or stable_id("diag", repr(issue)))
        self.diagnostic_roots[event_id].clear()
        diagnostic_id = f"diagnostic:{event_id}"
        primary_location = {
            "kind": "primary_diagnostic",
            "file": self._normalize_issue_file(issue),
            "line": _int_or_zero(issue.get("line")),
            "column": _int_or_zero(issue.get("column")),
        }
        location_chain = [primary_location]
        for note in issue.get("related_notes") or []:
            if not isinstance(note, dict):
                continue
            location_chain.append(
                {
                    "kind": str(note.get("kind") or "compiler_note"),
                    "file": self._normalize_issue_file(note),
                    "line": _int_or_zero(note.get("line")),
                    "column": _int_or_zero(note.get("column")),
                    "symbol": note.get("symbol"),
                }
            )
        driver_root = normalize_relpath(
            str(getattr(self.spec, "driver_root", "") or "")
        )
        old_root = self.root_for_tree("old")
        primary_file = str(primary_location.get("file") or "")
        primary_is_driver_source = bool(
            primary_file
            and (
                not driver_root
                or primary_file == driver_root
                or primary_file.startswith(driver_root.rstrip("/") + "/")
            )
            and old_root is not None
            and (old_root / primary_file).is_file()
        )
        macro_invocation_sites = [
            site
            for site in location_chain
            if site.get("kind") == "macro_expansion_note"
            and str(site.get("file") or "")
            and (
                not driver_root
                or str(site["file"]) == driver_root
                or str(site["file"]).startswith(driver_root.rstrip("/") + "/")
            )
            and old_root is not None
            and (old_root / str(site["file"])).is_file()
        ]
        effective_location = (
            primary_location
            if primary_is_driver_source or not macro_invocation_sites
            else macro_invocation_sites[-1]
        )
        relative_path = str(effective_location.get("file") or "")
        line = _int_or_zero(effective_location.get("line"))
        column = _int_or_zero(effective_location.get("column"))
        used_macro_invocation = effective_location is not primary_location
        compiler_suggestions = [
            {
                "symbol": str(item.get("symbol") or "").strip(),
                "relation": str(
                    item.get("relation") or "suggested_contract"
                ).strip(),
                "authority": "compiler",
            }
            for item in issue.get("compiler_suggestions") or []
            if isinstance(item, dict)
            and str(item.get("symbol") or "").strip()
            and str(item.get("authority") or "").strip() == "compiler"
        ]
        diagnostic_added = self.graph.add_node(
            diagnostic_id,
            "Diagnostic",
            budget_exempt=True,
            diagnostic_event_id=event_id,
            file=relative_path or None,
            line=line or None,
            column=column or None,
            primary_file=primary_location.get("file") or None,
            primary_line=primary_location.get("line") or None,
            primary_column=primary_location.get("column") or None,
            effective_location_kind=effective_location.get("kind"),
            location_chain=location_chain,
            route=issue.get("diagnostic_route"),
            repair_relevance=issue.get("repair_relevance"),
            symbol=issue.get("symbol"),
            compiler_suggestions=compiler_suggestions,
            module=issue.get("module"),
            message=str(issue.get("message") or issue.get("raw") or "")[:500],
        )
        if not diagnostic_added:
            raise RuntimeError(f"failed to retain diagnostic graph node {diagnostic_id}")
        mapped_nodes: list[str] = []
        root_nodes: set[str] = set()
        mapping_status = "unmapped"
        donor_position_mapping: dict[str, Any] = {}
        macro_argument_mapping: dict[str, Any] = {}
        baseline_contract_mapping: dict[str, Any] = {}
        symbol_mapping: dict[str, Any] = {}
        graph_contract_anchor_mapping: dict[str, Any] = {}
        compiler_suggestion_mapping: dict[str, Any] = {}
        structural_container_mapping: dict[str, Any] = {}
        missing_member_container_mapping: dict[str, Any] = {}
        callback_binding_mapping: dict[str, Any] = {}
        call_argument_mapping: dict[str, Any] = {}
        macro_pasted_mapping: dict[str, Any] = {}
        local_return_mapping: dict[str, Any] = {}
        focused_diagnostic_parse: dict[str, Any] = {}
        reached_consumer_mapping: dict[str, Any] = {}
        exact_mapping: dict[str, Any] = {
            "mapping_kind": "unavailable",
            "references": [],
            "position": None,
        }
        mapped_references: list[dict[str, Any]] = []
        stable_root_evidence: dict[str, Any] = {}
        effective_file_node_id: str | None = None
        macro_argument_roots: set[str] = set()
        diagnostic_owner_node_ids: set[str] = set()
        diagnostic_transaction_owner_node_ids: set[str] = set()
        references: list[dict[str, Any]] = []

        generated_export_origin = self._map_generated_export_origin(
            diagnostic_id=diagnostic_id,
            relative_path=relative_path,
            line=line,
        )

        missing_object_mapping = self._map_missing_object_prerequisite(
            diagnostic_id=diagnostic_id,
            issue=issue,
            relative_path=relative_path,
        )
        if missing_object_mapping.get("root_node_ids"):
            root_nodes.update(
                missing_object_mapping.get("root_node_ids") or []
            )
            mapped_nodes.extend(
                missing_object_mapping.get("mapped_node_ids") or []
            )
            mapping_status = str(missing_object_mapping.get("status"))

        stable_anchor = (
            issue.get("stable_root_anchor")
            if isinstance(issue.get("stable_root_anchor"), dict)
            else {}
        )
        if stable_anchor:
            anchor_members = [
                dict(item)
                for item in stable_anchor.get("members") or []
                if isinstance(item, dict)
            ] or [dict(stable_anchor)]
            member_evidence: list[dict[str, Any]] = []
            anchored_root_ids: set[str] = set()
            unresolved_members = 0
            for member in anchor_members:
                anchor_tree = str(member.get("tree") or "old")
                anchor_file = normalize_relpath(
                    str(member.get("file") or "")
                )
                anchor_contract = str(
                    member.get("contract_key") or ""
                )
                anchor_kind = str(
                    member.get("entity_kind") or ""
                ).lower()
                anchor_symbol = str(member.get("symbol") or "")
                if (
                    anchor_file
                    and anchor_contract
                    and anchor_kind
                    and anchor_tree in self.contexts
                ):
                    self.index_file(anchor_tree, anchor_file)
                candidates = [
                    entity
                    for entity in self.entities_by_file.get(
                        (anchor_tree, anchor_file),
                        [],
                    )
                    if anchor_contract
                    and str(
                        entity.get("contract_key")
                        or entity.get("key")
                        or ""
                    )
                    == anchor_contract
                    and anchor_kind
                    and str(
                        entity.get("kind")
                        or entity.get("entity_kind")
                        or ""
                    ).lower()
                    == anchor_kind
                    and (
                        not anchor_symbol
                        or str(entity.get("symbol") or "")
                        == anchor_symbol
                    )
                    and (
                        member.get("is_definition") is None
                        or bool(entity.get("is_definition"))
                        == bool(member.get("is_definition"))
                    )
                ]
                member_root_ids = {
                    self._retain_diagnostic_entity_node(candidate)
                    for candidate in candidates
                }
                member_root_ids.discard("")
                if not member_root_ids:
                    unresolved_members += 1
                else:
                    anchored_root_ids.update(member_root_ids)
                member_evidence.append(
                    {
                        "anchor": member,
                        "candidate_count": len(candidates),
                        "root_node_ids": sorted(member_root_ids),
                        "resolved": bool(member_root_ids),
                    }
                )

            all_members_resolved = bool(anchor_members) and not unresolved_members
            anchor_trees = {
                str(item.get("tree") or "old") for item in anchor_members
            }
            if all_members_resolved:
                if len(anchor_members) > 1:
                    status = "mapped_contract_anchor_set"
                elif anchor_trees == {"old"}:
                    status = "mapped_current_tree_contract_anchor"
                else:
                    status = "mapped_donor_contract_anchor"
            else:
                status = "unresolved_contract_anchor_set"
            stable_root_evidence = {
                "status": status,
                "anchor": dict(stable_anchor),
                "member_count": len(anchor_members),
                "resolved_member_count": (
                    len(anchor_members) - unresolved_members
                ),
                "candidate_count": sum(
                    int(item.get("candidate_count") or 0)
                    for item in member_evidence
                ),
                "members": member_evidence,
                # An atomic root set never degrades to partial authority.
                "root_node_ids": (
                    sorted(anchored_root_ids)
                    if all_members_resolved
                    else []
                ),
            }

        if relative_path:
            if old_root is not None and (old_root / relative_path).is_file():
                self.index_file("old", relative_path)
            if (
                old_root is not None
                and relative_path.endswith((".h", ".hpp"))
                and not any(
                    int(reference.get("line") or 0) == line
                    for reference in self.references_by_file.get(
                        ("old", relative_path), []
                    )
                )
            ):
                trigger_files = [
                    *list(issue.get("trigger_files") or []),
                    *[
                        entry.get("file")
                        for entry in issue.get("include_stack") or []
                        if isinstance(entry, dict)
                    ],
                ]
                for raw_trigger in trigger_files:
                    translation_unit = self._normalize_issue_file(
                        {"file": raw_trigger}
                    )
                    if (
                        not translation_unit.endswith((".c", ".cc", ".cpp"))
                        or not (old_root / translation_unit).is_file()
                    ):
                        continue
                    report = self.index_file_from_translation_unit(
                        "old", relative_path, translation_unit
                    )
                    if (
                        report.get("parse_ok")
                        and any(
                            int(reference.get("line") or 0) == line
                            for reference in self.references_by_file.get(
                                ("old", relative_path), []
                            )
                        )
                    ):
                        focused_diagnostic_parse = {
                            "translation_unit_file": translation_unit,
                            "file": relative_path,
                            "parse_ok": True,
                        }
                        break
            old_file_id = self._retain_diagnostic_file_node("old", relative_path)
            if old_file_id:
                effective_file_node_id = old_file_id
                self.graph.add_edge(
                    diagnostic_id,
                    old_file_id,
                    "occurs_at",
                    budget_exempt=True,
                )
                mapped_nodes.append(old_file_id)
                if mapping_status == "unmapped":
                    mapping_status = "mapped_file"

            includes = [
                include
                for include in self.includes_by_file.get(("old", relative_path), [])
                if not line or int(include.get("line") or 0) == line
            ]
            for include in includes:
                include_id = self._retain_diagnostic_include_node(include)
                if include_id:
                    self.graph.add_edge(
                        diagnostic_id,
                        include_id,
                        "diagnoses",
                        budget_exempt=True,
                    )
                    mapped_nodes.append(include_id)
                target = normalize_relpath(str(include.get("target_file") or ""))
                if target:
                    root_nodes.add(f"file:old:{target}")
                    mapping_status = "mapped_include_dependency"
                    continue
                include_name = normalize_relpath(str(include.get("name") or ""))
                if include_id and include_name:
                    unresolved_id = stable_id("unresolved-include", "old", include_name)
                    self.graph.add_node(
                        unresolved_id,
                        "UnresolvedInclude",
                        budget_exempt=True,
                        name=include_name,
                        status="unresolved_at_diagnostic_location",
                    )
                    self.graph.add_edge(
                        include_id,
                        unresolved_id,
                        "unresolved_dependency",
                        budget_exempt=True,
                    )
                    include_candidates = {
                        (
                            str(candidate.get("tree") or ""),
                            normalize_relpath(str(candidate.get("target_file") or "")),
                        )
                        for records in self.includes_by_file.values()
                        for candidate in records
                        if normalize_relpath(str(candidate.get("name") or "")) == include_name
                        and candidate.get("target_file")
                    }
                    for donor_index, donor_root in enumerate(
                        getattr(self.spec, "donor_root_paths", []) or []
                    ):
                        donor_tree = f"donor_{donor_index}"
                        donor_root = Path(donor_root)
                        for donor_include in parse_include_directives(
                            root=donor_root,
                            driver_root=donor_root / self.spec.driver_root,
                            relative_path=relative_path,
                            tree=donor_tree,
                        ):
                            if (
                                donor_include.get("name") == include_name
                                and donor_include.get("delimiter")
                                == include.get("delimiter")
                                and donor_include.get("target_file")
                            ):
                                include_candidates.add(
                                    (
                                        donor_tree,
                                        normalize_relpath(
                                            str(donor_include["target_file"])
                                        ),
                                    )
                                )
                    for tree, candidate_path in sorted(include_candidates):
                        candidate_id = f"file:{tree}:{candidate_path}"
                        self.graph.add_node(
                            candidate_id,
                            "File",
                            budget_exempt=True,
                            tree=tree,
                            path=candidate_path,
                            exists=True,
                        )
                        self.graph.add_edge(
                            unresolved_id,
                            candidate_id,
                            "candidate_resolution",
                            budget_exempt=True,
                        )
                    root_nodes.add(unresolved_id)
                    mapped_nodes.append(unresolved_id)
                    mapping_status = "mapped_unresolved_include"

            references = self.references_by_file.get(("old", relative_path), [])
            old_root = self.root_for_tree("old")
            exact_mapping = (
                _exact_reference_records_at_position(
                    old_root,
                    relative_path,
                    line,
                    column,
                    references,
                )
                if old_root is not None
                else {"mapping_kind": "unavailable", "references": [], "position": None}
            )
            mapped_references = list(exact_mapping.get("references") or [])
            if mapped_references:
                for reference in mapped_references:
                    reference_id = self._retain_diagnostic_reference_node(reference)
                    if reference_id:
                        self.graph.add_edge(
                            diagnostic_id,
                            reference_id,
                            "diagnoses",
                            budget_exempt=True,
                        )
                        mapped_nodes.append(reference_id)
                    target_record = (
                        reference.get("referenced_entity")
                        if isinstance(reference.get("referenced_entity"), dict)
                        else {}
                    )
                    owner_record = (
                        reference.get("owner_entity")
                        if isinstance(reference.get("owner_entity"), dict)
                        else {}
                    )
                    target_id = str(
                        reference.get("referenced_entity_id")
                        or target_record.get("id")
                        or ""
                    )
                    owner_id = str(
                        reference.get("owner_entity_id")
                        or owner_record.get("id")
                        or ""
                    )
                    target_node = self.graph.nodes.get(target_id) or {}
                    if (
                        target_id
                        and target_id in self.graph.nodes
                        and _entity_has_source_extent(target_node)
                    ):
                        if owner_id and owner_id in self.graph.nodes:
                            diagnostic_transaction_owner_node_ids.add(
                                owner_id
                            )
                            self.graph.add_edge(
                                owner_id,
                                target_id,
                                "diagnostic_contract_dependency",
                                budget_exempt=True,
                                reference_id=reference_id,
                                use_kind=reference.get("use_kind"),
                            )
                        root_nodes.add(target_id)
                        continue
                    symbol = str(reference.get("symbol") or "").strip()
                    if reference_id and symbol:
                        reference_contract = str(reference.get("contract_key") or symbol)
                        provider_contract_keys = reference_provider_contract_keys(
                            reference
                        )
                        unresolved_id = stable_id(
                            "unresolved-reference",
                            "old",
                            reference_contract,
                        )
                        self.graph.add_node(
                            unresolved_id,
                            "UnresolvedEntity",
                            budget_exempt=True,
                            spelling=symbol,
                            contract_key=reference_contract,
                            provider_contract_keys=list(provider_contract_keys),
                            entity_kind=reference.get("kind"),
                            status="unresolved_at_diagnostic_location",
                        )
                        self.graph.add_edge(
                            reference_id,
                            unresolved_id,
                            "unresolved_dependency",
                            budget_exempt=True,
                        )
                        if owner_id and owner_id in self.graph.nodes:
                            diagnostic_transaction_owner_node_ids.add(
                                owner_id
                            )
                            self.graph.add_edge(
                                owner_id,
                                unresolved_id,
                                "diagnostic_contract_dependency",
                                budget_exempt=True,
                                reference_id=reference_id,
                                use_kind=reference.get("use_kind"),
                            )
                        self._link_entity_to_semantic_contracts(
                            unresolved_id,
                            provider_contract_keys,
                        )
                        provider_candidates = canonical_entity_records(
                            candidate
                            for provider_contract in provider_contract_keys
                            for candidate in self.entities_by_contract.get(
                                provider_contract,
                                [],
                            )
                        )
                        aligned_candidates: list[dict[str, Any]] = []
                        old_same_file_provider = any(
                            candidate.get("tree") == "old"
                            and normalize_relpath(
                                str(candidate.get("file") or "")
                            ) == relative_path
                            and _entity_has_source_extent(candidate)
                            for candidate in provider_candidates
                        )
                        if not old_same_file_provider:
                            for donor_index, _ in enumerate(
                                getattr(self.spec, "donor_root_paths", []) or []
                            ):
                                donor_report = self.index_structural_file(
                                    f"donor_{donor_index}", relative_path
                                )
                                if not donor_report.get("parse_ok"):
                                    continue
                                aligned_candidates.extend(
                                    candidate
                                    for candidate in donor_report.get("entities") or []
                                    if str(candidate.get("contract_key") or "")
                                    in provider_contract_keys
                                    and _entity_has_source_extent(candidate)
                                    and (
                                        str(candidate.get("kind") or "").lower()
                                        == "macro"
                                        or not self._extent_overlaps_error_nodes(
                                            dict(candidate.get("extent") or {}),
                                            donor_report.get("error_nodes") or [],
                                        )
                                    )
                                )
                        provider_candidates = canonical_entity_records(
                            [*provider_candidates, *aligned_candidates]
                        )
                        for candidate in provider_candidates:
                            if not _entity_has_source_extent(candidate):
                                continue
                            candidate_id = self._retain_diagnostic_entity_node(candidate)
                            if not candidate_id:
                                continue
                            self.graph.add_edge(
                                unresolved_id,
                                candidate_id,
                                "candidate_resolution",
                                budget_exempt=True,
                            )
                        root_nodes.add(unresolved_id)
                        mapped_nodes.append(unresolved_id)
                callback_providers = (
                    _callback_binding_provider_references(
                        mapped_references,
                        references,
                        source_root=old_root,
                    )
                )
                callback_binding_records: list[dict[str, Any]] = []
                for provider_reference in callback_providers:
                    provider_reference_id = (
                        self._retain_diagnostic_reference_node(
                            provider_reference
                        )
                    )
                    provider_entity = (
                        provider_reference.get("referenced_entity")
                        if isinstance(
                            provider_reference.get("referenced_entity"),
                            dict,
                        )
                        else {}
                    )
                    provider_id = self._retain_diagnostic_entity_node(
                        provider_entity
                    )
                    if not provider_reference_id or not provider_id:
                        continue
                    self.graph.add_edge(
                        diagnostic_id,
                        provider_reference_id,
                        "diagnoses_callback_binding_provider",
                        budget_exempt=True,
                    )
                    self.graph.add_edge(
                        provider_reference_id,
                        provider_id,
                        "callback_provider_binding",
                        budget_exempt=True,
                    )
                    binding_relationships: list[
                        tuple[str, tuple[tuple[str, int], ...]]
                    ] = []
                    for field_reference in references:
                        if str(
                            field_reference.get("kind") or ""
                        ).lower() != "field":
                            continue
                        relationship = _callback_binding_relationship(
                            old_root,
                            field_reference,
                            provider_reference,
                        )
                        if relationship is None:
                            continue
                        binding_relationships.append(relationship)
                        field_reference_id = (
                            self._retain_diagnostic_reference_node(
                                field_reference
                            )
                        )
                        if field_reference_id:
                            self.graph.add_edge(
                                field_reference_id,
                                provider_reference_id,
                                "callback_binding_pair",
                                budget_exempt=True,
                            )
                    root_nodes.add(provider_id)
                    for consumer_reference in [provider_reference, *references]:
                        if consumer_reference is not provider_reference and (
                            consumer_reference.get("referenced_entity_id") != provider_id
                            or consumer_reference.get("use_kind") != "call"
                        ):
                            continue
                        consumer = consumer_reference.get("owner_entity") or {}
                        if consumer.get("tree") != "old" or not _entity_has_source_extent(consumer):
                            continue
                        consumer_id = self._retain_diagnostic_entity_node(consumer)
                        if consumer_id and consumer_id != provider_id:
                            self.graph.add_edge(
                                provider_id, consumer_id, "callback_contract_consumer",
                                budget_exempt=True,
                                reference_id=consumer_reference.get("id"),
                                relation="binding" if consumer_reference is provider_reference else "direct_call",
                            )
                    mapped_nodes.extend(
                        [provider_reference_id, provider_id]
                    )
                    binding_kind, binding_branch = (
                        binding_relationships[0]
                        if binding_relationships
                        else ("", ())
                    )
                    callback_binding_records.append(
                        {
                            "provider_reference_id": (
                                provider_reference_id
                            ),
                            "provider_entity_id": provider_id,
                            "owner_entity_id": provider_reference.get(
                                "owner_entity_id"
                            ),
                            "binding_kind": binding_kind,
                            "binding_branch": [
                                {
                                    "cursor_kind": kind,
                                    "child_index": child_index,
                                }
                                for kind, child_index in binding_branch
                            ],
                            "initializer_branch": [
                                {
                                    "cursor_kind": kind,
                                    "child_index": child_index,
                                }
                                for kind, child_index in (
                                    _initializer_branch_key(
                                        provider_reference
                                    )
                                )
                            ],
                        }
                    )
                if callback_binding_records:
                    callback_binding_mapping = {
                        "status": "mapped_callback_binding_provider",
                        "authority": (
                            "exact_typed_callback_binding"
                        ),
                        "records": callback_binding_records,
                    }
                mapping_status = (
                    "mapped_callback_binding_provider"
                    if callback_binding_mapping
                    else (
                        "mapped_exact_reference"
                        if len(mapped_references) == 1
                        else "mapped_exact_reference_set"
                    )
                )

            if used_macro_invocation:
                macro_argument_mapping = self._map_macro_argument_contracts(
                    diagnostic_id=diagnostic_id,
                    relative_path=relative_path,
                    line=line,
                    column=column,
                )
                macro_argument_roots = set(
                    macro_argument_mapping.get("root_node_ids") or []
                )
                if macro_argument_roots:
                    # A type+field pair spelled in the invocation is more
                    # specific than the implementation macro that expanded.
                    root_nodes = macro_argument_roots
                    mapped_nodes.extend(
                        macro_argument_mapping.get("mapped_node_ids") or []
                    )
                    mapping_status = str(
                        macro_argument_mapping.get("status")
                        or "mapped_typed_macro_argument_contract"
                    )

            has_concrete_root = any(
                str((self.graph.nodes.get(root_id) or {}).get("kind") or "")
                not in _UNRESOLVED_PROVIDER_NODE_KINDS
                for root_id in root_nodes
            )
            if not has_concrete_root:
                baseline_symbol = str(
                    exact_mapping.get("identifier")
                    or next(
                        (
                            reference.get("symbol")
                            for reference in mapped_references
                            if reference.get("symbol")
                        ),
                        "",
                    )
                    or ""
                ).strip()
                baseline_contract_mapping = (
                    self._map_baseline_contract_successor(
                        diagnostic_id=diagnostic_id,
                        relative_path=relative_path,
                        symbol=baseline_symbol,
                    )
                    if baseline_symbol
                    else {}
                )
                baseline_roots = set(
                    baseline_contract_mapping.get("root_node_ids") or []
                )
                if baseline_roots:
                    root_nodes = baseline_roots
                    mapped_nodes.extend(
                        baseline_contract_mapping.get("mapped_node_ids")
                        or []
                    )
                    mapping_status = str(
                        baseline_contract_mapping.get("status")
                        or "mapped_baseline_contract_successor"
                    )

            has_concrete_root = any(
                str((self.graph.nodes.get(root_id) or {}).get("kind") or "")
                not in _UNRESOLVED_PROVIDER_NODE_KINDS
                for root_id in root_nodes
            )
            if not has_concrete_root:
                donor_position_mapping = self._map_donor_reference_at_position(
                    diagnostic_id=diagnostic_id,
                    relative_path=relative_path,
                    line=line,
                    column=column,
                )
                donor_roots = set(donor_position_mapping.get("root_node_ids") or [])
                if donor_roots:
                    root_nodes.update(donor_roots)
                    mapped_nodes.extend(donor_position_mapping.get("mapped_node_ids") or [])
                    donor_evidence = list(donor_position_mapping.get("evidence") or [])
                    mapping_kinds = {
                        str(item.get("mapping_kind") or "lexical_identifier")
                        for item in donor_evidence
                        if isinstance(item, dict)
                    }
                    status_base = (
                        "mapped_donor_operator_position"
                        if mapping_kinds == {"operator_operand"}
                        else "mapped_donor_position"
                    )
                    mapping_status = status_base if len(donor_evidence) == 1 else f"{status_base}_set"

            source_position = exact_mapping.get("position") or {}
            semantic_column = int(source_position.get("source_column") or column)
            macro_owner_candidates = canonical_entity_records(
                reference.get("owner_entity")
                for reference in references
                if used_macro_invocation
                and int(reference.get("line") or 0) == line
                and str(reference.get("cursor_kind") or "")
                == "MACRO_INSTANTIATION"
                and isinstance(reference.get("owner_entity"), dict)
                and _entity_has_source_extent(
                    reference.get("owner_entity") or {}
                )
            )
            semantic_owner_candidates = canonical_entity_records(
                entity
                for entity in self.entities_by_file.get(
                    ("old", relative_path),
                    [],
                )
                if self._contains_position(
                    entity.get("extent") or {},
                    line,
                    semantic_column,
                )
            )
            containing = (
                macro_owner_candidates
                if used_macro_invocation
                else semantic_owner_candidates
            )
            if not containing:
                structural_report = self.index_structural_file(
                    "old",
                    relative_path,
                )
                containing = canonical_entity_records(
                    entity
                    for entity in structural_report.get("entities") or []
                    if isinstance(entity, dict)
                    and bool(entity.get("is_definition"))
                    and self._contains_position(
                        entity.get("extent") or {},
                        line,
                        semantic_column,
                    )
                    and not self._extent_overlaps_error_nodes(
                        dict(entity.get("extent") or {}),
                        structural_report.get("error_nodes") or [],
                    )
                )
            if containing:
                ordered_owners = sorted(containing, key=self._extent_order)
                minimum_extent = self._extent_order(ordered_owners[0])[:3]
                owners = [
                    owner
                    for owner in ordered_owners
                    if self._extent_order(owner)[:3] == minimum_extent
                ]
                owner_node_ids: set[str] = set()
                for owner in owners:
                    owner_id = self._retain_diagnostic_entity_node(owner)
                    if not owner_id:
                        continue
                    owner_node_ids.add(owner_id)
                    diagnostic_owner_node_ids.add(owner_id)
                    self.graph.add_edge(
                        diagnostic_id,
                        owner_id,
                        "diagnoses",
                        budget_exempt=True,
                    )
                    mapped_nodes.append(owner_id)
                    if not root_nodes:
                        root_nodes.add(owner_id)
                transaction_owners = canonical_entity_records(
                    owner
                    for owner in ordered_owners
                    if str(owner.get("cursor_kind") or "")
                    in {
                        "FUNCTION_DECL",
                        "CXX_METHOD",
                        "CONSTRUCTOR",
                        "DESTRUCTOR",
                    }
                )
                for owner in transaction_owners:
                    owner_id = self._retain_diagnostic_entity_node(owner)
                    if not owner_id:
                        continue
                    diagnostic_owner_node_ids.add(owner_id)
                    diagnostic_transaction_owner_node_ids.add(owner_id)
                    self.graph.add_edge(
                        diagnostic_id,
                        owner_id,
                        "diagnoses",
                        budget_exempt=True,
                        mapping_kind="exact_ast_repair_transaction_owner",
                    )
                    mapped_nodes.append(owner_id)
                if (
                    used_macro_invocation
                    and not macro_argument_roots
                    and owner_node_ids
                ):
                    # The compiler may report a type error inside a shared
                    # macro definition even though the incompatible operands
                    # come from one concrete invocation. Keep the expanded
                    # provider as evidence, but repair the invoking owner.
                    root_nodes = owner_node_ids
                if mapping_status == "mapped_file":
                    mapping_status = "mapped_entity" if len(owners) == 1 else "mapped_entity_set"
            elif (
                used_macro_invocation
                and not macro_argument_roots
                and effective_file_node_id
            ):
                root_nodes = {effective_file_node_id}

        related_location_mapping = self._map_related_diagnostic_locations(
            diagnostic_id=diagnostic_id,
            location_chain=location_chain[1:],
        )
        related_roots = set(
            related_location_mapping.get("root_node_ids") or []
        )
        if related_roots and not used_macro_invocation:
            root_nodes.update(related_roots)
        mapped_nodes.extend(
            related_location_mapping.get("mapped_node_ids") or []
        )

        trigger_translation_units: list[str] = []
        for item in issue.get("trigger_files") or []:
            normalized_trigger = self._normalize_issue_file({"file": item})
            if (
                normalized_trigger
                and normalized_trigger not in trigger_translation_units
            ):
                trigger_translation_units.append(normalized_trigger)
        trigger_translation_units.sort()
        source_contract_resolution_domain = (
            "translation_units:" + ",".join(trigger_translation_units)
            if trigger_translation_units
            else f"source_file:{relative_path}"
        )
        exact_identifier = str(exact_mapping.get("identifier") or "").strip()
        root_matches_exact_identifier = any(
            str(
                (self.graph.nodes.get(root_id) or {}).get("symbol")
                or (self.graph.nodes.get(root_id) or {}).get("name")
                or ""
            )
            == exact_identifier
            for root_id in root_nodes
        )
        position_symbol_issue = (
            {
                **issue,
                "symbol": exact_identifier,
                "_source_contract_evidence": {
                    "authority": "exact_source_position",
                    "mapping_kind": exact_mapping.get("mapping_kind"),
                    "source_file": relative_path,
                    "line": line,
                    "column": int(
                        (exact_mapping.get("position") or {}).get(
                            "source_column"
                        )
                        or column
                    ),
                    "identifier": exact_identifier,
                    "namespace": "c_identifier_lookup",
                    "resolution_domain": source_contract_resolution_domain,
                },
            }
            if exact_identifier
            and not used_macro_invocation
            and not mapped_references
            and not root_matches_exact_identifier
            else None
        )
        source_symbol_evidence = (
            dict(issue.get("source_symbol_evidence") or {})
            if isinstance(issue.get("source_symbol_evidence"), dict)
            else {}
        )
        compiler_missing_declaration = bool(
            str(issue.get("kind") or "")
            == "compile_missing_declaration"
            and str(issue.get("symbol") or "").strip()
            and source_symbol_evidence.get("authority")
            == "compiler_missing_declaration"
            and source_symbol_evidence.get("relation")
            == "requires_visible_declaration"
        )
        located_unresolved_declaration = bool(
            compiler_missing_declaration
            and not used_macro_invocation
            and mapped_references
            and not any(
                str((self.graph.nodes.get(node_id) or {}).get("symbol") or "")
                == str(issue.get("symbol") or "")
                and str((self.graph.nodes.get(node_id) or {}).get("kind") or "")
                in {"Function", "Variable"}
                for node_id in root_nodes
            )
        )
        symbol_mapping = {}
        if (
            not relative_path
            or used_macro_invocation
            or position_symbol_issue
            or located_unresolved_declaration
        ):
            symbol_mapping = self._map_symbol_diagnostic(
                diagnostic_id=diagnostic_id,
                issue=position_symbol_issue or issue,
            )
        symbol_roots = set(symbol_mapping.get("root_node_ids") or [])
        compiler_missing_type = bool(
            str(issue.get("kind") or "") == "compile_missing_type"
            and str(issue.get("symbol") or "").strip()
            and source_symbol_evidence.get("authority")
            == "compiler_missing_type"
            and source_symbol_evidence.get("relation")
            == "requires_complete_type_definition"
        )
        incomplete_type_cause: dict[str, Any] = {}
        if (
            str(issue.get("kind") or "") == "compile_error"
            and "incomplete type" in str(issue.get("message") or "").lower()
            and not stable_root_evidence
        ):
            type_references = [
                (node_id, self.graph.nodes.get(node_id) or {})
                for node_id in root_nodes | diagnostic_owner_node_ids
                if str((self.graph.nodes.get(node_id) or {}).get("kind") or "")
                == "Type"
                and str((self.graph.nodes.get(node_id) or {}).get("tree") or "")
                == "old"
                and not (self.graph.nodes.get(node_id) or {}).get("is_definition")
            ]
            report = self.index_structural_file("old", relative_path)
            if not type_references and report.get("parse_ok"):
                structural_types = [
                    entity
                    for entity in report.get("entities") or []
                    if entity.get("kind") == "type"
                    and not entity.get("is_definition")
                    and self._contains_position(
                        entity.get("extent") or {}, line, column
                    )
                    and not self._extent_overlaps_error_nodes(
                        dict(entity.get("extent") or {}),
                        report.get("error_nodes") or [],
                    )
                ]
                if len(structural_types) == 1:
                    node_id = self._retain_diagnostic_entity_node(
                        structural_types[0]
                    )
                    type_references = [
                        (node_id, self.graph.nodes.get(node_id) or {})
                    ]
            if len(type_references) == 1:
                type_name = str(type_references[0][1].get("symbol") or "")
                type_contract = str(
                    type_references[0][1].get("contract_key") or ""
                )
                if IDENTIFIER_RE.fullmatch(type_name):
                    declarations = [
                        entity
                        for entity in report.get("entities") or []
                        if entity.get("kind") == "variable"
                        and entity.get("is_definition")
                        and self._contains_position(
                            entity.get("extent") or {}, line, column
                        )
                    ]
                    if len(declarations) == 1:
                        type_issue = {
                            **issue,
                            "kind": "compile_missing_type",
                            "symbol": type_name,
                            "source_symbol_evidence": {
                                "authority": "compiler_missing_type",
                                "relation": "requires_complete_type_definition",
                            },
                        }
                        type_mapping = self._map_symbol_diagnostic(
                            diagnostic_id=diagnostic_id,
                            issue=type_issue,
                        )
                        provider_roots = [
                            node_id
                            for node_id in type_mapping.get("root_node_ids") or []
                            if str((self.graph.nodes.get(node_id) or {}).get("kind") or "")
                            == "Type"
                            and str((self.graph.nodes.get(node_id) or {}).get("symbol") or "")
                            == type_name
                            and str(
                                (self.graph.nodes.get(node_id) or {}).get("contract_key")
                                or ""
                            ) == type_contract
                            and str(
                                (self.graph.nodes.get(node_id) or {}).get("tree")
                                or ""
                            ).startswith("donor_")
                            and (self.graph.nodes.get(node_id) or {}).get("is_definition")
                        ]
                        if len(provider_roots) == 1:
                            symbol_mapping = {
                                **type_mapping,
                                "root_node_ids": provider_roots,
                                "unresolved": False,
                                "status": "mapped_incomplete_type_provider",
                            }
                            compiler_missing_type = True
                            incomplete_type_cause = {
                                "type_symbol": type_name,
                                "declaration_file": relative_path,
                                "declaration_extent": dict(
                                    declarations[0].get("extent") or {}
                                ),
                                "provider_root_node_id": provider_roots[0],
                            }
        if compiler_missing_type and not incomplete_type_cause:
            symbol_mapping = self._map_symbol_diagnostic(
                diagnostic_id=diagnostic_id,
                issue=issue,
            )
        symbol_roots = set(symbol_mapping.get("root_node_ids") or [])
        resolved_missing_declaration_roots = [
            node_id
            for node_id in symbol_roots
            if str((self.graph.nodes.get(node_id) or {}).get("symbol") or "")
            == str(issue.get("symbol") or "")
            and str((self.graph.nodes.get(node_id) or {}).get("kind") or "")
            in {"Function", "Variable"}
            and _entity_has_source_extent(self.graph.nodes.get(node_id) or {})
        ]
        unique_missing_declaration = bool(
            located_unresolved_declaration
            and not symbol_mapping.get("unresolved")
            and len(symbol_roots) == len(resolved_missing_declaration_roots) == 1
        )
        if symbol_roots:
            if (
                position_symbol_issue
                and symbol_mapping.get("exact_contract_mapped")
                and not symbol_mapping.get("unresolved")
            ):
                # An exact provider for the identifier demanded at the
                # compiler location is stronger evidence than a historical
                # structural successor with a different contract identity.
                # Keep the historical mapping as provenance, but repair the
                # exact old/donor provider.
                root_nodes = symbol_roots
                mapped_nodes.extend(
                    symbol_mapping.get("mapped_node_ids") or []
                )
                mapping_status = str(
                    symbol_mapping.get("status") or mapping_status
                )
            elif (
                used_macro_invocation
                and compiler_missing_declaration
            ):
                root_nodes = symbol_roots
                mapping_status = str(
                    symbol_mapping.get("status") or mapping_status
                )
            elif unique_missing_declaration:
                root_nodes = set(resolved_missing_declaration_roots)
                mapping_status = str(
                    symbol_mapping.get("status") or mapping_status
                )
            elif compiler_missing_type:
                root_nodes = symbol_roots
                mapping_status = str(
                    symbol_mapping.get("status") or mapping_status
                )
            elif (
                macro_argument_mapping.get("root_node_ids")
                or baseline_contract_mapping.get("root_node_ids")
            ):
                mapped_nodes.extend(symbol_mapping.get("mapped_node_ids") or [])
            elif position_symbol_issue:
                root_nodes = symbol_roots
            else:
                if not used_macro_invocation and not located_unresolved_declaration:
                    root_nodes.update(symbol_roots)
            mapped_nodes.extend(symbol_mapping.get("mapped_node_ids") or [])
            if (
                not used_macro_invocation
                and not located_unresolved_declaration
                and not macro_argument_mapping.get("root_node_ids")
                and not baseline_contract_mapping.get("root_node_ids")
            ):
                mapping_status = str(
                    symbol_mapping.get("status") or mapping_status
                )

        graph_contract_anchor = (
            dict(issue.get("graph_contract_anchor") or {})
            if isinstance(
                issue.get("graph_contract_anchor"),
                dict,
            )
            else {}
        )
        if graph_contract_anchor:
            graph_contract_anchor_mapping = (
                self._map_graph_contract_anchor(
                    diagnostic_id=diagnostic_id,
                    anchor=graph_contract_anchor,
                )
            )
            contract_roots = set(
                graph_contract_anchor_mapping.get(
                    "root_node_ids"
                )
                or []
            )
            if contract_roots:
                root_nodes = contract_roots
                mapped_nodes.extend(
                    graph_contract_anchor_mapping.get(
                        "mapped_node_ids"
                    )
                    or []
                )
                mapping_status = str(
                    graph_contract_anchor_mapping.get("status")
                    or mapping_status
                )

        compiler_suggestion_mapping = self._map_compiler_suggestions(
            diagnostic_id=diagnostic_id,
            suggestions=compiler_suggestions,
        )
        # A compiler spelling suggestion is navigation evidence, not causal
        # proof. Replacing the exact missing contract with an existing
        # similarly named entity can make the repair target the wrong API.
        # Keep the suggestion edge in the graph while preserving the demanded
        # identifier (or its unresolved contract) as the primary root.
        mapped_nodes.extend(
            compiler_suggestion_mapping.get("mapped_node_ids") or []
        )

        if stable_root_evidence:
            anchored_roots = set(
                stable_root_evidence.get("root_node_ids") or []
            )
            if anchored_roots:
                root_nodes = anchored_roots
                mapped_nodes.extend(sorted(anchored_roots))
                for anchor_node_id in sorted(anchored_roots):
                    self.graph.add_edge(
                        diagnostic_id,
                        anchor_node_id,
                        "diagnoses_stable_contract",
                        budget_exempt=True,
                    )
                mapping_status = str(stable_root_evidence["status"])
            else:
                # A stale compiler line is not a valid replacement for an
                # entity identity that disappeared or became ambiguous after
                # an earlier patch in the same round.
                root_nodes = set()
                mapping_status = str(stable_root_evidence["status"])

        # Structural placement is relevant only when the normal diagnostic
        # mapping already proved that this source location belongs to an
        # old-tree declaration. Most API/provider diagnostics therefore keep
        # the original fast path without another structural parse.
        structural_container_mapping = (
            self._map_structural_container_mismatch(
                diagnostic_id=diagnostic_id,
                relative_path=relative_path,
                line=line,
                column=column,
            )
            if diagnostic_owner_node_ids and not stable_root_evidence
            else {}
        )
        structural_roots = set(
            structural_container_mapping.get("root_node_ids") or []
        )
        if structural_roots:
            # This is the same CodeIndex root-selection path as API and
            # provider diagnostics.  The only difference is that a proven
            # source-containment divergence is more causal than its later
            # compiler symptom, so it replaces the local owner root.
            root_nodes = structural_roots
            mapped_nodes.extend(
                structural_container_mapping.get("mapped_node_ids") or []
            )
            mapping_status = str(
                structural_container_mapping.get("status")
            )

        if (
            issue.get("repair_relevance") == "migration_contract"
            and not stable_root_evidence
        ):
            call_binding = _call_argument_mismatch_binding(
                mapped_references,
                references,
            )
            owner = (
                call_binding.get("owner")
                if isinstance(call_binding.get("owner"), dict)
                else {}
            )
            owner_id = self._retain_diagnostic_entity_node(owner)
            if owner_id:
                diagnostic_owner_node_ids.add(owner_id)
                diagnostic_transaction_owner_node_ids.add(owner_id)
                self.graph.add_edge(
                    diagnostic_id,
                    owner_id,
                    "diagnoses",
                    budget_exempt=True,
                    mapping_kind="exact_ast_call_argument_binding_owner",
                )
                endpoint_entity_ids: set[str] = set()
                for reference in [
                    *(call_binding.get("callee_references") or []),
                    *(call_binding.get("argument_references") or []),
                ]:
                    if not isinstance(reference, dict):
                        continue
                    reference_id = self._retain_diagnostic_reference_node(
                        reference
                    )
                    target = (
                        reference.get("referenced_entity")
                        if isinstance(
                            reference.get("referenced_entity"),
                            dict,
                        )
                        else {}
                    )
                    target_id = self._retain_diagnostic_entity_node(target)
                    if reference_id:
                        self.graph.add_edge(
                            diagnostic_id,
                            reference_id,
                            "diagnoses_call_argument_contract",
                            budget_exempt=True,
                        )
                        mapped_nodes.append(reference_id)
                    if target_id:
                        endpoint_entity_ids.add(target_id)
                        self.graph.add_edge(
                            owner_id,
                            target_id,
                            "call_argument_contract_endpoint",
                            budget_exempt=True,
                        )
                root_nodes = {owner_id}
                mapped_nodes.append(owner_id)
                mapping_status = str(call_binding.get("status"))
                call_argument_mapping = {
                    "status": call_binding.get("status"),
                    "authority": call_binding.get("authority"),
                    "owner_entity_id": owner_id,
                    "endpoint_entity_ids": sorted(endpoint_entity_ids),
                    "exact_argument_entity_ids": list(
                        call_binding.get("exact_argument_entity_ids") or []
                    ),
                    "call_branch": list(
                        call_binding.get("call_branch") or []
                    ),
                }

        if (
            old_root is not None
            and not stable_root_evidence
            and not structural_roots
            and not graph_contract_anchor_mapping.get("root_node_ids")
            and not call_argument_mapping
        ):
            binding = _local_return_owner_binding(
                mapped_references,
                references,
                source_root=old_root,
            )
            owner = binding.get("owner") or {}
            owner_id = self._retain_diagnostic_entity_node(owner)
            if owner_id:
                self.graph.add_edge(
                    diagnostic_id,
                    owner_id,
                    "diagnoses_local_return_contract",
                    budget_exempt=True,
                    local_reference_id=binding.get("local_reference_id"),
                    member_reference_id=binding.get("member_reference_id"),
                )
                root_nodes = {owner_id}
                mapped_nodes.append(owner_id)
                diagnostic_owner_node_ids.add(owner_id)
                diagnostic_transaction_owner_node_ids.add(owner_id)
                mapping_status = "mapped_local_return_owner"
                local_return_mapping = {
                    "owner_entity_id": owner_id,
                    "local_reference_id": binding.get("local_reference_id"),
                    "member_reference_id": binding.get("member_reference_id"),
                    "member_entity_id": binding.get("member_entity_id"),
                }

        if (
            not stable_root_evidence
            and not structural_roots
            and not graph_contract_anchor_mapping.get("root_node_ids")
            and not call_argument_mapping
            and root_nodes
            and all(
                str((self.graph.nodes.get(root_id) or {}).get("kind") or "").lower()
                == "type"
                and str((self.graph.nodes.get(root_id) or {}).get("tree") or "")
                == "old"
                for root_id in root_nodes
            )
        ):
            macro_pasted_mapping = self._map_macro_pasted_provider(
                diagnostic_id=diagnostic_id,
                relative_path=relative_path,
                line=line,
                column=column,
                diagnostic_owner_node_ids=diagnostic_owner_node_ids,
            )
            if macro_pasted_mapping.get("root_node_ids"):
                root_nodes = set(macro_pasted_mapping["root_node_ids"])
                mapped_nodes.extend(macro_pasted_mapping.get("mapped_node_ids") or [])
                mapping_status = str(macro_pasted_mapping["status"])

        if (
            not stable_root_evidence
            and not structural_roots
            and not graph_contract_anchor_mapping.get("root_node_ids")
            and not macro_argument_mapping.get("root_node_ids")
            and not call_argument_mapping
            and not used_macro_invocation
        ):
            missing_member_container_mapping = self._map_missing_member_container(
                diagnostic_id=diagnostic_id,
                issue=issue,
                relative_path=relative_path,
                line=line,
                column=column,
            )
            if missing_member_container_mapping.get("root_node_ids"):
                root_nodes = set(
                    missing_member_container_mapping["root_node_ids"]
                )
                mapped_nodes.extend(
                    missing_member_container_mapping["mapped_node_ids"]
                )
                mapping_status = str(
                    missing_member_container_mapping["status"]
                )

        if generated_export_origin.get("root_node_ids"):
            root_nodes = set(generated_export_origin["root_node_ids"])
            mapped_nodes.extend(generated_export_origin["mapped_node_ids"])
            diagnostic_owner_node_ids.update(root_nodes)
            diagnostic_transaction_owner_node_ids.update(root_nodes)
            mapping_status = str(generated_export_origin["status"])

        reached_consumer_mapping = (
            self._map_reached_donor_contract_consumers(
                diagnostic_id=diagnostic_id,
                root_node_ids=root_nodes,
                diagnostic_file=relative_path,
            )
            if not stable_root_evidence
            else {}
        )
        reached_consumer_owner_ids = set(
            reached_consumer_mapping.get("owner_node_ids") or []
        )
        diagnostic_owner_node_ids.update(reached_consumer_owner_ids)
        mapped_nodes.extend(sorted(reached_consumer_owner_ids))

        has_concrete_root = any(
            str((self.graph.nodes.get(root_id) or {}).get("kind") or "")
            not in _UNRESOLVED_PROVIDER_NODE_KINDS
            for root_id in root_nodes
        )
        unmapped = not root_nodes or bool(
            symbol_mapping.get("unresolved")
            and not symbol_mapping.get("exact_contract_mapped")
            and not has_concrete_root
        ) or bool(missing_member_container_mapping.get("unresolved"))
        if unmapped and not stable_root_evidence:
            mapping_status = str(
                missing_member_container_mapping.get("status")
                or symbol_mapping.get("status")
                or "unmapped_dependency"
            )
        exact_binding_provider_node_ids = {
            str(record.get("provider_entity_id") or "")
            for record in callback_binding_mapping.get("records") or []
            if isinstance(record, dict)
            and str(record.get("provider_entity_id") or "")
        }
        if (
            stable_root_evidence
            or structural_roots
            or graph_contract_anchor_mapping.get("root_node_ids")
            or macro_argument_mapping.get("root_node_ids")
        ):
            exact_binding_provider_node_ids = set()
        causal_root_nodes = _diagnostic_causal_root_frontier(
            root_nodes,
            diagnostic_transaction_owner_node_ids,
            exact_binding_provider_node_ids=(
                exact_binding_provider_node_ids
            ),
        )
        self.diagnostic_roots[event_id].update(causal_root_nodes)
        root_owner_alignment_evidence = self._diagnostic_root_owner_alignments(
            source_file=relative_path,
            root_node_ids=causal_root_nodes,
            existing_evidence=(
                symbol_mapping.get("provider_owner_evidence") or []
            ),
        )
        mapping = {
            "diagnostic_event_id": event_id,
            "diagnostic_node_id": diagnostic_id,
            "status": mapping_status,
            "mapped_node_ids": _ordered_unique(mapped_nodes),
            "diagnostic_owner_node_ids": sorted(
                diagnostic_owner_node_ids
            ),
            "diagnostic_transaction_owner_node_ids": sorted(
                diagnostic_transaction_owner_node_ids
            ),
            "root_candidate_node_ids": sorted(causal_root_nodes),
            "donor_position_evidence": list(donor_position_mapping.get("evidence") or []),
            "macro_argument_evidence": macro_argument_mapping.get("evidence"),
            "macro_pasted_provider_evidence": macro_pasted_mapping.get("evidence"),
            "local_return_evidence": local_return_mapping or None,
            "focused_diagnostic_parse": focused_diagnostic_parse or None,
            "baseline_contract_evidence": baseline_contract_mapping.get(
                "evidence"
            ),
            "root_owner_alignment_evidence": root_owner_alignment_evidence,
            "symbol_evidence": symbol_mapping,
            "graph_contract_anchor_evidence": (
                graph_contract_anchor_mapping
            ),
            "compiler_suggestion_evidence": compiler_suggestion_mapping,
            "structural_container_evidence": structural_container_mapping,
            "missing_member_container_evidence": (
                missing_member_container_mapping or None
            ),
            "incomplete_type_cause": incomplete_type_cause or None,
            "callback_binding_evidence": callback_binding_mapping,
            "call_argument_evidence": call_argument_mapping,
            "missing_object_evidence": missing_object_mapping or None,
            "generated_export_origin_evidence": generated_export_origin or None,
            "reached_consumer_evidence": reached_consumer_mapping,
            "stable_root_evidence": stable_root_evidence or None,
            "diagnostic_location_chain": location_chain,
            "related_location_evidence": list(
                related_location_mapping.get("evidence") or []
            ),
            "effective_diagnostic_location": effective_location,
            "unmapped": unmapped,
        }
        self.diagnostic_mappings[event_id] = mapping
        diagnostic_node = self.graph.nodes.get(diagnostic_id)
        if diagnostic_node is None:
            raise RuntimeError(f"diagnostic graph node disappeared during mapping: {diagnostic_id}")
        diagnostic_node["mapping_status"] = mapping_status
        diagnostic_node["root_candidate_node_ids"] = sorted(
            causal_root_nodes
        )
        return mapping

    def _reattribute_diagnostic_cascades(
        self,
        issues: list[dict[str, Any]],
        mappings: list[dict[str, Any]],
    ) -> None:
        """Bind secondary diagnostics to a proven prerequisite contract."""

        def locus(issue: dict[str, Any]) -> tuple[str, int, str]:
            phase = issue.get("_phase_index") or issue.get("phase_name") or ""
            return (
                self._normalize_issue_file(issue),
                _int_or_zero(issue.get("line")),
                str(phase),
            )

        def owners(mapping: dict[str, Any]) -> set[str]:
            return {
                str(node_id)
                for node_id in mapping.get("diagnostic_owner_node_ids") or []
                if str(node_id)
            }

        missing_members: dict[
            tuple[str, int, str],
            list[tuple[str, str, set[str], int]],
        ] = defaultdict(list)
        for issue, mapping in zip(issues, mappings):
            if str(issue.get("kind") or "") != "compile_missing_struct_member":
                continue
            event_id = str(issue.get("diagnostic_event_id") or "")
            location = locus(issue)
            roots = set(self.diagnostic_roots.get(event_id) or set())
            if not event_id or not location[0] or location[1] <= 0 or len(roots) != 1:
                continue
            root_id = next(iter(roots))
            if str((self.graph.nodes.get(root_id) or {}).get("kind") or "") != "Field":
                continue
            missing_members[location].append(
                (
                    event_id,
                    root_id,
                    owners(mapping),
                    _int_or_zero(issue.get("_issue_index")),
                )
            )

        for issue, mapping in zip(issues, mappings):
            if (
                str(issue.get("kind") or "") != "compile_error"
                or str(issue.get("warning_flag") or "")
                != "incompatible-pointer-types"
            ):
                continue
            event_id = str(issue.get("diagnostic_event_id") or "")
            issue_index = _int_or_zero(issue.get("_issue_index"))
            root_ids = set(self.diagnostic_roots.get(event_id) or set())
            root_kinds = {
                str((self.graph.nodes.get(root_id) or {}).get("kind") or "")
                for root_id in root_ids
            }
            if (
                not event_id
                or issue_index <= 0
                or not owners(mapping)
                or not root_ids
                or not root_kinds.issubset({"Function", "Variable"})
            ):
                continue
            causes = [
                candidate
                for candidate in missing_members.get(locus(issue), [])
                if candidate[3] > 0
                and candidate[3] < issue_index
                and bool(candidate[2] & owners(mapping))
            ]
            cause_roots = {candidate[1] for candidate in causes}
            if len(cause_roots) != 1:
                continue
            causal_root = next(iter(cause_roots))
            cause_event_ids = sorted(candidate[0] for candidate in causes)
            diagnostic_id = str(
                mapping.get("diagnostic_node_id")
                or f"diagnostic:{event_id}"
            )
            self.graph.add_edge(
                diagnostic_id,
                causal_root,
                "diagnoses_cascade_root",
                budget_exempt=True,
                causal_diagnostic_event_ids=cause_event_ids,
                relation=(
                    "same_ast_owner_initializer_after_missing_member"
                ),
            )
            self.diagnostic_roots[event_id] = {causal_root}
            mapping.update(
                {
                    "status": (
                        "mapped_compiler_cascade_to_structural_root"
                    ),
                    "root_candidate_node_ids": [causal_root],
                    "cascade_root_evidence": {
                        "causal_diagnostic_event_ids": cause_event_ids,
                        "causal_root_node_id": causal_root,
                        "superseded_root_candidate_node_ids": sorted(
                            root_ids
                        ),
                        "relation": (
                            "same_ast_owner_initializer_after_"
                            "missing_member"
                        ),
                    },
                    "unmapped": False,
                }
            )
            self.diagnostic_mappings[event_id] = mapping
            diagnostic_node = self.graph.nodes.get(diagnostic_id)
            if diagnostic_node is not None:
                diagnostic_node["mapping_status"] = mapping["status"]
                diagnostic_node["root_candidate_node_ids"] = [
                    causal_root
                ]

        incomplete_types = [
            (
                position,
                issue,
                mapping,
                mapping.get("incomplete_type_cause") or {},
            )
            for position, (issue, mapping) in enumerate(zip(issues, mappings))
            if isinstance(mapping.get("incomplete_type_cause"), dict)
        ]
        for position, (issue, mapping) in enumerate(zip(issues, mappings)):
            if str(issue.get("kind") or "") != "compile_missing_struct_member":
                continue
            container = issue.get("member_container") or {}
            if not isinstance(container, dict):
                continue
            matches = []
            for cause_position, cause_issue, cause_mapping, cause in incomplete_types:
                extent = cause.get("declaration_extent") or {}
                if (
                    cause_position >= position
                    or locus(cause_issue)[0] != locus(issue)[0]
                    or locus(cause_issue)[2] != locus(issue)[2]
                    or str(container.get("container_name") or "")
                    != str(cause.get("type_symbol") or "")
                    or not self._contains_position(
                        extent,
                        _int_or_zero(issue.get("line")),
                        _int_or_zero(issue.get("column")),
                    )
                ):
                    continue
                root_id = str(cause.get("provider_root_node_id") or "")
                if root_id and self.diagnostic_roots.get(
                    str(cause_issue.get("diagnostic_event_id") or "")
                ) == {root_id}:
                    matches.append((cause_issue, root_id))
            if len(matches) != 1:
                continue
            cause_issue, causal_root = matches[0]
            event_id = str(issue.get("diagnostic_event_id") or "")
            if not event_id:
                continue
            previous_roots = sorted(self.diagnostic_roots.get(event_id) or [])
            diagnostic_id = str(
                mapping.get("diagnostic_node_id") or f"diagnostic:{event_id}"
            )
            self.graph.add_edge(
                diagnostic_id,
                causal_root,
                "diagnoses_cascade_root",
                budget_exempt=True,
                causal_diagnostic_event_ids=[
                    str(cause_issue.get("diagnostic_event_id") or "")
                ],
                relation="same_declaration_incomplete_type_member",
            )
            self.diagnostic_roots[event_id] = {causal_root}
            mapping.update(
                {
                    "status": "mapped_incomplete_type_member_cascade",
                    "root_candidate_node_ids": [causal_root],
                    "cascade_root_evidence": {
                        "causal_diagnostic_event_ids": [
                            str(cause_issue.get("diagnostic_event_id") or "")
                        ],
                        "causal_root_node_id": causal_root,
                        "superseded_root_candidate_node_ids": previous_roots,
                        "relation": "same_declaration_incomplete_type_member",
                    },
                    "unmapped": False,
                }
            )
            self.diagnostic_mappings[event_id] = mapping
            diagnostic_node = self.graph.nodes.get(diagnostic_id)
            if diagnostic_node is not None:
                diagnostic_node["mapping_status"] = mapping["status"]
                diagnostic_node["root_candidate_node_ids"] = [causal_root]

    def add_diagnostics(
        self,
        issues: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized = [
            dict(issue) for issue in issues if isinstance(issue, dict)
        ]
        mappings = [self.add_diagnostic(issue) for issue in normalized]
        self._reattribute_diagnostic_cascades(normalized, mappings)
        return mappings

    def exact_provider_owner_bindings_for_diagnostics(
        self,
        issues: Iterable[dict[str, Any]],
        *,
        semantic_gap_ids: Iterable[object],
    ) -> dict[str, Any]:
        """Resolve provider-check diagnostics to exact editable graph ends.

        The referenced provider and the source entity containing an exact
        current-tree diagnostic location are both part of the atomic repair
        relation. Either endpoint becomes editable only after typed identity,
        source extent, and edit permission are proven. Ambiguous or stale
        locations remain evidence and never widen the repair boundary.
        """

        gap_ids = _ordered_unique(semantic_gap_ids)
        normalized_issues = [
            dict(issue) for issue in issues if isinstance(issue, dict)
        ]
        mappings = self.add_diagnostics(normalized_issues)
        bindings: dict[str, dict[str, Any]] = {}
        resolutions: list[dict[str, Any]] = []

        def mutation_identity(entity: dict[str, Any]) -> tuple[object, ...]:
            relative_path = normalize_relpath(
                str(entity.get("file") or entity.get("path") or "")
            )
            kind = str(
                entity.get("kind") or entity.get("entity_kind") or ""
            )
            contract_key = str(
                entity.get("contract_key") or entity.get("key") or ""
            )
            container = (
                entity.get("container")
                if isinstance(entity.get("container"), dict)
                else {}
            )
            if kind in {"field", "enum_constant"}:
                return (
                    relative_path,
                    kind,
                    str(container.get("container_kind") or ""),
                    str(container.get("container_name") or ""),
                    contract_key,
                )
            return (relative_path, kind, contract_key)

        def exact_equivalence_proof(
            records: list[dict[str, Any]],
        ) -> dict[str, Any] | None:
            old_records = [
                entity
                for entity in records
                if str(entity.get("tree") or "") == "old"
            ]
            donor_records = [
                entity
                for entity in records
                if str(entity.get("tree") or "").startswith("donor_")
            ]
            for old_entity in old_records:
                for donor_entity in donor_records:
                    comparison = compare_entity_fingerprints(
                        old_entity,
                        donor_entity,
                    )
                    entity_kind = str(
                        old_entity.get("kind")
                        or old_entity.get("entity_kind")
                        or ""
                    ).lower()
                    complete_source_match = (
                        comparison.get("exact_definition_source")
                        == "equivalent"
                    )
                    complete_typed_match = all(
                        comparison.get(dimension) == "equivalent"
                        for dimension in (
                            "signature",
                            "implementation",
                            "interface",
                        )
                    )
                    complete_macro_match = (
                        entity_kind == "macro"
                        and comparison.get("implementation")
                        == "equivalent"
                    )
                    if not (
                        complete_source_match
                        or complete_typed_match
                        or complete_macro_match
                    ):
                        continue
                    return {
                        "old_entity_id": str(
                            old_entity.get("id")
                            or old_entity.get("entity_id")
                            or ""
                        ),
                        "donor_entity_id": str(
                            donor_entity.get("id")
                            or donor_entity.get("entity_id")
                            or ""
                        ),
                        "comparison": comparison,
                    }
            return None

        def bind_consumer_owner(
            *,
            diagnostic_id: str,
            current_descriptor: dict[str, Any] | None,
            donor_descriptor: dict[str, Any] | None,
            exact_node_refs: Iterable[object],
        ) -> str | None:
            current = dict(current_descriptor or {})
            donor = dict(donor_descriptor or {})
            descriptor = current or donor
            relative_path = normalize_relpath(
                str(descriptor.get("file") or descriptor.get("path") or "")
            )
            identity = mutation_identity(descriptor)
            classify = getattr(self.spec, "classify_edit_root", None)
            permission = (
                str(classify(relative_path))
                if callable(classify) and relative_path
                else "forbidden"
            )
            if (
                not relative_path
                or not all(identity)
                or permission not in {"editable", "escalation"}
            ):
                return None
            owner_key = stable_id(
                "typed-diagnostic-consumer-owner",
                *[str(item) for item in identity],
            )
            binding = {
                "owner_key": owner_key,
                "role": "diagnostic_consumer_owner",
                "authority": "typed_diagnostic_consumer_resolution",
                "resolver_revision": (
                    TYPED_DIAGNOSTIC_PROVIDER_RESOLUTION_REVISION
                ),
                "file": relative_path,
                "contract_key": str(
                    descriptor.get("contract_key")
                    or descriptor.get("key")
                    or ""
                ),
                "symbol": descriptor.get("symbol") or descriptor.get("name"),
                "entity_kind": (
                    descriptor.get("kind")
                    or descriptor.get("entity_kind")
                ),
                "exact_node_refs": _ordered_unique(
                    [diagnostic_id, *list(exact_node_refs)]
                ),
                "target_presence": (
                    "existing_provider_consumer"
                    if current
                    else "donor_provider_consumer"
                ),
                "current_descriptor": deepcopy(current),
                "donor_descriptor": deepcopy(donor),
                "edit_permission": permission,
                "semantic_gap_ids": list(gap_ids),
            }
            existing = bindings.get(owner_key)
            if existing is None:
                bindings[owner_key] = binding
            else:
                existing["exact_node_refs"] = _ordered_unique(
                    [
                        *(existing.get("exact_node_refs") or []),
                        *(binding.get("exact_node_refs") or []),
                    ]
                )
                if current:
                    existing["current_descriptor"] = deepcopy(current)
                    existing["target_presence"] = (
                        "existing_provider_consumer"
                    )
                if donor:
                    existing["donor_descriptor"] = deepcopy(donor)
            return owner_key

        for issue, mapping in zip(normalized_issues, mappings):
            symbol = str(issue.get("symbol") or "").strip()
            diagnostic_id = str(mapping.get("diagnostic_node_id") or "")
            provider_mapping = self._map_symbol_diagnostic(
                diagnostic_id=diagnostic_id,
                issue={
                    "symbol": symbol,
                    "module": issue.get("module"),
                },
            )
            root_ids = {
                str(node_id)
                for node_id in provider_mapping.get("root_node_ids") or []
                if str(node_id)
            }
            query_matches = [
                dict(match)
                for query in provider_mapping.get(
                    "provider_definition_queries"
                )
                or []
                if isinstance(query, dict)
                for match in query.get("matches") or []
                if isinstance(match, dict)
                and _entity_has_source_extent(match)
            ]
            root_candidates = [
                entity
                for entity in self.entities_by_name.get(symbol, [])
                if str(
                    entity.get("id") or entity.get("entity_id") or ""
                )
                in root_ids
                and _entity_has_source_extent(entity)
            ]
            seed_candidates = canonical_entity_records(
                [*root_candidates, *query_matches]
            )
            seed_identities = {
                mutation_identity(entity)
                for entity in seed_candidates
                if all(mutation_identity(entity))
            }
            candidates = canonical_entity_records(
                [
                    *seed_candidates,
                    *[
                        entity
                        for entity in self.entities_by_name.get(symbol, [])
                        if _entity_has_source_extent(entity)
                        and mutation_identity(entity) in seed_identities
                    ],
                ]
            )
            grouped: dict[tuple[object, ...], list[dict[str, Any]]] = (
                defaultdict(list)
            )
            for entity in candidates:
                identity = mutation_identity(entity)
                if all(identity):
                    grouped[identity].append(entity)
            equivalent_identities: dict[
                tuple[object, ...], dict[str, Any]
            ] = {}
            for identity, records in grouped.items():
                proof = exact_equivalence_proof(records)
                if proof is not None:
                    equivalent_identities[identity] = proof
            actionable_groups = {
                identity: records
                for identity, records in grouped.items()
                if identity not in equivalent_identities
            }
            resolution = {
                "diagnostic_event_id": mapping.get("diagnostic_event_id"),
                "diagnostic_node_id": diagnostic_id,
                "symbol": symbol,
                "diagnostic_mapping_status": mapping.get("status"),
                "provider_mapping_status": provider_mapping.get("status"),
                "root_candidate_node_ids": sorted(root_ids),
                "provider_definition_queries": deepcopy(
                    provider_mapping.get("provider_definition_queries")
                    or []
                ),
                "candidate_mutation_identities": [
                    list(identity) for identity in sorted(grouped)
                ],
                "consumer_owner_node_ids": list(
                    mapping.get("diagnostic_owner_node_ids") or []
                ),
                "equivalent_provider_identities": [
                    {
                        "mutation_identity": list(identity),
                        "proof": deepcopy(equivalent_identities[identity]),
                    }
                    for identity in sorted(equivalent_identities)
                ],
            }
            if str(issue.get("location_provenance") or "") in {
                "exact_unique_unchanged_diff_block",
                "exact_unchanged_overlay_file",
            }:
                consumer_ids = {
                    str(node_id)
                    for node_id in mapping.get(
                        "diagnostic_owner_node_ids"
                    )
                    or []
                    if str(node_id)
                }
                consumer_records = canonical_entity_records(
                    entity
                    for entity in self.entities_by_file.get(
                        (
                            "old",
                            normalize_relpath(
                                str(issue.get("file") or "")
                            ),
                        ),
                        [],
                    )
                    if str(
                        entity.get("id")
                        or entity.get("entity_id")
                        or ""
                    )
                    in consumer_ids
                    and _entity_has_source_extent(entity)
                )
                consumer_bindings: list[str] = []
                for consumer in consumer_records:
                    consumer_file = normalize_relpath(
                        str(consumer.get("file") or "")
                    )
                    consumer_contract = str(
                        consumer.get("contract_key")
                        or consumer.get("key")
                        or ""
                    )
                    donor_consumers = canonical_entity_records(
                        entity
                        for entity in self.entities_by_contract.get(
                            consumer_contract,
                            [],
                        )
                        if str(entity.get("tree") or "").startswith(
                            "donor_"
                        )
                        and normalize_relpath(
                            str(entity.get("file") or "")
                        )
                        == consumer_file
                        and str(
                            entity.get("kind")
                            or entity.get("entity_kind")
                            or ""
                        )
                        == str(
                            consumer.get("kind")
                            or consumer.get("entity_kind")
                            or ""
                        )
                        and bool(entity.get("is_definition"))
                        == bool(consumer.get("is_definition"))
                    )
                    consumer_owner_key = bind_consumer_owner(
                        diagnostic_id=diagnostic_id,
                        current_descriptor=consumer,
                        donor_descriptor=(
                            donor_consumers[0]
                            if len(donor_consumers) == 1
                            else None
                        ),
                        exact_node_refs=[
                            str(
                                consumer.get("id")
                                or consumer.get("entity_id")
                                or ""
                            )
                        ],
                    )
                    if consumer_owner_key:
                        consumer_bindings.append(consumer_owner_key)
                if consumer_bindings:
                    resolution["consumer_owner_keys"] = sorted(
                        set(consumer_bindings)
                    )
            if len(actionable_groups) != 1:
                if not actionable_groups and equivalent_identities:
                    resolution["status"] = "provider_already_equivalent"
                else:
                    resolution["status"] = (
                        "provider_unresolved"
                        if not grouped
                        else "provider_ambiguous"
                    )
                resolutions.append(resolution)
                continue

            identity, records = next(iter(actionable_groups.items()))
            donor_records = [
                entity
                for entity in records
                if str(entity.get("tree") or "").startswith("donor_")
            ]
            old_records = [
                entity
                for entity in records
                if str(entity.get("tree") or "") == "old"
            ]
            descriptor = canonical_entity_records(
                donor_records or old_records or records
            )[0]
            relative_path = normalize_relpath(
                str(descriptor.get("file") or descriptor.get("path") or "")
            )
            classify = getattr(self.spec, "classify_edit_root", None)
            permission = (
                str(classify(relative_path))
                if callable(classify)
                else "forbidden"
            )
            if permission not in {"editable", "escalation"}:
                resolution.update(
                    {
                        "status": "provider_not_editable",
                        "provider_file": relative_path,
                        "edit_permission": permission,
                    }
                )
                resolutions.append(resolution)
                continue

            provider_node_ids = _ordered_unique(
                str(entity.get("id") or entity.get("entity_id") or "")
                for entity in records
            )
            owner_key = stable_id(
                "typed-diagnostic-provider-owner",
                *[str(item) for item in identity],
            )
            binding = {
                "owner_key": owner_key,
                "role": "diagnostic_provider_owner",
                "authority": "typed_diagnostic_provider_resolution",
                "resolver_revision": (
                    TYPED_DIAGNOSTIC_PROVIDER_RESOLUTION_REVISION
                ),
                "file": relative_path,
                "contract_key": str(
                    descriptor.get("contract_key")
                    or descriptor.get("key")
                    or ""
                ),
                "symbol": descriptor.get("symbol") or symbol,
                "entity_kind": (
                    descriptor.get("kind")
                    or descriptor.get("entity_kind")
                ),
                "exact_node_refs": _ordered_unique(
                    [diagnostic_id, *provider_node_ids]
                ),
                "target_presence": (
                    "donor_provider"
                    if donor_records
                    else "existing_provider"
                ),
                "current_descriptor": (
                    deepcopy(canonical_entity_records(old_records)[0])
                    if old_records
                    else {}
                ),
                "donor_descriptor": (
                    deepcopy(canonical_entity_records(donor_records)[0])
                    if donor_records
                    else deepcopy(descriptor)
                ),
                "edit_permission": permission,
                "semantic_gap_ids": list(gap_ids),
            }
            bindings.setdefault(owner_key, binding)

            direct_consumer_owner_keys: list[str] = []
            direct_consumer_reference_ids: list[str] = []
            consumer_probe_reports: list[dict[str, Any]] = []
            donor_provider_ids_by_tree: dict[str, set[str]] = defaultdict(set)
            for provider in donor_records:
                provider_tree = str(provider.get("tree") or "")
                provider_id = str(
                    provider.get("id") or provider.get("entity_id") or ""
                )
                if provider_tree.startswith("donor_") and provider_id:
                    donor_provider_ids_by_tree[provider_tree].add(provider_id)

            for provider_tree, donor_provider_ids in sorted(
                donor_provider_ids_by_tree.items()
            ):
                frontier = self.compile_dependency_frontier(provider_tree)
                candidate_search = self.exact_token_candidates_in_files(
                    provider_tree,
                    [symbol],
                    frontier.get("files") or [],
                )
                candidate_files = _ordered_unique(
                    (candidate_search.get("candidates_by_symbol") or {}).get(
                        symbol,
                        [],
                    )
                )
                probe_reports = self.query_reference_files_to_entities(
                    provider_tree,
                    candidate_files,
                    donor_provider_ids,
                )
                for probe in probe_reports:
                    self.add_reference_probe_report(probe)
                    references = [
                        dict(reference)
                        for reference in probe.get("references") or []
                        if isinstance(reference, dict)
                        and str(
                            reference.get("referenced_entity_id") or ""
                        )
                        in donor_provider_ids
                    ]
                    consumer_probe_reports.append(
                        {
                            "tree": provider_tree,
                            "file": normalize_relpath(
                                str(probe.get("file") or "")
                            ),
                            "reference_count": len(references),
                            "parse_ok": bool(probe.get("parse_ok")),
                            "parse_error": probe.get("parse_error"),
                        }
                    )
                    probe_entities = {
                        str(entity.get("id") or entity.get("entity_id") or ""):
                        dict(entity)
                        for entity in probe.get("entities")
                        or probe.get("definitions")
                        or []
                        if isinstance(entity, dict)
                        and str(
                            entity.get("id")
                            or entity.get("entity_id")
                            or ""
                        )
                    }
                    for reference in references:
                        donor_consumer = (
                            dict(reference.get("owner_entity") or {})
                            if isinstance(
                                reference.get("owner_entity"),
                                dict,
                            )
                            else {}
                        )
                        if not donor_consumer:
                            donor_consumer = probe_entities.get(
                                str(reference.get("owner_entity_id") or ""),
                                {},
                            )
                        if not _entity_has_source_extent(donor_consumer):
                            continue
                        consumer_file = normalize_relpath(
                            str(donor_consumer.get("file") or "")
                        )
                        consumer_contract = str(
                            donor_consumer.get("contract_key")
                            or donor_consumer.get("key")
                            or ""
                        )
                        consumer_kind = str(
                            donor_consumer.get("kind")
                            or donor_consumer.get("entity_kind")
                            or ""
                        )
                        old_root = self.root_for_tree("old")
                        if (
                            old_root is not None
                            and consumer_file
                            and (old_root / consumer_file).is_file()
                        ):
                            self.index_file("old", consumer_file)
                        current_consumers = canonical_entity_records(
                            entity
                            for entity in self.entities_by_contract.get(
                                consumer_contract,
                                [],
                            )
                            if str(entity.get("tree") or "") == "old"
                            and normalize_relpath(
                                str(entity.get("file") or "")
                            )
                            == consumer_file
                            and str(
                                entity.get("kind")
                                or entity.get("entity_kind")
                                or ""
                            )
                            == consumer_kind
                            and bool(entity.get("is_definition"))
                            == bool(donor_consumer.get("is_definition"))
                            and _entity_has_source_extent(entity)
                        )
                        if len(current_consumers) > 1:
                            continue
                        reference_id = str(
                            reference.get("id")
                            or reference.get("reference_id")
                            or ""
                        )
                        consumer_owner_key = bind_consumer_owner(
                            diagnostic_id=diagnostic_id,
                            current_descriptor=(
                                current_consumers[0]
                                if current_consumers
                                else None
                            ),
                            donor_descriptor=donor_consumer,
                            exact_node_refs=[
                                reference_id,
                                str(
                                    reference.get("referenced_entity_id")
                                    or ""
                                ),
                                str(
                                    donor_consumer.get("id")
                                    or donor_consumer.get("entity_id")
                                    or ""
                                ),
                                *(
                                    [
                                        str(
                                            current_consumers[0].get("id")
                                            or current_consumers[0].get(
                                                "entity_id"
                                            )
                                            or ""
                                        )
                                    ]
                                    if current_consumers
                                    else []
                                ),
                            ],
                        )
                        if consumer_owner_key:
                            direct_consumer_owner_keys.append(
                                consumer_owner_key
                            )
                            direct_consumer_reference_ids.append(reference_id)
            resolution.update(
                {
                    "status": "provider_resolved",
                    "owner_key": owner_key,
                    "provider_file": relative_path,
                    "provider_node_ids": provider_node_ids,
                    "direct_consumer_owner_keys": _ordered_unique(
                        direct_consumer_owner_keys
                    ),
                    "direct_consumer_reference_ids": _ordered_unique(
                        direct_consumer_reference_ids
                    ),
                    "consumer_reference_probes": consumer_probe_reports,
                }
            )
            resolutions.append(resolution)

        return {
            "schema": "typed_diagnostic_provider_expansion_v1",
            "resolver_revision": (
                TYPED_DIAGNOSTIC_PROVIDER_RESOLUTION_REVISION
            ),
            "status": "expanded" if bindings else "no_unique_provider",
            "semantic_gap_ids": list(gap_ids),
            "diagnostic_count": len(normalized_issues),
            "resolutions": resolutions,
            "owner_bindings": [
                deepcopy(bindings[key]) for key in sorted(bindings)
            ],
            "allowed_fileset": _ordered_unique(
                binding.get("file") for binding in bindings.values()
            ),
        }

    def query_definition_batches(
        self,
        *,
        tree: str,
        symbols: Iterable[str],
        cursor: int = 0,
        max_files: int = 192,
        candidate_files: Iterable[object] | None = None,
        candidate_source: str = "automatic",
        previously_attempted_files: Iterable[object] = (),
        contract_keys: Iterable[object] = (),
        require_definition: bool = False,
        exhaustive_candidate_scan: bool = False,
        query_only_files: Iterable[object] = (),
        definition_probe_files: Iterable[object] = (),
    ) -> dict[str, Any]:
        """Continue exact contract lookup inside the realized compile frontier."""

        wanted = {str(symbol).strip() for symbol in symbols if str(symbol).strip()}
        wanted_contracts = _normalized_contract_keys(contract_keys)
        compile_files = self.compile_command_files(tree)
        context_available = self.contexts.get(tree) is not None
        start = max(0, int(cursor))
        provided_candidate_files = (
            None
            if candidate_files is None
            else _ordered_unique(candidate_files)
        )
        query_available = bool(
            context_available or provided_candidate_files is not None
        )
        attempted_files = set(_ordered_unique(previously_attempted_files))
        query_only = set(_ordered_unique(query_only_files))
        definition_probes = set(_ordered_unique(definition_probe_files))

        def concrete_matches() -> list[dict[str, Any]]:
            return build_conditioned_provider_records([
                record
                for symbol in sorted(wanted)
                for record in self.entities_by_name.get(symbol, [])
                if str(record.get("tree") or "") == tree
                and _entity_has_source_extent(record)
                and (
                    not wanted_contracts
                    or str(
                        record.get("contract_key")
                        or record.get("key")
                        or ""
                    )
                    in wanted_contracts
                )
            ])

        def query_satisfied(records: list[dict[str, Any]]) -> bool:
            records_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                records_by_symbol[str(record.get("symbol") or "")].append(record)
            for symbol in wanted:
                symbol_records = records_by_symbol.get(symbol, [])
                if not symbol_records:
                    return False
                if any(bool(record.get("is_definition")) for record in symbol_records):
                    continue
                if require_definition:
                    return False
                if any(
                    str(record.get("kind") or "").lower()
                    in {"function", "variable"}
                    for record in symbol_records
                ):
                    return False
            return True

        def retain_query_matches(
            records: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            self.retain_query_definition_records(
                record
                for record in records
                if str(record.get("symbol") or "") in wanted
            )
            return concrete_matches()

        existing_matches = concrete_matches()
        if (
            query_satisfied(existing_matches)
            and not exhaustive_candidate_scan
        ):
            existing_matches = retain_query_matches(existing_matches)
            return {
                "schema": "code_index_query_page_v3",
                "tree": tree,
                "symbols": sorted(wanted),
                "contract_keys": sorted(wanted_contracts),
                "cursor": start,
                "next_cursor": None,
                "scanned_file_count": 0,
                "skipped_full_parse_count": 0,
                "total_file_count": len(
                    provided_candidate_files
                    if provided_candidate_files is not None
                    else compile_files
                ),
                "complete": True,
                "truncated": False,
                "unavailable": False,
                "unresolved_reason": None,
                "matches": existing_matches,
                "resolution_source": "existing_exact_ast_entity",
                "candidate_source": candidate_source,
                "attempted_files": sorted(attempted_files),
                "resolved": True,
            }

        if provided_candidate_files is None:
            frontier = self.compile_dependency_frontier(
                tree,
                within_root="",
            )
            bounded = self.exact_token_candidates_in_files(
                tree,
                wanted,
                frontier.get("files") or [],
            )
            ordered_files = _ordered_unique(
                relative_path
                for symbol in sorted(wanted)
                for relative_path in (
                    bounded.get("candidates_by_symbol") or {}
                ).get(symbol, [])
            )
            lexical_candidates = list(ordered_files)
            structural_candidate_files = []
            candidate_source = "case_dependency_frontier_exact_token"
        else:
            ordered_files = list(provided_candidate_files)
            lexical_candidates = (
                list(ordered_files)
                if candidate_source
                in {
                    "case_compile_frontier_exact_token",
                    "case_dependency_frontier_exact_token",
                    "demand_provider_exact_token",
                }
                else []
            )
            # A staged query validates structural ownership only for files on
            # the current page instead of pre-parsing every lexical candidate.
            structural_candidate_files = []
        end = min(len(ordered_files), start + max(1, int(max_files)))
        lexical_set = set(lexical_candidates)
        scanned = 0
        skipped = 0
        resolved_from_lexical_candidate = False
        resolved_from_structural_candidate = False
        for relative_path in ordered_files[start:end]:
            file_node = self.graph.nodes.get(f"file:{tree}:{relative_path}") or {}
            if relative_path in attempted_files or (
                not self.graph.truncated
                and bool(file_node.get("full_parse_attempted"))
            ):
                skipped += 1
            else:
                if relative_path in query_only or relative_path in definition_probes:
                    root = self.root_for_tree(tree)
                    driver_root = self.driver_root_for_tree(tree)
                    if root is not None and driver_root is not None:
                        report = parse_source_file(
                            self.contexts.get(tree),
                            root.resolve(),
                            driver_root,
                            relative_path,
                            detailed_processing=True,
                            tree=tree,
                        )
                        if report.get("parse_ok"):
                            definitions = [
                                record
                                for record in report.get("entities") or []
                                if record.get("is_definition")
                                and str(record.get("symbol") or "") in wanted
                                and (
                                    not wanted_contracts
                                    or str(
                                        record.get("contract_key")
                                        or record.get("key")
                                        or ""
                                    ) in wanted_contracts
                                )
                            ]
                            if definitions and relative_path not in query_only:
                                self.index_file(tree, relative_path, detailed_processing=True)
                            else:
                                self.retain_query_definition_records(definitions)
                        else:
                            self.index_file(tree, relative_path, detailed_processing=True)
                    else:
                        self.index_file(tree, relative_path, detailed_processing=True)
                else:
                    self.index_file(tree, relative_path, detailed_processing=True)
                scanned += 1
            attempted_files.add(relative_path)
            current_matches = concrete_matches()
            defined_here = {
                str(record.get("symbol") or "")
                for record in current_matches
                if normalize_relpath(str(record.get("file") or "")) == relative_path
                and record.get("is_definition")
            }
            if relative_path in lexical_set and (
                wanted - defined_here
                and (exhaustive_candidate_scan or not query_satisfied(current_matches))
            ):
                structural_matches = self.materialize_structural_definition_candidates(
                    tree,
                    relative_path,
                    wanted,
                    contract_keys=wanted_contracts,
                )
                if structural_matches:
                    resolved_from_structural_candidate = True
                    current_matches = concrete_matches()
            if (
                query_satisfied(current_matches)
                and not exhaustive_candidate_scan
            ):
                current_matches = retain_query_matches(current_matches)
                resolved_from_lexical_candidate = relative_path in lexical_set
                return {
                    "schema": "code_index_query_page_v3",
                    "tree": tree,
                    "symbols": sorted(wanted),
                    "contract_keys": sorted(wanted_contracts),
                    "cursor": start,
                    "next_cursor": None,
                    "scanned_file_count": scanned,
                    "skipped_full_parse_count": skipped,
                    "total_file_count": len(ordered_files),
                    "compile_command_file_count": len(compile_files),
                    "lexical_candidate_count": len(lexical_candidates),
                    "structural_candidate_file_count": len(
                        structural_candidate_files
                    ),
                    "complete": True,
                    "truncated": False,
                    "unavailable": False,
                    "unresolved_reason": None,
                    "matches": current_matches,
                    "resolution_source": (
                        "lexical_candidate_verified_by_structural_definition"
                        if resolved_from_structural_candidate
                        else "lexical_candidate_verified_by_ast"
                        if resolved_from_lexical_candidate
                        else "compile_command_candidate_verified_by_ast"
                    ),
                    "candidate_source": candidate_source,
                    "attempted_files": sorted(attempted_files),
                    "resolved": True,
                }
        matches = concrete_matches()
        complete = end >= len(ordered_files)
        resolved = bool(complete and query_satisfied(matches))
        if resolved:
            matches = retain_query_matches(matches)
        return {
            "schema": "code_index_query_page_v3",
            "tree": tree,
            "symbols": sorted(wanted),
            "contract_keys": sorted(wanted_contracts),
            "cursor": start,
            "next_cursor": None if complete else end,
            "scanned_file_count": scanned,
            "skipped_full_parse_count": skipped,
            "total_file_count": len(ordered_files),
            "compile_command_file_count": len(compile_files),
            "lexical_candidate_count": len(lexical_candidates),
            "structural_candidate_file_count": len(structural_candidate_files),
            "complete": complete,
            "truncated": query_available and not complete,
            "unavailable": not query_available,
            "unresolved_reason": (
                None
                if query_available
                else self.context_reports.get(tree, {}).get("reason")
            ),
            "matches": matches,
            "resolution_source": (
                "exhaustive_exact_candidate_definition_query"
                if exhaustive_candidate_scan
                else "ordered_ast_definition_query"
            ),
            "candidate_source": candidate_source,
            "attempted_files": sorted(attempted_files),
            "resolved": resolved,
        }

    def _natural_root_record(self, node_id: str) -> dict[str, Any]:
        node = dict(self.graph.nodes.get(node_id) or {"id": node_id, "kind": "Unknown"})
        symbol = str(node.get("symbol") or node.get("name") or node.get("spelling") or "")
        file = normalize_relpath(str(node.get("file") or node.get("path") or ""))
        contract = str(node.get("contract_key") or node.get("key") or symbol)
        scope = f"{file}:{contract}" if file and contract and contract != file else file or contract or node_id
        return {
            "key": node_id,
            "node_id": node_id,
            "title": "code_dependency_root",
            "scope": scope,
            "entity_kind": node.get("kind"),
            "entity_id": node_id,
            "tree": node.get("tree"),
            "file": file or None,
            "symbol": symbol or None,
            "contract_key": contract or None,
            "source": "code_index_common_upstream_partition_v3",
        }

    def _upstream_reachable_nodes(self, starts: Iterable[str]) -> set[str]:
        upstream_edge_types = {
            "candidate_resolution",
            "diagnostic_contract_dependency",
            "instance_of_contract",
            "part_of_patch_transaction",
            "depends_on_patch_transaction",
            "member_of",
            "has_type",
            "owned_by_contract_file",
        }
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in self.graph.edges:
            if str(edge.get("type") or "") in upstream_edge_types:
                adjacency[str(edge.get("from") or "")].append(str(edge.get("to") or ""))
        reachable: set[str] = set()
        queue = deque(str(item) for item in starts if str(item))
        while queue:
            node_id = queue.popleft()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            queue.extend(adjacency.get(node_id, []))
        return reachable

    def _upstream_root_nodes_from(
        self,
        immediate_nodes: Iterable[object],
    ) -> set[str]:
        """Return typed semantic roots reachable from an event frontier."""

        immediate = {str(item) for item in immediate_nodes if str(item)}
        if not immediate:
            return set()
        reachable = self._upstream_reachable_nodes(immediate)
        resolved_frontiers = {
            str(edge.get("from") or "")
            for edge in self.graph.edges
            if edge.get("type") == "candidate_resolution"
        }
        causal_file_nodes = {
            str(edge.get("to") or "")
            for edge in self.graph.edges
            if edge.get("type") in {"candidate_resolution", "resolves_to"}
            and str(edge.get("from") or "") in reachable
        }
        eligible: set[str] = set()
        semantic_kinds = {
            "Function",
            "Variable",
            "Type",
            "Field",
            "Enum_Constant",
            "Macro",
            "Top_Level_Call_Expression",
            "Entity",
            "Contract",
            "PatchTransaction",
            "UnresolvedContract",
            "UnresolvedEntity",
            "UnresolvedInclude",
        }
        for node_id in reachable:
            node_kind = str((self.graph.nodes.get(node_id) or {}).get("kind") or "")
            if (
                node_kind in _UNRESOLVED_PROVIDER_NODE_KINDS
                and node_id in resolved_frontiers
            ):
                continue
            if node_kind in semantic_kinds:
                eligible.add(node_id)
            elif node_kind == "File" and (node_id in immediate or node_id in causal_file_nodes):
                # The diagnostic occurrence file is excluded; include targets and
                # graph-resolved file dependencies remain valid upstream roots.
                eligible.add(node_id)
        return eligible

    def upstream_root_nodes(self, event_id: str) -> set[str]:
        """Return the most-specific responsible roots for one event."""

        return set(
            self._most_specific_roots(
                self._upstream_root_nodes_from(
                    self.diagnostic_roots.get(event_id) or set()
                )
            )
        )

    def upstream_dependency_nodes(self, node_ids: Iterable[str]) -> set[str]:
        """Return typed semantic dependencies reachable from graph nodes."""

        return set(self._upstream_reachable_nodes(node_ids))

    def semantic_contract_binding(
        self,
        node_ids: Iterable[object],
    ) -> dict[str, Any]:
        reachable = self._upstream_reachable_nodes(
            str(item) for item in node_ids if str(item)
        )
        semantic_nodes = [
            dict(self.graph.nodes[node_id])
            for node_id in sorted(reachable)
            if str((self.graph.nodes.get(node_id) or {}).get("kind") or "")
            in {"Contract", "PatchTransaction"}
            and (self.graph.nodes.get(node_id) or {}).get("semantic_gap_ids")
        ]
        contract_nodes = [
            node
            for node in semantic_nodes
            if str(node.get("kind") or "") == "Contract"
        ]
        transaction_nodes = [
            node
            for node in semantic_nodes
            if str(node.get("kind") or "") == "PatchTransaction"
        ]
        gap_ids = sorted(
            {
                str(gap_id)
                for node in semantic_nodes
                for gap_id in node.get("semantic_gap_ids") or []
                if str(gap_id)
            }
        )
        return {
            "authority": "typed_code_graph_exact_contract",
            "semantic_gap_ids": gap_ids,
            "contract_node_ids": [str(node.get("id") or "") for node in contract_nodes],
            "transaction_node_ids": [
                str(node.get("id") or "")
                for node in transaction_nodes
            ],
            "contract_identities": sorted(
                {
                    str(identity)
                    for node in contract_nodes
                    for identity in node.get("contract_identities") or []
                    if str(identity)
                }
            ),
        }

    def _most_specific_roots(self, roots: set[str]) -> list[str]:
        if not roots:
            return []
        less_specific: set[str] = set()
        for candidate in roots:
            reachable = self._upstream_reachable_nodes([candidate])
            less_specific.update(
                other
                for other in roots
                if other != candidate and other in reachable
            )
        return sorted(roots - less_specific)

    @staticmethod
    def _event_partitions(
        event_ids: list[str],
        roots_by_event: dict[str, set[str]],
    ) -> list[dict[str, Any]]:
        """Partition graph events without transitive overlap merging.

        Each root induces the events that can reach it. Maximal induced sets
        are valid common-root groups. An event covered by multiple incomparable
        maximal sets is a bridge and remains in an explicit ambiguous
        partition, so no event appears in two work units.
        """

        root_events: dict[str, set[str]] = defaultdict(set)
        for event_id in event_ids:
            for root_id in roots_by_event.get(event_id) or set():
                root_events[root_id].add(event_id)

        coverage_roots: dict[frozenset[str], set[str]] = defaultdict(set)
        for root_id, covered_events in root_events.items():
            if covered_events:
                coverage_roots[frozenset(covered_events)].add(root_id)
        coverages = list(coverage_roots)
        maximal_coverages = [
            coverage
            for coverage in coverages
            if not any(coverage < other for other in coverages)
        ]

        memberships: dict[str, list[frozenset[str]]] = defaultdict(list)
        for coverage in maximal_coverages:
            for event_id in coverage:
                memberships[event_id].append(coverage)

        partitions: list[dict[str, Any]] = []
        for coverage in maximal_coverages:
            assigned = sorted(
                event_id
                for event_id in coverage
                if len(memberships.get(event_id) or []) == 1
            )
            if assigned:
                partitions.append(
                    {
                        "event_ids": assigned,
                        "partition_reason": "common_upstream_root",
                    }
                )

        bridge_events: dict[frozenset[str], list[str]] = defaultdict(list)
        for event_id in event_ids:
            if len(memberships.get(event_id) or []) > 1:
                bridge_events[frozenset(roots_by_event.get(event_id) or set())].append(event_id)
        for root_set, covered_events in bridge_events.items():
            partitions.append(
                {
                    "event_ids": sorted(covered_events),
                    "partition_reason": "overlapping_upstream_roots",
                    "overlapping_root_node_ids": sorted(root_set),
                }
            )

        for event_id in event_ids:
            if not roots_by_event.get(event_id):
                partitions.append(
                    {
                        "event_ids": [event_id],
                        "partition_reason": "unmapped_event",
                    }
                )

        assigned_events = [
            event_id
            for partition in partitions
            for event_id in partition["event_ids"]
        ]
        if len(assigned_events) != len(set(assigned_events)) or set(assigned_events) != set(event_ids):
            raise RuntimeError(
                "common-root partition must assign every event exactly once"
            )
        return sorted(
            partitions,
            key=lambda partition: tuple(partition["event_ids"]),
        )

    @classmethod
    def _diagnostic_partitions(
        cls,
        event_ids: list[str],
        roots_by_event: dict[str, set[str]],
    ) -> list[dict[str, Any]]:
        return [
            {
                **{
                    key: value
                    for key, value in partition.items()
                    if key != "event_ids"
                },
                "diagnostic_event_ids": list(
                    partition.get("event_ids") or []
                ),
            }
            for partition in cls._event_partitions(
                event_ids,
                roots_by_event,
            )
        ]

    def explanation_components(
        self,
        issues: list[dict[str, Any]],
        *,
        semantic_state: dict[str, Any] | None = None,
        semantic_contract_evidence_by_ref: (
            dict[str, dict[str, Any]] | None
        ) = None,
    ) -> list[dict[str, Any]]:
        if semantic_state:
            self.add_semantic_gap_ledger(
                semantic_state,
                contract_evidence_by_ref=(
                    semantic_contract_evidence_by_ref
                ),
            )
        self.add_diagnostics(issues)
        if semantic_state:
            self.bind_diagnostics_to_semantic_transactions(
                issues,
                semantic_state,
            )
        event_ids = list(
            dict.fromkeys(
                str(issue.get("diagnostic_event_id") or stable_id("diag", repr(issue)))
                for issue in issues
            )
        )
        roots_by_event = {event_id: self.upstream_root_nodes(event_id) for event_id in event_ids}
        partitions = self._diagnostic_partitions(event_ids, roots_by_event)

        issue_by_event = {
            str(issue.get("diagnostic_event_id") or stable_id("diag", repr(issue))): issue
            for issue in issues
        }
        result: list[dict[str, Any]] = []
        for partition in partitions:
            component_events = list(partition["diagnostic_event_ids"])
            sets = [roots_by_event[event_id] for event_id in component_events]
            common = set.intersection(*sets) if sets and all(sets) else set()
            roots = self._most_specific_roots(common)
            root_records = [self._natural_root_record(node_id) for node_id in roots]
            concrete_root_records = [
                record
                for record in root_records
                if str(record.get("entity_kind") or "")
                not in _UNMAPPED_ROOT_NODE_KINDS
            ]
            ambiguous = len(root_records) != 1
            if len(root_records) == 1:
                selected = dict(root_records[0])
                if not concrete_root_records:
                    selected["title"] = "unmapped_diagnostic"
            else:
                unresolved_key = (
                    stable_id("root-set", *roots)
                    if root_records
                    else stable_id("unmapped", *component_events)
                )
                selected = {
                    "key": unresolved_key,
                    "node_id": None,
                    "title": "code_dependency_root_set" if root_records else "unmapped_diagnostic",
                    "scope": unresolved_key,
                    "source": "code_index_common_upstream_partition_v3",
                    "root_shape": "mapped_root_set" if root_records else "unmapped",
                    "ambiguous": bool(root_records),
                    "unmapped": not root_records,
                }
            selected["diagnostic_event_ids"] = component_events
            selected["reachable_diagnostic_count"] = len(component_events)
            selected["alternatives"] = root_records if ambiguous else []
            selected["ambiguous"] = ambiguous and bool(root_records)
            selected["unmapped"] = not concrete_root_records
            result.append(
                {
                    "diagnostic_event_ids": component_events,
                    "issues": [issue_by_event[event_id] for event_id in component_events],
                    "root": selected,
                    "root_alternatives": root_records if ambiguous else [],
                    "common_root_node_ids": sorted(common),
                    "mapping_statuses": [self.diagnostic_mappings[event_id] for event_id in component_events],
                    "partition_reason": partition["partition_reason"],
                    "overlapping_root_node_ids": list(partition.get("overlapping_root_node_ids") or []),
                }
            )
        return result

    def diagnostic_graph_dict(
        self,
        issues: list[dict[str, Any]],
        *,
        selected_root: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_ids = {
            str(issue.get("diagnostic_event_id") or stable_id("diag", repr(issue)))
            for issue in issues
        }
        mappings = [self.diagnostic_mappings[event_id] for event_id in sorted(event_ids) if event_id in self.diagnostic_mappings]
        roots = {root_id for event_id in event_ids for root_id in self.upstream_root_nodes(event_id)}
        root_records = [self._natural_root_record(root_id) for root_id in sorted(roots)]
        selected_key = str((selected_root or {}).get("key") or "")
        selected_root_id = str(
            (selected_root or {}).get("node_id")
            or (selected_root or {}).get("entity_id")
            or ""
        )
        target_roots = {selected_root_id} if selected_root_id else set(roots)
        diagnostic_ids = {f"diagnostic:{event_id}" for event_id in event_ids}

        reference_edges = {
            "call",
            "memberref",
            "typeref",
            "macro",
            "declref",
            "callback_provider_binding",
            "callback_binding_pair",
        }
        entity_edges = {
            "candidate_resolution",
            "diagnostic_contract_dependency",
            "instance_of_contract",
            "part_of_patch_transaction",
            "depends_on_patch_transaction",
            "member_of",
            "has_type",
            "owned_by_contract_file",
            "same_contract_as",
        }

        def is_causal_edge(edge: dict[str, Any]) -> bool:
            source_id = str(edge.get("from") or "")
            edge_type = str(edge.get("type") or "")
            source_kind = str((self.graph.nodes.get(source_id) or {}).get("kind") or "")
            if source_kind == "Diagnostic":
                return edge_type in {
                    "diagnoses",
                    "diagnoses_callback_binding_provider",
                    "compiler_suggests_contract",
                }
            if source_kind == "Reference":
                return edge_type in reference_edges or edge_type == "unresolved_dependency"
            if source_kind == "Include":
                return edge_type in {"resolves_to", "unresolved_dependency"}
            if source_kind in _UNRESOLVED_PROVIDER_NODE_KINDS:
                return edge_type == "candidate_resolution"
            return edge_type in entity_edges

        causal_edges = [edge for edge in self.graph.edges if is_causal_edge(edge)]
        adjacency: dict[str, set[str]] = defaultdict(set)
        reverse: dict[str, set[str]] = defaultdict(set)
        for edge in causal_edges:
            source = str(edge.get("from") or "")
            target = str(edge.get("to") or "")
            adjacency[source].add(target)
            reverse[target].add(source)

        def reachable(starts: Iterable[str], graph: dict[str, set[str]]) -> set[str]:
            result: set[str] = set()
            queue = deque(str(item) for item in starts if str(item))
            while queue:
                node_id = queue.popleft()
                if node_id in result:
                    continue
                result.add(node_id)
                queue.extend(graph.get(node_id, set()))
            return result

        if target_roots:
            relevant_nodes = reachable(diagnostic_ids, adjacency) & reachable(target_roots, reverse)
            relevant_nodes.update(diagnostic_ids)
            relevant_nodes.update(target_roots)
        else:
            relevant_nodes = set(diagnostic_ids)
            relevant_nodes.update(
                str(node_id)
                for mapping in mappings
                for node_id in mapping.get("mapped_node_ids") or []
                if str(node_id)
            )

        # Exact source-position mappings are evidence owned by the diagnostic,
        # even when the selected upstream root is an unresolved contract.  A
        # root-path intersection alone cannot retain the donor side of these
        # owner alignments because the unresolved node is not their parent.
        exact_dependency_evidence = [
            dict(mapping.get("symbol_evidence") or {})
            for mapping in mappings
            if isinstance(mapping.get("symbol_evidence"), dict)
        ]
        exact_dependency_owner_ids = {
            str(node_id)
            for evidence in exact_dependency_evidence
            for node_id in evidence.get("exact_dependency_owner_ids") or []
            if str(node_id)
        }
        exact_dependency_counterpart_ids = {
            str(node_id)
            for evidence in exact_dependency_evidence
            for node_id in evidence.get("exact_dependency_counterpart_ids") or []
            if str(node_id)
        }
        exact_dependency_ids = (
            exact_dependency_owner_ids | exact_dependency_counterpart_ids
        )
        relevant_nodes.update(exact_dependency_ids)

        context_edges: list[dict[str, Any]] = []
        diagnostic_dependency_targets: set[str] = set()
        for edge in self.graph.edges:
            source = str(edge.get("from") or "")
            target = str(edge.get("to") or "")
            edge_type = str(edge.get("type") or "")
            target_node = self.graph.nodes.get(target) or {}
            target_tree = str(target_node.get("tree") or "")
            if edge_type == "occurs_at" and source in diagnostic_ids:
                relevant_nodes.add(target)
                context_edges.append(edge)
            elif edge_type == "callback_contract_consumer" and source in target_roots:
                relevant_nodes.add(target)
                context_edges.append(edge)
            elif (
                edge_type == "same_contract_as"
                and source in exact_dependency_ids
                and target in exact_dependency_ids
            ):
                context_edges.append(edge)
            elif (
                edge_type == "diagnostic_contract_dependency"
                and source in relevant_nodes
            ):
                relevant_nodes.add(target)
                diagnostic_dependency_targets.add(target)
                context_edges.append(edge)
            elif edge_type == "reported_for_module" and source in diagnostic_ids:
                relevant_nodes.add(target)
                context_edges.append(edge)
            elif edge_type == "occurs_in" and source in relevant_nodes and target_tree == "old":
                relevant_nodes.add(target)
                context_edges.append(edge)
            elif (
                edge_type == "diagnoses"
                and source in diagnostic_ids
                and target_tree == "old"
                and target_node.get("extent")
            ):
                relevant_nodes.add(target)
                context_edges.append(edge)

        diagnostic_dependency_closure = self._upstream_reachable_nodes(
            diagnostic_dependency_targets
        )
        if diagnostic_dependency_closure:
            relevant_nodes.update(diagnostic_dependency_closure)
            context_edges.extend(
                edge
                for edge in causal_edges
                if str(edge.get("from") or "")
                in diagnostic_dependency_closure
                and str(edge.get("to") or "")
                in diagnostic_dependency_closure
            )

        module_node_ids = {
            str(edge.get("to") or "")
            for edge in context_edges
            if str(edge.get("type") or "") == "reported_for_module"
        }
        module_object_ids = {
            str(edge.get("from") or "")
            for edge in self.graph.edges
            if str(edge.get("type") or "") == "links_into"
            and str(edge.get("to") or "") in module_node_ids
        }
        module_file_ids = {
            str(edge.get("from") or "")
            for edge in self.graph.edges
            if str(edge.get("type") or "") == "compiles_to"
            and str(edge.get("to") or "") in module_object_ids
        }
        relevant_nodes.update(module_object_ids)
        relevant_nodes.update(module_file_ids)
        context_edges.extend(
            edge
            for edge in self.graph.edges
            if (
                str(edge.get("type") or "") == "links_into"
                and str(edge.get("from") or "") in module_object_ids
                and str(edge.get("to") or "") in module_node_ids
            )
            or (
                str(edge.get("type") or "") == "compiles_to"
                and str(edge.get("from") or "") in module_file_ids
                and str(edge.get("to") or "") in module_object_ids
            )
        )

        repair_dependency_target_ids: set[str] = set()
        repair_dependency_reference_ids: set[str] = set()
        repair_dependency_owner_ids: set[str] = set()
        donor_implementation_root_ids = {
            root_id
            for root_id in target_roots
            if str((self.graph.nodes.get(root_id) or {}).get("tree") or "").startswith("donor_")
        }
        donor_comparison_counterpart_ids: set[str] = set()
        for edge in self.graph.edges:
            if (
                str(edge.get("type") or "") == "same_contract_as"
                and str(edge.get("from") or "") in target_roots
            ):
                counterpart_id = str(edge.get("to") or "")
                counterpart_tree = str(
                    (self.graph.nodes.get(counterpart_id) or {}).get("tree") or ""
                )
                if counterpart_tree.startswith("donor_"):
                    donor_comparison_counterpart_ids.add(counterpart_id)

        implementation_owners_by_reference: dict[str, set[str]] = defaultdict(set)
        for edge in self.graph.edges:
            reference_id = str(edge.get("from") or "")
            owner_id = str(edge.get("to") or "")
            if (
                str(edge.get("type") or "") == "occurs_in"
                and owner_id in donor_implementation_root_ids
            ):
                implementation_owners_by_reference[reference_id].add(owner_id)
        implementation_reference_ids = set(implementation_owners_by_reference)
        reference_edge_types = {"call", "memberref", "typeref", "macro", "declref"}
        old_counterparts_by_entity: dict[str, set[str]] = defaultdict(set)
        for edge in self.graph.edges:
            if str(edge.get("type") or "") != "same_contract_as":
                continue
            source_id = str(edge.get("from") or "")
            old_id = str(edge.get("to") or "")
            if str((self.graph.nodes.get(old_id) or {}).get("tree") or "") == "old":
                old_counterparts_by_entity[source_id].add(old_id)
        for edge in self.graph.edges:
            reference_id = str(edge.get("from") or "")
            edge_type = str(edge.get("type") or "")
            if reference_id not in implementation_reference_ids or edge_type not in reference_edge_types:
                continue
            target_id = str(edge.get("to") or "")
            target = self.graph.nodes.get(target_id) or {}
            target_tree = str(target.get("tree") or "")
            target_file = normalize_relpath(str(target.get("file") or target.get("path") or ""))
            if not target_tree.startswith("donor_") or not target_file:
                continue
            owner_ids = implementation_owners_by_reference.get(reference_id) or set()
            if not any(
                reference_crosses_entity_boundary(
                    {
                        **dict(self.graph.nodes.get(reference_id) or {}),
                        "owner_entity": self.graph.nodes.get(owner_id) or {},
                        "referenced_entity": target,
                    }
                )
                for owner_id in owner_ids
            ):
                continue
            dependency_scope = str(
                (self.graph.nodes.get(reference_id) or {}).get("dependency_scope")
                or "unresolved"
            )
            old_counterpart_ids = old_counterparts_by_entity.get(
                target_id,
                set(),
            )
            target_digest = str(target.get("digest") or "")
            equivalent_old_contract = bool(
                target_digest
                and old_counterpart_ids
                and any(
                    str(
                        (self.graph.nodes.get(old_id) or {}).get("digest")
                        or ""
                    )
                    == target_digest
                    for old_id in old_counterpart_ids
                )
            )
            if dependency_scope != "local_driver" and equivalent_old_contract:
                continue
            for owner_id in owner_ids:
                repair_dependency_owner_ids.add(owner_id)
                repair_dependency_target_ids.add(target_id)
                repair_dependency_reference_ids.add(reference_id)
                context_edges.append(
                    {
                        "from": owner_id,
                        "to": target_id,
                        "type": "implementation_depends_on",
                        "reference_id": reference_id,
                        "use_kind": edge_type,
                        "dependency_scope": dependency_scope,
                    }
                )

        repair_context_ids = {
            *donor_implementation_root_ids,
            *donor_comparison_counterpart_ids,
            *repair_dependency_owner_ids,
            *repair_dependency_reference_ids,
            *repair_dependency_target_ids,
        }
        if repair_context_ids:
            relevant_nodes.update(repair_context_ids)
            for edge in self.graph.edges:
                source = str(edge.get("from") or "")
                target = str(edge.get("to") or "")
                edge_type = str(edge.get("type") or "")
                if edge_type == "same_contract_as" and (
                    source in target_roots | repair_dependency_target_ids
                    or target in target_roots | repair_dependency_target_ids
                ):
                    relevant_nodes.update([source, target])
                    context_edges.append(edge)
                elif edge_type in {
                    "occurs_in",
                    "call",
                    "memberref",
                    "typeref",
                    "macro",
                    "declref",
                } and source in repair_context_ids and target in repair_context_ids:
                    context_edges.append(edge)
                elif edge_type in {"defined_in", "declared_in", "member_of", "has_type"}:
                    if source in repair_dependency_target_ids:
                        relevant_nodes.add(target)
                        context_edges.append(edge)

        causal_path_edges = [
            edge
            for edge in causal_edges
            if str(edge.get("from") or "") in relevant_nodes
            and str(edge.get("to") or "") in relevant_nodes
        ]
        edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for edge in [*causal_path_edges, *context_edges]:
            key = (
                str(edge.get("from") or ""),
                str(edge.get("to") or ""),
                str(edge.get("type") or ""),
            )
            edges_by_key.setdefault(key, edge)
        edges = [edges_by_key[key] for key in sorted(edges_by_key)]
        nodes = [
            self.graph.nodes[node_id]
            for node_id in sorted(relevant_nodes)
            if node_id in self.graph.nodes
        ]
        return {
            "schema": "diagnostic_code_dependency_graph_v3",
            "nodes": nodes,
            "edges": edges,
            "diagnostic_mappings": mappings,
            "reachable_upstream_root_clusters": [
                {
                    **record,
                    "selected": record.get("key") == selected_key,
                    "diagnostic_event_ids": sorted(
                        event_id
                        for event_id in event_ids
                        if record.get("node_id") in self.upstream_root_nodes(event_id)
                    ),
                }
                for record in root_records
            ],
            "summary": {
                "diagnostic_count": len(event_ids),
                "mapped_diagnostic_count": sum(not item.get("unmapped") for item in mappings),
                "unmapped_diagnostic_count": sum(bool(item.get("unmapped")) for item in mappings),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "reachable_upstream_root_count": len(root_records),
                "selected_root_cause_key": selected_key or None,
                "direct_repair_dependency_count": len(repair_dependency_target_ids),
                "direct_repair_dependency_node_ids": sorted(repair_dependency_target_ids),
                "truncated": self.graph.truncated,
                "truncation_reasons": sorted(self.graph.truncation_reasons),
            },
        }
