#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def _enum_name(value: object) -> str:
    return str(getattr(value, "name", "") or "")


def _digest(payload: object) -> str:
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8", errors="ignore")).hexdigest()


def _canonical_type_spelling(value: object) -> str:
    if value is None:
        return ""
    try:
        canonical = value.get_canonical()
    except Exception:
        canonical = value
    return str(getattr(canonical, "spelling", "") or getattr(value, "spelling", "") or "")


def _cursor_kind(cursor: object | None) -> str:
    return _enum_name(getattr(cursor, "kind", None)) if cursor is not None else ""


def _cursor_children(cursor: object | None) -> Iterable[object]:
    if cursor is None:
        return ()
    try:
        return tuple(cursor.get_children())
    except Exception:
        return ()


def _walk_cursor(cursor: object | None) -> Iterable[object]:
    for child in _cursor_children(cursor):
        yield child
        yield from _walk_cursor(child)


def _cursor_arguments(cursor: object) -> list[object]:
    try:
        return list(cursor.get_arguments())
    except Exception:
        return []


def _is_variadic(cursor: object) -> bool:
    cursor_type = getattr(cursor, "type", None)
    try:
        return bool(cursor_type.is_function_variadic())
    except Exception:
        return False


def cursor_signature_payload(cursor: object) -> dict[str, Any]:
    kind = _cursor_kind(cursor)
    arguments = _cursor_arguments(cursor)
    result_type = getattr(cursor, "result_type", None)
    return {
        "cursor_kind": kind,
        "canonical_type": _canonical_type_spelling(getattr(cursor, "type", None)),
        "result_type": _canonical_type_spelling(result_type),
        "argument_types": [
            _canonical_type_spelling(getattr(argument, "type", None))
            for argument in arguments
        ],
        "variadic": _is_variadic(cursor),
    }


_STORAGE_CLASS_CURSOR_KINDS = {
    "FUNCTION_DECL",
    "CXX_METHOD",
    "CONSTRUCTOR",
    "DESTRUCTOR",
    "VAR_DECL",
}

_LINKAGE_CURSOR_KINDS = {
    *_STORAGE_CLASS_CURSOR_KINDS,
    "STRUCT_DECL",
    "UNION_DECL",
    "CLASS_DECL",
    "ENUM_DECL",
    "TYPEDEF_DECL",
    "TYPE_ALIAS_DECL",
}


def cursor_interface_payload(cursor: object) -> dict[str, Any]:
    kind = _cursor_kind(cursor)
    return {
        "storage_class": (
            _enum_name(getattr(cursor, "storage_class", None))
            if kind in _STORAGE_CLASS_CURSOR_KINDS
            else ""
        ),
        "linkage": (
            _enum_name(getattr(cursor, "linkage", None))
            if kind in _LINKAGE_CURSOR_KINDS
            else ""
        ),
        "visibility": (
            _enum_name(getattr(cursor, "visibility", None))
            if kind in _LINKAGE_CURSOR_KINDS
            else ""
        ),
        "availability": _enum_name(getattr(cursor, "availability", None)),
    }


def _local_identifier_names(cursor: object) -> list[str]:
    names: list[str] = []
    for node in [*_cursor_arguments(cursor), *_walk_cursor(cursor)]:
        if _cursor_kind(node) not in {"PARM_DECL", "VAR_DECL"}:
            continue
        spelling = str(getattr(node, "spelling", "") or "").strip()
        if spelling and spelling not in names:
            names.append(spelling)
    return names


def _token_records(cursor: object) -> list[tuple[str, str]]:
    try:
        return [
            (_enum_name(getattr(token, "kind", None)), str(getattr(token, "spelling", "") or ""))
            for token in cursor.get_tokens()
        ]
    except Exception:
        return []


def _implementation_token_slice(
    records: list[tuple[str, str]],
    *,
    cursor_kind: str,
) -> list[tuple[str, str]]:
    if cursor_kind in {"FUNCTION_DECL", "CXX_METHOD", "CONSTRUCTOR", "DESTRUCTOR"}:
        body_start = next(
            (index for index, (_kind, spelling) in enumerate(records) if spelling == "{"),
            None,
        )
        if body_start is None:
            return []
        return records[body_start:]
    if cursor_kind == "VAR_DECL":
        initializer_start = next(
            (index for index, (_kind, spelling) in enumerate(records) if spelling == "="),
            None,
        )
        return records[initializer_start:] if initializer_start is not None else []
    return records


def cursor_implementation_payload(cursor: object, *, symbol: str) -> dict[str, Any] | None:
    cursor_kind = _cursor_kind(cursor)
    records = _implementation_token_slice(_token_records(cursor), cursor_kind=cursor_kind)
    if not records:
        return None

    local_names = _local_identifier_names(cursor)
    local_map = {name: f"<local:{index}>" for index, name in enumerate(local_names)}
    normalized: list[str] = []
    for token_kind, spelling in records:
        if not spelling:
            continue
        if token_kind == "COMMENT":
            continue
        if token_kind == "IDENTIFIER":
            if spelling == symbol:
                normalized.append("<self>")
            elif spelling in local_map:
                normalized.append(local_map[spelling])
            else:
                normalized.append(spelling)
        else:
            normalized.append(spelling)
    return {
        "normalized_tokens": normalized,
        "normalized_token_count": len(normalized),
    }


def cursor_fingerprints(cursor: object, *, symbol: str) -> dict[str, Any]:
    signature = cursor_signature_payload(cursor)
    interface = cursor_interface_payload(cursor)
    implementation = cursor_implementation_payload(cursor, symbol=symbol)
    return {
        "signature": signature,
        "signature_fingerprint": _digest(signature),
        "interface": interface,
        "interface_fingerprint": _digest(interface),
        "implementation": implementation,
        "implementation_fingerprint": _digest(implementation) if implementation is not None else None,
    }


def entity_is_internally_linked(record: dict[str, Any]) -> bool:
    interface = record.get("interface") if isinstance(record.get("interface"), dict) else {}
    storage = str(interface.get("storage_class") or record.get("storage_class") or "").upper()
    linkage = str(interface.get("linkage") or record.get("linkage") or "").upper()
    return storage == "STATIC" or linkage in {"INTERNAL", "NO_LINKAGE"}


def compare_entity_fingerprints(
    old: dict[str, Any] | None,
    donor: dict[str, Any] | None,
) -> dict[str, Any]:
    if old is None or donor is None:
        return {
            "old_present": old is not None,
            "donor_present": donor is not None,
            "signature": "unavailable",
            "implementation": "unavailable",
            "interface": "unavailable",
        }

    def relation(field: str) -> str:
        old_value = old.get(field)
        donor_value = donor.get(field)
        if old_value in (None, "") or donor_value in (None, ""):
            return "unavailable"
        return "equivalent" if old_value == donor_value else "different"

    signature = relation("signature_fingerprint")
    implementation = relation("implementation_fingerprint")
    interface = relation("interface_fingerprint")
    exact_definition_source = "unavailable"
    if (
        bool(old.get("is_definition"))
        and bool(donor.get("is_definition"))
        and str(old.get("kind") or "")
        == str(donor.get("kind") or "")
        and str(old.get("contract_key") or "")
        == str(donor.get("contract_key") or "")
    ):
        old_source_fingerprint = str(
            old.get("syntax_fingerprint")
            or old.get("source_fingerprint")
            or old.get("digest")
            or ""
        )
        donor_source_fingerprint = str(
            donor.get("syntax_fingerprint")
            or donor.get("source_fingerprint")
            or donor.get("digest")
            or ""
        )
        if old_source_fingerprint and donor_source_fingerprint:
            exact_definition_source = (
                "equivalent"
                if old_source_fingerprint == donor_source_fingerprint
                else "different"
            )
    if exact_definition_source == "equivalent":
        # A structural fingerprint covers the complete definition extent.
        # Exact equality therefore proves all three source-level dimensions
        # even when libclang could not materialize their individual payloads.
        if signature == "unavailable":
            signature = "equivalent"
        if implementation == "unavailable":
            implementation = "equivalent"
        if interface == "unavailable":
            interface = "equivalent"
    if (
        implementation == "unavailable"
        and bool(old.get("is_definition"))
        and bool(donor.get("is_definition"))
        and str(old.get("kind") or "")
        == str(donor.get("kind") or "")
        and str(old.get("contract_key") or "")
        == str(donor.get("contract_key") or "")
        and old.get("digest") not in (None, "")
        and donor.get("digest") not in (None, "")
    ):
        implementation = (
            "equivalent"
            if old["digest"] == donor["digest"]
            else "different"
        )

    return {
        "old_present": True,
        "donor_present": True,
        "signature": signature,
        "implementation": implementation,
        "interface": interface,
        "exact_definition_source": exact_definition_source,
        "old_internal_linkage": entity_is_internally_linked(old),
        "donor_internal_linkage": entity_is_internally_linked(donor),
    }
