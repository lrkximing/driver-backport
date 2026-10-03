#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from typing import Any

from driver_backport.core.common import normalize_relpath, stable_id


PROMPT_EVIDENCE_SCHEMA = "prompt_evidence_view_v3"
PROMPT_REFERENCE_TABLE_SCHEMA = "prompt_reference_table_v1"
PROMPT_CONTENT_STORE_SCHEMA = "content_addressed_prompt_payload_v1"
PROMPT_RECORD_TABLE_SCHEMA = "prompt_record_table_v1"
PROMPT_CONTENT_REF_KEY = "prompt_content_ref"

_REFERENCE_OCCURRENCE_FIELDS = {
    "id",
    "reference_id",
    "contract_evidence_id",
    "line",
    "column",
    "extent",
}


def _json_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_bytes(value: Any) -> int:
    return len(_canonical_json(value).encode("utf-8"))


def _content_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def compact_prompt_references(values: object) -> list[dict[str, Any]] | dict[str, Any]:
    """Store common graph-edge facts once while preserving every occurrence.

    A reference is a graph edge plus a concrete source occurrence.  Grouping
    is based on the complete projected edge identity, including full endpoint
    content, rather than on a symbol or entity id.  This prevents entities
    with the same logical id but different preprocessor-context extents from
    being merged.
    """

    projected = [
        record
        for raw in (values if isinstance(values, (list, tuple)) else [])
        if (record := prompt_reference_identity(raw)) is not None
    ]
    if len(projected) < 2:
        return projected

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    shared_values: dict[str, dict[str, Any]] = {}
    for ordinal, record in enumerate(projected):
        shared = {
            key: value
            for key, value in record.items()
            if key not in _REFERENCE_OCCURRENCE_FIELDS
        }
        signature = _content_digest(shared)
        previous = shared_values.setdefault(signature, shared)
        if _canonical_json(previous) != _canonical_json(shared):
            raise ValueError("prompt reference content digest collision")
        grouped.setdefault(signature, []).append((ordinal, record))

    groups: list[dict[str, Any]] = []
    for signature, records in grouped.items():
        shared = dict(shared_values[signature])
        groups.append(
            {
                "shared": shared,
                "occurrences": [
                    {
                        "ordinal": ordinal,
                        **{
                            key: value
                            for key, value in record.items()
                            if key in _REFERENCE_OCCURRENCE_FIELDS
                        },
                    }
                    for ordinal, record in records
                ],
            }
        )

    table = {
        "schema": PROMPT_REFERENCE_TABLE_SCHEMA,
        "record_count": len(projected),
        "groups": groups,
    }
    restored: list[dict[str, Any] | None] = [None] * len(projected)
    for group in groups:
        for occurrence in group["occurrences"]:
            row = dict(occurrence)
            ordinal = int(row.pop("ordinal"))
            restored[ordinal] = {
                **group["shared"],
                **row,
            }
    if _canonical_json(restored) != _canonical_json(projected):
        raise ValueError("prompt reference table is not losslessly reversible")
    return table if _json_bytes(table) < _json_bytes(projected) else projected


def _expand_prompt_record_table(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema") != (
        PROMPT_RECORD_TABLE_SCHEMA
    ):
        return [
            dict(record)
            for record in value
            if isinstance(record, dict)
        ] if isinstance(value, list) else []
    columns = [str(column) for column in value.get("columns") or []]
    if value.get("row_encoding") == "presence_bitmap_and_values":
        restored: list[dict[str, Any]] = []
        for row in value.get("rows") or []:
            if not isinstance(row, list) or len(row) != 2:
                continue
            presence, fields = row
            if (
                not isinstance(presence, str)
                or len(presence) != len(columns)
                or set(presence) - {"0", "1"}
                or not isinstance(fields, list)
                or presence.count("1") != len(fields)
            ):
                continue
            restored.append(
                dict(
                    zip(
                        [
                            column
                            for column, present in zip(columns, presence)
                            if present == "1"
                        ],
                        fields,
                    )
                )
            )
        return restored
    return [
        dict(zip(columns, row))
        for row in value.get("rows") or []
        if isinstance(row, list) and len(row) == len(columns)
    ]


def compact_prompt_records(
    values: object,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Encode repeated dictionaries without dropping fields or presence."""

    if not isinstance(values, list) or any(
        not isinstance(record, dict) for record in values
    ):
        return list(values) if isinstance(values, list) else []
    records = [dict(record) for record in values]
    if len(records) < 4:
        return records
    if any(
        not isinstance(key, str)
        for record in records
        for key in record
    ):
        return records
    columns = sorted({key for record in records for key in record})
    candidates: list[dict[str, Any]] = []
    if all(set(record) == set(columns) for record in records):
        candidates.append(
            {
                "schema": PROMPT_RECORD_TABLE_SCHEMA,
                "columns": columns,
                "rows": [
                    [record[column] for column in columns]
                    for record in records
                ],
            }
        )
    candidates.append(
        {
            "schema": PROMPT_RECORD_TABLE_SCHEMA,
            "row_encoding": "presence_bitmap_and_values",
            "columns": columns,
            "rows": [
                [
                    "".join(
                        "1" if column in record else "0"
                        for column in columns
                    ),
                    [
                        record[column]
                        for column in columns
                        if column in record
                    ],
                ]
                for record in records
            ],
        }
    )
    table = min(candidates, key=_json_bytes)
    if _canonical_json(_expand_prompt_record_table(table)) != (
        _canonical_json(records)
    ):
        raise ValueError("prompt record table is not losslessly reversible")
    return (
        table
        if _json_bytes(table) * 10 <= _json_bytes(records) * 9
        else records
    )


def _collect_repeated_prompt_records(
    value: Any,
    *,
    counts: dict[str, int],
    records: dict[str, Any],
) -> None:
    if not isinstance(value, (dict, list)):
        return
    digest = _content_digest(value)
    previous = records.setdefault(digest, value)
    if _canonical_json(previous) != _canonical_json(value):
        raise ValueError("prompt content digest collision")
    counts[digest] = counts.get(digest, 0) + 1
    children = value.values() if isinstance(value, dict) else value
    for child in children:
        _collect_repeated_prompt_records(
            child,
            counts=counts,
            records=records,
        )


def _replace_prompt_records(
    value: Any,
    *,
    aliases: dict[str, str],
    preserve_digest: str | None = None,
) -> Any:
    if isinstance(value, (dict, list)):
        digest = _content_digest(value)
        if digest in aliases and digest != preserve_digest:
            return {PROMPT_CONTENT_REF_KEY: aliases[digest]}
        if isinstance(value, dict):
            return {
                key: _replace_prompt_records(child, aliases=aliases)
                for key, child in value.items()
            }
        return [
            _replace_prompt_records(child, aliases=aliases)
            for child in value
        ]
    return value


def _expand_prompt_payload(value: dict[str, Any]) -> Any:
    catalog = {
        str(record["id"]): record
        for record in value.get("records") or []
    }
    cache: dict[str, Any] = {}

    def expand(item: Any) -> Any:
        if isinstance(item, dict) and set(item) == {PROMPT_CONTENT_REF_KEY}:
            record_id = str(item[PROMPT_CONTENT_REF_KEY])
            if record_id not in cache:
                cache[record_id] = expand(catalog[record_id]["value"])
            return cache[record_id]
        if isinstance(item, dict):
            return {key: expand(child) for key, child in item.items()}
        if isinstance(item, list):
            return [expand(child) for child in item]
        return item

    return expand(value.get("root"))


def encode_prompt_payload(value: Any) -> tuple[Any, dict[str, Any]]:
    """Encode repeated structured evidence once, without dropping any facts.

    Candidate selection is determined by exact serialized cost.  The encoded
    representation is used only when it is smaller than the original and its
    expansion exactly matches the original canonical JSON.
    """

    original_bytes = _json_bytes(value)
    counts: dict[str, int] = {}
    records: dict[str, Any] = {}
    _collect_repeated_prompt_records(value, counts=counts, records=records)
    candidate_digests: list[str] = []
    for digest, count in counts.items():
        if count < 2:
            continue
        record = records[digest]
        alias = "record-000000"
        inline_bytes = count * _json_bytes(record)
        reference_bytes = count * _json_bytes({PROMPT_CONTENT_REF_KEY: alias})
        catalog_bytes = _json_bytes(
            {
                "id": alias,
                "sha256": digest,
                "value": record,
            }
        )
        if reference_bytes + catalog_bytes < inline_bytes:
            candidate_digests.append(digest)
    candidate_digests.sort(
        key=lambda digest: (
            -(counts[digest] - 1) * _json_bytes(records[digest]),
            digest,
        )
    )
    aliases = {
        digest: f"record-{index:06d}"
        for index, digest in enumerate(candidate_digests, start=1)
    }
    encoded = {
        "schema": PROMPT_CONTENT_STORE_SCHEMA,
        "reference_semantics": (
            "recursively replace each prompt_content_ref object with the "
            "matching records[id].value"
        ),
        "root": _replace_prompt_records(value, aliases=aliases),
        "records": [
            {
                "id": aliases[digest],
                "sha256": digest,
                "value": _replace_prompt_records(
                    records[digest],
                    aliases=aliases,
                    preserve_digest=digest,
                ),
            }
            for digest in candidate_digests
        ],
    }
    encoded_bytes = _json_bytes(encoded)
    applied = bool(candidate_digests) and encoded_bytes < original_bytes
    if applied:
        expanded = _expand_prompt_payload(encoded)
        if _canonical_json(expanded) != _canonical_json(value):
            raise ValueError("content-addressed prompt payload is not lossless")
        rendered: Any = encoded
    else:
        rendered = value
        encoded_bytes = original_bytes
    return rendered, {
        "schema": PROMPT_CONTENT_STORE_SCHEMA,
        "applied": applied,
        "lossless_verified": True,
        "record_count": len(candidate_digests) if applied else 0,
        "original_serialized_bytes": original_bytes,
        "encoded_serialized_bytes": encoded_bytes,
        "saved_serialized_bytes": original_bytes - encoded_bytes,
    }


def prompt_transport_instruction(value: Any) -> str:
    """Describe only the transport encodings actually present in a prompt."""

    def contains_schema(item: Any, schema: str) -> bool:
        if isinstance(item, dict):
            if item.get("schema") == schema:
                return True
            return any(
                contains_schema(child, schema)
                for child in item.values()
            )
        if isinstance(item, list):
            return any(contains_schema(child, schema) for child in item)
        return False

    instructions: list[str] = []
    if contains_schema(value, PROMPT_CONTENT_STORE_SCHEMA):
        instructions.append(
            "For content_addressed_prompt_payload_v1, recursively expand "
            "each prompt_content_ref through records before reasoning. "
            "record-* identifiers are lossless transport aliases rather "
            "than domain evidence or action identifiers; cite identifiers "
            "inside the expanded values when the response requires evidence."
        )
    if contains_schema(value, PROMPT_REFERENCE_TABLE_SCHEMA):
        instructions.append(
            "For prompt_reference_table_v1, merge each group's shared fields "
            "with every occurrence and restore occurrence order by ordinal; "
            "no reference location has been omitted."
        )
    if contains_schema(value, PROMPT_RECORD_TABLE_SCHEMA):
        instructions.append(
            "For prompt_record_table_v1, dense rows zip directly with "
            "columns. For presence_bitmap_and_values rows, select columns "
            "whose mask position is 1 and zip them with the row values. "
            "No field, absent-field distinction, or record has been omitted."
        )
    return " ".join(instructions)


def render_prompt_payload(
    value: Any,
    *,
    indent: int | None = None,
    sort_keys: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    """Render every structured model input through the shared transport."""

    encoded, metrics = encode_prompt_payload(value)
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": sort_keys,
    }
    if indent is None:
        options["separators"] = (",", ":")
    else:
        options["indent"] = indent
    return (
        json.dumps(encoded, **options),
        prompt_transport_instruction(encoded),
        metrics,
    )


def _text_digest(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _source_record_entity_ids(record: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            str(item)
            for item in [
                record.get("entity_id"),
                record.get("code_graph_node_id"),
                record.get("node_id"),
                record.get("materialized_entity_id"),
                *list(record.get("entity_ids") or []),
            ]
            if str(item or "")
        )
    )


def _source_record_alias(record: dict[str, Any]) -> dict[str, Any]:
    entity_ids = _source_record_entity_ids(record)
    return {
        key: value
        for key, value in {
            "id": record.get("id"),
            "entity_ids": entity_ids,
            "symbol": record.get("symbol"),
            "contract_key": record.get("contract_key"),
            "source_kind": record.get("source_kind"),
            "line_start": record.get("line_start"),
            "line_end": record.get("line_end"),
            "roles": list(record.get("roles") or []),
            "reason": record.get("reason"),
            "edit_permission": record.get("edit_permission"),
        }.items()
        if value not in (None, "", [], {})
    }


def _merged_source_record_aliases(
    *records: dict[str, Any],
) -> list[dict[str, Any]]:
    aliases: dict[str, dict[str, Any]] = {}
    for record in records:
        candidates = [
            dict(item)
            for item in record.get("source_aliases") or []
            if isinstance(item, dict)
        ]
        if not candidates:
            candidates = [_source_record_alias(record)]
        for alias in candidates:
            aliases.setdefault(_canonical_json(alias), alias)
    return [aliases[key] for key in sorted(aliases)]


def _source_record_inspect_handle_ids(
    record: dict[str, Any],
) -> list[str]:
    return list(
        dict.fromkeys(
            str(item)
            for item in [
                record.get("inspect_handle_id"),
                *list(record.get("inspect_handle_ids") or []),
            ]
            if str(item or "")
        )
    )


def source_evidence_record(
    record: dict[str, Any],
    *,
    role: str,
) -> dict[str, Any] | None:
    tree = str(record.get("tree") or ("donor" if "donor" in role else "old"))
    relative_path = normalize_relpath(
        str(record.get("file") or record.get("path") or "")
    )
    text = str(record.get("text") or "")
    if not relative_path or not text:
        return None
    entity_ids = _source_record_entity_ids(record)
    entity_id = entity_ids[0] if entity_ids else ""
    start = int(record.get("line_start") or 0)
    end = int(record.get("line_end") or start)
    content_digest = _text_digest(text)
    record_id = str(
        record.get("id")
        or stable_id(
            "prompt-source",
            tree,
            relative_path,
            entity_id,
            str(start),
            str(end),
            content_digest,
        )
    )
    return {
        "id": record_id,
        "tree": tree,
        "file": relative_path,
        "entity_id": entity_id or None,
        "entity_ids": entity_ids if len(entity_ids) > 1 else None,
        "symbol": record.get("symbol"),
        "contract_key": record.get("contract_key"),
        "line_start": start or None,
        "line_end": end or None,
        "source_kind": record.get("source_kind"),
        "reason": record.get("reason"),
        "edit_permission": record.get("edit_permission"),
        "roles": [role],
        "content_digest": content_digest,
        "text": text,
    }


def prompt_entity_identity(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    extent = (
        value.get("extent")
        if isinstance(value.get("extent"), dict)
        else {}
    )
    compact_extent = (
        {
            "start_line": extent.get("start_line"),
            "start_column": extent.get("start_column"),
            "end_line": extent.get("end_line"),
            "end_column": extent.get("end_column"),
        }
        if extent
        else None
    )
    record = {
        key: field
        for key, field in {
            "id": value.get("id") or value.get("entity_id"),
            "tree": value.get("tree"),
            "file": value.get("file") or value.get("source_file"),
            "kind": value.get("kind") or value.get("entity_kind"),
            "symbol": value.get("symbol") or value.get("name"),
            "contract_key": value.get("contract_key") or value.get("key"),
            "is_definition": value.get("is_definition"),
            "extent": compact_extent,
            "container": value.get("container"),
            "signature_fingerprint": value.get("signature_fingerprint"),
            "implementation_fingerprint": value.get(
                "implementation_fingerprint"
            ),
            "source_fingerprint": value.get("source_fingerprint"),
        }.items()
        if field not in (None, "", [], {})
    }
    return record or None


def prompt_reference_identity(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    extent = (
        value.get("extent")
        if isinstance(value.get("extent"), dict)
        else {}
    )
    record = {
        key: field
        for key, field in {
            "id": value.get("id") or value.get("reference_id"),
            "reference_id": value.get("reference_id"),
            "contract_evidence_id": value.get("contract_evidence_id"),
            "contract_identity": value.get("contract_identity"),
            "kind": value.get("kind"),
            "entity_kind": value.get("entity_kind"),
            "tree": value.get("tree"),
            "source_file": value.get("source_file"),
            "line": value.get("line"),
            "column": value.get("column"),
            "extent": (
                {
                    "start_line": extent.get("start_line"),
                    "start_column": extent.get("start_column"),
                    "end_line": extent.get("end_line"),
                    "end_column": extent.get("end_column"),
                }
                if extent
                else None
            ),
            "symbol": value.get("symbol"),
            "use_kind": value.get("use_kind"),
            "cursor_kind": value.get("cursor_kind"),
            "dependency_scope": value.get("dependency_scope"),
            "contract_key": value.get("contract_key"),
            "owner_entity": prompt_entity_identity(value.get("owner_entity")),
            "referenced_entity": prompt_entity_identity(
                value.get("referenced_entity")
            ),
        }.items()
        if field not in (None, "", [], {})
    }
    return record or None


def _prompt_alignment_proof(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    proof = (
        value.get("alignment")
        if isinstance(value.get("alignment"), dict)
        else value
    )
    return {
        key: field
        for key, field in {
            "donor_reference_id": value.get("donor_reference_id"),
            "old_reference_id": (
                (value.get("old_reference") or {}).get("id")
                if isinstance(value.get("old_reference"), dict)
                else value.get("old_reference_id")
            ),
            "status": proof.get("status"),
            "equivalence_proven": proof.get("equivalence_proven"),
            "reason": proof.get("reason"),
            "exact_context_candidate_count": proof.get(
                "exact_context_candidate_count"
            ),
            "owner_target_candidate_count": proof.get(
                "owner_target_candidate_count"
            ),
        }.items()
        if field not in (None, "", [], {})
    }


def prompt_contract_evidence(value: object) -> dict[str, Any]:
    """Project exact contract facts without repeating full graph entities."""

    if not isinstance(value, dict):
        return {}
    references = [
        item
        for item in value.get("consumer_references") or []
        if isinstance(item, dict)
    ]
    alignments = [
        item
        for item in value.get("consumer_reference_alignments") or []
        if isinstance(item, dict)
    ]
    alignment = (
        value.get("alignment")
        if isinstance(value.get("alignment"), dict)
        else {}
    )
    alignments.extend(
        item
        for item in alignment.get("reference_alignments") or []
        if isinstance(item, dict)
    )
    alignment_summary: dict[tuple[str, object], int] = {}
    unresolved_alignments: list[dict[str, Any]] = []
    for item in alignments:
        compact = _prompt_alignment_proof(item)
        key = (
            str(compact.get("status") or "unavailable"),
            compact.get("equivalence_proven"),
        )
        alignment_summary[key] = alignment_summary.get(key, 0) + 1
        if compact.get("equivalence_proven") is False:
            unresolved_alignments.append(compact)

    excluded = {
        "consumer_reference",
        "consumer_references",
        "consumer_reference_alignments",
        "alignment",
        "old_entity",
        "donor_entity",
        "old_declarations",
        "donor_declarations",
        "old_exports",
        "donor_exports",
        "post_materialization_validation",
    }
    result = {
        key: field
        for key, field in value.items()
        if key not in excluded and field not in (None, "", [], {})
    }
    result["old_entity"] = prompt_entity_identity(value.get("old_entity"))
    result["donor_entity"] = prompt_entity_identity(
        value.get("donor_entity")
    )
    for key in (
        "old_declarations",
        "donor_declarations",
        "old_exports",
        "donor_exports",
    ):
        records = [
            compact
            for item in value.get(key) or []
            if (compact := prompt_entity_identity(item)) is not None
        ]
        if records:
            result[key] = records
    single_reference = prompt_reference_identity(
        value.get("consumer_reference")
    )
    if single_reference:
        result["consumer_reference"] = single_reference
    result["consumer_reference_count"] = int(
        value.get("consumer_reference_count") or len(references)
    )
    result["consumer_reference_ids"] = [
        str(item.get("id") or "")
        for item in references
        if str(item.get("id") or "")
    ]
    result["consumer_reference_alignment_summary"] = [
        {
            "status": status,
            "equivalence_proven": proof,
            "count": count,
        }
        for (status, proof), count in sorted(
            alignment_summary.items(),
            key=lambda item: (item[0][0], str(item[0][1])),
        )
    ]
    if unresolved_alignments:
        result["unresolved_consumer_reference_alignments"] = (
            unresolved_alignments
        )
    post_validation = (
        value.get("post_materialization_validation")
        if isinstance(value.get("post_materialization_validation"), dict)
        else {}
    )
    if post_validation:
        result["post_materialization_validation"] = {
            key: field
            for key, field in post_validation.items()
            if key != "consumer_references"
            and field not in (None, "", [], {})
        }
    result["alignment"] = {
        key: field
        for key, field in alignment.items()
        if key != "reference_alignments"
        and field not in (None, "", [], {})
    }
    return {
        key: field
        for key, field in result.items()
        if field not in (None, "", [], {})
    }


def deduplicate_source_evidence(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        identity = (
            record.get("tree"),
            record.get("file"),
            record.get("line_start"),
            record.get("line_end"),
            record.get("content_digest"),
        )
        existing = unique.get(identity)
        if existing is None:
            unique[identity] = record
            continue
        source_aliases = _merged_source_record_aliases(
            existing,
            record,
        )
        existing["roles"] = list(
            dict.fromkeys(
                [
                    *list(existing.get("roles") or []),
                    *list(record.get("roles") or []),
                ]
            )
        )
        existing["entity_ids"] = list(
            dict.fromkeys(
                [
                    *_source_record_entity_ids(existing),
                    *_source_record_entity_ids(record),
                ]
            )
        )
        existing["source_aliases"] = source_aliases

    ordered = sorted(
        unique.values(),
        key=lambda item: (
            str(item.get("tree") or ""),
            str(item.get("file") or ""),
            int(item.get("line_start") or 0),
            -int(item.get("line_end") or 0),
            str(item.get("id") or ""),
        ),
    )
    retained: list[dict[str, Any]] = []
    for record in ordered:
        covered_by = next(
            (
                parent
                for parent in retained
                if parent.get("tree") == record.get("tree")
                and parent.get("file") == record.get("file")
                and int(parent.get("line_start") or 0) > 0
                and int(parent.get("line_start") or 0)
                <= int(record.get("line_start") or 0)
                and int(parent.get("line_end") or 0)
                >= int(record.get("line_end") or 0)
                and str(record.get("text") or "")
                in str(parent.get("text") or "")
            ),
            None,
        )
        if covered_by is None:
            retained.append(record)
            continue
        source_aliases = _merged_source_record_aliases(
            covered_by,
            record,
        )
        covered_by["covered_record_ids"] = list(
            dict.fromkeys(
                [
                    *list(covered_by.get("covered_record_ids") or []),
                    str(record.get("id") or ""),
                    *list(record.get("covered_record_ids") or []),
                ]
            )
        )
        covered_by["roles"] = list(
            dict.fromkeys(
                [
                    *list(covered_by.get("roles") or []),
                    *list(record.get("roles") or []),
                ]
            )
        )
        covered_by["entity_ids"] = list(
            dict.fromkeys(
                [
                    *_source_record_entity_ids(covered_by),
                    *_source_record_entity_ids(record),
                ]
            )
        )
        covered_by["source_aliases"] = source_aliases
    return retained


def _tree_family(value: object) -> str:
    tree = str(value or "").strip()
    return "donor" if tree.startswith("donor") else tree


def _bind_source_inspect_handles(
    source_records: list[dict[str, Any]],
    inspectable_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    direct_ids = {
        str(record.get("id") or "")
        for record in inspectable_entities
        if isinstance(record, dict) and str(record.get("id") or "")
    }
    identities: dict[tuple[str, str, str, str], list[str]] = {}
    for record in inspectable_entities:
        if not isinstance(record, dict):
            continue
        handle_id = str(record.get("id") or "")
        tree = _tree_family(record.get("tree"))
        path = normalize_relpath(str(record.get("path") or ""))
        if not handle_id or not tree or not path:
            continue
        for kind, identity in (
            ("contract", str(record.get("contract_key") or "")),
            ("symbol", str(record.get("symbol") or "")),
        ):
            if identity:
                identities.setdefault(
                    (tree, path, kind, identity),
                    [],
                ).append(handle_id)

    bound: list[dict[str, Any]] = []
    for raw in source_records:
        record = dict(raw)
        record.pop("inspect_handle_id", None)
        record.pop("inspect_handle_ids", None)
        identity_records = [
            record,
            *[
                alias
                for alias in record.get("source_aliases") or []
                if isinstance(alias, dict)
            ],
        ]
        candidates: list[str] = []
        for identity_record in identity_records:
            candidates.extend(
                entity_id
                for entity_id in _source_record_entity_ids(identity_record)
                if entity_id in direct_ids
            )
            tree = _tree_family(
                identity_record.get("tree") or record.get("tree")
            )
            path = normalize_relpath(
                str(
                    identity_record.get("file")
                    or identity_record.get("path")
                    or record.get("file")
                    or ""
                )
            )
            for kind, identity in (
                (
                    "contract",
                    str(identity_record.get("contract_key") or ""),
                ),
                ("symbol", str(identity_record.get("symbol") or "")),
            ):
                if not identity:
                    continue
                matches = identities.get(
                    (tree, path, kind, identity),
                    [],
                )
                if len(matches) == 1:
                    candidates.append(matches[0])
        unique_candidates = sorted(set(candidates))
        if unique_candidates:
            record["inspect_handle_id"] = unique_candidates[0]
        if len(unique_candidates) > 1:
            record["inspect_handle_ids"] = unique_candidates
        bound.append(record)
    return bound


def _reference_rendered_source_bodies(
    value: Any,
    *,
    source_records: list[dict[str, Any]],
) -> Any:
    """Replace exact duplicate source bodies with content-addressed references."""

    references_by_text: dict[str, list[str]] = {}
    digest_by_text: dict[str, str] = {}
    for record in source_records:
        text = record.get("text")
        if not isinstance(text, str) or not text:
            continue
        record_id = str(record.get("id") or "")
        if record_id:
            references_by_text.setdefault(text, []).append(record_id)
        digest_by_text[text] = str(
            record.get("content_digest") or _text_digest(text)
        )

    def replace(item: Any) -> Any:
        if isinstance(item, str) and item in references_by_text:
            return {
                "source_evidence_refs": list(
                    dict.fromkeys(references_by_text[item])
                ),
                "content_digest": digest_by_text[item],
                "body_rendered_in": "source_evidence",
            }
        if isinstance(item, dict):
            return {key: replace(child) for key, child in item.items()}
        if isinstance(item, list):
            return [replace(child) for child in item]
        if isinstance(item, tuple):
            return [replace(child) for child in item]
        return item

    return replace(value)


def _range_covered_by_sources(
    record: dict[str, Any],
    source_records: list[dict[str, Any]],
) -> bool:
    tree = str(record.get("tree") or "")
    path = normalize_relpath(str(record.get("path") or ""))
    start = int(record.get("line_start") or 0)
    end = int(record.get("line_end") or start)
    return bool(
        path
        and start > 0
        and any(
            source.get("tree") == tree
            and source.get("file") == path
            and int(source.get("line_start") or 0) > 0
            and int(source.get("line_start") or 0) <= start
            and int(source.get("line_end") or 0) >= end
            for source in source_records
        )
    )


def compact_inspectable_ranges(
    records: list[dict[str, Any]],
    *,
    source_records: list[dict[str, Any]],
) -> list[list[Any]]:
    unique: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, dict):
            continue
        tree = str(raw.get("tree") or "")
        path = normalize_relpath(str(raw.get("path") or ""))
        start = int(raw.get("line_start") or 0)
        end = int(raw.get("line_end") or start)
        if not tree or not path or start <= 0 or end < start:
            continue
        record = {
            "tree": tree,
            "path": path,
            "line_start": start,
            "line_end": end,
        }
        if _range_covered_by_sources(record, source_records):
            continue
        unique.setdefault((tree, path, start, end), record)
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item["tree"],
            item["path"],
            item["line_start"],
            -item["line_end"],
        ),
    )
    retained: list[dict[str, Any]] = []
    for record in ordered:
        if any(
            parent["tree"] == record["tree"]
            and parent["path"] == record["path"]
            and parent["line_start"] <= record["line_start"]
            and parent["line_end"] >= record["line_end"]
            for parent in retained
        ):
            continue
        retained.append(record)
    return [
        [
            record["tree"],
            record["path"],
            record["line_start"],
            record["line_end"],
        ]
        for record in retained
    ]


def compact_inspectable_entities(
    records: list[dict[str, Any]],
) -> list[list[Any]]:
    unique: dict[str, list[Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("id"):
            continue
        entity_id = str(record["id"])
        unique.setdefault(
            entity_id,
            [
                entity_id,
                record.get("tree"),
                record.get("path"),
                record.get("kind"),
                record.get("symbol"),
                record.get("contract_key"),
                record.get("is_definition"),
            ],
        )
    return [unique[key] for key in sorted(unique)]


def _exact_navigation_entity_ids(
    control_state: dict[str, Any],
    causal_graph: dict[str, Any],
) -> set[str]:
    entity_ids: set[str] = set()
    edit_scope = (
        control_state.get("edit_scope")
        if isinstance(control_state.get("edit_scope"), dict)
        else {}
    )
    for unit in edit_scope.get("graph_units") or []:
        if not isinstance(unit, dict):
            continue
        root_node_id = str(unit.get("root_node_id") or "")
        if root_node_id:
            entity_ids.add(root_node_id)
        for field in (
            "root_alternative_node_ids",
            "materializable_entity_ids",
            "semantic_consumer_entity_ids",
        ):
            entity_ids.update(
                str(item)
                for item in unit.get(field) or []
                if str(item)
            )
    for unit in causal_graph.get("units") or []:
        if not isinstance(unit, dict):
            continue
        entity_ids.update(
            str(item)
            for item in unit.get("root_node_ids") or []
            if str(item)
        )
        entity_ids.update(
            str(node.get("id") or "")
            for node in unit.get("nodes") or []
            if isinstance(node, dict) and str(node.get("id") or "")
        )
    return entity_ids


def _exact_navigation_files(
    control_state: dict[str, Any],
    causal_graph: dict[str, Any],
    source_records: list[dict[str, Any]],
) -> set[str]:
    paths = {
        normalize_relpath(str(record.get("file") or ""))
        for record in source_records
        if normalize_relpath(str(record.get("file") or ""))
    }
    edit_scope = (
        control_state.get("edit_scope")
        if isinstance(control_state.get("edit_scope"), dict)
        else {}
    )
    for field in (
        "editable_files",
        "escalation_files",
        "read_only_evidence_files",
    ):
        paths.update(
            normalize_relpath(str(item))
            for item in edit_scope.get(field) or []
            if normalize_relpath(str(item))
        )
    for unit in causal_graph.get("units") or []:
        if not isinstance(unit, dict):
            continue
        paths.update(
            normalize_relpath(str(node.get("file") or ""))
            for node in unit.get("nodes") or []
            if isinstance(node, dict)
            and normalize_relpath(str(node.get("file") or ""))
        )
    return paths


def _missing_header_consumer_entity_ids(
    obligations: dict[str, Any],
) -> set[str]:
    """Return consumers that remain navigable behind an exact include locus."""

    deferred: set[str] = set()
    for unit in obligations.get("units") or []:
        if not isinstance(unit, dict):
            continue
        owners = [
            owner
            for owner in unit.get("repair_owners") or []
            if isinstance(owner, dict)
        ]
        absent_donor_headers = {
            normalize_relpath(str(owner.get("owner_file") or ""))
            for owner in owners
            if str(owner.get("owner_role") or "") == "root_owner"
            and str(owner.get("owner_entity_kind") or "").lower() == "file"
            and str(owner.get("owner_entity_tree") or "").startswith(
                "donor"
            )
            and owner.get("old_exists") is False
        }
        if not absent_donor_headers:
            continue
        for owner in owners:
            if (
                str(owner.get("owner_role") or "")
                != "include_relation_owner"
                or owner.get("allowed_as_patch_target") is not True
                or normalize_relpath(
                    str(owner.get("provider_header_file") or "")
                )
                not in absent_donor_headers
            ):
                continue
            deferred.update(
                str(entity_id)
                for entity_id in owner.get("consumer_owner_node_ids") or []
                if str(entity_id)
            )
    return deferred


def _compact_source_alias_tables(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for raw in records:
        record = dict(raw)
        aliases = record.get("source_aliases")
        if isinstance(aliases, list):
            record["source_aliases"] = compact_prompt_records(aliases)
        compacted.append(record)
    return compacted


def _compact_obligation_record_tables(
    obligations: dict[str, Any],
) -> dict[str, Any]:
    compacted = dict(obligations)
    units: list[dict[str, Any]] = []
    for raw in obligations.get("units") or []:
        if not isinstance(raw, dict):
            continue
        unit = dict(raw)
        for field in ("repair_owners", "patch_targets"):
            records = unit.get(field)
            if isinstance(records, list):
                unit[field] = compact_prompt_records(records)
        localization = unit.get("localization")
        if isinstance(localization, dict):
            localization = dict(localization)
            targets = localization.get("targets")
            if isinstance(targets, list):
                localization["targets"] = compact_prompt_records(targets)
            unit["localization"] = localization
        units.append(unit)
    compacted["units"] = units
    return compacted


def assemble_prompt_evidence_view(
    *,
    control_state: dict[str, Any],
    diagnostics: dict[str, Any],
    causal_graph: dict[str, Any],
    obligations: dict[str, Any],
    source_records: list[dict[str, Any]],
    latest_failure_delta: dict[str, Any] | None,
    inspectable_ranges: list[dict[str, Any]],
    inspectable_entities: list[dict[str, Any]],
    optional_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    deduplicated_sources = _bind_source_inspect_handles(
        deduplicate_source_evidence(source_records),
        inspectable_entities,
    )
    root_entity_ids = _exact_navigation_entity_ids(
        control_state,
        causal_graph,
    )
    deferred_header_consumer_ids = _missing_header_consumer_entity_ids(
        obligations
    )

    def source_entity_ids(record: dict[str, Any]) -> set[str]:
        return set(_source_record_entity_ids(record))

    def render_source(record: dict[str, Any]) -> bool:
        entity_ids = source_entity_ids(record)
        roles = set(record.get("roles") or [])
        inspect_handles = _source_record_inspect_handle_ids(record)
        deferred_header_consumer = bool(
            entity_ids
            and entity_ids <= deferred_header_consumer_ids
            and inspect_handles
        )
        return bool(
            not entity_ids
            or entity_ids & root_entity_ids
            or (
                "old_edit_source" in roles
                and not deferred_header_consumer
            )
            or not inspect_handles
            or any(str(role).startswith("dynamic_") for role in roles)
        )

    rendered_sources = [
        record for record in deduplicated_sources if render_source(record)
    ]
    deferred_header_source_count = sum(
        1
        for record in deduplicated_sources
        if not render_source(record)
        and source_entity_ids(record)
        <= deferred_header_consumer_ids
    )
    rendered_obligations = _compact_obligation_record_tables(
        _reference_rendered_source_bodies(
            obligations,
            source_records=rendered_sources,
        )
    )
    prompt_sources = _compact_source_alias_tables(rendered_sources)
    required_payload = {
        "control_state": control_state,
        "diagnostics": diagnostics,
        "causal_graph": causal_graph,
        "obligations": rendered_obligations,
        "source_evidence": prompt_sources,
        "latest_failure_delta": latest_failure_delta or {},
    }
    required_chars = _json_size(required_payload)
    all_compact_ranges = compact_inspectable_ranges(
        inspectable_ranges,
        source_records=rendered_sources,
    )
    exact_files = _exact_navigation_files(
        control_state,
        causal_graph,
        rendered_sources,
    )
    ranges = [
        record
        for record in all_compact_ranges
        if normalize_relpath(str(record[1] or "")) in exact_files
    ]
    all_compact_entities = compact_inspectable_entities(
        inspectable_entities
    )
    exact_entity_ids = _exact_navigation_entity_ids(
        control_state,
        causal_graph,
    )
    exact_entity_ids.update(
        handle_id
        for record in deduplicated_sources
        for handle_id in _source_record_inspect_handle_ids(record)
    )
    entities = [
        record
        for record in all_compact_entities
        if str(record[0] or "") in exact_entity_ids
    ]
    selected_optional_records = list(optional_records or [])
    metrics = {
        "required_serialized_chars": required_chars,
        "source_record_input_count": len(source_records),
        "source_record_deduplicated_count": len(deduplicated_sources),
        "source_record_rendered_count": len(rendered_sources),
        "source_record_handle_only_count": (
            len(deduplicated_sources) - len(rendered_sources)
        ),
        "missing_header_consumer_handle_only_count": (
            deferred_header_source_count
        ),
        "obligation_unit_count": len(
            rendered_obligations.get("units") or []
        ),
        "inspectable_range_rendered_count": len(ranges),
        "inspectable_range_input_count": len(all_compact_ranges),
        "inspectable_entity_rendered_count": len(entities),
        "inspectable_entity_input_count": len(all_compact_entities),
        "navigation_entity_anchor_count": len(exact_entity_ids),
        "navigation_file_anchor_count": len(exact_files),
    }
    return {
        "schema": PROMPT_EVIDENCE_SCHEMA,
        "evidence_policy": {
            "required_evidence": "lossless",
            "source_body_deduplication": "content_addressed_exact_body",
            "source_navigation": (
                "root_and_writable_owner_bodies_plus_graph_authorized_"
                "entity_handles"
            ),
            "missing_header_consumer_navigation": (
                "exact_include_locus_inline_with_alternative_consumer_"
                "entity_handles"
            ),
            "repeated_record_transport": (
                "lossless_column_table_when_at_least_ten_percent_smaller"
            ),
            "optional_evidence": "complete_current_unit_records",
            "navigation_evidence": "exact_root_and_contract_handles",
        },
        "control_state": control_state,
        "diagnostics": diagnostics,
        "causal_graph": causal_graph,
        "obligations": rendered_obligations,
        "source_evidence": prompt_sources,
        "latest_failure_delta": latest_failure_delta or {},
        "optional_evidence": selected_optional_records,
        "inspectable_ranges": ranges,
        "inspectable_entities": entities,
        "metrics": metrics,
    }
