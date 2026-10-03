from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from driver_backport.core.common import (
    ordered_unique_strings as _ordered_strings,
)
from driver_backport.planning.apply_protocol import (
    extract_json_object as extract_protocol_json_object,
)
from driver_backport.planning.prompt_evidence import (
    render_prompt_payload,
)


SEMANTIC_REVIEW_RESPONSE_SCHEMA = "semantic_review_response_v1"
SEMANTIC_REPAIR_RESPONSE_SCHEMA = "semantic_repair_response_v1"

_REPAIR_ACTION_LIST_FIELDS = {
    "source_edits",
    "inspect_handle_ids",
    "owner_keys",
    "dependency_edge_ids",
}
_REPAIR_ACTION_STRING_FIELDS = {
    "dependency_direction",
    "materialization_handle_id",
}
_REPAIR_ACTION_FIELDS = (
    _REPAIR_ACTION_LIST_FIELDS | _REPAIR_ACTION_STRING_FIELDS
)
_REPAIR_ENVELOPE_FIELDS = (
    "action",
    "rationale",
    "gap_ids",
    "source_edits",
    "inspect_handle_ids",
    "owner_keys",
    "dependency_direction",
    "dependency_edge_ids",
    "materialization_handle_id",
)
_REPAIR_ACTION_CONTRACTS = {
    "replace_source": {
        "required_nonempty": {"source_edits"},
        "optional_nonempty": set(),
    },
    "inspect": {
        "required_nonempty": {"inspect_handle_ids"},
        "optional_nonempty": set(),
    },
    "expand_dependencies": {
        "required_nonempty": {
            "owner_keys",
            "dependency_direction",
        },
        "optional_nonempty": {"dependency_edge_ids"},
    },
    "materialize_entity_from_donor": {
        "required_nonempty": {"materialization_handle_id"},
        "optional_nonempty": set(),
    },
    "blocked": {
        "required_nonempty": set(),
        "optional_nonempty": set(),
    },
}

class SemanticRepairProtocolError(ValueError):
    """A rejected semantic action envelope."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object without repairing or guessing model output."""

    value = extract_protocol_json_object(text)
    if value is not None:
        return value
    raise ValueError("model response does not contain a JSON object")


def _action_field_is_nonempty(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return value is not None


def _action_contract(action: str) -> dict[str, Any]:
    contract = _REPAIR_ACTION_CONTRACTS.get(action)
    if contract is None:
        raise SemanticRepairProtocolError(
            f"unsupported semantic repair action: {action or '<empty>'}"
        )
    required = set(contract["required_nonempty"])
    optional = set(contract["optional_nonempty"])
    allowed = required | optional
    return {
        "required_nonempty": sorted(required),
        "optional_nonempty": sorted(optional),
        "must_be_empty": sorted(_REPAIR_ACTION_FIELDS - allowed),
    }


def _action_contract_prompt() -> dict[str, Any]:
    return {
        action: _action_contract(action)
        for action in _REPAIR_ACTION_CONTRACTS
    }


def semantic_review_json_schema() -> dict[str, Any]:
    """Return the provider-facing strict schema for review decisions."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "gap_id",
                        "disposition",
                        "rationale",
                        "closure_basis",
                        "covered_dimensions",
                        "evidence_ids",
                    ],
                    "properties": {
                        "gap_id": {"type": "string"},
                        "disposition": {
                            "type": "string",
                            "enum": [
                                "no_change_required",
                                "change_required",
                            ],
                        },
                        "rationale": {"type": "string"},
                        "closure_basis": {
                            "type": "string",
                            "enum": [
                                "",
                                "exact_current_tree_satisfied",
                                "target_contract_not_required",
                            ],
                        },
                        "covered_dimensions": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            }
        },
    }


def semantic_repair_json_schema() -> dict[str, Any]:
    """Return one fixed action envelope for strict structured output."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "rationale",
            "gap_ids",
            "source_edits",
            "inspect_handle_ids",
            "owner_keys",
            "dependency_direction",
            "dependency_edge_ids",
            "materialization_handle_id",
        ],
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "replace_source",
                    "inspect",
                    "expand_dependencies",
                    "materialize_entity_from_donor",
                    "blocked",
                ],
            },
            "rationale": {"type": "string"},
            "gap_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "source_edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "edit_handle_id",
                        "expected_source_sha256",
                        "replacement_source",
                    ],
                    "properties": {
                        "edit_handle_id": {"type": "string"},
                        "expected_source_sha256": {"type": "string"},
                        "replacement_source": {"type": "string"},
                    },
                },
            },
            "inspect_handle_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "owner_keys": {
                "type": "array",
                "items": {"type": "string"},
            },
            "dependency_direction": {
                "type": "string",
                "enum": ["", "outgoing", "incoming", "both"],
            },
            "dependency_edge_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "materialization_handle_id": {"type": "string"},
        },
    }


def _strict_fields(
    value: dict[str, Any],
    *,
    allowed: set[str],
    label: str,
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(
            f"{label} contains unexpected fields: "
            + ", ".join(unexpected)
        )


def _review_gap_payload(gap: dict[str, Any]) -> dict[str, Any]:
    initial = (
        gap.get("contract_evidence")
        if isinstance(gap.get("contract_evidence"), dict)
        else {}
    )
    initial_locator = {
        key: deepcopy(initial.get(key))
        for key in (
            "id",
            "contract_identity",
            "kind",
            "symbol",
            "dependency_scope",
            "semantic_requirement",
        )
        if initial.get(key) not in (None, "", [], {})
    }
    payload = {
        key: deepcopy(gap.get(key))
        for key in (
            "id",
            "contract_identity",
            "status",
            "review_failures",
            "authoritative_noop_evidence_refs",
            "contract_evidence_ref",
            "compile_valid_evaluation_ref",
            "compile_valid_evaluation",
        )
        if gap.get(key) not in (None, "", [], {})
    } | {
        "initial_contract_locator": initial_locator,
    }
    required_evidence_ids = _ordered_strings(
        [
            gap.get("contract_evidence_ref"),
            gap.get("compile_valid_evaluation_ref"),
        ]
    )
    if required_evidence_ids:
        payload["required_evidence_ids"] = required_evidence_ids
    return payload


def _prompt_resolution(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: deepcopy(value.get(key))
        for key in (
            "status",
            "parse_error",
            "resolution_basis",
        )
        if value.get(key) not in (None, "", [], {})
    }


def _prompt_demand_reference(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: deepcopy(value.get(key))
        for key in (
            "id",
            "reference_id",
            "source_file",
            "line",
            "column",
            "extent",
            "symbol",
            "entity_kind",
            "use_kind",
            "cursor_kind",
            "dependency_scope",
            "contract_key",
        )
        if value.get(key) not in (None, "", [], {})
    }


def _prompt_owner(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = {
        key: deepcopy(value.get(key))
        for key in (
            "owner_key",
            "role",
            "file",
            "contract_key",
            "symbol",
            "entity_kind",
            "target_presence",
        )
        if value.get(key) not in (None, "", [], {})
    }
    demand = _prompt_demand_reference(value.get("demand_reference"))
    if demand:
        result["demand_reference"] = demand
    demand_relation = value.get("demand_reference_relation")
    if isinstance(demand_relation, dict) and demand_relation:
        result["demand_reference_relation"] = deepcopy(demand_relation)
    current = _prompt_resolution(value.get("current_resolution"))
    donor = _prompt_resolution(value.get("donor_resolution"))
    if current:
        result["current_resolution"] = current
    if donor:
        result["donor_resolution"] = donor
    return result


_PROMPT_EVIDENCE_ID_FIELDS = frozenset(
    {
        "id",
        "entity_id",
        "reference_id",
        "owner_key",
        "handle_id",
        "source_record_id",
        "hunk_id",
        "gap_id",
        "contract_key",
        "contract_evidence_ref",
        "compile_valid_evaluation_ref",
        "evaluation_ref",
        "record_ref",
    }
)
_PROMPT_EVIDENCE_ID_LIST_FIELDS = frozenset(
    {
        "entity_ids",
        "owner_keys",
        "gap_ids",
        "edit_handle_ids",
        "materialization_handle_ids",
        "read_handle_ids",
        "donor_source_record_ids",
        "exact_node_refs",
        "evidence_ids",
        "required_evidence_ids",
        "evidence_refs",
    }
)


def _semantic_review_payload(request: dict[str, Any]) -> dict[str, Any]:
    """Build the single payload used for review transport and proof."""

    expected = _ordered_strings(
        request.get("expected_gap_ids") or []
    )
    context = (
        request.get("context")
        if isinstance(request.get("context"), dict)
        else {}
    )
    gaps = [
        dict(item)
        for item in context.get("gaps") or []
        if isinstance(item, dict)
    ]
    observed = _ordered_strings(
        item.get("id") for item in gaps
    )
    if not expected or set(observed) != set(expected):
        raise ValueError(
            "semantic review context must contain exactly the expected gaps"
        )
    return {
        "task": (
            "Decide whether each exact donor/current contract difference "
            "still requires a source change in the compile-valid target."
        ),
        "round": int(request.get("round") or 0),
        "work_unit_id": str(request.get("work_unit_id") or ""),
        "expected_gap_ids": expected,
        "rules": [
            (
                "An observed version difference is not automatically a "
                "required migration."
            ),
            (
                "compile_valid_evaluation is the authoritative structural "
                "observation of the current target and donor trees. An "
                "unresolved relation means the exact donor graph shape is "
                "absent; it does not by itself prove that target behavior is "
                "missing or inadequate. "
                "initial_contract_locator identifies the original contract "
                "only and must not override current observations."
            ),
            (
                "Choose no_change_required only when the supplied current-"
                "tree evidence shows that the target contract and its "
                "relevant integration behavior are already adequate."
            ),
            (
                "For no_change_required, closure_basis must distinguish an "
                "exactly satisfied current graph from a target-specific "
                "semantic judgment, and covered_dimensions must list every "
                "currently unresolved or unavailable dimension."
            ),
            (
                "evidence_ids must cite only exact IDs or record references "
                "present in this request. Copy every ID from the gap's "
                "required_evidence_ids into evidence_ids. A missing relation cannot "
                "be described as exact_current_tree_satisfied."
            ),
            (
                "Choose change_required when the supplied evidence shows a "
                "missing or incompatible contract/integration behavior that "
                "the migrated driver needs. A donor-only helper, provider, "
                "API spelling, or direct reference is not sufficient: compare "
                "the target implementation and choose target_contract_not_"
                "required when target-version code already realizes the "
                "required behavior through a different valid sequence."
            ),
            (
                "The relation target_alternative_review_required means only "
                "that the donor's direct reference shape is absent while the "
                "requirement permits a target-specific representation. It is "
                "not evidence of missing behavior. Inspect the exact old and "
                "donor source_records for that gap; choose change_required "
                "only when those sources demonstrate that the demanded "
                "behavior itself is absent or incompatible."
            ),
            (
                "For change_required on target_alternative_review_required, "
                "evidence_ids must include the exact contract and evaluation "
                "references plus at least one old-tree and one donor-tree "
                "source_record id for that gap."
            ),
            (
                "Do not propose a patch, request more files, classify the "
                "error, or invent evidence in this decision."
            ),
            (
                "Return one decision for every expected gap and no decision "
                "for any other gap."
            ),
        ],
        "gaps": [_review_gap_payload(gap) for gap in gaps],
        "decision_evidence": deepcopy(
            request.get("decision_evidence") or {}
        ),
        "response_schema": {
            "decisions": [
                {
                    "gap_id": "<exact expected gap id>",
                    "disposition": (
                        "no_change_required | change_required"
                    ),
                    "rationale": (
                        "<specific rationale grounded in supplied evidence>"
                    ),
                    "closure_basis": (
                        "<empty for change_required; otherwise "
                        "exact_current_tree_satisfied or "
                        "target_contract_not_required>"
                    ),
                    "covered_dimensions": [
                        "<dimensions justified by a no-change decision>"
                    ],
                    "evidence_ids": [
                        "<exact supplied record/entity/edge ids>"
                    ],
                }
            ]
        },
    }


def semantic_review_prompt_evidence_ids(
    request: dict[str, Any],
) -> list[str]:
    """List domain evidence identifiers present in the review payload."""

    prompt_evidence = _semantic_review_payload(request)
    identifiers: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                collect(item)
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key in _PROMPT_EVIDENCE_ID_FIELDS:
                normalized = str(item or "")
                if normalized:
                    identifiers.add(normalized)
            elif key in _PROMPT_EVIDENCE_ID_LIST_FIELDS:
                identifiers.update(_ordered_strings(item or []))
            collect(item)

    identifiers.update(
        _ordered_strings(prompt_evidence.get("expected_gap_ids") or [])
    )
    collect(prompt_evidence.get("gaps") or [])
    collect(prompt_evidence.get("decision_evidence") or {})
    return sorted(identifiers)


def project_semantic_tool_evidence_for_prompt(
    evidence: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Remove duplicate catalog transport while preserving exact evidence."""

    if evidence.get("schema") != "semantic_closure_tool_evidence_v1":
        return deepcopy(evidence)
    if mode not in {"review", "repair"}:
        raise ValueError(f"unsupported semantic evidence prompt mode: {mode}")
    projected: dict[str, Any] = {
        key: deepcopy(evidence.get(key))
        for key in (
            "schema",
            "work_unit_id",
            "hunks",
            "executable_materializations",
            "unavailable_materializations",
            "unavailable_insertion_anchors",
            "mutation_targets",
            "source_records",
        )
        if evidence.get(key) not in (None, "", [], {})
    }
    owners = [
        _prompt_owner(item)
        for item in evidence.get("owners") or []
        if isinstance(item, dict)
    ]
    if owners:
        projected["owners"] = owners
    if mode == "review":
        # Every dependency edge in the base catalog is already represented by
        # its owner's exact demand_reference and node refs.
        return projected

    read_handle_fields = {
        "id",
        "kind",
        "tree",
        "file",
        "start_line",
        "end_line",
        "entity_id",
        "entity_ids",
        "authorities",
        "owner_keys",
        "gap_ids",
        "source_record_id",
        "hunk_id",
    }
    read_handles = [
        {
            key: deepcopy(value)
            for key, value in item.items()
            if key in read_handle_fields
            and value not in (None, "", [], {})
        }
        for item in evidence.get("read_handles") or []
        if isinstance(item, dict)
    ]
    if read_handles:
        projected["read_handles"] = read_handles

    edit_handle_fields = {
        "id",
        "file",
        "start_line",
        "end_line",
        "source_sha256",
        "edit_mode",
        "authorities",
        "owner_keys",
        "gap_ids",
        "repair_dimensions",
        "read_handle_ids",
        "source_record_id",
        "donor_source_record_ids",
    }
    edit_handles = [
        {
            key: deepcopy(value)
            for key, value in item.items()
            if key in edit_handle_fields
            and value not in (None, "", [], {})
        }
        for item in evidence.get("edit_handles") or []
        if isinstance(item, dict)
    ]
    if edit_handles:
        projected["edit_handles"] = edit_handles
    # Base dependency edges duplicate owner demand provenance. The model uses
    # an owner_key to list current direct edges through the tool, then selects
    # returned edge ids in a later action. Full graph provenance remains in
    # the durable evidence artifact.
    return projected


def build_semantic_review_prompt(
    request: dict[str, Any],
) -> str:
    """Build a compact decision-only prompt for exact contract gaps."""

    payload = _semantic_review_payload(request)
    rendered_payload, transport_instruction, _ = render_prompt_payload(
        payload,
        sort_keys=True,
    )
    return (
        "You are reviewing exact Linux driver migration contract evidence.\n"
        "Return only one JSON object matching response_schema.\n"
        + (
            "Prompt transport encoding: " + transport_instruction + "\n"
            if transport_instruction
            else ""
        )
        + rendered_payload
        + "\n"
    )


def parse_semantic_review_response(
    text: str,
    *,
    expected_gap_ids: Iterable[object],
) -> dict[str, Any]:
    payload = extract_json_object(text)
    _strict_fields(
        payload,
        allowed={"decisions"},
        label="semantic review response",
    )
    expected = _ordered_strings(expected_gap_ids)
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError(
            "semantic review response.decisions must be a list"
        )
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_decisions):
        if not isinstance(raw, dict):
            raise ValueError(
                f"semantic review decisions[{index}] must be an object"
            )
        _strict_fields(
            raw,
            allowed={
                "gap_id",
                "disposition",
                "rationale",
                "closure_basis",
                "covered_dimensions",
                "evidence_ids",
            },
            label=f"semantic review decisions[{index}]",
        )
        gap_id = str(raw.get("gap_id") or "").strip()
        disposition = str(
            raw.get("disposition") or ""
        ).strip()
        rationale = str(raw.get("rationale") or "").strip()
        closure_basis = str(raw.get("closure_basis") or "").strip()
        if not isinstance(raw.get("covered_dimensions"), list):
            raise ValueError(
                "semantic review covered_dimensions must be a list: "
                f"{gap_id}"
            )
        if not isinstance(raw.get("evidence_ids"), list):
            raise ValueError(
                f"semantic review evidence_ids must be a list: {gap_id}"
            )
        covered_dimensions = _ordered_strings(
            raw.get("covered_dimensions") or []
        )
        evidence_ids = _ordered_strings(raw.get("evidence_ids") or [])
        if not gap_id or gap_id in seen:
            raise ValueError(
                f"semantic review has an invalid duplicate gap: {gap_id}"
            )
        if disposition not in {
            "no_change_required",
            "change_required",
        }:
            raise ValueError(
                f"unsupported semantic disposition: {disposition}"
            )
        if not rationale:
            raise ValueError(
                f"semantic review decision has no rationale: {gap_id}"
            )
        if disposition == "change_required":
            if closure_basis or covered_dimensions:
                raise ValueError(
                    "change-required review decision contains a no-change "
                    f"proof claim: {gap_id}"
                )
        elif (
            closure_basis
            not in {
                "exact_current_tree_satisfied",
                "target_contract_not_required",
            }
            or not covered_dimensions
            or not evidence_ids
        ):
            raise ValueError(
                f"no-change review decision has no structured proof claim: "
                f"{gap_id}"
            )
        seen.add(gap_id)
        decisions.append(
            {
                "gap_id": gap_id,
                "disposition": disposition,
                "rationale": rationale,
                "closure_basis": closure_basis,
                "covered_dimensions": covered_dimensions,
                "evidence_ids": evidence_ids,
            }
        )
    if seen != set(expected):
        raise ValueError(
            "semantic review decision ids do not exactly match the request: "
            f"missing={sorted(set(expected) - seen)} "
            f"unexpected={sorted(seen - set(expected))}"
        )
    return {
        "schema": SEMANTIC_REVIEW_RESPONSE_SCHEMA,
        "decisions": decisions,
    }


def _repair_contract(request: dict[str, Any]) -> dict[str, Any]:
    kind = str(request.get("kind") or "")
    if kind != "semantic_repair":
        raise ValueError(f"unsupported semantic repair kind: {kind}")
    context = (
        request.get("context")
        if isinstance(request.get("context"), dict)
        else {}
    )
    work_unit = (
        context.get("work_unit")
        if isinstance(context.get("work_unit"), dict)
        else {}
    )
    gap_ids = _ordered_strings(work_unit.get("gap_ids") or [])
    if not gap_ids:
        raise ValueError("semantic repair request has no exact gaps")
    return {
        "kind": kind,
        "gap_ids": gap_ids,
        "work_unit_id": str(
            request.get("work_unit_id")
            or work_unit.get("id")
            or work_unit.get("work_unit_id")
            or ""
        ),
    }


def _repair_prompt_context(
    context: dict[str, Any],
) -> dict[str, Any]:
    """Remove fields already represented by the exact evidence catalog."""

    work_unit = (
        context.get("work_unit")
        if isinstance(context.get("work_unit"), dict)
        else {}
    )
    compact_work_unit = {
        key: deepcopy(work_unit.get(key))
        for key in (
            "id",
            "work_unit_id",
            "unit_kind",
            "gap_ids",
            "gap_statuses",
            "source_work_unit_ids",
            "allowed_fileset",
            "depends_on_work_unit_ids",
            "owner_mapping_complete",
        )
        if work_unit.get(key) not in (None, "", [], {})
    }
    gaps = []
    for raw in context.get("gaps") or []:
        if not isinstance(raw, dict):
            continue
        gaps.append(
            {
                key: deepcopy(raw.get(key))
                for key in (
                    "id",
                    "contract_identity",
                    "status",
                    "decision",
                    "review_failures",
                    "compile_valid_evaluation_ref",
                )
                if raw.get(key) not in (None, "", [], {})
            }
            | {
                "requirement": {
                    key: deepcopy(
                        (raw.get("contract_evidence") or {}).get(key)
                    )
                    for key in (
                        "kind",
                        "symbol",
                        "dependency_scope",
                        "semantic_requirement",
                    )
                    if isinstance(raw.get("contract_evidence"), dict)
                    and (raw.get("contract_evidence") or {}).get(key)
                    not in (None, "", [], {})
                },
                "current_evaluation": {
                    key: deepcopy(
                        (raw.get("compile_valid_evaluation") or {}).get(key)
                    )
                    for key in (
                        "status",
                        "required_dimensions",
                        "relations",
                        "unresolved_dimensions",
                        "unavailable_dimensions",
                    )
                    if isinstance(
                        raw.get("compile_valid_evaluation"),
                        dict,
                    )
                    and (raw.get("compile_valid_evaluation") or {}).get(key)
                    not in (None, "", [], {})
                },
            }
        )
    return {
        "schema": context.get("schema"),
        "work_unit": compact_work_unit,
        "gaps": gaps,
    }


def _prompt_evidence_record(
    value: object,
    *,
    available_source_sha256: set[str] | None = None,
) -> object:
    """Keep decision evidence while removing transport and patch duplication."""

    if isinstance(value, list):
        return [
            _prompt_evidence_record(
                item,
                available_source_sha256=available_source_sha256,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return deepcopy(value)
    if value.get("schema") == "apply_failure_v1":
        # This record is already the compact, typed failure contract emitted
        # by the shared apply protocol.  Keeping it intact is necessary for a
        # later action to distinguish representation, grounding, policy, and
        # transaction failures without replaying the failed patch blindly.
        return deepcopy(value)
    source_sha256 = str(value.get("source_sha256") or "")
    source_is_replayed = bool(
        source_sha256
        and available_source_sha256
        and source_sha256 in available_source_sha256
        and "source" in value
    )
    retained = {
        "schema",
        "action",
        "status",
        "reason",
        "error",
        "apply_ref",
        "apply_result",
        "apply_failure",
        "canonical_apply_status",
        "mechanical_repair",
        "patch_rebase",
        "patch_signature",
        "validation_failure",
        "gap_ids",
        "inspect_handle_ids",
        "owner_keys",
        "direction",
        "materialization_handle_id",
        "requests",
        "evidence",
        "dependency_edge_ids",
        "dependency_edges",
        "denied_changed_ranges",
        "returncode",
        "stdout",
        "stderr",
        "stage",
        "recoverability",
        "finding_kind",
        "deterministic_attempts",
        "details",
        "repaired_hunk_count",
        "results",
        "kind",
        "id",
        "handle_id",
        "tree",
        "file",
        "entity_id",
        "descriptor",
        "target",
        "reference",
        "selected_owner_key",
        "symbol",
        "contract_key",
        "source",
        "source_sha256",
        "start_line",
        "end_line",
        "hunk_id",
        "record",
        "selected_owner_keys",
        "allowed_fileset",
        "touched_files",
        "remaining_mutation_frontier",
        "executable_mutation_frontier_empty",
        "open_dimensions",
        "edit_handle_ids",
        "materialization_handle_ids",
    }
    projected = {
        str(key): _prompt_evidence_record(
            item,
            available_source_sha256=available_source_sha256,
        )
        for key, item in value.items()
        if str(key) in retained
        and not (source_is_replayed and str(key) == "source")
    }
    if source_is_replayed:
        projected["source_reused_from_current_evidence"] = source_sha256
    return projected


def _compact_previous_outcomes(
    values: Iterable[object],
    *,
    available_source_sha256: Iterable[object] = (),
) -> dict[str, Any]:
    """Project prior actions; shared prompt transport deduplicates records."""

    actions: list[dict[str, Any]] = []
    available_sources = set(_ordered_strings(available_source_sha256))
    for raw in values:
        if not isinstance(raw, dict):
            continue
        actions.append(
            {
                key: deepcopy(raw.get(key))
                for key in (
                    "round",
                    "step",
                    "evidence_action",
                    "budget_kind",
                    "action_signature",
                    "outcome",
                    "duplicate",
                )
                if raw.get(key) not in (None, "", [], {})
            }
            | {
                "evidence": [
                    _prompt_evidence_record(
                        evidence.get("record") or {},
                        available_source_sha256=available_sources,
                    )
                    for evidence in raw.get("evidence") or []
                    if isinstance(evidence, dict)
                ]
            }
        )
    return {"actions": actions}


def build_semantic_repair_prompt(
    request: dict[str, Any],
    *,
    evidence: dict[str, Any],
) -> str:
    """Build a repair prompt from exact graph-selected evidence only."""

    contract = _repair_contract(request)
    instructions = {
        "task": (
            "Repair one exact semantic contract unit."
        ),
        "rules": [
            (
                "Use only the supplied exact evidence and advertised handles."
            ),
            (
                "evidence.mutation_targets is the controller-derived mapping "
                "from each unresolved contract dimension to the exact owners "
                "and write handles that can satisfy it. Source records without "
                "a listed edit handle are comparison evidence, not mutation "
                "targets."
            ),
            (
                "Never write patch syntax. For a source change, return the "
                "complete replacement_source for one or more exact edit "
                "handles. The controller verifies each source digest and "
                "constructs the atomic unified diff."
            ),
            (
                "Read-only inspection and write authorization are distinct. "
                "Inspect only advertised read handles; edit only advertised "
                "edit handles."
            ),
            (
                "An edit handle with edit_mode insert_after_anchor or "
                "insert_before_anchor authorizes only an insertion at that "
                "exact structural boundary. Preserve the current anchor "
                "source byte-for-byte as the replacement prefix or suffix, "
                "respectively; donor_source_record_ids identify the exact "
                "missing contracts associated with that insertion handle."
            ),
            (
                "source_records contain complete exact current/donor source "
                "already available in this request. A handle's "
                "source_record_id points to that record; do not request an "
                "inspection merely to read the same source again."
            ),
            (
                "If evidence is insufficient, request exact read handles or "
                "list direct dependency edges instead of guessing. The first "
                "expand_dependencies action must leave dependency_edge_ids "
                "empty; it returns compact exact edge ids. A later "
                "expand_dependencies action may bind only explicitly selected "
                "returned edge ids."
            ),
            (
                "A previous typed apply_failure is authoritative evidence "
                "about the current-tree transaction. Do not submit the same "
                "source edits again against an unchanged evidence state. "
                "Use its stage, finding_kind, recoverability, diagnostics, "
                "and deterministic attempts to revise the edit or request "
                "the exact missing evidence."
            ),
            (
                "Do not introduce a donor-only symbol into target source "
                "unless the evidence shows that its provider already exists "
                "in the target or mutation_targets contains an executable "
                "materialization that closes the provider contract. A donor "
                "provider in comparison evidence is not target availability."
            ),
            (
                "Use materialize_entity_from_donor only with an advertised "
                "executable materialization handle."
            ),
            (
                "blocked reports that the controller has no executable exact "
                "mutation action. It is invalid while mutation_targets still "
                "advertise an edit or materialization handle; use that handle "
                "or request the exact evidence needed to edit it."
            ),
            (
                "Do not edit Kconfig, Makefile, or Kbuild in this source-"
                "repair phase."
            ),
            (
                "A semantic patch must claim exactly the gaps it repairs "
                "from this work unit."
            ),
            (
                "Return every fixed envelope field, but populate only fields "
                "allowed by the selected action contract. Use [] for an "
                "empty list and the literal JSON string \"\" for an empty "
                "string; never write the word 'empty'."
            ),
        ],
        "response_fields": list(_REPAIR_ENVELOPE_FIELDS),
        "action_contracts": _action_contract_prompt(),
        "source_edit_fields": [
            "edit_handle_id",
            "expected_source_sha256",
            "replacement_source",
        ],
    }
    payload = {
        "round": int(request.get("round") or 0),
        "work_unit_id": contract["work_unit_id"],
        "kind": contract["kind"],
        "gap_ids": contract["gap_ids"],
        "context": _repair_prompt_context(
            request.get("context")
            if isinstance(request.get("context"), dict)
            else {},
        ),
        "evidence": project_semantic_tool_evidence_for_prompt(
            evidence,
            mode="repair",
        ),
        "previous_outcomes": _compact_previous_outcomes(
            request.get("previous_outcomes") or [],
            available_source_sha256=(
                item.get("source_sha256")
                for item in evidence.get("source_records") or []
                if isinstance(item, dict)
            ),
        ),
    }
    rendered_instructions, instruction_transport, _ = (
        render_prompt_payload(instructions)
    )
    rendered_payload, payload_transport, _ = render_prompt_payload(payload)
    transport_instructions = " ".join(
        dict.fromkeys(
            instruction
            for instruction in (
                instruction_transport,
                payload_transport,
            )
            if instruction
        )
    )
    return (
        "You are repairing a Linux driver migration from exact code-graph "
        "evidence. Return only one JSON object matching the fixed envelope.\n"
        + (
            "Prompt transport encoding: "
            + transport_instructions
            + "\n"
            if transport_instructions
            else ""
        )
        + rendered_instructions
        + "\nDYNAMIC_INPUT\n"
        + rendered_payload
        + "\n"
    )


def _parse_gap_ids(
    payload: dict[str, Any],
    *,
    expected_gap_ids: list[str],
) -> list[str]:
    raw_gap_ids = payload.get("gap_ids")
    if not isinstance(raw_gap_ids, list):
        raise SemanticRepairProtocolError("gap_ids must be a list")
    gap_ids = _ordered_strings(raw_gap_ids)
    if not gap_ids or not set(gap_ids) <= set(expected_gap_ids):
        raise SemanticRepairProtocolError(
            "semantic repair action must claim only exact work-unit gaps"
        )
    return gap_ids


def parse_semantic_repair_response(
    text: str,
    *,
    kind: str,
    expected_gap_ids: Iterable[object],
) -> dict[str, Any]:
    if kind != "semantic_repair":
        raise ValueError(f"unsupported semantic repair kind: {kind}")
    payload = extract_json_object(text)
    action = str(payload.get("action") or "").strip()
    expected = _ordered_strings(expected_gap_ids)
    common_fields = set(_REPAIR_ENVELOPE_FIELDS)
    unexpected = sorted(set(payload) - common_fields)
    if unexpected:
        raise SemanticRepairProtocolError(
            "semantic repair action contains unexpected fields: "
            + ", ".join(unexpected)
        )
    missing = sorted(common_fields - set(payload))
    if missing:
        raise SemanticRepairProtocolError(
            "semantic repair action is missing fixed envelope fields: "
            + ", ".join(missing)
        )
    contract = _action_contract(action)
    disallowed_nonempty_fields = [
        field
        for field in contract["must_be_empty"]
        if _action_field_is_nonempty(payload.get(field))
    ]
    if disallowed_nonempty_fields:
        raise SemanticRepairProtocolError(
            f"{action} requires empty fields: "
            + ", ".join(disallowed_nonempty_fields)
        )
    rationale_value = payload.get("rationale")
    if not isinstance(rationale_value, str) or not rationale_value.strip():
        raise SemanticRepairProtocolError(
            "semantic repair action requires a rationale"
        )
    rationale = rationale_value.strip()
    gap_ids = _parse_gap_ids(
        payload,
        expected_gap_ids=expected,
    )
    source_edits = payload.get("source_edits")
    inspect_handles = payload.get("inspect_handle_ids")
    owner_keys = payload.get("owner_keys")
    dependency_direction = payload.get("dependency_direction")
    dependency_edge_ids = payload.get("dependency_edge_ids")
    materialization_handle_id = payload.get("materialization_handle_id")
    invalid_list_fields = [
        field
        for field, value in (
            ("source_edits", source_edits),
            ("inspect_handle_ids", inspect_handles),
            ("owner_keys", owner_keys),
            ("dependency_edge_ids", dependency_edge_ids),
        )
        if not isinstance(value, list)
    ]
    invalid_string_fields = [
        field
        for field, value in (
            ("dependency_direction", dependency_direction),
            ("materialization_handle_id", materialization_handle_id),
        )
        if not isinstance(value, str)
    ]
    invalid_types = invalid_list_fields + invalid_string_fields
    if invalid_types:
        raise SemanticRepairProtocolError(
            "semantic repair action fields have invalid types: "
            + ", ".join(invalid_types)
        )
    missing_action_fields = [
        field
        for field in contract["required_nonempty"]
        if not _action_field_is_nonempty(payload.get(field))
    ]
    if missing_action_fields:
        raise SemanticRepairProtocolError(
            f"{action} requires non-empty fields: "
            + ", ".join(missing_action_fields)
        )

    common_result: dict[str, Any] = {
        "schema": SEMANTIC_REPAIR_RESPONSE_SCHEMA,
        "action": action,
        "rationale": rationale,
    }

    if action == "replace_source":
        normalized_edits: list[dict[str, str]] = []
        for index, raw in enumerate(source_edits):
            if not isinstance(raw, dict):
                raise SemanticRepairProtocolError(
                    f"source_edits[{index}] must be an object"
                )
            required_edit_fields = {
                "edit_handle_id",
                "expected_source_sha256",
                "replacement_source",
            }
            invalid_edit_fields = sorted(set(raw) - required_edit_fields)
            missing_edit_fields = sorted(required_edit_fields - set(raw))
            if invalid_edit_fields or missing_edit_fields:
                raise SemanticRepairProtocolError(
                    f"source_edits[{index}] must contain exactly: "
                    + ", ".join(sorted(required_edit_fields))
                )
            handle_id = str(raw.get("edit_handle_id") or "").strip()
            digest = str(
                raw.get("expected_source_sha256") or ""
            ).strip()
            replacement = raw.get("replacement_source")
            if not handle_id or not digest or not isinstance(
                replacement,
                str,
            ):
                raise SemanticRepairProtocolError(
                    f"source_edits[{index}] has invalid values"
                )
            normalized_edits.append(
                {
                    "edit_handle_id": handle_id,
                    "expected_source_sha256": digest,
                    "replacement_source": replacement,
                }
            )
        return {
            **common_result,
            "gap_ids": gap_ids,
            "source_edits": normalized_edits,
        }

    if action == "inspect":
        normalized = _ordered_strings(inspect_handles)
        if not normalized:
            raise SemanticRepairProtocolError(
                "inspect requires at least one non-empty inspect handle"
            )
        return {
            **common_result,
            "inspect_handle_ids": normalized,
        }

    if action == "expand_dependencies":
        normalized_owner_keys = _ordered_strings(owner_keys)
        normalized_edge_ids = _ordered_strings(dependency_edge_ids)
        if not normalized_owner_keys:
            raise SemanticRepairProtocolError(
                "expand_dependencies requires non-empty owner_keys"
            )
        if dependency_direction not in {
            "outgoing",
            "incoming",
            "both",
        }:
            raise SemanticRepairProtocolError(
                "dependency_direction must be outgoing, incoming, or both"
            )
        return {
            **common_result,
            "owner_keys": normalized_owner_keys,
            "direction": dependency_direction,
            "dependency_edge_ids": normalized_edge_ids,
        }

    if action == "materialize_entity_from_donor":
        return {
            **common_result,
            "materialization_handle_id": materialization_handle_id.strip(),
            "gap_ids": gap_ids,
        }

    if action == "blocked":
        return {
            **common_result,
            "gap_ids": gap_ids,
        }

    raise AssertionError(f"unhandled semantic repair action: {action}")
