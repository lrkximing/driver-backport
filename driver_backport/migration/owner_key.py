#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths,
    ordered_unique_strings,
    stable_id,
)


BUILD_METADATA_NAMES = {"Kconfig", "Makefile", "Kbuild"}

_UNSPECIFIED_MUTATION_DIMENSION = "unspecified_contract_dimension"
_PROVIDER_MUTATION_DIMENSIONS = frozenset(
    {
        "name",
        "signature",
        "implementation",
        "interface",
        "structural_source",
        "export",
        "module_boundary",
    }
)
_DECLARATION_MUTATION_DIMENSIONS = frozenset(
    {
        "name",
        "signature",
        "interface",
        "declaration",
        "export",
        "module_boundary",
    }
)
_REFERENCE_COUPLED_PROVIDER_DIMENSIONS = frozenset(
    {
        "name",
        "signature",
        "integration_reference",
    }
)

_INITIAL_SEMANTIC_OWNER_ROLES = frozenset(
    {
        "repair_locus",
        "demand_owner",
        "contract_entity",
        "contract_declaration",
    }
)


def is_build_path(relative_path: str) -> bool:
    return Path(normalize_relpath(relative_path)).name in BUILD_METADATA_NAMES


def semantic_open_mutation_dimensions(
    gap: dict[str, Any],
) -> list[str]:
    """Return contract dimensions that still require a source decision."""

    evaluation = next(
        (
            gap.get(key)
            for key in (
                "compile_valid_evaluation",
                "compile_valid_handoff_evaluation",
            )
            if isinstance(gap.get(key), dict)
        ),
        {},
    )
    if evaluation:
        return ordered_unique_strings(
            [
                *list(evaluation.get("unresolved_dimensions") or []),
                *list(evaluation.get("unavailable_dimensions") or []),
            ]
        )
    contract_evidence = (
        gap.get("contract_evidence")
        if isinstance(gap.get("contract_evidence"), dict)
        else {}
    )
    requirement = (
        contract_evidence.get("semantic_requirement")
        if isinstance(contract_evidence.get("semantic_requirement"), dict)
        else {}
    )
    required = ordered_unique_strings(
        requirement.get("required_dimensions") or []
    )
    return required or [_UNSPECIFIED_MUTATION_DIMENSION]


def semantic_owner_mutation_dimensions(
    binding: dict[str, Any],
    open_dimensions: list[str] | tuple[str, ...] | set[str],
) -> list[str]:
    """Map open dimensions to the exact owner role that can change them."""

    dimensions = set(ordered_unique_strings(open_dimensions))
    if not dimensions:
        return []
    if _UNSPECIFIED_MUTATION_DIMENSION in dimensions:
        return [_UNSPECIFIED_MUTATION_DIMENSION]
    role = str(binding.get("role") or "")
    if role == "demand_owner":
        # An exact reference owner is also part of the atomic mutation locus
        # when the referenced provider changes identity or call shape.  This
        # is a graph relation between two contracts, not a symbol-name or
        # file-level affinity.  Implementation-only provider changes leave
        # the consumer read-only.
        supported = set(_REFERENCE_COUPLED_PROVIDER_DIMENSIONS)
    elif role == "contract_declaration":
        supported = set(_DECLARATION_MUTATION_DIMENSIONS)
    elif role in {"repair_locus", "contract_entity"}:
        supported = set(_PROVIDER_MUTATION_DIMENSIONS)
    elif role in {
        "diagnostic_provider_owner",
        "diagnostic_consumer_owner",
    }:
        supported = dimensions
    elif role == "direct_dependency_owner":
        # This owner is the provider of a dependency used by the open
        # contract, not the owner of the open contract's outgoing integration
        # edge.  In particular, an unmet MODULE_DEVICE_TABLE/callback/call
        # reference must be repaired at its demand owner; granting the same
        # integration_reference authority to an already-closed dependency can
        # place the reference in an unrelated header or provider body.
        supported = set(_PROVIDER_MUTATION_DIMENSIONS)
    else:
        supported = set()
    return sorted(dimensions.intersection(supported))


def semantic_mutable_owner_key(
    binding: dict[str, Any],
) -> str | None:
    """Return a tree-independent identity for one exact mutable entity.

    Reference IDs and evidence roles identify observations, not mutation
    owners.  This key deliberately uses the resolved entity contract so that
    multiple references into the same function (or multiple gaps incident on
    it) share one transaction owner across old and donor trees.
    """

    descriptors = [
        descriptor
        for descriptor in (
            binding.get("current_descriptor"),
            binding.get("donor_descriptor"),
        )
        if isinstance(descriptor, dict) and descriptor
    ]
    descriptor = next(
        (
            candidate
            for candidate in descriptors
            if normalize_relpath(
                str(candidate.get("file") or candidate.get("path") or "")
            )
            and str(
                candidate.get("kind")
                or candidate.get("entity_kind")
                or binding.get("entity_kind")
                or ""
            )
            and str(
                candidate.get("contract_key")
                or candidate.get("key")
                or binding.get("contract_key")
                or ""
            )
        ),
        descriptors[0] if descriptors else {},
    )
    relative_path = normalize_relpath(
        str(
            descriptor.get("file")
            or descriptor.get("path")
            or binding.get("file")
            or ""
        )
    )
    entity_kind = str(
        descriptor.get("kind")
        or descriptor.get("entity_kind")
        or binding.get("entity_kind")
        or ""
    )
    container = (
        descriptor.get("container")
        if isinstance(descriptor.get("container"), dict)
        else {}
    )
    if entity_kind in {"field", "enum_constant"}:
        container_kind = str(container.get("container_kind") or "")
        container_name = str(container.get("container_name") or "")
        if (
            relative_path
            and container_kind in {"struct", "union", "enum"}
            and container_name
        ):
            return stable_id(
                "semantic-mutable-container",
                relative_path,
                container_kind,
                container_name,
            )
        return None

    contract = str(
        descriptor.get("contract_key")
        or descriptor.get("key")
        or binding.get("contract_key")
        or ""
    )
    if not relative_path or not entity_kind or not contract:
        return None
    definition_kind = (
        "definition"
        if descriptor.get("is_definition") is True
        else "declaration"
        if descriptor.get("is_definition") is False
        else "entity"
    )
    return stable_id(
        "semantic-mutable-entity",
        relative_path,
        entity_kind,
        contract,
        definition_kind,
    )


def semantic_observation_owner_key(
    binding: dict[str, Any],
) -> str | None:
    """Identify one initial graph owner independently of reference count.

    A demand owner is the exact relation between one enclosing consumer entity
    and one demanded contract. Individual call/reference nodes are provenance
    on that relation, not separate mutation owners.
    """

    role = str(binding.get("role") or "")
    if role not in _INITIAL_SEMANTIC_OWNER_ROLES:
        return None
    relative_path = normalize_relpath(str(binding.get("file") or ""))
    contract_key = str(binding.get("contract_key") or "").strip()
    entity_kind = str(binding.get("entity_kind") or "").strip()
    symbol = str(binding.get("symbol") or "").strip()
    demand_reference = (
        binding.get("demand_reference")
        if isinstance(binding.get("demand_reference"), dict)
        else {}
    )
    demanded_contract = (
        str(
            binding.get("demand_contract_key")
            or demand_reference.get("contract_key")
            or ""
        ).strip()
        if role == "demand_owner"
        else ""
    )
    if not relative_path or not (
        contract_key or symbol or demanded_contract
    ):
        return None
    return stable_id(
        "semantic-owner-v2",
        role,
        relative_path,
        contract_key,
        entity_kind,
        symbol,
        demanded_contract,
    )


def _graph_root_ids(unit: dict[str, Any]) -> list[str]:
    roots: list[str] = []
    for key in ("code_graph_root_cause_cluster", "root_cause_cluster"):
        root = unit.get(key)
        if not isinstance(root, dict):
            continue
        node_id = str(root.get("node_id") or "").strip()
        if node_id:
            roots.append(node_id)
    return list(dict.fromkeys(roots))


def owner_key_for_unit(
    *,
    unit: dict[str, Any],
    touched_files: list[str] | None = None,
) -> str:
    roots = _graph_root_ids(unit)
    if roots:
        return "code_graph_root:" + stable_id("roots", *sorted(roots))

    work_unit_id = str(unit.get("work_unit_id") or unit.get("id") or "").strip()
    if work_unit_id:
        return "work_unit:" + stable_id("work-unit", work_unit_id)

    touched = sorted(ordered_unique_relpaths(touched_files or []))
    if touched:
        return "patch_files:" + stable_id("files", *touched)
    return "unmapped_unit:" + stable_id(
        "unit",
        "repair",
        str(unit.get("title") or ""),
        str(unit.get("scope") or ""),
    )
