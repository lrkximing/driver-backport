from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable

from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths as _ordered_paths,
    ordered_unique_strings as _ordered_strings,
    stable_id,
)
from driver_backport.migration.owner_key import (
    semantic_mutable_owner_key,
    semantic_observation_owner_key,
    semantic_open_mutation_dimensions,
    semantic_owner_mutation_dimensions,
)
from driver_backport.migration.semantic_closure_state import (
    CHANGE_REQUIRED,
    PATCH_PENDING_COMPILE,
    PENDING_DECISION,
    semantic_gap_id,
)


SEMANTIC_CLOSURE_PROJECTION_SCHEMA = "semantic_closure_projection_v1"
SEMANTIC_CLOSURE_WORK_UNIT_SCHEMA = "semantic_closure_work_unit_v1"
_EXPANDED_OWNER_AUTHORITIES = {
    "typed_direct_dependency_edge",
    "typed_diagnostic_provider_resolution",
    "typed_diagnostic_consumer_resolution",
}
_PROVIDER_OWNER_ROLES = {
    "repair_locus",
    "contract_entity",
    "contract_declaration",
}


def _compact_entity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: deepcopy(value.get(key))
        for key in (
            "id",
            "entity_id",
            "tree",
            "file",
            "symbol",
            "name",
            "kind",
            "entity_kind",
            "cursor_kind",
            "contract_key",
            "key",
            "usr",
            "container",
            "is_definition",
            "extent",
            "signature_fingerprint",
            "interface_fingerprint",
            "implementation_fingerprint",
            "syntax_fingerprint",
            "source_fingerprint",
            "digest",
        )
        if value.get(key) not in (None, "", [], {})
    }


def _compact_source_unit(
    unit_id: str,
    unit: dict[str, Any],
) -> dict[str, Any]:
    """Keep only fields consumed by semantic scheduling and file scope."""

    return {"id": str(unit_id)} | {
        key: deepcopy(unit.get(key))
        for key in (
            "depends_on_work_unit_ids",
            "allowed_fileset",
        )
        if unit.get(key) not in (None, "", [], {})
    }


def _compact_owner_binding(
    *,
    role: str,
    file: object,
    contract_key: object,
    symbol: object,
    entity_kind: object,
    exact_node_refs: Iterable[object],
    current_descriptor: object = None,
    donor_descriptor: object = None,
    target_presence: object = None,
    old_extent: object = None,
    donor_extent: object = None,
    demand_contract_key: object = None,
) -> dict[str, Any] | None:
    relative_path = normalize_relpath(str(file or ""))
    normalized_contract = str(contract_key or "").strip()
    normalized_symbol = str(symbol or "").strip()
    normalized_kind = str(entity_kind or "").strip()
    node_refs = _ordered_strings(exact_node_refs)
    if not relative_path or not (
        normalized_contract or normalized_symbol or node_refs
    ):
        return None
    binding = {
        "role": str(role),
        "file": relative_path,
        "contract_key": normalized_contract or None,
        "symbol": normalized_symbol or None,
        "entity_kind": normalized_kind or None,
        "exact_node_refs": node_refs,
        "current_descriptor": _compact_entity(current_descriptor),
        "donor_descriptor": _compact_entity(donor_descriptor),
        "target_presence": str(target_presence or "") or None,
        # Extents are audit hints only. Patch authorization must resolve the
        # descriptor again in the current tree before every apply attempt.
        "old_extent_hint": (
            deepcopy(old_extent) if isinstance(old_extent, dict) else {}
        ),
        "donor_extent_hint": (
            deepcopy(donor_extent)
            if isinstance(donor_extent, dict)
            else {}
        ),
    }
    normalized_demand_contract = str(demand_contract_key or "").strip()
    if normalized_demand_contract:
        binding["demand_contract_key"] = normalized_demand_contract
    owner_key = semantic_observation_owner_key(binding)
    if owner_key is None:
        return None
    return {"owner_key": owner_key, **binding}


def _add_owner_binding(
    bindings: dict[str, dict[str, Any]],
    binding: dict[str, Any],
) -> None:
    owner_key = str(binding["owner_key"])
    existing = bindings.get(owner_key)
    if existing is None:
        bindings[owner_key] = binding
        return
    existing["exact_node_refs"] = _ordered_strings(
        [
            *list(existing.get("exact_node_refs") or []),
            *list(binding.get("exact_node_refs") or []),
        ]
    )


def _owner_bindings_for_unit(
    unit: dict[str, Any],
    *,
    evidence_ids: set[str],
) -> list[dict[str, Any]]:
    bindings: dict[str, dict[str, Any]] = {}
    for raw_locus in unit.get("repair_loci") or []:
        if not isinstance(raw_locus, dict):
            continue
        evidence_id = str(
            raw_locus.get("contract_evidence_id") or ""
        )
        if evidence_ids and evidence_id not in evidence_ids:
            continue
        binding = _compact_owner_binding(
            role="repair_locus",
            file=raw_locus.get("file"),
            contract_key=raw_locus.get("contract_key"),
            symbol=raw_locus.get("symbol"),
            entity_kind=raw_locus.get("entity_kind"),
            exact_node_refs=[
                raw_locus.get("id"),
                raw_locus.get("old_entity_id"),
                raw_locus.get("donor_entity_id"),
            ],
            current_descriptor={
                "id": raw_locus.get("old_entity_id"),
                "file": raw_locus.get("file"),
                "symbol": raw_locus.get("symbol"),
                "kind": raw_locus.get("entity_kind"),
                "contract_key": raw_locus.get("contract_key"),
                "extent": raw_locus.get("old_extent"),
                "is_definition": True,
            },
            donor_descriptor={
                "id": raw_locus.get("donor_entity_id"),
                "file": raw_locus.get("file"),
                "symbol": raw_locus.get("symbol"),
                "kind": raw_locus.get("entity_kind"),
                "contract_key": raw_locus.get("contract_key"),
                "extent": raw_locus.get("donor_extent"),
                "is_definition": True,
            },
            target_presence=raw_locus.get("target_presence"),
            old_extent=raw_locus.get("old_extent"),
            donor_extent=raw_locus.get("donor_extent"),
        )
        if binding is not None:
            _add_owner_binding(bindings, binding)

    for raw_anchor in unit.get("demand_anchors") or []:
        if not isinstance(raw_anchor, dict):
            continue
        evidence_id = str(
            raw_anchor.get("contract_evidence_id") or ""
        )
        if evidence_ids and evidence_id not in evidence_ids:
            continue
        owner = (
            raw_anchor.get("owner_entity")
            if isinstance(raw_anchor.get("owner_entity"), dict)
            else {}
        )
        referenced = (
            raw_anchor.get("referenced_entity")
            if isinstance(raw_anchor.get("referenced_entity"), dict)
            else {}
        )
        demand_contract_key = str(
            raw_anchor.get("contract_key")
            or referenced.get("contract_key")
            or ""
        )
        binding = _compact_owner_binding(
            role="demand_owner",
            file=(
                owner.get("file")
                or raw_anchor.get("source_file")
            ),
            contract_key=owner.get("contract_key"),
            symbol=owner.get("symbol"),
            entity_kind=owner.get("kind"),
            exact_node_refs=[
                raw_anchor.get("id"),
                raw_anchor.get("reference_id"),
                owner.get("id"),
                referenced.get("id"),
            ],
            current_descriptor=owner,
            donor_descriptor=owner,
            target_presence="existing_consumer",
            donor_extent=owner.get("extent"),
            demand_contract_key=demand_contract_key,
        )
        if binding is not None:
            binding["demand_reference"] = {
                key: deepcopy(raw_anchor.get(key))
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
                if raw_anchor.get(key) not in (None, "", [], {})
            }
            _add_owner_binding(bindings, binding)
    return list(bindings.values())


def _demand_owner_bindings_for_evidence(
    evidence: dict[str, Any],
    *,
    evidence_id: str,
) -> list[dict[str, Any]]:
    """Preserve demand owners even when a contract needs no repair unit."""

    references = [
        dict(item)
        for item in evidence.get("consumer_references") or []
        if isinstance(item, dict)
    ]
    if not references and isinstance(
        evidence.get("consumer_reference"), dict
    ):
        references = [dict(evidence["consumer_reference"])]

    bindings: dict[str, dict[str, Any]] = {}
    for reference in references:
        owner = (
            reference.get("owner_entity")
            if isinstance(reference.get("owner_entity"), dict)
            else {}
        )
        referenced = (
            reference.get("referenced_entity")
            if isinstance(reference.get("referenced_entity"), dict)
            else {}
        )
        reference_id = str(reference.get("id") or "")
        binding = _compact_owner_binding(
            role="demand_owner",
            file=owner.get("file") or reference.get("source_file"),
            contract_key=owner.get("contract_key"),
            symbol=owner.get("symbol"),
            entity_kind=owner.get("kind"),
            exact_node_refs=[
                evidence_id,
                reference_id,
                owner.get("id"),
                referenced.get("id"),
            ],
            current_descriptor=owner,
            donor_descriptor=owner,
            target_presence="existing_consumer",
            donor_extent=owner.get("extent"),
            demand_contract_key=(
                reference.get("contract_key")
                or referenced.get("contract_key")
            ),
        )
        if binding is None:
            continue
        binding["demand_reference"] = {
            key: deepcopy(reference.get(key))
            for key in (
                "id",
                "source_file",
                "line",
                "column",
                "extent",
                "symbol",
                "kind",
                "use_kind",
                "cursor_kind",
                "dependency_scope",
                "contract_key",
            )
            if reference.get(key) not in (None, "", [], {})
        }
        if reference_id:
            binding["demand_reference"]["reference_id"] = reference_id
        _add_owner_binding(bindings, binding)
    return list(bindings.values())


def _declaration_owner_bindings(
    evidence: dict[str, Any],
    *,
    evidence_id: str,
) -> list[dict[str, Any]]:
    """Project donor declarations as exact mutation owners.

    A declaration is paired with an old declaration only when both resolve to
    the same source file and contract.  Implicit declarations recovered at a
    call site therefore cannot masquerade as the donor header declaration.
    """

    old_declarations = [
        dict(item)
        for item in evidence.get("old_declarations") or []
        if isinstance(item, dict)
    ]
    old_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for declaration in old_declarations:
        identity = (
            normalize_relpath(str(declaration.get("file") or "")),
            str(
                declaration.get("contract_key")
                or declaration.get("key")
                or ""
            ),
            str(
                declaration.get("kind")
                or declaration.get("entity_kind")
                or ""
            ),
        )
        extent = declaration.get("extent")
        if (
            all(identity)
            and isinstance(extent, dict)
            and int(extent.get("start_line") or 0) > 0
        ):
            old_by_identity.setdefault(identity, declaration)

    bindings: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in evidence.get("donor_declarations") or []:
        if not isinstance(raw, dict):
            continue
        donor = dict(raw)
        identity = (
            normalize_relpath(str(donor.get("file") or "")),
            str(donor.get("contract_key") or donor.get("key") or ""),
            str(donor.get("kind") or donor.get("entity_kind") or ""),
        )
        if not all(identity) or identity in bindings:
            continue
        current = old_by_identity.get(identity, {})
        binding = _compact_owner_binding(
            role="contract_declaration",
            file=identity[0],
            contract_key=identity[1],
            symbol=(
                donor.get("symbol")
                or donor.get("name")
                or current.get("symbol")
                or current.get("name")
            ),
            entity_kind=identity[2],
            exact_node_refs=[
                evidence_id,
                current.get("id") or current.get("entity_id"),
                donor.get("id") or donor.get("entity_id"),
            ],
            current_descriptor=current,
            donor_descriptor=donor,
            target_presence=(
                "existing_old_declaration"
                if current
                else "missing_old_declaration"
            ),
            old_extent=current.get("extent"),
            donor_extent=donor.get("extent"),
        )
        if binding is not None:
            bindings[identity] = binding
    return list(bindings.values())


def project_initial_semantic_differences(
    analysis: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Project every reachable driver demand into the closure ledger.

    ChangeAtoms identify demands that own an initial repair or validation
    unit. Demands that were satisfied or evidence-blocked during source
    analysis remain no-edit ledger entries and are re-evaluated against the
    compile-valid current tree.
    """

    atoms = [
        dict(item)
        for item in analysis.get("change_atoms") or []
        if isinstance(item, dict)
    ]
    raw_ledger = analysis.get("demand_ledger")
    evidence_records = [
        dict(item)
        for item in (
            raw_ledger
            if isinstance(raw_ledger, list)
            else analysis.get("contract_evidence") or []
        )
        if isinstance(item, dict)
    ]
    evidence_by_id: dict[str, dict[str, Any]] = {}
    evidence_by_identity: dict[str, dict[str, Any]] = {}
    for evidence in evidence_records:
        evidence_id = str(evidence.get("id") or "").strip()
        contract_identity = str(
            evidence.get("contract_identity") or ""
        ).strip()
        if not evidence_id or not contract_identity:
            raise ValueError(
                "every demand-ledger record requires id and "
                "contract_identity"
            )
        if evidence_id in evidence_by_id:
            raise ValueError(
                f"duplicate demand-ledger evidence id: {evidence_id}"
            )
        if contract_identity in evidence_by_identity:
            raise ValueError(
                "duplicate demand-ledger contract identity: "
                f"{contract_identity}"
            )
        evidence_by_id[evidence_id] = evidence
        evidence_by_identity[contract_identity] = evidence
    units = [
        dict(item)
        for item in plan.get("work_unit_plans") or []
        if isinstance(item, dict)
    ]
    units_by_id = {
        str(unit.get("id") or unit.get("work_unit_id") or ""): unit
        for unit in units
        if str(unit.get("id") or unit.get("work_unit_id") or "")
    }
    unit_ids_by_atom: dict[str, list[str]] = defaultdict(list)
    for unit_id, unit in units_by_id.items():
        for atom_id in unit.get("difference_atom_ids") or []:
            normalized = str(atom_id or "")
            if normalized and unit_id not in unit_ids_by_atom[normalized]:
                unit_ids_by_atom[normalized].append(unit_id)

    atoms_by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for atom in atoms:
        atom_id = str(atom.get("id") or "").strip()
        contract_identity = str(
            atom.get("contract_identity") or ""
        ).strip()
        evidence_id = str(
            atom.get("contract_evidence_id") or ""
        ).strip()
        if not atom_id or not contract_identity or not evidence_id:
            raise ValueError(
                "every semantic ChangeAtom requires id, contract_identity, "
                "and contract_evidence_id"
            )
        if evidence_id not in evidence_by_id:
            raise ValueError(
                f"ChangeAtom {atom_id} references missing evidence "
                f"{evidence_id}"
            )
        if str(
            evidence_by_id[evidence_id].get("contract_identity") or ""
        ) != contract_identity:
            raise ValueError(
                f"ChangeAtom {atom_id} contract identity does not match "
                f"evidence {evidence_id}"
            )
        if atom_id not in unit_ids_by_atom:
            raise ValueError(
                f"ChangeAtom {atom_id} is not owned by a DifferenceUnit"
            )
        atoms_by_contract[contract_identity].append(atom)

    gaps: list[dict[str, Any]] = []
    source_unit_ids: set[str] = set()
    for contract_identity, evidence in sorted(
        evidence_by_identity.items()
    ):
        evidence_id = str(evidence.get("id") or "")
        evidence_ids = {evidence_id}
        contract_atoms = atoms_by_contract.get(contract_identity, [])
        contract_unit_ids = _ordered_strings(
            unit_id
            for atom in contract_atoms
            for unit_id in unit_ids_by_atom[str(atom["id"])]
        )
        contract_units = [
            units_by_id[unit_id] for unit_id in contract_unit_ids
        ]
        source_unit_ids.update(contract_unit_ids)
        owner_bindings: dict[str, dict[str, Any]] = {}
        for unit in contract_units:
            for binding in _owner_bindings_for_unit(
                unit,
                evidence_ids=evidence_ids,
            ):
                _add_owner_binding(owner_bindings, binding)
        for binding in _demand_owner_bindings_for_evidence(
            evidence,
            evidence_id=evidence_id,
        ):
            _add_owner_binding(owner_bindings, binding)
        donor_entity = (
            evidence.get("donor_entity")
            if isinstance(evidence.get("donor_entity"), dict)
            else {}
        )
        old_entity = (
            evidence.get("old_entity")
            if isinstance(evidence.get("old_entity"), dict)
            else {}
        )
        contract_binding = _compact_owner_binding(
            role="contract_entity",
            file=(
                old_entity.get("file")
                or donor_entity.get("file")
            ),
            contract_key=(
                old_entity.get("contract_key")
                or donor_entity.get("contract_key")
            ),
            symbol=(
                old_entity.get("symbol")
                or donor_entity.get("symbol")
            ),
            entity_kind=(
                old_entity.get("kind")
                or donor_entity.get("kind")
            ),
            exact_node_refs=[
                old_entity.get("id"),
                donor_entity.get("id"),
                evidence_id,
            ],
            current_descriptor=old_entity or donor_entity,
            donor_descriptor=donor_entity or old_entity,
            target_presence=(
                "existing_old_entity"
                if old_entity
                else "missing_old_entity"
            ),
            old_extent=old_entity.get("extent"),
            donor_extent=donor_entity.get("extent"),
        )
        if contract_binding is not None:
            _add_owner_binding(owner_bindings, contract_binding)
        for declaration_binding in _declaration_owner_bindings(
            evidence,
            evidence_id=evidence_id,
        ):
            _add_owner_binding(owner_bindings, declaration_binding)
        gaps.append(
            {
                "id": semantic_gap_id(contract_identity),
                "contract_identity": contract_identity,
                "raw_hunk_ids": _ordered_strings(
                    hunk_id
                    for atom in contract_atoms
                    for hunk_id in atom.get("raw_hunk_ids") or []
                ),
                "contract_evidence": deepcopy(evidence),
                "source_work_unit_ids": contract_unit_ids,
                "owner_bindings": list(owner_bindings.values()),
            }
        )

    referenced_unit_ids = {
        unit_id
        for gap in gaps
        for unit_id in gap["source_work_unit_ids"]
    }
    if referenced_unit_ids != source_unit_ids:
        raise ValueError("semantic source-unit projection is inconsistent")
    source_units = {
        unit_id: _compact_source_unit(
            unit_id,
            units_by_id[unit_id],
        )
        for unit_id in sorted(referenced_unit_ids)
    }
    return {
        "schema": SEMANTIC_CLOSURE_PROJECTION_SCHEMA,
        "analysis_schema": analysis.get("schema"),
        "analysis_revision": analysis.get("analysis_revision"),
        "plan_schema": plan.get("schema"),
        "difference_unit_revision": plan.get(
            "difference_unit_revision"
        ),
        "gaps": gaps,
        "source_units": source_units,
        "summary": {
            "projected_demand_count": len(evidence_by_identity),
            "change_atom_count": len(atoms),
            "demand_with_change_atom_count": len(atoms_by_contract),
            "no_edit_demand_count": (
                len(evidence_by_identity) - len(atoms_by_contract)
            ),
            "semantic_gap_count": len(gaps),
            "source_work_unit_count": len(source_units),
        },
    }


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _direct_closed_provider_bindings(
    state: dict[str, Any],
    member_gaps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose one-hop providers that are already closed but still editable.

    A closed provider gap records exact donor dependency edges through its
    demand owners. If an open component owns that exact consumer entity, the
    provider is a support locus for the open repair. It is not reopened and it
    does not merge components; fresh batch validation still rechecks any
    surviving edit to it.
    """

    open_gap_ids_by_owner: dict[str, list[str]] = defaultdict(list)
    for gap in member_gaps:
        gap_id = str(gap.get("id") or "")
        open_dimensions = set(semantic_open_mutation_dimensions(gap))
        # A pure integration-reference obligation belongs to the exact
        # consumer/demand owner.  A closed dependency provider may be useful
        # read-only evidence, but it is not a writable support locus for that
        # edge.  Keeping it out of the unit also prevents its file from
        # entering the automatic allowed-fileset.
        if open_dimensions and open_dimensions <= {"integration_reference"}:
            continue
        for binding in gap.get("owner_bindings") or []:
            if (
                not isinstance(binding, dict)
                or str(binding.get("role") or "")
                not in _PROVIDER_OWNER_ROLES
            ):
                continue
            mutable_key = semantic_mutable_owner_key(binding)
            if mutable_key and gap_id:
                open_gap_ids_by_owner[mutable_key].append(gap_id)
    if not open_gap_ids_by_owner:
        return []

    support: dict[str, dict[str, Any]] = {}
    for closed_gap in (state.get("gaps") or {}).values():
        if (
            not isinstance(closed_gap, dict)
            or str(closed_gap.get("status") or "")
            not in {"closed_modified", "closed_no_change"}
        ):
            continue
        incident_open_gap_ids: list[str] = []
        edge_refs: list[str] = []
        for binding in closed_gap.get("owner_bindings") or []:
            if (
                not isinstance(binding, dict)
                or str(binding.get("role") or "") != "demand_owner"
            ):
                continue
            mutable_key = semantic_mutable_owner_key(binding)
            if mutable_key not in open_gap_ids_by_owner:
                continue
            incident_open_gap_ids.extend(
                open_gap_ids_by_owner[mutable_key]
            )
            edge_refs.extend(binding.get("exact_node_refs") or [])
        incident_open_gap_ids = _ordered_strings(incident_open_gap_ids)
        if not incident_open_gap_ids:
            continue
        for binding in closed_gap.get("owner_bindings") or []:
            if (
                not isinstance(binding, dict)
                or str(binding.get("role") or "")
                not in _PROVIDER_OWNER_ROLES
            ):
                continue
            provider = deepcopy(binding)
            provider_mutable_key = semantic_mutable_owner_key(provider)
            if not provider_mutable_key:
                continue
            provider["owner_key"] = stable_id(
                "semantic-direct-support-owner",
                provider_mutable_key,
                *sorted(incident_open_gap_ids),
                *sorted(_ordered_strings(edge_refs)),
            )
            existing = support.get(str(provider["owner_key"]), {})
            provider["role"] = "direct_dependency_owner"
            provider["authority"] = "typed_direct_dependency_edge"
            provider["semantic_gap_ids"] = incident_open_gap_ids
            provider["source_owner_keys"] = _ordered_strings(
                [
                    *(existing.get("source_owner_keys") or []),
                    binding.get("owner_key"),
                ]
            )
            provider["exact_node_refs"] = _ordered_strings(
                [
                    *(binding.get("exact_node_refs") or []),
                    *edge_refs,
                    closed_gap.get("id"),
                ]
            )
            if existing:
                provider["semantic_gap_ids"] = _ordered_strings(
                    [
                        *(existing.get("semantic_gap_ids") or []),
                        *provider["semantic_gap_ids"],
                    ]
                )
                provider["exact_node_refs"] = _ordered_strings(
                    [
                        *(existing.get("exact_node_refs") or []),
                        *provider["exact_node_refs"],
                    ]
                )
                provider["source_owner_keys"] = _ordered_strings(
                    [
                        *(existing.get("source_owner_keys") or []),
                        *provider["source_owner_keys"],
                    ]
                )
            support[str(provider["owner_key"])] = provider
    return list(support.values())


def materialize_semantic_work_units(
    state: dict[str, Any],
    *,
    source_units: dict[str, dict[str, Any]],
    statuses: Iterable[str] = (
        PENDING_DECISION,
        CHANGE_REQUIRED,
        PATCH_PENDING_COMPILE,
    ),
) -> list[dict[str, Any]]:
    """Group open gaps by exact DifferenceUnit and mutable code-graph owner.

    File names, symbol spelling similarity, and error categories never merge
    gaps. Two gaps share a work unit only when the initial dependency graph
    already put them in the same DifferenceUnit, their open dimensions bind
    to the same exact mutation owner, or member contracts resolve to the same
    enclosing struct, union, or enum entity. A shared read-only consumer does
    not merge independent repairs.
    """

    selected_statuses = {str(item) for item in statuses}
    gaps = {
        str(gap_id): gap
        for gap_id, gap in (state.get("gaps") or {}).items()
        if isinstance(gap, dict)
        and str(gap.get("status") or "") in selected_statuses
    }
    if not gaps:
        return []
    disjoint = _DisjointSet(gaps)
    gap_ids_by_source_unit: dict[str, list[str]] = defaultdict(list)
    gap_ids_by_mutation_owner: dict[str, list[str]] = defaultdict(list)
    gap_ids_by_mutable_owner: dict[str, list[str]] = defaultdict(list)
    for gap_id, gap in gaps.items():
        for unit_id in gap.get("source_work_unit_ids") or []:
            if str(unit_id) in source_units:
                gap_ids_by_source_unit[str(unit_id)].append(gap_id)
        for binding in gap.get("owner_bindings") or []:
            if not isinstance(binding, dict):
                continue
            mutation_dimensions = semantic_owner_mutation_dimensions(
                binding,
                semantic_open_mutation_dimensions(gap),
            )
            if not mutation_dimensions:
                continue
            owner_key = str(binding.get("owner_key") or "")
            if owner_key:
                gap_ids_by_mutation_owner[owner_key].append(gap_id)
            mutable_key = semantic_mutable_owner_key(binding)
            if mutable_key:
                gap_ids_by_mutable_owner[mutable_key].append(gap_id)
    for members in [
        *gap_ids_by_source_unit.values(),
        *gap_ids_by_mutation_owner.values(),
        *gap_ids_by_mutable_owner.values(),
    ]:
        first = members[0]
        for other in members[1:]:
            disjoint.union(first, other)

    components: dict[str, list[str]] = defaultdict(list)
    for gap_id in sorted(gaps):
        components[disjoint.find(gap_id)].append(gap_id)

    unit_id_by_source_id: dict[str, str] = {}
    work_units: list[dict[str, Any]] = []
    for member_gap_ids in components.values():
        member_gaps = [gaps[gap_id] for gap_id in member_gap_ids]
        member_source_ids = sorted(
            {
                str(unit_id)
                for gap in member_gaps
                for unit_id in gap.get("source_work_unit_ids") or []
                if str(unit_id) in source_units
            }
        )
        owner_bindings: dict[str, dict[str, Any]] = {}
        for gap in member_gaps:
            for binding in gap.get("owner_bindings") or []:
                if isinstance(binding, dict) and binding.get("owner_key"):
                    owner_key = str(binding["owner_key"])
                    merged = {
                        **owner_bindings.get(owner_key, {}),
                        **deepcopy(binding),
                    }
                    for field in ("exact_node_refs", "semantic_gap_ids"):
                        merged[field] = _ordered_strings(
                            [
                                *list(
                                    owner_bindings.get(owner_key, {}).get(
                                        field
                                    )
                                    or []
                                ),
                                *list(binding.get(field) or []),
                            ]
                        )
                    owner_bindings[owner_key] = merged
        for binding in _direct_closed_provider_bindings(
            state,
            member_gaps,
        ):
            owner_bindings.setdefault(str(binding["owner_key"]), binding)
        base_allowed_files = _ordered_paths(
            path
            for source_id in member_source_ids
            for path in (
                source_units[source_id].get("allowed_fileset") or []
            )
        )
        allowed_files = _ordered_paths(
            [
                *base_allowed_files,
                *(
                    binding.get("file")
                    for binding in owner_bindings.values()
                ),
            ]
        )
        root_owner_keys = sorted(
            owner_key
            for owner_key, binding in owner_bindings.items()
            if str(binding.get("authority") or "")
            not in _EXPANDED_OWNER_AUTHORITIES
        )
        work_unit_id = stable_id(
            "semantic-closure-unit",
            *sorted(member_gap_ids),
            *member_source_ids,
            *root_owner_keys,
        )
        for source_id in member_source_ids:
            unit_id_by_source_id[source_id] = work_unit_id
        work_units.append(
            {
                "schema": SEMANTIC_CLOSURE_WORK_UNIT_SCHEMA,
                "id": work_unit_id,
                "work_unit_id": work_unit_id,
                "unit_kind": (
                    "semantic_gap_repair"
                    if any(
                        str(gap.get("status") or "")
                        in {CHANGE_REQUIRED, PATCH_PENDING_COMPILE}
                        for gap in member_gaps
                    )
                    else "semantic_gap_review"
                ),
                "gap_ids": sorted(member_gap_ids),
                "gap_statuses": {
                    gap_id: str(gaps[gap_id].get("status") or "")
                    for gap_id in sorted(member_gap_ids)
                },
                "source_work_unit_ids": member_source_ids,
                "owner_bindings": list(owner_bindings.values()),
                "base_allowed_fileset": base_allowed_files,
                "allowed_fileset": allowed_files,
                "depends_on_work_unit_ids": [],
            }
        )

    work_units_by_id = {
        str(unit["id"]): unit for unit in work_units
    }
    for work_unit in work_units:
        dependencies: set[str] = set()
        for source_id in work_unit["source_work_unit_ids"]:
            source = source_units[source_id]
            for dependency_source_id in (
                source.get("depends_on_work_unit_ids") or []
            ):
                dependency_unit_id = unit_id_by_source_id.get(
                    str(dependency_source_id)
                )
                if (
                    dependency_unit_id
                    and dependency_unit_id != work_unit["id"]
                    and dependency_unit_id in work_units_by_id
                ):
                    dependencies.add(dependency_unit_id)
        work_unit["depends_on_work_unit_ids"] = sorted(dependencies)
    return sorted(
        work_units,
        key=lambda unit: (
            len(unit.get("depends_on_work_unit_ids") or []),
            str(unit.get("id") or ""),
        ),
    )
