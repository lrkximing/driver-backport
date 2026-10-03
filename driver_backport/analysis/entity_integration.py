#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from driver_backport.analysis.code_index import (
    CodeIndex,
    SOURCE_SUFFIXES,
    parse_structural_entity_file,
    reference_crosses_entity_boundary,
)
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths,
    stable_id,
)
from driver_backport.planning.execution_phase import (
    EXECUTION_PHASE_SEMANTIC_CLOSURE,
)


TRANSLATION_UNIT_SUFFIXES = {".c", ".cc", ".cpp"}
HEADER_SUFFIXES = {".h", ".hpp"}
INTEGRATION_ANALYSIS_REVISION = "post_materialization_integration_v4"


def _contract(record: object) -> str:
    if not isinstance(record, dict):
        return ""
    return str(record.get("contract_key") or record.get("key") or "")


def _entity_id(record: object) -> str:
    if not isinstance(record, dict):
        return ""
    return str(record.get("id") or record.get("entity_id") or "")


def _entity_kind(record: object) -> str:
    if not isinstance(record, dict):
        return ""
    return str(record.get("kind") or record.get("entity_kind") or "")


def _compact_entity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "id",
            "entity_id",
            "tree",
            "file",
            "kind",
            "entity_kind",
            "symbol",
            "contract_key",
            "key",
            "is_definition",
            "extent",
            "container",
            "storage_class",
            "linkage",
            "signature",
            "signature_fingerprint",
            "interface",
            "interface_fingerprint",
            "source_fingerprint",
        )
        if record.get(key) not in (None, "", [], {})
    }


def _definition_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in report.get("entities") or []
        if isinstance(record, dict)
        and record.get("is_definition") is True
        and _contract(record)
    ]


def _evidence_side_records(
    evidence: dict[str, Any],
    side: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    entity = evidence.get(f"{side}_entity")
    if isinstance(entity, dict):
        records.append(dict(entity))
    records.extend(
        dict(item)
        for item in evidence.get(f"{side}_declarations") or []
        if isinstance(item, dict)
    )
    return records


def consumer_facing_contract_change(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Prove whether a provider's typed external contract changed.

    This compares parser-produced signature and interface fingerprints only.
    Implementation/source differences are deliberately excluded because they
    do not by themselves imply that direct consumers must be edited.
    """

    comparison = (
        evidence.get("comparison")
        if isinstance(evidence.get("comparison"), dict)
        else {}
    )
    relations: dict[str, dict[str, Any]] = {}
    changed_dimensions: list[str] = []
    for dimension in ("signature", "interface"):
        field = f"{dimension}_fingerprint"
        old_values = sorted(
            {
                str(record.get(field) or "")
                for record in _evidence_side_records(evidence, "old")
                if str(record.get(field) or "")
            }
        )
        donor_values = sorted(
            {
                str(record.get(field) or "")
                for record in _evidence_side_records(evidence, "donor")
                if str(record.get(field) or "")
            }
        )
        observed_relation = str(comparison.get(dimension) or "")
        proven_different = bool(
            observed_relation == "different"
            or (
                old_values
                and donor_values
                and set(old_values) != set(donor_values)
            )
        )
        if proven_different:
            changed_dimensions.append(dimension)
        relations[dimension] = {
            "relation": (
                "different"
                if proven_different
                else "equivalent"
                if old_values
                and donor_values
                and set(old_values) == set(donor_values)
                else observed_relation or "unavailable"
            ),
            "old_fingerprints": old_values,
            "donor_fingerprints": donor_values,
        }
    return {
        "schema": "consumer_facing_contract_change_v1",
        "changed": bool(changed_dimensions),
        "changed_dimensions": changed_dimensions,
        "relations": relations,
        "authority": "typed_entity_and_declaration_fingerprints",
    }


def semantic_integration_requirements_from_unit(
    unit: dict[str, Any],
    *,
    touched_files: Iterable[object],
) -> list[dict[str, Any]]:
    """Build exact incoming-edge requirements for one semantic transaction."""

    if (
        str(unit.get("execution_phase") or "")
        != EXECUTION_PHASE_SEMANTIC_CLOSURE
    ):
        return []
    touched = set(ordered_unique_relpaths(touched_files))
    anchors_by_evidence: dict[str, set[str]] = {}
    for anchor in unit.get("demand_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        evidence_id = str(anchor.get("contract_evidence_id") or "")
        reference_id = str(
            anchor.get("reference_id")
            or anchor.get("donor_reference_id")
            or ""
        )
        if evidence_id and reference_id:
            anchors_by_evidence.setdefault(evidence_id, set()).add(
                reference_id
            )

    requirements: list[dict[str, Any]] = []
    for evidence in unit.get("contract_evidence") or []:
        if not isinstance(evidence, dict):
            continue
        change = consumer_facing_contract_change(evidence)
        if not change["changed"]:
            continue
        evidence_id = str(evidence.get("id") or "")
        old_records = _evidence_side_records(evidence, "old")
        donor_records = _evidence_side_records(evidence, "donor")
        provider_files = set(
            ordered_unique_relpaths(
                record.get("file")
                for record in [*old_records, *donor_records]
            )
        )
        if touched and not touched.intersection(provider_files):
            continue
        target_candidates = [
            record
            for record in donor_records
            if _contract(record) and _entity_kind(record)
        ]
        donor_entity = (
            dict(evidence.get("donor_entity") or {})
            if isinstance(evidence.get("donor_entity"), dict)
            else {}
        )
        definition_candidates = [
            record
            for record in target_candidates
            if record.get("is_definition") is True
        ]
        target = (
            donor_entity
            if _contract(donor_entity) and _entity_kind(donor_entity)
            else definition_candidates[0]
            if definition_candidates
            else target_candidates[0]
            if target_candidates
            else {}
        )
        target_contract = _contract(target)
        target_kind = _entity_kind(target)
        if not target_contract or not target_kind:
            continue
        reference_ids = set(anchors_by_evidence.get(evidence_id) or set())
        reference_ids.update(
            str(item.get("id") or "")
            for item in evidence.get("consumer_references") or []
            if isinstance(item, dict) and str(item.get("id") or "")
        )
        reference_ids.update(
            str(item.get("donor_reference_id") or "")
            for item in evidence.get(
                "consumer_reference_alignments"
            )
            or []
            if isinstance(item, dict)
            and str(item.get("donor_reference_id") or "")
        )
        if not reference_ids:
            continue
        consumer_references = [
            dict(item)
            for item in evidence.get("consumer_references") or []
            if isinstance(item, dict)
            and str(item.get("id") or "")
        ]
        requirements.append(
            {
                "schema": "semantic_integration_requirement_v1",
                "id": stable_id(
                    "semantic-integration-requirement",
                    evidence_id,
                    target_contract,
                    *sorted(reference_ids),
                ),
                "contract_evidence_id": evidence_id,
                "target_contract_key": target_contract,
                "target_symbol": target.get("symbol"),
                "target_kind": target_kind,
                "target_file": normalize_relpath(
                    str(target.get("file") or "")
                ),
                "target_entity": _compact_entity(target),
                "provider_files": sorted(provider_files),
                "consumer_reference_ids": sorted(reference_ids),
                "consumer_references": consumer_references,
                "contract_change": change,
            }
        )
    return requirements


def structural_entity_additions_from_transaction(
    spec: CaseSpec,
    *,
    transaction_dir: Path,
    touched_files: Iterable[object],
) -> dict[str, Any]:
    """Find exact donor-backed definitions introduced by one atomic patch."""

    transaction_dir = transaction_dir.resolve()
    backup_root = transaction_dir / "backups"
    normalized_touched_files = ordered_unique_relpaths(touched_files)
    added: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    parse_blockers: list[dict[str, Any]] = []
    for relative_path in normalized_touched_files:
        if Path(relative_path).suffix not in SOURCE_SUFFIXES:
            continue
        current_path = spec.resolve_old(relative_path)
        if not current_path.is_file():
            continue
        current_report = parse_structural_entity_file(
            root=spec.old_root_path,
            relative_path=relative_path,
            tree="old",
        )
        if not current_report.get("parse_ok"):
            parse_blockers.append(
                {
                    "tree": "old",
                    "file": relative_path,
                    "reason": "postimage_structural_parse_failed",
                    "parse_error": current_report.get("parse_error"),
                }
            )
            continue
        backup_path = backup_root / relative_path
        if backup_path.is_file():
            before_report = parse_structural_entity_file(
                root=backup_root,
                relative_path=relative_path,
                tree="pre_patch",
            )
            if not before_report.get("parse_ok"):
                parse_blockers.append(
                    {
                        "tree": "pre_patch",
                        "file": relative_path,
                        "reason": "preimage_structural_parse_failed",
                        "parse_error": before_report.get("parse_error"),
                    }
                )
                continue
            before_contracts = {
                _contract(record)
                for record in _definition_records(before_report)
            }
        else:
            before_contracts = set()

        donor_path = spec.resolve_donor(relative_path)
        donor_report = (
            parse_structural_entity_file(
                root=spec.primary_donor_root,
                relative_path=relative_path,
                tree="donor_0",
            )
            if donor_path.is_file()
            else {
                "parse_ok": False,
                "parse_error": "donor_file_missing",
                "entities": [],
            }
        )
        donor_by_contract: dict[str, list[dict[str, Any]]] = {}
        for donor_entity in _definition_records(donor_report):
            donor_by_contract.setdefault(
                _contract(donor_entity),
                [],
            ).append(donor_entity)

        for current_entity in _definition_records(current_report):
            contract_key = _contract(current_entity)
            if contract_key in before_contracts:
                continue
            donor_matches = donor_by_contract.get(contract_key) or []
            exact_matches = [
                donor_entity
                for donor_entity in donor_matches
                if _entity_kind(donor_entity)
                == _entity_kind(current_entity)
            ]
            if len(exact_matches) == 1:
                added.append(
                    {
                        "schema": "post_patch_integration_seed_v1",
                        "contract_key": contract_key,
                        "symbol": current_entity.get("symbol"),
                        "kind": _entity_kind(current_entity),
                        "file": relative_path,
                        "current_entity": _compact_entity(current_entity),
                        "donor_entity": _compact_entity(exact_matches[0]),
                    }
                )
            else:
                unmatched.append(
                    {
                        "contract_key": contract_key,
                        "symbol": current_entity.get("symbol"),
                        "kind": _entity_kind(current_entity),
                        "file": relative_path,
                        "donor_match_count": len(exact_matches),
                        "reason": (
                            "new_definition_has_no_unique_donor_contract"
                        ),
                    }
                )

    seeds_by_contract = {
        str(seed["contract_key"]): seed
        for seed in added
        if str(seed.get("contract_key") or "")
    }
    return {
        "schema": "post_patch_structural_additions_v1",
        "status": (
            "ready"
            if not parse_blockers
            else "partially_available"
            if seeds_by_contract
            else "unavailable"
        ),
        "transaction_dir": str(transaction_dir),
        "touched_files": normalized_touched_files,
        "seed_entities": [
            seeds_by_contract[key] for key in sorted(seeds_by_contract)
        ],
        "unmatched_additions": unmatched,
        "parse_blockers": parse_blockers,
        "summary": {
            "seed_count": len(seeds_by_contract),
            "unmatched_addition_count": len(unmatched),
            "parse_blocker_count": len(parse_blockers),
        },
    }


def _matching_references(
    reports: Iterable[dict[str, Any]],
    *,
    target_entity_ids: set[str],
    target_contracts: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    selected: dict[tuple[str, ...], dict[str, Any]] = {}
    accepted_contracts = set(target_contracts or set())
    for report in reports:
        translation_unit_file = normalize_relpath(
            str(report.get("translation_unit_file") or "")
        )
        for raw_reference in report.get("references") or []:
            if not isinstance(raw_reference, dict):
                continue
            reference = dict(raw_reference)
            referenced_entity = (
                reference.get("referenced_entity")
                if isinstance(
                    reference.get("referenced_entity"),
                    dict,
                )
                else {}
            )
            referenced_identity = (
                _contract(referenced_entity),
                _entity_kind(referenced_entity),
            )
            if not (
                str(reference.get("referenced_entity_id") or "")
                in target_entity_ids
                or referenced_identity in accepted_contracts
            ) or not reference_crosses_entity_boundary(reference):
                continue
            owner = (
                reference.get("owner_entity")
                if isinstance(reference.get("owner_entity"), dict)
                else {}
            )
            target = (
                reference.get("referenced_entity")
                if isinstance(
                    reference.get("referenced_entity"),
                    dict,
                )
                else {}
            )
            if (
                str(reference.get("dependency_scope") or "")
                != "local_driver"
                or not _contract(owner)
                or not _contract(target)
            ):
                continue
            identity = (
                normalize_relpath(
                    str(reference.get("source_file") or "")
                ),
                _contract(owner),
                _contract(target),
                str(reference.get("cursor_kind") or ""),
                str(reference.get("use_kind") or ""),
                str(
                    reference.get(
                        "reference_context_fingerprint"
                    )
                    or ""
                ),
                str(reference.get("reference_use_fingerprint") or ""),
            )
            reference["translation_unit_file"] = (
                translation_unit_file or None
            )
            selected.setdefault(identity, reference)
    return list(selected.values())


def _index_candidate_reports(
    index: CodeIndex,
    *,
    tree: str,
    relative_path: str,
    target_entity_ids: set[str],
    target_contracts: set[tuple[str, str]] | None = None,
    compile_source_files: list[str],
    preferred_translation_units: Iterable[object] = (),
) -> list[dict[str, Any]]:
    reports = [
        index.index_file(
            tree,
            relative_path,
            detailed_processing=True,
        )
    ]
    if Path(relative_path).suffix not in HEADER_SUFFIXES:
        return reports

    translation_units = ordered_unique_relpaths(
        [
            *list(preferred_translation_units),
            *compile_source_files,
        ]
    )
    for translation_unit in translation_units:
        if Path(translation_unit).suffix not in TRANSLATION_UNIT_SUFFIXES:
            continue
        report = index.index_file_from_translation_unit(
            tree,
            relative_path,
            translation_unit,
            detailed_processing=True,
        )
        reports.append(report)
        if _matching_references(
            [report],
            target_entity_ids=target_entity_ids,
            target_contracts=target_contracts,
        ):
            break
    return reports


def _reference_alternative(reference: dict[str, Any]) -> dict[str, Any]:
    owner = dict(reference.get("owner_entity") or {})
    target = dict(reference.get("referenced_entity") or {})
    return {
        "donor_reference_id": reference.get("id"),
        "source_file": normalize_relpath(
            str(reference.get("source_file") or "")
        ),
        "translation_unit_file": normalize_relpath(
            str(reference.get("translation_unit_file") or "")
        )
        or None,
        "line": reference.get("line"),
        "column": reference.get("column"),
        "cursor_kind": reference.get("cursor_kind"),
        "use_kind": reference.get("use_kind"),
        "reference_context_fingerprint": reference.get(
            "reference_context_fingerprint"
        ),
        "reference_use_contract": dict(
            reference.get("reference_use_contract") or {}
        ),
        "reference_use_fingerprint": reference.get(
            "reference_use_fingerprint"
        ),
        "owner_contract_key": _contract(owner),
        "owner_entity": _compact_entity(owner),
        "target_contract_key": _contract(target),
        "target_entity": _compact_entity(target),
    }


def discover_integration_obligations(
    spec: CaseSpec,
    *,
    seed_entities: Iterable[dict[str, Any]],
    validation_requirements: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Create one finite incoming-reachability obligation per new entity.

    This performs one graph step only. It never recursively materializes an
    owner. A later patch may introduce another donor-backed entity, which is
    registered as a new seed and processed once by stable contract identity.
    """

    seeds = [
        dict(seed)
        for seed in seed_entities
        if isinstance(seed, dict)
        and str(seed.get("contract_key") or "")
        and str(seed.get("symbol") or "")
        and normalize_relpath(str(seed.get("file") or ""))
    ]
    seeds_by_contract = {
        str(seed["contract_key"]): seed for seed in seeds
    }
    required_reference_ids_by_contract: dict[str, set[str]] = {}
    required_references_by_contract: dict[
        str, dict[str, dict[str, Any]]
    ] = {}
    for requirement in validation_requirements:
        if not isinstance(requirement, dict):
            continue
        contract_key = str(
            requirement.get("target_contract_key") or ""
        )
        if not contract_key:
            continue
        required_reference_ids_by_contract.setdefault(
            contract_key,
            set(),
        ).update(
            str(item)
            for item in requirement.get("consumer_reference_ids") or []
            if str(item)
        )
        contract_references = required_references_by_contract.setdefault(
            contract_key,
            {},
        )
        for reference in requirement.get("consumer_references") or []:
            if not isinstance(reference, dict):
                continue
            reference_id = str(reference.get("id") or "")
            referenced_entity = (
                reference.get("referenced_entity")
                if isinstance(
                    reference.get("referenced_entity"),
                    dict,
                )
                else {}
            )
            referenced_contract = str(
                referenced_entity.get("contract_key")
                or referenced_entity.get("key")
                or reference.get("contract_key")
                or ""
            )
            if (
                reference_id
                and referenced_contract == contract_key
            ):
                contract_references.setdefault(
                    reference_id,
                    dict(reference),
                )
    if not seeds_by_contract:
        return {
            "schema": "post_materialization_integration_report_v1",
            "status": "not_required",
            "obligations": [],
            "coverage_blockers": [],
            "summary": {
                "seed_count": 0,
                "obligation_count": 0,
                "coverage_blocker_count": 0,
            },
        }

    index = CodeIndex(spec, graph_projection="records")
    frontier = index.compile_dependency_frontier(
        "donor_0",
        within_root=spec.driver_root,
    )
    frontier_files = ordered_unique_relpaths(frontier.get("files") or [])
    compile_source_files = ordered_unique_relpaths(
        frontier.get("compile_source_files") or []
    )
    if str(frontier.get("status") or "") != "ready":
        return {
            "schema": "post_materialization_integration_report_v1",
            "status": "unavailable",
            "obligations": [],
            "coverage_blockers": [
                {
                    "reason": (
                        frontier.get("reason")
                        or "donor_compile_dependency_frontier_unavailable"
                    ),
                    "seed_contract_keys": sorted(seeds_by_contract),
                }
            ],
            "frontier": frontier,
            "summary": {
                "seed_count": len(seeds_by_contract),
                "obligation_count": 0,
                "coverage_blocker_count": 1,
            },
        }

    candidate_report = index.exact_token_candidates_in_files(
        "donor_0",
        [
            str(seed.get("symbol") or "")
            for seed in seeds_by_contract.values()
        ],
        frontier_files,
    )
    obligations: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    visited_contract_keys: list[str] = []
    structural_report_cache: dict[str, dict[str, Any]] = {}
    for contract_key in sorted(seeds_by_contract):
        seed = seeds_by_contract[contract_key]
        symbol = str(seed.get("symbol") or "")
        seed_file = normalize_relpath(str(seed.get("file") or ""))
        target_report = index.index_file(
            "donor_0",
            seed_file,
            detailed_processing=True,
        )
        contract_target_entities = [
            entity
            for entity in target_report.get("entities") or []
            if isinstance(entity, dict)
            and entity.get("is_definition") is True
            and _contract(entity) == contract_key
            and _entity_kind(entity) == str(seed.get("kind") or "")
        ]
        seed_entity_id = _entity_id(seed)
        exact_target_entities = [
            entity
            for entity in contract_target_entities
            if seed_entity_id
            and _entity_id(entity) == seed_entity_id
        ]
        target_entities = (
            exact_target_entities
            if len(exact_target_entities) == 1
            else contract_target_entities
        )
        structural_target_entities: list[dict[str, Any]] = []
        structural_target_proof = False
        if len(target_entities) != 1:
            structural_report = structural_report_cache.get(seed_file)
            if structural_report is None:
                structural_report = parse_structural_entity_file(
                    root=spec.primary_donor_root,
                    relative_path=seed_file,
                    tree="donor_0",
                )
                structural_report_cache[seed_file] = structural_report
            structural_target_entities = [
                entity
                for entity in structural_report.get("entities") or []
                if isinstance(entity, dict)
                and entity.get("is_definition") is True
                and _contract(entity) == contract_key
                and _entity_kind(entity) == str(seed.get("kind") or "")
            ]
            structural_target_proof = bool(
                structural_report.get("parse_ok")
                and len(structural_target_entities) == 1
                and seed_entity_id
            )
            if structural_target_proof:
                target_entities = [seed]
        if len(target_entities) != 1:
            blockers.append(
                {
                    "seed_contract_key": contract_key,
                    "seed_file": seed_file,
                    "reason": (
                        "donor_integration_seed_is_absent_or_ambiguous"
                    ),
                    "donor_match_count": len(target_entities),
                    "contract_match_count": len(
                        contract_target_entities
                    ),
                    "exact_seed_entity_id": seed_entity_id or None,
                    "exact_seed_match_count": len(
                        exact_target_entities
                    ),
                    "structural_match_count": len(
                        structural_target_entities
                    ),
                }
            )
            continue
        target_entity = target_entities[0]
        target_ids = {
            _entity_id(target_entity),
            *(
                {
                    _entity_id(structural_target_entities[0])
                }
                if structural_target_proof
                else set()
            ),
        }
        target_ids.discard("")
        target_contracts = {
            (
                _contract(target_entity),
                _entity_kind(target_entity),
            )
        }
        target_contracts.discard(("", ""))
        required_reference_ids = (
            required_reference_ids_by_contract.get(
                contract_key,
                set(),
            )
        )
        supplied_required_references = (
            required_references_by_contract.get(contract_key, {})
        )
        if (
            required_reference_ids
            and required_reference_ids.issubset(
                supplied_required_references
            )
        ):
            for reference_id in sorted(required_reference_ids):
                reference = supplied_required_references[reference_id]
                alternative = _reference_alternative(reference)
                owner_contract = str(
                    alternative.get("owner_contract_key") or ""
                )
                obligation_id = stable_id(
                    "post-materialization-integration-obligation",
                    contract_key,
                    owner_contract,
                    reference_id,
                )
                obligations.append(
                    {
                        "schema": (
                            "post_materialization_integration_"
                            "obligation_v1"
                        ),
                        "id": obligation_id,
                        "status": "pending_validation",
                        "target_contract_key": contract_key,
                        "target_symbol": symbol,
                        "target_kind": _entity_kind(target_entity),
                        "target_file": seed_file,
                        "target_entity": _compact_entity(
                            target_entity
                        ),
                        "selected_owner_contract_key": (
                            owner_contract
                        ),
                        "selected_owner_entity": _compact_entity(
                            reference.get("owner_entity") or {}
                        ),
                        "alternatives": [alternative],
                        "selection_basis": (
                            "persisted_explicit_validation_requirement"
                        ),
                    }
                )
            visited_contract_keys.append(contract_key)
            continue

        candidate_files = ordered_unique_relpaths(
            (
                candidate_report.get("candidates_by_symbol") or {}
            ).get(symbol)
            or []
        )
        reports: list[dict[str, Any]] = []
        for candidate_file in sorted(
            candidate_files,
            key=lambda path: (
                Path(path).suffix in HEADER_SUFFIXES,
                path,
            ),
        ):
            reports.extend(
                _index_candidate_reports(
                    index,
                    tree="donor_0",
                    relative_path=candidate_file,
                    target_entity_ids=target_ids,
                    target_contracts=target_contracts,
                    compile_source_files=compile_source_files,
                    preferred_translation_units=[seed_file],
                )
            )
        incoming = _matching_references(
            reports,
            target_entity_ids=target_ids,
            target_contracts=target_contracts,
        )
        if not incoming:
            if required_reference_ids:
                blockers.append(
                    {
                        "seed_contract_key": contract_key,
                        "seed_file": seed_file,
                        "reason": (
                            "required_donor_consumer_references_"
                            "not_observed"
                        ),
                        "missing_consumer_reference_ids": sorted(
                            required_reference_ids
                        ),
                        "candidate_files": candidate_files,
                    }
                )
                continue
            requires_local_incoming = bool(
                _entity_kind(target_entity) == "field"
                or str(
                    target_entity.get("storage_class") or ""
                ).upper()
                == "STATIC"
                or str(target_entity.get("linkage") or "").upper()
                == "INTERNAL"
            )
            if requires_local_incoming:
                blockers.append(
                    {
                        "seed_contract_key": contract_key,
                        "seed_file": seed_file,
                        "reason": (
                            "internal_donor_entity_has_no_typed_"
                            "incoming_reference"
                        ),
                        "candidate_files": candidate_files,
                    }
                )
            continue

        if required_reference_ids:
            incoming_by_id = {
                str(reference.get("id") or ""): reference
                for reference in incoming
                if str(reference.get("id") or "")
            }
            missing_reference_ids = (
                required_reference_ids - set(incoming_by_id)
            )
            if missing_reference_ids:
                blockers.append(
                    {
                        "seed_contract_key": contract_key,
                        "seed_file": seed_file,
                        "reason": (
                            "required_donor_consumer_references_"
                            "not_observed"
                        ),
                        "missing_consumer_reference_ids": sorted(
                            missing_reference_ids
                        ),
                        "candidate_files": candidate_files,
                    }
                )
            for reference_id in sorted(
                required_reference_ids & set(incoming_by_id)
            ):
                reference = incoming_by_id[reference_id]
                alternative = _reference_alternative(reference)
                owner_contract = str(
                    alternative.get("owner_contract_key") or ""
                )
                obligation_id = stable_id(
                    "post-materialization-integration-obligation",
                    contract_key,
                    owner_contract,
                    reference_id,
                )
                obligations.append(
                    {
                        "schema": (
                            "post_materialization_integration_"
                            "obligation_v1"
                        ),
                        "id": obligation_id,
                        "status": "pending_validation",
                        "target_contract_key": contract_key,
                        "target_symbol": symbol,
                        "target_kind": _entity_kind(target_entity),
                        "target_file": seed_file,
                        "target_entity": _compact_entity(
                            target_entity
                        ),
                        "selected_owner_contract_key": (
                            owner_contract
                        ),
                        "selected_owner_entity": _compact_entity(
                            reference.get("owner_entity") or {}
                        ),
                        "alternatives": [alternative],
                        "selection_basis": (
                            "explicit_validation_requirement"
                        ),
                    }
                )
            visited_contract_keys.append(contract_key)
            continue

        executable = [
            reference
            for reference in incoming
            if _entity_kind(reference.get("owner_entity"))
            == "function"
        ]
        eligible = (
            executable
            if _entity_kind(target_entity) == "field" and executable
            else incoming
        )
        eligible.sort(
            key=lambda reference: (
                normalize_relpath(
                    str(reference.get("source_file") or "")
                ),
                _contract(reference.get("owner_entity")),
                int(reference.get("line") or 0),
                str(reference.get("use_kind") or ""),
            )
        )
        selected_owner_contract = _contract(
            eligible[0].get("owner_entity")
        )
        alternatives = [
            _reference_alternative(reference)
            for reference in eligible
            if _contract(reference.get("owner_entity"))
            == selected_owner_contract
        ]
        obligation_id = stable_id(
            "post-materialization-integration-obligation",
            contract_key,
            selected_owner_contract,
            *sorted(
                str(
                    alternative.get(
                        "reference_context_fingerprint"
                    )
                    or alternative.get("donor_reference_id")
                    or ""
                )
                for alternative in alternatives
            ),
        )
        obligations.append(
            {
                "schema": (
                    "post_materialization_integration_obligation_v1"
                ),
                "id": obligation_id,
                "status": "pending_validation",
                "target_contract_key": contract_key,
                "target_symbol": symbol,
                "target_kind": _entity_kind(target_entity),
                "target_file": seed_file,
                "target_entity": _compact_entity(target_entity),
                "selected_owner_contract_key": (
                    selected_owner_contract
                ),
                "selected_owner_entity": _compact_entity(
                    eligible[0].get("owner_entity") or {}
                ),
                "alternatives": alternatives,
                "selection_basis": (
                    "typed_executable_consumer"
                    if executable
                    and _entity_kind(target_entity) == "field"
                    else "typed_incoming_reference"
                ),
            }
        )
        visited_contract_keys.append(contract_key)

    report = {
        "schema": "post_materialization_integration_report_v1",
        "status": (
            "ready"
            if obligations and not blockers
            else "partially_available"
            if obligations
            else "unavailable"
        ),
        "obligations": obligations,
        "coverage_blockers": blockers,
        "frontier": {
            "status": frontier.get("status"),
            "scope_root": frontier.get("scope_root"),
            "compile_source_file_count": len(compile_source_files),
            "file_count": len(frontier_files),
        },
        "termination": {
            "kind": "finite_one_step_worklist",
            "seed_contract_keys": sorted(seeds_by_contract),
            "visited_contract_keys": sorted(
                set(visited_contract_keys)
            ),
            "deduplication_key": "stable_contract_and_reference_identity",
            "recursive_materialization": False,
        },
        "summary": {
            "seed_count": len(seeds_by_contract),
            "obligation_count": len(obligations),
            "coverage_blocker_count": len(blockers),
        },
    }
    validation = revalidate_integration_obligations(
        spec,
        obligations=obligations,
    )
    report["obligations"] = validation["obligations"]
    report["validation"] = {
        key: validation.get(key)
        for key in ("status", "summary")
    }
    return report


def _reference_matches_alternative(
    reference: dict[str, Any],
    alternative: dict[str, Any],
) -> bool:
    owner = reference.get("owner_entity")
    target = reference.get("referenced_entity")
    if not isinstance(owner, dict) or not isinstance(target, dict):
        return False
    if (
        _contract(owner)
        != str(alternative.get("owner_contract_key") or "")
        or _contract(target)
        != str(alternative.get("target_contract_key") or "")
        or str(reference.get("cursor_kind") or "")
        != str(alternative.get("cursor_kind") or "")
        or str(reference.get("use_kind") or "")
        != str(alternative.get("use_kind") or "")
    ):
        return False
    use_contract = (
        reference.get("reference_use_contract")
        if isinstance(reference.get("reference_use_contract"), dict)
        else {}
    )
    if (
        str(reference.get("use_kind") or "") == "call"
        and use_contract.get("arity_compatible") is False
    ):
        return False
    return True


def revalidate_integration_obligations(
    spec: CaseSpec,
    *,
    obligations: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Validate exact incoming edges in the current old tree."""

    records = [
        dict(obligation)
        for obligation in obligations
        if isinstance(obligation, dict)
        and str(obligation.get("id") or "")
    ]
    if not records:
        return {
            "schema": "integration_obligation_validation_v1",
            "status": "not_required",
            "obligations": [],
            "summary": {
                "obligation_count": 0,
                "satisfied_count": 0,
                "unresolved_count": 0,
            },
        }
    index = CodeIndex(spec, graph_projection="records")
    frontier = index.compile_dependency_frontier(
        "old",
        within_root=spec.driver_root,
    )
    compile_source_files = ordered_unique_relpaths(
        frontier.get("compile_source_files") or []
    )
    reports_by_file: dict[
        tuple[str, tuple[str, ...]], list[dict[str, Any]]
    ] = {}
    validated: list[dict[str, Any]] = []
    for obligation in records:
        matches: list[dict[str, Any]] = []
        alternatives = [
            dict(item)
            for item in obligation.get("alternatives") or []
            if isinstance(item, dict)
        ]
        for alternative in alternatives:
            source_file = normalize_relpath(
                str(alternative.get("source_file") or "")
            )
            if not source_file or not spec.resolve_old(source_file).is_file():
                continue
            target_contract = str(
                alternative.get("target_contract_key") or ""
            )
            owner_contract = str(
                alternative.get("owner_contract_key") or ""
            )
            cache_key = (
                source_file,
                (target_contract, owner_contract),
            )
            reports = reports_by_file.get(cache_key)
            if reports is None:
                target_report = index.index_file(
                    "old",
                    source_file,
                    detailed_processing=True,
                )
                target_ids = {
                    _entity_id(entity)
                    for entity in target_report.get("entities") or []
                    if isinstance(entity, dict)
                    and _contract(entity) == target_contract
                }
                reports = [target_report]
                if (
                    Path(source_file).suffix in HEADER_SUFFIXES
                    and target_ids
                ):
                    reports = _index_candidate_reports(
                        index,
                        tree="old",
                        relative_path=source_file,
                        target_entity_ids=target_ids,
                        compile_source_files=compile_source_files,
                        preferred_translation_units=[
                            alternative.get(
                                "translation_unit_file"
                            )
                        ],
                    )
                reports_by_file[cache_key] = reports
            for report in reports:
                for reference in report.get("references") or []:
                    if (
                        isinstance(reference, dict)
                        and _reference_matches_alternative(
                            reference,
                            alternative,
                        )
                    ):
                        matches.append(
                            {
                                "reference_id": reference.get("id"),
                                "source_file": source_file,
                                "line": reference.get("line"),
                                "column": reference.get("column"),
                                "owner_contract_key": owner_contract,
                                "target_contract_key": target_contract,
                                "reference_context_fingerprint": (
                                    reference.get(
                                        "reference_context_fingerprint"
                                    )
                                ),
                                "reference_use_fingerprint": (
                                    reference.get(
                                        "reference_use_fingerprint"
                                    )
                                ),
                            }
                        )
            if matches:
                break
        status = "satisfied" if matches else "unresolved"
        validated.append(
            {
                **obligation,
                "status": status,
                "current_matches": matches,
                "validation_reason": (
                    "one typed owner-to-target incoming-edge alternative "
                    "exists and is structurally compatible with the current "
                    "provider contract in the old-tree code graph"
                    if matches
                    else (
                        "no structurally compatible incoming-edge "
                        "alternative exists in the current old-tree code "
                        "graph"
                    )
                ),
            }
        )
    satisfied_count = sum(
        item.get("status") == "satisfied" for item in validated
    )
    unresolved_count = len(validated) - satisfied_count
    return {
        "schema": "integration_obligation_validation_v1",
        "status": (
            "satisfied"
            if validated and not unresolved_count
            else "unresolved"
        ),
        "obligations": validated,
        "summary": {
            "obligation_count": len(validated),
            "satisfied_count": satisfied_count,
            "unresolved_count": unresolved_count,
        },
    }


def enforce_post_materialization_validation_requirement(
    report: dict[str, Any],
    validation_requirement: dict[str, Any] | None,
) -> dict[str, Any]:
    requirement = (
        dict(validation_requirement)
        if isinstance(validation_requirement, dict)
        else {}
    )
    result = dict(report)
    result["validation_requirement"] = requirement or None
    required_dimensions = {
        str(item)
        for item in requirement.get("required_dimensions") or []
        if str(item)
    }
    if "integration_reference" not in required_dimensions:
        return result

    target_contract_key = str(
        requirement.get("target_contract_key") or ""
    )
    target_obligations = [
        item
        for item in result.get("obligations") or []
        if isinstance(item, dict)
        and (
            not target_contract_key
            or str(item.get("target_contract_key") or "")
            == target_contract_key
        )
    ]
    relevant_blockers = [
        item
        for item in result.get("coverage_blockers") or []
        if isinstance(item, dict)
        and (
            not target_contract_key
            or not str(item.get("seed_contract_key") or "")
            or str(item.get("seed_contract_key") or "")
            == target_contract_key
        )
        and (
            not target_contract_key
            or not item.get("seed_contract_keys")
            or target_contract_key
            in {
                str(value)
                for value in item.get("seed_contract_keys") or []
                if str(value)
            }
        )
    ]
    required_reference_ids = {
        str(item)
        for item in requirement.get("consumer_reference_ids") or []
        if str(item)
    }
    observed_reference_ids = {
        str(alternative.get("donor_reference_id") or "")
        for obligation in target_obligations
        for alternative in obligation.get("alternatives") or []
        if isinstance(alternative, dict)
        and str(alternative.get("donor_reference_id") or "")
    }
    missing_reference_ids = (
        required_reference_ids - observed_reference_ids
    )
    if relevant_blockers or (
        target_obligations and not missing_reference_ids
    ):
        return result

    result["status"] = "unavailable"
    result["coverage_blockers"] = [
        *[
            dict(item)
            for item in result.get("coverage_blockers") or []
            if isinstance(item, dict)
        ],
        {
            "reason": (
                "required_post_materialization_consumer_edges_absent"
                if missing_reference_ids
                else (
                    "required_post_materialization_integration_"
                    "evidence_absent"
                )
            ),
            "seed_contract_key": target_contract_key or None,
            "consumer_reference_ids": list(
                requirement.get("consumer_reference_ids") or []
            ),
            "missing_consumer_reference_ids": sorted(
                missing_reference_ids
            ),
        },
    ]
    return result


def analyze_post_patch_integration(
    spec: CaseSpec,
    *,
    transaction_dir: Path,
    touched_files: Iterable[object],
    validation_requirements: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    normalized_touched_files = ordered_unique_relpaths(touched_files)
    requirements_by_id: dict[str, dict[str, Any]] = {}
    for raw_requirement in validation_requirements:
        if not isinstance(raw_requirement, dict):
            continue
        requirement = dict(raw_requirement)
        requirement_id = str(
            requirement.get("id")
            or stable_id(
                "post-patch-integration-requirement",
                str(requirement.get("target_contract_key") or ""),
                *[
                    str(item)
                    for item in requirement.get(
                        "consumer_reference_ids"
                    )
                    or []
                    if str(item)
                ],
            )
        )
        requirement["id"] = requirement_id
        requirements_by_id[requirement_id] = requirement
    requirements = [
        requirements_by_id[key] for key in sorted(requirements_by_id)
    ]
    additions = structural_entity_additions_from_transaction(
        spec,
        transaction_dir=transaction_dir,
        touched_files=normalized_touched_files,
    )
    seeds_by_contract = {
        str(seed.get("contract_key") or ""): dict(seed)
        for seed in additions.get("seed_entities") or []
        if isinstance(seed, dict)
        and str(seed.get("contract_key") or "")
    }
    for requirement in requirements:
        target = (
            dict(requirement.get("target_entity") or {})
            if isinstance(requirement.get("target_entity"), dict)
            else {}
        )
        contract_key = str(
            requirement.get("target_contract_key")
            or _contract(target)
            or ""
        )
        symbol = str(
            requirement.get("target_symbol")
            or target.get("symbol")
            or ""
        )
        kind = str(
            requirement.get("target_kind")
            or _entity_kind(target)
            or ""
        )
        relative_path = normalize_relpath(
            str(
                requirement.get("target_file")
                or target.get("file")
                or ""
            )
        )
        if not contract_key or not symbol or not kind or not relative_path:
            continue
        seeds_by_contract[contract_key] = {
            "schema": "post_patch_integration_seed_v1",
            "change_kind": "existing_contract_transaction",
            "contract_key": contract_key,
            "symbol": symbol,
            "kind": kind,
            "file": relative_path,
            "current_entity": {},
            "donor_entity": _compact_entity(target),
            **_compact_entity(target),
        }
    integration = discover_integration_obligations(
        spec,
        seed_entities=[
            seeds_by_contract[key] for key in sorted(seeds_by_contract)
        ],
        validation_requirements=requirements,
    )
    required_contracts = {
        str(item.get("target_contract_key") or "")
        for item in requirements
        if str(item.get("target_contract_key") or "")
    }
    required_reference_ids = {
        str(reference_id)
        for item in requirements
        for reference_id in item.get("consumer_reference_ids") or []
        if str(reference_id)
    }
    required_obligations = [
        dict(item)
        for item in integration.get("obligations") or []
        if isinstance(item, dict)
        and str(item.get("target_contract_key") or "")
        in required_contracts
    ]
    observed_reference_ids = {
        str(alternative.get("donor_reference_id") or "")
        for obligation in required_obligations
        for alternative in obligation.get("alternatives") or []
        if isinstance(alternative, dict)
        and str(alternative.get("donor_reference_id") or "")
    }
    unresolved_required = [
        item
        for item in required_obligations
        if str(item.get("status") or "") != "satisfied"
    ]
    required_blockers = [
        dict(item)
        for item in integration.get("coverage_blockers") or []
        if isinstance(item, dict)
        and (
            str(item.get("seed_contract_key") or "")
            in required_contracts
            or required_contracts.intersection(
                {
                    str(value)
                    for value in item.get("seed_contract_keys") or []
                    if str(value)
                }
            )
        )
    ]
    missing_required_reference_ids = sorted(
        required_reference_ids - observed_reference_ids
    )
    atomic_transaction_required = bool(requirements)
    atomic_transaction_complete = bool(
        not atomic_transaction_required
        or (
            not unresolved_required
            and not required_blockers
            and not missing_required_reference_ids
        )
    )
    report = {
        "schema": "post_patch_integration_analysis_v1",
        "analysis_revision": INTEGRATION_ANALYSIS_REVISION,
        "status": integration.get("status"),
        "revalidation_request": {
            "schema": "post_patch_integration_revalidation_request_v1",
            "transaction_dir": str(transaction_dir.resolve()),
            "touched_files": normalized_touched_files,
            "validation_requirements": requirements,
        },
        "structural_additions": additions,
        "obligations": list(integration.get("obligations") or []),
        "coverage_blockers": [
            *list(additions.get("parse_blockers") or []),
            *[
                {
                    **dict(item),
                    "seed_contract_key": item.get("contract_key"),
                }
                for item in additions.get("unmatched_additions") or []
                if isinstance(item, dict)
            ],
            *list(integration.get("coverage_blockers") or []),
        ],
        "frontier": integration.get("frontier"),
        "termination": integration.get("termination"),
        "atomic_transaction_required": (
            atomic_transaction_required
        ),
        "atomic_transaction_complete": (
            atomic_transaction_complete
        ),
        "required_integration": {
            "requirement_ids": sorted(requirements_by_id),
            "target_contract_keys": sorted(required_contracts),
            "consumer_reference_ids": sorted(
                required_reference_ids
            ),
            "observed_consumer_reference_ids": sorted(
                observed_reference_ids
            ),
            "missing_consumer_reference_ids": (
                missing_required_reference_ids
            ),
            "unresolved_obligation_ids": sorted(
                str(item.get("id") or "")
                for item in unresolved_required
                if str(item.get("id") or "")
            ),
            "coverage_blockers": required_blockers,
        },
        "summary": {
            "seed_count": int(
                len(seeds_by_contract)
            ),
            "required_seed_count": len(requirements),
            "obligation_count": len(
                integration.get("obligations") or []
            ),
            "unresolved_count": sum(
                str(item.get("status") or "") == "unresolved"
                for item in integration.get("obligations") or []
                if isinstance(item, dict)
            ),
            "coverage_blocker_count": len(
                [
                    *list(additions.get("parse_blockers") or []),
                    *list(
                        integration.get("coverage_blockers")
                        or []
                    ),
                ]
            ),
            "required_unresolved_count": len(
                unresolved_required
            ),
            "required_coverage_blocker_count": len(
                required_blockers
            ),
            "missing_required_reference_count": len(
                missing_required_reference_ids
            ),
        },
    }
    return report
