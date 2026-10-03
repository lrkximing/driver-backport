from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from driver_backport.analysis.code_index import CodeIndex
from driver_backport.analysis.cross_version_contract import (
    canonical_semantic_requirement,
    reference_is_source_contract,
)
from driver_backport.analysis.entity_fingerprint import (
    compare_entity_fingerprints,
    entity_is_internally_linked,
)
from driver_backport.analysis.entity_resolution import (
    case_aware_entity_report,
    entity_contract_key,
    entity_kind,
)
from driver_backport.analysis.export_index import (
    export_contract_relation,
    index_export_contracts,
    module_ownership_for_sources,
)
from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_strings as _ordered_strings,
    utc_now,
)
from driver_backport.core.executor import run_unified_diff
from driver_backport.migration.semantic_closure_state import (
    CLOSED_MODIFIED,
    PENDING_DECISION,
    activate_semantic_gap_snapshot,
    record_compile_valid_handoff_evaluations,
    semantic_closure_summary,
)
from driver_backport.migration.semantic_closure_store import (
    assert_semantic_source_artifacts_integrity,
    load_semantic_closure_state,
    load_semantic_record,
    put_semantic_record,
    write_semantic_closure_state,
)
COMPILE_VALID_HANDOFF_SCHEMA = "compile_valid_semantic_handoff_v1"
COMPILE_VALID_HANDOFF_KEY = "compile_valid_semantic_handoff"
COMPILE_VALID_HANDOFF_EVALUATOR_REVISION = (
    "demand-facet-current-tree-proof-v6"
)

_SOURCE_SUFFIXES = {".c", ".cc", ".cpp"}
_SATISFIED_RELATIONS = {
    "consumer_compile_binding": {"compile_valid"},
    "name": {"equivalent", "not_required"},
    "signature": {"equivalent", "not_required"},
    "implementation": {"equivalent", "not_required"},
    "interface": {"equivalent", "not_required"},
    "declaration": {"equivalent", "not_required"},
    "structural_source": {"equivalent", "not_required"},
    "export": {"equivalent", "not_required"},
    "module_boundary": {
        "same_module",
        "cross_module",
        "not_required",
    },
    "integration_reference": {"equivalent", "not_required"},
}
_DIFFERENCE_RELATIONS = {
    "different",
    "missing_in_old",
    "target_alternative_review_required",
    "required_missing",
    "donor_only",
    "old_only",
}
def _compact_entity(entity: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(entity, dict):
        return None
    return {
        key: deepcopy(entity.get(key))
        for key in (
            "id",
            "tree",
            "file",
            "contract_key",
            "key",
            "kind",
            "symbol",
            "is_definition",
            "extent",
            "signature_fingerprint",
            "implementation_fingerprint",
            "interface_fingerprint",
            "syntax_fingerprint",
            "source_fingerprint",
            "digest",
            "interface",
            "storage_class",
            "linkage",
            "conditional_context",
        )
        if entity.get(key) not in (None, "", [], {})
    }


def _relation_from_candidates(
    comparisons: Iterable[dict[str, Any]],
    dimension: str,
) -> str:
    values = [
        str(comparison.get(dimension) or "unavailable")
        for comparison in comparisons
    ]
    if "equivalent" in values:
        return "equivalent"
    if values and all(value == "different" for value in values):
        return "different"
    return "unavailable"


def _exact_source_contract_equivalent(
    current: dict[str, Any],
    donor: dict[str, Any],
) -> bool:
    if (
        entity_contract_key(current) != entity_contract_key(donor)
        or entity_kind(current) != entity_kind(donor)
    ):
        return False
    current_source = str(
        current.get("syntax_fingerprint")
        or current.get("source_fingerprint")
        or current.get("digest")
        or ""
    )
    donor_source = str(
        donor.get("syntax_fingerprint")
        or donor.get("source_fingerprint")
        or donor.get("digest")
        or ""
    )
    return bool(current_source and current_source == donor_source)


def _descriptor_matches(
    entity: dict[str, Any],
    descriptor: dict[str, Any],
    *,
    definition: bool | None,
) -> bool:
    contract = entity_contract_key(descriptor)
    kind = entity_kind(descriptor)
    symbol = str(
        descriptor.get("symbol") or descriptor.get("name") or ""
    )
    if contract and entity_contract_key(entity) != contract:
        return False
    if kind and entity_kind(entity) != kind:
        return False
    if symbol and str(
        entity.get("symbol") or entity.get("name") or ""
    ) != symbol:
        return False
    descriptor_context = tuple(
        str(item)
        for item in descriptor.get("conditional_context") or []
    )
    if descriptor_context and tuple(
        str(item)
        for item in entity.get("conditional_context") or []
    ) != descriptor_context:
        return False
    if definition is not None and bool(
        entity.get("is_definition")
    ) is not definition:
        return False
    return True


class ExactCurrentTreeEvidence:
    """Collect bounded fresh evidence from both case-specific code trees."""

    def __init__(self, spec: object) -> None:
        self.spec = spec
        self.index = CodeIndex(
            spec,
            graph_projection="records",
        )
        self.report_cache: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        self.module_cache: dict[
            tuple[str, str], list[dict[str, Any]]
        ] = {}
        self.export_cache: dict[
            tuple[str, str, str], list[dict[str, Any]]
        ] = {}
        self.reference_cache: dict[
            tuple[str, str, tuple[str, ...]], dict[str, Any]
        ] = {}

    def entity_candidates(
        self,
        descriptor: dict[str, Any],
        *,
        tree: str,
        definition: bool | None,
    ) -> dict[str, Any]:
        relative_path = normalize_relpath(
            str(
                descriptor.get("file")
                or descriptor.get("path")
                or ""
            )
        )
        if not relative_path:
            return {
                "status": "descriptor_has_no_file",
                "file": "",
                "candidates": [],
            }
        report = case_aware_entity_report(
            self.spec,
            relative_path=relative_path,
            tree=tree,
            index=self.index,
            report_cache=self.report_cache,
        )
        candidates = [
            dict(entity)
            for entity in report.get("entities") or []
            if isinstance(entity, dict)
            and _descriptor_matches(
                entity,
                descriptor,
                definition=definition,
            )
        ]
        if candidates:
            status = "resolved" if len(candidates) == 1 else "resolved_multiple"
        elif not report.get("parse_ok"):
            status = "parse_unavailable"
        elif report.get("owner_presence_authoritative"):
            status = "entity_absent"
        else:
            status = "absence_unproven"
        return {
            "status": status,
            "file": relative_path,
            "backend": report.get("backend"),
            "compile_aware": bool(report.get("compile_aware")),
            "parse_error": report.get("parse_error"),
            "candidates": candidates,
        }

    def module_owners(
        self,
        tree: str,
        relative_path: str,
    ) -> list[dict[str, Any]]:
        relative_path = normalize_relpath(relative_path)
        normalized_tree = (
            "donor_0" if tree in {"donor", "donor_0"} else "old"
        )
        key = (normalized_tree, relative_path)
        if key not in self.module_cache:
            root = (
                Path(getattr(self.spec, "primary_donor_root"))
                if normalized_tree == "donor_0"
                else Path(getattr(self.spec, "old_root_path"))
            )
            result = module_ownership_for_sources(
                root,
                [relative_path],
                tree=normalized_tree,
            )
            self.module_cache[key] = list(
                result.get(relative_path) or []
            )
        return list(self.module_cache[key])

    def exports(
        self,
        tree: str,
        relative_path: str,
        symbol: str,
    ) -> list[dict[str, Any]]:
        relative_path = normalize_relpath(relative_path)
        normalized_tree = (
            "donor_0" if tree in {"donor", "donor_0"} else "old"
        )
        key = (normalized_tree, relative_path, str(symbol))
        if key not in self.export_cache:
            root = (
                Path(getattr(self.spec, "primary_donor_root"))
                if normalized_tree == "donor_0"
                else Path(getattr(self.spec, "old_root_path"))
            )
            indexed = index_export_contracts(
                root,
                [relative_path],
                tree=normalized_tree,
            )
            for indexed_symbol, records in indexed.items():
                self.export_cache[
                    (
                        normalized_tree,
                        relative_path,
                        str(indexed_symbol),
                    )
                ] = list(records)
            self.export_cache.setdefault(key, [])
        return list(self.export_cache[key])

    def references(
        self,
        *,
        tree: str,
        relative_path: str,
        translation_units: Iterable[object],
    ) -> dict[str, Any]:
        relative_path = normalize_relpath(relative_path)
        normalized_tree = (
            "donor_0" if tree in {"donor", "donor_0"} else "old"
        )
        selected_translation_units = tuple(
            translation_unit
            for translation_unit in _ordered_strings(translation_units)
            if Path(translation_unit).suffix.lower()
            in _SOURCE_SUFFIXES
        )
        cache_key = (
            normalized_tree,
            relative_path,
            selected_translation_units,
        )
        if cache_key in self.reference_cache:
            return deepcopy(self.reference_cache[cache_key])
        reports = [
            self.index.index_file(
                normalized_tree,
                relative_path,
                detailed_processing=True,
            )
        ]
        for translation_unit in selected_translation_units:
            reports.append(
                self.index.index_file_from_translation_unit(
                    normalized_tree,
                    relative_path,
                    translation_unit,
                    detailed_processing=True,
                )
            )
        parse_ok = any(report.get("parse_ok") for report in reports)
        references = [
            dict(reference)
            for reference in self.index.references_by_file.get(
                (normalized_tree, relative_path),
                [],
            )
            if isinstance(reference, dict)
        ]
        result = {
            "status": "resolved" if parse_ok else "parse_unavailable",
            "file": relative_path,
            "tree": normalized_tree,
            "parse_reports": [
                {
                    "translation_unit_file": report.get(
                        "translation_unit_file"
                    ),
                    "parse_ok": bool(report.get("parse_ok")),
                    "parse_error": report.get("parse_error"),
                }
                for report in reports
            ],
            "references": references,
        }
        self.reference_cache[cache_key] = deepcopy(result)
        return result


def _current_entity_relations(
    collector: ExactCurrentTreeEvidence,
    evidence: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    donor_locator = (
        evidence.get("donor_entity")
        if isinstance(evidence.get("donor_entity"), dict)
        else {}
    )
    if not donor_locator:
        return {}, {
            "status": "donor_entity_unavailable",
            "current_entities": [],
            "donor_entities": [],
        }
    definition = (
        True if donor_locator.get("is_definition") is True else None
    )
    donor_resolution = collector.entity_candidates(
        donor_locator,
        tree="donor_0",
        definition=definition,
    )
    donor_candidates = list(
        donor_resolution.get("candidates") or []
    )
    if not donor_candidates:
        return {
            "name": "unavailable",
            "signature": "unavailable",
            "implementation": "unavailable",
            "interface": "unavailable",
            "structural_source": "unavailable",
        }, {
            "status": "fresh_donor_entity_unavailable",
            "donor_resolution": {
                key: value
                for key, value in donor_resolution.items()
                if key != "candidates"
            },
            "current_entities": [],
            "donor_entities": [],
        }

    current_resolutions = [
        collector.entity_candidates(
            donor,
            tree="old",
            definition=definition,
        )
        for donor in donor_candidates
    ]
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for resolution in current_resolutions:
        for candidate in resolution.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            identity = str(
                candidate.get("id")
                or candidate.get("entity_id")
                or repr(candidate)
            )
            candidates_by_id.setdefault(identity, candidate)
    candidates = list(candidates_by_id.values())
    if not candidates:
        authoritative_absence = bool(current_resolutions) and all(
            resolution.get("status") == "entity_absent"
            for resolution in current_resolutions
        )
        relation = (
            "missing_in_old"
            if authoritative_absence
            else "unavailable"
        )
        return {
            "name": relation,
            "signature": "unavailable",
            "implementation": "unavailable",
            "interface": "unavailable",
            "structural_source": "unavailable",
        }, {
            "status": (
                "entity_absent"
                if authoritative_absence
                else "current_entity_unavailable"
            ),
            "donor_resolution": {
                key: value
                for key, value in donor_resolution.items()
                if key != "candidates"
            },
            "current_resolutions": [
                {
                    key: value
                    for key, value in resolution.items()
                    if key != "candidates"
                }
                for resolution in current_resolutions
            ],
            "current_entities": [],
            "donor_entities": [
                _compact_entity(candidate)
                for candidate in donor_candidates
            ],
        }

    comparisons = [
        compare_entity_fingerprints(candidate, donor)
        for candidate in candidates
        for donor in donor_candidates
    ]
    relations = {
        "name": "equivalent",
        "signature": _relation_from_candidates(
            comparisons,
            "signature",
        ),
        "implementation": _relation_from_candidates(
            comparisons,
            "implementation",
        ),
        "interface": _relation_from_candidates(
            comparisons,
            "interface",
        ),
        "structural_source": _relation_from_candidates(
            comparisons,
            "exact_definition_source",
        ),
    }
    return relations, {
        "status": (
            "resolved"
            if len(candidates) == 1 and len(donor_candidates) == 1
            else "resolved_multiple"
        ),
        "donor_resolution": {
            key: value
            for key, value in donor_resolution.items()
            if key != "candidates"
        },
        "current_resolutions": [
            {
                key: value
                for key, value in resolution.items()
                if key != "candidates"
            }
            for resolution in current_resolutions
        ],
        "current_entities": [
            _compact_entity(candidate) for candidate in candidates
        ],
        "donor_entities": [
            _compact_entity(candidate)
            for candidate in donor_candidates
        ],
        "fingerprint_comparisons": comparisons,
    }


def _declaration_relation(
    collector: ExactCurrentTreeEvidence,
    evidence: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    if str(evidence.get("dependency_scope") or "") == "kernel_tree":
        return "not_required", []
    donor_declarations = [
        declaration
        for declaration in evidence.get("donor_declarations") or []
        if isinstance(declaration, dict)
    ]
    if not donor_declarations:
        return "not_required", []

    observations: list[dict[str, Any]] = []
    any_unavailable = False
    any_missing = False
    for donor_locator in donor_declarations:
        donor_resolution = collector.entity_candidates(
            donor_locator,
            tree="donor_0",
            definition=None,
        )
        donor_candidates = list(
            donor_resolution.get("candidates") or []
        )
        if not donor_candidates:
            any_unavailable = True
            observations.append(
                {
                    "donor_declaration_locator": _compact_entity(
                        donor_locator
                    ),
                    "donor_resolution_status": (
                        donor_resolution.get("status")
                    ),
                    "donor_declarations": [],
                    "current_declarations": [],
                    "fingerprint_comparisons": [],
                    "matched": False,
                }
            )
            continue
        current_resolutions = [
            collector.entity_candidates(
                donor,
                tree="old",
                definition=None,
            )
            for donor in donor_candidates
        ]
        candidates = [
            candidate
            for resolution in current_resolutions
            for candidate in resolution.get("candidates") or []
            if isinstance(candidate, dict)
        ]
        comparisons = [
            compare_entity_fingerprints(candidate, donor)
            for candidate in candidates
            for donor in donor_candidates
        ]
        matched = any(
            _exact_source_contract_equivalent(candidate, donor)
            for candidate in candidates
            for donor in donor_candidates
        ) or any(
            (
                comparison.get("exact_definition_source")
                == "equivalent"
            )
            or (
                comparison.get("signature") == "equivalent"
                and comparison.get("interface")
                in {"equivalent", "unavailable"}
            )
            for comparison in comparisons
        )
        if not matched:
            if current_resolutions and all(
                resolution.get("status") == "entity_absent"
                for resolution in current_resolutions
            ):
                any_missing = True
            else:
                any_unavailable = True
        observations.append(
            {
                "donor_declaration_locator": _compact_entity(
                    donor_locator
                ),
                "donor_resolution_status": donor_resolution.get(
                    "status"
                ),
                "donor_declarations": [
                    _compact_entity(candidate)
                    for candidate in donor_candidates
                ],
                "current_resolution_statuses": [
                    resolution.get("status")
                    for resolution in current_resolutions
                ],
                "current_declarations": [
                    _compact_entity(candidate)
                    for candidate in candidates
                ],
                "fingerprint_comparisons": comparisons,
                "matched": matched,
            }
        )
    if any_missing:
        return "missing_in_old", observations
    if any_unavailable:
        return "unavailable", observations
    return "equivalent", observations


def _export_shape(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("macro") or ""),
        str(record.get("namespace_or_modules") or ""),
    )


def _export_relation(
    collector: ExactCurrentTreeEvidence,
    evidence: dict[str, Any],
    current_entities: Iterable[dict[str, Any]],
    donor_entities: Iterable[dict[str, Any]],
    *,
    module_boundary: str,
) -> tuple[str, dict[str, Any]]:
    symbol = str(evidence.get("symbol") or "")
    current_provider_files = _ordered_strings(
        entity.get("file")
        for entity in current_entities
        if isinstance(entity, dict)
    )
    donor_provider_files = _ordered_strings(
        entity.get("file")
        for entity in donor_entities
        if isinstance(entity, dict)
    )
    current_exports = [
        record
        for relative_path in current_provider_files
        for record in collector.exports("old", relative_path, symbol)
    ]
    donor_exports = [
        record
        for relative_path in donor_provider_files
        for record in collector.exports(
            "donor_0",
            relative_path,
            symbol,
        )
    ]
    donor_shapes = {_export_shape(record) for record in donor_exports}
    current_shapes = {_export_shape(record) for record in current_exports}
    export_required = not (
        str(evidence.get("dependency_scope") or "") == "kernel_tree"
        or any(
            entity_is_internally_linked(entity)
            for entity in current_entities
            if isinstance(entity, dict)
        )
        or module_boundary == "same_module"
    ) and module_boundary == "cross_module"
    if module_boundary == "cross_module":
        relation = export_contract_relation(
            current_exports,
            donor_exports,
            required=export_required,
        )
    elif any(
        entity_is_internally_linked(entity)
        for entity in current_entities
        if isinstance(entity, dict)
    ):
        relation = "not_required"
    elif module_boundary == "same_module":
        relation = "not_required"
    elif donor_exports and current_shapes == donor_shapes:
        relation = "equivalent"
    elif donor_exports and not current_shapes:
        relation = "missing_in_old"
    elif not donor_exports:
        relation = "not_required"
    else:
        relation = "unavailable"
    return relation, {
        "current_provider_files": current_provider_files,
        "donor_provider_files": donor_provider_files,
        "module_boundary": module_boundary,
        "donor_exports": donor_exports,
        "current_exports": current_exports,
    }


def _module_names(records: Iterable[dict[str, Any]]) -> set[str]:
    return {
        normalize_relpath(str(record.get("module") or ""))
        for record in records
        if str(record.get("module") or "")
    }


def _module_boundary_relation(
    collector: ExactCurrentTreeEvidence,
    evidence: dict[str, Any],
    current_entities: Iterable[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    entities = [
        entity for entity in current_entities if isinstance(entity, dict)
    ]
    if (
        str(evidence.get("dependency_scope") or "") == "kernel_tree"
        or any(entity_is_internally_linked(entity) for entity in entities)
    ):
        return "not_required", {}
    provider_files = _ordered_strings(
        entity.get("file") for entity in entities
    )
    consumer_files = _ordered_strings(
        reference.get("source_file")
        for reference in evidence.get("consumer_references") or []
        if isinstance(reference, dict)
    )
    provider_modules = {
        module
        for relative_path in provider_files
        for module in _module_names(
            collector.module_owners("old", relative_path)
        )
    }
    consumer_modules = {
        module
        for relative_path in consumer_files
        for module in _module_names(
            collector.module_owners("old", relative_path)
        )
    }
    if not provider_modules or not consumer_modules:
        relation = "unavailable"
    elif consumer_modules <= provider_modules:
        relation = "same_module"
    else:
        relation = "cross_module"
    return relation, {
        "provider_files": provider_files,
        "consumer_files": consumer_files,
        "provider_modules": sorted(provider_modules),
        "consumer_modules": sorted(consumer_modules),
    }


def reference_shape(
    reference: dict[str, Any],
    *,
    target_representation_agnostic: bool = False,
) -> tuple[str, str, str, str]:
    owner = (
        reference.get("owner_entity")
        if isinstance(reference.get("owner_entity"), dict)
        else {}
    )
    target = (
        reference.get("referenced_entity")
        if isinstance(reference.get("referenced_entity"), dict)
        else {}
    )
    target_identity = (
        str(
            reference.get("symbol")
            or target.get("symbol")
            or target.get("name")
            or ""
        )
        if target_representation_agnostic
        else str(
            reference.get("contract_key")
            or entity_contract_key(target)
            or ""
        )
    )
    owner_identity = (
        ""
        if target_representation_agnostic
        else entity_contract_key(owner)
    )
    return (
        normalize_relpath(str(reference.get("source_file") or "")),
        owner_identity,
        target_identity,
        str(reference.get("use_kind") or ""),
    )


def _unique_references(
    references: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    for reference in references:
        shape = reference_shape(reference)
        identity = (
            *shape,
            int(reference.get("line") or 0),
            int(reference.get("column") or 0),
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(reference)
    return result


def integration_reference_relation(
    collector: ExactCurrentTreeEvidence,
    evidence: dict[str, Any],
    *,
    target_representation_agnostic: bool = False,
) -> tuple[str, dict[str, Any]]:
    recorded_reference_locators = _unique_references(
        [
            reference
            for reference in evidence.get("consumer_references") or []
            if isinstance(reference, dict)
        ]
    )
    reference_locators = [
        reference
        for reference in recorded_reference_locators
        if reference_is_source_contract(reference)
    ]
    if not reference_locators:
        return (
            "not_required"
            if recorded_reference_locators
            else "unavailable"
        ), {
            "reason": (
                "recorded references are compiler-generated and not "
                "independent source contracts"
                if recorded_reference_locators
                else "contract evidence has no exact donor reference"
            ),
            "ignored_generated_reference_shapes": [
                list(reference_shape(reference))
                for reference in recorded_reference_locators
                if not reference_is_source_contract(reference)
            ],
        }
    by_file: dict[str, list[dict[str, Any]]] = {}
    for reference in reference_locators:
        relative_path = normalize_relpath(
            str(reference.get("source_file") or "")
        )
        by_file.setdefault(relative_path, []).append(reference)

    donor_references: list[dict[str, Any]] = []
    current_references: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for relative_path, locators in sorted(by_file.items()):
        translation_units = [
            reference.get("translation_unit_file")
            for reference in locators
        ]
        donor_report = collector.references(
            tree="donor_0",
            relative_path=relative_path,
            translation_units=translation_units,
        )
        current_report = collector.references(
            tree="old",
            relative_path=relative_path,
            translation_units=translation_units,
        )
        reports.append(
            {
                "file": relative_path,
                "donor": {
                    key: deepcopy(donor_report.get(key))
                    for key in ("status", "tree", "parse_reports")
                },
                "current": {
                    key: deepcopy(current_report.get(key))
                    for key in ("status", "tree", "parse_reports")
                },
            }
        )
        if (
            donor_report.get("status") != "resolved"
            or current_report.get("status") != "resolved"
        ):
            return "unavailable", {
                "reference_locator_shapes": [
                    list(reference_shape(reference))
                    for reference in reference_locators
                ],
                "donor_reference_shapes": [],
                "current_reference_shapes": [],
                "reports": reports,
            }

        expected_shapes = {
            reference_shape(
                reference,
                target_representation_agnostic=(
                    target_representation_agnostic
                ),
            )
            for reference in locators
        }
        selected_donor = [
            reference
            for reference in donor_report.get("references") or []
            if isinstance(reference, dict)
            and reference_is_source_contract(reference)
            and reference_shape(
                reference,
                target_representation_agnostic=(
                    target_representation_agnostic
                ),
            )
            in expected_shapes
        ]
        if not selected_donor:
            return "unavailable", {
                "reason": (
                    "fresh donor graph no longer resolves the exact "
                    "reference locator"
                ),
                "reference_locator_shapes": [
                    list(shape) for shape in sorted(expected_shapes)
                ],
                "donor_reference_shapes": [],
                "current_reference_shapes": [],
                "reports": reports,
            }
        donor_references.extend(selected_donor)
        current_references.extend(
            reference
            for reference in current_report.get("references") or []
            if isinstance(reference, dict)
            and reference_is_source_contract(reference)
            and reference_shape(
                reference,
                target_representation_agnostic=(
                    target_representation_agnostic
                ),
            )
            in expected_shapes
        )

    donor_shapes = Counter(
        reference_shape(
            reference,
            target_representation_agnostic=(
                target_representation_agnostic
            ),
        )
        for reference in _unique_references(donor_references)
    )
    current_shapes = Counter(
        reference_shape(
            reference,
            target_representation_agnostic=(
                target_representation_agnostic
            ),
        )
        for reference in _unique_references(current_references)
    )
    missing = donor_shapes - current_shapes
    relation = (
        "target_alternative_review_required"
        if missing and target_representation_agnostic
        else "missing_in_old"
        if missing
        else "equivalent"
    )
    return relation, {
        "donor_reference_shapes": [
            {"shape": list(shape), "count": count}
            for shape, count in sorted(donor_shapes.items())
        ],
        "current_reference_shapes": [
            {"shape": list(shape), "count": count}
            for shape, count in sorted(current_shapes.items())
        ],
        "missing_reference_shapes": [
            {"shape": list(shape), "count": count}
            for shape, count in sorted(missing.items())
        ],
        "reports": reports,
        "target_representation_agnostic": bool(
            target_representation_agnostic
        ),
        "comparison_limit": (
            "donor_reference_shape_only; inspect exact current target "
            "source before deciding that behavior is absent"
            if missing and target_representation_agnostic
            else None
        ),
    }


def evaluate_integration_reference_relation(
    collector: ExactCurrentTreeEvidence,
    evidence: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Apply the canonical requirement policy to one reference relation."""

    representation_agnostic = (
        integration_reference_target_representation_agnostic(evidence)
    )
    return integration_reference_relation(
        collector,
        evidence,
        target_representation_agnostic=representation_agnostic,
    )


def _semantic_requirement_for_evidence(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    persisted = (
        dict(evidence.get("semantic_requirement"))
        if isinstance(evidence.get("semantic_requirement"), dict)
        else {}
    )
    dependency_scope = str(evidence.get("dependency_scope") or "")
    if dependency_scope and not persisted.get("dependency_scopes"):
        persisted["dependency_scopes"] = [dependency_scope]
    return canonical_semantic_requirement(persisted)


def integration_reference_target_representation_agnostic(
    evidence: dict[str, Any],
) -> bool:
    """Return the canonical reference-identity policy for one contract."""

    requirement = _semantic_requirement_for_evidence(evidence)
    dependency_scopes = {
        str(item)
        for item in requirement.get("dependency_scopes") or []
        if str(item)
    }
    required_dimensions = set(
        requirement.get("required_dimensions") or []
    )
    return bool(
        dependency_scopes == {"kernel_tree"}
        and "consumer_compile_binding" in required_dimensions
    )


def evaluate_exact_cross_module_export_obligation(
    collector: ExactCurrentTreeEvidence,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Prove whether one exact donor export can be materialized now."""

    required_dimensions = set(
        (evidence.get("semantic_requirement") or {}).get(
            "required_dimensions"
        )
        or []
    )
    recorded_donor_exports = [
        record
        for record in evidence.get("donor_exports") or []
        if isinstance(record, dict) and str(record.get("raw") or "").strip()
    ]
    if "export" not in required_dimensions or len(recorded_donor_exports) != 1:
        return {
            "status": "not_applicable",
            "reason": "contract has no single exact donor export obligation",
        }

    _relations, provider = _current_entity_relations(collector, evidence)
    current_entities = [
        entity
        for entity in provider.get("current_entities") or []
        if isinstance(entity, dict)
    ]
    donor_entities = [
        entity
        for entity in provider.get("donor_entities") or []
        if isinstance(entity, dict)
    ]
    module_boundary, module_observation = _module_boundary_relation(
        collector,
        evidence,
        current_entities,
    )
    export_relation, export_observation = _export_relation(
        collector,
        evidence,
        current_entities,
        donor_entities,
        module_boundary=module_boundary,
    )
    relations = {
        "module_boundary": module_boundary,
        "export": export_relation,
    }
    observations = {
        "provider_entity": provider,
        "module_boundary": module_observation,
        "exports": export_observation,
    }
    if export_relation == "equivalent":
        relations["integration_reference"] = "not_evaluated"
        observations["integration_reference"] = {
            "status": "not_evaluated",
            "reason": "exact current export is already satisfied",
        }
        observations["external_consumer_binding"] = {
            "status": "not_evaluated",
            "reason": "no export mutation is required",
        }
        return {
            "status": "already_satisfied",
            "relations": relations,
            "observations": observations,
        }

    provider_modules = set(
        module_observation.get("provider_modules") or []
    )
    external_references: list[dict[str, Any]] = []
    external_reference_files: dict[str, list[str]] = {}
    for reference in evidence.get("consumer_references") or []:
        if not isinstance(reference, dict):
            continue
        relative_path = normalize_relpath(
            str(reference.get("source_file") or "")
        )
        consumer_modules = {
            module
            for module in _module_names(
                collector.module_owners("old", relative_path)
            )
        }
        if consumer_modules and not consumer_modules <= provider_modules:
            external_references.append(reference)
            external_reference_files[relative_path] = sorted(
                consumer_modules
            )
    reference_relation, reference_observation = (
        integration_reference_relation(
            collector,
            {
                **evidence,
                "consumer_references": external_references,
            },
            target_representation_agnostic=True,
        )
    )
    relations["integration_reference"] = reference_relation
    observations["integration_reference"] = reference_observation
    observations["external_consumer_binding"] = {
        "reference_count": len(external_references),
        "consumer_modules_by_file": external_reference_files,
    }

    recorded_export = recorded_donor_exports[0]
    recorded_shape = _export_shape(recorded_export)
    fresh_donor_exports = [
        record
        for record in export_observation.get("donor_exports") or []
        if isinstance(record, dict)
        and _export_shape(record) == recorded_shape
        and str(record.get("raw") or "").strip()
        == str(recorded_export.get("raw") or "").strip()
    ]
    ready = bool(
        provider.get("status") == "resolved"
        and len(current_entities) == 1
        and module_boundary == "cross_module"
        and bool(external_references)
        and reference_relation == "equivalent"
        and export_relation in {"missing_in_old", "required_missing"}
        and len(fresh_donor_exports) == 1
    )
    return {
        "status": "ready" if ready else "blocked",
        "reason": (
            None
            if ready
            else "current provider, consumer binding, module boundary, and exact donor export are not jointly proven"
        ),
        "relations": relations,
        "observations": observations,
        **(
            {
                "current_entity": current_entities[0],
                "donor_export": fresh_donor_exports[0],
            }
            if ready
            else {}
        ),
    }


def _reverse_patch_text_is_current(
    root: Path,
    diff_text: str,
    *,
    source: str,
) -> dict[str, Any]:
    if not diff_text.strip():
        return {
            "status": "unavailable",
            "patch_source": str(source),
            "reason": "patch_artifact_empty",
        }
    try:
        returncode, stdout, stderr = run_unified_diff(
            root,
            diff_text,
            dry_run=True,
            reverse=True,
        )
    except OSError as exc:
        return {
            "status": "unavailable",
            "patch_source": str(source),
            "reason": f"{exc.__class__.__name__}: {exc}",
        }
    return {
        "status": "present" if returncode == 0 else "not_present",
        "patch_source": str(source),
        "returncode": int(returncode),
        "stdout": stdout,
        "stderr": stderr,
    }


def _reverse_patch_is_current(
    root: Path,
    patch_path: Path,
) -> dict[str, Any]:
    """Check whether an accepted patch is still present without writing."""

    if not patch_path.is_file():
        return {
            "status": "unavailable",
            "patch_source": str(patch_path),
            "reason": "patch_artifact_missing",
        }
    try:
        diff_text = patch_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as exc:
        return {
            "status": "unavailable",
            "patch_source": str(patch_path),
            "reason": f"{exc.__class__.__name__}: {exc}",
        }
    return _reverse_patch_text_is_current(
        root,
        diff_text,
        source=str(patch_path),
    )


def _initial_outcomes_for_state(
    state_root: Path,
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    snapshot_ref = str(state.get("initial_unit_outcomes_ref") or "")
    if not snapshot_ref:
        raise ValueError(
            "compile-valid semantic handoff requires the immutable B0 "
            "outcome snapshot"
        )
    snapshot = load_semantic_record(state_root, snapshot_ref)
    if not isinstance(snapshot, dict):
        raise ValueError("initial unit outcome snapshot is not an object")
    return {
        str(unit_id): dict(outcome)
        for unit_id, outcome in (
            snapshot.get("unit_outcomes") or {}
        ).items()
        if isinstance(outcome, dict)
    }


def _initial_treatment_for_gap(
    spec: object,
    gap: dict[str, Any],
    *,
    state_root: Path,
    unit_outcomes: dict[str, dict[str, Any]],
    patch_presence_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_unit_ids = _ordered_strings(
        gap.get("source_work_unit_ids") or []
    )
    outcomes = [
        unit_outcomes.get(unit_id)
        for unit_id in source_unit_ids
    ]
    if not source_unit_ids:
        return {
            "status": "no_initial_action",
            "source_work_unit_ids": [],
            "reason": "demand_has_no_change_atom",
        }
    if any(not isinstance(outcome, dict) for outcome in outcomes):
        return {
            "status": "unresolved",
            "source_work_unit_ids": source_unit_ids,
            "reason": "source_unit_outcome_unavailable",
        }
    normalized = [
        dict(outcome)
        for outcome in outcomes
        if isinstance(outcome, dict)
    ]
    accepted = {"modified", "already_satisfied", "validation_only"}
    if any(
        str(outcome.get("status") or "") not in accepted
        for outcome in normalized
    ):
        return {
            "status": "unresolved",
            "source_work_unit_ids": source_unit_ids,
            "unit_outcomes": normalized,
            "reason": "one_or_more_source_units_were_not_completed",
        }

    modified = [
        outcome
        for outcome in normalized
        if outcome.get("status") == "modified"
    ]
    if not modified:
        return {
            "status": (
                "validation_only"
                if any(
                    outcome.get("status") == "validation_only"
                    for outcome in normalized
                )
                else "already_satisfied"
            ),
            "source_work_unit_ids": source_unit_ids,
            "unit_outcomes": normalized,
            "patch_presence": "not_applicable",
        }

    patch_backed = all(
        bool(
            outcome.get("patch_record_refs")
            or outcome.get("patch_paths")
        )
        for outcome in modified
    )
    patch_checks: list[dict[str, Any]] = []
    if patch_backed:
        for patch_ref in _ordered_strings(
            reference
            for outcome in modified
            for reference in outcome.get("patch_record_refs") or []
        ):
            cache_key = f"record:{patch_ref}"
            if cache_key not in patch_presence_cache:
                patch_record = load_semantic_record(
                    state_root,
                    patch_ref,
                )
                if not isinstance(patch_record, dict):
                    raise ValueError(
                        "initial patch record is not an object"
                    )
                patch_presence_cache[cache_key] = (
                    _reverse_patch_text_is_current(
                        Path(getattr(spec, "old_root_path")),
                        str(patch_record.get("diff_text") or ""),
                        source=patch_ref,
                    )
                )
            patch_checks.append(
                deepcopy(patch_presence_cache[cache_key])
            )
        for patch_path in _ordered_strings(
            path
            for outcome in modified
            if not outcome.get("patch_record_refs")
            for path in outcome.get("patch_paths") or []
        ):
            if patch_path not in patch_presence_cache:
                patch_presence_cache[patch_path] = (
                    _reverse_patch_is_current(
                        Path(getattr(spec, "old_root_path")),
                        Path(patch_path),
                    )
                )
            patch_checks.append(
                deepcopy(patch_presence_cache[patch_path])
            )
    patch_presence = (
        "present"
        if patch_checks
        and all(check.get("status") == "present" for check in patch_checks)
        else "not_present"
        if patch_checks
        and any(
            check.get("status") == "not_present"
            for check in patch_checks
        )
        else "unavailable"
        if patch_checks
        else "static_validation_required"
    )
    return {
        "status": "modified",
        "source_work_unit_ids": source_unit_ids,
        "unit_outcomes": normalized,
        "all_modified_units_patch_backed": patch_backed,
        "patch_presence": patch_presence,
        "patch_checks": patch_checks,
    }


def evaluate_contract_evidence(
    collector: ExactCurrentTreeEvidence,
    *,
    gap_id: str,
    evidence: dict[str, Any],
    checkpoint_ref: str,
    initial_treatment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one initial contract against the compile-valid current tree."""

    persisted_required_dimensions = _ordered_strings(
        (
            evidence.get("semantic_requirement")
            if isinstance(evidence.get("semantic_requirement"), dict)
            else {}
        ).get("required_dimensions")
        or []
    )
    requirement = _semantic_requirement_for_evidence(evidence)
    required_dimensions = _ordered_strings(
        requirement.get("required_dimensions") or []
    )
    if not required_dimensions:
        raise ValueError(
            f"semantic contract has no required dimensions: {gap_id}"
        )

    relations: dict[str, str] = {}
    observations: dict[str, Any] = {}
    if "consumer_compile_binding" in required_dimensions:
        relations["consumer_compile_binding"] = "compile_valid"
        observations["consumer_compile_binding"] = {
            "status": "compile_valid",
            "checkpoint_ref": str(checkpoint_ref),
            "authority": "target_driver_compile_checkpoint",
        }
    entity_dimensions = {
        "name",
        "signature",
        "implementation",
        "interface",
        "structural_source",
    }
    if entity_dimensions.intersection(required_dimensions):
        entity_relations, entity_observation = (
            _current_entity_relations(collector, evidence)
        )
        relations.update(entity_relations)
        observations["provider_entity"] = entity_observation
    current_entities = [
        entity
        for entity in (
            observations.get("provider_entity") or {}
        ).get("current_entities")
        or []
        if isinstance(entity, dict)
    ]
    donor_entities = [
        entity
        for entity in (
            observations.get("provider_entity") or {}
        ).get("donor_entities")
        or []
        if isinstance(entity, dict)
    ]

    if "declaration" in required_dimensions:
        relation, observation = _declaration_relation(
            collector,
            evidence,
        )
        relations["declaration"] = relation
        observations["declarations"] = observation
    module_boundary = "not_required"
    if {
        "module_boundary",
        "export",
    }.intersection(required_dimensions):
        module_boundary, observation = _module_boundary_relation(
            collector,
            evidence,
            current_entities,
        )
        if "module_boundary" in required_dimensions:
            relations["module_boundary"] = module_boundary
        observations["module_boundary"] = observation
    if "export" in required_dimensions:
        relation, observation = _export_relation(
            collector,
            evidence,
            current_entities,
            donor_entities,
            module_boundary=module_boundary,
        )
        relations["export"] = relation
        observations["exports"] = observation
    if "integration_reference" in required_dimensions:
        relation, observation = evaluate_integration_reference_relation(
            collector,
            evidence,
        )
        relations["integration_reference"] = relation
        observations["integration_reference"] = observation

    unresolved_dimensions: list[str] = []
    unavailable_dimensions: list[str] = []
    for dimension in required_dimensions:
        relation = str(relations.get(dimension) or "unavailable")
        if relation in _SATISFIED_RELATIONS.get(dimension, set()):
            continue
        if relation in _DIFFERENCE_RELATIONS:
            unresolved_dimensions.append(dimension)
        else:
            unavailable_dimensions.append(dimension)
    treatment = dict(initial_treatment or {})
    observations["initial_treatment"] = treatment
    current_patch_treatment = bool(
        treatment.get("status") == "modified"
        and treatment.get("all_modified_units_patch_backed")
        and treatment.get("patch_presence") == "present"
    )
    # Patch lineage proves that a change survives in the current tree.  It is
    # provenance, not evidence that any required contract dimension holds.
    observations["initial_patch_resolved_dimensions"] = []
    if unresolved_dimensions:
        status = "unresolved"
        resolution_basis = "fresh_cross_version_contract_difference"
    elif unavailable_dimensions:
        status = "evidence_unavailable"
        resolution_basis = "fresh_cross_version_evidence_incomplete"
    else:
        status = (
            "satisfied_modified"
            if current_patch_treatment
            else "satisfied"
        )
        resolution_basis = (
            "fresh_contract_satisfied_after_initial_modification"
            if status == "satisfied_modified"
            else "fresh_cross_version_contract_satisfied"
        )

    return {
        "schema": COMPILE_VALID_HANDOFF_SCHEMA,
        "gap_id": str(gap_id),
        "contract_identity": str(
            evidence.get("contract_identity") or ""
        ),
        "checkpoint_ref": str(checkpoint_ref),
        "status": status,
        "resolution_basis": resolution_basis,
        "persisted_required_dimensions": persisted_required_dimensions,
        "required_dimensions": required_dimensions,
        "relations": {
            dimension: str(relations.get(dimension) or "unavailable")
            for dimension in required_dimensions
        },
        "unresolved_dimensions": unresolved_dimensions,
        "unavailable_dimensions": unavailable_dimensions,
        "observations": observations,
        "evaluated_at": utc_now(),
    }


def evaluate_compile_valid_semantic_handoff(
    spec: object,
    *,
    state_root: Path,
    checkpoint_ref: str,
) -> dict[str, Any]:
    """Run the one-time bounded static handoff after compilation succeeds."""

    checkpoint_ref = str(checkpoint_ref or "").strip()
    if not checkpoint_ref:
        raise ValueError("compile-valid semantic handoff needs a checkpoint")
    state = load_semantic_closure_state(
        state_root,
        case_id=str(getattr(spec, "case_id")),
    )
    if state is None:
        return {
            "schema": COMPILE_VALID_HANDOFF_SCHEMA,
            "status": "not_initialized",
        }
    previous = state.get(COMPILE_VALID_HANDOFF_KEY)
    refreshing_checkpoint = False
    if (
        isinstance(previous, dict)
        and previous.get("status") == "complete"
    ):
        previous_revision = str(
            previous.get("evaluator_revision") or ""
        )
        if (
            previous_revision
            != COMPILE_VALID_HANDOFF_EVALUATOR_REVISION
        ):
            return {
                "schema": COMPILE_VALID_HANDOFF_SCHEMA,
                "status": "stale_handoff",
                "expected_evaluator_revision": (
                    COMPILE_VALID_HANDOFF_EVALUATOR_REVISION
                ),
                "actual_evaluator_revision": (
                    previous_revision or "unversioned"
                ),
                "checkpoint_ref": str(
                    previous.get("checkpoint_ref") or checkpoint_ref
                ),
                "required_action": (
                    "initialize a fresh semantic ledger from the immutable "
                    "initial analysis and plan; do not resume semantic "
                    "decisions derived from the stale handoff"
                ),
                "summary": semantic_closure_summary(state),
            }
        if str(previous.get("checkpoint_ref") or "") == checkpoint_ref:
            return {
                **deepcopy(previous),
                "status": "already_complete",
                "summary": semantic_closure_summary(state),
            }
        refreshing_checkpoint = True
    if state.get("active_batch_id"):
        raise ValueError(
            "compile-valid handoff cannot refresh an unvalidated batch"
        )
    if not refreshing_checkpoint and state.get("rounds"):
        raise ValueError(
            "compile-valid handoff must precede semantic repair rounds"
        )
    assert_semantic_source_artifacts_integrity(state_root, state)

    if refreshing_checkpoint:
        pending_gap_ids = sorted(
            str(gap_id)
            for gap_id, gap in (state.get("gaps") or {}).items()
            if isinstance(gap, dict)
            and gap.get("status") != CLOSED_MODIFIED
            and not isinstance(
                gap.get("post_batch_contract_evaluation"),
                dict,
            )
        )
    else:
        pending_gap_ids = sorted(
            str(gap_id)
            for gap_id, gap in (state.get("gaps") or {}).items()
            if isinstance(gap, dict)
            and gap.get("status") == PENDING_DECISION
        )
    unit_outcomes = _initial_outcomes_for_state(
        state_root,
        state,
    )
    collector = ExactCurrentTreeEvidence(spec)
    patch_presence_cache: dict[str, dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []
    initial_treatments: dict[str, dict[str, Any]] = {}
    for gap_id in pending_gap_ids:
        gap = state["gaps"][gap_id]
        initial_treatment = _initial_treatment_for_gap(
            spec,
            gap,
            state_root=state_root,
            unit_outcomes=unit_outcomes,
            patch_presence_cache=patch_presence_cache,
        )
        initial_treatments[gap_id] = initial_treatment
        evidence_ref = str(
            gap.get("contract_evidence_ref") or ""
        )
        evidence = load_semantic_record(
            state_root,
            evidence_ref,
        )
        if not isinstance(evidence, dict):
            raise ValueError(
                f"semantic contract evidence is not an object: {gap_id}"
            )
        evaluation = evaluate_contract_evidence(
            collector,
            gap_id=gap_id,
            evidence=evidence,
            checkpoint_ref=checkpoint_ref,
            initial_treatment=initial_treatment,
        )
        evaluation_ref = put_semantic_record(
            state_root,
            kind="compile-valid-evaluation",
            record=evaluation,
        )
        evaluations.append(
            {
                **evaluation,
                "evaluation_ref": evaluation_ref,
            }
        )

    transition = record_compile_valid_handoff_evaluations(
        state,
        evaluations=evaluations,
        checkpoint_ref=checkpoint_ref,
        refresh=refreshing_checkpoint,
    )
    if refreshing_checkpoint:
        state_path = write_semantic_closure_state(
            state_root,
            state,
        )
        return {
            **deepcopy(state[COMPILE_VALID_HANDOFF_KEY]),
            "status": "refreshed",
            "evaluated_gap_count": len(evaluations),
            "refresh_transition": transition,
            "state_path": str(state_path),
            "summary": semantic_closure_summary(state),
        }
    marker = {
        "schema": COMPILE_VALID_HANDOFF_SCHEMA,
        "evaluator_revision": (
            COMPILE_VALID_HANDOFF_EVALUATOR_REVISION
        ),
        "status": "complete",
        "checkpoint_ref": checkpoint_ref,
        "evaluated_gap_count": len(evaluations),
        "registered_candidate_gap_count": len(
            state.get("gaps") or {}
        ),
        "initially_handled_gap_count": sum(
            treatment.get("status")
            in {"modified", "already_satisfied"}
            for treatment in initial_treatments.values()
        ),
        "initially_unresolved_gap_count": sum(
            treatment.get("status") == "unresolved"
            for treatment in initial_treatments.values()
        ),
        "closed_modified_gap_count": sum(
            evaluation.get("status") == "satisfied_modified"
            for evaluation in evaluations
        ),
        "closed_no_change_gap_count": sum(
            evaluation.get("status") == "satisfied"
            for evaluation in evaluations
        ),
        "closed_gap_count": len(
            transition["closed_gap_ids"]
        ),
        "unresolved_gap_count": len(
            transition["unresolved_gap_ids"]
        ),
        "evidence_unavailable_gap_count": len(
            transition["evidence_unavailable_gap_ids"]
        ),
        "closed_gap_ids": list(
            transition["closed_gap_ids"]
        ),
        "unresolved_gap_ids": list(
            transition["unresolved_gap_ids"]
        ),
        "evidence_unavailable_gap_ids": list(
            transition["evidence_unavailable_gap_ids"]
        ),
        "completed_at": utc_now(),
    }
    state[COMPILE_VALID_HANDOFF_KEY] = marker
    snapshot_transition = activate_semantic_gap_snapshot(
        state,
        checkpoint_ref=checkpoint_ref,
        evaluator_revision=COMPILE_VALID_HANDOFF_EVALUATOR_REVISION,
    )
    state_path = write_semantic_closure_state(
        state_root,
        state,
    )
    return {
        **marker,
        "semantic_gap_snapshot": snapshot_transition,
        "state_path": str(state_path),
        "summary": semantic_closure_summary(state),
    }


def evaluate_active_batch_semantic_contracts(
    spec: object,
    *,
    state_root: Path,
    state: dict[str, Any],
    batch: dict[str, Any],
    checkpoint_ref: str,
) -> dict[str, Any]:
    """Revalidate every claimed gap after semantic compile recovery."""

    evaluation_gap_ids = _ordered_strings(
        [
            *list(batch.get("primary_gap_ids") or []),
            *list(batch.get("affected_gap_ids") or []),
        ]
    )
    realized_gap_ids = {
        str(gap_id)
        for region in batch.get("active_patch_regions") or []
        if isinstance(region, dict)
        for gap_id in region.get("direct_gap_ids") or []
        if str(gap_id)
    }
    realized_gap_ids.update(
        str(gap_id)
        for gap_id in batch.get("affected_gap_ids") or []
        if str(gap_id)
    )
    collector: ExactCurrentTreeEvidence | None = None
    evaluations: list[dict[str, Any]] = []
    for gap_id in evaluation_gap_ids:
        gap = (state.get("gaps") or {}).get(gap_id)
        if not isinstance(gap, dict):
            raise ValueError(
                f"active batch references unknown semantic gap: {gap_id}"
            )
        evidence = load_semantic_record(
            state_root,
            str(gap.get("contract_evidence_ref") or ""),
        )
        if not isinstance(evidence, dict):
            raise ValueError(
                f"semantic contract evidence is not an object: {gap_id}"
            )
        if collector is None:
            collector = ExactCurrentTreeEvidence(spec)
        treatment = {
            "status": (
                "modified"
                if gap_id in realized_gap_ids
                else "unresolved"
            ),
            "all_modified_units_patch_backed": (
                gap_id in realized_gap_ids
            ),
            "patch_presence": (
                "present"
                if gap_id in realized_gap_ids
                else "not_present"
            ),
            "source": "verified_patch_lineage_and_current_tree_evidence",
        }
        evaluation = {
            **evaluate_contract_evidence(
                collector,
                gap_id=gap_id,
                evidence=evidence,
                checkpoint_ref=checkpoint_ref,
                initial_treatment=treatment,
            ),
            "evaluation_phase": "post_semantic_batch_compile",
        }
        evaluation_ref = put_semantic_record(
            state_root,
            kind="post-batch-contract-evaluation",
            record=evaluation,
        )
        evaluations.append(
            {
                **evaluation,
                "evaluation_ref": evaluation_ref,
            }
        )
    counts = Counter(
        str(evaluation.get("status") or "")
        for evaluation in evaluations
    )
    return {
        "schema": "post_semantic_batch_contract_evaluation_v1",
        "batch_id": str(batch.get("id") or ""),
        "checkpoint_ref": str(checkpoint_ref),
        "evaluations": evaluations,
        "summary": {
            "evaluated_gap_count": len(evaluations),
            "satisfied_modified_count": int(
                counts["satisfied_modified"]
            ),
            "satisfied_no_change_count": int(counts["satisfied"]),
            "unresolved_count": int(counts["unresolved"]),
            "evidence_unavailable_count": int(
                counts["evidence_unavailable"]
            ),
        },
    }
