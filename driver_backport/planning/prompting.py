#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from driver_backport.core.case_spec import CaseSpec
from driver_backport.planning.prompt_evidence import (
    compact_prompt_references,
    deduplicate_source_evidence,
    prompt_contract_evidence,
    render_prompt_payload,
    source_evidence_record,
)


REPAIR_PLAN_KIND = "repair"
REPAIR_PLAN_UNITS_KEY = "work_unit_plans"


def repair_plan_units(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if REPAIR_PLAN_UNITS_KEY in plan:
        return list(plan.get(REPAIR_PLAN_UNITS_KEY) or [])
    raise ValueError("unrecognized repair plan schema")


def _compact_prior_unresolved_attempt_for_prompt(attempt: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: attempt.get(key)
        for key in (
            "source",
            "attempt_number",
            "cycle",
            "work_unit_id",
            "plan_unit_id",
            "status",
            "tree_change",
            "validation_relation",
            "diagnostic_component_match",
            "unresolved_diagnostic_signatures",
            "resolved_diagnostic_signatures_since_attempt",
            "new_diagnostic_signatures_since_attempt",
            "touched_files",
            "patch_path",
            "patch_excerpt_omitted_reason",
            "patch_delta_summary",
            "diagnostic_snapshot",
            "apply_failures",
        )
        if attempt.get(key) not in (None, "", [], {})
    }
    note = str(attempt.get("note") or "").strip()
    if note:
        compact["note"] = note
    patch_excerpt = [str(line) for line in attempt.get("patch_excerpt_lines") or []]
    if patch_excerpt:
        compact["patch_excerpt_lines"] = patch_excerpt
    return compact


def _source_record_header(record: dict[str, Any]) -> str:
    tree = str(record.get("tree") or "<tree>")
    file_name = str(record.get("file") or "<file>")
    role = str(
        record.get("role")
        or ",".join(str(item) for item in record.get("roles") or [])
        or "<role>"
    )
    source_kind = str(record.get("source_kind") or "<source>")
    line_start = record.get("line_start")
    line_end = record.get("line_end")
    location = ""
    if line_start is not None and line_end is not None:
        location = f":{line_start}-{line_end}"
    elif record.get("line_numbers"):
        location = ":" + ",".join(str(item) for item in record.get("line_numbers") or [])
    symbol = record.get("symbol")
    symbol_suffix = f" symbol={symbol}" if symbol else ""
    return f"--- {tree} {role} {file_name}{location} ({source_kind}){symbol_suffix}"


def _compact_initial_difference_unit(
    unit: dict[str, Any],
) -> dict[str, Any]:
    atoms = [
        item
        for item in unit.get("difference_atoms") or []
        if isinstance(item, dict)
    ]
    anchors = [
        item
        for item in unit.get("demand_anchors") or []
        if isinstance(item, dict)
    ]
    loci = [
        item
        for item in unit.get("repair_loci") or []
        if isinstance(item, dict)
    ]
    evidence = [
        item
        for item in unit.get("contract_evidence") or []
        if isinstance(item, dict)
    ]
    return {
        "schema": "initial_difference_unit_prompt_v4",
        "difference_atoms": [dict(atom) for atom in atoms],
        "demand_anchors": compact_prompt_references(anchors),
        "repair_loci": [dict(locus) for locus in loci],
        "contract_evidence": [
            prompt_contract_evidence(item) for item in evidence
        ],
        "depends_on_work_unit_ids": list(
            unit.get("depends_on_work_unit_ids") or []
        ),
    }


def _compact_source_pack_summary(summary: object) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    return dict(summary)


def _structured_prompt_lines(value: Any) -> list[str]:
    rendered, instruction, _ = render_prompt_payload(value)
    return [
        *(
            ["Prompt transport encoding: " + instruction]
            if instruction
            else []
        ),
        rendered,
    ]


def _render_source_records(
    records: list[dict[str, Any]],
) -> list[str]:
    if not records:
        return ["<none>"]
    lines: list[str] = []
    for record in records:
        lines.append(_source_record_header(record))
        reason = str(record.get("reason") or "").strip()
        if reason:
            lines.append(f"reason: {reason}")
        if not record.get("exists", True):
            lines.append("<missing>")
            continue
        text = str(record.get("text") or "").rstrip()
        lines.append(text if text else "<no source text captured>")
    return lines


def _required_source_evidence(
    source_pack: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for section, default_role in (
        ("allowed_edit_sources", "old_edit_source"),
        ("donor_reference_sources", "donor_reference_source"),
        ("read_only_evidence_sources", "read_only_source"),
    ):
        for raw in source_pack.get(section) or []:
            if not isinstance(raw, dict):
                continue
            record = source_evidence_record(
                raw,
                role=str(raw.get("role") or default_role),
            )
            if record is None:
                continue
            if raw.get("reason"):
                record["reason"] = raw.get("reason")
            records.append(record)
    return deduplicate_source_evidence(records)


def _render_repair_source_pack(source_pack: dict[str, Any]) -> str:
    if not source_pack:
        return ""
    lines: list[str] = []
    lines.append("Repair source pack:")
    lines.extend(
        _structured_prompt_lines(
            _compact_source_pack_summary(source_pack.get("summary") or {})
        )
    )
    policy = source_pack.get("policy") or {}
    if policy:
        lines.append("Source-pack policy:")
        for key, value in policy.items():
            lines.append(f"- {key}: {value}")
    blockers = list(source_pack.get("localization_blockers") or [])
    if blockers:
        lines.append("Localization blockers:")
        lines.extend(_structured_prompt_lines(blockers))
    lines.append("")
    lines.append("Required source evidence:")
    lines.extend(
        _render_source_records(
            _required_source_evidence(source_pack),
        )
    )
    return "\n".join(lines).strip()


def build_initial_compatibility_prompt(
    spec: CaseSpec,
    *,
    unit: dict[str, Any],
) -> str:
    if unit.get("unit_kind") != "initial_compatibility":
        raise ValueError(
            "initial compatibility prompt requires an initial_compatibility unit"
        )
    decision = unit.get("repair_decision") or {}
    allowed_fileset = list(unit.get("allowed_fileset") or [])
    read_only_evidence_files = list(unit.get("read_only_evidence_files") or [])
    escalation_candidates = list(unit.get("escalation_candidates") or [])
    forbidden_candidates = list(unit.get("forbidden_candidates") or [])
    localization = dict(unit.get("localization") or {})
    patch_targets = list(unit.get("patch_targets") or [])
    repair_source_pack = unit.get("repair_source_pack") or {}
    prior_unresolved_patch_attempts = list(unit.get("prior_unresolved_patch_attempts") or [])

    lines = [
        "You are helping with an agentic Linux kernel driver backport.",
        f"Case ID: {spec.case_id}",
        f"Plan kind: {REPAIR_PLAN_KIND}",
        f"Unit title: {unit.get('title')}",
        f"Unit scope: {unit.get('scope')}",
        "Unit kind: initial_compatibility",
        f"Decision mode: {decision.get('mode')}",
    ]
    if prior_unresolved_patch_attempts:
        lines.append(
            "Prior patch attempts that did not resolve or could not be "
            "atomically applied for this work unit:"
        )
        compact_attempts = [
            _compact_prior_unresolved_attempt_for_prompt(item)
            for item in prior_unresolved_patch_attempts
            if isinstance(item, dict)
        ]
        lines.extend(_structured_prompt_lines(compact_attempts))
        lines.append(
            "Interpret each attempt by tree_change: not_applied means the patch "
            "never changed the candidate, while rolled_back means an applied "
            "transaction failed a postcondition and was restored. Do not "
            "repeat the same patch shape unless the new patch directly "
            "addresses the recorded failure."
        )
    lines.append("")
    lines.append("Initial compatibility demanded-difference unit:")
    lines.extend(
        _structured_prompt_lines(
            _compact_initial_difference_unit(unit)
        )
    )
    lines.append(
        "Resolve this atomic difference component. The demand anchors "
        "explain why each difference matters; repair loci identify the "
        "old-tree locations at which it can be realized."
    )
    if localization or patch_targets:
        lines.append("")
        lines.append("Patch target localization:")
        lines.extend(
            _structured_prompt_lines(
                {
                    "localization": localization,
                    "patch_targets": patch_targets,
                }
            )
        )

    rendered_source_pack = _render_repair_source_pack(repair_source_pack)
    if rendered_source_pack:
        lines.append("")
        lines.append(rendered_source_pack)

    lines.append("")
    lines.append("Allowed fileset:")
    lines.extend(f"- {item}" for item in allowed_fileset)
    if not allowed_fileset:
        lines.append("- <none>")

    lines.append("")
    lines.append("Read-only evidence files:")
    lines.extend(f"- {item}" for item in read_only_evidence_files)
    if not read_only_evidence_files:
        lines.append("- <none>")

    lines.append("")
    lines.append("Escalation candidates:")
    lines.extend(f"- {item}" for item in escalation_candidates)
    if not escalation_candidates:
        lines.append("- <none>")

    lines.append("")
    lines.append("Forbidden candidates:")
    lines.extend(f"- {item}" for item in forbidden_candidates)
    if not forbidden_candidates:
        lines.append("- <none>")

    lines.append("")
    lines.append("Graph decision rationale:")
    lines.extend(f"- {item}" for item in decision.get("reasoning") or [])
    if not decision.get("reasoning"):
        lines.append("- <none>")

    lines.append("")
    lines.append("Response protocol:")
    lines.append("The response is a structured decision envelope. The patch is only the patch.content_lines field inside that JSON object.")
    if unit.get("requires_context_expansion"):
        lines.append("This repair unit is blocked by repair-owner localization. Return needs_escalation; do not generate a patch for this unit.")
    lines.append('Return exactly one JSON object using this schema:')
    lines.append('{')
    lines.append('  "action": "apply_patch" | "no_patch" | "needs_escalation",')
    lines.append('  "decision": {')
    lines.append('    "mode": "<decision mode>",')
    lines.append(
        '    "difference_atom_ids": ["initial-difference-atom-id"],'
    )
    lines.append('    "rationale": "<one concise paragraph>"')
    lines.append('  },')
    lines.append('  "patch_plan": {')
    lines.append('    "strategy": "<short strategy>",')
    lines.append('    "affected_files": ["relative/path.c"],')
    lines.append('    "expected_outcome": "<what this patch should fix>"')
    lines.append('  },')
    lines.append('  "missing_context": {')
    lines.append('    "diagnostic_event_ids": ["diagnostic-node-id"],')
    lines.append('    "graph_node_ids": ["CodeIndex-node-id"],')
    lines.append('    "continuation": {},')
    lines.append('    "reason": "<exact unresolved graph edge or permission boundary>"')
    lines.append('  },')
    lines.append('  "patch": {')
    lines.append('    "format": "unified_diff",')
    lines.append('    "content_lines": ["diff --git a/relative/path.c b/relative/path.c", "..."]')
    lines.append('  }')
    lines.append('}')
    lines.append("- Return plain JSON only. Do not wrap the response in markdown prose.")
    lines.append('- This unit is generated before the first target compile. Return no_patch only when every listed DifferenceAtom is already satisfied in the current old tree.')
    lines.append('- Emit a patch when a demanded difference is unresolved, or return needs_escalation when no listed repair locus authorizes the required edit.')
    lines.append('- If patching, set "action" to "apply_patch" and include one unified diff in patch.content_lines.')
    lines.append('- If escalation is required, set "action" to "needs_escalation" and explain why in decision.rationale.')
    lines.append('- For needs_escalation, identify the unresolved diagnostic and CodeIndex node ids and preserve the stored continuation; do not propose guessed files or symbols.')
    lines.append('- In patch.content_lines, put one diff line per JSON array element, without trailing newline characters.')
    lines.append('- Do not put raw tab control characters inside JSON strings. Prefer spaces in patch.content_lines; if exact tab context is required, use the JSON escape \\t.')
    lines.append("")
    lines.append("Rules:")
    lines.append("- Work on the entire atomic unit, not one diff hunk at a time.")
    lines.append("- No compiler diagnostic exists yet; decide only from demand anchors, DifferenceAtoms, repair loci, and exact old/donor source evidence.")
    lines.append("- no_patch is valid only with a concrete explanation that every listed demanded difference is already satisfied.")
    lines.append("- Do not expand from a type name to an entire structure unless a demand anchor explicitly requires complete layout.")
    if unit.get("requires_context_expansion"):
        lines.append("- This unit has requires_context_expansion=true, so patch generation is forbidden until graph owner localization is rerun with expanded context.")
    lines.append("- Use needs_escalation, not no_patch, to express insufficient scope or missing evidence.")
    lines.append("- Preserve old-tree ABI and behavior unless the graph evidence proves that a contract extension is required.")
    lines.append("- Use exact old/donor definitions for constants and declarations; never invent values or interfaces that are absent from the evidence.")
    lines.append("- Do not invent a same-named compatibility macro, wrapper, or stub merely to make a missing donor API compile. Such a compatibility provider is valid only when its complete behavior and all direct dependencies are present in the supplied evidence and realizable in the old tree; otherwise adapt every listed consumer to an existing old-tree primitive.")
    lines.append("- Preserve acquisition, release, cleanup, and every early-return path when replacing a scoped helper. Remove any compatibility definition that becomes unused after its consumers are adapted.")
    lines.append("- If no editable repair locus is available, return its exact unresolved identity through needs_escalation instead of guessing a file.")
    lines.append("- Do not copy donor shared files wholesale into the old tree; patch the smallest compatible surface.")
    lines.append("- Base the final decision on the typed graph and exact source evidence above.")
    lines.append("- Prefer a coordinated multi-file patch across the allowed fileset when needed.")
    lines.append("- Do not touch files outside the allowed fileset.")
    lines.append("- Use Allowed edit sources as the current source for diff context; donor and read-only evidence sources are reference-only.")
    lines.append("- Donor reference sources may justify exact constants, declarations, fields, and macro bodies, but donor paths must not appear in the diff.")
    lines.append("- Read-only evidence sources may explain the diagnostic or consumer use-site, but they must not appear in patch_plan.affected_files or in the diff unless also listed as an allowed edit source.")
    lines.append("- Read-only evidence files may inform the decision, but they must not appear in patch_plan.affected_files or in the diff.")
    lines.append("- Treat shared contracts, build boundaries, data-path interfaces, and runtime behavior as first-class migration concerns when evidence points there.")
    lines.append("- Keep the decision consistent with the diff you emit.")
    lines.append("- Use relative paths rooted at the old tree.")
    return "\n".join(lines).strip() + "\n"
