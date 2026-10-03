#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    ensure_dir,
    normalize_relpath,
    ordered_unique_relpaths,
    utc_now,
    write_json,
)
from driver_backport.core.executor import (
    backup_file,
    finalize_transaction,
    rollback_transaction,
    run_unified_diff,
    start_transaction,
)
from driver_backport.llm.model_util import gpt_api_with_metadata
from driver_backport.planning.diagnostic_routing import (
    is_build_metadata_file,
    is_build_prep_issue,
    validation_phase_issue_pairs,
)
from driver_backport.planning.response_apply import (
    diff_from_structured_payload,
    extract_json_payload,
    preflight_patch,
    repair_unified_diff_structure,
    structured_json_error_summary,
    touched_files_from_diff,
)
from driver_backport.planning.prompt_evidence import render_prompt_payload
from driver_backport.validation.build_utils import (
    case_build_env,
    olddefconfig_command,
    parse_kconfig_assignment,
    scripts_config_command,
)
from driver_backport.validation.kconfig_sync import kconfig_index
from driver_backport.validation.validator import run_validation


def _blocking_build_config_issues(validation_report: dict[str, Any]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for phase, issue in validation_phase_issue_pairs(validation_report):
        if not is_build_prep_issue(phase, issue):
            continue
        blockers.append(
            {
                "phase": phase.get("name"),
                "phase_command": phase.get("command"),
                "phase_log_path": phase.get("log_path"),
                "issue": issue,
            }
        )
    return blockers


def _has_build_metadata_repair_evidence(
    spec: CaseSpec,
    validation_report: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> bool:
    metadata_issue_kinds = {
        "kconfig_recursive_dependency",
        "kconfig_symbol_missing",
        "make_no_rule_to_make_target",
        "missing_expected_object",
    }
    metadata_module_reasons = {
        "module_target_missing_from_makefile",
        "makefile_config_not_enabled",
    }
    for blocker in blockers:
        issue = blocker.get("issue") or {}
        issue_kind = str(issue.get("kind") or "")
        if issue_kind in metadata_issue_kinds:
            return True
        diagnosis = issue.get("build_output_diagnosis") or {}
        if str(diagnosis.get("reason") or "") in metadata_module_reasons:
            return True
    assignment_not_enabled = any(
        str((blocker.get("issue") or {}).get("kind") or "")
        == "kconfig_assignment_not_enabled"
        for blocker in blockers
    )
    if assignment_not_enabled:
        old_index = kconfig_index(
            (str(spec.old_root_path.resolve()),)
        )
        donor_index = kconfig_index(
            (str(spec.primary_donor_root.resolve()),)
        )
        requested_symbols = {
            str(config).removeprefix("CONFIG_")
            for config in spec.build_kconfig
        }
        if any(
            symbol not in old_index and symbol in donor_index
            for symbol in requested_symbols
        ):
            return True
    for phase in validation_report.get("phases") or []:
        if not isinstance(phase, dict) or phase.get("name") != "kconfig-sync":
            continue
        closure = phase.get("dependency_closure") or {}
        if closure.get("undefined_symbols"):
            return True
        seed_symbols = {
            str(config).removeprefix("CONFIG_")
            for config in closure.get("seed_configs") or []
        }
        old_index = donor_index = None
        for blocked in closure.get("blocked_dependency_expressions") or []:
            symbol = str(blocked.get("symbol") or "")
            expression = str(blocked.get("expr") or "").strip()
            if not symbol or not expression or symbol in seed_symbols:
                continue
            if old_index is None:
                old_index = kconfig_index((str(spec.old_root_path.resolve()),))
                donor_index = kconfig_index((str(spec.primary_donor_root.resolve()),))
            old_entry = old_index.get(symbol) or {}
            donor_entry = donor_index.get(symbol) or {}
            old_expressions = {
                str(value).strip()
                for key in ("depends", "visibility_depends")
                for value in old_entry.get(key) or []
            }
            donor_expressions = {
                str(value).strip()
                for key in ("depends", "visibility_depends")
                for value in donor_entry.get(key) or []
            }
            if expression in old_expressions and expression not in donor_expressions:
                return True
    return False


def _path_permission(
    spec: CaseSpec,
    path: str,
    *,
    allow_escalation: bool,
    dynamic_editable_files: set[str] | None = None,
) -> str:
    normalized = normalize_relpath(path)
    if allow_escalation and dynamic_editable_files and normalized in dynamic_editable_files:
        return "dynamic_escalation"
    permission = spec.classify_edit_root(path)
    if permission == "editable":
        return "editable"
    if permission == "escalation" and allow_escalation:
        return "escalation"
    return "read_only"


def _ancestor_build_files(spec: CaseSpec) -> list[str]:
    candidates: list[str] = []
    current = Path(spec.driver_root)
    while str(current) not in {"", "."}:
        for name in ("Kconfig", "Makefile", "Kbuild"):
            relative = normalize_relpath((current / name).as_posix())
            if spec.resolve_old(relative).exists() or spec.resolve_donor(relative).exists():
                candidates.append(relative)
        if current.name == "drivers":
            break
        current = current.parent
    return ordered_unique_relpaths(candidates)


def _issue_build_files(validation_report: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for _phase, issue in validation_phase_issue_pairs(validation_report):
        for key in ("file", "path", "relative_path"):
            value = str(issue.get(key) or "").strip()
            if value and is_build_metadata_file(value):
                candidates.append(value)
        for value in issue.get("related_files") or []:
            text = str(value or "").strip()
            if text and is_build_metadata_file(text):
                candidates.append(text)
    return ordered_unique_relpaths(candidates)


def _allowed_build_files(
    spec: CaseSpec,
    validation_report: dict[str, Any],
    *,
    allow_escalation: bool,
    metadata_edits_allowed: bool,
    dynamic_editable_files: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    candidates = ordered_unique_relpaths(
        [
            *_ancestor_build_files(spec),
            *_issue_build_files(validation_report),
        ]
    )
    editable: list[str] = []
    read_only: list[str] = []
    for path in candidates:
        if not is_build_metadata_file(path):
            continue
        permission = _path_permission(
            spec,
            path,
            allow_escalation=allow_escalation,
            dynamic_editable_files=dynamic_editable_files,
        )
        if metadata_edits_allowed and permission in {
            "editable",
            "escalation",
            "dynamic_escalation",
        }:
            editable.append(path)
        else:
            read_only.append(path)
    dynamic_paths = (
        sorted(dynamic_editable_files or set())
        if metadata_edits_allowed
        else []
    )
    for path in dynamic_paths:
        if is_build_metadata_file(path):
            editable.append(path)
    return ordered_unique_relpaths(editable), ordered_unique_relpaths(read_only)


def _tail_text(text: str, *, max_lines: int = 120, max_chars: int = 16000) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = ["... <truncated>"] + lines[-max_lines:]
    rendered = "\n".join(lines)
    if len(rendered) > max_chars:
        rendered = "... <truncated>\n" + rendered[-max_chars:]
    return rendered


def _read_file(path: Path, *, max_lines: int = 260, max_chars: int = 24000) -> str:
    if not path.exists() or not path.is_file():
        return "<missing>"
    return _tail_text(path.read_text(encoding="utf-8", errors="replace"), max_lines=max_lines, max_chars=max_chars)


_KCONFIG_ENTRY_LINE_RE = re.compile(
    r"^\s*(?:menuconfig|config)\s+([A-Za-z0-9_]+)\b"
)
_KCONFIG_SOURCE_LINE_RE = re.compile(
    r"^\s*(?:o?r?source|source)\s+[\"']?([^\"'\s]+)[\"']?"
)
_KCONFIG_SYMBOL_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _build_context_selectors(
    spec: CaseSpec,
    validation_report: dict[str, Any],
    blockers: list[dict[str, Any]],
    files: list[str],
) -> dict[str, Any]:
    symbols: set[str] = set()
    issue_files: set[str] = set()
    causal_files: set[str] = set()
    lines_by_file: dict[str, set[int]] = {}
    target_paths: set[str] = set()

    def add_symbol(value: object) -> None:
        text = str(value or "").strip()
        if not text:
            return
        text = text.split("=", 1)[0].removeprefix("CONFIG_")
        if _KCONFIG_SYMBOL_NAME_RE.fullmatch(text):
            symbols.add(text)

    def add_file(value: object, *, direct: bool = False) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        normalized = normalize_relpath(text)
        if not is_build_metadata_file(normalized):
            return ""
        causal_files.add(normalized)
        if direct:
            issue_files.add(normalized)
        return normalized

    def add_line(path: object, value: object) -> None:
        normalized = add_file(path)
        try:
            line = int(value or 0)
        except (TypeError, ValueError):
            line = 0
        if normalized and line > 0:
            lines_by_file.setdefault(normalized, set()).add(line)

    def add_target(value: object) -> None:
        text = str(value or "").strip()
        if not text:
            return
        normalized = normalize_relpath(text)
        if normalized:
            target_paths.add(normalized)

    for assignment in spec.build_kconfig:
        try:
            name, _value = parse_kconfig_assignment(str(assignment))
        except ValueError:
            name = str(assignment).split("=", 1)[0]
        add_symbol(name)
    for path in [
        *spec.entry_files,
        *spec.expected_modules,
        *spec.expected_objects,
    ]:
        add_target(path)

    for blocker in blockers:
        issue = blocker.get("issue") or {}
        if not isinstance(issue, dict):
            continue
        direct_path = ""
        for key in ("file", "path", "relative_path"):
            if issue.get(key):
                direct_path = add_file(issue.get(key), direct=True)
                add_line(direct_path, issue.get("line"))
        for value in issue.get("related_files") or []:
            add_file(value)
        for value in issue.get("symbols") or []:
            add_symbol(value)
        for key in (
            "symbol",
            "config",
            "guessed_config",
            "requested_config",
        ):
            add_symbol(issue.get(key))
        for key in (
            "target",
            "needed_by",
            "expected_object_path",
            "expected_module_path",
        ):
            add_target(issue.get(key))
        for step in issue.get("dependency_chain") or []:
            if not isinstance(step, dict):
                continue
            add_symbol(step.get("symbol"))
            add_line(step.get("file"), step.get("line"))
        diagnosis = issue.get("build_output_diagnosis") or {}
        if isinstance(diagnosis, dict):
            add_symbol(diagnosis.get("guessed_config"))
            add_target(diagnosis.get("module_stem"))
            makefile = diagnosis.get("makefile") or {}
            if isinstance(makefile, dict):
                add_file(makefile.get("makefile"))
                for membership in makefile.get("obj_membership") or []:
                    if not isinstance(membership, dict):
                        continue
                    add_symbol(membership.get("config"))
                    add_target(membership.get("object"))

    for phase in validation_report.get("phases") or []:
        if not isinstance(phase, dict):
            continue
        closure = phase.get("dependency_closure") or {}
        if not isinstance(closure, dict):
            continue
        for record in closure.get("symbols") or []:
            if isinstance(record, dict):
                add_symbol(record.get("config") or record.get("symbol"))
        for record in closure.get("blocked_dependency_expressions") or []:
            if isinstance(record, dict):
                add_symbol(record.get("symbol"))

    return {
        "symbols": symbols,
        "issue_files": issue_files,
        "causal_files": causal_files,
        "lines_by_file": lines_by_file,
        "target_paths": target_paths,
        "candidate_files": set(files),
    }


def _merge_evidence_ranges(
    ranges: list[tuple[int, int, str]],
) -> list[tuple[int, int, list[str]]]:
    merged: list[tuple[int, int, list[str]]] = []
    for start, end, reason in sorted(ranges):
        if start <= 0 or end < start:
            continue
        if merged and start <= merged[-1][1] + 1:
            old_start, old_end, reasons = merged[-1]
            if reason not in reasons:
                reasons.append(reason)
            merged[-1] = (old_start, max(old_end, end), reasons)
            continue
        merged.append((start, end, [reason]))
    return merged


def _kconfig_evidence_ranges(
    text: str,
    *,
    relative: str,
    selectors: dict[str, Any],
) -> list[tuple[int, int, list[str]]]:
    lines = text.splitlines()
    entries: list[tuple[int, int, str]] = []
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines, start=1):
        match = _KCONFIG_ENTRY_LINE_RE.match(line)
        if match:
            starts.append((index, match.group(1)))
    for position, (start, symbol) in enumerate(starts):
        end = starts[position + 1][0] - 1 if position + 1 < len(starts) else len(lines)
        entries.append((start, end, symbol))

    ranges: list[tuple[int, int, str]] = []
    selected_symbols = selectors["symbols"]
    for start, end, symbol in entries:
        if symbol in selected_symbols:
            ranges.append((start, end, f"kconfig_symbol:{symbol}"))

    for line_number in selectors["lines_by_file"].get(relative, set()):
        containing = next(
            (
                (start, end, symbol)
                for start, end, symbol in entries
                if start <= line_number <= end
            ),
            None,
        )
        if containing is not None:
            start, end, symbol = containing
            ranges.append((start, end, f"diagnostic_line:{line_number}:{symbol}"))
        elif 1 <= line_number <= len(lines):
            ranges.append((line_number, line_number, f"diagnostic_line:{line_number}"))

    relation_paths = {
        *selectors["candidate_files"],
        *selectors["causal_files"],
    }
    for index, line in enumerate(lines, start=1):
        match = _KCONFIG_SOURCE_LINE_RE.match(line)
        if match is None:
            continue
        source_path = normalize_relpath(match.group(1))
        if source_path in relation_paths:
            ranges.append((index, index, f"kconfig_source:{source_path}"))

    if not ranges and relative in selectors["issue_files"]:
        ranges.append((1, len(lines), "unstructured_direct_blocker"))
    return _merge_evidence_ranges(ranges)


def _make_evidence_ranges(
    text: str,
    *,
    relative: str,
    selectors: dict[str, Any],
) -> list[tuple[int, int, list[str]]]:
    lines = text.splitlines()
    parent = Path(relative).parent
    terms = {f"CONFIG_{symbol}" for symbol in selectors["symbols"]}
    for target in selectors["target_paths"]:
        target_path = Path(target)
        terms.update({target_path.name, target_path.stem})
        try:
            local = target_path.relative_to(parent)
        except ValueError:
            continue
        terms.add(local.as_posix())
        if len(local.parts) > 1:
            terms.add(local.parts[0] + "/")
    for candidate in selectors["candidate_files"]:
        candidate_path = Path(candidate)
        if candidate == relative:
            continue
        try:
            local = candidate_path.relative_to(parent)
        except ValueError:
            continue
        if len(local.parts) > 1:
            terms.add(local.parts[0] + "/")

    ranges: list[tuple[int, int, str]] = []
    index = 0
    while index < len(lines):
        start = index
        end = index
        while end < len(lines) - 1 and lines[end].rstrip().endswith("\\"):
            end += 1
        statement = "\n".join(lines[start : end + 1])
        matched = sorted(term for term in terms if term and term in statement)
        if matched:
            ranges.append(
                (start + 1, end + 1, "build_relation:" + ",".join(matched))
            )
        index = end + 1

    for line_number in selectors["lines_by_file"].get(relative, set()):
        if 1 <= line_number <= len(lines):
            ranges.append((line_number, line_number, f"diagnostic_line:{line_number}"))
    if not ranges and relative in selectors["issue_files"]:
        ranges.append((1, len(lines), "unstructured_direct_blocker"))
    return _merge_evidence_ranges(ranges)


def _build_file_evidence(
    path: Path,
    *,
    relative: str,
    selectors: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {
            "exists": False,
            "selection": "missing_file",
            "fragments": [],
        }
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    if Path(relative).name == "Kconfig":
        ranges = _kconfig_evidence_ranges(
            text,
            relative=relative,
            selectors=selectors,
        )
    else:
        ranges = _make_evidence_ranges(
            text,
            relative=relative,
            selectors=selectors,
        )
    lines = text.splitlines()
    fragments = [
        {
            "line_start": start,
            "line_end": end,
            "reasons": reasons,
            "text": "\n".join(lines[start - 1 : end]),
        }
        for start, end, reasons in ranges
    ]
    return {
        "exists": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "line_count": len(lines),
        "selection": (
            "causal_structural_fragments"
            if fragments
            else "no_causal_fragment"
        ),
        "fragments": fragments,
    }


def _phase_log_tails(validation_report: dict[str, Any]) -> list[dict[str, str]]:
    logs: list[dict[str, str]] = []
    for phase in validation_report.get("phases") or []:
        if not isinstance(phase, dict) or phase.get("ok"):
            continue
        log_path = Path(str(phase.get("log_path") or ""))
        if not log_path.exists():
            continue
        logs.append(
            {
                "phase": str(phase.get("name") or ""),
                "log_path": str(log_path.resolve()),
                "tail": _read_file(log_path, max_lines=120, max_chars=16000),
            }
        )
    return logs


def _build_file_context(
    spec: CaseSpec,
    files: list[str],
    *,
    validation_report: dict[str, Any],
    blockers: list[dict[str, Any]],
    allow_escalation: bool,
    dynamic_editable_files: set[str] | None = None,
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    selectors = _build_context_selectors(
        spec,
        validation_report,
        blockers,
        files,
    )
    for relative in files:
        permission = _path_permission(
            spec,
            relative,
            allow_escalation=allow_escalation,
            dynamic_editable_files=dynamic_editable_files,
        )
        old_evidence = _build_file_evidence(
            spec.resolve_old(relative),
            relative=relative,
            selectors=selectors,
        )
        donor_evidence = _build_file_evidence(
            spec.resolve_donor(relative),
            relative=relative,
            selectors=selectors,
        )
        record: dict[str, Any] = {
            "file": relative,
            "permission": permission,
            "old_exists": bool(old_evidence["exists"]),
            "donor_exists": bool(donor_evidence["exists"]),
        }
        if old_evidence == donor_evidence:
            record["shared_old_donor_evidence"] = old_evidence
        else:
            record["old_evidence"] = old_evidence
            record["donor_evidence"] = donor_evidence
        context.append(record)
    return context


def _compact_validation(validation_report: dict[str, Any], blockers: list[dict[str, Any]]) -> dict[str, Any]:
    phases = []
    for phase in validation_report.get("phases") or []:
        if not isinstance(phase, dict):
            continue
        phases.append(
            {
                "name": phase.get("name"),
                "command": phase.get("command"),
                "ok": phase.get("ok"),
                "returncode": phase.get("returncode"),
                "log_path": phase.get("log_path"),
                "issues": phase.get("issues") or [],
            }
        )
    return {
        "overall_ok": validation_report.get("overall_ok"),
        "summary": validation_report.get("summary") or {},
        "phases": phases,
        "build_config_blockers": blockers,
    }


def _build_config_preflight_json_schema() -> dict[str, Any]:
    """Return the strict envelope accepted by the existing apply protocol."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action",
            "reason",
            "patch_plan",
            "patch",
            "config_updates",
            "file_edits",
        ],
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "apply_patch",
                    "apply_config",
                    "apply_patch_and_config",
                    "apply_structured_edits",
                    "apply_structured_edits_and_config",
                    "no_patch",
                    "needs_escalation",
                ],
            },
            "reason": {"type": "string"},
            "patch_plan": {
                "type": "object",
                "additionalProperties": False,
                "required": ["affected_files", "intent"],
                "properties": {
                    "affected_files": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "intent": {"type": "string"},
                },
            },
            "patch": {
                "type": "object",
                "additionalProperties": False,
                "required": ["format", "content"],
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["unified_diff"],
                    },
                    "content": {"type": "string"},
                },
            },
            "config_updates": {
                "type": "object",
                "additionalProperties": False,
                "required": ["assignments", "intent"],
                "properties": {
                    "assignments": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "intent": {"type": "string"},
                },
            },
            "file_edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "file",
                        "operation",
                        "anchor",
                        "content",
                    ],
                    "properties": {
                        "file": {"type": "string"},
                        "operation": {
                            "type": "string",
                            "enum": [
                                "append",
                                "insert_after",
                                "insert_before",
                                "delete_exact",
                                "replace_exact",
                            ],
                        },
                        "anchor": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
    }


def _build_prompt(
    spec: CaseSpec,
    validation_report: dict[str, Any],
    blockers: list[dict[str, Any]],
    editable_files: list[str],
    read_only_files: list[str],
    *,
    allow_escalation: bool,
    metadata_edits_allowed: bool,
    prior_feedback: list[dict[str, Any]] | None = None,
    dynamic_editable_files: set[str] | None = None,
) -> str:
    all_files = ordered_unique_relpaths([*editable_files, *read_only_files])
    payload = {
        "schema": "build_config_preflight_prompt_v1",
        "goal": (
            "Fix only the Kconfig/Makefile/Kbuild closure needed to let the target module enter "
            "the old-kernel build. Do not adapt driver C source APIs in this step."
        ),
        "case": {
            "case_id": spec.case_id,
            "driver_root": spec.driver_root,
            "entry_files": list(spec.entry_files),
            "build_kconfig": list(spec.build_kconfig),
            "expected_modules": list(spec.expected_modules),
            "build_env": dict(spec.build_env),
            "allow_escalation": allow_escalation,
            "metadata_edits_allowed": metadata_edits_allowed,
        },
        "validation": _compact_validation(validation_report, blockers),
        "failed_logs": _phase_log_tails(validation_report),
        "editable_build_files": editable_files,
        "read_only_build_files": read_only_files,
        "dynamically_escalated_build_files": sorted(dynamic_editable_files or set()),
        "file_context": _build_file_context(
            spec,
            all_files,
            validation_report=validation_report,
            blockers=blockers,
            allow_escalation=allow_escalation,
            dynamic_editable_files=dynamic_editable_files,
        ),
        "prior_attempt_feedback": list(prior_feedback or []),
        "hard_constraints": [
            "Return exactly one JSON object. Do not wrap it in prose.",
            (
                "Return every protocol field: action, reason, patch_plan, "
                "patch, config_updates, and file_edits. Use empty strings "
                "or lists for fields unused by the selected action."
            ),
            (
                "The action must name every non-empty operation: use "
                "apply_patch_and_config or "
                "apply_structured_edits_and_config for combined actions; "
                "otherwise leave the unused operation fields empty."
            ),
            "Only patch files listed in editable_build_files.",
            "Only edit Kconfig, Makefile, Kbuild, or .mk files.",
            "Do not edit .c, .h, device-tree, generated files, or config output files.",
            "Do not add driver-specific compatibility source changes in this step.",
            "Use config_updates when the old-kernel .config needs extra dependency symbols enabled or disabled.",
            "config_updates are applied to the case build .config with scripts/config followed by make olddefconfig.",
            "Do not put CONFIG_* assignments in the patch unless the Kconfig/Makefile metadata itself is wrong.",
            (
                "file_context contains exact causal structural fragments with original line numbers and full-file "
                "fingerprints. A no_causal_fragment record grants no evidence about unseen contents; do not infer "
                "or rewrite those contents. shared_old_donor_evidence means the projected evidence is byte-identical "
                "in both trees."
            ),
            "If the editable files are insufficient, return action=needs_escalation and list the missing files.",
            "For needs_escalation, put exact Kconfig/Makefile/Kbuild/.mk paths in patch_plan.affected_files.",
            (
                "For structured edits, operation must be append, "
                "insert_after, insert_before, delete_exact, or "
                "replace_exact; provide an exact unique complete-line "
                "anchor except for append. Use replace_exact for an atomic "
                "replacement instead of a delete followed by an insertion."
            ),
            "Return action=no_patch only when no patch, structured edit, or config update should be applied.",
            "If prior_attempt_feedback reports invalid JSON, return exactly one valid JSON object with no prose.",
            "If prior_attempt_feedback reports patch_dry_run_failed or apply_failed, fix the unified diff format, use apply_structured_edits, or use apply_config instead.",
            *(
                []
                if metadata_edits_allowed
                else [
                    "No structural build-metadata defect was observed. Kconfig, Makefile, Kbuild, and .mk files are read-only in this attempt.",
                    "Return apply_config when exact .config assignments can satisfy the target-tree constraints; otherwise return no_patch with the unsatisfied constraint.",
                ]
            ),
        ],
    }
    rendered_payload, _, _ = render_prompt_payload(payload, indent=2)
    return rendered_payload


def _sanitize_config_assignments(items: Any) -> list[str]:
    if not isinstance(items, list):
        raise ValueError("config_updates.assignments must be a list")
    assignments: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        if "\n" in text or "\r" in text or "\x00" in text:
            raise ValueError(f"invalid Kconfig assignment contains control characters: {text!r}")
        name, value = parse_kconfig_assignment(text)
        if not name.removeprefix("CONFIG_"):
            raise ValueError(f"invalid Kconfig symbol name: {text!r}")
        if any(char.isspace() for char in value):
            raise ValueError(f"invalid Kconfig value contains whitespace: {text!r}")
        normalized = f"{name}={value}"
        if normalized in seen:
            continue
        seen.add(normalized)
        assignments.append(normalized)
    return assignments


def _config_updates_from_payload(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("config_updates") or {}
    if not isinstance(raw, dict):
        raise ValueError("config_updates must be an object")
    return _sanitize_config_assignments(raw.get("assignments") or [])


def _apply_config_updates(
    spec: CaseSpec,
    *,
    assignments: list[str],
    attempt_dir: Path,
) -> dict[str, Any]:
    if not assignments:
        return {"status": "skipped", "reason": "no config assignments"}

    ensure_dir(attempt_dir)
    ensure_dir(spec.build_config_path.parent)
    scripts_config = spec.old_root_path / "scripts" / "config"
    log_path = attempt_dir / "config_updates.log"
    backup_path = attempt_dir / "config.before-build-config-preflight"
    after_path = attempt_dir / "config.after-build-config-preflight"
    config_existed = spec.build_config_path.exists()
    lines: list[str] = [
        f"config_path={spec.build_config_path}",
        "assignments:",
        *[f"  {assignment}" for assignment in assignments],
    ]

    if config_existed:
        backup_path.write_bytes(spec.build_config_path.read_bytes())
    if not scripts_config.exists():
        lines.append(f"missing scripts/config: {scripts_config}")
        log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return {"status": "failed", "reason": "missing_scripts_config", "log_path": str(log_path.resolve())}

    env = case_build_env(spec)
    commands: list[dict[str, Any]] = []

    def restore_config() -> None:
        if config_existed and backup_path.exists():
            spec.build_config_path.write_bytes(backup_path.read_bytes())
        elif not config_existed and spec.build_config_path.exists():
            spec.build_config_path.unlink()

    for assignment in assignments:
        name, value = parse_kconfig_assignment(assignment)
        command = scripts_config_command(
            scripts_config,
            name,
            value,
            config_path=spec.build_config_path if spec.build_dir_path is not None else None,
        )
        process = subprocess.run(
            command,
            cwd=spec.old_root_path,
            env=env,
            text=True,
            capture_output=True,
        )
        commands.append(
            {
                "command": command,
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
        lines.append("$ " + " ".join(command))
        if process.stdout:
            lines.append(process.stdout.rstrip())
        if process.stderr:
            lines.append(process.stderr.rstrip())
        if process.returncode != 0:
            restore_config()
            log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
            return {
                "status": "failed",
                "reason": f"scripts_config_failed_for_{assignment}",
                "assignments": assignments,
                "commands": commands,
                "log_path": str(log_path.resolve()),
            }

    olddefconfig = olddefconfig_command(spec)
    process = subprocess.run(
        olddefconfig,
        cwd=spec.old_root_path,
        env=env,
        text=True,
        capture_output=True,
    )
    commands.append(
        {
            "command": olddefconfig,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    )
    lines.append("$ " + " ".join(olddefconfig))
    if process.stdout:
        lines.append(process.stdout.rstrip())
    if process.stderr:
        lines.append(process.stderr.rstrip())
    if process.returncode != 0:
        restore_config()
        log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return {
            "status": "failed",
            "reason": "olddefconfig_failed",
            "assignments": assignments,
            "commands": commands,
            "log_path": str(log_path.resolve()),
        }

    if spec.build_config_path.exists():
        after_path.write_bytes(spec.build_config_path.read_bytes())
    log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "status": "applied",
        "assignments": assignments,
        "commands": commands,
        "config_before_path": str(backup_path.resolve()) if backup_path.exists() else None,
        "config_after_path": str(after_path.resolve()) if after_path.exists() else None,
        "log_path": str(log_path.resolve()),
    }


def _backup_preflight_file(
    transaction: Any,
    spec: CaseSpec,
    relative_path: str,
    *,
    dynamic_editable_files: set[str] | None,
) -> Path:
    normalized = normalize_relpath(relative_path)
    if not dynamic_editable_files or normalized not in dynamic_editable_files:
        return backup_file(transaction, spec, normalized)

    target = spec.resolve_old(normalized)
    backup_path = transaction.backups_dir / normalized
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    if target.exists():
        backup_path.write_bytes(target.read_bytes())

    payload = transaction.manifest()
    payload["entries"].append(
        {
            "type": "backup",
            "relative_path": normalized,
            "target": str(target),
            "backup": str(backup_path.resolve()),
            "existed": existed,
            "permission": "dynamic_build_metadata_escalation",
        }
    )
    transaction.save_manifest(payload)
    return backup_path


def _parse_preflight_response(response_text: str) -> dict[str, Any]:
    payload = extract_json_payload(response_text)
    if not payload:
        raise ValueError(structured_json_error_summary(response_text))
    action = str(payload.get("action") or "")
    allowed = {
        "apply_patch",
        "apply_config",
        "apply_patch_and_config",
        "apply_structured_edits",
        "apply_structured_edits_and_config",
        "no_patch",
        "needs_escalation",
    }
    if action not in allowed:
        raise ValueError(f"unsupported build/config preflight action: {action}")
    assignments = _config_updates_from_payload(payload)
    file_edits = payload.get("file_edits") or []
    if not isinstance(file_edits, list):
        raise ValueError("file_edits must be a list")
    patch_content = str(
        (payload.get("patch") or {}).get("content") or ""
    ).strip()
    action_uses_patch = action in {
        "apply_patch",
        "apply_patch_and_config",
    }
    action_uses_config = action in {
        "apply_config",
        "apply_patch_and_config",
        "apply_structured_edits_and_config",
    }
    action_uses_structured_edits = action in {
        "apply_structured_edits",
        "apply_structured_edits_and_config",
    }
    payload_contracts = (
        ("patch.content", bool(patch_content), action_uses_patch),
        ("config_updates.assignments", bool(assignments), action_uses_config),
        ("file_edits", bool(file_edits), action_uses_structured_edits),
    )
    for field, present, required in payload_contracts:
        if present != required:
            expectation = "non-empty" if required else "empty"
            raise ValueError(
                f"{field} must be {expectation} for action={action}"
            )
    diff_text = ""
    if action_uses_patch:
        diff_text = diff_from_structured_payload(payload)
        diff_text, _ = repair_unified_diff_structure(diff_text)
    return {"action": action, "payload": payload, "diff_text": diff_text}


def _validate_structured_file_edits(
    spec: CaseSpec,
    payload: dict[str, Any],
    *,
    editable_files: list[str],
    allow_escalation: bool,
    dynamic_editable_files: set[str] | None,
) -> list[dict[str, str]]:
    raw_edits = payload.get("file_edits") or []
    if not isinstance(raw_edits, list):
        raise ValueError("file_edits must be a list")
    editable_set = {normalize_relpath(path) for path in editable_files}
    edits: list[dict[str, str]] = []
    for index, raw in enumerate(raw_edits):
        if not isinstance(raw, dict):
            raise ValueError(f"file_edits[{index}] must be an object")
        relative = normalize_relpath(str(raw.get("file") or ""))
        operation = str(raw.get("operation") or "").strip()
        content = str(raw.get("content") or "")
        anchor = str(raw.get("anchor") or "")
        if operation == "delete_exact" and content.strip():
            raise ValueError(
                f"file_edits[{index}].content must be empty for "
                "delete_exact; use replace_exact to replace the anchor"
            )
        if operation != "delete_exact" and not content.strip():
            continue
        if not relative:
            raise ValueError(f"file_edits[{index}].file is required")
        if relative not in editable_set:
            raise ValueError(f"file_edits[{index}].file is not editable: {relative}")
        if not is_build_metadata_file(relative):
            raise ValueError(f"file_edits[{index}].file is not build metadata: {relative}")
        permission = _path_permission(
            spec,
            relative,
            allow_escalation=allow_escalation,
            dynamic_editable_files=dynamic_editable_files,
        )
        if permission not in {"editable", "escalation", "dynamic_escalation"}:
            raise ValueError(f"file_edits[{index}].file is not permitted: {relative}")
        if operation not in {
            "append",
            "insert_after",
            "insert_before",
            "delete_exact",
            "replace_exact",
        }:
            raise ValueError(f"file_edits[{index}].operation is invalid: {operation}")
        if operation != "append" and not anchor:
            raise ValueError(f"file_edits[{index}].anchor is required for {operation}")
        edits.append(
            {
                "file": relative,
                "operation": operation,
                "anchor": anchor,
                "content": content,
            }
        )
    if not edits:
        raise ValueError("file_edits is empty")
    return edits


def _apply_structured_edit_to_text(
    text: str,
    edit: dict[str, str],
) -> str:
    """Apply one anchor-checked edit without mutating the filesystem."""

    operation = edit["operation"]
    relative = edit["file"]
    content = edit["content"]

    def complete_line_anchor_index(anchor: str) -> int:
        if text.count(anchor) != 1:
            raise ValueError(
                f"{operation} anchor must match once in {relative}: "
                f"{anchor[:120]!r}"
            )
        index = text.index(anchor)
        end = index + len(anchor)
        if index > 0 and text[index - 1] != "\n":
            raise ValueError(
                f"{operation} anchor must start at a line boundary in "
                f"{relative}"
            )
        if (
            not anchor.endswith("\n")
            and end < len(text)
            and text[end] != "\n"
        ):
            raise ValueError(
                f"{operation} anchor must end at a line boundary in "
                f"{relative}"
            )
        return index

    if operation in {"delete_exact", "replace_exact"}:
        anchor = edit["anchor"]
        index = complete_line_anchor_index(anchor)
        end = index + len(anchor)
        replacement = "" if operation == "delete_exact" else content
        if (
            operation == "replace_exact"
            and anchor.endswith("\n")
            and replacement
            and not replacement.endswith("\n")
        ):
            replacement += "\n"
        return text[:index] + replacement + text[end:]

    if operation == "append":
        if not content.endswith("\n"):
            content += "\n"
        prefix = text
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        return prefix + content

    if not content.endswith("\n"):
        content += "\n"
    anchor = edit["anchor"]
    index = complete_line_anchor_index(anchor)
    if operation == "insert_after":
        offset = index + len(anchor)
        if not anchor.endswith("\n"):
            if offset < len(text) and text[offset] == "\n":
                offset += 1
            elif offset == len(text):
                content = "\n" + content
            else:
                raise ValueError(
                    f"insert_after anchor must end at a line boundary in {relative}"
                )
        return text[:offset] + content + text[offset:]

    if operation != "insert_before":
        raise ValueError(f"unsupported structured edit operation: {operation}")
    return text[:index] + content + text[index:]


def _apply_structured_file_edits(
    spec: CaseSpec,
    *,
    state_root: Path,
    attempt: int,
    edits: list[dict[str, str]],
    dynamic_editable_files: set[str] | None,
) -> dict[str, Any]:
    touched_files = ordered_unique_relpaths([edit["file"] for edit in edits])
    # Simulate the complete edit sequence first.  This prevents an invalid
    # later anchor from leaving an earlier edit visible even transiently and
    # also makes repeated edits to one metadata file deterministic.
    staged_text: dict[str, str] = {}
    try:
        for relative_path in touched_files:
            path = spec.resolve_old(relative_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"structured edit target does not exist: {relative_path}"
                )
            staged_text[relative_path] = path.read_text(
                encoding="utf-8", errors="replace"
            )
        for edit in edits:
            relative_path = edit["file"]
            staged_text[relative_path] = _apply_structured_edit_to_text(
                staged_text[relative_path], edit
            )
    except Exception as exc:
        return {
            "status": "structured_edit_failed",
            "error": str(exc),
            "touched_files": touched_files,
            "transaction_dir": None,
            "rollback": None,
        }

    transaction = start_transaction(
        state_root,
        case_id=spec.case_id,
        cycle=0,
        work_unit_id=f"build-config-preflight-structured-{attempt}",
    )
    for relative_path in touched_files:
        _backup_preflight_file(
            transaction,
            spec,
            relative_path,
            dynamic_editable_files=dynamic_editable_files,
        )

    try:
        for relative_path in touched_files:
            spec.resolve_old(relative_path).write_text(
                staged_text[relative_path], encoding="utf-8"
            )
    except Exception as exc:
        finalize_transaction(transaction, status="failed", notes=[str(exc)])
        rollback = rollback_transaction(spec, transaction.root)
        return {
            "status": "structured_edit_failed",
            "error": str(exc),
            "touched_files": touched_files,
            "transaction_dir": str(transaction.root.resolve()),
            "rollback": rollback,
        }

    finalize_transaction(transaction, status="applied", notes=[f"build config structured edit attempt {attempt}"])
    return {
        "status": "applied",
        "touched_files": touched_files,
        "transaction_dir": str(transaction.root.resolve()),
        "edits": edits,
    }


def _apply_preflight_response(
    spec: CaseSpec,
    *,
    state_root: Path,
    attempt: int,
    attempt_dir: Path,
    response_text: str,
    editable_files: list[str],
    allow_escalation: bool,
    dynamic_editable_files: set[str] | None = None,
) -> dict[str, Any]:
    try:
        interpreted = _parse_preflight_response(response_text)
    except Exception as exc:
        return {"status": "invalid_response", "error": str(exc)}

    action = str(interpreted.get("action") or "")
    if action in {"no_patch", "needs_escalation"}:
        return {
            "status": action or "invalid_response",
            "payload": interpreted.get("payload") or {},
        }
    if action not in {
        "apply_patch",
        "apply_config",
        "apply_patch_and_config",
        "apply_structured_edits",
        "apply_structured_edits_and_config",
    }:
        return {"status": "invalid_response", "payload": interpreted.get("payload") or {}}

    applied_actions: list[str] = []
    payload = interpreted.get("payload") or {}
    patch_result: dict[str, Any] | None = None
    metadata_transaction_root: Path | None = None
    diff_text = str(interpreted.get("diff_text") or "")
    touched_files: list[str] = []
    if action in {"apply_patch", "apply_patch_and_config"}:
        touched_files = touched_files_from_diff(diff_text)
        editable_set = {normalize_relpath(path) for path in editable_files}
        disallowed = [
            path
            for path in touched_files
            if normalize_relpath(path) not in editable_set
            or not is_build_metadata_file(path)
            or _path_permission(
                spec,
                path,
                allow_escalation=allow_escalation,
                dynamic_editable_files=dynamic_editable_files,
            )
            not in {"editable", "escalation", "dynamic_escalation"}
        ]
        if disallowed:
            return {
                "status": "disallowed_targets",
                "touched_files": touched_files,
                "disallowed_targets": disallowed,
                "payload": payload,
            }

        dry_run_ok, dry_run_stdout, dry_run_stderr = preflight_patch(
            spec,
            diff_text=diff_text,
            touched_files=touched_files,
        )
        if not dry_run_ok:
            return {
                "status": "patch_dry_run_failed",
                "touched_files": touched_files,
                "stdout": dry_run_stdout,
                "stderr": dry_run_stderr,
                "rebase": {
                    "status": "not_attempted",
                    "reason": (
                        "build metadata patches require current-tree "
                        "context"
                    ),
                },
                "payload": payload,
            }

        transaction = start_transaction(
            state_root,
            case_id=spec.case_id,
            cycle=0,
            work_unit_id=f"build-config-preflight-{attempt}",
        )
        for relative_path in touched_files:
            _backup_preflight_file(
                transaction,
                spec,
                relative_path,
                dynamic_editable_files=dynamic_editable_files,
            )

        returncode, stdout, stderr = run_unified_diff(
            spec.old_root_path,
            diff_text,
        )
        if returncode != 0:
            finalize_transaction(transaction, status="failed", notes=[stdout, stderr])
            rollback = rollback_transaction(spec, transaction.root)
            return {
                "status": "apply_failed",
                "touched_files": touched_files,
                "stdout": stdout,
                "stderr": stderr,
                "transaction_dir": str(transaction.root.resolve()),
                "rollback": rollback,
                "payload": payload,
            }

        finalize_transaction(transaction, status="applied", notes=[f"build config preflight attempt {attempt}"])
        metadata_transaction_root = transaction.root
        applied_actions.append("patch")
        patch_result = {
            "status": "applied",
            "touched_files": touched_files,
            "transaction_dir": str(transaction.root.resolve()),
        }

    structured_result: dict[str, Any] | None = None
    if action in {"apply_structured_edits", "apply_structured_edits_and_config"}:
        try:
            edits = _validate_structured_file_edits(
                spec,
                payload,
                editable_files=editable_files,
                allow_escalation=allow_escalation,
                dynamic_editable_files=dynamic_editable_files,
            )
        except Exception as exc:
            return {"status": "invalid_structured_edits", "error": str(exc), "payload": payload}
        structured_result = _apply_structured_file_edits(
            spec,
            state_root=state_root,
            attempt=attempt,
            edits=edits,
            dynamic_editable_files=dynamic_editable_files,
        )
        if structured_result.get("status") != "applied":
            return {
                "status": "structured_edit_failed",
                "structured_result": structured_result,
                "payload": payload,
            }
        metadata_transaction_root = Path(str(structured_result.get("transaction_dir") or "")).resolve()
        touched_files = ordered_unique_relpaths([*touched_files, *(structured_result.get("touched_files") or [])])
        applied_actions.append("structured_edits")

    config_result: dict[str, Any] | None = None
    if action in {"apply_config", "apply_patch_and_config", "apply_structured_edits_and_config"}:
        try:
            assignments = _config_updates_from_payload(payload)
        except Exception as exc:
            rollback = rollback_transaction(spec, metadata_transaction_root) if metadata_transaction_root else None
            return {"status": "invalid_config_updates", "error": str(exc), "rollback": rollback, "payload": payload}
        if not assignments:
            rollback = rollback_transaction(spec, metadata_transaction_root) if metadata_transaction_root else None
            return {
                "status": "invalid_config_updates",
                "error": "no assignments provided",
                "rollback": rollback,
                "payload": payload,
            }
        config_result = _apply_config_updates(
            spec,
            assignments=assignments,
            attempt_dir=attempt_dir,
        )
        if config_result.get("status") != "applied":
            rollback = rollback_transaction(spec, metadata_transaction_root) if metadata_transaction_root else None
            return {
                "status": "config_apply_failed",
                "config_result": config_result,
                "patch_result": patch_result,
                "structured_result": structured_result,
                "rollback": rollback,
                "payload": payload,
            }
        applied_actions.append("config")

    if not applied_actions:
        return {"status": "invalid_response", "error": "no patch or config update was applied", "payload": payload}

    return {
        "status": "applied",
        "applied_actions": applied_actions,
        "touched_files": touched_files,
        "patch_result": patch_result,
        "structured_result": structured_result,
        "config_result": config_result,
        "payload": payload,
    }


def _requested_escalation_build_files(payload: dict[str, Any]) -> list[str]:
    patch_plan = payload.get("patch_plan") or {}
    if not isinstance(patch_plan, dict):
        return []
    requested: list[str] = []
    for value in patch_plan.get("affected_files") or []:
        text = str(value or "").strip()
        if not text:
            continue
        normalized = normalize_relpath(text)
        if is_build_metadata_file(normalized):
            requested.append(normalized)
    return ordered_unique_relpaths(requested)


def _grant_dynamic_build_metadata_escalation(
    spec: CaseSpec,
    *,
    apply_result: dict[str, Any],
    allow_escalation: bool,
    dynamic_editable_files: set[str],
) -> list[str]:
    if not allow_escalation or apply_result.get("status") != "needs_escalation":
        return []
    payload = apply_result.get("payload") or {}
    if not isinstance(payload, dict):
        return []
    granted: list[str] = []
    for path in _requested_escalation_build_files(payload):
        if path in dynamic_editable_files:
            continue
        if not is_build_metadata_file(path):
            continue
        if not spec.resolve_old(path).exists() and not spec.resolve_donor(path).exists():
            continue
        dynamic_editable_files.add(path)
        granted.append(path)
    return granted


def run_build_config_preflight(
    spec: CaseSpec,
    *,
    state_root: Path,
    model: str | None,
    max_completion_tokens: int,
    allow_escalation: bool,
    max_attempts: int = 2,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    preflight_root = ensure_dir(state_root / "build_config_preflight")
    attempts: list[dict[str, Any]] = []
    prior_feedback: list[dict[str, Any]] = []
    dynamic_editable_files: set[str] = set()
    max_attempts = max(0, int(max_attempts))

    def notify(message: str) -> None:
        if progress:
            progress(message)

    for attempt in range(0, max_attempts + 1):
        attempt_dir = ensure_dir(preflight_root / f"attempt-{attempt:02d}")
        validation_dir = ensure_dir(attempt_dir / "validation")
        notify(f"build/config preflight validation attempt={attempt} log_root={validation_dir}")
        validation = run_validation(
            spec,
            artifacts_dir=validation_dir,
            build_preparation_only=True,
            progress=progress,
        )
        validation_path = attempt_dir / "validation_report.json"
        write_json(validation_path, validation)
        blockers = _blocking_build_config_issues(validation)
        metadata_edits_allowed = _has_build_metadata_repair_evidence(
            spec,
            validation,
            blockers,
        )
        editable_files, read_only_files = _allowed_build_files(
            spec,
            validation,
            allow_escalation=allow_escalation,
            metadata_edits_allowed=metadata_edits_allowed,
            dynamic_editable_files=dynamic_editable_files,
        )
        attempt_record: dict[str, Any] = {
            "attempt": attempt,
            "validation_path": str(validation_path.resolve()),
            "validation_overall_ok": bool(validation.get("overall_ok")),
            "validation_summary": validation.get("summary") or {},
            "blocker_count": len(blockers),
            "blockers": blockers,
            "metadata_edits_allowed": metadata_edits_allowed,
            "editable_build_files": editable_files,
            "read_only_build_files": read_only_files,
        }
        attempts.append(attempt_record)

        if validation.get("overall_ok"):
            result = {
                "status": "validated",
                "ok": True,
                "reason": "build/config preflight validation passed",
                "attempts": attempts,
                "final_validation_path": str(validation_path.resolve()),
            }
            write_json(preflight_root / "result.json", result)
            return result
        if not blockers:
            result = {
                "status": "entered_compile",
                "ok": True,
                "reason": "validation reached non-build/config diagnostics; source migration can start",
                "attempts": attempts,
                "final_validation_path": str(validation_path.resolve()),
            }
            write_json(preflight_root / "result.json", result)
            return result
        if attempt >= max_attempts:
            break
        prompt = _build_prompt(
            spec,
            validation,
            blockers,
            editable_files,
            read_only_files,
            allow_escalation=allow_escalation,
            metadata_edits_allowed=metadata_edits_allowed,
            prior_feedback=prior_feedback,
            dynamic_editable_files=dynamic_editable_files,
        )
        prompt_path = attempt_dir / "build_config_preflight.prompt.json"
        prompt_path.write_text(prompt, encoding="utf-8")
        notify(f"build/config preflight LLM attempt={attempt} prompt={prompt_path}")
        try:
            response = gpt_api_with_metadata(
                prompt,
                model=model,
                max_completion_tokens=max_completion_tokens,
                response_schema=_build_config_preflight_json_schema(),
                response_schema_name="build_config_preflight",
            )
            response_text = str(response.get("text") or "")
            response_path = attempt_dir / "build_config_preflight.response.txt"
            response_path.write_text(response_text, encoding="utf-8")
            write_json(
                attempt_dir / "build_config_preflight.response_meta.json",
                {
                    "model": response.get("model"),
                    "api_mode": response.get("api_mode"),
                    "response_id": response.get("response_id"),
                    "usage": response.get("usage") or {},
                    "cost": response.get("cost") or {},
                },
            )
        except Exception as exc:
            attempt_record["llm"] = {
                "status": "failed",
                "error": str(exc),
            }
            break

        apply_result = _apply_preflight_response(
            spec,
            state_root=state_root,
            attempt=attempt,
            attempt_dir=attempt_dir,
            response_text=response_text,
            editable_files=editable_files,
            allow_escalation=allow_escalation,
            dynamic_editable_files=dynamic_editable_files,
        )
        granted_escalation = _grant_dynamic_build_metadata_escalation(
            spec,
            apply_result=apply_result,
            allow_escalation=allow_escalation,
            dynamic_editable_files=dynamic_editable_files,
        )
        attempt_record["llm"] = {
            "status": "completed",
            "prompt_path": str(prompt_path.resolve()),
            "response_path": str(response_path.resolve()),
            "apply_result": apply_result,
            "granted_dynamic_build_metadata_escalation": granted_escalation,
        }
        if apply_result.get("status") != "applied":
            prior_feedback.append(
                {
                    "attempt": attempt,
                    "status": apply_result.get("status"),
                    "granted_dynamic_build_metadata_escalation": granted_escalation,
                    "error": apply_result.get("error"),
                    "reason": apply_result.get("reason"),
                    "stderr": _tail_text(str(apply_result.get("stderr") or ""), max_lines=40, max_chars=8000),
                    "stdout": _tail_text(str(apply_result.get("stdout") or ""), max_lines=20, max_chars=4000),
                    "disallowed_targets": apply_result.get("disallowed_targets") or [],
                    "touched_files": apply_result.get("touched_files") or [],
                    "config_result": apply_result.get("config_result") or {},
                    "structured_result": apply_result.get("structured_result") or {},
                    "instruction": (
                        "The previous build/config preflight response was not applied. "
                        "Return a corrected JSON response. Prefer apply_config for config visibility gates; "
                        "use apply_patch or apply_structured_edits only when build metadata is structurally wrong. "
                        "If dynamic escalation was granted, the next prompt includes those exact files as editable."
                    ),
                }
            )
            continue

    metadata_edits_required = any(
        bool(item.get("metadata_edits_allowed"))
        for item in attempts
    )
    result = {
        "status": "blocked",
        "ok": False,
        "reason": (
            "build/config preflight still has build metadata blockers"
            if metadata_edits_required
            else "target configuration constraints remain unsatisfied"
        ),
        "failure_domain": (
            "build_metadata"
            if metadata_edits_required
            else "config_environment"
        ),
        "environment_failure": not metadata_edits_required,
        "attempts": attempts,
        "completed_at": utc_now(),
    }
    write_json(preflight_root / "result.json", result)
    return result
