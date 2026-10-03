#!/usr/bin/env python3
from __future__ import annotations

import difflib
import hashlib
import json
import re
import shutil
import tempfile
from bisect import bisect_left
from pathlib import Path
from typing import Any

from driver_backport.analysis.code_index import (
    CodeIndex,
    parse_module_metadata_calls,
    parse_include_directives,
    parse_structural_entity_file,
)
from driver_backport.analysis.entity_resolution import (
    entity_contract_key,
    entity_kind,
    resolve_entity_in_case_tree,
)
from driver_backport.analysis.entity_integration import (
    INTEGRATION_ANALYSIS_REVISION,
    analyze_post_patch_integration,
    semantic_integration_requirements_from_unit,
)
from driver_backport.analysis.export_index import parse_export_contracts
from driver_backport.core.case_spec import CaseSpec
from driver_backport.core.common import (
    cycle_hint_from_path,
    ensure_dir,
    normalize_relpath,
    stable_id,
    write_json,
)
from driver_backport.analysis.provider_contract import validate_provider_contract_patch
from driver_backport.core.executor import (
    backup_file,
    finalize_transaction,
    rollback_transaction,
    run_unified_diff,
    start_transaction,
)
from driver_backport.planning.apply_protocol import (
    ResponseInterpretationError,
    build_apply_failure,
    extract_json_object,
)
from driver_backport.planning.patch_transaction import build_patch_transaction_preview
from driver_backport.planning.plan_store import write_repair_plan
from driver_backport.planning.prompting import repair_plan_units
from driver_backport.planning.root_binding import graph_root_binding_key
from driver_backport.planning.sanity_check import (
    HUNK_HEADER_RE,
    build_sanity_report,
    normalize_diff_path as _normalize_diff_path,
)
from driver_backport.planning.execution_phase import (
    EXECUTION_PHASE_COMPILER_REPAIR,
    semantic_controls_enabled,
    unit_execution_phase,
)


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_BLOCK_RE = re.compile(r"```(?:diff|patch)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PATCH_CONFLICT_MAX_HUNKS = 12
PATCH_CONFLICT_MAX_EXCERPT_LINES = 120


def _token_pastes_cannot_form_identifier(source: bytes, identifier: bytes) -> bool:
    try:
        from tree_sitter_languages import get_parser

        root = get_parser("c").parse(source).root_node
    except (ImportError, RuntimeError, ValueError):
        return False
    remaining = bytearray(source)
    pending = [root]
    chain_pattern = re.compile(
        rb"[A-Za-z_][A-Za-z0-9_]*(?:[ \t]*##[ \t]*[A-Za-z_][A-Za-z0-9_]*)+"
    )
    while pending:
        node = pending.pop()
        if str(node.type) in {"preproc_def", "preproc_function_def"}:
            macro = source[node.start_byte:node.end_byte]
            match = re.match(
                rb"[ \t]*#[ \t]*define[ \t]+[A-Za-z_][A-Za-z0-9_]*"
                rb"(?:\(([^)]*)\))?",
                macro,
            )
            if match is None:
                return False
            parameters: set[bytes] = set()
            variadic_parameters: set[bytes] = set()
            for part in (match.group(1) or b"").split(b","):
                parameter = part.strip()
                if not parameter:
                    continue
                if parameter == b"...":
                    parameters.add(b"__VA_ARGS__")
                    variadic_parameters.add(b"__VA_ARGS__")
                    continue
                if parameter.endswith(b"..."):
                    parameter = parameter[:-3].strip()
                    variadic_parameters.add(parameter)
                if re.fullmatch(rb"[A-Za-z_][A-Za-z0-9_]*", parameter) is None:
                    return False
                parameters.add(parameter)
            body = re.sub(
                rb",[ \t]*##[ \t]*([A-Za-z_][A-Za-z0-9_]*)",
                lambda item: (
                    b" " * len(item.group())
                    if item.group(1) in variadic_parameters
                    else item.group()
                ),
                macro[match.end():],
            )
            chains = list(chain_pattern.finditer(body))
            unclassified = bytearray(body)
            for chain in chains:
                operands = re.split(rb"[ \t]*##[ \t]*", chain.group())
                # Unknown arguments cannot erase a fixed operand of a paste.
                possible_identifier = b"".join(
                    b".*" if operand in parameters else re.escape(operand)
                    for operand in operands
                )
                if re.fullmatch(possible_identifier, identifier):
                    return False
                unclassified[chain.start():chain.end()] = b" " * (
                    chain.end() - chain.start()
                )
            if b"##" in unclassified:
                return False
            remaining[node.start_byte:node.end_byte] = b" " * (
                node.end_byte - node.start_byte
            )
        else:
            pending.extend(node.children)
    return b"##" not in remaining


def _exact_declaration_absent_from_header(
    spec: CaseSpec, relative_path: str, symbol: str,
) -> bool:
    """Prove a header has no exact declaration without mistaking calls for one."""

    if (
        Path(relative_path).suffix.lower() != ".h"
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol) is None
    ):
        return False
    try:
        source = spec.resolve_old(relative_path).read_bytes()
    except (OSError, ValueError):
        return False
    if not isinstance(source, bytes):
        return False
    identifier = symbol.encode("ascii")
    occurrences = list(re.finditer(
        rb"(?<![A-Za-z0-9_])" + re.escape(identifier) + rb"(?![A-Za-z0-9_])",
        source,
    ))
    if any(marker in source for marker in (b"%:%:", b"??=", b"\x00")):
        return False
    if b"##" in source and not _token_pastes_cannot_form_identifier(
        source, identifier
    ):
        return False
    if not occurrences:
        return True
    try:
        from tree_sitter_languages import get_parser

        parser = get_parser("c")
        root = parser.parse(source).root_node
    except (ImportError, RuntimeError, ValueError):
        return False
    call_positions: set[tuple[int, int]] = set()
    pending = [(root, source, 0)]
    while pending:
        node, content, offset = pending.pop()
        if str(node.type) == "preproc_arg":
            replacement = content[node.start_byte:node.end_byte]
            if identifier in replacement:
                # Keep byte positions stable when parsing opaque macro bodies.
                replacement = re.sub(rb"\\(?=\r?\n)", b" ", replacement)
                prefix = b"void driver_backport_macro_body(void) {\n"
                wrapped = prefix + replacement + b";\n}\n"
                replacement_root = parser.parse(wrapped).root_node
                if replacement_root.has_error:
                    return False
                pending.append((
                    replacement_root, wrapped,
                    offset + node.start_byte - len(prefix),
                ))
            continue
        if (
            str(node.type) == "identifier"
            and content[node.start_byte:node.end_byte] == identifier
        ):
            parent = node.parent
            function = (
                parent.child_by_field_name("function")
                if parent is not None and str(parent.type) == "call_expression"
                else None
            )
            if (
                function is None
                or function.start_byte != node.start_byte
                or function.end_byte != node.end_byte
                or bool(getattr(parent, "has_error", False))
            ):
                return False
            ancestor = node
            while ancestor is not None and ancestor is not root:
                if str(ancestor.type) == "ERROR" or bool(
                    getattr(ancestor, "is_missing", False)
                ):
                    return False
                ancestor = ancestor.parent
            call_positions.add((offset + node.start_byte, offset + node.end_byte))
        pending.extend((child, content, offset) for child in node.children)
    return call_positions == {
        (match.start(), match.end()) for match in occurrences
    }


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _unit_root_binding(unit: dict[str, Any]) -> dict[str, str]:
    decision = (
        unit.get("repair_decision")
        if isinstance(unit.get("repair_decision"), dict)
        else {}
    )
    root = (
        unit.get("code_graph_root_cause_cluster")
        if isinstance(unit.get("code_graph_root_cause_cluster"), dict)
        else unit.get("root_cause_cluster")
        if isinstance(unit.get("root_cause_cluster"), dict)
        else {}
    )
    return {
        "work_unit_id": str(unit.get("work_unit_id") or unit.get("id") or ""),
        "root_node_id": str(
            decision.get("root_node_id")
            or root.get("node_id")
            or root.get("entity_id")
            or ""
        ),
        "root_key": str(
            decision.get("root_key")
            or graph_root_binding_key(root)
            or root.get("contract_key")
            or ""
        ),
    }


def _unit_source_preimage_digest(
    spec: CaseSpec,
    unit: dict[str, Any],
    touched_files: list[str],
) -> str:
    records: list[dict[str, Any]] = []
    paths = sorted(
        {
            normalize_relpath(path)
            for path in touched_files
            if normalize_relpath(path)
        }
    )
    for relative_path in paths:
        path = spec.resolve_old(relative_path)
        if path.is_file():
            content_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(
                {
                    "file": relative_path,
                    "exists": True,
                    "content_digest": content_digest,
                }
            )
        else:
            records.append({"file": relative_path, "exists": False})
    if not records:
        source_pack = (
            unit.get("repair_source_pack")
            if isinstance(unit.get("repair_source_pack"), dict)
            else {}
        )
        for role in (
            "allowed_edit_sources",
            "donor_reference_sources",
            "read_only_evidence_sources",
        ):
            for record in source_pack.get(role) or []:
                if not isinstance(record, dict):
                    continue
                records.append(
                    {
                        "role": role,
                        "tree": record.get("tree"),
                        "file": record.get("file"),
                        "entity_id": (
                            record.get("entity_id")
                            or record.get("code_graph_node_id")
                            or record.get("node_id")
                        ),
                        "line_start": record.get("line_start"),
                        "line_end": record.get("line_end"),
                        "content_digest": hashlib.sha256(
                            str(record.get("text") or "").encode("utf-8")
                        ).hexdigest(),
                    }
                )
    return _sha256_json(records)


def _structural_parse_reports_for_files(
    spec: CaseSpec,
    touched_files: list[str],
) -> dict[str, dict[str, Any]]:
    """Capture bounded syntax state for source files in one patch transaction."""

    reports: dict[str, dict[str, Any]] = {}
    for relative_path in sorted(
        {
            normalize_relpath(path)
            for path in touched_files
            if normalize_relpath(path)
        }
    ):
        if Path(relative_path).suffix.lower() not in {
            ".c",
            ".h",
            ".cc",
            ".cpp",
            ".hpp",
        }:
            continue
        reports[relative_path] = parse_structural_entity_file(
            root=spec.old_root_path,
            relative_path=relative_path,
            tree="old",
        )
    return reports


def _structural_parse_postcondition(
    spec: CaseSpec,
    *,
    touched_files: list[str],
    before_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Reject a patch that makes a previously clean source file malformed.

    This is an incremental transaction postcondition, not another compile
    phase. It intentionally leaves pre-existing parse errors to normal
    compiler recovery, while preventing one unit from introducing a new
    malformed declaration boundary that would cascade into other units.
    """

    after_reports = _structural_parse_reports_for_files(spec, touched_files)
    files: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    unavailable = 0
    for relative_path in sorted(set(before_reports) | set(after_reports)):
        before = dict(before_reports.get(relative_path) or {})
        after = dict(after_reports.get(relative_path) or {})
        before_available = bool(before.get("parse_ok"))
        after_available = bool(after.get("parse_ok"))
        before_has_errors = bool(before.get("has_error_nodes"))
        after_has_errors = bool(after.get("has_error_nodes"))
        entry = {
            "file": relative_path,
            "before_parse_ok": before_available,
            "after_parse_ok": after_available,
            "before_has_error_nodes": before_has_errors,
            "after_has_error_nodes": after_has_errors,
        }
        if not before_available or not after_available:
            unavailable += 1
            entry["status"] = "unavailable"
            entry["before_parse_error"] = before.get("parse_error")
            entry["after_parse_error"] = after.get("parse_error")
        elif not before_has_errors and after_has_errors:
            entry["status"] = "regressed"
            entry["after_error_nodes"] = list(after.get("error_nodes") or [])
            regressions.append(entry)
        else:
            entry["status"] = "preserved"
        files.append(entry)
    return {
        "schema": "transaction_structural_parse_postcondition_v1",
        "status": "regressed" if regressions else "preserved",
        "files": files,
        "regressions": regressions,
        "unavailable_file_count": unavailable,
    }


def _unit_evidence_set_digest(unit: dict[str, Any]) -> tuple[str, list[str]]:
    evidence_ids: list[str] = []
    source_pack = (
        unit.get("repair_source_pack")
        if isinstance(unit.get("repair_source_pack"), dict)
        else {}
    )
    for role in (
        "allowed_edit_sources",
        "donor_reference_sources",
        "read_only_evidence_sources",
    ):
        for record in source_pack.get(role) or []:
            if not isinstance(record, dict):
                continue
            evidence_id = str(
                record.get("id")
                or record.get("entity_id")
                or record.get("code_graph_node_id")
                or record.get("node_id")
                or ""
            )
            if evidence_id:
                evidence_ids.append(evidence_id)
    evidence_ids = list(dict.fromkeys(evidence_ids))
    return _sha256_json(sorted(evidence_ids)), evidence_ids


def _unit_diagnostic_set_digest(unit: dict[str, Any]) -> str:
    signatures = [
        str(item)
        for item in unit.get("diagnostic_signatures") or []
        if str(item)
    ]
    if not signatures:
        signatures = [
            str(item)
            for item in unit.get("diagnostic_event_ids") or []
            if str(item)
        ]
    return _sha256_json(sorted(dict.fromkeys(signatures)))


def _attach_apply_failure(
    spec: CaseSpec,
    unit: dict[str, Any],
    result: dict[str, Any],
    *,
    stage: str,
    recoverability: str,
    finding_kind: str,
    touched_files: list[str] | None = None,
    patch_signature: str | None = None,
    deterministic_attempts: list[dict[str, Any]] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_files = list(touched_files or result.get("touched_files") or [])
    evidence_set_digest, evidence_ids = _unit_evidence_set_digest(unit)
    root_binding = _unit_root_binding(unit)
    semantic_action_id = (
        stable_id(
            "semantic-action",
            root_binding.get("work_unit_id") or "",
            root_binding.get("root_node_id") or root_binding.get("root_key") or "",
            patch_signature or "",
        )
        if patch_signature
        else None
    )
    return {
        **result,
        "apply_failure": build_apply_failure(
            stage=stage,
            recoverability=recoverability,
            finding_kind=finding_kind,
            unit_id=str(unit.get("id") or ""),
            work_unit_id=str(
                unit.get("work_unit_id") or unit.get("id") or ""
            ),
            root_binding=root_binding,
            semantic_action_id=semantic_action_id,
            source_preimage_digest=_unit_source_preimage_digest(
                spec,
                unit,
                local_files,
            ),
            evidence_set_digest=evidence_set_digest,
            diagnostic_set_digest=_unit_diagnostic_set_digest(unit),
            patch_signature=patch_signature,
            deterministic_attempts=deterministic_attempts,
            evidence_ids=evidence_ids,
            details=details,
        ),
    }


def extract_json_payload(text: str) -> dict[str, Any] | None:
    return extract_json_object(text, required_key="action")


def structured_json_error_summary(text: str) -> str:
    candidates = [text.strip()]
    for match in JSON_BLOCK_RE.finditer(text):
        candidates.append(match.group(1).strip())

    errors: list[str] = []
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate, strict=False)
        except json.JSONDecodeError as exc:
            errors.append(f"{exc.msg} at line {exc.lineno} column {exc.colno}")
            continue
        if not isinstance(payload, dict):
            errors.append("top-level JSON value is not an object")
        elif not payload.get("action"):
            errors.append("top-level JSON object is missing action")

    if errors:
        return "; ".join(errors[:3])
    return "no JSON object with top-level action was found"


def _extract_raw_unified_diff(text: str) -> str | None:
    candidates = [text.strip()]
    candidates.extend(match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text))

    for candidate in candidates:
        lines = candidate.splitlines()
        start_index = None
        for index, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("diff --git ", "--- ", "*** Begin Patch")):
                start_index = index
                break
        if start_index is None:
            continue
        diff_text = "\n".join(lines[start_index:]).strip()
        if not diff_text:
            continue
        return diff_text if diff_text.endswith("\n") else diff_text + "\n"
    return None


def diff_from_structured_payload(payload: dict[str, Any]) -> str:
    patch = dict(payload.get("patch") or {})
    if patch.get("format") != "unified_diff":
        raise ValueError("structured response patch.format must be unified_diff")
    content_lines = patch.get("content_lines")
    if isinstance(content_lines, list):
        content = "\n".join(_normalize_structured_diff_line(str(line).rstrip("\n")) for line in content_lines)
    else:
        content = _normalize_structured_diff_line(str(patch.get("content") or ""))
    if not content.strip():
        raise ValueError("structured response patch.content_lines is empty")
    if not any(content.lstrip().startswith(prefix) for prefix in ("diff --git ", "--- ", "*** ")):
        raise ValueError("structured response patch content does not look like a unified diff")
    return content if content.endswith("\n") else content + "\n"


def _normalize_structured_diff_line(line: str) -> str:
    # Models sometimes double-escape tabs inside JSON content_lines, producing
    # literal "\\t" in the generated patch. patch(1) treats those as ordinary
    # characters, so context lines no longer match and some hunks become
    # malformed. Only normalize common patch-line escapes; do not interpret the
    # whole string as a Python/JSON escape sequence.
    return line.replace("\\t", "\t").replace("\\r", "\r")


def touched_files_from_diff(diff_text: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    lines = diff_text.splitlines()
    for index, line in enumerate(lines[:-1]):
        if not line.startswith("--- ") or not lines[index + 1].startswith(
            "+++ "
        ):
            continue
        old_path = line[4:].strip().split("\t", 1)[0]
        new_path = lines[index + 1][4:].strip().split("\t", 1)[0]
        path = old_path if new_path == "/dev/null" else new_path
        if path == "/dev/null":
            continue
        normalized = _normalize_diff_path(path)
        if normalized not in seen:
            seen.add(normalized)
            files.append(normalized)
    if not files:
        for line in diff_text.splitlines():
            if line.startswith(("--- ", "+++ ")):
                path = line[4:].strip().split("\t", 1)[0]
                if path == "/dev/null":
                    continue
                normalized = _normalize_diff_path(path)
                if normalized not in seen:
                    seen.add(normalized)
                    files.append(normalized)
    return files


def _materialize_patch_targets(spec: CaseSpec, touched_files: list[str], temp_root: Path) -> None:
    for relative_path in touched_files:
        source = spec.resolve_old(relative_path)
        target = temp_root / normalize_relpath(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copyfile(source, target)


def preflight_patch(
    spec: CaseSpec,
    *,
    diff_text: str,
    touched_files: list[str],
    reverse: bool = False,
) -> tuple[bool, str, str]:
    """Dry-run a complete patch in either direction on a target-tree copy."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir).resolve()
        _materialize_patch_targets(spec, touched_files, temp_root)
        returncode, stdout, stderr = run_unified_diff(
            temp_root,
            diff_text,
            dry_run=True,
            reverse=reverse,
        )
    return returncode == 0, stdout, stderr


def _split_unified_diff_records(diff_text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_hunk: list[str] | None = None
    lines = diff_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("--- ") and index + 1 < len(lines) and lines[index + 1].startswith("+++ "):
            old_path = _normalize_diff_path(line[4:].strip())
            new_path = _normalize_diff_path(lines[index + 1][4:].strip())
            relative_path = new_path if new_path != "/dev/null" else old_path
            current = {
                "file": relative_path,
                "old_header": line,
                "new_header": lines[index + 1],
                "hunks": [],
            }
            records.append(current)
            current_hunk = None
            index += 2
            continue
        if current is not None and line.startswith("@@ "):
            current_hunk = [line]
            current["hunks"].append(current_hunk)
            index += 1
            continue
        if current_hunk is not None:
            if line.startswith("diff --git ") or (
                line.startswith("--- ") and index + 1 < len(lines) and lines[index + 1].startswith("+++ ")
            ):
                current_hunk = None
                continue
            current_hunk.append(line)
        index += 1
    return records


def _render_unified_diff_record(record: dict[str, Any]) -> str | None:
    old_header = str(record.get("old_header") or "")
    new_header = str(record.get("new_header") or "")
    hunks = [
        list(hunk)
        for hunk in record.get("hunks") or []
        if isinstance(hunk, list)
    ]
    if not old_header or not new_header or not hunks:
        return None
    lines = [old_header, new_header]
    for hunk in hunks:
        lines.extend(str(line) for line in hunk)
    return "\n".join(lines) + "\n"


def _hunk_count(match: re.Match[str], name: str) -> int:
    raw = match.group(name)
    return int(raw) if raw is not None else 1


def _validate_unified_diff_structure(diff_text: str) -> tuple[bool, str | None]:
    records = _split_unified_diff_records(diff_text)
    if not records:
        return False, "unified_diff_structure: no file records found"

    errors: list[str] = []
    for record in records:
        relative_path = normalize_relpath(str(record.get("file") or ""))
        hunks = list(record.get("hunks") or [])
        if not hunks:
            errors.append(f"{relative_path or '<unknown>'}: no hunks found")
            continue
        for hunk_index, hunk in enumerate(hunks, start=1):
            if not hunk:
                errors.append(f"{relative_path}: hunk {hunk_index} is empty")
                continue
            header = str(hunk[0])
            match = HUNK_HEADER_RE.match(header)
            if not match:
                errors.append(f"{relative_path}: hunk {hunk_index} has invalid header: {header}")
                continue
            expected_old = _hunk_count(match, "old_count")
            expected_new = _hunk_count(match, "new_count")
            actual_old = 0
            actual_new = 0
            for line_index, line in enumerate(hunk[1:], start=2):
                if line.startswith("\\"):
                    continue
                if not line:
                    errors.append(
                        f"{relative_path}: hunk {hunk_index} line {line_index} is missing a unified-diff prefix"
                    )
                    continue
                prefix = line[0]
                if prefix == " ":
                    actual_old += 1
                    actual_new += 1
                elif prefix == "-":
                    actual_old += 1
                elif prefix == "+":
                    actual_new += 1
                else:
                    sample = line[:80].replace("\t", "\\t")
                    errors.append(
                        f"{relative_path}: hunk {hunk_index} line {line_index} has invalid prefix "
                        f"{prefix!r}: {sample}"
                    )
            if actual_old != expected_old or actual_new != expected_new:
                errors.append(
                    f"{relative_path}: hunk {hunk_index} line counts mismatch "
                    f"(header old/new={expected_old}/{expected_new}, body old/new={actual_old}/{actual_new})"
                )
    if errors:
        return False, "unified_diff_structure: " + " | ".join(errors[:8])
    return True, None


def _repair_unified_diff_missing_hunk_ranges(
    diff_text: str,
) -> tuple[str, dict[str, Any] | None]:
    """Give range-less ``@@`` hunks a valid placeholder for exact rebasing."""

    lines = diff_text.splitlines()
    repaired = list(lines)
    in_file_record = False
    repaired_hunks = 0
    for index, line in enumerate(lines):
        if (
            line.startswith("--- ")
            and index + 1 < len(lines)
            and lines[index + 1].startswith("+++ ")
        ):
            in_file_record = True
            continue
        if line.startswith("diff --git "):
            in_file_record = False
            continue
        if in_file_record and line.strip() == "@@":
            # The real line is established later by the exact-context rebaser.
            # Counts are corrected mechanically after missing prefixes are fixed.
            # Line zero marks the location as unknown. The relocator may then
            # accept only one exact/normalized context match; it must not use a
            # fabricated line number to choose between ambiguous matches.
            repaired[index] = "@@ -0,1 +0,1 @@"
            repaired_hunks += 1
    if not repaired_hunks:
        return diff_text, None
    repaired_text = "\n".join(repaired)
    if diff_text.endswith("\n"):
        repaired_text += "\n"
    return repaired_text, {
        "kind": "unified_diff_missing_hunk_range_repair",
        "repaired_hunk_count": repaired_hunks,
        "placement": "placeholder_for_exact_context_rebase",
    }


def _repair_unified_diff_hunk_counts(diff_text: str) -> tuple[str, dict[str, Any] | None]:
    valid, error = _validate_unified_diff_structure(diff_text)
    if valid or not error:
        return diff_text, None
    error_parts = str(error).removeprefix("unified_diff_structure: ").split(" | ")
    if not error_parts or any("line counts mismatch" not in item for item in error_parts):
        return diff_text, None

    lines = diff_text.splitlines()
    repaired = list(lines)
    repaired_hunks = 0
    for index, line in enumerate(lines):
        match = HUNK_HEADER_RE.match(line)
        if not match:
            continue
        actual_old = 0
        actual_new = 0
        cursor = index + 1
        while cursor < len(lines):
            body_line = lines[cursor]
            if body_line.startswith("@@ ") or body_line.startswith("diff --git "):
                break
            if (
                body_line.startswith(("--- a/", "--- /dev/null"))
                and cursor + 1 < len(lines)
                and lines[cursor + 1].startswith(("+++ b/", "+++ /dev/null"))
            ):
                break
            if body_line.startswith("\\"):
                cursor += 1
                continue
            if not body_line or body_line[0] not in {" ", "+", "-"}:
                return diff_text, None
            if body_line[0] in {" ", "-"}:
                actual_old += 1
            if body_line[0] in {" ", "+"}:
                actual_new += 1
            cursor += 1
        expected_old = _hunk_count(match, "old_count")
        expected_new = _hunk_count(match, "new_count")
        if (expected_old, expected_new) == (actual_old, actual_new):
            continue
        repaired[index] = (
            f"@@ -{match.group('old_start')},{actual_old} "
            f"+{match.group('new_start')},{actual_new} @@"
            f"{line[match.end():]}"
        )
        repaired_hunks += 1

    if not repaired_hunks:
        return diff_text, None
    repaired_text = "\n".join(repaired)
    if diff_text.endswith("\n"):
        repaired_text += "\n"
    repaired_valid, _ = _validate_unified_diff_structure(repaired_text)
    if not repaired_valid:
        return diff_text, None
    return repaired_text, {
        "kind": "unified_diff_hunk_count_repair",
        "repaired_hunk_count": repaired_hunks,
        "reason": error,
    }


def _remove_unified_diff_noop_hunks(
    diff_text: str,
) -> tuple[str, dict[str, Any] | None]:
    """Remove zero-change hunks without projecting away real edits."""

    records = _split_unified_diff_records(diff_text)
    retained: list[tuple[dict[str, Any], list[list[str]]]] = []
    removed_count = 0
    for record in records:
        substantive_hunks: list[list[str]] = []
        for hunk in record.get("hunks") or []:
            _, old_lines, new_lines, _, _ = _hunk_old_new_lines(hunk)
            if old_lines == new_lines:
                removed_count += 1
                continue
            substantive_hunks.append(hunk)
        if substantive_hunks:
            retained.append((record, substantive_hunks))

    # Preserve an all-noop response so the no-effect rejection path reports
    # it as such. Only canonicalize a transaction with at least one real edit.
    if not removed_count or not retained:
        return diff_text, None

    lines: list[str] = []
    for record, hunks in retained:
        old_header = str(record.get("old_header") or "")
        new_header = str(record.get("new_header") or "")
        if not old_header or not new_header:
            return diff_text, None
        lines.extend([old_header, new_header])
        for hunk in hunks:
            lines.extend(hunk)
    canonical = "\n".join(lines)
    if diff_text.endswith("\n"):
        canonical += "\n"
    valid, _ = _validate_unified_diff_structure(canonical)
    if not valid:
        return diff_text, None
    return canonical, {
        "kind": "unified_diff_noop_hunk_removal",
        "removed_hunk_count": removed_count,
        "retained_file_count": len(retained),
    }


def repair_unified_diff_structure(
    diff_text: str,
) -> tuple[str, dict[str, Any] | None]:
    repaired, range_repair = _repair_unified_diff_missing_hunk_ranges(diff_text)
    repaired, count_repair = _repair_unified_diff_hunk_counts(repaired)
    repaired, noop_repair = _remove_unified_diff_noop_hunks(repaired)
    repairs = [
        item
        for item in (range_repair, count_repair, noop_repair)
        if item is not None
    ]
    if not repairs:
        return diff_text, None
    if len(repairs) == 1:
        return repaired, repairs[0]
    return repaired, {
        "kind": "unified_diff_mechanical_repair",
        "repairs": repairs,
    }


def _hunk_old_new_lines(hunk: list[str]) -> tuple[int, list[str], list[str], list[str], list[str]]:
    if not hunk:
        return 0, [], [], [], []
    header_match = HUNK_HEADER_RE.match(hunk[0])
    old_start = int(header_match.group("old_start")) if header_match else 0
    old_lines: list[str] = []
    new_lines: list[str] = []
    removed_lines: list[str] = []
    added_lines: list[str] = []
    for line in hunk[1:]:
        if line.startswith("\\"):
            continue
        if not line:
            prefix = " "
            content = ""
        else:
            prefix = line[0]
            content = line[1:]
        if prefix == " ":
            old_lines.append(content)
            new_lines.append(content)
        elif prefix == "-":
            old_lines.append(content)
            removed_lines.append(content)
        elif prefix == "+":
            new_lines.append(content)
            added_lines.append(content)
    return old_start, old_lines, new_lines, removed_lines, added_lines


def _hunk_changes_within_owner(
    hunk: list[str],
    *,
    preimage_start: int,
    owner_start: int,
    owner_end: int,
) -> bool:
    cursor = preimage_start
    block_start = cursor
    block_removed: list[int] = []
    block_added = False
    in_block = False
    saw_change = False
    for line in [*hunk[1:], " "]:
        prefix = line[:1] if line else " "
        if prefix in {"+", "-"}:
            if not in_block:
                block_start = cursor
                block_removed = []
                block_added = False
                in_block = True
            saw_change = True
            if prefix == "-":
                block_removed.append(cursor)
                cursor += 1
            else:
                block_added = True
            continue
        if prefix == "\\":
            continue
        if in_block:
            if block_removed:
                if any(
                    not owner_start <= position < owner_end
                    for position in block_removed
                ):
                    return False
            elif block_added and not (
                owner_start < block_start < owner_end
            ):
                return False
            in_block = False
        cursor += 1
    return saw_change


def _has_current_patch_authority(constraint: dict[str, Any]) -> bool:
    return "repair_owner.authorized_current_entity" in (
        constraint.get("sources") or [constraint.get("source")]
    )


def _range_is_allowed(
    start: int,
    end: int,
    allowed_ranges: list[tuple[int, int]] | None,
) -> bool:
    return allowed_ranges is None or any(
        start >= allowed_start and end <= allowed_end
        for allowed_start, allowed_end in allowed_ranges
    )


def _find_sequence(
    lines: list[str],
    needle: list[str],
    *,
    allowed_ranges: list[tuple[int, int]] | None = None,
) -> list[int]:
    if not needle or len(needle) > len(lines):
        return []
    width = len(needle)
    return [
        index
        for index in range(0, len(lines) - width + 1)
        if _range_is_allowed(index, index + width, allowed_ranges)
        and lines[index : index + width] == needle
    ]


def _normalize_c_fragment(lines: list[str]) -> str:
    return re.sub(r"\s+", "", " ".join(line.strip() for line in lines))


def _find_normalized_sequence(
    lines: list[str],
    needle: list[str],
    *,
    allowed_ranges: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    target = _normalize_c_fragment(needle)
    if not target:
        return []

    normalized_lines = [re.sub(r"\s+", "", line) for line in lines]
    offsets = [0]
    for line in normalized_lines:
        offsets.append(offsets[-1] + len(line))
    normalized_source = "".join(normalized_lines)

    matches: list[tuple[int, int]] = []
    for start in range(len(lines)):
        target_end_offset = offsets[start] + len(target)
        end = bisect_left(
            offsets,
            target_end_offset,
            lo=start + 1,
        )
        if end >= len(offsets) or offsets[end] != target_end_offset:
            continue
        if not _range_is_allowed(start, end, allowed_ranges):
            continue
        if normalized_source[offsets[start] : offsets[end]] == target:
            matches.append((start, end))
    return matches


def _choose_match(matches: list[int], *, old_start: int) -> int | None:
    return matches[0] if len(matches) == 1 else None


def _choose_range_match(matches: list[tuple[int, int]], *, old_start: int) -> tuple[int, int] | None:
    return matches[0] if len(matches) == 1 else None


def _unique_sequence_range_match(
    lines: list[str],
    needle: list[str],
    *,
    allowed_ranges: list[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int] | None, str]:
    if not needle:
        return None, "empty context"
    exact = [
        (index, index + len(needle))
        for index in _find_sequence(
            lines,
            needle,
            allowed_ranges=allowed_ranges,
        )
    ]
    if len(exact) == 1:
        return exact[0], "matched_unique_context"
    if len(exact) > 1:
        return None, "exact context is ambiguous"
    normalized = _find_normalized_sequence(
        lines,
        needle,
        allowed_ranges=allowed_ranges,
    )
    if len(normalized) == 1:
        return normalized[0], "matched_unique_normalized_context"
    if len(normalized) > 1:
        return None, "normalized context is ambiguous"
    return None, "context not found"


def _insertion_contexts(hunk: list[str]) -> tuple[list[str], list[str]]:
    leading: list[str] = []
    trailing: list[str] = []
    seen_addition = False
    for line in hunk[1:]:
        if line.startswith("\\"):
            continue
        if not line:
            prefix = " "
            content = ""
        else:
            prefix = line[0]
            content = line[1:]
        if prefix == "+":
            seen_addition = True
        elif prefix == " ":
            if seen_addition:
                trailing.append(content)
            else:
                leading.append(content)
    return leading, trailing


def _complete_hunk_postimage_match(
    lines: list[str],
    hunk: list[str],
    *,
    allowed_ranges: list[tuple[int, int]] | None = None,
) -> tuple[bool, str]:
    """Prove that one hunk's complete postimage is uniquely present."""

    _old_start, _old_lines, new_lines, _removed_lines, _added_lines = (
        _hunk_old_new_lines(hunk)
    )
    if not new_lines:
        return False, "hunk has no postimage lines"
    exact_matches = _find_sequence(
        lines,
        new_lines,
        allowed_ranges=allowed_ranges,
    )
    if len(exact_matches) == 1:
        return True, "complete hunk postimage matched exactly"
    if len(exact_matches) > 1:
        return False, "complete hunk postimage is ambiguous"
    normalized_matches = _find_normalized_sequence(
        lines,
        new_lines,
        allowed_ranges=allowed_ranges,
    )
    if len(normalized_matches) == 1:
        return True, "complete hunk postimage matched after normalization"
    if len(normalized_matches) > 1:
        return False, "normalized complete hunk postimage is ambiguous"
    return False, "complete hunk postimage not found"


def _apply_insertion_hunk(
    lines: list[str],
    hunk: list[str],
    *,
    old_lines: list[str],
    new_lines: list[str],
    added_lines: list[str],
    allowed_ranges: list[tuple[int, int]] | None = None,
) -> tuple[list[str] | None, str]:
    if not added_lines:
        return None, "insertion hunk has no added lines"
    postimage_present, postimage_reason = _complete_hunk_postimage_match(
        lines,
        hunk,
        allowed_ranges=allowed_ranges,
    )
    if postimage_present:
        return lines, f"already_applied:{postimage_reason}"

    full_match, reason = _unique_sequence_range_match(
        lines,
        old_lines,
        allowed_ranges=allowed_ranges,
    )
    if full_match is not None:
        start, end = full_match
        return lines[:start] + new_lines + lines[end:], f"inserted_by_full_context:{reason}"

    normalized_full_match = _choose_range_match(
        _find_normalized_sequence(
            lines,
            old_lines,
            allowed_ranges=allowed_ranges,
        ),
        old_start=int(HUNK_HEADER_RE.match(hunk[0]).group("old_start") or 0),
    )
    if normalized_full_match is not None:
        start, end = normalized_full_match
        return (
            lines[:start] + new_lines + lines[end:],
            "inserted_by_normalized_full_context",
        )

    conditional_directive = re.compile(
        r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b"
    )
    if any(conditional_directive.match(line) for line in added_lines):
        return None, (
            "preprocessor conditional insertion requires complete-hunk "
            "relocation"
        )

    return None, "complete insertion context not found"


def _apply_unique_context_insertion_hunk(
    lines: list[str],
    hunk: list[str],
) -> tuple[list[str] | None, str]:
    (
        _old_start,
        old_lines,
        new_lines,
        removed_lines,
        added_lines,
    ) = _hunk_old_new_lines(hunk)
    if removed_lines:
        return None, "unique-context insertion contains removed lines"
    return _apply_insertion_hunk(
        lines,
        hunk,
        old_lines=old_lines,
        new_lines=new_lines,
        added_lines=added_lines,
    )


def _apply_relocatable_hunk(
    lines: list[str],
    hunk: list[str],
    *,
    allowed_ranges: list[tuple[int, int]] | None = None,
) -> tuple[list[str] | None, str]:
    old_start, old_lines, new_lines, removed_lines, added_lines = _hunk_old_new_lines(hunk)
    if not removed_lines:
        return _apply_insertion_hunk(
            lines,
            hunk,
            old_lines=old_lines,
            new_lines=new_lines,
            added_lines=added_lines,
            allowed_ranges=allowed_ranges,
        )

    matches = _find_sequence(
        lines,
        old_lines,
        allowed_ranges=allowed_ranges,
    )
    match_index = _choose_match(matches, old_start=old_start)
    if match_index is not None:
        return lines[:match_index] + new_lines + lines[match_index + len(old_lines) :], "matched_full_hunk"

    normalized_full_match = _choose_range_match(
        _find_normalized_sequence(
            lines,
            old_lines,
            allowed_ranges=allowed_ranges,
        ),
        old_start=old_start,
    )
    if normalized_full_match is not None:
        start, end = normalized_full_match
        return (
            lines[:start] + new_lines + lines[end:],
            "relocated_by_normalized_full_hunk",
        )

    conditional_directive = re.compile(
        r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b"
    )
    if any(
        conditional_directive.match(line)
        for line in [*removed_lines, *added_lines]
    ):
        postimage_present, postimage_reason = _complete_hunk_postimage_match(
            lines,
            hunk,
            allowed_ranges=allowed_ranges,
        )
        if postimage_present:
            return lines, f"already_applied:{postimage_reason}"
        return None, (
            "preprocessor conditional change requires complete-hunk "
            "relocation"
        )

    change_block_count = 0
    in_change_block = False
    for line in hunk[1:]:
        prefix = line[:1] if line else " "
        if prefix in {"+", "-"}:
            if not in_change_block:
                change_block_count += 1
                in_change_block = True
        elif prefix != "\\":
            in_change_block = False
    removed_line_fallback_safe = change_block_count == 1
    unsafe_removed_line_match = False
    matches = _find_sequence(
        lines,
        removed_lines,
        allowed_ranges=allowed_ranges,
    )
    match_index = _choose_match(matches, old_start=old_start)
    if match_index is not None and removed_line_fallback_safe:
        return (
            lines[:match_index] + added_lines + lines[match_index + len(removed_lines) :],
            "relocated_by_removed_lines",
        )
    if match_index is not None:
        unsafe_removed_line_match = True

    range_match = _choose_range_match(
        _find_normalized_sequence(
            lines,
            removed_lines,
            allowed_ranges=allowed_ranges,
        ),
        old_start=old_start,
    )
    if range_match is not None and removed_line_fallback_safe:
        start, end = range_match
        return lines[:start] + added_lines + lines[end:], "relocated_by_normalized_removed_lines"
    if range_match is not None:
        unsafe_removed_line_match = True

    postimage_present, postimage_reason = _complete_hunk_postimage_match(
        lines,
        hunk,
        allowed_ranges=allowed_ranges,
    )
    if postimage_present:
        return lines, f"already_applied:{postimage_reason}"
    if unsafe_removed_line_match:
        return None, "removed-line relocation is unsafe for a multi-block hunk"
    return None, "removed lines not found"


def _disconnected_change_blocks(
    hunk: list[str],
) -> list[dict[str, list[str]]]:
    """Return source-order +/- blocks separated by unchanged context."""

    blocks: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] | None = None
    for line in hunk[1:]:
        if line.startswith("\\"):
            continue
        prefix = line[:1] if line else " "
        if prefix not in {"+", "-"}:
            current = None
            continue
        if current is None:
            current = {"removed": [], "added": []}
            blocks.append(current)
        current["removed" if prefix == "-" else "added"].append(line[1:])
    return blocks


def _structural_entities_for_lines(
    *,
    relative_path: str,
    lines: list[str],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="driver-backport-contract-block-"
    ) as temp_dir:
        root = Path(temp_dir)
        path = root / normalize_relpath(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(lines)
        if lines:
            text += "\n"
        path.write_text(text, encoding="utf-8")
        return parse_structural_entity_file(
            root=root,
            relative_path=relative_path,
            tree="contract_block",
        )


def _single_export_contract(
    added_lines: list[str],
    *,
    relative_path: str,
) -> dict[str, Any] | None:
    text = "\n".join(added_lines).strip()
    if not text:
        return None
    exports = parse_export_contracts(
        text,
        file=relative_path,
        tree="patch",
    )
    if len(exports) != 1:
        return None
    raw = str(exports[0].get("raw") or "").strip()
    if re.sub(r"\s+", "", raw) != re.sub(r"\s+", "", text):
        return None
    return exports[0]


def _apply_disconnected_contract_blocks(
    lines: list[str],
    hunk: list[str],
    *,
    relative_path: str,
    allowed_ranges: list[tuple[int, int]] | None = None,
) -> tuple[list[str] | None, str]:
    """Materialize independently proven blocks from one collapsed model hunk."""

    conditional_directive = re.compile(
        r"^\s*#\s*(?:if|ifdef|ifndef|elif|else|endif)\b"
    )
    if any(
        line[:1] in {"+", "-"}
        and conditional_directive.match(line[1:])
        for line in hunk[1:]
        if line and not line.startswith(("+++", "---"))
    ):
        return None, (
            "disconnected relocation cannot split a preprocessor "
            "conditional change; relocate the complete hunk"
        )

    blocks = _disconnected_change_blocks(hunk)
    if len(blocks) < 2:
        return None, "hunk does not contain disconnected change blocks"

    modified = list(lines)
    last_change_end = -1
    outcomes: list[str] = []
    for block in blocks:
        removed = list(block["removed"])
        added = list(block["added"])
        if not removed:
            export = _single_export_contract(
                added,
                relative_path=relative_path,
            )
            if export is None:
                return None, (
                    "disconnected pure insertion is not a typed export contract"
                )
            existing_exports = parse_export_contracts(
                "\n".join(modified),
                file=relative_path,
                tree="current",
            )
            matching_existing = [
                item
                for item in existing_exports
                if str(item.get("contract_key") or "")
                == str(export.get("contract_key") or "")
            ]
            if matching_existing:
                change_start = max(
                    0,
                    int(matching_existing[0].get("line") or 1) - 1,
                )
                change_end = change_start + 1
                outcomes.append("export_already_present")
            else:
                report = _structural_entities_for_lines(
                    relative_path=relative_path,
                    lines=modified,
                )
                if not report.get("parse_ok"):
                    return None, (
                        "typed export target requires a parseable current owner"
                    )
                symbol = str(export.get("symbol") or "")
                providers = [
                    entity
                    for entity in report.get("entities") or []
                    if entity.get("is_definition")
                    and str(entity.get("kind") or "")
                    in {"function", "variable"}
                    and str(entity.get("symbol") or "") == symbol
                ]
                if len(providers) != 1:
                    return None, (
                        "typed export target is absent or ambiguous in current owner"
                    )
                extent = dict(providers[0].get("extent") or {})
                provider_start = int(extent.get("start_line") or 0) - 1
                provider_end = int(extent.get("end_line") or 0)
                if (
                    provider_start < 0
                    or provider_end <= provider_start
                    or not _range_is_allowed(
                        provider_start,
                        provider_end,
                        allowed_ranges,
                    )
                ):
                    return None, (
                        "typed export provider is outside the graph-proven range"
                    )
                change_start = provider_end
                change_end = change_start + len(added)
                modified[change_start:change_start] = added
                outcomes.append("inserted_export_after_unique_provider")
        elif added:
            exact_matches = _find_sequence(
                modified,
                removed,
                allowed_ranges=allowed_ranges,
            )
            normalized_matches = _find_normalized_sequence(
                modified,
                removed,
                allowed_ranges=allowed_ranges,
            )
            if len(exact_matches) == 1:
                change_start = exact_matches[0]
                change_end = change_start + len(added)
                modified[
                    change_start : change_start + len(removed)
                ] = added
                outcomes.append("replaced_unique_exact_block")
            elif not exact_matches and len(normalized_matches) == 1:
                change_start, old_end = normalized_matches[0]
                change_end = change_start + len(added)
                modified[change_start:old_end] = added
                outcomes.append("replaced_unique_normalized_block")
            else:
                return None, (
                    "disconnected replacement block is absent or ambiguous"
                )
        else:
            return None, (
                "disconnected pure deletion has no typed materialization contract"
            )

        if change_start < last_change_end:
            return None, (
                "disconnected contract blocks do not preserve source order"
            )
        last_change_end = change_end

    if modified == lines:
        return None, "disconnected contract blocks made no source change"
    return (
        modified,
        "materialized_disconnected_contract_blocks:"
        + ",".join(outcomes),
    )


def _diff_already_applied(
    spec: CaseSpec,
    *,
    diff_text: str,
    touched_files: list[str],
) -> tuple[bool, dict[str, Any]]:
    touched = {normalize_relpath(item) for item in touched_files}
    records = _split_unified_diff_records(diff_text)
    if not records:
        return False, {"status": "not_already_applied", "reason": "no unified diff records"}

    has_pure_deletion = any(
        bool(removed_lines and not added_lines)
        for record in records
        for hunk in record.get("hunks") or []
        for (
            _old_start,
            _old_lines,
            _new_lines,
            removed_lines,
            added_lines,
        ) in [_hunk_old_new_lines(hunk)]
    )
    reverse_ok, reverse_stdout, reverse_stderr = preflight_patch(
        spec,
        diff_text=diff_text,
        touched_files=touched_files,
        reverse=True,
    )
    if reverse_ok and not has_pure_deletion:
        return True, {
            "status": "already_applied",
            "reason": "reverse patch dry-run succeeded against the current tree",
            "verification": "reverse_patch_dry_run",
            "stdout": reverse_stdout,
            "stderr": reverse_stderr,
        }

    details: list[dict[str, Any]] = []
    hunk_count = 0
    for record in records:
        relative_path = normalize_relpath(str(record.get("file") or ""))
        if relative_path not in touched:
            return False, {"status": "not_already_applied", "reason": f"unexpected diff file {relative_path}"}
        path = spec.resolve_old(relative_path)
        if not path.exists():
            return False, {"status": "not_already_applied", "reason": f"target file missing: {relative_path}"}
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for hunk_index, hunk in enumerate(record.get("hunks") or [], start=1):
            hunk_count += 1
            _, _old_lines, _new_lines, removed_lines, added_lines = _hunk_old_new_lines(hunk)
            already, reason = _complete_hunk_postimage_match(lines, hunk)
            details.append({"file": relative_path, "hunk": hunk_index, "reason": reason})
            if not already:
                return False, {
                    "status": "not_already_applied",
                    "reason": reason,
                    "details": details,
                }
    if has_pure_deletion:
        return False, {
            "status": "not_already_applied",
            "reason": (
                "a pure deletion requires reverse-patch proof; unchanged context "
                "alone cannot prove that the deleted entity is absent"
            ),
            "details": details,
            "reverse_patch_stdout": reverse_stdout,
            "reverse_patch_stderr": reverse_stderr,
        }
    return hunk_count > 0, {
        "status": "already_applied",
        "hunk_count": hunk_count,
        "details": details,
        "verification": "per_hunk_new_content",
    }


def build_patch_conflict_evidence(
    spec: CaseSpec,
    *,
    diff_text: str,
    touched_files: list[str] | None = None,
) -> dict[str, Any]:
    """Inspect failed diff hunks against the exact current target contents."""

    allowed = {
        normalize_relpath(path)
        for path in (
            touched_files
            if touched_files is not None
            else touched_files_from_diff(diff_text)
        )
    }
    records: list[dict[str, Any]] = []
    omitted = 0
    for diff_record in _split_unified_diff_records(diff_text):
        relative_path = normalize_relpath(str(diff_record.get("file") or ""))
        if not relative_path or relative_path not in allowed:
            continue
        path = spec.resolve_old(relative_path)
        if not path.is_file():
            records.append(
                {
                    "file": relative_path,
                    "status": "target_file_missing",
                }
            )
            continue
        current_lines = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
        for hunk_index, hunk in enumerate(
            diff_record.get("hunks") or [],
            start=1,
        ):
            if len(records) >= PATCH_CONFLICT_MAX_HUNKS:
                omitted += 1
                continue
            (
                old_start,
                old_lines,
                _new_lines,
                removed_lines,
                added_lines,
            ) = _hunk_old_new_lines(hunk)
            relocated, relocation_reason = _apply_relocatable_hunk(
                current_lines,
                hunk,
            )
            expected_start = max(0, old_start - 1)
            expected_end = min(
                len(current_lines),
                expected_start
                + max(
                    1,
                    min(
                        PATCH_CONFLICT_MAX_EXCERPT_LINES,
                        len(old_lines) + 12,
                    ),
                ),
            )
            records.append(
                {
                    "file": relative_path,
                    "hunk": hunk_index,
                    "old_start": old_start,
                    "static_relocation_possible": relocated is not None,
                    "static_relocation_changes_source": (
                        relocated is not None
                        and relocated != current_lines
                    ),
                    "static_relocation_outcome": (
                        "source_change"
                        if relocated is not None
                        and relocated != current_lines
                        else "already_present"
                        if relocated is not None
                        else "not_relocatable"
                    ),
                    "static_relocation_reason": relocation_reason,
                    "removed_exact_match_count": len(
                        _find_sequence(current_lines, removed_lines)
                    ),
                    "removed_normalized_match_count": len(
                        _find_normalized_sequence(
                            current_lines,
                            removed_lines,
                        )
                    ),
                    "old_context_exact_match_count": len(
                        _find_sequence(current_lines, old_lines)
                    ),
                    "old_context_normalized_match_count": len(
                        _find_normalized_sequence(
                            current_lines,
                            old_lines,
                        )
                    ),
                    "removed_lines": removed_lines[
                        :PATCH_CONFLICT_MAX_EXCERPT_LINES
                    ],
                    "added_lines": added_lines[
                        :PATCH_CONFLICT_MAX_EXCERPT_LINES
                    ],
                    "current_source": {
                        "line_start": expected_start + 1,
                        "line_end": expected_end,
                        "text": "\n".join(
                            f"{line_number}: {current_lines[line_number - 1]}"
                            for line_number in range(
                                expected_start + 1,
                                expected_end + 1,
                            )
                        ),
                    },
                }
            )
    return {
        "schema": "patch_conflict_evidence_v1",
        "hunks": records,
        "hunk_count": len(records),
        "omitted_hunk_count": omitted,
    }


def canonical_patch_signature(diff_text: str) -> str:
    records_payload: list[dict[str, Any]] = []
    for record in _split_unified_diff_records(diff_text):
        file_payload: dict[str, Any] = {
            "file": normalize_relpath(str(record.get("file") or "")),
            "hunks": [],
        }
        for hunk in record.get("hunks") or []:
            _, old_lines, new_lines, _removed_lines, _added_lines = _hunk_old_new_lines(hunk)
            file_payload["hunks"].append(
                {
                    "old": old_lines,
                    "new": new_lines,
                }
            )
        records_payload.append(file_payload)
    if records_payload:
        encoded = json.dumps(records_payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    else:
        encoded = diff_text.strip().encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unit_structural_rebase_ranges(
    spec: CaseSpec,
    *,
    unit: dict[str, Any],
    touched_files: list[str],
) -> tuple[
    dict[str, list[tuple[int, int]]],
    list[dict[str, Any]],
]:
    touched = {normalize_relpath(item) for item in touched_files}
    candidates: dict[str, list[dict[str, Any]]] = {
        relative_path: [] for relative_path in touched
    }

    def add_entity_candidate(
        raw_entity: object,
        *,
        source: str,
        exact_graph_owner: bool = False,
        require_definition: bool = True,
        exact_absent_entity_insertion_allowed: bool = False,
    ) -> None:
        if not isinstance(raw_entity, dict):
            return
        entity = dict(raw_entity)
        relative_path = normalize_relpath(
            str(entity.get("file") or entity.get("path") or "")
        )
        if relative_path not in touched:
            return
        container = (
            entity.get("container")
            if isinstance(entity.get("container"), dict)
            else {}
        )
        container_kind = str(
            container.get("container_kind") or ""
        )
        container_name = str(
            container.get("container_name") or ""
        )
        if container_kind and container_name:
            candidates[relative_path].append(
                {
                    "owner_type": "container",
                    "container_kind": container_kind,
                    "container_name": container_name,
                    "source": source,
                    "exact_graph_owner": exact_graph_owner,
                    "donor_member_extent": dict(
                        entity.get("extent") or {}
                    ),
                }
            )
            return
        contract = entity_contract_key(entity)
        kind = entity_kind(entity)
        if (
            not contract
            or kind
            not in {
                "function",
                "variable",
                "type",
                "field",
                "enum_constant",
                "macro",
                "top_level_call_expression",
            }
            or (
                kind == "top_level_call_expression"
                and entity.get("is_source_anchor") is not True
            )
        ):
            return
        candidates[relative_path].append(
            {
                "owner_type": "entity",
                "descriptor": entity,
                "contract_key": contract,
                "entity_kind": kind,
                "symbol": (
                    entity.get("symbol")
                    or entity.get("name")
                ),
                "source": source,
                "exact_graph_owner": exact_graph_owner,
                "require_definition": (
                    require_definition
                    and entity.get("is_definition") is not False
                ),
                "donor_extent": dict(entity.get("extent") or {}),
                "planned_current_extent": (
                    dict(
                        entity.get("source_extent")
                        or entity.get("extent")
                        or {}
                    )
                    if exact_graph_owner
                    and str(entity.get("tree") or "") == "old"
                    else {}
                ),
                "exact_absent_entity_insertion_allowed": bool(
                    exact_absent_entity_insertion_allowed
                    or entity.get(
                        "exact_absent_entity_insertion_allowed"
                    )
                ),
                "exact_absent_declaration_insertion_allowed": bool(
                    entity.get("exact_absent_declaration_insertion_allowed")
                ),
            }
        )

    for evidence_record in unit.get("contract_evidence") or []:
        if not isinstance(evidence_record, dict):
            continue
        add_entity_candidate(
            evidence_record.get("donor_entity"),
            source="contract_evidence.donor_entity",
        )
        add_entity_candidate(
            evidence_record.get("old_entity"),
            source="contract_evidence.old_entity",
        )

    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in unit.get("contract_evidence") or []
        if isinstance(item, dict) and item.get("id")
    }
    authorized_locus_ids_by_file: dict[str, set[str]] = {}
    for target in unit.get("patch_targets") or []:
        if not isinstance(target, dict):
            continue
        relative_path = normalize_relpath(str(target.get("file") or ""))
        if relative_path not in touched:
            continue
        authorized_locus_ids_by_file.setdefault(relative_path, set()).update(
            str(item)
            for item in target.get("repair_locus_ids") or []
            if str(item)
        )
    for locus in unit.get("repair_loci") or []:
        if not isinstance(locus, dict):
            continue
        relative_path = normalize_relpath(str(locus.get("file") or ""))
        locus_id = str(locus.get("id") or "")
        contract_key = str(locus.get("contract_key") or "")
        if (
            relative_path not in touched
            or not locus_id
            or locus_id
            not in authorized_locus_ids_by_file.get(relative_path, set())
            or not contract_key
        ):
            continue
        evidence_record = evidence_by_id.get(
            str(locus.get("contract_evidence_id") or "")
        )
        if not isinstance(evidence_record, dict):
            continue
        target_presence = str(locus.get("target_presence") or "")
        entity_field = (
            "donor_entity"
            if target_presence == "missing_old_entity"
            else "old_entity"
            if target_presence == "existing_old_entity"
            else ""
        )
        entity = evidence_record.get(entity_field)
        if (
            not isinstance(entity, dict)
            or normalize_relpath(str(entity.get("file") or ""))
            != relative_path
            or entity_contract_key(entity) != contract_key
            or entity_kind(entity)
            != str(locus.get("entity_kind") or "").lower()
        ):
            continue
        add_entity_candidate(
            entity,
            source="repair_locus.exact_contract_owner",
            exact_graph_owner=True,
            exact_absent_entity_insertion_allowed=(
                target_presence == "missing_old_entity"
            ),
        )
    for anchor in unit.get("demand_anchors") or []:
        if isinstance(anchor, dict):
            add_entity_candidate(
                anchor.get("owner_entity"),
                source="demand_anchor.owner_entity",
            )
    for graph_key in (
        "root_cause_cluster",
        "code_graph_root_cause_cluster",
    ):
        add_entity_candidate(
            unit.get(graph_key),
            source=graph_key,
        )
    source_pack = (
        unit.get("repair_source_pack")
        if isinstance(unit.get("repair_source_pack"), dict)
        else {}
    )
    for source_group in (
        "allowed_edit_sources",
        "donor_reference_sources",
    ):
        for record in source_pack.get(source_group) or []:
            add_entity_candidate(
                record,
                source=f"repair_source_pack.{source_group}",
            )

    graph_scope = (
        unit.get("code_graph_scope")
        if isinstance(unit.get("code_graph_scope"), dict)
        else {}
    )
    graph_nodes: dict[str, dict[str, Any]] = {}
    for graph in (
        graph_scope,
        (
            unit.get("diagnostic_graph")
            if isinstance(unit.get("diagnostic_graph"), dict)
            else {}
        ),
    ):
        for node in graph.get("nodes") or []:
            if isinstance(node, dict) and node.get("id"):
                graph_nodes[str(node["id"])] = dict(node)
    include_relation_owners: list[tuple[dict[str, Any], str]] = [
        (owner, "code_graph_scope.include_relation_owner")
        for owner in graph_scope.get("owner_candidates") or []
        if isinstance(owner, dict)
        and str(owner.get("owner_role") or "")
        == "include_relation_owner"
    ]
    include_relation_owners.extend(
        (owner, "repair_owner.include_relation_owner")
        for owner in unit.get("repair_owners") or []
        if isinstance(owner, dict)
        and str(owner.get("owner_role") or "")
        == "include_relation_owner"
        and owner.get("allowed_as_patch_target") is True
        and normalize_relpath(str(owner.get("owner_file") or ""))
        in {
            normalize_relpath(str(item or ""))
            for item in owner.get("allowed_patch_files") or []
            if str(item or "")
        }
    )
    seen_include_relations: set[tuple[object, ...]] = set()
    for owner, owner_source in include_relation_owners:
        if (
            str(owner.get("owner_role") or "")
            != "include_relation_owner"
        ):
            continue
        relative_path = normalize_relpath(
            str(owner.get("file") or owner.get("owner_file") or "")
        )
        relation_mode = str(owner.get("include_relation_mode") or "")
        current_include_name = normalize_relpath(
            str(owner.get("current_include_name") or "")
        )
        required_include_name = normalize_relpath(
            str(owner.get("required_include_name") or "")
        )
        if (
            relative_path not in touched
            or relation_mode not in {"insert_provider", "mutate_unresolved"}
            or (
                relation_mode == "insert_provider"
                and not required_include_name
            )
            or (
                relation_mode == "mutate_unresolved"
                and not current_include_name
            )
        ):
            continue
        relation_identity = (
            relative_path,
            relation_mode,
            current_include_name,
            required_include_name,
            int(owner.get("expected_line") or 0),
            normalize_relpath(str(owner.get("provider_header_file") or "")),
        )
        if relation_identity in seen_include_relations:
            continue
        seen_include_relations.add(relation_identity)
        candidates[relative_path].append(
            {
                "owner_type": "include_relation",
                "contract_key": (
                    f"include:{required_include_name}"
                    if relation_mode == "insert_provider"
                    else f"include:{current_include_name}"
                ),
                "include_relation_mode": relation_mode,
                "current_include_name": current_include_name,
                "required_include_name": required_include_name,
                "expected_line": int(owner.get("expected_line") or 0),
                "provider_header_file": owner.get("provider_header_file"),
                "source": owner_source,
                "exact_graph_owner": True,
            }
        )
    root = unit.get("root_cause_cluster") or unit.get(
        "code_graph_root_cause_cluster"
    ) or {}
    root_file = normalize_relpath(str(root.get("file") or ""))
    root_node_id = str(root.get("node_id") or "")
    mapped_import_events = {
        str(mapping.get("diagnostic_event_id") or "")
        for mapping in unit.get("diagnostic_mappings") or []
        if isinstance(mapping, dict)
        and mapping.get("status") == "mapped_module_namespace_import_relation"
        and root_node_id in (mapping.get("root_candidate_node_ids") or [])
    }
    if root_file in touched and root_node_id == f"file:old:{root_file}":
        for issue in (
            unit.get("source_issues")
            or unit.get("source_issue_samples")
            or []
        ):
            if (
                isinstance(issue, dict)
                and issue.get("kind") == "modpost_namespace_import_missing"
                and str(issue.get("diagnostic_event_id") or "")
                in mapped_import_events
            ):
                candidates[root_file].append(
                    {
                        "owner_type": "module_import_relation",
                        "contract_key": "module_import:"
                        + str(issue.get("namespace") or ""),
                        "namespace": str(issue.get("namespace") or ""),
                        "source": "diagnostic_mapping.module_import_relation",
                        "exact_graph_owner": True,
                    }
                )
    authorized_current_repair_owner_node_ids: set[str] = set()
    for owner in unit.get("repair_owners") or []:
        if not isinstance(owner, dict):
            continue
        owner_node_id = str(owner.get("owner_entity_id") or "")
        owner_file = normalize_relpath(str(owner.get("owner_file") or ""))
        owner_tree = str(owner.get("owner_entity_tree") or "")
        owner_kind = str(owner.get("owner_entity_kind") or "").lower()
        owner_contract = str(owner.get("owner_contract_key") or "")
        allowed_patch_files = {
            normalize_relpath(str(item or ""))
            for item in owner.get("allowed_patch_files") or []
            if str(item or "")
        }
        if (
            owner_node_id
            and owner_file in touched
            and owner_tree == "old"
            and owner_kind
            in {
                "function",
                "variable",
                "type",
                "field",
                "enum_constant",
                "macro",
                "top_level_call_expression",
            }
            and (
                owner_kind != "top_level_call_expression"
                or (graph_nodes.get(owner_node_id) or {}).get("is_source_anchor") is True
            )
            and owner_contract
            and owner.get("allowed_as_patch_target") is True
            and owner_file in allowed_patch_files
        ):
            authorized_current_repair_owner_node_ids.add(owner_node_id)
            add_entity_candidate(
                {
                    "id": owner_node_id,
                    "file": owner_file,
                    "tree": owner_tree,
                    "kind": owner_kind,
                    "symbol": owner.get("owner_symbol"),
                    "contract_key": owner_contract,
                    "container": owner.get("owner_entity_container"),
                    "is_definition": owner.get("owner_is_definition"),
                    "extent": dict(
                        owner.get("owner_entity_extent") or {}
                    ),
                    "is_source_anchor": (
                        (graph_nodes.get(owner_node_id) or {}).get("is_source_anchor")
                    ),
                },
                source="repair_owner.authorized_current_entity",
                exact_graph_owner=True,
            )
        if not owner_node_id:
            continue
        owner_node = {
            "id": owner_node_id,
            "file": owner_file,
            "tree": owner_tree,
            "kind": owner_kind,
            "symbol": owner.get("owner_symbol"),
            "contract_key": owner_contract,
            "container": owner.get("owner_entity_container"),
            "is_definition": owner.get("owner_is_definition"),
            "extent": dict(owner.get("owner_entity_extent") or {}),
            "is_source_anchor": (
                (graph_nodes.get(owner_node_id) or {}).get("is_source_anchor")
            ),
            "exact_absent_entity_insertion_allowed": bool(
                owner.get("exact_absent_entity_insertion_allowed")
                or owner.get(
                    "exact_absent_declaration_insertion_allowed"
                )
            ),
            "exact_absent_declaration_insertion_allowed": bool(
                owner.get("exact_absent_declaration_insertion_allowed")
            ),
        }
        if owner_node_id not in graph_nodes:
            graph_nodes[owner_node_id] = owner_node
        else:
            graph_nodes[owner_node_id] = {
                **owner_node,
                **graph_nodes[owner_node_id],
                "exact_absent_entity_insertion_allowed": bool(
                    owner_node[
                        "exact_absent_entity_insertion_allowed"
                    ]
                    or graph_nodes[owner_node_id].get(
                        "exact_absent_entity_insertion_allowed"
                    )
                ),
                "exact_absent_declaration_insertion_allowed": bool(
                    owner_node[
                        "exact_absent_declaration_insertion_allowed"
                    ]
                    or graph_nodes[owner_node_id].get(
                        "exact_absent_declaration_insertion_allowed"
                    )
                ),
            }

    root_node_ids: set[str] = set()
    exact_owner_node_ids: set[str] = set(
        authorized_current_repair_owner_node_ids
    )
    diagnostic_transaction_owner_node_ids: set[str] = set()

    def add_graph_identity(raw: object, *, root: bool) -> None:
        if isinstance(raw, str):
            node_id = raw
        elif isinstance(raw, dict):
            node_id = str(
                raw.get("node_id")
                or raw.get("entity_id")
                or raw.get("id")
                or ""
            )
        else:
            node_id = ""
        if not node_id:
            return
        exact_owner_node_ids.add(node_id)
        if root:
            root_node_ids.add(node_id)

    for cluster in graph_scope.get("root_cause_clusters") or []:
        add_graph_identity(cluster, root=True)
    for graph_key in (
        "root_cause_cluster",
        "code_graph_root_cause_cluster",
    ):
        root_record = unit.get(graph_key)
        if isinstance(root_record, dict):
            add_graph_identity(root_record, root=True)
            for alternative in root_record.get("alternatives") or []:
                add_graph_identity(alternative, root=True)
    for owner in graph_scope.get("owner_candidates") or []:
        if not isinstance(owner, dict):
            continue
        owner_file = normalize_relpath(
            str(owner.get("file") or "")
        )
        if owner_file not in touched:
            continue
        owner_node_id = str(owner.get("node_id") or "")
        owner_root_node_id = str(owner.get("root_node_id") or "")
        owner_role = str(owner.get("owner_role") or "")
        if owner_node_id and owner_node_id not in graph_nodes:
            graph_nodes[owner_node_id] = dict(owner)
        add_graph_identity(owner_root_node_id, root=True)
        if (
            owner_node_id == owner_root_node_id
            or owner_role
            in {
                "root_owner",
                "provider_definition_owner",
                "provider_declaration_owner",
                "consumer_integration_owner",
            }
        ):
            add_graph_identity(owner_node_id, root=False)

    diagnostic_root_fields = (
        "root_node_id",
        "preferred_entity_id",
        "selected_root_node_id",
    )
    diagnostic_root_list_fields = (
        "root_node_ids",
        "mapped_root_node_ids",
        "candidate_root_node_ids",
    )
    for mapping in [
        *list(graph_scope.get("diagnostic_mappings") or []),
        *list(unit.get("diagnostic_mappings") or []),
    ]:
        if not isinstance(mapping, dict):
            continue
        for field in diagnostic_root_fields:
            add_graph_identity(mapping.get(field), root=True)
        for field in diagnostic_root_list_fields:
            for node_id in mapping.get(field) or []:
                add_graph_identity(node_id, root=True)

    repair_decision = (
        unit.get("repair_decision")
        if isinstance(unit.get("repair_decision"), dict)
        else {}
    )
    transaction_owner_authorized = all(
        repair_decision.get(field) is True
        for field in (
            "graph_complete",
            "root_mapping_complete",
            "scope_complete",
        )
    )
    if transaction_owner_authorized:
        for mapping in [
            *list(graph_scope.get("diagnostic_mappings") or []),
            *list(unit.get("diagnostic_mappings") or []),
        ]:
            if not isinstance(mapping, dict) or mapping.get("unmapped"):
                continue
            for node_id in (
                mapping.get("diagnostic_transaction_owner_node_ids")
                or []
            ):
                node_id = str(node_id or "")
                node = graph_nodes.get(node_id) or {}
                relative_path = normalize_relpath(
                    str(node.get("file") or node.get("path") or "")
                )
                if (
                    node_id
                    and relative_path in touched
                    and str(node.get("tree") or "") == "old"
                ):
                    exact_owner_node_ids.add(node_id)
                    diagnostic_transaction_owner_node_ids.add(node_id)

    structural_graph_kinds = {
        "function",
        "variable",
        "type",
        "field",
        "enum_constant",
        "macro",
        "top_level_call_expression",
    }
    root_contract_descriptors: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    for node_id in sorted(exact_owner_node_ids):
        node = graph_nodes.get(node_id)
        if not isinstance(node, dict):
            continue
        kind = entity_kind(node)
        contract = entity_contract_key(node)
        if kind not in structural_graph_kinds or not contract:
            continue
        add_entity_candidate(
            node,
            source=(
                "diagnostic_mapping.exact_transaction_owner"
                if node_id
                in diagnostic_transaction_owner_node_ids
                else "repair_owner.authorized_current_entity"
                if node_id
                in authorized_current_repair_owner_node_ids
                else "code_graph_scope.exact_owner"
            ),
            exact_graph_owner=True,
        )
        if node_id in root_node_ids and kind != "top_level_call_expression":
            root_contract_descriptors.setdefault(
                (kind, contract),
                node,
            )

    # A definition and its declarations are separate AST entities but share
    # one exact contract. When a model changes that contract in another
    # already-authorized touched file, resolve the declaration structurally
    # instead of falling back to a file-level write window.
    for descriptor in root_contract_descriptors.values():
        descriptor_file = normalize_relpath(
            str(
                descriptor.get("file")
                or descriptor.get("path")
                or ""
            )
        )
        for relative_path in sorted(touched):
            if relative_path == descriptor_file:
                continue
            if any(
                candidate.get("owner_type") == "entity"
                and candidate.get("exact_graph_owner")
                and candidate.get("entity_kind") == entity_kind(descriptor)
                and candidate.get("contract_key") == entity_contract_key(descriptor)
                for candidate in candidates[relative_path]
            ):
                continue
            add_entity_candidate(
                {
                    **descriptor,
                    "file": relative_path,
                    "path": relative_path,
                    "extent": {},
                    "source_extent": {},
                    "container": None,
                    "is_definition": None,
                    "exact_absent_entity_insertion_allowed": False,
                    "exact_absent_declaration_insertion_allowed": False,
                },
                source=(
                    "code_graph_scope.exact_contract_instance"
                ),
                exact_graph_owner=True,
                require_definition=False,
            )

    ranges_by_file: dict[str, list[tuple[int, int]]] = {}
    evidence: list[dict[str, Any]] = []
    if not any(candidates.values()):
        return (
            {
                relative_path: []
                for relative_path in sorted(touched)
            },
            [
                {
                    "file": relative_path,
                    "owners": [],
                    "status": "exact_owner_evidence_absent",
                    "matched_ranges": [],
                    "owner_constraints": [],
                    "unresolved_owner_ids": [],
                }
                for relative_path in sorted(touched)
            ],
        )
    code_index = CodeIndex(spec, graph_projection="records")
    report_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for relative_path in sorted(touched):
        raw_candidates = candidates.get(relative_path) or []
        unique_candidates: dict[
            tuple[str, str, str], dict[str, Any]
        ] = {}
        for candidate in raw_candidates:
            if candidate["owner_type"] == "container":
                identity = (
                    "container",
                    str(candidate.get("container_kind") or ""),
                    str(candidate.get("container_name") or ""),
                )
            elif candidate["owner_type"] == "include_relation":
                identity = (
                    "include_relation",
                    str(candidate.get("include_relation_mode") or ""),
                    str(
                        candidate.get("current_include_name")
                        or candidate.get("required_include_name")
                        or ""
                    ),
                )
            elif candidate["owner_type"] == "module_import_relation":
                identity = (
                    "module_import_relation",
                    str(candidate.get("namespace") or ""),
                    "",
                )
            else:
                identity = (
                    "entity",
                    str(candidate.get("entity_kind") or ""),
                    str(candidate.get("contract_key") or ""),
                )
            existing = unique_candidates.get(identity)
            if existing is None:
                unique_candidates[identity] = dict(candidate)
                continue
            exact_graph_owner = bool(
                existing.get("exact_graph_owner")
                or candidate.get("exact_graph_owner")
            )
            require_definition = bool(
                existing.get("require_definition", True)
                and candidate.get("require_definition", True)
            )
            sources = {
                str(item)
                for item in [
                    existing.get("source"),
                    *existing.get("sources", []),
                    candidate.get("source"),
                    *candidate.get("sources", []),
                ]
                if str(item or "")
            }

            def descriptor_precedence(
                item: dict[str, Any],
            ) -> tuple[int, int]:
                descriptor = item.get("descriptor")
                tree = (
                    str(descriptor.get("tree") or "")
                    if isinstance(descriptor, dict)
                    else ""
                )
                source = str(item.get("source") or "")
                return (
                    3
                    if tree == "old"
                    else 0
                    if source
                    == "code_graph_scope.exact_contract_instance"
                    else 1
                    if tree.startswith("donor")
                    else 2,
                    1 if item.get("exact_graph_owner") else 0,
                )

            selected = (
                candidate
                if descriptor_precedence(candidate)
                > descriptor_precedence(existing)
                else existing
            )
            merged = dict(selected)
            merged["exact_graph_owner"] = exact_graph_owner
            merged["require_definition"] = require_definition
            merged["sources"] = sorted(sources)
            if not merged.get("donor_extent"):
                merged["donor_extent"] = dict(
                    existing.get("donor_extent")
                    or candidate.get("donor_extent")
                    or {}
                )
            if not merged.get("planned_current_extent"):
                merged["planned_current_extent"] = dict(
                    existing.get("planned_current_extent")
                    or candidate.get("planned_current_extent")
                    or {}
                )
            merged["exact_absent_entity_insertion_allowed"] = bool(
                existing.get(
                    "exact_absent_entity_insertion_allowed"
                )
                or candidate.get(
                    "exact_absent_entity_insertion_allowed"
                )
            )
            merged["exact_absent_declaration_insertion_allowed"] = bool(
                existing.get("exact_absent_declaration_insertion_allowed")
                or candidate.get("exact_absent_declaration_insertion_allowed")
            )
            unique_candidates[identity] = merged

        owner_constraints: list[dict[str, Any]] = []
        structural_old_report: dict[str, Any] | None = None
        structural_donor_report: dict[str, Any] | None = None
        for identity, candidate in sorted(
            unique_candidates.items()
        ):
            if candidate["owner_type"] == "module_import_relation":
                namespace = str(candidate.get("namespace") or "")
                calls = parse_module_metadata_calls(
                    spec.resolve_old(relative_path).read_text(
                        encoding="utf-8", errors="ignore"
                    )
                )
                imports = [
                    call for call in calls
                    if call["name"] == "MODULE_IMPORT_NS"
                    and call["namespace"] == namespace
                ]
                licenses = [
                    call for call in calls if call["name"] == "MODULE_LICENSE"
                ]
                selected = (
                    imports[0] if len(imports) == 1 else
                    licenses[0] if not imports and len(licenses) == 1 else None
                )
                mode = "replace" if len(imports) == 1 else "insert"
                current_line = int(selected["line"]) if selected else 0
                current_status = (
                    "resolved"
                    if namespace and selected
                    and selected["line"] == selected["end_line"]
                    else "module_import_relation_ambiguous"
                )
                owner_constraints.append(
                    {
                        "id": stable_id(
                            "structural-rebase-owner", relative_path,
                            "module_import_relation", namespace,
                        ),
                        "file": relative_path,
                        "owner_kind": "module_import_relation",
                        "contract_key": candidate["contract_key"],
                        "namespace": namespace,
                        "relation_mode": mode,
                        "source": candidate["source"],
                        "sources": [candidate["source"]],
                        "donor_extent": None,
                        "current_extent": (
                            {"start_line": current_line, "end_line": current_line}
                            if current_status == "resolved" else None
                        ),
                        "current_status": current_status,
                        "exact_graph_owner": True,
                    }
                )
                continue
            if candidate["owner_type"] == "include_relation":
                include_boundary = _current_include_relation_boundary(
                    spec,
                    relative_path=relative_path,
                    relation_mode=str(
                        candidate.get("include_relation_mode") or ""
                    ),
                    current_include_name=str(
                        candidate.get("current_include_name") or ""
                    ),
                    required_include_name=str(
                        candidate.get("required_include_name") or ""
                    ),
                    expected_line=int(candidate.get("expected_line") or 0),
                )
                current_status = str(
                    include_boundary.get("status") or ""
                )
                current_start = int(
                    include_boundary.get("start_line") or 0
                )
                current_end = int(
                    include_boundary.get("end_line") or current_start
                )
                owner_constraints.append(
                    {
                        "id": stable_id(
                            "structural-rebase-owner",
                            relative_path,
                            "include_relation",
                            candidate.get("include_relation_mode"),
                            candidate.get("current_include_name"),
                            candidate.get("required_include_name"),
                        ),
                        "file": relative_path,
                        "owner_kind": "include_relation",
                        "contract_key": candidate.get("contract_key"),
                        "include_relation_mode": candidate.get(
                            "include_relation_mode"
                        ),
                        "current_include_name": candidate.get(
                            "current_include_name"
                        ),
                        "required_include_name": candidate.get(
                            "required_include_name"
                        ),
                        "provider_header_file": candidate.get(
                            "provider_header_file"
                        ),
                        "source": candidate.get("source"),
                        "sources": [candidate.get("source")],
                        "donor_extent": None,
                        "current_extent": (
                            {
                                "start_line": current_start,
                                "end_line": current_end,
                            }
                            if current_status == "resolved"
                            and current_start > 0
                            and current_end >= current_start
                            else None
                        ),
                        "current_status": current_status,
                        "current_extent_authority": {
                            "status": current_status,
                            "backend": "current_include_relation",
                        },
                        "donor_extent_authority": None,
                        "exact_graph_owner": True,
                        "exact_absent_entity_insertion_allowed": False,
                    }
                )
                continue
            if candidate["owner_type"] == "container":
                if structural_old_report is None:
                    structural_old_report = (
                        parse_structural_entity_file(
                            root=spec.old_root_path,
                            relative_path=relative_path,
                            tree="old",
                        )
                    )
                if structural_donor_report is None:
                    donor_root = getattr(
                        spec,
                        "primary_donor_root",
                        (
                            list(
                                getattr(
                                    spec,
                                    "donor_root_paths",
                                    [],
                                )
                                or []
                            )
                            or [spec.old_root_path]
                        )[0],
                    )
                    structural_donor_report = (
                        parse_structural_entity_file(
                            root=Path(donor_root),
                            relative_path=relative_path,
                            tree="donor",
                        )
                    )
                container_kind = str(
                    candidate.get("container_kind") or ""
                )
                container_name = str(
                    candidate.get("container_name") or ""
                )
                expected_cursor = (
                    f"TREE_SITTER_{container_kind.upper()}_SPECIFIER"
                )

                def container_matches(
                    report: dict[str, Any],
                ) -> list[dict[str, Any]]:
                    return [
                        entity
                        for entity in report.get("entities") or []
                        if isinstance(entity, dict)
                        and str(entity.get("kind") or "") == "type"
                        and bool(entity.get("is_definition"))
                        and str(entity.get("symbol") or "")
                        == container_name
                        and str(entity.get("cursor_kind") or "")
                        == expected_cursor
                    ]

                old_matches = container_matches(
                    structural_old_report
                )
                donor_matches = container_matches(
                    structural_donor_report
                )
                current_entity = (
                    old_matches[0]
                    if len(old_matches) == 1
                    else None
                )
                donor_entity = (
                    donor_matches[0]
                    if len(donor_matches) == 1
                    else None
                )
                current_status = (
                    "resolved"
                    if current_entity is not None
                    else "entity_absent"
                    if not old_matches
                    and structural_old_report.get("parse_ok")
                    and not structural_old_report.get(
                        "has_error_nodes"
                    )
                    else "ambiguous_or_unavailable"
                )
                donor_extent = dict(
                    (donor_entity or {}).get("extent")
                    or candidate.get("donor_member_extent")
                    or {}
                )
                current_extent = dict(
                    (current_entity or {}).get("extent") or {}
                )
                current_extent_authority = {
                    "status": "structural_container_extent",
                    "backend": structural_old_report.get("backend"),
                }
                donor_extent_authority = {
                    "status": "structural_container_extent",
                    "backend": structural_donor_report.get("backend"),
                }
                contract_key = (
                    f"{container_kind}:{container_name}"
                )
                owner_kind = "type"
            else:
                descriptor = dict(
                    candidate.get("descriptor") or {}
                )
                require_definition = bool(
                    candidate.get("require_definition", True)
                )
                if candidate.get(
                    "exact_absent_declaration_insertion_allowed"
                ):
                    descriptor["is_definition"] = False
                    require_definition = False
                current_resolution = resolve_entity_in_case_tree(
                    spec,
                    descriptor=descriptor,
                    relative_path=relative_path,
                    tree="old",
                    require_definition=require_definition,
                    index=code_index,
                    report_cache=report_cache,
                )
                donor_resolution = resolve_entity_in_case_tree(
                    spec,
                    descriptor=descriptor,
                    relative_path=relative_path,
                    tree="donor",
                    require_definition=require_definition,
                    index=code_index,
                    report_cache=report_cache,
                )
                current_entity = (
                    current_resolution.get("entity")
                    if current_resolution.get("status")
                    == "resolved"
                    else None
                )
                donor_entity = (
                    donor_resolution.get("entity")
                    if donor_resolution.get("status")
                    == "resolved"
                    else None
                )
                current_status = str(
                    current_resolution.get("status") or ""
                )
                physical_declaration_absence = bool(
                    current_status == "absence_unproven"
                    and current_resolution.get("parse_ok") is True
                    and candidate.get("exact_graph_owner")
                    and candidate.get(
                        "exact_absent_declaration_insertion_allowed"
                    )
                    and descriptor.get("is_definition") is False
                    and not require_definition
                    and donor_entity is not None
                    and _exact_declaration_absent_from_header(
                        spec,
                        relative_path,
                        str(candidate.get("symbol") or ""),
                    )
                )
                if physical_declaration_absence:
                    current_status = "entity_absent"
                donor_extent = dict(
                    (donor_entity or {}).get("source_extent")
                    or (donor_entity or {}).get("extent")
                    or candidate.get("donor_extent")
                    or {}
                )
                current_extent = dict(
                    (current_entity or {}).get("source_extent")
                    or (current_entity or {}).get("extent")
                    or {}
                )
                current_extent_authority: dict[str, Any] = {}
                if physical_declaration_absence:
                    current_extent_authority = {
                        "status": "exact_header_identifier_absence",
                        "backend": current_resolution.get("backend"),
                        "downstream_resolution_status": "absence_unproven",
                    }
                planned_current_extent = dict(
                    candidate.get("planned_current_extent") or {}
                )
                if (
                    current_status == "absence_unproven"
                    and candidate.get("exact_graph_owner")
                    and int(
                        planned_current_extent.get("start_line") or 0
                    )
                    > 0
                    and int(
                        planned_current_extent.get("end_line")
                        or planned_current_extent.get("start_line")
                        or 0
                    )
                    >= int(
                        planned_current_extent.get("start_line") or 0
                    )
                ):
                    current_extent = planned_current_extent
                    current_status = "resolved"
                    current_extent_authority = {
                        "status": "fresh_planned_current_owner_extent",
                        "backend": current_resolution.get("backend"),
                        "downstream_resolution_status": (
                            current_resolution.get("status")
                        ),
                    }
                donor_extent_authority = (
                    (donor_entity or {}).get("source_extent_authority")
                    or {
                        "status": "semantic_extent_fallback",
                        "backend": donor_resolution.get("backend"),
                    }
                )
                current_extent_authority = (
                    current_extent_authority
                    if current_extent_authority.get("status") in {
                        "fresh_planned_current_owner_extent",
                        "exact_header_identifier_absence",
                    }
                    else (
                        (current_entity or {}).get(
                            "source_extent_authority"
                        )
                        or {
                            "status": "semantic_extent_fallback",
                            "backend": current_resolution.get(
                                "backend"
                            ),
                        }
                    )
                )
                contract_key = str(
                    candidate.get("contract_key") or ""
                )
                owner_kind = str(
                    candidate.get("entity_kind") or ""
                )
                if (
                    candidate.get("source")
                    == "code_graph_scope.exact_contract_instance"
                    and current_entity is None
                    and donor_entity is None
                ):
                    continue
            current_start = int(
                current_extent.get("start_line") or 0
            )
            current_end = int(
                current_extent.get("end_line") or current_start
            )
            donor_start = int(
                donor_extent.get("start_line") or 0
            )
            donor_end = int(
                donor_extent.get("end_line") or donor_start
            )
            owner_constraints.append(
                {
                    "id": stable_id(
                        "structural-rebase-owner",
                        relative_path,
                        owner_kind,
                        contract_key,
                    ),
                    "file": relative_path,
                    "owner_kind": owner_kind,
                    "contract_key": contract_key,
                    "source": candidate.get("source"),
                    "sources": sorted(
                        {
                            str(item)
                            for item in [
                                candidate.get("source"),
                                *candidate.get("sources", []),
                            ]
                            if str(item or "")
                        }
                    ),
                    "donor_extent": (
                        {
                            "start_line": donor_start,
                            "end_line": donor_end,
                        }
                        if donor_start > 0
                        and donor_end >= donor_start
                        else None
                    ),
                    "current_extent": (
                        {
                            "start_line": current_start,
                            "end_line": current_end,
                        }
                        if current_start > 0
                        and current_end >= current_start
                        else None
                    ),
                    "current_status": current_status,
                    "current_extent_authority": current_extent_authority,
                    "donor_extent_authority": donor_extent_authority,
                    "exact_graph_owner": bool(
                        candidate.get("exact_graph_owner")
                    ),
                    "exact_absent_entity_insertion_allowed": bool(
                        candidate.get("exact_graph_owner")
                        and (
                            candidate.get(
                                "exact_absent_entity_insertion_allowed"
                            )
                            or candidate.get(
                                "exact_absent_declaration_insertion_allowed"
                            )
                        )
                        and current_status == "entity_absent"
                        and donor_entity is not None
                    ),
                }
            )
        ranges = {
            (
                int(constraint["current_extent"]["start_line"])
                - 1,
                int(constraint["current_extent"]["end_line"]),
            )
            for constraint in owner_constraints
            if constraint.get("current_status") == "resolved"
            and isinstance(
                constraint.get("current_extent"), dict
            )
        }
        ranges_by_file[relative_path] = sorted(ranges)
        unresolved_owners = [
            constraint
            for constraint in owner_constraints
            if constraint.get("current_status") != "resolved"
        ]
        evidence.append(
            {
                "file": relative_path,
                "owners": [
                    {
                        "kind": constraint.get("owner_kind"),
                        "contract_key": constraint.get(
                            "contract_key"
                        ),
                    }
                    for constraint in owner_constraints
                ],
                "status": (
                    "matched"
                    if owner_constraints
                    and not unresolved_owners
                    and ranges
                    else "owner_evidence_incomplete"
                    if owner_constraints
                    else "exact_owner_evidence_absent"
                ),
                "matched_ranges": [
                    {"start_line": start + 1, "end_line": end}
                    for start, end in sorted(ranges)
                ],
                "owner_constraints": owner_constraints,
                "unresolved_owner_ids": [
                    str(item.get("id") or "")
                    for item in unresolved_owners
                ],
            }
        )
    return ranges_by_file, evidence


def _unit_transaction_authority_ranges(
    unit: dict[str, Any],
    *,
    touched_files: list[str],
) -> dict[str, list[tuple[int, int]]]:
    """Return exact current postimage ranges owned by active patch transactions."""

    touched = {normalize_relpath(path) for path in touched_files}
    ranges: dict[str, set[tuple[int, int]]] = {
        relative_path: set() for relative_path in touched
    }
    for record in unit.get("repair_transaction_authorities") or []:
        if not isinstance(record, dict):
            continue
        if str(record.get("authority") or "") != (
            "active_patch_transaction_postimage"
        ):
            continue
        relative_path = normalize_relpath(str(record.get("file") or ""))
        start_line = int(record.get("start_line") or 0)
        end_line = int(record.get("end_line") or start_line)
        if (
            relative_path not in touched
            or start_line <= 0
            or end_line < start_line
            or not record.get("patch_ids")
        ):
            continue
        ranges[relative_path].add((start_line - 1, end_line))
    return {
        relative_path: sorted(file_ranges)
        for relative_path, file_ranges in ranges.items()
    }


def _shift_line_ranges_after_edit(
    ranges: list[tuple[int, int]],
    *,
    before: list[str],
    after: list[str],
) -> list[tuple[int, int]]:
    if len(before) == len(after):
        return ranges
    change_index = 0
    for change_index, (old_line, new_line) in enumerate(zip(before, after)):
        if old_line != new_line:
            break
    else:
        change_index = min(len(before), len(after))
    delta = len(after) - len(before)
    shifted: list[tuple[int, int]] = []
    for start, end in ranges:
        if change_index < start:
            shifted.append((max(0, start + delta), max(1, end + delta)))
        elif change_index <= end:
            shifted.append((start, max(start + 1, end + delta)))
        else:
            shifted.append((start, end))
    return shifted


def _extent_overlaps_interval(
    extent: object,
    *,
    start_line: int,
    end_line: int,
) -> bool:
    if not isinstance(extent, dict):
        return False
    owner_start = int(extent.get("start_line") or 0)
    owner_end = int(extent.get("end_line") or owner_start)
    return bool(
        owner_start > 0
        and owner_end >= owner_start
        and start_line <= owner_end
        and owner_start <= end_line
    )


def _physical_owner_extent_key(
    constraint: dict[str, Any],
    *,
    extent_field: str,
) -> tuple[str, int, int] | None:
    extent = constraint.get(extent_field)
    if not isinstance(extent, dict):
        return None
    start_line = int(extent.get("start_line") or 0)
    end_line = int(extent.get("end_line") or start_line)
    if start_line <= 0 or end_line < start_line:
        return None
    return (
        normalize_relpath(str(constraint.get("file") or "")),
        start_line,
        end_line,
    )


def _most_specific_owner_extent_groups(
    groups: dict[tuple[str, int, int], list[Any]],
) -> dict[tuple[str, int, int], list[Any]]:
    """Drop containing owners when a nested exact owner also matches."""

    result: dict[tuple[str, int, int], list[Any]] = {}
    for key, values in groups.items():
        relative_path, start_line, end_line = key
        contains_more_specific = any(
            other_path == relative_path
            and start_line <= other_start
            and other_end <= end_line
            and (start_line, end_line) != (other_start, other_end)
            for other_path, other_start, other_end in groups
        )
        if not contains_more_specific:
            result[key] = values
    return result


def _canonical_source_heading(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _disambiguate_owner_groups_by_current_hunk_section(
    groups: dict[tuple[str, int, int], list[Any]],
    *,
    hunk: list[str],
    current_lines: list[str],
) -> tuple[
    dict[tuple[str, int, int], list[Any]],
    dict[str, Any],
]:
    """Narrow existing owner proofs using a verified current heading.

    Unified-diff section text is model supplied, so it cannot grant write
    authority. It can only distinguish physical owners that already satisfy
    the hunk's independent current-text proof, and only after matching the
    exact heading at a current structural owner's start line.
    """

    if not hunk:
        return groups, {"status": "section_absent"}
    header_match = HUNK_HEADER_RE.match(hunk[0])
    section = (
        _canonical_source_heading(hunk[0][header_match.end() :])
        if header_match is not None
        else ""
    )
    if not section:
        return groups, {"status": "section_absent"}

    matches: dict[tuple[str, int, int], list[Any]] = {}
    for extent_key, values in groups.items():
        _relative_path, start_line, _end_line = extent_key
        if start_line <= 0 or start_line > len(current_lines):
            continue
        current_heading = _canonical_source_heading(
            current_lines[start_line - 1]
        )
        if current_heading == section:
            matches[extent_key] = values

    evidence = {
        "status": (
            "verified_unique_current_owner_heading"
            if len(matches) == 1
            else "current_owner_heading_not_found"
            if not matches
            else "current_owner_heading_ambiguous"
        ),
        "section": section,
        "matching_owner_extents": [
            {
                "file": relative_path,
                "start_line": start_line,
                "end_line": end_line,
            }
            for relative_path, start_line, end_line in sorted(matches)
        ],
    }
    if len(matches) == 1:
        return matches, evidence
    if len(matches) > 1:
        return matches, evidence
    return groups, evidence


def _physical_owner_alias_bundle(
    constraints: list[dict[str, Any]],
    *,
    relative_path: str,
    extent_field: str,
) -> dict[str, Any]:
    """Represent contract aliases backed by one exact source extent."""

    if not constraints:
        raise ValueError("physical owner bundle requires constraints")
    if len(constraints) == 1:
        return constraints[0]

    extent_keys = {
        _physical_owner_extent_key(
            constraint,
            extent_field=extent_field,
        )
        for constraint in constraints
    }
    if None in extent_keys or len(extent_keys) != 1:
        raise ValueError(
            "physical owner aliases must share one exact source extent"
        )
    _file, start_line, end_line = next(iter(extent_keys))
    owner_ids = sorted(
        {
            str(constraint.get("id") or "")
            for constraint in constraints
            if str(constraint.get("id") or "")
        }
    )
    contract_keys = sorted(
        {
            str(constraint.get("contract_key") or "")
            for constraint in constraints
            if str(constraint.get("contract_key") or "")
        }
    )
    owner_kinds = sorted(
        {
            str(constraint.get("owner_kind") or "")
            for constraint in constraints
            if str(constraint.get("owner_kind") or "")
        }
    )
    selected: dict[str, Any] = {
        "id": stable_id(
            "structural-rebase-physical-owner",
            normalize_relpath(relative_path),
            str(start_line),
            str(end_line),
            *owner_ids,
        ),
        "file": normalize_relpath(relative_path),
        "owner_kind": "exact_physical_owner_alias_bundle",
        "owner_kinds": owner_kinds,
        "contract_key": stable_id(
            "physical-owner-contract",
            *contract_keys,
        ),
        "contract_keys": contract_keys,
        "owner_ids": owner_ids,
        "current_status": (
            "resolved"
            if extent_field == "current_extent"
            else "entity_absent"
        ),
        extent_field: {
            "start_line": start_line,
            "end_line": end_line,
        },
    }
    other_extent_field = (
        "donor_extent"
        if extent_field == "current_extent"
        else "current_extent"
    )
    other_extents = {
        (
            int(extent.get("start_line") or 0),
            int(extent.get("end_line") or 0),
        )
        for constraint in constraints
        for extent in [constraint.get(other_extent_field)]
        if isinstance(extent, dict)
        and int(extent.get("start_line") or 0) > 0
        and int(extent.get("end_line") or 0)
        >= int(extent.get("start_line") or 0)
    }
    if len(other_extents) == 1:
        other_start, other_end = next(iter(other_extents))
        selected[other_extent_field] = {
            "start_line": other_start,
            "end_line": other_end,
        }
    return selected


def _structural_contracts_from_source(
    source: str,
    *,
    relative_path: str,
) -> set[tuple[str, str]]:
    """Parse exact entity contracts from a prospective inserted source block."""

    if not str(source).strip():
        return set()
    suffix = Path(relative_path).suffix or ".c"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        snippet_path = root / f"semantic-insertion{suffix}"
        snippet_path.write_text(str(source), encoding="utf-8")
        report = parse_structural_entity_file(
            root=root,
            relative_path=snippet_path.name,
            tree="prospective_patch",
        )
    if not report.get("parse_ok"):
        return set()
    return {
        (entity_kind(entity), entity_contract_key(entity))
        for entity in report.get("entities") or []
        if isinstance(entity, dict)
        and entity_kind(entity)
        and entity_contract_key(entity)
    }


def _current_top_level_include_block(
    spec: CaseSpec,
    *,
    relative_path: str,
    include_name: str,
) -> dict[str, Any]:
    """Resolve the current pre-declaration include block for one C source."""

    relative_path = normalize_relpath(relative_path)
    include_name = normalize_relpath(include_name)
    if not relative_path.endswith(".c") or not include_name:
        return {"status": "unsupported_consumer"}
    includes = parse_include_directives(
        root=spec.old_root_path,
        driver_root=spec.resolve_old(spec.driver_root),
        relative_path=relative_path,
        tree="old",
    )
    include_names = {
        normalize_relpath(str(item.get("name") or ""))
        for item in includes
        if isinstance(item, dict)
    }
    if include_name in include_names:
        return {"status": "already_present"}
    include_lines = sorted(
        int(item.get("line") or 0)
        for item in includes
        if isinstance(item, dict) and int(item.get("line") or 0) > 0
    )
    if not include_lines:
        return {"status": "include_block_absent"}
    path = spec.resolve_old(relative_path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    include_line_set = set(include_lines)
    top_level_include_lines: list[int] = []
    block_comment = False
    directive_continuation = False
    for line_number, line in enumerate(lines, start=1):
        if line_number in include_line_set:
            top_level_include_lines.append(line_number)
            directive_continuation = line.rstrip().endswith("\\")
            continue
        stripped = line.strip()
        if directive_continuation:
            directive_continuation = line.rstrip().endswith("\\")
            continue
        while block_comment:
            end = stripped.find("*/")
            if end < 0:
                stripped = ""
                break
            stripped = stripped[end + 2 :].lstrip()
            block_comment = False
        while stripped.startswith("/*"):
            end = stripped.find("*/", 2)
            if end < 0:
                block_comment = True
                stripped = ""
                break
            stripped = stripped[end + 2 :].lstrip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.startswith("#"):
            directive_continuation = line.rstrip().endswith("\\")
            continue
        break
    if not top_level_include_lines:
        return {"status": "include_block_not_top_level"}
    return {
        "status": "resolved",
        "start_line": top_level_include_lines[0],
        "end_line": top_level_include_lines[-1],
        "include_name": include_name,
    }


def _current_include_relation_boundary(
    spec: CaseSpec,
    *,
    relative_path: str,
    relation_mode: str,
    current_include_name: str,
    required_include_name: str,
    expected_line: int,
) -> dict[str, Any]:
    if relation_mode == "insert_provider":
        return _current_top_level_include_block(
            spec,
            relative_path=relative_path,
            include_name=required_include_name,
        )
    if relation_mode != "mutate_unresolved":
        return {"status": "unsupported_relation_mode"}
    current_include_name = normalize_relpath(current_include_name)
    matches = [
        item
        for item in parse_include_directives(
            root=spec.old_root_path,
            driver_root=spec.resolve_old(spec.driver_root),
            relative_path=normalize_relpath(relative_path),
            tree="old",
        )
        if isinstance(item, dict)
        and normalize_relpath(str(item.get("name") or ""))
        == current_include_name
        and int(item.get("line") or 0) > 0
    ]
    if expected_line > 0:
        matches = [
            item
            for item in matches
            if int(item.get("line") or 0) == expected_line
        ]
    if len(matches) != 1:
        return {
            "status": (
                "include_relation_absent"
                if not matches
                else "include_relation_ambiguous"
            )
        }
    line = int(matches[0]["line"])
    return {
        "status": "resolved",
        "start_line": line,
        "end_line": line,
        "include_name": current_include_name,
    }


def _include_directive_name(line: str) -> str:
    match = re.fullmatch(
        r"\s*#\s*include\s*[<\"](?P<name>[^>\"]+)[>\"]\s*",
        line,
    )
    if match is None:
        return ""
    return normalize_relpath(match.group("name"))


def _include_relation_constraints_for_hunk(
    hunk: list[str],
    constraints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Bind a pure include insertion, replacement, or deletion."""

    _old_start, _old_lines, _new_lines, removed_lines, added_lines = (
        _hunk_old_new_lines(hunk)
    )
    removed = [line for line in removed_lines if line.strip()]
    added = [line for line in added_lines if line.strip()]
    removed_names = [_include_directive_name(line) for line in removed]
    added_names = [_include_directive_name(line) for line in added]
    touches_include = bool(
        any(removed_names)
        or any(added_names)
        or any(re.match(r"^\s*#\s*include\b", line) for line in removed + added)
    )
    if not touches_include:
        return [], False
    if (
        not removed
        and len(added) == 1
        and len(added_names) == 1
        and added_names[0]
    ):
        return [
            constraint
            for constraint in constraints
            if str(constraint.get("owner_kind") or "")
            == "include_relation"
            and str(constraint.get("include_relation_mode") or "")
            == "insert_provider"
            and normalize_relpath(
                str(constraint.get("required_include_name") or "")
            )
            == added_names[0]
        ], True
    if (
        len(removed) == 1
        and bool(removed_names[0])
        and len(added) <= 1
        and all(added_names)
    ):
        return [
            constraint
            for constraint in constraints
            if str(constraint.get("owner_kind") or "")
            == "include_relation"
            and str(constraint.get("include_relation_mode") or "")
            == "mutate_unresolved"
            and normalize_relpath(
                str(constraint.get("current_include_name") or "")
            )
            == removed_names[0]
        ], True
    return [], True


def _exact_include_insertion_binding(
    current_lines: list[str],
    hunk: list[str],
    constraints: list[dict[str, Any]],
    *,
    relative_path: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Anchor one exact include insertion to a flat current include block."""

    _old_start, old_lines, _new_lines, removed_lines, added_lines = (
        _hunk_old_new_lines(hunk)
    )
    if old_lines or removed_lines or len(added_lines) != 1:
        return None, None
    added_include = _include_directive_name(added_lines[0])
    if not added_include:
        return None, None
    matches = [
        constraint
        for constraint in constraints
        if str(constraint.get("owner_kind") or "") == "include_relation"
        and str(constraint.get("include_relation_mode") or "")
        == "insert_provider"
        and normalize_relpath(
            str(constraint.get("required_include_name") or "")
        )
        == added_include
        and constraint.get("current_status") == "resolved"
        and isinstance(constraint.get("current_extent"), dict)
    ]
    matches_by_extent: dict[
        tuple[str, int, int], list[dict[str, Any]]
    ] = {}
    for constraint in matches:
        extent_key = _physical_owner_extent_key(
            constraint,
            extent_field="current_extent",
        )
        if extent_key is not None:
            matches_by_extent.setdefault(extent_key, []).append(constraint)
    if len(matches_by_extent) != 1:
        return None, None

    physical_matches = next(iter(matches_by_extent.values()))
    selected = _physical_owner_alias_bundle(
        physical_matches,
        relative_path=relative_path,
        extent_field="current_extent",
    )
    extent = selected.get("current_extent") or {}
    start = int(extent.get("start_line") or 0) - 1
    end = int(extent.get("end_line") or 0)
    if start < 0 or end <= start or end > len(current_lines):
        return None, None

    block_comment = False
    include_count = 0
    for line in current_lines[start:end]:
        stripped = line.strip()
        if block_comment:
            close = stripped.find("*/")
            if close < 0:
                continue
            stripped = stripped[close + 2 :].strip()
            block_comment = False
        while stripped.startswith("/*"):
            close = stripped.find("*/", 2)
            if close < 0:
                block_comment = True
                stripped = ""
                break
            stripped = stripped[close + 2 :].strip()
        if not stripped or stripped.startswith("//"):
            continue
        if not _include_directive_name(stripped):
            return None, None
        include_count += 1
    if block_comment or include_count == 0:
        return None, None

    conditional_depth = 0
    conditional_directive = re.compile(
        r"^\s*#\s*(?P<kind>if|ifdef|ifndef|endif)\b"
    )
    for line in current_lines[:end]:
        match = conditional_directive.match(line)
        if match is None:
            continue
        if match.group("kind") == "endif":
            conditional_depth -= 1
            if conditional_depth < 0:
                return None, None
        else:
            conditional_depth += 1
    if conditional_depth != 0:
        return None, None
    if any(
        _include_directive_name(line) == added_include
        for line in current_lines
    ):
        return None, None

    return selected, {
        "status": "bound",
        "basis": "current_exact_include_relation_insertion",
        "owner_id": selected.get("id"),
        "owner_kind": selected.get("owner_kind"),
        "contract_key": selected.get("contract_key"),
        "contract_keys": selected.get("contract_keys"),
        "owner_ids": selected.get("owner_ids"),
        "required_include_name": added_include,
        "insertion_index": end,
        "hunk_old_extent": {
            "start_line": int(
                HUNK_HEADER_RE.match(hunk[0]).group("old_start") or 0
            ),
            "end_line": int(
                HUNK_HEADER_RE.match(hunk[0]).group("old_start") or 0
            ),
        },
    }


def _bind_rebase_hunk_to_owner(
    hunk: list[str],
    owner_constraints: list[dict[str, Any]],
    *,
    current_lines: list[str] | None = None,
    relative_path: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Bind against current graph owners before using stale donor extents."""

    old_start, old_lines, _new_lines, removed_lines, added_lines = (
        _hunk_old_new_lines(hunk)
    )
    old_end = max(old_start, old_start + len(old_lines) - 1)
    if current_lines is not None:
        import_constraints = [
            constraint for constraint in owner_constraints
            if constraint.get("owner_kind") == "module_import_relation"
        ]
        if import_constraints and any(
            re.search(r"\bMODULE_IMPORT_NS\s*\(", line)
            for line in [*removed_lines, *added_lines]
        ):
            removed = [line for line in removed_lines if line.strip()]
            added = [line for line in added_lines if line.strip()]

            def import_namespace(line: str) -> str:
                calls = parse_module_metadata_calls(line)
                return (
                    str(calls[0]["namespace"] or "")
                    if len(calls) == 1
                    and calls[0]["name"] == "MODULE_IMPORT_NS"
                    and line.strip() == calls[0]["raw"].strip()
                    else ""
                )

            namespace = import_namespace(added[0]) if len(added) == 1 else ""
            mode = "replace" if len(removed) == 1 else "insert" if not removed else ""
            matches = [
                constraint for constraint in import_constraints
                if constraint.get("namespace") == namespace
                and constraint.get("relation_mode") == mode
                and constraint.get("current_status") == "resolved"
            ]
            if namespace and len(matches) == 1:
                selected = matches[0]
                line_index = int(
                    (selected.get("current_extent") or {}).get("start_line") or 0
                ) - 1
                if 0 <= line_index < len(current_lines):
                    current_line = current_lines[line_index]
                    valid = (
                        mode == "replace"
                        and import_namespace(removed[0]) == namespace
                        and current_line == removed[0]
                    ) or (
                        mode == "insert"
                        and len(parse_module_metadata_calls(current_line)) == 1
                        and parse_module_metadata_calls(current_line)[0]["name"]
                        == "MODULE_LICENSE"
                    )
                    if valid:
                        return selected, {
                            "status": "bound",
                            "basis": f"current_exact_module_import_{mode}",
                            "owner_id": selected["id"],
                            "owner_kind": "module_import_relation",
                            "contract_key": selected["contract_key"],
                            "replacement_index": line_index,
                            "insertion_index": line_index + 1,
                            "replacement_line": added[0],
                        }
            return None, {
                "status": "unbound",
                "reason": "module import mutation lacks one exact consumer relation",
                "hunk_old_extent": {"start_line": old_start, "end_line": old_end},
            }
        resolved_constraints = [
            constraint
            for constraint in owner_constraints
            if constraint.get("current_status") == "resolved"
            and isinstance(
                constraint.get("current_extent"), dict
            )
        ]
        include_constraints, touches_include = (
            _include_relation_constraints_for_hunk(
                hunk,
                resolved_constraints,
            )
        )
        if touches_include:
            if not include_constraints:
                return None, {
                    "status": "unbound",
                    "reason": (
                        "include mutation lacks one exact include-relation owner"
                    ),
                    "hunk_old_extent": {
                        "start_line": old_start,
                        "end_line": old_end,
                    },
                }
            resolved_constraints = include_constraints
            owner_constraints = include_constraints
            include_owner, include_binding = (
                _exact_include_insertion_binding(
                    current_lines,
                    hunk,
                    include_constraints,
                    relative_path=relative_path,
                )
            )
            if include_owner is not None and include_binding is not None:
                return include_owner, include_binding

        # A pure insertion can declare an exact graph entity that is absent
        # from the current tree while its surrounding context mentions other
        # resolved contracts. Bind the insertion by parsing the added entity,
        # not by guessing from stale line numbers or symbol substrings.
        if not removed_lines and added_lines:
            added_contracts = _structural_contracts_from_source(
                "\n".join(added_lines) + "\n",
                relative_path=relative_path,
            )
            absent_contract_matches = [
                constraint
                for constraint in owner_constraints
                if constraint.get("exact_absent_entity_insertion_allowed")
                and (
                    str(constraint.get("owner_kind") or ""),
                    str(constraint.get("contract_key") or ""),
                )
                in added_contracts
            ]
            matches_by_contract: dict[
                tuple[str, str], list[dict[str, Any]]
            ] = {}
            for constraint in absent_contract_matches:
                key = (
                    str(constraint.get("owner_kind") or ""),
                    str(constraint.get("contract_key") or ""),
                )
                matches_by_contract.setdefault(key, []).append(constraint)
            if len(matches_by_contract) == 1:
                contract_identity, physical_matches = next(
                    iter(matches_by_contract.items())
                )
                selected = _physical_owner_alias_bundle(
                    physical_matches,
                    relative_path=relative_path,
                    extent_field="donor_extent",
                )
                return selected, {
                    "status": "bound",
                    "basis": "added_entity_exact_absent_contract",
                    "owner_id": selected.get("id"),
                    "owner_kind": selected.get("owner_kind"),
                    "contract_key": selected.get("contract_key"),
                    "contract_keys": selected.get("contract_keys"),
                    "owner_ids": selected.get("owner_ids"),
                    "added_contract": {
                        "entity_kind": contract_identity[0],
                        "contract_key": contract_identity[1],
                    },
                    "hunk_old_extent": {
                        "start_line": old_start,
                        "end_line": old_end,
                    },
                }
            if len(matches_by_contract) > 1:
                return None, {
                    "status": "ambiguous",
                    "reason": (
                        "added source defines multiple exact graph entities "
                        "that are absent from the current tree"
                    ),
                    "added_contracts": [
                        {
                            "entity_kind": kind,
                            "contract_key": contract_key,
                        }
                        for kind, contract_key in sorted(
                            matches_by_contract
                        )
                    ],
                    "candidate_owner_ids": sorted(
                        str(item.get("id") or "")
                        for item in absent_contract_matches
                        if str(item.get("id") or "")
                    ),
                    "hunk_old_extent": {
                        "start_line": old_start,
                        "end_line": old_end,
                    },
                }

        # A pure insertion may use context on both sides of an owner
        # boundary. Prior edits can separate those sides while leaving one
        # complete side intact, so accept only a unique complete two-sided,
        # trailing, or leading context. Never shorten a context to guess the
        # owner. Existing lines remain immutable; only the proven boundary is
        # opened for insertion.
        if not removed_lines and added_lines and old_lines:
            leading_context, trailing_context = (
                _insertion_contexts(hunk)
            )
            inserted_export = _single_export_contract(
                added_lines, relative_path=relative_path
            )

            def unique_complete_context_match(
                context: list[str],
                *,
                context_kind: str,
            ) -> tuple[tuple[int, int], str] | None:
                if not context:
                    return None
                exact_matches = _find_sequence(
                    current_lines,
                    context,
                )
                if len(exact_matches) == 1:
                    start = exact_matches[0]
                    return (
                        (start, start + len(context)),
                        f"exact_{context_kind}",
                    )
                if exact_matches:
                    return None
                normalized_matches = _find_normalized_sequence(
                    current_lines,
                    context,
                )
                if len(normalized_matches) == 1:
                    return (
                        normalized_matches[0],
                        f"normalized_{context_kind}",
                    )
                return None

            context_proofs: list[
                tuple[tuple[int, int], str, int, str]
            ] = []
            full_match = unique_complete_context_match(
                old_lines,
                context_kind="full_hunk_preimage",
            )
            if full_match is not None:
                context_proofs.append(
                    (
                        full_match[0],
                        full_match[1],
                        len(leading_context),
                        "trailing" if trailing_context else "leading",
                    )
                )
            trailing_match = unique_complete_context_match(
                trailing_context,
                context_kind="full_trailing_context",
            )
            if trailing_match is not None:
                context_proofs.append(
                    (
                        trailing_match[0],
                        trailing_match[1],
                        0,
                        "trailing",
                    )
                )
            leading_match = unique_complete_context_match(
                leading_context,
                context_kind="full_leading_context",
            )
            if leading_match is not None:
                context_proofs.append(
                    (
                        leading_match[0],
                        leading_match[1],
                        len(leading_context),
                        "leading",
                    )
                )

            for (
                insertion_match,
                insertion_match_basis,
                insertion_offset,
                boundary_side,
            ) in context_proofs:
                insertion_index = (
                    insertion_match[0] + insertion_offset
                )
                boundary_candidates: list[dict[str, Any]] = []
                for constraint in resolved_constraints:
                    extent = constraint["current_extent"]
                    start = int(
                        extent.get("start_line") or 1
                    ) - 1
                    end = int(extent.get("end_line") or 0)
                    contains_boundary = (
                        start <= insertion_index < end
                        if boundary_side == "trailing"
                        else start < insertion_index <= end
                    )
                    if inserted_export is not None and insertion_index == end:
                        provider_kind = str(constraint.get("owner_kind") or "")
                        provider_symbol = str(inserted_export.get("symbol") or "")
                        contains_boundary = contains_boundary or bool(
                            _has_current_patch_authority(constraint)
                            and provider_kind in {"function", "variable"}
                            and provider_symbol
                            and constraint.get("contract_key")
                            == f"{provider_kind}:{provider_symbol}"
                        )
                    if contains_boundary:
                        boundary_candidates.append(constraint)
                candidates_by_extent: dict[
                    tuple[str, int, int],
                    list[dict[str, Any]],
                ] = {}
                for constraint in boundary_candidates:
                    extent_key = _physical_owner_extent_key(
                        constraint,
                        extent_field="current_extent",
                    )
                    if extent_key is None:
                        continue
                    candidates_by_extent.setdefault(
                        extent_key,
                        [],
                    ).append(constraint)
                if len(candidates_by_extent) == 1:
                    physical_matches = next(
                        iter(candidates_by_extent.values())
                    )
                    selected = _physical_owner_alias_bundle(
                        physical_matches,
                        relative_path=relative_path,
                        extent_field="current_extent",
                    )
                    return selected, {
                        "status": "bound",
                        "basis": (
                            "current_owner_exact_insertion_boundary"
                        ),
                        "preimage_basis": insertion_match_basis,
                        "boundary_side": boundary_side,
                        "insertion_line": insertion_index + 1,
                        "owner_id": selected.get("id"),
                        "owner_kind": selected.get("owner_kind"),
                        "contract_key": selected.get("contract_key"),
                        "contract_keys": selected.get("contract_keys"),
                        "owner_ids": selected.get("owner_ids"),
                        "physical_owner_alias_bundle": (
                            len(physical_matches) > 1
                        ),
                        "hunk_old_extent": {
                            "start_line": old_start,
                            "end_line": old_end,
                        },
                    }
                if len(candidates_by_extent) > 1:
                    return None, {
                        "status": "ambiguous",
                        "reason": (
                            "exact insertion boundary belongs to multiple "
                            "current owner contracts"
                        ),
                        "preimage_basis": insertion_match_basis,
                        "boundary_side": boundary_side,
                        "insertion_line": insertion_index + 1,
                        "candidate_owner_ids": sorted(
                            {
                                str(item.get("id") or "")
                                for matches in (
                                    candidates_by_extent.values()
                                )
                                for item in matches
                                if str(item.get("id") or "")
                            }
                        ),
                        "hunk_old_extent": {
                            "start_line": old_start,
                            "end_line": old_end,
                        },
                    }

        # A complete hunk preimage is an exact owner proof. Resolve it before
        # weaker single-block relocation, which can legitimately match the
        # same changed line in several different functions.
        for normalized in (False, True):
            preimage_matches: list[dict[str, Any]] = []
            if old_lines:
                for constraint in resolved_constraints:
                    current_extent = constraint["current_extent"]
                    allowed_ranges = [
                        (
                            int(
                                current_extent.get("start_line")
                                or 1
                            )
                            - 1,
                            int(
                                current_extent.get("end_line")
                                or 0
                            ),
                        )
                    ]
                    matches = (
                        _find_normalized_sequence(
                            current_lines,
                            old_lines,
                            allowed_ranges=allowed_ranges,
                        )
                        if normalized
                        else _find_sequence(
                            current_lines,
                            old_lines,
                            allowed_ranges=allowed_ranges,
                        )
                    )
                    if matches:
                        preimage_matches.append(constraint)
            preimage_matches_by_extent: dict[
                tuple[str, int, int],
                list[dict[str, Any]],
            ] = {}
            for constraint in preimage_matches:
                extent_key = _physical_owner_extent_key(
                    constraint,
                    extent_field="current_extent",
                )
                if extent_key is None:
                    continue
                preimage_matches_by_extent.setdefault(
                    extent_key,
                    [],
                ).append(constraint)
            preimage_matches_by_extent = (
                _most_specific_owner_extent_groups(
                    preimage_matches_by_extent
                )
            )
            hunk_section_binding: dict[str, Any] | None = None
            if len(preimage_matches_by_extent) > 1:
                (
                    preimage_matches_by_extent,
                    hunk_section_binding,
                ) = _disambiguate_owner_groups_by_current_hunk_section(
                    preimage_matches_by_extent,
                    hunk=hunk,
                    current_lines=current_lines,
                )
            if len(preimage_matches_by_extent) == 1:
                physical_matches = next(
                    iter(preimage_matches_by_extent.values())
                )
                selected = _physical_owner_alias_bundle(
                    physical_matches,
                    relative_path=relative_path,
                    extent_field="current_extent",
                )
                alias_bundle = len(physical_matches) > 1
                return selected, {
                    "status": "bound",
                    "basis": (
                        "current_owner_normalized_full_hunk_preimage"
                        if normalized
                        else "current_owner_exact_full_hunk_preimage"
                    ),
                    "owner_id": selected.get("id"),
                    "owner_kind": selected.get("owner_kind"),
                    "contract_key": selected.get("contract_key"),
                    "contract_keys": selected.get("contract_keys"),
                    "owner_ids": selected.get("owner_ids"),
                    "physical_owner_alias_bundle": alias_bundle,
                    "hunk_section_binding": hunk_section_binding,
                    "hunk_old_extent": {
                        "start_line": old_start,
                        "end_line": old_end,
                    },
                }
            if len(preimage_matches_by_extent) > 1:
                return None, {
                    "status": "ambiguous",
                    "reason": (
                        "complete hunk preimage occurs within multiple "
                        "current owner contracts"
                    ),
                    "match_basis": (
                        "normalized_full_hunk_preimage"
                        if normalized
                        else "exact_full_hunk_preimage"
                    ),
                    "hunk_section_binding": hunk_section_binding,
                    "hunk_old_extent": {
                        "start_line": old_start,
                        "end_line": old_end,
                    },
                    "candidate_owner_ids": sorted(
                        {
                            str(item.get("id") or "")
                            for matches in (
                                preimage_matches_by_extent.values()
                            )
                            for item in matches
                            if str(item.get("id") or "")
                        }
                    ),
                }

        # The unchanged context may name a neighboring entity even when all
        # edit blocks are inside one owner. Require one exact current preimage
        # before separating context placement from write authority.
        exact_file_matches = _find_sequence(current_lines, old_lines)
        if len(exact_file_matches) == 1:
            preimage_start = exact_file_matches[0]
            changed_owner_groups: dict[
                tuple[str, int, int], list[dict[str, Any]]
            ] = {}
            for constraint in resolved_constraints:
                if not _has_current_patch_authority(constraint):
                    continue
                extent = constraint["current_extent"]
                if not _hunk_changes_within_owner(
                    hunk,
                    preimage_start=preimage_start,
                    owner_start=int(extent.get("start_line") or 1) - 1,
                    owner_end=int(extent.get("end_line") or 0),
                ):
                    continue
                extent_key = _physical_owner_extent_key(
                    constraint,
                    extent_field="current_extent",
                )
                if extent_key is not None:
                    changed_owner_groups.setdefault(extent_key, []).append(
                        constraint
                    )
            changed_owner_groups = _most_specific_owner_extent_groups(
                changed_owner_groups
            )
            if len(changed_owner_groups) == 1:
                physical_matches = next(iter(changed_owner_groups.values()))
                selected = _physical_owner_alias_bundle(
                    physical_matches,
                    relative_path=relative_path,
                    extent_field="current_extent",
                )
                return selected, {
                    "status": "bound",
                    "basis": "current_owner_exact_full_hunk_changed_lines",
                    "owner_id": selected.get("id"),
                    "owner_kind": selected.get("owner_kind"),
                    "contract_key": selected.get("contract_key"),
                    "contract_keys": selected.get("contract_keys"),
                    "owner_ids": selected.get("owner_ids"),
                    "preimage_start_line": preimage_start + 1,
                    "hunk_old_extent": {
                        "start_line": old_start,
                        "end_line": old_end,
                    },
                }
            if len(changed_owner_groups) > 1:
                return None, {
                    "status": "ambiguous",
                    "reason": "changed hunk lines belong to multiple current owner contracts",
                    "candidate_owner_ids": sorted(
                        str(item.get("id") or "")
                        for group in changed_owner_groups.values()
                        for item in group
                        if str(item.get("id") or "")
                    ),
                    "hunk_old_extent": {
                        "start_line": old_start,
                        "end_line": old_end,
                    },
                }

        current_text_matches: list[
            tuple[dict[str, Any], str]
        ] = []
        for constraint in resolved_constraints:
            current_extent = constraint["current_extent"]
            allowed_ranges = [
                (
                    int(current_extent.get("start_line") or 1) - 1,
                    int(current_extent.get("end_line") or 0),
                )
            ]
            relocated, relocation_reason = _apply_relocatable_hunk(
                current_lines,
                hunk,
                allowed_ranges=allowed_ranges,
            )
            if relocated is None and relative_path:
                relocated, relocation_reason = (
                    _apply_disconnected_contract_blocks(
                        current_lines,
                        hunk,
                        relative_path=relative_path,
                        allowed_ranges=allowed_ranges,
                    )
                )
            if relocated is not None:
                current_text_matches.append(
                    (constraint, relocation_reason)
                )
        current_matches_by_extent: dict[
            tuple[str, int, int],
            list[tuple[dict[str, Any], str]],
        ] = {}
        for constraint, relocation_reason in current_text_matches:
            extent_key = _physical_owner_extent_key(
                constraint,
                extent_field="current_extent",
            )
            if extent_key is None:
                continue
            current_matches_by_extent.setdefault(
                extent_key,
                [],
            ).append((constraint, relocation_reason))
        current_matches_by_extent = _most_specific_owner_extent_groups(
            current_matches_by_extent
        )
        hunk_section_binding: dict[str, Any] | None = None
        if len(current_matches_by_extent) > 1:
            (
                current_matches_by_extent,
                hunk_section_binding,
            ) = _disambiguate_owner_groups_by_current_hunk_section(
                current_matches_by_extent,
                hunk=hunk,
                current_lines=current_lines,
            )
        if len(current_matches_by_extent) == 1:
            physical_matches = next(
                iter(current_matches_by_extent.values())
            )
            selected = _physical_owner_alias_bundle(
                [item for item, _reason in physical_matches],
                relative_path=relative_path,
                extent_field="current_extent",
            )
            relocation_reason = physical_matches[0][1]
            alias_bundle = len(physical_matches) > 1
            return selected, {
                "status": "bound",
                "basis": (
                    "current_physical_owner_alias_"
                    "text_relocation"
                    if alias_bundle
                    else "current_owner_text_relocation"
                ),
                "owner_id": selected.get("id"),
                "owner_kind": selected.get("owner_kind"),
                "contract_key": selected.get("contract_key"),
                "contract_keys": selected.get("contract_keys"),
                "owner_ids": selected.get("owner_ids"),
                "relocation_reason": relocation_reason,
                "hunk_section_binding": hunk_section_binding,
                "hunk_old_extent": {
                    "start_line": old_start,
                    "end_line": old_end,
                },
            }
        if len(current_matches_by_extent) > 1:
            return None, {
                "status": "ambiguous",
                "reason": (
                    "hunk text relocates within multiple current owner "
                    "contracts"
                ),
                "hunk_section_binding": hunk_section_binding,
                "hunk_old_extent": {
                    "start_line": old_start,
                    "end_line": old_end,
                },
                "candidate_owner_ids": [
                    str(item.get("id") or "")
                    for item, _reason in current_text_matches
                ],
            }

        positional_bundle_candidates = [
            constraint
            for constraint in resolved_constraints
            if _extent_overlaps_interval(
                constraint.get("donor_extent"),
                start_line=old_start,
                end_line=old_end,
            )
            or _extent_overlaps_interval(
                constraint.get("current_extent"),
                start_line=old_start,
                end_line=old_end,
            )
        ]
        if len(positional_bundle_candidates) > 1:
            bundle_start = min(
                int(
                    constraint["current_extent"].get(
                        "start_line"
                    )
                    or 1
                )
                for constraint in positional_bundle_candidates
            )
            bundle_end = max(
                int(
                    constraint["current_extent"].get(
                        "end_line"
                    )
                    or 0
                )
                for constraint in positional_bundle_candidates
            )
            bundle_constraints = [
                constraint
                for constraint in resolved_constraints
                if _extent_overlaps_interval(
                    constraint.get("current_extent"),
                    start_line=bundle_start,
                    end_line=bundle_end,
                )
            ]
            if len(bundle_constraints) > 1 and all(
                _has_current_patch_authority(constraint)
                for constraint in bundle_constraints
            ):
                bundle_ranges = [
                    (bundle_start - 1, bundle_end)
                ]
                relocated, relocation_reason = (
                    _apply_relocatable_hunk(
                        current_lines,
                        hunk,
                        allowed_ranges=bundle_ranges,
                    )
                )
                if relocated is None and relative_path:
                    relocated, relocation_reason = (
                        _apply_disconnected_contract_blocks(
                            current_lines,
                            hunk,
                            relative_path=relative_path,
                            allowed_ranges=bundle_ranges,
                        )
                    )
                if relocated is not None:
                    owner_ids = sorted(
                        str(item.get("id") or "")
                        for item in bundle_constraints
                        if str(item.get("id") or "")
                    )
                    contract_keys = sorted(
                        {
                            str(
                                item.get("contract_key")
                                or ""
                            )
                            for item in bundle_constraints
                            if str(
                                item.get("contract_key")
                                or ""
                            )
                        }
                    )
                    donor_extents = [
                        item.get("donor_extent")
                        for item in bundle_constraints
                        if isinstance(
                            item.get("donor_extent"), dict
                        )
                    ]
                    donor_start = min(
                        (
                            int(
                                extent.get("start_line")
                                or 0
                            )
                            for extent in donor_extents
                            if int(
                                extent.get("start_line")
                                or 0
                            )
                            > 0
                        ),
                        default=0,
                    )
                    donor_end = max(
                        (
                            int(
                                extent.get("end_line")
                                or 0
                            )
                            for extent in donor_extents
                        ),
                        default=0,
                    )
                    selected = {
                        "id": stable_id(
                            "structural-rebase-owner-bundle",
                            relative_path,
                            *owner_ids,
                        ),
                        "file": relative_path,
                        "owner_kind": (
                            "exact_structural_owner_bundle"
                        ),
                        "contract_key": stable_id(
                            "contract-bundle",
                            *contract_keys,
                        ),
                        "contract_keys": contract_keys,
                        "owner_ids": owner_ids,
                        "current_status": "resolved",
                        "current_extent": {
                            "start_line": bundle_start,
                            "end_line": bundle_end,
                        },
                        "donor_extent": (
                            {
                                "start_line": donor_start,
                                "end_line": donor_end,
                            }
                            if donor_start > 0
                            and donor_end >= donor_start
                            else None
                        ),
                    }
                    return selected, {
                        "status": "bound",
                        "basis": (
                            "current_exact_owner_bundle_"
                            "text_relocation"
                        ),
                        "owner_id": selected["id"],
                        "owner_kind": selected["owner_kind"],
                        "contract_key": selected[
                            "contract_key"
                        ],
                        "contract_keys": contract_keys,
                        "owner_ids": owner_ids,
                        "relocation_reason": relocation_reason,
                        "hunk_old_extent": {
                            "start_line": old_start,
                            "end_line": old_end,
                        },
                    }

        donor_matches = [
            constraint
            for constraint in owner_constraints
            if _extent_overlaps_interval(
                constraint.get("donor_extent"),
                start_line=old_start,
                end_line=old_end,
            )
        ]
        absent_donor_matches = [
            constraint
            for constraint in donor_matches
            if constraint.get("current_status") != "resolved"
            or not isinstance(
                constraint.get("current_extent"), dict
            )
        ]
        absent_matches_by_extent: dict[
            tuple[str, int, int],
            list[dict[str, Any]],
        ] = {}
        for constraint in absent_donor_matches:
            extent_key = _physical_owner_extent_key(
                constraint,
                extent_field="donor_extent",
            )
            if extent_key is None:
                continue
            absent_matches_by_extent.setdefault(
                extent_key,
                [],
            ).append(constraint)
        if len(absent_matches_by_extent) == 1:
            physical_matches = next(
                iter(absent_matches_by_extent.values())
            )
            selected = _physical_owner_alias_bundle(
                physical_matches,
                relative_path=relative_path,
                extent_field="donor_extent",
            )
            alias_bundle = len(physical_matches) > 1
            return selected, {
                "status": "bound",
                "basis": (
                    "donor_physical_owner_alias_current_absent"
                    if alias_bundle
                    else "donor_identity_current_owner_absent"
                ),
                "owner_id": selected.get("id"),
                "owner_kind": selected.get("owner_kind"),
                "contract_key": selected.get("contract_key"),
                "contract_keys": selected.get("contract_keys"),
                "owner_ids": selected.get("owner_ids"),
                "hunk_old_extent": {
                    "start_line": old_start,
                    "end_line": old_end,
                },
            }
        if len(absent_matches_by_extent) > 1:
            return None, {
                "status": "ambiguous",
                "reason": (
                    "hunk identifies multiple donor entities absent from "
                    "the current tree"
                ),
                "hunk_old_extent": {
                    "start_line": old_start,
                    "end_line": old_end,
                },
                "candidate_owner_ids": [
                    str(item.get("id") or "")
                    for item in absent_donor_matches
                ],
            }
        if len(owner_constraints) == 1:
            only_owner = owner_constraints[0]
            if (
                only_owner.get("current_status") != "resolved"
                or not isinstance(
                    only_owner.get("current_extent"), dict
                )
            ):
                return only_owner, {
                    "status": "bound",
                    "basis": "only_exact_current_owner_absent",
                    "owner_id": only_owner.get("id"),
                    "owner_kind": only_owner.get("owner_kind"),
                    "contract_key": only_owner.get("contract_key"),
                    "hunk_old_extent": {
                        "start_line": old_start,
                        "end_line": old_end,
                    },
                }
        return None, {
            "status": "unbound",
            "reason": (
                "hunk text does not relocate within exactly one current "
                "graph owner"
            ),
            "hunk_old_extent": {
                "start_line": old_start,
                "end_line": old_end,
            },
            "candidate_owner_ids": [
                str(item.get("id") or "")
                for item in owner_constraints
            ],
        }
    raise ValueError(
        "structural rebase requires current-source owner binding"
    )


def _shift_owner_constraints_after_edit(
    constraints: list[dict[str, Any]],
    *,
    before: list[str],
    after: list[str],
    insertion_index: int | None = None,
) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    insertion_delta = len(after) - len(before)
    for constraint in constraints:
        current_extent = constraint.get("current_extent")
        updated = dict(constraint)
        if isinstance(current_extent, dict):
            start = int(current_extent.get("start_line") or 1) - 1
            end = int(current_extent.get("end_line") or 0)
            if insertion_index is not None and insertion_delta > 0:
                if end <= insertion_index:
                    ranges = [(start, end)]
                elif start >= insertion_index:
                    ranges = [
                        (start + insertion_delta, end + insertion_delta)
                    ]
                else:
                    ranges = [(start, end + insertion_delta)]
            else:
                ranges = _shift_line_ranges_after_edit(
                    [(start, end)],
                    before=before,
                    after=after,
                )
            if ranges:
                start, end = ranges[0]
                updated["current_extent"] = {
                    "start_line": start + 1,
                    "end_line": end,
                }
        shifted.append(updated)
    return shifted


def rebase_unified_diff_to_current_tree(
    spec: CaseSpec,
    *,
    diff_text: str,
    touched_files: list[str],
    structural_ranges_by_file: dict[
        str,
        list[tuple[int, int]],
    ] | None = None,
    structural_owner_constraints_by_file: dict[
        str,
        list[dict[str, Any]],
    ] | None = None,
    transaction_authority_ranges_by_file: dict[
        str,
        list[tuple[int, int]],
    ] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    valid_structure, structure_error = _validate_unified_diff_structure(diff_text)
    if not valid_structure:
        return None, {"status": "not_rebased", "reason": structure_error or "invalid unified diff structure"}

    records = _split_unified_diff_records(diff_text)
    if not records:
        return None, {"status": "not_rebased", "reason": "no unified diff records"}

    changed_files: dict[str, tuple[list[str], list[str]]] = {}
    current_text_records: list[str] = []
    details: list[dict[str, Any]] = []
    for record in records:
        relative_path = normalize_relpath(str(record.get("file") or ""))
        if relative_path not in {normalize_relpath(item) for item in touched_files}:
            return None, {"status": "not_rebased", "reason": f"unexpected diff file {relative_path}"}
        path = spec.resolve_old(relative_path)
        if not path.exists():
            return None, {"status": "not_rebased", "reason": f"target file missing: {relative_path}"}
        original = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        modified = list(original)
        allowed_ranges = (
            list(structural_ranges_by_file[relative_path])
            if structural_ranges_by_file is not None
            and relative_path in structural_ranges_by_file
            else None
        )
        owner_constraints = (
            [
                dict(item)
                for item in structural_owner_constraints_by_file[
                    relative_path
                ]
                if isinstance(item, dict)
            ]
            if structural_owner_constraints_by_file is not None
            and relative_path
            in structural_owner_constraints_by_file
            else None
        )
        transaction_authority_ranges = (
            list(
                transaction_authority_ranges_by_file[relative_path]
            )
            if transaction_authority_ranges_by_file is not None
            and relative_path in transaction_authority_ranges_by_file
            else []
        )
        if not owner_constraints and not transaction_authority_ranges:
            record_text = _render_unified_diff_record(record)
            if record_text is None:
                return None, {
                    "status": "not_rebased",
                    "reason": f"invalid diff record for {relative_path}",
                    "details": details,
                }
            current_text_records.append(record_text)
            details.append(
                {
                    "file": relative_path,
                    "status": "current_text_required",
                    "reason": (
                        "authorized file-level record has no entity "
                        "relocation authority"
                    ),
                }
            )
            continue
        if not owner_constraints and transaction_authority_ranges:
            owner_constraints = None
            allowed_ranges = list(transaction_authority_ranges)
        for hunk_index, hunk in enumerate(record.get("hunks") or [], start=1):
            before_hunk = modified
            owner_binding: dict[str, Any] | None = (
                {
                    "status": "bound",
                    "basis": "active_patch_transaction_postimage",
                    "transaction_ranges": [
                        {
                            "start_line": start + 1,
                            "end_line": end,
                        }
                        for start, end in transaction_authority_ranges
                    ],
                }
                if owner_constraints is None
                and transaction_authority_ranges
                else None
            )
            hunk_allowed_ranges = allowed_ranges
            exact_absent_insertion = False
            exact_relation_edit: dict[str, Any] | None = None
            transaction_updated: list[str] | None = None
            transaction_reason: str | None = None
            if owner_constraints is not None:
                (
                    _old_start,
                    _old_lines,
                    _new_lines,
                    removed_lines,
                    _added_lines,
                ) = _hunk_old_new_lines(hunk)
                bound_owner, owner_binding = (
                    _bind_rebase_hunk_to_owner(
                        hunk,
                        owner_constraints,
                        current_lines=modified,
                        relative_path=relative_path,
                    )
                )
                if bound_owner is None and not exact_absent_insertion:
                    transaction_updated, transaction_reason = (
                        _apply_relocatable_hunk(
                            modified,
                            hunk,
                            allowed_ranges=(
                                transaction_authority_ranges
                            ),
                        )
                        if transaction_authority_ranges
                        else (None, "no active transaction range")
                    )
                    if (
                        transaction_updated is None
                        and transaction_authority_ranges
                    ):
                        (
                            transaction_updated,
                            transaction_reason,
                        ) = _apply_disconnected_contract_blocks(
                            modified,
                            hunk,
                            relative_path=relative_path,
                            allowed_ranges=(
                                transaction_authority_ranges
                            ),
                        )
                    if transaction_updated is not None:
                        bound_owner = {
                            "id": stable_id(
                                "transaction-repair-authority",
                                relative_path,
                                *[
                                    f"{start}:{end}"
                                    for start, end in (
                                        transaction_authority_ranges
                                    )
                                ],
                            ),
                            "current_status": "resolved",
                            "transaction_authority": True,
                            "current_extent": {
                                "start_line": min(
                                    start
                                    for start, _end in (
                                        transaction_authority_ranges
                                    )
                                )
                                + 1,
                                "end_line": max(
                                    end
                                    for _start, end in (
                                        transaction_authority_ranges
                                    )
                                ),
                            },
                        }
                        owner_binding = {
                            "status": "bound",
                            "basis": (
                                "active_patch_transaction_postimage"
                            ),
                            "owner_id": bound_owner["id"],
                            "transaction_ranges": [
                                {
                                    "start_line": start + 1,
                                    "end_line": end,
                                }
                                for start, end in (
                                    transaction_authority_ranges
                                )
                            ],
                            "relocation_reason": transaction_reason,
                        }
                        hunk_allowed_ranges = list(
                            transaction_authority_ranges
                        )
                    else:
                        details.append(
                            {
                                "file": relative_path,
                                "hunk": hunk_index,
                                "reason": owner_binding.get(
                                    "reason"
                                ),
                                "owner_binding": owner_binding,
                                "transaction_authority": {
                                    "ranges": [
                                        {
                                            "start_line": start + 1,
                                            "end_line": end,
                                        }
                                        for start, end in (
                                            transaction_authority_ranges
                                        )
                                    ],
                                    "relocation_reason": (
                                        transaction_reason
                                    ),
                                },
                            }
                        )
                        return None, {
                            "status": "not_rebased",
                            "reason": owner_binding.get(
                                "reason"
                            ),
                            "details": details,
                        }
                current_extent = (
                    bound_owner.get("current_extent")
                    if bound_owner is not None
                    else None
                )
                if not exact_absent_insertion and (
                    bound_owner.get("current_status") != "resolved"
                    or not isinstance(current_extent, dict)
                ):
                    if bound_owner.get(
                        "exact_absent_entity_insertion_allowed"
                    ) and not removed_lines:
                        hunk_allowed_ranges = None
                        exact_absent_insertion = True
                        owner_binding = {
                            **owner_binding,
                            "current_status": bound_owner.get(
                                "current_status"
                            ),
                            "relocation_basis": (
                                "exact_graph_absent_entity_insertion"
                                if bound_owner.get(
                                    "exact_absent_entity_insertion_allowed"
                                )
                                else None
                            ),
                        }
                    else:
                        reason = (
                            "bound donor owner is not uniquely present in "
                            "the current tree; entity materialization is "
                            "required"
                        )
                        details.append(
                            {
                                "file": relative_path,
                                "hunk": hunk_index,
                                "reason": reason,
                                "owner_binding": owner_binding,
                                "current_status": bound_owner.get(
                                    "current_status"
                                ),
                            }
                        )
                        return None, {
                            "status": "not_rebased",
                            "reason": reason,
                            "details": details,
                        }
                elif (
                    not exact_absent_insertion
                    and not bound_owner.get("transaction_authority")
                ):
                    hunk_allowed_ranges = (
                        None
                        if (
                            owner_binding.get("basis")
                            == "current_owner_exact_full_hunk_changed_lines"
                            or (
                                not removed_lines
                                and owner_binding.get("basis")
                                == "current_owner_exact_insertion_boundary"
                                and owner_binding.get("preimage_basis")
                                == "exact_full_hunk_preimage"
                            )
                        )
                        else [
                            (
                                int(current_extent.get("start_line") or 1) - 1,
                                int(current_extent.get("end_line") or 0),
                            )
                        ]
                    )
                if owner_binding.get("basis") in {
                    "current_exact_include_relation_insertion",
                    "current_exact_module_import_insert",
                    "current_exact_module_import_replace",
                }:
                    exact_relation_edit = owner_binding
            if transaction_updated is not None:
                updated, reason = (
                    transaction_updated,
                    str(transaction_reason or "transaction postimage match"),
                )
            elif exact_absent_insertion:
                updated, reason = (
                    _apply_unique_context_insertion_hunk(
                        modified,
                        hunk,
                    )
                )
            elif exact_relation_edit is not None:
                (
                    _old_start,
                    old_lines,
                    _new_lines,
                    removed_lines,
                    added_lines,
                ) = _hunk_old_new_lines(hunk)
                insertion_index = int(
                    exact_relation_edit.get("insertion_index", -1)
                )
                replacement_index = int(
                    exact_relation_edit.get("replacement_index", -1)
                )
                relation_basis = str(exact_relation_edit.get("basis") or "")
                if (
                    relation_basis == "current_exact_module_import_replace"
                    and len(added_lines) == 1
                    and 0 <= replacement_index < len(modified)
                ):
                    updated = list(modified)
                    updated[replacement_index] = added_lines[0]
                    reason = "replaced_by_exact_module_import_relation"
                elif (
                    relation_basis in {
                        "current_exact_include_relation_insertion",
                        "current_exact_module_import_insert",
                    }
                    and (not old_lines or relation_basis.endswith("_insert"))
                    and not removed_lines
                    and len(added_lines) == 1
                    and 0 <= insertion_index <= len(modified)
                ):
                    updated = (
                        modified[:insertion_index]
                        + added_lines
                        + modified[insertion_index:]
                    )
                    reason = "inserted_by_exact_relation"
                else:
                    updated = None
                    reason = "exact relation edit proof became invalid"
            else:
                updated, reason = _apply_relocatable_hunk(
                    modified,
                    hunk,
                    allowed_ranges=hunk_allowed_ranges,
                )
            if updated is None:
                contract_updated, contract_reason = (
                    _apply_disconnected_contract_blocks(
                        modified,
                        hunk,
                        relative_path=relative_path,
                        allowed_ranges=hunk_allowed_ranges,
                    )
                )
                if contract_updated is not None:
                    updated = contract_updated
                    reason = contract_reason
                else:
                    reason = (
                        f"{reason}; static contract materialization failed: "
                        f"{contract_reason}"
                    )
            details.append(
                {
                    "file": relative_path,
                    "hunk": hunk_index,
                    "reason": reason,
                    "owner_binding": owner_binding,
                }
            )
            if updated is None:
                return None, {
                    "status": "not_rebased",
                    "reason": reason,
                    "details": details,
                }
            modified = updated
            if allowed_ranges is not None:
                allowed_ranges = _shift_line_ranges_after_edit(
                    allowed_ranges,
                    before=before_hunk,
                    after=modified,
                )
            if owner_constraints is not None:
                owner_constraints = (
                    _shift_owner_constraints_after_edit(
                        owner_constraints,
                        before=before_hunk,
                        after=modified,
                        insertion_index=(
                            int(owner_binding["insertion_line"]) - 1
                            if (
                                not removed_lines
                                and owner_binding.get("basis")
                                == "current_owner_exact_insertion_boundary"
                                and owner_binding.get("preimage_basis")
                                == "exact_full_hunk_preimage"
                                and owner_binding.get("insertion_line")
                            )
                            else None
                        ),
                    )
                )
            if transaction_authority_ranges:
                transaction_authority_ranges = (
                    _shift_line_ranges_after_edit(
                        transaction_authority_ranges,
                        before=before_hunk,
                        after=modified,
                    )
                )
        if modified != original:
            changed_files[relative_path] = (original, modified)

    if not changed_files and not current_text_records:
        return None, {"status": "not_rebased", "reason": "rebased diff would be empty", "details": details}

    rebased_lines: list[str] = []
    for relative_path, (original, modified) in changed_files.items():
        rebased_lines.extend(
            difflib.unified_diff(
                original,
                modified,
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
                lineterm="",
                n=3,
            )
        )
    rebased = "\n".join(rebased_lines)
    if rebased and not rebased.endswith("\n"):
        rebased += "\n"
    rebased += "".join(current_text_records)
    return rebased, {"status": "rebased", "details": details}


def _write_generated_patch(response_dir: Path, unit_id: str, diff_text: str) -> Path:
    patch_dir = ensure_dir(response_dir / "patches")
    safe_unit_id = re.sub(r"[^A-Za-z0-9._-]+", "-", unit_id).strip("-") or "unit"
    patch_path = patch_dir / f"{safe_unit_id}.patch"
    patch_path.write_text(diff_text, encoding="utf-8")
    return patch_path.resolve()


def _allowed_response_targets(unit: dict[str, Any], *, allow_escalation: bool) -> set[str]:
    allowed = {normalize_relpath(item) for item in unit.get("allowed_fileset") or []}
    if allow_escalation:
        allowed.update(normalize_relpath(item) for item in unit.get("escalation_candidates") or [])
    return allowed



def _interpret_response_text(response_text: str) -> dict[str, Any]:
    structured = extract_json_payload(response_text)
    if structured is None:
        raw_diff = _extract_raw_unified_diff(response_text)
        if raw_diff is not None:
            raw_diff, mechanical_repair = repair_unified_diff_structure(
                raw_diff
            )
            valid, error = _validate_unified_diff_structure(raw_diff)
            if not valid:
                raise ResponseInterpretationError(
                    error or "invalid unified diff structure",
                    stage="unified_diff_structure",
                    finding_kind="invalid_raw_unified_diff",
                    immutable_patch_text=raw_diff,
                )
            return {
                "kind": "raw_unified_diff",
                "action": "apply_patch",
                "payload": {},
                "diff_text": raw_diff,
                "mechanical_repair": mechanical_repair,
            }
        raise ResponseInterpretationError(
            "response is not valid structured_json_v1: "
            + structured_json_error_summary(response_text),
            stage="response_envelope",
            finding_kind="unparseable_response_envelope",
        )

    action = str(structured.get("action") or "").strip()
    if not action and str(structured.get("status") or "").strip() == "needs_escalation":
        action = str(structured.get("status") or "").strip()
    if action not in {"apply_patch", "no_patch", "needs_escalation"}:
        raise ResponseInterpretationError(
            f"unsupported structured response action: {action}",
            stage="response_envelope",
            finding_kind="unsupported_response_action",
        )
    if action == "needs_escalation":
        missing_context = structured.get("missing_context")
        if not isinstance(missing_context, dict):
            raise ResponseInterpretationError(
                "needs_escalation response requires missing_context object",
                stage="response_envelope",
                finding_kind="invalid_escalation_envelope",
            )
        unexpected = sorted(
            set(missing_context)
            - {
                "work_unit_id",
                "diagnostic_event_ids",
                "graph_node_ids",
                "continuation",
                "reason",
            }
        )
        if unexpected:
            raise ResponseInterpretationError(
                "needs_escalation cannot request guessed files or symbols, unexpected fields: "
                + ", ".join(unexpected),
                stage="target_policy",
                finding_kind="guessed_scope_target",
            )
    if action == "apply_patch":
        try:
            diff_text = diff_from_structured_payload(structured)
        except ValueError as exc:
            patch = (
                structured.get("patch")
                if isinstance(structured.get("patch"), dict)
                else {}
            )
            raw_content = patch.get("content")
            if raw_content is None and isinstance(
                patch.get("content_lines"),
                list,
            ):
                raw_content = "\n".join(
                    str(line) for line in patch["content_lines"]
                )
            immutable_patch_text = (
                str(raw_content)
                if raw_content is not None and str(raw_content).strip()
                else None
            )
            raise ResponseInterpretationError(
                str(exc),
                stage=(
                    "unified_diff_structure"
                    if immutable_patch_text
                    else "response_envelope"
                ),
                finding_kind=(
                    "invalid_structured_patch_representation"
                    if immutable_patch_text
                    else "missing_structured_patch_payload"
                ),
                immutable_patch_text=immutable_patch_text,
            ) from exc
        diff_text, mechanical_repair = repair_unified_diff_structure(diff_text)
        valid, error = _validate_unified_diff_structure(diff_text)
        if not valid:
            raise ResponseInterpretationError(
                error or "invalid unified diff structure",
                stage="unified_diff_structure",
                finding_kind="invalid_structured_unified_diff",
                immutable_patch_text=diff_text,
            )
        return {
            "kind": "structured",
            "action": action,
            "payload": structured,
            "diff_text": diff_text,
            "mechanical_repair": mechanical_repair,
        }
    if action == "no_patch":
        decision = structured.get("decision")
        if not isinstance(decision, dict) or not str(decision.get("rationale") or "").strip():
            raise ResponseInterpretationError(
                "no_patch response requires decision.rationale",
                stage="response_envelope",
                finding_kind="invalid_no_patch_envelope",
            )
    return {
        "kind": "structured",
        "action": action,
        "payload": structured,
        "diff_text": None,
    }


def _validate_structured_patch_targets(payload: dict[str, Any], touched_files: list[str]) -> tuple[bool, list[str]]:
    patch_plan = dict(payload.get("patch_plan") or {})
    affected_files = [normalize_relpath(item) for item in patch_plan.get("affected_files") or [] if str(item).strip()]
    if not affected_files:
        return True, []
    touched = {normalize_relpath(item) for item in touched_files}
    mismatches = [item for item in affected_files if item not in touched]
    return not mismatches, mismatches


def _manual_macro_shim_finding(
    unit: dict[str, Any],
    diff_text: str,
) -> dict[str, Any] | None:
    """Reject guessed definitions for an exact missing macro contract.

    A manual same-named macro often clears one parser error while silently
    omitting the shared provider's behavior or dependencies. Exact provider
    materialization has its own graph-bound operation; an apply_patch repair
    for a missing macro must therefore adapt the authorized consumers instead
    of synthesizing a local shim.
    """

    if unit.get("controller_entity_materialization"):
        return None

    contract_names: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                collect(nested)
            return
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                collect(nested)
            return
        if not isinstance(value, str):
            return
        match = re.fullmatch(r"macro:([A-Za-z_][A-Za-z0-9_]*)", value)
        if match:
            contract_names.add(match.group(1))

    collect(unit)
    if not contract_names:
        return None
    added_definitions = {
        match.group(1)
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
        for match in [
            re.match(
                r"^\+\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                line,
            )
        ]
        if match is not None
    }
    blocked = sorted(contract_names.intersection(added_definitions))
    if not blocked:
        return None
    return {
        "schema": "manual_macro_shim_finding_v1",
        "status": "blocked",
        "macro_names": blocked,
        "reason": (
            "manual apply_patch attempted to define an exact missing macro "
            "contract; use graph-bound provider materialization or adapt all "
            "authorized consumers to target primitives"
        ),
    }


def _manual_missing_provider_definition_finding(
    unit: dict[str, Any],
    provider_contract_report: dict[str, Any],
) -> dict[str, Any] | None:
    """Reject model-defined implementations for graph-selected absent roots."""

    if unit.get("controller_entity_materialization"):
        return None

    root_records: list[dict[str, Any]] = []

    def collect_root_records(value: object) -> None:
        if isinstance(value, dict):
            contract_key = str(value.get("contract_key") or "")
            tree = str(value.get("tree") or "")
            if contract_key and tree:
                root_records.append(
                    {
                        "contract_key": contract_key,
                        "tree": tree,
                        "entity_kind": str(
                            value.get("entity_kind")
                            or value.get("kind")
                            or ""
                        ).lower(),
                        "file": normalize_relpath(
                            str(value.get("file") or "")
                        ),
                        "entity_id": str(
                            value.get("entity_id")
                            or value.get("node_id")
                            or ""
                        ),
                    }
                )
            for nested in value.values():
                collect_root_records(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                collect_root_records(nested)

    for field in (
        "root_cause_cluster",
        "code_graph_root_cause_cluster",
    ):
        collect_root_records(unit.get(field))

    trees_by_contract: dict[str, set[str]] = {}
    records_by_contract: dict[str, list[dict[str, Any]]] = {}
    for record in root_records:
        contract_key = record["contract_key"]
        trees_by_contract.setdefault(contract_key, set()).add(
            record["tree"]
        )
        records_by_contract.setdefault(contract_key, []).append(record)
    donor_only_contracts = {
        contract_key
        for contract_key, trees in trees_by_contract.items()
        if any(tree.startswith("donor_") for tree in trees)
        and not any(tree == "old" or tree.startswith("old_") for tree in trees)
    }
    if not donor_only_contracts:
        return None

    provider_kinds = {"function", "macro", "variable"}
    representation_authority = dict(
        unit.get("target_representation_adaptation_authority") or {}
    )
    repair_decision = dict(unit.get("repair_decision") or {})
    authority_root_id = str(
        representation_authority.get("donor_entity_id") or ""
    )
    selected_root_id = str(repair_decision.get("root_node_id") or "")

    def target_representation_adaptation_authorized(
        record: dict[str, Any],
    ) -> bool:
        return bool(
            representation_authority.get("status") == "authorized"
            and authority_root_id
            and selected_root_id == authority_root_id
            and str(record.get("key") or "")
            == str(representation_authority.get("contract_key") or "")
            and str(record.get("kind") or "").lower()
            == str(
                representation_authority.get("entity_kind") or ""
            ).lower()
            and normalize_relpath(str(record.get("file") or ""))
            == normalize_relpath(
                str(representation_authority.get("owner_file") or "")
            )
        )

    blocked = [
        dict(record)
        for record in provider_contract_report.get(
            "added_contract_definitions"
        )
        or []
        if isinstance(record, dict)
        and str(record.get("kind") or "").lower()
        in provider_kinds
        and str(record.get("key") or "") in donor_only_contracts
        and not target_representation_adaptation_authorized(record)
    ]
    if not blocked:
        return None

    blocked_contracts = sorted(
        {str(record.get("key") or "") for record in blocked}
    )
    return {
        "schema": "manual_missing_provider_definition_finding_v1",
        "status": "blocked",
        "contract_keys": blocked_contracts,
        "added_definitions": blocked,
        "root_evidence": [
            record
            for contract_key in blocked_contracts
            for record in records_by_contract.get(contract_key, [])
        ],
        "reason": (
            "manual apply_patch attempted to define a graph-selected "
            "provider that is absent from the current tree; import the "
            "complete graph-bound donor entity through "
            "materialize_entity_from_donor or adapt the authorized consumer "
            "to existing target providers"
        ),
    }


def _validate_static_patch_sanity(
    spec: CaseSpec,
    *,
    unit: dict[str, Any],
    diff_text: str,
    touched_files: list[str],
    expected_provider_obligations: list[dict[str, Any]] | None = None,
) -> tuple[bool, str | None, str | None, dict[str, Any] | None]:
    valid_structure, structure_error = _validate_unified_diff_structure(diff_text)
    if not valid_structure:
        return False, structure_error, "UnifiedDiffStructureChecker", None
    shim_finding = _manual_macro_shim_finding(unit, diff_text)
    if shim_finding is not None:
        return (
            False,
            str(shim_finding["reason"]),
            "CompatibilityShimChecker",
            shim_finding,
        )
    valid_provider_contract, provider_contract_error, provider_contract_report = validate_provider_contract_patch(
        spec,
        diff_text=diff_text,
        touched_files=touched_files,
        expected_provider_obligations=(
            expected_provider_obligations
        ),
    )
    if not valid_provider_contract:
        return False, provider_contract_error, "ProviderContractChecker", provider_contract_report
    provider_definition_finding = (
        _manual_missing_provider_definition_finding(
            unit,
            provider_contract_report,
        )
    )
    if provider_definition_finding is not None:
        rejected_report = {
            **provider_contract_report,
            "status": "rejected",
            "blocking_findings": [
                *list(
                    provider_contract_report.get("blocking_findings")
                    or []
                ),
                provider_definition_finding,
            ],
            "provider_materialization_authority": (
                provider_definition_finding
            ),
        }
        return (
            False,
            str(provider_definition_finding["reason"]),
            "ProviderMaterializationAuthorityChecker",
            rejected_report,
        )
    return True, None, None, provider_contract_report


def _write_patch_transaction_preview(
    spec: CaseSpec,
    *,
    units: list[dict[str, Any]],
    response_dir: Path,
    plan_path: Path | None,
) -> dict[str, Any]:
    ensure_dir(response_dir)
    candidates: list[dict[str, Any]] = []

    for index, unit in enumerate(units):
        unit_id = str(unit.get("id") or "")
        response_path = response_dir / f"{unit_id}.response.txt"
        base = {
            "index": index,
            "unit_id": unit_id,
            "response_path": str(response_path.resolve()),
            "unit": unit,
        }
        if not response_path.exists():
            candidates.append(
                {
                    **base,
                    "status": "missing_response",
                    "action": "missing_response",
                    "touched_files": [],
                }
            )
            continue
        try:
            interpreted = _interpret_response_text(
                response_path.read_text(encoding="utf-8"),
            )
        except (OSError, ResponseInterpretationError) as exc:
            candidates.append(
                {
                    **base,
                    "status": "invalid_response",
                    "action": "invalid_response",
                    "error": str(exc),
                    "touched_files": [],
                }
            )
            continue
        if interpreted["action"] != "apply_patch":
            candidates.append(
                {
                    **base,
                    "status": str(interpreted["action"]),
                    "action": str(interpreted["action"]),
                    "touched_files": [],
                }
            )
            continue
        diff_text = str(interpreted["diff_text"] or "")
        touched_files = touched_files_from_diff(diff_text)
        candidates.append(
            {
                **base,
                "status": "candidate",
                "action": "apply_patch",
                "touched_files": touched_files,
                "patch_signature": canonical_patch_signature(diff_text),
                "diff_text": diff_text,
                "mechanical_repair": interpreted.get("mechanical_repair"),
                "allowed_fileset": list(unit.get("allowed_fileset") or []),
                "escalation_candidates": list(unit.get("escalation_candidates") or []),
                "unit_kind": unit.get("unit_kind"),
            }
        )

    preview = build_patch_transaction_preview(
        case_id=spec.case_id,
        source_plan_path=str(plan_path.resolve()) if plan_path else None,
        response_dir=str(response_dir.resolve()),
        candidates=candidates,
    )
    sanity_report = build_sanity_report(
        candidates=candidates,
        transaction_preview=preview,
    )
    preview["sanity_report"] = sanity_report
    preview.setdefault("summary", {})["sanity_warning_count"] = int(
        (sanity_report.get("summary") or {}).get("warning_count") or 0
    )
    write_json(response_dir / "patch_transaction_preview.json", preview)
    return preview


def _sanity_errors_by_unit(transaction_preview: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for warning in (transaction_preview.get("sanity_report") or {}).get("warnings") or []:
        if not isinstance(warning, dict):
            continue
        if str(warning.get("severity") or "warning") != "error":
            continue
        unit_id = str(warning.get("unit_id") or "").strip()
        if not unit_id:
            continue
        by_unit.setdefault(unit_id, []).append({key: value for key, value in warning.items() if key != "unit"})
    return by_unit


def apply_plan_responses(
    spec: CaseSpec,
    plan: dict[str, Any],
    *,
    response_dir: Path,
    state_root: Path,
    allow_escalation: bool = False,
    plan_path: Path | None = None,
) -> list[dict[str, Any]]:
    units = repair_plan_units(plan)
    cycle_hint = cycle_hint_from_path(plan_path)
    results: list[dict[str, Any]] = []
    mechanical_repairs_by_unit: dict[str, dict[str, Any]] = {}
    applied_patch_signatures: set[str] = set()
    transaction_preview = _write_patch_transaction_preview(
        spec,
        units=units,
        response_dir=response_dir,
        plan_path=plan_path,
    )
    hard_sanity_errors_by_unit = _sanity_errors_by_unit(transaction_preview)
    plan_execution_phase = str(plan.get("execution_phase") or "").strip()
    for unit in units:
        unit_id = str(unit["id"])
        execution_phase = unit_execution_phase(unit)
        if (
            plan_execution_phase
            and execution_phase != plan_execution_phase
        ):
            raise ValueError(
                "repair plan and work unit execution phases differ: "
                f"{plan_execution_phase} != {execution_phase}"
            )
        semantic_controlled = semantic_controls_enabled(unit)
        patch_outcome_candidate = (
            dict(unit.get("patch_outcome_candidate") or {})
            if isinstance(unit.get("patch_outcome_candidate"), dict)
            else None
        )
        response_path = response_dir / f"{unit_id}.response.txt"
        if not response_path.exists():
            results.append(
                {
                    "unit_id": unit_id,
                    "status": "missing_response",
                    "response_path": str(response_path.resolve()),
                }
            )
            continue

        response_text = response_path.read_text(encoding="utf-8")
        try:
            interpreted = _interpret_response_text(
                response_text,
            )
        except ResponseInterpretationError as exc:
            immutable_patch_text = exc.immutable_patch_text
            patch_path: Path | None = None
            patch_signature: str | None = None
            if immutable_patch_text:
                patch_path = _write_generated_patch(
                    response_dir,
                    f"{unit_id}.invalid",
                    immutable_patch_text,
                )
                patch_signature = canonical_patch_signature(
                    immutable_patch_text
                )
            result = {
                "unit_id": unit_id,
                "status": "invalid_response",
                "response_path": str(response_path.resolve()),
                "error": str(exc),
                **(
                    {"patch_path": str(patch_path)}
                    if patch_path is not None
                    else {}
                ),
                **(
                    {"patch_signature": patch_signature}
                    if patch_signature
                    else {}
                ),
            }
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    result,
                    stage=exc.stage,
                    recoverability="outer_unit_replan",
                    finding_kind=exc.finding_kind,
                    patch_signature=patch_signature,
                    details={
                        "parser_error": str(exc),
                        "immutable_patch_body_available": bool(
                            immutable_patch_text
                        ),
                    },
                )
            )
            continue
        except Exception as exc:
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    {
                        "unit_id": unit_id,
                        "status": "invalid_response",
                        "response_path": str(response_path.resolve()),
                        "error": str(exc),
                    },
                    stage="response_envelope",
                    recoverability="terminal_infrastructure",
                    finding_kind="unexpected_response_parser_failure",
                    details={"exception_type": exc.__class__.__name__},
                )
            )
            continue
        if interpreted.get("mechanical_repair"):
            mechanical_repairs_by_unit[unit_id] = dict(interpreted["mechanical_repair"])

        if interpreted["action"] == "needs_escalation":
            results.append(
                {
                    "unit_id": unit_id,
                    "work_unit_id": unit.get("work_unit_id") or unit_id,
                    "status": "needs_escalation",
                    "response_kind": interpreted["kind"],
                    "payload": interpreted["payload"],
                    "missing_context": interpreted["payload"].get("missing_context") or {},
                    "allowed_fileset": list(unit.get("allowed_fileset") or []),
                    "requires_context_expansion": bool(unit.get("requires_context_expansion")),
                }
            )
            continue
        if interpreted["action"] == "no_patch":
            if str(unit.get("unit_kind") or "") != "initial_compatibility":
                results.append(
                    _attach_apply_failure(
                        spec,
                        unit,
                        {
                            "unit_id": unit_id,
                            "status": "invalid_response",
                            "response_path": str(response_path.resolve()),
                            "error": "no_patch is only accepted for pre-compile initial compatibility units",
                        },
                        stage="response_envelope",
                        recoverability="outer_unit_replan",
                        finding_kind="no_patch_not_allowed_for_repair",
                    )
                )
                continue
            results.append(
                {
                    "unit_id": unit_id,
                    "work_unit_id": unit.get("work_unit_id") or unit_id,
                    "status": "no_patch",
                    "response_kind": interpreted["kind"],
                    "payload": interpreted["payload"],
                    "allowed_fileset": list(unit.get("allowed_fileset") or []),
                }
            )
            continue

        hard_sanity_errors = hard_sanity_errors_by_unit.get(unit_id) or []
        if hard_sanity_errors:
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    {
                        "unit_id": unit_id,
                        "status": "invalid_response",
                        "response_path": str(response_path.resolve()),
                        "error": "patch sanity hard error",
                        "sanity_errors": hard_sanity_errors,
                    },
                    stage="target_policy",
                    recoverability="outer_unit_replan",
                    finding_kind="transaction_preview_conflict",
                    details={
                        "checker_ids": sorted(
                            {
                                str(item.get("checker") or "")
                                for item in hard_sanity_errors
                                if isinstance(item, dict)
                            }
                        )
                    },
                )
            )
            continue

        diff_text = str(interpreted["diff_text"] or "")
        touched_files = touched_files_from_diff(diff_text)
        patch_signature = canonical_patch_signature(diff_text)
        patch_path = _write_generated_patch(response_dir, unit_id, diff_text)
        if interpreted["payload"]:
            valid_targets, mismatches = _validate_structured_patch_targets(interpreted["payload"], touched_files)
            if not valid_targets:
                results.append(
                    _attach_apply_failure(
                        spec,
                        unit,
                        {
                            "unit_id": unit_id,
                            "status": "invalid_response",
                            "response_path": str(response_path.resolve()),
                            "patch_path": str(patch_path),
                            "error": f"patch_plan.affected_files not covered by diff: {mismatches}",
                            "validation_failure": {
                                "checker": "StructuredResponsePolicyChecker",
                                "kind": "affected_files",
                            },
                        },
                        stage="target_policy",
                        recoverability="outer_unit_replan",
                        finding_kind="declared_target_not_in_patch",
                        touched_files=touched_files,
                        patch_signature=patch_signature,
                        details={"mismatches": mismatches},
                    )
                )
                continue
        allowed_targets = _allowed_response_targets(unit, allow_escalation=allow_escalation)
        disallowed_targets = [item for item in touched_files if normalize_relpath(item) not in allowed_targets]
        if disallowed_targets:
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    {
                        "unit_id": unit_id,
                        "status": "disallowed_targets",
                        "response_path": str(response_path.resolve()),
                        "patch_path": str(patch_path),
                        "touched_files": touched_files,
                        "disallowed_targets": disallowed_targets,
                        "response_kind": interpreted["kind"],
                    },
                    stage="target_policy",
                    recoverability="outer_unit_replan",
                    finding_kind="patch_target_outside_graph_write_set",
                    touched_files=touched_files,
                    patch_signature=patch_signature,
                    details={
                        "disallowed_targets": disallowed_targets,
                        "allowed_target_count": len(allowed_targets),
                    },
                )
            )
            continue

        (
            structural_ranges_by_file,
            structural_rebase_evidence,
        ) = _unit_structural_rebase_ranges(
            spec,
            unit=unit,
            touched_files=touched_files,
        )
        transaction_authority_ranges_by_file = (
            _unit_transaction_authority_ranges(
                unit,
                touched_files=touched_files,
            )
        )
        structural_owner_constraints_by_file = {
            normalize_relpath(
                str(item.get("file") or "")
            ): [
                dict(owner)
                for owner in item.get(
                    "owner_constraints"
                )
                or []
                if isinstance(owner, dict)
                and owner.get("exact_graph_owner")
            ]
            for item in structural_rebase_evidence
            if isinstance(item, dict)
            and normalize_relpath(
                str(item.get("file") or "")
            )
        }
        has_exact_repair_authority = (
            any(
                structural_owner_constraints_by_file.get(
                    normalize_relpath(relative_path)
                )
                for relative_path in touched_files
            )
            or any(
                transaction_authority_ranges_by_file.get(
                    normalize_relpath(relative_path)
                )
                for relative_path in touched_files
            )
        ) and not bool(
            unit.get("controller_entity_materialization")
        )
        patch_rebase: dict[str, Any] | None = None
        if has_exact_repair_authority:
            owner_scoped_diff, owner_scope_info = (
                rebase_unified_diff_to_current_tree(
                    spec,
                    diff_text=diff_text,
                    touched_files=touched_files,
                    structural_ranges_by_file=(
                        structural_ranges_by_file
                    ),
                    structural_owner_constraints_by_file=(
                        structural_owner_constraints_by_file
                    ),
                    transaction_authority_ranges_by_file=(
                        transaction_authority_ranges_by_file
                    ),
                )
            )
            patch_rebase = {
                **owner_scope_info,
                "mode": "mandatory_owner_scoped_canonicalization",
                "structural_owner_constraints": (
                    structural_rebase_evidence
                ),
                "original_patch_path": str(patch_path),
            }
            if owner_scoped_diff is None:
                results.append(
                    _attach_apply_failure(
                        spec,
                        unit,
                        {
                            "unit_id": unit_id,
                            "status": "patch_apply_incomplete",
                            "response_path": str(
                                response_path.resolve()
                            ),
                            "patch_path": str(patch_path),
                            "touched_files": touched_files,
                            "patch_rebase": patch_rebase,
                            "error": owner_scope_info.get(
                                "reason"
                            ),
                        },
                        stage="structural_rebase",
                        recoverability="outer_unit_replan",
                        finding_kind=(
                            "exact_owner_patch_binding_failed"
                        ),
                        touched_files=touched_files,
                        patch_signature=patch_signature,
                        details={
                            "structural_rebase": patch_rebase
                        },
                    )
                )
                continue
            diff_text = owner_scoped_diff
            touched_files = touched_files_from_diff(diff_text)
            patch_signature = canonical_patch_signature(
                diff_text
            )
            patch_path = _write_generated_patch(
                response_dir,
                f"{unit_id}.owner-scoped",
                diff_text,
            )
            patch_rebase.update(
                {
                    "status": "owner_scoped",
                    "canonical_patch_path": str(patch_path),
                    "canonical_patch_signature": (
                        patch_signature
                    ),
                }
            )

        dry_run_ok, dry_run_stdout, dry_run_stderr = preflight_patch(
            spec,
            diff_text=diff_text,
            touched_files=touched_files,
        )
        provider_contract_report: dict[str, Any] | None = None
        if not dry_run_ok:
            already_applied, already_info = _diff_already_applied(
                spec,
                diff_text=diff_text,
                touched_files=touched_files,
            )
            if already_applied:
                duplicate = patch_signature in applied_patch_signatures
                applied_patch_signatures.add(patch_signature)
                results.append(
                    {
                        "unit_id": unit_id,
                        "status": (
                            "already_applied_duplicate"
                            if duplicate
                            else "already_applied"
                        ),
                        "response_path": str(response_path.resolve()),
                        "patch_path": str(patch_path),
                        "touched_files": touched_files,
                        "plan_kind": "repair",
                        "response_kind": interpreted["kind"],
                        "payload": interpreted["payload"],
                        "patch_rebase": patch_rebase,
                        "already_applied": already_info,
                    }
                )
                continue
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    {
                        "unit_id": unit_id,
                        "status": "apply_conflict",
                        "response_path": str(response_path.resolve()),
                        "patch_path": str(patch_path),
                        "touched_files": touched_files,
                        "response_kind": interpreted["kind"],
                        "error": (
                            "patch does not apply to the current tree after "
                            "any required owner-scoped canonicalization"
                        ),
                        "stdout": dry_run_stdout,
                        "stderr": dry_run_stderr,
                        "patch_rebase": patch_rebase,
                        "atomic_patch": True,
                    },
                    stage="structural_rebase",
                    recoverability="outer_unit_replan",
                    finding_kind="current_tree_patch_not_grounded",
                    touched_files=touched_files,
                    patch_signature=patch_signature,
                    deterministic_attempts=[
                        *(
                            [
                                {
                                    "kind": (
                                        "owner_scoped_"
                                        "canonicalization"
                                    ),
                                    "succeeded": True,
                                    "result": patch_rebase,
                                }
                            ]
                            if patch_rebase is not None
                            else []
                        ),
                        {
                            "kind": "current_tree_preflight",
                            "succeeded": False,
                        },
                    ],
                    details={
                        "atomic_patch": True,
                        "plan_kind": "repair",
                        "current_tree_preflight": {
                            "stdout": dry_run_stdout,
                            "stderr": dry_run_stderr,
                        },
                        "structural_rebase": patch_rebase or {},
                    },
                )
            )
            continue

        (
            valid_static_sanity,
            static_sanity_error,
            static_sanity_checker,
            provider_contract_report,
        ) = _validate_static_patch_sanity(
            spec,
            unit=unit,
            diff_text=diff_text,
            touched_files=touched_files,
            expected_provider_obligations=[
                dict(item)
                for item in unit.get(
                    "post_materialization_provider_obligations"
                )
                or []
                if isinstance(item, dict)
            ],
        )
        if patch_rebase is not None:
            patch_rebase.update(
                {
                    "rebased_static_sanity_ok": valid_static_sanity,
                    "rebased_static_sanity_error": static_sanity_error,
                    "rebased_static_sanity_checker": static_sanity_checker,
                    "rebased_provider_contract_report": (
                        provider_contract_report
                    ),
                }
            )
        if not valid_static_sanity:
            provider_status = str(
                (provider_contract_report or {}).get("status") or ""
            )
            if static_sanity_checker == "ProviderContractChecker":
                failure_stage = "provider_contract"
                if provider_status == "unavailable":
                    failure_recoverability = "terminal_infrastructure"
                    finding_kind = "checker_infrastructure_failure"
                elif (provider_contract_report or {}).get(
                    "blocking_findings"
                ):
                    failure_recoverability = "outer_unit_replan"
                    finding_kind = "contract_violation"
                else:
                    failure_recoverability = "outer_unit_replan"
                    finding_kind = "insufficient_graph_evidence"
            elif static_sanity_checker in {
                "CompatibilityShimChecker",
                "ProviderMaterializationAuthorityChecker",
            }:
                failure_stage = "provider_contract"
                failure_recoverability = "outer_unit_replan"
                finding_kind = "contract_violation"
            else:
                failure_stage = "unified_diff_structure"
                failure_recoverability = "terminal_infrastructure"
                finding_kind = "inconsistent_static_revalidation"
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    {
                        "unit_id": unit_id,
                        "status": "invalid_response",
                        "response_path": str(response_path.resolve()),
                        "patch_path": str(patch_path),
                        "touched_files": touched_files,
                        "error": static_sanity_error,
                        "validation_failure": {
                            "checker": static_sanity_checker,
                            "kind": "static_patch_sanity",
                        },
                        "patch_rebase": patch_rebase,
                        "provider_contract_report": provider_contract_report,
                    },
                    stage=failure_stage,
                    recoverability=failure_recoverability,
                    finding_kind=finding_kind,
                    touched_files=touched_files,
                    patch_signature=patch_signature,
                    deterministic_attempts=[
                        {
                            "kind": "current_tree_preflight",
                            "succeeded": True,
                        },
                        {
                            "kind": "provider_static_validation",
                            "succeeded": False,
                            "checker": static_sanity_checker,
                        },
                    ],
                    details={
                        "checker": static_sanity_checker,
                        "provider_status": provider_status or None,
                        "provider_reason": (
                            (provider_contract_report or {}).get("reason")
                        ),
                        "provider_blocking_findings": list(
                            (provider_contract_report or {}).get(
                                "blocking_findings"
                            )
                            or []
                        ),
                    },
                )
            )
            continue

        if patch_signature in applied_patch_signatures:
            already_applied, already_info = _diff_already_applied(
                spec,
                diff_text=diff_text,
                touched_files=touched_files,
            )
            if already_applied:
                results.append(
                    {
                        "unit_id": unit_id,
                        "status": "already_applied_duplicate",
                        "response_path": str(response_path.resolve()),
                        "patch_path": str(patch_path),
                        "touched_files": touched_files,
                        "plan_kind": "repair",
                        "response_kind": interpreted["kind"],
                        "payload": interpreted["payload"],
                        "patch_signature": patch_signature,
                        "already_applied": already_info,
                    }
                )
                continue

        transaction = start_transaction(
            state_root,
            case_id=spec.case_id,
            cycle=cycle_hint,
            work_unit_id=unit_id,
        )
        for relative_path in touched_files:
            backup_file(transaction, spec, relative_path)
        transaction_manifest = transaction.manifest()
        transaction_manifest.update(
            {
                "patch_signature": patch_signature,
                "patch_path": str(patch_path),
                "touched_files": touched_files,
                **(
                    {"patch_outcome_candidate": patch_outcome_candidate}
                    if patch_outcome_candidate is not None
                    else {}
                ),
            }
        )
        transaction.save_manifest(transaction_manifest)

        pre_apply_digest = _unit_source_preimage_digest(
            spec,
            unit,
            touched_files,
        )
        pre_apply_structural_reports = _structural_parse_reports_for_files(
            spec,
            touched_files,
        )
        returncode, stdout, stderr = run_unified_diff(
            spec.old_root_path,
            diff_text,
        )

        if returncode != 0:
            finalize_transaction(
                transaction,
                status="failed",
                notes=[stdout, stderr],
            )
            rollback_result = rollback_transaction(spec, transaction.root)
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    {
                        "unit_id": unit_id,
                        "status": "apply_failed",
                        "response_path": str(response_path.resolve()),
                        "patch_path": str(patch_path),
                        "touched_files": touched_files,
                        "stdout": stdout,
                        "stderr": stderr,
                        "plan_kind": "repair",
                        "response_kind": interpreted["kind"],
                        "transaction_dir": str(transaction.root.resolve()),
                        "rollback": rollback_result,
                    },
                    stage="transaction_apply",
                    recoverability="terminal_infrastructure",
                    finding_kind="apply_failed_after_successful_dry_run",
                    touched_files=touched_files,
                    patch_signature=patch_signature,
                    deterministic_attempts=[
                        {
                            "kind": "current_tree_preflight",
                            "succeeded": True,
                        },
                        {
                            "kind": "atomic_patch_apply",
                            "succeeded": False,
                        },
                        {
                            "kind": "transaction_rollback",
                            "succeeded": bool(
                                rollback_result.get("restored_count")
                                is not None
                            ),
                        },
                    ],
                    details={
                        "atomic_patch": True,
                        "rollback_status": rollback_result.get("status"),
                    },
                )
            )
            continue

        post_apply_digest = _unit_source_preimage_digest(
            spec,
            unit,
            touched_files,
        )
        if post_apply_digest == pre_apply_digest:
            finalize_transaction(
                transaction,
                status="no_effect",
                notes=[
                    (
                        "patch command succeeded but the complete touched-file "
                        "postimage is byte-identical to the preimage"
                    )
                ],
            )
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    {
                        "unit_id": unit_id,
                        "status": "invalid_response",
                        "response_path": str(response_path.resolve()),
                        "patch_path": str(patch_path),
                        "touched_files": touched_files,
                        "stdout": stdout,
                        "stderr": stderr,
                        "plan_kind": "repair",
                        "response_kind": interpreted["kind"],
                        "transaction_dir": str(transaction.root.resolve()),
                        "error": (
                            "patch command produced no byte-level change in "
                            "any touched file"
                        ),
                    },
                    stage="structural_postcondition",
                    recoverability="outer_unit_replan",
                    finding_kind="no_effect_patch",
                    touched_files=touched_files,
                    patch_signature=patch_signature,
                    deterministic_attempts=[
                        {
                            "kind": "current_tree_preflight",
                            "succeeded": True,
                        },
                        {
                            "kind": "atomic_patch_apply",
                            "succeeded": True,
                        },
                        {
                            "kind": "postimage_state_change",
                            "succeeded": False,
                        },
                    ],
                    details={
                        "pre_apply_digest": pre_apply_digest,
                        "post_apply_digest": post_apply_digest,
                        "atomic_patch": True,
                    },
                )
            )
            continue

        structural_postcondition = _structural_parse_postcondition(
            spec,
            touched_files=touched_files,
            before_reports=pre_apply_structural_reports,
        )
        if structural_postcondition.get("status") == "regressed":
            finalize_transaction(
                transaction,
                status="postcondition_failed",
                notes=[
                    (
                        "patch introduced a tree-sitter structural parse error "
                        "in a source file that was structurally clean before "
                        "this transaction"
                    )
                ],
            )
            rollback_result = rollback_transaction(spec, transaction.root)
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    {
                        "unit_id": unit_id,
                        "status": "invalid_response",
                        "response_path": str(response_path.resolve()),
                        "patch_path": str(patch_path),
                        "touched_files": touched_files,
                        "stdout": stdout,
                        "stderr": stderr,
                        "plan_kind": "repair",
                        "response_kind": interpreted["kind"],
                        "transaction_dir": str(transaction.root.resolve()),
                        "rollback": rollback_result,
                        "structural_postcondition": structural_postcondition,
                        "error": (
                            "patch introduced a structural parse error in a "
                            "previously clean source file"
                        ),
                    },
                    stage="structural_postcondition",
                    recoverability="outer_unit_replan",
                    finding_kind="introduced_structural_parse_error",
                    touched_files=touched_files,
                    patch_signature=patch_signature,
                    deterministic_attempts=[
                        {
                            "kind": "current_tree_preflight",
                            "succeeded": True,
                        },
                        {
                            "kind": "atomic_patch_apply",
                            "succeeded": True,
                        },
                        {
                            "kind": "transaction_structural_parse_check",
                            "succeeded": False,
                            "result": structural_postcondition,
                        },
                        {
                            "kind": "transaction_rollback",
                            "succeeded": bool(
                                rollback_result.get("restored_count")
                                is not None
                            ),
                        },
                    ],
                    details={
                        "atomic_patch": True,
                        "structural_postcondition": structural_postcondition,
                    },
                )
            )
            continue

        post_patch_integration: dict[str, Any] | None = None
        integration_requirements: list[dict[str, Any]] = []
        if semantic_controlled:
            integration_requirements = (
                semantic_integration_requirements_from_unit(
                    unit,
                    touched_files=touched_files,
                )
            )
            try:
                post_patch_integration = analyze_post_patch_integration(
                    spec,
                    transaction_dir=transaction.root,
                    touched_files=touched_files,
                    validation_requirements=(
                        integration_requirements
                    ),
                )
            except Exception as exc:
                post_patch_integration = {
                    "schema": "post_patch_integration_analysis_v1",
                    "analysis_revision": INTEGRATION_ANALYSIS_REVISION,
                    "status": "unavailable",
                    "revalidation_request": {
                        "schema": (
                            "post_patch_integration_revalidation_request_v1"
                        ),
                        "transaction_dir": str(
                            transaction.root.resolve()
                        ),
                        "touched_files": list(touched_files),
                    },
                    "obligations": [],
                    "coverage_blockers": [
                        {
                            "reason": (
                                "post_patch_integration_analysis_failed"
                            ),
                            "error": f"{exc.__class__.__name__}: {exc}",
                        }
                    ],
                    "summary": {
                        "seed_count": 0,
                        "obligation_count": 0,
                        "unresolved_count": 0,
                        "coverage_blocker_count": 1,
                    },
                    "atomic_transaction_required": bool(
                        integration_requirements
                    ),
                    "atomic_transaction_complete": False,
                }
        if (
            post_patch_integration is not None
            and bool(
                post_patch_integration.get(
                    "atomic_transaction_required"
                )
            )
            and not bool(
                post_patch_integration.get(
                    "atomic_transaction_complete"
                )
            )
        ):
            finalize_transaction(
                transaction,
                status="postcondition_failed",
                notes=[
                    (
                        "semantic provider transaction did not satisfy "
                        "its exact incoming-edge obligations"
                    )
                ],
            )
            rollback_result = rollback_transaction(
                spec,
                transaction.root,
            )
            required_integration = dict(
                post_patch_integration.get(
                    "required_integration"
                )
                or {}
            )
            unresolved_ids = {
                str(item)
                for item in required_integration.get(
                    "unresolved_obligation_ids"
                )
                or []
                if str(item)
            }
            unresolved_obligations = [
                dict(item)
                for item in post_patch_integration.get(
                    "obligations"
                )
                or []
                if isinstance(item, dict)
                and (
                    not unresolved_ids
                    or str(item.get("id") or "")
                    in unresolved_ids
                )
                and str(item.get("status") or "")
                != "satisfied"
            ]
            results.append(
                _attach_apply_failure(
                    spec,
                    unit,
                    {
                        "unit_id": unit_id,
                        "status": "invalid_response",
                        "response_path": str(
                            response_path.resolve()
                        ),
                        "patch_path": str(patch_path),
                        "touched_files": touched_files,
                        "stdout": stdout,
                        "stderr": stderr,
                        "plan_kind": "repair",
                        "response_kind": interpreted["kind"],
                        "transaction_dir": str(
                            transaction.root.resolve()
                        ),
                        "rollback": rollback_result,
                        "post_patch_integration": (
                            post_patch_integration
                        ),
                        "error": (
                            "the atomic semantic patch changed a "
                            "consumer-facing provider contract without "
                            "satisfying every exact incoming-edge "
                            "obligation"
                        ),
                    },
                    stage="semantic_transaction_postcondition",
                    recoverability="outer_unit_replan",
                    finding_kind=(
                        "incomplete_incoming_contract_transaction"
                    ),
                    touched_files=touched_files,
                    patch_signature=patch_signature,
                    deterministic_attempts=[
                        {
                            "kind": "current_tree_preflight",
                            "succeeded": True,
                        },
                        {
                            "kind": "atomic_patch_apply",
                            "succeeded": True,
                        },
                        {
                            "kind": (
                                "typed_incoming_edge_revalidation"
                            ),
                            "succeeded": False,
                        },
                        {
                            "kind": "transaction_rollback",
                            "succeeded": bool(
                                rollback_result.get(
                                    "restored_count"
                                )
                                is not None
                            ),
                        },
                    ],
                    details={
                        "integration_requirements": (
                            integration_requirements
                        ),
                        "integration_obligations": (
                            unresolved_obligations
                        ),
                        "integration_coverage_blockers": list(
                            required_integration.get(
                                "coverage_blockers"
                            )
                            or []
                        ),
                        "missing_consumer_reference_ids": list(
                            required_integration.get(
                                "missing_consumer_reference_ids"
                            )
                            or []
                        ),
                        "atomic_patch": True,
                        "rollback_status": (
                            rollback_result.get("status")
                        ),
                    },
                )
            )
            continue
        finalize_transaction(transaction, status="applied")
        applied_patch_signatures.add(patch_signature)
        results.append(
            {
                "unit_id": unit_id,
                "status": "applied",
                "patch_path": str(patch_path),
                "touched_files": touched_files,
                "plan_kind": "repair",
                "transaction_dir": str(transaction.root.resolve()),
                "response_kind": interpreted["kind"],
                "payload": interpreted["payload"],
                "patch_rebase": patch_rebase,
                "provider_contract_report": provider_contract_report,
                **(
                    {
                        "post_patch_integration": (
                            post_patch_integration
                        )
                    }
                    if post_patch_integration is not None
                    else {}
                ),
            }
        )
    units_by_id = {
        str(unit.get("id") or ""): unit
        for unit in units
        if isinstance(unit, dict) and unit.get("id")
    }
    enriched_results: list[dict[str, Any]] = []
    for item in results:
        unit_id = str(item.get("unit_id") or "")
        source_unit = units_by_id.get(unit_id) or {}
        candidate = source_unit.get("patch_outcome_candidate")
        enriched_results.append(
            {
                **item,
                **(
                    {"patch_outcome_candidate": dict(candidate)}
                    if isinstance(candidate, dict)
                    else {}
                ),
                **(
                    {
                        "mechanical_repair": mechanical_repairs_by_unit[
                            unit_id
                        ]
                    }
                    if unit_id in mechanical_repairs_by_unit
                    else {}
                ),
            }
        )
    return enriched_results


def apply_single_generated_diff(
    spec: CaseSpec,
    *,
    state_root: Path,
    artifact_dir: Path,
    plan_unit: dict[str, Any],
    diff_text: str,
    allow_escalation: bool = False,
) -> dict[str, Any]:
    """Apply one generated diff through the canonical response protocol."""

    unit = dict(plan_unit)
    unit_id = str(unit.get("id") or "").strip()
    if not unit_id:
        raise ValueError("single generated diff requires a plan unit id")
    execution_phase = unit_execution_phase(unit)
    unit["execution_phase"] = execution_phase
    output_dir = ensure_dir(artifact_dir)
    response_dir = ensure_dir(output_dir / "responses")
    plan = {
        "execution_phase": execution_phase,
        "work_unit_plans": [unit],
    }
    plan_path = output_dir / "apply_plan.json"
    write_repair_plan(
        plan_path,
        plan,
        state_root=state_root,
    )
    response_path = response_dir / f"{unit_id}.response.txt"
    response_path.write_text(
        diff_text if diff_text.endswith("\n") else diff_text + "\n",
        encoding="utf-8",
    )
    results = apply_plan_responses(
        spec,
        plan,
        response_dir=response_dir,
        state_root=state_root,
        allow_escalation=allow_escalation,
        plan_path=plan_path,
    )
    if len(results) != 1:
        raise RuntimeError(
            "single generated diff returned "
            f"{len(results)} apply results"
        )
    return {
        **dict(results[0]),
        "apply_plan_path": str(plan_path.resolve()),
    }
