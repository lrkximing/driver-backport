from __future__ import annotations

import re
from typing import Any


_CONDITIONAL_DIRECTIVE = re.compile(
    r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$"
)


def _conditional_directives(
    lines: list[str],
) -> list[tuple[int, str, str]]:
    """Return real conditional directives, excluding commented text."""

    directives: list[tuple[int, str, str]] = []
    in_block_comment = False
    for line_no, line in enumerate(lines, start=1):
        visible: list[str] = []
        offset = 0
        while offset < len(line):
            if in_block_comment:
                end = line.find("*/", offset)
                if end < 0:
                    offset = len(line)
                    break
                in_block_comment = False
                offset = end + 2
                continue
            block = line.find("/*", offset)
            single = line.find("//", offset)
            if single >= 0 and (block < 0 or single < block):
                visible.append(line[offset:single])
                offset = len(line)
                break
            if block < 0:
                visible.append(line[offset:])
                offset = len(line)
                break
            visible.append(line[offset:block])
            in_block_comment = True
            offset = block + 2
        match = _CONDITIONAL_DIRECTIVE.match("".join(visible))
        if match:
            directives.append(
                (
                    line_no,
                    match.group(1),
                    re.sub(r"\s+", " ", match.group(2).strip()),
                )
            )
    return directives


def preprocessor_conditional_integrity(
    lines: list[str],
) -> dict[str, Any]:
    """Validate the nesting grammar of C preprocessor conditionals."""

    directives = _conditional_directives(lines)
    stack: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for line_no, directive, expression in directives:
        if directive in {"if", "ifdef", "ifndef"}:
            stack.append(
                {
                    "line": line_no,
                    "directive": directive,
                    "expression": expression,
                    "else_line": None,
                }
            )
            continue
        if directive == "elif":
            if not stack:
                issues.append(
                    {"kind": "orphan_elif", "line": line_no}
                )
            elif stack[-1]["else_line"] is not None:
                issues.append(
                    {
                        "kind": "elif_after_else",
                        "line": line_no,
                        "else_line": stack[-1]["else_line"],
                    }
                )
            continue
        if directive == "else":
            if not stack:
                issues.append(
                    {"kind": "orphan_else", "line": line_no}
                )
            elif stack[-1]["else_line"] is not None:
                issues.append(
                    {
                        "kind": "duplicate_else",
                        "line": line_no,
                        "first_else_line": stack[-1]["else_line"],
                    }
                )
            else:
                stack[-1]["else_line"] = line_no
            continue
        if not stack:
            issues.append(
                {"kind": "orphan_endif", "line": line_no}
            )
        else:
            stack.pop()
    issues.extend(
        {
            "kind": "unterminated_conditional",
            "line": int(block["line"]),
            "directive": str(block["directive"]),
            "expression": str(block["expression"]),
        }
        for block in stack
    )
    return {
        "schema": "preprocessor_conditional_integrity_v1",
        "status": "valid" if not issues else "invalid",
        "directive_count": len(directives),
        "issue_count": len(issues),
        "issues": issues,
    }


def preprocessor_conditional_regressions(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only newly introduced conditional-grammar defects by file."""

    regressions: list[dict[str, Any]] = []
    for relative_path in sorted(set(before) | set(after)):
        prior = dict(before.get(relative_path) or {})
        current = dict(after.get(relative_path) or {})
        prior_counts: dict[str, int] = {}
        current_counts: dict[str, int] = {}
        for issue in prior.get("issues") or []:
            kind = str(issue.get("kind") or "unknown")
            prior_counts[kind] = prior_counts.get(kind, 0) + 1
        for issue in current.get("issues") or []:
            kind = str(issue.get("kind") or "unknown")
            current_counts[kind] = current_counts.get(kind, 0) + 1
        introduced = {
            kind: count - prior_counts.get(kind, 0)
            for kind, count in current_counts.items()
            if count > prior_counts.get(kind, 0)
        }
        if introduced:
            regressions.append(
                {
                    "file": relative_path,
                    "introduced_issue_counts": introduced,
                    "before": prior,
                    "after": current,
                }
            )
    return regressions


def conditional_blocks(lines: list[str]) -> list[dict[str, Any]]:
    """Parse nested C preprocessor conditionals without evaluating expressions."""

    blocks: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    for line_no, directive, expression in _conditional_directives(lines):
        if directive in {"if", "ifdef", "ifndef"}:
            block = {
                "start_line": line_no,
                "end_line": None,
                "signature": f"{directive} {expression}".strip(),
                "nesting_signatures": [
                    str(item["signature"]) for item in stack
                ],
                "branch_starts": [line_no],
            }
            stack.append(block)
            continue
        if directive in {"elif", "else"}:
            if stack:
                stack[-1]["branch_starts"].append(line_no)
            continue
        if directive == "endif" and stack:
            block = stack.pop()
            block["end_line"] = line_no
            blocks.append(block)
    return blocks


def conditional_variant_family(
    *,
    donor_lines: list[str],
    old_lines: list[str],
    target_entities: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Map duplicate contracts in mutually exclusive branches to one block."""

    target_lines = sorted(
        {
            int((entity.get("extent") or {}).get("start_line") or 0)
            for entity in target_entities
            if int((entity.get("extent") or {}).get("start_line") or 0) > 0
        }
    )
    if len(target_lines) < 2:
        return None

    donor_candidates: list[dict[str, Any]] = []
    for block in conditional_blocks(donor_lines):
        end_line = int(block.get("end_line") or 0)
        if not end_line or not all(
            int(block["start_line"]) < line_no < end_line
            for line_no in target_lines
        ):
            continue
        branch_starts = [
            int(value) for value in block.get("branch_starts") or []
        ]
        branch_indexes = {
            max(
                index
                for index, start in enumerate(branch_starts)
                if start < line_no
            )
            for line_no in target_lines
        }
        if len(branch_indexes) == len(target_lines):
            donor_candidates.append(block)
    if not donor_candidates:
        return None
    donor_block = min(
        donor_candidates,
        key=lambda item: int(item["end_line"]) - int(item["start_line"]),
    )

    old_candidates = [
        block
        for block in conditional_blocks(old_lines)
        if block.get("signature") == donor_block.get("signature")
        and block.get("nesting_signatures")
        == donor_block.get("nesting_signatures")
    ]
    if len(old_candidates) != 1:
        return None
    old_block = old_candidates[0]
    entities_by_line: dict[int, dict[str, Any]] = {}
    for entity in target_entities:
        if not isinstance(entity, dict):
            continue
        line_no = int((entity.get("extent") or {}).get("start_line") or 0)
        if line_no in target_lines:
            entities_by_line.setdefault(line_no, entity)
    return {
        "schema": "conditional_contract_variant_set_v1",
        "variant_count": len(target_lines),
        "variants": [
            {
                "line": line_no,
                "kind": str(entities_by_line.get(line_no, {}).get("kind") or ""),
                "contract_key": str(
                    entities_by_line.get(line_no, {}).get("contract_key")
                    or entities_by_line.get(line_no, {}).get("key")
                    or ""
                ),
                "role": (
                    "definition"
                    if bool(entities_by_line.get(line_no, {}).get("is_definition"))
                    else "declaration"
                ),
            }
            for line_no in target_lines
        ],
        "signature": donor_block.get("signature"),
        "nesting_signatures": list(
            donor_block.get("nesting_signatures") or []
        ),
        "donor_start_line": int(donor_block["start_line"]),
        "donor_end_line": int(donor_block["end_line"]),
        "old_start_line": int(old_block["start_line"]),
        "old_end_line": int(old_block["end_line"]),
    }


def conditional_variant_insertions(
    *,
    donor_lines: list[str],
    old_lines: list[str],
    target_entities: list[dict[str, Any]],
    donor_entities: list[dict[str, Any]],
    old_entities: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Map a donor-only conditional entity family to exact old branch anchors.

    This is deliberately structural: it uses matching preprocessor blocks,
    branch ordinals, entity identity, and exact source gaps. Existing variants
    are not copied again. It does not infer behavior from diagnostic text.
    """

    family = conditional_variant_family(
        donor_lines=donor_lines,
        old_lines=old_lines,
        target_entities=target_entities,
    )
    if family is None:
        return None

    donor_block = next(
        (
            block
            for block in conditional_blocks(donor_lines)
            if int(block.get("start_line") or 0)
            == int(family["donor_start_line"])
            and int(block.get("end_line") or 0)
            == int(family["donor_end_line"])
        ),
        None,
    )
    old_block = next(
        (
            block
            for block in conditional_blocks(old_lines)
            if int(block.get("start_line") or 0)
            == int(family["old_start_line"])
            and int(block.get("end_line") or 0)
            == int(family["old_end_line"])
        ),
        None,
    )
    if donor_block is None or old_block is None:
        return None

    def branch_ranges(block: dict[str, Any]) -> list[dict[str, int]]:
        branch_starts = [int(value) for value in block.get("branch_starts") or []]
        end_line = int(block.get("end_line") or 0)
        return [
            {
                "branch_index": index,
                "directive_line": start,
                "content_start_line": start + 1,
                "content_end_line": (
                    branch_starts[index + 1] - 1
                    if index + 1 < len(branch_starts)
                    else end_line - 1
                ),
                "boundary_line": (
                    branch_starts[index + 1]
                    if index + 1 < len(branch_starts)
                    else end_line
                ),
            }
            for index, start in enumerate(branch_starts)
        ]

    donor_branches = branch_ranges(donor_block)
    old_branches = branch_ranges(old_block)
    if not donor_branches or len(donor_branches) != len(old_branches):
        return None
    donor_directives = {
        line: (directive, expression)
        for line, directive, expression in _conditional_directives(donor_lines)
    }
    old_directives = {
        line: (directive, expression)
        for line, directive, expression in _conditional_directives(old_lines)
    }
    if [donor_directives.get(branch["directive_line"]) for branch in donor_branches] != [
        old_directives.get(branch["directive_line"]) for branch in old_branches
    ]:
        return None

    def identity(entity: dict[str, Any]) -> tuple[str, str, bool]:
        return (
            str(entity.get("contract_key") or entity.get("key") or ""),
            str(entity.get("kind") or entity.get("entity_kind") or "").lower(),
            bool(entity.get("is_definition")),
        )

    def entities_in_branch(
        entities: list[dict[str, Any]], branch: dict[str, int]
    ) -> list[dict[str, Any]]:
        return [
            entity
            for entity in entities
            if isinstance(entity, dict)
            and int((entity.get("extent") or {}).get("start_line") or 0)
            >= branch["content_start_line"]
            and int((entity.get("extent") or {}).get("end_line") or 0)
            <= branch["content_end_line"]
        ]

    def source_extent(lines: list[str], entity: dict[str, Any]) -> list[str]:
        extent = entity.get("extent") or {}
        start = int(extent.get("start_line") or 0)
        end = int(extent.get("end_line") or 0)
        return lines[start - 1 : end] if start > 0 and end >= start else []

    insertions: list[dict[str, Any]] = []
    ordered_targets = sorted(
        target_entities,
        key=lambda entity: (
            int((entity.get("extent") or {}).get("start_line") or 0),
            int((entity.get("extent") or {}).get("end_line") or 0),
        ),
    )
    for entity in ordered_targets:
        extent = dict(entity.get("extent") or {})
        start_line = int(extent.get("start_line") or 0)
        end_line = int(extent.get("end_line") or start_line)
        matching_branches = [
            branch
            for branch in donor_branches
            if int(branch["content_start_line"]) <= start_line
            and end_line <= int(branch["content_end_line"])
        ]
        if len(matching_branches) != 1:
            return None
        donor_branch = matching_branches[0]
        branch_index = int(donor_branch["branch_index"])
        old_branch = old_branches[branch_index]
        donor_context = tuple(str(item) for item in entity.get("conditional_context") or [])
        donor_branch_entities = [
            item
            for item in entities_in_branch(donor_entities, donor_branch)
            if tuple(str(value) for value in item.get("conditional_context") or [])
            == donor_context
        ]
        old_branch_entities = [
            item
            for item in entities_in_branch(old_entities, old_branch)
            if tuple(str(value) for value in item.get("conditional_context") or [])
            == donor_context
        ]
        if not donor_context or not donor_branch_entities:
            return None
        contract = identity(entity)
        old_same_contract = [
            item for item in old_branch_entities if identity(item)[0] == contract[0]
        ]
        if old_same_contract:
            if (
                len(old_same_contract) != 1
                or identity(old_same_contract[0]) != contract
                or source_extent(donor_lines, entity)
                != source_extent(old_lines, old_same_contract[0])
            ):
                return None
            continue

        predecessors = sorted(
            [
                item
                for item in donor_branch_entities
                if int((item.get("extent") or {}).get("end_line") or 0) < start_line
            ],
            key=lambda item: int((item.get("extent") or {}).get("end_line") or 0),
            reverse=True,
        )
        donor_predecessor_end = donor_branch["directive_line"]
        old_predecessor_end = old_branch["directive_line"]
        for predecessor in predecessors:
            matches = [
                item
                for item in old_branch_entities
                if identity(item) == identity(predecessor)
                and source_extent(donor_lines, predecessor)
                == source_extent(old_lines, item)
            ]
            if len(matches) != 1:
                continue
            donor_predecessor_end = int(
                (predecessor.get("extent") or {}).get("end_line") or 0
            )
            old_predecessor_end = int(
                (matches[0].get("extent") or {}).get("end_line") or 0
            )
            break

        donor_before = donor_lines[donor_predecessor_end : start_line - 1]
        old_prefix = old_lines[
            old_predecessor_end : old_predecessor_end + len(donor_before)
        ]
        insertion_index = old_predecessor_end + len(donor_before)
        leading_blank_count = 0
        placement = "exact_branch_local_source_prefix"
        if (
            old_prefix != donor_before
            and donor_before
            and all(not line.strip() for line in donor_before)
            and old_predecessor_end == old_branch["content_end_line"]
        ):
            insertion_index = old_predecessor_end
            leading_blank_count = len(donor_before)
        if old_prefix != donor_before and not leading_blank_count:
            donor_next_entities = sorted(
                [
                    item
                    for item in donor_branch_entities
                    if int((item.get("extent") or {}).get("start_line") or 0)
                    > end_line
                ],
                key=lambda item: int(
                    (item.get("extent") or {}).get("start_line") or 0
                ),
            )
            if not predecessors or not donor_next_entities:
                return None
            left, right = predecessors[0], donor_next_entities[0]
            left_end = int((left.get("extent") or {}).get("end_line") or 0)
            right_start = int((right.get("extent") or {}).get("start_line") or 0)
            old_left = [
                item for item in old_branch_entities
                if identity(item) == identity(left)
            ]
            old_right = [
                item for item in old_branch_entities
                if identity(item) == identity(right)
                and source_extent(old_lines, item)
                == source_extent(donor_lines, right)
            ]
            if len(old_left) != 1 or len(old_right) != 1:
                return None
            old_left_end = int(
                (old_left[0].get("extent") or {}).get("end_line") or 0
            )
            old_right_start = int(
                (old_right[0].get("extent") or {}).get("start_line") or 0
            )
            donor_left_gap = donor_lines[left_end : start_line - 1]
            donor_right_gap = donor_lines[end_line : right_start - 1]
            old_gap = old_lines[old_left_end : old_right_start - 1]
            if (
                left_end >= start_line
                or right_start <= end_line
                or old_left_end >= old_right_start
                or len(donor_right_gap) > 1
                or any(line.strip() for line in (
                    donor_left_gap + donor_right_gap + old_gap
                ))
                or any(
                    item is not old_left[0]
                    and old_left_end
                    < int((item.get("extent") or {}).get("start_line") or 0)
                    < old_right_start
                    for item in old_branch_entities
                )
            ):
                return None
            insertion_index = old_right_start - 1
            placement = "identity_predecessor_exact_successor_bracket"
        if (
            insertion_index > old_branch["content_end_line"]
            or not source_extent(donor_lines, entity)
        ):
            return None

        for successor in donor_branch_entities:
            if int((successor.get("extent") or {}).get("start_line") or 0) <= end_line:
                continue
            matches = [
                item
                for item in old_branch_entities
                if identity(item) == identity(successor)
                and source_extent(donor_lines, successor)
                == source_extent(old_lines, item)
            ]
            if len(matches) == 1 and int(
                (matches[0].get("extent") or {}).get("start_line") or 0
            ) <= insertion_index:
                return None

        donor_next = donor_lines[end_line : end_line + 1]
        old_next = old_lines[insertion_index : insertion_index + 1]
        trailing_blank = bool(
            donor_next and not donor_next[0].strip()
            and old_next and old_next[0].strip()
        )

        insertions.append(
            {
                "branch_index": branch_index,
                "placement": placement,
                "old_insertion_index": insertion_index,
                "donor_start_line": start_line,
                "donor_end_line": end_line,
                "donor_leading_blank_start_line": (
                    donor_predecessor_end + 1 if leading_blank_count else 0
                ),
                "donor_leading_blank_count": leading_blank_count,
                "donor_trailing_blank": trailing_blank,
                "entity_id": str(entity.get("id") or ""),
                "contract_key": str(
                    entity.get("contract_key") or entity.get("key") or ""
                ),
            }
        )

    if not insertions:
        return None
    return {
        "schema": "conditional_contract_insertion_plan_v1",
        "family": family,
        "insertions": insertions,
    }


def apply_conditional_variant_insertions(
    *,
    donor_lines: list[str],
    old_lines: list[str],
    insertion_plan: dict[str, Any],
) -> list[str]:
    """Apply a proven conditional insertion plan without changing other lines."""

    materialized = list(old_lines)
    insertions = sorted(
        [
            dict(item)
            for item in insertion_plan.get("insertions") or []
            if isinstance(item, dict)
        ],
        key=lambda item: int(item.get("old_insertion_index") or 0),
        reverse=True,
    )
    if not insertions:
        raise ValueError("conditional insertion plan has no insertions")
    for insertion in insertions:
        donor_start = int(insertion.get("donor_start_line") or 0)
        donor_end = int(insertion.get("donor_end_line") or donor_start)
        old_index = int(insertion.get("old_insertion_index") or 0)
        if (
            donor_start <= 0
            or donor_end < donor_start
            or donor_end > len(donor_lines)
            or old_index < 0
            or old_index > len(materialized)
        ):
            raise ValueError("conditional insertion extent is outside its source")
        entity_lines = list(donor_lines[donor_start - 1 : donor_end])
        if entity_lines and not entity_lines[-1].endswith(("\n", "\r")):
            entity_lines[-1] += "\n"
        leading_start = int(insertion.get("donor_leading_blank_start_line") or 0)
        leading_count = int(insertion.get("donor_leading_blank_count") or 0)
        if leading_count:
            entity_lines[:0] = donor_lines[
                leading_start - 1 : leading_start - 1 + leading_count
            ]
        if insertion.get("donor_trailing_blank"):
            entity_lines.append("\n")
        materialized[old_index:old_index] = entity_lines
    return materialized
