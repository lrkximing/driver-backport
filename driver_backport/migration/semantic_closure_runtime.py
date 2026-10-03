from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from driver_backport.core.common import (
    ordered_unique_strings as _ordered_strings,
)
from driver_backport.llm.model_util import gpt_api_with_metadata
from driver_backport.migration.semantic_closure_model import (
    build_semantic_repair_prompt,
    build_semantic_review_prompt,
    parse_semantic_repair_response,
    parse_semantic_review_response,
    project_semantic_tool_evidence_for_prompt,
    semantic_repair_json_schema,
    semantic_review_prompt_evidence_ids,
    semantic_review_json_schema,
)
from driver_backport.migration.semantic_closure_tools import (
    SemanticClosureEvidenceTools,
)
SEMANTIC_CLOSURE_RUNTIME_SCHEMA = "semantic_closure_runtime_v1"
ValidationCall = Callable[[dict[str, Any]], dict[str, Any]]


def _executable_mutation_frontier(
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return exact write actions that can still mutate this work unit.

    A semantic patch that was accepted leaves the unit immediately, so every
    advertised write handle in a subsequent repair call is still executable.
    Rejected attempts do not consume that authority; the current catalog is
    rebuilt from the current tree and remains the source of truth.
    """

    frontier: list[dict[str, Any]] = []
    for raw in evidence.get("mutation_targets") or []:
        if not isinstance(raw, dict):
            continue
        edit_handle_ids = _ordered_strings(
            raw.get("edit_handle_ids") or []
        )
        materialization_handle_ids = _ordered_strings(
            raw.get("materialization_handle_ids") or []
        )
        if not edit_handle_ids and not materialization_handle_ids:
            continue
        frontier.append(
            {
                "gap_id": str(raw.get("gap_id") or ""),
                "open_dimensions": _ordered_strings(
                    raw.get("open_dimensions") or []
                ),
                "owner_keys": _ordered_strings(
                    raw.get("owner_keys") or []
                ),
                "edit_handle_ids": edit_handle_ids,
                "materialization_handle_ids": (
                    materialization_handle_ids
                ),
            }
        )
    return frontier


def _review_decision_evidence(
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Project exact gap-bound evidence for the semantic reviewer."""

    return project_semantic_tool_evidence_for_prompt(
        evidence,
        mode="review",
    )


class DefaultSemanticClosureRuntime:
    """Production adapter for the isolated semantic closure controller."""

    def __init__(
        self,
        spec: object,
        *,
        state_root: Path,
        model: str | None = None,
        max_completion_tokens: int = 8192,
        validation_call: ValidationCall,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.state_root = state_root.resolve()
        self.model = model
        self.max_completion_tokens = max(
            0,
            int(max_completion_tokens),
        )
        self._validation_call = validation_call
        self.progress = progress
        self.tools = SemanticClosureEvidenceTools(
            spec,
            state_root=self.state_root,
        )

    def _notify(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _invoke_model(
        self,
        prompt: str,
        *,
        purpose: str,
        response_schema: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        prompt_size_bytes = len(prompt.encode("utf-8"))
        self._notify(
            f"semantic model call: {purpose} "
            f"prompt_bytes={prompt_size_bytes}"
        )
        response = gpt_api_with_metadata(
            prompt,
            model=self.model,
            max_completion_tokens=self.max_completion_tokens,
            max_retries=None,
            response_schema=response_schema,
            response_schema_name=purpose,
        )
        text = str(response.get("text") or "")
        metadata = {
            key: value
            for key, value in response.items()
            if key != "text"
        }
        metadata = {
            **metadata,
            "purpose": purpose,
            "prompt_size_bytes": prompt_size_bytes,
            "response_size_bytes": len(text.encode("utf-8")),
        }
        return text, metadata

    def review(self, request: dict[str, Any]) -> dict[str, Any]:
        context = (
            request.get("context")
            if isinstance(request.get("context"), dict)
            else {}
        )
        evidence, _catalog = self.tools.build_catalog(
            context,
            include_write_actions=False,
        )
        review_request = {
            **request,
            "decision_evidence": _review_decision_evidence(evidence),
        }
        prompt = build_semantic_review_prompt(review_request)
        text, metadata = self._invoke_model(
            prompt,
            purpose="semantic_review",
            response_schema=semantic_review_json_schema(),
        )
        prompt_evidence_ids = semantic_review_prompt_evidence_ids(
            review_request
        )
        prompt_source_records = [
            {
                "id": str(record.get("id") or ""),
                "tree": str(record.get("tree") or ""),
                "gap_ids": _ordered_strings(record.get("gap_ids") or []),
            }
            for record in (
                review_request.get("decision_evidence") or {}
            ).get("source_records") or []
            if isinstance(record, dict) and str(record.get("id") or "")
        ]
        try:
            parsed = parse_semantic_review_response(
                text,
                expected_gap_ids=request.get("expected_gap_ids") or [],
            )
        except ValueError as exc:
            return {
                "schema": SEMANTIC_CLOSURE_RUNTIME_SCHEMA,
                "status": "review_failed",
                "failure_kind": "model_protocol_invalid",
                "retryable": True,
                "error": f"{exc.__class__.__name__}: {exc}",
                "decisions": [],
                "prompt_evidence_ids": prompt_evidence_ids,
                "prompt_source_records": prompt_source_records,
                "model_call": metadata,
                "response_text": text,
            }
        return {
            **parsed,
            "status": "decided",
            "prompt_evidence_ids": prompt_evidence_ids,
            "prompt_source_records": prompt_source_records,
            "model_call": metadata,
            "response_text": text,
        }

    def repair(self, request: dict[str, Any]) -> dict[str, Any]:
        if str(request.get("kind") or "") != "semantic_repair":
            raise ValueError(
                "semantic runtime cannot execute compiler recovery"
            )
        context = (
            request.get("context")
            if isinstance(request.get("context"), dict)
            else {}
        )
        evidence, catalog = self.tools.build_catalog(context)
        prompt = build_semantic_repair_prompt(
            request,
            evidence=evidence,
        )
        text, metadata = self._invoke_model(
            prompt,
            purpose="semantic_repair",
            response_schema=semantic_repair_json_schema(),
        )
        work_unit = (
            context.get("work_unit")
            if isinstance(context.get("work_unit"), dict)
            else {}
        )
        expected_gap_ids = work_unit.get("gap_ids") or []
        try:
            action = parse_semantic_repair_response(
                text,
                kind=str(request.get("kind") or ""),
                expected_gap_ids=expected_gap_ids,
            )
        except ValueError as exc:
            return {
                "schema": SEMANTIC_CLOSURE_RUNTIME_SCHEMA,
                "action": "protocol_invalid",
                "status": "protocol_invalid",
                "reason": f"{exc.__class__.__name__}: {exc}",
                "model_call": metadata,
                "response_text": text,
            }
        common = {
            "schema": SEMANTIC_CLOSURE_RUNTIME_SCHEMA,
            "action": action["action"],
            "rationale": action.get("rationale"),
            "model_call": metadata,
            "response_text": text,
        }

        if action["action"] == "replace_source":
            try:
                structured_patch = self.tools.build_source_edit_patch(
                    source_edits=action.get("source_edits") or [],
                    catalog=catalog,
                )
            except (OSError, ValueError) as exc:
                return {
                    **common,
                    "status": "tool_request_rejected",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "source_edits": list(
                        action.get("source_edits") or []
                    ),
                }
            claimed_gap_ids = set(
                _ordered_strings(action.get("gap_ids") or [])
            )
            covered_gap_ids = set(
                _ordered_strings(
                    structured_patch.get("direct_gap_ids") or []
                )
            )
            if not claimed_gap_ids <= covered_gap_ids:
                return {
                    **common,
                    "status": "tool_request_rejected",
                    "reason": (
                        "structured source edits do not cover every claimed "
                        "semantic gap"
                    ),
                    "claimed_gap_ids": sorted(claimed_gap_ids),
                    "covered_gap_ids": sorted(covered_gap_ids),
                    "source_edits": list(
                        action.get("source_edits") or []
                    ),
                }
            if structured_patch.get("status") == "already_satisfied":
                return {
                    **common,
                    "status": "already_satisfied",
                    "gap_ids": list(action.get("gap_ids") or []),
                    "structured_source_patch": structured_patch,
                }
            return {
                **common,
                "status": "patch",
                "gap_ids": list(action.get("gap_ids") or []),
                "diff_text": str(
                    structured_patch.get("diff_text") or ""
                ),
                "structured_source_patch": structured_patch,
            }
        if action["action"] == "inspect":
            try:
                inspection = self.tools.inspect(
                    handle_ids=(
                        action.get("inspect_handle_ids") or []
                    ),
                    catalog=catalog,
                )
            except (OSError, ValueError) as exc:
                return {
                    **common,
                    "status": "tool_request_rejected",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "inspect_handle_ids": list(
                        action.get("inspect_handle_ids") or []
                    ),
                }
            return {
                **common,
                "status": "evidence_added",
                "inspect_handle_ids": list(
                    action.get("inspect_handle_ids") or []
                ),
                "evidence": inspection,
            }
        if action["action"] == "expand_dependencies":
            try:
                expansion = self.tools.expand_dependencies(
                    owner_keys=action.get("owner_keys") or [],
                    direction=str(action.get("direction") or ""),
                    dependency_edge_ids=(
                        action.get("dependency_edge_ids") or []
                    ),
                    catalog=catalog,
                )
            except (OSError, ValueError) as exc:
                return {
                    **common,
                    "status": "tool_request_rejected",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "owner_keys": list(
                        action.get("owner_keys") or []
                    ),
                    "direction": str(action.get("direction") or ""),
                    "dependency_edge_ids": list(
                        action.get("dependency_edge_ids") or []
                    ),
                }
            return {
                **common,
                "status": "evidence_added",
                "owner_keys": list(action.get("owner_keys") or []),
                "direction": str(action.get("direction") or ""),
                "dependency_edge_ids": list(
                    expansion.get("dependency_edge_ids") or []
                ),
                "evidence": expansion,
                "owner_bindings": list(
                    expansion.get("owner_bindings") or []
                ),
                "expanded_gap_ids": list(
                    expansion.get("expanded_gap_ids") or []
                ),
            }
        if action["action"] == "materialize_entity_from_donor":
            try:
                result = self.tools.materialize_entity(
                    materialization_handle_id=str(
                        action.get("materialization_handle_id") or ""
                    ),
                    gap_ids=action.get("gap_ids") or [],
                    rationale=str(action.get("rationale") or ""),
                    catalog=catalog,
                )
            except (OSError, ValueError) as exc:
                return {
                    **common,
                    "status": "tool_request_rejected",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "materialization_handle_id": str(
                        action.get("materialization_handle_id") or ""
                    ),
                    "gap_ids": list(action.get("gap_ids") or []),
                }
            return {
                **common,
                "status": "patch",
                "materialization_handle_id": str(
                    action.get("materialization_handle_id") or ""
                ),
                "gap_ids": list(result.get("gap_ids") or []),
                "diff_text": str(result.get("diff_text") or ""),
                "insertion_authorizations": list(
                    result.get("insertion_authorizations") or []
                ),
            }
        mutation_frontier = _executable_mutation_frontier(evidence)
        if mutation_frontier:
            return {
                **common,
                "status": "tool_request_rejected",
                "reason": (
                    "blocked is not a valid terminal action while the current "
                    "exact catalog still advertises executable mutation "
                    "targets; use one of those targets or request exact "
                    "evidence needed to edit it"
                ),
                "remaining_mutation_frontier": mutation_frontier,
                "executable_mutation_frontier_empty": False,
            }
        return {
            **common,
            "status": "blocked",
            "reason": action.get("rationale"),
            "remaining_mutation_frontier": [],
            "executable_mutation_frontier_empty": True,
        }

    def validate(self, request: dict[str, Any]) -> dict[str, Any]:
        report = self._validation_call(dict(request))
        if not isinstance(report, dict):
            raise ValueError(
                "semantic validation callback returned no report"
            )
        return {
            **report,
            "compile_ok": bool(report.get("overall_ok")),
            "validation_authority": "canonical_validation_cycle",
        }
