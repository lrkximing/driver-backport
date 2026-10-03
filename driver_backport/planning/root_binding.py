#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from driver_backport.core.common import stable_id


def mapped_root_alternative_ids(root: dict[str, Any]) -> list[str]:
    """Return the exact graph node ids that define a mapped root set."""

    alternatives = [
        item
        for item in root.get("alternatives") or []
        if isinstance(item, dict)
    ]
    if (
        not root.get("ambiguous")
        or root.get("unmapped")
        or not alternatives
    ):
        return []

    node_ids = [
        str(item.get("node_id") or item.get("entity_id") or "")
        for item in alternatives
    ]
    if any(not node_id for node_id in node_ids):
        return []
    unique_node_ids = sorted(set(node_ids))
    if len(unique_node_ids) < 2:
        return []
    return unique_node_ids


def graph_root_binding_key(root: dict[str, Any]) -> str:
    """Return a stable binding key for a single root or mapped root set."""

    explicit_key = str(root.get("key") or "")
    if explicit_key:
        return explicit_key

    alternative_ids = mapped_root_alternative_ids(root)
    if not alternative_ids:
        return ""

    existing_scope = str(root.get("scope") or "")
    if existing_scope:
        return existing_scope
    return stable_id("root-set", *alternative_ids)


def bind_graph_root_identity(root: dict[str, Any]) -> dict[str, Any]:
    """Persist a canonical key whenever the graph identifies a root set."""

    bound = dict(root)
    binding_key = graph_root_binding_key(bound)
    if binding_key:
        bound["key"] = binding_key
        bound.setdefault("scope", binding_key)
    return bound
