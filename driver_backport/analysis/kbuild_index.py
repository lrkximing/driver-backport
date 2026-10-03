#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable

from driver_backport.core.common import (
    normalize_relpath,
    ordered_unique_relpaths as _ordered_unique,
    stable_id,
)


MAKE_ASSIGNMENT_RE = re.compile(
    r"^(?P<lhs>[A-Za-z0-9_.$()/{}/+-]+)\s*(?P<operator>\+=|:=|=)\s*(?P<rhs>.*)$"
)
CONFIG_OBJECT_LHS_RE = re.compile(r"^obj-\$\((CONFIG_[A-Za-z0-9_]+)\)$")
FIXED_OBJECT_LHS_RE = re.compile(r"^obj-(?:y|m)$")
MODULE_OBJECT_LHS_RE = re.compile(
    r"^(?P<module>[A-Za-z0-9_./+-]+)-(?P<selector>y|m|objs|\$\([^)]*\))$"
)
OBJECT_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./+-]+\.o$")
DIRECTORY_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./+-]+/$")


def _logical_blocks(text: str) -> list[tuple[int, int, list[str]]]:
    lines = text.splitlines()
    result: list[tuple[int, int, list[str]]] = []
    index = 0
    while index < len(lines):
        start = index
        block = [lines[index]]
        while block[-1].rstrip().endswith("\\") and index + 1 < len(lines):
            index += 1
            block.append(lines[index])
        result.append((start + 1, index + 1, block))
        index += 1
    return result


def _logical_text(lines: list[str]) -> str:
    parts: list[str] = []
    for line in lines:
        content = line.split("#", 1)[0].rstrip()
        if content.endswith("\\"):
            content = content[:-1]
        if content.strip():
            parts.append(content.strip())
    return " ".join(parts)


def _rhs_items(rhs: str) -> tuple[list[str], list[str]]:
    objects: list[str] = []
    unresolved: list[str] = []
    for token in rhs.split():
        token = token.strip().rstrip(";")
        if not token:
            continue
        if "$" in token or "%" in token:
            unresolved.append(token)
        elif OBJECT_TOKEN_RE.fullmatch(token) or DIRECTORY_TOKEN_RE.fullmatch(token):
            objects.append(normalize_relpath(token))
    return _ordered_unique(objects), list(dict.fromkeys(unresolved))


@dataclass(frozen=True)
class KbuildAssignment:
    id: str
    file: str
    start_line: int
    end_line: int
    lines: tuple[str, ...]
    lhs: str
    operator: str
    rhs: str
    object_items: tuple[str, ...]
    unresolved_items: tuple[str, ...]
    relation: str
    config: str | None = None
    module: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "lines": list(self.lines),
            "lhs": self.lhs,
            "operator": self.operator,
            "rhs": self.rhs,
            "object_items": list(self.object_items),
            "unresolved_items": list(self.unresolved_items),
            "relation": self.relation,
            "config": self.config,
            "module": self.module,
        }


def parse_kbuild_text(
    text: str,
    *,
    file: str = "Makefile",
) -> list[KbuildAssignment]:
    records: list[KbuildAssignment] = []
    relative_file = normalize_relpath(file)
    for start_line, end_line, lines in _logical_blocks(text):
        logical = _logical_text(lines)
        match = MAKE_ASSIGNMENT_RE.match(logical)
        if match is None:
            continue
        lhs = match.group("lhs")
        operator = match.group("operator")
        rhs = match.group("rhs")
        objects, unresolved = _rhs_items(rhs)
        relation = "other"
        config: str | None = None
        module: str | None = None
        config_match = CONFIG_OBJECT_LHS_RE.match(lhs)
        module_match = MODULE_OBJECT_LHS_RE.match(lhs)
        if config_match:
            relation = "config_membership"
            config = config_match.group(1)
        elif FIXED_OBJECT_LHS_RE.match(lhs):
            relation = "fixed_membership"
        elif module_match:
            relation = "module_membership"
            module = normalize_relpath(module_match.group("module"))
        elif "$" in lhs or "%" in lhs:
            relation = "unresolved_dynamic_lhs"
        records.append(
            KbuildAssignment(
                id=stable_id("kbuild-assignment", relative_file, str(start_line), logical),
                file=relative_file,
                start_line=start_line,
                end_line=end_line,
                lines=tuple(lines),
                lhs=lhs,
                operator=operator,
                rhs=rhs,
                object_items=tuple(objects),
                unresolved_items=tuple(unresolved),
                relation=relation,
                config=config,
                module=module,
            )
        )
    return records


class KbuildIndex:
    """Canonical object/module/config graph for one Kbuild directory."""

    def __init__(
        self,
        assignments: Iterable[KbuildAssignment],
        *,
        directory: str,
        root: Path | None = None,
    ) -> None:
        self.assignments = list(assignments)
        self.directory = normalize_relpath(directory)
        self.root = root.resolve() if root is not None else None
        self.module_assignments: dict[str, list[KbuildAssignment]] = defaultdict(list)
        self.object_to_modules: dict[str, set[str]] = defaultdict(set)
        self.top_assignments: list[KbuildAssignment] = []
        for record in self.assignments:
            if record.relation == "module_membership" and record.module:
                self.module_assignments[record.module].append(record)
                for object_name in record.object_items:
                    self.object_to_modules[object_name].add(record.module)
            elif record.relation in {"config_membership", "fixed_membership"}:
                self.top_assignments.append(record)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        directory: str,
        file: str | None = None,
    ) -> "KbuildIndex":
        build_file = file or normalize_relpath(f"{directory}/Makefile")
        return cls(
            parse_kbuild_text(text, file=build_file),
            directory=directory,
        )

    @classmethod
    def from_directory(cls, root: Path, directory: str) -> "KbuildIndex":
        root = root.resolve()
        assignments: list[KbuildAssignment] = []
        for name in ("Makefile", "Kbuild"):
            path = (root / normalize_relpath(directory) / name).resolve()
            if not path.is_file():
                continue
            assignments.extend(
                parse_kbuild_text(
                    path.read_text(encoding="utf-8", errors="replace"),
                    file=normalize_relpath(f"{directory}/{name}"),
                )
            )
        return cls(assignments, directory=directory, root=root)

    def source_object(self, source_file: str) -> str | None:
        source = PurePosixPath(normalize_relpath(source_file))
        directory = PurePosixPath(self.directory) if self.directory else PurePosixPath(".")
        try:
            relative = source.relative_to(directory)
        except ValueError:
            return None
        if relative.suffix.lower() not in {".c", ".cc", ".cpp", ".s"}:
            return None
        return normalize_relpath(relative.with_suffix(".o").as_posix())

    def source_path_for_object(self, object_name: str) -> str | None:
        object_path = PurePosixPath(normalize_relpath(object_name))
        for suffix in (".c", ".cc", ".cpp", ".S", ".s"):
            source = PurePosixPath(self.directory) / object_path.with_suffix(suffix)
            normalized = normalize_relpath(source.as_posix())
            if self.root is None or (self.root / normalized).is_file():
                return normalized
        return None

    def assignment_closure_for_sources(self, source_files: Iterable[str]) -> dict[str, Any]:
        seed_objects = {
            object_name
            for source in source_files
            for object_name in [self.source_object(source)]
            if object_name
        }
        needed_objects = set(seed_objects)
        selected_ids: set[str] = set()
        reached_modules: set[str] = set()
        changed = True
        while changed:
            changed = False
            for object_name in list(needed_objects):
                for module in self.object_to_modules.get(object_name, set()):
                    if module not in reached_modules:
                        reached_modules.add(module)
                        needed_objects.add(f"{module}.o")
                        changed = True
            for module in list(reached_modules):
                for record in self.module_assignments.get(module, []):
                    if record.id not in selected_ids:
                        selected_ids.add(record.id)
                        changed = True
            for record in self.top_assignments:
                reaches_needed = any(
                    item in needed_objects
                    or (
                        item.endswith("/")
                        and any(needed.startswith(item) for needed in needed_objects)
                    )
                    for item in record.object_items
                )
                if reaches_needed and record.id not in selected_ids:
                    selected_ids.add(record.id)
                    changed = True

        selected = [record for record in self.assignments if record.id in selected_ids]
        top = [record for record in selected if record in self.top_assignments]
        module_targets = sorted(
            {
                object_name
                for record in top
                for object_name in record.object_items
                if object_name in needed_objects and object_name.endswith(".o")
            }
        )
        unresolved = [
            record
            for record in self.assignments
            if record.relation == "unresolved_dynamic_lhs" or record.unresolved_items
        ]
        return {
            "schema": "kbuild_assignment_closure_v1",
            "seed_source_files": _ordered_unique(source_files),
            "seed_objects": sorted(seed_objects),
            "needed_objects": sorted(needed_objects),
            "reached_modules": sorted(reached_modules),
            "configs": sorted({record.config for record in top if record.config}),
            "module_targets": module_targets,
            "expected_modules": [
                normalize_relpath(
                    (PurePosixPath(self.directory) / PurePosixPath(item).with_suffix(".ko")).as_posix()
                )
                for item in module_targets
                if any(record.relation == "config_membership" and item in record.object_items for record in top)
            ],
            "assignment_ids": [record.id for record in selected],
            "assignments": [record.to_dict() for record in selected],
            "unresolved_assignments": [record.to_dict() for record in unresolved],
            "complete": bool(seed_objects) and not bool(unresolved),
        }

    def component_sources(self, object_name: str) -> list[str]:
        """Resolve one Kbuild target to its complete source component set."""

        queue: deque[str] = deque([normalize_relpath(object_name)])
        seen: set[str] = set()
        sources: list[str] = []
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            module = str(PurePosixPath(current).with_suffix(""))
            members = [
                item
                for record in self.module_assignments.get(module, [])
                for item in record.object_items
            ]
            if members:
                queue.extend(members)
                continue
            source = self.source_path_for_object(current)
            if source and source not in sources:
                sources.append(source)
        return sources

    def build_contract_graph(self) -> dict[str, Any]:
        config_to_modules: dict[str, set[str]] = defaultdict(set)
        module_to_sources: dict[str, set[str]] = defaultdict(set)
        source_to_configs: dict[str, set[str]] = defaultdict(set)
        records: list[dict[str, Any]] = []
        for assignment in self.top_assignments:
            config = assignment.config
            if not config:
                continue
            config_name = config.removeprefix("CONFIG_")
            for target in assignment.object_items:
                module = str(PurePosixPath(target).with_suffix(""))
                sources = self.component_sources(target)
                config_to_modules[config_name].add(module)
                module_to_sources[module].update(sources)
                for source in sources:
                    source_to_configs[source].add(config_name)
                    records.append(
                        {
                            "build_file": assignment.file,
                            "line": assignment.start_line,
                            "object": str(PurePosixPath(source).with_suffix("")),
                            "source": source,
                            "configs": [config_name],
                            "module": module,
                            "assignment_id": assignment.id,
                            "text": f"CONFIG_{config_name} -> {target} -> {PurePosixPath(source).name}",
                        }
                    )
        return {
            "schema": "kbuild_contract_graph_v1",
            "records": records,
            "config_to_modules": {key: sorted(value) for key, value in config_to_modules.items()},
            "module_to_sources": {key: sorted(value) for key, value in module_to_sources.items()},
            "source_to_configs": {key: sorted(value) for key, value in source_to_configs.items()},
            "unresolved_assignments": [
                record.to_dict()
                for record in self.assignments
                if record.relation == "unresolved_dynamic_lhs" or record.unresolved_items
            ],
        }

    def to_graph(self, *, tree: str) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, str]] = []

        def node(node_id: str, kind: str, **payload: Any) -> None:
            payload.pop("id", None)
            nodes.setdefault(node_id, {"id": node_id, "kind": kind, **payload})

        for record in self.assignments:
            assignment_id = f"kbuild:{tree}:{record.id}"
            node(assignment_id, "KbuildAssignment", tree=tree, **record.to_dict())
            if record.config:
                config_id = f"kconfig:{tree}:{record.config}"
                node(config_id, "KconfigSymbol", tree=tree, symbol=record.config)
                edges.append({"from": config_id, "to": assignment_id, "type": "configures"})
            for object_name in record.object_items:
                object_id = f"object:{tree}:{self.directory}:{object_name}"
                node(object_id, "Object", tree=tree, object=object_name, directory=self.directory)
                edges.append({"from": assignment_id, "to": object_id, "type": "builds"})
                module = record.module
                if (
                    not module
                    and object_name.endswith(".o")
                    and (
                        record.relation == "config_membership"
                        or record.lhs == "obj-m"
                    )
                ):
                    module = str(PurePosixPath(object_name).with_suffix(""))
                if module:
                    module_id = f"module:{tree}:{self.directory}:{module}"
                    node(module_id, "Module", tree=tree, module=module, directory=self.directory)
                    edges.append({"from": object_id, "to": module_id, "type": "links_into"})
                source = self.source_path_for_object(object_name)
                if source:
                    file_id = f"file:{tree}:{source}"
                    node(file_id, "File", tree=tree, path=source)
                    edges.append({"from": file_id, "to": object_id, "type": "compiles_to"})
        return {
            "schema": "typed_kbuild_graph_v1",
            "nodes": list(nodes.values()),
            "edges": edges,
            "summary": {
                "assignment_count": len(self.assignments),
                "node_count": len(nodes),
                "edge_count": len(edges),
                "unresolved_assignment_count": sum(
                    record.relation == "unresolved_dynamic_lhs" or bool(record.unresolved_items)
                    for record in self.assignments
                ),
            },
        }
