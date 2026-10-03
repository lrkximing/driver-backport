#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path


def normalize_path(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return Path(os.path.abspath(str(expanded)))


def apply_path_maps(value: str, path_maps: list[tuple[str, str]]) -> str:
    for old_prefix, new_prefix in path_maps:
        if value == old_prefix:
            return new_prefix
        if value.startswith(old_prefix.rstrip("/") + "/"):
            suffix = value[len(old_prefix.rstrip("/")) :]
            return new_prefix.rstrip("/") + suffix
    return value


def apply_path_maps_to_command(command: str, path_maps: list[tuple[str, str]]) -> list[str]:
    tokens = shlex.split(command)
    path_prefix_flags = ("-I", "-isystem", "-iquote", "-idirafter", "--sysroot=")
    prefix_map_flags = ("-fmacro-prefix-map=", "-ffile-prefix-map=", "-fdebug-prefix-map=")
    remapped: list[str] = []
    for token in tokens:
        updated = apply_path_maps(token, path_maps)
        if updated == token:
            updated = token
            for prefix in path_prefix_flags:
                if token.startswith(prefix) and len(token) > len(prefix):
                    suffix = token[len(prefix) :]
                    updated = prefix + apply_path_maps(suffix, path_maps)
                    break
            if updated == token:
                for prefix in prefix_map_flags:
                    if token.startswith(prefix) and len(token) > len(prefix):
                        value = token[len(prefix) :]
                        old_path, sep, remainder = value.partition("=")
                        remapped_old = apply_path_maps(old_path, path_maps)
                        updated = prefix + remapped_old + (sep + remainder if sep else "")
                        break
        remapped.append(updated)
    return remapped


@dataclass(frozen=True)
class CompileCommand:
    directory: Path
    file: Path
    arguments: list[str]
    raw: dict


class CompileCommands:
    def __init__(self, entries: list[CompileCommand]) -> None:
        self.entries = entries
        self.by_file = {normalize_path(entry.file): entry for entry in entries}

    @classmethod
    def from_file(
        cls,
        json_path: Path,
        *,
        path_maps: list[tuple[str, str]] | None = None,
    ) -> "CompileCommands":
        path_maps = path_maps or []
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        return cls.from_records(payload, path_maps=path_maps)

    @classmethod
    def from_records(
        cls,
        payload: list[dict],
        *,
        path_maps: list[tuple[str, str]] | None = None,
    ) -> "CompileCommands":
        path_maps = path_maps or []
        entries: list[CompileCommand] = []
        for item in payload:
            directory = normalize_path(apply_path_maps(item["directory"], path_maps))
            file_path = normalize_path(apply_path_maps(item["file"], path_maps))
            if "arguments" in item:
                arguments = [apply_path_maps(str(arg), path_maps) for arg in item["arguments"]]
            else:
                arguments = apply_path_maps_to_command(item["command"], path_maps)
            entries.append(
                CompileCommand(
                    directory=directory,
                    file=file_path,
                    arguments=arguments,
                    raw=item,
                )
            )
        return cls(entries)

    def find(self, file_path: Path) -> CompileCommand | None:
        file_path = normalize_path(file_path)
        entry = self.by_file.get(file_path)
        if entry is not None:
            return entry

        suffix_matches = [
            candidate
            for candidate in self.entries
            if str(candidate.file).endswith("/" + file_path.as_posix().lstrip("/"))
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return None

    def architecture_contexts(self) -> list[str]:
        """Return architectures named by the database's kernel include paths."""

        contexts = {
            match.group(1)
            for entry in self.entries
            for value in [str(entry.directory), *entry.arguments]
            for match in re.finditer(
                r"(?:^|/)arch/([^/]+)/include(?:/|$)",
                str(value).replace("\\", "/"),
            )
        }
        return sorted(contexts)
