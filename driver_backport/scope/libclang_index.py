#!/usr/bin/env python3
from __future__ import annotations

import ctypes.util
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import sysconfig
from typing import Any

from driver_backport.analysis.kbuild_index import KbuildIndex
from driver_backport.scope.compile_commands import CompileCommand, CompileCommands
from driver_backport.scope.compile_context_materialization import (
    exact_target_kbuild_command,
)


DROP_NEXT_ARG_FLAGS = {
    "-o",
    "-MF",
    "-MT",
    "-MQ",
    "-MJ",
    "--serialize-diagnostics",
}

DROP_EXACT_FLAGS = {
    "-c",
    "-MD",
    "-MMD",
    "-MP",
    "-femit-struct-debug-baseonly",
}

DROP_PREFIX_FLAGS = (
    "-Wp,",
    "-fdiagnostics-color=",
    "-fasan-shadow-offset=",
    "-fstrict-flex-arrays=",
    "-ftrivial-auto-var-init=",
    "-mpreferred-stack-boundary=",
    "-mindirect-branch=",
    "-mfunction-return=",
    "-mstack-protector-guard-symbol=",
    "-falign-jumps=",
    "-falign-loops=",
)

DROP_UNSUPPORTED_FLAGS = {
    "-mindirect-branch-register",
    "-mindirect-branch-cs-prefix",
    "-fno-allow-store-data-races",
    "-fconserve-stack",
    "-mskip-rax-setup",
}

DIRECT_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
COMPILER_TARGET_RE = re.compile(
    r"^(?P<target>.+)-(?:clang\+\+|clang|g\+\+|gcc|c\+\+|cc)"
    r"(?:-\d+(?:\.\d+)*)?$"
)


def _is_compiler_token(value: str) -> bool:
    name = Path(str(value)).name
    return name in {"cc", "gcc", "g++", "clang", "clang++"} or bool(
        COMPILER_TARGET_RE.match(name)
    )


def compiler_arguments_from_kbuild_dry_run(
    output: str,
    *,
    source_file: str,
) -> list[str] | None:
    """Extract the exact compiler segment from a Kbuild ``make -n`` trace."""

    source_suffix = "/" + source_file
    candidates: list[list[str]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if source_file not in line or " -c " not in line:
            continue
        try:
            lexer = shlex.shlex(
                line,
                posix=True,
                punctuation_chars=";",
            )
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            continue
        segment_start = 0
        for index in range(len(tokens) + 1):
            if index < len(tokens) and tokens[index] != ";":
                continue
            segment = tokens[segment_start:index]
            segment_start = index + 1
            if "-c" not in segment or not any(
                token == source_file or token.endswith(source_suffix)
                for token in segment
            ):
                continue
            compiler_index = next(
                (
                    item_index
                    for item_index, token in enumerate(segment)
                    if _is_compiler_token(token)
                ),
                None,
            )
            if compiler_index is not None:
                candidates.append(segment[compiler_index:])
    return candidates[-1] if candidates else None


def infer_path_map_from_entry(raw_entry_file: str, entry_path: Path, kernel_root: Path) -> tuple[str, str] | None:
    try:
        relative_entry = entry_path.resolve().relative_to(kernel_root.resolve())
    except ValueError:
        return None

    raw_entry = str(raw_entry_file).rstrip("/")
    relative_suffix = "/" + relative_entry.as_posix()
    if not raw_entry.endswith(relative_suffix):
        return None

    old_kernel_root = raw_entry[: -len(relative_suffix)]
    if not old_kernel_root:
        return None

    new_kernel_root = str(kernel_root.resolve())
    if old_kernel_root == new_kernel_root:
        return None

    old_source_root = str(Path(old_kernel_root).parent)
    new_source_root = str(Path(new_kernel_root).parent)
    if Path(old_kernel_root).name == Path(new_kernel_root).name:
        if old_source_root == new_source_root:
            return None
        return old_source_root.rstrip("/"), new_source_root.rstrip("/")
    return old_kernel_root.rstrip("/"), new_kernel_root.rstrip("/")


def _python_site_packages_candidates() -> list[Path]:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sysconfig.get_paths().get("purelib", "")),
        Path(sys.prefix) / "lib" / version / "site-packages",
        Path(sys.base_prefix) / "lib" / version / "site-packages",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "lib" / version / "site-packages")
    candidates.extend(
        [
        Path(f"/opt/homebrew/opt/llvm/lib/{version}/site-packages"),
        Path("/opt/homebrew/opt/llvm/lib/python3.14/site-packages"),
        Path("/opt/homebrew/opt/llvm/lib/python3.13/site-packages"),
        Path("/opt/homebrew/opt/llvm/lib/python3.12/site-packages"),
        Path("/opt/homebrew/opt/llvm/lib/python3.11/site-packages"),
        Path("/opt/homebrew/opt/llvm/lib/python3.10/site-packages"),
        Path("/opt/homebrew/opt/llvm/lib/python3.9/site-packages"),
        ]
    )
    explicit = os.environ.get("LIBCLANG_PYTHON_SITE")
    if explicit:
        candidates.insert(0, Path(explicit))
    return [candidate for candidate in candidates if str(candidate)]


def _library_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit_file = os.environ.get("LIBCLANG_FILE")
    explicit_path = os.environ.get("LIBCLANG_PATH")
    conda_prefix = os.environ.get("CONDA_PREFIX")

    if explicit_file:
        candidates.append(Path(explicit_file))
    library_dirs: list[Path] = []
    if explicit_path:
        library_dirs.append(Path(explicit_path))
    for prefix in (conda_prefix, sys.prefix, sys.base_prefix):
        if prefix:
            library_dirs.append(Path(prefix) / "lib")
    library_dirs.extend([Path("/usr/local/lib"), Path("/usr/lib/x86_64-linux-gnu")])
    if sys.platform == "darwin":
        library_dirs.append(Path("/opt/homebrew/opt/llvm/lib"))
    library_dirs.extend(sorted(Path("/usr/lib").glob("llvm-*/lib")))

    names = ("libclang.so", "libclang.dylib", "libclang.dll")
    patterns = ("libclang.so.*", "libclang-*.so*", "libclang*.dylib")
    for directory in library_dirs:
        candidates.extend(directory / name for name in names)
        for pattern in patterns:
            candidates.extend(sorted(directory.glob(pattern)))
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        identity = str(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(candidate)
    return result


def ensure_clang_bindings() -> Any:
    for candidate in _python_site_packages_candidates():
        candidate_text = str(candidate.resolve())
        if candidate.exists() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
    try:
        from clang import cindex  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "python clang bindings are not available. Set LIBCLANG_PYTHON_SITE or install the clang Python package."
        ) from exc

    if getattr(cindex.Config, "loaded", False):
        return cindex
    if getattr(cindex.Config, "library_file", None) or getattr(cindex.Config, "library_path", None):
        return cindex

    for libclang_file in _library_file_candidates():
        if libclang_file.exists():
            cindex.Config.set_library_file(str(libclang_file))
            return cindex

    explicit_path = os.environ.get("LIBCLANG_PATH")
    if explicit_path and Path(explicit_path).is_dir():
        cindex.Config.set_library_path(explicit_path)
        return cindex

    loader_name = ctypes.util.find_library("clang")
    if loader_name:
        cindex.Config.set_library_file(loader_name)
        return cindex
    raise RuntimeError(
        "libclang shared library was not found. Set LIBCLANG_FILE or LIBCLANG_PATH; "
        f"runtime_prefix={sys.prefix}"
    )


def _split_command(entry: CompileCommand) -> list[str]:
    if len(entry.arguments) == 1 and " " in entry.arguments[0]:
        return shlex.split(entry.arguments[0])
    return list(entry.arguments)


def _compiler_target(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    match = COMPILER_TARGET_RE.fullmatch(Path(tokens[0]).name)
    return match.group("target") if match is not None else None


def _has_explicit_target(arguments: list[str]) -> bool:
    return any(
        token in {"-target", "--target"}
        or token.startswith("-target=")
        or token.startswith("--target=")
        for token in arguments
    )


def _clang_resource_include() -> Path | None:
    explicit = os.environ.get("LIBCLANG_RESOURCE_DIR")
    if explicit:
        explicit_path = Path(explicit).expanduser()
        include_path = (
            explicit_path
            if explicit_path.name == "include"
            else explicit_path / "include"
        )
        if (include_path / "stdarg.h").is_file():
            return include_path.resolve()

    candidates = [
        path
        for prefix in (sys.prefix, sys.base_prefix)
        for path in (Path(prefix) / "lib" / "clang").glob("*/include")
        if (path / "stdarg.h").is_file()
    ]
    if not candidates:
        return None

    def version_key(path: Path) -> tuple[int, ...]:
        return tuple(
            int(part) if part.isdigit() else 0
            for part in path.parent.name.split(".")
        )

    return max(candidates, key=version_key).resolve()


def sanitize_cindex_args(entry: CompileCommand) -> list[str]:
    tokens = _split_command(entry)
    if not tokens:
        raise ValueError(f"empty compile command for {entry.file}")

    sanitized: list[str] = []
    source_path = str(entry.file.resolve())
    compiler_target = _compiler_target(tokens)
    skip_next = False

    for token in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-W"):
            continue
        if token in DROP_EXACT_FLAGS:
            continue
        if token in DROP_NEXT_ARG_FLAGS:
            skip_next = True
            continue
        if token in DROP_UNSUPPORTED_FLAGS:
            continue
        if (
            token == "-mabi=lp64"
            and compiler_target is not None
            and compiler_target.startswith("aarch64")
        ):
            continue
        if any(token.startswith(prefix) for prefix in DROP_PREFIX_FLAGS):
            continue
        if token == source_path:
            continue
        sanitized.append(token)

    if compiler_target and not _has_explicit_target(sanitized):
        sanitized.append(f"--target={compiler_target}")
    if "-fsyntax-only" not in sanitized:
        sanitized.append("-fsyntax-only")
    if "-Qunused-arguments" not in sanitized:
        sanitized.append("-Qunused-arguments")
    if "-Wno-unknown-warning-option" not in sanitized:
        sanitized.append("-Wno-unknown-warning-option")
    if "-Wno-duplicate-decl-specifier" not in sanitized:
        sanitized.append("-Wno-duplicate-decl-specifier")
    if "-Wno-enum-enum-conversion" not in sanitized:
        sanitized.append("-Wno-enum-enum-conversion")
    if "-Wno-gnu-variable-sized-type-not-at-end" not in sanitized:
        sanitized.append("-Wno-gnu-variable-sized-type-not-at-end")
    resource_include = _clang_resource_include()
    if resource_include is not None and "-nostdinc" in sanitized:
        resource_path = str(resource_include)
        explicit_system_paths = {
            sanitized[index + 1]
            for index, token in enumerate(sanitized[:-1])
            if token == "-isystem"
        }
        if resource_path not in explicit_system_paths:
            sanitized.extend(("-isystem", resource_path))

    return sanitized


def resolved_path(value: Any) -> Path | None:
    if value is None:
        return None
    try:
        return Path(os.path.abspath(os.path.expanduser(str(value))))
    except OSError:
        return None


def location_dict(location: Any) -> dict | None:
    file_path = resolved_path(getattr(location, "file", None))
    if file_path is None:
        return None
    return {
        "file": str(file_path),
        "line": int(getattr(location, "line", 0) or 0),
        "column": int(getattr(location, "column", 0) or 0),
    }


def diagnostic_dict(diagnostic: Any) -> dict[str, Any]:
    """Serialize one libclang diagnostic without discarding its subject range."""

    source_ranges: list[dict[str, Any]] = []
    for source_range in getattr(diagnostic, "ranges", ()) or ():
        start = location_dict(getattr(source_range, "start", None))
        end = location_dict(getattr(source_range, "end", None))
        if start is None or end is None:
            continue
        source_ranges.append({"start": start, "end": end})
    return {
        "severity": int(getattr(diagnostic, "severity", 0) or 0),
        "spelling": str(getattr(diagnostic, "spelling", "") or ""),
        "location": location_dict(getattr(diagnostic, "location", None)),
        "ranges": source_ranges,
    }


def is_under(path: Path, root: Path) -> bool:
    path_str = str(path)
    root_str = str(root)
    return path_str == root_str or path_str.startswith(root_str.rstrip("/") + "/")


def relative_to_root(path: Path, root: Path) -> str:
    path_str = str(path)
    root_str = str(root).rstrip("/")
    if path_str == root_str:
        return "."
    prefix = root_str + "/"
    if not path_str.startswith(prefix):
        raise ValueError(f"{path_str} is not under {root_str}")
    return path_str[len(prefix) :]


@dataclass(frozen=True)
class ParseResult:
    translation_unit: Any
    diagnostics: list[dict]
    parse_seconds: float
    compile_context_origin: dict[str, Any] | None


class ClangCompileContext:
    def __init__(
        self,
        compile_commands_path: Path,
        kernel_root: Path,
        *,
        path_maps: list[tuple[str, str]] | None = None,
        make_bin: str | None = None,
        build_env: dict[str, str] | None = None,
        candidate_build_dir: Path | None = None,
    ) -> None:
        self.cindex = ensure_clang_bindings()
        self.compile_commands_path = compile_commands_path.resolve()
        self.kernel_root = kernel_root.resolve()
        self.path_maps = list(path_maps or [])
        self.make_bin = str(make_bin or "make")
        self.build_env = {
            str(key): str(value)
            for key, value in (build_env or {}).items()
        }
        self.candidate_build_dir = (
            candidate_build_dir.resolve()
            if candidate_build_dir is not None
            else None
        )
        self._db = CompileCommands.from_file(self.compile_commands_path, path_maps=self.path_maps)
        self._index = self.cindex.Index.create()
        self._materialized_entries: dict[Path, CompileCommand] = {}
        self._materialization_failures: dict[Path, str] = {}

    def _reload_db(self) -> None:
        self._db = CompileCommands.from_file(self.compile_commands_path, path_maps=self.path_maps)

    def _kernel_relative(self, file_path: Path) -> str | None:
        try:
            return file_path.resolve().relative_to(self.kernel_root).as_posix()
        except ValueError:
            return None

    def _raw_entry_paths(self, entry: CompileCommand) -> list[str]:
        paths = [str(entry.raw.get("file") or ""), str(entry.file)]
        return [path.rstrip("/") for path in paths if path]

    def _infer_path_map_from_relative_match(self, file_path: Path) -> tuple[str, str] | None:
        relative_path = self._kernel_relative(file_path)
        if relative_path is None:
            return None

        exact_suffix = "/" + relative_path
        parent_relative = str(Path(relative_path).parent)
        parent_suffix = "" if parent_relative == "." else "/" + parent_relative + "/"
        old_kernel_roots: set[str] = set()

        for entry in self._db.entries:
            for raw_path in self._raw_entry_paths(entry):
                if raw_path.endswith(exact_suffix):
                    old_kernel_roots.add(raw_path[: -len(exact_suffix)].rstrip("/"))
                elif file_path.suffix in {".h", ".hpp"} and parent_suffix and parent_suffix in raw_path:
                    if raw_path.endswith(".mod.c"):
                        continue
                    old_kernel_roots.add(raw_path.split(parent_suffix, 1)[0].rstrip("/"))

        old_kernel_roots = {root for root in old_kernel_roots if root}
        source_tree_roots = {root for root in old_kernel_roots if Path(root).name == self.kernel_root.name}
        if source_tree_roots:
            old_kernel_roots = source_tree_roots
        if len(old_kernel_roots) != 1:
            return None

        old_kernel_root = next(iter(old_kernel_roots))
        new_kernel_root = str(self.kernel_root)
        if old_kernel_root == new_kernel_root:
            return None

        old_source_root = str(Path(old_kernel_root).parent)
        new_source_root = str(self.kernel_root.parent)
        if Path(old_kernel_root).name == self.kernel_root.name:
            if old_source_root == new_source_root:
                return None
            return old_source_root.rstrip("/"), new_source_root.rstrip("/")
        return old_kernel_root.rstrip("/"), new_kernel_root.rstrip("/")

    def _add_path_map(self, inferred: tuple[str, str] | None) -> bool:
        if inferred is None or inferred in self.path_maps:
            return False
        self.path_maps.append(inferred)
        self._reload_db()
        return True

    def _infer_and_reload_for(self, file_path: Path, entry: CompileCommand | None = None) -> bool:
        inferred = None
        if entry is not None:
            inferred = infer_path_map_from_entry(entry.raw.get("file", ""), file_path, self.kernel_root)
        if inferred is None:
            inferred = self._infer_path_map_from_relative_match(file_path)
        return self._add_path_map(inferred)

    def _entry_for_kernel_relative(self, relative_path: str) -> CompileCommand | None:
        expected = (self.kernel_root / relative_path).resolve()
        matches = [
            entry
            for entry in [
                *self._db.entries,
                *self._materialized_entries.values(),
            ]
            if entry.file.resolve() == expected
        ]
        return matches[0] if len(matches) == 1 else None

    def _kbuild_source_and_build_contexts(self) -> list[tuple[Path, Path]]:
        contexts: set[tuple[Path, Path]] = set()
        for entry in self._db.entries:
            try:
                # Copy-on-write overlays symlink untouched sources outside the
                # active root; their mapped path still identifies the TU.
                mapped_file = Path(os.path.abspath(entry.file))
                relative = mapped_file.relative_to(
                    self.kernel_root
                ).as_posix()
            except ValueError:
                mapped_file = Path(entry.file.parent.resolve()) / entry.file.name
                try:
                    relative = mapped_file.relative_to(
                        self.kernel_root
                    ).as_posix()
                except ValueError:
                    continue
            if not relative or not entry.file.is_file():
                continue
            resolved_file = entry.file.resolve()
            if not is_under(resolved_file, self.kernel_root):
                mapped_source = False
                for source_prefix, target_prefix in self.path_maps:
                    try:
                        suffix = mapped_file.relative_to(
                            Path(os.path.abspath(target_prefix))
                        )
                    except ValueError:
                        continue
                    if resolved_file == (Path(source_prefix) / suffix).resolve():
                        mapped_source = True
                        break
                if not mapped_source:
                    continue
            raw_file = str(entry.raw.get("file") or "").rstrip("/")
            suffix = "/" + relative
            if raw_file.endswith(suffix):
                source_root = Path(raw_file[: -len(suffix)]).resolve()
            elif entry.file.resolve() == (self.kernel_root / relative).resolve():
                source_root = self.kernel_root
            else:
                continue
            raw_directory = str(entry.raw.get("directory") or "").strip()
            build_directory = Path(
                raw_directory or entry.directory
            ).expanduser().resolve()
            if source_root.is_dir() and build_directory.is_dir():
                contexts.add((source_root, build_directory))
        return sorted(contexts, key=lambda item: (str(item[0]), str(item[1])))

    def _materialize_kbuild_entry(self, file_path: Path) -> CompileCommand | None:
        desired = file_path.resolve()
        cached = self._materialized_entries.get(desired)
        if cached is not None:
            return cached
        relative = self._kernel_relative(desired)
        if (
            relative is None
            or desired.suffix.lower() not in {".c", ".cc", ".cpp", ".s"}
            or not desired.is_file()
        ):
            return None
        contexts = self._kbuild_source_and_build_contexts()
        if len(contexts) != 1:
            self._materialization_failures[desired] = (
                "kbuild_source_or_build_context_not_unique:"
                f"{len(contexts)}"
            )
            return None
        source_root, build_directory = contexts[0]
        if not (build_directory / ".config").is_file():
            self._materialization_failures[desired] = (
                "kbuild_configuration_not_materialized"
            )
            return None
        object_target = Path(relative).with_suffix(".o").as_posix()
        command = [
            self.make_bin,
            f"O={build_directory}",
            "V=1",
            "-n",
            object_target,
        ]
        env = os.environ.copy()
        env.update(self.build_env)
        env.pop("KCONFIG_CONFIG", None)
        env["LC_ALL"] = "C"
        env["LANG"] = "C"
        try:
            process = subprocess.run(
                command,
                cwd=source_root,
                env=env,
                text=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._materialization_failures[desired] = (
                f"kbuild_command_capture_failed:{exc.__class__.__name__}"
            )
            return None
        output = process.stdout + process.stderr
        arguments = compiler_arguments_from_kbuild_dry_run(
            output,
            source_file=relative,
        )
        expected_output = "/" + object_target
        output_arguments = [
            arguments[index + 1]
            for index, token in enumerate((arguments or [])[:-1])
            if token == "-o"
        ]
        if arguments is None or not any(
            value == object_target or value.endswith(expected_output)
            for value in output_arguments
        ):
            self._materialization_failures[desired] = (
                "exact_kbuild_compiler_invocation_not_observed:"
                f"returncode={process.returncode}"
            )
            return None
        rebased_arguments = [
            str(argument).replace(
                str(source_root),
                str(self.kernel_root),
            )
            for argument in arguments
        ]
        entry = CompileCommand(
            directory=build_directory,
            file=desired,
            arguments=rebased_arguments,
            raw={
                "directory": str(build_directory),
                "file": str(desired),
                "arguments": rebased_arguments,
                "driver_backport_context_origin": {
                    "schema": "on_demand_kbuild_compile_context_v1",
                    "kind": "exact_kbuild_dry_run_compiler_invocation",
                    "source_file": relative,
                    "object_target": object_target,
                    "source_root": str(source_root),
                    "active_root": str(self.kernel_root),
                    "build_directory": str(build_directory),
                    "make_returncode": int(process.returncode),
                },
            },
        )
        self._materialized_entries[desired] = entry
        self._materialization_failures.pop(desired, None)
        return entry

    def _materialize_candidate_build_entry(
        self,
        file_path: Path,
    ) -> CompileCommand | None:
        """Load an exact compiler invocation from the active candidate build."""

        desired = file_path.resolve()
        cached = self._materialized_entries.get(desired)
        if cached is not None:
            return cached
        relative = self._kernel_relative(desired)
        if (
            self.candidate_build_dir is None
            or relative is None
            or desired.suffix.lower() not in {".c", ".cc", ".cpp", ".s"}
            or not desired.is_file()
        ):
            return None
        raw = exact_target_kbuild_command(
            source_file=relative,
            build_dir=self.candidate_build_dir,
            old_root=self.kernel_root,
            target_build_dir=self.candidate_build_dir,
        )
        if raw is None:
            return None
        if isinstance(raw.get("arguments"), list):
            arguments = [str(argument) for argument in raw["arguments"]]
        else:
            try:
                arguments = shlex.split(str(raw.get("command") or ""))
            except ValueError:
                arguments = []
        if not arguments:
            self._materialization_failures[desired] = (
                "candidate_kbuild_command_arguments_unavailable"
            )
            return None
        entry = CompileCommand(
            directory=Path(str(raw["directory"])).resolve(),
            file=desired,
            arguments=arguments,
            raw=raw,
        )
        self._materialized_entries[desired] = entry
        self._materialization_failures.pop(desired, None)
        return entry

    def _kbuild_borrow_candidates(self, desired: Path) -> list[CompileCommand]:
        relative = self._kernel_relative(desired)
        if relative is None or desired.suffix.lower() not in {".c", ".cc", ".cpp", ".s"}:
            return []
        directory = desired.parent.resolve()
        while is_under(directory, self.kernel_root):
            relative_directory = relative_to_root(directory, self.kernel_root)
            index = KbuildIndex.from_directory(self.kernel_root, relative_directory)
            graph = index.build_contract_graph()
            modules = {
                module
                for module, sources in (graph.get("module_to_sources") or {}).items()
                if relative in set(sources or [])
            }
            related_sources = {
                source
                for module in modules
                for source in (graph.get("module_to_sources") or {}).get(module) or []
                if source != relative
            }
            if not related_sources:
                desired_configs = set((graph.get("source_to_configs") or {}).get(relative) or [])
                related_sources = {
                    source
                    for source, configs in (graph.get("source_to_configs") or {}).items()
                    if source != relative and desired_configs & set(configs or [])
                }
            candidates = [
                entry
                for source in sorted(related_sources)
                for entry in [self._entry_for_kernel_relative(source)]
                if entry is not None
            ]
            if candidates:
                return candidates
            if directory == self.kernel_root:
                break
            directory = directory.parent
        return []

    def _header_includer_candidates(self, desired: Path) -> list[CompileCommand]:
        if desired.suffix.lower() not in {".h", ".hpp"}:
            return []
        candidates: list[CompileCommand] = []
        entries = [
            *self._db.entries,
            *self._materialized_entries.values(),
        ]
        for entry in entries:
            source = entry.file.resolve()
            if source.suffix.lower() not in {".c", ".cc", ".cpp"} or not source.is_file():
                continue
            try:
                source_lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line in source_lines:
                match = DIRECT_INCLUDE_RE.match(line)
                if not match:
                    continue
                include_name = match.group(2).strip()
                possible_paths = [(source.parent / include_name).resolve()]
                if match.group(1) == "<":
                    possible_paths.extend(
                        [
                            (self.kernel_root / "include" / include_name).resolve(),
                            (self.kernel_root / include_name).resolve(),
                        ]
                    )
                if desired in possible_paths:
                    candidates.append(entry)
                    break
        unique: dict[str, CompileCommand] = {}
        for entry in candidates:
            unique[str(entry.file.resolve())] = entry
        return list(unique.values())

    @staticmethod
    def _borrow_arguments(entry: CompileCommand, desired: Path) -> list[str]:
        borrowed_file = entry.file.resolve()
        arguments: list[str] = []
        for argument in entry.arguments:
            raw = str(argument)
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = entry.directory / candidate
            try:
                is_source_argument = candidate.resolve() == borrowed_file
            except OSError:
                is_source_argument = False
            arguments.append(str(desired) if is_source_argument else raw)
        return arguments

    def _borrow_entry(self, file_path: Path) -> CompileCommand | None:
        desired = file_path.resolve()
        is_header = desired.suffix.lower() in {".h", ".hpp"}
        candidates = (
            self._header_includer_candidates(desired)
            if is_header
            else self._kbuild_borrow_candidates(desired)
        )
        candidate_count = len(candidates)
        borrow_relation = "unique_kbuild_peer"
        if is_header:
            borrow_relation = "unique_direct_includer"
            if len(candidates) > 1:
                same_stem = [
                    entry
                    for entry in candidates
                    if entry.file.resolve().stem == desired.stem
                ]
                if len(same_stem) == 1:
                    candidates = same_stem
                    borrow_relation = "unique_same_stem_direct_includer"
        if len(candidates) != 1:
            return None
        borrowed_from = candidates[0]
        if not borrowed_from.directory.is_dir():
            borrowed_from = (
                self._materialize_candidate_build_entry(borrowed_from.file)
                or borrowed_from
            )
        raw = dict(borrowed_from.raw)
        raw["borrowed_from"] = str(borrowed_from.file)
        raw["borrow_relation"] = borrow_relation
        raw["borrow_candidate_count"] = candidate_count
        raw["file"] = str(desired)
        return CompileCommand(
            directory=borrowed_from.directory,
            file=desired,
            arguments=self._borrow_arguments(borrowed_from, desired),
            raw=raw,
        )

    def resolve_entry(self, file_path: Path) -> CompileCommand:
        desired = file_path.resolve()
        materialized = self._materialized_entries.get(desired)
        if materialized is not None:
            return materialized
        entry = self._db.find(desired)
        if entry is None:
            if self._infer_and_reload_for(desired):
                entry = self._db.find(desired)
        if entry is None or not entry.directory.is_dir():
            materialized = self._materialize_candidate_build_entry(desired)
            if materialized is not None:
                return materialized
        if entry is None:
            materialized = self._materialize_kbuild_entry(desired)
            if materialized is not None:
                return materialized
            borrowed = self._borrow_entry(desired)
            if borrowed is not None:
                if borrowed.file.exists() and borrowed.directory.exists():
                    return borrowed
                raise FileNotFoundError(
                    "borrowed compile context is not materialized for "
                    f"{desired}: source={borrowed.file} directory={borrowed.directory}"
                )
            detail = self._materialization_failures.get(desired)
            suffix = f"; {detail}" if detail else ""
            raise FileNotFoundError(
                f"no compile_commands entry found for {desired}{suffix}"
            )
        if entry.file == desired and entry.file.exists() and entry.directory.exists():
            return entry

        if self._infer_and_reload_for(desired, entry):
            entry = self._db.find(desired)
            if entry is None:
                borrowed = self._borrow_entry(desired)
                if borrowed is not None:
                    if borrowed.file.exists() and borrowed.directory.exists():
                        return borrowed
                    raise FileNotFoundError(
                        "borrowed compile context is not materialized for "
                        f"{desired}: source={borrowed.file} directory={borrowed.directory}"
                    )
                raise FileNotFoundError(f"no compile_commands entry found for {desired}")
            if entry.file == desired and entry.file.exists() and entry.directory.exists():
                return entry
        if entry.file.exists() and entry.directory.exists():
            return entry
        raise FileNotFoundError(
            "compile context is not materialized for "
            f"{desired}: source={entry.file} directory={entry.directory}"
        )

    def parse(
        self,
        file_path: Path,
        *,
        detailed_processing: bool,
        skip_function_bodies: bool,
        exhaustive_diagnostics: bool = False,
    ) -> ParseResult:
        desired = file_path.resolve()
        entry = self.resolve_entry(desired)
        args = sanitize_cindex_args(entry)
        if exhaustive_diagnostics and "-ferror-limit=0" not in args:
            args.append("-ferror-limit=0")

        options = 0
        if detailed_processing:
            options |= self.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        if skip_function_bodies:
            options |= self.cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES

        working_directory = entry.directory
        previous_cwd = Path.cwd()
        start = os.times()[4]
        try:
            os.chdir(working_directory)
            translation_unit = self._index.parse(str(entry.file), args=args, options=options)
        finally:
            os.chdir(previous_cwd)
        end = os.times()[4]

        diagnostics = [
            diagnostic_dict(diagnostic)
            for diagnostic in translation_unit.diagnostics
        ]
        return ParseResult(
            translation_unit=translation_unit,
            diagnostics=diagnostics,
            parse_seconds=max(0.0, end - start),
            compile_context_origin=(
                dict(entry.raw.get("driver_backport_context_origin") or {})
                or None
            ),
        )

    def compile_files(self, *, root: Path | None = None) -> list[str]:
        """Return stable kernel-relative source paths represented by this database."""

        relative_root = (root or self.kernel_root).resolve()
        files: list[str] = []
        seen: set[str] = set()
        for entry in sorted(self._db.entries, key=lambda item: str(item.file)):
            path = entry.file.resolve()
            if not is_under(path, relative_root):
                continue
            relative = normalize_compile_file(path, relative_root)
            if relative and relative not in seen:
                seen.add(relative)
                files.append(relative)
        return files


def normalize_compile_file(path: Path, root: Path) -> str:
    return relative_to_root(path.resolve(), root.resolve()).replace("\\", "/").lstrip("/")
