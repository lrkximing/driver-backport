#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any

from driver_backport.core.common import load_json, normalize_relpath, resolve_under, slugify


def _roots_overlap(left: str, right: str) -> bool:
    left = left.rstrip("/")
    right = right.rstrip("/")
    if left == right:
        return True
    return left.startswith(right + "/") or right.startswith(left + "/")


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: str
    cwd: str | None = None
    timeout_sec: int = 1800
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_name: str) -> "CommandSpec":
        name = str(raw.get("name") or default_name)
        command = str(raw.get("command") or "").strip()
        if not command:
            raise ValueError(f"phase {name!r} is missing command")
        cwd = raw.get("cwd")
        return cls(
            name=name,
            command=command,
            cwd=normalize_relpath(cwd) if cwd else None,
            timeout_sec=int(raw.get("timeout_sec") or 1800),
            env={str(key): str(value) for key, value in dict(raw.get("env") or {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "cwd": self.cwd,
            "timeout_sec": self.timeout_sec,
            "env": dict(self.env),
        }


@dataclass(frozen=True)
class RetrievalTuningSpec:
    max_rounds: int = 4
    max_candidate_files: int = 256

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "RetrievalTuningSpec":
        raw = dict(raw or {})
        return cls(
            max_rounds=max(1, int(raw.get("max_rounds") or 4)),
            max_candidate_files=max(32, int(raw.get("max_candidate_files") or 256)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "max_candidate_files": self.max_candidate_files,
        }


def _parse_phase_list(raw: dict[str, Any], key: str, default_name: str) -> list[CommandSpec]:
    payload = raw.get(key, [])
    if isinstance(payload, dict):
        if "phases" in payload:
            payload = payload["phases"]
        else:
            payload = [payload]

    if not payload:
        return []
    if not isinstance(payload, list):
        raise ValueError(f"{key} must be a list of phase objects")

    phases: list[CommandSpec] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{key}[{index - 1}] must be an object")
        phases.append(CommandSpec.from_dict(item, default_name=f"{default_name}-{index}"))
    return phases


def parse_path_map_item(item: Any) -> tuple[str, str]:
    if isinstance(item, str):
        left, sep, right = item.partition("=")
        if not sep:
            raise ValueError(f"path map must use OLD=NEW format: {item!r}")
        return str(Path(left).expanduser()).rstrip("/"), str(Path(right).expanduser()).rstrip("/")
    if isinstance(item, dict):
        left = item.get("from", item.get("old", item.get("source")))
        right = item.get("to", item.get("new", item.get("target")))
        if not left or not right:
            raise ValueError(f"path map object must contain from/to: {item!r}")
        return str(Path(str(left)).expanduser()).rstrip("/"), str(Path(str(right)).expanduser()).rstrip("/")
    raise ValueError(f"path map must be a string or object: {item!r}")


def parse_path_maps(raw: Any) -> list[tuple[str, str]]:
    if not raw:
        return []
    if isinstance(raw, (str, dict)):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("path maps must be a list")
    return [parse_path_map_item(item) for item in raw]


_ARCH_CROSS_COMPILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "arm64": ("aarch64-linux-gnu-",),
    "arm": ("arm-linux-gnueabihf-", "arm-linux-gnueabi-"),
    "riscv": ("riscv64-linux-gnu-",),
    "mips": ("mips-linux-gnu-", "mipsel-linux-gnu-"),
    "powerpc": ("powerpc64le-linux-gnu-", "powerpc-linux-gnu-"),
}


def _auto_fill_cross_compile(build_env: dict[str, str]) -> dict[str, str]:
    env = dict(build_env)
    arch = str(env.get("ARCH") or "").strip().lower()
    if not arch or env.get("CROSS_COMPILE"):
        return env
    for prefix in _ARCH_CROSS_COMPILE_CANDIDATES.get(arch, ()):
        if shutil.which(prefix + "gcc"):
            env["CROSS_COMPILE"] = prefix
            break
    return env


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    old_root: str
    donor_roots: list[str]
    driver_root: str
    entry_files: list[str]
    editable_roots: list[str]
    escalation_edit_roots: list[str]
    forbidden_roots: list[str]
    build_phases: list[CommandSpec]
    build_kconfig: list[str] = field(default_factory=list)
    build_kconfig_unset: list[str] = field(default_factory=list)
    build_kconfig_exclusive_prefixes: list[str] = field(default_factory=list)
    build_dir: str | None = None
    make_bin: str | None = None
    build_env: dict[str, str] = field(default_factory=dict)
    expected_modules: list[str] = field(default_factory=list)
    expected_objects: list[str] = field(default_factory=list)
    require_load_validation: bool = False
    external_edit_policy: str = "allow-listed-escalation"
    load_phases: list[CommandSpec] = field(default_factory=list)
    compile_commands: str | None = None
    donor_compile_commands: list[str] = field(default_factory=list)
    target_compile_command_build_dirs: list[str] = field(default_factory=list)
    compile_path_maps: list[tuple[str, str]] = field(default_factory=list)
    donor_compile_path_maps: list[tuple[str, str]] = field(default_factory=list)
    retrieval_tuning: RetrievalTuningSpec = field(default_factory=RetrievalTuningSpec)
    state_root: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> "CaseSpec":
        return cls.from_dict(load_json(path))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CaseSpec":
        donor_roots = [
            str(Path(str(item)).expanduser())
            for item in list(raw.get("donor_roots") or [])
        ]

        driver_root = normalize_relpath(raw.get("driver_root") or "")
        entry_files = [normalize_relpath(item) for item in raw.get("entry_files") or []]
        editable_roots = [normalize_relpath(item) for item in raw.get("editable_roots") or [driver_root]]
        escalation_edit_roots = [normalize_relpath(item) for item in raw.get("escalation_edit_roots") or []]
        forbidden_roots = [normalize_relpath(item) for item in raw.get("forbidden_roots") or []]
        compile_commands = raw.get("compile_commands")
        donor_compile_commands_raw = raw.get("donor_compile_commands")
        if isinstance(donor_compile_commands_raw, (str, Path)):
            donor_compile_commands = [str(Path(str(donor_compile_commands_raw)).expanduser())]
        else:
            donor_compile_commands = [
                str(Path(str(item)).expanduser())
                for item in list(donor_compile_commands_raw or [])
                if str(item).strip()
            ]
        target_compile_command_build_dirs = [
            str(Path(str(item)).expanduser())
            for item in list(raw.get("target_compile_command_build_dirs") or [])
            if str(item).strip()
        ]
        build_env = raw.get("build_env") or {}
        if not isinstance(build_env, dict):
            raise ValueError("build_env must be an object")
        compile_path_maps = parse_path_maps(raw.get("compile_path_maps"))
        donor_compile_path_maps = parse_path_maps(raw.get("donor_compile_path_maps"))
        retrieval_tuning = RetrievalTuningSpec.from_dict(raw.get("retrieval_tuning"))

        case_id = str(raw.get("case_id") or slugify(driver_root or "driver-backport"))
        spec = cls(
            case_id=case_id,
            old_root=str(Path(str(raw.get("old_root") or "")).expanduser()),
            donor_roots=donor_roots,
            driver_root=driver_root,
            entry_files=entry_files,
            editable_roots=editable_roots,
            escalation_edit_roots=escalation_edit_roots,
            forbidden_roots=forbidden_roots,
            build_kconfig=[str(item).strip() for item in raw.get("build_kconfig") or [] if str(item).strip()],
            build_kconfig_unset=[
                str(item).strip()
                for item in raw.get("build_kconfig_unset") or []
                if str(item).strip()
            ],
            build_kconfig_exclusive_prefixes=[
                str(item).strip()
                for item in raw.get("build_kconfig_exclusive_prefixes") or []
                if str(item).strip()
            ],
            build_dir=str(Path(str(raw.get("build_dir"))).expanduser()) if raw.get("build_dir") else None,
            make_bin=str(raw.get("make_bin")).strip() if raw.get("make_bin") else None,
            build_env=_auto_fill_cross_compile({str(key): str(value) for key, value in build_env.items()}),
            expected_modules=[
                normalize_relpath(item)
                for item in raw.get("expected_modules") or []
                if str(item).strip()
            ],
            expected_objects=[
                normalize_relpath(item)
                for item in raw.get("expected_objects") or []
                if str(item).strip()
            ],
            require_load_validation=bool(raw.get("require_load_validation", False)),
            external_edit_policy=str(raw.get("external_edit_policy") or "allow-listed-escalation"),
            build_phases=_parse_phase_list(raw, "build_phases", "build"),
            load_phases=_parse_phase_list(raw, "load_phases", "load"),
            compile_commands=str(Path(str(compile_commands)).expanduser()) if compile_commands else None,
            donor_compile_commands=donor_compile_commands,
            target_compile_command_build_dirs=target_compile_command_build_dirs,
            compile_path_maps=compile_path_maps,
            donor_compile_path_maps=donor_compile_path_maps,
            retrieval_tuning=retrieval_tuning,
            state_root=str(Path(str(raw.get("state_root"))).expanduser()) if raw.get("state_root") else None,
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.old_root:
            raise ValueError("old_root is required")
        if not self.driver_root:
            raise ValueError("driver_root is required")
        if not self.donor_roots:
            raise ValueError("at least one donor root is required")
        if not self.entry_files:
            raise ValueError("entry_files is required")
        if not self.build_phases:
            raise ValueError("at least one build phase is required")
        if self.require_load_validation and not self.load_phases:
            raise ValueError(
                "require_load_validation=true requires at least one load phase; "
                "configure load_phases or disable the requirement"
            )
        if self.external_edit_policy not in {"disabled", "allow-listed-escalation"}:
            raise ValueError(
                "external_edit_policy must be one of: disabled, allow-listed-escalation"
            )
        for editable_root in self.editable_roots:
            for forbidden_root in self.forbidden_roots:
                if _roots_overlap(editable_root, forbidden_root):
                    raise ValueError(
                        f"editable_roots overlaps forbidden_roots: {editable_root} vs {forbidden_root}"
                    )
        for escalation_root in self.escalation_edit_roots:
            for forbidden_root in self.forbidden_roots:
                if _roots_overlap(escalation_root, forbidden_root):
                    raise ValueError(
                        f"escalation_edit_roots overlaps forbidden_roots: {escalation_root} vs {forbidden_root}"
                    )

        old_root = self.old_root_path
        if not old_root.exists():
            raise ValueError(f"old_root does not exist: {old_root}")
        rooted_fields = {
            "driver_root": [self.driver_root],
            "entry_files": self.entry_files,
            "editable_roots": self.editable_roots,
            "escalation_edit_roots": self.escalation_edit_roots,
            "forbidden_roots": self.forbidden_roots,
            "expected_modules": self.expected_modules,
            "expected_objects": self.expected_objects,
        }
        for field_name, values in rooted_fields.items():
            for value in values:
                try:
                    self.resolve_old(value)
                except ValueError as exc:
                    raise ValueError(
                        f"{field_name} path escapes old_root: {value}"
                    ) from exc
        for donor_root in self.donor_root_paths:
            if not donor_root.exists():
                raise ValueError(f"donor root does not exist: {donor_root}")
            if donor_root == old_root or donor_root.is_relative_to(old_root):
                raise ValueError(f"donor root must not be inside old_root: {donor_root}")
            if old_root.is_relative_to(donor_root):
                raise ValueError(f"old_root must not be inside donor root: {old_root}")

        driver_abs = self.resolve_old(self.driver_root)
        if not driver_abs.exists():
            raise ValueError(f"driver_root does not exist under old_root: {self.driver_root}")

        if not self.compile_commands:
            raise ValueError("compile_commands is required; generate it from the old target tree")
        compile_commands_path = Path(self.compile_commands).resolve()
        if not compile_commands_path.exists():
            raise ValueError(f"compile_commands does not exist: {compile_commands_path}")
        if not self.donor_compile_commands:
            raise ValueError("donor_compile_commands is required; generate it from the donor tree")
        for donor_compile_commands in self.donor_compile_commands:
            donor_compile_commands_path = Path(donor_compile_commands).resolve()
            if not donor_compile_commands_path.exists():
                raise ValueError(f"donor_compile_commands does not exist: {donor_compile_commands_path}")
        for build_dir in self.target_compile_command_build_dirs:
            build_dir_path = Path(build_dir).resolve()
            if not build_dir_path.is_dir():
                raise ValueError(
                    "target compile-command build directory does not exist: "
                    f"{build_dir_path}"
                )

        driver_prefix = self.driver_root.rstrip("/") + "/"
        for entry in self.entry_files:
            if not entry.startswith(driver_prefix):
                raise ValueError(f"entry file must be under driver_root: {entry}")
            if self.resolve_old(entry).exists():
                continue
            if any(self.resolve_donor(entry, donor_index=index).exists() for index, _ in enumerate(self.donor_root_paths)):
                continue
            raise ValueError(f"entry file does not exist under old_root or donor_roots: {entry}")
        for module in self.expected_modules:
            if not module.startswith(driver_prefix) or not module.endswith(".ko"):
                raise ValueError(
                    "expected module must be a .ko path under driver_root: "
                    f"{module}"
                )

    @property
    def old_root_path(self) -> Path:
        return Path(self.old_root).resolve()

    @property
    def donor_root_paths(self) -> list[Path]:
        return [Path(item).resolve() for item in self.donor_roots]

    @property
    def primary_donor_root(self) -> Path:
        return self.donor_root_paths[0]

    @property
    def build_dir_path(self) -> Path | None:
        if not self.build_dir:
            return None
        return Path(self.build_dir).resolve()

    @property
    def build_config_path(self) -> Path:
        if self.build_dir_path is not None:
            return self.build_dir_path / ".config"
        return self.old_root_path / ".config"

    def resolve_old(self, relative_path: str | Path) -> Path:
        return resolve_under(self.old_root_path, relative_path)

    def resolve_donor(self, relative_path: str | Path, *, donor_index: int = 0) -> Path:
        return resolve_under(self.donor_root_paths[donor_index], relative_path)

    def state_root_path(self, default_base: Path) -> Path:
        if self.state_root:
            return Path(self.state_root).resolve()
        return (default_base / slugify(self.case_id)).resolve()

    def classify_edit_root(self, relative_path: str | Path) -> str:
        normalized = normalize_relpath(relative_path)
        target = self.resolve_old(normalized)

        if any(target == self.resolve_old(root) or target.is_relative_to(self.resolve_old(root)) for root in self.forbidden_roots):
            return "forbidden"
        if any(target == self.resolve_old(root) or target.is_relative_to(self.resolve_old(root)) for root in self.editable_roots):
            return "editable"
        if self.external_edit_policy != "disabled":
            if any(
                target == self.resolve_old(root) or target.is_relative_to(self.resolve_old(root))
                for root in self.escalation_edit_roots
            ):
                return "escalation"
        return "forbidden"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "old_root": str(self.old_root_path),
            "donor_roots": [str(path) for path in self.donor_root_paths],
            "driver_root": self.driver_root,
            "entry_files": list(self.entry_files),
            "editable_roots": list(self.editable_roots),
            "escalation_edit_roots": list(self.escalation_edit_roots),
            "forbidden_roots": list(self.forbidden_roots),
            "build_kconfig": list(self.build_kconfig),
            "build_kconfig_unset": list(self.build_kconfig_unset),
            "build_kconfig_exclusive_prefixes": list(self.build_kconfig_exclusive_prefixes),
            "build_dir": str(self.build_dir_path) if self.build_dir_path else None,
            "make_bin": self.make_bin,
            "build_env": dict(self.build_env),
            "expected_modules": list(self.expected_modules),
            "expected_objects": list(self.expected_objects),
            "require_load_validation": self.require_load_validation,
            "external_edit_policy": self.external_edit_policy,
            "build_phases": [phase.to_dict() for phase in self.build_phases],
            "load_phases": [phase.to_dict() for phase in self.load_phases],
            "compile_commands": str(Path(self.compile_commands).resolve()) if self.compile_commands else None,
            "donor_compile_commands": [
                str(Path(item).resolve()) for item in self.donor_compile_commands
            ],
            "target_compile_command_build_dirs": [
                str(Path(item).resolve())
                for item in self.target_compile_command_build_dirs
            ],
            "compile_path_maps": [
                {"from": old, "to": new}
                for old, new in self.compile_path_maps
            ],
            "donor_compile_path_maps": [
                {"from": old, "to": new}
                for old, new in self.donor_compile_path_maps
            ],
            "retrieval_tuning": self.retrieval_tuning.to_dict(),
            "state_root": self.state_root,
        }
