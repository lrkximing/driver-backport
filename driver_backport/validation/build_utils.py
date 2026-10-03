#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


MAKE_VERSION_RE = re.compile(r"GNU Make\s+(\d+)\.(\d+)(?:\.(\d+))?")
MINIMUM_GNU_MAKE_VERSION = (4, 0, 0)


def case_build_env(spec: Any) -> dict[str, str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env.update(spec.build_env)
    if spec.build_dir_path is not None:
        env.setdefault("KCONFIG_CONFIG", str(spec.build_config_path))
    return env


def scripts_config_command(
    scripts_config: Path,
    config_name: str,
    value: str,
    *,
    config_path: Path | None,
) -> list[str]:
    symbol = config_name.removeprefix("CONFIG_")
    command = [str(scripts_config)]
    if config_path is not None:
        command.extend(["--file", str(config_path)])
    if value == "y":
        return command + ["--enable", symbol]
    if value == "m":
        return command + ["--module", symbol]
    if value == "n":
        return command + ["--disable", symbol]
    return command + ["--set-val", symbol, value]


def olddefconfig_command(spec: Any) -> list[str]:
    command = [find_make_binary(spec.make_bin)]
    if spec.build_dir_path is not None:
        command.append(f"O={spec.build_dir_path}")
    command.append("olddefconfig")
    return command


def _version_key(text: str) -> tuple[int, int, int] | None:
    match = MAKE_VERSION_RE.search(text)
    if not match:
        return None
    return (
        int(match.group(1) or 0),
        int(match.group(2) or 0),
        int(match.group(3) or 0),
    )


def _probe_make(candidate: str) -> tuple[tuple[int, int, int] | None, str]:
    process = subprocess.run(
        [candidate, "--version"],
        text=True,
        capture_output=True,
    )
    output = process.stdout + process.stderr
    if process.returncode != 0:
        return None, output.strip()
    return _version_key(output), output.strip()


def find_make_binary(explicit: str | None = None) -> str:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str | None) -> None:
        if not value:
            return
        resolved = shutil.which(value) or value
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    add(explicit)
    add(os.environ.get("GMAKE"))
    add(os.environ.get("MAKE"))
    add("gmake")
    add("/opt/homebrew/bin/gmake")
    add("/usr/local/bin/gmake")
    add("make")

    failures: list[str] = []
    for candidate in candidates:
        version, _output = _probe_make(candidate)
        if version is None:
            failures.append(f"{candidate}: unavailable")
            continue
        if version >= MINIMUM_GNU_MAKE_VERSION:
            return candidate
        failures.append(
            f"{candidate}: version {version[0]}.{version[1]}.{version[2]} < "
            f"{MINIMUM_GNU_MAKE_VERSION[0]}."
            f"{MINIMUM_GNU_MAKE_VERSION[1]}."
            f"{MINIMUM_GNU_MAKE_VERSION[2]}"
        )

    detail = "; ".join(failures) if failures else "no candidates tried"
    raise RuntimeError(f"no suitable GNU Make found; {detail}")


def parse_kconfig_assignment(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    name = name.strip()
    value = value.strip()
    if not sep or not name.startswith("CONFIG_") or not value:
        raise ValueError(f"invalid Kconfig assignment: {text!r}")
    return name, value


def kconfig_symbol_name(value: str) -> str:
    """Return the bare symbol from CONFIG_FOO or CONFIG_FOO=value."""

    return str(value or "").strip().split("=", 1)[0].strip().removeprefix("CONFIG_")
