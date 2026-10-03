#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, *, pretty: bool = True) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2 if pretty else None,
            )
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_relpath(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def ordered_unique_relpaths(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_relpath(str(value or ""))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def ordered_unique_strings(values: Iterable[object]) -> list[str]:
    return list(
        dict.fromkeys(
            str(value)
            for value in values
            if str(value or "").strip()
        )
    )


def append_unique_string(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def int_or_zero(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def tri_not(value: bool | None) -> bool | None:
    return None if value is None else not value


def tri_and(left: bool | None, right: bool | None) -> bool | None:
    if left is False or right is False:
        return False
    if left is True and right is True:
        return True
    return None


def tri_or(left: bool | None, right: bool | None) -> bool | None:
    if left is True or right is True:
        return True
    if left is False and right is False:
        return False
    return None


def normalize_root_relative(
    value: str | Path,
    *,
    root: Path,
    preferred_prefix: str | Path | None = None,
) -> str | None:
    text = str(value).strip().replace("\\", "/")
    if not text:
        return None

    path = Path(text)
    if path.is_absolute():
        try:
            return normalize_relpath(path.resolve().relative_to(root.resolve()).as_posix())
        except ValueError:
            pass

    normalized = normalize_relpath(text)
    prefix = normalize_relpath(preferred_prefix) if preferred_prefix else ""
    if prefix:
        if normalized == prefix or normalized.startswith(prefix.rstrip("/") + "/"):
            return normalized

    if path.is_absolute():
        return None
    return normalized


def resolve_under(root: Path, relative_path: str | Path) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / normalize_relpath(relative_path)).resolve()
    if resolved != resolved_root and not resolved.is_relative_to(resolved_root):
        raise ValueError(
            f"path escapes declared root {resolved_root}: {relative_path}"
        )
    return resolved


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
    return slug or "case"


def stable_id(prefix: str, *parts: str) -> str:
    payload = "::".join(parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    return f"{prefix}-{digest}"


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cycle_hint_from_path(path: Path | None) -> int:
    if path is None:
        return 0
    for part in path.parts:
        if part.startswith("cycle-"):
            suffix = part.split("-", 1)[1]
            if suffix.isdigit():
                return int(suffix)
    return 0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
