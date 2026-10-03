from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from driver_backport.core.common import stable_json_sha256, write_json


CONTENT_STORE_DIR_NAME = "content_store"
CONTENT_RECORD_REF_PREFIX = "content-record"


def _record_kind(value: str) -> str:
    normalized = str(value or "").strip().replace("_", "-")
    if not normalized or not all(
        character.isalnum() or character == "-"
        for character in normalized
    ):
        raise ValueError(f"invalid content record kind: {value!r}")
    return normalized


def _record_path(
    state_root: Path,
    *,
    kind: str,
    digest: str,
) -> Path:
    return (
        state_root.resolve()
        / CONTENT_STORE_DIR_NAME
        / "records"
        / _record_kind(kind)
        / f"{digest}.json"
    )


def put_content_record(
    state_root: Path,
    *,
    kind: str,
    record: Any,
) -> str:
    """Persist one immutable JSON value and return its stable reference."""

    normalized_kind = _record_kind(kind)
    digest = stable_json_sha256(record)
    path = _record_path(
        state_root,
        kind=normalized_kind,
        digest=digest,
    )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_json_sha256(existing) != digest:
            raise ValueError(f"content record digest mismatch at {path}")
    else:
        write_json(path, record, pretty=False)
    return f"{CONTENT_RECORD_REF_PREFIX}:{normalized_kind}:{digest}"


def parse_content_record_ref(reference: str) -> tuple[str, str]:
    parts = str(reference or "").split(":")
    if len(parts) != 3 or parts[0] != CONTENT_RECORD_REF_PREFIX:
        raise ValueError(f"invalid content record reference: {reference}")
    kind = _record_kind(parts[1])
    digest = str(parts[2])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"invalid content record digest: {reference}")
    return kind, digest


def load_content_record(state_root: Path, reference: str) -> Any:
    kind, digest = parse_content_record_ref(reference)
    path = _record_path(state_root, kind=kind, digest=digest)
    if not path.is_file():
        raise FileNotFoundError(
            f"content record does not exist: {reference}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if stable_json_sha256(payload) != digest:
        raise ValueError(
            f"content record content does not match {reference}"
        )
    return payload
