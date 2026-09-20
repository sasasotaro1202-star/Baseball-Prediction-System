"""Atomic filesystem primitives for production artifacts.

All writes stay on the destination filesystem and are committed with os.replace,
which is atomic on POSIX. This prevents partially-written JSON/manifests from
being mistaken for successful pipeline outputs after an interrupted runner.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
        try:
            dir_fd = os.open(dst.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Directory fsync is a durability enhancement and is not portable
            # to every filesystem used by local development environments.
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: str | Path, obj: Any) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_write_text(path, text)


def file_sha256(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
