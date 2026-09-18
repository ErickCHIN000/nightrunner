"""Hashing helpers (sha256 everywhere; chunked so multi-GB packs stream)."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

CHUNK = 1 << 20
# whole-file hashes keyed by (path, size, mtime); game packs are multi-GB and hashed on every extract/build
_CACHE: dict[str, list] = {}
_LOCK = threading.Lock()
_DISK = Path(os.environ.get("NIGHTRUNNER_HASH_CACHE", Path(__file__).resolve().parents[2] / ".cache" / "sha256.json"))
_loaded = False


def _load():
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        _CACHE.update(json.loads(_DISK.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass


def _save():
    try:
        _DISK.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DISK.with_suffix(".tmp")
        tmp.write_text(json.dumps(_CACHE), encoding="utf-8")
        os.replace(tmp, _DISK)
    except OSError:
        pass


def sha256_bytes(data) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, offset: int = 0, size: int | None = None) -> str:
    if offset == 0 and size is None:
        st = os.stat(path)
        key = os.path.normcase(os.path.abspath(path))
        stamp = [st.st_size, st.st_mtime_ns]
        with _LOCK:
            _load()
            hit = _CACHE.get(key)
            if hit and hit[:2] == stamp:
                return hit[2]
        digest = _hash(path, 0, None)
        with _LOCK:
            _CACHE[key] = stamp + [digest]
            _save()
        return digest
    return _hash(path, offset, size)


def _hash(path, offset, size) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(offset)
        remaining = size
        while True:
            want = CHUNK if remaining is None else min(CHUNK, remaining)
            if want <= 0:
                break
            chunk = fh.read(want)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()
