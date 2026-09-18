"""Resource-name helpers: engine case folding and filesystem-safe names."""

from __future__ import annotations

import re

_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def engine_fold(name: bytes) -> bytes:
    """Fold exactly like FindLogicalResourceUsingName (ResourceCore +0x19250): ASCII 'A'..'Z' only."""
    return bytes(b + 0x20 if 0x41 <= b <= 0x5A else b for b in name)


def decode_name(raw: bytes) -> str:
    return raw.decode("utf-8", "surrogateescape")


def encode_name(text: str) -> bytes:
    return text.encode("utf-8", "surrogateescape")


def safe_filename(name: str, max_len: int = 120) -> str:
    """Filesystem-safe rendering of a resource name (never used as the identity — pack.json holds the exact bytes)."""
    s = _BAD.sub("_", name)
    s = s.strip(" .")
    if not s:
        s = "_"
    stem = s.split(".")[0].upper()
    if stem in _RESERVED:
        s = "_" + s
    if len(s) > max_len:
        s = s[:max_len]
    return s


def resource_dirname(index: int, name: str) -> str:
    return f"{index:06d}_{safe_filename(name)}"
