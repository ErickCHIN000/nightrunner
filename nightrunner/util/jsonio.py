"""JSON helpers: stable formatting, atomic writes, hex rendering for raw words."""

from __future__ import annotations

import json
import os
from pathlib import Path


def dump_json(obj, path: Path | str, indent: int | None = 2) -> None:
    """Atomically write *obj* as JSON. `indent=None` writes it compact, without the separator
    padding that would leave a large file with neither readable indentation nor a small size."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=indent, separators=(",", ":") if indent is None else None,
                  ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def hx(value: int, width: int = 8) -> str:
    return f"0x{value:0{width}X}"
