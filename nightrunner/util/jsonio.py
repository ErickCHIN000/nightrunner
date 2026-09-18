"""JSON helpers: stable formatting, atomic writes, hex rendering for raw words."""

from __future__ import annotations

import json
import os
from pathlib import Path


def dump_json(obj, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=indent, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def hx(value: int, width: int = 8) -> str:
    return f"0x{value:0{width}X}"
