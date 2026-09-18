"""Small binary helpers shared by every codec."""

from __future__ import annotations

import struct


def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    return (value + alignment - 1) & ~(alignment - 1)


def u32(buf, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def u16(buf, off: int) -> int:
    return struct.unpack_from("<H", buf, off)[0]


def u64(buf, off: int) -> int:
    return struct.unpack_from("<Q", buf, off)[0]


def i16(buf, off: int) -> int:
    return struct.unpack_from("<h", buf, off)[0]


def f32(buf, off: int) -> float:
    return struct.unpack_from("<f", buf, off)[0]


def cstring(buf, off: int, limit: int | None = None) -> bytes:
    """NUL-terminated byte string starting at *off*. Raises if no NUL before *limit*/end."""
    end = len(buf) if limit is None else min(len(buf), off + limit)
    mv = memoryview(buf)[off:end]
    idx = bytes(mv).find(b"\0")
    if idx < 0:
        raise ValueError(f"unterminated string at 0x{off:X}")
    return bytes(mv[:idx])


def hexdump(data: bytes, width: int = 16, max_bytes: int = 256) -> str:
    lines = []
    data = bytes(data[:max_bytes])
    for i in range(0, len(data), width):
        chunk = data[i : i + width]
        hexs = " ".join(f"{b:02X}" for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{i:08X}  {hexs:<{width * 3}} {asc}")
    return "\n".join(lines)
