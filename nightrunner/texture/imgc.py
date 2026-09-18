"""IMGC texture header (RPACK part type 0x20) and bitmap payload layout (part type 0x21).

Provenance: notes/survey/03-textures-materials-sdb.md §2 (header, tier A: fields parsed independently against all
49,885 corpus headers; native identities from imagelib IL::ToStr +0x50E10 / GetDescription +0x45D10 / DDS
writer +0x3E230) and §4 (payload layout, tier A: every corpus bitmap size reproduces exactly). Fields marked (C)
are preserved raw and never interpreted.

Header (80-byte fixed part; the stored part is (header_size + 15) & ~15 bytes long)
-----------------------------------------------------------------------------------
    off  size  field         meaning                                                      tier
    0x00 4     magic         "IMGC"                                                       A
    0x04 4     version       0x20191127 — only value seen (49,885/49,885)                 A
    0x08 4     header_size   80; 96 / 106 on the two header-only records                  A
    0x0C 4     flags         0x64 ×43,713  0x44 ×6,107  0x04 ×48  0x54 ×17; bit 0x02 = no bitmap, extension holds a
                             reference name. Other bits: meaning unknown, preserved raw   A (values) / C (meaning)
    0x10 16    minimum[4]    float4 per-channel statistics                                 C (producer unknown)
    0x20 16    maximum[4]
    0x30 16    mean[4]
    0x40 2     width  u16
    0x42 2     height u16
    0x44 1     depth  u8     1 for 2D/cube, slice count for volumes                        A
    0x45 1     reserved      preserved verbatim                                           C
    0x46 1     format        IL::Format id (formats.py), NOT DXGI                          A
    0x47 1     packed        type = packed & 3 (0 2D, 1 cube, 2 volume, 3 invalid); mip_count = packed >> 2   A
    0x48 8     mip_split     u64, 0 in every sampled header; semantics unknown            C
    0x50 …     extension     header_size − 80 bytes (only seen on flag-0x02 records: NUL-terminated reference)  A
    …    …     tail          zero padding up to the 16-byte stored length (preserved raw)

Payload layout (mip-major; survey §4)
------------------------------------
    faces = 6 if cube else 1
    for mip in 0..mip_count-1:
        mw, mh, md = max(1, w >> mip), max(1, h >> mip), max(1, d >> mip)
        length = block ? ceil(mw/4)*ceil(mh/4)*md*unit : mw*mh*md*unit
        stride = (length + 15) & ~15                # every level padded to 16, including the last
        for face in 0..faces-1: level(mip, face) at offset; offset += stride
    payload_size == bitmap part size (49,885/49,885)
Volume slices are consecutive inside a level (slice k at level.offset + k*length/md). Cube face index → ±X/±Y/±Z
is assumed to be DDS order (C, survey §4 / §10.6).

Third-party padding variant (A by census 2026-09-15, notes/FORMATS/imgc.md "Third-party padding variant"): the 12
4096² BC1/BC4 textures of custom_rpacks/frank.rpack store every level back-to-back with NO per-level padding
(part = Σ tight level sizes = 11,184,824 = stock 11,184,848 − 3 × 8). The decoder accepts that layout when — and
only when — the part size equals the tight sum (`detect_level_padding`), reports it as `level_padding = 0`
(stock: 16) and the encoder reproduces whichever value the sidecar carries. Every stock texture is padded.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ..errors import FormatError, ValidationError
from ..util.binio import align_up
from . import formats

MAGIC = b"IMGC"
VERSION = 0x20191127
FIXED_SIZE = 80
FLAG_NO_BITMAP = 0x02        # (A) survey §1.4 / §2
TYPE_2D, TYPE_CUBE, TYPE_VOLUME, TYPE_INVALID = 0, 1, 2, 3
TYPE_NAMES = {TYPE_2D: "2D", TYPE_CUBE: "cube", TYPE_VOLUME: "volume", TYPE_INVALID: "invalid"}
MAX_MIPS = 63
LEVEL_PADDING_STOCK = 16      # (A) every stock bitmap: each level padded to a 16-byte stride
LEVEL_PADDING_TIGHT = 0       # (A) custom_rpacks/frank.rpack, 12 textures: levels back-to-back, no padding
LEVEL_PADDINGS = (LEVEL_PADDING_STOCK, LEVEL_PADDING_TIGHT)   # detection order: stock first

_FIXED = struct.Struct("<4sIII48sHHBBBBQ")   # magic, version, header_size, flags, stats, w, h, d, res, fmt, packed, mip_split
assert _FIXED.size == FIXED_SIZE
_STATS = struct.Struct("<12f")


@dataclass
class Level:
    """One (mip, face) surface inside the bitmap part. `size` is tight, `padded_size` includes the 16-byte pad."""
    mip: int
    face: int
    width: int
    height: int
    depth: int
    offset: int
    size: int
    padded_size: int

    @property
    def slice_size(self) -> int:
        return self.size // self.depth


@dataclass
class ImgcHeader:
    version: int = VERSION
    header_size: int = FIXED_SIZE
    flags: int = 0
    stats_raw: bytes = bytes(48)      # the three float4s exactly as stored (bit-exact round trip, NaN-safe)
    width: int = 0
    height: int = 0
    depth: int = 1
    reserved: int = 0
    format: int = 0
    packed: int = 0                   # type | mip_count << 2
    mip_split: int = 0
    extension: bytes = b""            # header_size - 80 bytes
    tail: bytes = b""                 # zero padding after header_size up to the stored (16-aligned) length
    magic: bytes = MAGIC

    # ---- derived -------------------------------------------------------------------------------------------

    @property
    def tex_type(self) -> int:
        return self.packed & 3

    @property
    def type_name(self) -> str:
        return TYPE_NAMES[self.tex_type]

    @property
    def mip_count(self) -> int:
        return self.packed >> 2

    @property
    def is_cube(self) -> bool:
        return self.tex_type == TYPE_CUBE

    @property
    def is_volume(self) -> bool:
        return self.tex_type == TYPE_VOLUME

    @property
    def faces(self) -> int:
        return 6 if self.is_cube else 1

    @property
    def header_only(self) -> bool:
        return bool(self.flags & FLAG_NO_BITMAP)

    @property
    def reference(self) -> str | None:
        """NUL-terminated name stored in the extension of header-only records (survey section 1.4)."""
        if not self.extension:
            return None
        return self.extension.split(b"\0", 1)[0].decode("utf-8", "replace")

    @property
    def stored_size(self) -> int:
        return align_up(self.header_size, 16)

    @property
    def minimum(self) -> tuple[float, float, float, float]:
        return tuple(_STATS.unpack(self.stats_raw)[0:4])

    @property
    def maximum(self) -> tuple[float, float, float, float]:
        return tuple(_STATS.unpack(self.stats_raw)[4:8])

    @property
    def mean(self) -> tuple[float, float, float, float]:
        return tuple(_STATS.unpack(self.stats_raw)[8:12])

    @property
    def format_name(self) -> str:
        return formats.name_of(self.format)

    def set_stats(self, minimum, maximum, mean) -> None:
        self.stats_raw = _STATS.pack(*minimum, *maximum, *mean)

    def set_geometry(self, width: int, height: int, depth: int, tex_type: int, mip_count: int, il_format: int) -> None:
        if not (1 <= width <= 0xFFFF and 1 <= height <= 0xFFFF):
            raise ValidationError(f"IMGC width/height must be 1..65535 (got {width}x{height})")
        if not 1 <= depth <= 0xFF:
            raise ValidationError(f"IMGC depth must be 1..255 (got {depth})")
        if tex_type not in (TYPE_2D, TYPE_CUBE, TYPE_VOLUME):
            raise ValidationError(f"IMGC type {tex_type} is invalid")
        if not 1 <= mip_count <= MAX_MIPS:
            raise ValidationError(f"IMGC mip count must be 1..63 (got {mip_count})")
        if il_format not in formats.FORMATS:
            raise ValidationError(f"IL format {il_format} is not in the native enum")
        if tex_type != TYPE_VOLUME and depth != 1:
            raise ValidationError("depth must be 1 for 2D and cube textures (corpus: 49,874/49,874)")
        self.width, self.height, self.depth = width, height, depth
        self.packed = (tex_type & 3) | (mip_count << 2)
        self.format = il_format

    # ---- validation ----------------------------------------------------------------------------------------

    def check_geometry(self) -> None:
        """The checks the old reader applied before laying out a bitmap (survey section 2 'Validation asserted')."""
        if self.width == 0 or self.height == 0 or self.depth == 0:
            raise FormatError(f"IMGC: zero dimension {self.width}x{self.height}x{self.depth}")
        if self.mip_count == 0:
            raise FormatError("IMGC: mip count 0")
        if self.tex_type == TYPE_INVALID:
            raise FormatError("IMGC: texture type 3 (invalid)")
        if self.format not in formats.FORMATS:
            raise FormatError(f"IMGC: IL format id {self.format} is not in the native enum")

    # ---- (de)serialisation ---------------------------------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "version": f"0x{self.version:08X}", "header_size": self.header_size, "flags": f"0x{self.flags:02X}",
            "stats": {"minimum": list(self.minimum), "maximum": list(self.maximum), "mean": list(self.mean),
                      "raw_hex": self.stats_raw.hex()},
            "width": self.width, "height": self.height, "depth": self.depth, "reserved": self.reserved,
            "format": self.format, "format_name": self.format_name,
            "type": self.tex_type, "type_name": self.type_name, "mip_count": self.mip_count,
            "mip_split": f"0x{self.mip_split:016X}", "extension_hex": self.extension.hex(), "tail_hex": self.tail.hex(),
            "header_only": self.header_only, "reference": self.reference,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ImgcHeader":
        st = d.get("stats") or {}
        if st.get("raw_hex"):
            stats_raw = bytes.fromhex(st["raw_hex"])
        else:
            stats_raw = _STATS.pack(*st.get("minimum", [0] * 4), *st.get("maximum", [0] * 4), *st.get("mean", [0] * 4))
        h = cls(version=int(str(d.get("version", VERSION)), 0), header_size=int(d.get("header_size", FIXED_SIZE)),
                flags=int(str(d.get("flags", 0)), 0), stats_raw=stats_raw,
                width=int(d.get("width", 0)), height=int(d.get("height", 0)), depth=int(d.get("depth", 1)),
                reserved=int(d.get("reserved", 0)), format=int(d.get("format", 0)),
                packed=(int(d.get("type", 0)) & 3) | (int(d.get("mip_count", 0)) << 2),
                mip_split=int(str(d.get("mip_split", 0)), 0),
                extension=bytes.fromhex(d.get("extension_hex", "")), tail=bytes.fromhex(d.get("tail_hex", "")))
        return h


def parse_header(raw, *, strict_length: bool = True) -> ImgcHeader:
    """Parse the bytes of a 0x20 part. With strict_length the buffer must be exactly the 16-byte-padded
    header_size (what every shipped part is); otherwise trailing bytes are ignored."""
    raw = bytes(raw)
    if len(raw) < FIXED_SIZE:
        raise FormatError(f"IMGC header is {len(raw)} bytes, shorter than the 80-byte fixed part")
    magic, version, header_size, flags, stats, w, h, d, res, fmt, packed, mip_split = _FIXED.unpack_from(raw, 0)
    if magic != MAGIC:
        raise FormatError(f"IMGC: bad magic {magic!r}")
    if version != VERSION:
        raise FormatError(f"IMGC: version 0x{version:08X} (only 0x{VERSION:08X} is known)")
    if header_size < FIXED_SIZE:
        raise FormatError(f"IMGC: header_size {header_size} < 80")
    stored = align_up(header_size, 16)
    if len(raw) < stored:
        raise FormatError(f"IMGC: part is {len(raw)} bytes but header_size {header_size} pads to {stored}")
    if strict_length and len(raw) != stored:
        raise FormatError(f"IMGC: part is {len(raw)} bytes, expected the padded header size {stored}")
    return ImgcHeader(version=version, header_size=header_size, flags=flags, stats_raw=stats, width=w, height=h,
                      depth=d, reserved=res, format=fmt, packed=packed, mip_split=mip_split,
                      extension=raw[FIXED_SIZE:header_size], tail=raw[header_size:stored], magic=magic)


def pack_header(h: ImgcHeader) -> bytes:
    """Serialise to the exact stored part bytes (16-byte padded)."""
    if len(h.extension) != h.header_size - FIXED_SIZE:
        raise ValidationError(f"IMGC: extension is {len(h.extension)} bytes but header_size {h.header_size} implies "
                              f"{h.header_size - FIXED_SIZE}")
    if len(h.stats_raw) != 48:
        raise ValidationError("IMGC: stats_raw must be 48 bytes")
    if not (0 <= h.width <= 0xFFFF and 0 <= h.height <= 0xFFFF and 0 <= h.depth <= 0xFF and 0 <= h.format <= 0xFF
            and 0 <= h.packed <= 0xFF and 0 <= h.reserved <= 0xFF):
        raise ValidationError("IMGC: field out of range")
    fixed = _FIXED.pack(h.magic, h.version, h.header_size, h.flags, h.stats_raw, h.width, h.height, h.depth,
                        h.reserved, h.format, h.packed, h.mip_split)
    stored = h.stored_size
    tail = h.tail if len(h.tail) == stored - h.header_size else bytes(stored - h.header_size)
    return fixed + h.extension + tail


# --------------------------------------------------------------------------------------------------------------
# Payload layout
# --------------------------------------------------------------------------------------------------------------

def mip_dims(width: int, height: int, depth: int, mip: int) -> tuple[int, int, int]:
    return max(1, width >> mip), max(1, height >> mip), max(1, depth >> mip)


def _check_padding(level_padding: int) -> int:
    if level_padding not in LEVEL_PADDINGS:
        raise ValidationError(f"IMGC: level_padding {level_padding!r} is neither {LEVEL_PADDING_STOCK} (stock) nor "
                              f"{LEVEL_PADDING_TIGHT} (third-party tight layout)")
    return level_padding


def level_layout(h: ImgcHeader, level_padding: int = LEVEL_PADDING_STOCK) -> list[Level]:
    """Mip-major list of (mip, face) surfaces with offsets inside the bitmap part (survey section 4).
    `level_padding` 16 = stock stride (every level padded to 16); 0 = the tight third-party layout."""
    h.check_geometry()
    _check_padding(level_padding)
    levels: list[Level] = []
    offset = 0
    for mip in range(h.mip_count):
        mw, mh, md = mip_dims(h.width, h.height, h.depth, mip)
        length = formats.level_bytes(h.format, mw, mh, md)
        stride = align_up(length, level_padding) if level_padding else length
        for face in range(h.faces):
            levels.append(Level(mip, face, mw, mh, md, offset, length, stride))
            offset += stride
    return levels


def payload_size(h: ImgcHeader, level_padding: int = LEVEL_PADDING_STOCK) -> int:
    levels = level_layout(h, level_padding)
    return levels[-1].offset + levels[-1].padded_size if levels else 0


def detect_level_padding(h: ImgcHeader, bitmap_size: int) -> int:
    """The level padding whose layout reproduces *bitmap_size*: 16 (stock, tried first) or 0 (tight). Raises
    FormatError when neither matches — the size is never repaired or guessed."""
    sizes = {}
    for pad in LEVEL_PADDINGS:
        sizes[pad] = payload_size(h, pad)
        if sizes[pad] == bitmap_size:
            return pad
    raise FormatError(f"IMGC: computed payload {sizes[LEVEL_PADDING_STOCK]} bytes (padded) / "
                      f"{sizes[LEVEL_PADDING_TIGHT]} bytes (tight) != bitmap part {bitmap_size} bytes "
                      f"({h.width}x{h.height}x{h.depth} {h.format_name} {h.type_name} mips={h.mip_count})")


def check_payload(h: ImgcHeader, bitmap_size: int, level_padding: int | None = None) -> list[Level]:
    """Level layout validated against the actual part size. With `level_padding` None the padding is detected
    (16 first, then 0); with an explicit value that layout must match exactly."""
    if level_padding is None:
        level_padding = detect_level_padding(h, bitmap_size)
    levels = level_layout(h, level_padding)
    total = levels[-1].offset + levels[-1].padded_size
    if total != bitmap_size:
        raise FormatError(f"IMGC: computed payload {total} bytes (level_padding {level_padding}) != bitmap part "
                          f"{bitmap_size} bytes ({h.width}x{h.height}x{h.depth} {h.format_name} {h.type_name} "
                          f"mips={h.mip_count})")
    return levels


def split_payload(h: ImgcHeader, bitmap, level_padding: int | None = None) -> list[memoryview]:
    """Tight (unpadded) per-level views, mip-major, in the order of `level_layout`. No copies. The padding is
    detected from the part size unless given."""
    mv = memoryview(bitmap)
    levels = check_payload(h, len(mv), level_padding)
    return [mv[lv.offset: lv.offset + lv.size] for lv in levels]


def join_payload(h: ImgcHeader, levels, level_padding: int = LEVEL_PADDING_STOCK) -> bytes:
    """Inverse of split_payload: concatenate tight levels (mip-major order) re-inserting the per-level padding
    (16-byte stride by default; 0 reproduces the third-party tight layout)."""
    layout = level_layout(h, level_padding)
    if len(levels) != len(layout):
        raise ValidationError(f"IMGC: {len(levels)} levels given, layout has {len(layout)}")
    out = bytearray(layout[-1].offset + layout[-1].padded_size)
    for lv, data in zip(layout, levels):
        if len(data) != lv.size:
            raise ValidationError(f"IMGC: level mip{lv.mip} face{lv.face} is {len(data)} bytes, expected {lv.size}")
        out[lv.offset: lv.offset + lv.size] = data
    return bytes(out)


def find_level(levels: list[Level], mip: int, face: int = 0) -> Level:
    for lv in levels:
        if lv.mip == mip and lv.face == face:
            return lv
    raise ValidationError(f"no level mip={mip} face={face}")
