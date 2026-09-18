"""Native IL::Format enum ↔ DXGI ↔ DDS FourCC ↔ block/unit sizes.

Provenance (see notes/survey/03-textures-materials-sdb.md §3 and notes/FORMATS/dds-mapping.md):

* Names, ids and holes of the 70-entry enum: (A) imagelib_x64_rwdi.dll IL::ToStr +0x50E10 (BN 2026-09-10),
  cross-checked with dltb-mf sdk/generated/assets/image-formats.json. Holes at 4, 5, 11, 43, 72; unknown = 255.
* DXGI equivalents, byte order and legacy FourCC for the 21 ids marked tier "A": (A) imagelib native DDS writer
  +0x3E230 ("IL_FormatToDDSHeader") and GetDescription +0x45D10 (block bytes), plus the corpus census
  (49,885 textures, 18 distinct ids actually stored; BC2/BC3/BC6H_SF16 are decoder-only, 0 occurrences).
* Everything marked tier "C" is *derived from the enum name alone* (the obvious DXGI format with the same
  channel/width/type spelling). Those ids never occur in the shipped corpus; they are here so that a parser can
  compute a level layout and refuse/accept an import with a precise message, not because the engine's own
  id→DXGI table was read for them. Where the name does not identify a DXGI layout unambiguously (BGRA8/BGRX8/
  XBGR8 byte order, A2RGB10, D24FS8, BGR32F, R8_NO_TYPELESS) `dxgi` is None and the id is refused on export.
* sRGB: (A) IL::IsSrgb (imagelib +0x48DC0) is true only for 33, 70, 71, 73, 74. No sRGB id appears in the corpus;
  how colour textures are flagged sRGB is a hole (survey §10.2).
* `unit` = bytes per pixel (uncompressed) or bytes per 4×4 block (BC). (A) for the 21 mapped ids (GetDescription
  byte 5 = block bytes; base row = ceil(w/4)×unit). (C) for the rest (from the name's bit widths).

Corpus census 2026-09-10 (survey §3.1), per id: 63:12,002  59:14,664  68:7,427  64:7,703  62:1,188  0:4,055
38:1,363  66:1,255  48:96  46:61  16:53  65:4  39:4  32:3  40:3  8:2  15:1  47:1  (sum 49,885).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ILFormat:
    id: int
    name: str
    unit: int                      # bytes per pixel, or per 4×4 block when `block`
    block: bool
    dxgi: int | None               # DXGI_FORMAT number, None = no unambiguous equivalent (export refused)
    dxgi_name: str | None
    fourcc: bytes | None           # legacy DDS FourCC the native writer / DirectXTex use for this layout
    channels: tuple[str, ...]      # storage order of channels for previews ("B","G","R","A" = bytes B,G,R,A)
    kind: str                      # "unorm" | "snorm" | "uint" | "sint" | "float" | "depth" | "packed" | "bc"
    tier: str                      # "A" (native + corpus) | "C" (name-derived, never observed)
    corpus: int = 0                # occurrences in the 49,885-texture census (survey §3.1)
    srgb: bool = False             # IL::IsSrgb true
    linear_id: int | None = None   # for sRGB ids: the linear id with the same layout
    srgb_id: int | None = None     # for linear ids: the sRGB id with the same layout


# id, name, unit, block, dxgi, dxgi_name, fourcc, channels, kind, tier, corpus
_ROWS = [
    (0,  "R8",            1, False, 61, "R8_UNORM",            None,     ("R",),             "unorm", "A", 4055),
    (1,  "R8_SNORM",      1, False, 63, "R8_SNORM",            None,     ("R",),             "snorm", "C", 0),
    (2,  "R8_UINT",       1, False, 62, "R8_UINT",             None,     ("R",),             "uint",  "C", 0),
    (3,  "R8_SINT",       1, False, 64, "R8_SINT",             None,     ("R",),             "sint",  "C", 0),
    (6,  "R16F",          2, False, 54, "R16_FLOAT",           None,     ("R",),             "float", "C", 0),
    (7,  "R16",           2, False, 56, "R16_UNORM",           None,     ("R",),             "unorm", "C", 0),
    (8,  "R16_SNORM",     2, False, 58, "R16_SNORM",           None,     ("R",),             "snorm", "A", 2),
    (9,  "R16_UINT",      2, False, 57, "R16_UINT",            None,     ("R",),             "uint",  "C", 0),
    (10, "R16_SINT",      2, False, 59, "R16_SINT",            None,     ("R",),             "sint",  "C", 0),
    (12, "R32F",          4, False, 41, "R32_FLOAT",           None,     ("R",),             "float", "C", 0),
    (13, "R32_UINT",      4, False, 42, "R32_UINT",            None,     ("R",),             "uint",  "C", 0),
    (14, "R32_SINT",      4, False, 43, "R32_SINT",            None,     ("R",),             "sint",  "C", 0),
    (15, "RG8",           2, False, 49, "R8G8_UNORM",          None,     ("R", "G"),         "unorm", "A", 1),
    (16, "RG8_SNORM",     2, False, 51, "R8G8_SNORM",          None,     ("R", "G"),         "snorm", "A", 53),
    (17, "RG8_UINT",      2, False, 50, "R8G8_UINT",           None,     ("R", "G"),         "uint",  "C", 0),
    (18, "RG8_SINT",      2, False, 52, "R8G8_SINT",           None,     ("R", "G"),         "sint",  "C", 0),
    (19, "RG16F",         4, False, 34, "R16G16_FLOAT",        None,     ("R", "G"),         "float", "C", 0),
    (20, "RG16",          4, False, 35, "R16G16_UNORM",        None,     ("R", "G"),         "unorm", "C", 0),
    (21, "RG16_SNORM",    4, False, 37, "R16G16_SNORM",        None,     ("R", "G"),         "snorm", "C", 0),
    (22, "RG16_UINT",     4, False, 36, "R16G16_UINT",         None,     ("R", "G"),         "uint",  "C", 0),
    (23, "RG16_SINT",     4, False, 38, "R16G16_SINT",         None,     ("R", "G"),         "sint",  "C", 0),
    (24, "RG32F",         8, False, 16, "R32G32_FLOAT",        None,     ("R", "G"),         "float", "C", 0),
    (25, "RG32_UINT",     8, False, 17, "R32G32_UINT",         None,     ("R", "G"),         "uint",  "C", 0),
    (26, "RG32_SINT",     8, False, 18, "R32G32_SINT",         None,     ("R", "G"),         "sint",  "C", 0),
    (27, "R5G6B5",        2, False, 85, "B5G6R5_UNORM",        None,     ("R", "G", "B"),    "packed", "C", 0),
    (28, "RGB8",          3, False, None, None,                None,     ("R", "G", "B"),    "unorm", "C", 0),
    (29, "BGR8",          3, False, None, None,                None,     ("B", "G", "R"),    "unorm", "C", 0),
    (30, "R11G11B10F",    4, False, 26, "R11G11B10_FLOAT",     None,     ("R", "G", "B"),    "packed", "C", 0),
    (31, "BGR32F",       12, False, None, None,                None,     ("B", "G", "R"),    "float", "C", 0),
    (32, "ARGB8",         4, False, 87, "B8G8R8A8_UNORM",      None,     ("B", "G", "R", "A"), "unorm", "A", 3),
    (33, "ARGB8_SRGB",    4, False, 91, "B8G8R8A8_UNORM_SRGB", None,     ("B", "G", "R", "A"), "unorm", "C", 0),
    (34, "XRGB8",         4, False, 88, "B8G8R8X8_UNORM",      None,     ("B", "G", "R", "X"), "unorm", "C", 0),
    (35, "BGRA8",         4, False, None, None,                None,     ("?", "?", "?", "?"), "unorm", "C", 0),
    (36, "BGRX8",         4, False, None, None,                None,     ("?", "?", "?", "?"), "unorm", "C", 0),
    (37, "XBGR8",         4, False, None, None,                None,     ("?", "?", "?", "?"), "unorm", "C", 0),
    (38, "RGBA8",         4, False, 28, "R8G8B8A8_UNORM",      None,     ("R", "G", "B", "A"), "unorm", "A", 1363),
    (39, "RGBA8_SNORM",   4, False, 31, "R8G8B8A8_SNORM",      None,     ("R", "G", "B", "A"), "snorm", "A", 4),
    (40, "RGBA8_UINT",    4, False, 30, "R8G8B8A8_UINT",       None,     ("R", "G", "B", "A"), "uint",  "A", 3),
    (41, "RGBA8_SINT",    4, False, 32, "R8G8B8A8_SINT",       None,     ("R", "G", "B", "A"), "sint",  "C", 0),
    (42, "A2RGB10",       4, False, None, None,                None,     ("?", "?", "?", "?"), "packed", "C", 0),
    (44, "RGB10A2",       4, False, 24, "R10G10B10A2_UNORM",   None,     ("R", "G", "B", "A"), "packed", "C", 0),
    (45, "RGB10A2_UINT",  4, False, 25, "R10G10B10A2_UINT",    None,     ("R", "G", "B", "A"), "packed", "C", 0),
    (46, "RGBA16F",       8, False, 10, "R16G16B16A16_FLOAT",  None,     ("R", "G", "B", "A"), "float", "A", 61),
    (47, "RGBA16",        8, False, 11, "R16G16B16A16_UNORM",  None,     ("R", "G", "B", "A"), "unorm", "A", 1),
    (48, "RGBA16_SNORM",  8, False, 13, "R16G16B16A16_SNORM",  None,     ("R", "G", "B", "A"), "snorm", "A", 96),
    (49, "RGBA16_UINT",   8, False, 12, "R16G16B16A16_UINT",   None,     ("R", "G", "B", "A"), "uint",  "C", 0),
    (50, "RGBA16_SINT",   8, False, 14, "R16G16B16A16_SINT",   None,     ("R", "G", "B", "A"), "sint",  "C", 0),
    (51, "RGBA32F",      16, False, 2,  "R32G32B32A32_FLOAT",  None,     ("R", "G", "B", "A"), "float", "C", 0),
    (52, "RGBA32_UINT",  16, False, 3,  "R32G32B32A32_UINT",   None,     ("R", "G", "B", "A"), "uint",  "C", 0),
    (53, "RGBA32_SINT",  16, False, 4,  "R32G32B32A32_SINT",   None,     ("R", "G", "B", "A"), "sint",  "C", 0),
    (54, "D16",           2, False, 55, "D16_UNORM",           None,     ("D",),             "depth", "C", 0),
    (55, "D24_S8",        4, False, 45, "D24_UNORM_S8_UINT",   None,     ("D", "S"),         "depth", "C", 0),
    (56, "D32F",          4, False, 40, "D32_FLOAT",           None,     ("D",),             "depth", "C", 0),
    (57, "D24FS8",        4, False, None, None,                None,     ("D", "S"),         "depth", "C", 0),
    (58, "D32F_S8",       8, False, 20, "D32_FLOAT_S8X24_UINT", None,    ("D", "S"),         "depth", "C", 0),
    (59, "BC1",           8, True,  71, "BC1_UNORM",           b"DXT1",  ("R", "G", "B", "A"), "bc", "A", 14664),
    (60, "BC2",          16, True,  74, "BC2_UNORM",           b"DXT3",  ("R", "G", "B", "A"), "bc", "A", 0),
    (61, "BC3",          16, True,  77, "BC3_UNORM",           b"DXT5",  ("R", "G", "B", "A"), "bc", "A", 0),
    (62, "BC4_SNORM",     8, True,  81, "BC4_SNORM",           b"BC4S",  ("R",),             "bc", "A", 1188),
    (63, "BC4",           8, True,  80, "BC4_UNORM",           b"ATI1",  ("R",),             "bc", "A", 12002),
    (64, "BC5_SNORM",    16, True,  84, "BC5_SNORM",           b"BC5S",  ("R", "G"),         "bc", "A", 7703),
    (65, "BC5",          16, True,  83, "BC5_UNORM",           b"ATI2",  ("R", "G"),         "bc", "A", 4),
    (66, "BC6H_UF16",    16, True,  95, "BC6H_UF16",           None,     ("R", "G", "B"),    "bc", "A", 1255),
    (67, "BC6H_SF16",    16, True,  96, "BC6H_SF16",           None,     ("R", "G", "B"),    "bc", "A", 0),
    (68, "BC7",          16, True,  98, "BC7_UNORM",           None,     ("R", "G", "B", "A"), "bc", "A", 7427),
    (69, "R8_NO_TYPELESS", 1, False, None, None,               None,     ("R",),             "unorm", "C", 0),
    (70, "BC1_SRGB",      8, True,  72, "BC1_UNORM_SRGB",      None,     ("R", "G", "B", "A"), "bc", "C", 0),
    (71, "BC3_SRGB",     16, True,  78, "BC3_UNORM_SRGB",      None,     ("R", "G", "B", "A"), "bc", "C", 0),
    (73, "BC7_SRGB",     16, True,  99, "BC7_UNORM_SRGB",      None,     ("R", "G", "B", "A"), "bc", "C", 0),
    (74, "RGBA8_SRGB",    4, False, 29, "R8G8B8A8_UNORM_SRGB", None,     ("R", "G", "B", "A"), "unorm", "C", 0),
]

# (A) IL::IsSrgb +0x48DC0: sRGB id → linear id with the same layout
SRGB_PAIRS = {33: 32, 70: 59, 71: 61, 73: 68, 74: 38}

UNKNOWN_ID = 255   # IL::Format "unknown" sentinel (A: ToStr)

FORMATS: dict[int, ILFormat] = {}
for _r in _ROWS:
    _id = _r[0]
    FORMATS[_id] = ILFormat(*_r, srgb=_id in SRGB_PAIRS, linear_id=SRGB_PAIRS.get(_id),
                            srgb_id=next((s for s, l in SRGB_PAIRS.items() if l == _id), None))
del _r, _id

NAME_TO_ID: dict[str, int] = {f.name: f.id for f in FORMATS.values()}

# ids present in the shipped corpus (census 2026-09-10; survey §3.1)
OBSERVED_IDS = tuple(sorted(f.id for f in FORMATS.values() if f.corpus > 0))
# ids whose DXGI mapping was read from the native DDS writer (tier A), observed or decoder-only
NATIVE_MAPPED_IDS = tuple(sorted(f.id for f in FORMATS.values() if f.tier == "A"))

DXGI_TO_IL: dict[int, int] = {}
for _f in FORMATS.values():
    if _f.dxgi is not None and _f.dxgi not in DXGI_TO_IL:
        DXGI_TO_IL[_f.dxgi] = _f.id
del _f

# legacy DDS FourCC → IL id. The native writer emits DX10 for everything the tool touches; these are the codes
# DirectXTex / texconv and older tools emit for the same layouts (accepted on import only).
FOURCC_TO_IL: dict[bytes, int] = {
    b"DXT1": 59, b"DXT3": 60, b"DXT5": 61,
    b"ATI1": 63, b"BC4U": 63, b"BC4S": 62,
    b"ATI2": 65, b"BC5U": 65, b"BC5S": 64,
}
# D3DFMT numeric FourCC codes (DirectXTex legacy) for uncompressed layouts
D3DFMT_TO_IL: dict[int, int] = {
    36: 47,    # D3DFMT_A16B16G16R16   → RGBA16
    110: 48,   # D3DFMT_Q16W16V16U16   → RGBA16_SNORM
    111: 6,    # D3DFMT_R16F           → R16F
    112: 19,   # D3DFMT_G16R16F        → RG16F
    113: 46,   # D3DFMT_A16B16G16R16F  → RGBA16F
    114: 12,   # D3DFMT_R32F           → R32F
    115: 24,   # D3DFMT_G32R32F        → RG32F
    116: 51,   # D3DFMT_A32B32G32R32F  → RGBA32F
}

# Uncompressed legacy DDS pixel formats that identify an IL layout unambiguously:
# (pf flags mask relevant bits, bit count, R, G, B, A masks) → IL id
DDPF_ALPHAPIXELS = 0x1
DDPF_ALPHA = 0x2
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40
DDPF_LUMINANCE = 0x20000
MASKS_TO_IL: dict[tuple[int, int, int, int, int], int] = {
    # (bitcount, rmask, gmask, bmask, amask)
    (32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000): 32,   # A8R8G8B8 = bytes B,G,R,A = ARGB8 (native writer masks)
    (32, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000): 38,   # A8B8G8R8 = bytes R,G,B,A = RGBA8
    (32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000): 34,   # X8R8G8B8 = XRGB8
    (16, 0x000000FF, 0x0000FF00, 0x00000000, 0x00000000): 15,   # 8:8 R,G                = RG8
    (8,  0x000000FF, 0x00000000, 0x00000000, 0x00000000): 0,    # L8 / R8
    (16, 0x0000FFFF, 0x00000000, 0x00000000, 0x00000000): 7,    # L16 / R16
    (24, 0x00FF0000, 0x0000FF00, 0x000000FF, 0x00000000): 29,   # R8G8B8 (D3D) = bytes B,G,R = BGR8
    (16, 0x0000F800, 0x000007E0, 0x0000001F, 0x00000000): 27,   # R5G6B5
}


def get(il_id: int) -> ILFormat:
    f = FORMATS.get(il_id)
    if f is None:
        raise KeyError(f"IL format id {il_id} is not in the native enum (holes: 4, 5, 11, 43, 72; unknown = 255)")
    return f


def name_of(il_id: int) -> str:
    f = FORMATS.get(il_id)
    return f.name if f else f"IL_{il_id}"


def level_bytes(il_id: int, width: int, height: int, depth: int = 1) -> int:
    """Tightly packed bytes of one mip level (one face) - survey section 4: block ? ceil(w/4)*ceil(h/4)*d*unit : w*h*d*unit."""
    f = get(il_id)
    if f.block:
        return ((width + 3) // 4) * ((height + 3) // 4) * depth * f.unit
    return width * height * depth * f.unit


def dxgi_for(il_id: int) -> int:
    f = get(il_id)
    if f.dxgi is None:
        raise KeyError(f"IL format {f.id} ({f.name}) has no unambiguous DXGI equivalent")
    return f.dxgi
