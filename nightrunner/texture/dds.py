"""DDS reader/writer for IMGC textures.

Export always writes a DX10 header (148 bytes, exact layout below — identical to the DyingLightExplorer writer,
survey §5.1 (B), so files from either tool compare byte-for-byte). Import accepts DX10 headers, legacy FourCC
(DXT1/DXT3/DXT5/ATI1/BC4U/BC4S/ATI2/BC5U/BC5S and the numeric D3DFMT codes) and legacy uncompressed pixel
formats whose bit masks identify exactly one IL layout (formats.MASKS_TO_IL). Texel data must be tightly packed
(no row alignment) and exactly Σ level sizes long — anything else is refused, not repaired.

DDS header we emit (offsets in bytes)
-------------------------------------
    0    4   "DDS "
    4    4   dwSize = 124
    8    4   dwFlags = 0x1007 (CAPS|HEIGHT|WIDTH|PIXELFORMAT) | (block ? 0x80000 LINEARSIZE : 0x8 PITCH)
                        | (mips > 1 ? 0x20000 MIPMAPCOUNT : 0) | (volume ? 0x800000 DEPTH : 0)
    12   4   dwHeight
    16   4   dwWidth
    20   4   dwPitchOrLinearSize = block ? bytes of (mip 0, face 0) : width * unit
    24   4   dwDepth = volume ? depth : 0
    28   4   dwMipMapCount = mip_count (written even when 1)
    32   44  dwReserved1[11] = 0
    76   32  DDS_PIXELFORMAT { dwSize 32, dwFlags 0x4 FOURCC, dwFourCC "DX10", bitcount 0, masks 0 }
    108  4   dwCaps = 0x1000 TEXTURE | (mips > 1 ? 0x400008 MIPMAP|COMPLEX : 0) | (cube or volume ? 0x8 COMPLEX : 0)
    112  4   dwCaps2 = cube ? 0xFE00 (CUBEMAP + all six faces) : volume ? 0x200000 VOLUME : 0
    116  12  dwCaps3, dwCaps4, dwReserved2 = 0
    128  4   DX10.dxgiFormat        (formats.py, tier A for the corpus ids)
    132  4   DX10.resourceDimension = volume ? 4 (TEXTURE3D) : 3 (TEXTURE2D)   (cube = 2D + miscFlag)
    136  4   DX10.miscFlag          = cube ? 0x4 (TEXTURECUBE) : 0
    140  4   DX10.arraySize         = 1
    144  4   DX10.miscFlags2        = 0 (alpha mode unknown)
    148  …   texels, FACE-MAJOR: for face in 0..5: for mip in 0..n-1  (volumes: slices consecutive per mip),
             16-byte IMGC level padding removed
IMGC stores levels MIP-major (imgc.level_layout); cube faces are reordered on both paths.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ..errors import FormatError, UnsupportedError, ValidationError
from . import formats
from .imgc import ImgcHeader, Level, TYPE_2D, TYPE_CUBE, TYPE_VOLUME, level_layout, check_payload

MAGIC = b"DDS "
HEADER_SIZE = 124
PF_SIZE = 32
LEGACY_DATA_OFFSET = 4 + HEADER_SIZE            # 128
DX10_DATA_OFFSET = LEGACY_DATA_OFFSET + 20      # 148
FOURCC_DX10 = b"DX10"

DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PITCH = 0x1, 0x2, 0x4, 0x8
DDSD_PIXELFORMAT, DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE, DDSD_DEPTH = 0x1000, 0x20000, 0x80000, 0x800000
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000
DDSCAPS2_CUBEMAP, DDSCAPS2_ALLFACES, DDSCAPS2_VOLUME = 0x200, 0xFC00, 0x200000
DX10_DIMENSION_1D, DX10_DIMENSION_2D, DX10_DIMENSION_3D = 2, 3, 4
DX10_MISC_TEXTURECUBE = 0x4

_HDR = struct.Struct("<4s7I44s")          # magic, size, flags, height, width, pitch, depth, mips, reserved1
_PF = struct.Struct("<II4sIIIII")         # size, flags, fourcc, bitcount, r, g, b, a
_CAPS = struct.Struct("<5I")              # caps, caps2, caps3, caps4, reserved2
_DX10 = struct.Struct("<5I")              # dxgi, dimension, misc, arraySize, misc2
assert _HDR.size + _PF.size + _CAPS.size == LEGACY_DATA_OFFSET


@dataclass
class DdsFile:
    """Parsed DDS header, normalised to IMGC terms."""
    width: int
    height: int
    depth: int
    mip_count: int
    tex_type: int                   # imgc.TYPE_*
    il_format: int
    dxgi: int | None
    data_offset: int
    identified_by: str              # "dx10" | "fourcc" | "d3dfmt" | "masks"
    array_size: int = 1
    alpha_mode: int = 0
    flags: int = 0
    caps: int = 0
    caps2: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def faces(self) -> int:
        return 6 if self.tex_type == TYPE_CUBE else 1

    def to_json(self) -> dict:
        f = formats.FORMATS.get(self.il_format)
        return {"width": self.width, "height": self.height, "depth": self.depth, "mip_count": self.mip_count,
                "type": self.tex_type, "il_format": self.il_format, "format_name": f.name if f else None,
                "dxgi": self.dxgi, "identified_by": self.identified_by, "data_offset": self.data_offset,
                "array_size": self.array_size, "alpha_mode": self.alpha_mode, "warnings": self.warnings}


# --------------------------------------------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------------------------------------------

def dds_header_for(h: ImgcHeader, levels: list[Level] | None = None) -> bytes:
    """The 148-byte DX10 header for an IMGC header (layout documented at the top of this module)."""
    f = formats.get(h.format)
    if f.dxgi is None:
        raise UnsupportedError(f"IL format {f.id} ({f.name}) has no DXGI equivalent; DDS export refused")
    if h.header_only:
        raise UnsupportedError("header-only IMGC record (flag 0x02) has no texels to export")
    levels = levels or level_layout(h)
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | (DDSD_LINEARSIZE if f.block else DDSD_PITCH)
    caps = DDSCAPS_TEXTURE
    caps2 = 0
    if h.mip_count > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps |= DDSCAPS_MIPMAP | DDSCAPS_COMPLEX
    if h.is_cube:
        caps |= DDSCAPS_COMPLEX
        caps2 = DDSCAPS2_CUBEMAP | DDSCAPS2_ALLFACES
    if h.is_volume:
        flags |= DDSD_DEPTH
        caps |= DDSCAPS_COMPLEX
        caps2 = DDSCAPS2_VOLUME
    pitch = levels[0].size if f.block else h.width * f.unit
    out = _HDR.pack(MAGIC, HEADER_SIZE, flags, h.height, h.width, pitch, h.depth if h.is_volume else 0,
                    h.mip_count, bytes(44))
    out += _PF.pack(PF_SIZE, formats.DDPF_FOURCC, FOURCC_DX10, 0, 0, 0, 0, 0)
    out += _CAPS.pack(caps, caps2, 0, 0, 0)
    out += _DX10.pack(f.dxgi, DX10_DIMENSION_3D if h.is_volume else DX10_DIMENSION_2D,
                      DX10_MISC_TEXTURECUBE if h.is_cube else 0, 1, 0)
    assert len(out) == DX10_DATA_OFFSET
    return out


def face_major(levels: list[Level]) -> list[Level]:
    """IMGC mip-major -> DDS face-major order."""
    return sorted(levels, key=lambda lv: (lv.face, lv.mip))


def iter_dds(h: ImgcHeader, bitmap, *, chunk: int = 1 << 20):
    """Yield the DDS file (header, then tight levels in DDS order) as memoryview/bytes chunks. Never copies the
    whole bitmap; safe on the mmapped 29 GB pack."""
    mv = memoryview(bitmap)
    levels = check_payload(h, len(mv))
    yield dds_header_for(h, levels)
    for lv in face_major(levels):
        for off in range(lv.offset, lv.offset + lv.size, chunk):
            yield mv[off: min(off + chunk, lv.offset + lv.size)]


def imgc_to_dds(h: ImgcHeader, bitmap) -> bytes:
    return b"".join(bytes(c) for c in iter_dds(h, bitmap, chunk=1 << 30))


def dds_size_for(h: ImgcHeader) -> int:
    levels = level_layout(h)
    return DX10_DATA_OFFSET + sum(lv.size for lv in levels)


# --------------------------------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------------------------------

def _identify_legacy(pf_flags: int, fourcc: bytes, bitcount: int, masks: tuple[int, int, int, int]) -> tuple[int, str]:
    if pf_flags & formats.DDPF_FOURCC:
        if fourcc in formats.FOURCC_TO_IL:
            return formats.FOURCC_TO_IL[fourcc], "fourcc"
        code = struct.unpack("<I", fourcc)[0]
        if code in formats.D3DFMT_TO_IL:
            return formats.D3DFMT_TO_IL[code], "d3dfmt"
        if fourcc in (b"DXT2", b"DXT4"):
            raise UnsupportedError(f"DDS FourCC {fourcc!r} is premultiplied-alpha BC2/BC3; the engine has no such format")
        raise UnsupportedError(f"DDS FourCC {fourcc!r} / D3DFMT {code} is not a known IL layout")
    if pf_flags & (formats.DDPF_RGB | formats.DDPF_LUMINANCE | formats.DDPF_ALPHA):
        key = (bitcount,) + tuple(masks)
        il = formats.MASKS_TO_IL.get(key)
        if il is None:
            raise UnsupportedError(f"DDS legacy pixel format (bits={bitcount}, masks R=0x{masks[0]:08X} G=0x{masks[1]:08X} "
                                   f"B=0x{masks[2]:08X} A=0x{masks[3]:08X}) does not identify an IL layout; save as DX10")
        return il, "masks"
    raise UnsupportedError(f"DDS pixel format flags 0x{pf_flags:X} carry neither FourCC nor RGB masks")


def read_dds(buf, *, srgb_to_linear: bool = False, allow_unobserved: bool = False) -> DdsFile:
    """Parse and validate a DDS file held in *buf* (bytes/memoryview). Raises FormatError for framing errors,
    UnsupportedError for layouts the engine has no IL id for, ValidationError when the texel length is wrong.

    srgb_to_linear: map *_SRGB DXGI formats onto the linear IL id with the same payload (BC7_UNORM_SRGB → BC7).
    allow_unobserved: accept IL ids that never occur in the shipped corpus (tier C) instead of refusing.
    """
    mv = memoryview(buf)
    if len(mv) < LEGACY_DATA_OFFSET:
        raise FormatError(f"DDS: {len(mv)} bytes, shorter than the 128-byte header")
    magic, size, flags, height, width, pitch, depth, mips, _res = _HDR.unpack_from(mv, 0)
    if magic != MAGIC:
        raise FormatError(f"DDS: bad magic {magic!r}")
    if size != HEADER_SIZE:
        raise FormatError(f"DDS: dwSize {size} != 124")
    pf_size, pf_flags, fourcc, bitcount, rm, gm, bm, am = _PF.unpack_from(mv, _HDR.size)
    if pf_size != PF_SIZE:
        raise FormatError(f"DDS: pixel format dwSize {pf_size} != 32")
    caps, caps2, _c3, _c4, _r2 = _CAPS.unpack_from(mv, _HDR.size + _PF.size)
    warnings: list[str] = []

    mip_count = max(1, mips) if flags & DDSD_MIPMAPCOUNT else 1
    if mips > 1 and not flags & DDSD_MIPMAPCOUNT:
        # some writers set the count without the flag; DirectXTex honours the count → follow it but say so
        mip_count = mips
        warnings.append(f"dwMipMapCount={mips} without DDSD_MIPMAPCOUNT; using it")
    is_cube = bool(caps2 & DDSCAPS2_CUBEMAP)
    is_volume = bool(caps2 & DDSCAPS2_VOLUME) or (bool(flags & DDSD_DEPTH) and depth > 1)
    data_offset = LEGACY_DATA_OFFSET
    dxgi = None
    array_size = 1
    alpha_mode = 0

    if pf_flags & formats.DDPF_FOURCC and fourcc == FOURCC_DX10:
        if len(mv) < DX10_DATA_OFFSET:
            raise FormatError("DDS: truncated DX10 header")
        dxgi, dimension, misc, array_size, misc2 = _DX10.unpack_from(mv, LEGACY_DATA_OFFSET)
        data_offset = DX10_DATA_OFFSET
        if dimension == DX10_DIMENSION_3D:
            is_volume = True
        elif dimension == DX10_DIMENSION_2D:
            pass
        elif dimension == DX10_DIMENSION_1D:
            raise UnsupportedError("DDS: 1D textures (DX10 resourceDimension 2) have no IMGC type")
        else:
            raise FormatError(f"DDS: DX10 resourceDimension {dimension} is invalid")
        if misc & DX10_MISC_TEXTURECUBE:
            is_cube = True
        if array_size != 1:
            raise UnsupportedError(f"DDS: arraySize {array_size}; IMGC has no texture-array field (survey section 5.1)")
        alpha_mode = misc2 & 0x7
        if alpha_mode > 4:
            raise FormatError(f"DDS: DX10 miscFlags2 alpha mode {alpha_mode} is invalid")
        il = formats.DXGI_TO_IL.get(dxgi)
        if il is None:
            raise UnsupportedError(f"DDS: DXGI format {dxgi} has no IL equivalent (notes/FORMATS/dds-mapping.md)")
        f = formats.FORMATS[il]
        if f.srgb:
            if srgb_to_linear:
                warnings.append(f"DXGI {dxgi} {f.dxgi_name} mapped to linear IL {f.linear_id} "
                                f"{formats.name_of(f.linear_id)} (payload identical; srgb_to_linear)")
                il = f.linear_id
            elif not allow_unobserved:
                raise UnsupportedError(
                    f"DDS: DXGI {dxgi} {f.dxgi_name} -> IL {f.id} {f.name}: no sRGB IL id occurs in the shipped corpus "
                    f"(survey section 3.1). Re-export as {formats.name_of(f.linear_id)} (DXGI {formats.FORMATS[f.linear_id].dxgi}) "
                    f"or set import.srgb_to_linear / import.allow_unobserved in the sidecar")
        identified = "dx10"
    else:
        il, identified = _identify_legacy(pf_flags, fourcc, bitcount, (rm, gm, bm, am))
        dxgi = formats.FORMATS[il].dxgi

    f = formats.FORMATS[il]
    if f.tier != "A" and not allow_unobserved:
        raise UnsupportedError(f"DDS: IL {f.id} {f.name} never occurs in the shipped corpus (tier C mapping); "
                               f"set import.allow_unobserved to import it anyway")
    if is_cube and is_volume:
        raise FormatError("DDS: both cube and volume flags set")
    if is_cube:
        faces_bits = caps2 & DDSCAPS2_ALLFACES
        # legacy headers must list all six faces; DX10 writers may leave caps2 faces empty (miscFlag is authoritative)
        if faces_bits != DDSCAPS2_ALLFACES and (data_offset == LEGACY_DATA_OFFSET or faces_bits != 0):
            raise UnsupportedError(f"DDS: partial cube map (caps2 0x{caps2:X}); IMGC cubes carry all six faces")
        if depth not in (0, 1):
            raise FormatError(f"DDS: cube with depth {depth}")
        tex_type, depth = TYPE_CUBE, 1
    elif is_volume:
        if depth < 1:
            raise FormatError("DDS: volume with depth 0")
        if depth > 255:
            raise UnsupportedError(f"DDS: volume depth {depth} exceeds the IMGC u8 depth field")
        tex_type = TYPE_VOLUME
    else:
        if depth > 1:
            raise FormatError(f"DDS: dwDepth {depth} on a non-volume texture")
        tex_type, depth = TYPE_2D, 1
    if width < 1 or height < 1:
        raise FormatError(f"DDS: {width}x{height}")
    if width > 0xFFFF or height > 0xFFFF:
        raise UnsupportedError(f"DDS: {width}x{height} exceeds the IMGC u16 dimension fields")
    if mip_count > 63:
        raise UnsupportedError(f"DDS: {mip_count} mips exceed the IMGC 6-bit mip count")

    d = DdsFile(width, height, depth, mip_count, tex_type, il, dxgi, data_offset, identified,
                array_size, alpha_mode, flags, caps, caps2, warnings)
    expected = data_offset + sum(lv.size for lv in dds_levels(d))
    if len(mv) != expected:
        raise ValidationError(f"DDS: file is {len(mv)} bytes but a tightly packed {width}x{height}x{depth} {f.name} "
                              f"{'cube ' if is_cube else ''}with {mip_count} mips needs exactly {expected} "
                              f"(data at {data_offset}); row-padded or truncated files are refused")
    return d


def dds_levels(d: DdsFile) -> list[Level]:
    """Surfaces in DDS file order (face-major) with offsets relative to the texel data start (tight)."""
    h = ImgcHeader()
    h.set_geometry(d.width, d.height, d.depth, d.tex_type, d.mip_count, d.il_format)
    out = []
    off = 0
    for lv in face_major(level_layout(h)):
        out.append(Level(lv.mip, lv.face, lv.width, lv.height, lv.depth, off, lv.size, lv.size))
        off += lv.size
    return out


def dds_to_levels(d: DdsFile, buf) -> list[memoryview]:
    """Tight per-level views of a parsed DDS in IMGC (mip-major) order - feed to imgc.join_payload."""
    mv = memoryview(buf)
    by_key = {}
    for lv in dds_levels(d):
        start = d.data_offset + lv.offset
        by_key[(lv.mip, lv.face)] = mv[start: start + lv.size]
    h = ImgcHeader()
    h.set_geometry(d.width, d.height, d.depth, d.tex_type, d.mip_count, d.il_format)
    return [by_key[(lv.mip, lv.face)] for lv in level_layout(h)]


def apply_dds_geometry(template: ImgcHeader, d: DdsFile) -> ImgcHeader:
    """Copy of *template* with width/height/depth/type/mips/format taken from the DDS. Flags, stats, reserved,
    mip_split, extension and version are untouched (the caller decides about stats)."""
    h = ImgcHeader(version=template.version, header_size=template.header_size, flags=template.flags,
                   stats_raw=template.stats_raw, width=template.width, height=template.height, depth=template.depth,
                   reserved=template.reserved, format=template.format, packed=template.packed,
                   mip_split=template.mip_split, extension=template.extension, tail=template.tail, magic=template.magic)
    h.set_geometry(d.width, d.height, d.depth, d.tex_type, d.mip_count, d.il_format)
    return h
