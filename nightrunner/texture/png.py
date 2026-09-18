"""Preview decoding (IMGC level → RGBA8) and PNG → IMGC import.

Preview decoders (inspection only — none of this is a data path for building packs):
* BC1/BC2/BC3/BC6H/BC7 → Pillow's 'bcn' decoder (DdsImagePlugin). 8-bit output, HDR (BC6H) clipped to [0,1].
* BC4/BC5 (unsigned and SNORM) → own vectorised numpy decoder below (`bc4_channels`), float output. This one is
  exact (integer palette math per the D3D spec) and is also used to recompute IMGC statistics on import.
* Uncompressed → numpy dtype table (`UNCOMPRESSED`), storage channel order swizzled to RGBA for the preview.
Signed values are shown as (v+1)/2, floats clipped to [0,1], integer (UINT/SINT) formats shown raw (8-bit).

PNG import (`png_to_imgc`) follows the DyingLightExplorer prepare_rgba path that was confirmed in game
(survey §6.2, tier B+): RGBA8 (IL 38), R8 (IL 0) or RG8_SNORM (IL 16, normal map: XY = rgb/127.5−1 renormalised,
packed as int8·127), full mip chain down to 1×1 with a stored-space BOX filter (no sRGB awareness — hole
§10.2), statistics = per-channel min/max/mean of the mip-0 float values, header fields that cannot be derived
(flags, version, header_size, extension, reserved, mip_split) copied from a template header.
"""

from __future__ import annotations

import numpy as np

from ..errors import UnsupportedError, ValidationError
from . import formats
from .imgc import ImgcHeader, TYPE_2D, LEVEL_PADDING_STOCK, level_layout, join_payload

# IL id → (numpy dtype, channel count, kind, swizzle from storage order to RGBA order or None)
# kinds: unorm (÷max), snorm (÷max, clamp −1), float (as is), uint/sint (raw integers)
UNCOMPRESSED: dict[int, tuple[str, int, str, tuple[int, ...] | None]] = {
    0: ("u1", 1, "unorm", None), 1: ("i1", 1, "snorm", None), 2: ("u1", 1, "uint", None), 3: ("i1", 1, "sint", None),
    6: ("<f2", 1, "float", None), 7: ("<u2", 1, "unorm", None), 8: ("<i2", 1, "snorm", None),
    9: ("<u2", 1, "uint", None), 10: ("<i2", 1, "sint", None),
    12: ("<f4", 1, "float", None), 13: ("<u4", 1, "uint", None), 14: ("<i4", 1, "sint", None),
    15: ("u1", 2, "unorm", None), 16: ("i1", 2, "snorm", None), 17: ("u1", 2, "uint", None), 18: ("i1", 2, "sint", None),
    19: ("<f2", 2, "float", None), 20: ("<u2", 2, "unorm", None), 21: ("<i2", 2, "snorm", None),
    22: ("<u2", 2, "uint", None), 23: ("<i2", 2, "sint", None),
    24: ("<f4", 2, "float", None), 25: ("<u4", 2, "uint", None), 26: ("<i4", 2, "sint", None),
    28: ("u1", 3, "unorm", None), 29: ("u1", 3, "unorm", (2, 1, 0)), 31: ("<f4", 3, "float", (2, 1, 0)),
    32: ("u1", 4, "unorm", (2, 1, 0, 3)), 33: ("u1", 4, "unorm", (2, 1, 0, 3)), 34: ("u1", 4, "unorm", (2, 1, 0, 3)),
    38: ("u1", 4, "unorm", None), 39: ("i1", 4, "snorm", None), 40: ("u1", 4, "uint", None), 41: ("i1", 4, "sint", None),
    46: ("<f2", 4, "float", None), 47: ("<u2", 4, "unorm", None), 48: ("<i2", 4, "snorm", None),
    49: ("<u2", 4, "uint", None), 50: ("<i2", 4, "sint", None),
    51: ("<f4", 4, "float", None), 52: ("<u4", 4, "uint", None), 53: ("<i4", 4, "sint", None),
    69: ("u1", 1, "unorm", None), 74: ("u1", 4, "unorm", None),
}

PILLOW_BCN = {59: (1, "BC1", "RGBA"), 60: (2, "BC2", "RGBA"), 61: (3, "BC3", "RGBA"),
              66: (6, "BC6H", "RGB"), 67: (6, "BC6HS", "RGB"), 68: (7, "BC7", "RGBA"),
              70: (1, "BC1", "RGBA"), 71: (3, "BC3", "RGBA"), 73: (7, "BC7", "RGBA")}
BC45 = {62: (1, True), 63: (1, False), 64: (2, True), 65: (2, False)}   # id → (channels, signed)

PREVIEW_NOTE = "8-bit preview; signed channels remapped [-1,1] -> [0,1]; HDR clipped to [0,1]; BC1/2/3/6H/7 via Pillow"


# --------------------------------------------------------------------------------------------------------------
# BC4 / BC5 (own decoder, exact)
# --------------------------------------------------------------------------------------------------------------

def bc4_channels(raw, width: int, height: int, channels: int, signed: bool) -> np.ndarray:
    """Decode BC4 (channels=1) or BC5 (channels=2, blocks stored [R-block][G-block]) to float32 (h, w, channels).
    Unsigned → [0,1]; signed → [-1,1] (−128 clamps to −1 per the D3D11 spec)."""
    bw, bh = (width + 3) // 4, (height + 3) // 4
    blocks = np.frombuffer(raw, np.uint8)
    if len(blocks) != bw * bh * channels * 8:
        raise ValidationError(f"BC4/5 data is {len(blocks)} bytes, expected {bw * bh * channels * 8} for {width}x{height}")
    blocks = blocks.reshape(-1, channels, 8)
    ends = blocks[:, :, :2].astype(np.float32)
    if signed:
        ends = np.maximum(np.where(ends > 127, ends - 256, ends) / 127.0, -1.0)
    else:
        ends = ends / 255.0
    a, b = ends[:, :, 0], ends[:, :, 1]
    palette = np.empty(a.shape + (8,), np.float32)
    palette[..., 0] = a
    palette[..., 1] = b
    for i in range(2, 8):
        seven = ((8 - i) * a + (i - 1) * b) / 7.0
        if i < 6:
            five = ((6 - i) * a + (i - 1) * b) / 5.0
        elif i == 6:
            five = np.full_like(a, -1.0 if signed else 0.0)
        else:
            five = np.full_like(a, 1.0)
        palette[..., i] = np.where(a > b, seven, five)
    bits = np.zeros(a.shape, np.uint64)
    for i in range(6):
        bits |= blocks[:, :, i + 2].astype(np.uint64) << np.uint64(8 * i)
    codes = ((bits[..., None] >> (np.arange(16, dtype=np.uint64) * np.uint64(3))) & np.uint64(7)).astype(np.intp)
    values = np.take_along_axis(palette, codes, axis=2)          # (blocks, channels, 16)
    img = values.reshape(bh, bw, channels, 4, 4).transpose(0, 3, 1, 4, 2).reshape(bh * 4, bw * 4, channels)
    return np.ascontiguousarray(img[:height, :width])


# --------------------------------------------------------------------------------------------------------------
# Level → float values / RGBA8 preview
# --------------------------------------------------------------------------------------------------------------

def decode_level(raw, width: int, height: int, il_format: int) -> tuple[np.ndarray, str]:
    """Decode one 2D surface to float32 (h, w, c) in RGBA channel order and native value range.
    Returns (values, decoder) where decoder ∈ {'pillow-bcn', 'bc45', 'uncompressed'}; 'pillow-bcn' values are
    8-bit-quantised previews (never a data path)."""
    if il_format in PILLOW_BCN:
        from PIL import Image
        n, pf, mode = PILLOW_BCN[il_format]
        im = Image.frombytes(mode, (width, height), bytes(raw), "bcn", (n, pf))
        arr = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
        return arr[:, :, : (3 if mode == "RGB" else 4)], "pillow-bcn"
    if il_format in BC45:
        ch, signed = BC45[il_format]
        return bc4_channels(raw, width, height, ch, signed), "bc45"
    spec = UNCOMPRESSED.get(il_format)
    if spec is None:
        raise UnsupportedError(f"no preview decoder for IL {il_format} ({formats.name_of(il_format)})")
    dtype, ch, kind, swz = spec
    arr = np.frombuffer(raw, dtype)
    if arr.size != width * height * ch:
        raise ValidationError(f"level data is {len(raw)} bytes, expected {width * height * ch * np.dtype(dtype).itemsize}")
    arr = arr.reshape(height, width, ch)
    if swz:
        arr = arr[:, :, list(swz)]
    if kind == "unorm":
        vals = arr.astype(np.float32) / float(np.iinfo(dtype).max)
    elif kind == "snorm":
        vals = np.maximum(arr.astype(np.float32) / float(np.iinfo(dtype).max), -1.0)
    elif kind == "float":
        vals = np.nan_to_num(arr.astype(np.float32))
    else:  # uint / sint: raw integers
        vals = arr.astype(np.float32)
    return np.ascontiguousarray(vals), "uncompressed"


def to_preview_rgba8(values: np.ndarray, il_format: int) -> np.ndarray:
    """float (h, w, c) -> uint8 (h, w, 4): signed formats (v+1)/2, floats/unorm clipped, integers raw."""
    f = formats.get(il_format)
    kind = f.kind
    v = values
    if kind == "snorm" or il_format in (62, 64, 67):
        v = (v + 1.0) / 2.0
    if kind in ("uint", "sint"):
        out = np.clip(v, 0, 255).astype(np.uint8)
    else:
        out = np.rint(np.clip(v, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w, c = out.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[:, :, 3] = 255
    rgba[:, :, :min(c, 4)] = out[:, :, :4]
    return rgba


def decode_preview(raw, width: int, height: int, il_format: int) -> np.ndarray:
    values, _ = decode_level(raw, width, height, il_format)
    return to_preview_rgba8(values, il_format)


def png_bytes(rgba8: np.ndarray) -> bytes:
    import io
    from PIL import Image
    out = io.BytesIO()
    Image.fromarray(rgba8, "RGBA").save(out, format="PNG")
    return out.getvalue()


# --------------------------------------------------------------------------------------------------------------
# Statistics (float4 min/max/mean at +0x10/+0x20/+0x30)
# --------------------------------------------------------------------------------------------------------------

def compute_stats(stored: np.ndarray, source01: np.ndarray | None = None) -> tuple[list[float], list[float], list[float]]:
    """The three float4 statistics, following the stock-corpus convention (notes/FORMATS/imgc.md §5, census
    2026-09-15): vec0/vec1 = per-channel min/max of the *stored* channels in their stored numeric range (SNORM
    negative, HDR unbounded), 0.0 for channels the format does not store; vec2 = per-channel mean of the *source*
    image in [0,1] (RGBA, 0/0/0/1 fill). `stored` is (…, c) float, `source01` (…, ≤4) float in [0,1]; when
    source01 is None the stored values are mapped to [0,1] ((v+1)/2 for negatives, clip) for vec2."""
    v = stored.reshape(-1, stored.shape[-1]).astype(np.float64)
    c = v.shape[1]
    mn = [float(x) for x in v.min(axis=0)] + [0.0] * (4 - c)
    mx = [float(x) for x in v.max(axis=0)] + [0.0] * (4 - c)
    if source01 is None:
        s = v
        if float(v.min()) < 0.0:
            s = (v + 1.0) / 2.0
        s = np.clip(s, 0.0, 1.0)
    else:
        s = source01.reshape(-1, source01.shape[-1]).astype(np.float64)
    cs = s.shape[1]
    me = [float(x) for x in s.mean(axis=0)] + [0.0, 0.0, 0.0, 1.0][cs:4]
    return mn[:4], mx[:4], me[:4]


# --------------------------------------------------------------------------------------------------------------
# PNG → IMGC
# --------------------------------------------------------------------------------------------------------------

IMPORT_FORMATS = {38: "RGBA8", 0: "R8", 16: "RG8_SNORM"}


def _full_chain(width: int, height: int) -> int:
    return max(width, height).bit_length()


def png_to_imgc(image, il_format: int, template: ImgcHeader, *, mip_count: int | None = None,
                max_side: int = 16384, level_padding: int = LEVEL_PADDING_STOCK) -> tuple[ImgcHeader, bytes, dict]:
    """Build (header, bitmap) for a 2D texture from a PIL image (or (h, w, 4) uint8 array).

    il_format: 38 RGBA8 | 0 R8 | 16 RG8_SNORM (normal map, XY from RGB).
    mip_count: None = full chain to 1×1 (max(w,h).bit_length()), 1 = no mips, else that many (≤ full chain).
    level_padding: 16 (stock stride) or 0 (third-party tight layout, imgc.py) — the sidecar value on rebuilds.
    Returns (header, bitmap_bytes, info) with info = {"mips", "stats", ...}.
    """
    from PIL import Image
    if il_format not in IMPORT_FORMATS:
        raise UnsupportedError(f"PNG import produces IL 38 RGBA8, 0 R8 or 16 RG8_SNORM; IL {il_format} "
                               f"({formats.name_of(il_format)}) would need a BC encoder (out of scope)")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    w, h = image.size
    if not (1 <= w <= max_side and 1 <= h <= max_side):
        raise ValidationError(f"image {w}x{h}: sides must be 1..{max_side}")
    if w > 0xFFFF or h > 0xFFFF:
        raise ValidationError(f"image {w}x{h} exceeds the IMGC u16 dimensions")
    full = _full_chain(w, h)
    mips = full if mip_count is None else int(mip_count)
    if not 1 <= mips <= full:
        raise ValidationError(f"mip_count {mips} must be 1..{full} for {w}x{h}")

    header = ImgcHeader(version=template.version, header_size=template.header_size, flags=template.flags,
                        stats_raw=template.stats_raw, reserved=template.reserved, mip_split=template.mip_split,
                        extension=template.extension, tail=template.tail, magic=template.magic)
    if header.header_only:
        raise ValidationError("template header has flag 0x02 (no bitmap); pick a texture with texels as template")
    header.set_geometry(w, h, 1, TYPE_2D, mips, il_format)

    src = np.asarray(image, dtype=np.float32) / 255.0       # (h, w, 4) in [0,1]

    def encode(mip_img):
        """Level bytes + the stored channel values (float, stored range) for the statistics."""
        if il_format == 38:
            arr = np.asarray(mip_img, dtype=np.uint8)
            return arr.tobytes(), arr.astype(np.float32) / 255.0
        if il_format == 0:
            arr = np.ascontiguousarray(np.asarray(mip_img, dtype=np.uint8)[:, :, 0])
            return arr.tobytes(), arr[:, :, None].astype(np.float32) / 255.0
        nv = np.asarray(mip_img, dtype=np.float32)[:, :, :3] / 127.5 - 1.0
        nv /= np.maximum(np.linalg.norm(nv, axis=2, keepdims=True), 1e-12)
        q = np.rint(np.clip(nv[:, :, :2], -1.0, 1.0) * 127.0).astype(np.int8)
        return q.tobytes(), q.astype(np.float32) / 127.0

    levels = []
    stats = None
    for lv in level_layout(header):
        mip = image if (lv.width, lv.height) == (w, h) else image.resize((lv.width, lv.height), Image.Resampling.BOX)
        data, stored = encode(mip)
        levels.append(data)
        if stats is None:
            stats = compute_stats(stored, src)      # vec0/vec1 from the stored mip-0 channels, vec2 from the source
    header.set_stats(*stats)
    bitmap = join_payload(header, levels, level_padding)
    info = {"width": w, "height": h, "mip_count": mips, "format": il_format, "format_name": formats.name_of(il_format),
            "stats": {"minimum": stats[0], "maximum": stats[1], "mean": stats[2]}, "filter": "BOX (stored space)"}
    return header, bitmap, info
