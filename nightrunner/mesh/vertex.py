"""Vertex formats 0/3/6/8: numpy dtypes, float decode, qtangent maths, exact re-encode, weight quantisation.

Layouts (survey 02 §5; strides 40/80 (A) renderer CreateSurfaceVF +0x52510, 16/32 (A) corpus; per-attribute
encodings (B) plus the 2026-09-15 census in notes/FORMATS/mesh.md). Field names starting with `raw_` are holes
carried verbatim.

    format 0 (16 B, static, half precision)
        0x00 f16[3] pos | 0x06 u16 raw_06 (census: always 0x3C00 = f16 1.0) | 0x08 i8[4] qtan (÷127) | 0x0C f16[2] uv0
    format 3 (32 B, static)
        0x00 f32[3] pos | 0x0C i16[4] qtan (÷32767) | 0x14 f16[2] uv0 | 0x18 f16[2] uv1 | 0x1C u32 raw_tail
    format 6 (40 B, skinned)
        0x00 f32[3] pos | 0x0C u8[4] weights | 0x10 u8[4] joints (submesh-palette index) | 0x14 i16[4] qtan |
        0x1C f16[2] uv0 | 0x20 f16[2] uv1 | 0x24 u32 raw_tail
    format 8 (80 B, skinned + extension)
        format 6 | 0x28 u8[40] raw_ext

QTangent (A for the handedness rule: native DXBC eye pass; B for the frame): q = (x, y, z, w)/scale normalised;
tangent = R·X, bitangent = R·Y = normal × tangent, normal = R·Z. The shader's bitangent sign is the sign of the
largest-|component| of the *stored* quaternion, ties resolved z > y > x > w. q and −q encode the same rotation, so
the encoder picks the sign of the quad to carry the handedness bit.
"""

from __future__ import annotations

import numpy as np

from ..errors import FormatError, UnsupportedError

VERTEX_DTYPES: dict[int, np.dtype] = {
    0: np.dtype([("pos", "<f2", 3), ("raw_06", "<u2"), ("qtan", "i1", 4), ("uv0", "<f2", 2)]),
    3: np.dtype([("pos", "<f4", 3), ("qtan", "<i2", 4), ("uv0", "<f2", 2), ("uv1", "<f2", 2), ("raw_tail", "<u4")]),
    6: np.dtype([("pos", "<f4", 3), ("weights", "u1", 4), ("joints", "u1", 4), ("qtan", "<i2", 4),
                 ("uv0", "<f2", 2), ("uv1", "<f2", 2), ("raw_tail", "<u4")]),
    8: np.dtype([("pos", "<f4", 3), ("weights", "u1", 4), ("joints", "u1", 4), ("qtan", "<i2", 4),
                 ("uv0", "<f2", 2), ("uv1", "<f2", 2), ("raw_tail", "<u4"), ("raw_ext", "u1", 40)]),
}
STRIDES = {f: dt.itemsize for f, dt in VERTEX_DTYPES.items()}
QTAN_SCALE = {0: 127.0, 3: 32767.0, 6: 32767.0, 8: 32767.0}
SKINNED_FORMATS = (6, 8)          # A: ((format − 6) & 0xFD) == 0 (RM CreateSurface +0xD8D0)
VERTEX_BLOCK_ALIGN = 160          # census: every entry's vertex window is zero-padded to a multiple of 160 bytes
INDEX_BASE_ALIGN = 4              # census: entry index bases are 4-byte aligned
INDEX_BUFFER_ALIGN = 16           # census: the index part is zero-padded to 16 bytes
assert STRIDES == {0: 16, 3: 32, 6: 40, 8: 80}


def align_to(value: int, alignment: int) -> int:
    """Round up to a multiple of *alignment* (any positive integer; util.binio.align_up is power-of-two only)."""
    return (value + alignment - 1) // alignment * alignment


def is_skinned(fmt: int) -> bool:
    return ((fmt - 6) & 0xFD) == 0


def supported(fmt: int) -> bool:
    return fmt in VERTEX_DTYPES


def raw_block(buffer, base: int, count: int, fmt: int) -> np.ndarray:
    """Zero-copy structured view of *count* vertices of format *fmt* at byte *base* of the vertex buffer."""
    if fmt not in VERTEX_DTYPES:
        raise UnsupportedError(f"vertex format {fmt} is not one of 0/3/6/8")
    dt = VERTEX_DTYPES[fmt]
    end = base + count * dt.itemsize
    if base < 0 or end > len(buffer):
        raise FormatError(f"vertex window [{base}, {end}) exceeds the vertex buffer ({len(buffer)} bytes)")
    return np.frombuffer(buffer, dtype=dt, count=count, offset=base)


# ---- qtangent --------------------------------------------------------------------------------------------------

def qtangent_sign(q: np.ndarray) -> np.ndarray:
    """Handedness (+1/−1) of stored quads *q* (N×4, int or float): sign of the largest |component|, ties z>y>x>w."""
    q = np.asarray(q)
    a = np.abs(q.astype(np.float64))
    m = a.max(axis=1)
    idx = np.full(len(q), 3)
    for i in (0, 1, 2):                       # later assignments win ⇒ z beats y beats x beats w on ties
        idx = np.where(a[:, i] == m, i, idx)
    picked = q[np.arange(len(q)), idx]
    return np.where(picked >= 0, 1, -1).astype(np.int8)


def decode_qtangent(qraw: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (tangent, normal, sign) float32 (N×3, N×3, N int8). Normalises q before building the frame."""
    q = qraw.astype(np.float32) / np.float32(scale)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(n, np.float32(1e-12))
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    tangent = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), axis=1)
    normal = np.stack((2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)), axis=1)
    return tangent.astype(np.float32), normal.astype(np.float32), qtangent_sign(qraw)


def quaternion_from_frames(t: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Rotation matrices with columns [t | b | n] → unit quaternions (x, y, z, w), vectorised, branch on the largest
    diagonal term for numerical stability. Inputs are re-orthonormalised (n kept, t projected, b = n × t)."""
    n = n.astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    t = t.astype(np.float64)
    t -= n * np.sum(t * n, axis=1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    b = np.cross(n, t)
    m00, m10, m20 = t[:, 0], t[:, 1], t[:, 2]
    m01, m11, m21 = b[:, 0], b[:, 1], b[:, 2]
    m02, m12, m22 = n[:, 0], n[:, 1], n[:, 2]
    c = np.stack((1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22, 1 + m00 + m11 + m22), axis=1)
    k = c.argmax(axis=1)
    s = np.sqrt(np.maximum(c[np.arange(len(t)), k], 1e-30)) * 2
    q = np.zeros((len(t), 4))
    i = k == 0
    q[i, 0] = s[i] / 4; q[i, 1] = (m01 + m10)[i] / s[i]; q[i, 2] = (m02 + m20)[i] / s[i]; q[i, 3] = (m21 - m12)[i] / s[i]
    i = k == 1
    q[i, 1] = s[i] / 4; q[i, 0] = (m01 + m10)[i] / s[i]; q[i, 2] = (m12 + m21)[i] / s[i]; q[i, 3] = (m02 - m20)[i] / s[i]
    i = k == 2
    q[i, 2] = s[i] / 4; q[i, 0] = (m02 + m20)[i] / s[i]; q[i, 1] = (m12 + m21)[i] / s[i]; q[i, 3] = (m10 - m01)[i] / s[i]
    i = k == 3
    q[i, 3] = s[i] / 4; q[i, 0] = (m21 - m12)[i] / s[i]; q[i, 1] = (m02 - m20)[i] / s[i]; q[i, 2] = (m10 - m01)[i] / s[i]
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return q


def encode_qtangent(tangent: np.ndarray, normal: np.ndarray, sign: np.ndarray, fmt: int) -> np.ndarray:
    """(tangent, normal, sign) → stored quads of the format's integer type, sign bit honoured."""
    scale = QTAN_SCALE[fmt]
    dt = VERTEX_DTYPES[fmt]["qtan"].base
    q = quaternion_from_frames(tangent, np.cross(normal, tangent), normal)
    packed = np.rint(np.clip(q, -1.0, 1.0) * scale).astype(np.int64)
    flip = qtangent_sign(packed) != np.asarray(sign)
    packed[flip] *= -1
    # a zero quad (degenerate frame) has sign +1 by the rule; nothing more can be done for it
    return packed.astype(dt)


def tangents_from_uv(positions: np.ndarray, normals: np.ndarray, uv: np.ndarray, faces: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex tangent + handedness from UV derivatives (the standard per-triangle accumulation; policy for
    Casts that carry no tangent buffer — the official Blender exporter writes none). Returns (tangents N×3 f32
    orthonormalised against *normals*, sign N int8 = sign of det[dUV] accumulated per vertex; +1 when a vertex has
    no non-degenerate triangle, its tangent then being an arbitrary perpendicular of the normal)."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    tan = np.zeros_like(p)
    bit = np.zeros_like(p)
    if len(f):
      with np.errstate(invalid="ignore", divide="ignore"):
          e1 = p[f[:, 1]] - p[f[:, 0]]
          e2 = p[f[:, 2]] - p[f[:, 0]]
          d1 = t[f[:, 1]] - t[f[:, 0]]
          d2 = t[f[:, 2]] - t[f[:, 0]]
          det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
          ok = np.abs(det) > 1e-20
          r = np.zeros(len(f))
          r[ok] = 1.0 / det[ok]
          sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
          tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
          for k in range(3):
              np.add.at(tan, f[:, k], sdir)
              np.add.at(bit, f[:, k], tdir)
    # Gram-Schmidt against the normal
    nn = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    tt = tan - nn * np.sum(tan * nn, axis=1, keepdims=True)
    ln = np.linalg.norm(tt, axis=1)
    bad = ln < 1e-12
    if bad.any():
        # arbitrary perpendicular: cross with the axis least aligned with the normal
        ax = np.zeros((int(bad.sum()), 3))
        idx = np.argmin(np.abs(nn[bad]), axis=1)
        ax[np.arange(len(ax)), idx] = 1.0
        alt = np.cross(nn[bad], ax)
        tt[bad] = alt
        ln = np.linalg.norm(tt, axis=1)
    tt /= np.maximum(ln, 1e-12)[:, None]
    sign = np.where(np.sum(np.cross(nn, tt) * bit, axis=1) < 0, -1, 1).astype(np.int8)
    return tt.astype(np.float32), sign


# ---- weights ---------------------------------------------------------------------------------------------------

def quantize_weights(w: np.ndarray) -> np.ndarray:
    """Float weights (N×4, any positive scale) → u8 rows summing to exactly 255 (B: DLE quantize_weights; census:
    every skinned vertex row in the corpus sums to 255). Rows that sum to 0 stay all-zero."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 4:
        raise FormatError("weights must be N×4")
    s = w.sum(axis=1, keepdims=True)
    ok = s[:, 0] > 0
    norm = np.zeros_like(w)
    norm[ok] = w[ok] / s[ok]
    scaled = norm * 255.0
    base = np.floor(scaled).astype(np.int64)
    frac = scaled - base
    remainder = 255 - base.sum(axis=1)
    remainder[~ok] = 0
    order = np.argsort(-frac, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(4)[None, :].repeat(len(w), 0), axis=1)
    add = (rank < remainder[:, None]).astype(np.int64)
    out = base + add
    return out.astype(np.uint8)


# ---- decode / encode of a whole block ------------------------------------------------------------------------

class DecodedVertices:
    """Float view of one vertex window plus the exact raw records it came from.

    positions (N×3 f32), uv0 (N×2 f32), uv1 (N×2 f32 | None), normals/tangents (N×3 f32), tangent_sign (N int8),
    weights (N×4 f32, raw/255) and joints (N×4 u8, submesh-palette indices) for skinned formats, else None.
    `raw` is the structured array of the on-disk records (never modified) — the encoder uses it to reproduce
    every field bit-exactly for vertices whose float attributes are unchanged.
    """

    __slots__ = ("format", "count", "raw", "positions", "uv0", "uv1", "normals", "tangents", "tangent_sign",
                 "weights", "joints")

    def __init__(self, fmt: int, raw: np.ndarray):
        self.format = fmt
        self.raw = raw
        self.count = len(raw)
        self.positions = raw["pos"].astype(np.float32)
        self.uv0 = raw["uv0"].astype(np.float32)
        self.uv1 = raw["uv1"].astype(np.float32) if "uv1" in raw.dtype.names else None
        self.tangents, self.normals, self.tangent_sign = decode_qtangent(raw["qtan"], QTAN_SCALE[fmt])
        if is_skinned(fmt):
            self.weights = raw["weights"].astype(np.float32) / np.float32(255.0)
            self.joints = raw["joints"].copy()
        else:
            self.weights = None
            self.joints = None

    @property
    def skinned(self) -> bool:
        return is_skinned(self.format)

    def weight_sums(self) -> np.ndarray | None:
        return None if self.weights is None else self.raw["weights"].astype(np.int32).sum(axis=1)

    @classmethod
    def from_floats(cls, fmt: int, positions, uv0, normals, tangents, tangent_sign, *, uv1=None, weights=None,
                    joints=None) -> "DecodedVertices":
        """Build a float vertex set without raw records (phase-2 import path). `raw` is None; `encode_block`
        then needs explicit *raw*/*raw_ids* for bit-exact reuse and *hole_defaults* for the rest."""
        if fmt not in VERTEX_DTYPES:
            raise UnsupportedError(f"vertex format {fmt} is not one of 0/3/6/8")
        self = cls.__new__(cls)
        self.format = fmt
        self.raw = None
        self.positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
        self.count = len(self.positions)
        self.uv0 = np.ascontiguousarray(uv0, dtype=np.float32).reshape(-1, 2)
        self.uv1 = None if uv1 is None else np.ascontiguousarray(uv1, dtype=np.float32).reshape(-1, 2)
        self.normals = np.ascontiguousarray(normals, dtype=np.float32).reshape(-1, 3)
        self.tangents = np.ascontiguousarray(tangents, dtype=np.float32).reshape(-1, 3)
        self.tangent_sign = np.ascontiguousarray(tangent_sign, dtype=np.int8).reshape(-1)
        if is_skinned(fmt):
            if weights is None or joints is None:
                raise FormatError(f"format {fmt} needs weights and joints")
            self.weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1, 4)
            self.joints = np.ascontiguousarray(joints, dtype=np.uint8).reshape(-1, 4)
        else:
            self.weights = None
            self.joints = None
        for name, a in (("uv0", self.uv0), ("normals", self.normals), ("tangents", self.tangents),
                        ("tangent_sign", self.tangent_sign), ("weights", self.weights), ("joints", self.joints)):
            if a is not None and len(a) != self.count:
                raise FormatError(f"{name}: {len(a)} rows for {self.count} vertices")
        return self


def decode_block(buffer, base: int, count: int, fmt: int) -> DecodedVertices:
    return DecodedVertices(fmt, raw_block(buffer, base, count, fmt))


def hole_defaults(fmt: int, window: np.ndarray | None) -> dict:
    """Values for the raw holes of NEW vertices (phase-2 policy, notes/FORMATS/mesh.md §Holes):
    raw_06 (format 0) = 0x3C00 (census 2026-09-15: 9,147/9,147 format-0 entries carry only that value);
    raw_tail (formats 3/6/8) = the most common raw_tail of the entry's original window (14,824/16,964 format-3
    and 1,676/1,818 format-6 entries hold a single value; format 8 is always 0), or 0 for an entry with no
    original vertices; raw_ext (format 8) = 40 zero bytes (42/42 shipped format-8 entries contain all-zero
    rows, ≥ 20.3 % of each entry; 201,203/634,247 rows overall). *window*: the entry's original raw records."""
    dt = VERTEX_DTYPES[fmt]
    out: dict = {}
    if "raw_06" in dt.names:
        out["raw_06"] = np.uint16(0x3C00)
    if "raw_tail" in dt.names:
        if window is not None and len(window):
            vals, cnt = np.unique(window["raw_tail"], return_counts=True)
            out["raw_tail"] = vals[cnt.argmax()]
        else:
            out["raw_tail"] = np.uint32(0)
    if "raw_ext" in dt.names:
        out["raw_ext"] = np.zeros(40, dtype=np.uint8)
    return out


def encode_block(v: DecodedVertices, *, raw: np.ndarray | None = None, raw_ids: np.ndarray | None = None,
                 holes: dict | None = None) -> np.ndarray:
    """Float attributes → structured records of `v.format`.

    Positions/UVs are written from the float arrays (f32 → f32 exact; f32 → f16 exact for values that came from
    f16). For the quantised fields (qtan, weights, joints) and the raw holes (raw_06, raw_tail, raw_ext) the encoder
    consults *raw* (default: `v.raw`), indexed by *raw_ids* (default: identity): a vertex whose float frame equals
    the decode of its raw record keeps the raw quad (the stored quads are not unit length, so re-quantising would
    differ by 1 LSB on ~13 % of vertices: census 2026-09-15 requant_off_by_1 5,382,243/39,804,375); otherwise the
    frame is re-quantised. Weights are re-quantised to sum 255 whenever the float row no longer equals raw/255.
    Vertices without a raw record (raw_ids == −1) get the hole values from *holes* (default: `hole_defaults`
    computed over *raw*).
    """
    fmt = v.format
    dt = VERTEX_DTYPES[fmt]
    n = v.count
    out = np.zeros(n, dtype=dt)
    out["pos"] = v.positions.astype(dt["pos"].base)
    out["uv0"] = v.uv0.astype(np.float16)
    if "uv1" in dt.names:
        if v.uv1 is None:
            raise FormatError(f"format {fmt} needs uv1")
        out["uv1"] = v.uv1.astype(np.float16)
    if raw is None:
        raw = v.raw
    if raw_ids is None:
        raw_ids = np.arange(n) if raw is not None and len(raw) == n else np.full(n, -1)
    raw_ids = np.asarray(raw_ids)
    has = raw_ids >= 0
    src = raw[raw_ids[has]] if raw is not None and has.any() else None

    # qtangent: keep raw quads where the decoded frame is unchanged
    qtan = encode_qtangent(v.tangents, v.normals, v.tangent_sign, fmt)
    if src is not None:
        t0, n0, s0 = decode_qtangent(src["qtan"], QTAN_SCALE[fmt])
        same = (np.all(t0 == v.tangents[has], axis=1) & np.all(n0 == v.normals[has], axis=1)
                & (s0 == v.tangent_sign[has]))
        keep = np.zeros(n, dtype=bool)
        keep[np.nonzero(has)[0][same]] = True
        qtan[keep] = src["qtan"][same]
    out["qtan"] = qtan

    if is_skinned(fmt):
        if v.weights is None or v.joints is None:
            raise FormatError(f"format {fmt} needs weights and joints")
        w = quantize_weights(v.weights)
        if src is not None:
            w0 = src["weights"].astype(np.float32) / np.float32(255.0)
            same = np.all(w0 == v.weights[has], axis=1)
            keep = np.zeros(n, dtype=bool)
            keep[np.nonzero(has)[0][same]] = True
            w[keep] = src["weights"][same]
        out["weights"] = w
        out["joints"] = v.joints.astype(np.uint8)

    # raw holes
    if holes is None:
        holes = hole_defaults(fmt, raw)
    for name in ("raw_06", "raw_tail", "raw_ext"):
        if name not in dt.names:
            continue
        if src is not None:
            out[name][has] = src[name]
        if not has.all():
            out[name][~has] = holes[name]
    return out
