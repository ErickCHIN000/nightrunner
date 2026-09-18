"""Material-aware preview images for the 3D views (eyes, hair cut-outs, opacity-blended surfaces).

The 3D preview cannot run the game's shaders, so for a few well-understood shader families it bakes the texture
layers into one RGBA image plus an alpha mode. The recipes are ports of DyingLightExplorer's bounded previews
(`materials.py` eye_preview / cutout_preview / opacity_preview, docs "Eye materials.md"), whose inputs were checked
against the DX11 pixel programs (renderer +0x3AB50 / +0x3C5F0 chains, BN 2026-09-10/11):

* **eye_layers** — tokens ⊇ {eyes_blicks_on, od1_tex, od2_tex, off_tex}: iris `od2_tex` blended toward sclera
  `od1_tex` by the sclera alpha (of1_msk_range_min/max, of1_msk_opacity), then toward the veins map `dif_0_tex`
  by the red channel of `off_tex` (off_msk_range_min/max, off_msk_opacity). `dif_0_tex` alone is only the veins map
  (why eyes rendered red/white). View-dependent parallax (of*_factor) and the emissive iris layer are not baked.
* **dither_cutout** — `dit_0_tex` (hair, beards): colour = `dif_0_tex` × `dif_0_val`, alpha = red of `dit_0_tex`,
  alpha-tested at 0.25 (the native depth-pass cutoff).
* **diffuse_opacity** — `opc_0_tex` (forearm hair, eye shadow, wet eye): colour = `dif_0_tex` × `dif_0_val`,
  alpha = (red of `opc_0_tex` − low) / (high − low) with `opc_3_ranges` (the diffuse channel of the
  "Diffuse;0 Normal,1 Specular,…" ranges), alpha-blended. Eye shadow / wet eye drawn opaque white covered the eyes.
* **plain** — `dif_0_tex` as is. Materials whose every shader variant has zero render passes (`null.mat`) are
  **hidden**.

Tiers: recipes are (B) — an older tool's shader reading, not re-verified here; runtime appearance is not claimed.
Unsupported variations (UV1/runtime opacity, extra offset layers, transformed layer UVs) are reported in
`warnings` and fall back to the plain diffuse. Everything here is worker-thread safe (no widgets; QImage only).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PySide6.QtGui import QImage

TEXTURE_TYPE = 0x20
OPAQUE, MASK, BLEND = "opaque", "mask", "blend"
HAIR_CUTOFF = 0.25

_EYE_TOKENS = {"eyes_blicks_on", "od1_tex", "od2_tex", "off_tex"}
_EYE_UNSUPPORTED = {"od3_tex", "od4_tex", "off_frs_on"}
_OPC_UNSUPPORTED = {"opc_uv_1_on", "opc_frs_on", "opc_usr_on", "opc_noise_tex", "opc_ovr_on"}
_DIT_UNSUPPORTED = {"dit_1_tex", "dit_usr_on", "dit_threshold_usr_on", "dit_0_clamp_u_on", "dit_0_clamp_v_on"}
_EYE_UV_DEFAULTS: dict[str, Any] = {"off_prop": (1, 1), "uv_0_offset": (0, 0, 0)}
for _l in (1, 2):
    _EYE_UV_DEFAULTS.update({f"of{_l}_scale": 1, f"of{_l}_prop": (1, 1), f"of{_l}_pos": (0, 0),
                             f"of{_l}_clamp_u_on": False, f"of{_l}_clamp_v_on": False})


@dataclass
class SurfacePreview:
    image: QImage | None = None
    alpha_mode: str = OPAQUE
    cutoff: float = 0.5
    recipe: str = "none"
    hidden: bool = False
    textures: dict = field(default_factory=dict)     # role -> texture name
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        if self.hidden:
            return "hidden (non-rendering material)"
        s = self.recipe
        if self.alpha_mode != OPAQUE:
            s += f", {self.alpha_mode}" + (f" @{self.cutoff:g}" if self.alpha_mode == MASK else "")
        if self.warnings:
            s += " — " + "; ".join(self.warnings[:2])
        return s


# ---- SDB facts --------------------------------------------------------------------------------------------------

def material_facts(info: dict, bindings_override: dict[str, str] | None = None) -> tuple[set, dict, dict, list]:
    """(tokens, parameter values, texture per parameter name, warnings) merged over every route / variant.
    *bindings_override* (param name -> texture) replaces bindings, e.g. `.model` type-7 rttiValues."""
    tokens: set[str] = set()
    values: dict[str, Any] = {}
    tex: dict[str, str] = {}
    warnings: list[str] = []
    for r in info.get("routes", []) or []:
        tokens |= {t for t in (r.get("tokens") or "").split(";") if t}
        for p in r.get("parameters", []) or []:
            if p.get("name") and p["name"] not in values:
                values[p["name"]] = p.get("value")
        for v in r.get("variants", []) or []:
            for b in v.get("texture_bindings", []) or []:
                pn, tn = b.get("param"), b.get("texture")
                if not pn or not tn:
                    continue
                if pn in tex and tex[pn] != tn:
                    msg = f"variants disagree on {pn}"
                    if msg not in warnings:
                        warnings.append(msg)
                    continue
                tex.setdefault(pn, tn)
    for pn, v in values.items():            # parameters of type string that the shader variants did not bind
        if pn.endswith("_tex") and isinstance(v, str) and v:
            tex.setdefault(pn, v)
    for pn, tn in (bindings_override or {}).items():
        if pn and tn:
            tex[pn] = tn
    return tokens, values, tex, warnings


def _num(v, default):
    if v is None:
        return default
    try:
        a = np.asarray(v, dtype=np.float64)
        return a if a.size > 1 else float(a.reshape(-1)[0])
    except (TypeError, ValueError):
        return default


# ---- texture fetch ----------------------------------------------------------------------------------------------

class TextureFetcher:
    """Texture name -> float32 RGBA array (0..1), decoded once through the catalog + texpreview (cached as uint8
    to keep a whole model's textures small)."""

    def __init__(self, catalog, max_dim: int = 1024):
        self.catalog, self.max_dim = catalog, max_dim
        self._cache: dict[str, np.ndarray | None] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def __call__(self, name: str) -> np.ndarray | None:
        key = name.lower()
        with self._lock:
            if key in self._cache:
                c = self._cache[key]
                return None if c is None else c.astype(np.float32) / 255.0
        arr, err = None, None
        gids = self.catalog.lookup(name, TEXTURE_TYPE) if name else []
        if not gids:
            err = "not in any loaded pack"
        else:
            try:
                from .texpreview import decode_texture
                e, i = self.catalog.split(gids[0])
                img, info = decode_texture(e.pack, i, mip=None, max_dim=self.max_dim)
                if img.isNull():
                    err = info.get("error") or "empty image"
                else:
                    arr = qimage_to_u8(img)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._cache[key] = arr
            if err:
                self._errors[key] = err
        return None if arr is None else arr.astype(np.float32) / 255.0

    def error(self, name: str) -> str | None:
        return self._errors.get(name.lower())


def qimage_to_u8(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    buf = np.frombuffer(bytes(img.constBits())[: img.sizeInBytes()], dtype=np.uint8)
    stride = img.bytesPerLine()
    return buf.reshape(h, stride)[:, : w * 4].reshape(h, w, 4).copy()


def qimage_to_array(img: QImage) -> np.ndarray:
    return qimage_to_u8(img).astype(np.float32) / 255.0


def array_to_qimage(a: np.ndarray) -> QImage:
    px = np.ascontiguousarray(np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8))
    h, w = px.shape[:2]
    return QImage(px.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888).copy()


def _resize(a: np.ndarray, w: int, h: int) -> np.ndarray:
    if a.shape[1] == w and a.shape[0] == h:
        return a
    from PIL import Image
    chans = [np.asarray(Image.fromarray(a[:, :, c]).resize((w, h), Image.Resampling.BILINEAR)) for c in range(4)]
    return np.stack(chans, axis=2).astype(np.float32)


def _common(arrays: list[np.ndarray]) -> list[np.ndarray]:
    w = max(a.shape[1] for a in arrays)
    h = max(a.shape[0] for a in arrays)
    return [_resize(a, w, h) for a in arrays]


def _tint(values: dict) -> np.ndarray:
    t = _num(values.get("dif_0_val"), None)
    if isinstance(t, np.ndarray) and t.shape[-1] >= 3 and np.isfinite(t).all():
        return t.reshape(-1)[:3].astype(np.float32)
    return np.ones(3, dtype=np.float32)


# ---- recipes ------------------------------------------------------------------------------------------------------

def compose(info: dict | None, fetch: Callable[[str], np.ndarray | None],
            bindings_override: dict[str, str] | None = None) -> SurfacePreview:
    """Preview for one SDB material dict (`Sdb.material()`); *fetch* maps texture name -> RGBA float array."""
    out = SurfacePreview()
    if not info:
        out.warnings.append("material not in SDB")
        return out
    if info.get("non_rendering"):
        out.hidden, out.recipe = True, "non_rendering"
        return out
    tokens, values, tex, warns = material_facts(info, bindings_override)
    out.warnings += warns
    dif = tex.get("dif_0_tex")

    if _EYE_TOKENS <= tokens and all(tex.get(k) for k in ("dif_0_tex", "od1_tex", "od2_tex", "off_tex")):
        res = _eye(tokens, values, tex, fetch, out)
        if res is not None:
            return res
    if tex.get("dit_0_tex") and "opc_0_tex" not in tokens and dif:
        res = _cutout(tokens, values, tex, fetch, out)
        if res is not None:
            return res
    if tex.get("opc_0_tex") and dif:
        res = _opacity(tokens, values, tex, fetch, out)
        if res is not None:
            return res
    if dif:
        a = fetch(dif)
        out.textures["diffuse"] = dif
        if a is not None:
            out.recipe = "plain"
            a = a.copy()
            a[:, :, 3] = 1.0
            out.image = array_to_qimage(a)
        else:
            out.warnings.append(f"{dif}: {getattr(fetch, 'error', lambda n: None)(dif) or 'unavailable'}")
    else:
        out.warnings.append("no dif_0_tex binding")
    return out


def _missing(fetch, names: list[str], out: SurfacePreview) -> list[np.ndarray] | None:
    arrays = [fetch(n) for n in names]
    bad = [n for n, a in zip(names, arrays) if a is None]
    if bad:
        err = getattr(fetch, "error", lambda n: None)
        out.warnings.append("missing " + ", ".join(f"{n} ({err(n) or 'unavailable'})" for n in bad))
        return None
    return arrays


def _eye(tokens, values, tex, fetch, out: SurfacePreview) -> SurfacePreview | None:
    if tokens & _EYE_UNSUPPORTED:
        out.warnings.append("eye: extra offset layers / Fresnel not baked")
    for k, d in _EYE_UV_DEFAULTS.items():
        v = values.get(k)
        if v is not None and not np.allclose(np.asarray(v, dtype=np.float64), np.asarray(d, dtype=np.float64),
                                             atol=1e-6):
            out.warnings.append(f"eye: layer UV setting {k} not baked")
            break
    names = [tex["dif_0_tex"], tex["od1_tex"], tex["od2_tex"], tex["off_tex"]]
    arrays = _missing(fetch, names, out)
    if arrays is None:
        return None
    veins, sclera, iris, mask = _common(arrays)
    s = [float(_num(values.get(k), d)) for k, d in (("of1_msk_range_min", 0.0), ("of1_msk_range_max", 1.0),
                                                     ("of1_msk_opacity", 1.0), ("off_msk_range_min", 0.0),
                                                     ("off_msk_range_max", 1.0), ("off_msk_opacity", 1.0))]
    amin, amax, opacity, mmin, mmax, mopacity = s
    if not np.isfinite(s).all() or amax <= amin or mmax <= mmin:
        out.warnings.append("eye: invalid mask range")
        return None
    alpha = np.clip((sclera[:, :, 3:4] - amin) / (amax - amin), 0, 1) * opacity
    weight = 1 - mopacity + np.clip((mask[:, :, 0:1] - mmin) / (mmax - mmin), 0, 1) * mopacity
    color = iris[:, :, :3] * (1 - alpha) + sclera[:, :, :3] * alpha
    color = color * (1 - weight) + veins[:, :, :3] * weight
    img = np.ones(color.shape[:2] + (4,), dtype=np.float32)
    img[:, :, :3] = color
    out.image, out.recipe = array_to_qimage(img), "eye_layers"
    out.textures.update(veins=names[0], sclera=names[1], iris=names[2], mask=names[3])
    return out


def _cutout(tokens, values, tex, fetch, out: SurfacePreview) -> SurfacePreview | None:
    if tokens & _DIT_UNSUPPORTED:
        out.warnings.append("cutout: runtime / clamped dither not evaluated")
    names = [tex["dif_0_tex"], tex["dit_0_tex"]]
    arrays = _missing(fetch, names, out)
    if arrays is None:
        return None
    color, cov = _common(arrays)
    img = color.copy()
    img[:, :, :3] = np.clip(img[:, :, :3] * _tint(values), 0, 1)
    img[:, :, 3] = cov[:, :, 0]
    out.image, out.recipe, out.alpha_mode, out.cutoff = array_to_qimage(img), "dither_cutout", MASK, HAIR_CUTOFF
    out.textures.update(diffuse=names[0], coverage=names[1])
    return out


def _opacity(tokens, values, tex, fetch, out: SurfacePreview) -> SurfacePreview | None:
    if tokens & _OPC_UNSUPPORTED:
        out.warnings.append("opacity: UV1 / runtime mask not evaluated — shown opaque")
        return None
    lim = _num(values.get("opc_3_ranges"), None)
    low, high = (0.0, 255.0)
    if isinstance(lim, np.ndarray) and lim.size >= 2:
        low, high = float(lim.reshape(-1)[0]), float(lim.reshape(-1)[1])
    if not (np.isfinite([low, high]).all() and high > low):
        out.warnings.append("opacity: invalid opc_3_ranges")
        return None
    names = [tex["dif_0_tex"], tex["opc_0_tex"]]
    arrays = _missing(fetch, names, out)
    if arrays is None:
        return None
    color, mask = _common(arrays)
    img = color.copy()
    img[:, :, :3] = np.clip(img[:, :, :3] * _tint(values), 0, 1)
    img[:, :, 3] = np.clip((mask[:, :, 0] * 255.0 - low) / (high - low), 0, 1)
    out.image, out.recipe, out.alpha_mode = array_to_qimage(img), "diffuse_opacity", BLEND
    out.textures.update(diffuse=names[0], opacity=names[1])
    return out


def preview_for_material(ctx, name: str, fetch: TextureFetcher | None = None,
                         bindings_override: dict[str, str] | None = None) -> SurfacePreview:
    """Convenience for tabs: SDB lookup through ctx.sdb, textures through ctx.catalog."""
    fetch = fetch or TextureFetcher(ctx.catalog)
    info = ctx.sdb.material_info(name) if name else None
    return compose(info, fetch, bindings_override)
