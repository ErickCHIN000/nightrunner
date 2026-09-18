"""Texture decoding for the GUI (worker-thread safe) and a zoomable preview widget.

`read_header(pack, index)` parses only the 0x20 IMGC part (cheap: list columns). `decode_texture(...)` decodes ONE
surface (mip / face / slice) of a texture into a QImage plus an info dict; it never raises for bad data — the
error text is returned in ``info["error"]`` and the image is null. Decoding reuses nightrunner.texture.png
(`decode_preview`: own BC4/BC5 + uncompressed decoders, Pillow for BC1/2/3/6H/7) and is done in horizontal bands
so a 16k texture never needs more than a band's worth of temporary float buffers.

`TexturePreview` is a QWidget (QGraphicsView based): wheel zoom, drag pan, Fit / 1:1, checkerboard or solid
background, R/G/B/A channel toggles (a single channel shows as greyscale), mip / face / slice selectors that emit
`levelRequested` (the owner re-decodes; this widget never touches packs).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QColorDialog, QComboBox, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView,
                               QHBoxLayout, QLabel, QSpinBox, QToolButton, QVBoxLayout, QWidget)

from ..container.rp6l import Pack
from ..texture import formats
from ..texture.codec import part_ordinals
from ..texture.imgc import ImgcHeader, check_payload, detect_level_padding, parse_header
from ..texture.png import BC45, PILLOW_BCN, UNCOMPRESSED, decode_preview

TEXTURE_TYPE = 0x20
BAND_PIXELS = 1 << 22             # decode at most ~4 M pixels per band (bounded temporary memory)


def decodable(il_format: int) -> bool:
    """True when nightrunner.texture.png has a preview decoder for this IL format."""
    return il_format in UNCOMPRESSED or il_format in PILLOW_BCN or il_format in BC45


def supported_formats() -> list[str]:
    """Names of every IL format the preview can show (for messages / docs)."""
    return [f.name for i, f in sorted(formats.FORMATS.items()) if decodable(i)]


def read_header(pack: Pack, index: int) -> tuple[ImgcHeader, int, int | None]:
    """(header, physical index of the 0x20 part, physical index of the 0x21 part or None). Raises on bad data."""
    lg = pack.logicals[index]
    parts = range(lg.first_part, lg.first_part + lg.part_count)
    hdr_k, bmp_k = part_ordinals([pack.part_type(i) for i in parts])
    if hdr_k is None:
        raise ValueError("no 0x20 IMGC header part")
    hdr_i = parts[hdr_k]
    mv = pack.read_part(hdr_i)
    try:
        h = parse_header(mv, strict_length=False)
    finally:
        mv.release()
    bmp_i = None if (bmp_k is None or h.header_only) else parts[bmp_k]
    return h, hdr_i, bmp_i


def header_summary(pack: Pack, index: int) -> tuple[str, str, str]:
    """(format, "WxH[xD] type", mips) strings for list columns; ("?", "?", "?") on garbage."""
    try:
        h, _, _ = read_header(pack, index)
    except Exception:  # noqa: BLE001 - garbage headers are expected (stress packs, third-party packs)
        return "?", "?", "?"
    if h.header_only:
        return "header-only", "→ " + (h.reference or "?"), "0"
    size = f"{h.width}x{h.height}"
    if h.is_volume:
        size += f"x{h.depth}"
    if h.is_cube:
        size += " cube"
    return h.format_name, size, str(h.mip_count)


def pick_mip(h: ImgcHeader, max_dim: int) -> int:
    """Smallest mip whose longer side is still >= max_dim (mip 0 when the texture is already small)."""
    best = 0
    for m in range(h.mip_count):
        if max(1, max(h.width, h.height) >> m) >= max_dim:
            best = m
    return best


def _fast_unorm8(data, width: int, height: int, il_format: int) -> np.ndarray | None:
    """Byte-exact shortcut of png.decode_preview for 8-bit UNORM layouts (no float round trip), else None."""
    spec = UNCOMPRESSED.get(il_format)
    if spec is None or spec[0] != "u1" or spec[2] != "unorm":
        return None
    _, ch, _, swz = spec
    src = np.frombuffer(data, np.uint8)
    if src.size != width * height * ch:
        return None
    src = src.reshape(height, width, ch)
    if swz:
        src = src[:, :, list(swz)]
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[:, :, 3] = 255
    rgba[:, :, :min(ch, 4)] = src[:, :, :4]
    return rgba


def _preview(data, width: int, height: int, il_format: int) -> np.ndarray:
    fast = _fast_unorm8(data, width, height, il_format)
    return fast if fast is not None else decode_preview(data, width, height, il_format)


def _decode_surface(data, width: int, height: int, il_format: int, step: int) -> np.ndarray:
    """RGBA8 (h', w', 4) of one surface, decoded in bands of whole rows (block rows for BC). `step` > 1 keeps every
    step-th pixel (cheap downscale for thumbnails / huge previews)."""
    f = formats.get(il_format)
    sub = step // 4 if f.block else step
    if sub > 1:
        # Subsample whole blocks / pixels before decoding: blocks are independent, so a thumbnail of a 16k
        # texture decodes only ~1/sub² of the data.
        n_units = (((width + 3) // 4) * ((height + 3) // 4)) if f.block else width * height
        if len(data) != n_units * f.unit:
            raise ValueError(f"surface is {len(data)} bytes, expected {n_units * f.unit}")
        cols = (width + 3) // 4 if f.block else width
        units = np.frombuffer(data, np.uint8).reshape(-1, cols, f.unit)[::sub, ::sub]
        sw = units.shape[1] * (4 if f.block else 1)
        sh = units.shape[0] * (4 if f.block else 1)
        if f.block:        # keep the image the size of whole blocks, trimmed like the original edge
            sw = min(sw, -(-width // sub))
            sh = min(sh, -(-height // sub))
        return _decode_surface(np.ascontiguousarray(units).tobytes(), sw, sh, il_format, step // sub)
    if f.block:
        row_h, row_bytes = 4, ((width + 3) // 4) * f.unit
    else:
        row_h, row_bytes = 1, width * f.unit
    rows_total = (height + row_h - 1) // row_h
    rows_per_band = max(1, BAND_PIXELS // max(1, width * row_h))
    if step == 1:
        fast = _fast_unorm8(data, width, height, il_format)      # no float temporaries: no banding needed
        if fast is not None:
            return fast
        if rows_per_band >= rows_total:
            return decode_preview(data, width, height, il_format)
    out = []
    y = 0
    for r0 in range(0, rows_total, rows_per_band):
        r1 = min(rows_total, r0 + rows_per_band)
        band_h = min(height, r1 * row_h) - r0 * row_h
        band = _preview(data[r0 * row_bytes: r1 * row_bytes], width, band_h, il_format)
        if step > 1:
            first = (-y) % step              # keep global rows 0, step, 2*step, …
            band = band[first::step, ::step]
        out.append(band)
        y += band_h
    return np.ascontiguousarray(np.concatenate(out, axis=0))


def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    """Owned QImage (RGBA8888) from an (h, w, 4) uint8 array. Safe in worker threads."""
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return img.copy()


def qimage_to_rgba(img: QImage) -> np.ndarray:
    """Copy of a QImage as an (h, w, 4) uint8 RGBA array."""
    img = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)[:, : 4 * w]
    return arr.reshape(h, w, 4).copy()


def texture_info(pack: Pack, logical_index: int) -> dict[str, Any]:
    """Everything the info panel shows except the decoded image: header, format, levels, parts. Never raises;
    ``error`` holds the first problem found."""
    info: dict[str, Any] = {"index": logical_index, "error": None, "header": None, "levels": [], "parts": []}
    try:
        res = pack.resource(logical_index)
        info["name"] = res.name
        info["type"] = res.type
        for i in res.part_indices:
            p = pack.physicals[i]
            st = pack.part_storage(i)
            info["parts"].append({"index": i, "type": pack.part_type(i), "offset": pack.part_offset(i),
                                  "size": p.size, "method": st.method, "child": bool(p.child),
                                  "direct": pack.part_is_direct(i)})
        h, hdr_i, bmp_i = read_header(pack, logical_index)
        info["header"] = h
        info["imgc"] = h.to_json()
        f = formats.FORMATS.get(h.format)
        info["format"] = {"il": h.format, "name": h.format_name, "dxgi": f.dxgi if f else None,
                          "dxgi_name": f.dxgi_name if f else None, "unit": f.unit if f else None,
                          "block": f.block if f else None, "kind": f.kind if f else None,
                          "tier": f.tier if f else None, "srgb": f.srgb if f else None,
                          "decodable": decodable(h.format)}
        info["bitmap_part"] = bmp_i
        if h.header_only:
            info["error"] = f"header-only record (flag 0x02): no texels; reference {h.reference!r}"
            return info
        if bmp_i is None:
            info["error"] = "no 0x21 bitmap part"
            return info
        size = pack.physicals[bmp_i].size
        pad = detect_level_padding(h, size)
        info["level_padding"] = pad
        info["payload_size"] = size
        info["levels"] = check_payload(h, size, pad)
    except Exception as exc:  # noqa: BLE001 - shown in the tab
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def decode_texture(pack: Pack, logical_index: int, mip: int | None = 0, face: int = 0, slice: int = 0,
                   max_dim: int | None = None) -> tuple[QImage, dict[str, Any]]:
    """Decode one surface to a QImage (worker-thread safe). `mip=None` picks the smallest mip that is still at
    least `max_dim` on its longer side; `max_dim` also downsamples (integer stride) the decoded surface so its
    longer side is <= max_dim·2 (thumbnails) — with max_dim None the surface is returned at full size.
    Returns (QImage (null on error), info) — info is `texture_info()` plus mip/face/slice/width/height/decoder/
    decode_ms, and `error` set instead of raising."""
    t0 = time.perf_counter()
    info = texture_info(pack, logical_index)
    if info["error"]:
        return QImage(), info
    h: ImgcHeader = info["header"]
    try:
        if not decodable(h.format):
            info["error"] = (f"no preview decoder for IL {h.format} ({h.format_name}); DDS / raw export still "
                             f"work. Previewable: uncompressed, BC1–BC7.")
            return QImage(), info
        if mip is None:
            mip = pick_mip(h, max_dim) if max_dim else 0
        mip = max(0, min(int(mip), h.mip_count - 1))
        face = max(0, min(int(face), h.faces - 1))
        lv = next(lv for lv in info["levels"] if lv.mip == mip and lv.face == face)
        slice = max(0, min(int(slice), lv.depth - 1))
        step = 1
        if max_dim:
            step = max(1, max(lv.width, lv.height) // (2 * max_dim))
        mv = pack.read_part(info["bitmap_part"])
        try:
            start = lv.offset + slice * lv.slice_size
            data = mv[start: start + lv.slice_size]
            try:
                rgba = _decode_surface(data, lv.width, lv.height, h.format, step)
            finally:
                data.release()
        finally:
            mv.release()
        img = rgba_to_qimage(rgba)
        info.update(mip=mip, face=face, slice=slice, width=lv.width, height=lv.height, step=step,
                    decoder="pillow-bcn" if h.format in PILLOW_BCN else "bc45" if h.format in BC45 else "numpy")
        info["decode_ms"] = (time.perf_counter() - t0) * 1000.0
        return img, info
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        return QImage(), info


def apply_channels(img: QImage, r: bool, g: bool, b: bool, a: bool) -> QImage:
    """Channel view: one colour channel alone → greyscale; otherwise disabled channels are zeroed; alpha off →
    opaque. Pure function (worker-safe)."""
    if r and g and b and a:
        return img
    arr = qimage_to_rgba(img)
    on = [r, g, b]
    if sum(on) == 1:
        c = on.index(True)
        out = np.empty_like(arr)
        out[..., 0] = out[..., 1] = out[..., 2] = arr[..., c]
        out[..., 3] = arr[..., 3] if a else 255
    elif sum(on) == 0 and a:
        out = np.empty_like(arr)
        out[..., 0] = out[..., 1] = out[..., 2] = arr[..., 3]
        out[..., 3] = 255
    else:
        out = arr
        for c in range(3):
            if not on[c]:
                out[..., c] = 0
        if not a:
            out[..., 3] = 255
    return rgba_to_qimage(out)


# ----------------------------------------------------------------------------------------------------------------
# widget
# ----------------------------------------------------------------------------------------------------------------

def checker_brush(size: int = 8) -> QBrush:
    pm = QPixmap(size * 2, size * 2)
    pm.fill(QColor(0x99, 0x99, 0x99))
    p = QPainter(pm)
    p.fillRect(0, 0, size, size, QColor(0x66, 0x66, 0x66))
    p.fillRect(size, size, size, size, QColor(0x66, 0x66, 0x66))
    p.end()
    return QBrush(pm)


class _View(QGraphicsView):
    zoomChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.FastTransformation)
        self.scene().addItem(self.item)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.bg: QBrush = checker_brush()
        self.fit_mode = True

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.resetTransform()
        painter.fillRect(self.viewport().rect(), self.bg)
        painter.restore()

    def set_pixmap(self, pm: QPixmap) -> None:
        self.item.setPixmap(pm)
        self.scene().setSceneRect(QRectF(pm.rect()))
        if self.fit_mode:
            self.fit()

    def zoom(self) -> float:
        return self.transform().m11()

    def set_zoom(self, z: float) -> None:
        z = max(0.01, min(64.0, z))
        self.resetTransform()
        self.scale(z, z)
        self.zoomChanged.emit(z)

    def fit(self) -> None:
        self.fit_mode = True
        r = self.item.boundingRect()
        if r.isEmpty():
            return
        self.fitInView(r, Qt.KeepAspectRatio)
        self.zoomChanged.emit(self.zoom())

    def wheelEvent(self, e) -> None:
        d = e.angleDelta().y()
        if not d:
            return
        self.fit_mode = False
        f = 1.25 if d > 0 else 0.8
        z = max(0.01, min(64.0, self.zoom() * f))
        self.scale(z / self.zoom(), z / self.zoom())
        self.zoomChanged.emit(self.zoom())

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        if self.fit_mode:
            self.fit()


class TexturePreview(QWidget):
    """Image viewer. Owners call `show_image(img, info)` / `show_message(text)` and listen to `levelRequested`
    (mip, face, slice). Pass a TaskRunner to do channel filtering of big images off the GUI thread."""
    levelRequested = Signal(int, int, int)

    BACKGROUNDS = ("checker", "black", "grey", "white", "custom…")

    def __init__(self, runner=None, parent=None):
        super().__init__(parent)
        self.runner = runner
        self._img = QImage()
        self._info: dict = {}
        self._custom = QColor(40, 40, 60)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.btn_fit = QToolButton(text="Fit")
        self.btn_one = QToolButton(text="1:1")
        self.zoom_label = QLabel("")
        self.zoom_label.setMinimumWidth(48)
        bar.addWidget(self.btn_fit)
        bar.addWidget(self.btn_one)
        bar.addWidget(self.zoom_label)
        self.ch: dict[str, QToolButton] = {}
        colors = {"R": "#e05050", "G": "#50c050", "B": "#5080ff", "A": "#c0c0c0"}
        for c in "RGBA":
            b = QToolButton(text=c, checkable=True, checked=True)
            b.setStyleSheet(f"QToolButton:checked {{ color: {colors[c]}; font-weight: bold; }}")
            b.toggled.connect(self._refilter)
            self.ch[c] = b
            bar.addWidget(b)
        self.spin: dict[str, QSpinBox] = {}
        for key in ("mip", "face", "slice"):
            lab = QLabel(key)
            sp = QSpinBox()
            sp.setRange(0, 0)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._level_changed)
            self.spin[key] = sp
            bar.addWidget(lab)
            bar.addWidget(sp)
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(self.BACKGROUNDS)
        self.bg_combo.activated.connect(self._bg_changed)
        bar.addWidget(self.bg_combo)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.view = _View()
        lay.addWidget(self.view, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        self.btn_fit.clicked.connect(self.view.fit)
        self.btn_one.clicked.connect(self._one_to_one)
        self.view.zoomChanged.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self._suppress = False

    # ---- public -------------------------------------------------------------------------------------------------
    def set_levels(self, mips: int, faces: int, depth: int, mip: int = 0, face: int = 0, slice: int = 0) -> None:
        """Configure the selector ranges without emitting levelRequested."""
        self._suppress = True
        for key, n, v in (("mip", mips, mip), ("face", faces, face), ("slice", depth, slice)):
            sp = self.spin[key]
            sp.setRange(0, max(0, n - 1))
            sp.setValue(v)
            sp.setEnabled(n > 1)
        self._suppress = False

    def level(self) -> tuple[int, int, int]:
        return self.spin["mip"].value(), self.spin["face"].value(), self.spin["slice"].value()

    def show_image(self, img: QImage, info: dict | None = None) -> None:
        self._img = img
        self._info = info or {}
        if img.isNull():
            self.show_message((info or {}).get("error") or "nothing to show")
            return
        i = self._info
        parts = [f"{img.width()}×{img.height()}"]
        if i.get("step", 1) > 1:
            parts.append(f"(downsampled 1/{i['step']})")
        if "mip" in i:
            parts.append(f"mip {i['mip']} face {i.get('face', 0)} slice {i.get('slice', 0)}")
        if i.get("decoder"):
            parts.append(f"decoder {i['decoder']}")
        if i.get("decode_ms") is not None:
            parts.append(f"{i['decode_ms']:.0f} ms")
        parts.append("8-bit preview (signed → [0,1], HDR clipped)")
        self.status.setText("  ·  ".join(parts))
        self._refilter()

    def show_message(self, text: str) -> None:
        self._img = QImage()
        self.view.set_pixmap(QPixmap())
        self.status.setText(text)

    def clear(self) -> None:
        self.show_message("")

    def image(self) -> QImage:
        return self._img

    # ---- internals ----------------------------------------------------------------------------------------------
    def _channels(self) -> tuple[bool, bool, bool, bool]:
        return tuple(self.ch[c].isChecked() for c in "RGBA")  # type: ignore[return-value]

    def _refilter(self, *_):
        img = self._img
        if img.isNull():
            return
        chans = self._channels()
        if all(chans):
            self._set(img)
        elif self.runner is not None and img.width() * img.height() > 1 << 20:
            self.runner.submit(("tex-channels", id(self)), apply_channels, img, *chans,
                               on_done=lambda out, src=img: self._set(out) if src is self._img else None)
        else:
            self._set(apply_channels(img, *chans))

    def _set(self, img: QImage) -> None:
        self.view.set_pixmap(QPixmap.fromImage(img))

    def _one_to_one(self) -> None:
        self.view.fit_mode = False
        self.view.set_zoom(1.0)

    def _level_changed(self, *_):
        if not self._suppress:
            self.levelRequested.emit(*self.level())

    def _bg_changed(self, i: int) -> None:
        name = self.BACKGROUNDS[i]
        if name == "checker":
            self.view.bg = checker_brush()
        elif name == "custom…":
            c = QColorDialog.getColor(self._custom, self, "Preview background")
            if c.isValid():
                self._custom = c
            self.view.bg = QBrush(self._custom)
        else:
            self.view.bg = QBrush(QColor({"black": "#000000", "grey": "#808080", "white": "#ffffff"}[name]))
        self.view.viewport().update()
