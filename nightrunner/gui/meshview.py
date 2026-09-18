"""MeshView: an embeddable 3D viewer for decoded mesh buffers, built on Qt Quick 3D (ships with PySide6).

    view = MeshView()
    view.add_mesh("e0/s0", positions, indices, normals, uv, color=(0.8, 0.5, 0.2), texture=qimage)
    view.frame_all()

Rendering goes through Qt's RHI (Direct3D 11 on Windows, Metal on macOS, OpenGL on Linux) inside a QQuickWidget, so
no extra dependency is needed. When Qt Quick 3D cannot render (software scene graph, offscreen/minimal platform,
missing module, scene-graph error), the widget shows a message instead and every method keeps working as a no-op
(bounds and visibility are still tracked, so callers never need to special-case it).

Controls: LMB drag orbit · MMB drag or Shift+LMB pan · wheel zoom · F frame all · W wireframe · N normals view ·
double-click frame all.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Q_ARG, Q_RETURN_ARG, QByteArray, QEvent, QMetaObject, QObject, QPointF, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QQuaternion, QVector3D
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

try:
    from PySide6.QtQml import QQmlComponent
    from PySide6.QtQuick import QQuickWindow, QSGRendererInterface
    from PySide6.QtQuick3D import QQuick3DGeometry, QQuick3DTextureData
    from PySide6.QtQuickWidgets import QQuickWidget
    HAVE_QUICK3D = True
    _IMPORT_ERROR = ""
except Exception as _exc:  # noqa: BLE001 - optional Qt modules
    HAVE_QUICK3D = False
    _IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

_ALPHA = {"opaque": 0, "mask": 1, "blend": 2}

NO_3D_PLATFORMS = ("offscreen", "minimal", "minimalegl", "vnc", "linuxfb")

QML = b"""
import QtQuick
import QtQuick3D

Item {
    id: root
    property vector3d camPos: Qt.vector3d(0, 0, 5)
    property quaternion camRot: Qt.quaternion(1, 0, 0, 0)
    property real near: 0.01
    property real far: 10000
    property bool wireframe: false
    property int shading: 0            // 0 lit colour, 1 normals, 2 unlit colour
    property color background: "#2b2d30"
    property bool flipV: false

    function addPart(geom, col, tex, amode, cutoff) {
        return partComp.createObject(parts, {"geometry": geom, "col": col, "texData": tex,
                                             "alphaModeI": amode, "cutoffV": cutoff});
    }

    Component {
        id: partComp
        Model {
            id: m
            property color col: "gray"
            property var texData: null
            property int alphaModeI: 0         // 0 opaque, 1 mask (alpha test), 2 blend
            property real cutoffV: 0.5
            Texture {
                id: tex
                textureData: m.texData ? m.texData : null
                flipV: root.flipV
                generateMipmaps: true
                mipFilter: Texture.Linear
            }
            materials: [
                PrincipledMaterial {
                    baseColor: m.texData ? "white" : m.col
                    baseColorMap: m.texData ? tex : null
                    roughness: 0.75
                    metalness: 0.0
                    cullMode: Material.NoCulling
                    alphaMode: (m.texData && m.alphaModeI === 1) ? PrincipledMaterial.Mask
                             : (m.texData && m.alphaModeI === 2) ? PrincipledMaterial.Blend
                             : PrincipledMaterial.Opaque
                    alphaCutoff: m.cutoffV
                    lighting: root.shading === 2 ? PrincipledMaterial.NoLighting : PrincipledMaterial.FragmentLighting
                }
            ]
        }
    }

    View3D {
        id: view
        anchors.fill: parent
        environment: SceneEnvironment {
            clearColor: root.background
            backgroundMode: SceneEnvironment.Color
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
            debugSettings: DebugSettings {
                wireframeEnabled: root.wireframe
                materialOverride: root.shading === 1 ? DebugSettings.Normals : DebugSettings.None
            }
        }
        PerspectiveCamera {
            id: cam
            position: root.camPos
            rotation: root.camRot
            clipNear: root.near
            clipFar: root.far
            fieldOfView: 45
            DirectionalLight { brightness: 0.9 }
        }
        DirectionalLight { eulerRotation.x: -70; eulerRotation.y: 30; brightness: 0.35 }
        DirectionalLight { eulerRotation.x: 60; eulerRotation.y: 200; brightness: 0.15 }
        Node { id: parts }
    }
}
"""


def smooth_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (used when a caller passes no normals)."""
    p = np.asarray(positions, dtype=np.float64)
    tri = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    n = np.zeros_like(p)
    if len(tri):
        fn = np.cross(p[tri[:, 1]] - p[tri[:, 0]], p[tri[:, 2]] - p[tri[:, 0]])
        for k in range(3):
            np.add.at(n, tri[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 0, n / np.maximum(ln, 1e-30), [0.0, 1.0, 0.0])
    return n.astype(np.float32)


if HAVE_QUICK3D:
    class MeshGeometry(QQuick3DGeometry):
        """Interleaved position/normal/uv triangle geometry fed from numpy."""

        def __init__(self, positions: np.ndarray, indices: np.ndarray, normals: np.ndarray, uv: np.ndarray | None):
            super().__init__()
            A = QQuick3DGeometry.Attribute
            n = len(positions)
            cols = [positions.astype(np.float32, copy=False), normals.astype(np.float32, copy=False)]
            if uv is not None:
                cols.append(uv.astype(np.float32, copy=False))
            data = np.ascontiguousarray(np.hstack(cols), dtype=np.float32) if n else np.zeros((0, 8), np.float32)
            self.setVertexData(QByteArray(data.tobytes()))
            self.setStride(data.shape[1] * 4 if n else 24)
            self.setIndexData(QByteArray(np.ascontiguousarray(indices, dtype=np.uint32).tobytes()))
            self.setPrimitiveType(QQuick3DGeometry.PrimitiveType.Triangles)
            self.addAttribute(A.PositionSemantic, 0, A.F32Type)
            self.addAttribute(A.NormalSemantic, 12, A.F32Type)
            if uv is not None:
                self.addAttribute(A.TexCoord0Semantic, 24, A.F32Type)
            self.addAttribute(A.IndexSemantic, 0, A.U32Type)
            if n:
                lo, hi = positions.min(axis=0), positions.max(axis=0)
                self.setBounds(QVector3D(*map(float, lo)), QVector3D(*map(float, hi)))

    class ImageTexture(QQuick3DTextureData):
        """RGBA8 texture data from a QImage."""

        def __init__(self, image: QImage, transparent: bool = False):
            super().__init__()
            img = image.convertToFormat(QImage.Format_RGBA8888)
            self.setSize(img.size())
            self.setFormat(QQuick3DTextureData.Format.RGBA8)
            self.setHasTransparency(transparent)
            self.setTextureData(QByteArray(bytes(img.constBits())[:img.sizeInBytes()]))


@dataclass
class _Part:
    key: str
    lo: np.ndarray
    hi: np.ndarray
    visible: bool = True
    color: tuple = (0.7, 0.7, 0.7)
    geom: object = None
    tex: object = None
    obj: object = None
    alpha_mode: str = "opaque"
    cutoff: float = 0.5


class MeshView(QWidget):
    """3D preview of any number of triangle meshes, addressed by string keys."""
    status = Signal(str)          # one-line render/camera status

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parts: dict[str, _Part] = {}
        self._yaw, self._pitch, self._dist = 35.0, -20.0, 5.0
        self._target = np.zeros(3)
        self._radius = 1.0
        self._drag: tuple[str, QPointF] | None = None
        self._wire = False
        self._shading = 0
        self._flip_v = False
        self.error: str | None = None
        self.quick: QQuickWidget | None = None
        self.root = None

        self._stack = QStackedLayout()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(self._stack)
        self._msg = QLabel()
        self._msg.setAlignment(Qt.AlignCenter)
        self._msg.setWordWrap(True)
        self._msg.setStyleSheet("color: #aaa; padding: 20px;")
        self._stack.addWidget(self._msg)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(200, 150)
        self._init_quick()

    # ---- setup ------------------------------------------------------------------------------------------------
    def _init_quick(self) -> None:
        if not HAVE_QUICK3D:
            self._fail(f"Qt Quick 3D is not available ({_IMPORT_ERROR}).")
            return
        plat = QGuiApplication.platformName().lower()
        if plat in NO_3D_PLATFORMS and not os.environ.get("NIGHTRUNNER_FORCE_3D"):
            self._fail(f"3D preview is not available on the '{plat}' Qt platform.")
            return
        api = QQuickWindow.graphicsApi()
        if not QSGRendererInterface.isApiRhiBased(api) and api != QSGRendererInterface.GraphicsApi.Unknown:
            self._fail(f"3D preview needs a GPU scene graph (current: {api.name}).")
            return
        try:
            w = QQuickWidget(self)
            w.setResizeMode(QQuickWidget.SizeRootObjectToView)
            w.setFocusPolicy(Qt.StrongFocus)
            comp = QQmlComponent(w.engine())
            comp.setData(QML, QUrl("file:///nightrunner/meshview.qml"))
            if comp.status() != QQmlComponent.Ready:
                raise RuntimeError("; ".join(e.toString() for e in comp.errors()))
            root = comp.create()
            if root is None:
                raise RuntimeError("; ".join(e.toString() for e in comp.errors()) or "QML root not created")
            w.setContent(QUrl("file:///nightrunner/meshview.qml"), comp, root)
            self._comp = comp
        except Exception as exc:  # noqa: BLE001
            self._fail(f"3D preview could not start: {exc}")
            return
        self.quick, self.root = w, root
        w.installEventFilter(self)
        w.sceneGraphError.connect(lambda _e, msg: self._fail(f"3D rendering error: {msg}"))
        qw = w.quickWindow()
        if qw is not None:
            qw.sceneGraphInitialized.connect(self._check_api)
        self._stack.addWidget(w)
        self._stack.setCurrentWidget(w)
        self._apply_camera()

    def _check_api(self) -> None:
        qw = self.quick.quickWindow() if self.quick else None
        ri = qw.rendererInterface() if qw else None
        if ri is not None and not QSGRendererInterface.isApiRhiBased(ri.graphicsApi()):
            self._fail("3D preview needs a GPU scene graph (Qt Quick fell back to software rendering).")

    def _fail(self, msg: str) -> None:
        self.error = msg
        self._msg.setText(msg + "\n\nMesh details, export and the other tabs still work.")
        self._stack.setCurrentWidget(self._msg)

    @property
    def available(self) -> bool:
        """True while a Qt Quick 3D scene is live (False → message label shown)."""
        return self.root is not None and self.error is None

    def set_message(self, text: str) -> None:
        """Show *text* over the view (empty = back to the 3D scene)."""
        if text or not self.available:
            self._msg.setText(text or (self.error or ""))
            self._stack.setCurrentWidget(self._msg)
        elif self.quick is not None:
            self._stack.setCurrentWidget(self.quick)

    # ---- contract -------------------------------------------------------------------------------------------
    def clear(self) -> None:
        for p in self._parts.values():
            if p.obj is not None:
                try:
                    p.obj.setParent(None)
                    p.obj.setProperty("visible", False)
                    p.obj.deleteLater()
                except RuntimeError:
                    pass
        self._parts.clear()

    def add_mesh(self, key: str, positions, indices, normals=None, uv=None, color=(0.7, 0.7, 0.7),
                 texture: QImage | None = None, alpha_mode: str = "opaque", alpha_cutoff: float = 0.5) -> None:
        """Add (or replace) a triangle mesh. *color* is RGB 0..1; *texture* is used as the base colour map.
        *alpha_mode* ("opaque" | "mask" | "blend") applies the texture alpha (see gui/matpreview.py)."""
        if key in self._parts:
            self.remove(key)
        pos = np.ascontiguousarray(np.asarray(positions, dtype=np.float32).reshape(-1, 3))
        idx = np.ascontiguousarray(np.asarray(indices, dtype=np.uint32).reshape(-1))
        idx = idx[:len(idx) - len(idx) % 3]
        if len(pos):
            finite = np.isfinite(pos).all(axis=1)
            if not finite.all():
                pos = pos.copy()
                pos[~finite] = 0.0
            lo, hi = pos.min(axis=0).astype(np.float64), pos.max(axis=0).astype(np.float64)
        else:
            lo = hi = np.zeros(3)
        if len(idx) and int(idx.max()) >= len(pos):
            idx = idx.reshape(-1, 3)
            idx = idx[(idx < len(pos)).all(axis=1)].reshape(-1)
        part = _Part(key, lo, hi, True, tuple(float(c) for c in color), alpha_mode=alpha_mode,
                     cutoff=float(alpha_cutoff))
        self._parts[key] = part
        if not self.available or not len(pos):
            return
        nrm = None if normals is None else np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        if nrm is None or len(nrm) != len(pos):
            nrm = smooth_normals(pos, idx)
        tuv = None if uv is None else np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        if tuv is not None and len(tuv) != len(pos):
            tuv = None
        part.geom = MeshGeometry(pos, idx, nrm, tuv)
        part.tex = (ImageTexture(texture, alpha_mode != "opaque")
                    if texture is not None and not texture.isNull() and tuv is not None else None)
        part.obj = QMetaObject.invokeMethod(self.root, "addPart", Q_RETURN_ARG("QVariant"),
                                            Q_ARG("QVariant", part.geom), Q_ARG("QVariant", self._qcolor(part.color)),
                                            Q_ARG("QVariant", part.tex), Q_ARG("QVariant", _ALPHA.get(alpha_mode, 0)),
                                            Q_ARG("QVariant", float(alpha_cutoff)))
        if part.obj is None:
            self._fail("3D preview: the QML scene refused a mesh part.")

    def remove(self, key: str) -> None:
        p = self._parts.pop(key, None)
        if p is not None and p.obj is not None:
            try:
                p.obj.setProperty("visible", False)
                p.obj.deleteLater()
            except RuntimeError:
                pass

    def set_visible(self, key: str, on: bool) -> None:
        p = self._parts.get(key)
        if p is None:
            return
        p.visible = bool(on)
        if p.obj is not None:
            p.obj.setProperty("visible", p.visible)

    def set_color(self, key: str, color) -> None:
        p = self._parts.get(key)
        if p is None:
            return
        p.color = tuple(float(c) for c in color)
        if p.obj is not None:
            p.obj.setProperty("col", self._qcolor(p.color))

    def set_texture(self, key: str, texture: QImage | None, alpha_mode: str | None = None,
                    alpha_cutoff: float | None = None) -> None:
        """Replace (or with None remove) a part's base colour map. Ignored for parts added without UVs.
        *alpha_mode* / *alpha_cutoff* default to the part's current values."""
        p = self._parts.get(key)
        if p is None or p.obj is None:
            return
        if alpha_mode is not None:
            p.alpha_mode = alpha_mode
        if alpha_cutoff is not None:
            p.cutoff = float(alpha_cutoff)
        g = p.geom
        if texture is None or texture.isNull():
            p.tex = None
            p.alpha_mode = "opaque"
        elif g is not None and g.attributeCount() >= 4:
            p.tex = ImageTexture(texture, p.alpha_mode != "opaque")
        else:
            return
        p.obj.setProperty("alphaModeI", _ALPHA.get(p.alpha_mode, 0))
        p.obj.setProperty("cutoffV", p.cutoff)
        p.obj.setProperty("texData", p.tex)

    def keys(self) -> list[str]:
        return list(self._parts)

    def is_visible(self, key: str) -> bool:
        p = self._parts.get(key)
        return bool(p and p.visible)

    def bounds(self, visible_only: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        ps = [p for p in self._parts.values() if p.visible or not visible_only]
        if not ps:
            return None
        return np.min([p.lo for p in ps], axis=0), np.max([p.hi for p in ps], axis=0)

    def frame_all(self) -> None:
        b = self.bounds() or self.bounds(False)
        if b is None:
            self._target = np.zeros(3)
            self._radius = 1.0
        else:
            lo, hi = b
            self._target = (lo + hi) / 2
            self._radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-3)
        self._dist = self._radius / math.sin(math.radians(22.5)) * 0.9
        self._apply_camera()

    def reset_view(self) -> None:
        self._yaw, self._pitch = 35.0, -20.0
        self.frame_all()

    def screenshot(self) -> QImage:
        if self.available and self.quick is not None:
            img = self.quick.grabFramebuffer()
            if not img.isNull():
                return img
        return self.grab().toImage()

    # ---- display options --------------------------------------------------------------------------------------
    def set_wireframe(self, on: bool) -> None:
        self._wire = bool(on)
        if self.root is not None:
            self.root.setProperty("wireframe", self._wire)

    def wireframe(self) -> bool:
        return self._wire

    def set_shading(self, mode: int) -> None:
        """0 lit colour/texture, 1 normals, 2 unlit colour/texture."""
        self._shading = int(mode)
        if self.root is not None:
            self.root.setProperty("shading", self._shading)

    def set_flip_v(self, on: bool) -> None:
        self._flip_v = bool(on)
        if self.root is not None:
            self.root.setProperty("flipV", self._flip_v)

    def set_background(self, color: QColor) -> None:
        if self.root is not None:
            self.root.setProperty("background", color)

    # ---- camera -----------------------------------------------------------------------------------------------
    def camera(self) -> dict:
        return {"yaw": self._yaw, "pitch": self._pitch, "distance": self._dist, "target": self._target.tolist()}

    def _rotation(self) -> QQuaternion:
        return QQuaternion.fromEulerAngles(self._pitch, self._yaw, 0.0)

    def _apply_camera(self) -> None:
        if self.root is None:
            return
        q = self._rotation()
        off = q.rotatedVector(QVector3D(0, 0, self._dist))
        t = self._target
        self.root.setProperty("camPos", QVector3D(float(t[0]) + off.x(), float(t[1]) + off.y(), float(t[2]) + off.z()))
        self.root.setProperty("camRot", q)
        r = max(self._radius, 1e-3)
        near = max(self._dist - 3 * r, self._dist * 0.002, r * 1e-4)
        self.root.setProperty("near", float(near))
        self.root.setProperty("far", float(self._dist + 6 * r))

    def orbit(self, dx: float, dy: float) -> None:
        self._yaw = (self._yaw - dx * 0.4) % 360.0
        self._pitch = max(-89.0, min(89.0, self._pitch - dy * 0.4))
        self._apply_camera()

    def pan(self, dx: float, dy: float) -> None:
        q = self._rotation()
        right = q.rotatedVector(QVector3D(1, 0, 0))
        up = q.rotatedVector(QVector3D(0, 1, 0))
        h = max(self.height(), 1)
        k = 2 * self._dist * math.tan(math.radians(22.5)) / h
        self._target = self._target + (-dx * k) * np.array([right.x(), right.y(), right.z()]) \
            + (dy * k) * np.array([up.x(), up.y(), up.z()])
        self._apply_camera()

    def zoom(self, steps: float) -> None:
        self._dist = max(self._radius * 1e-3, min(self._radius * 1e3, self._dist * (0.85 ** steps)))
        self._apply_camera()

    # ---- input ------------------------------------------------------------------------------------------------
    def eventFilter(self, obj: QObject, ev: QEvent) -> bool:
        if obj is not self.quick:
            return False
        t = ev.type()
        if t == QEvent.MouseButtonPress:
            b = ev.button()
            if b == Qt.LeftButton and not (ev.modifiers() & Qt.ShiftModifier):
                self._drag = ("orbit", ev.position())
            elif b in (Qt.MiddleButton, Qt.LeftButton):
                self._drag = ("pan", ev.position())
            elif b == Qt.RightButton:
                self._drag = ("zoom", ev.position())
            self.quick.setFocus()
            return True
        if t == QEvent.MouseMove and self._drag is not None:
            mode, last = self._drag
            p = ev.position()
            dx, dy = p.x() - last.x(), p.y() - last.y()
            self._drag = (mode, p)
            if mode == "orbit":
                self.orbit(dx, dy)
            elif mode == "pan":
                self.pan(dx, dy)
            else:
                self.zoom(-dy / 40.0)
            return True
        if t == QEvent.MouseButtonRelease:
            self._drag = None
            return True
        if t == QEvent.MouseButtonDblClick:
            self.frame_all()
            return True
        if t == QEvent.Wheel:
            self.zoom(ev.angleDelta().y() / 120.0)
            return True
        if t == QEvent.KeyPress and self._key(ev.key()):
            return True
        return False

    def keyPressEvent(self, ev) -> None:
        if not self._key(ev.key()):
            super().keyPressEvent(ev)

    def _key(self, key: int) -> bool:
        if key == Qt.Key_F:
            self.frame_all()
        elif key == Qt.Key_W:
            self.set_wireframe(not self._wire)
        elif key == Qt.Key_N:
            self.set_shading(0 if self._shading == 1 else 1)
        elif key == Qt.Key_R:
            self.reset_view()
        else:
            return False
        return True

    @staticmethod
    def _qcolor(c) -> QColor:
        return QColor.fromRgbF(*(max(0.0, min(1.0, float(x))) for x in c[:3]))


def palette_color(i: int) -> tuple[float, float, float]:
    """Distinct, muted colours for submesh colouring (golden-angle hue walk)."""
    c = QColor.fromHsvF(((i * 0.618034) + 0.08) % 1.0, 0.45, 0.85)
    return (c.redF(), c.greenF(), c.blueF())

