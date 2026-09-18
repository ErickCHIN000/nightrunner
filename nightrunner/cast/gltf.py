"""glTF 2.0 (.gltf / .glb) ↔ the in-memory Cast scene every nightrunner exporter builds and every importer reads.

Writers keep producing a `castlib.Cast`; `save_scene(cast, path)` writes Cast, glTF or GLB by extension.
Readers call `load_scene(path)`, which returns a `castlib.Cast` for any of the three, so `cast/import_.py` (per-mesh
re-import) and `cast/split.py` (single model Cast) accept glTF / GLB unchanged.

Mapping (Cast → glTF):

* coordinates: the numbers are written unchanged (native engine space, Y up — glTF's up axis; no unit or handedness
  conversion, exactly like the Cast files, so a round trip is lossless);
* Skeleton → one node per bone (local TRS from `lp` / `lr`, scale 1), parent links from `p`; one `skin` per file
  whose `joints` are the bones in Cast order and whose inverse bind matrices are the inverse rest globals composed
  from those local TRS (so a viewer shows the mesh exactly at its bind pose);
* Mesh → one node + one mesh with one triangle primitive (node and mesh named like the Cast mesh):
  POSITION, NORMAL, TANGENT (w = `bp_tangent_sign`, default +1), TEXCOORD_0/1, JOINTS_0 + WEIGHTS_0 (4 lanes,
  weights renormalised to sum 1 for glTF validators), indices uint32, and the custom attribute `_BP_VERTEX_ID`
  (float, exact for < 2^24) when the Cast carries `bp_vertex_id`; every other `bp_*` property → mesh `extras`;
* Material → PBR material (baseColorTexture from the `albedo` slot, normalTexture from `normal`, metallic 0,
  roughness 0.75, double-sided); `bp_alpha_mode` mask/blend → glTF alphaMode MASK (cutoff 0.5, the exported alpha
  is already hard 0/1) / BLEND — viewers and Blender's glTF importer need no helper script; `bp_*` → `extras`;
* images: `.gltf` references the PNG files by relative URI; `.glb` embeds them.

Mapping (glTF → Cast) for re-import: the first skin gives the skeleton (joint order = bone order, parents from the
node tree, local TRS from the joint nodes); every node with a mesh gives one Cast mesh per triangle primitive
(named after the node; additional primitives get `#<n>`); `extras` and `_BP_VERTEX_ID` come back as `bp_*`
properties; TANGENT.w becomes `bp_tangent_sign`; JOINTS_n / WEIGHTS_n sets are concatenated. Unskinned meshes are
transformed by their node's world matrix; skinned meshes are read in bind space as stored (glTF ignores their node
transform).
"""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from . import castlib

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT, UINT, USHORT, UBYTE = 5126, 5125, 5123, 5121
_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
EXTENSIONS = (".cast", ".gltf", ".glb")


class GltfError(ValueError):
    pass


# ---- dispatch -----------------------------------------------------------------------------------------------------

def load_scene(path: Path | str) -> castlib.Cast:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".gltf", ".glb"):
        return gltf_to_cast(p)
    return castlib.Cast.load(str(p))


def save_scene(cast: castlib.Cast, path: Path | str, base_dir: Path | None = None) -> None:
    """Write *cast* as Cast / glTF / GLB by extension. Material File paths are relative to *base_dir* (default:
    the output folder)."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".cast":
        cast.save(str(p))
    elif ext in (".gltf", ".glb"):
        cast_to_gltf(cast, p, base_dir=base_dir)
    else:
        raise GltfError(f"unknown scene extension {ext!r} (use {', '.join(EXTENSIONS)})")


# ---- math ---------------------------------------------------------------------------------------------------------

def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-20:
        return np.eye(3)
    s = 2.0 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def _trs(t, q, s=(1.0, 1.0, 1.0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quat_to_mat(q) * np.asarray(s, dtype=np.float64)
    m[:3, 3] = np.asarray(t, dtype=np.float64)
    return m


def _mat_to_quat(r: np.ndarray) -> list[float]:
    from .export import quaternion_xyzw
    return quaternion_xyzw(r)


def _node_matrix(n: dict) -> np.ndarray:
    if "matrix" in n:
        return np.asarray(n["matrix"], dtype=np.float64).reshape(4, 4).T
    return _trs(n.get("translation", (0, 0, 0)), n.get("rotation", (0, 0, 0, 1)), n.get("scale", (1, 1, 1)))


# ---- writer -------------------------------------------------------------------------------------------------------

class _Buf:
    def __init__(self):
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, raw: bytes, target: int | None = None) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        v = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target:
            v["target"] = target
        self.data += raw
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr: np.ndarray, ctype: int, kind: str, target: int | None = None, minmax=False) -> int:
        arr = np.ascontiguousarray(arr, dtype=_COMP[ctype])
        a = {"bufferView": self.view(arr.tobytes(), target), "componentType": ctype,
             "count": int(arr.shape[0]), "type": kind}
        if minmax and len(arr):
            a["min"] = [float(x) for x in arr.reshape(len(arr), -1).min(axis=0)]
            a["max"] = [float(x) for x in arr.reshape(len(arr), -1).max(axis=0)]
        self.accessors.append(a)
        return len(self.accessors) - 1


def _unit(values) -> np.ndarray:
    """float32 rows; rows already unit to 1e-5 are kept bit-exact (decoded qtangent frames), others normalised."""
    v32 = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    ln = np.linalg.norm(v32.astype(np.float64), axis=1, keepdims=True)
    fixed = (v32 / np.maximum(ln, 1e-12)).astype(np.float32)
    return np.where(np.abs(ln - 1) <= 1e-5, v32, fixed)


def _props(node, prefix="bp_") -> dict:
    out = {}
    for k, p in node.properties.items():
        if k.startswith(prefix):
            v = list(p.values)
            out[k] = v[0] if len(v) == 1 else v
    return out


def cast_to_gltf(cast: castlib.Cast, path: Path, base_dir: Path | None = None) -> None:
    path = Path(path)
    binary = path.suffix.lower() == ".glb"
    base_dir = Path(base_dir) if base_dir else path.parent
    root = cast.Roots()[0]
    mdl = root.ChildOfType(castlib.Model)
    if mdl is None:
        raise GltfError("the scene has no Model")
    buf = _Buf()
    nodes: list[dict] = []
    doc: dict[str, Any] = {"asset": {"version": "2.0", "generator": _generator(root)},
                           "scene": 0, "scenes": [{"name": mdl.Name() or "model", "nodes": []}]}
    top = doc["scenes"][0]["nodes"]

    # skeleton
    joints: list[int] = []
    skel = mdl.Skeleton()
    bones = skel.Bones() if skel is not None else []
    if bones:
        glob = []
        for i, b in enumerate(bones):
            lp = b.LocalPosition() or (0, 0, 0)
            lr = b.LocalRotation() or (0, 0, 0, 1)
            q = np.asarray(lr, dtype=np.float64)
            q = q / max(float(np.linalg.norm(q)), 1e-12)
            node = {"name": b.Name() or f"bone_{i}", "translation": [float(x) for x in lp],
                    "rotation": [float(np.clip(round(float(x), 9), -1.0, 1.0)) for x in q]}
            ex = _props(b)
            if ex:
                node["extras"] = ex
            nodes.append(node)
            joints.append(len(nodes) - 1)
        parents = [int(b.ParentIndex()) for b in bones]
        # one non-joint root node holds every root bone: glTF skins need a common root
        nodes.append({"name": (mdl.Name() or "model") + "_armature", "children": []})
        arm = len(nodes) - 1
        top.append(arm)
        for i, p in enumerate(parents):
            if 0 <= p < len(bones):
                nodes[joints[p]].setdefault("children", []).append(joints[i])
            else:
                nodes[arm]["children"].append(joints[i])
        glob = [None] * len(bones)

        def world(i):
            if glob[i] is None:
                n = nodes[joints[i]]
                loc = _trs(n["translation"], n["rotation"])
                p = parents[i]
                glob[i] = world(p) @ loc if 0 <= p < len(bones) and p != i else loc
            return glob[i]

        ibm = np.stack([np.linalg.inv(world(i)).T for i in range(len(bones))]).astype(np.float32)
        doc["skins"] = [{"name": (mdl.Name() or "model") + "_skin", "joints": joints,
                         "inverseBindMatrices": buf.accessor(ibm.reshape(-1, 16), FLOAT, "MAT4"),
                         "skeleton": arm}]

    # materials + images
    images: list[dict] = []
    image_index: dict[str, int] = {}
    textures: list[dict] = []
    materials: list[dict] = []
    mat_index: dict[int, int] = {}

    def texture(file_node) -> int | None:
        rel = file_node.Path() if file_node is not None else None
        if not rel:
            return None
        if rel not in image_index:
            src = (base_dir / rel)
            img: dict[str, Any] = {"name": Path(rel).stem}
            if binary:
                if not src.is_file():
                    return None
                img["bufferView"] = buf.view(src.read_bytes())
                img["mimeType"] = "image/png" if src.suffix.lower() == ".png" else "image/jpeg"
            else:
                import os
                from urllib.parse import quote
                img["uri"] = quote(Path(os.path.relpath(src, path.parent)).as_posix())
            images.append(img)
            textures.append({"source": len(images) - 1, "sampler": 0})
            image_index[rel] = len(textures) - 1
        return image_index[rel]

    for m in mdl.Materials():
        slots = m.Slots()
        g: dict[str, Any] = {"name": m.Name() or "material", "doubleSided": True,
                             "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.75}}
        t = texture(slots.get("albedo") or slots.get("diffuse"))
        if t is not None:
            g["pbrMetallicRoughness"]["baseColorTexture"] = {"index": t}
        t = texture(slots.get("normal"))
        if t is not None:
            g["normalTexture"] = {"index": t}
        ex = _props(m)
        mode = ex.get("bp_alpha_mode")
        if mode == "mask":
            g["alphaMode"], g["alphaCutoff"] = "MASK", 0.5
        elif mode == "blend":
            g["alphaMode"] = "BLEND"
        if ex:
            g["extras"] = ex
        materials.append(g)
        mat_index[m.Hash()] = len(materials) - 1

    # meshes
    meshes: list[dict] = []
    for k, m in enumerate(mdl.Meshes()):
        name = m.Name() or f"mesh_{k}"
        pos = np.asarray(m.VertexPositionBuffer() or [], dtype=np.float32).reshape(-1, 3)
        n = len(pos)
        if n == 0:
            continue
        attrs: dict[str, int] = {"POSITION": buf.accessor(pos, FLOAT, "VEC3", 34962, minmax=True)}
        nrm = m.VertexNormalBuffer()
        if nrm:
            attrs["NORMAL"] = buf.accessor(_unit(nrm), FLOAT, "VEC3", 34962)
        tan = m.VertexTangentBuffer()
        if tan:
            t = _unit(tan)
            sign = m.properties.get("bp_tangent_sign")
            w = np.ones(n) if sign is None or len(sign.values) != n else \
                np.where(np.asarray(sign.values, dtype=np.float64) < 0, -1.0, 1.0)
            attrs["TANGENT"] = buf.accessor(np.column_stack([t, w.astype(np.float32)]), FLOAT, "VEC4", 34962)
        nonfinite = {}
        for layer in range(int(m.UVLayerCount() or 0)):
            uv = m.VertexUVLayerBuffer(layer)
            if uv:
                a = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
                bad = ~np.isfinite(a)
                if bad.any():
                    # glTF forbids inf / NaN (shipped meshes carry some): write 0 and keep the exact values in extras
                    rows = np.flatnonzero(bad.any(axis=1))
                    nonfinite[str(layer)] = {"rows": rows.tolist(), "hex": a[rows].tobytes().hex()}
                    a = np.where(bad, 0.0, a)
                attrs[f"TEXCOORD_{layer}"] = buf.accessor(a, FLOAT, "VEC2", 34962)
        mi = int(m.MaximumWeightInfluence() or 0)
        skinned = bool(mi and bones and m.VertexWeightBoneBuffer())
        if skinned:
            wb = np.asarray(m.VertexWeightBoneBuffer(), dtype=np.int64).reshape(n, mi)
            wv = np.asarray(m.VertexWeightValueBuffer(), dtype=np.float64).reshape(n, mi)
            if mi > 4:
                order = np.argsort(-wv, axis=1)[:, :4]
                wb, wv = np.take_along_axis(wb, order, 1), np.take_along_axis(wv, order, 1)
            elif mi < 4:
                wb = np.pad(wb, ((0, 0), (0, 4 - mi)))
                wv = np.pad(wv, ((0, 0), (0, 4 - mi)))
            wb = np.where(wv > 0, wb, 0)
            wv32 = wv.astype(np.float32)
            s = wv32.astype(np.float64).sum(axis=1, keepdims=True)
            # rows already summing to 1 (every shipped row: raw/255) are written bit-exact; others renormalised
            wv = np.where(np.abs(s - 1) <= 1e-5, wv32,
                          np.where(s > 0, wv / np.maximum(s, 1e-12), np.array([[1.0, 0, 0, 0]])))
            ctype = UBYTE if len(bones) <= 256 else USHORT
            attrs["JOINTS_0"] = buf.accessor(wb, ctype, "VEC4", 34962)
            attrs["WEIGHTS_0"] = buf.accessor(wv, FLOAT, "VEC4", 34962)
        vid = m.properties.get("bp_vertex_id")
        if vid is not None and len(vid.values) == n:
            attrs["_BP_VERTEX_ID"] = buf.accessor(np.asarray(vid.values, dtype=np.float32), FLOAT, "SCALAR", 34962)
        faces = np.asarray(m.FaceBuffer() or [], dtype=np.uint32)
        prim: dict[str, Any] = {"attributes": attrs, "mode": 4,
                                "indices": buf.accessor(faces, UINT, "SCALAR", 34963)}
        mh = (m.properties.get("m").values[0] if m.properties.get("m") else None)
        if mh in mat_index:
            prim["material"] = mat_index[mh]
        gm: dict[str, Any] = {"name": name, "primitives": [prim]}
        ex = {k2: v2 for k2, v2 in _props(m).items() if k2 not in ("bp_vertex_id", "bp_tangent_sign")}
        if nonfinite:
            ex["bp_nonfinite_uv"] = nonfinite
        if ex:
            gm["extras"] = ex
        meshes.append(gm)
        node = {"name": name, "mesh": len(meshes) - 1}
        if skinned:
            node["skin"] = 0
        nodes.append(node)
        top.append(len(nodes) - 1)

    doc["nodes"] = nodes
    if meshes:
        doc["meshes"] = meshes
    if materials:
        doc["materials"] = materials
    if textures:
        doc["textures"] = textures
        doc["images"] = images
        doc["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    doc["accessors"] = buf.accessors
    doc["bufferViews"] = buf.views
    while len(buf.data) % 4:
        buf.data.append(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    if binary:
        doc["buffers"] = [{"byteLength": len(buf.data)}]
        js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        js += b" " * (-len(js) % 4)
        total = 12 + 8 + len(js) + 8 + len(buf.data)
        with open(tmp, "wb") as fh:
            fh.write(struct.pack("<III", GLB_MAGIC, 2, total))
            fh.write(struct.pack("<II", len(js), CHUNK_JSON) + js)
            fh.write(struct.pack("<II", len(buf.data), CHUNK_BIN) + bytes(buf.data))
    else:
        bin_path = path.with_suffix(".bin")
        bin_path.write_bytes(bytes(buf.data))
        doc["buffers"] = [{"byteLength": len(buf.data), "uri": bin_path.name}]
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(path)


def _generator(root) -> str:
    meta = root.ChildOfType(castlib.Metadata)
    return (meta.Software() if meta is not None and meta.Software() else "nightrunner")


# ---- reader -------------------------------------------------------------------------------------------------------

def _load(path: Path) -> tuple[dict, list[bytes]]:
    raw = path.read_bytes()
    if raw[:4] == struct.pack("<I", GLB_MAGIC):
        _magic, _ver, _total = struct.unpack_from("<III", raw, 0)
        off, doc, chunks = 12, None, []
        while off + 8 <= len(raw):
            ln, kind = struct.unpack_from("<II", raw, off)
            body = raw[off + 8: off + 8 + ln]
            if kind == CHUNK_JSON:
                doc = json.loads(body.decode("utf-8"))
            elif kind == CHUNK_BIN:
                chunks.append(body)
            off += 8 + ln
        if doc is None:
            raise GltfError(f"{path}: GLB without a JSON chunk")
        bufs = []
        for i, b in enumerate(doc.get("buffers", [])):
            if "uri" in b:
                bufs.append(_uri(path, b["uri"]))
            else:
                bufs.append(chunks[0] if chunks else b"")
        return doc, bufs
    doc = json.loads(raw.decode("utf-8"))
    return doc, [_uri(path, b.get("uri", "")) for b in doc.get("buffers", [])]


def _uri(path: Path, uri: str) -> bytes:
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    from urllib.parse import unquote
    return (path.parent / unquote(uri)).read_bytes()


def _accessor(doc: dict, bufs: list[bytes], idx: int) -> np.ndarray:
    a = doc["accessors"][idx]
    dt = np.dtype(_COMP[a["componentType"]])
    nc = _NCOMP[a["type"]]
    count = a["count"]
    if "sparse" in a:
        raise GltfError("sparse accessors are not supported")
    if "bufferView" not in a:
        out = np.zeros((count, nc), dtype=dt)
    else:
        v = doc["bufferViews"][a["bufferView"]]
        data = bufs[v["buffer"]]
        start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = v.get("byteStride") or dt.itemsize * nc
        if stride == dt.itemsize * nc:
            out = np.frombuffer(data, dtype=dt, count=count * nc, offset=start).reshape(count, nc)
        else:
            rows = [np.frombuffer(data, dtype=dt, count=nc, offset=start + i * stride) for i in range(count)]
            out = np.stack(rows) if rows else np.zeros((0, nc), dtype=dt)
    if a.get("normalized") and dt.kind in "iu":
        out = out.astype(np.float64) / np.iinfo(dt).max
    return out


def gltf_to_cast(path: Path | str) -> castlib.Cast:
    path = Path(path)
    doc, bufs = _load(path)
    nodes = doc.get("nodes", [])
    parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []):
            parent[c] = i

    def world(i):
        m = _node_matrix(nodes[i])
        while i in parent:
            i = parent[i]
            m = _node_matrix(nodes[i]) @ m
        return m

    cast = castlib.Cast()
    root = cast.CreateRoot()
    meta = root.CreateMetadata()
    meta.SetUpAxis("y")
    meta.SetSoftware(str((doc.get("asset") or {}).get("generator") or "glTF"))
    mdl = root.CreateModel()
    scenes = doc.get("scenes") or [{}]
    mdl.SetName(scenes[doc.get("scene", 0)].get("name") or path.stem)

    skins = doc.get("skins") or []
    joints = skins[0]["joints"] if skins else []
    joint_pos = {j: k for k, j in enumerate(joints)}
    if joints:
        sk = mdl.CreateSkeleton()
        for k, j in enumerate(joints):
            n = nodes[j]
            b = sk.CreateBone()
            b.SetName(n.get("name") or f"joint_{k}")
            p = parent.get(j)
            while p is not None and p not in joint_pos:     # skip non-joint helper nodes (armature objects)
                p = parent.get(p)
            b.SetParentIndex(joint_pos[p] if p is not None else -1)
            m = _node_matrix(n)
            if p is None and j in parent:                   # root joint under a helper: bake the helper's transform
                m = world(j)
            t = m[:3, 3]
            sc = np.linalg.norm(m[:3, :3], axis=0)
            r = m[:3, :3] / np.where(sc > 1e-12, sc, 1.0)
            b.SetLocalPosition([float(x) for x in t])
            b.SetLocalRotation(_mat_to_quat(r))
            b.SetScale([float(x) for x in sc])
            b.SetSegmentScaleCompensate(False)
            w = world(j)
            ws = np.linalg.norm(w[:3, :3], axis=0)
            b.SetWorldPosition([float(x) for x in w[:3, 3]])
            b.SetWorldRotation(_mat_to_quat(w[:3, :3] / np.where(ws > 1e-12, ws, 1.0)))
            for key, val in (n.get("extras") or {}).items():
                if key.startswith("bp_"):
                    _set_prop(b, key, val)

    mat_hash: dict[int, int] = {}
    for i, g in enumerate(doc.get("materials", [])):
        mn = mdl.CreateMaterial()
        mn.SetName(g.get("name") or f"material_{i}")
        mn.SetType("pbr")
        for key, val in (g.get("extras") or {}).items():
            if key.startswith("bp_"):
                _set_prop(mn, key, val)
        mat_hash[i] = mn.Hash()

    for ni, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        gm = doc["meshes"][n["mesh"]]
        name = n.get("name") or gm.get("name") or f"mesh_{ni}"
        skinned = "skin" in n
        xf = None if skinned else world(ni)
        # the node's skin may differ from skin 0: remap its joints into skin 0's order by node index
        jmap = None
        if skinned and n["skin"] != 0:
            jmap = np.array([joint_pos.get(j, 0) for j in skins[n["skin"]]["joints"]], dtype=np.int64)
        for pi, prim in enumerate(gm.get("primitives", [])):
            if prim.get("mode", 4) != 4:
                continue
            at = prim["attributes"]
            pos = _accessor(doc, bufs, at["POSITION"]).astype(np.float64)
            nrm = _accessor(doc, bufs, at["NORMAL"]).astype(np.float64) if "NORMAL" in at else None
            tan = _accessor(doc, bufs, at["TANGENT"]).astype(np.float64) if "TANGENT" in at else None
            if xf is not None:
                pos = pos @ xf[:3, :3].T + xf[:3, 3]
                if nrm is not None:
                    nrm = nrm @ np.linalg.inv(xf[:3, :3])
                if tan is not None:
                    tan[:, :3] = tan[:, :3] @ xf[:3, :3].T
            idx = _accessor(doc, bufs, prim["indices"]).reshape(-1) if "indices" in prim else np.arange(len(pos))
            me = mdl.CreateMesh()
            me.SetName(name if pi == 0 else f"{name}#{pi}")
            me.SetVertexPositionBuffer(pos.astype(np.float32).tolist())
            if nrm is not None:
                me.SetVertexNormalBuffer(nrm.astype(np.float32).tolist())
            if tan is not None:
                me.SetVertexTangentBuffer(tan[:, :3].astype(np.float32).tolist())
                me.CreateProperty("bp_tangent_sign", "f").values = [float(x) for x in np.where(tan[:, 3] < 0, -1, 1)]
            layers = 0
            nonfinite = (gm.get("extras") or {}).get("bp_nonfinite_uv") or {}
            while f"TEXCOORD_{layers}" in at:
                uv = _accessor(doc, bufs, at[f"TEXCOORD_{layers}"]).astype(np.float32).copy()
                nf = nonfinite.get(str(layers)) if isinstance(nonfinite, dict) else None
                if nf and pi == 0:
                    rows = np.asarray(nf["rows"], dtype=np.int64)
                    vals = np.frombuffer(bytes.fromhex(nf["hex"]), dtype=np.float32).reshape(-1, 2)
                    if len(rows) == len(vals) and (rows < len(uv)).all():
                        # only rows the editor left at the 0 placeholder get their original value back
                        placeholder = np.where(np.isfinite(vals), vals, 0.0)
                        keep = (uv[rows] == placeholder).all(axis=1)
                        uv[rows[keep]] = vals[keep]
                me.SetVertexUVLayerBuffer(layers, uv.tolist())
                layers += 1
            me.SetUVLayerCount(layers)
            me.SetColorLayerCount(0)
            me.SetFaceBuffer(idx.astype(np.int64).tolist())
            if skinned:
                sets = []
                k = 0
                while f"JOINTS_{k}" in at and f"WEIGHTS_{k}" in at:
                    jj = _accessor(doc, bufs, at[f"JOINTS_{k}"]).astype(np.int64)
                    ww = _accessor(doc, bufs, at[f"WEIGHTS_{k}"]).astype(np.float64)
                    if jmap is not None:
                        jj = jmap[jj]
                    sets.append((jj, ww))
                    k += 1
                if sets:
                    jb = np.concatenate([s[0] for s in sets], axis=1)
                    wv = np.concatenate([s[1] for s in sets], axis=1)
                    me.SetMaximumWeightInfluence(jb.shape[1])
                    me.SetSkinningMethod("linear")
                    me.SetVertexWeightBoneBuffer(jb.reshape(-1).tolist())
                    me.SetVertexWeightValueBuffer(wv.reshape(-1).tolist())
            if "_BP_VERTEX_ID" in at:
                vid = _accessor(doc, bufs, at["_BP_VERTEX_ID"]).reshape(-1)
                me.CreateProperty("bp_vertex_id", "i").values = [int(round(float(x))) for x in vid]
            for key, val in {**(gm.get("extras") or {}), **(n.get("extras") or {})}.items():
                if key.startswith("bp_") and key not in ("bp_vertex_id", "bp_tangent_sign", "bp_nonfinite_uv"):
                    _set_prop(me, key, val)
            if "material" in prim and prim["material"] in mat_hash:
                me.SetMaterial(mat_hash[prim["material"]])
    return cast


def _set_prop(node, key: str, val) -> None:
    vals = val if isinstance(val, list) else [val]
    if not vals:
        return
    if all(isinstance(v, bool) for v in vals):
        node.CreateProperty(key, "b").values = [int(v) for v in vals]
    elif all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        node.CreateProperty(key, "i").values = vals
    elif all(isinstance(v, (int, float)) for v in vals):
        node.CreateProperty(key, "f").values = [float(v) for v in vals]
    else:
        node.CreateProperty(key, "s").values = [str(v) for v in vals]
