"""MeshCodec — logical type 0x10.

extract()   model.cast (Cast scene, native coordinates) + mesh.json (sidecar) next to the raw parts.
roundtrip() decode → re-encode vertex (0xF0), index (0xF1) and fixups (0x11) in memory; `nr roundtrip` compares
            them with the stored bytes across the corpus (primary correctness gate).
build()     phase 2: raw parts + model.cast + mesh.json → new 0x10/0x11/0xF0/0xF1 bytes (cast.import_ +
            mesh.rebuild). Called by `nr build` when model.cast or mesh.json changed, on `--force-codec`, or when
            the spec flags `needs_identity_fix`; an unedited Cast reproduces every part byte-identically (apart
            from the embedded-name rewrite when the logical name differs).
"""

from __future__ import annotations

from pathlib import Path

from ..codecs import BuildContext, Codec, ExtractContext, register
from ..container.rp6l import PartSource, Resource
from ..errors import BuildError
from ..util.hashing import sha256_bytes
from ..util.jsonio import dump_json
from .decode import decode_parts, decode_resource, PART_FIXUPS, PART_VERTEX, PART_INDEX, PART_IMAGE, PART_SKIN, PART_CLOTH
from .encode import encode_index_buffer, encode_vertex_buffer
from .model import Model
from .sidecar import build_sidecar

CAST_FILE = "model.cast"
GLTF_FILES = ("model.glb", "model.gltf")      # optional glTF edits; they win over model.cast
SIDECAR_FILE = "mesh.json"
_PART_KEYS = {PART_IMAGE: "image", PART_FIXUPS: "fixups", PART_SKIN: "skin", PART_VERTEX: "vertex",
              PART_INDEX: "index", PART_CLOTH: "cloth"}
REGENERATED_TYPES = (PART_IMAGE, PART_FIXUPS, PART_VERTEX, PART_INDEX)


class MeshCodec(Codec):
    kind = "cast"

    # ---- extract --------------------------------------------------------------------------------------------

    def extract(self, ctx: ExtractContext, res: Resource, out_dir: Path) -> dict:
        from ..cast.export import export_cast
        model = decode_resource(res)
        for w in model.warnings:
            ctx.warn(f"{res.name!r}: {w}")
        rep = export_cast(model, out_dir / CAST_FILE)
        shas = {}
        for i in res.part_indices:
            t = ctx.pack.part_type(i)
            key = _PART_KEYS.get(t, f"part_{t:02X}")
            if ctx.pack.part_is_direct(i):
                shas[key] = sha256_bytes(ctx.pack.read_part(i))
        cast_info = {
            "file": CAST_FILE, "nodes": rep.nodes, "skipped": rep.skipped, "bone_frame_residual": rep.bone_frame_residual,
            "meshes": [{"name": m.name, "entry": m.entry, "submesh": m.submesh, "vertices": m.vertex_count,
                        "faces": m.face_count, "vertex_ids_b64": _b64_u32(m.vertex_ids)} for m in rep.meshes],
        }
        side = build_sidecar(model, part_sha256=shas, cast=cast_info,
                             source={"pack": str(ctx.pack.path), "index": res.index, "name": res.name})
        dump_json(side, out_dir / SIDECAR_FILE)
        files = {}
        files.update(self.file_record(out_dir / CAST_FILE, CAST_FILE))
        files.update(self.file_record(out_dir / SIDECAR_FILE, SIDECAR_FILE))
        summ = model.summary()
        return {"kind": self.kind, "files": files, "cast_nodes": rep.nodes, "entities": summ["entities"],
                "geometry_entries": summ["geometry_entries"], "submeshes": summ["submeshes"],
                "formats": summ["formats"], "vertices": summ["vertices"], "triangles": summ["triangles"],
                "warnings": list(model.warnings)}

    # ---- round trip -----------------------------------------------------------------------------------------

    def roundtrip(self, res: Resource) -> dict[int, bytes]:
        model = decode_resource(res)
        return self.reencode(res, model)

    @staticmethod
    def reencode(res: Resource, model: Model) -> dict[int, bytes]:
        """Per-ordinal bytes for the fixups, vertex and index parts regenerated from *model*."""
        out: dict[int, bytes] = {}
        for k, i in enumerate(res.part_indices):
            t = res.pack.part_type(i)
            if t == PART_FIXUPS:
                out[k] = model.fixups.to_bytes()
            elif t == PART_VERTEX:
                out[k] = encode_vertex_buffer(model)
            elif t == PART_INDEX:
                out[k] = encode_index_buffer(model)
        return out

    # ---- build ----------------------------------------------------------------------------------------------

    def build(self, ctx: BuildContext, entry: dict, res_dir: Path) -> dict[int, PartSource]:
        parts_by_type: dict[int, bytes] = {}
        ordinals: dict[int, int] = {}
        for k, prec in enumerate(entry["parts"]):
            t = int(prec["type"], 0)
            raw = prec.get("raw")
            if raw is None:
                raise BuildError(f"{entry['name']!r}: part {k} (0x{t:02X}) has no raw file; the mesh codec needs every raw part")
            p = ctx.rpx_dir / raw
            if not p.exists():
                raise BuildError(f"{entry['name']!r}: raw part file missing: {p}")
            if t in parts_by_type:
                raise BuildError(f"{entry['name']!r}: duplicate part type 0x{t:02X}")
            parts_by_type[t] = p.read_bytes()
            ordinals[t] = k
        for t in (PART_IMAGE, PART_FIXUPS):
            if t not in parts_by_type:
                raise BuildError(f"{entry['name']!r}: missing part 0x{t:02X}")
        name_bytes = bytes.fromhex(entry["name_hex"]) if entry.get("name_hex") else entry["name"].encode("utf-8", "surrogateescape")
        f08 = int(ctx.spec.get("header", {}).get("field08", "0"), 0)
        result = rebuild_from_files(
            find_edit_file(res_dir), res_dir / SIDECAR_FILE, parts_by_type, name_bytes,
            pad_index_part=bool(f08 & 0x1000), ignore_bone_changes=bool(ctx.options.get("ignore_bone_changes")))
        for w in result.report.get("warnings", []):
            ctx.warn(f"{entry['name']!r}: {w}")
        out: dict[int, PartSource] = {}
        produced = {PART_IMAGE: result.image, PART_FIXUPS: result.fixups, PART_VERTEX: result.vertex, PART_INDEX: result.index}
        for t, data in produced.items():
            if t in ordinals and data is not None:
                out[ordinals[t]] = PartSource(data)
        return out


def rebuild_from_files(cast_path: Path, sidecar_path: Path, parts_by_type: dict[int, bytes], logical_name: bytes | str,
                       *, pad_index_part: bool = True, ignore_bone_changes: bool = False, verify_result: bool = True):
    """Shared by MeshCodec.build and `nr mesh import`: decode the raw parts, read + resolve the Cast, rebuild."""
    from ..cast.import_ import load_sidecar, resolve_import
    from .rebuild import rebuild, verify
    name = logical_name.decode("utf-8", "surrogateescape") if isinstance(logical_name, bytes) else logical_name
    if not Path(cast_path).exists():
        raise BuildError(f"{cast_path}: model.cast missing")
    if not Path(sidecar_path).exists():
        raise BuildError(f"{sidecar_path}: mesh.json missing")
    side = load_sidecar(sidecar_path)
    # decode under the name the parts were extracted as: a renamed / cloned mesh keeps the source's scene node
    # names (<source>.eN.sM); the new logical name only goes into the rebuilt parts
    model = decode_parts(side.get("name") or name, parts_by_type[PART_IMAGE], parts_by_type[PART_FIXUPS],
                         vertex=parts_by_type.get(PART_VERTEX), index=parts_by_type.get(PART_INDEX),
                         skin=parts_by_type.get(PART_SKIN), cloth=parts_by_type.get(PART_CLOTH))
    from .rebuild import refuse_layout
    refuse_layout(model)          # DL2 meshes are read-only (clear error before the Cast is even read)
    scene = load_edit_scene(cast_path, model)
    imp = resolve_import(scene, side)
    result = rebuild(model, imp, logical_name=logical_name, pad_index_part=pad_index_part,
                     ignore_bone_changes=ignore_bone_changes)
    if verify_result:
        problems = verify(result, model, imp, name)
        if problems:
            raise BuildError(f"{name!r}: rebuilt parts fail the self-check: " + "; ".join(problems[:5]))
    return result


def load_edit_scene(cast_path: Path, model):
    """read_cast, except that glTF / GLB edits (Blender's glTF round trip moves every value by float noise and
    splits a few vertices) are first snapped back onto the original mesh (cast/split.normalize_mesh_scene)."""
    from ..cast.import_ import read_cast
    p = Path(cast_path)
    if p.suffix.lower() not in (".gltf", ".glb"):
        return read_cast(p)
    from ..cast.gltf import load_scene
    from ..cast.split import normalize_mesh_scene
    cast, notes = normalize_mesh_scene(load_scene(p), model)
    scene = read_cast(p, cast=cast)
    scene.warnings.extend(notes)
    return scene


def find_edit_file(res_dir: Path) -> Path:
    """The mesh edit in a resource folder: model.glb / model.gltf when present (they win over model.cast)."""
    res_dir = Path(res_dir)
    alt = [res_dir / n for n in GLTF_FILES if (res_dir / n).is_file()]
    if len(alt) > 1:
        raise BuildError(f"{res_dir}: both model.glb and model.gltf — keep one")
    return alt[0] if alt else res_dir / CAST_FILE


def _b64_u32(a) -> str:
    import base64
    import numpy as np
    return base64.b64encode(np.ascontiguousarray(a, dtype="<u4").tobytes()).decode("ascii")


register(0x10, MeshCodec())
