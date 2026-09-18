"""`nr mesh …` — single-resource helpers.

    nr mesh info   <rpack> <index|name>              summary: entities, entries, submeshes, formats, materials, warnings
    nr mesh export <rpack> <index|name> <out.cast>   write a Cast scene (+ <out>.mesh.json sidecar unless --no-sidecar)
    nr mesh dump   <rpack> <index|name> [--vertices N] JSON of the decoded model (sidecar layout, raw blobs omitted)
    nr mesh census <rpack> [--limit N]               per-resource one-line census (formats, entries, warnings)
    nr mesh import <model.cast|.glb|.gltf> <mesh.json> --out-dir DIR [--parts-dir DIR] [--name NAME] [--no-pad-index] [--ignore-bones]
                                                     rebuild image/fixups/vertex/index parts from an edited Cast
                                                     (raw parts read from --parts-dir, default: the sidecar's folder)
    nr mesh diff   <rpx-resource-dir> [--name NAME] [--ignore-bones] [--json]
                                                     report what the edited model.cast changes (no parts written)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..container.rp6l import Pack
from ..errors import NightrunnerError
from .decode import decode_resource, PART_IMAGE, PART_FIXUPS, PART_SKIN, PART_VERTEX, PART_INDEX, PART_CLOTH
from .sidecar import build_sidecar

PART_FILES = {PART_IMAGE: "image.bin", PART_FIXUPS: "fixups.bin", PART_SKIN: "skin.bin", PART_VERTEX: "vertex.bin",
              PART_INDEX: "index.bin", PART_CLOTH: "cloth.bin"}


def _resolve(pk: Pack, key: str) -> int:
    if key.isdigit():
        i = int(key)
        if not 0 <= i < len(pk):
            raise NightrunnerError(f"index {i} out of range ({len(pk)} resources)")
        return i
    hits = pk.find(key, type_id=0x10)
    if not hits:
        raise NightrunnerError(f"no mesh named {key!r}")
    if len(hits) > 1:
        print(f"note: {len(hits)} meshes named {key!r}, using index {hits[0]}", file=sys.stderr)
    return hits[0]


def cmd_info(a) -> int:
    with Pack.open(a.pack) as pk:
        r = pk.resource(_resolve(pk, a.key))
        m = decode_resource(r)
    s = m.summary()
    s["index"] = r.index
    s["parts"] = [f"0x{t:02X}" for t in r.part_types]
    s["entries"] = [{"index": e.index, "owner_entity": e.owner_entity, "format": e.format, "vertex_base": e.vertex_base,
                     "vertex_count": e.vertex_count, "index_base": e.index_base,
                     "submeshes": [{"material": m.material_name(x.material_slot), "indices": x.index_count,
                                    "palette": len(x.palette)} for x in e.submeshes]} for e in m.geometry_entries]
    s["entity_types"] = {}
    for en in m.entities:
        s["entity_types"][str(en.type)] = s["entity_types"].get(str(en.type), 0) + 1
    print(json.dumps(s, indent=2, ensure_ascii=False))
    return 0


def cmd_export(a) -> int:
    from ..cast.export import export_cast
    with Pack.open(a.pack) as pk:
        r = pk.resource(_resolve(pk, a.key))
        m = decode_resource(r)
        src = {"pack": str(pk.path), "index": r.index, "name": r.name}
    out = Path(a.out)
    rep = export_cast(m, out)
    print(json.dumps(rep.to_json(), indent=2))
    if not a.no_sidecar:
        from ..util.jsonio import dump_json
        side = build_sidecar(m, cast={"file": out.name, "nodes": rep.nodes, "skipped": rep.skipped,
                                      "meshes": [{"name": x.name, "entry": x.entry, "submesh": x.submesh} for x in rep.meshes]},
                             source=src)
        sp = out.with_name(out.stem + ".mesh.json")
        dump_json(side, sp)
        print(f"sidecar: {sp}")
    for w in m.warnings:
        print("warning:", w, file=sys.stderr)
    return 0


def cmd_dump(a) -> int:
    with Pack.open(a.pack) as pk:
        r = pk.resource(_resolve(pk, a.key))
        m = decode_resource(r)
        src = {"pack": str(pk.path), "index": r.index, "name": r.name}
    side = build_sidecar(m, source=src)
    for e in side["geometry_entries"]:
        e.pop("vertex_raw_b64", None)
    if a.vertices:
        for e, ge in zip(side["geometry_entries"], m.geometry_entries):
            v = ge.vertices
            if v is None:
                continue
            n = min(a.vertices, v.count)
            e["vertex_preview"] = [{
                "id": i, "pos": [float(x) for x in v.positions[i]], "uv0": [float(x) for x in v.uv0[i]],
                "normal": [round(float(x), 5) for x in v.normals[i]], "tangent": [round(float(x), 5) for x in v.tangents[i]],
                "sign": int(v.tangent_sign[i]),
                **({"weights": [int(x) for x in v.raw["weights"][i]], "joints": [int(x) for x in v.raw["joints"][i]]} if v.skinned else {}),
            } for i in range(n)]
    print(json.dumps(side, indent=2, ensure_ascii=False))
    return 0


def cmd_census(a) -> int:
    with Pack.open(a.pack) as pk:
        n = 0
        for r in pk.resources_of_type(0x10):
            if a.limit and n >= a.limit:
                break
            n += 1
            try:
                m = decode_resource(r)
                s = m.summary()
                print(json.dumps({"index": r.index, "name": r.name, "entities": s["entities"], "entries": s["geometry_entries"],
                                  "submeshes": s["submeshes"], "formats": s["formats"], "vertices": s["vertices"],
                                  "warnings": s["warnings"]}, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"index": r.index, "name": r.name, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
    return 0


def _read_parts(parts_dir: Path) -> dict[int, bytes]:
    parts = {t: (parts_dir / n).read_bytes() for t, n in PART_FILES.items() if (parts_dir / n).exists()}
    for t in (PART_IMAGE, PART_FIXUPS):
        if t not in parts:
            raise NightrunnerError(f"{parts_dir / PART_FILES[t]} missing (the raw parts of the extracted resource are needed)")
    return parts


def _logical_name(side: dict, override: str | None) -> bytes:
    if override:
        return override.encode("utf-8", "surrogateescape")
    if side.get("name_hex"):
        return bytes.fromhex(side["name_hex"])
    if side.get("name"):
        return side["name"].encode("utf-8", "surrogateescape")
    emb = bytes.fromhex(side["embedded_name_hex"])
    return emb[:-4] if emb.lower().endswith(b".msh") else emb


def cmd_import(a) -> int:
    from ..util.jsonio import dump_json, load_json
    from .codec import rebuild_from_files
    cast_path, side_path = Path(a.cast), Path(a.sidecar)
    parts_dir = Path(a.parts_dir) if a.parts_dir else side_path.parent
    side = load_json(side_path)
    parts = _read_parts(parts_dir)
    name = _logical_name(side, a.name)
    res = rebuild_from_files(cast_path, side_path, parts, name, pad_index_part=not a.no_pad_index,
                             ignore_bone_changes=a.ignore_bones)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced = {PART_IMAGE: res.image, PART_FIXUPS: res.fixups, PART_VERTEX: res.vertex, PART_INDEX: res.index}
    written = []
    for t, data in produced.items():
        if data is not None:
            (out / PART_FILES[t]).write_bytes(data)
            written.append(PART_FILES[t])
    for t in (PART_SKIN, PART_CLOTH):
        if t in parts:
            (out / PART_FILES[t]).write_bytes(parts[t])
            written.append(PART_FILES[t] + " (copied)")
    rep = dict(res.report)
    rep["identical"] = {PART_FILES[t]: (data == parts.get(t)) for t, data in produced.items() if data is not None}
    rep["written"] = written
    rep["out_dir"] = str(out)
    dump_json(rep, out / "import_report.json")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    for w in rep.get("warnings", []):
        print("warning:", w, file=sys.stderr)
    return 0


def cmd_diff(a) -> int:
    from ..cast.import_ import load_sidecar, resolve_import
    from .decode import decode_parts
    from .rebuild import make_plan
    d = Path(a.dir)
    from .codec import find_edit_file, load_edit_scene
    side_path, cast_path = d / "mesh.json", find_edit_file(d)
    if not side_path.exists() or not cast_path.exists():
        raise NightrunnerError(f"{d}: needs model.cast (or model.glb / model.gltf) and mesh.json "
                             "(an extracted mesh resource directory)")
    parts = _read_parts(d)
    side = load_sidecar(side_path)
    name = _logical_name(side, a.name)
    model = decode_parts(name.decode("utf-8", "surrogateescape"), parts[PART_IMAGE], parts[PART_FIXUPS],
                         vertex=parts.get(PART_VERTEX), index=parts.get(PART_INDEX), skin=parts.get(PART_SKIN),
                         cloth=parts.get(PART_CLOTH))
    imp = resolve_import(load_edit_scene(cast_path, model), side)
    plan = make_plan(model, imp, name)
    rep = plan.report()
    rep["cast"] = {"software": imp.scene.software, "up_axis": imp.scene.up_axis, "bones": len(imp.scene.bones),
                   "bones_matched_by": imp.bones_matched_by, "meshes": len(imp.scene.meshes)}
    rep["would_refuse"] = None
    if plan.bone_changes and not a.ignore_bones:
        rep["would_refuse"] = "bone transforms differ from the sidecar (pass --ignore-bones to keep the native skeleton)"
    if a.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0
    print(f"{name.decode('utf-8', 'replace')}: Cast by {imp.scene.software!r}, up {imp.scene.up_axis!r}, "
          f"{len(imp.scene.meshes)} meshes, {len(imp.scene.bones)} bones (matched by {imp.bones_matched_by})")
    for e in rep["entries"]:
        print(f"  entry {e['entry']} fmt {e['format']}: {e['path']}, vertices {e['vertex_count'][0]} -> {e['vertex_count'][1]}, "
              f"indices {e['index_count'][0]} -> {e['index_count'][1]}, submeshes {e['submesh_count'][0]} -> {e['submesh_count'][1]}")
        for s in e["submeshes"]:
            print(f"    s{s['submesh']} {s['mesh']!r} material {s['material']!r} (slot {s['material_slot']}, {s['material_matched_by']}): "
                  f"indices {s['index_count'][0]} -> {s['index_count'][1]}, moved {s['moved']}, edited {s['edited']}, "
                  f"new {s['new_vertices']}, palette {s['palette'][0]} -> {s['palette'][1]}"
                  + (f", derived {s['derived']}" if s["derived"] else ""))
    if rep["new_materials"]:
        print("  new materials:", rep["new_materials"])
    if rep["identity_rename"]:
        print("  embedded name:", rep["identity_rename"][0], "->", rep["identity_rename"][1])
    if rep["bone_changes"]:
        print("  bone changes:", rep["bone_changes"][:10])
    for w in rep["warnings"]:
        print("  warning:", w)
    if rep["would_refuse"]:
        print("  BUILD WOULD REFUSE:", rep["would_refuse"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nr mesh")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("info"); p.add_argument("pack"); p.add_argument("key"); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("export"); p.add_argument("pack"); p.add_argument("key"); p.add_argument("out")
    p.add_argument("--no-sidecar", action="store_true"); p.set_defaults(fn=cmd_export)
    p = sub.add_parser("dump"); p.add_argument("pack"); p.add_argument("key")
    p.add_argument("--vertices", type=int, default=0, help="include the first N decoded vertices per entry"); p.set_defaults(fn=cmd_dump)
    p = sub.add_parser("census"); p.add_argument("pack"); p.add_argument("--limit", type=int, default=0); p.set_defaults(fn=cmd_census)
    p = sub.add_parser("import", help="edited model.cast + mesh.json + raw parts → new parts")
    p.add_argument("cast"); p.add_argument("sidecar"); p.add_argument("--out-dir", required=True)
    p.add_argument("--parts-dir", help="folder holding image.bin/fixups.bin/vertex.bin/index.bin (default: the sidecar's)")
    p.add_argument("--name", help="logical resource name (default: the sidecar's); the embedded .msh name follows it")
    p.add_argument("--no-pad-index", action="store_true", help="do not pad the index part to 16 (field08 == 0 packs)")
    p.add_argument("--ignore-bones", action="store_true", help="keep the native skeleton when the Cast bones moved")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("diff", help="report what an edited model.cast changes, without building")
    p.add_argument("dir"); p.add_argument("--name"); p.add_argument("--ignore-bones", action="store_true")
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_diff)
    return ap


def run(args) -> int:
    a = build_parser().parse_args(args.args)
    return a.fn(a)
