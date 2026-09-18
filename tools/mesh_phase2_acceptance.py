"""Phase-2 acceptance on the PC (writes out/reports/mesh/phase2_acceptance.txt):

  1. extract common_meshes_pc.rpack --limit N --types 0x10 into out/test/phase2/cm.rpx
  2. (a) `nr build --force-codec` (every mesh goes through Cast import + rebuild) → `nr validate` → every part of
         every resource byte-identical with the source pack
  3. (b) move one vertex of one mesh in its model.cast → build → validate → decode: the vertex moved, every other
         byte of its vertex buffer identical, image/fixups/index identical
  4. (c) delete one triangle → build --layout auto → validate → counts updated, buffers re-laid out per the census
         rule; `nr roundtrip` on the rebuilt pack
  5. in-memory export → import → rebuild over a wider slice of the corpus (no disk tree): byte identity per part

    python tools/mesh_phase2_acceptance.py [--limit 200] [--wide 1500]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.cast import castlib  # noqa: E402
from nightrunner.cast.export import export_cast  # noqa: E402
from nightrunner.cli import main as bp_main  # noqa: E402
from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.container.validate import validate  # noqa: E402
from nightrunner.mesh.codec import rebuild_from_files  # noqa: E402
from nightrunner.mesh.decode import decode_resource, decode_parts  # noqa: E402
from nightrunner.mesh.sidecar import build_sidecar  # noqa: E402
from nightrunner.util.jsonio import dump_json  # noqa: E402
from tests.paths import ASSETS  # noqa: E402

OUT = ROOT / "out" / "test" / "phase2"
LOG = ROOT / "out" / "reports" / "mesh" / "phase2_acceptance.txt"
PART_FILES = {0x10: "image.bin", 0x11: "fixups.bin", 0x12: "skin.bin", 0xF0: "vertex.bin", 0xF1: "index.bin", 0xF3: "cloth.bin"}


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        line = " ".join(str(x) for x in a)
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()


def bp(*argv) -> int:
    return bp_main([str(a) for a in argv])


def compare_packs(log, src: Path, dst: Path, names: list[str]) -> tuple[int, int]:
    ident = diff = 0
    with Pack.open(src) as a, Pack.open(dst) as b:
        for r2 in b:
            hits = a.find(r2.name, type_id=0x10, fold=False)
            r1 = a.resource(hits[0])
            p1 = [bytes(a.read_part(i)) for i in r1.part_indices]
            p2 = [bytes(b.read_part(i)) for i in r2.part_indices]
            if p1 == p2:
                ident += 1
            else:
                diff += 1
                bad = [f"0x{t:02X}" for t, x, y in zip(r1.part_types, p1, p2) if x != y]
                log(f"  DIFF {r2.name!r}: parts {bad}")
    return ident, diff


def pick_edit_target(rpx: Path) -> dict:
    spec = json.loads((rpx / "pack.json").read_text(encoding="utf-8"))
    for e in spec["resources"]:
        ed = e["editable"]
        if ed.get("kind") == "cast" and ed.get("submeshes", 0) >= 2 and 6 in ed.get("formats", []) and ed.get("vertices", 0) < 20000:
            return e
    return next(e for e in spec["resources"] if e["editable"].get("kind") == "cast" and e["editable"].get("vertices", 0) > 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--wide", type=int, default=1500, help="meshes of common_meshes for the in-memory check (0 = all packs entirely)")
    a = ap.parse_args()
    log = Log(LOG)
    t0 = time.time()
    src = ASSETS / "common_meshes_pc.rpack"
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rpx = OUT / "cm.rpx"
    log(f"=== phase 2 acceptance {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"--- extract {src.name} --limit {a.limit} --types 0x10")
    rc = bp("extract", src, rpx, "--limit", a.limit, "--types", "0x10")
    log("extract rc", rc)
    spec = json.loads((rpx / "pack.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in spec["resources"]]
    log(f"resources: {len(names)}; formats: {sorted({f for e in spec['resources'] for f in e['editable'].get('formats', [])})}")

    # (a) force-codec build: every mesh through import + rebuild
    log("--- (a) build --force-codec --layout auto (all meshes through Cast import)")
    out_a = OUT / "cm_forced.rpack"
    rc = bp("build", rpx, out_a, "--layout", "auto", "--force-codec")
    log("build rc", rc)
    rc = bp("validate", out_a)
    log("validate rc", rc)
    ident, diff = compare_packs(log, src, out_a, names)
    log(f"(a) byte-identical resources: {ident}/{ident + diff}  differing: {diff}")

    # (b) move one vertex
    target = pick_edit_target(rpx)
    d = rpx / target["dir"]
    log(f"--- (b) move one vertex by (0.01, 0, 0): {target['name']!r} formats {target['editable']['formats']} "
        f"vertices {target['editable']['vertices']} submeshes {target['editable']['submeshes']}")
    orig_parts = {t: (d / n).read_bytes() for t, n in PART_FILES.items() if (d / n).exists()}
    c = castlib.Cast.load(str(d / "model.cast"))
    mdl = c.Roots()[0].ChildOfType(castlib.Model)
    mesh = mdl.Meshes()[0]
    vp = list(mesh.properties["vp"].values)
    vp[0] += 0.01
    mesh.properties["vp"].values = vp
    vid = int(mesh.properties["bp_vertex_id"].values[0])
    entry_i = int(mesh.properties["bp_entry"].values[0])
    c.save(str(d / "model.cast"))
    out_b = OUT / "cm_moved.rpack"
    rc = bp("build", rpx, out_b, "--layout", "auto")
    log("build rc", rc, "validate rc", bp("validate", out_b))
    with Pack.open(out_b) as pk:
        r = pk.resource(pk.find(target["name"], type_id=0x10, fold=False)[0])
        m2 = decode_resource(r)
        new_v = bytes(r.read_part_by_type(0xF0))
        e = m2.geometry_entries[entry_i]
        stride = e.vertex_end - e.vertex_base
        stride = {0: 16, 3: 32, 6: 40, 8: 80}[e.format]
        base = e.vertex_base + vid * stride
        diffs = [i for i in range(len(new_v)) if new_v[i] != orig_parts[0xF0][i]]
        ok = bool(diffs) and all(base <= i < base + 12 for i in diffs)
        m1 = decode_parts("x", orig_parts[0x10], orig_parts[0x11], vertex=orig_parts[0xF0], index=orig_parts[0xF1])
        moved = m2.geometry_entries[entry_i].vertices.positions[vid] - m1.geometry_entries[entry_i].vertices.positions[vid]
        log(f"(b) vertex {vid}: delta {np.round(moved, 5).tolist()}; differing vertex bytes {len(diffs)} all inside the vertex: {ok}; "
            f"image identical {bytes(r.read_part_by_type(0x10)) == orig_parts[0x10]}; fixups identical "
            f"{bytes(r.read_part_by_type(0x11)) == orig_parts[0x11]}; index identical {bytes(r.read_part_by_type(0xF1)) == orig_parts[0xF1]}")
    # (c) delete one triangle (from the moved cast)
    log(f"--- (c) delete one triangle from {mesh.Name()!r}")
    f = list(mesh.properties["f"].values)
    mesh.properties["f"].values = f[3:]
    c.save(str(d / "model.cast"))
    out_c = OUT / "cm_deltri.rpack"
    rc = bp("build", rpx, out_c, "--layout", "auto")
    log("build rc", rc, "validate rc", bp("validate", out_c))
    with Pack.open(out_c) as pk:
        r = pk.resource(pk.find(target["name"], type_id=0x10, fold=False)[0])
        m3 = decode_resource(r)
        e3 = m3.geometry_entries[entry_i]
        e1 = m1.geometry_entries[entry_i]
        log(f"(c) entry {entry_i}: index count {e1.index_count} -> {e3.index_count}; vertex count {e1.vertex_count} -> {e3.vertex_count}; "
            f"submesh counts {[s.index_count for s in e1.submeshes]} -> {[s.index_count for s in e3.submeshes]}; "
            f"index part {len(orig_parts[0xF1])} -> {len(bytes(r.read_part_by_type(0xF1)))} bytes (mult of 16: {len(bytes(r.read_part_by_type(0xF1))) % 16 == 0}); "
            f"bases {[(x.vertex_base, x.index_base) for x in m3.geometry_entries]}; warnings {m3.warnings}")
        rep = validate(pk)
        log(f"(c) validate: ok={rep.ok} errors={len(rep.errors)}")
    log("--- roundtrip of the rebuilt pack")
    rc = bp("roundtrip", out_c, "--types", "0x10", "--report", str(OUT / "roundtrip_deltri.json"))
    log("roundtrip rc", rc, json.loads((OUT / "roundtrip_deltri.json").read_text(encoding="utf-8"))["summary"])
    log("--- mesh diff (edited tree)")
    bp("mesh", "diff", d)

    # wide in-memory identity check
    log(f"--- (wide) in-memory export -> import -> rebuild, byte identity per part")
    wide = a.wide or None                       # --wide 0: the whole corpus
    packs = [("common_meshes_pc.rpack", wide), ("dlc_frontier_pc.rpack", None if wide is None else wide // 3), ("engine_pc.rpack", None),
             ("dlc_ft_prologue_pc.rpack", None), ("menu_level_ft_pc.rpack", None)]
    tmp = OUT / "wide"
    tmp.mkdir(exist_ok=True)
    total = ident = 0
    fails: list[str] = []
    t1 = time.time()
    for pname, lim in packs:
        p = ASSETS / pname
        if not p.exists():
            continue
        with Pack.open(p) as pk:
            n = 0
            for r in pk.resources_of_type(0x10):
                if lim is not None and n >= lim:
                    break
                n += 1
                total += 1
                try:
                    m = decode_resource(r)
                    rep = export_cast(m, tmp / "m.cast")
                    side = build_sidecar(m, cast={"meshes": [{"name": x.name, "entry": x.entry, "submesh": x.submesh} for x in rep.meshes]})
                    dump_json(side, tmp / "mesh.json")
                    parts = {}
                    for i in r.part_indices:
                        parts[pk.part_type(i)] = bytes(pk.read_part(i))
                    res = rebuild_from_files(tmp / "m.cast", tmp / "mesh.json", parts, r.name_raw)
                    same = (res.image == parts[0x10] and res.fixups == parts[0x11] and res.vertex == parts.get(0xF0)
                            and res.index == parts.get(0xF1))
                    if same:
                        ident += 1
                    else:
                        fails.append(f"{pname} #{r.index} {r.name!r}: differs")
                except Exception as exc:  # noqa: BLE001
                    fails.append(f"{pname} #{r.index} {r.name!r}: {type(exc).__name__}: {exc}")
            log(f"  {pname}: {n} meshes")
    log(f"(wide) identical {ident}/{total} in {time.time() - t1:.0f}s")
    for x in fails[:40]:
        log("  ", x)
    log(f"=== done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
