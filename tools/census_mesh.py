"""Corpus census of every type-0x10 resource: ClassReader records/slots, mesh class graph, vertex formats, vertex
field holes, buffer layout rules, skin weights, qtangent requantisation. Every mesh is fully decoded with
nightrunner.mesh.decode, so this is also the "every mesh decodes" evidence.

    python tools/census_mesh.py [--packs a.rpack b.rpack ...] [--out out/reports/mesh]

Writes census_<pack>.json per pack and census_summary.json (all packs merged) with exact counts.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.classreader.graph import CLASS_PALETTE, PALETTE_DESC_SIZE  # noqa: E402
from nightrunner.mesh.decode import decode_resource  # noqa: E402
from nightrunner.mesh.vertex import (STRIDES, QTAN_SCALE, decode_qtangent, encode_qtangent,  # noqa: E402
                                   VERTEX_BLOCK_ALIGN, INDEX_BASE_ALIGN, INDEX_BUFFER_ALIGN, is_skinned, align_to)
from tests.paths import ASSETS  # noqa: E402

MESH_PACKS = ["common_meshes_pc.rpack", "dlc_frontier_pc.rpack", "dlc_ft_prologue_pc.rpack", "menu_level_ft_pc.rpack",
              "engine_pc.rpack"]


class Tally:
    def __init__(self):
        self.c: dict[str, Counter] = {}
        self.n: Counter = Counter()
        self.lists: dict[str, list] = {}
        self.minmax: dict[str, list] = {}

    def count(self, key: str, value=1):
        self.n[key] += value

    def hist(self, key: str, value):
        self.c.setdefault(key, Counter())[value] += 1

    def note(self, key: str, item, limit: int = 50):
        lst = self.lists.setdefault(key, [])
        if len(lst) < limit:
            lst.append(item)
        self.n[key + "_total"] += 1

    def mm(self, key: str, value):
        v = self.minmax.get(key)
        if v is None:
            self.minmax[key] = [value, value]
        else:
            v[0] = min(v[0], value)
            v[1] = max(v[1], value)

    def to_json(self, top: int = 40) -> dict:
        def key_str(k):
            return k if isinstance(k, str) else str(k)
        return {
            "counts": dict(sorted(self.n.items())),
            "hist": {k: {key_str(a): b for a, b in sorted(v.items(), key=lambda kv: (-kv[1], str(kv[0])))[:top]}
                     | ({"__distinct__": len(v)} if len(v) > top else {}) for k, v in sorted(self.c.items())},
            "minmax": dict(sorted(self.minmax.items())),
            "examples": dict(sorted(self.lists.items())),
        }

    def merge(self, other: "Tally"):
        for k, v in other.n.items():
            self.n[k] += v
        for k, v in other.c.items():
            self.c.setdefault(k, Counter()).update(v)
        for k, v in other.lists.items():
            lst = self.lists.setdefault(k, [])
            lst.extend(v[: max(0, 50 - len(lst))])
        for k, v in other.minmax.items():
            self.mm(k, v[0]); self.mm(k, v[1])


def census_pack(path: Path, t: Tally, *, sample_limit: int | None = None) -> None:
    with Pack.open(path) as pk:
        meshes = list(pk.resources_of_type(0x10))
        if sample_limit:
            meshes = meshes[:sample_limit]
        for r in meshes:
            t.count("meshes")
            t.hist("part_types", ",".join(f"{x:02X}" for x in r.part_types))
            try:
                m = decode_resource(r)
            except Exception as exc:  # noqa: BLE001
                t.note("decode_failures", {"pack": path.name, "index": r.index, "name": r.name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            for w in m.warnings:
                t.hist("decode_warnings", w.split(":")[0] if ":" in w else w)
                t.note("decode_warning_examples", {"index": r.index, "name": r.name, "warning": w}, 100)
            _census_fixups(m, t)
            _census_graph(m, t, r)
            _census_geometry(m, t, r)
            _census_vertices(m, t, r)
            _census_buffers(m, t, r)


def _census_fixups(m, t: Tally) -> None:
    fx = m.fixups
    t.hist("fixups.object_count_raw", f"0x{fx.object_count_raw:08X}")
    t.count("fixups.secondary_present", int(fx.secondary_present))
    t.hist("fixups.trailing_len", len(fx.trailing))
    t.count("fixups.trailing_nonzero", int(any(fx.trailing)))
    t.count("fixups.part_len_mult16", int(len(m.fixups.to_bytes()) % 16 == 0))
    t.count("image.part_len_mult16", int(len(m.image.data) % 16 == 0))
    t.count("image.padding_nonzero", int(any(m.image.padding())))
    t.hist("image.padding_len", len(m.image.padding()))
    offs = [r.offset for r in fx.records if not r.secondary]
    t.count("fixups.records_ascending", int(offs == sorted(offs)))
    t.count("fixups.record0_at_0", int(bool(fx.records) and fx.records[0].offset == 0))
    for r in fx.records:
        t.hist("class_id", r.class_id)
        t.hist("class_hi", f"0x{r.class_hi:02X}")
        t.count("records")
        t.count("records.secondary", int(r.secondary))
        t.count("records.reverse", int(r.reverse))
        if r.class_id != 0:
            t.hist(f"class{r.class_id}.count", r.count)
    for cid in fx.class_census():
        t.hist("meshes_with_class", cid)
    for s in fx.slots:
        t.hist("slot_kind", s.kind)
        t.count("slots")
        if s.offset % 8:
            t.count("slots.misaligned")
    slot_offs = [s.offset for s in fx.slots]
    t.count("fixups.slots_ascending", int(slot_offs == sorted(slot_offs)))
    # tag census for tagged slots
    for s in fx.slots:
        if s.kind & 2:
            raw = m.image.u64(s.offset)
            t.hist("slot_tag", f"0x{raw >> 48:04X}")


def _census_graph(m, t: Tally, r) -> None:
    img = m.image
    root = m.root_raw
    t.hist("root.slots", ",".join(f"{p.slot:#x}" for p in _root_slots(m)))
    for rel in (0x5C, 0x60, 0x64, 0x68, 0x6C):
        t.hist(f"root.u32_{rel:#x}", struct.unpack_from("<I", root, rel)[0])
    t.hist("root.u32_0x58_eq_entities", int(struct.unpack_from("<I", root, 0x58)[0] == len(m.entities)))
    t.count("embedded_name_ok", int(m.embedded_name == r.name_raw + b".msh"))
    t.hist("scr_name_present", int(m.scr_name is not None))
    if m.scr_name is not None:
        t.hist("scr_name_matches", int(m.scr_name == r.name_raw + b".scr"))
    t.mm("entities", len(m.entities))
    t.count("entities_total", len(m.entities))
    for e in m.entities:
        t.hist("entity.type", e.type)
        t.hist("entity.flags", f"0x{e.flags:X}")
        t.hist("entity.geometry_count", e.geometry_count)
        t.count("entity.parent_forward", int(e.parent > e.index))
        t.count("entity.root", int(e.parent < 0))
        t.count("entity.geometry_link_ok", int((e.geometry_array_record is not None) == (e.geometry_count > 0)))
        t.count("entity.aux_present", int(e.aux_offset is not None))
        if e.raw_aux:
            t.count("entity.aux_all_zero", int(not any(e.raw_aux)))
            t.hist("entity.aux_len", len(e.raw_aux))
            if any(e.raw_aux):
                t.note("entity.aux_nonzero_examples", {"index": r.index, "name": r.name, "entity": e.index, "hex": e.raw_aux.hex()}, 20)
        t.count("entity.raw_90_nonzero", int(any(e.raw_90)))
        t.count("entity.raw_ca_nonzero", int(any(e.raw_ca)))
        t.hist("entity.raw_90_hex", e.raw_90.hex())
        t.hist("entity.raw_ca_hex", e.raw_ca.hex())
        if e.geometry_count:
            t.hist("entity.type_with_geometry", e.type)
    t.hist("materials.count", len(m.materials))
    t.hist("materials.capacity_minus_count", m.material_capacity - len(m.materials))
    for mat in m.materials:
        t.hist("material.tag", f"0x{mat.name_tag:04X}")
        w0, w2 = struct.unpack_from("<QQ", mat.raw, 0)[0], struct.unpack_from("<Q", mat.raw, 0x10)[0]
        u18, u1c = struct.unpack_from("<II", mat.raw, 0x18)
        t.hist("material.raw_00", f"0x{w0:X}")
        t.hist("material.raw_10", f"0x{w2:X}")
        t.hist("material.raw_18", u18)
        t.hist("material.raw_1c", u1c)
        t.hist("material.name_ext", mat.name.rsplit(b".", 1)[-1].decode("latin1") if b"." in mat.name else "")
        # the {u32 len, u32 len} prefix before the string
        t.count("material.name_inline", int(mat.name_inline))
        tgt = None if mat.name_inline else img.pointer(mat.offset + 8).target
        if tgt is not None and tgt >= 8:
            l1, l2 = struct.unpack_from("<II", img.data, tgt - 8)
            t.count("material.len_prefix_ok", int(l1 == len(mat.name) and l2 == len(mat.name)))
    for o in m.opaque:
        t.hist("opaque.class", o.class_id)
        if o.class_id == 12:
            t.hist("class12.magic", o.data[:4].hex())
            t.hist("class12.size", o.size)
        if o.class_id == 13:
            t.hist("class13.size", o.size)
            t.hist("class13.hex", o.data.hex())


def _root_slots(m):
    return [p for p in (m.image.pointer(off) for off in range(0, 0x70, 8) if m.image.is_slot(off))]


def _census_geometry(m, t: Tally, r) -> None:
    img = m.image
    t.hist("geometry.arrays_per_mesh", len({e.array_record for e in m.geometry_entries}))
    t.hist("geometry.entries_per_mesh", len(m.geometry_entries))
    per_array = Counter(e.array_record for e in m.geometry_entries)
    for c in per_array.values():
        t.hist("geometry.entries_per_array", c)
    fmts = sorted({e.format for e in m.geometry_entries})
    t.hist("geometry.formats_per_mesh", ",".join(map(str, fmts)))
    for e in m.geometry_entries:
        t.hist("format", e.format)
        t.hist("geometry.submesh_count", e.submesh_count)
        t.hist("geometry.raw_00", e.raw_00.hex())
        t.hist("geometry.raw_12_eq_nsub", int(e.raw_12 == e.submesh_count))
        t.hist("geometry.raw_14_eq_materials", int(e.raw_14 == len(m.materials)))
        t.hist("geometry.raw_17", e.raw_17)
        t.hist("geometry.raw_34_head", e.raw_34[:4].hex())
        t.hist("geometry.raw_38", struct.unpack_from("<H", e.raw_34, 4)[0])
        t.hist("geometry.raw_3a", struct.unpack_from("<H", e.raw_34, 6)[0])
        t.hist("geometry.raw_3c", struct.unpack_from("<H", e.raw_34, 8)[0])
        t.hist("geometry.raw_3e", struct.unpack_from("<H", e.raw_34, 10)[0])
        t.mm("geometry.vertex_count", e.vertex_count)
        t.count("geometry.vertex_count_gt_65535", int(e.vertex_count > 65535))
        t.hist("geometry.owner_type", None if e.owner_entity is None else m.entities[e.owner_entity].type)
        # palette record: +0x20 should target a class-7 record with count == nsub
        if e.submesh_count:
            pt = img.pointer(e.offset + 0x20).target
            ri = img.record_at(pt) if pt is not None else None
            ok = ri is not None and img.records[ri].class_id == CLASS_PALETTE and img.records[ri].count == e.submesh_count
            t.count("geometry.palette_record_ok", int(ok))
            t.count("geometry.palette_record_checked")
        for s in e.submeshes:
            t.count("submeshes")
            t.hist("submesh.palette_count", len(s.palette))
            t.count("submesh.index_count_mod3", int(s.index_count % 3 == 0))
            t.count("submesh.empty", int(s.index_count == 0))
            t.count("submesh.palette_null_when_zero", int((len(s.palette) == 0) == (s.palette_offset is None)))
            if e.vertices is not None and e.vertices.skinned:
                t.count("submesh.skinned")
                if len(s.palette) == 0:
                    t.note("submesh.skinned_without_palette", {"index": r.index, "name": r.name, "entry": e.index, "sub": s.index})
            elif len(s.palette):
                t.note("submesh.static_with_palette", {"index": r.index, "name": r.name, "entry": e.index, "sub": s.index, "n": len(s.palette)})
            if len(s.indices) and e.vertex_count:
                t.count("submesh.index_in_range", int(int(s.indices.max()) < e.vertex_count))
                t.count("submesh.index_checked")


def _census_vertices(m, t: Tally, r) -> None:
    for e in m.geometry_entries:
        v = e.vertices
        if v is None:
            continue
        raw = v.raw
        n = v.count
        t.count(f"vertices.fmt{e.format}", n)
        t.count("vertices", n)
        fin = np.isfinite(v.positions).all() and np.isfinite(v.uv0).all() and (v.uv1 is None or np.isfinite(v.uv1).all())
        if not fin:
            t.note("vertices.nonfinite", {"index": r.index, "name": r.name, "entry": e.index})
        if "raw_06" in raw.dtype.names:
            for val, c in zip(*np.unique(raw["raw_06"], return_counts=True)):
                t.hist("fmt0.raw_06", f"0x{int(val):04X}")
                t.count("fmt0.raw_06_rows", int(c))
        if "raw_tail" in raw.dtype.names:
            vals, cnt = np.unique(raw["raw_tail"], return_counts=True)
            for val, c in zip(vals, cnt):
                t.c.setdefault(f"fmt{e.format}.raw_tail", Counter())[f"0x{int(val):08X}"] += int(c)
            t.hist(f"fmt{e.format}.raw_tail_distinct_per_entry", len(vals))
        if v.uv1 is not None:
            same = np.all(raw["uv1"].view(np.uint16) == raw["uv0"].view(np.uint16), axis=1)
            t.count(f"fmt{e.format}.uv1_eq_uv0", int(same.sum()))
            t.count(f"fmt{e.format}.uv1_zero", int(np.all(raw["uv1"].view(np.uint16) == 0, axis=1).sum()))
            t.count(f"fmt{e.format}.uv1_rows", n)
        if "raw_ext" in raw.dtype.names:
            z = np.all(raw["raw_ext"] == 0, axis=1)
            t.count("fmt8.ext_zero_rows", int(z.sum()))
            t.count("fmt8.ext_rows", n)
            nz = raw["raw_ext"][~z]
            if len(nz):
                # which of the 10 dwords are ever non-zero
                dw = nz.view("<u4").reshape(-1, 10)
                for k in range(10):
                    t.count(f"fmt8.ext_dword{k}_nonzero", int((dw[:, k] != 0).sum()))
        if v.skinned:
            sums = v.weight_sums()
            for val, c in zip(*np.unique(sums, return_counts=True)):
                t.c.setdefault("skin.weight_sum", Counter())[int(val)] += int(c)
            w = raw["weights"]
            j = raw["joints"]
            active = w > 0
            npal = max((len(s.palette) for s in e.submeshes), default=0)
            t.count("skin.joint_ge_maxpalette", int((j[active] >= npal).sum()) if npal else int(active.sum()))
            t.count("skin.lanes_active", int(active.sum()))
            t.count("skin.lanes", int(w.size))
            t.count("skin.inactive_joint_nonzero", int(((~active) & (j != 0)).sum()))
            infl = active.sum(axis=1)
            for k, c in zip(*np.unique(infl, return_counts=True)):
                t.c.setdefault("skin.influences", Counter())[int(k)] += int(c)
            # are weights sorted descending across lanes?
            t.count("skin.rows_sorted_desc", int(np.all(np.diff(w.astype(np.int16), axis=1) <= 0, axis=1).sum()))
            t.count("skin.rows", n)
        # qtangent: requantisation exactness and norm
        q = raw["qtan"].astype(np.int64)
        norm = np.linalg.norm(q / QTAN_SCALE[e.format], axis=1)
        if n:
            t.mm(f"fmt{e.format}.qtan_norm_min", float(norm.min()))
            t.mm(f"fmt{e.format}.qtan_norm_max", float(norm.max()))
            t.count("qtan.zero_quads", int((np.abs(q).sum(axis=1) == 0).sum()))
            enc = encode_qtangent(v.tangents, v.normals, v.tangent_sign, e.format).astype(np.int64)
            d = np.abs(enc - q).max(axis=1)
            t.count("qtan.requant_exact", int((d == 0).sum()))
            t.count("qtan.requant_off_by_1", int((d == 1).sum()))
            t.count("qtan.requant_off_more", int((d > 1).sum()))
            t.count("qtan.rows", n)
            t.count("qtan.sign_neg", int((v.tangent_sign < 0).sum()))


def _census_buffers(m, t: Tally, r) -> None:
    if m.vertex_buffer is None:
        t.count("buffers.absent")
        return
    vb = np.frombuffer(bytes(m.vertex_buffer), dtype=np.uint8)
    ib = np.frombuffer(bytes(m.index_buffer), dtype=np.uint8) if m.index_buffer is not None else np.zeros(0, np.uint8)
    entries = m.geometry_entries
    # rule: entries consecutive in entry order, vertex windows padded to 160, index bases aligned 4, part padded 16
    vcur = icur = 0
    rule_ok = True
    covered_v = np.zeros(len(vb), dtype=bool)
    covered_i = np.zeros(len(ib), dtype=bool)
    order_ok = all(entries[i].vertex_base <= entries[i + 1].vertex_base for i in range(len(entries) - 1))
    t.count("buffers.entries_in_vertex_order", int(order_ok))
    for e in entries:
        if e.format not in STRIDES:
            rule_ok = False
            continue
        vsize = e.vertex_count * STRIDES[e.format]
        if e.vertex_base != vcur or e.index_base != align_to(icur, INDEX_BASE_ALIGN):
            rule_ok = False
        covered_v[e.vertex_base : e.vertex_base + vsize] = True
        covered_i[e.index_base : e.index_base + e.index_count * 2] = True
        vcur = align_to(e.vertex_base + vsize, VERTEX_BLOCK_ALIGN)
        icur = e.index_base + e.index_count * 2
    vsize_ok = len(vb) == vcur
    isize_padded = len(ib) == align_to(icur, INDEX_BUFFER_ALIGN)
    isize_exact = len(ib) == icur
    size_ok = vsize_ok and (isize_padded or isize_exact)
    t.count("buffers.layout_rule_ok", int(rule_ok and size_ok))
    t.count("buffers.layout_rule_bases_ok", int(rule_ok))
    t.count("buffers.layout_rule_vertex_size_ok", int(vsize_ok))
    t.count("buffers.index_part_padded16", int(isize_padded))
    t.count("buffers.index_part_exact_unpadded", int(isize_exact and not isize_padded))
    if not (rule_ok and size_ok):
        t.note("buffers.layout_rule_violations", {"index": r.index, "name": r.name, "vsize": len(vb), "vplan": vcur,
                                                   "isize": len(ib), "iplan": align_to(icur, INDEX_BUFFER_ALIGN),
                                                   "entries": [(e.format, e.vertex_base, e.vertex_count, e.index_base, e.index_count) for e in entries]})
    t.count("buffers.vertex_gap_bytes", int((~covered_v).sum()))
    t.count("buffers.vertex_gap_nonzero", int(vb[~covered_v].any()))
    t.count("buffers.index_gap_bytes", int((~covered_i).sum()))
    t.count("buffers.index_gap_nonzero", int(ib[~covered_i].any()))
    t.hist("buffers.vertex_size_mod160", len(vb) % 160)
    t.hist("buffers.index_size_mod16", len(ib) % 16)
    t.count("buffers.cloth", int(m.cloth_raw is not None))
    if m.cloth_raw is not None:
        t.hist("cloth.magic", m.cloth_raw[:4].hex())
    if m.skin_raw is not None:
        t.hist("skin_part.size", len(m.skin_raw))
        t.hist("skin_part.head", m.skin_raw[:8].hex())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="*", help="pack paths (default: the 5 mesh packs of the game install)")
    ap.add_argument("--out", default=str(ROOT / "out" / "reports" / "mesh"))
    ap.add_argument("--limit", type=int, default=None, help="meshes per pack (testing)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    packs = [Path(p) for p in a.packs] if a.packs else [ASSETS / n for n in MESH_PACKS]
    total = Tally()
    t_all = time.time()
    for p in packs:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            continue
        t = Tally()
        t0 = time.time()
        census_pack(p, t, sample_limit=a.limit)
        secs = round(time.time() - t0, 1)
        rep = {"pack": str(p), "seconds": secs, **t.to_json()}
        with open(out / f"census_{p.stem}.json", "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1, ensure_ascii=False, default=str)
        print(f"{p.name:32s} meshes={t.n['meshes']:6d} failures={t.n.get('decode_failures_total', 0)} "
              f"fmt={dict(t.c.get('format', {}))} {secs}s")
        total.merge(t)
    rep = {"packs": [str(p) for p in packs], "seconds": round(time.time() - t_all, 1), **total.to_json()}
    with open(out / "census_summary.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: v for k, v in rep["counts"].items()}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
