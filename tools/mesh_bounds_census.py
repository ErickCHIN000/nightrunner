"""Corpus census: how entity bounds (+0x60 centre, +0x6C half extent) relate to the positions of the geometry
entries the entity owns (entity +0x88 → class-6 array). Establishes the rule the phase-2 encoder uses when it
recomputes bounds after a geometry edit.

    python tools/mesh_bounds_census.py [--packs a.rpack ...] [--out out/reports/mesh/bounds_census.json]

For every entity with geometry_count > 0 and decodable vertices, computes the AABB of
  (a) entry 0's raw positions (all vertices of the window, finite only),
  (b) the union over every owned entry,
and reports the max abs deviation of stored (centre, half) from each, bucketed, plus containment
(stored box ⊇ union box) and per-format breakdowns.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.mesh.decode import decode_resource  # noqa: E402
from tests.paths import ASSETS  # noqa: E402

MESH_PACKS = ["common_meshes_pc.rpack", "dlc_frontier_pc.rpack", "dlc_ft_prologue_pc.rpack", "menu_level_ft_pc.rpack",
              "engine_pc.rpack"]
BUCKETS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, float("inf")]


def bucket(x: float) -> str:
    for b in BUCKETS:
        if x <= b:
            return f"<={b:g}"
    return ">1"


def aabb(p: np.ndarray):
    p = p.astype(np.float64)
    p = p[np.isfinite(p).all(axis=1)]
    if not len(p):
        return None
    lo, hi = p.min(0), p.max(0)
    return (lo + hi) / 2, (hi - lo) / 2, lo, hi


def census(paths: list[Path]) -> dict:
    n = Counter()
    h: dict[str, Counter] = {}
    examples: list = []
    t0 = time.time()
    for path in paths:
        with Pack.open(path) as pk:
            for r in pk.resources_of_type(0x10):
                try:
                    m = decode_resource(r)
                except Exception as exc:  # noqa: BLE001
                    n["decode_failures"] += 1
                    continue
                for en in m.entities:
                    if not en.geometry_entries:
                        continue
                    ents = [m.geometry_entries[i] for i in en.geometry_entries]
                    if any(e.vertices is None for e in ents):
                        n["entity.no_vertices"] += 1
                        continue
                    n["entity.with_geometry"] += 1
                    b0 = aabb(ents[0].vertices.positions)
                    allp = np.concatenate([e.vertices.positions for e in ents])
                    bu = aabb(allp)
                    if b0 is None or bu is None:
                        n["entity.empty"] += 1
                        continue
                    sc, sh = en.bounds_center.astype(np.float64), en.bounds_half.astype(np.float64)
                    d0 = max(np.abs(b0[0] - sc).max(), np.abs(b0[1] - sh).max())
                    du = max(np.abs(bu[0] - sc).max(), np.abs(bu[1] - sh).max())
                    fmt = ",".join(str(e.format) for e in ents)
                    h.setdefault("dev_entry0", Counter())[bucket(d0)] += 1
                    h.setdefault("dev_union", Counter())[bucket(du)] += 1
                    h.setdefault(f"dev_union_fmt{ents[0].format}", Counter())[bucket(du)] += 1
                    h.setdefault(f"dev_union_entries{len(ents)}", Counter())[bucket(du)] += 1
                    lo_s, hi_s = sc - sh, sc + sh
                    contains = bool((lo_s <= bu[2] + 1e-4).all() and (hi_s >= bu[3] - 1e-4).all())
                    n["union_contained"] += int(contains)
                    # relative deviation (vs extent) for the non-tiny cases
                    ext = max(float(sh.max()), 1e-9)
                    h.setdefault("dev_union_rel", Counter())[bucket(du / ext)] += 1
                    if du > 1e-3 and len(examples) < 60:
                        examples.append({"pack": path.name, "index": r.index, "name": r.name, "entity": en.index,
                                         "formats": fmt, "entries": len(ents), "dev_entry0": d0, "dev_union": du,
                                         "stored_center": sc.tolist(), "stored_half": sh.tolist(),
                                         "union_center": bu[0].tolist(), "union_half": bu[1].tolist(),
                                         "contains": contains})
                    n["entity.multi_entry"] += int(len(ents) > 1)
                    if len(ents) > 1:
                        h.setdefault("dev_entry0_multi", Counter())[bucket(d0)] += 1
                        h.setdefault("dev_union_multi", Counter())[bucket(du)] += 1
                # entities WITHOUT geometry: are their bounds zero?
                for en in m.entities:
                    if en.geometry_entries:
                        continue
                    z = bool((en.bounds_center == 0).all() and (en.bounds_half == 0).all())
                    n["entity.no_geometry"] += 1
                    n["entity.no_geometry_bounds_zero"] += int(z)
    return {"packs": [str(p) for p in paths], "seconds": round(time.time() - t0, 1),
            "counts": dict(sorted(n.items())),
            "hist": {k: dict(sorted(v.items())) for k, v in sorted(h.items())}, "examples": examples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="*")
    ap.add_argument("--out", default=str(ROOT / "out" / "reports" / "mesh" / "bounds_census.json"))
    a = ap.parse_args()
    paths = [Path(p) for p in a.packs] if a.packs else [ASSETS / p for p in MESH_PACKS if (ASSETS / p).exists()]
    rep = census(paths)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"counts": rep["counts"], "hist": rep["hist"]}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
