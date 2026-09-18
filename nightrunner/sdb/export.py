"""Whole-database JSON dumps of a shader database: materials, presets, textures.

Qt-free, so the GUI, the tests and any script share one implementation.

Three documents, each self-describing (`schema`, `tool`, `database`, `generated`, `counts`):

* **materials** - every material with its routes: the preset it uses, every parameter with its decoded value, and
  every shader variant with its texture bindings. Optionally what uses each material.
* **presets** - every preset with its declared parameters, their types and decoded defaults.
* **textures** - every texture name the database binds, with the materials and parameters that bind it.

No raw bytes. The reader hands out hex for record blobs, packed words and offsets; none of that survives into these
documents, because a dump is for reading and diffing, not for rebuilding a database. `nr sdb material --raw` and
the SDB tab's raw view remain the way to see bytes. There is no SDB writer, so nothing here round-trips.

Everything decoded is the reader's own output (`Sdb.material`, `Sdb.preset`), so the provenance of every field is
whatever `notes/FORMATS/sdb.md` records for it; fields the reader marks as undecoded are simply absent.
"""
from __future__ import annotations

import datetime as _dt
import struct
from pathlib import Path
from typing import Callable, Iterable

from .. import __version__
from ..util.jsonio import dump_json
from .reader import VALUE_FORMATS, VALUE_TYPES, Sdb

SCHEMA_MATERIALS = "nightrunner.sdb.materials/1"
SCHEMA_PRESETS = "nightrunner.sdb.presets/1"
SCHEMA_TEXTURES = "nightrunner.sdb.textures/1"

FILE_NAMES = {"materials": "sdb_materials.json", "presets": "sdb_presets.json", "textures": "sdb_textures.json"}

#: Indentation per document. The presets file is small enough to read by eye; the other two are tens of thousands
#: of records, where indenting doubles the size of a file nobody scrolls through by hand (materials: 113 MB
#: compact against 236 MB at indent=2, measured 2026-09-17). Pretty-print them with `jq` when you need to look.
INDENT = {"materials": None, "presets": 2, "textures": None}


class Cancelled(RuntimeError):
    """Raised out of a dump when the caller's cancel callback asked it to stop."""


def type_name(type_id: int) -> str:
    return VALUE_TYPES[type_id] if 0 <= type_id < len(VALUE_TYPES) else f"unknown_{type_id}"


def decode_default(sdb: Sdb, type_id: int, default_hex: str):
    """A preset default decoded with the parameter's value type; None when the bytes do not fit the type."""
    fmt = VALUE_FORMATS.get(type_id)
    if fmt is None:
        return None
    raw = bytes.fromhex(default_hex or "")
    if len(raw) != struct.calcsize(fmt):
        return None
    val = struct.unpack(fmt, raw)
    if type_id == 7:
        sv = sdb.string_value(val[0])
        return sv["name"] if sv.get("name") is not None else f"<runtime {sv['runtime_index']}>"
    return val[0] if len(val) == 1 else list(val)


def _header(sdb: Sdb, schema: str) -> dict:
    return {
        "schema": schema,
        "tool": f"nightrunner {__version__}",
        "generated": _dt.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "database": {"name": Path(sdb.path).name, "path": str(sdb.path), "size": len(sdb.data)},
    }


def _tick(progress: Callable[[int, int], None] | None, cancel: Callable[[], bool] | None, i: int, n: int) -> None:
    if cancel is not None and cancel():
        raise Cancelled(f"cancelled after {i:,} of {n:,}")
    if progress is not None and (i % 256 == 0 or i == n):
        progress(i, n)


# ---- materials -------------------------------------------------------------------------------------------------

def _clean_binding(b: dict) -> dict:
    """One texture binding: which parameter, which texture, and whether the material set it or the shader did."""
    out = {"param": b.get("param"), "texture": b.get("texture"), "source": b.get("source")}
    if b.get("texture") is None and "runtime_index" in b:
        out["runtime_index"] = b["runtime_index"]           # a string the database resolves at runtime
    return out


def _clean_parameter(p: dict) -> dict:
    out = {"id": p.get("id"), "name": p.get("name"), "declared": p.get("declared")}
    if "type" in p:
        out["type"] = p["type"]
    if "value" in p:
        out["value"] = p["value"]
    return out


def _clean_parameters(params: list[dict]) -> list[dict]:
    """Route parameters with exact duplicates collapsed.

    A route has one slot per parameter occurrence, and the same parameter id can appear in several slots. census
    2026-09-17 (runtime_dx11, 348,774 parameter rows): 2,415 rows repeat an id already seen, and 2,412 of those
    carry an identical value - noise worth collapsing. The other **3 disagree**: `det_0_a_rot_ang` is 90 and 270
    in both `dlc_ft_collapsing_plank.mat` and `man_srv_scarf_a.mat`, and `det_0_a_dtm_dif` is 0.6 and 0.75 in
    `sh_baker.mat`, each pair sitting at different offsets in the value blob. Which one the engine uses is not
    decoded, so only byte-identical rows are collapsed; a genuine disagreement stays in the document as two
    entries with the same id rather than being silently resolved here.
    """
    out, seen = [], set()
    for p in params:
        rec = _clean_parameter(p)
        key = repr(sorted(rec.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _group_variants(variants: list[dict]) -> list[dict]:
    """Shader variants grouped by the texture bindings they share.

    A material carries one variant per shader selector, and 68% of them (census 2026-09-17, every 13th material)
    bind exactly the same textures as another variant of the same material. Listing each variant in full repeats
    those bindings for no gain and buries the thing a reader is looking for: which textures this material uses
    under which conditions. Grouping keeps every selector and shader id, and states each distinct binding set once.
    """
    groups: list[dict] = []
    index: dict[str, int] = {}
    for v in variants:
        textures = [_clean_binding(b) for b in v.get("texture_bindings") or []]
        passes = v.get("render_pass_count")
        key = repr((passes, textures))
        i = index.get(key)
        if i is None:
            index[key] = len(groups)
            groups.append({"render_passes": passes, "textures": textures, "selectors": []})
            i = index[key]
        groups[i]["selectors"].append({"selector": v.get("selector_hex"), "shader": v.get("shader")})
    return groups


def clean_material(m: dict) -> dict:
    """A resolved material (`Sdb.material`) reduced to decoded meaning, with the convenience lists a reader wants."""
    routes, presets, textures = [], [], []
    for r in m.get("routes") or []:
        if r.get("preset") and r["preset"] not in presets:
            presets.append(r["preset"])
        for v in r.get("variants") or []:
            for b in v.get("texture_bindings") or []:
                if b.get("texture") and b["texture"] not in textures:
                    textures.append(b["texture"])
        groups = _group_variants(r.get("variants") or [])
        routes.append({"preset": r.get("preset"), "tokens": r.get("tokens"),
                       "variant_count": sum(len(g["selectors"]) for g in groups),
                       "parameters": _clean_parameters(r.get("parameters") or []),
                       "variants": groups})
    rec = {"index": m.get("index"), "name": m.get("name"), "non_rendering": m.get("non_rendering"),
           "textures": sorted(textures)}
    if len(routes) == 1:
        # census 2026-09-17: every material has exactly one route - 26,670/26,670 (runtime_dx11) and
        # 26,665/26,665 (runtime_dx12). A one-element "routes" array would be a nesting level that never
        # branches, so the single route is flattened onto the material. A material with any other number keeps
        # the array, so a database that does branch is represented honestly rather than silently truncated.
        rec.update({k: v for k, v in routes[0].items() if k != "variant_count"})
        rec["variant_count"] = routes[0]["variant_count"]
    else:
        rec["presets"] = presets
        rec["routes"] = routes
    return rec


def materials_document(sdb: Sdb, *, used_by: dict | None = None, progress=None, cancel=None,
                       names: Iterable[str] | None = None) -> dict:
    """Every material (or just *names*), cleaned, with `used_by` attached when the caller has it.

    *used_by* is ``{"models": {casefold name: [...]}, "meshes": {casefold name: [...]}, "complete": bool}``. It is
    the caller's job to produce it (the GUI scans the PAKs and the mesh corpus for it); a material with no rows in
    either index gets an empty record rather than being left out, so "nothing uses this" is stated, not implied.
    """
    mats = list(names) if names is not None else sdb.materials()
    models = (used_by or {}).get("models") or {}
    meshes = (used_by or {}).get("meshes") or {}
    out, failed = [], []
    n = len(mats)
    for i, name in enumerate(mats, 1):
        _tick(progress, cancel, i, n)
        try:
            rec = clean_material(sdb.material(name))
        except Exception as exc:                            # a material that will not resolve is reported, not fatal
            failed.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if used_by is not None:
            key = (rec["name"] or "").casefold()
            rec["used_by"] = {"models": models.get(key) or [], "meshes": meshes.get(key) or []}
        out.append(rec)
    doc = _header(sdb, SCHEMA_MATERIALS)
    doc["counts"] = {"materials": len(out), "failed": len(failed),
                     "textures": len({t for m in out for t in m["textures"]})}
    if used_by is not None:
        doc["used_by"] = {"complete": bool((used_by or {}).get("complete")),
                          "models_scanned": (used_by or {}).get("models_scanned"),
                          "meshes_scanned": (used_by or {}).get("meshes_scanned")}
    if failed:
        doc["failed"] = failed
    doc["materials"] = out
    return doc


# ---- presets ---------------------------------------------------------------------------------------------------

def clean_preset(sdb: Sdb, p: dict) -> dict:
    params = []
    for q in p.get("parameters") or []:
        params.append({"id": q.get("id"), "name": q.get("name"), "type": q.get("type_name"),
                       "expression": q.get("expression") or None, "annotation": q.get("annotation") or None,
                       "default": decode_default(sdb, q.get("type", -1), q.get("default_hex", ""))})
    return {"index": p.get("index"), "name": p.get("name"), "key": p.get("key"), "flags": p.get("flags"),
            "parameters": params, "groups": [{"key": g.get("key"), "values": g.get("values")}
                                             for g in p.get("groups_b") or []]}


def presets_document(sdb: Sdb, *, progress=None, cancel=None) -> dict:
    n = sdb.table(0xAA).count
    out, failed = [], []
    for i in range(n):
        _tick(progress, cancel, i + 1, n)
        try:
            out.append(clean_preset(sdb, sdb.preset(i)))
        except Exception as exc:
            failed.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
    doc = _header(sdb, SCHEMA_PRESETS)
    doc["counts"] = {"presets": len(out), "failed": len(failed),
                     "parameters": sum(len(p["parameters"]) for p in out)}
    if failed:
        doc["failed"] = failed
    doc["presets"] = out
    return doc


# ---- textures --------------------------------------------------------------------------------------------------

def variant_groups(material: dict) -> list[dict]:
    """The variant groups of a cleaned material, whether its single route was flattened onto it or not."""
    if "variants" in material:
        return material.get("variants") or []
    return [v for r in material.get("routes") or [] for v in r.get("variants") or []]


def textures_document(sdb: Sdb, materials: dict, *, providers: dict | None = None) -> dict:
    """The reverse index, derived from an already-built materials document.

    Built from `materials` rather than from a second walk of the database, so the two documents can never disagree.
    *providers* maps a texture name to where it lives in the game (the GUI passes catalog pack labels); textures
    the game does not ship are marked `in_game: false` when it is given.
    """
    rows: dict[str, dict] = {}
    for m in materials.get("materials") or []:
        for v in variant_groups(m):
            for b in v.get("textures") or []:
                if True:
                    name = b.get("texture")
                    if not name:
                        continue
                    row = rows.setdefault(name, {"name": name, "materials": [], "parameters": [], "bindings": 0})
                    row["bindings"] += 1
                    if m["name"] not in row["materials"]:
                        row["materials"].append(m["name"])
                    p = b.get("param")
                    if p and p not in row["parameters"]:
                        row["parameters"].append(p)
    out = []
    for name in sorted(rows, key=str.casefold):
        row = rows[name]
        row["materials"].sort(key=str.casefold)
        row["parameters"].sort(key=str.casefold)
        row["material_count"] = len(row["materials"])
        if providers is not None:
            packs = providers.get(name) or providers.get(name.casefold()) or []
            row["in_game"] = bool(packs)
            row["packs"] = list(packs)
        out.append(row)
    doc = _header(sdb, SCHEMA_TEXTURES)
    doc["counts"] = {"textures": len(out), "bindings": sum(r["bindings"] for r in out)}
    if providers is not None:
        doc["counts"]["missing_from_game"] = sum(1 for r in out if not r["in_game"])
    doc["source"] = {"schema": materials.get("schema"), "materials": (materials.get("counts") or {}).get("materials")}
    doc["textures"] = out
    return doc


# ---- all three -------------------------------------------------------------------------------------------------

def export_all(sdb: Sdb, out_dir: Path, *, used_by: dict | None = None, providers: dict | None = None,
               progress=None, cancel=None, names: Iterable[str] | None = None, indent: int | None = None) -> dict:
    """Write all three documents into *out_dir*. Returns {kind: {"path", "count", "bytes"}}.

    Indentation follows `INDENT` unless *indent* overrides it.

    `progress(stage, done, total)` is called as it goes; `cancel()` returning True raises `Cancelled` before the
    next material, leaving whatever was already written in place.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = (lambda s: (lambda d, t: progress(s, d, t))) if progress is not None else (lambda s: None)

    mats = materials_document(sdb, used_by=used_by, progress=stage("materials"), cancel=cancel, names=names)
    pres = presets_document(sdb, progress=stage("presets"), cancel=cancel)
    texs = textures_document(sdb, mats, providers=providers)

    written = {}
    for kind, doc, count in (("materials", mats, len(mats["materials"])),
                             ("presets", pres, len(pres["presets"])),
                             ("textures", texs, len(texs["textures"]))):
        path = out_dir / FILE_NAMES[kind]
        dump_json(doc, path, indent=INDENT[kind] if indent is None else indent)
        written[kind] = {"path": str(path), "count": count, "bytes": path.stat().st_size}
    return written
