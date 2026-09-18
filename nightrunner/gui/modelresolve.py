"""Full resolution of a `.model` document: skeleton → slots → meshes → submeshes → materials → SDB → textures.

Pure (no widgets) and worker-safe. `resolve_model(ctx_like, model_json, variant=None)` only needs
``ctx_like.catalog`` (`lookup(name, type)`, `entry(gid).label`, `split(gid)` for decoding found meshes) and
``ctx_like.sdb`` (`material_info(name)` → `Sdb.material()` dict). Tests pass stubs.

Chain (notes/GUI-models.md; tiers from notes/FORMATS/pak-and-model.md):

  preset.skeletonName → catalog type 0x10 (all providers)
  slot → meshResources.resources[] (every entry; `chosen` = selected, else first)                         (A schema)
  mesh .msh name → catalog type 0x10 gids (all providers) | missing
  submeshes: decoded from the first provider (entry, submesh, material slot, embedded material name);
             missing mesh → one pseudo-submesh per materialsData row (embedded names as the model states them)
  mesh variant (part 0x12, optional): embedded name → variant material name, applied BEFORE the join
  join: name → materialsData.number → materialsResources[number] → selected/first                         (B)
  effective material = selected .mat (base, MatLoaded first: A) + rttiValues                              (A parse)
  SDB: route parameters + texture bindings (union over shader variants, first texture per parameter)      (A)
  type-7 rttiValue on a parameter name → texture replaced (original kept); type 2/4 → value override        (A/C)
  every final texture → catalog type 0x20 gids | missing
"""
from __future__ import annotations

import threading
from typing import Any, Callable

MESH_TYPE = 0x10
TEXTURE_TYPE = 0x20
RTTI_KIND = {7: "texture", 2: "float", 4: "vec3"}

# SdbService / Sdb build lazy indices on first use; one resolution at a time keeps them race-free.
_SDB_LOCK = threading.RLock()


# ---- small helpers ---------------------------------------------------------------------------------------------

def _mesh_key(name: str) -> str:
    k = (name or "").strip().casefold()
    return k if k.endswith(".msh") else k + ".msh"


def _mat_key(name: str | None) -> str:
    k = (name or "").strip().casefold()
    return k[:-4] if k.endswith(".mat") else k


def _selected(resources: list) -> tuple[dict | None, str | None]:
    """(chosen entry, problem) — selected entry, else the first; several selected → first selected + a note."""
    items = [r for r in resources or [] if isinstance(r, dict)]
    sel = [r for r in items if r.get("selected")]
    if len(sel) > 1:
        return sel[0], f"{len(sel)} entries are 'selected' (engine precedence unknown, P4); first one used"
    if sel:
        return sel[0], None
    return (items[0], "no entry is 'selected'; first one used") if items else (None, None)


def _rtti_value(v: dict) -> Any:
    t = v.get("type")
    if t == 7:
        return v.get("val_str")
    if t == 2:
        return v.get("val_float")
    if t == 4:
        return v.get("val_vec3")
    return next((v[k] for k in v if isinstance(k, str) and k.startswith("val_")), None)


def _providers(catalog, gids: list[int]) -> list[str]:
    out = []
    for g in gids:
        try:
            out.append(catalog.entry(g).label)
        except Exception:  # noqa: BLE001 - stub catalogs may not know labels
            out.append(f"gid {g}")
    return out


def _lookup(catalog, name: str, type_id: int) -> list[int]:
    if not name:
        return []
    try:
        return [int(g) for g in catalog.lookup(name, type_id)]
    except Exception:  # noqa: BLE001
        return []


def _status(gids: list[int], providers: list[str]) -> str:
    return f"found in {', '.join(providers)}" if gids else "missing"


# ---- mesh reading ----------------------------------------------------------------------------------------------

def read_mesh_info(catalog, gid: int) -> dict:
    """Submesh table (+ named variants when the part-0x12 decoder is available) of one catalog mesh.

    Returns {"submeshes": [{entry, submesh, slot, material}], "materials": [...], "variants": [...],
    "error": str|None, "variants_error": str|None}. Uses nightrunner.mesh decoding directly (cheap: no vertex
    maths) so it works whether or not gui/meshdata.py is present; meshdata.mesh_materials is the last resort."""
    info: dict = {"submeshes": [], "materials": [], "variants": [], "error": None, "variants_error": None}
    try:
        entry, index = catalog.split(gid)
        res = entry.pack.resource(index)
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"cannot open resource: {type(exc).__name__}: {exc}"
        return info
    model = None
    try:
        from ..mesh.decode import decode_resource
        model = decode_resource(res)
        info["materials"] = [m.name_str for m in model.materials]
        for e in model.geometry_entries:
            for s in e.submeshes:
                info["submeshes"].append({"entry": e.index, "submesh": s.index, "slot": s.material_slot,
                                          "material": model.material_name(s.material_slot),
                                          "triangles": s.triangle_count})
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"mesh decode failed: {type(exc).__name__}: {exc}"
        try:
            from . import meshdata  # type: ignore[attr-defined]
            info["materials"] = list(meshdata.mesh_materials(entry.pack, index))
        except Exception:  # noqa: BLE001
            pass
    try:
        skin = res.read_part_by_type(0x12)
    except Exception:  # noqa: BLE001
        skin = None
    if skin is not None:
        try:
            from ..mesh import variants as mv  # type: ignore[attr-defined]
            doc = mv.decode(bytes(skin), model)
            # pairs index the FULL material table (incl. variant-only entries past `count`), see mesh-variants.md
            full = mv.material_table(model) if model is not None else []
            info["variants"] = normalize_variants(doc, full or info["materials"])
        except ImportError:
            info["variants_error"] = "nightrunner.mesh.variants not available"
        except Exception as exc:  # noqa: BLE001
            info["variants_error"] = f"{type(exc).__name__}: {exc}"
    return info


def normalize_variants(doc: Any, materials: list[str]) -> list[dict]:
    """Variant decoder output → [{"name", "map": {embedded material casefold: variant material}}].

    `material_map` is accepted as {material name | slot index: material name | slot index} or as a list aligned
    with the mesh material table (None / "" = unchanged)."""
    raw = doc.get("variants") if isinstance(doc, dict) else doc
    out = []
    for v in raw or []:
        if not isinstance(v, dict):
            continue
        mm = v.get("material_map")
        mapping: dict[str, str] = {}

        def name_of(x):
            if isinstance(x, int) or (isinstance(x, str) and x.isdigit()):
                i = int(x)
                return materials[i] if 0 <= i < len(materials) else None
            if isinstance(x, dict):
                return x.get("name") or x.get("material")
            return x

        if isinstance(mm, dict):
            items = list(mm.items())
        elif isinstance(mm, list):
            items = []
            for i, x in enumerate(mm):
                if isinstance(x, dict) and "slot" in x and "material" in x:
                    # nightrunner.mesh.variants: {slot, material[, slot_name, material_name]} table indices
                    items.append((x.get("slot_name") or x["slot"], x.get("material_name") or x["material"]))
                elif isinstance(x, dict) and ("from" in x or "default" in x):
                    items.append((x.get("from", x.get("default")), x.get("to", x.get("material"))))
                else:
                    items.append((i, x))
        else:
            items = []
        for k, val in items:
            src, dst = name_of(k), name_of(val)
            if src and dst and str(src).casefold() != str(dst).casefold():
                mapping[str(src).casefold()] = str(dst)
        out.append({"name": str(v.get("name", "")), "map": mapping, "material_map": mm})
    return out


# ---- SDB --------------------------------------------------------------------------------------------------------

def sdb_material(sdb, name: str, cache: dict) -> dict:
    """{found, name, index, preset, tokens, non_rendering, parameters[{name,type,value}],
    bindings[{param, param_id, texture, source, variants}]} for an SDB material (case-insensitive, .mat optional)."""
    key = _mat_key(name)
    if key in cache:
        return cache[key]
    info = None
    if sdb is not None and name:
        with _SDB_LOCK:
            try:
                info = sdb.material_info(name)
                if info is None and not name.lower().endswith(".mat"):
                    info = sdb.material_info(name + ".mat")
            except Exception:  # noqa: BLE001
                info = None
    if not info:
        out = {"found": False, "name": name, "parameters": [], "bindings": []}
        cache[key] = out
        return out
    params, bindings, seen_p = [], {}, set()
    presets, tokens = [], []
    for route in info.get("routes") or []:
        if route.get("preset"):
            presets.append(route["preset"])
        tokens.append(route.get("tokens"))
        for p in route.get("parameters") or []:
            k = (p.get("name"), p.get("offset"))
            if k in seen_p:
                continue
            seen_p.add(k)
            params.append({"name": p.get("name"), "id": p.get("id"), "type": p.get("type"), "value": p.get("value"),
                           "string": p.get("string")})
        for var in route.get("variants") or []:
            for b in var.get("texture_bindings") or []:
                pname = b.get("param") or f"binding_{b.get('binding')}"
                cur = bindings.get(pname)
                if cur is None:
                    bindings[pname] = {"param": pname, "param_id": b.get("param_id"), "texture": b.get("texture"),
                                       "source": b.get("source"), "variants": 1, "others": []}
                else:
                    cur["variants"] += 1
                    t = b.get("texture")
                    if t and t != cur["texture"] and t not in cur["others"]:
                        cur["others"].append(t)
    out = {"found": True, "name": info.get("name", name), "index": info.get("index"),
           "preset": ", ".join(dict.fromkeys(presets)) or None, "tokens": tokens[0] if tokens else None,
           "non_rendering": bool(info.get("non_rendering")), "parameters": params,
           "bindings": list(bindings.values())}
    cache[key] = out
    return out


# ---- main entry --------------------------------------------------------------------------------------------------

def mesh_variant_names(resolution: dict) -> list[str]:
    """Union of variant names over every mesh of a resolution, "Default" first."""
    names: list[str] = []
    for slot in resolution.get("slots", []):
        for m in slot.get("meshes", []):
            for v in m.get("variants", []):
                n = v.get("name")
                if n and n not in names:
                    names.append(n)
    rest = [n for n in names if n.casefold() != "default"]
    return ["Default"] + rest


def resolve_model(ctx_like, model_json: dict, variant: str | None = None, *, name: str | None = None,
                  mesh_reader: Callable[[Any, int], dict] | None = None) -> dict:
    """The complete mapping of a `.model` document (see module docstring). `variant` = a mesh variant name
    (None / "Default" = embedded materials). `mesh_reader(catalog, gid)` overrides `read_mesh_info` (tests)."""
    catalog = ctx_like.catalog
    sdb = getattr(ctx_like, "sdb", None)
    reader = mesh_reader or read_mesh_info
    use_variant = variant if variant and variant.casefold() != "default" else None
    sdb_cache: dict = {}
    mesh_cache: dict[int, dict] = {}
    notes: list[str] = []

    preset = model_json.get("preset") or {}
    skel = preset.get("skeletonName")
    skel_gids = _lookup(catalog, skel, MESH_TYPE) if skel else []
    skel_prov = _providers(catalog, skel_gids)

    slots_out = []
    for slot in model_json.get("slots") or []:
        if not isinstance(slot, dict):
            notes.append("non-object slot skipped")
            continue
        resources = (slot.get("meshResources") or {}).get("resources") or []
        chosen, problem = _selected(resources)
        meshes_out = []
        for r in resources:
            if not isinstance(r, dict):
                continue
            meshes_out.append(_resolve_mesh(catalog, sdb, reader, r, r is chosen, use_variant, sdb_cache,
                                            mesh_cache))
        slots_out.append({"name": slot.get("name"), "slotUid": slot.get("slotUid"),
                          "filterText": slot.get("filterText"), "problem": problem, "meshes": meshes_out,
                          "has_cloth": bool(slot.get("clothResources"))})

    res = {"name": name, "version": model_json.get("version"),
           "skeleton": {"name": skel, "gids": skel_gids, "providers": skel_prov,
                        "status": _status(skel_gids, skel_prov) if skel else "none"},
           "properties": (model_json.get("data") or {}).get("properties") or [],
           "variant": use_variant or "Default", "slots": slots_out, "notes": notes,
           "has_pose_items": bool(model_json.get("poseItems"))}
    res["variants"] = mesh_variant_names(res)
    if use_variant and use_variant not in res["variants"]:
        notes.append(f"variant {use_variant!r} is not defined by any mesh of this model")
    res.update(_summarize(res))
    return res


def _resolve_mesh(catalog, sdb, reader, r: dict, chosen: bool, variant: str | None, sdb_cache: dict,
                  mesh_cache: dict) -> dict:
    name = r.get("name") or ""
    gids = _lookup(catalog, name, MESH_TYPE)
    prov = _providers(catalog, gids)
    md_rows = [md for md in r.get("materialsData") or [] if isinstance(md, dict)]
    out = {"name": name, "selected": bool(r.get("selected")), "chosen": chosen, "layoutId": r.get("layoutId"),
           "gids": gids, "providers": prov, "status": _status(gids, prov), "variants": [], "error": None,
           "variants_error": None, "mesh_materials": [], "submeshes": [], "problems": []}
    info = None
    if gids:
        g = gids[0]
        if g not in mesh_cache:
            try:
                mesh_cache[g] = reader(catalog, g)
            except Exception as exc:  # noqa: BLE001
                mesh_cache[g] = {"error": f"{type(exc).__name__}: {exc}"}
        info = mesh_cache[g]
        out["error"] = info.get("error")
        out["variants_error"] = info.get("variants_error")
        out["variants"] = [{"name": v["name"], "material_map": v.get("material_map"), "map": v.get("map", {})}
                           for v in info.get("variants") or []]
        out["mesh_materials"] = list(info.get("materials") or [])
    vmap: dict[str, str] = {}
    if variant:
        for v in out["variants"]:
            if v["name"].casefold() == variant.casefold():
                vmap = v.get("map") or {}
    subs = (info or {}).get("submeshes") or []
    if subs:
        source = "mesh"
        rows = [dict(s) for s in subs]
    else:
        # mesh missing (or undecodable): the model's materialsData rows are what the model says the mesh embeds
        source = "model materialsData"
        rows = [{"entry": None, "submesh": None, "slot": None, "material": md.get("name")} for md in md_rows]
    listed = {(md.get("name") or "").casefold() for md in md_rows}
    for s in rows:
        embedded = s.get("material") or ""
        vmat = vmap.get(embedded.casefold())
        lookup_name = vmat or embedded
        sub = {"entry": s.get("entry"), "submesh": s.get("submesh"), "slot": s.get("slot"),
               "triangles": s.get("triangles"), "embedded_material": embedded, "embedded_source": source,
               "variant_material": vmat}
        sub.update(_join(r, md_rows, lookup_name))
        if sub["base_material"] is None:
            sub["base_material"] = lookup_name
            sub["material_source"] = "variant" if vmat else "mesh default"
        elif vmat:
            sub["material_source"] = "variant"
        sub.update(_material_textures(catalog, sdb, sub, sdb_cache))
        out["submeshes"].append(sub)
    if subs:
        seen = {(s.get("material") or "").casefold() for s in subs}
        unused = sorted(n for n in listed - seen if n and not n.startswith(("auto_shadow_caster", "shadowcaster")))
        if unused:
            out["problems"].append(f"materialsData names not used by any submesh: {', '.join(unused)}")
    return out


def _join(r: dict, md_rows: list[dict], material: str) -> dict:
    """The join rule for one submesh material name (pak-and-model.md §3). Errors are recorded, never raised."""
    out = {"number": None, "base_material": None, "material_source": None, "alternatives": [], "rtti": [],
           "join_error": None, "loadFlags": None}
    hits = [md for md in md_rows if (md.get("name") or "").casefold() == (material or "").casefold()]
    if not hits:
        return out
    if len(hits) > 1:
        out["join_error"] = f"{material!r} listed {len(hits)} times in materialsData"
        return out
    num = hits[0].get("number")
    out["number"] = num
    groups = [gr for gr in r.get("materialsResources") or [] if isinstance(gr, dict) and gr.get("number") == num]
    if not groups:
        out["join_error"] = f"no materialsResources group for number {num}"
        return out
    if len(groups) > 1:
        out["join_error"] = f"materialsResources number {num} is duplicated; first group used"
    entries = [e for e in groups[0].get("resources") or [] if isinstance(e, dict)]
    choice, problem = _selected(entries)
    if choice is None:
        out["join_error"] = f"materialsResources group {num} is empty"
        return out
    if problem:
        out["join_error"] = problem
    out["base_material"] = choice.get("name")
    out["loadFlags"] = choice.get("loadFlags")
    remapped = (choice.get("name") or "").casefold() != (material or "").casefold()
    out["material_source"] = "model (remap)" if remapped else "model"
    out["alternatives"] = [e.get("name") for e in entries if e is not choice]
    for v in choice.get("rttiValues") or []:
        if isinstance(v, dict):
            out["rtti"].append({"name": v.get("name"), "type": v.get("type"),
                                "kind": RTTI_KIND.get(v.get("type"), f"type {v.get('type')}"),
                                "value": _rtti_value(v)})
    return out


def _material_textures(catalog, sdb, sub: dict, sdb_cache: dict) -> dict:
    base = sub["base_material"]
    m = sdb_material(sdb, base, sdb_cache)
    tex_rows = []
    by_param = {}
    for b in m["bindings"]:
        row = {"param": b["param"], "texture": b["texture"], "original": None,
               "source": "sdb override" if b["source"] == "override" else
               "sdb shader default" if b["source"] == "shader_default" else f"sdb {b['source']}",
               "others": b.get("others", [])}
        tex_rows.append(row)
        by_param[(b["param"] or "").casefold()] = row
    pvals = {(p["name"] or "").casefold(): p for p in m["parameters"]}
    overrides = []
    for o in sub["rtti"]:
        key = (o["name"] or "").casefold()
        ov = dict(o)
        if o["type"] == 7:
            row = by_param.get(key)
            if row is not None:
                ov["original"] = row["texture"]
                ov["applied"] = True
                row["original"], row["texture"], row["source"] = row["texture"], o["value"], "model override"
            else:
                ov["original"] = None
                ov["applied"] = False
                ov["note"] = ("parameter not bound by the SDB material" if m["found"] else "material not in SDB")
                row = {"param": o["name"], "texture": o["value"], "original": None, "source": "model override",
                       "others": [], "unbound": True}
                tex_rows.append(row)
        else:
            p = pvals.get(key)
            ov["original"] = p["value"] if p else None
            ov["applied"] = p is not None
            if p is None:
                ov["note"] = ("parameter not in the SDB route" if m["found"] else "material not in SDB")
        overrides.append(ov)
    for row in tex_rows:
        gids = _lookup(catalog, row["texture"], TEXTURE_TYPE) if row["texture"] else []
        row["gids"] = gids
        row["providers"] = _providers(catalog, gids)
        row["status"] = _status(gids, row["providers"]) if row["texture"] else "no texture"
        if row["original"]:
            og = _lookup(catalog, row["original"], TEXTURE_TYPE)
            row["original_gids"] = og
            row["original_status"] = _status(og, _providers(catalog, og))
    return {"sdb": {k: m[k] for k in m if k not in ("bindings",)}, "overrides": overrides, "textures": tex_rows}


def _summarize(res: dict) -> dict:
    meshes: dict[str, dict] = {}
    textures: dict[str, dict] = {}
    materials: dict[str, dict] = {}
    n_rtti = n_remap = n_tex_ov = 0
    for slot in res["slots"]:
        for m in slot["meshes"]:
            meshes.setdefault(m["name"].casefold(), {"name": m["name"], "gids": m["gids"],
                                                    "providers": m["providers"]})
            for s in m["submeshes"]:
                n_rtti += len(s["overrides"])
                n_tex_ov += sum(1 for o in s["overrides"] if o["type"] == 7)
                if s["material_source"] in ("model (remap)", "variant"):
                    n_remap += 1
                mk = (s["base_material"] or "").casefold()
                if mk:
                    mat = materials.setdefault(mk, {"name": s["base_material"], "in_sdb": s["sdb"]["found"],
                                                    "preset": s["sdb"].get("preset"), "users": 0})
                    mat["users"] += 1
                for t in s["textures"]:
                    for tn, tg in ((t["texture"], t["gids"]), (t.get("original"), t.get("original_gids"))):
                        if tn and tn.casefold() not in textures:
                            textures[tn.casefold()] = {"name": tn, "gids": tg or [],
                                                       "overridden_only": tn == t.get("original")}
                        elif tn and tn != t.get("original"):
                            textures[tn.casefold()]["overridden_only"] = False
    tex_list = sorted(textures.values(), key=lambda t: t["name"].casefold())
    mesh_list = sorted(meshes.values(), key=lambda t: t["name"].casefold())
    mat_list = sorted(materials.values(), key=lambda t: t["name"].casefold())
    return {
        "unique_meshes": mesh_list,
        "unique_textures": [{"name": t["name"], "gids": t["gids"], "found": bool(t["gids"]),
                             "only_as_overridden_original": t["overridden_only"]} for t in tex_list],
        "unique_materials": mat_list,
        "summary": {
            "meshes": len(mesh_list), "meshes_found": sum(1 for m in mesh_list if m["gids"]),
            "textures": len(tex_list), "textures_found": sum(1 for t in tex_list if t["gids"]),
            "materials": len(mat_list), "materials_in_sdb": sum(1 for m in mat_list if m["in_sdb"]),
            "overrides": n_rtti, "texture_overrides": n_tex_ov, "material_remaps": n_remap,
            "slots": len(res["slots"]),
            "submeshes": sum(len(m["submeshes"]) for s in res["slots"] for m in s["meshes"]),
        },
    }


def json_safe(obj: Any) -> Any:
    """Resolution → JSON-serialisable (numpy scalars, tuples, sets)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return obj.item()
    except Exception:  # noqa: BLE001
        return str(obj)
