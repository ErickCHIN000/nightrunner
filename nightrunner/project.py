"""Build projects — a mod described as edits, built into one new rpack + (optionally) one new dataN.pak.

A project (`<name>.nrproj`, JSON, schema nightrunner.project/1) lists:

* **items** — resources for the mod rpack. Each item names a *target* game resource (pack label relative to the
  assets folder + logical name + type) that serves as the template, a *source* file, and an optional *new_name*
  (None = same-name override of the target, else a new resource cloned from the target):
    - ``texture``  source .png / .dds (PNG imports as RGBA8 / R8 / RG8 normal; DDS keeps its own format)
    - ``mesh``     source model.cast / .glb / .gltf edited from the target's per-mesh export
    - ``scene``    source single-model .cast/.glb/.gltf + its export report (``options.report``); split into one
                   mesh per exported part at build time (``options.renames`` {logical name: new name})
    - ``raw``      source file replacing one part (``options.part`` ordinal) byte for byte
* **models** — `.model` overrides: per role (tpp / fpp / lodcc / ui) the original PAK member and the full edited
  document. They are written, unchanged apart from the user's edits, into the project PAK.

Nothing here touches the game folder: outputs go to ``output_dir``; installing is copying two files.
Default output names: the first free ``assets_N_pc.rpack`` (N from 2) in the game's assets folder and the first free
``dataN.pak`` (N 0..8, never a stock archive of the game profile) in ``<data>/source`` (``ph_ft`` for DLTB, ``ph``
for DL2). ``game`` records the profile id the project was made for (files without it are DLTB projects).
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from . import games as G
from .errors import BuildError
from .util.jsonio import dump_json, load_json
from .util.schema import matches as schema_matches

SCHEMA = "nightrunner.project/1"
SUFFIX = ".nrproj"
KINDS = ("texture", "mesh", "scene", "raw")
TYPE_OF_KIND = {"texture": 0x20, "mesh": 0x10, "scene": 0x10}
ROLES = ("tpp", "fpp", "lodcc", "ui")
ROLE_KEYS = {"ModelTpp": "tpp", "ModelFpp": "fpp", "ModelTppLodCC": "lodcc", "ModelUI": "ui"}
PATH_MODES = ("root", "original", "both")
OUTFIT_SCRIPT = G.DLTB.outfit_script            # scripts/player/player_outfit_slots.scr (both games)
OUTFIT_SCRIPT_LEGACY = "scripts/player_outfit_slots.scr"
APPEARANCES_SCRIPT = G.DLTB.appearances_script
PLAYER_MODELS = set(G.DLTB.player_models)
OUTFIT_TEMPLATE = """
// mapping of outfit part slots to model slots. Done per game.
// Emptied by nightrunner: equipped gear no longer replaces model slots (the look comes from the .model only).


sub main()
{

}
"""
PAK_RANGE = range(0, 9)
RPACK_FIRST = 2
TEXTURE_FORMATS = {"auto": None, "rgba8": 38, "r8": 0, "normal": 16}
SCENE_EXTS = (".cast", ".glb", ".gltf")
TEXTURE_EXTS = (".png", ".dds", ".jpg", ".jpeg", ".tga", ".bmp")


class ProjectError(BuildError):
    pass


class Cancelled(Exception):
    pass


# ---- data -------------------------------------------------------------------------------------------------------

@dataclass
class Item:
    kind: str
    source: str
    target_name: str
    target_pack: str = ""            # label relative to the assets folder; "" = first provider
    new_name: str | None = None
    enabled: bool = True
    options: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def type(self) -> int:
        if self.kind == "raw":
            return int(self.options.get("type", 0))
        return TYPE_OF_KIND[self.kind]

    @property
    def output_name(self) -> str:
        return self.new_name or self.target_name


@dataclass
class ModelRole:
    member: str                      # original PAK member path, e.g. models/player/player_tpp_skeleton.model
    source_pak: str = "data0.pak"
    doc: dict | None = None          # edited document (None = not loaded yet / unchanged original)
    original: dict | None = None     # snapshot of the original (for diffs); not required to build
    stash: dict = field(default_factory=dict)   # slot name -> mesh entries removed by "off" (restored by "on")


@dataclass
class ModelOverride:
    label: str
    roles: dict[str, ModelRole] = field(default_factory=dict)
    path_mode: str = "root"
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    no_gear: bool = False          # write an empty player_outfit_slots.scr (gear stops swapping model slots)

    def is_player(self) -> bool:
        return any(r.member.replace("\\", "/").rsplit("/", 1)[-1].lower() in PLAYER_MODELS for r in self.roles.values())

    def members(self) -> dict[str, ModelRole]:
        """Unique members (roles pointing at the same member collapse onto the first role)."""
        out: dict[str, ModelRole] = {}
        for role in ROLES:
            r = self.roles.get(role)
            if r is not None and r.member.lower() not in {m.lower() for m in out}:
                out[r.member] = r
        return out


@dataclass
class Project:
    name: str = "untitled"
    output_dir: str = ""
    rpack_name: str = ""             # "" = auto
    game: str = G.DEFAULT_ID         # game profile id (nightrunner.games)
    pak_name: str = ""               # "" = auto
    field08: int | None = None       # None = auto (0x1000 when any mesh, else the targets' common value)
    options: dict = field(default_factory=lambda: {"validate": True, "report": True, "keep_work": False})
    items: list[Item] = field(default_factory=list)
    models: list[ModelOverride] = field(default_factory=list)
    path: str = ""                   # where it was loaded from / saved to (not stored)

    # ---- persistence -------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        base = Path(self.path).parent if self.path else None

        def rel(p: str) -> str:
            if base and p:
                try:
                    return str(Path(p).resolve().relative_to(base.resolve())).replace("\\", "/")
                except ValueError:
                    pass
            return p

        items = []
        for it in self.items:
            d = asdict(it)
            d["source"] = rel(it.source)
            if it.kind == "scene" and it.options.get("report"):
                d["options"] = dict(it.options, report=rel(it.options["report"]))
            items.append(d)
        models = []
        for m in self.models:
            models.append({"id": m.id, "label": m.label, "path_mode": m.path_mode, "enabled": m.enabled,
                           "no_gear": m.no_gear,
                           "roles": {k: {"member": r.member, "source_pak": r.source_pak, "doc": r.doc, "stash": r.stash}
                                     for k, r in m.roles.items()}})
        return {"schema": SCHEMA, "name": self.name, "output_dir": self.output_dir, "rpack_name": self.rpack_name,
                "game": self.game, "pak_name": self.pak_name, "field08": None if self.field08 is None else f"0x{self.field08:X}",
                "options": self.options, "items": items, "models": models}

    @classmethod
    def from_json(cls, d: dict, path: str = "") -> "Project":
        if not schema_matches(d.get("schema"), SCHEMA):
            raise ProjectError(f"not a nightrunner project (schema {d.get('schema')!r})")
        base = Path(path).parent if path else None

        def absol(p: str) -> str:
            if base and p and not Path(p).is_absolute():
                return str((base / p).resolve())
            return p

        items = []
        for x in d.get("items", []):
            it = Item(kind=x["kind"], source=absol(x["source"]), target_name=x["target_name"],
                      target_pack=x.get("target_pack", ""), new_name=x.get("new_name"),
                      enabled=x.get("enabled", True), options=dict(x.get("options") or {}), id=x.get("id") or uuid.uuid4().hex[:8])
            if it.kind not in KINDS:
                raise ProjectError(f"item {it.id}: unknown kind {it.kind!r}")
            if it.kind == "scene" and it.options.get("report"):
                it.options["report"] = absol(it.options["report"])
            items.append(it)
        models = []
        for x in d.get("models", []):
            roles = {k: ModelRole(member=v["member"], source_pak=v.get("source_pak", "data0.pak"), doc=v.get("doc"),
                                  stash=dict(v.get("stash") or {}))
                     for k, v in (x.get("roles") or {}).items()}
            models.append(ModelOverride(label=x.get("label", "model"), roles=roles,
                                        path_mode=x.get("path_mode", "root"), enabled=x.get("enabled", True),
                                        no_gear=bool(x.get("no_gear", False)),
                                        id=x.get("id") or uuid.uuid4().hex[:8]))
        f08 = d.get("field08")
        return cls(name=d.get("name", "untitled"), output_dir=d.get("output_dir", ""),
                   rpack_name=d.get("rpack_name", ""), game=str(d.get("game") or G.DEFAULT_ID).lower(), pak_name=d.get("pak_name", ""),
                   field08=None if f08 in (None, "") else int(str(f08), 0),
                   options=dict(d.get("options") or {}), items=items, models=models, path=path)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path or self.path)
        if not str(p):
            raise ProjectError("project has no path")
        if p.suffix.lower() != SUFFIX:
            p = p.with_suffix(SUFFIX)
        self.path = str(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".partial")
        tmp.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        p = Path(path)
        return cls.from_json(json.loads(p.read_text(encoding="utf-8")), str(p))

    def item(self, item_id: str) -> Item:
        for it in self.items:
            if it.id == item_id:
                return it
        raise KeyError(item_id)


# ---- game environment -------------------------------------------------------------------------------------------

class GameEnv:
    """Qt-free view of an install: rpack lookup by (name, type) and PAK paths. `packs` (label -> path) may be
    given directly (tests); otherwise it is the install's rpacks, sorted like the explorer lists them."""

    def __init__(self, assets: Path | None = None, source: Path | None = None,
                 packs: dict[str, Path] | None = None, profile=None):
        self.profile = G.profile(profile)
        self.assets = Path(assets) if assets else None
        self.source = Path(source) if source else None
        if packs is None:
            packs = {}
            if self.assets and self.assets.is_dir():
                for p in sorted(self.assets.rglob("*.rpack"), key=lambda q: str(q.relative_to(self.assets)).lower()):
                    packs[str(p.relative_to(self.assets)).replace("\\", "/")] = p
        self.packs = dict(packs)
        self._tables: dict[str, dict] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_game(cls, game) -> "GameEnv":
        return cls(game.assets, game.source, profile=getattr(game, "profile", None))

    @property
    def data_name(self) -> str:
        """Data folder name for install hints (the source folder's parent, else the profile's first)."""
        if self.source is not None and self.source.parent.name:
            name = self.source.parent.name
            if name.lower() in {d.lower() for d in self.profile.data_dirs}:
                return name
        return self.profile.data_dirs[0]

    def _table(self, label: str) -> dict:
        with self._lock:
            t = self._tables.get(label)
            if t is None:
                from .container.rp6l import Pack
                from .util.names import engine_fold
                t = {}
                with Pack.open(self.packs[label]) as pk:
                    for i, lg in enumerate(pk.logicals):
                        t.setdefault((lg.type, engine_fold(pk.name_bytes(lg.name_index))), []).append(i)
                self._tables[label] = t
            return t

    def find(self, name: str, type_id: int, pack: str = "") -> list[tuple[str, int]]:
        """[(pack label, logical index)] providing (name, type); one pack when *pack* is given."""
        from .util.names import engine_fold
        key = (type_id, engine_fold(name.encode("utf-8", "surrogateescape")))
        labels = [pack] if pack else list(self.packs)
        out = []
        for lb in labels:
            if lb not in self.packs:
                continue
            for i in self._table(lb).get(key, []):
                out.append((lb, i))
        return out

    def resolve(self, item: Item) -> tuple[str, int]:
        hits = self.find(item.target_name, item.type, item.target_pack)
        if not hits:
            where = f" in {item.target_pack}" if item.target_pack else ""
            raise ProjectError(f"{item.target_name!r} (type 0x{item.type:02X}) not found{where}")
        return hits[0]

    def pak(self, name: str) -> Path | None:
        if self.source is None:
            return None
        p = self.source / name
        return p if p.is_file() else None


def next_free_pak(source_dir: Path | None, taken: Iterable[str] = (), profile=None) -> str:
    """First dataN.pak (N in PAK_RANGE) that neither exists in *source_dir* nor is a stock archive of *profile*
    (no profile: existing files only)."""
    names = {n.lower() for n in taken}
    if profile is not None:
        names |= {n.lower() for n in G.profile(profile).stock_paks}
    if source_dir and Path(source_dir).is_dir():
        names |= {p.name.lower() for p in Path(source_dir).iterdir()}
    for n in PAK_RANGE:
        if f"data{n}.pak" not in names:
            return f"data{n}.pak"
    raise ProjectError(f"no free dataN.pak name (N {PAK_RANGE.start}..{PAK_RANGE.stop - 1})")


def next_free_rpack(assets_dir: Path | None, taken: Iterable[str] = (), profile=None) -> str:
    names = {n.lower() for n in taken}
    if assets_dir and Path(assets_dir).is_dir():
        names |= {p.name.lower() for p in Path(assets_dir).iterdir()}
    n = G.profile(profile).rpack_first if profile is not None else RPACK_FIRST
    while f"assets_{n}_pc.rpack" in names:
        n += 1
    return f"assets_{n}_pc.rpack"


def output_names(project: Project, env: GameEnv) -> tuple[str, str]:
    prof = getattr(env, "profile", None)
    rp = project.rpack_name or next_free_rpack(env.assets, profile=prof)
    pk = project.pak_name or next_free_pak(env.source, profile=prof)
    return rp, pk


# ---- adding files -----------------------------------------------------------------------------------------------

def guess_kind(path: Path) -> str | None:
    s = path.suffix.lower()
    if s in TEXTURE_EXTS:
        return "texture"
    if s in SCENE_EXTS:
        # per-mesh edits sit in an extracted resource folder (mesh.json); anything else is a whole-model scene
        side = (path.parent / "mesh.json").is_file() or path.with_suffix(".mesh.json").is_file()
        if scene_report_for(path) or not side:
            return "scene"
        return "mesh"
    if s == ".bin":
        return "raw"
    return None


def scene_report_for(path: Path) -> Path | None:
    """The single-model export report next to a scene file (<stem>.cast.json)."""
    for cand in (path.with_suffix(".cast.json"), path.parent / f"{path.stem}.cast.json"):
        if cand.is_file():
            try:
                if load_json(cand).get("format", "").startswith("nightrunner.model_cast"):
                    return cand
            except Exception:  # noqa: BLE001
                pass
    return None


_OBJ_CACHE: dict[tuple[str, int, int], list[str]] = {}


def scene_objects(path: str | Path) -> list[str]:
    """Object (mesh node) names of a scene file, Blender .NNN suffixes kept (cached by size + mtime)."""
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _OBJ_CACHE:
        _OBJ_CACHE.clear() if len(_OBJ_CACHE) > 32 else None
        _OBJ_CACHE[key] = _scene_objects(p)
    return list(_OBJ_CACHE[key])


def _base_name(n: str) -> str:
    return re.sub(r"\.\d{3}$", "", n)


def effective_assign(item: "Item") -> dict[str, str]:
    """options.assign plus, with options.skip_rest, "" for every object that has no target of its own."""
    assign = dict(item.options.get("assign") or {})
    if item.options.get("skip_rest") and Path(item.source).is_file():
        names = {t["name"] for t in scene_targets(item.options["report"])} if item.options.get("report") else set()
        for o in scene_objects(item.source):
            if o not in assign and _base_name(o) not in assign and _base_name(o) not in names:
                assign[_base_name(o)] = ""
    return assign


def _scene_objects(path: Path) -> list[str]:
    from .cast import castlib
    from .cast.gltf import load_scene
    c = load_scene(Path(path))
    out = []
    for root in c.Roots():
        for mdl in root.ChildrenOfType(castlib.Model):
            out += [m.Name() or "" for m in mdl.Meshes()]
    return out


def scene_targets(report: str | Path | dict) -> list[dict]:
    """Export submeshes a scene object can replace: [{name, slot, mesh, material}]."""
    rep = report if isinstance(report, dict) else load_json(report)
    parts = rep.get("parts") or []
    out = []
    for m in rep.get("mesh_map") or []:
        pr = parts[m["part"]] if 0 <= m["part"] < len(parts) else {}
        out.append({"name": m["name"], "slot": pr.get("slot", ""), "mesh": pr.get("logical_name", ""),
                    "material": m.get("material", "")})
    return out


def guess_target(path: Path, kind: str) -> str:
    """Best-effort target name from a dropped file: textures keep the file name (names carry .png/.dds);
    per-mesh scenes use the folder name when the file is model.cast/.glb/.gltf, else the stem."""
    if kind == "texture":
        return path.name
    if kind in ("mesh",) and path.stem.lower() == "model":
        return path.parent.name.split("_", 1)[1] if re.match(r"^\d{6}_", path.parent.name) else path.parent.name
    if kind == "mesh":
        m = path.parent / "mesh.json"
        if m.is_file():
            try:
                return load_json(m).get("source", {}).get("name") or path.stem
            except Exception:  # noqa: BLE001
                pass
    return path.stem


def add_files(project: Project, paths: Iterable[Path], env: GameEnv | None = None) -> list[Item]:
    """Create items for dropped files (folders are walked one level for known files). Targets are matched by
    name when *env* is given; unmatched items keep target_pack "" and show up in validate()."""
    new: list[Item] = []
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(q for q in p.iterdir() if q.is_file() and guess_kind(q))
        elif p.is_file():
            files.append(p)
    for f in files:
        kind = guess_kind(f)
        if kind is None:
            continue
        opts: dict = {}
        if kind == "scene":
            rep = scene_report_for(f)
            target = f.stem
            if rep is not None:
                opts["report"] = str(rep)
                target = load_json(rep).get("model") or f.stem
        else:
            target = guess_target(f, kind)
        it = Item(kind=kind, source=str(f), target_name=target, options=opts)
        if env is not None and kind in ("texture", "mesh"):
            hits = env.find(target, it.type)
            if not hits and kind == "texture":
                # same stem with the other extension (white.png dropped for white.dds)
                for ext in (".dds", ".png"):
                    alt = Path(target).stem + ext
                    if alt != target and env.find(alt, it.type):
                        it.target_name = target = alt
                        hits = env.find(alt, it.type)
                        break
            if hits:
                it.target_pack = hits[0][0]
            elif kind == "texture":
                # a new texture name: keep it as the output name, target to be picked
                it.new_name = target
        project.items.append(it)
        new.append(it)
    return new


# ---- .model editing ---------------------------------------------------------------------------------------------

def _msh(name: str) -> str:
    return name if name.lower().endswith(".msh") else name + ".msh"


def slots(doc: dict) -> list[dict]:
    return doc.get("slots") or []


def slot(doc: dict, name: str) -> dict:
    for s in slots(doc):
        if s.get("name") == name:
            return s
    raise KeyError(name)


def slot_resources(s: dict) -> list[dict]:
    return (s.get("meshResources") or {}).get("resources") or []


def slot_mesh(s: dict) -> str | None:
    """Selected mesh name of a slot (None = slot off / empty)."""
    for r in slot_resources(s):
        if r.get("selected"):
            return r.get("name")
    return None


def _slot_off(s: dict, stash: dict | None) -> None:
    # An empty resource list is the only "off" the stock data proves (ARMS / HANDS ship empty); a slot whose
    # entries are all selected=false may fall back to its first entry ("selected else first"). The entries are
    # kept in *stash* (ModelRole.stash, saved in the project, never written to the PAK).
    res = s.setdefault("meshResources", {}).setdefault("resources", [])
    if res and stash is not None:
        stash[s["name"]] = copy.deepcopy(res)
    res.clear()


def _slot_restore(s: dict, stash: dict | None) -> list[dict]:
    res = s.setdefault("meshResources", {}).setdefault("resources", [])
    if not res and stash and s.get("name") in stash:
        res.extend(stash.pop(s["name"]))
    return res


def set_slot_enabled(doc: dict, slot_name: str, on: bool, stash: dict | None = None) -> None:
    s = slot(doc, slot_name)
    if not on:
        _slot_off(s, stash)
        return
    res = _slot_restore(s, stash)
    if res and not any(r.get("selected") for r in res):
        res[0]["selected"] = True


def set_slot_mesh(doc: dict, slot_name: str, mesh: str | None, stash: dict | None = None) -> dict | None:
    """Select *mesh* in the slot (None = off: the slot is emptied, entries go to *stash*). A mesh not listed yet
    is added as a copy of the currently selected (else first) entry — materialsData / materialsResources
    included — with only the name changed."""
    s = slot(doc, slot_name)
    if mesh is None:
        _slot_off(s, stash)
        return None
    res = _slot_restore(s, stash)
    want = _msh(mesh).lower()
    hit = next((r for r in res if _msh(r.get("name", "")).lower() == want), None)
    if hit is None:
        tmpl = next((r for r in res if r.get("selected")), res[0] if res else None)
        if tmpl is not None:
            hit = copy.deepcopy(tmpl)
        else:
            hit = {"layoutId": 4, "userData": [0, 0, 0, 0], "materialsData": [], "materialsResources": []}
        hit["name"] = _msh(mesh)
        res.append(hit)
    for r in res:
        r["selected"] = r is hit
    return hit


def slot_alternatives(s: dict, stash: dict | None = None) -> list[str]:
    names = [r.get("name", "") for r in slot_resources(s)]
    names += [r.get("name", "") for r in (stash or {}).get(s.get("name"), [])]
    return list(dict.fromkeys(n for n in names if n))


def free_slot_uid(doc: dict) -> int:
    """Lowest unused slotUid from 100 (stock body slots use 100..199; hands 500+, gloves 800+)."""
    used = {x.get("slotUid") for x in slots(doc)}
    n = 100
    while n in used:
        n += 1
    return n


def next_torso_slot(doc: dict) -> str:
    names = {x.get("name") for x in slots(doc)}
    n = 1
    while f"TORSO_PART_{n}" in names:
        n += 1
    return f"TORSO_PART_{n}"


def add_slot(doc: dict, name: str, filter_text: str = "torso") -> dict:
    """A new empty slot (same fields as the stock ones). Existing name -> that slot."""
    try:
        return slot(doc, name)
    except KeyError:
        pass
    s = {"slotUid": free_slot_uid(doc), "name": name, "filterText": filter_text, "tagsBits": 0,
         "shadowMaps": 15, "meshResources": {"resources": []}}
    doc.setdefault("slots", []).append(s)
    return s


def move_slot_mesh(doc: dict, src: str, dst: str, stash: dict | None = None) -> dict | None:
    """Move the selected mesh entry of *src* (materials and rttiValues included) into *dst* as its only
    resource; *src* is turned off, the old *dst* entries go to *stash*. *dst* is created (torso) if missing."""
    s = slot(doc, src)
    ent = next((r for r in slot_resources(s) if r.get("selected")), None)
    if ent is None:
        return None
    s["meshResources"]["resources"].remove(ent)
    _slot_off(s, stash)
    d = add_slot(doc, dst)
    _slot_off(d, stash)
    ent["selected"] = True
    d["meshResources"]["resources"].append(ent)
    return ent


HIDDEN_SLOT_FILTERS = ("head", "legs")


def mirror_role(src: "ModelRole", dst: "ModelRole") -> None:
    """Every slot of *dst* takes *src*'s meshes (and stash); slots only in *src* are added with a free uid."""
    dslots = dst.doc.setdefault("slots", [])
    by = {x.get("name"): x for x in dslots}
    for ss in slots(src.doc):
        name = ss.get("name")
        ds = by.get(name)
        if ds is None:
            ds = copy.deepcopy(ss)
            if ds.get("slotUid") in {x.get("slotUid") for x in dslots}:
                ds["slotUid"] = free_slot_uid(dst.doc)
            dslots.append(ds)
            by[name] = ds
        else:
            ds["meshResources"] = copy.deepcopy(ss.get("meshResources") or {"resources": []})
        if name in src.stash:
            dst.stash[name] = copy.deepcopy(src.stash[name])
        else:
            dst.stash.pop(name, None)


def mesh_entry(doc: dict, slot_name: str, mesh: str) -> dict:
    want = _msh(mesh).lower()
    for r in slot_resources(slot(doc, slot_name)):
        if _msh(r.get("name", "")).lower() == want:
            return r
    raise KeyError(f"{slot_name}/{mesh}")


def material_entry(mres: dict, material: str, *, base: str | None = None, create: bool = True) -> dict | None:
    """The selected materialsResources entry for submesh *material* of a mesh entry (created on demand with
    base = *base* or the submesh material itself)."""
    mdata = mres.setdefault("materialsData", [])
    row = next((m for m in mdata if m.get("name", "").lower() == material.lower()), None)
    if row is None:
        if not create:
            return None
        num = max([int(m.get("number", 0)) for m in mdata] + [-1]) + 1
        row = {"number": num, "name": material, "layoutId": 4, "loadFlags": "S"}
        mdata.append(row)
    groups = mres.setdefault("materialsResources", [])
    grp = next((g for g in groups if g.get("number") == row["number"]), None)
    if grp is None:
        if not create:
            return None
        grp = {"number": row["number"], "resources": []}
        groups.append(grp)
    ents = grp.setdefault("resources", [])
    ent = next((e for e in ents if e.get("selected")), ents[0] if ents else None)
    if ent is None:
        if not create:
            return None
        ent = {"name": base or material, "selected": True, "layoutId": 4, "loadFlags": "S", "rttiValues": []}
        ents.append(ent)
    elif base:
        ent["name"] = base
    return ent


def set_rtti(ent: dict, param: str, value: Any) -> None:
    """Set / remove (value None) one rttiValues override. str -> type 7, float -> 2, 3-seq -> 4."""
    vals = ent.setdefault("rttiValues", [])
    vals[:] = [v for v in vals if v.get("name") != param]
    if value is None:
        return
    if isinstance(value, str):
        vals.append({"name": param, "type": 7, "val_str": value})
    elif isinstance(value, (int, float)):
        vals.append({"name": param, "type": 2, "val_float": float(value)})
    else:
        v = [float(x) for x in value]
        if len(v) != 3:
            raise ProjectError(f"{param}: vec3 expected")
        vals.append({"name": param, "type": 4, "val_vec3": v})


def rtti_value(v: dict) -> Any:
    return v.get("val_str", v.get("val_float", v.get("val_vec3")))


def diff_docs(a: Any, b: Any, path: str = "") -> list[str]:
    """Readable JSON diff; list elements with a "name" are keyed by it (slots, meshes, rtti values)."""
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            p = f"{path}.{k}" if path else k
            if k not in b:
                out.append(f"- {p}")
            elif k not in a:
                out.append(f"+ {p} = {json.dumps(b[k])[:120]}")
            else:
                out += diff_docs(a[k], b[k], p)
    elif isinstance(a, list) and isinstance(b, list):
        key = "name"
        keyed = all(isinstance(x, dict) and "name" in x for x in a + b)
        if not keyed and a + b and all(isinstance(x, dict) and "number" in x for x in a + b):
            keyed, key = True, "number"
        if keyed and len({x[key] for x in a}) == len(a) and len({x[key] for x in b}) == len(b):
            am = {x[key]: x for x in a}
            bm = {x[key]: x for x in b}
            for k in list(am) + [k for k in bm if k not in am]:
                p = f"{path}[{k}]"
                if k not in bm:
                    out.append(f"- {p}")
                elif k not in am:
                    out.append(f"+ {p}")
                else:
                    out += diff_docs(am[k], bm[k], p)
        elif len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff_docs(x, y, f"{path}[{i}]")
        else:
            out.append(f"~ {path}: {len(a)} → {len(b)} items")
    elif a != b:
        out.append(f"~ {path}: {json.dumps(a)[:60]} → {json.dumps(b)[:60]}")
    return out


def parse_appearances(text: str) -> list[dict]:
    """playerappearances.scr -> [{character, appearance, tpp, fpp, lodcc, ui}] (model file names)."""
    out: list[dict] = []
    char = None
    cur: dict | None = None
    for line in text.splitlines():
        line = re.sub(r"//.*", "", line)
        m = re.search(r'Character\(\s*"([^"]*)"', line)
        if m:
            char = m.group(1)
            continue
        m = re.search(r'Appearance\(\s*"([^"]*)"', line)
        if m:
            cur = {"character": char, "appearance": m.group(1)}
            out.append(cur)
            continue
        m = re.search(r'\b(ModelTppLodCC|ModelTpp|ModelFpp|ModelUI)\(\s*"([^"]*)"', line)
        if m and cur is not None:
            cur[ROLE_KEYS[m.group(1)]] = m.group(2)
    return out


def load_override(label: str, members: dict[str, str], pak: Path, source_pak: str = "") -> ModelOverride:
    """New override for *members* {role: member name or path} read from *pak* (names resolved with PakIndex.find)."""
    from .pak.model_json import PakIndex
    mo = ModelOverride(label=label)
    with PakIndex.open(pak) as ix:
        for role, name in members.items():
            if not name:
                continue
            mem = ix.find(name)
            doc = ix.load_model(mem)
            mo.roles[role] = ModelRole(member=mem.name, source_pak=source_pak or Path(pak).name,
                                       doc=copy.deepcopy(doc), original=doc)
    mo.no_gear = mo.is_player()
    return mo


def empty_outfit_script(text: str | None = None) -> str:
    """player_outfit_slots.scr with an empty main() (comments kept) — what gear-proof player mods ship."""
    if not text:
        return OUTFIT_TEMPLATE
    m = re.search(r"sub\s+main\s*\(\s*\)\s*\{", text)
    if m is None:
        return OUTFIT_TEMPLATE
    depth, i = 1, m.end()
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[:m.end()] + "\n\n" + text[i - 1:]


def ensure_originals(mo: ModelOverride, env: GameEnv) -> None:
    from .pak.model_json import PakIndex
    for r in mo.roles.values():
        if r.original is None:
            p = env.pak(r.source_pak)
            if p is None:
                continue
            with PakIndex.open(p) as ix:
                r.original = ix.load_model(r.member)
            if r.doc is None:
                r.doc = copy.deepcopy(r.original)


def member_paths(member: str, mode: str) -> list[str]:
    bare = member.replace("\\", "/").rsplit("/", 1)[-1]
    if mode == "root":
        return [bare]
    if mode == "original":
        return [member]
    return [bare] if bare == member else [bare, member]


# ---- validation -------------------------------------------------------------------------------------------------

@dataclass
class Problem:
    level: str        # "error" | "warning"
    where: str        # item id / override id / "project"
    message: str


def validate(project: Project, env: GameEnv, *, check_models: bool = True) -> list[Problem]:
    probs: list[Problem] = []
    out_names: dict[tuple[int, str], str] = {}
    for it in project.items:
        if not it.enabled:
            continue
        src = Path(it.source)
        if not src.is_file():
            probs.append(Problem("error", it.id, f"source missing: {src}"))
        if it.kind == "texture" and src.suffix.lower() not in TEXTURE_EXTS:
            probs.append(Problem("error", it.id, f"unsupported texture file ({src.name})"))
        if it.kind in ("mesh", "scene") and src.suffix.lower() not in SCENE_EXTS:
            probs.append(Problem("error", it.id, f"mesh source must be .cast/.glb/.gltf ({src.name})"))
        if it.kind == "scene":
            rep = it.options.get("report")
            if not rep or not Path(rep).is_file():
                probs.append(Problem("error", it.id, "scene has no export report (<model>.cast.json)"))
            else:
                known = {t["name"] for t in scene_targets(rep)}
                assign = it.options.get("assign") or {}
                hide = set(it.options.get("hide") or [])
                bad = sorted(({v for v in assign.values() if v} | hide) - known)
                if bad:
                    probs.append(Problem("error", it.id, f"unknown target {bad[0]}"))
                both = sorted(hide & set(assign.values()))
                if both:
                    probs.append(Problem("error", it.id, f"{both[0]} is assigned and hidden"))
                if src.is_file():
                    try:
                        eff = effective_assign(it)
                        loose = [o for o in scene_objects(src)
                                 if o not in eff and _base_name(o) not in eff and _base_name(o) not in known]
                    except Exception as exc:  # noqa: BLE001
                        loose = []
                        probs.append(Problem("error", it.id, f"cannot read scene: {exc}"))
                    if loose:
                        probs.append(Problem("error", it.id, f"{len(loose)} objects have no target ({loose[0]}…)"))
        if it.kind == "raw" and "part" not in it.options:
            probs.append(Problem("error", it.id, "raw item needs options.part"))
        if it.kind != "scene":
            try:
                env.resolve(it)
            except ProjectError as exc:
                probs.append(Problem("error", it.id, f"no target: {exc}"))
            key = (it.type, it.output_name.lower())
            if key in out_names:
                probs.append(Problem("error", it.id, f"duplicate output {it.output_name!r} (also {out_names[key]})"))
            out_names[key] = it.id
            if it.new_name and env.find(it.new_name, it.type):
                probs.append(Problem("warning", it.id, f"{it.new_name!r} also exists in the game (overrides it)"))
    if project.pak_name and not re.match(r"^[A-Za-z0-9_\-]+\.pak$", project.pak_name):
        probs.append(Problem("error", "project", f"bad PAK name {project.pak_name!r}"))
    if project.rpack_name and not project.rpack_name.lower().endswith(".rpack"):
        probs.append(Problem("error", "project", f"rpack name must end in .rpack ({project.rpack_name!r})"))
    prof = getattr(env, "profile", G.profile(None))
    if (project.game or G.DEFAULT_ID) != prof.id:
        other = G.PROFILES.get(project.game)
        probs.append(Problem("warning", "project", f"made for {other.name if other else project.game}, "
                                                   f"current game is {prof.name}"))
    if project.pak_name and project.pak_name.lower() in {n.lower() for n in prof.stock_paks}:
        probs.append(Problem("error", "project", f"{project.pak_name} is a game archive name"))
    if not any(it.enabled for it in project.items) and not any(m.enabled for m in project.models):
        probs.append(Problem("error", "project", "nothing to build"))

    if check_models:
        provided = {(it.type, it.output_name.lower()) for it in project.items if it.enabled and it.kind != "scene"}
        for sc in (it for it in project.items if it.enabled and it.kind == "scene"):
            for new in (sc.options.get("renames") or {}).values():
                provided.add((0x10, new.lower()))
        members_seen: dict[str, str] = {}
        for mo in project.models:
            if not mo.enabled:
                continue
            if mo.path_mode not in PATH_MODES:
                probs.append(Problem("error", mo.id, f"path mode {mo.path_mode!r}"))
            for member, r in mo.members().items():
                for mp in member_paths(member, mo.path_mode):
                    if mp.lower() in members_seen:
                        probs.append(Problem("error", mo.id, f"{mp} also written by {members_seen[mp.lower()]}"))
                    members_seen[mp.lower()] = mo.label
                if r.doc is None:
                    probs.append(Problem("error", mo.id, f"{member}: document not loaded"))
                    continue
                if r.original is None:
                    try:
                        ensure_originals(mo, env)
                    except Exception:  # noqa: BLE001 - no PAK: check every slot
                        pass
                orig_slots = {x.get("name"): x for x in slots(r.original or {})}
                for s in slots(r.doc):
                    if orig_slots.get(s.get("name")) == s:
                        continue                  # untouched game data is the game's business
                    m = slot_mesh(s)
                    if m and mo.is_player() and (s.get("filterText") or "").lower() in HIDDEN_SLOT_FILTERS:
                        stem_ = m[:-4] if m.lower().endswith(".msh") else m
                        if (0x10, stem_.lower()) in provided:
                            probs.append(Problem("warning", mo.id, f"{member} {s.get('name')}: new meshes in "
                                                                   f"head/legs slots are hidden in game — use a TORSO slot"))
                    if not m:
                        continue
                    stem = m[:-4] if m.lower().endswith(".msh") else m
                    if (0x10, stem.lower()) not in provided and not env.find(stem, 0x10):
                        probs.append(Problem("error", mo.id, f"{member} {s.get('name')}: mesh {m} not found"))
                    for grp in (_selected_mesh_entry(s) or {}).get("materialsResources", []):
                        for e in grp.get("resources", []):
                            if not e.get("selected", True):
                                continue
                            for v in e.get("rttiValues", []):
                                if v.get("type") == 7 and v.get("name", "").endswith("_tex"):
                                    tex = v.get("val_str", "")
                                    if tex and (0x20, tex.lower()) not in provided and not env.find(tex, 0x20):
                                        probs.append(Problem("error", mo.id,
                                                             f"{member} {s.get('name')}: texture {tex} not found"))
            if mo.no_gear and not prof.outfit_script:
                probs.append(Problem("warning", mo.id, f"No gear is not supported for {prof.name} (ignored)"))
            elif mo.no_gear:
                for mp in member_paths(prof.outfit_script, mo.path_mode):
                    owner = members_seen.get(mp.lower())
                    if owner is not None and owner != "gear":
                        probs.append(Problem("error", mo.id, f"{mp} also written by {owner}"))
                    members_seen[mp.lower()] = "gear"
            elif mo.is_player():
                probs.append(Problem("warning", mo.id, "gear still replaces slots (No gear off)"))
            if "tpp" in mo.roles and "fpp" not in mo.roles:
                probs.append(Problem("warning", mo.id, "no FPP model"))
    return probs


def _selected_mesh_entry(s: dict) -> dict | None:
    return next((r for r in slot_resources(s) if r.get("selected")), None)


# ---- build ------------------------------------------------------------------------------------------------------

def _texture_stage(it: Item, res_dir: Path, entry: dict) -> None:
    ed = entry.get("editable") or {}
    sc_rel = ed.get("sidecar")
    if not sc_rel:
        raise ProjectError(f"{it.target_name}: extracted texture has no sidecar")
    sc_path = res_dir / sc_rel
    sc = load_json(sc_path)
    src = Path(it.source)
    if src.suffix.lower() in (".png", ".dds"):
        dst_name = "source" + src.suffix.lower()
        shutil.copyfile(src, res_dir / dst_name)
    else:
        # other raster formats go through the PNG importer
        from PIL import Image
        dst_name = "source.png"
        with Image.open(src) as im:
            im.convert("RGBA").save(res_dir / dst_name)
    sc["source"] = dst_name
    imp = dict(sc.get("import") or {})
    fmt = it.options.get("format", "auto")
    if fmt not in TEXTURE_FORMATS:
        raise ProjectError(f"{it.target_name}: texture format {fmt!r}")
    if TEXTURE_FORMATS[fmt] is not None:
        imp["png_format"] = TEXTURE_FORMATS[fmt]
    if it.options.get("mips") == "none":
        imp["mip_count"] = 1
    if it.options.get("srgb_to_linear"):
        imp["srgb_to_linear"] = True
    sc["import"] = imp
    if src.suffix.lower() != ".dds":
        # PNG import changes the format, so the original per-mip byte sums are stale (imgc.md §6)
        sc.setdefault("imgc", {})
        if sc["imgc"].get("mip_split"):
            sc["imgc"]["mip_split"] = 0
    dump_json(sc, sc_path)


def _mesh_stage(it: Item, res_dir: Path) -> None:
    from .mesh.codec import GLTF_FILES
    src = Path(it.source)
    for n in GLTF_FILES:
        (res_dir / n).unlink(missing_ok=True)
    ext = src.suffix.lower()
    if ext == ".cast":
        shutil.copyfile(src, res_dir / "model.cast")
    elif ext == ".glb":
        shutil.copyfile(src, res_dir / "model.glb")
    else:
        # .gltf: copy the document as model.gltf and every relative buffer/image beside it
        doc = json.loads(src.read_text(encoding="utf-8"))
        for coll in ("buffers", "images"):
            for b in doc.get(coll, []):
                uri = b.get("uri", "")
                if uri and not uri.startswith("data:") and "://" not in uri:
                    s = (src.parent / uri).resolve()
                    d = (res_dir / uri).resolve()
                    if res_dir.resolve() not in d.parents:
                        raise ProjectError(f"{src.name}: uri {uri!r} leaves the folder")
                    d.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(s, d)
        shutil.copyfile(src, res_dir / "model.gltf")


def _raw_stage(it: Item, tree_dir: Path, entry: dict) -> None:
    k = int(it.options["part"])
    parts = entry["parts"]
    if not 0 <= k < len(parts) or not parts[k].get("raw"):
        raise ProjectError(f"{it.target_name}: part {k} has no raw file")
    shutil.copyfile(it.source, tree_dir / parts[k]["raw"])


def game_doc(doc: dict) -> dict:
    """The .model document as the game must see it: every slot holds at most ONE mesh entry.

    (RT, 2026-09-16) The game does not honour `selected` — with several entries in a slot it draws the stock one
    listed first and ignores ours (TORSO/TORSO_PART_1/_3 of the N2B build: new meshes invisible, the stock
    forearms shown instead). No stock .model in data0/data1 has more than one entry per slot. The project keeps
    alternatives for editing; the written PAK gets only the selected entry (else the first)."""
    out = copy.deepcopy(doc)
    for s in out.get("slots") or []:
        mr = s.get("meshResources")
        res = (mr or {}).get("resources") or []
        if len(res) > 1:
            chosen = next((r for r in res if r.get("selected")), res[0])
            chosen["selected"] = True
            mr["resources"] = [chosen]
    return out


def build_project(project: Project, env: GameEnv, *, progress: Callable[[str], Any] | None = None,
                  cancel: threading.Event | None = None, work_dir: Path | None = None,
                  out_dir: Path | None = None) -> dict:
    """Build the project. Returns the report (also written as <out>/<name>.build.json)."""
    from .build import build
    from .container.rp6l import Pack
    from .extract import extract
    from .pak.model_json import write_models_pak
    from .select import Pick, Tree, write_selection

    t0 = time.time()
    say = progress or (lambda m: None)

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    probs = validate(project, env)
    errors = [p for p in probs if p.level == "error"]
    if errors:
        raise ProjectError("; ".join(f"[{p.where}] {p.message}" for p in errors[:8]))
    out = Path(out_dir or project.output_dir or (Path(project.path).parent / "out" if project.path else "out"))
    out.mkdir(parents=True, exist_ok=True)
    rp_name, pak_name = output_names(project, env)
    work = Path(work_dir or out / f".{project.name}.work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    report: dict[str, Any] = {"schema": "nightrunner.project_build/1", "project": project.name,
                              "warnings": [p.message for p in probs], "items": [], "outputs": {}}
    try:
        # ---- expand scenes into per-mesh items ---------------------------------------------------------------
        items: list[tuple[Item, Path | None]] = []          # (item, pre-staged resource dir or None)
        for it in project.items:
            if not it.enabled:
                continue
            if it.kind != "scene":
                items.append((it, None))
                continue
            check()
            say(f"split {Path(it.source).name}")
            from .cast.split import split_model_cast
            rep = load_json(it.options["report"])
            pack_paths = {pr["pack_path"]: str(env.packs[_label_for(env, pr["pack_path"])])
                          for pr in rep["parts"] if _label_for(env, pr["pack_path"])}
            sres = split_model_cast(it.source, rep, work / "split" / it.id, pack_paths=pack_paths,
                                    tol=float(it.options.get("tol", 1e-4)),
                                    assign=effective_assign(it) or None, hide=it.options.get("hide") or None,
                                    lods=bool(it.options.get("lods", True)),
                                    double_sided=it.options.get("double_sided") or None)
            probs_s = [p for m in sres["meshes"] for p in m["problems"]] + sres["problems"]
            if probs_s:
                raise ProjectError(f"{Path(it.source).name}: " + "; ".join(probs_s[:5]))
            renames = it.options.get("renames") or {}
            only = set(it.options.get("meshes") or [])
            for pr, m in zip(rep["parts"], sres["meshes"]):
                if only and m["logical_name"] not in only:
                    continue
                chk = m.get("check") or {}
                changed = any(s.get("status") == "applied" for s in m["submeshes"]) and not chk.get("unchanged")
                if not changed and m["logical_name"] not in renames and m["logical_name"] not in only:
                    continue
                label = _label_for(env, pr["pack_path"])
                sub = Item(kind="mesh", source=str(Path(m["dir"]) / "model.cast"), target_name=m["logical_name"],
                           target_pack=label or "", new_name=renames.get(m["logical_name"]),
                           options={"from_scene": it.id}, id=f"{it.id}.{len(items)}")
                items.append((sub, None))
            report["items"].append({"id": it.id, "kind": "scene", "split": [
                {"mesh": m["logical_name"], "applied": sum(s.get("status") == "applied" for s in m["submeshes"])}
                for m in sres["meshes"]]})

        # ---- extract every target on its own (one tree per item) ---------------------------------------------
        picks = []
        f08s = set()
        has_mesh = False
        for n, (it, _) in enumerate(items):
            check()
            label, index = env.resolve(it)
            say(f"[{n + 1}/{len(items)}] {it.output_name}")
            tdir = work / "items" / f"{n:04d}_{it.id}.rpx"
            with Pack.open(env.packs[label]) as pk:
                f08s.add(pk.header.field08)
                extract(pk, tdir, indices={index}, progress=False)
            tree = Tree(tdir)
            entry = tree.by_index(index)
            res_dir = tree.dir / entry["dir"]
            if it.kind == "texture":
                _texture_stage(it, res_dir, entry)
            elif it.kind == "mesh":
                _mesh_stage(it, res_dir)
                has_mesh = True
            elif it.kind == "raw":
                _raw_stage(it, tree.dir, entry)
            pk_ = Pick(tree, entry)
            if it.new_name:
                pk_.name_hex = it.new_name.encode("utf-8", "surrogateescape").hex()
                if pk_.type == 0x10 and pk_.name_hex != entry["name_hex"]:
                    pk_.needs_identity_fix = True
            picks.append(pk_)
            report["items"].append({"id": it.id, "kind": it.kind, "target": f"{label}:{index}",
                                    "name": it.output_name, "source": it.source})

        if picks:
            check()
            if project.field08 is not None:
                f08 = project.field08
            elif has_mesh:
                f08 = 0x1000
            else:
                f08 = f08s.pop() if len(f08s) == 1 else 0x1000
            spec_dir = work / "mod.rpx"
            spec = write_selection(picks, {"field08": f08, "flags": 1, "trees": list({id(p.tree): p.tree for p in picks}.values())},
                                   spec_dir)
            report["warnings"] += spec.get("warnings", [])
            say(f"build {rp_name}")
            res = build(spec_dir, out / rp_name, options={"ignore_bone_changes": True},
                        validate_output=bool(project.options.get("validate", True)))
            report["warnings"] += res.get("warnings", [])
            report["outputs"]["rpack"] = {"path": res["output"], "size": res["size"], "layout": res["layout"],
                                          "resources": res["logicals"], "field08": f"0x{f08:X}",
                                          "regenerated": res["regenerated"],
                                          "validation_ok": (res.get("validation") or {}).get("ok")}

        # ---- PAK -------------------------------------------------------------------------------------------------
        docs: dict[str, dict] = {}
        for mo in project.models:
            if not mo.enabled:
                continue
            for member, r in mo.members().items():
                for mp in member_paths(member, mo.path_mode):
                    docs[mp] = game_doc(r.doc)
        texts: dict[str, str] = {}
        prof = getattr(env, "profile", G.profile(None))
        outfit = prof.outfit_script
        gear = [mo for mo in project.models if mo.enabled and mo.no_gear] if outfit else []
        if gear:
            src_text = None
            pak0 = env.pak(prof.base_pak)
            if pak0 is not None:
                from .pak.model_json import PakIndex
                for member in dict.fromkeys((outfit, OUTFIT_SCRIPT, OUTFIT_SCRIPT_LEGACY)):
                    try:
                        with PakIndex.open(pak0) as ix:
                            src_text = ix.read(member).decode("utf-8", "replace")
                        break
                    except Exception:  # noqa: BLE001 - member missing: try the next spelling
                        src_text = None
            for mp in {p for mo in gear for p in member_paths(outfit, mo.path_mode)}:
                texts[mp] = empty_outfit_script(src_text)
        if docs or texts:
            check()
            say(f"write {pak_name}")
            prep = write_models_pak(out / pak_name, docs, texts=texts, overwrite=True)
            report["outputs"]["pak"] = {"path": prep["path"], "members": prep["members"], "size": prep["bytes"]}
    finally:
        if not project.options.get("keep_work"):
            shutil.rmtree(work, ignore_errors=True)
    report["install"] = {
        "rpack": f"{getattr(env, 'data_name', 'ph_ft')}/work/data_platform/pc/assets/{rp_name}" if "rpack" in report["outputs"] else None,
        "pak": f"{getattr(env, 'data_name', 'ph_ft')}/source/{pak_name}" if "pak" in report["outputs"] else None,
    }
    report["game"] = env.profile.id if hasattr(env, "profile") else G.DEFAULT_ID
    report["seconds"] = round(time.time() - t0, 2)
    if project.options.get("report", True):
        dump_json(report, out / f"{project.name}.build.json")
    return report


def _label_for(env: GameEnv, pack_path: str) -> str | None:
    """Map a recorded pack path (export report) to an env label (by label suffix, then file name)."""
    p = pack_path.replace("\\", "/")
    for lb, path in env.packs.items():
        if p.endswith("/" + lb) or p == lb or str(path).replace("\\", "/") == p:
            return lb
    name = p.rsplit("/", 1)[-1].lower()
    cands = [lb for lb in env.packs if lb.rsplit("/", 1)[-1].lower() == name]
    return cands[0] if len(cands) == 1 else None
