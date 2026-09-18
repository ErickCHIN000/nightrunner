""".model (JSON v6) reader over PAK archives + models.pak override writer.

Provenance (survey 03 §9, 04 §5; notes/FORMATS/pak-and-model.md):
  (A) `dataN.pak` / `.mpak` are plain ZIP archives (every data0.pak member deflate, method 8; census 2026-09-15:
      18,012 members, 809 `.model`, 0 `.models`, all version 6 JSON objects).
  (A) `.model` schema as observed on all 809 members (census 2026-09-15):
        version 6 | preset {skeletonName} | data {meshAttribute[4], properties[{name,value}]} | slots[] (808/809)
        | poseItems (162/809, not interpreted)
        slot: slotUid, name, filterText, tagsBits, shadowMaps, meshResources{resources[]}, clothResources (286 slots)
        mesh resource: name (.msh), selected, layoutId, userData[4], materialsData[{number,name,layoutId,loadFlags}],
                       materialsResources[{number, resources[{name (.mat), selected, layoutId, loadFlags, rttiValues[]}]}]
        rttiValues: {name, type, val_str|val_float|val_vec3} — types seen in stock: 7 (340), 4 (97), 2 (16)
  (A) engine +0xBE0A60 parses selected/layoutId/loadFlags/rttiValues; +0xC0C7E0 MatLoads the base material BEFORE
      MatClone applies the overrides (so base-material textures must resolve).
  (B) join rule (DyingLightExplorer geometry.apply_model_materials, confirmed necessary for FPP sleeves):
      submesh material name → materialsData[].number → materialsResources[number].resources → selected (else first)
      → base .mat name + rttiValues overrides.
  (RT, user-confirmed, NOT native-verified) a `data3.pak` whose members sit at the ZIP ROOT with bare names
      (`player_tpp_skeleton.model`, `playerappearances.scr`) overrides `models/player/player_tpp_skeleton.model` and
      `scripts/playerappearances.scr` from data0.pak. Whether the rule is basename lookup, a root search path, or
      numeric dataN precedence is (C). `write_models_pak` therefore writes members at exactly the paths it is given
      and documents both conventions rather than choosing one.
  (C) semantics of layoutId, loadFlags "I"/"S", shadowMaps, tagsBits, slotUid, userData, meshAttribute, poseItems.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..errors import BuildError, FormatError

MODEL_SUFFIXES = (".model", ".models")
MAX_MEMBER_BYTES = 16 << 20
_SAFE_MEMBER = re.compile(r"^[A-Za-z0-9_\-./ ]+$")


@dataclass(frozen=True)
class Member:
    index: int
    name: str
    size: int
    compressed: int
    compress_type: int
    crc: int

    def to_json(self) -> dict:
        return {"index": self.index, "name": self.name, "size": self.size, "compressed": self.compressed,
                "compress_type": self.compress_type, "crc32": f"0x{self.crc:08X}"}


class PakIndex:
    """ZIP central-directory view of a PAK; members are inflated only on request."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        try:
            self._zip = zipfile.ZipFile(self.path)
        except zipfile.BadZipFile as exc:
            raise FormatError(f"{self.path}: not a ZIP/PAK archive ({exc})") from None
        self.members = [Member(i, zi.filename, zi.file_size, zi.compress_size, zi.compress_type, zi.CRC)
                        for i, zi in enumerate(self._zip.infolist())]
        self._by_name: dict[str, list[Member]] = {}
        for m in self.members:
            self._by_name.setdefault(m.name.replace("\\", "/").casefold(), []).append(m)

    @classmethod
    def open(cls, path: Path | str) -> "PakIndex":
        return cls(path)

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "PakIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.members)

    # ---- listing -----------------------------------------------------------------------------------------------

    def list_members(self, query: str | None = None, suffixes: tuple[str, ...] | None = None) -> list[Member]:
        q = query.casefold() if query else None
        out = []
        for m in self.members:
            n = m.name.casefold()
            if suffixes and not n.endswith(suffixes):
                continue
            if q and q not in n:
                continue
            out.append(m)
        return out

    def list_models(self, query: str | None = None) -> list[Member]:
        return self.list_members(query, MODEL_SUFFIXES)

    def find(self, member: str | int) -> Member:
        """Member by index, exact path, case-insensitive path, or unique basename."""
        if isinstance(member, int) or (isinstance(member, str) and member.isdigit()):
            i = int(member)
            if not 0 <= i < len(self.members):
                raise FormatError(f"member index {i} out of range")
            return self.members[i]
        key = member.replace("\\", "/").casefold()
        hits = self._by_name.get(key)
        if not hits:
            base = key.rsplit("/", 1)[-1]
            hits = [m for m in self.members if m.name.replace("\\", "/").casefold().rsplit("/", 1)[-1] == base]
        if not hits:
            raise FormatError(f"{member!r}: no such member in {self.path.name}")
        if len(hits) > 1:
            raise FormatError(f"{member!r}: {len(hits)} members match in {self.path.name}: {[m.name for m in hits]}")
        return hits[0]

    # ---- reading -----------------------------------------------------------------------------------------------

    def read(self, member: str | int | Member, max_bytes: int = MAX_MEMBER_BYTES) -> bytes:
        m = member if isinstance(member, Member) else self.find(member)
        if m.size > max_bytes:
            raise FormatError(f"{m.name}: {m.size} bytes exceeds the {max_bytes} byte limit")
        with self._zip.open(self._zip.infolist()[m.index]) as fh:
            data = fh.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise FormatError(f"{m.name}: inflated size exceeds the {max_bytes} byte limit")
        return data

    def load_json(self, member: str | int | Member) -> dict:
        m = member if isinstance(member, Member) else self.find(member)
        try:
            doc = json.loads(self.read(m).decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FormatError(f"{m.name}: not JSON ({exc})") from None
        if not isinstance(doc, dict):
            raise FormatError(f"{m.name}: JSON root is not an object")
        return doc

    def load_model(self, member: str | int | Member) -> dict:
        m = member if isinstance(member, Member) else self.find(member)
        if not m.name.casefold().endswith(MODEL_SUFFIXES):
            raise FormatError(f"{m.name}: not a .model member")
        doc = self.load_json(m)
        if doc.get("version") != 6:
            raise FormatError(f"{m.name}: version {doc.get('version')!r} (only 6 is known)")
        return doc


def load_model(pak: Path | str, member: str | int) -> dict:
    with PakIndex.open(pak) as idx:
        return idx.load_model(member)


# ---- model helpers ---------------------------------------------------------------------------------------------

def _selected(resources: list[dict]) -> dict | None:
    sel = [r for r in resources if isinstance(r, dict) and r.get("selected")]
    if len(sel) > 1:
        raise FormatError("multiple 'selected' entries")
    if sel:
        return sel[0]
    return resources[0] if resources else None


def mesh_refs(model: dict) -> list[dict]:
    """Every mesh resource referenced by the model: slot identity, .msh name, selected flag, material table."""
    out = []
    slots = model.get("slots") or []
    if not isinstance(slots, list):
        raise FormatError("model 'slots' is not a list")
    for s in slots:
        if not isinstance(s, dict):
            raise FormatError("slot is not an object")
        res = (s.get("meshResources") or {}).get("resources") or []
        chosen = _selected(res)
        for r in res:
            mats = []
            for md in r.get("materialsData") or []:
                mats.append({"number": md.get("number"), "embedded_name": md.get("name"), "layoutId": md.get("layoutId"),
                             "loadFlags": md.get("loadFlags"), "resolved": _resolve_number(r, md.get("number"))})
            out.append({"slot": s.get("name"), "slotUid": s.get("slotUid"), "mesh": r.get("name"),
                        "selected": bool(r.get("selected")), "chosen": r is chosen, "layoutId": r.get("layoutId"),
                        "userData": r.get("userData"), "materials": mats})
    return out


def _resolve_number(mesh_res: dict, number) -> dict | None:
    groups = [g for g in mesh_res.get("materialsResources") or [] if g.get("number") == number]
    if not groups:
        return None
    if len(groups) > 1:
        raise FormatError(f"materialsResources number {number} is duplicated")
    choice = _selected(groups[0].get("resources") or [])
    if choice is None:
        return None
    values = choice.get("rttiValues") or []
    overrides = {}
    for v in values:
        if not isinstance(v, dict) or "name" not in v:
            raise FormatError("rttiValues entry without a name")
        t = v.get("type")
        val = v.get("val_str") if t == 7 else v.get("val_float") if t == 2 else v.get("val_vec3") if t == 4 else \
            next((v[k] for k in v if k.startswith("val_")), None)
        overrides[v["name"]] = {"type": t, "value": val}
    return {"base": choice.get("name"), "layoutId": choice.get("layoutId"), "loadFlags": choice.get("loadFlags"),
            "rttiValues": values, "overrides": overrides, "alternatives": len(groups[0].get("resources") or [])}


def material_for_submesh(model: dict, mesh_name: str, material_name: str) -> dict | None:
    """The join rule: (mesh .msh name, submesh material name embedded in the mesh) → materialsData.number →
    materialsResources[number] → selected/first → {base, overrides}. Returns None when the model does not
    override that submesh (the mesh's embedded material applies unchanged)."""
    key = mesh_name.casefold()
    if not key.endswith(".msh"):
        key += ".msh"
    for s in model.get("slots") or []:
        for r in (s.get("meshResources") or {}).get("resources") or []:
            if (r.get("name") or "").casefold() != key:
                continue
            matches = [md for md in r.get("materialsData") or [] if (md.get("name") or "").casefold() == material_name.casefold()]
            if not matches:
                continue
            if len(matches) > 1:
                raise FormatError(f"{mesh_name}: material {material_name!r} listed {len(matches)} times in materialsData")
            resolved = _resolve_number(r, matches[0].get("number"))
            if resolved is None:
                continue
            resolved["slot"] = s.get("name")
            resolved["embedded_name"] = material_name
            return resolved
    return None


# ---- models.pak writer -----------------------------------------------------------------------------------------

def write_models_pak(out_path: Path | str, documents: dict[str, dict], *, texts: dict[str, str] | None = None,
                     overwrite: bool = False) -> dict:
    """Write a deflated PAK (ZIP) with each JSON document at exactly the member path given.

    Member-path conventions (both documented, neither native-verified):
      * `models/player/player_tpp_skeleton.model` — the path as it appears in data0.pak;
      * `player_tpp_skeleton.model` at the ZIP root — the form the RavenWolf data3.pak used, (RT) user-confirmed to
        override the data0.pak member in game (survey 04 §5.2). Which lookup rule makes that work is (C).
    `texts` adds plain UTF-8 members (e.g. a patched `playerappearances.scr`). Paths are validated (relative, no
    `..`, no drive letters); the archive is verified with testzip() before the result is returned.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise BuildError(f"{out_path} exists (pass overwrite=True)")
    entries: list[tuple[str, bytes]] = []
    for name, doc in documents.items():
        entries.append((_check_member_path(name), json.dumps(doc, indent=2, ensure_ascii=True, allow_nan=False).encode("utf-8")))
    for name, text in (texts or {}).items():
        entries.append((_check_member_path(name), text.encode("utf-8")))
    seen = set()
    for name, _ in entries:
        if name.casefold() in seen:
            raise BuildError(f"duplicate member path {name!r}")
        seen.add(name.casefold())
    if not entries:
        raise BuildError("nothing to write")
    tmp = out_path.with_name(out_path.name + ".partial")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in entries:
            z.writestr(name, data)
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad is not None:
            tmp.unlink()
            raise BuildError(f"written PAK failed verification at member {bad!r}")
    os.replace(tmp, out_path)
    return {"path": str(out_path), "members": [n for n, _ in entries], "bytes": out_path.stat().st_size,
            "root_level_override": "(RT) user-confirmed for data3.pak root members; not native-verified"}


def _check_member_path(name: str) -> str:
    n = name.replace("\\", "/")
    if not n or n.startswith("/") or ".." in n.split("/") or ":" in n or not _SAFE_MEMBER.match(n):
        raise BuildError(f"invalid PAK member path {name!r}")
    return n
