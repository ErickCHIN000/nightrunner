""".rpx tree (pack.json spec) → RPACK.

Rules
-----
* Every resource in `spec["resources"]` is written, in that order (order = logical order = engine lookup order).
* Each part's bytes come from its `raw` file unless the resource's codec regenerates that part because one of its
  editable files changed (sha256 differs from the recorded value) — or `--force-codec` is given.
* A mesh entry flagged `needs_identity_fix` (by `nr select --rename/--replace`) gets its embedded `.msh` name
  rewritten to `<logical name>.msh` (mesh.identity.rename) so the validator's mesh_name check and MeshMgr agree.
* Layout policy: `preserve` when the spec carries the original storages and no part size changed and the
  resource set is complete; otherwise `auto` (contiguous for field08 bit 12, grouped otherwise).
* Output is written to `<out>.partial` and validated (container.validate) before it is renamed into place; a
  failing build never touches an existing output.
"""

from __future__ import annotations

import os
from pathlib import Path

from .codecs import BuildContext, codec_for, codec_import_error
from .container.rp6l import Pack, PackWriter, ResourceSpec, PartSpec, PartSource, Storage
from .container.validate import validate
from .errors import BuildError
from .util.hashing import sha256_file
from .util.jsonio import load_json
from .util.schema import matches as schema_matches


def _changed_files(entry: dict, res_dir: Path) -> list[str]:
    files = (entry.get("editable") or {}).get("files") or {}
    changed = []
    for rel, rec in files.items():
        p = res_dir / rel
        if not p.exists():
            changed.append(rel + " (missing)")
            continue
        if rec.get("sha256") and sha256_file(p) != rec["sha256"]:
            changed.append(rel)
    if (entry.get("editable") or {}).get("kind") == "cast" or int(str(entry.get("type", "0")), 0) == 0x10:
        from .mesh.codec import GLTF_FILES
        changed += [n for n in GLTF_FILES if (res_dir / n).is_file() and n not in files]
    return changed


def _part_bytes(entry: dict, k: int, rpx_dir: Path, replacements: dict[int, PartSource]) -> bytes:
    """Bytes of part ordinal *k*: the codec's replacement when there is one, else the raw file."""
    src = replacements.get(k)
    if src is not None:
        if src.data is not None:
            return bytes(src.data)
        with open(src.path, "rb") as fh:
            fh.seek(src.offset)
            return fh.read(src.size)
    raw = entry["parts"][k].get("raw")
    if raw is None:
        raise BuildError(f"{entry['name']!r}: part {k} has no raw file")
    rp = rpx_dir / raw
    if not rp.exists():
        raise BuildError(f"{entry['name']!r}: raw part file missing: {rp}")
    return rp.read_bytes()


def _apply_identity_fix(entry: dict, rpx_dir: Path, replacements: dict[int, PartSource]) -> dict:
    """`nr select --rename` / `--replace` flags a mesh (0x10) whose logical name no longer matches the mesh it was
    taken from with `needs_identity_fix`. MeshMgr registers the mesh under the `.msh` string embedded in the
    ClassReader image, not under the logical name (A: ResourceManagement GetFileName +0x17A60, survey 02 §3), so
    the image (0x10) + fixups (0x11) parts are rewritten here with `mesh.identity.rename` (appends the new string,
    retargets the root's name slot, keeps every other byte). Returns identity.rename's info dict."""
    from .mesh import identity          # lazy: the mesh package is optional at import time
    types = [int(p["type"], 0) for p in entry["parts"]]
    if 0x10 not in types or 0x11 not in types:
        raise BuildError(f"{entry['name']!r}: needs_identity_fix on a mesh without image (0x10) + fixups (0x11) parts")
    ki, kf = types.index(0x10), types.index(0x11)
    image, fixups, info = identity.rename(_part_bytes(entry, ki, rpx_dir, replacements),
                                          _part_bytes(entry, kf, rpx_dir, replacements),
                                          bytes.fromhex(entry["name_hex"]))
    if info["changed"]:
        replacements[ki] = PartSource(image)
        replacements[kf] = PartSource(fixups)
    return info


def load_spec(spec_path: Path | str) -> tuple[dict, Path]:
    spec_path = Path(spec_path)
    if spec_path.is_dir():
        spec_path = spec_path / "pack.json"
    spec = load_json(spec_path)
    if not schema_matches(spec.get("schema"), "nightrunner.rpx/1"):
        raise BuildError(f"{spec_path}: unknown schema {spec.get('schema')!r}")
    return spec, spec_path.parent


def build(spec_path: Path | str, out_path: Path | str, *, layout: str = "auto", field08: int | None = None,
          flags: int | None = None, force_codec: bool = False, options: dict | None = None,
          validate_output: bool = True) -> dict:
    spec, rpx_dir = load_spec(spec_path)
    out_path = Path(out_path)
    ctx = BuildContext(rpx_dir, spec, dict(options or {}))
    hdr = spec["header"]
    f08 = int(hdr["field08"], 0) if field08 is None else field08
    fl = int(hdr["flags"], 0) if flags is None else flags
    template_storages = [Storage.from_json(s) for s in spec.get("storages", [])]
    storage_order = [s.key for s in template_storages] or None

    resources: list[ResourceSpec] = []
    sizes_unchanged = True
    regenerated = []
    for entry in spec["resources"]:
        res_dir = rpx_dir / entry["dir"]
        codec = codec_for(int(entry["type"], 0))
        changed = _changed_files(entry, res_dir) if entry.get("editable") else []
        # parts extracted with --no-raw (textures) have no raw file: the codec must regenerate them even when the
        # editable files are untouched (QA 2026-09-15: `extract --no-raw` + `build` used to fail with "no raw file")
        no_raw = [k for k, p in enumerate(entry["parts"]) if p.get("raw") is None and not p.get("unsupported")]
        replacements: dict[int, PartSource] = {}
        if changed or no_raw or (force_codec and entry.get("editable")):
            err = codec_import_error(int(entry["type"], 0))
            if err is not None:
                raise BuildError(f"{entry['name']!r}: codec for type {entry['type']} unavailable ({err}); "
                                 f"refusing to drop edits {changed or no_raw}")
            replacements = codec.build(ctx, entry, res_dir)
            if changed and not replacements:
                ctx.warn(f"{entry['name']!r}: {changed} changed but the {codec.kind} codec regenerated no part "
                         "(edit ignored; raw parts written)")
            regenerated.append({"resource": entry["index"], "name": entry["name"], "changed": changed,
                                "no_raw": no_raw, "parts": sorted(replacements)})
        if entry.get("needs_identity_fix") and int(entry["type"], 0) == 0x10:
            info = _apply_identity_fix(entry, rpx_dir, replacements)
            regenerated.append({"resource": entry["index"], "name": entry["name"], "changed": ["embedded .msh name"],
                                "parts": sorted(replacements), "identity": {k: (v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
                                                                             for k, v in info.items()}})
        parts: list[PartSpec] = []
        for k, prec in enumerate(entry["parts"]):
            stype = int(prec["type"], 0)
            sidx = prec["storage_index"]
            st = template_storages[sidx] if sidx < len(template_storages) else None
            if k in replacements:
                src = replacements[k]
            else:
                raw = prec.get("raw")
                if raw is None:
                    raise BuildError(f"{entry['name']!r}: part {k} has no raw file and no codec regenerated it")
                rp = rpx_dir / raw
                if not rp.exists():
                    raise BuildError(f"{entry['name']!r}: raw part file missing: {rp}")
                src = PartSource(path=rp, offset=0, size=rp.stat().st_size)
            if src.size != prec["size"]:
                sizes_unchanged = False
            parts.append(PartSpec(
                type=stype, source=src,
                align_raw=st.align_raw if st else 8,
                storage_flags=st.flags if st else 0,
                storage_metadata=st.metadata if st else 0,
                flag_bits=int(prec["flag_bits"], 0), fc=int(prec["fc"], 0),
                offset_units=prec.get("offset_units"), storage_index=sidx,
            ))
        resources.append(ResourceSpec(name=bytes.fromhex(entry["name_hex"]), type=int(entry["type"], 0),
                                      flags=int(entry["flags"], 0), parts=parts, name_index=entry.get("name_index")))

    eff = layout
    complete = bool(spec.get("complete", False)) and len(resources) == int(hdr["logical_count"])
    if layout == "preserve" and not (complete and sizes_unchanged and template_storages):
        raise BuildError("'preserve' layout requires a complete spec with unchanged part sizes")
    if layout == "auto" and complete and sizes_unchanged and template_storages and f08 == int(hdr["field08"], 0):
        eff = "preserve"
    writer = PackWriter(f08, fl, eff, storage_order=storage_order, template_storages=template_storages or None,
                        name_blob_order=spec.get("name_blob_order") if complete else None)
    for r in resources:
        writer.add(r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fill = None
    final_size = None
    if eff == "preserve" and spec.get("source", {}).get("path") and Path(spec["source"]["path"]).exists() \
            and spec["source"].get("size") and Path(spec["source"]["path"]).stat().st_size == spec["source"]["size"]:
        # gaps are zero in every shipped pack, but copying the original bytes is the strict guarantee
        fill = Path(spec["source"]["path"])      # read per gap, not into memory (review F11)
        final_size = spec["source"]["size"]
    rep = writer.write(out_path, fill=fill, final_size=final_size, replace=not validate_output)
    result = {"output": str(out_path), "size": rep.size, "layout": rep.layout, "storages": rep.storages,
              "physicals": rep.physicals, "logicals": rep.logicals, "regenerated": regenerated,
              "warnings": rep.warnings + ctx.warnings}
    if validate_output:
        # validate the .partial; an existing output is only replaced by a valid pack (review F6)
        tmp = Path(rep.path)
        try:
            with Pack.open(tmp) as pk:
                v = validate(pk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        result["validation"] = v.to_json()
        if not v.ok:
            bad = out_path.with_name(out_path.name + ".invalid")
            os.replace(tmp, bad)
            result["output"] = str(bad)
            raise BuildError(f"built pack failed validation ({len(v.errors)} errors); kept as {bad}"
                             + ("; existing output left untouched" if out_path.exists() else ""))
        os.replace(tmp, out_path)
    return result
