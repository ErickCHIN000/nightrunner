"""TextureCodec — logical type 0x20 (parts: 0x20 IMGC header, 0x21 bitmap) ↔ DDS + tex.json sidecar.

extract   <stem>.dds (DX10, padding stripped, face-major) + <stem>.tex.json (every header field that the DDS
          cannot carry, sha256 of the original parts, notes). Header-only records (flag 0x02) get the sidecar only.
roundtrip IMGC → DDS bytes → IMGC, both parts regenerated in memory (the corpus gate proves both directions).
build     parts 0x20/0x21 regenerated from the DDS (or a PNG named by the sidecar "source"): geometry, format and
          mips come from the image; flags/version/header_size/extension/reserved/mip_split from the sidecar;
          statistics kept when the texels are unchanged, recomputed otherwise (uncompressed + BC4/BC5; other BC
          formats keep the sidecar values unless "import.stats" says otherwise).

Sidecar (tex.json) keys the builder reads:
    imgc            ImgcHeader.to_json() of the original header (template for non-derivable fields)
    source          relative file to build from (default "<stem>.dds"; ".png" selects the PNG importer)
    import          {"stats": "auto"|"keep"|"recompute", "stats_values": {minimum,maximum,mean},
                     "png_format": 38|0|16, "mip_count": N|null, "srgb_to_linear": bool, "allow_unobserved": bool}
    sha256          {"header": …, "bitmap": …} of the original parts (texel-change detection)
    level_padding   16 (stock: every level padded to a 16-byte stride) or 0 (third-party tight layout, seen on 12
                    textures of custom_rpacks/frank.rpack); detected at extraction, reproduced by build/roundtrip
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .. import codecs
from ..container.rp6l import PartSource, Resource
from ..errors import BuildError, FormatError, UnsupportedError
from ..util.jsonio import dump_json, load_json
from ..util.names import safe_filename
from . import formats
from .dds import apply_dds_geometry, dds_to_levels, imgc_to_dds, iter_dds, read_dds
from .imgc import (ImgcHeader, LEVEL_PADDING_STOCK, LEVEL_PADDINGS, check_payload, detect_level_padding, join_payload,
                   level_layout, pack_header, parse_header, split_payload)
from .png import compute_stats, decode_level, png_to_imgc
from ..util.schema import matches as schema_matches

SIDECAR_SCHEMA = "nightrunner.tex/1"
PART_HEADER, PART_BITMAP = 0x20, 0x21
STATS_DECODABLE = set(formats.FORMATS) - {59, 60, 61, 66, 67, 68, 70, 71, 73}   # own decoders only (no Pillow BCn)


def texture_stem(name: str) -> str:
    """Resource name -> file stem: sanitised, one trailing source extension (".png"/".dds"/...) removed."""
    s = safe_filename(name)
    if "." in s:
        stem, ext = s.rsplit(".", 1)
        if stem and 1 <= len(ext) <= 5 and ext.isalnum():
            s = stem
    return s


def part_ordinals(part_types) -> tuple[int | None, int | None]:
    """(ordinal of the 0x20 part, ordinal of the 0x21 part) within the resource."""
    hdr = bmp = None
    for k, t in enumerate(part_types):
        if t == PART_HEADER and hdr is None:
            hdr = k
        elif t == PART_BITMAP and bmp is None:
            bmp = k
    return hdr, bmp


def _sha(data) -> str:
    return hashlib.sha256(data).hexdigest()


def stats_from_texels(h: ImgcHeader, bitmap) -> tuple[list[float], list[float], list[float]]:
    """min/max/mean over the mip-0 texels (all faces / slices) using the exact decoders only."""
    if h.format not in STATS_DECODABLE:
        raise UnsupportedError(f"statistics for {h.format_name} would need a BC decoder (Pillow's is preview-only)")
    levels = level_layout(h, detect_level_padding(h, len(bitmap)))
    views = split_payload(h, bitmap)
    chunks = []
    for lv, data in zip(levels, views):
        if lv.mip != 0:
            break
        per_slice = lv.slice_size
        for s in range(lv.depth):
            vals, _ = decode_level(data[s * per_slice:(s + 1) * per_slice], lv.width, lv.height, h.format)
            chunks.append(vals.reshape(-1, vals.shape[-1]))
    return compute_stats(np.concatenate(chunks, axis=0))


class TextureCodec(codecs.Codec):
    kind = "dds"

    # ---- extract ---------------------------------------------------------------------------------------------

    def extract(self, ctx, res: Resource, out_dir: Path) -> dict:
        hdr_k, bmp_k = part_ordinals(res.part_types)
        if hdr_k is None:
            raise FormatError(f"{res.name!r}: no 0x20 IMGC header part (parts: {[hex(t) for t in res.part_types]})")
        pack = res.pack
        hdr_i = res.part_indices[hdr_k]
        raw_hdr = pack.read_part(hdr_i)
        h = parse_header(raw_hdr)
        stem = texture_stem(res.name)
        dds_rel, json_rel = f"{stem}.dds", f"{stem}.tex.json"
        notes: list[str] = []
        sidecar = {
            "schema": SIDECAR_SCHEMA, "name": res.name, "index": res.index,
            "imgc": h.to_json(), "dxgi": formats.FORMATS[h.format].dxgi if h.format in formats.FORMATS else None,
            "format_tier": formats.FORMATS[h.format].tier if h.format in formats.FORMATS else None,
            "parts": {"header_ordinal": hdr_k, "bitmap_ordinal": bmp_k},
            "sha256": {"header": _sha(raw_hdr), "bitmap": None},
            "source": None, "import": {"stats": "auto", "png_format": None, "mip_count": None,
                                       "srgb_to_linear": False, "allow_unobserved": False},
            "notes": notes,
        }
        files: dict = {}
        record: dict = {"kind": self.kind, "sidecar": json_rel, "dds": None, "format": h.format_name,
                        "header_only": h.header_only}
        if h.header_only:
            notes.append(f"header-only record (flag 0x02): no bitmap part; reference {h.reference!r} kept in the "
                         f"extension bytes. Whether the engine substitutes the referenced texture is unverified (C).")
            if bmp_k is not None:
                notes.append("unexpected 0x21 part on a flag-0x02 record: kept raw only")
                ctx.warn(f"{res.name!r}: flag 0x02 with a bitmap part")
        else:
            if bmp_k is None:
                raise FormatError(f"{res.name!r}: no 0x21 bitmap part and flag 0x02 is clear")
            bitmap = pack.read_part(res.part_indices[bmp_k])
            padding = detect_level_padding(h, len(bitmap))
            levels = check_payload(h, len(bitmap), padding)
            sidecar["sha256"]["bitmap"] = _sha(bitmap)
            sidecar["payload_size"] = len(bitmap)
            sidecar["levels"] = len(levels)
            sidecar["level_padding"] = padding
            if padding != LEVEL_PADDING_STOCK:
                notes.append(f"bitmap stored with level_padding {padding} (levels back-to-back, third-party layout; "
                             f"stock packs always pad each level to 16 — notes/FORMATS/imgc.md). The builder "
                             f"reproduces this value; set level_padding to 16 for the stock layout.")
            sidecar["source"] = dds_rel
            with open(out_dir / dds_rel, "wb") as fh:
                for chunk in iter_dds(h, bitmap):
                    fh.write(chunk)
            files.update(self.file_record(out_dir / dds_rel, dds_rel))
            record["dds"] = dds_rel
        dump_json(sidecar, out_dir / json_rel)
        files.update(self.file_record(out_dir / json_rel, json_rel))
        record["files"] = files
        return record

    # ---- roundtrip -------------------------------------------------------------------------------------------

    def roundtrip(self, res: Resource) -> dict[int, bytes]:
        hdr_k, bmp_k = part_ordinals(res.part_types)
        if hdr_k is None:
            raise FormatError(f"{res.name!r}: no 0x20 part")
        pack = res.pack
        h = parse_header(pack.read_part(res.part_indices[hdr_k]))
        out = {hdr_k: pack_header(h)}
        if h.header_only:
            return out
        if bmp_k is None:
            raise FormatError(f"{res.name!r}: no 0x21 part and flag 0x02 clear")
        bitmap = pack.read_part(res.part_indices[bmp_k])
        padding = detect_level_padding(h, len(bitmap))          # 16 stock / 0 third-party tight (sidecar value)
        dds_bytes = imgc_to_dds(h, bitmap)                      # IMGC → DDS
        d = read_dds(dds_bytes)                                 # DDS → geometry/format
        h2 = apply_dds_geometry(h, d)                           # header from DDS + sidecar-equivalent fields
        out[hdr_k] = pack_header(h2)
        out[bmp_k] = join_payload(h2, dds_to_levels(d, dds_bytes), padding)
        return out

    # ---- build -----------------------------------------------------------------------------------------------

    def build(self, ctx, entry: dict, res_dir: Path) -> dict[int, PartSource]:
        ed = entry.get("editable") or {}
        name = entry.get("name", "?")
        json_rel = ed.get("sidecar")
        if not json_rel or not (res_dir / json_rel).exists():
            raise BuildError(f"{name!r}: texture sidecar missing ({json_rel})")
        sc = load_json(res_dir / json_rel)
        if not schema_matches(sc.get("schema"), SIDECAR_SCHEMA):
            raise BuildError(f"{name!r}: sidecar schema {sc.get('schema')!r} (expected {SIDECAR_SCHEMA})")
        template = ImgcHeader.from_json(sc["imgc"])
        part_types = [int(p["type"], 0) for p in entry["parts"]]
        hdr_k, bmp_k = part_ordinals(part_types)
        if hdr_k is None:
            raise BuildError(f"{name!r}: pack.json lists no 0x20 part")
        opts = dict(sc.get("import") or {})
        source = sc.get("source") or ed.get("dds")
        padding = sc.get("level_padding", LEVEL_PADDING_STOCK)
        if padding not in LEVEL_PADDINGS:
            raise BuildError(f"{name!r}: sidecar level_padding {padding!r} must be 16 (stock) or 0 (tight layout)")

        if template.header_only:
            if source and (res_dir / source).exists() and bmp_k is None:
                raise BuildError(f"{name!r}: header-only record (flag 0x02) has no 0x21 part in pack.json; a bitmap "
                                 f"cannot be added without a part record (clear flag 0x02 and add the part)")
            return {hdr_k: PartSource(pack_header(template))}
        if bmp_k is None:
            raise BuildError(f"{name!r}: pack.json lists no 0x21 part")
        if not source:
            raise BuildError(f"{name!r}: sidecar names no source file")
        src = res_dir / source
        if not src.exists():
            raise BuildError(f"{name!r}: source {src} missing")

        if src.suffix.lower() == ".png":
            h, bitmap, info = self._from_png(src, template, opts, name, padding)
            geometry_changed = True
            stats_mode = "recomputed (PNG import)"
        else:
            buf = src.read_bytes()
            d = read_dds(buf, srgb_to_linear=bool(opts.get("srgb_to_linear")),
                         allow_unobserved=bool(opts.get("allow_unobserved")))
            for w in d.warnings:
                ctx.warn(f"{name!r}: {w}")
            h = apply_dds_geometry(template, d)
            bitmap = join_payload(h, dds_to_levels(d, buf), padding)
            geometry_changed = (h.width, h.height, h.depth, h.packed, h.format) != \
                               (template.width, template.height, template.depth, template.packed, template.format)
            h, stats_mode = self._apply_stats(h, bitmap, template, sc, opts, geometry_changed, ctx, name)
        if geometry_changed:
            ctx.warn(f"{name!r}: geometry/format changed {template.width}x{template.height}x{template.depth} "
                     f"{template.format_name} mips={template.mip_count} -> {h.width}x{h.height}x{h.depth} "
                     f"{h.format_name} mips={h.mip_count}; flags 0x{h.flags:02X}/mip_split/extension kept from the sidecar")
            if h.mip_split:
                ctx.warn(f"{name!r}: mip_split 0x{h.mip_split:016X} encodes the ORIGINAL mip byte sums "
                         f"(notes/FORMATS/imgc.md section 6) and is now stale; set imgc.mip_split to 0 in the sidecar "
                         f"(and clear flag 0x10) unless you know the engine ignores it")
        if formats.FORMATS[h.format].tier != "A":
            ctx.warn(f"{name!r}: IL {h.format} {h.format_name} never occurs in the shipped corpus (untested in game)")
        hdr_bytes = pack_header(h)
        check_payload(h, len(bitmap), padding)
        if padding != LEVEL_PADDING_STOCK:
            ctx.warn(f"{name!r}: bitmap written with level_padding {padding} (third-party tight layout per the sidecar; "
                     f"stock packs pad every level to 16)")
        if stats_mode.startswith("kept (texels changed"):
            ctx.warn(f"{name!r}: stats {stats_mode}")
        return {hdr_k: PartSource(hdr_bytes), bmp_k: PartSource(bitmap)}

    def _from_png(self, src: Path, template: ImgcHeader, opts: dict, name: str, padding: int = LEVEL_PADDING_STOCK):
        from PIL import Image
        fmt = opts.get("png_format")
        if fmt is None:
            fmt = template.format if template.format in (38, 0, 16) else 38
        mips = opts.get("mip_count")
        with Image.open(src) as im:
            im.load()
            h, bitmap, info = png_to_imgc(im, int(fmt), template, mip_count=mips, level_padding=padding)
        return h, bitmap, info

    def _apply_stats(self, h: ImgcHeader, bitmap: bytes, template: ImgcHeader, sc: dict, opts: dict,
                     geometry_changed: bool, ctx, name: str) -> tuple[ImgcHeader, str]:
        mode = (opts.get("stats") or "auto").lower()
        explicit = opts.get("stats_values")
        if explicit:
            h.set_stats(explicit["minimum"], explicit["maximum"], explicit["mean"])
            return h, "explicit (sidecar import.stats_values)"
        if mode == "keep":
            return h, "kept (import.stats = keep)"
        unchanged = (not geometry_changed) and sc.get("sha256", {}).get("bitmap") == _sha(bitmap)
        if mode == "auto" and unchanged:
            return h, "kept (texels unchanged)"
        if h.format not in STATS_DECODABLE:
            if mode == "recompute":
                raise BuildError(f"{name!r}: import.stats = recompute but {h.format_name} has no exact decoder here "
                                 f"(Pillow's BCn decoder is preview-only); use keep or stats_values")
            return h, f"kept (texels changed but {h.format_name} statistics need a BC decoder; set import.stats_values)"
        h.set_stats(*stats_from_texels(h, bitmap))
        return h, "recomputed from the new texels"


codecs.register(0x20, TextureCodec())
