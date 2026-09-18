"""mesh.json — everything about a mesh resource that model.cast cannot carry.

Schema "nightrunner.mesh/1". Byte blobs are base64 (`*_b64`), small opaque words are hex strings, floats are
plain JSON numbers (positions/UVs are NOT here — they are in the Cast; the raw vertex windows are, so that an
unedited vertex re-encodes bit-exactly and the raw holes (qtangent quads, weights, tails, format-8 ext) survive
a Cast round trip through tools that drop custom properties).
"""

from __future__ import annotations

import base64

import numpy as np

from .. import __version__
from .model import Model
from .vertex import VERTEX_DTYPES, STRIDES

SCHEMA = "nightrunner.mesh/1"


def _b64(data) -> str:
    return base64.b64encode(bytes(data)).decode("ascii")


def _floats(a) -> list[float]:
    return [float(x) for x in np.asarray(a).reshape(-1)]


def _text(raw: bytes) -> str:
    """JSON-safe rendering of a name: UTF-8 with U+FFFD for invalid bytes (json cannot carry lone surrogates;
    common_meshes #2572 has an entity named 'dlc_ft_ce_a_hatch_1x_a_anm\x8c'). The exact bytes are in *_hex."""
    return raw.decode("utf-8", "replace")


def field_table(fmt: int) -> list[dict]:
    dt = VERTEX_DTYPES[fmt]
    out = []
    for name in dt.names:
        sub, off = dt.fields[name][0], dt.fields[name][1]
        out.append({"name": name, "offset": off, "dtype": sub.base.str, "count": int(np.prod(sub.shape)) if sub.shape else 1})
    return out


def build_sidecar(model: Model, *, part_sha256: dict[str, str] | None = None, cast: dict | None = None,
                  source: dict | None = None) -> dict:
    entries = []
    for e in model.geometry_entries:
        rec = {
            "index": e.index, "array_record": e.array_record, "element": e.element, "image_offset": e.offset,
            "owner_entity": e.owner_entity, "format": e.format, "vertex_base": e.vertex_base,
            "vertex_count": e.vertex_count, "index_base": e.index_base,
            "raw_00": e.raw_00.hex(), "raw_12": e.raw_12, "raw_14": e.raw_14, "raw_17": e.raw_17, "raw_34": e.raw_34.hex(),
            "material_slots_offset": e.material_slots_offset, "index_counts_offset": e.index_counts_offset,
            **({"stream_offset": e.stream_offset, "raw_stream": e.raw_stream.hex()} if e.stream_offset is not None else {}),
            "submeshes": [{
                "index": s.index, "material_slot": s.material_slot, "index_count": s.index_count,
                "index_base": s.index_base, "palette": [int(x) for x in s.palette],
                "palette_desc_offset": s.palette_desc_offset, "palette_offset": s.palette_offset,
            } for s in e.submeshes],
        }
        if e.vertices is not None:
            v = e.vertices
            rec["vertex_stride"] = STRIDES[e.format]
            rec["vertex_fields"] = field_table(e.format)
            rec["vertex_raw_b64"] = _b64(v.raw.tobytes())      # exact on-disk records, keyed by vertex id
            sums = v.weight_sums()
            if sums is not None:
                u, c = np.unique(sums, return_counts=True)
                rec["weight_sum_histogram"] = {int(a): int(b) for a, b in zip(u, c)}
        entries.append(rec)
    entities = [{
        "index": en.index, "image_offset": en.offset, "name": _text(en.name), "name_hex": en.name.hex(),
        "parent": en.parent, "type": en.type, "flags": f"0x{en.flags:08X}", "geometry_count": en.geometry_count,
        "geometry_array_record": en.geometry_array_record, "geometry_entries": list(en.geometry_entries),
        "local_3x4": _floats(en.local), "inv_bind_3x4": _floats(en.inv_bind),
        "bounds_center": _floats(en.bounds_center), "bounds_half": _floats(en.bounds_half),
        "aux_offset": en.aux_offset, "raw_aux": en.raw_aux.hex(), "raw_90": en.raw_90.hex(), "raw_ca": en.raw_ca.hex(),
    } for en in model.entities]
    materials = {
        "header_offset": model.material_header_offset, "count": len(model.materials), "capacity": model.material_capacity,
        "entries": [{"index": m.index, "image_offset": m.offset, "name": _text(m.name), "name_hex": m.name.hex(),
                     "name_tag": f"0x{m.name_tag:04X}", "name_inline": m.name_inline, "raw": m.raw.hex()} for m in model.materials],
    }
    opaque = [{"record": o.record, "class_id": o.class_id, "offset": o.offset, "size": o.size,
               **({"hex": o.data.hex()} if o.class_id != 0 else {})} for o in model.opaque]
    fx = model.fixups
    return {
        "schema": SCHEMA, "tool": f"nightrunner {__version__}",
        "name": _text(model.name.encode("utf-8", "surrogateescape")), "name_hex": model.name.encode("utf-8", "surrogateescape").hex(),
        "layout": model.layout,
        "embedded_name": _text(model.embedded_name),
        "embedded_name_hex": model.embedded_name.hex(),
        "scr_name": None if model.scr_name is None else _text(model.scr_name),
        "source": source or {},
        "sha256": part_sha256 or {},
        "buffers": {"vertex_size": model.vertex_buffer_size, "index_size": model.index_buffer_size,
                    "skin_size": None if model.skin_raw is None else len(model.skin_raw),
                    "cloth_size": None if model.cloth_raw is None else len(model.cloth_raw)},
        "fixups": fx.to_json(),
        "image_size": model.image.size, "image_part_size": len(model.image.data),
        "root_raw": model.root_raw.hex(),
        "entities": entities,
        "materials": materials,
        "geometry_entries": entries,
        "opaque_records": opaque,
        "cast": cast or {},
        "warnings": list(model.warnings),
    }
