"""Mesh resources (logical type 0x10): decode to a Model, export to Cast, re-encode buffers.

    vertex.py    vertex formats 0/3/6/8 (numpy), qtangent maths, weight quantisation, exact re-encode
    model.py     dataclasses (Model, Entity, GeometryEntry, Submesh, Material, OpaqueObject)
    decode.py    Resource / raw parts → Model (all geometry arrays, all entries, all submeshes)
    encode.py    Model → vertex/index buffers (original layout; new-layout planner for phase 2)
    identity.py  embedded `.msh` name read / verify / rename
    rebuild.py   phase 2: Model + resolved Cast import → new part bytes (in-place / rebuild paths, layout, verify)
    imagepatch.py phase 2: the image/fixups edits of a rebuild plan (entries, arrays, palettes, materials, bounds, name)
    codec.py     MeshCodec (extract → model.cast + mesh.json, roundtrip, build) registered for type 0x10
    cli.py       `nr mesh info|export|dump|census|import|diff`
"""

from .decode import decode_resource, decode_parts  # noqa: F401
from .model import Model, Entity, GeometryEntry, Submesh, Material  # noqa: F401
