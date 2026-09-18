"""Embedded `.msh` identity: read, verify, rename.

MeshMgr registers a CCompactMesh under the string at root+0x00 of the ClassReader image, not under the RPACK
logical name (A: ResourceManagement GetFileName +0x17A60, Create +0xC3F0; 4 runtime controls, survey 02 §3).
A duplicated/renamed mesh must carry `<logical name>.msh` there, exact case.
"""

from __future__ import annotations

from ..classreader.fixups import Fixups
from ..classreader.image import Image, ImagePatch, embedded_mesh_name
from ..errors import FormatError

SUFFIX = b".msh"


def expected_embedded_name(logical_name: bytes | str) -> bytes:
    raw = logical_name if isinstance(logical_name, bytes) else logical_name.encode("utf-8", "surrogateescape")
    return raw if raw.lower().endswith(SUFFIX) else raw + SUFFIX


def verify(image_bytes, fixups_bytes, logical_name: bytes | str) -> tuple[bool, bytes, bytes]:
    """→ (ok, embedded, expected)."""
    emb = embedded_mesh_name(image_bytes, fixups_bytes)
    exp = expected_embedded_name(logical_name)
    return emb == exp, emb, exp


def rename(image_bytes, fixups_bytes, logical_name: bytes | str) -> tuple[bytes, bytes, dict]:
    """Rewrite the embedded name to `<logical name>.msh` by appending a new string at the end of the primary
    image and retargeting the root's existing name slot (never overwrites shared string storage; keeps every
    other byte). Returns (image, fixups, {"before", "after", "changed"})."""
    exp = expected_embedded_name(logical_name)
    if not exp or b"\0" in exp or len(exp) > 65535:
        raise FormatError("invalid embedded mesh name")
    before = embedded_mesh_name(image_bytes, fixups_bytes)
    if before == exp:
        return bytes(image_bytes), bytes(fixups_bytes), {"before": before, "after": exp, "changed": False}
    img = Image(image_bytes, Fixups.parse(fixups_bytes))
    root = img.records[0].offset
    patch = ImagePatch(img)
    off = patch.append_string(exp)
    patch.retarget(root, off)
    new_image, new_fixups = patch.finish()
    after = embedded_mesh_name(new_image, new_fixups)
    if after != exp:
        raise FormatError("embedded name rename verification failed")
    return new_image, new_fixups, {"before": before, "after": exp, "changed": True}
