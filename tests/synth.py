"""Shared helpers for the container-layer tests.

* synthetic packs built with `PackWriter` (stock storage attributes per part type, census 2026-09-15),
* byte-level patching of a rendered pack (to break specific fields deliberately),
* discovery of the six small shipped packs (PC: tests/paths.ASSETS; cloud: the staged uploads),
* temp directories inside the workspace (never outside F:\\DLTB / the cloud workspace).

stdlib only.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nightrunner.container.rp6l import (  # noqa: E402
    Pack, PackWriter, ResourceSpec, PartSpec, PartSource, HEADER_SIZE, STORAGE_SIZE, PHYSICAL_SIZE, LOGICAL_SIZE,
    PHYS_BIT8, LOGICAL_FLAGS_DEFAULT, LOGICAL_FLAGS_ONDEMAND_MESH,
)
from nightrunner.util.binio import align_up  # noqa: E402
from tests import paths  # noqa: E402

# ---- sample packs ---------------------------------------------------------------------------------------------

CLOUD_ASSETS = Path("/mnt/user-data/uploads/Dying Light The Beast/ph_ft/work/data_platform/pc/assets")

# relative to the assets dir; the six small shipped packs used by the unit tests
SMALL_PACKS = [
    "dlc_ft_prologue_envprobes_pc.rpack",               # 146 × 0x55, one storage, grouped
    "menu_level_ft/menu_level_ft_persistent_pc.rpack",  # 1 × 0x61 (two method-0 storages)
    "dlc_frontier/reg_buffer_cst_pa_pc.rpack",          # 1 × 0x61, 976 bytes
    "dlc_frontier/reg_pc.rpack",                        # 1 × 0x5A + 1 × 0x61
    "dlc_frontier/dlc_frontier_cb_region_0_pc.rpack",   # 4 × 0x5A, name blob order [3,0,1,2]
    "dlc_ft_prologue/reg1_pc.rpack",                    # 5 × 0x5A + 1 × 0x61, name blob order [3,2,1,0,4,5]
]


def assets_dir() -> Path | None:
    if paths.ASSETS.exists():
        return paths.ASSETS
    if CLOUD_ASSETS.exists():
        return CLOUD_ASSETS
    return None


def small_packs() -> list[Path]:
    a = assets_dir()
    if a is None:
        return []
    return [a / rel for rel in SMALL_PACKS if (a / rel).exists()]


def small_pack(name: str) -> Path | None:
    for p in small_packs():
        if p.name == name:
            return p
    return None


# ---- temp dirs inside the workspace -------------------------------------------------------------------------------

def _tmp_base() -> Path:
    """Where test files go: $NIGHTRUNNER_TEST_TMP, else out/test (the PC: F:\\DLTB\\out\\test), else — in the cloud,
    where the workspace is a slow FUSE mount — the session scratchpad that the environment designates for
    temporary files. Never anywhere else."""
    env = os.environ.get("NIGHTRUNNER_TEST_TMP")
    if env:
        return Path(env)
    if os.name == "nt" or paths.have_game():
        return paths.OUT
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if sid:
        sp = Path("/tmp/claude-0/-home-claude") / sid / "scratchpad"
        if sp.is_dir():
            return sp / "nightrunner_tests"
    return paths.OUT


def tmpdir(prefix: str = "t_") -> tempfile.TemporaryDirectory:
    base = _tmp_base()
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix=prefix, dir=str(base))


# ---- stock storage attributes (align_raw, flags, metadata) per part type ---------------------------------------------
# census 2026-09-15 (47/47 shipped packs): every storage has align_raw 8 (16-byte alignment); flags/metadata per type
# as below. Mesh family: on-demand packs (method 1, bit 3) vs engine_pc (method 0).

STOCK = {
    0x10: (8, 0xC9, 0x03), 0x11: (8, 0xC9, 0x03), 0x12: (8, 0xD9, 0x00),
    0xF0: (8, 0x59, 0x00), 0xF1: (8, 0x49, 0x00), 0xF3: (8, 0x29, 0x00),
    0x20: (8, 0xB1, 0x00), 0x21: (8, 0xB1, 0x00),
    0x40: (8, 0x40, 0x00), 0x42: (8, 0x40, 0x00), 0x43: (8, 0x40, 0x00),
    0x44: (8, 0x20, 0x00), 0x45: (8, 0x29, 0x00),
    0x47: (8, 0xC0, 0x08), 0x48: (8, 0xC0, 0x08), 0x49: (8, 0x40, 0x00), 0x4A: (8, 0x40, 0x00),
    0x61: (8, 0x80, 0x00), 0x62: (8, 0x80, 0x00),
    0x5A: (8, 0x21, 0x00), 0x55: (8, 0x21, 0x00), 0x56: (8, 0x21, 0x00),
}
METHOD0_MESH = {0x10: (8, 0xC0, 0x03), 0x11: (8, 0xC0, 0x03), 0x12: (8, 0xD0, 0x00), 0xF0: (8, 0x50, 0x00), 0xF1: (8, 0x40, 0x00)}

MESH_SHAPE = (0x10, 0x12, 0x11, 0xF0, 0xF1)


def payload(n: int, seed: int = 0) -> bytes:
    """Deterministic non-zero bytes so parts are distinguishable (seed byte + running counter)."""
    return bytes(((seed * 37 + i * 7 + 1) & 0xFF) or 1 for i in range(n))


def part(type_id: int, data: bytes, *, align_raw: int | None = None, flags: int | None = None,
         metadata: int | None = None, flag_bits: int = 0, fc: int = 0, table: dict = STOCK) -> PartSpec:
    a, f, m = table[type_id]
    return PartSpec(type=type_id, source=PartSource(data),
                    align_raw=a if align_raw is None else align_raw,
                    storage_flags=f if flags is None else flags,
                    storage_metadata=m if metadata is None else metadata,
                    flag_bits=flag_bits, fc=fc)


def mesh(name: bytes, sizes=(200, 40, 100, 1000, 300), *, seed: int = 1, ondemand: bool = True,
         flag_bits: int | None = None, table: dict | None = None) -> ResourceSpec:
    """A 5-part mesh resource in stock logical order (image, skin, fixups, vertex, index)."""
    tbl = table or (STOCK if ondemand else METHOD0_MESH)
    fb = (PHYS_BIT8 if ondemand else 0) if flag_bits is None else flag_bits
    parts = [part(t, payload(s, seed + k), flag_bits=fb, table=tbl) for k, (t, s) in enumerate(zip(MESH_SHAPE, sizes))]
    return ResourceSpec(name=name, type=0x10, flags=LOGICAL_FLAGS_ONDEMAND_MESH if ondemand else LOGICAL_FLAGS_DEFAULT, parts=parts)


def texture(name: bytes, header_size: int = 80, bitmap_size: int = 1000, *, seed: int = 9) -> ResourceSpec:
    return ResourceSpec(name=name, type=0x20, flags=LOGICAL_FLAGS_DEFAULT,
                        parts=[part(0x20, payload(header_size, seed)), part(0x21, payload(bitmap_size, seed + 1))])


def single(name: bytes, type_id: int, size: int, *, seed: int = 5) -> ResourceSpec:
    """One-part resource (area 0x5A, envprobe 0x55, voxelizer 0x56, anim 0x40)."""
    return ResourceSpec(name=name, type=type_id, flags=LOGICAL_FLAGS_DEFAULT, parts=[part(type_id, payload(size, seed))])


def prefab(name: bytes = b"Prefabs", sizes=(600, 220), *, seed: int = 3) -> ResourceSpec:
    return ResourceSpec(name=name, type=0x61, flags=LOGICAL_FLAGS_DEFAULT,
                        parts=[part(0x61, payload(sizes[0], seed)), part(0x62, payload(sizes[1], seed + 1))])


def write_pack(resources, path: Path, *, field08: int = 0, flags: int = 1, layout: str = "auto", **kw):
    w = PackWriter(field08, flags, layout, **kw)
    for r in resources:
        w.add(r)
    return w.write(path)


def build_bytes(resources, *, field08: int = 0, flags: int = 1, layout: str = "auto", **kw) -> bytes:
    """Write a synthetic pack into a workspace temp dir and return its bytes."""
    with tmpdir("synth_") as d:
        p = Path(d) / "synth.rpack"
        write_pack(resources, p, field08=field08, flags=flags, layout=layout, **kw)
        return p.read_bytes()


def open_bytes(data: bytes, name: str = "<synthetic>") -> Pack:
    return Pack.from_bytes(data, name)


# ---- byte patching ------------------------------------------------------------------------------------------------

def patch(data: bytes, offset: int, new: bytes) -> bytes:
    return data[:offset] + new + data[offset + len(new):]


def patch_u32(data: bytes, offset: int, value: int) -> bytes:
    return patch(data, offset, struct.pack("<I", value & 0xFFFFFFFF))


def patch_u8(data: bytes, offset: int, value: int) -> bytes:
    return patch(data, offset, bytes([value & 0xFF]))


def storage_off(pk: Pack, i: int) -> int:
    return HEADER_SIZE + i * STORAGE_SIZE


def physical_off(pk: Pack, i: int) -> int:
    return pk.physical_offset + i * PHYSICAL_SIZE


def logical_off(pk: Pack, i: int) -> int:
    return pk.logical_offset + i * LOGICAL_SIZE


def name_offset_off(pk: Pack, i: int) -> int:
    return pk.name_offset_offset + i * 4


def table_layout(data: bytes) -> Pack:
    """Open the tables of a rendered pack (no validation beyond the reader's own)."""
    return Pack.from_bytes(data)


def run_cli(*argv) -> tuple[int, str]:
    """Run `bp <argv>` in-process; returns (exit code, stdout)."""
    import contextlib
    import io
    from nightrunner.cli import main
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = main([str(a) for a in argv])
    return rc, out.getvalue()


def expected_table_end(n_storages: int, n_parts: int, n_resources: int, names: list[bytes]) -> int:
    return (HEADER_SIZE + n_storages * STORAGE_SIZE + n_parts * PHYSICAL_SIZE + n_resources * LOGICAL_SIZE
            + 4 * len(names) + sum(len(n) + 1 for n in names))


__all__ = [
    "Pack", "PackWriter", "ResourceSpec", "PartSpec", "PartSource", "align_up",
    "STOCK", "METHOD0_MESH", "MESH_SHAPE", "payload", "part", "mesh", "texture", "single", "prefab",
    "write_pack", "build_bytes", "open_bytes", "patch", "patch_u32", "patch_u8",
    "storage_off", "physical_off", "logical_off", "name_offset_off", "expected_table_end",
    "small_packs", "small_pack", "assets_dir", "tmpdir", "os", "run_cli",
]
