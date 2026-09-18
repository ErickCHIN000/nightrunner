"""SDB layout selection by the inner MDB version word: DLTB (0x24012001) and DL2 (0x23062801).

Synthetic databases built here in either layout (stdlib only, no game data). Real-file checks for DL2 live in the
validation notes (notes/FORMATS/sdb.md §9); the DLTB real-file checks are in test_gui_sdb / test_gui_matpreview.
"""
from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nightrunner.sdb.reader import (  # noqa: E402
    INNER_VERSION, INNER_VERSION_DL2, LAYOUT_DL2, LAYOUT_DLTB, OUTER_VERSION, PROGRAM_KINDS, Sdb, SdbError,
)


def compact(v: int) -> bytes:
    if v < 1 << 6:
        return bytes([v << 2])
    if v < 1 << 14:
        return struct.pack("<H", (v << 2) | 1)
    return struct.pack("<I", (v << 2) | 2)


def vec(items: list[bytes], unit: int) -> bytes:
    out = compact(len(items))
    for it in items:
        assert len(it) % unit == 0
        out += compact(len(it) // unit) + it
    return out


def flat(items: list[bytes], width: int) -> bytes:
    assert all(len(i) == width for i in items)
    return compact(len(items)) + b"".join(items)


AE = ["", "dif_0_tex", "roughness", "default_dif.png", "noise.dds", "override_dif.png"]
PROGRAM_BLOB = b"DXBC"


def build(layout, *, version: int | None = None) -> bytes:
    """One material `test_mat.mat` → route 0 → preset `opaque` (dif_0_tex: string, roughness: float) →
    shader 1 (texture array 1: dif_0_tex [override] + an unnamed binding [shader default noise.dds])."""
    s = lambda xs: [x.encode() for x in xs]  # noqa: E731
    shader = bytearray(25)
    struct.pack_into("<H", shader, 0x08, 0)        # pass 0
    struct.pack_into("<H", shader, 0x12, 1)        # texture array 1
    shader[0x17] = 1                               # one render pass
    d2 = struct.pack("<H", 5) + struct.pack("<f", 0.25)
    preset = struct.pack("<QII", 0x1122334455667788, 0, 0) + bytes(7)
    preset += compact(2)
    preset += struct.pack("<4H", 1, 1, 2, 0x8007) + compact(1) + b"\x00"
    preset += struct.pack("<4H", 2, 1, 2, 2) + compact(4) + struct.pack("<f", 0.5)
    preset += compact(1) + struct.pack("<I", 7) + compact(2) + struct.pack("<2I", 1, 2)
    content = {
        0x9A: flat([bytes(25), bytes(shader)], 25),
        0xA2: vec([b"", struct.pack("<4H", 1, 3, 0, 4)], 4),
        0xC2: vec([struct.pack("<Q", (1 << 48) | 0xABC)], 8),
        0xCA: flat([struct.pack("<5H", 0, 0, 0, 0, 0)], 10),
        0xCE: vec([struct.pack("<2I", 1 | (0 << 16), 2 | (2 << 16))], 4),
        0xD2: vec([d2], 1),
        0xD6: flat([bytes(8)], 8),
        0xAE: vec(s(AE), 1),
        0xB2: vec(s(["test_mat.mat"]), 1),
        0xB6: vec(s(["opaque;dif_0_tex;"]), 1),
        0xBA: vec(s(["opaque", "IN.x", "display(X)"]), 1),
        0xAA: compact(1) + preset,
    }
    a = b""
    for key, shape, width in layout.order_a:
        if key == 0x62:
            a += flat([bytes([i]) * width for i in range(3)], width)       # width matters for the walk
        elif key == 0x28:
            a += flat([bytes([i]) * width for i in range(2)], width)
        else:
            a += content.get(key, compact(0))
    for kind in PROGRAM_KINDS:
        blobs = [PROGRAM_BLOB] if kind == 2 else [b""]
        a += compact(len(blobs)) + b"".join(struct.pack("<I", len(x)) for x in blobs)
    for key, shape, width in layout.order_b:
        a += content.get(key, compact(0))
    block_b = PROGRAM_BLOB
    inner = b"MDB " + struct.pack("<I", layout.inner_version if version is None else version) + bytes(28)
    block_a = inner + a
    return b"MDBR" + struct.pack("<III", OUTER_VERSION, len(block_a), len(block_b)) + block_a + block_b


class LayoutTests(unittest.TestCase):
    def check_resolves(self, layout):
        s = Sdb(build(layout))
        self.assertIs(s.layout, layout)
        self.assertEqual(s.inner_version, layout.inner_version)
        self.assertEqual(s.materials(), ["test_mat.mat"])
        self.assertEqual(s.tables[0x62].count, 3)
        self.assertEqual(s.tables[0x28].count, 2)
        self.assertEqual(bytes(s.program(2, 0)), PROGRAM_BLOB)
        m = s.material("TEST_MAT")
        (r,) = m["routes"]
        self.assertEqual(r["preset"], "opaque")
        self.assertEqual([(p["name"], p["type"], p["value"]) for p in r["parameters"]],
                         [("dif_0_tex", "string", "override_dif.png"), ("roughness", "float", 0.25)])
        (v,) = r["variants"]
        self.assertEqual((v["shader"], v["render_pass_count"], v["texture_array"]), (1, 1, 1))
        self.assertEqual([(b["texture"], b["source"]) for b in v["texture_bindings"]],
                         [("override_dif.png", "override"), ("noise.dds", "shader_default")])
        self.assertEqual(s.textures_for_material(0), {"override_dif.png", "noise.dds"})
        p = s.preset(0)
        self.assertTrue(p["complete"])
        self.assertEqual(p["name"], "opaque")
        v = s.validate_all()["totals"]
        self.assertEqual((v["materials"], v["errors"], v["bindings"], v["texture_names"]), (1, 0, 2, 2))
        st = s.stats()
        self.assertEqual([t["key"] for t in st["tables"]], [f"0x{k:02X}" for k, _, _ in layout.order])
        return s

    def test_dltb_layout(self):
        s = self.check_resolves(LAYOUT_DLTB)
        self.assertEqual(len(s.tables), 39)
        self.assertEqual(s.tables[0x62].width, 44)

    def test_dl2_layout(self):
        s = self.check_resolves(LAYOUT_DL2)
        self.assertEqual(len(s.tables), 38)
        self.assertNotIn(0x48, s.tables)
        self.assertEqual((s.tables[0x62].width, s.tables[0x28].width), (43, 25))
        with self.assertRaises(SdbError):
            s.table(0x48)

    def test_layout_constants(self):
        self.assertEqual(LAYOUT_DLTB.inner_version, INNER_VERSION)
        self.assertEqual(LAYOUT_DL2.inner_version, INNER_VERSION_DL2)
        self.assertEqual(len(LAYOUT_DL2.order), len(LAYOUT_DLTB.order) - 1)
        diff = {k: (w1, w2) for (k, _, w1), (_, _, w2) in
                zip([t for t in LAYOUT_DLTB.order if t[0] != 0x48], LAYOUT_DL2.order) if w1 != w2}
        self.assertEqual(diff, {0x62: (44, 43), 0x28: (27, 25)})

    def test_layout_follows_version_word_not_bytes(self):
        # DL2 table stream labelled with the DLTB version (and vice versa) must fail loudly, never mis-resolve.
        with self.assertRaises(SdbError):
            Sdb(build(LAYOUT_DL2, version=INNER_VERSION))
        with self.assertRaises(SdbError):
            Sdb(build(LAYOUT_DLTB, version=INNER_VERSION_DL2))

    def test_unknown_version(self):
        with self.assertRaisesRegex(SdbError, "0x24012001.*0x23062801.*0x12345678"):
            Sdb(build(LAYOUT_DL2, version=0x12345678))


def _dl2_sdbs() -> list[Path]:
    """Installed DL2 SDBs (NIGHTRUNNER_DL2_ROOT or the default Steam path; data folder `ph` or `ph_ft`)."""
    import os
    root = Path(os.environ.get("NIGHTRUNNER_DL2_ROOT") or r"C:\Program Files (x86)\Steam\steamapps\common\Dying Light 2")
    return [f for ph in ("ph", "ph_ft") for f in sorted((root / ph / "work/data_platform/pc/assets").glob("runtime_dx1?.sdb"))]


@unittest.skipUnless(_dl2_sdbs(), "no Dying Light 2 install with runtime_dx11/dx12.sdb")
class RealDl2Tests(unittest.TestCase):
    def test_open_and_resolve_sample(self):
        for path in _dl2_sdbs():
            with self.subTest(path.name), Sdb.open(path) as s:
                self.assertIs(s.layout, LAYOUT_DL2)
                self.assertGreater(s.tables[0xB2].count, 20000)             # 25,760 dx11 / 25,741 dx12 (2026-09-16)
                self.assertEqual(s.tables[0xAA].count, 523)
                n = s.tables[0xB2].count
                texs = set()
                for i in range(0, n, max(1, n // 300)):
                    m = s.material(i)
                    self.assertEqual(len(m["routes"]), 1)
                    self.assertTrue(m["routes"][0]["preset_indices"])
                    texs |= s.textures_for_material(i)
                self.assertTrue(any(t.endswith((".dds", ".png")) for t in texs))


if __name__ == "__main__":
    unittest.main()
