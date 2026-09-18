"""Texture layer tests: IMGC header, level layout, DDS <-> IMGC, previews, PNG import, codec on the sample pack
(out/samples/textures.rpack + textures.rpx from tools/make_samples_texture.py) and a light corpus check."""

import io
import shutil
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.codecs import BuildContext, codec_for  # noqa: E402
from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.errors import FormatError, UnsupportedError, ValidationError  # noqa: E402
from nightrunner.texture import dds, formats, imgc, png  # noqa: E402
from nightrunner.texture.codec import TextureCodec, part_ordinals, texture_stem  # noqa: E402
from nightrunner.util.jsonio import dump_json, load_json  # noqa: E402
from tests.paths import ASSETS, OUT, SAMPLES, have_game  # noqa: E402

RNG = np.random.default_rng(1234)
SAMPLE_PACK = SAMPLES / "textures.rpack"
SAMPLE_RPX = SAMPLES / "textures.rpx"


def synth_header(w, h, d, tex_type, mips, fmt, flags=0x44) -> imgc.ImgcHeader:
    hd = imgc.ImgcHeader(flags=flags)
    hd.set_geometry(w, h, d, tex_type, mips, fmt)
    hd.set_stats([0.1, 0.2, 0.3, 1.0], [0.9, 0.8, 0.7, 1.0], [0.5, 0.5, 0.5, 1.0])
    return hd


def synth_bitmap(hd: imgc.ImgcHeader) -> bytes:
    return imgc.join_payload(hd, [bytes(RNG.integers(0, 256, lv.size, dtype=np.uint8)) for lv in imgc.level_layout(hd)])


class TestHeader(unittest.TestCase):
    def test_pack_unpack_identity(self):
        hd = synth_header(256, 128, 1, imgc.TYPE_2D, 9, 68, flags=0x64)
        hd.reserved = 7
        hd.mip_split = 0x1000016800000400
        raw = imgc.pack_header(hd)
        self.assertEqual(len(raw), 80)
        self.assertEqual(raw[:4], b"IMGC")
        self.assertEqual(struct.unpack_from("<I", raw, 4)[0], 0x20191127)
        back = imgc.parse_header(raw)
        self.assertEqual(imgc.pack_header(back), raw)
        self.assertEqual((back.width, back.height, back.depth, back.mip_count, back.tex_type, back.format, back.flags),
                         (256, 128, 1, 9, 0, 68, 0x64))
        self.assertEqual(back.reserved, 7)
        self.assertEqual(back.mip_split, 0x1000016800000400)
        self.assertAlmostEqual(back.minimum[0], 0.1, places=6)

    def test_extension_and_tail(self):
        hd = imgc.ImgcHeader(flags=0x02, header_size=106, extension=b"error_no_pow2_tex_org.png\0")
        raw = imgc.pack_header(hd)
        self.assertEqual(len(raw), 112)
        back = imgc.parse_header(raw)
        self.assertTrue(back.header_only)
        self.assertEqual(back.reference, "error_no_pow2_tex_org.png")
        self.assertEqual(back.tail, bytes(6))
        self.assertEqual(imgc.pack_header(back), raw)
        # non-zero tail bytes survive
        raw2 = raw[:106] + b"\x01\x02\x03\x04\x05\x06"
        self.assertEqual(imgc.pack_header(imgc.parse_header(raw2)), raw2)

    def test_stats_raw_bit_exact(self):
        hd = synth_header(4, 4, 1, imgc.TYPE_2D, 1, 0)
        hd.stats_raw = bytes(range(48))     # arbitrary bit patterns (incl. NaN-ish floats) must survive
        raw = imgc.pack_header(hd)
        self.assertEqual(imgc.parse_header(raw).stats_raw, bytes(range(48)))
        j = hd.to_json()
        self.assertEqual(imgc.ImgcHeader.from_json(j).stats_raw, bytes(range(48)))

    def test_json_roundtrip(self):
        hd = synth_header(64, 32, 1, imgc.TYPE_2D, 7, 59, flags=0x54)
        hd.mip_split = 0x1000016800000400
        back = imgc.ImgcHeader.from_json(hd.to_json())
        self.assertEqual(imgc.pack_header(back), imgc.pack_header(hd))

    def test_bad_inputs(self):
        with self.assertRaises(FormatError):
            imgc.parse_header(b"IMGC" + bytes(70))
        with self.assertRaises(FormatError):
            imgc.parse_header(b"IMGX" + bytes(76))
        raw = imgc.pack_header(synth_header(4, 4, 1, imgc.TYPE_2D, 1, 0))
        with self.assertRaises(FormatError):
            imgc.parse_header(raw + bytes(16))            # strict length
        imgc.parse_header(raw + bytes(16), strict_length=False)
        with self.assertRaises(ValidationError):
            synth_header(4, 4, 2, imgc.TYPE_2D, 1, 0)      # depth on a 2D
        with self.assertRaises(ValidationError):
            synth_header(4, 4, 1, imgc.TYPE_2D, 64, 0)     # mips > 63
        with self.assertRaises(ValidationError):
            synth_header(4, 4, 1, imgc.TYPE_2D, 1, 4)      # enum hole


class TestLayout(unittest.TestCase):
    def sizes(self, hd):
        return [(lv.mip, lv.face, lv.width, lv.height, lv.depth, lv.offset, lv.size, lv.padded_size)
                for lv in imgc.level_layout(hd)]

    def test_2d_block(self):
        hd = synth_header(256, 256, 1, imgc.TYPE_2D, 9, 59)
        s = self.sizes(hd)
        self.assertEqual(s[0], (0, 0, 256, 256, 1, 0, 32768, 32768))
        self.assertEqual(s[6], (6, 0, 4, 4, 1, 43680, 8, 16))
        self.assertEqual(s[8], (8, 0, 1, 1, 1, 43712, 8, 16))
        self.assertEqual(imgc.payload_size(hd), 43728)

    def test_2d_nonblock_npot(self):
        hd = synth_header(160, 554, 1, imgc.TYPE_2D, 10, 46)
        s = self.sizes(hd)
        self.assertEqual(s[0][6], 160 * 554 * 8)
        self.assertEqual(s[1][2:5], (80, 277, 1))
        self.assertEqual(s[1][6], 80 * 277 * 8)
        self.assertEqual(s[9][2:5], (1, 1, 1))
        total = sum(lv.padded_size for lv in imgc.level_layout(hd))
        self.assertEqual(imgc.payload_size(hd), total)
        self.assertEqual(imgc.payload_size(hd) % 16, 0)

    def test_cube(self):
        hd = synth_header(64, 64, 1, imgc.TYPE_CUBE, 7, 66)
        s = self.sizes(hd)
        self.assertEqual(len(s), 42)
        self.assertEqual([x[:2] for x in s[:7]], [(0, f) for f in range(6)] + [(1, 0)])   # mip-major
        self.assertEqual(s[5][5], 5 * 4096)                                              # face 5 of mip 0
        self.assertEqual(s[6][5], 6 * 4096)                                              # mip 1 face 0
        self.assertEqual(imgc.payload_size(hd), 6 * (4096 + 1024 + 256 + 64 + 16 + 16 + 16))

    def test_volume(self):
        hd = synth_header(32, 16, 8, imgc.TYPE_VOLUME, 6, 38)
        s = self.sizes(hd)
        self.assertEqual(s[0][2:5], (32, 16, 8))
        self.assertEqual(s[0][6], 32 * 16 * 8 * 4)
        self.assertEqual(s[3][2:5], (4, 2, 1))
        self.assertEqual(s[5][2:5], (1, 1, 1))
        lv = imgc.level_layout(hd)[1]
        self.assertEqual(lv.slice_size, 16 * 8 * 4)
        hd2 = synth_header(2, 2, 3, imgc.TYPE_VOLUME, 1, 59)        # engine_pc default_env.dds
        self.assertEqual(imgc.payload_size(hd2), 32)

    def test_check_payload_and_split_join(self):
        hd = synth_header(37, 5, 1, imgc.TYPE_2D, 6, 63)
        bm = synth_bitmap(hd)
        levels = imgc.split_payload(hd, bm)
        self.assertEqual(len(levels), 6)
        self.assertEqual(imgc.join_payload(hd, levels), bm)
        with self.assertRaises(FormatError):
            imgc.check_payload(hd, len(bm) + 16)

    def test_level_bytes_table(self):
        for il, (w, h, want) in {59: (5, 5, 32), 63: (1, 1, 8), 68: (4, 4, 16), 0: (7, 3, 21), 46: (2, 2, 32)}.items():
            self.assertEqual(formats.level_bytes(il, w, h), want, il)


class TestLevelPadding(unittest.TestCase):
    """Third-party tight layout (custom_rpacks/frank.rpack: 12 × 4096² BC1/BC4, part 24 bytes short = the three 8-byte
    tail mips stored without their 16-byte stride). Synthetic 64² BC1 with 7 mips has the same 3 × 8 → 3 × 16 tail."""

    def setUp(self):
        self.hd = synth_header(64, 64, 1, imgc.TYPE_2D, 7, 59)
        self.levels = [bytes(RNG.integers(0, 256, lv.size, dtype=np.uint8)) for lv in imgc.level_layout(self.hd)]

    def test_layout_and_detection(self):
        padded = imgc.level_layout(self.hd)
        tight = imgc.level_layout(self.hd, 0)
        self.assertEqual([lv.size for lv in padded], [2048, 512, 128, 32, 8, 8, 8])
        self.assertEqual([lv.padded_size for lv in padded], [2048, 512, 128, 32, 16, 16, 16])
        self.assertEqual([lv.padded_size for lv in tight], [2048, 512, 128, 32, 8, 8, 8])
        self.assertEqual([lv.offset for lv in tight], [0, 2048, 2560, 2688, 2720, 2728, 2736])
        self.assertEqual((imgc.payload_size(self.hd), imgc.payload_size(self.hd, 0)), (2768, 2744))
        self.assertEqual(imgc.detect_level_padding(self.hd, 2768), 16)
        self.assertEqual(imgc.detect_level_padding(self.hd, 2744), 0)
        for bad in (2745, 2760, 2769, 0):
            with self.assertRaises(FormatError):
                imgc.detect_level_padding(self.hd, bad)
        with self.assertRaises(ValidationError):
            imgc.level_layout(self.hd, 8)
        # a texture whose every level is already a multiple of 16 has one layout only: it is reported as stock
        h16 = synth_header(16, 16, 1, imgc.TYPE_2D, 1, 38)
        self.assertEqual(imgc.payload_size(h16), imgc.payload_size(h16, 0))
        self.assertEqual(imgc.detect_level_padding(h16, 1024), 16)

    def test_split_join_both_layouts(self):
        bm16 = imgc.join_payload(self.hd, self.levels)
        bm0 = imgc.join_payload(self.hd, self.levels, 0)
        self.assertEqual((len(bm16), len(bm0)), (2768, 2744))
        self.assertEqual(bm0, b"".join(self.levels))                      # back-to-back, nothing inserted
        self.assertEqual([bytes(v) for v in imgc.split_payload(self.hd, bm0)], self.levels)     # auto-detected
        self.assertEqual([bytes(v) for v in imgc.split_payload(self.hd, bm16)], self.levels)
        self.assertEqual(imgc.join_payload(self.hd, imgc.split_payload(self.hd, bm0), 0), bm0)
        self.assertEqual(imgc.join_payload(self.hd, imgc.split_payload(self.hd, bm0), 16), bm16)
        with self.assertRaises(FormatError):
            imgc.split_payload(self.hd, bm0, 16)                            # explicit padding must match exactly
        # DDS export is identical from either layout (padding is not part of the DDS)
        self.assertEqual(dds.imgc_to_dds(self.hd, bm0), dds.imgc_to_dds(self.hd, bm16))

    def test_codec_roundtrip_extract_build(self):
        """A synthetic pack holding one tight and one padded texture: roundtrip identical, extract records the
        padding in tex.json, build honours it (byte-identical), and flipping it to 16 re-pads the part."""
        from tests.synth import ResourceSpec, part, write_pack, tmpdir, run_cli
        hdr = imgc.pack_header(self.hd)
        bm0 = imgc.join_payload(self.hd, self.levels, 0)
        bm16 = imgc.join_payload(self.hd, self.levels)
        with tmpdir("lp_") as d:
            d = Path(d)
            pack = d / "lp.rpack"
            write_pack([ResourceSpec(name=b"tight.png", type=0x20, flags=1, parts=[part(0x20, hdr), part(0x21, bm0)]),
                        ResourceSpec(name=b"padded.png", type=0x20, flags=1, parts=[part(0x20, hdr), part(0x21, bm16)])], pack)
            codec = codec_for(0x20)
            with Pack.open(pack) as pk:
                for res in pk:
                    out = codec.roundtrip(res)
                    for k, i in enumerate(res.part_indices):
                        self.assertEqual(bytes(pk.read_part(i)), out[k], (res.name, k))
            rpx = d / "lp.rpx"
            self.assertEqual(run_cli("extract", pack, rpx)[0], 0)
            spec = load_json(rpx / "pack.json")
            self.assertEqual(spec["warnings"], [])
            sc = {e["name"]: load_json(rpx / e["dir"] / e["editable"]["sidecar"]) for e in spec["resources"]}
            self.assertEqual((sc["tight.png"]["level_padding"], sc["padded.png"]["level_padding"]), (0, 16))
            self.assertEqual((sc["tight.png"]["payload_size"], sc["padded.png"]["payload_size"]), (2744, 2768))
            self.assertTrue(any("level_padding 0" in n for n in sc["tight.png"]["notes"]))
            self.assertEqual(sc["padded.png"]["notes"], [])
            # byte identity without edits, and with --force-codec (both parts regenerated through the codec)
            for extra in ((), ("--force-codec",)):
                rc, text = run_cli("build", rpx, d / "lp_out.rpack", *extra)
                self.assertEqual(rc, 0, text)
                self.assertEqual((d / "lp_out.rpack").read_bytes(), pack.read_bytes(), extra)
            # the sidecar value is the authority: 0 → 16 re-pads the tight texture (part grows by 24 bytes)
            e = next(e for e in spec["resources"] if e["name"] == "tight.png")
            sc0 = sc["tight.png"]
            sc0["level_padding"] = 16
            dump_json(sc0, rpx / e["dir"] / e["editable"]["sidecar"])
            rc, text = run_cli("build", rpx, d / "lp_pad.rpack")
            self.assertEqual(rc, 0, text)
            with Pack.open(d / "lp_pad.rpack") as pk:
                self.assertEqual(bytes(pk.read_part(1)), bm16)
                self.assertEqual(bytes(pk.read_part(3)), bm16)
            # an edited DDS keeps the tight layout when the sidecar says 0
            sc0["level_padding"] = 0
            dump_json(sc0, rpx / e["dir"] / e["editable"]["sidecar"])
            new_levels = [bytes(RNG.integers(0, 256, len(lv), dtype=np.uint8)) for lv in self.levels]
            (rpx / e["dir"] / e["editable"]["dds"]).write_bytes(dds.imgc_to_dds(self.hd, imgc.join_payload(self.hd, new_levels)))
            rc, text = run_cli("build", rpx, d / "lp_edit.rpack")
            self.assertEqual(rc, 0, text)
            with Pack.open(d / "lp_edit.rpack") as pk:
                self.assertEqual(bytes(pk.read_part(1)), b"".join(new_levels))
                self.assertEqual(pk.physicals[1].size, 2744)
            # a bogus sidecar value is refused
            sc0["level_padding"] = 8
            dump_json(sc0, rpx / e["dir"] / e["editable"]["sidecar"])
            self.assertEqual(run_cli("build", rpx, d / "lp_bad.rpack", "--force-codec")[0], 2)


class TestDds(unittest.TestCase):
    CASES = [(256, 256, 1, 0, 9, 59), (64, 64, 1, 1, 7, 66), (32, 16, 8, 2, 6, 38), (160, 554, 1, 0, 10, 46),
             (5, 3, 1, 0, 1, 63), (1, 1, 1, 0, 1, 0), (128, 128, 1, 1, 1, 64), (8, 8, 4, 2, 4, 61),
             (12774, 100, 1, 0, 1, 16), (3, 3, 1, 0, 2, 32), (16, 16, 16, 2, 1, 8)]

    def test_roundtrip_all_geometries(self):
        for case in self.CASES:
            hd = synth_header(*case)
            bm = synth_bitmap(hd)
            d = dds.imgc_to_dds(hd, bm)
            self.assertEqual(len(d), dds.dds_size_for(hd), case)
            df = dds.read_dds(d)
            h2 = dds.apply_dds_geometry(hd, df)
            self.assertEqual(imgc.pack_header(h2), imgc.pack_header(hd), case)
            self.assertEqual(imgc.join_payload(h2, dds.dds_to_levels(df, d)), bm, case)

    def test_header_bytes(self):
        hd = synth_header(64, 64, 1, imgc.TYPE_CUBE, 7, 66)
        hdr = dds.dds_header_for(hd)
        self.assertEqual(len(hdr), 148)
        magic, size, flags, height, width, pitch, depth, mips = struct.unpack_from("<4s7I", hdr, 0)
        self.assertEqual((magic, size, flags, height, width, pitch, depth, mips),
                         (b"DDS ", 124, 0x1007 | 0x80000 | 0x20000, 64, 64, 4096, 0, 7))
        self.assertEqual(struct.unpack_from("<II4s", hdr, 76), (32, 4, b"DX10"))
        caps, caps2 = struct.unpack_from("<II", hdr, 108)
        self.assertEqual((caps, caps2), (0x1000 | 0x400008, 0xFE00))
        self.assertEqual(struct.unpack_from("<5I", hdr, 128), (95, 3, 4, 1, 0))
        hv = synth_header(32, 16, 8, imgc.TYPE_VOLUME, 6, 38)
        hdr = dds.dds_header_for(hv)
        self.assertEqual(struct.unpack_from("<I", hdr, 8)[0], 0x1007 | 0x8 | 0x20000 | 0x800000)
        self.assertEqual(struct.unpack_from("<I", hdr, 20)[0], 32 * 4)
        self.assertEqual(struct.unpack_from("<I", hdr, 24)[0], 8)
        self.assertEqual(struct.unpack_from("<II", hdr, 108), (0x1000 | 0x400008, 0x200000))
        self.assertEqual(struct.unpack_from("<5I", hdr, 128), (28, 4, 0, 1, 0))

    def test_face_order(self):
        hd = synth_header(4, 4, 1, imgc.TYPE_CUBE, 2, 63)
        levels = [bytes([face * 16 + mip] * lv.size) for lv in imgc.level_layout(hd) for face, mip in [(lv.face, lv.mip)]]
        bm = imgc.join_payload(hd, levels)
        d = dds.imgc_to_dds(hd, bm)
        body = d[148:]
        # DDS is face-major: face0 mip0, face0 mip1, face1 mip0, ...
        self.assertEqual(body[:8], bytes([0x00] * 8))
        self.assertEqual(body[8:16], bytes([0x01] * 8))
        self.assertEqual(body[16:24], bytes([0x10] * 8))

    def _legacy(self, w, h, mips, fourcc=None, masks=None, bitcount=0, pf_flags=0):
        flags = 0x1007 | (0x20000 if mips > 1 else 0)
        hdr = struct.pack("<4s7I44s", b"DDS ", 124, flags, h, w, 0, 0, mips, bytes(44))
        if fourcc:
            hdr += struct.pack("<II4sIIIII", 32, 0x4, fourcc, 0, 0, 0, 0, 0)
        else:
            hdr += struct.pack("<II4sIIIII", 32, pf_flags, b"\0\0\0\0", bitcount, *masks)
        hdr += struct.pack("<5I", 0x1000, 0, 0, 0, 0)
        return hdr

    def test_legacy_fourcc_and_masks(self):
        hd = synth_header(8, 8, 1, imgc.TYPE_2D, 4, 59)
        body = b"".join(bytes(lv.size) for lv in imgc.level_layout(hd))
        d = dds.read_dds(self._legacy(8, 8, 4, fourcc=b"DXT1") + body)
        self.assertEqual((d.il_format, d.identified_by, d.data_offset, d.mip_count), (59, "fourcc", 128, 4))
        d = dds.read_dds(self._legacy(8, 8, 1, fourcc=b"ATI2") + bytes(64))
        self.assertEqual(d.il_format, 65)
        d = dds.read_dds(self._legacy(8, 8, 1, fourcc=b"BC5S") + bytes(64))
        self.assertEqual(d.il_format, 64)
        d = dds.read_dds(self._legacy(2, 2, 1, masks=(0x00FF0000, 0xFF00, 0xFF, 0xFF000000), bitcount=32, pf_flags=0x41) + bytes(16))
        self.assertEqual((d.il_format, d.identified_by), (32, "masks"))
        d = dds.read_dds(self._legacy(2, 2, 1, masks=(0xFF, 0xFF00, 0xFF0000, 0xFF000000), bitcount=32, pf_flags=0x41) + bytes(16))
        self.assertEqual(d.il_format, 38)
        d = dds.read_dds(self._legacy(2, 2, 1, masks=(0xFF, 0, 0, 0), bitcount=8, pf_flags=0x20000) + bytes(4))
        self.assertEqual(d.il_format, 0)
        d = dds.read_dds(self._legacy(2, 2, 1, fourcc=struct.pack("<I", 113)) + bytes(32))
        self.assertEqual(d.il_format, 46)

    def test_refusals(self):
        hd = synth_header(8, 8, 1, imgc.TYPE_2D, 4, 59)
        good = dds.imgc_to_dds(hd, synth_bitmap(hd))
        with self.assertRaises(ValidationError):
            dds.read_dds(good + b"\0")                       # wrong length
        bad = bytearray(good)
        struct.pack_into("<I", bad, 140, 2)                  # arraySize 2
        with self.assertRaises(UnsupportedError):
            dds.read_dds(bytes(bad))
        bad = bytearray(good)
        struct.pack_into("<I", bad, 128, 72)                 # BC1_UNORM_SRGB
        with self.assertRaises(UnsupportedError):
            dds.read_dds(bytes(bad))
        d = dds.read_dds(bytes(bad), srgb_to_linear=True)
        self.assertEqual(d.il_format, 59)
        d = dds.read_dds(bytes(bad), allow_unobserved=True)
        self.assertEqual(d.il_format, 70)
        bad = bytearray(good)
        struct.pack_into("<I", bad, 128, 9999)               # unknown DXGI
        with self.assertRaises(UnsupportedError):
            dds.read_dds(bytes(bad))
        with self.assertRaises(UnsupportedError):
            dds.read_dds(self._legacy(8, 8, 1, fourcc=b"DXT2") + bytes(32))
        with self.assertRaises(FormatError):
            dds.read_dds(b"DDX " + good[4:])
        hv = imgc.ImgcHeader()
        hv.set_geometry(4, 4, 1, imgc.TYPE_2D, 1, 35)         # BGRA8: no DXGI
        with self.assertRaises(UnsupportedError):
            dds.dds_header_for(hv)


class TestPng(unittest.TestCase):
    def test_bc4_decoder(self):
        # unsigned: a=255 > b=0 → 8-entry ramp; all indices 0 → 1.0 everywhere; indices 1 → 0.0
        block = bytes([255, 0]) + bytes(6)
        v = png.bc4_channels(block, 4, 4, 1, False)
        self.assertEqual(v.shape, (4, 4, 1))
        self.assertTrue(np.all(v == 1.0))
        block = bytes([255, 0]) + b"\x49\x92\x24" * 2         # index 1 for every texel (001 001 ...)
        v = png.bc4_channels(block, 4, 4, 1, False)
        self.assertTrue(np.all(v == 0.0))
        # signed: a=-128 → clamps to -1 ; b = 127 → 1 ; a<b → 6-entry ramp, code 6 = -1, code 7 = 1
        block = bytes([0x80, 0x7F]) + b"\xff" * 6              # index 7 everywhere
        v = png.bc4_channels(block, 4, 4, 1, True)
        self.assertTrue(np.all(v == 1.0))
        block = bytes([0x80, 0x7F]) + b"\xb6\x6d\xdb" * 2      # index 6 everywhere (110 110 ...)
        v = png.bc4_channels(block, 4, 4, 1, True)
        self.assertTrue(np.all(v == -1.0))
        # BC5: [R block][G block]
        two = bytes([255, 0]) + bytes(6) + bytes([0, 255]) + bytes(6)
        v = png.bc4_channels(two, 4, 4, 2, False)
        self.assertEqual(v.shape, (4, 4, 2))
        self.assertTrue(np.all(v[:, :, 0] == 1.0) and np.all(v[:, :, 1] == 0.0))
        # partial block crop
        v = png.bc4_channels(bytes([128, 0]) + bytes(6), 3, 2, 1, False)
        self.assertEqual(v.shape, (2, 3, 1))

    def test_uncompressed_decode_and_preview(self):
        raw = bytes([1, 2, 3, 4, 5, 6, 7, 8])                    # 2×1 ARGB8 → bytes B,G,R,A
        vals, dec = png.decode_level(raw, 2, 1, 32)
        self.assertEqual(dec, "uncompressed")
        self.assertEqual([round(x * 255) for x in vals[0, 0]], [3, 2, 1, 4])
        prev = png.to_preview_rgba8(vals, 32)
        self.assertEqual(list(prev[0, 0]), [3, 2, 1, 4])
        raw = struct.pack("<2b", -127, 127)                      # 1×1 RG8_SNORM
        vals, _ = png.decode_level(raw, 1, 1, 16)
        self.assertEqual(list(vals[0, 0]), [-1.0, 1.0])
        prev = png.to_preview_rgba8(vals, 16)
        self.assertEqual(list(prev[0, 0]), [0, 255, 0, 255])
        raw = struct.pack("<4e", 2.0, -1.0, 0.5, float("nan"))   # RGBA16F clipped
        prev = png.decode_preview(raw, 1, 1, 46)
        self.assertEqual(list(prev[0, 0]), [255, 0, 128, 0])
        prev = png.decode_preview(bytes(8), 2, 2, 59)             # BC1 via Pillow (preview only)
        self.assertEqual(prev.shape, (2, 2, 4))

    def test_png_to_imgc(self):
        from PIL import Image
        template = synth_header(4, 4, 1, imgc.TYPE_2D, 1, 38, flags=0x44)
        template.reserved = 0
        arr = np.zeros((6, 10, 4), np.uint8)
        arr[..., 0] = 200
        arr[..., 1] = 100
        arr[..., 3] = 255
        arr[0, 0] = (0, 0, 0, 0)
        h, bm, info = png.png_to_imgc(Image.fromarray(arr, "RGBA"), 38, template)
        self.assertEqual((h.width, h.height, h.mip_count, h.format, h.flags), (10, 6, 4, 38, 0x44))
        self.assertEqual(len(bm), imgc.payload_size(h))
        self.assertEqual(h.minimum, (0.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(h.maximum[0], 200 / 255, places=5)
        self.assertAlmostEqual(h.mean[3], 59 / 60, places=5)
        lv = imgc.split_payload(h, bm)
        self.assertEqual(bytes(lv[0][4:8]), bytes([200, 100, 0, 255]))
        self.assertEqual(len(lv[-1]), 4)
        # R8 and RG8_SNORM
        h, bm, _ = png.png_to_imgc(Image.fromarray(arr, "RGBA"), 0, template, mip_count=1)
        self.assertEqual((h.format, h.mip_count, len(bm)), (0, 1, 64))
        self.assertEqual(bytes(imgc.split_payload(h, bm)[0][:2]), bytes([0, 200]))
        nrm = np.full((4, 4, 4), 128, np.uint8)
        nrm[..., 2] = 255
        h, bm, _ = png.png_to_imgc(Image.fromarray(nrm, "RGBA"), 16, template)
        self.assertEqual((h.format, h.mip_count), (16, 3))
        x, y = struct.unpack_from("<2b", bytes(imgc.split_payload(h, bm)[0]), 0)
        self.assertTrue(abs(x) <= 1 and abs(y) <= 1)
        with self.assertRaises(UnsupportedError):
            png.png_to_imgc(Image.fromarray(arr, "RGBA"), 68, template)


@unittest.skipUnless(SAMPLE_PACK.exists() and (SAMPLE_RPX / "pack.json").exists(),
                     "out/samples/textures.rpack missing (tools/make_samples_texture.py)")
class TestSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pk = Pack.open(SAMPLE_PACK)
        cls.codec = codec_for(0x20)
        cls.spec = load_json(SAMPLE_RPX / "pack.json")
        cls.entries = {e["index"]: e for e in cls.spec["resources"]}

    @classmethod
    def tearDownClass(cls):
        cls.pk.close()

    def test_codec_registered(self):
        self.assertIsInstance(self.codec, TextureCodec)

    def test_roundtrip_every_sample(self):
        seen_types, seen_formats = set(), set()
        for res in self.pk.resources_of_type(0x20):
            produced = self.codec.roundtrip(res)
            for k, i in enumerate(res.part_indices):
                if k in produced:
                    self.assertEqual(bytes(self.pk.read_part(i)), produced[k], f"{res.name} part {k}")
            hk, _ = part_ordinals(res.part_types)
            h = imgc.parse_header(self.pk.read_part(res.part_indices[hk]))
            seen_types.add(h.type_name)
            seen_formats.add(h.format)
        self.assertEqual(seen_types, {"2D", "cube", "volume"})
        self.assertTrue(len(seen_formats) >= 15, seen_formats)

    def test_extract_files_and_build_identity(self):
        ctx = BuildContext(SAMPLE_RPX, self.spec, {"force_codec": True})
        n_header_only = 0
        for res in self.pk.resources_of_type(0x20):
            entry = self.entries[res.index]
            res_dir = SAMPLE_RPX / entry["dir"]
            ed = entry["editable"]
            self.assertEqual(ed["kind"], "dds")
            self.assertTrue((res_dir / ed["sidecar"]).exists())
            if ed["header_only"]:
                n_header_only += 1
                self.assertIsNone(ed["dds"])
            else:
                self.assertTrue((res_dir / ed["dds"]).exists())
            out = self.codec.build(ctx, entry, res_dir)
            for k, i in enumerate(res.part_indices):
                self.assertIn(k, out, f"{res.name}: part {k} not regenerated")
                self.assertEqual(bytes(self.pk.read_part(i)), bytes(out[k].data), f"{res.name} part {k}")
        self.assertEqual(n_header_only, 2)

    def _first(self, pred):
        for res in self.pk.resources_of_type(0x20):
            hk, _ = part_ordinals(res.part_types)
            h = imgc.parse_header(self.pk.read_part(res.part_indices[hk]))
            if pred(h):
                return res, h
        self.skipTest("no matching sample")

    def test_png_import_changes_geometry(self):
        from PIL import Image
        res, h = self._first(lambda h: h.format == 38 and h.tex_type == 0 and h.mip_count > 1)
        entry = self.entries[res.index]
        src_dir = SAMPLE_RPX / entry["dir"]
        work = OUT / "texture_png_import" / entry["dir"]
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(src_dir, work)
        ed = entry["editable"]
        sc = load_json(work / ed["sidecar"])
        # preview the original mip 0, shrink it to an odd size, import as PNG
        _, _, bmp_i = res, h, res.part_indices[part_ordinals(res.part_types)[1]]
        levels = imgc.split_payload(h, self.pk.read_part(bmp_i))
        rgba = png.decode_preview(bytes(levels[0]), h.width, h.height, 38)
        im = Image.fromarray(rgba, "RGBA").resize((max(1, h.width // 2 + 3), max(1, h.height // 2 + 1)))
        im.save(work / "edited.png")
        sc["source"] = "edited.png"
        dump_json(sc, work / ed["sidecar"])
        ctx = BuildContext(work.parent, self.spec, {})
        out = self.codec.build(ctx, entry, work)
        hk, bk = part_ordinals([int(p["type"], 0) for p in entry["parts"]])
        h2 = imgc.parse_header(bytes(out[hk].data))
        self.assertEqual((h2.width, h2.height), im.size)
        self.assertEqual(h2.mip_count, max(im.size).bit_length())
        self.assertEqual((h2.flags, h2.format, h2.mip_split, h2.reserved), (h.flags, 38, h.mip_split, h.reserved))
        self.assertNotEqual(h2.stats_raw, h.stats_raw)
        self.assertEqual(len(out[bk].data), imgc.payload_size(h2))
        self.assertTrue(any("geometry/format changed" in w for w in ctx.warnings))

    def test_dds_import_with_format_change(self):
        res, h = self._first(lambda h: h.format == 68 and h.tex_type == 0)
        entry = self.entries[res.index]
        work = OUT / "texture_dds_import" / entry["dir"]
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(SAMPLE_RPX / entry["dir"], work)
        ed = entry["editable"]
        # replace the DDS with a synthetic BC1 one of another size: allowed, header recomputed, flags kept
        hd = synth_header(24, 8, 1, imgc.TYPE_2D, 5, 59)
        (work / ed["dds"]).write_bytes(dds.imgc_to_dds(hd, synth_bitmap(hd)))
        ctx = BuildContext(work.parent, self.spec, {})
        out = self.codec.build(ctx, entry, work)
        hk, bk = part_ordinals([int(p["type"], 0) for p in entry["parts"]])
        h2 = imgc.parse_header(bytes(out[hk].data))
        self.assertEqual((h2.width, h2.height, h2.mip_count, h2.format, h2.flags), (24, 8, 5, 59, h.flags))
        self.assertEqual(h2.stats_raw, h.stats_raw)        # BC1: no exact decoder → kept, warned
        self.assertTrue(any("stats kept" in w for w in ctx.warnings))
        self.assertEqual(len(out[bk].data), imgc.payload_size(hd))

    def test_header_only_build_refuses_bitmap(self):
        res, h = self._first(lambda h: h.header_only)
        entry = self.entries[res.index]
        work = OUT / "texture_header_only" / entry["dir"]
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(SAMPLE_RPX / entry["dir"], work)
        ed = entry["editable"]
        sc = load_json(work / ed["sidecar"])
        (work / "x.dds").write_bytes(b"DDS ")
        sc["source"] = "x.dds"
        dump_json(sc, work / ed["sidecar"])
        from nightrunner.errors import BuildError
        with self.assertRaises(BuildError):
            self.codec.build(BuildContext(work.parent, self.spec, {}), entry, work)

    def test_stem(self):
        self.assertEqual(texture_stem("foo_dif.png"), "foo_dif")
        self.assertEqual(texture_stem("noise_msv.dds"), "noise_msv")
        self.assertEqual(texture_stem("plain"), "plain")
        self.assertEqual(texture_stem("a.b.png"), "a.b")


@unittest.skipUnless(have_game(), "game not installed")
class TestCorpus(unittest.TestCase):
    def test_common_textures_head_and_header_only(self):
        codec = codec_for(0x20)
        with Pack.open(ASSETS / "common_textures_0_pc.rpack") as pk:
            n = 0
            for res in pk.resources_of_type(0x20):
                if n >= 200:
                    break
                produced = codec.roundtrip(res)
                for k, i in enumerate(res.part_indices):
                    self.assertEqual(bytes(pk.read_part(i)), produced[k], res.name)
                n += 1
            hits = pk.find("bullet_trail_0000_fxd.dds", 0x20)   # by name: the index moves with every game patch
            self.assertEqual(len(hits), 1)
            res = pk.resource(hits[0])
            self.assertEqual(res.part_types, (0x20,))            # header-only: no 0x21 bitmap part
            produced = codec.roundtrip(res)
            self.assertEqual(list(produced), [0])
            self.assertEqual(bytes(pk.read_part(res.part_indices[0])), produced[0])

    def test_engine_cubes_and_volumes(self):
        codec = codec_for(0x20)
        with Pack.open(ASSETS / "engine_pc.rpack") as pk:
            kinds = set()
            for res in pk.resources_of_type(0x20):
                hk, _ = part_ordinals(res.part_types)
                h = imgc.parse_header(pk.read_part(res.part_indices[hk]))
                if h.tex_type == 0:
                    continue
                kinds.add(h.type_name)
                produced = codec.roundtrip(res)
                for k, i in enumerate(res.part_indices):
                    self.assertEqual(bytes(pk.read_part(i)), produced[k], res.name)
            self.assertEqual(kinds, {"cube", "volume"})


if __name__ == "__main__":
    unittest.main()
