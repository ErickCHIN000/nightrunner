"""AESP containers, Wwise soundbanks and the wwisepinhead registry (nightrunner/audio/).

The structural tests build their own containers and banks, so they run without the game. The corpus class at the
end re-measures the numbers `notes/FORMATS/aesp.md` claims, against the shipped files.
"""
from __future__ import annotations

import contextlib
import os
import struct
import tempfile
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nightrunner.audio import aesp, bnk, pinhead, preview, resolve  # noqa: E402
from nightrunner.errors import FormatError, UnsupportedError  # noqa: E402
from tests.paths import have_game  # noqa: E402
from tests.synth import tmpdir  # noqa: E402


@contextlib.contextmanager
def tmproot(prefix: str):
    """tests.synth.tmpdir yields a str; everything here wants a Path."""
    with tmpdir(prefix) as d:
        yield Path(d)

NL = chr(10)
AUDIO = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Dying Light The Beast\ph_ft\work\data\audio")


# ---- synthetic builders -------------------------------------------------------------------------------------

def make_aesp(members: list[tuple[str, bytes]], name: str = "test", ids: dict[str, int] | None = None) -> bytes:
    """A container in the shipped shape: header, table at 0xB8, payloads tight and in table order."""
    table = aesp.HEADER_MIN
    pos = table + len(members) * aesp.ENTRY_SIZE
    head = bytearray(table)
    struct.pack_into("<II", head, 0, 0x00020000, 0)
    head[0x08:0x08 + len(name)] = name.encode()
    struct.pack_into("<I", head, aesp.MAGIC_OFFSET_COUNT, len(members))
    struct.pack_into("<I", head, aesp.MAGIC_OFFSET_TABLE, table)
    rows, blobs = bytearray(), bytearray()
    for mname, data in members:
        row = bytearray(aesp.ENTRY_SIZE)
        row[:len(mname.encode())] = mname.encode()
        mid = (ids or {}).get(mname, aesp.expected_id(mname))
        struct.pack_into("<IIQQ", row, aesp.NAME_SIZE, mid, 0, pos, len(data))
        rows += row
        blobs += data
        pos += len(data)
    return bytes(head + rows + blobs)


def hirc_object(otype: int, oid: int, body: bytes = b"") -> bytes:
    payload = struct.pack("<I", oid) + body
    return bytes([otype]) + struct.pack("<I", len(payload)) + payload


def sound_object(oid: int, source_id: int, stream_type: int, plugin: int = 0x00040001, tail: bytes = b"") -> bytes:
    return hirc_object(bnk.OBJECT_SOUND, oid, struct.pack("<IBI", plugin, stream_type, source_id) + tail)


def make_bank(objects: list[bytes], version: int = 150, bank_id: int = 1234, extra: list[tuple[bytes, bytes]] = ()) -> bytes:
    bkhd = struct.pack("<II", version, bank_id) + b"\0" * 8
    out = bytearray(bnk.BKHD + struct.pack("<I", len(bkhd)) + bkhd)
    if objects:
        body = struct.pack("<I", len(objects)) + b"".join(objects)
        out += bnk.HIRC + struct.pack("<I", len(body)) + body
    for tag, body in extra:
        out += tag + struct.pack("<I", len(body)) + body
    return bytes(out)


# ---- ids ----------------------------------------------------------------------------------------------------

class WwiseIdTests(unittest.TestCase):
    def test_known_hashes(self):
        """Values read out of the shipped meta.aesp file table on 2026-09-17."""
        self.assertEqual(aesp.wwise_id("room_big_furnished"), 24721640)
        self.assertEqual(aesp.wwise_id("npc"), 662417162)
        self.assertEqual(aesp.wwise_id("weapons_pre"), 125528870)

    def test_case_is_folded_before_hashing(self):
        self.assertEqual(aesp.wwise_id("Room_Big_Furnished"), aesp.wwise_id("room_big_furnished"))
        self.assertEqual(aesp.wwise_id("ABC"), aesp.wwise_id("abc"))

    def test_expected_id_rule(self):
        self.assertEqual(aesp.expected_id("747664"), 747664)                   # numeric members are their own id
        self.assertEqual(aesp.expected_id("npc"), aesp.wwise_id("npc"))

    def test_hash_is_fnv1_not_fnv1a(self):
        """multiply then xor, which is what the shipped data follows."""
        h = 2166136261
        h = (h * 16777619) & 0xFFFFFFFF
        h ^= ord("a")
        self.assertEqual(aesp.wwise_id("a"), h)


# ---- container ----------------------------------------------------------------------------------------------

class AespTests(unittest.TestCase):
    def test_reads_members_in_table_order(self):
        blob = make_aesp([("747664", b"AAAA"), ("npc", b"BBBBBB")])
        a = aesp.Aesp(memoryview(blob), "t.aesp")
        self.assertEqual(a.name, "test")
        self.assertEqual([m.name for m in a], ["747664", "npc"])
        self.assertEqual(bytes(a.read(0)), b"AAAA")
        self.assertEqual(bytes(a.read("npc")), b"BBBBBB")

    def test_find_is_case_insensitive_and_read_accepts_a_name(self):
        a = aesp.Aesp(memoryview(make_aesp([("NPC", b"x")])), "t.aesp")
        self.assertIsNotNone(a.find("npc"))
        self.assertEqual(bytes(a.read("npc")), b"x")
        self.assertIsNone(a.find("nope"))
        with self.assertRaises(KeyError):
            a.read("nope")

    def test_ids_follow_the_documented_rule(self):
        a = aesp.Aesp(memoryview(make_aesp([("747664", b"a"), ("npc", b"b")])), "t.aesp")
        self.assertTrue(all(m.id_matches_name for m in a))
        self.assertEqual(a.layout()["ids_matching_name"], 2)

    def test_a_wrong_id_is_reported_not_refused(self):
        """An id that breaks the rule is interesting, not fatal: the container still reads."""
        blob = make_aesp([("npc", b"abc")], ids={"npc": 999})
        a = aesp.Aesp(memoryview(blob), "t.aesp")
        self.assertFalse(a.members[0].id_matches_name)
        self.assertEqual(a.layout()["ids_matching_name"], 0)
        self.assertEqual(bytes(a.read(0)), b"abc")

    def test_layout_reports_contiguity(self):
        a = aesp.Aesp(memoryview(make_aesp([("a", b"1234"), ("b", b"5678")])), "t.aesp")
        L = a.layout()
        self.assertEqual((L["contiguous"], L["gaps"], L["trailing_bytes"]), (2, 0, 0))
        self.assertEqual(L["payload_bytes"], 8)

    def test_a_gap_is_counted_not_refused(self):
        blob = bytearray(make_aesp([("a", b"1234"), ("b", b"5678")]))
        base = aesp.HEADER_MIN + aesp.ENTRY_SIZE + aesp.NAME_SIZE
        off = struct.unpack_from("<Q", blob, base + 8)[0]
        struct.pack_into("<Q", blob, base + 8, off + 2)       # push the second member along
        blob += b"\0\0"
        a = aesp.Aesp(memoryview(bytes(blob)), "t.aesp")
        self.assertEqual(a.layout()["gaps"], 1)

    def test_too_short_to_hold_a_header(self):
        with self.assertRaises(FormatError):
            aesp.Aesp(memoryview(b"\0" * 32), "t.aesp")

    def test_table_past_the_end_is_refused(self):
        blob = bytearray(make_aesp([("a", b"1")]))
        struct.pack_into("<I", blob, aesp.MAGIC_OFFSET_COUNT, 100000)
        with self.assertRaises(FormatError):
            aesp.Aesp(memoryview(bytes(blob)), "t.aesp")

    def test_member_payload_past_the_end_is_refused(self):
        blob = bytearray(make_aesp([("a", b"1234")]))
        struct.pack_into("<Q", blob, aesp.HEADER_MIN + aesp.NAME_SIZE + 16, 1 << 40)
        a = aesp.Aesp(memoryview(bytes(blob)), "t.aesp")
        with self.assertRaises(FormatError):
            _ = a.members

    def test_open_and_close_a_real_file(self):
        with tmpdir("aesp") as d:
            p = Path(d) / "x.aesp"
            p.write_bytes(make_aesp([("a", b"hello")]))
            with aesp.Aesp.open(p) as a:
                self.assertEqual(bytes(a.read("a")), b"hello")
            self.assertEqual(a._mm, None)                      # the mapping is released


# ---- soundbank ----------------------------------------------------------------------------------------------

class BankTests(unittest.TestCase):
    def test_header_and_chunks(self):
        b = bnk.Bank(make_bank([], version=150, bank_id=77), "t.bnk")
        self.assertEqual((b.version, b.bank_id), (150, 77))
        self.assertEqual([c.tag for c in b.chunks], ["BKHD"])
        self.assertEqual(b.trailing, 0)

    def test_hirc_walk_and_sound_fields(self):
        b = bnk.Bank(make_bank([sound_object(11, 4242, bnk.STREAM_STREAMED),
                                hirc_object(3, 12),
                                sound_object(13, 99, bnk.STREAM_EMBEDDED)]), "t.bnk")
        self.assertTrue(b.hirc_exact)
        self.assertEqual(len(b.objects), 3)
        self.assertEqual(b.type_histogram(), {2: 2, 3: 1})
        self.assertEqual([s.source_id for s in b.sounds], [4242, 99])
        self.assertEqual(b.stream_histogram(), {"streamed": 1, "embedded": 1})
        self.assertEqual(b.source_ids(), {4242, 99})
        self.assertEqual(b.sounds[0].plugin_name, "vorbis")

    def test_stream_type_offset_points_at_the_byte(self):
        """The one byte UTM-AIO patches. If this is wrong, a future writer corrupts a bank."""
        blob = make_bank([sound_object(11, 4242, bnk.STREAM_EMBEDDED)])
        b = bnk.Bank(blob, "t.bnk")
        s = b.sounds[0]
        self.assertEqual(blob[s.stream_type_offset], bnk.STREAM_EMBEDDED)
        patched = bytearray(blob)
        patched[s.stream_type_offset] = bnk.STREAM_STREAMED
        again = bnk.Bank(bytes(patched), "t.bnk").sounds[0]
        self.assertEqual(again.stream_type, bnk.STREAM_STREAMED)
        self.assertEqual(again.source_id, 4242)                # nothing else moved
        self.assertEqual(len(patched), len(blob))              # and the bank kept its size

    def test_unknown_object_types_are_kept_numeric(self):
        b = bnk.Bank(make_bank([hirc_object(99, 5)]), "t.bnk")
        self.assertEqual(b.objects[0].type_name, "type_99")
        self.assertEqual(b.sounds, [])

    def test_sound_tail_is_preserved_raw(self):
        b = bnk.Bank(make_bank([sound_object(11, 1, 0, tail=b"\xde\xad\xbe\xef")]), "t.bnk")
        self.assertEqual(b.sounds[0].body_raw, b"\xde\xad\xbe\xef")

    def test_other_chunks_are_exposed(self):
        b = bnk.Bank(make_bank([], extra=[(bnk.DIDX, b"\1" * 12), (bnk.DATA, b"\2" * 4)]), "t.bnk")
        self.assertEqual([c.tag for c in b.chunks], ["BKHD", "DIDX", "DATA"])
        self.assertEqual(b.chunk_data(bnk.DIDX), b"\1" * 12)

    def test_a_chunk_past_the_end_is_refused(self):
        blob = bytearray(make_bank([]))
        struct.pack_into("<I", blob, 4, 1 << 20)
        with self.assertRaises(FormatError):
            bnk.Bank(bytes(blob), "t.bnk")

    def test_no_bkhd_is_refused(self):
        with self.assertRaises(FormatError):
            bnk.Bank(b"JUNK" + struct.pack("<I", 0), "t.bnk")

    def test_is_bank(self):
        self.assertTrue(bnk.is_bank(make_bank([])))
        self.assertFalse(bnk.is_bank(b"RIFF1234"))


# ---- registry -----------------------------------------------------------------------------------------------

REGISTRY_XML = """<?xml version="1.0"?>
<Mapping Version="768">
    <Preloads>
        <Preload name="Npc" id="111" managed="true"><FileData name="Npc.bnk" /></Preload>
        <Preload name="Music" id="222" localized="true"><FileData name="Music.bnk" /></Preload>
    </Preloads>
    <Events>
        <Event name="wpn_pistol_reload" id="333" duration="0.5"><WwiseEvent name="wpn_pistol_reload" /></Event>
    </Events>
    <AuxBuses>
        <AuxBus name="Hud_FX" id="444"><WwiseAuxBus name="Hud_FX" /></AuxBus>
    </AuxBuses>
</Mapping>
"""


class PinheadTests(unittest.TestCase):
    def setUp(self):
        self.ph = pinhead.Pinhead(REGISTRY_XML)

    def test_version_and_preloads(self):
        self.assertEqual(self.ph.version, "768")
        self.assertEqual([p.name for p in self.ph.preloads], ["Npc", "Music"])
        self.assertTrue(self.ph.preloads[0].managed)
        self.assertTrue(self.ph.preloads[1].localized)
        self.assertEqual(self.ph.preloads[0].files, ["Npc.bnk"])

    def test_shipped_preloads_carry_no_whitelist(self):
        """0 of 128 in the shipped registry; the <File> list is UTM-AIO's own addition."""
        self.assertEqual([p.whitelist for p in self.ph.preloads], [[], []])
        self.assertEqual(self.ph.to_json()["preloads_with_whitelist"], 0)

    def test_id_to_name(self):
        self.assertEqual(self.ph.name_of(333), "wpn_pistol_reload")
        self.assertEqual(self.ph.name_of(444), "Hud_FX")
        self.assertIsNone(self.ph.name_of(999))

    def test_by_name_is_case_insensitive(self):
        self.assertEqual(self.ph.by_name("WPN_PISTOL_RELOAD").id, 333)

    def test_only_id_carrying_elements_are_indexed(self):
        """<WwiseEvent> carries a name but no id: it is the Wwise-side alias, not an object of its own."""
        self.assertEqual(self.ph.kind_histogram(), {"Event": 1, "AuxBus": 1, "Preload": 2})
        self.assertNotIn("WwiseEvent", self.ph.kind_histogram())

    def test_extra_attributes_are_kept(self):
        self.assertEqual(self.ph.of_kind("Event")[0].attrs["duration"], "0.5")

    def test_names_table(self):
        self.assertEqual(self.ph.names()[333], "wpn_pistol_reload")

    def test_from_container(self):
        blob = make_aesp([(pinhead.MEMBER_NAME, REGISTRY_XML.encode())], name="meta")
        a = aesp.Aesp(memoryview(blob), Path("meta.aesp"))
        self.assertEqual(pinhead.from_container(a).version, "768")
        empty = aesp.Aesp(memoryview(make_aesp([("npc", b"x")])), Path("t.aesp"))
        self.assertIsNone(pinhead.from_container(empty))


# ---- the shipped data ---------------------------------------------------------------------------------------

@unittest.skipUnless(have_game() and AUDIO.is_dir(), "game audio not installed")
class CorpusTests(unittest.TestCase):
    """Re-measures what notes/FORMATS/aesp.md claims. These numbers are the provenance for that file."""

    def test_every_container_reads_and_is_contiguous(self):
        totals = {}
        for stem in aesp.CONTAINERS:
            p = AUDIO / f"{stem}.aesp"
            if not p.is_file():
                continue
            with aesp.Aesp.open(p) as a:
                L = a.layout()
                self.assertEqual(L["gaps"], 0, stem)
                self.assertEqual(L["contiguous"], L["members"], stem)
                self.assertEqual(L["trailing_bytes"], 0, stem)
                # the u32 at +0x84 is zero everywhere except the registry member: 30,171/30,172
                self.assertEqual(L["reserved_nonzero"], 1 if stem == "meta" else 0, stem)
                totals[stem] = L["members"]
        self.assertEqual(totals.get("sfx"), 26517)
        self.assertEqual(totals.get("streams"), 3530)

    def test_only_the_registry_member_carries_a_reserved_value(self):
        """The one entry that breaks both the id rule and the zero-reserved rule, so the pair is worth a look."""
        odd = []
        for stem in aesp.CONTAINERS:
            p = AUDIO / f"{stem}.aesp"
            if not p.is_file():
                continue
            with aesp.Aesp.open(p) as a:
                odd += [(m.name, m.reserved) for m in a if m.reserved]
        self.assertEqual(odd, [(pinhead.MEMBER_NAME, 0x25019676)])

    def test_ids_follow_the_rule_everywhere_but_the_registry(self):
        """30,170/30,172 - the two misses are the registry member and nothing else."""
        misses = []
        for stem in aesp.CONTAINERS:
            p = AUDIO / f"{stem}.aesp"
            if not p.is_file():
                continue
            with aesp.Aesp.open(p) as a:
                misses += [m.name for m in a if not m.id_matches_name]
        self.assertEqual(misses, [pinhead.MEMBER_NAME])

    def test_every_bank_parses_on_one_version(self):
        with aesp.Aesp.open(AUDIO / "meta.aesp") as a:
            versions, exact, banks = set(), 0, 0
            for m in a:
                data = a.read(m)
                if not bnk.is_bank(data):
                    continue
                b = bnk.Bank(data, m.name)
                banks += 1
                versions.add(b.version)
                self.assertEqual(b.trailing, 0, m.name)
                if b.hirc_exact:
                    exact += 1
            self.assertEqual(banks, 123)
            self.assertEqual(versions, {150})
            self.assertEqual(exact, 121)

    def test_registry_names_the_events(self):
        with aesp.Aesp.open(AUDIO / "meta.aesp") as a:
            ph = pinhead.from_container(a)
        self.assertEqual(ph.version, "768")
        self.assertEqual(len(ph.preloads), 128)
        self.assertEqual(ph.kind_histogram()["Event"], 23904)
        self.assertEqual(sum(1 for p in ph.preloads if p.whitelist), 0)

    def test_a_streamed_sound_resolves_to_a_stream_member(self):
        """The link the stream-type technique depends on: a streamed source id is a member of streams.aesp."""
        with aesp.Aesp.open(AUDIO / "meta.aesp") as meta, aesp.Aesp.open(AUDIO / "streams.aesp") as st:
            stream_ids = {m.id for m in st}
            found = checked = 0
            for m in meta:
                data = meta.read(m)
                if not bnk.is_bank(data):
                    continue
                for s in bnk.Bank(data, m.name).sounds:
                    if s.stream_type == bnk.STREAM_STREAMED:
                        checked += 1
                        found += s.source_id in stream_ids
                if checked >= 50:
                    break
        self.assertGreater(checked, 0)
        self.assertEqual(found, checked, "every streamed source id should name a streams.aesp member")


if __name__ == "__main__":
    unittest.main()


# ---- the join -----------------------------------------------------------------------------------------------

class ResolveTests(unittest.TestCase):
    """AudioIndex joins banks to containers. Built on disk because AudioIndex opens files by name."""

    def build(self, d: Path, *, bank_objects, didx=(), sfx=(), streams=(), registry: str | None = None):
        extra = []
        if didx:
            idx, data = bytearray(), bytearray()
            for wid, blob in didx:
                idx += struct.pack("<III", wid, len(data), len(blob))
                data += blob
            extra = [(bnk.DIDX, bytes(idx)), (bnk.DATA, bytes(data))]
        members = [("mybank", make_bank(bank_objects, extra=extra))]
        if registry:
            members.append((pinhead.MEMBER_NAME, registry.encode()))
        (d / "meta.aesp").write_bytes(make_aesp(members, name="meta"))
        (d / "sfx.aesp").write_bytes(make_aesp([(str(i), b) for i, b in sfx], name="sfx"))
        (d / "streams.aesp").write_bytes(make_aesp([(str(i), b) for i, b in streams], name="streams"))
        return resolve.AudioIndex(d)

    def test_resolves_each_sound_to_its_store(self):
        with tmproot("resolve") as d:
            with self.build(d,
                            bank_objects=[sound_object(1, 100, bnk.STREAM_EMBEDDED),
                                          sound_object(2, 200, bnk.STREAM_STREAMED),
                                          sound_object(3, 300, bnk.STREAM_EMBEDDED),
                                          sound_object(4, 999, bnk.STREAM_EMBEDDED)],
                            didx=[(300, b"BAKED")],
                            sfx=[(100, b"SFXDATA")],
                            streams=[(200, b"STREAMDATA")]) as idx:
                got = {r.source_id: r.source for r in idx.resolve_bank("mybank")}
        self.assertEqual(got, {100: resolve.SOURCE_SFX, 200: resolve.SOURCE_STREAMS,
                               300: resolve.SOURCE_BANK, 999: resolve.SOURCE_MISSING})

    def test_reads_audio_from_wherever_it_lives(self):
        with tmproot("resolve_read") as d:
            with self.build(d,
                            bank_objects=[sound_object(1, 100, 0), sound_object(2, 200, 2), sound_object(3, 300, 0),
                                          sound_object(4, 999, 0)],
                            didx=[(300, b"BAKED")], sfx=[(100, b"SFXDATA")], streams=[(200, b"STREAMDATA")]) as idx:
                by_id = {r.source_id: r for r in idx.resolve_bank("mybank")}
                self.assertEqual(idx.read_audio(by_id[100]), b"SFXDATA")
                self.assertEqual(idx.read_audio(by_id[200]), b"STREAMDATA")
                self.assertEqual(idx.read_audio(by_id[300]), b"BAKED")
                self.assertIsNone(idx.read_audio(by_id[999]))

    def test_in_bank_flag_marks_what_needs_the_stream_type_flip(self):
        with tmproot("resolve_flag") as d:
            with self.build(d, bank_objects=[sound_object(1, 300, 0), sound_object(2, 100, 0)],
                            didx=[(300, b"X")], sfx=[(100, b"Y")]) as idx:
                rows = {r.source_id: r for r in idx.resolve_bank("mybank")}
        self.assertTrue(rows[300].in_bank)
        self.assertFalse(rows[100].in_bank)

    def test_didx_ids(self):
        b = bnk.Bank(make_bank([], extra=[(bnk.DIDX, struct.pack("<III", 7, 0, 3)), (bnk.DATA, b"abc")]), "b")
        self.assertEqual(resolve.didx_ids(b), {7: (0, 3)})
        self.assertEqual(resolve.didx_ids(bnk.Bank(make_bank([]), "b")), {})

    def test_event_names_come_from_the_name_hash(self):
        """The registry's <Event id> is Techland's; the Wwise object id is the hash of the name."""
        ev_name = "menu_back"
        ev_id = aesp.wwise_id(ev_name)
        action_id = 5555
        sound_id = 42
        event_body = struct.pack("<I", ev_id) + bytes([1]) + struct.pack("<I", action_id)
        event_obj = bytes([resolve.OBJECT_EVENT]) + struct.pack("<I", len(event_body)) + event_body
        action_body = struct.pack("<IHI", action_id, 0x0403, sound_id)
        action_obj = bytes([resolve.OBJECT_ACTION]) + struct.pack("<I", len(action_body)) + action_body
        registry = ('<?xml version="1.0"?><Mapping Version="768"><Preloads>'
                    '<Preload name="mybank" id="1"><FileData name="mybank.bnk" /></Preload></Preloads>'
                    f'<Events><Event name="{ev_name}" id="777" preload_id="1" /></Events></Mapping>')
        with tmproot("resolve_ev") as d:
            with self.build(d, bank_objects=[sound_object(sound_id, 100, 0), event_obj, action_obj],
                            sfx=[(100, b"A")], registry=registry) as idx:
                names = idx.event_names_by_sound("mybank")
        self.assertEqual(names, {sound_id: [ev_name]})

    def test_an_action_pointing_at_a_container_is_not_credited_to_a_sound(self):
        """3,198 of 14,430 actions target a Sound directly; the rest must not be attached to a guess."""
        ev_id = aesp.wwise_id("somewhere_else")
        event_body = struct.pack("<I", ev_id) + bytes([1]) + struct.pack("<I", 5555)
        event_obj = bytes([resolve.OBJECT_EVENT]) + struct.pack("<I", len(event_body)) + event_body
        action_body = struct.pack("<IHI", 5555, 0x0403, 987654)        # target is not a Sound in this bank
        action_obj = bytes([resolve.OBJECT_ACTION]) + struct.pack("<I", len(action_body)) + action_body
        registry = ('<?xml version="1.0"?><Mapping Version="768"><Preloads>'
                    '<Preload name="mybank" id="1"><FileData name="mybank.bnk" /></Preload></Preloads>'
                    '<Events><Event name="somewhere_else" id="777" preload_id="1" /></Events></Mapping>')
        with tmproot("resolve_ev2") as d:
            with self.build(d, bank_objects=[sound_object(42, 100, 0), event_obj, action_obj],
                            sfx=[(100, b"A")], registry=registry) as idx:
                self.assertEqual(idx.event_names_by_sound("mybank"), {})


@unittest.skipUnless(have_game() and AUDIO.is_dir(), "game audio not installed")
class ResolveCorpusTests(unittest.TestCase):
    def test_menu_bank_is_fully_resolved_and_named(self):
        """The main-menu case: no audio baked into the bank, every sound found, most named by an event."""
        with resolve.AudioIndex(AUDIO) as idx:
            rows = idx.resolve_bank("menu")
            names = idx.event_names_by_sound("menu")
        self.assertEqual(len(rows), 33)
        self.assertTrue(all(r.source == resolve.SOURCE_SFX for r in rows))
        self.assertTrue(all(r.found for r in rows))
        self.assertGreaterEqual(len(names), 25)
        self.assertIn("menu_crafting", {n for v in names.values() for n in v})

    def test_every_event_of_a_bank_resolves_by_name_hash(self):
        """10,833/10,833 corpus-wide; checked here on the menu preload's 39 events."""
        with resolve.AudioIndex(AUDIO) as idx:
            b = idx.bank("menu")
            by_id = {o.id for o in b.objects}
            evs = idx.events_for_preload(1962405007)
            self.assertEqual(len(evs), 39)
            self.assertTrue(all(aesp.wwise_id(e.name) in by_id for e in evs))


# ---- preview ------------------------------------------------------------------------------------------------

def riff(codec: int = 0xFFFF, channels: int = 2, rate: int = 44100, data: bytes = b"\0" * 16) -> bytes:
    fmt = struct.pack("<HHIIHH", codec, channels, rate, rate * 4, 4, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


class WemInfoTests(unittest.TestCase):
    def test_reads_the_fmt_chunk(self):
        info = preview.read_info(riff())
        self.assertEqual((info.codec, info.channels, info.sample_rate), (0xFFFF, 2, 44100))
        self.assertTrue(info.is_wwise_vorbis)
        self.assertEqual(info.data_bytes, 16)

    def test_every_shipped_wem_is_wwise_vorbis(self):
        """census 2026-09-17: 302/302 sfx and 321/321 streams members sampled carry codec 0xFFFF."""
        self.assertEqual(preview.WWISE_VORBIS, 0xFFFF)
        self.assertTrue(preview.read_info(riff(0xFFFF)).is_wwise_vorbis)
        self.assertFalse(preview.read_info(riff(0x0001)).is_wwise_vorbis)

    def test_not_riff(self):
        self.assertIsNone(preview.read_info(b"BKHD\0\0\0\0"))
        self.assertIsNone(preview.read_info(b""))

    def test_riff_without_a_fmt_chunk(self):
        body = b"WAVE" + b"data" + struct.pack("<I", 4) + b"abcd"
        self.assertIsNone(preview.read_info(b"RIFF" + struct.pack("<I", len(body)) + body))


class DecoderTests(unittest.TestCase):
    """The decoder is external, so these drive a stub executable rather than needing vgmstream installed."""

    def stub(self, d: Path, *, ok: bool = True) -> Path:
        """A tiny Python 'decoder' invoked as [exe, -o, out, in]."""
        script = d / "stub_decoder.py"
        script.write_text(NL.join([
            "import sys, pathlib",
            "out = sys.argv[sys.argv.index('-o') + 1]",
            f"ok = {ok!r}",
            "if not ok:",
            "    print('stub refused', file=sys.stderr); sys.exit(3)",
            "src = pathlib.Path(sys.argv[-1]).read_bytes()",
            "pathlib.Path(out).write_bytes(b'RIFFdecoded' + src[:4])",
        ]), encoding="utf-8")
        exe = d / "run_decoder.cmd"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
        return exe

    def test_decodes_through_the_external_tool(self):
        with tmproot("wem_decode") as d:
            out = preview.decode_to_wav(riff(), d / "out.wav", decoder=self.stub(d))
            self.assertTrue(out.is_file())
            self.assertTrue(out.read_bytes().startswith(b"RIFFdecoded"))

    def test_a_failing_decoder_reports_its_own_message(self):
        with tmproot("wem_fail") as d:
            with self.assertRaises(UnsupportedError) as cm:
                preview.decode_to_wav(riff(), d / "out.wav", decoder=self.stub(d, ok=False))
        self.assertIn("refused", str(cm.exception))

    def test_a_missing_decoder_names_what_is_needed(self):
        with tmproot("wem_missing") as d:
            with self.assertRaises(UnsupportedError) as cm:
                preview.decode_to_wav(riff(), d / "out.wav", decoder=d / "nope.exe")
        self.assertIn("could not run", str(cm.exception).casefold())

    def test_no_decoder_at_all_refuses_with_the_hint(self):
        old = os.environ.pop(preview.ENV_VAR, None)
        real_which = preview.shutil.which
        preview.shutil.which = lambda name: None
        try:
            self.assertIsNone(preview.find_decoder())
            with tmproot("wem_none") as d:
                with self.assertRaises(UnsupportedError) as cm:
                    preview.decode_to_wav(riff(), d / "out.wav")
            self.assertIn("vgmstream", str(cm.exception))
            self.assertIn(preview.ENV_VAR, str(cm.exception))
        finally:
            preview.shutil.which = real_which
            if old is not None:
                os.environ[preview.ENV_VAR] = old

    def test_env_var_wins(self):
        with tmproot("wem_env") as d:
            exe = self.stub(d)
            os.environ[preview.ENV_VAR] = str(exe)
            try:
                self.assertEqual(preview.find_decoder(), exe)
                self.assertTrue(preview.available())
            finally:
                os.environ.pop(preview.ENV_VAR, None)

    def test_the_temp_wem_is_cleaned_up(self):
        before = set(Path(tempfile.gettempdir()).glob("*.wem"))
        with tmproot("wem_tmp") as d:
            preview.decode_to_wav(riff(), d / "out.wav", decoder=self.stub(d))
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("*.wem")) - before, set())

    def test_the_old_generic_name_is_not_searched_for_on_path(self):
        """Git for Windows ships usr/bin/test.exe. Finding that would make Play look available and then fail."""
        self.assertIn("test.exe", preview.EXE_NAMES)
        self.assertNotIn("test.exe", preview.PATH_NAMES)
        seen = []
        real = preview.shutil.which
        preview.shutil.which = lambda name: seen.append(name) or None
        old = os.environ.pop(preview.ENV_VAR, None)
        try:
            preview.find_decoder()
        finally:
            preview.shutil.which = real
            if old is not None:
                os.environ[preview.ENV_VAR] = old
        self.assertEqual(seen, list(preview.PATH_NAMES))
