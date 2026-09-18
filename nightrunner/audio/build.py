"""Writing AESP containers, and swapping a sound's audio.

The reader in `aesp.py` measured the format well enough to write it back: a header, a 152-byte-per-member file
table at `0xB8`, then the payloads tight and in table order. The member id is computable (`expected_id`), so
nothing has to be copied blind.

The gate on any of this is that an unchanged rebuild must come back **byte-identical**. That is the same bar the
RPACK writer is held to, and `nr audio rebuild` checks it.

What this does not do is install anything. It writes containers to an output folder; copying them into a game is
the user's step, with the game closed.

**None of this is confirmed in a running game.** See `notes/FORMATS/aesp.md` §9 — in particular E12, the 16-byte
`hash` chunk of a wem, whose algorithm is unidentified. A replacement wem built here carries no hash unless one
is copied from the sound it replaces.
"""
from __future__ import annotations

import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from ..errors import BuildError
from .aesp import ENTRY_SIZE, HEADER_MIN, MAGIC_OFFSET_COUNT, MAGIC_OFFSET_TABLE, NAME_SIZE, Aesp, expected_id
from .bnk import Bank
from .wem import PLUGIN_PCM, describe

_ENTRY_TAIL = struct.Struct("<IIQQ")


@dataclass
class Entry:
    """A member to write. `data` is its payload; `id` defaults to the rule the shipped tables follow."""
    name: str
    data: bytes
    id: int | None = None
    reserved: int = 0

    def member_id(self) -> int:
        return expected_id(self.name) if self.id is None else self.id


def build_container(entries: list[Entry], name: str, header: bytes | None = None) -> bytes:
    """An AESP holding *entries*, in the shipped shape.

    *header* supplies the first `0xB8` bytes when rebuilding an existing container, so the words this project
    does not understand (hole E1) are carried over rather than invented. Without it they are zeroed, which is
    what the other tooling does, and which is untested against the engine.
    """
    if header is not None and len(header) < HEADER_MIN:
        raise BuildError(f"header must be at least {HEADER_MIN} bytes, got {len(header)}")
    head = bytearray(header[:HEADER_MIN] if header is not None else bytes(HEADER_MIN))
    if header is None:
        struct.pack_into("<II", head, 0, 0x00020000, 0)
        raw = name.encode("utf-8")[:16]
        head[0x08:0x08 + len(raw)] = raw
    struct.pack_into("<I", head, MAGIC_OFFSET_COUNT, len(entries))
    struct.pack_into("<I", head, MAGIC_OFFSET_TABLE, HEADER_MIN)

    table = bytearray()
    payload = bytearray()
    pos = HEADER_MIN + len(entries) * ENTRY_SIZE
    for e in entries:
        raw = e.name.encode("utf-8")
        if len(raw) >= NAME_SIZE:
            raise BuildError(f"member name {e.name!r} does not fit in {NAME_SIZE} bytes")
        row = bytearray(ENTRY_SIZE)
        row[:len(raw)] = raw
        _ENTRY_TAIL.pack_into(row, NAME_SIZE, e.member_id(), e.reserved, pos, len(e.data))
        table += row
        payload += e.data
        pos += len(e.data)
    return bytes(head + table + payload)


def entries_of(container: Aesp) -> list[Entry]:
    """Every member of an open container, ready to write back."""
    return [Entry(m.name, bytes(container.read(m)), m.id, m.reserved) for m in container]


def rebuild(path: Path | str) -> bytes:
    """Read a container and write it straight back out. Should be byte-identical to the source."""
    with Aesp.open(path) as c:
        return build_container(entries_of(c), c.name, header=bytes(c.header_raw))


#: Rebuilding in memory needs about twice the file; above this a container is checked without doing that.
FULL_COMPARE_LIMIT = 512 * 1024 * 1024


def verify_rebuild(path: Path | str, full: bool | None = None) -> dict:
    """Rebuild a container and compare it with the original. Never raises on a mismatch; reports it.

    `full=True` builds the whole thing and compares every byte. For a multi-gigabyte container that needs twice
    its size in memory, so by default anything over `FULL_COMPARE_LIMIT` is checked the cheap way instead: the
    header and the whole file table are compared byte for byte, and the payload region is confirmed to be
    contiguous, in table order, and to end exactly at the end of the file.

    That is not a weaker claim than it looks. `build_container` writes the carried-over header, then the table,
    then each member's bytes back to back in table order. If the table it produces is identical and the original
    payloads sit exactly where that table says with nothing between them, the rebuilt file is the original.
    """
    path = Path(path)
    size = path.stat().st_size
    if full is None:
        full = size <= FULL_COMPARE_LIMIT
    if full:
        original = path.read_bytes()
        built = rebuild(path)
        out = {"path": str(path), "size": len(original), "rebuilt": len(built),
               "identical": built == original, "compared": "every byte"}
        if not out["identical"]:
            n = min(len(built), len(original))
            out["first_difference"] = next((i for i in range(n) if built[i] != original[i]), n)
        return out

    with Aesp.open(path) as c:
        table_end = c.table_offset + c.count * ENTRY_SIZE
        head_and_table = bytes(c.data[:table_end])
        entries = [Entry(m.name, b"", m.id, m.reserved) for m in c]
        sizes = [m.size for m in c]
        ours = bytearray(build_container([], c.name, header=bytes(c.header_raw))[:HEADER_MIN])
        struct.pack_into("<I", ours, MAGIC_OFFSET_COUNT, c.count)
        pos = table_end
        for e, n in zip(entries, sizes):
            raw = e.name.encode("utf-8")
            row = bytearray(ENTRY_SIZE)
            row[:len(raw)] = raw
            _ENTRY_TAIL.pack_into(row, NAME_SIZE, e.member_id(), e.reserved, pos, n)
            ours += row
            pos += n
        contiguous = all(m.offset == (table_end if i == 0 else c.members[i - 1].offset + c.members[i - 1].size)
                         for i, m in enumerate(c.members))
        out = {"path": str(path), "size": size, "rebuilt": pos, "compared": "header, table and layout",
               "table_identical": bytes(ours) == head_and_table,
               "payload_contiguous": contiguous, "ends_at_eof": pos == size}
    out["identical"] = out["table_identical"] and out["payload_contiguous"] and out["ends_at_eof"]
    if not out["table_identical"]:
        n = min(len(ours), len(head_and_table))
        out["first_difference"] = next((i for i in range(n) if ours[i] != head_and_table[i]), n)
    return out


# ---- replacing a sound ---------------------------------------------------------------------------------------

@dataclass
class Swap:
    """One sound to replace: the bank it lives in, its source id, and the new audio."""
    bank: str
    source_id: int
    wem: bytes
    keep_hash: bool = True          # carry the original's `hash` chunk over (E12: we cannot compute one)


def _chunks(data: bytes):
    pos = 12
    while pos + 8 <= len(data):
        tag = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if pos + 8 + size > len(data):
            break
        yield tag, pos, pos + 8, size
        pos += 8 + size + (size & 1)


def copy_hash_chunk(new_wem: bytes, old_wem: bytes) -> bytes:
    """Insert the old wem's `hash` chunk into the new one, after `fmt `.

    The algorithm behind that hash is unidentified (E12), so this carries the original's bytes rather than
    computing anything. Whether the engine cares - and whether a stale hash is worse than none - is untested.
    """
    old = next((old_wem[s:s + n] for tag, _, s, n in _chunks(old_wem) if tag == b"hash"), None)
    if old is None or any(tag == b"hash" for tag, _, _, _ in _chunks(new_wem)):
        return new_wem
    after_fmt = next((start + size for tag, _, start, size in _chunks(new_wem) if tag == b"fmt "), None)
    if after_fmt is None:
        return new_wem
    chunk = b"hash" + struct.pack("<I", len(old)) + old
    body = new_wem[12:after_fmt] + chunk + new_wem[after_fmt:]
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def patch_sound_plugin(bank_data: bytes, source_id: int, plugin: int = PLUGIN_PCM) -> tuple[bytes, int]:
    """Point every Sound with *source_id* at a different codec. Returns (bank, how many were patched).

    Four bytes at the start of the Sound body, so the bank keeps its exact size and layout. Needed because a
    Sound names its codec: handing a Vorbis-declared sound some PCM audio would have the engine decode it as
    Vorbis (E13 - reasoned from the object layout, not tested).
    """
    b = Bank(bank_data, "patch")
    out = bytearray(bank_data)
    n = 0
    for s in b.sounds:
        if s.source_id == source_id:
            struct.pack_into("<I", out, s.body_offset, plugin)
            n += 1
    return bytes(out), n


def apply_swaps(audio_dir: Path | str, out_dir: Path | str, swaps: list[Swap], *,
                patch_plugin: bool = True, progress=None) -> dict:
    """Write copies of the containers with *swaps* applied. The game folder is never touched.

    Only the containers that actually change are written: the one holding each replaced wem, and `meta.aesp`
    when a plugin id is patched.
    """
    audio_dir, out_dir = Path(audio_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"swaps": [], "written": {}, "warnings": []}

    from .resolve import SOURCE_BANK, SOURCE_MISSING, AudioIndex
    with AudioIndex(audio_dir) as idx:
        touched: dict[str, dict[int, bytes]] = {}
        bank_edits: dict[str, bytes] = {}
        for sw in swaps:
            rows = [r for r in idx.resolve_bank(sw.bank) if r.source_id == sw.source_id]
            if not rows:
                raise BuildError(f"{sw.bank}: no sound with source id {sw.source_id}")
            r = rows[0]
            if r.source == SOURCE_MISSING:
                raise BuildError(f"{sw.bank}: source {sw.source_id} is not in this install's containers")
            if r.source == SOURCE_BANK:
                raise BuildError(f"{sw.bank}: source {sw.source_id} is baked into the bank's DATA chunk; "
                                 "only container-resolved sounds can be swapped this way")
            payload = sw.wem
            if sw.keep_hash:
                payload = copy_hash_chunk(payload, idx.read_audio(r) or b"")
            touched.setdefault(r.source, {})[sw.source_id] = payload
            entry = {"bank": sw.bank, "source_id": sw.source_id, "container": f"{r.source}.aesp",
                     "old_bytes": r.size, "new_bytes": len(payload), "new": describe(payload)}
            if patch_plugin:
                base = bank_edits.get(sw.bank)
                if base is None:
                    b = idx.bank(sw.bank)
                    if b is None:
                        raise BuildError(f"no bank called {sw.bank!r}")
                    base = b.data
                base, n = patch_sound_plugin(base, sw.source_id)
                bank_edits[sw.bank] = base
                entry["plugin_patched"] = n
            report["swaps"].append(entry)

        for stem, replacements in touched.items():
            src = audio_dir / f"{stem}.aesp"
            if progress:
                progress(f"rebuilding {src.name}")
            with Aesp.open(src) as c:
                entries = entries_of(c)
                for sid, data in replacements.items():
                    hits = [e for e in entries if e.member_id() == sid]
                    if not hits:
                        raise BuildError(f"{src.name}: no member with id {sid}")
                    if len(hits) > 1:
                        # census 2026-09-18: sfx.aesp holds 26,517 members under 24,890 distinct names, so
                        # 1,560 ids appear more than once. Which one the engine resolves is not known (E14),
                        # so a duplicate is refused rather than silently resolved to one of them.
                        raise BuildError(f"{src.name}: {len(hits)} members share id {sid}; which one the engine "
                                         "uses is undecoded, so this swap is refused")
                    hits[0].data = data
                blob = build_container(entries, c.name, header=bytes(c.header_raw))
            dest = out_dir / src.name
            dest.write_bytes(blob)
            report["written"][src.name] = {"path": str(dest), "bytes": len(blob)}

        if bank_edits:
            if progress:
                progress("rebuilding meta.aesp")
            meta = audio_dir / "meta.aesp"
            with Aesp.open(meta) as c:
                entries = entries_of(c)
                by_name = {e.name.lower(): e for e in entries}
                for bank_name, data in bank_edits.items():
                    e = by_name.get(bank_name.lower())
                    if e is None:
                        raise BuildError(f"meta.aesp has no member {bank_name!r}")
                    e.data = data
                blob = build_container(entries, c.name, header=bytes(c.header_raw))
            dest = out_dir / meta.name
            dest.write_bytes(blob)
            report["written"][meta.name] = {"path": str(dest), "bytes": len(blob)}
    return report


def back_up(audio_dir: Path | str, names: list[str], backup_dir: Path | str) -> dict:
    """Copy the named containers aside before anything is installed. Never overwrites an existing backup."""
    audio_dir, backup_dir = Path(audio_dir), Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in names:
        src = audio_dir / name
        if not src.is_file():
            continue
        dest = backup_dir / name
        if dest.exists():
            out[name] = {"path": str(dest), "skipped": "a backup is already there"}
            continue
        shutil.copy2(src, dest)
        out[name] = {"path": str(dest), "bytes": dest.stat().st_size}
    return out
