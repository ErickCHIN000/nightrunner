"""Offline replay of the engine's container contracts.

Each check mirrors a specific native routine (survey 01):

* structure   — OpenPack +0x26320: magic/version/table bounds; name table integrity.
* owner       — GetPhysicalResourceParentName +0x1EB00: physical.packed>>16 must be the owning logical index.
* parts       — runtime logical word packs part_count into 4 bits (+0x247F0): 1..15 parts.
* storage     — physical storage index in range; storage versions match the native catalogue.
* bounds      — every direct part lies inside the file, after the tables.
* alignment   — every part offset is a multiple of its storage alignment (1 << ((w>>9)&15)).
* ondemand    — for header.field08 bit 12 packs, replay OnDemandScheduleResourcePackCompatible (+0xCEB530) and
                OnDemandPreRegisterOnDemandReady_Mesh (+0xCEAA60) for every type-0x10 resource: all parts
                method 1, no part with physical 0x1000, and every slice offset computed from one read at the
                first part must equal the part's own file offset.
* mesh_name   — MeshMgr registers CCompactMesh::GetFileName = the string embedded in the ClassReader image
                (ResourceManagement +0x17A60); it must equal "<logical name>.msh" (checked when the
                classreader package is available).
* unknown     — census of bits with no established semantics (never a failure): physical bits 12/14/15,
                storage flags bit 2/3, align_raw low bit and bits 5..7, logical flags byte, physical.fc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..util.binio import align_up
from . import catalogue
from .rp6l import Pack, PHYS_SPECIAL, PHYS_BIT14, PHYS_BIT15, FIELD08_ONDEMAND, MAX_PARTS


@dataclass
class Finding:
    level: str          # 'error' | 'warning' | 'info'
    check: str
    message: str
    resource: int | None = None
    part: int | None = None

    def to_json(self) -> dict:
        d = {"level": self.level, "check": self.check, "message": self.message}
        if self.resource is not None:
            d["resource"] = self.resource
        if self.part is not None:
            d["part"] = self.part
        return d


@dataclass
class Report:
    path: str
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, level: str, check: str, message: str, resource: int | None = None, part: int | None = None) -> None:
        self.findings.append(Finding(level, check, message, resource, part))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json(self) -> dict:
        return {"path": self.path, "ok": self.ok, "stats": self.stats,
                "findings": [f.to_json() for f in self.findings]}


def ondemand_slices(pack: Pack, logical_index: int) -> tuple[list[int], int] | None:
    """Replay +0xCEB530/+0xCEAA60 for one resource: returns (expected offsets per part, total read size)
    or None if the resource is not on-demand compatible (method != 1 or physical 0x1000 on any part)."""
    r = pack.resource(logical_index)
    idxs = list(r.part_indices)
    if not idxs:
        return None
    total = 0
    offsets = []
    base = pack.part_offset(idxs[0])
    for i in idxs:
        p = pack.physicals[i]
        s = pack.storages[p.storage_index]
        if p.packed & PHYS_SPECIAL or s.method != 1:
            return None
        a = s.alignment
        total = align_up(total, a)          # ((total - 1 + A) & -A) for total>0; identical for total == 0
        offsets.append(base + total)
        total += p.size
    return offsets, total


def validate(pack: Pack, *, check_mesh_names: bool = True, max_findings: int = 2000) -> Report:
    rep = Report(str(pack.path))
    h = pack.header
    S, P, L = h.storage_count, h.physical_count, h.logical_count
    rep.stats.update(storages=S, physicals=P, logicals=L, field08=f"0x{h.field08:X}", flags=f"0x{h.flags:X}",
                     ondemand=bool(h.field08 & FIELD08_ONDEMAND), types=pack.type_histogram())

    # The cap only limits how many findings are *stored*; errors are always counted so `Report.ok` can never be
    # true for a pack that violated a contract after the cap was hit (review 2026-09-15, F1).
    dropped = {"error": 0, "warning": 0, "info": 0}

    def add(level, *a, **k):
        if level == "error" or len(rep.findings) < max_findings:
            rep.add(level, *a, **k)
        else:
            dropped[level] += 1

    # structure ---------------------------------------------------------------------------------------------
    if h.name_count != h.logical_count:
        add("warning", "structure", f"name_count {h.name_count} != logical_count {h.logical_count} (47/47 shipped packs are equal)")
    for ni, off in enumerate(pack.name_offsets):
        if off >= len(pack.name_blob):
            add("error", "structure", f"name offset {off} outside blob ({len(pack.name_blob)})")
        elif pack.name_blob.find(b"\0", off) < 0:
            add("error", "structure", f"name {ni} not NUL-terminated")
    if h.flags != 1:
        add("info", "structure", f"header.flags = 0x{h.flags:X} (all shipped packs use 1)")
    if h.field08 & ~FIELD08_ONDEMAND:
        add("info", "structure", f"header.field08 = 0x{h.field08:X} has bits other than 0x1000 (never seen shipped)")

    # storage -----------------------------------------------------------------------------------------------
    for si, s in enumerate(pack.storages):
        v = catalogue.type_version(s.type)
        if v is None:
            add("error", "storage", f"storage {si}: unknown type 0x{s.type:02X}")
        elif v != s.version:
            add("error", "storage", f"storage {si}: type 0x{s.type:02X} version {s.version} != catalogue {v}")
        if s.method >= 2:
            add("warning", "storage", f"storage {si}: compressed method {s.method} (codec {s.codec}) — unsupported by this tool, never seen shipped")
        if s.codec:
            add("info", "unknown", f"storage {si}: codec nibble {s.codec}")
        if s.flag_bit2:
            add("info", "unknown", f"storage {si}: flags bit 2 set (never seen shipped)")
        if s.align_raw & 0x1 or s.align_raw & 0xE0:
            add("info", "unknown", f"storage {si}: align_raw 0x{s.align_raw:02X} has bits outside the alignment field")
        if s.compressed:
            add("info", "unknown", f"storage {si}: compressed size {s.compressed} (shipped packs: 0)")

    # group size/count bookkeeping (descriptive; not a known engine check)
    gsize = [0] * S
    gcount = [0] * S
    for p in pack.physicals:
        s = pack.storages[p.storage_index]
        gsize[p.storage_index] += align_up(p.size, max(16, s.alignment))
        gcount[p.storage_index] += 1
    for si, s in enumerate(pack.storages):
        if s.count != (gcount[si] & 0xFFFF):
            add("warning", "storage", f"storage {si}: count {s.count} != {gcount[si]} parts referencing it")
        elif gcount[si] > 0xFFFF:
            add("info", "storage", f"storage {si}: {gcount[si]} parts, count field wrapped to {s.count} (stock behaviour)")
        if s.size != gsize[si]:
            add("warning", "storage", f"storage {si}: size {s.size} != Σ align_up(part.size) = {gsize[si]}")

    # parts / owner / bounds / alignment ----------------------------------------------------------------------
    owned = [None] * P
    payload_start = align_up(pack.table_end, 16)
    for r in pack:
        if not 1 <= r.logical.part_count <= MAX_PARTS:
            add("error", "parts", f"{r.name!r}: {r.logical.part_count} parts (runtime supports 1..{MAX_PARTS})", r.index)
        for i in r.part_indices:
            if owned[i] is not None:
                add("error", "parts", f"physical {i} referenced by logical {owned[i]} and {r.index}", r.index, i)
            owned[i] = r.index
            p = pack.physicals[i]
            if p.owner != (r.index & 0xFFFF):
                add("error", "owner", f"{r.name!r}: part {i} owner {p.owner} != {r.index}", r.index, i)
            elif r.index > 0xFFFF and i == r.logical.first_part:
                add("info", "owner", f"{r.name!r}: logical index {r.index} > 65535, owner field wrapped (stock behaviour; parent-name lookup is wrong for it)", r.index, i)
            s = pack.storages[p.storage_index]
            off = pack.part_offset(i)
            if not p.child and s.method in (0, 1):
                if off < pack.table_end:
                    add("error", "bounds", f"{r.name!r}: part {i} at 0x{off:X} overlaps the tables (end 0x{pack.table_end:X})", r.index, i)
                if off + p.size > pack.size:
                    add("error", "bounds", f"{r.name!r}: part {i} 0x{off:X}+0x{p.size:X} exceeds file 0x{pack.size:X}", r.index, i)
                if off % s.alignment:
                    add("error", "alignment", f"{r.name!r}: part {i} offset 0x{off:X} not aligned to {s.alignment}", r.index, i)
            if p.packed & PHYS_BIT14:
                add("info", "unknown", f"{r.name!r}: part {i} physical bit 14", r.index, i)
            if p.packed & PHYS_BIT15:
                add("info", "unknown", f"{r.name!r}: part {i} physical bit 15", r.index, i)
            if p.packed & PHYS_SPECIAL:
                add("info", "unknown", f"{r.name!r}: part {i} physical bit 12 (0x1000)", r.index, i)
            if p.fc:
                add("info", "unknown", f"{r.name!r}: part {i} fc = 0x{p.fc:X}", r.index, i)
    for i, o in enumerate(owned):
        if o is None:
            add("warning", "parts", f"physical {i} is not referenced by any logical resource", part=i)

    # logical flags census (unknown semantics)
    lf: dict[int, int] = {}
    for l in pack.logicals:
        lf[l.flags] = lf.get(l.flags, 0) + 1
    rep.stats["logical_flags"] = {f"0x{k:02X}": v for k, v in sorted(lf.items())}
    pf: dict[int, int] = {}
    for p in pack.physicals:
        pf[p.flag_bits] = pf.get(p.flag_bits, 0) + 1
    rep.stats["physical_flag_bits"] = {f"0x{k:04X}": v for k, v in sorted(pf.items())}

    # on-demand contract -------------------------------------------------------------------------------------
    if h.field08 & FIELD08_ONDEMAND:
        n_mesh = n_ok = n_incompatible = n_viol = 0
        for r in pack.resources_of_type(0x10):
            n_mesh += 1
            res = ondemand_slices(pack, r.index)
            if res is None:
                n_incompatible += 1
                add("error", "ondemand", f"{r.name!r}: not on-demand compatible (method != 1 or physical 0x1000) in a bit-12 pack", r.index)
                continue
            expected, total = res
            bad = False
            for k, i in enumerate(r.part_indices):
                actual = pack.part_offset(i)
                if actual != expected[k]:
                    bad = True
                    add("error", "ondemand", f"{r.name!r}: part {i} (type 0x{pack.part_type(i):02X}) stored at 0x{actual:X}, engine slices at 0x{expected[k]:X}", r.index, i)
                    break
            if bad:
                n_viol += 1
            else:
                n_ok += 1
        rep.stats["ondemand_meshes"] = dict(total=n_mesh, ok=n_ok, violating=n_viol, incompatible=n_incompatible)

    # mesh identity ------------------------------------------------------------------------------------------
    if check_mesh_names:
        try:
            from ..classreader.image import embedded_mesh_name  # lazy: optional package
        except Exception:  # noqa: BLE001
            embedded_mesh_name = None
        if embedded_mesh_name is not None:
            n = bad = 0
            for r in pack.resources_of_type(0x10):
                img = r.part_by_type(0x10)
                fix = r.part_by_type(0x11)
                if img is None or fix is None:
                    add("error", "mesh_name", f"{r.name!r}: mesh without image/fixups parts", r.index)
                    continue
                try:
                    emb = embedded_mesh_name(pack.read_part(img), pack.read_part(fix))
                except Exception as exc:  # noqa: BLE001
                    add("error", "mesh_name", f"{r.name!r}: cannot read embedded name: {exc}", r.index)
                    continue
                n += 1
                if emb != r.name_raw + b".msh":
                    bad += 1
                    add("error", "mesh_name", f"{r.name!r}: embedded name {emb!r} != {r.name_raw + b'.msh'!r}", r.index)
            rep.stats["mesh_names"] = dict(checked=n, mismatched=bad)
    if dropped["warning"] or dropped["info"]:
        rep.stats["findings_dropped"] = dict(dropped)
    return rep
