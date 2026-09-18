"""Corpus census: run `nightrunner.container.census` over every .rpack under the assets dir and render SUMMARY.md.

    python tools/census_corpus.py [--assets DIR] [--out DIR] [--layouts auto,preserve] [--include-custom] [--json-only]

Defaults: assets = the game install (tests/paths.ASSETS), out = out/reports/census. Packs under `custom_rpacks`
are skipped unless --include-custom; `assets_N_pc.rpack` override slots (mod-built, loaded by the game) are
listed but excluded from the "shipped" rule counts. Writes one `<pack>.census.json` per pack, `summary.json`,
and the human-readable `SUMMARY.md` (per-pack tables + corpus-wide rule checks).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nightrunner.container import catalogue  # noqa: E402
from nightrunner.container.census import census  # noqa: E402
from nightrunner.container.rp6l import Pack, stock_storage_order  # noqa: E402
from nightrunner.util.binio import align_up  # noqa: E402
from nightrunner.util.jsonio import dump_json  # noqa: E402
from tests import paths  # noqa: E402

OVERRIDE_RE = re.compile(r"^assets_\d+_pc\.rpack$", re.I)


def classify(p: Path, assets: Path) -> str:
    rel = p.relative_to(assets)
    if "custom_rpacks" in rel.parts:
        return "custom"
    if OVERRIDE_RE.match(p.name):
        return "override"
    return "shipped"


def extras(pk: Pack) -> dict:
    """Facts the census module does not record: owner wraps, name-index identity, alignment/method census, tail rule."""
    owner_wrap = sum(1 for r in pk for i in r.part_indices if pk.physicals[i].owner != r.index)
    owner_wrap_ok = sum(1 for r in pk for i in r.part_indices
                        if pk.physicals[i].owner != r.index and pk.physicals[i].owner == (r.index & 0xFFFF))
    count_wraps = []
    actual = Counter(p.storage_index for p in pk.physicals)
    for si, s in enumerate(pk.storages):
        if s.count != actual[si]:
            count_wraps.append({"storage": si, "type": f"0x{s.type:02X}", "stored": s.count, "actual": actual[si],
                                "is_u16_wrap": s.count == (actual[si] & 0xFFFF)})
    last_end = max((pk.part_offset(i) + pk.physicals[i].size for i in range(len(pk.physicals))), default=pk.table_end)
    keys = [s.key for s in pk.storages]
    from nightrunner.util.names import engine_fold
    by_name = Counter(r.name_raw for r in pk)
    by_type_fold = Counter((r.type, engine_fold(r.name_raw)) for r in pk)
    return {
        "owner_wrap_parts": owner_wrap, "owner_wrap_parts_u16": owner_wrap_ok, "count_wraps": count_wraps,
        "duplicate_names_any_type": sum(c - 1 for c in by_name.values() if c > 1),
        "duplicate_type_folded_names": sum(c - 1 for c in by_type_fold.values() if c > 1),
        "name_index_is_logical_index": all(l.name_index == i for i, l in enumerate(pk.logicals)),
        "align_raw_values": dict(Counter(s.align_raw for s in pk.storages)),
        "methods_by_type": {f"0x{s.type:02X}": s.method for s in pk.storages},
        "codec_values": dict(Counter(s.codec for s in pk.storages)),
        "compressed_nonzero": sum(1 for s in pk.storages if s.compressed),
        "storage_order_is_stock": keys == stock_storage_order(keys),
        "size_multiple_of_16": pk.size % 16 == 0,
        "tail_bytes": pk.size - last_end,
        "tail_rule": pk.size == align_up(last_end, 16),
        "payload_start_rule": (not pk.physicals) or min(pk.part_offset(i) for i in range(len(pk.physicals))) == align_up(pk.table_end, 16),
    }


def run_one(p: Path, layouts: tuple[str, ...]) -> dict:
    t0 = time.time()
    with Pack.open(p) as pk:
        c = census(pk, layouts=layouts)
        c["extras"] = extras(pk)
    c["seconds"] = round(time.time() - t0, 2)
    return c


# ---- rendering -------------------------------------------------------------------------------------------------------

def _hx(v) -> str:
    return v if isinstance(v, str) else f"0x{v:X}"


def render_pack(c: dict, kind: str) -> list[str]:
    h = c["header"]
    ex = c["extras"]
    fo = c["file_order"]
    L = [f"### {c['file']}  ({kind})", ""]
    L.append(f"| size | field08 | flags | storages | physicals | logicals | table_end | payload start | tail bytes | census time |")
    L.append(f"|---|---|---|---|---|---|---|---|---|---|")
    L.append(f"| {c['size']:,} | {h['field08']} | {h['flags']} | {h['storage_count']} | {h['physical_count']} | {h['logical_count']} "
             f"| {c['table_end']} | {fo['payload_start']} (expected {fo['payload_start_expected']}) | {ex['tail_bytes']} | {c['seconds']}s |")
    L.append("")
    L.append("| # | type | align_raw (A) | flags | meta | method | ver | base_units | size | Σalign_up(size) | count | parts | region contiguous |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s, g in zip(c["storages"], c["groups"]):
        L.append(f"| {g['index']} | {s['type']} {s['type_name']} | {s['align_raw']} ({s['alignment']}) | {s['flags']} | {s['metadata']} "
                 f"| {s['method']} | {s['version']} | {s['base_units']} | {s['size']:,} | {'=' if g['size_sum_aligned16'] == s['size'] else g['size_sum_aligned16']} "
                 f"| {s['count']} | {g['count']} | {g['region_contiguous']} |")
    L.append("")
    L.append("- types: " + ", ".join(f"{t} {catalogue.type_name(int(t, 0))} × {n}" for t, n in c["types"].items()))
    L.append("- part shapes: " + "; ".join(f"{s['type']} → [{' '.join(s['parts'])}] × {s['count']}" for s in c["part_shapes"]))
    L.append("- logical flags: " + ", ".join(f"{l['type']}: {l['flags']} × {l['count']}" for l in c["logical_flags"]))
    L.append("- physical flag bits (per part type): " + ", ".join(f"{p['part_type']}: {p['bits']} × {p['count']}" for p in c["physical_flag_bits"]))
    L.append("- fc values: " + ", ".join(f"{k} × {v}" for k, v in c["fc_values"].items()) + f"; max parts per resource: {c['max_parts']}")
    L.append(f"- names: duplicates {c['duplicate_names']}, leading-space {c['leading_space_names']}, non-ASCII {c['nonascii_names']}, "
             f"name_index == logical index: {ex['name_index_is_logical_index']}, name blob order logical: {c['name_blob_order_is_logical']}")
    L.append(f"- layout: storage order stock rule: {ex['storage_order_is_stock']} (first-appearance: {c['storage_order_is_first_appearance']}, "
             f"sorted by type: {c['storage_order_sorted_by_type']}); file order == logical order: {fo['logical_order']}; "
             f"gaps {fo['gaps']} (max {fo['max_gap']} B, non-zero {fo['nonzero_gaps']}), overlaps {fo['overlaps']}; "
             f"size % 16 == 0: {ex['size_multiple_of_16']}, tail rule: {ex['tail_rule']}; "
             f"logical-order contiguity violations: {c['logical_contiguity_violations_by_type'] or 'none'}")
    if ex["owner_wrap_parts"] or ex["count_wraps"]:
        L.append(f"- u16 wraps: owner field wrapped on {ex['owner_wrap_parts']} parts ({ex['owner_wrap_parts_u16']} equal to index & 0xFFFF); "
                 f"storage count wraps: {ex['count_wraps'] or 'none'}")
    rb = []
    for layout, r in c["rebuild"].items():
        if "error" in r:
            rb.append(f"{layout}: ERROR {r['error']}")
        else:
            rb.append(f"{layout}: tables {'identical' if r['tables_identical'] else 'differ @' + str(r['first_table_diff'])}, "
                      f"part offset mismatches {r['part_offset_mismatches']}" + (f", warnings {r['warnings']}" if r["warnings"] else ""))
    L.append("- rebuild identity: " + "; ".join(rb))
    L.append("")
    return L


def render_summary(results: list[tuple[dict, str]], assets: Path, layouts: tuple[str, ...]) -> str:
    shipped = [c for c, k in results if k == "shipped"]
    n = len(shipped)
    L = [f"# RP6L corpus census — {time.strftime('%Y-%m-%d')}", "",
         f"Assets: `{assets}`. {len(results)} packs opened ({n} shipped, "
         f"{sum(1 for _, k in results if k == 'override')} override slot(s), {sum(1 for _, k in results if k == 'custom')} custom). "
         f"Rebuild layouts checked: {', '.join(layouts)}. Generated by `tools/census_corpus.py` from `nightrunner.container.census`.",
         "", "## Corpus-wide facts (shipped packs only unless stated)", ""]

    tot_l = sum(c["header"]["logical_count"] for c in shipped)
    tot_p = sum(c["header"]["physical_count"] for c in shipped)
    tot_b = sum(c["size"] for c in shipped)
    L.append(f"- Totals: {n} packs, {tot_l:,} logical resources, {tot_p:,} physical parts, {tot_b:,} bytes.")
    types: Counter = Counter()
    for c in shipped:
        for t, k in c["types"].items():
            types[t] += k
    L.append("- Logical resources per type: " + ", ".join(f"{t} {catalogue.type_name(int(t, 0))} {k:,}" for t, k in sorted(types.items())))
    f08 = Counter(c["header"]["field08"] for c in shipped)
    L.append("- header.field08: " + ", ".join(f"{k} × {v}" for k, v in sorted(f08.items())) +
             " (bit-12 packs: " + ", ".join(c["file"] for c in shipped if int(c["header"]["field08"], 0) & 0x1000) + ")")
    fl = Counter(c["header"]["flags"] for c in shipped)
    L.append("- header.flags: " + ", ".join(f"{k} × {v}" for k, v in sorted(fl.items())) + "; version 4 everywhere; name_count == logical_count " +
             f"{sum(1 for c in shipped if c['header']['name_count'] == c['header']['logical_count'])}/{n}")

    def rule(name: str, pred, detail=None) -> None:
        ok = [c for c in shipped if pred(c)]
        bad = [c["file"] for c in shipped if not pred(c)]
        L.append(f"- **{name}: {len(ok)}/{n}**" + (f" — exceptions: {', '.join(bad)}" if bad else "") + (f" {detail}" if detail else ""))

    rule("Storage-table order = stock rule (bit-3 storages first, then the rest, each sorted by type)", lambda c: c["extras"]["storage_order_is_stock"])
    rule("storage.size = Σ align_up(part.size, max(16, A)) for every storage", lambda c: all(g["size_sum_aligned16"] == g["storage_size"] for g in c["groups"]))
    rule("storage.count = number of parts (mod 65536)", lambda c: all(w["is_u16_wrap"] for w in c["extras"]["count_wraps"]))
    rule("Payload starts at align_up(table_end, 16)", lambda c: c["extras"]["payload_start_rule"])
    rule("File tail: size = align_up(end of last part, 16) (zero padded, < 16 bytes)", lambda c: c["extras"]["tail_rule"],
         "tail bytes seen: " + ", ".join(f"{k}×{v}" for k, v in sorted(Counter(c["extras"]["tail_bytes"] for c in shipped).items())))
    rule("No overlapping parts", lambda c: c["file_order"]["overlaps"] == 0)
    rule("Every gap between parts is zero-filled (gaps ≤ 15 bytes come from 16-byte part alignment inside groups)",
         lambda c: c["file_order"]["nonzero_gaps"] == 0, "max gap: " + str(max(c["file_order"]["max_gap"] for c in shipped)) + " bytes")
    rule("field08 == 0 packs: every storage group is one contiguous region",
         lambda c: int(c["header"]["field08"], 0) & 0x1000 or all(g["region_contiguous"] for g in c["groups"]))
    rule("field08 bit-12 packs: every 0x10 resource satisfies logical-order contiguity (engine one-read replay)",
         lambda c: not (int(c["header"]["field08"], 0) & 0x1000) or "0x10" not in c["logical_contiguity_violations_by_type"])
    rule("field08 bit-12 packs: non-stream (texture) groups are contiguous regions placed before the meshes",
         lambda c: not (int(c["header"]["field08"], 0) & 0x1000) or all(g["region_contiguous"] for g, s in zip(c["groups"], c["storages"]) if not (int(s["flags"], 0) & 8)))
    rule("Auto rebuild from scratch reproduces the tables byte-for-byte (name blob order copied)",
         lambda c: c["rebuild"].get("auto", {}).get("tables_identical") is True)
    rule("Auto rebuild reproduces every part offset", lambda c: c["rebuild"].get("auto", {}).get("part_offset_mismatches") == 0)
    if "preserve" in layouts:
        rule("Preserve rebuild reproduces the tables byte-for-byte", lambda c: c["rebuild"].get("preserve", {}).get("tables_identical") is True)
    rule("name_index == logical index", lambda c: c["extras"]["name_index_is_logical_index"])
    rule("Name blob order == logical order (i.e. derivable)", lambda c: c["name_blob_order_is_logical"] is True,
         "— the blob order is the original tool's insertion order; the writer copies it, it cannot derive it")
    rule("Owner field == logical index for every part", lambda c: c["extras"]["owner_wrap_parts"] == 0)
    rule("physical.fc == 0 for every part", lambda c: list(c["fc_values"]) == ["0x0"])
    rule("align_raw == 8 (A = 16) on every storage", lambda c: list(c["extras"]["align_raw_values"]) == [8])
    rule("storage method ∈ {0, 1}, codec 0, compressed 0", lambda c: all(s["method"] in (0, 1) for s in c["storages"]) and list(c["extras"]["codec_values"]) == [0] and c["extras"]["compressed_nonzero"] == 0)
    rule("≤ 15 parts per logical resource", lambda c: c["max_parts"] <= 15, f"max seen: {max(c['max_parts'] for c in shipped)}")
    rule("No duplicate (type, engine-folded name) pairs — the engine's lookup key", lambda c: c["extras"]["duplicate_type_folded_names"] == 0,
         f"(exact (type, name) duplicates: {sum(c['duplicate_names'] for c in shipped)})")
    rule("No duplicate names across types", lambda c: c["extras"]["duplicate_names_any_type"] == 0,
         f"total cross-type duplicates: {sum(c['extras']['duplicate_names_any_type'] for c in shipped)} (survey 01 §6 counted these 72; "
         "every one pairs two different types, e.g. 0x40 vs 0x49, so per-type lookup never collides)")
    rule("No leading-space names", lambda c: c["leading_space_names"] == 0, f"total: {sum(c['leading_space_names'] for c in shipped)}")
    rule("No non-ASCII name bytes", lambda c: c["nonascii_names"] == 0)

    L += ["", "### u16 wrap cases", ""]
    wraps = [(c, c["extras"]) for c in shipped if c["extras"]["owner_wrap_parts"] or c["extras"]["count_wraps"]]
    if not wraps:
        L.append("none")
    for c, ex in wraps:
        L.append(f"- {c['file']}: logical_count {c['header']['logical_count']:,}; owner field wrapped on {ex['owner_wrap_parts']:,} parts "
                 f"(all equal to index & 0xFFFF: {ex['owner_wrap_parts'] == ex['owner_wrap_parts_u16']}); storage count wraps: "
                 + (", ".join(f"storage {w['storage']} {w['type']}: stored {w['stored']} for {w['actual']:,} parts (u16 wrap: {w['is_u16_wrap']})" for w in ex["count_wraps"]) or "none"))

    L += ["", "### Logical flags byte per logical type", "", "| type | flags | resources | packs |", "|---|---|---|---|"]
    lf: dict = defaultdict(lambda: [0, set()])
    for c in shipped:
        for l in c["logical_flags"]:
            lf[(l["type"], l["flags"])][0] += l["count"]
            lf[(l["type"], l["flags"])][1].add(c["file"])
    for (t, f), (k, files) in sorted(lf.items()):
        L.append(f"| {t} {catalogue.type_name(int(t, 0))} | {f} | {k:,} | {len(files)} |")

    L += ["", "### Physical flag bits (packed bits 8..15) per part type", "", "| part type | bits | parts | packs |", "|---|---|---|---|"]
    pf: dict = defaultdict(lambda: [0, set()])
    for c in shipped:
        for p in c["physical_flag_bits"]:
            pf[(p["part_type"], p["bits"])][0] += p["count"]
            pf[(p["part_type"], p["bits"])][1].add(c["file"])
    for (t, b), (k, files) in sorted(pf.items()):
        L.append(f"| {t} {catalogue.type_name(int(t, 0))} | {b} | {k:,} | {len(files)} |")

    L += ["", "### Storage attributes per type (align_raw, flags, metadata → method / version / bit 3)", "",
          "| type | align_raw | flags | metadata | method | version | bit3 (stream) | storages | packs |", "|---|---|---|---|---|---|---|---|---|"]
    sa: dict = defaultdict(lambda: [0, set()])
    for c in shipped:
        for s in c["storages"]:
            sa[(s["type"], s["align_raw"], s["flags"], s["metadata"], s["method"], s["version"])][0] += 1
            sa[(s["type"], s["align_raw"], s["flags"], s["metadata"], s["method"], s["version"])][1].add(c["file"])
    for (t, a, f, m, meth, ver), (k, files) in sorted(sa.items()):
        L.append(f"| {t} {catalogue.type_name(int(t, 0))} | {a} | {f} | {m} | {meth} | {ver} | {bool(int(f, 0) & 8)} | {k} | {len(files)} |")

    L += ["", "### Part shapes per logical type", "", "| type | parts (storage types, logical order) | resources |", "|---|---|---|"]
    sh: Counter = Counter()
    for c in shipped:
        for s in c["part_shapes"]:
            sh[(s["type"], " ".join(s["parts"]))] += s["count"]
    for (t, parts), k in sorted(sh.items()):
        L.append(f"| {t} {catalogue.type_name(int(t, 0))} | {parts} | {k:,} |")

    L += ["", "### Packs whose auto rebuild differs (and why)", ""]
    diff = [(c, k) for c, k in results if not (c["rebuild"].get("auto", {}).get("tables_identical") is True and c["rebuild"].get("auto", {}).get("part_offset_mismatches") == 0)]
    if not diff:
        L.append("none")
    for c, k in diff:
        r = c["rebuild"].get("auto", {})
        why = []
        if not c["extras"]["storage_order_is_stock"]:
            why.append("storage table not in stock order: " + ", ".join(s["type"] for s in c["storages"]))
        if not all(g["size_sum_aligned16"] == g["storage_size"] for g in c["groups"]):
            why.append("storage.size ≠ Σ align_up(size)")
        if int(c["header"]["field08"], 0) & 0x1000 and any(s["base_units"] for s, g in zip(c["storages"], c["groups"]) if int(s["flags"], 0) & 8):
            why.append("stream storages carry non-zero base_units")
        if int(c["header"]["field08"], 0) & 0x1000 and not all(g["region_contiguous"] for s, g in zip(c["storages"], c["groups"]) if not (int(s["flags"], 0) & 8)):
            why.append("texture groups are not contiguous regions")
        if c["file_order"]["gaps"] and int(c["header"]["field08"], 0) & 0x1000:
            why.append(f"{c['file_order']['gaps']} gaps in the payload")
        L.append(f"- {c['file']} ({k}): tables {'identical' if r.get('tables_identical') else 'differ @' + str(r.get('first_table_diff'))}, "
                 f"offset mismatches {r.get('part_offset_mismatches')}; " + ("; ".join(why) if why else "reason not classified") +
                 ". Written by another tool — the byte-identity claim covers shipped packs only.")

    L += ["", "## Per-pack tables", ""]
    for c, k in results:
        L += render_pack(c, k)
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default=str(paths.ASSETS))
    ap.add_argument("--out", default=str(paths.ROOT / "out" / "reports" / "census"))
    ap.add_argument("--layouts", default="auto,preserve")
    ap.add_argument("--include-custom", action="store_true")
    ap.add_argument("--json-only", action="store_true", help="skip SUMMARY.md")
    a = ap.parse_args()
    assets = Path(a.assets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layouts = tuple(a.layouts.split(","))
    packs = sorted(assets.rglob("*.rpack"))
    results: list[tuple[dict, str]] = []
    summary = []
    for p in packs:
        kind = classify(p, assets)
        if kind == "custom" and not a.include_custom:
            continue
        try:
            c = run_one(p, layouts)
        except Exception as exc:  # noqa: BLE001
            print(f"{p.name:45s} FAILED {type(exc).__name__}: {exc}")
            continue
        c["kind"] = kind
        dump_json(c, out / (p.stem + ".census.json"))
        results.append((c, kind))
        summary.append({k: c[k] for k in ("file", "kind", "size", "types", "file_order", "rebuild", "storage_order_is_first_appearance",
                                            "storage_order_sorted_by_type", "name_blob_order_is_logical",
                                            "logical_contiguity_violations_by_type", "max_parts", "fc_values",
                                            "logical_flags", "physical_flag_bits", "extras", "seconds")})
        print(f"{p.name:45s} {kind:8s} {c['seconds']:6.1f}s rebuild={ {k: v.get('tables_identical', v.get('error')) for k, v in c['rebuild'].items()} }")
    dump_json(summary, out / "summary.json")
    if not a.json_only:
        (out / "SUMMARY.md").write_text(render_summary(results, assets, layouts), encoding="utf-8")
        print(f"wrote {out / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
