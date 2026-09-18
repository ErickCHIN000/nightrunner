"""Print the interesting parts of census JSONs: python tools/inspect_census.py NAME..."""
import json
import sys

for name in sys.argv[1:]:
    c = json.load(open(f"F:/DLTB/out/reports/census/{name}.census.json"))
    print("==", name, c["size"], c["header"])
    print("  types", c["types"], "max_parts", c["max_parts"], "fc", c["fc_values"])
    print("  logical_flags", c["logical_flags"])
    print("  phys flag bits", c["physical_flag_bits"])
    print("  file_order", c["file_order"])
    print("  storage_order first_app", c["storage_order_is_first_appearance"], "sorted", c["storage_order_sorted_by_type"], "nbo_logical", c["name_blob_order_is_logical"])
    print("  contig viol", c["logical_contiguity_violations_by_type"])
    print("  part shapes", c["part_shapes"])
    for s, g in zip(c["storages"], c["groups"]):
        print("   ", s["type"], "align", s["alignment"], "flags", s["flags"], "meta", s["metadata"], "base_units", s["base_units"],
              "size", s["size"], "count", s["count"], "| grp count", g["count"], "sum16", g["size_sum_aligned16"],
              "min_off", g["min_offset"], "contig", g["region_contiguous"], "mono", g["offset_units_monotonic"])
