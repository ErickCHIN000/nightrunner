"""Non-mesh/non-texture resource families: lossless raw bundles + structural inspection dumps + corpus censuses.

Families (logical type → module):

    0x40 Animation (plain [40] or stream pair [44,45])   anim.py
    0x42 AnimationScr [42,43]                             animscr.py
    0x47 AnimGraphBank [47,48]                            animgraph.py
    0x49 AnimCustomResource [49,4A,49,4A]                 animcustom.py
    0x55 EnvprobeBin [55]                                 envprobe.py
    0x56 VoxelizerBin [56]                                voxelizer.py
    0x5A Area [5A]                                        area.py
    0x61 Prefab [61,62]                                   prefab.py

None of these bodies has an editable intermediate. `raw.py` writes the lossless JSON sidecar; each family module
adds a *structural* dump (words, tags, section tables, strings) whose every named field carries a provenance tier:
(A) verified by census or Binary Ninja, (B) behaviour of older code, (C) unknown. Field names are only used where the
survey/vault establishes them; everything else is reported as "word at offset". See notes/FORMATS/types.md (one
section per family, holes T1-T9).
"""

from __future__ import annotations

FAMILY_MODULES = {
    0x40: "anim",
    0x42: "animscr",
    0x47: "animgraph",
    0x49: "animcustom",
    0x55: "envprobe",
    0x56: "voxelizer",
    0x5A: "area",
    0x61: "prefab",
}

CENSUS_DATE = "2026-09-15"
