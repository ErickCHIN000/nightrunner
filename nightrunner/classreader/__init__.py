"""ClassReader blobs: the serialised object graph used by meshes (parts 0x10 + 0x11) and prefabs (0x61 + 0x62).

    fixups.py   part 0x11: sizes, object records, relocation slots, optional secondary stream (parse + serialise)
    image.py    part 0x10: typed readers over the primary image, pointer resolution through the slot table,
                embedded_mesh_name(), and an in-place patch builder for encoders
    graph.py    typed views of the mesh classes (3 root, 4 entity, 6 geometry, 7 palette, 10/11 material) and
                raw spans for everything else

Provenance: notes/survey/02-mesh-classreader-rig.md §2 (engine_core Initialize@ClassReader +0x7ED40,
Resolve@ClassReader +0x90280) and the corpus census recorded in notes/FORMATS/classreader.md.
"""

from .fixups import Fixups, Record, Slot  # noqa: F401
from .image import Image, embedded_mesh_name  # noqa: F401
