"""Read-only SDB (runtime_dx11.sdb / runtime_dx12.sdb) material database: MDBR container, the 39 Block-A tables,
compact integers, and the material → route → parameters → textures resolver. See notes/FORMATS/sdb.md."""

from .reader import Sdb, SdbError, VALUE_TYPES  # noqa: F401
