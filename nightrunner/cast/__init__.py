"""Cast (dtzxporter) interchange: vendored library + Model → .cast export (+ phase-2 import).

    castlib.py   vendored Cast reader/writer (MIT; CAST-LICENSE.txt next to it)
    export.py    Model → Cast scene per notes/FORMATS/cast-mapping.md
    import_.py   Cast scene + mesh.json → resolved edit description (read_cast, resolve_import)
"""

from .export import build_cast, export_cast  # noqa: F401
