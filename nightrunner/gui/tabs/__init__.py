"""One module per main-window tab. Each exposes `class Tab(QWidget)` with `TITLE`, `__init__(self, ctx)`, and
optional navigation slots (`open_gid(gid)`, `open_material(name)`, `open_model(name)`) plus `shutdown()`.

Tab order (fixed): raw, textures, meshes, models, sdb, build.
"""
TAB_MODULES = ("raw", "textures", "meshes", "models", "sdb", "build")
