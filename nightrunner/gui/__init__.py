"""Nightrunner — Qt (PySide6) GUI.

Phase 1 is the *raw* explorer: RP6L packs as a tree (header, storage table, resources grouped by type, parts),
a hex view of any part / table region / the whole file, a data inspector, raw export and the offline validator.
Nothing here decodes a resource body; the GUI only uses `nightrunner.container`. Launch with
`Nightrunner.pyw` (repo root) or `python -m nightrunner.gui`.
"""

APP_NAME = "Nightrunner"
