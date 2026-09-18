"""Shared pieces of the single-file model Cast (gui/modelcast.py writes it, cast/split.py undoes it).

* `MergedSkeleton` — the preset skeleton's bones first, then every part bone it lacks, matched by name
  (ASCII case-insensitive), parents matched by name, rest globals from whoever supplied the bone.
* `rebind_matrices` — per part entity `G_skel[mapped] · inv_bind_part` (fallback `inv(G_part)`): the matrix that
  moves a part vertex from its own bind space to the merged skeleton's rest pose (what the engine's skinning gives
  at that pose). The splitter applies the inverse of the weighted blend.
"""
from __future__ import annotations

import numpy as np


def _m4(m34) -> np.ndarray:
    out = np.eye(4)
    out[:3, :4] = np.asarray(m34, dtype=np.float64)
    return out


class MergedSkeleton:
    """Merged bone list: names, parents, 4×4 rest globals."""

    def __init__(self):
        self.names: list[bytes] = []
        self.parents: list[int] = []
        self.globals: list[np.ndarray] = []
        self.index: dict[bytes, int] = {}
        self.source: list[str] = []

    def add_model(self, model, label: str) -> None:
        g = model.entity_globals()
        for e in model.entities:
            self._add(e.name, model, g, label)

    def _add(self, name: bytes, model, g, label: str) -> int:
        key = name.lower()
        if key in self.index:
            return self.index[key]
        e = next(x for x in model.entities if x.name == name)
        parent = -1
        if e.parent >= 0:
            parent = self._add(model.entities[e.parent].name, model, g, label)
        self.index[key] = len(self.names)
        self.names.append(name)
        self.parents.append(parent)
        self.globals.append(np.asarray(g[e.index], dtype=np.float64))
        self.source.append(label)
        return self.index[key]

    def map_model(self, model, label: str) -> np.ndarray:
        """Part entity index -> merged bone index (appending unknown bones)."""
        g = model.entity_globals()
        return np.array([self._add(e.name, model, g, label) for e in model.entities], dtype=np.int64)


def rebind_matrices(model, skel: MergedSkeleton, emap: np.ndarray) -> tuple[np.ndarray, float]:
    """Per part entity: G_skel[mapped] · inv_bind_part (fallback inv(G_part)); and max |M − I|."""
    g = model.entity_globals()
    out = np.zeros((len(model.entities), 4, 4))
    worst = 0.0
    for e in model.entities:
        ib = _m4(e.inv_bind)
        if not np.isfinite(ib).all() or abs(np.linalg.det(ib[:3, :3])) < 1e-8:
            ib = np.linalg.inv(g[e.index])
        m = skel.globals[int(emap[e.index])] @ ib
        out[e.index] = m
        worst = max(worst, float(np.abs(m - np.eye(4)).max()))
    return out, worst


