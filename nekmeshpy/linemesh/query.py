"""Read-only ``LineMesh`` queries -- the operations that leave the ladder."""

from __future__ import annotations

import numpy as np

from .._typing import (
    BoolArray,
    IntArray,
)
from .linemesh import LineMesh


def boundary_points(mesh: LineMesh) -> IntArray:
    """Sorted unique topological boundary point ids: the degree-1 ends. Empty
    for a closed loop; the two ends for an open chain."""
    if mesh.lines.size == 0:
        return np.zeros(0, dtype=np.int64)
    pids, counts = np.unique(mesh.lines.ravel(), return_counts=True)
    return pids[counts == 1]

def boundary_elements(mesh: LineMesh) -> IntArray:
    """Sorted unique line ids with at least one degree-1 end point -- the 1-D
    sibling of ``QuadMesh.boundary_elements`` / ``HexMesh.boundary_elements``,
    and vectorized the same way."""
    ends = boundary_points(mesh)
    if ends.size == 0:
        return np.zeros(0, dtype=np.int64)
    is_end: BoolArray = np.zeros(mesh.points.shape[0], dtype=bool)
    is_end[ends] = True
    return np.flatnonzero(is_end[mesh.lines].any(axis=1)).astype(np.int64)

__all__ = [
    "boundary_elements",
    "boundary_points",
]
