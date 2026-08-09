"""Read-only ``LineMesh`` queries -- the operations that leave the ladder."""

from __future__ import annotations

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    Point,
    PointArray,
)
from ..core import measure
from .linemesh import LineMesh


def element_blocks(mesh: LineMesh) -> PointArray:
    """``(L, order+1, 3)`` -- each line's own node block, assembled natively from the
    shared corner points and this mesh's private interior nodes.  Nothing is resampled or
    deduplicated: the block *is* what the B-rep already stores, gathered into the element
    lattice order (start corner -> interior -> end corner).

    The line rung of a query every rung above spells the same way -- see
    :func:`quadmesh.element_blocks <nekmeshpy.quadmesh.query.element_blocks>`."""
    order = mesh.order
    lines = mesh.lines
    out: PointArray = np.empty((lines.shape[0], order + 1, 3), dtype=float)
    out[:, 0, :] = mesh.points[lines[:, 0]]
    out[:, order, :] = mesh.points[lines[:, 1]]
    out[:, 1:order, :] = mesh.interior
    return out


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

def _blocks(mesh: LineMesh, high_order: bool) -> PointArray:
    """The node blocks a measure integrates: the curved ones the mesh stores, or the
    straight-sided corner blocks it reduces to."""
    if high_order:
        return element_blocks(mesh)
    return measure.corner_blocks(mesh.points, mesh.lines, 1)


def bounds(mesh: LineMesh, *, high_order: bool = False) -> measure.Bounds:
    """The axis-aligned bounding box of the mesh's nodes.

    At ``high_order=False`` that is the **corners** only, which does not enclose a
    curved element; pass ``high_order=True`` to include the stored interior nodes.
    Neither bounds the polynomial itself -- a curve can bulge past its own nodes -- so
    read this as an extent, not a guarantee."""
    return measure.bounds_of(_blocks(mesh, high_order) if high_order else mesh.points)


def element_lengths(mesh: LineMesh, *, high_order: bool = False) -> FloatArray:
    """``(L,)`` arc length of each line element.

    Straight chord by default; ``high_order=True`` integrates the curve the mesh
    actually stores, which is the reading that differs on anything built at
    ``order > 1``.  The line rung of :func:`quadmesh.element_areas
    <nekmeshpy.quadmesh.query.element_areas>` / :func:`hexmesh.element_volumes
    <nekmeshpy.hexmesh.query.element_volumes>`."""
    return measure.integrate(_blocks(mesh, high_order), 1)[0]


def length(mesh: LineMesh, *, high_order: bool = False) -> float:
    """Total arc length -- :func:`element_lengths` summed."""
    return float(element_lengths(mesh, high_order=high_order).sum())


def centroid(mesh: LineMesh, *, high_order: bool = False) -> Point:
    """The **length-weighted** centroid ``integral x ds / integral ds`` -- the mass
    property, not the mean of the points (which would weight a dense region of nodes
    over a long sparse one)."""
    return measure.centroid_of(_blocks(mesh, high_order), 1, "linemesh.centroid")


__all__ = [
    "bounds",
    "boundary_elements",
    "boundary_points",
    "centroid",
    "element_blocks",
    "element_lengths",
    "length",
]
