"""Read-only ``QuadMesh`` queries -- the operations that leave the ladder."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
    Vec3,
)
from ..model import frames
from ..model.quality import QualitySummary
from .quadmesh import QuadMesh


def _boundary_mask(quads: IntArray) -> tuple[IntArray, BoolArray]:
    """``(edges, is_boundary)``: every quad edge ``(4M,2)`` element-major
    (row ``4q+e``), and a mask of those borne by a single quad."""
    Q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    edges: IntArray = Q[:, QuadMesh.EDGE_POINTS].reshape(-1, 2)
    keys = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    return edges, counts[inverse.ravel()] == 1

def boundary_edges(mesh: QuadMesh) -> IntArray:
    """``(K,2)`` array of ``[quad id, local edge (1-4)]`` for every edge on
    the section boundary (borne by a single quad)."""
    _, mask = _boundary_mask(mesh.quads)
    rows = np.flatnonzero(mask)
    return np.column_stack([rows // 4, rows % 4 + 1]).astype(np.int64)

def boundary_elements(mesh: QuadMesh) -> IntArray:
    """Sorted unique quad ids with at least one edge on the section boundary."""
    return np.unique(boundary_edges(mesh)[:, 0])

def boundary_points(mesh: QuadMesh) -> IntArray:
    """Sorted unique point ids lying on the section boundary."""
    edges, mask = _boundary_mask(mesh.quads)
    be = edges[mask]
    return np.unique(be) if be.size else np.zeros(0, dtype=np.int64)

def scaled_jacobian(mesh: QuadMesh, *, high_order: bool = False) -> FloatArray:
    """Per-quad minimum scaled Jacobian ``(n_quads,)``."""
    from . import quality
    if high_order:
        return quality.scaled_jacobian_ho(mesh, mesh.order)
    return quality.scaled_jacobian(mesh.points, mesh.quads)

def quality_summary(mesh: QuadMesh, *, high_order: bool = False) -> QualitySummary:
    """Aggregate scaled-Jacobian statistics (see :func:`scaled_jacobian <nekmeshpy.quadmesh.query.scaled_jacobian>` for the
    ``high_order`` flag)."""
    from . import quality
    if high_order:
        return quality.summary_ho(mesh, mesh.order)
    return quality.summary(mesh.points, mesh.quads)

def plane_normal(mesh: QuadMesh, *,
                 hint: Vec3 | Sequence[float] | None = None,
                 check: bool = True) -> Vec3:
    """The unit normal of the plane a **planar** section lies in: the smallest right
    singular vector of its centred points, i.e. the exact least-squares plane."""
    normal = None
    if not check:
        P: PointArray = np.asarray(mesh.points, dtype=float)
        c = P.mean(axis=0)
        normal = np.linalg.svd(P - c)[2][2]
    R, _ = frames.plane_frame(mesh.points, normal=normal, hint=hint)
    w: Vec3 = R[:, 2]
    return w


__all__ = [
    "boundary_edges",
    "boundary_elements",
    "boundary_points",
    "plane_normal",
    "quality_summary",
    "scaled_jacobian",
]
