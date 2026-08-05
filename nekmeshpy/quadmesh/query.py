"""Read-only ``QuadMesh`` queries -- the operations that leave the ladder.

They take the mesh (or bare connectivity) and return plain arrays, counts or a named
tuple of statistics, never another mesh.

Free functions bound onto :class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>` by ``quadmesh/__init__.py``;
internal toolkit code imports them from here directly rather than through the bound
``QuadMesh.<name>`` sugar.
"""

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
    """Per-quad minimum scaled Jacobian ``(n_quads,)``.

    Defaults to the corner metric (the pinned linear numbers).  With
    ``high_order=True`` it is sampled at the ``(order+1)**2`` GLL nodes of the
    curved block (:func:`~nekmeshpy.quadmesh.quality.scaled_jacobian_ho`); at
    order 1 the two agree."""
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
    singular vector of its centred points, i.e. the exact least-squares plane.

    The question a caller asks of a cross-section before bridging or stubbing off it --
    "which way does this disc face?" -- and the honest answer for that is the section's
    own fitted plane, not the direction between two centroids, which a disc not
    perfectly centred on its nominal position tilts by a small angle.  ``sweep``'s
    ``"fixed"`` orientation makes the section exactly perpendicular to whatever tangent
    it is handed, so that small tilt lands the first station a little off the very disc
    it was supposed to reproduce.

    ``hint=`` flips the sign to agree with a direction; without it the sign is
    whichever the SVD returns, which is deterministic for a given point array but
    carries no outward/inward meaning.  ``check=False`` skips the planarity check for a
    section known not to be planar (a T-junction's saddle-shaped footprint disc, say),
    where the fitted plane is still the best available answer but
    :func:`frames.plane_frame <nekmeshpy.model.frames.plane_frame>` would rather refuse
    than guess.

    Only the normal: use ``frames.plane_frame`` directly for the full ``(R, origin)``
    frame, which this delegates to."""
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
