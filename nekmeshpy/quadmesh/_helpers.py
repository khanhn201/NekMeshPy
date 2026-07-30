"""Shared internals for the :class:`~nekmeshpy.QuadMesh` factory functions.

``_apply_smoothing`` and ``_check_boundary`` are used by both the core container
(``quadmesh.py``) and the split-out factory files (``_open.py``); they live here so
those files can share them without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._typing import CurvedBlock, IntArray, PointArray
from ..linemesh import LineMesh
from ..model.interp import (
    corner_indices,
    quad_edge_indices,
    subdivide_quads,
)

if TYPE_CHECKING:
    from .quadmesh import QuadMesh

#: One wall overlay: ``(quad ids, quad side 1-4, per-edge curved nodes)`` -- the
#: true boundary curve to stamp onto those quads' side, replacing the straight guess.
Overlay = tuple[IntArray, int, CurvedBlock]


def _elevate(qm: QuadMesh, order: int,
             overlays: list[Overlay] | None = None) -> QuadMesh:
    """Return the order-N form of a linear (post-smoothing) region ``qm``.

    At ``order == 1`` returns ``qm`` unchanged (the golden no-op).  Otherwise the
    interior is straight-subdivided (:func:`~nekmeshpy.model.interp.subdivide_quads`)
    and each ``overlays`` entry stamps a true boundary curve onto the named quad side
    (auto-oriented to the side's ``start->end`` corner), so region walls follow the
    exact loop while the interior stays a straight order-N fill.  Corner nodes are
    pinned to ``points[quads]`` so the block stays corner-consistent."""
    if order == 1:
        return qm
    from .quadmesh import QuadMesh
    curved: CurvedBlock = subdivide_quads(qm.points, qm.quads, order)
    for quad_ids, side, blocks in (overlays or []):
        idx = quad_edge_indices(side, order)
        v0 = QuadMesh.EDGE_POINTS[side - 1, 0]
        for k in range(int(np.asarray(quad_ids).shape[0])):
            q = int(quad_ids[k])
            edge = np.asarray(blocks[k], dtype=float).reshape(-1, 3)
            start = qm.points[qm.quads[q, v0]]
            if (np.linalg.norm(edge[0] - start)
                    > np.linalg.norm(edge[-1] - start)):
                edge = edge[::-1]
            curved[q, idx, :] = edge
    curved[:, corner_indices(order, 2), :] = qm.points[qm.quads]
    return QuadMesh.from_corners(qm.points, qm.quads, qm.boundaries, qm.boundary_tags,
                                 element_tags=qm.element_tags, order=order,
                                 curved=curved)


def _apply_smoothing(qm: QuadMesh, smoothing_method: str | None) -> QuadMesh:
    """Reposition ``qm``'s interior points in place (``None`` = no smoothing)."""
    if smoothing_method is not None:
        from . import smoothing
        smoothing.set_section_smoothing(qm, smoothing_method)
    return qm


def _check_boundary(obj: LineMesh, name: str,
                    closed: bool, min_pts: int) -> PointArray:
    """Validate a ``LineMesh`` factory argument (open/closed topology, minimum
    point count, finite coordinates), returning its ``(N,3)`` points."""
    if not isinstance(obj, LineMesh):
        raise TypeError("%s must be a LineMesh, got %s"
                        % (name, type(obj).__name__))
    if obj.is_closed != closed:
        raise TypeError("%s must be a %s LineMesh"
                        % (name, "closed" if closed else "open"))
    pts = obj.points
    if pts.shape[0] < min_pts:
        raise ValueError("%s needs at least %d points, got %d"
                         % (name, min_pts, pts.shape[0]))
    if not np.all(np.isfinite(pts)):
        raise ValueError("%s has non-finite coordinates" % name)
    return pts
