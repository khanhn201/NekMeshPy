"""Shared internals for the :class:`~nekmeshpy.QuadMesh` factory functions.

``_apply_smoothing`` and ``_check_boundary`` are used by both the core container
(``quadmesh.py``) and the split-out factory files (``_open.py``); they live here so
those files can share them without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._typing import FloatArray, IntArray, PointArray
from ..linemesh import LineMesh
from ..model import conform
from ..model.interp import tensor_nodes

if TYPE_CHECKING:
    from .quadmesh import QuadMesh

#: One wall overlay: ``(quad ids, quad side 1-4, wall curve)`` -- the true boundary
#: ``LineMesh`` whose line ``k`` is the exact geometry of ``quad_ids[k]``'s named side,
#: replacing that side's straight guess.  Its ``interior`` nodes are the payload; its
#: ``points``/``lines`` give the traversal direction to match against the quad's.
Overlay = tuple[IntArray, int, LineMesh]


def _elevate(qm: QuadMesh, order: int,
             overlays: list[Overlay] | None = None) -> QuadMesh:
    """Return the order-N form of a linear (post-smoothing) region ``qm``.

    At ``order == 1`` returns ``qm`` unchanged (the golden no-op).  Otherwise the
    high-order nodes are built **natively as B-rep entities** -- no
    ``(Q,(order+1)**2,3)`` block is ever materialized:

    * every quad's four element-local edge interiors and its private quad interior are
      filled by straight (bilinear) subdivision of its CCW corners -- the same
      ``tensor_nodes(order, 2)`` weights a full straight-sided block would use,
      evaluated only at the entity slots;
    * each ``overlays`` entry then **overwrites** the element-local edge interior of the
      named side with the wall curve's own private ``interior`` nodes (reversed where
      the quad traverses that boundary line the other way), so region walls follow the
      exact loop while the interior stays a straight order-N fill;
    * the element-local edge copies are reconciled into the shared canonical table by
      :func:`~nekmeshpy.model.conform.scatter_edge_nodes` (owner-wins + verify).

    Corners are never stored -- they stay single-sourced by ``points[quads]``."""
    if order == 1:
        return qm
    from .quadmesh import QuadMesh, _edge_interior_slots, _quad_interior_slots
    points: PointArray = qm.points
    quads: IntArray = qm.quads
    nq = quads.shape[0]
    params = tensor_nodes(order, 2)                     # (M,2) in [0,1], i fastest
    u, v = params[:, 0], params[:, 1]
    # weights in quad CCW corner order [(0,0),(1,0),(1,1),(0,1)]
    W = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], axis=1)
    c = points[quads]                                   # (Q,4,3) CCW corners
    eslots = _edge_interior_slots(order)                # (4,order-1) traversal order
    local: FloatArray = np.einsum(
        "mk,qkd->qmd", W[eslots.ravel()], c).reshape(nq, 4, order - 1, 3)
    interior: FloatArray = np.einsum(
        "mk,qkd->qmd", W[_quad_interior_slots(order)], c)

    for quad_ids, side, wall in (overlays or []):
        ends: PointArray = wall.points[wall.lines]      # (L,2,3) directed endpoints
        inner: FloatArray = wall.interior               # (L,order-1,3) private nodes
        v0 = QuadMesh.EDGE_POINTS[side - 1, 0]
        ids: IntArray = np.asarray(quad_ids, dtype=np.int64).ravel()
        for k in range(ids.shape[0]):
            q = int(ids[k])
            start = points[quads[q, v0]]
            # the quad's side runs start->end; reverse the wall curve when it is
            # stored the other way round, exactly as the old side-stamp did.
            if (np.linalg.norm(ends[k, 0] - start)
                    > np.linalg.norm(ends[k, 1] - start)):
                local[q, side - 1] = inner[k][::-1]
            else:
                local[q, side - 1] = inner[k]

    edges, elem_edges, flip = conform.unique_edges(quads, 2)
    edge_nodes = conform.scatter_edge_nodes(
        local, elem_edges, flip, edges.shape[0],
        conform.entity_tol(points), "QuadMesh._elevate")
    lm = LineMesh(points, edges, order=order, interior=edge_nodes)
    return QuadMesh(lm, elem_edges, flip, interior,
                    qm.boundaries, qm.boundary_tags,
                    element_tags=qm.element_tags, order=order)


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
