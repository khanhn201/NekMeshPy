"""Shared internals for the :class:`QuadMesh <nekmeshpy.quadmesh.quadmesh.QuadMesh>`
factory functions."""

from __future__ import annotations

import numpy as np

from .._typing import IntArray, PointArray
from ..core import conform
from ..core.fields import gll_nodes
from ..core.interp import tensor_nodes
from ..linemesh import LineMesh
from .quadmesh import QuadMesh, _coons_at, _edge_interior_slots, _quad_interior_slots

#: One wall overlay: ``(quad ids, quad side 1-4, wall curve)`` -- the true boundary
#: ``LineMesh`` whose line ``k`` is the exact geometry of ``quad_ids[k]``'s named side,
#: replacing that side's straight guess.  Its ``interior`` nodes are the payload; its
#: ``points``/``lines`` give the traversal direction to match against the quad's.
Overlay = tuple[IntArray, int, LineMesh]


def _elevate(qm: QuadMesh, order: int,
             overlays: list[Overlay] | None = None) -> QuadMesh:
    """Return the order-N form of a linear (post-smoothing) region ``qm``."""
    if order == 1:
        return qm
    points: PointArray = qm.points
    quads: IntArray = qm.quads
    nq = quads.shape[0]
    params = tensor_nodes(order, 2)                     # (M,2) in [0,1], i fastest
    u, v = params[:, 0], params[:, 1]
    # weights in quad CCW corner order [(0,0),(1,0),(1,1),(0,1)]
    W = np.stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v], axis=1)
    c = points[quads]                                   # (Q,4,3) CCW corners
    eslots = _edge_interior_slots(order)                # (4,order-1) traversal order
    local: PointArray = np.einsum(
        "mk,qkd->qmd", W[eslots.ravel()], c).reshape(nq, 4, order - 1, 3)

    for quad_ids, side, wall in (overlays or []):
        ends: PointArray = wall.points[wall.lines]      # (L,2,3) directed endpoints
        inner: PointArray = wall.interior               # (L,order-1,3) private nodes
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

    # Private quad interiors: the transfinite (Coons) patch of the element's own four
    # edge curves, evaluated **after** the overlays.  A curved side therefore bows the
    # interior with it, instead of leaving a straight bilinear fill inside a curved
    # boundary.  With four straight edges the patch reduces to that bilinear fill.
    g = gll_nodes(order)
    row = order + 1
    islots = _quad_interior_slots(order)

    def side_curve(s: int) -> PointArray:
        """Side ``s``'s ``(Q,order+1,3)`` curve, start corner -> end corner."""
        v0, v1 = QuadMesh.EDGE_POINTS[s - 1]
        return np.concatenate([c[:, v0, None, :], local[:, s - 1],
                               c[:, v1, None, :]], axis=1)

    # _coons_at wants both families running with the lattice: bottom/top along i
    # (v0->v1 / v3->v2), left/right along j (v0->v3 / v1->v2).  Sides 3 and 4 are
    # stored v2->v3 and v3->v0, i.e. against it, so they are reversed.
    interior: PointArray = _coons_at(
        side_curve(1), side_curve(3)[:, ::-1], side_curve(4)[:, ::-1],
        side_curve(2), g, islots % row, islots // row)

    edges, elem_edges, flip = conform.unique_edges(quads, 2)
    edge_nodes = conform.scatter_edge_nodes(
        local, elem_edges, flip, edges.shape[0],
        conform.entity_tol(points), "QuadMesh._elevate")
    # the edge table is rebuilt here, so the tags are carried onto the new ids
    # rather than reused: local edge ``qm.quad[q, s]`` becomes ``elem_edges[q, s]``
    mine: IntArray = np.full(qm.lines.n_lines, -1, dtype=np.int64)
    mine[np.asarray(qm.quad, dtype=np.int64).ravel()] = np.asarray(
        elem_edges, dtype=np.int64).ravel()
    lm = LineMesh(points, edges, interior=edge_nodes,
                  element_tags=qm.edge_tags.renumber(mine))
    return QuadMesh(lm, elem_edges, flip, interior, qm.element_tags)


def entities_from_blocks(blocks: PointArray, quads: IntArray, points: PointArray,
                         order: int, who: str) -> tuple[LineMesh, IntArray,
                                                        IntArray, PointArray]:
    """Decompose per-element curved blocks into the B-rep tables."""
    local: PointArray = blocks[:, _edge_interior_slots(order)]
    interior: PointArray = blocks[:, _quad_interior_slots(order)]
    edges, elem_edges, flip = conform.unique_edges(quads, 2)
    edge_nodes = conform.scatter_edge_nodes(
        local, elem_edges, flip, edges.shape[0], conform.entity_tol(points), who)
    lm = LineMesh(points, edges, interior=edge_nodes)
    return lm, elem_edges, flip, interior


def _apply_smoothing(qm: QuadMesh, smoothing_method: str | None) -> QuadMesh:
    """Reposition ``qm``'s interior points in place (``None`` = no smoothing)."""
    if smoothing_method is not None:
        from . import smoothing
        smoothing.set_section_smoothing(qm, smoothing_method)
    return qm


def _check_boundary(obj: LineMesh, name: str, min_pts: int) -> PointArray:
    """Validate a ``LineMesh`` factory argument (type, minimum point count, finite
    coordinates), returning its ``(N,3)`` points."""
    if not isinstance(obj, LineMesh):
        raise TypeError("%s must be a LineMesh, got %s"
                        % (name, type(obj).__name__))
    pts = obj.points
    if pts.shape[0] < min_pts:
        raise ValueError("%s needs at least %d points, got %d"
                         % (name, min_pts, pts.shape[0]))
    if not np.all(np.isfinite(pts)):
        raise ValueError("%s has non-finite coordinates" % name)
    return pts
