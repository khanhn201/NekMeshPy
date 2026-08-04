"""Shared internals for the :class:`~nekmeshpy.QuadMesh` factory functions.

``_apply_smoothing`` and ``_check_boundary`` are used by both the core container
(``quadmesh.py``) and the split-out factory files (``_open.py``); they live here so
those files can share them without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._typing import IntArray, PointArray
from ..linemesh import LineMesh
from ..model import conform
from ..model.fields import gll_nodes
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

    * every quad's four element-local edge interiors start as the straight (bilinear)
      subdivision of its CCW corners -- the same ``tensor_nodes(order, 2)`` weights a
      full straight-sided block would use, evaluated only at the entity slots;
    * each ``overlays`` entry then **overwrites** the element-local edge interior of the
      named side with that curve's own private ``interior`` nodes (reversed where the
      quad traverses the boundary line the other way), so the named sides follow the
      exact input curve.  A caller may overlay any side, not just a region wall:
      ``ogrid``/``half_ogrid`` pass one pair per O-ring so curvature reaches inward;
    * the private quad ``interior`` is then the **transfinite (Coons) patch of the
      element's own four edge curves**, taken *after* the overlays, so a curved side
      bows the nodes inside it.  With four straight edges the patch is the bilinear
      corner fill it replaces;
    * the element-local edge copies are reconciled into the shared canonical table by
      :func:`~nekmeshpy.model.conform.scatter_edge_nodes` (owner-wins + verify).

    Corners are never stored -- they stay single-sourced by ``points[quads]``."""
    if order == 1:
        return qm
    from .quadmesh import QuadMesh, _coons_at, _edge_interior_slots, _quad_interior_slots
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
    lm = LineMesh(points, edges, order=order, interior=edge_nodes)
    return QuadMesh(lm, elem_edges, flip, interior,
                    qm.edge_tags, qm.element_tags, order=order)


def entities_from_blocks(blocks: PointArray, quads: IntArray, points: PointArray,
                         order: int, who: str) -> tuple[LineMesh, IntArray,
                                                        IntArray, PointArray]:
    """Decompose per-element curved blocks into the B-rep tables.

    ``blocks`` is ``(Q,(order+1)**2,3)`` in the lexicographic (``i`` fastest) frame --
    the transient full-block form -- and is split into the shared-edge ``LineMesh``
    (reconciled owner-wins by
    :func:`~nekmeshpy.model.conform.scatter_edge_nodes`) plus the private per-quad
    ``interior``.  Returns ``(edge LineMesh, elem_edges, flip, interior)``.

    This is the inverse of the entity -> block gather, for a factory that can evaluate
    its region's true geometry at every node at once (``structured``) rather than
    subdividing a linear guess and stamping walls back on (``_elevate``)."""
    from .quadmesh import _edge_interior_slots, _quad_interior_slots
    local: PointArray = blocks[:, _edge_interior_slots(order)]
    interior: PointArray = blocks[:, _quad_interior_slots(order)]
    edges, elem_edges, flip = conform.unique_edges(quads, 2)
    edge_nodes = conform.scatter_edge_nodes(
        local, elem_edges, flip, edges.shape[0], conform.entity_tol(points), who)
    lm = LineMesh(points, edges, order=order, interior=edge_nodes)
    return lm, elem_edges, flip, interior


def _apply_smoothing(qm: QuadMesh, smoothing_method: str | None) -> QuadMesh:
    """Reposition ``qm``'s interior points in place (``None`` = no smoothing)."""
    if smoothing_method is not None:
        from . import smoothing
        smoothing.set_section_smoothing(qm, smoothing_method)
    return qm


def _check_boundary(obj: LineMesh, name: str, min_pts: int) -> PointArray:
    """Validate a ``LineMesh`` factory argument (type, minimum point count, finite
    coordinates), returning its ``(N,3)`` points.

    Open-vs-closed is deliberately *not* checked here: it is a property of the
    ``lines`` connectivity, stored nowhere, so each factory's own point-count and
    connectivity requirements are what constrain the input."""
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
