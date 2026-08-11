"""Quad mesh of a single cross-section slice."""

from __future__ import annotations

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
)
from ..core import conform
from ..core.interp import quad_edge_indices
from ..core.tags import ElementTags
from ..linemesh import LineMesh
from ..linemesh.linemesh import _repr_tags

#: Tag sentinel meaning "leave this side unnamed": a side carrying it emits no
#: side-tag row.  Equal to ``""`` so it reads as "unnamed" everywhere.
NO_TAG: str = ""

# default sweep axis / origin for extrude (module-level singletons; read-only)
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_ORIGIN = np.array([0.0, 0.0, 0.0])

# ``from_grid`` side name -> the ``loft`` channel that carries it, and the local quad
# side it lands on.  The grid is lofted column by column (profile = the ``i`` chain,
# sweep = the ``j`` axis), so the two ``i``-ends of the profile ride the loft's tagged
# **boundary points** onto quad sides 4 / 2, and the two ``j``-ends ride its
# ``first_tag`` / ``last_tag`` end caps onto sides 1 / 3.  Entries are
# ``("side", quad side 1-4)`` / ``("cap", quad side 1-4)``.  ``HexMesh`` keeps the same
# table one rung up under the same name (side name -> channel + Nek face).
_GRID_SIDES: dict[str, tuple[str, int]] = {
    "x_min": ("side", 4), "x_max": ("side", 2),
    "y_min": ("cap", 1), "y_max": ("cap", 3),
}


# -- entity slots of the lexicographic quad block -----------------------
def _edge_interior_slots(order: int) -> IntArray:
    """``(4, order-1)`` lexicographic (``i`` fastest) block slots strictly inside each
    CCW local edge, in **element traversal order** (that edge's start corner -> end
    corner) -- the frame :func:`~nekmeshpy.core.conform.scatter_edge_nodes` expects."""
    return np.stack([quad_edge_indices(s, order)[1:-1] for s in (1, 2, 3, 4)])


def _quad_interior_slots(order: int) -> IntArray:
    """``((order-1)**2,)`` lexicographic block slots strictly interior to the quad
    (interior on **both** axes) -- the private per-quad nodes, in the same ascending
    slot order the container's ``interior`` is stored in."""
    row = order + 1
    m: IntArray = np.arange(row * row, dtype=np.int64)
    i, j = m % row, m // row
    return m[(i > 0) & (i < order) & (j > 0) & (j < order)]


def _coons_at(bottom: PointArray, top: PointArray, left: PointArray,
              right: PointArray, g: FloatArray, ii: IntArray,
              jj: IntArray) -> PointArray:
    """A loft column's transfinite (Coons) patch evaluated at the block slots whose
    lattice coordinates are ``(ii, jj)`` -- ``i`` the profile axis, ``j`` the sweep
    axis."""
    uu = g[ii].reshape(1, -1, 1)
    vv = g[jj].reshape(1, -1, 1)
    bo, tp = bottom[:, ii, :], top[:, ii, :]
    lf, rt = left[:, jj, :], right[:, jj, :]
    P00, P10 = bottom[:, 0], bottom[:, -1]
    P01, P11 = top[:, 0], top[:, -1]
    return ((1 - vv) * bo + vv * tp + (1 - uu) * lf + uu * rt
            - ((1 - uu) * (1 - vv) * P00[:, None, :]
               + uu * (1 - vv) * P10[:, None, :]
               + (1 - uu) * vv * P01[:, None, :]
               + uu * vv * P11[:, None, :]))


class QuadMesh:
    """A quadrilateral surface / cross-section mesh in **B-rep** form."""

    # local quad edges (CCW); row e is edge e+1
    EDGE_POINTS = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)

    def __init__(
        self,
        lines: LineMesh,
        quad: IntArray,
        flip: BoolArray,
        interior: PointArray | None = None,
        element_tags: ElementTags | None = None,
    ) -> None:
        """Construct from the B-rep directly: ``lines`` (a ``LineMesh`` holding every
        shared edge -- its ``points`` are the shared corners, its ``lines`` the shared
        edge connectivity, its ``interior`` the shared edge-interior HO nodes), ``quad``
        ``(Q,4)`` edge indices into ``lines.lines`` (CCW local edge order), ``flip``
        ``(Q,4)`` bool (True where the quad traverses that edge anti-canonically), and
        ``interior`` ``(Q,(order-1)**2,3)`` private per-quad nodes (omit / ``None`` at
        order 1)."""
        if not isinstance(lines, LineMesh):
            raise TypeError("QuadMesh: lines must be a LineMesh, got %s"
                            % type(lines).__name__)
        self.lines = lines

        self.quad: IntArray = np.asarray(quad, dtype=np.int64).reshape(-1, 4)
        self.flip: BoolArray = np.asarray(flip, dtype=bool).reshape(-1, 4)
        if self.flip.shape[0] != self.quad.shape[0]:
            raise ValueError("QuadMesh: flip length (%d) must match quad (%d)"
                             % (self.flip.shape[0], self.quad.shape[0]))
        F = lines.n_lines                        # the rung below: shared edges
        if self.quad.size and (self.quad.min() < 0 or self.quad.max() >= F):
            raise ValueError(
                "QuadMesh: quad must index the %d shared edges of ``lines``; got ids "
                "in [%d, %d]" % (F, int(self.quad.min()), int(self.quad.max())))
        E = self.quad.shape[0]

        if interior is None:
            if self.order > 1:
                raise ValueError(
                    "QuadMesh: order %d > 1 requires the per-quad private interior "
                    "nodes (pass interior=(Q,(order-1)**2,3), or build the section "
                    "with a factory -- the region fills inherit their order from their "
                    "input boundary, e.g. "
                    "QuadMesh.ogrid(LineMesh.circle(r, n, order=%d), n_side, radial))"
                    % (self.order, self.order))
            interior = np.zeros((E, 0, 3), dtype=float)
        self.interior: PointArray = np.asarray(interior, dtype=float)
        k = (self.order - 1) ** 2
        if self.interior.shape != (E, k, 3):
            raise ValueError(
                "QuadMesh: interior must be (Q,(order-1)**2,3) = (%d,%d,3), got %s"
                % (E, k, self.interior.shape))

        self.element_tags = ElementTags.empty() if element_tags is None else element_tags
        self.element_tags.check_within(E)

        # corner connectivity is derived from quad/flip and immutable post-construction
        # (point moves don't change it), so memoize it once.
        self._corners: IntArray = self._derive_corners()

    @classmethod
    def from_corners(
        cls,
        points: PointArray,
        quads: IntArray,
        element_tags: ElementTags | None = None,
        *,
        order: int = 1,
    ) -> QuadMesh:
        """Build a **linear** ``QuadMesh`` from corner ``points`` ``(P,3)`` + CCW
        ``quads`` ``(Q,4)`` connectivity -- the corner -> B-rep bridge every factory
        routes through."""
        if order != 1:
            raise ValueError(
                "QuadMesh.from_corners builds the linear (order 1) B-rep from corner "
                "points alone; got order=%d, whose shared edge / private interior "
                "nodes it cannot know. Build with a factory instead, which places those "
                "nodes on the true geometry: the region fills and the sweeps inherit "
                "their order from their inputs (e.g. "
                "QuadMesh.ogrid(LineMesh.circle(r, n, order=%d), n_side, radial), or "
                "QuadMesh.loft(slices) from order-%d profiles), while the grid / "
                "analytic factories take it directly (QuadMesh.from_grid(P, order=%d), "
                "QuadMesh.rectangle(corners, nx, ny, order=%d)). Or construct "
                "QuadMesh(lines, quad, flip, interior=...) directly from the entity "
                "fields, where the order rides in on lines."
                % (order, order, order, order, order))
        pts: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
        conn: IntArray = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
        edges, elem_edges, flip = conform.unique_edges(conn, 2)
        lm = LineMesh(pts, edges)
        return cls(lm, elem_edges, flip, None, element_tags)

    @property
    def edge_tags(self) -> ElementTags:
        """The tags on the shared edges, over **edge ids**.

        This is ``lines``' own ``element_tags`` read through, not a table of its own:
        rung *N*'s side tags are rung *N-1*'s element tags. An edge is one stored
        object every incident quad references, so naming it is naming that object --
        which is why there is no ``(quad, side)`` here to disagree with itself."""
        return self.lines.element_tags

    def _derive_corners(self) -> IntArray:
        """Corner connectivity ``(Q,4)`` recovered from the edge indices + flip: column
        ``k`` of quad ``q`` is the directed **start** of its local edge ``k`` --
        ``lines.lines[quad[q,k], 1 if flip[q,k] else 0]``."""
        ln = self.lines.lines                          # (Ne,2) canonical edges
        eid = self.quad                                # (Q,4) edge ids
        start = np.where(self.flip, ln[eid, 1], ln[eid, 0])   # (Q,4)
        return start.astype(np.int64)

    def __repr__(self) -> str:
        """One-line REPL summary: element / point counts, ``order``, and the tag
        vocabulary -- the same field set :class:`LineMesh <nekmeshpy.linemesh.LineMesh>`
        and :class:`HexMesh <nekmeshpy.hexmesh.HexMesh>` render, so the three read as a
        family."""
        try:
            return ("<QuadMesh %d points, %d quads, order %d, element_tags=%s, "
                    "edge_tags=%s>"
                    % (self.lines.points.shape[0], self.quad.shape[0], self.order,
                       _repr_tags(self.element_group_tags),
                       _repr_tags(self.edge_group_tags)))
        except Exception:                     # a repr must never break a debug session
            return "<QuadMesh (unprintable)>"

    @property
    def order(self) -> int:
        """Global polynomial order (1 = linear)."""
        return self.lines.order

    @property
    def points(self) -> PointArray:
        """The ``(P,3)`` shared corner points -- a live view of the edge
        ``LineMesh``'s ``points`` (the single source of truth), so an
        in-place edit (``mesh.points[:] = X``) moves the shared corners for every quad."""
        return self.lines.points

    @property
    def quads(self) -> IntArray:
        """``(Q,4)`` CCW corner connectivity, derived (memoized) from the stored edge
        indices + flip.  Read-only; the B-rep ``quad`` / ``flip`` are the source of
        truth."""
        return self._corners

    @property
    def edges(self) -> IntArray:
        """``(Ne,2)`` unique undirected quad edges (canonical: min corner id first) --
        the shared edge topology (the ``lines`` of the edge ``LineMesh``).
        Non-empty at every order (edges are first-class B-rep storage)."""
        return self.lines.lines

    @property
    def edge_nodes(self) -> PointArray:
        """``(Ne, order-1, 3)`` shared high-order interior nodes of each unique
        :attr:`edges` entry, in canonical (min->max corner) order.  Empty at order 1;
        a shared edge resolves to the same nodes from either incident quad."""
        return self.lines.interior

    # -- sizes -----------------------------------------------------------
    @property
    def n_points(self) -> int:
        """Number of (shared) points."""
        return self.points.shape[0]

    @property
    def n_quads(self) -> int:
        """Number of quadrilaterals."""
        return self.quads.shape[0]

    @property
    def n_edge_tags(self) -> int:
        """Number of tagged edges."""
        return len(self.edge_tags)

    @property
    def edge_group_tags(self) -> list[str]:
        """Sorted unique tags of the tagged edges."""
        return self.edge_tags.group_tags

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-quad element tags present on the section."""
        return self.element_tags.group_tags

