"""Quad mesh of a single cross-section slice.

``QuadMesh`` is a pure container: ``points`` ``(nn,3)`` and quad connectivity
``quads`` ``(nq,4)``, plus a dense per-quad ``element_tags`` and a sparse tagged
boundary-edge list ``boundaries`` ``(Nbc,2)`` = ``[quad id, side 1-4]`` with a
coupled tags.  Factory classmethods fill a bounded region with quads;
``extrude``/``loft`` sweep a ``LineMesh`` into a quad section.

This file stays a **pure container**: storage, validation, ``from_corners``, and the
derived views.  Every operation on a finished section lives beside it as a free function
bound onto the class in ``quadmesh/__init__.py``, split by arity and by rung delta --
``_assemble.py`` (the n-ary ``loft`` / ``merge``, which build a new index space),
``_lift.py`` (``extrude`` / ``from_grid``, which delegate to it), ``_morph.py``
(``blend``), ``_query.py`` (read-only queries), ``_open.py`` (region fills) and
``_closed.py`` (closed surfaces).  Adding an operation touches only the sibling module,
never this one; the shared ``_apply_smoothing`` / ``_check_boundary`` / ``_elevate``
factory internals live in ``_helpers.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
    StrArray,
)
from ..linemesh import LineMesh
from ..linemesh.linemesh import _repr_tags
from ..model import conform
from ..model.interp import quad_edge_indices
from ..model.tags import (
    NO_TAGS,
    BoundaryTable,
    ElementTags,
    check_tag_range,
)

#: Boundary-name sentinel meaning "not a boundary": a side carrying this name emits
#: no boundary row.  Equal to ``""`` so it reads as "unnamed" everywhere.
NO_BOUNDARY: str = ""

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
    corner) -- the frame
    :func:`~nekmeshpy.model.conform.scatter_edge_nodes` expects."""
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
    lattice coordinates are ``(ii, jj)`` -- ``i`` the profile axis, ``j`` the sweep axis.

    ``bottom`` / ``top`` are the ``(Q, order+1, 3)`` profile curves at the column's two
    bounding levels; ``left`` / ``right`` the straight GLL sweeps at its two profile
    corners.  Returns ``(Q, len(ii), 3)``: the same elementwise expression the full
    ``(Q, order+1, order+1, 3)`` patch uses, restricted to the requested slots, so the
    entity nodes are bit-identical to the corresponding slices of that patch."""
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
    """A quadrilateral surface / cross-section mesh in **B-rep** form.

    Storage is the boundary representation (source of truth): ``lines`` -- a shared
    ``LineMesh`` holding every edge (its ``points`` are the shared corners) -- plus
    per-quad edge indices ``quad`` ``(Q,4)`` into ``lines.lines``, a per-quad-per-edge
    ``flip`` ``(Q,4)`` orientation bit, and private per-quad ``interior`` nodes.  A
    shared edge is thus literally one stored object referenced by every incident quad
    (structural conformality), exactly as corners are one row of ``points``.  The
    familiar ``points`` ``(P,3)`` / ``quads`` ``(Q,4)`` CCW connectivity views are
    **derived** on read; build from corners with :meth:`from_corners`.  Also
    carries a dense per-quad ``element_tags`` and a sparse tagged-boundary list
    ``boundaries`` ``(Nbc,2)`` = ``[quad id, side 1-4]`` with a parallel
    coupled tags."""

    def __init__(
        self,
        lines: LineMesh,
        quad: IntArray,
        flip: BoolArray,
        interior: PointArray | None = None,
        boundaries: BoundaryTable | None = None,
        element_tags: ElementTags | None = None,
        *,
        order: int = 1,
    ) -> None:
        """Construct from the B-rep directly: ``lines`` (a ``LineMesh``
        holding every shared edge -- its ``points`` are the shared corners, its ``lines``
        the shared edge connectivity, its ``interior`` the shared edge-interior HO
        nodes), ``quad`` ``(Q,4)`` edge indices into ``lines.lines`` (CCW local edge
        order), ``flip`` ``(Q,4)`` bool (True where the quad traverses that edge
        anti-canonically), and ``interior`` ``(Q,(order-1)**2,3)`` private per-quad
        nodes (omit / ``None`` at order 1).  Also an optional dense per-quad
        ``element_tags`` ``(Q,)`` and a tagged-boundary list ``boundaries`` ``(Nbc,2)``
        = ``[quad id, side 1-4]`` coupled with its tags.

        ``.points`` / ``.quads`` are **derived** views over this B-rep, so a shared
        edge is literally one stored object referenced by every incident quad
        (structural conformality).  :meth:`from_corners` is the linear
        corner -> B-rep bridge for callers that only hold corner connectivity; a
        caller that already owns the edge ``LineMesh`` (``loft``, ``blend``, the
        section factories) builds through here directly and nothing is re-derived.
        ``re2`` export stays linear; only ``vtu`` reads the high-order nodes."""
        if not isinstance(lines, LineMesh):
            raise TypeError("QuadMesh: lines must be a LineMesh, got %s"
                            % type(lines).__name__)
        self._order = int(order)
        if lines.order != self._order:
            raise ValueError("QuadMesh: lines.order (%d) must match order (%d)"
                             % (lines.order, self._order))
        self.lines = lines
        self.quad: IntArray = np.asarray(quad, dtype=np.int64).reshape(-1, 4)
        self.flip: BoolArray = np.asarray(flip, dtype=bool).reshape(-1, 4)
        if self.flip.shape[0] != self.quad.shape[0]:
            raise ValueError("QuadMesh: flip length (%d) must match quad (%d)"
                             % (self.flip.shape[0], self.quad.shape[0]))
        Q = self.quad.shape[0]
        k = (self._order - 1) ** 2
        if interior is None:
            if self._order > 1:
                raise ValueError(
                    "QuadMesh: order %d > 1 requires the per-quad private interior "
                    "nodes (pass interior=(Q,(order-1)**2,3), or build the section "
                    "with a factory -- the region fills inherit their order from their "
                    "input boundary, e.g. "
                    "QuadMesh.ogrid(LineMesh.circle(r, n, order=%d), n_side, radial))"
                    % (self._order, self._order))
            self.interior: PointArray = np.zeros((Q, 0, 3), dtype=float)
        else:
            ia: PointArray = np.asarray(interior, dtype=float)
            if ia.shape != (Q, k, 3):
                raise ValueError(
                    "QuadMesh: interior must be (Q,(order-1)**2,3) = (%d,%d,3), got %s"
                    % (Q, k, ia.shape))
            self.interior = ia
        # dense per-quad region/material tag ("" = untagged)
        #: which quads carry a region tag (sparse -- untagged stores nothing)
        self.element_tags: ElementTags = (
            NO_TAGS if element_tags is None else element_tags)
        # tagged boundary edges: [quad id, side 1-4] coupled with their names
        self.boundaries: BoundaryTable = (
            BoundaryTable.empty() if boundaries is None else boundaries)
        check_tag_range(self.element_tags, self.boundaries, Q, 4, "quads")

        # corner connectivity is derived from quad/flip and immutable post-construction
        # (point moves don't change it), so memoize it once.
        self._corners: IntArray = self._derive_corners()

    # local quad edges (CCW); row e is edge e+1
    EDGE_POINTS = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)

    @classmethod
    def from_corners(
        cls,
        points: PointArray,
        quads: IntArray,
        boundaries: BoundaryTable | None = None,
        element_tags: ElementTags | None = None,
        *,
        order: int = 1,
    ) -> QuadMesh:
        """Build a **linear** ``QuadMesh`` from corner ``points`` ``(P,3)`` + CCW
        ``quads`` ``(Q,4)`` connectivity -- the corner -> B-rep bridge every factory
        routes through.  Decomposes the shared edges with ``conform.unique_edges``
        (lossless, so ``.quads`` round-trips the input exactly).

        Corners are all a linear mesh has, so this is an ``order == 1`` constructor:
        at ``order > 1`` there is no way to invent the shared edge / private interior
        nodes without silently straight-subdividing, so it raises.  Build with a
        factory (``ogrid`` / ``structured`` / ``box`` / ``sphere`` / ``loft`` / ...) at
        ``order=N``, which places those nodes on the true geometry, or construct
        ``QuadMesh(lines, quad, flip, interior, ..., order=N)`` directly from the
        entity fields if you already hold them."""
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
                "QuadMesh(lines, quad, flip, interior=..., order=%d) directly from the "
                "entity fields."
                % (order, order, order, order, order, order))
        pts: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
        conn: IntArray = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
        edges, elem_edges, flip = conform.unique_edges(conn, 2)
        lm = LineMesh(pts, edges)
        return cls(lm, elem_edges, flip, None, boundaries, element_tags, order=1)

    def _derive_corners(self) -> IntArray:
        """Corner connectivity ``(Q,4)`` recovered from the edge indices + flip: column
        ``k`` of quad ``q`` is the directed **start** of its local edge ``k`` --
        ``lines.lines[quad[q,k], 1 if flip[q,k] else 0]``.  Lossless inverse of
        ``conform.unique_edges``, so it reproduces the corner
        connectivity the mesh was built from byte-for-byte."""
        ln = self.lines.lines                          # (Ne,2) canonical edges
        eid = self.quad                                # (Q,4) edge ids
        start = np.where(self.flip, ln[eid, 1], ln[eid, 0])   # (Q,4)
        return start.astype(np.int64)

    def __repr__(self) -> str:
        """One-line REPL summary: element / point counts, ``order``, and the tag
        vocabulary -- the same field set
        :class:`LineMesh <nekmeshpy.linemesh.LineMesh>` and
        :class:`HexMesh <nekmeshpy.hexmesh.HexMesh>` render, so the three read as a
        family.

        Counts come from the stored B-rep (``lines.points`` and the ``quad`` incidence
        table) rather than the derived ``points`` / ``quads`` views, so it stays cheap
        and correct even on an instance whose memoized ``_corners`` never got built.
        Never raises -- see
        :meth:`LineMesh.__repr__ <nekmeshpy.linemesh.LineMesh.__repr__>`."""
        try:
            return ("<QuadMesh %d points, %d quads, order %d, element_tags=%s, "
                    "boundary_tags=%s>"
                    % (self.lines.points.shape[0], self.quad.shape[0], self._order,
                       _repr_tags(self.element_group_tags),
                       _repr_tags(self.boundary_group_tags)))
        except Exception:                     # a repr must never break a debug session
            return "<QuadMesh (unprintable)>"

    @property
    def order(self) -> int:
        """Global polynomial order (1 = linear)."""
        return self._order

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

    @property
    def n_points(self) -> int:
        """Number of (shared) points."""
        return self.points.shape[0]

    @property
    def n_quads(self) -> int:
        """Number of quadrilaterals."""
        return self.quads.shape[0]

    @property
    def n_boundaries(self) -> int:
        """Number of tagged boundary edges."""
        return len(self.boundaries)

    @property
    def boundary_group_tags(self) -> list[str]:
        """Sorted unique tags of the tagged boundary edges."""
        return self.boundaries.group_tags

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-quad element tags present on the section."""
        return self.element_tags.group_tags

    # -- helpers for the operation modules -----------------------------
    @staticmethod
    def _cap_tags(cap: str | Sequence[str] | StrArray, L: int) -> list[str]:
        """Normalize a cap tag to one tag per section line (length ``L``): a scalar
        ``str`` tags the whole cap, an array-like is a per-line tag."""
        if isinstance(cap, str):
            return [cap] * L
        arr = np.asarray(cap, dtype=np.str_).reshape(-1)
        if arr.shape[0] != L:
            raise ValueError("cap tags length (%d) must match section lines (%d)"
                             % (arr.shape[0], L))
        return [str(x) for x in arr.tolist()]


