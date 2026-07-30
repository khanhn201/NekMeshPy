"""Quad mesh of a single cross-section slice.

``QuadMesh`` is a pure container: ``points`` ``(nn,3)`` and quad connectivity
``quads`` ``(nq,4)``, plus a dense per-quad ``element_tags`` and a sparse tagged
boundary-edge list ``boundaries`` ``(Nbc,2)`` = ``[quad id, side 1-4]`` with a
parallel ``boundary_tags``.  Factory classmethods fill a bounded region with quads;
``extrude``/``loft`` sweep a ``LineMesh`` into a quad section.

The region-fill and closed-surface factories live beside this file as free functions
and are bound onto the class in ``quadmesh/__init__.py``: ``_open.py`` (``structured``
/ ``rectangle`` / ``ogrid`` / ``half_ogrid`` / ``annulus``) and ``_closed.py``
(``box`` / ``sphere``).  This file stays a pure container -- the core constructors
(``__init__`` / ``from_grid`` / ``merge`` / ``extrude`` / ``loft``) and all queries --
so adding a factory touches only the sibling module, never this one; the shared
``_apply_smoothing`` / ``_check_boundary`` helpers live in ``_helpers.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    Point,
    PointArray,
    StrArray,
    Vec3,
)
from ..linemesh import LineMesh
from ..model import conform
from ..model.fields import gll_nodes, reject_loop_caps, validate_layers
from ..model.interp import quad_edge_indices

#: Boundary-name sentinel meaning "not a boundary": a side carrying this name emits
#: no boundary row.  Equal to ``""`` so it reads as "unnamed" everywhere.
NO_BOUNDARY: str = ""

# default sweep axis / origin for extrude (module-level singletons; read-only)
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_ORIGIN = np.array([0.0, 0.0, 0.0])

# ``from_grid`` side name -> the ``loft`` channel that carries it.  The grid is lofted
# column by column (profile = the ``i`` chain, sweep = the ``j`` axis), so the two
# ``i``-ends of the profile ride the loft's tagged **boundary points** onto local quad
# sides 4 / 2, and the two ``j``-ends ride its ``first_tag`` / ``last_tag`` end caps
# onto sides 1 / 3.  Entries are ``("wall", profile vertex 1-2)`` / ``("cap", which)``.
_GRID_EDGES: dict[str, tuple[str, int]] = {
    "x_min": ("wall", 1), "x_max": ("wall", 2),
    "y_min": ("cap", 0), "y_max": ("cap", 1),
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
    ``boundary_tags``."""

    def __init__(
        self,
        lines: LineMesh,
        quad: IntArray,
        flip: BoolArray,
        interior: PointArray | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
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
        = ``[quad id, side 1-4]`` with a parallel ``boundary_tags``.

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
                    "QuadMesh: order %d > 1 requires interior nodes" % self._order)
            self.interior: PointArray = np.zeros((Q, 0, 3), dtype=float)
        else:
            ia: PointArray = np.asarray(interior, dtype=float)
            if ia.shape != (Q, k, 3):
                raise ValueError(
                    "QuadMesh: interior must be (Q,(order-1)**2,3) = (%d,%d,3), got %s"
                    % (Q, k, ia.shape))
            self.interior = ia
        # dense per-quad region/material tag ("" = untagged)
        if element_tags is None:
            self.element_tags: StrArray = np.full(Q, "", dtype=np.str_)
        else:
            et = np.asarray(element_tags, dtype=np.str_).reshape(-1)
            if et.shape[0] != Q:
                raise ValueError("element_tags length (%d) must match quads (%d)"
                                 % (et.shape[0], Q))
            self.element_tags = et
        # tagged boundary edges [quad id, side 1-4] parallel with boundary_tags
        self.boundaries: IntArray = (
            np.zeros((0, 2), np.int64) if boundaries is None
            else np.asarray(boundaries, np.int64).reshape(-1, 2))
        self.boundary_tags: StrArray = (
            np.empty(0, dtype=np.str_) if boundary_tags is None
            else np.asarray(boundary_tags, dtype=np.str_).reshape(-1))
        if self.boundary_tags.shape[0] != self.boundaries.shape[0]:
            raise ValueError("boundary_tags length (%d) must match boundaries (%d)"
                             % (self.boundary_tags.shape[0], self.boundaries.shape[0]))

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
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
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
                "nodes it cannot know. Build with a factory at order=%d (e.g. "
                "QuadMesh.ogrid(..., order=%d) or QuadMesh.loft(slices) from order-%d "
                "profiles), which places those nodes on the true geometry, or "
                "construct QuadMesh(lines, quad, flip, interior=..., order=%d) "
                "directly from the entity fields."
                % (order, order, order, order, order))
        pts: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
        conn: IntArray = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
        edges, elem_edges, flip = conform.unique_edges(conn, 2)
        lm = LineMesh(pts, edges)
        return cls(lm, elem_edges, flip, None, boundaries, boundary_tags,
                   element_tags, order=1)

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
        return self.boundaries.shape[0]

    @property
    def boundary_group_tags(self) -> list[str]:
        """Sorted unique tags of the tagged boundary edges."""
        return sorted(set(self.boundary_tags.tolist()))

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-quad element tags present on the section."""
        return sorted({t for t in self.element_tags.tolist() if t})

    # -- quality ---------------------------------------------------------
    def scaled_jacobian(self, *, high_order: bool = False) -> FloatArray:
        """Per-quad minimum scaled Jacobian ``(n_quads,)``.

        Defaults to the corner metric (the pinned linear numbers).  With
        ``high_order=True`` it is sampled at the ``(order+1)**2`` GLL nodes of the
        curved block (:func:`~nekmeshpy.quadmesh.quality.scaled_jacobian_ho`); at
        order 1 the two agree."""
        from . import quality
        if high_order:
            return quality.scaled_jacobian_ho(self, self.order)
        return quality.scaled_jacobian(self.points, self.quads)

    def quality_summary(self, *, high_order: bool = False) -> dict[str, Any]:
        """Aggregate scaled-Jacobian statistics (see :meth:`scaled_jacobian` for the
        ``high_order`` flag)."""
        from . import quality
        if high_order:
            return quality.summary_ho(self, self.order)
        return quality.summary(self.points, self.quads)

    @staticmethod
    def _order_bnd(
        bnd: Sequence[Sequence[int]] | IntArray,
        names: Sequence[str] | StrArray,
    ) -> tuple[IntArray, StrArray]:
        """Stably order boundary rows by ``(quad id, side)``, permuting the
        parallel tags array to match."""
        b: IntArray = np.asarray(bnd, dtype=np.int64).reshape(-1, 2)
        nm: StrArray = np.asarray(names, dtype=np.str_).reshape(-1)
        if b.shape[0]:
            order = np.lexsort((b[:, 1], b[:, 0]))
            b = b[order]
            nm = nm[order]
        return b, nm

    # -- boundary queries (topological section outline) -----------------
    @staticmethod
    def _boundary_mask(quads: IntArray) -> tuple[IntArray, BoolArray]:
        """``(edges, is_boundary)``: every quad edge ``(4M,2)`` element-major
        (row ``4q+e``), and a mask of those borne by a single quad."""
        Q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
        edges: IntArray = Q[:, QuadMesh.EDGE_POINTS].reshape(-1, 2)
        keys = np.sort(edges, axis=1)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        return edges, counts[inverse.ravel()] == 1

    def boundary_edges(self) -> IntArray:
        """``(K,2)`` array of ``[quad id, local edge (1-4)]`` for every edge on
        the section boundary (borne by a single quad)."""
        _, mask = self._boundary_mask(self.quads)
        rows = np.flatnonzero(mask)
        return np.column_stack([rows // 4, rows % 4 + 1]).astype(np.int64)

    def boundary_elements(self) -> IntArray:
        """Sorted unique quad ids with at least one edge on the section boundary."""
        return np.unique(self.boundary_edges()[:, 0])

    def boundary_points(self) -> IntArray:
        """Sorted unique point ids lying on the section boundary."""
        edges, mask = self._boundary_mask(self.quads)
        be = edges[mask]
        return np.unique(be) if be.size else np.zeros(0, dtype=np.int64)

    # -- assembly --------------------------------------------------------
    @classmethod
    def merge(cls, meshes: list[QuadMesh], *, tol: float | None = None) -> QuadMesh:
        """Merge quad sections into one, welding coincident boundary points.
        ``tol`` is the absolute coincidence distance (default ``1e-7`` x the extent).
        Tagged ``boundaries`` and dense ``element_tags`` concatenate with each
        block's quad ids offset; an interior seam is not auto-dropped."""
        meshes = list(meshes)
        pos = [np.asarray(m.points, dtype=float).reshape(-1, 3) for m in meshes]
        counts = [p.shape[0] for p in pos]
        P = np.concatenate(pos, axis=0) if pos else np.zeros((0, 3))
        total = P.shape[0]

        remap = np.arange(total, dtype=np.int64)
        is_bnd: BoolArray = np.zeros(total, dtype=bool)
        noff = 0
        for m, c in zip(meshes, counts):
            edges, mask = cls._boundary_mask(m.quads)
            is_bnd[noff + np.unique(edges[mask])] = True
            noff += c
        bidx = np.flatnonzero(is_bnd)
        if bidx.size:
            scl = float(np.max(P.max(axis=0) - P.min(axis=0)))
            t = tol if tol is not None else (1e-7 * scl if scl > 0 else 1.0)
            keys = np.round(P[bidx, :] / t).astype(np.int64)
            _, first_local, inverse = np.unique(
                keys, axis=0, return_index=True, return_inverse=True)
            remap[bidx] = bidx[first_local][inverse.ravel()]

        survivors = np.unique(remap)
        new_id: IntArray = np.empty(total, dtype=np.int64)
        new_id[survivors] = np.arange(survivors.size)
        point_id = new_id[remap]
        points = P[survivors, :]

        quad_list, bnd_list, name_list, etag_list = [], [], [], []
        noff = qoff = 0
        for m, c in zip(meshes, counts):
            quad_list.append(point_id[m.quads + noff])   # local -> welded id
            etag_list.append(m.element_tags)
            if m.boundaries.shape[0]:
                b: IntArray = m.boundaries.copy()
                b[:, 0] += qoff                          # shift quad ids; sides local
                bnd_list.append(b)
                name_list.append(m.boundary_tags)
            noff += c
            qoff += m.n_quads
        quads = np.concatenate(quad_list, axis=0) if quad_list else np.zeros((0, 4), np.int64)
        etags = (np.concatenate(etag_list) if etag_list
                 else np.empty(0, dtype=np.str_))
        bnd = np.concatenate(bnd_list, axis=0) if bnd_list else np.zeros((0, 2), np.int64)
        names = np.concatenate(name_list) if name_list else np.empty(0, dtype=np.str_)
        b_ord, n_ord = cls._order_bnd(bnd, names)

        # order-N: the private per-quad interiors just concatenate, but the shared edge
        # tables must be rebuilt against the *merged* topology -- gather each block's
        # nodes into its own element traversal order, concatenate in merged element
        # order, then re-scatter.  That scatter is the conformal-weld guard: two blocks
        # that disagree on a welded shared edge raise instead of silently welding.
        order = meshes[0].order if meshes else 1
        if any(m.order != order for m in meshes):
            raise ValueError("merge: all sections must share the same order")
        edges, elem_edges, flip = conform.unique_edges(quads, 2)
        edge_nodes: PointArray | None = None
        interior: PointArray | None = None
        if order > 1:
            local: PointArray = np.concatenate(
                [conform.gather_edge_nodes(m.lines.interior, m.quad, m.flip)
                 for m in meshes], axis=0)                     # (Q,4,order-1,3)
            edge_nodes = conform.scatter_edge_nodes(
                local, elem_edges, flip, edges.shape[0],
                conform.entity_tol(points), "QuadMesh.merge")
            interior = np.concatenate([m.interior for m in meshes], axis=0)
        lm = LineMesh(points, edges, order=order, interior=edge_nodes)
        return cls(lm, elem_edges, flip, interior, b_ord, n_ord,
                   element_tags=etags, order=order)

    @classmethod
    def blend(cls, a: QuadMesh, b: QuadMesh,
              fractions: FloatArray | Sequence[float]) -> list[QuadMesh]:
        """Linearly morph between two conformal sections ``a`` and ``b`` (identical
        ``quads``, equal point count), one section per fraction ``t`` with points
        ``(1-t)*a + t*b`` -- ``t=0`` reproduces ``a``, ``t=1`` reproduces ``b``.  Each
        result carries ``a``'s ``quads``, ``boundaries`` and ``boundary_tags``
        (positional BC markers follow the morph); per-quad ``element_tags`` are left
        for the consuming ``loft`` caps to assign, so a blended stack lofts directly.
        This is the profile-positioning step behind ``HexMesh.annulus``.

        The morph is delegated one rung **down the B-rep ladder**: the shared corners
        and the shared edge-interior nodes are exactly the edge ``LineMesh``, so
        :meth:`LineMesh.blend <nekmeshpy.linemesh.LineMesh.blend>` produces the blended
        edge mesh and this method only lerps what a quad owns privately -- its
        per-quad ``interior`` -- while keeping ``a``'s ``quad`` / ``flip`` incidence
        verbatim."""
        A: PointArray = np.asarray(a.points, dtype=float).reshape(-1, 3)
        B: PointArray = np.asarray(b.points, dtype=float).reshape(-1, 3)
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "blend: sections must have equal point counts (got %d, %d); build "
                "one from the other's points so they pair by index"
                % (A.shape[0], B.shape[0]))
        if not np.array_equal(a.quads, b.quads):
            raise ValueError(
                "blend: sections must share identical connectivity (paired by index)")
        if a.order != b.order:
            raise ValueError("blend: sections must have the same order "
                             "(got %d, %d)" % (a.order, b.order))
        # identical connectivity => identical edge tables, so a's and b's shared edge
        # nodes and private interiors already pair one-for-one and each morphs with the
        # same lerp the corners get from the blended points.  The shared corners *are*
        # the edge LineMesh's points and the shared edge nodes *are* its interior, so
        # that whole half of the morph is one ``LineMesh.blend`` of the rung below; the
        # result reuses ``a``'s per-quad edge indices / flips verbatim -- a blend is a
        # pure point-space morph, so nothing is re-derived.  At order 1 both entity
        # tables are empty and this is exactly the plain point blend.
        ho = a.order > 1
        ai, bi = a.interior, b.interior
        fr: FloatArray = np.asarray(fractions, dtype=float).ravel()
        return [cls(lm, a.quad, a.flip,
                    (1.0 - t) * ai + t * bi if ho else None,
                    boundaries=a.boundaries, boundary_tags=a.boundary_tags,
                    order=a.order)
                for t, lm in zip(fr, LineMesh.blend(a.lines, b.lines, fr))]

    @classmethod
    def from_grid(
        cls,
        P: PointArray,
        *,
        edge_tags: Mapping[str, str] | None = None,
        element_tag: str = "",
        order: int = 1,
    ) -> QuadMesh:
        """Build quads from a structured point grid ``P`` ``(ni+1,nj+1,3)``.
        ``edge_tags`` maps side names (``x_min`` / ``x_max`` / ``y_min`` / ``y_max``)
        to boundary tags on the four outer edges; a side left out (or mapped to
        ``NO_BOUNDARY``) emits no boundary row.  ``element_tag`` is written to every
        quad's dense ``element_tags``.

        ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
        each quad carries ``(order+1)**2`` straight-sided GLL nodes (a flat grid cell
        is exact under this subdivision).

        Built as a :meth:`loft` of the grid's **column profiles**, each of which is
        itself a :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>` of the
        grid's ``i`` points -- profile ``j`` is the open chain lofted from
        ``P[:, j, :]`` running ``i = 0..ni`` -- and the sweep runs ``j = 0..nj``, so
        the whole ladder ``HexMesh.from_grid -> from_grid -> LineMesh.loft`` is
        composed at every rung.  That is the orientation whose column quad
        ``[a_j, b_j, b_{j+1}, a_{j+1}]`` *is* the grid cell's CCW corner list
        ``[(i,j), (i+1,j), (i+1,j+1), (i,j+1)]``, and it routes all four tagged sides
        through channels ``loft`` already has: the profile's tagged **end points**
        become the ``x_min`` / ``x_max`` walls and the sweep's caps the ``y_min`` /
        ``y_max`` ones.  So the shared edges come out of the layer-by-layer B-rep
        assembly rather than a ``unique_edges`` re-derivation from corners -- this rung
        is composed from the one below like every other.

        **Ordering is the loft's, carried up unchanged** -- composing the rung below
        means accepting its numbering, so nothing is relabelled here.  The grid is
        therefore numbered **sweep-major** (``i`` fastest): grid node ``(i, j)`` is
        point ``j*(ni+1) + i`` and grid cell ``(i, j)`` is quad ``j*ni + i``, i.e.
        ``points`` equals ``P.transpose(1, 0, 2).reshape(-1, 3)`` -- *not* the
        ``P.reshape(-1, 3)`` (``j``-fastest) order this factory used historically.
        ``boundaries`` stays lexsorted by ``(quad, side)``, so its row order follows
        the quad ids; each tagged row still names the same physical side."""
        P = np.asarray(P, dtype=float)
        ni1, nj1, _ = P.shape
        ni = ni1 - 1
        tags = {s: n for s, n in (edge_tags or {}).items() if n}
        for side in tags:
            _GRID_EDGES[side]        # reject an unknown side name (KeyError)

        # -- the column profile, shared by every level -----------------------
        # np.full width-infers from the fill value (dtype=np.str_ would clip to <U1)
        line_tags: StrArray = np.full(ni, element_tag)
        # tagged profile end points -> the two swept walls (loft: vertex 1 -> quad side
        # 4, vertex 2 -> side 2), which is exactly x_min / x_max.
        pbnd: list[list[int]] = []
        pnames: list[str] = []
        for side in ("x_min", "x_max"):
            if side in tags:
                vertex = _GRID_EDGES[side][1]
                pbnd.append([0 if vertex == 1 else ni - 1, vertex])
                pnames.append(tags[side])
        pbnd_a: IntArray = np.asarray(pbnd, dtype=np.int64).reshape(-1, 2)
        # each profile is itself a ``LineMesh.loft`` of its ``i`` points: the rung below
        # builds the open ``i = 0..ni`` chain and, at order > 1, each segment's private
        # interior as the straight GLL blend of its two endpoints.  ``loft`` here builds
        # the sweep-direction rungs the same way and the quad interiors as the Coons
        # patch of the two, so a flat grid cell stays exact.
        slices = [LineMesh.loft(P[:, j, :], element_tags=line_tags,
                                boundaries=pbnd_a, boundary_tags=pnames, order=order)
                  for j in range(nj1)]
        # the loft *is* the result: its sweep-major numbering is carried up unchanged.
        return cls.loft(slices, first_tag=tags.get("y_min", ""),
                        last_tag=tags.get("y_max", ""))

    # -- line -> quad sweep (LineMesh one dimension down) ---------------
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

    @classmethod
    def extrude(
        cls,
        line: LineMesh,
        *,
        axis: Vec3 = _Z_AXIS,
        length: float,
        layers: FloatArray,
        origin: Point = _ORIGIN,
        first_tag: str | Sequence[str] | StrArray = "",
        last_tag: str | Sequence[str] | StrArray = "",
    ) -> QuadMesh:
        """Sweep a ``LineMesh`` a distance ``length`` along ``axis`` into a quad
        section (the straight special case of :meth:`loft`).

        The ``line`` is translated rigidly along ``axis``; ``origin`` shifts the whole
        section.  ``layers`` are normalized positions along ``axis`` (strictly
        increasing in ``[0, 1]`` with the last ``1``), giving ``layers.size - 1``
        layers.  The line's ``element_tags`` ride onto the swept quads and its tagged
        boundary points onto the side-wall edges; ``first_tag`` / ``last_tag`` name
        the near / far cap edges."""
        shift: Vec3 = np.asarray(origin, dtype=float)
        base = np.asarray(line.points, dtype=float).reshape(-1, 3) + shift
        axis_u: Vec3 = np.asarray(axis, dtype=float)
        axis_u = axis_u / np.linalg.norm(axis_u)
        offsets = validate_layers(layers, "extrude layers") * float(length)
        # the sweep is a rigid translation, so each slice's private high-order interior
        # nodes ride along by the same vector as its corner points -- no block anywhere.
        # Each slice reuses ``line.lines`` verbatim, so the swept strip wraps exactly
        # when the input curve's own connectivity does -- there is no flag to carry.
        ho = line.order > 1
        slices = [LineMesh(base + d * axis_u[None, :], line.lines,
                           element_tags=line.element_tags,
                           boundaries=line.boundaries,
                           boundary_tags=line.boundary_tags,
                           order=line.order,
                           interior=(line.interior + (shift + d * axis_u)[None, None, :]
                                     if ho else None))
                  for d in offsets]
        return cls.loft(slices, first_tag=first_tag, last_tag=last_tag)

    @classmethod
    def loft(
        cls,
        slices: Sequence[LineMesh],
        *,
        loop: bool = False,
        first_tag: str | Sequence[str] | StrArray = "",
        last_tag: str | Sequence[str] | StrArray = "",
    ) -> QuadMesh:
        """Loft a stack of conformal ``LineMesh`` profiles into a quad section
        (the general primitive behind :meth:`extrude`, and the middle rung of the
        uniform sweep shared with
        :meth:`LineMesh.loft <nekmeshpy.linemesh.LineMesh.loft>` and
        :meth:`HexMesh.loft <nekmeshpy.hexmesh.HexMesh.loft>`).

        ``slices`` is ``nz+1`` line profiles sharing the same ``lines``,
        ``element_tags``, and ``boundaries``; consecutive profiles form ``nz`` quad
        layers.  For line ``(a, b)`` at layer ``i`` the column quad is
        ``[a_i, b_i, b_{i+1}, a_{i+1}]``.  The line's ``element_tags`` ride onto every
        quad in its column and tagged boundary points onto the swept wall edges;
        ``first_tag`` / ``last_tag`` name the near / far cap edges (scalar or per-line
        array).

        ``loop=True`` makes the sweep **periodic**: the last profile is joined back to
        the *first*, so ``M`` profiles give ``M`` layers instead of ``M-1``.  It falls
        out of the layer-by-layer assembly as one extra iteration -- exactly one more
        rung block, appended once at the end, with the first profile's lines *not*
        duplicated -- so the seam is a genuine shared entity and the closed tube has
        no free boundary edge in the sweep direction.  A closed sweep has no near/far
        cap, so ``first_tag`` / ``last_tag`` with ``loop=True`` raise ``ValueError``
        rather than being silently dropped, and no cap boundary row is emitted.
        (Side-wall boundaries derived from the profiles' own boundary points are
        unaffected.)

        The section's B-rep is assembled **layer by layer**, never re-derived from
        corner connectivity: each profile's own ``LineMesh`` is merged into the growing
        global edge mesh (its ``lines`` and their private ``interior`` nodes carry
        through untouched), then the *rung* lines joining that level's points to the
        previous level's are appended (their ``interior`` is the straight GLL blend
        between the two corner points).  A quad is then just its four line indices plus
        four ``flip`` bits -- no ``unique_edges`` dedupe and no owner-wins
        ``scatter_edge_nodes`` reconciliation, because no shared edge is ever
        duplicated in the first place."""
        slices = list(slices)
        if loop:
            reject_loop_caps("QuadMesh.loft", first_tag, last_tag)
        lines = np.asarray(slices[0].lines, dtype=np.int64).reshape(-1, 2)
        L = lines.shape[0]
        n_prof = len(slices)
        # periodic: profile M-1 sweeps back onto profile 0, so there are M layers.
        nz = n_prof if loop else n_prof - 1
        S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                      for s in slices], axis=0)              # (n_prof, nn, 3)
        nn = S.shape[1]
        points = S.reshape(n_prof * nn, 3)                   # global id = i*nn + v

        order = slices[0].order
        if any(s.order != order for s in slices):
            raise ValueError("loft: all slices must share the same order")
        ho = order > 1

        a = lines[:, 0]
        b = lines[:, 1]
        # only points actually carried by a profile line get a rung (an isolated point
        # borders no quad, so a rung there would be a dangling edge).
        used: IntArray = (np.unique(lines.ravel()) if L
                          else np.zeros(0, dtype=np.int64))
        nu = used.shape[0]
        rung_slot: IntArray = np.full(nn, -1, dtype=np.int64)
        rung_slot[used] = np.arange(nu, dtype=np.int64)

        # -- grow the global edge LineMesh layer by layer -------------------
        # level i's profile lines occupy [prof_off[i], prof_off[i]+L); the rungs of
        # transition i->next(i) occupy [rung_off[i], rung_off[i]+nu).  ``nxt`` is the
        # profile each layer sweeps *to*: i+1 normally, wrapping to 0 for the closing
        # layer of a periodic sweep -- the single extra iteration below.
        nxt: IntArray = (np.arange(1, nz + 1, dtype=np.int64) % n_prof if nz
                         else np.zeros(0, dtype=np.int64))
        gi: FloatArray = gll_nodes(order)[1:order] if ho else np.zeros(0)
        prof_off: IntArray = np.empty(n_prof, dtype=np.int64)
        rung_off: IntArray = np.empty(max(nz, 0), dtype=np.int64)
        line_rows: list[IntArray] = []
        inter_rows: list[PointArray] = []
        cur = 0

        def _append_rung(layer: int) -> None:
            """Append the rung block of layer ``layer`` (profile ``layer`` ->
            ``nxt[layer]``) to the growing edge mesh -- appended exactly once per
            layer, so the seam rung of a periodic sweep is never duplicated."""
            nonlocal cur
            j = int(nxt[layer])
            rung_off[layer] = cur
            line_rows.append(np.column_stack([layer * nn + used, j * nn + used]))
            if ho:
                lo, hi = S[layer, used, :], S[j, used, :]
                inter_rows.append(lo[:, None, :]
                                  + gi[None, :, None] * (hi - lo)[:, None, :])
            cur += nu

        for i in range(n_prof):
            prof_off[i] = cur                                # merge level i's LineMesh
            line_rows.append(lines + i * nn)
            if ho:
                inter_rows.append(np.asarray(slices[i].interior, dtype=float))
            cur += L
            if i:                                            # rungs (i-1) -> i
                _append_rung(i - 1)
        if loop and nz:
            # the one extra iteration: the closing rung (M-1) -> 0.  No profile is
            # re-appended, so the seam shares profile 0's lines and nodes outright.
            _append_rung(nz - 1)
        all_lines: IntArray = (np.concatenate(line_rows, axis=0) if line_rows
                               else np.zeros((0, 2), dtype=np.int64))
        edge_nodes: PointArray | None = (
            np.concatenate(inter_rows, axis=0) if ho else None)

        # -- quads purely by line index -------------------------------------
        # quad row for (layer i, line l) = i*L + l; corners [a_i, b_i, b_{i+1}, a_{i+1}].
        # local edge 1 = level-i profile line l (forward), 2 = rung at b (forward),
        # 3 = level-(i+1) profile line l (reversed), 4 = rung at a (reversed).
        i_idx: IntArray = np.repeat(np.arange(nz, dtype=np.int64), L)
        l_idx = np.tile(np.arange(L, dtype=np.int64), nz)
        j_idx: IntArray = nxt[i_idx] if nz else i_idx        # the swept-to profile
        av = a[l_idx]
        bv = b[l_idx]
        quad: IntArray = np.stack([
            prof_off[i_idx] + l_idx,
            rung_off[i_idx] + rung_slot[bv],
            prof_off[j_idx] + l_idx,
            rung_off[i_idx] + rung_slot[av]], axis=1)
        flip: BoolArray = np.tile(
            np.array([False, False, True, True]), (nz * L, 1))
        etags: StrArray = np.asarray(slices[0].element_tags, dtype=np.str_)[l_idx]

        # order-N: each column quad is a transfinite (Coons) patch -- curved along the
        # profile line (from the slices' own points + private interior nodes), straight
        # along the sweep between consecutive slices.  Its boundary already lives on the
        # merged/rung lines, so the patch is evaluated only at the private interior
        # slots.
        interior: PointArray | None = None
        if ho:
            g = gll_nodes(order)
            row = order + 1
            # each profile's own high-order curve, assembled natively from the shared
            # corner points and that profile's private interior nodes
            Scur: PointArray = np.empty((n_prof, L, row, 3), dtype=float)
            Scur[:, :, 0, :] = S[:, a, :]
            Scur[:, :, order, :] = S[:, b, :]
            Scur[:, :, 1:order, :] = np.stack(
                [np.asarray(s.interior, dtype=float) for s in slices], axis=0)
            bottom = Scur[i_idx, l_idx]                     # (Q,row,3) a->b at i
            top = Scur[j_idx, l_idx]                        # (Q,row,3) a->b at next(i)
            a_lo, a_hi = S[i_idx, av], S[j_idx, av]         # (Q,3) sweep at a
            b_lo, b_hi = S[i_idx, bv], S[j_idx, bv]         # (Q,3) sweep at b
            gg = g[None, :, None]
            left = a_lo[:, None, :] + gg * (a_hi - a_lo)[:, None, :]   # (Q,row,3)
            right = b_lo[:, None, :] + gg * (b_hi - b_lo)[:, None, :]
            islots = _quad_interior_slots(order)
            interior = _coons_at(bottom, top, left, right, g,
                                 islots % row, islots // row)

        # tagged boundary point -> swept wall edge: vertex 0 -> side 4, vertex 1 -> 2
        sec_b = np.asarray(slices[0].boundaries, dtype=np.int64).reshape(-1, 2)
        sec_t = slices[0].boundary_tags
        bnd: list[list[int]] = []
        names: list[str] = []
        for r in range(sec_b.shape[0]):
            tag = str(sec_t[r])
            if tag == NO_BOUNDARY:
                continue
            l0 = int(sec_b[r, 0])
            qside = 4 if int(sec_b[r, 1]) == 1 else 2
            for ii in range(nz):
                bnd.append([ii * L + l0, qside])
                names.append(tag)
        # caps: scalar tags the whole cap, an array tags per section line.  A periodic
        # sweep has no near/far cap edge at all, so it emits no cap row (and rejected
        # the tags above).
        if not loop:
            first_caps = cls._cap_tags(first_tag, L)
            last_caps = cls._cap_tags(last_tag, L)
            for l0 in range(L):
                if first_caps[l0]:
                    bnd.append([l0, 1])
                    names.append(first_caps[l0])
            if nz:
                for l0 in range(L):
                    if last_caps[l0]:
                        bnd.append([(nz - 1) * L + l0, 3])
                        names.append(last_caps[l0])
        b_ord, n_ord = cls._order_bnd(bnd, names)
        lm = LineMesh(points, all_lines, order=order, interior=edge_nodes)
        return cls(lm, quad, flip, interior, b_ord, n_ord,
                   element_tags=etags, order=order)
