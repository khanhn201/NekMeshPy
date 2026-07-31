"""All-hex mesh container.

``HexMesh`` stores ``points`` ``(P,3)``, ``hexes`` ``(N,8)`` connectivity in Nek
order, a sparse tagged ``boundaries`` ``(Nbc,2)`` = ``[element id, face 1-6]`` with
parallel ``boundary_tags``, and a dense per-hex ``element_tags``. Boundary tags map
to Nek BC codes only at export.

It is built complete, not incrementally: from arrays or via the factory
classmethods ``loft`` / ``extrude`` / ``annulus`` / ``merge`` / ``from_grid``. The
topology is fixed at construction, but coordinates may be repositioned in place.

This file stays a **pure container**: storage, validation, ``from_corners``, and the
derived views.  Every operation on a finished block lives beside it as a free function
bound onto the class in ``hexmesh/__init__.py``, split by arity and by rung delta --
``_assemble.py`` (the n-ary ``loft`` / ``merge``, which build a new index space),
``_lift.py`` (``extrude`` / ``annulus`` / ``from_grid``, which delegate to it),
``_morph.py`` (``blend``) and ``_query.py`` (queries, topology and reporting).  Adding
an operation touches only the sibling module, never this one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .._typing import (
    FloatArray,
    IntArray,
    PointArray,
    StrArray,
)
from ..model import conform
from ..model.interp import corner_indices
from ..quadmesh import QuadMesh

# default sweep axis / origin for extrude
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_ORIGIN = np.array([0.0, 0.0, 0.0])

# ``from_grid`` side name -> the ``loft`` channel that carries it, and the Nek face it
# lands on -- the same table ``QuadMesh``'s ``_GRID_SIDES`` holds one rung down, in the
# same shape.  The x/y sides ride the section's own ``edge_tags`` (section side ``s`` ->
# hex face ``s``) and the z sides the sweep's end caps.  ``from_grid`` routes every side
# through those channels, so this is the *reference* mapping the composition reproduces
# (and the set of legal side names it validates against), not a lookup it indexes a
# plane with.
_GRID_SIDES: dict[str, tuple[str, int]] = {
    "x_min": ("side", 4), "x_max": ("side", 2),
    "y_min": ("side", 1), "y_max": ("side", 3),
    "z_min": ("cap", 5), "z_max": ("cap", 6),
}


# -- native (block-free) high-order helpers -----------------------------
def _slice_block(s: QuadMesh, order: int) -> PointArray:
    """``(Q,(order+1)**2,3)`` in-plane high-order block of one loft profile, assembled
    natively from that section's B-rep (shared corners ++ shared edge-interior nodes in
    element traversal order ++ private quad interiors).  A loft column's geometry is a
    straight sweep between two such in-plane blocks, so this is the only intermediate
    :meth:`HexMesh.loft` needs -- no per-element hex block is ever materialized."""
    row = order + 1
    out: PointArray = np.empty((s.quads.shape[0], row * row, 3), dtype=float)
    out[:, corner_indices(order, 2), :] = s.points[s.quads]
    out[:, conform._edge_slots(2, order)[:, 1:-1], :] = conform.gather_edge_nodes(
        s.lines.interior, s.quad, s.flip)
    out[:, conform._interior_slots(2, order), :] = s.interior
    return out


def _sweep_at(bottom: PointArray, top: PointArray, g: FloatArray,
              slots: IntArray, m2: int) -> PointArray:
    """A loft column's straight GLL sweep evaluated at the hex block ``slots``.

    Hex lexicographic slot ``s`` decomposes as ``s = k*m2 + m`` (in-plane index ``m``
    fastest, sweep index ``k`` slowest), so the node there is
    ``(1-g[k])*bottom[m] + g[k]*top[m]`` -- the same elementwise expression the full
    ``(E,order+1,m2,3)`` sweep uses, restricted to the requested slots.  Returns
    ``(E, len(slots), 3)``."""
    k = slots // m2
    m = slots % m2
    gg = g[k].reshape(1, -1, 1)
    return (1.0 - gg) * bottom[:, m, :] + gg * top[:, m, :]


class HexMesh:
    """An all-hexahedral volume mesh in shared-point form.

    Stores ``points`` ``(P,3)``, ``hexes`` ``(N,8)`` connectivity (Nek order), a
    sparse tagged ``boundaries`` with parallel ``boundary_tags``, and a dense
    per-hex ``element_tags``. Immutable topology; build via a factory or the array
    constructor."""

    # Nek face -> the 4 corner point positions (0-based); row f is face f+1.
    FACE_POINTS = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                           [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)

    def __init__(
        self,
        quads: QuadMesh,
        hex: IntArray,
        face_orient: IntArray,
        interior: PointArray | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
        *,
        order: int = 1,
    ) -> None:
        """Construct from the B-rep directly: ``quads`` (a ``QuadMesh`` holding every
        shared face -- its ``points`` are the shared corners, its ``quads`` the shared
        face connectivity, its edges / ``interior`` the shared face-boundary / interior
        HO nodes), ``hex`` ``(E,6)`` face indices into ``quads.quads`` (Nek local-face
        order), ``face_orient`` ``(E,6)`` D4 codes (element-local face frame ->
        canonical), and ``interior`` ``(E,(order-1)**3,3)`` private per-hex nodes (omit /
        ``None`` at order 1).  Also an optional dense per-hex ``element_tags`` ``(E,)``
        and a tagged-boundary list ``boundaries`` ``(Nbc,2)`` = ``[hex id, face 1-6]``
        with a parallel ``boundary_tags``.

        ``.points`` / ``.hexes`` are **derived** views over this B-rep, so
        a shared face is literally one stored object referenced by every incident hex
        (structural conformality).  :meth:`from_corners` is the linear corner -> B-rep
        bridge for callers that only hold corner connectivity; a caller that already
        owns the shared-face ``QuadMesh`` builds through here directly.  ``re2`` export
        stays linear; only ``vtu`` reads the high-order nodes."""
        if not isinstance(quads, QuadMesh):
            raise TypeError("HexMesh: quads must be a QuadMesh, got %s"
                            % type(quads).__name__)
        self._order = int(order)
        if quads.order != self._order:
            raise ValueError("HexMesh: quads.order (%d) must match order (%d)"
                             % (quads.order, self._order))
        self.quads = quads
        self.hex: IntArray = np.asarray(hex, dtype=np.int64).reshape(-1, 6)
        self.face_orient: IntArray = np.asarray(
            face_orient, dtype=np.int64).reshape(-1, 6)
        if self.face_orient.shape[0] != self.hex.shape[0]:
            raise ValueError("HexMesh: face_orient length (%d) must match hex (%d)"
                             % (self.face_orient.shape[0], self.hex.shape[0]))
        E = self.hex.shape[0]
        k = (self._order - 1) ** 3
        if interior is None:
            if self._order > 1:
                raise ValueError(
                    "HexMesh: order %d > 1 requires the per-hex private interior "
                    "nodes (pass interior=(E,(order-1)**3,3), or build the block "
                    "with a factory such as HexMesh.extrude(section, ...) from an "
                    "order-%d section)" % (self._order, self._order))
            self.interior: PointArray = np.zeros((E, 0, 3), dtype=float)
        else:
            ia: PointArray = np.asarray(interior, dtype=float)
            if ia.shape != (E, k, 3):
                raise ValueError(
                    "HexMesh: interior must be (E,(order-1)**3,3) = (%d,%d,3), got %s"
                    % (E, k, ia.shape))
            self.interior = ia
        if element_tags is None:
            self.element_tags: StrArray = np.full(E, "", dtype=np.str_)
        else:
            et = np.asarray(element_tags, dtype=np.str_).reshape(-1)
            if et.shape[0] != E:
                raise ValueError("element_tags length (%d) must match hexes (%d)"
                                 % (et.shape[0], E))
            self.element_tags = et
        self.boundaries: IntArray = (
            np.zeros((0, 2), np.int64) if boundaries is None
            else np.asarray(boundaries, np.int64).reshape(-1, 2))
        self.boundary_tags: StrArray = (
            np.empty(0, dtype=np.str_) if boundary_tags is None
            else np.asarray(boundary_tags, dtype=np.str_).reshape(-1))
        if self.boundary_tags.shape[0] != self.boundaries.shape[0]:
            raise ValueError("boundary_tags length (%d) must match boundaries (%d)"
                             % (self.boundary_tags.shape[0], self.boundaries.shape[0]))

        # corner connectivity + per-hex edge incidence are derived from the shared
        # faces and immutable post-construction (point moves don't change them), so
        # memoize once.
        self._corners: IntArray = self._derive_corners()
        _, self._elem_edges, self._edge_flip = conform.unique_edges(self._corners, 3)

    @classmethod
    def from_corners(
        cls,
        points: PointArray,
        hexes: IntArray,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
        *,
        order: int = 1,
    ) -> HexMesh:
        """Build a **linear** ``HexMesh`` from corner ``points`` ``(P,3)`` + Nek-order
        ``hexes`` ``(E,8)`` connectivity -- the corner -> B-rep bridge every factory
        routes through.  Decomposes the shared faces with ``conform.canonical_faces``
        (lossless, so ``.hexes`` round-trips the input exactly).

        Corners are all a linear mesh has, so this is an ``order == 1`` constructor:
        at ``order > 1`` there is no way to invent the shared edge / face / private
        interior nodes without silently straight-subdividing, so it raises.  Build with
        a factory (``extrude`` / ``loft`` / ``annulus`` / ``from_grid``) at ``order=N``,
        which places those nodes on the true geometry, or construct
        ``HexMesh(quads, hex, face_orient, interior, ..., order=N)`` directly from the
        entity fields if you already hold them."""
        if order != 1:
            raise ValueError(
                "HexMesh.from_corners builds the linear (order 1) B-rep from corner "
                "points alone; got order=%d, whose shared edge / face / private "
                "interior nodes it cannot know. Build with a factory at order=%d "
                "(e.g. HexMesh.extrude(section, ...) / HexMesh.loft(slices) from "
                "order-%d sections), which places those nodes on the true geometry, "
                "or construct HexMesh(quads, hex, face_orient, interior=..., "
                "order=%d) directly from the entity fields (the shared-face QuadMesh "
                "carries the edge / face nodes)."
                % (order, order, order, order))
        pts: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
        conn: IntArray = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
        canonical_conn, elem_faces, face_orient = conform.canonical_faces(conn)
        quads = QuadMesh.from_corners(pts, canonical_conn)
        return cls(quads, elem_faces, face_orient, None, boundaries,
                   boundary_tags, element_tags, order=1)

    def _derive_corners(self) -> IntArray:
        """Corner connectivity ``(E,8)`` (Nek order) recovered from the shared faces via
        ``conform.hex_corners_from_faces`` -- the lossless inverse of
        ``conform.canonical_faces``, so it reproduces the connectivity the mesh was built
        from byte-for-byte."""
        return conform.hex_corners_from_faces(
            self.quads.quads, self.hex, self.face_orient)

    @property
    def order(self) -> int:
        """Global polynomial order (1 = linear)."""
        return self._order

    @property
    def points(self) -> PointArray:
        """The ``(P,3)`` shared corner points -- a live view of the shared-face
        ``QuadMesh``'s ``points`` (the single source of truth), so an in-place edit
        (``mesh.points[:] = X``) moves the shared corners for every hex."""
        return self.quads.points

    @property
    def hexes(self) -> IntArray:
        """``(E,8)`` Nek-order corner connectivity, derived (memoized) from the shared
        faces.  Read-only; the B-rep ``quads`` / ``hex`` / ``face_orient`` are the source
        of truth."""
        return self._corners

    @property
    def edges(self) -> IntArray:
        """``(Ne,2)`` unique undirected hex edges (canonical: min corner id first) -- the
        shared edge topology (the ``edges`` of the shared-face ``QuadMesh``).  Non-empty
        at every order (edges are first-class B-rep storage)."""
        return self.quads.edges

    @property
    def edge_nodes(self) -> PointArray:
        """``(Ne, order-1, 3)`` shared high-order interior nodes of each unique
        :attr:`edges` entry, in canonical (min->max corner) order.  Empty at order 1."""
        return self.quads.edge_nodes

    @property
    def faces(self) -> IntArray:
        """``(Nf,4)`` unique hex faces (canonical: sorted corner ids) -- the shared face
        topology.  Non-empty at every order (faces are first-class B-rep storage)."""
        return np.sort(self.quads.quads, axis=1)

    @property
    def face_nodes(self) -> PointArray:
        """``(Nf, (order-1)**2, 3)`` shared high-order interior nodes of each unique
        :attr:`faces` entry, in the canonical D4-normalized frame.  Empty at order 1; a
        shared face resolves to the same nodes from either incident hex."""
        return self.quads.interior

    # -- sizes -----------------------------------------------------------
    @property
    def n_hexes(self) -> int:
        """Number of hexahedra."""
        return self.hexes.shape[0]

    @property
    def n_points(self) -> int:
        """Number of (shared) points."""
        return self.points.shape[0]

    @property
    def n_boundaries(self) -> int:
        """Number of tagged boundary faces."""
        return self.boundaries.shape[0]

    @property
    def boundary_group_tags(self) -> list[str]:
        """Sorted unique tags of the tagged boundary faces."""
        return sorted(set(self.boundary_tags.tolist()))

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-hex element tags present on the mesh."""
        return sorted({t for t in self.element_tags.tolist() if t})

    # -- helpers for the operation modules -----------------------------
    @staticmethod
    def _cap_tags(cap: str | Sequence[str] | StrArray, M: int) -> list[str]:
        """Normalize a cap tag to one tag per section quad (length ``M``): a scalar
        ``str`` tags the whole cap, an array-like is per-quad."""
        if isinstance(cap, str):
            return [cap] * M
        arr = np.asarray(cap, dtype=np.str_).reshape(-1)
        if arr.shape[0] != M:
            raise ValueError("cap tags length (%d) must match section quads (%d)"
                             % (arr.shape[0], M))
        return [str(x) for x in arr.tolist()]

    @staticmethod
    def _signed_vol(P: PointArray) -> float:
        """Sign proxy of the trilinear Jacobian at the hex centre (Nek order)."""
        P = np.asarray(P, dtype=float)
        r = P[[1, 2, 5, 6], :].mean(axis=0) - P[[0, 3, 4, 7], :].mean(axis=0)
        s = P[[2, 3, 6, 7], :].mean(axis=0) - P[[0, 1, 4, 5], :].mean(axis=0)
        t = P[[4, 5, 6, 7], :].mean(axis=0) - P[[0, 1, 2, 3], :].mean(axis=0)
        return float(np.dot(np.cross(r, s), t))

    @staticmethod
    def _order_bnd(
        bnd: Sequence[Sequence[int]] | IntArray,
        names: Sequence[str] | StrArray,
    ) -> tuple[IntArray, StrArray]:
        """Stably order boundary rows by ``(element id, face)``, applying the same
        permutation to the parallel ``names`` array."""
        b: IntArray = np.asarray(bnd, dtype=np.int64).reshape(-1, 2)
        nm: StrArray = np.asarray(names, dtype=np.str_).reshape(-1)
        if b.shape[0]:
            order = np.lexsort((b[:, 1], b[:, 0]))
            b = b[order]
            nm = nm[order]
        return b, nm
