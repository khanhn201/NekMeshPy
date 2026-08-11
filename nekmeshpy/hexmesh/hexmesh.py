"""All-hex mesh container."""

from __future__ import annotations

import numpy as np

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
)
from ..core import conform
from ..core.tags import ElementTags
from ..linemesh.linemesh import _repr_tags
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
def _sweep_at(bottom: PointArray, top: PointArray, g: FloatArray,
              slots: IntArray, m2: int) -> PointArray:
    """A loft column's straight GLL sweep evaluated at the hex block ``slots``."""
    k = slots // m2
    m = slots % m2
    gg = g[k].reshape(1, -1, 1)
    return (1.0 - gg) * bottom[:, m, :] + gg * top[:, m, :]


class HexMesh:
    """An all-hexahedral volume mesh in shared-point form."""

    # Nek face -> the 4 corner point positions (0-based); row f is face f+1.
    FACE_POINTS = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6],
                           [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)

    def __init__(
        self,
        quads: QuadMesh,
        hex: IntArray,
        face_orient: IntArray,
        interior: PointArray | None = None,
        element_tags: ElementTags | None = None,
    ) -> None:
        """Construct from the B-rep directly: ``quads`` (a ``QuadMesh`` holding every
        shared face -- its ``points`` are the shared corners, its ``quads`` the shared
        face connectivity, its edges / ``interior`` the shared face-boundary / interior
        HO nodes), ``hex`` ``(E,6)`` face indices into ``quads.quads`` (Nek local-face
        order), ``face_orient`` ``(E,6)`` D4 codes (element-local face frame ->
        canonical), and ``interior`` ``(E,(order-1)**3,3)`` private per-hex nodes (omit
        / ``None`` at order 1)."""
        if not isinstance(quads, QuadMesh):
            raise TypeError("HexMesh: quads must be a QuadMesh, got %s"
                            % type(quads).__name__)
        self.quads = quads

        self.hex: IntArray = np.asarray(hex, dtype=np.int64).reshape(-1, 6)
        self.face_orient: IntArray = np.asarray(
            face_orient, dtype=np.int64).reshape(-1, 6)
        if self.face_orient.shape[0] != self.hex.shape[0]:
            raise ValueError("HexMesh: face_orient length (%d) must match hex (%d)"
                             % (self.face_orient.shape[0], self.hex.shape[0]))
        F = quads.n_quads                        # the rung below: shared faces
        if self.hex.size and (self.hex.min() < 0 or self.hex.max() >= F):
            raise ValueError(
                "HexMesh: hex must index the %d shared faces of ``quads``; got ids in "
                "[%d, %d]" % (F, int(self.hex.min()), int(self.hex.max())))
        E = self.hex.shape[0]

        if interior is None:
            if self.order > 1:
                raise ValueError(
                    "HexMesh: order %d > 1 requires the per-hex private interior "
                    "nodes (pass interior=(E,(order-1)**3,3), or build the block "
                    "with a factory such as HexMesh.extrude(section, ...) from an "
                    "order-%d section)" % (self.order, self.order))
            interior = np.zeros((E, 0, 3), dtype=float)
        self.interior: PointArray = np.asarray(interior, dtype=float)
        k = (self.order - 1) ** 3
        if self.interior.shape != (E, k, 3):
            raise ValueError(
                "HexMesh: interior must be (E,(order-1)**3,3) = (%d,%d,3), got %s"
                % (E, k, self.interior.shape))

        self.element_tags = ElementTags.empty() if element_tags is None else element_tags
        self.element_tags.check_within(E)

        # Corner connectivity is derived from the shared faces and immutable
        # post-construction (point moves don't change it), so memoize it once.
        self._corners: IntArray = self._derive_corners()
        # The per-hex edge incidence is the same kind of read -- never a fresh dedup, the
        # ids come out of ``quads``' own table -- but only export, quality and ``merge``'s
        # gather ever ask for it, so it is deferred rather than paid by every
        # construction.  ``merge`` in particular builds its own and would never look.
        self._edge_tables: tuple[IntArray, BoolArray] | None = None

    def _edge_incidence(self) -> tuple[IntArray, BoolArray]:
        """``(elem_edges (E,12), edge_flip (E,12))``, read off the shared faces on first
        ask and kept."""
        if self._edge_tables is None:
            self._edge_tables = conform.hex_edges_from_faces(
                self.hex, self.face_orient, self.quads.quad, self.quads.flip)
        return self._edge_tables

    @property
    def _elem_edges(self) -> IntArray:
        """``(E,12)`` per-hex incidence on the shared edge table."""
        return self._edge_incidence()[0]

    @property
    def _edge_flip(self) -> BoolArray:
        """``(E,12)`` ``True`` where the hex walks that edge against its stored row."""
        return self._edge_incidence()[1]

    @classmethod
    def from_corners(
        cls,
        points: PointArray,
        hexes: IntArray,
        element_tags: ElementTags | None = None,
        *,
        order: int = 1,
    ) -> HexMesh:
        """Build a **linear** ``HexMesh`` from corner ``points`` ``(P,3)`` + Nek-order
        ``hexes`` ``(E,8)`` connectivity -- the corner -> B-rep bridge every factory
        routes through."""
        if order != 1:
            raise ValueError(
                "HexMesh.from_corners builds the linear (order 1) B-rep from corner "
                "points alone; got order=%d, whose shared edge / face / private "
                "interior nodes it cannot know. Build with a factory at order=%d "
                "(e.g. HexMesh.extrude(section, ...) / HexMesh.loft(slices) from "
                "order-%d sections), which places those nodes on the true geometry, "
                "or construct HexMesh(quads, hex, face_orient, interior=...) directly "
                "from the entity fields (the shared-face QuadMesh carries the edge / "
                "face nodes, and the order with them)."
                % (order, order, order))
        pts: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
        conn: IntArray = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
        canonical_conn, elem_faces, face_orient = conform.canonical_faces(conn)
        quads = QuadMesh.from_corners(pts, canonical_conn)
        return cls(quads, elem_faces, face_orient, None, element_tags)

    @property
    def face_tags(self) -> ElementTags:
        """The tags on the shared faces, over **face ids**.

        This is ``quads``' own ``element_tags`` read through, not a table of its own:
        rung *N*'s side tags are rung *N-1*'s element tags. A face is one stored object
        both its hexes reference, so naming it is naming that object -- which is why
        there is no ``(hex, side)`` here for the two sides to disagree over. The
        ``(element, face)`` rows the ``.re2`` boundary block wants are reconstructed at
        export from the hexes that carry each named face (see :func:`face_tag_rows
        <nekmeshpy.hexmesh.query.face_tag_rows>`)."""
        return self.quads.element_tags

    def _derive_corners(self) -> IntArray:
        """Corner connectivity ``(E,8)`` (Nek order) recovered from the shared faces via
        ``conform.hex_corners_from_faces`` -- the lossless inverse of
        ``conform.canonical_faces``, so it reproduces the connectivity the mesh was built
        from byte-for-byte."""
        return conform.hex_corners_from_faces(
            self.quads.quads, self.hex, self.face_orient)

    def __repr__(self) -> str:
        """One-line REPL summary: element / point counts, ``order``, and the tag
        vocabulary -- the same field set :class:`LineMesh <nekmeshpy.linemesh.LineMesh>`
        and :class:`QuadMesh <nekmeshpy.quadmesh.QuadMesh>` render, so the three read as
        a family."""
        try:
            return ("<HexMesh %d points, %d hexes, order %d, element_tags=%s, "
                    "face_tags=%s>"
                    % (self.quads.lines.points.shape[0], self.hex.shape[0], self.order,
                       _repr_tags(self.element_group_tags),
                       _repr_tags(self.face_group_tags)))
        except Exception:                     # a repr must never break a debug session
            return "<HexMesh (unprintable)>"

    @property
    def order(self) -> int:
        """Global polynomial order (1 = linear)."""
        return self.quads.order

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
    def n_points(self) -> int:
        """Number of (shared) points."""
        return self.points.shape[0]

    @property
    def n_hexes(self) -> int:
        """Number of hexahedra."""
        return self.hexes.shape[0]

    @property
    def n_face_tags(self) -> int:
        """Number of tagged faces."""
        return len(self.face_tags)

    @property
    def face_group_tags(self) -> list[str]:
        """Sorted unique tags of the tagged faces."""
        return self.face_tags.group_tags

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-hex element tags present on the mesh."""
        return self.element_tags.group_tags

    # -- helpers for the operation modules -----------------------------
    @staticmethod
    def _signed_vol(P: PointArray) -> float:
        """Sign proxy of the trilinear Jacobian at the hex centre (Nek order)."""
        P = np.asarray(P, dtype=float)
        r = P[[1, 2, 5, 6], :].mean(axis=0) - P[[0, 3, 4, 7], :].mean(axis=0)
        s = P[[2, 3, 6, 7], :].mean(axis=0) - P[[0, 1, 4, 5], :].mean(axis=0)
        t = P[[4, 5, 6, 7], :].mean(axis=0) - P[[0, 1, 2, 3], :].mean(axis=0)
        return float(np.dot(np.cross(r, s), t))

