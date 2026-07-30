"""All-hex mesh container.

``HexMesh`` stores ``points`` ``(P,3)``, ``hexes`` ``(N,8)`` connectivity in Nek
order, a sparse tagged ``boundaries`` ``(Nbc,2)`` = ``[element id, face 1-6]`` with
parallel ``boundary_tags``, and a dense per-hex ``element_tags``. Boundary tags map
to Nek BC codes only at export.

It is built complete, not incrementally: from arrays or via the factory
classmethods ``loft`` / ``extrude`` / ``annulus`` / ``merge`` / ``from_grid``. The
topology is fixed at construction, but coordinates may be repositioned in place.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .._typing import (
    BoolArray,
    CurvedBlock,
    FloatArray,
    IntArray,
    Point,
    PointArray,
    StrArray,
    Vec3,
)
from ..linemesh import LineMesh
from ..model import conform
from ..model.fields import gll_nodes, validate_layers
from ..model.interp import (
    blend_ho,
    corner_indices,
    subdivide_hexes,
)
from ..quadmesh import NO_BOUNDARY, QuadMesh

# default sweep axis / origin for extrude
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_ORIGIN = np.array([0.0, 0.0, 0.0])

# side name -> (Nek face number, axis, which end) for structured grids
_GRID_SIDES = {
    "x_min": (4, 0, 0), "x_max": (2, 0, -1),
    "y_min": (1, 1, 0), "y_max": (3, 1, -1),
    "z_min": (5, 2, 0), "z_max": (6, 2, -1),
}


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
        interior: FloatArray | None = None,
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

        ``.points`` / ``.hexes`` / ``.curved`` are **derived** views over this B-rep, so
        a shared face is literally one stored object referenced by every incident hex
        (structural conformality).  Prefer :meth:`from_corners` to build from corner
        points + hex connectivity; the factories all route through it.  ``re2`` export
        stays linear; only ``vtu`` reads the curved nodes."""
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
                    "HexMesh: order %d > 1 requires interior nodes" % self._order)
            self.interior: FloatArray = np.zeros((E, 0, 3), dtype=float)
        else:
            ia: FloatArray = np.asarray(interior, dtype=float)
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
        self._hexes: IntArray = self._derive_hexes()
        _, self._elem_edges, self._edge_flip = conform.unique_edges(self._hexes, 3)

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
        curved: CurvedBlock | None = None,
    ) -> HexMesh:
        """Build a ``HexMesh`` from corner ``points`` ``(P,3)`` + Nek-order ``hexes``
        ``(E,8)`` connectivity -- the corner -> B-rep bridge every factory routes
        through.  Decomposes the shared faces with ``conform.canonical_faces`` (lossless,
        so ``.hexes`` round-trips the input exactly) and, at ``order > 1``, validates +
        scatters the ``curved`` block ``(E,(order+1)**3,3)`` via ``conform.split`` (shape
        + corner-consistency, owner-wins edge / face nodes).  Same signature and
        semantics as the old array constructor."""
        pts: PointArray = np.asarray(points, dtype=float).reshape(-1, 3)
        conn: IntArray = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
        canonical_conn, elem_faces, face_orient = conform.canonical_faces(conn)
        interior: FloatArray | None
        if order > 1:
            t = conform.split(order, curved, pts, conn, 3, "HexMesh")
            q_edges, q_elem_edges, q_flip = conform.unique_edges(canonical_conn, 2)
            eb: CurvedBlock = np.empty((q_edges.shape[0], order + 1, 3), dtype=float)
            eb[:, corner_indices(order, 1), :] = pts[q_edges]
            eb[:, 1:order, :] = t.edge_nodes
            edge_lm = LineMesh(pts, q_edges, order=order, curved=eb)
            quads = QuadMesh(edge_lm, q_elem_edges, q_flip, interior=t.face_nodes,
                             order=order)
            interior = t.interior
        else:
            # order 1: split still validates curved (corner-consistency) but returns
            # empty tables; the shared-face topology comes from canonical_faces.
            conform.split(order, curved, pts, conn, 3, "HexMesh")
            quads = QuadMesh.from_corners(pts, canonical_conn)
            interior = None
        return cls(quads, elem_faces, face_orient, interior, boundaries,
                   boundary_tags, element_tags, order=order)

    def _derive_hexes(self) -> IntArray:
        """Corner connectivity ``(E,8)`` (Nek order) recovered from the shared faces via
        ``conform.hex_corners_from_faces`` -- the lossless inverse of
        ``conform.canonical_faces``, so it reproduces the connectivity the mesh was built
        from byte-for-byte."""
        return conform.hex_corners_from_faces(
            self.quads.quads, self.hex, self.face_orient)

    def _tables(self) -> conform.EntityTables:
        """A transient :class:`~nekmeshpy.model.conform.EntityTables` assembled from the
        stored B-rep fields -- the vehicle for the tested ``assemble`` / ``to_conformal``
        readers (not storage).  Edges / faces come from the shared-face ``QuadMesh``; the
        per-hex edge incidence is the cached ``unique_edges`` of the derived corners."""
        return conform.EntityTables(
            order=self._order, dim=3,
            edges=self.quads.edges,
            edge_nodes=self.quads.edge_nodes,
            elem_edges=self._elem_edges,
            edge_flip=self._edge_flip,
            faces=np.sort(self.quads.quads, axis=1),
            face_nodes=self.quads.interior,
            elem_faces=self.hex,
            face_orient=self.face_orient,
            interior=self.interior)

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
        return self._hexes

    @property
    def curved(self) -> CurvedBlock:
        """The full high-order node block ``(N, (order+1)**3, 3)`` in lexicographic GLL
        order (``i`` fastest), reassembled on read from the authoritative corners
        ``points[hexes]`` and the stored non-corner nodes -- so corners are never
        duplicated and an in-place ``points`` edit is reflected.  At order 1 it holds
        the 8 corners."""
        return conform.assemble(self._tables(), self.points, self.hexes)

    @property
    def edges(self) -> IntArray:
        """``(Ne,2)`` unique undirected hex edges (canonical: min corner id first) -- the
        shared edge topology (the ``edges`` of the shared-face ``QuadMesh``).  Non-empty
        at every order (edges are first-class B-rep storage)."""
        return self.quads.edges

    @property
    def edge_nodes(self) -> CurvedBlock:
        """``(Ne, order-1, 3)`` shared high-order interior nodes of each unique
        :attr:`edges` entry, in canonical (min->max corner) order.  Empty at order 1."""
        return self.quads.edge_nodes

    @property
    def faces(self) -> IntArray:
        """``(Nf,4)`` unique hex faces (canonical: sorted corner ids) -- the shared face
        topology.  Non-empty at every order (faces are first-class B-rep storage)."""
        return np.sort(self.quads.quads, axis=1)

    @property
    def face_nodes(self) -> CurvedBlock:
        """``(Nf, (order-1)**2, 3)`` shared high-order interior nodes of each unique
        :attr:`faces` entry, in the canonical D4-normalized frame.  Empty at order 1; a
        shared face resolves to the same nodes from either incident hex."""
        return self.quads.interior

    def to_conformal(self) -> tuple[PointArray, IntArray]:
        """Conformal high-order view ``(nodes (M,3), conn (N,(order+1)**3))``: every node
        (corner, shared edge / face-interior, private cell-interior) numbered once in one
        global array with dense per-hex connectivity into it -- the high-order analog of
        ``points`` + ``hexes``.  Shared edges and faces resolve to the same node ids from
        every incident hex.  At order 1 this is ``points`` + ``hexes`` in block order."""
        return conform.to_conformal(self._tables(), self.points, self.hexes)

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

    # -- quality ---------------------------------------------------------
    def scaled_jacobian(self, *, high_order: bool = False) -> FloatArray:
        """Per-hex minimum scaled Jacobian ``(n_hexes,)``.

        Defaults to the corner metric (the pinned linear numbers).  With
        ``high_order=True`` it is sampled at the ``(order+1)**3`` GLL nodes of the
        curved block (:func:`~nekmeshpy.hexmesh.quality.scaled_jacobian_ho`); at order
        1 the two agree."""
        from . import quality
        if high_order:
            return quality.scaled_jacobian_ho(self.curved, self.order)
        return quality.scaled_jacobian(self.points, self.hexes)

    def quality_summary(self, *, high_order: bool = False) -> dict[str, Any]:
        """Aggregate scaled-Jacobian statistics (see :meth:`scaled_jacobian` for the
        ``high_order`` flag)."""
        from . import quality
        if high_order:
            return quality.summary_ho(self.curved, self.order)
        return quality.summary(self.points, self.hexes)

    # -- orientation -----------------------------------------------------
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

    # -- factories -------------------------------------------------------
    @classmethod
    def extrude(
        cls,
        section: QuadMesh,
        *,
        axis: Vec3 = _Z_AXIS,
        length: float,
        layers: FloatArray,
        origin: Point = _ORIGIN,
        first_tag: str | Sequence[str] | StrArray = "",
        last_tag: str | Sequence[str] | StrArray = "",
    ) -> HexMesh:
        """Sweep a single quad ``section`` a distance ``length`` along ``axis`` into
        a hex block.

        The section is translated rigidly along ``axis`` (its placement is
        preserved); ``origin`` shifts the whole block. ``layers`` are the normalized
        copy-plane positions in ``[0, 1]``, strictly increasing, last ``1``;
        ``layers[0]`` is the near cap and ``layers.size - 1`` hex layers span it to
        ``1``. ``first_tag`` / ``last_tag`` name the caps. The straight special case
        of ``loft``."""
        base = np.asarray(section.points, dtype=float).reshape(-1, 3) \
            + np.asarray(origin, dtype=float)
        axis_u: Vec3 = np.asarray(axis, dtype=float)
        axis_u = axis_u / np.linalg.norm(axis_u)
        offsets = validate_layers(layers, "extrude layers") * float(length)
        sc = None if section.curved is None else np.asarray(section.curved, float)
        slices = [QuadMesh.from_corners(
                      base + d * axis_u[None, :],
                      section.quads, boundaries=section.boundaries,
                      boundary_tags=section.boundary_tags,
                      element_tags=section.element_tags,
                      order=section.order,
                      curved=None if sc is None else sc + d * axis_u[None, None, :])
                  for d in offsets]
        return cls.loft(slices, first_tag=first_tag, last_tag=last_tag)

    @classmethod
    def loft(
        cls,
        slices: Sequence[QuadMesh],
        *,
        first_tag: str | Sequence[str] | StrArray = "",
        last_tag: str | Sequence[str] | StrArray = "",
    ) -> HexMesh:
        """Loft a stack of conformal quad profiles into a hex block (the general
        primitive behind ``extrude``).

        ``slices`` is ``nz+1`` profiles sharing the same quad connectivity,
        ``boundary_tags``, and ``element_tags``; consecutive profiles form ``nz`` hex
        layers. ``first_tag`` names the first bottom cap (face 5), ``last_tag`` the
        last top cap (face 6) -- each a scalar or a per-quad array. Side faces are
        named from the section's ``boundary_tags`` (unnamed or ``NO_BOUNDARY`` edges
        stay untagged), and every hex inherits its quad's ``element_tags``. Points
        are shared by construction."""
        slices = list(slices)
        quads = np.asarray(slices[0].quads, dtype=np.int64).reshape(-1, 4)
        # section (quad, side) -> name; each swept side face inherits its section edge
        sec_bnd = np.asarray(slices[0].boundaries, dtype=np.int64).reshape(-1, 2)
        sec_tags = slices[0].boundary_tags
        side_name: dict[tuple[int, int], str] = {
            (int(sec_bnd[r, 0]), int(sec_bnd[r, 1])): str(sec_tags[r])
            for r in range(sec_bnd.shape[0])}
        tag_sides = bool(side_name)
        qtag = np.asarray(slices[0].element_tags, dtype=np.str_).reshape(-1)
        M = quads.shape[0]
        nz = len(slices) - 1
        S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                      for s in slices], axis=0)             # (nz+1, nn, 3)
        nn = S.shape[1]
        points = S.reshape((nz + 1) * nn, 3)                 # global id = i*nn + v

        # Decide handedness once from the first layer and flip the quad template if
        # left-handed; reject a mixed-winding section rather than invert elements.
        signs = np.array([cls._signed_vol(np.vstack([S[0, quads[q], :], S[1, quads[q], :]]))
                          for q in range(M)]) if nz else np.zeros(0)
        if nz and not (np.all(signs > 0) or np.all(signs < 0)):
            raise ValueError(
                "extrude: section is not consistently wound (mixed hex "
                "orientation) -- the section mesher must emit uniform winding")
        flip = bool(nz and signs[0] < 0)
        qw = quads[:, [0, 3, 2, 1]] if flip else quads

        # caps stay faces 5/6 by q (the flip only reorders a quad's 4 corners)
        first_caps = cls._cap_tags(first_tag, M)
        last_caps = cls._cap_tags(last_tag, M)

        hexes = np.empty((nz * M, 8), dtype=np.int64)
        etags: StrArray = np.empty(nz * M, dtype=np.str_)
        bnd: list[list[int]] = []
        names: list[str] = []
        e = 0
        for i in range(nz):
            for q in range(M):
                v = qw[q, :]
                hexes[e] = np.concatenate([i * nn + v, (i + 1) * nn + v])
                etags[e] = qtag[q] if qtag.size else ""
                if tag_sides:
                    # section side s -> hex face s, or 5-s when the quad was flipped
                    for s in (1, 2, 3, 4):
                        nm = side_name.get((q, s))
                        if nm is None or nm == NO_BOUNDARY:
                            continue
                        bnd.append([e, (5 - s) if flip else s])
                        names.append(nm)
                if i == 0 and first_caps[q]:
                    bnd.append([e, 5])
                    names.append(first_caps[q])
                if i == nz - 1 and last_caps[q]:
                    bnd.append([e, 6])
                    names.append(last_caps[q])
                e += 1

        order = slices[0].order
        if any(s.order != order for s in slices):
            raise ValueError("loft: all slices must share the same order")
        curved: CurvedBlock | None = None
        if order > 1:
            g = gll_nodes(order)
            row = order + 1
            m2 = row * row
            SC = np.stack([np.asarray(s.curved, dtype=float) for s in slices], axis=0)
            # (nz, M, m2, 3) in-plane blocks of the bottom/top slice of each layer,
            # flattened to hex order e = i*M + q
            bottom = SC[:-1].reshape(nz * M, m2, 3)
            top = SC[1:].reshape(nz * M, m2, 3)
            if flip:
                kk = np.arange(m2)
                trans = (kk // row) + row * (kk % row)    # transpose the in-plane grid
                bottom = bottom[:, trans, :]
                top = top[:, trans, :]
            # straight GLL sweep between the two in-plane blocks: (E, row_k, m2, 3)
            gg = g.reshape(1, row, 1, 1)
            block = (1.0 - gg) * bottom[:, None, :, :] + gg * top[:, None, :, :]
            # lexicographic hex order: in-plane index m fastest, sweep k slowest
            curved = block.reshape(nz * M, row * m2, 3)
            curved[:, corner_indices(order, 3), :] = points[hexes]
        return cls.from_corners(points, hexes, *cls._order_bnd(bnd, names),
                                element_tags=etags, order=order, curved=curved)

    @classmethod
    def annulus(
        cls,
        inner: QuadMesh,
        outer: QuadMesh,
        radial: FloatArray,
        *,
        inner_tag: str = "",
        outer_tag: str = "",
    ) -> HexMesh:
        """Shell O-grid filling the region between an inner and an outer closed quad
        surface (e.g. a sphere inside a cubic far-field box).

        The two surfaces are paired by index: equal point count ``P`` and identical
        ``quads`` connectivity, with point ``p`` of ``inner`` joined radially to point
        ``p`` of ``outer``. ``radial`` are the shell positions in ``[0, 1]``, strictly
        increasing; ``radial[0]`` is the inner shell and the last is ``1``, so
        ``radial.size - 1`` shell layers blend inner -> outer directly in 3-D.

        Wall faces are tagged from the surfaces' per-quad ``element_tags`` (a closed
        surface has no free boundary edges): inner caps face 5, outer caps face 6. A
        non-empty scalar ``inner_tag`` / ``outer_tag`` overrides and names the whole
        wall."""
        radial = validate_layers(radial, "annulus radial")
        A: FloatArray = np.asarray(inner.points, dtype=float).reshape(-1, 3)
        B: FloatArray = np.asarray(outer.points, dtype=float).reshape(-1, 3)
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "annulus: inner and outer surfaces must have equal point counts "
                "(got %d, %d); build one from the other's points so they pair by "
                "index" % (A.shape[0], B.shape[0]))
        if not np.array_equal(inner.quads, outer.quads):
            raise ValueError(
                "annulus: inner and outer surfaces must share identical quad "
                "connectivity (they are paired by index)")
        if float(np.min(np.linalg.norm(B - A, axis=1))) <= 0.0:
            raise ValueError("annulus: inner and outer surfaces touch or cross")

        # shell t is the straight-chord blend inner -> outer sharing inner's quads;
        # consecutive shells loft into hex layers.
        shells = QuadMesh.blend(inner, outer, radial)
        # wall tags from the surfaces' per-quad element_tags; scalar arg overrides
        inner_caps: str | StrArray = (
            inner_tag if inner_tag
            else (inner.element_tags if inner.element_group_tags else ""))
        outer_caps: str | StrArray = (
            outer_tag if outer_tag
            else (outer.element_tags if outer.element_group_tags else ""))
        return cls.loft(shells, first_tag=inner_caps, last_tag=outer_caps)

    @classmethod
    def merge(
        cls,
        meshes: Sequence[HexMesh],
        *,
        tol: float | None = None,
    ) -> HexMesh:
        """Stitch several hex blocks into one, coordinate-welding coincident seam
        points in a single pass.  ``tol`` is the absolute coincidence distance
        (default ``1e-7`` x the merged bounding-box extent).

        Only points on each block's domain boundary (faces carried by a single hex)
        are weld candidates; interior points are always kept distinct."""
        meshes = list(meshes)
        pos = [m.points for m in meshes]
        counts = [p.shape[0] for p in pos]
        P = np.concatenate(pos, axis=0) if pos else np.zeros((0, 3))
        total = P.shape[0]

        # remap: concat point index -> representative concat index (self by default)
        remap = np.arange(total, dtype=np.int64)
        is_bnd: BoolArray = np.zeros(total, dtype=bool)
        noff = 0
        for m, c in zip(meshes, counts):
            is_bnd[noff + cls._boundary_points(m.hexes)] = True
            noff += c
        bidx = np.flatnonzero(is_bnd)
        if bidx.size:
            scl = float(np.max(P.max(axis=0) - P.min(axis=0)))
            t = tol if tol is not None else (1e-7 * scl if scl > 0 else 1.0)
            keys = np.round(P[bidx, :] / t).astype(np.int64)
            _, first_local, inverse = np.unique(
                keys, axis=0, return_index=True, return_inverse=True)
            remap[bidx] = bidx[first_local][inverse.ravel()]

        survivors = np.unique(remap)                    # concat indices kept
        new_id: IntArray = np.empty(total, dtype=np.int64)
        new_id[survivors] = np.arange(survivors.size)
        point_id = new_id[remap]                         # concat index -> final id
        points = P[survivors, :]

        hex_list, bnd_list, name_list, etag_list = [], [], [], []
        noff = eoff = 0
        for m, c in zip(meshes, counts):
            hex_list.append(point_id[m.hexes + noff])    # local -> concat -> welded id
            etag_list.append(np.asarray(m.element_tags, dtype=np.str_).reshape(-1))
            if m.boundaries.shape[0]:
                b: IntArray = m.boundaries.copy()
                b[:, 0] += eoff
                bnd_list.append(b)
                name_list.append(m.boundary_tags)
            noff += c
            eoff += m.hexes.shape[0]
        hexes = (np.concatenate(hex_list, axis=0) if hex_list
                 else np.zeros((0, 8), np.int64))
        etags = (np.concatenate(etag_list) if etag_list
                 else np.empty(0, dtype=np.str_))
        bnd = np.concatenate(bnd_list, axis=0) if bnd_list else np.zeros((0, 2), np.int64)
        names = (np.concatenate(name_list) if name_list
                 else np.empty(0, dtype=np.str_))
        order = meshes[0].order if meshes else 1
        if any(mm.order != order for mm in meshes):
            raise ValueError("merge: all blocks must share the same order")
        curved: CurvedBlock | None = None
        if order > 1:
            curved = np.concatenate(
                [np.asarray(mm.curved, dtype=float) for mm in meshes], axis=0)
            curved[:, corner_indices(order, 3), :] = points[hexes]
        return cls.from_corners(points, hexes, *cls._order_bnd(bnd, names),
                                element_tags=etags, order=order, curved=curved)

    @classmethod
    def blend(cls, a: HexMesh, b: HexMesh,
              fractions: FloatArray | Sequence[float]) -> list[HexMesh]:
        """Linearly morph between two conformal blocks ``a`` and ``b`` (identical
        ``hexes``, equal point count), one block per fraction ``t`` with points
        ``(1-t)*a + t*b`` -- ``t=0`` reproduces ``a``, ``t=1`` reproduces ``b``.  Each
        result carries ``a``'s ``hexes``, ``boundaries`` and ``boundary_tags``
        (positional BC markers follow the morph); per-hex ``element_tags`` are left
        for the caller to assign.  The 3-D sibling of
        :meth:`QuadMesh.blend <nekmeshpy.quadmesh.QuadMesh.blend>`."""
        A: PointArray = np.asarray(a.points, dtype=float).reshape(-1, 3)
        B: PointArray = np.asarray(b.points, dtype=float).reshape(-1, 3)
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "blend: blocks must have equal point counts (got %d, %d); build one "
                "from the other's points so they pair by index"
                % (A.shape[0], B.shape[0]))
        if not np.array_equal(a.hexes, b.hexes):
            raise ValueError(
                "blend: blocks must share identical connectivity (paired by index)")
        if a.order != b.order:
            raise ValueError("blend: blocks must share the same order")
        out: list[HexMesh] = []
        for t in np.asarray(fractions, dtype=float).ravel():
            cb = blend_ho(a.curved, b.curved, float(t))
            out.append(cls.from_corners((1.0 - t) * A + t * B, a.hexes,
                                        a.boundaries, a.boundary_tags,
                                        order=a.order, curved=cb))
        return out

    # -- boundary queries (topological domain surface) ------------------
    @staticmethod
    def _boundary_mask(hexes: IntArray) -> tuple[IntArray, BoolArray]:
        """``(faces, is_boundary)``: every hex quad face ``(6N,4)`` in Nek order,
        element-major (row ``6e+f``), and a mask of those on the domain boundary."""
        HC = np.asarray(hexes, dtype=np.int64).reshape(-1, 8)
        faces: IntArray = HC[:, HexMesh.FACE_POINTS].reshape(-1, 4)
        keys = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        return faces, counts[inverse.ravel()] == 1

    @staticmethod
    def _boundary_points(hexes: IntArray) -> IntArray:
        faces, mask = HexMesh._boundary_mask(hexes)
        bf = faces[mask]
        return np.unique(bf) if bf.size else np.zeros(0, dtype=np.int64)

    def boundary_faces(self) -> IntArray:
        """``(K,2)`` of ``[element id, local face (1-6)]`` for every face on the
        topological domain boundary (a quad carried by a single hex). Distinct from
        the tagged ``boundaries``, which may also carry interior planes."""
        _, mask = self._boundary_mask(self.hexes)
        rows = np.flatnonzero(mask)
        return np.column_stack([rows // 6, rows % 6 + 1]).astype(np.int64)

    def boundary_elements(self) -> IntArray:
        """Sorted unique element ids with at least one face on the domain boundary."""
        return np.unique(self.boundary_faces()[:, 0])

    def boundary_points(self) -> IntArray:
        """Sorted unique point ids lying on the domain boundary."""
        return self._boundary_points(self.hexes)

    @classmethod
    def from_grid(
        cls,
        P: FloatArray,
        *,
        face_tags: dict[str, str] | None = None,
        element_tag: str = "",
        order: int = 1,
    ) -> HexMesh:
        """Build hexes from a structured point grid ``P`` ``(ni+1,nj+1,nk+1,3)``.
        ``face_tags`` maps side names (``x_min``/``x_max``/``y_min``/``y_max``/
        ``z_min``/``z_max``) to boundary names on the six outer sides; a side left out
        or mapped to ``NO_BOUNDARY`` emits no boundary row. ``element_tag`` is written
        to every hex's ``element_tags``.

        ``order`` (default 1 = linear) sets the polynomial order: at ``order > 1``
        each hex carries ``(order+1)**3`` straight-sided (trilinear) GLL nodes."""
        P = np.asarray(P, dtype=float)
        ni1, nj1, nk1, _ = P.shape
        ni, nj, nk = ni1 - 1, nj1 - 1, nk1 - 1
        points = P.reshape(-1, 3)
        ids = np.arange(ni1 * nj1 * nk1, dtype=np.int64).reshape(ni1, nj1, nk1)

        hexes = np.empty((ni * nj * nk, 8), dtype=np.int64)
        e = 0
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    hexes[e] = [ids[i, j, k], ids[i + 1, j, k],
                                ids[i + 1, j + 1, k], ids[i, j + 1, k],
                                ids[i, j, k + 1], ids[i + 1, j, k + 1],
                                ids[i + 1, j + 1, k + 1], ids[i, j + 1, k + 1]]
                    e += 1
        bnd: list[list[int]] = []
        names: list[str] = []
        cell = np.arange(ni * nj * nk).reshape(ni, nj, nk)
        for side, name in (face_tags or {}).items():
            if not name:
                continue
            face, axis, end = _GRID_SIDES[side]
            plane: IntArray = cell.take(0 if end == 0 else -1, axis=axis).ravel()
            for eid in plane:
                bnd.append([int(eid), face])
                names.append(name)
        # np.full width-infers from the fill value (dtype=np.str_ would clip to <U1)
        etags: StrArray = np.full(hexes.shape[0], element_tag)
        curved: CurvedBlock | None = (subdivide_hexes(points, hexes, order)
                                      if order > 1 else None)
        return cls.from_corners(points, hexes, *cls._order_bnd(bnd, names),
                                element_tags=etags, order=order, curved=curved)

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

    # -- shared-point view ------------------------------------------------
    def weld(self) -> tuple[PointArray, IntArray, int]:
        """Shared-point view ``(points, hexes, n_points)``; the live positions array
        can be mutated in place to reposition the mesh."""
        return self.points, self.hexes, self.n_points

    def classify_points(self, wall: str) -> tuple[BoolArray, BoolArray]:
        """Flag welded points: ``(is_wall, is_fixed)``.  Faces named ``wall`` are
        wall; all other tagged faces are fixed.  A point on both is treated as
        fixed."""
        _, HC, nu = self.weld()
        is_wall: BoolArray = np.zeros(nu, dtype=bool)
        is_fixed: BoolArray = np.zeros(nu, dtype=bool)
        for b in range(self.boundaries.shape[0]):
            elem = int(self.boundaries[b, 0])
            face = int(self.boundaries[b, 1])
            ids = HC[elem, self.FACE_POINTS[face - 1, :]]
            if self.boundary_tags[b] == wall:
                is_wall[ids] = True
            else:
                is_fixed[ids] = True
        is_wall[is_fixed] = False
        return is_wall, is_fixed

    # -- topology / validity --------------------------------------------
    def topology_report(self) -> dict[str, Any]:
        """Watertightness / connectivity report of the welded mesh."""
        from ..model import topology
        X, HC, _ = self.weld()
        return topology.hex_report(X, HC)

    def is_watertight(self) -> bool:
        """``True`` if the mesh boundary is a closed, leak-tight 2-manifold and the
        mesh is a single connected component. Does not imply conformity."""
        rep = self.topology_report()
        return bool(rep["watertight"] and rep["n_components"] == 1)

    def is_conforming(self) -> bool:
        """``True`` if the mesh has no hanging points (no T-junctions)."""
        return bool(self.topology_report()["conformal"])

    def report(self) -> str:
        """Human-readable summary: element/point counts, scaled-Jacobian quality,
        per-name tagged-face counts, and the topology report."""
        from ..model import topology
        from . import quality
        lines = ["%d hex elements, %d points" % (self.n_hexes, self.n_points)]
        lines.append(quality.format_report(quality.summary(self.points, self.hexes)))
        for name in self.boundary_group_tags:
            n = int(np.sum(self.boundary_tags == name))
            lines.append("  %-14s : %d faces" % (name, n))
        lines.append(topology.format_report(topology.hex_report(self.points, self.hexes)))
        return "\n".join(lines)

    # -- connectivity helpers (used by interior / smoothing) ------------
    @staticmethod
    def _unique_edges(HC: IntArray, he: IntArray) -> IntArray:
        Ei = HC[:, he[:, 0]].ravel()
        Ej = HC[:, he[:, 1]].ravel()
        return np.unique(np.sort(np.column_stack([Ei, Ej]), axis=1), axis=0)
