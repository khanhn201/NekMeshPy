"""Quad mesh of a single cross-section slice.

``QuadMesh`` is a pure container and the quad sibling of
:class:`~nekmeshpy.geometry.trimesh.TriMesh`: point coordinates ``points`` (nn,3)
and quad connectivity ``quads`` (nq,4), with matching ``n_points`` / ``n_quads``
size properties.  It additionally records the set of quad edges lying on the
outer wall boundary (``boundaries``, each a ``frozenset({i, j})`` of point
indices).

Besides the array constructor, three factory classmethods fill a bounded region
with quads (mirroring the :class:`~nekmeshpy.geometry.hexmesh.HexMesh` factories):
:meth:`structured` (transfinite grid over four edge curves), :meth:`ogrid` (butterfly
O-grid inside a closed loop), and :meth:`half_ogrid` (half-disc O-grid split along
a spine).  A stack of these slices (sharing connectivity) is recombined into
hexes by :meth:`~nekmeshpy.geometry.hexmesh.HexMesh.loft`, or a single section is
swept along a straight axis by :meth:`~nekmeshpy.geometry.hexmesh.HexMesh.extrude`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, PointArray
from ..model.fields import validate_layers
from .curve import Curve, CurveLoop

#: Boundary-name sentinel meaning "this face is *not* a boundary": a section edge
#: (or swept face / grid side) carrying this name emits **no** boundary row, so it
#: is left as a raw topological surface -- or, when two blocks are stitched, as the
#: welded-away seam.  Marking the touching faces ``NO_BOUNDARY`` before
#: :meth:`HexMesh.merge` lets merge stay a plain concatenate: there is simply no
#: stale tag on the face that becomes interior.  Equal to ``""`` so it also reads
#: as "unnamed" everywhere an empty name is already skipped.
NO_BOUNDARY: str = ""


def _apply_interior(qm: QuadMesh, interior_method: str | None) -> QuadMesh:
    """Reposition ``qm``'s interior points in place via ``ops.interior`` (``None``
    leaves the raw algebraic fill).  Imported lazily to avoid an import cycle."""
    if interior_method is not None:
        from ..ops import interior
        interior.set_section_interior(qm, interior_method)
    return qm


def _check_boundary(obj: Curve | CurveLoop, name: str,
                    expected: type[Curve] | type[CurveLoop], min_pts: int) -> PointArray:
    """Validate a curve/loop factory argument, returning its ``(N,3)`` points.
    Enforces the exact type (so the open/closed :class:`Curve` vs
    :class:`CurveLoop` distinction holds at runtime, not just for the type
    checker), a minimum point count, and finite coordinates."""
    if not isinstance(obj, expected):
        raise TypeError("%s must be a %s, got %s"
                        % (name, expected.__name__, type(obj).__name__))
    pts = obj.points
    if pts.shape[0] < min_pts:
        raise ValueError("%s needs at least %d points, got %d"
                         % (name, min_pts, pts.shape[0]))
    if not np.all(np.isfinite(pts)):
        raise ValueError("%s has non-finite coordinates" % name)
    return pts


class QuadMesh:
    def __init__(
        self,
        points: PointArray,
        quads: IntArray,
        boundaries: Iterable[Iterable[int]] | None = None,
        boundary_names: Mapping[frozenset[int], str] | None = None,
    ) -> None:
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.quads = np.asarray(quads, dtype=np.int64)
        # per-edge boundary names carried to the swept side faces by HexMesh.loft
        # (``frozenset({i, j})`` -> name).  A NO_BOUNDARY value marks an edge that
        # should stay untagged (e.g. one that will be welded away by merge).
        self.boundary_names: dict[frozenset[int], str] = (
            {} if boundary_names is None
            else {frozenset(e): n for e, n in boundary_names.items()})
        # walls (edges held fixed by interior methods): explicit set if given,
        # else every named edge, else none -- so passing only ``boundary_names`` still
        # pins the section boundary during interior repositioning.
        self.boundaries: set[frozenset[int]] = (
            {frozenset(e) for e in boundaries} if boundaries is not None
            else set(self.boundary_names) if boundary_names is not None
            else set())

    # local quad edges (CCW); row e is edge e+1
    EDGE_POINTS = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def n_quads(self) -> int:
        return self.quads.shape[0]

    # -- boundary queries (topological section outline) -----------------
    @staticmethod
    def _boundary_mask(quads: IntArray) -> tuple[IntArray, BoolArray]:
        """``(edges, is_boundary)``: every quad edge ``(4M,2)`` in CCW order,
        element-major (row ``4q+e`` is quad ``q``, local edge ``e``), and a mask
        of those borne by a single quad (the section boundary)."""
        Q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
        edges: IntArray = Q[:, QuadMesh.EDGE_POINTS].reshape(-1, 2)
        keys = np.sort(edges, axis=1)
        _, inverse, counts = np.unique(
            keys, axis=0, return_inverse=True, return_counts=True)
        return edges, counts[inverse.ravel()] == 1

    def boundary_edges(self) -> IntArray:
        """``(K,2)`` array of ``[quad id, local edge (1-4)]`` for every edge on
        the section boundary (an edge borne by a single quad).  Distinct from
        :attr:`boundaries` (the wall subset -- a half-disk section's boundary is
        the wall arc *plus* the flat spine).  An edge's point ids are
        ``self.quads[q, self.EDGE_POINTS[e - 1]]``."""
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
        """Merge quad sections into one, welding coincident **boundary** points in
        a single pass (interior points are exclusive to their section).  ``tol`` is
        the absolute coincidence distance (default ``1e-7`` x the extent).  Edges
        that become shared drop out of the outline; ``boundaries`` and
        ``boundary_names`` are unioned and remapped to the merged point ids (a shared
        edge named in one section keeps that name)."""
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

        quad_list, wall = [], set()
        enames: dict[frozenset[int], str] = {}
        noff = 0
        for m, c in zip(meshes, counts):
            quad_list.append(point_id[m.quads + noff])
            for edge in m.boundaries:
                i, j = (int(x) for x in edge)
                wall.add(frozenset((int(point_id[i + noff]), int(point_id[j + noff]))))
            for edge, name in m.boundary_names.items():
                i, j = (int(x) for x in edge)
                enames.setdefault(
                    frozenset((int(point_id[i + noff]), int(point_id[j + noff]))), name)
            noff += c
        quads = np.concatenate(quad_list, axis=0) if quad_list else np.zeros((0, 4), np.int64)
        return cls(points, quads, boundaries=wall, boundary_names=enames)

    # -- factories (2-D section meshers) --------------------------------
    # Each fills a bounded region with quads and returns a QuadMesh whose
    # ``boundaries`` is the outer boundary.  The topology is chosen by *which*
    # factory you call -- structured and o-grid are different topologies -- and an
    # optional ``interior_method`` (``"conduction"`` / ``"winslow"`` /
    # ``"bilinear"``; ``None`` = raw algebraic fill) repositions the interior
    # points via :func:`nekmeshpy.ops.interior.set_section_interior`.
    @classmethod
    def structured(cls, edges: list[Curve], *,
                   boundary_names: Mapping[str, str] | None = None,
                   interior_method: str | None = None) -> QuadMesh:
        """Transfinite (Coons-patch) quad grid over the surface bounded by four
        edge curves ``edges = [bottom, right, top, left]`` given in CCW loop order
        (each a :class:`~nekmeshpy.geometry.curve.Curve` of 3-D points; the Coons
        blend runs directly on the 3-D edges, so the section may lie in any plane).
        The curves must share corners, i.e. form a closed loop: ``bottom`` ends where
        ``right`` begins, ``right`` where ``top`` begins, and so on.

        The grid resolution and node distribution come **directly from the edge
        curves' own points** -- there is no resampling.  ``bottom`` and ``top``
        must carry the same number of points (``nx+1``, the u-direction) and
        ``left`` and ``right`` the same number (``ny+1``, the v-direction); the
        interior is filled by bilinearly-blended transfinite (Coons) interpolation,
        giving ``nx`` x ``ny`` cells.  Because the caller supplies the sampling, a
        graded edge (e.g. built with :meth:`~nekmeshpy.geometry.curve.Curve.resample`
        at clustered fractions) yields a graded grid -- thinner cells near a wall --
        directly.  Straight, uniformly-sampled edges reduce this exactly to a
        uniform bilinear grid.

        ``boundary_names`` names the section's outer edges at build time, keyed by
        side -- ``"bottom"`` / ``"right"`` / ``"top"`` / ``"left"`` (matching the
        ``edges`` order) -> boundary name -- so the names ride through
        :meth:`~nekmeshpy.geometry.hexmesh.HexMesh.loft` / ``extrude`` onto the
        swept side faces (a side left out stays untagged; the
        :data:`NO_BOUNDARY` sentinel names a side but emits no boundary row, e.g.
        an edge welded away by :meth:`merge`).
        """
        if len(edges) != 4:
            raise ValueError("structured needs exactly 4 edge curves "
                             "[bottom, right, top, left]")
        bottom, right, top, left = edges
        for nm, e in (("bottom", bottom), ("right", right),
                      ("top", top), ("left", left)):
            _check_boundary(e, "structured " + nm + " edge", Curve, 2)
        # resolution comes from the edges' own point counts (no resampling): the
        # opposite edges of each family must be sampled to matching counts.
        if bottom.points.shape[0] != top.points.shape[0]:
            raise ValueError(
                "structured: bottom and top edges must have equal point counts "
                "(got %d, %d); resample them to the same nx+1 first"
                % (bottom.points.shape[0], top.points.shape[0]))
        if left.points.shape[0] != right.points.shape[0]:
            raise ValueError(
                "structured: left and right edges must have equal point counts "
                "(got %d, %d); resample them to the same ny+1 first"
                % (left.points.shape[0], right.points.shape[0]))
        nx = bottom.points.shape[0] - 1
        ny = left.points.shape[0] - 1
        # the four edges must close into a loop (share corners) in CCW order
        allpts = np.vstack([e.points for e in edges])
        scale = float(np.max(allpts.max(axis=0) - allpts.min(axis=0)))
        tol = 1e-6 * scale if scale > 0 else 1e-9
        for lbl, p, q in (("bottom->right", bottom.points[-1], right.points[0]),
                          ("right->top", right.points[-1], top.points[0]),
                          ("top->left", top.points[-1], left.points[0]),
                          ("left->bottom", left.points[-1], bottom.points[0])):
            gap = float(np.linalg.norm(p - q))
            if gap > tol:
                raise ValueError(
                    "structured: edges must form a closed loop in CCW order "
                    "[bottom, right, top, left] with shared corners; "
                    "gap %.3g at %s" % (gap, lbl))
        # orient the two edge families so both run corner c0->c1 (u) / c0->c3 (v);
        # the caller's node distribution is used verbatim (no resampling):
        cb = bottom.points                                     # c0 -> c1
        ct = top.points[::-1]                                  # c3 -> c2
        cl = left.points[::-1]                                 # c0 -> c3
        cr = right.points                                      # c1 -> c2
        P00, P10, P01, P11 = cb[0], cb[-1], ct[0], ct[-1]      # shared corners

        u = np.linspace(0.0, 1.0, nx + 1)[:, None, None]       # (nx+1,1,1)
        v = np.linspace(0.0, 1.0, ny + 1)[None, :, None]       # (1,ny+1,1)
        # Coons blend: (edge terms) - (bilinear corner correction)
        S = ((1 - v) * cb[:, None, :] + v * ct[:, None, :]
             + (1 - u) * cl[None, :, :] + u * cr[None, :, :]
             - ((1 - u) * (1 - v) * P00 + u * (1 - v) * P10
                + (1 - u) * v * P01 + u * v * P11))
        points = S.reshape(-1, 3)                              # nid(i,j)=i*row+j
        row = ny + 1

        def nid(i: int, j: int) -> int:
            return i * row + j

        # quads in i-major / j-minor order (i in [0,nx), j in [0,ny))
        qi: IntArray = np.repeat(np.arange(nx, dtype=np.int64), ny)
        qj = np.tile(np.arange(ny, dtype=np.int64), nx)
        quads = np.stack([qi * row + qj, (qi + 1) * row + qj,
                          (qi + 1) * row + qj + 1, qi * row + qj + 1], axis=1)
        wall = set()
        side_edges: dict[str, list[frozenset[int]]] = {
            "left": [frozenset((nid(0, j), nid(0, j + 1))) for j in range(ny)],
            "right": [frozenset((nid(nx, j), nid(nx, j + 1))) for j in range(ny)],
            "bottom": [frozenset((nid(i, 0), nid(i + 1, 0))) for i in range(nx)],
            "top": [frozenset((nid(i, ny), nid(i + 1, ny))) for i in range(nx)],
        }
        for edges_on_side in side_edges.values():
            wall.update(edges_on_side)
        enames: dict[frozenset[int], str] = {}
        for side, nm in (boundary_names or {}).items():
            if side not in side_edges:
                raise ValueError("structured boundary_names side must be one of "
                                 "bottom/right/top/left, got %r" % side)
            for edge in side_edges[side]:
                enames[edge] = nm
        return _apply_interior(
            cls(points, quads, boundaries=wall, boundary_names=enames),
            interior_method)

    @classmethod
    def ogrid(cls, boundary: CurveLoop, n_side: int, radial: FloatArray, *,
              center_scale: float = 0.5,
              wall_name: str = "", interior_method: str | None = None) -> QuadMesh:
        """Butterfly O-grid filling the closed ``boundary``
        (a :class:`~nekmeshpy.geometry.curve.CurveLoop`, e.g. from
        ``CurveLoop.circle``): a central
        ``n_side x n_side`` block at the loop centroid, surrounded by O-ring
        layers blending its perimeter out to the boundary (no collapsed cell at the
        centre).  ``center_scale`` sizes the block (fraction of the mean radius).

        The whole grid is built in 3-D with **no projection to a plane**, so a
        curvy / non-planar boundary keeps its true shape (the wall ring sits exactly
        on the boundary, resampled by arc length to ``4*n_side`` points).  The
        block-and-ring construction is only an initial guess; pass
        ``interior_method="conduction"`` to relax the interior harmonically onto the
        curved surface spanned by the fixed boundary ring.  For a planar boundary the
        result stays coplanar with it.

        ``radial`` are the O-ring layer positions with the **initial position
        explicit** (the same convention as :meth:`half_ogrid`,
        :meth:`annulus`, and :meth:`~nekmeshpy.geometry.hexmesh.HexMesh.extrude`'s
        ``layers``): strictly increasing values in ``[0, 1]`` -- ``radial[0]`` is the
        central block perimeter (``0``) and the last is ``1`` (the wall) -- so
        ``radial.size - 1`` rings blend the block perimeter out to the boundary.
        Pass ``geometric_spacing(k, ratio)`` to cluster rings toward the wall.

        ``wall_name`` names the outer ring (the section's single wall) at build
        time, so the name rides through
        :meth:`~nekmeshpy.geometry.hexmesh.HexMesh.loft` / ``extrude`` onto the
        swept side faces (see :meth:`structured`); left empty the wall stays
        untagged.

        ``n_side`` counts the central block cells per side and ``radial`` the ring
        layers of the butterfly topology; they are not interchangeable with the
        ``nx``/``ny`` of :meth:`structured`."""
        if n_side < 1:
            raise ValueError("ogrid needs n_side >= 1")
        if not 0.0 < center_scale < 1.0:
            raise ValueError("ogrid needs center_scale in (0, 1)")
        radial = validate_layers(radial, "ogrid radial")
        n_radial = radial.size - 1
        _check_boundary(boundary, "ogrid boundary", CurveLoop, 3)
        # The butterfly is built entirely in 3-D -- nothing is projected to a plane,
        # so a curvy / non-planar boundary keeps its true shape.  The wall ring is
        # the boundary resampled (by 3-D arc length) to the P = 4*n_side ring points;
        # the central block corners are 4 of those points pulled toward the centroid,
        # bilinearly filled; the rings are straight-chord blends between the block
        # perimeter and the wall.  This is only an initial guess -- pass
        # ``interior_method="conduction"`` to relax the interior harmonically onto
        # the (possibly curved) surface spanned by the fixed boundary ring.
        P = 4 * n_side
        outer_pos: PointArray = boundary.resample(
            np.linspace(0.0, 1.0, P, endpoint=False)).points        # (P,3) true wall
        centroid = outer_pos.mean(axis=0)
        rad = float(np.mean(np.linalg.norm(outer_pos - centroid, axis=1)))
        if rad <= 0.0:
            raise ValueError("ogrid: boundary is degenerate (all points coincide)")
        row = n_side + 1

        def cid(i: int, j: int) -> int:
            return i * row + j

        # central block: 4 corners are the boundary points at arc-length quarters,
        # scaled toward the centroid, then bilinearly interpolated into a 3-D patch.
        C00 = centroid + center_scale * (outer_pos[0] - centroid)
        C10 = centroid + center_scale * (outer_pos[n_side] - centroid)
        C11 = centroid + center_scale * (outer_pos[2 * n_side] - centroid)
        C01 = centroid + center_scale * (outer_pos[3 * n_side] - centroid)
        t_lat = np.arange(row) / n_side
        U = t_lat[:, None]                                           # (row,1)  i / n_side
        V = t_lat[None, :]                                           # (1,row)  j / n_side
        block = (((1 - U) * (1 - V))[..., None] * C00
                 + (U * (1 - V))[..., None] * C10
                 + (U * V)[..., None] * C11
                 + ((1 - U) * V)[..., None] * C01).reshape(-1, 3)    # (row*row, 3)
        bi: IntArray = np.repeat(np.arange(n_side, dtype=np.int64), n_side)
        bj = np.tile(np.arange(n_side, dtype=np.int64), n_side)
        cquads = np.stack([bi * row + bj, (bi + 1) * row + bj,
                           (bi + 1) * row + bj + 1, bi * row + bj + 1], axis=1)

        peri_ids = np.array([cid(i, 0) for i in range(row)]
                            + [cid(n_side, j) for j in range(1, row)]
                            + [cid(i, n_side) for i in range(n_side - 1, -1, -1)]
                            + [cid(0, j) for j in range(n_side - 1, 0, -1)],
                            dtype=np.int64)
        peri_pos = block[peri_ids, :]                                # (P,3), index-aligned
                                                                     # with outer_pos
        # n_radial O-ring layers blending the block perimeter out to the boundary;
        # radial[0] (== 0) is the block perimeter itself, so skip it
        fracs = radial[1:]
        layers = [block]
        ring = [peri_ids]
        nprev = block.shape[0]
        for t in fracs:
            layers.append((1.0 - t) * peri_pos + t * outer_pos)
            ring.append(nprev + np.arange(P, dtype=np.int64))
            nprev += P
        points = np.vstack(layers)

        k: IntArray = np.arange(P, dtype=np.int64)
        kn = (k + 1) % P
        ring_quads = [np.stack([b[k], b[kn], a[kn], a[k]], axis=1)    # CCW
                      for a, b in zip(ring[:-1], ring[1:])]
        quads = np.vstack([cquads, *ring_quads])

        outer = ring[n_radial]
        wall = {frozenset((int(outer[m]), int(outer[(m + 1) % P]))) for m in range(P)}
        enames = {e: wall_name for e in wall} if wall_name else None
        qm = cls(points, quads, boundaries=wall, boundary_names=enames)
        return _apply_interior(qm, interior_method)

    @classmethod
    def half_ogrid(cls, arc: Curve, spine: Curve,
                   radial: FloatArray, *, center_scale: float = 0.5,
                   wall_name: str = "",
                   interior_method: str | None = None) -> QuadMesh:
        """Structured HALF-circle O-grid over a half-disk split along the ``spine``
        curve (A1..A2); the wall ``arc`` (``(4*Ntheta+1, 3)``, arc[0]=A1,
        arc[-1]=A2) is the open boundary, recorded as ``boundaries``.  ``radial``
        are the O-ring layer positions with the **initial position explicit** (the
        same convention as :meth:`ogrid` / :meth:`annulus`): strictly increasing
        values in ``[0, 1]`` -- ``radial[0]`` is the inner block perimeter (``0``)
        and the last is ``1`` (the wall) -- so ``radial.size - 1`` rings are laid
        out; ``center_scale`` is the inner block extent as a fraction of the spine.
        ``wall_name`` names
        the ``arc`` wall at build time so it rides through
        :meth:`~nekmeshpy.geometry.hexmesh.HexMesh.loft` / ``extrude`` onto the
        swept side faces (see :meth:`structured`); left empty the wall stays
        untagged."""
        apts = _check_boundary(arc, "half_ogrid arc", Curve, 5)   # (na,3) backing array
        _check_boundary(spine, "half_ogrid spine", Curve, 2)
        na = apts.shape[0]
        if (na - 1) % 4 != 0:
            raise ValueError("half_ogrid: arc must have 4*Ntheta+1 points (Ntheta >= 1)")
        Nt = (na - 1) // 4
        if not 0.0 < center_scale < 1.0:
            raise ValueError("half_ogrid needs center_scale in (0, 1)")
        radial = validate_layers(radial, "half_ogrid radial")
        Nr = radial.size - 1
        cs = center_scale

        O = spine.resample(0.5).points[0]
        sN = (1 - cs) / 2
        sS = (1 + cs) / 2

        fe = spine.resample(np.linspace(sN, sS, 2 * Nt + 1)).points
        Q_N = O + cs * (apts[Nt, :] - O)
        Q_S = O + cs * (apts[3 * Nt, :] - O)
        ae = Q_N + (np.arange(2 * Nt + 1)[:, None] / (2 * Nt)) * (Q_S - Q_N)
        P_N = fe[0, :]
        P_S = fe[-1, :]

        ni = 2 * Nt
        nj = Nt
        rid: IntArray = np.zeros((ni + 1, nj + 1), dtype=np.int64)
        point_list = []
        for i in range(ni + 1):
            u = i / ni
            for j in range(nj + 1):
                v = j / nj
                left = (1 - v) * P_N + v * Q_N
                right = (1 - v) * P_S + v * Q_S
                bott = fe[i, :]
                top = ae[i, :]
                C = ((1 - v) * bott + v * top + (1 - u) * left + u * right
                     - ((1 - u) * (1 - v) * P_N + u * (1 - v) * P_S
                        + (1 - u) * v * Q_N + u * v * Q_S))
                point_list.append(C)
                rid[i, j] = len(point_list) - 1

        quads = []
        for i in range(ni):
            for j in range(nj):
                quads.append([rid[i, j], rid[i + 1, j], rid[i + 1, j + 1], rid[i, j + 1]])

        peri = np.concatenate([rid[0, 0:nj + 1], rid[1:ni + 1, nj], rid[ni, nj - 1::-1]])
        points = np.array(point_list, dtype=float)
        peripts = points[peri, :]

        lid = [peri]
        for r in range(Nr):
            tau = radial[r + 1]                 # radial[0] == 0 is the block perimeter
            pts = (1 - tau) * peripts + tau * apts
            pts[0, :] = spine.resample((1 - tau) * sN).points[0]
            pts[-1, :] = spine.resample(sS + tau * (1 - sS)).points[0]
            base = points.shape[0]
            points = np.vstack([points, pts])
            lid.append(base + np.arange(pts.shape[0]))

        for r in range(Nr):
            a = lid[r]
            b = lid[r + 1]
            for k in range(4 * Nt):
                quads.append([a[k], a[k + 1], b[k + 1], b[k]])

        arc_ids = lid[Nr]
        boundaries = {frozenset((int(arc_ids[k]), int(arc_ids[k + 1])))
                      for k in range(arc_ids.shape[0] - 1)}
        enames = {e: wall_name for e in boundaries} if wall_name else None
        qm = cls(points, np.array(quads, dtype=np.int64),
                 boundaries=boundaries, boundary_names=enames)
        return _apply_interior(qm, interior_method)

    @classmethod
    def annulus(cls, inner: CurveLoop, outer: CurveLoop, radial: FloatArray, *,
                interior_method: str | None = None,
                inner_name: str = "", outer_name: str = "",
                ) -> QuadMesh:
        """Ring O-grid filling the region *between* an inner and an outer closed
        loop (both :class:`~nekmeshpy.geometry.curve.CurveLoop`) -- e.g. a circular
        body inside a square far-field box for a flow-past-cylinder section.

        The two loops are paired **by index**: they must carry the same number of
        points ``N`` (the azimuthal resolution), and point ``i`` of ``inner`` is
        joined radially to point ``i`` of ``outer``.  The caller is responsible for
        supplying a correctly sized and oriented outer loop -- there is no
        resampling here.  To align a coarse far-field box loop to a finer body loop
        so the radial lines do not skew, project it first with
        :meth:`~nekmeshpy.geometry.curve.CurveLoop.radial_match`::

            outer = CurveLoop([...box corners...]).radial_match(inner)

        ``radial`` are the ring positions with the **initial position explicit**
        (the same convention as :meth:`ogrid` / :meth:`half_ogrid` and
        :meth:`~nekmeshpy.geometry.hexmesh.HexMesh.extrude`'s ``layers``): strictly
        increasing values in ``[0, 1]`` -- ``radial[0]`` is the inner ring (``0`` for
        a ring flush with the body, or e.g. ``0.5`` to start the mesh halfway out)
        and the last is ``1`` (the outer loop) -- so ``radial.size - 1`` ring layers
        blend ``radial[0]`` -> outer.  Pass
        ``geometric_spacing(k, ratio)`` to cluster rings toward the inner body for a
        boundary layer, or ``uniform_spacing(k)`` / ``numpy.linspace(a, 1, k + 1)``.
        Both the inner and outer rings are recorded as ``boundaries``, so an
        ``interior_method`` holds them fixed while relaxing the ring interior.

        The ring blend runs directly in 3-D (no projection to a plane), so the two
        loops need not be planar or coplanar: a curvy / non-planar inner-outer pair
        keeps its true shape and an ``interior_method`` (e.g. ``conduction``)
        relaxes the ring interior onto the resulting curved surface.

        ``inner_name`` and ``outer_name`` name the whole inner and outer rings at
        build time -- each a single string (e.g. an embedded body and a uniform,
        round far field).  The names ride through
        :meth:`~nekmeshpy.geometry.hexmesh.HexMesh.loft` / ``extrude`` onto the
        swept side faces (see :meth:`structured`).  ``annulus`` deliberately has
        **no per-side outer tagging**: to split a far field into distinct sides
        (inlet / outlet / top / bottom), build it from several
        :meth:`structured` patches and stitch them with
        :meth:`~nekmeshpy.geometry.hexmesh.HexMesh.merge`, as
        ``examples/flow_past_cylinder.py`` does (the 2-D analogue of the
        cubed-sphere shell in ``examples/flow_past_sphere.py``).

        Gives ``N x (radial.size - 1)`` quads wound CCW.  ``radial`` here sets the
        ring layers and is not interchangeable with the ``nx``/``ny`` of
        :meth:`structured` or the ``n_side`` of :meth:`ogrid`."""
        radial = validate_layers(radial, "annulus radial")
        n_radial = radial.size - 1
        ipts = _check_boundary(inner, "annulus inner", CurveLoop, 3)
        opts = _check_boundary(outer, "annulus outer", CurveLoop, 3)
        if ipts.shape[0] != opts.shape[0]:
            raise ValueError(
                "annulus: inner and outer loops must have equal point counts "
                "(got %d, %d); align the outer loop to the inner first, e.g. "
                "outer.radial_match(inner)" % (ipts.shape[0], opts.shape[0]))
        # blend the two rings directly in 3-D (no projection): ring k is the
        # straight-chord interpolation between the true 3-D inner and outer points,
        # so a curvy / non-planar pair of loops keeps its shape and an
        # ``interior_method`` (e.g. conduction) relaxes the ring interior onto it.
        A: FloatArray = ipts                                         # (N,3)
        N = A.shape[0]
        B: FloatArray = opts                                         # (N,3) by index
        if float(np.min(np.linalg.norm(B - A, axis=1))) <= 0.0:
            raise ValueError("annulus: inner and outer loops touch or cross")

        # n_radial+1 rings inner(radial[0]) -> outer(radial[-1]=1); id(k,i) = k*N + i
        fracs = radial                                               # (n_radial+1,)
        pts3d = np.vstack([(1.0 - t) * A + t * B for t in fracs])    # ((n_radial+1)*N,3)

        k: IntArray = np.repeat(np.arange(n_radial, dtype=np.int64), N)
        i = np.tile(np.arange(N, dtype=np.int64), n_radial)
        inn = (i + 1) % N
        # radial-out, +theta, radial-in, -theta -> CCW in the section's plane
        quads = np.stack([k * N + i, (k + 1) * N + i,
                          (k + 1) * N + inn, k * N + inn], axis=1)

        base = n_radial * N
        inner_ring = [frozenset((int(m), int((m + 1) % N))) for m in range(N)]
        outer_ring = [frozenset((int(base + m), int(base + (m + 1) % N)))
                      for m in range(N)]
        wall = set(inner_ring) | set(outer_ring)

        enames: dict[frozenset[int], str] = {}
        if inner_name:
            for edge in inner_ring:
                enames[edge] = inner_name
        if outer_name:
            for edge in outer_ring:
                enames[edge] = outer_name
        qm = cls(pts3d, quads, boundaries=wall, boundary_names=enames)
        return _apply_interior(qm, interior_method)
