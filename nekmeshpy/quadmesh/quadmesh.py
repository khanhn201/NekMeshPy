"""Quad mesh of a single cross-section slice.

``QuadMesh`` is a pure container: ``points`` ``(nn,3)`` and quad connectivity
``quads`` ``(nq,4)``, plus a dense per-quad ``element_tags`` and a sparse tagged
boundary-edge list ``boundaries`` ``(Nbc,2)`` = ``[quad id, side 1-4]`` with a
parallel ``boundary_tags``.  Factory classmethods fill a bounded region with quads;
``extrude``/``loft`` sweep a ``LineMesh`` into a quad section.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Point, PointArray, StrArray, Vec3
from ..linemesh import LineMesh
from ..model.fields import validate_layers

#: Boundary-name sentinel meaning "not a boundary": a side carrying this name emits
#: no boundary row.  Equal to ``""`` so it reads as "unnamed" everywhere.
NO_BOUNDARY: str = ""

# default sweep axis / origin for extrude (module-level singletons; read-only)
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_ORIGIN = np.array([0.0, 0.0, 0.0])

# grid side name -> (quad edge side 1-4, axis, which end) for from_grid.
_GRID_EDGES = {
    "x_min": (4, 0, 0), "x_max": (2, 0, -1),
    "y_min": (1, 1, 0), "y_max": (3, 1, -1),
}


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


class QuadMesh:
    """A quadrilateral surface / cross-section mesh in shared-point form.

    Stores ``points`` ``(P,3)`` and ``quads`` ``(Q,4)`` CCW connectivity, a dense
    per-quad ``element_tags``, and a sparse tagged-boundary list ``boundaries``
    ``(Nbc,2)`` = ``[quad id, side 1-4]`` with a parallel ``boundary_tags``."""

    def __init__(
        self,
        points: PointArray,
        quads: IntArray,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
    ) -> None:
        """Construct from arrays: ``points`` ``(P,3)``, ``quads`` ``(Q,4)`` CCW
        indices, an optional dense per-quad ``element_tags`` ``(Q,)``, and an
        optional tagged-boundary list ``boundaries`` ``(Nbc,2)`` = ``[quad id,
        side 1-4]`` with a parallel ``boundary_tags``."""
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
        # dense per-quad region/material tag ("" = untagged)
        if element_tags is None:
            self.element_tags: StrArray = np.full(
                self.quads.shape[0], "", dtype=np.str_)
        else:
            et = np.asarray(element_tags, dtype=np.str_).reshape(-1)
            if et.shape[0] != self.quads.shape[0]:
                raise ValueError("element_tags length (%d) must match quads (%d)"
                                 % (et.shape[0], self.quads.shape[0]))
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

    # local quad edges (CCW); row e is edge e+1
    EDGE_POINTS = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)

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
    def scaled_jacobian(self) -> FloatArray:
        """Per-quad minimum corner scaled Jacobian ``(n_quads,)``."""
        from . import quality
        return quality.scaled_jacobian(self.points, self.quads)

    def quality_summary(self) -> dict[str, Any]:
        """Aggregate scaled-Jacobian statistics."""
        from . import quality
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
        return cls(points, quads, b_ord, n_ord, element_tags=etags)

    @classmethod
    def from_grid(
        cls,
        P: FloatArray,
        *,
        edge_tags: Mapping[str, str] | None = None,
        element_tag: str = "",
    ) -> QuadMesh:
        """Build quads from a structured point grid ``P`` ``(ni+1,nj+1,3)``.
        ``edge_tags`` maps side names (``x_min`` / ``x_max`` / ``y_min`` / ``y_max``)
        to boundary tags on the four outer edges; a side left out (or mapped to
        ``NO_BOUNDARY``) emits no boundary row.  ``element_tag`` is written to every
        quad's dense ``element_tags``."""
        P = np.asarray(P, dtype=float)
        ni1, nj1, _ = P.shape
        ni, nj = ni1 - 1, nj1 - 1
        points = P.reshape(-1, 3)
        ids = np.arange(ni1 * nj1, dtype=np.int64).reshape(ni1, nj1)

        quads = np.empty((ni * nj, 4), dtype=np.int64)
        e = 0
        for i in range(ni):
            for j in range(nj):
                quads[e] = [ids[i, j], ids[i + 1, j],
                            ids[i + 1, j + 1], ids[i, j + 1]]   # CCW
                e += 1
        bnd: list[list[int]] = []
        names: list[str] = []
        cell = np.arange(ni * nj).reshape(ni, nj)
        for side, name in (edge_tags or {}).items():
            if not name:
                continue
            edge, axis, end = _GRID_EDGES[side]
            strip: IntArray = cell.take(0 if end == 0 else -1, axis=axis).ravel()
            for qid in strip:
                bnd.append([int(qid), edge])
                names.append(name)
        # np.full width-infers from the fill value (dtype=np.str_ would clip to <U1)
        etags: StrArray = np.full(quads.shape[0], element_tag)
        return cls(points, quads, *cls._order_bnd(bnd, names), element_tags=etags)

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
        base = np.asarray(line.points, dtype=float).reshape(-1, 3) \
            + np.asarray(origin, dtype=float)
        axis_u: Vec3 = np.asarray(axis, dtype=float)
        axis_u = axis_u / np.linalg.norm(axis_u)
        offsets = validate_layers(layers, "extrude layers") * float(length)
        slices = [LineMesh(base + d * axis_u[None, :], line.lines,
                           element_tags=line.element_tags,
                           boundaries=line.boundaries,
                           boundary_tags=line.boundary_tags,
                           closed=line.is_closed)
                  for d in offsets]
        return cls.loft(slices, first_tag=first_tag, last_tag=last_tag)

    @classmethod
    def loft(
        cls,
        slices: Sequence[LineMesh],
        *,
        first_tag: str | Sequence[str] | StrArray = "",
        last_tag: str | Sequence[str] | StrArray = "",
    ) -> QuadMesh:
        """Loft a stack of conformal ``LineMesh`` profiles into a quad section
        (the general primitive behind :meth:`extrude`).

        ``slices`` is ``nz+1`` line profiles sharing the same ``lines``,
        ``element_tags``, and ``boundaries``; consecutive profiles form ``nz`` quad
        layers.  For line ``(a, b)`` at layer ``i`` the column quad is
        ``[a_i, b_i, b_{i+1}, a_{i+1}]``.  The line's ``element_tags`` ride onto every
        quad in its column and tagged boundary points onto the swept wall edges;
        ``first_tag`` / ``last_tag`` name the near / far cap edges (scalar or per-line
        array)."""
        slices = list(slices)
        lines = np.asarray(slices[0].lines, dtype=np.int64).reshape(-1, 2)
        L = lines.shape[0]
        nz = len(slices) - 1
        S = np.stack([np.asarray(s.points, dtype=float).reshape(-1, 3)
                      for s in slices], axis=0)              # (nz+1, nn, 3)
        nn = S.shape[1]
        points = S.reshape((nz + 1) * nn, 3)                 # global id = i*nn + v

        a = lines[:, 0]
        b = lines[:, 1]
        # quad row for (layer i, line l) = i*L + l; vertices [a_i, b_i, b_{i+1}, a_{i+1}]
        i_idx: IntArray = np.repeat(np.arange(nz, dtype=np.int64), L)
        l_idx = np.tile(np.arange(L, dtype=np.int64), nz)
        av = a[l_idx]
        bv = b[l_idx]
        quads = np.stack([i_idx * nn + av, i_idx * nn + bv,
                          (i_idx + 1) * nn + bv, (i_idx + 1) * nn + av], axis=1)
        etags: StrArray = np.asarray(slices[0].element_tags, dtype=np.str_)[l_idx]

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
        # caps: scalar tags the whole cap, an array tags per section line.
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
        return cls(points, quads, b_ord, n_ord, element_tags=etags)

    # -- factories (2-D section meshers) --------------------------------
    # Each fills a bounded region with quads.  An optional ``smoothing_method``
    # ("conduction" / "winslow" / "bilinear"; None = raw fill) repositions the
    # interior points.
    @classmethod
    def rectangle(cls, corners: PointArray | Sequence[Point], nx: int, ny: int, *,
                  x_frac: FloatArray | None = None,
                  y_frac: FloatArray | None = None,
                  side_tags: Mapping[str, str] | None = None,
                  smoothing_method: str | None = None) -> QuadMesh:
        """Structured quad grid over the rectangle with four CCW corners
        ``corners = [c0, c1, c2, c3]``: ``nx`` cells along ``c0->c1`` (bottom/top),
        ``ny`` along ``c1->c2`` (left/right).  ``x_frac`` / ``y_frac`` are optional
        node fractions in ``[0,1]`` (length ``nx+1`` / ``ny+1``) for grading, else
        uniform.  ``side_tags`` (keyed ``bottom`` / ``right`` / ``top`` / ``left``)
        names the outer sides; an absent side stays untagged."""
        c = np.asarray(corners, dtype=float).reshape(-1, 3)
        if c.shape[0] != 4:
            raise ValueError("rectangle needs exactly 4 corners")
        xf = (np.linspace(0.0, 1.0, nx + 1) if x_frac is None
              else np.asarray(x_frac, dtype=float).ravel())
        yf = (np.linspace(0.0, 1.0, ny + 1) if y_frac is None
              else np.asarray(y_frac, dtype=float).ravel())
        st = side_tags or {}
        specs = (("bottom", c[0], c[1], xf), ("right", c[1], c[2], yf),
                 ("top", c[2], c[3], xf), ("left", c[3], c[0], yf))
        edges = [LineMesh.line(a, b, frac, element_tag=st.get(side, ""))
                 for side, a, b, frac in specs]
        return cls.structured(edges, smoothing_method=smoothing_method)

    @classmethod
    def structured(cls, edges: list[LineMesh], *,
                   boundary_tags: Mapping[str, str] | None = None,
                   smoothing_method: str | None = None) -> QuadMesh:
        """Transfinite (Coons-patch) quad grid over the surface bounded by four
        open edge lines ``edges = [bottom, right, top, left]`` in CCW loop order.
        The lines must share corners (form a closed loop).

        Resolution and node distribution come directly from the edge lines' own
        points (no resampling): ``bottom``/``top`` must share a point count
        (``nx+1``) and ``left``/``right`` another (``ny+1``), giving ``nx`` x ``ny``
        cells.

        Each side is named from its own edge line's uniform ``element_tags``;
        ``boundary_tags`` (keyed by ``"bottom"`` / ``"right"`` / ``"top"`` /
        ``"left"``) overrides that -- a non-empty entry replaces the side's tag, a
        present-but-empty entry suppresses the side.
        """
        if len(edges) != 4:
            raise ValueError("structured needs exactly 4 edge lines "
                             "[bottom, right, top, left]")
        bottom, right, top, left = edges
        for nm, e in (("bottom", bottom), ("right", right),
                      ("top", top), ("left", left)):
            _check_boundary(e, "structured " + nm + " edge", False, 2)
        # resolution comes from the edges' own point counts (no resampling)
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
        # orient the two edge families so both run c0->c1 (u) / c0->c3 (v)
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
        points = S.reshape(-1, 3)                              # id(i,j) = i*row + j
        row = ny + 1

        # quads in i-major / j-minor order (i in [0,nx), j in [0,ny))
        qi: IntArray = np.repeat(np.arange(nx, dtype=np.int64), ny)
        qj = np.tile(np.arange(ny, dtype=np.int64), nx)
        quads = np.stack([qi * row + qj, (qi + 1) * row + qj,
                          (qi + 1) * row + qj + 1, qi * row + qj + 1], axis=1)
        # boundary edges as [quad id, side]; quad id = i*ny + j (i-major).  With
        # v0=(i,j) v1=(i+1,j) v2=(i+1,j+1) v3=(i,j+1): bottom (j=0) is side 1,
        # right (i=nx-1) side 2, top (j=ny-1) side 3, left (i=0) side 4.
        side_rows: dict[str, list[tuple[int, int]]] = {
            "bottom": [(i * ny + 0, 1) for i in range(nx)],
            "right": [((nx - 1) * ny + j, 2) for j in range(ny)],
            "top": [(i * ny + (ny - 1), 3) for i in range(nx)],
            "left": [(0 * ny + j, 4) for j in range(ny)],
        }
        side_edges = {"bottom": bottom, "right": right, "top": top, "left": left}
        bt = boundary_tags or {}
        for side in bt:
            if side not in side_rows:
                raise ValueError("structured boundary_tags side must be one of "
                                 "bottom/right/top/left, got %r" % side)
        bnd: list[list[int]] = []
        names: list[str] = []
        # each side is named by its edge's uniform element tag; a non-empty
        # boundary_tags[side] overrides, a present-but-empty entry suppresses it.
        for side, rows in side_rows.items():
            if side in bt:
                nm = bt[side]
            else:
                et = side_edges[side].element_group_tags
                nm = et[0] if len(et) == 1 else ""
            if not nm:                       # NO_BOUNDARY / "" / untagged -> no row
                continue
            for q, s in rows:
                bnd.append([q, s])
                names.append(nm)
        return _apply_smoothing(
            cls(points, quads, *cls._order_bnd(bnd, names)),
            smoothing_method)

    @classmethod
    def ogrid(cls, boundary: LineMesh, n_side: int, radial: FloatArray, *,
              center_scale: float = 0.5,
              wall_tag: str = "", smoothing_method: str | None = None) -> QuadMesh:
        """O-grid filling the closed ``boundary``: a central ``n_side x n_side``
        block at the loop centroid, surrounded by O-ring layers blending its
        perimeter out to the boundary.  ``center_scale`` sizes the block (fraction
        of the mean radius).

        Built in 3-D with no projection, so a curvy / non-planar boundary keeps its
        true shape; the block-and-ring build is only an initial guess, relaxed by
        ``smoothing_method="conduction"``.

        ``radial`` are the O-ring layer positions with the initial position explicit:
        strictly increasing in ``[0, 1]`` (``radial[0]`` = block perimeter, last =
        ``1`` = wall), giving ``radial.size - 1`` rings.

        The outer ring (wall) is named from ``boundary``'s per-line ``element_tags``;
        a non-empty scalar ``wall_tag`` overrides that for the whole wall."""
        if n_side < 1:
            raise ValueError("ogrid needs n_side >= 1")
        if not 0.0 < center_scale < 1.0:
            raise ValueError("ogrid needs center_scale in (0, 1)")
        radial = validate_layers(radial, "ogrid radial")
        n_radial = radial.size - 1
        bpts = _check_boundary(boundary, "ogrid boundary", True, 3)
        # wall ring = the boundary loop itself, meshed exactly: it must already carry
        # P = 4*n_side points (the caller sizes it, e.g. circle(R, 4*n_side)).  Block
        # corners are 4 of those pulled toward the centroid and bilinearly filled;
        # rings are straight-chord blends (an initial guess for smoothing).
        P = 4 * n_side
        if bpts.shape[0] != P:
            raise ValueError(
                "ogrid boundary must have exactly 4*n_side = %d points to be meshed "
                "exactly (got %d); size the loop to match, e.g. circle(R, %d)"
                % (P, bpts.shape[0], P))
        outer_pos: PointArray = bpts                                # (P,3) true wall
        centroid = outer_pos.mean(axis=0)
        rad = float(np.mean(np.linalg.norm(outer_pos - centroid, axis=1)))
        if rad <= 0.0:
            raise ValueError("ogrid: boundary is degenerate (all points coincide)")
        row = n_side + 1

        def cid(i: int, j: int) -> int:
            return i * row + j

        # central block: 4 corners at arc-length quarters, scaled toward the
        # centroid, bilinearly interpolated into a 3-D patch.
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
        peri_pos = block[peri_ids, :]                                # (P,3)
        # O-ring layers blending block perimeter out to boundary; radial[0]==0 is
        # the perimeter itself, so skip it
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

        # wall edges = side 1 of the outermost ring's quads (rows n_side^2 +
        # (n_radial-1)*P onward).
        wall_q0 = n_side * n_side + (n_radial - 1) * P
        # wall named from the boundary loop's per-segment tags; a non-empty scalar
        # wall_tag overrides that for the whole wall.
        wall_seg = boundary._seg_tags()
        bnd: list[list[int]] = []
        names: list[str] = []
        for m in range(P):
            nm = wall_tag if wall_tag else (wall_seg[m] if wall_seg is not None else "")
            if nm:
                bnd.append([wall_q0 + m, 1])
                names.append(nm)
        qm = cls(points, quads, *cls._order_bnd(bnd, names))
        return _apply_smoothing(qm, smoothing_method)

    @staticmethod
    def half_ogrid_spine_fractions(arc: LineMesh, center_scale: float,
                                   radial: FloatArray) -> FloatArray:
        """Canonical arc-length fractions along the spine that :meth:`half_ogrid`
        indexes, in order ``[fan (2*Ntheta+1), north caps (Nradial), south caps
        (Nradial)]``.  Sample the spine at these fractions (analytically for a
        straight spine, or ``trimesh.ops.resample_polyline`` for a curved one) so the
        spine handed to ``half_ogrid`` is meshed exactly and can't drift.

        With ``sN = (1-cs)/2``, ``sS = (1+cs)/2``: the fan is
        ``linspace(sN, sS, 2*Ntheta+1)``; north cap ``r`` (``r = 1..Nradial``) is
        ``(1-radial[r])*sN`` and south cap ``r`` is ``sS + radial[r]*(1-sS)``."""
        na = np.asarray(arc.points, dtype=float).reshape(-1, 3).shape[0]
        if (na - 1) % 4 != 0:
            raise ValueError("half_ogrid: arc must have 4*Ntheta+1 points (Ntheta >= 1)")
        Nt = (na - 1) // 4
        if not 0.0 < center_scale < 1.0:
            raise ValueError("half_ogrid needs center_scale in (0, 1)")
        rad = validate_layers(radial, "half_ogrid radial")
        cs = center_scale
        sN = (1 - cs) / 2
        sS = (1 + cs) / 2
        fan = np.linspace(sN, sS, 2 * Nt + 1)
        north = (1.0 - rad[1:]) * sN
        south = sS + rad[1:] * (1.0 - sS)
        return np.concatenate([fan, north, south])

    @classmethod
    def half_ogrid(cls, arc: LineMesh, spine: LineMesh,
                   radial: FloatArray, *, center_scale: float = 0.5,
                   wall_tag: str = "",
                   smoothing_method: str | None = None) -> QuadMesh:
        """Structured half-circle O-grid over a half-disk split along the ``spine``
        line (A1..A2); the wall ``arc`` (``(4*Ntheta+1, 3)``, arc[0]=A1, arc[-1]=A2)
        is the open boundary.  ``radial`` are the O-ring layer positions with the
        initial position explicit (strictly increasing in ``[0, 1]``, ``radial[0]`` =
        inner block perimeter, last = ``1`` = wall); ``center_scale`` is the inner
        block extent as a fraction of the spine.

        The ``spine`` is meshed exactly: its points must be the canonical samples this
        method indexes, ``2*Ntheta+1 + 2*Nradial`` of them in the order ``[fan, north
        caps, south caps]`` -- build them with :meth:`half_ogrid_spine_fractions` and
        sample the spine curve at those fractions.  The ``arc`` wall is named from the
        arc's per-segment ``element_tags``; a non-empty scalar ``wall_tag`` overrides
        that for the whole wall."""
        apts = _check_boundary(arc, "half_ogrid arc", False, 5)   # (na,3) backing array
        na = apts.shape[0]
        if (na - 1) % 4 != 0:
            raise ValueError("half_ogrid: arc must have 4*Ntheta+1 points (Ntheta >= 1)")
        Nt = (na - 1) // 4
        if not 0.0 < center_scale < 1.0:
            raise ValueError("half_ogrid needs center_scale in (0, 1)")
        radial = validate_layers(radial, "half_ogrid radial")
        Nr = radial.size - 1
        cs = center_scale

        # spine is meshed exactly: its points must be the canonical samples half_ogrid
        # indexes, in order [fan (2Nt+1), north caps (Nr), south caps (Nr)] -- build
        # them with half_ogrid_spine_fractions so the caller can't drift.
        sp = _check_boundary(spine, "half_ogrid spine", False, 2)
        n_spine = 2 * Nt + 1 + 2 * Nr
        if sp.shape[0] != n_spine:
            raise ValueError(
                "half_ogrid spine must have exactly 2*Ntheta+1 + 2*Nradial = %d points "
                "(got %d); build it with QuadMesh.half_ogrid_spine_fractions"
                % (n_spine, sp.shape[0]))

        fe = sp[0:2 * Nt + 1]                       # the fan, sN..sS
        O = fe[Nt]                                  # spine midpoint (fan is symmetric)
        north = sp[2 * Nt + 1:2 * Nt + 1 + Nr]      # north caps, per radial layer
        south = sp[2 * Nt + 1 + Nr:]                # south caps, per radial layer
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
            pts[0, :] = north[r]                # spine sample at (1-tau)*sN
            pts[-1, :] = south[r]              # spine sample at sS + tau*(1-sS)
            base = points.shape[0]
            points = np.vstack([points, pts])
            lid.append(base + np.arange(pts.shape[0]))

        for r in range(Nr):
            a = lid[r]
            b = lid[r + 1]
            for k in range(4 * Nt):
                quads.append([a[k], a[k + 1], b[k + 1], b[k]])

        # wall arc edges = side 3 of the outermost ring's quads (rows (ni*nj) +
        # (Nr-1)*(4*Nt) onward); wall edge k tracks arc segment k.
        wall_q0 = ni * nj + (Nr - 1) * (4 * Nt)
        # wall named from the arc's per-segment tags; a non-empty scalar wall_tag
        # overrides that for the whole wall.
        wall_seg = arc._seg_tags()
        bnd: list[list[int]] = []
        names: list[str] = []
        for k in range(4 * Nt):
            nm = wall_tag if wall_tag else (wall_seg[k] if wall_seg is not None else "")
            if nm:
                bnd.append([wall_q0 + k, 3])
                names.append(nm)
        qm = cls(points, np.array(quads, dtype=np.int64),
                 *cls._order_bnd(bnd, names))
        return _apply_smoothing(qm, smoothing_method)

    @classmethod
    def annulus(cls, inner: LineMesh, outer: LineMesh, radial: FloatArray, *,
                smoothing_method: str | None = None,
                inner_tag: str = "", outer_tag: str = "",
                ) -> QuadMesh:
        """Ring O-grid filling the region between an inner and an outer closed loop
        -- e.g. a circular body inside a square far-field box.

        The two loops are paired by index: they must carry the same number of points
        ``N``, and point ``i`` of ``inner`` joins radially to point ``i`` of
        ``outer`` (no resampling; build the outer loop index-aligned to the inner,
        e.g. ``LineMesh.far_field_box(inner, ...)``).  ``radial`` are the ring positions with
        the initial position explicit (strictly increasing in ``[0, 1]``,
        ``radial[0]`` = inner ring, last = ``1`` = outer loop), giving
        ``radial.size - 1`` ring layers.  ``smoothing_method`` relaxes the ring
        interior with the inner/outer rings held fixed.

        Boundary tags come from the loops' per-line ``element_tags`` (each ring edge
        tagged from the matching loop segment, so a named box splits the outer ring
        into distinct sides).  A non-empty scalar ``inner_tag`` / ``outer_tag``
        overrides that for the whole inner / outer ring.

        Built by :meth:`loft`-ing the blended rings; the inner / outer rings are the
        loft's near / far caps.  Gives ``N x (radial.size - 1)`` quads."""
        radial = validate_layers(radial, "annulus radial")
        A: FloatArray = _check_boundary(inner, "annulus inner", True, 3)   # (N,3)
        B: FloatArray = _check_boundary(outer, "annulus outer", True, 3)   # (N,3)
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "annulus: inner and outer loops must have equal point counts "
                "(got %d, %d); build the outer loop index-aligned to the inner, "
                "e.g. LineMesh.far_field_box(inner, ...)" % (A.shape[0], B.shape[0]))
        if float(np.min(np.linalg.norm(B - A, axis=1))) <= 0.0:
            raise ValueError("annulus: inner and outer loops touch or cross")

        # ring k is the straight-chord blend inner -> outer, all sharing inner's
        # wrapping line connectivity; consecutive rings loft into quad layers.  The
        # loops' per-segment tags become the inner (side 1) / outer (side 3) caps.
        rings = [LineMesh((1.0 - t) * A + t * B, inner.lines, closed=True)
                 for t in radial]
        # tags from each loop's per-segment element_tags; a non-empty scalar
        # inner_tag / outer_tag overrides that for the whole ring.
        inner_caps: str | StrArray = (
            inner_tag if inner_tag
            else (inner.element_tags if inner.element_group_tags else ""))
        outer_caps: str | StrArray = (
            outer_tag if outer_tag
            else (outer.element_tags if outer.element_group_tags else ""))
        qm = cls.loft(rings, first_tag=inner_caps, last_tag=outer_caps)
        return _apply_smoothing(qm, smoothing_method)

    # -- factories (closed 3-D surfaces) --------------------------------
    # the six box faces: outward normal n with right-handed tangents (u x v = n),
    # each mapped to its {x,y,z}_{min,max} side key.
    _BOX_FACES = [
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "x_max"),
        ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), "x_min"),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), "y_max"),
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "y_min"),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "z_max"),
        ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), "z_min"),
    ]

    @classmethod
    def box(cls, half_sizes: float | Sequence[float] | FloatArray,
            n: int | Sequence[int] | IntArray, *,
            face_tags: Mapping[str, str] | None = None) -> QuadMesh:
        """Closed box surface centred at the origin: six quad patches welded with
        :meth:`merge`.  ``half_sizes`` is a scalar (cube) or ``(sx, sy, sz)``; ``n``
        is a scalar or ``(nx, ny, nz)`` cells per axis.  ``face_tags`` (keyed
        ``x_min`` / ``x_max`` / ... / ``z_max``) writes each face's dense per-quad
        ``element_tags`` -- e.g. the far-field side it forms; an absent face stays
        untagged so ``merge`` welds shared edges cleanly."""
        hs: FloatArray = np.asarray(half_sizes, dtype=float).ravel()
        if hs.size == 1:
            hs = np.full(3, float(hs[0]))
        elif hs.size != 3:
            raise ValueError("half_sizes must be a scalar or 3 values (sx, sy, sz)")
        na: IntArray = np.asarray(n, dtype=np.int64).ravel()
        if na.size == 1:
            n_axis = (int(na[0]), int(na[0]), int(na[0]))
        elif na.size == 3:
            n_axis = (int(na[0]), int(na[1]), int(na[2]))
        else:
            raise ValueError("n must be a scalar or 3 counts (nx, ny, nz)")
        ft = face_tags or {}
        patches: list[QuadMesh] = []
        for nrm, u, v, key in cls._BOX_FACES:
            nv: FloatArray = np.asarray(nrm, dtype=float)
            uv: FloatArray = np.asarray(u, dtype=float)
            vv: FloatArray = np.asarray(v, dtype=float)
            au = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(uv)))] + 1)
            av = np.linspace(-1.0, 1.0, n_axis[int(np.argmax(np.abs(vv)))] + 1)
            A: FloatArray
            B: FloatArray
            A, B = np.meshgrid(au, av, indexing="ij")
            face = hs * (nv + A[..., None] * uv + B[..., None] * vv)
            patches.append(cls.from_grid(face, element_tag=ft.get(key, "")))
        return cls.merge(patches)

    @classmethod
    def sphere(cls, radius: float, n: int | Sequence[int] | IntArray, *,
               element_tag: str = "sphere") -> QuadMesh:
        """Closed cubed-sphere surface of ``radius`` about the origin: a unit
        :meth:`box` projected radially onto the sphere (same connectivity, so it
        pairs by index with a same-``n`` box for
        :meth:`HexMesh.annulus <nekmeshpy.hexmesh.HexMesh.annulus>`).  Every
        quad carries ``element_tag`` (default ``sphere``)."""
        cube = cls.box(1.0, n)
        pts = radius * cube.points / np.linalg.norm(cube.points, axis=1, keepdims=True)
        return cls(pts, cube.quads,
                   element_tags=np.full(cube.n_quads, element_tag))
