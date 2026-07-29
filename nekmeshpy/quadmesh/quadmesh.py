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
    def blend(cls, a: QuadMesh, b: QuadMesh,
              fractions: FloatArray | Sequence[float]) -> list[QuadMesh]:
        """Linearly morph between two conformal sections ``a`` and ``b`` (identical
        ``quads``, equal point count), one section per fraction ``t`` with points
        ``(1-t)*a + t*b`` -- ``t=0`` reproduces ``a``, ``t=1`` reproduces ``b``.  Each
        result carries ``a``'s ``quads``, ``boundaries`` and ``boundary_tags``
        (positional BC markers follow the morph); per-quad ``element_tags`` are left
        for the consuming ``loft`` caps to assign, so a blended stack lofts directly.
        This is the profile-positioning step behind ``HexMesh.annulus``."""
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
        return [cls((1.0 - t) * A + t * B, a.quads, boundaries=a.boundaries,
                    boundary_tags=a.boundary_tags)
                for t in np.asarray(fractions, dtype=float).ravel()]

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
