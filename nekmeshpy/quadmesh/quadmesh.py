"""Quad mesh of a single cross-section slice.

``QuadMesh`` is a pure container and the quad sibling of
:class:`~nekmeshpy.trimesh.TriMesh`: point coordinates ``points`` (nn,3)
and quad connectivity ``quads`` (nq,4), with matching ``n_points`` / ``n_quads``
size properties.  It carries the two tag systems used throughout the toolkit: a
dense per-quad ``element_tags`` ``(nq,)`` (region/material, ``""`` = untagged), and
tagged boundary edges recorded exactly as
:class:`~nekmeshpy.hexmesh.HexMesh` records faces: ``boundaries`` is an
``(Nbc,2)`` array of ``[quad id (0-based), side (1-4)]`` with a parallel
``boundary_tags`` ``(Nbc,)`` naming each tagged edge.  Side ``s`` spans the local
edge ``EDGE_POINTS[s-1]`` -- side 1 = pt1-2, 2 = pt2-3,
3 = pt3-4, 4 = pt4-1.  Untagged boundary edges are not stored; recover the full
topological outline with :meth:`~QuadMesh.boundary_edges`.

Besides the array constructor, four factory classmethods fill a bounded region
with quads (mirroring the :class:`~nekmeshpy.hexmesh.HexMesh` factories):
:meth:`structured` (transfinite grid over four edge lines), :meth:`ogrid` (butterfly
O-grid inside a closed loop), :meth:`half_ogrid` (half-disc O-grid split along a
spine), and :meth:`annulus` (ring O-grid).  A stack of these slices (sharing
connectivity) is recombined into hexes by
:meth:`~nekmeshpy.hexmesh.HexMesh.loft`, or a single section is swept along
a straight axis by :meth:`~nekmeshpy.hexmesh.HexMesh.extrude`.  One
dimension down, :meth:`extrude` / :meth:`loft` sweep a
:class:`~nekmeshpy.linemesh.LineMesh` into this quad section, carrying the
line's ``element_tags`` onto the swept quads and its tagged boundary **points** onto
the swept side-wall **edges**.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .._typing import BoolArray, FloatArray, IntArray, Point, PointArray, StrArray, Vec3
from ..linemesh import LineMesh
from ..model.fields import validate_layers

#: Boundary-name sentinel meaning "this face is *not* a boundary": a section edge
#: (or swept face / grid side) carrying this name emits **no** boundary row, so it
#: is left as a raw topological surface -- or, when two blocks are stitched, as the
#: welded-away seam.  Marking the touching faces ``NO_BOUNDARY`` before
#: :meth:`~nekmeshpy.hexmesh.HexMesh.merge` lets merge stay a plain concatenate: there is simply no
#: stale tag on the face that becomes interior.  Equal to ``""`` so it also reads
#: as "unnamed" everywhere an empty name is already skipped.
NO_BOUNDARY: str = ""

# default sweep axis / origin for extrude (module-level singletons; read-only)
_Z_AXIS = np.array([0.0, 0.0, 1.0])
_ORIGIN = np.array([0.0, 0.0, 0.0])

# grid side name -> (quad edge side 1-4, axis, which end) for from_grid; mirrors
# HexMesh._GRID_SIDES one dimension down (edge sides match QuadMesh.EDGE_POINTS).
_GRID_EDGES = {
    "x_min": (4, 0, 0), "x_max": (2, 0, -1),
    "y_min": (1, 1, 0), "y_max": (3, 1, -1),
}


def _apply_smoothing(qm: QuadMesh, smoothing_method: str | None) -> QuadMesh:
    """Reposition ``qm``'s interior points in place via the sibling
    :mod:`~nekmeshpy.quadmesh.smoothing` module (``None`` leaves the raw
    algebraic fill).  Imported lazily to avoid an import cycle."""
    if smoothing_method is not None:
        from . import smoothing
        smoothing.set_section_smoothing(qm, smoothing_method)
    return qm


def _check_boundary(obj: LineMesh, name: str,
                    closed: bool, min_pts: int) -> PointArray:
    """Validate a :class:`~nekmeshpy.linemesh.LineMesh` factory argument,
    returning its ``(N,3)`` points.  Enforces the required open/closed topology (so
    the distinction holds at runtime, not just for the type checker), a minimum point
    count, and finite coordinates."""
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
    ``(Nbc,2)`` = ``[quad id, side 1-4]`` with a parallel ``boundary_tags`` --
    mirroring :class:`~nekmeshpy.hexmesh.HexMesh` one dimension down.  Build a
    section with the factory classmethods (:meth:`structured` / :meth:`ogrid` /
    :meth:`half_ogrid` / :meth:`annulus` / :meth:`from_grid`), or sweep a
    :class:`~nekmeshpy.linemesh.LineMesh` with :meth:`extrude` / :meth:`loft`;
    both tag systems then ride up onto the swept hex faces."""

    def __init__(
        self,
        points: PointArray,
        quads: IntArray,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
    ) -> None:
        """Construct from arrays: ``points`` ``(P,3)``, ``quads`` ``(Q,4)`` indices
        (CCW), an optional dense per-quad ``element_tags`` ``(Q,)`` (``""`` =
        untagged; length must equal ``len(quads)``), and an optional tagged-boundary
        list mirroring :class:`~nekmeshpy.hexmesh.HexMesh`: ``boundaries``
        ``(Nbc,2)`` = ``[quad id (0-based), side (1-4)]`` with a parallel
        ``boundary_tags`` ``(Nbc,)`` naming each tagged edge.  Side ``s`` spans the
        local edge ``EDGE_POINTS[s-1]`` (side 1 = pt1-2, 2 = pt2-3,
        3 = pt3-4, 4 = pt4-1).  Untagged boundary edges are *not* stored here; recover
        the full topological outline with :meth:`boundary_edges`.  Use the factory
        classmethods (:meth:`structured` / :meth:`ogrid` / :meth:`half_ogrid` /
        :meth:`annulus`) for the usual build paths."""
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
        # dense per-quad region/material tag ("" = untagged), carried onto the swept
        # hexes by HexMesh.loft / extrude (the element-tag chain, one level up).
        if element_tags is None:
            self.element_tags: StrArray = np.full(
                self.quads.shape[0], "", dtype=np.str_)
        else:
            et = np.asarray(element_tags, dtype=np.str_).reshape(-1)
            if et.shape[0] != self.quads.shape[0]:
                raise ValueError("element_tags length (%d) must match quads (%d)"
                                 % (et.shape[0], self.quads.shape[0]))
            self.element_tags = et
        # tagged boundary edges [quad id, side 1-4] parallel with boundary_tags,
        # carried to the swept side faces by HexMesh.loft.  A NO_BOUNDARY tag marks
        # an edge that should stay untagged (e.g. one welded away by merge).
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
        """Sorted unique tags of the tagged boundary edges present on the section
        (a Nek BC code / id is assigned only at export)."""
        return sorted(set(self.boundary_tags.tolist()))

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-quad element tags present on the section."""
        return sorted({t for t in self.element_tags.tolist() if t})

    # -- quality ---------------------------------------------------------
    def scaled_jacobian(self) -> FloatArray:
        """Per-quad minimum corner scaled Jacobian ``(n_quads,)`` (see
        :func:`nekmeshpy.quadmesh.quality.scaled_jacobian`)."""
        from . import quality
        return quality.scaled_jacobian(self.points, self.quads)

    def quality_summary(self) -> dict[str, Any]:
        """Aggregate scaled-Jacobian statistics (see
        :func:`nekmeshpy.quadmesh.quality.summary`)."""
        from . import quality
        return quality.summary(self.points, self.quads)

    @staticmethod
    def _order_bnd(
        bnd: Sequence[Sequence[int]] | IntArray,
        names: Sequence[str] | StrArray,
    ) -> tuple[IntArray, StrArray]:
        """Stably order boundary rows by ``(quad id, side)`` so a section is
        independent of insertion order, applying the same permutation to the
        parallel tags array (mirrors :meth:`HexMesh._order_bnd`)."""
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
        ``boundaries`` (the wall subset -- a half-disk section's boundary is
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
        the absolute coincidence distance (default ``1e-7`` x the extent).

        Mirrors :meth:`~nekmeshpy.hexmesh.HexMesh.merge`: connectivity welds by point, but the tagged
        ``boundaries`` are ``[quad id, side]`` rows, so they simply concatenate with
        each block's quad ids offset -- ``boundary_tags`` rides along, and the dense
        ``element_tags`` concatenate one per quad.  A face that becomes an interior
        seam is *not* auto-dropped; leave the touching edges untagged (or
        ``NO_BOUNDARY``) so no stale tag lands on the weld."""
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
            quad_list.append(point_id[m.quads + noff])   # local -> concat -> welded id
            etag_list.append(m.element_tags)
            if m.boundaries.shape[0]:
                b: IntArray = m.boundaries.copy()
                b[:, 0] += qoff                          # quad ids shift; sides are local
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
        """Build quads from a structured point grid ``P`` ``(ni+1,nj+1,3)`` (the
        sibling of :meth:`~nekmeshpy.hexmesh.HexMesh.from_grid` one dimension
        down).  ``edge_tags`` maps side names (``x_min`` / ``x_max`` / ``y_min`` /
        ``y_max``) to boundary **tags** on the four outer edges; a side left out (or
        mapped to ``NO_BOUNDARY``) emits no boundary row -- use that for an edge
        that will be welded away by :meth:`merge`, so merge stays a plain concatenate
        with no stale tag.  ``element_tag`` (default untagged) is written to every
        quad's dense ``element_tags`` (e.g. tag a whole cube-face patch with the
        far-field side it forms, then :meth:`~nekmeshpy.hexmesh.HexMesh.annulus`
        turns it into a wall face)."""
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
        """Normalize a cap tag to one tag per section line (length ``L``).  A scalar
        ``str`` tags the whole cap (``""`` = untagged everywhere); an array-like is a
        per-line tag (``""`` entries stay untagged), used by :meth:`annulus` to tag a
        cap from a ring loop's per-segment ``element_tags`` (mirrors
        :meth:`~nekmeshpy.hexmesh.HexMesh._cap_tags` one dimension down)."""
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
        """Sweep a :class:`~nekmeshpy.linemesh.LineMesh` a distance
        ``length`` along ``axis`` into a quad section (the line sibling of
        :meth:`~nekmeshpy.hexmesh.HexMesh.extrude`, one dimension down).

        The ``line`` is taken as a real curve in 3-D and translated **rigidly** along
        ``axis``; ``origin`` shifts the whole section by a constant offset.  ``layers``
        are the normalized copy-line positions along ``axis`` as fractions of
        ``length`` -- strictly increasing values in ``[0, 1]`` with the last ``1`` --
        so ``layers.size - 1`` quad layers span ``layers[0]..1`` (same convention as
        :meth:`~nekmeshpy.hexmesh.HexMesh.extrude`).

        The line's dense ``element_tags`` ride onto the swept quads (each line sweeps
        into a column of quads), its tagged boundary **points** ride onto the swept
        side-wall **edges** (vertex-0 point -> quad side 4, vertex-1 point -> side 2),
        and ``first_tag`` / ``last_tag`` name the near / far cap edges (sides 1 / 3).
        The straight special case of :meth:`loft`."""
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
        """Loft a stack of conformal :class:`~nekmeshpy.linemesh.LineMesh`
        profiles into a quad section (the general primitive behind :meth:`extrude`,
        one dimension down from :meth:`~nekmeshpy.hexmesh.HexMesh.loft`).

        ``slices`` is ``nz+1`` line profiles sharing the same ``lines`` connectivity,
        ``element_tags``, and tagged ``boundaries``; consecutive profiles form ``nz``
        quad layers.  For line ``(a, b)`` at layer ``i`` the column quad is
        ``[a_i, b_i, b_{i+1}, a_{i+1}]`` (side 1 = line@i, 2 = vertex-b wall, 3 =
        line@{i+1}, 4 = vertex-a wall), so the construction is topologically uniform
        for every line and layer -- no winding flip is needed.  The line's
        ``element_tags`` ride onto every quad in its column; a tagged boundary point
        (side 1 -> local vertex 0, side 2 -> vertex 1) rides onto the swept wall edge
        (vertex 0 -> quad side 4, vertex 1 -> side 2), skipping the
        ``NO_BOUNDARY`` sentinel; ``first_tag`` / ``last_tag`` name the near / far
        cap edges (side 1 of the layer-0 quads / side 3 of the last-layer quads) --
        each a scalar ``str`` tagging the whole cap, or a per-line array (one tag per
        section line, ``""`` = untagged) so a cap can be tagged from a ring loop's own
        per-segment ``element_tags`` (see :meth:`annulus`)."""
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

        # tagged boundary point -> swept wall edge (per layer); vertex 0 (side 1) ->
        # quad side 4, vertex 1 (side 2) -> quad side 2.
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
        # caps: scalar tags the whole cap, an array tags per section line l0.
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
    # Each fills a bounded region with quads and returns a QuadMesh whose
    # ``boundaries`` is the outer boundary.  The topology is chosen by *which*
    # factory you call -- structured and o-grid are different topologies -- and an
    # optional ``smoothing_method`` (``"conduction"`` / ``"winslow"`` /
    # ``"bilinear"``; ``None`` = raw algebraic fill) repositions the interior
    # points via :func:`nekmeshpy.quadmesh.smoothing.set_section_smoothing`.
    @classmethod
    def structured(cls, edges: list[LineMesh], *,
                   boundary_tags: Mapping[str, str] | None = None,
                   smoothing_method: str | None = None) -> QuadMesh:
        """Transfinite (Coons-patch) quad grid over the surface bounded by four
        edge lines ``edges = [bottom, right, top, left]`` given in CCW loop order
        (each an open :class:`~nekmeshpy.linemesh.LineMesh` of 3-D points;
        the Coons blend runs directly on the 3-D edges, so the section may lie in any
        plane).  The lines must share corners, i.e. form a closed loop: ``bottom``
        ends where ``right`` begins, ``right`` where ``top`` begins, and so on.

        The grid resolution and node distribution come **directly from the edge
        lines' own points** -- there is no resampling.  ``bottom`` and ``top``
        must carry the same number of points (``nx+1``, the u-direction) and
        ``left`` and ``right`` the same number (``ny+1``, the v-direction); the
        interior is filled by bilinearly-blended transfinite (Coons) interpolation,
        giving ``nx`` x ``ny`` cells.  Because the caller supplies the sampling, a
        graded edge (e.g. built with
        :meth:`~nekmeshpy.linemesh.LineMesh.resample` at clustered
        fractions) yields a graded grid -- thinner cells near a wall -- directly.
        Straight, uniformly-sampled edges reduce this exactly to a uniform bilinear
        grid.

        Each side is named at the **lowest level** from its own edge line's uniform
        ``element_tags`` (a single non-empty tag shared by every segment of that
        edge), so the tag rides through
        :meth:`~nekmeshpy.hexmesh.HexMesh.loft` / ``extrude`` onto the swept side
        faces.  ``boundary_tags`` (keyed by side -- ``"bottom"`` / ``"right"`` /
        ``"top"`` / ``"left"``, matching the ``edges`` order) is the **override**: a
        non-empty entry replaces that side's edge tag, and a present-but-empty entry
        (the ``NO_BOUNDARY`` sentinel or ``""``) suppresses the side, naming it
        but emitting no boundary row (e.g. an edge welded away by :meth:`merge`).  A
        side with neither an override nor a uniform edge tag stays untagged.
        """
        if len(edges) != 4:
            raise ValueError("structured needs exactly 4 edge lines "
                             "[bottom, right, top, left]")
        bottom, right, top, left = edges
        for nm, e in (("bottom", bottom), ("right", right),
                      ("top", top), ("left", left)):
            _check_boundary(e, "structured " + nm + " edge", False, 2)
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
        # each side is named by its own edge line's uniform element tag (the
        # lowest-level place to declare it); a non-empty boundary_tags[side]
        # OVERRIDES that, and a present-but-empty entry (NO_BOUNDARY / "")
        # suppresses the side entirely.
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
        """Butterfly O-grid filling the closed ``boundary``
        (a closed :class:`~nekmeshpy.linemesh.LineMesh`, e.g. from
        ``LineMesh.circle``): a central
        ``n_side x n_side`` block at the loop centroid, surrounded by O-ring
        layers blending its perimeter out to the boundary (no collapsed cell at the
        centre).  ``center_scale`` sizes the block (fraction of the mean radius).

        The whole grid is built in 3-D with **no projection to a plane**, so a
        curvy / non-planar boundary keeps its true shape (the wall ring sits exactly
        on the boundary, resampled by arc length to ``4*n_side`` points).  The
        block-and-ring construction is only an initial guess; pass
        ``smoothing_method="conduction"`` to relax the interior harmonically onto the
        curved surface spanned by the fixed boundary ring.  For a planar boundary the
        result stays coplanar with it.

        ``radial`` are the O-ring layer positions with the **initial position
        explicit** (the same convention as :meth:`half_ogrid`,
        :meth:`annulus`, and :meth:`~nekmeshpy.hexmesh.HexMesh.extrude`'s
        ``layers``): strictly increasing values in ``[0, 1]`` -- ``radial[0]`` is the
        central block perimeter (``0``) and the last is ``1`` (the wall) -- so
        ``radial.size - 1`` rings blend the block perimeter out to the boundary.
        Pass ``geometric_spacing(k, ratio)`` to cluster rings toward the wall.

        The outer ring (the section's single wall) is named at the **lowest level**
        from ``boundary``'s own per-line ``element_tags`` (see
        :class:`~nekmeshpy.linemesh.LineMesh`), resampled per segment to the
        ``4*n_side`` wall points, so the tag rides through
        :meth:`~nekmeshpy.hexmesh.HexMesh.loft` / ``extrude`` onto the swept side
        faces (see :meth:`structured`).  A non-empty scalar ``wall_tag`` is the
        **override** -- it replaces the loop tags and names the whole wall; left
        empty with an untagged boundary the wall stays untagged.

        ``n_side`` counts the central block cells per side and ``radial`` the ring
        layers of the butterfly topology; they are not interchangeable with the
        ``nx``/``ny`` of :meth:`structured`."""
        if n_side < 1:
            raise ValueError("ogrid needs n_side >= 1")
        if not 0.0 < center_scale < 1.0:
            raise ValueError("ogrid needs center_scale in (0, 1)")
        radial = validate_layers(radial, "ogrid radial")
        n_radial = radial.size - 1
        _check_boundary(boundary, "ogrid boundary", True, 3)
        # The butterfly is built entirely in 3-D -- nothing is projected to a plane,
        # so a curvy / non-planar boundary keeps its true shape.  The wall ring is
        # the boundary resampled (by 3-D arc length) to the P = 4*n_side ring points;
        # the central block corners are 4 of those points pulled toward the centroid,
        # bilinearly filled; the rings are straight-chord blends between the block
        # perimeter and the wall.  This is only an initial guess -- pass
        # ``smoothing_method="conduction"`` to relax the interior harmonically onto
        # the (possibly curved) surface spanned by the fixed boundary ring.
        P = 4 * n_side
        wall_loop = boundary.resample(np.linspace(0.0, 1.0, P, endpoint=False))
        outer_pos: PointArray = wall_loop.points                    # (P,3) true wall
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

        # wall edges = side 1 of the outermost ring's quads (built as
        # [outer[k], outer[kn], inner[kn], inner[k]] so side 1 = outer[k]-outer[kn]);
        # the outermost ring occupies quad rows n_side^2 + (n_radial-1)*P onward.
        wall_q0 = n_side * n_side + (n_radial - 1) * P
        # the wall is named at the lowest level from the boundary loop's per-line
        # element tags (wall edge m tracks resampled wall segment m); a non-empty
        # scalar wall_tag OVERRIDES that and tags the whole wall; else untagged.
        wall_seg = wall_loop._seg_tags()
        bnd: list[list[int]] = []
        names: list[str] = []
        for m in range(P):
            nm = wall_tag if wall_tag else (wall_seg[m] if wall_seg is not None else "")
            if nm:
                bnd.append([wall_q0 + m, 1])
                names.append(nm)
        qm = cls(points, quads, *cls._order_bnd(bnd, names))
        return _apply_smoothing(qm, smoothing_method)

    @classmethod
    def half_ogrid(cls, arc: LineMesh, spine: LineMesh,
                   radial: FloatArray, *, center_scale: float = 0.5,
                   wall_tag: str = "",
                   smoothing_method: str | None = None) -> QuadMesh:
        """Structured HALF-circle O-grid over a half-disk split along the ``spine``
        line (A1..A2); the wall ``arc`` (``(4*Ntheta+1, 3)``, arc[0]=A1,
        arc[-1]=A2) is the open boundary, recorded as ``boundaries`` (both open
        :class:`~nekmeshpy.linemesh.LineMesh`).  ``radial``
        are the O-ring layer positions with the **initial position explicit** (the
        same convention as :meth:`ogrid` / :meth:`annulus`): strictly increasing
        values in ``[0, 1]`` -- ``radial[0]`` is the inner block perimeter (``0``)
        and the last is ``1`` (the wall) -- so ``radial.size - 1`` rings are laid
        out; ``center_scale`` is the inner block extent as a fraction of the spine.
        The ``arc`` wall is named at the **lowest level** from the arc's own
        per-segment ``element_tags`` (wall edge ``k`` tracks arc segment ``k``), so
        the tag rides through :meth:`~nekmeshpy.hexmesh.HexMesh.loft` / ``extrude``
        onto the swept side faces (see :meth:`structured`).  A non-empty scalar
        ``wall_tag`` is the **override** -- it replaces the arc tags and names the
        whole wall; left empty with an untagged arc the wall stays untagged."""
        apts = _check_boundary(arc, "half_ogrid arc", False, 5)   # (na,3) backing array
        _check_boundary(spine, "half_ogrid spine", False, 2)
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

        # wall arc edges = side 3 of the outermost ring's quads (appended as
        # [a[k], a[k+1], arc[k+1], arc[k]] so side 3 = arc[k+1]-arc[k]); the outer
        # ring occupies quad rows (ni*nj) + (Nr-1)*(4*Nt) onward.  Wall edge k
        # tracks arc segment k (1:1).
        wall_q0 = ni * nj + (Nr - 1) * (4 * Nt)
        # the wall is named at the lowest level from the arc's per-segment element
        # tags; a non-empty scalar wall_tag OVERRIDES that for the whole wall.
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
        """Ring O-grid filling the region *between* an inner and an outer closed
        loop (both closed :class:`~nekmeshpy.linemesh.LineMesh`) -- e.g. a
        circular body inside a square far-field box for a flow-past-cylinder section.

        The two loops are paired **by index**: they must carry the same number of
        points ``N`` (the azimuthal resolution), and point ``i`` of ``inner`` is
        joined radially to point ``i`` of ``outer``.  The caller is responsible for
        supplying a correctly sized and oriented outer loop -- there is no
        resampling here.  To align a coarse far-field box loop to a finer body loop
        so the radial lines do not skew, project it first with
        :meth:`~nekmeshpy.linemesh.LineMesh.radial_match`::

            outer = LineMesh.loop([...box corners...]).radial_match(inner)

        ``radial`` are the ring positions with the **initial position explicit**
        (the same convention as :meth:`ogrid` / :meth:`half_ogrid` and
        :meth:`~nekmeshpy.hexmesh.HexMesh.extrude`'s ``layers``): strictly
        increasing values in ``[0, 1]`` -- ``radial[0]`` is the inner ring (``0`` for
        a ring flush with the body, or e.g. ``0.5`` to start the mesh halfway out)
        and the last is ``1`` (the outer loop) -- so ``radial.size - 1`` ring layers
        blend ``radial[0]`` -> outer.  Pass
        ``geometric_spacing(k, ratio)`` to cluster rings toward the inner body for a
        boundary layer, or ``uniform_spacing(k)`` / ``numpy.linspace(a, 1, k + 1)``.
        An ``smoothing_method`` holds the section's topological boundary (the inner
        and outer rings) fixed while relaxing the ring interior.

        The ring blend runs directly in 3-D (no projection to a plane), so the two
        loops need not be planar or coplanar: a curvy / non-planar inner-outer pair
        keeps its true shape and an ``smoothing_method`` (e.g. ``conduction``)
        relaxes the ring interior onto the resulting curved surface.

        Boundary tags come from **the lowest level -- the loops themselves**: if a
        loop carries per-line ``element_tags`` (see
        :class:`~nekmeshpy.linemesh.LineMesh`), each ring edge is tagged from
        the corresponding loop segment (they pair by index), so a named far-field box
        splits the outer ring into distinct sides (inlet / outlet / top / bottom)
        automatically -- tag once on the loop, e.g.::

            outer = LineMesh.loop([...4 corners...],
                                  element_tags=["bottom", "outlet", "top", "inlet"]
                                  ).radial_match(inner)   # tags ride through the match

        as ``examples/flow_past_cylinder.py`` does.  A non-empty scalar ``inner_tag``
        / ``outer_tag`` is the **override** -- it replaces that loop's per-line tags
        and names the whole inner / outer ring (an embedded body and a uniform, round
        far field).  Either way the tags ride on through
        :meth:`~nekmeshpy.hexmesh.HexMesh.loft` / ``extrude`` onto the swept side
        faces (see :meth:`structured`).

        Built by :meth:`loft`-ing ``radial.size`` blended rings (the ring layers) --
        the periodic ring topology rides in the loops' wrapping ``lines``, exactly as
        :meth:`~nekmeshpy.hexmesh.HexMesh.annulus` rides on a closed surface's
        ``quads`` one dimension up; the inner / outer rings are the loft's near /
        far caps (quad sides 1 / 3).  Gives ``N x (radial.size - 1)`` quads.
        ``radial`` here sets the ring layers and is not interchangeable with the
        ``nx``/``ny`` of :meth:`structured` or the ``n_side`` of :meth:`ogrid`."""
        radial = validate_layers(radial, "annulus radial")
        A: FloatArray = _check_boundary(inner, "annulus inner", True, 3)   # (N,3)
        B: FloatArray = _check_boundary(outer, "annulus outer", True, 3)   # (N,3)
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "annulus: inner and outer loops must have equal point counts "
                "(got %d, %d); align the outer loop to the inner first, e.g. "
                "outer.radial_match(inner)" % (A.shape[0], B.shape[0]))
        if float(np.min(np.linalg.norm(B - A, axis=1))) <= 0.0:
            raise ValueError("annulus: inner and outer loops touch or cross")

        # ring k is the straight-chord blend inner(radial[0]) -> outer(radial[-1]=1),
        # all sharing inner's (wrapping) line connectivity; consecutive rings loft
        # into quad layers.  The periodic ring topology rides in inner.lines' closing
        # segment [N-1, 0] -- no modular arithmetic here (mirrors HexMesh.annulus,
        # whose wrap rides in the closed surface's quads).  The rings carry no
        # element_tags -> the quads stay region-untagged; the loops' per-segment tags
        # become the inner (near, side 1) / outer (far, side 3) loft caps.
        rings = [LineMesh((1.0 - t) * A + t * B, inner.lines, closed=True)
                 for t in radial]
        # tags come from the lowest level: each loop's per-segment element_tags name
        # its ring (a named far-field box tags each side).  A non-empty scalar
        # inner_tag / outer_tag OVERRIDES that and tags the whole ring (an embedded
        # body / a uniform round far field); an empty ("") tag leaves it untagged.
        inner_caps: str | StrArray = (
            inner_tag if inner_tag
            else (inner.element_tags if inner.element_group_tags else ""))
        outer_caps: str | StrArray = (
            outer_tag if outer_tag
            else (outer.element_tags if outer.element_group_tags else ""))
        qm = cls.loft(rings, first_tag=inner_caps, last_tag=outer_caps)
        return _apply_smoothing(qm, smoothing_method)
