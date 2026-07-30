"""1-D mesh container: line elements ``(L,2)`` over a shared ``(N,3)`` point array.

The line sibling of QuadMesh/HexMesh; it can branch rather than being a single
ordered path. It carries a dense per-line ``element_tags`` and sparse tagged
``boundaries`` (a line's boundary is a point), both of which sweep up on extrude.
Open vs closed is a property of the ``lines`` connectivity itself -- a loop is a
cycle of line elements with no degree-1 end point -- and is stored nowhere;
factories build the common cases (``loft`` / ``open`` / ``loop`` / ``line`` /
``circle`` / ``rectangle`` / ``from_segments``) and every curve is meshed exactly at
the points given -- there is no resampling here.

``lines`` is a **required** constructor argument: the container never invents
connectivity, so there is nothing in ``LineMesh`` that could imply a wrap.  The
bottom rung of the uniform sweep primitive, :meth:`LineMesh.loft`, is what authors it
-- one dimension below ``QuadMesh.loft``/``HexMesh.loft``, each "profile" is a single
point and the rungs joining consecutive profiles *are* the line elements, with
``loop=True`` adding the closing rung from the last point back to the first.  ``open``
and ``loop`` are thin wrappers over it (``loop=False`` / ``loop=True``).

The parametric shape factories live beside this file as free functions and are
bound onto the class in ``linemesh/__init__.py``: ``_closed.py`` (``circle`` /
``rectangle``) and ``_open.py`` (``line``). This file stays a pure container -- the
core constructors (``loft`` / ``open`` / ``loop`` / ``from_segments``) and all
queries -- so adding a shape touches only the sibling module, never this one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._typing import (
    BoolArray,
    FloatArray,
    IntArray,
    PointArray,
    StrArray,
)
from ..model.fields import gll_nodes, reject_loop_caps


def _as_points(points: NDArray[Any]) -> PointArray:
    """Normalize an array-like to a validated ``(N,3)`` float point array, raising
    the one actionable "boundaries live in 3-D" error for anything else.  Shared by
    ``LineMesh.__init__`` and :meth:`LineMesh.loft` so both report it identically."""
    a: PointArray = np.asarray(points, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(
            f"boundary points must be (N,3) 3-D coordinates; got "
            f"{a.shape} -- add a z column (all boundaries live in 3-D)")
    return a


class LineMesh:
    """A 1-D mesh: an ``(N,3)`` point array with ``(L,2)`` line connectivity, a
    dense per-line ``element_tags``, and ``[line id, side 1-2]`` ``boundaries`` with
    parallel ``boundary_tags``. Build with ``loft`` / ``open`` / ``loop`` /
    ``circle`` / ``from_segments``."""

    # local line "edges": row s-1 is side s -> the single local vertex it names.
    EDGE_POINTS = np.array([[0], [1]], dtype=np.int64)

    def __init__(
        self,
        points: NDArray[Any],
        lines: IntArray,
        element_tags: StrArray | Sequence[str] | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        *,
        order: int = 1,
        interior: PointArray | None = None,
    ) -> None:
        """Construct from arrays: ``points`` ``(N,3)`` (must be 3-D), the **required**
        ``lines`` ``(L,2)`` connectivity, optional dense ``element_tags`` ``(L,)``,
        and an optional tagged-boundary list ``boundaries`` ``(Nbc,2)`` with parallel
        ``boundary_tags``.

        The container never synthesizes connectivity -- there is no "consecutive
        chain" default and therefore nothing here that could imply a wrap.  Callers
        either own their ``lines`` outright (``from_segments``' chained loop,
        ``merge``'s rewelded lines, ``blend``'s copy of ``a.lines``, the quad/hex
        edge meshes built from ``conform.unique_edges``) or author them one rung up
        with :meth:`loft` -- and ``open`` / ``loop`` are just ``loft(..., loop=False)``
        / ``loft(..., loop=True)``.

        ``order`` is the global polynomial order (default 1 = linear).  At
        ``order > 1`` pass ``interior``: the per-line *private* interior nodes
        ``(L, order-1, 3)`` in ascending GLL order, i.e. the nodes strictly between
        each line's two endpoints.  A line element has no shared edges or faces, so
        its endpoints (owned by ``points[lines]``) are its only conformal nodes and
        there is nothing to reconcile between lines -- ``interior`` is the whole
        high-order state of a ``LineMesh``.

        ``re2`` export stays linear; only ``vtu`` reads the high-order nodes."""
        self.points: PointArray = _as_points(points)
        self.lines: IntArray = np.asarray(lines, dtype=np.int64).reshape(-1, 2)

        if element_tags is None:
            self.element_tags: StrArray = np.full(
                self.lines.shape[0], "", dtype=np.str_)
        else:
            et = np.asarray(element_tags, dtype=np.str_).reshape(-1)
            if et.shape[0] != self.lines.shape[0]:
                raise ValueError(
                    "element_tags length (%d) must match lines (%d)"
                    % (et.shape[0], self.lines.shape[0]))
            self.element_tags = et

        # tagged boundary points [line id, side 1-2] parallel with boundary_tags.
        self.boundaries: IntArray = (
            np.zeros((0, 2), np.int64) if boundaries is None
            else np.asarray(boundaries, np.int64).reshape(-1, 2))
        self.boundary_tags: StrArray = (
            np.empty(0, dtype=np.str_) if boundary_tags is None
            else np.asarray(boundary_tags, dtype=np.str_).reshape(-1))
        if self.boundary_tags.shape[0] != self.boundaries.shape[0]:
            raise ValueError("boundary_tags length (%d) must match boundaries (%d)"
                             % (self.boundary_tags.shape[0], self.boundaries.shape[0]))

        self._order = int(order)
        #: ``(L, order-1, 3)`` per-line private high-order interior nodes (ascending GLL
        #: order).  A line has no shared edges/faces, so its endpoints (owned by
        #: ``points[lines]``) are its only conformal nodes and every interior node is
        #: private -- there is nothing to reconcile between lines.  Empty at order 1.
        self.interior: PointArray = self._check_interior(interior)

    def _check_interior(self, interior: PointArray | None) -> PointArray:
        """Validate the **native** per-line private interior ``(L, order-1, 3)`` and
        return it as float.  Nothing is shared between line elements beyond the
        endpoints in ``points``, so there is no corner check to run here -- the corners
        are single-sourced by construction.  ``interior=None`` is allowed only at
        order 1 (empty interior)."""
        order = self._order
        if order < 1:
            raise ValueError("LineMesh: order must be >= 1, got %d" % order)
        n_lines = self.lines.shape[0]
        k = order - 1
        if interior is None:
            if order > 1:
                raise ValueError(
                    "LineMesh: order %d > 1 requires the per-line interior nodes "
                    "(pass interior=(L, order-1, 3), or build the curve with a "
                    "factory such as LineMesh.circle(..., order=%d))"
                    % (order, order))
            return np.zeros((n_lines, 0, 3), dtype=float)
        ia: PointArray = np.asarray(interior, dtype=float)
        if ia.shape != (n_lines, k, 3):
            raise ValueError(
                "LineMesh: interior must be (L, order-1, 3) = (%d,%d,3), got %s"
                % (n_lines, k, ia.shape))
        return ia

    @property
    def order(self) -> int:
        """Global polynomial order (1 = linear)."""
        return self._order

    # -- construction factories -----------------------------------------
    @classmethod
    def loft(
        cls,
        points: NDArray[Any],
        *,
        loop: bool = False,
        element_tags: StrArray | Sequence[str] | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        first_tag: str = "",
        last_tag: str = "",
        order: int = 1,
        interior: PointArray | None = None,
    ) -> LineMesh:
        """Loft a stack of point "profiles" into a 1-D mesh -- the bottom rung of the
        uniform sweep primitive shared with
        :meth:`QuadMesh.loft <nekmeshpy.quadmesh.QuadMesh.loft>` and
        :meth:`HexMesh.loft <nekmeshpy.hexmesh.HexMesh.loft>`.

        One dimension below a quad loft each profile is a **single point**, so the
        rungs joining consecutive profiles *are* the line elements: ``points``
        ``(N,3)`` lofts into the consecutive chain ``[[0,1], ..., [N-2,N-1]]``, and
        ``loop=True`` appends one more rung -- from the last point back to the first
        (``[N-1,0]``) -- closing the curve.  The seam rung is appended exactly once
        and no profile is duplicated, so a lofted loop carries ``N`` points and ``N``
        lines with no degree-1 end.  (Note the unavoidable name collision: the
        ``loop=`` keyword here is the periodic flag, while :meth:`loop` is the closed
        factory that wraps this method with ``loop=True``.)

        ``element_tags`` is the dense per-line tag array (line ``m`` = point ``m`` ->
        ``m+1``, and for ``loop=True`` line ``N-1`` = point ``N-1`` -> ``0``);
        ``boundaries`` / ``boundary_tags`` are passed through verbatim.
        ``first_tag`` / ``last_tag`` name the near / far **end points** of the chain
        (the 1-D end caps: line ``0`` side ``1`` and line ``L-1`` side ``2``).  A
        closed sweep has no near/far cap, so passing either with ``loop=True``
        raises ``ValueError`` rather than silently dropping it.

        At ``order > 1`` an explicit ``interior`` ``(L, order-1, 3)`` is used as-is
        (that is how ``circle`` stamps true-arc nodes); when it is omitted each line's
        private interior is built here as the **straight GLL blend** between its two
        endpoints, which is exactly what a straight-sided curve wants."""
        pts = _as_points(points)
        n = pts.shape[0]
        if loop:
            reject_loop_caps("LineMesh.loft", first_tag, last_tag)
        idx = np.arange(n, dtype=np.int64)
        if n < 2:
            lines: IntArray = np.zeros((0, 2), dtype=np.int64)
        elif loop:
            lines = np.column_stack([idx, np.roll(idx, -1)])
        else:
            lines = np.column_stack([idx[:-1], idx[1:]])

        bnd = boundaries
        names = boundary_tags
        if first_tag or last_tag:
            rows = [[int(r[0]), int(r[1])]
                    for r in np.asarray(bnd if bnd is not None else
                                        np.zeros((0, 2), np.int64),
                                        dtype=np.int64).reshape(-1, 2)]
            tags = [str(t) for t in np.asarray(
                names if names is not None else np.empty(0, dtype=np.str_),
                dtype=np.str_).reshape(-1).tolist()]
            L = lines.shape[0]
            if first_tag and L:
                rows.append([0, 1])
                tags.append(first_tag)
            if last_tag and L:
                rows.append([L - 1, 2])
                tags.append(last_tag)
            bnd, names = cls._order_bnd(rows, tags)

        if order > 1 and interior is None:
            # straight GLL blend between each line's two endpoints -- the same
            # expression the straight-sided factories (``line`` / ``rectangle``) use.
            a: PointArray = pts[lines[:, 0]]
            b: PointArray = pts[lines[:, 1]]
            g = gll_nodes(order)[1:order]              # interior GLL nodes only
            interior = a[:, None, :] + g[None, :, None] * (b - a)[:, None, :]
        return cls(pts, lines, element_tags, bnd, names,
                   order=order, interior=interior)

    @classmethod
    def open(
        cls,
        points: NDArray[Any],
        element_tags: StrArray | Sequence[str] | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        *,
        order: int = 1,
        interior: PointArray | None = None,
    ) -> LineMesh:
        """An open line mesh through ``points`` in order: a consecutive chain.
        A thin wrapper over :meth:`loft` with ``loop=False``.
        Pass ``order`` with ``interior`` (the per-line private high-order nodes) to
        carry high-order geometry (see ``__init__``); omit ``interior`` at
        ``order > 1`` and ``loft`` places the straight GLL blend."""
        return cls.loft(points, loop=False, element_tags=element_tags,
                        boundaries=boundaries, boundary_tags=boundary_tags,
                        order=order, interior=interior)

    @classmethod
    def loop(
        cls,
        points: NDArray[Any],
        element_tags: StrArray | Sequence[str] | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        *,
        order: int = 1,
        interior: PointArray | None = None,
    ) -> LineMesh:
        """A closed loop through ``points`` (last rejoins first): ``N`` line
        elements. A thin wrapper over :meth:`loft` with ``loop=True``, which appends
        the single closing rung ``[N-1, 0]`` -- the loop's closedness lives in the
        ``lines`` ``loft`` authors (``[[0,1], ..., [N-1,0]]``), nowhere else.

        A per-element ``element_tags`` (element ``m`` = point ``m`` ->
        ``(m+1) % N``) is carried through the ordered ops and consumed by the section
        factories, e.g. to split a far-field box into named sides. Pass ``order``
        with ``interior`` (the per-line private high-order nodes) to carry
        high-order geometry (see ``__init__``)."""
        return cls.loft(points, loop=True, element_tags=element_tags,
                        boundaries=boundaries, boundary_tags=boundary_tags,
                        order=order, interior=interior)

    @classmethod
    def from_segments(cls, segs: FloatArray | None) -> LineMesh | None:
        """Chain unordered 3D segments into a single closed ordered loop (the
        largest connected component), or ``None`` if ``segs`` forms no loop."""
        if segs is None or len(segs) == 0:
            return None
        segs = np.asarray(segs, dtype=float)
        ns = segs.shape[0]
        pts_raw = np.vstack([segs[:, 0:3], segs[:, 3:6]])

        # weld coincident endpoints on a scale-relative grid
        scl = float(np.max(pts_raw.max(axis=0) - pts_raw.min(axis=0)))
        tol = 1e-6 * scl if scl > 0 else 1.0
        key = np.round(pts_raw / tol).astype(np.int64)
        _, ic = np.unique(key, axis=0, return_inverse=True)
        ic = ic.ravel()
        npts = int(ic.max()) + 1

        coord = np.zeros((npts, 3))
        cnt = np.zeros(npts)
        for i in range(2 * ns):
            coord[ic[i], :] += pts_raw[i, :]
            cnt[ic[i]] += 1
        coord = coord / cnt[:, None]

        # node -> incident neighbours and each node pair -> its segment indices,
        # so the walk finds the next segment by incidence.
        seg_ids = np.column_stack([ic[:ns], ic[ns:]])
        adj: list[list[int]] = [[] for _ in range(npts)]
        pair_segs: dict[tuple[int, int], list[int]] = {}
        for s in range(ns):
            i, j = int(seg_ids[s, 0]), int(seg_ids[s, 1])
            adj[i].append(j)
            adj[j].append(i)
            pair_segs.setdefault((i, j) if i <= j else (j, i), []).append(s)

        visited_seg = np.zeros(ns, dtype=bool)
        loops = []
        for s in range(ns):
            if visited_seg[s]:
                continue
            start = int(seg_ids[s, 0])
            cur = int(seg_ids[s, 1])
            visited_seg[s] = True
            order = [start, cur]
            while cur != start:
                found = False
                for nb in adj[cur]:
                    for s2 in pair_segs[(cur, nb) if cur <= nb else (nb, cur)]:
                        if visited_seg[s2]:
                            continue
                        visited_seg[s2] = True
                        cur = nb
                        order.append(cur)
                        found = True
                        break
                    if found:
                        break
                if not found:
                    break
            loops.append(cls.loop(coord[order, :]))
        if not loops:
            return None
        return max(loops, key=len)

    @classmethod
    def blend(cls, a: LineMesh, b: LineMesh,
              fractions: FloatArray | Sequence[float]) -> list[LineMesh]:
        """Linearly morph between two conformal profiles ``a`` and ``b`` (equal point
        count and identical ``lines`` connectivity -- which is exactly what makes
        both open or both closed), one profile per fraction
        ``t`` with points ``(1-t)*a + t*b`` -- so ``t=0`` reproduces ``a`` and ``t=1``
        reproduces ``b``.  Each result carries ``a``'s connectivity, ``boundaries``
        and ``boundary_tags`` (positional BC markers follow the morph); per-element
        ``element_tags`` are left for the consuming factory/``loft`` to assign, so a
        blended stack feeds straight into ``loft`` or a section factory.  This is the
        profile-positioning step behind ``annulus`` (and any morphing sweep).

        High-order profiles morph too: ``a``/``b`` share the same order ``N`` (so
        their private per-line ``interior`` nodes pair by index) and each result
        carries the blended interior ``(1-t)*a.interior + t*b.interior`` -- the same
        lerp the corners get from the blended ``points``, so the result stays
        corner-consistent by construction and a high-order blended stack feeds
        ``loft`` unchanged.  At order 1 ``interior`` is empty and the result is
        byte-identical to the plain linear morph."""
        A: PointArray = np.asarray(a.points, dtype=float).reshape(-1, 3)
        B: PointArray = np.asarray(b.points, dtype=float).reshape(-1, 3)
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                "blend: profiles must have equal point counts (got %d, %d); build "
                "one from the other's points so they pair by index"
                % (A.shape[0], B.shape[0]))
        if not np.array_equal(a.lines, b.lines):
            raise ValueError(
                "blend: profiles must share identical connectivity (paired by index)")
        if a.order != b.order:
            raise ValueError("blend: profiles must have the same order (got %d, %d)"
                             % (a.order, b.order))
        out: list[LineMesh] = []
        for t in np.asarray(fractions, dtype=float).ravel():
            # the private interiors take the same lerp as the corners (which ride in
            # the blended points); at order 1 both interiors are empty, so this is a
            # no-op and the result equals the plain point blend.
            ia: PointArray = (1.0 - t) * a.interior + t * b.interior
            out.append(cls((1.0 - t) * A + t * B, a.lines, boundaries=a.boundaries,
                           boundary_tags=a.boundary_tags,
                           order=a.order, interior=ia))
        return out

    @classmethod
    def merge(cls, meshes: list[LineMesh], *,
              tol: float | None = None) -> LineMesh:
        """Merge line meshes into one, welding coincident **topological end
        points** (the degree-1 chain ends -- the 1-D analogue of the boundary
        vertices ``QuadMesh.merge``/``HexMesh.merge`` weld).  ``tol`` is the
        absolute coincidence distance (default ``1e-7`` x the extent).  Dense
        ``element_tags`` and tagged ``boundaries`` concatenate with each block's
        line ids offset; interior points are never welded.  Closedness is not
        tracked anywhere -- it simply falls out of the welded connectivity: if no
        degree-1 end survives the result *is* a loop, so two shared-endpoint
        ``A1 -> A2`` arcs (reverse one so the traversal doesn't cross) weld at
        ``A1`` and ``A2`` into a single cycle, the clean way to close a ring from
        two half-arcs."""
        meshes = list(meshes)
        pos = [np.asarray(m.points, dtype=float).reshape(-1, 3) for m in meshes]
        counts = [p.shape[0] for p in pos]
        P = np.concatenate(pos, axis=0) if pos else np.zeros((0, 3))
        total = P.shape[0]

        remap = np.arange(total, dtype=np.int64)
        is_bnd: BoolArray = np.zeros(total, dtype=bool)
        noff = 0
        for m, c in zip(meshes, counts):
            is_bnd[noff + m.boundary_points()] = True
            noff += c
        bidx = np.flatnonzero(is_bnd)
        if bidx.size:
            scl = float(np.max(P.max(axis=0) - P.min(axis=0))) if total else 0.0
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

        line_list: list[IntArray] = []
        bnd_list: list[IntArray] = []
        name_list: list[StrArray] = []
        etag_list: list[StrArray] = []
        noff = loff = 0
        for m, c in zip(meshes, counts):
            line_list.append(point_id[m.lines + noff])   # local -> welded id
            etag_list.append(m.element_tags)
            if m.boundaries.shape[0]:
                b: IntArray = m.boundaries.copy()
                b[:, 0] += loff                          # shift line ids; sides local
                bnd_list.append(b)
                name_list.append(m.boundary_tags)
            noff += c
            loff += m.n_lines
        lines = (np.concatenate(line_list, axis=0) if line_list
                 else np.zeros((0, 2), np.int64))
        etags = (np.concatenate(etag_list) if etag_list
                 else np.empty(0, dtype=np.str_))
        bnd = (np.concatenate(bnd_list, axis=0) if bnd_list
               else np.zeros((0, 2), np.int64))
        names = (np.concatenate(name_list) if name_list
                 else np.empty(0, dtype=np.str_))

        # order-N: welding only touches endpoints (corners, which are re-numbered into
        # the merged points), and every high-order node of a line is *private*, so the
        # merged interior is just the blocks concatenated in the same order the lines
        # were -- nothing to reconcile, nothing to re-pin.
        order = meshes[0].order if meshes else 1
        if any(m.order != order for m in meshes):
            raise ValueError("LineMesh.merge: all meshes must share the same order")
        interior: PointArray | None = None
        if meshes:
            interior = np.concatenate([m.interior for m in meshes], axis=0)

        return cls(points, lines, etags, bnd, names, order=order,
                   interior=interior)

    # -- sizes / topology -----------------------------------------------
    def __len__(self) -> int:
        """Number of points."""
        return self.points.shape[0]

    def __array__(self, dtype: np.dtype[Any] | None = None) -> NDArray[Any]:
        """Expose the ``(N,3)`` point array to ``numpy.asarray``."""
        return self.points if dtype is None else self.points.astype(dtype)

    @property
    def n_points(self) -> int:
        """Number of points."""
        return self.points.shape[0]

    @property
    def n_lines(self) -> int:
        """Number of line elements."""
        return self.lines.shape[0]

    @property
    def n_boundaries(self) -> int:
        """Number of tagged boundary points."""
        return self.boundaries.shape[0]

    @property
    def element_group_tags(self) -> list[str]:
        """Sorted unique non-empty per-line element tags present on the mesh."""
        return sorted({t for t in self.element_tags.tolist() if t})

    @property
    def boundary_group_tags(self) -> list[str]:
        """Sorted unique tags of the tagged boundary points present on the mesh."""
        return sorted(set(self.boundary_tags.tolist()))

    def boundary_points(self) -> IntArray:
        """Sorted unique topological boundary point ids: the degree-1 ends. Empty
        for a closed loop; the two ends for an open chain."""
        if self.lines.size == 0:
            return np.zeros(0, dtype=np.int64)
        pids, counts = np.unique(self.lines.ravel(), return_counts=True)
        return pids[counts == 1]

    def boundary_elements(self) -> IntArray:
        """Sorted unique line ids with at least one degree-1 end point."""
        ends = set(self.boundary_points().tolist())
        if not ends:
            return np.zeros(0, dtype=np.int64)
        rows = [e for e in range(self.lines.shape[0])
                if int(self.lines[e, 0]) in ends or int(self.lines[e, 1]) in ends]
        return np.asarray(sorted(rows), dtype=np.int64)

    @staticmethod
    def _order_bnd(
        bnd: Sequence[Sequence[int]] | IntArray,
        tags: Sequence[str] | StrArray,
    ) -> tuple[IntArray, StrArray]:
        """Stably order boundary rows by ``(line id, side)``, applying the same
        permutation to the parallel ``tags`` array."""
        b: IntArray = np.asarray(bnd, dtype=np.int64).reshape(-1, 2)
        nm: StrArray = np.asarray(tags, dtype=np.str_).reshape(-1)
        if b.shape[0]:
            order = np.lexsort((b[:, 1], b[:, 0]))
            b = b[order]
            nm = nm[order]
        return b, nm

    # -- per-segment tags for the ordered ops ---------------------------
    def _seg_tags(self) -> list[str] | None:
        """The dense ``element_tags`` as a ``list[str]`` for the ordered ops, or
        ``None`` if every element is untagged (so an untagged mesh stays untagged)."""
        if self.element_tags.size and np.any(self.element_tags != ""):
            return [str(x) for x in self.element_tags.tolist()]
        return None

    # -- arc length ------------------------------------------------------
    @property
    def length(self) -> float:
        """Total (open) arc length through the points in index order."""
        P = self.points
        return float(np.sum(np.sqrt(np.sum(np.diff(P, axis=0) ** 2, axis=1))))

