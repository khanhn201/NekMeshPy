"""1-D mesh container: line elements ``(L,2)`` over a shared ``(N,3)`` point array.

The line sibling of QuadMesh/HexMesh; it can branch rather than being a single
ordered path. It carries a dense per-line ``element_tags`` and sparse tagged
``boundaries`` (a line's boundary is a point), both of which sweep up on extrude.
Open vs closed is a topological property (``is_closed`` / ``is_open``), not a
subclass; factories build the common cases (``open`` / ``loop`` / ``line`` /
``circle`` / ``rectangle`` / ``far_field_box`` / ``from_segments``) and every
curve is meshed exactly at the points given -- there is no resampling here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._typing import FloatArray, IntArray, Point, PointArray, StrArray, Vec3
from ._plane import _in_plane_axes


class LineMesh:
    """A 1-D mesh: an ``(N,3)`` point array with ``(L,2)`` line connectivity, a
    dense per-line ``element_tags``, and ``[line id, side 1-2]`` ``boundaries`` with
    parallel ``boundary_tags``. Build with ``open`` / ``loop`` / ``circle`` /
    ``from_segments``."""

    # local line "edges": row s-1 is side s -> the single local vertex it names.
    EDGE_POINTS = np.array([[0], [1]], dtype=np.int64)

    def __init__(
        self,
        points: NDArray[Any],
        lines: IntArray | None = None,
        element_tags: StrArray | Sequence[str] | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
        *,
        closed: bool = False,
    ) -> None:
        """Construct from arrays: ``points`` ``(N,3)`` (must be 3-D), optional
        ``lines`` ``(L,2)`` connectivity (defaults to a consecutive chain, wrapping
        when ``closed``), optional dense ``element_tags`` ``(L,)``, and an optional
        tagged-boundary list ``boundaries`` ``(Nbc,2)`` with parallel
        ``boundary_tags``. Prefer the ``open`` / ``loop`` factories."""
        a = np.asarray(points, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(
                f"boundary points must be (N,3) 3-D coordinates; got "
                f"{a.shape} -- add a z column (all boundaries live in 3-D)")
        self.points: PointArray = a
        self._closed = bool(closed)
        n = a.shape[0]

        if lines is None:
            if n < 2:
                self.lines: IntArray = np.zeros((0, 2), dtype=np.int64)
            else:
                idx = np.arange(n, dtype=np.int64)
                seg = np.column_stack([idx[:-1], idx[1:]])
                if self._closed:
                    seg = np.vstack([seg, [[n - 1, 0]]])
                self.lines = seg
        else:
            self.lines = np.asarray(lines, dtype=np.int64).reshape(-1, 2)

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

    # -- construction factories -----------------------------------------
    @classmethod
    def open(
        cls,
        points: NDArray[Any],
        element_tags: StrArray | Sequence[str] | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
    ) -> LineMesh:
        """An open line mesh through ``points`` in order: a consecutive chain."""
        return cls(points, None, element_tags, boundaries, boundary_tags,
                   closed=False)

    @classmethod
    def loop(
        cls,
        points: NDArray[Any],
        element_tags: StrArray | Sequence[str] | None = None,
        boundaries: IntArray | None = None,
        boundary_tags: StrArray | Sequence[str] | None = None,
    ) -> LineMesh:
        """A closed loop through ``points`` (last rejoins first): ``N`` line
        elements. A per-element ``element_tags`` (element ``m`` = point ``m`` ->
        ``(m+1) % N``) is carried through the ordered ops and consumed by the section
        factories, e.g. to split a far-field box into named sides."""
        return cls(points, None, element_tags, boundaries, boundary_tags,
                   closed=True)

    @classmethod
    def line(cls, start: Point, end: Point, fractions: float | FloatArray, *,
             element_tag: str = "") -> LineMesh:
        """A straight open line from ``start`` to ``end`` sampled at normalized
        arc-length ``fractions`` in ``[0, 1]`` (``0`` = start, ``1`` = end): the
        graded-edge sibling of ``circle``/``rectangle``. The points are placed
        exactly at ``start + f*(end - start)`` -- no resampling. ``element_tag``
        names every resulting line element (e.g. to tag a structured edge as one
        wall)."""
        frac = np.atleast_1d(np.asarray(fractions, dtype=float))
        s: Point = np.asarray(start, dtype=float).ravel()
        e: Point = np.asarray(end, dtype=float).ravel()
        pts = s + frac[:, None] * (e - s)
        tags = [element_tag] * (pts.shape[0] - 1) if element_tag else None
        return cls.open(pts, element_tags=tags)

    @classmethod
    def circle(cls, radius: float, n: int, *,
               center: Point = (0.0, 0.0, 0.0),
               normal: Vec3 = (0.0, 0.0, 1.0),
               start_theta: float = 0.0,
               element_tags: StrArray | Sequence[str] | None = None) -> LineMesh:
        """A closed loop of ``n`` points evenly spaced on a circle of ``radius``
        about ``center`` in the plane with the given ``normal`` (default ``+z``).
        Point ``k`` sits at angle ``2*pi*k/n + start_theta`` from the in-plane
        ``e1`` axis, so ``start_theta`` rotates the whole loop -- e.g. to align its
        index 0 with another loop or a far-field box before an index-paired
        :meth:`QuadMesh.annulus`. ``element_tags`` tags the loop's line elements at
        construction."""
        th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + float(start_theta)
        c: Point = np.asarray(center, dtype=float).ravel()
        e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
        local = radius * np.cos(th)[:, None] * e1 + radius * np.sin(th)[:, None] * e2
        return cls.loop(c + local, element_tags)

    @classmethod
    def rectangle(cls, width: float, height: float, *,
                  center: Point = (0.0, 0.0, 0.0),
                  normal: Vec3 = (0.0, 0.0, 1.0),
                  element_tags: StrArray | Sequence[str] | None = None) -> LineMesh:
        """A closed 4-corner loop of the given ``width`` x ``height`` about
        ``center`` in the plane with the given ``normal`` (default ``+z``). The four
        line elements run CCW from the lower-left corner (bottom / right / top /
        left); ``element_tags`` names them, e.g. to split a far-field box into sides."""
        c: Point = np.asarray(center, dtype=float).ravel()
        e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
        hw, hh = width / 2.0, height / 2.0
        corners = np.array([c - hw * e1 - hh * e2, c + hw * e1 - hh * e2,
                            c + hw * e1 + hh * e2, c - hw * e1 + hh * e2])
        return cls.loop(corners, element_tags)

    @classmethod
    def far_field_box(cls, inner: LineMesh, half_width: float,
                      half_height: float | None = None, *,
                      center: Point = (0.0, 0.0, 0.0),
                      normal: Vec3 = (0.0, 0.0, 1.0),
                      side_tags: Sequence[str] | None = None) -> LineMesh:
        """A rectangular far-field loop **index-aligned** to ``inner``: one outer
        point per ``inner`` point, placed where the ray from ``center`` through that
        point meets the box of half-extents ``half_width`` x ``half_height``
        (``half_height`` defaults to ``half_width``). The result carries the same
        point count as ``inner``, so it feeds :meth:`QuadMesh.annulus` directly
        without any matching/resampling step -- the box follows the body loop the
        way a cubed-``sphere`` follows a ``box`` one dimension up.

        ``side_tags`` (length-4 ``[bottom, right, top, left]``) names the four box
        sides: each line element takes ``right``/``left`` where its midpoint
        direction is dominated by the in-plane ``e1`` axis (sign of the ``e1``
        component) else ``top``/``bottom`` (sign of ``e2``). ``side_tags=None``
        leaves the loop untagged."""
        if not isinstance(inner, LineMesh):
            raise TypeError("far_field_box inner must be a LineMesh, got %s"
                            % type(inner).__name__)
        if not inner.is_closed:
            raise TypeError("far_field_box inner must be a closed loop")
        pin: PointArray = np.asarray(inner.points, dtype=float).reshape(-1, 3)
        hw = float(half_width)
        hh = hw if half_height is None else float(half_height)
        c: Point = np.asarray(center, dtype=float).ravel()
        e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
        d = pin - c                                             # (N,3) rays
        a = d @ e1                                              # in-plane components
        b = d @ e2
        na = np.abs(a) / hw
        nb = np.abs(b) / hh
        denom = np.maximum(na, nb)
        if np.any(denom <= 0.0):
            raise ValueError("far_field_box: an inner point coincides with center")
        s = 1.0 / denom                                        # ray-box hit scale
        outer = c + (s * a)[:, None] * e1 + (s * b)[:, None] * e2
        tags: list[str] | None = None
        if side_tags is not None:
            st = list(side_tags)
            if len(st) != 4:
                raise ValueError("far_field_box side_tags must be 4 names "
                                 "[bottom, right, top, left]")
            bottom, right, top, left = st
            # each line element (point m -> m+1) named by its midpoint direction
            am = 0.5 * (a + np.roll(a, -1))
            bm = 0.5 * (b + np.roll(b, -1))
            horiz = np.abs(am) / hw >= np.abs(bm) / hh
            tags = [(right if am[m] >= 0 else left) if horiz[m]
                    else (top if bm[m] >= 0 else bottom)
                    for m in range(pin.shape[0])]
        return cls.loop(outer, tags)

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

    # -- sizes / topology -----------------------------------------------
    def __len__(self) -> int:
        """Number of points."""
        return self.points.shape[0]

    def __array__(self, dtype: np.dtype[Any] | None = None) -> NDArray[Any]:
        """Expose the ``(N,3)`` point array to ``numpy.asarray``."""
        return self.points if dtype is None else self.points.astype(dtype)

    @property
    def is_closed(self) -> bool:
        """``True`` if this is a closed loop (last point rejoins the first)."""
        return self._closed

    @property
    def is_open(self) -> bool:
        """``True`` if this is an open line mesh (distinct end points)."""
        return not self._closed

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

