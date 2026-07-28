"""1-D mesh container -- the line sibling of :class:`~nekmeshpy.quadmesh.QuadMesh`
and :class:`~nekmeshpy.hexmesh.HexMesh`.

Following the same 2 / 4 / 8-vertices-per-element ladder, a :class:`LineMesh` is a
list of **line elements** ``lines`` ``(L,2)`` over a shared ``(N,3)`` point array,
so it can branch (a T-junction, a star) rather than being a single ordered path.
It carries the two tag systems used throughout the toolkit, one dimension down:

* ``element_tags`` ``(L,)`` -- a dense per-line region/material tag (``""`` =
  untagged).  On extrude a line sweeps into a column of quads, which inherit its
  tag; on the next level those quads sweep into hexes.  This is the renamed,
  generalized successor of the old per-segment ``segment_names``.
* ``boundaries`` ``(Nbc,2)`` = ``[line id (0-based), side (1-2)]`` with a parallel
  ``boundary_tags`` ``(Nbc,)``.  The **boundary of a line is a point** (side ``s``
  = local vertex ``s-1`` = ``lines[e, s-1]``); on
  :meth:`~nekmeshpy.quadmesh.QuadMesh.extrude` a tagged boundary point
  sweeps into a quad boundary **edge**, and thence (via
  :meth:`~nekmeshpy.hexmesh.HexMesh.extrude`) into a hex boundary **face**
  -- the same one-way tag chain as ``QuadMesh.boundary_tags``, one level down.

**Open vs closed is a topological property, not a subclass.** The old gmsh
``Curve`` (open) / ``CurveLoop`` (closed) distinction becomes a single ``_closed``
flag exposed as :attr:`is_closed` / :attr:`is_open`; a loop simply rejoins its last
point to its first.  The factory :meth:`open` / :meth:`loop` / :meth:`circle` /
:meth:`from_segments` build the common cases, and the ordered ops (:meth:`resample`,
:meth:`resample_spline`, :meth:`align_to`, :meth:`radial_match`,
:meth:`split_by_fraction`) operate on the points in index order (as a path or a
loop) exactly as before.

All numerics are ported verbatim from the former ``Curve`` / ``CurveLoop`` helpers
(``resample_path``, ``resample_loop``, ``chain_segments``, and the O-grid
``spine_at`` / ``_resample_spline`` / ``_split_ring_by_fraction`` /
``_align_ring_to``), so coordinate results are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from .._typing import FloatArray, IntArray, Point, PointArray, StrArray, Vec3
from ._plane import _in_plane_axes, from_plane, plane_frame, to_plane


def _radial_project(loop: FloatArray, centroid: FloatArray,
                    dirs: FloatArray) -> FloatArray:
    """Point on the closed 2-D ``loop`` in the radial direction of each ``dirs``
    row (from ``centroid``), by angular interpolation -- keeps a matched ring
    radially aligned so its quads do not tangle (star-shaped loops).  Returns
    ``(len(dirs), 2)``.  Shared by :meth:`LineMesh.radial_match` and the O-grid
    ring projection."""
    ang = np.arctan2(loop[:, 1] - centroid[1], loop[:, 0] - centroid[0])
    order = np.argsort(ang)
    a = np.concatenate([ang[order], ang[order][:1] + 2 * np.pi])
    x = np.concatenate([loop[order, 0], loop[order][:1, 0]])
    y = np.concatenate([loop[order, 1], loop[order][:1, 1]])
    t = np.arctan2(dirs[:, 1], dirs[:, 0])
    t = np.where(t < a[0], t + 2 * np.pi, t)
    return np.column_stack([np.interp(t, a, x), np.interp(t, a, y)])


def _lerp_along(P: PointArray, arclen: FloatArray, targets: FloatArray) -> PointArray:
    """Piecewise-linear resample of polyline ``P`` at each query arc length in
    ``targets`` (``arclen`` = ``P``'s cumulative arc length); returns
    ``(len(targets), 3)``.  Shared core of the four arc-length interpolations
    below (ported verbatim from ``resample_path`` / ``resample_loop`` /
    ``spine_at`` / ``_interp_al``); callers pass already-clamped ``targets``."""
    K = P.shape[0]
    out = np.zeros((targets.shape[0], 3))
    for k in range(targets.shape[0]):
        s = targets[k]
        idx = int(np.flatnonzero(arclen <= s)[-1])
        idx = min(idx, K - 2)
        span = arclen[idx + 1] - arclen[idx]
        t = 0.0
        if span > 0:
            t = (s - arclen[idx]) / span
        out[k, :] = P[idx, :] + t * (P[idx + 1, :] - P[idx, :])
    return out


def _names_at(src_names: list[str], bounds: FloatArray,
              queries: FloatArray) -> list[str]:
    """Per-segment tags for a resampled line: ``src_names[m]`` labels the source
    segment spanning ``[bounds[m], bounds[m + 1])`` (arc length or CCW angle); each
    query position (a new segment's midpoint) inherits the tag of the source
    segment containing it.  ``queries`` must already lie within ``bounds``."""
    idx = np.searchsorted(bounds, queries, side="right") - 1
    idx = np.clip(idx, 0, len(src_names) - 1)
    return [src_names[int(m)] for m in idx]


def _names_by_sector(src_names: list[str], src_pts2d: FloatArray,
                     out_pts2d: FloatArray, centroid: FloatArray) -> list[str]:
    """Per-segment tags for a radially-matched loop: source segment ``m`` (point
    ``m`` -> ``(m+1) % Ns``) covers the CCW azimuthal arc between the two points'
    angles about ``centroid``; each output segment inherits the source tag of the
    arc containing its midpoint direction.  Assumes a simple star-shaped source loop
    wound CCW (the :meth:`LineMesh.radial_match` precondition)."""
    two_pi = 2.0 * np.pi
    sa = np.arctan2(src_pts2d[:, 1] - centroid[1],
                    src_pts2d[:, 0] - centroid[0]) % two_pi        # (Ns,)
    ns = sa.shape[0]
    k0 = int(np.argmin(sa))
    order = (k0 + np.arange(ns)) % ns                              # CCW from min angle
    bounds = np.concatenate([sa[order], sa[order][:1] + two_pi])   # (Ns+1,) increasing
    mid = (out_pts2d + np.roll(out_pts2d, -1, axis=0)) / 2.0       # segment midpoints
    ma = np.arctan2(mid[:, 1] - centroid[1], mid[:, 0] - centroid[0])
    ma = bounds[0] + (ma - bounds[0]) % two_pi                     # into [b0, b0+2pi)
    j = np.clip(np.searchsorted(bounds, ma, side="right") - 1, 0, ns - 1)
    return [src_names[int(order[int(jj)])] for jj in j]


class LineMesh:
    """A 1-D mesh: an ``(N,3)`` point array with ``(L,2)`` line connectivity, a
    dense per-line ``element_tags``, and ``[line id, side 1-2]`` ``boundaries`` with
    parallel ``boundary_tags``.  The line sibling of
    :class:`~nekmeshpy.quadmesh.QuadMesh`.  Open vs closed is the
    :attr:`is_closed` topological property (see the module docstring); build with
    :meth:`open` / :meth:`loop` / :meth:`circle` / :meth:`from_segments`."""

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
        """Construct from arrays: ``points`` ``(N,3)`` (input **must be 3-D**),
        optional ``lines`` ``(L,2)`` connectivity (defaults to a consecutive chain
        ``[[0,1],[1,2],...]``, plus ``[N-1,0]`` when ``closed``), an optional dense
        ``element_tags`` ``(L,)`` (``""`` = untagged; length must equal ``len(lines)``),
        and an optional tagged-boundary list ``boundaries`` ``(Nbc,2)`` =
        ``[line id, side 1-2]`` with a parallel ``boundary_tags`` ``(Nbc,)``.  Use
        the :meth:`open` / :meth:`loop` factories rather than ``closed=`` directly."""
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

        # tagged boundary points [line id, side 1-2] parallel with boundary_tags,
        # carried to the swept quad edges by QuadMesh.extrude / loft.
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
        """An **open** line mesh through ``points`` in order (gmsh ``Curve``): a
        consecutive chain with ``N-1`` line elements."""
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
        """A **closed** loop through ``points`` (last rejoins first; gmsh
        ``Curve Loop``): ``N`` line elements.  A per-element ``element_tags`` -- one
        entry per point, element ``m`` = point ``m`` -> ``(m+1) % N`` -- is carried
        through :meth:`resample` / :meth:`radial_match` / :meth:`align_to` and
        consumed by the section factories (e.g.
        :meth:`~nekmeshpy.quadmesh.QuadMesh.annulus`), which copy it onto
        the section boundary edges.  This is how a far-field box is split into named
        sides -- e.g. ``LineMesh.loop([...4 corners...], element_tags=["bottom",
        "outlet", "top", "inlet"])`` -- without post-hoc geometry detection (see
        ``examples/flow_past_cylinder.py``)."""
        return cls(points, None, element_tags, boundaries, boundary_tags,
                   closed=True)

    @classmethod
    def circle(cls, radius: float, n: int, *,
               center: Point = (0.0, 0.0, 0.0),
               normal: Vec3 = (0.0, 0.0, 1.0),
               element_tags: StrArray | Sequence[str] | None = None) -> LineMesh:
        """A **closed** loop of ``n`` points evenly spaced on a circle of ``radius``
        about the 3-D ``center`` in the plane with the given ``normal`` (endpoint
        not repeated).  The defaults -- ``center`` at the origin, ``normal = +z`` --
        give the unit-circle-style loop in the ``xy`` plane; pass ``normal`` to place
        the circle in any plane (e.g. ``normal=(1, 0, 0)`` for a ``yz`` circle).
        ``element_tags`` (one entry per point, ``n`` total) tags the loop's line
        elements at construction -- the lowest-level place to name a wall -- so the
        section factories (:meth:`~nekmeshpy.quadmesh.QuadMesh.ogrid` /
        :meth:`~nekmeshpy.quadmesh.QuadMesh.annulus`) pick it up without a scalar
        ``wall_tag`` / ``inner_tag`` argument (see :meth:`loop`)."""
        th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        c: Point = np.asarray(center, dtype=float).ravel()
        e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
        local = radius * np.cos(th)[:, None] * e1 + radius * np.sin(th)[:, None] * e2
        return cls.loop(c + local, element_tags)

    @classmethod
    def from_segments(cls, segs: FloatArray | None) -> LineMesh | None:
        """Chain unordered 3D segments (from marching triangles) into a single
        closed ordered loop -- the largest connected component -- or ``None`` if
        ``segs`` is empty / forms no loop.  A segment set may contain several
        disjoint loops; only the largest (most points) is returned, since that is
        the cross-section ring of interest.  (Port of ``chain_segments``.)"""
        if segs is None or len(segs) == 0:
            return None
        segs = np.asarray(segs, dtype=float)
        ns = segs.shape[0]
        pts_raw = np.vstack([segs[:, 0:3], segs[:, 3:6]])

        # weld coincident endpoints on a scale-relative grid (was an absolute
        # ``* 1e6``; ``/(1e-6 * extent)`` keeps the dedup robust to the
        # geometry's units, matching QuadMesh.merge's scale-relative tolerance)
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

        # node -> incident neighbours (segment-index order) and each unordered
        # node pair -> its segment indices (ascending), so the walk finds the
        # next segment by incidence instead of rescanning all ns segments.
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
        """Number of points (so ``len(line)`` is the point count)."""
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
        """Sorted unique tags of the tagged boundary points present on the mesh
        (a Nek BC code / id is assigned only at export)."""
        return sorted(set(self.boundary_tags.tolist()))

    def boundary_points(self) -> IntArray:
        """Sorted unique **topological** boundary point ids: the degree-1 ends (a
        point touched by a single line).  For a closed loop this is empty; for an
        open chain it is the two ends.  Distinct from the *tagged* ``boundaries``."""
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
        """Stably order boundary rows by ``(line id, side)`` so a mesh is
        independent of insertion order, applying the same permutation to the
        parallel ``tags`` array (mirrors :meth:`QuadMesh._order_bnd`)."""
        b: IntArray = np.asarray(bnd, dtype=np.int64).reshape(-1, 2)
        nm: StrArray = np.asarray(tags, dtype=np.str_).reshape(-1)
        if b.shape[0]:
            order = np.lexsort((b[:, 1], b[:, 0]))
            b = b[order]
            nm = nm[order]
        return b, nm

    # -- per-segment tags for the ordered ops ---------------------------
    def _seg_tags(self) -> list[str] | None:
        """The dense ``element_tags`` as a ``list[str]`` for the ordered ops'
        carry-through helpers, or ``None`` if every element is untagged (so an
        untagged mesh stays untagged, mirroring the old ``names is None`` path).
        The ordered ops treat the points in index order as a path/loop, so the
        per-element tag is the per-segment tag."""
        if self.element_tags.size and np.any(self.element_tags != ""):
            return [str(x) for x in self.element_tags.tolist()]
        return None

    # -- arc length ------------------------------------------------------
    def _seg_arclen(self) -> FloatArray:
        P = self.points
        return np.concatenate(
            [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(P, axis=0) ** 2, axis=1)))])

    @property
    def length(self) -> float:
        """Total (open) arc length through the points in index order."""
        P = self.points
        return float(np.sum(np.sqrt(np.sum(np.diff(P, axis=0) ** 2, axis=1))))

    # -- ordered ops (index-order path / loop) --------------------------
    def resample(self, fractions: float | FloatArray) -> LineMesh:
        """Resample at the given normalized arc-length ``fractions`` in ``[0, 1]``,
        returning a new :class:`LineMesh` of the same open/closed kind.  For an
        **open** mesh ``0`` = start, ``1`` = end; pass ``uniform_spacing(n)`` for
        ``n+1`` arc-length-even points, ``geometric_spacing(n, r)`` to grade toward
        one end, or ``numpy.linspace(a, b, k)`` for an explicit set (a scalar
        returns a single point).  For a **closed** loop ``1`` wraps back to the
        start, so keep fractions in ``[0, 1)`` (e.g.
        ``numpy.linspace(0, 1, m, endpoint=False)`` for ``m`` even points).  (Ports
        ``resample_path`` / ``spine_at`` / ``resample_loop``.)"""
        seg_tags = self._seg_tags()
        if self._closed:
            pts = self.points
            closed = np.vstack([pts, pts[0, :]])
            seglen = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
            arclen = np.concatenate([[0.0], np.cumsum(seglen)])
            fr = np.atleast_1d(np.asarray(fractions, dtype=float))
            clipped = np.clip(fr, 0.0, 1.0)
            targets = clipped * arclen[-1]
            new_tags = None
            if seg_tags is not None and clipped.size >= 1:
                # each new (closed) segment k joins new point k -> (k+1)%M; its
                # midpoint arc-length fraction (wrapping the last back to the
                # first) picks the source segment tag.
                nxt = np.roll(clipped, -1)
                nxt[-1] += 1.0
                mid = (((clipped + nxt) / 2.0) % 1.0) * arclen[-1]
                new_tags = _names_at(seg_tags, arclen, mid)
            return LineMesh.loop(_lerp_along(closed, arclen, targets), new_tags)
        arclen = self._seg_arclen()
        fr = np.atleast_1d(np.asarray(fractions, dtype=float))
        targets = np.clip(fr, 0.0, 1.0) * arclen[-1]
        new_tags = None
        if seg_tags is not None and targets.size >= 2:
            # each new (open) segment inherits the source tag at its midpoint
            mid = (targets[:-1] + targets[1:]) / 2.0
            new_tags = _names_at(seg_tags, arclen, mid)
        return LineMesh.open(_lerp_along(self.points, arclen, targets), new_tags)

    def resample_spline(self, n: int) -> LineMesh:
        """Interpolating cubic spline through the points, resampled to ``n``
        arc-length-even points; endpoints pinned.  Returns an **open**
        :class:`LineMesh`.  (Port of the O-grid ``_resample_spline``.)"""
        P = self.points
        m = P.shape[0]
        t = self._seg_arclen()
        if m < 2 or t[-1] == 0:
            return LineMesh.open(np.tile(P[0, :], (n, 1)))
        td = np.linspace(0.0, t[-1], max(10 * m, 50))
        cs = CubicSpline(t, P, axis=0)              # not-a-knot (MATLAB 'spline')
        Xd = cs(td)
        ad = np.concatenate(
            [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(Xd, axis=0) ** 2, axis=1)))])
        aq = np.linspace(0.0, ad[-1], n)
        out = np.column_stack([np.interp(aq, ad, Xd[:, c]) for c in range(3)])
        out[0, :] = P[0, :]
        out[-1, :] = P[-1, :]
        return LineMesh.open(out)

    def align_to(self, other: LineMesh) -> LineMesh:
        """Cyclically shift (and possibly flip) this **closed** loop to best match
        ``other`` in least squares; returns a new closed :class:`LineMesh`.
        (Port of ``_align_ring_to``.)"""
        B = other.points
        A = self.points
        M = A.shape[0]
        best = np.inf
        bestA = A
        best_f, best_s = 0, 0
        for f in (0, 1):
            Af = A[::-1, :].copy() if f else A
            for s in range(M):
                As = np.roll(Af, s, axis=0)
                d = np.sum((As - B) ** 2)
                if d < best:
                    best = d
                    bestA = As
                    best_f, best_s = f, s
        seg_tags = self._seg_tags()
        new_tags = None
        if seg_tags is not None:
            tags_arr = np.array(seg_tags, dtype=object)
            # transform the per-segment tags by the same flip+roll as the points:
            # a flip reverses the loop (segment m -> M-2-m), then roll by ``s``.
            if best_f:
                tags_arr = np.roll(tags_arr[::-1], -1)
            new_tags = [str(x) for x in np.roll(tags_arr, best_s)]
        return LineMesh.loop(bestA, new_tags)

    def radial_match(self, other: LineMesh) -> LineMesh:
        """Resample this **closed** loop to ``len(other)`` points, one per ``other``
        point at its azimuthal angle about the *shared centroid* (``other``'s
        centroid), by angular interpolation along this loop.  Returns a new closed
        :class:`LineMesh` (in ``other``'s plane) radially aligned to ``other`` --
        e.g. to align a coarse far-field box loop to a finer body loop before
        :meth:`~nekmeshpy.quadmesh.QuadMesh.annulus`, so the annulus rings
        do not skew::

            outer = LineMesh.loop([...box corners...]).radial_match(inner)
            QuadMesh.annulus(inner, outer, radial)   # equal point counts, aligned

        Both loops are projected into ``other``'s best-fit plane, matched there, and
        lifted back, so they need not lie in the ``xy`` plane (but should be coplanar
        for the match to be meaningful).

        Per-segment ``element_tags`` are carried by **angular sector**: each output
        segment inherits the tag of the source segment whose azimuthal span (about
        the shared centroid) contains the output segment's midpoint direction -- so a
        4-corner tagged box maps each output segment onto the box side it faces."""
        c, e1, e2, _ = plane_frame(other.points)
        A: FloatArray = to_plane(other.points, c, e1, e2)
        centroid = A.mean(axis=0)
        self2d: FloatArray = to_plane(self.points, c, e1, e2)
        matched = _radial_project(self2d, centroid, A - centroid)
        seg_tags = self._seg_tags()
        new_tags = None
        if seg_tags is not None:
            new_tags = _names_by_sector(seg_tags, self2d, matched, centroid)
        return LineMesh.loop(from_plane(matched, c, e1, e2), new_tags)

    def split_by_fraction(self, f: float, nh: int) -> LineMesh:
        """Resample this **closed** loop into ``2*nh`` points so index 0 is
        ``points[0]`` (A1 rail) and index ``nh`` sits at arc-length fraction ``f``
        (A2 rail).  Returns an **open** :class:`LineMesh`.  (Port of
        ``_split_ring_by_fraction``.)"""
        R = self.points
        Rc = np.vstack([R, R[0, :]])
        al = np.concatenate(
            [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(Rc, axis=0) ** 2, axis=1)))])
        tot = al[-1]
        p1 = self._interp_al(Rc, al, np.linspace(0.0, f * tot, nh + 1))
        p2 = self._interp_al(Rc, al, np.linspace(f * tot, tot, nh + 1))
        return LineMesh.open(np.vstack([p1[0:nh, :], p2[0:nh, :]]))

    @staticmethod
    def _interp_al(Rc: PointArray, al: FloatArray, tq: FloatArray) -> PointArray:
        """Linear interpolation of polyline ``Rc`` (cumulative arc length
        ``al``) at query positions ``tq``."""
        return _lerp_along(Rc, al, np.clip(tq, 0.0, al[-1]))
