"""1-D geometry value objects, following gmsh's vocabulary.

gmsh's model is *points -> curves -> curve loops -> surfaces*: a **Curve** is an
open 1-D entity with two endpoints, and a **Curve Loop** is an ordered, closed
assembly used to bound surfaces.  Mirroring that:

* :class:`Curve` -- an **open** ordered sequence of 3-D points.
* :class:`CurveLoop` -- a **closed** loop (last point joins the first).

A ``CurveLoop`` is deliberately **not** a subclass of ``Curve`` (a curve loop is a
distinct concept in gmsh, not a curve), so a parameter typed ``Curve`` rejects a
``CurveLoop`` and vice versa.

A curve stores its coordinates as a single ``(N,3)`` NumPy array, exposed as
``curve.points`` -- consumers work with that array directly (a row is a plain
``(3,)`` array).  Curve-producing methods return a ``Curve`` / ``CurveLoop``;
``length`` returns a ``float``.

All numerics are ported verbatim from the original functional helpers
(``resample_path``, ``resample_loop``, ``chain_segments``, and the O-grid
``spine_at`` / ``_resample_spline`` / ``_split_ring_by_fraction`` /
``_align_ring_to``) so results are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from .._typing import FloatArray, Point, PointArray, Vec3
from ._plane import _in_plane_axes, from_plane, plane_frame, to_plane


def _radial_project(loop: FloatArray, centroid: FloatArray,
                    dirs: FloatArray) -> FloatArray:
    """Point on the closed 2-D ``loop`` in the radial direction of each ``dirs``
    row (from ``centroid``), by angular interpolation -- keeps a matched ring
    radially aligned so its quads do not tangle (star-shaped loops).  Returns
    ``(len(dirs), 2)``.  Shared by :meth:`CurveLoop.radial_match` and the O-grid
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


class _PointSeq:
    """A 1-D curve backed by an ``(N,3)`` coordinate array (``self.points``).
    Private base of :class:`Curve` and :class:`CurveLoop`; not used directly."""

    def __init__(self, points: NDArray[Any]) -> None:
        a = np.asarray(points, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(
                f"boundary points must be (N,3) 3-D coordinates; got "
                f"{a.shape} -- add a z column (all boundaries live in 3-D)")
        self.points: PointArray = a

    def __len__(self) -> int:
        return self.points.shape[0]

    def __array__(self, dtype: np.dtype[Any] | None = None) -> NDArray[Any]:
        return self.points if dtype is None else self.points.astype(dtype)

    # -- arc length ------------------------------------------------------
    def _seg_arclen(self) -> FloatArray:
        P = self.points
        return np.concatenate(
            [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(P, axis=0) ** 2, axis=1)))])

    @property
    def length(self) -> float:
        """Total (open) arc length."""
        P = self.points
        return float(np.sum(np.sqrt(np.sum(np.diff(P, axis=0) ** 2, axis=1))))

    def resample_spline(self, n: int) -> Curve:
        """Interpolating cubic spline through the points, resampled to ``n``
        arc-length-even points; endpoints pinned.  Returns a :class:`Curve`.
        (Port of the O-grid ``_resample_spline``.)"""
        P = self.points
        m = P.shape[0]
        t = self._seg_arclen()
        if m < 2 or t[-1] == 0:
            return Curve(np.tile(P[0, :], (n, 1)))
        td = np.linspace(0.0, t[-1], max(10 * m, 50))
        cs = CubicSpline(t, P, axis=0)              # not-a-knot (MATLAB 'spline')
        Xd = cs(td)
        ad = np.concatenate(
            [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(Xd, axis=0) ** 2, axis=1)))])
        aq = np.linspace(0.0, ad[-1], n)
        out = np.column_stack([np.interp(aq, ad, Xd[:, c]) for c in range(3)])
        out[0, :] = P[0, :]
        out[-1, :] = P[-1, :]
        return Curve(out)


class Curve(_PointSeq):
    """An **open** 1-D curve: an ordered sequence of 3-D points with two
    endpoints (gmsh ``Curve``)."""

    def resample(self, fractions: float | FloatArray) -> Curve:
        """Resample the curve at the given normalized arc-length ``fractions`` in
        ``[0, 1]`` (``0`` = start, ``1`` = end), returning a new :class:`Curve`.
        The caller supplies *where* to sample, so a graded array puts the points
        where you want them.  Pass the spacing helpers for the common cases:
        ``resample(uniform_spacing(n))`` for ``n+1`` arc-length-even points,
        ``resample(geometric_spacing(n, r))`` to grade toward one end, or
        ``numpy.linspace(a, b, k)`` for an explicit set.  A scalar returns a
        single-point curve.  (Ports ``resample_path`` / ``spine_at``.)"""
        arclen = self._seg_arclen()
        fr = np.atleast_1d(np.asarray(fractions, dtype=float))
        targets = np.clip(fr, 0.0, 1.0) * arclen[-1]
        return Curve(_lerp_along(self.points, arclen, targets))


class CurveLoop(_PointSeq):
    """A **closed** loop (last point joins the first), gmsh ``Curve Loop`` --
    the boundary concept used to bound a surface.  Not a :class:`Curve`."""

    @classmethod
    def circle(cls, radius: float, n: int, *,
               center: Point = (0.0, 0.0, 0.0),
               normal: Vec3 = (0.0, 0.0, 1.0)) -> CurveLoop:
        """A closed loop of ``n`` points evenly spaced on a circle of ``radius``
        about the 3-D ``center`` in the plane with the given ``normal`` (endpoint
        not repeated).  The defaults -- ``center`` at the origin, ``normal = +z``
        -- give the unit-circle-style loop in the ``xy`` plane; pass ``normal`` to
        place the circle in any plane (e.g. ``normal=(1, 0, 0)`` for a ``yz``
        circle)."""
        th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        c: Point = np.asarray(center, dtype=float).ravel()
        e1, e2 = _in_plane_axes(np.asarray(normal, dtype=float))
        local = radius * np.cos(th)[:, None] * e1 + radius * np.sin(th)[:, None] * e2
        return cls(c + local)

    def resample(self, fractions: float | FloatArray) -> CurveLoop:
        """Resample the closed loop at the given normalized arc-length
        ``fractions`` (``0`` = the loop start; ``1`` wraps back to it, so keep
        fractions in ``[0, 1)`` to avoid duplicating the start), returning a new
        :class:`CurveLoop`.  ``resample(numpy.linspace(0, 1, m, endpoint=False))``
        gives ``m`` arc-length-even points around the loop.  (Port of
        ``resample_loop``.)"""
        pts = self.points
        closed = np.vstack([pts, pts[0, :]])
        seglen = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
        arclen = np.concatenate([[0.0], np.cumsum(seglen)])
        fr = np.atleast_1d(np.asarray(fractions, dtype=float))
        targets = np.clip(fr, 0.0, 1.0) * arclen[-1]
        return CurveLoop(_lerp_along(closed, arclen, targets))

    def align_to(self, other: CurveLoop) -> CurveLoop:
        """Cyclically shift (and possibly flip) this loop to best match
        ``other`` in least squares; returns a new :class:`CurveLoop`.
        (Port of ``_align_ring_to``.)"""
        B = other.points
        A = self.points
        M = A.shape[0]
        best = np.inf
        bestA = A
        for f in (0, 1):
            Af = A[::-1, :].copy() if f else A
            for s in range(M):
                As = np.roll(Af, s, axis=0)
                d = np.sum((As - B) ** 2)
                if d < best:
                    best = d
                    bestA = As
        return CurveLoop(bestA)

    def radial_match(self, other: CurveLoop) -> CurveLoop:
        """Resample this loop to ``len(other)`` points, one per ``other`` point at
        its azimuthal angle about the *shared centroid* (``other``'s centroid), by
        angular interpolation along this loop.  Returns a new :class:`CurveLoop`
        (in ``other``'s plane) radially aligned to ``other`` -- e.g. to align a
        coarse far-field box loop to a finer body loop before
        :meth:`~nekmeshpy.geometry.quadmesh.QuadMesh.annulus`, so the annulus rings
        do not skew::

            outer = CurveLoop([...box corners...]).radial_match(inner)
            QuadMesh.annulus(inner, outer, radial)   # equal point counts, aligned

        Both loops are projected into ``other``'s best-fit plane, matched there,
        and lifted back, so they need not lie in the ``xy`` plane (but should be
        coplanar for the match to be meaningful)."""
        c, e1, e2, _ = plane_frame(other.points)
        A: FloatArray = to_plane(other.points, c, e1, e2)
        centroid = A.mean(axis=0)
        self2d: FloatArray = to_plane(self.points, c, e1, e2)
        matched = _radial_project(self2d, centroid, A - centroid)
        return CurveLoop(from_plane(matched, c, e1, e2))

    def split_by_fraction(self, f: float, nh: int) -> Curve:
        """Resample the loop into ``2*nh`` points so index 0 is ``points[0]``
        (A1 rail) and index ``nh`` sits at arc-length fraction ``f`` (A2 rail).
        Returns a :class:`Curve`.  (Port of ``_split_ring_by_fraction``.)"""
        R = self.points
        Rc = np.vstack([R, R[0, :]])
        al = np.concatenate(
            [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(Rc, axis=0) ** 2, axis=1)))])
        tot = al[-1]
        p1 = self._interp_al(Rc, al, np.linspace(0.0, f * tot, nh + 1))
        p2 = self._interp_al(Rc, al, np.linspace(f * tot, tot, nh + 1))
        return Curve(np.vstack([p1[0:nh, :], p2[0:nh, :]]))

    @staticmethod
    def _interp_al(Rc: PointArray, al: FloatArray, tq: FloatArray) -> PointArray:
        """Linear interpolation of polyline ``Rc`` (cumulative arc length
        ``al``) at query positions ``tq``."""
        return _lerp_along(Rc, al, np.clip(tq, 0.0, al[-1]))

    @classmethod
    def chain(cls, segs: FloatArray | None) -> CurveLoop | None:
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
            loops.append(cls(coord[order, :]))
        if not loops:
            return None
        return max(loops, key=len)
