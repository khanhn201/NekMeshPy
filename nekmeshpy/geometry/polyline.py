"""Polyline value objects: open arcs and closed rings.

These wrap an ``(k,3)`` point array and carry the arc-length resampling /
alignment operations the mesher needs.  All numerics are ported verbatim from
the original functional helpers (``resample_path``, ``resample_loop``,
``chain_segments``, and the O-grid ``spine_at`` / ``_resample_spline`` /
``_split_ring_by_fraction`` / ``_align_ring_to``) so results are unchanged.
"""

import numpy as np
from scipy.interpolate import CubicSpline


class Polyline:
    """An ordered sequence of 3D points."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

    def __len__(self):
        return self.points.shape[0]

    def __array__(self, dtype=None):
        return self.points if dtype is None else self.points.astype(dtype)

    # -- arc length ------------------------------------------------------
    def _seg_arclen(self):
        return np.concatenate(
            [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(self.points, axis=0) ** 2, axis=1)))])

    @property
    def length(self):
        """Total (open) arc length."""
        return float(np.sum(np.sqrt(np.sum(np.diff(self.points, axis=0) ** 2, axis=1))))

    def point_at_fraction(self, s):
        """Point(s) on the (open) polyline at arc-length fraction(s) ``s`` in
        [0,1].  Returns an ``(m,3)`` array.  (Port of ``spine_at``.)"""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        al = self._seg_arclen()
        tot = al[-1]
        P = self.points
        n = P.shape[0]
        q = np.zeros((s.size, 3))
        for m in range(s.size):
            ss = min(max(s[m], 0.0), 1.0) * tot
            idx = int(np.flatnonzero(al <= ss)[-1])
            idx = min(idx, n - 2)
            span = al[idx + 1] - al[idx]
            t = 0.0
            if span > 0:
                t = (ss - al[idx]) / span
            q[m, :] = P[idx, :] + t * (P[idx + 1, :] - P[idx, :])
        return q

    def resample_spline(self, n):
        """Interpolating cubic spline through the points, resampled to ``n``
        arc-length-even points; endpoints pinned.  Returns an ``(n,3)`` array.
        (Port of the O-grid ``_resample_spline``.)"""
        P = self.points
        m = P.shape[0]
        t = self._seg_arclen()
        if m < 2 or t[-1] == 0:
            return np.tile(P[0, :], (n, 1))
        td = np.linspace(0.0, t[-1], max(10 * m, 50))
        cs = CubicSpline(t, P, axis=0)              # not-a-knot (MATLAB 'spline')
        Xd = cs(td)
        ad = np.concatenate(
            [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(Xd, axis=0) ** 2, axis=1)))])
        aq = np.linspace(0.0, ad[-1], n)
        out = np.column_stack([np.interp(aq, ad, Xd[:, c]) for c in range(3)])
        out[0, :] = P[0, :]
        out[-1, :] = P[-1, :]
        return out


class Arc(Polyline):
    """An open polyline (endpoints not joined)."""

    def resample(self, n):
        """Resample to ``n`` arc-length-even points, keeping the endpoints.
        Returns an ``(n,3)`` array.  (Port of ``resample_path``.)"""
        P = self.points
        seg = np.sqrt(np.sum(np.diff(P, axis=0) ** 2, axis=1))
        arclen = np.concatenate([[0.0], np.cumsum(seg)])
        total = arclen[-1]
        targ = np.linspace(0.0, total, n)
        out = np.zeros((n, 3))
        K = P.shape[0]
        for k in range(n):
            s = targ[k]
            idx = int(np.flatnonzero(arclen <= s)[-1])
            idx = min(idx, K - 2)
            span = arclen[idx + 1] - arclen[idx]
            t = 0.0
            if span > 0:
                t = (s - arclen[idx]) / span
            out[k, :] = P[idx, :] + t * (P[idx + 1, :] - P[idx, :])
        return out


class Ring(Polyline):
    """A closed polyline (loops back to its first point)."""

    def resample(self, m):
        """Resample the loop to ``m`` arc-length-even points; returns a new
        :class:`Ring`.  (Port of ``resample_loop``.)"""
        pts = self.points
        closed = np.vstack([pts, pts[0, :]])
        seglen = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
        arclen = np.concatenate([[0.0], np.cumsum(seglen)])
        total = arclen[-1]
        targets = np.linspace(0.0, total, m + 1)[:-1]
        out = np.zeros((m, 3))
        n = pts.shape[0]
        for k in range(m):
            s = targets[k]
            idx = int(np.flatnonzero(arclen <= s)[-1])
            idx = min(idx, n - 1)                    # closed has n+1 rows
            span = arclen[idx + 1] - arclen[idx]
            t = 0.0
            if span > 0:
                t = (s - arclen[idx]) / span
            out[k, :] = closed[idx, :] + t * (closed[idx + 1, :] - closed[idx, :])
        return Ring(out)

    def align_to(self, other):
        """Cyclically shift (and possibly flip) this ring to best match
        ``other`` (Ring or array) in least squares; returns a new :class:`Ring`.
        (Port of ``_align_ring_to``.)"""
        B = other.points if isinstance(other, Polyline) else np.asarray(other, float)
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
        return Ring(bestA)

    def split_by_fraction(self, f, nh):
        """Resample the ring into ``2*nh`` points so index 0 is ``points[0]``
        (A1 rail) and index ``nh`` sits at arc-length fraction ``f`` (A2 rail).
        Returns a ``(2*nh,3)`` array.  (Port of ``_split_ring_by_fraction``.)"""
        R = self.points
        Rc = np.vstack([R, R[0, :]])
        al = np.concatenate(
            [[0.0], np.cumsum(np.sqrt(np.sum(np.diff(Rc, axis=0) ** 2, axis=1)))])
        tot = al[-1]
        p1 = self._interp_al(Rc, al, np.linspace(0.0, f * tot, nh + 1))
        p2 = self._interp_al(Rc, al, np.linspace(f * tot, tot, nh + 1))
        return np.vstack([p1[0:nh, :], p2[0:nh, :]])

    @staticmethod
    def _interp_al(Rc, al, tq):
        """Linear interpolation of polyline ``Rc`` (cumulative arc length
        ``al``) at query positions ``tq``."""
        P = np.zeros((tq.size, 3))
        n = Rc.shape[0]
        for i in range(tq.size):
            s = min(max(tq[i], 0.0), al[-1])
            idx = int(np.flatnonzero(al <= s)[-1])
            idx = min(idx, n - 2)
            span = al[idx + 1] - al[idx]
            t = 0.0
            if span > 0:
                t = (s - al[idx]) / span
            P[i, :] = Rc[idx, :] + t * (Rc[idx + 1, :] - Rc[idx, :])
        return P

    @classmethod
    def chain(cls, segs):
        """Chain unordered 3D segments (from marching triangles) into closed
        ordered loops; returns ``list[Ring]``.  (Port of ``chain_segments``.)"""
        if segs is None or len(segs) == 0:
            return []
        segs = np.asarray(segs, dtype=float)
        ns = segs.shape[0]
        pts_raw = np.vstack([segs[:, 0:3], segs[:, 3:6]])

        key = np.round(pts_raw * 1e6).astype(np.int64)
        _, ic = np.unique(key, axis=0, return_inverse=True)
        ic = ic.ravel()
        npts = int(ic.max()) + 1

        coord = np.zeros((npts, 3))
        cnt = np.zeros(npts)
        for i in range(2 * ns):
            coord[ic[i], :] += pts_raw[i, :]
            cnt[ic[i]] += 1
        coord = coord / cnt[:, None]

        seg_ids = np.column_stack([ic[:ns], ic[ns:]])
        adj = [[] for _ in range(npts)]
        for s in range(ns):
            i, j = seg_ids[s]
            adj[i].append(j)
            adj[j].append(i)

        visited_seg = np.zeros(ns, dtype=bool)
        loops = []
        for s in range(ns):
            if visited_seg[s]:
                continue
            start = seg_ids[s, 0]
            cur = seg_ids[s, 1]
            visited_seg[s] = True
            order = [start, cur]
            while cur != start:
                found = False
                for nb in adj[cur]:
                    for s2 in range(ns):
                        if visited_seg[s2]:
                            continue
                        a, b = seg_ids[s2]
                        if (a == cur and b == nb) or (b == cur and a == nb):
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
        return loops
