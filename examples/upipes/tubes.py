"""Tube-level geometry on top of geom.py: one merged canal surface per pipe.

A pipe's faces (legs/bends/tops) are G1-joined, so for distance purposes the pipe is
one surface (distance to its whole spine minus R).  The faces are still kept apart
where the paper needs them: a ray from one part of a pipe may end on another,
non-adjacent part of the same pipe (the self-face between the two legs of a U, or
the focal axis of a bend) -- that is what `self_dist` sees.

Surface parameters: s = arclength along the spine (s = 0 at the plate on the first
leg), phi = angle around the spine in the frame (N, B), B = normal of the pipe's
plane, N = B x T.  Spines are planar, so this frame does not twist.
"""
import numpy as np
from geom import build, R, PLATE, BELOW, wall_dist, WALLS

SEGS, OF_PIPE, BOX = build()
PIPES = sorted(OF_PIPE)


class Tube:
    def __init__(self, pid):
        self.pid = pid
        self.segs = [SEGS[g] for g in OF_PIPE[pid]]
        self.s0 = np.cumsum([0.0] + [g.L for g in self.segs])[:-1] - BELOW   # s at segment start
        self.L = self.s0[-1] + self.segs[-1].L - BELOW                        # s at the far plate
        arcs = [g for g in self.segs if g.kind == "arc"]
        n = arcs[0].n
        self.B = n / np.linalg.norm(n)

    # ---- spine --------------------------------------------------------------
    def seg_of(self, s):
        s = np.asarray(s, float)
        k = np.clip(np.searchsorted(self.s0, s, side="right") - 1, 0, len(self.segs) - 1)
        return k

    def spine(self, s):
        s = np.asarray(s, float)
        k = self.seg_of(s)
        out = np.empty(s.shape + (3,))
        for j, g in enumerate(self.segs):
            m = k == j
            if m.any():
                out[m] = g.point(s[m] - self.s0[j])
        return out

    def tangent(self, s):
        s = np.asarray(s, float)
        k = self.seg_of(s)
        out = np.empty(s.shape + (3,))
        for j, g in enumerate(self.segs):
            m = k == j
            if m.any():
                out[m] = g.tangent(s[m] - self.s0[j])
        return out

    def frame(self, s):
        T = self.tangent(s)
        B = np.broadcast_to(self.B, T.shape)
        N = np.cross(B, T)
        return T, N, B

    def surf(self, s, phi):
        """surface point and outward unit normal"""
        s, phi = np.broadcast_arrays(np.asarray(s, float), np.asarray(phi, float))
        T, N, B = self.frame(s)
        n = np.cos(phi)[..., None] * N + np.sin(phi)[..., None] * B
        return self.spine(s) + R * n, n

    # ---- distances ----------------------------------------------------------
    def dist(self, X):
        """distance to the pipe surface, and spine parameter of the foot"""
        best = np.full(X.shape[:-1], np.inf)
        sb = np.zeros(X.shape[:-1])
        for j, g in enumerate(self.segs):
            d, s, _ = g.foot(X)
            m = d < best
            best = np.where(m, d, best)
            sb = np.where(m, s + self.s0[j], sb)
        return best - R, sb

    def self_dist(self, X, s_own, gap):
        """distance from X to the spine excluding |s - s_own| < gap  (minus R)"""
        best = np.full(X.shape[:-1], np.inf)
        lo_ex, hi_ex = s_own - gap, s_own + gap
        for j, g in enumerate(self.segs):
            a0 = self.s0[j]
            a1 = a0 + g.L
            # allowed parameter set on this segment: [a0,a1] minus (lo_ex, hi_ex)
            for (u0, u1) in ((np.full_like(s_own, a0), np.minimum(lo_ex, a1)),
                             (np.maximum(hi_ex, a0), np.full_like(s_own, a1))):
                ok = u1 > u0 + 1e-9
                if not ok.any():
                    continue
                d = _clamped_seg_dist(g, X, u0 - a0, u1 - a0)
                best = np.where(ok, np.minimum(best, d), best)
        return best - R


def _clamped_seg_dist(g, X, t0, t1):
    """distance from X to segment g restricted to arclength [t0, t1] (arrays)"""
    if g.kind == "line":
        s = np.einsum("...k,k->...", X - g.p0, g.t)
        s = np.clip(s, t0, t1)
        return np.linalg.norm(X - g.p0 - s[..., None] * g.t, axis=-1)
    v = X - g.c
    x1 = np.einsum("...k,k->...", v, g.e1)
    x2 = np.einsum("...k,k->...", v, g.e2)
    a = np.arctan2(x2, x1)
    a0, a1 = t0 / g.r, t1 / g.r
    # clamp angle a (mod 2pi) into [a0, a1]: candidates a, a+2pi, a-2pi, endpoints
    best = np.full(np.shape(a), np.inf)
    for aa in (a, a + 2 * np.pi, a - 2 * np.pi):
        ac = np.clip(aa, a0, a1)
        P = g.c + g.r * (np.cos(ac)[..., None] * g.e1 + np.sin(ac)[..., None] * g.e2)
        best = np.minimum(best, np.linalg.norm(X - P, axis=-1))
    return best


TUBES = {p: Tube(p) for p in PIPES}


def spine_samples(step=1.0):
    out, owner = [], []
    for p, T in TUBES.items():
        s = np.arange(-BELOW, T.L + BELOW + 1e-9, step)
        out.append(T.spine(s))
        owner.append(np.full(len(s), p))
    return np.concatenate(out), np.concatenate(owner)


def neighbours(pid, reach):
    """pipes whose spine comes within `reach` of pid's spine"""
    P, own = spine_samples(2.0)
    me = P[own == pid]
    from scipy.spatial import cKDTree
    t = cKDTree(P[own != pid])
    hits = t.query_ball_point(me, reach)
    ids = np.unique(np.concatenate([own[own != pid][h] for h in hits if h] or [np.array([], int)]))
    return [int(i) for i in ids]
