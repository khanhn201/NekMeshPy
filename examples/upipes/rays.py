"""Foot partition of every governor: march each surface normal ray until another
governor ties.  The tie point is a point of the Voronoi diagram (medial surface), the
label says which governor shares it.  Labels:
    >= 0      : another pipe id
    -1 .. -5  : box walls x-, x+, z-, z+, y+
    SELF (-10): the same pipe, a non-adjacent part (the face between the two legs)
    FOCAL(-11): the same pipe's bend axis (a continuum of feet -- the paper's open case)
"""
import sys, time, numpy as np
from geom import R, PLATE, wall_dist
from tubes import TUBES, BOX, neighbours

SELF, FOCAL = -10, -11
TMAX = 70.0
GAP = 3.0


def march(T, s, phi, cand, iters=40):
    """rays of tube T at (s, phi) -> t*, label, tie point"""
    a, n = T.surf(s, phi)
    lo = np.zeros(len(s))
    hi = np.full(len(s), TMAX)

    def g(t):
        p = a + t[:, None] * n
        best = np.full(len(s), np.inf)
        lab = np.full(len(s), -99)
        for q in cand:
            d, _ = TUBES[q].dist(p)
            m = d < best
            best = np.where(m, d, best)
            lab = np.where(m, q, lab)
        W = wall_dist(p, BOX)
        for k in range(W.shape[1]):
            m = W[:, k] < best
            best = np.where(m, W[:, k], best)
            lab = np.where(m, -1 - k, lab)
        ds = T.self_dist(p, s, GAP)
        m = ds < best
        best = np.where(m, ds, best)
        lab = np.where(m, SELF, lab)
        return best - t, lab

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v, _ = g(mid)
        pos = v > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    t = 0.5 * (lo + hi)
    _, lab = g(t + 1e-6)
    p = a + t[:, None] * n
    # focal: self tie lying on the axis of the bend that owns the ray
    foc = np.zeros(len(s), bool)
    k = T.seg_of(s)
    for j, gseg in enumerate(T.segs):
        if gseg.kind != "arc":
            continue
        m = (lab == SELF) & (k == j)
        if m.any():
            v = p[m] - gseg.c
            off = v - np.outer(v @ gseg.n, gseg.n)
            foc[np.where(m)[0][np.linalg.norm(off, axis=1) < 0.35 * gseg.r]] = True
    lab = np.where(foc, FOCAL, lab)
    return t, lab, p


def tube_map(pid, ds=0.5, nphi=96, chunk=6000):
    T = TUBES[pid]
    cand = neighbours(pid, 2 * TMAX)
    S = np.arange(0.0, T.L + 1e-9, ds)
    Ph = np.arange(nphi) * 2 * np.pi / nphi
    SS, PP = np.meshgrid(S, Ph, indexing="ij")
    s, ph = SS.ravel(), PP.ravel()
    t = np.empty(len(s)); lab = np.empty(len(s), int); P = np.empty((len(s), 3))
    for i in range(0, len(s), chunk):
        sl = slice(i, i + chunk)
        t[sl], lab[sl], P[sl] = march(T, s[sl], ph[sl], cand)
    sh = SS.shape
    return dict(S=S, Ph=Ph, t=t.reshape(sh), lab=lab.reshape(sh), P=P.reshape(sh + (3,)), cand=cand)


if __name__ == "__main__":
    pids = [int(a) for a in sys.argv[1:]] or sorted(TUBES)
    for pid in pids:
        t0 = time.time()
        M = tube_map(pid)
        np.savez_compressed(f"maps/tube_{pid}.npz", **M)
        u, c = np.unique(M["lab"], return_counts=True)
        print(pid, f"{time.time()-t0:.1f}s", dict(zip(u.tolist(), c.tolist())), flush=True)
