"""constraint curves where two boundary surfaces meet: pipe/plate circles, wall/plate lines,
box edges.  snap onto the curve and project a direction onto its tangent."""
import numpy as np
from tubes import TUBES, BOX
from geom import R, PLATE
lo, hi = BOX
WALLS = [(0, lo[0]), (0, hi[0]), (2, lo[2]), (2, hi[2]), (1, hi[1])]


def kind(surfs):
    s = sorted(surfs, key=str)
    if len(s) != 2:
        return None
    a, b = s
    if {a[0], b[0]} == {"p", "t"}:
        return ("circle", (a if a[0] == "t" else b)[1])
    if {a[0], b[0]} == {"p", "w"}:
        return ("line", (WALLS[(a if a[0] == "w" else b)[1]][0], 1),
                (WALLS[(a if a[0] == "w" else b)[1]][1], PLATE))
    if a[0] == "w" and b[0] == "w":
        (ax1, v1), (ax2, v2) = WALLS[a[1]], WALLS[b[1]]
        if ax1 == ax2:
            return None
        return ("line", (ax1, ax2), (v1, v2))
    return None


def snap(k, P):
    P = np.array(P, float, copy=True)
    if k[0] == "circle":
        T = TUBES[k[1]]
        P[..., 1] = PLATE
        _, s = T.dist(P.reshape(-1, 3)); c = T.spine(s).reshape(P.shape)
        v = P - c; v[..., 1] = 0
        P = c + R * v / np.linalg.norm(v, axis=-1, keepdims=True)
        P[..., 1] = PLATE
        return P
    (a1, a2), (v1, v2) = k[1], k[2]
    P[..., a1] = v1; P[..., a2] = v2
    return P


def tangent(k, P):
    P = np.atleast_2d(P)
    if k[0] == "circle":
        T = TUBES[k[1]]
        _, s = T.dist(P); v = P - T.spine(s); v[:, 1] = 0
        t = np.cross([0, 1.0, 0], v)
        return t / np.linalg.norm(t, axis=1, keepdims=True)
    t = np.zeros_like(P)
    free = [a for a in range(3) if a not in k[1]][0]
    t[:, free] = 1.0
    return t
