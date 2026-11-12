"""columns: sweep every medial quad to its governors; hex quality."""
import os
import numpy as np
from geom import wall_dist
from tubes import TUBES, BOX
from mesher import graph

_NL = {}


def NL_OF(G):
    """layers from the medial surface to governor G: per pipe, from its 95th-percentile
    ray length (constant per governor, which is all conformity needs)"""
    if G not in _NL:
        if G < 0:
            _NL[G] = int(os.environ.get("EVG_NLWALL", 3))
        else:
            t = np.load(f"maps/tube_{G}.npz")["t"]
            _NL[G] = int(np.clip(round(np.percentile(t, 95) / float(os.environ.get("EVG_NLDIV", 4.0))),
                                 int(os.environ.get("EVG_NLMIN", 2)), int(os.environ.get("EVG_NLMAX", 3))))
    return _NL[G]

LO, HI = BOX
# wall plane: (axis, value)
WALL = {-1: (0, LO[0]), -2: (0, HI[0]), -3: (2, LO[2]), -4: (2, HI[2]), -5: (1, HI[1])}


def node_govs(nodes):
    edges, verts = graph()
    g = {}
    for k, i in nodes.key.items():
        if k[0] == "v":
            g[i] = set(verts[k[1]]["gov"])
        elif k[0] == "e":
            g[i] = set(edges[k[1]]["gov"])
    return g


def _seg_foot(g, X, t0, t1):
    """exact foot on segment g restricted to arclength [t0, t1] (arrays); (dist, point, t)"""
    if g.kind == "line":
        t = np.clip(np.einsum("ij,j->i", X - g.p0, g.t), t0, t1)
        P = g.p0 + t[:, None] * g.t
        return np.linalg.norm(X - P, axis=1), P, t
    v = X - g.c
    a = np.arctan2(v @ g.e2, v @ g.e1)
    best = np.full(len(X), np.inf); BP = np.zeros_like(X); BT = np.zeros(len(X))
    for aa in (a, a + 2 * np.pi, a - 2 * np.pi):
        ac = np.clip(aa, t0 / g.r, t1 / g.r)
        P = g.c + g.r * (np.cos(ac)[:, None] * g.e1 + np.sin(ac)[:, None] * g.e2)
        d = np.linalg.norm(X - P, axis=1)
        m = d < best
        best = np.where(m, d, best); BP = np.where(m[:, None], P, BP); BT = np.where(m, ac * g.r, BT)
    return best, BP, BT


def tube_foot(pid, X, s_ref, win=1e9):
    """foot on tube pid; the global nearest point unless it lies away from s_ref
    (a degenerate node equidistant to two parts of the pipe), then the nearest
    point within win of s_ref.  Deterministic per (node, branch)."""
    from geom import R
    T = TUBES[pid]
    d, s = T.dist(X)
    far = np.abs(s - s_ref) > win
    S = s.copy()
    dg = d + R                              # distance to the spine
    if far.any():
        Xf = X[far]; sr = s_ref[far]
        best = np.full(len(Xf), np.inf); bs = np.zeros(len(Xf))
        for j, g in enumerate(T.segs):
            a0 = T.s0[j]
            lo = np.clip(sr - win - a0, 0, g.L); hi = np.clip(sr + win - a0, 0, g.L)
            ok = hi > lo
            dd, P, t = _seg_foot(g, Xf, lo, hi)
            dd = np.where(ok, dd, np.inf)
            m = dd < best
            best = np.where(m, dd, best); bs = np.where(m, t + a0, bs)
        # keep the global foot unless the windowed one is (nearly) as close: only a
        # node equidistant to two parts of the pipe switches branch
        use = best <= dg[far] + 0.05
        idx = np.where(far)[0][use]
        S[idx] = bs[use]
    sp = T.spine(S)
    n = X - sp
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return sp + R * n, S


def wall_foot(w, X, govs):
    """projection on wall w; on another wall's edge (paper type [1]) -> the box edge"""
    Y = X.copy()
    ax, val = WALL[w]
    Y[:, ax] = val
    for i in range(len(X)):
        for w2 in govs[i]:
            if w2 < 0 and w2 != w and w2 >= -5:
                a2, v2 = WALL[w2]
                Y[i, a2] = v2
    return Y


def sweep(nodes, Q, govs, hexes, owner, tag):
    """append hexes of the columns of quads Q (face A|X) to hexes"""
    XYZ = nodes.arr()
    gv = node_govs(nodes)
    for G in govs:
        Pq = XYZ[Q]                          # (nq, 4, 3)
        flat = Pq.reshape(-1, 3)
        if G >= 0:
            from tubes import TUBES as TT
            _, sc = TT[G].dist(Pq.mean(1))
            F, S = tube_foot(G, flat, np.repeat(sc, 4))
            fkey = np.round(S, 1)
        else:
            F = wall_foot(G, flat, [gv.get(i, set()) for i in Q.ravel()])
            fkey = np.zeros(len(flat))
        from geom import PLATE
        onp = np.abs(flat[:, 1] - PLATE) < 1e-6
        F[onp, 1] = PLATE                     # a plate node's column stays in the plate
        F = F.reshape(-1, 4, 3); fkey = fkey.reshape(-1, 4).astype(object)
        if G < 0:
            # a foot on a box edge/corner is shared by every wall column reaching it
            for qi, c in np.ndindex(len(Q), 4):
                ws = sorted(w for w in gv.get(Q[qi, c], set()) if -5 <= w < 0)
                if len(ws) >= 2:
                    fkey[qi, c] = ("edge",) + tuple(ws)
        NL = NL_OF(G)
        lay = np.empty((len(Q), 4, NL + 1), int)
        for qi in range(len(Q)):
            for c in range(4):
                n0 = Q[qi, c]
                lay[qi, c, 0] = n0
                for l in range(1, NL + 1):
                    x = XYZ[n0] + (l / NL) * (F[qi, c] - XYZ[n0])
                    fk = fkey[qi, c]
                    gk = "W" if isinstance(fk, tuple) else G
                    lay[qi, c, l] = nodes.get(("c", n0, gk, fk, l), x)
        for l in range(NL):
            h = np.concatenate([lay[:, :, l], lay[:, :, l + 1]], 1)
            hexes.extend(h.tolist()); owner.extend([(tag, G)] * len(h))


# ---------------------------------------------------------------- quality
def scaled_jacobian(P):
    """P (n, 8, 3) hex corners (VTK order) -> min scaled Jacobian over the 8 corners"""
    nb = [(1, 3, 4), (2, 0, 5), (3, 1, 6), (0, 2, 7), (7, 5, 0), (4, 6, 1), (5, 7, 2), (6, 4, 3)]
    out = np.full(len(P), np.inf)
    for c, (a, b, d) in enumerate(nb):
        e1 = P[:, a] - P[:, c]; e2 = P[:, b] - P[:, c]; e3 = P[:, d] - P[:, c]
        j = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
        n = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(e3, axis=1)
        out = np.minimum(out, j / np.maximum(n, 1e-30))
    return out


def orient(H, XYZ):
    """flip hexes with negative mean Jacobian"""
    H = np.array(H)
    P = XYZ[H]
    c = P.mean(1)
    e1 = P[:, [1, 2, 6, 5]].mean(1) - P[:, [0, 3, 7, 4]].mean(1)
    e2 = P[:, [3, 2, 6, 7]].mean(1) - P[:, [0, 1, 5, 4]].mean(1)
    e3 = P[:, 4:].mean(1) - P[:, :4].mean(1)
    s = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
    H[s < 0] = H[s < 0][:, [4, 5, 6, 7, 0, 1, 2, 3]]
    return H
