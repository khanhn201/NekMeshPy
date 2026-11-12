"""Smooth every pipe's surface mesh in the pipe's own (s, R*phi) plane, then rebuild the
column layers as straight segments from each medial node to its (moved) foot.

The surface mesh of a pipe is the projection of the Voronoi-face meshes; where that
projection is strongly stretched (rays converging at an inner bend, a core end seen
edge-on) it leaves rows of sliver quads.  The topology is right -- every pipe is covered
exactly once -- so a constrained Laplacian in the pipe's parameter plane can even them out.
Moves that would turn any incident quad non-convex are rejected; plate nodes stay fixed.
"""
import sys, pickle, numpy as np
from collections import defaultdict
from tubes import TUBES
from geom import R, PLATE
from columns import NL_OF

src = sys.argv[1] if len(sys.argv) > 1 else "out_full"
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
ASP, RINGS = 4.0, 3
D = np.load(src + ".npz"); X = D["X"].copy(); H = D["H"]
keys = pickle.load(open(src + "_keys.pkl", "rb"))

# columns: foot node -> (medial node, pipe, layer ids)
cols = defaultdict(dict)
for k, i in keys.items():
    if k[0] == "c" and isinstance(k[2], (int, np.integer)) and k[2] >= 0:
        cols[(k[1], k[2], k[3])][k[4]] = i
F6 = np.concatenate([H[:, [0, 3, 2, 1]], H[:, [4, 5, 6, 7]], H[:, [0, 1, 5, 4]], H[:, [1, 2, 6, 5]], H[:, [2, 3, 7, 6]], H[:, [3, 0, 4, 7]]])
u, idx, inv, cnt = np.unique(np.sort(F6, 1), axis=0, return_index=True, return_inverse=True, return_counts=True)
BQ = F6[idx[cnt == 1]]


def uv_of(T, P):
    _, s = T.dist(P)
    Tg, N, B = T.frame(s)
    v = P - T.spine(s)
    return s, np.arctan2(np.einsum("ij,ij->i", v, B), np.einsum("ij,ij->i", v, N))


def corner_sine(uvq, sgn):
    """min signed corner sine over the quads (negative: reflex / inverted corner)"""
    e1 = np.roll(uvq, -1, 1) - uvq; e2 = np.roll(uvq, 1, 1) - uvq
    cr = sgn * (e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0])
    return (cr / np.maximum(np.linalg.norm(e1, axis=-1) * np.linalg.norm(e2, axis=-1), 1e-30)).min()


SMIN = 0.1      # a move must keep every incident corner sine above SMIN, or raise the worst one


stats = []
for p, T in TUBES.items():
    d = np.abs(T.dist(X[BQ].reshape(-1, 3))[0]).reshape(-1, 4)
    Q = BQ[(d < 5e-3).all(1)]
    nodes = np.unique(Q)
    loc = {n: i for i, n in enumerate(nodes)}
    q = np.vectorize(loc.get)(Q)
    s, ph = uv_of(T, X[nodes])
    fixed = np.abs(X[nodes, 1] - PLATE) < 1e-6
    # neighbours along quad edges
    nb = defaultdict(set)
    for a in range(4):
        for i, j in zip(q[:, a], q[:, (a + 1) % 4]):
            nb[i].add(j); nb[j].add(i)
    nbl = [np.array(sorted(nb[i])) for i in range(len(nodes))]
    inc = defaultdict(list)
    for k, row in enumerate(q):
        for i in row: inc[i].append(k)
    incl = [np.array(inc[i]) for i in range(len(nodes))]

    def quad_uv(s, ph, qq):
        P = np.stack([s[qq], R * ph[qq]], -1)
        P[..., 1] = P[..., :1, 1] + (P[..., 1] - P[..., :1, 1] + np.pi * R) % (2 * np.pi * R) - np.pi * R
        return P

    a = quad_uv(s, ph, q)
    area = 0.5 * sum(a[:, k, 0] * a[:, (k + 1) % 4, 1] - a[:, (k + 1) % 4, 0] * a[:, k, 1] for k in range(4))
    sgn = np.sign(area.sum())
    L0 = np.linalg.norm(np.roll(a, -1, 1) - a, axis=2); ar0 = (L0.max(1) / L0.min(1)).max()
    # only around slivers: nodes within RINGS quad-rings of a quad with aspect > ASP
    act = np.zeros(len(nodes), bool); act[q[(L0.max(1) / L0.min(1)) > ASP].ravel()] = True
    for _ in range(RINGS):
        act[q[act[q].any(1)].ravel()] = True
    fixed = fixed | ~act
    for it in range(ITERS):
        moved = 0
        for i in np.where(~fixed)[0]:
            j = nbl[i]
            dph = (ph[j] - ph[i] + np.pi) % (2 * np.pi) - np.pi
            ts, tph = s[j].mean(), ph[i] + dph.mean()
            m0 = corner_sine(quad_uv(s, ph, q[incl[i]]), sgn)
            for w in (0.5, 0.2):
                s0, p0 = s[i], ph[i]
                s[i] = s0 + w * (ts - s0); ph[i] = p0 + w * (tph - p0)
                m1 = corner_sine(quad_uv(s, ph, q[incl[i]]), sgn)
                if m1 >= min(m0 + 1e-9, SMIN) and abs(s[i] - s0) + R * abs(ph[i] - p0) > 1e-7:
                    moved += 1; break
                s[i], ph[i] = s0, p0
        if moved == 0:
            break
    a = quad_uv(s, ph, q)
    L1 = np.linalg.norm(np.roll(a, -1, 1) - a, axis=2); ar1 = (L1.max(1) / L1.min(1)).max()
    Pn, _ = T.surf(s, ph)
    X[nodes[~fixed]] = Pn[~fixed]
    stats.append((p, len(q), ar0, ar1))
    print("pipe %3d: %5d quads, max aspect in (s,R phi) %.1f -> %.1f" % (p, len(q), ar0, ar1), flush=True)

# rebuild column layers on pipes: straight from the medial node to the moved foot
for (n0, G, fk), lay in cols.items():
    nl = NL_OF(G)
    if nl not in lay:
        continue
    f = lay[nl]
    for l, i in lay.items():
        if 0 < l < nl:
            X[i] = X[n0] + (l / nl) * (X[f] - X[n0])
np.savez(src + "_ss.npz", X=X, H=H, part=D["part"], gov=D["gov"], sj=D["sj"])
import shutil
shutil.copy(src + "_tags.pkl", src + "_ss_tags.pkl")
shutil.copy(src + "_keys.pkl", src + "_ss_keys.pkl")
