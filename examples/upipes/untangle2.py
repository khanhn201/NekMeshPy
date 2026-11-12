"""Local min-max untangling: Gauss-Seidel over free nodes near poor hexes; each node moves
to maximise the worst scaled Jacobian of its incident hexes (pattern search).  Nodes on one
boundary surface slide on it (snapped back); nodes on two surfaces are fixed."""
import sys, numpy as np, meshio
from collections import defaultdict
from columns import scaled_jacobian
from geom import PLATE, R
from tubes import TUBES, BOX

src = sys.argv[1] if len(sys.argv) > 1 else "out_full"
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
D = np.load(src + ".npz"); X, H = D["X"].copy(), D["H"]
used = np.unique(H); remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
X, H = X[used], remap[H]
F = np.concatenate([H[:, [0,1,2,3]], H[:, [4,5,6,7]], H[:, [0,1,5,4]], H[:, [1,2,6,5]], H[:, [2,3,7,6]], H[:, [3,0,4,7]]])
u, inv, cnt = np.unique(np.sort(F, 1), axis=0, return_inverse=True, return_counts=True)
bnode = np.zeros(len(X), bool); bnode[F[cnt[inv] == 1].ravel()] = True
lo, hi = BOX
WALLS = [(0, lo[0]), (0, hi[0]), (2, lo[2]), (2, hi[2]), (1, hi[1])]
surf = [[] for _ in range(len(X))]
for i in np.where(bnode & (np.abs(X[:, 1] - PLATE) < 1e-6))[0]: surf[i].append(("p",))
for k, (ax, v) in enumerate(WALLS):
    for i in np.where(bnode & (np.abs(X[:, ax] - v) < 1e-6))[0]: surf[i].append(("w", k))
for p, T in TUBES.items():
    for i in np.where(bnode & (np.abs(T.dist(X)[0]) < 1e-3))[0]: surf[i].append(("t", p))
import curves as CV
ckind = [CV.kind(s) if len(s) == 2 else None for s in surf]
fixed = np.array([(len(s) >= 2 and ck is None) or (bnode[i] and len(s) == 0) for i, (s, ck) in enumerate(zip(surf, ckind))])

def snap(i, P):
    if not surf[i]:
        return P
    if ckind[i] is not None:
        return CV.snap(ckind[i], P)
    s = surf[i][0]
    P = P.copy()
    if s[0] == "p":
        P[:, 1] = PLATE
    elif s[0] == "w":
        P[:, WALLS[s[1]][0]] = WALLS[s[1]][1]
    else:
        T = TUBES[s[1]]; d, sp = T.dist(P); c = T.spine(sp); v = P - c
        P = c + R * v / np.linalg.norm(v, axis=1, keepdims=True)
    return P

EP = np.array([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)])
L0 = np.linalg.norm(X[H[:, EP[:, 0]]] - X[H[:, EP[:, 1]]], axis=2).min(1)
def lattice_sj(P, n=3):
    r = np.linspace(0, 1, n); worst = np.full(len(P), np.inf)
    for u in r:
        for v in r:
            for w in r:
                du = ((1-v)*(1-w)*(P[:,1]-P[:,0]) + v*(1-w)*(P[:,2]-P[:,3]) + (1-v)*w*(P[:,5]-P[:,4]) + v*w*(P[:,6]-P[:,7]))
                dv = ((1-u)*(1-w)*(P[:,3]-P[:,0]) + u*(1-w)*(P[:,2]-P[:,1]) + (1-u)*w*(P[:,7]-P[:,4]) + u*w*(P[:,6]-P[:,5]))
                dw = ((1-u)*(1-v)*(P[:,4]-P[:,0]) + u*(1-v)*(P[:,5]-P[:,1]) + (1-u)*v*(P[:,7]-P[:,3]) + u*v*(P[:,6]-P[:,2]))
                j = np.einsum("ij,ij->i", np.cross(du, dv), dw)
                worst = np.minimum(worst, j / (np.linalg.norm(du, axis=1) * np.linalg.norm(dv, axis=1) * np.linalg.norm(dw, axis=1) + 1e-300))
    return worst


FP = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 3, 2, 1], [4, 5, 6, 7]])
def twisted(P):
    q = P[:, FP]
    turn = np.cross(np.roll(q, -1, axis=2) - q, np.roll(q, 1, axis=2) - q)
    mean = turn.sum(axis=2, keepdims=True)
    agree = np.einsum("efkc,efkc->efk", turn, np.broadcast_to(mean, turn.shape))
    m = np.linalg.norm(turn, axis=3) * np.linalg.norm(mean, axis=3)
    return (agree / np.maximum(m, 1e-300)).min(axis=(1, 2))      # cosine; <= 0 is twisted


def merit(P, idx):
    """min scaled Jacobian over corners and a 3^3 lattice; -1 if an edge shrinks below 40%
    of the hex's original shortest edge"""
    q = np.minimum(np.minimum(scaled_jacobian(P), lattice_sj(P)), twisted(P))
    L = np.linalg.norm(P[:, EP[:, 0]] - P[:, EP[:, 1]], axis=2).min(1)
    return np.where(L < 0.4 * L0[idx], -1.0, q)

inc = defaultdict(list)
for h, row in enumerate(H):
    for n in row: inc[n].append(h)
dirs = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)], float)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
sj = merit(X[H], np.arange(len(H)))
print("before: inverted %d, <%.2f: %d, min %.3f" % ((sj < 0).sum(), THR, (sj < THR).sum(), sj.min()))
for sweep_ in range(int(sys.argv[3]) if len(sys.argv) > 3 else 40):
    bad = np.where(sj < THR)[0]
    if not len(bad):
        break
    cand = np.zeros(len(X), bool); cand[H[bad].ravel()] = True
    for _ in range(2):
        hs = cand[H].any(1); cand[H[hs].ravel()] = True
    cand &= ~fixed
    moved = 0
    for v in np.where(cand)[0]:
        hs = np.array(inc[v])
        f0 = sj[hs].min()
        if f0 >= THR:
            continue
        E = H[hs]
        nb = np.unique(E.ravel()); nb = nb[nb != v]
        ell = np.min(np.linalg.norm(X[nb] - X[v], axis=1))
        best, bestP = f0, None
        for step in (1.0, 0.5, 0.25, 0.1, 0.04, 0.015):
            P = X[v] + step * ell * dirs
            P = snap(v, P)
            Pa = X[E][None].repeat(len(P), 0)          # (k, nh, 8, 3)
            mask = (E == v)
            Pa[:, mask] = P[:, None, :].repeat(mask.sum(), 1)
            f = merit(Pa.reshape(-1, 8, 3), np.tile(hs, len(P))).reshape(len(P), len(hs)).min(1)
            k = int(np.argmax(f))
            if f[k] > best + 1e-6:
                best, bestP = f[k], P[k]
        if bestP is not None:
            X[v] = bestP; sj[hs] = merit(X[H[hs]], hs); moved += 1
    print("sweep %d: moved %d, inverted %d, <%.2f: %d, min %.3f" % (sweep_, moved, (sj < 0).sum(), THR, (sj < THR).sum(), sj.min()))
    if moved == 0:
        break
sj = scaled_jacobian(X[H])
np.savez(src + "_u.npz", X=X, H=H, sj=sj, part=D["part"], gov=D["gov"])
import shutil; shutil.copy(src + "_tags.pkl", src + "_u_tags.pkl")
meshio.write(src + "_u.vtu", meshio.Mesh(X, [("hexahedron", H)], cell_data={"sj": [sj], "part": [D["part"]], "gov": [D["gov"]]}))
