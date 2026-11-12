"""Global smoothing of the finished mesh (paper sec. 5.2: the decomposition is removed and
the mesh is smoothed as a whole).

1. surfaces: every pipe in its own (s, R*phi) map, every box wall and the plate in-plane;
   nodes on two surfaces (pipe/plate circles, box edges, wall/plate lines) stay fixed; a
   move is kept only if every incident surface quad stays convex.
2. columns: layer nodes rebuilt as straight segments medial node -> moved foot.
3. volume: the repo's CG shape optimizer over all hexes, boundary nodes sliding on their
   surface (projected gradient + snap), inversion never accepted.
usage: python gsmooth.py SRC [surface_iters] [volume_iters]  -> SRC_g.npz
"""
import os
import sys, pickle, numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from nekmeshpy.hexmesh.smoother.smooth import quality, gradient, _eval
from nekmeshpy.hexmesh.smoother.constrain import vertex_length
from nekmeshpy.core.smoother import descent
from nekmeshpy.core.smoother.metric import KAPPA, EPS, DELTA
from tubes import TUBES, BOX
from geom import R, PLATE
from columns import NL_OF, scaled_jacobian

src = sys.argv[1]
SIT = int(sys.argv[2]) if len(sys.argv) > 2 else 60
VIT = int(sys.argv[3]) if len(sys.argv) > 3 else 200
D = np.load(src + ".npz"); X = D["X"].copy(); H = D["H"]
keys = pickle.load(open(src + "_keys.pkl", "rb")) if SIT > 0 else None
lo, hi = BOX
WALLS = [(0, lo[0]), (0, hi[0]), (2, lo[2]), (2, hi[2]), (1, hi[1])]

F6 = np.concatenate([H[:, [0,3,2,1]], H[:, [4,5,6,7]], H[:, [0,1,5,4]], H[:, [1,2,6,5]], H[:, [2,3,7,6]], H[:, [3,0,4,7]]])
u, idx, inv, cnt = np.unique(np.sort(F6, 1), axis=0, return_index=True, return_inverse=True, return_counts=True)
BQ = F6[idx[cnt == 1]]
bnode = np.zeros(len(X), bool); bnode[BQ.ravel()] = True

# surface membership
surf = defaultdict(list)                       # node -> list of surface ids
SURF_Q = {}                                     # surface id -> quads
for k, (ax, v) in enumerate(WALLS):
    m = np.all(np.abs(X[BQ][..., ax] - v) < 1e-6, 1); SURF_Q[("w", k)] = BQ[m]
    for i in np.where(bnode & (np.abs(X[:, ax] - v) < 1e-6))[0]: surf[i].append(("w", k))
m = np.all(np.abs(X[BQ][..., 1] - PLATE) < 1e-6, 1); SURF_Q[("p",)] = BQ[m]
for i in np.where(bnode & (np.abs(X[:, 1] - PLATE) < 1e-6))[0]: surf[i].append(("p",))
for p, T in TUBES.items():
    d = np.abs(T.dist(X[BQ].reshape(-1, 3))[0]).reshape(-1, 4)
    SURF_Q[("t", p)] = BQ[(d < 5e-3).all(1)]
    for i in np.where(bnode & (np.abs(T.dist(X)[0]) < 5e-3))[0]: surf[i].append(("t", p))
fixed_curve = np.array([len(surf.get(i, [])) >= 2 for i in range(len(X))])


def to2(sid, P):
    """surface coordinates (n,2) of points P on surface sid"""
    if sid[0] == "t":
        T = TUBES[sid[1]]; _, s = T.dist(P); Tg, N, B = T.frame(s); v = P - T.spine(s)
        return np.stack([s, R * np.arctan2(np.einsum("ij,ij->i", v, B), np.einsum("ij,ij->i", v, N))], 1)
    ax = WALLS[sid[1]][0] if sid[0] == "w" else 1
    o = [a for a in range(3) if a != ax]
    return P[:, o]


def to3(sid, uv, P0):
    if sid[0] == "t":
        T = TUBES[sid[1]]; Pn, _ = T.surf(uv[:, 0], uv[:, 1] / R); return Pn
    ax = WALLS[sid[1]][0] if sid[0] == "w" else 1
    o = [a for a in range(3) if a != ax]
    P = P0.copy(); P[:, o] = uv; return P


def surface_smooth(sid, iters):
    Q = SURF_Q[sid]
    if not len(Q):
        return
    nodes = np.unique(Q); loc = -np.ones(len(X), int); loc[nodes] = np.arange(len(nodes))
    q = loc[Q]
    uv = to2(sid, X[nodes])
    per = 2 * np.pi * R if sid[0] == "t" else None
    free = ~fixed_curve[nodes]
    nb = defaultdict(set); inc = defaultdict(list)
    for k, row in enumerate(q):
        for a in range(4):
            nb[row[a]].add(row[(a + 1) % 4]); nb[row[(a + 1) % 4]].add(row[a]); inc[row[a]].append(k)
    nbl = {i: np.array(sorted(nb[i])) for i in range(len(nodes))}
    incl = {i: np.array(inc[i]) for i in range(len(nodes))}

    def quv(qq):
        P = uv[q[qq]].copy()
        if per:
            P[..., 1] = P[..., :1, 1] + (P[..., 1] - P[..., :1, 1] + per / 2) % per - per / 2
        return P

    def turns(qq):
        P = quv(qq); e1 = np.roll(P, -1, 1) - P; e2 = np.roll(P, 1, 1) - P
        return (e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]) / (np.linalg.norm(e1, axis=2) * np.linalg.norm(e2, axis=2) + 1e-300)

    sg = np.sign(np.median(turns(np.arange(len(q)))))
    before = (sg * turns(np.arange(len(q)))).min()
    for it in range(iters):
        moved = 0
        for i in np.where(free)[0]:
            j = nbl[i]; d = uv[j] - uv[i]
            if per:
                d[:, 1] = (d[:, 1] + per / 2) % per - per / 2
            t = uv[i] + d.mean(0)
            o = uv[i].copy(); m0 = (sg * turns(incl[i])).min()
            for w in (0.5, 0.2):
                uv[i] = o + w * (t - o)
                if (sg * turns(incl[i])).min() >= min(m0, 0.5) - 1e-12:
                    moved += 1; break
                uv[i] = o
        if not moved:
            break
    after = (sg * turns(np.arange(len(q)))).min()
    X[nodes[free]] = to3(sid, uv[free], X[nodes[free]])
    return before, after


from nekmeshpy.quadmesh.smoother import smooth as QS


def surf_normal(sid, P):
    if sid[0] == "t":
        T = TUBES[sid[1]]; _, sp = T.dist(P); v = P - T.spine(sp)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    n = np.zeros_like(P); n[:, WALLS[sid[1]][0] if sid[0] == "w" else 1] = 1.0
    return n


def surf_snap(sid, P):
    if sid[0] == "t":
        T = TUBES[sid[1]]; _, sp = T.dist(P); c = T.spine(sp); v = P - c
        return c + R * v / np.linalg.norm(v, axis=1, keepdims=True)
    P = P.copy()
    if sid[0] == "w":
        P[:, WALLS[sid[1]][0]] = WALLS[sid[1]][1]
    else:
        P[:, 1] = PLATE
    return P


def param_check(sid, Pn, q):
    """(non-convex quads, total unrolled area) of a pipe's surface mesh in (s, R phi);
    a fold shows as extra area (the tiling of the cylinder is otherwise exact)"""
    if sid[0] != "t":
        return 0, 0.0
    T = TUBES[sid[1]]
    _, s_ = T.dist(Pn); Tg, Nf, Bf = T.frame(s_); v = Pn - T.spine(s_)
    ph = np.arctan2(np.einsum("ij,ij->i", v, Bf), np.einsum("ij,ij->i", v, Nf))
    uv = np.stack([s_[q], R * ph[q]], -1)
    uv[..., 1] = uv[..., :1, 1] + (uv[..., 1] - uv[..., :1, 1] + np.pi * R) % (2 * np.pi * R) - np.pi * R
    e1 = np.roll(uv, -1, 1) - uv; e2 = np.roll(uv, 1, 1) - uv
    cr = e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]
    sg = np.sign(np.median(cr))
    area = 0.5 * sum(uv[:, k, 0] * uv[:, (k + 1) % 4, 1] - uv[:, (k + 1) % 4, 0] * uv[:, k, 1] for k in range(4))
    return int(((sg * cr) <= 0).any(1).sum()), float(np.abs(area).sum())


def surface_opt(sid, iters, chunk=25):
    """shape optimisation of one boundary surface's quad mesh (repo quad metric, CG),
    nodes sliding on the exact surface (curve nodes on their curve), in rounds with the
    reference normals refreshed; a round that folds a pipe surface is reverted"""
    Q = SURF_Q[sid]
    if not len(Q):
        return None
    nodes = np.unique(Q); loc = -np.ones(len(X), int); loc[nodes] = np.arange(len(nodes))
    q = loc[Q]; P = X[nodes].copy(); P0 = P.copy()
    mov = ~fixed_curve[nodes]
    def nref_of(Z):
        r = surf_normal(sid, Z[q].mean(1))
        ng = np.cross(Z[q[:, 2]] - Z[q[:, 0]], Z[q[:, 3]] - Z[q[:, 1]])
        return r * np.sign(np.median(np.einsum("ij,ij->i", ng, r)))
    before = QS.quality(P, q, nref_of(P))
    nb0, ar0 = param_check(sid, P, q)
    ell = np.full((len(P), 1), 1.0)
    L = np.linalg.norm(P[q] - P[np.roll(q, -1, 1)], axis=2)
    for k in range(4):
        np.minimum.at(ell[:, 0], q[:, k], L[:, k])
    def proj(d, Z):
        d = np.where(mov[:, None], d, 0.0); n = surf_normal(sid, Z)
        return d - np.einsum("ij,ij->i", d, n)[:, None] * n
    rounds, stop = 0, ""
    for r0 in range(0, iters, chunk):
        nref = nref_of(P)
        res = descent.minimize(
            P, evaluate=lambda Z: QS._eval(Z, q, nref, KAPPA, EPS, DELTA, None),
            gradient=lambda Z: QS.gradient(Z, q, nref, kappa=KAPPA, eps=EPS, delta=DELTA),
            project=proj, snap=lambda Z: np.where(mov[:, None], surf_snap(sid, Z), P0),
            summary=lambda Z: (0, 0.0, 0.0), ell=ell, movable=mov, tangent=mov,
            n_used=len(q), iters=min(chunk, iters - r0), alpha_max=0.5, tol_rel=1e-8)
        Pn = np.where(mov[:, None], surf_snap(sid, res.P), P0)
        nb, ar = param_check(sid, Pn, q)
        if QS.quality(Pn, q, nref_of(Pn)).n_inverted > before.n_inverted or nb > nb0 or ar > ar0 * (1 + 5e-4):
            stop = " (stopped: fold, round %d)" % rounds
            break
        P = Pn; rounds += 1
    if sid[0] == "t" and rounds == 0:
        P, stop = straight_opt(sid, P0, q, mov, iters, chunk, before, nb0, ar0, nref_of)
    after = QS.quality(P, q, nref_of(P))
    X[nodes] = P
    return before, after, stop


def straight_opt(sid, P0, q, mov, iters, chunk, before, nb0, ar0, nref_of):
    """same optimisation on the pipe's straightened copy (s, R phi) -> (R cos phi, R sin phi, s):
    isometric to the parameter plane, so the inner-bend compression cannot fold it"""
    T = TUBES[sid[1]]
    _, s_ = T.dist(P0); Tg, Nf, Bf = T.frame(s_); v = P0 - T.spine(s_)
    ph = np.arctan2(np.einsum("ij,ij->i", v, Bf), np.einsum("ij,ij->i", v, Nf))
    Y0 = np.stack([R * np.cos(ph), R * np.sin(ph), s_], 1)
    def back(Y):
        return T.surf(Y[:, 2], np.arctan2(Y[:, 1], Y[:, 0]))[0]
    def snapY(Y):
        Y = Y.copy(); Y[:, :2] *= R / np.linalg.norm(Y[:, :2], axis=1, keepdims=True)
        return np.where(mov[:, None], Y, Y0)
    def nY(Y):
        n = Y.copy(); n[:, 2] = 0
        return n / np.linalg.norm(n, axis=1, keepdims=True)
    def nrefY(Y):
        r = nY(Y[q].mean(1))
        ng = np.cross(Y[q[:, 2]] - Y[q[:, 0]], Y[q[:, 3]] - Y[q[:, 1]])
        return r * np.sign(np.median(np.einsum("ij,ij->i", ng, r)))
    def proj(d, Z):
        d = np.where(mov[:, None], d, 0.0); n = nY(Z)
        return d - np.einsum("ij,ij->i", d, n)[:, None] * n
    ell = np.full((len(Y0), 1), 1.0)
    L = np.linalg.norm(Y0[q] - Y0[np.roll(q, -1, 1)], axis=2)
    for k in range(4):
        np.minimum.at(ell[:, 0], q[:, k], L[:, k])
    Y, P, rounds = Y0, P0, 0
    for r0 in range(0, iters, chunk):
        nref = nrefY(Y)
        res = descent.minimize(
            Y, evaluate=lambda Z: QS._eval(Z, q, nref, KAPPA, EPS, DELTA, None),
            gradient=lambda Z: QS.gradient(Z, q, nref, kappa=KAPPA, eps=EPS, delta=DELTA),
            project=proj, snap=snapY, summary=lambda Z: (0, 0.0, 0.0), ell=ell, movable=mov, tangent=mov,
            n_used=len(q), iters=min(chunk, iters - r0), alpha_max=0.5, tol_rel=1e-8)
        Yn = snapY(res.P); Pn = np.where(mov[:, None], back(Yn), P0)
        nb, ar = param_check(sid, Pn, q)
        if QS.quality(Pn, q, nref_of(Pn)).n_inverted > before.n_inverted or nb > nb0 or ar > ar0 * (1 + 5e-4):
            return P, " (straightened: %d rounds, then fold)" % rounds
        Y, P, rounds = Yn, Pn, rounds + 1
    return P, " (straightened: %d rounds)" % rounds


if SIT > 0:
    for sid in SURF_Q:
        r = surface_opt(sid, SIT)
        if r:
            b, a, stop = r
            print("surface %-9s quads %5d: inverted %d -> %d, min sJ %.2f -> %.2f, worst1%% %.2f -> %.2f%s"
                  % (str(sid), len(SURF_Q[sid]), b.n_inverted, a.n_inverted, b.sj_min, a.sj_min, b.sj_q01, a.sj_q01, stop), flush=True)
    for sid in []:
        r = surface_smooth(sid, SIT)
        if r and sid[0] != "t":
            print("surface", sid, "min corner sine %.2f -> %.2f" % r, flush=True)
    print("pipes smoothed", flush=True)
    # columns: straight segments medial node -> foot (pipes and walls)
    cols = defaultdict(dict)
    for k, i in keys.items():
        if k[0] == "c":
            cols[(k[1], k[2], k[3])][k[4]] = i
    for (n0, G, fk), lay in cols.items():
        nl = max(lay)
        f = lay[nl]
        for l, i in lay.items():
            if 0 < l < nl:
                X[i] = X[n0] + (l / nl) * (X[f] - X[n0])

# volume optimisation, boundary nodes sliding
used = np.unique(H); remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
Xc, Hc = X[used], remap[H]
surfc = [surf.get(i, []) for i in used]
import curves as CV
ckind = [CV.kind(s) if len(s) == 2 else None for s in surfc]
fixed = np.array([(len(s) >= 2 and ck is None) for s, ck in zip(surfc, ckind)]) | np.array([bnode[i] and len(surf.get(i, [])) == 0 for i in used])
cgroups = defaultdict(list)
for k, ck in enumerate(ckind):
    if ck is not None:
        cgroups[ck].append(k)
cgroups = {g: np.array(v) for g, v in cgroups.items()}
print("curve nodes sliding:", sum(len(v) for v in cgroups.values()), "fixed:", int(fixed.sum()), flush=True)
slide = np.array([len(s) == 1 for s in surfc])
groups = defaultdict(list)
for k, s in enumerate(surfc):
    if len(s) == 1:
        groups[s[0]].append(k)
groups = {g: np.array(v) for g, v in groups.items()}


def normals(Q):
    Nn = np.zeros_like(Q)
    for g, idx in groups.items():
        if g[0] == "t":
            T = TUBES[g[1]]; _, sp = T.dist(Q[idx]); v = Q[idx] - T.spine(sp)
            Nn[idx] = v / np.linalg.norm(v, axis=1, keepdims=True)
        elif g[0] == "w":
            Nn[idx, WALLS[g[1]][0]] = 1.0
        else:
            Nn[idx, 1] = 1.0
    return Nn


movable = ~fixed


def project(d, Q):
    d = np.where(movable[:, None], d, 0.0)
    Nn = normals(Q)
    d = d - np.einsum("ij,ij->i", d, Nn)[:, None] * Nn
    for g, idx in cgroups.items():
        t = CV.tangent(g, Q[idx])
        d[idx] = np.einsum("ij,ij->i", d[idx], t)[:, None] * t
    return d


def snap(Q):
    Q = Q.copy()
    for g, idx in groups.items():
        if g[0] == "t":
            T = TUBES[g[1]]; _, sp = T.dist(Q[idx]); c = T.spine(sp); v = Q[idx] - c
            Q[idx] = c + R * v / np.linalg.norm(v, axis=1, keepdims=True)
        elif g[0] == "w":
            Q[idx, WALLS[g[1]][0]] = WALLS[g[1]][1]
        else:
            Q[idx, 1] = PLATE
    for g, idx in cgroups.items():
        Q[idx] = CV.snap(g, Q[idx])
    return Q


print("before volume:", quality(Xc, Hc), flush=True)
if VIT > 0:
    ell = vertex_length(Xc, Hc)[:, None]
    res = descent.minimize(
        Xc, evaluate=lambda Q: _eval(Q, Hc, KAPPA, EPS, DELTA, None),
        gradient=lambda Q: gradient(Q, Hc, kappa=KAPPA, eps=EPS, delta=DELTA),
        project=project, snap=snap, summary=lambda Q: (0, 0.0, 0.0), ell=ell, movable=movable,
        tangent=slide | np.array([ck is not None for ck in ckind]), n_used=len(Hc), iters=VIT, alpha_max=0.5, tol_rel=1e-7)
    Xn = snap(res.P)
    print("volume CG: %d its (%s, %s)" % (res.done, res.stopped, res.reason), flush=True)
    qa = quality(Xn, Hc)
    if qa.n_inverted == 0:
        Xc = Xn
    print("after volume:", qa, flush=True)
sj = scaled_jacobian(Xc[Hc])
np.savez(src + "_g.npz", X=Xc, H=Hc, sj=sj, part=D["part"], gov=D["gov"])
import shutil
shutil.copy(src + "_tags.pkl", src + "_g_tags.pkl")
