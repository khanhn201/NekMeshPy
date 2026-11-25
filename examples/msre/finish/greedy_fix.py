"""Twist-aware greedy polish of a linear model: around clusters of hexes with corner sj < SJ, interior
(non-boundary) nodes move to the best of a set of candidates; the objective per node is lexicographic
(min face-turn agreement capped at FT, then min corner sj) over its incident hexes, so no face is ever
twisted further and the worst Jacobian only rises.  Zones run in parallel.  Boundary nodes never move,
so the walls and every seam stay exactly where they are.
usage: greedy_fix.py in.pkl out.pkl   env SJ RINGS ROUNDS NPROC FT SKIP_VANE BOX"""
import os, sys, pickle, time
import numpy as np
from multiprocessing import Pool
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from nekmeshpy import HexMesh, hexmesh
from nekmeshpy.hexmesh import smoother as hs
from nekmeshpy.hexmesh.quality import corner_scaled_jacobian

SJ = float(os.environ.get("SJ", "0.15")); RINGS = int(os.environ.get("RINGS", "2"))
ROUNDS = int(os.environ.get("ROUNDS", "30")); FT = float(os.environ.get("FT", "0.3"))
EFRAC = float(os.environ.get("EFRAC", "0.7"))          # no edge at a moved node below this fraction of its original shortest
E12 = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
FP = HexMesh.FACE_POINTS
t0 = time.time()
m = pickle.load(open(sys.argv[1], "rb"))
P = np.asarray(m.points, float).copy(); H = np.asarray(m.corners)


def csj(Pt, hx):
    return np.asarray(corner_scaled_jacobian(Pt, hx))


def face_turn(Pt, hx):
    q = Pt[hx[:, FP]]
    turn = np.cross(np.roll(q, -1, axis=2) - q, np.roll(q, 1, axis=2) - q)
    mean = turn.sum(axis=2, keepdims=True)
    ag = np.einsum("efkc,efkc->efk", turn, np.broadcast_to(mean, turn.shape))
    ag = ag / (np.linalg.norm(turn, axis=3) * np.linalg.norm(mean, axis=3) + 1e-300)
    return ag.min(axis=(1, 2))


sj = csj(P, H); cen = P[H].mean(1)
bad = sj < SJ
if os.environ.get("BOX"):
    b = [float(x) for x in os.environ["BOX"].split(",")]
    bad &= (cen[:, 0] > b[0]) & (cen[:, 0] < b[1]) & (cen[:, 1] > b[2]) & (cen[:, 1] < b[3]) & (cen[:, 2] > b[4]) & (cen[:, 2] < b[5])
if os.environ.get("SKIP_VANE"):
    r = np.hypot(cen[:, 0], cen[:, 1]); bad &= ~((cen[:, 2] < 300) & (r > 455) & (r < 745))
E = len(H)
A = coo_matrix((np.ones(8 * E), (np.repeat(np.arange(E), 8), H.ravel())), shape=(E, len(P))).tocsr()
zone = bad.copy()
for _ in range(RINGS):
    pts = np.asarray(A[zone].sum(0)).ravel() > 0
    zone = np.asarray(A[:, pts].sum(1)).ravel() > 0
zid = np.where(zone)[0]
Az = A[zid]; G = (Az @ Az.T).tocoo()
nc, lab = connected_components(coo_matrix((np.ones(len(G.row)), (G.row, G.col)), shape=(len(zid), len(zid))), directed=False)
# big zones are split into blocks of <= ZMAXH hexes (recursive median bisection, xy rotated by ROT degrees so a
# second pass cuts elsewhere); a block only moves nodes whose every hex lies in it, so blocks never collide
ZMAXH = int(os.environ.get("ZMAXH", "1500")); ROT = np.radians(float(os.environ.get("ROT", "0")))
_cz = P[H[zid]].mean(1)
_cz = np.c_[_cz[:, 0] * np.cos(ROT) - _cz[:, 1] * np.sin(ROT), _cz[:, 0] * np.sin(ROT) + _cz[:, 1] * np.cos(ROT), _cz[:, 2]]


def _bisect(idx):
    if len(idx) <= ZMAXH:
        return [idx]
    ax = int(np.argmax(np.ptp(_cz[idx], axis=0)))
    med = np.median(_cz[idx, ax])
    lo, hi = idx[_cz[idx, ax] <= med], idx[_cz[idx, ax] > med]
    if not len(lo) or not len(hi):
        return [idx]
    return _bisect(lo) + _bisect(hi)


_new = -np.ones(len(zid), int); _nb = 0
for c_ in range(nc):
    for part in _bisect(np.where(lab == c_)[0]):
        _new[part] = _nb; _nb += 1
nc, lab = _nb, _new
fk = np.sort(H[:, FP].reshape(-1, 4), axis=1)
_, fi, fcnt = np.unique(fk, axis=0, return_inverse=True, return_counts=True)
bnd = np.zeros(len(P), bool); bnd[H[:, FP].reshape(-1, 4)[fcnt[fi] == 1].ravel()] = True
# flat-wall sliding: a boundary node whose boundary faces are all coplanar (and carry one tag) moves in that plane
PLANE = {}
if os.environ.get("SLIDE", "1") == "1":
    BF = H[:, FP].reshape(-1, 4)[fcnt[fi] == 1]
    q = P[BF]
    nrm = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300
    from collections import defaultdict
    nb_ = defaultdict(list)
    for f, row in enumerate(BF):
        for v in row:
            nb_[v].append(f)
    cosmax = np.cos(np.radians(float(os.environ.get("FLAT_DEG", "0.5"))))
    for v, fs in nb_.items():
        nn = nrm[fs]; mu = nn.mean(0); mu /= np.linalg.norm(mu)
        if (nn @ mu).min() > cosmax:
            # the plane must contain every corner of those faces (truly flat, not just parallel)
            if np.abs((P[BF[fs]].reshape(-1, 3) - P[v]) @ mu).max() < 1e-6:
                PLANE[v] = mu
print("%d hexes, %d below sj %.2f (min %.3f); %d zones, %d hexes (%.0fs)" % (E, bad.sum(), SJ, sj.min(), nc, len(zid), time.time() - t0), flush=True)
Acsc = A.tocsc()
from scipy.spatial import cKDTree
from nekmeshpy.core import topology as T, conform
CEN0 = P[H].mean(1); RAD0 = np.linalg.norm(P[H] - CEN0[:, None], axis=2).max(1)
CTREE = cKDTree(CEN0); TOL = 1e-6 * conform.bbox_scale(P)


def ovl_pairs(Pt, ids):
    """overlapping pairs (a, b) with a in ids and b any hex near it (radius broad phase)."""
    R = 2.0 * RAD0[ids].max() + 1e-9
    near = np.unique(np.concatenate(CTREE.query_ball_point(CEN0[ids], R)))
    I = np.repeat(ids, len(near)); J = np.tile(near, len(ids)); k = I != J; I, J = I[k], J[k]
    ci, cj = Pt[H[I]].mean(1), Pt[H[J]].mean(1)
    ri = np.linalg.norm(Pt[H[I]] - ci[:, None], axis=2).max(1); rj = np.linalg.norm(Pt[H[J]] - cj[:, None], axis=2).max(1)
    k = np.linalg.norm(ci - cj, axis=1) < ri + rj; I, J = I[k], J[k]
    out = set()
    for lo in range(0, len(I), 100_000):
        a, b = I[lo:lo + 100_000], J[lo:lo + 100_000]
        _, _, va, na = T._hex_geometry(Pt, H[a]); _, _, vb, nb = T._hex_geometry(Pt, H[b])
        hit = T._points_inside(T._hex_sample_points(Pt, H[a]), vb, nb, TOL) | T._points_inside(T._hex_sample_points(Pt, H[b]), va, na, TOL)
        out |= {(min(x, y), max(x, y)) for x, y in zip(a[hit].tolist(), b[hit].tolist())}
    return out


def run(k):
    rng = np.random.default_rng(k)
    ids = zid[lab == k]
    nodes = np.unique(H[ids]); nodes = nodes[~bnd[nodes] | np.isin(nodes, list(PLANE.keys()))]
    idset = np.zeros(E, bool); idset[ids] = True
    nodes = np.array([v for v in nodes if idset[Acsc[:, v].indices].all()], dtype=np.int64)   # own nodes only
    if not len(nodes):
        return k, None, "no own nodes"
    Pt = P.copy()
    inc = {v: Acsc[:, v].indices for v in nodes}
    nbr = {}
    for v in nodes:
        hh = H[inc[v]]; s_ = set()
        for row in hh:
            j = int(np.where(row == v)[0][0])
            for a, b in E12:
                if a == j: s_.add(int(row[b]))
                elif b == j: s_.add(int(row[a]))
        nbr[v] = np.array(sorted(s_))

    def score(hx):
        return (min(face_turn(Pt, hx).min(), FT), csj(Pt, hx).min())
    L0 = {v: np.linalg.norm(P[nbr[v]] - P[v], axis=1).min() for v in nodes}
    s0 = csj(P, H[ids]).min()
    for rd in range(ROUNDS):
        moved = 0
        cur_sj = csj(Pt, H[ids])
        order = sorted(nodes, key=lambda v: cur_sj[np.isin(ids, inc[v])].min() if np.isin(ids, inc[v]).any() else 1.0)
        for v in order:
            hx = H[inc[v]]
            cur = score(hx); old = Pt[v].copy(); best, bp = cur, None
            c0 = Pt[nbr[v]].mean(0); hl = 0.5 * np.linalg.norm(Pt[nbr[v]] - old, axis=1).mean()
            for p in [c0, 0.5 * (c0 + old)] + [old + hl * f * rng.normal(size=3) for f in (0.03, 0.1, 0.3) for _ in range(8)]:
                if v in PLANE:
                    p = p - ((p - old) @ PLANE[v]) * PLANE[v]
                Pt[v] = p
                if np.linalg.norm(Pt[nbr[v]] - p, axis=1).min() < EFRAC * L0[v]:
                    continue
                sc = score(hx)
                if sc[0] > best[0] + 1e-9 or (sc[0] >= best[0] - 1e-12 and sc[1] > best[1] + 1e-6):
                    best, bp = sc, p.copy()
            Pt[v] = bp if bp is not None else old
            moved += bp is not None
        if not moved:
            break
    mv = nodes[np.linalg.norm(Pt[nodes] - P[nodes], axis=1) > 0]
    chk = np.unique(Acsc[:, mv].indices) if len(mv) else ids       # every hex a moved node touches
    before = ovl_pairs(P, chk)
    for _ in range(6):
        new = ovl_pairs(Pt, chk) - before
        if not new:
            break
        back = np.unique(H[np.array(sorted({h for pr in new for h in pr}))])
        back = back[np.isin(back, nodes)]
        if not len(back):
            break
        Pt[back] = P[back]
    if ovl_pairs(Pt, chk) - before:
        return k, None, "overlap"
    s1 = csj(Pt, H[ids]).min()
    a0, a1 = face_turn(P, H[ids]).min(), face_turn(Pt, H[ids]).min()
    if s1 > s0 + 1e-3 and a1 >= min(a0, FT) - 1e-9:
        return k, (nodes, Pt[nodes]), "%.3f -> %.3f, face turn %.2f -> %.2f (%d hexes, %d nodes)" % (s0, s1, a0, a1, len(ids), len(nodes))
    return k, None, "no gain %.3f -> %.3f (face turn %.2f -> %.2f)" % (s0, s1, a0, a1)


with Pool(int(os.environ.get("NPROC", "32"))) as pool:
    outs = pool.map(run, range(nc), chunksize=1)
kept = 0
for k, r_, msg in outs:
    c = cen[zid[lab == k]].mean(0)
    print("  zone %d at r %.0f th %.0f z %.0f: %s" % (k, np.hypot(*c[:2]), np.degrees(np.arctan2(c[1], c[0])), c[2], msg), flush=True)
    if r_ is not None:
        P[r_[0]] = r_[1]; kept += 1
sj2 = csj(P, H)
v = hexmesh.validity(hs.with_points(m, P))
print("kept %d of %d zones; below sj %.2f: %d -> %d, min %.3f -> %.3f; inverted %d twisted %d (%.0fs)" % (
    kept, nc, SJ, (sj < SJ).sum(), (sj2 < SJ).sum(), sj.min(), sj2.min(), v.inverted.size, v.twisted.size, time.time() - t0), flush=True)
pickle.dump(hs.with_points(m, P), open(sys.argv[2], "wb"))
