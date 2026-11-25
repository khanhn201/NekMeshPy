"""Post-assembly local smoothing of a linear model: clusters of hexes with corner sj < SJ (optionally only
inside a box), each a zone RINGS deep, smoothed with the toolkit (untangle, then shape); wall nodes slide on
the zone's fitted patches, the zone's cut and named-boundary corners are frozen.  A zone is kept only when its
min corner sj improves, nothing inverts or twists, and nothing in it starts to overlap.
usage: region_fix.py in.pkl out.pkl   env SJ RINGS OUTER NPROC BOX="xmin,xmax,ymin,ymax,zmin,zmax" SKIP_VANE"""
import os, sys, pickle, time
import numpy as np
from multiprocessing import Pool
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from nekmeshpy import HexMesh, hexmesh
from nekmeshpy.hexmesh import smoother as hs
from nekmeshpy.core import topology as T, conform

SJ = float(os.environ.get("SJ", "0.15")); RINGS = int(os.environ.get("RINGS", "2"))
t0 = time.time()
m = pickle.load(open(sys.argv[1], "rb"))
P = np.asarray(m.points, float).copy(); H = np.asarray(m.corners)
sj = hs.corner_metrics(P, H).sj.min(axis=1)
cen = P[H].mean(1)
bad = sj < SJ
if os.environ.get("BOX"):
    b = [float(x) for x in os.environ["BOX"].split(",")]
    bad &= (cen[:, 0] > b[0]) & (cen[:, 0] < b[1]) & (cen[:, 1] > b[2]) & (cen[:, 1] < b[3]) & (cen[:, 2] > b[4]) & (cen[:, 2] < b[5])
if os.environ.get("SKIP_VANE"):
    r = np.hypot(cen[:, 0], cen[:, 1]); bad &= ~((cen[:, 2] < 300) & (r > 455) & (r < 745))
print("%d hexes, %d below sj %.2f (min %.3f) (%.0fs)" % (len(H), bad.sum(), SJ, sj.min(), time.time() - t0), flush=True)
# hex adjacency through shared points -> zones grown RINGS deep, then clustered
E = len(H)
rows = np.repeat(np.arange(E), 8); A = coo_matrix((np.ones(8 * E), (rows, H.ravel())), shape=(E, len(P))).tocsr()
zone = bad.copy()
for _ in range(RINGS):
    pts = np.asarray(A[zone].sum(0)).ravel() > 0
    zone = np.asarray(A[:, pts].sum(1)).ravel() > 0
zid = np.where(zone)[0]
Az = A[zid]; G = (Az @ Az.T).tocoo()
nc, lab = connected_components(coo_matrix((np.ones(len(G.row)), (G.row, G.col)), shape=(len(zid), len(zid))), directed=False)
print("%d zones, %d hexes in total" % (nc, len(zid)), flush=True)
# frozen: named-face corner points? keep tags: points on faces whose tag differs from neighbours' -> simply freeze
# points shared by two differently tagged boundary faces (feature lines between BCs)
ftag = m.face_tags.to_dense()
fc = np.asarray(m.quad_mesh.corners); bfi = np.where(np.asarray(hexmesh.boundary_face_ids(m)))[0]
from collections import defaultdict
ptags = defaultdict(set)
for f in bfi:
    for v in fc[f]:
        ptags[v].add(ftag[f])
multi = np.array(sorted(v for v, s in ptags.items() if len(s) > 1), int)
FROZ = np.zeros(len(P), bool); FROZ[multi] = True
TOL = 1e-6 * conform.bbox_scale(P)
OUTER = int(os.environ.get("OUTER", "4"))


from scipy.spatial import cKDTree
CEN0 = P[H].mean(1); RAD0 = np.linalg.norm(P[H] - CEN0[:, None], axis=2).max(1); CTREE = cKDTree(CEN0)


def overlaps(Pt, ids):
    """number of overlapping pairs (a, b): a in ids, b any hex within reach of it (radius broad phase)."""
    nb = np.unique(np.concatenate(CTREE.query_ball_point(CEN0[ids], 2.0 * RAD0[ids].max() + 1e-9)))
    c, r, v, n = T._hex_geometry(Pt, H[nb]); li = {h: i for i, h in enumerate(nb)}
    s = T._hex_sample_points(Pt, H[nb])
    ai = np.array([li[x] for x in ids])
    a = np.repeat(ai, len(nb)); b = np.tile(np.arange(len(nb)), len(ai))
    k = (nb[a] != nb[b]); a, b = a[k], b[k]
    k = np.linalg.norm(c[a] - c[b], axis=1) < r[a] + r[b]; a, b = a[k], b[k]
    out = set()
    for lo in range(0, len(a), 100_000):
        x, y = a[lo:lo + 100_000], b[lo:lo + 100_000]
        hit = T._points_inside(s[x], v[y], n[y], TOL) | T._points_inside(s[y], v[x], n[x], TOL)
        out |= {(min(p, q), max(p, q)) for p, q in zip(nb[x[hit]].tolist(), nb[y[hit]].tolist())}
    return len(out)


def face_turn(Pt, hx):
    """per hex: the worst agreement of a face corner's turn with its face's mean turn (<= 0: a bow-tie face)."""
    q = Pt[hx[:, HexMesh.FACE_POINTS]]
    turn = np.cross(np.roll(q, -1, axis=2) - q, np.roll(q, 1, axis=2) - q)
    mean = turn.sum(axis=2, keepdims=True)
    ag = np.einsum("efkc,efkc->efk", turn, np.broadcast_to(mean, turn.shape))
    ag = ag / (np.linalg.norm(turn, axis=3) * np.linalg.norm(mean, axis=3) + 1e-300)
    return ag.min(axis=(1, 2))


def run(k):
    ids = zid[lab == k]
    sp, si = np.unique(H[ids].ravel(), return_inverse=True)
    sub = HexMesh.from_corners(P[sp], si.reshape(-1, 8))
    out_ = np.ones(len(P), bool); out_[sp] = False
    rest = np.ones(E, bool); rest[ids] = False
    touch = np.zeros(len(P), bool); touch[H[rest].ravel()] = True
    fz = np.where(touch[sp] | FROZ[sp])[0]
    s0 = sj[ids].min(); msgs = []; best = None
    for bmode in ("slide", "frozen"):
        try:
            res = hs.resolve(sub, [hs.Frozen(fz)], boundary=bmode)
            sched = hs.untangle_schedule() + hs.user_schedule(outer=OUTER)
            o, _ = hs.run_schedule(sub, res, sched)
        except Exception as e:
            msgs.append("%s: error %s" % (bmode, e)); continue
        Q = np.asarray(o.points)
        Pt = P.copy(); Pt[sp] = Q
        s1 = hs.corner_metrics(Pt, H[ids]).sj.min()
        a0 = face_turn(P, H[ids]).min(); a1 = face_turn(Pt, H[ids]).min()
        if s1 <= s0 + 1e-3 or a1 <= 0 or a1 < min(a0, 0.5):
            msgs.append("%s: no gain %.3f -> %.3f (face turn %.2f -> %.2f)" % (bmode, s0, s1, a0, a1)); continue
        if overlaps(Pt, ids) > overlaps(P, ids):
            msgs.append("%s: overlap" % bmode); continue
        if best is None or s1 > best[0]:
            best = (s1, Q, bmode)
        if bmode == "slide":
            break
    if best is None:
        return k, None, "; ".join(msgs)
    return k, (sp, best[1]), "%.3f -> %.3f (%d hexes, %s)" % (s0, best[0], len(ids), best[2])


with Pool(int(os.environ.get("NPROC", "32"))) as pool:
    outs = pool.map(run, range(nc), chunksize=1)
kept = 0
for k, r_, msg in outs:
    ids = zid[lab == k]
    c = cen[ids].mean(0)
    print("  zone %d at r %.0f th %.0f z %.0f: %s" % (k, np.hypot(*c[:2]), np.degrees(np.arctan2(c[1], c[0])), c[2], msg), flush=True)
    if r_ is not None:
        P[r_[0]] = r_[1]; kept += 1
sj2 = hs.corner_metrics(P, H).sj.min(axis=1)
v = hexmesh.validity(hs.with_points(m, P))
print("kept %d of %d zones; below sj %.2f: %d -> %d, min %.3f -> %.3f; inverted %d twisted %d (%.0fs)" % (
    kept, nc, SJ, (sj < SJ).sum(), (sj2 < SJ).sum(), sj.min(), sj2.min(), v.inverted.size, v.twisted.size, time.time() - t0), flush=True)
pickle.dump(hs.with_points(m, P), open(sys.argv[2], "wb"))
