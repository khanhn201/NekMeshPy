"""Overlap repair for the vskin lower head: move free nodes of overlapping hex pairs (toolkit's
half-space overlap test) to the candidate that first removes overlap, then keeps the twist-aware score.
Same node classes as vskin_twist (vane skin nodes stay on the rounded vane, shell nodes on the shell,
other boundary nodes fixed).  IN / OUT env."""
import os, sys, pickle, time, collections
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("STOP2D", "1")
from nekmeshpy import HexMesh, hexmesh
from nekmeshpy.core import topology as T, conform
from nekmeshpy.hexmesh import smoother as hs
import head as HD
import vskin_twist_lib as VT

t0 = time.time()
m = pickle.load(open(os.environ.get("IN", "head_L2_skin_lin3.pkl"), "rb"))
P = np.asarray(m.points).copy(); H = np.asarray(m.corners)
free, proj = VT.node_classes(P, H)
TOL = 1e-6 * conform.bbox_scale(P)
TGT = float(os.environ.get("TGT", "0.05"))
HMIN = float(os.environ.get("HMIN", "0.5"))            # shortest edge wanted (mm)
EU = np.unique(np.sort(H[:, VT.E12].reshape(-1, 2), 1), axis=0)


def short_nodes(P):
    L = np.linalg.norm(P[EU[:, 0]] - P[EU[:, 1]], axis=1)
    return set(np.unique(EU[L < HMIN]).tolist()), int((L < HMIN).sum()), float(L.min())


def ovl_pairs(P, hx_idx, cand_idx):
    """overlapping pairs (exact trilinear test) between hexes hx_idx and cand_idx, as frozensets."""
    sub = np.unique(np.r_[hx_idx, cand_idx])
    O = T.overlapping_pairs(P, H[sub], workers=1)
    hx = set(np.asarray(hx_idx).tolist())
    return {frozenset((int(sub[a]), int(sub[b]))) for a, b in O if int(sub[a]) in hx or int(sub[b]) in hx}


_c0 = P[H].mean(1)
ZONE = np.where((np.hypot(_c0[:, 0], _c0[:, 1]) > float(os.environ.get("RMIN", "430"))) & (_c0[:, 2] < float(os.environ.get("ZMAX", "320"))))[0]


def all_pairs(P):
    """overlapping pairs inside ZONE (the toolkit's exact, parallel search)."""
    O = T.overlapping_pairs(P, H[ZONE])
    return {frozenset((int(ZONE[a]), int(ZONE[b]))) for a, b in O}


inc = collections.defaultdict(list); nbr = collections.defaultdict(set)
for e, q in enumerate(H):
    for a in range(8):
        inc[q[a]].append(e)
for q in H:
    for a, b in VT.E12:
        nbr[q[a]].add(q[b]); nbr[q[b]].add(q[a])
cen = P[H].mean(1); RAD = float(np.linalg.norm(P[H] - cen[:, None], axis=2).max())
tree = cKDTree(cen)
rng = np.random.default_rng(int(os.environ.get("SEED", "0")))
pairs = all_pairs(P)
sn, nshort, lmin = short_nodes(P)
print("start: %d overlapping pairs, %d edges < %.2f (min %.4f) (%.0fs)" % (len(pairs), nshort, HMIN, lmin, time.time() - t0), flush=True)
for rd in range(int(os.environ.get("ROUNDS", "40"))):
    sn, nshort, lmin = short_nodes(P)
    if not pairs and not sn:
        break
    hx = sorted({h for p in pairs for h in p})
    zone = set(np.unique(H[hx]).tolist()) if hx else set()
    zone |= sn
    for _ in range(int(os.environ.get("GROW", "1"))):
        zone |= {w for x in list(zone) for w in nbr[x]}
    for x in sorted(zone):
        if not free[x]:
            continue
        hs_ = np.array(inc[x])
        near = np.array(sorted(set(tree.query_ball_point(P[x], 2.5 * RAD))))
        nb_ = list(nbr[x])

        def obj():
            le = np.linalg.norm(P[nb_] - P[x], axis=1).min()
            return (len(ovl_pairs(P, hs_, near)), -min(VT.hex_score(P, H[hs_]).min(), TGT), -min(le, HMIN))
        cur = obj(); old = P[x].copy(); best, bp = cur, None
        if cur[0] == 0 and -cur[1] >= TGT and -cur[2] >= HMIN:
            continue
        c0 = P[nb_].mean(0)
        hl = 0.5 * np.mean(np.linalg.norm(P[nb_] - P[x], axis=1))
        cands = [c0, 0.5 * (c0 + old)] + [old + hl * f * rng.normal(size=3) for f in (0.05, 0.15, 0.3, 0.6) for _ in range(10)]
        dd = np.linalg.norm(P[nb_] - old, axis=1)
        for w in np.array(nb_)[dd < HMIN]:                  # away from a too-close neighbour
            u = old - P[w]; u = u / max(np.linalg.norm(u), 1e-12)
            if np.linalg.norm(old - P[w]) < 1e-9:
                u = rng.normal(size=3); u /= np.linalg.norm(u)
            cands += [old + f * HMIN * u for f in (0.3, 0.6, 1.0, 1.5)]
        floor = -cur[1]                                    # never push an incident hex below min(its score, TGT)
        for p in cands:
            P[x] = proj(x, p)
            o = obj()
            if -o[1] < floor - 1e-12:
                continue
            if o < best:
                best, bp = o, P[x].copy()
        P[x] = bp if bp is not None else old
    pairs = all_pairs(P) if rd % 5 == 4 else {p for p in pairs if ovl_pairs(P, np.array(sorted(p))[:1], np.array(sorted(p))[1:])} | \
        {pp for h in hx for pp in ovl_pairs(P, np.array([h]), np.array(sorted(set(tree.query_ball_point(P[H[h]].mean(0), 2.5 * RAD)))))}
    sn, nshort, lmin = short_nodes(P)
    print("  round %d: %d overlapping pairs, %d short edges (min %.4f), min score %.3f (%.0fs)" % (rd, len(pairs), nshort, lmin, VT.hex_score(P, H).min(), time.time() - t0), flush=True)
pairs = all_pairs(P)
sn, nshort, lmin = short_nodes(P)
out = hs.with_points(m, P)
v = hexmesh.validity(out)
print("short edges < %.2f: %d (min %.4f)" % (HMIN, nshort, lmin))
print("end: %d overlapping pairs, inverted %d, twisted %d, min score %.3f (%.0fs)" % (
    len(pairs), v.inverted.size, v.twisted.size, VT.hex_score(P, H).min(), time.time() - t0), flush=True)
pickle.dump(out, open(os.environ.get("OUT", "head_L2_skin_lin4") + ".pkl", "wb"))
