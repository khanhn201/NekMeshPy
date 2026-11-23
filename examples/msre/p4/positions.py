"""Positions of the global bottom section at each level -> pos_<level>.npz (checkpoints).
usage: python positions.py [level ...]   (default: all)"""
import os, sys, pickle, collections, time
import numpy as np
from q2 import corner_sj
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import levels as lv
from levels import bsec, rules, secA, PITCH, R_W
import bores
from nekmeshpy.quadmesh.quadmesh import QuadMesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth

BAND = 35.0
T = pickle.load(open(os.path.join(HERE, "topo.pkl"), "rb"))
QUADS, INNER, OUTER, CELLS = T["quads"], T["inner"], T["outer"], T["cells"]
KEYS = {}
for q, _, _ in QUADS:
    for k in q:
        KEYS.setdefault(k, len(KEYS))
INV = [None] * len(KEYS)
for k, i in KEYS.items():
    INV[i] = k
Q = np.array([[KEYS[k] for k in q] for q, _, _ in QUADS])
CORE_XY = {secA.ckey(p): tuple(p) for p in secA.CORE.X}
BK = np.array((1, 1), float) * PITCH
_core = secA.CORE
BK_KEYS = {secA.ckey(_core.X[i]) for q, l in zip(_core.quads, _core.lab) if l.startswith("bk_") or
           (l == "thim" and np.linalg.norm(np.mean([_core.X[j] for j in q], 0) - BK) < 40) for i in q}
PW = collections.defaultdict(list)
for q, _, _ in QUADS:
    if isinstance(q[3], tuple) and q[3][0] == "pw" and not (isinstance(q[0], tuple) and q[0][0] == "pw"):
        PW[q[3][1]].append((q[3][2], q[0]))


def cell_of(k):
    if isinstance(k[0], (float, np.floating)):
        return lv.ROLE[k][1] if k in lv.ROLE else None
    if isinstance(k[0], str) and k[0] in ("ring", "sqb", "sqi", "mj", "mq"):
        return k[1]
    return None


def raw(L, pool):
    cache = {}
    X = np.zeros((len(INV), 2))
    for i, k in enumerate(INV):
        if not (isinstance(k, tuple) and k[0] == "pw"):
            X[i] = lv.coords(k, L, INNER, cache, CORE_XY)
    sig = {c: lv.tile_signature(c, L, CORE_XY) for c in CELLS}
    rep = {}
    for c, s in sig.items():
        rep.setdefault(s, c)
    t0 = time.time()
    res = pool.map(lv.smooth_tile, [(c, L, CORE_XY) for c in rep.values()], chunksize=1)
    tiles = {s: d for s, d in zip(rep.keys(), res)}
    print("  %s: %d tiles for %d cells (%.0fs)" % (L[0], len(tiles), len(CELLS), time.time() - t0), flush=True)
    for i, k in enumerate(INV):
        c = cell_of(k)
        if c is None or c not in sig:
            continue
        dv = tiles[sig[c]].get(lv.rel_key(k, c))
        if dv is not None:
            X[i] += dv
    return X


def basket(X, z):
    f = bores.basket_xy(BK)
    ids = [i for i, k in enumerate(INV) if k in BK_KEYS]
    X[ids] = f(np.array([INV[i] for i in ids], float), z)


def pillows(X):
    for qi, lst in PW.items():
        P = np.array([X[KEYS[c0]] for _, c0 in sorted(lst)]); cen = P.mean(0)
        for j, c0 in sorted(lst):
            X[KEYS[("pw", qi, j)]] = X[KEYS[c0]] + 0.5 * (cen - X[KEYS[c0]])


def wall(X, L):
    """snap the outline onto R692.15 (pins / posts crossing the wall keep their circle up to the
    intersection), then smooth the wall band with the toolkit smoother on a submesh."""
    X = X.copy()
    n = len(OUTER)
    oid = [KEYS[k] for k in OUTER]
    frozen = set()
    anchors = {}
    ring_wall = collections.defaultdict(list)
    for j, k in enumerate(OUTER):
        if isinstance(k, tuple) and k[0] == "ring" and k[1] in lv.SITES:
            cc = np.array(k[1], float) * PITCH
            rr = lv.ring_radius(k[1], L)
            if np.linalg.norm(cc) + rr > R_W:
                ring_wall[k[1]].append(j)
    for c, js in ring_wall.items():
        cc = np.array(c, float) * PITCH; rr = lv.ring_radius(c, L)
        d = np.linalg.norm(cc); a = (R_W ** 2 - rr ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(max(R_W ** 2 - a ** 2, 0)); u = cc / d; nr = np.array([-u[1], u[0]])
        cand = [a * u - h * nr, a * u + h * nr]
        for j in js:                              # junction = outline ring node next to an inner one
            for jn in ((j - 1) % n, (j + 1) % n):
                kn = OUTER[jn]
                if not (isinstance(kn, tuple) and kn[0] == "ring" and kn[1] == c):
                    i = oid[j]
                    pass                          # (no pin/wall anchors: arc length only)
    # place the outline on the wall: radial angles, made strictly increasing and diffused a little
    # so edges that point along a radius do not collapse
    P0 = X[oid]
    th = np.unwrap(np.arctan2(P0[:, 1], P0[:, 0]))
    for _ in range(40):
        ext = np.concatenate([[th[-1] - 2 * np.pi], th, [th[0] + 2 * np.pi]])
        th = 0.5 * th + 0.25 * (ext[:-2] + ext[2:])
    for j in range(n):
        if L[2] == "top" and isinstance(OUTER[j][0], (float, np.floating)):
            continue
        X[oid[j]] = R_W * np.array([np.cos(th[j]), np.sin(th[j])])
    pillows(X)
    r = np.hypot(*X.T)
    free = r > R_W - BAND
    for i, k in enumerate(INV):
        if isinstance(k[0], (float, np.floating)) or (isinstance(k[0], str) and k[0] == "X"):
            free[i] = False
    for i in frozen:
        free[i] = False
    qs = np.where(free[Q].any(1))[0]
    sub = Q[qs]
    nodes = np.unique(sub)
    loc = -np.ones(len(X), int); loc[nodes] = np.arange(len(nodes))
    SQ = loc[sub]
    P = X[nodes]
    if os.environ.get("DBG"):
        e_ = np.concatenate([SQ[:, [0, 1]], SQ[:, [1, 2]], SQ[:, [2, 3]], SQ[:, [3, 0]]])
        ln = np.linalg.norm(P[e_[:, 0]] - P[e_[:, 1]], axis=1)
        a_ = 0.5 * np.sum(P[SQ][..., 0] * np.roll(P[SQ][..., 1], -1, 1) - np.roll(P[SQ][..., 0], -1, 1) * P[SQ][..., 1], 1)
        print("  submesh", len(SQ), "quads; zero edges", (ln < 1e-9).sum(), "neg area", (a_ <= 0).sum(), "nan", np.isnan(P).sum())
        dup = collections.Counter(map(tuple, np.round(P, 9)))
        print("  duplicate positions", sum(v > 1 for v in dup.values()))
        np.savez("dbg_sub.npz", P=P, SQ=SQ)
    qm = QuadMesh.from_corners(np.c_[P, np.zeros(len(P))], SQ)
    fz = [loc[i] for i in nodes if not free[i]]
    cons = [cn.Frozen(np.array(sorted(fz)))]
    ow = [loc[i] for i in oid if free[i]]
    cons.append(cn.OnCurve(np.array(ow), cn.circle(np.zeros(3), R_W)))
    rings = collections.defaultdict(list)
    lines = collections.defaultdict(list)
    for i in nodes:
        if not free[i] or i in set(oid):
            continue
        k = INV[i]
        if k[0] == "ring":
            rings[k[1]].append(loc[i])
        elif k[0] == "mq":
            jn = bsec.side_lines(k[1], k[2])[1]
            lines[(jn[0], round(lv.line_pos(jn, L), 6))].append(loc[i])
    for c, li in rings.items():
        cc = np.array(c, float) * PITCH
        cons.append(cn.OnCurve(np.array(li), cn.circle(np.array([cc[0], cc[1], 0.0]), lv.ring_radius(c, L))))
    for (ax, v), li in lines.items():
        o = np.array([v, 0.0, 0.0]) if ax == "x" else np.array([0.0, v, 0.0])
        dd = np.array([0.0, 1.0, 0.0]) if ax == "x" else np.array([1.0, 0.0, 0.0])
        cons.append(cn.OnCurve(np.array(li), cn.line(o, dd)))
    res = cn.resolve(qm, cons, boundary="frozen")
    nrm = np.tile([0.0, 0.0, 1.0], (len(SQ), 1))
    out, _ = smooth.run_schedule(qm, res, smooth.untangle_schedule(), normals=nrm)
    out, _ = smooth.run_schedule(out, res, smooth.user_schedule(outer=8), normals=nrm)
    X[nodes] = np.asarray(out.points)[:, :2]
    return X


def repair(X, L, grow=3, thr=0.0, slide=False, keep=()):
    """untangle the neighbourhood of any inverted quad (toolkit smoother; rings stay on their
    circles, coordinate keys / X stay frozen, the outline stays on the wall)."""
    bad = np.where(corner_sj(X, Q) <= thr)[0]
    if not len(bad):
        return X
    X = X.copy()
    inc = collections.defaultdict(list)
    for qi, q in enumerate(Q):
        for v in q:
            inc[v].append(qi)
    zone = set(bad.tolist())
    for _ in range(grow):
        zone |= {qj for qi in list(zone) for v in Q[qi] for qj in inc[v]}
    qs = np.array(sorted(zone))
    sub = Q[qs]
    nodes = np.unique(sub)
    loc = -np.ones(len(X), int); loc[nodes] = np.arange(len(nodes))
    SQ = loc[sub]
    qm = QuadMesh.from_corners(np.c_[X[nodes], np.zeros(len(nodes))], SQ)
    oset = set(KEYS[k] for k in OUTER)
    fz, rings, wallv = [], collections.defaultdict(list), []
    hl = collections.defaultdict(list)
    inner = set(np.unique(Q[bad]).tolist())
    for qi in bad:
        for v in Q[qi]:
            inner |= {w for qj in inc[v] for w in Q[qj]}
    keep = set(keep)
    for i in nodes:
        k = INV[i]
        if i in keep:
            fz.append(loc[i]); continue
        if slide and isinstance(k[0], (float, np.floating)):
            pass                                   # level-static node next to a fold: free
        elif isinstance(k[0], (float, np.floating)) or (isinstance(k[0], str) and k[0] == "X"):
            fz.append(loc[i])
        elif i in oset:
            wallv.append(loc[i])
        elif k[0] == "ring":
            rings[k[1]].append(loc[i])
    cons = [cn.Frozen(np.array(sorted(fz), int))]
    if wallv:
        cons.append(cn.OnCurve(np.array(wallv), cn.circle(np.zeros(3), R_W)))
    for yv, li in hl.items():
        cons.append(cn.OnCurve(np.array(li), cn.line(np.array([0.0, yv, 0.0]), np.array([1.0, 0.0, 0.0]))))
    for c, li in rings.items():
        cc = np.array(c, float) * PITCH
        cons.append(cn.OnCurve(np.array(li), cn.circle(np.array([cc[0], cc[1], 0.0]), lv.ring_radius(c, L))))
    res = cn.resolve(qm, cons, boundary="frozen")
    nrm = np.tile([0.0, 0.0, 1.0], (len(SQ), 1))
    out, _ = smooth.run_schedule(qm, res, smooth.untangle_schedule(), normals=nrm)
    out, _ = smooth.run_schedule(out, res, smooth.user_schedule(outer=4), normals=nrm)
    X[nodes] = np.asarray(out.points)[:, :2]
    print("  repair: %d bad quads, zone %d" % (len(bad), len(qs)), flush=True)
    return X


AB_PRISM = {(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1),
            (12, 6), (6, 12), (-12, 6), (-6, 12), (12, -6), (6, -12), (-12, -6), (-6, -12)}


def level_positions(name, pool):
    L = lv.LV[name]
    t0 = time.time()
    X = raw(L, pool)
    z = L[1] if L[1] is not None else 206.5
    basket(X, z)
    pillows(X)
    X = wall(X, L)
    if not os.environ.get("NOREPAIR"):
        X = repair(X, L, grow=3, thr=0.1)
    def score(Y):
        s_ = corner_sj(Y, Q)
        return ((s_ <= 0.02).sum(), -s_.min())
    for g, sl in ((2, False), (2, True), (3, True), (1, True), (5, True), (8, True)):
        if score(X)[0] == 0 or os.environ.get("NOREPAIR"):
            break
        Y = repair(X, L, grow=g, thr=0.02, slide=sl)
        if score(Y) < score(X):
            X = Y
    a = 0.5 * np.sum(X[Q][..., 0] * np.roll(X[Q][..., 1], -1, 1) - np.roll(X[Q][..., 0], -1, 1) * X[Q][..., 1], 1)
    if name == "AB":                               # band A: prism O-grids where the top arcs fold
        XT = np.load(os.path.join(HERE, "pos_top.npz"))["X"]
        for i, k in enumerate(INV):
            if isinstance(k, tuple) and k[0] in ("ring", "sqb", "sqi") and k[1] in AB_PRISM:
                X[i] = XT[i]
    if corner_sj(X, Q).min() <= 0.05:
        import ab_fix
        X = ab_fix.fix(X, L, AB_PRISM if name == "AB" else set())
    sj = corner_sj(X, Q)
    print("  %s done (%.0fs): min area %.3g, neg %d, min corner sj %.3f, n<=0.02 %d" % (name, time.time() - t0, a.min(), (a <= 0).sum(), sj.min(), (sj <= 0.02).sum()), flush=True)
    np.savez(os.path.join(HERE, "pos_%s.npz" % name), X=X)
    return X


if __name__ == "__main__":
    names = sys.argv[1:] or [L[0] for L in lv.LEVELS]
    with Pool(int(os.environ.get("NPROC", "60"))) as pool:
        for nm in names:
            level_positions(nm, pool)
