"""Lower head zone (L2): global section + radial annulus out to the vessel, the 48 vanes morphed in
(selection-first, whole channels, max-2 rule, pillowed single-quad turns), surfaces, and the
2D section/positions for L2a (vanes) / L2b (morph back) / transition.  -> head_topo.pkl"""
import os, sys, pickle, collections, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import levels as lv
from levels import bsec, rules, secA, PITCH, R_W
import positions as PS

R_CAN, R_VES = 711.2, 738.4990768478899
R_MID = 0.5 * (R_CAN + R_VES)
ROWS = [R_W, R_CAN, R_MID, R_VES]
R0V = 459.1                      # vane inner edge
VHW = 6.35                       # vane half thickness
NV = 48


# ------------------------------------------------------------------ surfaces --------------
def head_z(r):
    """dish / knuckle 1 / knuckle 2 (exact circle fits)."""
    r = np.asarray(r, float)
    z = 1401.56 - np.sqrt(np.maximum(1376.159 ** 2 - (r + 3.03) ** 2, 0))
    z = np.where(r > 652.95, 404.44 - np.sqrt(np.maximum(232.739 ** 2 - (r - 558.34) ** 2, 0)), z)
    z = np.where(r > 708.34, 298.99 - np.sqrt(np.maximum(102.215 ** 2 - (r - 636.28) ** 2, 0)), z)
    return z


def s2(r):
    return lv.gz(r)                          # grid bottom (= 298.99 beyond the cone)


def s0(r):
    return np.minimum(head_z(r), s2(r) - 6.0)


def s1(r):
    r = np.asarray(r, float)
    tau = (216.88 - head_z(R0V)) / (s2(R0V) - head_z(R0V))
    inner = s0(r) + tau * (s2(r) - s0(r))
    vt = np.minimum(216.88 + 0.2549 * (r - R0V), 277.94)
    z = np.where(r >= R0V, vt, inner)
    return np.clip(z, s0(r) + 1.5, s2(r) - 4.5)


# ------------------------------------------------------------------ section ---------------
NPLAIN = 2                       # 1088 = 3*362 + 2: 362 3:1 templates + 2 plain cells -> 364 columns


def transition_plan(n):
    """per template: number of inner edges (3 or 1); spread the plain ones evenly."""
    npl = n % 3 if n % 3 else 3                     # plain cells so the rest divides by 3 (1088 -> 2)
    k = (n - npl) // 3
    assert 3 * k + npl == n
    plan = [3] * k
    for i in range(npl):
        plan.insert(int((i + 0.5) * len(plan) / npl) + i, 1)
    return plan


def build_section():
    quads = list(PS.QUADS)
    outer = PS.OUTER
    n = len(outer)
    plan = transition_plan(n)
    m = len(plan)
    j = 0
    for t, w in enumerate(plan):
        b0, b1 = ("ann", 1, t), ("ann", 1, (t + 1) % m)
        a = [outer[(j + i) % n] for i in range(w + 1)]
        if w == 1:
            quads.append(((a[0], b0, b1, a[1]), "ann0", ("ann",)))
        else:
            m1, m2 = ("tm", t, 1), ("tm", t, 2)
            quads.append(((a[0], b0, m1, a[1]), "ann0", ("ann",)))
            quads.append(((a[1], m1, m2, a[2]), "ann0", ("ann",)))
            quads.append(((a[2], m2, b1, a[3]), "ann0", ("ann",)))
            quads.append(((m1, b0, b1, m2), "ann0", ("ann",)))
        j += w
    for row in (1, 2):
        a = [("ann", row, t) for t in range(m)]
        b = [("ann", row + 1, t) for t in range(m)]
        for t in range(m):
            k = (t + 1) % m
            quads.append(((a[t], b[t], b[k], a[k]), "ann%d" % row, ("ann",)))
    return quads


def positions_kn(quads, XKN):
    keys = {}
    for q, _, _ in quads:
        for k in q:
            keys.setdefault(k, len(keys))
    X = np.zeros((len(keys), 2))
    for k, i in keys.items():
        if k in PS.KEYS:
            X[i] = XKN[PS.KEYS[k]]
    n = len(PS.OUTER)
    th = np.unwrap(np.array([np.arctan2(*XKN[PS.KEYS[k]][::-1]) for k in PS.OUTER]))
    plan = transition_plan(n)
    tb, j = [], 0
    for t, w in enumerate(plan):
        tb.append(th[j]); j += w
    tb = np.array(tb)
    # outer columns: start at each template's first inner node, then evened out a little
    j = 0
    for t, w in enumerate(plan):
        if w == 3:
            for i in (1, 2):
                X[keys[("tm", t, i)]] = 0.5 * (R_W + R_CAN) * np.array([np.cos(th[j + i]), np.sin(th[j + i])])
        j += w
    for k, i in keys.items():
        if isinstance(k, tuple) and k[0] == "ann":
            X[i] = ROWS[k[1]] * np.array([np.cos(tb[k[2]]), np.sin(tb[k[2]])])
    # the template nodes are free (the head's top face between the bands (R_W) and the downcomer (R_CAN) is a
    # wall): relax them to the middle of their neighbours, so no template quad is a sliver when two wall nodes
    # sit close together
    tm = [i for k, i in keys.items() if isinstance(k, tuple) and k[0] == "tm"]
    nb = {i: set() for i in tm}
    for q, _, _ in quads:
        ids = [keys[k] for k in q]
        for a_ in range(4):
            u_, v_ = ids[a_], ids[(a_ + 1) % 4]
            if u_ in nb:
                nb[u_].add(v_)
            if v_ in nb:
                nb[v_].add(u_)
    EF = float(os.environ.get("TM_EMIN", "1.0"))          # no template edge shorter than this

    def edge_ok(i, p):
        return np.linalg.norm(X[list(nb[i])] - p, axis=1).min() >= EF
    for _ in range(int(os.environ.get("TM_RELAX", "50"))):
        for i in tm:
            p = 0.5 * X[i] + 0.5 * X[list(nb[i])].mean(0)
            if edge_ok(i, p):
                X[i] = p
    # then per node: the spot (of a few candidates) that maximises the worst corner sj of its quads
    from q2 import corner_sj
    Q = np.array([[keys[k] for k in q] for q, _, _ in quads])
    inc = {i: [] for i in tm}
    for qi, row in enumerate(Q):
        for i in row:
            if i in inc:
                inc[i].append(qi)
    rng = np.random.default_rng(0)
    for _ in range(30):
        moved = 0
        for i in tm:
            qs = Q[inc[i]]
            cur = corner_sj(X, qs).min()
            if cur > 0.3:
                continue
            old = X[i].copy(); best, bp = cur, None
            h = 0.3 * np.mean([np.linalg.norm(X[j] - old) for j in nb[i]])
            for p in [old + h * f * rng.normal(size=2) for f in (0.1, 0.3, 0.6, 1.0) for _ in range(10)]:
                if not edge_ok(i, p):
                    continue
                X[i] = p
                v = corner_sj(X, qs).min()
                if v > best + 1e-6:
                    best, bp = v, p.copy()
            X[i] = bp if bp is not None else old
            moved += bp is not None
        if not moved:
            break
    return keys, X


if __name__ == "__main__":
    quads = build_section()
    XKN = np.load(os.path.join(HERE, "pos_KN.npz"))["X"]
    keys, X = positions_kn(quads, XKN)
    print("L2 section: %d quads, %d nodes" % (len(quads), len(keys)))
    pickle.dump(dict(quads=quads, keys=keys, X_kn=X), open(os.path.join(HERE, "head_sec.pkl"), "wb"))
