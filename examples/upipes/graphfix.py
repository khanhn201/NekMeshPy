"""Approximate-graph fixes in the hub (paper sec. 2: elements smaller than the
tolerance are merged).  -> maps/graph2.pkl

* The vacancies at (+-49.7, +-h) are 4-way degenerate vertical edges (four legs at
  19.90 +- 0.02 mm).  Extraction split them into near-coincident 3-way pieces; here they
  become one 4-governor edge with an explicit cyclic face list, ending at the vertex
  where the structure splits.
* The hub centre (6-way edge + 83's focal point) is replaced by the hub core, so its
  fragments are removed.
"""
import numpy as np, pickle
from graph import gname

E, V = pickle.load(open("maps/graph.pkl", "rb"))
for e in E:
    e.setdefault("faces", None)


def eid(gov, near):
    best = None
    for i, e in enumerate(E):
        if set(e["gov"]) == set(gov) and e.get("alive", True):
            d = min(np.linalg.norm(e["pts"] - np.asarray(near), axis=1))
            if best is None or d < best[0]:
                best = (d, i)
    return best[1]


def vertical_departure(P, X0, tol=0.6):
    """index where polyline P leaves the vertical line through X0 (after starting on it)"""
    d = np.hypot(P[:, 0] - X0[0], P[:, 2] - X0[2])
    k = np.argmax(d > tol)
    return k


def add_vertex(X, gov):
    V.append(dict(X=np.asarray(X, float), ends=[], gov=tuple(sorted(gov)), plate=False))
    return len(V) - 1


def fix_vacancy(base, govs, faces, kill, cut):
    """base: plate point of the vacancy line; kill: edges lying on it entirely;
    cut: edges that run up it and then leave"""
    X0 = np.asarray(base, float)
    ys = []
    for g in cut:
        i = eid(g, X0)
        P = E[i]["pts"]
        if np.linalg.norm(P[-1] - X0) < np.linalg.norm(P[0] - X0):
            P = P[::-1]; E[i]["pts"] = P; E[i]["v"] = E[i]["v"][::-1]
        k = vertical_departure(P, X0)
        ys.append(P[max(k - 1, 0), 1])
    ytop = float(np.median(ys))
    top = add_vertex([X0[0], ytop, X0[2]], govs)
    for g in cut:
        i = eid(g, X0)
        P = E[i]["pts"]
        k = np.searchsorted(-(P[:, 1] < ytop - 1e-9).astype(int) * 0 + np.cumsum(P[:, 1] >= ytop), 1)
        k = int(np.argmax(P[:, 1] >= ytop - 0.3)) if (P[:, 1] >= ytop - 0.3).any() else len(P) - 1
        Q = np.vstack([[V[top]["X"]], P[k + 1:]])
        E[i]["pts"] = Q
        E[i]["v"] = [top, E[i]["v"][1]]
        E[i]["L"] = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
    for g in kill:
        i = eid(g, X0)
        E[i]["alive"] = False
    plate_v = [v for v in range(len(V)) if V[v]["plate"] and np.linalg.norm(V[v]["X"] - X0) < 1.0]
    pv = plate_v[0]
    n = 40
    P = np.stack([np.full(n + 1, X0[0]), np.linspace(X0[1], ytop, n + 1), np.full(n + 1, X0[2])], 1)
    E.append(dict(gov=tuple(sorted(govs)), pts=P, L=float(ytop - X0[1]), n=0, v=[pv, top], faces=faces))
    return top


import os
from tubes import TUBES
from geom import PLATE as _PLATE
VACSPLIT = float(os.environ.get("EVG_VACSPLIT", 0.0))


def split_vacancy(base, pair, third, kill, delta):
    """resolve the 4-way vacancy by letting the diagonal pair touch all the way down: the two
    edges bounding face pair = (P, Q) run up the vacancy line offset by +-delta/2 across the
    P|Q bisector plane (towards `third` for the edge that contains it), each from its own
    plate vertex; the other on-line fragments are dropped (approximate graph)"""
    X0 = np.asarray(base, float)
    P_, Q_ = pair
    def leg(p):
        T = TUBES[p]; _, s = T.dist(X0[None]); return T.spine(s)[0]
    nrm = leg(Q_) - leg(P_); nrm[1] = 0; nrm /= np.linalg.norm(nrm)
    u = np.cross([0, 1.0, 0], nrm)
    if (leg(third) - X0) @ u < 0:
        u = -u
    for g in kill:
        i = eid(g, X0); E[i]["alive"] = False
    for g, sgn in (((min(P_, Q_), max(P_, Q_), third) if True else None, +1),):
        pass
    for i, e in enumerate(E):
        if not e.get("alive", True) or not {P_, Q_} <= set(e["gov"]) or len(e["gov"]) != 3:
            continue
        P = e["pts"]
        d = np.hypot(P[:, 0] - X0[0], P[:, 2] - X0[2])
        if d.min() > 0.6:
            continue
        sgn = +1 if third in e["gov"] else -1
        if np.linalg.norm(P[-1] - X0) < np.linalg.norm(P[0] - X0) or (P[0][1] > P[-1][1] and d[-1] < d[0]):
            if d[-1] < d[0]:
                P = P[::-1]; e["v"] = e["v"][::-1]; d = d[::-1]
        if P[0][1] > _PLATE + 1e-6:                       # starts above the plate on the line: drop down
            n = max(2, int((P[0][1] - _PLATE) / 0.5))
            Pre = np.stack([np.full(n, X0[0]), np.linspace(_PLATE, P[0][1], n, endpoint=False), np.full(n, X0[2])], 1)
            P = np.vstack([Pre, P]); d = np.r_[np.zeros(n), d]
        on = d < 0.6
        last = np.where(on)[0].max()
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        w = np.where(on, 1.0, np.clip(1.0 - (L - L[last]) / 6.0, 0.0, 1.0))
        w[: last + 1] = 1.0
        P = P + sgn * 0.5 * delta * w[:, None] * u[None, :]
        P[0, 1] = _PLATE
        pv = len(V)
        V.append(dict(X=P[0].copy(), ends=[], gov=tuple(sorted(set(e["gov"]))), plate=True))
        e["pts"] = P; e["v"] = [pv, e["v"][1]]
        e["L"] = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
        print("   vacancy split: edge", i, e["gov"], "offset", sgn, "-> plate vertex", pv, P[0].round(2))


if VACSPLIT > 0:
    split_vacancy([49.7, 1938.811, 1.4], (81, 90), 73, kill=[(73, 81, 91), (73, 90, 91)], delta=VACSPLIT)
    split_vacancy([-49.7, 1938.811, -33.15], (82, 95), 80, kill=[(80, 82, 94)], delta=VACSPLIT)
else:
    # 4-way vacancy edges (cyclic face lists from the lattice: sites at 0/60/120/180 deg)
    fix_vacancy([49.7, 1938.811, 1.4], (73, 81, 90, 91),
                faces=[(73, 90), (90, 91), (81, 91), (73, 81)],
                kill=[(73, 81, 91), (73, 90, 91)], cut=[(73, 81, 90), (81, 90, 91)])
    fix_vacancy([-49.7, 1938.811, -33.15], (80, 82, 94, 95),
                faces=[(80, 95), (94, 95), (82, 94), (80, 82)],
                kill=[(80, 82, 94)], cut=[(80, 82, 95), (82, 94, 95)])

# hub centre: drop the 6-way/focal fragments (the hub core replaces them)
for i, e in enumerate(E):
    g = set(e["gov"])
    c = e["pts"].mean(0)
    if g in ({81, 82, 83}, {81, 83, 102}, {82, 83, 102}) and np.hypot(c[0], c[2] + 15.88) < 14 and c[1] < 1975:
        e["alive"] = False
    if g in ({81, 83, 102}, {82, 83, 102}) and e["L"] < 6:
        e["alive"] = False

alive = [i for i, e in enumerate(E) if e.get("alive", True)]
print("edges alive", len(alive), "of", len(E))
pickle.dump((E, V), open("maps/graph2.pkl", "wb"))
for i, e in enumerate(E):
    if e.get("faces") or (e.get("alive", True) and set(e["gov"]) & {81, 82} and set(e["gov"]) & {90, 91, 94, 95}):
        print(i, "/".join(gname(g) for g in e["gov"]), "L=%.1f" % e["L"], e["v"], "alive" if e.get("alive", True) else "dead",
              e["pts"][0].round(1), e["pts"][-1].round(1))

# ---- snap 4-governor vertices to their exact position (Newton), blend attached edges
from vexact import solve
moved = []
for i, v in enumerate(V):
    if v["plate"] or len(v["gov"]) != 4 or any(g >= 1000 for g in v["gov"]):
        continue
    att = [k for k, e in enumerate(E) if e.get("alive", True) and i in e["v"]]
    if not att:
        continue
    p, r = solve(v["gov"], v["X"])
    dX = p - v["X"]
    if r > 1e-6 or np.linalg.norm(dX) > 3.0:
        continue
    v["X"] = p
    moved.append((i, round(float(np.linalg.norm(dX)), 2)))
    for k in att:
        P = E[k]["pts"]
        first = E[k]["v"][0] == i
        keep = np.linalg.norm(P - p, axis=1) > 3.0
        keep[-1 if first else 0] = True
        Q = P[keep]
        Q = np.vstack([[p], Q]) if first else np.vstack([Q, [p]])
        E[k]["pts"] = Q
        E[k]["L"] = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
pickle.dump((E, V), open("maps/graph2.pkl", "wb"))
print("snapped vertices:", sorted(moved, key=lambda t: -t[1])[:8], "...", len(moved))


# ---- T1 splits: a 4-gov vertex where two opposite faces both taper to a tip (on the pipe walls)
# becomes a short 4-way edge a-b; both faces then start with a cross side (ring-grid walls)
T1 = float(os.environ.get("EVG_T1", 0.0))


def _trace_dir(vid, g3):
    import items as IT
    from hubcore import HubCore
    IT.HUB["core"] = HubCore(); IT.HUB.pop("geom", None)
    g = IT.hub_geom()
    sgn = +1 if 81 in g3 else -1
    d = g[("line", sgn)][-1] - V[vid]["X"]
    return d / np.linalg.norm(d)


def split_t1(vid, w1, ell, blend=8.0):
    gov = set(V[vid]["gov"])
    P_, Q_ = w1
    R_, S_ = sorted(gov - {P_, Q_})
    arms = {}
    for k, e in enumerate(E):
        if e.get("alive", True) and vid in e["v"] and len(e["gov"]) == 3:
            P = e["pts"] if e["v"][0] == vid else e["pts"][::-1]
            d = P[min(4, len(P) - 1)] - P[0]
            arms[frozenset(e["gov"])] = ("e", k, d / np.linalg.norm(d))
    for g3 in ([81, 83, 102], [82, 83, 102]):
        if set(g3) <= gov and frozenset(g3) not in arms:
            arms[frozenset(g3)] = ("t", None, _trace_dir(vid, g3))
    e1, e2 = arms[frozenset((P_, Q_, R_))], arms[frozenset((P_, Q_, S_))]
    e3, e4 = arms[frozenset((P_, R_, S_))], arms[frozenset((Q_, R_, S_))]
    opp = lambda x, y: -(x[2] @ y[2])
    if opp(e1, e3) + opp(e2, e4) >= opp(e1, e4) + opp(e2, e3):
        A, B = (e1, e3), (e2, e4)
        cyc = [(P_, Q_), (Q_, R_), (R_, S_), (S_, P_)]
    else:
        A, B = (e1, e4), (e2, e3)
        cyc = [(P_, Q_), (P_, R_), (R_, S_), (Q_, S_)]
    n = e1[2] - e2[2]; n /= np.linalg.norm(n)
    v0 = V[vid]["X"].copy()
    a, b = v0 + 0.5 * ell * n, v0 - 0.5 * ell * n
    bid = len(V)
    V.append(dict(X=b, ends=[], gov=V[vid]["gov"], plate=False))
    V[vid]["X"] = a
    for grp, sh, vnew in ((A, +0.5 * ell * n, vid), (B, -0.5 * ell * n, bid)):
        for arm in grp:
            if arm[0] != "e":
                continue
            e = E[arm[1]]
            first = e["v"][0] == vid
            P = e["pts"] if first else e["pts"][::-1]
            L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            w = np.clip(1.0 - L / blend, 0.0, 1.0)
            P = P + w[:, None] * sh[None, :]
            e["pts"] = P if first else P[::-1]
            e["L"] = float(np.linalg.norm(np.diff(e["pts"], axis=0), axis=1).sum())
            if vnew != vid:
                e["v"] = [vnew if x == vid else x for x in e["v"]]
    trace_side = "a" if any(x[0] == "t" for x in A) else ("b" if any(x[0] == "t" for x in B) else None)
    cyc = [tuple(sorted(f)) for f in cyc]
    E.append(dict(gov=tuple(sorted(gov)), pts=np.linspace(a, b, 12), L=float(ell), n=0, v=[vid, bid],
                  faces=cyc, alive=True))
    print("   T1 split at vertex %d: wedges %s|%s, cross edge %d-%d (%.1f mm), faces %s, trace end %s"
          % (vid, (P_, Q_), (R_, S_), vid, bid, ell, cyc, trace_side))
    return trace_side, bid


T1_INFO = {}
if T1 > 0:
    for vid, w1 in ((88, (81, 92)), (90, (81, 102)), (98, (82, 93)), (99, (82, 102))):
        T1_INFO[vid] = split_t1(vid, w1, T1)
    pickle.dump((E, V), open("maps/graph2.pkl", "wb"))
pickle.dump(T1_INFO, open("maps/t1_info.pkl", "wb"))
