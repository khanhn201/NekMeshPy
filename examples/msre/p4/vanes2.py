"""Vane outlines as shortest EDGE paths hugging the exact vane outline (side -, inner end, side +),
strip = flood fill inside path + vessel wall, pillows = turn quads.  -> vanes.pkl (same format)"""
import os, sys, collections, heapq, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vanes as V
from head import R0V, VHW, NV, R_VES

BETA = float(os.environ.get("VBETA", "4.0"))
HN = float(os.environ.get("VHN", "2.0"))


def seg_dist(P, a, b):
    d = b - a
    s = np.clip((P - a) @ d / (d @ d), 0, 1)
    return np.linalg.norm(P - (a + s[:, None] * d), axis=1)


def dijkstra(cx, adj, src, dst, cost_node, banned):
    X = cx.X
    dist = {src: 0.0}; prev = {}
    h = [(0.0, src)]
    while h:
        d, u = heapq.heappop(h)
        if u == dst:
            break
        if d > dist.get(u, np.inf):
            continue
        for v in adj[u]:
            if v in banned and v != dst:
                continue
            L = np.linalg.norm(X[u] - X[v])
            nd = d + L * (1 + BETA * 0.5 * (cost_node[u] + cost_node[v]))
            if nd < dist.get(v, np.inf):
                dist[v] = nd; prev[v] = u
                heapq.heappush(h, (nd, v))
    if dst not in prev:
        return None
    p = [dst]
    while p[-1] != src:
        p.append(prev[p[-1]])
    return p[::-1]


def shortcut3(cx, path):
    """replace any 3 consecutive path edges around one quad by the quad's 4th edge."""
    changed = True
    while changed:
        changed = False
        for k in range(len(path) - 3):
            a, b, c, d = path[k:k + 4]
            for qi in cx.inc[a]:
                q = list(cx.Q[qi])
                if all(v in q for v in (a, b, c, d)):
                    path = path[:k + 1] + path[k + 3:]
                    changed = True
                    break
            if changed:
                break
    return path


def violations(cx, sel, nn, zone):
    nb = cx.nb
    out = []
    for i in zone:
        d = [k for k in range(4) if nb[i, k] >= 0 and sel[nb[i, k]] != sel[i]]
        if len(d) >= 3:
            out.append(i)
        elif len(d) == 2 and d[1] - d[0] == 2:
            a_, b_ = nb[i, d[0]], nb[i, d[1]]
            if not (sel[i] and np.sign(nn[a_]) != np.sign(nn[b_])):
                out.append(i)
    return out


def select(cx, t, n, taken):
    """centroid inside the exact strip; then single-quad toggles (cheapest first) until no quad has
    3 outline edges or two opposite ones (except a one-quad-wide strip)."""
    tt, nn = cx.cen @ t, cx.cen @ n
    sel = (tt > R0V) & (np.abs(nn) < VHW) & ~taken
    P = cx.X[cx.Q]; pn = P @ n; pt = P @ t
    ann = np.char.startswith(cx.lab.astype(str), "ann")
    sel |= ann & (pn.min(1) < 0) & (pn.max(1) > 0) & (pt.min(1) > 0) & ~taken
    cost = np.minimum(np.abs(np.abs(nn) - VHW), np.where(np.abs(nn) < VHW, np.abs(tt - R0V), 1e9))
    zone = np.where((tt > R0V - 30) & (np.abs(nn) < 40))[0]
    zs = set(zone.tolist())
    for _ in range(500):
        bad = violations(cx, sel, nn, zone)
        if not bad:
            break
        best = None
        for i in bad:
            for c in [i] + [j for j in cx.nb[i] if j >= 0]:
                if c not in zs or (taken[c] and not sel[c]):
                    continue
                loc = list({c} | {j for j in cx.nb[c] if j >= 0} | {k for j in cx.nb[c] if j >= 0 for k in cx.nb[j] if k >= 0})
                b0 = len(violations(cx, sel, nn, loc))
                sel[c] = ~sel[c]
                b1 = len(violations(cx, sel, nn, loc))
                sel[c] = ~sel[c]
                if b1 < b0 and (best is None or (b0 - b1, -cost[c]) > (best[1], -cost[best[0]])):
                    best = (c, b0 - b1)
        if best is None:
            break
        sel[best[0]] = ~sel[best[0]]
    # keep the largest edge-connected piece
    comp = -np.ones(len(sel), int); big = []
    for s0 in np.where(sel)[0]:
        if comp[s0] >= 0:
            continue
        st = [s0]; comp[s0] = s0; mem = []
        while st:
            i = st.pop(); mem.append(i)
            for j in cx.qnb[i]:
                if sel[j] and comp[j] < 0:
                    comp[j] = s0; st.append(j)
        if len(mem) > len(big):
            big = mem
    out = np.zeros(len(sel), bool); out[big] = True
    return out, len(violations(cx, out, nn, zone))


def run(cx):
    taken = np.zeros(len(cx.Q), bool)
    res = []
    for kv in range(NV):
        th = np.radians(7.5 * kv)
        t = np.array([np.cos(th), np.sin(th)]); n = np.array([-t[1], t[0]])
        sel, nv = select(cx, t, n, taken)
        path = V.outline(cx, sel)
        if path is None or nv:
            print("vane %d: %d violations, outline %s" % (kv, nv, "ok" if path else "NOT one chain"), flush=True)
            if path is None:
                continue
        taken |= sel
        res.append((kv, t, n, sel, path))
    return res, taken


def turn_quads(cx, path):
    done = set()
    for j in range(1, len(path) - 1):
        u, v, w = path[j - 1], path[j], path[j + 1]
        for qi in cx.inc[v]:
            q = list(cx.Q[qi]); a = q.index(v)
            if {q[(a - 1) % 4], q[(a + 1) % 4]} == {u, w}:
                done.add(qi)
    return done


if __name__ == "__main__":
    t0 = time.time()
    H = pickle.load(open(os.path.join(HERE, "head_sec.pkl"), "rb"))
    cx = V.Ctx(H["quads"], H["keys"], H["X_kn"])
    vs, taken = run(cx)
    pw = set()
    for kv, t, n, sel, path in vs:
        pw |= turn_quads(cx, path)
    dev = [np.abs(np.abs(cx.X[p] @ n) - VHW)[np.abs(cx.X[p] @ n) > 0] for kv, t, n, sel, p in vs]
    print("vanes: %d of %d, %d quads, %d pillows, chain nodes %d (%.0fs)" % (len(vs), NV, taken.sum(), len(pw),
          sum(len(v[4]) for v in vs), time.time() - t0))
    pickle.dump(dict(vanes=vs, taken=taken, pillow=sorted(pw)), open(os.path.join(HERE, os.environ.get("OUTV", "vanes2.pkl")), "wb"))
    os.makedirs(os.path.join(HERE, "img"), exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, ax = plt.subplots(figsize=(16, 9))
    P = cx.X[cx.Q]
    ax.add_collection(PolyCollection(P, facecolors=np.where(taken[:, None], [[0.9, 0.4, 0.3, 1]], [[1, 1, 1, 0]]), edgecolors="k", linewidths=0.2))
    for qi in pw:
        ax.fill(*cx.X[cx.Q[qi]].T, color="b", alpha=0.5)
    for kv, t, n, sel, p in vs:
        th = np.radians(7.5 * kv); tt = np.array([np.cos(th), np.sin(th)]); nn = np.array([-tt[1], tt[0]])
        for sgn in (-1, 1):
            a, b = R0V * tt + sgn * VHW * nn, R_VES * tt + sgn * VHW * nn
            ax.plot([a[0], b[0]], [a[1], b[1]], "g-", lw=0.8)
    ax.set_xlim(440, 740); ax.set_ylim(-110, 110); ax.set_aspect("equal")
    plt.savefig(os.path.join(HERE, "img", "vanes2_zoom.png"), dpi=120, bbox_inches="tight")
