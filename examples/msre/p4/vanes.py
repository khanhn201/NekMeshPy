"""The 48 anti-swirl vanes on the L2 section: per vane a selection-first strip (whole channels,
max-2 rule fixpoint, manifold), its outline chain (vessel wall -> side -> inner end -> side -> wall),
pillows for single-quad turns, and the vane-config positions (chains on the exact vane lines,
toolkit smoother around them)."""
import os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from head import R0V, VHW, NV, R_VES


class Ctx:
    def __init__(self, quads, keys, X):
        self.quads = list(quads)
        self.keys = keys
        self.X = X
        self.Q = np.array([[keys[k] for k in q] for q, _, _ in quads])
        self.lab = np.array([l for _, l, _ in quads])
        P = X[self.Q]
        self.cen = P.mean(1)
        self.area = np.abs(0.5 * np.sum(P[..., 0] * np.roll(P[..., 1], -1, 1) - np.roll(P[..., 0], -1, 1) * P[..., 1], 1))
        self.edge_q = collections.defaultdict(list)
        for qi, q in enumerate(self.Q):
            for a in range(4):
                self.edge_q[frozenset((q[a], q[(a + 1) % 4]))].append(qi)
        self.nb = -np.ones((len(self.Q), 4), int)
        for qi, q in enumerate(self.Q):
            for a in range(4):
                for qj in self.edge_q[frozenset((q[a], q[(a + 1) % 4]))]:
                    if qj != qi:
                        self.nb[qi, a] = qj
        self.qnb = [set(x for x in row if x >= 0) for row in self.nb]
        self.inc = collections.defaultdict(list)
        for qi, q in enumerate(self.Q):
            for v in q:
                self.inc[v].append(qi)
        bedges = [e for e, qs in self.edge_q.items() if len(qs) == 1]
        self.bnodes = {v for e in bedges for v in e}
        # channels = edge-connected components of 'chan' quads
        self.group = -np.ones(len(self.Q), int)
        g = 0
        for s0 in np.where(self.lab == "chan")[0]:
            if self.group[s0] >= 0:
                continue
            stack = [s0]; self.group[s0] = g
            while stack:
                i = stack.pop()
                for j in self.qnb[i]:
                    if self.lab[j] == "chan" and self.group[j] < 0:
                        self.group[j] = g; stack.append(j)
            g += 1
        self.groups = collections.defaultdict(list)
        for i, gi in enumerate(self.group):
            if gi >= 0:
                self.groups[gi].append(i)


def select(cx, t, n, taken):
    tt, nn = cx.cen @ t, cx.cen @ n
    ideal = (tt > R0V) & (np.abs(nn) < VHW) & ~taken
    P = cx.X[cx.Q]; pn = P @ n; pt = P @ t
    cross = (pn.min(1) < 0) & (pn.max(1) > 0) & (pt.min(1) > 0)
    ann = np.char.startswith(cx.lab.astype(str), "ann")
    ideal |= ann & cross & ~taken
    axis = (tt > R0V) & (np.abs(nn) < 2.0)
    sel = ideal.copy()
    cost = np.minimum(np.abs(np.abs(nn) - VHW), np.where(np.abs(nn) < VHW, np.abs(tt - R0V), 1e9))
    zone0 = np.where((tt > R0V - 20) & (np.abs(nn) < 30))[0]
    zset = set(zone0.tolist())
    nb = cx.nb

    def viol(i):
        d = [k for k in range(4) if nb[i, k] >= 0 and sel[nb[i, k]] != sel[i]]
        if len(d) >= 3:
            return 1
        if len(d) == 2 and d[1] - d[0] == 2:
            a_, b_ = nb[i, d[0]], nb[i, d[1]]
            if not (sel[i] and np.sign(nn[a_]) != np.sign(nn[b_]) and abs(nn[a_]) > 3 and abs(nn[b_]) > 3):
                return 1
        return 0

    def local(i):
        s = {i} | cx.qnb[i]
        out = set(s)
        for j in s:
            out |= cx.qnb[j]
        return out

    def fixpoint(zone):
        locked = np.zeros(len(cx.Q), bool)
        zone = list(zone)
        for _ in range(4000):
            bad = [i for i in zone if viol(i)]
            if not bad:
                break
            i = min(bad, key=lambda k: cost[k])
            best = None
            nbi = list(cx.qnb[i])
            moves = [(c,) for c in [i] + nbi] + [(i, j) for j in nbi] + \
                    [(j, k) for j in nbi for k in cx.qnb[j] if k != i]
            for mv in moves:
                if any(c not in zset or locked[c] or (taken[c] and not sel[c]) for c in mv):
                    continue
                region = set().union(*[local(c) for c in mv])
                before = sum(viol(k) for k in region)
                for c in mv:
                    sel[c] = not sel[c]
                after = sum(viol(k) for k in region)
                for c in mv:
                    sel[c] = not sel[c]
                cst = sum(cost[c] for c in mv)
                if after < before and (best is None or (before - after, -cst) > (best[2], -best[1])):
                    best = (mv, cst, before - after)
            if best is None:
                locked[i] = True
                zone.remove(i)
                continue
            for c in best[0]:
                sel[c] = not sel[c]
                locked[c] = True

    def prune():
        comp = -np.ones(len(cx.Q), int); bestc = None
        for s0 in np.where(sel)[0]:
            if comp[s0] >= 0:
                continue
            stack = [s0]; comp[s0] = s0; members = []
            while stack:
                i = stack.pop(); members.append(i)
                for j in cx.qnb[i]:
                    if sel[j] and comp[j] < 0:
                        comp[j] = s0; stack.append(j)
            if bestc is None or len(members) > len(bestc):
                bestc = members
        sel[:] = False
        sel[bestc] = True

    def manifold():
        for _ in range(50):
            seen = np.zeros(len(cx.Q), bool); filled = False
            for s0 in np.where(~sel & (np.abs(nn) < 40) & (tt > R0V - 40))[0]:
                if seen[s0]:
                    continue
                stack = [s0]; seen[s0] = True; comp = []
                while stack:
                    i = stack.pop(); comp.append(i)
                    for j in cx.qnb[i]:
                        if not sel[j] and not seen[j]:
                            seen[j] = True; stack.append(j)
                    if len(comp) > 40:
                        break
                if len(comp) <= 40 and not any(v in cx.bnodes for i in comp for v in cx.Q[i]):
                    sel[comp] = True; filled = True
            deg = collections.Counter()
            for e, qs in cx.edge_q.items():
                if len(qs) == 2 and sel[qs[0]] != sel[qs[1]]:
                    for v in e:
                        deg[v] += 1
            pinch = [v for v, d in deg.items() if d > 2]
            if not pinch and not filled:
                break
            for v in pinch:
                cand = [qi for qi in cx.inc[v] if not sel[qi] and not taken[qi]]
                if cand:
                    sel[min(cand, key=lambda i: abs(nn[i]))] = True

    for _ in range(8):
        fixpoint(zone0)
        prune()
        manifold()
        if not any(viol(i) for i in zone0):
            break
    prune()
    nv = sum(viol(i) for i in zone0)
    if nv:
        print("   %d quads break the 2-edge rule" % nv, flush=True)
    return sel


def outline(cx, sel):
    nxt = collections.defaultdict(list)
    for e, qs in cx.edge_q.items():
        if len(qs) == 2 and sel[qs[0]] != sel[qs[1]]:
            u, v = tuple(e)
            nxt[u].append(v); nxt[v].append(u)
    ends = [v for v in nxt if len(nxt[v]) == 1]
    if len(ends) != 2 or any(len(v) > 2 for v in nxt.values()):
        return None
    path = [ends[0]]; prev = -1
    while True:
        cand = [w for w in nxt[path[-1]] if w != prev]
        if not cand:
            break
        prev = path[-1]; path.append(cand[0])
    return path


def pillow(cx, sel, path):
    """pillow every non-channel element that is the only quad inside a turn of the path."""
    new_quads = list(cx.quads)
    new_sel = list(sel)
    keyinv = {v: k for k, v in cx.keys.items()}
    done = set()
    for j in range(1, len(path) - 1):
        u, v, w = path[j - 1], path[j], path[j + 1]
        for qi in cx.inc[v]:
            if qi in done:
                continue
            q = list(cx.Q[qi]); a = q.index(v)
            if {q[(a - 1) % 4], q[(a + 1) % 4]} == {u, w}:
                done.add(qi)
    for qi in sorted(done):
        q, lab, c = cx.quads[qi]
        inner = [("pv", qi, j) for j in range(4)]
        new_quads[qi] = (tuple(inner), lab, c)
        for j in range(4):
            new_quads.append(((q[j], q[(j + 1) % 4], inner[(j + 1) % 4], inner[j]), lab, c))
            new_sel.append(new_sel[qi])
    return new_quads, np.array(new_sel), sorted(done)


def spread_targets(cx, path, t, n):
    """target position of every chain node on the exact vane outline."""
    P = cx.X[path]
    cm, cp = R0V * t - VHW * n, R0V * t + VHW * n
    im = int(np.argmin(np.linalg.norm(P - cm, axis=1)))
    ip = int(np.argmin(np.linalg.norm(P - cp, axis=1)))
    if im > ip:
        path = path[::-1]; P = P[::-1]; im, ip = len(path) - 1 - im, len(path) - 1 - ip
    if im == ip:                                   # one node nearest both corners: split there
        if np.linalg.norm(P[im + 1] - cp) < np.linalg.norm(P[im - 1] - cm):
            ip = im + 1
        else:
            im = im - 1
    tw = np.sqrt(R_VES ** 2 - VHW ** 2)
    tgt = {}

    def spread(idx, p0, p1):
        seg = P[idx]
        L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))])
        for k, f in zip(idx, L / max(L[-1], 1e-12)):
            tgt[path[k]] = p0 + f * (p1 - p0)

    spread(list(range(im, -1, -1)), cm, tw * t - VHW * n)
    spread(list(range(im, ip + 1)), cm, cp)
    spread(list(range(ip, len(path))), cp, tw * t + VHW * n)
    return tgt


def run(cx):
    taken = np.zeros(len(cx.Q), bool)
    out = []
    for kv in range(NV):
        th = np.radians(7.5 * kv)
        t = np.array([np.cos(th), np.sin(th)]); n = np.array([-t[1], t[0]])
        sel = select(cx, t, n, taken)
        path = outline(cx, sel)
        if path is None:
            print("vane %d: outline not a single chain" % kv, flush=True)
            continue
        taken |= sel
        out.append((kv, t, n, sel.copy(), path))
    return out, taken


if __name__ == "__main__":
    import pickle, time
    t0 = time.time()
    H = pickle.load(open(os.path.join(HERE, "head_sec.pkl"), "rb"))
    cx = Ctx(H["quads"], H["keys"], H["X_kn"])
    vs, taken = run(cx)
    print("vanes: %d of %d, %d quads selected (%.0fs)" % (len(vs), NV, taken.sum(), time.time() - t0))
    pw = set()
    for kv, t, n, sel, path in vs:
        _, _, done = pillow(cx, taken, path)
        pw |= set(done)
    print("pillow quads:", len(pw), "of which annulus:", sum(1 for qi in pw if str(cx.lab[qi]).startswith("ann")))
    pickle.dump(dict(vanes=vs, taken=taken, pillow=sorted(pw)), open(os.path.join(HERE, "vanes.pkl"), "wb"))
    os.makedirs(os.path.join(HERE, "img"), exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 14))
    from matplotlib.collections import PolyCollection
    P = cx.X[cx.Q]
    ax.add_collection(PolyCollection(P, facecolors=np.where(taken[:, None], [[0.9, 0.4, 0.3, 1]], [[1, 1, 1, 0]]), edgecolors="k", linewidths=0.1))
    for qi in pw:
        ax.fill(*cx.X[cx.Q[qi]].T, color="b", alpha=0.6)
    ax.set_xlim(-760, 760); ax.set_ylim(-760, 760); ax.set_aspect("equal")
    plt.savefig(os.path.join(HERE, "img", "vanes_all.png"), dpi=110)
    ax.set_xlim(420, 760); ax.set_ylim(-120, 220)
    plt.savefig(os.path.join(HERE, "img", "vanes_zoom.png"), dpi=110)
