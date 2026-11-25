"""Global bottom-zone section topology, rim by LOFT (prototype): the lattice keeps every quad lying
GMIN inside the wall R692.15 (+ the core's channel quads), cleaned to one star-shaped outline loop;
each outline node is lofted radially onto the wall with one quad strip -- no clipping, no pillows
(the core section's own edge construction).  -> topo2.pkl (same fields as topo.pkl; 'outer' is
now the wall loop of new ("wl", k) keys, 'lattice_outline' the loop they hang from)"""
import os, sys, pickle, collections
os.environ.setdefault("LOFT_RIM", "0")          # no ring shrinking: the O-grid is not physical, cut it anywhere
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "p3"), os.path.join(HERE, "..", "p1")]
import bsec, rules, secA, cells_A, edgeA
from geom import CENTRAL

R_W = rules.R_RING
TOP = dict(z=rules.Z["top"], ring=rules.R_PIN)
GMIN = float(os.environ.get("GMIN", "2.0"))          # strip at least this thick


def base_quads():
    cells = cells_A.template_cells() + edgeA.edge_cells() + sorted(CENTRAL)
    quads, inner = bsec.build(cells)
    X = np.array(secA.CORE.X)
    for qi, (q, lab) in enumerate(zip(secA.CORE.quads, secA.CORE.lab)):
        if lab in secA.BORE_LABS:
            quads.append((tuple(secA.ckey(X[i]) for i in q), lab, ("bore", qi)))
    quads, inner, alias = secA.unify(quads, inner)
    import levels as lv
    core_xy = {secA.ckey(p): tuple(p) for p in secA.CORE.X}
    cache = {}
    L = lv.LV["top"]
    fixed = []
    pos = {}
    for q, lab, c in quads:
        P = np.array([lv.coords(k, L, inner, cache, core_xy) for k in q])
        a = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        fixed.append((q if a > 0 else q[::-1], lab, c))
        for k, p in zip(q, P):
            pos[k] = p
    return cells, fixed, inner, alias, pos


def loops_of(Q):
    e = collections.Counter()
    for q in Q:
        for a in range(4):
            e[(q[a], q[(a + 1) % 4])] += 1
    und = collections.Counter()
    for (u, v), n in e.items():
        und[frozenset((u, v))] += n
    bedges = [(u, v) for (u, v) in e if und[frozenset((u, v))] == 1]
    nxt = collections.defaultdict(list)
    for u, v in bedges:
        nxt[u].append(v)
    loops, used, bad = [], set(), set()
    for u0, v0 in bedges:
        if (u0, v0) in used:
            continue
        loop = [u0]; used.add((u0, v0)); cur = v0
        while cur != u0:
            loop.append(cur)
            cand = [w for w in nxt[cur] if (cur, w) not in used]
            if len(cand) != 1:
                bad.add(cur)
                break
            used.add((cur, cand[0])); cur = cand[0]
        loops.append(loop)
    return loops, bad


def level_rmax(quads, inner):
    """largest node radius of each quad over EVERY level's raw layout (the gap must hold at all of them)."""
    import levels as lv
    core_xy = {secA.ckey(p): tuple(p) for p in secA.CORE.X}
    keys = sorted({k for q, _, _ in quads for k in q}, key=str)
    rk = np.zeros(len(keys))
    for L in lv.LEVELS:
        cache = {}
        for j, k in enumerate(keys):
            p = np.asarray(lv.coords(k, L, inner, cache, core_xy), float)[:2]
            rk[j] = max(rk[j], np.hypot(*p))
    rofk = dict(zip(keys, rk))
    return np.array([max(rofk[k] for k in q) for q, _, _ in quads])


def select(quads, pos, inner):
    rmax = level_rmax(quads, inner)
    labs = np.array([l for _, l, _ in quads])
    sel = rmax < R_W - GMIN
    # plain per-quad selection: any part of any O-grid may go
    ogrid = np.isin(labs, ("oring", "disc"))
    cellof = np.array([str(c) for _, _, c in quads])
    disc = np.zeros(len(quads), bool)
    sel |= labs == "chan"                                  # the core's channels stay open into band A
    ok_add = rmax < R_W - 0.5 * GMIN
    exf = os.path.join(HERE, "rim_exclude.npy")       # quads that came out bad at some level: unselect
    if os.path.exists(exf):
        from scipy.spatial import cKDTree
        ex = np.load(exf)
        if len(ex):
            cen = np.array([np.mean([pos[k] for k in q], 0) for q, _, _ in quads])
            d, _ = cKDTree(ex).query(cen)
            excl = (d < 1e-6) & (labs != "chan")
            sel &= ~excl; ok_add &= ~excl
            print("excluded %d quads that came out bad at some level" % excl.sum(), flush=True)
    # node -> quads, quad adjacency
    inc = collections.defaultdict(list)
    for i, (q, _, _) in enumerate(quads):
        for k in q:
            inc[k].append(i)
    NIT = int(os.environ.get("ITERS", "400"))
    for it in range(NIT):
        smooth_phase = it < NIT // 4                       # phase 1 smooths, phase 2 only repairs (star wins)
        Q = [quads[i][0] for i in np.where(sel)[0]]
        loops, bad = loops_of(Q)
        loops.sort(key=len, reverse=True)
        outer = loops[0]
        changed = 0
        # holes / extra loops near the rim: fill the quads around them
        for L in loops[1:]:
            rr = np.hypot(*np.array([pos[k] for k in L]).T)
            if rr.max() > R_W - 60:
                for k in L:
                    for qi in inc[k]:
                        if not sel[qi] and ok_add[qi]:
                            sel[qi] = True; changed += 1
        # pinch nodes: fill where possible, otherwise drop the outermost selected quad there
        for k in bad:
            add = [qi for qi in inc[k] if not sel[qi] and ok_add[qi]]
            if add:
                for qi in add:
                    sel[qi] = True; changed += 1
            else:
                cand = [qi for qi in inc[k] if sel[qi] and labs[qi] != "chan"]
                if cand:
                    sel[max(cand, key=lambda qi: rmax[qi])] = False; changed += 1
        # smooth the outline: drop one-quad bumps, fill one-quad notches (at most two outline edges, adjacent)
        if os.environ.get("RIMSMOOTH", "1") == "1" and smooth_phase:
            ekey_ = collections.defaultdict(list)
            for qi in range(len(quads)):
                q = quads[qi][0]
                for a_ in range(4):
                    ekey_[frozenset((q[a_], q[(a_ + 1) % 4]))].append(qi)
            for qi in np.where(rmax > R_W - 60)[0]:
                q = quads[qi][0]
                d_ = []
                for a_ in range(4):
                    nb_ = [j for j in ekey_[frozenset((q[a_], q[(a_ + 1) % 4]))] if j != qi]
                    other = sel[nb_[0]] if nb_ else False
                    if other != sel[qi]:
                        d_.append(a_)
                bad_ = len(d_) >= 3 or (len(d_) == 2 and (d_[1] - d_[0]) % 4 == 2)
                if not bad_:
                    continue
                if sel[qi] and labs[qi] != "chan":
                    sel[qi] = False; changed += 1                 # a bump: drop it
                elif not sel[qi] and ok_add[qi]:
                    sel[qi] = True; changed += 1                  # a notch: fill it
        # star-shapedness of the outer loop: angle must increase along it (ccw)
        P = np.array([pos[k] for k in outer])
        if 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0:
            outer = outer[::-1]; P = P[::-1]
        ang = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
        d = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
        back = np.where(d < -1e-9)[0]
        ekey = collections.defaultdict(list)
        for qi in np.where(sel)[0]:
            q = quads[qi][0]
            for a_ in range(4):
                ekey[frozenset((q[a_], q[(a_ + 1) % 4]))].append(qi)
        for j in back:                                      # a backtracking edge: trim the overhang inside it
            e = frozenset((outer[j], outer[(j + 1) % len(outer)]))
            inside = [qi for qi in ekey.get(e, []) if sel[qi]]
            trimmed = False
            for qi in inside:
                if labs[qi] != "chan":
                    sel[qi] = False; changed += 1; trimmed = True
                    if disc[qi]:                            # a disc goes with its whole O-grid
                        sel[ogrid & (cellof == cellof[qi])] = False
            if not trimmed:                                 # a channel: fill the notch instead
                for k in e:
                    for qi in inc[k]:
                        if not sel[qi] and ok_add[qi]:
                            sel[qi] = True; changed += 1
        if not changed and not smooth_phase:
            break
    return sel, outer, len(back), bad


def main():
    cells, quads, inner, alias, pos = base_quads()
    sel, outline, nback, bad = select(quads, pos, inner)
    P = np.array([pos[k] for k in outline])
    r = np.hypot(*P.T)
    print("lattice: %d of %d quads; outline %d nodes, r %.2f..%.2f (gap to wall >= %.2f); backtracking edges %d, pinch nodes %d" % (
        sel.sum(), len(quads), len(outline), r.min(), r.max(), R_W - r.max(), nback, len(bad)), flush=True)
    # radial loft: wall nodes ("wl", k) at R_W, angles from the outline made strictly increasing
    ang = np.unwrap(np.arctan2(P[:, 1], P[:, 0]))
    n = len(outline)
    G = [quads[i] for i in np.where(sel)[0]]
    wl = [("wl", k) for k in outline]
    for t in range(n):
        j = (t + 1) % n
        G.append(((outline[t], wl[t], wl[j], outline[j]), "rim0", ("rim0",)))     # ccw: the strip lies outside the ccw outline
    for t, k in enumerate(wl):
        pos[k] = R_W * np.array([np.cos(ang[t]), np.sin(ang[t])])
    out = dict(cells=cells, quads=G, inner=inner, alias=alias, outer=wl, lattice_outline=outline)
    pickle.dump(out, open(os.path.join(HERE, "topo2.pkl"), "wb"))
    # quick 2D check at the top config
    from q2 import corner_sj
    keys = {}
    for q, _, _ in G:
        for k in q:
            keys.setdefault(k, len(keys))
    X = np.zeros((len(keys), 2))
    for k, i in keys.items():
        X[i] = pos[k]
    Q = np.array([[keys[k] for k in q] for q, _, _ in G])
    s = corner_sj(X, Q)
    strip = np.array([l == "rim0" for _, l, _ in G])
    e = np.linalg.norm(np.roll(X[Q], -1, 1) - X[Q], axis=2).min(1)
    print("section %d quads: strip %d (min sj %.3f, min edge %.3f), lattice min sj %.3f" % (
        len(Q), strip.sum(), s[strip].min(), e[strip].min(), s[~strip].min()), flush=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for ax, (cxy, w) in zip(axs, (((480, 480), 60), ((0, 680), 40))):
        m = (np.abs(X[Q].mean(1) - np.array(cxy)) < w * 1.5).all(1)
        ax.add_collection(PolyCollection(X[Q[m]], array=s[m], cmap="RdYlGn", clim=(-0.2, 1), edgecolors="k", linewidths=0.3))
        th = np.linspace(0, 2 * np.pi, 3000); ax.plot(R_W * np.cos(th), R_W * np.sin(th), "b-", lw=0.8)
        ax.set_xlim(cxy[0] - w, cxy[0] + w); ax.set_ylim(cxy[1] - w, cxy[1] + w); ax.set_aspect("equal")
    axs[0].set_title("rim by radial loft (top config): lattice + one strip to R692.15, colour corner sj")
    os.makedirs(os.path.join(HERE, "img"), exist_ok=True)
    plt.savefig(os.path.join(HERE, "img", "topo2_rim.png"), dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
