"""Band-A section topology over the whole plenum: lattice template cells (incl. T-cells next to
the bores) + the core's bore quads + central / edge channel footprints, unified on one key set.
The edge band (lattice outline -> R692.15) is filled separately (edgefill.py)."""
import os, sys, pickle, collections
import numpy as np
import bsec, rules, cells_A
from geom import CENTRAL

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = pickle.load(open(os.path.join(HERE, "..", "p1", "section.pkl"), "rb"))
BORE_LABS = ("thim", "th_hole", "th_disc", "bk_A", "bk_AB", "bk_BC", "bk_CD")


def ckey(p):
    return (round(float(p[0]), 6) + 0.0, round(float(p[1]), 6) + 0.0)


def collect():
    import edgeA
    template = cells_A.template_cells() + edgeA.edge_cells()
    quads, inner = bsec.build(template)
    tset = set(template)
    for (c, s), kk in bsec.FP_HALVES:
        if c in tset:
            continue
        lab = "chan" if c in CENTRAL else "chan_e"
        for q in bsec.fp.QUADS:
            quads.append((tuple(kk[r] for r in q), lab, c))
    X = np.array(CORE.X)
    for qi, (q, lab) in enumerate(zip(CORE.quads, CORE.lab)):
        if lab in BORE_LABS:
            quads.append((tuple(ckey(X[i]) for i in q), lab, ("bore", qi)))
    return template, quads, inner


def unify(quads, inner):
    """alias symbolic keys onto coordinate keys that coincide at the TOP config."""
    TOP = dict(z=rules.Z["top"], ring=rules.R_PIN)
    BOT = dict(z=rules.Z["AB"], ring=rules.R_PIN)
    ct, cb = {}, {}
    allk = {k for q, _, _ in quads for k in q}
    pos = {}
    for k in allk:
        pt = bsec.coords(k, TOP, inner, ct)
        pos.setdefault(ckey(pt), []).append(k)
    alias = {}
    for p, ks in pos.items():
        if len(ks) == 1:
            continue
        coordk = [k for k in ks if isinstance(k[0], (float, np.floating))]
        tgt = coordk[0] if coordk else ks[0]
        pb = [bsec.coords(k, BOT, inner, cb) for k in ks]
        spread = max(np.linalg.norm(a - pb[0]) for a in pb)
        if spread > 1e-9:
            raise RuntimeError("keys %s coincide at TOP but not at BOT (%.3g)" % (ks, spread))
        for k in ks:
            if k != tgt:
                alias[k] = tgt
    out = []
    for q, lab, c in quads:
        q2 = tuple(alias.get(k, k) for k in q)
        P = np.array([bsec.coords(k, TOP, inner, ct) for k in q])
        a = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        out.append((q2 if a > 0 else q2[::-1], lab, c))
    inner2 = {c: [alias.get(k, k) for k in b] for c, b in inner.items()}
    return out, inner2, alias


def boundary_loops(quads):
    e = collections.Counter()
    for q, _, _ in quads:
        for a in range(4):
            u, v = q[a], q[(a + 1) % 4]
            e[(u, v)] += 1
    und = collections.Counter()
    for (u, v), n in e.items():
        und[frozenset((u, v))] += n
    bedges = [(u, v) for (u, v) in e if und[frozenset((u, v))] == 1]
    nxt = collections.defaultdict(list)
    for u, v in bedges:
        nxt[u].append(v)
    loops, used = [], set()
    for u0, v0 in bedges:
        if (u0, v0) in used:
            continue
        loop = [u0]; used.add((u0, v0)); cur = v0
        while cur != u0:
            loop.append(cur)
            cand = [w for w in nxt[cur] if (cur, w) not in used]
            if len(cand) != 1:
                raise RuntimeError("non-manifold boundary at %s (%d)" % (cur, len(cand)))
            used.add((cur, cand[0])); cur = cand[0]
        loops.append(loop)
    return loops


if __name__ == "__main__":
    template, quads, inner = collect()
    quads, inner, alias = unify(quads, inner)
    print("quads", len(quads), collections.Counter(l for _, l, _ in quads), "aliases", len(alias))
    TOP = dict(z=rules.Z["top"], ring=rules.R_PIN)
    cache = {}
    loops = boundary_loops(quads)
    for L in sorted(loops, key=len, reverse=True)[:12]:
        P = np.array([bsec.coords(k, TOP, inner, cache) for k in L])
        r = np.hypot(*P.T)
        print("loop", len(L), "r %.1f..%.1f" % (r.min(), r.max()))
    print("loops:", len(loops))
    pickle.dump((template, quads, inner, alias), open("secA_topo.pkl", "wb"))
