"""Global bottom-zone section topology (all bands A..KN), r < R692.15:
lattice template on every cell (incl. the 5 central cells, T-cells, edge cells) + the core's bore
quads; quads with centroid inside the wall, cleaned (max-2 rule), outer outline pillowed."""
import os, sys, pickle, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "p3"), os.path.join(HERE, "..", "p1")]
import bsec, rules, secA, cells_A, edgeA, wallA
from geom import CENTRAL

R_W = rules.R_RING
TOP = dict(z=rules.Z["top"], ring=rules.R_PIN)


def main():
    cells = cells_A.template_cells() + edgeA.edge_cells() + sorted(CENTRAL)
    quads, inner = bsec.build(cells)
    X = np.array(secA.CORE.X)
    for qi, (q, lab) in enumerate(zip(secA.CORE.quads, secA.CORE.lab)):
        if lab in secA.BORE_LABS:
            quads.append((tuple(secA.ckey(X[i]) for i in q), lab, ("bore", qi)))
    quads, inner, alias = secA.unify(quads, inner)
    # orientation from the real top-level layout (central plug ring inside the bore crescents)
    import levels as lv
    core_xy = {secA.ckey(p): tuple(p) for p in secA.CORE.X}
    cache = {}
    L = lv.LV["top"]
    fixed = []
    for q, lab, c in quads:
        P = np.array([lv.coords(k, L, inner, cache, core_xy) for k in q])
        a = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        fixed.append((q if a > 0 else q[::-1], lab, c))
    quads = fixed
    cache = {}
    cen = np.array([np.mean([bsec.coords(k, TOP, inner, cache) for k in q], 0) for q, _, _ in quads])
    labs = np.array([l for _, l, _ in quads])
    sel = np.hypot(*cen.T) < R_W
    SITES = set(rules.sites())
    inwall = lambda c: isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], int) and c in SITES and \
        np.hypot(*np.array(c, float)) * rules.P + rules.R_PIN < R_W
    locked_in = (labs == "chan") | np.array([inwall(c) and l in ("oring", "disc") for _, l, c in quads])
    sel[locked_in] = True
    solid = np.zeros(len(quads), bool)
    sel = wallA.clean(quads, sel.copy(), cen, solid, locked_in)
    loops = secA.boundary_loops([q for q, s in zip(quads, sel) if s])
    loops.sort(key=len, reverse=True)
    print("loops:", [len(L) for L in loops[:4]], "...", len(loops))
    outer = loops[0]
    quads2, sel2, npil = wallA.pillow_turns(quads, sel, outer)
    G = [quads2[i] for i in range(len(quads2)) if sel2[i]]
    print("global section: %d quads, outline %d nodes, pillowed %d, other loops %d" % (len(G), len(outer), npil, len(loops) - 1))
    print(collections.Counter(l for _, l, _ in G))
    pickle.dump(dict(cells=cells, quads=G, inner=inner, alias=alias, outer=outer), open(os.path.join(HERE, "topo.pkl"), "wb"))


if __name__ == "__main__":
    main()
