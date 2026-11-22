"""Band-A fluid selection on the unified section + its outlines."""
import pickle, collections
import numpy as np
import bsec, rules, secA

R_W = rules.R_RING
SOLID = ("disc", "bk_AB", "bk_BC")
SITES = set(rules.sites())


def select(quads, inner):
    TOP = dict(z=rules.Z["top"], ring=rules.R_PIN)
    cache = {}
    sel = np.zeros(len(quads), bool)
    cen = np.zeros((len(quads), 2))
    for i, (q, lab, c) in enumerate(quads):
        P = np.array([bsec.coords(k, TOP, inner, cache) for k in q])
        cen[i] = P.mean(0)
        if lab == "disc" and not (isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], int) and c in SITES):
            solid = False                          # no pin at this site: the disc is fluid
        else:
            solid = lab in SOLID
        sel[i] = (not solid) and np.hypot(*cen[i]) < R_W
        # a pin wholly inside the wall keeps its whole ring (the snap squeezes it to the wall)
        if lab == "oring" and c in SITES and np.hypot(*np.array(c, float)) * rules.P + rules.R_PIN < R_W:
            sel[i] = True
    return sel, cen


if __name__ == "__main__":
    template, quads, inner = secA.collect()
    quads, inner, alias = secA.unify(quads, inner)
    sel, cen = select(quads, inner)
    print("quads", len(quads), "selected", sel.sum())
    loops = secA.boundary_loops([q for q, s in zip(quads, sel) if s])
    TOP = dict(z=rules.Z["top"], ring=rules.R_PIN); cache = {}
    for L in sorted(loops, key=len, reverse=True)[:6]:
        P = np.array([bsec.coords(k, TOP, inner, cache) for k in L]); r = np.hypot(*P.T)
        print("loop", len(L), "r %.1f..%.1f" % (r.min(), r.max()))
    print("loops", len(loops))
    pickle.dump((template, quads, inner, alias, sel), open("selA.pkl", "wb"))
