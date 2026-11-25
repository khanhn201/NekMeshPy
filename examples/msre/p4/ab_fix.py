"""AB level: after the prism copy, untangle the footprint nodes next to the kept O-grids."""
import numpy as np, positions as PS, levels as lv
from levels import bsec
from q2 import corner_sj, untangle


def fix(X, L, kept=None):
    kept = PS.AB_PRISM if kept is None else kept
    INV = PS.INV
    free = np.zeros(len(X), bool)
    oset = set(PS.KEYS[k] for k in PS.OUTER)
    con = {}
    for i, k in enumerate(INV):
        if i in oset:
            if getattr(PS, "LOFT", False):               # loft rim: wall nodes slide on the wall
                free[i] = True; con[i] = ("wall",)
            continue
        if isinstance(k[0], (float, np.floating)) and k in lv.ROLE:
            r, c2, s = lv.ROLE[k]
            if r in ("Bm", "Bp", "Ym", "Y0", "Yp", "M"):
                free[i] = True
            elif r in ("Am", "Ap", "Im", "I0", "Ip"):
                jn = bsec.side_lines(c2, s)[1]
                lp = lv.line_pos(jn, L)
                if np.isfinite(lp) and abs(lp) < lv.R_W:        # a slot line outside the lattice: leave it be
                    free[i] = True; con[i] = ("line", jn[0], lp)
        elif isinstance(k[0], str) and k[0] == "ring" and k[1] not in kept:
            free[i] = True
            if not getattr(PS, "LOFT", False) or lv.ring_held(k[1]):
                con[i] = ("circ", np.array(k[1], float) * lv.PITCH, lv.ring_radius(k[1], L), k[1])
            else:
                con[i] = ("in",)                         # a template ring: free, inside the wall
        elif isinstance(k[0], str) and k[0] in ("sqb", "sqi", "mq", "mj", "pw") and k[1] not in kept:
            free[i] = True

    def proj(i, p):
        c = con.get(i)
        if c is None:
            return p
        if c[0] == "wall":
            return lv.R_W * p / np.linalg.norm(p)
        if c[0] == "in":
            r = np.hypot(*p); return p if r <= lv.R_W - 1.5 else p * (lv.R_W - 1.5) / r
        if c[0] == "circ":
            if getattr(PS, "LOFT", False):
                return lv.clamp_ring(c[3], L, p)
            d = p - c[1]; return c[1] + c[2] * d / np.linalg.norm(d)
        p = p.copy()
        if c[1] == "x":
            p[0] = c[2]
        else:
            p[1] = c[2]
        return p
    import os
    if getattr(PS, "LOFT", False):                       # loft rim: only the rim needs this
        free &= np.hypot(X[:, 0], X[:, 1]) > lv.R_W - 40
    return untangle(X, PS.Q, free, proj, target=0.1, rounds=int(os.environ.get("ABF_ROUNDS", "300")),
                    grow=int(os.environ.get("ABF_GROW", "3")))


if __name__ == "__main__":
    X = np.load("pos_AB.npz")["X"]
    Y = fix(X, lv.LV["AB"])
    s = corner_sj(Y, PS.Q)
    print("min csj %.3f n<=0 %d n<0.05 %d" % (s.min(), (s <= 0).sum(), (s < 0.05).sum()))
    np.savez("pos_AB_try.npz", X=Y)
