"""Band-A edge, structured: lattice template on every cell meeting the wall R692.15, fluid quads
inside the wall selected, the outer outline snapped onto the wall."""
import collections, pickle
import numpy as np
import bsec, rules, cells_A
from geom import CENTRAL, H

R_W = rules.R_RING


def edge_cells():
    tset = set(cells_A.template_cells())
    out = []
    for c, t in bsec.fp.bs.CELLS.items():
        if c in tset or c in CENTRAL or t not in ("full", "cut"):
            continue
        cc = np.array(c, float) * rules.P
        rmin = max(0.0, np.hypot(max(abs(cc[0]) - H, 0), max(abs(cc[1]) - H, 0)))
        if rmin < R_W:
            out.append(c)
    return out


if __name__ == "__main__":
    ec = edge_cells()
    print("edge cells", len(ec), collections.Counter(bsec.fp.bs.CELLS[c] for c in ec))
    quads, inner = bsec.build(ec)
    TOP = dict(z=rules.Z["top"], ring=rules.R_PIN); BOT = dict(z=rules.Z["AB"], ring=rules.R_PIN)
    a0 = bsec.check(quads, inner, TOP); sg = np.sign(a0)
    a1 = bsec.check(quads, inner, BOT)
    print("quads", len(quads), "TOP min|a| %.3g  BOT flips %d" % (np.abs(a0).min(), ((np.sign(a1) != sg) | (np.abs(a1) < 1e-9)).sum()))
