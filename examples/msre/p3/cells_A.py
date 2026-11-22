"""Cell categories for band A full coverage."""
import numpy as np
import bsec, rules
from geom import CENTRAL, H

D4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
R_WALL = rules.R_RING                       # 692.15, band A's outer wall
BORES = {(1, 1), (-1, 1), (-1, -1), (1, -1)}


def corners_R(c):
    cc = np.array(c, float) * rules.P
    return max(np.hypot(*(cc + H * np.array(s))) for s in ((1, 1), (1, -1), (-1, 1), (-1, -1)))


def template_cells(margin=2.0):
    """full stringer cells the plain template can take in band A: not central, not next to a bore,
    whole cell inside the outer wall."""
    out = []
    for c, t in bsec.fp.bs.CELLS.items():
        if t != "full" or c in CENTRAL:
            continue
        if corners_R(c) > R_WALL - margin:
            continue
        out.append(c)
    return out


if __name__ == "__main__":
    tc = template_cells()
    allfull = [c for c, t in bsec.fp.bs.CELLS.items() if t == "full"]
    print("template cells", len(tc), "of", len(allfull), "full; regular", len(bsec.regular_cells()))
    rest = [c for c in allfull if c not in set(tc)]
    print("full cells left for special handling:", len(rest), sorted(rest)[:12])
