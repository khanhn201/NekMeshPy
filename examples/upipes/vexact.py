"""exact 4-governor vertices (Newton on 3 equalities) for merged graph vertices"""
import numpy as np, pickle
from tubes import TUBES, BOX
from geom import wall_dist

def dist_grad(g, p):
    p = np.atleast_2d(p)
    if g >= 0:
        d, s = TUBES[g].dist(p); v = p - TUBES[g].spine(s)
        return d[0], (v / np.linalg.norm(v, axis=1, keepdims=True))[0]
    lo, hi = BOX
    ax, val, sg = {-1: (0, lo[0], 1), -2: (0, hi[0], -1), -3: (2, lo[2], 1), -4: (2, hi[2], -1), -5: (1, hi[1], -1)}[g]
    gr = np.zeros(3); gr[ax] = sg
    return sg * (p[0, ax] - val), gr

def solve(govs, p0, it=50):
    p = np.asarray(p0, float).copy()
    for _ in range(it):
        dg = [dist_grad(g, p) for g in govs]
        f = np.array([dg[0][0] - dg[k][0] for k in range(1, 4)])
        J = np.array([dg[0][1] - dg[k][1] for k in range(1, 4)])
        dp = -np.linalg.lstsq(J, f, rcond=None)[0]
        p += np.clip(dp, -1, 1)
        if np.linalg.norm(dp) < 1e-10: break
    return p, max(abs(f))

if __name__ == "__main__":
    E, V = pickle.load(open("maps/graph2.pkl", "rb"))
    worst = []
    for i, v in enumerate(V):
        if v["plate"] or len(v["gov"]) != 4 or any(g >= 1000 for g in v["gov"]): continue
        if not any(e.get("alive", True) and i in e["v"] for e in E): continue
        p, r = solve(v["gov"], v["X"])
        worst.append((np.linalg.norm(p - v["X"]), i, v["gov"], r))
    worst.sort(reverse=True)
    for w in worst[:12]: print("vertex %d %s moved %.2f mm (residual %.1e)" % (w[1], w[2], w[0], w[3]))
