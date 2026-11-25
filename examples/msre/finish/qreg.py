"""Per-region corner scaled Jacobian + min edge length of a linear .re2 (whole model).
usage: qreg.py model.re2 [worst-N]  -> prints the table, the worst elements, saves <model>_q.npz"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p4"))
from re2check import read_re2
from nekmeshpy.hexmesh.quality import corner_scaled_jacobian
X = read_re2(sys.argv[1]); E = len(X)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 25
sj = np.concatenate([np.asarray(corner_scaled_jacobian(X[s:s + 200000].reshape(-1, 3), np.arange(min(200000, E - s) * 8).reshape(-1, 8))) for s in range(0, E, 200000)])
E12 = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
h = np.linalg.norm(X[:, E12[:, 0]] - X[:, E12[:, 1]], axis=2)
hmin, hmax = h.min(1), h.max(1)
c = X.mean(1); r = np.hypot(c[:, 0], c[:, 1]); z = c[:, 2]
reg = np.full(E, "other", object)
reg[z < 300] = "lower head (off vanes)"
reg[(z < 300) & (r > 455) & (r < 745)] = "lower head: vane annulus"
reg[(z >= 300) & (z < 398.6)] = "bottom bands B-KN"
reg[(z >= 300) & (z < 398.6) & (r > 670) & (r < 710)] = "bottom bands: rim"
reg[(z >= 398.6) & (z < 411.3)] = "band A (plenum)"
reg[(z >= 411.3) & (z < 1990)] = "core"
reg[z >= 1990] = "top head + pocket"
reg[(r > 745) & (z > 290) & (z < 2000)] = "downcomer / volute / inlet"
G = ["lower head: vane annulus", "lower head (off vanes)", "bottom bands B-KN", "bottom bands: rim", "band A (plenum)",
     "core", "top head + pocket", "downcomer / volute / inlet", "other"]
print("%d hexes; corner sj min %.3f, inverted %d; min edge %.4f" % (E, sj.min(), (sj <= 0).sum(), hmin.min()))
print("%-28s %9s | %6s %6s %6s | %6s %6s %6s | %8s %8s" % ("region", "hexes", "sj<.05", "sj<.1", "sj<.2", "h<.1", "h<.3", "h<.5", "min sj", "min h"))
for g in G:
    m = reg == g
    if m.any():
        print("%-28s %9d | %6d %6d %6d | %6d %6d %6d | %8.3f %8.3f" % (g, m.sum(), (sj[m] < .05).sum(), (sj[m] < .1).sum(), (sj[m] < .2).sum(),
              (hmin[m] < .1).sum(), (hmin[m] < .3).sum(), (hmin[m] < .5).sum(), sj[m].min(), hmin[m].min()))
for name, key in (("corner sj", sj), ("min edge", hmin)):
    print("worst by %s:" % name)
    for i in np.argsort(key)[:N]:
        print("  #%d sj %.3f hmin %.3f hmax %.1f  r %.1f th %.1f z %.1f  %s" % (i, sj[i], hmin[i], hmax[i], r[i], np.degrees(np.arctan2(c[i, 1], c[i, 0])), z[i], reg[i]))
np.savez(os.path.splitext(sys.argv[1])[0] + "_q.npz", sj=sj, hmin=hmin, hmax=hmax, c=c)
