"""global + local untangling of a whole mesh with every boundary node fixed
usage: python outer_untangle.py SRC.npz OUT.npz"""
import os
import sys, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psmooth import global_untangle, untangle, merit, edge_stats
from inflate import face_table
D = np.load(sys.argv[1]); X, H = D["X"].copy(), D["H"]
t = time.time()
ft = face_table(H)
bnd = set(int(v) for k, v in ft.items() if len(v) == 1 for v in k)
sj = merit(X[H]); print("start: inverted %d min %.3f" % ((sj < 0).sum(), sj.min()), flush=True)
for tau in (0.2, 0.25, 0.3):
    X = global_untangle(X, H, bnd, set(), tau=tau, lam=0.005, iters=800, mu=1.0, lfrac=0.25)
X, sj = untangle(X, H, bnd, set(), thr=0.15, sweeps=60, verbose=True, lfrac=0.25)
sj = merit(X[H]); e, a = edge_stats(X, H)
print("final: inverted %d, min %.3f p1 %.3f median %.3f, min edge %.3f max aspect %.1f (%.0fs)" % ((sj < 0).sum(), sj.min(), np.percentile(sj, 1), np.median(sj), e, a, time.time() - t), flush=True)
out = dict(D); out["X"] = X
np.savez(sys.argv[2], **out)
