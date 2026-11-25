"""Undo node moves that fold elements between their nodes: hexes inverted when the trilinear map is sampled at
ORDER (the solver's) get their nodes put back to BASE (same connectivity), repeated until clean at ORDER and
overlap-free.  usage: revert_sampled.py base.pkl new.pkl out.pkl   env ORDER (default SCAN_ORDER)"""
import os, sys, pickle, time
import numpy as np
from nekmeshpy import hexmesh
from nekmeshpy.core import quality as Q, topology as T
from nekmeshpy.core.interp import sampled_scaled_jacobian
from nekmeshpy.hexmesh import smoother as hs

t0 = time.time()
base = pickle.load(open(sys.argv[1], "rb")); new = pickle.load(open(sys.argv[2], "rb"))
X0 = np.asarray(base.points, float); X = np.asarray(new.points, float).copy(); H = np.asarray(new.corners)
assert np.array_equal(np.asarray(base.corners), H), "connectivity differs"
N = int(os.environ.get("ORDER", Q.SCAN_ORDER))


from nekmeshpy.core.interp import corner_indices
CI = corner_indices(1, 3)


def sampled(Xa):
    blk = np.empty((len(H), 8, 3)); blk[:, CI, :] = Xa[H]        # the .re2 trilinear map (as the toolkit reads it)
    return sampled_scaled_jacobian(blk, 1, N, 3)


sb = sampled(X0)
print("base: sampled at N=%d min %.4f, %d inverted (%.0fs)" % (N, sb.min(), (sb <= 0).sum(), time.time() - t0), flush=True)
for it in range(10):
    s = sampled(X)
    bad = np.where(s <= 0)[0]
    O = T.overlapping_pairs(X, H)
    print("pass %d: %d inverted at N=%d (min %.4f), %d overlapping pairs (%.0fs)" % (it, len(bad), N, s.min(), len(O), time.time() - t0), flush=True)
    if not len(bad) and not len(O):
        break
    hexes = np.unique(np.r_[bad, O.ravel()])
    nodes = np.unique(H[hexes])
    moved = nodes[np.linalg.norm(X[nodes] - X0[nodes], axis=1) > 0]
    if not len(moved):
        print("  not caused by moved nodes -- stopping"); break
    X[moved] = X0[moved]
    print("  reverted %d nodes" % len(moved), flush=True)
pickle.dump(hs.with_points(new, X), open(sys.argv[3], "wb"))
print("done (%.0fs)" % (time.time() - t0))
