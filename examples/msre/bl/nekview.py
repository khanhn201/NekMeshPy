"""The geometry Nek builds from corners + 'm' edge midpoints: every face and the centre by Gordon-Hall
(Coons) from the edges.  Sample it at N=2 and SAMPLE_ORDER (7).  usage: nekview.py mesh.pkl"""
import sys, os, pickle, numpy as np
from nekmeshpy import hexmesh
from nekmeshpy.core import interp
m = pickle.load(open(sys.argv[1], "rb"))
blk = np.array(hexmesh.element_blocks(m))
lat = np.array([(i, j, k) for k in range(3) for j in range(3) for i in range(3)])
n1 = (lat == 1).sum(1)
for f in np.where(n1 == 2)[0]:                                     # face centres from their 4 edges and 4 corners
    ax = int(np.where(lat[f] != 1)[0][0]); v = lat[f][ax]
    on = (lat[:, ax] == v)
    e = np.where(on & (n1 == 1))[0]; c = np.where(on & (n1 == 0))[0]
    blk[:, f] = 0.5 * blk[:, e].sum(1) - 0.25 * blk[:, c].sum(1)
blk[:, 13] = 0.5 * blk[:, n1 == 2].sum(1) - 0.25 * blk[:, n1 == 1].sum(1) + 0.125 * blk[:, n1 == 0].sum(1)
for N in (2, int(os.environ.get("SAMPLE_ORDER", 7))):
    sj = np.asarray(interp.sampled_scaled_jacobian(blk, 2, N, 3))
    print("Nek view (edges only), N=%d: min %+.4f, %d inverted" % (N, sj.min(), (sj <= 0).sum()), flush=True)
