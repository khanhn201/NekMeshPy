"""Smooth the bore interiors on the LINEAR section (toolkit smoother): wall circles frozen, helper
rings + inner square free; helper-ring edges become straight (not walls), wall arcs stay exact."""
import pickle, numpy as np
from q2d import Sec
from geom import PITCH, THIMBLES, R_THIM, R_BASKET
from nekmeshpy.quadmesh.quadmesh import QuadMesh
from nekmeshpy.quadmesh.smoother import constrain as cn
from nekmeshpy.quadmesh.smoother import smooth

S = pickle.load(open("section_presmooth.pkl", "rb"))
X = np.array(S.X); Q = np.array(S.quads); lab = np.array(S.lab)
bore = np.array([l.startswith("th_") or l.startswith("bk_") for l in lab])
inb = np.zeros(len(X), bool); inb[np.unique(Q[bore])] = True
walls = {R_THIM: [12.7, R_THIM], R_BASKET: [7.5038, 12.7, 25.79, R_BASKET]}
helper = {R_THIM: [22.0], R_BASKET: []}
onwall = np.zeros(len(X), bool)
helper_nodes = set()
for c, rt in THIMBLES.items():
    cc = np.array(c, float) * PITCH
    r = np.hypot(*(X - cc).T)
    for R in walls[rt]:
        onwall |= inb & (np.abs(r - R) < 1e-6)
    for R in helper[rt]:
        helper_nodes |= set(np.where(inb & (np.abs(r - R) < 1e-6))[0].tolist())
free = inb & ~onwall
qm = QuadMesh.from_corners(np.c_[X, np.zeros(len(X))], Q)
res = cn.resolve(qm, [cn.Frozen(np.where(~free)[0])])
out, _ = smooth.run_schedule(qm, res, smooth.user_schedule(outer=10), normals=np.tile([0.0, 0.0, 1.0], (len(Q), 1)))
Y = np.asarray(out.points)[:, :2]
# helper-ring arcs no longer hold (nodes moved): make those edges straight
for k in list(S.cur):
    a, b, spec = S.cur[k]
    if a in helper_nodes and b in helper_nodes and spec[0] == "arc":
        del S.cur[k]
S.X = [tuple(p) for p in Y]
print("bore nodes moved max %.2f mm" % np.abs(Y - X)[free].max())
pickle.dump(S, open("section.pkl", "wb"))
