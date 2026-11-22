"""Inlet pipe from the STEP: cylinder axis/radius, end plane, and its trim curve on the torus."""
import gmsh, numpy as np
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
import os
STEP = os.environ.get("MSRE_STEP", "core_fluid_cleanup_v5.STEP")
gmsh.model.occ.importShapes(STEP); gmsh.model.occ.synchronize()
def sample(tag, n=30):
    lo, hi = gmsh.model.getParametrizationBounds(2, tag)
    UV = np.array([(u, v) for u in np.linspace(lo[0], hi[0], n) for v in np.linspace(lo[1], hi[1], n)]).ravel()
    return np.array(gmsh.model.getValue(2, tag, UV)).reshape(-1, 3), np.array(gmsh.model.getNormal(tag, UV)).reshape(-1, 3)
P, N = sample(14097)
ax = np.linalg.svd(N)[2][-1]
e1 = np.cross([0, 0, 1.0], ax); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
q = np.c_[P @ e1, P @ e2]
A = np.c_[2 * q, np.ones(len(q))]; b = (q ** 2).sum(1)
sol = np.linalg.lstsq(A, b, rcond=None)[0]
r = np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2); c = sol[0] * e1 + sol[1] * e2
t = (P - c) @ ax
print("pipe: axis", ax.round(6), "point", c.round(4), "r %.4f" % r, "t range %.2f..%.2f" % (t.min(), t.max()))
Pp, Np = sample(14103, 10)
n = Np[0]; print("end plane normal", n.round(6), "point", Pp.mean(0).round(3), "axis.normal %.6f" % abs(n @ ax), "t at plane %.3f" % ((Pp.mean(0) - c) @ ax))
# trim curve: boundary curves of the pipe face
bc = gmsh.model.getBoundary([(2, 14097)], oriented=False)
for dim, tag in bc:
    lo, hi = gmsh.model.getParametrizationBounds(1, abs(tag))
    X = np.array(gmsh.model.getValue(1, abs(tag), np.linspace(lo[0], hi[0], 200))).reshape(-1, 3)
    tt = (X - c) @ ax
    rr = np.hypot(X[:, 0], X[:, 1])
    print("  boundary curve", tag, gmsh.model.getType(1, abs(tag)), "t %.2f..%.2f" % (tt.min(), tt.max()), "r %.2f..%.2f" % (rr.min(), rr.max()), "z %.2f..%.2f" % (X[:, 2].min(), X[:, 2].max()))
    pass
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p4", "pipe_axis.npy"), np.r_[c, ax, r])
gmsh.finalize()
