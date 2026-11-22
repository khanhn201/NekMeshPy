"""Exact bore axes from the STEP: per r9.525 cylinder face, axis direction (normals) and the axis
line point at the row height (circle fit in the plane normal to the axis). -> p4/bore_axes.npy
rows: cx, cy, cz, ax, ay, az, radius"""
import gmsh, numpy as np, pickle
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
import os
STEP = os.environ.get("MSRE_STEP", "core_fluid_cleanup_v5.STEP")
gmsh.model.occ.importShapes(STEP); gmsh.model.occ.synchronize()
rec = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces.pkl"), "rb"))
rows = []
for s, t, a, c, bb in rec:
    if not (t == "Cylinder" and abs(a - 1766.7) < 1 and 1600 < c[2] < 1950):
        continue
    lo, hi = gmsh.model.getParametrizationBounds(2, s)
    UV = np.array([(u, v) for u in np.linspace(lo[0], hi[0], 16) for v in np.linspace(lo[1], hi[1], 6)]).ravel()
    P = np.array(gmsh.model.getValue(2, s, UV)).reshape(-1, 3)
    N = np.array(gmsh.model.getNormal(s, UV)).reshape(-1, 3)
    ax = np.linalg.svd(N)[2][-1]
    if ax @ np.array([c[0], c[1], 0.0]) < 0:
        ax = -ax
    e1 = np.cross([0, 0, 1.0], ax); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
    q = np.c_[P @ e1, P @ e2]
    A = np.c_[2 * q, np.ones(len(q))]; b = (q ** 2).sum(1)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    r = np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)
    cen = sol[0] * e1 + sol[1] * e2                     # axis point (component along ax = 0)
    rows.append(np.r_[cen, ax, r])
rows = np.array(rows)
o = np.lexsort((np.arctan2(rows[:, 1] + rows[:, 4] * 760, rows[:, 0] + rows[:, 3] * 760), np.round(rows[:, 2], 1)))
rows = rows[o]
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p4", "bore_axes.npy"), rows)
print(len(rows), "bores; radius %.4f..%.4f; z rows" % (rows[:, 6].min(), rows[:, 6].max()), sorted(set(np.round(rows[:, 2], 3))))
print("axis z-components max %.2e" % np.abs(rows[:, 5]).max())
gmsh.finalize()
