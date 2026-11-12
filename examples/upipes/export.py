"""final mesh -> tagged nekmeshpy HexMesh (+ validity, .vtu)"""
import os
import sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from nekmeshpy import hexmesh
from nekmeshpy.hexmesh.hexmesh import HexMesh
from nekmeshpy.hexmesh.query import boundary_face_ids
from geom import PLATE
from tubes import TUBES, BOX

src = sys.argv[1]
D = np.load(src + ".npz"); X, H = D["X"], D["H"]
used = np.unique(H); remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
X, H = X[used], remap[H]
mesh = HexMesh.from_corners(X, H)
q = mesh.quad_mesh
fid = np.where(boundary_face_ids(mesh))[0]
C = q.points[q.corners[fid]]                       # (nb, 4, 3)
lo, hi = BOX
names = np.full(len(fid), "", dtype=object)
on = lambda ax, v: np.all(np.abs(C[..., ax] - v) < 1e-6, 1)
names[on(1, PLATE)] = "plate"
for nm, ax, v in (("x_min", 0, lo[0]), ("x_max", 0, hi[0]), ("z_min", 2, lo[2]), ("z_max", 2, hi[2]), ("top", 1, hi[1])):
    names[on(ax, v)] = nm
dmin = np.full(C.shape[:2], np.inf)
for p, T in TUBES.items():
    dmin = np.minimum(dmin, np.abs(T.dist(C.reshape(-1, 3))[0]).reshape(C.shape[:2]))
names[np.all(dmin < 5e-3, 1)] = "pipes"     # pipes.txt spines have ~1e-3 steps at bend junctions
bad = names == ""
if bad.any():
    print("untagged:", bad.sum()); print(C[bad][:4].round(4)); print(dmin[bad][:4])
    raise SystemExit
mesh = hexmesh.tag_faces(mesh, fid, names.astype(str))
from collections import Counter
print("boundary faces:", dict(Counter(names)))
v = hexmesh.validity(mesh) if hasattr(hexmesh, "validity") else None
print("validity:", v)
print(hexmesh.quality_summary(mesh) if hasattr(hexmesh, "quality_summary") else "")
from nekmeshpy.io.writer import to_vtu
to_vtu(mesh, src + "_tagged.vtu", bc={n: "W  " for n in set(names)})
import pickle; pickle.dump(mesh, open(src + "_hexmesh.pkl", "wb"))
