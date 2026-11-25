"""Boundary-condition coverage of a .re2: every open face has exactly one BC row (no leak), no row on an
interior face, IDs used, and boundary faces lying on top of each other (a hidden internal wall)."""
import sys, collections
import numpy as np
from scipy.spatial import cKDTree
from nekmeshpy import HexMesh

fn = sys.argv[1]
with open(fn, "rb") as f:
    hdr = f.read(80).decode(); E = int(hdr[5:21]); nbc = int(hdr[40:44]) if len(hdr) > 44 and hdr[40:44].strip() else 1
    f.read(4)
    X = np.frombuffer(f.read(E * 25 * 8), "<f8").reshape(E, 25)[:, 1:].reshape(E, 3, 8).transpose(0, 2, 1)
    ncurve = int(np.frombuffer(f.read(8), "<f8")[0])
    f.read(ncurve * 8 * 8)
    blocks = []
    for b in range(nbc):
        raw = f.read(8)
        if len(raw) < 8:
            break
        n = int(np.frombuffer(raw, "<f8")[0])
        rows = np.frombuffer(f.read(n * 8 * 8), "<f8").reshape(n, 8)
        blocks.append(rows)
print("header: %s" % hdr.strip())
print("%d elements, %d BC block(s)" % (E, len(blocks)))
P, inv = np.unique(X.reshape(-1, 3), axis=0, return_inverse=True)
H = inv.reshape(E, 8)
FP = HexMesh.FACE_POINTS                      # Nek face (1-6) -> 4 corner slots
faces = H[:, FP]                              # (E,6,4)
key = np.sort(faces.reshape(-1, 4), axis=1)
_, finv, fcnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
open_ = (fcnt[finv] == 1).reshape(E, 6)
print("open (boundary) faces: %d; interior faces: %d; faces shared by >2: %d" % (open_.sum(), (fcnt == 2).sum(), (fcnt > 2).sum()))
rows = blocks[0]
el = rows[:, 0].astype(int) - 1; fc = rows[:, 1].astype(int) - 1
bid = rows[:, 6].astype(int)
code = np.array([r.tobytes()[56:64].decode("ascii", "replace").strip() for r in rows])
has = np.zeros((E, 6), int)
np.add.at(has, (el, fc), 1)
print("BC rows: %d" % len(rows))
print("  by boundaryID / code:", sorted(collections.Counter(zip(bid.tolist(), code.tolist())).items()))
print("  open faces with NO row (leaks): %d" % (open_ & (has == 0)).sum())
print("  open faces with >1 row: %d" % (has > 1).sum())
print("  rows on interior faces: %d" % (~open_[el, fc]).sum())
if (open_ & (has == 0)).any():
    e_, f_ = np.where(open_ & (has == 0))
    c = X[e_][np.arange(len(e_))[:, None], FP[f_]].mean(1)
    print("  first unassigned faces at:", np.round(c[:8], 1).tolist())
# boundary faces on top of each other (distinct faces, same place): an internal wall inside the fluid
e_, f_ = np.where(open_)
C = X[e_][np.arange(len(e_))[:, None], FP[f_]]
cen = C.mean(1); sz = np.linalg.norm(C[:, 0] - C[:, 2], axis=1)
pairs = cKDTree(cen).query_pairs(r=1e-6 * max(1.0, np.abs(cen).max()), output_type="ndarray")
print("coincident boundary-face pairs (internal walls): %d" % len(pairs))
for a, b in pairs[:8]:
    print("   at", np.round(cen[a], 2).tolist(), "elements", e_[a] + 1, e_[b] + 1)
