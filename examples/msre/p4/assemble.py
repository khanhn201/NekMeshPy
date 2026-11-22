"""core + bottom bands A..KN (+ lower head L2 if built) + top head -> one HexMesh."""
import os, sys, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "p1")]
from q2d import Sec                                              # noqa: F401
from nekmeshpy import hexmesh
from nekmeshpy.io.writer import to_vtu

t0 = time.time()
parts = {}
parts["core"] = pickle.load(open(os.path.join(HERE, "..", "p1", "upper_o%s_nz%s.pkl" % (os.environ.get("CORE_O", "2"), os.environ.get("CORE_NZ", "32"))), "rb"))
OUT = os.environ.get("OUT", "msre")
for b in ["A", "B", "C", "D", "E", "F3", "F2", "F1", "F0", "H", "I", "J", "KN"]:
    parts["band_" + b] = pickle.load(open(os.path.join(HERE, "band_%s.pkl" % b), "rb"))
L2 = os.path.join(HERE, "head_L2.pkl")
if os.path.exists(L2) and "--no-l2" not in sys.argv:
    parts["L2"] = pickle.load(open(L2, "rb"))
DC = os.path.join(HERE, "dc3.pkl")
if os.path.exists(DC):
    parts["downcomer"] = pickle.load(open(DC, "rb"))["mesh"]
VO = os.path.join(HERE, "volute.pkl")
if os.path.exists(VO):
    parts["volute"] = pickle.load(open(VO, "rb"))
parts["head_top"] = pickle.load(open(os.path.join(HERE, "..", "p5", "head_top.pkl"), "rb"))
PK = os.path.join(HERE, "..", "p5", "pocket.pkl")
if os.path.exists(PK) and "--no-pocket" not in sys.argv:
    parts["pocket"] = pickle.load(open(PK, "rb"))
for k, m in parts.items():
    print("%-10s %8d hexes" % (k, len(m.hexes)), flush=True)
mesh = hexmesh.merge(list(parts.values()), clear_seam_tags=True)
if not os.environ.get("NOFIX"):
    import hexfix
    mesh = hexfix.straighten_bad(mesh)
# every remaining name except the open interfaces is a wall
KEEP = ({"outlet", "KN_bot"} if "L2" not in parts else {"outlet"}) | {"inlet", "channel"}
dense = mesh.face_tags.to_dense()
mesh = hexmesh.retag_face(mesh, {t: "wall" for t in set(dense.tolist()) if t and t not in KEEP})
dense = mesh.face_tags.to_dense()
un = np.where(np.asarray(hexmesh.boundary_face_ids(mesh)) & (dense == ""))[0]
if len(un):
    mesh = hexmesh.tag_faces(mesh, un, "wall")
# end planes by geometry: every boundary face lying in the outlet plane (x = 635 at the branch) /
# the inlet end plane (pipe axis t = -635) gets that name, skin rings included
bfi = np.where(np.asarray(hexmesh.boundary_face_ids(mesh)))[0]
fcp = np.asarray(mesh.quad_mesh.points)[np.asarray(mesh.quad_mesh.corners)[bfi]]
out_f = bfi[(np.abs(fcp[..., 0] - 635.0) < 1e-6).all(1) & (fcp[..., 2] > 2400).all(1)]
PA = np.load(os.path.join(HERE, "pipe_axis.npy"))
tpa = (fcp - PA[:3]) @ PA[3:6]
in_f = bfi[(np.abs(tpa + 635.0) < 1e-6).all(1)]
mesh = hexmesh.tag_faces(mesh, out_f, "outlet")
mesh = hexmesh.tag_faces(mesh, in_f, "inlet")
dense = mesh.face_tags.to_dense()
for nm in ("inlet", "outlet"):
    ids_ = np.where(dense == nm)[0]
    ar = sum(0.5 * np.linalg.norm(np.cross(q[2] - q[0], q[3] - q[1])) for q in np.asarray(mesh.quad_mesh.points)[np.asarray(mesh.quad_mesh.corners)[ids_]])
    print("%s: %d faces, area %.0f mm2 (pipe section %.0f)" % (nm, len(ids_), ar, np.pi * (64.135 if nm == "inlet" else 64.1) ** 2), flush=True)
print("merged: %d hexes (%.0fs)" % (len(mesh.hexes), time.time() - t0), flush=True)
print(hexmesh.topology_report(mesh), flush=True)
print(hexmesh.tag_report(mesh), flush=True)
print("curved:", hexmesh.quality_summary(mesh), flush=True)
print("corner:", hexmesh.corner_summary(mesh), flush=True)
v = hexmesh.validity(mesh)
print("validity: inverted %d, twisted %d" % (len(v.inverted), len(v.twisted)), flush=True)
pickle.dump(mesh, open(os.path.join(HERE, OUT + ".pkl"), "wb"))
from nekmeshpy.io.writer import to_re2
if not len(v.inverted) and not len(v.twisted):
    to_re2(mesh, os.path.join(HERE, OUT + ".re2"), bc={"inlet": ("v  ", 1), "outlet": ("O  ", 2), "wall": ("W  ", 3), "channel": ("W  ", 4)})
    print("re2 written", flush=True)
if "--vtu" in sys.argv:
    to_vtu(mesh, os.path.join(HERE, OUT + ".vtu"))
print("done (%.0fs)" % (time.time() - t0))
