"""The top head carried down the side like the CAD: the pocket between the core can (R711.2) and the
vessel wall (R735.42) from the floor z1947.74 up, plus the ring above the can top (RC -> R711.2),
up to the dome's cap ring at z2070 (RC -> R2070).  A revolved (r, z) section on the core outline's
theta nodes; its inner side (r = RC, z 2014.48 -> 2070) welds to the core's outer faces, its top
(z 2070, RC -> R2070) to the dome skin's cap faces.  Can top taken at the core level 2014.48 (CAD
2016.25).  Brackets and rim screw holes are not modelled.  -> pocket.pkl / pocket.vtu"""
import os, sys, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "p1")]
from q2d import Sec
from nekmeshpy import quadmesh, hexmesh
from nekmeshpy.core.fields import gll_nodes

ORDER = int(os.environ.get("ORDER", "2"))
G = gll_nodes(ORDER)
RC, R_CAN, R_VW = 704.85, 711.2, 735.42265387044
Z_FLOOR, Z0 = 1947.74, 2070.0
CK, RK = np.array([646.556, 2019.62]), 88.867
NRP = int(os.environ.get("NRP", "2"))            # radial elements across the annulus
DZ = float(os.environ.get("POCKET_DZ", "14"))
PHI_TOP = np.radians(float(os.environ.get("PHI_TOP", "20")))
t0 = time.time()

core = pickle.load(open(os.path.join(HERE, "..", "p1", "upper_o%s_nz%s.pkl" % (os.environ.get("CORE_O", "2"), os.environ.get("CORE_NZ", "32"))), "rb"))
qm = core.quad_mesh
bf = np.where(np.asarray(hexmesh.boundary_face_ids(core)))[0]
cp = np.asarray(qm.points)[np.asarray(qm.corners)[bf]]
side = (np.abs(np.hypot(cp[..., 0], cp[..., 1]) - RC) < 1e-4).all(1)
pts = cp[side].reshape(-1, 3)
THETA = np.unique(np.round(np.arctan2(pts[:, 1], pts[:, 0]), 12))
ZL = np.unique(np.round(pts[:, 2], 6))
ZU = ZL[ZL >= 2014.0]                             # core levels on the interface (can top -> cap)
R2070 = CK[0] + np.sqrt(RK ** 2 - (Z0 - CK[1]) ** 2)
print("core outline: %d theta nodes; interface levels %s; R2070 %.3f" % (len(THETA), ZU.tolist(), R2070), flush=True)


def wall_pt(s):
    """outer wall by arclength from (R_VW, ZU[0]): vertical, then the knuckle."""
    lv = CK[1] - ZU[0]
    if s <= lv:
        return np.array([R_VW, ZU[0] + s])
    a = (s - lv) / RK
    return CK + RK * np.array([np.cos(a), np.sin(a)])


nu = len(ZU) - 1                                  # interface rows (5)
PHI_W = np.radians(float(os.environ.get("PHI_W", "22")))
L_w = (CK[1] - ZU[0]) + RK * PHI_W
S = Sec()
up = {}
for j, z in enumerate(ZU):                        # column 0 (r = RC) and the can line (711.2 -> R2070)
    f = (z - ZU[0]) / (Z0 - ZU[0])
    up[(0, j)] = S.P((RC, z))
    up[(1, j)] = S.P((R_CAN + f * (R2070 - R_CAN), z))
for j in range(nu):                               # U1: RC .. can line, all interface rows; top = cap
    S.quad((up[(0, j)], up[(1, j)], up[(1, j + 1)], up[(0, j + 1)]), "pocket")
# U2 (triangle can corner / wall / cap corner) as three quads: Q1 (can side), Q2 (wall side), Q3 (cap corner)
W = [S.P(wall_pt(L_w * j / (nu - 1))) for j in range(nu)]          # B .. W[nu-1] on the wall
Lc = [np.array(S.X[up[(1, j)]]) for j in range(nu + 1)]
Wc = [np.array(S.X[w]) for w in W]
cen = 0.25 * (Lc[nu - 1] + Wc[nu - 1] + Lc[nu - 2] + Wc[nu - 2])
mid0 = 0.5 * (Lc[0] + Wc[0])
Mi = [S.P(mid0 + (j / (nu - 1)) * (cen - mid0)) for j in range(nu)]
for j in range(nu - 1):
    S.quad((up[(1, j)], Mi[j], Mi[j + 1], up[(1, j + 1)]), "pocket")
    S.quad((Mi[j], W[j], W[j + 1], Mi[j + 1]), "pocket")
S.quad((up[(1, nu - 1)], Mi[nu - 1], W[nu - 1], up[(1, nu)]), "pocket")
for j in range(nu - 1):                           # wall edges
    za, zb = Wc[j][1], Wc[j + 1][1]
    if zb <= CK[1] + 1e-9:
        continue
    if za >= CK[1] - 1e-9:
        S.arc(W[j], W[j + 1], CK, RK)
    else:
        S.comp([W[j], np.array([R_VW, CK[1]]), W[j + 1]], [("line",), ("arc", CK[0], CK[1], RK)])
S.arc(W[nu - 1], up[(1, nu)], CK, RK)              # knuckle up to the cap corner (R2070, Z0)
up[(2, 0)], up[(3, 0)] = Mi[0], W[0]
NRP = 2
# lower annulus: R711.2 .. wall, floor -> ZU[0]
nl = max(2, int(round((ZU[0] - Z_FLOOR) / DZ)))
zl = np.linspace(Z_FLOOR, ZU[0], nl + 1)
lo = {}
for j, z in enumerate(zl):
    for k in range(NRP + 1):
        lo[(k, j)] = up[(1 + k, 0)] if j == nl else S.P((R_CAN + (R_VW - R_CAN) * k / NRP, z))
for j in range(nl):
    for k in range(NRP):
        S.quad((lo[(k, j)], lo[(k + 1, j)], lo[(k + 1, j + 1)], lo[(k, j + 1)]), "pocket")
sec = S.to_quadmesh(ORDER, 0.0)
print("pocket section: %d quads (%.0fs)" % (len(S.quads), time.time() - t0), flush=True)


def rev(th):
    return lambda P: np.c_[P[:, 0] * np.cos(th), P[:, 0] * np.sin(th), P[:, 1]]


th = THETA
sl = [quadmesh.transform_fn(sec, rev(t)) for t in th]
sw = []
for i in range(len(th)):
    a0, a1 = th[i], th[(i + 1) % len(th)] + (2 * np.pi if i == len(th) - 1 else 0)
    sw.append([quadmesh.transform_fn(sec, rev(a0 + t * (a1 - a0))) for t in G[1:-1]])
pocket = hexmesh.loft(sl, loop=True, sweep_nodes=sw, element_tags="fluid", first_tag="", last_tag="")
print("pocket: %d hexes, corner %s" % (len(pocket.hexes), hexmesh.corner_summary(pocket)), flush=True)
print(hexmesh.quality_summary(pocket))
top_h = pickle.load(open(os.path.join(HERE, "head_top.pkl"), "rb"))
for nm, other in (("core", core), ("head_top", top_h)):
    try:
        m = hexmesh.merge([other, pocket]); print("  merge ok with", nm, hexmesh.topology_report(m).n_components, "component(s)", flush=True)
    except ValueError as ex:
        print("  merge FAIL with", nm, str(ex)[:90], flush=True)
pickle.dump(pocket, open(os.path.join(HERE, "pocket.pkl"), "wb"))
from nekmeshpy.io.writer import to_vtu
to_vtu(pocket, os.path.join(HERE, "pocket.vtu"))
print("done (%.0fs)" % (time.time() - t0))
