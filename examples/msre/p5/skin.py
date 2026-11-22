"""One skin layer from the inner head volume's free boundary onto the CAD surfaces.
Every stored boundary node (corner / edge / face GLL node) gets a target from the classes of the
faces it touches, so shared edges land on intersection curves.  -> head_top.pkl / .vtu"""
import os, sys, pickle, time, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import head_top as HT
from head_top import (Z0, RC, RP, ZF, ZT, ZB, RB, XE, CK, RK, THIM, key2)
from nekmeshpy import quadmesh, hexmesh
from nekmeshpy.linemesh import LineMesh
from nekmeshpy.pointmesh import PointMesh
from nekmeshpy.quadmesh import QuadMesh

t0 = time.time()
D = pickle.load(open(os.path.join(HERE, "inner.pkl"), "rb"))
inner, TH, LP, ta, tt = D["inner"], D["TH"], D["LP"], D["ta"], D["tt"]
ZB0, ZT0 = ZF - HT.D0, ZT
Zs = HT.Zs
YLO, YHI = D["ysq"]
R2070 = CK[0] + np.sqrt(RK ** 2 - (Z0 - CK[1]) ** 2)

Bq = hexmesh.boundary_mesh(inner)
lm = Bq.line_mesh
PTS = np.asarray(lm.points); LINES = np.asarray(lm.lines)
LI = np.asarray(lm.interior); FI = np.asarray(Bq.interior)
FQ = np.asarray(Bq.quads); FC = np.asarray(Bq.corners)
print("boundary: %d faces (%.0fs)" % (len(FQ), time.time() - t0), flush=True)

# ---- face classes
cp = PTS[FC]                                   # (F,4,3)
r_ = np.hypot(cp[..., 0], cp[..., 1])
outl = np.array([[key2(p) in TH for p in f] for f in cp])
cls = np.full(len(FC), "dome", dtype=object)
th_hit = np.zeros(len(FC), bool)
for c, rr in THIM:
    th_hit |= (np.abs(np.hypot(cp[..., 0] - c[0], cp[..., 1] - c[1]) - rr) < 1e-6).all(1)
cls[(np.abs(r_ - RC) < 1e-6).all(1)] = "rim"
cls[outl.all(1) & (cp[..., 2] > ZB0 + 1e-6).any(1)] = "pipe"
cls[(cp[..., 2] > 2400).all(1) & (cp[..., 0] > 130).any(1)] = "duct"
cls[(np.abs(cp[..., 2] - ZT0) < 1e-6).all(1)] = "top"
cls[th_hit] = "thim"
cls[(np.abs(cp[..., 0] - XE) < 1e-6).all(1)] = "outlet"
cls[(np.abs(cp[..., 2] - Z0) < 1e-6).all(1)] = "cap"
print("classes:", collections.Counter(cls), flush=True)

# node class sets
pcls = collections.defaultdict(set)
for f, q in enumerate(FC):
    for v in q:
        pcls[v].add(cls[f])
lcls = collections.defaultdict(set)
for f, q in enumerate(FQ):
    for l in q:
        lcls[l].add(cls[f])

# outline interpolation for dome nodes next to the pipe
oa = np.mod(ta, 2 * np.pi); order = np.argsort(oa)
OA = oa[order]; OR = np.hypot(*LP.T)[order]; ODT = (tt - ta)[order]
OA_ = np.r_[OA[-1] - 2 * np.pi, OA, OA[0] + 2 * np.pi]
interp = lambda a, v: np.interp(np.mod(a, 2 * np.pi), OA_, np.r_[v[-1], v, v[0]])
LB = 30.0
RHO_FAR = 2.8
OLP = LP[order]


def ray_r(a):
    """radius of the (star-shaped) outline polygon along angle a."""
    am = np.mod(a, 2 * np.pi)
    j = np.searchsorted(OA, am) - 1
    p, q = OLP[j % len(OLP)], OLP[(j + 1) % len(OLP)]
    d = np.array([np.cos(a), np.sin(a)])
    M = np.array([d, p - q]).T
    tq = np.linalg.solve(M, p)
    return tq[0]


def theta_of(p):
    k = key2(p)
    return TH.get(k)


def pipe_xyz(th, z):
    """pipe wall point at theta / column height z, opened for the branch."""
    y, v = RP * np.sin(th), Zs(z) - ZB if z is not None else None
    if np.cos(th) > 0 and v is not None:
        sy = max(abs(YLO), abs(YHI))
        rho = max(abs(y) / sy, abs(v) / HT.SW)
        if rho < RHO_FAR:
            wv = np.array([y, v]); phi = 1.0 - np.clip((rho - 1.0) / (RHO_FAR - 1.0), 0, 1)
            phi = phi * phi * (3 - 2 * phi)
            wn = wv + phi * (RB * wv / np.linalg.norm(wv) - wv / rho)
            return np.array([np.sqrt(max(RP ** 2 - wn[0] ** 2, 0.0)), wn[0], ZB + wn[1]])
    return np.array([RP * np.cos(th), RP * np.sin(th), Zs(z)])


def target(p, cs):
    x, y, z = p
    if "cap" in cs:
        r = np.hypot(x, y); return np.array([R2070 * x / r, R2070 * y / r, Z0])
    if "outlet" in cs:
        w = np.array([y, z - ZB]); w = RB * w / np.linalg.norm(w); return np.array([XE, w[0], ZB + w[1]])
    if "thim" in cs:
        return np.array([x, y, ZT])
    if "duct" in cs and "pipe" in cs:
        th = theta_of(p)
        w = np.array([RP * np.sin(th), Zs(z) - ZB]); w = RB * w / np.linalg.norm(w)
        return np.array([np.sqrt(RP ** 2 - w[0] ** 2), w[0], ZB + w[1]])
    if "duct" in cs:
        w = np.array([y, z - ZB]); w = RB * w / np.linalg.norm(w); return np.array([x, w[0], ZB + w[1]])
    if "pipe" in cs:
        th = theta_of(p)
        if "top" in cs:
            return np.array([RP * np.cos(th), RP * np.sin(th), ZT])
        if "dome" in cs:
            return np.array([RP * np.cos(th), RP * np.sin(th), ZF])
        return pipe_xyz(th, z)
    if cs == {"top"}:
        r = np.hypot(x, y); a = np.arctan2(y, x); ro = ray_r(a)
        phi = np.clip(1 - (ro - r) / 35.0, 0, 1) if r <= ro + 1e-6 else 0.0
        r = r + (RP - ro) * phi; a = a + interp(a, ODT) * phi
        return np.array([r * np.cos(a), r * np.sin(a), ZT])
    if "rim" in cs and not (cs - {"rim"}):
        # rim face: knuckle-arc angle linear in height between the cap ring and the rim top's
        # closest point (closest-point projection bunches the rim nodes toward the top)
        sr = np.clip((z - Z0) / HT.H_RIM, 0.0, 1.0)
        f0 = np.arctan2(Z0 - CK[1], R2070 - CK[0])
        qt = HT.closest(np.array([[RC, Z0 + HT.H_RIM]]))[0]
        f1 = np.arctan2(qt[1] - CK[1], qt[0] - CK[0])
        ph = f0 + sr * (f1 - f0)
        a = np.arctan2(y, x)
        rho, zz = CK[0] + RK * np.cos(ph), CK[1] + RK * np.sin(ph)
        return np.array([rho * np.cos(a), rho * np.sin(a), zz])
    # dome / rim: radial push next to the pipe, then closest point on the profile
    r = np.hypot(x, y); a = np.arctan2(y, x)
    ro = ray_r(a)
    if r >= ro - 1e-6 and z < ZF:
        phi = np.clip(1 - (r - ro) / LB, 0, 1)
        r = r + (RP - ro) * phi; a = a + interp(a, ODT) * phi
    q = HT.closest(np.array([[r, z]]))[0]
    return np.array([q[0] * np.cos(a), q[0] * np.sin(a), q[1]])


newP = PTS.copy()
for v in range(len(PTS)):
    cs = pcls.get(v, set()) - {"thim_"}
    if cs and cs - {"cap", "thim", "outlet", "top"}:
        newP[v] = target(PTS[v], cs)
newLI = LI.copy()
for l in range(len(LINES)):
    cs = lcls.get(l, set())
    if cs and cs - {"cap", "thim", "outlet", "top"}:
        for g in range(LI.shape[1]):
            newLI[l, g] = target(LI[l, g], cs)
newFI = FI.copy()
for f in range(len(FC)):
    if cls[f] not in ("cap", "thim", "outlet", "top"):
        for g in range(FI.shape[1]):
            newFI[f, g] = target(FI[f, g], {cls[f]})
print("targets (%.0fs)" % (time.time() - t0), flush=True)
Bo = QuadMesh(LineMesh(PointMesh(newP, lm.point_mesh.element_tags), LINES, newLI, lm.element_tags),
              FQ, np.asarray(Bq.orient), newFI, Bq.element_tags)
mask = ~np.isin(cls, ("cap", "thim", "outlet", "top"))
Pn = newP[FC]; Po = PTS[FC]
nrm = np.cross(Po[:, 1] - Po[:, 0], Po[:, 3] - Po[:, 0]) + np.cross(Po[:, 2] - Po[:, 1], Po[:, 0] - Po[:, 1])
mv = Pn.mean(1) - Po.mean(1)
dot = (nrm * mv).sum(1)
for c in ("dome", "rim", "pipe", "top", "duct"):
    m = (cls == c)
    print("  %s: %d faces, inward %d, outward %d, zero-move %d" % (c, m.sum(), (m & (dot < 0)).sum(), (m & (dot > 0)).sum(), (m & (np.linalg.norm(mv, axis=1) < 1e-9)).sum()))
bad = mask & (dot <= 0)
if bad.any():
    print("  e.g.", [(cls[f], np.round(Po[f].mean(0), 1), np.round(mv[f], 2)) for f in np.where(bad)[0][:6]])
from nekmeshpy import HexMesh
fm = np.where(mask)[0]
hx = np.c_[FC[fm], FC[fm] + len(PTS)]
lin = HexMesh.from_corners(np.r_[PTS, newP], hx)
csj = np.asarray(hexmesh.corner_scaled_jacobian(lin)); csj = csj.min(1) if csj.ndim > 1 else csj
neg = csj <= 0
if neg.any():
    print("  linear skin: %d inverted (of %d); by class %s" % (neg.sum(), len(fm), collections.Counter(cls[fm[neg]])), flush=True)
    for c in ("pipe", "dome", "top", "duct"):
        mm = neg & (cls[fm] == c)
        if mm.any():
            cc_ = PTS[FC[fm[mm]]].mean(1)
            print("   %s at r %s z %s" % (c, np.round(np.hypot(*cc_[:6, :2].T), 1).tolist(), np.round(cc_[:6, 2], 1).tolist()), flush=True)
    for f in [fm[neg & (cls[fm] == c_)][0] for c_ in ("dome", "top", "duct") if (neg & (cls[fm] == c_)).any()]:
        print("   face", np.round(PTS[FC[f]], 2).tolist(), "->", np.round(newP[FC[f]], 2).tolist())
        print("     theta", [round(float(np.degrees(TH.get(key2(PTS[v]), np.nan))), 3) for v in FC[f]], "atan2", [round(float(np.degrees(np.arctan2(PTS[v][1], PTS[v][0]))), 3) for v in FC[f]])
    for v in FC[fm[neg][0]]:
        p_ = PTS[v]; a_ = np.arctan2(p_[1], p_[0])
        print("    node", np.round(p_, 2), pcls[v], "r %.2f ro %.2f" % (np.hypot(p_[0], p_[1]), ray_r(a_)))
    if os.environ.get("STOP"): sys.exit()
sk_in, sk_out = quadmesh.select(Bq, mask), quadmesh.select(Bo, mask)
skin = hexmesh.loft([sk_in, sk_out], element_tags="fluid", first_tag="", last_tag="wall")
print("skin: %d hexes, corner %s (%.0fs)" % (len(skin.hexes), hexmesh.corner_summary(skin), time.time() - t0), flush=True)
mesh = hexmesh.merge([inner, skin], clear_seam_tags=True)
print("head_top: %d hexes" % len(mesh.hexes), hexmesh.topology_report(mesh), flush=True)
print(hexmesh.quality_summary(mesh)); print("corner:", hexmesh.corner_summary(mesh))
sys.path.insert(0, os.path.join(HERE, "..", "p4"))
import hexfix
mesh = hexfix.straighten_bad(mesh)
pickle.dump(mesh, open(os.path.join(HERE, "head_top.pkl"), "wb"))
from nekmeshpy.io.writer import to_vtu
to_vtu(mesh, os.path.join(HERE, "head_top.vtu"))
print("done (%.0fs)" % (time.time() - t0))
