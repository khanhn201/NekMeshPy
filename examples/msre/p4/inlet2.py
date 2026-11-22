"""Inlet pipe on the volute with an O-grid window: the D section's outer NSH layers are rebuilt as a
shell over the (theta, tube-angle) grid with ONE hole block whose ring is the pipe/torus trim curve
(its two grazing tips clipped), transitioned to the plain revolve in two pillow-cap stages (as the
volute band does for the bores); the window disk is slid along the pipe axis to a cap plane and
extruded straight to the end plane (inlet).  Pipe wall = exact cylinder away from the clipped tips.
-> volute.pkl / volute.vtu"""
import os, sys, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "p1")]
from q2d import Sec
from nekmeshpy import quadmesh, hexmesh
from nekmeshpy.core.fields import gll_nodes
import dc3 as DC

ORDER = int(os.environ.get("ORDER", "2"))
G = gll_nodes(ORDER)
RV, ZC, RD = DC.R_VOL, DC.Z_DC, DC.R_D
PA = np.load(os.path.join(HERE, "pipe_axis.npy"))
C0, AX, RP = PA[:3], PA[3:6], PA[6]
E1 = np.cross([0, 0, 1.0], AX); E1 /= np.linalg.norm(E1); E2 = np.cross(AX, E1)
T_END = -635.0
T_CAP = float(os.environ.get("T_CAP", "-490"))
N1 = int(os.environ.get("N1", "8"))
DX = float(os.environ.get("DX", "25"))
CLIP = float(os.environ.get("CLIP", "0.80"))    # keep this fraction of the window's half length
MARG = float(os.environ.get("MARG", "12"))      # block rectangle margin around the ring (mm)
RA = RV + RD
t0 = time.time()

VI = pickle.load(open(os.path.join(HERE, "volute_inner.pkl"), "rb"))
inner, thv = VI["inner"], VI["thv"]
LAY = VI["L_mid"]                               # D-plane layers, inner (lambda=0) .. arc (lambda=1)
NSH = VI["NSH"]
M = len(LAY[0])
cD = np.array(VI["cD"])
ALPHA = np.unwrap(np.arctan2(LAY[-1][:, 1] - cD[1], LAY[-1][:, 0] - cD[0]))   # tube angle of the arc rows
SROW = -ALPHA * RD                               # row coordinate (increasing top -> bottom)
assert np.all(np.diff(SROW) > 0)


def pipe_coords(X):
    R = np.asarray(X, float) - C0
    return R @ AX, R @ E1, R @ E2


def tube(X):
    X = np.asarray(X, float)
    r = np.hypot(X[..., 0], X[..., 1])
    return np.arctan2(X[..., 1], X[..., 0]), r - RV, X[..., 2] - ZC


def tdist(X):
    _, dr, dz = tube(X)
    return np.hypot(dr, dz)


def gen(psi, t, rho=RP):
    return C0 + np.asarray(t)[..., None] * AX + np.asarray(rho)[..., None] * (np.cos(psi)[..., None] * E1 + np.sin(psi)[..., None] * E2)


def t_trim(psi):
    ts = np.linspace(T_END, 0.0, 1300)
    inside = tdist(gen(np.full(len(ts), psi), ts)) < RD
    k = int(np.argmax(inside)) if inside.any() else len(ts) - 1
    lo, hi = ts[max(k - 1, 0)], ts[k]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if tdist(gen(np.array(psi), np.array(mid))) < RD:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ shell placement -----------
TH0 = thv[0]


def to_shell(X):
    """3D torus point -> shell coords (a, s), a on the column branch."""
    th, dr, dz = tube(X)
    th = TH0 + np.mod(th - TH0, 2 * np.pi)
    return np.stack([th * RA, -np.arctan2(dz, dr) * RD], -1)


def place(P2, q):
    """shell coords (a, s) + level q in [0, NSH] (0 = torus, NSH = the D core surface) -> xyz."""
    P2 = np.atleast_2d(P2)
    q = np.broadcast_to(np.asarray(q, float), (len(P2),))
    th = P2[:, 0] / RA
    s = P2[:, 1]
    j = np.clip(np.searchsorted(SROW, s, side="right") - 1, 0, M - 2)
    f = np.clip((s - SROW[j]) / (SROW[j + 1] - SROW[j]), 0, 1)[:, None]
    lam = np.clip(NSH - q, 0, NSH)                 # layer coordinate 0 (inner) .. NSH (arc)
    k = np.clip(np.floor(lam).astype(int), 0, NSH - 1)
    g = (lam - k)[:, None]
    out = np.empty((len(P2), 2))
    for kk in range(NSH):
        m = k == kk
        if not m.any():
            continue
        La = (1 - f[m]) * LAY[kk][j[m]] + f[m] * LAY[kk][j[m] + 1]
        if kk + 1 == NSH:                          # the arc: exact torus
            al = -s[m] / RD
            Lb = np.c_[cD[0] + RD * np.cos(al), cD[1] + RD * np.sin(al)]
            end = (j[m] == 0) & (f[m][:, 0] < 1e-12) | (j[m] == M - 2) & (f[m][:, 0] > 1 - 1e-12)
            Lb[end] = ((1 - f[m]) * LAY[kk + 1][j[m]] + f[m] * LAY[kk + 1][j[m] + 1])[end]
        else:
            Lb = (1 - f[m]) * LAY[kk + 1][j[m]] + f[m] * LAY[kk + 1][j[m] + 1]
        out[m] = (1 - g[m]) * La + g[m] * Lb
    return np.c_[out[:, 0] * np.cos(th), out[:, 0] * np.sin(th), out[:, 1]]


# ------------------------------------------------------------------ the window ring ------------
pg = np.linspace(-np.pi, np.pi, 1441)[:-1]
TR = np.array([t_trim(p) for p in pg])
OV = to_shell(gen(pg, TR))
CO = OV.mean(0)
bet = lambda P: np.arctan2(P[..., 1] - CO[1], P[..., 0] - CO[0])
rad = lambda P: np.hypot(P[..., 0] - CO[0], P[..., 1] - CO[1])


def polar_interp(bs, vals):
    o = np.argsort(np.mod(bs, 2 * np.pi)); B = np.mod(bs, 2 * np.pi)[o]; V = np.asarray(vals)[o]
    B_ = np.r_[B[-1] - 2 * np.pi, B, B[0] + 2 * np.pi]; V_ = np.r_[V[-1], V, V[0]]
    return lambda x: np.interp(np.mod(x, 2 * np.pi), B_, V_)


R_OV = polar_interp(bet(OV), rad(OV))
HA = 0.5 * (OV[:, 0].max() - OV[:, 0].min())
bg = np.linspace(-np.pi, np.pi, 1441)[:-1]
rc = np.minimum(R_OV(bg), CLIP * HA / np.maximum(np.abs(np.cos(bg)), 1e-9))
for _ in range(30):                                # round the clip corners
    rc = np.minimum(rc, 0.5 * rc + 0.25 * (np.roll(rc, 1) + np.roll(rc, -1)))
R_RING = polar_interp(bg, rc)
ring_pt = lambda b: CO + R_RING(b)[..., None] * np.stack([np.cos(b), np.sin(b)], -1)
print("window: half length %.1f, half height %.1f mm (shell coords); clipped to %.0f%% of the length (%.0fs)" % (
    HA, 0.5 * (OV[:, 1].max() - OV[:, 1].min()), 100 * CLIP, time.time() - t0), flush=True)

# ------------------------------------------------------------------ shell sections ------------
n = len(thv)
A_COL = thv * RA
Lp = 2 * np.pi * RA
rA = ring_pt(bg)
i0 = int(np.searchsorted(A_COL, rA[:, 0].min() - MARG, side="right")) - 1
i1 = int(np.searchsorted(A_COL, rA[:, 0].max() + MARG, side="left"))
j0 = int(np.searchsorted(SROW, rA[:, 1].min() - MARG, side="right")) - 1
j1 = int(np.searchsorted(SROW, rA[:, 1].max() + MARG, side="left"))
assert 0 <= i0 and i1 <= n and 0 <= j0 and j1 <= M - 1, (i0, i1, j0, j1)
na, nb = i1 - i0, j1 - j0
print("block: columns %d..%d (%d), rows %d..%d (%d)" % (i0, i1, na, j0, j1, nb), flush=True)


def shell_sec(stage):
    S = Sec()
    ids = np.array([[S.P((A_COL[i] if i < n else A_COL[0] + Lp, SROW[j])) for j in range(M)] for i in range(n + 1)])
    inblk = {(i, j) for i in range(i0, i1) for j in range(j0, j1)} if stage > 0 else set()
    for i in range(n):
        for j in range(M - 1):
            if (i, j) not in inblk:
                S.quad((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]), "shell")
    meta = None
    if stage > 0:
        walk = [(t, 0) for t in range(na)] + [(na, t) for t in range(nb)] + \
               [(na - t, nb) for t in range(na)] + [(0, nb - t) for t in range(nb)]
        bd = [ids[i0 + a, j0 + b] for a, b in walk]
        P = np.array([S.X[b_] for b_ in bd])
        ang = bet(P)
        lam = np.full(2, 0.72 * np.min(R_RING(ang) / rad(P)))
        (x0_, y0_), (x1_, y1_) = S.X[ids[i0, j0]], S.X[ids[i1, j1]]
        la = float(os.environ.get("LAMA", "0.9"))
        if la > 0:
            lam = np.array([la * np.min(np.abs(rA[:, 0] - CO[0]).max() / max(CO[0] - x0_, x1_ - CO[0])) / 1.0,
                            la * np.abs(rA[:, 1] - CO[1]).max() / max(CO[1] - y0_, y1_ - CO[1])])
            cor = CO + lam * (np.array([[x0_, y0_], [x1_, y0_], [x1_, y1_], [x0_, y1_]]) - CO)
            while (rad(cor) > R_RING(bet(cor)) * 0.85).any():
                lam = lam * 0.97
                cor = CO + lam * (np.array([[x0_, y0_], [x1_, y0_], [x1_, y1_], [x0_, y1_]]) - CO)
        sq = {}
        for a in range(na + 1):
            for b in range(nb + 1):
                g_ = np.array(S.X[ids[i0 + a, j0 + b]])
                sq[(a, b)] = S.P(CO + lam * (g_ - CO))
        mm = len(bd)
        if stage == 2:
            ring = [S.P(ring_pt(np.array(b_))) for b_ in ang]
            for k in range(mm):
                l = (k + 1) % mm
                db = (ang[l] - ang[k] + np.pi) % (2 * np.pi) - np.pi
                S.curve(ring[k], ring[l], ("fn", lambda t, b0=ang[k], db=db: ring_pt(b0 + t * db)))
                S.quad((bd[k], bd[l], ring[l], ring[k]), "ringB")
                S.quad((ring[k], ring[l], sq[walk[l]], sq[walk[k]]), "ringA")
        else:
            for k in range(mm):
                l = (k + 1) % mm
                S.quad((bd[k], bd[l], sq[walk[l]], sq[walk[k]]), "ringA")
        for a in range(na):
            for b in range(nb):
                S.quad((sq[(a, b)], sq[(a + 1, b)], sq[(a + 1, b + 1)], sq[(a, b + 1)]), "square")
        meta = dict(cen=CO, lam=lam, bd=P, ang=ang, rect=(S.X[ids[i0, j0]], S.X[ids[i1, j1]]))
    return S, meta


S2, MT = shell_sec(2)
S1, _ = shell_sec(1)
S0, _ = shell_sec(0)
q2, q1, q0 = S2.to_quadmesh(ORDER, 0.0), S1.to_quadmesh(ORDER, 0.0), S0.to_quadmesh(ORDER, 0.0)
print("shell sections: %d / %d / %d quads (%.0fs)" % (len(S2.quads), len(S1.quads), len(S0.quads), time.time() - t0), flush=True)


def in_rect(p):
    (x0, y0), (x1, y1) = MT["rect"]
    return x0 - 1e-9 <= p[0] <= x1 + 1e-9 and y0 - 1e-9 <= p[1] <= y1 + 1e-9


def in_square(p):
    (x0, y0), (x1, y1) = MT["rect"]
    c, l = CO, MT["lam"]
    return (c[0] + l[0] * (x0 - c[0]) - 1e-9 <= p[0] <= c[0] + l[0] * (x1 - c[0]) + 1e-9 and
            c[1] + l[1] * (y0 - c[1]) - 1e-9 <= p[1] <= c[1] + l[1] * (y1 - c[1]) + 1e-9)


def in_ring(p):
    return rad(np.asarray(p)) <= R_RING(bet(np.asarray(p))) + 1e-9


def arc_c(k, u):
    a0, a1 = MT["ang"][k], MT["ang"][(k + 1) % len(MT["ang"])]
    return ring_pt(np.array(a0 + u * ((a1 - a0 + np.pi) % (2 * np.pi) - np.pi)))


def rect_c(k, u):
    P = MT["bd"]
    return (1 - u) * P[k] + u * P[(k + 1) % len(P)]


def sq_c(k, u):
    P = CO + MT["lam"] * (MT["bd"] - CO)
    return (1 - u) * P[k] + u * P[(k + 1) % len(P)]


def ringq(p, inner_c, outer_c):
    ang = MT["ang"]; nn = len(ang)
    rel = np.mod(bet(np.asarray(p)) - ang, 2 * np.pi)
    span = np.mod(np.roll(ang, -1) - ang, 2 * np.pi)
    k0 = int(np.argmin(np.where(rel <= span + 1e-12, rel, 9)))
    best = None
    for k in (k0, (k0 - 1) % nn, (k0 + 1) % nn):
        uv = np.array([0.5, 0.5])
        for _ in range(60):
            u, v = uv
            a_, b_ = inner_c(k, u), outer_c(k, u)
            x = (1 - v) * a_ + v * b_
            h = 1e-7
            du = ((1 - v) * inner_c(k, u + h) + v * outer_c(k, u + h) - x) / h
            step = np.linalg.solve(np.c_[du, b_ - a_], np.asarray(p, float)[:2] - x)
            uv = uv + step
            if np.abs(step).max() < 1e-13:
                break
        bad = max(0.0, -uv[0], uv[0] - 1)
        if best is None or bad < best[0]:
            best = (bad, k, uv[0], uv[1])
        if bad < 1e-9:
            break
    return best[1], best[2], best[3]


Q0, Q1, Q2 = 0.0, NSH / 2.0, float(NSH)
qmB = Q0 + float(os.environ.get("DIPB", "0.25")) * NSH
qmA = Q1 + float(os.environ.get("DIPA", "0.05")) * NSH
at = lambda qfun: (lambda P: place(P[:, :2], qfun(P)))

# stage B, layer 1: window O-grid (torus, q=Q0) -> mid surface (Q1 outside, dipped to qmB inside the ring)
def midB(P):
    out = np.full(len(P), Q1)
    for i, p in enumerate(P):
        if not in_rect(p):
            continue
        if in_ring(p[:2]):
            out[i] = qmB
        else:
            k, u, v = ringq(p[:2], arc_c, rect_c)
            out[i] = qmB + v * (Q1 - qmB)
    return out


c1B = hexmesh.loft([quadmesh.transform_fn(q2, at(lambda P: np.full(len(P), Q0))), quadmesh.transform_fn(q2, at(midB))],
                   element_tags="fluid", first_tag="wall", last_tag="")
lab2 = np.array(S2.lab)
disk = quadmesh.select(q2, np.isin(lab2, ("square", "ringA")))


def expB(t):
    def f(P):
        out = np.empty((len(P), 2)); qq = np.empty(len(P))
        for i, p in enumerate(P):
            if in_square(p[:2]):
                out[i] = p[:2]; qq[i] = (1 - t) * qmB + t * Q1
                continue
            k, u, v = ringq(p[:2], sq_c, arc_c)
            q_ = (1 - v) * sq_c(k, u) + v * rect_c(k, u)
            out[i] = (1 - t) * p[:2] + t * q_
            qq[i] = qmB + t * (Q1 - qmB) if v > 1 - 1e-7 else (1 - t) * qmB + t * Q1
        return place(out, qq)
    return f


c2B = hexmesh.loft([quadmesh.transform_fn(disk, expB(0.0)), quadmesh.transform_fn(disk, expB(1.0))],
                   sweep_nodes=[[quadmesh.transform_fn(disk, expB(t)) for t in G[1:-1]]],
                   element_tags="fluid", first_tag="", last_tag="")


def midA(P):
    out = np.full(len(P), Q2)
    for i, p in enumerate(P):
        if not in_rect(p):
            continue
        if in_square(p[:2]):
            out[i] = qmA
        else:
            k, u, v = ringq(p[:2], sq_c, rect_c)
            out[i] = qmA + v * (Q2 - qmA)
    return out


c1A = hexmesh.loft([quadmesh.transform_fn(q1, at(lambda P: np.full(len(P), Q1))), quadmesh.transform_fn(q1, at(midA))],
                   element_tags="fluid", first_tag="", last_tag="")
sqm = quadmesh.select(q1, np.array(S1.lab) == "square")


def expA(t):
    def f(P):
        out = np.empty((len(P), 2)); qq = np.empty(len(P))
        (x0, y0), (x1, y1) = MT["rect"]; c, l = CO, MT["lam"]
        for i, p in enumerate(P):
            out[i] = (1 - t) * p[:2] + t * (c + (p[:2] - c) / l)
            onb = min(abs(p[0] - (c[0] + l[0] * (x0 - c[0]))), abs(p[0] - (c[0] + l[0] * (x1 - c[0]))),
                      abs(p[1] - (c[1] + l[1] * (y0 - c[1]))), abs(p[1] - (c[1] + l[1] * (y1 - c[1])))) < 1e-9
            qq[i] = qmA + t * (Q2 - qmA) if onb else (1 - t) * qmA + t * Q2
        return place(out, qq)
    return f


c2A = hexmesh.loft([quadmesh.transform_fn(sqm, expA(0.0)), quadmesh.transform_fn(sqm, expA(1.0))],
                   sweep_nodes=[[quadmesh.transform_fn(sqm, expA(t)) for t in G[1:-1]]],
                   element_tags="fluid", first_tag="", last_tag="")
for nm, b_ in (("c1B", c1B), ("c2B", c2B), ("c1A", c1A), ("c2A", c2A)):
    print("  %s: %d hexes, corner %s" % (nm, len(b_.hexes), hexmesh.corner_summary(b_)), flush=True)

# ------------------------------------------------------------------ duct -----------------------
win = quadmesh.transform_fn(disk, at(lambda P: np.full(len(P), Q0)))      # window disk on the torus
Wp = np.asarray(win.points)
tW, uW, vW = pipe_coords(Wp)
psiW = np.arctan2(vW, uW)
rhoW = np.hypot(uW, vW)
# ring radius in the pipe frame (RP except at the clipped tips) as a function of psi
Pr = ring_pt(bg); Xr = place(Pr, np.zeros(len(Pr)))
tr_, ur_, vr_ = pipe_coords(Xr)
RHO_R = polar_interp(np.arctan2(vr_, ur_), np.hypot(ur_, vr_))
print("ring in the pipe frame: rho %.2f..%.2f (RP %.3f), t %.1f..%.1f" % (np.hypot(ur_, vr_).min(), np.hypot(ur_, vr_).max(), RP, tr_.min(), tr_.max()), flush=True)


def slide(f):
    gf = f * f * (3 - 2 * f)
    def fn(P):
        t, u, v = pipe_coords(P)
        psi = np.arctan2(v, u)
        sc = 1 + gf * (RP / RHO_R(psi) - 1)                 # clipped tips: back to the full radius
        return C0 + ((1 - f) * t + f * T_CAP)[:, None] * AX + (sc * u)[:, None] * E1 + (sc * v)[:, None] * E2
    return fn


fr = np.linspace(0, 1, N1 + 1)
sl = [quadmesh.transform_fn(win, slide(f)) for f in fr]
sw = [[quadmesh.transform_fn(win, slide(f + g * (fr[k + 1] - f))) for g in G[1:-1]] for k, f in enumerate(fr[:-1])]
cap = sl[-1]
n2 = max(2, int(round((T_CAP - T_END) / DX)))
for k in range(1, n2 + 1):
    tt = T_CAP + (T_END - T_CAP) * k / n2
    t_prev = T_CAP + (T_END - T_CAP) * (k - 1) / n2
    sl.append(quadmesh.transform_fn(cap, lambda P, tt=tt: P + (tt - T_CAP) * AX))
    sw.append([quadmesh.transform_fn(cap, lambda P, a=t_prev + g * (tt - t_prev): P + (a - T_CAP) * AX) for g in G[1:-1]])
win_t = quadmesh.tag_edges(win, quadmesh.boundary_edges(win), "wall")
sl[0] = win_t
duct = hexmesh.loft(sl, sweep_nodes=sw, element_tags="fluid", first_tag="", last_tag="inlet")
cs = hexmesh.corner_summary(duct)
print("duct: %d hexes, corner %s (%.0fs)" % (len(duct.hexes), cs, time.time() - t0), flush=True)

parts = [inner, c1B, c2B, c1A, c2A, duct]
names = ("inner", "c1B", "c2B", "c1A", "c2A", "duct")
for i in range(len(parts) - 1):
    for j in range(i + 1, len(parts)):
        try:
            hexmesh.merge([parts[i], parts[j]])
        except ValueError as ex:
            print("  FAIL", names[i], names[j], str(ex)[:70], flush=True)
mesh = hexmesh.merge(parts, clear_seam_tags=True)
print("volute + inlet: %d hexes" % len(mesh.hexes), hexmesh.topology_report(mesh), flush=True)
import hexfix
mesh = hexfix.straighten_bad(mesh)
print(hexmesh.quality_summary(mesh)); print("corner:", hexmesh.corner_summary(mesh))
v = hexmesh.validity(mesh); print("validity: inverted %d, twisted %d" % (v.inverted.size, v.twisted.size))
print(hexmesh.tag_report(mesh))
pickle.dump(mesh, open(os.path.join(HERE, "volute.pkl"), "wb"))
from nekmeshpy.io.writer import to_vtu
to_vtu(mesh, os.path.join(HERE, "volute.vtu"))
print("done (%.0fs)" % (time.time() - t0))
