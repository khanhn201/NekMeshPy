"""Inlet volute: a thin transition band on the R760.8 wall (hole O-grids at the bore ends -> plain
(theta, z) grid, two pillow-cap stages) and the half-torus D section (r101.6) revolved over the
downcomer's columns.  -> volute.pkl / .vtu"""
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
RV = DC.R_VOL
ZC, RD = DC.Z_DC, DC.R_D
TB = 24.0                                     # band thickness
RH0, RH1, RH2 = RV, RV + TB / 2, RV + TB      # band levels: O-grid -> one ring -> plain
A = DC.A_HOLE
NR = 3                                       # C-ring layers of the D section
NA = 4                                       # core columns


import tilt as TL
DELTA = TL.delta(DC.R_HI, RV)                 # rotation of the tilted slide R_HI -> RV


def zscale(rho):
    return np.sqrt(np.maximum(RD ** 2 - (rho - RV) ** 2, 0)) / RD


H = None                                      # Holes over the stage-2 band (set in main)


def theta_v(P, rho):
    """downcomer unrolled (s, z) -> volute angle: the exact slide along the bore axis inside each
    opening, the uniform rotation DELTA away from the holes, blended across ringB and faded out
    radially between RH0 (bore ends) and RH1."""
    P = np.atleast_2d(P)
    rho = np.broadcast_to(np.asarray(rho, float), (len(P),))
    th = P[:, 0] / DC.RREF + DELTA
    wr = np.clip((RH1 - rho) / (RH1 - RH0), 0.0, 1.0)
    if H is None:
        return th
    j = H.which(P)
    for i, p in enumerate(P):
        m = H.meta[j[i]]
        if wr[i] <= 0 or not in_rect(m, p):
            continue
        uv = DC.din_inv(m["k"], p[None, :2])[0]
        if np.hypot(*uv) <= A + 1e-9:
            w = 1.0
        else:
            kk, u, v = H.ringq(m, p[:2], arc_c, rect_c)
            w = 1.0 - v
        w *= wr[i]
        if w <= 0:
            continue
        X = TL.wall_xyz(m["k"], uv[0], uv[1], RV)
        a = np.arctan2(X[1], X[0])
        a = th[i] + ((a - th[i] + np.pi) % (2 * np.pi) - np.pi)
        th[i] = (1 - w) * th[i] + w * a
    return th


def place(P2, rho):
    """2D downcomer coords + radius -> xyz on the volute side (z squeezed onto the D arc)."""
    th_ = theta_v(P2, rho)
    z = ZC + (P2[:, 1] - ZC) * zscale(rho)
    return np.c_[rho * np.cos(th_), rho * np.sin(th_), z]


def place_rho(P2, rho):
    th_ = theta_v(P2, rho)
    z = ZC + (P2[:, 1] - ZC) * zscale(rho)
    return np.c_[rho * np.cos(th_), rho * np.sin(th_), z]


def to3d(rho_fn):
    return lambda P: place_rho(P[:, :2], rho_fn(P))


# ------------------------------------------------------------------ band section ---------
def band_rows():
    """band rows: the downcomer's hole-block rows inside [Z_DBOT, Z_DTOP], filled to ~15 mm."""
    zs_dc = DC.rows(DC.holes[:, 1])[0]
    rows = [DC.Z_DBOT]
    blk_rows = sorted(set(np.round(zs_dc[(zs_dc > DC.Z_DBOT + 1) & (zs_dc <= DC.Z_DTOP + 1e-9)], 9)))
    for z in blk_rows:
        if z - rows[-1] > 20:
            m = int(round((z - rows[-1]) / 15.0))
            rows += list(np.linspace(rows[-1], z, m + 1)[1:-1])
        rows.append(z)
    if rows[-1] < DC.Z_DTOP - 1e-6:
        rows.append(DC.Z_DTOP)
    return np.array(rows), zs_dc


def build_band_sec(th, zs, zs_dc, stage):
    S, info = DC.hole_section(th, zs, stage, dc_rows=zs_dc)
    meta = []
    for d in info:
        cen = d["cen"]
        ang = np.arctan2(d["bd"][:, 1] - cen[1], d["bd"][:, 0] - cen[0])
        meta.append(dict(cen=cen, lam=d["lam"], rect=d["rect"], bd=d["bd"], ang=ang, phi=d["phi"], k=d["k"], sref=d["sref"]))
    return S, meta


# ------------------------------------------------------------------ ring parametrisation --
def rect_point(rect, cen, phi):
    (x0, y0), (x1, y1) = rect
    d = np.array([np.cos(phi), np.sin(phi)])
    ts = []
    for k, lim in ((0, x0), (0, x1), (1, y0), (1, y1)):
        if abs(d[k]) > 1e-15:
            t = (lim - cen[k]) / d[k]
            if t > 0:
                q = cen + t * d
                if x0 - 1e-9 <= q[0] <= x1 + 1e-9 and y0 - 1e-9 <= q[1] <= y1 + 1e-9:
                    ts.append(t)
    return min(ts)


class Holes:
    def __init__(self, meta):
        from scipy.spatial import cKDTree
        self.meta = meta
        self.tree = cKDTree(np.array([m["cen"] for m in meta]))

    def which(self, P):
        return self.tree.query(P[:, :2])[1]

    def ringq(self, m, p, inner, outer):
        """param (k, u, v) of p in the ring between curves inner(k,u) and outer(k,u)."""
        uvp = DC.din_inv(m["k"], np.asarray(p, float)[None, :2])[0]
        phi = np.arctan2(uvp[1], uvp[0])
        ph = m["phi"]; n = len(ph)
        rel = np.mod(phi - ph, 2 * np.pi)
        span = np.mod(np.roll(ph, -1) - ph, 2 * np.pi)
        k0 = int(np.argmin(np.where(rel <= span + 1e-12, rel, 9)))
        best = None
        for k in (k0, (k0 - 1) % n, (k0 + 1) % n):
            uv = np.array([0.5, 0.5])
            for _ in range(50):
                u, v = uv
                a_, b_ = inner(m, k, u), outer(m, k, u)
                x = (1 - v) * a_ + v * b_
                h = 1e-7
                du = ((1 - v) * inner(m, k, u + h) + v * outer(m, k, u + h) - x) / h
                J = np.c_[du, b_ - a_]
                step = np.linalg.solve(J, np.asarray(p, float)[:2] - x)
                uv = uv + step
                if np.abs(step).max() < 1e-13:
                    break
            bad = max(0.0, -uv[0], uv[0] - 1)
            if best is None or bad < best[0]:
                best = (bad, k, uv[0], uv[1])
            if bad < 1e-9:
                break
        return best[1], best[2], best[3]


def arc_c(m, k, u):
    f0 = m["phi"][k]; f1 = m["phi"][(k + 1) % len(m["phi"])]
    f = f0 + u * ((f1 - f0 + np.pi) % (2 * np.pi) - np.pi)
    return DC.din(m["k"], A * np.cos(f), A * np.sin(f), m["sref"])


def rect_c(m, k, u):
    P = m["bd"]
    return (1 - u) * P[k] + u * P[(k + 1) % len(P)]


def sq_c(m, k, u):
    P = m["cen"] + m["lam"] * (m["bd"] - m["cen"])
    return (1 - u) * P[k] + u * P[(k + 1) % len(P)]


def in_square(m, p):
    (x0, y0), (x1, y1) = m["rect"]
    c, l = m["cen"], m["lam"]
    return (c[0] + l * (x0 - c[0]) - 1e-9 <= p[0] <= c[0] + l * (x1 - c[0]) + 1e-9 and
            c[1] + l * (y0 - c[1]) - 1e-9 <= p[1] <= c[1] + l * (y1 - c[1]) + 1e-9)


def in_rect(m, p):
    (x0, y0), (x1, y1) = m["rect"]
    return x0 - 1e-9 <= p[0] <= x1 + 1e-9 and y0 - 1e-9 <= p[1] <= y1 + 1e-9


if __name__ == "__main__":
    t0 = time.time()
    D3 = pickle.load(open(os.path.join(HERE, "dc3.pkl"), "rb"))
    th = D3["th"]
    zs, zs_dc = band_rows()
    S2, meta = build_band_sec(th, zs, zs_dc, 2)
    S1, _ = build_band_sec(th, zs, zs_dc, 1)
    S0, _ = build_band_sec(th, zs, zs_dc, 0)
    H = Holes(meta)
    globals()['H'] = H
    q2, q1, q0 = S2.to_quadmesh(ORDER, 0.0), S1.to_quadmesh(ORDER, 0.0), S0.to_quadmesh(ORDER, 0.0)
    print("band sections: %d / %d / %d quads (%.0fs)" % (len(S2.quads), len(S1.quads), len(S0.quads), time.time() - t0), flush=True)
    rmB, rmA = RH0 + TB / 4, RH1 + TB / 4

    # stage B, layer 1: P2 (O-grid) at RH0 -> mid surface (RH1 outside, dip to rmB inside the circle)
    def midB(P):
        out = np.full(len(P), RH1)
        j = H.which(P)
        for i, p in enumerate(P):
            m = H.meta[j[i]]
            if not in_rect(m, p):
                continue
            if np.hypot(*DC.din_inv(m["k"], p[None, :2])[0]) <= A + 1e-9:
                out[i] = rmB
            else:
                k, u, v = H.ringq(m, p[:2], arc_c, rect_c)
                out[i] = rmB + v * (RH1 - rmB)
        return out
    c1B = hexmesh.loft([quadmesh.transform_fn(q2, to3d(lambda P: np.full(len(P), RH0))),
                        quadmesh.transform_fn(q2, to3d(midB))], element_tags="fluid", first_tag="", last_tag="wall")
    # stage B, layer 2: disk (square + ringA) at rmB -> expanded (square + ringA') at RH1
    lab2 = np.array(S2.lab)
    disk = quadmesh.select(q2, np.isin(lab2, ("square", "ringA")))

    def expB(t):
        def f(P):
            out = np.empty((len(P), 2)); rho = np.empty(len(P))
            j = H.which(P)
            for i, p in enumerate(P):
                m = H.meta[j[i]]
                if in_square(m, p[:2]):
                    out[i] = p[:2]; rho[i] = (1 - t) * rmB + t * RH1
                    continue
                k, u, v = H.ringq(m, p[:2], sq_c, arc_c)
                q = (1 - v) * sq_c(m, k, u) + v * rect_c(m, k, u)
                out[i] = (1 - t) * p[:2] + t * q
                rho[i] = rmB + t * (RH1 - rmB) if v > 1 - 1e-7 else (1 - t) * rmB + t * RH1
            return place_rho(out, rho)
        return f
    c2B = hexmesh.loft([quadmesh.transform_fn(disk, expB(0.0)), quadmesh.transform_fn(disk, expB(1.0))],
                       sweep_nodes=[[quadmesh.transform_fn(disk, expB(t)) for t in G[1:-1]]],
                       element_tags="fluid", first_tag="", last_tag="wall")
    print("stage B: c1 %d, c2 %d hexes (%.0fs)" % (len(c1B.hexes), len(c2B.hexes), time.time() - t0), flush=True)

    # stage A, layer 1: P1 at RH1 -> mid surface (RH2 outside, dip to rmA inside the square)
    def midA(P):
        out = np.full(len(P), RH2)
        j = H.which(P)
        for i, p in enumerate(P):
            m = H.meta[j[i]]
            if not in_rect(m, p):
                continue
            if in_square(m, p[:2]):
                out[i] = rmA
            else:
                k, u, v = H.ringq(m, p[:2], sq_c, rect_c)
                out[i] = rmA + v * (RH2 - rmA)
        return out
    c1A = hexmesh.loft([quadmesh.transform_fn(q1, to3d(lambda P: np.full(len(P), RH1))),
                        quadmesh.transform_fn(q1, to3d(midA))], element_tags="fluid", first_tag="", last_tag="wall")
    lab1 = np.array(S1.lab)
    sqm = quadmesh.select(q1, lab1 == "square")

    def expA(t):
        def f(P):
            out = np.empty((len(P), 2)); rho = np.empty(len(P))
            j = H.which(P)
            for i, p in enumerate(P):
                m = H.meta[j[i]]
                q = m["cen"] + (p[:2] - m["cen"]) / m["lam"]
                out[i] = (1 - t) * p[:2] + t * q
                (x0, y0), (x1, y1) = m["rect"]
                c, l = m["cen"], m["lam"]
                onb = min(abs(p[0] - (c[0] + l * (x0 - c[0]))), abs(p[0] - (c[0] + l * (x1 - c[0]))),
                          abs(p[1] - (c[1] + l * (y0 - c[1]))), abs(p[1] - (c[1] + l * (y1 - c[1])))) < 1e-9
                rho[i] = rmA + t * (RH2 - rmA) if onb else (1 - t) * rmA + t * RH2
            return place_rho(out, rho)
        return f
    c2A = hexmesh.loft([quadmesh.transform_fn(sqm, expA(0.0)), quadmesh.transform_fn(sqm, expA(1.0))],
                       sweep_nodes=[[quadmesh.transform_fn(sqm, expA(t)) for t in G[1:-1]]],
                       element_tags="fluid", first_tag="", last_tag="")
    print("stage A: c1 %d, c2 %d hexes (%.0fs)" % (len(c1A.hexes), len(c2A.hexes), time.time() - t0), flush=True)

    # D section (r, z) for r >= RH2, flat side rows = band rows squeezed at RH2
    f2 = zscale(RH2)
    zf = ZC + (zs - ZC) * f2
    nz = len(zf)
    lo, hi = NR, nz - 1 - NR
    a_core = 0.35 * RD
    rc = RH2 + a_core * np.arange(NA + 1) / NA
    D = Sec()
    core = {}
    for i in range(NA + 1):
        for j in range(lo, hi + 1):
            core[(i, j)] = D.P((rc[i], zf[j]))
    for i in range(NA):
        for j in range(lo, hi):
            D.quad((core[(i, j)], core[(i + 1, j)], core[(i + 1, j + 1)], core[(i, j + 1)]), "core")
    bnd = [core[(i, hi)] for i in range(NA + 1)] + [core[(NA, j)] for j in range(hi - 1, lo - 1, -1)] + \
          [core[(i, lo)] for i in range(NA - 1, -1, -1)]
    Bp = np.array([D.X[b] for b in bnd])
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(Bp, axis=0), axis=1))]); L /= L[-1]
    cD = np.array([RV, ZC])
    aT = np.arctan2(zf[-1] - ZC, RH2 - RV); aB = np.arctan2(zf[0] - ZC, RH2 - RV)
    arcp = [cD + RD * np.array([np.cos(aT + f * (aB - aT)), np.sin(aT + f * (aB - aT))]) for f in L]
    layers = [bnd]
    M = len(bnd)
    HS = float(os.environ.get("VOL_HS", "0"))          # >0: thin outermost layer (skin variant)
    NSH = int(os.environ.get("VOL_NSH", "2"))           # outer layers that form the inlet shell
    for k in range(1, NR + 1):
        tt = k / NR
        tT = (zf[hi + k] - zf[hi]) / (arcp[0][1] - zf[hi])
        tB = (zf[lo - k] - zf[lo]) / (arcp[-1][1] - zf[lo])
        row = []
        for jj in range(M):
            w = L[jj]
            tk = tt if 0 < jj < M - 1 else (tT if jj == 0 else tB)
            if 0 < jj < M - 1:
                tk = tt
                if k == NR - 1 and HS > 0:
                    tk = max(tt, 1.0 - HS / np.linalg.norm(arcp[jj] - Bp[jj]))
            p = (1 - tk) * Bp[jj] + tk * arcp[jj]
            if jj == 0:
                p = np.array([RH2, zf[hi + k]])
            if jj == M - 1:
                p = np.array([RH2, zf[lo - k]])
            if k == NR and 0 < jj < M - 1:
                p = arcp[jj]
            row.append(D.P(p))
        layers.append(row)
    for k in range(NR):
        for jj in range(M - 1):
            D.quad((layers[k][jj], layers[k][jj + 1], layers[k + 1][jj + 1], layers[k + 1][jj]), "ringo" if k >= NR - NSH else "ring")
    for jj in range(M - 1):
        D.arc(layers[NR][jj], layers[NR][jj + 1], cD, RD)
    dq = D.to_quadmesh(ORDER, 0.0)
    print("D section: %d quads (%.0fs)" % (len(D.quads), time.time() - t0), flush=True)
    # revolve over the volute columns (loop)
    thv_ = th + DELTA

    def rev(theta):
        return lambda P: np.c_[P[:, 0] * np.cos(theta), P[:, 0] * np.sin(theta), P[:, 1]]
    sl = [quadmesh.transform_fn(dq, rev(t)) for t in thv_]
    sw = []
    for i in range(len(thv_)):
        a0, a1 = thv_[i], thv_[(i + 1) % len(thv_)] + (2 * np.pi if i == len(thv_) - 1 else 0)
        sw.append([quadmesh.transform_fn(dq, rev(a0 + t * (a1 - a0))) for t in G[1:-1]])
    vol = hexmesh.loft(sl, loop=True, sweep_nodes=sw, element_tags="fluid", first_tag="", last_tag="")
    # inner volume for the inlet build: the same revolve without the outermost (skin) layer
    dlab = np.array(D.lab)
    dq_in = quadmesh.select(dq, dlab != "ringo")
    sl_in = [quadmesh.transform_fn(dq_in, rev(t)) for t in thv_]
    sw_in = []
    for i in range(len(thv_)):
        a0, a1 = thv_[i], thv_[(i + 1) % len(thv_)] + (2 * np.pi if i == len(thv_) - 1 else 0)
        sw_in.append([quadmesh.transform_fn(dq_in, rev(a0 + t * (a1 - a0))) for t in G[1:-1]])
    vol_in = hexmesh.loft(sl_in, loop=True, sweep_nodes=sw_in, element_tags="fluid", first_tag="", last_tag="")
    print("volute D: %d hexes, corner %s (%.0fs)" % (len(vol.hexes), hexmesh.corner_summary(vol), time.time() - t0), flush=True)
    parts = [c1B, c2B, c1A, c2A, vol]
    for nm, p in zip(("c1B", "c2B", "c1A", "c2A", "vol"), parts):
        print("  %s corner %s" % (nm, hexmesh.corner_summary(p)), flush=True)
    from scipy.spatial import cKDTree as _KD
    def edge_diff(A_, B_):
        la, lb = A_.quad_mesh.line_mesh, B_.quad_mesh.line_mesh
        Pa, Pb = np.asarray(la.points), np.asarray(lb.points)
        d, ia = _KD(Pa).query(Pb)
        ea = {frozenset(e): i for i, e in enumerate(np.asarray(la.lines).tolist())}
        Ia, Ib = np.asarray(la.interior), np.asarray(lb.interior)
        worst = []
        for li, (u, v) in enumerate(np.asarray(lb.lines)):
            if d[u] < 1e-6 and d[v] < 1e-6:
                k = ea.get(frozenset((ia[u], ia[v])))
                if k is None:
                    continue
                A2 = Ia[k] if np.asarray(la.lines)[k][0] == ia[u] else Ia[k][::-1]
                worst.append((np.abs(A2 - Ib[li]).max(), Pb[u], Pb[v], A2, Ib[li]))
        worst.sort(key=lambda w: -w[0])
        return worst[:3]
    for w in edge_diff(c1B, c1A) + edge_diff(c1B, c2B):
        print("  diff %.3g at %s-%s\n    a %s\n    b %s" % (w[0], np.round(w[1], 2), np.round(w[2], 2), np.round(w[3], 3).tolist(), np.round(w[4], 3).tolist()), flush=True)
    names = ("c1B", "c2B", "c1A", "c2A", "vol")
    for i in range(len(parts) - 1):
        for j in range(i + 1, len(parts)):
            try:
                hexmesh.merge([parts[i], parts[j]]); print("  ok", names[i], names[j], flush=True)
            except ValueError as ex:
                print("  FAIL", names[i], names[j], str(ex)[:60], flush=True)
    mesh = hexmesh.merge(parts, clear_seam_tags=True)
    print("volute: %d hexes" % len(mesh.hexes), hexmesh.topology_report(mesh), flush=True)
    print(hexmesh.quality_summary(mesh)); print("corner:", hexmesh.corner_summary(mesh))
    pickle.dump(mesh, open(os.path.join(HERE, "volute.pkl"), "wb"))
    inner = hexmesh.merge([c1B, c2B, c1A, c2A, vol_in], clear_seam_tags=True)
    print("volute inner (no skin layer): %d hexes" % len(inner.hexes), hexmesh.topology_report(inner), flush=True)
    pickle.dump(dict(inner=inner, zf=zf, RH2=RH2, HS=HS, cD=cD, top=(RH2, zf[-1]), bot=(RH2, zf[0]),
                     inner_corner=(zf[hi + NR - 1], zf[lo - NR + 1]), thv=thv_, NSH=NSH, NR=NR, hi=hi, lo=lo,
                     L_in=np.array([D.X[i] for i in layers[NR - NSH]]), L_mid=[np.array([D.X[i] for i in layers[k]]) for k in range(NR - NSH, NR + 1)],
                     arcp=np.array([D.X[i] for i in layers[NR]])), open(os.path.join(HERE, "volute_inner.pkl"), "wb"))
    from nekmeshpy.io.writer import to_vtu
    to_vtu(mesh, os.path.join(HERE, "volute.vtu"))
