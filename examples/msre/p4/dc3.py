"""Downcomer: structured (theta, z) grid on the lower head's 364 annulus columns, R711.2 -> outer wall,
z 298.987 -> 1888.81, two radial layers (matching the annulus rows), 84 hole O-grid blocks on the
outer wall, and the radial bores (r9.525) through the vessel wall to R760.82.  -> dc3.pkl / .vtu"""
import os, sys, pickle, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "p1")]
from q2d import Sec
from nekmeshpy import quadmesh, hexmesh
from nekmeshpy.core.fields import gll_nodes

ORDER = int(os.environ.get("ORDER", "2"))
RCO, R_LO, R_HI, R_VOL = 711.2, 738.4990768478899, 735.42265387044, 760.8226538704399
Z_BOT, Z_C0, Z_C1, Z_TOP = 298.98724095080007, 1594.8034715253302, 1657.712836025, 1888.8105344373203
A_HOLE = 9.525
Z_DC, R_D = 1784.1, 101.6               # volute D section: centre height, radius
Z_DTOP, Z_DBOT = Z_DC + R_D, Z_DC - R_D
RREF = R_HI
ROWH = 40.0
NB = 3                                   # hole block: NB columns x NB rows
HR = 12.7                                # row pitch through the hole bands
NT = 3                                   # layers along each bore
import tilt as TL
holes = np.c_[TL.wall_angle(np.arange(TL.NH), R_HI), TL.C[:, 2]]     # (84, 2): theta on the downcomer wall, z
G = gll_nodes(ORDER)


def r_out(z):
    z = np.asarray(z, float)
    return np.where(z <= Z_C0, R_LO, np.where(z >= Z_C1, R_HI, R_LO + (R_HI - R_LO) * (z - Z_C0) / (Z_C1 - Z_C0)))


def columns():
    H = pickle.load(open(os.path.join(HERE, "head_sec.pkl"), "rb"))
    keys, X = H["keys"], H["X_kn"]
    m = max(k[2] for k in keys if isinstance(k, tuple) and k[0] == "ann") + 1
    th = np.array([np.arctan2(*X[keys[("ann", 1, t)]][::-1]) for t in range(m)])
    return np.unwrap(th)


def rows(holes_z):
    zs = [Z_BOT]
    bands = []
    for zh in sorted(set(np.round(holes_z, 6))):
        lo = zh - (NB / 2) * HR
        b = [lo + k * HR for k in range(NB + 1)]
        if b[-1] > Z_DTOP:
            b[-1] = Z_DTOP                 # the volute's D corner: block ends on it
        bands.append(b)
    marks = [Z_C0, Z_C1]
    for b in bands:
        stops = sorted([m_ for m_ in marks if zs[-1] < m_ < b[0] - 5] + [b[0]])
        for zz in stops:
            n = max(1, int(round((zz - zs[-1]) / ROWH)))
            zs += list(np.linspace(zs[-1], zz, n + 1)[1:])
        zs += b[1:]
    if zs[-1] < Z_TOP - 1e-9:
        n = max(1, int(round((Z_TOP - zs[-1]) / ROWH)))
        zs += list(np.linspace(zs[-1], Z_TOP, n + 1)[1:])
    return np.array(zs), bands


def din(k, u, v, sref):
    """bore cross-section (u, v) -> downcomer unrolled (s, z) (s branch near sref)."""
    X = TL.wall_xyz(k, u, v, R_HI)
    a = np.arctan2(X[..., 1], X[..., 0])
    a = sref / RREF + ((a - sref / RREF + np.pi) % (2 * np.pi) - np.pi)
    return np.stack([a * RREF, X[..., 2]], -1)


def din_inv(k, P):
    """downcomer unrolled (s, z) -> bore cross-section (u, v)."""
    P = np.atleast_2d(P)
    th = P[:, 0] / RREF
    X = np.c_[R_HI * np.cos(th), R_HI * np.sin(th), P[:, 1]]
    return TL.uv_of(k, X)


def ring_fn(k, f0, f1, sref):
    df = (f1 - f0 + np.pi) % (2 * np.pi) - np.pi
    return lambda t: din(k, A_HOLE * np.cos(f0 + t * df), A_HOLE * np.sin(f0 + t * df), sref)


def hole_section(th, zs, stage=2, dc_rows=None):
    """unrolled (s = theta*RREF, z) grid on columns th / rows zs with the 84 tilted-bore hole blocks:
    stage 2 = square + ringA + ringB (exact wall opening of the bore), 1 = square + ringA' (square
    straight to the block rectangle), 0 = plain grid.  Hole block rows come from dc_rows (the
    downcomer's own rows, default zs).  Returns Sec, per-hole info."""
    n = len(th)
    dc_rows = zs if dc_rows is None else dc_rows
    s_col = th * RREF
    Lp = 2 * np.pi * RREF
    S = Sec()
    ids = np.array([[S.P((s_col[i] if i < n else s_col[0] + Lp, zs[j])) for j in range(len(zs))] for i in range(n + 1)])
    blocks, inblk = [], set()
    for k in range(TL.NH):
        thh, zh = holes[k]
        sh = (th[0] + ((thh - th[0]) % (2 * np.pi))) * RREF
        ring = din(k, A_HOLE * np.cos(np.linspace(0, 2 * np.pi, 73)), A_HOLE * np.sin(np.linspace(0, 2 * np.pi, 73)), sh)
        hw = np.abs(ring[:, 0] - sh).max()
        i0 = int(np.searchsorted(s_col, sh - hw - 3.0, side="right")) - 1
        i1 = int(np.searchsorted(s_col, sh + hw + 3.0, side="left"))
        na = max(2, i1 - i0)
        zb = dc_rows[int(np.argmin(np.abs(dc_rows - (zh - (NB / 2) * HR))))]
        j0 = int(np.argmin(np.abs(zs - zb)))
        assert abs(zs[j0] - zb) < 1e-6 and 0 <= i0 and i0 + na <= n
        blocks.append((k, i0, j0, sh, zh, thh, na))
        if stage > 0:
            for di in range(na):
                for dj in range(NB):
                    inblk.add((i0 + di, j0 + dj))
    for i in range(n):
        for j in range(len(zs) - 1):
            if (i, j) not in inblk:
                S.quad((ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]), "dc")
    info = []
    for (k, i0, j0, sh, zh, thh, na) in blocks:
        walk = [(t, 0) for t in range(na)] + [(na, t) for t in range(NB)] + \
               [(na - t, NB) for t in range(na)] + [(0, NB - t) for t in range(NB)]
        bd = [ids[i0 + a, j0 + b] for a, b in walk]
        cen = din(k, 0.0, 0.0, sh)
        P = np.array([S.X[b_] for b_ in bd])
        lam = 0.72 * A_HOLE / np.linalg.norm(P - cen, axis=1).max()
        uv = din_inv(k, P)
        phi = np.arctan2(uv[:, 1], uv[:, 0])
        sq = {}
        for a in range(na + 1):
            for b in range(NB + 1):
                g = np.array(S.X[ids[i0 + a, j0 + b]])
                sq[(a, b)] = S.P(cen + lam * (g - cen))
        m = len(bd)
        if stage == 2:
            ring = [S.P(din(k, A_HOLE * np.cos(f), A_HOLE * np.sin(f), sh)) for f in phi]
            for q in range(m):
                l = (q + 1) % m
                S.curve(ring[q], ring[l], ("fn", ring_fn(k, phi[q], phi[l], sh)))
                S.quad((bd[q], bd[l], ring[l], ring[q]), "ringB")
                S.quad((ring[q], ring[l], sq[walk[l]], sq[walk[q]]), "ringA")
        elif stage == 1:
            for q in range(m):
                l = (q + 1) % m
                S.quad((bd[q], bd[l], sq[walk[l]], sq[walk[q]]), "ringA")
        if stage > 0:
            for a in range(na):
                for b in range(NB):
                    S.quad((sq[(a, b)], sq[(a + 1, b)], sq[(a + 1, b + 1)], sq[(a, b + 1)]), "square")
        info.append(dict(k=k, cen=cen, sref=sh, theta=thh, lam=lam, i0=i0, j0=j0, na=na, phi=phi, bd=P,
                         rect=(S.X[ids[i0, j0]], S.X[ids[i0 + na, j0 + NB]])))
    return S, info


def build_unrolled(th):
    zs, bands = rows(holes[:, 1])
    S, info = hole_section(th, zs, 2)
    return S, zs, info


def cyl(rfun):
    """unrolled (s, z) -> cylinder at radius rfun(z)."""
    return lambda P: np.c_[rfun(P[:, 1]) * np.cos(P[:, 0] / RREF), rfun(P[:, 1]) * np.sin(P[:, 0] / RREF), P[:, 1]]


if __name__ == "__main__":
    t0 = time.time()
    th = columns()
    S, zs, info = build_unrolled(th)
    print("unrolled: %d quads, %d rows, %d holes (%.0fs)" % (len(S.quads), len(zs), len(info), time.time() - t0), flush=True)
    sec = S.to_quadmesh(ORDER, 0.0)
    flat = quadmesh.transform_fn(sec, lambda P: np.c_[P[:, 0], P[:, 2], np.zeros(len(P))]) if False else sec
    # downcomer annulus: R711.2 -> mid -> outer wall (radial loft, straight)
    rmid = lambda z: 0.5 * (RCO + r_out(z))
    radii = [lambda z: np.full(len(np.atleast_1d(z)), RCO), rmid, r_out]
    sl = [quadmesh.transform_fn(sec, cyl(rf)) for rf in radii]
    sw = []
    for k in range(2):
        mids = []
        for t in G[1:-1]:
            ra, rb = radii[k], radii[k + 1]
            mids.append(quadmesh.transform_fn(sec, cyl(lambda z, ra=ra, rb=rb, t=t: (1 - t) * ra(z) + t * rb(z))))
        sw.append(mids)
    dc = hexmesh.loft(sl, sweep_nodes=sw, element_tags="fluid", first_tag="wall", last_tag="wall")
    print("downcomer: %d hexes, corner %s (%.0fs)" % (len(dc.hexes), hexmesh.corner_summary(dc), time.time() - t0), flush=True)
    # bores: hole disks (square + ringA) from the outer wall to R_VOL
    lab = np.array(S.lab)
    disk = quadmesh.select(sec, np.isin(lab, ("square", "ringA")))
    cens = np.array([d["cen"] for d in info])
    from scipy.spatial import cKDTree
    ct = cKDTree(cens)

    def bore(t):
        def f(P):
            _, j = ct.query(P[:, :2])
            out = np.empty((len(P), 3))
            for h in np.unique(j):
                m_ = j == h
                k = info[h]["k"]
                uv = din_inv(k, P[m_, :2])
                lm = (1 - t) * TL.lam(k, uv[:, 0], uv[:, 1], R_HI) + t * TL.lam(k, uv[:, 0], uv[:, 1], R_VOL)
                out[m_] = TL.xyz(k, uv[:, 0], uv[:, 1], lm)
            return out
        return f
    fr = np.linspace(0, 1, NT + 1)
    sl = [quadmesh.transform_fn(disk, bore(f)) for f in fr]
    sw = [[quadmesh.transform_fn(disk, bore(f + t * (fr[k + 1] - f))) for t in G[1:-1]] for k, f in enumerate(fr[:-1])]
    bores = hexmesh.loft(sl, sweep_nodes=sw, element_tags="fluid", first_tag="", last_tag="inlet")
    print("bores: %d hexes, corner %s (%.0fs)" % (len(bores.hexes), hexmesh.corner_summary(bores), time.time() - t0), flush=True)
    mesh = hexmesh.merge([dc, bores], clear_seam_tags=True)
    print("dc3: %d hexes" % len(mesh.hexes), hexmesh.topology_report(mesh), flush=True)
    print(hexmesh.quality_summary(mesh)); print("corner:", hexmesh.corner_summary(mesh))
    pickle.dump(dict(mesh=mesh, info=info, zs=zs, th=th), open(os.path.join(HERE, "dc3.pkl"), "wb"))
    from nekmeshpy.io.writer import to_vtu
    to_vtu(mesh, os.path.join(HERE, "dc3.vtu"))
