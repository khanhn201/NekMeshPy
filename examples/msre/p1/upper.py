"""Core + stringer tops as terrain-following layers over the one 2D section.

Every zone is a loft of the section (or a subset of it) between two height fields
evaluated at every node, so the 45-degree pyramid faces land exactly on hex faces:
the pyramid is planar over each quad because its creases (joints = valleys,
diagonals = ridges) are section edges."""
import pickle, sys, time
import numpy as np
from geom import *
from q2d import Sec
from nekmeshpy import quadmesh, hexmesh
from nekmeshpy.io.writer import to_vtu

ORDER = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NZ = int(sys.argv[2]) if len(sys.argv) > 2 else 32
Z_TOP = 2070.0
NZ_EDGE = 2
LAT = pickle.load(open("lattice.pkl", "rb"))
CELLS = LAT["cells"]
TALL = {(i, 0) for i in range(-13, 14) if abs(i) >= 3} | {(0, j) for j in range(-13, 14) if abs(j) >= 3}
PYR = {c for c, t in CELLS.items() if t == "full" and c not in CENTRAL}
TALL &= PYR
NORMAL = PYR - TALL


def cell_uv(p):
    c = np.round(p[:, :2] / PITCH).astype(int)
    d = np.abs(p[:, :2] - c * PITCH)
    return [tuple(x) for x in c], d.max(1)


def owner_cells(p, tol=1e-6):
    """all lattice cells whose closed square contains p (1, 2 or 4 of them)."""
    out = []
    base = np.floor(p[:2] / PITCH + 0.5 - 1e-9).astype(int)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            c = (base[0] + di, base[1] + dj)
            if np.all(np.abs(p[:2] - np.array(c) * PITCH) <= H + tol):
                out.append(c)
    return out


OFFS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIRS = [np.array(o, float) for o in OFFS]


def pyr(p, c, z0):
    return z0 + H - np.abs(p[:2] - np.array(c) * PITCH).max()


def L1_point(p):
    cs = owner_cells(p)
    for c in cs:
        if c in PYR:
            return pyr(p, c, Z_PYR)
    for c in cs:
        if c in CENTRAL:                  # flat shoulder, ramped down 45 deg to pyramid-topped neighbours
            d = p[:2] - np.array(c) * PITCH
            ds = [H - d @ n for n, o in zip(DIRS, OFFS) if (c[0] + o[0], c[1] + o[1]) in PYR]
            return Z_PYR + min([Z_SHOULDER - Z_PYR] + ds)
    return Z_PYR


def L2_point(p):
    for c in owner_cells(p):
        if c in TALL:
            return pyr(p, c, Z_PYR_TALL)
    return Z_PYR_TALL


def thimble_outer(p):
    """for a node inside a thimble site: the point straight out on the bore boundary."""
    for c, rt in THIMBLES.items():
        cc = np.array(c, float) * PITCH
        d = p[:2] - cc
        r = np.linalg.norm(d)
        if r < R_OPEN + 1e-6 or np.abs(d).max() < H + 1e-6:
            u = d / r
            s_sq = H / np.abs(u).max()
            return np.concatenate([cc + max(s_sq, R_OPEN) * u, [0]])
    return p


def field(fn):
    cache = {}

    def f(P):
        out = np.empty(len(P))
        for i, p in enumerate(P):
            k = (round(p[0], 7), round(p[1], 7))
            v = cache.get(k)
            if v is None:
                v = fn(thimble_outer(p))
                cache[k] = v
            out[i] = v
        return out
    return f


F1, F2 = field(L1_point), field(L2_point)


def flat(z):
    return lambda P: np.full(len(P), z)


def zone(sec, lo, hi, n, first, last):
    slices = []
    for k in range(n + 1):
        t = k / n
        slices.append(quadmesh.transform_fn(
            sec, lambda P, t=t: np.stack([P[:, 0], P[:, 1], (1 - t) * lo(P) + t * hi(P)], 1)))
    return hexmesh.loft(slices, element_tags="fluid", first_tag=first, last_tag=last)


if __name__ == "__main__":
    t0 = time.time()
    S = pickle.load(open("section.pkl", "rb"))
    lab = np.array(S.lab)
    X = np.array(S.X)
    Q = np.array(S.quads)
    cen = X[Q].mean(1)
    qcell = [tuple(c) for c in np.round(cen / PITCH).astype(int)]
    tallq = np.array([c in TALL for c in qcell])
    graph = np.isin(lab, ("gr", "gr_post", "gr_tip"))
    full = S.to_quadmesh(ORDER)

    def sub(mask):
        """fluid sub-section; side walls of fuel-channel quads are named "channel", the rest "wall"."""
        s = quadmesh.select(full, mask)
        rows = np.asarray(quadmesh.boundary_edges(s))
        return quadmesh.tag_edges(s, rows, np.where(lab[mask][rows[:, 0]] == "chan", "channel", "wall"))

    FL = np.isin(lab, ("chan", "gap"))                   # half-channels are not extruded (graphite)
    BORE = np.array([l.startswith("th_") or l.startswith("bk_") for l in lab])   # solid above the bores
    m_U1a = (lab != "post") & ~(tallq & graph) & ~BORE
    m_U1b = ~(tallq & graph) & ~BORE
    secs = {k: sub(m) for k, m in (("C", FL), ("U1a", m_U1a), ("U1b", m_U1b), ("U2", ~BORE))}
    print("sections built %.1fs" % (time.time() - t0))
    lo_slab = hexmesh.extrude(secs["C"], 20.0, 1, origin=(0, 0, Z_CORE0 - 20.0),
                              element_tags="fluid", first_tag="inlet", last_tag="")
    blocks = [
              zone(secs["C"], flat(Z_CORE0), F1, NZ, "", ""),
              zone(secs["U1a"], F1, flat(Z_POST), 3, "wall", "wall"),
              zone(secs["U1b"], flat(Z_POST), F2, 3, "wall", "wall"),
              zone(secs["U2"], F2, flat(Z_TOP), 2, "wall", "outlet")]
    import bores
    blocks += bores.build(full, lab, cen, F1, NZ)
    print("zones built %.1fs" % (time.time() - t0), [len(b.hexes) for b in blocks])
    mesh = hexmesh.merge(blocks, clear_seam_tags=True)
    print("merged %d hexes %.1fs" % (len(mesh.hexes), time.time() - t0))
    print(hexmesh.topology_report(mesh))
    print(hexmesh.tag_report(mesh))
    print(hexmesh.quality_summary(mesh))
    print("corner:", hexmesh.corner_summary(mesh))
    pickle.dump(mesh, open(f"upper_o{ORDER}_nz{NZ}.pkl", "wb"))
    to_vtu(mesh, f"upper_o{ORDER}_nz{NZ}.vtu", bc={"wall": "W  ", "inlet": "v  ", "outlet": "O  "})
    print("done %.1fs" % (time.time() - t0))
