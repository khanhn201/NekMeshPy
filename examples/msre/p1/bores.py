"""The four centre bores below/around the thimble caps and the sample basket.

Each site is its own loft (the sites share no nodes with each other or with the
channels below the stringer tops), with its own z levels:
  thimble: r12.7 hole 347.786-398.586 | whole bore 398.586-411.29 |
           411.29 -> cap height field (sphere R31.4795 @z447.636 + r5.333 fillet) | ring -> F1
  basket : hole 347.786-381.001 | drain r7.504 381-398.586 | drain + outside the bottom
           cone (r19.68->25.79, 398.586-408.711) | funnel r7.504->19.69 (410.543-430.789)
           up to 434.43 | outside r25.79 up to 444.78 | ring -> F1
The basket spigot's 1.5 mm annular pocket (z 381-391.19) is dropped."""
import numpy as np
from geom import *
from nekmeshpy import quadmesh, hexmesh

Z_HOLE0, Z_FLOOR = 347.786, 398.586
R_HOLE = 12.7
# thimble cap
CAP_C, CAP_R, CAP_Z1, FIL_R = 447.636, 31.4795, 433.077, 5.333
# basket
BK_A0, BK_C0, BK_D = 7.5038, 25.79, R_BASKET


def cap_z(r):
    rc = R_THIM - FIL_R
    rt = rc * CAP_R / (CAP_R - FIL_R)              # sphere/fillet tangent radius
    zs = CAP_C - np.sqrt(np.maximum(CAP_R ** 2 - r ** 2, 0))
    zf = CAP_Z1 - np.sqrt(np.maximum(FIL_R ** 2 - (np.minimum(r, R_THIM) - rc) ** 2, 0))
    return np.where(r <= rt, zs, zf)


def rA(z):
    return np.interp(z, [410.543, 430.789], [BK_A0, 19.69])


def rC(z):
    return np.interp(z, [Z_FLOOR, 408.711], [19.68, BK_C0])


def rB(z):
    return np.where(z <= Z_FLOOR + 1e-9, R_HOLE, 0.5 * (rA(z) + rC(z)))


def basket_xy(cc):
    def f(P, z):
        d = P[:, :2] - cc
        rho = np.hypot(d[:, 0], d[:, 1])
        k_b = np.array([BK_A0, R_HOLE, BK_C0, BK_D])
        k_a = np.array([rA(z), rB(z), rC(z), BK_D])
        r_new = np.where(rho <= BK_A0, rho * k_a[0] / BK_A0, np.interp(rho, k_b, k_a))
        r_new = np.where(rho >= BK_D, rho, r_new)
        s = np.where(rho > 1e-12, r_new / np.maximum(rho, 1e-12), 1.0)
        out = P.copy()
        out[:, :2] = cc + d * s[:, None]
        return out
    return f


def loft_band(sec, lo, hi, n, first, last, xy=None):
    slices = []
    for k in range(n + 1):
        t = k / n

        def m(P, t=t):
            z = (1 - t) * lo(P) + t * hi(P)
            Q = P.copy() if xy is None else xy(P, z[0] if np.ndim(z) else z)
            Q[:, 2] = z
            return Q
        slices.append(quadmesh.transform_fn(sec, m))
    return hexmesh.loft(slices, element_tags="fluid", first_tag=first, last_tag=last)


def flat(z):
    return lambda P: np.full(len(P), z)


def build(full, lab, cen, F1, NZ):
    """list of HexMesh blocks, one band at a time, for all four sites."""
    blocks = []

    def sub(mask):
        s = quadmesh.select(full, mask)
        return quadmesh.tag_edges(s, quadmesh.boundary_edges(s), "wall")

    for c, rt in THIMBLES.items():
        cc = np.array(c, float) * PITCH
        near = np.hypot(*(cen - cc).T) < R_OPEN + 10
        m = lambda *ls: near & np.isin(lab, ls)
        if rt == R_THIM:
            def cap(P, cc=cc):
                r = np.hypot(*(P[:, :2] - cc).T)
                return np.where(r <= R_THIM + 1e-9, cap_z(r), CAP_Z1)
            # below Z_CORE0 the bores belong to the bottom-zone bands (band A ...)
            bands = [(m("th_hole", "th_disc", "thim"), flat(Z_CORE0), cap, 3, "", "wall", None),
                     (m("thim"), cap, F1, NZ, "wall", "", None)]
        else:
            xy = basket_xy(cc)
            A, AB, CD, RG_ = "bk_A", "bk_AB", "bk_CD", "thim"
            bands = [(m(A, CD, RG_), flat(Z_CORE0), flat(430.789), 3, "", "", xy),
                     (m(A, CD, RG_), flat(430.789), flat(434.43), 1, "", "wall", xy),
                     (m(CD, RG_), flat(434.43), flat(444.78), 2, "", "wall", xy),
                     (m(RG_), flat(444.78), F1, NZ, "wall", "", xy)]
        for mask, lo, hi, n, f_, l_, xyf in bands:
            blocks.append(loft_band(sub(mask), lo, hi, n, f_, l_, xyf))
    return blocks
