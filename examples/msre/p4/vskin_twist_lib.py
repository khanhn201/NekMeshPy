"""vskin_twist helpers: rounded vane model, twist-aware hex score, boundary node classes."""
import numpy as np
from nekmeshpy import HexMesh
import head as HD
FP = HexMesh.FACE_POINTS
R0V, VHW, NV = HD.R0V, HD.VHW, HD.NV
TH = np.radians(7.5 * np.arange(NV))
_ug = np.linspace(R0V + 50, HD.R_VES, 4000); _h = HD.s1(_ug) - HD.s0(_ug)
UEND = min(float(_ug[np.argmax(_h < 2 * VHW)]), HD.R_MID - 1.5 - 1.0); UW = UEND - VHW
E12 = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])


def onvane(p):
    k = np.round(np.arctan2(p[:, 1], p[:, 0]) / np.radians(7.5)).astype(int) % NV
    t = np.c_[np.cos(TH[k]), np.sin(TH[k])]
    u = np.einsum("ij,ij->i", p[:, :2], t); uc = np.clip(u, R0V + VHW, UW)
    mm = np.c_[uc[:, None] * t, np.minimum(p[:, 2], HD.s1(uc) - VHW)]
    d = p - mm; L = np.linalg.norm(d, axis=1); L = np.where(L < 1e-12, 1, L)
    return mm + VHW * d / L[:, None]


def hex_score(Pt, hx):
    from nekmeshpy.hexmesh.quality import corner_scaled_jacobian
    sj = np.asarray(corner_scaled_jacobian(Pt, hx))
    q = Pt[hx[:, FP]]                                         # (n,6,4,3)
    turn = np.cross(np.roll(q, -1, axis=2) - q, np.roll(q, 1, axis=2) - q)
    mean = turn.sum(axis=2, keepdims=True)
    ag = np.einsum("efkc,efkc->efk", turn, np.broadcast_to(mean, turn.shape))
    ag = ag / (np.linalg.norm(turn, axis=3) * np.linalg.norm(mean, axis=3) + 1e-300)
    return np.minimum(sj, ag.min(axis=(1, 2)))


def node_classes(P, H, vtol=0.05):
    fk = np.sort(H[:, FP].reshape(-1, 4), axis=1)
    _, fi, fc = np.unique(fk, axis=0, return_inverse=True, return_counts=True)
    bfaces = H[:, FP].reshape(-1, 4)[fc[fi] == 1]
    bnd = np.zeros(len(P), bool); bnd[bfaces.ravel()] = True
    r_ = np.hypot(P[:, 0], P[:, 1])
    shell = bnd & (np.abs(P[:, 2] - HD.s0(r_)) < 1e-6)
    vane = bnd & ~shell & (np.linalg.norm(onvane(P) - P, axis=1) < vtol)
    free = ~bnd | vane | shell

    def proj(i, p):
        if vane[i]:
            p = onvane(p[None])[0]
        if shell[i]:
            p = p.copy(); p[2] = HD.s0(np.hypot(p[0], p[1]))
        return p
    return free, proj
