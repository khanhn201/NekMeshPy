"""Tilted bores: per hole k an axis line X = c_k + u e1_k + v e2 + lam a_k (a_k horizontal,
e1_k = horizontal normal to a_k, e2 = z).  (u, v) are the bore cross-section coordinates (the
bore wall is u^2 + v^2 = A^2); every wall point is reached by sliding along a_k."""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
AX = np.load(os.path.join(HERE, "bore_axes.npy"))
C = AX[:, :3]; Adir = AX[:, 3:6]
E1 = np.c_[-Adir[:, 1], Adir[:, 0], np.zeros(len(AX))]          # z x a (horizontal)
NH = len(AX)


def lam(k, u, v, R):
    """slide distance from the cross-section plane to the cylinder r = R (outward root)."""
    k = np.asarray(k); u = np.asarray(u, float)
    p = C[k, :2] + u[..., None] * E1[k, :2]
    pa = (p * Adir[k, :2]).sum(-1)
    return -pa + np.sqrt(pa ** 2 - (p ** 2).sum(-1) + R ** 2)


def xyz(k, u, v, lm):
    k = np.asarray(k)
    return C[k] + np.asarray(u, float)[..., None] * E1[k] + np.asarray(v, float)[..., None] * np.array([0, 0, 1.0]) + np.asarray(lm, float)[..., None] * Adir[k]


def wall_xyz(k, u, v, R):
    return xyz(k, u, v, lam(k, u, v, R))


def uv_of(k, X):
    """cross-section coords of 3D point(s) X (slides along a_k)."""
    k = np.asarray(k); X = np.asarray(X, float)
    return np.stack([((X - C[k]) * E1[k]).sum(-1), X[..., 2] - C[k, 2]], -1)


def wall_angle(k, R):
    X = wall_xyz(k, 0.0, 0.0, R)
    return np.arctan2(X[..., 1], X[..., 0])


def delta(R0, R1):
    """global rotation of a point slid from r=R0 to r=R1 along the bores' tilt."""
    ar = np.mean((Adir[:, :2] * C[:, :2]).sum(1) / np.linalg.norm(C[:, :2], axis=1)) if False else None
    # tilt from the fitted axes: angle between a_k and the radial direction at the hole on R0
    th = wall_angle(np.arange(NH), R0)
    rad = np.c_[np.cos(th), np.sin(th)]
    cb = (Adir[:, :2] * rad).sum(1).mean(); sb = np.sqrt(1 - cb ** 2)
    lmb = -R0 * cb + np.sqrt(R0 ** 2 * cb ** 2 - R0 ** 2 + R1 ** 2)
    return np.arctan2(lmb * sb, R0 + lmb * cb)
