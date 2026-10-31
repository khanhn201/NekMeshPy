"""Small CCW quad grids shared by the smoother tests."""

import numpy as np


def grid(n=6, jitter=0.0, seed=0, z=None):
    """``n x n`` unit-square quads (CCW) as ``(points, corners)``, interior jittered."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    interior = ((np.abs(P[:, 0] - 0.5) < 0.5 - 1e-9) & (np.abs(P[:, 1] - 0.5) < 0.5 - 1e-9))
    P[interior, :2] += rng.uniform(-jitter, jitter, (interior.sum(), 2)) / n
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    return P, quads
