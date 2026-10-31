"""Small structured hex blocks shared by the hex smoother tests (kept tiny: the smoother is slow)."""

import numpy as np

from nekmeshpy.hexmesh import HexMesh
from nekmeshpy.tags import Tags


def block_arrays(n=4, jitter=0.0, seed=0, size=(1.0, 1.0, 1.0)):
    """``n^3`` hexes of the box ``[0, size]`` as ``(points, hexes)`` in Nek corner order, the interior
    vertices jittered by up to ``jitter`` cells."""
    rng = np.random.default_rng(seed)
    xs = [np.linspace(0.0, s, n + 1) for s in size]
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    lo, hi = np.zeros(3), np.asarray(size, float)
    inner = ((P > lo + 1e-9) & (P < hi - 1e-9)).all(axis=1)
    P[inner] += rng.uniform(-jitter, jitter, (int(inner.sum()), 3)) * (hi / n)
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    c = [idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[1:, 1:, :-1], idx[:-1, 1:, :-1],
         idx[:-1, :-1, 1:], idx[1:, :-1, 1:], idx[1:, 1:, 1:], idx[:-1, 1:, 1:]]
    return P, np.stack(c, axis=-1).reshape(-1, 8)


def block(n=4, jitter=0.0, seed=0, size=(1.0, 1.0, 1.0), tags=None):
    """The same block as a linear :class:`HexMesh`; ``tags`` a ``(E,)`` list of element tags."""
    P, H = block_arrays(n, jitter, seed, size)
    et = None if tags is None else Tags.from_dense(np.asarray(tags))
    return HexMesh.from_corners(P, H, et)
